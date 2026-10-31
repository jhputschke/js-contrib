"""
tests/test_multi_hard.py

The event's other hard scatterings (docs/Plans/PLAN_multi_hard.md): the split of an event's
partons into the trigger's and the extras' showers, the particlize file's scatterings/ and
initiators/scatter, the extra_frag hadron files (groups and per_scattering), the readers'
full event (bulk_jet + jet_frag + extra_frag) and run_prod_jet.py's option handling.  No
compiled extension: a stub core stands in for Colorless.  The production side (PythiaGun's
<extraHard>, the separate random streams) needs MUSIC and is checked by the plan's V1-V3.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests"))

h5py = pytest.importorskip("h5py")

from jetscape import hadronizer as hz  # noqa: E402
from jetscape.hadrons_h5 import (FIELDS, ORIGIN, TAGS, HadronFile, HadronH5Writer,  # noqa: E402
                                 JetEvents)
from jetscape.particlize_h5 import (FINAL_PARTON_COLUMNS, SCATTER_COLUMNS,  # noqa: E402
                                    ParticlizeFile, split_by_scatter)
from jetscape.showers import INITIATOR_COLUMNS  # noqa: E402

EXAMPLE = ROOT / "example" / "prod_AuAu_0_10_jet"


def _load(name):
    spec = importlib.util.spec_from_file_location(name[:-3], EXAMPLE / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _partons(showers):
    """Two partons per shower, E = 10 x (shower + 1)."""
    rows = []
    for sh in showers:
        for k in range(2):
            r = np.zeros(len(FINAL_PARTON_COLUMNS))
            r[0], r[1], r[3] = sh, 21, 10.0 * (sh + 1) + k
            rows.append(r)
    return np.asarray(rows).reshape(-1, len(FINAL_PARTON_COLUMNS))


def _initiators(n):
    ini = np.zeros((n, len(INITIATOR_COLUMNS)))
    ini[:, 0] = np.arange(n)
    ini[:, 1] = 21
    return ini


def _scatterings(scatters):
    sc = np.zeros((len(scatters), len(SCATTER_COLUMNS)))
    sc[:, 0] = scatters
    sc[:, 1] = 15.0 + np.arange(len(scatters))
    return sc


class _Core:
    """Stub: Colorless gives one hadron per parton with E = seed % 1000."""

    PYTHIA_VERSION = 8312

    def __init__(self):
        self.calls = []

    def hadronize_partons(self, module, partons, seed):
        self.calls.append((len(partons), int(seed)))
        n = len(partons)
        return {"pid": np.full(n, 211, np.int32), "pstat": np.zeros(n, np.int32),
                "p": np.tile(np.array([seed % 1000, 0.1, 0, 0], np.float32), (n, 1)),
                "x": np.zeros((n, 4), np.float32)}


# ── the split ────────────────────────────────────────────────────────────────────
def test_split_keeps_the_triggers_rows_as_they_are():
    partons = _partons([0, 1, 2, 3, 4])
    ini = _initiators(5)
    scat = np.array([0, 0, 1, 1, 2])
    (p0, i0), (px, ix, sx) = split_by_scatter(partons, ini, scat)
    # the trigger's: showers 0 and 1, rows and shower index unchanged
    assert np.array_equal(p0, partons[:4]) and np.array_equal(i0, ini[:2])
    assert np.array_equal(px, partons[4:]) and np.array_equal(ix, ini[2:])
    assert sx.tolist() == [1, 1, 2]
    # no extras: everything is the trigger's
    (p0, i0), (px, ix, sx) = split_by_scatter(partons, ini, np.zeros(5))
    assert np.array_equal(p0, partons) and len(px) == len(ix) == len(sx) == 0
    with pytest.raises(ValueError, match="scatter values"):
        split_by_scatter(partons, ini, scat[:3])


# ── the particlize file ──────────────────────────────────────────────────────────
def test_particlize_stores_scatterings_and_initiator_scatter(tmp_path):
    from test_particlize_h5 import _writer

    w, jet, bg, mgr = _writer(tmp_path, extra_hard=True)
    mgr.rows = _partons([0, 1, 2])
    mgr.get_shower_initiating_partons = lambda: []
    import jetscape.particlize_h5 as ph
    orig = ph.initiators_from_manager
    ph.initiators_from_manager = lambda m: _initiators(3)
    try:
        w.Exec(0, scatterings=_scatterings([0, 1]), shower_scatter=[0, 0, 1])
        mgr.rows = _partons([0])
        ph.initiators_from_manager = lambda m: _initiators(1)
        w.Exec(1, scatterings=_scatterings([0]), shower_scatter=[0])
    finally:
        ph.initiators_from_manager = orig
    w.Finish()
    with ParticlizeFile(tmp_path / "p.h5") as pf:
        assert pf.extra_hard
        assert pf.initiator_scatter(0).tolist() == [0, 0, 1]
        assert pf.initiator_scatter(1).tolist() == [0]
        assert pf.scatterings(0)[:, 0].tolist() == [0, 1]
        assert pf.f["scatterings"].attrs["data_columns"].tolist() == list(SCATTER_COLUMNS)


def test_particlize_without_extras_has_no_scatterings(tmp_path):
    from test_particlize_h5 import _writer

    w, jet, bg, mgr = _writer(tmp_path)
    w.Exec(0)
    w.Finish()
    with ParticlizeFile(tmp_path / "p.h5") as pf:
        assert not pf.extra_hard and pf.scatterings(0) is None
        assert "initiators/scatter" not in pf.f


def test_particlize_sink_hands_over_the_scatterings(tmp_path):
    from test_particlize_h5 import _Leg, _Mgr, _JS
    from jetscape.particlize_h5 import ParticlizeH5Writer

    got = []
    jet, bg, mgr = _Leg("MUSIC_2"), _Leg("MUSIC_1"), _Mgr()
    w = ParticlizeH5Writer(None, legs=("jet", "bg"), music_input="", sink=got.append,
                           extra_hard=True)
    w.attach(_JS(bg, jet), manager=mgr)
    import jetscape.particlize_h5 as ph
    orig = ph.initiators_from_manager
    ph.initiators_from_manager = lambda m: _initiators(2)
    try:
        w.Exec(0, scatterings=_scatterings([0, 1]), shower_scatter=[0, 1])
    finally:
        ph.initiators_from_manager = orig
    rec = got[0]
    assert rec["initiator_scatter"].tolist() == [0, 1]
    assert rec["scatterings"].shape == (2, len(SCATTER_COLUMNS))


# ── extra_frag ───────────────────────────────────────────────────────────────────
def test_tags_and_seed_keys():
    assert TAGS[:3] == ("bulk_jet", "bulk_bg", "jet_frag") and "extra_frag" in TAGS
    # the seeds' key of the old tags is unchanged; extra_frag has its own
    assert [hz.TAG_INDEX[t] for t in ("bulk_jet", "bulk_bg", "jet_frag")] == [0, 1, 2]
    assert hz.unit_seed(1, "extra_frag", 3) != hz.unit_seed(1, "jet_frag", 3)


@pytest.mark.parametrize("mode", hz.EXTRA_HADRONIZE)
def test_extra_fragments_write_extra_frag(tmp_path, mode):
    core = _Core()
    h = hz.Hadronizer(core, colorless=object(), base_seed=1, prod=7)
    h.open("jet_frag", tmp_path / "s_hadrons_jet_frag.h5", n_samples=2, attrs={},
           initiators=True, partons=True)
    h.open("extra_frag", tmp_path / "s_hadrons_extra_frag.h5", n_samples=2, attrs={},
           initiators=True, partons=True, hadron_scatter=mode == "per_scattering")
    partons, ini = _partons([0, 1, 2, 3]), _initiators(4)
    scat = np.array([0, 1, 1, 2])
    (p0, i0), (px, ix, sx) = split_by_scatter(partons, ini, scat)
    jf = h.fragments(0, p0, [h.seed("jet_frag", 0, s) for s in range(2)], initiators=i0)
    xf = h.extra_fragments(0, px, 2, initiators=ix, initiator_scatter=sx,
                           scatterings=_scatterings([0, 1, 2]), mode=mode, n_extra=2)
    h.close()
    assert [len(f["pid"]) for f in jf] == [2, 2]
    assert [len(f["pid"]) for f in xf] == [6, 6]
    if mode == "groups":
        assert core.calls[2:] == [(6, h.seed("extra_frag", 0, 0)),
                                  (6, h.seed("extra_frag", 0, 1))]
    else:                                # scattering 1 (4 partons), 2 (2), per fragmentation
        assert [c[0] for c in core.calls[2:]] == [4, 2, 4, 2]
        assert xf[0]["scatter"].tolist() == [1, 1, 1, 1, 2, 2]
    with HadronFile(tmp_path / "s_hadrons_extra_frag.h5") as hf:
        assert hf.tag == "extra_frag" and hf.n_samples(0) == 2
        assert hf.scatterings(0)[:, 0].tolist() == [0, 1, 2]
        assert hf.initiator_scatter(0).tolist() == [1, 1, 2]
        assert np.array_equal(hf.partons(0), px)
        assert int(hf.units["n_extra"][0]) == 2
        ev = hf.sample_event(0, 1)
        assert ("scatter" in ev) == (mode == "per_scattering")
    with HadronFile(tmp_path / "s_hadrons_jet_frag.h5") as hf:
        assert np.array_equal(hf.partons(0), p0)          # the trigger's alone
        assert hf.initiator_scatter(0).tolist() == [0]


def test_extra_fragments_of_an_event_without_extras(tmp_path):
    core = _Core()
    h = hz.Hadronizer(core, colorless=object())
    h.open("extra_frag", tmp_path / "s_hadrons_extra_frag.h5", n_samples=1, attrs={},
           initiators=True, hadron_scatter=True)
    xf = h.extra_fragments(0, np.zeros((0, 14)), 1, initiators=np.zeros((0, 11)),
                           initiator_scatter=[], scatterings=_scatterings([0]),
                           mode="per_scattering")
    h.close()
    assert len(xf[0]["pid"]) == 0 and core.calls == []
    with HadronFile(tmp_path / "s_hadrons_extra_frag.h5") as hf:
        assert hf.n_samples(0) == 1 and len(hf.sample_event(0, 0)["pid"]) == 0


def test_writer_options_are_tag_bound(tmp_path):
    with pytest.raises(ValueError, match="hadron_scatter"):
        HadronH5Writer(tmp_path / "a.h5", tag="jet_frag", n_samples=1, hadron_scatter=True)
    HadronH5Writer(tmp_path / "b.h5", tag="extra_frag", n_samples=1, partons=True,
                   initiators=True).close()


# ── the full event ───────────────────────────────────────────────────────────────
def _hadrons(n, e):
    return {"pid": np.full(n, 211, np.int32), "pstat": np.zeros(n, np.int32),
            "p": np.tile(np.array([e, 0.1, 0, 0], np.float32), (n, 1)),
            "x": np.zeros((n, 4), np.float32)}


def test_jet_event_is_bulk_plus_jet_and_extra_fragments(tmp_path):
    stem = tmp_path / "s"
    for tag, n in (("bulk_jet", 5), ("jet_frag", 2), ("extra_frag", 3), ("bulk_bg", 4)):
        with HadronH5Writer(f"{stem}_hadrons_{tag}.h5", tag=tag, n_samples=1,
                            attrs={"source_uuid": ""}) as w:
            w.append_unit([_hadrons(n, 1.0)], unit=0, event=0)
    with JetEvents.from_stem(str(stem)) as je:
        ev = je.jet_event(0, 0)
        assert np.bincount(ev["origin"]).tolist() == [5, 2, 3]
        assert ORIGIN["extra"] == 2 and set(ev) == set(FIELDS) | {"origin"}
        assert len(je.jet_event(0, 0, extras=False)["pid"]) == 7
        assert len(je.jet_event(0, 0, fragments=False)["pid"]) == 5
        assert je.scatterings(0) is not None


def test_jet_event_without_extra_frag_is_as_before(tmp_path):
    stem = tmp_path / "s"
    for tag, n in (("bulk_jet", 5), ("jet_frag", 2)):
        with HadronH5Writer(f"{stem}_hadrons_{tag}.h5", tag=tag, n_samples=1) as w:
            w.append_unit([_hadrons(n, 1.0)], unit=0, event=0)
    with JetEvents.from_stem(str(stem)) as je:
        assert je.extra_frag is None and je.scatterings(0) is None
        assert np.bincount(je.jet_event(0, 0)["origin"]).tolist() == [5, 2]


# ── run_prod_jet.py ──────────────────────────────────────────────────────────────
def _args(**kw):
    base = dict(extra_hard=False, extra_pthat_min=None, extra_n=None, extra_warmup=None,
                extra_hadronize="groups", hard="pythia", pthat_bins=None, pthat_min=None,
                pthat_max=None, parton_ymax=None, trigger=None, trigger_ymax=None,
                lepton_ptmin=None, lepton_etamax=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_run_prod_jet_extra_hard_options():
    rpj = _load("run_prod_jet.py")
    rpj.check_extra_hard(_args())
    rpj.check_extra_hard(_args(extra_hard=True, extra_pthat_min=10, extra_n=2))
    rpj.check_extra_hard(_args(extra_hard=True, hard="none"))
    for bad, match in ((dict(hard="none"), "needs --extra-hard"),
                       (dict(extra_n=3), "need"),
                       (dict(extra_hadronize="per_scattering"), "need"),
                       (dict(extra_hard=True, hard="pgun"), "PythiaGun"),
                       (dict(extra_hard=True, extra_pthat_min=0), "> 0"),
                       (dict(extra_hard=True, hard="none", pthat_bins="10-20"), "no triggering"),
                       (dict(extra_hard=True, hard="none", trigger="Z"), "no triggering")):
        with pytest.raises(ValueError, match=match):
            rpj.check_extra_hard(_args(**bad))
    assert rpj.hadron_tags(_args(extra_hard=True)) == ("bulk_jet", "bulk_bg", "jet_frag",
                                                      "extra_frag")
    assert rpj.hadron_tags(_args(extra_hard=True, hard="none")) == ("bulk_jet", "bulk_bg",
                                                                   "extra_frag")
    assert rpj.hadron_tags(_args()) == ("bulk_jet", "bulk_bg", "jet_frag")


def test_run_prod_jet_writes_the_extra_hard_block():
    import xml.etree.ElementTree as ET
    rpj = _load("run_prod_jet.py")
    pg = ET.fromstring("<PythiaGun><extraHard><on>1</on><pTHatMin>10</pTHatMin>"
                       "<fixedN>4</fixedN></extraHard></PythiaGun>")
    a = _args()
    rpj.configure_extra_hard(a, pg)            # off: whatever the template says
    assert pg.find("extraHard/on").text.strip() == "0"
    assert pg.find("extraHard/fixedN") is None and a.extra_pthat_min_xml == "10"
    a = _args(extra_hard=True, hard="none", extra_pthat_min=12.5, extra_n=2)
    rpj.configure_extra_hard(a, pg)
    got = {c.tag: c.text.strip() for c in pg.find("extraHard")}
    assert got == {"on": "1", "noTrigger": "1", "pTHatMin": "12.5", "fixedN": "2"}
    pg = ET.fromstring("<PythiaGun/>")
    a = _args(extra_hard=True)
    rpj.configure_extra_hard(a, pg)
    assert a.extra_pthat_min_xml == "15"      # PythiaGun's default


# ── HadronFileReader ─────────────────────────────────────────────────────────────
def test_jet_minus_background_adds_the_extras_to_the_fragments(tmp_path):
    from extras_campaign import zjet_files
    from jetscape.hadrons_h5 import HadronFileReader

    zjet_files(tmp_path / "x", extras=True)
    zjet_files(tmp_path / "n", extras=False)
    zjet_files(tmp_path / "m", minbias=True)
    bins = np.linspace(0, 50, 6)
    rx, rn, rm = (HadronFileReader(str(tmp_path / d)) for d in "xnm")
    full, _ = rx.jet_minus_background("pt", bins, fragments=True)
    trig, _ = rx.jet_minus_background("pt", bins, fragments=True, extras=False)
    bulk, _ = rx.jet_minus_background("pt", bins)
    extra, _ = rx.hist("extra_frag", "pt", bins)
    frag, _ = rx.hist("jet_frag", "pt", bins)
    assert np.allclose(full, bulk + frag + extra) and np.allclose(trig, bulk + frag)
    assert rx.has_extras() and not rn.has_extras()
    # without extras as before; minimum bias: no jet_frag, the extras alone
    assert np.allclose(rn.jet_minus_background("pt", bins, fragments=True)[0],
                       rn.jet_minus_background("pt", bins)[0]
                       + rn.hist("jet_frag", "pt", bins)[0])
    assert np.allclose(rm.jet_minus_background("pt", bins, fragments=True)[0],
                       rm.jet_minus_background("pt", bins)[0]
                       + rm.hist("extra_frag", "pt", bins)[0])
    # the trigger's initiators never come from extra_frag
    ini = rx.event_info(0).initiators()
    assert np.array_equal(ini, rn.event_info(0).initiators())
