"""
tests/test_hadrons_only.py

run_prod_jet.py --hadrons-only (docs/Plans/PLAN_AuAu_hadrons_only.md): the shared per-unit
code (jetscape.hadronizer), the jet file's partons/, the readers without a particlize file,
the driver's option handling and hadronize.py on a hadrons-only jet file.  No compiled
extension: a stub core stands in for iSS and Colorless.  The end-to-end equality with two
stages (hadronize.py on a particlize run of the same seed) needs MUSIC and is in the plan's
validation section.
"""

from __future__ import annotations

import importlib.util
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from jetscape import hadronizer as hz  # noqa: E402
from jetscape.hadrons_h5 import (PARTON_COLUMNS, HadronFile, HadronFileReader,  # noqa: E402
                                 HadronH5Writer, JetEvents)

EXAMPLE = ROOT / "example" / "prod_AuAu_0_10_jet"


def _load(name):
    spec = importlib.util.spec_from_file_location(name[:-3], EXAMPLE / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _hadrons(n, e):
    return {"pid": np.full(n, 211, np.int32), "pstat": np.zeros(n, np.int32),
            "p": np.tile(np.array([e, 0.1, 0.0, 0.0], np.float32), (n, 1)),
            "x": np.zeros((n, 4), np.float32)}


class _Core:
    """Stub pyjetscape_core: iSS gives n_cells hadrons per sample with E = seed % 1000 +
    sample, its primaries 2x as many; Colorless one hadron per parton with E = seed % 1000."""

    SOFT_HAS_SAMPLE_CELLS = True
    PYTHIA_VERSION = 8312
    ISS_DECAYS_SMASH_LIST = True

    def __init__(self):
        self.last = None
        self.prim = None

    def soft_sample_cells(self, iss, cells, seed, n):
        self.last = seed
        parts = [_hadrons(len(cells), seed % 1000 + k) for k in range(n)]
        self.prim = [_hadrons(2 * len(cells), seed % 1000 + k) for k in range(n)]
        return self._flat(parts)

    @staticmethod
    def _flat(parts):
        d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        d["sample_counts"] = np.array([len(p["pid"]) for p in parts])
        return d

    def soft_primaries_numpy(self, iss):
        return self._flat(self.prim[:iss.keep])

    def soft_last_random_seed(self, iss):
        return self.last

    def hadronize_partons(self, module, partons, seed):
        return _hadrons(len(partons), seed % 1000)


# ── jetscape.hadronizer ─────────────────────────────────────────────────────────────
def test_helpers_are_hadronize_py_s():
    hd = _load("hadronize.py")
    assert hd.unit_seed is hz.unit_seed and hd.TAGS == hz.TAGS
    assert hz.unit_seed(1, "jet_frag", 3, 1, prod=5) == hd.unit_seed(1, "jet_frag", 3, 1, 5)
    assert hz.unit_seed(1, "bulk_bg", 0) != hz.unit_seed(1, "bulk_bg", 0, prod=1)
    assert hz.samples_per_background(3, None, 9, 100) == (3, False)
    assert hz.samples_per_background(3, "7", 9, 100) == (7, False)
    assert hz.samples_per_background(3, "auto", 4, 100) == (12, False)
    assert hz.samples_per_background(3, "per-pthat-bin", 50, 100) == (100, True)
    with pytest.raises(ValueError, match="oversample-bg"):
        hz.samples_per_background(3, "zero", 1, 100)
    a = hz.tag_attrs("bulk_bg", _Core(), {"hadron_list": "SMASH", "iss_decays": "x",
                                          "iss_commit": "c", "decayed": True},
                     common_seeds=True, correlated=True)
    assert a["correlated_sampling"] and a["hadron_list"] == "SMASH" and a["decayed"]
    assert hz.tag_attrs("jet_frag", _Core(), {})["pythia_version"] == 8312


def test_configure_hadronize_xml():
    root = ET.parse(EXAMPLE.parent / "config" / "hadronize.xml").getroot()
    hz.configure_hadronize_xml(root, xscape="/xs", seed=7, n_events=0, oversample=3,
                               keep_primaries=hz.KEEP_ALL_PRIMARIES, correlated=True,
                               correlated_block="0.25,0.5,0.75")
    iss = root.find("SoftParticlization/iSS")
    assert root.findtext("nEvents") == "1" and root.findtext("Random/seed") == "7"
    assert iss.findtext("iSS_table_path") == "/xs/external_packages/iSS/iSS_tables"
    assert iss.findtext("number_of_repeated_sampling") == "3"
    assert int(iss.findtext("keep_primaries")) == hz.KEEP_ALL_PRIMARIES
    assert iss.findtext("correlated_sampling") == "1"
    assert iss.findtext("correlated_block_deta") == "0.75"
    with pytest.raises(ValueError, match="DTAU,DX,DETA"):
        hz.configure_hadronize_xml(root, xscape="/xs", seed=1, n_events=1, oversample=1,
                                   keep_primaries=0, correlated=True, correlated_block="1,2")


def test_hadronizer_writes_tags_primaries_and_partons(tmp_path):
    core = _Core()
    iss = SimpleNamespace(keep=2)
    h = hz.Hadronizer(core, iss=iss, colorless=object(), base_seed=1, prod=99)
    stem = tmp_path / "job"
    common = {"source_uuid": "u"}
    h.open("bulk_jet", f"{stem}_hadrons_bulk_jet.h5", n_samples=3, attrs=common,
           initiators=True, primary_tag="bulk_jet_primary",
           primary_path=f"{stem}_hadrons_bulk_jet_primary.h5", keep_primaries=2)
    h.open("bulk_bg", f"{stem}_hadrons_bulk_bg.h5", n_samples=0, attrs=common,
           primary_tag="bulk_bg_primary", primary_path=f"{stem}_hadrons_bulk_bg_primary.h5",
           keep_primaries=hz.KEEP_ALL_PRIMARIES)
    h.open("jet_frag", f"{stem}_hadrons_jet_frag.h5", n_samples=2, attrs=common,
           initiators=True, partons=True)
    cells = np.zeros((4, 32), np.float32)
    ini = np.zeros((1, 11))
    partons = np.ones((3, len(PARTON_COLUMNS)))
    seed = h.bulk_jet_seed(0, 0)
    had, used = h.bulk("bulk_jet", 0, cells, seed, 3, event=0, initiators=ini,
                       keep_primaries=2, pthat=25.0, bg_unit=0)
    assert used == seed == hz.unit_seed(1, "bulk_jet", 0, prod=99) and len(had["pid"]) == 12
    iss.keep = 5
    h.bulk("bulk_bg", 0, cells, h.seed("bulk_bg", 0), 5, event=0,
           keep_primaries=hz.KEEP_ALL_PRIMARIES)
    h.fragments(0, partons, [h.seed("jet_frag", 0, s) for s in range(2)], initiators=ini,
                pthat=25.0, bg_unit=0)
    # an empty surface: an empty unit, no iSS call
    core.last = None
    assert h.bulk("bulk_jet", 1, cells[:0], 5, 3, event=1, initiators=ini,
                  keep_primaries=2)[0] is None and core.last is None
    h.fragments(1, partons[:0], [1, 2], initiators=ini)
    h.set_attrs({"nevents": 2})
    h.close()
    with HadronFile(f"{stem}_hadrons_bulk_jet.h5") as f:
        assert f.n_samples(0) == 3 and f.n_samples(1) == 0 and f.attrs["nevents"] == 2
        assert f.units["pthat"][0] == 25.0 and f.units["seed"][0] == seed
    with HadronFile(f"{stem}_hadrons_bulk_jet_primary.h5") as f:
        assert f.n_samples(0) == 2 and len(f.sample_event(0, 1)["pid"]) == 8
        assert f.attrs["primaries_of"] == "bulk_jet" and not f.attrs["decayed"]
    with HadronFile(f"{stem}_hadrons_bulk_bg_primary.h5") as f:
        assert f.n_samples(0) == 5 and f.attrs["keep_primaries"] == "all"
        assert f.attrs["n_samples"] == 0                 # per unit, as bulk_bg
    with HadronFile(f"{stem}_hadrons_jet_frag.h5") as f:
        np.testing.assert_array_equal(f.partons(0), partons)
        assert f.partons(1).shape == (0, len(PARTON_COLUMNS))
        assert f.f["partons"].attrs["x_had"] == "none" and f.units["n_partons"][0] == 3


def test_partons_group_only_on_jet_frag(tmp_path):
    with pytest.raises(ValueError, match="only jet_frag"):
        HadronH5Writer(tmp_path / "x.h5", tag="bulk_jet", n_samples=1, partons=True)
    with HadronH5Writer(tmp_path / "y.h5", tag="jet_frag", n_samples=1) as w:
        with pytest.raises(ValueError, match="partons=True"):
            w.append_unit([_hadrons(1, 1.0)], partons=np.zeros((1, 14)), unit=0)
    with HadronFile(tmp_path / "y.h5") as f:
        assert not f.has_partons and f.partons(0) is None


# ── readers without a particlize file ───────────────────────────────────────────────
def _hadrons_only_job(tmp_path, name="AuAu_seed0001", uuid="u-1"):
    """Three events, event 1 reuses background 0, as run_prod_jet.py --hadrons-only writes
    them: the event record in the jet files' units/, no particlize file."""
    stem = tmp_path / name
    attrs = {"hadrons_only": True, "source_uuid": uuid, "prod_seed": 1,
             "pthat_bins": np.array([[20.0, 40.0], [40.0, 60.0]])}
    record = [{"bg_unit": 0, "pthat_bin": 0, "pthat": 25.0, "bg_key": "k0"},
              {"bg_unit": 0, "pthat_bin": 1, "pthat": 45.0, "bg_key": "k0"},
              {"bg_unit": 1, "pthat_bin": 0, "pthat": 30.0, "bg_key": "k1"}]
    ini = np.zeros((1, 11))
    with HadronH5Writer(f"{stem}_hadrons_bulk_jet.h5", tag="bulk_jet", n_samples=2,
                        attrs=attrs, initiators=True) as w:
        for e, row in enumerate(record):
            w.append_unit([_hadrons(2, 10 * e + k) for k in range(2)], ini, unit=e, event=e,
                          seed=e, **row)
    with HadronH5Writer(f"{stem}_hadrons_jet_frag.h5", tag="jet_frag", n_samples=1,
                        attrs=attrs, initiators=True, partons=True) as w:
        for e, row in enumerate(record):
            w.append_unit([_hadrons(1, 100 + e)], ini, partons=np.ones((2, 14)), unit=e,
                          event=e, seed=e, **row)
    with HadronH5Writer(f"{stem}_hadrons_bulk_bg.h5", tag="bulk_bg", n_samples=0,
                        attrs=attrs) as w:
        for u, first in ((0, 0), (1, 2)):
            w.append_unit([_hadrons(3, 50 + u) for _ in range(4)], unit=u, event=first,
                          seed=u)
    return stem


def test_reader_needs_no_particlize_file_for_hadrons_only_jobs(tmp_path):
    stem = _hadrons_only_job(tmp_path)
    with HadronFileReader(str(tmp_path)) as r:
        assert r.n_events == 3 and r.tags() == ("bulk_jet", "bulk_bg", "jet_frag")
        assert [r.event_info(e).bg_unit for e in range(3)] == [0, 0, 1]
        assert r.event_info(1).pthat_bin == 1 and r.event_info(0).seed == 1
        assert np.all(r.background_event(1, 3)["p"][:, 0] == 50)    # reused background 0
        assert np.all(r.jet_event(2, 1)["p"][:2, 0] == 21)
        assert r.event_info(2).initiators().shape == (1, 11)
        assert r.pthat_bins.shape == (2, 2)
        assert list(r.pthat_bin_events(0)) == [0, 2]
        n, _ = r.total("bulk_bg")
        assert n == pytest.approx(3.0)
    with JetEvents.from_stem(str(stem)) as je:
        assert [je.bg_unit(e) for e in range(3)] == [0, 0, 1]
    # a hadron file of another run cannot pair up
    with HadronH5Writer(f"{stem}_hadrons_bulk_bg.h5", tag="bulk_bg", n_samples=1,
                        attrs={"hadrons_only": True, "source_uuid": "other"}) as w:
        w.append_unit([_hadrons(1, 1.0)], unit=0, event=0, seed=0)
    with pytest.raises(ValueError, match="not the same run"):
        HadronFileReader(str(tmp_path))


def test_hadron_files_without_particlize_file_or_hadrons_only_are_refused(tmp_path):
    stem = tmp_path / "plain"
    with HadronH5Writer(f"{stem}_hadrons_jet_frag.h5", tag="jet_frag", n_samples=1) as w:
        w.append_unit([_hadrons(1, 1.0)], unit=0, event=0, seed=0)
    with pytest.raises(FileNotFoundError):
        HadronFileReader(str(tmp_path))                 # the directory: nothing found
    with pytest.raises(FileNotFoundError, match="particlize"):
        HadronFileReader(f"{stem}_hadrons_jet_frag.h5")


# ── run_prod_jet.py --hadrons-only options ──────────────────────────────────────────
def _args(**kw):
    a = dict(hadrons_only=True, validate_inline=False, particlize_only=False,
             write_particlize="none", pthat_windows=None, jets_per_bin=1, reuse=3, events=10,
             hadronize_seed=None, oversample=None, oversample_bg=None, oversample_bg_max=None,
             n_frag=None, keep_primaries=None, hadron_keep_bits_p=None,
             hadron_keep_bits_x=None, eta_max=None, charged=False, no_x=False,
             correlated=False, correlated_block=None)
    a.update(kw)
    return SimpleNamespace(**a)


def test_run_prod_jet_hadrons_only_defaults_and_refusals():
    rpj = _load("run_prod_jet.py")
    a = _args()
    rpj.check_hadrons_only(a)
    assert (a.write_particlize, a.particlize_only) == ("both", True)
    assert (a.oversample, a.n_frag, a.oversample_bg, a.oversample_bg_max) == (1, 1, "auto",
                                                                              2000)
    assert a.keep_primaries == "all" and a.keep_primaries_n == hz.KEEP_ALL_PRIMARIES
    assert a.hadronize_seed == 1
    a = _args(pthat_windows=[(20, 40), (40, 60)], keep_primaries="0")
    rpj.check_hadrons_only(a)
    assert a.oversample_bg == "per-pthat-bin" and a.keep_primaries_n == 0
    for bad in (dict(particlize_only=True), dict(write_particlize="jet"),
                dict(validate_inline=True), dict(oversample_bg="per-pthat-bin"),
                dict(keep_primaries="some"), dict(correlated_block="1,1,1"),
                dict(hadron_keep_bits_p=30), dict(oversample=0)):
        with pytest.raises(SystemExit):
            rpj.check_hadrons_only(_args(**bad))
    # hadron options without --hadrons-only
    with pytest.raises(SystemExit, match="--oversample needs --hadrons-only"):
        rpj.check_hadrons_only(_args(hadrons_only=False, oversample=3))
    rpj.check_hadrons_only(_args(hadrons_only=False))        # nothing given: fine


def test_background_uses_follow_reuse_and_windows():
    rpj = _load("run_prod_jet.py")
    ho = rpj.HadronsOnly.__new__(rpj.HadronsOnly)
    ho.a = _args(events=10, reuse=3, oversample_bg="auto")
    ho.per_bg = 3
    assert [ho._background_uses(i) for i in (0, 3, 9)] == [3, 3, 1]
    ho.a = _args(events=7, pthat_windows=[(1, 2), (2, 3)], jets_per_bin=2,
                 oversample_bg="per-pthat-bin")
    ho.per_bg = 4
    assert [ho._background_uses(i) for i in (0, 4)] == [2, 2]   # 3 events left, 2 windows
    assert ho._background_uses(6) == 1


# ── hadronize.py on a --hadrons-only jet file ───────────────────────────────────────
def test_hadronize_reads_a_hadrons_only_jet_file(tmp_path):
    hd = _load("hadronize.py")
    stem = _hadrons_only_job(tmp_path, uuid="12345678-1234-5678-1234-567812345678")
    pf = hd.open_input(f"{stem}_hadrons_jet_frag.h5")
    assert isinstance(pf, hd.JetFileInput) and pf.legs == () and pf.nevents == 3
    assert hd.production_key(pf) == hz.uuid_key("12345678-1234-5678-1234-567812345678")
    assert pf.partons(1).shape == (2, 14) and list(pf.events("bg_unit")) == [0, 0, 1]
    data, off = pf.initiators_all()
    assert data.shape == (3, 11) and list(off) == [0, 1, 2, 3]
    assert pf.bg_units()[0] == 0
    pf.close()
    plain = tmp_path / "plain_hadrons_jet_frag.h5"         # a hadronize.py jet file
    with HadronH5Writer(plain, tag="jet_frag", n_samples=1) as w:
        w.append_unit([_hadrons(1, 1.0)], unit=0, event=0, seed=0)
    with pytest.raises(ValueError, match="partons"):
        hd.open_input(str(plain))
