"""
tests/test_particlize_h5.py

Tests for the hadronization-input and hadron files (PLAN_particlize_h5.md):
jetscape.particlize_h5 (surfaces + final partons), jetscape.hadrons_h5 (samples of hadrons)
and PairH5Writer's keep_surface.  Nothing here needs the compiled extension or MUSIC; the
column lists are checked against the extension when it is importable.

    pytest tests/test_particlize_h5.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

h5py = pytest.importorskip("h5py")

from jetscape.hadrons_h5 import HadronH5Writer, Hadrons  # noqa: E402
from jetscape.particlize_h5 import (FINAL_PARTON_COLUMNS, SURFACE_COLUMNS,  # noqa: E402
                                    ParticlizeFile, ParticlizeH5Writer)

NCOL = len(SURFACE_COLUMNS)


class _Leg:
    """Duck-types MpiMusic's surface side."""

    def __init__(self, name, skip=False):
        self.name, self.skip, self.cells = name, skip, np.zeros((0, NCOL), np.float32)

    def GetId(self):
        return self.name

    def get_skip_surface(self):
        return self.skip

    def set_skip_surface(self, v):
        self.skip = bool(v)

    def surface_to_numpy(self):
        return self.cells


class _Mgr:
    def __init__(self):
        self.rows = np.zeros((0, len(FINAL_PARTON_COLUMNS)))

    def final_partons_numpy(self):
        return self.rows


class _JS:
    def __init__(self, *tasks):
        self.tasks = tasks

    def GetTaskList(self):
        return list(self.tasks)


def _cells(n, value):
    c = np.full((n, NCOL), value, np.float32)
    c[:, 0] = np.arange(n)                          # tau: distinguishes the rows
    return c


def _partons(n, shower=0):
    p = np.zeros((n, len(FINAL_PARTON_COLUMNS)))
    p[:, 0], p[:, 1], p[:, 3] = shower, 21, 10.0
    return p


def _writer(tmp_path, legs=("jet", "bg"), **kw):
    jet, bg, mgr = _Leg("MUSIC_2"), _Leg("MUSIC_1"), _Mgr()
    w = ParticlizeH5Writer(tmp_path / "p.h5", legs=legs, music_input="EOS_to_use 9\n",
                           T_fo=0.15, pair_file="/x/pair.h5", **kw)
    w.attach(_JS(bg, jet), manager=mgr)
    return w, jet, bg, mgr


def test_surfaces_partons_and_background_reuse(tmp_path):
    w, jet, bg, mgr = _writer(tmp_path)
    # three events, the second reuses the first one's background (bg_id 0)
    for k, (bg_id, nj, nb, npart) in enumerate([(0, 5, 4, 3), (0, 6, 99, 0), (2, 2, 3, 2)]):
        jet.cells, bg.cells, mgr.rows = _cells(nj, k), _cells(nb, 10 + k), _partons(npart)
        assert w.Exec(k, bg_id=bg_id, bg_key=f"key{bg_id}", E_droplets=1.5 * k) == k
    w.write_events(1, wall_s=3.0)
    w.Finish()

    with ParticlizeFile(tmp_path / "p.h5") as pf:
        assert pf.nevents == 3 and pf.legs == ("jet", "bg")
        assert pf.attrs["complete"] and pf.attrs["T_fo"] == 0.15
        assert pf.attrs["pair_file"] == "pair.h5"
        assert pf.music_input() == "EOS_to_use 9\n"
        assert list(pf.attrs["surface_columns"]) == list(SURFACE_COLUMNS)
        np.testing.assert_array_equal(pf.surface("jet", 1), _cells(6, 1))
        # event 1 reused background 0: its 99-cell surface was never read
        n_bg, ids = pf.bg_units()
        assert n_bg == 2 and list(ids) == [0, 2]
        assert list(pf.events("bg_unit")) == [0, 0, 1]
        np.testing.assert_array_equal(pf.surface("bg", 1), _cells(4, 10))
        np.testing.assert_array_equal(pf.surface("bg", 2), _cells(3, 12))
        assert list(pf.events("n_cells_bg")) == [4, 4, 3]
        assert pf.partons(1).shape == (0, len(FINAL_PARTON_COLUMNS))
        assert pf.partons(2).shape == (2, len(FINAL_PARTON_COLUMNS))
        assert list(pf.events("n_partons")) == [3, 0, 2]
        assert list(pf.events("E_droplets")) == [0.0, 1.5, 3.0]
        wall = pf.events("wall_s")
        assert np.isnan(wall[0]) and wall[1] == 3.0
        assert [x.decode() if isinstance(x, bytes) else x for x in pf.events("bg_key")] \
            == ["key0", "key0", "key2"]
        assert pf.f["surface/jet/cells"].dtype == np.float32


def test_empty_surface_is_an_empty_unit(tmp_path):
    w, jet, bg, mgr = _writer(tmp_path, legs=("jet",))
    jet.cells = _cells(3, 1)
    w.Exec(0)
    jet.cells = np.zeros((0, NCOL), np.float32)
    with pytest.warns(RuntimeWarning, match="empty surface"):
        w.Exec(1)
    w.Finish()
    with ParticlizeFile(tmp_path / "p.h5") as pf:
        assert pf.legs == ("jet",) and "surface/bg" not in pf.f
        assert list(pf.events("n_cells_jet")) == [3, 0]
        assert pf.surface("jet", 1).shape == (0, NCOL)


def test_a_stored_leg_with_skip_surface_is_switched_on(tmp_path):
    jet, bg = _Leg("MUSIC_2", skip=True), _Leg("MUSIC_1", skip=True)
    w = ParticlizeH5Writer(tmp_path / "p.h5", legs=("jet",), music_input="")
    with pytest.warns(RuntimeWarning, match="skip_surface"):
        w.attach(_JS(bg, jet), manager=_Mgr())
    assert not jet.skip and bg.skip                  # only the stored leg
    w.Finish()


def test_event_index_must_follow_the_pair_writer(tmp_path):
    w, jet, bg, mgr = _writer(tmp_path)
    w.Exec(0)
    with pytest.raises(ValueError, match="skip the same events"):
        w.Exec(2)
    w.Finish()


def test_columns_match_the_extension():
    core = pytest.importorskip("jetscape.pyjetscape_core")
    assert tuple(core.SURFACE_CELL_COLUMNS) == SURFACE_COLUMNS
    assert tuple(core.FINAL_PARTON_COLUMNS) == FINAL_PARTON_COLUMNS


def test_pair_writer_keeps_only_the_listed_surfaces(tmp_path):
    from test_pair_h5 import _evo, _pair_writer

    w, bg, jet, liq = _pair_writer(tmp_path, _evo(3, 1.0), _evo(3, 1.0), keep_surface=("jet",))
    assert not jet.skip and bg.skip
    w.Exec()
    assert w.last_bg_key is not None and len(w.last_bg_key) == 32
    w.Finish()
    with pytest.raises(ValueError, match="keep_surface"):
        _pair_writer(tmp_path, _evo(3, 1.0), _evo(3, 1.0), keep_surface=("both",))


# ───────────────────────────────────────────── hadron files
def _hadrons(n, pid=211, e=1.0):
    return {"pid": np.full(n, pid, np.int32), "pstat": np.zeros(n, np.int32),
            "p": np.tile(np.array([e, 0.5, 0.0, 0.1], np.float32), (n, 1)),
            "x": np.zeros((n, 4), np.float32)}


def test_hadron_file_keeps_samples_and_units_apart(tmp_path):
    p = tmp_path / "h.h5"
    with HadronH5Writer(p, tag="bulk_jet", n_samples=2, attrs={"source": "p.h5"}) as w:
        both = {k: np.concatenate([_hadrons(3)[k], _hadrons(1, pid=-211)[k]])
                for k in ("pid", "pstat", "p", "x")}
        both["sample_counts"] = np.array([3, 1])
        assert w.append_unit(both, unit=0, event=0, seed=11) == 0     # soft_hadrons form
        assert w.append_unit([], unit=1, event=1, seed=12) == 1        # empty surface
        assert w.append_unit([_hadrons(2), _hadrons(4)], unit=2, event=2, seed=13) == 2

    h = Hadrons.from_h5(p)
    assert h.attrs["tag"] == "bulk_jet" and h.attrs["complete"]
    assert h.n_units == 3 and list(h.samples_per_unit) == [2, 0, 2]
    assert list(np.diff(h.sample_offsets)) == [3, 1, 2, 4]
    assert list(h.units["seed"]) == [11, 12, 13]
    assert list(h.unit) == [0, 0, 0, 0, 2, 2, 2, 2, 2, 2]
    # per-sample average over all units: 10 hadrons / 4 samples (unit 1 has none)
    n, err = h.total()
    assert n == pytest.approx(10 / 4) and err == pytest.approx(np.sqrt(10) / 4)
    n0, _ = h.total(units=[0])
    assert n0 == pytest.approx(4 / 2)
    npi, _ = h.total(mask=h.species("pi+"), units=[0])
    assert npi == pytest.approx(3 / 2)
    hist, _ = h.hist(h.pt, np.array([0.0, 1.0]), units=[2])
    assert hist[0] == pytest.approx(6 / 2)

    sub = Hadrons.from_h5(p, units=[2])
    assert sub.n_units == 1 and len(sub.pid) == 6 and list(sub.units["seed"]) == [13]


def test_hadron_file_rejects_unknown_tags(tmp_path):
    with pytest.raises(ValueError, match="tag"):
        HadronH5Writer(tmp_path / "h.h5", tag="soft", n_samples=1)


# ───────────────────────────────────────────── single events and JetEvents
def _tagged(n, pid, e):
    """n hadrons of one pid whose energy e identifies the sample they came from."""
    return _hadrons(n, pid=pid, e=e)


def _campaign(tmp_path):
    """Three events; event 1 reuses event 0's background; event 2 has an empty surface.

    bulk_jet: 3 oversamples per event (energy 10*event + k); jet_frag: 2 fragmentations
    (energy 100 + 10*event + j); bulk_bg: 3 oversamples per background (energy 50 + unit)."""
    from jetscape.hadrons_h5 import HadronH5Writer

    stem = tmp_path / "run"
    with HadronH5Writer(f"{stem}_hadrons_bulk_jet.h5", tag="bulk_jet", n_samples=3) as w:
        for ev in range(3):
            samples = [] if ev == 2 else [_tagged(2 + k, 211, 10 * ev + k) for k in range(3)]
            w.append_unit(samples, unit=ev, event=ev, seed=ev)
    with HadronH5Writer(f"{stem}_hadrons_jet_frag.h5", tag="jet_frag", n_samples=2) as w:
        for ev in range(3):
            w.append_unit([_tagged(1, 2212, 100 + 10 * ev + j) for j in range(2)],
                          unit=ev, event=ev, seed=ev)
    with HadronH5Writer(f"{stem}_hadrons_bulk_bg.h5", tag="bulk_bg", n_samples=3) as w:
        for unit, first in ((0, 0), (1, 2)):
            w.append_unit([_tagged(4, -211, 50 + unit) for _ in range(3)], unit=unit,
                          event=first, seed=unit)
    # the particlize file only has to carry events/bg_unit for JetEvents
    pw, jet, bg, mgr = _writer(tmp_path, legs=("jet", "bg"))
    for k, bg_id in enumerate((0, 0, 2)):
        jet.cells, bg.cells = _cells(1, k), _cells(1, 10 + k)
        pw.Exec(k, bg_id=bg_id)
    pw.Finish()
    (tmp_path / "p.h5").rename(f"{stem}_particlize.h5")
    return stem


def test_every_oversample_is_an_event(tmp_path):
    from jetscape.hadrons_h5 import HadronFile, Hadrons

    stem = _campaign(tmp_path)
    path = f"{stem}_hadrons_bulk_jet.h5"
    h = Hadrons.from_h5(path)
    with HadronFile(path) as hf:
        for src in (h, hf):
            assert src.n_samples(1) == 3 and src.n_samples(2) == 0
            ev = src.sample_event(1, 2)                  # event 1, oversample 2
            assert len(ev["pid"]) == 4 and np.all(ev["p"][:, 0] == 12)
            assert ev["p"].dtype == np.float32 and ev["x"].shape == (4, 4)
            with pytest.raises(IndexError, match="empty surface"):
                src.sample_event(2, 0)
            with pytest.raises(IndexError, match="3 sample"):
                src.sample_event(0, 3)
            with pytest.raises(IndexError, match="no unit 7"):
                src.sample_event(7, 0)
    # a subset keeps the recorded unit ids
    sub = Hadrons.from_h5(path, units=[1])
    assert np.all(sub.sample_event(1, 0)["p"][:, 0] == 10)


def test_jet_events_combine_bulk_and_fragments_per_event(tmp_path):
    from jetscape.hadrons_h5 import ORIGIN, JetEvents

    stem = _campaign(tmp_path)
    with JetEvents.from_stem(str(stem)) as je:
        assert je.n_oversamples(0) == 3 and je.n_frag(0) == 2
        ev = je.jet_event(1, 2)                          # oversample 2 -> fragmentation 0
        bulk, frag = ev["origin"] == ORIGIN["bulk"], ev["origin"] == ORIGIN["frag"]
        assert bulk.sum() == 4 and np.all(ev["p"][bulk, 0] == 12)
        assert frag.sum() == 1 and ev["p"][frag, 0][0] == 110 and ev["pid"][frag][0] == 2212
        assert je.jet_event(1, 1)["p"][-1, 0] == 111     # k mod n_frag
        assert je.jet_event(1, 1, frag_sample=0)["p"][-1, 0] == 110
        assert np.all(je.jet_event(0, 0, fragments=False)["origin"] == ORIGIN["bulk"])
        assert len(list(je.iter_jet_events(0))) == 3
        assert list(je.iter_jet_events(2)) == []         # empty surface: no oversamples
        # backgrounds through events/bg_unit: event 1 reused event 0's
        assert [je.bg_unit(e) for e in range(3)] == [0, 0, 1]
        assert np.all(je.background_event(1, 0)["p"][:, 0] == 50)
        assert np.all(je.background_event(2, 2)["p"][:, 0] == 51)
    # without the particlize file only events that started a background resolve
    with JetEvents(bulk_bg=f"{stem}_hadrons_bulk_bg.h5") as je:
        assert je.bg_unit(2) == 1
        with pytest.raises(ValueError, match="particlize"):
            je.bg_unit(1)
    with pytest.raises(ValueError, match="tag"):
        JetEvents(bulk_jet=f"{stem}_hadrons_jet_frag.h5")


# ───────────────────────────────────────────── HadronFileReader: a campaign
PHI_JET = {("A", 0): 0.3, ("A", 1): 2.0, ("B", 0): -1.0, ("B", 1): 1.2}


def _hads(phis, pid, e):
    n = len(phis)
    p = np.zeros((n, 4), np.float32)
    p[:, 0], p[:, 1], p[:, 2] = e, np.cos(phis), np.sin(phis)
    return {"pid": np.full(n, pid, np.int32), "pstat": np.zeros(n, np.int32), "p": p,
            "x": np.zeros((n, 4), np.float32)}


def _seed_files(tmp_path, name, bg_ids, empty_jet=(), n_os=2, n_jet=3):
    """One production file: particlize (with bg map + uuid), pair file (initiators), and
    the three hadron files.  Jet-leg hadrons sit at phi_jet + 0.1 (energy 10*event + k + 1),
    background hadrons at a fixed phi 0.5 (energy 100 + unit), fragments along the jet."""
    import h5py

    from jetscape.hadrons_h5 import HadronH5Writer

    stem = str(tmp_path / f"run_{name}")
    jet, bg, mgr = _Leg("MUSIC_2"), _Leg("MUSIC_1"), _Mgr()
    pw = ParticlizeH5Writer(f"{stem}_particlize.h5", legs=("jet", "bg"), music_input="",
                            pair_file=f"/elsewhere/run_{name}.h5",
                            extra_attrs={"prod_seed": ord(name)})
    pw.attach(_JS(bg, jet), manager=mgr)
    for k, bg_id in enumerate(bg_ids):
        jet.cells, bg.cells = _cells(1, k), _cells(1, 10 + k)
        pw.Exec(k, bg_id=bg_id)
    pw.Finish()
    with ParticlizeFile(f"{stem}_particlize.h5") as pf:
        uuid, bg_unit = pf.attrs["file_uuid"], pf.events("bg_unit")
    n_ev = len(bg_ids)
    with h5py.File(f"{stem}.h5", "w") as f:                          # the pair file
        ini = np.zeros((n_ev, 11))
        for e in range(n_ev):
            ini[e, 3], ini[e, 4] = np.cos(PHI_JET[(name, e)]), np.sin(PHI_JET[(name, e)])
        f["shower/initiators"] = ini
        f["shower/initiator_offsets"] = np.arange(n_ev + 1)
    attrs = {"source_uuid": uuid}
    with HadronH5Writer(f"{stem}_hadrons_bulk_jet.h5", tag="bulk_jet", n_samples=2,
                        attrs=attrs) as w:
        for e in range(n_ev):
            s = [] if e in empty_jet else [_hads([PHI_JET[(name, e)] + 0.1] * n_jet, 211,
                                                 10 * e + k + 1) for k in range(n_os)]
            w.append_unit(s, unit=e, event=e, seed=e)
    with HadronH5Writer(f"{stem}_hadrons_bulk_bg.h5", tag="bulk_bg", n_samples=2,
                        attrs=attrs) as w:
        for u in range(int(bg_unit.max()) + 1):
            w.append_unit([_hads([0.5, 0.5], -211, 100 + u) for _ in range(2)], unit=u,
                          event=int(np.flatnonzero(bg_unit == u)[0]), seed=u)
    with HadronH5Writer(f"{stem}_hadrons_jet_frag.h5", tag="jet_frag", n_samples=1,
                        attrs=attrs) as w:
        for e in range(n_ev):
            w.append_unit([_hads([PHI_JET[(name, e)]], 2212, 50)], unit=e, event=e, seed=e)
    return stem


def _two_seeds(tmp_path, empty_jet_b=()):
    # seed A: both events share background 0; seed B: one background per event
    return (_seed_files(tmp_path, "A", (0, 0)), _seed_files(tmp_path, "B", (0, 1),
                                                            empty_jet=empty_jet_b))


def test_reader_finds_and_indexes_a_campaign(tmp_path):
    from jetscape.hadrons_h5 import HadronFileReader, JetEvents

    a, b = _two_seeds(tmp_path)
    for source in (str(tmp_path), str(tmp_path / "*_hadrons_bulk_jet.h5"), [b, a],
                   [a + "_particlize.h5", b + "_particlize.h5"]):
        with HadronFileReader(source) as r:
            assert r.stems == [a, b] and r.n_files == 2 and r.n_events == 4
    with HadronFileReader(str(tmp_path)) as r:
        assert r.tags() == ("bulk_jet", "bulk_bg", "jet_frag")
        assert r.locate(3) == (1, 1) and r.global_event(1, 0) == 2
        info = r.event_info(1)
        assert (info.stem, info.local_event, info.bg_unit, info.seed) == (a, 1, 0, ord("A"))
        assert np.allclose(info.initiators()[0, 3:5],
                           [np.cos(PHI_JET[("A", 1)]), np.sin(PHI_JET[("A", 1)])])
        assert r.n_oversamples(2) == 2 and r.n_frag(2) == 1
        with JetEvents.from_stem(b) as je:
            ref = je.jet_event(1, 1)
        ev = r.jet_event(3, 1)                           # global 3 = seed B, event 1
        assert all(np.array_equal(ev[k], ref[k]) for k in ("pid", "p", "origin"))
        assert np.all(r.background_event(1, 0)["p"][:, 0] == 100)   # reused bg unit 0
        with pytest.raises(IndexError):
            r.locate(4)


def test_reader_histograms_follow_each_event_and_count_reused_backgrounds(tmp_path):
    from jetscape.hadrons_h5 import HadronFileReader

    _two_seeds(tmp_path)
    with HadronFileReader(str(tmp_path)) as r:
        # counts: 3 per jet sample, 2 per background sample (4 events x 2 samples each)
        assert r.total("bulk_jet") == pytest.approx((3.0, np.sqrt(24) / 8))
        n, err = r.total("bulk_bg", weights="E")
        # events A0 and A1 both use seed A's unit 0 (E = 100); B0, B1 units 0, 1 (100, 101)
        e_sum = 2 * 2 * (2 * 100) + 2 * (2 * 100) + 2 * (2 * 101)
        assert n == pytest.approx(e_sum / 8)
        # unit A0's hadrons are counted twice in the same bin: their weights add first
        sq = 4 * (2 * 100) ** 2 + 4 * 100 ** 2 + 4 * 101 ** 2
        assert err == pytest.approx(np.sqrt(sq) / 8)

        def dphi(ev, info):                              # relative to this event's jet
            ini = info.initiators()[0]
            return np.mod(ev.phi - np.arctan2(ini[4], ini[3]), 2 * np.pi)

        bins = np.linspace(0, 2 * np.pi, 64)
        h, _ = r.hist("bulk_jet", dphi, bins)
        assert h[np.searchsorted(bins, 0.1) - 1] == pytest.approx(3.0)   # all at +0.1
        # the reused background lands in a different bin for each of its two events
        hb, eb = r.hist("bulk_bg", dphi, bins, events=[0, 1])
        i0 = np.searchsorted(bins, np.mod(0.5 - 0.3, 2 * np.pi)) - 1
        i1 = np.searchsorted(bins, np.mod(0.5 - 2.0, 2 * np.pi)) - 1
        assert hb[i0] == pytest.approx(1.0) and hb[i1] == pytest.approx(1.0)
        assert eb[i0] == pytest.approx(np.sqrt(4) / 4)                 # not correlated
        # 2-d, with a name as mask
        h2, _ = r.hist("bulk_jet", lambda ev, info: (ev.pt, ev.phi),
                       (np.array([0.0, 2.0]), np.array([-np.pi, np.pi])), mask="charged")
        assert h2.shape == (1, 1) and h2[0, 0] == pytest.approx(3.0)


def test_reader_jet_minus_background_uses_common_events(tmp_path):
    from jetscape.hadrons_h5 import HadronFileReader

    _two_seeds(tmp_path, empty_jet_b=(1,))               # seed B event 1: empty surface
    with HadronFileReader(str(tmp_path)) as r:
        assert r.n_oversamples(3) == 0
        bins = np.array([0.0, 10.0])
        d, e = r.jet_minus_background("pt", bins)
        common = [0, 1, 2]                               # event 3 has no jet samples
        hj, ej = r.hist("bulk_jet", "pt", bins, events=common)
        hb, eb = r.hist("bulk_bg", "pt", bins, events=common)
        assert d == pytest.approx(hj - hb) and e == pytest.approx(np.hypot(ej, eb))
        assert d[0] == pytest.approx(3 - 2)
        df, _ = r.jet_minus_background("pt", bins, fragments=True)
        assert df[0] == pytest.approx(3 - 2 + 1)


def test_reader_refuses_files_from_another_run(tmp_path):
    import h5py

    from jetscape.hadrons_h5 import HadronFileReader

    a, b = _two_seeds(tmp_path)
    with h5py.File(b + "_hadrons_bulk_bg.h5", "a") as f:
        f.attrs["source_uuid"] = "someone-else"
    with pytest.raises(ValueError, match="not the same run"):
        HadronFileReader(str(tmp_path))
    with HadronFileReader(str(tmp_path), check_uuid=False) as r:
        assert r.n_events == 4
    import os
    os.remove(a + "_hadrons_jet_frag.h5")
    with HadronFileReader(str(tmp_path), check_uuid=False) as r:
        assert r.tags() == ("bulk_jet", "bulk_bg") and "jet_frag" in r.tags(1)
        with pytest.raises(ValueError, match="no jet_frag file"):
            r.total("jet_frag")


def test_reader_weighs_every_event_the_same(tmp_path):
    from jetscape.hadrons_h5 import HadronFileReader

    _seed_files(tmp_path, "A", (0, 0), n_os=2, n_jet=3)      # 3 hadrons per sample
    _seed_files(tmp_path, "B", (0, 1), n_os=4, n_jet=5)      # 5, and twice the samples
    with HadronFileReader(str(tmp_path)) as r:
        n, err = r.total("bulk_jet")
        assert n == pytest.approx((3 + 3 + 5 + 5) / 4)     # not the sample-weighted 4.33
        assert err == pytest.approx(np.sqrt(2 * 6 / 2 ** 2 + 2 * 20 / 4 ** 2) / 4)



# ───────────────────────────────────────────── hadronize.py / run_hadronize.py helpers
def _load_example(name):
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "example" / "prod_AuAu_0_10_jet" / name
    spec = importlib.util.spec_from_file_location(name[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _particlize_with_backgrounds(path, bg_ids, complete=True):
    jet, bg = _Leg("MUSIC_2"), _Leg("MUSIC_1")
    w = ParticlizeH5Writer(path, legs=("jet", "bg"), music_input="")
    w.attach(_JS(bg, jet), manager=_Mgr())
    for k, b in enumerate(bg_ids):
        jet.cells, bg.cells = _cells(1, k), _cells(1, 10 + k)
        w.Exec(k, bg_id=b)
    w.Finish(complete=complete)


def test_oversample_bg_per_background(tmp_path):
    hz = _load_example("hadronize.py")
    p = tmp_path / "x_particlize.h5"
    _particlize_with_backgrounds(p, (0, 0, 0, 3))        # background 0 used 3x, 1 once
    with ParticlizeFile(p) as pf:
        assert hz.background_samples(pf, 100, None, 2000) == ({0: 100, 1: 100}, [])
        assert hz.background_samples(pf, 100, "7", 2000) == ({0: 7, 1: 7}, [])
        assert hz.background_samples(pf, 100, "auto", 2000) == ({0: 300, 1: 100}, [])
        assert hz.background_samples(pf, 100, "auto", 250) == ({0: 250, 1: 100}, [0])
        with pytest.raises(ValueError):
            hz.background_samples(pf, 100, "0", 2000)


def test_run_hadronize_finds_complete_inputs_and_finished_outputs(tmp_path):
    from jetscape.hadrons_h5 import HadronH5Writer

    rh = _load_example("run_hadronize.py")
    done, todo, busy = (str(tmp_path / f"{n}_particlize.h5") for n in ("done", "todo", "busy"))
    for p in (done, todo):
        _particlize_with_backgrounds(p, (0, 1))
    _particlize_with_backgrounds(busy, (0,), complete=False)   # its job is still running
    for tag in ("bulk_jet", "bulk_bg"):
        with HadronH5Writer(str(tmp_path / f"done_hadrons_{tag}.h5"), tag=tag, n_samples=1):
            pass
    with HadronH5Writer(str(tmp_path / "todo_hadrons_bulk_jet.h5"), tag="bulk_jet",
                        n_samples=1) as w:
        w.close(complete=False)                                   # an interrupted pass
    assert rh.scan([str(tmp_path)]) == sorted([busy, done, todo])
    assert rh.scan([str(tmp_path / "d*")]) == [done]
    assert [rh.is_complete(p) for p in (done, todo, busy)] == [True, True, False]
    plan = rh.Plan(["--tags", "bulk_jet,bulk_bg", "--oversample", "100",
                    "--oversample-bg", "auto"])
    assert "--skip-complete" in plan.args
    assert plan.all_done(done) and not plan.all_done(todo)
    assert plan.memory_gb(done) == pytest.approx(rh.GB_BASE + rh.GB_PER_OVERSAMPLE * 100)
    assert not rh.Plan(["--force"]).all_done(done)
    assert rh.Plan(["--tags", "jet_frag"]).memory_gb(done) == rh.GB_FRAG_ONLY
