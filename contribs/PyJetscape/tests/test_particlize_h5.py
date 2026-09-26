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
