"""
tests/test_pair_h5.py

Tests for the pair extensions of jetscape.fno_h5_writer (a second evolution, ragged
per-event tables, per-event diagnostics, tau-axis discovery in repad_to).  Nothing here
needs the compiled extension or MUSIC.

    pytest tests/test_pair_h5.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from jetscape.fno_h5_writer import (FREEZEOUT_CONVENTION_ID, FnoH5Writer, grid_attrs,
                                    repad_to)

h5py = pytest.importorskip("h5py")

NX, NY, NETA = 5, 4, 3


def _attrs(choose_ntau=1):
    return grid_attrs(NX, NY, NETA, 0.5, 0.5, 0.4, tau_min=0.5, dtau=0.1,
                      choose_ntau=choose_ntau)


def _frame(value):
    return np.full((4, NX, NY, NETA), value, dtype=np.float32)


def _write_pair(w, i, n_jet, n_bg, jet_value=2.0, bg_value=1.0):
    """One pair event: the jet leg lives n_jet frames, the background n_bg."""
    w.ensure_capacity(nevents=i + 1, choose_ntau=max(n_jet, n_bg))
    for t in range(n_bg):
        w.write_frame(i, t, _frame(bg_value), dataset="arr_bg")
    for t in range(n_jet):
        w.write_frame(i, t, _frame(jet_value))
    w.set_event_meta(i, n_bg, 0.5 + n_bg * 0.1, dataset="arr_bg")
    w.set_event_meta(i, n_jet, 0.5 + n_jet * 0.1)


# ───────────────────────────────────────────── a second evolution
def test_second_evolution_keeps_arr_shape_with_its_own_freezeout(tmp_path):
    p = tmp_path / "pair.h5"
    with FnoH5Writer(p, _attrs()) as w:
        w.add_evolution("arr_bg", fo_suffix="_bg")
        _write_pair(w, 0, n_jet=6, n_bg=4)
        _write_pair(w, 1, n_jet=3, n_bg=5)          # here the background outlives the jet
    with h5py.File(p, "r") as f:
        arr, bg = f["arr"], f["arr_bg"]
        assert arr.shape == bg.shape == (2, 4, NX, NY, NETA, 6)
        assert bg.chunks == arr.chunks and bg._filters == arr._filters
        assert bg.attrs["compression"] == arr.attrs["compression"]
        assert bg.maxshape == arr.maxshape
        assert arr.attrs["tau_axis"] == 5 and bg.attrs["tau_axis"] == 5
        assert list(f["ntau_freezeout"][:]) == [6, 3]
        assert list(f["ntau_freezeout_bg"][:]) == [4, 5]
        # each leg is exactly zero after its own last frame
        assert np.all(bg[0, ..., 4:] == 0) and np.all(bg[0, ..., :4] == 1)
        assert np.all(arr[1, ..., 3:] == 0) and np.all(arr[1, ..., :3] == 2)
        assert f.attrs["nevents"] == 2 and f.attrs["choose_ntau"] == 6
        assert f.attrs["nevents_written"] == 2 and f.attrs["complete"]
        assert f.attrs["freezeout_convention_id"] == FREEZEOUT_CONVENTION_ID
        assert not f.attrs["has_source"]


def test_only_the_primary_evolution_marks_an_event_written(tmp_path):
    p = tmp_path / "pair.h5"
    with FnoH5Writer(p, _attrs()) as w:
        w.add_evolution("arr_bg", fo_suffix="_bg")
        w.ensure_capacity(nevents=1, choose_ntau=2)
        w.set_event_meta(0, 2, 0.7, dataset="arr_bg")
        assert w.nevents_written == 0
        w.set_event_meta(0, 2, 0.7)
        assert w.nevents_written == 1
        assert w.evolutions == ("arr", "arr_bg")


def test_an_evolution_added_late_reads_zero_for_earlier_events(tmp_path):
    p = tmp_path / "late.h5"
    with FnoH5Writer(p, _attrs()) as w:
        w.append_event(0, np.ones((4, NX, NY, NETA, 3), np.float32), 3, 0.8)
        w.add_evolution("arr_bg", fo_suffix="_bg")
        assert w.f["arr_bg"].shape == w.arr.shape
    with h5py.File(p, "r") as f:
        assert np.all(f["arr_bg"][0] == 0)


def test_add_evolution_refuses_a_duplicate(tmp_path):
    with FnoH5Writer(tmp_path / "dup.h5", _attrs()) as w:
        w.add_evolution("arr_bg", fo_suffix="_bg")
        with pytest.raises(ValueError, match="already exists"):
            w.add_evolution("arr_bg", fo_suffix="_bg2")


# ───────────────────────────────────────────── ragged tables
def test_ragged_tables_stay_aligned_with_events_including_empty_ones(tmp_path):
    p = tmp_path / "ragged.h5"
    counts = [3, 0, 5]
    with FnoH5Writer(p, _attrs()) as w:
        g = w.ragged("source", "offsets", {"droplets": (np.float64, (8,))},
                     attrs={"droplet_columns": ["tau", "x", "y", "eta", "E", "px", "py", "pz"]})
        for i, n in enumerate(counts):
            w.ensure_capacity(nevents=i + 1, choose_ntau=1)
            w.write_frame(i, 0, _frame(1.0))
            g.append({"droplets": np.full((n, 8), float(i))} if n else None)
            w.set_event_meta(i, 1, 0.6)
    with h5py.File(p, "r") as f:
        off = f["source/offsets"][:]
        d = f["source/droplets"][:]
        assert list(np.diff(off)) == counts
        assert d.shape == (sum(counts), 8) and off[-1] == len(d)
        assert np.all(d[off[2]:off[3]] == 2.0)          # event 2's rows, not event 0's
        assert off[1] == off[2]                          # event 1 is empty, not missing
        assert list(f["source"].attrs["droplet_columns"])[4] == "E"


def test_ragged_fields_share_offsets_and_must_agree_on_row_count(tmp_path):
    with FnoH5Writer(tmp_path / "shared.h5", _attrs()) as w:
        g = w.ragged("hadrons", "offsets",
                     {"p": (np.float32, (4,)), "pid": (np.int32, ())})
        assert g.append({"p": np.zeros((2, 4)), "pid": [211, -211]}) == 2
        with pytest.raises(ValueError, match="share one offsets"):
            g.append({"p": np.zeros((2, 4)), "pid": [211]})
        with pytest.raises(KeyError):
            g.append({"nope": [1]})


def test_a_ragged_table_created_after_some_events_starts_with_empty_ones(tmp_path):
    p = tmp_path / "lazy.h5"
    with FnoH5Writer(p, _attrs()) as w:
        for i in range(2):
            w.append_event(i, np.ones((4, NX, NY, NETA, 1), np.float32), 1, 0.6)
        g = w.ragged("shower", "parton_offsets", {"partons": (np.float64, (13,))})
        assert g.units_written == 2
        w.ensure_capacity(nevents=3, choose_ntau=1)
        g.append({"partons": np.ones((4, 13))})
        w.set_event_meta(2, 1, 0.6)
    with h5py.File(p, "r") as f:
        assert list(f["shower/parton_offsets"][:]) == [0, 0, 0, 4]


def test_set_event_meta_refuses_an_event_a_ragged_table_missed(tmp_path):
    with FnoH5Writer(tmp_path / "missed.h5", _attrs()) as w:
        w.ragged("source", "offsets", {"droplets": (np.float64, (8,))})
        w.ensure_capacity(nevents=1, choose_ntau=1)
        with pytest.raises(ValueError, match="append exactly"):
            w.set_event_meta(0, 1, 0.6)


def test_a_free_ragged_table_is_not_checked_per_event(tmp_path):
    with FnoH5Writer(tmp_path / "free.h5", _attrs()) as w:
        g = w.ragged("samples", "offsets", {"x": (np.float32, ())}, unit="free")
        g.append({"x": [1, 2]})
        g.append({"x": [3]})
        w.append_event(0, np.ones((4, NX, NY, NETA, 1), np.float32), 1, 0.6)


# ───────────────────────────────────────────── diagnostics
def test_diag_is_written_per_event_with_fast_data_dtypes(tmp_path):
    p = tmp_path / "diag.h5"
    with FnoH5Writer(p, _attrs()) as w:
        for i in range(3):
            w.append_event(i, np.ones((4, NX, NY, NETA, 1), np.float32), 1, 0.6)
        w.write_diag(0, n_droplets=3, E_droplets=1.5, prod_seed=7, note="first")
        w.write_diag(2, n_droplets=5, flag=np.array([True]))
        assert w.f["diag/n_droplets"][0] == 3          # on disk already, not held to close
    with h5py.File(p, "r") as f:
        d = f["diag"]
        assert d["n_droplets"].shape == (3,)
        assert list(d["n_droplets"][[0, 2]]) == [3, 5] and np.isnan(d["n_droplets"][1])
        assert np.isnan(d["E_droplets"][1]) and np.isnan(d["E_droplets"][2])
        assert d["prod_seed"].dtype == np.uint64 and d["prod_seed"][0] == 7
        assert d["note"][0].decode() == "first"
        assert d["flag"][2] == 1.0 and np.isnan(d["flag"][0])


def test_diag_refuses_an_array(tmp_path):
    with FnoH5Writer(tmp_path / "bad.h5", _attrs()) as w:
        with pytest.raises(ValueError, match="not a scalar"):
            w.write_diag(0, v=np.zeros(3))


# ───────────────────────────────────────────── repad
def _pair_file(path, n_jet, n_bg):
    with FnoH5Writer(path, _attrs()) as w:
        w.add_evolution("arr_bg", fo_suffix="_bg")
        _write_pair(w, 0, n_jet=n_jet, n_bg=n_bg)
    return path


def test_repad_grows_both_legs_of_a_pair_together(tmp_path):
    a = _pair_file(tmp_path / "a.h5", 6, 4)
    b = _pair_file(tmp_path / "b.h5", 9, 7)
    target, changed = repad_to([a, b], verbose=False)
    assert target == 9 and changed == [str(a)]
    with h5py.File(a, "r") as f:
        assert f["arr"].shape[5] == f["arr_bg"].shape[5] == f.attrs["choose_ntau"] == 9
        assert np.all(f["arr_bg"][0, ..., 4:] == 0)


def test_repad_grows_arr_bg_in_files_without_tau_axis_attributes(tmp_path):
    """A FastHydro pair file (fast_data writer) has no tau_axis attributes."""
    p = tmp_path / "legacy.h5"
    shape = (1, 4, NX, NY, NETA, 5)
    with h5py.File(p, "w") as f:
        for k, v in _attrs(5).items():
            f.attrs[k] = v
        f.attrs["nFeatures"] = 4
        for name in ("arr", "arr_bg"):
            f.create_dataset(name, shape, dtype="f4", maxshape=shape[:-1] + (None,),
                             chunks=shape)
    target, changed = repad_to(p, 8, verbose=False)
    assert target == 8 and changed == [str(p)]
    with h5py.File(p, "r") as f:
        assert f["arr"].shape[5] == f["arr_bg"].shape[5] == 8


# ───────────────────────────────────────────── liquefier_io
class _FakeLiquefier:
    """Duck-types the bound CausalLiquefier: droplets_numpy() and params()."""

    def __init__(self, rows, with_derived=True):
        self._rows = rows
        if with_derived:
            self.c_diff, self.gamma_relax = 0.9, 5.0

    def droplets_numpy(self):
        return self._rows

    def params(self):
        return {"dtau": 0.02, "tau_delay": 2.0, "time_relax": 0.1, "d_diff": 0.08,
                "width_delta": 0.1}


def test_droplets_come_out_as_an_m_by_8_float64_array():
    from jetscape.liquefier_io import DROPLET_COLUMNS, droplets

    d = droplets(_FakeLiquefier([[1, 2, 3, 4, 5, 6, 7, 8]] * 3))
    assert d.shape == (3, 8) and d.dtype == np.float64 and len(DROPLET_COLUMNS) == 8
    assert droplets(_FakeLiquefier([])).shape == (0, 8)


def test_liquefier_params_prefer_the_objects_derived_values():
    from jetscape.liquefier_io import PARAM_KEYS, liquefier_params

    p = liquefier_params(_FakeLiquefier([], with_derived=True))
    assert tuple(p) == PARAM_KEYS and p["c_diff"] == 0.9 and p["gamma_relax"] == 5.0
    q = liquefier_params(_FakeLiquefier([], with_derived=False))
    assert q["c_diff"] == pytest.approx(np.sqrt(0.8)) and q["gamma_relax"] == pytest.approx(5.0)


# ───────────────────────────────────────────── PairH5Writer with stub legs
class _Bulk:
    """Duck-types EvolutionHistory: grid scalars + to_numpy_full (e, T, vx, vy, vz)."""

    def __init__(self, evo, tau_min=0.4, dtau=0.1):
        self.evo = evo                                   # (ntau, nx, ny, neta, 4)
        self.ntau, self.nx, self.ny, self.neta = evo.shape[:4]
        self.dx = self.dy = 0.5
        self.deta = 0.4
        self.x_min = -0.5 * (self.nx - 1) * self.dx
        self.y_min = -0.5 * (self.ny - 1) * self.dy
        self.eta_min = -0.5 * (self.neta - 1) * self.deta
        self.tau_min, self.dtau = tau_min, dtau
        self.boost_invariant = False

    def to_numpy_full(self, n_features=5):
        e, v = self.evo[..., :1], self.evo[..., 1:]
        return np.concatenate([e, np.full_like(e, 0.3), v], axis=-1)


class _Music:
    """Duck-types MpiMusic for one leg (framework copy or native store)."""

    def __init__(self, name, evo, liquefier=None):
        self.name, self.evo, self.liq = name, evo, liquefier
        self.dump, self.skip, self.cleared = False, False, 0

    def GetId(self):
        return self.name

    def get_bulk_info(self):
        return _Bulk(self.evo)

    def get_dump_hydro_only(self):
        return self.dump

    def set_dump_hydro_only(self, v):
        self.dump = bool(v)

    def set_skip_surface(self, v):
        self.skip = bool(v)

    def get_native_evolution_numpy(self, tau_stride=1):
        return self.evo[::tau_stride]

    def clear_hydro_info_from_memory(self):
        self.cleared += 1

    def get_liquefier(self):
        return self.liq


class _Liq(_FakeLiquefier):
    """tau_delay 0.2 fm, so a droplet at tau 0.6 deposits inside the stub evolutions."""

    def __init__(self, rows):
        super().__init__(np.asarray(rows, dtype=float).reshape(-1, 8))
        self.clears = 0

    def params(self):
        return dict(super().params(), tau_delay=0.2)

    def ClearTask(self):
        self.clears += 1


class _Manager:
    def __init__(self, n_showers=0):
        self.n = n_showers

    def get_showers(self):
        return [_Shower() for _ in range(self.n)]

    def get_shower_initiating_partons(self):
        return []

    def GetTaskList(self):
        return []


class _Shower:
    def vertices_to_numpy(self):
        return np.array([[0, 0, 0, 0, 0.0], [1, 0, 0, 0, 1.0]])

    def to_numpy(self):
        return np.array([[0, 1, 21, 0, 1, 0, 0, 10, 0, 0, 0, 0.5]], dtype=float)


class _JS:
    def __init__(self, *tasks):
        self.tasks = tasks

    def GetTaskList(self):
        return list(self.tasks)


def _evo(ntau, value, nx=NX, ny=NY, neta=NETA):
    e = np.zeros((ntau, nx, ny, neta, 4), np.float32)
    e[..., 0] = value + np.arange(ntau, dtype=np.float32)[:, None, None, None]
    e[..., 1:] = 0.1
    return e


def _pair_writer(tmp_path, bg_evo, jet_evo, *, drops=(), deposition=True, **kw):
    from jetscape.pair_h5 import PairH5Writer

    liq = _Liq(drops)
    bg = _Music("MUSIC_1", bg_evo)
    jet = _Music("MUSIC_2", jet_evo, liquefier=liq if deposition else None)
    w = PairH5Writer(tmp_path / "pair.h5", grid_mode="native", **kw)
    w.attach(_JS(bg, jet), manager=_Manager(1), liquefier=None if deposition else liq)
    return w, bg, jet, liq


def test_pair_writer_writes_both_legs_in_the_fasthydro_layout(tmp_path):
    bg_evo = _evo(4, 1.0)
    jet_evo = _evo(6, 1.0)
    jet_evo[2:, 1, 1, 1, 0] += 0.5              # the jet shows up from frame 2 on
    drop = [[0.6, 0, 0, 0, 3.0, 1, 0, 0], [9.0, 0, 0, 0, 2.0, 1, 0, 0]]   # 2nd is late
    w, bg, jet, liq = _pair_writer(tmp_path, bg_evo, jet_evo, drops=drop)
    assert jet.dump and not bg.dump and jet.skip and bg.skip
    assert w.Exec(wall_s=1.5) == 0
    w.Finish()
    assert jet.cleared == 1 and liq.clears == 0     # MUSIC_2 clears its own droplets

    with h5py.File(tmp_path / "pair.h5", "r") as f:
        assert f["arr"].shape == f["arr_bg"].shape == (1, 4, NX, NY, NETA, 6)
        assert f["ntau_freezeout"][0] == 6 and f["ntau_freezeout_bg"][0] == 4
        # arr_bg comes from the framework copy's (e, vx, vy, vz), not temperature
        assert np.allclose(f["arr_bg"][0, 1, ..., :4], 0.1)
        assert np.all(f["arr_bg"][0, ..., 4:] == 0)
        np.testing.assert_array_equal(f["arr_bg"][0, 0, ..., :4],
                                      np.moveaxis(bg_evo[..., 0], 0, -1))
        d = f["diag"]
        assert d["frames_identical"][0] == 2
        assert d["n_droplets"][0] == 2 and d["E_droplets"][0] == 5.0
        assert d["n_droplets_late"][0] == 1 and d["E_droplets_late"][0] == 2.0
        assert d["wall_s"][0] == 1.5 and d["bg_id"][0] == 0
        assert d["n_showers"][0] == 1 and d["n_partons"][0] == 1
        assert list(f["source/offsets"][:]) == [0, 2]
        assert f["source/droplets"].shape == (2, 8)
        assert list(f["shower/parton_offsets"][:]) == [0, 1]
        a = f.attrs
        assert a["pairing"] == "bg_jet" and a["has_shower"] and not a["has_source"]
        assert a["deposition"] == "on" and a["liquefier_tau_delay"] == 0.2
        assert a["freezeout_convention_id"] == "frames_written"


def test_the_null_test_clears_droplets_and_flags_identical_legs(tmp_path):
    evo = _evo(5, 2.0)
    w, bg, jet, liq = _pair_writer(tmp_path, evo, evo.copy(),
                                   drops=[[0.6, 0, 0, 0, 1.0, 1, 0, 0]], deposition=False)
    assert w.deposition is False
    w.Exec()
    w.Finish()
    assert liq.clears == 1                          # nothing else would clear them
    with h5py.File(tmp_path / "pair.h5", "r") as f:
        np.testing.assert_array_equal(f["arr"][:], f["arr_bg"][:])
        assert f["diag/frames_identical"][0] == 5
        assert f.attrs["deposition"] == "off"


def test_a_jet_leg_identical_despite_droplets_is_flagged(tmp_path):
    evo = _evo(5, 2.0)
    w, *_ = _pair_writer(tmp_path, evo, evo.copy(), drops=[[0.6, 0, 0, 0, 4.0, 1, 0, 0]])
    with pytest.warns(RuntimeWarning, match="MUSIC ignored the liquefier"):
        w.Exec()
    w.Finish()


def test_different_initial_conditions_are_flagged(tmp_path):
    w, *_ = _pair_writer(tmp_path, _evo(3, 1.0), _evo(3, 7.0))
    with pytest.warns(RuntimeWarning, match="first frame"):
        w.Exec()
    w.Finish()


def test_a_reused_background_keeps_its_bg_id(tmp_path):
    bg_evo = _evo(3, 1.0)
    w, bg, jet, liq = _pair_writer(tmp_path, bg_evo, bg_evo.copy())
    for k in range(3):
        jet.evo = _evo(3, 1.0)
        jet.evo[1:, 0, 0, 0, 0] += k + 1           # a different jet every event
        if k == 2:
            bg.evo = _evo(3, 5.0)                  # a new background run
            jet.evo = _evo(3, 5.0)
            jet.evo[1:, 0, 0, 0, 0] += 1
        w.Exec()
    w.Finish()
    with h5py.File(tmp_path / "pair.h5", "r") as f:
        assert list(f["diag/bg_id"][:]) == [0, 0, 2]
        np.testing.assert_array_equal(f["arr_bg"][0], f["arr_bg"][1])


def _reused_run(tmp_path, name, **kw):
    """Five events: background A for events 0-2, B for 3-4, a different jet every event."""
    bg_evo = _evo(3, 1.0)
    w, bg, jet, liq = _pair_writer(tmp_path / name, bg_evo, bg_evo.copy(), **kw)
    for k in range(5):
        if k == 3:
            bg.evo = _evo(4, 5.0)                  # a new, longer background run
        jet.evo = bg.evo.copy()
        jet.evo[1:, 0, 0, 0, 0] += k + 1
        w.Exec()
    w.Finish()
    return tmp_path / name / "pair.h5", w


def test_shared_background_reads_like_the_full_layout(tmp_path):
    full, wf = _reused_run(tmp_path, "full", bg_layout="full")
    shared, ws = _reused_run(tmp_path, "shared", bg_layout="shared")
    assert (wf.bg_layout, ws.bg_layout) == ("full", "shared")
    with h5py.File(full, "r") as a, h5py.File(shared, "r") as b:
        assert "bg_layout" not in a.attrs and not a["arr_bg"].is_virtual
        assert b.attrs["bg_layout"] == "shared" and b["arr_bg"].is_virtual
        assert b["arr_bg"].shape == b["arr"].shape == a["arr_bg"].shape
        np.testing.assert_array_equal(b["arr_bg"][:], a["arr_bg"][:])
        np.testing.assert_array_equal(b["arr"][:], a["arr"][:])
        for k in ("ntau_freezeout_bg", "tau_freezeout_bg", "diag/bg_id",
                  "diag/frames_identical"):
            np.testing.assert_array_equal(b[k][:], a[k][:])
        assert list(b["arr_bg_rows"][:]) == [0, 0, 0, 3, 3]
        # only the two backgrounds' rows hold data: 3 + 4 frames, one chunk each
        assert b["arr_bg_store"].id.get_num_chunks() == 7
        assert a["arr_bg"].id.get_num_chunks() == 3 * 3 + 2 * 4


def test_store_evolution_off_keeps_the_bookkeeping_and_writes_no_file(tmp_path):
    """--particlize-only: the same indices, diag and background keys, no pair file."""
    def run(name, **kw):
        bg_evo = _evo(3, 1.0)
        w, bg, jet, liq = _pair_writer(tmp_path / name, bg_evo, bg_evo.copy(),
                                       drops=[[0.6, 0, 0, 0, 3.0, 1, 0, 0]], reuse=3, **kw)
        out = []
        for k in range(5):
            if k == 3:
                bg.evo = _evo(4, 5.0)              # a new, longer background run
            jet.evo = bg.evo.copy()
            jet.evo[1:, 0, 0, 0, 0] += k + 1
            out.append((w.Exec(pthat=10.0 + k), w.last_event_diag, w.last_bg_key))
        w.Finish()
        w.Finish()                                 # idempotent
        return out, w, jet

    full, wf, _ = run("full")
    slim, ws, jet = run("slim", store_evolution=False)
    assert (tmp_path / "full" / "pair.h5").exists()
    assert not (tmp_path / "slim" / "pair.h5").exists()
    assert jet.cleared == 5                        # the native store is still released
    assert ws.GetNumberOfEventsWritten() == wf.GetNumberOfEventsWritten() == 5
    assert [i for i, _, _ in slim] == [i for i, _, _ in full] == list(range(5))
    assert [k for _, _, k in slim] == [k for _, _, k in full]
    for (_, ds, _), (_, df, _) in zip(slim, full):
        assert {k: v for k, v in df.items() if k not in ("n_showers", "n_partons")} == ds
    assert [d["bg_id"] for _, d, _ in slim] == [0, 0, 0, 3, 3]


def test_store_evolution_off_still_clips_at_a_pinned_ntau(tmp_path):
    w, *_ = _pair_writer(tmp_path, _evo(4, 1.0), _evo(5, 1.0), store_evolution=False,
                         choose_ntau=3)
    with pytest.warns(RuntimeWarning, match="clipping"):
        w.Exec()
    w.Finish()
    assert w.n_clipped == 2
    assert (w.last_event_diag["ntau_jet"], w.last_event_diag["ntau_bg"]) == (3, 3)


def test_read_jet_off_takes_the_jet_leg_from_musics_metadata(tmp_path):
    """--particlize-only: the jet leg's evolution is never read, its native store still
    released; the background bookkeeping is unchanged."""
    from jetscape.pair_h5 import PairH5Writer

    def run(name, **kw):
        bg_evo = _evo(3, 1.0)
        w, bg, jet, liq = _pair_writer(tmp_path / name, bg_evo, bg_evo.copy(),
                                       drops=[[0.6, 0, 0, 0, 3.0, 1, 0, 0],
                                              [9.0, 0, 0, 0, 2.0, 1, 0, 0]],
                                       store_evolution=False, tau_stride=2, **kw)
        native = jet.get_native_evolution_numpy
        jet.reads = 0

        def counted(tau_stride=1):
            jet.reads += 1
            return native(tau_stride)
        jet.get_native_evolution_numpy = counted
        out = []
        for k in range(4):
            if k == 2:
                bg.evo = _evo(4, 5.0)              # a new background run
            jet.evo = np.concatenate([bg.evo, bg.evo[-1:]])   # one step longer
            jet.evo[1:, 0, 0, 0, 0] += k + 1
            out.append((w.Exec(), w.last_event_diag, w.last_bg_key))
        w.Finish()
        run.n_hit_edge = w.n_hit_edge
        return out, jet

    with pytest.warns(RuntimeWarning, match="on the transverse edge"):  # e >= 1 up to the edge
        read, jet_read = run("read", edge_e_threshold=0.3)
    assert run.n_hit_edge == {"bg": 4, "jet": 4}
    with pytest.warns(RuntimeWarning, match="the bg leg has e"):
        skip, jet = run("skip", read_jet=False, edge_e_threshold=0.3)
    assert run.n_hit_edge == {"bg": 4, "jet": None}           # the jet leg: not measured
    assert jet_read.reads == 4 and jet.reads == 0
    assert jet.cleared == 4                        # the native store is still released
    assert [k for _, _, k in skip] == [k for _, _, k in read]
    for (i, ds, _), (j, dr, _) in zip(skip, read):
        assert i == j
        assert "frames_identical" not in ds and "ntau_jet" not in ds
        assert ds["ntau_jet_music"] == dr["ntau_jet"]          # native grid, stride 2
        same = {k: v for k, v in dr.items()
                if k not in ("frames_identical", "ntau_jet", "jet_hit_edge")
                and not k.startswith("jet_edge")}
        assert {k: v for k, v in ds.items() if k != "ntau_jet_music"} == same
    assert [d["bg_id"] for _, d, _ in skip] == [0, 0, 2, 2]
    assert [d["n_droplets_late"] for _, d, _ in skip] == [1, 1, 1, 1]
    with pytest.raises(ValueError, match="store_evolution=False"):
        PairH5Writer(tmp_path / "x.h5", read_jet=False)


class _FramedBulk(_Bulk):
    """A build with EvolutionHistory.frame_numpy: one (nx, ny, neta, 4) frame at a time."""

    def __init__(self, evo, log):
        super().__init__(evo)
        self.log = log

    def get_data_size(self):
        return self.evo[..., 0].size

    def frame_numpy(self, id_tau):
        self.log.append(id_tau)
        return self.evo[id_tau].copy()


@pytest.mark.parametrize("grid_mode", ["native", "grid"])
def test_background_read_frame_by_frame_writes_the_same_file(tmp_path, grid_mode):
    """F3: the framework copy fetched per tau frame gives a byte-identical pair file."""
    from jetscape.bulk_sources import Grid

    out = Grid.from_bounds((-0.6, 0.6, 3), (-0.4, 0.4, 2), (-0.3, 0.3, 2),
                           tau_min=0.4, dtau=0.15)

    def run(name, framed):
        log = []
        bg_evo = _evo(5, 1.0)
        w, bg, jet, liq = _pair_writer(tmp_path / name, bg_evo, bg_evo.copy(), tau_stride=2)
        if grid_mode == "grid":                    # resampled onto a smaller grid
            w._grid_mode, w._out_grid = "grid", out
        if framed:
            bg.get_bulk_info = lambda: _FramedBulk(bg.evo, log)
        for k in range(3):
            if k == 2:
                bg.evo = _evo(6, 4.0)
            jet.evo = bg.evo.copy()
            jet.evo[1:, 0, 0, 0, 0] += k + 1
            assert w.Exec() == k
        w.Finish()
        return tmp_path / name / "pair.h5", log

    plain, _ = run("plain", False)
    framed, log = run("framed", True)
    assert log and set(log) <= {0, 2, 4}           # strided frames only, never all at once
    with h5py.File(plain, "r") as a, h5py.File(framed, "r") as b:
        names = []
        a.visititems(lambda n, o: names.append(n) if isinstance(o, h5py.Dataset) else None)
        for n in names:
            np.testing.assert_array_equal(a[n][()], b[n][()], err_msg=n)


def test_framework_frames_index_like_the_whole_array():
    from jetscape.bulk_sources import FrameworkFrames

    evo = _evo(5, 1.0)
    f = FrameworkFrames(_FramedBulk(evo, []), tau_stride=2)
    want = evo[::2]
    assert f.shape == want.shape and len(f) == 3
    np.testing.assert_array_equal(f[1], want[1])
    np.testing.assert_array_equal(f[-1, 1:3, :, 0, :], want[-1, 1:3, :, 0, :])
    with pytest.raises(IndexError):
        f[3]


class _FramedMusic:
    """Adds MpiMusic's per-frame native access (get_native_ntau, get_native_frame_numpy)
    to a _Music leg, logging every frame fetched."""

    def __init__(self, music, log):
        self.music, self.log = music, log

    def install(self):
        m = self.music
        m.get_native_ntau = lambda: m.evo.shape[0]
        m.get_native_frame_numpy = self.frame
        return m

    def frame(self, it):
        self.log.append(it)
        return self.music.evo[it].copy()


def test_native_frames_index_like_the_whole_array():
    from jetscape.bulk_sources import NativeFrames, native_frames

    evo = _evo(5, 1.0)
    log = []
    m = _FramedMusic(_Music("jet", evo), log).install()
    f = NativeFrames(m, tau_stride=2)
    want = evo[::2]
    assert f.shape == want.shape and len(f) == 3
    np.testing.assert_array_equal(f[1], want[1])
    np.testing.assert_array_equal(f[-1, 1:3, :, 0, :], want[-1, 1:3, :, 0, :])
    with pytest.raises(IndexError):
        f[3]
    assert log == [2, 4]                            # strided frames, each fetched once
    assert native_frames(_Music("old", evo)) is None   # a build without the binding
    m.dump = False
    with pytest.raises(RuntimeError, match="dump_hydro_only"):
        native_frames(m)


@pytest.mark.parametrize("grid_mode", ["native", "grid"])
def test_jet_leg_read_frame_by_frame_writes_the_same_file(tmp_path, grid_mode):
    """The jet leg fetched per tau frame (NativeFrames) gives a byte-identical pair file."""
    from jetscape.bulk_sources import Grid

    out = Grid.from_bounds((-0.6, 0.6, 3), (-0.4, 0.4, 2), (-0.3, 0.3, 2),
                           tau_min=0.4, dtau=0.15)

    def run(name, framed):
        log = []
        bg_evo = _evo(5, 1.0)
        w, bg, jet, liq = _pair_writer(tmp_path / name, bg_evo, bg_evo.copy(), tau_stride=2)
        if grid_mode == "grid":                    # resampled onto a smaller grid
            w._grid_mode, w._out_grid = "grid", out
        if framed:
            _FramedMusic(jet, log).install()
        for k in range(3):
            if k == 2:
                bg.evo = _evo(6, 4.0)
            jet.evo = bg.evo.copy()
            jet.evo[1:, 0, 0, 0, 0] += k + 1
            assert w.Exec() == k
        w.Finish()
        return tmp_path / name / "pair.h5", log

    plain, _ = run("plain", False)
    framed, log = run("framed", True)
    assert log and set(log) <= {0, 2, 4}           # strided frames only, never all at once
    with h5py.File(plain, "r") as a, h5py.File(framed, "r") as b:
        names = []
        a.visititems(lambda n, o: names.append(n) if isinstance(o, h5py.Dataset) else None)
        for n in names:
            np.testing.assert_array_equal(a[n][()], b[n][()], err_msg=n)


def test_bg_layout_auto_follows_reuse(tmp_path):
    from jetscape.pair_h5 import PairH5Writer

    one = PairH5Writer(tmp_path / "a.h5", reuse=1)
    three = PairH5Writer(tmp_path / "b.h5", reuse=3)
    assert (one.bg_layout, three.bg_layout) == ("full", "shared")
    assert PairH5Writer(tmp_path / "c.h5", reuse=3, bg_layout="full").bg_layout == "full"
    with pytest.raises(ValueError):
        PairH5Writer(tmp_path / "d.h5", bg_layout="copies")


def test_repad_grows_the_store_and_rebuilds_the_view(tmp_path):
    shared, _ = _reused_run(tmp_path, "shared", bg_layout="shared")
    with h5py.File(shared, "r") as f:
        before, n0 = f["arr_bg"][:], f["arr"].shape[5]
    target, changed = repad_to(shared, n0 + 3, verbose=False)
    assert target == n0 + 3 and changed == [str(shared)]
    with h5py.File(shared, "r") as f:
        assert f["arr_bg"].is_virtual
        assert f["arr"].shape == f["arr_bg"].shape == f["arr_bg_store"].shape
        assert f["arr_bg"].shape[5] == n0 + 3
        np.testing.assert_array_equal(f["arr_bg"][..., :n0], before)
        assert np.all(f["arr_bg"][..., n0:] == 0)


def test_legs_on_different_grids_skip_the_event(tmp_path):
    w, bg, jet, liq = _pair_writer(tmp_path, _evo(3, 1.0), _evo(3, 1.0, nx=NX + 2))
    with pytest.warns(RuntimeWarning, match="different grids"):
        assert w.Exec() is None
    assert jet.cleared == 1                         # released even when skipped
    w.Finish()


def test_attach_refuses_a_background_that_keeps_no_framework_copy(tmp_path):
    from jetscape.pair_h5 import PairH5Writer

    bg, jet = _Music("MUSIC_1", _evo(2, 1.0)), _Music("MUSIC_2", _evo(2, 1.0))
    bg.dump = True
    with pytest.raises(RuntimeError, match="Matter/LBT would see no medium"):
        PairH5Writer(tmp_path / "x.h5").attach(_JS(bg, jet), manager=_Manager())
    with pytest.raises(RuntimeError, match="no task with id"):
        PairH5Writer(tmp_path / "y.h5", jet_id="MUSIC_X").attach(
            _JS(_Music("MUSIC_1", _evo(2, 1.0)), jet), manager=_Manager())


def test_grid_mode_resamples_both_legs_onto_one_output_grid(tmp_path):
    pytest.importorskip("scipy")
    from jetscape.bulk_sources import Grid

    out = Grid.from_bounds((-0.5, 0.5, 3), (-0.5, 0.5, 3), (0.0, 0.0, 1),
                           tau_min=0.5, dtau=0.1)
    from jetscape.pair_h5 import PairH5Writer
    bg = _Music("MUSIC_1", _evo(4, 1.0))
    jet = _Music("MUSIC_2", _evo(5, 1.0), liquefier=_Liq([]))
    w = PairH5Writer(tmp_path / "grid.h5", grid_mode="grid", out_grid=out)
    w.attach(_JS(bg, jet), manager=_Manager())
    w.Exec()
    w.Finish()
    with h5py.File(tmp_path / "grid.h5", "r") as f:
        assert f["arr"].shape[2:5] == (3, 3, 1) == f["arr_bg"].shape[2:5]
        # tau 0.4 + k*0.1 resampled from 0.5: jet 4 frames, background 3
        assert f["ntau_freezeout"][0] == 4 and f["ntau_freezeout_bg"][0] == 3
        assert f.attrs["tau_min"] == pytest.approx(0.5)
        assert f["diag/frames_identical"][0] == 3


def test_bg_id_changes_with_the_background_even_when_frame_0_is_all_zero(tmp_path):
    """On MUSIC's native grid the first frame is at tau0, before any string deposits."""
    def evo_with_zero_first_frame(value):
        e = _evo(3, value)
        e[0] = 0.0
        return e

    w, bg, jet, liq = _pair_writer(tmp_path, evo_with_zero_first_frame(1.0),
                                   evo_with_zero_first_frame(1.0))
    for value in (1.0, 2.0, 3.0):
        bg.evo = evo_with_zero_first_frame(value)
        jet.evo = evo_with_zero_first_frame(value)
        w.Exec()
    w.Finish()
    with h5py.File(tmp_path / "pair.h5", "r") as f:
        assert list(f["diag/bg_id"][:]) == [0, 1, 2]


def test_a_leg_stopped_at_the_grid_boundary_is_flagged(tmp_path):
    w, bg, jet, liq = _pair_writer(tmp_path, _evo(3, 1.0), _evo(3, 1.0))
    bg.get_hit_grid_boundary = lambda: False
    jet.get_hit_grid_boundary = lambda: True
    with pytest.warns(RuntimeWarning, match="jet leg's freeze-out surface reached"):
        w.Exec()
    w.Finish()
    with h5py.File(tmp_path / "pair.h5", "r") as f:
        assert f["diag/jet_hit_boundary"][0] == 1 and f["diag/bg_hit_boundary"][0] == 0



def _edge_evo(ntau=3, hot=True):
    """e = 1 inside, 0.1 on every face, and (hot) in frame 1 a hot spot 0.5 on the last x
    plane and 0.7 on the first eta plane (away from the x/y faces)."""
    e = _evo(ntau, 0.0)
    e[..., 0] = 0.1
    e[:, 1:-1, 1:-1, 1:-1, 0] = 1.0
    if hot:
        e[1, -1, 2, 1, 0] = 0.5
        e[1, 2, 2, 0, 0] = 0.7
    return e


def test_edge_e_max_is_recorded_per_leg_on_the_output_grid_faces(tmp_path):
    w, bg, jet, liq = _pair_writer(tmp_path, _edge_evo(hot=False), _edge_evo())
    w.Exec()
    w.Finish()
    assert w.n_hit_edge == {"bg": 0, "jet": 0}
    with h5py.File(tmp_path / "pair.h5", "r") as f:
        d = f["diag"]
        assert d["jet_edge_e_max"][0] == pytest.approx(0.5)      # x/y faces only
        assert d["jet_edge_e_max_eta"][0] == pytest.approx(0.7)
        assert d["bg_edge_e_max"][0] == d["bg_edge_e_max_eta"][0] == pytest.approx(0.1)
        assert "jet_hit_edge" not in d and "bg_hit_edge" not in d  # no threshold, no flag
        assert "edge_e_threshold" not in f.attrs


def test_a_leg_above_the_threshold_on_the_edge_is_flagged(tmp_path):
    w, bg, jet, liq = _pair_writer(tmp_path, _edge_evo(hot=False), _edge_evo(),
                                   edge_e_threshold=0.3)
    with pytest.warns(RuntimeWarning, match="jet leg has e = 0.500 GeV/fm.3 > 0.3"):
        w.Exec()
    jet.evo = _edge_evo()
    jet.evo[1, -1, 2, 1, 0] = 0.2                    # below: no flag, no warning
    w.Exec()
    w.Finish()
    assert w.n_hit_edge == {"bg": 0, "jet": 1}
    with h5py.File(tmp_path / "pair.h5", "r") as f:
        assert list(f["diag/jet_hit_edge"][:]) == [1, 0]
        assert list(f["diag/bg_hit_edge"][:]) == [0, 0]
        assert f.attrs["edge_e_threshold"] == 0.3


def test_edge_e_max_follows_the_output_grid_not_musics(tmp_path):
    """grid mode: the faces are those of the resampled box, here the inner 3 x 2 cells."""
    pytest.importorskip("scipy")
    from jetscape.bulk_sources import Grid
    from jetscape.pair_h5 import PairH5Writer

    evo = _edge_evo()                     # NX x NY = 5 x 4, dx = dy = 0.5, x_min -1
    out = Grid.from_bounds((-0.5, 0.5, 3), (-0.25, 0.25, 2), (-0.4, 0.4, 3),
                           tau_min=0.4, dtau=0.1)
    bg = _Music("MUSIC_1", evo)
    jet = _Music("MUSIC_2", evo.copy(), liquefier=_Liq([]))
    w = PairH5Writer(tmp_path / "grid.h5", grid_mode="grid", out_grid=out)
    w.attach(_JS(bg, jet), manager=_Manager())
    w.Exec()
    w.Finish()
    with h5py.File(tmp_path / "grid.h5", "r") as f:
        # x = +-0.5 and y = +-0.25 lie inside MUSIC's grid, where e = 1 (the 0.1 faces
        # and the hot spot on MUSIC's last x plane are outside the box)
        assert f["diag/bg_edge_e_max"][0] == pytest.approx(1.0)
        assert f["diag/jet_edge_e_max"][0] == pytest.approx(1.0)


def test_a_single_slice_axis_has_no_faces(tmp_path):
    from jetscape.pair_h5 import _EdgeMax

    m = _EdgeMax()
    m.add(np.arange(6.0).reshape(3, 2, 1))       # neta = 1: no eta faces
    assert m.diag("bg", 2.0) == {"bg_edge_e_max": 5.0, "bg_hit_edge": 1}
    m = _EdgeMax()
    m.add(np.ones((1, 1, 4)))                    # a single x, y point: no x/y faces
    assert m.diag("jet", 0.5) == {"jet_edge_e_max_eta": 1.0}

# ───────────────────────────────────────────── compression and mantissa rounding
@pytest.mark.parametrize("spec", ["lzf", "gzip", None, "blosc-zstd", "blosc-lz4"])
def test_every_evolution_gets_the_same_filter_and_reads_back_exactly(tmp_path, spec):
    from jetscape.h5_compression import HAVE_HDF5PLUGIN, h5_filter_kwargs
    if str(spec).startswith("blosc") and not HAVE_HDF5PLUGIN:
        pytest.skip("hdf5plugin not installed")
    p = tmp_path / "c.h5"
    rng = np.random.default_rng(7)
    frames = rng.standard_normal((2, 3, 4, NX, NY, NETA)).astype(np.float32)
    with FnoH5Writer(p, _attrs(), compression=spec) as w:
        w.add_evolution("arr_bg", fo_suffix="_bg")
        w.ensure_capacity(nevents=1, choose_ntau=3)
        for t in range(3):
            w.write_frame(0, t, frames[1, t], dataset="arr_bg")
            w.write_frame(0, t, frames[0, t])
        w.set_event_meta(0, 3, 0.8, dataset="arr_bg")
        w.set_event_meta(0, 3, 0.8)
    with h5py.File(p, "r") as f:
        label = h5_filter_kwargs(spec)[1]
        for name, k in (("arr", 0), ("arr_bg", 1)):
            ds = f[name]
            assert ds.attrs["compression"] == label
            assert ds._filters == f["arr"]._filters
            assert np.array_equal(np.moveaxis(ds[0], -1, 0), frames[k])


def test_keep_bits_rounds_both_legs_identically(tmp_path):
    from jetscape.h5_compression import HAVE_HDF5PLUGIN
    if not HAVE_HDF5PLUGIN:
        pytest.skip("hdf5plugin not installed")
    p = tmp_path / "k.h5"
    base = np.random.default_rng(8).standard_normal((4, NX, NY, NETA)).astype(np.float32)
    jet = base.copy()
    jet[0, 0, 0, 0] += 1.0                       # the legs differ in one cell only
    with FnoH5Writer(p, _attrs(), keep_bits=12) as w:
        w.add_evolution("arr_bg", fo_suffix="_bg")
        w.ensure_capacity(nevents=1, choose_ntau=2)
        w.write_frame(0, 0, base, dataset="arr_bg")
        w.write_frame(0, 0, jet)
        w.set_event_meta(0, 1, 0.6, dataset="arr_bg")
        w.set_event_meta(0, 1, 0.6)
    with h5py.File(p, "r") as f:
        arr, bg = f["arr"][0, ..., 0], f["arr_bg"][0, ..., 0]
        for name in ("arr", "arr_bg"):
            assert int(f[name].attrs["keep_mantissa_bits"]) == 12
            assert f[name].attrs["compression"] == "blosc-zstd:3+shuffle"
        assert np.all(f["arr"][0, ..., 1] == 0), "unwritten frames stay exactly zero"
    live = base != 0
    assert (np.abs(bg[live] - base[live]) / np.abs(base[live])).max() <= 2.0 ** -13
    diff = arr - bg
    diff[0, 0, 0, 0] = 0.0
    assert np.all(diff == 0), "arr - arr_bg is exactly zero wherever the legs agree"
