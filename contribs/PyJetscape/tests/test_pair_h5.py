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
        assert bg.chunks == arr.chunks and bg.compression == arr.compression
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
