"""
tests/test_h5_bulk.py

Tests for the pure-Python HDF5 bulk writer (jetscape.fno_h5_writer,
jetscape.bulk_sources, jetscape.fast_h5_bulk).

Nothing here runs MUSIC: the interpolation is checked against a literal transcription of
the C++ EvolutionHistory::get(), and the schema is checked against FNO4d's loaders when
FNO4d is importable.  Run a real A/B comparison with example/validate_h5_vs_root.py.

    pytest tests/test_h5_bulk.py -q

The transcription below evaluates in float64, so agreement with resample() should be at
round-off (~1e-12).  That is deliberate: it pins the *semantics* (range masking, index
truncation, edge clamping, the tau blend), which is what could actually be wrong.  Against
the real C++ the gap is ~1e-7 instead, because Jetscape::real is float
(src/framework/RealType.h:26) so the C++ does its own blend in float32.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from jetscape.bulk_sources import Grid, attrs_from_grids, resample, resolve_out_grid
from jetscape.fno_h5_writer import SCALAR_KEYS, FnoH5Writer

h5py = pytest.importorskip("h5py")
pytest.importorskip("scipy")


# ───────────────────────────────────────────── literal C++ transcription
def _clamp(i, n):
    return min(n - 1, max(0, i))


def _at_step(data, g, id_tau, x, y, eta):
    """EvolutionHistory::GetAtTimeStep + TrilinearInt (LinearInterpolation.h:121)."""
    id_x = int((x - g.x_min) / g.dx)
    id_y = int((y - g.y_min) / g.dy)
    id_eta = 0 if g.boost_invariant else int((eta - g.eta_min) / g.deta)

    def cell(a, b, c):
        # GetFluidCell forces id_eta=0 for neta<=1; CellIndex clamps every index.
        ie = 0 if g.neta <= 1 else _clamp(id_eta + c, g.neta)
        return data[_clamp(id_tau, g.ntau), _clamp(id_x + a, g.nx),
                    _clamp(id_y + b, g.ny), ie].astype(np.float64)

    x0 = g.x_min + id_x * g.dx
    y0 = g.y_min + id_y * g.dy
    eta0 = g.eta_min + id_eta * g.deta
    t = (x - x0) / g.dx
    u = (y - y0) / g.dy
    v = 0.0 if g.deta == 0 else (eta - eta0) / g.deta

    return ((1 - t) * (1 - u) * (1 - v) * cell(0, 0, 0)
            + (1 - t) * (1 - u) * v * cell(0, 0, 1)
            + (1 - t) * u * (1 - v) * cell(0, 1, 0)
            + (1 - t) * u * v * cell(0, 1, 1)
            + t * (1 - u) * (1 - v) * cell(1, 0, 0)
            + t * (1 - u) * v * cell(1, 0, 1)
            + t * u * (1 - v) * cell(1, 1, 0)
            + t * u * v * cell(1, 1, 1))


def cpp_get(data, g, tau, x, y, eta):
    """EvolutionHistory::get (FluidEvolutionHistory.cc:359), including CheckInRange."""
    n_feat = data.shape[-1]
    x_max = g.x_min + (g.nx - 1) * g.dx
    y_max = g.y_min + (g.ny - 1) * g.dy
    eta_max = g.eta_min + (g.neta - 1) * g.deta
    if not (g.tau_min <= tau <= g.tau_max):
        return np.zeros(n_feat)
    if not (g.x_min <= x <= x_max) or not (g.y_min <= y <= y_max):
        return np.zeros(n_feat)
    if not g.boost_invariant and not (g.eta_min <= eta <= eta_max):
        return np.zeros(n_feat)

    id_tau = int((tau - g.tau_min) / g.dtau)
    tau0 = g.tau_min + id_tau * g.dtau
    tau1 = g.tau_min + (id_tau + 1) * g.dtau
    b0 = _at_step(data, g, id_tau, x, y, eta)
    b1 = _at_step(data, g, id_tau + 1, x, y, eta)
    return ((tau - tau0) * b1 + (tau1 - tau) * b0) / (tau1 - tau0)


def brute_force(data, src, out):
    res = np.zeros((out.ntau, out.nx, out.ny, out.neta, data.shape[-1]))
    for j in range(out.ntau):
        tau = out.tau_min + j * out.dtau
        for ix in range(out.nx):
            for iy in range(out.ny):
                for ie in range(out.neta):
                    res[j, ix, iy, ie] = cpp_get(
                        data, src, tau,
                        out.x_min + ix * out.dx,
                        out.y_min + iy * out.dy,
                        out.eta_min + ie * out.deta)
    return res


# ───────────────────────────────────────────────────────── fixtures
def make_source(nx=7, ny=6, neta=5, ntau=4, seed=0):
    g = Grid(nx=nx, ny=ny, neta=neta, ntau=ntau,
             x_min=-3.0, dx=1.0, y_min=-2.5, dy=1.0,
             eta_min=-2.0, deta=1.0, tau_min=0.6, dtau=0.2)
    rng = np.random.default_rng(seed)
    data = rng.random((ntau, nx, ny, neta, 4)).astype(np.float32)
    return data, g


# ──────────────────────────────────────────────── resample semantics
def test_resample_matches_cpp_get_on_a_shifted_grid():
    data, src = make_source()
    out = Grid(nx=9, ny=8, neta=7, ntau=5,
               x_min=-3.3, dx=0.8, y_min=-2.7, dy=0.7,
               eta_min=-2.1, deta=0.6, tau_min=0.55, dtau=0.15)
    got = resample(data, src, out)
    assert got.shape == (out.ntau, out.nx, out.ny, out.neta, 4)
    np.testing.assert_allclose(got, brute_force(data, src, out), rtol=0, atol=2e-6)


def test_resample_reproduces_the_source_on_an_identical_grid():
    data, src = make_source()
    got = resample(data, src, src)
    np.testing.assert_allclose(got, data, rtol=0, atol=2e-6)


def test_resample_zeroes_points_outside_the_source_box():
    """CheckInRange is a hard cut, not a clamp -- one axis out of range zeroes the cell."""
    data, src = make_source()
    out = Grid(nx=5, ny=5, neta=5, ntau=3,
               x_min=-8.0, dx=4.0, y_min=-8.0, dy=4.0,
               eta_min=-4.0, deta=2.0, tau_min=0.6, dtau=0.2)
    got = resample(data, src, out)
    # x = -8 and +8 are both outside [-3, 3]
    assert np.all(got[:, 0] == 0) and np.all(got[:, -1] == 0)
    # ... while the centre row is not
    assert np.any(got[:, 2, 2] != 0)
    np.testing.assert_allclose(got, brute_force(data, src, out), rtol=0, atol=2e-6)


def test_resample_zeroes_tau_outside_the_source_range():
    data, src = make_source()
    out = Grid(nx=src.nx, ny=src.ny, neta=src.neta, ntau=6,
               x_min=src.x_min, dx=src.dx, y_min=src.y_min, dy=src.dy,
               eta_min=src.eta_min, deta=src.deta,
               tau_min=src.tau_min, dtau=src.dtau)
    got = resample(data, src, out)
    assert out.tau_min + 4 * out.dtau > src.tau_max        # frames 4,5 are past the end
    assert np.all(got[4:] == 0)
    assert np.any(got[3] != 0)


def test_flat_eta_source_warns_when_the_output_grid_has_eta_structure():
    data, src = make_source(neta=1)
    out = resolve_out_grid(src, {"deta": 0.5, "eta_min": -1.0})
    assert out.neta > 1
    with pytest.warns(RuntimeWarning, match="boost_invariant"):
        got = resample(data, src, out)
    # every output eta gets the single source slice
    for ie in range(1, out.neta):
        np.testing.assert_allclose(got[..., ie, :], got[..., 0, :], rtol=0, atol=0)


def test_resolve_out_grid_returns_the_source_when_nothing_is_specified():
    """An all-zero spec must not silently re-derive an odd origin-centred grid."""
    _, src = make_source()
    assert resolve_out_grid(src, None) is src
    assert resolve_out_grid(src, {"dx": 0, "dtau": None}) is src


def test_resolve_out_grid_uses_the_cpp_derivation_once_a_key_is_given():
    _, src = make_source()
    out = resolve_out_grid(src, {"dx": 0.5, "x_min": -2.0})
    assert (out.dx, out.x_min, out.nx) == (0.5, -2.0, 2 * int(2.0 / 0.5) + 1)
    assert (out.dy, out.y_min) == (src.dy, src.y_min)      # untouched keys follow the source


# ───────────────────────────────────────────────────── writer schema
def write_two_ragged_events(path, ntaus=(4, 7), nx=3, ny=3, neta=2):
    src = Grid(nx=nx, ny=ny, neta=neta, ntau=max(ntaus),
               x_min=-1.0, dx=1.0, y_min=-1.0, dy=1.0,
               eta_min=-0.5, deta=0.5, tau_min=0.6, dtau=0.1)
    attrs = attrs_from_grids(src, src)
    attrs["choose_ntau"] = ntaus[0]
    rng = np.random.default_rng(7)
    payload = []
    with FnoH5Writer(path, attrs, nevents=0, force=True) as w:
        for i, nt in enumerate(ntaus):
            ev = rng.random((nt, nx, ny, neta, 4)).astype(np.float32) + 1.0
            payload.append(ev)
            w.ensure_capacity(nevents=i + 1, choose_ntau=nt)
            for t in range(nt):
                w.write_frame(i, t, np.ascontiguousarray(ev[t].transpose(3, 0, 1, 2)))
            w.set_event_meta(i, nt, src.tau_min + (nt - 1) * src.dtau)
    return src, payload


def test_writer_matches_the_fno4d_contract(tmp_path):
    p = tmp_path / "evo.h5"
    src, payload = write_two_ragged_events(p)
    with h5py.File(p, "r") as f:
        arr = f["arr"]
        assert arr.shape == (2, 4, src.nx, src.ny, src.neta, 7)
        assert arr.dtype == np.float32
        assert arr.chunks[0] == 1, "one event per chunk"
        assert arr.chunks[-1] == 1, "one tau frame per chunk -- see fno_h5_writer.write_frame"
        assert arr.compression == "lzf"
        assert f["ntau_freezeout"].dtype == np.int32
        assert f["tau_freezeout"].dtype == np.float32
        assert list(f["ntau_freezeout"][:]) == [4, 7]

        a = f.attrs
        assert int(a["nevents"]) == arr.shape[0], "read_3d_data_hdf5 vs MultiH5Array"
        assert int(a["choose_ntau"]) == arr.shape[5]
        assert bool(a["complete"])
        for k in SCALAR_KEYS:
            assert k in a, k
        assert np.dtype(a["nx"]) == np.int64 and np.dtype(a["dx"]) == np.float64

        # the short event's tail is exactly zero (unallocated chunks -> fill value)
        assert np.all(f["arr"][0, :, :, :, :, 4:] == 0.0)
        np.testing.assert_array_equal(
            f["arr"][0, ..., :4], np.moveaxis(payload[0], (0, 4), (4, 0)))


def test_padding_costs_no_disk_space(tmp_path):
    """Chunks past an event's end are never allocated, so the pad is free."""
    p = tmp_path / "evo.h5"
    write_two_ragged_events(p, ntaus=(2, 40), nx=8, ny=8, neta=4)
    with h5py.File(p, "r") as f:
        stored = f["arr"].id.get_storage_size()
        dense = int(np.prod(f["arr"].shape)) * 4
    assert stored < dense * 0.6, f"{stored} B stored vs {dense} B dense"


def test_pinned_choose_ntau_does_not_grow(tmp_path):
    p = tmp_path / "evo.h5"
    src = Grid(nx=2, ny=2, neta=1, ntau=5, x_min=-0.5, dx=1.0, y_min=-0.5, dy=1.0,
               eta_min=0.0, deta=1.0, tau_min=0.6, dtau=0.1)
    attrs = attrs_from_grids(src, src)
    attrs["choose_ntau"] = 3
    with FnoH5Writer(p, attrs, nevents=0, growable_tau=False, force=True) as w:
        w.ensure_capacity(nevents=1, choose_ntau=99)
        assert w.choose_ntau == 3
        assert w.nevents == 1


def test_reader_round_trip(tmp_path):
    from jetscape.fast_h5_bulk import read_fast_h5_bulk

    p = tmp_path / "evo.h5"
    _, payload = write_two_ragged_events(p)
    d = read_fast_h5_bulk(p)
    assert len(d["events"]) == 2
    assert list(d["ntau"]) == [4, 7]
    for got, want in zip(d["events"], payload):
        assert got.shape == want.shape           # (ntau, nx, ny, neta, 4), as the ROOT reader
        np.testing.assert_array_equal(got, want)
    assert d["features"] == ("energy_density", "vx", "vy", "vz")
    assert d["grid"]["x"].shape == (3,)

    sub = read_fast_h5_bulk(p, entry_start=1, entry_stop=2)
    assert len(sub["events"]) == 1
    np.testing.assert_array_equal(sub["events"][0], payload[1])


# ───────────────────────────────────────────── FNO4d loader compatibility
FNO4D = Path("/Users/du8478/FNO4d")


@pytest.mark.skipif(not FNO4D.exists(), reason="FNO4d checkout not present")
def test_fno4d_loaders_accept_the_file(tmp_path):
    # loc_libs/data/sources.py does a bare `from read_3d_hdf5 import ...`, so loc_libs
    # itself has to be on the path, not just the checkout root.
    for d in (FNO4D, FNO4D / "loc_libs"):
        sys.path.insert(0, str(d))
    read_3d_hdf5 = pytest.importorskip("loc_libs.read_3d_hdf5")

    p = tmp_path / "evo.h5"
    write_two_ragged_events(p)

    d = read_3d_hdf5.read_3d_data_hdf5([str(p)], lazy=False)
    assert d["arr"].shape[0] == 2
    assert int(d["nevents"]) == 2

    m = read_3d_hdf5.MultiH5Array([str(p)])
    assert len(m) == 2

    from loc_libs.data.dataset import live_tau_lengths
    assert list(live_tau_lengths(np.asarray(d["arr"]))) == [4, 7]


# ───────────────────────────────────── framework-mode channel selection
class _StubBulkInfo:
    """Just enough of a bound EvolutionHistory for event_array(grid_mode='framework')."""

    def __init__(self, arr5, g):
        self._arr5, self._g = arr5, g
        for k in ("nx", "ny", "neta", "ntau", "x_min", "dx", "y_min", "dy",
                  "eta_min", "deta", "tau_min", "dtau", "boost_invariant"):
            setattr(self, k, getattr(g, k))

    def to_numpy_full(self, n_features=5):
        assert n_features == 5, "framework mode must ask for all five fields"
        return self._arr5[..., :n_features]


class _StubHydro:
    def __init__(self, b):
        self._b = b

    def get_bulk_info(self):
        return self._b


def test_framework_mode_picks_e_vx_vy_vz_and_not_temperature():
    """to_numpy_full is [e, T, vx, vy, vz]; slicing [:4] would store T as vx."""
    from jetscape.bulk_sources import event_array

    _, g = make_source(nx=4, ny=4, neta=3, ntau=3)
    rng = np.random.default_rng(3)
    arr5 = rng.random((g.ntau, g.nx, g.ny, g.neta, 5)).astype(np.float32)
    hydro = _StubHydro(_StubBulkInfo(arr5, g))

    arr, src, out = event_array(hydro, grid_mode="framework")
    assert arr.shape[-1] == 4
    assert out == src, "no out_spec -> the source grid"
    for want, got in zip((0, 2, 3, 4), range(4)):
        np.testing.assert_allclose(arr[..., got], arr5[..., want], rtol=0, atol=2e-6)
    # the temperature channel must not appear anywhere
    assert not np.allclose(arr[..., 1], arr5[..., 1])


# ──────────────────────────── choose_ntau as a cross-file contract (README)
def test_pinning_choose_ntau_generously_is_free(tmp_path):
    """Over-estimating choose_ntau must cost no disk: the pad chunks are never allocated.

    This is what makes "pin it generously for the whole campaign" the right advice, so it
    is a promise the README makes.
    """
    sizes = {}
    for pin in (0, 200):
        p = tmp_path / f"pin{pin}.h5"
        src = Grid(nx=16, ny=16, neta=4, ntau=14, x_min=-2.0, dx=0.3, y_min=-2.0, dy=0.3,
                   eta_min=-0.5, deta=0.2, tau_min=0.6, dtau=0.1)
        attrs = attrs_from_grids(src, src)
        attrs["choose_ntau"] = pin or 1
        rng = np.random.default_rng(1)
        with FnoH5Writer(p, attrs, nevents=0, force=True, growable_tau=not pin) as w:
            for i, nt in enumerate((10, 14)):
                w.ensure_capacity(nevents=i + 1, choose_ntau=nt)
                for t in range(nt):
                    w.write_frame(i, t, rng.random((4, 16, 16, 4)).astype(np.float32) + 1)
                w.set_event_meta(i, nt, src.tau_min + nt * src.dtau)
        with h5py.File(p, "r") as f:
            sizes[pin] = f["arr"].id.get_storage_size()
            assert f["arr"].shape[-1] == (200 if pin else 14)
    assert sizes[200] == sizes[0], f"generous pin cost {sizes[200] - sizes[0]} extra bytes"


def test_tau_axis_can_be_repadded_in_place(tmp_path):
    """maxshape leaves the tau axis open, so mismatched files can be reconciled in place.

    No rewrite, no size change, and the per-event lifetimes must survive -- the new region
    is unallocated chunks that read back as exactly 0.0.
    """
    paths = []
    for name, ntaus in (("jobA", (10, 14)), ("jobB", (9, 21))):
        p = tmp_path / f"{name}.h5"
        write_two_ragged_events(p, ntaus=ntaus)
        paths.append(p)

    with h5py.File(paths[0], "r+") as f:
        before = f["arr"].id.get_storage_size()
        f["arr"].resize(21, axis=5)
        f.attrs["choose_ntau"] = 21
        assert f["arr"].id.get_storage_size() == before, "re-pad must not move data"
        assert np.all(f["arr"][1, ..., 14:] == 0.0), "grown region must read back as 0.0"
        live = [int((np.abs(f["arr"][i, 0]).max(axis=(0, 1, 2)) > 0).sum()) for i in (0, 1)]
    assert live == [10, 14], "per-event lifetimes must survive the re-pad"

    with h5py.File(paths[1], "r") as f:
        assert int(f.attrs["choose_ntau"]) == 21          # now agrees with jobA


# ───────────────────────────────────────────────────── repad_to()
def _job(tmp_path, name, ntaus):
    p = tmp_path / f"{name}.h5"
    write_two_ragged_events(p, ntaus=ntaus)
    return p


def test_repad_to_reconciles_files_and_keeps_them_loadable(tmp_path):
    from jetscape.fno_h5_writer import repad_to

    a, b = _job(tmp_path, "jobA", (10, 14)), _job(tmp_path, "jobB", (9, 21))
    with h5py.File(a, "r") as f:
        before = f["arr"].id.get_storage_size()

    target, changed = repad_to([a, b], verbose=False)
    assert target == 21 and changed == [str(a)]

    with h5py.File(a, "r") as f:
        assert f["arr"].shape[-1] == 21
        assert int(f.attrs["choose_ntau"]) == 21
        assert f["arr"].id.get_storage_size() == before, "re-pad must not move data"
        assert np.all(f["arr"][1, ..., 14:] == 0.0)
    # the whole point: the two now concatenate
    from jetscape.fast_h5_bulk import read_fast_h5_bulk
    assert [e.shape[0] for e in read_fast_h5_bulk(a)["events"]] == [10, 14]


def test_repad_to_accepts_a_generous_target_and_is_idempotent(tmp_path):
    from jetscape.fno_h5_writer import repad_to

    a, b = _job(tmp_path, "jobA", (10, 14)), _job(tmp_path, "jobB", (9, 21))
    target, changed = repad_to([a, b], choose_ntau=200, verbose=False)
    assert target == 200 and sorted(changed) == sorted([str(a), str(b)])
    assert repad_to([a, b], choose_ntau=200, verbose=False)[1] == [], "second run is a no-op"


def test_repad_to_refuses_to_shrink(tmp_path):
    """Shrinking an HDF5 dataset discards data permanently -- never silently."""
    from jetscape.fno_h5_writer import repad_to

    a, b = _job(tmp_path, "jobA", (10, 14)), _job(tmp_path, "jobB", (9, 21))
    with pytest.raises(ValueError, match="discards data permanently"):
        repad_to([a, b], choose_ntau=14, verbose=False)
    with h5py.File(b, "r") as f:
        assert f["arr"].shape[-1] == 21, "nothing may have been written"


def test_repad_to_refuses_mismatched_spatial_dims(tmp_path):
    from jetscape.fno_h5_writer import repad_to

    a = _job(tmp_path, "jobA", (10, 14))
    b = tmp_path / "other.h5"
    write_two_ragged_events(b, ntaus=(10, 14), nx=5)          # different nx
    with pytest.raises(ValueError, match="could never be concatenated"):
        repad_to([a, b], verbose=False)


def test_repad_to_refuses_a_fixed_tau_axis(tmp_path):
    """An FNO4d-written file has no maxshape, so it needs a rewrite, not a resize."""
    from jetscape.fno_h5_writer import repad_to

    a = _job(tmp_path, "jobA", (10, 14))
    fixed = tmp_path / "fixed.h5"
    with h5py.File(a, "r") as src, h5py.File(fixed, "w") as dst:
        for k, v in src.attrs.items():
            dst.attrs[k] = v
        dst.attrs["choose_ntau"] = 5
        dst.create_dataset("arr", data=src["arr"][..., :5])   # no maxshape
    with pytest.raises(ValueError, match="not growable"):
        repad_to([a, fixed], verbose=False)


def test_repad_to_grows_the_source_group_with_arr(tmp_path):
    """fast_data's /source datasets carry the same tau axis; leaving them behind would
    silently desync the file."""
    from jetscape.fno_h5_writer import repad_to

    p = tmp_path / "withsource.h5"
    with h5py.File(p, "w") as f:
        f.attrs["choose_ntau"] = 14
        f.create_dataset("arr", (2, 4, 8, 8, 2, 14), dtype=np.float32,
                         maxshape=(2, 4, 8, 8, 2, None), chunks=(1, 4, 8, 8, 2, 1))
        g = f.create_group("source")
        g.create_dataset("S", (2, 4, 8, 8, 2, 14), dtype=np.float32,
                         maxshape=(2, 4, 8, 8, 2, None), chunks=(1, 4, 8, 8, 2, 1))
        g.create_dataset("P_cart", (2, 14, 4), dtype=np.float64,
                         maxshape=(2, None, 4), chunks=(1, 14, 4))

    repad_to(p, choose_ntau=30, verbose=False)
    with h5py.File(p, "r") as f:
        assert f["arr"].shape[5] == 30
        assert f["source/S"].shape[5] == 30, "/source/S must track arr"
        assert f["source/P_cart"].shape[1] == 30, "/source/P_cart must track arr"


def test_repad_to_rejects_an_already_inconsistent_file(tmp_path):
    from jetscape.fno_h5_writer import repad_to

    p = tmp_path / "skew.h5"
    with h5py.File(p, "w") as f:
        f.attrs["choose_ntau"] = 14
        f.create_dataset("arr", (2, 4, 8, 8, 2, 14), dtype=np.float32,
                         maxshape=(2, 4, 8, 8, 2, None), chunks=(1, 4, 8, 8, 2, 1))
        f.create_group("source").create_dataset(
            "S", (2, 4, 8, 8, 2, 9), dtype=np.float32,          # already out of step
            maxshape=(2, 4, 8, 8, 2, None), chunks=(1, 4, 8, 8, 2, 1))
    with pytest.raises(ValueError, match="already inconsistent"):
        repad_to(p, choose_ntau=30, verbose=False)


# ───────────────────────── the HDF5 tooling must not need the X-SCAPE build
def test_h5_tooling_imports_without_the_compiled_extension(tmp_path):
    """`repad_to` / `FnoH5Writer` touch only h5py+numpy, so they must stay usable on a
    training machine with no X-SCAPE build.

    Run in a subprocess against a copy of the package with the extension removed -- the
    package __init__ used to hard-import it, which defeated the point.
    """
    import shutil
    import subprocess

    pkg = Path(__file__).resolve().parents[1] / "python" / "jetscape"
    dst = tmp_path / "jetscape"
    shutil.copytree(pkg, dst, ignore=shutil.ignore_patterns("__pycache__", "*.so"))

    code = (
        "import jetscape as js;"
        "assert js.HAS_CORE is False, js.HAS_CORE;"
        "assert js.HAS_H5PY is True;"
        "assert js.repad_to and js.FnoH5Writer and js.grid_attrs and js.read_fast_h5_bulk;"
        "\ntry:\n"
        "    js.JetScape\n"
        "except ImportError as e:\n"
        "    assert 'pyjetscape_core' in str(e), e\n"
        "else:\n"
        "    raise AssertionError('a core name must raise ImportError, not resolve')\n"
        "print('ok')"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
                       capture_output=True, text=True)
    assert r.returncode == 0, f"stdout={r.stdout!r} stderr={r.stderr!r}"
    assert "ok" in r.stdout


def test_repad_h5_runs_as_a_plain_script(tmp_path):
    """The two files can be copied anywhere and used as a utility, with no package."""
    import shutil
    import subprocess

    pkg = Path(__file__).resolve().parents[1] / "python" / "jetscape"
    for name in ("fno_h5_writer.py", "repad_h5.py"):
        shutil.copy(pkg / name, tmp_path / name)

    a, b = tmp_path / "a.h5", tmp_path / "b.h5"
    write_two_ragged_events(a, ntaus=(4, 7))
    write_two_ragged_events(b, ntaus=(5, 11))

    r = subprocess.run([sys.executable, str(tmp_path / "repad_h5.py"), str(a), str(b)],
                       cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "choose_ntau = 11" in r.stdout
    with h5py.File(a, "r") as f:
        assert f["arr"].shape[-1] == 11
