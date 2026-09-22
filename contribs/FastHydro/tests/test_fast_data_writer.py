"""Gates for the HDF5 writer: the file must be a drop-in for the existing training loaders."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

h5py = pytest.importorskip("h5py")

from fast_data import writer                              # noqa: E402

# data.dataset, data.sources and read_3d_hdf5 are FNO4d training-side loaders (loc_libs/),
# not part of fast_data, so they are not vendored here.  The point of these assertions is
# that the written file stays a drop-in for those loaders, so the test is kept verbatim and
# simply skips unless FNO4d is on PYTHONPATH.
live_tau_lengths = pytest.importorskip(
    "data.dataset",
    reason="FNO4d's loc_libs training loaders are not vendored by FastHydro").live_tau_lengths
build_source = pytest.importorskip("data.sources").build_source          # noqa: E402
_r3 = pytest.importorskip("read_3d_hdf5")                                # noqa: E402
MultiH5Array, read_3d_data_hdf5 = _r3.MultiH5Array, _r3.read_3d_data_hdf5

NX = NY = 8
NZ, NT, NEV = 4, 6, 2


def _attrs():
    return writer.grid_attrs(NX, NY, NZ, 0.5, 0.5, 0.5, tau_min=0.6, dtau=0.1, choose_ntau=NT)


def _event(seed=0, ntau_fo=4):
    rng = np.random.default_rng(seed)
    a = rng.random((4, NX, NY, NZ, NT)).astype(np.float32) + 0.5
    a[..., ntau_fo - 1:] = 0.0            # the measured convention: live == ntau_fo - 1
    return a


def _write(path, nev=NEV, **kw):
    with writer.FnoH5Writer(path, _attrs(), nev, force=True, **kw) as w:
        for i in range(nev):
            w.append_event(i, _event(i, 4 + i), 4 + i, 0.6 + (3 + i) * 0.1)
    return str(path)


def test_w1_the_file_loads_through_every_canonical_reader(tmp_path):
    p = _write(tmp_path / "a.h5")
    d = read_3d_data_hdf5([p])
    assert d["arr"].shape == (NEV, 4, NX, NY, NZ, NT)
    assert d["arr"].dtype == np.float32
    for k in writer.SCALAR_KEYS:
        assert k in d, f"missing required attribute {k}"
    assert d["choose_ntau"] == NT and d["nevents"] == NEV
    a = MultiH5Array([p])
    assert len(a) == NEV and a[0].shape == (4, NX, NY, NZ, NT)
    assert len(build_source([p])) == NEV


def test_w2_nevents_attribute_matches_the_dataset_shape(tmp_path):
    """read_3d_data_hdf5 trusts the attribute, MultiH5Array trusts the shape; a mismatch
    silently corrupts the per-event metadata slicing."""
    p = _write(tmp_path / "b.h5")
    with h5py.File(p) as f:
        assert int(f.attrs["nevents"]) == f["arr"].shape[0]
        assert f["arr"].chunks[0] == 1        # one event per chunk -> true random access
        assert f["arr"].compression == "lzf"


def test_w3_live_tau_lengths_agrees_with_ntau_freezeout(tmp_path):
    p = _write(tmp_path / "c.h5")
    d = read_3d_data_hdf5([p])
    lt = live_tau_lengths(MultiH5Array([p]))
    assert list(lt) == [n - 1 for n in d["ntau_freezeout"]]


def test_w4_tau_freezeout_follows_the_measured_relation(tmp_path):
    p = _write(tmp_path / "d.h5")
    d = read_3d_data_hdf5([p])
    assert np.allclose(d["tau_freezeout"],
                       d["tau_min"] + (d["ntau_freezeout"] - 1) * d["dtau"], atol=1e-5)


def test_w5_two_files_concatenate(tmp_path):
    """The multi-file consistency check is on (nFeatures, nx, ny, neta, choose_ntau)."""
    p1, p2 = _write(tmp_path / "e1.h5"), _write(tmp_path / "e2.h5")
    a = MultiH5Array([p1, p2])
    assert len(a) == 2 * NEV


def test_w6_extra_attributes_survive_the_round_trip(tmp_path):
    p = _write(tmp_path / "f.h5", extra_attrs={"generator": "fast_data", "answer": 42})
    d = read_3d_data_hdf5([p])
    assert d["generator"] == "fast_data" and int(d["answer"]) == 42


def test_w7_source_group_layout(tmp_path):
    path = str(tmp_path / "g.h5")
    S = np.zeros((4, NX, NY, NZ, NT), dtype=np.float32)
    S[..., 2] = 1.0
    drops = np.arange(16, dtype=np.float64).reshape(2, 8)
    with writer.FnoH5Writer(path, _attrs(), 1, force=True, write_source=True) as w:
        w.append_event(0, _event(), 4, 0.9, S_ev=S, P_cart=np.ones((NT, 4)), droplets=drops)
    with h5py.File(path) as f:
        assert (f["source/S"][0, ..., 0] == 0).all(), "frame 0 is the IC and carries no source"
        assert f["source/offsets"][-1] == f["source/droplets"].shape[0]
        for k in ("convention", "units", "channels", "droplet_columns"):
            assert k in f["source"].attrs
        assert f.attrs["has_source"]


def test_w8_refuses_to_clobber_and_marks_completeness(tmp_path):
    p = str(tmp_path / "h.h5")
    _write(p)
    with pytest.raises(FileExistsError):
        writer.FnoH5Writer(p, _attrs(), NEV)
    with h5py.File(p) as f:
        assert bool(f.attrs["complete"]) is True

    # an interrupted run is structurally valid but explicitly marked incomplete
    q = str(tmp_path / "i.h5")
    w = writer.FnoH5Writer(q, _attrs(), 3, force=True)
    w.append_event(0, _event(), 4, 0.9)
    w.close()
    with h5py.File(q) as f:
        assert bool(f.attrs["complete"]) is False
        assert int(f.attrs["nevents_written"]) == 1
        assert f["arr"].shape[0] == 3
