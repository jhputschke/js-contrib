"""The dataset format: FNO4d's HDF5 schema, written by the vendored FnoH5Writer.

The .npz the drivers can also emit is a convenience for looking at one event. This is the
format the training pipeline reads, and a file written here must be indistinguishable from
one written by fast_data's own generate.py -- otherwise every downstream reader needs a
special case, which is exactly what vendoring the writer was meant to avoid.
"""

import copy

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
pytest.importorskip("torch")

from fast_data.config import DEFAULTS                                  # noqa: E402
from fasthydro.config import resolve as fh_resolve                     # noqa: E402
from fasthydro.grid import GridSpec                                    # noqa: E402
from fasthydro.h5_writer import PairedH5Writer                         # noqa: E402


def _cfg():
    c = copy.deepcopy(DEFAULTS)
    c["grid"].update(nx=9, ny=7, neta=5, dx=0.6, dy=0.5, deta=0.4)
    c["time"].update(tau0=0.6, record_dtau=0.2, choose_ntau=6)
    c["eos"].update(kind="conformal", dof=42.25)
    c["run"].update(device="cpu", dtype="float64")
    c["fasthydro"] = fh_resolve({})
    return c


class _FakeHydro:
    """Stands in for a FastHydro that has run, so the writer can be tested on its own."""

    def __init__(self, g, seed, with_source):
        rng = np.random.default_rng(seed)
        shape = (4, g.nx, g.ny, g.neta, g.ntau)
        self.g = g
        self.arr = rng.uniform(0, 10, shape).astype(np.float32)
        self.src = rng.uniform(-1, 1, shape).astype(np.float32) if with_source else None
        self.ic_sha256 = "a" * 64
        self.diag = {"ntau_freezeout": g.ntau - 1, "tau_freezeout": 1.4, "n_steps": 7,
                     "wall_s": 0.5, "frozen_out": True,
                     "P_cart": np.zeros((g.ntau, 4)), "T_max": np.zeros(g.ntau)}


class _FakeBridge:
    def __init__(self, n):
        from fast_data.liquefier.droplets import DropletArray
        d = np.tile(np.array([[1.0, 0.2, -0.3, 0.1, 5.0, 1.0, 0.0, 0.5]]), (n, 1))
        self.droplets = DropletArray(d, np.array([0, n], dtype=np.int64))
        self.n_late, self.n_early, self.E_in_window = 0, 0, float(d[:, 4].sum())


@pytest.fixture
def written(tmp_path):
    cfg = _cfg()
    g = GridSpec.from_cfg(cfg)
    path = tmp_path / "pair.h5"
    counts = [3, 5]
    with PairedH5Writer(path, cfg, len(counts)) as w:
        for i, n in enumerate(counts):
            w.append(i, _FakeHydro(g, 10 + i, False), _FakeHydro(g, 100 + i, True),
                     _FakeBridge(n))
    return path, cfg, g, counts


def test_is_a_fast_data_file(written):
    """The format tag and grid attrs every FNO4d reader keys off."""
    path, cfg, g, _ = written
    with h5py.File(path) as f:
        assert f.attrs["format"] == "fast_data/hydro_evolution"
        assert f.attrs["nevents_written"] == 2 and bool(f.attrs["complete"])
        for k, v in (("nx", g.nx), ("ny", g.ny), ("neta", g.neta), ("choose_ntau", g.ntau)):
            assert int(f.attrs[k]) == v
        for k, v in (("dx", g.dx), ("deta", g.deta), ("tau_min", g.tau0),
                     ("dtau", g.record_dtau)):
            assert float(f.attrs[k]) == pytest.approx(v)
        # cell-centred mins, as fast_data writes them
        assert float(f.attrs["eta_min"]) == pytest.approx(g.eta_min)


def test_arr_is_the_jet_leg_and_arr_bg_the_background(written):
    """fast_data's convention is that arr already contains the source, so arr is the JET leg.
    Getting this backwards would train the FNO on the wrong target."""
    path, _, g, _ = written
    with h5py.File(path) as f:
        assert f["arr"].shape == (2, 4, g.nx, g.ny, g.neta, g.ntau)
        assert f["arr_bg"].shape == f["arr"].shape
        assert not np.array_equal(f["arr"][0], f["arr_bg"][0])
        assert "jet" in f.attrs["arr_is"] and "background" in f.attrs["arr_bg_is"]
        # the source belongs to the leg stored in arr
        assert f["source/S"].shape == f["arr"].shape
        assert np.abs(f["source/S"][0]).max() > 0


def test_droplets_are_stored_per_event(written):
    path, _, _, counts = written
    with h5py.File(path) as f:
        off = f["source/offsets"][:]
        assert list(np.diff(off)) == counts
        assert f["source/droplets"].shape == (sum(counts), 8)
        assert f["source/P_cart"].shape[0] == 2


def test_freezeout_bookkeeping_for_both_legs(written):
    path, _, g, _ = written
    with h5py.File(path) as f:
        for k in ("ntau_freezeout", "ntau_freezeout_bg",
                  "tau_freezeout", "tau_freezeout_bg"):
            assert f[k].shape == (2,)
        assert int(f["ntau_freezeout"][0]) == g.ntau - 1


def test_provenance_is_recorded(written):
    path, _, _, _ = written
    with h5py.File(path) as f:
        assert f.attrs["generator"] == "fasthydro"
        assert f.attrs["pairing"] == "bg_jet"
        assert f.attrs["source_mode"] == "conservative"
        assert f.attrs["hard_vertex"] == "ncoll"
        import json
        json.loads(f.attrs["config_json"])          # must be round-trippable
        json.loads(f.attrs["fasthydro_json"])


def test_the_vendored_reader_loads_it(written):
    """The whole point of using FnoH5Writer unmodified. Skips unless FNO4d is on the path,
    since its training loaders are not vendored here."""
    path, _, g, _ = written
    reader = pytest.importorskip(
        "read_3d_hdf5", reason="FNO4d's loc_libs readers are not vendored by FastHydro")
    d = reader.read_3d_data_hdf5(str(path))
    assert d["arr"].shape == (2, 4, g.nx, g.ny, g.neta, g.ntau)
    assert float(d["dtau"]) == pytest.approx(g.record_dtau)
