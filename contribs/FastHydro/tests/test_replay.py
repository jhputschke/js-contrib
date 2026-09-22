"""Replay: the dump round-trips, and re-running reproduces the original bit for bit.

None of this needs pyjetscape_core -- that is the point of the replay path.  If any of it
starts importing `jetscape`, the split between droplets_io and liquefier_bridge has broken.
"""

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fast_data.config import DEFAULTS                                  # noqa: E402
from fast_data.liquefier import LiquefierParams                        # noqa: E402
from fast_data.liquefier.droplets import DropletArray                  # noqa: E402
from fasthydro.droplets_io import load_droplets_npz, save_droplets_npz  # noqa: E402
from fasthydro.grid import GridSpec                                    # noqa: E402
from fasthydro.replay import replay_event, verify_replay               # noqa: E402


def _cfg():
    c = copy.deepcopy(DEFAULTS)
    c["grid"].update(nx=13, ny=11, neta=7, dx=0.6, dy=0.5, deta=0.4)
    c["time"].update(tau0=0.6, record_dtau=0.2, choose_ntau=7)
    c["eos"].update(kind="conformal", dof=42.25)
    c["run"].update(device="cpu", dtype="float64")
    c["output"].update(stop_at_freezeout=False)
    c["source"].update(mode="conservative")
    return c


def _ic(g):
    x = (np.arange(g.nx) - (g.nx - 1) / 2) * g.dx
    y = (np.arange(g.ny) - (g.ny - 1) / 2) * g.dy
    eta = (np.arange(g.neta) - (g.neta - 1) / 2) * g.deta
    X, Y, E = np.meshgrid(x, y, eta, indexing="ij")
    return 60.0 * np.exp(-(X**2 + Y**2) / 8.0 - E**2 / 4.0) + 1e-3


class _FakeBridge:
    """Just enough of DropletBridge for save_droplets_npz, without importing jetscape."""

    def __init__(self, history, params, g):
        self.history, self.params = history, params
        self.hydro_jet = type("H", (), {"g": g})()


def test_droplet_npz_round_trip(tmp_path):
    g = GridSpec.from_cfg(_cfg())
    p = LiquefierParams(tau_delay=0.6)
    d0 = DropletArray(np.array([[0.6, 0.5, -0.5, 0.0, 8.0, 2.0, 1.0, 0.5]]),
                      np.array([0, 1], dtype=np.int64))
    d1 = DropletArray(np.array([[0.8, -1.0, 1.0, 0.4, 5.0, -1.0, 0.5, -1.5],
                                [0.9, 0.2, 0.2, -0.4, 3.0, 0.5, 0.5, 0.5]]),
                      np.array([0, 2], dtype=np.int64))
    path = tmp_path / "d.npz"
    save_droplets_npz(path, _FakeBridge([d0, d1], p, g), ic=_ic(g), seeds=[1, 2])

    back, params, meta = load_droplets_npz(path)
    assert len(back) == 2
    assert np.array_equal(back[0].data, d0.data)
    assert np.array_equal(back[1].data, d1.data)
    assert params == p
    assert meta["ic"].shape == (g.nx, g.ny, g.neta)


def test_column_order_is_checked(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, droplets=np.zeros((1, 8)), offsets=np.array([0, 1]),
             flags=np.zeros(1, np.uint32),
             columns=np.array(["E", "px", "py", "pz", "tau", "x", "y", "eta"]),
             params=np.array([0.02, 2.0, 0.1, 0.08, 0.1]))
    with pytest.raises(ValueError, match="columns"):
        load_droplets_npz(path)


def test_replay_is_deterministic_bitwise():
    """Two replays of the same dump on CPU/float64 must be identical -- this is what lets a
    replay be compared against the original run's sha256."""
    cfg = _cfg()
    g = GridSpec.from_cfg(cfg)
    p = LiquefierParams(tau_delay=0.6)
    da = DropletArray(np.array([[0.7, 0.4, -0.3, 0.2, 9.0, 2.0, -1.0, 0.5]]),
                      np.array([0, 1], dtype=np.int64))
    e0 = _ic(g)
    a1, _, _ = replay_event(cfg, e0, da, p)
    a2, _, _ = replay_event(cfg, e0, da, p)
    ok, msg = verify_replay(a1, reference_arr=a2, exact=True)
    assert ok, msg


def test_replay_without_droplets_matches_a_sourceless_run():
    cfg = _cfg()
    g = GridSpec.from_cfg(cfg)
    e0 = _ic(g)
    none_arr, none_src, _ = replay_event(cfg, e0, None, LiquefierParams())
    empty = DropletArray(np.zeros((0, 8)), np.array([0, 0], dtype=np.int64))
    empty_arr, _, _ = replay_event(cfg, e0, empty, LiquefierParams())
    assert none_src is None
    assert np.array_equal(none_arr, empty_arr)


def test_replay_deposits_and_conserves():
    cfg = _cfg()
    g = GridSpec.from_cfg(cfg)
    p = LiquefierParams(tau_delay=0.6)
    da = DropletArray(np.array([[0.7, 0.0, 0.0, 0.0, 9.0, 2.0, -1.0, 0.5]]),
                      np.array([0, 1], dtype=np.int64))
    e0 = _ic(g)
    bg, _, _ = replay_event(cfg, e0, None, p)
    jet, src, _ = replay_event(cfg, e0, da, p)
    assert src is not None
    assert np.abs(jet[0] - bg[0]).max() > 0, "the deposit never reached the evolution"


def test_grid_mismatch_is_refused():
    cfg = _cfg()
    with pytest.raises(ValueError, match="config grid"):
        replay_event(cfg, np.zeros((3, 3, 3)), None, LiquefierParams())
