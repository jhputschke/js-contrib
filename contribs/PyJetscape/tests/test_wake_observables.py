"""wake_observables.py: the EoS lookup, the flux sums and the prefetch give what the
straightforward code gave (np.interp per column, the stacked cells, reading in the loop)."""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "python"))
sys.path.insert(0, str(HERE.parent / "example" / "analysis"))

pytest.importorskip("h5py")
pytest.importorskip("scipy")

import wake_observables as wo  # noqa: E402


def _table(path, e):
    """A MUSIC-format EoS table (e, P, s, T as little-endian float64 rows)."""
    P = e / 3 - 0.01 * np.sqrt(e)
    s = 5.0 * e ** 0.75
    T = 0.2 * e ** 0.25
    np.column_stack([e, P, s, T]).astype("<f8").tofile(path)
    return path


def _reference(eos, e):
    """The EoS as wake_observables computed it before: np.interp in log e per column."""
    e = np.asarray(e, dtype=np.float64)
    le = np.log(np.maximum(e, eos.e[0]))
    P, s, T = (np.interp(le, eos.le, c) for c in (eos.P, eos.s, eos.T))
    low = e < eos.e[0]
    if np.any(low):
        r = np.clip(e[low], 0.0, None) / eos.e[0]
        P[low] *= r
        s[low] *= r ** 0.75
        T[low] *= r ** 0.25
    return P, s, T


@pytest.mark.parametrize("uniform", [True, False])
def test_eos_equals_np_interp_in_log_e(tmp_path, uniform):
    rng = np.random.default_rng(1)
    e_tab = (np.linspace(4e-4, 50.0, 5001) if uniform
             else np.sort(rng.uniform(4e-4, 50.0, 5001)))   # MUSIC's are uniform
    eos = wo.EoS(_table(tmp_path / "eos.dat", e_tab))
    assert (eos._de is not None) == uniform
    e = np.concatenate([
        rng.uniform(0.0, 60.0, 20000),                   # inside and above the table
        rng.uniform(0.0, 4e-4, 100),                     # below: P ~ e, s ~ e^3/4
        e_tab[::7], np.nextafter(e_tab[::7], 0), np.nextafter(e_tab[::7], 100),  # at nodes
        [0.0, e_tab[0], e_tab[-1], 1e3, np.nan]])
    for got, ref in zip(eos(e), _reference(eos, e)):
        assert np.allclose(got, ref, rtol=1e-12, atol=0, equal_nan=True)
    # scalars, as the droplet and apex lookups pass them
    for x in (0.37, 1e-5, 100.0):
        assert np.allclose([float(v) for v in eos(x)],
                           [float(v[0]) for v in _reference(eos, [x])], rtol=1e-12, atol=0)


class _Grid:
    """What flux_cells needs of wake_observables.Grid."""

    def __init__(self, nx, ny, neta):
        self.dx, self.dy, self.deta = 0.3, 0.3, 0.2
        eta = self.deta * (np.arange(neta) - neta // 2)
        self.ch = np.cosh(eta)[None, None, :]
        self.sh = np.sinh(eta)[None, None, :]

    def tau(self, k):
        return 0.6 + 0.1 * np.asarray(k)


def _reference_cells(frame, tau, eos, g):
    """flux_cells as it was."""
    e = frame[0].astype(np.float64)
    vx, vy, vz = (frame[i].astype(np.float64) for i in (1, 2, 3))
    P, s, _ = _reference(eos, e)
    g2 = 1.0 / (1.0 - np.clip(vx * vx + vy * vy + vz * vz, 0.0, 1.0 - 1e-12))
    w = (e + P) * g2
    ch, sh = g.ch, g.sh
    dv = tau * g.dx * g.dy * g.deta
    p0 = dv * (ch * (w - P) - sh * w * vz)
    px = dv * w * vx * (ch - sh * vz)
    py = dv * w * vy * (ch - sh * vz)
    pz = dv * (ch * w * vz - sh * (w * vz * vz + P))
    S = dv * s * np.sqrt(g2) * (ch - sh * vz)
    return np.stack([p0, px, py, pz]), S


def test_flux_equals_the_stacked_cells(tmp_path):
    rng = np.random.default_rng(2)
    eos = wo.EoS(_table(tmp_path / "eos.dat", np.linspace(4e-4, 50.0, 5001)))
    g = _Grid(9, 8, 7)
    arr = np.empty((4, 9, 8, 7, 5), np.float32)          # e, vx, vy, vz per frame
    arr[0] = rng.uniform(0.0, 30.0, arr.shape[1:])
    arr[1:] = rng.uniform(-0.55, 0.55, (3,) + arr.shape[1:])
    P, S = wo.flux_series(arr, eos, g)
    for k in range(arr.shape[-1]):
        c, s = _reference_cells(arr[..., k], g.tau(k), eos, g)
        assert np.allclose(P[k], c.sum(axis=(1, 2, 3)), rtol=1e-12, atol=0)
        assert np.isclose(S[k], s.sum(), rtol=1e-12, atol=0)
        cells, s_cells = wo.flux_cells(arr[..., k], g.tau(k), eos, g)
        assert np.allclose(cells, c, rtol=1e-12, atol=0)
        assert np.allclose(s_cells, s, rtol=1e-12, atol=0)


def test_prefetch_keeps_the_order_and_loads_ahead():
    main = threading.get_ident()
    started = [threading.Event() for _ in range(5)]
    in_thread = []

    def load(k):
        in_thread.append(threading.get_ident() != main)
        started[k].set()
        return k * k

    got = []
    for k, v in enumerate(wo.prefetch(range(5), load)):
        got.append(v)
        if k + 1 < 5:            # item k + 1 loads while the caller still holds item k
            assert started[k + 1].wait(10)
    assert got == [0, 1, 4, 9, 16] and all(in_thread)
    assert list(wo.prefetch([], load)) == []


def test_prefetch_passes_a_load_error_on():
    def load(k):
        if k == 2:
            raise OSError("chunk 2 unreadable")
        return k

    with pytest.raises(OSError, match="chunk 2"):
        list(wo.prefetch(range(4), load))
