"""
Tests for the framework SurfaceFinder, driven through PyFluidDynamics.

A synthetic 3+1D evolution with a uniform flow and a temperature that falls with tau only,
T = T0 (tau0/tau)^(1/3), has a flat freeze-out surface at the tau where T = T_sw.  Every
surface cell must then carry a normalised four-velocity, u.u = 1, whose Milne components are
the boost of gamma(1, v) by eta.  Before the fix in PrepareASurfaceCell the flow was built as
u0 = sqrt(1 + v^2), u^i = u0 v^i, which gives u.u = 1 - v^4 (13% short at v = 0.7).

    pytest tests/test_surface_finder.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

core = pytest.importorskip("jetscape.pyjetscape_core")

T0, TAU0, T_SW = 0.30, 0.6, 0.15
TAU_SW = TAU0 * (T0 / T_SW) ** 3          # 4.8 fm/c


def _surface(v, *, dtau=0.0, dx=0.0, deta=0.0):
    """Surface cells of a uniform-v, T(tau) evolution on tau x [-2,2]^2 x [-1,1]."""
    ntau, nx, ny, neta = 61, 9, 9, 9
    tau = TAU0 + 0.1 * np.arange(ntau)
    x_min, dxg = -2.0, 0.5
    eta_min, detag = -1.0, 0.25

    fd = core.FluidDynamics()
    fd.set_hydro_grid_info(
        tau_min=TAU0, dtau=0.1, ntau=ntau,
        x_min=x_min, dx=dxg, nx=nx, y_min=x_min, dy=dxg, ny=ny,
        eta_min=eta_min, deta=detag, neta=neta, boost_inv=False, tau_eta_is_tz=False)

    shape = (nx, ny, neta, ntau)
    T = np.broadcast_to(T0 * (TAU0 / tau) ** (1.0 / 3.0), shape)
    arr = np.stack([T, 10.0 * T ** 4,
                    np.full(shape, v[0]), np.full(shape, v[1]), np.full(shape, v[2])])
    fd.store_fluid_cells_from_numpy_3d(np.ascontiguousarray(arr, dtype=np.float32),
                                       ["temperature", "energy_density", "vx", "vy", "vz"])
    fd.set_hydro_status_finished()
    fd.find_freezeout_surface(T_SW, dtau, dx, deta)
    return fd.get_surface_cells()


@pytest.mark.parametrize("v", [
    (0.0, 0.0, 0.0),
    (0.3, 0.0, 0.0),
    (0.5, 0.5, 0.0),            # |v| = 0.71
    (0.0, 0.0, 0.7),            # longitudinal: exercises the Milne boost
    (0.5, 0.3, 0.6),            # |v| = 0.84
    (0.0, 0.0, 0.9),
])
def test_umu_is_normalised_and_boosted(v):
    cells = _surface(v)
    assert len(cells) > 0

    vx, vy, vz = v
    gamma = 1.0 / np.sqrt(1.0 - (vx * vx + vy * vy + vz * vz))
    n_eta = 0
    for c in cells:
        u = np.asarray(c.umu, dtype=np.float64)
        # SurfaceCellInfo stores umu as float, so round-off is ~1e-7 relative
        assert u[0] ** 2 - u[1] ** 2 - u[2] ** 2 - u[3] ** 2 == pytest.approx(1.0, abs=1e-5)
        ch, sh = np.cosh(c.eta), np.sinh(c.eta)
        assert u[0] == pytest.approx(gamma * (ch - vz * sh), rel=1e-5)
        assert u[1] == pytest.approx(gamma * vx, abs=1e-5)
        assert u[2] == pytest.approx(gamma * vy, abs=1e-5)
        assert u[3] == pytest.approx(gamma * (vz * ch - sh), abs=1e-5)
        n_eta += abs(c.eta) > 0.1
    assert n_eta > 0, "no cell at eta != 0; the boost was not tested"


def test_surface_sits_at_tau_sw():
    cells = _surface((0.0, 0.0, 0.0))
    taus = np.array([c.tau for c in cells])
    # T is linearly interpolated in tau between stored frames, so allow one frame
    assert np.all(np.abs(taus - TAU_SW) < 0.1)
    # flat surface: time-like normal only.  Cornelius works in float (~1e-5 relative) and
    # emits a few degenerate elements (|dsigma| ~ 1e-30 .. 1e-11) where corners sit on T_sw; skip them.
    d0_max = max(abs(c.d3sigma_mu[0]) for c in cells)
    for c in cells:
        d = np.asarray(c.d3sigma_mu, dtype=np.float64)
        if abs(d[0]) < 1e-6 * d0_max:
            continue
        assert np.allclose(d[1:], 0.0, atol=1e-4 * abs(d[0]))


def test_lattice_spacing_is_honoured():
    fine = _surface((0.0, 0.0, 0.0))                              # 0.2 fm x 0.2 in eta
    coarse = _surface((0.0, 0.0, 0.0), dtau=0.2, dx=0.5, deta=0.5)
    # a flat surface over the same area: element count scales with 1/(dx dy deta)
    assert len(coarse) < len(fine)
    # Both lattices must cover the whole grid, [-2,2]^2 x [-1,1] = 32 fm^2.  Before the
    # fix, float truncation (4 / 0.2f = 19.99999 -> 19 cells) dropped the last cell in every
    # direction and the default lattice covered only 3.8 * 3.8 * 1.8 = 26.
    area = lambda cells: sum(abs(c.d3sigma_mu[0]) for c in cells)  # noqa: E731
    assert area(fine) == pytest.approx(32.0, rel=1e-3)
    assert area(coarse) == pytest.approx(32.0, rel=1e-3)
