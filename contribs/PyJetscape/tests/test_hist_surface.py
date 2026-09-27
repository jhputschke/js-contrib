"""
Tests for example/hydro_hist_vs_surface/make_surfaces.py: the freeze-out surface built from
a stored hydro history (e, vx, vy, vz on the output grid), as the hist variant builds it.

A synthetic evolution at rest with an energy density that falls with tau only has a flat
surface at the tau where T(e) = T_sw.  The cells must then sit on the isotherm (e and P from
the EoS at T_sw, not the separately interpolated e), carry no viscous or charge fields, and
cover the grid's area.  The low-temperature cap adds one tau = const cell per grid point with
0.05 < e < e(T_sw) at the first frame, and none when switched off.

    pytest tests/test_hist_surface.py -q
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

core = pytest.importorskip("jetscape.pyjetscape_core")
from jetscape.particlize_h5 import SURFACE_COLUMNS  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "make_surfaces", ROOT / "example" / "hydro_hist_vs_surface" / "make_surfaces.py")
ms = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ms)
ms.COL = {n: i for i, n in enumerate(SURFACE_COLUMNS)}
C = ms.COL

T_SW = 0.15


class ToyEos(ms.Eos):
    """A conformal-like table, e = a T^4, P = e/3, in MUSIC's (e, P, s, T) layout."""

    def __init__(self, a=460.0):      # e(T_sw = 0.15) = 0.233, as hotQCD
        T = np.linspace(0.01, 0.6, 3000)
        self.e, self.T = a * T ** 4, T
        self.P = self.e / 3
        self.s = (self.e + self.P) / T
        self.path = "toy"


def _history(eos, *, e_first_cold=None):
    """(4, nx, ny, neta, ntau) in Bjorken flow (v_z = z/t = tanh eta, so u = (1, 0, 0, 0) in
    Milne components); T = 0.3 (0.6/tau)^(1/3) crosses T_sw at tau 4.8."""
    nx = ny = 9
    neta, ntau = 9, 61
    tau = 0.6 + 0.1 * np.arange(ntau)
    T = 0.30 * (0.6 / tau) ** (1 / 3)
    e = np.broadcast_to(np.interp(T, eos.T, eos.e), (nx, ny, neta, ntau)).copy()
    if e_first_cold is not None:          # one cold corner cell at the first frame
        e[0, 0, 0, 0] = e_first_cold
    arr = np.zeros((4, nx, ny, neta, ntau), dtype=np.float32)
    arr[0] = e
    eta = -1.0 + 0.25 * np.arange(neta)
    arr[3] = np.tanh(eta)[None, None, :, None]
    grid = {"tau_min": 0.6, "dtau": 0.1, "x_min": -2.0, "dx": 0.5, "y_min": -2.0, "dy": 0.5,
            "eta_min": -1.0, "deta": 0.25}
    return arr, grid


def test_flat_surface_on_the_isotherm():
    eos = ToyEos()
    arr, grid = _history(eos)
    cells, diag = ms.hist_surface(core, arr, arr.shape[-1], grid, eos, T_SW,
                                  (0.1, 0.5, 0.25), cap=False)
    assert len(cells) > 0 and diag["hist_n_cap"] == 0
    assert np.all(np.abs(cells[:, C["tau"]] - 4.8) < 0.1)
    np.testing.assert_allclose(cells[:, C["T"]], T_SW, rtol=1e-3)
    e_sw = float(eos.e_of_T(T_SW))
    np.testing.assert_allclose(cells[:, C["e"]], e_sw, rtol=5e-3)
    np.testing.assert_allclose(cells[:, C["P"]], e_sw / 3, rtol=5e-3)
    for name in ms.VISCOUS + ms.CHARGES:
        assert np.all(cells[:, C[name]] == 0), name
    # Bjorken flow: u = (1, 0, 0, 0) in Milne components; the area covers [-2, 2]^2 x [-1, 1]
    # (v_z = tanh eta is interpolated linearly between the eta grid points: ~6e-5 off)
    np.testing.assert_allclose(cells[:, C["u0"]], 1, atol=2e-4)
    d0 = cells[:, C["ds0"]].astype(float)
    assert d0.sum() == pytest.approx(32.0, rel=1e-3)


def test_first_frame_cap():
    eos = ToyEos()
    arr, grid = _history(eos, e_first_cold=0.1)          # 0.05 < 0.1 < e(T_sw)
    cells, diag = ms.hist_surface(core, arr, arr.shape[-1], grid, eos, T_SW,
                                  (0.1, 0.5, 0.25), cap=True)
    assert diag["hist_n_cap"] == 1
    cap = cells[np.isclose(cells[:, C["tau"]], grid["tau_min"])]
    cap = cap[np.isclose(cap[:, C["ds0"]], 0.5 * 0.5 * 0.25)]
    assert len(cap) == 1
    c = cap[0]
    assert (c[C["x"]], c[C["y"]], c[C["eta"]]) == pytest.approx((-2.0, -2.0, -1.0))
    assert c[C["e"]] == pytest.approx(0.1, rel=1e-6)
    assert c[C["T"]] == pytest.approx(float(eos.T_of_e(0.1)), rel=1e-4)
    assert c[C["T"]] < T_SW
    # below MUSIC's 0.05 GeV/fm^3 there is no cap cell
    arr, grid = _history(eos, e_first_cold=0.01)
    _, diag = ms.hist_surface(core, arr, arr.shape[-1], grid, eos, T_SW, (0.1, 0.5, 0.25),
                              cap=True)
    assert diag["hist_n_cap"] == 0


def test_zeroed_frames_after_freeze_out_are_dropped():
    eos = ToyEos()
    arr, grid = _history(eos)
    # the leg froze out after frame 50: MUSIC writes zeros after it; the finder must not see
    # the drop to 0 as a second surface (T at frame 50 is below T_sw already)
    arr[..., 51:] = 0
    full, _ = ms.hist_surface(core, arr, arr.shape[-1], grid, eos, T_SW, (0.1, 0.5, 0.25),
                              cap=False)
    cut, diag = ms.hist_surface(core, arr, 51, grid, eos, T_SW, (0.1, 0.5, 0.25), cap=False)
    assert diag["hist_ntau_used"] == 52
    assert len(cut) == len(full)
    np.testing.assert_array_equal(cut, full)


def test_milne_u_matches_surface_finder():
    u = ms.milne_u(np.array([0.3]), np.array([0.1]), np.array([0.5]), np.array([0.7]))[0]
    assert u[0] ** 2 - u[1] ** 2 - u[2] ** 2 - u[3] ** 2 == pytest.approx(1.0)
    g = 1 / np.sqrt(1 - 0.35)
    assert u[0] == pytest.approx(g * (np.cosh(0.7) - 0.5 * np.sinh(0.7)))
    assert u[3] == pytest.approx(g * (0.5 * np.cosh(0.7) - np.sinh(0.7)))
