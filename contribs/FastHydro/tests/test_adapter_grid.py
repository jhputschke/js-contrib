"""GridSpec: the one place fast_data's axes and X-SCAPE's are reconciled.

These are cheap and need no build, but they catch the failure mode that is hardest to notice
downstream -- a transposed or half-cell-shifted axis still produces a perfectly plausible
fireball.
"""

import numpy as np
import pytest

from fasthydro.grid import GridSpec

# deliberately asymmetric: nx != ny != neta and three different spacings, so no two axes can
# be swapped without a test failing
ASYM = dict(nx=21, ny=15, neta=9, dx=0.5, dy=0.6, deta=0.4,
            tau0=0.6, record_dtau=0.2, ntau=11)


def test_cell_centred_axes():
    g = GridSpec(**ASYM)
    assert g.x_min == pytest.approx(-0.5 * (g.nx - 1) * g.dx)
    assert g.y_min == pytest.approx(-0.5 * (g.ny - 1) * g.dy)
    assert g.eta_min == pytest.approx(-0.5 * (g.neta - 1) * g.deta)
    assert g.tau_max == pytest.approx(g.tau0 + (g.ntau - 1) * g.record_dtau)


@pytest.mark.parametrize("n,d", [(65, 0.3125), (25, 0.5), (21, 0.5), (15, 0.6),
                                 (9, 0.4), (33, 0.3125), (13, 0.4), (2, 1.0)])
def test_setranges_round_trips_to_n(n, d):
    """GetXSize() is ceil(2*grid_max/grid_step); grid_max = n*d/2 must recover n exactly."""
    g = GridSpec(n, n, n, d, d, d, 0.6, 0.1, 2)
    assert g.expected_is_sizes() == (n, n, n)


def test_setranges_round_trips_on_an_asymmetric_grid():
    g = GridSpec(**ASYM)
    assert g.expected_is_sizes() == (g.nx, g.ny, g.neta)


def test_axes_reproduce_fv_grid_exactly():
    """What EvolutionHistory reconstructs (eta_min + i*deta) must be fv.Grid's own axis."""
    pytest.importorskip("torch")
    g = GridSpec(**ASYM)
    fvg = g.to_fv_grid()
    for lo, d, n, ax in ((g.x_min, g.dx, g.nx, fvg.x),
                         (g.y_min, g.dy, g.ny, fvg.y),
                         (g.eta_min, g.deta, g.neta, fvg.eta)):
        assert np.max(np.abs((lo + d * np.arange(n)) - ax.numpy())) < 1e-12


def test_from_cfg_derives_ntau_both_ways():
    base = {"grid": dict(nx=8, ny=8, neta=4, dx=0.5, dy=0.5, deta=0.5)}
    a = GridSpec.from_cfg({**base, "time": dict(tau0=0.6, record_dtau=0.2, choose_ntau=11)})
    b = GridSpec.from_cfg({**base, "time": dict(tau0=0.6, record_dtau=0.2,
                                                choose_ntau=None, tau_end=2.6)})
    assert a.ntau == b.ntau == 11
    assert a.tau_max == pytest.approx(2.6)


def test_tau_grid_is_exact_and_uniform():
    g = GridSpec(**ASYM)
    t = g.tau_grid()
    assert len(t) == g.ntau
    assert t[0] == pytest.approx(g.tau0)
    # tau0 + k*dtau accumulates a ULP or two; 1e-12 fm/c is far below anything physical
    assert np.allclose(np.diff(t), g.record_dtau, atol=1e-12, rtol=0)
