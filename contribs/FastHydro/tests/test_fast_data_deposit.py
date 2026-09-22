"""Gates for the grid deposition and the Milne basis conversion.

The exact-conservation test (D2) is the real check of the eta basis: the liquefier returns the
eta component in the ORTHONORMAL Milne tetrad while the solver's q is CONTRAVARIANT, and a
missing factor of tau there produces data that looks perfectly reasonable and is wrong.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

torch = pytest.importorskip("torch")

from fast_data import fv                                              # noqa: E402
from fast_data.liquefier import (CausalLiquefierSource, DropletFlags,  # noqa: E402
                                 LiquefierParams, deposit, droplet_weights,
                                 milne_dq_from_cartesian, support_box)

P = LiquefierParams()


def _grid(n=41, nz=33, d=0.3125):
    return fv.Grid(n, n, nz, d, d, d, dtype=torch.float64)


def test_d1_milne_round_trip_reproduces_cpp_unit_test():
    """TEST_GRID_TAU_ETA_CONSERVATION (causal_liquifier.cc:141-211): a droplet at Milne
    (1,0,0,1) with Cartesian p = (1,1,1,1), recovered on a fine grid with the measure
    dvolume = tau*dx*dy*deta and an inverse boost carrying NO factor of tau -- which is only
    consistent with the eta component being orthonormal."""
    from fast_data.liquefier import kernel as K
    p = LiquefierParams(dtau=0.1)
    drop = np.array([1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    dx = 0.05
    n_xy, n_eta = 160, 100
    tau = 1.0 + p.tau_delay
    x = (np.arange(n_xy) - 0.5 * n_xy) * dx
    e = (np.arange(n_eta) - 0.5 * n_eta) * dx
    X, Y, E = np.meshgrid(x, x, e, indexing="ij")
    jmu = K.smearing_kernel_jmu(tau, X, Y, E, drop, p)
    dvol = tau * dx * dx * dx
    ch, sh = np.cosh(E), np.sinh(E)
    P4 = np.array([
        np.sum(p.dtau * (jmu[..., 0] * ch + jmu[..., 3] * sh) * dvol),
        np.sum(p.dtau * jmu[..., 1] * dvol),
        np.sum(p.dtau * jmu[..., 2] * dvol),
        np.sum(p.dtau * (jmu[..., 0] * sh + jmu[..., 3] * ch) * dvol),
    ])
    assert P4[1] == pytest.approx(1.0, abs=0.05), "the C++ gate is 5% on the px component"
    assert np.allclose(P4, [1.0, 1.0, 1.0, 1.0], atol=0.05), P4


def test_d2_exact_cartesian_conservation_including_holes_and_forward_droplets():
    """The whole point of the scalar renormalisation: one number fixes all four components, so
    the four-momentum the grid receives is exact regardless of how well the shape is resolved.
    Checked through the SOLVER's own cartesian_four_momentum, not our own algebra."""
    g = _grid()
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(40):
        E = rng.uniform(-20, 20)                       # negative E = a recoil hole: must work
        th, ph = rng.uniform(0, np.pi), rng.uniform(0, 2 * np.pi)
        pm = abs(E) * rng.uniform(0.2, 1.0)
        drop = np.array([rng.uniform(0.3, 3.0), rng.uniform(-3, 3), rng.uniform(-3, 3),
                         rng.uniform(-5, 5), E,
                         pm * np.sin(th) * np.cos(ph), pm * np.sin(th) * np.sin(ph),
                         pm * np.cos(th)])
        tau_q = drop[0] + P.tau_delay
        patch = droplet_weights(drop, tau_q, g, P)
        if patch.is_empty:
            continue
        q = torch.zeros(1, 4, g.nx, g.ny, g.neta, dtype=torch.float64)
        q[0, :, patch.ix, patch.iy, patch.ie] = torch.as_tensor(
            milne_dq_from_cartesian(patch, drop[4:8], tau_q, g))
        got = fv.cartesian_four_momentum(q, tau_q, g)[0].numpy()
        want = drop[4:8] * patch.in_grid_fraction
        worst = max(worst, np.max(np.abs(got - want)) / max(np.max(np.abs(drop[4:8])), 1e-12))
    assert worst < 1e-12, f"worst relative error {worst:.3e}"


def test_d3_weights_sum_to_one_after_renormalisation():
    g = _grid()
    for eta_d in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0):
        patch = droplet_weights(np.array([1., 0., 0., eta_d, 1., 0., 0., 0.]), 3.0, g, P)
        if not patch.is_empty:
            assert patch.w.sum() == pytest.approx(1.0, abs=1e-12), f"eta_d={eta_d}"


def test_d4_point_sampling_is_not_conservative_on_this_grid():
    """Documents WHY conservative mode exists: the C++ cell-centre sampling, which is fine on
    MUSIC's fine grid, loses or multiplies the deposit on the FNO grid."""
    g = _grid(65, 33)
    got = {}
    for eta_d in (0.0, 2.0, 3.0):
        patch = droplet_weights(np.array([1., 0., 0., eta_d, 1., 0., 0., 0.]), 3.0, g, P,
                                mode="xscape")
        got[eta_d] = patch.n_raw
    assert got[0.0] == pytest.approx(1.0, abs=0.01)      # fine near midrapidity
    assert got[2.0] > 1.5                                 # badly over-counted
    assert got[3.0] == pytest.approx(0.0, abs=1e-9)       # lost entirely


def test_d5_conservative_mode_resolves_what_point_sampling_cannot():
    g = _grid(65, 33)
    for eta_d in (0.0, 1.0, 2.0, 3.0):
        patch = droplet_weights(np.array([1., 0., 0., eta_d, 1., 0., 0., 0.]), 3.0, g, P)
        assert abs(patch.n_raw - 1.0) < 0.05, f"eta_d={eta_d}: n_raw={patch.n_raw}"


def test_d6_unresolved_deposits_are_flagged_not_hidden():
    """A forward droplet whose support is a fraction of a cell still conserves exactly, but the
    shape is not trustworthy -- so it must carry a flag."""
    g = _grid(65, 33)
    patch = droplet_weights(np.array([1., 0., 0., 4.0, 1., 0., 0., 0.]), 3.0, g, P)
    assert patch.flags & (DropletFlags.UNDER_RESOLVED | DropletFlags.SUBCELL
                          | DropletFlags.SHELL_DOMINATED)
    assert patch.w.sum() == pytest.approx(1.0, abs=1e-12)     # still exactly conservative


def test_d7_in_grid_fraction_reports_what_falls_off_the_edge():
    g = _grid(21, 17, 0.3125)
    inside = droplet_weights(np.array([1., 0., 0., 0., 1., 0., 0., 0.]), 2.0, g, P)
    assert inside.in_grid_fraction == pytest.approx(1.0, abs=1e-6)
    edge = droplet_weights(np.array([1., 3.0, 0., 0., 1., 0., 0., 0.]), 2.0, g, P)
    assert edge.in_grid_fraction < 1.0
    assert edge.flags & DropletFlags.OUT_OF_GRID


def test_d8_support_box_is_a_true_superset():
    g = _grid()
    from fast_data.liquefier import kernel as K
    drop = np.array([1.0, 0.3, -0.2, 0.7, 1.0, 0.0, 0.0, 0.0])
    lo, hi, rmax, ok = support_box(drop, 2.4, g, P)
    assert ok
    rng = np.random.default_rng(0)
    eta = rng.uniform(hi + 0.05, hi + 2.0, 500)
    kt = K.k_tau(2.4, drop[1], drop[2], eta, drop, P)
    assert np.all(kt == 0.0), "kernel is non-zero outside the claimed support"


def test_d9_milne_conversion_agrees_with_the_solvers_own_helper():
    """An independent check of the eta basis conversion.

    test_d2 validates it through fv.cartesian_four_momentum; this pins it directly against
    fv.cartesian_vector_to_milne_q, the solver's own already-validated Cartesian -> Milne
    routine (which does J^eta = (cosh J^z - sinh J^t)/tau and divides by dV).  Two independent
    routes to the same numbers means a sign or a factor of tau cannot hide in either.
    """
    g = _grid(21, 17)
    drop = np.array([1.2, 0.4, -0.3, 0.6, 7.0, 2.0, -1.5, 3.0])
    tau_q = drop[0] + P.tau_delay
    patch = droplet_weights(drop, tau_q, g, P)
    assert not patch.is_empty

    mine = milne_dq_from_cartesian(patch, drop[4:8], tau_q, g)

    # the solver's helper wants the normalised kernel K on the full grid, with sum K dV = 1
    K = torch.zeros(1, 1, g.nx, g.ny, g.neta, dtype=torch.float64)
    dV = float(g.dx) * float(g.dy) * float(g.deta)
    K[0, 0, patch.ix, patch.iy, patch.ie] = torch.as_tensor(patch.w) / dV
    assert float(K.sum()) * dV == pytest.approx(1.0, abs=1e-12)
    dP = torch.as_tensor(drop[4:8], dtype=torch.float64).view(1, 1, 4)
    theirs = fv.cartesian_vector_to_milne_q(dP, K, tau_q, g)

    got = torch.zeros_like(theirs)
    got[0, :, patch.ix, patch.iy, patch.ie] = torch.as_tensor(mine)
    assert torch.allclose(got, theirs, atol=1e-13, rtol=0), \
        f"max |diff| = {float((got - theirs).abs().max()):.3e}"


def test_d10_emulate_float32_reproduces_the_cpp_precision_boundary():
    """Jetscape::real is float, so the C++ rounds the droplet on ingest and jmu on egress while
    computing in double between.  The flag exists for bit-comparable regression, so it has to
    actually change the result by about float32 epsilon and nothing more."""
    from fast_data.liquefier import kernel as K
    drop = np.array([1.0, 0.2, -0.1, 0.5, 3.0, 1.0, 0.5, -2.0])
    tau = drop[0] + P.tau_delay
    rng = np.random.default_rng(3)
    x, y = rng.uniform(-2, 2, 400), rng.uniform(-2, 2, 400)
    eta = rng.uniform(-1.5, 1.5, 400)
    f64 = K.smearing_kernel_jmu(tau, x, y, eta, drop, P, emulate_float32=False)
    f32 = K.smearing_kernel_jmu(tau, x, y, eta, drop, P, emulate_float32=True)
    nz = np.abs(f64) > 0
    assert nz.any(), "the test points missed the deposit entirely"
    rel = np.abs(f32[nz] - f64[nz]) / np.abs(f64[nz])
    assert rel.max() < 2e-6, f"float32 emulation differs by {rel.max():.2e}"
    # and it must not move the support: zeros stay zeros
    assert np.array_equal(f64 == 0, f32 == 0)
