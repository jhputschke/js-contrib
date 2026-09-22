"""Gates for the conserved-state -> training-channel conversion and the freeze-out policy.

The lab-rest test is the sharp one.  The solver carries the CONTRAVARIANT u^eta while the
training data stores Cartesian lab velocities built from the ORTHONORMAL tau*u^eta.  Every way of
getting that wrong -- a missing tau, a swapped cosh/sinh, a covariant/contravariant mix-up --
still yields finite, smooth, |v| < 1 data.  Only a case with a known exact answer catches it, and
a fluid at rest in the lab at eta != 0 is that case: it must give v = 0 identically.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

torch = pytest.importorskip("torch")

from fast_data import convert, fv                       # noqa: E402


def _grid(nz=17, deta=0.5):
    return fv.Grid(6, 6, nz, 0.5, 0.5, deta, dtype=torch.float64)


def test_c1_fluid_at_rest_in_the_lab_gives_exactly_zero_velocity():
    """A fluid at rest in the lab has Cartesian u = (1,0,0,0).  Transforming with
    tau = sqrt(t^2-z^2), eta = artanh(z/t) gives u^tau = cosh(eta) and u^eta = -sinh(eta)/tau
    (note the sign), which normalises as -u_tau^2 + tau^2 u_eta^2 = -1.  Its lab velocity is
    zero by construction, so the conversion must return exactly zero."""
    g = _grid(nz=21, deta=0.4)          # eta from -4 to +4; see test_c1b for why not +/-5
    eos = fv.ConformalEoS(47.5)
    tau = 1.7
    eta = g.eta.view(1, 1, 1, -1)
    u = torch.zeros(1, 4, g.nx, g.ny, g.neta, dtype=torch.float64)
    u[:, 0] = torch.cosh(eta)
    u[:, 3] = -torch.sinh(eta) / tau      # note the sign; derived in the docstring above
    e = torch.full((1, g.nx, g.ny, g.neta), 4.0, dtype=torch.float64)
    q = fv.conserved_from_primitives(e, u, tau, eos)
    frame, _ = convert.q_to_fno_frame(q, None, None, tau, g, eos)
    v = frame[0, 1:4].numpy()
    assert np.max(np.abs(v)) < 1e-10, f"max |v| = {np.max(np.abs(v)):.3e} for a lab-rest fluid"
    assert np.allclose(frame[0, 0].numpy(), 4.0, rtol=1e-10)


def test_c1b_the_velocity_cap_bounds_how_far_out_in_eta_this_is_exact():
    """A fluid at rest in the lab has Milne speed tanh(eta), so beyond |eta| ~ 4.6 it exceeds the
    solver's V_CAP = 0.9999 and the Landau match clamps it.  The FNO grid runs to |eta| = 5, so
    its outermost cells are inside that regime -- which is where the large |v| in dilute edge
    cells comes from.  Recorded here so the boundary is a known number, not a surprise."""
    eos = fv.ConformalEoS(47.5)
    tau = 1.7
    got = {}
    for eta_max in (2.0, 4.0, 5.0):
        g = fv.Grid(4, 4, 11, 0.5, 0.5, 2 * eta_max / 10, dtype=torch.float64)
        eta = g.eta.view(1, 1, 1, -1)
        u = torch.zeros(1, 4, g.nx, g.ny, g.neta, dtype=torch.float64)
        u[:, 0] = torch.cosh(eta)
        u[:, 3] = -torch.sinh(eta) / tau
        q = fv.conserved_from_primitives(
            torch.full((1, 4, 4, 11), 4.0, dtype=torch.float64), u, tau, eos)
        frame, _ = convert.q_to_fno_frame(q, None, None, tau, g, eos)
        got[eta_max] = float(np.abs(frame[0, 1:4].numpy()).max())
    assert got[2.0] < 1e-13 and got[4.0] < 1e-10          # exact where tanh(eta) < V_CAP
    # Beyond the cap exactness is simply lost.  Assert that, not the size of the artefact:
    # how far off it lands depends on the Landau clamp and is not a well-defined number.
    assert got[5.0] > 1e-6
    assert got[5.0] > 1e4 * got[4.0]
    assert np.tanh(5.0) > fv.V_CAP > np.tanh(4.5)


def test_c2_bjorken_gives_vz_equal_tanh_eta():
    """u^eta = 0 is Bjorken flow: the lab velocity is purely longitudinal, vz = tanh(eta)."""
    g = _grid(nz=21, deta=0.5)
    eos = fv.ConformalEoS(47.5)
    tau = 2.3
    u = torch.zeros(1, 4, g.nx, g.ny, g.neta, dtype=torch.float64)
    u[:, 0] = 1.0
    e = torch.full((1, g.nx, g.ny, g.neta), 3.0, dtype=torch.float64)
    q = fv.conserved_from_primitives(e, u, tau, eos)
    frame, _ = convert.q_to_fno_frame(q, None, None, tau, g, eos)
    want = np.tanh(g.eta.numpy()).reshape(1, 1, -1)
    assert np.allclose(frame[0, 3].numpy(), want, atol=1e-12)
    assert np.max(np.abs(frame[0, 1:3].numpy())) < 1e-12


def test_c3_transverse_boost_is_recovered():
    g = _grid()
    eos = fv.ConformalEoS(47.5)
    tau, vx = 1.0, 0.5
    gam = 1.0 / np.sqrt(1 - vx ** 2)
    u = torch.zeros(1, 4, g.nx, g.ny, g.neta, dtype=torch.float64)
    u[:, 0], u[:, 1] = gam, gam * vx
    e = torch.full((1, g.nx, g.ny, g.neta), 5.0, dtype=torch.float64)
    q = fv.conserved_from_primitives(e, u, tau, eos)
    frame, _ = convert.q_to_fno_frame(q, None, None, tau, g, eos)
    mid = g.neta // 2
    assert frame[0, 1, :, :, mid].numpy() == pytest.approx(vx, abs=1e-10)


def test_c4_u_tau_is_the_orthonormal_norm():
    """prim['u_tau'] must equal sqrt(1 + ux^2 + uy^2 + (tau u^eta)^2): the orthonormal
    normalisation, which is what justifies using it directly in the conversion."""
    g = _grid()
    eos = fv.ConformalEoS(47.5)
    tau = 1.4
    torch.manual_seed(0)
    u = torch.zeros(1, 4, g.nx, g.ny, g.neta, dtype=torch.float64)
    u[:, 1] = torch.rand_like(u[:, 1]) * 0.4
    u[:, 2] = torch.rand_like(u[:, 2]) * 0.4
    u[:, 3] = (torch.rand_like(u[:, 3]) - 0.5) * 0.4 / tau
    u[:, 0] = torch.sqrt(1 + u[:, 1] ** 2 + u[:, 2] ** 2 + (tau * u[:, 3]) ** 2)
    e = torch.full((1, g.nx, g.ny, g.neta), 6.0, dtype=torch.float64)
    q = fv.conserved_from_primitives(e, u, tau, eos)
    prim = fv.primitive_recovery(q, None, None, tau, eos)
    want = torch.sqrt(1 + prim["u_x"] ** 2 + prim["u_y"] ** 2 + (tau * prim["u_eta"]) ** 2)
    assert torch.allclose(prim["u_tau"], want, atol=1e-12)


@pytest.mark.parametrize("T,expect_ntau,expect_live", [
    (np.linspace(0.4, 0.05, 20), 15, 14),      # cools partway through
    (np.full(20, 0.4), 20, 20),                # never cools: nothing is dead, nothing zeroed
    (np.full(20, 0.01), 1, 0),                 # already cold at tau0
])
def test_c5_freezeout_matches_the_measured_music_convention(T, expect_ntau, expect_live):
    """Measured on all 25 events of data/dAu_25ev_mb.h5:
    ntau_freezeout == (non-zero frames) + 1, and tau_freezeout is the tau of the first zeroed
    frame.  live_tau_lengths reads channel 0, so the tail really must be zero."""
    n = convert.freezeout_index(T, 0.150)
    assert n == expect_ntau
    arr = np.ones((4, 2, 2, 2, len(T)), dtype=np.float32)
    convert.zero_after_freezeout(arr, n)
    alive = np.abs(arr[0]).max(axis=(0, 1, 2)) > 0
    live = int(np.flatnonzero(alive)[-1]) + 1 if alive.any() else 0
    assert live == expect_live


def test_c6_cartesian_P_of_source_matches_the_solver():
    """convert.cartesian_P_of_source must agree with fv.cartesian_four_momentum."""
    g = _grid()
    torch.manual_seed(1)
    S = torch.randn(1, 4, g.nx, g.ny, g.neta, dtype=torch.float64)
    tau = 1.9
    want = fv.cartesian_four_momentum(S, tau, g)[0].numpy()
    got = convert.cartesian_P_of_source(S[0].numpy(), tau, g)
    assert np.allclose(got, want, rtol=1e-12, atol=1e-12)
