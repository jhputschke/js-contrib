"""CausalLiquefier's <frame>: bjorken (the default), the frame of the Bjorken flow at the
droplet's eta_s, and lab, as before (X-SCAPE with XSCAPE_CAUSAL_LIQUEFIER_FRAME).

* bjorken frame: a droplet at eta_d deposits exactly as a droplet at eta_s = 0 does, shifted
  by eta_d (the flux through the tau slice, per unit lab energy);
* both frames agree at eta_d = 0;
* the C++ kernel matches the Python port (example/liquefier_frame_test/liquefier_kernel.py).

fluid (XSCAPE_CAUSAL_LIQUEFIER_FLUID_FRAME), the rest frame of the background fluid where the
droplet was made:
* with the fluid moving with the Bjorken flow it is the Bjorken frame; without a usable fluid
  (not looked up, or e < fluid_e_min) it is exactly the Bjorken frame;
* with transverse flow the C++ matches the port, and a fine grid catches a flux of 1;
* the droplets carry the fluid through add_droplets_numpy / droplet_fluid_numpy.
"""
import os
import sys

import numpy as np
import pytest

core = pytest.importorskip("jetscape.pyjetscape_core")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "example", "liquefier_frame_test"))
import liquefier_kernel as port  # noqa: E402

TAU_D = 1.0


@pytest.fixture(scope="module")
def liq():
    l = core.CausalLiquefier(0.02, 0.3, 0.3, 0.2)       # no XML: tau_delay 2, the defaults
    if not hasattr(l, "bjorken_frame"):
        pytest.skip("X-SCAPE without <CausalLiquefier><frame>")
    return l


def flux(liq, bjorken, eta_d, tau, x, y, eta):
    """J^tau of a unit droplet at rest (lab energy 1), at one point of its deposit slice."""
    liq.bjorken_frame = bjorken
    d = core.Droplet([TAU_D, 0.03, -0.05, eta_d], [1.0, 0.0, 0.0, 0.0])
    j = liq.smearing_kernel(tau, x, y, eta, d)
    return np.cosh(eta) * j[0] + np.sinh(eta) * j[3]


def points(n=300, seed=3):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(-1.2, 1.2, n), rng.uniform(-1.2, 1.2, n),
                            rng.uniform(-0.7, 0.7, n)])


def test_the_default_is_the_bjorken_frame():
    assert core.CausalLiquefier(0.02, 0.3, 0.3, 0.2).bjorken_frame is True


def test_bjorken_frame_is_the_midrapidity_deposit_shifted(liq):
    tau = TAU_D + liq.tau_delay
    nonzero = 0
    for x, y, deta in points():
        ref = flux(liq, False, 0.0, tau, x, y, deta)
        for eta_d in (1.0, 2.4, -3.1):
            got = flux(liq, True, eta_d, tau, x, y, eta_d + deta)
            assert got == pytest.approx(ref, rel=2e-4, abs=1e-6 * max(1.0, abs(ref)))
        nonzero += ref != 0
    assert nonzero > 20


def test_both_frames_agree_at_midrapidity(liq):
    tau = TAU_D + liq.tau_delay
    for x, y, eta in points(100, seed=5):
        assert flux(liq, True, 0.0, tau, x, y, eta) == flux(liq, False, 0.0, tau, x, y, eta)


@pytest.mark.parametrize("bjorken", [False, True])
def test_cpp_matches_the_python_port(liq, bjorken):
    k = port.Kernel(liq.time_relax, liq.d_diff, liq.width_delta, liq.tau_delay)
    tau, eta_d = TAU_D + liq.tau_delay, 1.7
    grid = dict(x_min=-1.2, dx=0.3, nx=9, y_min=-1.2, dy=0.3, ny=9, eta_min=eta_d - 1.4,
                deta=0.2, neta=15)
    w = k.deposit((TAU_D, 0.03, -0.05, eta_d), grid, "bjorken" if bjorken else "lab", tau_q=tau)[0]
    dV = tau * grid["dx"] * grid["dy"] * grid["deta"]
    for ie in range(grid["neta"]):
        for ix in range(grid["nx"]):
            for iy in range(grid["ny"]):
                eta = grid["eta_min"] + ie * grid["deta"]
                x, y = grid["x_min"] + ix * grid["dx"], grid["y_min"] + iy * grid["dy"]
                # the C++ kernel is per unit time (divided by dtau, the step it fires in)
                c = flux(liq, bjorken, eta_d, tau, x, y, eta) * dV * liq.dtau
                assert c == pytest.approx(w[ie, ix, iy], rel=2e-4, abs=1e-7)
    liq.bjorken_frame = True


# ── the fluid frame ─────────────────────────────────────────────────────────────────────
@pytest.fixture()
def fluid_liq(liq):
    if not hasattr(liq, "frame"):
        pytest.skip("X-SCAPE without <CausalLiquefier><frame>fluid</frame>")
    liq.frame = "fluid"
    yield liq
    liq.frame = "bjorken"


def droplet(eta_d, fluid=None, x=0.03, y=-0.05):
    d = core.Droplet([TAU_D, x, y, eta_d], [1.0, 0.0, 0.0, 0.0])
    if fluid is not None:
        d.set_fluid(list(fluid[:3]), fluid[3])
    return d


def flux_of(liq, d, tau, x, y, eta):
    j = liq.smearing_kernel(tau, x, y, eta, d)
    return np.cosh(eta) * j[0] + np.sinh(eta) * j[3]


def test_frame_names(liq):
    if not hasattr(liq, "frame"):
        pytest.skip("X-SCAPE without the fluid frame")
    assert core.CausalLiquefier(0.02, 0.3, 0.3, 0.2).frame == "bjorken"
    for name in ("lab", "fluid", "bjorken"):
        liq.frame = name
        assert liq.frame == name
        assert liq.bjorken_frame is (name == "bjorken")


def test_fluid_without_a_fluid_is_the_bjorken_frame(fluid_liq):
    tau = TAU_D + fluid_liq.tau_delay
    for fl in (None, (0.6, 0.0, np.tanh(1.7), 0.5 * fluid_liq.fluid_e_min)):
        d = droplet(1.7, fl)
        assert not fluid_liq.uses_fluid_velocity(d)
        for x, y, deta in points(60, seed=21):
            fluid_liq.frame = "bjorken"
            ref = flux_of(fluid_liq, d, tau, x, y, 1.7 + deta)
            fluid_liq.frame = "fluid"
            assert flux_of(fluid_liq, d, tau, x, y, 1.7 + deta) == ref


def test_fluid_with_the_bjorken_flow_is_the_bjorken_frame(fluid_liq):
    tau = TAU_D + fluid_liq.tau_delay
    for eta_d in (0.0, 1.0, 2.4):
        d = droplet(eta_d, (0.0, 0.0, np.tanh(eta_d), 1.0))
        assert fluid_liq.uses_fluid_velocity(d)
        assert np.abs(fluid_liq.kernel_frame_velocity(d)).max() < 1e-5   # float storage
        for x, y, deta in points(80, seed=23):
            fluid_liq.frame = "bjorken"
            ref = flux_of(fluid_liq, d, tau, x, y, eta_d + deta)
            fluid_liq.frame = "fluid"
            got = flux_of(fluid_liq, d, tau, x, y, eta_d + deta)
            assert got == pytest.approx(ref, rel=1e-4, abs=1e-5 * max(1.0, abs(ref)))


def test_relative_velocity_matches_the_port(fluid_liq):
    rng = np.random.default_rng(29)
    for _ in range(20):
        eta_d = rng.uniform(-3, 3)
        vT, phi = rng.uniform(0, 0.8), rng.uniform(0, 2 * np.pi)
        fl = (vT * np.cos(phi) / np.cosh(eta_d), vT * np.sin(phi) / np.cosh(eta_d),
              np.tanh(eta_d), 1.0)
        np.testing.assert_allclose(fluid_liq.kernel_frame_velocity(droplet(eta_d, fl)),
                                   port.relative_velocity(eta_d, fl), atol=2e-6)
        # transverse flow on top of the Bjorken flow: v_T relative to it, at any eta_d
        assert np.linalg.norm(port.relative_velocity(eta_d, fl)) == pytest.approx(vT, abs=1e-9)


@pytest.mark.parametrize("eta_d, fluid", [(0.0, (0.6, 0.0, 0.0, 1.0)),
                                          (1.7, (0.3 / np.cosh(1.7), -0.5 / np.cosh(1.7),
                                                 np.tanh(1.9), 2.0))])
def test_fluid_cpp_matches_the_python_port(fluid_liq, eta_d, fluid):
    k = port.Kernel(fluid_liq.time_relax, fluid_liq.d_diff, fluid_liq.width_delta,
                    fluid_liq.tau_delay)
    tau = TAU_D + fluid_liq.tau_delay
    grid = dict(x_min=-1.5, dx=0.3, nx=11, y_min=-1.5, dy=0.3, ny=11, eta_min=eta_d - 1.4,
                deta=0.2, neta=15)
    w = k.deposit((TAU_D, 0.03, -0.05, eta_d), grid, "fluid", tau_q=tau, fluid=fluid)[0]
    d = droplet(eta_d, fluid)
    dV = tau * grid["dx"] * grid["dy"] * grid["deta"]
    nz = 0
    for ie in range(grid["neta"]):
        eta = grid["eta_min"] + ie * grid["deta"]
        for ix in range(grid["nx"]):
            for iy in range(grid["ny"]):
                x, y = grid["x_min"] + ix * grid["dx"], grid["y_min"] + iy * grid["dy"]
                c = flux_of(fluid_liq, d, tau, x, y, eta) * dV * fluid_liq.dtau
                assert c == pytest.approx(w[ie, ix, iy], rel=2e-4, abs=1e-6)
                nz += c != 0
    assert nz > 20


@pytest.mark.parametrize("vT", [0.3, 0.6, 0.8])
def test_fluid_kernel_flux_is_one_on_a_fine_grid(vT):
    """The rest-frame current boosted back is conserved: its flux through the slice is 1."""
    k = port.Kernel(0.1, 0.08, 0.1, 1.0)
    eta_d = 2.4
    fl = (vT / np.cosh(eta_d), 0.0, np.tanh(eta_d), 1.0)
    fine = dict(x_min=-2.0, dx=0.02, nx=201, y_min=-2.0, dy=0.02, ny=201, eta_min=eta_d - 0.8,
                deta=0.01, neta=161)
    w, _, etas, _ = k.deposit((1.0, 0.03, -0.05, eta_d), fine, "fluid", tau_q=2.0, fluid=fl)
    assert w.sum() == pytest.approx(1.0, abs=0.005)
    # carried along the flow (x): the centroid moves in +x, eta_s stays at eta_d
    xs = fine["x_min"] + fine["dx"] * np.arange(fine["nx"])
    assert (w.sum(axis=(0, 2)) * xs).sum() / w.sum() > 0.03 + 0.5 * vT * 0.5
    assert (w.sum(axis=(1, 2)) * etas).sum() / w.sum() == pytest.approx(eta_d, abs=0.01)


def test_droplets_carry_the_fluid():
    l = core.CausalLiquefier(0.02, 0.3, 0.3, 0.2)
    if not hasattr(l, "droplet_fluid_numpy"):
        pytest.skip("X-SCAPE whose droplets carry no fluid")
    rows = np.array([[1.0, 0.1, 0.2, 0.3, 5.0, 1.0, 0.5, 0.2, 0.4, -0.1, 0.29, 3.5],
                     [2.0, -1.0, 0.0, -2.0, 1.0, 0.0, 0.0, -0.9, 0.0, 0.0, -0.96, 0.05]])
    l.add_droplets_numpy(rows)
    l.add_droplets_numpy(rows[:, :8])                  # no fluid: e = -1
    np.testing.assert_allclose(l.droplets_numpy()[:2], rows[:, :8], rtol=1e-6)
    fl = l.droplet_fluid_numpy()
    np.testing.assert_allclose(fl[:2], rows[:, 8:], rtol=1e-6)
    np.testing.assert_array_equal(fl[2:], [[0, 0, 0, -1]] * 2)
    l.ClearTask()
