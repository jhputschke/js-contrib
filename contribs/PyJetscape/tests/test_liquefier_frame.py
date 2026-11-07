"""CausalLiquefier's <frame>: the lab frame as before, and the frame of the Bjorken fluid at the
droplet's eta_s (X-SCAPE with XSCAPE_CAUSAL_LIQUEFIER_FRAME).

* fluid frame: a droplet at eta_d deposits exactly as a droplet at eta_s = 0 does, shifted
  by eta_d (the flux through the tau slice, per unit lab energy);
* both frames agree at eta_d = 0;
* the C++ kernel matches the Python port (example/prod_AuAu_0_10_jet/liquefier_kernel.py).
"""
import os
import sys

import numpy as np
import pytest

core = pytest.importorskip("jetscape.pyjetscape_core")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "example", "prod_AuAu_0_10_jet"))
import liquefier_kernel as port  # noqa: E402

TAU_D = 1.0


@pytest.fixture(scope="module")
def liq():
    l = core.CausalLiquefier(0.02, 0.3, 0.3, 0.2)       # no XML: tau_delay 2, the defaults
    if not hasattr(l, "fluid_frame"):
        pytest.skip("X-SCAPE without <CausalLiquefier><frame>")
    return l


def flux(liq, fluid, eta_d, tau, x, y, eta):
    """J^tau of a unit droplet at rest (lab energy 1), at one point of its deposit slice."""
    liq.fluid_frame = fluid
    d = core.Droplet([TAU_D, 0.03, -0.05, eta_d], [1.0, 0.0, 0.0, 0.0])
    j = liq.smearing_kernel(tau, x, y, eta, d)
    return np.cosh(eta) * j[0] + np.sinh(eta) * j[3]


def points(n=300, seed=3):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(-1.2, 1.2, n), rng.uniform(-1.2, 1.2, n),
                            rng.uniform(-0.7, 0.7, n)])


def test_fluid_frame_is_the_midrapidity_deposit_shifted(liq):
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


@pytest.mark.parametrize("fluid", [False, True])
def test_cpp_matches_the_python_port(liq, fluid):
    k = port.Kernel(liq.time_relax, liq.d_diff, liq.width_delta, liq.tau_delay)
    tau, eta_d = TAU_D + liq.tau_delay, 1.7
    grid = dict(x_min=-1.2, dx=0.3, nx=9, y_min=-1.2, dy=0.3, ny=9, eta_min=eta_d - 1.4,
                deta=0.2, neta=15)
    w = k.deposit((TAU_D, 0.03, -0.05, eta_d), grid, "fluid" if fluid else "lab", tau_q=tau)[0]
    dV = tau * grid["dx"] * grid["dy"] * grid["deta"]
    for ie in range(grid["neta"]):
        for ix in range(grid["nx"]):
            for iy in range(grid["ny"]):
                eta = grid["eta_min"] + ie * grid["deta"]
                x, y = grid["x_min"] + ix * grid["dx"], grid["y_min"] + iy * grid["dy"]
                # the C++ kernel is per unit time (divided by dtau, the step it fires in)
                c = flux(liq, fluid, eta_d, tau, x, y, eta) * dV * liq.dtau
                assert c == pytest.approx(w[ie, ix, iy], rel=2e-4, abs=1e-7)
    liq.fluid_frame = False
