"""The pure-Python CausalLiquefier port against the actual C++ the run uses.

FNO4d validated the port against X-SCAPE's `examples/unittests/causal_liquifier.cc` to 4e-14.
That checked it against a *transcription* of the C++.  These call the real
`Jetscape::CausalLiquefier` through the bindings, so the reference is the code the pipeline
actually executes -- if the C++ kernel ever changes, this notices and the unit-test
transcription would not.

Tolerance: Jetscape::real is float, so anything that crosses the C++ boundary is float32.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.needs_xscape

from fast_data.liquefier import LiquefierParams          # noqa: E402
from fast_data.liquefier import kernel as K              # noqa: E402

DTAU, DX, DY, DETA = 0.02, 0.3, 0.3, 0.2


@pytest.fixture(scope="module")
def cpp():
    from jetscape.pyjetscape_core import CausalLiquefier
    return CausalLiquefier(DTAU, DX, DY, DETA)


@pytest.fixture(scope="module")
def params(cpp):
    """Read the parameters off the live C++ object rather than assuming the defaults."""
    from fasthydro.liquefier_bridge import params_from_liquefier
    return params_from_liquefier(cpp)


def test_derived_constants_agree(cpp, params):
    assert params.c_diff == pytest.approx(cpp.c_diff, rel=1e-12)
    assert params.gamma_relax == pytest.approx(cpp.gamma_relax, rel=1e-12)


@pytest.mark.parametrize("t", [0.05, 0.2, 0.5, 1.0, 2.0])
@pytest.mark.parametrize("r", [0.01, 0.1, 0.3, 0.7, 1.5])
def test_smooth_kernels_agree(cpp, params, t, r):
    """rho_smooth / j_smooth are the Bessel part -- scipy's ive against GSL's I1/I2."""
    assert K.rho_smooth(t, r, params) == pytest.approx(cpp.rho_smooth(t, r), rel=1e-6, abs=1e-12)
    assert K.j_smooth(t, r, params) == pytest.approx(cpp.j_smooth(t, r), rel=1e-6, abs=1e-12)


@pytest.mark.parametrize("t,r", [(0.05, 0.02), (0.2, 0.15), (0.5, 0.4),
                                 (1.0, 0.85), (2.0, 1.7)])
def test_full_kernels_agree_in_strict_mode(cpp, params, t, r):
    """`strict_xscape=True` is the mode that reproduces the C++ delta-shell term exactly."""
    got_rho = K.kernel_rho(t, r, params, strict_xscape=True)
    got_j = K.kernel_j(t, r, params, strict_xscape=True)
    assert got_rho == pytest.approx(cpp.kernel_rho(t, r), rel=1e-6, abs=1e-12)
    assert got_j == pytest.approx(cpp.kernel_j(t, r), rel=1e-6, abs=1e-12)


def test_milne_rotations_agree(cpp, params):
    rng = np.random.default_rng(0)
    for _ in range(50):
        px, pz, eta = rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-3, 3)
        assert cpp.get_ptau(px, pz, eta) == pytest.approx(
            px * np.cosh(eta) - pz * np.sinh(eta), rel=1e-6, abs=1e-9)
        assert cpp.get_peta(px, pz, eta) == pytest.approx(
            pz * np.cosh(eta) - px * np.sinh(eta), rel=1e-6, abs=1e-9)


def test_get_source_matches_the_python_port(cpp, params):
    """The whole j^mu, on the C++ side vs `get_source_xscape` -- window, causality cut and
    Milne rotation together, not just the radial kernel."""
    drop = np.array([[1.0, 0.0, 0.0, 0.0, 10.0, 1.0, -2.0, 0.5]])
    cpp.ClearTask()
    cpp.add_droplets_numpy(drop)

    tau = 1.0 + params.tau_delay          # the firing window
    rng = np.random.default_rng(1)
    pts = np.column_stack([rng.uniform(-1.0, 1.0, 40),
                           rng.uniform(-1.0, 1.0, 40),
                           rng.uniform(-0.6, 0.6, 40)])
    n_nonzero = 0
    for x, y, eta in pts:
        c = np.asarray(cpp.get_source(tau, x, y, eta), dtype=np.float64)
        p = np.asarray(K.get_source_xscape(tau, x, y, eta, drop, params,
                                           emulate_float32=True), dtype=np.float64).ravel()
        scale = max(1e-9, np.abs(c).max())
        assert np.max(np.abs(c - p)) / scale < 2e-5, (x, y, eta, c, p)
        n_nonzero += int(np.abs(c).max() > 0)
    assert n_nonzero > 0, "every sample point was outside the causal support; test is vacuous"
    cpp.ClearTask()


def test_the_window_is_half_open(cpp, params):
    """A droplet fires in exactly one hydro step -- otherwise it deposits twice or not at all."""
    cpp.ClearTask()
    cpp.add_droplets_numpy(np.array([[1.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0]]))
    dep = 1.0 + params.tau_delay
    fired = [np.abs(np.asarray(cpp.get_source(dep + k * DTAU, 0.05, 0.0, 0.0))).max() > 0
             for k in (-2, -1, 0, 1, 2)]
    assert sum(fired) == 1, f"deposited in {sum(fired)} steps, expected exactly 1"
    cpp.ClearTask()
