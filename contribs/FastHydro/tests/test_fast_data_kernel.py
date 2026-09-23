"""Gates for the CausalLiquefier kernel port (loc_libs/fast_data/liquefier/kernel.py).

The reference is X-SCAPE's own unit test, examples/unittests/causal_liquifier.cc, which prints
no numbers -- its assertions ARE the targets.  Needs no data files and runs in a few seconds.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from fast_data.liquefier import kernel as K          # noqa: E402
from fast_data.liquefier import LiquefierParams      # noqa: E402

P = LiquefierParams()                                # the XML defaults


def test_k1_conservation_reproduces_cpp_unit_test():
    """TEST_CONSERVATION (causal_liquifier.cc:80-94), verbatim: the radial integral of the
    kernel is 1 for every t, to 5%.  The midpoint rule starting at r = 0.5*dr matters at that
    level, so it is reproduced exactly."""
    dr = 0.005
    worst = 0.0
    for t in np.arange(0.3, 3.0, 0.005):
        r = np.arange(0.5 * dr, 1.5 * t, dr)
        integral = np.sum(4.0 * np.pi * r * r * dr * K.kernel_rho(t, r, P))
        worst = max(worst, abs(integral - 1.0))
    assert worst < 0.05, f"C++ gate: {worst}"
    assert worst < 0.01, f"this port should do much better than the C++ tolerance: {worst}"


def test_k2_shell_mass_is_analytic_and_width_independent():
    """The delta shell carries (1 + gt + (gt)^2/2) e^{-gt} whatever its width -- which is what
    makes widening it (to the local node spacing) mass-preserving, and what lets the c*t <=
    width_delta case be regularised into a uniform ball without changing the physics."""
    for t in (0.2, 0.5, 1.0, 2.0, 3.0):
        want = float(K.shell_mass_fraction(t, P))
        for r_w in (0.05, 0.1, 0.2):
            R = P.c_diff * t
            r = np.linspace(max(R - r_w, 0.0), R * (1 - 1e-12), 200001)
            got = np.trapezoid(4 * np.pi * r * r * K.rho_delta(t, r, P, r_w=r_w), r)
            assert got == pytest.approx(want, rel=1e-4), f"t={t}, r_w={r_w}"


def test_k3_finite_at_the_hazards_the_cpp_leaves_open():
    """r -> 0 and c*t <= width_delta, where the C++'s unguarded 1/r^2 returns inf."""
    for t in (1e-6, 0.01, 0.05, 0.3, 1.0, 5.0):
        r = np.concatenate([[0.0, 1e-300, 1e-12], np.linspace(0, P.c_diff * t * 1.5, 2000)])
        for fn in (K.kernel_rho, K.kernel_j):
            v = fn(t, r, P)
            assert np.isfinite(v).all(), f"{fn.__name__} not finite at t={t}"
            assert (v >= 0).all(), f"{fn.__name__} went negative at t={t}"


def test_k3b_strict_xscape_keeps_the_cpp_divergence():
    """The regression path must reproduce the C++ exactly, inf included."""
    assert np.isinf(K.rho_delta(0.05, 0.0, P, strict_xscape=True))


def test_k4_causality_is_strict():
    """TEST_CAUSALITY: exactly zero AT the front as well as outside it (the C++ uses r < c*t)."""
    for scale in (1, 2, 3, 4):
        assert float(K.kernel_rho(2.0, P.c_diff * 2.0 * scale, P)) == 0.0
        assert float(K.kernel_j(2.0, P.c_diff * 2.0 * scale, P)) == 0.0


def test_k5_matches_the_literal_cpp_expression():
    """The A/B reformulation removes the divisions by u algebraically, so it must agree with a
    direct evaluation of the C++ formula wherever that formula is numerically sound."""
    from scipy.special import iv
    c, g = P.c_diff, P.gamma_relax
    for t in (0.05, 0.2, 0.5, 1.0, 1.3, 2.0, 3.0, 6.0):
        r = np.linspace(0, c * t * (1 - 1e-9), 3000)[1:]
        u = np.sqrt(c * c * t * t - r * r)
        x = g * u / c
        cpp = (g * g / c) * (iv(1, x) / (c * u) + iv(2, x) * t / u ** 2) * np.exp(-g * t) / (4 * np.pi)
        ours = K.kernel_rho(t, r, P) - K.rho_delta(t, r, P)
        assert np.max(np.abs(ours - cpp) / np.abs(cpp)) < 1e-12, f"t={t}"


def test_k6_light_front_limit_is_the_analytic_one():
    """As u -> 0 both smooth terms are 0/0 in the C++ form; the limits are finite."""
    c, g = P.c_diff, P.gamma_relax
    t = 1.3
    want = (g ** 3 / c ** 3) * (0.5 + g * t / 8.0) * np.exp(-g * t) / (4 * np.pi)
    r = c * t * (1 - 1e-15)
    got = float(K.kernel_rho(t, r, P) - K.rho_delta(t, r, P))
    assert got == pytest.approx(want, rel=1e-6)


def test_k7_bessel_helpers_match_scipy_and_are_continuous():
    """A(x) = I1(x)/x and B(x) = I2(x)/x^2, across the series/ive branch threshold."""
    from scipy.special import iv
    xs = np.array([1e-6, 1e-4, 5e-4, 9.9e-4, 1e-3, 1.001e-3, 2e-3, 1e-2, 0.1, 1.0, 10.0, 60.0])
    assert np.allclose(K.bessel_A(xs), iv(1, xs) / xs, rtol=1e-13)
    assert np.allclose(K.bessel_B(xs), iv(2, xs) / xs ** 2, rtol=1e-13)
    assert float(K.bessel_A(0.0)) == pytest.approx(0.5, rel=1e-15)
    assert float(K.bessel_B(0.0)) == pytest.approx(0.125, rel=1e-15)
    x0 = 1e-3
    assert K.bessel_A(x0 * (1 + 1e-9)) == pytest.approx(K.bessel_A(x0 * (1 - 1e-9)), rel=1e-12)


def test_k8_scaled_bessel_cannot_overflow():
    """u <= c*t implies x <= gamma*t, so exp(x - gamma*t) <= 1 -- the property that makes the
    scaled form safe where the C++'s unscaled gsl_sf_bessel_In overflows."""
    c, g = P.c_diff, P.gamma_relax
    worst = 0.0
    for t in np.linspace(0.01, 20.0, 300):
        r = np.linspace(0, c * t * (1 - 1e-12), 400)
        u = np.sqrt(np.maximum(c * c * t * t - r * r, 0.0))
        worst = max(worst, float(np.max(np.exp(g * u / c - g * t))))
    assert worst <= 1.0 + 1e-12


def test_k9_window_and_source_zero_outside_the_window():
    """smearing_kernel's half-open window is the only 'when' test in the C++."""
    drop = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    tau_dep = 1.0 + P.tau_delay
    assert np.all(K.smearing_kernel_jmu(0.99 * (tau_dep - 0.5 * P.dtau), 0.0, 0.0, 0.0, drop, P) == 0)
    assert np.all(K.smearing_kernel_jmu(1.01 * (tau_dep + 0.5 * P.dtau), 0.0, 0.0, 0.0, drop, P) == 0)
    assert np.any(K.smearing_kernel_jmu(tau_dep, 0.3, 0.0, 0.0, drop, P) != 0)


def test_k10_derived_constants_match_the_xml_defaults():
    assert P.c_diff == pytest.approx(np.sqrt(0.8), rel=1e-15)
    assert P.gamma_relax == pytest.approx(5.0, rel=1e-15)
    with pytest.raises(ValueError):
        LiquefierParams(d_diff=1.0, time_relax=0.1)          # superluminal
