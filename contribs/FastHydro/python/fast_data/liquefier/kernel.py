"""Green's function of causal (telegraph) diffusion — a pure-numpy port of X-SCAPE's CausalLiquefier.

Reference: `X-SCAPE/src/liquefier/CausalLiquefier.cc:104-212`.  With
``u = sqrt(c^2 t^2 - r^2)`` and ``x = gamma*u/c`` the C++ computes

    dumping(t) = exp(-gamma*t)/(4*pi)
    rho_smooth = (gamma^2/c) * [ I1(x)/(c*u) + I2(x)*t/u^2 ]      (r < c*t, else 0)
    j_smooth   = (gamma^2/c) * [ I2(x)*r/u^2 ]                    (r < c*t, else 0)
    rho_delta  = (1 + gamma*t + (gamma*t)^2/2) / (r_w * r^2)      on [c*t - r_w, c*t)
    j_delta    = c_diff * rho_delta
    kernel_rho = dumping * (rho_smooth + rho_delta)
    kernel_j   = dumping * (j_smooth   + j_delta)

and normalises so that ``\\int d^3r kernel_rho = 1``.

This port is algebraically identical but removes three numerical hazards the C++ leaves open.

1.  **Division by `u` at the light front.**  Both smooth terms look singular as ``r -> c*t``,
    though their limits are finite.  Substituting ``x = gamma*u/c`` (so ``x/u = gamma/c``)
    eliminates `u` from the denominators *exactly*:

        A(x) = I1(x)/x ,  B(x) = I2(x)/x^2
        rho_smooth = (gamma^3/c^3) * [ A(x) + gamma*t*B(x) ]
        j_smooth   = (gamma^4/c^3) * r * B(x)

    `A` and `B` are entire functions with A(0) = 1/2 and B(0) = 1/8, so there is nothing to guard.

2.  **Bessel overflow.**  The C++ calls the *unscaled* `gsl_sf_bessel_I1/In`, which overflow for
    large argument, and applies `exp(-gamma*t)` separately.  Because ``u <= c*t`` implies
    ``x <= gamma*t``, the combination ``exp(x - gamma*t) <= 1`` always, so the scaled
    ``ive(n,x) = I_n(x)*exp(-x)`` gives the same number and cannot overflow.

3.  **The unguarded 1/r^2 in the delta shell.**  When ``c*t <= width_delta`` the C++ clamps
    ``r_w = c*t``, the "shell" fills the whole ball, and ``r -> 0`` returns inf.  The shell's total
    mass is analytic and independent of `r_w`,

        f_delta(t) = (1 + gamma*t + (gamma*t)^2/2) * exp(-gamma*t),

    so that branch is replaced by a uniform ball of *the same mass*, which is finite everywhere.
    Pass ``strict_xscape=True`` to restore the raw 1/r^2 (and its inf at r = 0) for regression.

Everything is float64, broadcasting numpy.  There is deliberately no torch mirror: the deposit
weights depend only on the droplet, the parameters and the grid -- never on the fluid state -- so
they are computed once per droplet here and reused as a dense array by the solver-facing code.
That also sidesteps torch having no `iv` and no `i0e` on MPS.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ive

from .params import LiquefierParams

__all__ = ["bessel_A", "bessel_B", "dumping", "shell_mass_fraction",
           "rho_smooth", "j_smooth", "rho_delta", "j_delta",
           "kernel_rho", "kernel_j", "k_tau", "smearing_kernel_jmu", "get_source_xscape"]

#: Below this the ive() form loses relative precision through the 1/x^n; the Taylor series is
#: exact to ~1e-19 there.  The two branches agree to <= 5e-15 across a wide band around it.
_X_SMALL = 1e-3


def bessel_A(x, damp_exponent=None):
    """A(x) = I1(x)/x, optionally times exp(damp_exponent) folded in for overflow safety.

    `damp_exponent` should be ``x - gamma*t`` (always <= 0 in this kernel), in which case the
    result is ``I1(x)/x * exp(-gamma*t)``.
    """
    x = np.asarray(x, dtype=np.float64)
    scale = np.exp(damp_exponent) if damp_exponent is not None else np.exp(x)
    small = x <= _X_SMALL
    xb, sb = np.broadcast_arrays(x, scale)
    # Both branches return the SCALED value I1(x)/x * exp(-x), so multiplying by
    # exp(damp_exponent) = exp(x - gamma*t) leaves exactly I1(x)/x * exp(-gamma*t).
    with np.errstate(divide="ignore", invalid="ignore"):
        big = np.where(small, 1.0, ive(1, xb) / np.where(small, 1.0, xb))
    x2 = xb * xb
    ser = (0.5 + x2 / 16.0 + x2 * x2 / 384.0) * np.exp(-xb)   # I1(x)/x = 1/2 + x^2/16 + ...
    return np.where(small, ser, big) * sb


def bessel_B(x, damp_exponent=None):
    """B(x) = I2(x)/x^2, with the same optional exponent folding as `bessel_A`."""
    x = np.asarray(x, dtype=np.float64)
    scale = np.exp(damp_exponent) if damp_exponent is not None else np.exp(x)
    small = x <= _X_SMALL
    xb, sb = np.broadcast_arrays(x, scale)
    with np.errstate(divide="ignore", invalid="ignore"):
        big = np.where(small, 1.0, ive(2, xb) / np.where(small, 1.0, xb * xb))
    x2 = xb * xb
    ser = (0.125 + x2 / 96.0 + x2 * x2 / 3072.0) * np.exp(-xb)  # I2(x)/x^2 = 1/8 + x^2/96 + ...
    return np.where(small, ser, big) * sb


def dumping(t, p: LiquefierParams):
    """exp(-gamma*t)/(4 pi)  (CausalLiquefier.cc:160)."""
    return np.exp(-p.gamma_relax * np.asarray(t, dtype=np.float64)) / (4.0 * np.pi)


def shell_mass_fraction(t, p: LiquefierParams):
    """f_delta(t) = int 4 pi r^2 rho_delta dr = (1 + g t + (g t)^2/2) exp(-g t).

    Exact and independent of `width_delta` -- which is what makes regularising the shell
    mass-preserving.  It is also the fraction of the droplet's momentum carried by the wave front,
    so it is the natural "is this deposit shell-dominated?" diagnostic.
    """
    gt = p.gamma_relax * np.asarray(t, dtype=np.float64)
    return (1.0 + gt + 0.5 * gt * gt) * np.exp(-gt)


def _smooth_terms(t, r, p):
    """(A~, B~, inside) with the damping already folded in; zero outside the light cone."""
    t = np.asarray(t, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    c, g = p.c_diff, p.gamma_relax
    inside = r < c * t
    u2 = np.where(inside, c * c * t * t - r * r, 0.0)
    u = np.sqrt(np.maximum(u2, 0.0))
    x = g * u / c
    # x - gamma*t <= 0 always, because u <= c*t.  This is what makes the scaled form safe.
    A = bessel_A(x, damp_exponent=x - g * t)
    B = bessel_B(x, damp_exponent=x - g * t)
    return A, B, inside


def rho_smooth(t, r, p: LiquefierParams):
    """Undamped smooth part, matching the C++ `rho_smooth` term for term."""
    A, B, inside = _smooth_terms(t, r, p)
    c, g = p.c_diff, p.gamma_relax
    val = (g ** 3 / c ** 3) * (A + g * np.asarray(t, dtype=np.float64) * B)
    return np.where(inside, val * np.exp(p.gamma_relax * np.asarray(t, dtype=np.float64)), 0.0)


def j_smooth(t, r, p: LiquefierParams):
    """Undamped smooth current, matching the C++ `j_smooth`."""
    A, B, inside = _smooth_terms(t, r, p)
    c, g = p.c_diff, p.gamma_relax
    val = (g ** 4 / c ** 3) * np.asarray(r, dtype=np.float64) * B
    return np.where(inside, val * np.exp(p.gamma_relax * np.asarray(t, dtype=np.float64)), 0.0)


def rho_delta(t, r, p: LiquefierParams, r_w=None, strict_xscape=False):
    """Damped wave-front (delta-shell) part, normalised to carry mass `shell_mass_fraction(t)`.

    `r_w` overrides `width_delta` (used to widen the shell to the local node spacing, which is
    mass-preserving).  `strict_xscape=True` restores the C++'s raw 1/r^2, including its inf at
    r = 0 when c*t <= r_w.
    """
    t = np.asarray(t, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    R = p.c_diff * t
    w = float(p.width_delta if r_w is None else r_w)
    f = shell_mass_fraction(t, p)

    thin = R > w
    rw = np.where(thin, w, R)                                     # the C++ clamp
    shell = (r >= R - rw) & (r < R)
    with np.errstate(divide="ignore", invalid="ignore"):
        dens_thin = np.where(shell & (r > 0), 1.0 / (4.0 * np.pi * np.where(rw > 0, rw, 1.0)
                                                     * np.where(r > 0, r * r, 1.0)), 0.0)
        if strict_xscape:
            # raw C++: same expression in the R <= w branch too, hence inf at r -> 0
            dens = np.where(shell, np.where(r > 0, dens_thin, np.inf), 0.0)
            return f * dens
        # regularised: a uniform ball of identical total mass, finite at r = 0
        dens_ball = np.where((r < R) & (R > 0),
                             3.0 / (4.0 * np.pi * np.where(R > 0, R, 1.0) ** 3), 0.0)
        dens = np.where(thin, dens_thin, dens_ball)
    return f * dens


def j_delta(t, r, p: LiquefierParams, r_w=None, strict_xscape=False):
    """c_diff * rho_delta  (CausalLiquefier.cc:211)."""
    return p.c_diff * rho_delta(t, r, p, r_w=r_w, strict_xscape=strict_xscape)


def kernel_rho(t, r, p: LiquefierParams, r_w=None, strict_xscape=False):
    """Charge density of the causal-diffusion Green's function, in fm^-3.

    Normalised so that ``int 4 pi r^2 kernel_rho dr = 1`` for every t > 0; this is the C++ unit
    test `TEST_CONSERVATION` (which allows 5%; this port lands at ~0.6%).
    """
    A, B, inside = _smooth_terms(t, r, p)
    c, g = p.c_diff, p.gamma_relax
    tt = np.asarray(t, dtype=np.float64)
    smooth = np.where(inside, (g ** 3 / c ** 3) * (A + g * tt * B), 0.0) / (4.0 * np.pi)
    return smooth + rho_delta(t, r, p, r_w=r_w, strict_xscape=strict_xscape)


def kernel_j(t, r, p: LiquefierParams, r_w=None, strict_xscape=False):
    """Radial current magnitude of the same Green's function, in fm^-3."""
    A, B, inside = _smooth_terms(t, r, p)
    c, g = p.c_diff, p.gamma_relax
    rr = np.asarray(r, dtype=np.float64)
    smooth = np.where(inside, (g ** 4 / c ** 3) * rr * B, 0.0) / (4.0 * np.pi)
    return smooth + j_delta(t, r, p, r_w=r_w, strict_xscape=strict_xscape)


def k_tau(tau, x, y, eta, drop, p: LiquefierParams, r_w=None, strict_xscape=False):
    """The scalar the whole deposit is built from, in fm^-3:

        K_tau = K_rho * cosh(eta) - ((z - z_d)/dr) * K_j * sinh(eta)

    `drop` is (tau_d, x_d, y_d, eta_d, ...); only the position is used.  This is
    `CausalLiquefier.cc:127-137` with the 1/dtau factored out -- see `source.py` for why that
    factor cancels.  The droplet sits AT REST IN THE LAB at fixed z_d; the kernel diffuses in
    Cartesian lab coordinates, which is what makes forward droplets hard (see the README).
    """
    tau_d, x_d, y_d, eta_d = (float(drop[i]) for i in range(4))
    t_d, z_d = tau_d * np.cosh(eta_d), tau_d * np.sinh(eta_d)

    tau = np.asarray(tau, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    t, z = tau * np.cosh(eta), tau * np.sinh(eta)
    dt = t - t_d
    dz = z - z_d
    dr = np.sqrt((np.asarray(x, np.float64) - x_d) ** 2
                 + (np.asarray(y, np.float64) - y_d) ** 2 + dz ** 2)

    krho = kernel_rho(dt, dr, p, r_w=r_w, strict_xscape=strict_xscape)
    kj = kernel_j(dt, dr, p, r_w=r_w, strict_xscape=strict_xscape)
    # the C++ zeroes the longitudinal current when dr <= DBL_MIN (the direction cosine is 0/0)
    cos_z = np.where(dr > 0, dz / np.where(dr > 0, dr, 1.0), 0.0)
    return krho * np.cosh(eta) - cos_z * kj * np.sinh(eta)


def _f32(a):
    return np.asarray(a, dtype=np.float32).astype(np.float64)


def smearing_kernel_jmu(tau, x, y, eta, drop, p: LiquefierParams, emulate_float32=False):
    """Verbatim port of `CausalLiquefier::smearing_kernel` -> jmu (..., 4) in GeV/fm^4.

    Includes the half-open time window, which is the ONLY "when" test in the C++:
    ``tau - dtau/2 <= tau_d + tau_delay < tau + dtau/2``.  `emulate_float32` rounds the droplet
    on ingest and the result on egress, which is what `Jetscape::real = float` does.
    """
    drop = _f32(drop) if emulate_float32 else np.asarray(drop, dtype=np.float64)
    tau_d = float(drop[0])
    eta_arr = np.asarray(eta, dtype=np.float64)
    tau_arr = np.asarray(tau, dtype=np.float64)

    in_window = ((tau_arr - 0.5 * p.dtau <= tau_d + p.tau_delay)
                 & (tau_arr + 0.5 * p.dtau > tau_d + p.tau_delay))
    kt = k_tau(tau, x, y, eta, drop, p) / p.dtau

    E, px, py, pz = (float(drop[i]) for i in (4, 5, 6, 7))
    ch, sh = np.cosh(eta_arr), np.sinh(eta_arr)
    ptau = E * ch - pz * sh                    # get_ptau
    peta = pz * ch - E * sh                    # get_peta -- ORTHONORMAL, no factor of tau
    jmu = np.stack(np.broadcast_arrays(kt * ptau, kt * px, kt * py, kt * peta), axis=-1)
    jmu = np.where(np.asarray(in_window)[..., None], jmu, 0.0)
    return _f32(jmu) if emulate_float32 else jmu


def get_source_xscape(tau, x, y, eta, drops, p: LiquefierParams, emulate_float32=True):
    """Verbatim port of `LiquefierBase::get_source` -> jmu (..., 4), summed over droplets.

    The outer causality cut is ``tau >= tau_d`` and
    ``ds^2 = tau^2 + tau_d^2 - 2 tau tau_d cosh(eta - eta_d) - dx^2 - dy^2 >= 0``.
    Note it uses `tau_d`, NOT `tau_d + tau_delay`.  The cut is in fact redundant -- it equals
    ``dt^2 - dr^2 >= 0`` while the kernel already demands ``dr < c_diff*dt`` with c_diff < 1 --
    but it is kept so this function stays a faithful reference for regression.
    """
    drops = np.atleast_2d(np.asarray(drops, dtype=np.float64))
    tau_a, eta_a = np.asarray(tau, np.float64), np.asarray(eta, np.float64)
    out = np.zeros(np.broadcast(tau_a, np.asarray(x), np.asarray(y), eta_a).shape + (4,))
    for d in drops:
        ds2 = (tau_a ** 2 + d[0] ** 2 - 2.0 * tau_a * d[0] * np.cosh(eta_a - d[3])
               - (np.asarray(x, np.float64) - d[1]) ** 2 - (np.asarray(y, np.float64) - d[2]) ** 2)
        ok = (tau_a >= d[0]) & (ds2 >= 0.0)
        if not np.any(ok):
            continue
        out += np.where(np.asarray(ok)[..., None],
                        smearing_kernel_jmu(tau, x, y, eta, d, p, emulate_float32), 0.0)
    return out
