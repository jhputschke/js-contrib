"""
structure_preserving_hydro_fv.py  (rebuild, 2026-09)

Conservative finite-volume solver for 3+1D relativistic hydrodynamics in Milne
coordinates, written in PyTorch so it runs batched on the training GPU and is
differentiable end to end.  Its job in the fno4hic project is to be a *fast,
low-fidelity 3D training-data generator* — including jet energy–momentum
deposition as a hydro source term — for the FNO surrogate; MUSIC remains the
high-fidelity anchor.

Conventions (fixed; every function below assumes them)
-------------------------------------------------------
coordinates  x^mu = (tau, x, y, eta),  metric g = diag(-1, +1, +1, tau^2)   [mostly plus]
             sqrt(-g) = tau ; nonzero Christoffels: G^tau_{eta eta} = tau,
             G^eta_{tau eta} = G^eta_{eta tau} = 1/tau
units        GeV and fm, hbar c = 0.19733 GeV fm ; e, p in GeV/fm^3 ; T in GeV
tensors      channel-second:  (B, C, X, Y, Z)   with Z = eta ;  axes: x=2, y=3, eta=4
state        q  = tau * T^{tau nu}, nu in (tau, x, y, eta)         (B, 4, X, Y, Z)
             pi = pi^{mu nu} contravariant, 10 independent comps   (B, 10, X, Y, Z)
                  order: tt tx ty te xx xy xe yy ye ee
             Pi = bulk pressure                                    (B, 1, X, Y, Z)
T^{mu nu}    = (e + p + Pi) u^mu u^nu + (p + Pi) g^{mu nu} + pi^{mu nu}
equations    d_tau q^nu + d_i (tau T^{i nu}) = S^nu + tau J^nu
             S = (-tau^2 T^{eta eta}, 0, 0, -2 T^{tau eta})      [exact, cell-local]
             J^nu = jet energy–momentum deposition rate, Milne components.
             The same tau-weight multiplies q and the source, so an instantaneous
             deposit of DeltaP^nu (Milne components) into a cell is
             Delta q^nu = DeltaP^nu / (dx dy deta).
Cartesian    P^t = sum dV (cosh eta q^tau + tau sinh eta q^eta),  P^z = sum dV (sinh eta q^tau + tau cosh eta q^eta)
bookkeeping  P^{x,y} = sum dV q^{x,y}.   Vector components convert as
             J^tau = cosh eta J^t - sinh eta J^z,  J^eta = (cosh eta J^z - sinh eta J^t)/tau.

Scheme
------
* KT/Rusanov central flux on conserved variables with limited (undivided) slopes;
  interface primitives recovered by the same Landau-matching Newton iteration as
  the cell centres.  Conservation of the Milne components is exact (telescoping)
  *independently of the slopes* — which is what lets an FNO supply the slopes
  (see fno_slope_head.py).
* Heun (SSP-RK2) in tau for the ideal part; Strang split V/2 – I – V/2 with the
  Israel–Stewart relaxation integrated by exponential Euler (stiff-safe).
* Jet sources: continuous deposition along straight massless-parton trajectories
  (in Milne coordinates: fixed eta_s = y, transverse speed 1) and/or instantaneous
  four-momentum kicks at chosen tau; both exact in Cartesian four-momentum.

Validation status (run `python structure_preserving_hydro_fv.py --selftest`, all pass, float64, CPU)
--------------------------------------------------------------------------------------------
validated   Landau matching round trip  — 2e-15 (with u^eta != 0, projected pi, Pi)
            telescoping conservation    — 3e-16, verified with random (FNO-like) slopes
            Bjorken (ideal, conformal)  — 4e-7 at dtau = 0.002 (Heun, O(dtau^2))
            Gubser (ideal, conformal)   — transverse sector; L1 error 2% at dx = 0.25 fm, order ~1 (minmod)
            uniform Cartesian flow      — eta-sector with u^eta != 0 (both geometric sources, eta fluxes,
                                          Landau matching at gamma ~ 8); converges, order 1.6 (e) / 3.8 (u^eta)
            IS kinematics algebra       — theta = sigma = Du = 0 on uniform flow, Bjorken sigma/theta exact (1e-14)
            Bjorken Israel–Stewart      — e to 8e-5, phi = pi^xx + pi^yy to 8e-4 vs the reference ODE
            jet sources                 — Cartesian four-momentum bookkeeping 1e-13 (kick) / 3e-16 (continuous)
            wake linearity              — |wake(2E)| / |wake(E)| = 1.997 in paired rollouts
NOT yet     transverse-gradient viscous terms (needs viscous Gubser, arXiv:1307.6130)
            omitted second-order couplings: tau_pipi, phi_7, lambda_piPi, lambda_Pipi
            viscous advection of pi beyond uniform flow
Note        Cartesian energy is conserved only to truncation order (2.5% at deta = 0.3, ~10x smaller at
            deta = 0.15): the scheme telescopes the Milne per-component sums exactly, not cosh(eta)-weighted ones.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

HBARC = 0.19733  # GeV fm

#: A residual substep below this fraction of the nominal one is merged into the step before it
#: instead of being taken on its own.  `strang_step` forms d_tau u as a difference of two
#: Newton-recovered velocities DIVIDED BY THE STEP, so a residual of 1e-9 fm/c against a
#: nominal 0.02 is not a small step, it is recovery noise amplified by 1e7 -- straight into the
#: shear source, and the Israel-Stewart run dies a few frames later.  1e-6 sits well above any
#: float representation crumb in a tau axis (float32 carries ~1e-7 of them) and well below any
#: spacing anyone meant to ask for, so the two populations never mix.  `evolve.evolve_event`
#: uses the same constant, and additionally refuses an axis a fixed substep does not divide.
DTAU_ABSORB = 1e-6

# ----------------------------------------------------------------------------- grid


@dataclass
class Grid:
    """Cell-centred uniform grid.  Coordinates are cell centres, symmetric about 0."""
    nx: int
    ny: int
    neta: int
    dx: float
    dy: float
    deta: float
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float64

    def __post_init__(self):
        def centres(n, d):
            return (torch.arange(n, dtype=self.dtype, device=self.device) - 0.5 * (n - 1)) * d
        self.x = centres(self.nx, self.dx)
        self.y = centres(self.ny, self.dy)
        self.eta = centres(self.neta, self.deta)

    @property
    def dV(self) -> float:            # coordinate volume dx dy deta (times tau for proper volume)
        return self.dx * self.dy * self.deta

    def mesh(self):
        """Broadcastable coordinate fields: X (1,1,X,1,1), Y (1,1,1,Y,1), ETA (1,1,1,1,Z)."""
        return (self.x.view(1, 1, -1, 1, 1), self.y.view(1, 1, 1, -1, 1),
                self.eta.view(1, 1, 1, 1, -1))

    def max_dtau(self, tau: float, cfl: float = 0.4) -> float:
        """CFL bound with the coordinate light speed (1 in x,y ; 1/tau in eta)."""
        return cfl * min(self.dx, self.dy, tau * self.deta)

    def zeros(self, B: int, C: int) -> torch.Tensor:
        return torch.zeros(B, C, self.nx, self.ny, self.neta, dtype=self.dtype, device=self.device)


# ----------------------------------------------------------------------------- EoS


class ConformalEoS:
    """p = e/3 ; T from e = (pi^2/30) dof T^4 / (hbar c)^3."""

    def __init__(self, dof: float = 47.5):
        self.dof = dof
        self.c = (math.pi ** 2 / 30.0) * dof / HBARC ** 3   # e = c T^4

    def p(self, e):
        return e / 3.0

    def cs2(self, e):
        return torch.full_like(e, 1.0 / 3.0)

    def T(self, e):
        return (e.clamp_min(0.0) / self.c) ** 0.25


class TabulatedEoS:
    """Linear interpolation in log(e) of a monotone table; mu_B = 0 so s = (e + p)/T.

    The tables are built in float64 on the CPU (where torch.gradient is accurate) and then
    materialised ONCE on `device`/`dtype`.  Two reasons this matters:
      * MPS has no float64 at all, so a float64 table simply cannot be moved there
        (`TypeError: Cannot convert a MPS Tensor to float64 dtype`);
      * `primitive_recovery` calls the EoS inside its Newton loop, so a per-call
        `.to(e.device)` would copy the tables host->device thousands of times per step.
    Pass device=None/dtype=torch.float64 for the historical CPU behaviour.
    """

    #: points on the uniform log(e) lookup grid (4 tables x 64k x 4 B = 1 MB); see __init__
    N_UNIFORM = 65536

    def __init__(self, e_tab, p_tab, T_tab, device=None, dtype=torch.float64, n_uniform=None):
        e64 = torch.as_tensor(e_tab, dtype=torch.float64)
        p64 = torch.as_tensor(p_tab, dtype=torch.float64)
        T64 = torch.as_tensor(T_tab, dtype=torch.float64)
        # cs2 = dp/de from the table (centred), clipped to (0, 1)
        cs2 = torch.gradient(p64, spacing=(e64,))[0].clamp(1e-4, 0.999)

        # Resample onto a grid that is UNIFORM in log(e).  MUSIC's tables are only piecewise
        # uniform (a coarse low-e extension in front of the fine part), so a lookup needs a
        # binary search -- and primitive_recovery calls the EoS inside a multi-iteration Newton
        # loop, several times per step, which made a tabulated run ~30x slower than a conformal
        # one.  On a uniform grid the index is arithmetic, so the lookup is O(1) and vectorises
        # cleanly on the GPU.  N_UNIFORM is chosen finer than the source table everywhere, so
        # the resampling adds nothing on top of the linear interpolation already in the data.
        n_uniform = int(n_uniform or self.N_UNIFORM)
        loge_src = torch.log(e64)
        loge_u = torch.linspace(float(loge_src[0]), float(loge_src[-1]), n_uniform,
                                dtype=torch.float64)
        idx = torch.searchsorted(loge_src, loge_u).clamp(1, len(loge_src) - 1)
        x0, x1 = loge_src[idx - 1], loge_src[idx]
        w = ((loge_u - x0) / (x1 - x0)).clamp(0.0, 1.0)
        resample = lambda tab: tab[idx - 1] + w * (tab[idx] - tab[idx - 1])

        to = dict(device=device, dtype=dtype)
        self.loge0 = float(loge_u[0])
        self.dloge = float(loge_u[1] - loge_u[0])
        self.n = n_uniform
        self.p_tab = resample(p64).to(**to)
        self.T_tab = resample(T64).to(**to)
        self.cs2_tab = resample(cs2).to(**to)
        self.loge = loge_u.to(**to)          # kept for introspection and tests
        self.device, self.dtype = self.p_tab.device, self.p_tab.dtype

    def to(self, device=None, dtype=None):
        """Return a view of this EoS with its tables on another device/dtype."""
        to = dict(device=device or self.device, dtype=dtype or self.dtype)
        out = object.__new__(TabulatedEoS)
        for k in ("loge", "p_tab", "T_tab", "cs2_tab"):
            setattr(out, k, getattr(self, k).to(**to))
        for k in ("loge0", "dloge", "n"):
            setattr(out, k, getattr(self, k))
        out.device, out.dtype = out.p_tab.device, out.p_tab.dtype
        return out

    def _interp(self, e, tab):
        loge = torch.log(e.clamp_min(1e-30)).to(tab.dtype)
        f = ((loge - self.loge0) / self.dloge).clamp(0.0, self.n - 1 - 1e-6)
        # Bound the INDEX, not just f.  A NaN e survives both clamp_min and clamp (torch
        # propagates NaN through them), and NaN.long() is an out-of-range integer -- which
        # indexes the table out of bounds, fires a device-side assert, and takes the CUDA
        # context down with a traceback pointing nowhere near the divergence that caused it.
        # Clamping after the cast is robust to whatever the cast produced, and is free: it
        # replaces the clamp that guarded i + 1.  w still carries the NaN, so a diverged e
        # propagates as NaN and evolve._check_frame reports it as the divergence it is.
        i = f.long().clamp(0, self.n - 2)
        w = f - i.to(f.dtype)
        y0 = tab[i]
        y1 = tab[i + 1]
        return (y0 + w * (y1 - y0)).to(e.dtype)

    def p(self, e):
        return self._interp(e, self.p_tab)

    def T(self, e):
        return self._interp(e, self.T_tab)

    def cs2(self, e):
        return self._interp(e, self.cs2_tab)


# ----------------------------------------------------------------------------- small tensor helpers


def _pad_axis(t: torch.Tensor, axis: int) -> torch.Tensor:
    """Replicate-pad one of the last three axes (x=2, y=3, eta=4 counted on a 5D tensor)
    by one cell on each side.  Works for tensors with extra leading dims (e.g. (B,4,4,X,Y,Z))."""
    lead = t.shape[:-3]
    flat = t.reshape(-1, 1, *t.shape[-3:])              # (N, 1, X, Y, Z)
    pads = [0] * 6
    k = 2 * (4 - axis)                                   # F.pad orders from the last dim
    pads[k] = pads[k + 1] = 1
    out = F.pad(flat, pads, mode="replicate")
    return out.reshape(*lead, *out.shape[-3:])


def _slice(t: torch.Tensor, axis: int, start, stop) -> torch.Tensor:
    idx = [slice(None)] * t.dim()
    idx[axis] = slice(start, stop)
    return t[tuple(idx)]


def _shift(t: torch.Tensor, axis: int, k: int) -> torch.Tensor:
    """t[i+k] with replicate boundary (no periodic wrap).  `axis` is the 5D convention
    (x=2, y=3, eta=4); tensors with extra leading dims (B,4,4,X,Y,Z) are handled by
    counting from the end."""
    p = _pad_axis(t, axis)
    ax = axis - 5                                   # -3, -2, -1
    n = t.shape[ax]
    return _slice(p, ax, 1 + k, 1 + k + n)


def minmod_slope(q: torch.Tensor, axis: int) -> torch.Tensor:
    """Minmod-limited undivided slope (q-units per cell); boundary slabs zero.
    Identical contract to fno_slope_head.minmod_slope."""
    qm = _shift(q, axis, -1)
    qp = _shift(q, axis, +1)
    dl = q - qm
    dr = qp - q
    s = torch.where(dl * dr > 0, torch.sign(dl) * torch.minimum(dl.abs(), dr.abs()),
                    torch.zeros_like(q))
    n = q.shape[axis]
    mask = torch.ones(n, dtype=q.dtype, device=q.device)
    mask[0] = 0.0
    mask[-1] = 0.0
    shape = [1] * q.dim()
    shape[axis] = n
    return s * mask.view(shape)


def zero_slope(q: torch.Tensor, axis: int) -> torch.Tensor:
    """First-order (piecewise-constant) reconstruction."""
    return torch.zeros_like(q)


# pi^{mu nu}: 10-vector <-> full symmetric 4x4
_PI_IDX = {"tt": 0, "tx": 1, "ty": 2, "te": 3, "xx": 4, "xy": 5, "xe": 6, "yy": 7, "ye": 8, "ee": 9}


def pi_full(pi10: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """(B,10,...) -> (B,4,4,...) symmetric contravariant pi^{mu nu}."""
    if pi10 is None:
        return None
    c = lambda k: pi10[:, _PI_IDX[k]]
    rows = [torch.stack([c("tt"), c("tx"), c("ty"), c("te")], 1),
            torch.stack([c("tx"), c("xx"), c("xy"), c("xe")], 1),
            torch.stack([c("ty"), c("xy"), c("yy"), c("ye")], 1),
            torch.stack([c("te"), c("xe"), c("ye"), c("ee")], 1)]
    return torch.stack(rows, 1)


def pi_10(full: torch.Tensor) -> torch.Tensor:
    f = full
    return torch.stack([f[:, 0, 0], f[:, 0, 1], f[:, 0, 2], f[:, 0, 3], f[:, 1, 1], f[:, 1, 2],
                        f[:, 1, 3], f[:, 2, 2], f[:, 2, 3], f[:, 3, 3]], 1)


def metric_up(tau: float, like: torch.Tensor):
    """g^{mu mu} as a (1,4,1,1,1) tensor."""
    return torch.tensor([-1.0, 1.0, 1.0, 1.0 / tau ** 2], dtype=like.dtype, device=like.device).view(1, 4, 1, 1, 1)


def metric_low(tau: float, like: torch.Tensor):
    return torch.tensor([-1.0, 1.0, 1.0, tau ** 2], dtype=like.dtype, device=like.device).view(1, 4, 1, 1, 1)


# ----------------------------------------------------------------------------- primitive recovery


V_CAP = 0.9999      # |v| cap in primitive recovery (gamma ~ 70); only unphysical states hit it

#: Newton iterations in primitive_recovery, by working precision.  With a TABULATED EoS the
#: iteration is only linearly convergent (~1 digit per step): cs2 is read from its own table and
#: is not exactly dp/de of the p interpolant, so this is an inexact Newton, not a quadratic one.
#: Measured on central Au+Au, max rel. error in e against n_iter=200, over tau = 2, 5 and 8 fm
#: and |v| up to 0.996, and against shear up to |pi|/e = 0.4 and bulk over Pi/p in [-0.9, +0.3]
#: (viscous states are marginally BETTER conditioned -- pi lowers M^2/K^tau, so the v -> 1
#: initial guess starts nearer the root):
#:
#:      n_iter      4        6        8       10       12       16       20
#:      float32   2e-07    5e-07      0        0        0        0        0     (float32 eps)
#:      float64   2e-07    5e-10    1e-12    2e-14    1e-15      0        0
#:
#: float32 has fully settled by 8; float64 needs 16 to reach its own epsilon, so it keeps the
#: historical 20 and its results are unchanged.  A conformal EoS converges by 4 either way.
NEWTON_ITERS = {torch.float64: 20}
NEWTON_ITERS_DEFAULT = 8


def primitive_recovery(q: torch.Tensor, pi: Optional[torch.Tensor], Pi: Optional[torch.Tensor],
                       tau: float, eos, n_iter: Optional[int] = None, e_floor: float = 1e-12,
                       v_cap: float = V_CAP, tol: Optional[float] = None) -> Dict[str, torch.Tensor]:
    """Landau matching: (q, pi, Pi) -> (e, p, T, u^mu, ...).

    With K^nu = T^{tau nu} - pi^{tau nu} and P = p + Pi:
        K^tau + P = w (u^tau)^2,  K^i = w u^tau u^i,  w = e + P,
        M^2 = K_x^2 + K_y^2 + tau^2 (K^eta)^2 = (K^tau + P)^2 v^2,
        e   = K^tau - M^2 / (K^tau + P(e)).
    Newton on the last line (unrolled, differentiable).  f'(e) = 1 - v^2 cs^2 > 0 -- bounded
    below by 1 - cs2_max for any physical EoS, and independent of pi and Pi, which enter only as
    the constants K^nu and an e-independent shift of P.  `n_iter` therefore does not depend on
    the transport mode; None picks it from the working precision (see NEWTON_ITERS).  `tol` adds
    a relative-step early exit, which costs a device sync per iteration and so is off by default:
    on CUDA the loop is launch-bound, and a sync in it costs more than the iterations it saves.
    Vacuum is regularised by e_floor.  A solution with |v| < 1 exists iff M <= K^tau (+Pi terms);
    states violating this (they can only arise from reconstruction, never from a physical q)
    are capped at |v| = v_cap so that fluxes stay finite — the reconstruction fallback in
    flux_divergence_along makes this a last line of defence, not a regular occurrence.
    """
    Tt = q / tau
    if pi is not None:
        Kt = Tt[:, 0] - pi[:, 0]
        Kx = Tt[:, 1] - pi[:, 1]
        Ky = Tt[:, 2] - pi[:, 2]
        Ke = Tt[:, 3] - pi[:, 3]
    else:
        Kt, Kx, Ky, Ke = Tt[:, 0], Tt[:, 1], Tt[:, 2], Tt[:, 3]
    Pi_ = Pi[:, 0] if Pi is not None else torch.zeros_like(Kt)
    M2 = Kx ** 2 + Ky ** 2 + tau ** 2 * Ke ** 2
    tiny = torch.finfo(q.dtype).tiny * 1e10

    if n_iter is None:
        n_iter = NEWTON_ITERS.get(q.dtype, NEWTON_ITERS_DEFAULT)

    e = (Kt - torch.sqrt(M2 + tiny)).clamp_min(e_floor)      # v -> 1 guess; f(e) < 0 there
    for _ in range(n_iter):
        P = eos.p(e) + Pi_
        den = Kt + P
        f = e - Kt + M2 / (den + tiny)
        fp = 1.0 - M2 * eos.cs2(e) / (den ** 2 + tiny)
        de = f / fp.clamp_min(1e-3)
        e = (e - de).clamp_min(e_floor)
        if tol is not None and float((de.abs() / e.clamp_min(e_floor)).max()) < tol:
            break

    p = eos.p(e)
    P = p + Pi_
    w = e + P
    v2 = (M2 / ((Kt + P) ** 2 + tiny)).clamp(0.0, v_cap ** 2)
    u_tau = 1.0 / torch.sqrt(1.0 - v2)
    wut = (w * u_tau).clamp_min(tiny)
    u_x = Kx / wut
    u_y = Ky / wut
    u_eta = Ke / wut
    # u^tau comes from the CAPPED v^2, but u^i = K^i/(w u^tau) does not, so wherever the cap
    # (or e_floor, or the `tiny` guards) bites, u stops being a unit vector: u^tau saturates at
    # 1/sqrt(1 - v_cap^2) ~ 70 while u^i keeps growing without bound.  The ideal sector never
    # notices -- it only ever contracts u into fluxes -- but the Israel-Stewart sector takes
    # finite differences of u to build dU, theta and sigma, and those inherit the full
    # magnitude: measured on central Au+Au, |u^i| reached 1.2e12 against u^tau = 70.72 and
    # sigma overflowed float32 in 304 vacuum cells.  Rescale the spatial part back onto
    # u.u = -1.  The scale is exactly 1 in every consistent cell, so the ideal path is
    # bit-for-bit unchanged.
    # Gate on the EXACT conditions under which u is inconsistent -- the |v| cap saturated, or e
    # sitting on its floor -- not on a numerical |u.u + 1| test.  sp2 and need are equal only
    # analytically; they are computed by different float paths and disagree by a few eps, so a
    # tolerance test rescales healthy cells by 1 +- 1e-6 and perturbs the ideal sector
    # everywhere.  With this predicate every consistent cell keeps its bits exactly.
    sp2 = u_x ** 2 + u_y ** 2 + tau ** 2 * u_eta ** 2
    need = v2 * u_tau ** 2                         # = u_tau^2 - 1, without the cancellation
    capped = (v2 >= v_cap ** 2) | (e <= e_floor)
    scale = torch.where(capped, torch.sqrt(need / sp2.clamp_min(tiny)), torch.ones_like(sp2))
    u_x, u_y, u_eta = u_x * scale, u_y * scale, u_eta * scale
    u = torch.stack([u_tau, u_x, u_y, u_eta], 1)             # (B,4,X,Y,Z)
    return {"e": e, "p": p, "P": P, "w": w, "T": eos.T(e), "cs2": eos.cs2(e), "Pi": Pi_,
            "u": u, "u_tau": u_tau, "u_x": u_x, "u_y": u_y, "u_eta": u_eta,
            "capped": capped}    # cells whose u had to be regularised; the viscous sector
                                 # must not build velocity gradients out of these


_EAGER_RECOVERY = None


def enable_compiled_recovery(enable: bool = True, dynamic: bool = True,
                             recompile_limit: int = 128) -> bool:
    """Swap primitive_recovery for a torch.compile'd one, process-wide.  Returns True if on.

    The solver is launch-bound, not compute-bound: one strang_step issues ~7400 kernels whose
    mean duration is 2.7 us against 2.7 us of host time just to launch each, GPU busy sits at
    ~55%, and primitive_recovery is ~95% of the launches.  Fusing it is worth 29x on the
    function, 5.1x on a step, and 2.7x per event at steady state (4.38 -> 1.62 s/event on
    central Au+Au, CUDA, float32).

    `dynamic` must stay True.  primitive_recovery runs both on cell centres (nx) and, through
    flux_divergence_along, on reconstructed interface states (nx + 1 on the axis being
    differenced).  With static shapes Inductor compiles a kernel per shape -- measured 155
    s/event -- and under the DEFAULT recompile limit Dynamo gives up after 8 and silently falls
    back to eager, so you pay the compile and get nothing.  Hence the limit is raised here too.

    Two costs, which is why this is opt-in.  The fused kernel reassociates in float32, so the
    output is NOT bit-identical to an eager run.  Measured over 12 central Au+Au events, energy
    channel, against the same seeds run eagerly:

        max absolute difference            1.2e-04 GeV/fm^3
        max relative, e > 0.15 (above T_fo)  1.5e-06
        max relative, e > 1e-3 (dilute tail) 5.6e-02
        total energy per frame               agrees to 7 significant figures
        ntau_freezeout                       identical on all 12 events

    i.e. the hot fluid is untouched at 1e-06, and it is only in the dilute tail, where that same
    1e-04 absolute is a large fraction of e, that it shows.  Worth knowing before training on
    the tail.  The other cost is the first compile, ~16 s, cached under TORCHINDUCTOR_CACHE_DIR
    -- /tmp by default, so it is cold again after a reboot.  Break-even is 8 events cold, 2 warm.
    """
    global primitive_recovery, _EAGER_RECOVERY
    if enable:
        if _EAGER_RECOVERY is None:
            import torch._dynamo
            torch._dynamo.config.recompile_limit = max(
                getattr(torch._dynamo.config, "recompile_limit", 8), recompile_limit)
            _EAGER_RECOVERY = primitive_recovery
            primitive_recovery = torch.compile(primitive_recovery, dynamic=dynamic)
        return True
    if _EAGER_RECOVERY is not None:
        primitive_recovery = _EAGER_RECOVERY
        _EAGER_RECOVERY = None
    return False


def conserved_from_primitives(e, u, tau: float, eos, pi=None, Pi=None) -> torch.Tensor:
    """q = tau T^{tau nu} from e (B,X,Y,Z) and u (B,4,X,Y,Z) (+ optional pi, Pi)."""
    Pi_ = Pi[:, 0] if Pi is not None else torch.zeros_like(e)
    P = eos.p(e) + Pi_
    w = e + P
    T = w.unsqueeze(1) * u[:, 0:1] * u        # w u^tau u^nu
    g = metric_up(tau, e)
    T = T + P.unsqueeze(1) * g * torch.tensor([1.0, 0, 0, 0], dtype=e.dtype, device=e.device).view(1, 4, 1, 1, 1)
    if pi is not None:
        T = T + pi[:, 0:4]                    # pi^{tau nu}
    return tau * T


# ----------------------------------------------------------------------------- fluxes and sources


def milne_flux(prim: Dict[str, torch.Tensor], pi4: Optional[torch.Tensor], tau: float, axis: int) -> torch.Tensor:
    """tau * T^{i nu} for i = x (axis 2), y (3), eta (4).  (B,4,X,Y,Z)."""
    i = axis - 1                              # 4-vector component index
    u = prim["u"]
    ui = u[:, i:i + 1]
    Fl = prim["w"].unsqueeze(1) * ui * u      # w u^i u^nu
    g_ii = 1.0 if i < 3 else 1.0 / tau ** 2
    onehot = torch.zeros(1, 4, 1, 1, 1, dtype=u.dtype, device=u.device)
    onehot[0, i] = g_ii
    Fl = Fl + prim["P"].unsqueeze(1) * onehot
    if pi4 is not None:
        Fl = Fl + pi4[:, i]                   # pi^{i nu}
    return tau * Fl


def geometric_source(prim: Dict[str, torch.Tensor], tau: float, pi4: Optional[torch.Tensor] = None,
                     Pi: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Exact Milne geometric source for q = tau T^{tau nu}:
        S = (-tau^2 T^{eta eta}, 0, 0, -2 T^{tau eta}).
    Bjorken pins the first (d_tau(tau e) = -p); the 2 in the last comes from
    Gamma^eta_{tau eta} and Gamma^eta_{eta tau} both contributing.  (Pi already in prim["P"].)"""
    w, P, u = prim["w"], prim["P"], prim["u"]
    T_ee = w * u[:, 3] * u[:, 3] + P / tau ** 2
    T_te = w * u[:, 0] * u[:, 3]
    if pi4 is not None:
        T_ee = T_ee + pi4[:, 3, 3]
        T_te = T_te + pi4[:, 0, 3]
    zero = torch.zeros_like(T_ee)
    return torch.stack([-tau ** 2 * T_ee, zero, zero, -2.0 * T_te], 1)


def max_signal_speed(prim: Dict[str, torch.Tensor], tau: float, axis: int, speed: str = "local") -> torch.Tensor:
    """Largest |characteristic speed| in coordinate units along `axis`, shape (B,1,X,Y,Z).
    'light': 1 (x,y) or 1/tau (eta) — always safe.
    'local': relativistic sound characteristics
        lam+- = [v_i (1-cs2) +- cs sqrt((1-v^2)(1 - v^2 cs2 - v_i^2 (1-cs2)))] / (1 - v^2 cs2),
    with v_i the physical velocity along the axis (v_eta_phys = tau u^eta/u^tau); divided by tau for eta."""
    u = prim["u"]
    if speed == "light":
        a = torch.ones_like(u[:, 0:1])
        return a if axis < 4 else a / tau
    ut = u[:, 0]
    vx, vy, ve = u[:, 1] / ut, u[:, 2] / ut, tau * u[:, 3] / ut
    v2 = (vx ** 2 + vy ** 2 + ve ** 2).clamp(0.0, 1.0 - 1e-12)
    vi = {2: vx, 3: vy, 4: ve}[axis]
    cs2 = prim["cs2"]
    cs = torch.sqrt(cs2)
    root = torch.sqrt(((1.0 - v2) * (1.0 - v2 * cs2 - vi ** 2 * (1.0 - cs2))).clamp_min(0.0))
    den = 1.0 - v2 * cs2
    lp = (vi * (1.0 - cs2) + cs * root) / den
    lm = (vi * (1.0 - cs2) - cs * root) / den
    a = torch.maximum(lp.abs(), lm.abs()).clamp(max=1.0).unsqueeze(1)
    return a if axis < 4 else a / tau


def _is_physical(q: torch.Tensor, pi: Optional[torch.Tensor], tau: float, margin: float = 0.999) -> torch.Tensor:
    """M <= margin * K^tau (necessary and sufficient for a |v| < 1 Landau-matching solution at Pi = 0)."""
    Tt = q / tau
    if pi is not None:
        K = Tt - pi[:, 0:4]
    else:
        K = Tt
    M = torch.sqrt(K[:, 1] ** 2 + K[:, 2] ** 2 + tau ** 2 * K[:, 3] ** 2)
    return (K[:, 0] > 0) & (M <= margin * K[:, 0])


def flux_divergence_along(q: torch.Tensor, prim: Dict[str, torch.Tensor], slope: torch.Tensor, axis: int,
                          dx: float, tau: float, eos, pi: Optional[torch.Tensor] = None,
                          Pi: Optional[torch.Tensor] = None, speed: str = "local",
                          return_boundary: bool = False):
    """-(F_{i+1/2} - F_{i-1/2})/dx with KT/Rusanov interface fluxes.

    Interface states from the (undivided) slopes: q_L = q_i + s_i/2, q_R = q_{i+1} - s_{i+1}/2;
    pi and Pi at the interface by averaging.  Primitives at both interface states are
    recovered with the same Newton iteration as the cell centres.  Conservation is the
    telescoping of F over interior faces — independent of the slopes.  Compatible with the
    call in fno_slope_head.py (pi/Pi omitted -> ideal)."""
    qp = _pad_axis(q, axis)
    sp = _pad_axis(slope, axis)
    qLc, qRc = _slice(qp, axis, 0, -1), _slice(qp, axis, 1, None)          # first-order states
    qL = qLc + 0.5 * _slice(sp, axis, 0, -1)
    qR = qRc - 0.5 * _slice(sp, axis, 1, None)
    if pi is not None:
        pp = _pad_axis(pi, axis)
        pi_f = 0.5 * (_slice(pp, axis, 0, -1) + _slice(pp, axis, 1, None))
    else:
        pi_f = None
    # Fallback: a reconstructed state with M > K^tau has no |v| < 1 solution.  Drop that face
    # to first order (both sides) — conservation is untouched, only the interface states change.
    ok = (_is_physical(qL, pi_f, tau) & _is_physical(qR, pi_f, tau)).unsqueeze(1)
    qL = torch.where(ok, qL, qLc)
    qR = torch.where(ok, qR, qRc)
    if Pi is not None:
        Pp = _pad_axis(Pi, axis)
        Pi_f = 0.5 * (_slice(Pp, axis, 0, -1) + _slice(Pp, axis, 1, None))
    else:
        Pi_f = None
    pi4_f = pi_full(pi_f)
    primL = primitive_recovery(qL, pi_f, Pi_f, tau, eos)
    primR = primitive_recovery(qR, pi_f, Pi_f, tau, eos)
    FL = milne_flux(primL, pi4_f, tau, axis)
    FR = milne_flux(primR, pi4_f, tau, axis)
    a = torch.maximum(max_signal_speed(primL, tau, axis, speed), max_signal_speed(primR, tau, axis, speed))
    Fface = 0.5 * (FL + FR) - 0.5 * a * (qR - qL)                  # (B,4,...,N+1,...)
    div = -(_slice(Fface, axis, 1, None) - _slice(Fface, axis, 0, -1)) / dx
    if return_boundary:
        return div, _slice(Fface, axis, 0, 1), _slice(Fface, axis, -1, None)
    return div


def ideal_rhs(q, pi, Pi, tau: float, grid: Grid, eos, slope_fn: Callable = minmod_slope,
              speed: str = "local", extra_rate: Optional[Callable] = None):
    """d q / d tau from fluxes + geometric source (+ optional extra rate field, e.g. tau J^nu).
    Returns (rhs, prim)."""
    prim = primitive_recovery(q, pi, Pi, tau, eos)
    pi4 = pi_full(pi)
    rhs = geometric_source(prim, tau, pi4, Pi)
    for axis, dx in ((2, grid.dx), (3, grid.dy), (4, grid.deta)):
        rhs = rhs + flux_divergence_along(q, prim, slope_fn(q, axis), axis, dx, tau, eos, pi, Pi, speed)
    if extra_rate is not None:
        rhs = rhs + extra_rate(tau, prim)
    return rhs, prim


def ideal_step(q, pi, Pi, tau: float, dtau: float, grid: Grid, eos, slope_fn=minmod_slope,
               speed: str = "local", extra_rate=None):
    """Heun / SSP-RK2 for the ideal (flux + geometric source) part; pi, Pi frozen."""
    k1, prim0 = ideal_rhs(q, pi, Pi, tau, grid, eos, slope_fn, speed, extra_rate)
    q1 = q + dtau * k1
    k2, _ = ideal_rhs(q1, pi, Pi, tau + dtau, grid, eos, slope_fn, speed, extra_rate)
    return 0.5 * (q + q1 + dtau * k2), prim0


# ----------------------------------------------------------------------------- viscous sector (Israel–Stewart)


@dataclass
class Transport:
    """Transport coefficients.  eta_over_s / zeta_over_s: float or callable of T [GeV].

    `tau_pi_coeff` = 5.0 is the DNMR conformal value, and the one test_gubser_viscous validates
    against (the Marrochio et al. tauR_hat = c (eta/s)/That with c = 5).  It is also MUSIC's
    in-code default -- but NOT what a stock MUSIC run uses: `read_in_parameters.cpp:238` reads
    `shear_relax_time_factor` from the input file, and stock `music_input` ships **4.65**.
    Nothing in the XML or MusicWrapper overrides it, so "matching MUSIC" here means checking
    that file, not trusting this docstring.  The fv-vs-MUSIC study ran on the mismatch for its
    first round; see RESULTS_fv_vs_music.md.

    DNMR relaxation times: tau_pi = tau_pi_coeff * eta/(e+p),
    tau_Pi = zeta / (15 (1/3 - cs2)^2 (e+p)); second-order: delta_pipi = 4/3 tau_pi,
    delta_PiPi = 2/3 tau_Pi.  (tau_pipi, phi_7, lambda_piPi, lambda_Pipi omitted.)"""
    eta_over_s: object = 0.08
    zeta_over_s: object = 0.0
    tau_pi_coeff: float = 5.0
    delta_pipi: float = 4.0 / 3.0
    delta_PiPi: float = 2.0 / 3.0
    tau_min: float = 0.02      # fm; floor on relaxation times (avoids 1/0 in the vacuum)
    pi_rho_max: Optional[float] = 1.0   # rescale pi above this |pi|/(e+p); see regulate_pi
    pi_e_min: Optional[float] = 1e-3    # GeV/fm^3; freeze pi below this e, and wherever the
                                        # velocity had to be capped -- see viscous_step
    pi_advection: str = "centred"       # centred | upwind, the stencil pi is advected with
    Pi_p_bounds: Optional[Tuple[float, float]] = (-0.9, 0.3)   # bound Pi/p after every step;
                                        # None disables.  Inert at zeta = 0 -- see viscous_step

    def _eval(self, f, T):
        return f(T) if callable(f) else torch.full_like(T, float(f))

    def eta(self, prim):
        """Shear viscosity in GeV/fm^2:  eta = (eta/s) * s * hbar c,  s = (e+p)/T in fm^-3.
        Then pi_NS = 2 eta sigma is in GeV/fm^3 and tau_pi = 5 eta/(e+p) = 5 (eta/s) hbar c / T in fm."""
        s = (prim["e"] + prim["p"]) / prim["T"].clamp_min(1e-6)
        return self._eval(self.eta_over_s, prim["T"]) * s * HBARC

    def zeta(self, prim):
        s = (prim["e"] + prim["p"]) / prim["T"].clamp_min(1e-6)
        return self._eval(self.zeta_over_s, prim["T"]) * s * HBARC


def finite_difference_dU(u: torch.Tensor, dudtau: Optional[torch.Tensor], grid: Grid) -> torch.Tensor:
    """dU[:, mu, nu] = d_mu u^nu.  Spatial: centred differences with replicate boundary; tau: the
    d_tau u handed in by strang_step, or zero.

    The spatial stencil is deliberately plain.  MUSIC's generalised-minmod gradient (theta 1.8)
    and a noise-annihilating 5-point stencil were both tried against the Gubser solution and
    both made sigma WORSE, even from the exact u (6.1e-01 -> 1.9e-01 -> 5.9e-02 and 2.0e-01 ->
    5.1e-02 -> 1.3e-02 over n = 64/128/256, against 8.0e-02 -> 2.0e-02 -> 5.1e-03 centred): the
    Gubser velocity has genuine extrema that a limiter clips.  That sentence turned out to
    describe the real bottleneck, in the wrong place -- it is the limiter in the FLUX
    reconstruction, not in this stencil, that caps the viscous sector; see
    test_gubser_viscous.

    This stencil is not the problem, and that is now measured rather than inherited.  Taking
    it to fourth order on the solver's own u -- 150x more accurate, 1.0e-03 -> 6.7e-06 on the
    initial condition -- moves pi's n = 256 -> 512 convergence from order 0.16 to 0.17.  The
    older claim, "freezing them to exact values changed nothing", reached the same conclusion
    by an experiment that does not survive the d_tau u fix: exact gradients belong to a
    velocity field the solver does not have, and substituting them now makes the run diverge.
    """
    B = u.shape[0]
    dt = dudtau if dudtau is not None else torch.zeros_like(u)
    dxs = [(2, grid.dx), (3, grid.dy), (4, grid.deta)]
    spatial = [(_shift(u, ax, +1) - _shift(u, ax, -1)) / (2.0 * d) for ax, d in dxs]
    return torch.stack([dt] + spatial, 1)                     # (B,4,4,X,Y,Z)


def shear_and_expansion(u: torch.Tensor, dU: torch.Tensor, tau: float):
    """Milne-covariant kinematics from u^mu (B,4,...) and dU[mu,nu] = d_mu u^nu.

    nabla_mu u_nu = d_mu u_nu - Gamma^lam_{mu nu} u_lam, with u_nu = g_{nu nu} u^nu and the
    explicit tau-dependence of g_{eta eta} = tau^2 included in d_tau u_eta.
    theta = g^{mu nu} nabla_mu u_nu
    sigma_{mu nu} = -[ Delta^a_mu Delta^b_nu (nabla_(a u_b)) - Delta_{mu nu} theta/3 ]
        — the overall minus sign is the mostly-plus convention that makes
          pi^{mu nu}_NS = +2 eta sigma^{mu nu} physical (Bjorken: sigma^{eta eta} = -2/(3 tau^3)).
    Returns theta (B,...), sigma^{mu nu} (B,4,4,...), Du_nu = u^mu nabla_mu u_nu (B,4,...)."""
    dtype, dev = u.dtype, u.device
    gl = torch.tensor([-1.0, 1.0, 1.0, tau ** 2], dtype=dtype, device=dev)
    gu = torch.tensor([-1.0, 1.0, 1.0, 1.0 / tau ** 2], dtype=dtype, device=dev)
    ul = [gl[n] * u[:, n] for n in range(4)]                    # u_nu
    # d_mu u_nu
    A = [[gl[n] * dU[:, m, n] for n in range(4)] for m in range(4)]
    A[0][3] = A[0][3] + 2.0 * tau * u[:, 3]                     # d_tau (tau^2 u^eta)
    # - Gamma^lam_{mu nu} u_lam
    A[3][3] = A[3][3] - tau * ul[0]                             # Gamma^tau_{eta eta} = tau
    A[0][3] = A[0][3] - ul[3] / tau                             # Gamma^eta_{tau eta}
    A[3][0] = A[3][0] - ul[3] / tau                             # Gamma^eta_{eta tau}
    A = torch.stack([torch.stack(row, 1) for row in A], 1)      # (B,4,4,...) [mu][nu]
    theta = sum(gu[m] * A[:, m, m] for m in range(4))
    S = 0.5 * (A + A.transpose(1, 2))
    ulow = torch.stack(ul, 1)                                   # (B,4,...)
    eye = torch.eye(4, dtype=dtype, device=dev).view(1, 4, 4, *([1] * (u.dim() - 2)))
    Dmix = eye + u.unsqueeze(2) * ulow.unsqueeze(1)             # Delta^a_mu  [a][mu]
    Dlow = torch.diag(gl).view(1, 4, 4, *([1] * (u.dim() - 2))) + ulow.unsqueeze(2) * ulow.unsqueeze(1)
    proj = torch.einsum("bam...,bcn...,bac...->bmn...", Dmix, Dmix, S)
    sig_low = -(proj - Dlow * theta.unsqueeze(1).unsqueeze(1) / 3.0)
    sig_up = sig_low * gu.view(1, 4, 1, *([1] * (u.dim() - 2))) * gu.view(1, 1, 4, *([1] * (u.dim() - 2)))
    Du = torch.einsum("bm...,bmn...->bn...", u, A)
    return theta, sig_up, Du


def project_transverse_traceless(pi4: torch.Tensor, u: torch.Tensor, tau: float) -> torch.Tensor:
    """Delta^{mu nu}_{ab} pi^{ab}: enforce u_mu pi^{mu nu} = 0 and pi^mu_mu = 0."""
    dtype, dev = u.dtype, u.device
    gl = torch.tensor([-1.0, 1.0, 1.0, tau ** 2], dtype=dtype, device=dev)
    gu = torch.tensor([-1.0, 1.0, 1.0, 1.0 / tau ** 2], dtype=dtype, device=dev)
    sh = [1] * (u.dim() - 2)
    ulow = u * gl.view(1, 4, *sh)
    eye = torch.eye(4, dtype=dtype, device=dev).view(1, 4, 4, *sh)
    Dmix = eye + u.unsqueeze(2) * ulow.unsqueeze(1)                       # Delta^m_a
    Dup = torch.diag(gu).view(1, 4, 4, *sh) + u.unsqueeze(2) * u.unsqueeze(1)
    Dlow = torch.diag(gl).view(1, 4, 4, *sh) + ulow.unsqueeze(2) * ulow.unsqueeze(1)
    p = torch.einsum("bma...,bnc...,bac...->bmn...", Dmix, Dmix, pi4)
    tr = torch.einsum("bac...,bac...->b...", Dlow, pi4)
    return p - Dup * tr.unsqueeze(1).unsqueeze(1) / 3.0


def _upwind_advect(f: torch.Tensor, u: torch.Tensor, grid: Grid,
                   scheme: str = "centred") -> torch.Tensor:
    """-(u^i/u^tau) d_i f.  f: (B, ..., X, Y, Z) with u (B,4,X,Y,Z).

    "centred" is the second-order central difference, "upwind" the first-order donor cell the
    solver shipped with.  pi is advected ONLY here, so this one derivative sets the order of the
    viscous sector.  With d_tau u done right (strang_step), against Gubser inside |r| < 4 fm at
    n = 256, centred gives pi 5.3e-02 at orders 1.48/1.28 against 6.7e-02 at 1.38/0.87 for
    upwind, and e and u^x improve with it (9.0e-03, 7.4e-03 against 9.5e-03, 9.6e-03).  It is
    non-dissipative, and that is safe here for two reasons: pi is a relaxation variable, not a
    conserved density that shocks, and the one place it is rough -- the vacuum edge -- is
    handled by Transport.pi_e_min freezing it outright.  Checked on production Au+Au at
    eta/s = 0.005 ... 0.16: every run freezes out at the same frame as upwind.  Before the
    d_tau u fix the same change made things WORSE (2.2e-01 -> 2.9e-01), because two first-order
    errors of opposite sign were partly cancelling; that is why it was rejected once.
    """
    out = torch.zeros_like(f)
    ut = u[:, 0]
    for k, (ax, d) in enumerate([(2, grid.dx), (3, grid.dy), (4, grid.deta)]):
        v = u[:, k + 1] / ut                                              # (B,X,Y,Z)
        back = (f - _shift(f, ax, -1)) / d
        fwd = (_shift(f, ax, +1) - f) / d
        vb = v.view(v.shape[0], *([1] * (f.dim() - 4)), *v.shape[1:])
        slope = 0.5 * (back + fwd) if scheme == "centred" else torch.where(vb > 0, back, fwd)
        out = out - vb * slope
    return out


def regulate_pi(pi4: torch.Tensor, w: torch.Tensor, tau: float,
                rho_max: Optional[float]) -> torch.Tensor:
    """Rescale pi^{mu nu} so that sqrt(pi_{mu nu} pi^{mu nu}) <= rho_max * (e + p).

    pi is relaxed on the WHOLE grid, vacuum included, and in a vacuum cell (e pinned at e_floor,
    p ~ 0) the Navier-Stokes target it relaxes towards carries no scale that bounds it.  Left
    alone the ratio runs away -- measured on central Au+Au at eta/s = 0.08, live cells sit at
    0.10-0.13 for the whole evolution while vacuum cells go 0.44 -> 0.63 -> 9.5 -> 1.9e21 -> NaN
    in four steps.  Past ~1 the viscous term dominates K^nu = T^{tau nu} - pi^{tau nu} and takes
    the energy density itself non-finite, so the run dies in the ideal sector with no sign of
    where it started.  This is the usual MUSIC/DNMR-style rescaling; rho_max=None disables it.

    On its own this is NOT enough: pi can arrive here already 1e6 over the bound, having been
    amplified inside a single half-step.  Transport.pi_e_min is what prevents that; see the
    freeze in viscous_step.
    """
    if rho_max is None:
        return pi4
    sh = [1] * (pi4.dim() - 3)
    gl = torch.tensor([-1.0, 1.0, 1.0, tau ** 2], dtype=pi4.dtype, device=pi4.device)
    piSq = (pi4 ** 2 * gl.view(1, 4, 1, *sh) * gl.view(1, 1, 4, *sh)).sum((1, 2))
    rho = piSq.abs().sqrt() / w.clamp_min(1e-30)
    s = torch.where(rho > rho_max, rho_max / rho.clamp_min(1e-30), torch.ones_like(rho))
    return pi4 * s.unsqueeze(1).unsqueeze(1)


def viscous_step(pi10: torch.Tensor, Pi: torch.Tensor, prim: Dict[str, torch.Tensor],
                 dudtau: Optional[torch.Tensor], tau: float, dtau: float, grid: Grid, eos,
                 tr: Transport):
    """Israel–Stewart relaxation over dtau (exponential Euler), holding q fixed.

    D pi^{mu nu} = (u^mu pi^{a nu} + u^nu pi^{mu a}) Du_a
                   - [pi - 2 eta sigma + delta_pipi tau_pi pi theta] / tau_pi
    with D = u^lam nabla_lam expanded as u^tau d_tau + u^i d_i + Christoffel terms.
    The stiff relaxation is integrated exactly; the second-order theta term is explicit.
    Bulk: D Pi = -[Pi + zeta theta + delta_PiPi tau_Pi Pi theta]/tau_Pi.
    Afterwards pi is projected transverse-traceless w.r.t. the current u."""
    u = prim["u"]
    ut = u[:, 0]
    e, p = prim["e"], prim["p"]
    eta = tr.eta(prim)
    zeta = tr.zeta(prim)
    tau_pi = (tr.tau_pi_coeff * eta / (e + p).clamp_min(1e-30)).clamp_min(tr.tau_min)
    tau_Pi = (zeta / (15.0 * (1.0 / 3.0 - prim["cs2"]) ** 2 * (e + p)).clamp_min(1e-30)).clamp_min(tr.tau_min)

    dU = finite_difference_dU(u, dudtau, grid)
    theta, sig, Du = shear_and_expansion(u, dU, tau)
    pi4 = pi_full(pi10)

    # Christoffel part of u^lam nabla_lam pi^{mu nu}:  X^m_a pi^{a n} + (m<->n), X^m_a = u^lam Gamma^m_{lam a}
    sh = pi4.shape
    X = torch.zeros(sh[0], 4, 4, *sh[3:], dtype=pi4.dtype, device=pi4.device)
    X[:, 0, 3] = tau * u[:, 3]
    X[:, 3, 3] = u[:, 0] / tau
    X[:, 3, 0] = u[:, 3] / tau
    C = torch.einsum("bma...,ban...->bmn...", X, pi4)
    C = C + C.transpose(1, 2)
    piDu = torch.einsum("ban...,ba...->bn...", pi4, Du)                      # pi^{a nu} Du_a
    I = u.unsqueeze(2) * piDu.unsqueeze(1) + u.unsqueeze(1) * piDu.unsqueeze(2)
    adv = _upwind_advect(pi4, u, grid, tr.pi_advection)                      # -(u^i/u^tau) d_i pi
    ut4 = ut.unsqueeze(1).unsqueeze(1)
    A = adv + (I - C) / ut4                                                  # non-relaxation part of d_tau pi
    th4 = theta.unsqueeze(1).unsqueeze(1)
    target = 2.0 * eta.unsqueeze(1).unsqueeze(1) * sig - tr.delta_pipi * (tau_pi * theta).unsqueeze(1).unsqueeze(1) * pi4
    r = 1.0 / (tau_pi * ut)
    E = torch.exp(-r * dtau)
    phi = (1.0 - E) / r
    E4, phi4 = E.unsqueeze(1).unsqueeze(1), phi.unsqueeze(1).unsqueeze(1)
    pi_new = pi4 * E4 + target * (1.0 - E4) + A * phi4
    pi_new = project_transverse_traceless(pi_new, u, tau)
    # Freeze pi where the fluid is not a fluid.  Regulating after the fact cannot help here:
    # measured on central Au+Au, pi reaches the regulator already 1.6e6 times over the bound,
    # having grown from 0.059 to 3.7e5 inside ONE half-step.  The amplification is in A, which
    # carries Du and dU -- and those are finite differences of a u that was capped, i.e. of a
    # velocity field the recovery had to invent.  Holding pi at its incoming value in those
    # cells stops the garbage entering the relaxation instead of rescaling it afterwards.
    #
    # The freeze goes BEFORE the regulator, never after: a frozen cell must still be bounded,
    # or it keeps a stale pi while e + p collapses around it and the ratio runs to 1e17.
    if tr.pi_e_min is not None:
        frozen = (e < tr.pi_e_min) | prim["capped"]
        pi_new = torch.where(frozen.unsqueeze(1).unsqueeze(1), pi4, pi_new)
    pi_new = regulate_pi(pi_new, e + p, tau, tr.pi_rho_max)

    # bulk
    Pi0 = Pi[:, 0]
    advP = _upwind_advect(Pi0, u, grid, tr.pi_advection)
    targetP = -zeta * theta - tr.delta_PiPi * tau_Pi * theta * Pi0
    rP = 1.0 / (tau_Pi * ut)
    EP = torch.exp(-rP * dtau)
    Pi_new = Pi0 * EP + targetP * (1.0 - EP) + advP * (1.0 - EP) / rP
    if tr.pi_e_min is not None:                      # freeze first, then bound -- as for pi
        Pi_new = torch.where(frozen, Pi0, Pi_new)
    if tr.pi_rho_max is not None:                    # same bound, same reason as regulate_pi
        lim = tr.pi_rho_max * (e + p)                # inert at the default zeta_over_s = 0,
        Pi_new = Pi_new.clamp(-lim, lim)             # where Pi is identically zero
    # ...but for BULK, (e + p) is not the bound that matters: p + Pi is the pressure the
    # recovery works with, and (e + p) is 4-7 p near T_c.  Without a bound on Pi/p, 0-10% Au+Au
    # diverges at zeta/s = 0.04 within a couple of events, and at 0.12 about 1 event in 30
    # (tau = 5-6 fm/c, still hot).  The measured mechanism: a corona cell frozen below pi_e_min
    # keeps its Pi while its p falls -- Pi/p went -0.3 -> -5.8 over 2 fm/c -- until p + Pi < 0.
    # Raising pi_e_min only moves the corona.  So Pi is held to the range the recovery is
    # validated for (Pi/p in [-0.9, +0.3], see NEWTON_ITERS above).
    #
    # This is a REGULATOR, not only a corona guard.  At zeta/s = 0.12 (constant), the bulk
    # correction near T_c overshoots early: for tau < 3 fm/c the unbounded step gives Pi/p down
    # to -1.1, and the bound trims it in ~20-30% of cells at e = 0.24-1 GeV/fm^3 (2% above
    # 1 GeV/fm^3); after tau ~ 3 it acts almost only on the corona.  The effect on the medium
    # is small: at eta = 0, total energy +0.1% and e-weighted <v_T> +0.04% without it (10
    # events, tau = 3-9 fm/c).  At zeta = 0 Pi is zero and the bound is a no-op, bit for bit.
    if tr.Pi_p_bounds is not None:
        lo, hi = tr.Pi_p_bounds
        Pi_new = torch.maximum(torch.minimum(Pi_new, hi * p), lo * p)
    return pi_10(pi_new), Pi_new.unsqueeze(1)


# ----------------------------------------------------------------------------- Strang step and rollout


def strang_step(q, pi, Pi, tau: float, dtau: float, grid: Grid, eos, transport: Optional[Transport] = None,
                slope_fn=minmod_slope, source=None, dudtau=None, speed: str = "local"):
    """One tau step:  V(dtau/2) -> ideal Heun(dtau) [+ jet deposit] -> V(dtau/2).
    Returns (q, pi, Pi, dudtau_estimate).  transport=None (or pi=None) -> ideal."""
    viscous = transport is not None and pi is not None
    if viscous:
        prim = primitive_recovery(q, pi, Pi, tau, eos)
        pi, Pi = viscous_step(pi, Pi, prim, dudtau, tau, 0.5 * dtau, grid, eos, transport)
    q_new, prim_old = ideal_step(q, pi, Pi, tau, dtau, grid, eos, slope_fn, speed)
    if source is not None:
        q_new = q_new + source.step(tau, dtau, grid, prim_old)
    tau_new = tau + dtau
    prim_new = primitive_recovery(q_new, pi, Pi, tau_new, eos)
    if viscous:
        # d_tau u must be a difference of u's recovered WITH THEIR OWN pi.  u comes out of the
        # Landau match of K^nu = T^{tau nu} - pi^{tau nu}, so when pi moves, u moves; taking
        # (prim_new - prim_old)/dtau, both recovered at the pi left by the first half-step,
        # drops (du/dpi) d_tau pi altogether -- a missing term, O(1) relative wherever pi is
        # relaxing, not a truncation error, and it does not go away with resolution.  Measured
        # against the Marrochio et al. Gubser solution, and read the tense: with the PRE-FIX
        # d_tau u -- the (prim_new - prim_old) one this block replaced -- the shear stalled at
        # 2.5e-01 (order -0.18 from n = 128 to 256) and dragged e and u down with it; with the
        # exact d_tau u, every spatial gradient still numerical, pi converged at order 0.97.
        # What is written below is neither: it is the consistent finite difference, and it is
        # what test_gubser_viscous's ladder measures now (pi^xx order 1.48 at 64->128, 1.28 at
        # 128->256, 0.39 at 256->512 -- see that docstring; the stall is pushed out by two
        # refinements, not removed).  This is also what MUSIC does: MakeDTau differences
        # u_prev and u_curr, each recovered with its own W.
        #   V2 gets the step-so-far difference (incoming pi -> pi after the first half-step);
        #   the next step gets the fully consistent one (incoming pi -> final pi).
        dudtau_v2 = (prim_new["u"] - prim["u"]) / dtau
        pi, Pi = viscous_step(pi, Pi, prim_new, dudtau_v2, tau_new, 0.5 * dtau, grid, eos, transport)
        prim_final = primitive_recovery(q_new, pi, Pi, tau_new, eos)
        dudtau = (prim_final["u"] - prim["u"]) / dtau
    else:
        dudtau = (prim_new["u"] - prim_old["u"]) / dtau
    return q_new, pi, Pi, dudtau


def rollout(q, pi, Pi, tau0: float, tau_end: float, grid: Grid, eos, transport=None, source=None,
            cfl: float = 0.4, dtau_max: Optional[float] = None, record_dtau: Optional[float] = None,
            slope_fn=minmod_slope, speed: str = "local", callback: Optional[Callable] = None):
    """Advance from tau0 to tau_end, recording every `record_dtau` (default: only the end).
    Returns dict: tau (list), q (stack over records), pi, Pi (or None), src (deposit summed
    since previous record; the FNO source channel), plus the final dudtau."""
    tau = float(tau0)
    taus, qs, pis, Pis, srcs = [tau], [q], [pi], [Pi], []
    src_acc = torch.zeros_like(q)
    next_rec = tau + (record_dtau if record_dtau else float("inf"))
    dudtau = None
    while tau < tau_end - 1e-12:
        nominal = grid.max_dtau(tau, cfl)
        if dtau_max is not None:
            nominal = min(nominal, dtau_max)
        rem = min(tau_end - tau, next_rec - tau)
        dtau = min(nominal, rem)
        if 0.0 < rem - dtau < DTAU_ABSORB * nominal:
            dtau = rem          # absorb the stub; see DTAU_ABSORB -- d_tau u divides by dtau
        q, pi, Pi, dudtau = strang_step(q, pi, Pi, tau, dtau, grid, eos, transport, slope_fn, source, dudtau, speed)
        if source is not None and source.last_dq is not None:
            src_acc = src_acc + source.last_dq
        tau += dtau
        if callback is not None:
            callback(tau, q, pi, Pi)
        if tau >= next_rec - 1e-12 or tau >= tau_end - 1e-12:
            taus.append(tau); qs.append(q); pis.append(pi); Pis.append(Pi); srcs.append(src_acc)
            src_acc = torch.zeros_like(q)
            next_rec = tau + (record_dtau if record_dtau else float("inf"))
    out = {"tau": taus, "q": torch.stack(qs), "src": torch.stack(srcs) if srcs else None, "dudtau": dudtau}
    out["pi"] = torch.stack(pis) if pis[0] is not None else None
    out["Pi"] = torch.stack(Pis) if Pis[0] is not None else None
    return out


# ----------------------------------------------------------------------------- Cartesian bookkeeping


def cartesian_four_momentum(q: torch.Tensor, tau: float, grid: Grid) -> torch.Tensor:
    """Total Cartesian (P^t, P^x, P^y, P^z) of the slice, shape (B,4).  Exactly conserved by the
    continuum equations; discretely to truncation order (the Milne per-component sums, not these,
    are what the scheme telescopes exactly)."""
    _, _, ETA = grid.mesh()
    ch, sh = torch.cosh(ETA), torch.sinh(ETA)
    Pt = (ch * q[:, 0:1] + tau * sh * q[:, 3:4]).sum(dim=(1, 2, 3, 4))
    Pz = (sh * q[:, 0:1] + tau * ch * q[:, 3:4]).sum(dim=(1, 2, 3, 4))
    Px = q[:, 1].sum(dim=(1, 2, 3))
    Py = q[:, 2].sum(dim=(1, 2, 3))
    return torch.stack([Pt, Px, Py, Pz], 1) * grid.dV


def cartesian_vector_to_milne_q(dP: torch.Tensor, K: torch.Tensor, tau: float, grid: Grid) -> torch.Tensor:
    """Distribute Cartesian four-momenta dP (B,N,4) with normalised kernels K (B,N,X,Y,Z),
    sum K dV = 1, onto q: returns Delta q (B,4,X,Y,Z) with the per-cell vector conversion
    (J^tau = cosh eta J^t - sinh eta J^z, J^eta = (cosh eta J^z - sinh eta J^t)/tau)."""
    _, _, ETA = grid.mesh()
    ch, sh = torch.cosh(ETA), torch.sinh(ETA)                    # (1,1,1,1,Z)
    Pt, Px, Py, Pz = [dP[:, :, k].view(*dP.shape[:2], 1, 1, 1) for k in range(4)]
    dq_t = (K * (ch * Pt - sh * Pz)).sum(1)
    dq_x = (K * Px).sum(1)
    dq_y = (K * Py).sum(1)
    dq_e = (K * (ch * Pz - sh * Pt)).sum(1) / tau
    return torch.stack([dq_t, dq_x, dq_y, dq_e], 1)


def gaussian_kernel(grid: Grid, xp, yp, etap, sigma_perp: float, sigma_eta: float) -> torch.Tensor:
    """Discretely normalised Gaussian around (xp, yp, etap) [each (B,N)]: sum_cells K dV = 1.
    Kernels centred outside the grid get whatever mass falls inside (renormalised), so the
    caller must mask partons that left the box."""
    X, Y, ETA = grid.mesh()
    xp = xp.view(*xp.shape, 1, 1, 1)
    yp = yp.view(*yp.shape, 1, 1, 1)
    ep = etap.view(*etap.shape, 1, 1, 1)
    K = torch.exp(-((X - xp) ** 2 + (Y - yp) ** 2) / (2.0 * sigma_perp ** 2) - (ETA - ep) ** 2 / (2.0 * sigma_eta ** 2))
    norm = K.sum(dim=(2, 3, 4), keepdim=True) * grid.dV
    return K / norm.clamp_min(1e-300)


# ----------------------------------------------------------------------------- jet sources


def constant_loss(dEdtau: float):
    """dE/dtau = const [GeV/fm]."""
    return lambda E, T, tau: torch.full_like(E, dEdtau)


def power_loss(kappa: float, n: float = 3.0, T_min: float = 0.0):
    """dE/dtau = kappa T^n  (T in GeV, kappa in GeV^{1-n}/fm); switched off below T_min."""
    return lambda E, T, tau: kappa * T.clamp_min(0.0) ** n * (T > T_min)


class JetSource:
    """Continuous energy–momentum deposition by massless partons, batched (B, Nj).

    Kinematics.  A massless parton produced at tau_prod at transverse point (x0, y0) with
    rapidity y and azimuth phi has, in Milne coordinates,
        eta_s(tau) = y   (constant),   x_perp(tau) = x0 + n_phi (tau - tau_prod),
    i.e. it moves purely transversely at unit speed and its own p^eta vanishes.
    (t = tau cosh y  ->  x_perp = x0 + (t/cosh y) n_phi = x0 + tau n_phi.)

    Deposition.  Energy lost over a step, dE = min(E, rate*dtau), is deposited collinearly:
    Cartesian dP^mu = dE (1, cos phi / cosh y, sin phi / cosh y, tanh y), smeared with a
    normalised Gaussian and converted cell by cell to Milne components, so the Cartesian
    four-momentum of medium + parton is conserved to machine precision.

    rate = loss_rate(E, T_local, tau) [GeV/fm]; T_local sampled at the nearest cell.
    Set `active_from` to delay the deposition (e.g. formation time) independently of tau_prod.
    """

    def __init__(self, x0, y0, rapidity, phi, energy, loss_rate: Callable, sigma_perp: float = 0.5,
                 sigma_eta: float = 0.3, tau_prod=0.0, active_from=None):
        t = lambda a: torch.as_tensor(a, dtype=x0.dtype, device=x0.device).expand_as(x0).clone()
        self.x0, self.y0, self.y, self.phi = x0, y0, rapidity, phi
        self.E = t(energy)
        self.E0 = self.E.clone()
        self.tau_prod = t(tau_prod)
        self.active_from = t(active_from) if active_from is not None else self.tau_prod.clone()
        self.loss_rate = loss_rate
        self.sigma_perp, self.sigma_eta = sigma_perp, sigma_eta
        self.deposited = torch.zeros_like(self.E)
        self.last_dq = None

    def positions(self, tau: float):
        s = (tau - self.tau_prod).clamp_min(0.0)
        return self.x0 + torch.cos(self.phi) * s, self.y0 + torch.sin(self.phi) * s

    def _inside(self, grid: Grid, xp, yp):
        return ((xp.abs() <= grid.x.abs().max()) & (yp.abs() <= grid.y.abs().max()) &
                (self.y.abs() <= grid.eta.abs().max()))

    def _local_T(self, grid: Grid, prim, xp, yp):
        if prim is None:
            return torch.ones_like(xp)
        T = prim["T"]                                          # (B,X,Y,Z)
        ix = ((xp - grid.x[0]) / grid.dx).round().long().clamp(0, grid.nx - 1)
        iy = ((yp - grid.y[0]) / grid.dy).round().long().clamp(0, grid.ny - 1)
        ie = ((self.y - grid.eta[0]) / grid.deta).round().long().clamp(0, grid.neta - 1)
        b = torch.arange(T.shape[0], device=T.device).view(-1, 1).expand_as(ix)
        return T[b, ix, iy, ie]

    def step(self, tau: float, dtau: float, grid: Grid, prim=None) -> torch.Tensor:
        """Deposit over [tau, tau+dtau] (midpoint rule); returns Delta q (B,4,X,Y,Z) and updates E."""
        tmid = tau + 0.5 * dtau
        xp, yp = self.positions(tmid)
        active = (self.E > 0) & (tmid >= self.active_from) & self._inside(grid, xp, yp)
        rate = self.loss_rate(self.E, self._local_T(grid, prim, xp, yp), tmid)
        dE = torch.minimum(rate * dtau, self.E) * active
        self.E = self.E - dE
        self.deposited = self.deposited + dE
        chy, thy = torch.cosh(self.y), torch.tanh(self.y)
        dP = torch.stack([dE, dE * torch.cos(self.phi) / chy, dE * torch.sin(self.phi) / chy, dE * thy], -1)
        K = gaussian_kernel(grid, xp, yp, self.y, self.sigma_perp, self.sigma_eta) * active.view(*active.shape, 1, 1, 1)
        self.last_dq = cartesian_vector_to_milne_q(dP, K, tau + dtau, grid)
        return self.last_dq

    def clone(self):
        c = JetSource.__new__(JetSource)
        c.__dict__.update({k: (v.clone() if torch.is_tensor(v) else v) for k, v in self.__dict__.items()})
        return c


def kick_deposit(grid: Grid, tau: float, xp, yp, etap, dP_cart, sigma_perp: float, sigma_eta: float) -> torch.Tensor:
    """Instantaneous deposition of Cartesian four-momenta dP_cart (B,N,4) at (xp, yp, etap) [(B,N)]
    as a smeared Delta q to be *added to the conserved state* at tau:  q <- q + kick.
    This is the 'new initial condition' operation: exactly conservative, Landau-matched by the
    next primitive recovery."""
    K = gaussian_kernel(grid, xp, yp, etap, sigma_perp, sigma_eta)
    return cartesian_vector_to_milne_q(dP_cart, K, tau, grid)


# ----------------------------------------------------------------------------- background initial conditions


def smooth_initial_energy(grid: Grid, e0, R, a2=0.0, psi2=0.0, a3=0.0, psi3=0.0, eta_flat=2.0,
                          sigma_eta=1.0, x_shift=0.0, y_shift=0.0, e_floor: float = 1e-10) -> torch.Tensor:
    """Analytic parametric energy density (B,X,Y,Z) at tau0:
        transverse  e0 exp(-r^2 / (2 R(phi)^2)),  R(phi) = R [1 + a2 cos 2(phi-psi2) + a3 cos 3(phi-psi3)]
        longitudinal plateau |eta| < eta_flat with Gaussian tails of width sigma_eta.
    All parameters may be floats or (B,) tensors.  Use `eccentricity` to calibrate a_n -> eps_n."""
    X, Y, ETA = grid.mesh()
    bt = lambda a: torch.as_tensor(a, dtype=grid.dtype, device=grid.device).reshape(-1, 1, 1, 1)
    x = X[:, 0] - bt(x_shift)
    y = Y[:, 0] - bt(y_shift)
    r = torch.sqrt(x ** 2 + y ** 2)
    phi = torch.atan2(y, x)
    Rphi = bt(R) * (1.0 + bt(a2) * torch.cos(2.0 * (phi - bt(psi2))) + bt(a3) * torch.cos(3.0 * (phi - bt(psi3))))
    perp = torch.exp(-r ** 2 / (2.0 * Rphi ** 2))
    ae = ETA[:, 0].abs()
    longi = torch.exp(-((ae - bt(eta_flat)).clamp_min(0.0)) ** 2 / (2.0 * bt(sigma_eta) ** 2))
    return bt(e0) * perp * longi + e_floor


def initial_state_from_energy(e: torch.Tensor, tau0: float, eos, u=None) -> torch.Tensor:
    """q at tau0 from e (B,X,Y,Z); default flow u = (1,0,0,0) (Bjorken, u^eta = 0)."""
    if u is None:
        u = torch.zeros(e.shape[0], 4, *e.shape[1:], dtype=e.dtype, device=e.device)
        u[:, 0] = 1.0
    return conserved_from_primitives(e, u, tau0, eos)


def eccentricity(e: torch.Tensor, grid: Grid, n: int) -> torch.Tensor:
    """eps_n = |sum e r^n e^{i n phi}| / sum e r^n  (transverse, eta-integrated), per batch."""
    X, Y, _ = grid.mesh()
    r = torch.sqrt(X ** 2 + Y ** 2)[:, 0]
    phi = torch.atan2(Y, X)[:, 0]
    w = e * r ** n
    num = torch.sqrt((w * torch.cos(n * phi)).sum(dim=(1, 2, 3)) ** 2 + (w * torch.sin(n * phi)).sum(dim=(1, 2, 3)) ** 2)
    return num / w.sum(dim=(1, 2, 3)).clamp_min(1e-300)


def paired_rollout(q0, pi0, Pi0, tau0, tau_end, grid, eos, source: JetSource, transport=None, **kw):
    """Run the same background with and without the jet from identical initial data.
    Returns (run_nojet, run_jet); the wake is run_jet['q'] - run_nojet['q'] at each record."""
    base = rollout(q0.clone(), None if pi0 is None else pi0.clone(), None if Pi0 is None else Pi0.clone(),
                   tau0, tau_end, grid, eos, transport, None, **kw)
    jet = rollout(q0.clone(), None if pi0 is None else pi0.clone(), None if Pi0 is None else Pi0.clone(),
                  tau0, tau_end, grid, eos, transport, source, **kw)
    return base, jet


# ----------------------------------------------------------------------------- example generator


def example_generator(B: int = 2, n: int = 48, neta: int = 24, device: str = "cpu", dtype=torch.float32,
                      tau0: float = 0.6, tau_end: float = 6.0, record_dtau: float = 0.4, seed: int = 0):
    """Minimal end-to-end sample of the training generator: a batch of parametric backgrounds,
    one dijet per event with a temperature-dependent loss rate, paired (jet / no-jet) rollouts,
    and the per-record source channel.  Returns (base, jet, source, grid, eos)."""
    torch.manual_seed(seed)
    dev = torch.device(device)
    g = Grid(n, n, neta, 20.0 / n, 20.0 / n, 8.0 / neta, dev, dtype)
    eos = ConformalEoS()
    e0 = 15.0 + 10.0 * torch.rand(B, dtype=dtype, device=dev)
    a2 = 0.10 * torch.rand(B, dtype=dtype, device=dev)
    psi2 = math.pi * torch.rand(B, dtype=dtype, device=dev)
    e = smooth_initial_energy(g, e0=e0, R=3.0, a2=a2, psi2=psi2, eta_flat=1.5, sigma_eta=0.8)
    q0 = initial_state_from_energy(e, tau0, eos)
    # dijet: back-to-back, produced near the centre (in a real sampler: from T_A T_B of the event)
    x0 = 1.5 * torch.randn(B, 1, dtype=dtype, device=dev).repeat(1, 2)
    y0 = 1.5 * torch.randn(B, 1, dtype=dtype, device=dev).repeat(1, 2)
    phi1 = 2 * math.pi * torch.rand(B, 1, dtype=dtype, device=dev)
    phi = torch.cat([phi1, phi1 + math.pi], 1)
    rap = 0.5 * torch.randn(B, 1, dtype=dtype, device=dev).repeat(1, 2)
    src = JetSource(x0, y0, rap, phi, energy=30.0, loss_rate=power_loss(kappa=30.0, n=3.0, T_min=0.16),
                    sigma_perp=0.6, sigma_eta=0.4, tau_prod=0.0, active_from=tau0)
    base, jet = paired_rollout(q0, None, None, tau0, tau_end, g, eos, src, record_dtau=record_dtau)
    return base, jet, src, g, eos


# ============================================================================= self-tests
# Every test uses analytically derivable ground truth.  Run: python structure_preserving_hydro_fv.py --selftest


def gubser_ideal(tau, X, Y, q: float, e_hat: float):
    """Ideal conformal Gubser flow (Gubser 2010):
        e = e_hat (2q)^{8/3} / ( tau^{4/3} [1 + 2 q^2 (tau^2 + r^2) + q^4 (tau^2 - r^2)^2]^{4/3} ),
        v_perp = 2 q^2 tau r / (1 + q^2 tau^2 + q^2 r^2),  u^eta = 0."""
    r2 = X ** 2 + Y ** 2
    r = torch.sqrt(r2)
    e = e_hat * (2 * q) ** (8.0 / 3.0) / (tau ** (4.0 / 3.0) * (1 + 2 * q ** 2 * (tau ** 2 + r2) + q ** 4 * (tau ** 2 - r2) ** 2) ** (4.0 / 3.0))
    v = 2 * q ** 2 * tau * r / (1 + q ** 2 * tau ** 2 + q ** 2 * r2)
    ut = 1.0 / torch.sqrt(1.0 - v ** 2)
    ux = ut * v * X / r.clamp_min(1e-300)
    uy = ut * v * Y / r.clamp_min(1e-300)
    return e, ut, ux, uy


def uniform_cartesian_flow(tau, ETA, e0: float, u_cart):
    """Constant Cartesian u^mu = (u^t, u^x, u^y, u^z) with e = e0: an exact solution with u^eta != 0.
    u^tau = cosh eta u^t - sinh eta u^z,  u^eta = (cosh eta u^z - sinh eta u^t)/tau."""
    ut_c, ux_c, uy_c, uz_c = u_cart
    ch, sh = torch.cosh(ETA), torch.sinh(ETA)
    u_tau = ch * ut_c - sh * uz_c
    u_eta = (ch * uz_c - sh * ut_c) / tau
    return u_tau, torch.full_like(u_tau, ux_c), torch.full_like(u_tau, uy_c), u_eta


def _rel(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-300)).item()


def _full(g: Grid, a: torch.Tensor, B: int = 1) -> torch.Tensor:
    """Broadcast a (1,1,·,·,·) mesh-shaped field to (B, X, Y, Z)."""
    return a.expand(B, 1, g.nx, g.ny, g.neta)[:, 0].clone()


def test_recovery_roundtrip(dev, dt):
    torch.manual_seed(0)
    g = Grid(6, 5, 7, 0.3, 0.3, 0.2, dev, dt)
    eos = ConformalEoS()
    X, Y, ETA = g.mesh()
    e = _full(g, 5.0 * torch.exp(-(X ** 2 + Y ** 2) / 2.0 - ETA ** 2 / 3.0), 2) + 0.1
    tau = 1.3
    u_tau, ux, uy, ue = uniform_cartesian_flow(tau, ETA, 1.0, (math.sqrt(1 + 0.3 ** 2 + 0.2 ** 2 + 0.5 ** 2), 0.3, 0.2, 0.5))
    u = torch.stack([_full(g, a, 2) for a in (u_tau, ux, uy, ue)], 1)
    pi4 = 0.05 * torch.randn(2, 4, 4, *e.shape[1:], dtype=dt, device=dev)
    pi4 = project_transverse_traceless(0.5 * (pi4 + pi4.transpose(1, 2)), u, tau) * e.unsqueeze(1).unsqueeze(1)
    pi = pi_10(pi4)
    Pi = -0.02 * e.unsqueeze(1)
    q = conserved_from_primitives(e, u, tau, eos, pi, Pi)
    prim = primitive_recovery(q, pi, Pi, tau, eos)
    err = max(_rel(prim["e"], e), _rel(prim["u"], u))
    # constraints on the projected pi
    gl = torch.tensor([-1, 1, 1, tau ** 2], dtype=dt, device=dev).view(1, 4, 1, 1, 1)
    ortho = torch.einsum("bm...,bmn...->bn...", u * gl, pi4).abs().max().item()
    trace = (pi4[:, 0, 0] * -1 + pi4[:, 1, 1] + pi4[:, 2, 2] + pi4[:, 3, 3] * tau ** 2).abs().max().item()
    print(f"[recovery]   round-trip rel err {err:.2e}   |u.pi| {ortho:.1e}   |trace| {trace:.1e}")
    return err < 1e-11 and ortho < 1e-12 and trace < 1e-12


def test_telescoping(dev, dt):
    torch.manual_seed(1)
    g = Grid(9, 7, 8, 0.25, 0.3, 0.2, dev, dt)
    eos = ConformalEoS()
    X, Y, ETA = g.mesh()
    e = _full(g, 3.0 * torch.exp(-(X ** 2 + Y ** 2) / 2.0 - ETA ** 2), 2) + 0.05
    tau = 0.9
    u = torch.zeros(2, 4, *e.shape[1:], dtype=dt, device=dev)
    u[:, 1] = _full(g, 0.3 * torch.sin(X), 2); u[:, 3] = _full(g, 0.2 * torch.sin(ETA), 2) / tau
    u[:, 0] = torch.sqrt(1 + u[:, 1] ** 2 + u[:, 2] ** 2 + tau ** 2 * u[:, 3] ** 2)
    q = conserved_from_primitives(e, u, tau, eos)
    prim = primitive_recovery(q, None, None, tau, eos)
    worst = 0.0
    for axis, dx in ((2, g.dx), (3, g.dy), (4, g.deta)):
        for sl in (minmod_slope(q, axis), 0.3 * torch.randn_like(q) * q.abs()):    # random slopes too
            div, Flo, Fhi = flux_divergence_along(q, prim, sl, axis, dx, tau, eos, return_boundary=True)
            resid = (div.sum(dim=axis, keepdim=True) * dx + Fhi - Flo).abs().max() / Fhi.abs().max()
            worst = max(worst, resid.item())
    print(f"[telescoping] worst |sum div dx + F_hi - F_lo| / |F| = {worst:.1e}  (slope-independent)")
    return worst < 1e-12


def test_bjorken_ideal(dev, dt):
    g = Grid(4, 4, 4, 1.0, 1.0, 0.5, dev, dt)
    eos = ConformalEoS()
    e0, tau0, tau1 = 20.0, 0.5, 3.0
    e = torch.full((1, 4, 4, 4), e0, dtype=dt, device=dev)
    q = initial_state_from_energy(e, tau0, eos)
    out = rollout(q, None, None, tau0, tau1, g, eos, dtau_max=0.002)
    e_num = primitive_recovery(out["q"][-1], None, None, tau1, eos)["e"]
    exact = e0 * (tau0 / tau1) ** (4.0 / 3.0)
    err = _rel(e_num, torch.full_like(e_num, exact))
    print(f"[Bjorken]    e(tau=3)/exact - 1 = {err:.2e}   (dtau = 0.002, Heun)")
    return err < 1e-6


def test_gubser_ideal(dev, dt, speed="local"):
    eos = ConformalEoS()
    qg, e_hat, tau0, tau1, L = 0.5, 30.0, 1.0, 2.0, 16.0
    errs = []
    for n in (32, 64):
        g = Grid(n, n, 3, L / n, L / n, 0.5, dev, dt)
        X, Y, _ = g.mesh()
        e, ut, ux, uy = [_full(g, a) for a in gubser_ideal(tau0, X, Y, qg, e_hat)]
        u = torch.stack([ut, ux, uy, torch.zeros_like(ut)], 1)
        q = conserved_from_primitives(e, u, tau0, eos)
        out = rollout(q, None, None, tau0, tau1, g, eos, speed=speed)
        prim = primitive_recovery(out["q"][-1], None, None, tau1, eos)
        e_ex, _, ux_ex, _ = [_full(g, a) for a in gubser_ideal(tau1, X, Y, qg, e_hat)]
        m = _full(g, (torch.sqrt(X ** 2 + Y ** 2) < 5.0).to(dt))
        err_e = ((prim["e"] - e_ex).abs() * m).sum() / (e_ex.abs() * m).sum()
        err_u = ((prim["u_x"] - ux_ex).abs() * m).sum() / (ux_ex.abs() * m).sum()
        errs.append((err_e.item(), err_u.item()))
    rate_e = math.log2(errs[0][0] / errs[1][0]); rate_u = math.log2(errs[0][1] / errs[1][1])
    print(f"[Gubser {speed:5s}] L1 rel err e: {errs[0][0]:.2e} -> {errs[1][0]:.2e} (order {rate_e:.2f});"
          f"  u^x: {errs[0][1]:.2e} -> {errs[1][1]:.2e} (order {rate_u:.2f})")
    return rate_e > 0.8 and rate_u > 0.8 and errs[1][0] < 3e-2


def test_uniform_cartesian_flow(dev, dt):
    eos = ConformalEoS()
    e0, tau0, tau1 = 4.0, 1.0, 1.4
    u_c = (math.sqrt(1 + 0.2 ** 2 + math.sinh(0.3) ** 2), 0.2, 0.0, math.sinh(0.3))
    errs = []
    for n in (32, 64):
        g = Grid(3, 3, n, 0.5, 0.5, 5.0 / n, dev, dt)
        _, _, ETA = g.mesh()
        ut, ux, uy, ue = [_full(g, a) for a in uniform_cartesian_flow(tau0, ETA, e0, u_c)]
        u = torch.stack([ut, ux, uy, ue], 1)
        e = torch.full_like(ut, e0)
        q = conserved_from_primitives(e, u, tau0, eos)
        out = rollout(q, None, None, tau0, tau1, g, eos)
        prim = primitive_recovery(out["q"][-1], None, None, tau1, eos)
        ut1, _, _, ue1 = [_full(g, a) for a in uniform_cartesian_flow(tau1, ETA, e0, u_c)]
        m = _full(g, (ETA.abs() < 1.5).to(dt))            # causally disconnected from the eta boundaries
        err_e = ((prim["e"] - e0).abs() * m).max() / e0
        err_ue = ((prim["u_eta"] - ue1).abs() * m).max() / ue1.abs().max()
        errs.append((err_e.item(), err_ue.item()))
    rate_e = math.log2(errs[0][0] / errs[1][0]); rate_u = math.log2(errs[0][1] / errs[1][1])
    print(f"[eta-sector] uniform Cartesian flow (u^eta != 0): max rel err e {errs[0][0]:.2e} -> {errs[1][0]:.2e} "
          f"(order {rate_e:.2f}); u^eta {errs[0][1]:.2e} -> {errs[1][1]:.2e} (order {rate_u:.2f})")
    return rate_e > 1.5 and rate_u > 1.5


def test_kinematics_uniform_flow(dev, dt):
    """sigma = theta = 0 for constant Cartesian u, with *analytic* derivatives fed in — checks the
    Christoffel bookkeeping in shear_and_expansion to machine precision (Bjorken values too)."""
    g = Grid(3, 3, 9, 0.5, 0.5, 0.3, dev, dt)
    _, _, ETA = g.mesh()
    tau = 1.7
    u_c = (math.sqrt(1 + 0.25 ** 2 + math.sinh(0.4) ** 2), 0.25, 0.0, math.sinh(0.4))
    ut, ux, uy, ue = [_full(g, a) for a in uniform_cartesian_flow(tau, ETA, 1.0, u_c)]
    u = torch.stack([ut, ux, uy, ue], 1)
    ch, sh = _full(g, torch.cosh(ETA)), _full(g, torch.sinh(ETA))
    d_tau = torch.stack([torch.zeros_like(ut), torch.zeros_like(ut), torch.zeros_like(ut), -ue / tau], 1)
    d_eta = torch.stack([sh * u_c[0] - ch * u_c[3], torch.zeros_like(ut), torch.zeros_like(ut), (sh * u_c[3] - ch * u_c[0]) / tau], 1)
    dU = torch.stack([d_tau, torch.zeros_like(d_tau), torch.zeros_like(d_tau), d_eta], 1)
    theta, sig, Du = shear_and_expansion(u, dU, tau)
    err1 = max(theta.abs().max().item(), sig.abs().max().item(), Du.abs().max().item())
    # Bjorken: theta = 1/tau, sigma^{xx} = 1/(3 tau), sigma^{eta eta} = -2/(3 tau^3)
    ub = torch.zeros_like(u); ub[:, 0] = 1.0
    th_b, sig_b, _ = shear_and_expansion(ub, torch.zeros_like(dU), tau)
    err2 = max(abs(th_b.max().item() - 1 / tau), abs(sig_b[:, 1, 1].max().item() - 1 / (3 * tau)),
               abs(sig_b[:, 3, 3].max().item() + 2 / (3 * tau ** 3)), sig_b[:, 0].abs().max().item())
    print(f"[IS algebra] uniform flow: max|theta|,|sigma|,|Du| = {err1:.1e};  Bjorken sigma/theta err = {err2:.1e}")
    return err1 < 1e-12 and err2 < 1e-12


def test_bjorken_is(dev, dt):
    """Bjorken Israel–Stewart vs the reference ODE (phi = pi^xx + pi^yy = -tau^2 pi^{eta eta}):
        de/dtau = -(e+p)/tau + phi/tau,   dphi/dtau = -phi/tau_pi + (4/3)(eta/tau)/tau_pi - (4/3) phi/tau."""
    eos = ConformalEoS()
    tr = Transport(eta_over_s=0.16)
    e0, tau0, tau1 = 30.0, 0.5, 4.0
    g = Grid(3, 3, 3, 1.0, 1.0, 0.5, dev, dt)
    e = torch.full((1, 3, 3, 3), e0, dtype=dt, device=dev)
    q = initial_state_from_energy(e, tau0, eos)
    pi = g.zeros(1, 10); Pi = g.zeros(1, 1)
    out = rollout(q, pi, Pi, tau0, tau1, g, eos, transport=tr, dtau_max=0.004)
    prim = primitive_recovery(out["q"][-1], out["pi"][-1], out["Pi"][-1], tau1, eos)
    phi_num = (out["pi"][-1][0, 4] + out["pi"][-1][0, 7]).mean().item()
    e_num = prim["e"].mean().item()

    def rhs(tau, y):                                   # reference ODE, same coefficient conventions
        e_, phi_ = y
        p_ = e_ / 3
        T_ = (e_ / eos.c) ** 0.25
        s_ = (e_ + p_) / T_
        eta_ = 0.16 * s_ * HBARC
        taupi = max(5 * eta_ / (e_ + p_), tr.tau_min)
        return (-(e_ + p_) / tau + phi_ / tau, -phi_ / taupi + (4 / 3) * (eta_ / tau) / taupi - (4 / 3) * phi_ / tau)

    y, t, h = (e0, 0.0), tau0, 1e-4
    while t < tau1 - 1e-12:
        h = min(h, tau1 - t)
        k1 = rhs(t, y); k2 = rhs(t + h / 2, tuple(a + h / 2 * b for a, b in zip(y, k1)))
        k3 = rhs(t + h / 2, tuple(a + h / 2 * b for a, b in zip(y, k2))); k4 = rhs(t + h, tuple(a + h * b for a, b in zip(y, k3)))
        y = tuple(a + h / 6 * (b1 + 2 * b2 + 2 * b3 + b4) for a, b1, b2, b3, b4 in zip(y, k1, k2, k3, k4)); t += h
    err_e, err_phi = abs(e_num / y[0] - 1), abs(phi_num / y[1] - 1)
    print(f"[Bjorken IS] e rel err {err_e:.2e}, phi rel err {err_phi:.2e}  (e={e_num:.4f} vs {y[0]:.4f}; phi={phi_num:.4e} vs {y[1]:.4e})")
    return err_e < 1e-3 and err_phi < 1e-2


def test_jet_bookkeeping(dev, dt):
    eos = ConformalEoS()
    g = Grid(24, 24, 16, 0.5, 0.5, 0.25, dev, dt)
    tau0 = 1.0
    X, Y, ETA = g.mesh()
    e = smooth_initial_energy(g, e0=[10.0, 12.0], R=3.0, a2=0.1, eta_flat=1.0, sigma_eta=0.7)
    q = initial_state_from_energy(e, tau0, eos)
    # (a) instantaneous kick: Cartesian four-momentum bookkeeping to machine precision
    dP = torch.tensor([[[6.0, 3.0, 1.0, 2.0]], [[4.0, -1.0, 2.0, -1.5]]], dtype=dt, device=dev)
    kick = kick_deposit(g, tau0, torch.tensor([[0.5], [-1.0]], dtype=dt, device=dev),
                        torch.tensor([[0.2], [0.4]], dtype=dt, device=dev), torch.tensor([[0.3], [-0.5]], dtype=dt, device=dev),
                        dP, 0.6, 0.4)
    dPm = cartesian_four_momentum(q + kick, tau0, g) - cartesian_four_momentum(q, tau0, g)
    err_kick = (dPm - dP[:, 0]).abs().max().item() / dP.abs().max().item()
    # (b) continuous source: deposited Cartesian energy/momentum equals the parton's loss
    src = JetSource(x0=torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=dt, device=dev),
                    y0=torch.tensor([[0.0, 0.5], [0.0, -0.5]], dtype=dt, device=dev),
                    rapidity=torch.tensor([[0.4, -0.3], [0.0, 0.8]], dtype=dt, device=dev),
                    phi=torch.tensor([[0.3, 2.0], [1.0, -2.5]], dtype=dt, device=dev),
                    energy=20.0, loss_rate=constant_loss(3.0), sigma_perp=0.6, sigma_eta=0.4)
    got = 0.0; tau = tau0
    for _ in range(12):
        dq = src.step(tau, 0.1, g); tau += 0.1
        got = got + cartesian_four_momentum(dq, tau, g)        # each deposit at its own tau
    dE = src.deposited
    chy, thy = torch.cosh(src.y), torch.tanh(src.y)
    want = torch.stack([dE.sum(1), (dE * torch.cos(src.phi) / chy).sum(1), (dE * torch.sin(src.phi) / chy).sum(1), (dE * thy).sum(1)], 1)
    err_cont = (got - want).abs().max().item() / want.abs().max().item()
    print(f"[jet source] kick bookkeeping err {err_kick:.1e};  continuous-source bookkeeping err {err_cont:.1e}  (deposited {dE.sum().item():.1f} GeV)")
    return err_kick < 1e-12 and err_cont < 1e-12


def test_wake_linearity(dev, dt):
    """Paired rollouts: wake(2E) ~ 2 wake(E) in the linear-response regime; Cartesian energy drift small."""
    eos = ConformalEoS()
    g = Grid(28, 28, 24, 0.45, 0.45, 0.3, dev, dt)          # eta box +-3.6: no outflow, drift = truncation
    tau0, tau1 = 1.0, 2.2
    e = smooth_initial_energy(g, e0=15.0, R=3.0, eta_flat=0.5, sigma_eta=0.4)
    q0 = initial_state_from_energy(e, tau0, eos)
    resp = []
    for E in (1.0, 2.0):
        src = JetSource(x0=torch.tensor([[-1.0]], dtype=dt, device=dev), y0=torch.tensor([[0.0]], dtype=dt, device=dev),
                        rapidity=torch.tensor([[0.2]], dtype=dt, device=dev), phi=torch.tensor([[0.0]], dtype=dt, device=dev),
                        energy=E, loss_rate=constant_loss(E / 1.0), sigma_perp=0.7, sigma_eta=0.45)
        base, jet = paired_rollout(q0, None, None, tau0, tau1, g, eos, src)
        resp.append(jet["q"][-1] - base["q"][-1])
        P0 = cartesian_four_momentum(q0, tau0, g)
        P1 = cartesian_four_momentum(jet["q"][-1], tau1, g)
    ratio = (resp[1].abs().sum() / resp[0].abs().sum()).item()
    drift = ((P1[0, 0] - P0[0, 0] - src.deposited.sum()) / P0[0, 0]).abs().item()
    print(f"[wake]       |wake(2E)|/|wake(E)| = {ratio:.3f} (linear: 2);  Cartesian energy drift {drift:.1e} "
          f"(eta-truncation, ~x10 smaller at half deta)")
    return 1.8 < ratio < 2.2


def test_jet_israel_stewart(dev, dt):
    """The jet source with the viscous sector LIVE -- the combination nothing else covered.

    test_jet_bookkeeping and test_wake_linearity both run ideal (paired_rollout(q0, None, None,
    ...)), so until this gate nothing exercised a deposit landing in a fluid that carries pi.
    That is the interaction worth pinning, because of where the source sits in strang_step: it is
    added to q AFTER ideal_step and BEFORE prim_new, and prim_new is what sets d_tau u.  A
    deposit therefore reaches theta, sigma and the Navier-Stokes target as a Delta u / dtau spike
    in the very step it fires -- and the source itself never writes pi at all
    (PLAN_jet_source.md, J3: injection is in (e, u^mu) only; the viscous channels pass through).

      (a) J5 with pi live: a jet that deposits nothing leaves q AND pi bitwise alone.
      (b) the deposit stays exact in q with pi live.  viscous_step only ever returns (pi, Pi) --
          it does not touch q -- so after one step (q_jet - q_nojet) is the deposited Cartesian
          four-momentum to machine precision.  This is what would break if the source were ever
          moved inside the viscous substep.
      (c) pi RESPONDS, within a measured band.  Since J3 gives the deposit no shear of its own,
          the only route from jet to shear stress is the kinematics of the flow it modifies.
          This pins that the route exists and is about the size it is today (0.145).  It is a
          characterisation, not a sharp guard -- see the mutation note below.
      (d) the coupled run stays finite and pi stays bounded in the live fluid.  The d_tau u spike
          in (b) is the one place this combination could run away; the bound is loose because it
          is a catastrophe check, not a physics gate (production Au+Au reads 0.19-0.20).

    Mutation-tested, because a gate that cannot fail is worth nothing:
      * deposit moved BEFORE ideal_step, so q_new = ideal(q + dq) instead of ideal(q) + dq:
        CAUGHT.  (b) reads 2.8e-02 against a 2.2e-13 tolerance.
      * deposit moved AFTER prim_new, severing the jet from d_tau u in the step it fires:
        NOT CAUGHT, and that is the more useful result.  The pi response moves only 0.145 ->
        0.142, because the deposit is in q, so the NEXT step's prim recovers it and the coupling
        simply returns one step late.  The jet -> shear route is robust to that ordering; do not
        read (c) as defending it.
      Nothing else in the suite notices either mutation -- jet, wake, gubser_visc and bjorken_is
      all still pass -- which is the reason this gate exists.

    What this does NOT establish: that the deposit SHOULD carry no pi.  A parton depositing into
    the medium does generate shear stress directly, and that channel is absent by construction --
    J3 assigns it to the learned operator W, not to the solver.  This gate pins the behaviour the
    code intends, not the physics of the omission.
    """
    eos = ConformalEoS()
    g = Grid(24, 24, 16, 0.5, 0.5, 0.25, dev, dt)
    tau0, tau1 = 1.0, 2.0
    e = smooth_initial_energy(g, e0=[10.0], R=3.0, a2=0.1, eta_flat=1.0, sigma_eta=0.7)
    q0 = initial_state_from_energy(e, tau0, eos)
    tr = Transport(eta_over_s=0.08, zeta_over_s=0.0)
    T = lambda a: torch.tensor([[a]], dtype=dt, device=dev)
    mk = lambda E: JetSource(x0=T(0.0), y0=T(0.0), rapidity=T(0.0), phi=T(0.0), energy=E,
                             loss_rate=constant_loss(3.0), sigma_perp=0.6, sigma_eta=0.4)
    z10, z1 = g.zeros(1, 10), g.zeros(1, 1)

    # (a) an exhausted jet (E = 0) deposits nothing, so nothing may move -- bitwise, pi included
    b_a, n_a = paired_rollout(q0, z10, z1, tau0, 1.2, g, eos, mk(0.0), transport=tr)
    inert = bool(torch.equal(b_a["q"][-1], n_a["q"][-1]) and
                 torch.equal(b_a["pi"][-1], n_a["pi"][-1]))

    # (b) one step, pi live: Delta q is exactly the deposited Cartesian four-momentum
    dtau, src = 0.02, mk(20.0)
    qA, *_ = strang_step(q0.clone(), z10.clone(), z1.clone(), tau0, dtau, g, eos, tr,
                         minmod_slope, None, None)
    qB, *_ = strang_step(q0.clone(), z10.clone(), z1.clone(), tau0, dtau, g, eos, tr,
                         minmod_slope, src, None)
    dE, chy, thy = src.deposited, torch.cosh(src.y), torch.tanh(src.y)
    want = torch.stack([dE.sum(1), (dE * torch.cos(src.phi) / chy).sum(1),
                        (dE * torch.sin(src.phi) / chy).sum(1), (dE * thy).sum(1)], 1)
    # field by field: nothing in the viscous substep touches q, so the difference of the two
    # states IS the source's own last_dq.  This is the cancellation-free form of the check.
    err_field = float(((qB - qA) - src.last_dq).abs().max() / src.last_dq.abs().max())
    # and it arrives as the right Cartesian four-momentum.  This form differences two totals of
    # order P_tot to resolve a deposit of order |want|, so its floor is eps * (P_tot / want),
    # not eps -- about 4e3 here.  Scale by the measured cancellation rather than by a constant
    # tuned until it passed.
    got = (cartesian_four_momentum(qB, tau0 + dtau, g) -
           cartesian_four_momentum(qA, tau0 + dtau, g))
    amp = float(cartesian_four_momentum(qA, tau0 + dtau, g).abs().max() / want.abs().max())
    err_dep = float((got - want).abs().max() / want.abs().max())

    # (c) + (d) the full coupled run
    base, jet = paired_rollout(q0, z10, z1, tau0, tau1, g, eos, mk(20.0), transport=tr)
    piN, piJ = pi_full(base["pi"][-1]), pi_full(jet["pi"][-1])
    resp = float((piJ - piN).abs().max() / piN.abs().max())
    finite = bool(torch.isfinite(jet["q"][-1]).all() and torch.isfinite(jet["pi"][-1]).all())
    pr = primitive_recovery(jet["q"][-1], jet["pi"][-1], jet["Pi"][-1], tau1, eos)
    g4 = torch.tensor([-1.0, 1.0, 1.0, tau1 ** 2], dtype=dt, device=dev)
    rho = ((piJ ** 2 * g4.view(1, 4, 1, 1, 1, 1) * g4.view(1, 1, 4, 1, 1, 1)).sum((1, 2))
           .abs().sqrt() / (pr["e"] + pr["p"]).clamp_min(1e-30))
    live = pr["e"] >= Transport().pi_e_min
    worst = float(rho[live].max()) if bool(live.any()) else 0.0

    # dtype-aware, so this gate holds in float32 too -- one of the few here that does.  (a), (c)
    # and (d) are dtype-independent; only (b) is a floating-point identity.
    # NB the four-momentum tolerance is ~3e-01 in float32 (eps x 4.9e4), which is too loose to
    # be a gate.  That is the honest consequence of the cancellation, not something to tune away:
    # in float32 the field form above is the one carrying the check, and it stays at ~2e-05.
    eps = torch.finfo(dt).eps
    tol_field, tol_dep = 1e3 * eps, 50.0 * amp * eps
    print(f"[jet + IS]   zero-source inert {inert};  deposit in q {err_field:.1e} "
          f"(tol {tol_field:.1e}), as four-momentum {err_dep:.1e} (tol {tol_dep:.1e}, "
          f"cancellation x{amp:.0f});  pi response {resp:.3f} of |pi|;  finite {finite}, "
          f"worst rho {worst:.2f}")
    # two-sided on resp: 0 means the jet stopped reaching the shear sector at all, and a large
    # value means the d_tau u spike is driving pi instead of the physics.  Measured 0.145.
    return (inert and err_field < tol_field and err_dep < tol_dep
            and 0.02 < resp < 1.0 and finite and worst < 5.0)


def test_recovery_iterations(dev, dt):
    """The Newton default must be converged for the working precision, on a TABULATED EoS (the
    conformal one converges by 4 and would not exercise this), and a NaN must leave the table
    lookup as a NaN rather than as an out-of-bounds index."""
    torch.manual_seed(0)
    # a conformal table, so no download is needed; cs2 still comes from torch.gradient of p,
    # which is what makes the tabulated iteration inexact and only linearly convergent
    conf = ConformalEoS()
    e_tab = torch.logspace(-6, 3, 4000, dtype=torch.float64)
    eos = TabulatedEoS(e_tab, conf.p(e_tab), conf.T(e_tab), device=dev, dtype=dt)

    g = Grid(6, 5, 7, 0.3, 0.3, 0.2, dev, dt)
    X, Y, ETA = g.mesh()
    e = _full(g, 5.0 * torch.exp(-(X ** 2 + Y ** 2) / 2.0 - ETA ** 2 / 3.0), 2) + 0.1
    tau = 1.3
    u_tau, ux, uy, ue = uniform_cartesian_flow(
        tau, ETA, 1.0, (math.sqrt(1 + 0.3 ** 2 + 0.2 ** 2 + 0.5 ** 2), 0.3, 0.2, 0.5))
    u = torch.stack([_full(g, a, 2) for a in (u_tau, ux, uy, ue)], 1)
    pi4 = 0.05 * torch.randn(2, 4, 4, *e.shape[1:], dtype=dt, device=dev)
    pi4 = project_transverse_traceless(0.5 * (pi4 + pi4.transpose(1, 2)), u, tau) \
        * e.unsqueeze(1).unsqueeze(1)
    pi, Pi = pi_10(pi4), -0.02 * e.unsqueeze(1)
    q = conserved_from_primitives(e, u, tau, eos, pi, Pi)

    ref = primitive_recovery(q, pi, Pi, tau, eos, n_iter=200)["e"]
    got = primitive_recovery(q, pi, Pi, tau, eos)["e"]                 # the shipped default
    conv = float(((got - ref).abs() / ref.clamp_min(1e-30)).max())
    eps = torch.finfo(dt).eps
    n_def = NEWTON_ITERS.get(dt, NEWTON_ITERS_DEFAULT)

    # the early exit must land on the same answer as the fixed loop
    early = primitive_recovery(q, pi, Pi, tau, eos, n_iter=200, tol=1e-3 * eps)["e"]
    exit_err = float(((early - ref).abs() / ref.clamp_min(1e-30)).max())

    # a NaN must not index the table out of bounds; it must come back as a NaN
    bad = torch.tensor([float("nan"), float("inf"), -1.0, 1.0], dtype=dt, device=dev)
    out = eos.p(bad)
    nan_ok = bool(torch.isnan(out[0])) and bool(torch.isfinite(out[2:]).all())

    print(f"[iterations] tabulated, n_iter={n_def} ({dt}): rel {conv:.2e} (eps {eps:.1e})   "
          f"early-exit rel {exit_err:.2e}   NaN->NaN {nan_ok}")
    return conv < 20 * eps and exit_err < 20 * eps and nan_ok


def test_is_vacuum_stability(dev, dt):
    """The two things that made the Israel-Stewart sector die in vacuum cells.

    (a) primitive_recovery must return a unit four-velocity even where the |v| cap bites --
        u^tau saturates there but u^i = K^i/(w u^tau) does not, and the IS sector differentiates
        u, so an unnormalised u put sigma over the float32 range.
    (b) pi must stay bounded relative to (e + p) in vacuum, where the relaxation target carries
        no scale of its own.  Unregulated it reached 1.9e21 within four steps and then destroyed
        e itself through K^nu = T^{tau nu} - pi^{tau nu}.
    """
    eos = ConformalEoS()

    # (a) a deliberately unphysical cell: almost no energy, plenty of momentum -> M > K^tau
    g = Grid(4, 4, 4, 0.3, 0.3, 0.3, dev, dt)
    tau = 1.5
    q = g.zeros(1, 4)
    q[:, 0] = 1e-10 * tau
    q[:, 1] = 1.0 * tau                       # K^x >> K^tau: no subluminal solution exists
    q[:, 3] = 0.5 * tau
    u = primitive_recovery(q, None, None, tau, eos)["u"]
    norm = -u[:, 0] ** 2 + u[:, 1] ** 2 + u[:, 2] ** 2 + tau ** 2 * u[:, 3] ** 2
    unit = float((norm + 1.0).abs().max())

    # (b) regulate_pi must pull a runaway back to the bound and leave a healthy pi alone
    w = torch.full((1, 4, 4, 4), 2.0, dtype=dt, device=dev)
    big = torch.randn(1, 4, 4, 4, 4, 4, dtype=dt, device=dev) * 1e6
    big = project_transverse_traceless(0.5 * (big + big.transpose(1, 2)),
                                       torch.stack([torch.ones_like(w)] + [torch.zeros_like(w)] * 3, 1), tau)
    reg = regulate_pi(big, w, tau, 1.0)
    gl = torch.tensor([-1.0, 1.0, 1.0, tau ** 2], dtype=dt, device=dev)
    rho = lambda x: ((x ** 2 * gl.view(1, 4, 1, 1, 1, 1) * gl.view(1, 1, 4, 1, 1, 1)).sum((1, 2))
                     .abs().sqrt() / w)
    bounded = float(rho(reg).max())
    small = big * 1e-12
    untouched = bool(torch.equal(regulate_pi(small, w, tau, 1.0), small))

    # (c) an IS evolution on a fireball with vacuum around it must stay finite.  This needs a
    # TABULATED EoS (its lowest table entry floors T, which is what keeps tau_pi short enough in
    # vacuum for pi to be driven at all) and enough rapidity range for the Milne terms to bite:
    # at +-4 the unregulated run goes non-finite by step ~41, at +-1.6 it never develops.
    e_tab = torch.logspace(-6, 3, 4000, dtype=torch.float64)
    tab = TabulatedEoS(e_tab, eos.p(e_tab), eos.T(e_tab), device=dev, dtype=dt)
    g2 = Grid(20, 20, 16, 0.5, 0.5, 0.5, dev, dt)
    X, Y, ETA = g2.mesh()
    e0 = _full(g2, 15.0 * torch.exp(-(X ** 2 + Y ** 2) / 4.0 - ETA ** 2 / 2.0), 1) + 1e-12
    tr = Transport(eta_over_s=0.08, zeta_over_s=0.0)          # bulk off: the shipped default
    qq = initial_state_from_energy(e0, 0.6, tab)
    pi, Pi, t_, dud = g2.zeros(1, 10), g2.zeros(1, 1), 0.6, None
    worst, finite = 0.0, True
    for _ in range(50):
        dtau = g2.max_dtau(t_, 0.4)
        qq, pi, Pi, dud = strang_step(qq, pi, Pi, t_, dtau, g2, tab, tr, minmod_slope, None, dud)
        t_ += dtau
        if not (torch.isfinite(qq).all() and torch.isfinite(pi).all()):
            finite = False
            break
        pr = primitive_recovery(qq, pi, Pi, t_, tab)
        g4 = torch.tensor([-1.0, 1.0, 1.0, t_ ** 2], dtype=dt, device=dev)
        pi4 = pi_full(pi)
        rho_seen = ((pi4 ** 2 * g4.view(1, 4, 1, 1, 1, 1) * g4.view(1, 1, 4, 1, 1, 1)).sum((1, 2))
                    .abs().sqrt() / (pr["e"] + pr["p"]).clamp_min(1e-30))
        # measure in the LIVE fluid only.  Outside it, e + p is recomputed here after the step
        # and can have collapsed by orders of magnitude since the regulator last used it, so the
        # ratio is not well defined there -- pi is re-bounded on the next half-step anyway.
        live = pr["e"] >= Transport().pi_e_min
        if live.any():
            worst = max(worst, float(rho_seen[live].max()))

    # (d) the freeze must actually hold pi in capped cells.  The integration in (c) does not
    # reach the velocity cap, so without this the freeze could be deleted and (c) would pass.
    pr_d = primitive_recovery(qq, pi, Pi, t_, tab)
    pr_d["capped"] = torch.zeros_like(pr_d["e"], dtype=torch.bool)
    pr_d["capped"][..., :4, :, :] = True                  # an arbitrary slab of "capped" cells
    # The regulator is switched off for this sub-check on purpose: it isolates the freeze.  The
    # slab marked "capped" is deep vacuum, where e + p ~ 1e-12, so even this pi is ~1e4 times
    # the bound and the regulator would rescale it AFTER the freeze -- which is the intended
    # order (freeze, then bound) and exactly what part (b) tests.  Here we only ask whether the
    # relaxation update was held off.
    small = pi * 1e-6
    kept, _ = viscous_step(small, Pi, pr_d, None, t_, 1e-3, g2, tab,
                           Transport(eta_over_s=0.08, zeta_over_s=0.0, pi_rho_max=None))
    loose, _ = viscous_step(small, Pi, pr_d, None, t_, 1e-3, g2, tab,
                            Transport(eta_over_s=0.08, zeta_over_s=0.0, pi_e_min=None,
                                      pi_rho_max=None))
    m = pr_d["capped"]
    held = bool(torch.allclose(pi_full(kept)[..., m[0]], pi_full(small)[..., m[0]],
                               rtol=1e-6, atol=1e-30))
    moved = not bool(torch.allclose(pi_full(loose)[..., m[0]], pi_full(small)[..., m[0]],
                                    rtol=1e-6, atol=1e-30))
    print(f"[is_vacuum]  freeze holds pi in capped cells {held} (and it moves without it {moved})")

    print(f"[is_vacuum]  |u.u+1| at the cap {unit:.2e}   regulated rho {bounded:.3f} "
          f"(healthy pi untouched {untouched})   50 IS steps finite {finite}, worst rho {worst:.2f}")
    # `worst` is a catastrophe bound, not a physics gate: this toy is a 4-cell fireball, and its
    # post-step |pi|/(e+p) in cells that have just gone dilute reads 6-7 legitimately (the
    # regulator bounded pi against the e + p of the substep before).  Without the regulator it
    # is 556, then NaN.  On production Au+Au the same measure is 0.19 in the hot fluid.
    return (unit < 1e-5 and bounded <= 1.0 + 1e-5 and untouched and finite and worst < 20.0
            and held and moved)


def test_gubser_viscous(dev, dt):
    """Israel-Stewart against the Marrochio et al. semi-analytic Gubser solution.

    All three fields are gated: e, u^x and the shear stress all have to converge.  Measured
    with the shipped scheme, L1 relative inside |r| < 4 fm, tau 1 -> 2 fm, adaptive CFL:

        n        dx       e        u^x      pi^xx
        64     0.250   7.99e-02  7.52e-02  3.57e-01
        128    0.125   2.70e-02  2.12e-02  1.28e-01
        256    0.062   9.03e-03  7.44e-03  5.26e-02
        512    0.031   4.16e-03  4.35e-03  4.02e-02
        order 64->128     1.57      1.83      1.48
        order 128->256    1.58      1.51      1.28
        order 256->512    1.12      0.77      0.39

    and inside |r| < 1, pi goes 4.05e-01 -> 1.52e-01 -> 4.64e-02 (orders 1.42, 1.71).
    Both tables are `workflow_fastdata/gubser_ladder.py`, which exists so that they can be
    re-measured rather than believed.

    **Read the last row.**  The ladder used to stop at 256, and stopping there says "pi
    converges at ~1.3".  It does not: at 512 pi is stalling at 4.0e-02 and dragging e and u
    down with it -- the same shape as the pre-fix failure in `strang_step`, two refinements
    further out.

    The substep is NOT what is left.  Holding n = 128 and refining only dtau_max:

        dtau     steps      e        u^x      pi^xx
        0.05      20     2.696e-02  2.116e-02  1.282e-01     (= the CFL step at this n)
        0.02      50     2.504e-02  1.986e-02  1.215e-01
        0.01     100     2.462e-02  1.932e-02  1.251e-01
        0.005    200     2.446e-02  1.904e-02  1.276e-01
        0.002    500     2.437e-02  1.887e-02  1.296e-01

    e and u^x fall onto a spatial floor, ~10% below their CFL values.  pi moves by 6% over a
    25x range of dtau, non-monotonically, and settles near 1.3e-01 -- i.e. at this resolution
    pi's error is spatial, and the time integration is not what limits it.  A 1/dtau noise term
    from d_tau u (a difference of Newton-recovered velocities divided by the step) would show
    up here as pi diverging as dtau -> 0.  It does not, at float64: that mechanism is real but
    needs the ~1e-9 fm/c residual step that `evolve.evolve_event` used to leave behind, an
    amplification of 1e7 rather than of 25, which is what the crash in RESULTS_fv_vs_music.md
    was.  Holding dtau at 0.0125 for EVERY n says the same from the other side: the orders
    still collapse, to 0.98 / 0.68 / 0.37.

    Four controls say what it is not (all `workflow_fastdata/gubser_ladder.py`):

        vacuum guards off        bit-identical at 128, 256, 512  -> not the regulator or freeze
        ideal sector, same box   orders 1.74 (e), 1.82 (u^x)     -> not the box, eta, reference,
                                                                    recovery or ideal flux
        pi_advection: upwind     orders 0.87 then 0.55           -> not the advection stencil
        mask tightened to |r|<1  pi orders 1.71 then 1.64        -> not the core

    and a fifth says it is not the way it is being measured.  pi^xx and u^x both change sign
    across the transverse plane, so an L1 error whose reference nearly cancels is suspect;
    measured instead on sum|pi^{ab}| over all ten components and on the transverse speed,
    neither of which has a node, the 256 -> 512 orders are e 1.12, |u_T| 0.72, sum|pi| 0.16 --
    WORSE, not better.  The stall is real.

    Per annulus, node-free metric, 256 -> 512:

        r [fm]      order    share of the total |pi| error
        0.0-1.5     2.75 / 1.54 / 0.97     3.1%
        1.5-2.0    -0.30                  32.7%       <- error GROWS under refinement
        2.0-2.5     0.10                  47.1%
        2.5-4.0     1.38 / 0.65 / -1.38   17.0%

    80% of it is in 1.5 < r < 2.5 fm, and over the previous refinement that same annulus
    converged at 1.52.  An absolute error that grows when the grid is refined is not a
    truncation error; something switches on between those two resolutions.  It is the limiter
    -- see the paragraph below, and PLAN_fv_vs_music_next.md B1 for what to do about it.

    What the stall IS: `minmod_slope`.  Reconstructing with `zero_slope` instead -- no limiter,
    and only first-order accurate -- changes pi's 256 -> 512 order from 0.16 to 1.46, and lands
    at a LOWER absolute error than minmod does (3.56e-02 against 3.86e-02).  A first-order
    scheme beating a second-order one at fine resolution is what a floor looks like, and per
    annulus it is exactly the two rings that were stalling that come back: 1.5-2.0 fm goes
    -0.19 -> 1.34 and 2.0-2.5 fm goes 0.05 -> 3.11.

    Why the shear sector and not the ideal one: minmod returns ZERO wherever the one-sided
    differences disagree in sign, i.e. at every local extremum, so the reconstruction drops to
    first order on a set of cells whose position moves as the grid is refined.  The ideal
    sector only ever integrates that field.  The shear sector DIFFERENTIATES it -- sigma is
    built from d_mu u -- which is also why differencing it more accurately does not help (see
    below), and why `finite_difference_dU` had already noticed that "the Gubser velocity has
    genuine extrema that a limiter clips" about the gradient stencil.  Treat the mechanism as
    the best reading of the evidence rather than as proven: the fraction of cells the limiter
    zeroes falls roughly as dx everywhere (n = 128/256/512: 16.3/8.6/5.6% at 1.5-2.0 fm,
    5.4/2.6/1.3% at 2.0-2.5), so it is not that the limiter fires more and more.

    History, because it is the part worth keeping -- and re-run against the post-fix solver,
    because every row of it was measured while the pre-fix d_tau u dominated and so only ever
    established "not the biggest term in 2025".  Before that fix the shear stalled at 2.5e-01
    with order -0.18 from 128 to 256 and dragged e and u down with it (orders 0.44 and 0.72
    there).  The hunt ruled out, each by measurement:

      - the continuum terms (the assembled RHS matches the analytic d_tau pi at second order
        given exact inputs) and the kinematics (shear_and_expansion reproduces the exact
        Navier-Stokes sigma at clean second order from the exact u).  Both are statements about
        the algebra given exact inputs, so the d_tau u fix cannot have changed them.
      - the vacuum guards (disabling them is bit-identical here) -- RE-VERIFIED, still
        bit-identical at n = 128, 256 and 512.
      - the box (doubling it at fixed dx is bit-identical inside the mask, as locality
        requires) -- RE-VERIFIED, still bit-identical to every digit.
      - the spatial velocity gradients ("freezing them to exact values changed nothing") --
        the CONCLUSION survives and the METHOD does not.  Differencing the solver's own u to
        fourth order instead of second -- 150x more accurate on the initial condition,
        1.0e-03 -> 6.7e-06 -- moves pi's 256 -> 512 order from 0.16 to 0.17.  Nothing.  But
        substituting the *exact* gradients, as that row did, now makes the run diverge
        (orders -2.4, -4.1, -3.0): they belong to a velocity field the solver does not have,
        and the inconsistency is destabilising.  It was only ever readable because the term it
        was competing against was enormous.  Use --dxu-order 4, not --exact-dxu.
      - the Strang split (an unsplit Heun was no better) and the interface treatment of pi in
        the flux (reconstructing it one-sided changed nothing) -- NOT re-run.  Both need a code
        variant rather than a flag, and the limiter result above arrived first.

    What did it *then* was the time derivative: freezing only d_tau u to its exact value
    restored convergence everywhere, and the reason is in strang_step.  That term is fixed, and
    what it was hiding is the limiter.

    All of this is `workflow_fastdata/gubser_ladder.py`: --slope, --dxu-order, --exact-dxu,
    --lbox, --no-guards, --limiter-stats, --rmask, --r-bins, --metric, --dtau-max.

    The reference is independent of the solver: gubser.solve_de_sitter integrates the paper's
    ODEs and gubser_fields maps them to Milne; the conventions line up exactly (tau_pi =
    5 (eta/s) hbarc / T is the paper's tauR_hat = c (eta/s) / That, and both carry delta_pipi =
    4/3 and omit tau_pipi), and the initial state is exact to ~5e-16 in |u.u + 1|,
    |u_mu pi^{mu nu}| and |pi^mu_mu|.  This test runs 32 and 64 to stay inside a minute.
    """
    import numpy as np                      # fv.py itself is numpy-free; this test is not
    from . import gubser as _gub

    eta_s, c_relax, That0, qg = 0.2, 5.0, 1.2, 1.0
    tau0, tau1, Lbox, rmask = 1.0, 2.0, 16.0, 4.0
    That_fn, pibar_fn = _gub.solve_de_sitter(That0=That0, pibar0=0.0, eta_s=eta_s,
                                             c_relax=c_relax)
    eos = ConformalEoS(dof=42.25)                     # == gubser.A_E_DEFAULT

    def fields(tau, g):
        xs = g.x.detach().cpu().numpy()[:, None, None]
        ys = g.y.detach().cpu().numpy()[None, :, None]
        f = _gub.gubser_fields(tau, xs, ys, qg, That_fn, pibar_fn, _gub.A_E_DEFAULT)
        T = lambda a: torch.as_tensor(
            np.ascontiguousarray(np.broadcast_to(a, (g.nx, g.ny, g.neta))),
            dtype=dt, device=dev).unsqueeze(0)
        e = T(f["e"]) * HBARC                          # fm^-4 -> GeV/fm^3
        u = torch.cat([T(f["utau"]), T(f["ux"]), T(f["uy"]), T(f["ueta"])], 0).unsqueeze(0)
        z = torch.zeros_like(e)
        pi10 = torch.cat([T(f["pitautau"]), T(f["pitaux"]), T(f["pitauy"]), z,
                          T(f["pixx"]), T(f["pixy"]), z, T(f["piyy"]), z,
                          T(f["tau2_pietaeta"]) / tau ** 2], 0).unsqueeze(0) * HBARC
        return e, u, pi10

    errs = []
    for n in (32, 64):
        g = Grid(n, n, 3, Lbox / n, Lbox / n, 0.5, dev, dt)
        e0, u0, pi0 = fields(tau0, g)
        Pi0 = torch.zeros_like(e0).unsqueeze(1)
        q = conserved_from_primitives(e0, u0, tau0, eos, pi0, Pi0)
        tr = Transport(eta_over_s=eta_s, zeta_over_s=0.0, tau_pi_coeff=c_relax)
        out = rollout(q, pi0, Pi0, tau0, tau1, g, eos, transport=tr)
        prim = primitive_recovery(out["q"][-1], out["pi"][-1], out["Pi"][-1], tau1, eos)
        eE, uE, pE = fields(tau1, g)
        X, Y, _ = g.mesh()
        m = ((X ** 2 + Y ** 2).sqrt() < rmask).to(dt)
        rel = lambda a, b: float(((a - b).abs() * m).sum() / ((b.abs() * m).sum() + 1e-300))
        errs.append((rel(prim["e"], eE), rel(prim["u_x"], uE[:, 1]),
                     rel(pi_full(out["pi"][-1])[:, 1, 1], pE[:, 4])))
    o = [math.log2(errs[0][k] / errs[1][k]) for k in range(3)]
    print(f"[Gubser IS]  e {errs[0][0]:.2e} -> {errs[1][0]:.2e} (order {o[0]:.2f})   "
          f"u^x {errs[0][1]:.2e} -> {errs[1][1]:.2e} (order {o[1]:.2f})")
    print(f"[Gubser IS]  pi^xx {errs[0][2]:.2e} -> {errs[1][2]:.2e} (order {o[2]:.2f})")
    return (o[0] > 0.8 and errs[1][0] < 0.15 and
            o[1] > 0.8 and errs[1][1] < 0.15 and
            o[2] > 0.8 and errs[1][2] < 0.6)


def run_selftests(device="cpu", dtype=None):
    """Run the full physics suite.  The tolerances are tuned for float64; MPS has no float64,
    so a float32 run there is a smoke test, not a physics gate (use tests/test_fast_data_device.py
    for backend coverage).

    A float32 run prints a banner saying so, because the bare FAIL list reads like a broken port
    and is not one: on MPS, recovery/telescoping/bjorken/is_algebra/jet fail at ~1e-6 and
    is_vacuum at 7.3e-04 on |u.u+1|, and CPU float32 reproduces every one at the same magnitude.
    is_vacuum's `worst rho` is worth singling out -- in float32 that deep-vacuum toy is chaotic
    (cpu 4.4e+03 / mps 3.6e+01 before the d_tau u fix, cpu 8.6e+02 / mps 9.4e+02 after, against
    1.2 and 7.0 in float64), so the `worst < 20` bound carries no meaning outside float64.  The
    physics it stands in for is measured on production Au+Au instead, where |pi|/(e+p) in the hot
    fluid is 0.200 on cpu/float64, cpu/float32 and mps/float32 alike.

    The float32 run does still return False.  Reporting success there would mask a genuine
    backend regression; the banner is for reading the output, not for softening the gate."""
    dev = torch.device(device)
    if dtype is None:
        dtype = torch.float32 if dev.type == "mps" else torch.float64
    dt = dtype
    if dt == torch.float32:
        print(f"NOTE: running in float32 on {dev.type}.  Every tolerance below is tuned for "
              f"float64, so the\n      gates that measure an exact identity -- recovery, "
              f"telescoping, bjorken, is_algebra, jet,\n      and is_vacuum's |u.u+1| -- report "
              f"FAIL at ~1e-6 here purely from rounding.  That is\n      expected, and the exit "
              f"status will be non-zero.  To tell a float32 effect from a BACKEND\n      bug, run "
              f"the same suite with --device cpu --dtype float32: anything that fails there too\n"
              f"      is the dtype, not {dev.type}.  For real backend coverage use "
              f"tests/test_fast_data_device.py.\n")
    previous_default = torch.get_default_dtype()
    torch.set_default_dtype(dt)          # restored in the finally below, so importing and
    try:                                 # running this does not change global state for anyone else
        results = {
            "recovery": test_recovery_roundtrip(dev, dt),
            "iterations": test_recovery_iterations(dev, dt),
            "is_vacuum": test_is_vacuum_stability(dev, dt),
            "telescoping": test_telescoping(dev, dt),
            "bjorken": test_bjorken_ideal(dev, dt),
            "gubser_local": test_gubser_ideal(dev, dt, "local"),
            "gubser_light": test_gubser_ideal(dev, dt, "light"),
            "gubser_visc": test_gubser_viscous(dev, dt),
            "eta_sector": test_uniform_cartesian_flow(dev, dt),
            "is_algebra": test_kinematics_uniform_flow(dev, dt),
            "bjorken_is": test_bjorken_is(dev, dt),
            "jet": test_jet_bookkeeping(dev, dt),
            "wake": test_wake_linearity(dev, dt),
            "jet_is": test_jet_israel_stewart(dev, dt),
        }
    finally:
        torch.set_default_dtype(previous_default)
    print("\n" + "  ".join(f"{k}:{'PASS' if v else 'FAIL'}" for k, v in results.items()))
    if dt == torch.float32 and not all(results.values()):
        bad = [k for k, v in results.items() if not v]
        print(f"      ^ float32: {len(bad)} FAIL -- read the NOTE above before calling this a "
              f"regression.\n        {', '.join(bad)}\n        Compare against --device cpu "
              f"--dtype float32 first: what fails there too is the dtype.")
    # deliberately strict: a float32 run still returns False, so this can never mask a real
    # backend regression behind a dtype excuse.  The banner explains the exit status instead.
    return all(results.values())


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(
        prog="fast_data.fv",
        description="Structure-preserving finite-volume 3+1D Milne hydro (PyTorch).")
    ap.add_argument("--selftest", action="store_true", help="run the physics self-test suite")
    ap.add_argument("--device", default=None,
                    help="cpu | cuda | mps (default: cuda if --cuda, else cpu)")
    ap.add_argument("--cuda", action="store_true", help="shorthand for --device cuda")
    ap.add_argument("--dtype", default=None, choices=["float32", "float64"],
                    help="default: float64, or float32 on mps which has no float64")
    args = ap.parse_args(argv)

    if not args.selftest:
        print(__doc__)
        return 0

    device = args.device or ("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    dtype = {"float32": torch.float32, "float64": torch.float64}.get(args.dtype)
    return 0 if run_selftests(device, dtype) else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
