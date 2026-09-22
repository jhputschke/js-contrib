"""Turning one droplet into a Delta q patch on the hydro grid.

Why this file is not a one-liner
--------------------------------
The kernel is normalised in the continuum (``int d^3r K_rho = 1``), but the deposit has two
features far below the grid scale.  On the production grid (65x65x33, d = 0.3125 fm):

* the wave-front shell is ``width_delta = 0.1 fm`` thick, a third of a cell, and it carries
  ``f_delta(t) = (1 + g t + (g t)^2/2) e^{-g t}`` of the momentum -- 12% at Delta t = 1 fm, 54% at
  0.5 fm, 92% at 0.2 fm;
* for a droplet at large |eta_d| the ENTIRE causal support is sub-cell in eta, because the deposit
  happens over a short lab interval while ``dz = tau cosh(eta) deta`` is huge out there.

Sampling the kernel at cell centres therefore does not merely lose accuracy, it loses the deposit:
measured on that grid at tau_d = 1, deposit at tau = 3, the normalisation
``N = sum tau dV K_tau`` comes out 1.00 / 1.01 / 2.56 / 0.00 / 0.00 at eta_d = 0 / 1 / 2 / 3 / 4.
Refining naively does not fix it either -- at eta_d = 4 the sequence over n_sub = 1,2,4,8,16 is
0.00, 0.00, 0.10, 1.00, 0.11: aliasing, not convergence, because the nodes keep missing a support
thinner than the cell.

Three mechanisms, in order of importance
----------------------------------------
1. **Clip the quadrature to the support before placing nodes.**  Each cell's integration interval
   is the cell intersected with the causal support, and the cell weight carries the resulting
   volume fraction.  Nodes then always land inside the support, however thin it is.
2. **Sub-cell fallback.**  If the whole support fits inside one cell, quadrature is pointless:
   scatter the full weight to the support centroid with a trilinear (CIC) kernel.  That is the
   exact answer in that regime, and it removes the aliasing lottery entirely.
3. **One scalar renormalisation per droplet**, so that ``sum_cells tau dV K_tau = 1`` exactly.
   Because that single scalar multiplies all four components, Cartesian four-momentum conservation
   is then exact to round-off regardless of how well the shape is resolved (see the identity in
   `milne_dq_from_cartesian`).  The pre-renormalisation value `n_raw` is recorded, and is the
   honest measure of the shape error.

Conservation comes from (3); (1) and (2) only buy shape.  Keeping those separate is what makes the
deposit trustworthy on a grid that cannot resolve it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import kernel as _K
from .droplets import DropletFlags
from .params import LiquefierParams

__all__ = ["DepositPatch", "support_box", "droplet_weights", "milne_dq_from_cartesian"]


@dataclass
class DepositPatch:
    """Normalised cell weights for one droplet on one grid, plus the diagnostics that say how
    much to trust them."""

    ix: slice
    iy: slice
    ie: slice
    w: np.ndarray                    # (nx, ny, ne); sums to 1 after renormalisation
    tau_eval: float
    n_raw: float = 1.0               # sum before renormalisation: the shape/quadrature error
    in_grid_fraction: float = 1.0
    flags: int = 0
    shell_fraction: float = 0.0
    eta_centroid: float = 0.0
    n_cells: int = 0
    eta_nodes: np.ndarray = field(default=None, repr=False)   # eta of the patch's eta-slice

    @property
    def is_empty(self):
        return self.w.size == 0 or not np.any(self.w)


def _axis_cells(centres, d, lo, hi):
    """Indices of cells whose extent [c-d/2, c+d/2) overlaps [lo, hi], as a slice."""
    idx = np.nonzero((centres + 0.5 * d > lo) & (centres - 0.5 * d < hi))[0]
    if idx.size == 0:
        return slice(0, 0)
    return slice(int(idx[0]), int(idx[-1]) + 1)


def support_box(drop, tau_eval, grid, p: LiquefierParams, n_scan=2048):
    """The causal support of one droplet on the tau_eval slice.

    Returns (eta_lo, eta_hi, r_max, ok).  The support is
    ``{eta : |z(eta) - z_d| < c (t(eta) - t_d)}`` with ``t = tau cosh eta``, ``z = tau sinh eta``;
    the transverse extent at each eta is a disk of radius ``sqrt(R^2 - (z-z_d)^2)``.
    """
    tau_d, eta_d = float(drop[0]), float(drop[3])
    t_d, z_d = tau_d * np.cosh(eta_d), tau_d * np.sinh(eta_d)
    c = p.c_diff

    eta_all = np.asarray(grid.eta.detach().cpu(), dtype=np.float64)
    lo, hi = float(eta_all[0]) - float(grid.deta), float(eta_all[-1]) + float(grid.deta)
    # widen the scan so a support outside the grid is still found (for in_grid_fraction)
    span = max(abs(eta_d) + 1.0, abs(lo), abs(hi))
    e = np.linspace(-span, span, n_scan)
    t, z = tau_eval * np.cosh(e), tau_eval * np.sinh(e)
    dt, dz = t - t_d, z - z_d
    R = c * dt
    inside = (R > 0) & (np.abs(dz) < R)
    if not inside.any():
        return 0.0, 0.0, 0.0, False
    idx = np.nonzero(inside)[0]
    # refine the two edges by bisection so the box is tight but a true superset
    step = e[1] - e[0]
    eta_lo = e[idx[0]] - step
    eta_hi = e[idx[-1]] + step
    rmax = float(np.max(np.sqrt(np.maximum(R[inside] ** 2 - dz[inside] ** 2, 0.0))))
    return float(eta_lo), float(eta_hi), rmax, True


def _gauss_nodes(lo, hi, n):
    """n midpoint nodes on [lo, hi] and the interval length (midpoint rule, which is what the
    C++ conservation test uses and is second-order without endpoint bias)."""
    n = max(int(n), 1)
    h = (hi - lo) / n
    return lo + (np.arange(n) + 0.5) * h, (hi - lo)


def droplet_weights(drop, tau_eval, grid, p: LiquefierParams, *, mode="conservative",
                    n_sub="auto", n_sub_max=24, shell="widen", renorm="grid",
                    min_in_grid=0.99, subcell_beta=1.0):
    """-> DepositPatch.  `mode="xscape"` reproduces the C++ cell-centre point sampling."""
    ex = np.asarray(grid.x.detach().cpu(), dtype=np.float64)
    ey = np.asarray(grid.y.detach().cpu(), dtype=np.float64)
    ee = np.asarray(grid.eta.detach().cpu(), dtype=np.float64)
    dx, dy, de = float(grid.dx), float(grid.dy), float(grid.deta)
    tau_d, x_d, y_d, eta_d = (float(drop[i]) for i in range(4))
    flags = 0

    if mode == "xscape":
        # verbatim point sampling: no clipping, no renormalisation, no shell regularisation
        X, Y, E = np.meshgrid(ex, ey, ee, indexing="ij")
        kt = _K.k_tau(tau_eval, X, Y, E, drop, p, strict_xscape=True)
        w = tau_eval * dx * dy * de * np.nan_to_num(kt, nan=0.0, posinf=0.0, neginf=0.0)
        n_raw = float(w.sum())
        return DepositPatch(slice(None), slice(None), slice(None), w, tau_eval,
                            n_raw=n_raw, in_grid_fraction=1.0, flags=flags,
                            n_cells=int(w.size), eta_nodes=ee)

    eta_lo, eta_hi, r_max, ok = support_box(drop, tau_eval, grid, p)
    if not ok or r_max <= 0:
        return DepositPatch(slice(0, 0), slice(0, 0), slice(0, 0),
                            np.zeros((0, 0, 0)), tau_eval, n_raw=0.0, in_grid_fraction=0.0,
                            flags=flags, n_cells=0, eta_nodes=ee[:0])

    dt_typ = max(tau_eval * np.cosh(0.5 * (eta_lo + eta_hi)) - tau_d * np.cosh(eta_d), 1e-12)
    shell_frac = float(_K.shell_mass_fraction(dt_typ, p))
    if shell_frac > 0.2:
        flags |= DropletFlags.SHELL_DOMINATED

    # --- (2) sub-cell fallback: the whole support lives inside one cell -------------------
    sub_eta = (eta_hi - eta_lo) < subcell_beta * de
    sub_xy = (2.0 * r_max) < subcell_beta * min(dx, dy)
    if sub_eta and sub_xy:
        flags |= DropletFlags.SUBCELL
        return _cic_patch(drop, tau_eval, grid, 0.5 * (eta_lo + eta_hi), flags,
                          shell_frac, min_in_grid, renorm)

    # --- cell box ------------------------------------------------------------------------
    ix = _axis_cells(ex, dx, x_d - r_max, x_d + r_max)
    iy = _axis_cells(ey, dy, y_d - r_max, y_d + r_max)
    ie = _axis_cells(ee, de, eta_lo, eta_hi)
    if ix.stop == ix.start or iy.stop == iy.start or ie.stop == ie.start:
        # the support misses the grid entirely
        flags |= DropletFlags.OUT_OF_GRID
        return DepositPatch(slice(0, 0), slice(0, 0), slice(0, 0), np.zeros((0, 0, 0)),
                            tau_eval, n_raw=0.0, in_grid_fraction=0.0, flags=flags,
                            shell_fraction=shell_frac, n_cells=0, eta_nodes=ee[:0])

    cx, cy, ce = ex[ix], ey[iy], ee[ie]
    w = _quadrature(cx, cy, ce, dx, dy, de, drop, tau_eval, p,
                    n_sub=n_sub, n_sub_max=n_sub_max, shell=shell,
                    xlim=(x_d - r_max, x_d + r_max), ylim=(y_d - r_max, y_d + r_max),
                    elim=(eta_lo, eta_hi))
    n_grid = float(w.sum())

    # --- in-grid fraction, on a virtual lattice covering the whole support ----------------
    n_ext = _extended_norm(drop, tau_eval, grid, p, eta_lo, eta_hi, r_max,
                           n_sub=n_sub, n_sub_max=n_sub_max, shell=shell)
    in_grid = float(n_grid / n_ext) if n_ext > 0 else 0.0
    if in_grid < min_in_grid:
        flags |= DropletFlags.OUT_OF_GRID

    # |n_raw - 1| is the honest shape error: the total is made exact by the renormalisation
    # below whatever happens, but a deposit whose weights did not sum to 1 before that is one
    # the grid could not resolve.  Flag it rather than let it pass as if it were resolved.
    if abs((n_ext if n_ext > 0 else 0.0) - 1.0) > 0.05:
        flags |= DropletFlags.UNDER_RESOLVED

    n_raw = n_ext if renorm == "extended" else n_grid
    if n_raw <= 0:
        flags |= DropletFlags.OUT_OF_GRID
        w = np.zeros_like(w)
    else:
        w = w / n_raw

    eta_cen = float((w.sum(axis=(0, 1)) * ce).sum() / w.sum()) if w.sum() else 0.0
    return DepositPatch(ix, iy, ie, w, tau_eval, n_raw=float(n_grid if renorm == "grid" else n_ext),
                        in_grid_fraction=in_grid, flags=flags, shell_fraction=shell_frac,
                        eta_centroid=eta_cen, n_cells=int(w.size), eta_nodes=ce)


def _target_spacing(R, p):
    """The physical node spacing needed to resolve the deposit near radius R.

    Two scales compete: the smooth part varies on the causal radius R, the wave front on
    `width_delta`.  Resolving the finer of the two to a quarter, with R/16 as a floor so a
    well-resolved smooth deposit does not pay for a shell it barely has.
    """
    if R <= 0:
        return float(p.width_delta) / 4.0
    return max(min(float(p.width_delta), R) / 4.0, R / 16.0)


def _node_counts(extent_phys, n_sub, n_sub_max, s_target):
    """Nodes along one axis: enough that the PHYSICAL spacing resolves `s_target`.

    `extent_phys` is in fm, which for the eta axis means tau*cosh(eta)*d_eta -- at large |eta|
    that is huge, which is exactly why a fixed node count silently fails for forward droplets.
    """
    if n_sub != "auto":
        return max(int(n_sub), 1)
    if s_target <= 0:
        return 2
    return int(np.clip(np.ceil(extent_phys / s_target), 2, n_sub_max))


def _quadrature(cx, cy, ce, dx, dy, de, drop, tau_eval, p, *, n_sub, n_sub_max, shell,
                xlim, ylim, elim):
    """Support-clipped midpoint quadrature over the cell box -> (nx,ny,ne) weights."""
    tau_d, x_d, y_d, eta_d = (float(drop[i]) for i in range(4))
    t_d, z_d = tau_d * np.cosh(eta_d), tau_d * np.sinh(eta_d)
    w = np.zeros((len(cx), len(cy), len(ce)), dtype=np.float64)

    for k, ec in enumerate(ce):
        e_lo = max(ec - 0.5 * de, elim[0])
        e_hi = min(ec + 0.5 * de, elim[1])
        if e_hi <= e_lo:
            continue
        dt_c = tau_eval * np.cosh(ec) - t_d
        R_c = p.c_diff * max(dt_c, 0.0)
        s_target = _target_spacing(R_c, p)
        # dz/deta = tau cosh(eta): the eta axis is stretched enormously at large |eta|
        dz_deta = tau_eval * np.cosh(ec)
        ne_n = _node_counts(dz_deta * (e_hi - e_lo), n_sub, n_sub_max, s_target)
        en, e_len = _gauss_nodes(e_lo, e_hi, ne_n)
        f_e = (e_hi - e_lo) / de

        for i, xc in enumerate(cx):
            x_lo, x_hi = max(xc - 0.5 * dx, xlim[0]), min(xc + 0.5 * dx, xlim[1])
            if x_hi <= x_lo:
                continue
            nx_n = _node_counts(x_hi - x_lo, n_sub, n_sub_max, s_target)
            xn, _ = _gauss_nodes(x_lo, x_hi, nx_n)
            f_x = (x_hi - x_lo) / dx

            for j, yc in enumerate(cy):
                y_lo, y_hi = max(yc - 0.5 * dy, ylim[0]), min(yc + 0.5 * dy, ylim[1])
                if y_hi <= y_lo:
                    continue
                ny_n = _node_counts(y_hi - y_lo, n_sub, n_sub_max, s_target)
                yn, _ = _gauss_nodes(y_lo, y_hi, ny_n)
                f_y = (y_hi - y_lo) / dy

                X, Y, E = np.meshgrid(xn, yn, en, indexing="ij")
                r_w = None
                if shell == "widen":
                    # widen the shell to the local node spacing; mass-preserving, because the
                    # shell's integral does not depend on its width
                    hz = tau_eval * np.cosh(ec) * (e_hi - e_lo) / ne_n
                    h = max((x_hi - x_lo) / nx_n, (y_hi - y_lo) / ny_n, hz)
                    r_w = max(p.width_delta, h)
                kt = _K.k_tau(tau_eval, X, Y, E, drop, p, r_w=r_w)
                w[i, j, k] = float(np.mean(kt)) * f_x * f_y * f_e

    return w * (tau_eval * dx * dy * de)


def _extended_norm(drop, tau_eval, grid, p, eta_lo, eta_hi, r_max, *, n_sub, n_sub_max, shell):
    """The same quadrature on a virtual lattice of the same spacing covering the whole support,
    so `in_grid_fraction` measures what falls off the grid rather than hiding it."""
    dx, dy, de = float(grid.dx), float(grid.dy), float(grid.deta)
    x_d, y_d = float(drop[1]), float(drop[2])
    ex0 = float(np.asarray(grid.x.detach().cpu())[0])
    ey0 = float(np.asarray(grid.y.detach().cpu())[0])
    ee0 = float(np.asarray(grid.eta.detach().cpu())[0])

    def _aligned(c0, d, lo, hi):
        """Cell centres on the GRID's own lattice phase, covering [lo, hi] with one cell of
        margin.  Sharing the phase is what makes in_grid_fraction mean 'fell off the grid'
        rather than 'the two quadratures sampled differently'."""
        i_lo = int(np.floor((lo - c0) / d)) - 1
        i_hi = int(np.ceil((hi - c0) / d)) + 1
        return c0 + np.arange(i_lo, i_hi + 1) * d

    cx = _aligned(ex0, dx, x_d - r_max, x_d + r_max)
    cy = _aligned(ey0, dy, y_d - r_max, y_d + r_max)
    ce = _aligned(ee0, de, eta_lo, eta_hi)
    w = _quadrature(cx, cy, ce, dx, dy, de, drop, tau_eval, p,
                    n_sub=n_sub, n_sub_max=n_sub_max, shell=shell,
                    xlim=(x_d - r_max, x_d + r_max), ylim=(y_d - r_max, y_d + r_max),
                    elim=(eta_lo, eta_hi))
    return float(w.sum())


def _cic_patch(drop, tau_eval, grid, eta_c, flags, shell_frac, min_in_grid, renorm):
    """Whole support inside one cell: trilinear scatter of the full weight to its centroid."""
    ex = np.asarray(grid.x.detach().cpu(), dtype=np.float64)
    ey = np.asarray(grid.y.detach().cpu(), dtype=np.float64)
    ee = np.asarray(grid.eta.detach().cpu(), dtype=np.float64)
    dx, dy, de = float(grid.dx), float(grid.dy), float(grid.deta)
    x_d, y_d = float(drop[1]), float(drop[2])

    def cic(c, d, v):
        f = (v - c[0]) / d
        i0 = int(np.floor(f))
        t = f - i0
        pairs = [(i0, 1.0 - t), (i0 + 1, t)]
        return [(i, wt) for i, wt in pairs if 0 <= i < len(c) and wt != 0.0]

    px, py, pe = cic(ex, dx, x_d), cic(ey, dy, y_d), cic(ee, de, eta_c)
    if not px or not py or not pe:
        flags |= DropletFlags.OUT_OF_GRID
        return DepositPatch(slice(0, 0), slice(0, 0), slice(0, 0), np.zeros((0, 0, 0)),
                            tau_eval, n_raw=0.0, in_grid_fraction=0.0, flags=flags,
                            shell_fraction=shell_frac, n_cells=0, eta_nodes=ee[:0])

    ix = slice(min(i for i, _ in px), max(i for i, _ in px) + 1)
    iy = slice(min(i for i, _ in py), max(i for i, _ in py) + 1)
    ie = slice(min(i for i, _ in pe), max(i for i, _ in pe) + 1)
    w = np.zeros((ix.stop - ix.start, iy.stop - iy.start, ie.stop - ie.start))
    for i, wi in px:
        for j, wj in py:
            for k, wk in pe:
                w[i - ix.start, j - iy.start, k - ie.start] = wi * wj * wk
    total = float(w.sum())
    in_grid = total                      # the CIC weights that landed on the grid
    if in_grid < min_in_grid:
        flags |= DropletFlags.OUT_OF_GRID
    if total > 0:
        w = w / total
    return DepositPatch(ix, iy, ie, w, tau_eval, n_raw=1.0, in_grid_fraction=in_grid,
                        flags=flags, shell_fraction=shell_frac, eta_centroid=float(eta_c),
                        n_cells=int(w.size), eta_nodes=ee[ie])


def milne_dq_from_cartesian(patch: DepositPatch, P_cart, tau_q, grid):
    """Normalised cell weights + a Cartesian four-momentum -> Delta q on the patch, (4,nx,ny,ne).

    With `w` the normalised weights (sum = 1), dV = dx dy deta and eta the cell's rapidity:

        Delta q^tau = (w/dV) (E cosh eta - pz sinh eta)          [GeV/fm^2]
        Delta q^x   = (w/dV) px
        Delta q^y   = (w/dV) py
        Delta q^eta = (w/dV) (pz cosh eta - E sinh eta) / tau_q  [GeV/fm^3]

    The bracketed quantities are the ORTHONORMAL Milne (tetrad) components -- which is what
    X-SCAPE's get_ptau/get_peta return, with no factor of tau.  The solver's q is CONTRAVARIANT,
    hence the single 1/tau_q on the eta component and nowhere else.  There is no hbar c anywhere:
    MUSIC divides by it only because its own q is in fm^-4, while this solver keeps GeV.

    The conservation identity this makes exact: substituting the above into
    `fv.cartesian_four_momentum` (P^t = sum dV (cosh eta q^tau + tau sinh eta q^eta), etc.) the
    cosh/sinh pairs collapse via cosh^2 - sinh^2 = 1, leaving

        P^t = E sum(w),  P^x = px sum(w),  P^y = py sum(w),  P^z = pz sum(w)

    so the grid receives exactly (E, px, py, pz) if and only if

        sum_cells tau_eval * dx * dy * deta * <K_tau> = 1

    -- ONE scalar fixes all four components at once, which is why a single renormalisation buys
    exact conservation even when the shape is under-resolved.  That is also the measure the C++
    unit test TEST_GRID_TAU_ETA_CONSERVATION uses (dvolume = tau dx dy deta, inverse boost by
    cosh/sinh with no tau factor).
    """
    if patch.is_empty:
        return None
    E, px, py, pz = (float(v) for v in P_cart)
    eta = np.asarray(patch.eta_nodes, dtype=np.float64).reshape(1, 1, -1)
    ch, sh = np.cosh(eta), np.sinh(eta)
    dV = float(grid.dx) * float(grid.dy) * float(grid.deta)
    k = patch.w / dV
    return np.stack([k * (E * ch - pz * sh),
                     k * px * np.ones_like(ch),
                     k * py * np.ones_like(ch),
                     k * (pz * ch - E * sh) / float(tau_q)], axis=0)
