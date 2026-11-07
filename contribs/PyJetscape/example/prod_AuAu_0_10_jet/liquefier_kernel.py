#!/usr/bin/env python3
"""
liquefier_kernel.py -- X-SCAPE's CausalLiquefier kernel in Python, in the frame of the Bjorken flow
at the droplet's rapidity (<frame>bjorken</frame>, X-SCAPE's default) and in the lab frame
(<frame>lab</frame>, the liquefier's paper and X-SCAPE before 2026-11).

    python liquefier_kernel.py shapes                     # one droplet at eta_s = 0, 1, 1.8, 2.4
    python liquefier_kernel.py check PAIR.h5              # the C++'s recorded flux, reproduced
    python liquefier_kernel.py window PAIR.h5 [--eta 1]   # how much energy lands in |eta_s| < 1

The kernel is CausalLiquefier::smearing_kernel, sampled at MUSIC's cell centres as
LiquefierBase::sampled_flux does: the causal diffusion solution (smooth part plus a wave front of
width width_delta), evaluated on the slice tau_drop + tau_delay, weighted by its flux through
that slice, J^tau = cosh(eta) J^t - sinh(eta) J^z.  "flux" is the sum over the cell centres of
J^tau tau dx dy deta for a droplet of unit energy: 1 means the grid catches it exactly; the
C++ normalization (normalize_on_hydro_grid) divides by it.

frame = "lab":     t = tau cosh(eta), z = tau sinh(eta); the droplet at its lab position.
frame = "bjorken": t = tau cosh(eta - eta_d), z = tau sinh(eta - eta_d); the droplet at z = 0.
               The kernel then depends on eta only through eta - eta_d: every droplet deposits
               as one at midrapidity does.  The four-momentum part of the source (Milne
               components of the droplet's lab momentum) is the same in both.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
from scipy.special import iv

# the Au+Au production's liquefier (config/AuAu_MCGlauber_MUSIC_0_10_jet.xml); a pair file's
# liquefier_* attributes override them
PARAMS = dict(time_relax=0.1, d_diff=0.08, width_delta=0.1, tau_delay=1.0)
# MUSIC's grid in the production (set by the 3D Glauber initial state, not music_input)
NATIVE = dict(x_min=-15.0, dx=0.3, nx=100, y_min=-15.0, dy=0.3, ny=100,
              eta_min=-6.0, deta=0.2, neta=60)


class Kernel:
    def __init__(self, time_relax=0.1, d_diff=0.08, width_delta=0.1, tau_delay=1.0):
        self.c = np.sqrt(d_diff / time_relax)        # signal speed
        self.g = 0.5 / time_relax                    # damping rate
        self.wd = width_delta
        self.tau_delay = tau_delay

    def rho_j(self, t, r):
        """kernel_rho, kernel_j (radial) at lab age t (scalar) and distance r (array);
        and the smooth part of rho alone."""
        c, g = self.c, self.g
        damp = np.exp(-g * t) / (4 * np.pi)
        inside = r < c * t
        u = np.sqrt(np.maximum(c * c * t * t - r * r, 1e-300))
        x = g * u / c
        f = g * g / c
        rho_s = f * (iv(1, x) / (c * u) + iv(2, x) * t / (u * u))
        j_s = f * iv(2, x) * r / (u * u)
        rw = min(self.wd, c * t)
        front = inside & (r >= c * t - rw)
        rho_d = np.where(front, (1 + g * t + (g * t) ** 2 / 2) / rw / np.maximum(r, 1e-300) ** 2, 0.0)
        rho = np.where(inside, damp * (rho_s + rho_d), 0.0)
        j = np.where(inside, damp * (j_s + c * rho_d), 0.0)
        return rho, j, np.where(inside, damp * rho_s, 0.0)

    def deposit(self, drop, grid, frame="lab", tau_q=None):
        """The droplet's flux weights on the grid: (w[eta, x, y], smooth[eta, x, y], eta, age).
        w = J^tau tau dx dy deta for a unit droplet; the slice is tau_q (default
        tau_drop + tau_delay)."""
        tau_d, xd, yd, eta_d = drop[:4]
        tq = tau_d + self.tau_delay if tau_q is None else tau_q
        etas = grid["eta_min"] + grid["deta"] * np.arange(grid["neta"])
        xs = grid["x_min"] + grid["dx"] * np.arange(grid["nx"])
        ys = grid["y_min"] + grid["dy"] * np.arange(grid["ny"])
        if frame == "lab":
            td, zd, rel = tau_d * np.cosh(eta_d), tau_d * np.sinh(eta_d), etas
        elif frame == "bjorken":
            td, zd, rel = tau_d, 0.0, etas - eta_d
        else:
            raise ValueError(frame)
        w = np.zeros((len(etas), len(xs), len(ys)))
        sm = np.zeros_like(w)
        age = np.zeros(len(etas))
        reach = self.c * (tq * np.cosh(rel) - td)    # largest radius per slice
        for ie in np.nonzero(reach > 0)[0]:
            t, z = tq * np.cosh(rel[ie]), tq * np.sinh(rel[ie])
            dt = t - td
            if abs(z - zd) >= self.c * dt:
                continue
            R = np.sqrt((self.c * dt) ** 2 - (z - zd) ** 2) + max(grid["dx"], grid["dy"])
            ix = np.nonzero(np.abs(xs - xd) <= R)[0]
            iy = np.nonzero(np.abs(ys - yd) <= R)[0]
            if not len(ix) or not len(iy):
                continue
            X, Y = np.meshgrid(xs[ix], ys[iy], indexing="ij")
            r = np.sqrt((X - xd) ** 2 + (Y - yd) ** 2 + (z - zd) ** 2)
            rho, j, rho_s = self.rho_j(dt, r)
            jz = np.where(r > 0, (z - zd) / np.maximum(r, 1e-300) * j, 0.0)
            jtau = np.cosh(rel[ie]) * rho - np.sinh(rel[ie]) * jz
            dV = tq * grid["dx"] * grid["dy"] * grid["deta"]
            w[ie][np.ix_(ix, iy)] = jtau * dV
            sm[ie][np.ix_(ix, iy)] = np.cosh(rel[ie]) * rho_s * dV
            age[ie] = dt
        return w, sm, etas, age


def summary(w, sm, etas, age):
    tot = w.sum()
    we = w.sum(axis=(1, 2))
    if tot <= 0:
        return dict(flux=tot, cells=0, mean=np.nan, rms=np.nan, age=np.nan, smooth=np.nan)
    mean = (we * etas).sum() / tot
    return dict(flux=tot, cells=int((np.abs(w) > 1e-3 * np.abs(w).max()).sum()), mean=mean,
                rms=np.sqrt((we * (etas - mean) ** 2).sum() / tot),
                age=(we * age).sum() / tot, smooth=sm.sum() / tot)


def grid_of(f):
    a = f.attrs
    return dict(x_min=float(a["X_min_MUSIC"]), dx=float(a["dX_MUSIC"]), nx=int(a["nX_MUSIC"]),
                y_min=float(a["Y_min_MUSIC"]), dy=float(a["dY_MUSIC"]), ny=int(a["nY_MUSIC"]),
                eta_min=float(a["eta_min_MUSIC"]), deta=float(a["deta_MUSIC"]),
                neta=int(a["neta_MUSIC"]))


def kernel_of(f):
    a = f.attrs
    return Kernel(float(a["liquefier_time_relax"]), float(a["liquefier_d_diff"]),
                  float(a["liquefier_width_delta"]), float(a["liquefier_tau_delay"]))


def music_step(tau, tau0=0.4, dtau=0.02, late=0.05, switch=3.0):
    """The MUSIC step time whose deposit window [t - dt/2, t + dt/2) holds tau: steps of dtau
    from tau0, then of late from switch (the production's late_delta_tau, late_tau_switch)."""
    if tau < switch - dtau / 2:
        return tau0 + np.floor((tau - tau0) / dtau + 0.5) * dtau
    return switch + np.floor((tau - switch) / late + 0.5) * late


# ── commands ────────────────────────────────────────────────────────────────────────────
def cmd_shapes(a):
    k = Kernel(**PARAMS)
    fine = dict(x_min=-3.0, dx=0.05, nx=121, y_min=-3.0, dy=0.05, ny=121,
                eta_min=-5.0, deta=0.005, neta=2000)
    print(f"a droplet made at tau = 1 fm/c at x = 0.07, y = 0.11 fm, deposited on tau = 2 fm/c;"
          f" native grid {NATIVE['dx']} fm, deta {NATIVE['deta']}")
    print(f"{'frame':>7} {'eta_d':>5} | {'fine grid: mean':>15} {'rms':>6} {'age':>6} {'smooth':>6} |"
          f" {'native: flux':>12} {'cells':>6} {'mean':>6}")
    for frame in ("lab", "bjorken"):
        for eta_d in (0.0, 1.0, 1.8, 2.4):
            drop = (1.0, 0.07, 0.11, eta_d)
            s_f = summary(*k.deposit(drop, fine, frame, tau_q=2.0))
            s_n = summary(*k.deposit(drop, NATIVE, frame, tau_q=2.0))
            print(f"{frame:>7} {eta_d:5.1f} | {s_f['mean']:15.2f} {s_f['rms']:6.3f} {s_f['age']:6.2f}"
                  f" {s_f['smooth']:6.2f} | {s_n['flux']:12.3f} {s_n['cells']:6d} {s_n['mean']:6.2f}"
                  f"   (fine flux {s_f['flux']:.3f})")


def cmd_check(a):
    import h5py
    import hdf5plugin  # noqa: F401
    with h5py.File(a.pair, "r") as f:
        g, k = grid_of(f), kernel_of(f)
        d, fl = f["source/droplets"][:], f["source/flux"][:]
        tau0 = f["diag/tau0_music"][:]
        off = f["source/offsets"][:]
    ev = np.repeat(np.arange(len(off) - 1), np.diff(off))
    rows = []
    for i, drop in enumerate(d):
        if fl[i] <= 1e-6:                            # never deposited, or into one cell
            continue
        tq = music_step(drop[0] + k.tau_delay, tau0[ev[i]])
        s = summary(*k.deposit(drop, g, "lab", tau_q=tq))
        rows.append((ev[i], drop[3], fl[i], s["flux"]))
    r = np.array(rows)
    rel = r[:, 3] / r[:, 2] - 1
    print(f"{len(r)} droplets (of {len(fl)}): |Python / C++ flux - 1|: median {np.median(np.abs(rel)):.1e},"
          f" max {np.abs(rel).max():.1e}")


def cmd_window(a):
    import h5py
    import hdf5plugin  # noqa: F401
    with h5py.File(a.pair, "r") as f:
        g, k = grid_of(f), kernel_of(f)
        d, off = f["source/droplets"][:], f["source/offsets"][:]
        tau0 = f["diag/tau0_music"][:]
    etas = g["eta_min"] + g["deta"] * np.arange(g["neta"])
    win = np.abs(etas) < a.eta
    print(f"lab energy [GeV] at |eta_s| < {a.eta}: of droplets made there, and deposited there")
    print(f"{'event':>5} {'made':>8} | {'lab frame':>9} {'bjorken frame':>13}")
    for e in range(len(off) - 1):
        made, dep = 0.0, {"lab": 0.0, "bjorken": 0.0}
        for drop in d[off[e]:off[e + 1]]:
            E = drop[4]
            if abs(drop[3]) < a.eta:
                made += E
            tq = music_step(drop[0] + k.tau_delay, tau0[e])
            for frame in dep:
                w = k.deposit(drop, g, frame, tau_q=tq)[0].sum(axis=(1, 2))
                if w.sum() > 1e-6:
                    dep[frame] += E * w[win].sum() / w.sum()
                else:                                # the C++ fallback: one cell at eta_d
                    dep[frame] += E * (abs(drop[3]) < a.eta)
        print(f"{e:5d} {made:8.2f} | {dep['lab']:9.2f} {dep['bjorken']:13.2f}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("shapes")
    c = sub.add_parser("check"); c.add_argument("pair")
    w = sub.add_parser("window"); w.add_argument("pair"); w.add_argument("--eta", type=float, default=1.0)
    a = p.parse_args(argv)
    {"shapes": cmd_shapes, "check": cmd_check, "window": cmd_window}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
