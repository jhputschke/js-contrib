#!/usr/bin/env python3
"""
liquefier_kernel.py -- X-SCAPE's CausalLiquefier kernel in Python, in its three frames: the frame
of the Bjorken flow at the droplet's rapidity (<frame>bjorken</frame>, X-SCAPE's default), the lab
frame (<frame>lab</frame>, the liquefier's paper and X-SCAPE before 2026-11) and the rest frame of
the background fluid where the droplet was made (<frame>fluid</frame>).

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
               components of the droplet's lab momentum) is the same in all frames.
frame = "fluid":   the Bjorken frame, then a boost with the fluid's velocity relative to the
               Bjorken flow at the droplet (relative_velocity): the cell-droplet separation is
               boosted into the fluid's rest frame, rho and the radial j are evaluated at that
               age and distance, the current is boosted back and its flux through the slice
               taken.  The fluid is (vx, vy, vz, e) in the lab where the droplet was made
               (the pair file's source/fluid, or fluid_from_background for older files); with
               e < e_min (no fluid there) it is the Bjorken frame.
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


FRAMES = ("lab", "bjorken", "fluid")
FLUID_E_MIN, FLUID_V_MAX = 0.1, 0.99        # X-SCAPE's defaults (<fluid_e_min>, <fluid_v_max>)


def relative_velocity(eta_d, fluid, e_min=FLUID_E_MIN, v_max=FLUID_V_MAX):
    """CausalLiquefier::kernel_frame_velocity: the fluid's velocity relative to the Bjorken flow
    at eta_d, its speed capped at v_max; 0 without a usable fluid (fluid None, e < e_min)."""
    if fluid is None or not np.isfinite(eta_d):
        return np.zeros(3)
    vx, vy, vz, e = (float(c) for c in fluid)
    if not (e >= 0.0 and e >= e_min):
        return np.zeros(3)
    ut = np.cosh(eta_d) - vz * np.sinh(eta_d)
    uz = vz * np.cosh(eta_d) - np.sinh(eta_d)
    if not ut > 0:
        return np.zeros(3)
    v = np.array([vx / ut, vy / ut, uz / ut])
    s = np.sqrt(v @ v)
    return v * (v_max / s) if s > v_max else v


class Kernel:
    def __init__(self, time_relax=0.1, d_diff=0.08, width_delta=0.1, tau_delay=1.0):
        self.c = np.sqrt(d_diff / time_relax)        # signal speed
        self.g = 0.5 / time_relax                    # damping rate
        self.wd = width_delta
        self.tau_delay = tau_delay

    def rho_j(self, t, r):
        """kernel_rho, kernel_j (radial) at the kernel frame's age t and distance r (arrays
        that broadcast); and the smooth part of rho alone."""
        c, g = self.c, self.g
        t = np.asarray(t, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            return self._rho_j(t, r, c, g)

    def _rho_j(self, t, r, c, g):
        damp = np.exp(-g * t) / (4 * np.pi)
        inside = (r < c * t) & (t > 0)
        u = np.sqrt(np.maximum(c * c * t * t - r * r, 1e-300))
        x = g * u / c
        f = g * g / c
        rho_s = f * (iv(1, x) / (c * u) + iv(2, x) * t / (u * u))
        j_s = f * iv(2, x) * r / (u * u)
        rw = np.maximum(np.minimum(self.wd, c * t), 1e-300)
        front = inside & (r >= c * t - rw)
        rho_d = np.where(front, (1 + g * t + (g * t) ** 2 / 2) / rw / np.maximum(r, 1e-300) ** 2, 0.0)
        rho = np.where(inside, damp * (rho_s + rho_d), 0.0)
        j = np.where(inside, damp * (j_s + c * rho_d), 0.0)
        return rho, j, np.where(inside, damp * rho_s, 0.0)

    def deposit(self, drop, grid, frame="lab", tau_q=None, fluid=None, e_min=FLUID_E_MIN,
                v_max=FLUID_V_MAX):
        """The droplet's flux weights on the grid: (w[eta, x, y], smooth[eta, x, y], eta, age).
        w = J^tau tau dx dy deta for a unit droplet; the slice is tau_q (default
        tau_drop + tau_delay).  frame "fluid" needs fluid = (vx, vy, vz, e) at the droplet
        (None: the Bjorken frame); age is then the mean rest-frame age per slice."""
        tau_d, xd, yd, eta_d = drop[:4]
        tq = tau_d + self.tau_delay if tau_q is None else tau_q
        etas = grid["eta_min"] + grid["deta"] * np.arange(grid["neta"])
        xs = grid["x_min"] + grid["dx"] * np.arange(grid["nx"])
        ys = grid["y_min"] + grid["dy"] * np.arange(grid["ny"])
        v = np.zeros(3)
        if frame == "lab":
            td, zd, rel = tau_d * np.cosh(eta_d), tau_d * np.sinh(eta_d), etas
        elif frame in ("bjorken", "fluid"):
            td, zd, rel = tau_d, 0.0, etas - eta_d
            if frame == "fluid":
                v = relative_velocity(eta_d, fluid, e_min, v_max)
        else:
            raise ValueError(frame)
        v2 = float(v @ v)
        boosted = v2 > 0
        g = 1.0 / np.sqrt(1.0 - v2)
        gg = g * g / (1.0 + g)
        # the support lies inside the droplet's future light cone (any frame); without a
        # boost, inside the kernel's own signal front c dt
        cs = 1.0 if boosted else self.c
        w = np.zeros((len(etas), len(xs), len(ys)))
        sm = np.zeros_like(w)
        age = np.zeros(len(etas))
        reach = cs * (tq * np.cosh(rel) - td)        # largest radius per slice
        for ie in np.nonzero(reach > 0)[0]:
            t, z = tq * np.cosh(rel[ie]), tq * np.sinh(rel[ie])
            dt = t - td
            if abs(z - zd) >= cs * dt:
                continue
            R = np.sqrt((cs * dt) ** 2 - (z - zd) ** 2) + max(grid["dx"], grid["dy"])
            ix = np.nonzero(np.abs(xs - xd) <= R)[0]
            iy = np.nonzero(np.abs(ys - yd) <= R)[0]
            if not len(ix) or not len(iy):
                continue
            X, Y = np.meshgrid(xs[ix], ys[iy], indexing="ij")
            d3 = np.stack([X - xd, Y - yd, np.full_like(X, z - zd)])
            if boosted:
                vd = np.tensordot(v, d3, axes=1)
                dtp = g * (dt - vd)
                d3 = d3 + (gg * vd - g * dt) * v[:, None, None]
            else:
                dtp = np.full_like(X, dt)
            r = np.sqrt((d3 ** 2).sum(axis=0))
            with np.errstate(over="ignore", invalid="ignore"):
                rho, j, rho_s = self.rho_j(dtp, r)
            jr = np.where(r > 0, j / np.maximum(r, 1e-300), 0.0) * d3
            jt, jz = rho, jr[2]
            if boosted:
                vj = np.tensordot(v, jr, axes=1)
                jt = g * (rho + vj)
                jz = jr[2] + (gg * vj + g * rho) * v[2]
            jtau = np.cosh(rel[ie]) * jt - np.sinh(rel[ie]) * jz
            dV = tq * grid["dx"] * grid["dy"] * grid["deta"]
            w[ie][np.ix_(ix, iy)] = jtau * dV
            sm[ie][np.ix_(ix, iy)] = np.cosh(rel[ie]) * rho_s * dV
            age[ie] = dt if not boosted else (
                (dtp * np.abs(jtau)).sum() / max(np.abs(jtau).sum(), 1e-300))
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


def frame_of(f):
    """The liquefier frame a pair file was made in (files without the attribute: lab)."""
    fr = f.attrs.get("liquefier_frame", "lab")
    return fr.decode() if isinstance(fr, bytes) else str(fr)


def fluid_of(f):
    """(M, 4) = (vx, vy, vz, e) of the background fluid at every droplet of a pair file, where
    it was made: the file's source/fluid (an X-SCAPE whose droplets carry it), else
    interpolated from arr_bg (linear in tau, trilinear in x, y, eta_s; e = 0 before the
    hydro's first stored frame and off the grid, which the fluid frame treats as no fluid).
    The second value says which: "file" or "arr_bg"."""
    if "source/fluid" in f:
        return f["source/fluid"][:].astype(np.float64), "file"
    from scipy.interpolate import RegularGridInterpolator
    d, off = f["source/droplets"][:], f["source/offsets"][:]
    a = f.attrs
    axes = [a[k + "_min"] + a["d" + k] * np.arange(a["n" + k]) for k in ("x", "y", "eta")]
    t0, dt = float(a["tau_min"]), float(a["dtau"])
    out = np.zeros((len(d), 4))
    cache = {}

    def at(ev, k, pts):
        key = (ev, k)
        if key not in cache:
            fr = f["arr_bg"][ev, ..., k].astype(np.float64)        # (4, nx, ny, neta)
            cache[key] = [RegularGridInterpolator(axes, fr[c], bounds_error=False, fill_value=0.0)
                          for c in range(4)]
        return np.array([ip(pts) for ip in cache[key]]).T          # (n, 4): e, vx, vy, vz

    for ev in range(len(off) - 1):
        nfr = int(f["ntau_freezeout_bg"][ev]) if "ntau_freezeout_bg" in f else f["arr_bg"].shape[-1]
        for i in range(off[ev], off[ev + 1]):
            tau, x, y, eta = d[i, :4]
            u = (tau - t0) / dt
            if u < 0 or u > nfr - 1:
                continue                                           # no fluid stored there
            k = min(int(np.floor(u)), nfr - 2)
            w1 = u - k
            p = np.array([[x, y, eta]])
            c = (1 - w1) * at(ev, k, p)[0] + w1 * at(ev, k + 1, p)[0]
            out[i] = (c[1], c[2], c[3], c[0])
        cache.clear()
    return out, "arr_bg"


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
    print("fluid: the Bjorken flow plus a transverse flow v_T along x (v_T = 0: the Bjorken frame)")
    print(f"{'frame':>7} {'v_T':>4} {'eta_d':>5} | {'fine grid: mean':>15} {'rms':>6} {'age':>6} {'smooth':>6} |"
          f" {'native: flux':>12} {'cells':>6} {'mean':>6}")
    for frame, vT in (("lab", 0), ("bjorken", 0), ("fluid", 0.5), ("fluid", 0.8)):
        for eta_d in (0.0, 1.0, 1.8, 2.4):
            drop = (1.0, 0.07, 0.11, eta_d)
            fl = (vT / np.cosh(eta_d), 0.0, np.tanh(eta_d), 1.0)
            s_f = summary(*k.deposit(drop, fine, frame, tau_q=2.0, fluid=fl))
            s_n = summary(*k.deposit(drop, NATIVE, frame, tau_q=2.0, fluid=fl))
            print(f"{frame:>7} {vT:4.1f} {eta_d:5.1f} | {s_f['mean']:15.2f} {s_f['rms']:6.3f} {s_f['age']:6.2f}"
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
        frame = frame_of(f)
        fluid = fluid_of(f)[0] if frame == "fluid" else None
        e_min = float(f.attrs.get("liquefier_fluid_e_min", FLUID_E_MIN))
        v_max = float(f.attrs.get("liquefier_fluid_v_max", FLUID_V_MAX))
    ev = np.repeat(np.arange(len(off) - 1), np.diff(off))
    rows = []
    for i, drop in enumerate(d):
        if fl[i] <= 1e-6:                            # never deposited, or into one cell
            continue
        tq = music_step(drop[0] + k.tau_delay, tau0[ev[i]])
        s = summary(*k.deposit(drop, g, frame, tau_q=tq,
                               fluid=None if fluid is None else fluid[i], e_min=e_min,
                               v_max=v_max))
        rows.append((ev[i], drop[3], fl[i], s["flux"]))
    r = np.array(rows)
    rel = r[:, 3] / r[:, 2] - 1
    print(f"frame {frame}: {len(r)} droplets (of {len(fl)}): |Python / C++ flux - 1|: median "
          f"{np.median(np.abs(rel)):.1e}, max {np.abs(rel).max():.1e}")


def cmd_window(a):
    import h5py
    import hdf5plugin  # noqa: F401
    with h5py.File(a.pair, "r") as f:
        g, k = grid_of(f), kernel_of(f)
        d, off = f["source/droplets"][:], f["source/offsets"][:]
        tau0 = f["diag/tau0_music"][:]
        fluid = fluid_of(f)[0]
    etas = g["eta_min"] + g["deta"] * np.arange(g["neta"])
    win = np.abs(etas) < a.eta
    print(f"lab energy [GeV] at |eta_s| < {a.eta}: of droplets made there, and deposited there")
    print(f"{'event':>5} {'made':>8} | {'lab frame':>9} {'bjorken':>8} {'fluid':>8}")
    for e in range(len(off) - 1):
        made, dep = 0.0, {fr: 0.0 for fr in FRAMES}
        for i in range(off[e], off[e + 1]):
            drop = d[i]
            E = drop[4]
            if abs(drop[3]) < a.eta:
                made += E
            tq = music_step(drop[0] + k.tau_delay, tau0[e])
            for frame in dep:
                w = k.deposit(drop, g, frame, tau_q=tq, fluid=fluid[i])[0].sum(axis=(1, 2))
                if w.sum() > 1e-6:
                    dep[frame] += E * w[win].sum() / w.sum()
                else:                                # the C++ fallback: one cell at eta_d
                    dep[frame] += E * (abs(drop[3]) < a.eta)
        print(f"{e:5d} {made:8.2f} | {dep['lab']:9.2f} {dep['bjorken']:8.2f} {dep['fluid']:8.2f}")


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
