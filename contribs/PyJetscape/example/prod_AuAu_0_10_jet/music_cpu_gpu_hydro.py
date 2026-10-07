#!/usr/bin/env python3
"""
example/prod_AuAu_0_10_jet/music_cpu_gpu_hydro.py

MUSIC on the CPU (double precision) against MUSIC4GPU (single precision, vacuum cells put at
rest): the same events, both written on MUSIC's grid.  See docs/MUSIC_CPU_vs_GPU.md.

    python run_prod_jet.py --native --events 6 --reuse 2 --seed 11 --write-particlize both \\
        --outdir OUT/gpu --seed-registry none
    MUSIC_FORCE_CPU=1 python run_prod_jet.py ... --outdir OUT/cpu ...      # the same options
    python music_cpu_gpu_hydro.py OUT/gpu/AuAu_0_10_jet_seed0011.h5 OUT/cpu/AuAu_0_10_jet_seed0011.h5

Backgrounds start from identical initial conditions on both paths and compare cell by cell.
The jet legs get the same showers only when Matter and LBT happen to sample the slightly
different media the same way; those events (same droplets on both paths) are compared cell
by cell, the others only through the wake per deposited energy.

Energies are those of the ideal part of T^{mu nu} (the store has no viscous fields) through
the tau = const surface, in the lab frame:

    E = sum tau dx dy deta [(e + P) u^t u^tau - P cosh eta],   u^tau = u^t (cosh eta - v_z sinh eta)

with the stored lab-frame velocities v = u/u^t.  They are summed over real fluid only
(e >= E_CUT = 1e-4 GeV/fm^3): the CPU's vacuum cells (e ~ 1e-14 GeV/fm^3) move with u^tau up
to ~6000, and their float32 velocities round to |v| = 1, so their u^t cannot be recovered
from the file.  Vacuum cells are only counted.

Needs the environment PyJetscape was built in (js_fno on the GB10) and MUSIC's EOS 9 table.
"""

from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
XSCAPE = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "..", ".."))
DEFAULT_EOS = os.path.join(XSCAPE, "build_gpu", "EOS", "hotQCD", "hrg_hotqcd_eos_binary.dat")

#: MUSIC4GPU's GPU_VACUUM_E (1e-5 1/fm^4) in GeV/fm^3: below it, a cell with u^0 > 10 is put
#: at rest on the GPU
E_VAC = 1e-5 * 0.1973269804
#: real fluid, far above both paths' vacuum floors [GeV/fm^3]
E_CUT = 1e-4
#: frames shown per background [fm/c]
SHOW = (1.0, 2.0, 4.0, 6.0, 8.0, 10.0)


class Eos:
    """MUSIC's hotQCD table: rows (e, P, s, T), linear in e as MUSIC reads it."""

    def __init__(self, path, T_fo=0.15):
        if not os.path.exists(path):
            sys.exit(f"music_cpu_gpu_hydro.py: EoS table {path} not found (--eos)")
        self.tab = np.fromfile(path, dtype="<f8").reshape(-1, 4)
        self.e_fo = float(np.interp(T_fo, self.tab[:, 3], self.tab[:, 0]))

    def P(self, e):
        t = self.tab
        return np.where(e < t[0, 0], np.clip(e, 0, None) * t[0, 1] / t[0, 0],
                        np.interp(e, t[:, 0], t[:, 1]))

    def T(self, e):
        return np.where(e > 0, np.interp(e, self.tab[:, 0], self.tab[:, 3]), 0.0)


class Leg:
    """One leg of one event: frames (4, nx, ny, neta) of e, vx, vy, vz on MUSIC's grid."""

    def __init__(self, f, ds, row, nt):
        a = f.attrs
        if a.get("xscape_grid_mode") != "native":
            sys.exit("music_cpu_gpu_hydro.py: the pair files must be on MUSIC's grid "
                     "(run_prod_jet.py --native)")
        self.f, self.ds, self.row, self.nt = f, ds, row, nt
        self.eta = a["eta_min"] + a["deta"] * np.arange(int(a["neta"]))
        self.dV = float(a["dx"] * a["dy"] * a["deta"])
        self.tau0, self.dtau = float(a["tau_min"]), float(a["dtau"])

    def tau(self, t):
        return self.tau0 + self.dtau * t

    def frame(self, t):
        return None if t >= self.nt else self.f[self.ds][self.row, :, ..., t].astype(np.float64)


def background(f, b):
    return Leg(f, "arr_bg_store" if "arr_bg_store" in f else "arr_bg",
               int(f["arr_bg_rows"][b]) if "arr_bg_rows" in f else b,
               int(f["ntau_freezeout_bg"][b]))


def measures(eos, fr, eta, tau, dV):
    """Energies, vacuum counts and the momentum anisotropy of one frame."""
    e, vx, vy, vz = fr
    P = eos.P(e)
    om = 1.0 - (vx * vx + vy * vy + vz * vz)
    g2 = 1.0 / np.clip(om, 1e-12, None)
    ch, sh = np.cosh(eta)[None, None, :], np.sinh(eta)[None, None, :]
    utau = np.sqrt(g2) * (ch - vz * sh)
    E = tau * dV * ((e + P) * g2 * (ch - vz * sh) - P * ch)
    vac, fluid, hot = e < E_VAC, e >= E_CUT, e > eos.e_fo
    unres = om <= 1e-6                          # float32 velocities cannot give gamma here
    mid = (np.abs(eta) < 0.5)[None, None, :] & fluid
    txx = (e + P) * g2 * vx * vx + P
    tyy = (e + P) * g2 * vy * vy + P
    den = ((txx + tyy) * mid).sum()
    return {"E": float(E[fluid].sum()), "E_hot": float(E[hot].sum()),
            "E_dilute": float(E[fluid & ~hot].sum()),
            "E_low": float(E[~fluid & ~vac & ~unres].sum()),
            "n_fast_vac": int((vac & (utau > 10) & ~unres).sum()),
            "n_unres_vac": int((vac & unres).sum()),
            "n_unres_fluid": int((fluid & unres).sum()),
            "eps_p": float(((txx - tyy) * mid).sum() / den) if den > 0 else np.nan}


def field_diff(eos, a, b):
    """Cell-by-cell differences where either path is above freeze-out (energy-weighted)."""
    hot = (a[0] > eos.e_fo) | (b[0] > eos.e_fo)
    if not hot.any():
        return None
    em = 0.5 * (a[0] + b[0])[hot]
    rel = np.abs(a[0] - b[0])[hot] / em
    dT = np.abs(eos.T(a[0][hot]) - eos.T(b[0][hot])) / np.maximum(eos.T(b[0][hot]), 1e-9)
    dv = np.sqrt(((a[1:] - b[1:])[:, hot] ** 2).sum(0))
    return {"rel": float((rel * em).sum() / em.sum()), "rel_max": float(rel.max()),
            "dT": float((dT * em).sum() / em.sum()), "dv": float((dv * em).sum() / em.sum())}


def compare_background(eos, fg, fc, b):
    lg, lc = background(fg, b), background(fc, b)
    n = max(lg.nt, lc.nt)
    rows = []
    for t in range(n):
        a, c = lg.frame(t), lc.frame(t)
        if a is None or c is None:
            rows.append(None)
            continue
        tau = lg.tau(t)
        rows.append((tau, measures(eos, a, lg.eta, tau, lg.dV),
                     measures(eos, c, lc.eta, tau, lc.dV), field_diff(eos, a, c)))
    return lg, lc, rows


def same_showers(fg, fc, i, tol=1e-2):
    og, oc = fg["source/offsets"], fc["source/offsets"]
    dg = fg["source/droplets"][og[i]:og[i + 1]]
    dc = fc["source/droplets"][oc[i]:oc[i + 1]]
    return dg.shape == dc.shape and (len(dg) == 0 or np.abs(dg - dc).max() < tol)


def wake(eos, f, i, t):
    """Jet leg minus background at frame t: (fluid-energy difference, de field)."""
    b = int(f["diag/bg_id"][i])
    lj, lb = Leg(f, "arr", i, int(f["ntau_freezeout"][i])), background(f, b)
    a, c = lj.frame(t), lb.frame(t)
    tau = lj.tau(t)
    return (measures(eos, a, lj.eta, tau, lj.dV)["E"]
            - measures(eos, c, lb.eta, tau, lb.dV)["E"]), a[0] - c[0]


def rel(g, c):
    return (g - c) / c if c else np.nan


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("gpu_pair")
    p.add_argument("cpu_pair")
    p.add_argument("--eos", default=DEFAULT_EOS, help=f"MUSIC's EOS 9 table ({DEFAULT_EOS})")
    a = p.parse_args()
    import hdf5plugin  # noqa: F401  (Blosc-compressed pair files)

    eos = Eos(a.eos)
    fg, fc = h5py.File(a.gpu_pair, "r"), h5py.File(a.cpu_pair, "r")
    bg_ids = fg["diag/bg_id"][:].astype(int)
    if not np.array_equal(bg_ids, fc["diag/bg_id"][:].astype(int)):
        sys.exit("music_cpu_gpu_hydro.py: the two files have different backgrounds per event")
    print(f"e_fo = {eos.e_fo:.4f} GeV/fm^3, GPU vacuum threshold {E_VAC:.2e} GeV/fm^3, "
          f"fluid cut {E_CUT:g} GeV/fm^3")

    summary = []
    for b in sorted(set(bg_ids)):
        lg, lc, rows = compare_background(eos, fg, fc, b)
        print(f"\n=== background {b}: frames GPU {lg.nt} / CPU {lc.nt}")
        print("  tau | E fluid GPU [GeV] (G-C)/C | E hot (G-C)/C | E dilute (G-C)/C | "
              "vacuum cells u^tau>10: GPU CPU | |v|=1 in float: GPU CPU | eps_p GPU CPU | "
              "<|de|/e> max|de|/e <|dT|/T> <|dv|>")
        for t0 in SHOW:
            k = int(round((t0 - lg.tau0) / lg.dtau))
            if k >= len(rows) or rows[k] is None or rows[k][3] is None:
                continue
            tau, g, c, d = rows[k]
            print(f"  {t0:4.1f} | {g['E']:9.1f} {rel(g['E'], c['E']):+.2e} | "
                  f"{rel(g['E_hot'], c['E_hot']):+.2e} | {rel(g['E_dilute'], c['E_dilute']):+.2e} | "
                  f"{g['n_fast_vac']:6d} {c['n_fast_vac']:6d} | {g['n_unres_vac']:6d} "
                  f"{c['n_unres_vac']:6d} | {g['eps_p']:+.4f} {c['eps_p']:+.4f} | "
                  f"{d['rel']:.2e} {d['rel_max']:.2e} {d['dT']:.2e} {d['dv']:.2e}")
        ok = [r for r in rows if r is not None and r[3] is not None and r[0] >= 1.0 - 1e-9]
        summary.append({
            "b": b, "nt": (lg.nt, lc.nt),
            "E": np.median([rel(g["E"], c["E"]) for _, g, c, _ in ok]),
            "E_hot": np.median([rel(g["E_hot"], c["E_hot"]) for _, g, c, _ in ok]),
            "E_dilute": np.median([rel(g["E_dilute"], c["E_dilute"]) for _, g, c, _ in ok]),
            "rel": np.median([d["rel"] for *_, d in ok]),
            "rel_max": max(d["rel_max"] for *_, d in ok),
            "dT": np.median([d["dT"] for *_, d in ok]),
            "dv": np.median([d["dv"] for *_, d in ok]),
            "eps": np.median([g["eps_p"] - c["eps_p"] for _, g, c, _ in ok]),
            "fast_vac": (max(g["n_fast_vac"] for _, g, _, _ in ok),
                         max(c["n_fast_vac"] for _, _, c, _ in ok)),
            "unres_fluid": max(max(g["n_unres_fluid"], c["n_unres_fluid"]) for _, g, c, _ in ok)})

    print("\n=== backgrounds: median over frames with tau >= 1 fm/c (both paths running)")
    for s in summary:
        print(f"  bg {s['b']}: frames {s['nt'][0]}/{s['nt'][1]} | (G-C)/C: E fluid "
              f"{s['E']:+.2e}, E hot {s['E_hot']:+.2e}, E dilute {s['E_dilute']:+.2e} | "
              f"<|de|/e> {s['rel']:.2e} (max cell {s['rel_max']:.2e}), <|dT|/T> {s['dT']:.2e}, "
              f"<|dv|> {s['dv']:.2e} | eps_p G-C {s['eps']:+.1e} | fast vacuum cells (most) "
              f"GPU {s['fast_vac'][0]} CPU {s['fast_vac'][1]} | fluid cells with |v|=1 in "
              f"float {s['unres_fluid']}")

    print("\n=== jet legs: wake = E fluid(jet) - E fluid(bg) at the last frame both run")
    same = [i for i in range(fg["arr"].shape[0]) if same_showers(fg, fc, i)]
    for name, f in (("GPU", fg), ("CPU", fc)):
        r = []
        for i in range(f["arr"].shape[0]):
            b = int(f["diag/bg_id"][i])
            k = min(int(f["ntau_freezeout"][i]), int(f["ntau_freezeout_bg"][b])) - 1
            w, _ = wake(eos, f, i, k)
            dep = (f["diag/E_droplets"][i] - f["diag/E_droplets_late"][i]
                   - f["diag/E_droplets_early"][i])
            r.append((w, dep))
        q = np.array([w / d for w, d in r])
        print(f"  {name}: " + "  ".join(f"ev{i} {w:.1f}/{d:.1f} GeV" for i, (w, d) in enumerate(r))
              + f"  | wake/deposit {q.mean():.3f} +- {q.std(ddof=1) / np.sqrt(len(q)):.3f}")
    print(f"  same showers on both paths (droplets within 1e-2): events {same}")

    if same:
        print("\n=== same-shower events: wake de = e_jet - e_bg cell by cell")
        print("  ev  tau | wake GPU CPU [GeV] | |de_G - de_C| / |de_C| (L2) | peak de GPU CPU")
        for i in same:
            b = int(fg["diag/bg_id"][i])
            n = min(int(fg["ntau_freezeout"][i]), int(fg["ntau_freezeout_bg"][b]),
                    int(fc["ntau_freezeout"][i]), int(fc["ntau_freezeout_bg"][b]))
            for t0 in (2.0, 4.0, 6.0, 8.0, 10.0):
                k = int(round((t0 - float(fg.attrs["tau_min"])) / float(fg.attrs["dtau"])))
                if k >= n:
                    continue
                (wg, dg), (wc, dc) = wake(eos, fg, i, k), wake(eos, fc, i, k)
                err = np.sqrt(((dg - dc) ** 2).sum() / (dc ** 2).sum())
                print(f"  {i:2d}  {t0:4.1f} | {wg:7.2f} {wc:7.2f} | {err:.3f} | "
                      f"{dg.max():.3f} {dc.max():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
