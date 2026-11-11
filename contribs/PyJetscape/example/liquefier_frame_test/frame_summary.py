#!/usr/bin/env python3
"""
frame_summary.py -- the liquefier frame test (make_data.sh) reduced to one small file: the same
events in the lab, bjorken and fluid frames, at hydro and at hadron level.

    python frame_summary.py DATA [-o DATA/summary.npz] [--frames lab bjorken fluid]

DATA holds one directory per frame (make_data.sh), each with the pair file, the particlize
file and the hadron files of the same events.  Written (npz; f = frame):

checks       backgrounds and droplets identical in all frames
tau          the stored frames' tau [fm/c]
f/dE_full    (events, tau) the fluid's lab energy, jet leg - background, whole grid [GeV]
f/dE_win     the same at |eta_s| < 1
f/dE_eta     (events, tau, eta_s) the same per eta_s slice
f/prof       (events, PROF_TAUS, eta_s) the wake's energy per eta_s slice [GeV]
f/map        (events, MAP_TAUS, x, y) the wake's energy density summed over |eta_s| < 0.5
             times deta [GeV/fm^3] (around each event's map_center, 0 unless a caller such as
             gun_summary.py gives one)
drops, fluid, ev   the droplets (tau, x, y, eta, E, px, py, pz), the background fluid where
             each was made (vx, vy, vz, e: the fluid run's source/fluid), their event
f/flux       the C++ sampled flux of each droplet on MUSIC's grid (source/flux)
dep/f        (droplets, DEP_COLUMNS) each droplet's deposit on MUSIC's grid by the Python port
             (liquefier_kernel.py), for all three frames whichever runs DATA holds (the
             droplets and the background are the same): its centroid and widths, the cells
             it covers, and the share
             of its energy landing in cells MUSIC cannot invert (|M| >= T^tautau after the
             deposit, with the background interpolated from arr_bg)
f/had_*      hadron-level wake (bulk jet - background, correlated sampling), charged hadrons:
             dN/deta, the pT spectrum at |eta| < 1, dN/d(dphi) to the event's axis at
             |eta| < 1 (the trigger Z's phi when the events have one, the recoil then near
             pi; else the leading parton's), N and sum pT at |eta| < 1 and at |eta| < 2.4 over
             all events (had_tot: (4 quantities, value/error); the reader's error, which counts
             each background's samples once for all its events) and per event (had_*_ev).
             Two frames' hadrons are independent samples (their own background samples too):
             the error of a frame difference is the two totals' errors in quadrature; the
             spread of per-event differences misses the background samples' common offset.

The energy is the ideal part of T^{mu nu} through the tau surface, as in tutorial 06 and
liquefier_replay.py: E = sum tau dx dy deta [cosh(eta) T^00 - sinh(eta) T^03].
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import liquefier_kernel as lk  # noqa: E402

EOS_NPZ = os.path.join(HERE, "..", "..", "..", "..", "tutorials", "pyjetscape", "_common",
                       "eos_hotqcd_91.npz")
FRAMES = ("lab", "bjorken", "fluid")
PROF_TAUS = (2.0, 3.0, 4.0, 6.0, 8.0)
MAP_TAUS = (3.0, 6.0)
ETA_WIN = 1.0
DEP_COLUMNS = ("flux", "cells", "eta_mean", "eta_rms", "x_mean", "y_mean", "r_rms", "age",
               "bad_share", "max_ratio")
ETA_EDGES = np.arange(-5.0, 5.01, 0.5)
PT_EDGES = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
DPHI_EDGES = np.linspace(-np.pi / 2, 3 * np.pi / 2, 25)


def pressure_of(npz=EOS_NPZ):
    eos = np.load(npz)

    def pressure(e):
        e = np.maximum(e, 1e-12)
        low = e * eos["P"][0] / eos["e"][0]
        return np.where(e < eos["e"][0], low, np.interp(np.log(e), np.log(eos["e"]), eos["P"]))
    return pressure


def energy_per_eta(fr, tau, etas, dx, dy, deta, pressure):
    """(neta,) lab energy per eta_s slice of one frame (4, nx, ny, neta)."""
    e, vx, vy, vz = (fr[c].astype(np.float64) for c in range(4))
    g2 = 1.0 / (1.0 - np.clip(vx * vx + vy * vy + vz * vz, 0, 1 - 1e-12))
    p = pressure(e)
    w = (e + p) * g2
    dens = np.cosh(etas) * (w - p) - np.sinh(etas) * w * vz
    return dens.sum(axis=(0, 1)) * tau * dx * dy * deta


def stem_of(d):
    names = [n for n in os.listdir(d) if n.endswith(".h5") and "_" in n
             and not any(t in n for t in ("_particlize", "_hadrons", "_primary"))]
    if len(names) != 1:
        raise SystemExit(f"{d}: expected one pair file, found {names}")
    return os.path.join(d, names[0])


# ── hydro ───────────────────────────────────────────────────────────────────────────────
def hydro(pair, pressure, bg_curve=None, map_center=None):
    import h5py
    import hdf5plugin  # noqa: F401
    out = {}
    with h5py.File(pair, "r") as f:
        a = f.attrs
        etas = a["eta_min"] + a["deta"] * np.arange(a["neta"])
        dx, dy, de, t0, dt = (float(a[k]) for k in ("dx", "dy", "deta", "tau_min", "dtau"))
        nev, ntau = f["arr"].shape[0], f["arr"].shape[-1]
        taus = t0 + dt * np.arange(ntau)
        win = np.abs(etas) < ETA_WIN
        centers = np.zeros(nev) if map_center is None else np.asarray(map_center, float)
        nfj = f["ntau_freezeout"][:]
        nfb = f["ntau_freezeout_bg"][:]
        rows = f["arr_bg_rows"][:] if "arr_bg_rows" in f else np.arange(nev)
        # the background's energy per eta slice, once per background
        bg = {} if bg_curve is None else bg_curve
        for i in range(nev):
            key = int(rows[i])
            if key in bg:
                continue
            per = np.full((ntau, len(etas)), np.nan)
            for k in range(int(nfb[i])):
                per[k] = energy_per_eta(f["arr_bg"][i, ..., k], taus[k], etas, dx, dy, de, pressure)
            bg[key] = per
        dE_full = np.full((nev, ntau), np.nan)
        dE_win = np.full((nev, ntau), np.nan)
        dE_eta = np.full((nev, ntau, len(etas)), np.nan)
        prof = np.full((nev, len(PROF_TAUS), len(etas)), np.nan)
        maps = np.full((nev, len(MAP_TAUS), int(a["nx"]), int(a["ny"])), np.nan)
        kp = [int(round((t - t0) / dt)) for t in PROF_TAUS]
        km = [int(round((t - t0) / dt)) for t in MAP_TAUS]
        for i in range(nev):
            n = int(min(nfj[i], nfb[i]))
            b = bg[int(rows[i])]
            for k in range(n):
                fr = f["arr"][i, ..., k]
                per = energy_per_eta(fr, taus[k], etas, dx, dy, de, pressure) - b[k]
                dE_full[i, k], dE_win[i, k] = per.sum(), per[win].sum()
                dE_eta[i, k] = per
                if k in kp:
                    prof[i, kp.index(k)] = per
                if k in km:
                    mid = np.abs(etas - centers[i]) < 0.5
                    maps[i, km.index(k)] = ((fr[0] - f["arr_bg"][i, 0, ..., k])[..., mid]
                                            .sum(axis=-1) * de)
        off = f["source/offsets"][:]
        out.update(tau=taus, etas=etas, x=a["x_min"] + dx * np.arange(a["nx"]),
                   y=a["y_min"] + dy * np.arange(a["ny"]), dE_full=dE_full, dE_win=dE_win, dE_eta=dE_eta,
                   prof=prof, map=maps, drops=f["source/droplets"][:], offsets=off,
                   flux=f["source/flux"][:] if "source/flux" in f else np.full(off[-1], np.nan),
                   fluid=f["source/fluid"][:] if "source/fluid" in f else None,
                   frame=lk.frame_of(f), bg_rows=rows, ntau_fo=nfj, ntau_fo_bg=nfb,
                   bg_probe=f["arr_bg"][0, 0, ..., min(20, int(nfb[0]) - 1)][:],
                   tau0=f["diag/tau0_music"][:], e_min=float(a.get("liquefier_fluid_e_min",
                                                                   lk.FLUID_E_MIN)))
    return out, bg


# ── droplets: their deposits on MUSIC's grid ────────────────────────────────────────────
def deposits(pair, frames, fluid, pressure):
    """(droplets, DEP_COLUMNS) per frame, by the Python port on MUSIC's native grid."""
    import h5py
    import hdf5plugin  # noqa: F401
    from scipy.interpolate import RegularGridInterpolator
    with h5py.File(pair, "r") as f:
        g, k = lk.grid_of(f), lk.kernel_of(f)
        d, off = f["source/droplets"][:], f["source/offsets"][:]
        tau0 = f["diag/tau0_music"][:]
        a = f.attrs
        axes = [a[n + "_min"] + a["d" + n] * np.arange(a["n" + n]) for n in ("x", "y", "eta")]
        t0, dt = float(a["tau_min"]), float(a["dtau"])
        nfb = f["ntau_freezeout_bg"][:]
        rows = f["arr_bg_rows"][:] if "arr_bg_rows" in f else np.arange(len(off) - 1)
        cache = {}

        def background(i, tq, pts):
            """e, vx, vy, vz of the background at tau tq (nearest stored frame), points (n, 3)."""
            kf = int(round((tq - t0) / dt))
            if kf < 0 or kf >= nfb[i]:
                return None
            key = (int(rows[i]), kf)
            if key not in cache:
                if len(cache) > 8:
                    cache.clear()
                fr = f["arr_bg"][i, ..., kf].astype(np.float64)
                cache[key] = [RegularGridInterpolator(axes, fr[c], bounds_error=False,
                                                      fill_value=0.0) for c in range(4)]
            return np.array([ip(pts) for ip in cache[key]])

        xs = g["x_min"] + g["dx"] * np.arange(g["nx"])
        ys = g["y_min"] + g["dy"] * np.arange(g["ny"])
        es = g["eta_min"] + g["deta"] * np.arange(g["neta"])
        out = {fr: np.full((len(d), len(DEP_COLUMNS)), np.nan) for fr in frames}
        ev = np.repeat(np.arange(len(off) - 1), np.diff(off))
        for j in range(len(d)):
            drop = d[j]
            E, px, py, pz = drop[4:]
            tq = lk.music_step(drop[0] + k.tau_delay, tau0[ev[j]])
            V = tq * g["dx"] * g["dy"] * g["deta"]
            for fr in frames:
                w, _, _, age = k.deposit(drop, g, fr, tau_q=tq, fluid=fluid[j])
                tot = w.sum()
                if not tot > 1e-6:
                    out[fr][j, 0] = tot
                    continue
                ie, ix, iy = np.nonzero(np.abs(w) > 1e-6 * np.abs(w).max())
                ww = w[ie, ix, iy] / tot
                eta = es[ie]
                em = (ww * eta).sum()
                xm, ym = (ww * xs[ix]).sum(), (ww * ys[iy]).sum()
                we = w.sum(axis=(1, 2))
                row = [tot, int((np.abs(w) > 1e-3 * np.abs(w).max()).sum()), em,
                       np.sqrt(max((ww * (eta - em) ** 2).sum(), 0.0)), xm, ym,
                       np.sqrt(max((ww * ((xs[ix] - xm) ** 2 + (ys[iy] - ym) ** 2)).sum(), 0.0)),
                       (we * age).sum() / we.sum(), np.nan, np.nan]
                B = background(ev[j], tq, np.column_stack([xs[ix], ys[iy], eta]))
                if B is not None:
                    ch, sh = np.cosh(eta), np.sinh(eta)
                    dtt = ww * (E * ch - pz * sh) / V
                    dtx, dty = ww * px / V, ww * py / V
                    dte = ww * (pz * ch - E * sh) / V
                    e, vx, vy, vz = B
                    p = pressure(e)
                    gam = 1 / np.sqrt(np.maximum(1e-12, 1 - vx * vx - vy * vy - vz * vz))
                    ut, ux, uy, ue = gam * (ch - vz * sh), gam * vx, gam * vy, gam * (vz * ch - sh)
                    Ttt = (e + p) * ut * ut - p
                    M = np.sqrt(((e + p) * ut * ux + dtx) ** 2 + ((e + p) * ut * uy + dty) ** 2
                                + ((e + p) * ut * ue + dte) ** 2)
                    bad = (Ttt + dtt) <= M
                    Edep = np.abs(ww * E)
                    row[8] = Edep[bad].sum() / max(Edep.sum(), 1e-12)
                    row[9] = np.max(np.abs(dtt) / np.maximum(Ttt, 1e-9))
                out[fr][j] = row
    return out


# ── hadrons ─────────────────────────────────────────────────────────────────────────────
HAD_TOT = ("N |eta|<1", "sum pT |eta|<1", "N |eta|<2.4", "sum pT |eta|<2.4")


def hadron_totals(r, charged_mid, charged_cms):
    """(4, 2): N and sum pT at |eta| < 1 and < 2.4 per event over all events, value and error."""
    out = []
    for m in (charged_mid, charged_cms):
        for w in (None, "pt"):
            v, e = r.jet_minus_background("pt", np.array([0.0, 1e9]), mask=m, weights=w,
                                          per_event=False)
            out.append((v[0], e[0]))
    return np.array(out)


def hadrons(d, totals_only=False):
    from jetscape.hadrons_h5 import CHARGED, HadronFileReader
    r = HadronFileReader(d)

    def charged(ev, info):
        return np.isin(np.abs(ev.pid), CHARGED)

    def charged_mid(ev, info):
        return charged(ev, info) & (np.abs(ev.eta) < ETA_WIN)

    def axis_phi(info):
        """The trigger Z's phi, else the leading parton's (leptons and photons excluded)."""
        z = getattr(info, "z", None)
        if z and np.isfinite(z.get("z_phi", np.nan)):
            return z["z_phi"]
        ini = info.initiators()
        ini = ini[~np.isin(np.abs(ini[:, 1]), (11, 12, 13, 14, 15, 16, 22))]
        pt = np.hypot(ini[:, 3], ini[:, 4])
        L = ini[int(np.argmax(pt))]
        return np.arctan2(L[4], L[3])

    def dphi_lead(ev, info):
        return np.mod(ev.phi - axis_phi(info) + np.pi / 2, 2 * np.pi) - np.pi / 2

    def charged_cms(ev, info):
        return charged(ev, info) & (np.abs(ev.eta) < 2.4)

    out = {"correlated": r.correlated, "n_events": r.n_events,
           "had_tot": hadron_totals(r, charged_mid, charged_cms)}
    if totals_only:
        return out
    i0 = r.event_info(0) if hasattr(r, "event_info") else None
    out["had_axis"] = np.array("Z" if (i0 is not None and getattr(i0, "z", None)) else "parton")
    out["had_dndeta"] = r.jet_minus_background("eta", ETA_EDGES, mask=charged, per_event=False)
    out["had_pt"] = r.jet_minus_background("pt", PT_EDGES, mask=charged_mid, per_event=False)
    out["had_dphi"] = r.jet_minus_background(dphi_lead, DPHI_EDGES, mask=charged_mid,
                                             per_event=True)
    for tag, m in (("", charged_mid), ("_cms", charged_cms)):
        n, s = [], []
        for i in range(r.n_events):
            n.append(r.jet_minus_background("pt", np.array([0.0, 1e9]), mask=m,
                                            events=[i], per_event=False))
            s.append(r.jet_minus_background("pt", np.array([0.0, 1e9]), mask=m,
                                            weights="pt", events=[i], per_event=False))
        out[f"had_N_ev{tag}"] = np.array([[a[0][0], a[1][0]] for a in n])
        out[f"had_sumpt_ev{tag}"] = np.array([[a[0][0], a[1][0]] for a in s])
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("data")
    p.add_argument("-o", "--out", default=None)
    p.add_argument("--frames", nargs="+", default=list(FRAMES))
    p.add_argument("--no-hadrons", action="store_true")
    p.add_argument("--add-hadron-totals", action="store_true",
                   help="only add had_tot to an existing summary (-o), nothing else recomputed")
    a = p.parse_args(argv)
    out_path = a.out or os.path.join(a.data, "summary.npz")
    pressure = pressure_of()
    if a.add_hadron_totals:
        old = dict(np.load(out_path, allow_pickle=False))
        for fr in a.frames:
            old[f"{fr}/had_tot"] = hadrons(os.path.join(a.data, fr), totals_only=True)["had_tot"]
            print(f"{fr}: {old[f'{fr}/had_tot'][:, 0].round(2)} +- {old[f'{fr}/had_tot'][:, 1].round(2)}",
                  flush=True)
        old["had_tot_columns"] = np.array(HAD_TOT)
        np.savez_compressed(out_path, **old)
        return 0
    summarize(a.data, out_path, a.frames, pressure, hadrons_too=not a.no_hadrons)
    return 0


def summarize(data, out_path, frames=FRAMES, pressure=None, hadrons_too=True, map_center=None,
              hadron_fn=None, extra=None):
    """The summary of DATA's frames -> out_path.  map_center: per-event eta_s of the maps;
    hadron_fn(dir) -> dict of had_* arrays (default hadrons()); extra: more arrays to save."""
    from types import SimpleNamespace
    a = SimpleNamespace(data=data, frames=list(frames), no_hadrons=not hadrons_too)
    pressure = pressure or pressure_of()
    hadron_fn = hadron_fn or hadrons
    S, H = {}, {}
    t_start = time.time()
    for fr in a.frames:
        pair = stem_of(os.path.join(a.data, fr))
        h, _ = hydro(pair, pressure, map_center=map_center)
        if h["frame"] != fr:
            raise SystemExit(f"{pair}: liquefier_frame is {h['frame']!r}, expected {fr!r}")
        H[fr] = h
        print(f"[{time.time() - t_start:6.0f} s] {fr}: hydro, {len(h['drops'])} droplets in "
              f"{len(h['offsets']) - 1} events", flush=True)
    ref = H[a.frames[0]]
    checks = {
        "droplets_identical": all(np.array_equal(H[fr]["drops"], ref["drops"]) for fr in H),
        "backgrounds_identical": all(np.array_equal(H[fr]["bg_probe"], ref["bg_probe"])
                                     for fr in H),
    }
    print("checks:", checks, flush=True)
    fluid_frame = next((H[fr] for fr in H if H[fr]["fluid"] is not None), None)
    if fluid_frame is None:
        import h5py
        import hdf5plugin  # noqa: F401
        with h5py.File(stem_of(os.path.join(a.data, a.frames[0])), "r") as f:
            fluid, origin = lk.fluid_of(f)
    else:
        fluid, origin = fluid_frame["fluid"], "file"
    checks["fluid_from"] = origin
    dep = deposits(stem_of(os.path.join(a.data, a.frames[0])), FRAMES, fluid, pressure)
    print(f"[{time.time() - t_start:6.0f} s] deposits on MUSIC's grid", flush=True)
    for fr in a.frames:
        h = H[fr]
        for key in ("dE_full", "dE_win", "dE_eta", "prof", "map", "flux", "ntau_fo"):
            S[f"{fr}/{key}"] = h[key]
        if not a.no_hadrons:
            had = hadron_fn(os.path.join(a.data, fr))
            for key, v in had.items():
                if key.startswith(("had_", "hg_")) and isinstance(v, tuple):
                    S[f"{fr}/{key}"] = np.array(v)
                elif key.startswith(("had_", "hg_")):
                    S[f"{fr}/{key}"] = v
            S[f"{fr}/had_correlated"] = np.array(had["correlated"])
            print(f"[{time.time() - t_start:6.0f} s] {fr}: hadrons", flush=True)
    off = ref["offsets"]
    np.savez_compressed(
        out_path, **S, frames=np.array(a.frames), tau=ref["tau"], etas=ref["etas"],
        x=ref["x"], y=ref["y"], drops=ref["drops"], fluid=fluid,
        ev=np.repeat(np.arange(len(off) - 1), np.diff(off)), offsets=off,
        ntau_fo_bg=ref["ntau_fo_bg"], tau0=ref["tau0"], e_min=ref["e_min"],
        prof_taus=np.array(PROF_TAUS), map_taus=np.array(MAP_TAUS), eta_win=ETA_WIN,
        dep_columns=np.array(DEP_COLUMNS), eta_edges=ETA_EDGES, pt_edges=PT_EDGES,
        dphi_edges=DPHI_EDGES, checks=np.array(str(checks)), had_tot_columns=np.array(HAD_TOT),
        **{f"dep/{fr}": dep[fr] for fr in FRAMES}, **(extra or {}))
    print(f"-> {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB, {time.time() - t_start:.0f} s)")
    return checks


if __name__ == "__main__":
    sys.exit(main())
