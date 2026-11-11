#!/usr/bin/env python3
"""
gun_summary.py -- the single-parton frame test (make_gun_data.sh) reduced to one small file:
one quark per event from the origin at fixed (pT, y) points, the same events in the lab, bjorken
and fluid frames on one LHC background.

    python gun_summary.py DATA [-o DATA/summary.npz] [--no-hadrons]

DATA holds one directory per frame, as for frame_summary.py, whose arrays this writes too
(hydro per eta_s slice and tau, the droplets and their deposits in all three frames; f/map
centred on each event's gun rapidity).  Added (f = frame, K points):

gun_points       (K, 2) the gun's (pT [GeV], y) points; gun_phi, gun_parID
gun_point, gun_pt, gun_y   (events,) each event's point (diag/ of the pair file)
f/hg_dndeta      (K, 2, n_eta) charged wake hadrons (bulk jet - background, correlated
                 sampling, no fragments) per event per HG_ETA_EDGES bin: value, error
f/hg_pt          (K, 2, n_pt) their pT spectrum at |eta - y| < 1 (PT_EDGES)
f/hg_dphi        (K, 2, n_dphi) dN/d(phi - phi_gun) at |eta - y| < 1 (DPHI_EDGES: the gun
                 at 0)
f/hg_tot         (K, 6, 2) per event: N and sum pT at |eta - y| < 1, at |eta| < 2.4 (CMS's
                 acceptance) and at |eta| < 1; value, error (the reader's: each background
                 sample counted once)

Each frame hadronizes its own samples, so the error of a frame difference is the two errors in
quadrature.  The hydro (f/dE_eta etc.) has no sampling noise and the same droplets in every
frame: its differences are exact.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import frame_summary as fs  # noqa: E402

HG_ETA_EDGES = np.arange(-7.0, 7.01, 0.5)
HG_TOT = ("N |eta-y|<1", "sum pT |eta-y|<1", "N |eta|<2.4", "sum pT |eta|<2.4",
          "N |eta|<1", "sum pT |eta|<1")


def gun_of(pair):
    """(points (K, 2), per-event point, pT, y, phi, parID) from the pair file."""
    import h5py
    import hdf5plugin  # noqa: F401
    with h5py.File(pair, "r") as f:
        a = f.attrs
        if "pgun_points" not in a:
            raise SystemExit(f"{pair}: no pgun_points (not a run_prod_jet.py --pgun-grid job)")
        pts = np.asarray(a["pgun_points"], dtype=np.float64).reshape(-1, 2)
        k = f["diag/gun_point"][:].astype(int)
        pt, y = f["diag/gun_pt"][:], f["diag/gun_y"][:]
        phi = float(a.get("pgun_phi", 0.0))
        pid = int(a.get("pgun_id", 21))
    if not np.allclose(pts[k, 0], pt) or not np.allclose(pts[k, 1], y):
        raise SystemExit(f"{pair}: diag/gun_* do not match pgun_points")
    return pts, k, pt, y, phi, pid


def _point_hadrons(job):
    """One frame directory's hadron-level wake at one point -> (d, kp, dict)."""
    d, kp, y, evs, phi_gun = job
    from jetscape.hadrons_h5 import CHARGED, HadronFileReader
    r = HadronFileReader(d)

    def charged(ev, info):
        return np.isin(np.abs(ev.pid), CHARGED)

    def near(ev, info):
        return charged(ev, info) & (np.abs(ev.eta - y) < 1.0)

    def cms(ev, info):
        return charged(ev, info) & (np.abs(ev.eta) < 2.4)

    def mid(ev, info):
        return charged(ev, info) & (np.abs(ev.eta) < 1.0)

    def dphi(ev, info):
        return np.mod(ev.phi - phi_gun + np.pi / 2, 2 * np.pi) - np.pi / 2

    out = {"hg_dndeta": r.jet_minus_background("eta", HG_ETA_EDGES, mask=charged, events=evs,
                                               per_event=False),
           "hg_pt": r.jet_minus_background("pt", fs.PT_EDGES, mask=near, events=evs,
                                           per_event=False),
           "hg_dphi": r.jet_minus_background(dphi, fs.DPHI_EDGES, mask=near, events=evs,
                                             per_event=False)}
    tot = []
    for m in (near, cms, mid):
        for w in (None, "pt"):
            v, e = r.jet_minus_background("pt", np.array([0.0, 1e9]), mask=m, weights=w,
                                          events=evs, per_event=False)
            tot.append((v[0], e[0]))
    out["hg_tot"] = tot
    return d, kp, out, r.correlated, r.n_events


def all_hadrons(dirs, points, point_of, phi_gun, workers=None):
    """{dir: dict of hg_* arrays} for every frame directory, the (frame, point) pairs in
    parallel processes (one reader call reads ~5 events x the samples; ~9 calls per pair)."""
    import multiprocessing as mp
    import time
    jobs = [(d, kp, float(y), [int(i) for i in np.nonzero(point_of == kp)[0]], phi_gun)
            for d in dirs for kp, (_, y) in enumerate(points)]
    workers = workers or max(1, min(len(jobs), (os.cpu_count() or 4) - 2))
    res = {d: {} for d in dirs}
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers) as pool:
        for n, (d, kp, out, corr, nev) in enumerate(pool.imap_unordered(_point_hadrons, jobs), 1):
            if nev != len(point_of):
                raise SystemExit(f"{d}: {nev} hadron events, {len(point_of)} in the pair file")
            res[d][kp] = (out, corr)
            if n % 5 == 0 or n == len(jobs):
                print(f"  hadrons: {n}/{len(jobs)} (frame, point) pairs, {time.time() - t0:.0f} s",
                      flush=True)
    final = {}
    for d in dirs:
        K = len(points)
        final[d] = {k: np.array([res[d][kp][0][k] for kp in range(K)])
                    for k in ("hg_dndeta", "hg_pt", "hg_dphi", "hg_tot")}
        final[d]["correlated"] = res[d][0][1]
    return final


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("data")
    p.add_argument("-o", "--out", default=None)
    p.add_argument("--no-hadrons", action="store_true")
    p.add_argument("--workers", type=int, default=None,
                   help="processes for the hadrons (default: cores - 2)")
    a = p.parse_args(argv)
    out_path = a.out or os.path.join(a.data, "summary.npz")
    ref = fs.stem_of(os.path.join(a.data, fs.FRAMES[0]))
    pts, k, pt, y, phi, pid = gun_of(ref)
    for fr in fs.FRAMES[1:]:
        o = gun_of(fs.stem_of(os.path.join(a.data, fr)))
        if not (np.array_equal(o[0], pts) and np.array_equal(o[1], k)):
            raise SystemExit(f"{fr}: other gun points than {fs.FRAMES[0]}")
    print(f"gun: parID {pid}, phi {phi:g}, {len(pts)} points, {len(k)} events", flush=True)
    had = {}
    if not a.no_hadrons:
        had = all_hadrons([os.path.join(a.data, fr) for fr in fs.FRAMES], pts, k, phi,
                          workers=a.workers)
    fs.summarize(
        a.data, out_path, fs.FRAMES, hadrons_too=not a.no_hadrons, map_center=y,
        hadron_fn=lambda d: had[d],
        extra=dict(gun_points=pts, gun_point=k, gun_pt=pt, gun_y=y, gun_phi=phi,
                   gun_parID=pid, hg_eta_edges=HG_ETA_EDGES, hg_tot_columns=np.array(HG_TOT)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
