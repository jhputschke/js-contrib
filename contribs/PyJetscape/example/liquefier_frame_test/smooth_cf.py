#!/usr/bin/env python3
"""
smooth_cf.py -- the wake's hadron spectra without sampling noise: a smooth Cooper-Frye integral
over each event's jet-leg freeze-out surface minus its background's, from the particlize files
of make_gun_data.sh (or make_data.sh).

    python smooth_cf.py DATA [-o DATA/smooth_cf.npz] [--workers N]

Per frame directory of DATA, per event, per species (direct pi+, K+, p): dN/(dy pT dpT) on
PT_GRID within |y - y_c| < Y_HALF, y_c the event's gun rapidity (0 for events without a gun),
jet surface minus background surface.  Written: f/spec (events, species, pT) and the grids.

The integrand is iSS's (emissionfunction.cpp, the surface fields as iSS receives them from
X-SCAPE: particlize_h5.SURFACE_COLUMNS):

    dN/dy d^2pT = g / (2 pi)^3 sum_cells tau f0(p.u / T) (pt da0 + px da1 + py da2 + pz da3 / tau)

with pt = mT cosh(y - eta), pz = mT sinh(y - eta), p.u = pt u0 - px u1 - py u2 - pz u3.
Simplifications, the same in every frame (so they cancel in frame comparisons): Boltzmann f0
(the phi integral then analytic: I0 and I1), the ideal part only (no shear or bulk delta f),
direct (thermal) hadrons only (no resonance decays), mu = 0.  Units: dN/(dy pT dpT) per event
[GeV^-2], the surface's da [fm^3] converted with hbar c.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

HBARC = 0.19733
SPECIES = (("pi+", 0.13957, 1.0), ("K+", 0.49368, 1.0), ("p", 0.93827, 2.0))
PT_GRID = np.linspace(0.1, 4.0, 30)
Y_HALF = 1.0
NY = 7                                     # Gauss-Legendre points in |y - y_c| < Y_HALF
ETA_CUT = 3.0                              # cells farther than this from y_c in eta: negligible
COLS = ("tau", "x", "y", "eta", "ds0", "ds1", "ds2", "ds3", "u0", "u1", "u2", "u3", "e", "T")


def spectra(cells, yc, chunk=200_000):
    """(species, pT) dN/(dy pT dpT) averaged over |y - yc| < Y_HALF, one surface (N, 32)."""
    from scipy.special import i0e, i1e
    c = {k: cells[:, i].astype(np.float64) for i, k in enumerate(COLS)}
    keep = (np.abs(c["eta"] - yc) < ETA_CUT) & (c["T"] > 0)
    c = {k: v[keep] for k, v in c.items()}
    xg, wg = np.polynomial.legendre.leggauss(NY)
    ys = yc + Y_HALF * xg
    wy = wg / 2.0                                       # average over the window
    out = np.zeros((len(SPECIES), len(PT_GRID)))
    n = len(c["tau"])
    for s0 in range(0, n, chunk):
        sl = slice(s0, s0 + chunk)
        tau, eta, T = c["tau"][sl], c["eta"][sl], c["T"][sl]
        da0, da1, da2, da3 = c["ds0"][sl], c["ds1"][sl], c["ds2"][sl], c["ds3"][sl]
        u0, u1, u2, u3 = c["u0"][sl], c["u1"][sl], c["u2"][sl], c["u3"][sl]
        uperp = np.hypot(u1, u2)
        phiu = np.arctan2(u2, u1)
        dperp = da1 * np.cos(phiu) + da2 * np.sin(phiu)    # the in-flow part of (da1, da2)
        for js, (_, m, g) in enumerate(SPECIES):
            mT = np.sqrt(PT_GRID ** 2 + m * m)[:, None]     # (pT, 1)
            pT = PT_GRID[:, None]
            acc = np.zeros(len(PT_GRID))
            for y, w in zip(ys, wy):
                ch, sh = np.cosh(y - eta), np.sinh(y - eta)
                pt, pz = mT * ch, mT * sh                  # (pT, cells)
                A = pt * u0 - pz * u3                      # p.u without the transverse part
                a = pT * uperp / T                         # the Bessel argument
                expo = np.exp(-(A - pT * uperp) / T)       # e^{-(p.u)/T} / e^{a} (scaled Bessels)
                pds = (pt * da0 + pz * da3 / tau) * i0e(a) + pT * dperp * i1e(a)
                acc += w * (expo * pds * tau).sum(axis=1)
            out[js] += g / (2 * np.pi) ** 2 * acc / HBARC ** 3   # 2 pi from phi, (2 pi)^3
    return out


def _job(args):
    path, group, lo, hi, yc = args
    import h5py
    import hdf5plugin  # noqa: F401
    with h5py.File(path, "r") as f:
        cells = f[f"surface/{group}/cells"][lo:hi]
    return args, spectra(cells, yc)


def main(argv=None):
    import multiprocessing as mp
    import h5py
    import hdf5plugin  # noqa: F401
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("data")
    p.add_argument("-o", "--out", default=None)
    p.add_argument("--frames", nargs="+", default=["lab", "bjorken", "fluid"])
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--events", type=int, default=None, help="only the first N events (tests)")
    a = p.parse_args(argv)
    out_path = a.out or os.path.join(a.data, "smooth_cf.npz")
    jobs, meta = [], {}
    for fr in a.frames:
        d = os.path.join(a.data, fr)
        path = os.path.join(d, next(n for n in os.listdir(d) if n.endswith("_particlize.h5")))
        with h5py.File(path, "r") as f:
            off = f["surface/jet/offsets"][:]
            boff = f["surface/bg/offsets"][:]
            nev = len(off) - 1 if a.events is None else min(a.events, len(off) - 1)
            ev = f["events"]
            yc = ev["gun_y"][:nev] if "gun_y" in ev else np.zeros(nev)
            unit = ev["bg_unit"][:nev] if "bg_unit" in ev else np.zeros(nev, int)
        meta[fr] = (nev, yc, unit)
        for i in range(nev):
            jobs.append((path, "jet", int(off[i]), int(off[i + 1]), float(yc[i]), fr, "jet", i))
        for u in np.unique(unit):
            for y in np.unique(yc[unit == u]):
                jobs.append((path, "bg", int(boff[u]), int(boff[u + 1]), float(y), fr, "bg",
                             (int(u), float(y))))
    t0 = time.time()
    res = {}
    with mp.get_context("spawn").Pool(a.workers) as pool:
        keyed = {x[:5]: x[5:] for x in jobs}            # the surface -> (frame, leg, key)
        it = pool.imap_unordered(_job, list(keyed))
        for n, (args, spec) in enumerate(it, 1):
            res[keyed[args]] = spec
            if n % 10 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)} surfaces, {time.time() - t0:.0f} s", flush=True)
    S = {}
    for fr in a.frames:
        nev, yc, unit = meta[fr]
        jet = np.array([res[(fr, "jet", i)] for i in range(nev)])
        bg = np.array([res[(fr, "bg", (int(unit[i]), float(yc[i])))] for i in range(nev)])
        S[f"{fr}/spec"] = jet - bg
        S[f"{fr}/bg"] = bg
    np.savez_compressed(out_path, **S, pt=PT_GRID, species=np.array([s[0] for s in SPECIES]),
                        y_half=Y_HALF, y_center=meta[a.frames[0]][1])
    print(f"-> {out_path} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
