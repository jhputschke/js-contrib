#!/usr/bin/env python
"""(jet leg) - (background leg) at hadron level, from two run_particlize.py runs.

    python delta_spectra.py --out-dir out_particlize [--ymax 1] [--npz delta.npz]

For every event k present in both runs (and with the same IC hash -- refused otherwise) it
compares per-oversample averages:

    Delta X = <X>_jet - <X>_bg,     X = sum over hadrons of E, p_T along the deposit, dN/dphi...

and checks it against what the jet actually put into the fluid (``deposited_P``, recorded by
the jet run from the solver's source accounting).

Errors.  The hadron file does not separate iSS's oversamples within an event, so the error of
an oversample average of X = sum_i x_i uses the compound-Poisson variance, Var = sum_i x_i^2
(iSS samples each species' multiplicity independently from a Poisson distribution).  The
background leg's error is what limits Delta X; it falls as 1/sqrt(its oversamples).

What the numbers mean.  Delta E over all hadrons should equal the deposited energy, up to the
energy that leaves through the open eta edges of the grid and the iSS rapidity window.  Inside
|y| < ymax only part of the deposit arrives -- the jet's own rapidity spread decides how much.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

CHARGED = {211, 321, 2212, 11, 13, 3222, 3112, 3312, 3334}


def read_hadrons(path):
    """-> list of (n_rows, 5) arrays [pid, E, px, py, pz], one per event block."""
    events, cur = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                if line.startswith("#\tEvent") or line.startswith("# Event"):
                    if cur or events:
                        events.append(np.array(cur, dtype=float).reshape(-1, 5))
                    cur = []
                continue
            t = line.split()
            if len(t) >= 7:
                cur.append((float(t[1]), float(t[3]), float(t[4]), float(t[5]), float(t[6])))
    events.append(np.array(cur, dtype=float).reshape(-1, 5))
    return events


def observables(h, n_os, phi_d, ymax, nphi):
    """Per-oversample sums and their variances for one event's hadrons."""
    pid, E, px, py, pz = h.T
    pt = np.hypot(px, py)
    y = 0.5 * np.log(np.clip(E + pz, 1e-300, None) / np.clip(E - pz, 1e-300, None))
    mid = np.abs(y) < ymax
    ch = np.isin(np.abs(pid).astype(int), list(CHARGED))
    # transverse momentum along the deposit's transverse direction
    p_par = px * np.cos(phi_d) + py * np.sin(phi_d)
    dphi = np.angle(np.exp(1j * (np.arctan2(py, px) - phi_d)))       # (-pi, pi]

    def tot(x, m=None):
        x = x if m is None else x[m]
        return x.sum() / n_os, np.sqrt((x ** 2).sum()) / n_os

    edges = np.linspace(-np.pi, np.pi, nphi + 1)
    sel = mid & ch
    hist = np.histogram(dphi[sel], bins=edges, weights=pt[sel])[0] / n_os
    hvar = np.histogram(dphi[sel], bins=edges, weights=pt[sel] ** 2)[0] / n_os ** 2
    return {
        "E_all": tot(E), "E_mid": tot(E, mid), "N_ch_mid": tot(np.ones_like(E), sel),
        "p_par_all": tot(p_par), "p_par_mid": tot(p_par, mid),
        "pt_dphi": (hist, np.sqrt(hvar)), "phi_edges": edges,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="out_particlize")
    ap.add_argument("--ymax", type=float, default=1.0, help="mid-rapidity window |y| < ymax")
    ap.add_argument("--nphi", type=int, default=16, help="bins in phi - phi_deposit")
    ap.add_argument("--npz", default=None, help="write the per-event results here")
    a = ap.parse_args(argv)

    meta = {}
    for leg in ("jet", "bg"):
        with open(os.path.join(a.out_dir, f"{leg}_events.json")) as f:
            meta[leg] = json.load(f)
    had = {leg: read_hadrons(meta[leg]["hadron_file"]) for leg in meta}
    n_os = {leg: meta[leg]["n_oversample"] for leg in meta}

    n = min(len(meta["jet"]["events"]), len(meta["bg"]["events"]))
    print(f"jet: {len(meta['jet']['events'])} event(s) x {n_os['jet']} oversamples, "
          f"bg: {len(meta['bg']['events'])} event(s) x {n_os['bg']} oversamples; "
          f"pairing {n}, |y| < {a.ymax}")

    rows, hists = [], []
    for k in range(n):
        ej, eb = meta["jet"]["events"][k], meta["bg"]["events"][k]
        if ej["ic_sha256"] != eb["ic_sha256"]:
            raise SystemExit(f"event {k}: the IC differs between the runs "
                             f"({ej['ic_sha256'][:12]} vs {eb['ic_sha256'][:12]}); rerun both "
                             f"legs with the same seed and event count")
        P = np.asarray(ej.get("deposited_P") or [0, 0, 0, 0], dtype=float)
        phi_d = float(np.arctan2(P[2], P[1]))
        oj = observables(had["jet"][k], n_os["jet"], phi_d, a.ymax, a.nphi)
        ob = observables(had["bg"][k], n_os["bg"], phi_d, a.ymax, a.nphi)

        def delta(key):
            (mj, sj), (mb, sb) = oj[key], ob[key]
            return mj - mb, float(np.hypot(sj, sb))

        r = {"event": k, "E_dep": P[0], "pT_dep": float(np.hypot(P[1], P[2]))}
        for key in ("E_all", "E_mid", "N_ch_mid", "p_par_all", "p_par_mid"):
            r[key] = delta(key)
        rows.append(r)
        hj, sj = oj["pt_dphi"]
        hb, sb = ob["pt_dphi"]
        hists.append((hj - hb, np.hypot(sj, sb)))

        print(f"\nevent {k}: deposited E = {P[0]:.2f} GeV, p_T = {r['pT_dep']:.2f} GeV "
              f"at phi = {phi_d:+.2f}")
        for key, label in (("E_all", "Delta E, all hadrons"),
                           ("E_mid", f"Delta E, |y|<{a.ymax}"),
                           ("p_par_all", "Delta p_par, all hadrons"),
                           ("p_par_mid", f"Delta p_par, |y|<{a.ymax}"),
                           ("N_ch_mid", f"Delta N_ch, |y|<{a.ymax}")):
            m, s = r[key]
            print(f"  {label:26s} = {m:+9.2f} +- {s:7.2f}   ({abs(m) / s if s else 0:4.1f} sigma)")

    if n > 1:
        print(f"\naverage over {n} events:")
        for key in ("E_all", "E_mid", "p_par_all", "p_par_mid", "N_ch_mid"):
            m = np.mean([r[key][0] for r in rows])
            s = np.sqrt(np.sum([r[key][1] ** 2 for r in rows])) / n
            print(f"  {key:10s} = {m:+9.2f} +- {s:7.2f}")
        print(f"  E_dep      = {np.mean([r['E_dep'] for r in rows]):+9.2f}")

    if a.npz:
        np.savez(a.npz, rows=json.dumps(rows, default=float),
                 phi_edges=np.linspace(-np.pi, np.pi, a.nphi + 1),
                 dpt_dphi=np.array([h[0] for h in hists]),
                 dpt_dphi_err=np.array([h[1] for h in hists]))
        print(f"\nwrote {a.npz}  (dpt_dphi: charged p_T per oversample vs phi - phi_deposit, "
              f"|y|<{a.ymax})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
