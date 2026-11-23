#!/usr/bin/env python3
"""
analysis/OO-RHIC/oo_bias_jets.py -- the selection-only centrality bias of O+O jets as a function of
the jet pT (R4b, the full test, continued): the events of oo_bias.py (`mb`, `jets`; jet in vacuum,
--vacuum-jet), classes as oo_bias.py builds them, the jet yield per class from reconstructed jets
instead of the hard scattering itself.

    python oo_bias_jets.py                 # -> results/oo_bias_jets.json
    python oo_bias_jets.py --out DIR       # another output root of oo_bias.py (DIR/jets, DIR/mb)
    python oo_bias_jets.py --sample jets_quench   # the jet in the medium -> results/oo_bias_jets_jets_quench.json

Jets: anti-kT R = 0.4 (FastJet) on the hard event's hadrons (the jet_frag files: shower and
iColored remnant hadrons, no bulk; the jet is in vacuum, so there is nothing to subtract),
|eta_jet| < 0.7 as sPHENIX's, constituents pT > 0.2 GeV.  Which particles depends on the files:
samples stored with --charged (mb, jets) give charged-particle jets, whose pT is roughly 0.6-0.7 of
a full jet's (sPHENIX's 19-60 GeV correspond to about 12-40 GeV); samples with all hadrons
(jets_quench) give full jets, all final-state particles except neutrinos, as sPHENIX's calorimeter
jets.

Per jet-pT bin each jet counts with its event's weight N_coll x sigma_k / N_k (k = the pTHat
window; Pythia's cross section per window over the window's events): R_AA^vac per class =
(N_coll-weighted jet fraction in the class) / (N_coll-weighted minimum-bias fraction), R_CP from
the same sums (hstat.raa_vac, hstat.rcp; errors from both samples, jets of one event summed
before squaring).  The windows 9-11, 18-23, 30-40, 40-55 GeV leave gaps (11-18, 23-30 GeV), so the
"all windows" spectrum is not inclusive in pTHat; per window the same quantities are given too.
Classes: sPHENIX's MBD (3.51 < |eta| < 4.61, both arms) in 0-20/20-40/40-60/60-80/80-100 %, the EPD
(2.1 < |eta| < 5.1) in the same classes and in STAR's.  The jet events' class multiplicity counts
their bulk and their hard event's hadrons (oo_bias.load).  Model results, not vetted physics.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hstat as H  # noqa: E402
import oo_bias as B  # noqa: E402

INVISIBLE = (12, 14, 16)                              # neutrinos
JET_R = 0.4
JET_ETA = 0.7
CONST_PTMIN = 0.2
PT_EDGES = np.array([5, 8, 11, 14, 18, 23, 30, 40, 55.0])
RCP_PAIRS = {"sph": [(0, 3), (0, 2), (1, 3)], "star": [(0, 4), (0, 3), (1, 4)]}
ESTIMATORS = (("mbd", "mbd", B.SPH_CLASSES), ("epd_sph", "epd", B.SPH_CLASSES), ("epd", "epd", B.STAR_CLASSES))


def file_jets(path):
    """Per event of one jet_frag file: the number of jets in each pT bin, the pTHat window, and the
    file's per-window cross sections and accepted events."""
    import awkward as ak
    import fastjet
    import hdf5plugin  # noqa: F401
    import h5py
    with h5py.File(path) as h:
        n = int(h.attrs["nunits_written"])
        charged_only = bool(h.attrs.get("charged_only", False))
        p, counts, pid = B._units(h, n, charged=False, with_pid=True)
        win = h["units/pthat_bin"][:n].astype(np.int64)
        sig = np.asarray(h.attrs["pthat_bin_sigma_gen"], float)
        nacc = np.asarray(h.attrs["pthat_bin_n_accepted"], float)
    E, px, py, pz = p.T
    pt = np.hypot(px, py)
    pa = np.sqrt(px * px + py * py + pz * pz)
    eta = 0.5 * np.log(np.clip(pa + pz, 1e-12, None) / np.clip(pa - pz, 1e-12, None))
    keep = (pt > CONST_PTMIN) & (np.abs(eta) < JET_ETA + JET_R + 0.1) & ~np.isin(np.abs(pid), INVISIBLE)
    ev = np.repeat(np.arange(n), counts)[keep]
    cnt = np.bincount(ev, minlength=n)
    P = p[keep]
    parts = ak.zip({"px": ak.unflatten(P[:, 1], cnt), "py": ak.unflatten(P[:, 2], cnt),
                    "pz": ak.unflatten(P[:, 3], cnt), "E": ak.unflatten(P[:, 0], cnt)})
    cs = fastjet.ClusterSequence(parts, fastjet.JetDefinition(fastjet.antikt_algorithm, JET_R))
    jets = cs.inclusive_jets(min_pt=float(PT_EDGES[0]))
    nj = ak.to_numpy(ak.num(jets))
    out = np.zeros((n, len(PT_EDGES) - 1))
    if nj.sum():
        jx, jy, jz = (ak.to_numpy(ak.flatten(jets[k])) for k in ("px", "py", "pz"))
        jev = np.repeat(np.arange(n), nj)
        jpt = np.hypot(jx, jy)
        jp = np.sqrt(jx * jx + jy * jy + jz * jz)
        jeta = 0.5 * np.log(np.clip(jp + jz, 1e-12, None) / np.clip(jp - jz, 1e-12, None))
        ib = np.digitize(jpt, PT_EDGES) - 1
        ok = (np.abs(jeta) < JET_ETA) & (ib >= 0) & (ib < len(PT_EDGES) - 1)
        np.add.at(out, (jev[ok], ib[ok]), 1.0)
    return out, win, sig, nacc, charged_only


def classes(x_mb, x_jet, cls_def, rng):
    nc = len(cls_def)
    xm = x_mb + rng.random(len(x_mb))
    cuts = np.percentile(xm, [100 - hi for _, hi in cls_def])

    def cls(v):
        c = np.full(len(v), nc - 1)
        for k in range(nc - 1, -1, -1):
            c[v >= cuts[k]] = k
        return c
    return cls(xm), cls(x_jet + rng.random(len(x_jet)))


def per_bin(cm, ncoll_mb, cj, w_ev, jets_ev, cls_def, pairs):
    """R_AA^vac and R_CP per jet-pT bin for jets of events with weights w_ev."""
    nc = len(cls_def)
    out = {"raa_vac": [], "raa_err": [], "rcp": {}, "n_jets": []}
    labs = [f"{a}-{b}" for a, b in cls_def]
    for b in range(jets_ev.shape[1]):
        wj = w_ev * jets_ev[:, b]
        rv = H.raa_vac(cm, ncoll_mb, cj, wj, nc)
        out["raa_vac"].append(rv["raa"].tolist())
        out["raa_err"].append(rv["err"].tolist())
        out["n_jets"].append(int(jets_ev[:, b].sum()))
        for a, c in pairs:
            out["rcp"].setdefault(f"{labs[a]}/{labs[c]}", []).append(list(H.rcp(rv, a, c)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="output root of oo_bias.py (default its OUT)")
    ap.add_argument("-j", type=int, default=8)
    ap.add_argument("--sample", default="jets", help="jet sample of oo_bias.py (jets, jets_quench, ...)")
    a = ap.parse_args(argv)
    if a.out:
        B.OUT = Path(a.out)
    files = sorted(glob.glob(str(B.OUT / a.sample / "*_hadrons_jet_frag.h5")))
    if not files:
        sys.exit(f"oo_bias_jets.py: no jet files under {B.OUT / a.sample}")
    mb, jet = B.load("mb"), B.load(a.sample)       # the estimator counts, same file order as below
    with ProcessPoolExecutor(max_workers=a.j) as ex:
        res = list(ex.map(file_jets, files))
    jets_ev = np.concatenate([r[0] for r in res])
    win = np.concatenate([r[1] for r in res])
    if len(win) != len(jet["glauber_ncoll"]) or not np.array_equal(win, jet["pthat_bin"]):
        sys.exit(f"oo_bias_jets.py: the jet files and oo_bias.load('{a.sample}') do not line up")
    kinds = {r[4] for r in res}
    if len(kinds) != 1:
        sys.exit("oo_bias_jets.py: the sample mixes charged-only and all-hadron files")
    charged_only = kinds.pop()
    nacc = np.sum([r[3] for r in res], axis=0)
    sigma = np.sum([r[2] * r[3] for r in res], axis=0) / np.maximum(nacc, 1)
    nev = np.bincount(win, minlength=len(sigma)).astype(float)
    s_k = sigma / np.maximum(nev, 1)
    w_all = jet["glauber_ncoll"] * s_k[win]
    rng = np.random.default_rng(11)
    out = {"sample": a.sample,
           "jet_def": {"R": JET_R, "eta_max": JET_ETA, "const_ptmin": CONST_PTMIN, "charged_only": charged_only,
                       "particles": "charged" if charged_only else "all but neutrinos"},
           "pt_edges": PT_EDGES.tolist(), "windows": B.JET_WINDOWS, "sigma": sigma.tolist(),
           "n_events": {"mb": int(len(mb["glauber_ncoll"])), "jets": int(len(win))},
           "events_per_window": nev.astype(int).tolist(), "estimators": {}}
    for key, est, cls_def in ESTIMATORS:
        cm, cj = classes(mb[f"n_{est}"], jet[f"n_{est}"], cls_def, rng)
        pairs = RCP_PAIRS["sph" if cls_def is B.SPH_CLASSES else "star"]
        e = {"classes": cls_def, "all": per_bin(cm, mb["glauber_ncoll"], cj, w_all, jets_ev, cls_def, pairs),
             "per_window": {}}
        for k, (lo, hi) in enumerate(B.JET_WINDOWS):
            m = win == k
            e["per_window"][f"{lo}-{hi}"] = per_bin(cm, mb["glauber_ncoll"], cj[m], jet["glauber_ncoll"][m].astype(float),
                                                    jets_ev[m], cls_def, pairs)
        out["estimators"][key] = e
    default = HERE / "results" / ("oo_bias_jets.json" if a.sample == "jets" else f"oo_bias_jets_{a.sample}.json")
    path = Path(os.environ.get("OO_BIAS_JETS_JSON", default))
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    lab = [f"{lo:g}-{hi:g}" for lo, hi in zip(PT_EDGES[:-1], PT_EDGES[1:])]
    print(f"minimum bias {out['n_events']['mb']}, jet events {out['n_events']['jets']} "
          f"(per window {out['events_per_window']}); {'charged' if charged_only else 'full'} jets R = {JET_R}, |eta| < {JET_ETA}")
    for key in ("mbd", "epd_sph"):
        r = out["estimators"][key]["all"]
        print(f"{key}: R_CP(0-20/60-80) vs {'charged' if charged_only else 'full'}-jet pT [GeV] (all windows, sigma-weighted)")
        for l, (v, er), nj in zip(lab, r["rcp"]["0-20/60-80"], r["n_jets"]):
            print(f"   {l:>6s}  {v:6.3f} ± {er:5.3f}   ({nj} jets)")
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
