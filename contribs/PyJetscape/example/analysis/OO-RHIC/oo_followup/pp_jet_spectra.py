#!/usr/bin/env python3
"""
analysis/OO-RHIC/oo_followup/pp_jet_spectra.py -- the model's p+p jet cross sections in the experiments' own
definitions and bins, for jet_spectrum_slope.ipynb:

    python pp_jet_spectra.py --out /data/OO-RHIC_phys -j 8            # all jobs -> results/pp_jet_spectra.json
    python pp_jet_spectra.py --out /data/OO-RHIC_phys -j 8 --phase3   # phase 3 only -> results/pp_jet_spectra_phase3.json

- STAR (arXiv:2603.28695): anti-kT R = 0.5, |eta_jet| < 0.5 and 0.5 < |eta_jet| < 0.9, STAR's bins.
- sPHENIX (sPH-CONF-JET-2025-03, preliminary): anti-kT R = 0.4, |eta_jet| < 0.7, sPHENIX's bins.

Sample: pp_ref (the full chain from pp200_isr_rhictune.xml, jet in vacuum): all finished jobs,
including the 3e windows, or with --phase3 the phase-3 jobs only (as oo_analysis_rhic.py --phase3).  Jets from the hard event's visible particles (no
neutrinos), pT > 0.2 GeV, as oo_analysis_rhic.py's full jets.  There is no bulk in the jet file, so
there is no underlying event to subtract; the experiments subtract theirs.  pTHat windows stitched
with oo_analysis_rhic.stitch_pieces (sigma from Pythia, mb -> pb).  Errors from the summed squared
weights.  Model results, not vetted.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import _paths  # noqa: E402,F401  (analysis/OO-RHIC and its test directories on sys.path)
ROOT = _paths.ROOT                           # analysis/OO-RHIC: results/, refs/, data/
import oo_analysis_rhic as A  # noqa: E402
import oo_bias as B  # noqa: E402
import oo_bias_jets as J  # noqa: E402

DEFS = {"star_eta0": (0.5, 0.0, 0.5), "star_eta1": (0.5, 0.5, 0.9), "sphenix": (0.4, 0.0, 0.7)}


def edges():
    dd = json.loads((ROOT / "data" / "data.json").read_text())
    sp = json.loads((ROOT / "refs" / "sphenix_pp_jet" / "sphenix_pp_jet_prelim.json").read_text())["rows"]
    out = {}
    for k in ("star_eta0", "star_eta1"):
        e = np.array(dd[f"star_jets_{k[-4:]}"]["edges"])
        out[k] = np.append(e[:, 0], e[-1, 1])
    out["sphenix"] = np.append([r["lo"] for r in sp], sp[-1]["hi"])
    return out


def one_file(args):
    path, E = args
    import h5py
    import hdf5plugin  # noqa: F401
    with h5py.File(path) as h:
        n = int(h.attrs["nunits_written"])
        sig = np.asarray(h.attrs["pthat_bin_sigma_gen"], float)
        nacc = np.asarray(h.attrs["pthat_bin_n_accepted"], float)
        bins = [tuple(map(float, w)) for w in np.asarray(h.attrs["pthat_bins"]).reshape(-1, 2)]
        win = h["units/pthat_bin"][:n].astype(int)
        pthat = h["units/pthat"][:n].astype(float)
        pa, ca, pid = B._units(h, n, charged=False, with_pid=True)
    ev = np.repeat(np.arange(n), ca)
    pt, eta, _ = A._kin(pa)
    vis = ~np.isin(np.abs(pid), A.INVISIBLE) & (pt > A.CONST_PTMIN)
    hists = {}
    for R in sorted({d[0] for d in DEFS.values()}):
        m = vis & (np.abs(eta) < 0.9 + R + 0.1)
        je, jpt, jeta, _ = A._jets(pa[m], ev[m], n, R, None, ptmin=5.0)
        for k, (r, e0, e1) in DEFS.items():
            if r != R:
                continue
            s = (np.abs(jeta) >= e0) & (np.abs(jeta) < e1)
            hists[k] = A._hist(je[s], jpt[s], E[k], n)
    return bins, sig, nacc, win, pthat, hists


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/data/OO-RHIC_phys")
    ap.add_argument("-j", type=int, default=8)
    ap.add_argument("--phase3", action="store_true", help="phase-3 jobs only (no 3e windows)")
    a = ap.parse_args(argv)
    B.OUT = Path(a.out)
    E = edges()
    comps, _ = J.components("pp_ref")
    files = []
    if a.phase3:
        hij = {f"pp_ref_{B._w(lo, hi)}" for lo, hi in getattr(B, "HIJET_WINDOWS", {})}
        comps = [(nm, w) for nm, w in comps if nm not in hij]
    for nm, _ in comps:
        fs = J.finished_files(nm)
        if a.phase3:
            seeds = {s for _, s, _, _ in B.jobs(nm)}
            fs = [f for f in fs if int(f.rsplit("_seed", 1)[1].split("_")[0]) in seeds]
        files += fs
    with ProcessPoolExecutor(max_workers=a.j) as ex:
        res = list(ex.map(one_file, [(f, E) for f in files]))
    wl = sorted({w for bins, *_ in res for w in bins})
    sig_n, nacc = np.zeros(len(wl)), np.zeros(len(wl))
    win, pth, H = [], [], {k: [] for k in DEFS}
    for bins, sig, na, w, p, hs in res:
        g = np.array([wl.index(x) for x in bins])
        sig_n[g] += sig * na
        nacc[g] += na
        win.append(g[w]); pth.append(p)
        for k in DEFS:
            H[k].append(hs[k])
    win, pth = np.concatenate(win), np.concatenate(pth)
    w, *_ = A.stitch_pieces(wl, win, pth, sig_n / np.maximum(nacc, 1))
    sfx = "_phase3" if a.phase3 else ""
    out = {"sample": "pp_ref (phase 3)" if a.phase3 else "pp_ref (all jobs)", "events": int(len(win)), "files": len(files),
           "units": "d2sigma/dpT deta [pb/GeV]", "definitions": {k: {"R": d[0], "abs_eta": [d[1], d[2]]} for k, d in DEFS.items()}}
    for k, (R, e0, e1) in DEFS.items():
        h = np.concatenate(H[k])
        S, S2 = (w[:, None] * h).sum(0), ((w[:, None] * h) ** 2).sum(0)
        conv = 1e9 / np.diff(E[k]) / (2 * (e1 - e0))
        out[k] = {"edges": E[k].tolist(), "value": (S * conv).tolist(), "err": (np.sqrt(S2) * conv).tolist(),
                  "raw_jets": h.sum(0).astype(int).tolist()}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"pp_jet_spectra{sfx}.json").write_text(json.dumps(out, indent=1))
    print(f"pp_ref: {len(files)} files, {len(win)} events -> results/pp_jet_spectra{sfx}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
