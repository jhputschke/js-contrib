#!/usr/bin/env python3
"""
analysis/OO-RHIC/css_gap_diag.py -- the run C vs run 1 gap in Colorless's h+- R_AA without the close-pair shift
(colorless_shift_study.ipynb: |eta| < 1, 10-25 GeV, 0.78 +- 0.05 vs 0.94 +- 0.04), per pT-hat window, from stored files.

(0) stitching: per window k and h+- bin (8-25 GeV), the events N_k, the events contributing to the bin (non-zero
    row), and the weighted share of the bin's yield (w = sigma_k / N_k, auau_raa.py's collect), Au+Au and p+p;
    R_AA with a minimum-entries rule (a window enters a bin only with >= MIN entries in both Au+Au and p+p).
(2) parton level per window (the produced files): leading final jet parton pT (pstat 0), its ratio to the
    hard leading parton, E of the jet partons, recoils (pstat 1) and holes (pstat -1) per event, their energy.
(3) per window: R_AA_k = AA_k / pp_k, and window k's contribution to the gap, (AA_k(run 1) - AA_k(run C)) / pp
    (same p+p reference, so the contributions add up to the gap), plus R_AA leaving one window out; run 1 split
    into (a) and (b) for the low windows (without the shift).

    python css_gap_diag.py -j 16      # -> results/css_gap_diag.json
Model results, not vetted.
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
import auau_raa as RA  # noqa: E402
import oo_analysis_rhic as A  # noqa: E402

ROOT = Path(os.environ.get("CSS_ROOT", "/home/putschke/prod"))   # the GB10 layout; $CSS_ROOT elsewhere
RH = ROOT / "AuAu-RHIC_runs/rehad"
G = "*_hadrons_jet_frag.h5"
SAMPLES = {   # tag: (Au+Au glob, p+p glob, produced files for the parton level)
    "noshift": {"runC": f"{RH}/runC/cl_noshift/*/{G}", "run1": f"{RH}/run1/cl_noshift/*/{G}",
                "pp": f"{RH}/PP19/cl_noshift/*/{G}"},
    "shift": {"runC": f"{RH}/runC/cl_shift/*/{G}", "run1": f"{RH}/run1/cl_shift/*/{G}",
              "pp": f"{RH}/PP19/cl_shift/*/{G}"},
}
PRODUCED = {"runC": f"{ROOT}/AuAu-RHIC_jschain/js_chain_t3/{G}",
            "run1": f"{ROOT}/AuAu-RHIC_runs/run1/*/{G}"}
EDGES = RA.HAD_EDGES
BINS = [(8, 10), (10, 12), (12, 15), (15, 20), (20, 25)]
RANGES = [(10, 25), (10, 12), (8, 10)]
MIN_ENTRIES = 10


def collect(pattern, j):
    fs = RA.files(None, None, pattern)
    with ProcessPoolExecutor(max_workers=j) as ex:
        res = list(ex.map(RA.one_file, fs))
    wl = sorted({w for r in res for w in r[0]})
    sig_n, nacc = np.zeros(len(wl)), np.zeros(len(wl))
    win, pth, keep, rows = [], [], [], []
    for bins, sig, na, w, p, kp, o, _bk, _gr in res:
        g = np.array([wl.index(x) for x in bins])
        sig_n[g] += sig * na; nacc[g] += na
        win.append(g[w]); pth.append(p); keep.append(kp); rows.append(o["had"])
    win, pth, keep = np.concatenate(win), np.concatenate(pth), np.concatenate(keep)
    w, *_ = A.stitch_pieces(wl, win, pth, sig_n / np.maximum(nacc, 1))
    return {"wl": wl, "win": win, "w": w * keep, "rows": np.concatenate(rows), "files": len(fs)}


def per_window(d, wl_all):
    """yield, error^2, events, entries per (global window, bin)."""
    nw, nb = len(wl_all), len(EDGES) - 1
    Y, E2, N, NE = np.zeros((nw, nb)), np.zeros((nw, nb)), np.zeros(nw), np.zeros((nw, nb))
    g = np.array([wl_all.index(x) for x in d["wl"]])[d["win"]]
    x = d["w"][:, None] * d["rows"]
    np.add.at(Y, g, x); np.add.at(E2, g, x ** 2); np.add.at(N, g, 1.0); np.add.at(NE, g, (d["rows"] != 0).astype(float))
    return Y, E2, N, NE


def sel(lo, hi):
    return (EDGES[:-1] >= lo - 1e-9) & (EDGES[1:] <= hi + 1e-9)


def ratio(a, ea2, p, ep2):
    r = a / p if p > 0 else np.nan
    return r, abs(r) * np.sqrt(ea2 / a ** 2 + ep2 / p ** 2) if a != 0 and p > 0 else np.nan


def parton_one(path):
    import h5py
    import hdf5plugin  # noqa: F401
    with h5py.File(path) as h:
        n = int(h.attrs["nunits_written"])
        bins = [tuple(map(float, w)) for w in np.asarray(h.attrs["pthat_bins"]).reshape(-1, 2)]
        u = h["units"]
        win, lead0 = u["pthat_bin"][:n].astype(int), u["parton_pt_lead"][:n].astype(float)
        po = h["partons/offsets"][: n + 1]
        par = h["partons/data"][: po[-1]]
    ev = np.repeat(np.arange(n), np.diff(po))
    st = par[:, 2].astype(int)
    pt = np.hypot(par[:, 4], par[:, 5])
    out = np.zeros((n, 8))
    jet = st == 0
    lead = np.zeros(n); np.maximum.at(lead, ev[jet], pt[jet])
    out[:, 0] = lead
    out[:, 1] = np.divide(lead, lead0, out=np.zeros(n), where=lead0 > 0)
    np.add.at(out[:, 2], ev[jet], par[jet, 3])
    for c, s in ((3, 1), (5, -1)):
        m = st == s
        np.add.at(out[:, c], ev[m], 1.0); np.add.at(out[:, c + 1], ev[m], par[m, 3])
    out[:, 7] = lead0
    return [bins[k] for k in win], out


PCOLS = ["lead_pt_final", "lead_final_over_hard", "E_jet_partons", "n_recoils", "E_recoils", "n_holes", "E_holes",
         "lead_pt_hard"]


def partons(pattern, j):
    fs = sorted(glob.glob(pattern))
    with ProcessPoolExecutor(max_workers=j) as ex:
        res = list(ex.map(parton_one, fs))
    wins = [w for r in res for w in r[0]]
    X = np.concatenate([r[1] for r in res])
    out = {}
    for w in sorted(set(wins)):
        m = np.array([x == w for x in wins])
        out[f"{w[0]:g}-{w[1]:g}"] = {"events": int(m.sum()),
                                     **{c: [float(X[m, i].mean()), float(X[m, i].std() / np.sqrt(m.sum()))]
                                        for i, c in enumerate(PCOLS)}}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-j", type=int, default=16)
    a = ap.parse_args(argv)
    res = {"min_entries": MIN_ENTRIES, "bins": BINS, "ranges": RANGES}
    for arm, smp in SAMPLES.items():
        D = {k: collect(g, a.j) for k, g in smp.items()}
        wl_all = sorted({w for d in D.values() for w in d["wl"]})
        P = {k: per_window(d, wl_all) for k, d in D.items()}
        Yp, E2p, Np, NEp = P["pp"]
        o = {"windows": [f"{lo:g}-{hi:g}" for lo, hi in wl_all], "files": {k: d["files"] for k, d in D.items()},
             "events": {k: P[k][2].astype(int).tolist() for k in P}}
        # (0) entries and weighted shares per window and bin
        for k in P:
            Y, E2, N, NE = P[k]
            o[f"entries_{k}"] = {f"{lo:g}-{hi:g}": NE[:, sel(lo, hi)].sum(1).astype(int).tolist() for lo, hi in BINS}
            o[f"share_{k}"] = {f"{lo:g}-{hi:g}": (Y[:, sel(lo, hi)].sum(1) / Y[:, sel(lo, hi)].sum()).tolist()
                               for lo, hi in BINS}
        # R_AA, all windows / with the minimum-entries rule / leaving one window out; (3) per window, gap shares
        for k in ("runC", "run1"):
            Y, E2, N, NE = P[k]
            for lo, hi in RANGES + BINS:
                s = sel(lo, hi); key = f"{lo:g}-{hi:g}"
                r = ratio(Y[:, s].sum(), E2[:, s].sum(), Yp[:, s].sum(), E2p[:, s].sum())
                ok = ((NE[:, s] >= MIN_ENTRIES) & (NEp[:, s] >= MIN_ENTRIES))          # per window and bin
                rm = ratio((Y[:, s] * ok).sum(), (E2[:, s] * ok).sum(), (Yp[:, s] * ok).sum(), (E2p[:, s] * ok).sum())
                drop = 1 - (Y[:, s] * ok).sum() / Y[:, s].sum()
                loo = [ratio(np.delete(Y[:, s], i, 0).sum(), np.delete(E2[:, s], i, 0).sum(),
                             np.delete(Yp[:, s], i, 0).sum(), np.delete(E2p[:, s], i, 0).sum()) for i in range(len(wl_all))]
                pw = [ratio(Y[i, s].sum(), E2[i, s].sum(), Yp[i, s].sum(), E2p[i, s].sum()) for i in range(len(wl_all))]
                o.setdefault("raa", {}).setdefault(k, {})[key] = {
                    "all": r, "min_entries": rm, "dropped_share_auau": drop, "leave_one_out": loo, "per_window": pw}
        for lo, hi in RANGES + BINS:
            s = sel(lo, hi); key = f"{lo:g}-{hi:g}"; pp = Yp[:, s].sum()
            d = (P["run1"][0][:, s].sum(1) - P["runC"][0][:, s].sum(1)) / pp
            e = np.sqrt(P["run1"][1][:, s].sum(1) + P["runC"][1][:, s].sum(1)) / pp
            o.setdefault("gap", {})[key] = {"total": float(d.sum()), "per_window": d.tolist(), "per_window_err": e.tolist()}
        res[arm] = o
        print(f"[{arm}] windows {o['windows']}")
        for k in ("runC", "run1"):
            x = o["raa"][k]["10-25"]
            print(f"  {k}: h 10-25 all {x['all'][0]:.3f}+-{x['all'][1]:.3f}  min-entries {x['min_entries'][0]:.3f}"
                  f"+-{x['min_entries'][1]:.3f} (dropped {100 * x['dropped_share_auau']:.1f}% of the Au+Au yield)")
        print("  gap h 10-25 per window: " + " ".join(f"{w} {v:+.3f}+-{e:.3f}" for w, v, e in
                                                       zip(o["windows"], o["gap"]["10-25"]["per_window"],
                                                           o["gap"]["10-25"]["per_window_err"])))
    # run 1 split into (a) and (b) without the shift: the low windows' rate of 10-12 GeV hadrons, R_AA(10-25)
    sp = {"runC": SAMPLES["noshift"]["runC"], "run1a": f"{RH}/run1/cl_noshift/a/{G}",
          "run1b": f"{RH}/run1/cl_noshift/b/{G}", "pp": SAMPLES["noshift"]["pp"]}
    D = {k: collect(g, a.j) for k, g in sp.items()}
    wl_all = sorted({w for d in D.values() for w in d["wl"]})
    P = {k: per_window(d, wl_all) for k, d in D.items()}
    s, s12 = sel(10, 25), sel(10, 12)
    res["split_noshift"] = {"windows": [f"{lo:g}-{hi:g}" for lo, hi in wl_all], **{
        k: {"events": P[k][2].astype(int).tolist(), "entries_10_12": P[k][3][:, s12].sum(1).astype(int).tolist(),
            "raa_10_25": [ratio(P[k][0][i, s].sum(), P[k][1][i, s].sum(), P["pp"][0][i, s].sum(), P["pp"][1][i, s].sum())
                          if P[k][2][i] else [np.nan, np.nan] for i in range(len(wl_all))]} for k in sp}}
    res["partons"] = {k: partons(g, a.j) for k, g in PRODUCED.items()}
    fn = HERE / "results" / "css_gap_diag.json"
    fn.write_text(json.dumps(res, indent=1, default=float))
    print("->", fn)


if __name__ == "__main__":
    main()
