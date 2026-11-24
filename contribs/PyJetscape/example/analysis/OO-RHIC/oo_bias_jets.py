#!/usr/bin/env python3
"""
analysis/OO-RHIC/oo_bias_jets.py -- the selection-only centrality bias of O+O jets as a function of
the jet pT and of the charged-hadron pT (R4b, the full test, continued): the events of oo_bias.py
(`mb`, `jets`; jet in vacuum, --vacuum-jet; `jets_quench`, jet in the medium), classes as oo_bias.py
builds them, the yield per class from reconstructed jets and from charged hadrons instead of the hard
scattering itself.

    python oo_bias_jets.py                 # -> results/oo_bias_jets.json
    python oo_bias_jets.py --out DIR       # another output root of oo_bias.py (DIR/jets, DIR/mb)
    python oo_bias_jets.py --sample jets_quench   # the jet in the medium -> results/oo_bias_jets_jets_quench.json
    python oo_bias_jets.py --min-entries 20       # the stitching threshold (default 10, see below)

Jets: anti-kT R = 0.4 (FastJet) on the hard event's hadrons (the jet_frag files: shower and
iColored remnant hadrons, no bulk; the jet is in vacuum, so there is nothing to subtract),
|eta_jet| < 0.7 as sPHENIX's, constituents pT > 0.2 GeV.  Which particles depends on the files:
samples stored with --charged (mb, jets) give charged-particle jets, whose pT is roughly 0.6-0.7 of
a full jet's (sPHENIX's 19-60 GeV correspond to about 12-40 GeV); samples with all hadrons
(jets_quench) give full jets, all final-state particles except neutrinos, as sPHENIX's calorimeter
jets.  Hadrons: charged hadrons of the hard event, |eta| < 1 (STAR's acceptance), in the same files.

Per jet-pT bin each jet counts with its event's weight N_coll x sigma_k / N_k (k = the pTHat
window; Pythia's cross section per window over the window's events): R_AA^vac per class =
(N_coll-weighted jet fraction in the class) / (N_coll-weighted minimum-bias fraction), R_CP from
the same sums (hstat.raa_vac, hstat.rcp; errors from both samples, jets of one event summed
before squaring).  The windows 9-11, 18-23, 30-40, 40-55 GeV leave gaps (11-18, 23-30 GeV), so the
"all windows" spectrum is not inclusive in pTHat; per window the same quantities are given too.

Stitching the windows (as the experiments do with pTHat-binned simulations): a window contributes to
a jet- or hadron-pT bin only if its histogram has at least --min-entries (default 10) raw entries in
that bin in every class; otherwise it is left out of that bin in all classes (the class fractions and
R_CP then use the same windows).  The few high-weight entries a low window's tail puts into a high bin
would otherwise decide the value and its error.  Per window, a bin with fewer entries in any class is
nan.  The class sums (W, sum w^2 of the jet sample) are stored per bin, so ratios of two samples
(quenched / vacuum) can be formed with the jet samples' errors only (the minimum-bias sums cancel).
Classes: sPHENIX's MBD (3.51 < |eta| < 4.61, both arms) in 0-20/20-40/40-60/60-80/80-100 %, the EPD
(2.1 < |eta| < 5.1) in the same classes and in STAR's, and as the truth reference the impact parameter
in sPHENIX's classes (percentiles of the minimum-bias b; no selection there: the vacuum R_AA^vac is 1).
Per class also the mean N_coll of minimum-bias and of (hard-scattering weighted) jet events.  The jet events' class multiplicity counts
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
HAD_ETA = 1.0
HAD_EDGES = np.array([2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30.0])
MIN_ENTRIES = 10
NCOLL_EDGES = np.array([1, 2, 3, 5, 9, 16, 26, 60])
HAD_WIDE = [(6, 30), (8, 30)]                         # hadron-pT ranges summed per event (one error per event)
JET_R = 0.4
JET_ETA = 0.7
CONST_PTMIN = 0.2
PT_EDGES = np.array([5, 8, 11, 14, 18, 23, 30, 40, 55.0])
RCP_PAIRS = {"sph": [(0, 3), (0, 2), (1, 3)], "star": [(0, 4), (0, 3), (1, 4)]}
ESTIMATORS = (("mbd", "mbd", B.SPH_CLASSES), ("epd_sph", "epd", B.SPH_CLASSES), ("epd", "epd", B.STAR_CLASSES),
              ("b", "b", B.SPH_CLASSES))     # true geometry: percentiles of the minimum-bias impact parameter


def file_jets(path):
    """Per event of one jet_frag file: the number of jets and of charged hadrons in each pT bin, the
    pTHat window, the file's per-window cross sections and accepted events, and the colour audit's
    hole and partner energies (zero without a medium)."""
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
        holes = np.stack([h[f"units/colour_{k}_energy"][:n] if f"units/colour_{k}_energy" in h else np.zeros(n)
                          for k in ("hole", "partner")], axis=1)
    E, px, py, pz = p.T
    pt = np.hypot(px, py)
    pa = np.sqrt(px * px + py * py + pz * pz)
    eta = 0.5 * np.log(np.clip(pa + pz, 1e-12, None) / np.clip(pa - pz, 1e-12, None))
    evall = np.repeat(np.arange(n), counts)
    hb = np.digitize(pt, HAD_EDGES) - 1
    hk = (np.abs(eta) < HAD_ETA) & (hb >= 0) & (hb < len(HAD_EDGES) - 1) & np.isin(np.abs(pid), B.charged_ids())
    had = np.zeros((n, len(HAD_EDGES) - 1))
    np.add.at(had, (evall[hk], hb[hk]), 1.0)
    keep = (pt > CONST_PTMIN) & (np.abs(eta) < JET_ETA + JET_R + 0.1) & ~np.isin(np.abs(pid), INVISIBLE)
    ev = evall[keep]
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
    return out, win, sig, nacc, charged_only, had, holes


def classes(x_mb, x_jet, cls_def, rng, jitter=True):
    """Class index per event from percentiles of x_mb (larger x = more central); integer counts get a
    uniform jitter so that ties split at the percentile, continuous x (-b) none."""
    nc = len(cls_def)
    xm = x_mb + rng.random(len(x_mb)) if jitter else np.asarray(x_mb, float)
    cuts = np.percentile(xm, [100 - hi for _, hi in cls_def])

    def cls(v):
        c = np.full(len(v), nc - 1)
        for k in range(nc - 1, -1, -1):
            c[v >= cuts[k]] = k
        return c
    return cls(xm), cls(x_jet + rng.random(len(x_jet)) if jitter else np.asarray(x_jet, float))


def entries(cj, win, counts_ev, nc, nwin):
    """Raw entries per (window, class, pT bin)."""
    out = np.zeros((nwin, nc, counts_ev.shape[1]))
    np.add.at(out, (win, cj), counts_ev)
    return out


def per_bin(cm, ncoll_mb, cj, w_ev, counts_ev, cls_def, pairs, win, nwin, n_min):
    """R_AA^vac and R_CP per pT bin for the jets (hadrons) counts_ev of events with weights w_ev,
    stitched over the windows in win with the min-entries rule (module docstring)."""
    nc = len(cls_def)
    n_k = entries(cj, win, counts_ev, nc, nwin)                    # (window, class, bin)
    use = n_k.min(axis=1) >= n_min                                 # (window, bin)
    present = np.bincount(win, minlength=nwin) > 0
    out = {"raa_vac": [], "raa_err": [], "rcp": {}, "n": [], "n_used": [], "windows_used": [], "Wj": [], "W2j": []}
    labs = [f"{a}-{b}" for a, b in cls_def]
    for b in range(counts_ev.shape[1]):
        wj = w_ev * counts_ev[:, b] * use[win, b]
        rv = H.raa_vac(cm, ncoll_mb, cj, wj, nc)
        ok = bool(use[present, b].any())
        nanl = [float("nan")] * nc
        out["raa_vac"].append(rv["raa"].tolist() if ok else nanl)
        out["raa_err"].append(rv["err"].tolist() if ok else nanl)
        out["n"].append(int(counts_ev[:, b].sum()))
        out["n_used"].append(int((counts_ev[:, b] * use[win, b]).sum()))
        out["windows_used"].append([int(k) for k in np.flatnonzero(use[:, b] & present)])
        out["Wj"].append(rv["Wj"].tolist())
        out["W2j"].append(rv["W2j"].tolist())
        for a, c in pairs:
            out["rcp"].setdefault(f"{labs[a]}/{labs[c]}", []).append(list(H.rcp(rv, a, c)) if ok else [float("nan")] * 2)
    out["n_jets"] = out["n"]                                       # the name before hadrons were added
    return out


def estimator_block(cm, ncoll_mb, cj, w_all, ncoll_jet, win, counts_ev, cls_def, pairs, n_min):
    nwin = len(B.JET_WINDOWS)
    Wm, W2m = H.class_sums(cm, len(cls_def), ncoll_mb)
    nm, nme = H.class_mean(cm, ncoll_mb, len(cls_def))
    nj, nje = H.class_mean(cj, ncoll_jet, len(cls_def), w=w_all)      # the jets' geometry (hard-scattering weighted)
    e = {"classes": cls_def, "Wm": Wm.tolist(), "W2m": W2m.tolist(),
         "ncoll_mb": nm.tolist(), "ncoll_mb_err": nme.tolist(), "ncoll_jet": nj.tolist(), "ncoll_jet_err": nje.tolist(),
         "all": per_bin(cm, ncoll_mb, cj, w_all, counts_ev, cls_def, pairs, win, nwin, n_min),
         "per_window": {}}
    for k, (lo, hi) in enumerate(B.JET_WINDOWS):
        m = win == k
        e["per_window"][f"{lo}-{hi}"] = per_bin(cm, ncoll_mb, cj[m], ncoll_jet[m].astype(float), counts_ev[m], cls_def,
                                                pairs, np.zeros(int(m.sum()), np.int64), 1, n_min)
    return e


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="output root of oo_bias.py (default its OUT)")
    ap.add_argument("-j", type=int, default=8)
    ap.add_argument("--sample", default="jets", help="jet sample of oo_bias.py (jets, jets_quench, ...)")
    ap.add_argument("--min-entries", type=int, default=MIN_ENTRIES,
                    help=f"raw entries a window needs in a bin (every class) to enter it (default {MIN_ENTRIES})")
    a = ap.parse_args(argv)
    if a.out:
        B.OUT = Path(a.out)
    files = [f for f in sorted(glob.glob(str(B.OUT / a.sample / "*_hadrons_jet_frag.h5")))
             if os.path.exists(f.replace("_hadrons_jet_frag.h5", ".json"))]   # finished jobs, as oo_bias.load
    if not files:
        sys.exit(f"oo_bias_jets.py: no jet files under {B.OUT / a.sample}")
    mb, jet = B.load("mb"), B.load(a.sample)       # the estimator counts, same file order as below
    with ProcessPoolExecutor(max_workers=a.j) as ex:
        res = list(ex.map(file_jets, files))
    jets_ev = np.concatenate([r[0] for r in res])
    had_ev = np.concatenate([r[5] for r in res])
    had_wide = np.stack([had_ev[:, (HAD_EDGES[:-1] >= lo) & (HAD_EDGES[1:] <= hi)].sum(1) for lo, hi in HAD_WIDE], axis=1)
    holes = np.concatenate([r[6] for r in res])
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
           "min_entries": a.min_entries,
           "n_events": {"mb": int(len(mb["glauber_ncoll"])), "jets": int(len(win))},
           "events_per_window": nev.astype(int).tolist(), "estimators": {},
           "hadrons": {"eta_max": HAD_ETA, "pt_edges": HAD_EDGES.tolist(), "particles": "charged", "estimators": {},
                       "wide": {"ranges": HAD_WIDE, "estimators": {}}}}
    ncoll = jet["glauber_ncoll"]
    for key, est, cls_def in ESTIMATORS:
        if est == "b":
            cm, cj = classes(-mb["glauber_b"], -jet["glauber_b"], cls_def, rng, jitter=False)
        else:
            cm, cj = classes(mb[f"n_{est}"], jet[f"n_{est}"], cls_def, rng)
        pairs = RCP_PAIRS["sph" if cls_def is B.SPH_CLASSES else "star"]
        out["estimators"][key] = estimator_block(cm, mb["glauber_ncoll"], cj, w_all, ncoll, win, jets_ev, cls_def,
                                                 pairs, a.min_entries)
        out["hadrons"]["estimators"][key] = estimator_block(cm, mb["glauber_ncoll"], cj, w_all, ncoll, win, had_ev,
                                                            cls_def, pairs, a.min_entries)
        out["hadrons"]["wide"]["estimators"][key] = estimator_block(cm, mb["glauber_ncoll"], cj, w_all, ncoll, win, had_wide,
                                                                    cls_def, pairs, a.min_entries)
        if key == "mbd":
            cj_mbd = cj
    if holes.any():     # the medium's holes (dropped by iColored) and the partners put in for them, per event
        hv, he = H.profile(ncoll, holes[:, 0], NCOLL_EDGES)
        pv, pe_ = H.profile(ncoll, holes[:, 1], NCOLL_EDGES)
        hc, hce = H.class_mean(cj_mbd, holes[:, 0], len(B.SPH_CLASSES))
        pc_, pce = H.class_mean(cj_mbd, holes[:, 1], len(B.SPH_CLASSES))
        out["holes"] = {"ncoll_edges": NCOLL_EDGES.tolist(), "hole_E": hv.tolist(), "hole_E_err": he.tolist(),
                        "partner_E": pv.tolist(), "partner_E_err": pe_.tolist(),
                        "mbd_classes": B.SPH_CLASSES, "hole_E_class": hc.tolist(), "hole_E_class_err": hce.tolist(),
                        "partner_E_class": pc_.tolist(), "partner_E_class_err": pce.tolist()}
    default = HERE / "results" / ("oo_bias_jets.json" if a.sample == "jets" else f"oo_bias_jets_{a.sample}.json")
    path = Path(os.environ.get("OO_BIAS_JETS_JSON", default))
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    lab = [f"{lo:g}-{hi:g}" for lo, hi in zip(PT_EDGES[:-1], PT_EDGES[1:])]
    hlab = [f"{lo:g}-{hi:g}" for lo, hi in zip(HAD_EDGES[:-1], HAD_EDGES[1:])]
    print(f"minimum bias {out['n_events']['mb']}, jet events {out['n_events']['jets']} "
          f"(per window {out['events_per_window']}); {'charged' if charged_only else 'full'} jets R = {JET_R}, |eta| < {JET_ETA}; "
          f"windows stitched with >= {a.min_entries} entries per bin and class")
    for what, blk, labs_ in (("jet", out["estimators"], lab), ("charged-hadron", out["hadrons"]["estimators"], hlab)):
        for key in ("mbd", "epd_sph"):
            r = blk[key]["all"]
            print(f"{key}: R_CP(0-20/60-80) vs {what} pT [GeV] (all windows, sigma-weighted)")
            for l, (v, er), nj, nu, wu in zip(labs_, r["rcp"]["0-20/60-80"], r["n"], r["n_used"], r["windows_used"]):
                print(f"   {l:>6s}  {v:6.3f} ± {er:5.3f}   ({nj} entries, {nu} used; windows {wu})")
    if "holes" in out:
        print("holes dropped by iColored / partners added, mean energy per event [GeV] by N_coll: "
              + "  ".join(f"{lo}-{hi}: {h:.2f}/{q:.2f}" for lo, hi, h, q in
                          zip(NCOLL_EDGES[:-1], NCOLL_EDGES[1:], out["holes"]["hole_E"], out["holes"]["partner_E"])))
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
