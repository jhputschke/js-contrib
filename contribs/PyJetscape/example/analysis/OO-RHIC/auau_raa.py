#!/usr/bin/env python3
"""
analysis/OO-RHIC/auau_raa.py -- the Au+Au energy-loss calibration (B1 / B2, tuning page "A3 and the
proposed runs"): R_AA of the O+O chain in central Au+Au against STAR and PHENIX.

    python auau_raa.py --family auau_med -j 12      # B1 -> results/auau_raa_auau_med.json
    python auau_raa.py --family auau_t3 -j 12       # B2 -> results/auau_raa_auau_t3.json

Samples: <family>_w* in /data/AuAu-RHIC_phys (0-10 % by b < 4.7 fm, hard partons |y| < 1 "any"),
the p+p reference pp_ref_w* in /data/OO-RHIC_phys (phase 3 + 3e; the same |y| < 1 "any" cut applied
here through the stored hardest partons' rapidities, parton_y_lead / parton_y_sub).  Runs on partial
output (the finished jobs).  One hard scattering per Au+Au event, so
R_AA(pT) = sum_AA w dN/dpT / sum_pp w dN/dpT with w = sigma_window / N_window (oo_analysis_rhic's
stitch_pieces); errors from the summed squared weights.

Observables (hard event only, i.e. the jet_frag hadrons; no bulk, so no background subtraction):
- pions (pi+ + pi-)/2, |eta| < 0.35, PHENIX's pi0 bins (the pi0 are decayed in these files);
- charged hadrons, |eta| < 1;
- STAR's acceptances: pi+- |y| < 0.5 (pion_star, 1110.0579's bins) and (h+ + h-)/2 |eta| < 0.5 (had_star);
- charged jets anti-kT R = 0.2, 0.3, 0.4, constituents pT > 0.2 GeV, |eta_jet| < 1 - R, holes +
  partner copies subtracted with the charged fraction 2/3 (as oo_analysis_rhic.py), STAR's bins; with
  and without STAR's leading-charged-track cut pT,lead > 5 GeV (a charged hadron above 5 GeV within
  dR < R of the jet axis). STAR's R_AA^Pythia puts the cut on the Au+Au jets only (its PYTHIA reference
  has none): key chjet_R*_star = Au+Au with the cut / p+p without; chjet_R*_lead5 has the cut in both;
- full jets anti-kT R = 0.4, |eta_jet| < 0.5 (2204.01163 Fig. 15's prediction), holes + partner copies
  subtracted with the full four-momentum.
Every jet spectrum also without the subtraction (key suffix _raw), to show its size.  The hadron
observables are not hole-subtracted (negative_hadrons.py); the holes and partners are soft, so this is
taken as safe from ~6 GeV on.
Also the bulk (iSS) at mid-rapidity, 0-10 %: charged dN/deta |eta| < 0.5 and <pT> of pi, K, p.
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
import oo_analysis_rhic as A  # noqa: E402
import oo_bias as B  # noqa: E402

AUAU_ROOT, PP_ROOT = Path("/data/AuAu-RHIC_phys"), Path("/data/OO-RHIC_phys")
PI_EDGES = np.array([5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 12, 14, 16, 18, 20.0])
HAD_EDGES = np.array([5, 6, 7, 8, 10, 12, 15, 20, 25.0])
STAR_PI_EDGES = np.array([5, 5.5, 6, 6.5, 7, 8, 10, 12, 15.0])      # STAR pi+- R_AA (1110.0579) bins from 5 GeV
STAR_H_EDGES = np.array([5.1, 6, 7, 8, 9, 10, 12, 15.0])            # STAR h+- R_AA (nucl-ex/0305015) bins to 10 GeV, extended
CHJ_EDGES = np.array([8, 10, 12, 14, 16, 20, 25, 30.0])
FJ_EDGES = np.array([10, 15, 20, 25, 30, 35, 40, 50.0])
YMAX = 1.0
LEAD = 5.0


def files(root, prefix):
    out = []
    for d in sorted(glob.glob(str(root / f"{prefix}_w*"))):
        out += [f for f in sorted(glob.glob(f"{d}/*_hadrons_jet_frag.h5"))
                if os.path.exists(f.replace("_hadrons_jet_frag.h5", ".json"))]
    return out


def lead_ok(je, jeta, jphi, ev_c, pt_c, eta_c, phi_c, R):
    ok = np.zeros(len(je), bool)
    hi = pt_c > LEAD
    if not hi.any():
        return ok
    evh, etah, phih = ev_c[hi], eta_c[hi], phi_c[hi]
    for i in range(len(je)):
        q = evh == je[i]
        if q.any():
            ok[i] = (np.hypot(etah[q] - jeta[i], A._dphi(phih[q], jphi[i])) < R).any()
    return ok


def one_file(path):
    import h5py
    import hdf5plugin  # noqa: F401
    bf = path.replace("_jet_frag.h5", "_bulk_bg.h5")
    with h5py.File(path) as h:
        n = int(h.attrs["nunits_written"])
        sig = np.asarray(h.attrs["pthat_bin_sigma_gen"], float)
        nacc = np.asarray(h.attrs["pthat_bin_n_accepted"], float)
        bins = [tuple(map(float, w)) for w in np.asarray(h.attrs["pthat_bins"]).reshape(-1, 2)]
        win = h["units/pthat_bin"][:n].astype(int)
        pthat = h["units/pthat"][:n].astype(float)
        yl, ys = h["units/parton_y_lead"][:n], h["units/parton_y_sub"][:n]
        ycut = "parton_ymax" in h.attrs
        pa, ca, pid = B._units(h, n, charged=False, with_pid=True)
        neg = A.negative_partons(h, n)
    # the |y| < 1 "any" cut: already applied in the Au+Au jobs; applied here to p+p
    keep = np.ones(n, bool) if ycut else (np.abs(yl) < YMAX) | (np.abs(ys) < YMAX)
    ev = np.repeat(np.arange(n), ca)
    pt, eta, phi = A._kin(pa)
    ch = np.isin(np.abs(pid), B.charged_ids())
    out = {}
    m = (np.abs(pid) == 211) & (np.abs(eta) < 0.35)
    out["pion"] = A._hist(ev[m], pt[m], PI_EDGES, n) / 2.0
    m = ch & (np.abs(eta) < 1.0)
    out["had"] = A._hist(ev[m], pt[m], HAD_EDGES, n)
    # STAR's acceptances: pi+- |y| < 0.5 (1110.0579), (h+ + h-)/2 |eta| < 0.5 (nucl-ex/0305015)
    E_, pz_ = pa[:, 0], pa[:, 3]
    y = 0.5 * np.log(np.clip(E_ + pz_, 1e-12, None) / np.clip(E_ - pz_, 1e-12, None))
    m = (np.abs(pid) == 211) & (np.abs(y) < 0.5)
    out["pion_star"] = A._hist(ev[m], pt[m], STAR_PI_EDGES, n) / 2.0
    m = ch & (np.abs(eta) < 0.5)
    out["had_star"] = A._hist(ev[m], pt[m], STAR_H_EDGES, n) / 2.0
    mc = ch & (pt > A.CONST_PTMIN) & (np.abs(eta) < 1.0)
    for R in (0.2, 0.3, 0.4):
        je, jpt, jeta, jphi, jx, jy = A._jets(pa[mc], ev[mc], n, R, None, ptmin=CHJ_EDGES[0] * 0.5, full=True)
        jsub = A.subtract_negatives(je, jeta, jphi, jx, jy, neg, R, frac=A.CHARGED_FRAC)
        k = np.abs(jeta) < 1.0 - R
        out[f"chjet_R{R}"] = A._hist(je[k], jsub[k], CHJ_EDGES, n)
        out[f"chjet_R{R}_raw"] = A._hist(je[k], jpt[k], CHJ_EDGES, n)              # holes not subtracted
        lk = lead_ok(je, jeta, jphi, ev[mc], pt[mc], eta[mc], phi[mc], R)
        out[f"chjet_R{R}_lead5"] = A._hist(je[k & lk], jsub[k & lk], CHJ_EDGES, n)
        out[f"chjet_R{R}_lead5_raw"] = A._hist(je[k & lk], jpt[k & lk], CHJ_EDGES, n)
    vis = ~np.isin(np.abs(pid), A.INVISIBLE) & (pt > A.CONST_PTMIN) & (np.abs(eta) < 1.0)
    je, jpt, jeta, jphi, jx, jy = A._jets(pa[vis], ev[vis], n, 0.4, None, ptmin=FJ_EDGES[0] * 0.5, full=True)
    jsub = A.subtract_negatives(je, jeta, jphi, jx, jy, neg, 0.4)
    k = np.abs(jeta) < 0.5
    out["fjet_R0.4"] = A._hist(je[k], jsub[k], FJ_EDGES, n)
    out["fjet_R0.4_raw"] = A._hist(je[k], jpt[k], FJ_EDGES, n)                     # holes not subtracted
    # bulk at mid-rapidity (Au+Au only): charged dN/deta |eta| < 0.5, <pT> of pi, K, p (|y| < 0.5 ~ |eta|)
    bulk = None
    if ycut and os.path.exists(bf):
        with h5py.File(bf) as hb:
            pb, cb, pidb = B._units(hb, n, charged=False, with_pid=True)
        ptb, etab, _ = A._kin(pb)
        mid = np.abs(etab) < 0.5
        chb = np.isin(np.abs(pidb), B.charged_ids())
        bulk = {"n_ev": n, "nch_mid": float((mid & chb).sum())}
        for nm, code in (("pi", 211), ("K", 321), ("p", 2212)):
            s = mid & (np.abs(pidb) == code)
            bulk[f"{nm}_n"] = float(s.sum()); bulk[f"{nm}_pt"] = float(ptb[s].sum())
    return bins, sig, nacc, win, pthat, keep, out, bulk


def collect(fs, j):
    with ProcessPoolExecutor(max_workers=j) as ex:
        res = list(ex.map(one_file, fs))
    wl = sorted({w for r in res for w in r[0]})
    sig_n, nacc = np.zeros(len(wl)), np.zeros(len(wl))
    win, pth, keep, H, bulk = [], [], [], {}, []
    for bins, sig, na, w, p, kp, o, bk in res:
        g = np.array([wl.index(x) for x in bins])
        sig_n[g] += sig * na; nacc[g] += na
        win.append(g[w]); pth.append(p); keep.append(kp)
        for k, v in o.items():
            H.setdefault(k, []).append(v)
        if bk:
            bulk.append(bk)
    win, pth, keep = np.concatenate(win), np.concatenate(pth), np.concatenate(keep)
    w, *_ = A.stitch_pieces(wl, win, pth, sig_n / np.maximum(nacc, 1))
    w = w * keep
    spec = {k: ((w[:, None] * np.concatenate(v)).sum(0), np.sqrt(((w[:, None] * np.concatenate(v)) ** 2).sum(0)))
            for k, v in H.items()}
    return spec, len(win), int(keep.sum()), wl, bulk


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", default="auau_med")
    ap.add_argument("-j", type=int, default=8)
    a = ap.parse_args(argv)
    fa, fp = files(AUAU_ROOT, a.family), files(PP_ROOT, "pp_ref")
    if not fa:
        sys.exit(f"auau_raa.py: no finished {a.family} jobs in {AUAU_ROOT}")
    SA, na, _, wla, bulk = collect(fa, a.j)
    SP, npp, npk, wlp, _ = collect(fp, a.j)
    edges = {"pion": PI_EDGES, "had": HAD_EDGES, "fjet_R0.4": FJ_EDGES, "pion_star": STAR_PI_EDGES, "had_star": STAR_H_EDGES}
    out = {"family": a.family, "auau_events": na, "auau_files": len(fa), "auau_windows": wla,
           "pp_events": npp, "pp_events_in_cut": npk, "raa": {}}
    for k in SA:
        e = edges.get(k, CHJ_EDGES)
        a_, ea = SA[k]; p_, ep = SP[k]
        r = np.divide(a_, p_, out=np.full_like(a_, np.nan), where=p_ > 0)
        er = r * np.sqrt(np.divide(ea, a_, out=np.zeros_like(a_), where=a_ > 0) ** 2
                         + np.divide(ep, p_, out=np.zeros_like(p_), where=p_ > 0) ** 2)
        out["raa"][k] = {"edges": e.tolist(), "raa": r.tolist(), "err": er.tolist(),
                         "auau": a_.tolist(), "pp": p_.tolist()}
    # STAR's R_AA^Pythia (2006.00582 Sec. X B): the leading-track cut on the Au+Au jets only, "No p_T,lead
    # cut is imposed on this reference jet population" -> Au+Au (lead > 5) / p+p (no lead cut)
    for R in (0.2, 0.3, 0.4):
        for sfx in ("", "_raw"):
            a_, ea = SA[f"chjet_R{R}_lead5{sfx}"]; p_, ep = SP[f"chjet_R{R}{sfx}"]
            r = np.divide(a_, p_, out=np.full_like(a_, np.nan), where=p_ > 0)
            er = r * np.sqrt(np.divide(ea, a_, out=np.zeros_like(a_), where=a_ > 0) ** 2
                             + np.divide(ep, p_, out=np.zeros_like(p_), where=p_ > 0) ** 2)
            out["raa"][f"chjet_R{R}_star{sfx}"] = {"edges": CHJ_EDGES.tolist(), "raa": r.tolist(), "err": er.tolist(),
                                                   "auau": a_.tolist(), "pp": p_.tolist()}
    if bulk:
        nev = sum(b["n_ev"] for b in bulk)
        out["bulk_0_10"] = {"events": nev, "dnch_deta_mid": sum(b["nch_mid"] for b in bulk) / nev,
                            **{f"mean_pt_{nm}": sum(b[f"{nm}_pt"] for b in bulk) / max(sum(b[f"{nm}_n"] for b in bulk), 1)
                               for nm in ("pi", "K", "p")}}
    (HERE / "results").mkdir(exist_ok=True)
    fn = HERE / "results" / f"auau_raa_{a.family}.json"
    fn.write_text(json.dumps(out, indent=1))
    print(f"{a.family}: {na} Au+Au events ({len(fa)} files, windows {wla}); pp_ref {npp} events ({npk} in |y| < {YMAX:g}) -> {fn}")
    for k, v in out["raa"].items():
        print(f"  {k:16s} " + " ".join(f"{x:.2f}" if np.isfinite(x) else "  - " for x in v["raa"]))
    if bulk:
        print("  bulk 0-10 %:", {k: round(v, 3) for k, v in out["bulk_0_10"].items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
