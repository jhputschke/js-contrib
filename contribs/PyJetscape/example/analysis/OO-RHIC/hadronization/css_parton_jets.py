#!/usr/bin/env python3
"""
analysis/OO-RHIC/hadronization/css_parton_jets.py -- the parton-level reference of colorless_shift_study.ipynb: which hadronization
keeps the parton-level jet suppression, and the energy balance of Colorless's hole subtraction.

For JS T3-78k (run 1) and XS-ISR T3 (reuse) (run 1'), and their p+p references (PP19, XS-ISR pp), from the stored
final partons (the produced files' partons/) and the re-hadronized hadrons (rehad/<sample>/<kind>/, sample 0 of the
n_frag 10, i.e. one fragmentation per event):

(1) parton-level jets, anti-kT R = 0.2 and 0.4 (fastjet), |eta_jet| < 1 - R, as quench_diag.py A1:
      "par":  shower + ISR partons (pstat 0, 1000, 22), no medium response;
      "resp": the same plus LBT's recoils (pstat 1), the holes (pstat -1) subtracted inside the cone
              (oo_analysis_rhic.subtract_negatives, full four-momentum) -- the parton-level reference;
    their spectra with auau_raa.py's window weights (sigma_k / N_k, oo_analysis_rhic.stitch_pieces; the p+p
    files' |y| < 1 "any" parton cut as auau_raa.py), the parton-level jet R_AA and n_eff;
    hadron-level jets of the same definition (visible hadrons pT > 0.2 GeV, |eta| < 1; holes subtracted as
    auau_raa.py: Colorless's pstat -1 hadrons, iColored's holes + partner copies), their R_AA, and per parton jet
    ("resp", pT > 8 GeV) the matched hadron jet (the hardest within dR < R): r = pT_had / pT_parton, its weighted
    mean per parton-jet pT bin in Au+Au and p+p, and the double ratio <r>_AA / <r>_pp.
(2) Colorless's energy balance per event (CL+CPS, CL; iColored for comparison): sum E of the positive hadrons, of
    the pstat -1 hadrons, against the partons (pstat >= 0, with the recoils) and the holes; inclusive and inside
    dR < 0.4 of the parton jets ("resp", pT > 10 GeV, |eta| < 0.6), and the charged vs visible pT in that cone.

    python css_parton_jets.py -j 14        # -> results/css_parton_jets.json
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
sys.path.insert(0, str(HERE.parent))
import _paths  # noqa: E402,F401  (analysis/OO-RHIC and its test directories on sys.path)
ANA = _paths.ROOT                            # analysis/OO-RHIC: results/ (ROOT below: the data)
import oo_analysis_rhic as A  # noqa: E402
import oo_bias as B  # noqa: E402

ROOT = Path(os.environ.get("CSS_ROOT", "/home/putschke/prod"))
RH = ROOT / "AuAu-RHIC_runs/rehad"
F3 = ROOT / "AuAu-RHIC_hadtests/from3090"
G = "*_hadrons_jet_frag.h5"
SAMPLES = {   # name: (Au+Au produced glob, Au+Au rehad name, p+p produced glob, p+p rehad name, kinds)
    "JS": (f"{ROOT}/AuAu-RHIC_runs/run1/*/{G}", "run1", f"{F3}/pp19_icdecays*/{G}", "PP19",
           ("cl_shift", "cl_noshift", "icol_beam_reco", "icol_beam_fake")),
    "XS": (f"{ROOT}/AuAu-RHIC_runs/run1p/*/{G}", "run1p", f"{F3}/pp_ref_w*/{G}", "pp_ref",
           ("cl_shift", "cl_noshift", "icol_remn")),
}
RS = (0.2, 0.4)
EDGES = np.array([8, 10, 12, 14, 16, 20, 25, 30, 40, 50.0])
PT_MATCH = 8.0
CONE = 0.4
YMAX = 1.0


def rehad_path(f, name, kind):
    return str(RH / name / kind / os.path.basename(os.path.dirname(f)) / os.path.basename(f))


def _match(je, jpt, jeta, jphi, ev, eta, phi, R):
    """pT of the hardest jet within dR < R of each (event, eta, phi); 0 if none (quench_diag.py's _match; je sorted
    by event, so each event's jets are one slice)."""
    out = np.zeros(len(ev))
    lo, hi = np.searchsorted(je, ev, "left"), np.searchsorted(je, ev, "right")
    for i in np.flatnonzero(hi > lo):
        a, b = lo[i], hi[i]
        k = np.hypot(jeta[a:b] - eta[i], A._dphi(jphi[a:b], phi[i])) < R
        if k.any():
            out[i] = jpt[a:b][k].max()
    return out


def _sub_neg(je, jeta, jphi, jx, jy, neg, R):
    """oo_analysis_rhic.subtract_negatives (frac 1) with per-event slices: the jets' pT after subtracting the
    negatives' (holes, pstat -1 hadrons, partner copies) transverse momenta inside dR < R."""
    sx, sy = np.zeros(len(je)), np.zeros(len(je))
    if neg is not None and len(neg) and len(je):
        neg = neg[np.argsort(neg[:, 0], kind="stable")]
        nev = neg[:, 0].astype(int)
        _, neta, nphi = A._kin(neg[:, 1:5])
        lo, hi = np.searchsorted(nev, je, "left"), np.searchsorted(nev, je, "right")
        for i in np.flatnonzero(hi > lo):
            a, b = lo[i], hi[i]
            ins = np.hypot(neta[a:b] - jeta[i], A._dphi(nphi[a:b], jphi[i])) < R
            sx[i] = neg[a:b][ins, 2].sum(); sy[i] = neg[a:b][ins, 3].sum()
    return np.hypot(jx - sx, jy - sy)


def _cone_sums(ev, eta, phi, cols, cone_ev, cone_eta, cone_phi):
    """per cone (event, eta, phi): the sums of each column of cols (N, k) over the particles of its event inside
    dR < CONE (ev sorted)."""
    out = np.zeros((len(cone_ev), cols.shape[1]))
    lo, hi = np.searchsorted(ev, cone_ev, "left"), np.searchsorted(ev, cone_ev, "right")
    for i in range(len(cone_ev)):
        a, b = lo[i], hi[i]
        ins = np.hypot(eta[a:b] - cone_eta[i], A._dphi(phi[a:b], cone_phi[i])) < CONE
        out[i] = cols[a:b][ins].sum(0)
    return out


def _hist_w(ev, x, n):
    """per event histogram on EDGES (counts)."""
    return A._hist(ev, x, EDGES, n)


def one_file(args):
    """One produced file and its re-hadronizations: per event the window record, parton jets, and per kind the
    hadron jets, the matched r and the energy balance."""
    f, name, kinds = args
    import h5py
    import hdf5plugin  # noqa: F401
    with h5py.File(f) as h:
        n = int(h.attrs["nunits_written"])
        sig = np.asarray(h.attrs["pthat_bin_sigma_gen"], float)
        nacc = np.asarray(h.attrs["pthat_bin_n_accepted"], float)
        bins = [tuple(map(float, w)) for w in np.asarray(h.attrs["pthat_bins"]).reshape(-1, 2)]
        win = h["units/pthat_bin"][:n].astype(int); pth = h["units/pthat"][:n].astype(float)
        yl, ys = h["units/parton_y_lead"][:n], h["units/parton_y_sub"][:n]
        ycut = "parton_ymax" in h.attrs
        po = h["partons/offsets"][: n + 1]; par = h["partons/data"][: po[-1]].astype(float)
    keep = np.ones(n, bool) if ycut else (np.abs(yl) < YMAX) | (np.abs(ys) < YMAX)
    evp = np.repeat(np.arange(n), np.diff(po))
    st = par[:, 2].astype(int); p4 = par[:, 3:7]
    base = np.isin(st, (0, 1000, 22)); rec = st == 1; hol = st == -1
    hneg = np.column_stack([evp[hol], p4[hol]]) if hol.any() else np.zeros((0, 5))
    out = {"n": n, "bins": bins, "sig": sig, "nacc": nacc, "win": win, "pthat": pth, "keep": keep,
           "E_partons": np.bincount(evp[base | rec], weights=p4[base | rec, 0], minlength=n),
           "E_holes": np.bincount(evp[hol], weights=p4[hol, 0], minlength=n)}
    pj = {}
    for R in RS:
        je, jpt, jeta, jphi = A._jets(p4[base], evp[base], n, R, None, ptmin=3.0)
        k = np.abs(jeta) < 1 - R
        out[f"par_R{R}"] = _hist_w(je[k], jpt[k], n)
        sel = base | rec
        je, jpt, jeta, jphi, jx, jy = A._jets(p4[sel], evp[sel], n, R, None, ptmin=3.0, full=True)
        jsub = _sub_neg(je, jeta, jphi, jx, jy, hneg, R)
        k = np.abs(jeta) < 1 - R
        out[f"resp_R{R}"] = _hist_w(je[k], jsub[k], n)
        m = k & (jsub > PT_MATCH)
        pj[R] = (je[m], jsub[m], jeta[m], jphi[m])
    # the cone around the parton jets (R = 0.4, resp, pT > 10, |eta| < 0.6) for the energy balance
    je4, pt4, eta4, phi4 = pj[0.4]
    cj = (pt4 > 10) & (np.abs(eta4) < 0.6)
    cone_ev, cone_eta, cone_phi = je4[cj], eta4[cj], phi4[cj]
    ppt = np.hypot(p4[:, 1], p4[:, 2]); peta = np.arcsinh(p4[:, 3] / np.maximum(ppt, 1e-9)); pphi = np.arctan2(p4[:, 2], p4[:, 1])
    pr = (base | rec).astype(float); ph = hol.astype(float)
    out["cone_ev"] = cone_ev; out["cone_pt"] = pt4[cj]
    out["cone_par"] = _cone_sums(evp, peta, pphi, np.column_stack([ppt * pr, ppt * ph, p4[:, 0] * pr, p4[:, 0] * ph]),
                                 cone_ev, cone_eta, cone_phi)
    for kind in kinds:
        g = rehad_path(f, name, kind)
        if not os.path.exists(g):
            continue
        with h5py.File(g) as h:
            if int(h.attrs["nunits_written"]) != n:
                continue
            so, uo = h["hadrons/sample_offsets"][:], h["hadrons/unit_offsets"][: n + 1]
            s0 = uo[:-1]                                            # sample 0 of each unit
            lo, hi = so[s0], so[s0 + 1]
            cnt = (hi - lo).astype(int)
            idx = (np.repeat(lo - np.r_[0, np.cumsum(cnt)[:-1]], cnt) + np.arange(cnt.sum())).astype(np.int64)
            ev = np.repeat(np.arange(n), cnt)
            top = int(hi.max()) if n else 0
            pa = h["hadrons/p"][:top][idx].astype(float) if len(idx) else np.zeros((0, 4))
            pid = h["hadrons/pid"][:top][idx]
            pst = h["hadrons/pstat"][:top][idx] if "hadrons/pstat" in h else np.zeros(len(idx), int)
            neg_ic = A.negative_partons(h, n) if not (pst == -1).any() else None
        pt, eta, phi = A._kin(pa)
        negh = pst == -1
        vis = ~np.isin(np.abs(pid), A.INVISIBLE)
        ch = np.isin(np.abs(pid), B.charged_ids())
        d = {"E_pos": np.bincount(ev[~negh], weights=pa[~negh, 0], minlength=n),
             "E_neg": np.bincount(ev[negh], weights=pa[negh, 0], minlength=n)}
        for R in RS:
            m = vis & (pt > A.CONST_PTMIN) & (np.abs(eta) < 1.0)
            je, jpt, jeta, jphi, jx, jy = A._jets(pa[m & ~negh], ev[m & ~negh], n, R, None, ptmin=3.0, full=True)
            if negh.any():
                mn = m & negh
                neg = np.column_stack([ev[mn], pa[mn]]) if mn.any() else np.zeros((0, 5))
            else:
                neg = neg_ic
            jsub = _sub_neg(je, jeta, jphi, jx, jy, neg, R)
            k = np.abs(jeta) < 1 - R
            d[f"had_R{R}"] = _hist_w(je[k], jsub[k], n)
            pe, ppt_, peta_, pphi_ = pj[R]
            d[f"r_R{R}"] = _match(je[k], jsub[k], jeta[k], jphi[k], pe, peta_, pphi_, R) / ppt_
        # energy balance in the parton-jet cone: visible / charged pT, positive and pstat -1 hadrons
        cols = np.column_stack([pt * (vis & ~negh), pt * (vis & negh), pt * (ch & ~negh), pt * (ch & negh),
                                pa[:, 0] * ~negh, pa[:, 0] * negh])
        d["cone"] = _cone_sums(ev, eta, phi, cols, cone_ev, cone_eta, cone_phi)
        out[kind] = d
    out["pj_ev"] = {R: pj[R][0] for R in RS}
    out["pj_pt"] = {R: pj[R][1] for R in RS}
    return out


def collect(fs, name, kinds, j):
    """one_file over the files (those with a .json), the window weights stitched as auau_raa.py."""
    fs = [f for f in fs if os.path.exists(f.replace("_hadrons_jet_frag.h5", ".json"))]
    with ProcessPoolExecutor(max_workers=j) as ex:
        res = list(ex.map(one_file, [(f, name, kinds) for f in fs]))
    wl = sorted({w for r in res for w in r["bins"]})
    sig_n, nacc = np.zeros(len(wl)), np.zeros(len(wl))
    win, pth, keep, off = [], [], [], [0]
    for r in res:
        g = np.array([wl.index(x) for x in r["bins"]])
        sig_n[g] += r["sig"] * r["nacc"]; nacc[g] += r["nacc"]
        win.append(g[r["win"]]); pth.append(r["pthat"]); keep.append(r["keep"]); off.append(off[-1] + r["n"])
    win, pth, keep = np.concatenate(win), np.concatenate(pth), np.concatenate(keep)
    w, *_ = A.stitch_pieces(wl, win, pth, sig_n / np.maximum(nacc, 1))
    return res, w * keep, np.array(off), len(fs)


def spec(res, w, off, key, kind=None):
    """weighted spectrum (sum, sum w^2) over the files, per EDGES bin, per GeV."""
    s, s2 = np.zeros(len(EDGES) - 1), np.zeros(len(EDGES) - 1)
    for r, o in zip(res, off[:-1]):
        if kind and kind not in r:                          # a re-hadronization missing for this file
            continue
        d = r[kind] if kind else r
        x = w[o:o + r["n"], None] * d[key]
        s += x.sum(0); s2 += (x ** 2).sum(0)
    return s, s2


def r_mean(res, w, off, R, kind):
    """weighted <r> per parton-jet pT bin (EDGES), its error and n_eff, and the unmatched fraction."""
    nb = len(EDGES) - 1
    sw, swr, swr2, sw2, unm = (np.zeros(nb) for _ in range(5))
    for r, o in zip(res, off[:-1]):
        if kind not in r:
            continue
        ev, pt, rr = r["pj_ev"][R], r["pj_pt"][R], r[kind][f"r_R{R}"]
        ww = w[o + ev]; b = np.digitize(pt, EDGES) - 1; ok = (b >= 0) & (b < nb)
        np.add.at(sw, b[ok], ww[ok]); np.add.at(swr, b[ok], (ww * rr)[ok]); np.add.at(swr2, b[ok], (ww * rr ** 2)[ok])
        np.add.at(sw2, b[ok], (ww ** 2)[ok]); np.add.at(unm, b[ok], (ww * (rr == 0))[ok])
    mean = np.divide(swr, sw, out=np.full(nb, np.nan), where=sw > 0)
    var = np.divide(swr2, sw, out=np.zeros(nb), where=sw > 0) - mean ** 2
    neff = np.divide(sw ** 2, sw2, out=np.zeros(nb), where=sw2 > 0)
    err = np.sqrt(np.maximum(var, 0) / np.maximum(neff, 1))
    return mean, err, neff, np.divide(unm, sw, out=np.zeros(nb), where=sw > 0)


def balance(res, w, off, kind):
    """weighted per-event means of the energy balance, and per cone (visible / charged pT)."""
    tot = {k: 0.0 for k in ("w", "E_partons", "E_holes", "E_pos", "E_neg")}
    cone = np.zeros(10); cw = 0.0
    for r, o in zip(res, off[:-1]):
        if kind not in r:
            continue
        ww = w[o:o + r["n"]]
        tot["w"] += ww.sum()
        for k in ("E_partons", "E_holes"):
            tot[k] += (ww * r[k]).sum()
        for k in ("E_pos", "E_neg"):
            tot[k] += (ww * r[kind][k]).sum()
        if len(r["cone_ev"]):
            wc = w[o + r["cone_ev"]]
            cone[:4] += (wc[:, None] * r["cone_par"]).sum(0)
            cone[4:] += (wc[:, None] * r[kind]["cone"]).sum(0)
            cw += wc.sum()
    inc = {k: tot[k] / tot["w"] for k in ("E_partons", "E_holes", "E_pos", "E_neg")}
    names = ("par_pt", "hole_pt", "par_E", "hole_E", "vis_pos_pt", "vis_neg_pt", "ch_pos_pt", "ch_neg_pt", "pos_E", "neg_E")
    return inc, dict(zip(names, (cone / max(cw, 1e-300)).tolist()))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-j", type=int, default=14)
    ap.add_argument("--max-files", type=int, default=0, help="test: the first N files per sample")
    a = ap.parse_args(argv)
    out = {"edges": EDGES.tolist(), "R": list(RS), "pt_match": PT_MATCH}
    for smp, (ga, na, gp, npp, kinds) in SAMPLES.items():
        fa, fp = sorted(glob.glob(ga)), sorted(glob.glob(gp))
        if a.max_files:                                      # test: a few files of each
            fa, fp = fa[:: max(1, len(fa) // a.max_files)][: a.max_files], fp[:: max(1, len(fp) // a.max_files)][: a.max_files]
        RA_, wA, oA, nfa = collect(fa, na, kinds, a.j)
        RP_, wP, oP, nfp = collect(fp, npp, kinds, a.j)
        o = {"files_auau": nfa, "files_pp": nfp, "events_auau": int(oA[-1]), "events_pp": int(oP[-1])}
        for R in RS:
            for lvl in ("par", "resp"):
                sa, sa2 = spec(RA_, wA, oA, f"{lvl}_R{R}"); sp, sp2 = spec(RP_, wP, oP, f"{lvl}_R{R}")
                o[f"{lvl}_R{R}"] = {"auau": sa.tolist(), "pp": sp.tolist(), "neff_auau": (sa ** 2 / np.maximum(sa2, 1e-300)).tolist(),
                                    "neff_pp": (sp ** 2 / np.maximum(sp2, 1e-300)).tolist(),
                                    "raa": np.divide(sa, sp, out=np.full_like(sa, np.nan), where=sp > 0).tolist(),
                                    "err": (np.divide(sa, sp, out=np.zeros_like(sa), where=sp > 0)
                                            * np.sqrt(np.divide(sa2, sa ** 2, out=np.zeros_like(sa), where=sa > 0)
                                                      + np.divide(sp2, sp ** 2, out=np.zeros_like(sp), where=sp > 0))).tolist()}
        for kind in kinds:
            d = {}
            for R in RS:
                sa, sa2 = spec(RA_, wA, oA, f"had_R{R}", kind); sp, sp2 = spec(RP_, wP, oP, f"had_R{R}", kind)
                d[f"had_R{R}"] = {"auau": sa.tolist(), "pp": sp.tolist(),
                                  "raa": np.divide(sa, sp, out=np.full_like(sa, np.nan), where=sp > 0).tolist(),
                                  "err": (np.divide(sa, sp, out=np.zeros_like(sa), where=sp > 0)
                                          * np.sqrt(np.divide(sa2, sa ** 2, out=np.zeros_like(sa), where=sa > 0)
                                                    + np.divide(sp2, sp ** 2, out=np.zeros_like(sp), where=sp > 0))).tolist(),
                                  "neff_auau": (sa ** 2 / np.maximum(sa2, 1e-300)).tolist(),
                                  "neff_pp": (sp ** 2 / np.maximum(sp2, 1e-300)).tolist()}
                ma, ea, na_, ua = r_mean(RA_, wA, oA, R, kind); mp, ep, np_, up = r_mean(RP_, wP, oP, R, kind)
                dr = ma / mp
                d[f"r_R{R}"] = {"auau": ma.tolist(), "auau_err": ea.tolist(), "auau_neff": na_.tolist(), "auau_unmatched": ua.tolist(),
                                "pp": mp.tolist(), "pp_err": ep.tolist(), "pp_neff": np_.tolist(), "pp_unmatched": up.tolist(),
                                "double": dr.tolist(), "double_err": (np.abs(dr) * np.hypot(ea / ma, ep / mp)).tolist()}
            inc_a, cone_a = balance(RA_, wA, oA, kind); inc_p, cone_p = balance(RP_, wP, oP, kind)
            d["balance"] = {"auau": inc_a, "auau_cone": cone_a, "pp": inc_p, "pp_cone": cone_p}
            o[kind] = d
        out[smp] = o
        print(f"[{smp}] Au+Au {o['events_auau']} events ({nfa} files), p+p {o['events_pp']} ({nfp} files)", flush=True)
        for R in RS:
            print(f"  R={R}: parton R_AA (resp) " + " ".join(f"{x:.2f}" for x in o[f'resp_R{R}']['raa']), flush=True)
            for kind in kinds:
                print(f"    {kind:15s} hadron R_AA " + " ".join(f"{x:.2f}" for x in o[kind][f'had_R{R}']['raa'])
                      + " | <r>AA/<r>pp " + " ".join(f"{x:.2f}" for x in o[kind][f'r_R{R}']['double']), flush=True)
    fn = ANA / "results" / ("css_parton_jets.json" if not a.max_files else "css_parton_jets_test.json")
    fn.write_text(json.dumps(out, indent=1, default=float))
    print("->", fn)


if __name__ == "__main__":
    main()
