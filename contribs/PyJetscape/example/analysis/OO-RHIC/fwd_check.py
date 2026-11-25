#!/usr/bin/env python3
"""
analysis/OO-RHIC/fwd_check.py -- forward activity of p+p jet events against minimum bias, in the
centrality acceptances (EPD 2.1 < |eta| < 5.1, MBD 3.51 < |eta| < 4.61), for the phase-2 p+p
samples of the physics run (MPI on, MPI off, string veto 0).
    python fwd_check.py --out /data/OO-RHIC_phys -j 16     # -> results/fwd_check.json, a table

Why.  The gate-G2 UE comparison is at |eta| < 1; the selection bias of the O+O centrality classes
is set by the forward activity of jet events relative to minimum bias.  This puts the two side by
side per leading-jet bin: transverse density / minimum bias (the G2 quantity) and EPD, MBD /
minimum bias, plus the EPD split into the leading jet's hemisphere (same sign of eta) and the
opposite one.  If the three settings spread forward as they do at mid-rapidity, the O+O
string-switch run is a bracket of the forward response; if not, it is not.
Same jet definition as ue_check.py (anti-kT R = 0.6, charged + neutral visible pT > 0.2 GeV,
|eta| < 1, leading jet |eta| < 0.4).  Forward counts: charged hadrons, no pT cut (as oo_bias's
estimators), jet and bulk hadrons of the event.  Minimum bias: the p+p bulk sample pp_tau0424
(bulk hadrons only), all events and NSD (STAR BBC 3.3-5.0 coincidence).  Events weighted by
sigma_k / N_k per pTHat window.  Leading-jet bins below ~11 GeV are incomplete (no pTHat < 5).
Model numbers with the frozen tune (XML 788ca1e), not a vetted physics result.
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
import config as C  # noqa: E402
import observables as O  # noqa: E402
import ue_check as U  # noqa: E402
import g2_check as G  # noqa: E402

EPD, MBD, BBC = (2.1, 5.1), (3.51, 4.61), (3.3, 5.0)
VARIANTS = G.UE_VARIANTS
QUANT = ("n_trans", "n_epd", "n_epd_same", "n_epd_opp", "n_mbd", "mbd_coinc", "has_bulk")
TRANS_AREA = 2.0 * 2 * np.pi / 3


def file_sums(path):
    import awkward as ak
    import fastjet
    fd = O.read_file(path)
    nwin, nb, nev = len(fd["sigma"]), len(U.LEAD_EDGES) - 1, len(fd["counts"])
    pt, _, eta, phi = O._kin(fd["p"])
    ev = np.repeat(np.arange(nev), fd["counts"])
    pid = np.abs(fd["pid"])
    vis = ~np.isin(pid, C.INVISIBLE) & (pt > C.CONST_PTMIN) & (np.abs(eta) < U.TRK_ETA)
    cc = np.bincount(ev[vis], minlength=nev)
    P = fd["p"][vis]
    parts = ak.zip({k: ak.unflatten(P[:, i], cc)
                    for k, i in (("px", 1), ("py", 2), ("pz", 3), ("E", 0))})
    js = fastjet.ClusterSequence(parts, fastjet.JetDefinition(
        fastjet.antikt_algorithm, U.JET_R_UE)).inclusive_jets(min_pt=2.0)
    jpt = np.hypot(js.px, js.py)
    jp = np.sqrt(js.px ** 2 + js.py ** 2 + js.pz ** 2)
    jeta = 0.5 * np.log(np.maximum(jp + js.pz, 1e-12) / np.maximum(jp - js.pz, 1e-12))
    jpt = ak.where(np.abs(jeta) < U.JET_ETA, jpt, -1.0)
    imax = ak.argmax(jpt, axis=1, keepdims=True)
    lead_pt = ak.to_numpy(ak.fill_none(ak.firsts(jpt[imax]), -1.0))
    lead_phi = ak.to_numpy(ak.fill_none(ak.firsts(np.arctan2(js.py, js.px)[imax]), 0.0))
    lead_eta = ak.to_numpy(ak.fill_none(ak.firsts(jeta[imax]), 0.0))
    lb = np.digitize(lead_pt, U.LEAD_EDGES) - 1
    good = (lead_pt > 0) & (lb >= 0) & (lb < nb)

    ch = np.isin(pid, O._charged_ids())
    e_ch, eta_ch = ev[ch], eta[ch]
    mid = (pt[ch] > 0.2) & (np.abs(eta_ch) < 1.0)
    dphi = np.abs((phi[ch] - lead_phi[e_ch] + np.pi) % (2 * np.pi) - np.pi)
    tr = mid & (dphi > np.pi / 3) & (dphi < 2 * np.pi / 3)
    epd = (np.abs(eta_ch) > EPD[0]) & (np.abs(eta_ch) < EPD[1])
    same = np.sign(eta_ch) == np.where(lead_eta[e_ch] >= 0, 1.0, -1.0)
    mpos = (eta_ch > MBD[0]) & (eta_ch < MBD[1])
    mneg = (-eta_ch > MBD[0]) & (-eta_ch < MBD[1])
    bc = lambda m: np.bincount(e_ch[m], minlength=nev)  # noqa: E731
    npos, nneg = bc(mpos), bc(mneg)
    q = {"n_trans": bc(tr), "n_epd": bc(epd), "n_epd_same": bc(epd & same),
         "n_epd_opp": bc(epd & ~same), "n_mbd": npos + nneg,
         "mbd_coinc": (npos > 0) & (nneg > 0),
         "has_bulk": (fd["n_bulk"] > 0 if fd.get("n_bulk") is not None else np.zeros(nev, bool))}
    flat = fd["win"][good] * nb + lb[good]
    out = {"n": np.bincount(flat, minlength=nwin * nb).reshape(nwin, nb).astype(float)}
    for k in QUANT:
        v = q[k][good].astype(float)
        out[k] = np.bincount(flat, weights=v, minlength=nwin * nb).reshape(nwin, nb)
        out[k + "_2"] = np.bincount(flat, weights=v * v, minlength=nwin * nb).reshape(nwin, nb)
    return (tuple(map(tuple, fd["bins"].tolist())), fd["sigma"], fd["n_acc"],
            np.bincount(fd["win"], minlength=nwin), out)


def sums(files, j):
    with ProcessPoolExecutor(max_workers=j) as ex:
        res = list(ex.map(file_sums, files))
    edges = sorted({w for r in res for w in r[0]})
    nwin, nb = len(edges), len(U.LEAD_EDGES) - 1
    acc = {k: np.zeros((nwin, nb)) for k in res[0][4]}
    sig_w, nacc, nev = np.zeros(nwin), np.zeros(nwin), np.zeros(nwin)
    for bins, sig, na, ne, out in res:
        g = np.array([edges.index(w) for w in bins])
        for k in acc:
            acc[k][g] += out[k]
        sig_w[g] += na * sig
        nacc[g] += na
        nev[g] += ne
    acc.update(sigma=sig_w / np.maximum(nacc, 1), n_events=nev)
    return acc


def means(acc):
    """Per leading-jet bin: sigma/N-weighted mean of each quantity and its error."""
    w = (acc["sigma"] / np.maximum(acc["n_events"], 1))[:, None]
    n = acc["n"]
    den = (w * n).sum(0)
    out = {"n_events": n.sum(0)}
    for k in QUANT:
        s1, s2 = acc[k], acc[k + "_2"]
        var_k = np.where(n > 1, s2 / np.maximum(n, 1) - (s1 / np.maximum(n, 1)) ** 2, 0.0)
        m = (w * s1).sum(0) / np.maximum(den, 1e-300)
        e = np.sqrt((w ** 2 * n * var_k).sum(0)) / np.maximum(den, 1e-300)
        out[k], out[k + "_err"] = m, e
    return out


def mb_forward(root, sample):
    """Minimum bias (bulk hadrons only): mean EPD, MBD per event, MBD coincidence, transverse-
    equivalent mid-rapidity density (pT > 0.2, |eta| < 1, per d eta d phi); all events and NSD."""
    import h5py
    import hdf5plugin  # noqa: F401
    import oo_bias as B
    rec = {k: [] for k in ("epd", "mbd", "coinc", "mid", "nsd")}
    for bf in sorted(glob.glob(f"{root}/{sample}/*_hadrons_bulk_bg.h5")):
        if not os.path.exists(bf.replace("_hadrons_bulk_bg.h5", ".json")):
            continue
        with h5py.File(bf) as h:
            n = int(h.attrs["nunits_written"])
            p, c = B._units(h, n)
        E, px, py, pz = p.T
        pt = np.hypot(px, py)
        pa = np.sqrt(px * px + py * py + pz * pz)
        eta = 0.5 * np.log(np.clip(pa + pz, 1e-12, None) / np.clip(pa - pz, 1e-12, None))
        ev = np.repeat(np.arange(n), c)
        bc = lambda m: np.bincount(ev[m], minlength=n)  # noqa: E731
        npos, nneg = bc((eta > MBD[0]) & (eta < MBD[1])), bc((-eta > MBD[0]) & (-eta < MBD[1]))
        rec["epd"].append(bc((np.abs(eta) > EPD[0]) & (np.abs(eta) < EPD[1])))
        rec["mbd"].append(npos + nneg)
        rec["coinc"].append((npos > 0) & (nneg > 0))
        rec["mid"].append(bc((np.abs(eta) < 1.0) & (pt > 0.2)))
        rec["nsd"].append((bc((eta > BBC[0]) & (eta < BBC[1])) > 0)
                          & (bc((-eta > BBC[0]) & (-eta < BBC[1])) > 0))
    rec = {k: np.concatenate(v) for k, v in rec.items()}
    out = {"sample": f"{root}/{sample}", "events": int(len(rec["epd"])),
           "frac_nsd": float(rec["nsd"].mean())}
    for lab, m in (("all", np.ones(len(rec["epd"]), bool)), ("nsd", rec["nsd"])):
        for k, x, norm in (("epd", rec["epd"], 1.0), ("mbd", rec["mbd"], 1.0),
                           ("coinc", rec["coinc"], 1.0), ("mid", rec["mid"], 2 * 2 * np.pi)):
            x = x[m].astype(float)
            out[f"{lab}_{k}"] = float(x.mean() / norm)
            out[f"{lab}_{k}_err"] = float(x.std() / np.sqrt(len(x)) / norm)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/data/OO-RHIC_phys")
    ap.add_argument("-j", type=int, default=16)
    a = ap.parse_args(argv)
    mb = mb_forward(*G.MB_PP)
    res = {"lead_edges": U.LEAD_EDGES.tolist(), "complete_from_gev": G.COMPLETE_FROM, "mb": mb,
           "variants": {}}
    cen = 0.5 * (U.LEAD_EDGES[:-1] + U.LEAD_EDGES[1:])
    for v, lab in VARIANTS.items():
        files = G.ue_files(Path(a.out), v)
        if not files:
            print(f"{v}: no files", file=sys.stderr)
            continue
        m = means(sums(files, a.j))
        r = {"label": lab, "files": len(files)}
        for k in QUANT:
            r[k], r[k + "_err"] = m[k].tolist(), m[k + "_err"].tolist()
        r["n_events"] = m["n_events"].tolist()
        for base in ("all", "nsd"):
            r[f"trans_over_mb_{base}"] = (m["n_trans"] / TRANS_AREA / mb[f"{base}_mid"]).tolist()
            r[f"epd_over_mb_{base}"] = (m["n_epd"] / mb[f"{base}_epd"]).tolist()
            r[f"mbd_over_mb_{base}"] = (m["n_mbd"] / mb[f"{base}_mbd"]).tolist()
            r[f"epd_same_over_mb_{base}"] = (m["n_epd_same"] / (0.5 * mb[f"{base}_epd"])).tolist()
            r[f"epd_opp_over_mb_{base}"] = (m["n_epd_opp"] / (0.5 * mb[f"{base}_epd"])).tolist()
        res["variants"][v] = r
    Path(HERE / "results").mkdir(exist_ok=True)
    with open(HERE / "results" / "fwd_check.json", "w") as f:
        json.dump(res, f, indent=1)

    print(f"minimum bias {mb['sample']}: {mb['events']} events, NSD {mb['frac_nsd']:.3f}")
    for base in ("all", "nsd"):
        print(f"  {base}: EPD {mb[base + '_epd']:.2f}  MBD {mb[base + '_mbd']:.2f}  "
              f"MBD coinc {mb[base + '_coinc']:.3f}  mid dN/(deta dphi) {mb[base + '_mid']:.3f}")
    for v, r in res["variants"].items():
        print(f"\n{v} ({r['label']}): ratios to minimum bias (all events) | NSD in brackets")
        print(f"{'lead jet':>10} {'events':>7} {'trans':>12} {'EPD':>12} {'MBD':>12} "
              f"{'EPD same':>9} {'EPD opp':>8} {'coinc':>6} {'bulk':>5}")
        for i, c in enumerate(cen):
            lo, hi = U.LEAD_EDGES[i], U.LEAD_EDGES[i + 1]
            print(f"{lo:4.0f}-{hi:<4.0f}{'*' if lo < G.COMPLETE_FROM else ' '}"
                  f"{r['n_events'][i]:7.0f} "
                  f"{r['trans_over_mb_all'][i]:5.2f} ({r['trans_over_mb_nsd'][i]:4.2f}) "
                  f"{r['epd_over_mb_all'][i]:5.2f} ({r['epd_over_mb_nsd'][i]:4.2f}) "
                  f"{r['mbd_over_mb_all'][i]:5.2f} ({r['mbd_over_mb_nsd'][i]:4.2f}) "
                  f"{r['epd_same_over_mb_all'][i]:9.2f} {r['epd_opp_over_mb_all'][i]:8.2f} "
                  f"{r['mbd_coinc'][i]:6.3f} {r['has_bulk'][i]:5.2f}")
        print("  (* incomplete leading-jet bin; relative errors of the EPD ratios: "
              + " ".join(f"{e / max(x, 1e-9):.1%}" for x, e in zip(r["n_epd"], r["n_epd_err"])) + ")")
    return 0


if __name__ == "__main__":
    sys.exit(main())
