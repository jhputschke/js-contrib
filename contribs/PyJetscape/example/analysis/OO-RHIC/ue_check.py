#!/usr/bin/env python3
"""
analysis/OO-RHIC/ue_check.py -- the underlying event of the p+p hard runs against STAR's
measurement vs the leading jet (PRD 101, 052004 (2020), arXiv:1912.08187; HEPData ins1771348).

    python ue_check.py best best_mpi4 best_nompi -j 8      # -> results/ue_<run>.npz, a table
    python ue_check.py trunc --check-truncation            # share of pTHat 4-5 GeV per bin

What it shows.  The runs are vacuum-mode p+p (no strings, no hydro): the underlying event is
the beam remnants (hadronized by iColored with the jet) and the MPI above the 5 GeV cut.  The
comparison says how much of STAR's transverse activity this hard chain makes by itself; the
rest has to come from the soft sector (strings + hydro, or softer MPI) -- step 3 of the plan.

STAR's definition (as in the paper): leading jet anti-kT R = 0.6 from charged tracks and
towers (pT, ET > 0.2 GeV, |eta| < 1), |eta_jet| < 1 - R = 0.4; charged particles pT > 0.2 (and
> 0.5) GeV, |eta| < 1, in the regions toward |dphi| < 60 deg, transverse 60-120 deg, away
> 120 deg relative to the leading jet; densities dN_ch / (deta dphi) with deta = 2 and
dphi = 2 pi / 3 per region (the transverse region is both sides, 60-120 deg; TransMax and
TransMin one side each, pi / 3).  Leading-jet bins (centres 6, 8, 10, 13, 17.5, 22.5, 30,
40 GeV in HEPData) taken as 5-7, 7-9, 9-11, 11-15, 15-20, 20-25, 25-35, 35-45 GeV.

Model: all visible final particles (no neutrinos) for the jets; charged = jetscape's charged
list.  pi0 and eta are undecayed in these runs (their eta -> 3 pi charged pions are missing,
a percent-level effect on the densities).  Events weighted by sigma_k / N_k per pTHat window;
the leading-jet bins below ~9-11 GeV are incomplete (no pTHat < 5 GeV; --check-truncation).
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import config as C
import observables as O

UE_REC = "ins1771348"
JET_R_UE = 0.6
JET_ETA = 0.4
TRK_ETA = 1.0
LEAD_EDGES = np.array([5, 7, 9, 11, 15, 20, 25, 35, 45], dtype=np.float64)
REGIONS = ("toward", "transverse", "away")
#: quantities per event, accumulated per (window, leading-jet bin)
QUANT = ("n_toward", "n_transverse", "n_away", "n_trans05", "sumpt_transverse",
         "n_transmax", "n_transmin")


def load_star():
    """STAR's tables (fetch: fetch_data.download) -> dict of arrays."""
    import yaml
    import fetch_data as F
    out = {"edges": LEAD_EDGES.tolist()}
    for fig, keys in (("Figure 2", ("toward", "away", "transverse")),
                      ("Figure 3", ("transverse", "transverse05")),
                      ("Figure 5", ("meanpt_transverse", "meanpt_transverse05"))):
        path = F.raw_path(UE_REC, fig)
        if not path.exists():
            F.download(UE_REC, fig)
        d = yaml.safe_load(path.read_text())
        for key, dv in zip(keys, d["dependent_variables"]):
            val = np.array([v["value"] for v in dv["values"]])
            st = np.array([e["symerror"] for v in dv["values"] for e in v["errors"]
                           if e["label"] == "stat"])
            sy = np.array([max(abs(e["asymerror"]["minus"]), abs(e["asymerror"]["plus"]))
                           if "asymerror" in e else abs(e["symerror"])
                           for v in dv["values"] for e in v["errors"] if e["label"] == "sys"])
            out[key] = dict(value=val, stat=st, sys=sy)
    return out


def file_sums(path):
    """Per (window, leading-jet bin): number of events and sums of QUANT and their squares."""
    import awkward as ak
    import fastjet
    fd = O.read_file(path)
    nwin = len(fd["sigma"])
    nb = len(LEAD_EDGES) - 1
    nev = len(fd["counts"])
    pt, _, eta, phi = O._kin(fd["p"])
    ev = np.repeat(np.arange(nev), fd["counts"])
    pid = np.abs(fd["pid"])
    vis = ~np.isin(pid, C.INVISIBLE) & (pt > C.CONST_PTMIN) & (np.abs(eta) < TRK_ETA)
    cc = np.bincount(ev[vis], minlength=nev)
    P = fd["p"][vis]
    parts = ak.zip({k: ak.unflatten(P[:, i], cc)
                    for k, i in (("px", 1), ("py", 2), ("pz", 3), ("E", 0))})
    js = fastjet.ClusterSequence(parts, fastjet.JetDefinition(
        fastjet.antikt_algorithm, JET_R_UE)).inclusive_jets(min_pt=2.0)
    jpt = np.hypot(js.px, js.py)
    jp = np.sqrt(js.px ** 2 + js.py ** 2 + js.pz ** 2)
    jeta = 0.5 * np.log(np.maximum(jp + js.pz, 1e-12) / np.maximum(jp - js.pz, 1e-12))
    jpt = ak.where(np.abs(jeta) < JET_ETA, jpt, -1.0)          # leading jet in |eta| < 0.4
    imax = ak.argmax(jpt, axis=1, keepdims=True)
    lead_pt = ak.to_numpy(ak.fill_none(ak.firsts(jpt[imax]), -1.0))
    lead_phi = ak.to_numpy(ak.fill_none(ak.firsts(np.arctan2(js.py, js.px)[imax]), 0.0))
    lb = np.digitize(lead_pt, LEAD_EDGES) - 1
    good = (lead_pt > 0) & (lb >= 0) & (lb < nb)
    # charged tracks relative to the leading jet
    ch = np.isin(pid, O._charged_ids()) & (pt > 0.2) & (np.abs(eta) < TRK_ETA)
    e_ch = ev[ch]
    dphi = np.abs((phi[ch] - lead_phi[e_ch] + np.pi) % (2 * np.pi) - np.pi)
    # the two transverse sides: sign of the azimuth difference
    sgn = np.sign(((phi[ch] - lead_phi[e_ch] + np.pi) % (2 * np.pi) - np.pi))
    tow, aw = dphi < np.pi / 3, dphi > 2 * np.pi / 3
    tr = ~tow & ~aw
    q = {
        "n_toward": np.bincount(e_ch[tow], minlength=nev),
        "n_transverse": np.bincount(e_ch[tr], minlength=nev),
        "n_away": np.bincount(e_ch[aw], minlength=nev),
        "n_trans05": np.bincount(e_ch[tr & (pt[ch] > 0.5)], minlength=nev),
        "sumpt_transverse": np.bincount(e_ch[tr], weights=pt[ch][tr], minlength=nev),
    }
    s1 = np.bincount(e_ch[tr & (sgn > 0)], minlength=nev)
    s2 = np.bincount(e_ch[tr & (sgn <= 0)], minlength=nev)
    q["n_transmax"], q["n_transmin"] = np.maximum(s1, s2), np.minimum(s1, s2)
    win = fd["win"]
    flat = win[good] * nb + lb[good]
    out = {"n": np.bincount(flat, minlength=nwin * nb).reshape(nwin, nb).astype(float)}
    for k in QUANT:
        v = q[k][good].astype(float)
        out[k] = np.bincount(flat, weights=v, minlength=nwin * nb).reshape(nwin, nb)
        out[k + "_2"] = np.bincount(flat, weights=v * v, minlength=nwin * nb).reshape(nwin, nb)
    return (tuple(map(tuple, fd["bins"].tolist())), fd["sigma"], fd["n_acc"],
            np.bincount(win, minlength=nwin), out)


def run_sums(run, j=8):
    files = sorted(glob.glob(str(C.point_dir(run) / "*_hadrons_jet_frag.h5")))
    if not files:
        sys.exit(f"ue_check.py: no files for {run}")
    with ProcessPoolExecutor(max_workers=j) as ex:
        res = list(ex.map(file_sums, files))
    edges = sorted({w for r in res for w in r[0]})
    nwin, nb = len(edges), len(LEAD_EDGES) - 1
    acc = {k: np.zeros((nwin, nb)) for k in res[0][4]}
    sig_w, nacc, nev = np.zeros(nwin), np.zeros(nwin), np.zeros(nwin)
    for bins, sig, na, ne, out in res:
        g = np.array([edges.index(w) for w in bins])
        for k in acc:
            acc[k][g] += out[k]
        sig_w[g] += na * sig
        nacc[g] += na
        nev[g] += ne
    acc.update(sigma=sig_w / np.maximum(nacc, 1), n_events=nev,
               pthat_bins=np.asarray(edges, dtype=float))
    return acc


def densities(acc, windows=None):
    """Weighted mean per leading-jet bin (events of each window weighted sigma_k / N_k) and
    its error (ratio estimator), as densities dN/(deta dphi) (sumpt: dpT/(deta dphi))."""
    k = np.arange(len(acc["sigma"])) if windows is None else np.asarray(windows)
    s = (acc["sigma"][k] / np.maximum(acc["n_events"][k], 1))[:, None]
    W = (s * acc["n"][k]).sum(0)
    area_reg = 2 * TRK_ETA * 2 * np.pi / 3          # every region spans 120 deg of azimuth
    area_side = 2 * TRK_ETA * np.pi / 3
    res = {"weight": W}
    for q in QUANT:
        S1, S2, n = (s * acc[q][k]), (s ** 2 * acc[q + "_2"][k]), acc["n"][k]
        m = np.divide(S1.sum(0), W, out=np.full(W.shape, np.nan), where=W > 0)
        # sum over events of w^2 (x - m)^2 per window: S2 - 2 m s S1 + m^2 s^2 n
        var = (S2 - 2 * m * s * S1 + m ** 2 * s ** 2 * n).sum(0)
        err = np.divide(np.sqrt(np.clip(var, 0, None)), W, out=np.full(W.shape, np.nan),
                        where=W > 0)
        area = area_side if q in ("n_transmax", "n_transmin") else area_reg
        res[q] = (m / area, err / area)
    res["meanpt_transverse"] = (res["sumpt_transverse"][0] / res["n_transverse"][0], None)
    res["n_events"] = acc["n"][k].sum(0)
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+")
    p.add_argument("-j", type=int, default=8)
    p.add_argument("--check-truncation", action="store_true",
                   help="also the share of the lowest pTHat window (run 'trunc')")
    a = p.parse_args(argv)
    star = load_star()
    (C.HERE / "results").mkdir(exist_ok=True)
    lo, hi = LEAD_EDGES[:-1], LEAD_EDGES[1:]
    print("STAR transverse dNch/detadphi (pT > 0.2): "
          + " ".join(f"{v:.3f}" for v in star["transverse"]["value"]))
    summary = {}
    for run in a.runs:
        acc = run_sums(run, a.j)
        np.savez(C.HERE / "results" / f"ue_{run}.npz", **acc)
        d = densities(acc)
        print(f"\n{run}: leading jet [GeV]   events   toward  transverse (STAR)   away   "
              "trans pT>0.5 (STAR)  TransMax TransMin")
        rows = []
        for i in range(len(lo)):
            t, te = d["n_transverse"][0][i], d["n_transverse"][1][i]
            print(f"  {lo[i]:4.0f}-{hi[i]:<4.0f} {int(d['n_events'][i]):9d}   "
                  f"{d['n_toward'][0][i]:.3f}   {t:.3f}+-{te:.3f} ({star['transverse']['value'][i]:.3f})"
                  f"   {d['n_away'][0][i]:.3f}   {d['n_trans05'][0][i]:.3f} "
                  f"({star['transverse05']['value'][i]:.3f})   {d['n_transmax'][0][i]:.3f}  "
                  f"{d['n_transmin'][0][i]:.3f}")
            rows.append({k: float(d[k][0][i]) for k in
                         ("n_toward", "n_transverse", "n_away", "n_trans05", "n_transmax",
                          "n_transmin", "meanpt_transverse")}
                        | {"n_transverse_err": float(te), "events": int(d["n_events"][i])})
        summary[run] = rows
        if a.check_truncation:
            low = [i for i, w in enumerate(acc["pthat_bins"].tolist()) if w[0] < 5]
            rest = [i for i in range(len(acc["sigma"])) if i not in low]
            if low:
                W_all = densities(acc)["weight"]
                W_rest = densities(acc, rest)["weight"]
                share = 1 - W_rest / np.where(W_all > 0, W_all, np.nan)
                print("  share of the leading-jet bin from pTHat < 5 GeV: "
                      + " ".join(f"{100 * x:.1f}%" for x in share))
                summary[run + "_low_share"] = share.tolist()
    (C.HERE / "results" / "ue_check.json").write_text(json.dumps(
        {"star": {k: (v if k == "edges" else {kk: np.asarray(vv).tolist()
                                             for kk, vv in v.items()})
                  for k, v in star.items()}, "runs": summary}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
