#!/usr/bin/env python3
"""
analysis/OO-RHIC/chain_diag.py -- where between the partons and the hadrons the Au+Au samples differ
(auau_jet_raa.ipynb, "Where B1's under-quenching comes from"):

    python chain_diag.py -j 6          # -> results/chain_diag.json

Per sample and pTHat window, per event (|eta| < 1 where noted):
- e_medium: the energy given to the medium, sum E(recoils, pstat 1) - sum E(holes, pstat -1) [GeV];
- holes: holes per event; ini_pt5 / ini_pt10: shower initiators with pT > 5 / 10 GeV (all showers);
- par_pt10 / par_pt15: final partons (pstat >= 0, |eta| < 1) above 10 / 15 GeV;
- had_pt8 / had_pt12: charged hadrons (pstat >= 0, |eta| < 1) above 8 / 12 GeV; neg_pt8: hole hadrons
  (pstat -1, Colorless) above 8 GeV;
- had_per_par = had_pt12 / par_pt15: high-p_T hadrons per high-p_T parton, the hadronization's yield.
T1 (GB10, 2026-11-26): B1's Au+Au job with --matter-only (no LBT, no recoils, no holes, no closure), windows
18-23 / 23-30 / 30-40 GeV, 500 events each, /data/AuAu-RHIC_phys/hadtests; T2 the same with --colour-closure
thermal (GB10); T3 with --colour-closure fake (3090s, oo_bias.py hadtest_t3, /data/AuAu-RHIC_phys/auau_fake_w*).
Parton counts compare only within one shower scheme: the A+A scheme (Matter in_vac 0) showers to QS
0.9 GeV, pp_ref's vacuum scheme (in_vac 1) to Q0 2 GeV, PP19 to Q0 1 GeV.  Finished jobs only (.json).
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

R = "/data/AuAu-RHIC_phys"
SAMPLES = {
    "B1 (our chain, Type 2, iColored)": f"{R}/auau_med_w*/*_hadrons_jet_frag.h5",
    "B1 rehadronized with Colorless": f"{R}/rehad_colorless/auau_med_w*/*_hadrons_jet_frag.h5",
    "B2 (our chain, Type 3, iColored)": f"{R}/auau_t3_w*/*_hadrons_jet_frag.h5",
    "B2 rehadronized with Colorless": f"{R}/rehad_colorless/auau_t3_w*/*_hadrons_jet_frag.h5",
    "T1 (our chain, MATTER only, iColored)": f"{R}/hadtests/auau_matteronly_w*/*_hadrons_jet_frag.h5",
    "T1 rehadronized with Colorless": f"{R}/hadtests/rehad/auau_matteronly_w*/*_hadrons_jet_frag.h5",
    "T2 (our chain, closure thermal, iColored)": f"{R}/hadtests/auau_thermal_w*/*_hadrons_jet_frag.h5",
    "T2 rehadronized with Colorless": f"{R}/hadtests/rehad/auau_thermal_w*/*_hadrons_jet_frag.h5",
    "T3 (our chain, closure fake, iColored)": f"{R}/auau_fake_w*/*_hadrons_jet_frag.h5",
    "T3 rehadronized with Colorless": f"{R}/hadtests/rehad/auau_fake_w*/*_hadrons_jet_frag.h5",
    "run C (JETSCAPE chain, Type 3, Colorless)": f"{R}/js_chain_t3/*_hadrons_jet_frag.h5",
    "pp_ref (iColored, vacuum)": "/data/OO-RHIC_phys/pp_ref_w*/*_hadrons_jet_frag.h5",
    "pp_ref rehadronized with Colorless": f"{R}/rehad_colorless/pp_ref_w*/*_hadrons_jet_frag.h5",
    "PP19 (Colorless, vacuum)": "/data/OO-RHIC_pp_refs/pp19_icdecays/*_hadrons_jet_frag.h5",
}
KEYS = ("n", "e_medium", "holes", "ini_pt5", "ini_pt10", "par_pt10", "par_pt15", "had_pt8", "had_pt12", "neg_pt8")


def one(path):
    import h5py
    import hdf5plugin  # noqa: F401
    with h5py.File(path) as h:
        n = int(h.attrs["nunits_written"])
        bins = [tuple(map(float, w)) for w in np.asarray(h.attrs["pthat_bins"]).reshape(-1, 2)]
        win = h["units/pthat_bin"][:n].astype(int)
        po = h["partons/offsets"][: n + 1]; par = h["partons/data"][: po[-1]]
        io = h["initiators/offsets"][: n + 1]; ini = h["initiators/data"][: io[-1]]
        pa, ca, pid = B._units(h, n, charged=False, with_pid=True)
        so, uo = h["hadrons/sample_offsets"][:], h["hadrons/unit_offsets"][: n + 1]
        pst = h["hadrons/pstat"][: int(so[uo[-1]])] if "hadrons/pstat" in h else np.zeros(len(pa), int)
    evp = np.repeat(np.arange(n), np.diff(po))
    evi = np.repeat(np.arange(n), np.diff(io))
    evh = np.repeat(np.arange(n), ca)
    ppt = np.hypot(par[:, 4], par[:, 5]); peta = np.arcsinh(par[:, 6] / np.maximum(ppt, 1e-9))
    ipt = np.hypot(ini[:, 3], ini[:, 4])                      # initiators: shower pid pstat px py pz E x y z t
    hpt, heta, _ = A._kin(pa)
    ch = np.isin(np.abs(pid), B.charged_ids()) & (np.abs(heta) < 1)
    good = (par[:, 2] >= 0) & (np.abs(peta) < 1)
    def cnt(ev, m):
        return np.bincount(ev[m], minlength=n).astype(float)
    per = {"e_medium": np.bincount(evp, np.where(par[:, 2] == 1, par[:, 3], 0.0) - np.where(par[:, 2] == -1, par[:, 3], 0.0), minlength=n),
           "holes": cnt(evp, par[:, 2] == -1), "ini_pt5": cnt(evi, ipt > 5), "ini_pt10": cnt(evi, ipt > 10),
           "par_pt10": cnt(evp, good & (ppt > 10)), "par_pt15": cnt(evp, good & (ppt > 15)),
           "had_pt8": cnt(evh, ch & (pst >= 0) & (hpt > 8)), "had_pt12": cnt(evh, ch & (pst >= 0) & (hpt > 12)),
           "neg_pt8": cnt(evh, ch & (pst == -1) & (hpt > 8))}
    out = {}
    for k, b in enumerate(bins):
        m = win == k
        if m.any():
            out[f"{b[0]:g}-{b[1]:g}"] = {"n": int(m.sum()), **{q: float(v[m].sum()) for q, v in per.items()}}
    return out


def collect(pattern, j):
    fs = [f for f in sorted(glob.glob(pattern)) if os.path.exists(f.replace("_hadrons_jet_frag.h5", ".json"))]
    if not fs:
        return None
    with ProcessPoolExecutor(max_workers=j) as ex:
        res = list(ex.map(one, fs))
    tot = {}
    for r in res:
        for w, d in r.items():
            t = tot.setdefault(w, {k: 0.0 for k in KEYS})
            for k in KEYS:
                t[k] += d[k]
    out = {}
    for w, t in tot.items():
        n = t["n"]
        d = {"events": int(n), **{k: t[k] / n for k in KEYS if k != "n"}}
        d["had_per_par"] = d["had_pt12"] / d["par_pt15"] if d["par_pt15"] > 0 else float("nan")
        out[w] = d
    return {"files": len(fs), "windows": dict(sorted(out.items(), key=lambda kv: float(kv[0].split("-")[0])))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-j", type=int, default=6)
    a = ap.parse_args(argv)
    res = {}
    for name, pat in SAMPLES.items():
        r = collect(pat, a.j)
        if r is not None:
            res[name] = {"glob": pat, **r}
            print(f"{name}: {r['files']} files")
    (HERE / "results" / "chain_diag.json").write_text(json.dumps(res, indent=1))
    print("->", HERE / "results" / "chain_diag.json")


if __name__ == "__main__":
    sys.exit(main())
