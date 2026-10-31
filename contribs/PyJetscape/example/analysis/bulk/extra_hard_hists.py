#!/usr/bin/env python3
"""
extra_hard_hists.py -- per-event histograms of the Pb+Pb 0-10 % extras campaign
(~/prod/PbPb_0_10_v2x) for extra_hard_bulk_jets.ipynb and extra_hard_bulk_jets_hist.ipynb
(this directory); the output goes to the campaign's git-ignored run folder,
../runs/pbpb_extra_hard/data/.

One pass over every event and every oversample k, three compositions per sample:

    bg     bulk_bg, sample k of the event's background          (bulk, no extras)
    jet    bulk_jet, sample k of the jet leg                    (bulk + the extras' wakes)
    full   bulk_jet + extra_frag (fragmentation k % n_frag)     (+ the extras' own hadrons)

Per event, averaged over its samples (so every event weighs the same):

    eta     charged, pT > 0.15 GeV, dN/deta in |eta| < 2.6 (the files' eta_max)
    phi     charged, pT > 0.15 GeV, |eta| < 2.4, dN/dphi in the lab
    dphi2   the same against the background's Psi_2 (phi - Psi_2 mod pi; Psi_2 from all
            bulk_bg samples of the event, charged 0.2 < pT < 3 GeV, |eta| < 2.4)
    pt      charged, |eta| < 0.8, dN/dpT
    jets    anti-kt R = 0.4, unsubtracted, constituents pT > 0.15 GeV and |eta| < 2.4, jets
            |eta| < 2.0 (inclusive dN/dpT and the leading jet), charged-particle jets
            ("ch") and jets of all hadrons ("all")

and the extras: N_coll, the scatterings (pTHat, scalar pT sum of their partons) and the
initiators (shower-initiating partons: pT, eta, scatter).  Errors are left to the
notebook (jackknife over the hydro backgrounds, ``group``).

    python extra_hard_hists.py [--prod DIR] [--out ../runs/pbpb_extra_hard/data/extra_hard_hists.npz]
                               [--workers 12]
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PYJ = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(PYJ, "python"))

from jetscape.hadrons_h5 import CHARGED, HadronFileReader  # noqa: E402

PROD = os.path.expanduser("~/prod/PbPb_0_10_v2x")
RUN = os.path.join(HERE, "..", "runs", "pbpb_extra_hard")      # git-ignored results
COMPS = ("bg", "jet", "full")
JET_KINDS = ("ch", "all")

EDGES = {
    "eta": np.linspace(-2.6, 2.6, 53),
    "phi": np.linspace(-np.pi, np.pi, 37),
    "dphi2": np.linspace(0.0, np.pi, 37),
    "pt": np.concatenate([np.arange(0.15, 3.0, 0.1), np.arange(3.0, 8.0, 0.5),
                          np.arange(8.0, 20.0, 2.0), [20.0, 30.0, 50.0]]),
    "jet": np.arange(0.0, 402.0, 6.0),
}
PT_MIN, ETA_CONST, ETA_JET, R = 0.15, 2.4, 2.0, 0.4


def kin(p):
    pt = np.hypot(p[:, 1], p[:, 2])
    pabs = np.sqrt(pt ** 2 + p[:, 3] ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = 0.5 * np.log((pabs + p[:, 3]) / (pabs - p[:, 3]))
    return pt, eta, np.arctan2(p[:, 2], p[:, 1])


def psi2(samples):
    """Psi_2 of the background from all its samples together."""
    q = 0j
    for ev in samples:
        pt, eta, phi = kin(ev["p"])
        m = (np.isin(np.abs(ev["pid"]), CHARGED) & (pt > 0.2) & (pt < 3.0)
             & (np.abs(eta) < ETA_CONST))
        q += np.exp(2j * phi[m]).sum()
    return 0.5 * np.angle(q)


def cluster(evs, charged_only):
    """Inclusive and leading jet pT (|eta_jet| < ETA_JET) of each sample, vectorized."""
    import awkward as ak
    import fastjet as fj
    ps = []
    for ev in evs:
        p = ev["p"].astype(np.float64)
        pt, eta, _ = kin(p)
        m = (pt > PT_MIN) & (np.abs(eta) < ETA_CONST) & np.isfinite(eta)
        if charged_only:
            m &= np.isin(np.abs(ev["pid"]), CHARGED)
        ps.append(p[m])
    n = [len(p) for p in ps]
    p = np.concatenate(ps)
    arr = ak.unflatten(ak.zip({"px": p[:, 1], "py": p[:, 2], "pz": p[:, 3], "E": p[:, 0]}), n)
    jets = fj.ClusterSequence(arr, fj.JetDefinition(fj.antikt_algorithm, R)).inclusive_jets(
        min_pt=EDGES["jet"][0] + 1.0)
    out = []
    for j in ak.to_list(jets):
        jp = np.array([[x["px"], x["py"], x["pz"], x["E"]] for x in j]).reshape(-1, 4)
        jpt = np.hypot(jp[:, 0], jp[:, 1])
        pabs = np.linalg.norm(jp[:, :3], axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            jeta = 0.5 * np.log((pabs + jp[:, 2]) / (pabs - jp[:, 2]))
        out.append(jpt[np.abs(jeta) < ETA_JET])
    return out


def hist_samples(values_per_sample, edges):
    """Mean over samples of the per-sample histograms."""
    h = np.zeros(len(edges) - 1)
    for v in values_per_sample:
        h += np.histogram(v, edges)[0]
    return h / max(len(values_per_sample), 1)


def one_file(stem_glob):
    import warnings
    warnings.filterwarnings("ignore")
    r = HadronFileReader(stem_glob)
    rows = []
    for g in r.events():
        info = r.event_info(g)
        K = r.n_oversamples(g)
        evs = {"bg": [r.background_event(g, k) for k in range(K)],
               "jet": [r.jet_event(g, k, extras=False) for k in range(K)],
               "full": [r.jet_event(g, k) for k in range(K)]}
        ps2 = psi2(evs["bg"])
        row = {"seed": info.seed, "local": info.local_event, "bg_unit": info.bg_unit,
               "n_coll": info.n_coll, "n_samples": K, "psi2": ps2}
        for c in COMPS:
            v = {"eta": [], "phi": [], "dphi2": [], "pt": []}
            for ev in evs[c]:
                pt, eta, phi = kin(ev["p"])
                ch = np.isin(np.abs(ev["pid"]), CHARGED) & (pt > PT_MIN) & np.isfinite(eta)
                v["eta"].append(eta[ch])
                m = ch & (np.abs(eta) < ETA_CONST)
                v["phi"].append(phi[m])
                v["dphi2"].append(np.mod(phi[m] - ps2, np.pi))
                v["pt"].append(pt[ch & (np.abs(eta) < 0.8)])
            for q in v:
                row[f"{c}.{q}"] = hist_samples(v[q], EDGES[q])
            for kind in JET_KINDS:
                jpt = cluster(evs[c], kind == "ch")
                row[f"{c}.jet_{kind}"] = hist_samples(jpt, EDGES["jet"])
                lead = [j.max(keepdims=True) if len(j) else j for j in jpt]
                row[f"{c}.lead_{kind}"] = hist_samples(lead, EDGES["jet"])
        sc = info.scatterings()
        sc = sc[sc[:, 0] > 0] if sc is not None else np.zeros((0, 8))
        row["n_extra"] = len(sc)
        row["extra_pthat"] = sc[:, 1].copy()
        row["extra_ptsum"] = sc[:, 7].copy()
        hf = r._jet_events(info.file_index).extra_frag
        ini = hf.initiators(info.local_event)
        row["n_initiators"] = len(ini)
        ipt = np.hypot(ini[:, 3], ini[:, 4])
        ip = np.sqrt(ipt ** 2 + ini[:, 5] ** 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            row["ini_eta"] = 0.5 * np.log((ip + ini[:, 5]) / (ip - ini[:, 5]))
        row["ini_pt"] = ipt
        rows.append(row)
    r.close()
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--prod", default=PROD)
    ap.add_argument("--out", default=os.path.join(RUN, "data", "extra_hard_hists.npz"))
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    stems = sorted(glob.glob(os.path.join(a.prod, "*_hadrons_bulk_jet.h5")))
    t0 = time.time()
    rows = []
    with Pool(a.workers) as pool:
        for i, rr in enumerate(pool.imap(one_file, stems)):
            rows += rr
            print(f"  {i + 1}/{len(stems)} files, {len(rows)} events, "
                  f"{time.time() - t0:.0f} s", flush=True)
    out = {f"edges_{k}": v for k, v in EDGES.items()}
    for key in ("seed", "local", "bg_unit", "n_coll", "n_samples", "psi2", "n_extra",
                "n_initiators"):
        out[key] = np.array([r[key] for r in rows])
    for key in rows[0]:
        if "." in key:
            out[key] = np.stack([r[key] for r in rows])
    for key in ("extra_pthat", "extra_ptsum", "ini_pt", "ini_eta"):
        out[key] = np.concatenate([r[key] for r in rows])
    out["ini_event"] = np.repeat(np.arange(len(rows)), [r["n_initiators"] for r in rows])
    out["extra_event"] = np.repeat(np.arange(len(rows)), [r["n_extra"] for r in rows])
    out["group"] = out["seed"] * 1000 + out["bg_unit"]
    np.savez_compressed(a.out, **out)
    print(f"wrote {a.out}: {len(rows)} events in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
