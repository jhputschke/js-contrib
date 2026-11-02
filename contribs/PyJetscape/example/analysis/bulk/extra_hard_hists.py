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

v2(pT) of charged hadrons in the bins ``edges_v2`` (ALICE's up to 17 GeV, then CMS-like to
100 GeV), per composition, as sums the notebooks combine over events (means over the event's
samples, like the histograms; the estimators are ratios of the sums over all events):

    gap1    v2{2, |Delta eta| > 1}, ALICE's (arXiv:1804.02944): POIs and reference particles
            (charged, 0.2 < pT < 5 GeV) in |eta| < 0.8, subevents eta < -0.5 and eta > 0.5;
            v2(pT) = <<2'>> / sqrt(<<2>>), <<2'>> = sum Re(p_A Q_B* + p_B Q_A*) / sum (m_A M_B
            + m_B M_A), <<2>> = sum Re(Q_A Q_B*) / sum M_A M_B
    gap15   the same with POIs in |eta| < 1 (-1 < eta < 0 against reference particles at
            1.5 < eta < 2.6, 0 < eta < 1 against -2.6 < eta < -1.5): |Delta eta| > 1.5, as
            near to CMS's scalar product with the HF (|Delta eta| > 3) as |eta| < 2.6 allows
    psi     <cos 2(phi - Psi_2,bg)> of the POIs in |eta| < 0.8, Psi_2 of the background from
            all its samples (charged, 0.2 < pT < 3 GeV, 1 < |eta| < 2.6): no non-flow of the
            extras; ``psi2_res`` = cos 2(Psi_A - Psi_B) of its two halves (eta < -1, eta > 1)
            per event, for the resolution

Keys: ``<c>.v2_<m>_num``, ``<c>.v2_<m>_den`` (per bin), ``<c>.v2_<m>_rnum``,
``<c>.v2_<m>_rden`` (the reference, gap1 and gap15).

    python extra_hard_hists.py [--prod DIR] [--out ../runs/pbpb_extra_hard/data/extra_hard_hists.npz]
                               [--workers 12] [--chunks-per-worker 4]

The events go to the processes in contiguous ranges within a file (about --chunks-per-worker
ranges per process), so the cores stay busy to the end; the output is the same, row for
row, as with one range per file.
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

#: v2(pT) bins: ALICE's (1804.02944, 5-10 %) to 17 GeV, then CMS-like (1702.00630) to 100 GeV
EDGES["v2"] = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0,
                        2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0,
                        8.0, 9.0, 10.0, 12.0, 14.0, 17.0, 20.0, 25.0, 35.0, 45.0, 60.0, 80.0,
                        100.0])
#: the two-particle methods: (POI A, ref B), (POI B, ref A) as eta ranges, and the reference
#: subevents A, B; reference particles charged with REF_PT[0] < pT < REF_PT[1]
V2_METHODS = {
    "gap1":  dict(poi_a=(-0.8, -0.5), poi_b=(0.5, 0.8), ref_a=(-0.8, -0.5), ref_b=(0.5, 0.8)),
    "gap15": dict(poi_a=(-1.0, 0.0), poi_b=(0.0, 1.0), ref_a=(-2.6, -1.5), ref_b=(1.5, 2.6)),
}
REF_PT = (0.2, 5.0)
PSI_POI_ETA, PSI_ETA = 0.8, (1.0, 2.6)


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


def psi2_plane(samples):
    """Psi_2 of the background from all its samples at 1 < |eta| < 2.6 (charged, 0.2 < pT <
    3 GeV), and cos 2(Psi_A - Psi_B) of its halves eta < -1 and eta > 1 (the resolution)."""
    qa = qb = 0j
    for ev in samples:
        pt, eta, phi = kin(ev["p"])
        m = np.isin(np.abs(ev["pid"]), CHARGED) & (pt > 0.2) & (pt < 3.0)
        e2 = np.exp(2j * phi)
        qa += e2[m & (eta < -PSI_ETA[0]) & (eta > -PSI_ETA[1])].sum()
        qb += e2[m & (eta > PSI_ETA[0]) & (eta < PSI_ETA[1])].sum()
    return 0.5 * np.angle(qa + qb), float(np.cos(np.angle(qa) - np.angle(qb)))


def v2_sums(pt, eta, phi, ch, psi):
    """One sample's v2(pT) sums of every method (see the module docstring)."""
    e = EDGES["v2"]
    nb = len(e) - 1
    ib = np.searchsorted(e, pt, side="right") - 1
    inb = ch & (ib >= 0) & (ib < nb)
    q2 = np.exp(2j * phi)
    ref = ch & (pt > REF_PT[0]) & (pt < REF_PT[1])
    out = {}

    def within(r):
        return (eta > r[0]) & (eta < r[1])

    def binned(m, w):
        return np.bincount(ib[m], weights=w[m], minlength=nb)

    for name, mth in V2_METHODS.items():
        ra, rb = ref & within(mth["ref_a"]), ref & within(mth["ref_b"])
        Qa, Qb, Ma, Mb = q2[ra].sum(), q2[rb].sum(), ra.sum(), rb.sum()
        pa, pb = inb & within(mth["poi_a"]), inb & within(mth["poi_b"])
        num = (binned(pa, (q2 * np.conj(Qb)).real) + binned(pb, (q2 * np.conj(Qa)).real))
        den = (np.bincount(ib[pa], minlength=nb) * Mb + np.bincount(ib[pb], minlength=nb) * Ma)
        out[name] = (num, den.astype(float), float((Qa * np.conj(Qb)).real), float(Ma * Mb))
    m = inb & (np.abs(eta) < PSI_POI_ETA)
    out["psi"] = (binned(m, np.cos(2 * (phi - psi))), np.bincount(ib[m], minlength=nb).astype(float))
    return out


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


def one_file(task):
    """The rows of one event range: ``task`` = (production file, its local events) or just
    the file (all its events)."""
    import warnings
    warnings.filterwarnings("ignore")
    stem_glob, events = task if isinstance(task, tuple) else (task, None)
    r = HadronFileReader(stem_glob)
    rows = []
    for g in r.events(events):
        info = r.event_info(g)
        K = r.n_oversamples(g)
        evs = {"bg": [r.background_event(g, k) for k in range(K)],
               "jet": [r.jet_event(g, k, extras=False) for k in range(K)],
               "full": [r.jet_event(g, k) for k in range(K)]}
        ps2 = psi2(evs["bg"])
        psi_v2, res = psi2_plane(evs["bg"])
        row = {"seed": info.seed, "local": info.local_event, "bg_unit": info.bg_unit,
               "n_coll": info.n_coll, "n_samples": K, "psi2": ps2, "psi2_res": res}
        for c in COMPS:
            v = {"eta": [], "phi": [], "dphi2": [], "pt": []}
            v2 = {}
            for ev in evs[c]:
                pt, eta, phi = kin(ev["p"])
                ch = np.isin(np.abs(ev["pid"]), CHARGED) & (pt > PT_MIN) & np.isfinite(eta)
                for mth, sums in v2_sums(pt, eta, phi, ch, psi_v2).items():
                    v2[mth] = sums if mth not in v2 else tuple(a + b for a, b in
                                                               zip(v2[mth], sums))
                v["eta"].append(eta[ch])
                m = ch & (np.abs(eta) < ETA_CONST)
                v["phi"].append(phi[m])
                v["dphi2"].append(np.mod(phi[m] - ps2, np.pi))
                v["pt"].append(pt[ch & (np.abs(eta) < 0.8)])
            for q in v:
                row[f"{c}.{q}"] = hist_samples(v[q], EDGES[q])
            for mth, sums in v2.items():                  # means over the samples
                for nm, x in zip(("num", "den", "rnum", "rden"), sums):
                    row[f"{c}.v2_{mth}_{nm}"] = np.asarray(x, float) / max(K, 1)
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


def event_ranges(stems, workers, per_worker=4):
    """(file, local events) tasks: each file's events in contiguous ranges of about
    total / (workers * per_worker) events, in file order."""
    import warnings
    counts = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for st in stems:
            r = HadronFileReader(st)
            counts.append(r.n_events)
            r.close()
    size = max(1, int(np.ceil(sum(counts) / max(1, workers * per_worker))))
    tasks = []
    for st, n in zip(stems, counts):
        for part in np.array_split(np.arange(n), max(1, int(np.ceil(n / size)))):
            if len(part):
                tasks.append((st, part))
    return tasks


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--prod", default=PROD)
    ap.add_argument("--out", default=os.path.join(RUN, "data", "extra_hard_hists.npz"))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--chunks-per-worker", type=int, default=4,
                    help="event ranges per process (1 per file: 0)")
    a = ap.parse_args()
    stems = sorted(glob.glob(os.path.join(a.prod, "*_hadrons_bulk_jet.h5")))
    t0 = time.time()
    tasks = (event_ranges(stems, a.workers, a.chunks_per_worker) if a.chunks_per_worker > 0
             else stems)
    rows = []
    with Pool(a.workers) as pool:
        for i, rr in enumerate(pool.imap(one_file, tasks)):
            rows += rr
            print(f"  {i + 1}/{len(tasks)} ranges of {len(stems)} files, {len(rows)} events, "
                  f"{time.time() - t0:.0f} s", flush=True)
    out = {f"edges_{k}": v for k, v in EDGES.items()}
    for key in ("seed", "local", "bg_unit", "n_coll", "n_samples", "psi2", "psi2_res",
                "n_extra", "n_initiators"):
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
