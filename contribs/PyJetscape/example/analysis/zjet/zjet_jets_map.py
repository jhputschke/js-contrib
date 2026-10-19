#!/usr/bin/env python3
"""The map step of the Z+jet analysis (CMS 1702.01060) in Python: the jets of every oversample
of the Z events of a production directory, one ``<stem>_jets.root`` per production file.
The same algorithm and output as the C++ map step, example/analysis_root/zjet/zjets.cxx
(its header has the definitions); check_cxx.py jets compares the two, zjet_jets.py is the
final stage of both.

    python zjet_jets_map.py <h5 dir or quoted glob> <out dir> [--R 0.3] [--seed 1]
                            [--samples N] [--emb-samples N] [--cones 4] [--ptmin 5]
                            [--qa-samples 1] [--files i:j]

--files i:j maps only production files i .. j-1 (in the reader's order), for several processes
at once.  Every output stands alone: its ``file_info`` tree (the production file's name,
events, hydro backgrounds, pTHat windows: ``jetcorr.mapio``) lets zjet_jets.load_jets rebuild
the event and background ids and weight_mb of any set of outputs, from one or several
campaigns (``jetcorr.weights.campaign_ids``, ``combine_windows``); the stored columns event,
file, bg and weight_mb are those of the mapped directory (for older readers).  ``run_info``:
the settings (R, ghost area, --ptmin, --cones, --seed, ...).

Jets (``jetcorr.jetmap``): anti-k_T R = --R (0.3) from all final particles but neutrinos and
the Z's leptons; Pb+Pb with explicit ghosts on a fixed grid and rho(jet) from the k_T R = 0.4 median in a strip
|y - y_jet| < 1 (|y| < 2.2), p_T = p_T,raw - rho A; the fragments embedded in bulk_bg (the
matched reference jets within R / 2, and all its jets above --ptmin as `nw`: the same event
without the wake, the final stage's "no wake" variant); random cones of radius R in bulk_bg;
the Z cone (zcone_*: the p_T sum within R of (y_Z, phi_Z), rho there and sum - rho pi R^2, in
the full event and in the event without the wake, zcone_nw_*: jet1's p_T variant of the
dijet proxy, PLAN_dijet_deta_wake A2); the jets of every background's bulk_bg samples (tree bg_jets, the final
stage's B); per jet also its mass, the number of its (non-ghost) constituents and the leading
one's p_T.  The jet QA (trees qa_jets, qa_bg_jets): the first --qa-samples samples of every Z
event and background with all their jets (p_T,raw > 1 GeV, no cut on the subtracted p_T), for
zjet_jets_qa.py.  --emb-samples N: the embedded event (the fragments alone and in bulk_bg: emb,
nw, zcone_nw) only for the first N oversamples, about half the time of the others; the jets,
the Z cone and the random cones of every oversample are unchanged (default: all).  Slow in
Python (~0.2 s per Pb+Pb oversample): use --samples for checks.
Exact shortcuts, the outputs unchanged: rho(jet) once per strip content (StripRho), no rho for
jets that cannot pass the p_T cut, the background's particles and estimator kept per hydro
background and sample, the random cones' particles preselected in numpy."""
import argparse
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                        # jetcorr
try:
    import jetscape  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "python"))
import zjet_wake as zw  # noqa: E402
from jetcorr import mapio  # noqa: E402
from jetcorr import weights as jw  # noqa: E402
from jetcorr.jetmap import (ALL, GHOST_AREA, JET_NAMES, R_JET, Clusterer, cone,  # noqa: E402
                            rap_phi)

PT_REF, ETA_RC = 10.0, 1.6
MAP_VERSION = 2                     # 2: file_info, run_info, bg_unit, zcone_*, --R
RC_SALT = 0x5243000000000000
NU = (12, 14, 16)


def keep_mask(pid, p, info):
    """All but neutrinos and the Z's two leptons (zjet_wake.not_z_lepton's match)."""
    m = ~np.isin(np.abs(pid), NU)
    z = info.z
    if z:
        p3 = p[:, 1:4].astype(np.float64)
        for i in (1, 2):
            lp = zw.lepton_p3(z[f"lep{i}_pt"], z[f"lep{i}_eta"], z[f"lep{i}_phi"])
            tol = 1e-3 * max(float(np.linalg.norm(lp)), 1.0)
            m &= ~((pid == int(z[f"lep{i}_pid"])) & (np.linalg.norm(p3 - lp, axis=1) < tol))
    return m


def pseudojets(fj, ev, info):
    p, pid = np.asarray(ev["p"]), np.asarray(ev["pid"])
    m = keep_mask(pid, p, info)
    return [fj.PseudoJet(float(q[1]), float(q[2]), float(q[3]), float(q[0])) for q in p[m]]


def with_rap_phi(fj, ev, info):
    """pseudojets() and the kept particles' rapidities and phis (for the cones)."""
    p, pid = np.asarray(ev["p"]), np.asarray(ev["pid"])
    m = keep_mask(pid, p, info)
    parts = [fj.PseudoJet(float(q[1]), float(q[2]), float(q[3]), float(q[0])) for q in p[m]]
    return (parts,) + rap_phi(ev, m)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("h5")
    ap.add_argument("out")
    ap.add_argument("--R", type=float, default=R_JET, help="the anti-k_T radius (0.3)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--samples", type=int, default=1 << 30)
    ap.add_argument("--emb-samples", type=int, default=1 << 30,
                    help="the embedded event (emb, nw, zcone_nw) only for the first N oversamples")
    ap.add_argument("--cones", type=int, default=4)
    ap.add_argument("--ptmin", type=float, default=5.0)
    ap.add_argument("--qa-samples", type=int, default=1,
                    help="the first N samples' jets without the p_T cut "
                         "(trees qa_jets, qa_bg_jets)")
    ap.add_argument("--files", default=None, help="i:j, only production files i .. j-1")
    a = ap.parse_args()
    import awkward as ak
    import fastjet as fj
    import uproot
    from jetscape.hadrons_h5 import EventHadrons, HadronFileReader, Hadrons

    r = HadronFileReader(a.h5)
    infos = [r.event_info(g) for g in range(r.n_events)]
    bulk = "bulk_jet" in r.tags()
    pbin = np.array([-1 if i.pthat_bin is None else i.pthat_bin for i in infos])
    nk = np.bincount(np.clip(pbin, 0, None)[pbin >= 0], minlength=1)
    sig = ([r.pthat_bin_sigma(k)[0] for k in range(len(r.pthat_bins))]
           if r.pthat_bins is not None else [])
    loc = np.array([(i.file_index, i.bg_unit if bulk else i.local_event) for i in infos])
    _, bgid = np.unique(loc, axis=0, return_inverse=True)
    bgid = bgid.ravel()
    c = Clusterer(fj, a.R)
    dr_match = c.R / 2
    run_info = {"R": c.R, "ghost_area": GHOST_AREA, "ptmin": a.ptmin, "cones": a.cones,
                "seed": a.seed, "qa_samples": a.qa_samples, "samples": a.samples,
                "emb_samples": a.emb_samples, "version": MAP_VERSION}
    os.makedirs(a.out, exist_ok=True)
    lo, hi = 0, len(r.stems)
    if a.files:
        i, _, j = a.files.partition(":")
        lo, hi = max(0, int(i)), min(hi, int(j) if j else int(i) + 1)
    for fi, stem in enumerate(r.stems):
        if not lo <= fi < hi:
            continue
        t0 = time.time()
        name = os.path.basename(stem)
        h = {t: Hadrons.from_h5(f"{stem}_hadrons_{t}.h5")
             for t in (("bulk_jet", "bulk_bg", "jet_frag") if bulk else ("jet_frag",))}

        def sample(tag, unit, k):
            ev = EventHadrons(h[tag], h[tag].unit_index(unit))
            s = ev.sample == k
            return {"pid": ev.pid[s], "p": ev.p[s]}

        evs = {k: [] for k in ("event", "local_event", "file", "bg", "bg_unit", "pthat_bin",
                               "weight_mb", "n_coll", "n_samples", "z_pt", "z_y", "z_phi", "z_m",
                               "lep1_pid", "lep1_pt", "lep1_eta", "lep1_phi", "lep2_pid",
                               "lep2_pt", "lep2_eta", "lep2_phi")}
        rows = {k: [] for k in ("event", "sample", "frag_sample")}
        jet, nw, emb, rc = [], [], [], []
        zcone = {k: [] for k in ("zcone_sum", "zcone_rho", "zcone_dpt", "zcone_nw_sum",
                                 "zcone_nw_rho", "zcone_nw_dpt")}
        qa_rows, qa_jet = {"event": [], "sample": []}, []
        bg_cache = {}            # (bg_unit, k) -> the background's particles, rho, rap, phi

        def background(info, k):
            """bulk_bg sample k of the event's background: its particles (as pseudojets(),
            the event's Z leptons removed), their StripRho, rapidities and phis; shared by the
            background's events (one background at a time) unless a Z lepton is among them."""
            ev = sample("bulk_bg", info.bg_unit, k)
            pid, p = np.asarray(ev["pid"]), np.asarray(ev["p"])
            m = keep_mask(pid, p, info)
            shared = np.array_equal(m, ~np.isin(np.abs(pid), NU))
            key = (info.bg_unit, k)
            if shared and key in bg_cache:
                return bg_cache[key]
            bp = pseudojets(fj, ev, info)
            out = (bp, c.bge(bp)) + rap_phi(ev, m)
            if shared:
                if bg_cache and next(iter(bg_cache))[0] != info.bg_unit:
                    bg_cache.clear()
                bg_cache[key] = out
            return out
        for g in range(int(r._offsets[fi]), int(r._offsets[fi + 1])):
            info = infos[g]
            z = info.z
            if not z or not (60 < z["z_m"] < 120 and z["z_pt"] > 30):
                continue
            le = info.local_event
            nf = h["jet_frag"].n_samples(le)
            ns = h["bulk_jet"].n_samples(le) if bulk else nf
            if ns <= 0 or nf <= 0:
                continue
            n_used = min(ns, a.samples)
            k_ = info.pthat_bin if info.pthat_bin is not None else -1
            for key, v in (("event", g), ("local_event", le), ("file", fi), ("bg", bgid[g]),
                           ("bg_unit", info.bg_unit if bulk else le), ("pthat_bin", k_),
                           ("weight_mb", sig[k_] / nk[k_] if k_ >= 0 and sig else math.nan),
                           ("n_coll", info.n_coll if info.n_coll is not None else -1),
                           ("n_samples", n_used)):
                evs[key].append(v)
            for key in list(evs)[9:]:
                evs[key].append(z[key])
            key = zw.event_key(a.seed, name, le)
            for k in range(n_used):
                fk = k % nf if bulk else k
                rows["event"].append(g)
                rows["sample"].append(k)
                rows["frag_sample"].append(fk)
                frag = sample("jet_frag", le, fk)
                if bulk:
                    bj = sample("bulk_jet", le, k)
                    full = {"pid": np.concatenate([bj["pid"], frag["pid"]]),
                            "p": np.concatenate([bj["p"], frag["p"]])}
                else:
                    full = frag
                parts, py, pphi = with_rap_phi(fj, full, info)
                pb = c.bge(parts) if bulk else None
                every = c.jets(parts, bulk, ALL if k < a.qa_samples else a.ptmin, bge=pb)
                _, zs, zr, zd = cone(fj, parts, py, pphi, pb, z["z_y"], z["z_phi"], c.R)
                for col, v in (("zcone_sum", zs), ("zcone_rho", zr), ("zcone_dpt", zd)):
                    zcone[col].append(v)
                jet.append([x for x in every if (x[1] > a.ptmin if bulk else x[1] >= a.ptmin)])
                if k < a.qa_samples:
                    qa_rows["event"].append(g)
                    qa_rows["sample"].append(k)
                    qa_jet.append(every)
                if not bulk:
                    nw.append([])
                    emb.append([])
                    rc.append([])
                    for col in ("zcone_nw_sum", "zcone_nw_rho", "zcone_nw_dpt"):
                        zcone[col].append(math.nan)
                    continue
                bp, b, by, bphi = background(info, k)
                if k < a.emb_samples:
                    fp, fy, fphi = with_rap_phi(fj, frag, info)
                    ref = c.jets(fp, False, PT_REF)
                    nwp = bp + fp
                    nwb = c.bge(nwp)
                    ej = c.jets(nwp, True, 1.0, bge=nwb)
                    nw.append([x for x in ej if x[1] > a.ptmin])    # the event without the wake
                    _, zs, zr, zd = cone(fj, nwp, np.concatenate([by, fy]),
                                         np.concatenate([bphi, fphi]), nwb, z["z_y"],
                                         z["z_phi"], c.R)
                    pairs = []
                    for rj in ref:
                        best, dr_best = math.nan, math.inf
                        for x in ej:
                            dr = math.hypot(x[2] - rj[2], zw.wrap_dphi(x[3] - rj[3], -math.pi))
                            if dr < dr_best:
                                dr_best, best = dr, x[1]
                        pairs.append((rj[1], rj[2], rj[3],
                                      best if dr_best < dr_match else math.nan,
                                      dr_best if math.isfinite(dr_best) else math.nan))
                else:                                   # not embedded (--emb-samples)
                    nw.append([])
                    zs = zr = zd = math.nan
                    pairs = []
                for col, v in (("zcone_nw_sum", zs), ("zcone_nw_rho", zr),
                               ("zcone_nw_dpt", zd)):
                    zcone[col].append(v)
                emb.append(pairs)
                cones = []
                for q in range(a.cones):
                    s = RC_SALT | (k << 16) | q
                    eta = -ETA_RC + 2 * ETA_RC * zw.key_uniform(key, 2 * s)
                    phi = -math.pi + 2 * math.pi * zw.key_uniform(key, 2 * s + 1)
                    axis, tot, rho, dpt = cone(fj, bp, by, bphi, b, eta, phi, c.R)
                    cones.append((eta, axis.phi_std(), tot, rho, dpt))
                rc.append(cones)

        # the background jets: every hydro background's bulk_bg samples, as zjets.cxx
        bgj = {k: [] for k in ("bg", "bg_unit", "sample", "n_coll")}
        bg_rows = []
        qbj = {k: [] for k in bgj}
        qa_bg_rows = []
        if bulk:
            seen = set()
            no_z = type("NoZ", (), {"z": None})()
            for g in range(int(r._offsets[fi]), int(r._offsets[fi + 1])):
                info = infos[g]
                if info.bg_unit is None or info.bg_unit in seen:
                    continue
                seen.add(info.bg_unit)
                n_bg = h["bulk_bg"].n_samples(info.bg_unit)
                for k in range(min(n_bg, a.samples)):
                    ids = (("bg", bgid[g]), ("bg_unit", info.bg_unit), ("sample", k),
                           ("n_coll", info.n_coll if info.n_coll is not None else -1))
                    for key, v in ids:
                        bgj[key].append(v)
                    bp = pseudojets(fj, sample("bulk_bg", info.bg_unit, k), no_z)
                    every = c.jets(bp, True, ALL if k < a.qa_samples else a.ptmin)
                    bg_rows.append([x for x in every if x[1] > a.ptmin])
                    if k < a.qa_samples:
                        for key, v in ids:
                            qbj[key].append(v)
                        qa_bg_rows.append(every)

        def jet_record(lists, names=JET_NAMES):
            counts = np.array([len(row) for row in lists], np.int64)
            return ak.zip({nm: ak.unflatten(np.array([x[i] for row in lists for x in row],
                                                     np.float64), counts)
                           for i, nm in enumerate(names)})

        def bg_tree(ids, lists):
            return {"bg": np.asarray(ids["bg"], np.int64),
                    "bg_unit": np.asarray(ids["bg_unit"], np.int32),
                    "sample": np.asarray(ids["sample"], np.int32),
                    "n_coll": np.asarray(ids["n_coll"], np.int32),
                    "jet": jet_record(lists)}

        with uproot.recreate(os.path.join(a.out, f"{name}_jets.root")) as fo:
            fo["jet_events"] = {k: np.asarray(v, np.int64 if k in ("event", "bg") else
                                              np.int32 if k in ("local_event", "file",
                                                                "bg_unit", "pthat_bin", "n_coll",
                                                                "n_samples", "lep1_pid",
                                                                "lep2_pid") else np.float64)
                                for k, v in evs.items()}
            out = {"event": np.asarray(rows["event"], np.int64),
                   "sample": np.asarray(rows["sample"], np.int32),
                   "frag_sample": np.asarray(rows["frag_sample"], np.int32)}
            for gname, lists, names in (
                    ("jet", jet, JET_NAMES),
                    ("nw", nw, JET_NAMES),
                    ("emb", emb, ("ref_pt", "ref_eta", "ref_phi", "pt", "dr")),
                    ("rc", rc, ("eta", "phi", "sum", "rho", "dpt"))):
                out[gname] = jet_record(lists, names)
            for k, v in zcone.items():
                out[k] = np.asarray(v, np.float64)
            fo["jets"] = out
            fo["qa_jets"] = {"event": np.asarray(qa_rows["event"], np.int64),
                             "sample": np.asarray(qa_rows["sample"], np.int32),
                             "jet": jet_record(qa_jet)}
            if bg_rows:
                fo["bg_jets"] = bg_tree(bgj, bg_rows)
                fo["qa_bg_jets"] = bg_tree(qbj, qa_bg_rows)
            fo["file_info"] = mapio.file_info_tree(jw.file_record(r, fi))
            fo["run_info"] = mapio.run_info_tree(run_info)
        print(f"{name}: {len(evs['event'])} Z events, {sum(len(j) for j in jet)} jets, "
              f"{time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
