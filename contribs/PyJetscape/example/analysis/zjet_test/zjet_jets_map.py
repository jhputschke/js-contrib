#!/usr/bin/env python3
"""The map step of the Z+jet analysis (CMS 1702.01060) in Python: the jets of every oversample
of the Z events of a production directory, one ``<stem>_jets.root`` per production file.
The same algorithm and output as the C++ map step, example/analysis_root/zjet/zjets.cxx
(its header has the definitions); check_cxx.py jets compares the two, zjet_jets.py is the
final stage of both.

    python zjet_jets_map.py <h5 dir or quoted glob> <out dir> [--seed 1] [--samples N]
                            [--cones 4] [--ptmin 5] [--qa-samples 1] [--files i:j]

--files i:j maps only production files i .. j-1 (in the reader's order), for several processes
at once; the weights still come from all files, so the slices combine.

Jets: anti-k_T R = 0.3 from all final particles but neutrinos and the Z's leptons; Pb+Pb with
explicit ghosts on a fixed grid and rho(jet) from the k_T R = 0.4 median in a strip
|y - y_jet| < 1 (|y| < 2.2), p_T = p_T,raw - rho A; the fragments embedded in bulk_bg; random
cones in bulk_bg; the jets of every background's bulk_bg samples (tree bg_jets, the final
stage's B); per jet also its mass, the number of its (non-ghost) constituents and the leading
one's p_T.  The jet QA (trees qa_jets, qa_bg_jets): the first --qa-samples samples of every Z
event and background with all their jets (p_T,raw > 1 GeV, no cut on the subtracted p_T), for
zjet_jets_qa.py.  Slow in Python (~0.3 s per Pb+Pb oversample): use --samples for checks.
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
sys.path.insert(0, os.path.dirname(HERE))
try:
    import jetscape  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "python"))
import zjet_wake as zw  # noqa: E402

R_JET, ETA_JET, PT_REF, DR_MATCH, ETA_RC = 0.3, 2.0, 10.0, 0.15, 1.6
RC_SALT = 0x5243000000000000
NU = (12, 14, 16)
JET_NAMES = ("pt_raw", "pt", "eta", "phi", "area", "rho", "m", "nconst", "lead_pt")
ALL = -math.inf


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


STRIP = 1.0          # the rho strip's half width in rapidity (Clusterer.rho_sel)


class StripRho:
    """rho(jet) of JetMedianBackgroundEstimator(SelectorAbsRapMax(2.2) * SelectorStrip(1.0)),
    computed once per strip content.  With explicit ghosts the median depends only on which
    k_T jets lie in the strip |y - y_jet| < 1, and that set changes only where y_jet +- 1
    crosses a k_T jet's rapidity: the jets in between share one value, so each distinct
    content (the number of k_T jets below y_jet - 1 and below y_jet + 1) is computed once.
    A k_T jet within 1e-9 of a strip edge (where the selector's comparison decides) is never
    cached.  The same values as the estimator's, bit for bit."""

    def __init__(self, fj, cs, sel):
        self.cs = cs                             # the estimator's jets refer to it
        jets = cs.inclusive_jets()
        self.b = fj.JetMedianBackgroundEstimator(sel)
        self.b.set_jets(jets)
        self.ys = np.sort(np.array([j.rap() for j in jets]))
        self.cache = {}

    def rho(self, jet):
        y, ys = jet.rap(), self.ys
        key = []
        for edge in (y - STRIP, y + STRIP):
            i = int(np.searchsorted(ys, edge))
            if (i < len(ys) and abs(ys[i] - edge) < 1e-9) or (i and abs(ys[i - 1] - edge) < 1e-9):
                return self.b.rho(jet)
            key.append(i)
        key = tuple(key)
        if key not in self.cache:
            self.cache[key] = self.b.rho(jet)
        return self.cache[key]


def rap_phi(ev, keep):
    """Rapidity and phi of the kept particles, as FastJet computes them (for preselecting the
    random cones' particles; the cone sum itself uses PseudoJet.delta_R)."""
    p = np.asarray(ev["p"], np.float64)[keep]
    E, px, py, pz = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    kt2 = px * px + py * py
    m2 = np.maximum(E * E - kt2 - pz * pz, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = 0.5 * np.log((kt2 + m2) / (E + np.abs(pz)) ** 2)
    y = np.where(pz > 0, -y, y)
    return np.nan_to_num(y, nan=0.0, posinf=1e5, neginf=-1e5), np.arctan2(py, px)


class Clusterer:
    def __init__(self, fj):
        self.fj = fj
        self.akt = fj.JetDefinition(fj.antikt_algorithm, R_JET)
        self.kt = fj.JetDefinition(fj.kt_algorithm, 0.4)
        self.area = fj.AreaDefinition(fj.active_area_explicit_ghosts,
                                      fj.GhostedAreaSpec(2.6, 1, 0.01, 0.0, 0.0))
        self.rho_sel = fj.SelectorAbsRapMax(2.2) * fj.SelectorStrip(1.0)

    def bge(self, parts):
        """The k_T jets' rho estimator of a particle list (StripRho: rho(jet) as
        JetMedianBackgroundEstimator's, computed once per strip content)."""
        return StripRho(self.fj, self.fj.ClusterSequenceArea(parts, self.kt, self.area),
                        self.rho_sel)

    @staticmethod
    def _jet(j, pt, a, rho, ghosts):
        n, lead = 0, 0.0
        for q in j.constituents():
            if not ghosts or not q.is_pure_ghost():
                n += 1
                lead = max(lead, q.pt())
        return (j.pt(), pt, j.eta(), j.phi_std(), a, rho, j.m(), float(n), lead)

    def jets(self, parts, subtract, ptmin):
        """-> list of (pt_raw, pt, eta, phi, area, rho, m, nconst, lead_pt), as zjets.cxx's
        jets_of: p_T,raw > 1 GeV, |eta| < 2, p_T > ptmin (ALL: no cut)."""
        fj = self.fj
        out = []
        if not subtract:
            cs = fj.ClusterSequence(parts, self.akt)
            for j in fj.sorted_by_pt(cs.inclusive_jets(1.0)):
                if abs(j.eta()) < ETA_JET and j.pt() >= ptmin:
                    out.append(self._jet(j, j.pt(), 0.0, 0.0, False))
            return out
        b = self.bge(parts)
        cs = fj.ClusterSequenceArea(parts, self.akt, self.area)
        for j in fj.sorted_by_pt(cs.inclusive_jets(1.0)):
            if not abs(j.eta()) < ETA_JET:
                continue
            if j.pt() <= ptmin:                  # rho >= 0: p_T <= p_T,raw cannot pass
                continue
            rho, a = b.rho(j), j.area()
            pt = j.pt() - rho * a
            if pt > ptmin:
                out.append(self._jet(j, pt, a, rho, True))
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("h5")
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--samples", type=int, default=1 << 30)
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
    c = Clusterer(fj)
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

        evs = {k: [] for k in ("event", "local_event", "file", "bg", "pthat_bin", "weight_mb",
                               "n_coll", "n_samples", "z_pt", "z_y", "z_phi", "z_m",
                               "lep1_pid", "lep1_pt", "lep1_eta", "lep1_phi", "lep2_pid",
                               "lep2_pt", "lep2_eta", "lep2_phi")}
        rows = {k: [] for k in ("event", "sample", "frag_sample")}
        jet, emb, rc = [], [], []
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
                           ("pthat_bin", k_),
                           ("weight_mb", sig[k_] / nk[k_] if k_ >= 0 and sig else math.nan),
                           ("n_coll", info.n_coll if info.n_coll is not None else -1),
                           ("n_samples", n_used)):
                evs[key].append(v)
            for key in list(evs)[8:]:
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
                every = c.jets(pseudojets(fj, full, info), bulk,
                               ALL if k < a.qa_samples else a.ptmin)
                jet.append([x for x in every if (x[1] > a.ptmin if bulk else x[1] >= a.ptmin)])
                if k < a.qa_samples:
                    qa_rows["event"].append(g)
                    qa_rows["sample"].append(k)
                    qa_jet.append(every)
                if not bulk:
                    emb.append([])
                    rc.append([])
                    continue
                fp = pseudojets(fj, frag, info)
                bp, b, by, bphi = background(info, k)
                ref = c.jets(fp, False, PT_REF)
                ej = c.jets(bp + fp, True, 1.0)
                pairs = []
                for rj in ref:
                    best, dr_best = math.nan, math.inf
                    for x in ej:
                        dr = math.hypot(x[2] - rj[2], zw.wrap_dphi(x[3] - rj[3], -math.pi))
                        if dr < dr_best:
                            dr_best, best = dr, x[1]
                    pairs.append((rj[1], rj[2], rj[3], best if dr_best < DR_MATCH else math.nan,
                                  dr_best if math.isfinite(dr_best) else math.nan))
                emb.append(pairs)
                cones = []
                for q in range(a.cones):
                    s = RC_SALT | (k << 16) | q
                    eta = -ETA_RC + 2 * ETA_RC * zw.key_uniform(key, 2 * s)
                    phi = -math.pi + 2 * math.pi * zw.key_uniform(key, 2 * s + 1)
                    axis = fj.PseudoJet()
                    axis.reset_PtYPhiM(1.0, eta, phi, 0.0)
                    # preselect in numpy (0.01 margin), then FastJet's delta_R in the same
                    # order: the same particles and the same sum as over all of bp
                    dphi = np.abs(bphi - axis.phi())
                    dphi = np.minimum(dphi, 2 * np.pi - dphi)
                    near = np.flatnonzero((by - axis.rap()) ** 2 + dphi ** 2
                                          < (R_JET + 0.01) ** 2)
                    tot = sum(bp[i].pt() for i in near if bp[i].delta_R(axis) < R_JET)
                    rho = b.rho(axis)
                    cones.append((eta, axis.phi_std(), tot, rho, tot - rho * math.pi * R_JET ** 2))
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
                                                                "pthat_bin", "n_coll",
                                                                "n_samples", "lep1_pid",
                                                                "lep2_pid") else np.float64)
                                for k, v in evs.items()}
            out = {"event": np.asarray(rows["event"], np.int64),
                   "sample": np.asarray(rows["sample"], np.int32),
                   "frag_sample": np.asarray(rows["frag_sample"], np.int32)}
            for gname, lists, names in (
                    ("jet", jet, JET_NAMES),
                    ("emb", emb, ("ref_pt", "ref_eta", "ref_phi", "pt", "dr")),
                    ("rc", rc, ("eta", "phi", "sum", "rho", "dpt"))):
                out[gname] = jet_record(lists, names)
            fo["jets"] = out
            fo["qa_jets"] = {"event": np.asarray(qa_rows["event"], np.int64),
                             "sample": np.asarray(qa_rows["sample"], np.int32),
                             "jet": jet_record(qa_jet)}
            if bg_rows:
                fo["bg_jets"] = bg_tree(bgj, bg_rows)
                fo["qa_bg_jets"] = bg_tree(qbj, qa_bg_rows)
        print(f"{name}: {len(evs['event'])} Z events, {sum(len(j) for j in jet)} jets, "
              f"{time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
