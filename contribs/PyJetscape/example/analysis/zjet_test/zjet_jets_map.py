#!/usr/bin/env python3
"""The map step of the Z+jet analysis (CMS 1702.01060) in Python: the jets of every oversample
of the Z events of a production directory, one ``<stem>_jets.root`` per production file.
The same algorithm and output as the C++ map step, example/analysis_root/zjet/zjets.cxx
(its header has the definitions); check_cxx.py jets compares the two, zjet_jets.py is the
final stage of both.

    python zjet_jets_map.py <h5 dir> <out dir> [--seed 1] [--samples N] [--cones 4]
                            [--ptmin 5]

Jets: anti-k_T R = 0.3 from all final particles but neutrinos and the Z's leptons; Pb+Pb with
explicit ghosts on a fixed grid and rho(jet) from the k_T R = 0.4 median in a strip
|y - y_jet| < 1 (|y| < 2.2), p_T = p_T,raw - rho A; the fragments embedded in bulk_bg; random
cones in bulk_bg.  Slow in Python (~1 s per Pb+Pb oversample): use --samples for checks."""
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


class Clusterer:
    def __init__(self, fj):
        self.fj = fj
        self.akt = fj.JetDefinition(fj.antikt_algorithm, R_JET)
        self.kt = fj.JetDefinition(fj.kt_algorithm, 0.4)
        self.area = fj.AreaDefinition(fj.active_area_explicit_ghosts,
                                      fj.GhostedAreaSpec(2.6, 1, 0.01, 0.0, 0.0))
        self.rho_sel = fj.SelectorAbsRapMax(2.2) * fj.SelectorStrip(1.0)

    def bge(self, parts):
        b = self.fj.JetMedianBackgroundEstimator(self.rho_sel, self.kt, self.area)
        b.set_particles(parts)
        return b

    def jets(self, parts, subtract, ptmin):
        """-> list of (pt_raw, pt, eta, phi, area, rho), as zjets.cxx's jets_of."""
        fj = self.fj
        out = []
        if not subtract:
            cs = fj.ClusterSequence(parts, self.akt)
            for j in fj.sorted_by_pt(cs.inclusive_jets(ptmin)):
                if abs(j.eta()) < ETA_JET:
                    out.append((j.pt(), j.pt(), j.eta(), j.phi_std(), 0.0, 0.0))
            return out
        b = self.bge(parts)
        cs = fj.ClusterSequenceArea(parts, self.akt, self.area)
        for j in fj.sorted_by_pt(cs.inclusive_jets(1.0)):
            if not abs(j.eta()) < ETA_JET:
                continue
            rho, a = b.rho(j), j.area()
            pt = j.pt() - rho * a
            if pt > ptmin:
                out.append((j.pt(), pt, j.eta(), j.phi_std(), a, rho))
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("h5")
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--samples", type=int, default=1 << 30)
    ap.add_argument("--cones", type=int, default=4)
    ap.add_argument("--ptmin", type=float, default=5.0)
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
    for fi, stem in enumerate(r.stems):
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
                js = c.jets(pseudojets(fj, full, info), bulk, a.ptmin)
                jet.append(js)
                if not bulk:
                    emb.append([])
                    rc.append([])
                    continue
                fp = pseudojets(fj, frag, info)
                bg = sample("bulk_bg", info.bg_unit, k)
                bp = pseudojets(fj, bg, info)
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
                b = c.bge(bp)
                cones = []
                for q in range(a.cones):
                    s = RC_SALT | (k << 16) | q
                    eta = -ETA_RC + 2 * ETA_RC * zw.key_uniform(key, 2 * s)
                    phi = -math.pi + 2 * math.pi * zw.key_uniform(key, 2 * s + 1)
                    axis = fj.PseudoJet()
                    axis.reset_PtYPhiM(1.0, eta, phi, 0.0)
                    tot = sum(p.pt() for p in bp if p.delta_R(axis) < R_JET)
                    rho = b.rho(axis)
                    cones.append((eta, axis.phi_std(), tot, rho, tot - rho * math.pi * R_JET ** 2))
                rc.append(cones)

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
                    ("jet", jet, ("pt_raw", "pt", "eta", "phi", "area", "rho")),
                    ("emb", emb, ("ref_pt", "ref_eta", "ref_phi", "pt", "dr")),
                    ("rc", rc, ("eta", "phi", "sum", "rho", "dpt"))):
                counts = np.array([len(row) for row in lists], np.int64)
                cols = {nm: ak.unflatten(np.array([x[i] for row in lists for x in row],
                                                  np.float64), counts)
                        for i, nm in enumerate(names)}
                out[gname] = ak.zip(cols)
            fo["jets"] = out
        print(f"{name}: {len(evs['event'])} Z events, {sum(len(j) for j in jet)} jets, "
              f"{time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
