#!/usr/bin/env python3
"""
analysis/OO-RHIC/validate_holes.py -- validation of the hole and partner treatment for jets in
the medium (oo_bias.py sample `jets_holes`: Matter + LBT with recoils, iColored, stored with
--store-partons --store-hadronization-input; negative hadrons from
prod_OO_isr/negative_hadrons.py, <stem>_hadrons_jet_neg.h5 next to every jet_frag file).

    python oo_bias.py run jets_holes -j 8 --mps
    for f in ~/prod/OO-RHIC_OO/bias/jets_holes/*_jet_frag.h5; do python ../../prod_OO_isr/negative_hadrons.py $f; done
    python validate_holes.py               # -> results/validate_holes.json

Checks:
  1. bookkeeping: the stored holes and the partners found in the Pythia input against
     iColored's colour audit (count and energy, per event); the negative pass's
     four-momentum against holes + partner copies; how many events were written as
     negative partons;
  2. the hole and partner energy per class (the peripheral limit: -> 0);
  3. full jets (anti-kT R = 0.4, |eta| < 0.7, all particles but neutrinos, constituents
     pT > 0.2 GeV): uncorrected; holes + partners subtracted at parton level; holes alone at
     parton level; the negative hadrons (and pstat -2 partons) subtracted.  Subtraction: the
     positive hadrons are clustered (jets above 3 GeV), then the negative four-momenta with
     Delta R < R to the jet axis are subtracted and pT, eta recomputed.  Per jet the
     parton-level and negative-hadron subtractions are compared; R_CP(0-20/60-80, MBD) per
     window for every variant;
  4. charged jets: the negative charged hadrons against 2/3 of the holes' + partners'
     momenta (the charged fraction as an approximation);
  5. leading charged hadrons (|eta| < 1, 6-30 GeV): yield with and without the negative
     charged hadrons.
Model results, not vetted.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hstat as H  # noqa: E402
import oo_bias as B  # noqa: E402
import oo_bias_jets as J  # noqa: E402

SAMPLE = "jets_holes"
R = 0.4
ETA_JET = 0.7
PT_EDGES = np.array([5, 8, 11, 14, 18, 23, 30, 40, 55.0])
VARIANTS = ("none", "parton", "parton_holes", "neg")          # full jets
CVARIANTS = ("none", "neg", "two_thirds")                    # charged jets
INVISIBLE = (12, 14, 16)


def _eta_phi(p):
    px, py, pz = p[:, 1], p[:, 2], p[:, 3]
    pa = np.sqrt(px * px + py * py + pz * pz)
    eta = 0.5 * np.log(np.clip(pa + pz, 1e-12, None) / np.clip(pa - pz, 1e-12, None))
    return eta, np.arctan2(py, px)


def _ragged(f, grp, n):
    off = f[f"{grp}/offsets"][: n + 1]
    return f[f"{grp}/data"][int(off[0]):int(off[-1])], off - off[0]


def cluster(p, ptmin=3.0):
    """anti-kT jets of one event's particles p (K, 4) [E, px, py, pz] -> (J, 4)."""
    import fastjet
    if len(p) == 0:
        return np.zeros((0, 4))
    parts = [fastjet.PseudoJet(float(x[1]), float(x[2]), float(x[3]), float(x[0])) for x in p]
    cs = fastjet.ClusterSequence(parts, fastjet.JetDefinition(fastjet.antikt_algorithm, R))
    js = cs.inclusive_jets(ptmin)
    return np.array([[j.E(), j.px(), j.py(), j.pz()] for j in js]).reshape(-1, 4)


def subtract(jets, neg, scale=1.0):
    """Subtract scale x the four-momenta of neg within Delta R < R of each jet axis."""
    if len(jets) == 0 or len(neg) == 0:
        return jets.copy()
    ej, fj = _eta_phi(jets)
    en, fn = _eta_phi(neg)
    dphi = np.abs((fj[:, None] - fn[None, :] + np.pi) % (2 * np.pi) - np.pi)
    inside = (ej[:, None] - en[None, :]) ** 2 + dphi ** 2 < R * R
    return jets - scale * inside.astype(float) @ neg


def binned(jets):
    out = np.zeros(len(PT_EDGES) - 1)
    if len(jets):
        pt = np.hypot(jets[:, 1], jets[:, 2])
        eta, _ = _eta_phi(jets)
        ib = np.digitize(pt, PT_EDGES) - 1
        ok = (np.abs(eta) < ETA_JET) & (ib >= 0) & (ib < len(out))
        np.add.at(out, ib[ok], 1.0)
    return out


def file_events(path):
    """Per event of one jet_frag file and its jet_neg file: the checks' raw numbers."""
    import hdf5plugin  # noqa: F401
    import h5py
    neg_path = path.replace("_hadrons_jet_frag.h5", "_hadrons_jet_neg.h5")
    charged = B.charged_ids()
    with h5py.File(path) as h, h5py.File(neg_path) as hn:
        n = int(h.attrs["nunits_written"])
        assert int(hn.attrs["nunits_written"]) == n, neg_path
        p, counts, pid = B._units(h, n, charged=False, with_pid=True)
        q, qcounts, qid = B._units(hn, n, charged=False, with_pid=True)
        par, poff = _ragged(h, "partons", n)
        hin, hoff = _ragged(h, "hadronization_input", n)
        audit = {k: h[f"units/colour_{k}"][:n] for k in ("holes", "hole_energy", "partners", "partner_energy")}
        nu = {k: hn[f"units/{k}"][:n] for k in ("neg_in_E", "neg_out_E", "neg_in_px", "neg_out_px",
                                                  "neg_in_pz", "neg_out_pz", "neg_mode", "neg_soft_E")}
        win = h["units/pthat_bin"][:n].astype(int)
    off = np.r_[0, np.cumsum(counts)]
    qoff = np.r_[0, np.cumsum(qcounts)]
    nb = len(PT_EDGES) - 1
    res = {"win": win, "jets": {v: np.zeros((n, nb)) for v in VARIANTS},
           "cjets": {v: np.zeros((n, nb)) for v in CVARIANTS},
           "had": np.zeros((n, 2)), "hole_E": np.zeros(n), "partner_E": np.zeros(n),
           "n_holes": np.zeros(n, int), "n_partners": np.zeros(n, int), "dpt": [],
           "audit": audit, "neg": nu}
    for e in range(n):
        P = p[off[e]:off[e + 1]]
        Pid = pid[off[e]:off[e + 1]]
        Q = q[qoff[e]:qoff[e + 1]]
        Qid = qid[qoff[e]:qoff[e + 1]]
        pr = par[poff[e]:poff[e + 1]]
        holes = pr[pr[:, 2] == -1]
        hi = hin[hoff[e]:hoff[e + 1]]
        partners = []
        for hl in holes:
            j = np.flatnonzero(np.abs(hi[:, 4:7] - hl[4:7]).sum(1) < 1e-6)
            if len(j):
                partners.append(hi[j[0], 3:7])
        partners = np.asarray(partners).reshape(-1, 4)
        hole4 = holes[:, 3:7]
        res["n_holes"][e], res["hole_E"][e] = len(holes), hole4[:, 0].sum()
        res["n_partners"][e], res["partner_E"][e] = len(partners), partners[:, 0].sum() if len(partners) else 0.0
        both = np.vstack([hole4, partners])
        # full jets
        pt = np.hypot(P[:, 1], P[:, 2])
        eta, _ = _eta_phi(P) if len(P) else (np.zeros(0), None)
        keep = (pt > 0.2) & (np.abs(eta) < ETA_JET + R + 0.1) & ~np.isin(np.abs(Pid), INVISIBLE)
        jets = cluster(P[keep])
        res["jets"]["none"][e] = binned(jets)
        jp = subtract(jets, both)
        jn = subtract(jets, Q)
        res["jets"]["parton"][e] = binned(jp)
        res["jets"]["parton_holes"][e] = binned(subtract(jets, hole4))
        res["jets"]["neg"][e] = binned(jn)
        if len(jets):
            pt0 = np.hypot(jets[:, 1], jets[:, 2])
            e0, _ = _eta_phi(jets)
            sel = (pt0 > 10) & (np.abs(e0) < ETA_JET)
            for a, b, c in zip(pt0[sel], np.hypot(jp[sel, 1], jp[sel, 2]), np.hypot(jn[sel, 1], jn[sel, 2])):
                res["dpt"].append((e, a, b, c))
        # charged jets
        ch = keep & np.isin(np.abs(Pid), charged)
        cj = cluster(P[ch])
        qh = Q[np.isin(np.abs(Qid), charged)]
        res["cjets"]["none"][e] = binned(cj)
        res["cjets"]["neg"][e] = binned(subtract(cj, qh))
        res["cjets"]["two_thirds"][e] = binned(subtract(cj, both, 2.0 / 3.0))
        # leading charged hadrons
        hsel = (np.abs(eta) < 1.0) & (pt > 6) & (pt < 30) & np.isin(np.abs(Pid), charged)
        qpt = np.hypot(qh[:, 1], qh[:, 2]) if len(qh) else np.zeros(0)
        qeta = _eta_phi(qh)[0] if len(qh) else np.zeros(0)
        res["had"][e] = hsel.sum(), np.sum((np.abs(qeta) < 1.0) & (qpt > 6) & (qpt < 30))
    res["dpt"] = np.asarray(res["dpt"]).reshape(-1, 4)
    return res


def main():
    files = sorted(glob.glob(str(B.OUT / SAMPLE / "*_hadrons_jet_frag.h5")))
    files = [f for f in files if os.path.exists(f.replace("_hadrons_jet_frag.h5", ".json"))
             and os.path.exists(f.replace("_hadrons_jet_frag.h5", "_hadrons_jet_neg.h5"))]
    if not files:
        sys.exit(f"validate_holes.py: no jet_frag + jet_neg files under {B.OUT / SAMPLE}")
    with ProcessPoolExecutor(8) as ex:
        res = list(ex.map(file_events, files))
    cat = lambda k: np.concatenate([r[k] for r in res])           # noqa: E731
    win = cat("win")
    jet, mb = B.load(SAMPLE), B.load("mb")
    assert np.array_equal(win, jet["pthat_bin"]), "files and oo_bias.load do not line up"
    ncoll = jet["glauber_ncoll"].astype(float)
    WIN = B.SAMPLES[SAMPLE][2]                                     # this sample's pTHat windows
    windows = [WIN[k] for k in sorted(set(win))]
    cm, cj = J.classes(mb["n_mbd"], jet["n_mbd"], B.SPH_CLASSES, np.random.default_rng(11))
    out = {"sample": SAMPLE, "n_events": int(len(win)), "files": len(files), "windows": windows}
    # 1. bookkeeping
    aud = {k: np.concatenate([r["audit"][k] for r in res]) for k in res[0]["audit"]}
    neg = {k: np.concatenate([r["neg"][k] for r in res]) for k in res[0]["neg"]}
    nh, eh, npar, epar = cat("n_holes"), cat("hole_E"), cat("n_partners"), cat("partner_E")
    out["bookkeeping"] = {
        "holes_match_audit": bool(np.array_equal(nh, aud["holes"]) and np.allclose(eh, aud["hole_energy"], atol=1e-6)),
        "partners_match_audit": bool(np.array_equal(npar, aud["partners"])
                                     and np.allclose(epar, aud["partner_energy"], atol=1e-6)),
        "max_dE_negative_pass": float(np.max(np.abs(neg["neg_out_E"] - neg["neg_in_E"]))),
        "max_dp_negative_pass": float(max(np.max(np.abs(neg["neg_out_px"] - neg["neg_in_px"])),
                                          np.max(np.abs(neg["neg_out_pz"] - neg["neg_in_pz"])))),
        "neg_mode_fractions": {str(m): float(np.mean(neg["neg_mode"] == m)) for m in (0, 1, 2)},
        "neg_in_E_equals_holes_plus_partners": bool(np.allclose(neg["neg_in_E"], eh + epar, atol=1e-5)),
        "soft_E_mean": float(neg["neg_soft_E"].mean())}
    # 2. per class
    nc = len(B.SPH_CLASSES)
    out["per_class"] = {"classes": B.SPH_CLASSES}
    for k, v in (("hole_E", eh), ("partner_E", epar), ("n_holes", nh.astype(float))):
        m, e = H.class_mean(cj, v, nc)
        out["per_class"][k], out["per_class"][k + "_err"] = m.tolist(), e.tolist()
    # 3./4. jets: R_CP per window
    out["jets"], out["cjets"] = {}, {}
    for key, vars_ in (("jets", VARIANTS), ("cjets", CVARIANTS)):
        for v in vars_:
            arr = np.concatenate([r[key][v] for r in res])
            out[key][v] = {}
            for k in sorted(set(win)):
                m = win == k
                pb = J.per_bin(cm, mb["glauber_ncoll"], cj[m], ncoll[m], arr[m], B.SPH_CLASSES,
                               [(0, 3)], np.zeros(int(m.sum()), np.int64), 1, J.MIN_ENTRIES)
                out[key][v]["-".join(map(str, WIN[k]))] = {"rcp": pb["rcp"]["0-20/60-80"], "n": pb["n"]}
    first = np.r_[0, np.cumsum([len(r["win"]) for r in res])][:-1]
    dpt = np.concatenate([r["dpt"] + np.array([f0, 0, 0, 0]) for r, f0 in zip(res, first)])
    if len(dpt):
        cls_j = cj[dpt[:, 0].astype(int)]
        for lab, col in (("shift_parton", 2), ("shift_neg", 3)):
            m_, e_ = H.class_mean(cls_j, dpt[:, 1] - dpt[:, col], nc)
            out.setdefault("per_jet_class", {"classes": B.SPH_CLASSES})[lab] = m_.tolist()
            out["per_jet_class"][lab + "_err"] = e_.tolist()
        m_, _ = H.class_mean(cls_j, dpt[:, 1], nc)
        out["per_jet_class"]["mean_pt"] = m_.tolist()
        d = dpt[:, 2] - dpt[:, 3]
        ev = dpt[:, 0].astype(int)
        out["per_jet"] = {"n_jets": int(len(d)), "mean_pt_none": float(dpt[:, 1].mean()),
                          "mean_shift_parton": float((dpt[:, 1] - dpt[:, 2]).mean()),
                          "mean_shift_neg": float((dpt[:, 1] - dpt[:, 3]).mean()),
                          "mean_parton_minus_neg": float(d.mean()), "rms_parton_minus_neg": float(d.std())}
    # 5. leading hadrons
    had = cat("had")
    out["leading_hadrons"] = {"positive": float(had[:, 0].sum()), "negative": float(had[:, 1].sum()),
                              "ratio_with_negative": float(1 - had[:, 1].sum() / max(had[:, 0].sum(), 1))}
    path = HERE / "results" / "validate_holes.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    # summary
    b = out["bookkeeping"]
    print(f"{out['n_events']} events in {out['files']} files, windows {windows}")
    print(f"1. holes = audit: {b['holes_match_audit']}; partners = audit: {b['partners_match_audit']}; "
          f"negative pass: max |dE| {b['max_dE_negative_pass']:.1e}, max |dp| {b['max_dp_negative_pass']:.1e} GeV, "
          f"input = holes + partners: {b['neg_in_E_equals_holes_plus_partners']}; modes {b['neg_mode_fractions']}")
    pc = out["per_class"]
    print("2. per MBD class: hole E " + "  ".join(f"{v:.2f}±{e:.2f}" for v, e in zip(pc["hole_E"], pc["hole_E_err"]))
          + " | partner E " + "  ".join(f"{v:.2f}" for v in pc["partner_E"]))
    if "per_jet" in out:
        pj = out["per_jet"]
        print(f"3. per jet (pT > 10 GeV, {pj['n_jets']} jets, <pT> {pj['mean_pt_none']:.1f}): shift parton-level "
              f"{pj['mean_shift_parton']:.3f}, negative hadrons {pj['mean_shift_neg']:.3f} GeV; "
              f"parton - neg {pj['mean_parton_minus_neg']:.3f} ± rms {pj['rms_parton_minus_neg']:.3f} GeV")
    if "per_jet_class" in out:
        q = out["per_jet_class"]
        print("   per MBD class, jets pT > 10 GeV: shift parton-level "
              + "  ".join(f"{v:.3f}±{e:.3f}" for v, e in zip(q["shift_parton"], q["shift_parton_err"]))
              + " | negative hadrons " + "  ".join(f"{v:.3f}±{e:.3f}" for v, e in zip(q["shift_neg"], q["shift_neg_err"]))
              + " GeV | <pT> " + " ".join(f"{v:.1f}" for v in q["mean_pt"]))
    lab = [f"{lo:g}-{hi:g}" for lo, hi in zip(PT_EDGES[:-1], PT_EDGES[1:])]
    for key, vars_ in (("jets", VARIANTS), ("cjets", CVARIANTS)):
        for w in out[key][vars_[0]]:
            print(f"   {key} pTHat {w}: R_CP(0-20/60-80) per jet-pT bin")
            for v in vars_:
                r = out[key][v][w]
                print(f"      {v:12s} " + "  ".join(f"{l} {x:.2f}±{e:.2f}" for l, (x, e), n in
                                                  zip(lab, r["rcp"], r["n"]) if n > 100))
    lh = out["leading_hadrons"]
    print(f"5. leading charged hadrons 6-30 GeV: {lh['positive']:.0f} positive, {lh['negative']:.0f} negative "
          f"(yield x {lh['ratio_with_negative']:.4f})")
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
