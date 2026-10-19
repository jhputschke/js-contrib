"""Z as jet1 of CMS HIN-25-012 (2602.19431): Z+jet events per dijet eta-gap class.

    python gap_count.py <pbpb jets dir[,dir...]> <pp jets dir[,dir...]>

(several directories, comma-separated: campaigns mapped apart, combined by zjet_jets.load_jets)

Z: zjet_jets.z_selection's mass and lepton cuts, |y_Z| < 1, p_T^Z > a threshold.
jet2: the leading map-step jet (|eta| < 2; Pb+Pb rhoA-subtracted, the map step's R) of each
oversample, p_T > a threshold and Delta phi(Z, jet) > 5pi/6.  Gap |y_Z - eta_jet|, CMS's
classes.  Per event the fraction of its oversamples in each class; N_eff of the selected
Z's (sum w)^2 / sum w^2 times the class's weighted fraction is the experiment-like count.
"""
import os
import sys

import awkward as ak
import numpy as np

ANALYSIS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ANALYSIS)                                     # jetcorr
sys.path.insert(0, os.path.join(ANALYSIS, "zjet"))                # zjet_jets
import zjet_jets as zj  # noqa: E402

GAPS = [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0)]
CUTS = [(40, 30), (60, 30), (60, 40), (80, 50), (100, 50), (130, 50)]


def leading(J, prefix="jet"):
    pt = J[f"{prefix}_pt"]
    i = ak.argmax(pt, axis=1, keepdims=True)
    has = ak.to_numpy(ak.num(pt) > 0)
    f = lambda k: ak.to_numpy(ak.fill_none(ak.firsts(J[f"{prefix}_{k}"][i]), np.nan))
    return f("pt"), f("eta"), f("phi"), has


def count(E, J, ncoll, label, prefix="jet", embedded=False):
    """embedded: only the oversamples with the embedded event (nw: the map step's
    --emb-samples, per map output)."""
    base = (E["z_m"] > 70) & (E["z_m"] < 110) \
        & zj.lep_ok(E["lep1_pid"], E["lep1_eta"]) & zj.lep_ok(E["lep2_pid"], E["lep2_eta"])
    idx = {g: i for i, g in enumerate(E["event"])}
    ev = np.array([idx[g] for g in J["event"]])
    lim = E["emb_samples"][ev] if embedded else np.full(len(ev), 1 << 30)
    pt, eta, phi, has = leading(J, prefix)
    has = has & (np.asarray(J["sample"]) < lim)
    ns = np.minimum(E["n_samples"][ev], lim).astype(float)
    w0 = E["weight_mb"].astype(float) * (E["n_coll"] if ncoll else 1.0)
    print(f"\n== {label}: {len(E['event'])} Z events, {len(J['event'])} oversample rows "
          f"(n_samples {np.unique(E['n_samples'])})")
    print(f"{'pTZ>':>5} {'jet>':>5} | {'N_Z':>5} {'Neff_Z':>7} | {'f_jet2':>6} | "
          + " | ".join(f"{lo:.1f}-{hi:.1f}: frac  Neff " for lo, hi in GAPS))
    for zpt, jpt in CUTS:
        sel = base & (np.abs(E["z_y"]) < 1.0) & (E["z_pt"] > zpt)
        w = np.where(sel, w0, 0.0)
        neff = w.sum() ** 2 / (w ** 2).sum()
        ok = has & (pt > jpt) & (zj.dphi(phi, E["z_phi"][ev]) > 5 * np.pi / 6)
        gap = np.abs(E["z_y"][ev] - eta)
        f_rows = []
        for lo, hi in [(0.0, np.inf)] + GAPS:
            r = ok & (gap >= lo) & (gap < hi)
            p = np.bincount(ev, weights=r / ns, minlength=len(E["event"]))   # per-event fraction
            f_rows.append((w * p).sum() / w.sum())
        cells = " | ".join(f"{f:10.3f} {neff * f:6.0f}" for f in f_rows[1:])
        print(f"{zpt:5d} {jpt:5d} | {sel.sum():5d} {neff:7.0f} | {f_rows[0]:6.3f} | {cells}")


Ep, Jp, _ = zj.load_jets(sys.argv[2].split(","))
count(Ep, Jp, False, f"pp {sys.argv[2]}")
Eb, Jb, _ = zj.load_jets(sys.argv[1].split(","))
count(Eb, Jb, True, f"Pb+Pb {sys.argv[1]} (full event, rhoA)")
count(Eb, Jb, True, f"Pb+Pb {sys.argv[1]} (no-wake twin nw)", prefix="nw", embedded=True)
