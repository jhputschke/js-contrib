#!/usr/bin/env python3
"""
analysis/OO-RHIC/centrality_norm.py -- how STAR's O+O centrality classes map onto the model's, from
the results of oo_bulk.py (no events are read).

    python centrality_norm.py [SAMPLES]        # default: every results/oo_bulk_oo_*.json
                                               # -> results/centrality_norm.json

Why.  STAR's classes are fractions of a cross section that STAR does not measure directly:
  * flow paper (arXiv:2510.19645, appendix): centrality from N_ch (0.2 < pT < 2 GeV, |eta| < 1.5)
    through a nucleon-Glauber + negative-binomial fit to p(N_ch); the measured p(N_ch) (Fig. 5a)
    is depleted at low N_ch (trigger and vertex efficiency), the fit supplies the missing events;
  * jet paper (arXiv:2604.13935): EA classes as edges in the EPD's raw nMIP (Fig. 1).
The model's own percentiles count every event with at least one NN collision.  If the two total
cross sections differ, every class edge shifts, most in peripheral classes: <N_ch> vs centrality
then tests the normalisation more than the bulk physics.

What it does.
  1. STAR alone: the missing area A below the measured p(N_ch) is fixed by requiring that the
     class means of Fig. 5b follow from Fig. 5a with total area sum(p) + A (the same procedure as
     STAR's, with the Glauber + NBD fit replaced by its integral).  This gives STAR's class edges in
     N_ch.
  2. Per model sample: the model centrality at STAR's N_ch edges (fraction of model events above
     the edge) and at STAR's EPD edges (nMIP / k, k the scale fitted to the shape of STAR's
     minimum-bias nMIP distribution above nMIP 78, which does not depend on the normalisation).
     If the model's p(N_ch) has STAR's shape above the edges and only the normalisation differs,
     c_model = f * c_STAR with one f for all edges.
  3. Per model sample: the f for which the model's classes, taken as fractions of the f highest-
     multiplicity events, give STAR's <N_ch> per class with the flattest model/STAR ratio.
  4. The same with STAR's minimum-bias trigger emulated in the model (oo_bulk.py "trigger": TOF
     plus a forward coincidence in the VPD or the EPD acceptance): does the trigger explain f?

Caveats: the EPD comparison uses the model's n_epd without the per-particle response width (the
fit's best width, 1 MIP, broadens the distribution by ~sqrt(N) MIPs: a few % at the 60-80 % edge);
model results, not vetted physics results.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
STAR = HERE / "refs" / "star_oo" / "star_oo_bulk.json"
OUT = RES / "centrality_norm.json"
FIG5B_CLASSES = [(i, i + 1) for i in range(10)] + [(10, 15), (15, 20)] + [(i, i + 10) for i in range(20, 80, 10)]
EA_EDGE_PCT = {"0-10%": 10, "10-20%": 20, "20-40%": 40, "40-60%": 60, "60-80%": 80}


# ── STAR ──────────────────────────────────────────────────────────────────────────────────
def _top_down(x, w):
    """Bins sorted from high x down, as (lower edge, upper edge, weight) of 2-wide bins."""
    o = np.argsort(x)[::-1]
    return x[o] - 1.0, x[o] + 1.0, w[o]


def hist_class_means(x, p, total, classes):
    """<x> per class (percent of `total` from the top) of a histogram with 2-wide bins centred at
    x, uniform within a bin; the area beyond sum(p) sits below the lowest bin."""
    lo, hi, w = _top_down(np.asarray(x, float), np.asarray(p, float))
    cum = np.concatenate([[0.0], np.cumsum(w)])
    out = []
    for a, b in classes:
        A, Bq = a / 100 * total, b / 100 * total
        num = den = 0.0
        for k in range(len(w)):
            s, e = max(A, cum[k]), min(Bq, cum[k + 1])
            if e <= s or w[k] <= 0:
                continue
            # the bin's area from the top: x runs from hi (cum[k]) down to lo (cum[k+1])
            xs = hi[k] - (s - cum[k]) / w[k] * 2.0
            xe = hi[k] - (e - cum[k]) / w[k] * 2.0
            num += (e - s) * 0.5 * (xs + xe)
            den += e - s
        out.append(num / den if den > 0 else np.nan)
    return np.array(out)


def hist_edge(x, p, total, pct):
    """x at which the area above is pct % of total (uniform within 2-wide bins)."""
    lo, hi, w = _top_down(np.asarray(x, float), np.asarray(p, float))
    cum = np.concatenate([[0.0], np.cumsum(w)])
    A = pct / 100 * total
    k = int(np.searchsorted(cum, A) - 1)
    if k >= len(w):
        return float(lo[-1])
    return float(hi[k] - (A - cum[k]) / w[k] * 2.0)


def star_normalisation(S):
    """The missing area (fraction of the measured area) that makes Fig. 5a reproduce Fig. 5b."""
    x, p = np.array(S["flow_fig5a"]["nch"]), np.array(S["flow_fig5a"]["p"])
    target = np.array(S["flow_fig5b"]["nch"])
    meas = p.sum()
    scan = np.arange(0.0, 0.3001, 0.001)
    chi = []
    for a in scan:
        m = hist_class_means(x, p, meas * (1 + a), FIG5B_CLASSES)
        chi.append(np.sum(((m - target) / (0.02 * target)) ** 2))   # 2 %: the read-off precision
    chi = np.nan_to_num(np.array(chi), nan=np.inf)      # a class beyond the histogram: excluded
    i = int(np.argmin(chi))
    a = float(scan[i])
    ok = scan[chi <= chi[i] + 1.0]
    total = meas * (1 + a)
    means = hist_class_means(x, p, total, FIG5B_CLASSES)
    edges = {f"{b:g}": hist_edge(x, p, total, b) for _, b in FIG5B_CLASSES}
    # fraction of STAR's total below N_ch = 20 that is not in the measured histogram
    return {"missing_area_frac": a, "missing_area_frac_range": [float(ok.min()), float(ok.max())],
            "chi2": float(chi[i]), "n_classes": len(FIG5B_CLASSES),
            "measured_frac_of_total": float(1 / (1 + a)),
            "class_means_reproduced": means.tolist(), "class_means_star": target.tolist(),
            "nch_edges": edges}


# ── model ─────────────────────────────────────────────────────────────────────────────────
def model_centrality(qpct, qval, v):
    """Model centrality (percent of events above v) from the quantile table."""
    qv = np.asarray(qval, float)
    below = np.interp(v, qv, qpct, left=0.0, right=100.0)
    return float(100.0 - below)


def class_means_top_f(qpct, qval, f, classes):
    """<x> per class with the classes taken as fractions of the f highest events."""
    q, v = np.asarray(qpct, float), np.asarray(qval, float)
    out = []
    for a, b in classes:
        lo, hi = 100 - b * f, 100 - a * f            # quantile range of the class
        g = np.linspace(lo, hi, 41)
        out.append(float(np.mean(np.interp(g, q, v))))
    return np.array(out)


def analyse_sample(name, star_n, S):
    R = json.loads((RES / f"oo_bulk_{name}.json").read_text())
    if "quantiles" not in R:
        return None
    Q = R["quantiles"]
    out = {"n_mb": R["n_mb"]}
    # 2a. N_ch edges
    out["flow_edges"] = {c: {"nch": e, "c_star": float(c),
                             "c_model": model_centrality(Q["pct"], Q["nch_flow"], e)}
                         for c, e in star_n["nch_edges"].items()}
    for c, d in out["flow_edges"].items():
        d["f"] = d["c_model"] / d["c_star"]
    # 2b. EPD edges
    k = R["epd"]["k_best"]
    out["epd_k"] = k
    out["epd_edges"] = {}
    for lab, nmip in R["epd"]["edges_star_nmip"].items():
        cs = EA_EDGE_PCT[lab]
        cm = model_centrality(Q["pct"], Q["n_epd"], nmip / k)
        out["epd_edges"][lab] = {"nmip": nmip, "c_star": cs, "c_model": cm, "f": cm / cs}
    # 3. best f for <N_ch> vs centrality
    target = np.array(S["flow_fig5b"]["nch"])
    best = None
    for f in np.arange(0.60, 1.0001, 0.005):
        r = class_means_top_f(Q["pct"], Q["nch_flow"], f, FIG5B_CLASSES) / target
        sl = float(np.std(np.log(r)))
        if best is None or sl < best[0]:
            best = (sl, float(f), r)
    r1 = class_means_top_f(Q["pct"], Q["nch_flow"], 1.0, FIG5B_CLASSES) / target
    out["flat_f"] = best[1]
    out["flat_f_spread"] = best[0]
    out["ratio_f1"] = r1.tolist()
    out["ratio_flat_f"] = best[2].tolist()
    out["ratio_flat_f_mean"] = float(np.exp(np.mean(np.log(best[2]))))
    # 4. with the emulated minimum-bias trigger (oo_bulk.py: TOF + forward coincidence): classes as
    #    percentiles of the triggered events, and the f that is still needed on top of it
    out["trigger"] = {}
    for k, d in R.get("trigger", {}).items():
        r = class_means_top_f(Q["pct"], d["nch_flow_q"], 1.0, FIG5B_CLASSES) / target
        fs = [(float(np.std(np.log(class_means_top_f(Q["pct"], d["nch_flow_q"], f, FIG5B_CLASSES) / target))), float(f))
              for f in np.arange(0.60, 1.0001, 0.005)]
        sl, fb = min(fs)
        out["trigger"][k] = {"eff": d["eff"], "ratio": r.tolist(), "extra_f": fb,
                             "total_f": fb * d["eff"]}
    return out


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    S = json.loads(STAR.read_text())
    star_n = star_normalisation(S)
    names = argv or sorted(Path(f).stem.replace("oo_bulk_", "")
                           for f in glob.glob(str(RES / "oo_bulk_oo_*.json")))
    res = {"star": star_n, "classes": FIG5B_CLASSES, "samples": {},
           "note": "model, not vetted; see the module docstring"}
    print(f"STAR: measured p(N_ch) is {100 * star_n['measured_frac_of_total']:.1f} % of the total that "
          f"reproduces Fig. 5b (missing area {star_n['missing_area_frac']:.3f}, range "
          f"{star_n['missing_area_frac_range'][0]:.3f}-{star_n['missing_area_frac_range'][1]:.3f}, "
          f"chi2 {star_n['chi2']:.1f} / {star_n['n_classes']} classes at 2 %)")
    print("   STAR N_ch edges at 1/10/20/40/60/80 %: " + " / ".join(
        f"{star_n['nch_edges'][c]:.1f}" for c in ("1", "10", "20", "40", "60", "80")))
    for n in names:
        o = analyse_sample(n, star_n, S)
        if o is None:
            print(f"{n}: no quantiles (rerun oo_bulk.py analyse --sample {n})")
            continue
        res["samples"][n] = o
        fe = o["flow_edges"]
        print(f"{n}: model centrality at STAR's N_ch edges 10/20/40/60/80 %: " + " / ".join(
            f"{fe[c]['c_model']:.1f}" for c in ("10", "20", "40", "60", "80")))
        print("   at STAR's EPD edges 10/20/40/60/80 %: " + " / ".join(
            f"{o['epd_edges'][l]['c_model']:.1f}" for l in EA_EDGE_PCT) + f"  (k {o['epd_k']:.2f})")
        print(f"   flattest <N_ch> model/STAR with f = {o['flat_f']:.3f}: mean {o['ratio_flat_f_mean']:.3f}, "
              f"rms of log {o['flat_f_spread']:.3f}")
        for k, d in o["trigger"].items():
            print(f"   trigger {k:9s} eff {d['eff']:.3f}: <N_ch>/STAR 0-1/20-30/40-50/60-70/70-80 % " + " / ".join(
                f"{d['ratio'][i]:.2f}" for i in (0, 12, 14, 16, 17)) + f"; still needs f {d['extra_f']:.3f}")
    OUT.write_text(json.dumps(res, indent=1))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
