#!/usr/bin/env python3
"""
analysis/OO-RHIC/oo_bulk.py -- the O+O 200 GeV bulk against STAR (no retune): how the bulk set of
the O+O tune XML does on STAR's multiplicity distributions, from the events of the centrality-bias
test (oo_bias.py); no new events.

    python oo_bulk.py analyse          # -> results/oo_bulk.json
    jupyter nbconvert --to notebook --execute oo_bulk_comparison.ipynb --output results/oo_bulk_comparison_executed.ipynb

Model events (oo_bias.py): minimum bias = the `mb` stand-in (ISR chain without the subtraction and
with the veto off, bulk hadrons only; the same strings and geometry as minimum-bias events, see
oo_bias.py), MUSIC + iSS with one sample per event, charged hadrons with |eta| < 5.2, b 0-10 fm.
Jet events = `jets` (production settings; bulk + the hard event's hadrons), when present.
Every model number has its statistical error (`<key>_err`, hstat.py): one hydro and one iSS sample
per event, so events are independent; particle distributions use the event-to-event spread.

STAR data (refs/star_oo/star_oo_bulk.json, read from the vector figures of the arXiv sources):
  arXiv:2510.19645 Fig. 5   p(N_ch) and <N_ch> vs centrality, N_ch efficiency corrected,
                            0.2 < pT < 2 GeV, |eta| < 1.5; centrality from N_ch
  arXiv:2604.13935 Fig. 1/4 raw nMIP in the EPD (2.14 < |eta| < 5.09) for minimum bias and for
                            events with a charged trigger (5-7, 7-10, 10-30 GeV, |eta| < 1.5); EA
                            class edges.  nMIP is not N_ch: the model's EPD charged multiplicity is
                            mapped with one scale k (MIPs per primary charged particle), fitted, with
                            and without a Gaussian per-particle response width.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hstat as H  # noqa: E402
import oo_bias as B  # noqa: E402

STAR = HERE / "refs" / "star_oo" / "star_oo_bulk.json"
OUT_JSON = Path(os.environ.get("OO_BULK_JSON", HERE / "results" / "oo_bulk.json"))
EPD_STAR = (2.14, 5.09)
FLOW = dict(eta=1.5, ptmin=0.2, ptmax=2.0)
ETA_EDGES = np.round(np.arange(-5.2, 5.2001, 0.2), 3)
#: STAR Fig. 5b classes (centre points 0.5 ... 9.5, 12.5, 17.5, 25, ..., 75)
FIG5B_CLASSES = [(i, i + 1) for i in range(10)] + [(10, 15), (15, 20)] + [(i, i + 10) for i in range(20, 80, 10)]
EA_CLASSES = [(0, 10), (10, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
TRIG_BINS = {"trig_5_7": (5, 7), "trig_7_10": (7, 10), "trig_10_30": (10, 30)}


def _eta_pt(p):
    E, px, py, pz = p.T
    pt = np.hypot(px, py)
    pa = np.sqrt(px * px + py * py + pz * pz)
    eta = 0.5 * np.log(np.clip(pa + pz, 1e-12, None) / np.clip(pa - pz, 1e-12, None))
    return eta, pt


def per_event(p, counts):
    """Per event: N_ch in STAR's flow acceptance and in the EPD, the dN/deta histogram, and the
    leading charged pT at |eta| < 1.5."""
    eta, pt = _eta_pt(p)
    n = len(counts)
    ev = np.repeat(np.arange(n), counts)
    flow = (np.abs(eta) < FLOW["eta"]) & (pt > FLOW["ptmin"]) & (pt < FLOW["ptmax"])
    epd = (np.abs(eta) > EPD_STAR[0]) & (np.abs(eta) < EPD_STAR[1])
    ib = np.digitize(eta, ETA_EDGES) - 1
    ok = (ib >= 0) & (ib < len(ETA_EDGES) - 1)
    deta = np.zeros((n, len(ETA_EDGES) - 1), np.int32)
    np.add.at(deta, (ev[ok], ib[ok]), 1)
    lead = np.zeros(n)
    m = np.abs(eta) < 1.5
    np.maximum.at(lead, ev[m], pt[m])
    return {"nch_flow": np.bincount(ev[flow], minlength=n), "n_epd": np.bincount(ev[epd], minlength=n),
            "deta": deta, "lead": lead}


def load(name, with_jet):
    import hdf5plugin  # noqa: F401
    import h5py
    rec = {}
    for jf in sorted(glob.glob(str(B.OUT / name / "*_hadrons_jet_frag.h5"))):
        bf = jf.replace("_hadrons_jet_frag.h5", "_hadrons_bulk_bg.h5")
        if not os.path.exists(jf.replace("_hadrons_jet_frag.h5", ".json")):
            continue  # job still running (the summary is written at the end)
        with h5py.File(jf) as h:
            n = int(h.attrs["nunits_written"])
            u = {k: h[f"units/{k}"][:n] for k in ("glauber_ncoll", "glauber_npart", "glauber_b", "pthat_bin",
                                                   "has_medium")}
            pj, cj = B._units(h, n) if with_jet else (None, None)
        if os.path.exists(bf):
            with h5py.File(bf) as h:
                pb, cb = B._units(h, n)
        else:
            pb, cb = np.zeros((0, 4)), np.zeros(n, np.int64)
        if with_jet:
            # one combined particle list per event: bulk then jet
            off_b = np.concatenate([[0], np.cumsum(cb)])
            off_j = np.concatenate([[0], np.cumsum(cj)])
            p = np.concatenate([np.concatenate([pb[off_b[i]:off_b[i + 1]], pj[off_j[i]:off_j[i + 1]]])
                                for i in range(n)])
            c = cb + cj
        else:
            p, c = pb, cb
        for k, v in {**u, **per_event(p, c)}.items():
            rec.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in rec.items()} if rec else {}


def percentile_classes(x, classes, rng):
    """Class index per event from percentiles of x (high x = central), random tie-break."""
    xr = x + rng.random(len(x))
    cuts = np.percentile(xr, [100 - hi for _, hi in classes])
    c = np.full(len(x), len(classes) - 1)
    for k in range(len(classes) - 1, -1, -1):
        c[xr >= cuts[k]] = k
    return c, cuts


def smear(n, k, s, rng):
    """nMIP for N primary charged particles: k N with a Gaussian per-particle width s (in MIPs)."""
    mu = k * n
    return mu + (rng.normal(size=len(n)) * s * np.sqrt(np.maximum(n, 0)) * k if s > 0 else 0)


def hist_norm(x, w, lo, hi, err=False):
    """Self-normalised density in STAR's bins (unit area over the drawn range); with err=True
    (density, error)."""
    v, e = H.density(x, np.append(lo, hi[-1]), w)
    return (v, e) if err else v


EPD_FIT_MIN = 78.0


def match_area(m, star_mb, err=None):
    """Scale a model density to STAR's area over the fitted bins (nMIP >= EPD_FIT_MIN); with err
    (scaled density, scaled error)."""
    lo, hi, y = map(np.asarray, (star_mb["lo"], star_mb["hi"], star_mb["y"]))
    use, w = lo >= EPD_FIT_MIN, hi - lo
    f = np.sum((y * w)[use]) / max(np.sum((m * w)[use]), 1e-300)
    return m * f if err is None else (m * f, np.asarray(err) * f)


def fit_k(n_epd, star_mb, rng, widths=(0.0, 0.5, 1.0)):
    """Fit k (and pick the response width s) to the shape of STAR's minimum-bias nMIP distribution
    (Fig. 1) above nMIP 78, both normalised to the same area there (the two lowest, widest bins carry
    the trigger-efficiency correction and the low-activity events, which are tested separately)."""
    lo, hi, y = map(np.asarray, (star_mb["lo"], star_mb["hi"], star_mb["y"]))
    use = lo >= EPD_FIT_MIN
    best = {}
    for s in widths:
        ks = np.linspace(0.5, 4.0, 351)
        chi = []
        for k in ks:
            m = match_area(hist_norm(smear(n_epd, k, s, np.random.default_rng(5)), None, lo, hi), star_mb)
            # relative 3 % uncertainty per bin (STAR's are smaller than the markers): a shape metric
            chi.append(np.sum(((m[use] - y[use]) / (0.03 * y[use])) ** 2))
        i = int(np.argmin(chi))
        best[str(s)] = {"k": float(ks[i]), "chi2": float(chi[i]), "ndf": int(use.sum() - 1)}
    return best


def analyse():
    rng = np.random.default_rng(23)
    star = json.loads(STAR.read_text())
    mb = load("mb", with_jet=False)
    if not mb:
        sys.exit("oo_bulk.py: no minimum-bias events")
    nmb = len(mb["nch_flow"])
    res = {"n_mb": nmb, "eta_edges": ETA_EDGES.tolist()}

    # 1. p(N_ch) in STAR's flow acceptance
    edges = np.arange(-0.5, 260, 2.0)  # STAR's points sit at odd N_ch, 2 wide
    h, he = H.counts(mb["nch_flow"], edges)
    res["pnch"] = {"edges": edges.tolist(), "counts": h.tolist(), "counts_err": he.tolist(),
                   "mean": float(mb["nch_flow"].mean()),
                   "mean_err": float(mb["nch_flow"].std() / np.sqrt(nmb))}

    # 2. <N_ch> per centrality class, centrality from N_ch itself (as STAR does)
    c, cuts = percentile_classes(mb["nch_flow"], FIG5B_CLASSES, rng)
    res["nch_vs_cent"] = {"classes": FIG5B_CLASSES}
    for key, col in (("nch", "nch_flow"), ("npart", "glauber_npart"), ("ncoll", "glauber_ncoll")):
        v, e = H.class_mean(c, mb[col], len(FIG5B_CLASSES))
        res["nch_vs_cent"][key], res["nch_vs_cent"][key + "_err"] = v.tolist(), e.tolist()

    # 3. EPD: fit the nMIP scale, class edges
    fit = fit_k(mb["n_epd"], star["ea_fig1"]["mb"], rng)
    s_best = min(fit, key=lambda s: fit[s]["chi2"])
    k_best = fit[s_best]["k"]
    star_edges = star["ea_fig1"]["class_edges"]
    pct = {"0-10%": 90, "10-20%": 80, "20-40%": 60, "40-60%": 40, "60-80%": 20}
    model_edges = {lab: float(np.percentile(mb["n_epd"] + rng.random(nmb), p)) for lab, p in pct.items()}
    # statistical error of the percentile edges: bootstrap over the minimum-bias events
    brng = np.random.default_rng(31)                    # own stream: the classes below stay as they were
    boot = np.array([[np.percentile(x, p) for p in pct.values()]
                     for x in (mb["n_epd"][brng.integers(0, nmb, nmb)] + brng.random(nmb) for _ in range(100))])
    edge_err = dict(zip(pct, boot.std(0)))
    k_edges = {lab: star_edges[lab] / model_edges[lab] for lab in pct if model_edges[lab] > 0}
    res["epd"] = {"fit": fit, "s_best": float(s_best), "k_best": k_best,
                  "mean_n_epd": float(mb["n_epd"].mean()),
                  "edges_model_nch": model_edges, "edges_model_nch_err": {k: float(v) for k, v in edge_err.items()},
                  "edges_star_nmip": star_edges, "k_from_edges": k_edges,
                  "k_from_edges_err": {lab: float(k_edges[lab] * edge_err[lab] / model_edges[lab]) for lab in k_edges}}
    nm = smear(mb["n_epd"], k_best, float(s_best), np.random.default_rng(5))
    lo, hi = (np.asarray(star["ea_fig1"]["mb"][x]) for x in ("lo", "hi"))
    v, e = hist_norm(nm, None, lo, hi, err=True)
    vs, es = match_area(v, star["ea_fig1"]["mb"], e)
    res["epd"].update(mb_model=vs.tolist(), mb_model_err=es.tolist(), mb_model_full=v.tolist(), mb_model_full_err=e.tolist())
    res["epd"]["frac_below_fit_model"], res["epd"]["frac_below_fit_model_err"] = H.fraction(nm < EPD_FIT_MIN)
    sm = star["ea_fig1"]["mb"]
    res["epd"]["frac_below_fit_star"] = float(sum(y * (h - l) for l, h, y in zip(sm["lo"], sm["hi"], sm["y"])
                                                  if l < EPD_FIT_MIN))

    # 4. dN/deta per EA class (EPD percentiles of the model), all pT
    cE, _ = percentile_classes(mb["n_epd"], EA_CLASSES, rng)
    # per-event histograms: the error is the event-to-event spread (hstat.event_mean)
    v, e = H.event_mean(mb["deta"])
    D = {"classes": EA_CLASSES, "mb": (v / 0.2).tolist(), "mb_err": (e / 0.2).tolist(), "per_class": [], "per_class_err": []}
    for k in range(len(EA_CLASSES)):
        v, e = H.event_mean(mb["deta"][cE == k])
        D["per_class"].append((v / 0.2).tolist()); D["per_class_err"].append((e / 0.2).tolist())
    v, e = H.class_mean(cE, mb["glauber_npart"], len(EA_CLASSES))
    D["npart"], D["npart_err"] = v.tolist(), e.tolist()
    mid = (np.abs(np.asarray(ETA_EDGES[:-1]) + 0.1) < 0.5)
    v, e = H.class_mean(cE, mb["deta"][:, mid].sum(1) / (0.2 * mid.sum()), len(EA_CLASSES))
    D["mid05"], D["mid05_err"] = v.tolist(), e.tolist()
    res["dndeta"] = D

    # 5. trigger-selected EPD distributions (jets, production settings), when present
    jets = load("jets", with_jet=True)
    if jets:
        w = jets["glauber_ncoll"].astype(float)
        n_j = smear(jets["n_epd"], k_best, float(s_best), np.random.default_rng(6))
        out = {"n": int(len(w)), "windows": B.JET_WINDOWS, "per_window": {}, "per_trigger": {}}
        for iw, (a, b) in enumerate(B.JET_WINDOWS):
            m = (jets["pthat_bin"] == iw) & (jets["lead"] > 5)
            v, e = hist_norm(n_j[m], w[m], lo, hi, err=True)
            vs, es = match_area(v, star["ea_fig1"]["trig_5_7"], e)
            out["per_window"][f"{a}-{b}"] = {"n": int(m.sum()), "h": v.tolist(), "h_err": e.tolist(),
                                             "h_star_area": vs.tolist(), "h_star_area_err": es.tolist()}
        for name, (a, b) in TRIG_BINS.items():
            m = (jets["lead"] > a) & (jets["lead"] < b)
            # the windows have gaps (11-18, 23-30 GeV), so this is not an inclusive trigger sample
            v, e = hist_norm(n_j[m], w[m], lo, hi, err=True)
            out["per_trigger"][name] = {"n": int(m.sum()), "h": v.tolist(), "h_err": e.tolist()}
        out["mean_n_epd_weighted"] = float(np.average(jets["n_epd"], weights=w))
        res["jets"] = out
    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(res, indent=1))
    print(f"minimum bias: {nmb} events, <N_ch>(|eta|<1.5, 0.2-2 GeV) {res['pnch']['mean']:.1f}, "
          f"<N_ch,EPD> {res['epd']['mean_n_epd']:.1f}")
    print("  <N_ch> 0-1 % / 0-10 % / 70-80 %: model {:.1f} / {:.1f} / {:.1f}".format(
        res["nch_vs_cent"]["nch"][0], np.mean(res["nch_vs_cent"]["nch"][:10]), res["nch_vs_cent"]["nch"][-1]))
    print(f"  nMIP scale: k {k_best:.2f} (width {s_best}); from the class edges: "
          + ", ".join(f"{k} {v:.2f}" for k, v in res["epd"]["k_from_edges"].items()))
    print("  jets:", "yes" if jets else "not yet")
    print("wrote", OUT_JSON)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != "analyse":
        sys.exit(__doc__)
    sys.exit(analyse())
