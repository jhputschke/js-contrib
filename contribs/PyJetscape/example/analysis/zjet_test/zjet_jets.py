#!/usr/bin/env python3
"""Z+jet against CMS 1702.01060 (HEPData ins1512112): the final stage of the jets of a Pb+Pb and a
pp sample, from either map step (example/analysis_root/zjet/zjets or zjet_jets_map.py).

    python zjet_jets.py --pbpb data/jets_cxx/pbpb_campaign --pp data/jets_cxx/vacuum_all
    python zjet_jets.py ... --tag _c00 --blocks 100 [--n-mixed 40] [--n-nearest 20]

Z: an opposite-sign lepton pair with 70 < m < 110 GeV, p_T^Z > 40 GeV, muons |eta| < 2.4,
electrons |eta| < 2.5 outside 1.44-1.57 (both p_T > 20 from the generator; CMS: muons from
10 GeV).  Jets: anti-k_T R = 0.3, |eta| < 1.6, Delta R > 0.4 from the Z's leptons, p_T > 30 GeV
after the smearing.  Weights: window cross section / events (x N_coll for Pb+Pb), every
oversample 1 / n_samples.  Observables per Z: Delta phi_jZ and x_jZ (p_T^Z > 60), <x_jZ> and
R_jZ in p_T^Z classes (Delta phi > 7 pi / 8).

The Pb+Pb analysis in three steps, from the model-only ideal to the experiment's way:
  step 1  the reference: the jets of the fragments alone (the embedding's reference jets: the
          quenched jet without any background or medium response), CMS's recipe for theory
          (generator-level jets smeared with the full PbPb resolution), and unsmeared
  step 2  S - B_own: the rho A-subtracted jets of the whole event (S) minus the jets of the Z's
          own background (bulk_bg, the same fluid without the jet, sample by sample): the jets
          not from the Z's interaction removed with the model's twin, as only the model can
  step 3  S - B_mixed: B from --n-mixed bulk_bg samples of the --n-nearest other backgrounds
          closest in N_coll, as CMS's 40 minimum-bias events of the same centrality; every
          background rotated by its own random angle (default; --no-rotate keeps them as
          produced): 3dMCGlauber puts every event's impact parameter along x, so without it the
          mixed backgrounds would share the Z event's event plane, unlike CMS's minimum-bias
          events (the means do not change, B_mixed's fluctuations against S do)
The wake's effect: the map step also stores the jets of the embedded event, bulk_bg k + the
same fragments (the Z event without its wake: the same fluid sample, random numbers and
subtraction), and steps 2 and 3 are repeated on it ('no wake', unsmeared); step 2 - step 2
without the wake is what the wake does to the observables, with the jackknife of the
difference (the two share their events) -> figures/zjet_jets_wake{tag}.png.
Steps 2 and 3 in four smearing variants (CMS's smearing being optional): none; detector (C, S
of PbPb: the model's own fluctuations stand for N); full (C, S, N: counts them twice); match
(C, S and sqrt(N^2 - sigma_RC^2), sigma_RC the random cones' width).  pp (every step's top
row): smeared with PbPb's resolution against CMS's smeared-pp tables 1, 3, 5, 7 (the model is
at particle level, so the whole sigma_PbPb).  The pp figure (zjet_jets_pp): the four observables
unsmeared, with pp's and with PbPb's resolution, against tables 1, 3, 5, 7 (smeared pp) and
table 9 (x_jZ of pp at pp's resolution, CMS's only unsmeared pp table), with model / CMS below.
CMS's resolution (Appendix A): sigma / p_T = sqrt(C^2 + S^2 / p_T + N^2 / p_T^2), C 0.061;
pp S 0.95, N 0.001; PbPb 0-30 % S 1.24, N 8.08.  Errors: a jackknife over blocks of the Z's
hydro backgrounds (pp: of events).
-> figures/zjet_jets_pp{tag}.png, zjet_jets_step{1,2,3}{tag}.png, zjet_jets_embedding{tag}.png,
   zjet_jets_summary{tag}.txt."""
import argparse
import glob
import os

import numpy as np

import figdata

HERE = os.path.dirname(os.path.abspath(__file__))

C = 0.061
RES = {"pp": (0.95, 0.001), "PbPb": (1.24, 8.08)}
PBPB_SMEAR = ("none", "detector", "full", "match")
SEED = 7                                       # --seed's default
PHI_EDGES = np.array([0, np.pi / 5, 2 * np.pi / 5, 3 * np.pi / 5, 7 * np.pi / 10, 4 * np.pi / 5,
                      9 * np.pi / 10, 19 * np.pi / 20, np.pi])
X_EDGES = np.linspace(0, 2, 11)
ZPT_BINS = [(40, 50), (50, 60), (60, 80), (80, 120)]
EMB_BINS = [(10, 20), (20, 30), (30, 50), (50, 100)]
OBS = ("phi", "x", "mx", "rjz")
TABLES = {"phi": (1, 2), "x": (3, 4), "mx": (5, 6), "rjz": (7, 8)}
LABELS = {"phi": "Δφ_jZ, p_T^Z > 60", "x": "x_jZ, p_T^Z > 60, Δφ > 7π/8",
          "mx": "⟨x_jZ⟩ vs p_T^Z", "rjz": "R_jZ vs p_T^Z"}


def _flat_tree(t):
    """A tree's arrays with the Python map step's records flattened (jet -> jet_pt, ...)."""
    out = {}
    for k in t.fields:
        if k in ("njet", "nnw", "nemb", "nrc"):
            continue
        x = t[k]
        if x.fields:
            for sub in x.fields:
                out[f"{k}_{sub}"] = x[sub]
        else:
            out[k] = x
    return out


def load_jets(directory):
    """The map step's files of a directory -> (events, jets, background jets), each a dict of
    flat names (jet_pt, emb_ref_pt, rc_dpt, ...), concatenated over the files; from either map
    step.  Background jets: {} without (vacuum)."""
    import awkward as ak
    import uproot
    paths = sorted(glob.glob(os.path.join(directory, "*_jets.root")))
    if not paths:
        raise FileNotFoundError(f"no *_jets.root in {directory}")
    evs, jets, bgs = [], [], []
    for p in paths:
        with uproot.open(p) as f:
            evs.append(f["jet_events"].arrays())
            jets.append(f["jets"].arrays())
            if "bg_jets" in f:
                bgs.append(f["bg_jets"].arrays())
    E = {k: ak.to_numpy(v) for k, v in _flat_tree(ak.concatenate(evs)).items()}
    J = _flat_tree(ak.concatenate(jets))
    for k in ("event", "sample", "frag_sample"):
        J[k] = ak.to_numpy(J[k])
    B = _flat_tree(ak.concatenate(bgs)) if bgs else {}
    for k in ("bg", "bg_unit", "sample", "n_coll"):
        if k in B:
            B[k] = ak.to_numpy(B[k])
    return E, J, B


def load_qa_jets(directory):
    """The jet QA trees of a directory's map-step files -> (qa_jets, qa_bg_jets), dicts of flat
    names as load_jets; qa_bg_jets {} without (vacuum).  Files from before the QA trees:
    KeyError."""
    import awkward as ak
    import uproot
    paths = sorted(glob.glob(os.path.join(directory, "*_jets.root")))
    if not paths:
        raise FileNotFoundError(f"no *_jets.root in {directory}")
    qs, qbs = [], []
    for p in paths:
        with uproot.open(p) as f:
            if "qa_jets" not in f:
                raise KeyError(f"{p}: no qa_jets tree (made before the jet QA: rerun the map step)")
            qs.append(f["qa_jets"].arrays())
            if "qa_bg_jets" in f:
                qbs.append(f["qa_bg_jets"].arrays())
    Q = _flat_tree(ak.concatenate(qs))
    for k in ("event", "sample"):
        Q[k] = ak.to_numpy(Q[k])
    QB = _flat_tree(ak.concatenate(qbs)) if qbs else {}
    for k in ("bg", "bg_unit", "sample", "n_coll"):
        if k in QB:
            QB[k] = ak.to_numpy(QB[k])
    return Q, QB


def lep_ok(pid, eta):
    a = np.abs(eta)
    return np.where(np.abs(pid) == 13, a < 2.4, (a < 2.5) & ~((a > 1.44) & (a < 1.57)))


def dphi(a, b):
    d = np.abs(a - b) % (2 * np.pi)
    return np.where(d > np.pi, 2 * np.pi - d, d)


def z_selection(E):
    return ((E["z_m"] > 70) & (E["z_m"] < 110) & (E["z_pt"] > 40)
            & lep_ok(E["lep1_pid"], E["lep1_eta"]) & lep_ok(E["lep2_pid"], E["lep2_eta"]))


def weights(E, sel, ncoll):
    w = E["weight_mb"].astype(float).copy()
    if ncoll and np.all(E["n_coll"][sel] >= 0):
        w = w * E["n_coll"]
    w = np.where(sel, w, 0.0)
    return w / w[sel].mean() if sel.any() else w


def z_counts(E, sel, w):
    """Per event: N_Z for p_T^Z > 60 and per p_T^Z class (weighted)."""
    zpt = E["z_pt"]
    nz = np.where(sel & (zpt > 60), w, 0.0)
    nzb = np.stack([np.where(sel & (zpt >= lo) & (zpt < hi), w, 0.0) for lo, hi in ZPT_BINS], 1)
    return {"nz": nz, "nzb": nzb}


def pair_sums(E, sel, e, pt, eta, phi, wj):
    """Per event the weighted Z-jet pair sums of jets (e: their event's row, pt after the
    smearing, wj: their weight): the Delta phi and x_jZ histograms, the pairs and x sums per
    p_T^Z class (Delta phi > 7 pi / 8)."""
    n = len(E["event"])
    nzb = len(ZPT_BINS)
    out = {"phi": np.zeros((n, len(PHI_EDGES) - 1)), "x": np.zeros((n, len(X_EDGES) - 1)),
           "pairs": np.zeros((n, nzb)), "xsum": np.zeros((n, nzb))}
    if not len(e):
        return out
    zpt = E["z_pt"]
    ok = sel[e] & (np.abs(eta) < 1.6) & (pt > 30)
    for l in (1, 2):
        ok &= np.hypot(eta - E[f"lep{l}_eta"][e], dphi(phi, E[f"lep{l}_phi"][e])) > 0.4
    dp = dphi(phi, E["z_phi"][e])
    x = pt / zpt[e]
    hi = ok & (zpt[e] > 60)
    away = hi & (dp > 7 * np.pi / 8)
    for k, edges, val, m in (("phi", PHI_EDGES, dp, hi), ("x", X_EDGES, x, away)):
        b = np.clip(np.searchsorted(edges, val, side="right") - 1, 0, len(edges) - 2)
        inside = m & (val >= edges[0]) & (val < edges[-1] + 1e-12)
        np.add.at(out[k], (e[inside], b[inside]), wj[inside])
    back = ok & (dp > 7 * np.pi / 8)
    for bi, (lo, hi_) in enumerate(ZPT_BINS):
        m = back & (zpt[e] >= lo) & (zpt[e] < hi_)
        np.add.at(out["pairs"][:, bi], e[m], wj[m])
        np.add.at(out["xsum"][:, bi], e[m], (wj * x)[m])
    return out


def observables(c, keep):
    s = {k: v[keep].sum(axis=0) for k, v in c.items()}
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"phi": s["phi"] / s["nz"] / np.diff(PHI_EDGES),
                "x": s["x"] / s["nz"] / np.diff(X_EDGES),
                "mx": np.where(s["pairs"] > 0, s["xsum"] / np.where(s["pairs"] > 0,
                                                                    s["pairs"], 1), np.nan),
                "rjz": s["pairs"] / s["nzb"]}


def with_errors(c, groups, n_blocks):
    v = observables(c, np.ones(len(groups), bool))
    ug = np.unique(groups)
    blocks = [b for b in np.array_split(ug, min(n_blocks, len(ug))) if len(b)]
    if len(blocks) < 2:
        return v, {k: np.full_like(x, np.nan) for k, x in v.items()}
    reps = [observables(c, ~np.isin(groups, b)) for b in blocks]
    G = len(blocks)
    with np.errstate(invalid="ignore"):
        err = {k: np.sqrt((G - 1) / G * np.nansum((np.array([r[k] for r in reps]) - v[k]) ** 2,
                                                   axis=0)) for k in v}
    return v, err


def diff_with_errors(c1, c2, groups, n_blocks):
    """observables(c1) - observables(c2) and the jackknife of the difference over the same
    blocks: for two results on the same events (the wake's effect)."""
    one = np.ones(len(groups), bool)
    o1, o2 = observables(c1, one), observables(c2, one)
    v = {k: o1[k] - o2[k] for k in o1}
    ug = np.unique(groups)
    blocks = [b for b in np.array_split(ug, min(n_blocks, len(ug))) if len(b)]
    if len(blocks) < 2:
        return v, {k: np.full_like(x, np.nan) for k, x in v.items()}
    reps = []
    for b in blocks:
        keep = ~np.isin(groups, b)
        r1, r2 = observables(c1, keep), observables(c2, keep)
        reps.append({k: r1[k] - r2[k] for k in r1})
    G = len(blocks)
    with np.errstate(invalid="ignore"):
        err = {k: np.sqrt((G - 1) / G * np.nansum((np.array([r[k] for r in reps]) - v[k]) ** 2,
                                                   axis=0)) for k in v}
    return v, err


def smear(pt, S, N, rng, n_extra=None):
    """CMS's resolution; ``n_extra`` replaces N (the 'match' and 'detector' variants)."""
    N = N if n_extra is None else n_extra
    sig = np.sqrt(C ** 2 + S ** 2 / pt + N ** 2 / pt ** 2)
    return pt * (1 + sig * rng.standard_normal(len(pt)))


def flatten_rows(E, J, prefix, rows, row_weight, ev_of_row, row_dphi=None):
    """Jets of the jagged columns ``prefix``_pt/eta/phi at ``rows`` -> flat (e, pt, eta, phi,
    wj): every jet with its row's event and weight; ``row_dphi`` rotates each row's jets."""
    import awkward as ak
    pt = J[f"{prefix}_pt"][rows]
    counts = ak.to_numpy(ak.num(pt))
    rep = np.repeat(np.arange(len(rows)), counts)
    phi = ak.to_numpy(ak.flatten(J[f"{prefix}_phi"][rows]))
    if row_dphi is not None:
        phi = phi + np.asarray(row_dphi, float)[rep]
    return (ev_of_row[rep], ak.to_numpy(ak.flatten(pt)),
            ak.to_numpy(ak.flatten(J[f"{prefix}_eta"][rows])), phi, row_weight[rep])


def background_angles(bg_ids, seed):
    """A random angle in [0, 2 pi) per hydro background (one draw per distinct id, in sorted
    order) -> function of an array of background ids.  Rotating a background (its Z events,
    their jets, the background's own jets and random cones) is an exact symmetry; only jets
    paired with another background's Z change: a mixed jet of background b seen from a Z of
    background b' turns by angle(b) - angle(b').  3dMCGlauber puts every event's impact
    parameter along x (all event planes near phi = 0); the rotation makes them random, as
    for CMS's minimum-bias events (ANA_Zjet.md, *The backgrounds' event planes*)."""
    ids = np.unique(np.asarray(bg_ids))
    ang = np.random.default_rng(seed).uniform(0, 2 * np.pi, len(ids))
    return lambda b: ang[np.searchsorted(ids, np.asarray(b))]


def rotation_seed(seed, given=None):
    """The rotations' seed: ``given`` (--rotate-seed), else derived from --seed
    (zjet_jets_qa.py uses zjet_jets.py's default --seed, so that both draw the same angles)."""
    return given if given is not None else [int(seed), 0x726F74]


def fmt(m):
    return np.array2string(m, precision=3, separator=", ", suppress_small=True,
                           floatmode="fixed", max_line_width=200)


def cms(hep, i):
    import yaml
    with open(os.path.join(hep, f"data{i}.yaml")) as f:
        d = yaml.safe_load(f)
    v = d["dependent_variables"][0]["values"]
    y = np.array([float(b["value"]) for b in v])
    e = np.array([np.hypot(*[float(x["symerror"]) for x in b["errors"]]) for b in v])
    return y, e


def chi2(m, me, y, e):
    """chi^2 and its bins, model and CMS errors in quadrature (CMS: stat. and syst.).  Bins
    where CMS gives no error are left out: CMS's empty bins (0 +- 0, x_jZ 1.6-2.0 in PbPb)
    are not measurements, and the model's small values there with tiny errors would
    dominate the sum."""
    ee = np.where(np.isfinite(me), me, 0.0)
    good = np.isfinite(m) & np.isfinite(y) & (e > 0)
    return float(np.sum((m[good] - y[good]) ** 2 / (e[good] ** 2 + ee[good] ** 2))), int(good.sum())


def pp_figure(a, res, xs, lines, plt):
    """pp against CMS: the four observables (top) and model / CMS (bottom).  CMS's pp tables
    1, 3, 5, 7 are smeared to PbPb's resolution, so they go with the PbPb-smeared model; table 9
    (x_jZ) is pp at pp's own resolution and goes with the pp-smeared model.  The unsmeared
    (particle-level) model shows what the smearing does."""
    pairs = {obs: [(TABLES[obs][0], "pp, PbPb-smeared", "CMS pp, smeared to PbPb", "o", "k")]
             for obs in OBS}
    pairs["x"].append((9, "pp, pp-smeared", "CMS pp", "D", "C2"))
    style = {"pp, PbPb-smeared": "C0", "pp, pp-smeared": "C2", "pp, unsmeared": "0.5"}
    zedges = np.array([lo for lo, _ in ZPT_BINS] + [ZPT_BINS[-1][1]], float)
    edges = {"phi": PHI_EDGES, "x": X_EDGES, "mx": zedges, "rjz": zedges}
    lines.append("pp against CMS 1702.01060 (zjet_jets_pp): chi2 with model and CMS errors")
    fig, ax = figdata.subplots(2, 4, figsize=(20, 7.5), sharex="col",
                           gridspec_kw={"height_ratios": [3, 1.3]})
    for col, obs in enumerate(OBS):
        top, bot = ax[0, col], ax[1, col]
        for j, k in enumerate(style):
            m, me = res[k][0][obs], res[k][1][obs]
            top.errorbar(xs[obs] * (1 + 0.008 * (j - 1)), m, me, fmt="s--", ms=3.5, mfc="none",
                         lw=0.8, color=style[k], label=f"model {k[4:]}")
        for ti, k, lab, mk, colr in pairs[obs]:
            y, e = cms(a.hepdata, ti)
            top.errorbar(xs[obs], y, e, fmt=mk, color=colr, ms=4, label=f"{lab} (table {ti})")
            m, me = res[k][0][obs], res[k][1][obs]
            with np.errstate(invalid="ignore", divide="ignore"):
                r, re_, rd = m / y, me / y, e / y
            bot.errorbar(xs[obs], r, re_, fmt="s", ms=3.5, mfc="none", color=style[k],
                         label=f"{k[4:]} / table {ti}")
            bot.stairs(1 + rd, edges[obs], baseline=1 - rd, fill=True, color=colr, alpha=0.15)
            c, nb = chi2(m, me, y, e)
            lines.append(f"  {obs:4s} {k:32s} vs data{ti}: chi2 {c:6.1f}/{nb}   model {fmt(m)}")
        m = res["pp, unsmeared"][0][obs]
        lines.append(f"  {obs:4s} {'pp, unsmeared':32s} (no table)        model {fmt(m)}")
        top.set_title(LABELS[obs])
        if obs == "phi":
            top.set_yscale("log")
        top.legend(fontsize=6)
        bot.axhline(1, color="0.5", lw=0.6)
        bot.set_ylim(0.2, 1.8)
        bot.set_ylabel("model / CMS")
        bot.legend(fontsize=6)
    ax[1, 0].set_xlabel("Δφ_jZ")
    ax[1, 1].set_xlabel("x_jZ")
    ax[1, 2].set_xlabel("p_T^Z [GeV]")
    ax[1, 3].set_xlabel("p_T^Z [GeV]")
    fig.suptitle("Z+jet in pp, CMS 1702.01060: the model unsmeared, with pp's and with PbPb's "
                 "jet resolution (bands: CMS's errors)", y=0.995)
    fig.tight_layout()
    figdata.savefig(fig, os.path.join(a.outdir, f"zjet_jets_pp{a.tag}.png"), dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pbpb", required=True, help="the map step's Pb+Pb *_jets.root")
    ap.add_argument("--pp", required=True, help="the map step's vacuum *_jets.root")
    ap.add_argument("--hepdata", default=os.path.join(HERE, "data", "hepdata_zjet",
                                                      "HEPData-ins1512112-v1-yaml"))
    ap.add_argument("--outdir", default=os.path.join(HERE, "figures"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--blocks", type=int, default=100)
    ap.add_argument("--n-mixed", type=int, default=40,
                    help="bulk_bg samples per Z for step 3's B (CMS: 40 minimum-bias events)")
    ap.add_argument("--n-nearest", type=int, default=20,
                    help="step 3: the other backgrounds closest in N_coll to draw them from")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="the smearing's and the draws' random numbers")
    ap.add_argument("--no-rotate", dest="rotate", action="store_false",
                    help="step 3: keep the backgrounds as produced (their event planes all near "
                         "phi = 0) instead of rotating each by a random angle (the default)")
    ap.add_argument("--rotate-seed", type=int, default=None,
                    help="seed of the background rotations (default: derived from --seed)")
    a = ap.parse_args()
    import awkward as ak
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(a.seed)
    lines = []
    S, N = RES["PbPb"]
    # ── pp: one result, every step's top row
    Ep, Jp, _ = load_jets(a.pp)
    selp = z_selection(Ep)
    wp = weights(Ep, selp, ncoll=False)
    idxp = {g: i for i, g in enumerate(Ep["event"])}
    evp = np.array([idxp[g] for g in Jp["event"]])
    rows = np.arange(len(Jp["event"]))
    pp_flat = flatten_rows(Ep, Jp, "jet", rows, wp[evp] / Ep["n_samples"][evp], evp)
    res = {}
    for key, sn in (("pp, pp-smeared", RES["pp"]), ("pp, PbPb-smeared", RES["PbPb"]),
                    ("pp, unsmeared", None)):
        e, pt, eta, phi, wj = pp_flat
        ptx = pt if sn is None else smear(pt, *sn, rng)
        c = {**z_counts(Ep, selp, wp), **pair_sums(Ep, selp, e, ptx, eta, phi, wj)}
        res[key] = with_errors(c, Ep["event"], a.blocks)
    lines.append(f"pp: {int(selp.sum())} Z's (CMS 1702.01060's cuts) of {len(selp)} in {a.pp}")

    # ── Pb+Pb
    Eb, Jb, Bb = load_jets(a.pbpb)
    selb = z_selection(Eb)
    wb = weights(Eb, selb, ncoll=True)
    n_bg = len(np.unique(Eb["bg"][selb]))
    groups = Eb["bg"] if n_bg >= 2 else Eb["event"]
    idx = {g: i for i, g in enumerate(Eb["event"])}
    ev = np.array([idx[g] for g in Jb["event"]])
    rows = np.arange(len(Jb["event"]))
    w_row = wb[ev] / Eb["n_samples"][ev]
    S_flat = flatten_rows(Eb, Jb, "jet", rows, w_row, ev)
    NW_flat = (flatten_rows(Eb, Jb, "nw", rows, w_row, ev)          # the event without the wake
               if "nw_pt" in Jb else None)
    F_flat = flatten_rows(Eb, Jb, "emb_ref", rows, w_row, ev)
    rc = ak.to_numpy(ak.flatten(Jb["rc_dpt"])) if "rc_dpt" in Jb else np.zeros(0)
    sig_rc = float(np.std(rc)) if len(rc) else 0.0
    lines.append(f"Pb+Pb: {int(selb.sum())} Z's of {len(selb)} on {n_bg} hydro background(s) in "
                 f"{a.pbpb}" + ("" if n_bg >= 2 else
                                " (jackknife over events: errors underestimated)"))
    # B_own: the Z's own background, sample k of bulk_bg against sample k of the event
    B_own = B_mix = None
    if Bb:
        bmap = {(int(b), int(k)): i for i, (b, k) in enumerate(zip(Bb["bg"], Bb["sample"]))}
        own = np.array([bmap.get((int(Eb["bg"][e_]), int(k)), -1)
                        for e_, k in zip(ev, Jb["sample"])])
        ok = own >= 0
        B_own = flatten_rows(Eb, Bb, "jet", own[ok], w_row[ok], ev[ok])
        # B_mixed: per Z event n_mixed bulk_bg samples of the n_nearest other backgrounds
        bgs, first = np.unique(Bb["bg"], return_index=True)
        ncoll_bg = Bb["n_coll"][first].astype(float)
        by_bg = {int(b): np.flatnonzero(Bb["bg"] == b) for b in bgs}
        mrows, mev, mw = [], [], []
        for e_ in np.flatnonzero(selb):
            others = bgs != Eb["bg"][e_]
            if not others.any():
                continue
            cand = bgs[others][np.argsort(np.abs(ncoll_bg[others] - Eb["n_coll"][e_]),
                                          kind="stable")[:a.n_nearest]]
            pool = np.concatenate([by_bg[int(b)] for b in cand])
            pick = rng.choice(pool, size=a.n_mixed, replace=len(pool) < a.n_mixed)
            mrows.append(pick)
            mev.append(np.full(a.n_mixed, e_))
            mw.append(np.full(a.n_mixed, wb[e_] / a.n_mixed))
        if mrows:
            mrows, mev = np.concatenate(mrows), np.concatenate(mev)
            turn = None
            if a.rotate:
                angle = background_angles(np.concatenate([Bb["bg"], Eb["bg"]]),
                                          rotation_seed(a.seed, a.rotate_seed))
                turn = angle(Bb["bg"][mrows]) - angle(Eb["bg"][mev])
            B_mix = flatten_rows(Eb, Bb, "jet", mrows, np.concatenate(mw), mev, turn)
            if not a.rotate:
                how = "as produced (--no-rotate): event planes all near phi = 0"
            elif a.rotate_seed is not None:
                how = f"rotated by random angles (seed {a.rotate_seed})"
            else:
                how = f"rotated by random angles (from --seed {a.seed})"
            lines.append(f"step 3: the mixed backgrounds {how}")
    zc = z_counts(Eb, selb, wb)

    def result(sig_flat, bkg_flat, n_extra):
        return with_errors(counts(sig_flat, bkg_flat, n_extra), groups, a.blocks)

    def counts(sig_flat, bkg_flat, n_extra):
        e, pt, eta, phi, wj = sig_flat
        ptx = pt if n_extra == "none" else smear(pt, S, N, rng, n_extra=n_extra)
        c = {**zc, **pair_sums(Eb, selb, e, ptx, eta, phi, wj)}
        if bkg_flat is not None:
            e, pt, eta, phi, wj = bkg_flat
            ptx = pt if n_extra == "none" else smear(pt, S, N, rng, n_extra=n_extra)
            b = pair_sums(Eb, selb, e, ptx, eta, phi, wj)
            for k in b:
                c[k] = c[k] - b[k]
        return c

    n_extra = {"none": "none", "detector": 0.0, "full": None,
               "match": np.sqrt(max(N ** 2 - sig_rc ** 2, 0.0))}
    steps = {1: {"fragments, theory (full σ_PbPb)": result(F_flat, None, None),
                 "fragments, none": result(F_flat, None, "none")}}
    if B_own is not None:
        steps[2] = {f"S − B_own, {v}": result(S_flat, B_own, n_extra[v]) for v in PBPB_SMEAR}
        if B_mix is not None:
            steps[3] = {f"S − B_mixed, {v}": result(S_flat, B_mix, n_extra[v])
                        for v in PBPB_SMEAR}
        else:
            lines.append("step 3 skipped: no other hydro background to mix with")
        s_only = result(S_flat, None, n_extra["match"])
    # the jets not from the Z's interaction, per Z (R_jZ of B, 'match' smearing)
    if B_own is not None:
        for lab, bf in (("B_own", B_own), ("B_mixed", B_mix)):
            if bf is None:
                continue
            e, pt, eta, phi, wj = bf
            b = pair_sums(Eb, selb, e, smear(pt, S, N, rng, n_extra=n_extra["match"]), eta, phi, wj)
            with np.errstate(invalid="ignore", divide="ignore"):
                r = b["pairs"].sum(0) / zc["nzb"].sum(0)
            lines.append(f"{lab}: background jets per Z (Δφ > 7π/8, R_jZ of B, 'match') in the "
                         f"p_T^Z classes: {np.round(r, 3).tolist()}; S alone R_jZ "
                         f"{np.round(s_only[0]['rjz'], 3).tolist()}")
    # the embedding and the random cones
    lines.append(f"random cones (bulk_bg, R = 0.3, |eta| < 1.6): {len(rc)}, delta p_T mean "
                 f"{np.mean(rc) if len(rc) else np.nan:+.2f} GeV, sigma_RC {sig_rc:.2f} GeV "
                 f"(CMS's PbPb N {N:.2f} GeV; 'match' smears with sqrt(N^2 - sigma_RC^2) = "
                 f"{np.sqrt(max(N ** 2 - sig_rc ** 2, 0)):.2f} GeV)")
    ref = emb = None
    if "emb_ref_pt" in Jb and len(ak.flatten(Jb["emb_ref_pt"])):
        ref = ak.to_numpy(ak.flatten(Jb["emb_ref_pt"]))
        emb = ak.to_numpy(ak.flatten(Jb["emb_pt"]))
        for lo, hi in EMB_BINS:
            m = (ref >= lo) & (ref < hi)
            okm = m & np.isfinite(emb)
            d = emb[okm] - ref[okm]
            if m.sum():
                lines.append(f"embedding {lo}-{hi} GeV: {int(m.sum())} jets, matched "
                             f"{okm.sum() / m.sum():.3f}, <p_T,emb - p_T,ref> {np.mean(d):+.2f}, "
                             f"sd {np.std(d):.2f} GeV")
    lines.append("")

    # ── the figures, one per step: pp (top) and Pb+Pb (bottom) against CMS
    xs = {"phi": 0.5 * (PHI_EDGES[1:] + PHI_EDGES[:-1]), "x": 0.5 * (X_EDGES[1:] + X_EDGES[:-1]),
          "mx": np.array([45, 55, 70, 100.0]), "rjz": np.array([45, 55, 70, 100.0])}
    titles = {1: "step 1, the reference: the fragments' jets (no background, no medium response)",
              2: "step 2: S − B_own (the Z's own background, the model's twin)",
              3: "step 3: S − B_mixed (bulk_bg of the backgrounds closest in N_coll, as CMS)"}
    os.makedirs(a.outdir, exist_ok=True)
    pp_figure(a, res, xs, lines, plt)
    for st, variants in steps.items():
        lines.append(f"Step {st}: {titles[st].split(': ', 1)[1]}")
        fig, ax = figdata.subplots(2, 4, figsize=(20, 8.5))
        for col, obs in enumerate(OBS):
            ipp, ipb = TABLES[obs]
            for row, (ti, keys, lab) in enumerate((
                    (ipp, ["pp, PbPb-smeared"], "CMS pp, smeared to PbPb"),
                    (ipb, list(variants), "CMS PbPb 0–30 %"))):
                y, e = cms(a.hepdata, ti)
                axr = ax[row, col]
                axr.errorbar(xs[obs], y, e, fmt="o", color="k", ms=4, label=lab)
                for j, k in enumerate(keys):
                    m, me = (res[k] if row == 0 else variants[k])
                    m, me = m[obs], me[obs]
                    axr.errorbar(xs[obs] * (1 + 0.008 * (j - 1.5)), m, me, fmt="s--", ms=3.5,
                                 mfc="none", lw=0.8, label=f"model {k}")
                    chi, nb = chi2(m, me, y, e)
                    if row == 1:
                        lines.append(f"  {obs:4s} {k:32s} vs data{ti}: chi2 {chi:6.1f}/{nb}"
                                     f"   model {fmt(m)}")
                axr.set_title(LABELS[obs])
                if obs == "phi":
                    axr.set_yscale("log")
                axr.legend(fontsize=6)
        fig.suptitle(f"Z+jet, CMS 1702.01060: pp (top, smeared to PbPb's resolution) and "
                     f"Pb+Pb 0–30 % (bottom), {titles[st]}", y=0.995)
        fig.tight_layout()
        figdata.savefig(fig, os.path.join(a.outdir, f"zjet_jets_step{st}{a.tag}.png"), dpi=100)
        plt.close(fig)
    # the wake's effect: step 2 against the same on the event without the wake (unsmeared)
    if NW_flat is not None and B_own is not None:
        c_s, c_nw = counts(S_flat, B_own, "none"), counts(NW_flat, B_own, "none")
        nowake = {"S − B_own, none, no wake": with_errors(c_nw, groups, a.blocks)}
        if B_mix is not None:
            nowake["S − B_mixed, none, no wake"] = result(NW_flat, B_mix, "none")
        wake = diff_with_errors(c_s, c_nw, groups, a.blocks)
        lines.append("Without the wake: steps 2 and 3 on the embedded event (bulk_bg k + the same "
                     "fragments), unsmeared")
        fw, aw = figdata.subplots(2, 4, figsize=(20, 8.5))
        for col, obs in enumerate(OBS):
            ti = TABLES[obs][1]
            y, e = cms(a.hepdata, ti)
            top = aw[0, col]
            top.errorbar(xs[obs], y, e, fmt="o", color="k", ms=4, label="CMS PbPb 0–30 %")
            shown = [("step 2: S − B_own, none", steps[2]["S − B_own, none"], "C0"),
                     ("step 2 without the wake", nowake["S − B_own, none, no wake"], "C1"),
                     ("step 1: fragments, none", steps[1]["fragments, none"], "C2")]
            for j, (lab, (m, me), colr) in enumerate(shown):
                top.errorbar(xs[obs] * (1 + 0.008 * (j - 1)), m[obs], me[obs], fmt="s--",
                             ms=3.5, mfc="none", lw=0.8, color=colr, label=f"model {lab}")
            for k, (m, me) in nowake.items():
                chi, nb = chi2(m[obs], me[obs], y, e)
                lines.append(f"  {obs:4s} {k:32s} vs data{ti}: chi2 {chi:6.1f}/{nb}"
                             f"   model {fmt(m[obs])}")
            top.set_title(LABELS[obs])
            if obs == "phi":
                top.set_yscale("log")
            top.legend(fontsize=6)
            bot = aw[1, col]
            bot.errorbar(xs[obs], wake[0][obs], wake[1][obs], fmt="s", ms=4, color="C3",
                         label="the wake: step 2 − step 2 without the wake")
            bot.axhline(0, color="0.5", lw=0.8)
            bot.set_xlabel({"phi": "Δφ_jZ", "x": "x_jZ", "mx": "p_T^Z [GeV]",
                            "rjz": "p_T^Z [GeV]"}[obs])
            bot.legend(fontsize=6)
        lines.append("The wake's effect: step 2 - step 2 without the wake (S - B_own, unsmeared; "
                     "jackknife of the difference)")
        for obs in OBS:
            lines.append(f"  {obs:4s} {fmt(wake[0][obs])} ± {fmt(wake[1][obs])}")
        fw.suptitle("Z+jet, Pb+Pb 0–30 %: the wake's effect (top: step 2 with and without the "
                    "wake, step 1; bottom: their difference)", y=0.995)
        fw.tight_layout()
        figdata.savefig(fw, os.path.join(a.outdir, f"zjet_jets_wake{a.tag}.png"), dpi=100)
        plt.close(fw)
    # the embedding figure
    fe, ae = figdata.subplots(1, 2, figsize=(12, 4.2))
    if len(rc):
        ae[0].hist(rc, bins=60, histtype="step", density=True, label="random cones in bulk_bg")
        xx = np.linspace(rc.min(), rc.max(), 200)
        ae[0].plot(xx, np.exp(-xx ** 2 / (2 * N ** 2)) / np.sqrt(2 * np.pi) / N, "k--",
                   label=f"Gauss, CMS N = {N} GeV")
        ae[0].set_xlabel("δp_T = Σp_T − ρπR² [GeV]")
        ae[0].legend(fontsize=8)
    if ref is not None:
        ae[1].plot(ref, emb - ref, ".", ms=2, alpha=0.4)
        ae[1].axhline(0, color="0.5", lw=0.6)
        ae[1].set_xlabel("p_T of the fragments' jet [GeV]")
        ae[1].set_ylabel("p_T,embedded − p_T [GeV]")
    fe.tight_layout()
    figdata.savefig(fe, os.path.join(a.outdir, f"zjet_jets_embedding{a.tag}.png"), dpi=100)
    plt.close(fe)
    text = "\n".join(lines) + "\n"
    with open(os.path.join(a.outdir, f"zjet_jets_summary{a.tag}.txt"), "w") as f:
        f.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
