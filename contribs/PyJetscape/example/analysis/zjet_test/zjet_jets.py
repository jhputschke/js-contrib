#!/usr/bin/env python3
"""Z+jet against CMS 1702.01060 (HEPData ins1512112): the final stage of the jets of a Pb+Pb and a
pp sample, from either map step (example/analysis_root/zjet/zjets or zjet_jets_map.py).

    python zjet_jets.py --pbpb data/jets_cxx/pbpb_campaign --pp data/jets_cxx/vacuum_all
    python zjet_jets.py ... --tag _c01 --blocks 100

Z: an opposite-sign lepton pair with 70 < m < 110 GeV, p_T^Z > 40 GeV, muons |eta| < 2.4,
electrons |eta| < 2.5 outside 1.44-1.57 (both p_T > 20 from the generator).  Jets: anti-k_T
R = 0.3 (Pb+Pb with rho A subtracted in the map step), |eta| < 1.6, Delta R > 0.4 from the Z's
leptons, p_T > 30 GeV after the smearing.  Weights: window cross section / events (x N_coll for
Pb+Pb), every oversample 1 / n_samples.

CMS's jet resolution (Appendix A), sigma / p_T = sqrt(C^2 + S^2 / p_T + N^2 / p_T^2), C 0.061;
pp S 0.95, N 0.001; PbPb 0-30 % S 1.24, N 8.08 (N: the underlying event's fluctuations).
The pp model is smeared with pp's resolution (against CMS's unsmeared pp x_jZ, table 9) and
with PbPb's (against CMS's smeared-pp tables 1, 3, 5, 7).  The Pb+Pb model (tables 2, 4, 6, 8)
in four variants, the smearing being optional:
  none      the subtracted particle-level jets
  detector  C and S of PbPb only: the model's own background fluctuations stand for N
  full      C, S and N: N on top of the model's fluctuations (double counts them)
  match     C, S and sqrt(N^2 - sigma_RC^2): N reduced by the model's own fluctuations, the
            random-cone width sigma_RC of the map step, so that the total matches CMS's
Observables: Delta phi_jZ and x_jZ (p_T^Z > 60), <x_jZ> and R_jZ in p_T^Z classes (Delta phi
> 7 pi / 8), per Z.  Errors: a jackknife over blocks of hydro backgrounds (Pb+Pb; blocks of
events with a single background, which then underestimates them) and of events (pp).  The
map step's embedding (the fragments in bulk_bg) and random cones: the subtraction's offset and
the model's background fluctuations, against CMS's N.
-> figures/zjet_jets{tag}.png, zjet_jets_embedding{tag}.png, zjet_jets_summary{tag}.txt."""
import argparse
import glob
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

C = 0.061
RES = {"pp": (0.95, 0.001), "PbPb": (1.24, 8.08)}
PBPB_SMEAR = ("none", "detector", "full", "match")
PHI_EDGES = np.array([0, np.pi / 5, 2 * np.pi / 5, 3 * np.pi / 5, 7 * np.pi / 10, 4 * np.pi / 5,
                      9 * np.pi / 10, 19 * np.pi / 20, np.pi])
X_EDGES = np.linspace(0, 2, 11)
ZPT_BINS = [(40, 50), (50, 60), (60, 80), (80, 120)]
EMB_BINS = [(10, 20), (20, 30), (30, 50), (50, 100)]


def load_jets(directory):
    """The map step's files of a directory -> (events dict, jets dict of flat names:
    jet_pt, emb_ref_pt, rc_dpt, ...), concatenated; from either map step (the Python one's
    groups come back as records, the C++ one's as vectors)."""
    import awkward as ak
    import uproot
    paths = sorted(glob.glob(os.path.join(directory, "*_jets.root")))
    if not paths:
        raise FileNotFoundError(f"no *_jets.root in {directory}")
    evs, jets = [], []
    for p in paths:
        with uproot.open(p) as f:
            evs.append(f["jet_events"].arrays())
            jets.append(f["jets"].arrays())
    ev = ak.concatenate(evs)
    jt = ak.concatenate(jets)
    E = {k: ak.to_numpy(ev[k]) for k in ev.fields}
    J = {}
    for k in jt.fields:
        if k in ("njet", "nemb", "nrc"):
            continue
        x = jt[k]
        if x.fields:                                   # a record of the Python map step
            for sub in x.fields:
                J[f"{k}_{sub}"] = x[sub]
        else:
            J[k] = x
    for k in ("event", "sample", "frag_sample"):
        J[k] = ak.to_numpy(J[k])
    return E, J


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


def per_event(E, J, pt_smeared, sel, w):
    """Per selected event its contributions (weighted): N_Z (p_T^Z > 60, per p_T^Z class), the
    Delta phi and x_jZ histograms, the pairs and x sums per class -> dict of (n_events, ...)."""
    import awkward as ak
    idx = {g: i for i, g in enumerate(E["event"])}
    ev = np.array([idx[g] for g in J["event"]])
    n = len(E["event"])
    nzb = len(ZPT_BINS)
    out = {"nz": np.zeros(n), "nzb": np.zeros((n, nzb)), "phi": np.zeros((n, len(PHI_EDGES) - 1)),
           "x": np.zeros((n, len(X_EDGES) - 1)), "pairs": np.zeros((n, nzb)),
           "xsum": np.zeros((n, nzb))}
    zpt = E["z_pt"]
    for i in range(n):
        if sel[i]:
            out["nz"][i] = w[i] * (zpt[i] > 60)
            for b, (lo, hi) in enumerate(ZPT_BINS):
                out["nzb"][i, b] = w[i] * (lo <= zpt[i] < hi)
    counts = ak.to_numpy(ak.num(J["jet_pt"]))
    row = np.repeat(np.arange(len(counts)), counts)
    e = ev[row]
    eta = ak.to_numpy(ak.flatten(J["jet_eta"]))
    phi = ak.to_numpy(ak.flatten(J["jet_phi"]))
    ok = sel[e] & (np.abs(eta) < 1.6) & (pt_smeared > 30)
    for l in (1, 2):
        ok &= np.hypot(eta - E[f"lep{l}_eta"][e], dphi(phi, E[f"lep{l}_phi"][e])) > 0.4
    wj = w[e] / E["n_samples"][e]
    dp = dphi(phi, E["z_phi"][e])
    x = pt_smeared / zpt[e]
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
                "mx": s["xsum"] / s["pairs"], "rjz": s["pairs"] / s["nzb"]}


def with_errors(c, groups, n_blocks):
    allk = np.ones(len(groups), bool)
    v = observables(c, allk)
    ug = np.unique(groups)
    blocks = [b for b in np.array_split(ug, min(n_blocks, len(ug))) if len(b)]
    if len(blocks) < 2:
        return v, {k: np.full_like(x, np.nan) for k, x in v.items()}
    reps = [observables(c, ~np.isin(groups, b)) for b in blocks]
    G = len(blocks)
    err = {k: np.sqrt((G - 1) / G * np.nansum((np.array([r[k] for r in reps]) - v[k]) ** 2, axis=0))
           for k in v}
    return v, err


def smear(pt, S, N, rng, n_extra=None):
    """CMS's resolution; ``n_extra`` replaces N (the 'match' variant)."""
    N = N if n_extra is None else n_extra
    sig = np.sqrt(C ** 2 + S ** 2 / pt + N ** 2 / pt ** 2)
    return pt * (1 + sig * rng.standard_normal(len(pt)))


def cms(hep, i):
    import yaml
    with open(os.path.join(hep, f"data{i}.yaml")) as f:
        d = yaml.safe_load(f)
    v = d["dependent_variables"][0]["values"]
    y = np.array([float(b["value"]) for b in v])
    e = np.array([np.hypot(*[float(x["symerror"]) for x in b["errors"]]) for b in v])
    return y, e


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pbpb", required=True, help="the map step's Pb+Pb *_jets.root")
    ap.add_argument("--pp", required=True, help="the map step's vacuum *_jets.root")
    ap.add_argument("--hepdata", default=os.path.join(HERE, "data", "hepdata_zjet",
                                                      "HEPData-ins1512112-v1-yaml"))
    ap.add_argument("--outdir", default=os.path.join(HERE, "figures"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--blocks", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7, help="the smearing's random numbers")
    a = ap.parse_args()
    import awkward as ak
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(a.seed)
    lines = []
    res = {}
    # pp
    Ep, Jp = load_jets(a.pp)
    selp = z_selection(Ep)
    wp = weights(Ep, selp, ncoll=False)
    ptp = ak.to_numpy(ak.flatten(Jp["jet_pt"]))
    for key, (S, N) in (("pp, pp-smeared", RES["pp"]), ("pp, PbPb-smeared", RES["PbPb"])):
        c = per_event(Ep, Jp, smear(ptp, S, N, rng), selp, wp)
        res[key] = with_errors(c, Ep["event"], a.blocks)
    lines.append(f"pp: {int(selp.sum())} Z's (CMS 1702.01060's cuts) of {len(selp)} in {a.pp}")
    # Pb+Pb
    Eb, Jb = load_jets(a.pbpb)
    selb = z_selection(Eb)
    wb = weights(Eb, selb, ncoll=True)
    ptb = ak.to_numpy(ak.flatten(Jb["jet_pt"]))
    rc = ak.to_numpy(ak.flatten(Jb["rc_dpt"])) if "rc_dpt" in Jb else np.zeros(0)
    sig_rc = float(np.std(rc)) if len(rc) else 0.0
    S, N = RES["PbPb"]
    n_bg = len(np.unique(Eb["bg"][selb]))
    groups = Eb["bg"] if n_bg >= 2 else Eb["event"]
    for v in PBPB_SMEAR:
        pt = {"none": ptb, "detector": None, "full": None, "match": None}[v]
        if v == "detector":
            pt = smear(ptb, S, N, rng, n_extra=0.0)
        elif v == "full":
            pt = smear(ptb, S, N, rng)
        elif v == "match":
            pt = smear(ptb, S, N, rng, n_extra=np.sqrt(max(N ** 2 - sig_rc ** 2, 0.0)))
        c = per_event(Eb, Jb, pt, selb, wb)
        res[f"PbPb, {v}"] = with_errors(c, groups, a.blocks)
    lines.append(f"Pb+Pb: {int(selb.sum())} Z's of {len(selb)} on {n_bg} hydro background(s) "
                 f"in {a.pbpb}" + ("" if n_bg >= 2 else
                                   " (jackknife over events: errors underestimated)"))
    # the map step's embedding and random cones
    lines.append(f"random cones (bulk_bg, R = 0.3, |eta| < 1.6): {len(rc)}, delta p_T mean "
                 f"{np.mean(rc) if len(rc) else np.nan:+.2f} GeV, sigma_RC {sig_rc:.2f} GeV "
                 f"(CMS's PbPb N {N:.2f} GeV; 'match' smears with sqrt(N^2 - sigma_RC^2) = "
                 f"{np.sqrt(max(N ** 2 - sig_rc ** 2, 0)):.2f} GeV)")
    if "emb_ref_pt" in Jb and len(ak.flatten(Jb["emb_ref_pt"])):
        ref = ak.to_numpy(ak.flatten(Jb["emb_ref_pt"]))
        emb = ak.to_numpy(ak.flatten(Jb["emb_pt"]))
        for lo, hi in EMB_BINS:
            m = (ref >= lo) & (ref < hi)
            ok = m & np.isfinite(emb)
            d = emb[ok] - ref[ok]
            if m.sum():
                lines.append(f"embedding {lo}-{hi} GeV: {int(m.sum())} jets, matched "
                             f"{ok.sum() / m.sum():.3f}, <p_T,emb - p_T,ref> {np.mean(d):+.2f}, "
                             f"sd {np.std(d):.2f} GeV")
    lines.append("")
    # against CMS
    tables = {"phi": (1, 2), "x": (3, 4), "mx": (5, 6), "rjz": (7, 8)}
    labels = {"phi": "Δφ_jZ, p_T^Z > 60", "x": "x_jZ, p_T^Z > 60, Δφ > 7π/8",
              "mx": "⟨x_jZ⟩ vs p_T^Z", "rjz": "R_jZ vs p_T^Z"}
    xs = {"phi": 0.5 * (PHI_EDGES[1:] + PHI_EDGES[:-1]), "x": 0.5 * (X_EDGES[1:] + X_EDGES[:-1]),
          "mx": np.array([45, 55, 70, 100.0]), "rjz": np.array([45, 55, 70, 100.0])}
    fig, ax = plt.subplots(2, 4, figsize=(20, 8.5))
    for col, obs in enumerate(("phi", "x", "mx", "rjz")):
        ipp, ipb = tables[obs]
        for row, (ti, keys, lab) in enumerate((
                (ipp, ["pp, PbPb-smeared"], "CMS pp, smeared to PbPb"),
                (ipb, [f"PbPb, {v}" for v in PBPB_SMEAR], "CMS PbPb 0–30 %"))):
            y, e = cms(a.hepdata, ti)
            axr = ax[row, col]
            axr.errorbar(xs[obs], y, e, fmt="o", color="k", ms=4, label=lab)
            for j, k in enumerate(keys):
                m, me = res[k][0][obs], res[k][1][obs]
                axr.errorbar(xs[obs] * (1 + 0.008 * (j - 1.5)), m, me, fmt="s--", ms=3.5,
                             mfc="none", lw=0.8, label=f"model {k}")
                ee = np.where(np.isfinite(me), me, 0.0)
                good = np.isfinite(m) & np.isfinite(y) & (e ** 2 + ee ** 2 > 0)
                chi = float(np.sum((m[good] - y[good]) ** 2 / (e[good] ** 2 + ee[good] ** 2)))
                lines.append(f"{obs:4s} {k:18s} vs data{ti}: chi2 {chi:6.1f}/{int(good.sum())}"
                             f"   model {np.array2string(m, precision=3, separator=', ')}")
            axr.set_title(labels[obs])
            if obs == "phi":
                axr.set_yscale("log")
            axr.legend(fontsize=6)
        y9, e9 = cms(a.hepdata, 9)
        if obs == "x":
            m = res["pp, pp-smeared"][0]["x"]
            good = np.isfinite(m) & (e9 > 0)
            c9 = float(np.sum((m[good] - y9[good]) ** 2 / e9[good] ** 2))
            lines.append(f"x    pp, pp-smeared     vs data9: chi2 {c9:6.1f}/{int(good.sum())}")
    fig.suptitle("Z+jet, CMS 1702.01060: pp (top, smeared to PbPb's resolution) and Pb+Pb 0–30 % "
                 "(bottom; subtracted jets, four smearings)", y=0.995)
    fig.tight_layout()
    os.makedirs(a.outdir, exist_ok=True)
    fig.savefig(os.path.join(a.outdir, f"zjet_jets{a.tag}.png"), dpi=100)
    plt.close(fig)
    # embedding figure
    fe, ae = plt.subplots(1, 2, figsize=(12, 4.2))
    if len(rc):
        ae[0].hist(rc, bins=60, histtype="step", density=True, label="random cones in bulk_bg")
        xx = np.linspace(rc.min(), rc.max(), 200)
        ae[0].plot(xx, np.exp(-xx ** 2 / (2 * N ** 2)) / np.sqrt(2 * np.pi) / N, "k--",
                   label=f"Gauss, CMS N = {N} GeV")
        ae[0].set_xlabel("δp_T = Σp_T − ρπR² [GeV]")
        ae[0].legend(fontsize=8)
    if "emb_ref_pt" in Jb and len(ak.flatten(Jb["emb_ref_pt"])):
        ae[1].plot(ref, emb - ref, ".", ms=2, alpha=0.4)
        ae[1].axhline(0, color="0.5", lw=0.6)
        ae[1].set_xlabel("p_T of the fragments' jet [GeV]")
        ae[1].set_ylabel("p_T,embedded − p_T [GeV]")
    fe.tight_layout()
    fe.savefig(os.path.join(a.outdir, f"zjet_jets_embedding{a.tag}.png"), dpi=100)
    plt.close(fe)
    text = "\n".join(lines) + "\n"
    with open(os.path.join(a.outdir, f"zjet_jets_summary{a.tag}.txt"), "w") as f:
        f.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
