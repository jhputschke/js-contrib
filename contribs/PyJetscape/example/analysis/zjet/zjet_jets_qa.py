#!/usr/bin/env python3
"""The jet QA of the Z+jet analysis: the jet reconstruction's diagnostics from the map step's
QA trees (qa_jets, qa_bg_jets: every jet of the first --qa-samples samples, no cut on the
subtracted p_T) and its jets tree (embedding, random cones), from either map step
(example/analysis_root/zjet/zjets or zjet_jets_map.py).

    python zjet_jets_qa.py --pbpb data/jets_cxx/pbpb_campaign --pp data/jets_cxx/vacuum_all
    python zjet_jets_qa.py ... [--hadron-qa data/qa_py/pbpb_campaign] [--n-nearest 20] --tag _c00

The Z events: CMS 1702.01060's selection and weights (zjet_jets.py: window cross section /
events x N_coll for Pb+Pb, normalised per Z).  Jets per Z: S the Z events' jets (bulk_jet +
fragments), B_own the jets of the Z's own background (bulk_bg, the same oversample index),
B_mixed the jets of the --n-nearest other backgrounds closest in N_coll (zjet_jets.py's step 3,
each background rotated by zjet_jets.py's random angle; --no-rotate as produced), pp the
vacuum's.  The Z+jet cuts where a panel says so: |eta| < 1.6, Delta R > 0.4 from the Z's
leptons, the subtracted p_T > 30 GeV (no smearing here).

  qa_jets_subtraction  p_T,raw, rho A, p_T = p_T,raw - rho A (the negative tail), p_T,raw vs
                       rho A, the fake fraction B / S, S - B_own against pp
  qa_jets_area_rho     A / pi R^2 and against p_T, rho per sample, rho vs N_coll, rho vs eta,
                       rho of the Z event - rho of its own background
  qa_jets_response     random cones (delta p_T, sigma_RC vs N_coll and eta), the embedding's
                       response and resolution, matching efficiency, Delta R of the pairs
  qa_jets_geometry     eta, Delta phi to the Z, phi, jets per Z, Delta R to the Z's leptons
  qa_jets_constituents constituents, the leading one's p_T fraction, m / p_T
  qa_jets_mixing       B_mixed's backgrounds against the own one: N_coll, rho, the jet spectra
-> figures/qa_jets_*{tag}.png, qa_jets_summary{tag}.txt."""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))                        # jetcorr
from jetcorr import runs  # noqa: E402
from zjet_jets import (SEED, background_angles, dphi, load_jets,  # noqa: E402
                       load_qa_jets, rotation_seed, weights, z_selection)

R_JET = 0.3
AREA = np.pi * R_JET ** 2
ETA_CUT, PT_CUT, DR_LEP = 1.6, 30.0, 0.4
PT_SUB = np.linspace(-40, 120, 81)
PT_RAW = np.linspace(0, 160, 81)
RHO_A = np.linspace(0, 60, 61)
PT_CLASSES = ((-np.inf, 10), (10, 30), (30, np.inf))


def flat(T, rows, row_w, row_ev, names=("pt_raw", "pt", "eta", "phi", "area", "rho", "m",
                                         "nconst", "lead_pt"), row_dphi=None):
    """The jets of the rows ``rows`` of a QA/jets tree -> dict of flat arrays (the jet_* columns,
    their event row ``e`` and weight ``w``); ``row_dphi`` rotates each row's jets."""
    import awkward as ak
    out = {}
    first = T[f"jet_{names[0]}"][rows]
    counts = ak.to_numpy(ak.num(first))
    rep = np.repeat(np.arange(len(rows)), counts)
    for nm in names:
        out[nm] = ak.to_numpy(ak.flatten(T[f"jet_{nm}"][rows])).astype(float)
    if row_dphi is not None:
        out["phi"] = out["phi"] + np.asarray(row_dphi, float)[rep]
    out["e"] = np.asarray(row_ev)[rep]
    out["w"] = np.asarray(row_w, float)[rep]
    return out


def cuts(E, j, pt=True):
    """The Z+jet jet cuts (|eta| < 1.6, Delta R > 0.4 from the Z's leptons, p_T > 30)."""
    e = j["e"]
    ok = np.abs(j["eta"]) < ETA_CUT
    for lep in (1, 2):
        ok &= np.hypot(j["eta"] - E[f"lep{lep}_eta"][e],
                       dphi(j["phi"], E[f"lep{lep}_phi"][e])) > DR_LEP
    if pt:
        ok &= j["pt"] > PT_CUT
    return ok


def hist(x, edges, w, norm):
    h, _ = np.histogram(x, edges, weights=w)
    return h / norm / np.diff(edges)


def profile(x, y, edges, w=None):
    """Weighted mean and standard deviation of y in bins of x."""
    w = np.ones_like(x) if w is None else w
    b = np.digitize(x, edges) - 1
    m, s = np.full(len(edges) - 1, np.nan), np.full(len(edges) - 1, np.nan)
    for i in range(len(edges) - 1):
        k = (b == i) & np.isfinite(y)
        if w[k].sum() > 0:
            m[i] = np.average(y[k], weights=w[k])
            s[i] = np.sqrt(np.average((y[k] - m[i]) ** 2, weights=w[k]))
    return m, s


def mid(edges):
    return 0.5 * (edges[1:] + edges[:-1])


def per_row_mean(T, rows, field, eta_max):
    """Per row of a QA tree: the mean of jet_<field> over its jets with |eta| < eta_max."""
    import awkward as ak
    x, eta = T[f"jet_{field}"][rows], T["jet_eta"][rows]
    return ak.to_numpy(ak.fill_none(ak.mean(x[np.abs(eta) < eta_max], axis=1), np.nan))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pbpb", required=True, help="the map step's Pb+Pb *_jets.root")
    ap.add_argument("--pp", required=True, help="the map step's vacuum *_jets.root")
    ap.add_argument("--hadron-qa", default=None,
                    help="qa.py map's / zqa's output of the same Pb+Pb sample: rho against "
                         "dN_ch/deta of the background")
    ap.add_argument("--n-nearest", type=int, default=20,
                    help="B_mixed: the other backgrounds closest in N_coll (zjet_jets.py's)")
    ap.add_argument("--no-rotate", dest="rotate", action="store_false",
                    help="B_mixed: keep the backgrounds as produced (event planes all near "
                         "phi = 0) instead of zjet_jets.py's random rotations (the default)")
    ap.add_argument("--rotate-seed", type=int, default=None,
                    help="seed of the rotations (default: zjet_jets.py's, from its --seed 7)")
    ap.add_argument("--run", default="zjet_setup", help="runs/<run>/figures is the default --outdir (jetcorr.runs)")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    a.outdir = a.outdir or runs.figures(a.run)
    import awkward as ak
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from jetcorr import figdata

    os.makedirs(a.outdir, exist_ok=True)
    lines = []
    # ── the Z events and their QA jets
    Eb, Jb, Bb = load_jets(a.pbpb)
    Qb, QB = load_qa_jets(a.pbpb)
    Ep, Jp, _ = load_jets(a.pp)
    Qp, _ = load_qa_jets(a.pp)
    selb, selp = z_selection(Eb), z_selection(Ep)
    wb, wp = weights(Eb, selb, ncoll=True), weights(Ep, selp, ncoll=False)
    Wb, Wp = wb[selb].sum(), wp[selp].sum()
    row_of_b = {g: i for i, g in enumerate(Eb["event"])}
    row_of_p = {g: i for i, g in enumerate(Ep["event"])}

    def z_rows(Q, row_of, sel, w):
        """The QA rows of the selected Z events, their event row and weight per Z."""
        ev = np.array([row_of[g] for g in Q["event"]])
        n_q = np.bincount(ev, minlength=len(sel))
        keep = np.flatnonzero(sel[ev])
        return keep, ev[keep], w[ev[keep]] / n_q[ev[keep]]

    rs, es, ws = z_rows(Qb, row_of_b, selb, wb)
    S = flat(Qb, rs, ws, es)
    rp, ep, wq = z_rows(Qp, row_of_p, selp, wp)
    P = flat(Qp, rp, wq, ep)
    zsel = np.flatnonzero(selb)
    bg_of_z = Eb["bg"][zsel]
    lines.append(f"Pb+Pb: {len(zsel)} Z's (CMS 1702.01060's cuts), {len(np.unique(Qb['sample']))} "
                 f"QA sample(s) per event, {len(np.unique(Eb['bg'][selb]))} background(s); "
                 f"pp: {int(selp.sum())} Z's")

    # B_own and B_mixed per Z: the background QA rows, each Z's weight spread over them
    B_own = B_mix = None
    nn_info = None
    if QB:
        rows_of_bg = {int(b): np.flatnonzero(QB["bg"] == b) for b in np.unique(QB["bg"])}
        bgs = np.array(sorted(rows_of_bg))
        ncoll_bg = np.array([QB["n_coll"][rows_of_bg[b][0]] for b in bgs], float)
        orow, oev, ow, mrow, mev, mw, dnc = [], [], [], [], [], [], []
        for e_ in zsel:
            b = int(Eb["bg"][e_])
            if b in rows_of_bg:
                r_ = rows_of_bg[b]
                orow.append(r_)
                oev.append(np.full(len(r_), e_))
                ow.append(np.full(len(r_), wb[e_] / len(r_)))
            others = bgs != b
            if others.any():
                near = bgs[others][np.argsort(np.abs(ncoll_bg[others] - Eb["n_coll"][e_]),
                                              kind="stable")[:a.n_nearest]]
                r_ = np.concatenate([rows_of_bg[int(x)] for x in near])
                mrow.append(r_)
                mev.append(np.full(len(r_), e_))
                mw.append(np.full(len(r_), wb[e_] / len(r_)))
                dnc.append(np.abs(ncoll_bg[np.searchsorted(bgs, near)] - Eb["n_coll"][e_]))
        if orow:
            B_own = flat(QB, np.concatenate(orow), np.concatenate(ow), np.concatenate(oev))
        if mrow:
            mrow, mev = np.concatenate(mrow), np.concatenate(mev)
            turn = None
            if a.rotate:
                angle = background_angles(np.concatenate([QB["bg"], Eb["bg"]]),
                                          rotation_seed(SEED, a.rotate_seed))
                turn = angle(QB["bg"][mrow]) - angle(Eb["bg"][mev])
            B_mix = flat(QB, mrow, np.concatenate(mw), mev, row_dphi=turn)
            lines.append("B_mixed: the backgrounds " + (
                "rotated by zjet_jets.py's random angles" if a.rotate else
                "as produced (--no-rotate): event planes all near phi = 0"))
            nn_info = np.concatenate(dnc)
    sets = [("S (Z events)", S, Wb, "C3"), ("B_own", B_own, Wb, "C0"),
            ("B_mixed", B_mix, Wb, "C1"), ("pp", P, Wp, "k")]
    sets = [s for s in sets if s[1] is not None]

    # ── 1. the subtraction
    fig, ax = figdata.subplots(2, 3, figsize=(17, 9))
    for lab, j, W, col in sets:
        acc = np.abs(j["eta"]) < ETA_CUT
        ax[0, 0].stairs(hist(j["pt_raw"][acc], PT_RAW, j["w"][acc], W), PT_RAW, color=col,
                        label=lab)
        if lab != "pp":
            ax[0, 1].stairs(hist(j["rho"][acc] * j["area"][acc], RHO_A, j["w"][acc], W), RHO_A,
                            color=col, label=lab)
        ax[0, 2].stairs(hist(j["pt"][acc], PT_SUB, j["w"][acc], W), PT_SUB, color=col,
                        label=lab + (" (no subtraction)" if lab == "pp" else ""))
    ax[0, 0].set(yscale="log", xlabel="p_T,raw [GeV] (|η| < 1.6)", ylabel="jets per Z per GeV",
                 title="raw jet p_T")
    ax[0, 1].set(yscale="log", xlabel="ρA [GeV]", ylabel="jets per Z per GeV",
                 title="the subtracted ρA")
    ax[0, 2].set(yscale="log", xlabel="p_T = p_T,raw − ρA [GeV]", ylabel="jets per Z per GeV",
                 title="subtracted p_T (no p_T cut: the negative tail)")
    for x in (0, PT_CUT):
        ax[0, 2].axvline(x, color="0.6", lw=0.6, ls=":")
    acc = np.abs(S["eta"]) < ETA_CUT
    hh = ax[1, 0].hist2d(S["rho"][acc] * S["area"][acc], S["pt_raw"][acc], [RHO_A, PT_RAW],
                         weights=S["w"][acc], cmap="viridis", cmin=1e-12,
                         norm=matplotlib.colors.LogNorm())
    fig.colorbar(hh[3], ax=ax[1, 0])
    ax[1, 0].plot(RHO_A, RHO_A + PT_CUT, "r--", lw=0.8, label="p_T = 30 GeV")
    ax[1, 0].set(xlabel="ρA [GeV]", ylabel="p_T,raw [GeV]", title="S: p_T,raw against ρA")
    ax[1, 0].legend(fontsize=7)
    edges = np.array([0, 5, 10, 15, 20, 25, 30, 40, 50, 70, 100, 150.0])
    fake = {}
    for (lab, j), ls in zip((("B_own", B_own), ("B_mixed", B_mix)), ("-", "--")):
        if j is None:
            continue
        for kind, back in (("inclusive", False), ("Δφ_jZ > 7π/8", True)):
            ms, mb = cuts(Eb, S, pt=False), cuts(Eb, j, pt=False)
            if back:
                ms &= dphi(S["phi"], Eb["z_phi"][S["e"]]) > 7 * np.pi / 8
                mb &= dphi(j["phi"], Eb["z_phi"][j["e"]]) > 7 * np.pi / 8
            hs = hist(S["pt"][ms], edges, S["w"][ms], Wb)
            hb = hist(j["pt"][mb], edges, j["w"][mb], Wb)
            with np.errstate(invalid="ignore", divide="ignore"):
                r = hb / hs
            fake[(lab, kind)] = r
            ax[1, 1].plot(mid(edges), r, "o" + ls, ms=3,
                          color="C0" if kind == "inclusive" else "C2", label=f"{lab}, {kind}")
    ax[1, 1].axvline(PT_CUT, color="0.6", lw=0.6, ls=":")
    ax[1, 1].set(yscale="log", xlabel="p_T [GeV] (Z+jet cuts but p_T)", ylabel="B / S",
                 title="the fake fraction: background jets / all jets")
    ax[1, 1].legend(fontsize=7)
    for lab, j, W, col in sets:
        m = cuts(Eb if lab != "pp" else Ep, j, pt=False)
        h = hist(j["pt"][m], edges, j["w"][m], W)
        if lab == "S (Z events)" and B_own is not None:
            mb = cuts(Eb, B_own, pt=False)
            h = h - hist(B_own["pt"][mb], edges, B_own["w"][mb], Wb)
            lab = "S − B_own"
        elif lab not in ("pp",):
            continue
        ax[1, 2].plot(mid(edges), h, "o-", ms=3, color=col, label=lab)
    ax[1, 2].axvline(PT_CUT, color="0.6", lw=0.6, ls=":")
    ax[1, 2].set(yscale="log", xlabel="p_T [GeV]", ylabel="jets per Z per GeV",
                 title="S − B_own against pp (Z+jet cuts but p_T)")
    ax[1, 2].legend(fontsize=7)
    for x in ax.flat[:3]:
        x.legend(fontsize=7)
    fig.suptitle("Z+jet jet QA: the ρA subtraction (QA samples, all jets with p_T,raw > 1 GeV)")
    fig.tight_layout()
    figdata.savefig(fig, os.path.join(a.outdir, f"qa_jets_subtraction{a.tag}.png"), dpi=90)
    plt.close(fig)
    for (lab, kind), r in fake.items():
        i = np.searchsorted(edges, PT_CUT)
        lines.append(f"fake fraction {lab:8s} {kind:13s}: B/S at 30-40 GeV {r[i]:.3f}, "
                     f"at 50-70 GeV {r[i + 2]:.3f}")
    for lab, j, W, col in sets:
        acc = np.abs(j["eta"]) < ETA_CUT
        neg = np.sum(j["w"][acc & (j["pt"] < 0)]) / max(np.sum(j["w"][acc]), 1e-300)
        lines.append(f"{lab:13s}: jets per Z (|eta| < 1.6, p_T,raw > 1) "
                     f"{np.sum(j['w'][acc]) / W:7.2f}, the fraction with p_T < 0 {neg:.3f}, "
                     f"jets per Z with p_T > 30 "
                     f"{np.sum(j['w'][acc & (j['pt'] > PT_CUT)]) / W:.3f}")

    # ── 2. area and rho
    fig, ax = figdata.subplots(2, 3, figsize=(17, 9))
    a_edges = np.linspace(0, 2, 51)
    for (lo, hi), col in zip(PT_CLASSES, ("C0", "C1", "C3")):
        m = (np.abs(S["eta"]) < ETA_CUT) & (S["pt"] >= lo) & (S["pt"] < hi)
        if m.any():
            ax[0, 0].stairs(np.histogram(S["area"][m] / AREA, a_edges, weights=S["w"][m],
                                         density=True)[0], a_edges, color=col,
                            label="S, p_T " + (f"< {hi:g}" if lo == -np.inf else
                                                 f"> {lo:g}" if hi == np.inf else
                                                 f"{lo:g}–{hi:g}") + " GeV")
    if B_own is not None:
        m = np.abs(B_own["eta"]) < ETA_CUT
        ax[0, 0].stairs(np.histogram(B_own["area"][m] / AREA, a_edges, weights=B_own["w"][m],
                                     density=True)[0], a_edges, color="0.4", ls="--",
                        label="B_own, all")
    ax[0, 0].axvline(1, color="0.6", lw=0.6, ls=":")
    ax[0, 0].set(xlabel="A / πR²", ylabel="density", title="jet area")
    ax[0, 0].legend(fontsize=7)
    pe = np.array([-20, -10, 0, 5, 10, 20, 30, 40, 60, 80, 120.0])
    for lab, j, col in (("S", S, "C3"), ("B_own", B_own, "C0")):
        if j is None:
            continue
        m = np.abs(j["eta"]) < ETA_CUT
        mu, sd = profile(j["pt"][m], j["area"][m] / AREA, pe, j["w"][m])
        ax[0, 1].errorbar(mid(pe), mu, sd, fmt="o-", ms=3, color=col, label=f"{lab} (± sd)")
    ax[0, 1].axhline(1, color="0.6", lw=0.6, ls=":")
    ax[0, 1].set(xlabel="p_T [GeV]", ylabel="⟨A⟩ / πR²", title="area against p_T")
    ax[0, 1].legend(fontsize=7)
    # rho per sample: the mean over its jets of rho(jet) (|eta| < 1.6)
    rho_S = per_row_mean(Qb, rs, "rho", ETA_CUT)
    rho_bg = None
    if QB:
        rho_bg = per_row_mean(QB, np.arange(len(QB["bg"])), "rho", ETA_CUT)
    both = np.concatenate([rho_S, rho_bg if rho_bg is not None else []])
    lo_, hi_ = np.nanpercentile(both, [0.5, 99.5]) if len(both) else (0, 1)
    r_edges = np.linspace(lo_ - 0.2 * (hi_ - lo_) - 1e-9, hi_ + 0.2 * (hi_ - lo_) + 1e-9, 50)
    ax[0, 2].hist(rho_S, r_edges, weights=ws, density=True, histtype="step", color="C3",
                  label="S: Z events")
    if rho_bg is not None:
        ax[0, 2].hist(rho_bg, r_edges, density=True, histtype="step", color="C0",
                      label="bulk_bg samples")
    ax[0, 2].set(xlabel="ρ [GeV per unit area] (per sample, the mean of ρ(jet), |η| < 1.6)",
                 ylabel="density", title="ρ per sample")
    ax[0, 2].legend(fontsize=7)
    # rho against N_coll, per background, over every bulk_bg sample of bg_jets
    rho_bg_all = {}
    if Bb:
        rho_all = per_row_mean(Bb, np.arange(len(Bb["bg"])), "rho", ETA_CUT)
        ub = np.unique(Bb["bg"])
        nc = np.array([Bb["n_coll"][Bb["bg"] == b][0] for b in ub], float)
        mu = np.array([np.nanmean(rho_all[Bb["bg"] == b]) for b in ub])
        sd = np.array([np.nanstd(rho_all[Bb["bg"] == b]) for b in ub])
        rho_bg_all = {int(b): m_ for b, m_ in zip(ub, mu)}
        ax[1, 0].errorbar(nc, mu, sd, fmt="o", ms=3, color="C0",
                          label="per background (± sd of its samples)")
        lines.append(f"rho per background: {len(ub)} backgrounds, N_coll {nc.min():.0f}-"
                     f"{nc.max():.0f}, <rho> {np.nanmean(mu):.1f} GeV, sd over samples "
                     f"{np.nanmean(sd):.2f} GeV")
        if a.hadron_qa:
            from qa import load
            H, _ = load(a.hadron_qa)
            dn = {}
            for g, b, n05 in zip(H["event"], H["bg"], H["nch05_bg"]):
                if g in row_of_b:
                    dn.setdefault(int(Eb["bg"][row_of_b[g]]), []).append(n05)
            ratio = [mu[i] / np.mean(dn[int(b)]) for i, b in enumerate(ub) if int(b) in dn]
            if ratio:
                lines.append(f"rho / (dN_ch/deta |eta| < 0.5 of bulk_bg): {np.mean(ratio):.4f} "
                             f"GeV (sd {np.std(ratio):.4f} over {len(ratio)} backgrounds)")
    ax[1, 0].set(xlabel="N_coll", ylabel="⟨ρ⟩ [GeV]", title="ρ against N_coll")
    ax[1, 0].legend(fontsize=7)
    ee = np.linspace(-2, 2, 21)
    for lab, j, col in (("S", S, "C3"), ("B_own", B_own, "C0")):
        if j is None:
            continue
        mu, sd = profile(j["eta"], j["rho"], ee, j["w"])
        ax[1, 1].errorbar(mid(ee), mu, sd, fmt="o-", ms=3, color=col, label=f"{lab} (± sd)")
    ax[1, 1].set(xlabel="η of the jet", ylabel="ρ(jet) [GeV]",
                 title="ρ against η (the strip |Δy| < 1)")
    ax[1, 1].legend(fontsize=7)
    # rho of the Z event - rho of its own background, the same sample
    if QB:
        key = {(int(b), int(k)): i for i, (b, k) in enumerate(zip(QB["bg"], QB["sample"]))}
        own = np.array([key.get((int(Eb["bg"][e_]), int(k)), -1)
                        for e_, k in zip(es, Qb["sample"][rs])])
        ok = own >= 0
        d = rho_S[ok] - rho_bg[own[ok]]
        if len(d):
            de = np.linspace(min(-1, np.nanpercentile(d, 0.5)), max(1, np.nanpercentile(d, 99.5)),
                             41)
            ax[1, 2].hist(d, de, weights=ws[ok], histtype="step", color="C3")
            mu_d = np.average(d, weights=ws[ok])
            sd_d = np.sqrt(np.average((d - mu_d) ** 2, weights=ws[ok]))
            n_eff = ws[ok].sum() ** 2 / (ws[ok] ** 2).sum()
            ax[1, 2].axvline(mu_d, color="C3", ls="--", lw=0.8, label=f"mean {mu_d:+.2f} GeV")
            ax[1, 2].legend(fontsize=7)
            lines.append(f"rho(Z event) - rho(own bulk_bg, same sample): mean {mu_d:+.3f} "
                         f"± {sd_d / np.sqrt(n_eff):.3f} GeV, sd {sd_d:.3f} GeV (<rho> of the Z "
                         f"events {np.average(rho_S, weights=ws):.2f} GeV): what the wake and the "
                         f"fragments add to rho")
            if rho_bg_all:
                d2 = np.array([rho_S[i] - rho_bg_all.get(int(Eb["bg"][e_]), np.nan)
                               for i, e_ in enumerate(es)])
                k2 = np.isfinite(d2)
                if k2.any():
                    m2 = np.average(d2[k2], weights=ws[k2])
                    s2 = np.sqrt(np.average((d2[k2] - m2) ** 2, weights=ws[k2]))
                    n2 = ws[k2].sum() ** 2 / (ws[k2] ** 2).sum()
                    lines.append(f"rho(Z event) - <rho> of its background over all bulk_bg "
                                 f"samples: mean {m2:+.3f} ± {s2 / np.sqrt(n2):.3f} GeV")
    ax[1, 2].set(xlabel="ρ(Z event) − ρ(its bulk_bg) [GeV]", ylabel="Z's (weighted)",
                 title="the wake's and the jet's contribution to ρ")
    m = (np.abs(S["eta"]) < ETA_CUT) & (S["pt"] > PT_CUT)
    if m.any():
        lines.append(f"area of the jets above 30 GeV (S): <A/piR^2> "
                     f"{np.average(S['area'][m] / AREA, weights=S['w'][m]):.3f}")
    fig.suptitle("Z+jet jet QA: area and ρ")
    fig.tight_layout()
    figdata.savefig(fig, os.path.join(a.outdir, f"qa_jets_area_rho{a.tag}.png"), dpi=90)
    plt.close(fig)

    # ── 3. random cones and the embedding (the jets tree, every sample)
    fig, ax = figdata.subplots(2, 3, figsize=(17, 9))
    rc = ak.to_numpy(ak.flatten(Jb["rc_dpt"])) if "rc_dpt" in Jb else np.zeros(0)
    if len(rc):
        ev_rc = np.repeat(np.array([row_of_b[g] for g in Jb["event"]]),
                          ak.to_numpy(ak.num(Jb["rc_dpt"])))
        eta_rc = ak.to_numpy(ak.flatten(Jb["rc_eta"]))
        sig = float(np.std(rc))
        de = np.linspace(rc.min(), rc.max(), 61)
        ax[0, 0].hist(rc, de, density=True, histtype="step", label="random cones in bulk_bg")
        xx = np.linspace(de[0], de[-1], 200)
        for s_, ls, lab in ((sig, "-", f"Gauss σ_RC = {sig:.2f}"),
                            (8.08, "--", "Gauss, CMS N = 8.08")):
            ax[0, 0].plot(xx, np.exp(-(xx - rc.mean()) ** 2 / (2 * s_ ** 2)) / np.sqrt(2 * np.pi)
                          / s_, "k" + ls, lw=0.8, label=lab)
        ax[0, 0].set(yscale="log", ylim=(1e-5, None), xlabel="δp_T = Σp_T − ρπR² [GeV]",
                     ylabel="density", title="random cones (log: the tails)")
        ax[0, 0].legend(fontsize=7)
        ub = np.unique(Eb["bg"][ev_rc])
        nc = np.array([Eb["n_coll"][ev_rc][Eb["bg"][ev_rc] == b][0] for b in ub], float)
        sb = np.array([np.std(rc[Eb["bg"][ev_rc] == b]) for b in ub])
        mb_ = np.array([np.mean(rc[Eb["bg"][ev_rc] == b]) for b in ub])
        ax[0, 1].plot(nc, sb, "o", ms=3, label="σ_RC")
        ax[0, 1].plot(nc, mb_, "s", ms=3, mfc="none", label="⟨δp_T⟩")
        ax[0, 1].axhline(8.08, color="0.5", ls="--", lw=0.8, label="CMS N")
        ax[0, 1].axhline(0, color="0.6", lw=0.6)
        ax[0, 1].set(xlabel="N_coll", ylabel="[GeV]", title="random cones per background")
        ax[0, 1].legend(fontsize=7)
        ee = np.linspace(-1.6, 1.6, 9)
        mu, sd = profile(eta_rc, rc, ee)
        ax[0, 2].plot(mid(ee), sd, "o-", ms=3, label="σ_RC")
        ax[0, 2].plot(mid(ee), mu, "s--", ms=3, mfc="none", label="⟨δp_T⟩")
        ax[0, 2].axhline(0, color="0.6", lw=0.6)
        ax[0, 2].set(xlabel="η of the cone", ylabel="[GeV]", title="random cones against η")
        ax[0, 2].legend(fontsize=7)
        tail = np.mean(np.abs(rc - rc.mean()) > 3 * sig)
        lines.append(f"random cones: {len(rc)}, <delta p_T> {rc.mean():+.2f} GeV, sigma_RC "
                     f"{sig:.2f} GeV, beyond 3 sigma {tail:.4f} (Gauss 0.0027), per background "
                     f"sigma_RC {sb.min():.2f}-{sb.max():.2f} GeV")
    if "emb_ref_pt" in Jb and len(ak.flatten(Jb["emb_ref_pt"])):
        ref = ak.to_numpy(ak.flatten(Jb["emb_ref_pt"]))
        emb = ak.to_numpy(ak.flatten(Jb["emb_pt"]))
        dr = ak.to_numpy(ak.flatten(Jb["emb_dr"]))
        pe = np.array([10, 15, 20, 25, 30, 40, 50, 70, 100, 150.0])
        ok = np.isfinite(emb)
        mu, sd = profile(ref[ok], (emb[ok] - ref[ok]) / ref[ok], pe)
        ax[1, 0].errorbar(mid(pe), mu, sd, fmt="o-", ms=3, label="mean ± sd")
        ax[1, 0].axhline(0, color="0.6", lw=0.6)
        ax[1, 0].set(xlabel="p_T of the fragments' jet [GeV]",
                     ylabel="(p_T,embedded − p_T) / p_T", title="the embedding: JES and JER")
        ax[1, 0].legend(fontsize=7)
        n_all, _ = np.histogram(ref, pe)
        n_ok, _ = np.histogram(ref[ok], pe)
        with np.errstate(invalid="ignore", divide="ignore"):
            eff = n_ok / n_all
        ax[1, 1].errorbar(mid(pe), eff, np.sqrt(np.clip(eff * (1 - eff), 0, None)
                                                / np.maximum(n_all, 1)), fmt="o-", ms=3)
        ax[1, 1].set(ylim=(0, 1.05), xlabel="p_T of the fragments' jet [GeV]",
                     ylabel="matched (ΔR < 0.15)", title="matching efficiency")
        ax[1, 2].hist(dr[np.isfinite(dr)], np.linspace(0, 0.6, 61), histtype="step")
        ax[1, 2].axvline(0.15, color="0.5", ls="--", lw=0.8, label="match cut 0.15")
        ax[1, 2].set(yscale="log", xlabel="ΔR (reference, nearest embedded jet)",
                     ylabel="pairs", title="ΔR of the embedding's pairs")
        ax[1, 2].legend(fontsize=7)
        for lo, hi in ((20, 30), (30, 50), (50, 100)):
            m = (ref >= lo) & (ref < hi)
            if m.sum():
                r = (emb[m & ok] - ref[m & ok]) / ref[m & ok]
                lines.append(f"embedding {lo}-{hi} GeV: matched {np.mean(ok[m]):.3f}, JES "
                             f"{np.mean(r):+.3f}, JER {np.std(r):.3f} ({int(m.sum())} jets)")
    fig.suptitle("Z+jet jet QA: background fluctuations and the response (every sample)")
    fig.tight_layout()
    figdata.savefig(fig, os.path.join(a.outdir, f"qa_jets_response{a.tag}.png"), dpi=90)
    plt.close(fig)

    # ── 4. geometry and the selection
    fig, ax = figdata.subplots(2, 3, figsize=(17, 9))
    for lab, j, W, col in sets:
        E = Ep if lab == "pp" else Eb
        m = cuts(E, j)
        ax[0, 0].stairs(hist(j["eta"][m], np.linspace(-1.6, 1.6, 33), j["w"][m], W),
                        np.linspace(-1.6, 1.6, 33), color=col, label=lab)
        pe = np.linspace(0, np.pi, 33)
        ax[0, 1].stairs(hist(dphi(j["phi"][m], E["z_phi"][j["e"][m]]), pe, j["w"][m], W), pe,
                        color=col, label=lab)
        fe = np.linspace(-np.pi, np.pi, 37)
        ax[0, 2].stairs(hist(j["phi"][m], fe, j["w"][m], W), fe, color=col, label=lab)
        # jets per Z event after the cuts (a jet's weight over its event's: 1 / QA samples)
        sel = selp if lab == "pp" else selb
        w_ = wp if lab == "pp" else wb
        n_per = np.bincount(j["e"][m], weights=j["w"][m] / np.where(w_[j["e"][m]] > 0,
                                                                     w_[j["e"][m]], 1.0),
                            minlength=len(E["event"]))
        ne = np.arange(-0.5, 6.5, 1)
        ax[1, 0].stairs(np.histogram(np.round(n_per[sel]), ne, weights=w_[sel],
                                     density=True)[0], ne, color=col, label=lab)
        lines.append(f"{lab:13s}: jets per Z after the Z+jet cuts (p_T > 30) "
                     f"{np.sum(j['w'][m]) / W:.3f}")
    ax[0, 0].set(xlabel="η (Z+jet cuts)", ylabel="jets per Z per unit", title="jet η")
    ax[0, 1].set(yscale="log", xlabel="Δφ_jZ", ylabel="jets per Z per rad", title="Δφ to the Z")
    ax[0, 2].set(xlabel="φ", ylabel="jets per Z per rad", ylim=(0, None),
                 title="jet φ (flat: no detector holes, no flow imprint)")
    ax[1, 0].set(yscale="log", xlabel="jets per Z (QA sample)", ylabel="fraction of Z's",
                 title="jet multiplicity per Z (Z+jet cuts)")
    for x in (ax[0, 0], ax[0, 1], ax[0, 2], ax[1, 0]):
        x.legend(fontsize=7)
    # Delta R to the nearest Z lepton before the 0.4 cut (|eta| < 1.6, p_T > 30)
    dre = np.linspace(0, 1.5, 61)
    for lab, j, W, col in (("S (Z events)", S, Wb, "C3"), ("pp", P, Wp, "k")):
        E = Ep if lab == "pp" else Eb
        m = (np.abs(j["eta"]) < ETA_CUT) & (j["pt"] > PT_CUT)
        e = j["e"][m]
        drl = np.minimum(*[np.hypot(j["eta"][m] - E[f"lep{l}_eta"][e],
                                    dphi(j["phi"][m], E[f"lep{l}_phi"][e])) for l in (1, 2)])
        ax[1, 1].stairs(hist(drl, dre, j["w"][m], W), dre, color=col, label=lab)
        lines.append(f"{lab:13s}: jets (p_T > 30, |eta| < 1.6) within Delta R < 0.4 of a Z "
                     f"lepton per Z {np.sum(j['w'][m][drl < DR_LEP]) / W:.4f} (removed by the "
                     f"cut: e.g. the leptons' radiation)")
    ax[1, 1].axvline(DR_LEP, color="0.5", ls="--", lw=0.8, label="cut 0.4")
    ax[1, 1].set(yscale="log", xlabel="ΔR(jet, nearest Z lepton)", ylabel="jets per Z per unit",
                 title="the lepton cut (before it)")
    ax[1, 1].legend(fontsize=7)
    # the p_T^Z and x_jZ phase space of the jets
    m = cuts(Eb, S)
    hh = ax[1, 2].hist2d(Eb["z_pt"][S["e"][m]], S["pt"][m], [np.linspace(40, 200, 33),
                                                           np.linspace(30, 200, 35)],
                         weights=S["w"][m], cmin=1e-12, norm=matplotlib.colors.LogNorm())
    fig.colorbar(hh[3], ax=ax[1, 2])
    ax[1, 2].plot([40, 200], [40, 200], "w--", lw=0.8, label="x_jZ = 1")
    ax[1, 2].set(xlabel="p_T^Z [GeV]", ylabel="p_T,jet [GeV]", title="S: jet against Z p_T")
    ax[1, 2].legend(fontsize=7)
    fig.suptitle("Z+jet jet QA: geometry and the selection (QA samples)")
    fig.tight_layout()
    figdata.savefig(fig, os.path.join(a.outdir, f"qa_jets_geometry{a.tag}.png"), dpi=90)
    plt.close(fig)

    # ── 5. constituents
    fig, ax = figdata.subplots(1, 4, figsize=(21, 4.6))
    ne = np.arange(0.5, 200.5, 4)
    pe = np.array([-20, -10, 0, 10, 20, 30, 40, 60, 80, 120.0])
    for lab, j, W, col in sets:
        E = Ep if lab == "pp" else Eb
        m = cuts(E, j)
        ax[0].stairs(hist(j["nconst"][m], ne, j["w"][m], W), ne, color=col, label=lab)
        acc = np.abs(j["eta"]) < ETA_CUT
        mu, sd = profile(j["pt"][acc], j["nconst"][acc], pe, j["w"][acc])
        ax[1].errorbar(mid(pe), mu, sd, fmt="o-", ms=3, color=col, label=lab)
        ze = np.linspace(0, 1, 41)
        ax[2].stairs(hist(j["lead_pt"][m] / j["pt_raw"][m], ze, j["w"][m], W), ze, color=col,
                     label=lab)
        me = np.linspace(0, 1, 51)
        ax[3].stairs(hist(j["m"][m] / j["pt_raw"][m], me, j["w"][m], W), me, color=col,
                     label=lab)
        if m.any():
            wm = j["w"][m]
            zl = np.average(j["lead_pt"][m] / j["pt_raw"][m], weights=wm)
            mm = np.average(j["m"][m] / j["pt_raw"][m], weights=wm)
            lines.append(f"{lab:13s}: jets above 30 GeV: <n_const> "
                         f"{np.average(j['nconst'][m], weights=wm):.1f}, "
                         f"<p_T,lead / p_T,raw> {zl:.3f}, <m / p_T,raw> {mm:.3f}")
    ax[0].set(yscale="log", xlabel="constituents (Z+jet cuts)", ylabel="jets per Z per unit",
              title="constituents")
    ax[1].set(xlabel="p_T [GeV] (|η| < 1.6)", ylabel="⟨constituents⟩ ± sd",
              title="constituents against p_T")
    ax[2].set(yscale="log", xlabel="p_T,lead / p_T,raw (Z+jet cuts)", ylabel="jets per Z per unit",
              title="the leading constituent's fraction (fakes: soft)")
    ax[3].set(yscale="log", xlabel="m / p_T,raw (Z+jet cuts)", ylabel="jets per Z per unit",
              title="jet mass (unsubtracted)")
    for x in ax:
        x.legend(fontsize=7)
    fig.suptitle("Z+jet jet QA: the jets' constituents (all final particles but ν and the Z's "
                 "leptons; Pb+Pb before subtraction)")
    fig.tight_layout()
    figdata.savefig(fig, os.path.join(a.outdir, f"qa_jets_constituents{a.tag}.png"), dpi=90)
    plt.close(fig)

    # ── 6. the mixing: B_mixed's backgrounds against the own one
    fig, ax = figdata.subplots(1, 3, figsize=(17, 4.6))
    if B_mix is not None and nn_info is not None:
        ax[0].hist(nn_info, 40, histtype="step")
        ax[0].set(xlabel="|N_coll(mixed) − N_coll(own)|", ylabel="backgrounds drawn",
                  title=f"the {a.n_nearest} nearest backgrounds in N_coll")
        lines.append(f"B_mixed: |Delta N_coll| to the own background median "
                     f"{np.median(nn_info):.0f}, 90 % below {np.percentile(nn_info, 90):.0f}")
        if rho_bg is not None:
            bg_rho = {int(b): np.nanmean(rho_bg[QB["bg"] == b]) for b in np.unique(QB["bg"])}
            d = []
            bgs = np.array(sorted(bg_rho))
            nc_b = np.array([QB["n_coll"][QB["bg"] == b][0] for b in bgs], float)
            for e_ in zsel:
                b = int(Eb["bg"][e_])
                others = bgs != b
                if b not in bg_rho or not others.any():
                    continue
                near = bgs[others][np.argsort(np.abs(nc_b[others] - Eb["n_coll"][e_]),
                                              kind="stable")[:a.n_nearest]]
                d.append(np.mean([bg_rho[int(x)] for x in near]) - bg_rho[b])
            ax[1].hist(d, 40, histtype="step")
            ax[1].set(xlabel="⟨ρ⟩ of the mixed backgrounds − ρ of the own one [GeV]",
                      ylabel="Z's", title="ρ of the mixed backgrounds")
            if d:
                lines.append(f"B_mixed: <rho(mixed)> - rho(own) per Z: mean {np.mean(d):+.3f}, "
                             f"sd {np.std(d):.3f} GeV")
        edges = np.array([0, 5, 10, 15, 20, 25, 30, 40, 50, 70, 100.0])
        mo, mm = cuts(Eb, B_own, pt=False), cuts(Eb, B_mix, pt=False)
        ho = hist(B_own["pt"][mo], edges, B_own["w"][mo], Wb)
        hm = hist(B_mix["pt"][mm], edges, B_mix["w"][mm], Wb)
        with np.errstate(invalid="ignore", divide="ignore"):
            ax[2].plot(mid(edges), hm / ho, "o-", ms=3)
        ax[2].axhline(1, color="0.6", lw=0.6)
        ax[2].set(xlabel="p_T [GeV] (Z+jet cuts but p_T)", ylabel="B_mixed / B_own",
                  title="the background jets: mixed against own")
    else:
        for x in ax:
            x.text(0.5, 0.5, "needs ≥ 2 hydro backgrounds", ha="center", va="center",
                   transform=x.transAxes)
        lines.append("mixing QA skipped: one hydro background")
    fig.suptitle("Z+jet jet QA: B_mixed (step 3) against B_own (step 2)")
    fig.tight_layout()
    figdata.savefig(fig, os.path.join(a.outdir, f"qa_jets_mixing{a.tag}.png"), dpi=90)
    plt.close(fig)

    text = "\n".join(lines) + "\n"
    with open(os.path.join(a.outdir, f"qa_jets_summary{a.tag}.txt"), "w") as f:
        f.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
