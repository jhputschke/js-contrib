#!/usr/bin/env python3
"""The pp comparison (README.md): the campaign's pp reference (Matter Q0 = 2 GeV) and JETSCAPE's
PP19 tune (Q0 = 1 GeV), on the same Pythia events, against CMS's pp data in the Z-hadron
(2507.09307) and Z+jet (1702.01060) observables, and against each other.

    python compare_pp.py --data runs/pp_comp/data [--outdir runs/pp_comp/figures] [--blocks 100]

Inputs (make_data.sh): data/mix/<sample>/vacuum/*.npz (mix.py --pools-only) and
data/jets/<sample>/*_jets.root (zjet_jets_map.py, R = 0.3), for the samples campaign and pp19.

Z-hadron (CMS 2507.09307's Z selection): mix.py's pp result, the factorized S - B
(../MIXING_MODES.md section 4: B pooled from all other Z events, flat in Delta phi, Delta y
without the own event), Delta phi and Delta y (Z side) in the three track p_T classes, against
CMS's pp tables (stat + syst).  Z+jet (CMS 1702.01060's Z and jet cuts): zjet_jets.py's pp
observables (Delta phi_jZ, x_jZ, <x_jZ>, R_jZ) smeared to PbPb's jet resolution against CMS's
smeared-pp tables 1, 3, 5, 7, smeared to pp's against table 9 (x_jZ), and unsmeared.

The two samples share their events (the seed, the trigger and Pythia do not see Q0), so
PP19 - campaign is formed event by event with a jackknife over the same blocks of events: the
error of the difference, much smaller than either sample's.  The Z+jet difference is taken
unsmeared (the smearing's random numbers would add noise of their own).  chi^2 against CMS:
the model's and CMS's errors in quadrature (zjet_jets.chi2).
-> <outdir>/pp_comp_zhadron_dphi.png, pp_comp_zhadron_dy.png, pp_comp_zjet.png and
   pp_comp_summary.txt (every PNG with its .npz and .root, jetcorr.figdata)."""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ZJET = os.path.dirname(HERE)
sys.path.insert(0, ZJET)
sys.path.insert(0, os.path.dirname(ZJET))                        # jetcorr
import mix  # noqa: E402  (puts PyJetscape/python on the path when needed)
import zjet_jets as zj  # noqa: E402
import zjet_mixing as zm  # noqa: E402
from cms2507 import PT_LABELS, PT_TAGS  # noqa: E402
from cms2507 import cms as cms_zh  # noqa: E402
from jetcorr import figdata, runs  # noqa: E402

SAMPLES = ("campaign", "pp19")
LABEL = {"campaign": "campaign pp (Matter Q0 = 2 GeV)", "pp19": "PP19 (Matter Q0 = 1 GeV)"}
SHORT = {"campaign": "campaign", "pp19": "PP19"}
COLOR = {"campaign": "C0", "pp19": "C3"}
DIFF = "PP19 − campaign (paired)"


def sample_info(data, s):
    """The production's summary (run_prod_vacuum.py's <stem>.json) of a sample."""
    paths = sorted(glob.glob(os.path.join(data, s, "*_seed*.json")))
    if not paths:
        return {}
    with open(paths[0]) as f:
        return json.load(f)


def sig(d, e):
    """sum (d / e)^2 over the bins with an error, and their number."""
    ok = np.isfinite(d) & np.isfinite(e) & (e > 0)
    return float(np.sum((d[ok] / e[ok]) ** 2)), int(ok.sum())


# ── Z-hadron ─────────────────────────────────────────────────────────────────────
def zhadron(a, plt, lines):
    pools, P = {}, {}
    for s in SAMPLES:
        paths = sorted(glob.glob(os.path.join(a.data, "mix", s, "vacuum", "*.npz")))
        if not paths:
            raise SystemExit(f"no pools in {a.data}/mix/{s}/vacuum (make_data.sh pools)")
        pools[s] = pool = zm.Pool.load(paths)
        P[s] = mix.from_all(pool, "full", ("all", "self"), True, pool.blocks(a.blocks), 1)
    pa, pb = pools["campaign"], pools["pp19"]
    paired = (pa.n == pb.n and np.array_equal(pa.y, pb.y) and np.array_equal(pa.phi, pb.phi)
              and np.allclose(pa.w, pb.w))
    D = {}
    for var in ("phi", "y"):
        d = P["pp19"][var][0] - P["campaign"][var][0]
        if paired:
            e = zm.jackknife(P["pp19"][f"{var}_reps"] - P["campaign"][f"{var}_reps"])
        else:
            e = np.hypot(P["pp19"][var][1], P["campaign"][var][1])
        D[var] = (d, e)

    lines.append("Z-hadron, CMS 2507.09307 pp (factorized S - B; chi2 with the model's errors "
                 "and CMS stat + syst, over the independent half of the reflected bins; d at "
                 "Delta phi or Delta y > 0)")
    for s in SAMPLES:
        p = pools[s]
        per_z = p.bg_R["full"].sum(axis=(0, 2, 3)) / p.bg_W.sum()
        lines.append(f"  {LABEL[s]}: {p.n} Z's, N_eff {zm.n_eff(p.w):.0f}; charged tracks per Z "
                     f"(|eta| < 2.4) per class {zj.fmt(per_z)}")
    lines.append(f"  {DIFF}: " + ("the same Z's in both, jackknife of the difference"
                                  if paired else "NOT the same Z's: errors in quadrature"))
    xlab = {"phi": "Δφ_ch,Z", "y": "Δy_ch,Z (Z side, |Δφ| < π/2)"}
    for var in ("phi", "y"):
        fig, ax = figdata.subplots(3, 3, figsize=(16, 11), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1.5, 1.5]})
        for c in range(len(PT_TAGS)):
            cx, y, st, sy = cms_zh(a.hep_zh, var, "pp", PT_TAGS[c])
            tot = np.hypot(st, sy)
            top, mid, bot = ax[0, c], ax[1, c], ax[2, c]
            top.errorbar(cx, y, tot, fmt="o", color="k", ms=4, label="CMS pp (stat ⊕ syst)")
            mid.fill_between(cx, -tot, tot, step="mid", color="k", alpha=0.12,
                             label="CMS stat ⊕ syst")
            for j, s in enumerate(SAMPLES):
                m, me = P[s][var][0][c], P[s][var][1][c]
                dx = 0.03 * (j - 0.5) * (cx[1] - cx[0]) / 0.1
                top.errorbar(cx + dx, m, me, fmt="s", ms=3.5, mfc="none", color=COLOR[s],
                             label=LABEL[s])
                mid.errorbar(cx + dx, m - y, me, fmt="s", ms=3.5, mfc="none", color=COLOR[s],
                             label=f"{SHORT[s]} − CMS")
                chi, nb = mix.chi2(m, me, y, tot)
                lines.append(f"  {var:3s} {PT_LABELS[c]:20s} {LABEL[s]:34s} chi2 {chi:6.1f}/{nb}")
            d, e = D[var][0][c], D[var][1][c]
            bot.errorbar(cx, d, e, fmt="D", ms=3.5, color="C2", label=DIFF)
            # the bins are reflected (+- merged): their independent half
            half = cx > 0 if var == "y" else (cx > 0) & (cx < np.pi)
            s2, nb = sig(d[half], e[half])
            lines.append(f"  {var:3s} {PT_LABELS[c]:20s} {DIFF:34s} sum (d/err)^2 {s2:6.1f}/{nb}"
                         f"   d {zj.fmt(d[half])}")
            for axx in (top, mid, bot):
                axx.axhline(0, color="0.6", lw=0.6)
            top.set_title(PT_LABELS[c])
            top.legend(fontsize=7)
            mid.legend(fontsize=7)
            bot.legend(fontsize=7)
            sym = "φ" if var == "phi" else "y"
            top.set_ylabel(f"d⟨ΔN_ch⟩/dΔ{sym}")
            mid.set_ylabel("model − CMS")
            bot.set_ylabel("PP19 − campaign")
            bot.set_xlabel(xlab[var])
        fig.suptitle(f"Z–hadron Δ{'φ' if var == 'phi' else 'y'} in pp against CMS 2507.09307: "
                     "the campaign's pp (Matter Q0 = 2 GeV) and PP19 (Q0 = 1 GeV), the same "
                     "Pythia events", y=0.995)
        fig.tight_layout()
        path = os.path.join(a.outdir, f"pp_comp_zhadron_d{var}{a.tag}.png")
        figdata.savefig(fig, path, dpi=100)
        plt.close(fig)
        print(f"-> {path}")
    lines.append("")


# ── Z+jet ────────────────────────────────────────────────────────────────────────
def zjet(a, plt, lines):
    res, cnt, ev = {}, {}, {}
    for i, s in enumerate(SAMPLES):
        src = os.path.join(a.data, "jets", s)
        R = zj.map_run_info(src)["R"]
        if abs(R - 0.3) > 1e-9:
            raise SystemExit(f"{src}: R = {R:g}, CMS 1702.01060 needs R = 0.3")
        E, J, _ = zj.load_jets(src)
        sel = zj.z_selection(E)
        w = zj.weights(E, sel, ncoll=False)
        idx = {g: k for k, g in enumerate(E["event"])}
        evr = np.array([idx[g] for g in J["event"]])
        rows = np.arange(len(J["event"]))
        e, pt, eta, phi, wj = zj.flatten_rows(E, J, "jet", rows, w[evr] / E["n_samples"][evr],
                                              evr)
        res[s], cnt[s] = {}, {}
        for j, (key, sn) in enumerate((("unsmeared", None), ("pp-smeared", zj.RES["pp"]),
                                       ("PbPb-smeared", zj.RES["PbPb"]))):
            rng = np.random.default_rng([a.seed, i, j])
            ptx = pt if sn is None else zj.smear(pt, *sn, rng)
            c = {**zj.z_counts(E, sel, w), **zj.pair_sums(E, sel, e, ptx, eta, phi, wj)}
            cnt[s][key] = c
            res[s][key] = zj.with_errors(c, E["event"], a.blocks)
        ev[s] = (E, sel)
    (Ea, sela), (Eb, selb) = ev["campaign"], ev["pp19"]
    paired = (np.array_equal(Ea["event"], Eb["event"]) and np.array_equal(sela, selb)
              and np.allclose(Ea["z_pt"], Eb["z_pt"]))
    if paired:
        D = zj.diff_with_errors(cnt["pp19"]["unsmeared"], cnt["campaign"]["unsmeared"],
                                Ea["event"], a.blocks)
    else:
        vb, eb = res["pp19"]["unsmeared"]
        va, ea = res["campaign"]["unsmeared"]
        D = ({k: vb[k] - va[k] for k in va}, {k: np.hypot(eb[k], ea[k]) for k in va})

    lines.append("Z+jet, CMS 1702.01060 pp (R = 0.3; chi2 with the model's errors and CMS's)")
    for s in SAMPLES:
        E, sel = ev[s]
        v = res[s]["unsmeared"][0]
        lines.append(f"  {LABEL[s]}: {int(sel.sum())} Z's (p_T^Z > 40), "
                     f"{int((sel & (E['z_pt'] > 60)).sum())} with p_T^Z > 60; unsmeared "
                     f"<x_jZ> {zj.fmt(v['mx'])}, R_jZ {zj.fmt(v['rjz'])}")
    lines.append(f"  {DIFF}: " + ("the same Z's in both, jackknife of the difference"
                                  if paired else "NOT the same Z's: errors in quadrature"))
    xs = {"phi": 0.5 * (zj.PHI_EDGES[1:] + zj.PHI_EDGES[:-1]),
          "x": 0.5 * (zj.X_EDGES[1:] + zj.X_EDGES[:-1]),
          "mx": np.array([45, 55, 70, 100.0]), "rjz": np.array([45, 55, 70, 100.0])}
    zedges = np.array([lo for lo, _ in zj.ZPT_BINS] + [zj.ZPT_BINS[-1][1]], float)
    edges = {"phi": zj.PHI_EDGES, "x": zj.X_EDGES, "mx": zedges, "rjz": zedges}
    # (table, the model variant it goes with, label, marker, colour)
    tables = {obs: [(zj.TABLES[obs][0], "PbPb-smeared", "CMS pp smeared to PbPb", "o", "k")]
              for obs in zj.OBS}
    tables["x"].append((9, "pp-smeared", "CMS pp", "D", "C2"))
    fig, ax = figdata.subplots(3, 4, figsize=(20, 10.5), sharex="col",
                               gridspec_kw={"height_ratios": [3, 1.4, 1.4]})
    for col, obs in enumerate(zj.OBS):
        top, mid, bot = ax[0, col], ax[1, col], ax[2, col]
        for ti, key, lab, mk, colr in tables[obs]:
            y, e = zj.cms(a.hep_zj, ti)
            top.errorbar(xs[obs], y, e, fmt=mk, color=colr, ms=4, label=f"{lab} (table {ti})")
            with np.errstate(invalid="ignore", divide="ignore"):
                rd = e / y
            mid.stairs(1 + rd, edges[obs], baseline=1 - rd, fill=True, color=colr, alpha=0.12)
            for j, s in enumerate(SAMPLES):
                m, me = res[s][key][0][obs], res[s][key][1][obs]
                dx = (1 + 0.01 * (j - 0.5) * (2 if key == "pp-smeared" else 1))
                ls = "--" if key == "pp-smeared" else "-"
                top.errorbar(xs[obs] * dx, m, me, fmt="s" + ls, ms=3.5, mfc="none", lw=0.8,
                             color=COLOR[s], label=f"{SHORT[s]}, {key}")
                with np.errstate(invalid="ignore", divide="ignore"):
                    mid.errorbar(xs[obs] * dx, m / y, me / y, fmt="s", ms=3.5, mfc="none",
                                 color=COLOR[s], label=f"{SHORT[s]} {key} / table {ti}")
                chi, nb = zj.chi2(m, me, y, e)
                lines.append(f"  {obs:4s} {LABEL[s]:34s} {key:13s} vs data{ti}: "
                             f"chi2 {chi:6.1f}/{nb}   model {zj.fmt(m)}")
        d, de = D[0][obs], D[1][obs]
        bot.errorbar(xs[obs], d, de, fmt="D", ms=3.5, color="C2", label=f"{DIFF}, unsmeared")
        s2, nb = sig(d, de)
        lines.append(f"  {obs:4s} {DIFF:34s} unsmeared     sum (d/err)^2 {s2:6.1f}/{nb}   "
                     f"d {zj.fmt(d)}")
        top.set_title(zj.LABELS[obs])
        if obs == "phi":
            top.set_yscale("log")
        top.legend(fontsize=6)
        mid.axhline(1, color="0.5", lw=0.6)
        mid.set_ylim(0.2, 1.8)
        mid.set_ylabel("model / CMS")
        mid.legend(fontsize=6)
        bot.axhline(0, color="0.5", lw=0.6)
        bot.set_ylabel("PP19 − campaign")
        bot.legend(fontsize=6)
    ax[2, 0].set_xlabel("Δφ_jZ")
    ax[2, 1].set_xlabel("x_jZ")
    ax[2, 2].set_xlabel("p_T^Z [GeV]")
    ax[2, 3].set_xlabel("p_T^Z [GeV]")
    fig.suptitle("Z+jet in pp against CMS 1702.01060 (R = 0.3): the campaign's pp (Matter Q0 = "
                 "2 GeV) and PP19 (Q0 = 1 GeV), the same Pythia events (bands: CMS's errors)",
                 y=0.995)
    fig.tight_layout()
    path = os.path.join(a.outdir, f"pp_comp_zjet{a.tag}.png")
    figdata.savefig(fig, path, dpi=100)
    plt.close(fig)
    print(f"-> {path}")
    lines.append("")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=runs.data("pp_comp"),
                    help="make_data.sh's data directory (default runs/pp_comp/data)")
    ap.add_argument("--outdir", default=None, help="default runs/pp_comp/figures")
    ap.add_argument("--hep-zh", dest="hep_zh", default=runs.hepdata("HEPData-ins2945894-v1-yaml"))
    ap.add_argument("--hep-zj", dest="hep_zj", default=runs.hepdata("HEPData-ins1512112-v1-yaml"))
    ap.add_argument("--blocks", type=int, default=100, help="jackknife blocks of events")
    ap.add_argument("--seed", type=int, default=zj.SEED, help="the jet smearing's random numbers")
    ap.add_argument("--tag", default="", help="suffix of the output names")
    a = ap.parse_args()
    a.outdir = a.outdir or runs.figures("pp_comp")
    os.makedirs(a.outdir, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lines = ["The pp comparison: the campaign's pp reference against JETSCAPE's PP19 tune "
             "(arXiv:1910.05481), which differ in Matter's Q0 alone (check_settings.py)", ""]
    for s in SAMPLES:
        i = sample_info(a.data, s)
        lines.append(f"{LABEL[s]}: {i.get('events_written', '?')} events, seed {i.get('seed', '?')}, "
                     f"{i.get('n_frag', '?')} fragmentations, "
                     f"{i.get('particles_per_fragmentation', '?')} particles per fragmentation")
    lines.append("")
    zhadron(a, plt, lines)
    zjet(a, plt, lines)
    path = os.path.join(a.outdir, f"pp_comp_summary{a.tag}.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
