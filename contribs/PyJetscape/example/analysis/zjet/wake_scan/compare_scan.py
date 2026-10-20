#!/usr/bin/env python3
"""The scan's comparison (README.md): for every value of the scanned liquefier setting the
Z-side Delta y of Pb+Pb - pp split into the fragments' and the wake's parts (../wake_decomp.py),
and every value against the baseline, event by event.

    python compare_scan.py --data runs/wake_scan/data --pp <pp pool dir> \\
        --param Liquefier/CausalLiquefier/tau_delay --values 0.5 1.0 2.0 --base 1.0

Inputs (make_data.sh): data/mix/<name>/pbpb/ (version-4 pools, mix.py --pools-only), <name> =
<setting>_<value with . -> p>.  The values' jobs have the same seeds, so the same events,
backgrounds, jets and droplets (make_data.sh check): variant - baseline is formed per event
with a jackknife over the same blocks of hydro backgrounds, the error of the difference.  The
pp is the same for all values and cancels in the differences; the fragments' pieces differ
only by the hadronization's random numbers (each job its own), so they stay near zero there
and show the noise floor.
-> <outdir>/wake_scan_<setting>.png (with .npz/.root) and wake_scan_<setting>.txt."""
import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ZJET = os.path.dirname(HERE)
sys.path.insert(0, ZJET)
sys.path.insert(0, os.path.dirname(ZJET))                         # jetcorr
import wake_decomp as wd  # noqa: E402  (imports mix: the PyJetscape path)
import zjet_mixing as zm  # noqa: E402
from cms2507 import PT_LABELS, PT_TAGS  # noqa: E402
from jetcorr import figdata, runs  # noqa: E402

YMAX = wd.YMAX


def name_of(param, value):
    return f"{param.split('/')[-1]}_{value.replace('.', 'p')}"


def paired(pools, pp, base, width, n_blocks):
    """Per value: the pieces with errors, and value - base with the jackknife of the
    difference over the same background blocks (pp kept whole: it cancels)."""
    pb0 = pools[base]
    all_pp = np.ones(pp.n, bool)
    units = np.unique(pb0.zb)
    blocks = [b for b in np.array_split(units, min(n_blocks, len(units))) if len(b)]
    full = {v: wd.pieces(p, pp, np.ones(p.n, bool), all_pp, width) for v, p in pools.items()}
    reps = {v: [] for v in pools}
    for b in blocks:
        keep = ~np.isin(pb0.zb, b)
        for v, p in pools.items():
            reps[v].append(wd.pieces(p, pp, keep, all_pp, width))
    G = len(blocks)

    def jk(rs):
        r = np.array(rs)
        return np.sqrt((G - 1) / G * np.sum((r - r.mean(0)) ** 2, axis=0))
    diff, derr = {}, {}
    for v in pools:
        if v == base:
            continue
        diff[v] = {k: full[v][k] - full[base][k] for k in wd.PIECES}
        derr[v] = {k: jk([rv[k] - rb[k] for rv, rb in zip(reps[v], reps[base])])
                   for k in wd.PIECES}
    return diff, derr


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=runs.data("wake_scan"))
    ap.add_argument("--pp", nargs="+", required=True, help="pp pool directories (version 4)")
    ap.add_argument("--param", default="Liquefier/CausalLiquefier/tau_delay")
    ap.add_argument("--values", nargs="+", default=["0.5", "1.0", "2.0"])
    ap.add_argument("--base", default="1.0")
    ap.add_argument("--base-name", default=None,
                    help="the baseline's run name if not <setting>_<value> (make_data.sh REF: "
                         "an earlier scan's campaign-setting jobs)")
    ap.add_argument("--outdir", default=None, help="default runs/wake_scan/figures")
    ap.add_argument("--hepdata", default=runs.hepdata("HEPData-ins2945894-v1-yaml"))
    ap.add_argument("--blocks", type=int, default=100)
    ap.add_argument("--dy", type=float, default=0.8)
    a = ap.parse_args()
    a.outdir = a.outdir or runs.figures("wake_scan")
    if a.base not in a.values:
        raise SystemExit("--base must be one of --values")
    tag = a.param.split("/")[-1]
    names = {v: (a.base_name if (v == a.base and a.base_name) else name_of(a.param, v))
             for v in a.values}
    pools = {v: wd.load([os.path.join(a.data, "mix", names[v], "pbpb")]) for v in a.values}
    pp = wd.load(a.pp)
    p0 = pools[a.base]
    for v, p in pools.items():
        if not (p.n == p0.n and np.array_equal(p.event, p0.event) and np.array_equal(p.y, p0.y)
                and np.array_equal(p.bg, p0.bg) and np.allclose(p.w, p0.w)):
            raise SystemExit(f"{tag} = {v}: not the same Z events as {a.base} (make_data.sh check)")
    n_bins = int(round(2 * YMAX / a.dy))
    centres = -YMAX + a.dy * (np.arange(n_bins) + 0.5)
    pos = centres > 0
    first = int(np.argmax(pos))
    absval = {v: wd.with_errors(p, pp, a.dy, a.blocks) for v, p in pools.items()}
    diff, derr = paired(pools, pp, a.base, a.dy, a.blocks)
    ref = [wd.cms_difference(a.hepdata, c, a.dy) for c in range(len(PT_TAGS))]

    lines = [f"Scan of {a.param}: {' '.join(a.values)} (baseline {a.base}, run {names[a.base]}); "
             f"{p0.n} Pb+Pb Z's on "
             f"{p0.n_bg} hydro backgrounds (N_eff {zm.n_eff(p0.w):.0f}), the same in every value; "
             f"pp {pp.n} Z's from {' '.join(a.pp)}",
             "Z side, per Z per unit Delta y, +-merged x 0.5; the pieces of ../wake_decomp.py", ""]
    for where, sl, scale in ((f"first bin past 0 (0 < Delta y < {a.dy:g})", slice(first, first + 1), 1.0),
                             (f"integrated over 0 < Delta y < {YMAX:g} (per Z)", pos, a.dy)):
        lines.append(f"The values, {where}:")
        for c in range(len(PT_TAGS)):
            r = ref[c]
            cms_s = (f"   CMS {r[1][r[0] > 0][0]:+.3f}±{r[2][r[0] > 0][0]:.3f}"
                     if (r and scale == 1.0) else "")
            lines.append(f"  {PT_LABELS[c]}{cms_s}")
            for v in a.values:
                val, err = absval[v]
                lines.append(f"    {tag} {v:5s} " + " | ".join(
                    f"{k} {scale * val[k][c][sl].sum():+.3f}±"
                    f"{scale * np.sqrt((err[k][c][sl] ** 2).sum()):.3f}" for k in wd.PIECES))
        lines.append(f"Paired differences value - {a.base}, {where}:")
        for c in range(len(PT_TAGS)):
            lines.append(f"  {PT_LABELS[c]}")
            for v in diff:
                lines.append(f"    {tag} {v:5s} " + " | ".join(
                    f"{k} {scale * diff[v][k][c][sl].sum():+.3f}±"
                    f"{scale * np.sqrt((derr[v][k][c][sl] ** 2).sum()):.3f}" for k in wd.PIECES))
        lines.append("")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = figdata.subplots(2, len(PT_TAGS), figsize=(17, 9.5), sharex=True)
    cols = {v: f"C{i}" for i, v in enumerate(a.values)}
    for c in range(len(PT_TAGS)):
        top, bot = ax[0, c], ax[1, c]
        r = ref[c]
        if r:
            h = r[0] > 0
            top.errorbar(r[0][h], r[1][h], r[2][h], fmt="D", color="0.5", ms=5,
                         label="CMS Pb+Pb 0–30 % − pp")
        for i, v in enumerate(a.values):
            val, err = absval[v]
            top.errorbar(centres[pos] + 0.04 * i, val[wd.PIECES[0]][c][pos],
                         err[wd.PIECES[0]][c][pos], fmt="s-", ms=4, color=cols[v],
                         lw=1.4 if v == a.base else 0.9, label=f"{tag} = {v}")
        for i, v in enumerate(diff):
            for k, ls, mk in ((wd.PIECES[0], "-", "s"), ("wake, Z side", "--", "o"),
                              ("wake, level", ":", "v")):
                bot.errorbar(centres[pos] + 0.04 * i, diff[v][k][c][pos], derr[v][k][c][pos],
                             fmt=mk + ls, ms=4, color=cols[v], mfc="none" if "level" in k else cols[v],
                             label=f"{tag} {v} − {a.base}: {k}")
            fr = diff[v]["fragments, Z side"][c] + diff[v]["fragments, level"][c]
            bot.plot(centres[pos] + 0.04 * i, fr[pos], "x", color=cols[v], alpha=0.6,
                     label=f"{tag} {v} − {a.base}: fragments (noise floor)")
        for axx in (top, bot):
            axx.axhline(0, color="0.6", lw=0.6)
            axx.legend(fontsize=6)
        top.set_title(PT_LABELS[c])
        top.set_ylabel("Pb+Pb − pp: d⟨ΔN_ch⟩/dΔy")
        bot.set_ylabel(f"paired difference to {tag} = {a.base}")
        bot.set_xlabel("Δy_ch,Z (Z side, |Δφ| < π/2)")
    fig.suptitle(f"Scan of {a.param}: the Z-side Δy of Pb+Pb 0–30 % − pp (top) and each value "
                 f"against {a.base}, event by event (bottom); {p0.n} Z's on {p0.n_bg} backgrounds",
                 fontsize=10)
    fig.tight_layout()
    os.makedirs(a.outdir, exist_ok=True)
    path = os.path.join(a.outdir, f"wake_scan_{tag}.png")
    figdata.savefig(fig, path, dpi=100)
    plt.close(fig)
    txt = os.path.join(a.outdir, f"wake_scan_{tag}.txt")
    with open(txt, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"-> {path}\n-> {txt}")


if __name__ == "__main__":
    main()
