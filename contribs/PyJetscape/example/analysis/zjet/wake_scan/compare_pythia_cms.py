#!/usr/bin/env python3
"""CMS 2507.09307's observable per Pythia setting: the Z-side Delta y of Pb+Pb 0-30 % - pp
(the mean-subtracted Delta<N_ch> = S - B, mix.py's factorized result) and its split into the
fragments' and the wake's parts (../wake_decomp.py), for the campaign's settings, MPI off, and
ISR + MPI off, each against its own pp (the same Pythia settings in vacuum), with CMS's
HEPData.  The companion of compare_pythia.py (the wake on its own).

    python compare_pythia_cms.py [--data runs/wake_scan/data] [--e 2p0]
        [--setting NAME:PBPB_RUN:PP_POOLS:LABEL ...]      # default: the three below

Inputs: version-4 pools (make_data.sh pools): data/mix/<run>/pbpb and the pp's vacuum pools.
The settings have other hard events (the same hydro backgrounds): each is compared with CMS on
its own; the errors are each setting's jackknife (Pb+Pb backgrounds and pp events).
--pp-all: one pp for all settings (the campaign's, as the experiments use their best pp tune
whatever the nucleus does to MPI and ISR).
-> <outdir>/pythia_cms{tag}.png (with .npz/.root) and pythia_cms{tag}.txt."""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ZJET = os.path.dirname(HERE)
sys.path.insert(0, ZJET)
sys.path.insert(0, os.path.dirname(ZJET))                         # jetcorr
import wake_decomp as wd  # noqa: E402  (imports mix: the PyJetscape path)
from cms2507 import PT_LABELS, PT_TAGS  # noqa: E402
from jetcorr import figdata, runs  # noqa: E402

PP_CAMPAIGN = runs.data("pp_comp", "mix", "campaign", "vacuum")
DEFAULT = (f"campaign:{{tau}}:{PP_CAMPAIGN}:ISR + MPI on (campaign)",
           "mpi_off:mpi_off_e_threshold_{e}:{data}/mix/pp_mpi_off/vacuum:MPI off",
           "isr_mpi_off:isr_mpi_off_e_threshold_{e}:{data}/mix/pp_isr_mpi_off/vacuum:"
           "ISR + MPI off")
COLS = ("k", "C0", "C3", "C2", "C1")
SUMS = (("Pb+Pb − pp", (wd.PIECES[0],)),
        ("fragments (Z side + level)", (wd.PIECES[1], wd.PIECES[2])),
        ("wake (Z side + level)", (wd.PIECES[3], wd.PIECES[4])))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=runs.data("wake_scan"))
    ap.add_argument("--e", default="2p0", help="e_threshold of the runs: 2p0 or 3p0")
    ap.add_argument("--setting", action="append", default=None,
                    help="NAME:PBPB_RUN:PP_POOL_DIR:LABEL (data/mix/<run>/pbpb)")
    ap.add_argument("--outdir", default=None, help="default runs/wake_scan/figures")
    ap.add_argument("--hepdata", default=runs.hepdata("HEPData-ins2945894-v1-yaml"))
    ap.add_argument("--blocks", type=int, default=100)
    ap.add_argument("--dy", type=float, default=0.8)
    ap.add_argument("--pp-all", default=None, metavar="PP_POOL_DIR",
                    help="one pp for every setting instead of each its own (e.g. the campaign's: "
                         "the experiments' pp is the best tune, whatever the nucleus does)")
    ap.add_argument("--tag", default=None, help="suffix of the output names (default: from --e "
                    "and --pp-all)")
    a = ap.parse_args()
    a.outdir = a.outdir or runs.figures("wake_scan")
    tau = "tau_delay_1p0" if a.e == "2p0" else f"e_threshold_{a.e}"
    specs = [s.format(tau=tau, e=a.e, data=a.data).split(":", 3)
             for s in (a.setting or DEFAULT)]
    if a.pp_all:
        specs = [(n, r, a.pp_all, lab) for n, r, _, lab in specs]
    res = []
    for name, run, pp_dir, label in specs:
        pb = wd.load([os.path.join(a.data, "mix", run, "pbpb")])
        pp = wd.load([pp_dir])
        v, e = wd.with_errors(pb, pp, a.dy, a.blocks)
        res.append((name, label, run, pb, pp, v, e))
    n_bins = int(round(2 * wd.YMAX / a.dy))
    centres = -wd.YMAX + a.dy * (np.arange(n_bins) + 0.5)
    pos = centres > 0
    first = int(np.argmax(pos))
    ref = [wd.cms_difference(a.hepdata, c, a.dy) for c in range(len(PT_TAGS))]

    def summed(v, e, keys):
        return sum(v[k] for k in keys), np.sqrt(sum(e[k] ** 2 for k in keys))

    L = [f"CMS 2507.09307's Z-side Delta y of Pb+Pb - pp per Pythia setting, e_threshold "
         f"{a.e.replace('p', '.')} (compare_pythia_cms.py); per Z per unit Delta y, +-merged x 0.5; "
         f"pp: {a.pp_all or 'each setting its own'}",
         "chi2: the Pb+Pb - pp curve against CMS over 0 < Delta y < 4 (model and CMS errors "
         "in quadrature, CMS stat + syst; bins treated as independent)", ""]
    for name, label, run, pb, pp, v, e in res:
        L.append(f"{label} ({run}; pp {pp.n} Z's): {pb.n} Pb+Pb Z's on {pb.n_bg} backgrounds")
        for c in range(len(PT_TAGS)):
            r = ref[c]
            t, te = v[wd.PIECES[0]][c][pos], e[wd.PIECES[0]][c][pos]
            chi2 = float(np.sum((t - r[1][r[0] > 0]) ** 2 / (te ** 2 + r[2][r[0] > 0] ** 2)))
            parts = []
            for title, keys in SUMS:
                x, xe = summed(v, e, keys)
                parts.append(f"{title}: first {x[c][first]:+.3f}±{xe[c][first]:.3f}, integrated "
                             f"{a.dy * x[c][pos].sum():+.3f}±"
                             f"{a.dy * np.sqrt((xe[c][pos] ** 2).sum()):.3f}")
            L.append(f"  {PT_LABELS[c]:20s} chi2 {chi2:5.1f}/{int(pos.sum())} | " + " | ".join(parts))
        L.append("")
    L.append("CMS Pb+Pb 0-30 % - pp: " + " | ".join(
        f"{PT_LABELS[c]}: first {ref[c][1][ref[c][0] > 0][0]:+.3f}±{ref[c][2][ref[c][0] > 0][0]:.3f}"
        f", integrated {a.dy * ref[c][1][ref[c][0] > 0].sum():+.3f}±"
        f"{a.dy * np.sqrt((ref[c][2][ref[c][0] > 0] ** 2).sum()):.3f}"
        for c in range(len(PT_TAGS))))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = figdata.subplots(len(SUMS), len(PT_TAGS), figsize=(17, 4.2 * len(SUMS)),
                               squeeze=False)
    for r_, (title, keys) in enumerate(SUMS):
        for c in range(len(PT_TAGS)):
            aa = ax[r_, c]
            if r_ == 0:
                cr = ref[c]
                h = cr[0] > 0
                aa.errorbar(cr[0][h], cr[1][h], cr[2][h], fmt="D", color="0.5", ms=5,
                            label="CMS Pb+Pb 0–30 % − pp")
            for i, (name, label, run, pb, pp, v, e) in enumerate(res):
                x, xe = summed(v, e, keys)
                aa.errorbar(centres[pos] + 0.04 * i, x[c][pos], xe[c][pos], fmt="s-", ms=4,
                            lw=1.1, color=COLS[i], label=label)
            aa.axhline(0, color="0.6", lw=0.6)
            aa.set_title(f"{title}\n{PT_LABELS[c]}", fontsize=9)
            aa.set(xlabel="Δy_ch,Z (Z side, |Δφ| < π/2)", ylabel="d⟨ΔN_ch⟩/dΔy")
            if c == 0:
                aa.legend(fontsize=7)
    which = (f"all against the pp {os.path.basename(os.path.dirname(a.pp_all.rstrip('/')))}"
             if a.pp_all else "each with its own pp")
    fig.suptitle(f"CMS's Z-side Δy, Pb+Pb 0–30 % − pp, per Pythia setting ({which}), "
                 f"e_threshold {a.e.replace('p', '.')}: the total and its fragments' and "
                 "wake's parts", fontsize=10)
    fig.tight_layout()
    os.makedirs(a.outdir, exist_ok=True)
    tag = a.tag if a.tag is not None else (("" if a.e == "2p0" else f"_e{a.e}")
                                            + ("_pp_all" if a.pp_all else ""))
    p = os.path.join(a.outdir, f"pythia_cms{tag}.png")
    figdata.savefig(fig, p, dpi=90)
    plt.close(fig)
    txt = os.path.join(a.outdir, f"pythia_cms{tag}.txt")
    with open(txt, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"-> {p}\n-> {txt}")


if __name__ == "__main__":
    main()
