"""Per gap class the Z-side yields of deta.py's truth contents before R_asym - R_sym and the
sideband: where the model's wake (W) and wake + fragments (T) sit in Delta y for each jet2
rapidity (baseline cuts), with pp's F; jackknife errors.  For the interpretation of a missing
dip in R_asym - R_sym: whether the wake's depletion follows jet2 or stays at the Z.

    python deta_classes.py --pools <dir...> --vacuum-pools <dir...> --jets <dir...> \\
                           --pp-jets <dir...> --out <png> [--tag text] [--mix-classes 10]"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))         # deta (dijet/)
import deta  # noqa: E402
from jetcorr import deta_asym as da, figdata  # noqa: E402

GAPS = [f"{lo:.1f}-{hi:.1f}" for lo, hi in da.GAPS]
PTS = ("1-2 GeV", "2-4 GeV")


def main():
    ap = argparse.ArgumentParser()
    for k in ("pools", "vacuum-pools", "jets", "pp-jets"):
        ap.add_argument(f"--{k}", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mix-classes", type=int, default=10, help="as deta.py's")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    a.zpt, a.jet2, a.ymax, a.xcut, a.jet1, a.jet2_pick = 60.0, 30.0, 1.0, True, "bare", "leading"
    a.central, a.ncoll_min, a.blocks = False, deta.NCOLL_10, 100
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import zjet_jets as zj
    import zjet_mixing as zm
    pbp, ppp = zm.Pool.load(deta.pool_paths(a.pools)), zm.Pool.load(deta.pool_paths(a.vacuum_pools))
    Eb, Jb, Bb = zj.load_jets(a.jets)
    Ep, Jp, _ = zj.load_jets(a.pp_jets)
    pb = deta.Sample("Pb+Pb", pbp, Eb, Jb, a, False, Bb)
    pp = deta.Sample("pp", ppp, Ep, Jp, a, True)
    res = {}
    for s, c in ((pb, "W"), (pb, "T"), (pp, "F")):
        fn, shp = s.observable(c, subtract=False)
        v, e = da.jackknife(fn, da.blocks_of(s.groups, a.blocks))
        res[(s.label, c)] = (deta.split(v, shp)[1], deta.split(e, shp)[1])   # yc (4, N_PT, N_DY)
    x02 = da.rebin(np.zeros(da.N_DY))[1]
    lines = [f"deta_classes {a.tag}: per class, Z side, per trigger per unit Delta y, no sideband "
             "subtraction, 1-2 GeV; mean over Delta y windows (0.2 bins; reflected: jet2 at "
             "Delta y < 0); jet2's mean Delta y per class is about -(gap centre)"]
    wins = ((-2.5, -1.5), (-1.5, -0.5), (-0.5, 0.5), (0.5, 1.5), (1.5, 2.5))
    lines.append("  " + " " * 18 + "".join(f"{lo:+.1f}..{hi:+.1f}      " for lo, hi in wins))
    fig, ax = figdata.subplots(2, 3, figsize=(16, 8), sharex=True)
    for col, (key, title) in enumerate(((("Pb+Pb", "W"), "Pb+Pb W: the wake alone"),
                                        (("Pb+Pb", "T"), "Pb+Pb T: wake + fragments"),
                                        (("pp", "F"), "pp"))):
        yc, ye = res[key]
        for ip in range(2):
            axr = ax[ip, col]
            for c, g in enumerate(GAPS):
                y = da.rebin(yc[c, ip])[0]
                e = np.sqrt(da.rebin(ye[c, ip] ** 2)[0] / 2)
                axr.errorbar(x02 + 0.02 * (c - 1.5), y, e, fmt="o", ms=3, color=f"C{c}",
                             label=f"|Δη| {g}" + (" (R_sym)" if c == 0 else ""))
                if ip == 0:
                    m = [np.nanmean(y[(x02 > lo) & (x02 < hi)]) for lo, hi in wins]
                    em = [np.sqrt(np.nansum(e[(x02 > lo) & (x02 < hi)] ** 2)) / 5 for lo, hi in wins]
                    lines.append(f"  {key[0]:5s} {key[1]} {g:8s}  " + "".join(
                        f"{mm:+.3f}±{ee:.3f}  " for mm, ee in zip(m, em)))
            axr.axhline(0, color="0.5", lw=0.8)
            axr.set_title(f"{title}, {PTS[ip]}")
            axr.set_xlabel("Δy = η − y_Z (reflected: jet2 at Δy < 0)")
            axr.set_ylabel("per Z per unit Δy (no sideband)")
            axr.legend(fontsize=7)
    fig.suptitle(f"Per gap class, before R_asym − R_sym and the sideband {a.tag}", y=0.995)
    fig.tight_layout()
    figdata.savefig(fig, a.out, dpi=100)
    text = "\n".join(lines) + "\n"
    with open(os.path.splitext(a.out)[0] + ".txt", "w") as fo:
        fo.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
