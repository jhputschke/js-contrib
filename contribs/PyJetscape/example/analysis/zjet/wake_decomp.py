#!/usr/bin/env python3
"""The Z-side Delta y of Pb+Pb - pp (CMS 2507.09307, mix.py's factorized step 1) split into
the fragments' and the wake's parts, each into its Z-side yield and its level, per track p_T
class:

  D_PbPb - D_pp = [S_frag,PbPb - S_pp]  -  [B_frag,PbPb - B_pp]  +  S_wake  -  B_wake
                  fragments, Z side       fragments, level        wake,     wake,
                                                                  Z side    level

    python wake_decomp.py --pbpb <pool dir> [<pool dir> ...] --pp <pool dir> [...] \\
        [--run <run>] [--tag _c00-01] [--blocks 100] [--dy 0.8]

S is the Z-side (|Delta phi| < pi/2) yield per Z, B the mixed event: half of the other Z
events' pooled lab-eta density folded with each y_Z, the own event left out (MIXING_MODES.md
section 4, the rules of mix.py's factorized result); +-Delta y merged and scaled by 0.5, per
unit Delta y.  The contents are those of the version-4 pools (mix.py --pools-only or
zhadron_pools: Y_*, L_* on the 0.1 Delta y grid): the Z-correlated yield Y_zc (bulk_jet -
bulk_bg sample by sample + fragments), the fragments Y_frag, and the wake Y_zc - Y_frag; pp
its fragments (Y_full).  The level terms are the mean subtraction of CMS's observable: a
content's B integrates to half its total yield per Z, so tracks added on the jet side lower the
Z side and tracks removed there raise it.  The four pieces add up to Pb+Pb - pp exactly
(checked).  Errors: a jackknife over blocks of Pb+Pb hydro backgrounds and over blocks of pp
events, in quadrature.  CMS's Pb+Pb 0-30 % - pp (HEPData ins2945894, pp merged to 0.8) for
reference.
-> runs/<run>/figures/wake_decomp{tag}.png (with .npz/.root, jetcorr.figdata) and
   wake_decomp{tag}.txt."""
import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                        # jetcorr
import mix  # noqa: E402,F401  (puts PyJetscape/python on the path when needed)
import zjet_mixing as zm  # noqa: E402
from cms2507 import PT_LABELS, PT_TAGS, cms  # noqa: E402
from jetcorr import deta_asym as da  # noqa: E402
from jetcorr import figdata, runs  # noqa: E402

PIECES = ("Pb+Pb − pp", "fragments, Z side", "fragments, level", "wake, Z side", "wake, level")
YMAX = 4.0                                     # CMS's |Delta y| range


def load(dirs):
    paths = []
    for d in dirs:
        found = sorted(glob.glob(os.path.join(d, "*_pool.root")) or
                       glob.glob(os.path.join(d, "*.npz")))
        if not found:
            raise SystemExit(f"no pools (*.npz, *_pool.root) in {d}")
        paths += found
    return zm.Pool.load(paths)


def self_shift(Lt, y):
    """shift_i(Lt_i) for every event: its own lab density seen from its own Z (the self pair
    the factorized Delta y leaves out) -> (n, N_PT, N_DY)."""
    j = np.clip(np.searchsorted(da.DY, da.ETAC[None, :] - y[:, None], side="right") - 1, 0,
                da.N_DY - 1)
    n, n_pt, n_eta = Lt.shape
    out = np.zeros((n, n_pt, da.N_DY))
    rows = np.repeat(np.arange(n), n_eta)
    for p in range(n_pt):
        np.add.at(out[:, p], (rows, j.ravel()), Lt[:, p, :].ravel())
    return out


def s_and_b(Y, L, w, y, keep):
    """Z-side S and B per Z per unit Delta y, +-merged x 0.5 -> (N_PT, N_DY) each.  Y: (n,
    N_PT, 2, N_DY) in the Z frame per side, L: (n, N_PT, 2, N_ETA) lab eta per side."""
    Y, L = Y[keep].astype(float), L[keep].astype(float)
    w, y = np.asarray(w, float)[keep], np.asarray(y, float)[keep]
    W, W2 = w.sum(), (w * w).sum()
    S = np.tensordot(w, Y[:, :, 0, :], axes=1) / W
    Lt = L.sum(axis=2)                       # a partner's tracks: both sides of the Z_i
    pair = np.tensordot(w, da.shift_density(np.tensordot(w, Lt, axes=1), y), axes=1)
    own = np.tensordot(w * w, self_shift(Lt, y), axes=1)
    B = 0.5 * (pair - own) / (W * W - W2)    # the Z side: half of an uncorrelated partner

    def sym(x):
        return 0.5 * (x + x[..., ::-1]) / da.WIDTH
    return sym(S), sym(B)


def rebin(x, width):
    """|Delta y| < 4 in bins of ``width`` (the mean of its 0.1 bins)."""
    k = int(round(width / da.WIDTH))
    lo = int(round((YMAX + da.DY[0]) / -da.WIDTH))           # the first bin at -4
    x = x[..., lo:lo + int(round(2 * YMAX / da.WIDTH))]
    return x.reshape(x.shape[:-1] + (x.shape[-1] // k, k)).mean(-1)


def pieces(pb, pp, keep_pb, keep_pp, width):
    S_t, B_t = s_and_b(pb.Y_zc, pb.L_zc, pb.w, pb.y, keep_pb)
    S_f, B_f = s_and_b(pb.Y_frag, pb.L_frag, pb.w, pb.y, keep_pb)
    S_p, B_p = s_and_b(pp.Y_full, pp.L_full, pp.w, pp.y, keep_pp)
    out = dict(zip(PIECES, ((S_t - B_t) - (S_p - B_p), S_f - S_p, -(B_f - B_p), S_t - S_f,
                            -(B_t - B_f))))
    return {k: rebin(v, width) for k, v in out.items()}


def with_errors(pb, pp, width, n_blocks):
    """The pieces and their jackknife errors (Pb+Pb backgrounds, pp events, in quadrature)."""
    all_pb, all_pp = np.ones(pb.n, bool), np.ones(pp.n, bool)
    v = pieces(pb, pp, all_pb, all_pp, width)
    var = {k: 0.0 for k in v}
    for pool, side in ((pb, "pbpb"), (pp, "pp")):
        units = np.unique(pool.zb)
        blocks = [b for b in np.array_split(units, min(n_blocks, len(units))) if len(b)]
        G = len(blocks)
        if G < 2:
            continue
        reps = []
        for b in blocks:
            keep = ~np.isin(pool.zb, b)
            reps.append(pieces(pb, pp, keep if side == "pbpb" else all_pb,
                               keep if side == "pp" else all_pp, width))
        for k in v:
            r = np.array([x[k] for x in reps])
            var[k] = var[k] + (G - 1) / G * np.sum((r - r.mean(0)) ** 2, axis=0)
    return v, {k: np.sqrt(x) for k, x in var.items()}


def cms_difference(hep, c, width):
    """CMS's Pb+Pb 0-30 % - pp on the Z side (pp's 0.4 bins merged to 0.8) -> centres, value,
    error (stat + syst of both in quadrature); None unless ``width`` is 0.8."""
    if abs(width - 0.8) > 1e-9:
        return None
    cx, yb, sb, syb = cms(hep, "y", "pbpb", PT_TAGS[c])
    _, yq, sq, syq = cms(hep, "y", "pp", PT_TAGS[c])
    yq2 = yq.reshape(-1, 2).mean(1)
    tq2 = np.sqrt((np.hypot(sq, syq).reshape(-1, 2) ** 2).sum(1)) / 2
    return cx, yb - yq2, np.hypot(np.hypot(sb, syb), tq2)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pbpb", nargs="+", required=True,
                    help="Pb+Pb pool directories (version 4; several: campaigns built apart)")
    ap.add_argument("--pp", nargs="+", required=True, help="pp pool directories (version 4)")
    ap.add_argument("--run", default="zjet_test", help="runs/<run>/figures is the default --outdir")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--tag", default="", help="suffix of the output names")
    ap.add_argument("--hepdata", default=runs.hepdata("HEPData-ins2945894-v1-yaml"))
    ap.add_argument("--blocks", type=int, default=100, help="jackknife blocks")
    ap.add_argument("--dy", type=float, default=0.8,
                    help="Delta y bin width (CMS Pb+Pb: 0.8, pp: 0.4)")
    a = ap.parse_args()
    a.outdir = a.outdir or runs.figures(a.run)
    pb, pp = load(a.pbpb), load(a.pp)
    for p, name in ((pb, "--pbpb"), (pp, "--pp")):
        need = ("Y_zc", "Y_frag", "L_zc", "L_frag") if p is pb else ("Y_full", "L_full")
        if not all(hasattr(p, k) for k in need):
            raise SystemExit(f"{name}: pools without {'/'.join(need)} (version 4 needed: rebuild "
                             "with mix.py --pools-only or zhadron_pools)")
    v, e = with_errors(pb, pp, a.dy, a.blocks)
    total = sum(v[k] for k in PIECES[1:])
    if not np.allclose(total, v[PIECES[0]], atol=1e-9):
        raise SystemExit("the pieces do not add up to Pb+Pb - pp")
    n_bins = int(round(2 * YMAX / a.dy))
    centres = -YMAX + a.dy * (np.arange(n_bins) + 0.5)
    pos = centres > 0
    first = int(np.argmax(pos))
    lines = [f"Pb+Pb: {pb.n} Z's on {pb.n_bg} hydro backgrounds (N_eff {zm.n_eff(pb.w):.0f}) from "
             f"{' '.join(a.pbpb)}; pp: {pp.n} Z's from {' '.join(a.pp)}",
             "Z side, per Z per unit Delta y, +-merged x 0.5; errors: jackknife over blocks of "
             "Pb+Pb backgrounds and pp events", "",
             f"First bin past 0 (0 < Delta y < {a.dy:g}):"]
    ref = [cms_difference(a.hepdata, c, a.dy) for c in range(len(PT_TAGS))]
    for c in range(len(PT_TAGS)):
        r = ref[c]
        head = (f"CMS {r[1][r[0] > 0][0]:+.3f}±{r[2][r[0] > 0][0]:.3f} | " if r else "")
        lines.append(f"  {PT_LABELS[c]:20s} {head}" + " | ".join(
            f"{k} {v[k][c][first]:+.3f}±{e[k][c][first]:.3f}" for k in PIECES))
    lines += ["", f"Integrated over 0 < Delta y < {YMAX:g} (per Z; a content's Z-side integral "
              "minus its level is half its Z-side minus away-side yield):"]
    for c in range(len(PT_TAGS)):
        lines.append(f"  {PT_LABELS[c]:20s} " + " | ".join(
            f"{k} {a.dy * v[k][c][pos].sum():+.3f}±{a.dy * np.sqrt((e[k][c][pos] ** 2).sum()):.3f}"
            for k in PIECES))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style = {PIECES[0]: ("k", "s", "-"), PIECES[1]: ("C0", "o", "--"),
             PIECES[2]: ("C0", "v", ":"), PIECES[3]: ("C3", "o", "--"),
             PIECES[4]: ("C3", "v", ":")}
    fig, ax = figdata.subplots(1, len(PT_TAGS), figsize=(17, 5.2))
    for c in range(len(PT_TAGS)):
        r = ref[c]
        if r:
            h = r[0] > 0
            ax[c].errorbar(r[0][h], r[1][h], r[2][h], fmt="D", color="0.5", ms=5,
                           label="CMS Pb+Pb 0–30 % − pp")
        for i, (k, (col, mk, ls)) in enumerate(style.items()):
            ax[c].errorbar(centres[pos] + 0.03 * i, v[k][c][pos], e[k][c][pos], fmt=mk + ls,
                           color=col, ms=4, mfc="none" if "level" in k else col,
                           lw=1.2 if k == PIECES[0] else 0.8, label=k)
        ax[c].axhline(0, color="0.6", lw=0.6)
        ax[c].set_title(PT_LABELS[c])
        ax[c].set_xlabel("Δy_ch,Z (Z side, |Δφ| < π/2)")
        ax[c].set_ylabel("Pb+Pb − pp: d⟨ΔN_ch⟩/dΔy")
        ax[c].legend(fontsize=7)
    fig.suptitle(f"Z-side Δy, Pb+Pb 0–30 % − pp, split into the fragments and the wake, each into "
                 f"its Z-side yield and its level (the mean subtraction); {pb.n} Pb+Pb Z's on "
                 f"{pb.n_bg} backgrounds", fontsize=10)
    fig.tight_layout()
    os.makedirs(a.outdir, exist_ok=True)
    path = os.path.join(a.outdir, f"wake_decomp{a.tag}.png")
    figdata.savefig(fig, path, dpi=100)
    plt.close(fig)
    txt = os.path.join(a.outdir, f"wake_decomp{a.tag}.txt")
    with open(txt, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"-> {path}\n-> {txt}")


if __name__ == "__main__":
    main()
