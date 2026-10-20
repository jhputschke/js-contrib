"""The bulk's closure (B) of deta.py against a permutation null: every hydro background's bulk
in lab eta (its Z events' near-side L_full - L_zc, the mean over its Z's; in lab eta, since
the Delta y contents of Z's at different y_Z do not average) handed to the Z's of another
background (a random derangement of the backgrounds), shifted to each Z's rapidity as the mixed
event's kernel (da.shift_density); the Z's own y, weight, class and reflection kept.  The null
breaks any correlation between the bulk and the jet2 selection (fake jet2s from the bulk's own
fluctuations, the reflection by jet2's side) and keeps the noise of the bulk's event-by-event
eta shape.  Its mean tests the method on an uncorrelated bulk (must be 0), its spread is the
true noise of B's integral (all bins correlated), and the real B against it tells a fluctuation
from a selection bias.

    python deta_null.py --pools <dir...> --jets <dir...> --out <file.txt> [--perms 200]
                        [--mix-classes 10] [--ref-jets]"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))         # deta (dijet/)
import deta  # noqa: E402
from jetcorr import deta_asym as da  # noqa: E402

GAPS = [f"{lo:.1f}-{hi:.1f}" for lo, hi in da.GAPS[1:]]


def integrals(s, X, w):
    f, r, _, _ = s.classes["gap"]
    num, den = da.class_sums(X, w, f, r)
    me, _ = da.class_sums(s.K["pooled"], w, f, r)
    yc, acc = da.corrected(num, den, me, s.norm_bins, with_acceptance=True)
    dif, _ = da.asym_minus_sym(yc, acceptance=acc)
    return da.integral(dif[:, :2])                                 # (3 classes, 2 p_T)


def main():
    ap = argparse.ArgumentParser()
    for k in ("pools", "jets"):
        ap.add_argument(f"--{k}", nargs="+", required=True)
    for k in ("vacuum-pools", "pp-jets"):                             # accepted, not used
        ap.add_argument(f"--{k}", nargs="+", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mix-classes", type=int, default=10, help="as deta.py's")
    ap.add_argument("--ref-jets", action="store_true",
                    help="classify by the map step's reference jets (emb.ref_*: the fragments "
                         "alone, no bulk) instead of the full events' jets")
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    a.zpt, a.jet2, a.ymax, a.xcut, a.jet1, a.jet2_pick = 60.0, 30.0, 1.0, True, "bare", "leading"
    a.central, a.ncoll_min = False, deta.NCOLL_10
    import zjet_jets as zj
    import zjet_mixing as zm
    pbp = zm.Pool.load(deta.pool_paths(a.pools))
    Eb, Jb, Bb = zj.load_jets(a.jets)
    if a.ref_jets:                                  # no bulk in the selection
        Jb = dict(Jb)
        for k in ("pt", "eta", "phi"):
            Jb[f"jet_{k}"] = Jb[f"emb_ref_{k}"]
        Jb["jet_pt"] = Jb["jet_pt"] * (abs(Jb["jet_eta"]) < 2.0)   # the map step's |eta| < 2
        Bb = None
    pb = deta.Sample("Pb+Pb", pbp, Eb, Jb, a, False, Bb)
    L = np.asarray(pb.L["B"], float)[:, :, 0]                     # (n, N_PT, N_ETA), near side
    ub, inv = np.unique(pb.groups, return_inverse=True)
    inv = inv.ravel()
    cnt = np.bincount(inv, minlength=len(ub)).astype(float)
    Lbg = np.zeros((len(ub),) + L.shape[1:])
    np.add.at(Lbg, inv, L)
    Lbg /= cnt[:, None, None]
    members = [np.flatnonzero(inv == b) for b in range(len(ub))]
    shape = (pb.pool.n, L.shape[1], 2, da.N_DY)

    def bulk(src):
        """Every Z on background b gets background src[b]'s lab bulk at its own y."""
        X = np.zeros(shape)
        for b, m in enumerate(members):
            X[m, :, 0] = da.shift_density(Lbg[src[b]], pb.y[m])
        return X
    lines = [f"deta_null: {a.jets}", f"{pb.n_eff(pb.sel.astype(float)):.0f} effective Z's, "
             f"{len(ub)} backgrounds"]
    real = integrals(pb, np.asarray(pb.Y["B"], float), pb.w)
    own = integrals(pb, bulk(np.arange(len(ub))), pb.w)           # the backgrounds' own means
    rng = np.random.default_rng(a.seed)
    null = []
    for _ in range(a.perms):
        while True:                                               # a derangement
            p = rng.permutation(len(ub))
            if not np.any(p == np.arange(len(ub))):
                break
        null.append(integrals(pb, bulk(p), pb.w))
    null = np.array(null)
    mu, sd = null.mean(axis=0), null.std(axis=0, ddof=1)
    lines.append(f"B's integral over -2.5 < Delta y < -0.5 (per unit Delta y) against {a.perms} "
                 "derangements of the backgrounds' bulk: real B (own: the backgrounds' own lab means, shifted), null "
                 "mean ± sd (its error ± sd/sqrt(perms)), z = (real - null mean) / null sd")
    for p_, pt in enumerate(("1-2 GeV", "2-4 GeV")):
        for i, g in enumerate(GAPS):
            lines.append(f"  {pt} {g}: real {real[i, p_]:+8.3f} ({own[i, p_]:+8.3f}); null "
                         f"{mu[i, p_]:+7.3f} ± {sd[i, p_]:6.3f} (±{sd[i, p_] / np.sqrt(a.perms):.3f})"
                         f"; z {(real[i, p_] - mu[i, p_]) / sd[i, p_]:+.2f}")
    zs = (real - mu) / sd
    lines.append(f"  z over the 6 (class, p_T): mean {zs.mean():+.2f}, rms {np.sqrt((zs ** 2).mean()):.2f}"
                 " (classes share R_sym: not independent)")
    text = "\n".join(lines) + "\n"
    with open(a.out, "w") as fo:
        fo.write(text)
    np.savez_compressed(a.out + ".npz", real=real, own=own, null=null)
    print(text, end="")


if __name__ == "__main__":
    main()
