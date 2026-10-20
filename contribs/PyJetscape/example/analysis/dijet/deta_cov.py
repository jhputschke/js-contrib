"""deta.py's closure and signal with the full jackknife covariance of the bins: the integral over
-2.5 < Delta y < -0.5 jackknifed as a whole (deta.py adds the bins' errors in quadrature, as
if independent) and chi^2 against 0 over CMS's 10 bins of 0.2 with the bins' covariance
(deta.py: the diagonal only).  A coherent shift of the bulk's eta shape moves all bins
together; this tells a fluctuation from a bias.

    python deta_cov.py --pools <dir...> --vacuum-pools <dir...> --jets <dir...> \\
                       --pp-jets <dir...> --out <file.txt> [deta.py's cut options]
                       [--mix-classes 10] [--ref-jets]

Writes <out> (text) and <out>.npz (values, errors, replicas)."""
import argparse
import os
import sys

import numpy as np
from scipy.stats import chi2 as chi2d, f as fd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))         # deta (dijet/)
import deta  # noqa: E402  (puts jetcorr, zjet and the jetscape package on the path)
from jetcorr import deta_asym as da  # noqa: E402

GAPS = [f"{lo:.1f}-{hi:.1f}" for lo, hi in da.GAPS[1:]]
PTS = ("1-2 GeV", "2-4 GeV")
SIG = (da.rebin(np.zeros(da.N_DY))[1] > -2.5) & (da.rebin(np.zeros(da.N_DY))[1] < -0.5)


def reduce(dif):
    """dif (3, N_PT, N_DY) -> per class and p_T (1-2, 2-4): the 10 signal bins of 0.2 and the
    integral (per unit Delta y, deta.py's), flattened (3, 2, 11)."""
    out = np.empty((3, 2, 11))
    for i in range(3):
        for p in range(2):
            out[i, p, :10] = da.rebin(dif[i, p])[0][SIG]
            out[i, p, 10] = da.integral(dif[i, p])
    return out


def jk(fn, block):
    ids = np.unique(block)
    full = fn(np.ones(len(block), bool))
    reps = np.array([fn(block != b) for b in ids])
    return full, reps


def cov(reps):
    g = len(reps)
    d = reps - reps.mean(axis=0)
    return (g - 1) / g * np.einsum("r...i,r...j->...ij", d, d), g


def tests(x, C, g, name, lines):
    """x (3, 2, 11), C (3, 2, 11, 11): the integral and chi^2 with the full covariance."""
    for p in range(2):
        cells = []
        for i in range(3):
            v, c = x[i, p], C[i, p]
            e_int = np.sqrt(c[10, 10])
            e_diag = 0.2 * np.sqrt(np.sum(np.diag(c)[:10]))      # deta.py's (bins independent)
            m = np.isfinite(v[:10])
            vv, cc = v[:10][m], c[:10, :10][np.ix_(m, m)]
            t2 = float(vv @ np.linalg.solve(cc, vv))
            k = len(vv)
            fstat = (g - k) / (k * (g - 1)) * t2                 # Hotelling: estimated cov
            p_f = fd.sf(fstat, k, g - k)
            chi_d = float(np.sum(vv ** 2 / np.diag(cc)))
            cells.append(f"{GAPS[i]}: int {v[10]:+.3f} ± {e_int:.3f} (diag {e_diag:.3f}, "
                         f"{v[10] / e_int:+.1f}σ); chi2 full {t2:.1f}/{k} (p {p_f:.2f}), "
                         f"diag {chi_d:.1f}/{k}")
        lines.append(f"  {name:22s} {PTS[p]}: " + "; ".join(cells))


def main():
    ap = argparse.ArgumentParser()
    for k in ("pools", "vacuum-pools", "jets", "pp-jets"):
        ap.add_argument(f"--{k}", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref-jets", action="store_true",
                    help="classify by the map step's reference jets (emb.ref_*: the fragments "
                         "alone, no bulk) instead of the full events' jets")
    ap.add_argument("--zpt", type=float, default=60.0)
    ap.add_argument("--jet2", type=float, default=30.0)
    ap.add_argument("--ymax", type=float, default=1.0)
    ap.add_argument("--no-xcut", dest="xcut", action="store_false")
    ap.add_argument("--jet1", default="bare")
    ap.add_argument("--jet2-pick", default="leading")
    ap.add_argument("--central", action="store_true")
    ap.add_argument("--ncoll-min", type=float, default=deta.NCOLL_10)
    ap.add_argument("--blocks", type=int, default=100)
    ap.add_argument("--mix-classes", type=int, default=10, help="as deta.py's")
    a = ap.parse_args()
    import zjet_jets as zj
    import zjet_mixing as zm
    pbp = zm.Pool.load(deta.pool_paths(a.pools))
    ppp = zm.Pool.load(deta.pool_paths(a.vacuum_pools))
    Eb, Jb, Bb = zj.load_jets(a.jets)
    if a.ref_jets:                                  # no bulk in the selection
        Jb = dict(Jb)
        for k in ("pt", "eta", "phi"):
            Jb[f"jet_{k}"] = Jb[f"emb_ref_{k}"]
        Jb["jet_pt"] = Jb["jet_pt"] * (abs(Jb["jet_eta"]) < 2.0)   # the map step's |eta| < 2
        Bb = None
    Ep, Jp, _ = zj.load_jets(a.pp_jets)
    pb = deta.Sample("Pb+Pb", pbp, Eb, Jb, a, False, Bb)
    pp = deta.Sample("pp", ppp, Ep, Jp, a, True)
    bpb, bpp = da.blocks_of(pb.groups, a.blocks), da.blocks_of(pp.groups, a.blocks)

    def dif_fn(s, c):
        fn, shp = s.observable(c, strict=(c == "O"))
        return lambda keep: reduce(deta.split(fn(keep), shp)[0])

    R = {}
    for c in ("T", "W", "F", "N", "B", "O"):
        R[("Pb+Pb", c)] = jk(dif_fn(pb, c), bpb)
    fF, fT = dif_fn(pb, "F"), dif_fn(pb, "T")
    R[("Pb+Pb", "F - T")] = jk(lambda keep: fF(keep) - fT(keep), bpb)   # same blocks
    for c in ("F", "O"):
        R[("pp", c)] = jk(dif_fn(pp, c), bpp)

    lines = [f"deta_cov: {a.jets} | cuts zpt {a.zpt:g}, jet2 {a.jet2:g}, xcut {a.xcut}, jet1 "
             f"{a.jet1}, pick {a.jet2_pick}, central {a.central}",
             f"Pb+Pb {int(pb.sel.sum())} Z's on {len(np.unique(pb.groups[pb.sel]))} backgrounds, "
             f"{len(np.unique(bpb))} jackknife blocks; pp {int(pp.sel.sum())} Z's",
             "R_asym - R_sym over -2.5 < Delta y < -0.5: the integral (per unit Delta y) with its "
             "jackknife error as a whole (diag: deta.py's, bins independent), and chi^2 against 0 "
             "over the 10 bins of 0.2 with the full covariance (p: Hotelling F, the covariance "
             "estimated from the blocks) and diagonal only:"]
    npz = {}
    for key, (x, reps) in R.items():
        C, g = cov(reps)
        tests(x, C, g, " ".join(key), lines)
        npz[" ".join(key)] = x
        npz[" ".join(key) + " reps"] = reps
    for c, cp in (("F", "F"), ("T", "F"), ("O", "O")):
        x = R[("Pb+Pb", c)][0] - R[("pp", cp)][0]
        C = cov(R[("Pb+Pb", c)][1])[0] + cov(R[("pp", cp)][1])[0]
        g = min(len(R[("Pb+Pb", c)][1]), len(R[("pp", cp)][1]))
        tests(x, C, g, f"Pb+Pb {c} - pp {cp}", lines)
    # the bin-to-bin correlation of B's 10 bins (class 0.5-1.0, 1-2 GeV): how coherent
    Cb = cov(R[("Pb+Pb", "B")][1])[0][0, 0, :10, :10]
    d = np.sqrt(np.diag(Cb))
    rho = Cb / np.outer(d, d)
    lines.append(f"  B, 0.5-1.0, 1-2 GeV: mean bin-to-bin correlation of the 10 bins "
                 f"{(rho.sum() - 10) / 90:+.2f}")
    text = "\n".join(lines) + "\n"
    with open(a.out, "w") as fo:
        fo.write(text)
    np.savez_compressed(a.out + ".npz", **npz)
    print(text, end="")


if __name__ == "__main__":
    main()
