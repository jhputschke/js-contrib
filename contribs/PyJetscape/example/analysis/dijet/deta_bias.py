"""Does jet2's selection prefer the thinner side of its own hydro background, and where does
deta.py's bulk residual (B, the bulk alone through R_asym - R_sym) come from?

For every selected oversample (deta.py's baseline cuts: the Z of the pools with |y_Z| < 1 and
p_T^Z > 60, the leading jet2 above --jet2 with Delta phi > 5 pi / 6 and jet2 < jet1) the
contrast of its own background's bulk at jet2's lab eta against the mirror point,

    d = [rho_b(eta2) - rho_b(-eta2)] / [rho_b(eta2) + rho_b(-eta2)],

rho_b the background's mean lab density of the bulk (L_full - L_zc, both sides of the Z, the
mean over its Z's; 1-2 and 2-4 GeV) in a window of +-0.25 around +-eta2 (|eta2| > 0.3).  Weighted
mean over the rows (event weight / its oversamples) against a null: the same rows with the
bulk of another background (100 derangements).  d < 0: jet2 sits where its own medium is thinner.
Broken down by gap class, |eta_jet2|, centrality (N_coll above / below 1210), and with jet2
from the full event or from the bulk-free reference jets (emb.ref_*), at --jet2 30 and 50.

Then B (the bulk's R_asym - R_sym integral over -2.5 < Delta y < -0.5) with one mixed event for
all centralities, as deta.py had before --mix-classes: for central and less central events,
jackknifed as a whole, against the derangement null (deta_null.py's) and against a null
deranged only within --mix-classes N_coll classes (which keeps each Z's centrality); the bulk's
lab dN/deta shape against centrality; and B and F with the mixed event of each event's own
N_coll class (deta.mixed_event_of, deta.py's default since 2026-10-20) against the one for all.
On c00 + c01: no thinner side (d = 0 within 1e-3); the residual follows centrality, the bulk
narrows in eta with it, and the matched mixed event removes most of B and its noise.

    python deta_bias.py --pools <dir...> --jets <dir...> --out <file.txt> [--perms 100]
                        [--mix-classes 10]"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))         # deta (dijet/)
import deta  # noqa: E402
from jetcorr import deta_asym as da  # noqa: E402

GAPS = [f"{lo:.1f}-{hi:.1f}" for lo, hi in da.GAPS]
WIN = 0.25


def rows_of(pool, E, J, zpt, jet2, ymax=1.0):
    """deta.Sample's selection, kept per oversample: (event index, eta2, class) of the rows
    that pass, and each row's weight (event weight / the event's oversamples)."""
    idx = {int(g): i for i, g in enumerate(E["event"])}
    ie = np.array([idx[int(g)] for g in pool.event])
    y, phi = np.asarray(pool.y, float), np.asarray(pool.phi, float)
    sel = np.abs(y) < ymax
    pos = {int(r): i for i, r in enumerate(ie)}
    rows = np.flatnonzero(np.isin(J["event"], E["event"][ie[sel]]))
    ev = np.array([pos[idx[int(g)]] for g in J["event"][rows]], np.int64)
    pt2, eta2, phi2 = deta.leading_jet2(J, rows, phi[ev], "leading")
    pt1 = np.asarray(E["z_pt"], float)[ie][ev]
    ok = (pt1 > zpt) & (pt2 > jet2) & (deta.dphi(phi2, phi[ev]) > 5 * np.pi / 6) & (pt2 < pt1)
    n_rows = np.bincount(ev, minlength=pool.n).astype(float)
    cls, _ = da.gap_class(y[ev], np.where(ok, eta2, np.nan))
    m = ok & (cls >= 0)
    return ev[m], eta2[m], cls[m], np.asarray(pool.w, float)[ev[m]] / n_rows[ev[m]]


def contrast(rho, bg_row, eta2):
    """d per row from rho (n_bg, N_PT, N_ETA): the background bg_row's density at +-eta2."""
    a = np.abs(da.ETAC[None, :] - eta2[:, None]) < WIN               # (rows, N_ETA)
    b = np.abs(da.ETAC[None, :] + eta2[:, None]) < WIN
    r = rho[bg_row]                                                  # (rows, N_PT, N_ETA)
    pa = np.einsum("rpe,re->rp", r, a.astype(float))
    pb = np.einsum("rpe,re->rp", r, b.astype(float))
    return (pa - pb) / (pa + pb)                                     # (rows, N_PT)


def derangement(n, rng):
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", nargs="+", required=True)
    ap.add_argument("--jets", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--perms", type=int, default=100)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--mix-classes", type=int, default=10,
                    help="N_coll classes of the matched mixed event and the stratified null")
    a = ap.parse_args()
    import zjet_jets as zj
    import zjet_mixing as zm
    pool = zm.Pool.load(deta.pool_paths(a.pools))
    E, J, Bj = zj.load_jets(a.jets)
    Jref = dict(J)
    for k in ("pt", "eta", "phi"):
        Jref[f"jet_{k}"] = J[f"emb_ref_{k}"]
    Jref["jet_pt"] = Jref["jet_pt"] * (abs(Jref["jet_eta"]) < 2.0)

    # the bulk of every background in lab eta, both sides of the Z, 1-2 and 2-4 GeV
    L = (np.asarray(pool.L_full, float) - pool.L_zc).sum(axis=2)[:, :2]   # (n, 2, N_ETA)
    ub, inv = np.unique(np.asarray(pool.bg), return_inverse=True)
    inv = inv.ravel()
    rho = np.zeros((len(ub),) + L.shape[1:])
    np.add.at(rho, inv, L)
    rho /= np.bincount(inv, minlength=len(ub)).astype(float)[:, None, None]
    ncoll = np.asarray(pool.n_coll, float)
    central = ncoll > deta.NCOLL_10
    rng = np.random.default_rng(a.seed)
    perms = [derangement(len(ub), rng) for _ in range(a.perms)]
    near = rho.mean(axis=0)                                            # pooled, per unit eta
    lines = [f"deta_bias: {a.jets}",
             f"{pool.n} pooled Z's on {len(ub)} backgrounds; the bulk, both sides of the Z, "
             f"|eta| < 0.5, per Z event per unit eta: 1-2 GeV {near[0, np.abs(da.ETAC) < 0.5].mean() / da.WIDTH:.0f}, "
             f"2-4 GeV {near[1, np.abs(da.ETAC) < 0.5].mean() / da.WIDTH:.0f}",
             "",
             "Contrast d = [rho(eta2) - rho(-eta2)] / [rho(eta2) + rho(-eta2)] of jet2's own background "
             f"(windows +-{WIN}, |eta2| > 0.3), weighted mean in units of 1e-3, against "
             f"{a.perms} derangements of the backgrounds: real, null mean ± sd, z"]
    npz = {}

    def report(tag, ev, eta2, cls, wr, mask):
        m = mask & (np.abs(eta2) > 0.3)
        if m.sum() < 50:
            return f"{tag}: too few rows ({m.sum()})"
        bg_row = inv[ev[m]]
        wm = wr[m] / wr[m].sum()
        real = wm @ contrast(rho, bg_row, eta2[m])
        null = np.array([wm @ contrast(rho, p[bg_row], eta2[m]) for p in perms])
        mu, sd = null.mean(0), null.std(0, ddof=1)
        npz[tag] = np.concatenate([real, mu, sd])
        neff = 1.0 / np.sum(wm ** 2)
        return (f"  {tag:44s} rows {m.sum():6d} (N_eff {neff:5.0f}): " + "; ".join(
            f"{pt} {1e3 * real[p]:+6.1f} (null {1e3 * mu[p]:+5.1f} ± {1e3 * sd[p]:4.1f}, z {(real[p] - mu[p]) / sd[p]:+.1f})"
            for p, pt in enumerate(("1-2 GeV", "2-4 GeV"))))

    for jl, JJ in (("full-event jet2", J), ("reference jet2 (no bulk)", Jref)):
        for thr in (30.0, 50.0):
            ev, eta2, cls, wr = rows_of(pool, E, JJ, 60.0, thr)
            lines.append(f"{jl}, p_T^Z > 60, jet2 > {thr:g} GeV:")
            allm = np.ones(len(ev), bool)
            lines.append(report("all", ev, eta2, cls, wr, allm))
            for c, g in enumerate(GAPS):
                lines.append(report(f"class {g}", ev, eta2, cls, wr, cls == c))
            for lo, hi in ((0.3, 1.0), (1.0, 1.5), (1.5, 2.0)):
                lines.append(report(f"|eta_jet2| {lo}-{hi}", ev, eta2, cls, wr,
                                    (np.abs(eta2) >= lo) & (np.abs(eta2) < hi)))
            lines.append(report("N_coll > 1210 (0-10 %-like)", ev, eta2, cls, wr, central[ev]))
            lines.append(report("N_coll < 1210", ev, eta2, cls, wr, ~central[ev]))
    lines.append("")

    # B with one mixed event for all centralities (deta.py before --mix-classes), per centrality
    import zjet_jets  # noqa: F401
    ns = argparse.Namespace(zpt=60.0, jet2=30.0, ymax=1.0, xcut=True, jet1="bare",
                            jet2_pick="leading", central=False, ncoll_min=deta.NCOLL_10, blocks=100,
                            mix_classes=1)
    s = deta.Sample("Pb+Pb", pool, E, J, ns, False, Bj)
    blk = da.blocks_of(s.groups, 100)
    fB, shp = s.observable("B")
    Lh = np.asarray(s.L["B"], float)[:, :, 0]                       # near side, lab
    Lbg = np.zeros((len(ub),) + Lh.shape[1:])
    np.add.at(Lbg, inv, Lh)
    Lbg /= np.bincount(inv, minlength=len(ub)).astype(float)[:, None, None]
    members = [np.flatnonzero(inv == b) for b in range(len(ub))]
    f, r, _, _ = s.classes["gap"]

    def b_int(X, w):
        num, den = da.class_sums(X, w, f, r)
        me, _ = da.class_sums(s.K["pooled"], w, f, r)
        yc, acc = da.corrected(num, den, me, s.norm_bins, with_acceptance=True)
        dif, _ = da.asym_minus_sym(yc, acceptance=acc)
        return da.integral(dif[:, :2])                                  # (3, 2)

    def bulk(src):
        X = np.zeros((pool.n, Lh.shape[1], 2, da.N_DY))
        for b, m in enumerate(members):
            X[m, :, 0] = da.shift_density(Lbg[src[b]], s.y[m])
        return X
    nullX = [bulk(p) for p in perms[:min(a.perms, 60)]]
    lines.append("B (the bulk alone), one mixed event for all centralities: R_asym - R_sym integrated "
                 "over -2.5 < Delta y < -0.5, per centrality; real ± jackknife (integral as a whole), "
                 "derangement null mean ± sd, z:")
    for name, cm in (("all", np.ones(pool.n, bool)), ("N_coll > 1210", central),
                     ("N_coll < 1210", ~central)):
        def fn(keep, cm=cm):
            return deta.split(fB(keep & cm), shp)[0]
        full = da.integral(fn(np.ones(pool.n, bool))[:, :2])
        reps = np.array([da.integral(fn(blk != b)[:, :2]) for b in np.unique(blk)])
        g = len(reps)
        err = np.sqrt((g - 1) / g * ((reps - reps.mean(0)) ** 2).sum(0))
        nl = np.array([b_int(X, s.w * cm) for X in nullX])
        mu, sd = nl.mean(0), nl.std(0, ddof=1)
        neff = s.n_eff(s.classes["gap"][0].sum(1) * cm)
        for p, pt in enumerate(("1-2 GeV", "2-4 GeV")):
            lines.append(f"  {name:14s} {pt} (N_eff {neff:.0f}): " + "; ".join(
                f"{GAPS[i + 1]}: {full[i, p]:+7.2f} ± {err[i, p]:5.2f} (null {mu[i, p]:+5.2f} ± "
                f"{sd[i, p]:4.2f}, z {(full[i, p] - mu[i, p]) / sd[i, p]:+.1f})" for i in range(3)))
    lines.append("")

    # where B comes from: the classes' centrality composition and the bulk's eta shape
    lines.append("Centrality composition per class (weighted <N_coll> of the events, by their class "
                 "fractions) and the bulk's lab dN/deta shape against centrality:")
    wf = s.w[:, None] * f
    mean_nc = (wf * ncoll[:, None]).sum(0) / wf.sum(0)
    lines.append("  <N_coll> per class: " + ", ".join(f"{g}: {m:.0f}" for g, m in zip(GAPS, mean_nc)))
    nc_bg = np.zeros(len(ub))
    nc_bg[inv] = ncoll
    terc = np.quantile(nc_bg, [1 / 3, 2 / 3])
    for p_, pt in enumerate(("1-2 GeV", "2-4 GeV")):
        cells = []
        for name, m in (("low N_coll third", nc_bg < terc[0]), ("high N_coll third", nc_bg >= terc[1])):
            rr = rho[m, p_].mean(0)
            r0 = rr[np.abs(da.ETAC) < 0.5].mean()
            cells.append(f"{name}: " + ", ".join(
                f"|eta| {lo}-{hi} {rr[(np.abs(da.ETAC) >= lo) & (np.abs(da.ETAC) < hi)].mean() / r0:.4f}"
                for lo, hi in ((1.0, 1.5), (1.5, 2.0), (2.0, 2.4))))
        lines.append(f"  rho(eta) / rho(|eta| < 0.5), {pt}: " + "; ".join(cells))

    # B with a centrality-matched mixed event (the pooled density of the event's own N_coll
    # class, as CMS mixes within 0.5 % of centrality) and against a null deranged within them
    q_bg = da.quantile_classes(nc_bg, a.mix_classes)                  # class of each background
    s.K["matched"] = deta.mixed_event_of(pool, s.y, a.mix_classes)
    lines.append(f"B and F with a centrality-matched mixed event (the pooled lab density of the event's "
                 f"own N_coll class, {a.mix_classes} classes, as CMS mixes within 0.5 % of "
                 "centrality) against one for all; integral, jackknife:")
    for cont in ("B", "F"):
        for acc in ("pooled", "matched"):
            fn_, shp_ = s.observable(cont, acceptance=acc)
            full = da.integral(deta.split(fn_(np.ones(pool.n, bool)), shp_)[0][:, :2])
            reps = np.array([da.integral(deta.split(fn_(blk != b), shp_)[0][:, :2])
                             for b in np.unique(blk)])
            g = len(reps)
            err = np.sqrt((g - 1) / g * ((reps - reps.mean(0)) ** 2).sum(0))
            for p_, pt in enumerate(("1-2 GeV", "2-4 GeV")):
                lines.append(f"  {cont} {acc:8s} {pt}: " + "; ".join(
                    f"{GAPS[i + 1]}: {full[i, p_]:+7.2f} ± {err[i, p_]:5.2f}" for i in range(3)))

    def strat_derangement(rng):
        p = np.arange(len(ub))
        for q in range(q_bg.max() + 1):
            idx = np.flatnonzero(q_bg == q)
            p[idx] = idx[derangement(len(idx), rng)]
        return p
    rng2 = np.random.default_rng(a.seed + 1)
    sn = np.array([b_int(bulk(strat_derangement(rng2)), s.w) for _ in range(min(a.perms, 60))])
    real = b_int(np.asarray(s.Y["B"], float), s.w)
    lines.append(f"B (one mixed event for all) against a null deranged only within the {a.mix_classes} "
                 "N_coll classes (keeps each Z's centrality): real, null mean ± sd, z:")
    for p_, pt in enumerate(("1-2 GeV", "2-4 GeV")):
        lines.append(f"  {pt}: " + "; ".join(
            f"{GAPS[i + 1]}: {real[i, p_]:+7.2f} (null {sn[:, i, p_].mean():+6.2f} ± "
            f"{sn[:, i, p_].std(ddof=1):5.2f}, z {(real[i, p_] - sn[:, i, p_].mean()) / sn[:, i, p_].std(ddof=1):+.1f})"
            for i in range(3)))
    text = "\n".join(lines) + "\n"
    with open(a.out, "w") as fo:
        fo.write(text)
    np.savez_compressed(a.out + ".npz", **npz)
    print(text, end="")


if __name__ == "__main__":
    main()
