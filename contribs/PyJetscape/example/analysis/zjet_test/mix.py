#!/usr/bin/env python3
"""Event mixing (../zjet_mixing.py) against CMS 2507.09307: the pools of a Pb+Pb and a vacuum
directory (the map step, one job per production file, cached), then the mixing modes ->
figures/mixing_dphi.png, mixing_dy.png, mixing_errors.png and mixing_summary.txt.

  python mix.py                                     # the test: data/pbpb, data/vacuum
  python mix.py --pbpb <dir> --vacuum <dir> --pools <dir> --jobs 8 --tag <name>  # a campaign
  python mix.py --pbpb-pools <dir> --vacuum-pools <dir> --tag _cxx   # the C++ map step's pools
                                    # (example/analysis_root/zjet/zhadron_pools, *_pool.root)

Pb+Pb, CMS 2507.09307's selection (zjet_wake.CMS_2507), every event weighted w (window cross
section / events x N_coll):
  factorized  the Z-correlated yield (paired wake + fragments), compare.py's pair rules
  all         full events, every selected event a partner (the pooled density)
  cms10       full events, 10 partners from 10 other backgrounds, one oversample each
  strict      one Z per background (its own fluid), one oversample, 10 partners from 10 other
              backgrounds of that set; the error from the spread over the Z's
pp: the factorized result (no fluid: every event its own background).

Checks first: (1) factorized = compare.py (data/hists.npz) to rounding, Pb+Pb and pp; (2)
the own-background rule with the bulk off, within the errors of (1); (3) the closure: the
bulk term (all - all with the bulk off) against 0; (4) the exclusion bias of leaving the own
background out (zjet_mixing.exclusion_bias), predicted and measured; (5) the strict spread
error against the scatter of disjoint subsamples (``--strict-split``); (6) the noise budget
per Z from all Z's (zjet_mixing.per_z_noise, strict_noise): the strict error predicted with
and without each event's flat multiplicity level and Poisson only (``--efficiency`` scales the
Poisson part as efficiency-corrected tracks do).  Then the modes against CMS, and the strict
errors scaled to CMS's Z count (``--n-cms``) against CMS's statistical errors.  Errors:
jackknife over blocks of backgrounds (``--blocks``); strict: the spread."""
import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                        # zjet_wake, zjet_mixing
sys.path.insert(0, HERE)                                         # compare
try:
    import jetscape  # noqa: F401
except ImportError:                                              # the source tree
    sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "python"))
import zjet_mixing as zm                                         # noqa: E402
import zjet_wake as zw                                           # noqa: E402

POOL_VERSION = 3
MODES = ("factorized", "all", "cms10", "strict")
COLORS = {"factorized": "C3", "all": "C0", "cms10": "C2", "strict": "C1"}


# ── the map step ─────────────────────────────────────────────────────────────────
def _pool_job(stem, local, glob_, w, bg, seed, path):
    t0 = time.time()
    d = zm.file_pool(stem, local, glob_, w, bg, seed=seed)
    np.savez_compressed(path, **d)
    return os.path.basename(stem), len(d["event"]), time.time() - t0


def build_pools(src, outdir, jobs, seed, rebuild):
    """The pools of every production file of ``src`` in ``outdir`` (rebuilt when the
    files, the selection or the seed changed) -> their paths."""
    from jetscape.hadrons_h5 import HadronFileReader

    r = HadronFileReader(src)
    t = zw.event_table(r)
    sel = zw.select(t, **zw.CMS_2507)
    w = zw.event_weights(t, sel)
    loc = np.array([r.locate(g) for g in range(r.n_events)], np.int64).reshape(-1, 2)
    bulk = "bulk_jet" in r.tags()
    # the hydro background of every event; vacuum: every event its own
    unit = (np.array([r.event_info(g).bg_unit for g in range(r.n_events)], np.int64) if bulk
            else loc[:, 1])
    _, bgid = np.unique(np.stack([loc[:, 0], unit], axis=1), axis=0, return_inverse=True)
    bgid = bgid.ravel()
    stems = r.stems
    manifest = {"version": POOL_VERSION, "source": os.path.abspath(src), "seed": seed,
                "selection": "CMS_2507", "n_events": r.n_events, "n_selected": int(sel.sum()),
                "files": [[os.path.basename(s), os.path.getmtime(f"{s}_hadrons_jet_frag.h5")]
                          for s in stems]}
    os.makedirs(outdir, exist_ok=True)
    paths = [os.path.join(outdir, os.path.basename(s) + ".npz") for s in stems]
    mpath = os.path.join(outdir, "manifest.json")
    if not rebuild and os.path.exists(mpath) and all(map(os.path.exists, paths)):
        with open(mpath) as f:
            if json.load(f) == json.loads(json.dumps(manifest)):
                print(f"pools of {src}: cached in {outdir}", flush=True)
                return paths
    t0 = time.time()
    print(f"pools of {src}: {len(stems)} files, {int(sel.sum())} selected events, "
          f"{jobs} jobs", flush=True)
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = []
        for i, (stem, path) in enumerate(zip(stems, paths)):
            g = np.flatnonzero(sel & (loc[:, 0] == i))
            futs.append(ex.submit(_pool_job, stem, loc[g, 1], g, w[g], bgid[g], seed, path))
        for f in futs:
            name, n, dt = f.result()
            print(f"  {name}: {n} events, {dt:.0f} s", flush=True)
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"  {time.time() - t0:.0f} s", flush=True)
    return paths


# ── the modes ───────────────────────────────────────────────────────────────────
def rebin(h, k):
    """compare.rebin_y on values: |Delta y| < 4 in bins of 0.4 k."""
    from compare import rebin_y

    return rebin_y(h, np.zeros_like(h), k)[0]


def from_all(pool, content, rules, phi_flat, blocks, k):
    """S - B with all partners -> dict phi, y (value, jackknife error), y04 (value), and
    the replicates; ``rules`` (Delta phi, Delta y)."""
    out = {}
    for var, rule in zip(("phi", "y"), rules):
        D, reps = zm.all_partners(pool, content, rule, phi_flat, blocks)
        j = 0 if var == "phi" else 1
        v = zm.observables(D)[j]
        rv = np.array([zm.observables(d)[j] for d in reps])
        if var == "y":
            out["y04"], out["y04_reps"] = v, rv
            v, rv = rebin(v, k), rebin(rv, k)
        out[var] = (v, zm.jackknife(rv) if len(rv) > 1 else np.zeros_like(v))
        out[f"{var}_reps"] = rv
    return out


def from_explicit(pool, res, w, groups, blocks, k, error):
    """Per-Z explicit-mixing observables -> dict phi, y (value, error); ``error`` 'jk' (the
    jackknife over the blocks) or 'spread' (over the Z's), the other one in ``*_alt``."""
    out = {}
    for var, x in (("phi", res["dphi"]), ("y", rebin(res["dy"], k))):
        m, s, jk = zm.weighted_mean(x, w, groups, blocks if len(blocks) > 1 else ())
        jk = np.zeros_like(m) if jk is None else jk
        out[var], out[f"{var}_alt"] = ((m, jk), s) if error == "jk" else ((m, s), jk)
    return out


def strict_pulls(pool, zi, W, n_split, n_mix, seed, k):
    """The strict emulation in ``n_split`` disjoint subsamples of its Z's (partners inside
    each) -> {var: (n_PT,) pull width}: the scatter of the subsamples' results over their
    spread errors, the median over the bins of sqrt(sum_g (m_g - <m>)^2 / s_g^2 / (G - 1)).
    1 if the spread error is right (it ignores that partners are shared between Z's)."""
    rng = np.random.default_rng([int(seed), 0x5917])
    groups = np.array_split(rng.permutation(len(zi)), n_split)
    vals, errs = {"phi": [], "y": []}, {"phi": [], "y": []}
    for g in groups:
        res = zm.explicit_partners(pool, zi[g], zi[g], W[g], n_mix=n_mix, s="one", seed=seed)
        for var, x in (("phi", res["dphi"]), ("y", rebin(res["dy"], k))):
            m, s, _ = zm.weighted_mean(x, W[g])
            vals[var].append(m)
            errs[var].append(s)
    out = {}
    for var in vals:
        m, s = np.array(vals[var]), np.array(errs[var])
        pull2 = np.sum((m - m.mean(axis=0)) ** 2 / s ** 2, axis=0) / (len(groups) - 1)
        out[var] = np.median(np.sqrt(pull2), axis=-1)
    return out


def chi2(m, e, y, t):
    """compare.py's chi^2: over the independent half of the (reflected) bins."""
    return 0.5 * float(np.sum((m - y) ** 2 / (e ** 2 + t ** 2))), len(m) // 2


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pbpb", default=os.path.join(HERE, "data", "pbpb"))
    ap.add_argument("--vacuum", default=os.path.join(HERE, "data", "vacuum"))
    ap.add_argument("--pools", default=os.path.join(HERE, "data", "mix"),
                    help="pool cache (default data/mix/{pbpb,vacuum})")
    ap.add_argument("--pbpb-pools", default=None,
                    help="read these pools (*_pool.root of zhadron_pools, or *.npz) instead of "
                         "building them from --pbpb")
    ap.add_argument("--vacuum-pools", default=None, help="the same for --vacuum")
    ap.add_argument("--hists", default=os.path.join(HERE, "data", "hists.npz"),
                    help="accumulate.py's output, for the check against compare.py ('' none)")
    ap.add_argument("--hepdata", default=os.path.join(HERE, "data", "hepdata",
                                                      "HEPData-ins2945894-v1-yaml"))
    ap.add_argument("--outdir", default=os.path.join(HERE, "figures"))
    ap.add_argument("--tag", default="", help="suffix of the output names")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--rebuild", action="store_true", help="remake the pools")
    ap.add_argument("--n-mix", type=int, default=10, help="partners per Z (CMS: 10)")
    ap.add_argument("--blocks", type=int, default=100, help="jackknife blocks of backgrounds")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-cms", type=float, default=1550.0,
                    help="CMS's Z count in 0-30 %% (plan: ~1,550, 78 %% of ~2,000)")
    ap.add_argument("--strict-split", type=int, default=10,
                    help="disjoint subsamples for the check of the strict error")
    ap.add_argument("--efficiency", type=float, default=1.0,
                    help="tracking efficiency for the Poisson part of the predicted strict "
                         "errors (CMS's tracks are efficiency-corrected; 1: perfect)")
    a = ap.parse_args()
    from compare import PT_LABELS, PT_TAGS, Model, cms

    t0 = time.time()
    def pool_files(given, src, sub):
        if given:
            paths = sorted(glob.glob(os.path.join(given, "*_pool.root"))
                           or glob.glob(os.path.join(given, "*.npz")))
            if not paths:
                raise FileNotFoundError(f"no *_pool.root or *.npz in {given}")
            print(f"pools: {len(paths)} files from {given}", flush=True)
            return paths
        return build_pools(src, os.path.join(a.pools, sub), a.jobs, a.seed, a.rebuild)

    pb = zm.Pool.load(pool_files(a.pbpb_pools, a.pbpb, "pbpb"))
    pp = zm.Pool.load(pool_files(a.vacuum_pools, a.vacuum, "vacuum"))
    blocks, pblocks = pb.blocks(a.blocks), pp.blocks(a.blocks)
    R = {}
    R["factorized"] = from_all(pb, "zc", ("all", "self"), True, blocks, 2)
    R["all"] = from_all(pb, "full", ("all", "all"), False, blocks, 2)
    fact_all = from_all(pb, "zc", ("all", "all"), False, blocks, 2)          # (3)
    fact_bg = from_all(pb, "zc", ("background", "background"), False, blocks, 2)  # (2)
    full_bg = from_all(pb, "full", ("background", "background"), False, blocks, 2)  # (4)
    bias = zm.exclusion_bias(pb, "full")
    cms10 = zm.explicit_partners(pb, np.arange(pb.n), np.arange(pb.n), pb.w, n_mix=a.n_mix,
                                 s="full", seed=a.seed)
    R["cms10"] = from_explicit(pb, cms10, pb.w, pb.zb, blocks, 2, "jk")
    zi, W = zm.strict_set(pb, seed=a.seed)
    strict = zm.explicit_partners(pb, zi, zi, W, n_mix=a.n_mix, s="one", seed=a.seed)
    R["strict"] = from_explicit(pb, strict, W, pb.zb[zi], blocks, 2, "spread")
    P = from_all(pp, "full", ("all", "self"), True, pblocks, 1)
    P["y08"] = (rebin(P["y04"], 2), zm.jackknife(rebin(P["y04_reps"], 2)))
    # (5) the strict spread error checked against the scatter of disjoint subsamples
    pulls = {}
    need = a.strict_split * (a.n_mix + 1)
    pzi, pW = zm.strict_set(pp, seed=a.seed)
    for name, pool_, zi_, W_, k in (("Pb+Pb", pb, zi, W, 2), ("pp", pp, pzi, pW, 1)):
        if len(zi_) >= 2 * need:
            pulls[name] = strict_pulls(pool_, zi_, W_, a.strict_split, a.n_mix, a.seed, k)
    # (6) the noise budget per Z, from all Z's: the strict error predicted
    nz = zm.per_z_noise(pb)
    budget, sd = {}, {}
    for var, j, k in (("phi", 0, None), ("y", 1, 2)):
        sd[var] = {}
        for piece in zm.NOISE_PIECES:
            x = nz[piece][j] if k is None else rebin(nz[piece][j], k)
            m = np.tensordot(pb.w, x, axes=1) / pb.w.sum()
            sd[var][piece] = np.sqrt(np.tensordot(pb.w, (x - m) ** 2, axes=1) / pb.w.sum())
        n_s = 1.0 / np.average(1.0 / pb.n_samples, weights=pb.w)
        budget[var] = zm.strict_noise(sd[var], n_s, a.n_mix, a.efficiency)
    print(f"mixing: {time.time() - t0:.0f} s", flush=True)

    neff_pb, neff_strict = zm.n_eff(pb.w), zm.n_eff(W)
    lines = [f"Pb+Pb: {pb.n} selected Z's (effective {neff_pb:.0f}) on {pb.n_bg} hydro "
             f"backgrounds, jackknife over {len(blocks)} blocks; strict: {len(zi)} Z's "
             f"(effective {neff_strict:.0f}), partners per Z {strict['n_mix'].min()}-"
             f"{strict['n_mix'].max()}; cms10 partners {cms10['n_mix'].min()}-"
             f"{cms10['n_mix'].max()}", f"pp: {pp.n} selected Z's", ""]

    # (1) factorized = compare.py
    if a.hists and os.path.exists(a.hists):
        d = np.load(a.hists)
        for name, g in (("DETA", zm.DETA), ("DPHI", zm.DPHI), ("ETA", zm.ETA)):
            if not np.allclose(d[name], g):
                raise ValueError(f"{a.hists}: {name} is not zjet_mixing's grid")
        model = Model(d["DETA"], d["DPHI"], d["ETA"])
        gs = list(d["groups"])
        n = {g: int(d[f"g{g}_n"]) for g in gs}
        H = sum(n[g] * (d[f"g{g}_wake"] + d[f"g{g}_jet"]) for g in gs)
        T2 = sum(n[g] * (d[f"g{g}_wake_w2"] + d[f"g{g}_jet_w2"]) for g in gs)
        sel, w, yz = d["sel"], d["w"], d["y_z"]
        (p0, _), (y0, _) = model.observables(H, np.zeros_like(H), T2, yz[sel], w[sel])
        sv, wv = d["vac_sel"], d["vac_w"]
        nv = int(sv.sum())
        (q0, _), (z0, _) = model.observables(nv * d["vac"], np.zeros_like(d["vac"]),
                                             nv * d["vac_w2"], d["vac_y_z"][sv], wv[sv])
        lines.append("(1) factorized (bulk off, compare.py's pair rules) - compare.py, max |diff| "
                     "(relative to the largest |value|):")
        for lab, mine, ref in (("Pb+Pb dphi", R["factorized"]["phi"][0], p0),
                               ("Pb+Pb dy", R["factorized"]["y04"], y0),
                               ("pp dphi", P["phi"][0], q0), ("pp dy", P["y04"], z0)):
            diff = np.max(np.abs(mine - ref))
            lines.append(f"    {lab:11s} {diff:.2e} ({diff / np.max(np.abs(ref)):.1e})")
    else:
        lines.append("(1) no data/hists.npz: the check against compare.py skipped")
    # (2) own-background rule, bulk off; (3) the bulk term
    lines.append("(2) bulk off, own-background rule - factorized: max |diff| / error, "
                 "(3) closure: bulk term = all - all with the bulk off (every event a "
                 "partner), chi2/ndf against 0 (jackknife errors):")
    for var in ("phi", "y"):
        dv = R["all"][var][0] - fact_all[var][0]
        de = zm.jackknife(R["all"][f"{var}_reps"] - fact_all[f"{var}_reps"])
        for c in range(len(PT_TAGS)):
            r2 = np.max(np.abs(fact_bg[var][0][c] - R["factorized"][var][0][c])
                        / np.where(R["factorized"][var][1][c] > 0, R["factorized"][var][1][c],
                                   np.inf))
            c2, ndf = chi2(dv[c], de[c], 0.0, 0.0)
            lines.append(f"    d{var:3s} {PT_LABELS[c]:20s} (2) {r2:5.2f}   (3) {c2:6.1f}/{ndf}"
                         f"   bulk term at the first bin past 0: {dv[c][_first(var)]:+.3f} ± "
                         f"{de[c][_first(var)]:.3f}")
    # (4) what excluding the own background costs with weighted backgrounds
    meas = (full_bg["phi"][0] - R["all"]["phi"][0]).mean(axis=1)
    lines.append("(4) exclusion bias (the flat level by which leaving the Z's own background "
                 "out of its partners shifts S - B, per radian; zjet_mixing.exclusion_bias):")
    for c in range(len(PT_TAGS)):
        lines.append(f"    {PT_LABELS[c]:20s} pooled pairs {bias['pooled'][c]:+.4f} (measured "
                     f"{meas[c]:+.4f}), explicit draws (cms10, strict) "
                     f"{bias['explicit'][c]:+.4f}; error of cms10 at the first bin past 0 "
                     f"{R['cms10']['phi'][1][c][_first('phi')]:.4f}")
    lines.append(f"(5) strict spread error against the scatter of {a.strict_split} disjoint "
                 "subsamples (partners inside each): pull width, median over the bins "
                 "(1: the error is right):")
    if not pulls:
        lines.append(f"    too few Z's (each sample needs {2 * need}: one Z per background)")
    for name, pv in pulls.items():
        lines.append(f"    {name:6s} " + "   ".join(
            f"d{var} " + " ".join(f"{x:4.2f}" for x in pv[var]) for var in ("phi", "y"))
            + "   (p_T classes 1-2, 2-4, 4-10 GeV)")
    lines.append("")

    # the modes against CMS; the errors
    lines.append(f"{'':5s} {'':20s} {'mode':11s} {'Pb+Pb: model':>16s} {'CMS':>16s} "
                 f"{'chi2':>9s}   {'Pb+Pb - pp: model':>16s} {'CMS':>16s} {'chi2':>9s}")
    budget_lines = [
        f"(6) noise budget per Z (Pb+Pb, all {pb.n} Z's; per radian / unit Delta y at the first "
        f"bin past 0): the spreads over the Z's of the pooled mode, without each event's flat "
        f"level (shape), and of one oversample's Poisson noise; the fluid term; the strict error "
        f"they predict at CMS's {a.n_cms:.0f} Z's with {a.n_mix} partners, Poisson part / "
        f"efficiency {a.efficiency:g}: full / shape only / Poisson only; CMS's statistical error; "
        f"the median ratio predicted / CMS over the independent bins:",
        f"{'':5s} {'':20s} {'pooled':>7s} {'shape':>7s} {'Poisson':>7s} {'fluid':>7s} | "
        f"{'full':>6s} {'shape':>6s} {'Poiss.':>6s} {'CMS':>6s} | ratios full, shape, Poisson"]
    err_lines = [f"Errors at the first bin past 0, and the median over the independent bins of "
                 f"the strict error scaled to CMS's {a.n_cms:.0f} Z's / CMS's statistical "
                 f"error (strict: {neff_strict:.0f} effective Z's, x sqrt({neff_strict:.0f} / "
                 f"{a.n_cms:.0f}) = {np.sqrt(neff_strict / a.n_cms):.2f}):",
                 f"{'':5s} {'':20s} " + " ".join(f"{m:>10s}" for m in MODES)
                 + f" {'strict@CMS':>10s} {'CMS stat':>9s}   {'ratio':>6s}"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scale = np.sqrt(neff_strict / a.n_cms)
    for var in ("phi", "y"):
        fig, ax = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
        fe, ae = plt.subplots(1, 3, figsize=(16, 4.2))
        for c in range(len(PT_TAGS)):
            cx, yp, sp, yp_sy = cms(a.hepdata, var, "pbpb", PT_TAGS[c])
            _, yq, sq, yq_sy = cms(a.hepdata, var, "pp", PT_TAGS[c])
            tp, tq = np.hypot(sp, yp_sy), np.hypot(sq, yq_sy)
            if var == "y":                            # CMS pp: 0.4 in Delta y -> 0.8
                yq = yq.reshape(-1, 2).mean(1)
                tq = np.sqrt((tq.reshape(-1, 2) ** 2).sum(1)) / 2
            mq, eq = (P["phi"][0][c], P["phi"][1][c]) if var == "phi" else \
                (P["y08"][0][c], P["y08"][1][c])
            i0 = _first(var, len(cx))
            ax[0, c].errorbar(cx, yp, tp, fmt="o", color="k", ms=4,
                              label="CMS 2507.09307 (stat ⊕ syst)")
            ax[1, c].errorbar(cx, yp - yq, np.hypot(tp, tq), fmt="o", color="k", ms=4)
            for k, mode in enumerate(MODES):
                m, e = R[mode][var][0][c], R[mode][var][1][c]
                dx = (k - 1.5) * 0.03 * (cx[1] - cx[0]) / 0.314
                for row, (mm, ee) in enumerate(((m, e), (m - mq, np.hypot(e, eq)))):
                    ax[row, c].errorbar(cx + dx, mm, ee, fmt="s", ms=3.5, mfc="none",
                                        color=COLORS[mode], label=mode if row == 0 else None)
                c2a, ndf = chi2(m, e, yp, tp)
                c2b, _ = chi2(m - mq, np.hypot(e, eq), yp - yq, np.hypot(tp, tq))
                lines.append(f"d{var:4s} {PT_LABELS[c]:20s} {mode:11s} {m[i0]:+7.3f} ± "
                             f"{e[i0]:5.3f} {yp[i0]:+7.3f} ± {tp[i0]:5.3f} {c2a:6.1f}/{ndf}   "
                             f"{m[i0] - mq[i0]:+7.3f} ± {np.hypot(e, eq)[i0]:5.3f} "
                             f"{yp[i0] - yq[i0]:+7.3f} ± {np.hypot(tp, tq)[i0]:5.3f} "
                             f"{c2b:6.1f}/{ndf}")
                ae[c].plot(cx + dx, e, "s-" if mode != "strict" else "s:", ms=3.5,
                           color=COLORS[mode], mfc="none", lw=0.8,
                           label=mode + (" (jackknife)" if mode != "strict" else " (spread)"))
            es = R["strict"][var][1][c] * scale
            ae[c].plot(cx, es, "D-", ms=4, color="C1", label=f"strict at CMS's {a.n_cms:.0f} Z's")
            ae[c].plot(cx, sp, "o-", ms=4, color="k", label="CMS stat")
            half = cx > 0 if var == "y" else (cx > 0) & (cx < np.pi)
            ratio = np.median(es[half] / sp[half])
            pred = {v: budget[var][v][c] / np.sqrt(a.n_cms) for v in ("full", "shape", "poisson")}
            for v, ls in (("full", "--"), ("shape", "-."), ("poisson", ":")):
                ae[c].plot(cx, pred[v], ls, color="C4", lw=1.2,
                           label=f"predicted at CMS's {a.n_cms:.0f} Z's: {v}")
            pieces = (sd[var]["pooled"], sd[var]["shape"], sd[var]["poisson"],
                      budget[var]["fluid"])
            variants = ("full", "shape", "poisson")
            budget_lines.append(
                f"d{var:4s} {PT_LABELS[c]:20s} "
                + " ".join(f"{x[c][i0]:7.3f}" for x in pieces)
                + " | " + " ".join(f"{pred[v][i0]:6.3f}" for v in variants)
                + f" {sp[i0]:6.3f} | "
                + ", ".join(f"{np.median(pred[v][half] / sp[half]):.2f}" for v in variants))
            err_lines.append(f"d{var:4s} {PT_LABELS[c]:20s} "
                             + " ".join(f"{R[mode][var][1][c][i0]:10.3f}" for mode in MODES)
                             + f" {es[i0]:10.3f} {sp[i0]:9.3f}   {ratio:6.2f}")
            for row, lab in enumerate(("Pb+Pb 0–30 %", "Pb+Pb − pp")):
                ax[row, c].axhline(0, color="0.6", lw=0.6)
                ax[row, c].set_ylabel(f"{lab}: d⟨ΔN_ch⟩/dΔ{'φ' if var == 'phi' else 'y'}")
            ax[0, c].set_title(PT_LABELS[c] + ("" if var == "phi" else ",  |Δφ| < π/2"))
            ax[1, c].set_xlabel("Δφ_ch,Z" if var == "phi" else "Δy_ch,Z")
            ae[c].set_title(PT_LABELS[c])
            ae[c].set_xlabel("Δφ_ch,Z" if var == "phi" else "Δy_ch,Z")
            ae[c].set_ylabel(f"error of d⟨ΔN_ch⟩/dΔ{'φ' if var == 'phi' else 'y'} (Pb+Pb)")
            ae[c].set_yscale("log")
        ax[0, 0].legend(fontsize=8)
        ae[0].legend(fontsize=7)
        fig.suptitle(f"Z–hadron S − B by event mixing, Pb+Pb 0–30 % ({pb.n} Z's, {pb.n_bg} "
                     f"backgrounds) and pp", y=0.995)
        fig.tight_layout()
        fe.suptitle("Statistical errors of the mixing modes, Pb+Pb 0–30 %", y=0.995)
        fe.tight_layout()
        os.makedirs(a.outdir, exist_ok=True)
        for f_, name in ((fig, f"mixing_d{var}"), (fe, f"mixing_errors_d{var}")):
            path = os.path.join(a.outdir, f"{name}{a.tag}.png")
            f_.savefig(path, dpi=110)
            plt.close(f_)
            print(f"-> {path}")
    # step 1, the reference: the factorized result alone, with the pp row (compare.py's figure)
    ref_lines = ["Step 1, the reference (factorized: the exact Z-correlated yield, possible only "
                 "in the model; figures mixing_reference_d*): model and CMS at the first bin "
                 "past 0, chi2/ndf with the model's errors + CMS stat + syst:"]
    for var in ("phi", "y"):
        fig, ax = plt.subplots(3, 3, figsize=(16, 11), sharex=True)
        for c in range(len(PT_TAGS)):
            cx, yp, sp, yp_sy = cms(a.hepdata, var, "pbpb", PT_TAGS[c])
            cq, yq, sq, yq_sy = cms(a.hepdata, var, "pp", PT_TAGS[c])
            tp, tq = np.hypot(sp, yp_sy), np.hypot(sq, yq_sy)
            if var == "y":                            # pp 0.4 -> 0.8 for the difference
                yq2 = yq.reshape(-1, 2).mean(1)
                tq2 = np.sqrt((tq.reshape(-1, 2) ** 2).sum(1)) / 2
                mq2, eq2 = P["y08"][0][c], P["y08"][1][c]
            else:
                yq2, tq2 = yq, tq
                mq2, eq2 = P["phi"][0][c], P["phi"][1][c]
            mq, eq = P[var][0][c], P[var][1][c]
            m, e = R["factorized"][var][0][c], R["factorized"][var][1][c]
            for row, (mm, ee, xx, yy, tt, lab) in enumerate((
                    (mq, eq, cq, yq, tq, "pp"), (m, e, cx, yp, tp, "Pb+Pb 0–30 %"),
                    (m - mq2, np.hypot(e, eq2), cx, yp - yq2, np.hypot(tp, tq2), "Pb+Pb − pp"))):
                axr = ax[row, c]
                axr.errorbar(xx, yy, tt, fmt="o", color="k", ms=4,
                             label="CMS 2507.09307 (stat ⊕ syst)")
                axr.errorbar(xx, mm, ee, fmt="s", color="C3", ms=4, mfc="none",
                             label="model, factorized")
                axr.axhline(0, color="0.6", lw=0.6)
                axr.set_ylabel(f"{lab}: d⟨ΔN_ch⟩/dΔ{'φ' if var == 'phi' else 'y'}")
                c2, ndf = chi2(mm, ee, yy, tt)
                i0 = _first(var, len(xx))
                ref_lines.append(f"d{var:4s} {PT_LABELS[c]:20s} {lab:13s} {mm[i0]:+7.3f} ± "
                                 f"{ee[i0]:5.3f}  CMS {yy[i0]:+7.3f} ± {tt[i0]:5.3f}  "
                                 f"{c2:6.1f}/{ndf}")
            ax[0, c].set_title(PT_LABELS[c] + ("" if var == "phi" else ",  |Δφ| < π/2"))
            ax[2, c].set_xlabel("Δφ_ch,Z" if var == "phi" else "Δy_ch,Z")
        ax[0, 0].legend(fontsize=8)
        fig.suptitle(f"Step 1, the reference: Z–hadron S − B, factorized (paired wake + "
                     f"fragments), Pb+Pb 0–30 % ({pb.n} Z's) and pp ({pp.n} Z's)", y=0.995)
        fig.tight_layout()
        path = os.path.join(a.outdir, f"mixing_reference_d{var}{a.tag}.png")
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"-> {path}")
    lines = lines[:3] + ref_lines + [""] + lines[3:]
    lines += [""] + err_lines + [""] + budget_lines + [
        "", "strict@CMS: the strict emulation's spread error x sqrt(N_eff / N_CMS). CMS's "
        "statistical errors are of efficiency-corrected tracks (larger than at perfect "
        "efficiency where Poisson noise dominates); N_CMS is the plan's estimate."]
    text = "\n".join(lines) + "\n"
    with open(os.path.join(a.outdir, f"mixing_summary{a.tag}.txt"), "w") as f:
        f.write(text)
    print(text, end="")


def _first(var, n=None):
    """The first bin past 0: Delta phi pi/20 (index 5 of 20), Delta y the first > 0."""
    if var == "phi":
        return 5
    return (n or 10) // 2


if __name__ == "__main__":
    main()
