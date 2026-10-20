#!/usr/bin/env python3
"""CMS HIN-25-012's eta-separated dijet-hadron analysis (arXiv:2602.19431) on the model, with
the Z as jet1 (``--trigger z``, the Z+jet campaigns): the final stage, from the Z-hadron pools
(zjet/mix.py's or zhadron_pools', version 4: the per-event contents Y_*, L_*) and the jet map
step's outputs (zjet_jets_map.py / zjets at --R 0.4), Pb+Pb and pp
(docs/Plans/PLAN_dijet_deta_wake.md, Phase B; the step by step: ANA_Deta.md).

    python deta.py --pools runs/zjet_c00/data/mix/pbpb --vacuum-pools <pp pools> \\
                   --jets runs/zjet_c00/data/jets_r04 --pp-jets <pp jets> --run zjet_c00 \\
                   [--zpt 60] [--jet2 30] [--no-xcut] [--jet1 bare|zcone] \\
                   [--jet2-pick leading|b2b] [--central] [--blocks 100] [--tag _c00]

Several directories per option: campaigns mapped and pooled apart, combined here.

Selection (per oversample of every pooled Z event, the decisions of the plan's section 6):
the Z of the pools (CMS 2507.09307: 60-120 GeV, both leptons > 20 GeV in |eta| < 2.4),
|y_Z| < --ymax (1); jet1's p_T = p_T^Z (--jet1 bare) or p_T^Z + the Z cone's delta p_T
(--jet1 zcone) above --zpt; jet2 = the leading map-step jet (|eta| < 2; Pb+Pb rho A-subtracted)
above --jet2, Delta phi(Z, jet2) > 5 pi / 6 (--jet2-pick b2b: the leading among those
back to back), and jet2 below jet1 (x_jZ < 1, CMS's leading-jet ordering; --no-xcut drops
it); --central: N_coll > --ncoll-min (1210, the model's 0-10 %).  Classes: CMS's gaps
|y_Z - eta_jet2| < 0.5 (R_sym), 0.5-1, 1-1.5, 1.5-2 (R_asym); per event the fraction of its
oversamples in each, reflected (Delta y -> -Delta y) where jet2 is above the Z; the strict
mode (O) uses the event's one oversample k_one.  Yang-Wang's form (2508.04194): the lab-eta
distribution on the Z side for |eta_jet2| < 0.8 against 0.8-1.6 and for jet2 in the Z's
hemisphere or the other, reflected to eta_jet2 > 0.

The contents (Pb+Pb; pp: F and O):  T  wake + fragments (Y_zc), the truth;  W  the wake
(Y_zc - Y_frag);  F  the full event, 40 samples (Y_full);  O  one oversample (Y_one);  N  the
no-wake twin, bulk_bg + the same fragments (Y_full - Y_zc + Y_frag);  B  the bulk alone
(Y_full - Y_zc).  The observable (CMS): near side per trigger per unit Delta y, divided by the
mixed event (the triggers' rapidities folded with the pooled lab density of the sample's full
events; T and W with the geometric acceptance), normalized to 1 where the acceptance is full;
the 1.5-2.5 level (its bins weighted by the acceptance) subtracted per class; R_asym - R_sym; Pb+Pb - pp; the integral over
-2.5 < Delta y < -0.5 (CMS's 10 bins centred -2.4 ... -0.6).  Errors: jackknife over blocks
of hydro backgrounds (pp: of events).

Closure (B4): (1) the classes summed give the Z-side Delta y of S_full; (2) N and (3) B flat;
(4) F = T; (5) the sideband's share of T; (6) pp flat.  The noise per event (R4): the errors
scaled to one effective Z.

CMS (cms25012.py, HEPData ins3123406 in --hepdata; ../zjet/get_hepdata.sh 25012): the model in
CMS's 0.2 bins and convention (--cms-norm per-bin: CMS's tables as yields per bin of 0.2, the
reading its Table 4 implies; per-unit: per unit Delta eta) against Tables 1-3 (pp, PbPb 0-30 %
and PbPb - pp) and Table 4 (the integral), chi^2 with both errors.  With --trigger z this is
the Z with p_T^Z > --zpt against CMS's dijets with jet1 > 130 GeV: not like for like, a shape
reference and the comparison in place for the dijet productions.
-> runs/<run>/figures/deta_{asym,closure,yw,integrated,cms}<tag>.png (+ .npz/.root),
deta_summary<tag>.txt."""
import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
sys.path.insert(0, ANALYSIS)                                      # jetcorr
sys.path.insert(0, os.path.join(ANALYSIS, "zjet"))                # zjet_mixing, zjet_jets
try:
    import jetscape  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.join(ANALYSIS, "..", "..", "python"))
from jetcorr import deta_asym as da  # noqa: E402
from jetcorr import figdata, mapio, runs  # noqa: E402

import cms25012  # noqa: E402

PT_LABELS = ("1-2 GeV", "2-4 GeV", "4-10 GeV")
NCOLL_10 = 1210.0          # the model's 0-10 %: 200k min-bias 3dMCGlauber events (zjet/test)
CONTENTS_PBPB = ("T", "W", "F", "O", "N", "B")
GEOMETRIC = ("T", "W")     # the truth needs no mixed event of the bulk


def pool_paths(dirs):
    out = []
    for d in dirs:
        found = sorted(glob.glob(os.path.join(d, "*_pool.root")) or
                       glob.glob(os.path.join(d, "*.npz")))
        if not found:
            raise FileNotFoundError(f"no pools in {d}")
        out += found
    return out


def map_names(dirs):
    """The production files of a set of map outputs (their file records), sorted by name."""
    import uproot
    from zjet_jets import map_files
    names = []
    for p in map_files(dirs):
        with uproot.open(p) as f:
            rec = mapio.read_file_info(f)
        if rec is None:
            raise ValueError(f"{p}: no file_info (map step before 2026-10-19): rerun it")
        names.append(rec["name"])
    return sorted(names)


def leading_jet2(J, rows, phi_z_row, pick):
    """Per row: jet2's (pt, eta, phi), NaN without: the leading jet (pick 'leading') or the
    leading among the jets with Delta phi(Z, jet) > 5 pi / 6 ('b2b')."""
    import awkward as ak
    pt, eta, phi = J["jet_pt"][rows], J["jet_eta"][rows], J["jet_phi"][rows]
    if pick == "b2b":
        d = np.abs(phi - np.asarray(phi_z_row, float)) % (2 * np.pi)
        keep = np.minimum(d, 2 * np.pi - d) > 5 * np.pi / 6
        pt, eta, phi = pt[keep], eta[keep], phi[keep]
    i = ak.argmax(pt, axis=1, keepdims=True)

    def first(x):
        return ak.to_numpy(ak.fill_none(ak.firsts(x[i]), np.nan)).astype(float)
    return first(pt), first(eta), first(phi)


def dphi(a, b):
    d = np.abs(np.asarray(a) - np.asarray(b)) % (2 * np.pi)
    return np.where(d > np.pi, 2 * np.pi - d, d)


class Sample:
    """One sample (Pb+Pb or pp): its pooled Z events, their weights, classes and contents."""

    def __init__(self, label, pool, E, J, a, vacuum, B=None):
        self.label, self.pool, self.vacuum = label, pool, vacuum
        idx = {int(g): i for i, g in enumerate(E["event"])}
        missing = [int(g) for g in pool.event if int(g) not in idx]
        if missing:
            raise ValueError(f"{label}: {len(missing)} pooled Z events have no jets (e.g. "
                             f"event {missing[0]}): pools and jets of other files?")
        ie = np.array([idx[int(g)] for g in pool.event])          # pool event -> E row
        self.z_pt = E["z_pt"][ie]
        self.y = np.asarray(pool.y, float)
        self.phi = np.asarray(pool.phi, float)
        self.n_coll = np.asarray(pool.n_coll, float)
        sel = np.abs(self.y) < a.ymax
        if a.central and not vacuum:
            sel &= self.n_coll > a.ncoll_min
        # the oversample rows of the selected events
        pos = {int(e_row): i for i, e_row in enumerate(ie)}
        rows = np.flatnonzero(np.isin(J["event"], E["event"][ie[sel]]))
        ev_of_row = np.array([pos[idx[int(g)]] for g in J["event"][rows]], np.int64)
        pt2, eta2, phi2 = leading_jet2(J, rows, self.phi[ev_of_row], a.jet2_pick)
        pt1 = self.z_pt[ev_of_row].astype(float)
        if a.jet1 == "zcone":
            pt1 = pt1 + np.nan_to_num(np.asarray(J["zcone_dpt"])[rows])
        ok = (pt1 > a.zpt) & (pt2 > a.jet2) & (dphi(phi2, self.phi[ev_of_row]) > 5 * np.pi / 6)
        if a.xcut:
            ok &= pt2 < pt1
        eta2 = np.where(ok, eta2, np.nan)
        n_rows = np.bincount(ev_of_row, minlength=pool.n).astype(float)
        # fake jet2: the same cuts on the jets of the Z's own background, the same oversample
        self.fake = None
        if B:
            bmap = {(int(b), int(k)): i for i, (b, k) in enumerate(zip(B["bg"], B["sample"]))}
            ib = np.array([bmap.get((int(g), int(k)), -1) for g, k in
                           zip(E["bg"][ie][ev_of_row], np.asarray(J["sample"])[rows])])
            has = ib >= 0
            fpt, feta, fphi = (np.full(len(rows), np.nan) for _ in range(3))
            if has.any():
                q = leading_jet2(B, ib[has], self.phi[ev_of_row][has], a.jet2_pick)
                fpt[has], feta[has], fphi[has] = q
            fok = (pt1 > a.zpt) & (fpt > a.jet2) & (dphi(fphi, self.phi[ev_of_row]) > 5 * np.pi / 6)
            if a.xcut:
                fok &= fpt < pt1
            fcls, frefl = da.gap_class(self.y[ev_of_row], np.where(fok, feta, np.nan))
            self.fake = da.event_classes(ev_of_row, pool.n, n_rows, fcls, frefl,
                                         len(da.GAPS))[0]
        self.sel = sel & (n_rows > 0)
        self.n_rows = n_rows
        k_one = np.asarray(pool.k_one)[ev_of_row]
        one = np.asarray(J["sample"])[rows] == k_one
        self.classes = {}
        for scheme, (cls, refl, nc) in {
                "gap": da.gap_class(self.y[ev_of_row], eta2) + (len(da.GAPS),),
                "yw": da.jet_eta_class(eta2) + (2,),
                "hemi": da.hemisphere_class(self.y[ev_of_row], eta2) + (2,)}.items():
            f, r = da.event_classes(ev_of_row, pool.n, n_rows, cls, refl, nc)
            f1, r1 = da.event_classes(ev_of_row[one], pool.n, np.ones(pool.n), cls[one],
                                      refl[one], nc)
            self.classes[scheme] = (f * self.sel[:, None], r, f1 * self.sel[:, None], r1)
        self.w = np.where(self.sel, pool.w, 0.0)
        self.groups = (np.asarray(pool.bg) if not vacuum else np.asarray(pool.event))
        P = pool
        self.Y = {"F": P.Y_full, "O": P.Y_one}
        self.L = {"F": P.L_full}
        if not vacuum:
            self.Y.update({"T": P.Y_zc, "W": P.Y_zc.astype(float) - P.Y_frag,
                           "N": P.Y_full.astype(float) - P.Y_zc + P.Y_frag,
                           "B": P.Y_full.astype(float) - P.Y_zc})
            self.L.update({"T": P.L_zc, "W": P.L_zc.astype(float) - P.L_frag,
                           "N": P.L_full.astype(float) - P.L_zc + P.L_frag,
                           "B": P.L_full.astype(float) - P.L_zc})
        # the mixed events: the pooled lab density of the full events, and the geometric one
        rho = pool.bg_R["full"].sum(axis=(0, 3))                   # (N_PT, N_ETA)
        self.K = {"pooled": da.mixed_event(self.y, rho),
                  "geometric": da.mixed_event(self.y, da.flat_density(rho.shape[0]))}
        self.norm_bins = da.full_acceptance(a.ymax)

    def n_eff(self, f):
        x = self.w * f
        return float(x.sum() ** 2 / np.sum(x ** 2)) if x.sum() > 0 else 0.0

    def observable(self, content, scheme="gap", strict=False, acceptance=None,
                   subtract=True):
        """-> fn(keep) giving (dif (n_cls - 1, N_PT, N_DY), yc (n_cls, N_PT, N_DY), level
        (n_cls, N_PT)) flattened into one vector for the jackknife, and the shapes."""
        f, r, f1, r1 = self.classes[scheme]
        if strict:
            f, r = f1, r1
        X = self.Y[content]
        acc = acceptance or ("geometric" if content in GEOMETRIC else "pooled")
        K = self.K[acc]

        def fn(keep):
            w = self.w * keep
            num, den = da.class_sums(X, w, f, r)
            me, _ = da.class_sums(K, w, f, r)
            yc, acc_ = da.corrected(num, den, me, self.norm_bins, with_acceptance=True)
            dif, level = da.asym_minus_sym(yc, subtract=subtract, acceptance=acc_)
            return np.concatenate([dif.ravel(), yc.ravel(), level.ravel()])
        n_cls = f.shape[1]
        shapes = ((n_cls - 1, X.shape[1], da.N_DY), (n_cls, X.shape[1], da.N_DY),
                  (n_cls, X.shape[1]))
        return fn, shapes

    def lab(self, content, scheme):
        """Yang-Wang: per class the per-trigger dN/deta on the Z side -> fn(keep)."""
        f, r, _, _ = self.classes[scheme]
        X = self.L[content]

        def fn(keep):
            num, den = da.class_sums(X, self.w * keep, f, r)
            return da.lab_yields(num, den).ravel()
        return fn, (f.shape[1], X.shape[1], len(da.ETAC))


def split(v, shapes):
    out, i = [], 0
    for s in shapes:
        n = int(np.prod(s))
        out.append(v[i:i + n].reshape(s))
        i += n
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trigger", default="z", choices=("z",),
                    help="jet1: the Z (Phase C adds jet1 of the dijet productions)")
    ap.add_argument("--pools", nargs="+", required=True, help="Pb+Pb pool directories")
    ap.add_argument("--vacuum-pools", nargs="+", required=True, help="pp pool directories")
    ap.add_argument("--jets", nargs="+", required=True, help="Pb+Pb map outputs (R = 0.4)")
    ap.add_argument("--pp-jets", nargs="+", required=True, help="pp map outputs")
    ap.add_argument("--zpt", type=float, default=60.0, help="jet1 (the Z) p_T threshold")
    ap.add_argument("--jet2", type=float, default=30.0, help="jet2 p_T threshold")
    ap.add_argument("--ymax", type=float, default=1.0, help="|y_Z| <")
    ap.add_argument("--no-xcut", dest="xcut", action="store_false",
                    help="drop jet2 < jet1 (x_jZ < 1)")
    ap.add_argument("--jet1", default="bare", choices=("bare", "zcone"))
    ap.add_argument("--jet2-pick", default="leading", choices=("leading", "b2b"))
    ap.add_argument("--central", action="store_true", help="only N_coll > --ncoll-min")
    ap.add_argument("--ncoll-min", type=float, default=NCOLL_10)
    ap.add_argument("--blocks", type=int, default=100)
    ap.add_argument("--hepdata", default=runs.hepdata("HEPData-ins3123406-v1-yaml"),
                    help="CMS HIN-25-012's HEPData record (../zjet/get_hepdata.sh 25012)")
    ap.add_argument("--cms-norm", default="per-bin", choices=tuple(cms25012.NORMS),
                    help="CMS's tables as yields per 0.2 bin (Table 4's reading) or per unit "
                         "Delta eta")
    ap.add_argument("--run", default="zjet_test")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    a.outdir = a.outdir or runs.figures(a.run)
    os.makedirs(a.outdir, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import zjet_jets as zj
    import zjet_mixing as zm

    lines = []
    R = {k: zj.map_run_info(v)["R"] for k, v in (("Pb+Pb", a.jets), ("pp", a.pp_jets))}
    pbp, ppp = zm.Pool.load(pool_paths(a.pools)), zm.Pool.load(pool_paths(a.vacuum_pools))
    for lab_, p, d in (("Pb+Pb", pbp, a.jets), ("pp", ppp, a.pp_jets)):
        if not hasattr(p, "Y_full"):
            raise SystemExit(f"{lab_} pools without the dijet contents: rebuild them (version 4)")
        if sorted(p.file_names.tolist()) != map_names(d):
            raise SystemExit(f"{lab_}: the pools and the jets are of different production files")
    Eb, Jb, Bb = zj.load_jets(a.jets)
    Ep, Jp, _ = zj.load_jets(a.pp_jets)
    pb = Sample("Pb+Pb", pbp, Eb, Jb, a, False, Bb)
    pp = Sample("pp", ppp, Ep, Jp, a, True)
    cuts = (f"jet1 {a.jet1} > {a.zpt:g}, jet2 ({a.jet2_pick}) > {a.jet2:g} GeV, |y_Z| < "
            f"{a.ymax:g}, {'jet2 < jet1' if a.xcut else 'no x_jZ cut'}"
            + (f", N_coll > {a.ncoll_min:g}" if a.central else ""))
    lines.append(f"CMS HIN-25-012 with the Z as jet1: {cuts}; jets R = {R['Pb+Pb']:g} "
                 f"(pp {R['pp']:g})")
    if abs(R["Pb+Pb"] - 0.4) > 1e-9:
        lines.append(f"  note: CMS uses R = 0.4, these jets are R = {R['Pb+Pb']:g}")
    blk = {"Pb+Pb": da.blocks_of(pb.groups, a.blocks), "pp": da.blocks_of(pp.groups, a.blocks)}
    for s in (pb, pp):
        f = s.classes["gap"][0]
        lines.append(f"{s.label}: {int(s.sel.sum())} Z's with |y_Z| < {a.ymax:g} (of {s.pool.n} "
                     f"pooled), on {len(np.unique(s.groups[s.sel]))} "
                     f"{'events' if s.vacuum else 'hydro backgrounds'}; effective Z's per "
                     "class " + ", ".join(f"{lo:.1f}-{hi:.1f}: {s.n_eff(f[:, c]):.0f}"
                                          for c, (lo, hi) in enumerate(da.GAPS))
                     + f"; with a jet2 {(s.w * f.sum(1)).sum() / max((s.w * (s.z_pt > a.zpt)).sum(), 1e-300):.3f}"
                     f" of the Z's above {a.zpt:g} GeV")
    if pb.fake is not None:
        f = pb.classes["gap"][0]
        lines.append("Pb+Pb fake jet2 (the cuts on the Z's own bulk_bg, the same oversample), "
                     "relative to the full events' jet2, per class: " + ", ".join(
                         f"{lo:.1f}-{hi:.1f}: {(pb.w * pb.fake[:, c] * pb.sel).sum() / max((pb.w * f[:, c]).sum(), 1e-300):.3f}"
                         for c, (lo, hi) in enumerate(da.GAPS)))
    lines.append("")

    # ── the CMS form: every content
    res = {}
    for s, contents in ((pb, CONTENTS_PBPB), (pp, ("F", "O"))):
        for c in contents:
            fn, shp = s.observable(c, strict=(c == "O"))
            v, e = da.jackknife(fn, blk[s.label])
            res[(s.label, c)] = (split(v, shp), split(e, shp))
    # T without the sideband subtraction (the sideband's share, B4.5) and with the pooled ME
    fn, shp = pb.observable("T", subtract=False)
    v, e = da.jackknife(fn, blk["Pb+Pb"])
    res[("Pb+Pb", "T, no sideband")] = (split(v, shp), split(e, shp))
    fn, shp = pb.observable("T", acceptance="pooled")
    v, e = da.jackknife(fn, blk["Pb+Pb"])
    res[("Pb+Pb", "T, pooled ME")] = (split(v, shp), split(e, shp))

    def dif(key):
        (d, _, _), (de, _, _) = res[key]
        return d, de

    # Pb+Pb - pp (independent samples: errors in quadrature)
    sub = {}
    for c, cp in (("F", "F"), ("T", "F"), ("O", "O")):
        d1, e1 = dif(("Pb+Pb", c))
        d2, e2 = dif(("pp", cp))
        sub[c] = (d1 - d2, np.sqrt(e1 ** 2 + e2 ** 2))

    # ── summary: R_asym - R_sym in CMS's bins, integrals, closure
    gaps = [f"{lo:.1f}-{hi:.1f}" for lo, hi in da.GAPS[1:]]
    lines.append("R_asym - R_sym (near side, per trigger per unit Delta y, sideband-subtracted),"
                 " integrated over -2.5 < Delta y < -0.5 (CMS's 10 bins centred -2.4 ... -0.6;"
                 " CMS's Fig. 3 lists 5x this if its Fig. 2 is per unit Delta eta):")
    for ip, ptl in enumerate(PT_LABELS):
        lines.append(f"  p_T {ptl}")
        for key in [("Pb+Pb", c) for c in CONTENTS_PBPB] + [("pp", "F"), ("pp", "O")]:
            d, de = dif(key)
            vals = ", ".join(f"{g}: {da.integral(d[i, ip]):+.3f} ± "
                             f"{np.sqrt(da.integral(de[i, ip] ** 2) * da.WIDTH):.3f}"
                             for i, g in enumerate(gaps))
            lines.append(f"    {key[0]:5s} {key[1]:2s}  {vals}")
        for c, (d, de) in sub.items():
            vals = ", ".join(f"{g}: {da.integral(d[i, ip]):+.3f}" for i, g in enumerate(gaps))
            lines.append(f"    Pb+Pb {c} - pp  {vals}")
    lines.append("  (integral errors: the bins' jackknife errors added in quadrature, bins "
                 "treated as independent)")
    lines.append("")

    def flat_chi2(d, de, lo=da.SIGNAL_BINS[0], hi=da.SIGNAL_BINS[1]):
        m = (da.DYC > lo) & (da.DYC < hi)
        x, e = da.rebin(d)[0], da.rebin(de ** 2)[0]
        cm = da.rebin(np.where(m, 1.0, np.nan))[0]
        ok = np.isfinite(cm) & np.isfinite(x) & (e > 0)
        # rebin of de^2 averages the squares of 2 bins: the error of a 0.2 bin's mean
        return float(np.sum(x[ok] ** 2 / (e[ok] / 2))), int(ok.sum())

    lines.append("Closure (B4), 1-2 GeV and 2-4 GeV, chi^2 / bins against 0 over -2.5 < Delta y"
                 " < -0.5 in CMS's 0.2 bins, per gap class:")
    for name, key in (("(2) N flat", ("Pb+Pb", "N")), ("(3) B flat", ("Pb+Pb", "B")),
                      ("(6) pp flat", ("pp", "F"))):
        d, de = dif(key)
        lines.append(f"  {name:14s} " + "; ".join(
            f"{ptl}: " + ", ".join("{:.1f}/{}".format(*flat_chi2(d[i, ip], de[i, ip]))
                                   for i in range(len(gaps)))
            for ip, ptl in enumerate(PT_LABELS[:2])))
    dF, eF = dif(("Pb+Pb", "F"))
    dT, eT = dif(("Pb+Pb", "T"))
    lines.append("  (4) F - T      " + "; ".join(
        f"{ptl}: " + ", ".join("{:.1f}/{}".format(*flat_chi2(dF[i, ip] - dT[i, ip],
                                                              np.sqrt(eF[i, ip] ** 2)))
                               for i in range(len(gaps)))
        for ip, ptl in enumerate(PT_LABELS[:2])) + "  (F's errors only: F and T share events)")
    (_, ycT, levT), _ = res[("Pb+Pb", "T, no sideband")]
    lines.append("  (5) T's sideband level per class (per unit Delta y, 1-2 GeV): "
                 + ", ".join(f"{lo:.1f}-{hi:.1f}: {levT[c, 0]:+.4f}"
                             for c, (lo, hi) in enumerate(da.GAPS)))
    # (1) the classes summed against the Z-side Delta y of S_full (to rounding)
    f = pb.classes["gap"][0]
    wf = pb.w * f.sum(axis=1)
    y_sum = np.tensordot(wf, pb.pool.Y_full.astype(float)[:, :, 0], axes=1)
    s_side = np.tensordot(wf, pb.pool.S_full.astype(float)[..., zm.ZSIDE].sum(-1), axes=1)
    y04 = y_sum.reshape(y_sum.shape[0], -1, 4).sum(-1)
    rel = np.max(np.abs(y04 - s_side)) / max(np.max(np.abs(s_side)), 1e-300)
    lines.append(f"  (1) classes summed (unreflected, Z side, 0.4 bins) vs S_full's Z side: "
                 f"max relative difference {rel:.1e}")
    lines.append("")

    # R4: the noise per effective Z, per 0.2 bin, class 1.0-1.5, 1-2 GeV
    i_cls = 1
    fc = pb.classes["gap"][0][:, i_cls + 1]
    ne = pb.n_eff(fc)
    lines.append(f"Noise (R4), class {gaps[i_cls]}, 1-2 GeV, per 0.2 bin in -2.5..-0.5 (median), "
                 f"{ne:.0f} effective Z's in the class:")
    for c in ("T", "F", "O"):
        de = dif(("Pb+Pb", c))[1][i_cls, 0]
        e02 = np.sqrt(da.rebin(de ** 2)[0] / 2)
        m = (da.rebin(np.zeros(da.N_DY))[1] > -2.5) & (da.rebin(np.zeros(da.N_DY))[1] < -0.5)
        med = float(np.nanmedian(e02[m]))
        lines.append(f"  {c}: ±{med:.3f} per bin; per effective Z {med * np.sqrt(ne):.2f}; "
                     f"10x the events: ±{med / np.sqrt(10):.3f}")
    lines.append("")

    # ── Yang-Wang: lab eta on the Z side
    yw = {}
    for s, contents in ((pb, ("F", "T", "B")), (pp, ("F",))):
        for c in contents:
            for scheme in ("yw", "hemi"):
                fn, shp = s.lab(c, scheme)
                v, e = da.jackknife(fn, blk[s.label])
                yw[(s.label, c, scheme)] = (v.reshape(shp), e.reshape(shp))
    lines.append("Yang-Wang (2508.04194): Z-side dN/deta, |eta_jet2| 0.8-1.6 minus < 0.8 (yw) and "
                 "other minus same hemisphere (hemi), reflected to eta_jet2 > 0; the mean over "
                 "0.5 < eta < 1.5 at 1-2 GeV:")
    m = (da.ETAC > 0.5) & (da.ETAC < 1.5)
    for key, (v, e) in yw.items():
        d, de = v[1] - v[0], np.sqrt(e[1] ** 2 + e[0] ** 2)
        lines.append(f"  {key[0]:5s} {key[1]} {key[2]:4s}: {np.nanmean(d[0, m]):+.4f} ± "
                     f"{np.sqrt(np.nansum(de[0, m] ** 2)) / m.sum():.4f}")

    # ── figures
    x02 = da.rebin(np.zeros(da.N_DY))[1]

    def show(ax, d, de, **kw):
        y, e2 = da.rebin(d)[0], da.rebin(de ** 2)[0]
        ax.errorbar(x02, y, np.sqrt(e2 / 2), fmt=kw.pop("fmt", "o"), ms=3.5, **kw)

    fig, ax = figdata.subplots(2, 3, figsize=(16, 8), sharex=True)
    for ip in range(2):
        for i, g in enumerate(gaps):
            axr = ax[ip, i]
            for key, lab_, col in ((("Pb+Pb", "F"), "Pb+Pb, full event (F)", "C3"),
                                   (("Pb+Pb", "T"), "Pb+Pb, truth (T)", "C1"),
                                   (("pp", "F"), "pp", "C0")):
                d, de = dif(key)
                show(axr, d[i, ip], de[i, ip], color=col, label=lab_)
            d, de = sub["F"]
            show(axr, d[i, ip], de[i, ip], color="k", fmt="s", mfc="none", label="Pb+Pb − pp (F)")
            axr.axhline(0, color="0.5", lw=0.8)
            axr.axvspan(-2.5, -0.5, color="0.9", zorder=0)
            axr.set_title(f"R_asym ({g}) − R_sym, {PT_LABELS[ip]}")
            axr.set_xlabel("Δy = η − y_Z (reflected: jet2 at Δy < 0)")
            axr.legend(fontsize=7)
    fig.suptitle(f"The Z as jet1 of CMS HIN-25-012: {cuts}", y=0.995)
    fig.tight_layout()
    figdata.savefig(fig, os.path.join(a.outdir, f"deta_asym{a.tag}.png"), dpi=100)
    plt.close(fig)

    fig, ax = figdata.subplots(2, 3, figsize=(16, 8), sharex=True)
    for ip in range(2):
        for i, g in enumerate(gaps):
            axr = ax[ip, i]
            for key, lab_, col in ((("Pb+Pb", "N"), "N: no-wake twin", "C2"),
                                   (("Pb+Pb", "B"), "B: bulk alone", "C4"),
                                   (("Pb+Pb", "W"), "W: wake alone", "C1"),
                                   (("Pb+Pb", "O"), "O: one oversample", "C5")):
                d, de = dif(key)
                show(axr, d[i, ip], de[i, ip], color=col, label=lab_)
            show(axr, dF[i, ip] - dT[i, ip], eF[i, ip], color="k", fmt="s", mfc="none",
                 label="F − T")
            axr.axhline(0, color="0.5", lw=0.8)
            axr.set_title(f"closure: {g} − R_sym, {PT_LABELS[ip]}")
            axr.set_xlabel("Δy")
            axr.legend(fontsize=7)
    fig.tight_layout()
    figdata.savefig(fig, os.path.join(a.outdir, f"deta_closure{a.tag}.png"), dpi=100)
    plt.close(fig)

    fig, ax = figdata.subplots(2, 2, figsize=(12, 8), sharex=True)
    for col_, scheme, title in ((0, "yw", "|η_jet2| 0.8–1.6 minus < 0.8"),
                                (1, "hemi", "other minus same hemisphere")):
        for ip in range(2):
            axr = ax[ip, col_]
            for key, lab_, col in ((("Pb+Pb", "F", scheme), "Pb+Pb, full event", "C3"),
                                   (("Pb+Pb", "T", scheme), "Pb+Pb, truth", "C1"),
                                   (("Pb+Pb", "B", scheme), "Pb+Pb, bulk alone", "C4"),
                                   (("pp", "F", scheme), "pp", "C0")):
                v, e = yw[key]
                axr.errorbar(da.ETAC, v[1, ip] - v[0, ip], np.sqrt(e[1, ip] ** 2 + e[0, ip] ** 2),
                             fmt="o", ms=3, color=col, label=lab_)
            axr.axhline(0, color="0.5", lw=0.8)
            axr.set_title(f"Yang–Wang: Z side, {title}, {PT_LABELS[ip]}")
            axr.set_xlabel("η (reflected: jet2 at η > 0)")
            axr.legend(fontsize=7)
    fig.tight_layout()
    figdata.savefig(fig, os.path.join(a.outdir, f"deta_yw{a.tag}.png"), dpi=100)
    plt.close(fig)

    fig, ax = figdata.subplots(1, 2, figsize=(12, 4.5))
    for ip in range(2):
        axr = ax[ip]
        for j, (key, lab_, col) in enumerate(((("Pb+Pb", "F"), "Pb+Pb F", "C3"),
                                              (("Pb+Pb", "T"), "Pb+Pb T", "C1"),
                                              (("Pb+Pb", "N"), "Pb+Pb N", "C2"),
                                              (("pp", "F"), "pp", "C0"))):
            d, de = dif(key)
            v = [da.integral(d[i, ip]) for i in range(len(gaps))]
            e = [np.sqrt(da.integral(de[i, ip] ** 2) * da.WIDTH) for i in range(len(gaps))]
            axr.errorbar(np.arange(len(gaps)) + 0.08 * (j - 1.5), v, e, fmt="o", color=col,
                         label=lab_)
        axr.set_xticks(np.arange(len(gaps)))
        axr.set_xticklabels(gaps)
        axr.axhline(0, color="0.5", lw=0.8)
        axr.set_xlabel("|Δη_jet1,jet2| of R_asym")
        axr.set_ylabel("∫ (R_asym − R_sym) dΔy over −2.5..−0.5")
        axr.set_title(PT_LABELS[ip])
        axr.legend(fontsize=7)
    fig.tight_layout()
    figdata.savefig(fig, os.path.join(a.outdir, f"deta_integrated{a.tag}.png"), dpi=100)
    plt.close(fig)

    cms_comparison(a, lines, sub, res, plt, gaps)
    text = "\n".join(lines) + "\n"
    with open(os.path.join(a.outdir, f"deta_summary{a.tag}.txt"), "w") as fo:
        fo.write(text)
    print(text, end="")


def cms_comparison(a, lines, sub, res, plt, gaps):
    """The model against CMS HIN-25-012's HEPData (Tables 1-4) -> deta_cms<tag>.png and the
    summary's chi^2 lines; skipped with a note without the record."""
    if not os.path.isdir(a.hepdata):
        lines.append(f"CMS HIN-25-012: no HEPData record in {a.hepdata} (../zjet/get_hepdata.sh "
                     "25012): not compared")
        return
    t12, t3, t4 = (cms25012.table12(a.hepdata), cms25012.table3(a.hepdata),
                   cms25012.table4(a.hepdata))
    norm = a.cms_norm
    pts = {"1-2": 0, "2-4": 1}
    gi = {g: i for i, g in enumerate(gaps)}

    def model(d, de):
        """-> (values, errors, centres) in CMS's 0.2 bins and convention."""
        v, c = da.rebin(d)
        e = np.sqrt(da.rebin(de ** 2)[0] / 2)
        return cms25012.to_cms(v, norm), cms25012.to_cms(e, norm), c

    lines.append("")
    lines.append(f"CMS HIN-25-012 (HEPData ins3123406), the model in CMS's 0.2 bins, CMS's tables "
                 f"read as {norm} (Table 4 = the sum of Table 3's 10 bins):"
                 + ("  [--trigger z: the Z as jet1 against CMS's dijets, not like for like]"
                    if a.trigger == "z" else ""))
    fig, ax = figdata.subplots(3, 4, figsize=(21, 12))
    # Table 3: PbPb 0-30 % - pp
    for j, (key, data) in enumerate(sorted(t3.items(), key=lambda kv: (kv[0][0], kv[0][1]))):
        pt, gap = key
        axr = ax[0, j]
        axr.errorbar(data["x"], data["value"], data["total"], fmt="o", color="k", ms=4,
                     label="CMS PbPb 0–30 % − pp")
        for c, col in (("F", "C3"), ("T", "C1")):
            v, e, x = model(*(arr[gi[gap], pts[pt]] for arr in sub[c]))
            axr.errorbar(x + 0.02, v, e, fmt="s", ms=3.5, mfc="none", color=col,
                         label=f"model Pb+Pb {c} − pp")
            ch, nb = cms25012.chi2(v, e, data)
            lines.append(f"  Table 3 PbPb-pp {pt} GeV, {gap}: model {c} chi2 {ch:6.1f}/{nb}")
        axr.axhline(0, color="0.5", lw=0.8)
        axr.set_title(f"PbPb − pp, R_asym ({gap}) − R_sym, {pt} GeV")
        axr.set_xlabel("Δη (model: Δy)")
        axr.legend(fontsize=7)
    # Tables 1-2: pp and PbPb 0-30 %, R_asym 1.0-1.5
    for j, (pt, system) in enumerate((("1-2", "pp"), ("1-2", "0-30"), ("2-4", "pp"),
                                      ("2-4", "0-30"))):
        data = t12[(pt, system)]
        axr = ax[1, j]
        axr.errorbar(data["x"], data["value"], data["total"], fmt="o", color="k", ms=4,
                     label=f"CMS {'pp' if system == 'pp' else 'PbPb ' + system + ' %'}")
        keys = [("pp", "F")] if system == "pp" else [("Pb+Pb", "F"), ("Pb+Pb", "T")]
        for key, col in zip(keys, ("C0", "C3", "C1") if system == "pp" else ("C3", "C1")):
            (d, _, _), (de, _, _) = res[key]
            v, e, x = model(d[gi["1.0-1.5"], pts[pt]], de[gi["1.0-1.5"], pts[pt]])
            axr.errorbar(x + 0.02, v, e, fmt="s", ms=3.5, mfc="none", color=col,
                         label=f"model {key[0]} {key[1]}")
            ch, nb = cms25012.chi2(v, e, data)
            lines.append(f"  Table {1 if pt == '1-2' else 2} {system:5s} {pt} GeV, 1.0-1.5: "
                         f"model {key[0]} {key[1]} chi2 {ch:6.1f}/{nb}")
        axr.axhline(0, color="0.5", lw=0.8)
        axr.set_title(f"{'pp' if system == 'pp' else 'PbPb ' + system + ' %'}: R_asym (1.0–1.5) − "
                      f"R_sym, {pt} GeV")
        axr.set_xlabel("Δη (model: Δy)")
        axr.legend(fontsize=7)
    # Table 4: the integral against centrality; the model at 0-30 %
    for j, (key, data) in enumerate(sorted(t4.items(), key=lambda kv: (kv[0][1], kv[0][0]))):
        pt, gap = key
        axr = ax[2, j]
        axr.errorbar(data["x"], data["value"], data["total"], fmt="o", color="k", ms=4,
                     label="CMS PbPb − pp")
        for k_, (c, col) in enumerate((("F", "C3"), ("T", "C1"))):
            v, e, x = model(*(arr[gi[gap], pts[pt]] for arr in sub[c]))
            m = (x > -2.45) & (x < -0.55)
            iv, ie = np.nansum(v[m]), np.sqrt(np.nansum(e[m] ** 2))
            axr.errorbar([15 + 3 * (k_ - 0.5)], [iv], [ie], fmt="s", ms=5, mfc="none",
                         color=col, label=f"model Pb+Pb {c} − pp (0–30 %)")
            i30 = int(np.argmin(np.abs(data["x"] - 15)))
            pull = (iv - data["value"][i30]) / np.hypot(ie, data["total"][i30])
            lines.append(f"  Table 4 {pt} GeV, {gap}, 0-30 %: CMS {data['value'][i30]:+.3f} ± "
                         f"{data['total'][i30]:.3f}, model {c} {iv:+.3f} ± {ie:.3f} "
                         f"(pull {pull:+.1f})")
        axr.axhline(0, color="0.5", lw=0.8)
        axr.set_title(f"∫ over −2.4…−0.6, R_asym ({gap}), {pt} GeV")
        axr.set_xlabel("centrality [%]")
        axr.legend(fontsize=7)
    fig.suptitle(f"CMS HIN-25-012 against the model ({'the Z as jet1' if a.trigger == 'z' else 'jet1'};"
                 f" CMS's tables read as {norm})", y=0.995)
    fig.tight_layout()
    figdata.savefig(fig, os.path.join(a.outdir, f"deta_cms{a.tag}.png"), dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    main()
