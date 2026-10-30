#!/usr/bin/env python3
"""The Z-axis dipole of the soft sector as a wake measurement (WAKE_SYNTHESIS.md): (a) the
dipole relative to the Z in Delta R annuli around jet2 (the recoil), per track p_T class, a
"Z-tagged missing p_T"; (b) its profile in eta - eta_jet2, and its localization (core minus
ring).  Pb+Pb - pp per Pythia setting, from the maps of ../wake_iso.py (``zring_*``,
``zeta_*``), with the errors CMS would have: the noise of one measured event (``nzr_*``,
``nzc_*``, ``nze_*``: the full event's single-sample covariance plus the event-to-event spread)
scaled to CMS's Z count.

    python compare_dipole.py [--data runs/wake_scan/data] [--n-cms 1550] [--n-pp 6200]
        [--axis-eff 1.0] [--pp-all <pp map>] [--setting NAME:RUN:PP_RUN:LABEL ...]

Signs: d = -sum cos(phi - phi_Z) (tracks toward the recoil count positive), P_par = -sum p_T
cos(phi - phi_Z) [GeV], per Z.  The annuli are pairs: around jet2 and around its mirror at
(eta_jet2, phi_jet2 + pi) on the Z's side, every track in the annulus of the nearer axis.  The
pair is symmetric under phi -> phi + pi, so the bulk drops out on average (an annulus around
jet2 alone holds -N_bulk of cos ~ -1 and would need a mixed-event subtraction); the profile
in eta - eta_jet2 uses the full azimuth.  The Z's: those with |eta_rec| < 1 (the maps' J frame), with the
recoil parton's direction as jet2's (truth; a reconstructed jet2 adds its resolution and,
above a p_T threshold, loses the Z's whose jet fell below it: --axis-eff).  The bulk averages
to zero against the Z's direction and enters only the noise.  Projected errors:
sigma^2 = a^T Cov_PbPb a / N_PbPb + a^T Cov_pp a / N_pp with N_PbPb = n_cms x (the J
fraction) x axis_eff.  n_cms = 1550: CMS 2507.09307's effective Pb+Pb Z count
(ANALYSIS_c00_01.md); n_pp: an assumption (4x).  For comparison: CMS's own statistical error
of the Z-side Delta y in its first bin (HEPData ins2945894), 0.286 at 1-2 GeV and 0.123 at
2-4 GeV, against the model's value there (mix.py / wake_decomp.py).
-> <outdir>/dipole_annuli{tag}.png, dipole_eta{tag}.png (with .npz/.root), dipole{tag}.txt."""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ZJET = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ZJET)
sys.path.insert(0, os.path.dirname(ZJET))                         # jetcorr
import compare_iso as ci  # noqa: E402
import wake_iso as wi  # noqa: E402
from jetcorr import figdata, runs  # noqa: E402

DEFAULT = ("campaign:tau_delay_1p0:pp_campaign:ISR + MPI on (campaign)",
           "mpi_off:mpi_off_e_threshold_2p0:pp_mpi_off:MPI off",
           "isr_mpi_off:isr_mpi_off_e_threshold_2p0:pp_isr_mpi_off:ISR + MPI off")
COLS = ("k", "C0", "C3", "C2", "C1")
N_R, N_Q, N_S = len(wi.ZR_EDGES) - 1, len(wi.ZR_Q), 2       # sides: around jet2, its mirror
#: annulus regions in Delta R to jet2 (unions of ZR_EDGES bins)
REGIONS = (("ΔR < 0.3", 0.0, 0.3), ("0.3–0.8", 0.3, 0.8), ("0.8–1.6", 0.8, 1.6),
           ("> 1.6", 1.6, 99.0), ("all", 0.0, 99.0))
N_CLS = 3                                         # 0.5-1, 1-2, 2-4 GeV


def region_vector(c, lo, hi, q, sides=(0, 1)):
    """The weights a over the flattened zring vector (N_PT, N_R, N_S, N_Q) for p_T class c, the
    annuli in [lo, hi), quantity q, with the sign toward the recoil.  Both sides (the default:
    around jet2 and around its mirror on the Z's side) make the region symmetric under
    phi -> phi + pi, so the bulk drops out; one side alone is not pedestal-free."""
    a = np.zeros((wi.N_PT, N_R, N_S, N_Q))
    e = wi.ZR_EDGES
    sel = (e[:-1] >= lo - 1e-9) & (e[1:] <= hi + 1e-9)
    for sd in sides:
        a[c, sel, sd, q] = -1.0
    return a.ravel()


def sym(x):
    """A flattened zring vector -> (N_PT, N_R, N_Q), the two sides summed, toward the recoil."""
    return -x.reshape(wi.N_PT, N_R, N_S, N_Q).sum(2)


def cr_vector(c):
    """The weights over the flattened core/ring vector (N_PT, 2): core density minus ring
    density per unit eta - eta_rec, toward the recoil."""
    a = np.zeros((wi.N_PT, 2))
    a[c, 0] = -1.0 / (2 * wi.DIP_CORE)
    a[c, 1] = +1.0 / (2 * (wi.DIP_RING[1] - wi.DIP_RING[0]))
    return a.ravel()


def per_z_cov(v, name):
    """The noise of one measured event: E[C] + E[m m^T] - E[m] E[m]^T."""
    m = v.mean(f"{name}_m")
    return v.mean(f"{name}_C") + v.mean(f"{name}_mm") - np.outer(m, m)


def contents(v, pp):
    """The zring vectors of the wake, the fragments' Pb+Pb - pp and the total, flattened."""
    w = v.mean("zring_wake").ravel()
    f = v.mean("zring_frag").ravel() - pp.mean("zring_frag").ravel()
    return w, f, w + f


def cr_contents(v, pp):
    """The core/ring sums (N_PT, 2) flattened of the wake, fragments - pp, total."""
    a = np.abs(wi.ETA_C)
    core, ring = a < wi.DIP_CORE, (a > wi.DIP_RING[0]) & (a < wi.DIP_RING[1])

    def cr(x):
        return np.stack([x[:, core].sum(-1), x[:, ring].sum(-1)], -1).ravel()
    w = cr(v.mean("zeta_wake"))
    f = cr(v.mean("zeta_frag")) - cr(pp.mean("zeta_frag"))
    return w, f, w + f


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=runs.data("wake_scan"))
    ap.add_argument("--setting", action="append", default=None,
                    help="NAME:RUN:PP_RUN:LABEL (data/iso/<run>.npz)")
    ap.add_argument("--pp-all", default=None, help="one pp map for all settings")
    ap.add_argument("--n-cms", type=float, default=1550.0)
    ap.add_argument("--n-pp", type=float, default=None, help="default 4 x --n-cms")
    ap.add_argument("--axis-eff", type=float, default=1.0,
                    help="the fraction of the |eta_jet2| < 1 Z's with a usable jet2 axis")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--blocks", type=int, default=100)
    a = ap.parse_args()
    a.outdir = a.outdir or runs.figures("wake_scan")
    a.n_pp = a.n_pp or 4 * a.n_cms
    tag = a.tag if a.tag is not None else ("_pp_all" if a.pp_all else "")
    load = lambda n: wi.Maps(os.path.join(a.data, "iso", f"{n}.npz"))     # noqa: E731
    S = []
    for spec in (a.setting or DEFAULT):
        name, run, ppr, label = spec.split(":", 3)
        m = load(run)
        pp = wi.Maps(a.pp_all) if a.pp_all else load(ppr)
        for need in ("zring_wake", "nzr_C", "nzc_C"):
            if need not in m.sums:
                raise SystemExit(f"{run}: no {need} (rebuild with wake_iso.py map --rebuild)")
        S.append(dict(name=name, label=label, m=m, pp=pp, run=run))

    L = [f"The Z-axis dipole as a wake measurement (compare_dipole.py); pp: "
         f"{a.pp_all or 'each setting its own'}; projected errors at CMS's Z count: n_cms "
         f"{a.n_cms:g} x the J fraction x axis_eff {a.axis_eff:g}, n_pp {a.n_pp:g}",
         "d = -sum cos(phi - phi_Z) [count], P_par = -sum p_T cos(phi - phi_Z) [GeV], per Z; "
         "annuli in Delta R to jet2 (the recoil); values: the model's jackknife; sigma_CMS: the "
         "projected statistical error; S/sigma = value / sigma_CMS", ""]
    res = {}
    keys = [(q, c, title, lo, hi) for q in range(N_Q) for c in range(N_CLS)
            for title, lo, hi in REGIONS]
    A = np.array([region_vector(c, lo, hi, q) for q, c, _, lo, hi in keys])    # (n_keys, nv)
    Aj = np.array([region_vector(c, lo, hi, q, (0,)) for q, c, _, lo, hi in keys])
    Am = np.array([region_vector(c, lo, hi, q, (1,)) for q, c, _, lo, hi in keys])
    Acr = np.array([cr_vector(c) for c in range(N_CLS)])

    def summary(V):
        """The region sums (total, wake, fragments - pp) and the localization (same order)."""
        w, f, t = contents(V[0], V[1])
        cw, cf, ct = cr_contents(V[0], V[1])
        return np.concatenate([A @ t, A @ w, A @ f, Acr @ ct, Acr @ cw, Acr @ cf, Aj @ w,
                               Am @ w])

    for s in S:
        m, pp = s["m"], s["pp"]
        fJ = m.NJ.sum() / m.N.sum()
        n_pb = a.n_cms * fJ * a.axis_eff
        cov_pb, cov_pp = per_z_cov(wi.View(m), "nzr"), per_z_cov(wi.View(pp), "nzr")
        ccov_pb, ccov_pp = per_z_cov(wi.View(m), "nzc"), per_z_cov(wi.View(pp), "nzc")
        val, err = wi.jackknife(summary, [[m], [pp]], a.blocks)
        nk = len(keys)
        tv, wv, fv = val[:nk], val[nk:2 * nk], val[2 * nk:3 * nk]
        te = err[:nk]
        cbase = 3 * nk
        ct, cw, cf = (val[cbase:cbase + N_CLS], val[cbase + N_CLS:cbase + 2 * N_CLS],
                      val[cbase + 2 * N_CLS:cbase + 3 * N_CLS])
        cte = err[cbase:cbase + N_CLS]
        wj = val[cbase + 3 * N_CLS:cbase + 3 * N_CLS + nk]
        wm = val[cbase + 3 * N_CLS + nk:]
        sig = np.sqrt(np.einsum("ki,ij,kj->k", A, cov_pb, A) / n_pb
                      + np.einsum("ki,ij,kj->k", A, cov_pp, A) / a.n_pp)
        csig = np.sqrt(np.einsum("ki,ij,kj->k", Acr, ccov_pb, Acr) / n_pb
                       + np.einsum("ki,ij,kj->k", Acr, ccov_pp, Acr) / a.n_pp)
        L.append(f"{s['label']} ({s['run']}): {int(m.NJ.sum())} Z's with |eta_rec| < 1 of "
                 f"{m.n_z} (J fraction {fJ:.2f}) -> N_PbPb {n_pb:.0f}")
        rows = {}
        for k, (q, c, title, lo, hi) in enumerate(keys):
            rows[(q, c, title)] = (tv[k], te[k], sig[k], wv[k], fv[k], wj[k], wm[k])
        for q in range(N_Q):
            for c in range(N_CLS):
                L.append(f"  {'d' if q == 0 else 'P_par'} {wi.PT_LABELS[c]:20s} " + " | ".join(
                    f"{title}: total {rows[(q, c, title)][0]:+.3f} (model "
                    f"±{rows[(q, c, title)][1]:.3f}, σ_CMS {rows[(q, c, title)][2]:.3f}, S/σ "
                    f"{rows[(q, c, title)][0] / rows[(q, c, title)][2]:+.1f}) wake "
                    f"{rows[(q, c, title)][3]:+.3f} (jet2 side {rows[(q, c, title)][5]:+.3f}, "
                    f"mirror {rows[(q, c, title)][6]:+.3f}) frag {rows[(q, c, title)][4]:+.3f}"
                    for title, _, _ in REGIONS))
        crows = {c: (ct[c], cte[c], csig[c], cw[c], cf[c]) for c in range(N_CLS)}
        L.append("  localization in eta - eta_jet2 (core |d_eta| < "
                 f"{wi.DIP_CORE:g} minus ring {wi.DIP_RING[0]:g}-{wi.DIP_RING[1]:g}, d per unit "
                 "d_eta): " + " | ".join(
                     f"{wi.PT_LABELS[c]}: total {crows[c][0]:+.3f} (model ±{crows[c][1]:.3f}, "
                     f"σ_CMS {crows[c][2]:.3f}, S/σ {crows[c][0] / crows[c][2]:+.1f}) wake "
                     f"{crows[c][3]:+.3f} frag {crows[c][4]:+.3f}" for c in range(N_CLS)))
        L.append("")
        w_, f_, t_ = contents(wi.View(m), wi.View(pp))
        _, terr = wi.jackknife(lambda V: sym(contents(V[0], V[1])[2]).ravel(), [[m], [pp]],
                               a.blocks)
        res[s["name"]] = dict(rows=rows, crows=crows, w_sym=sym(w_), t_sym=sym(t_),
                              t_err=terr.reshape(wi.N_PT, N_R, N_Q), cov_pb=cov_pb,
                              cov_pp=cov_pp, n_pb=n_pb)
    L.append("For comparison: CMS 2507's Z-side Delta y, first bin: CMS stat 0.286 (1-2 GeV), "
             "0.123 (2-4 GeV), against the model's value there (mix.py, wake_decomp.py).")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # (a) the annuli
    e = wi.ZR_EDGES
    labels = [f"{lo:g}–{hi:g}" for lo, hi in zip(e[:-1], e[1:])]
    fig, ax = figdata.subplots(2, N_CLS, figsize=(5.4 * N_CLS, 8.0), squeeze=False)
    for r_, q in enumerate((0, 1)):
        for c in range(N_CLS):
            aa = ax[r_, c]
            x = np.arange(N_R)
            for i, s in enumerate(S):
                R = res[s["name"]]
                t, te = R["t_sym"][c, :, q], R["t_err"][c, :, q]
                sig = np.array([np.sqrt(region_vector(c, lo, hi, q) @ R["cov_pb"] @
                                        region_vector(c, lo, hi, q) / R["n_pb"] +
                                        region_vector(c, lo, hi, q) @ R["cov_pp"] @
                                        region_vector(c, lo, hi, q) / a.n_pp)
                                for lo, hi in zip(e[:-1], e[1:])])
                aa.errorbar(x + 0.1 * i, t, te, fmt="o-", ms=4, lw=1.0, color=COLS[i],
                            label=f"{s['label']}: Pb+Pb − pp")
                aa.fill_between(x + 0.1 * i, t - sig, t + sig, color=COLS[i], alpha=0.12,
                                lw=0)
                aa.plot(x + 0.1 * i, R["w_sym"][c, :, q], ":", color=COLS[i], lw=0.9)
            aa.axhline(0, color="0.6", lw=0.6)
            aa.set_xticks(np.arange(N_R))
            aa.set_xticklabels(labels, fontsize=7)
            aa.set(xlabel="ΔR to jet2 (the recoil)",
                   ylabel="d per Z" if q == 0 else "P_∥ per Z [GeV]",
                   title=f"{'d = −Σ cos(φ − φ_Z)' if q == 0 else 'P_∥ = −Σ p_T cos(φ − φ_Z)'}"
                         f"\n{wi.PT_LABELS[c]}")
            if r_ == 0 and c == 0:
                aa.legend(fontsize=6)
    fig.suptitle("The Z-axis dipole in annuli around jet2 plus their mirrors on the Z's side "
                 "(pedestal-free), Pb+Pb − pp (points: the model with its jackknife; band: the "
                 "statistical error at CMS's Z count; dotted: the wake alone)", fontsize=10)
    fig.tight_layout()
    os.makedirs(a.outdir, exist_ok=True)
    p1 = os.path.join(a.outdir, f"dipole_annuli{tag}.png")
    figdata.savefig(fig, p1, dpi=90)
    plt.close(fig)

    # (b) the eta profile
    fig, ax = figdata.subplots(1, N_CLS, figsize=(5.4 * N_CLS, 4.4), squeeze=False)
    for c in range(N_CLS):
        aa = ax[0, c]
        for i, s in enumerate(S):
            m, pp = s["m"], s["pp"]
            prof = lambda V: -(V[0].mean("zeta_wake") + V[0].mean("zeta_frag")   # noqa: E731
                               - V[1].mean("zeta_frag")) / wi.D_ETA
            val, err = wi.jackknife(prof, [[m], [pp]], a.blocks)
            vm, vp = wi.View(m), wi.View(pp)
            var = ((vm.mean("nze_v") + vm.mean("nze_m2") - vm.mean("nze_m") ** 2)
                   / res[s["name"]]["n_pb"] + (vp.mean("nze_v") + vp.mean("nze_m2")
                                               - vp.mean("nze_m") ** 2) / a.n_pp) / wi.D_ETA ** 2
            x, y = ci.rebin1(wi.ETA_C, val, 2)
            _, ye = ci.rebin1(wi.ETA_C, err ** 2, 2)
            _, ys = ci.rebin1(wi.ETA_C, var, 2)
            ye, ys = np.sqrt(ye / 2), np.sqrt(ys / 2)
            keep = np.abs(x) < 3.2
            aa.errorbar(x[keep] + 0.02 * i, y[c][keep], ye[c][keep], fmt="o-", ms=3, lw=0.9,
                        color=COLS[i], label=s["label"])
            aa.fill_between(x[keep], (y[c] - ys[c])[keep], (y[c] + ys[c])[keep],
                            color=COLS[i], alpha=0.12, lw=0)
        aa.axhline(0, color="0.6", lw=0.6)
        for xx in (wi.DIP_CORE, -wi.DIP_CORE):
            aa.axvline(xx, color="0.5", ls=":", lw=0.6)
        aa.set(xlabel="η − η_jet2", ylabel="d per Z per unit Δη",
               title=f"the Z-axis dipole against η − η_jet2, Pb+Pb − pp\n{wi.PT_LABELS[c]}")
        if c == 0:
            aa.legend(fontsize=7)
    fig.suptitle("The dipole's η profile around jet2 (points: the model; band: the statistical "
                 "error at CMS's Z count, bins of 0.4)", fontsize=10)
    fig.tight_layout()
    p2 = os.path.join(a.outdir, f"dipole_eta{tag}.png")
    figdata.savefig(fig, p2, dpi=90)
    plt.close(fig)
    txt = os.path.join(a.outdir, f"dipole{tag}.txt")
    with open(txt, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"-> {p1}\n-> {p2}\n-> {txt}")


if __name__ == "__main__":
    main()
