#!/usr/bin/env python3
"""The Z+jet test against CMS 2507.09307 (HEPData ins2945894), from data/hists.npz
(accumulate.py) -> figures/zjet_vs_cms2507_dphi.png, figures/zjet_vs_cms2507_dy.png and
figures/summary.txt.

CMS's observable: Delta<N_ch> = S - B, B the mixed-event yield (the Z of one event with the
tracks of another Z event), merged over +/- Delta phi (+/- Delta y) and scaled by 0.5 (the
mean of the +/- bins), per unit angle.  The model's S is its Z-correlated yield, wake
(<bulk_jet> - <bulk_bg>, paired) + jet fragments; the uncorrelated bulk cancels in S - B
and is left out of both.
* Delta phi (all Delta y): B is flat in Delta phi -> S minus its Delta phi average.
* Delta y (|Delta phi| < pi/2): B = half (the Z side) of the y_Z of event i folded with
  the lab-frame eta distribution of event j != i.
Errors: Pb+Pb per-event statistics (+) a jackknife over the hydro backgrounds (the events
of one background are correlated), in quadrature; pp per-event statistics.  chi^2 over the
independent bins (the reflected half), CMS stat (+) syst, bin-to-bin correlations ignored."""
import argparse
import os

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import figdata  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PT_TAGS = ["1_<_p_t_<_2", "2_<_p_t_<_4", "4_<_p_t_<_10"]
PT_LABELS = ["1 < p_T^ch < 2 GeV", "2 < p_T^ch < 4 GeV", "4 < p_T^ch < 10 GeV"]


def cms(hep, var, kind, pt_tag):
    """-> bin centres, value, stat, syst of a HEPData table; var 'phi' / 'y', kind
    'pbpb' (0-30 %) / 'pp'."""
    v = "\\delta\\phi_{ch,z}" if var == "phi" else "\\delta_y_{ch,z}"
    k = "in_pbpb_0-30%" if kind == "pbpb" else "spectra_in_pp"
    with open(os.path.join(hep, f"${v}$_{k}_for_${pt_tag}$_gev.yaml")) as f:
        d = yaml.safe_load(f)
    x = d["independent_variables"][0]["values"]
    vals = d["dependent_variables"][0]["values"]
    centre = np.array([0.5 * (b["low"] + b["high"]) for b in x])
    value = np.array([float(b["value"]) for b in vals])
    stat = np.array([float(b["errors"][0]["symerror"]) for b in vals])
    syst = np.array([float(b["errors"][1]["symerror"]) for b in vals])
    return centre, value, stat, syst


class Model:
    """CMS's observables from the summed histograms of a set of events."""

    def __init__(self, deta, dphi, eta):
        self.deta, self.dphi, self.eta = deta, dphi, eta
        self.cdeta = 0.5 * (deta[1:] + deta[:-1])
        self.cphi = 0.5 * (dphi[1:] + dphi[:-1])
        self.ceta = 0.5 * (eta[1:] + eta[:-1])
        # the bin at -dphi (dphi in [-pi/2, 3pi/2))
        neg = (-self.cphi + np.pi / 2) % (2 * np.pi) - np.pi / 2
        self.mirror = np.array([int(np.argmin(np.abs(x - self.cphi))) for x in neg])
        self.zside = np.abs(self.cphi) < np.pi / 2

    def observables(self, H, E, T2, yz, wz):
        """H, E: sum_i w_i x (pT, deta, dphi, eta) yields and their errors; T2: sum_i
        w_i^2 x (pT, deta) yields (all dphi); yz, wz: the events' y_Z and weights.
        -> dphi (value, error) per radian, (pT, 20); dy (value, error) per unit, (pT, 24)."""
        W, W2 = wz.sum(), (wz ** 2).sum()
        # Delta phi
        hp = H.sum(axis=(1, 3)) / W
        ep = np.sqrt((E ** 2).sum(axis=(1, 3))) / W
        hp = 0.5 * (hp + hp[:, self.mirror])
        ep = 0.5 * np.hypot(ep, ep[:, self.mirror])
        width = np.diff(self.dphi)
        dphi = ((hp - hp.mean(axis=1, keepdims=True)) / width, ep / width)
        # Delta y, Z side: same event - half the mixed events
        s = H[:, :, self.zside, :].sum(axis=(2, 3)) / W
        es = np.sqrt((E[:, :, self.zside, :] ** 2).sum(axis=(2, 3))) / W
        rho = H.sum(axis=(1, 2)) / W                       # (pT, eta_lab) per Z
        erho = np.sqrt((E ** 2).sum(axis=(1, 2))) / W
        nb = len(self.cdeta)
        conv = np.zeros_like(s)
        for y, wi in zip(yz, wz):                          # all pairs (i, j)
            j = np.clip(np.searchsorted(self.deta, self.ceta - y) - 1, 0, nb - 1)
            for p in range(len(rho)):
                conv[p] += wi / W * np.bincount(j, rho[p], nb)
        mixed = (W ** 2 * conv - T2) / (W ** 2 - W2)       # pairs i != j
        # its error: rho's errors, shifted by the mean y_Z
        j = np.clip(np.searchsorted(self.deta, self.ceta - np.average(yz, weights=wz)) - 1,
                    0, nb - 1)
        emixed = np.sqrt(np.array([np.bincount(j, e ** 2, nb) for e in erho]))
        dy = s - 0.5 * mixed
        edy = np.hypot(es, 0.5 * emixed)
        dy = 0.5 * (dy + dy[:, ::-1])
        edy = 0.5 * np.hypot(edy, edy[:, ::-1])
        bw = np.diff(self.deta)
        return dphi, (dy / bw, edy / bw)


def rebin_y(h, e, k):
    """|Delta y| < 4 (bins 2..21 of 0.4) into bins of 0.4 k: their mean."""
    h, e = h[..., 2:22], e[..., 2:22]
    shp = h.shape[:-1] + (h.shape[-1] // k, k)
    return h.reshape(shp).mean(-1), np.sqrt((e.reshape(shp) ** 2).sum(-1)) / k


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hists", default=os.path.join(HERE, "data", "hists.npz"))
    ap.add_argument("--hepdata", default=os.path.join(HERE, "data", "hepdata",
                                                      "HEPData-ins2945894-v1-yaml"))
    ap.add_argument("--outdir", default=os.path.join(HERE, "figures"))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    d = np.load(a.hists)
    model = Model(d["DETA"], d["DPHI"], d["ETA"])
    sel, w, yz, group = d["sel"], d["w"], d["y_z"], d["group"]
    groups = list(d["groups"])

    def pbpb(gs):
        n = {g: int(d[f"g{g}_n"]) for g in gs}
        H = sum(n[g] * (d[f"g{g}_wake"] + d[f"g{g}_jet"]) for g in gs)
        E = np.sqrt(sum(n[g] ** 2 * (d[f"g{g}_wake_err"] ** 2 + d[f"g{g}_jet_err"] ** 2)
                        for g in gs))
        T2 = sum(n[g] * (d[f"g{g}_wake_w2"] + d[f"g{g}_jet_w2"]) for g in gs)
        m = sel & np.isin(group, gs)
        (p, ep), (y, ey) = model.observables(H, E, T2, yz[m], w[m])
        y, ey = rebin_y(y, ey, 2)                          # CMS Pb+Pb: 0.8 in Delta y
        return dict(phi=(p, ep), y=(y, ey))

    pb = pbpb(groups)
    jack = [pbpb([g for g in groups if g != x]) for x in groups]
    G = len(groups)
    for var in ("phi", "y"):
        th = np.array([j[var][0] for j in jack])
        ejk = np.sqrt((G - 1) / G * ((th - th.mean(0)) ** 2).sum(0))
        pb[var] = (pb[var][0], np.hypot(pb[var][1], ejk), pb[var][1], ejk)

    sv, wv = d["vac_sel"], d["vac_w"]
    nv = int(sv.sum())
    (p, ep), (y, ey) = model.observables(nv * d["vac"], nv * d["vac_err"], nv * d["vac_w2"],
                                         d["vac_y_z"][sv], wv[sv])
    pp = dict(phi=(p, ep), y=rebin_y(y, ey, 1), y08=rebin_y(y, ey, 2))

    n_pb, ws = int(sel.sum()), w[sel]
    n_eff = ws.sum() ** 2 / (ws ** 2).sum()
    tag = f"X-SCAPE (Pb+Pb: {n_pb} Zs, {G} hydro events; pp: {nv:,} Zs)"
    lines = [f"Pb+Pb 0-30 %: {n_pb} selected Z events (effective {n_eff:.0f}), {G} hydro "
             f"backgrounds; pp: {nv} selected Z events",
             "jackknife / per-event error (Pb+Pb, median over bins): " + ", ".join(
                 f"{v} {l.split(' < p')[0]}-{l.split('< ')[-1]}: "
                 f"{np.median(pb[v][3][c] / pb[v][2][c]):.1f}"
                 for v in ("phi", "y") for c, l in enumerate(PT_LABELS)), "",
             f"{'':4s} {'':20s} {'':13s} {'model':>14s} {'CMS':>14s}   "
             f"{'model':>14s} {'CMS':>14s}   chi2/ndf",
             f"{'':4s} {'':20s} {'':13s} {'bin at 0':>30s}   {'bin at pi':>30s}"]
    xy = np.linspace(-3.6, 3.6, 10), np.linspace(-3.8, 3.8, 20)
    for var in ("phi", "y"):
        fig, ax = figdata.subplots(3, 3, figsize=(16, 11), sharex=True)
        for c in range(3):
            mp, ep = pb[var][0][c], pb[var][1][c]
            if var == "phi":
                mq, eq = pp["phi"][0][c], pp["phi"][1][c]
                mq2, eq2 = mq, eq
                xp = xq = model.cphi
            else:
                mq, eq = pp["y"][0][c], pp["y"][1][c]
                mq2, eq2 = pp["y08"][0][c], pp["y08"][1][c]
                xp, xq = xy
            cp, yp, sp, yp_sy = cms(a.hepdata, var, "pbpb", PT_TAGS[c])
            cq, yq, sq, yq_sy = cms(a.hepdata, var, "pp", PT_TAGS[c])
            tp, tq = np.hypot(sp, yp_sy), np.hypot(sq, yq_sy)
            if var == "y":                     # CMS pp: 0.4 in Delta y -> 0.8
                yq2 = yq.reshape(-1, 2).mean(1)
                tq2 = np.sqrt((tq.reshape(-1, 2) ** 2).sum(1)) / 2
            else:
                yq2, tq2 = yq, tq
            rows = ((mp, ep, xp, yp, tp, cp, "Pb+Pb 0–30 %"),
                    (mq, eq, xq, yq, tq, cq, "pp"),
                    (mp - mq2, np.hypot(ep, eq2), xp, yp - yq2, np.hypot(tp, tq2), cp,
                     "Pb+Pb − pp"))
            for k, (m_, e_, x_, y_, t_, cx, lab) in enumerate(rows):
                a_ = ax[k, c]
                a_.errorbar(cx, y_, t_, fmt="o", color="k", ms=4,
                            label="CMS 2507.09307 (stat ⊕ syst)")
                a_.errorbar(x_, m_, e_, fmt="s", color="C3", ms=4, mfc="none", label=tag)
                a_.axhline(0, color="0.6", lw=0.6)
                a_.set_ylabel(f"{lab}: d⟨ΔN_ch⟩/dΔ{'φ' if var == 'phi' else 'y'}")
                if k == 0:
                    a_.set_title(PT_LABELS[c] + ("" if var == "phi" else ",  |Δφ| < π/2"))
                if k == 2:
                    a_.set_xlabel("Δφ_ch,Z" if var == "phi" else "Δy_ch,Z = y_Z − η_ch")
                if c == 0 and k == 0:
                    a_.legend(fontsize=8)
                ndf = len(m_) // 2
                chi2 = 0.5 * np.sum((m_ - y_) ** 2 / (e_ ** 2 + t_ ** 2))
                i0 = int(np.argmin(np.abs(x_ - (0.157 if var == "phi" else x_[x_ > 0].min()))))
                far = ""
                if var == "phi":
                    ipi = int(np.argmin(np.abs(x_ - 2.985)))
                    far = f"{m_[ipi]:+6.2f} ± {e_[ipi]:4.2f}  {y_[ipi]:+6.2f} ± {t_[ipi]:4.2f}"
                lines.append(f"{'Δ' + var:4s} {PT_LABELS[c]:20s} {lab:13s} "
                             f"{m_[i0]:+6.2f} ± {e_[i0]:4.2f}  {y_[i0]:+6.2f} ± {t_[i0]:4.2f}   "
                             f"{far:30s}   {chi2:5.1f}/{ndf}")
        fig.suptitle("Z–hadron correlations, Pb+Pb 5.02 TeV 0–30 % and pp, "
                     "40 < p_T^Z < 350 GeV, |y_Z| < 2.4, |η_ch| < 2.4", y=0.995)
        fig.tight_layout()
        path = os.path.join(a.outdir, f"zjet_vs_cms2507_d{var}.png")
        figdata.savefig(fig, path, dpi=110)
        print(f"-> {path}")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(a.outdir, "summary.txt"), "w") as f:
        f.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
