#!/usr/bin/env python3
"""What CMS 2507.09307 would see if the wake could be isolated: the model's wake on its own
(paired bulk_jet - bulk_bg) and the isolated jet-correlated total (wake + fragments - pp, no
mixed event, no mean subtraction), in CMS's Delta phi and Delta y conventions, next to CMS's
Pb+Pb 0-30 % - pp and next to the model's result with CMS's procedure (the mean-subtracted
S - B), for every variation of the scans (tau_delay, e_threshold, the Pythia settings).

    python compare_cms_iso.py [--data runs/wake_scan/data] [--family NAME ...]

CMS's conventions (HEPData ins2945894): Delta phi over all Delta y (tracks |eta| < 2.4), pi/10
bins, +-Delta phi merged and halved, per radian; Delta y on the Z side (|Delta phi| < pi/2),
0.8 bins, +-Delta y merged and halved, per unit Delta y; all per Z.  The model's isolated
pieces are absolute yields (what an ideal background removal would leave); CMS's procedure
subtracts the mean: in Delta phi the yield's Delta phi average (B is flat in Delta phi: the
factorized result of mix.py), in Delta y the mixed event of the pools (../wake_decomp.py).
Inputs: data/iso/<run>.npz (wake_iso.py map, with cphi_*) and data/mix/<run>/pbpb + the pp
pools (make_data.sh pools).  Errors: the jackknife over hydro backgrounds and pp event blocks.
-> <outdir>/cms_iso_<family>.png (with .npz/.root) and cms_iso.txt."""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ZJET = os.path.dirname(HERE)
sys.path.insert(0, ZJET)
sys.path.insert(0, os.path.dirname(ZJET))                         # jetcorr
import wake_decomp as wd  # noqa: E402  (imports mix: the PyJetscape path)
import wake_iso as wi  # noqa: E402
from cms2507 import PT_LABELS, PT_TAGS, cms  # noqa: E402
from jetcorr import figdata, runs  # noqa: E402

PP_CAMPAIGN = ("pp_campaign", runs.data("pp_comp", "mix", "campaign", "vacuum"))
PP = {"campaign": PP_CAMPAIGN, "mpi_off": ("pp_mpi_off", "{data}/mix/pp_mpi_off/vacuum"),
      "isr_mpi_off": ("pp_isr_mpi_off", "{data}/mix/pp_isr_mpi_off/vacuum")}
#: family -> [(label, run, pp setting)], the first the family's reference
FAMILIES = {
    "tau_delay": [("τ_delay 0.5", "tau_delay_0p5", "campaign"),
                  ("τ_delay 1.0 (campaign)", "tau_delay_1p0", "campaign"),
                  ("τ_delay 2.0", "tau_delay_2p0", "campaign")],
    "e_threshold": [("e_threshold 1", "e_threshold_1p0", "campaign"),
                    ("e_threshold 2 (campaign)", "tau_delay_1p0", "campaign"),
                    ("e_threshold 3", "e_threshold_3p0", "campaign")],
    "pythia_e2": [("ISR + MPI on (campaign)", "tau_delay_1p0", "campaign"),
                  ("MPI off", "mpi_off_e_threshold_2p0", "mpi_off"),
                  ("ISR + MPI off", "isr_mpi_off_e_threshold_2p0", "isr_mpi_off")],
    "pythia_e3": [("ISR + MPI on, e_threshold 3", "e_threshold_3p0", "campaign"),
                  ("MPI off, e_threshold 3", "mpi_off_e_threshold_3p0", "mpi_off"),
                  ("ISR + MPI off, e_threshold 3", "isr_mpi_off_e_threshold_3p0", "isr_mpi_off")],
}
COLS = ("C0", "k", "C3")
#: the maps' p_T classes for CMS's three (wake_iso.PT_EDGES starts at 0.5 GeV)
CLS = (1, 2, 3)
DY_W = 0.8
YMAX = 4.0


# ── the conventions ────────────────────────────────────────────────────────────────
_PC = 0.5 * (wi.CMS_DPHI[1:] + wi.CMS_DPHI[:-1])
_MIRROR = np.array([int(np.argmin(np.abs(_PC - ((-c + np.pi / 2) % (2 * np.pi) - np.pi / 2))))
                    for c in _PC])


def cms_phi(h):
    """Counts per Z in CMS's Delta phi bins (..., N_CPHI) -> +-merged, halved, per radian."""
    return 0.5 * (h + h[..., _MIRROR]) / np.diff(wi.CMS_DPHI)


def mean_subtracted(x):
    """CMS's procedure in Delta phi: minus the Delta phi average (B flat in Delta phi)."""
    return x - np.average(x, axis=-1, weights=np.diff(wi.CMS_DPHI))[..., None]


def cms_dy(m):
    """A Z frame map (N_PT, N_ETA, N_PHI) -> the Z side (|Delta phi| < pi/2) in 0.8 bins over
    |Delta y| < 4, +-merged and halved, per unit Delta y, (N_PT, 10)."""
    z = m[:, :, np.abs(wi.PHI_C) < np.pi / 2].sum(-1)
    keep = np.abs(wi.ETA_C) < YMAX
    k = int(round(DY_W / wi.D_ETA))
    z = z[:, keep].reshape(z.shape[0], -1, k).sum(-1) / DY_W
    return 0.5 * (z + z[:, ::-1])


def isolated(v, pp):
    """The isolated pieces of one run: Delta phi (wake, total), Delta y (wake, total) for
    CMS's classes, flattened."""
    pw, pt = v.mean("cphi_wake"), v.mean("cphi_wake") + v.mean("cphi_frag") - pp.mean("cphi_frag")
    yw = v.mean("map_wake_Z")
    yt = yw + v.mean("map_frag_Z") - pp.mean("map_frag_Z")
    out = [cms_phi(pw)[list(CLS)], cms_phi(pt)[list(CLS)], mean_subtracted(cms_phi(pt))[list(CLS)],
           cms_dy(yw)[list(CLS)], cms_dy(yt)[list(CLS)]]
    return np.concatenate([o.ravel() for o in out])


SHAPES = (("phi_wake", 20), ("phi_total", 20), ("phi_cms", 20), ("dy_wake", 10),
          ("dy_total", 10))


def unpack(x):
    out, i = {}, 0
    for name, n in SHAPES:
        out[name] = x[i:i + 3 * n].reshape(3, n)
        i += 3 * n
    return out


def cms_data(hep):
    """CMS's Pb+Pb 0-30 % - pp: Delta phi (20 bins) and Delta y (pp merged to 0.8)."""
    phi, dy = [], []
    for c in range(len(PT_TAGS)):
        x, yb, sb, syb = cms(hep, "phi", "pbpb", PT_TAGS[c])
        _, yq, sq, syq = cms(hep, "phi", "pp", PT_TAGS[c])
        phi.append((x, yb - yq, np.hypot(np.hypot(sb, syb), np.hypot(sq, syq))))
        dy.append(wd.cms_difference(hep, c, DY_W))
    return phi, dy


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=runs.data("wake_scan"))
    ap.add_argument("--family", nargs="+", default=list(FAMILIES), choices=list(FAMILIES))
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--hepdata", default=runs.hepdata("HEPData-ins2945894-v1-yaml"))
    ap.add_argument("--blocks", type=int, default=100)
    a = ap.parse_args()
    a.outdir = a.outdir or runs.figures("wake_scan")
    cphi, cdy = cms_data(a.hepdata)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    L = ["What CMS would see with the wake isolated (compare_cms_iso.py); per Z; Delta phi per "
         "radian (+-merged x 0.5, all Delta y), Delta y per unit Delta y on the Z side (+-merged "
         "x 0.5); isolated = absolute (no mean subtraction), CMS-proc = the model with CMS's "
         "mean subtraction; values at the first bin past 0, at Delta phi ~ pi (the last bin) and "
         "integrated over 0 < Delta y < 4", ""]
    cache = {}
    paths = []
    for fam in a.family:
        rows = []
        for label, run, ppk in FAMILIES[fam]:
            if (run, ppk) not in cache:
                ppmap, pppool = PP[ppk]
                m = wi.Maps(os.path.join(a.data, "iso", f"{run}.npz"))
                p = wi.Maps(os.path.join(a.data, "iso", f"{ppmap}.npz"))
                if "cphi_wake" not in m.sums:
                    raise SystemExit(f"{run}: no cphi_* (rebuild with wake_iso.py map)")
                val, err = wi.jackknife(lambda V: isolated(V[0], V[1]), [[m], [p]], a.blocks)
                pb = wd.load([os.path.join(a.data, "mix", run, "pbpb")])
                pq = wd.load([pppool.format(data=a.data)])
                dv, de = wd.with_errors(pb, pq, DY_W, a.blocks)
                cache[(run, ppk)] = (unpack(val), unpack(err), dv[wd.PIECES[0]],
                                     de[wd.PIECES[0]], m.n_z)
            rows.append((label, run) + cache[(run, ppk)])
        fig, ax = figdata.subplots(4, 3, figsize=(16.5, 16.0), squeeze=False)
        phi_keep = (_PC > 0) & (_PC < np.pi)
        dyc = -YMAX + DY_W * (np.arange(int(2 * YMAX / DY_W)) + 0.5)
        pos = dyc > 0
        for c in range(3):
            xp, yp, ep = cphi[c]
            hk = (xp > 0) & (xp < np.pi)
            r = cdy[c]
            for r_ in range(4):
                aa = ax[r_, c]
                if r_ < 2:
                    aa.errorbar(xp[hk], yp[hk], ep[hk], fmt="D", color="0.55", ms=5,
                                label="CMS Pb+Pb 0–30 % − pp")
                else:
                    h = r[0] > 0
                    aa.errorbar(r[0][h], r[1][h], r[2][h], fmt="D", color="0.55", ms=5,
                                label="CMS Pb+Pb 0–30 % − pp")
            for i, (label, run, v, e, ydv, yde, nz) in enumerate(rows):
                col = COLS[i % len(COLS)]
                d = 0.03 * (i - 1)
                ax[0, c].errorbar(_PC[phi_keep] + d, v["phi_wake"][c][phi_keep],
                                  e["phi_wake"][c][phi_keep], fmt="o-", ms=3, lw=1.0,
                                  color=col, label=f"{label}: the wake")
                ax[1, c].errorbar(_PC[phi_keep] + d, v["phi_total"][c][phi_keep],
                                  e["phi_total"][c][phi_keep], fmt="o-", ms=3, lw=1.0,
                                  color=col, label=f"{label}: wake + fragments − pp")
                ax[1, c].plot(_PC[phi_keep] + d, v["phi_cms"][c][phi_keep], ":", color=col,
                              lw=1.3, label=f"{label}: with CMS's mean subtraction")
                ax[2, c].errorbar(dyc[pos] + d, v["dy_wake"][c][pos], e["dy_wake"][c][pos],
                                  fmt="o-", ms=3, lw=1.0, color=col, label=f"{label}: the wake")
                ax[3, c].errorbar(dyc[pos] + d, v["dy_total"][c][pos], e["dy_total"][c][pos],
                                  fmt="o-", ms=3, lw=1.0, color=col,
                                  label=f"{label}: wake + fragments − pp")
                ax[3, c].plot(dyc[pos] + d, ydv[c][pos], ":", color=col, lw=1.3,
                              label=f"{label}: with CMS's mean subtraction")
            for r_, (title, xl, yl) in enumerate((
                    ("Δφ: the wake on its own", "Δφ_ch,Z", "dN/dΔφ per Z"),
                    ("Δφ: wake + fragments − pp (solid), with CMS's subtraction (dotted)",
                     "Δφ_ch,Z", "dN/dΔφ per Z"),
                    ("Δy, Z side: the wake on its own", "Δy_ch,Z", "dN/dΔy per Z"),
                    ("Δy, Z side: wake + fragments − pp (solid), with CMS's subtraction "
                     "(dotted)", "Δy_ch,Z", "dN/dΔy per Z"))):
                aa = ax[r_, c]
                aa.axhline(0, color="0.6", lw=0.6)
                aa.set(xlabel=xl, ylabel=yl)
                aa.set_title(f"{title}\n{PT_LABELS[c]}", fontsize=8)
                if c == 0:
                    aa.legend(fontsize=6)
        fig.suptitle(f"What CMS would see if the wake could be isolated: {fam.replace('_', ' ')} "
                     "variations against CMS 2507.09307 (Pb+Pb 0–30 % − pp); the model's "
                     "isolated pieces are absolute, CMS's data mean-subtracted", fontsize=10)
        fig.tight_layout()
        os.makedirs(a.outdir, exist_ok=True)
        p = os.path.join(a.outdir, f"cms_iso_{fam}.png")
        figdata.savefig(fig, p, dpi=85)
        plt.close(fig)
        paths.append(p)
        L.append(f"== {fam}")
        for c in range(3):
            xp, yp, ep = cphi[c]
            hk = (xp > 0) & (xp < np.pi)
            r = cdy[c]
            h = r[0] > 0
            L.append(f"  {PT_LABELS[c]}: CMS dphi first {yp[hk][0]:+.3f}±{ep[hk][0]:.3f}, at pi "
                     f"{yp[hk][-1]:+.3f}±{ep[hk][-1]:.3f}; dy first {r[1][h][0]:+.3f}±"
                     f"{r[2][h][0]:.3f}, integrated {DY_W * r[1][h].sum():+.3f}±"
                     f"{DY_W * np.sqrt((r[2][h] ** 2).sum()):.3f}")
            for label, run, v, e, ydv, yde, nz in rows:
                def f(k, j):
                    return f"{v[k][c][j]:+.3f}±{e[k][c][j]:.3f}"
                ip = np.flatnonzero(phi_keep)
                iy = np.flatnonzero(pos)
                integ = {k: (DY_W * v[k][c][pos].sum(), DY_W * np.sqrt((e[k][c][pos] ** 2).sum()))
                         for k in ("dy_wake", "dy_total")}
                L.append(f"    {label:30s} dphi first: wake {f('phi_wake', ip[0])} total "
                         f"{f('phi_total', ip[0])} CMS-proc {v['phi_cms'][c][ip[0]]:+.3f} | at pi: "
                         f"wake {f('phi_wake', ip[-1])} total {f('phi_total', ip[-1])} CMS-proc "
                         f"{v['phi_cms'][c][ip[-1]]:+.3f} | dy first: wake {f('dy_wake', iy[0])} "
                         f"total {f('dy_total', iy[0])} CMS-proc {ydv[c][iy[0]]:+.3f}±"
                         f"{yde[c][iy[0]]:.3f} | dy integrated: wake {integ['dy_wake'][0]:+.3f}±"
                         f"{integ['dy_wake'][1]:.3f} total {integ['dy_total'][0]:+.3f}±"
                         f"{integ['dy_total'][1]:.3f} CMS-proc {DY_W * ydv[c][pos].sum():+.3f}")
        L.append("")
    txt = os.path.join(a.outdir, "cms_iso.txt")
    with open(txt, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    for p in paths + [txt]:
        print(f"-> {p}")


if __name__ == "__main__":
    main()
