#!/usr/bin/env python3
"""The scan's wake on its own (docs/Plans/PLAN_wake_isolated.md, step 1): for every value of the
scanned setting the paired wake <bulk_jet> - <bulk_bg> and the fragments in (Delta eta, Delta
phi) around the Z and around the recoiling parton (../wake_iso.py), and every value against the
baseline, background by background.  No mixed event, no mean subtraction.

    python compare_iso.py --data runs/wake_scan/data --param Liquefier/CausalLiquefier/tau_delay \\
        --values 0.5 1.0 2.0 --base 1.0 [--base-name tau_delay_1p0] [--pp <pp map .npz>]

Inputs (make_data.sh iso): data/iso/<name>.npz (wake_iso.py map, <name> = <setting>_<value with
. -> p>) and the pp's (data/iso/pp_campaign.npz).  The values' jobs have the same seeds, so the
same Z's and backgrounds (refused otherwise): a value minus the baseline is formed per
background, with a jackknife over the same background blocks.  The pp is independent: its
jackknife over blocks of its events adds in quadrature.

-> <outdir>/wake_iso_maps_{Z,J}.png (the baseline: the wake, the fragments, the fragments'
   Pb+Pb - pp), wake_iso_{proj,diff,frag,harm,harmdiff,spectra,diffmap}_<setting>.png (with
   .npz/.root) and wake_iso_<setting>.txt."""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ZJET = os.path.dirname(HERE)
sys.path.insert(0, ZJET)
sys.path.insert(0, os.path.dirname(ZJET))                         # jetcorr
import wake_iso as wi  # noqa: E402
from jetcorr import figdata, runs  # noqa: E402

AREA = wi.D_ETA * wi.D_PHI
J_ETA = 1.4                      # |d_eta| fully inside the track acceptance in the J frame


def name_of(param, value):
    return f"{param.split('/')[-1]}_{value.replace('.', 'p')}"


# ── the numbers of one run (a View) ──────────────────────────────────────────────────
def numbers(v, content):
    """The scalars of PLAN_wake_isolated.md §1 for one content -> dict name -> array."""
    out = {}
    hz = wi.harmonics(v, content, "Z")
    hj = wi.harmonics(v, content, "J", J_ETA)
    hzj = wi.harmonics(v, content, "ZJ", J_ETA)
    for n in range(wi.NH):
        out[f"c{n} Z"] = hz[:, n]
        out[f"c{n} ZJ"] = hzj[:, n]
        out[f"c{n} J"] = hj[:, n]
    for frame in ("Z", "J"):
        d, rho = wi.dip(v, content, frame)
        out[f"dip {frame}"] = d
        out[f"rho_in {frame}"] = rho[:, 0]
        out[f"rho_out {frame}"] = rho[:, 1]
    cw = wi.core_wings(v, content)
    out["core J"], out["wings J"] = cw[:, 0], cw[:, 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        out["v1 Z"] = hz[:, 1] / hz[:, 0]
        out["v1 J"] = hj[:, 1] / hj[:, 0]
    tot = v.mean(f"tot_{content}")
    for k, name in enumerate(wi.TOTALS):
        out[f"total {name}"] = np.array([tot[k]])
    if content == "wake":
        out["E_droplets"] = np.array([v.mean("edrop")])
    return out


def per_class_keys():
    keys = []
    for n in range(wi.NH):
        keys += [f"c{n} Z", f"c{n} ZJ", f"c{n} J"]
    return keys + ["v1 Z", "v1 J", "rho_in Z", "rho_out Z", "dip Z", "rho_in J", "rho_out J",
                   "dip J", "core J", "wings J"]


def flat(d):
    keys = sorted(d)
    return np.concatenate([np.atleast_1d(d[k]) for k in keys]), keys


def unflat(x, template):
    out, i = {}, 0
    for k in sorted(template):
        n = np.atleast_1d(template[k]).size
        out[k] = x[i:i + n]
        i += n
    return out


def jk_numbers(fn_dict, sets, n_blocks):
    """A dict-valued quantity through the jackknife -> (values, errors) dicts."""
    template = fn_dict([wi.View(m) for s in sets for m in s])
    val, err = wi.jackknife(lambda V: flat(fn_dict(V))[0], sets, n_blocks)
    return unflat(val, template), unflat(err, template)


def toward_z_share(spec):
    """The share of a (2, n) spectrum toward the Z, bin by bin (NaN where it is empty)."""
    tot = spec.sum(0)
    return spec[0] / np.where(tot != 0, tot, np.nan)


# ── figures ─────────────────────────────────────────────────────────────────────────
def zframe_axes(a, frame):
    a.axhline(np.pi, color="k", ls="--", lw=0.6)
    a.axhline(0.0, color="k", ls=":", lw=0.6)
    if frame == "J":
        for x in (-J_ETA, J_ETA):
            a.axvline(x, color="0.4", ls=":", lw=0.6)
        a.set_xlim(-3.4, 3.4)
        a.set(xlabel="Δη = η − η_rec", ylabel="Δφ = φ − φ_rec (Z near π)")
    else:
        a.set(xlabel="Δη = η − y_Z", ylabel="Δφ = φ − φ_Z (recoil near π)")


def merged(m, k):
    """A map (N_PT, N_ETA, N_PHI) of counts per Z with k x k bins merged -> density per Z per
    unit d_eta per rad."""
    x = m.reshape(m.shape[0], m.shape[1] // k, k, m.shape[2] // k, k)
    return x.sum((2, 4)) / (k * k * AREA)


def rebin1(x, y, k):
    """Centres and per-unit densities of a projection (..., n) with k bins merged."""
    return x.reshape(-1, k).mean(1), y.reshape(y.shape[:-1] + (-1, k)).mean(-1)


def draw_maps(base, pp, frame, path, base_label):
    import matplotlib.pyplot as plt

    vb, vp = wi.View(base), wi.View(pp)
    rows = [("wake (paired bulk_jet − bulk_bg)", vb.mean(f"map_wake_{frame}"), True),
            ("fragments, Pb+Pb", vb.mean(f"map_frag_{frame}"), False),
            ("fragments, Pb+Pb − pp", vb.mean(f"map_frag_{frame}") - vp.mean(f"map_frag_{frame}"),
             True)]
    fig, ax = figdata.subplots(len(rows), wi.N_PT, figsize=(5.0 * wi.N_PT, 3.9 * len(rows)),
                               squeeze=False)
    for i, (label, m, signed) in enumerate(rows):
        for c in range(wi.N_PT):
            a = ax[i, c]
            dens = merged(m, 2)[c]
            if frame == "J":
                show = np.abs(wi.ETA_C[::2] + wi.D_ETA / 2) < 3.4
                lim = np.nanpercentile(np.abs(dens[show]), 99) or 1.0
            else:
                lim = np.nanpercentile(np.abs(dens), 99) or 1.0
            pm = a.pcolormesh(wi.DETA[::2], wi.DPHI[::2], dens.T,
                              cmap="RdBu_r" if signed else "viridis",
                              vmin=-lim if signed else 0.0, vmax=lim)
            fig.colorbar(pm, ax=a, label="dN/dΔη dΔφ per Z")
            zframe_axes(a, frame)
            a.set_title(f"{label}\n{wi.PT_LABELS[c]}", fontsize=9)
    n = int(base.N.sum() if frame == "Z" else base.NJ.sum())
    what = ("around the Z, all selected Z's" if frame == "Z" else
            f"around the recoil, the Z's with |η_rec| < {base.meta['rec_eta']:g}")
    fig.suptitle(f"{base_label}: the wake and the fragments on their own, {what} ({n} Pb+Pb "
                 f"Z's; pp {int(pp.N.sum() if frame == 'Z' else pp.NJ.sum())}); bins 0.4 × π/8",
                 fontsize=10)
    fig.tight_layout()
    figdata.savefig(fig, path, dpi=90)
    plt.close(fig)


PROJ = (("Z frame: Δφ, all Δη", "Δφ − φ_Z", "dN/dΔφ per Z"),
        ("Z frame: Δη toward the Z (|Δφ| < π/2)", "Δη = η − y_Z", "dN/dΔη per Z"),
        (f"J frame: Δφ, |Δη| < {J_ETA:g}", "Δφ − φ_rec", "dN/dΔφ per Z"),
        ("J frame: Δη toward the recoil (|Δφ| < π/2)", "Δη = η − η_rec", "dN/dΔη per Z"))


def projections(v, content, k=1):
    """The four projections of PROJ, k bins merged -> list of (x, y (N_PT, n))."""
    return [rebin1(wi.PHI_C, wi.phi_proj(v, content, "Z"), k),
            rebin1(wi.ETA_C, wi.eta_proj(v, content, "Z", "z"), k),
            rebin1(wi.PHI_C, wi.phi_proj(v, content, "J", J_ETA), k),
            rebin1(wi.ETA_C, wi.eta_proj(v, content, "J", "rec"), k)]


def draw_proj(curves, path, title, ref=None, zero=False):
    """curves: list of (label, colour, [(x, value, error) per PROJ row], style)."""
    import matplotlib.pyplot as plt

    fig, ax = figdata.subplots(len(PROJ), wi.N_PT, figsize=(4.6 * wi.N_PT, 3.3 * len(PROJ)),
                               squeeze=False)
    for r, (rt, xl, yl) in enumerate(PROJ):
        for c in range(wi.N_PT):
            a = ax[r, c]
            for i, (label, col, rows, fmt) in enumerate(curves):
                x, y, e = rows[r]
                keep = np.ones(len(x), bool) if r != 3 else np.abs(x) < 3.4
                a.errorbar(x[keep] + 0.012 * i, y[c][keep], e[c][keep], fmt=fmt, color=col,
                           ms=3, lw=0.9, label=label)
            if r == 3:
                for xx in (-J_ETA, J_ETA):
                    a.axvline(xx, color="0.5", ls=":", lw=0.6)
            if r in (0, 2):
                a.axvline(np.pi, color="0.5", ls="--", lw=0.6)
            a.axhline(0, color="0.6", lw=0.6)
            a.set_title(f"{rt}\n{wi.PT_LABELS[c]}", fontsize=8)
            a.set(xlabel=xl, ylabel=yl)
            if r == 0 and c == 0:
                a.legend(fontsize=6)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    figdata.savefig(fig, path, dpi=90)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=runs.data("wake_scan"))
    ap.add_argument("--param", default="Liquefier/CausalLiquefier/tau_delay")
    ap.add_argument("--values", nargs="+", default=["0.5", "1.0", "2.0"])
    ap.add_argument("--base", default="1.0")
    ap.add_argument("--base-name", default=None,
                    help="the baseline's run name if not <setting>_<value> (make_data.sh REF)")
    ap.add_argument("--pp", default=None, help="the pp map (default <data>/iso/pp_campaign.npz)")
    ap.add_argument("--outdir", default=None, help="default runs/wake_scan/figures")
    ap.add_argument("--blocks", type=int, default=100)
    a = ap.parse_args()
    a.outdir = a.outdir or runs.figures("wake_scan")
    a.pp = a.pp or os.path.join(a.data, "iso", "pp_campaign.npz")
    if a.base not in a.values:
        raise SystemExit("--base must be one of --values")
    tag = a.param.split("/")[-1]
    names = {v: (a.base_name if (v == a.base and a.base_name) else name_of(a.param, v))
             for v in a.values}
    maps = {v: wi.Maps(os.path.join(a.data, "iso", f"{names[v]}.npz")) for v in a.values}
    pp = wi.Maps(a.pp)
    base = maps[a.base]
    for v, m in maps.items():
        if not m.same_events(base):
            raise SystemExit(f"{tag} = {v}: not the same Z's and backgrounds as {a.base}")
    others = [v for v in a.values if v != a.base]
    os.makedirs(a.outdir, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = lambda s: os.path.join(a.outdir, s)                     # noqa: E731
    paths = []
    for frame in ("Z", "J"):
        p = out(f"wake_iso_maps_{frame}.png")
        draw_maps(base, pp, frame, p, f"{tag} = {a.base} (the campaign's)")
        paths.append(p)

    # ── the projections, per value and as paired differences ──
    cols = {v: f"C{i}" for i, v in enumerate(a.values)}

    def proj_err(fn, sets):
        template = fn([wi.View(m) for s in sets for m in s])
        val, err = wi.jackknife(lambda V: np.concatenate([y.ravel() for _, y in fn(V)]), sets,
                                a.blocks)
        res, i = [], 0
        for x, y in template:
            n = y.size
            res.append((x, val[i:i + n].reshape(y.shape), err[i:i + n].reshape(y.shape)))
            i += n
        return res

    wake = {v: proj_err(lambda V: projections(V[0], "wake", 2), [[maps[v]]])
            for v in a.values}
    curves = [(f"{tag} = {v}", cols[v], wake[v], "o-" if v == a.base else "s--")
              for v in a.values]
    p = out(f"wake_iso_proj_{tag}.png")
    draw_proj(curves, p, f"The wake on its own (paired bulk_jet − bulk_bg) per Z, every value of "
                         f"{tag}; {base.n_z} Z's on {len(base.groups)} backgrounds (J frame: "
                         f"{int(base.NJ.sum())} with |η_rec| < {base.meta['rec_eta']:g}); bins "
                         "π/8, 0.4")
    paths.append(p)
    diff = {v: proj_err(lambda V: [(x, y0 - y1) for (x, y0), (_, y1) in
                                   zip(projections(V[0], "wake", 4),
                                       projections(V[1], "wake", 4))],
                        [[maps[v], base]]) for v in others}
    fdiff = {v: proj_err(lambda V: [(x, y0 - y1) for (x, y0), (_, y1) in
                                    zip(projections(V[0], "frag", 4),
                                        projections(V[1], "frag", 4))],
                         [[maps[v], base]]) for v in others}
    curves = []
    for v in others:
        curves.append((f"wake, {v} − {a.base}", cols[v], diff[v], "o-"))
        curves.append((f"fragments, {v} − {a.base}", cols[v], fdiff[v], "x:"))
    p = out(f"wake_iso_diff_{tag}.png")
    draw_proj(curves, p, f"Paired differences to {tag} = {a.base}, background by background: "
                         "the wake (solid) and the fragments (dotted; for a setting acting after "
                         "the energy loss only their sampling noise); bins π/4, 0.8")
    paths.append(p)
    frag = {v: proj_err(lambda V: projections(V[0], "frag", 2), [[maps[v]]])
            for v in a.values}
    ppf = proj_err(lambda V: projections(V[0], "frag", 2), [[pp]])
    curves = [(f"Pb+Pb, {tag} = {v}", cols[v], frag[v], "o-" if v == a.base else "s--")
              for v in a.values] + [("pp", "k", ppf, "-")]
    p = out(f"wake_iso_frag_{tag}.png")
    draw_proj(curves, p, f"The fragments on their own per Z (jet_frag without the Z's leptons), "
                         f"every value of {tag}, and pp")
    paths.append(p)

    # ── the numbers ──
    num = {v: {c: jk_numbers(lambda V, c=c: numbers(V[0], c), [[maps[v]]], a.blocks)
               for c in ("wake", "frag")} for v in a.values}
    num_pp = jk_numbers(lambda V: numbers(V[0], "frag"), [[pp]], a.blocks)
    def minus(x, y):
        return {k: x[k] - y[k] for k in x if k in y}

    dnum = {v: {c: jk_numbers(lambda V, c=c: minus(numbers(V[0], c), numbers(V[1], c)),
                              [[maps[v], base]], a.blocks)
                for c in ("wake", "frag")} for v in others}
    pbpp = {v: jk_numbers(lambda V: minus(numbers(V[0], "frag"), numbers(V[1], "frag")),
                          [[maps[v]], [pp]], a.blocks) for v in a.values}

    def budget(V):
        """(wake + fragments) Pb+Pb - fragments pp, the totals over every stored hadron."""
        t = V[0].mean("tot_wake") + V[0].mean("tot_frag") - V[1].mean("tot_frag")
        return t

    bud = {v: wi.jackknife(budget, [[maps[v]], [pp]], a.blocks) for v in a.values}

    # ── harmonics figure ──
    fig, ax = figdata.subplots(3, wi.NH, figsize=(3.8 * wi.NH, 9.5), squeeze=False)
    xs = np.arange(wi.N_PT)
    for r, frame in enumerate(("Z", "ZJ", "J")):
        for n in range(wi.NH):
            aa = ax[r, n]
            for i, v in enumerate(a.values):
                val, err = num[v]["wake"]
                aa.errorbar(xs + 0.08 * i, val[f"c{n} {frame}"], err[f"c{n} {frame}"], fmt="o",
                            color=cols[v], ms=4, label=f"{tag} = {v}")
            aa.axhline(0, color="0.6", lw=0.6)
            aa.set_xticks(xs)
            aa.set_xticklabels([f"{x:g}–{y:g}" for x, y in zip(wi.PT_EDGES[:-1],
                                                              wi.PT_EDGES[1:])], fontsize=7)
            where = {"Z": "Z frame, all Δη", "ZJ": f"Z frame, |Δη| < {J_ETA:g}, the J Z's",
                     "J": f"J frame, |Δη| < {J_ETA:g}"}[frame]
            aa.set_title(f"c_{n} = Σ cos({n}Δφ) per Z\n{where}", fontsize=8)
            aa.set_xlabel("track p_T [GeV]")
            if n == 0 and r == 0:
                aa.legend(fontsize=6)
    fig.suptitle(f"The wake's Δφ harmonics per Z (a part flat in Δφ enters c_0 only), every value "
                 f"of {tag}", fontsize=10)
    fig.tight_layout()
    p = out(f"wake_iso_harm_{tag}.png")
    figdata.savefig(fig, p, dpi=90)
    plt.close(fig)
    paths.append(p)

    # ── the harmonics' paired differences ──
    hkeys = [f"c{n}" for n in range(wi.NH)] + ["v1"]
    fig, ax = figdata.subplots(2, len(hkeys), figsize=(3.6 * len(hkeys), 6.6), squeeze=False)
    for r, frame in enumerate(("Z", "J")):
        for j, h in enumerate(hkeys):
            aa = ax[r, j]
            for i, v in enumerate(others):
                val, err = dnum[v]["wake"]
                aa.errorbar(xs + 0.1 * i, val[f"{h} {frame}"], err[f"{h} {frame}"], fmt="o",
                            color=cols[v], ms=4, label=f"{v} − {a.base}")
            aa.axhline(0, color="0.6", lw=0.6)
            aa.set_xticks(xs)
            aa.set_xticklabels([f"{x:g}–{y:g}" for x, y in zip(wi.PT_EDGES[:-1],
                                                              wi.PT_EDGES[1:])], fontsize=7)
            what = "v_1 = c_1 / c_0" if h == "v1" else f"c_{h[1]}"
            where = "Z frame, all Δη" if frame == "Z" else f"J frame, |Δη| < {J_ETA:g}"
            aa.set_title(f"Δ {what} (paired)\n{where}", fontsize=8)
            aa.set_xlabel("track p_T [GeV]")
            if j == 0 and r == 0:
                aa.legend(fontsize=6)
    fig.suptitle(f"The wake's harmonics, each value minus {tag} = {a.base}, background by "
                 "background", fontsize=10)
    fig.tight_layout()
    p = out(f"wake_iso_harmdiff_{tag}.png")
    figdata.savefig(fig, p, dpi=90)
    plt.close(fig)
    paths.append(p)

    # ── spectra ──
    spec = {v: wi.jackknife(lambda V: wi.spectra(V[0], "wake"), [[maps[v]]], a.blocks)
            for v in a.values}
    dspec = {v: wi.jackknife(lambda V: wi.spectra(V[0], "wake").sum(0)
                             - wi.spectra(V[1], "wake").sum(0), [[maps[v], base]], a.blocks)
             for v in others}
    centres = 0.5 * (wi.SPEC_EDGES[1:] + wi.SPEC_EDGES[:-1])
    low = centres < 5.0
    fig, ax = figdata.subplots(1, 4, figsize=(20, 4.6), squeeze=False)
    for i, v in enumerate(a.values):
        val, err = spec[v]
        for s_, lab in ((0, "toward the Z (|Δφ_Z| < π/2)"), (1, "toward the recoil")):
            ax[0, s_].errorbar(centres[low] + 0.02 * i, val[s_][low], err[s_][low], fmt="o-",
                               ms=3, color=cols[v], lw=0.8, label=f"{tag} = {v}")
            ax[0, s_].set_title(f"wake p_T spectrum, {lab}", fontsize=9)
        share, se = wi.jackknife(lambda V: toward_z_share(wi.spectra(V[0], "wake")),
                                 [[maps[v]]], a.blocks)
        ax[0, 2].errorbar(centres[low] + 0.02 * i, share[low], se[low], fmt="o-", ms=3,
                          color=cols[v], lw=0.8, label=f"{tag} = {v}")
    for i, v in enumerate(others):
        dv, de = dspec[v]
        ax[0, 3].errorbar(centres[low] + 0.02 * i, dv[low], de[low], fmt="o-", ms=3,
                          color=cols[v], lw=0.8, label=f"{v} − {a.base}")
    for s_ in (0, 1):
        ax[0, s_].set(xlabel="p_T [GeV]", ylabel="dN/dp_T per Z [1/GeV]")
        ax[0, s_].axhline(0, color="0.6", lw=0.6)
    ax[0, 2].axhline(0.5, color="0.6", lw=0.6, ls="--")
    ax[0, 2].set(xlabel="p_T [GeV]", ylabel="toward Z / all",
                 title="the wake's share toward the Z (0.5: flat in Δφ)")
    ax[0, 2].set_ylim(-0.2, 1.0)
    ax[0, 3].axhline(0, color="0.6", lw=0.6)
    ax[0, 3].set(xlabel="p_T [GeV]", ylabel="Δ dN/dp_T per Z [1/GeV]",
                 title="the wake's spectrum (all Δφ), paired difference")
    ax[0, 0].legend(fontsize=7)
    ax[0, 3].legend(fontsize=7)
    fig.suptitle(f"The wake's charged p_T spectra (|η| < 2.4), every value of {tag}", fontsize=10)
    fig.tight_layout()
    p = out(f"wake_iso_spectra_{tag}.png")
    figdata.savefig(fig, p, dpi=90)
    plt.close(fig)
    paths.append(p)

    # ── 2-d paired differences (bins merged 4 x 4) ──
    de2, dp2 = wi.DETA[::4], wi.DPHI[::4]
    rows = [(v, frame) for v in others for frame in ("Z", "J")]
    n_pt = 3                                                     # 0.5-1, 1-2, 2-4 GeV
    fig, ax = figdata.subplots(len(rows), 2 * n_pt, figsize=(4.0 * 2 * n_pt, 3.4 * len(rows)),
                               squeeze=False)
    for r, (v, frame) in enumerate(rows):
        dv, de = wi.jackknife(lambda V, f=frame: merged(V[0].mean(f"map_wake_{f}"), 4)
                              - merged(V[1].mean(f"map_wake_{f}"), 4), [[maps[v], base]],
                              a.blocks)
        for c in range(n_pt):
            for j, (arr, lab, cm, lim) in enumerate((
                    (dv[c], "difference", "RdBu_r", None),
                    (np.where(de[c] > 0, dv[c] / np.where(de[c] > 0, de[c], 1), 0.0),
                     "difference / error", "PuOr_r", 3.0))):
                aa = ax[r, 2 * c + j]
                lim = lim or (np.nanpercentile(np.abs(arr), 99) or 1.0)
                pm = aa.pcolormesh(de2, dp2, arr.T, cmap=cm, vmin=-lim, vmax=lim)
                fig.colorbar(pm, ax=aa)
                zframe_axes(aa, frame)
                aa.set_title(f"wake {v} − {a.base}, {lab}, {frame} frame\n{wi.PT_LABELS[c]}",
                             fontsize=8)
    fig.suptitle(f"The wake's paired differences to {tag} = {a.base} in (Δη, Δφ), bins 0.8 × π/4: "
                 "the difference per Z per unit Δη per rad and its significance", fontsize=10)
    fig.tight_layout()
    p = out(f"wake_iso_diffmap_{tag}.png")
    figdata.savefig(fig, p, dpi=80)
    plt.close(fig)
    paths.append(p)

    # ── the text ──
    def fmt(val, err, k, c=None):
        x = val[k] if c is None else val[k][c]
        e = err[k] if c is None else err[k][c]
        return f"{float(np.squeeze(x)):+.3f}±{float(np.squeeze(e)):.3f}"

    lines = [f"Wake on its own (wake_iso.py), scan of {a.param}: {' '.join(a.values)} (baseline "
             f"{a.base}, run {names[a.base]}); {base.n_z} Pb+Pb Z's on {len(base.groups)} hydro "
             f"backgrounds (N_eff {base.n_eff:.0f}), the same in every value; J frame "
             f"{int(base.NJ.sum())} Z's with |eta_rec| < {base.meta['rec_eta']:g}; pp "
             f"{pp.n_z} Z's ({int(pp.NJ.sum())} in J) from {pp.meta['sources'][0]}",
             "Per Z.  c_n = sum cos(n dphi) over charged tracks |eta| < 2.4 (Z frame: all deta; "
             f"ZJ, J: |deta| < {J_ETA:g}); rho_in/out: density toward the Z at |dphi - phi_Z| < "
             "pi/8 / pi/4..pi/2, |deta| < 1, per rad per unit deta, dip = in - out; core/wings: J "
             "frame toward the recoil (|dphi| < pi/2), |deta| < 0.4 / 0.4..1.4, per unit deta; "
             "totals: every stored hadron (|eta| < 2.6, all species, any p_T).  Errors: jackknife "
             "over hydro backgrounds (8: about 25 % on the errors) and pp event blocks.", ""]
    keys = per_class_keys()
    sections = [(f"wake, {tag} = {v}", num[v]["wake"]) for v in a.values]
    sections += [(f"fragments Pb+Pb, {tag} = {v}", num[v]["frag"]) for v in a.values]
    sections += [("fragments pp", num_pp)]
    sections += [(f"fragments Pb+Pb - pp, {tag} = {v}", pbpp[v]) for v in a.values]
    sections += [(f"wake, {v} - {a.base} (paired)", dnum[v]["wake"]) for v in others]
    sections += [(f"fragments, {v} - {a.base} (paired)", dnum[v]["frag"]) for v in others]
    for title, (val, err) in sections:
        lines.append(title)
        for c in range(wi.N_PT):
            lines.append(f"  {wi.PT_LABELS[c]:22s} " + " | ".join(
                f"{k} {fmt(val, err, k, c)}" for k in keys if k in val))
        lines.append("  totals: " + " | ".join(
            f"{k[6:]} {fmt(val, err, k)}" for k in sorted(val) if k.startswith("total ")) +
            (f" | E_droplets {fmt(val, err, 'E_droplets')}" if "E_droplets" in val else ""))
        lines.append("")
    lines.append("Budget: (wake + fragments) Pb+Pb - fragments pp, every stored hadron "
                 "(|eta| < 2.6), per Z:")
    for v in a.values:
        bv, be = bud[v]
        lines.append(f"  {tag} = {v:5s} " + " | ".join(
            f"{name} {bv[k]:+.2f}±{be[k]:.2f}" for k, name in enumerate(wi.TOTALS)))
    lines.append("")
    lines.append("The wake's energy in |eta| < 2.6 over the deposited droplet energy:")
    for v in a.values:
        f_, fe = wi.jackknife(lambda V: np.array([V[0].mean("tot_wake")[1]
                                                  / V[0].mean("edrop")]), [[maps[v]]],
                              a.blocks)
        lines.append(f"  {tag} = {v:5s} {f_[0]:.3f}±{fe[0]:.3f}")
    txt = out(f"wake_iso_{tag}.txt")
    with open(txt, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    for p in paths + [txt]:
        print(f"-> {p}")


if __name__ == "__main__":
    main()
