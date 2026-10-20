#!/usr/bin/env python3
"""The wake on its own under other Pythia settings (docs/Plans/PLAN_wake_isolated.md): the
campaign's (ISR and MPI on), MPI off, ISR and MPI off, each at e_threshold 2 and 3 GeV, from the
maps of ../wake_iso.py.  Without MPI and ISR the only parton the energy loss gets is the Z's
recoil, so the comparison tells which part of the wake (and of its part toward the Z) the other
partons make.

    python compare_pythia.py [--data runs/wake_scan/data] \\
        [--setting NAME:E2_RUN:E3_RUN:PP_RUN:LABEL ...]     # default: the three below

Inputs: data/iso/<run>.npz (make_data.sh iso, VARIANT=mpi_off / isr_mpi_off).  The settings'
jobs have the same seeds and so the same hydro backgrounds (check_pairing.py --level
backgrounds), but other hard events: two settings are compared per Z, with a jackknife over the
same background blocks (the backgrounds are the independent units of both), not event by event.
e_threshold 3 against 2 within one setting is paired event by event.  Each setting has its own
pp reference (the same Pythia settings in vacuum).
-> <outdir>/pythia_{kin,proj,harm,e3,frag}.png (with .npz/.root) and pythia_compare.txt."""
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

DEFAULT = ("campaign:tau_delay_1p0:e_threshold_3p0:pp_campaign:ISR + MPI on (campaign)",
           "mpi_off:mpi_off_e_threshold_2p0:mpi_off_e_threshold_3p0:pp_mpi_off:MPI off",
           "isr_mpi_off:isr_mpi_off_e_threshold_2p0:isr_mpi_off_e_threshold_3p0:pp_isr_mpi_off:"
           "ISR + MPI off")
N_CLS = 3                                         # 0.5-1, 1-2, 2-4 GeV (4-10: no wake)
COLS = ("k", "C0", "C3", "C2", "C1")


def zshare(v, content, frame):
    """The share of a content toward the Z (Z frame: |d_phi| < pi/2; J: |d_phi - pi| < pi/2),
    per track p_T class."""
    m = v.mean(f"map_{content}_{frame}")
    zs = wi.zside_phi(frame)
    return m[:, :, zs].sum((1, 2)) / m.sum((1, 2))


def zyield(v, content, frame):
    """A content's yield per Z on the Z's half (Z frame: |d_phi| < pi/2; J: |d_phi - pi| <
    pi/2), per track p_T class."""
    return v.mean(f"map_{content}_{frame}")[:, :, wi.zside_phi(frame)].sum((1, 2))


def numbers(v, content):
    """compare_iso's numbers, the share and the yield toward the Z, and the CMS-like
    Delta-y-integrated Z side (2/pi)(c_1 - c_3/3): half the Z-side minus away-side yield."""
    out = ci.numbers(v, content)
    for frame in ("Z", "J"):
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"zshare {frame}"] = zshare(v, content, frame)
        out[f"zyield {frame}"] = zyield(v, content, frame)
    out["cms Z"] = 2 / np.pi * (out["c1 Z"] - out["c3 Z"] / 3)
    return out


def minus(x, y):
    return {k: x[k] - y[k] for k in x if k in y}


def sets_of(*maps):
    """One paired set if the runs share their groups (hydro backgrounds), else one set each."""
    if all(np.array_equal(m.groups, maps[0].groups) for m in maps[1:]):
        return [list(maps)]
    return [[m] for m in maps]


def flat_keys(d):
    return [k for k in sorted(d)]


def kinematics(m):
    """The selected Z's weighted means: p_T^Z, the recoil's p_T, the deposited droplet energy,
    and the weighted distributions."""
    e = m.events
    w = e["w"] / e["w"].sum()
    return {"z_pt": float(np.sum(w * e["z_pt"])), "rec_pt": float(np.sum(w * e["rec_pt"])),
            "e_drop": float(np.sum(w * e["e_drop"])), "n": len(w),
            "n_eff": float(1.0 / np.sum(w * w)), "in_j": float(np.mean(e["in_j"]))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=runs.data("wake_scan"))
    ap.add_argument("--setting", action="append", default=None,
                    help="NAME:E2_RUN:E3_RUN:PP_RUN:LABEL (data/iso/<run>.npz); the first is "
                         "the reference")
    ap.add_argument("--outdir", default=None, help="default runs/wake_scan/figures")
    ap.add_argument("--blocks", type=int, default=100)
    a = ap.parse_args()
    a.outdir = a.outdir or runs.figures("wake_scan")
    specs = [s.split(":", 4) for s in (a.setting or DEFAULT)]
    load = lambda n: wi.Maps(os.path.join(a.data, "iso", f"{n}.npz"))     # noqa: E731
    S = [{"name": n, "e2": load(e2), "e3": load(e3), "pp": load(pp), "label": lab,
          "runs": (e2, e3, pp)} for n, e2, e3, pp, lab in specs]
    for s in S:
        if not s["e3"].same_events(s["e2"]):
            raise SystemExit(f"{s['name']}: e_threshold 3 and 2 are not the same events")
    ref = S[0]
    jk = lambda fn, sets: ci.jk_numbers(fn, sets, a.blocks)                # noqa: E731
    # ── the numbers ──
    R = {}
    for s in S:
        r = R[s["name"]] = {}
        for e in ("e2", "e3"):
            r[e] = {c: jk(lambda V, c=c: numbers(V[0], c), [[s[e]]]) for c in ("wake", "frag")}
            r[e]["frag-pp"] = jk(lambda V: minus(numbers(V[0], "frag"), numbers(V[1], "frag")),
                                 [[s[e]], [s["pp"]]])
            r[e]["budget"] = wi.jackknife(lambda V: V[0].mean("tot_wake") + V[0].mean("tot_frag")
                                          - V[1].mean("tot_frag"), [[s[e]], [s["pp"]]],
                                          a.blocks)
            r[e]["kin"] = kinematics(s[e])
        r["pp"] = jk(lambda V: numbers(V[0], "frag"), [[s["pp"]]])
        r["inc"] = {c: jk(lambda V, c=c: minus(numbers(V[0], c), numbers(V[1], c)),
                          [[s["e3"], s["e2"]]]) for c in ("wake", "frag")}
        if s is not ref:
            r["vs_ref"] = {c: jk(lambda V, c=c: minus(numbers(V[0], c), numbers(V[1], c)),
                                 sets_of(s["e2"], ref["e2"])) for c in ("wake",)}
            r["inc_vs_ref"] = jk(lambda V: minus(minus(numbers(V[0], "wake"),
                                                       numbers(V[1], "wake")),
                                                 minus(numbers(V[2], "wake"),
                                                       numbers(V[3], "wake"))),
                                 sets_of(s["e3"], s["e2"], ref["e3"], ref["e2"]))
            r["paired_bg"] = len(sets_of(s["e2"], ref["e2"])) == 1

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = lambda x: os.path.join(a.outdir, x)                              # noqa: E731
    os.makedirs(a.outdir, exist_ok=True)
    paths = []
    col = {s["name"]: COLS[i] for i, s in enumerate(S)}

    # ── kinematics ──
    fig, ax = figdata.subplots(1, 3, figsize=(16, 4.4), squeeze=False)
    for s in S:
        e = s["e2"].events
        for j, (k, edges, xl) in enumerate((("z_pt", np.linspace(40, 160, 25), "p_T^Z [GeV]"),
                                            ("rec_pt", np.linspace(0, 160, 33),
                                             "recoil parton p_T [GeV]"),
                                            ("e_drop", np.linspace(-50, 600, 27),
                                             "deposited droplet energy per event [GeV]"))):
            ax[0, j].hist(e[k], edges, weights=e["w"] / e["w"].sum(), histtype="step", lw=1.5,
                          color=col[s["name"]], label=s["label"])
            ax[0, j].set(xlabel=xl, ylabel="fraction of Z's (weighted)")
    ax[0, 0].legend(fontsize=7)
    fig.suptitle("The selected Z's per Pythia setting (e_threshold 2): p_T^Z, the recoil, the "
                 "energy the fluid gets", fontsize=10)
    fig.tight_layout()
    p = out("pythia_kin.png")
    figdata.savefig(fig, p, dpi=90)
    plt.close(fig)
    paths.append(p)

    # ── projections of the wake at e_threshold 2 ──
    curves = []
    for i, s in enumerate(S):
        t = s["e2"]
        tmpl = ci.projections(wi.View(t), "wake", 2)
        val, err = wi.jackknife(lambda V: np.concatenate(
            [y.ravel() for _, y in ci.projections(V[0], "wake", 2)]), [[t]], a.blocks)
        rows, j = [], 0
        for x, y in tmpl:
            n = y.size
            rows.append((x, val[j:j + n].reshape(y.shape), err[j:j + n].reshape(y.shape)))
            j += n
        curves.append((s["label"], col[s["name"]], rows, "o-" if i == 0 else "s--"))
    p = out("pythia_proj.png")
    ci.draw_proj(curves, p, "The wake on its own per Z at e_threshold 2, per Pythia setting "
                            "(bins π/8, 0.4)")
    paths.append(p)

    # ── harmonics, share toward the Z ──
    xs = np.arange(wi.N_PT)
    qty = (("c0", "c_0: wake hadrons per Z"), ("c1", "c_1 = Σ cos Δφ per Z"),
           ("zyield", "wake hadrons per Z toward the Z"), ("zshare", "share toward the Z"))
    fig, ax = figdata.subplots(2, len(qty), figsize=(4.2 * len(qty), 7.2), squeeze=False)
    for r_, frame in enumerate(("Z", "J")):
        for j, (q, title) in enumerate(qty):
            aa = ax[r_, j]
            for i, s in enumerate(S):
                for e, mfc, dx in (("e2", None, 0.0), ("e3", "none", 0.04)):
                    val, err = R[s["name"]][e]["wake"]
                    aa.errorbar(xs + 0.1 * i + dx, val[f"{q} {frame}"], err[f"{q} {frame}"],
                                fmt="o" if e == "e2" else "s", color=col[s["name"]], ms=4,
                                mfc=mfc, label=f"{s['label']}, e_threshold {e[1]}")
            aa.axhline(0, color="0.6", lw=0.6)
            if q == "zshare":
                aa.axhline(0.5, color="0.6", lw=0.6, ls="--")
            aa.set_xticks(xs)
            aa.set_xticklabels([f"{x:g}–{y:g}" for x, y in zip(wi.PT_EDGES[:-1],
                                                              wi.PT_EDGES[1:])], fontsize=7)
            where = "Z frame, all Δη" if frame == "Z" else f"J frame, |Δη| < {ci.J_ETA:g}"
            aa.set_title(f"{title}\n{where}", fontsize=8)
            aa.set_xlabel("track p_T [GeV]")
            if r_ == 0 and j == 0:
                aa.legend(fontsize=6)
    fig.suptitle("The wake's harmonics and its share toward the Z per Pythia setting "
                 "(filled: e_threshold 2, open: 3)", fontsize=10)
    fig.tight_layout()
    p = out("pythia_harm.png")
    figdata.savefig(fig, p, dpi=90)
    plt.close(fig)
    paths.append(p)

    # ── the e_threshold 3 increment ──
    fig, ax = figdata.subplots(2, 4, figsize=(18, 8), squeeze=False)
    for j, (q, frame, title) in enumerate((("c0", "Z", "Δc_0 (3 − 2)"), ("c1", "Z", "Δc_1, Z frame"),
                                           ("v1", "Z", "Δv_1, Z frame"),
                                           ("v1", "J", "Δv_1, J frame"))):
        aa = ax[0, j]
        for i, s in enumerate(S):
            val, err = R[s["name"]]["inc"]["wake"]
            aa.errorbar(xs + 0.1 * i, val[f"{q} {frame}"], err[f"{q} {frame}"], fmt="o",
                        color=col[s["name"]], ms=4, label=s["label"])
        aa.axhline(0, color="0.6", lw=0.6)
        aa.set_xticks(xs)
        aa.set_xticklabels([f"{x:g}–{y:g}" for x, y in zip(wi.PT_EDGES[:-1], wi.PT_EDGES[1:])],
                           fontsize=7)
        aa.set_title(f"the wake's e_threshold 3 − 2: {title} (paired)", fontsize=8)
        aa.set_xlabel("track p_T [GeV]")
        if j == 0:
            aa.legend(fontsize=7)
    for j, (frame, kind, title) in enumerate((("J", "phi", "J frame Δφ, |Δη| < 1.4"),
                                              ("J", "eta", "J frame Δη toward the recoil"),
                                              ("Z", "phi", "Z frame Δφ, all Δη"))):
        aa = ax[1, j]
        for i, s in enumerate(S):
            def f(V, frame=frame, kind=kind):
                g = (lambda v: wi.phi_proj(v, "wake", frame, ci.J_ETA if frame == "J" else None)
                     if kind == "phi" else wi.eta_proj(v, "wake", frame, "rec"))
                x = wi.PHI_C if kind == "phi" else wi.ETA_C
                return ci.rebin1(x, g(V[0])[1] - g(V[1])[1], 4)[1]
            val, err = wi.jackknife(f, [[s["e3"], s["e2"]]], a.blocks)
            x = ci.rebin1(wi.PHI_C if kind == "phi" else wi.ETA_C, np.zeros(wi.N_PHI if kind ==
                          "phi" else wi.N_ETA), 4)[0]
            keep = np.ones(len(x), bool) if kind == "phi" else np.abs(x) < 3.4
            aa.errorbar(x[keep] + 0.03 * i, val[keep], err[keep], fmt="o-", ms=3, lw=0.9,
                        color=col[s["name"]], label=s["label"])
        aa.axhline(0, color="0.6", lw=0.6)
        aa.set_title(f"the increment at 1 < p_T < 2 GeV: {title}", fontsize=8)
        aa.set(xlabel="Δφ" if kind == "phi" else "Δη",
               ylabel="dN/dΔφ per Z" if kind == "phi" else "dN/dΔη per Z")
    aa = ax[1, 3]
    labels = ("wake E", "fragments E", "wake + frag − pp E", "wake Σp_T cos Δφ_Z")
    for i, s in enumerate(S):
        r = R[s["name"]]
        vw, ew = r["inc"]["wake"]
        vf, ef = r["inc"]["frag"]
        b3, e3 = r["e3"]["budget"]
        b2, e2 = r["e2"]["budget"]
        vals = [vw["total E"][0], vf["total E"][0], b3[1] - b2[1], vw["total pT cos dphi_Z"][0]]
        errs = [ew["total E"][0], ef["total E"][0], np.hypot(e3[1], e2[1]),
                ew["total pT cos dphi_Z"][0]]
        aa.errorbar(np.arange(4) + 0.12 * i, vals, errs, fmt="o", color=col[s["name"]], ms=5,
                    label=s["label"])
    aa.axhline(0, color="0.6", lw=0.6)
    aa.set_xticks(np.arange(4))
    aa.set_xticklabels(labels, fontsize=7, rotation=15)
    aa.set(ylabel="e_threshold 3 − 2 per Z [GeV]", title="the increment's totals (|η| < 2.6)")
    fig.suptitle("What e_threshold 3 adds to the wake, per Pythia setting (paired within each)",
                 fontsize=10)
    fig.tight_layout()
    p = out("pythia_e3.png")
    figdata.savefig(fig, p, dpi=90)
    plt.close(fig)
    paths.append(p)

    # ── the fragments, Pb+Pb - pp ──
    fig, ax = figdata.subplots(1, N_CLS + 1, figsize=(4.6 * (N_CLS + 1), 4.2), squeeze=False)
    for i, s in enumerate(S):
        def f(V):
            return ci.rebin1(wi.PHI_C, wi.phi_proj(V[0], "frag", "Z")
                             - wi.phi_proj(V[1], "frag", "Z"), 2)[1]
        val, err = wi.jackknife(f, [[s["e2"]], [s["pp"]]], a.blocks)
        x = ci.rebin1(wi.PHI_C, np.zeros((1, wi.N_PHI)), 2)[0]
        for c in range(N_CLS):
            ax[0, c].errorbar(x + 0.02 * i, val[c], err[c], fmt="o-", ms=3, lw=0.9,
                              color=col[s["name"]], label=s["label"])
            ax[0, c].set(xlabel="Δφ − φ_Z", ylabel="dN/dΔφ per Z",
                         title=f"fragments, Pb+Pb − pp, all Δη\n{wi.PT_LABELS[c]}")
            ax[0, c].axhline(0, color="0.6", lw=0.6)
        val, err = R[s["name"]]["e2"]["frag-pp"]
        ax[0, N_CLS].errorbar(xs + 0.1 * i, val["c0 Z"], err["c0 Z"], fmt="o",
                              color=col[s["name"]], ms=4, label=f"{s['label']}: c_0")
        ax[0, N_CLS].errorbar(xs + 0.1 * i + 0.04, val["c1 Z"], err["c1 Z"], fmt="s",
                              color=col[s["name"]], ms=4, mfc="none", label=f"{s['label']}: c_1")
    ax[0, N_CLS].axhline(0, color="0.6", lw=0.6)
    ax[0, N_CLS].set_xticks(xs)
    ax[0, N_CLS].set_xticklabels([f"{x:g}–{y:g}" for x, y in zip(wi.PT_EDGES[:-1],
                                                                 wi.PT_EDGES[1:])], fontsize=7)
    ax[0, N_CLS].set(xlabel="track p_T [GeV]", title="fragments, Pb+Pb − pp: c_0 and c_1")
    ax[0, 0].legend(fontsize=7)
    ax[0, N_CLS].legend(fontsize=6)
    fig.suptitle("The fragments' loss, Pb+Pb − pp (each setting against its own pp), "
                 "e_threshold 2", fontsize=10)
    fig.tight_layout()
    p = out("pythia_frag.png")
    figdata.savefig(fig, p, dpi=90)
    plt.close(fig)
    paths.append(p)

    # ── the text ──
    def fe(v, e, k, c=None, p=3):
        x = np.atleast_1d(v[k])[0 if c is None else c]
        y = np.atleast_1d(e[k])[0 if c is None else c]
        return f"{x:+.{p}f}±{y:.{p}f}"

    per_cls = ("c0 Z", "c1 Z", "v1 Z", "zshare Z", "zyield Z", "cms Z", "c0 J", "c1 J", "v1 J",
               "zshare J", "zyield J",
               "rho_in Z", "rho_out Z", "dip Z", "rho_in J", "rho_out J", "dip J", "core J",
               "wings J")
    totals = [f"total {t}" for t in wi.TOTALS]
    L = [f"The wake on its own per Pythia setting (compare_pythia.py), from {a.data}/iso",
         "Per Z; c_n, v_1, zshare (the share toward the Z), rho_in/out, dip, core/wings as in "
         "compare_iso.py; totals over every stored hadron (|eta| < 2.6).  Errors: jackknife over "
         "hydro backgrounds (and pp event blocks); settings against the first share the "
         "background blocks when their backgrounds are the same.", ""]
    for s in S:
        L.append(f"{s['label']} ({s['name']}: {', '.join(s['runs'])})")
        for e in ("e2", "e3"):
            k = R[s["name"]][e]["kin"]
            L.append(f"  e_threshold {e[1]}: {k['n']} Z's (N_eff {k['n_eff']:.0f}, J frame "
                     f"{k['in_j']:.2f}), <p_T^Z> {k['z_pt']:.1f}, <p_T,rec> {k['rec_pt']:.1f}, "
                     f"<E_droplets> {k['e_drop']:.1f} GeV")
        if s is not ref:
            L.append(f"  backgrounds the same as {ref['name']}'s: {R[s['name']]['paired_bg']}")
    L.append("")

    def block(title, ve, keys, cls=True):
        v, e = ve
        L.append(title)
        if cls:
            for c in range(N_CLS):
                L.append(f"  {wi.PT_LABELS[c]:22s} " + " | ".join(
                    f"{k} {fe(v, e, k, c)}" for k in keys if k in v))
        tk = [k for k in totals if k in v] + (["E_droplets"] if "E_droplets" in v else [])
        if tk:
            L.append("  totals: " + " | ".join(f"{k.replace('total ', '')} {fe(v, e, k, p=2)}"
                                               for k in tk))

    for s in S:
        r = R[s["name"]]
        for e in ("e2", "e3"):
            block(f"wake, {s['label']}, e_threshold {e[1]}", r[e]["wake"], per_cls)
        block(f"fragments Pb+Pb, {s['label']}, e_threshold 2", r["e2"]["frag"],
              ("c0 Z", "c1 Z", "v1 Z"))
        block(f"fragments pp, {s['label']}", r["pp"], ("c0 Z", "c1 Z", "v1 Z"))
        block(f"fragments Pb+Pb - pp, {s['label']}, e_threshold 2", r["e2"]["frag-pp"],
              ("c0 Z", "c1 Z"))
        for e in ("e2", "e3"):
            bv, be = r[e]["budget"]
            L.append(f"budget (wake + fragments - pp), {s['label']}, e_threshold {e[1]}: " +
                     " | ".join(f"{t} {bv[i]:+.2f}±{be[i]:.2f}" for i, t in enumerate(wi.TOTALS)))
        block(f"wake, e_threshold 3 - 2 (paired), {s['label']}", r["inc"]["wake"], per_cls)
        block(f"fragments, e_threshold 3 - 2 (paired), {s['label']}", r["inc"]["frag"],
              ("c0 Z", "c1 Z"))
        if s is not ref:
            block(f"wake, {s['label']} - {ref['label']}, e_threshold 2", r["vs_ref"]["wake"],
                  per_cls)
            block(f"wake increment (3 - 2), {s['label']} - {ref['label']}", r["inc_vs_ref"],
                  ("c0 Z", "c1 Z", "v1 Z", "zshare Z", "c0 J", "v1 J", "core J", "wings J"))
        L.append("")
    txt = out("pythia_compare.txt")
    with open(txt, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    for p in paths + [txt]:
        print(f"-> {p}")


if __name__ == "__main__":
    main()
