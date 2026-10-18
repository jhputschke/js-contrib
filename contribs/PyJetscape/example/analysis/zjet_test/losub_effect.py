#!/usr/bin/env python3
"""The paired bulk term of the Z-correlated yield, <bulk_jet> - <bulk_bg>, in CMS
2507.09307's observables, against what the model misses (PLAN_hard_momentum_subtraction.md
§6) -> <outdir>/losub_effect{tag}.png (+ .npz/.root) and losub_effect{tag}.txt.

Inputs are accumulate.py's histograms (hists.npz), one per production:
* --hists: e.g. run C (--no-deposit --lo-hard-subtraction 1): the bulk term is then the
  LO hard subtraction's effect alone, no wake;
* --hists-wake (optional): a production with deposition and without the subtraction: the
  bulk term is the wake (e.g. data/hists.npz, the 96-event test);
* --reference: step 1 of the c00 + c01 mixing (mix.py's figure data,
  mixing_reference_d{phi,y}{tag}.npz) and CMS's points in it: CMS - model is what the model
  misses, with CMS's errors (stat + syst) and the model's in quadrature.

The bulk term goes through the same S - B as compare.py's model (Model.observables: Delta
phi around its mean; Delta y on the Z side minus half the mixed events, which here are
the bulk term's own eta density folded with the Z's), so it adds linearly to the fragments'
and the reference's.  Errors: per-event (+) a jackknife over the hydro backgrounds."""
import argparse
import json
import os

import numpy as np

import figdata
from compare import PT_LABELS, Model, rebin_y

HERE = os.path.dirname(os.path.abspath(__file__))


def bulk_term(path):
    """-> {'phi': (value, error), 'y': (value, error)} of <bulk_jet> - <bulk_bg> per Z,
    (pT class, bin); Delta y in CMS Pb+Pb's 0.8 bins; and the number of selected Z's."""
    d = np.load(path)
    model = Model(d["DETA"], d["DPHI"], d["ETA"])
    sel, w, yz, group = d["sel"], d["w"], d["y_z"], d["group"]
    groups = list(d["groups"])

    def obs(gs):
        n = {g: int(d[f"g{g}_n"]) for g in gs}
        H = sum(n[g] * d[f"g{g}_wake"] for g in gs)
        E = np.sqrt(sum(n[g] ** 2 * d[f"g{g}_wake_err"] ** 2 for g in gs))
        T2 = sum(n[g] * d[f"g{g}_wake_w2"] for g in gs)
        m = sel & np.isin(group, gs)
        (p, ep), (y, ey) = model.observables(H, E, T2, yz[m], w[m])
        return dict(phi=(p, ep), y=rebin_y(y, ey, 2))

    out = obs(groups)
    jack = [obs([g for g in groups if g != x]) for x in groups]
    G = len(groups)
    for var in ("phi", "y"):
        th = np.array([j[var][0] for j in jack])
        ejk = np.sqrt((G - 1) / G * ((th - th.mean(0)) ** 2).sum(0))
        out[var] = (out[var][0], np.hypot(out[var][1], ejk))
    out["x"] = dict(phi=model.cphi, y=np.linspace(-3.6, 3.6, 10))
    out["n_z"], out["n_bg"] = int(sel.sum()), G
    return out


def reference(path):
    """mix.py's step-1 figure data -> per pT class (x, CMS, CMS error, model, model error)
    of the Pb+Pb row (axes 3, 4, 5)."""
    d = np.load(path, allow_pickle=True)
    m = json.loads(str(d["__manifest__"]))
    rows = {}
    for c in m["calls"]:
        if c["method"] != "errorbar" or c["ax"] not in (3, 4, 5):
            continue
        x, y, e = (d[a["__array__"]] for a in c["args"])
        kind = "cms" if c["kwargs"].get("fmt") == "o" else "model"
        rows.setdefault(c["ax"] - 3, {})[kind] = (x, y, e)
    return [(r["cms"][0], r["cms"][1], r["cms"][2], r["model"][1], r["model"][2])
            for _, r in sorted(rows.items())]


def at(x, v, e, x0):
    i = int(np.argmin(np.abs(x - x0)))
    return v[i], e[i]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hists", required=True, help="accumulate.py's output of the run")
    ap.add_argument("--label", default="LO hard subtraction, no wake (run C)")
    ap.add_argument("--hists-wake", default=None,
                    help="accumulate.py's output of a production with the wake")
    ap.add_argument("--label-wake", default="wake (deposition, no subtraction)")
    ap.add_argument("--reference", default=os.path.join(HERE, "figures-c00-01"),
                    help="directory with mixing_reference_d{phi,y}<--ref-tag>.npz")
    ap.add_argument("--ref-tag", default="_c00_01")
    ap.add_argument("--outdir", default=os.path.join(HERE, "figures"))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    runs = [(a.label, bulk_term(a.hists), "C3")]
    if a.hists_wake:
        runs.append((a.label_wake, bulk_term(a.hists_wake), "C0"))
    ref = {v: reference(os.path.join(a.reference, f"mixing_reference_d{v}{a.ref_tag}.npz"))
           for v in ("phi", "y")}

    lines = [f"<bulk_jet> - <bulk_bg> in CMS 2507.09307's S - B, per Z, against CMS - model "
             f"(step 1, {a.ref_tag.strip('_')})"]
    for lab, r, _ in runs:
        lines.append(f"  {lab}: {r['n_z']} selected Z's, {r['n_bg']} hydro backgrounds "
                     f"({os.path.basename(a.hists if r is runs[0][1] else a.hists_wake)})")
    lines += ["", f"{'':5s} {'':20s} {'first bin':>10s} " + " ".join(
        f"{lab[:28]:>30s}" for lab, _, _ in runs) + f" {'CMS - model':>18s}"]
    fig, ax = figdata.subplots(2, 3, figsize=(16, 8.5))
    for k, var in enumerate(("y", "phi")):
        x0 = 0.4 if var == "y" else 0.157
        for c in range(3):
            a_ = ax[k, c]
            xr, ycms, ecms, ymod, emod = ref[var][c]
            miss, emiss = ycms - ymod, np.hypot(ecms, emod)
            a_.errorbar(xr, miss, emiss, fmt="o", color="k", ms=4,
                        label="CMS − model (c00+c01 step 1)")
            vals = []
            for lab, r, col in runs:
                v, e = r[var][0][c], r[var][1][c]
                a_.errorbar(r["x"][var], v, e, fmt="s", color=col, ms=4, mfc="none",
                            label=lab)
                vals.append(at(r["x"][var], v, e, x0))
            a_.axhline(0, color="0.6", lw=0.6)
            if k == 0:
                a_.set_title(PT_LABELS[c] + ",  |Δφ| < π/2")
            a_.set_xlabel("Δy_ch,Z" if var == "y" else "Δφ_ch,Z")
            a_.set_ylabel(f"Pb+Pb 0–30 %: d⟨ΔN_ch⟩/dΔ{'y' if var == 'y' else 'φ'}")
            if c == 0 and k == 0:
                a_.legend(fontsize=8)
            m0 = at(xr, miss, emiss, x0)
            lines.append(f"{'Δ' + var:5s} {PT_LABELS[c]:20s} {x0:10.3f} " + " ".join(
                f"{v:+14.3f} ± {e:5.3f}       " for v, e in vals)
                + f" {m0[0]:+8.3f} ± {m0[1]:5.3f}")
    fig.suptitle("Z–hadron S − B of the paired bulk term ⟨bulk_jet⟩ − ⟨bulk_bg⟩ against "
                 "what the model misses", fontsize=12)
    fig.tight_layout()
    png = os.path.join(a.outdir, f"losub_effect{a.tag}.png")
    figdata.savefig(fig, png)
    txt = os.path.join(a.outdir, f"losub_effect{a.tag}.txt")
    text = "\n".join(lines) + "\n"
    open(txt, "w").write(text)
    print(text, end="")
    print(f"-> {png}, {txt}")


if __name__ == "__main__":
    main()
