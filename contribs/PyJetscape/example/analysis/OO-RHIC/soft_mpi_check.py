#!/usr/bin/env python3
"""
analysis/OO-RHIC/soft_mpi_check.py -- the soft-MPI check after gate G2: p+p with hydro in the
production chain with Pythia's soft MPI at two pT0Ref (oo_bias.py group phys2_soft:
pp_bulk_soft142_w*, pp_bulk_soft145_w*, pp_bulk_soft148_w*, 1.5k events per window, windows 5-55 GeV), against STAR's
underlying event and against MPI on (the tune) and off from phase 2.
    python soft_mpi_check.py --out /data/OO-RHIC_phys -j 16    # -> results/soft_mpi_check.json

Per setting and leading-jet bin: transverse density (ue_check.py's definitions) and its ratio to
STAR, the pT > 0.5 GeV density, the transverse <pT>, has_bulk, and fwd_check.py's forward ratios
(EPD, MBD over the minimum bias of pp_tau0424).  Then the pT0Ref at which the transverse density
reaches STAR's, for three ranges of leading-jet bins: linear in ln(ratio) between the two soft
points that bracket it, and from a parabola in ln(ratio) through all soft points (with three).
The same for the chi2 over the complete bins (its minimum on the parabola).  The MPI on / off sums are taken from g2_check.py's cache when present.
Model results with the frozen tune XML plus the soft MPI lines, not vetted physics results.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fwd_check as F  # noqa: E402
import g2_check as G  # noqa: E402
import ue_check as U  # noqa: E402

VARIANTS = {"pp_bulk": ("MPI on (tune)", None), "pp_bulk_soft142": ("soft MPI, pT0Ref 1.42", 1.42),
            "pp_bulk_soft145": ("soft MPI, pT0Ref 1.45", 1.45), "pp_bulk_soft148": ("soft MPI, pT0Ref 1.48", 1.48),
            "pp_bulk_mpioff": ("MPI off", None)}
SOFT = ("pp_bulk_soft142", "pp_bulk_soft145", "pp_bulk_soft148")
#: leading-jet bin ranges for the interpolation (indices into ue_check.LEAD_EDGES)
RANGES = {"6-15 GeV (bins 5-15, three incomplete)": (0, 4), "11-20 GeV": (3, 5),
          "11-45 GeV (all complete bins)": (3, 8)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/data/OO-RHIC_phys")
    ap.add_argument("-j", type=int, default=16)
    a = ap.parse_args(argv)
    out = Path(a.out)
    star = U.load_star()
    sv = np.asarray(star["transverse"]["value"])
    se = np.hypot(star["transverse"]["stat"], star["transverse"]["sys"])
    s05 = np.asarray(star["transverse05"]["value"])
    smpt = np.asarray(star["meanpt_transverse"]["value"])
    comp = U.LEAD_EDGES[:-1] >= G.COMPLETE_FROM
    mb = F.mb_forward(*G.MB_PP)
    res = {"lead_edges": U.LEAD_EDGES.tolist(), "complete_from_gev": G.COMPLETE_FROM,
           "star": {"transverse": sv.tolist(), "transverse_err": se.tolist(),
                    "transverse05": s05.tolist(), "meanpt_transverse": smpt.tolist()},
           "variants": {}}
    for v, (lab, p) in VARIANTS.items():
        files = G.ue_files(out, v)
        if not files:
            print(f"soft_mpi_check.py: no finished files for {v}", file=sys.stderr)
            continue
        npz = HERE / "results" / f"ue_phys_{v}.npz"
        if npz.exists() and v not in SOFT:
            acc = dict(np.load(npz))
        else:
            acc = G.ue_sums(files, a.j)
            np.savez(npz, **acc)
        rows = G.ue_rows(acc)[0]
        t = np.array([r["n_transverse"] for r in rows])
        te = np.array([r["n_transverse_err"] for r in rows])
        m = F.means(F.sums(files, a.j))
        res["variants"][v] = {
            "label": lab, "pt0ref": p, "files": len(files), "events": int(acc["n_events"].sum()),
            "job": G.job_settings(out, f"{v}_w09_11"),
            "transverse": t.tolist(), "transverse_err": te.tolist(), "ratio_to_star": (t / sv).tolist(),
            "transverse05_ratio_to_star": (np.array([r["n_trans05"] for r in rows]) / s05).tolist(),
            "meanpt_ratio_to_star": (np.array([r["meanpt_transverse"] for r in rows]) / smpt).tolist(),
            "has_bulk": [r["has_bulk"] for r in rows],
            "chi2_complete_bins": float(np.sum((((t - sv) / np.hypot(te, se))[comp]) ** 2)),
            "n_complete_bins": int(comp.sum()),
            "epd_over_mb": (m["n_epd"] / mb["all_epd"]).tolist(),
            "mbd_over_mb": (m["n_mbd"] / mb["all_mbd"]).tolist(),
            "epd_over_mb_relerr": (m["n_epd_err"] / np.maximum(m["n_epd"], 1e-9)).tolist(),
            "mbd_over_mb_relerr": (m["n_mbd_err"] / np.maximum(m["n_mbd"], 1e-9)).tolist(),
        }
    V = res["variants"]
    pts = [s for s in SOFT if s in V]
    if len(pts) >= 2:
        x = np.array([V[s]["pt0ref"] for s in pts])
        res["interpolation"] = {}
        for name, (i, j) in RANGES.items():
            r = np.array([np.mean(V[s]["ratio_to_star"][i:j]) for s in pts])
            y = np.log(r)
            d = {"ratio": dict(zip(map(float, x), map(float, r)))}
            k = next((k for k in range(len(x) - 1) if y[k] * y[k + 1] <= 0), None)
            if k is not None:      # linear in ln(ratio) between the bracketing pair
                d["pt0ref_at_star_linear"] = float(x[k] - y[k] * (x[k + 1] - x[k]) / (y[k + 1] - y[k]))
            if len(x) >= 3:        # parabola through all points, the root inside the scanned range
                roots = [z.real for z in np.roots(np.polyfit(x, y, 2)) if abs(z.imag) < 1e-12
                         and x.min() - 0.02 <= z.real <= x.max() + 0.02]
                if roots:
                    d["pt0ref_at_star_parabola"] = float(roots[0])
            res["interpolation"][name] = d
        if len(x) >= 3:
            c2 = np.array([V[s]["chi2_complete_bins"] for s in pts])
            a2, b2, _ = np.polyfit(x, c2, 2)
            if a2 > 0:
                xm = -b2 / (2 * a2)
                res["chi2_minimum"] = {"pt0ref": float(xm), "chi2": float(np.polyval(np.polyfit(x, c2, 2), xm)),
                                       "n_bins": V[pts[0]]["n_complete_bins"], "points": dict(zip(map(float, x), map(float, c2)))}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "soft_mpi_check.json").write_text(json.dumps(res, indent=1))

    E = U.LEAD_EDGES
    print("lead jet  " + " ".join(f"{E[i]:.0f}-{E[i + 1]:.0f}".rjust(6) for i in range(len(E) - 1)))
    print("STAR      " + " ".join(f"{x:6.3f}" for x in sv))
    for v, r in V.items():
        print(f"{r['label']}: chi2 {r['chi2_complete_bins']:.1f} / {r['n_complete_bins']} complete bins")
        for k, f in (("transverse", "{:6.3f}"), ("ratio_to_star", "{:6.2f}"), ("epd_over_mb", "{:6.2f}"),
                     ("mbd_over_mb", "{:6.2f}"), ("meanpt_ratio_to_star", "{:6.2f}")):
            print(f"  {k[:8]:8s}" + " ".join(f.format(x) for x in r[k]))
    for name, d in res.get("interpolation", {}).items():
        print(f"{name}: ratio " + " / ".join(f"{v:.3f} ({k:.2f})" for k, v in d["ratio"].items())
              + "".join(f"; STAR's level {m} {d[f'pt0ref_at_star_{m}']:.3f}" for m in ("linear", "parabola")
                        if f"pt0ref_at_star_{m}" in d))
    if "chi2_minimum" in res:
        m = res["chi2_minimum"]
        print(f"chi2 over the complete bins: minimum {m['chi2']:.1f} / {m['n_bins']} at pT0Ref {m['pt0ref']:.3f} (parabola)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
