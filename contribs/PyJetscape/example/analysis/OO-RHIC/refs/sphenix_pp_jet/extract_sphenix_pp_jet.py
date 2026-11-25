#!/usr/bin/env python3
"""
refs/sphenix_pp_jet/extract_sphenix_pp_jet.py -- sPHENIX's preliminary inclusive jet cross section
in p+p at 200 GeV (sPH-CONF-JET-2025-03, April 2025; anti-kT R = 0.4, |eta_jet| < 0.7, 16.6 pb^-1,
unfolded to particle level), read from the note's vector Fig. 5 (page 9): the note has no tables.
The drawn statistical bars are the same in every bin (a drawing size; upper bounds).

    python extract_sphenix_pp_jet.py NOTE.pdf     # -> sphenix_pp_jet_prelim.json (this folder)

Per bin: the edges and the value from the systematic box (its horizontal extent is the bin), the
marker's centre as the value, the statistical bar's extent, and the box as the systematic range
(asymmetric on the log axis).  Axes mapped with the major ticks (x 20-70 GeV, y 1e-3 .. 1e5
pb/GeV in decades).  PRELIMINARY: for comparison only.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "sphenix_oo_jet"))
import extract_sphenix_oo_jet as X  # noqa: E402

PANEL = "matrix(0.61908, 0, 0, -0.61908, 130.5, 529.704)"
BLUE = "rgb(0%, 39.99939%, 79.998779%)"          # markers and statistical bars
BOX = "rgb(19.999695%, 59.999084%, 100%)"         # systematic boxes


def main(pdf):
    with tempfile.TemporaryDirectory() as d:
        svg = f"{d}/p9.svg"
        subprocess.run(["pdftocairo", "-svg", "-f", "9", "-l", "9", pdf, svg], check=True)
        ps = X.paths(svg)
    (x0, x1, y0, y1), xt, yt = X.axes(ps, PANEL)
    if len(xt) != 6 or len(yt) != 9:
        sys.exit(f"extract_sphenix_pp_jet.py: unexpected ticks {len(xt)} x, {len(yt)} y")
    fx = X.mapper(xt, [20, 30, 40, 50, 60, 70])
    fy = X.mapper(yt, list(range(-3, 6)))                          # log10, bottom to top
    boxes = [p["P"] for p in ps if p["fill"] == BOX and p["P"][:, 1].min() > y0 - 1
             and p["P"][:, 0].min() >= x0 - 1 and p["P"][:, 0].max() <= x1 + 1]
    marks = [p["P"] for p in ps if p["fill"] == BLUE and len(p["P"]) == 4]
    bars = [p["P"] for p in ps if p["tr"] == PANEL and p["stroke"] == BLUE and p["sw"] == "1.5"
            and abs(p["P"][0, 0] - p["P"][1, 0]) < 1e-3]
    rows = []
    for B in sorted(boxes, key=lambda B: B[:, 0].min()):
        xl, xh = B[:, 0].min(), B[:, 0].max()
        # a marker is a circle drawn with Bezier curves: its stored points are the start and the
        # curve ends, so the centre is the midpoint of the x range; it must lie inside the box
        inside = [M for M in marks if xl < (M[:, 0].min() + M[:, 0].max()) / 2 < xh
                  and B[:, 1].min() - 1 <= M[:, 1].mean() <= B[:, 1].max() + 1]
        if len(inside) != 1:                                           # the legend's box
            continue
        M = inside[0]
        xc = (M[:, 0].min() + M[:, 0].max()) / 2
        val = 10 ** float(fy(M[:, 1].mean()))
        vb = [S for S in bars if abs(S[0, 0] - xc) < 0.5 and np.ptp(S[:, 1]) > 1e-3]
        if not vb:                                                     # the legend's box and marker
            continue
        ys = np.concatenate([S[:, 1] for S in vb]) if vb else np.array([M[:, 1].mean()])
        lo, hi = 10 ** fy(ys.max()), 10 ** fy(ys.min())                # page y grows down
        slo, shi = 10 ** fy(B[:, 1].max()), 10 ** fy(B[:, 1].min())
        rows.append({"lo": round(float(fx(xl)), 2), "hi": round(float(fx(xh)), 2), "value": float(f"{val:.4g}"),
                     "stat_lo": float(f"{max(val - lo, 0):.3g}"), "stat_hi": float(f"{max(hi - val, 0):.3g}"),
                     "sys_lo": float(f"{val - slo:.3g}"), "sys_hi": float(f"{shi - val:.3g}")})
    out = {"source": "sPHENIX, sPH-CONF-JET-2025-03 (preliminary, April 2025), "
                     "https://www.sphenix.bnl.gov/sites/default/files/2025-04/sPH-CONF-JET-2025-03.pdf",
           "observable": "d2sigma/(deta dpT) [pb/GeV], anti-kT R = 0.4, |eta_jet| < 0.7, p+p 200 GeV, 16.6 pb^-1, "
                         "unfolded to particle level",
           "method": "Fig. 5 read from the vector graphics (pdftocairo -svg): marker centre = value, bars = stat, "
                     "boxes = syst (asymmetric in log), box extent = bin; axes from the major ticks",
           "caveat": "the drawn statistical bars have the same length in every bin (+-0.074 in log10, +-17 %): "
                     "a drawing size rather than the error (the note: statistical errors often smaller than "
                     "the marker, about 20 % above 50 GeV), so an upper bound; the systematic boxes are largely "
                     "correlated between bins (JES 6 %, unfolding prior)",
           "rows": rows}
    (HERE / "sphenix_pp_jet_prelim.json").write_text(json.dumps(out, indent=1))
    for r in rows:
        print(f"{r['lo']:5.1f}-{r['hi']:<5.1f} {r['value']:10.4g}  stat -{r['stat_lo']:.3g}/+{r['stat_hi']:.3g}  "
              f"syst -{r['sys_lo']:.3g}/+{r['sys_hi']:.3g}")


if __name__ == "__main__":
    main(sys.argv[1])
