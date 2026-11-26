#!/usr/bin/env python3
"""
refs/phobos_dau200/extract_phobos_dau200.py -- PHOBOS charged-particle dN/deta in d+Au at 200 GeV,
centrality 0-20, 20-40, 40-60, 60-80, 80-100 % and minimum bias, read from the vector Fig. 27
(page 19) of arXiv:1011.1940 (PRC 83, 024913 (2011)).  eta is relative to the deuteron beam
(+z); the Au-going side is eta < 0.  The grey band is 1 sigma (not 90 % C.L. as for Au+Au).

    python extract_phobos_dau200.py PAPER.pdf      # -> phobos_dau200_dndeta.json (this folder)

Same method as ../phobos_auau200/extract_phobos_auau200.py: marker centres, statistical bars
(smaller than the markers: 0), the band's two edge curves at each point; axes from the major
ticks (eta -4 .. 4; dN/deta 0-10 in the top row, 0-20 in the bottom row).  Also Table III
(<N_part^Au>, <N_part^d>, N_ch in |eta| < 5.4) and a check of the read points against it.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "sphenix_oo_jet"))
sys.path.insert(0, str(HERE.parent / "phobos_auau200"))
import extract_sphenix_oo_jet as X  # noqa: E402
import extract_phobos_auau200 as A  # noqa: E402  (tick finder, colours)

COLS = [(338.8, 413.0), (413.0, 487.3), (487.3, 561.5)]
ROWS = [(53.0, 144.2), (144.2, 235.3)]
CENT = {(0, 0): "80-100", (0, 1): "60-80", (0, 2): "40-60", (1, 0): "20-40", (1, 1): "0-20", (1, 2): "MB"}
YV = {0: [0, 5, 10], 1: [0, 10, 20]}
MAJOR = (3.8, 4.6)                              # major tick length in Fig. 27 (4.24 pt)
# Table III: <N_part^Au>, <N_part^d>, N_ch |eta| < 5.4 (90 % C.L.)
TAB = {"0-20": (13.5, 2.0, 157, 10), "20-40": (8.9, 1.9, 109, 7), "40-60": (5.4, 1.7, 74, 5),
       "60-80": (2.9, 1.4, 46, 3), "80-100": (1.6, 1.1, 28, 3), "MB": (6.6, 1.7, 82, 6)}


def main(pdf):
    with tempfile.TemporaryDirectory() as d:
        svg = f"{d}/p19.svg"
        subprocess.run(["pdftocairo", "-svg", "-f", "19", "-l", "19", pdf, svg], check=True)
        ps = [p for p in X.paths(svg) if p["P"][:, 0].min() > COLS[0][0] - 1 and p["P"][:, 1].max() < ROWS[-1][1] + 10]
    fy, fx = {}, {}
    for r, (ya, yb) in enumerate(ROWS):
        t = sorted([y for y in A.ticks(ps, x0=COLS[0][0], length=MAJOR) if ya - 0.3 <= y <= yb + 0.3], reverse=True)[:len(YV[r])]
        if len(t) != len(YV[r]):
            sys.exit(f"row {r}: ticks {t}")
        fy[r] = X.mapper(t, YV[r])
    for c, (xa, xb) in enumerate(COLS):
        t = [x for x in A.ticks(ps, y1=ROWS[-1][1], length=MAJOR) if xa + 0.5 < x < xb - 0.5]       # not the frame lines
        if len(t) != 5:
            sys.exit(f"column {c}: ticks {t}")
        fx[c] = X.mapper(t, [-4, -2, 0, 2, 4])
    marks = [p["P"] for p in ps if p["fill"] == A.BLACK and len(p["P"]) == 5
             and 1.4 < np.ptp(p["P"][:, 0]) < 2.2 and 1.4 < np.ptp(p["P"][:, 1]) < 2.2]
    pinks = [p["P"] for p in ps if p["stroke"] == A.PINK]
    out = {"source": "PHOBOS, PRC 83, 024913 (2011), arXiv:1011.1940, Fig. 27 (read from the vector figure)",
           "units": "dN_ch/deta; eta relative to the deuteron beam (Au-going side eta < 0); sys = 1 sigma band edges",
           "bins": {}}
    for (r, c), cen in CENT.items():
        (xa, xb), (ya, yb) = COLS[c], ROWS[r]
        inside = lambda P: xa < P[:, 0].mean() < xb and ya < P[:, 1].mean() < yb
        M = np.array([P.mean(0) for P in marks if inside(P)])
        M = M[np.argsort(M[:, 0])]
        E = sorted([P for P in pinks if inside(P) and np.ptp(P[:, 0]) > 0.8 * (xb - xa)], key=lambda P: P[:, 1].mean())
        if len(E) < 2:
            sys.exit(f"{cen}: {len(E)} band edges")
        E = [E[0], E[-1]]
        rows = []
        for mx, my in M:
            val = float(fy[r](my))
            edges = [float(fy[r](np.interp(mx, *zip(*sorted(zip(P[:, 0], P[:, 1])))))) for P in E]
            rows.append({"eta": round(float(fx[c](mx)), 3), "value": round(val, 3), "stat": 0.0,
                         "sys_hi": round(max(edges) - val, 3), "sys_lo": round(val - min(edges), 3)})
        na, nd, nch, nche = TAB[cen]
        out["bins"][cen] = {"npart_au": na, "npart_d": nd, "table_nch_5p4": nch, "table_nch_err": nche, "points": rows}
        e = np.array([p["eta"] for p in rows]); v = np.array([p["value"] for p in rows]); de = np.median(np.diff(e))
        print(f"{cen:6s} {len(rows):3d} points, eta {e.min():+.2f} .. {e.max():+.2f} (step {de:.3f}); "
              f"sum*deta {v.sum() * de:6.1f} (Table III N_ch 5.4: {nch} +- {nche}); peak {v.max():.2f} at eta {e[v.argmax()]:+.2f}")
    (HERE / "phobos_dau200_dndeta.json").write_text(json.dumps(out, indent=1))
    print("->", HERE / "phobos_dau200_dndeta.json")


if __name__ == "__main__":
    main(sys.argv[1])
