#!/usr/bin/env python3
"""
refs/phobos_auau200/extract_phobos_auau200.py -- PHOBOS charged-particle dN/deta in Au+Au at 200 GeV,
eleven centrality bins 0-3 % ... 45-50 %, read from the vector Fig. 18 (page 15) of arXiv:1011.1940
(PRC 83, 024913 (2011); INSPIRE 876609).  The per-eta values are only in the paper's EPAPS supplement
and no HEPData record was found, so the points are read from the figure:

    python extract_phobos_auau200.py PAPER.pdf      # -> phobos_auau200_dndeta.json (this folder)

Per panel: the filled markers (the measured points; their centres), the statistical bars drawn
through them (mostly smaller than the marker), and the 90 % C.L. systematic band (its two edge
curves, evaluated at each point).  Axes mapped with the major ticks: eta -4 .. 4 in steps of 2 per
column; dN/deta from 0 with the row's labelled ticks (0-150, 0-300, 0-400, 0-600).  Also the
paper's tables for the same bins (Table V: dN/deta at |eta| < 1 from the tracklets; Table VII:
N_ch in |eta| < 5.4 and the FWHM), and a check of the read points against them.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "sphenix_oo_jet"))
import extract_sphenix_oo_jet as X  # noqa: E402  (SVG path reader)

BLACK, PINK = "rgb(0%, 0%, 0%)", "rgb(73.631287%, 56.054688%, 56.054688%)"
# Fig. 18 is the right-hand figure of page 15: 3 columns x 4 rows of panels, (a)..(k)
COLS = [(349.9, 420.2), (420.2, 490.6), (490.6, 560.9)]
ROWS = [(53.2, 137.6), (137.6, 222.1), (222.1, 306.5), (306.5, 390.9)]
CENT = {(0, 0): "45-50", (0, 1): "40-45", (1, 0): "35-40", (1, 1): "30-35", (1, 2): "25-30",
        (2, 0): "20-25", (2, 1): "15-20", (2, 2): "10-15", (3, 0): "6-10", (3, 1): "3-6", (3, 2): "0-3"}
# Table V (dN/deta |eta|<1, tracklets) and Table VII (N_ch |eta|<5.4, FWHM), 200 GeV; 90 % C.L. syst.
TAB = {"0-3": (361, 691, 52, 5159, 258, 7.63), "3-6": (331, 619, 46, 4753, 238, 7.64),
       "6-10": (297, 540, 41, 4198, 210, 7.71), "10-15": (255, 465, 35, 3598, 180, 7.77),
       "15-20": (215, 384, 29, 2999, 150, 7.83), "20-25": (180, 313, 24, 2496, 125, 7.90),
       "25-30": (150, 257, 19, 2057, 103, 7.99), "30-35": (124, 208, 16, 1685, 84, 7.96),
       "35-40": (101, 165, 12, 1354, 68, 8.04), "40-45": (82, 133, 10, 1096, 55, 8.11),
       "45-50": (65, 100, 8, 847, 42, 8.23)}


def ticks(ps, x0=None, y1=None, length=(2.5, 3.5)):
    """Major tick positions (ticks of the given length range in pt; Fig. 18: 3.02) on the left axis
    (x0) or the bottom axis (y1)."""
    seg = [p["P"] for p in ps if p["stroke"] == BLACK and p["fill"] == "none" and len(p["P"]) == 2]
    if x0 is not None:
        return sorted({round(P[0, 1], 2) for P in seg if abs(P[0, 1] - P[1, 1]) < 1e-3
                       and abs(P[:, 0].min() - x0) < 0.6 and length[0] < np.ptp(P[:, 0]) < length[1]})
    return sorted({round(P[0, 0], 2) for P in seg if abs(P[0, 0] - P[1, 0]) < 1e-3
                   and abs(P[:, 1].max() - y1) < 0.6 and length[0] < np.ptp(P[:, 1]) < length[1]})


def main(pdf):
    with tempfile.TemporaryDirectory() as d:
        svg = f"{d}/p15.svg"
        subprocess.run(["pdftocairo", "-svg", "-f", "15", "-l", "15", pdf, svg], check=True)
        ps = [p for p in X.paths(svg) if p["P"][:, 0].min() > COLS[0][0] - 1]
    # axes: y per row from the left axis of column 0, x per column from the bottom axis
    yv = {0: [0, 50, 100, 150], 1: [0, 100, 200, 300], 2: [0, 200, 400], 3: [0, 200, 400, 600]}
    fy = {}
    for r, (ya, yb) in enumerate(ROWS):
        t = [y for y in ticks(ps, x0=COLS[0][0]) if ya - 0.3 <= y <= yb + 0.3]
        t = sorted(t, reverse=True)[:len(yv[r])]                      # from the bottom (0) up
        if len(t) != len(yv[r]):
            sys.exit(f"row {r}: found ticks {t}")
        fy[r] = X.mapper(t, yv[r])
    fx = {}
    for c, (xa, xb) in enumerate(COLS):
        t = [x for x in ticks(ps, y1=ROWS[-1][1]) if xa - 0.3 <= x <= xb + 0.3]
        if len(t) != 5:
            sys.exit(f"column {c}: found ticks {t}")
        fx[c] = X.mapper(t, [-4, -2, 0, 2, 4])
    marks = [p["P"] for p in ps if p["fill"] == BLACK and len(p["P"]) == 5
             and 1.5 < np.ptp(p["P"][:, 0]) < 2.2 and 1.5 < np.ptp(p["P"][:, 1]) < 2.2]
    bars = [p["P"] for p in ps if p["stroke"] == BLACK and p["fill"] == "none" and len(p["P"]) == 2
            and abs(p["P"][0, 0] - p["P"][1, 0]) < 1e-3]
    pinks = [p["P"] for p in ps if p["stroke"] == PINK]
    out = {"source": "PHOBOS, PRC 83, 024913 (2011), arXiv:1011.1940, Fig. 18 (read from the vector figure)",
           "units": "dN_ch/deta; sys = 90 % C.L. band edges at the point", "bins": {}}
    for (r, c), cen in CENT.items():
        (xa, xb), (ya, yb) = COLS[c], ROWS[r]
        inside = lambda P: xa < P[:, 0].mean() < xb and ya < P[:, 1].mean() < yb
        M = np.array([P.mean(0) for P in marks if inside(P)])
        M = M[np.argsort(M[:, 0])]
        # the band's two edge curves (pink polylines spanning most of the panel width)
        E = [P for P in pinks if inside(P) and np.ptp(P[:, 0]) > 0.8 * (xb - xa)]
        if len(E) < 2:
            sys.exit(f"{cen}: {len(E)} band edges")
        E = sorted(E, key=lambda P: P[:, 1].mean())[:1] + sorted(E, key=lambda P: P[:, 1].mean())[-1:]
        rows = []
        for mx, my in M:
            eta, val = float(fx[c](mx)), float(fy[r](my))
            b = [P for P in bars if abs(P[0, 0] - mx) < 0.3 and P[:, 1].min() - 0.3 <= my <= P[:, 1].max() + 0.3]
            st = (max(np.ptp(P[:, 1]) for P in b) / 2 if b else 0.0)
            st = float(abs(fy[r](my - st) - val)) if st > 0.95 else 0.0      # bars shorter than the marker: 0
            edges = [float(fy[r](np.interp(mx, *zip(*sorted(zip(P[:, 0], P[:, 1])))))) for P in E]
            rows.append({"eta": round(eta, 3), "value": round(val, 2), "stat": round(st, 2),
                         "sys_hi": round(max(edges) - val, 2), "sys_lo": round(val - min(edges), 2)})
        npart, mid, mide, nch, nche, fwhm = TAB[cen]
        out["bins"][cen] = {"npart": npart, "table_mid": mid, "table_mid_err": mide, "table_nch_5p4": nch,
                            "table_nch_err": nche, "table_fwhm": fwhm, "points": rows}
        e = np.array([p["eta"] for p in rows]); v = np.array([p["value"] for p in rows])
        de = np.median(np.diff(e))
        print(f"{cen:6s} {len(rows):3d} points, eta {e.min():+.2f} .. {e.max():+.2f} (step {de:.3f}); "
              f"|eta|<1 mean {v[np.abs(e) < 1].mean():6.1f} (table {mid}), sum*deta {v.sum() * de:7.0f} (N_ch 5.4: {nch})")
    (HERE / "phobos_auau200_dndeta.json").write_text(json.dumps(out, indent=1))
    print("->", HERE / "phobos_auau200_dndeta.json")


if __name__ == "__main__":
    main(sys.argv[1])
