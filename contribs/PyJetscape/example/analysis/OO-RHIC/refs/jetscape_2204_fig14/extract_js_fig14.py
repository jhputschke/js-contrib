#!/usr/bin/env python3
"""
refs/jetscape_2204_fig14/extract_js_fig14.py -- JETSCAPE's MATTER+LBT Type 3 (alpha_s^fix 0.3, Q_sw 2 GeV,
tau0 0.6, PP19 p+p, Colorless) R_AA in central Au+Au at 200 GeV, read from the vector Fig. 14 (page 23) of
arXiv:2204.01163 (PRC 107, 054901 (2023)):

    python extract_js_fig14.py PAPER.pdf      # -> js_fig14.json (this folder)

Panels: charged jets R = 0.2, 0.3, 0.4 (|eta_jet| < 1 - R, leading charged > 5 GeV, against STAR) and
(pi+ + pi-)/2 |y| < 0.35 (against PHENIX pi0).  The model is drawn as a red step histogram with red vertical
error bars at the bin centres.  Axes from the frame and the major ticks: y 0-1 (bottom, top of the frame),
x from the first and last major tick (jets 10-30 GeV, pions 8-20 GeV).  Check: STAR's error bars read with the
same mapping fall on STAR's values (0.25, 0.25, 0.34, 0.36 for R = 0.2 above 14 GeV).
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

PANELS = {"chjet_R0.2": ("matrix(0.52927, 0, 0, -0.52927, 77.87, 223.557)", 10, 30),
          "chjet_R0.3": ("matrix(0.52927, 0, 0, -0.52927, 309.583, 223.557)", 10, 30),
          "chjet_R0.4": ("matrix(0.52927, 0, 0, -0.52927, 77.87, 396.035)", 10, 30),
          "pion": ("matrix(0.52927, 0, 0, -0.52927, 309.583, 396.035)", 8, 20)}
RED, BLACK = "rgb(100%, 0%, 0%)", "rgb(0%, 0%, 0%)"


def panel(ps, tr, x0, x1):
    C = [p for p in ps if p["tr"] == tr]
    blk = [p["P"] for p in C if p["stroke"] == BLACK and p["sw"] == "1.5" and len(p["P"]) == 2]
    vert = [P for P in blk if abs(P[0, 0] - P[1, 0]) < 1e-3]
    frame_v = [P for P in vert if np.ptp(P[:, 1]) > 100]
    xl, xr = min(P[0, 0] for P in frame_v), max(P[0, 0] for P in frame_v)
    yb, yt = max(frame_v[0][:, 1]), min(frame_v[0][:, 1])       # local y grows downwards: bottom = 0, top = 1
    major = sorted({round(P[0, 0], 2) for P in vert if abs(max(P[:, 1]) - yb) < 0.2 and 3.8 < np.ptp(P[:, 1]) < 4.6})
    fx = lambda u: x0 + (np.asarray(u) - major[0]) * (x1 - x0) / (major[-1] - major[0])
    fy = lambda v: (yb - np.asarray(v)) / (yb - yt)
    red = [p["P"] for p in C if p["stroke"] == RED]
    step = max(red, key=len)
    err = [P for P in red if len(P) == 2 and abs(P[0, 0] - P[1, 0]) < 1e-3]
    pts = []
    for a, b in zip(step[0::2], step[1::2]):                      # horizontal pieces of the step line
        lo, hi = float(fx(min(a[0], b[0]))), float(fx(max(a[0], b[0])))
        if lo < float(fx(xl)) - 0.01 or hi > float(fx(xr)) + 0.01:  # steps outside the frame are clipped
            continue
        xc = 0.5 * (a[0] + b[0])
        e = [P for P in err if abs(P[0, 0] - xc) < 0.6]
        lo_e = hi_e = 0.0
        val = float(fy(a[1]))
        if e:
            lo_e, hi_e = val - float(fy(max(e[0][:, 1]))), float(fy(min(e[0][:, 1]))) - val
        pts.append({"lo": round(lo * 2) / 2, "hi": round(hi * 2) / 2, "value": round(val, 4),
                    "err_lo": round(lo_e, 4), "err_hi": round(hi_e, 4)})
    return pts


def main(pdf):
    with tempfile.TemporaryDirectory() as d:
        svg = f"{d}/p23.svg"
        subprocess.run(["pdftocairo", "-svg", "-f", "23", "-l", "23", pdf, svg], check=True)
        ps = X.paths(svg)
    res = {"source": "JETSCAPE, arXiv:2204.01163 (PRC 107, 054901), Fig. 14, read from the vector figure",
           "model": "MATTER+LBT, q-hat Type 3 (HTL running x f(Q^2)), alpha_s^fix 0.3, Q_sw 2 GeV, tau0 0.6 fm/c, "
                    "T_c 160 MeV; TRENTo + free streaming + VISHNU; PP19 p+p, Colorless; no retuning at RHIC",
           "observables": {"chjet_R*": "charged-jet R_AA, anti-kT, |eta_jet| < 1 - R, p_T,lead,ch > 5 GeV, Au+Au 0-10 %",
                           "pion": "(pi+ + pi-)/2 R_AA, |y| < 0.35, Au+Au 0-10 % (compared with PHENIX pi0)"},
           "panels": {}}
    for k, (tr, x0, x1) in PANELS.items():
        res["panels"][k] = panel(ps, tr, x0, x1)
        print(f"{k:11s} " + "  ".join(f"{q['lo']:g}-{q['hi']:g}:{q['value']:.3f}(+{q['err_hi']:.3f}/-{q['err_lo']:.3f})" for q in res["panels"][k]))
    (HERE / "js_fig14.json").write_text(json.dumps(res, indent=1))
    print("->", HERE / "js_fig14.json")


if __name__ == "__main__":
    main(sys.argv[1])
