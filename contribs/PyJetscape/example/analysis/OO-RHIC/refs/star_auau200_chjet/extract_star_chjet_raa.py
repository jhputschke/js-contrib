#!/usr/bin/env python3
"""
refs/star_auau200_chjet/extract_star_chjet_raa.py -- STAR's charged-jet R_AA^Pythia in Au+Au at 200 GeV,
0-10 % and 60-80 %, anti-kT R = 0.2, 0.3, 0.4, |eta_jet| < 1 - R, leading charged track > 5 GeV/c,
read from the vector Fig. 17 (page 22) of arXiv:2006.00582 (PRC 102, 054913 (2020)):

    python extract_star_chjet_raa.py PAPER.pdf     # -> star_chjet_raa.json (this folder)

The p+p reference of this R_AA is PYTHIA (Perugia), not a measurement.  Points: "unbiased" (blue;
above the dashed line, where the leading-track bias is small) and "biased" (grey).  Per point: the
bin from the horizontal error bar, the value at its height, the statistical error from the vertical
bar (0 if hidden by the marker), the correlated (magenta box) and shape (grey band) systematics as
the boxes' extents.  Normalisation uncertainties (T_AA, PYTHIA) are not per point.  Axes from the
ticks: log y with majors at 0.1 and 1; x 0-35 GeV in steps of 5 per panel.
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

ROWS = {"0-10": "matrix(0.85135, 0, 0, -0.85135, 66.703, 657.665)",
        "60-80": "matrix(0.85135, 0, 0, -0.85135, 66.703, 502.575)"}
RADII = (0.2, 0.3, 0.4)
BLACK, BLUE, GREY = "rgb(0%, 0%, 0%)", "rgb(0%, 0%, 100%)", "rgb(59.999084%, 59.999084%, 59.999084%)"
MAGENTA, SHAPE = "rgb(100%, 0%, 100%)", "rgb(79.998779%, 79.998779%, 79.998779%)"
THIN = "0.140953"


def row(ps, tr):
    C = [p for p in ps if p["tr"] == tr]
    frames = sorted([p["P"] for p in C if p["stroke"] == BLACK and p["sw"] == "0.75" and len(p["P"]) == 5
                     and np.ptp(p["P"][:, 0]) > 100], key=lambda P: P[:, 0].min())
    if len(frames) != 3:
        sys.exit(f"{tr}: {len(frames)} panel frames")
    y0, y1 = frames[0][:, 1].min(), frames[0][:, 1].max()           # page y grows down
    thin = [p["P"] for p in C if p["stroke"] == BLACK and p["sw"] == THIN and len(p["P"]) == 2]
    # y: log axis, the two long ticks on the left axis are 1 (upper) and 0.1 (lower)
    xl = frames[0][:, 0].min()
    yt = sorted({round(P[0, 1], 2) for P in thin if abs(P[0, 1] - P[1, 1]) < 1e-3
                 and abs(P[:, 0].min() - xl) < 0.5 and 3.5 < np.ptp(P[:, 0]) < 5})
    if len(yt) != 2:
        sys.exit(f"{tr}: y major ticks {yt}")
    def fy(y):
        return 10 ** (np.log10(1.0) - (np.asarray(y) - yt[0]) / (yt[1] - yt[0]))
    out = {}
    for k, F in enumerate(frames):
        xa, xb = F[:, 0].min(), F[:, 0].max()
        xt = sorted({round(P[0, 0], 2) for P in thin if abs(P[0, 0] - P[1, 0]) < 1e-3 and abs(P[:, 1].max() - y1) < 0.5
                     and xa - 0.3 <= P[0, 0] <= xb + 0.3 and 3.0 < np.ptp(P[:, 1]) < 4.5})
        xt = [x for x in xt if xa + 1 < x < xb - 1]                   # inner major ticks, 5 GeV apart
        step = np.median(np.diff(xt))                                  # every panel starts at 0 GeV
        fx = lambda u: 5.0 * (np.asarray(u) - xa) / step
        pts = []
        # the blue series holds every bin; the bins left of STAR's dashed line are drawn grey on top
        # (the same values): "biased" = a grey bar at the same height and place
        grey = [p["P"] for p in C if p["stroke"] == GREY and p["sw"] == "0.75" and len(p["P"]) == 2
                and xa < p["P"][:, 0].mean() < xb and abs(p["P"][0, 1] - p["P"][1, 1]) < 1e-3]
        for col in (BLUE,):
            seg = [p["P"] for p in C if p["stroke"] == col and p["sw"] == "0.75" and len(p["P"]) == 2
                   and xa < p["P"][:, 0].mean() < xb]
            hz = [P for P in seg if abs(P[0, 1] - P[1, 1]) < 1e-3]
            vt = [P for P in seg if abs(P[0, 0] - P[1, 0]) < 1e-3]
            groups = {}
            for P in hz:
                groups.setdefault(round(P[0, 1], 1), []).append(P)
            for yv, G in groups.items():
                lo, hi = min(P[:, 0].min() for P in G), max(P[:, 0].max() for P in G)
                if hi - lo < 6:                                       # marker strokes, not a bin
                    continue
                xc = 0.5 * (lo + hi)
                V = [P for P in vt if abs(P[0, 0] - xc) < 0.6]
                ys = [P[:, 1] for P in V]
                stat_lo = stat_hi = 0.0
                val = float(fy(yv))
                if ys:
                    ymin = min(a.min() for a in ys); ymax = max(a.max() for a in ys)
                    if ymax - ymin > 2.5:
                        stat_hi, stat_lo = float(fy(ymin)) - val, val - float(fy(ymax))
                boxes = {}
                for name, sel in (("sys_corr", lambda p: p["stroke"] == MAGENTA and p["fill"] == "none"),
                                  ("sys_shape", lambda p: p["fill"] == SHAPE)):
                    B = [p["P"] for p in C if sel(p) and p["P"][:, 0].min() - 0.5 <= xc <= p["P"][:, 0].max() + 0.5
                         and np.ptp(p["P"][:, 0]) > 4 and xa < p["P"][:, 0].mean() < xb
                         and p["P"][:, 1].min() - 0.5 <= yv <= p["P"][:, 1].max() + 0.5]
                    if B:
                        b = min(B, key=lambda P: np.ptp(P[:, 0]))
                        boxes[name + "_hi"] = round(float(fy(b[:, 1].min())) - val, 4)
                        boxes[name + "_lo"] = round(val - float(fy(b[:, 1].max())), 4)
                if not boxes:                                         # legend entries have no systematic box
                    continue
                label = "biased" if any(abs(P[0, 1] - yv) < 0.15 and lo - 0.5 <= P[:, 0].mean() <= hi + 0.5 for P in grey) else "unbiased"
                pts.append({"lo": round(2 * float(fx(lo))) / 2, "hi": round(2 * float(fx(hi))) / 2, "value": round(val, 4),
                            "stat_hi": round(stat_hi, 4), "stat_lo": round(stat_lo, 4), "kind": label, **boxes})
        out[f"R{RADII[k]}"] = sorted(pts, key=lambda q: q["lo"])
    return out


def main(pdf):
    with tempfile.TemporaryDirectory() as d:
        svg = f"{d}/p22.svg"
        subprocess.run(["pdftocairo", "-svg", "-f", "22", "-l", "22", pdf, svg], check=True)
        ps = X.paths(svg)
    res = {"source": "STAR, PRC 102, 054913 (2020), arXiv:2006.00582, Fig. 17 (read from the vector figure)",
           "observable": "R_AA^Pythia of charged jets, anti-kT, |eta_jet| < 1 - R, p_T,lead > 5 GeV/c; p+p reference from PYTHIA",
           "centrality": {}}
    for cen, tr in ROWS.items():
        res["centrality"][cen] = row(ps, tr)
        for r, pts in res["centrality"][cen].items():
            print(f"{cen:6s} {r}: " + "  ".join(f"{q['lo']:.1f}-{q['hi']:.1f}:{q['value']:.3f}{'u' if q['kind'] == 'unbiased' else 'b'}"
                                              f"(+{q['stat_hi']:.3f}/-{q['stat_lo']:.3f}; c{q.get('sys_corr_hi', 0):.3f} s{q.get('sys_shape_hi', 0):.3f})" for q in pts))
    (HERE / "star_chjet_raa.json").write_text(json.dumps(res, indent=1))
    print("->", HERE / "star_chjet_raa.json")


if __name__ == "__main__":
    main(sys.argv[1])
