#!/usr/bin/env python3
"""
refs/star_oo/extract_star_oo.py -- the STAR O+O 200 GeV bulk references, read exactly from the
vector figures of the arXiv sources (ROOT PDFs: every point is a marker path plus its error bars,
mapped to data coordinates through the axis ticks).  No HEPData exists for these figures.

    python extract_star_oo.py --flow <dir of arXiv:2510.19645 source> --jq <dir of arXiv:2604.13935v2 source>
    -> star_oo_bulk.json

  flow_fig5a  arXiv:2510.19645 Fig. 5a: p(N_ch) of minimum-bias O+O, N_ch efficiency corrected,
              0.2 < pT < 2 GeV, |eta| < 1.5 (arbitrary normalisation)
  flow_fig5b  Fig. 5b: <N_ch> vs centrality (centrality from a Glauber+NBD fit to p(N_ch))
  ea_fig1     arXiv:2604.13935 Fig. 1: raw nMIP in the EPD (2.14 < |eta| < 5.09), self-normalised,
              minimum bias (trigger-efficiency corrected) and events with a charged trigger in
              5-7, 7-10, 10-30 GeV (|eta| < 1.5); the EA class edges (dashed lines)
  ea_fig4     Fig. 4: the same distributions divided by the 5-7 GeV one
Statistical errors as drawn (vertical bars; most are smaller than the markers, then 0).
"""
import argparse
import json
import re
import zlib
from pathlib import Path


def stream(path):
    b = Path(path).read_bytes()
    out = []
    for m in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", b, re.S):
        try:
            out.append(zlib.decompress(m.group(1)))
        except zlib.error:
            pass
    return max(out, key=len).decode("latin1")


def segments(s):
    parts = re.split(r"(\S+ \S+ \S+ RG \S+ \S+ \S+ rg)", s)
    return [(tuple(round(float(v), 3) for v in parts[i].split()[:3]), parts[i + 1])
            for i in range(1, len(parts), 2)]


LINE = re.compile(r"([-\d.]+) ([-\d.]+) m ([-\d.]+) ([-\d.]+) l S")
CIRCLE = re.compile(r"([-\d.]+) ([-\d.]+) m ([-\d.]+) ([-\d.]+) ([-\d.]+) ([-\d.]+) ([-\d.]+) ([-\d.]+) c")


def binned(body, frame):
    """Points drawn as horizontal bars (bin edges, two pieces around the marker) + vertical bars."""
    x_lo, y_lo, x_hi, y_hi = frame
    hor, ver = [], []
    for a in LINE.findall(body):
        x0, y0, x1, y1 = map(float, a)
        if abs(y0 - y1) < 1e-3 and x1 > x0 and y_lo < y0 < y_hi:
            hor.append((x0, x1, y0))
        elif abs(x0 - x1) < 1e-3 and x_lo <= x0 <= x_hi:
            ver.append((x0, y0, y1))
    pts, i = [], 0
    while i + 1 < len(hor):
        a, b = hor[i], hor[i + 1]
        if abs(a[2] - b[2]) < 1e-3 and b[0] >= a[1] - 1e-3:
            lo, hi, y = a[0], b[1], a[2]
            xc = 0.5 * (a[1] + b[0])
            # error bars start at the marker edge (radius < 8 pt) and point away from it
            err = max([max(abs(y0 - y), abs(y1 - y)) for x, y0, y1 in ver
                       if abs(x - xc) < 0.05 and min(abs(y0 - y), abs(y1 - y)) < 8] or [0.0])
            pts.append((lo, hi, y, err))
            i += 2
        else:
            i += 1
    return pts


def circles(body):
    out = []
    for a in CIRCLE.findall(body):
        x0, yc, _, _, x1, _, x1b, ycb = map(float, a)
        if abs(ycb - yc) < 1e-3 and abs(x1 - x1b) < 1e-3:
            out.append((0.5 * (x0 + x1), yc))
    return out


def lin(p0, v0, p1, v1):
    return lambda p: v0 + (p - p0) * (v1 - v0) / (p1 - p0)


def logm(p0, d0, p1, d1):
    return lambda p: 10 ** (d0 + (p - p0) * (d1 - d0) / (p1 - p0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow", required=True)
    ap.add_argument("--jq", required=True)
    ap.add_argument("--out", default=str(Path(__file__).with_name("star_oo_bulk.json")))
    a = ap.parse_args()
    res = {"source": {"flow": "STAR, arXiv:2510.19645 (Fig. 5)", "jq": "STAR, arXiv:2604.13935v2 (Figs. 1, 4)"},
           "note": "read from the vector figures of the arXiv sources by extract_star_oo.py"}

    # flow Fig. 5: panel a x 0 at 51.9307, 50 at 104.23; y decades 1e-6 at 38.5118 ... 1e-2 at 227.48
    seg = segments(stream(Path(a.flow) / "fig_05.pdf"))
    blue = [b for c, b in seg if c == (0.0, 0.0, 1.0)]
    fx, fy = lin(51.9307, 0, 104.23, 50), logm(38.5118, -6, 227.48, -2)
    pa = [(fx(x), fy(y)) for x, y in circles(blue[0]) if x < 282]
    res["flow_fig5a"] = {"x": "N_ch (0.2<pT<2 GeV, |eta|<1.5, efficiency corrected)", "y": "p(N_ch), arbitrary scale",
                         "nch": [round(x, 2) for x, _ in pa], "p": [float(f"{y:.4g}") for _, y in pa]}
    # panel b: centrality 80 at 332.561, 0 at 557.07; y 10 at 96.96, 100 at 208.75
    gx, gy = lin(332.561, 80, 557.07, 0), logm(96.96, 1, 208.75, 2)
    pb = [(gx(x), gy(y)) for x, y in circles(blue[-1]) if x > 332]
    res["flow_fig5b"] = {"x": "centrality [%]", "y": "<N_ch>", "cent": [round(x, 2) for x, _ in pb],
                         "nch": [round(y, 2) for _, y in pb]}

    frame = (85.0394, 61.645, 510.236, 369.87)
    fx = lin(85.0394, 0, 132.283, 50)
    names = {(1.0, 0.702, 0.0): "trig_10_30", (0.055, 0.302, 0.651): "trig_7_10",
             (0.965, 0.2, 0.176): "trig_5_7", (0.0, 0.0, 0.0): "mb"}
    for key, fname, ylin in (("ea_fig1", "EA.pdf", lin(61.645, 0, 369.87, 6e-3)),
                             ("ea_fig4", "EA_ratio.pdf", lin(61.645, 0.4, 369.87, 1.6))):
        seg = segments(stream(Path(a.jq) / "figures" / fname))
        out = {"x": "nMIP_EPD (raw)", "y": "self-normalised entries" if key == "ea_fig1" else "ratio to 5-7 GeV"}
        for c, body in seg:
            n = names.get(c)
            if n is None or n in out:
                continue
            pts = binned(body, frame)
            if len(pts) < 5:
                continue
            out[n] = {"lo": [round(fx(p[0]), 2) for p in pts], "hi": [round(fx(p[1]), 2) for p in pts],
                      "y": [float(f"{ylin(p[2]):.5g}") for p in pts],
                      "err": [float(f"{(ylin(p[2] + p[3]) - ylin(p[2])):.3g}") for p in pts]}
        grey = [b for c, b in seg if c == (0.6, 0.6, 0.6)]
        if grey and key == "ea_fig1":
            xs = sorted({float(x0) for x0, y0, x1, y1 in LINE.findall(grey[0]) if x0 == x1})
            labels = re.findall(r"Tf \((\d+-\d+%)\)", grey[0])
            out["class_edges"] = {lab: round(fx(x), 2) for lab, x in zip(labels, xs)}
        res[key] = out
    Path(a.out).write_text(json.dumps(res, indent=1))
    for k in ("flow_fig5a", "flow_fig5b"):
        print(k, len(res[k][list(res[k])[2]]), "points")
    for k in ("ea_fig1", "ea_fig4"):
        print(k, {n: len(v["y"]) for n, v in res[k].items() if isinstance(v, dict) and "y" in v},
              res[k].get("class_edges", ""))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
