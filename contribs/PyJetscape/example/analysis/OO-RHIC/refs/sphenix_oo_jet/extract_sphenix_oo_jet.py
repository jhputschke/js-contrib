#!/usr/bin/env python3
"""
refs/sphenix_oo_jet/extract_sphenix_oo_jet.py -- sPHENIX's preliminary O+O jet R_AA and R_CP
(sPH-CONF-JET-2026-02, June 2026; R = 0.4, |eta| < 0.7, 19-60 GeV, MBD classes) read from the
note's vector figures: the note has no value tables, so the points, statistical bars and
systematic boxes are taken from the PDF's drawing paths (pdftocairo -svg), mapped to data
coordinates with the axes' major ticks.  Exact to the drawing precision, not digitised by eye.

    python extract_sphenix_oo_jet.py NOTE.pdf     # -> sphenix_oo_jet_prelim.json (this folder)

Panels: Fig. 6 left (R_AA 0-100 %), Fig. 7 (R_CP vs 60-80 % and vs 40-60 %); Fig. 5 (R_AA per
class) not yet.  The note's model curves ("theory"): Fig. 6 left, the no-quenching nPDF baseline
(MadGraph5_aMC@NLO + Pythia8 with nNNPDF3.0, EPPS21, TUJU21; Gebhard, Mazeliauskas, Takacs,
arXiv:2410.22405); Fig. 6 right, Figs. 9 and 10, JEWEL 2.4.0 (CT10 NLO, no nPDF, T_i 0.26 GeV in 0-20 %
scaled with <N_part>^(1/3), recoils off and on) for R_AA 0-100 % and R_CP; Fig. 8, Angantyr's
centrality-bias factors for R_CP (arXiv:2507.03603's approach).  Per curve the drawn vertices: the
central line's value and the band's lower and upper edge (the band is the model's statistical
uncertainty as drawn; for the nPDFs the 68 % CL band, no central line).  Per point: the value (the marker's horizontal bar), the statistical bar's extent
and the asymmetric systematic box.  The N_coll / T_AA normalisation boxes at unity are not points
(excluded by their width).  Caveat: the lowest pT bins show the same statistical bar, +-0.017 to
0.018, in every panel; that is probably the drawing's minimum (the bar is drawn outside the marker)
rather than the error, so there the statistical errors are upper bounds.  PRELIMINARY: for
comparison only.
"""
import json
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NS = "{http://www.w3.org/2000/svg}"
COLORS = {"rgb(59.999084%, 0%, 0%)": "red", "rgb(19.999695%, 19.999695%, 100%)": "blue",
          "rgb(0%, 59.999084%, 0%)": "green", "rgb(0%, 0%, 0%)": "black"}


def matrix(s):
    if not s:
        return np.eye(3)
    m = re.match(r"matrix\(([^)]*)\)", s)
    a, b, c, d, e, f = (float(x) for x in m.group(1).replace(",", " ").split())
    return np.array([[a, c, e], [b, d, f], [0, 0, 1]])


def points(d, T):
    nums = re.findall(r"[MLCZHV]|-?[\d.]+(?:e-?\d+)?", d)
    pts, cur, i, cmd = [], None, 0, None
    while i < len(nums):
        t = nums[i]
        if t in "MLCZHV":
            cmd = t; i += 1
            if t == "Z":
                continue
        if cmd in ("M", "L"):
            cur = (float(nums[i]), float(nums[i + 1])); i += 2
        elif cmd == "C":
            cur = (float(nums[i + 4]), float(nums[i + 5])); i += 6
        elif cmd == "H":
            cur = (float(nums[i]), cur[1]); i += 1
        elif cmd == "V":
            cur = (cur[0], float(nums[i])); i += 1
        else:
            i += 1; continue
        pts.append(cur)
    if not pts:
        return np.zeros((0, 2))
    P = np.c_[np.asarray(pts), np.ones(len(pts))] @ T.T
    return P[:, :2]


def paths(svg):
    out = []
    for el in ET.parse(svg).getroot().iter(NS + "path"):
        a = el.attrib
        if "d" not in a or el.get("id"):
            continue
        P = points(a["d"], matrix(a.get("transform")))
        if len(P):
            out.append({"P": P, "fill": a.get("fill"), "stroke": a.get("stroke"), "sw": a.get("stroke-width"),
                        "rule": a.get("fill-rule"), "tr": a.get("transform")})
    return out


def axes(ps, tr):
    """Frame box and the major ticks of the panel drawn with transform tr: (x0, x1, y0, y1) in page
    coordinates and the tick positions along each axis."""
    blk = [p for p in ps if p["tr"] == tr and p["stroke"] == "rgb(0%, 0%, 0%)" and p["fill"] == "none" and p["sw"] == "0.75"]
    big = max((p["P"] for p in blk if len(p["P"]) >= 4), key=lambda P: np.ptp(P[:, 0]) * np.ptp(P[:, 1]))
    x0, x1, y0, y1 = big[:, 0].min(), big[:, 0].max(), big[:, 1].min(), big[:, 1].max()   # page y grows down
    # ticks: consecutive point pairs of the axis polylines (ROOT draws an axis and its ticks as one path)
    segs = [p["P"][i:i + 2] for p in blk for i in range(len(p["P"]) - 1)]
    vt = [s for s in segs if abs(s[0, 0] - s[1, 0]) < 1e-3 and abs(max(s[:, 1]) - y1) < 0.5
          and 0.3 < np.ptp(s[:, 1]) < 0.1 * (y1 - y0)]
    ht = [s for s in segs if abs(s[0, 1] - s[1, 1]) < 1e-3 and abs(min(s[:, 0]) - x0) < 0.5
          and 0.3 < np.ptp(s[:, 0]) < 0.1 * (x1 - x0)]
    lv = max(np.ptp(s[:, 1]) for s in vt)
    lh = max(np.ptp(s[:, 0]) for s in ht)
    xt = sorted({round(s[0, 0], 2) for s in vt if np.ptp(s[:, 1]) > 0.8 * lv})
    yt = sorted({round(s[0, 1], 2) for s in ht if np.ptp(s[:, 0]) > 0.8 * lh}, reverse=True)
    return (x0, x1, y0, y1), xt, yt


def mapper(ticks, values):
    t, v = np.asarray(ticks), np.asarray(values[:len(ticks)])
    a, b = np.polyfit(t, v, 1)
    return lambda u: a * np.asarray(u) + b


def panel(ps, tr, xvals, yvals, colors, ymax_data):
    box, xt, yt = axes(ps, tr)
    fx, fy = mapper(xt, xvals), mapper(yt, yvals)
    res = {}
    for col in colors:
        cname = [k for k, v in COLORS.items() if v == col][0]
        boxes = [p["P"] for p in ps if p["tr"] == tr and p["rule"] == "evenodd" and p["stroke"] == cname]
        segs = [p["P"] for p in ps if p["tr"] == tr and p["fill"] == "none" and p["stroke"] == cname
                and p["sw"] == "1.5" and len(p["P"]) == 2]
        rows = []
        for B in boxes:
            xl, xh = sorted(fx([B[:, 0].min(), B[:, 0].max()]))
            bl, bh = sorted(fy([B[:, 1].min(), B[:, 1].max()]))
            if xh - xl < 2.0 or (bl + bh) / 2 > ymax_data:               # N_coll boxes at unity, legend
                continue
            inbin = [S for S in segs if B[:, 0].min() - 0.5 <= S[:, 0].min() and S[:, 0].max() <= B[:, 0].max() + 0.5]
            hor = [S for S in inbin if abs(S[0, 1] - S[1, 1]) < 1e-3 and np.ptp(S[:, 0]) > 0.5]
            ver = [S for S in inbin if abs(S[0, 0] - S[1, 0]) < 1e-3 and np.ptp(S[:, 1]) > 1e-3]
            if not hor:
                continue
            val = float(fy(np.median([S[0, 1] for S in hor])))           # the marker's horizontal bar
            ys = fy(np.concatenate([S[:, 1] for S in ver])) if ver else np.array([val])
            rows.append({"lo": round(float(xl), 2), "hi": round(float(xh), 2), "value": round(val, 4),
                         "stat_lo": round(float(val - ys.min()), 4), "stat_hi": round(float(ys.max() - val), 4),
                         "sys_lo": round(float(val - bl), 4), "sys_hi": round(float(bh - val), 4)})
        res[col] = sorted(rows, key=lambda r: r["lo"])
    return res


def theory(ps, tr, xv, yv, curves, keep=None):
    """Model curves of the panel drawn with transform tr.  curves: {name: (band fill, line stroke)}.
    The band is an untransformed fill-only polygon inside the frame, the line a stroked polyline of the
    panel.  Per band vertex x: the band's lower and upper edge, and the line's value there (None when the
    line is the band's outline, as for the nPDFs).  keep: (lo, hi), drop vertices outside lo < x < hi
    (JEWEL's curves are clipped at the axis range 19-60 GeV: the end vertices are not model points)."""
    (x0, x1, y0, y1), xt, yt = axes(ps, tr)
    fx, fy = mapper(xt, xv), mapper(yt, yv)
    out = {}
    for name, (fill, stroke) in curves.items():
        bands = [p["P"] for p in ps if p["tr"] is None and p["fill"] == fill and p["stroke"] is None and len(p["P"]) > 8
                 and p["P"][:, 0].min() >= x0 - 1 and p["P"][:, 0].max() <= x1 + 1
                 and p["P"][:, 1].min() >= y0 - 1 and p["P"][:, 1].max() <= y1 + 1]
        if len(bands) != 1:
            sys.exit(f"extract_sphenix_oo_jet.py: {len(bands)} bands of {fill} in panel {tr}")
        bx, by = fx(bands[0][:, 0]), fy(bands[0][:, 1])
        lines = [p["P"] for p in ps if p["tr"] == tr and p["fill"] == "none" and p["stroke"] == stroke and len(p["P"]) > 2]
        L = max(lines, key=len) if lines else None
        lx, ly = (fx(L[:, 0]), fy(L[:, 1])) if L is not None else (None, None)
        central = lx is not None and len(np.unique(lx.round(1))) == len(lx)
        rows = []
        for u in np.unique(bx.round(1)):
            if keep and not keep[0] < u < keep[1]:
                continue
            m = np.isclose(bx.round(1), u)
            o = np.argsort(lx) if central else None
            rows.append({"x": float(u), "value": round(float(np.interp(u, lx[o], ly[o])), 4) if central else None,
                         "lo": round(float(by[m].min()), 4), "hi": round(float(by[m].max()), 4)})
        out[name] = rows
    return out


def panels(ps, ty):
    """The transforms of the panels whose frame is at page offset ty, left to right."""
    trs = [t for t in {p["tr"] for p in ps if p["tr"]} if abs(matrix(t)[1, 2] - ty) < 0.01
           and sum(p["tr"] == t for p in ps) > 100]
    return sorted(trs, key=lambda t: matrix(t)[0, 2])


R60 = "rgb(59.999084%, 0%, 0%)"
B60 = "rgb(19.999695%, 19.999695%, 100%)"
G60 = "rgb(0%, 59.999084%, 0%)"
CLS_60_80 = {"0-20/60-80": (R60, R60), "20-40/60-80": (B60, B60), "40-60/60-80": (G60, G60)}
CLS_40_60 = {"0-20/40-60": (R60, R60), "20-40/40-60": (B60, B60)}
JEWEL_KEEP = (20.0, 59.9)


def main(pdf):
    out = {"source": "sPHENIX, sPH-CONF-JET-2026-02 (preliminary, June 2026), "
                     "https://www.sphenix.bnl.gov/PublicResults/sPH-CONF-JET-2026-02",
           "method": "points, statistical bars and systematic boxes read from the note's vector figures "
                     "(pdftocairo -svg), mapped with the axes' major ticks; preliminary, for comparison only",
           "jets": "anti-kT R = 0.4, |eta| < 0.7, calorimeter jets, unfolded; MBD centrality",
           "caveat": "the lowest pT bins show the same statistical bar (+-0.017-0.018) in every panel: probably "
                     "the drawing's minimum, so an upper bound there; values from the marker's bar, systematics "
                     "asymmetric from the boxes",
           "ncoll": {"0-20": [22.49, 0.94], "20-40": [12.16, 0.66], "40-60": [6.02, 0.33], "60-80": [2.92, 0.15]}}
    th = out["theory"] = {
        "about": "the note's model curves, read from its vector figures: per vertex x the central line's value and "
                 "the band's edges lo/hi (value null: band only).  Preliminary note, for comparison only.",
        "npdf_raa_0_100": "Fig. 6 left: no-quenching baseline, MadGraph5_aMC@NLO + Pythia8 with three nPDFs "
                          "(68 % CL bands), Gebhard, Mazeliauskas, Takacs, JHEP 04 (2025) 034, arXiv:2410.22405",
        "jewel": "Figs. 6 right, 9, 10: JEWEL 2.4.0, default configuration, CT10 NLO, no nPDF; T_i = 0.26 GeV in "
                 "0-20 % (from Habich, Nagle, Romatschke, EPJC 75 (2015) 15), scaled with <N_part>^(1/3); recoils off "
                 "and on.  Points at the bin centres 22.5-57.5 GeV (5 GeV bins); the drawn ends at 19 and 60 GeV are "
                 "clipping and dropped",
        "angantyr": "Fig. 8: centrality-bias factors on R_CP from Angantyr (Pythia 8; Bierlich et al., JHEP 10 (2018) "
                    "134), the approach of Park et al., PRC 112 (2025) 064916, arXiv:2507.03603; vertices at 19-29 GeV "
                    "in 2 GeV steps, then 32.5-57.5 GeV"}
    with tempfile.TemporaryDirectory() as d:
        for page, figs in ((10, ("fig6", "fig7")), (11, ("fig8", "fig9")), (12, ("fig10",))):
            svg = f"{d}/p{page}.svg"
            subprocess.run(["pdftocairo", "-svg", "-f", str(page), "-l", str(page), pdf, svg], check=True)
            ps = paths(svg)
            trs = sorted({p["tr"] for p in ps if p["tr"]}, key=lambda t: (-matrix(t)[1, 2], matrix(t)[0, 2]))
            if page == 10:
                f7 = [t for t in trs if abs(matrix(t)[1, 2] - 519.991) < 0.01]
                f6 = [t for t in trs if abs(matrix(t)[1, 2] - 265.626) < 0.01]
                f7.sort(key=lambda t: matrix(t)[0, 2]); f6.sort(key=lambda t: matrix(t)[0, 2])
                yv = [round(0.2 * k, 1) for k in range(10)]
                xv = list(range(20, 65, 5))
                out["rcp_vs_60_80"] = panel(ps, f7[0], xv, yv, ("red", "blue", "green"), 1.3)
                out["rcp_vs_60_80"]["classes"] = {"red": "0-20/60-80", "blue": "20-40/60-80", "green": "40-60/60-80"}
                out["rcp_vs_40_60"] = panel(ps, f7[1], xv, yv, ("red", "blue"), 1.3)
                out["rcp_vs_40_60"]["classes"] = {"red": "0-20/40-60", "blue": "20-40/40-60"}
                out["raa_0_100"] = panel(ps, f6[0], xv, yv, ("black",), 1.3)
                th["npdf_raa_0_100"] = theory(ps, f6[0], xv, yv, {
                    "nNNPDF3.0": ("rgb(0%, 79.998779%, 79.998779%)", "rgb(0%, 59.999084%, 59.999084%)"),
                    "EPPS21": ("rgb(79.998779%, 39.99939%, 100%)", "rgb(39.99939%, 0%, 79.998779%)"),
                    "TUJU21": ("rgb(100%, 59.999084%, 0%)", "rgb(100%, 59.999084%, 19.999695%)")})
                th["jewel_raa_0_100"] = theory(ps, f6[1], xv, yv, {
                    "recoils_off": ("rgb(0%, 0%, 79.998779%)",) * 2, "recoils_on": ("rgb(79.998779%, 0%, 0%)",) * 2},
                    keep=JEWEL_KEEP)
            elif page == 11:
                f8, f9 = panels(ps, 287.974), panels(ps, 606.959)
                th["angantyr_bias_rcp_vs_60_80"] = theory(ps, f8[0], xv, yv, CLS_60_80)
                th["angantyr_bias_rcp_vs_40_60"] = theory(ps, f8[1], xv, yv, CLS_40_60)
                th["jewel_rcp_vs_60_80"] = {"recoils_off": theory(ps, f9[0], xv, yv, CLS_60_80, keep=JEWEL_KEEP),
                                            "recoils_on": theory(ps, f9[1], xv, yv, CLS_60_80, keep=JEWEL_KEEP)}
            elif page == 12:
                f10 = panels(ps, 447.466)
                th["jewel_rcp_vs_40_60"] = {"recoils_off": theory(ps, f10[0], xv, yv, CLS_40_60, keep=JEWEL_KEEP),
                                            "recoils_on": theory(ps, f10[1], xv, yv, CLS_40_60, keep=JEWEL_KEEP)}
    (HERE / "sphenix_oo_jet_prelim.json").write_text(json.dumps(out, indent=1))
    def show(name, d):
        for k, v in d.items():
            if isinstance(v, dict):
                show(f"{name} {k}", v)
            else:
                print(f"  {name} {k}: " + " ".join(f"{r['x']:.1f}:{'-' if r['value'] is None else format(r['value'], '.3f')}"
                                                 f"[{r['lo']:.3f},{r['hi']:.3f}]" for r in v))
    print("theory")
    for k, v in th.items():
        if isinstance(v, dict):
            show(k, v)
    for k in ("rcp_vs_60_80", "rcp_vs_40_60", "raa_0_100"):
        print(k)
        for col, rows in out[k].items():
            if col == "classes":
                continue
            print(f"  {out[k].get('classes', {}).get(col, col)}:")
            for r in rows:
                print(f"    {r['lo']:5.1f}-{r['hi']:<5.1f} {r['value']:.3f} stat +{r['stat_hi']:.3f}/-{r['stat_lo']:.3f} "
                      f"sys +{r['sys_hi']:.3f}/-{r['sys_lo']:.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
