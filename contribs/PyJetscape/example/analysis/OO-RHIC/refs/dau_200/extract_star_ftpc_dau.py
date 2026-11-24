#!/usr/bin/env python3
"""
refs/dau_200/extract_star_ftpc_dau.py -- STAR d+Au 200 GeV charged dN/deta in the FTPCs
(2.5 < |eta| < 4 detectors; points at 3.0 <= |eta| <= 3.5) and the TPC (|eta| < 1), read exactly
from the vector figure dAu_dndeta_star_phobos_brahms.pdf (ROOT 3.03, 2004-05, "dN/deta FTPC" from the
STAR d+Au FTPC analysis; it also shows PHOBOS and BRAHMS, not read here).  The figure is not public
and not in the repository (.gitignore); star_dau_ftpc.json, its output, is.

    python3 -I extract_star_ftpc_dau.py      -> star_dau_ftpc.json   (needs mutool, the figure, and
                                                star_ins656934_Table2.yaml from dau_bulk.py fetch)

Classes 0-20, 20-40, 40-100 % (percentiles of the total d+Au cross section) and minimum bias.  The
centrality estimator differs by region: the |eta| < 1 (TPC) points use the Au-going FTPC multiplicity
(-3.8 < eta < -2.8, as in nucl-ex/0408016), the FTPC points the charged multiplicity at |eta| < 0.5
(as in that analysis).  Each point is a filled up-triangle drawn in
the PDF's coordinates; axis frame from the tick marks (eta -4 at x = 820, +4 at 4850; dN/deta 0 at
y = 425, 25 at 3462.5; units of 0.1 pt).  The marker's position is its bounding-box centre (ROOT's
triangle).  Error bars are the vertical segments of the same colour at the marker's x; ROOT omits a bar
shorter than the marker (half-height 0.40 in dN/deta): such points get err = null and
err_max = 0.40.  err is the mean of the up and down extent (the two halves differ by the marker
geometry, not by the data).

Check against STAR's published |eta| < 1 values (HEPData ins656934 Table 2): the figure's 0-20 and
20-40 % points agree within 0.7 % (rms 0.5 %); 40-100 % and minimum bias are 13 % and 9 % above the
published ones (this figure predates the final normalisation of those classes).  `norm_to_published`
is the published / figure ratio at |eta| < 1 per class; use it for the FTPC points of 40-100 % and
minimum bias (the same normalisation applies to all eta).
"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
PDF = HERE / "dAu_dndeta_star_phobos_brahms.pdf"
HEP = HERE / "star_ins656934_Table2.yaml"
X = lambda x: -4 + (x - 820) / 503.75
Y = lambda y: (y - 425) / 121.5
COLS = {(0.0, 0.0, 0.0): "0-20%", (1.0, 0.0, 0.0): "20-40%", (0.0, 1.0, 0.0): "40-100%", (0.0, 0.0, 1.0): "minbias"}


def items(pdf):
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "clean.pdf"
        subprocess.run(["mutool", "clean", "-d", str(pdf), str(out)], check=True)
        txt = out.read_bytes().decode("latin-1")
    s = txt[txt.index("stream", txt.index("8 0 obj")) + 6:]
    s = s[:s.index("endstream")]
    toks = re.findall(r"\((?:\\.|[^\\)])*\)|\[[^\]]*\]|/\S+|[-+]?\d*\.?\d+|[A-Za-z'\"*]+", s)
    fill = stroke = (0.0, 0.0, 0.0); st = []; path = []; cur = []; res = []
    for t in toks:
        try:
            st.append(float(t)); continue
        except ValueError:
            pass
        if t == "rg": fill = tuple(st[-3:])
        elif t == "RG": stroke = tuple(st[-3:])
        elif t == "g": fill = (st[-1],) * 3
        elif t == "G": stroke = (st[-1],) * 3
        elif t == "m":
            if cur: path.append(cur)
            cur = [tuple(st[-2:])]
        elif t == "l": cur.append(tuple(st[-2:]))
        elif t in ("f", "f*", "F", "S", "s", "B", "b", "n", "B*", "b*"):
            if cur: path.append(cur); cur = []
            res.append((t, tuple(map(float, fill if t[0] in "fFBb" else stroke)), path)); path = []
        st = []
    return res


def main():
    it = items(PDF)
    out = {"source": "figure dAu_dndeta_star_phobos_brahms.pdf (STAR d+Au FTPC analysis, ROOT 2004-05; not in the repository); see extract_star_ftpc_dau.py",
           "classes": {}}
    for col, lab in COLS.items():
        tri = [p[0] for op, c, p in it if op == "f*" and c == col and p and len(p[0]) == 3]
        segs = [p[0] for op, c, p in it if op == "S" and c == col and p and len(p[0]) == 2 and abs(p[0][0][0] - p[0][1][0]) < 0.6]
        pts = []
        for t in tri:
            xs, ys = [x for x, _ in t], [y for _, y in t]
            cx, cy, top, bot = 0.5 * (max(xs) + min(xs)), 0.5 * (max(ys) + min(ys)), max(ys), min(ys)
            if X(cx) > 2 and Y(cy) > 21:
                continue                                     # the legend marker
            yy = [y for s_ in segs if abs(s_[0][0] - cx) < 2.0 for _, y in s_]
            up = max([y for y in yy if y > top], default=None)
            dn = min([y for y in yy if y < bot], default=None)
            ext = [v for v in ((Y(up) - Y(cy)) if up else None, (Y(cy) - Y(dn)) if dn else None) if v is not None]
            pts.append({"eta": round(X(cx), 3), "y": round(Y(cy), 3),
                        "err": round(sum(ext) / len(ext), 3) if ext else None, "err_max": round(Y(top) - Y(cy), 3)})
        out["classes"][lab] = sorted(pts, key=lambda p: p["eta"])
    # normalisation against the published |eta| < 1 values
    d = yaml.safe_load(HEP.read_text())
    eta = [(v["low"] + v["high"]) / 2 for v in d["independent_variables"][0]["values"]]
    out["norm_to_published"] = {}
    for dv in d["dependent_variables"]:
        q = dv["qualifiers"][0]["value"]
        lab = "minbias" if "minimum bias" in q else q.split()[-1]
        fig = [p for p in out["classes"][lab] if abs(p["eta"]) < 1.01]
        fe, fy = [p["eta"] for p in fig], [p["y"] for p in fig]
        import numpy as np
        r = np.array([float(v["value"]) for v in dv["values"]]) / np.interp(eta, fe, fy)
        out["norm_to_published"][lab] = round(float(r.mean()), 4)
    path = HERE / "star_dau_ftpc.json"
    path.write_text(json.dumps(out, indent=1))
    for lab, pts in out["classes"].items():
        au = [p for p in pts if p["eta"] < -2]; dd = [p for p in pts if p["eta"] > 2]
        print(f"{lab:8s} {len(pts)} points (FTPC-Au {len(au)}, TPC {len(pts) - len(au) - len(dd)}, FTPC-d {len(dd)}); "
              f"published/figure at |eta| < 1: {out['norm_to_published'][lab]:.4f}")
    print("wrote", path)


if __name__ == "__main__":
    sys.exit(main())
