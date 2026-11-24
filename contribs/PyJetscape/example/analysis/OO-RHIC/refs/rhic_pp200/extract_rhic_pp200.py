#!/usr/bin/env python3
"""
refs/rhic_pp200/extract_rhic_pp200.py -- RHIC p+p 200 GeV charged-particle multiplicity references
besides UA5 (which pp_bulk.py fetches from HEPData).  None of these is on HEPData.

    python extract_rhic_pp200.py --phobos <dir of the arXiv:1011.1940 source>   -> rhic_pp200.json

  phobos_dndeta  PHOBOS, arXiv:1011.1940 (Phys. Rev. C 83, 024913), Fig. 28a: dN_ch/deta of
                 inelastic p+p at 200 GeV, -5.3 <= eta <= 5.3 in steps of 0.2 (hit counting,
                 corrected to inelastic with HIJING).  Read exactly from the Grace EPS of the source
                 (phmult_pp.eps): every point is a filled circle drawn in viewport coordinates;
                 panel (a)'s frame is x 0.15-0.55 for eta -6..6 and y 0.15-0.75 for 0..4 (from the
                 tick marks).  The figure shows no per-point errors, only a 90 % C.L. systematic
                 band (two curves around the double-Gaussian fit); `sys90_up` / `sys90_dn` are
                 the distances from each point to the band's edges, `sys` their mean / 1.645
                 (a symmetric 1 sigma).  The two eta halves
                 are separate measurements (the paper uses their difference as a systematic).
  phobos_table4  Table IV: dN_ch/deta at |eta| < 1 = 2.25 +0.37 -0.30, N_ch(|eta| < 5.4) = 19.3 +- 1.8,
                 N_ch total = 20.0 +- 1.8 (90 % C.L.), inelastic
  star_nsd       STAR, arXiv:0808.2041 (Phys. Rev. C 79, 034909), Table II: NSD p+p (BBC
                 coincidence, sigma_NSD = 30.0 +- 3.5 mb), dN_ch/deta at |eta| < 0.5 = 2.98 +- 0.34
                 (total; STAR notes the 2.43 of the p-pbar parametrisation differs by the NSD cross
                 section, 30 vs 35 mb)
  phenix_mb      PHENIX, arXiv:1509.06727 (Phys. Rev. C 93, 024901), Sec. VII: minimum-bias p+p,
                 (dN_ch/deta)/(0.5 N_part) = 2.38 +- 0.17 at mid-rapidity (|eta| < 0.35), i.e. dN_ch/deta
                 for N_part = 2 (total uncertainty)
"""
import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
FRAME_A = dict(x0=0.15, x1=0.55, eta0=-6.0, eta1=6.0, y0=0.15, y1=0.75, v0=0.0, v1=4.0)

CIRC = re.compile(r"^n\s*\n([\d.]+) ([\d.]+) ([\d.]+) [\d.]+ 0 360 EARC c\s*\n(?:[^\n]*SCS?\s*\n){0,2}fill", re.M)
PATH = re.compile(r"\nn\n((?:[\d.]+ [\d.]+ [ml]\n)+)s\n")


def to_data(x, y, f=FRAME_A):
    eta = f["eta0"] + (x - f["x0"]) / (f["x1"] - f["x0"]) * (f["eta1"] - f["eta0"])
    val = f["v0"] + (y - f["y0"]) / (f["y1"] - f["y0"]) * (f["v1"] - f["v0"])
    return round(eta, 3), round(val, 4)


def phobos(src):
    eps = (Path(src) / "phmult_pp.eps").read_text(errors="replace")
    body = eps[eps.index("%%EndProlog"):]
    f = FRAME_A
    pts = sorted({to_data(float(x), float(y)) for x, y, r in CIRC.findall(body)
                  if f["x0"] < float(x) < f["x1"] and float(r) > 0.004})
    # stroked polylines inside panel (a): the band's upper and lower curve and the fit; the band
    # curves are the ones that start and end at the outermost points' eta
    curves = []
    for m in PATH.finditer(body):
        xy = [tuple(map(float, ln.split()[:2])) for ln in m.group(1).strip().splitlines()]
        if len(xy) > 20 and all(f["x0"] <= x <= f["x1"] for x, _ in xy):
            curves.append([to_data(x, y) for x, y in xy])
    # three curves span the data: the band's upper and lower edge and the fit between them
    lo_eta, hi_eta = pts[0][0], pts[-1][0]
    band = [c for c in curves if abs(c[0][0] - lo_eta) < 0.05 and abs(c[-1][0] - hi_eta) < 0.05]
    if len(band) != 3:
        raise SystemExit(f"band curves not found ({len(curves)} curves in panel a)")
    mean_y = [sum(v for _, v in c) / len(c) for c in band]
    up, lo = band[mean_y.index(max(mean_y))], band[mean_y.index(min(mean_y))]

    def interp(c, e):
        for (e0, v0), (e1, v1) in zip(c, c[1:]):
            if e0 <= e <= e1:
                return v0 + (v1 - v0) * (e - e0) / max(e1 - e0, 1e-9)
        return c[0][1] if e < c[0][0] else c[-1][1]
    out = {"eta": [], "y": [], "sys90_up": [], "sys90_dn": [], "sys": []}
    for e, v in pts:
        u, d = interp(up, e) - v, v - interp(lo, e)
        out["eta"].append(e); out["y"].append(v)
        out["sys90_up"].append(round(u, 4)); out["sys90_dn"].append(round(d, 4))
        out["sys"].append(round(0.5 * (u + d) / 1.645, 4))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phobos", required=True, help="directory with the arXiv:1011.1940 source (phmult_pp.eps)")
    a = p.parse_args()
    ph = phobos(a.phobos)
    res = {
        "source": {"phobos": "PHOBOS, arXiv:1011.1940, Fig. 28a and Table IV",
                   "star": "STAR, arXiv:0808.2041, Table II", "phenix": "PHENIX, arXiv:1509.06727, Sec. VII"},
        "phobos_dndeta": {"norm": "inelastic", **ph},
        "phobos_table4": {"norm": "inelastic", "dndeta_eta1": 2.25, "dndeta_eta1_up90": 0.37, "dndeta_eta1_dn90": 0.30,
                          "nch_eta54": 19.3, "nch_eta54_err90": 1.8, "nch_tot": 20.0, "nch_tot_err90": 1.8},
        "star_nsd": {"norm": "NSD (sigma 30.0 +- 3.5 mb)", "eta_max": 0.5, "dndeta": 2.98, "err": 0.34},
        "phenix_mb": {"norm": "minimum bias", "eta_max": 0.35, "dndeta": 2.38, "err": 0.17},
    }
    out = HERE / "rhic_pp200.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"PHOBOS: {len(ph['eta'])} points, eta {ph['eta'][0]} .. {ph['eta'][-1]}; "
          f"|eta|<1 mean {sum(v for e, v in zip(ph['eta'], ph['y']) if abs(e) < 1) / sum(1 for e in ph['eta'] if abs(e) < 1):.3f} "
          f"(Table IV 2.25 +0.37 -0.30); band at eta 0.1: +{ph['sys90_up'][len(ph['eta']) // 2]} -{ph['sys90_dn'][len(ph['eta']) // 2]}")
    print("wrote", out)


if __name__ == "__main__":
    main()
