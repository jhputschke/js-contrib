#!/usr/bin/env python3
"""Minimum-bias 3dMCGlauber check of the b_max centrality cut of the Pb+Pb XMLs.

X-SCAPE selects centrality on the impact parameter only (MCGlauberWrapper samples b on
[b_min, b_max] with P(b) ~ b; cenMin/cenMax are ignored, see the <MCGlauber> comment of
../../config/PbPb_0_30_ZJet.xml).  This runs the standalone 3dMCGlb.e of an X-SCAPE build with
the XML's <MCGlauber> settings on top of the build's mcglauber.input, minimum bias (b < 20 fm,
only the event summary), and compares:
* the model's own sigma_inel (from the b distribution: every b below ~10 fm collides, so
  sigma = pi b0^2 N_all / N(b < b0)) with the 7.7 b behind b_max, and the geometric b of the
  class;
* the class defined by b with classes defined by event activity: the string count (the
  standalone code's own centrality cut), N_part, N_coll and a two-component mix.

    python centrality_check.py                       # PbPb_0_30_ZJet.xml, 4 x 50,000 events
    python centrality_check.py --xml ../../config/PbPb_0_10_jet.xml --percent 10

~8 min on the GB10 for 200,000 events (4 processes); output in data/glauber_mb/
(summary.txt)."""
import argparse
import glob
import os
import re
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
XSCAPE = os.environ.get("XSCAPE", os.path.normpath(os.path.join(HERE, *[".."] * 7)))


def make_input(xml, base, path):
    """The build's mcglauber.input with the XML's <MCGlauber> values, minimum bias."""
    s = open(xml).read()
    blk = s[s.index("<MCGlauber>"):s.index("</MCGlauber>")]
    blk = re.sub(r"<!--.*?-->", "", blk, flags=re.S)
    vals = dict(re.findall(r"<([A-Za-z_0-9]+)>\s*([^<]*?)\s*</\1>", blk))
    rename = {"projectile": "Projectile", "target": "Target", "sqrts": "roots"}
    over = {rename.get(k, k): v for k, v in vals.items() if k != "generateOnlyPositions"}
    over.update({"b_min": "0.", "b_max": "20.", "cenMin": "0.", "cenMax": "100.",
                 "only_event_statistics": "1", "batch_density_output": "0",
                 "batch_eccentricity_output": "0", "batch_2Ddensity_output": "0",
                 "outputInitialEst": "0"})
    out, seen = [], set()
    for line in open(base).read().splitlines():
        m = re.match(r"\s*([A-Za-z_0-9]+)\s+\S+", line)
        if m and m.group(1) in over:
            out.append(f"{m.group(1)} {over[m.group(1)]}")
            seen.add(m.group(1))
        else:
            out.append(line)
    out += [f"{k} {v}" for k, v in over.items() if k not in seen]
    open(path, "w").write("\n".join(out) + "\n")
    return float(vals["b_max"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--xml", default=os.path.join(HERE, "..", "..", "config", "PbPb_0_30_ZJet.xml"))
    ap.add_argument("--percent", type=float, default=30.0, help="the class the XML's b_max means")
    ap.add_argument("--build", default=os.path.join(XSCAPE, "build_gpu"))
    ap.add_argument("--events", type=int, default=50000, help="per seed")
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13, 14])
    ap.add_argument("--out", default=os.path.join(HERE, "data", "glauber_mb"))
    ap.add_argument("--sigma-assumed", type=float, default=7.7, help="b, behind the XML's b_max")
    a = ap.parse_args()
    exe = os.path.join(a.build, "external_packages", "3dMCGlauber", "src", "3dMCGlb.e")
    a.out = os.path.abspath(a.out)
    os.makedirs(a.out, exist_ok=True)
    inp = os.path.join(a.out, "mcglauber_mb.input")
    b_xml = make_input(a.xml, os.path.join(a.build, "mcglauber.input"), inp)
    procs = []
    for s in a.seeds:
        d = os.path.join(a.out, f"s{s}")
        os.makedirs(d, exist_ok=True)
        if not os.path.exists(os.path.join(d, "tables")):
            os.symlink(os.path.join(a.build, "tables"), os.path.join(d, "tables"))
        procs.append(subprocess.Popen([exe, str(a.events), inp, str(s)], cwd=d,
                                      stdout=open(os.path.join(d, "run.log"), "w"),
                                      stderr=subprocess.STDOUT))
    if any(p.wait() for p in procs):
        raise SystemExit(f"3dMCGlb.e failed: see {a.out}/s*/run.log")

    files = [os.path.join(a.out, f"s{s}", "events_summary.dat") for s in a.seeds]
    d = np.vstack([np.loadtxt(f, comments="#") for f in files])
    npart, ncoll, nstr, b = d[:, 1], d[:, 2], d[:, 3], d[:, 4]
    n, c = len(b), a.percent / 100
    sig = np.pi * 10.0**2 * n / (b < 10.0).sum() / 100                  # barn
    b_exact = np.sqrt(c * sig * 100 / np.pi)
    frac = 100 * c * (b_xml / b_exact) ** 2
    lines = [f"{os.path.basename(a.xml)}: b_max {b_xml} fm, meant as 0-{a.percent:g} % "
             f"(pi b^2 = {c:g} x {a.sigma_assumed} b)",
             f"{n} minimum-bias events ({exe}, seeds {a.seeds})",
             f"this model's sigma_inel = {sig:.2f} b -> b_max {b_xml} fm is its 0-{frac:.1f} %; "
             f"0-{a.percent:g} % exactly: b < {b_exact:.2f} fm", "",
             f"{'0-' + format(a.percent, 'g') + ' % by':30s} {'class at b > b_max':>18s} "
             f"{'b < b_max outside':>17s} {'<N_coll>':>8s} {'N_coll-weighted <b>':>19s}"]
    x = 0.1
    est = {"b (geometric)": -b, "string count (standalone cut)": nstr, "N_part": npart,
           "N_coll": ncoll, f"two-component, x = {x}": (1 - x) * npart / 2 + x * ncoll}
    jit = np.random.default_rng(1).random(n) * 1e-6                       # ties at the edge
    inside = b < b_xml
    for name, e in est.items():
        cls = (e + jit) >= np.quantile(e + jit, 1 - c)
        lines.append(f"{name:30s} {np.mean(b[cls] > b_xml):18.1%} {np.mean(~cls[inside]):17.1%} "
                     f"{ncoll[cls].mean():8.0f} {np.average(b[cls], weights=ncoll[cls]):17.2f} fm")
    text = "\n".join(lines) + "\n"
    open(os.path.join(a.out, "summary.txt"), "w").write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
