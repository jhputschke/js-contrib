#!/usr/bin/env python3
"""
analysis/OO-RHIC/make_design.py -- the design points of the tune and their user XMLs.

    python make_design.py                 # design/design.json + design/xml/p000.xml ...
    python make_design.py --add best 0.142 0.31 1.6   # an extra run (validation), by name
    python make_design.py --add best_nompi 0.142 0.31 1.6 --jobs 20 --pythia-line "PartonLevel:MPI = off"

A maximin Latin hypercube of config.N_DESIGN points over config.PARAMETERS (seeded: the same
design every time).  Each point gets a copy of the p+p template XML with its Q0 and vir_factor
in <Eloss><Matter> (iMATTER reads them there too); alpha_s goes to the driver as Pythia lines
(run_runs.py).  Extra runs (validation, truncation check) are added under their own name.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET

import numpy as np

import config as C

DESIGN = C.HERE / "design"


def lhs_maximin(n, d, seed, tries=2000):
    rng = np.random.default_rng(seed)
    best, best_d = None, -1.0
    for _ in range(tries):
        u = (np.argsort(rng.random((n, d)), axis=0) + rng.random((n, d))) / n
        diff = u[:, None, :] - u[None, :, :]
        dist = np.sqrt((diff ** 2).sum(-1))[np.triu_indices(n, 1)].min()
        if dist > best_d:
            best, best_d = u, dist
    return best, best_d


def write_xml(name, params, hydro=False):
    """The run's user XML: the p+p vacuum template with Q0 and vir_factor (iMATTER and
    Matter read them).  hydro: the O+O ISR template's blocks (3dMCGlauber with p+p and
    nucleon_configuration_from_file 0, MUSIC's grid, pre-equilibrium, MUSIC) with the p+p
    template's <Hard>, <Eloss> (Matter in vacuum, no LBT) and <JetHadronization>: the same
    hard sector, plus a medium from the strings 3dMCGlauber leaves."""
    tree = ET.parse(C.TEMPLATE_XML)
    root = tree.getroot()
    if hydro:
        oo = ET.parse(C.OO_TEMPLATE_XML).getroot()
        mcg = oo.find("IS/MCGlauber")
        mcg.find("projectile").text = "p"
        mcg.find("target").text = "p"
        mcg.find("nucleon_configuration_from_file").text = "0"
        for tag in ("Hard", "Eloss", "JetHadronization"):
            old = oo.find(tag)
            idx = list(oo).index(old)
            oo.remove(old)
            oo.insert(idx, root.find(tag))
        root = oo
        tree = ET.ElementTree(root)
    m = root.find("Eloss/Matter")
    m.find("Q0").text = f" {params['Q0']:.6g} "
    m.find("vir_factor").text = f" {params['Cv']:.6g} "
    path = DESIGN / "xml" / f"{name}.xml"
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path)
    return path


def floor_ok(params):
    """iMATTER radiates only if Cv pTHat^2 > Q0^2: the lowest window must be above."""
    return params["Q0"] / np.sqrt(params["Cv"]) <= C.PTHAT_WINDOWS[0][0] + 1e-9


def load():
    return json.loads((DESIGN / "design.json").read_text())


def save(design):
    (DESIGN / "design.json").write_text(json.dumps(design, indent=1))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--add", nargs=1 + len(C.PARAMETERS), metavar="X", default=None,
                   help="add a run: NAME " + " ".join(C.PARAMETERS))
    p.add_argument("--jobs", type=int, default=None,
                   help="with --add: jobs of this run (default config.VALIDATION_JOBS)")
    p.add_argument("--windows", default=None,
                   help="with --add: pTHat windows 'A-B,C-D,...' (default config's)")
    p.add_argument("--pythia-line", action="append", default=[], dest="pythia_lines",
                   help="with --add: extra Pythia lines of the hard process, after the fixed "
                        "ones and alpha_s, e.g. 'MultipartonInteractions:pTmin = 4.0' or "
                        "'PartonLevel:MPI = off' (repeatable)")
    p.add_argument("--hydro", action="store_true",
                   help="with --add: p+p with a medium (the O+O 3dMCGlauber/MUSIC blocks with "
                        "p+p; jets in vacuum; bulk hadrons in the job)")
    a = p.parse_args(argv)
    names = list(C.PARAMETERS)
    if a.add:
        design = load()
        name, vals = a.add[0], [float(x) for x in a.add[1:]]
        params = dict(zip(names, vals))
        windows = ([tuple(float(v) for v in w.split("-")) for w in a.windows.split(",")]
                   if a.windows else C.PTHAT_WINDOWS)
        if not params["Q0"] / np.sqrt(params["Cv"]) <= windows[0][0] + 1e-9:
            sys.exit(f"make_design.py: Q0/sqrt(Cv) = {params['Q0'] / np.sqrt(params['Cv']):.2f} "
                     f"GeV is above the lowest window ({windows[0][0]} GeV)")
        write_xml(name, params, hydro=a.hydro)
        old = design["extra"].get(name, {})
        base = old.get("seed_base", 900000 + 1000 * len(design["extra"]))
        design["extra"][name] = {"params": params, "jobs": a.jobs or C.VALIDATION_JOBS,
                                 "windows": [list(w) for w in windows], "seed_base": base,
                                 "pythia_lines": list(a.pythia_lines), "hydro": a.hydro}
        save(design)
        print(f"added {name}: {params}, {design['extra'][name]['jobs']} jobs")
        return 0
    u, dmin = lhs_maximin(C.N_DESIGN, len(names), C.DESIGN_SEED)
    lo = np.array([C.PARAMETERS[k][0] for k in names])
    hi = np.array([C.PARAMETERS[k][1] for k in names])
    x = lo + u * (hi - lo)
    points = {}
    for i, row in enumerate(x):
        params = dict(zip(names, (float(v) for v in row)))
        if not floor_ok(params):
            sys.exit(f"point {i}: Q0/sqrt(Cv) above the lowest window: {params}")
        name = f"p{i:03d}"
        write_xml(name, params)
        points[name] = {"params": params}
    extra = load()["extra"] if (DESIGN / "design.json").exists() else {}
    save({"parameters": {k: list(v) for k, v in C.PARAMETERS.items()},
          "n": C.N_DESIGN, "seed": C.DESIGN_SEED, "min_distance_unit_cube": dmin,
          "points": points, "extra": extra})
    print(f"{C.N_DESIGN} points (maximin LHS, min distance {dmin:.3f} in the unit cube) -> "
          f"{DESIGN / 'design.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
