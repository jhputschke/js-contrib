#!/usr/bin/env python3
"""
example/prod_AuAu_0_10_jet/run_prod_jet.py

prod_AuAu_0_10 with a jet: 0-10% Au+Au 200 GeV, 3D MC-Glauber strings -> the X-SCAPE
two-stage hydro (MUSIC_1 background -> Matter + LBT + CausalLiquefier -> MUSIC_2 with the
jet's deposition), written as background/jet PAIRS straight to FNO4d-schema HDF5 in
FastHydro's layout (jetscape.pair_h5.PairH5Writer): arr = jet leg, arr_bg = background,
source/droplets, shower/, diag/.

One call = one job = one seed = one .h5 file, as in ../prod_AuAu_0_10/run_prod.py, whose
grid YAMLs, grid checks and environment checks this reuses.

    conda activate js_fno
    python run_prod_jet.py --events 10 --seed 1                  # PythiaGun, deposition on
    python run_prod_jet.py --events 1 --seed 1 --no-deposit      # null test: arr == arr_bg
    python run_prod_jet.py --events 10 --seed 1 --hard pgun --pgun-pt 60
    python run_prod_jet.py --events 30 --seed 1 --reuse 3        # one background per 3 jets
    python run_prod_jet.py --events 10 --seed 1 --surface jet    # surface for the jet leg only
    python run_prod_jet.py --events 1 --seed 1 --dry-run         # check, print the plan
    ./run_jobs.sh 20 25 1                                        # 20 jobs x 25 events

Needs a MUSIC build with the jet source slot (MUSIC cee9460 + X-SCAPE PR #138 on the CPU,
the matching MUSIC4GPU port on the GPU).  Without it MUSIC_2 silently ignores the droplets;
PairH5Writer then warns that the jet leg is bit-identical to the background.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import socket
import sys
import time
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
PROD = os.path.join(os.path.dirname(HERE), "prod_AuAu_0_10")

# The single-leg production's helpers (grid YAML, MUSIC box, env checks).  Loading it also
# puts PyJetscape/python on sys.path.
_spec = importlib.util.spec_from_file_location("_run_prod", os.path.join(PROD, "run_prod.py"))
rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp)

USER_XML = os.path.join(HERE, "AuAu_MCGlauber_MUSIC_0_10_jet.xml")
GRID_YAML = os.path.join(PROD, "grid_fno.yaml")
BG_ID, JET_ID = "MUSIC_1", "MUSIC_2"

HARD_VERTEX = {
    "pythia": "3dMCGlauber binary-collision points (PythiaGun uses x, y; z = t = 0)",
    "pgun": "origin: PGun samples a vertex and zeroes it (PGun.cc:117-120)",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--events", type=int, default=10, help="events in this job (default 10)")
    p.add_argument("--seed", type=int, default=1,
                   help="framework seed (3dMCGlauber, Pythia, Matter/LBT). Must differ "
                        "between jobs and be > 0.")
    p.add_argument("--grid", default=GRID_YAML,
                   help="output grid YAML (default: ../prod_AuAu_0_10/grid_fno.yaml)")
    p.add_argument("--outdir", default=os.path.join(HERE, "out"),
                   help="output directory (default: ./out next to this script)")
    p.add_argument("--out", default=None,
                   help="output file name (default: AuAu_0_10_jet_seed<NNNN>.h5)")
    p.add_argument("--build", default=os.path.join(rp.XSCAPE, "build_gpu"),
                   help="X-SCAPE build tree (default: build_gpu)")
    p.add_argument("--main-xml", default=os.path.join(rp.XSCAPE, "config", "jetscape_main.xml"),
                   dest="main_xml")
    p.add_argument("--user-xml", default=USER_XML, dest="user_xml",
                   help="template user XML (default: the one in this folder)")
    p.add_argument("--native", action="store_true",
                   help="write MUSIC's own grid for both legs instead of the YAML's")
    p.add_argument("--hard", choices=("pythia", "pgun"), default="pythia",
                   help="hard process: PythiaGun (default; vertex from the Glauber "
                        "collisions) or PGun (fixed pT, vertex at the origin)")
    p.add_argument("--pthat-min", type=float, default=None, dest="pthat_min",
                   help="PythiaGun pTHatMin [GeV] (default: the XML's)")
    p.add_argument("--pthat-max", type=float, default=None, dest="pthat_max",
                   help="PythiaGun pTHatMax [GeV] (default: the XML's)")
    p.add_argument("--pgun-pt", type=float, default=60.0, dest="pgun_pt",
                   help="PGun parton pT [GeV] (default 60)")
    p.add_argument("--reuse", type=int, default=1,
                   help="jet events per background (setReuseHydro); 1 = a new background "
                        "every event (default)")
    p.add_argument("--no-deposit", action="store_true", dest="no_deposit",
                   help="null test: MUSIC_2 without the liquefier, so arr must equal arr_bg")
    p.add_argument("--no-showers", action="store_true", dest="no_showers",
                   help="do not store the parton showers (shower/)")
    p.add_argument("--surface", choices=("none", "bg", "jet", "both"), default="none",
                   help="which legs build MUSIC's freeze-out surface. It is only needed to "
                        "particlize a leg (e.g. 'jet' for hadrons from the jet leg); "
                        "'none' (default) is ~6 s per MUSIC run faster, with a "
                        "bit-identical evolution")
    p.add_argument("--dry-run", action="store_true", dest="dry_run",
                   help="check the XML and grid, write the job XML, print the plan; do not run")
    return p.parse_args()


def _text(node, path):
    n = node.find(path)
    return None if n is None or n.text is None else n.text.strip()


def _set(parent, tag, value):
    n = parent.find(tag)
    if n is None:
        n = ET.SubElement(parent, tag)
    n.text = f" {value} "
    return n


def music_delta_tau(build: str) -> float:
    """MUSIC's time step from the build tree's music_input."""
    with open(os.path.join(build, "music_input")) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "Delta_Tau":
                return float(parts[1])
    sys.exit(f"run_prod_jet.py: no Delta_Tau in {build}/music_input")


def job_xml(a, out_h5: str):
    """Check the template, apply this job's settings, write it next to the output."""
    tree = ET.parse(a.user_xml)
    root = tree.getroot()
    problems = []

    _set(root, "nEvents", a.events)
    root.find("Random/seed").text = str(a.seed)
    _set(root, "setReuseHydro", "true" if a.reuse > 1 else "false")
    _set(root, "nReuseHydro", max(1, a.reuse))

    # Exactly one hard process: the automatic task list would run every block present.
    hard = root.find("Hard")
    if hard is None:
        sys.exit(f"run_prod_jet.py: {a.user_xml} has no <Hard> block")
    for child in list(hard):
        hard.remove(child)
    template = ET.parse(a.user_xml).getroot().find("Hard")
    if a.hard == "pythia":
        pg = template.find("PythiaGun")
        if pg is None:
            sys.exit(f"run_prod_jet.py: {a.user_xml} has no <Hard><PythiaGun>")
        hard.append(pg)
        if a.pthat_min is not None:
            _set(pg, "pTHatMin", a.pthat_min)
        if a.pthat_max is not None:
            _set(pg, "pTHatMax", a.pthat_max)
    else:
        pgun = ET.SubElement(hard, "PGun")
        _set(pgun, "name", "PGun")
        _set(pgun, "pT", a.pgun_pt)

    hydros = root.findall("Hydro")
    names = [_text(h, "MUSIC/name") for h in hydros]
    if names != [BG_ID, JET_ID]:
        problems.append(f"need exactly two <Hydro><MUSIC> blocks named {BG_ID}, {JET_ID} "
                        f"(in that order), found {names}")
    else:
        music = hydros[0].find("MUSIC")
        if _text(music, "output_evolution_to_memory") != "1":
            problems.append(f"{BG_ID} needs <output_evolution_to_memory>1")
        if _text(music, "dump_hydro_only") != "0":
            problems.append(f"{BG_ID} needs <dump_hydro_only>0: Matter/LBT read its "
                            "framework medium (MUSIC_2 is switched to 1 in Python)")
        _set(hydros[1], "AddLiquefier", "false" if a.no_deposit else "true")
        # First block: the background leg and the default for every instance; MUSIC_2's
        # own block always gets an explicit value, which overrides it for the jet leg.
        _set(music, "freeze_out_surface", 1 if a.surface in ("bg", "both") else 0)
        _set(hydros[1].find("MUSIC"), "freeze_out_surface",
             1 if a.surface in ("jet", "both") else 0)
    if _text(root, "Preequilibrium/evolutionInMemory") != "0":
        problems.append("<Preequilibrium><evolutionInMemory> must be 0 with NullPreDynamics "
                        "and 3D-Glauber strings (otherwise the medium's tau axis is garbage)")
    if root.find("Eloss") is None or _text(root, "Eloss/AddLiquefier") != "true":
        problems.append("<Eloss> with <AddLiquefier>true is required")
    liq = root.find("Liquefier/CausalLiquefier")
    if liq is None:
        problems.append("<Liquefier><CausalLiquefier> is required")
    else:
        dtau, music_dtau = float(_text(liq, "dtau")), music_delta_tau(a.build)
        if abs(dtau - music_dtau) > 1e-9:
            problems.append(f"<CausalLiquefier><dtau> = {dtau} must equal MUSIC's Delta_Tau "
                            f"= {music_dtau}: the kernel is divided by it and deposited in "
                            "one hydro step")
    for tag in ("SoftParticlization", "Afterburner", "RootBulkWriter", "FastRootBulkWriter"):
        if root.find(tag) is not None:
            problems.append(f"remove <{tag}>: this job writes the hydro pair only")
    if problems:
        sys.exit(f"run_prod_jet.py: {a.user_xml}:\n  " + "\n  ".join(problems))

    path = os.path.splitext(out_h5)[0] + ".xml"
    tree.write(path)
    return path, root


def main() -> int:
    a = parse_args()
    for k in ("build", "outdir", "main_xml", "user_xml", "grid"):
        setattr(a, k, os.path.abspath(getattr(a, k)))
    rp.check_env(a)
    if a.reuse < 1:
        sys.exit("run_prod_jet.py: --reuse must be >= 1")

    grid, max_ntau, grid_text = rp.load_grid_yaml(a.grid)
    os.makedirs(a.outdir, exist_ok=True)
    out_h5 = os.path.join(a.outdir, a.out or f"AuAu_0_10_jet_seed{a.seed:04d}.h5")
    xml, root = job_xml(a, out_h5)
    box = rp.music_box(root, a.main_xml)
    grid_mode = "native" if a.native else "grid"
    if not a.native:
        rp.check_inside(grid, box)

    hard_desc = (f"PythiaGun pTHat {_text(root, 'Hard/PythiaGun/pTHatMin')}-"
                 f"{_text(root, 'Hard/PythiaGun/pTHatMax')} GeV" if a.hard == "pythia"
                 else f"PGun pT {a.pgun_pt:g} GeV")
    print(f"prod_AuAu_0_10_jet: {a.events} event(s), seed {a.seed}, grid_mode {grid_mode}, "
          f"{hard_desc}, reuse {a.reuse}, deposition {'OFF (null test)' if a.no_deposit else 'on'}, "
          f"freeze-out surface: {a.surface}")
    print(f"  build    {a.build}")
    print(f"  job XML  {xml}")
    print(f"  grid     {a.grid}")
    if a.native:
        print("  grid     MUSIC native: " + ", ".join(
            f"{ax} {b[0]:g} .. {b[1]:g} (n = {b[2]})" for ax, b in box.items())
            + f"; max_ntau = {max_ntau or 'auto'}")
    else:
        print(rp.describe(grid, max_ntau))
    print(f"  output   {out_h5}")
    if a.dry_run:
        return 0

    import h5py
    import jetscape as js
    from jetscape.pair_h5 import PairH5Writer

    provenance = {
        "prod": "PyJetscape/example/prod_AuAu_0_10_jet",
        "prod_seed": a.seed,
        "prod_host": socket.gethostname(),
        "prod_platform": platform.platform(),
        "prod_build": a.build,
        "prod_user_xml": open(xml).read(),
        "prod_grid_yaml": grid_text,
        "prod_hard": hard_desc,
        "prod_reuse": a.reuse,
        "prod_surface": a.surface,
        "system": "AuAu 200 GeV 0-10% (b in [0, 4.7] fm)",
        "initial_state_kind": "3dMCGlauber_strings",
        "hydro": "MUSIC (music4gpu)" if "gpu" in os.path.basename(a.build) else "MUSIC",
        "jet": "Matter + LBT, CausalLiquefier droplets into MUSIC_2",
        "T_fo": 0.15,
    }
    writer = PairH5Writer(
        out_h5, bg_id=BG_ID, jet_id=JET_ID, grid_mode=grid_mode,
        out_grid=None if a.native else grid, choose_ntau=max_ntau,
        store_showers=not a.no_showers,
        provenance={"hard_vertex": HARD_VERTEX[a.hard],
                    "eos_kind": "hotqcd (MUSIC EOS 9)",
                    "transport_mode": "MUSIC viscous: eta/s(T) and zeta/s(T) "
                                      "parametrization 3, second-order terms"},
        extra_attrs=provenance, verbose=True)

    # MUSIC / 3dMCGlauber resolve their input files relative to the working directory.
    os.chdir(a.build)
    jetscape = js.JetScapePerEvent()
    jetscape.SetXMLMainFileName(a.main_xml)
    jetscape.SetXMLUserFileName(xml)
    jetscape.Init()
    writer.attach(jetscape)                  # after Init: MUSIC_2 -> dump_hydro_only

    late = 0
    droplets = e_late = 0.0
    t_job = time.time()
    try:
        jetscape.ExecInit()
        for i in range(jetscape.GetNumberOfEvents()):
            t0 = time.time()
            jetscape.ExecPerEvent()
            idx = writer.Exec()              # both legs are live here
            jetscape.ClearPerEvent()
            wall = time.time() - t0
            if idx is None:
                print(f"prod_AuAu_0_10_jet: event {i + 1}/{a.events} SKIPPED (see warning)")
                continue
            writer.write_diag(idx, wall_s=wall)
            d = writer.last_event_diag
            droplets += d["n_droplets"]
            e_late += d["E_droplets_late"]
            msg = (f"prod_AuAu_0_10_jet: event {i + 1}/{a.events} done in {wall:.1f} s, "
                   f"MUSIC tau0 = {d['tau0_music']:.3f} fm/c, jet/bg frames "
                   f"{d['ntau_jet']}/{d['ntau_bg']}, {d['n_droplets']} droplets "
                   f"({d['E_droplets']:.1f} GeV, {d['E_droplets_late']:.1f} after freeze-out), "
                   f"{d['frames_identical']} leading frames identical, bg {d['bg_id']}")
            if not a.native and d["tau0_music"] > grid.tau_min + rp._TOL:
                late += 1
                msg += (f"  WARNING: tau0 after tau.min = {grid.tau_min:g}, so the first "
                        "frame(s) of this event are zeros")
            print(msg)
        jetscape.Finish()
    finally:
        writer.Finish()

    n = writer.GetNumberOfEventsWritten()
    if n:
        with h5py.File(out_h5, "a") as f:
            f.attrs["prod_wall_s_total"] = time.time() - t_job
    summary = {"out": out_h5, "seed": a.seed, "grid": a.grid, "hard": hard_desc,
               "reuse": a.reuse, "deposition": not a.no_deposit, "surface": a.surface,
               "events_requested": a.events, "events_written": n,
               "events_tau0_after_tau_min": late,
               "legs_cut_at_max_ntau": writer.n_clipped,
               "droplets_total": int(droplets), "E_droplets_late_total": round(e_late, 3),
               "wall_s": round(time.time() - t_job, 1)}
    with open(os.path.splitext(out_h5)[0] + ".json", "w") as f:
        json.dump(summary, f, indent=1)
    print("prod_AuAu_0_10_jet:", json.dumps(summary))
    return 0 if n == a.events else 1


if __name__ == "__main__":
    raise SystemExit(main())
