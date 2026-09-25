#!/usr/bin/env python3
"""
example/prod_AuAu_0_10/run_prod.py

Production of 0-10% Au+Au 200 GeV hydro evolutions -- 3D MC-Glauber (dynamical strings) ->
MUSIC (GPU in build_gpu) -- written straight to FNO4d-schema HDF5 from MUSIC's native
in-memory store.  No ROOT file, no framework copy of the evolution, no root2hdf5 step.

One call = one job = one seed = one .h5 file.  The output grid (x/y/eta range and cell
counts, tau start, spacing and max frames) comes from a YAML file, grid_fno.yaml by default.
Files from jobs with different seeds and the same grid YAML can be trained on together
(FNO4d MultiH5Array / read_3d_data_hdf5).

    conda activate js_fno
    python run_prod.py --events 10 --seed 1                 # -> out/AuAu_0_10_seed0001.h5
    python run_prod.py --events 10 --seed 1 --grid grid_x10_eta2p5.yaml
    python run_prod.py --events 50 --seed 7 --outdir /data/AuAu_0_10
    python run_prod.py --events 1 --seed 1 --dry-run        # check the grid, print the plan
    ./run_jobs.sh 20 25 1                                   # 20 jobs x 25 events, seeds 1..20

Everything this needs is in this folder, and it can be launched from anywhere.  Each job
runs in its own working directory (OUTDIR/work/<tag>, see enter_workdir()) with a private
copy of music_input, reading the shared assets from the build tree, so concurrent jobs never
touch the same file.  See README.md.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import sys
import time
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
PYJETSCAPE = os.path.dirname(os.path.dirname(HERE))          # contribs/PyJetscape
XSCAPE = os.path.abspath(os.path.join(PYJETSCAPE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(PYJETSCAPE, "python"))

from jetscape.bulk_sources import Grid, ntau_to_end         # noqa: E402  (pure Python)

USER_XML = os.path.join(HERE, "AuAu_MCGlauber_MUSIC_0_10_fast.xml")
GRID_YAML = os.path.join(HERE, "grid_fno.yaml")

#: slack for comparing grid edges; the MUSIC grid is float32
_TOL = 1e-4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--events", type=int, default=10, help="events in this job (default 10)")
    p.add_argument("--seed", type=int, default=1,
                   help="framework seed; drives 3dMCGlauber. Must differ between jobs, "
                        "and must be > 0 (0 = random, not reproducible).")
    p.add_argument("--grid", default=GRID_YAML,
                   help="output grid YAML (default: grid_fno.yaml next to this script)")
    p.add_argument("--outdir", default=os.path.join(HERE, "out"),
                   help="output directory (default: ./out next to this script)")
    p.add_argument("--out", default=None,
                   help="output file name (default: AuAu_0_10_seed<NNNN>.h5)")
    p.add_argument("--build", default=os.path.join(XSCAPE, "build_gpu"),
                   help="X-SCAPE build tree the PyJetscape extension is linked against "
                        "(default: build_gpu)")
    p.add_argument("--main-xml", default=os.path.join(XSCAPE, "config", "jetscape_main.xml"),
                   dest="main_xml")
    p.add_argument("--user-xml", default=USER_XML, dest="user_xml",
                   help="template user XML (default: the one in this folder)")
    p.add_argument("--native", action="store_true",
                   help="ignore the YAML's grid and tau min/dtau and write MUSIC's own grid "
                        "(100x100x60 here, ~10 MB per frame); max_ntau still applies")
    p.add_argument("--dry-run", action="store_true", dest="dry_run",
                   help="check the grid, write the job XML and print the plan; do not run")
    add_workdir_args(p)
    return p.parse_args()


# ─────────────────────────────────────────────────────────── working directory
#: read-only assets the vendored 3dMCGlauber / trento code still opens relative to the
#: working directory; X-SCAPE's examples/run_in_workdir.sh links the same set
WORKDIR_LINKS = ("tables", "eps09", "LHAPDF_Lib", "nucleusConfigs", "data_table")


def add_workdir_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--workdir", default=None,
                   help="this job's working directory (default: OUTDIR/work/<output name>)")
    p.add_argument("--keep-workdir", action="store_true", dest="keep_workdir",
                   help="keep the working directory after a successful job (it holds only "
                        "the modules' side files: 3dMCGlauber strings_event_*.dat, MUSIC's "
                        "momentum_anisotropy_*.dat, ...); a failed job always keeps it")
    p.add_argument("--in-build", action="store_true", dest="in_build",
                   help="run in the build tree itself, as before (shares music_input and "
                        "every side file with any other job running there)")


def _absolute_parent_paths(src: str, dst: str, build: str) -> None:
    """Copy an XML, making every element text that starts with '../' absolute.

    Such paths (iSS, SMASH, hydro-from-file, Martini, ... in jetscape_main.xml) are
    relative to the build tree, which is the working directory only with --in-build.
    './' paths (outputs such as ./FinalPartonsInfo.dat, iSS_working_path '.') stay
    relative: they belong in the job's own directory.
    """
    tree = ET.parse(src)
    for el in tree.iter():
        text = (el.text or "").strip()
        if text.startswith("../"):
            el.text = os.path.normpath(os.path.join(build, text))
    tree.write(dst)


def enter_workdir(a, workdir: str, job_xml_path: str) -> str:
    """Make ``workdir`` this job's working directory -> the main XML to use.

    The Python counterpart of X-SCAPE's examples/run_in_workdir.sh (which execs the
    runJetscape binary, so it cannot drive PyJetscape).  In the build tree, concurrent jobs
    share music_input -- MusicWrapper truncates and rewrites it at every MUSIC init, and a
    job reading it in that window spins forever in MUSIC's StringFind4 -- as well as every
    file the modules write to the working directory (3dMCGlauber strings_event_<N>.dat and
    events_summary.dat, MUSIC's momentum_anisotropy / eccentricities / meanpT files, ...).
    Here each job has its own directory with a private music_input, and the read-only
    assets come from the build tree:

    * ``XSCAPE_DATA_DIR`` -> mcglauber.input (MCGlauberWrapper), LBT-tables fallback
    * ``HYDROPROGRAMPATH`` -> MUSIC's EOS tables
    * ``LBT_TABLES_PATH`` -> LBT tables
    * symlinks for the dirs still opened relative to the working directory (WORKDIR_LINKS)
    * '../' paths in the main and job XML made absolute (see _absolute_parent_paths)
    """
    build = a.build
    os.makedirs(workdir, exist_ok=True)
    shutil.copyfile(os.path.join(build, "music_input"), os.path.join(workdir, "music_input"))
    for name in WORKDIR_LINKS:
        src, dst = os.path.join(build, name), os.path.join(workdir, name)
        if os.path.exists(src) and not os.path.lexists(dst):
            os.symlink(src, dst)
    os.environ["XSCAPE_DATA_DIR"] = build
    os.environ["HYDROPROGRAMPATH"] = build
    os.environ["LBT_TABLES_PATH"] = os.path.join(build, "LBT-tables")
    main_xml = os.path.join(workdir, "jetscape_main.xml")
    _absolute_parent_paths(a.main_xml, main_xml, build)
    _absolute_parent_paths(job_xml_path, job_xml_path, build)
    os.chdir(workdir)
    return main_xml


def job_workdir(a, out_h5: str) -> str:
    """This job's working directory: --workdir, else OUTDIR/work/<output name>."""
    if a.workdir:
        return os.path.abspath(a.workdir)
    return os.path.join(os.path.dirname(out_h5), "work",
                        os.path.splitext(os.path.basename(out_h5))[0])


def leave_workdir(a, workdir: str, ok: bool) -> None:
    """Remove the working directory after a successful job unless --keep-workdir."""
    if a.in_build:
        return
    os.chdir(os.path.dirname(workdir))
    if ok and not a.keep_workdir:
        shutil.rmtree(workdir, ignore_errors=True)
        if not a.workdir:                # OUTDIR/work, once its last job is gone
            try:
                os.rmdir(os.path.dirname(workdir))
            except OSError:
                pass
    else:
        print(f"  working directory kept: {workdir}")


# ───────────────────────────────────────────────────────────────── grid YAML
def load_grid_yaml(path: str):
    """Read the grid YAML -> (Grid with ntau = max_ntau, max_ntau, file text)."""
    import yaml

    text = open(path).read()
    cfg = yaml.safe_load(text) or {}

    def section(d, name, keys):
        if not isinstance(d.get(name), dict):
            sys.exit(f"run_prod.py: {path}: missing section '{name}'")
        s = d[name]
        extra, missing = set(s) - set(keys), [k for k in keys if k not in s]
        if extra or missing:
            sys.exit(f"run_prod.py: {path}: '{name}' needs exactly {list(keys)}"
                     + (f"; unknown {sorted(extra)}" if extra else "")
                     + (f"; missing {missing}" if missing else ""))
        return s

    unknown = set(cfg) - {"grid", "tau"}
    if unknown:
        sys.exit(f"run_prod.py: {path}: unknown top-level keys {sorted(unknown)}")
    g = section(cfg, "grid", ("x", "y", "eta"))
    axes = {ax: section(g, ax, ("min", "max", "n")) for ax in ("x", "y", "eta")}
    t = section(cfg, "tau", ("min", "dtau", "max_ntau"))
    for ax, s in axes.items():
        if not isinstance(s["n"], int):
            sys.exit(f"run_prod.py: {path}: grid.{ax}.n must be an integer, got {s['n']!r}")
    if not isinstance(t["max_ntau"], int) or t["max_ntau"] < 0:
        sys.exit(f"run_prod.py: {path}: tau.max_ntau must be an integer >= 0")
    if t["min"] <= 0:
        sys.exit(f"run_prod.py: {path}: tau.min must be > 0")

    try:
        grid = Grid.from_bounds(**{ax: (s["min"], s["max"], s["n"]) for ax, s in axes.items()},
                                tau_min=t["min"], dtau=t["dtau"], ntau=t["max_ntau"])
    except ValueError as exc:
        sys.exit(f"run_prod.py: {path}: {exc}")
    return grid, int(t["max_ntau"]), text


def music_box(user_root, main_xml: str) -> dict:
    """MUSIC's cell-centre range per axis, from the <IS> grid it inherits (Initial_profile
    131 uses the initial-state lattice): n = 2*max/step cells from -max, so the upper edge
    is max - step (x: -15 .. 14.7 for max 15, step 0.3)."""
    main_root = ET.parse(main_xml).getroot()

    def val(tag):
        for root in (user_root, main_root):
            node = root.find(f"IS/{tag}")
            if node is not None and node.text and node.text.strip():
                return float(node.text)
        sys.exit(f"run_prod.py: <IS><{tag}> not found in the user or main XML")

    box = {}
    for ax, tag in (("x", "x"), ("y", "y"), ("eta", "z")):
        hi, step = val(f"grid_max_{tag}"), val(f"grid_step_{tag}")
        n = int(round(2 * hi / step))
        box[ax] = (-hi, -hi + (n - 1) * step, n, step)
    return box


def check_inside(grid: Grid, box: dict) -> None:
    bad = []
    for ax in ("x", "y", "eta"):
        axis = grid.axis(ax)
        lo, hi = box[ax][0], box[ax][1]
        if axis[0] < lo - _TOL or axis[-1] > hi + _TOL:
            bad.append(f"{ax}: requested {axis[0]:g} .. {axis[-1]:g}, "
                       f"MUSIC covers {lo:g} .. {hi:g}")
    if bad:
        sys.exit("run_prod.py: the output grid leaves MUSIC's box (those cells would be "
                 "written as zeros):\n  " + "\n  ".join(bad))


def describe(grid: Grid, max_ntau: int) -> str:
    rows = []
    for ax, n, d in (("x", grid.nx, grid.dx), ("y", grid.ny, grid.dy),
                     ("eta", grid.neta, grid.deta)):
        a = grid.axis(ax)
        rows.append(f"{ax:>3} {a[0]:g} .. {a[-1]:g}, n = {n}, step = {d:g}")
    t_end = grid.tau_min + (max_ntau - 1) * grid.dtau if max_ntau else None
    rows.append(f"tau {grid.tau_min:g} + k*{grid.dtau:g}, max_ntau = {max_ntau or 'auto'}"
                + (f" (to {t_end:g} fm/c)" if t_end else ""))
    mb = 4 * 4 * grid.nx * grid.ny * grid.neta / 2**20
    rows.append(f"    {mb:.1f} MB per frame uncompressed")
    return "\n".join("           " + r for r in rows)


# ───────────────────────────────────────────────────────────────── job setup
def pythia_data_problem():
    """None if pyjetscape_core can be imported, else the reason.

    Pythia finds its xmldoc through PYTHIA8DATA or the path compiled into libpythia8.
    Homebrew's Pythia (macOS) has a working compiled-in path; a relocated conda Pythia
    (the js_fno env) may not, and then the import aborts the process -- so it is probed
    in a subprocess, and only when PYTHIA8DATA is unset.
    """
    if os.environ.get("PYTHIA8DATA"):
        return None
    import subprocess
    probe = f"import sys; sys.path.insert(0, {os.path.join(PYJETSCAPE, 'python')!r}); import jetscape"
    try:
        r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                           timeout=300)
    except subprocess.TimeoutExpired:
        return "importing pyjetscape_core timed out (PYTHIA8DATA is not set)"
    if r.returncode == 0:
        return None
    tail = " | ".join((r.stderr or r.stdout).strip().splitlines()[-3:])
    return ("PYTHIA8DATA is not set and importing pyjetscape_core fails without it: point it "
            "to Pythia's xmldoc (`conda activate js_fno` sets it). "
            f"Import error (exit {r.returncode}): {tail}")


def check_env(a) -> None:
    problems = []
    pythia = pythia_data_problem()
    if pythia:
        problems.append(pythia)
    for f in ("music_input", "mcglauber.input"):
        if not os.path.exists(os.path.join(a.build, f)):
            problems.append(f"{f} not found in the build tree {a.build}")
    if a.seed <= 0:
        problems.append("--seed must be > 0 (0 means a random seed; the job would not be "
                        "reproducible and two jobs could collide).")
    if problems:
        sys.exit("run_prod.py: cannot start:\n  " + "\n  ".join(problems))


def job_xml(a, out_h5: str):
    """Write the per-job user XML next to the output file -> (path, parsed root)."""
    tree = ET.parse(a.user_xml)
    root = tree.getroot()
    root.find("nEvents").text = f" {a.events} "
    root.find("Random/seed").text = str(a.seed)
    # Guard against a hand-edited template that would silently change what is written.
    music = root.find("Hydro/MUSIC")
    for tag, want in (("output_evolution_to_memory", "1"), ("dump_hydro_only", "1")):
        node = music.find(tag)
        if node is None or node.text.strip() != want:
            sys.exit(f"run_prod.py: {a.user_xml}: <Hydro><MUSIC><{tag}> must be {want}")
    for tag in ("Eloss", "SoftParticlization", "Afterburner", "RootBulkWriter",
                "FastRootBulkWriter"):
        if root.find(tag) is not None:
            sys.exit(f"run_prod.py: {a.user_xml}: remove <{tag}> -- this is a hydro-only "
                     "dump (the framework medium is left empty).")
    path = os.path.splitext(out_h5)[0] + ".xml"
    tree.write(path)
    return path, root


def main() -> int:
    a = parse_args()
    a.build = os.path.abspath(a.build)
    a.outdir = os.path.abspath(a.outdir)
    a.main_xml = os.path.abspath(a.main_xml)
    a.user_xml = os.path.abspath(a.user_xml)
    a.grid = os.path.abspath(a.grid)
    check_env(a)

    grid, max_ntau, grid_text = load_grid_yaml(a.grid)
    os.makedirs(a.outdir, exist_ok=True)
    out_h5 = os.path.join(a.outdir, a.out or f"AuAu_0_10_seed{a.seed:04d}.h5")
    xml, root = job_xml(a, out_h5)
    box = music_box(root, a.main_xml)
    grid_mode = "native" if a.native else "grid"
    if not a.native:
        check_inside(grid, box)

    print(f"prod_AuAu_0_10: {a.events} event(s), seed {a.seed}, grid_mode {grid_mode}")
    print(f"  build    {a.build}")
    print(f"  job XML  {xml}")
    print(f"  grid     {a.grid}")
    if a.native:
        print("  grid     MUSIC native: " + ", ".join(
            f"{ax} {b[0]:g} .. {b[1]:g} (n = {b[2]})" for ax, b in box.items())
            + f"; max_ntau = {max_ntau or 'auto'}")
    else:
        print(describe(grid, max_ntau))
    print(f"  output   {out_h5}")
    if a.dry_run:
        return 0

    import numpy as np
    import h5py
    import jetscape as js
    from jetscape.fast_h5_bulk import H5BulkWriter

    provenance = {
        "prod": "PyJetscape/example/prod_AuAu_0_10",
        "prod_seed": a.seed,
        "prod_host": socket.gethostname(),
        "prod_platform": platform.platform(),
        "prod_build": a.build,
        "prod_user_xml": open(xml).read(),
        "prod_grid_yaml": grid_text,
        "system": "AuAu 200 GeV 0-10% (b in [0, 4.7] fm)",
        "initial_state_kind": "3dMCGlauber_strings",
        "hydro": "MUSIC (music4gpu)" if "gpu" in os.path.basename(a.build) else "MUSIC",
        "T_fo": 0.15,
    }
    writer = H5BulkWriter(out_file_name=out_h5, grid_mode=grid_mode, tau_stride=1,
                          choose_ntau=max_ntau, out_grid=None if a.native else grid,
                          extra_attrs=provenance, verbose=True)

    # MUSIC / 3dMCGlauber resolve their input files relative to the working directory.
    if a.in_build:
        os.chdir(a.build)
        main_xml, workdir = a.main_xml, a.build
    else:
        workdir = job_workdir(a, out_h5)
        main_xml = enter_workdir(a, workdir, xml)
    print(f"  workdir  {workdir}")
    jetscape = js.JetScapePerEvent()
    jetscape.SetXMLMainFileName(main_xml)
    jetscape.SetXMLUserFileName(xml)
    jetscape.Init()

    wall, tau0 = [], []                      # per WRITTEN event
    late = truncated = 0
    t_job = time.time()
    writer.Init()
    try:
        jetscape.ExecInit()
        for i in range(jetscape.GetNumberOfEvents()):
            t0 = time.time()
            jetscape.ExecPerEvent()
            # With dump_hydro_only, bulk_info.tau_min is this event's MUSIC tau0.
            hydro = js.JetScapeSignalManager.Instance().GetHydroPointer()
            t0_music, n_full = np.nan, None
            if hydro is not None:
                b = hydro.get_bulk_info()
                t0_music = float(b.tau_min)
                # frames from tau.min to this event's end, before any max_ntau cut
                n_full = ntau_to_end(Grid.from_bulk_info(b), grid.tau_min, grid.dtau)
            n_before = writer.GetNumberOfEventsWritten()
            writer.Exec()                    # MUSIC's native store is live here
            jetscape.ClearPerEvent()
            if writer.GetNumberOfEventsWritten() > n_before:
                wall.append(time.time() - t0)
                tau0.append(t0_music)
            msg = (f"prod_AuAu_0_10: event {i + 1}/{a.events} done in "
                   f"{time.time() - t0:.1f} s, MUSIC tau0 = {t0_music:.3f} fm/c")
            if not a.native and t0_music > grid.tau_min + _TOL:
                late += 1
                msg += (f"  WARNING: after tau.min = {grid.tau_min:g}, so the first "
                        f"frame(s) of this event are zeros")
            if not a.native and max_ntau and n_full is not None and n_full > max_ntau:
                truncated += 1
                msg += (f", kept the first {max_ntau} of {n_full} frames "
                        f"(tau.max_ntau)")
            print(msg)
        jetscape.Finish()
    finally:
        # JetScape::Finish() does not propagate to sub-tasks: close the file by hand.
        writer.Finish()

    n = writer.GetNumberOfEventsWritten()
    if n:
        with h5py.File(out_h5, "a") as f:
            g = f.require_group("diag")
            for name, vals in (("wall_s", wall), ("tau0_music", tau0)):
                if name in g:
                    del g[name]
                g[name] = np.asarray(vals, dtype=np.float64)
            f.attrs["prod_wall_s_total"] = time.time() - t_job
    summary = {"out": out_h5, "seed": a.seed, "grid": a.grid, "events_requested": a.events,
               "events_written": n, "events_tau0_after_tau_min": late,
               "events_cut_at_max_ntau": truncated,
               "wall_s": round(time.time() - t_job, 1)}
    with open(os.path.splitext(out_h5)[0] + ".json", "w") as f:
        json.dump(summary, f, indent=1)
    print("prod_AuAu_0_10:", json.dumps(summary))
    leave_workdir(a, workdir, n == a.events)
    return 0 if n == a.events else 1


if __name__ == "__main__":
    raise SystemExit(main())
