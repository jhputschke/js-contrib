#!/usr/bin/env python3
"""
analysis/OO-RHIC/run_runs.py -- run the events of the design points (or extra runs) with the
X-SCAPE p+p vacuum chain: prod_OO_isr/run_prod_oo_isr.py --vacuum, one job per core.

    python run_runs.py --design -j 20          # all design points, config.JOBS_PER_POINT each
    python run_runs.py best default -j 20      # extra runs from design.json's "extra"
    python run_runs.py --low -j 20             # the extra low-window jobs (pi0 statistics)
    python run_runs.py --design --dry-run      # print the job commands

Resumable: a job whose .json says written + skipped == requested is skipped.  Every job has
its own seed (100000 + 1000 * point + job; low-window jobs 500000 + 1000 * point + job;
extra runs from 900000), its log next to its
output in PROD_DIR/events/<run>/.  Needs the js_fno environment (PyJetscape, PYTHIA8DATA).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import config as C
import make_design as D


def windows_text(windows):
    return ",".join(f"{lo:g}-{hi:g}" for lo, hi in windows)


def job_list(design, runs, use_design, use_low=False):
    jobs = []
    names = list(design["points"]) if use_design or use_low else []
    for idx, name in enumerate(names):
        prm = design["points"][name]["params"]
        if use_design:
            for j in range(C.JOBS_PER_POINT):
                seed = 100000 + 1000 * idx + j
                jobs.append((name, prm, C.REPLACED_SEEDS.get(seed, seed), C.PTHAT_WINDOWS, []))
        if use_low:                       # the extra low-window events (pi0 statistics)
            for j in range(C.LOW_JOBS_PER_POINT):
                jobs.append((name, prm, 500000 + 1000 * idx + j, C.LOW_WINDOWS, []))
    for idx, name in enumerate(runs):
        ex = design["extra"].get(name)
        if ex is None:
            sys.exit(f"run_runs.py: no extra run {name!r} in design.json (make_design.py --add)")
        windows = [tuple(w) for w in ex.get("windows", C.PTHAT_WINDOWS)]
        base = ex["seed_base"]
        for j in range(ex["jobs"]):
            jobs.append((name, ex["params"], base + j, windows, ex.get("pythia_lines", [])))
    return jobs


def command(name, prm, seed, windows, extra_lines=()):
    out = C.point_dir(name)
    nwin = len(windows)
    events = C.EVENTS_PER_JOB - C.EVENTS_PER_JOB % nwin      # the same number per window
    cmd = [sys.executable, str(C.DRIVER), "--vacuum",
           "--user-xml", str(D.DESIGN / "xml" / f"{name}.xml"),
           "--events", str(events), "--seed", str(seed),
           "--pthat-bins", windows_text(windows),
           "--eta-max", str(C.ETA_MAX), "--eloss-threads", "0", "--outdir", str(out),
           "--max-skipped", "30"]
    lines = (C.FIXED_PYTHIA + [f"SigmaProcess:alphaSvalue = {prm['alpha_s']:.6g}",
                               f"MultipartonInteractions:alphaSvalue = {prm['alpha_s']:.6g}"]
             + list(extra_lines))
    for line in lines:
        cmd += ["--pythia-line", line]
    for line in C.FIXED_HADRONIZATION:
        cmd += ["--hadronization-line", line]
    return cmd, out, events


def done(out, seed, events):
    js = out / f"pp_isr_vacuum_seed{seed:04d}.json"
    if not js.exists():
        return False
    try:
        d = json.loads(js.read_text())
    except ValueError:
        return False
    return d.get("events_written", 0) + d.get("events_skipped", 0) == events


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="*", help="extra runs (design.json 'extra')")
    p.add_argument("--design", action="store_true", help="the design points")
    p.add_argument("--low", action="store_true",
                   help="the design points' extra low-window jobs (config.LOW_WINDOWS)")
    p.add_argument("-j", type=int, default=os.cpu_count(), help="jobs at a time")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    design = D.load()
    jobs = job_list(design, a.runs, a.design, a.low)
    todo = []
    for name, prm, seed, windows, extra in jobs:
        cmd, out, events = command(name, prm, seed, windows, extra)
        if done(out, seed, events):
            continue
        todo.append((name, seed, cmd, out))
    print(f"run_runs.py: {len(jobs)} jobs, {len(todo)} to run, {a.j} at a time; output under "
          f"{C.PROD_DIR / 'events'}")
    if a.dry_run:
        for _, _, cmd, _ in todo[:3]:
            print(" ".join(cmd))
        return 0
    env = dict(os.environ, OMP_NUM_THREADS="1")
    t0 = time.time()
    n_ok = n_fail = 0

    def run(item):
        name, seed, cmd, out = item
        out.mkdir(parents=True, exist_ok=True)
        with open(out / f"pp_isr_vacuum_seed{seed:04d}.log", "w") as log:
            r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                               cwd=str(C.DRIVER.parent))
        return name, seed, r.returncode

    with ThreadPoolExecutor(max_workers=a.j) as ex:
        for k, (name, seed, rc) in enumerate(ex.map(run, todo), 1):
            n_ok += rc == 0
            n_fail += rc != 0
            print(f"[{k}/{len(todo)}] {name} seed {seed}: "
                  + ("ok" if rc == 0 else f"FAILED (rc {rc})") + f", {time.time() - t0:.0f} s",
                  flush=True)
    print(f"run_runs.py: {n_ok} ok, {n_fail} failed, {time.time() - t0:.0f} s")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
