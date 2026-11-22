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
                jobs.append((name, prm, C.REPLACED_SEEDS.get(seed, seed), C.PTHAT_WINDOWS, [],
                             False))
        if use_low:                       # the extra low-window events (pi0 statistics)
            for j in range(C.LOW_JOBS_PER_POINT):
                jobs.append((name, prm, 500000 + 1000 * idx + j, C.LOW_WINDOWS, [], False))
    for idx, name in enumerate(runs):
        ex = design["extra"].get(name)
        if ex is None:
            sys.exit(f"run_runs.py: no extra run {name!r} in design.json (make_design.py --add)")
        windows = [tuple(w) for w in ex.get("windows", C.PTHAT_WINDOWS)]
        base = ex["seed_base"]
        for j in range(ex["jobs"]):
            jobs.append((name, ex["params"], base + j, windows, ex.get("pythia_lines", []),
                         ex.get("hydro", False)))
    return jobs


def command(name, prm, seed, windows, extra_lines=(), hydro=False, events=None):
    out = C.point_dir(name)
    nwin = len(windows)
    events = events or C.EVENTS_PER_JOB
    events -= events % nwin                                   # the same number per window
    medium = (["--no-particlize", "--no-hydro-file",
               "--bulk-hadrons", str(C.HYDRO_BULK_OVERSAMPLES)] if hydro else ["--vacuum"])
    cmd = [sys.executable, str(C.DRIVER), *medium,
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
    js = next(iter(sorted(out.glob(f"*_seed{seed:04d}.json"))), out / "none.json")
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
    p.add_argument("--events", type=int, default=None,
                   help="events per job (default config.EVENTS_PER_JOB; e.g. a timing test)")
    p.add_argument("--mps", action="store_true",
                   help="share the GPU through a CUDA MPS daemon started for this call (as "
                        "run_jobs.sh --mps; the production setting for hydro runs on the GB10)")
    p.add_argument("--omp", type=int, default=1,
                   help="OMP_NUM_THREADS per job (default 1; production hydro on the GB10: 2 "
                        "with -j 8)")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    design = D.load()
    jobs = job_list(design, a.runs, a.design, a.low)
    todo = []
    for name, prm, seed, windows, extra, hydro in jobs:
        cmd, out, events = command(name, prm, seed, windows, extra, hydro, a.events)
        if done(out, seed, events):
            continue
        todo.append((name, seed, cmd, out))
    print(f"run_runs.py: {len(jobs)} jobs, {len(todo)} to run, {a.j} at a time; output under "
          f"{C.PROD_DIR / 'events'}")
    if a.dry_run:
        for _, _, cmd, _ in todo[:3]:
            print(" ".join(cmd))
        return 0
    env = dict(os.environ, OMP_NUM_THREADS=str(a.omp))
    mps_dir = None
    if a.mps:
        mps_dir = os.path.join(os.environ.get("XDG_RUNTIME_DIR", "/tmp"), f"xscape-mps.{os.getpid()}")
        os.makedirs(os.path.join(mps_dir, "pipe"), exist_ok=True)
        os.makedirs(os.path.join(mps_dir, "log"), exist_ok=True)
        env.update(CUDA_MPS_PIPE_DIRECTORY=os.path.join(mps_dir, "pipe"),
                   CUDA_MPS_LOG_DIRECTORY=os.path.join(mps_dir, "log"))
        if subprocess.run(["nvidia-cuda-mps-control", "-d"], env=env).returncode != 0:
            sys.exit("run_runs.py: could not start the CUDA MPS daemon")
        print(f"run_runs.py: MPS daemon started ({mps_dir})", flush=True)
    t0 = time.time()
    n_ok = n_fail = 0

    def run(item):
        name, seed, cmd, out = item
        out.mkdir(parents=True, exist_ok=True)
        with open(out / f"run_seed{seed:04d}.log", "w") as log:
            r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                               cwd=str(C.DRIVER.parent))
        return name, seed, r.returncode

    try:
        with ThreadPoolExecutor(max_workers=a.j) as ex:
            for k, (name, seed, rc) in enumerate(ex.map(run, todo), 1):
                n_ok += rc == 0
                n_fail += rc != 0
                print(f"[{k}/{len(todo)}] {name} seed {seed}: "
                      + ("ok" if rc == 0 else f"FAILED (rc {rc})")
                      + f", {time.time() - t0:.0f} s", flush=True)
    finally:
        if mps_dir:
            subprocess.run(["nvidia-cuda-mps-control"], input="quit\n", text=True, env=env)
            import shutil
            shutil.rmtree(mps_dir, ignore_errors=True)
            print("run_runs.py: MPS daemon stopped", flush=True)
    print(f"run_runs.py: {n_ok} ok, {n_fail} failed, {time.time() - t0:.0f} s")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
