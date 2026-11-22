#!/usr/bin/env python3
"""
analysis/OO-RHIC/refs/run_refs.py -- the other p+p 200 GeV models for the comparison, in the
tune's file layout under PROD_DIR/events/<name> (observables.py and ue_check.py read them):

    pp19             JETSCAPE's PP19 tune (arXiv:1910.05481): PythiaGun (ISR, MPI on, FSR off,
                     partons > 2 GeV) + Matter in vacuum (Q0 1, vir_factor 0.25) + Colorless;
                     prod_AuAu_0_10_jet/run_prod_vacuum.py with refs/pp19_200.xml and
                     refs/hadronize_pp19.xml
    pythia8_default  standalone Pythia 8.312 as it comes (Monash), refs/pythia_ref.py
    pythia8_detroit  STAR's Detroit tune, refs/pythia_ref.py --tune detroit

    python refs/run_refs.py -j 20               # all three, JOBS x config.EVENTS_PER_JOB each
    python refs/run_refs.py pp19 --dry-run

The pTHat windows are config.PTHAT_WINDOWS; pi0 and eta undecayed in all.  Resumable (a job
whose .json reports all events written is skipped).  Seeds: pp19 970000 + j, default
960000 + j, detroit 965000 + j.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import config as C  # noqa: E402

JOBS = 20
REFS = {"pp19": 970000, "pythia8_default": 960000, "pythia8_detroit": 965000}
VACUUM = C.EXAMPLE / "prod_AuAu_0_10_jet" / "run_prod_vacuum.py"


def windows_text():
    return ",".join(f"{lo:g}-{hi:g}" for lo, hi in C.PTHAT_WINDOWS)


def job(name, seed):
    out = C.point_dir(name)
    nwin = len(C.PTHAT_WINDOWS)
    events = C.EVENTS_PER_JOB - C.EVENTS_PER_JOB % nwin
    if name == "pp19":
        cmd = [sys.executable, str(VACUUM), "--events", str(events), "--seed", str(seed),
               "--user-xml", str(HERE / "pp19_200.xml"),
               "--hadronize-xml", str(HERE / "hadronize_pp19.xml"),
               "--pthat-bins", windows_text(), "--eta-max", str(C.ETA_MAX), "--n-frag", "1",
               "--outdir", str(out), "--seed-registry", "none"]
        js = out / f"vacuum_jet_seed{seed:04d}.json"
    else:
        tune = name.split("_", 1)[1]
        stem = out / f"{name}_seed{seed}_hadrons_jet_frag.h5"
        cmd = [sys.executable, str(HERE / "pythia_ref.py"), "--tune", tune,
               "--events", str(events), "--seed", str(seed), "--out", str(stem)]
        js = out / f"{name}_seed{seed}_hadrons_jet_frag.json"
    return cmd, out, js, events


def done(js, events):
    try:
        return json.loads(js.read_text()).get("events_written", 0) == events
    except (OSError, ValueError):
        return False


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("refs", nargs="*", default=list(REFS))
    p.add_argument("--jobs", type=int, default=JOBS, help="jobs per reference")
    p.add_argument("-j", type=int, default=os.cpu_count())
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    todo = []
    for name in a.refs:
        for j in range(a.jobs):
            cmd, out, js, events = job(name, REFS[name] + j)
            if not done(js, events):
                todo.append((name, REFS[name] + j, cmd, out))
    print(f"run_refs.py: {len(todo)} jobs to run, {a.j} at a time")
    if a.dry_run:
        for t in todo[:3]:
            print(" ".join(t[2]))
        return 0
    env = dict(os.environ, OMP_NUM_THREADS="1")
    t0 = time.time()

    def run(item):
        name, seed, cmd, out = item
        out.mkdir(parents=True, exist_ok=True)
        with open(out / f"{name}_seed{seed}.log", "w") as log:
            r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                               cwd=str(VACUUM.parent))
        return name, seed, r.returncode

    n_fail = 0
    with ThreadPoolExecutor(max_workers=a.j) as ex:
        for k, (name, seed, rc) in enumerate(ex.map(run, todo), 1):
            n_fail += rc != 0
            print(f"[{k}/{len(todo)}] {name} seed {seed}: "
                  + ("ok" if rc == 0 else f"FAILED (rc {rc})") + f", {time.time() - t0:.0f} s",
                  flush=True)
    print(f"run_refs.py: {len(todo) - n_fail} ok, {n_fail} failed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
