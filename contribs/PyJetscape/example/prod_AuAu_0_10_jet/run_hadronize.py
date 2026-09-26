#!/usr/bin/env python3
"""
example/prod_AuAu_0_10_jet/run_hadronize.py

Hadronize a campaign: hadronize.py on every complete ``*_particlize.h5``, P at a time.  The
run_jobs.sh counterpart for the hadronization step (PLAN_particlize_h5.md).

    ./run_hadronize.py out_had -j 4 --oversample 500 --n-frag 50
    ./run_hadronize.py out_had -j 4 --oversample 200 --oversample-bg auto      # --reuse runs
    ./run_hadronize.py out_had -j 4 --follow --oversample 500 --n-frag 50      # during
                                                                               # run_jobs.sh
    ./run_hadronize.py out_had --dry-run --oversample 500                      # the plan only

INPUTS are directories (their ``*_particlize.h5``), particlize files, or glob patterns.
Every option not listed under "run_hadronize options" goes to each hadronize.py unchanged
(--oversample, --oversample-bg, --n-frag, --tags, --seed, --events, --out-dir,
--hadronize-xml, --build, ...); they are checked once before anything starts.

- **Only complete inputs.**  A particlize file is written event by event while its job runs
  and marked ``complete`` when the job ends; only complete files are hadronized.  Files
  still being written are reported as incomplete (without --follow) or waited for.
- **Restarting.**  ``--skip-complete`` is passed unless you pass ``--force``: files whose
  hadron outputs are all complete are skipped without starting a process, missing or
  incomplete outputs are redone.  Re-running the same command finishes an interrupted pass.
- **--follow** hadronizes files as a running ``run_jobs.sh`` campaign completes them.  It
  stops when every input directory holds ``run_jobs.finished`` (written by run_jobs.sh at
  the end of a campaign) and nothing is left to do, or after ``--idle-exit`` minutes with no
  new work and nothing running.
- **Logs** are appended to ``<stem>_hadronize.log`` next to the outputs.  A summary at the
  end lists what was done, skipped, failed and left incomplete; the exit code is 1 if a
  hadronize.py run failed.  Ctrl-C stops the running processes: they close their outputs as
  incomplete (within one iSS pass, up to ~90 s; a second Ctrl-C kills them at once), and
  the next run redoes those files.
- **Memory.**  Each process needs ~1.4 GB plus ~2.4 MB per iSS oversample (of its largest
  surface, e.g. an --oversample-bg auto background).  A warning is printed when P processes
  would not fit into the available memory.

hadronize.py runs on one core, so P up to the number of free cores is useful; while GPU
jobs run on the same machine, leave them their cores (on the GB10 3-4 processes keep up
with a four-job campaign).
"""

from __future__ import annotations

import argparse
import collections
import glob
import importlib.util
import os
import re
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
HADRONIZE = os.path.join(HERE, "hadronize.py")

_spec = importlib.util.spec_from_file_location("_hadronize", HADRONIZE)
hz = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hz)                # also puts PyJetscape/python on sys.path

FINISHED_MARKER = "run_jobs.finished"
GB_BASE, GB_PER_OVERSAMPLE, GB_FRAG_ONLY = 1.4, 0.0024, 0.7


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="%(prog)s INPUTS... [-j P] [--follow] [hadronize.py options]")
    g = p.add_argument_group("run_hadronize options")
    g.add_argument("inputs", nargs="+", help="directories, particlize files or globs")
    g.add_argument("-j", "--jobs", type=int, default=1,
                   help="hadronize.py processes at once (default 1)")
    g.add_argument("--follow", action="store_true",
                   help="keep watching the inputs and hadronize files as they complete")
    g.add_argument("--idle-exit", type=float, default=30.0, dest="idle_exit",
                   help="--follow: stop after this many minutes without new work and with "
                        "nothing running (default 30)")
    g.add_argument("--poll", type=float, default=30.0,
                   help="--follow: seconds between scans of the inputs (default 30)")
    g.add_argument("--dry-run", action="store_true", dest="dry_run",
                   help="show what would be done and exit")
    a, passthrough = p.parse_known_args(argv)
    if a.jobs < 1:
        p.error("-j must be >= 1")
    return a, passthrough


def scan(inputs):
    """-> sorted particlize paths named by the inputs."""
    found = set()
    for item in inputs:
        if os.path.isdir(item):
            found.update(glob.glob(os.path.join(item, "*_particlize.h5")))
        elif any(c in item for c in "*?["):
            found.update(p for p in glob.glob(item) if p.endswith("_particlize.h5"))
        elif os.path.exists(item):
            found.add(item)
    return sorted(os.path.abspath(p) for p in found)


def is_complete(particlize):
    """True once the production job closed the particlize file as complete."""
    import h5py
    try:
        with h5py.File(particlize, "r") as f:
            return bool(f.attrs.get("complete", False))
    except OSError:                              # still open for writing, or truncated
        return False


class Plan:
    """What hadronize.py will be asked to do: tags, outputs, memory per process."""

    def __init__(self, passthrough):
        # hadronize.py's own parser checks the options once, before anything starts
        self.h = hz.parse_args(["CHECK_particlize.h5"] + passthrough)
        self.tags = [t.strip() for t in self.h.tags.split(",") if t.strip()]
        bad = set(self.tags) - set(hz.TAGS)
        if bad:
            raise SystemExit(f"run_hadronize.py: unknown tag(s) {sorted(bad)}")
        self.args = list(passthrough)
        if "--force" not in self.args and "--skip-complete" not in self.args:
            self.args.append("--skip-complete")
        import xml.etree.ElementTree as ET
        self.oversample = self.h.oversample or int(ET.parse(self.h.hadronize_xml).getroot()
                                                   .find("SoftParticlization/iSS/"
                                                         "number_of_repeated_sampling").text)

    def outputs(self, particlize):
        stem = os.path.basename(particlize)[:-len("_particlize.h5")]
        out_dir = (os.path.abspath(self.h.out_dir) if self.h.out_dir
                   else os.path.dirname(particlize))
        return out_dir, stem, {t: os.path.join(out_dir, f"{stem}_hadrons_{t}.h5")
                               for t in self.tags}

    def all_done(self, particlize):
        if "--force" in self.args:
            return False
        _, _, outs = self.outputs(particlize)
        return all(hz._complete(p) for p in outs.values())

    def memory_gb(self, particlize):
        """Rough peak memory of one hadronize.py on this file."""
        if not {"bulk_jet", "bulk_bg"} & set(self.tags):
            return GB_FRAG_ONLY
        n = self.oversample if "bulk_jet" in self.tags else 0
        if "bulk_bg" in self.tags:
            try:
                from jetscape.particlize_h5 import ParticlizeFile
                with ParticlizeFile(particlize) as pf:
                    bg, _ = hz.background_samples(pf, self.oversample, self.h.oversample_bg,
                                                  self.h.oversample_bg_max)
                n = max([n] + list(bg.values()))
            except Exception:                    # noqa: BLE001 - an estimate only
                n = max(n, self.oversample)
        return GB_BASE + GB_PER_OVERSAMPLE * n


def available_memory_gb():
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024 ** 2
    except OSError:
        pass
    try:                                         # macOS: free + inactive + speculative pages
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
        page = int(re.search(r"page size of (\d+) bytes", out).group(1))
        pages = {k.strip(): int(v.strip().rstrip("."))
                 for k, v in (l.split(":", 1) for l in out.splitlines()[1:] if ":" in l)}
        n = sum(pages.get(f"Pages {k}", 0) for k in ("free", "inactive", "speculative"))
        return n * page / 1024 ** 3
    except (OSError, ValueError, AttributeError, subprocess.CalledProcessError):
        pass
    return None


def finished_marker_present(inputs):
    dirs = [i for i in inputs if os.path.isdir(i)] or sorted(
        {os.path.dirname(os.path.abspath(p)) for p in scan(inputs)})
    return bool(dirs) and all(os.path.exists(os.path.join(d, FINISHED_MARKER)) for d in dirs)


def main(argv=None):
    a, passthrough = parse_args(argv)
    plan = Plan(passthrough)
    t_start = time.time()
    status = {}                                  # particlize path -> state
    queue = collections.deque()
    running = {}                                 # path -> (Popen, log handle, start time)
    results = collections.OrderedDict()          # path -> (state, seconds)
    mem_warned = False
    last_activity = time.time()
    last_scan = [0.0]

    def log(msg):
        print(f"[{time.strftime('%F %T')}] {msg}", flush=True)

    def refresh():
        nonlocal last_activity, mem_warned
        last_scan[0] = time.time()
        for p in scan(a.inputs):
            if status.get(p) not in (None, "incomplete"):
                continue
            if not is_complete(p):
                status[p] = "incomplete"
                continue
            if plan.all_done(p):
                status[p] = "already complete"
                results[p] = ("already complete", 0.0)
                continue
            status[p] = "queued"
            queue.append(p)
            last_activity = time.time()
            if not mem_warned:
                need, have = a.jobs * plan.memory_gb(p), available_memory_gb()
                if have is not None and need > have:
                    log(f"WARNING -- -j {a.jobs} x ~{plan.memory_gb(p):.1f} GB per process "
                        f"= ~{need:.0f} GB, but only ~{have:.0f} GB are available; lower -j "
                        "or the oversample count")
                mem_warned = True

    def start(p):
        out_dir, stem, _ = plan.outputs(p)
        os.makedirs(out_dir, exist_ok=True)
        fh = open(os.path.join(out_dir, f"{stem}_hadronize.log"), "a")
        cmd = [sys.executable, HADRONIZE, p] + plan.args
        fh.write(f"\n=== {time.strftime('%F %T')} run_hadronize.py: {' '.join(cmd)}\n")
        fh.flush()
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                start_new_session=True)
        running[p] = (proc, fh, time.time())
        status[p] = "running"
        log(f"start  {stem} ({len(running)} running, {len(queue)} queued)")

    def reap():
        nonlocal last_activity
        for p, (proc, fh, t0) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            fh.close()
            del running[p]
            dt = time.time() - t0
            state = "done" if rc == 0 else f"failed (exit {rc})"
            status[p] = state
            results[p] = (state, dt)
            last_activity = time.time()
            log(f"{'done  ' if rc == 0 else 'FAILED'} {os.path.basename(p)[:-len('_particlize.h5')]}"
                f" in {dt:.0f} s" + ("" if rc == 0 else f" (see its _hadronize.log)"))

    def stop_children(grace=90):
        # hadronize.py turns SIGTERM into an exception and closes its files as incomplete,
        # but only once control is back in Python: one iSS pass can take ~40 s.
        for proc, fh, _ in running.values():
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        log(f"waiting up to {grace} s for them to close their files (Ctrl-C again: kill now)")
        deadline = time.time() + grace
        for proc, fh, _ in running.values():
            try:
                proc.wait(timeout=max(0.1, deadline - time.time()))
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            fh.close()

    refresh()
    if a.dry_run:
        by = collections.Counter(status.values())
        print(f"run_hadronize.py (dry run): {len(status)} particlize file(s): "
              + ", ".join(f"{n} {s}" for s, n in sorted(by.items())))
        for p, s in status.items():
            print(f"  {s:17s} {p}")
        if queue:
            print("each queued file runs:\n  " + " ".join(
                [sys.executable, HADRONIZE, "<particlize>"] + plan.args))
            print(f"estimated memory per process: ~{plan.memory_gb(queue[0]):.1f} GB, "
                  f"x -j {a.jobs}")
        return 0

    log(f"run_hadronize.py: {len(status)} particlize file(s) found, -j {a.jobs}"
        + (", following" if a.follow else "") + f"; hadronize.py {' '.join(plan.args)}")
    interrupted = False
    try:
        while True:
            reap()
            while queue and len(running) < a.jobs:
                start(queue.popleft())
            idle = not queue and not running
            if not a.follow:
                if idle:
                    break
            else:
                waiting = [p for p, s in status.items() if s == "incomplete"]
                if idle and finished_marker_present(a.inputs):
                    refresh()                    # the last files may have just completed
                    if not queue:
                        break
                    continue
                if idle and time.time() - last_activity > 60 * a.idle_exit:
                    log(f"no new work for {a.idle_exit:g} min"
                        + (f" ({len(waiting)} file(s) still incomplete)" if waiting else "")
                        + "; stopping")
                    break
            time.sleep(1.0 if running else (min(a.poll, 5.0) if a.follow else 0.2))
            if a.follow and len(running) < a.jobs and time.time() - last_scan[0] >= a.poll:
                refresh()
    except KeyboardInterrupt:
        interrupted = True
        log(f"interrupted: stopping {len(running)} running hadronize.py process(es)")
        stop_children()

    # summary
    by = collections.Counter(s.split(" (")[0] for s, _ in results.values())
    incomplete = sorted(p for p, s in status.items() if s == "incomplete")
    failed = [p for p, (s, _) in results.items() if s.startswith("failed")]
    log(f"summary after {time.time() - t_start:.0f} s: "
        f"{by.get('done', 0)} hadronized, {by.get('already complete', 0)} already complete, "
        f"{len(failed)} failed, {len(incomplete)} incomplete (not hadronized)")
    for p in failed:
        print(f"  failed:     {p}")
    for p in incomplete:
        print(f"  incomplete: {p}")
    if interrupted:
        return 130
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
