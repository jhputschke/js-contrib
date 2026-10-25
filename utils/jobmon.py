#!/usr/bin/env python3
"""
utils/jobmon.py -- watch a production of run_jobs.sh jobs: progress, rate, ETA, failures, checks.

It only reads what the jobs already write into their OUTDIR: <stem>.log (the driver's
"event i/N done in ... s" lines, warnings, tracebacks), <stem>.json (the job summary, written
at the end) and run_jobs.{prefix,campaign,finished}.  The production scripts and the images
are not involved, so it works on Docker, SLURM and macOS output alike, during a run or after.

    utils/jobmon.py /data/zjet_c02                  # OUTBASE: its gpu0/, gpu1/, ... are taken
    utils/jobmon.py '/data/zjet_c0*' --njobs 100    # several directories; jobs not started yet
    utils/jobmon.py /data/zjet_c02 --watch          # live view (rich), refreshed every 10 s
    utils/jobmon.py /data/zjet_c02 --all            # every job, not only running and problems

Per job: done / running / FAILED / short (a summary with fewer events than asked: run_jobs.sh
reruns it) / waiting (only with --njobs).  A job counts as running while its driver process is
alive (seen with ps: on the machine that runs the jobs, containers included) or, if no driver
process is visible at all (another machine, a cluster), while its log changed within --stale s.

Rate: the steady state, the sum over running jobs of 3600 / (s per event over their last 5
events), as the benchmarks count it; in --watch also the events seen finishing in the last
10 minutes.  ETA: remaining events / that rate.  The drivers' "done in" lines reach the log
in blocks (Python buffers its output to a file; MUSIC's lines come at once), so a job's event
count jumps by ~10-15 at a time unless the jobs run with PYTHONUNBUFFERED=1; "log idle" shows
that a job is alive meanwhile.  The s per event themselves are exact.

Checks: tracebacks, crashes and kills in the logs; check_deposition.py's rule (droplets per
event below half the campaign median, a rare silent failure); events at the grid edge; skipped
events; the most frequent warnings.  Exits 1 if a job failed or is flagged (for cron and
scripts), else 0.  Standard library only; --watch uses rich if installed (the analysis venv
has it), else plain text; --tui is a full terminal UI (jobmon_tui.py, needs textual).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_deposition import rows as deposition_rows  # noqa: E402

ANSI = re.compile(r"\x1b\[[0-9;]*m")
EVENT_DONE = re.compile(r"^(\S+): event (\d+)/(\d+) done in ([0-9.]+) s")
WARNING = re.compile(r"\[(Warning|Error)\]\s*(.*)")
FUNC = re.compile(r"(\w+::\w+)\(")
NUMBER = re.compile(r"[-+]?\d+(\.\d+)?([eE][-+]?\d+)?")
FATAL = re.compile(r"Traceback \(most recent call last\)|Segmentation fault|core dumped|"
                   r"\bKilled\b|MemoryError|terminate called|CUDA error|Aborted|"
                   r"out of memory|std::bad_alloc")
NOT_JOB_LOGS = re.compile(r"(\.init|^upload|^run_jobs.*)$")
RECENT = 5          # events per job for its current s/event
WINDOW = 600.0      # s, the --watch rate over events seen finishing


@dataclass
class LogState:
    """What has been read of one job's log so far (read incrementally)."""
    offset: int = 0
    rest: bytes = b""
    driver: str = ""
    done: int = 0
    total: int = 0
    times: deque = field(default_factory=lambda: deque(maxlen=RECENT))
    sum_s: float = 0.0
    fatal: str = ""
    warnings: Counter = field(default_factory=Counter)
    tail: deque = field(default_factory=lambda: deque(maxlen=5))

    def feed(self, path):
        try:
            size = os.path.getsize(path)
        except OSError:
            return
        if size < self.offset:                       # rewritten (a job rerun): start over
            self.__init__()
        if size == self.offset:
            return
        with open(path, "rb") as f:
            f.seek(self.offset)
            data = self.rest + f.read()
        self.offset = size
        lines = data.split(b"\n")
        self.rest = lines.pop()                      # an unfinished last line
        for raw in lines:
            line = ANSI.sub("", raw.decode("utf-8", "replace")).rstrip()
            if not line:
                continue
            self.tail.append(line)
            m = EVENT_DONE.match(line)
            if m:
                self.driver = m.group(1)
                self.done, self.total = int(m.group(2)), int(m.group(3))
                s = float(m.group(4))
                self.times.append(s)
                self.sum_s += s
                continue
            w = WARNING.search(line)
            if w:
                self.warnings[warning_key(w.group(1), w.group(2))] += 1
            elif not self.fatal and FATAL.search(line):
                self.fatal = line[:200]

    def s_per_event(self):
        return sum(self.times) / len(self.times) if self.times else None


def warning_key(kind, text):
    """'[Warning] void Jetscape::A::f(args) : msg 1.2 GeV' -> 'Warning A::f: msg N GeV'."""
    f = FUNC.search(text)
    msg = text.split(") : ", 1)[1] if ") : " in text else text
    msg = NUMBER.sub("N", msg)[:110]
    return f"{kind} {f.group(1)}: {msg}" if f else f"{kind}: {msg}"


@dataclass
class Job:
    stem: str
    outdir: str
    log: LogState = field(default_factory=LogState)
    summary: dict | None = None
    state: str = "?"
    reason: str = ""
    log_age: float | None = None
    ended: bool = False                  # run_jobs.sh finished after the log's last change
    pid: int | None = None
    asked: int = 0                       # --events of its running driver

    @property
    def done(self):
        if self.summary:
            return self.summary.get("events_written", 0)
        return self.log.done

    @property
    def total(self):
        if self.summary:
            return self.summary.get("events_requested", self.log.total)
        return self.log.total or self.asked


def complete(r):
    """The test run_jobs.sh uses to skip a finished job (events skipped on purpose count)."""
    ev, sk = r.get("events_requested"), r.get("events_skipped", 0)
    pw = r.get("particlize_events_written")
    return (ev is not None and r.get("events_written", -1) + sk == ev
            and (pw is None or pw + sk == ev))


def job_dirs(patterns):
    """Directories holding job output; a directory without any takes its subdirectories."""
    def has_jobs(d):
        return (glob.glob(os.path.join(d, "*.log")) or glob.glob(os.path.join(d, "*.json"))
                or os.path.exists(os.path.join(d, "run_jobs.prefix")))
    out = []
    for pat in patterns:
        for d in sorted(glob.glob(pat)) or [pat]:
            if not os.path.isdir(d):
                sys.exit(f"jobmon: {d}: no such directory")
            if has_jobs(d):
                out.append(d)
            else:
                out += [s for s in sorted(glob.glob(os.path.join(d, "*")))
                        if os.path.isdir(s) and has_jobs(s)]
    return list(dict.fromkeys(out))


def driver_processes():
    """{stem: (pid, events asked)} of the running drivers (run_prod*.py) that can be tied to a
    job: on Linux by their stdout, which run_jobs.sh points at <stem>.log (any run_jobs.sh
    version, containers included); else by --out <stem>.h5 (run_jobs.sh of 2026-10-16 on).
    A driver started some other way is not tied to a job; the logs' age decides then."""
    try:
        ps = subprocess.run(["ps", "-axww", "-o", "pid=,command="], capture_output=True,
                            text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    procs = {}
    for line in ps.splitlines():
        pid, _, cmd = line.strip().partition(" ")
        if not re.search(r"python\S*\s+\S*run_prod\S*\.py\s", cmd):
            continue
        stem = None
        try:
            out = os.readlink(f"/proc/{pid}/fd/1")
            if out.endswith(".log"):
                stem = os.path.basename(out)[:-4]
        except OSError:
            pass
        m = re.search(r"--out\s+(\S+)\.h5", cmd)
        stem = stem or (os.path.basename(m.group(1)) if m else None)
        if stem:
            ev = re.search(r"--events\s+(\d+)", cmd)
            procs[stem] = (int(pid), int(ev.group(1)) if ev else 0)
    return procs


class Monitor:
    def __init__(self, a):
        self.a = a
        self.dirs = job_dirs(a.paths)
        if not self.dirs:
            sys.exit(f"jobmon: no run_jobs.sh output in {' '.join(a.paths)}")
        self.jobs: dict[str, Job] = {}
        self.seen: deque = deque()                   # (time, events) seen finishing
        self.t_start = time.time()
        self.scans = 0

    def scan(self):
        now = time.time()
        procs = driver_processes()
        visible = bool(procs)
        for d in self.dirs:
            fin = os.path.join(d, "run_jobs.finished")
            t_fin = os.path.getmtime(fin) if os.path.exists(fin) else None
            stems = set()
            for p in glob.glob(os.path.join(d, "*.log")):
                s = os.path.basename(p)[:-4]
                if not NOT_JOB_LOGS.search(s):
                    stems.add(s)
            stems |= {os.path.basename(p)[:-5] for p in glob.glob(os.path.join(d, "*.json"))}
            for s in stems:
                key = os.path.join(d, s)
                job = self.jobs.setdefault(key, Job(s, d))
                before = job.log.done
                logp = key + ".log"
                job.log.feed(logp)
                if job.log.done > before and self.scans:   # not what was there at the start
                    self.seen.append((now, job.log.done - before))
                t_log = os.path.getmtime(logp) if os.path.exists(logp) else None
                job.log_age = now - t_log if t_log is not None else None
                job.ended = t_fin is not None and t_log is not None and t_fin >= t_log
                if job.summary is None or not complete(job.summary):
                    job.summary = self.read_summary(key + ".json")
                job.pid, asked = procs.get(s, (None, 0))
                job.asked = asked or job.asked
                self.classify(job, visible)
        while self.seen and self.seen[0][0] < now - WINDOW:
            self.seen.popleft()
        self.scans += 1

    @staticmethod
    def read_summary(p):
        try:
            with open(p) as f:
                r = json.load(f)
        except (OSError, ValueError):
            return None
        return r if isinstance(r, dict) and "events_requested" in r else None

    def classify(self, job, procs_visible):
        r, lg = job.summary, job.log
        job.reason = ""
        if r is not None and complete(r):
            job.state = "done"
        elif job.pid is not None or (not procs_visible and not job.ended
                                     and job.log_age is not None and job.log_age < self.a.stale):
            job.state = "running"
        elif r is not None:
            job.state = "short"
            job.reason = (f"summary has {r.get('events_written')} of "
                          f"{r.get('events_requested')} events; run_jobs.sh reruns it")
        else:
            job.state = "FAILED"
            if lg.fatal:
                job.reason = lg.fatal
            elif job.log_age is not None:
                job.reason = f"ended without a summary; log idle {fmt_dt(job.log_age)}"
            else:
                job.reason = "no log"

    # -- the report --------------------------------------------------------------------------
    def report(self):
        jobs = sorted(self.jobs.values(), key=lambda j: (j.outdir, j.stem))
        by = Counter(j.state for j in jobs)
        waiting = max(0, self.a.njobs - len(jobs)) if self.a.njobs else 0
        totals = [j.total for j in jobs if j.total]
        per_job = statistics.median(totals) if totals else 0
        done_ev = sum(j.done for j in jobs)
        all_ev = sum(j.total or per_job for j in jobs) + waiting * per_job
        running = [j for j in jobs if j.state == "running"]
        rate = sum(3600 / s for j in running if (s := j.log.s_per_event()))
        seen_rate, span = None, min(WINDOW, time.time() - self.t_start)
        if self.scans > 1 and span > 120:
            seen_rate = sum(n for _, n in self.seen) * 3600 / span
        left = max(0, all_ev - done_ev)
        problems = [(j, j.reason) for j in jobs if j.state in ("FAILED", "short")]
        flagged, notes = self.checks(jobs)
        problems += flagged
        warn = Counter()
        for j in jobs:
            warn.update(j.log.warnings)
        return dict(jobs=jobs, by=by, waiting=waiting, done_ev=done_ev, all_ev=all_ev,
                    running=running, rate=rate, seen_rate=seen_rate, span=span, notes=notes,
                    eta=left * 3600 / rate if rate and left else None,
                    problems=problems, warnings=warn, disk=self.disk(jobs),
                    campaigns=self.campaigns(), machine=self.machine())

    def checks(self, jobs):
        """(flagged, notes): jobs to redo, and facts worth knowing (edge hits, skips)."""
        out, notes = [], []
        dep = deposition_rows(self.dirs)
        if len(dep) >= 3:
            med = statistics.median(x for x, _, _ in dep)
            for x, n, p in dep:
                if x < 0.5 * med:
                    j = self.jobs.get(p[:-5])
                    out.append((j, f"deposition: {x:.1f} droplets per event, campaign median "
                                   f"{med:.1f} (check_deposition.py rule: redo the job)"))
        for j in jobs:
            r = j.summary or {}
            edge = r.get("events_hit_edge") or {}
            n_edge = sum(v for v in edge.values() if v) if isinstance(edge, dict) else edge
            if n_edge:
                legs = (", ".join(f"{k} {v}" for k, v in edge.items() if v)
                        if isinstance(edge, dict) else str(edge))
                notes.append((j, f"grid edge: fluid above freeze-out at the edge of the stored "
                                 f"box in {legs} event(s)"))
            if r.get("events_skipped"):
                errs = Counter(s.get("error", "?") for s in r.get("skipped", []))
                notes.append((j, f"skipped {r['events_skipped']} event(s) on purpose: "
                                 + "; ".join(f"{e} x{n}" for e, n in errs.items())))
        return out, notes

    def campaigns(self):
        names = set()
        for d in self.dirs:
            p = os.path.join(d, "run_jobs.campaign")
            if os.path.exists(p):
                with open(p) as f:
                    names.add(f.read().strip())
        return sorted(names)

    def disk(self, jobs):
        size, done_size = 0, 0
        done = {(j.outdir, j.stem) for j in jobs if j.state == "done"}
        for d in self.dirs:
            with os.scandir(d) as it:
                for e in it:
                    if not e.is_file():
                        continue
                    b = e.stat().st_size
                    size += b
                    stem = re.sub(r"(_hadrons_\w+|_particlize|_primary)*\.\w+$", "", e.name)
                    if (d, stem) in done:
                        done_size += b
        done_ev = sum(j.done for j in jobs if j.state == "done")
        free = min(shutil.disk_usage(d).free for d in self.dirs)
        return dict(size=size, free=free, per_event=done_size / done_ev if done_ev else None)

    def machine(self):
        if self.a.no_machine:
            return None
        m = {}
        try:
            q = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,"
                                "memory.total", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=10)
            m["gpus"] = [tuple(x.strip() for x in l.split(",")) for l in q.stdout.splitlines()
                         if l.strip()]
        except (OSError, subprocess.SubprocessError):
            pass
        try:
            with open("/proc/meminfo") as f:
                mi = {l.split(":")[0]: int(l.split()[1]) for l in f}
            m["mem"] = ((mi["MemTotal"] - mi["MemAvailable"]) / 2**20, mi["MemTotal"] / 2**20)
        except (OSError, KeyError, ValueError):
            pass
        m["load"] = os.getloadavg()[0]
        m["cores"] = os.cpu_count()
        return m


# -- formatting -------------------------------------------------------------------------------
def fmt_dt(s):
    s = int(s)
    if s < 120:
        return f"{s} s"
    if s < 7200:
        return f"{s // 60} min"
    return f"{s // 3600} h {s % 3600 // 60:02d} min"


def fmt_b(n):
    for u in ("B", "kB", "MB", "GB", "TB"):
        if n < 1000 or u == "TB":
            return f"{n:.1f} {u}" if u != "B" else f"{n} B"
        n /= 1000


def summary_lines(r):
    by = r["by"]
    head = (", ".join(r["campaigns"]) + ": " if r["campaigns"] else "")
    out = [head + f"{len(r['jobs']) + r['waiting']} jobs: {by['done']} done, "
           f"{by['running']} running, {by['FAILED']} failed"
           + (f", {by['short']} short" if by["short"] else "")
           + (f", {r['waiting']} waiting" if r["waiting"] else "")]
    pct = 100 * r["done_ev"] / r["all_ev"] if r["all_ev"] else 0
    if r["rate"]:
        rate = f"{r['rate']:,.0f} events/h (running jobs, last {RECENT} events each)"
    elif r["running"]:
        rate = "not yet (no running job has finished an event)"
    else:
        rate = "none (no job running)"
    if r["seen_rate"] is not None:
        rate += f"; {r['seen_rate']:,.0f} seen finishing in the last {fmt_dt(r['span'])}"
    out.append(f"events: {r['done_ev']:,} of {r['all_ev']:,.0f} ({pct:.0f} %); rate {rate}"
               + (f"; ETA {fmt_dt(r['eta'])}" if r["eta"] else ""))
    d = r["disk"]
    out.append(f"disk: {fmt_b(d['size'])} written"
               + (f", {fmt_b(d['per_event'])} per event" if d["per_event"] else "")
               + f"; {fmt_b(d['free'])} free")
    m = r["machine"]
    if m:
        parts = [f"GPU{i} {u} % busy, {int(mu) / 1024:.1f}/{int(mt) / 1024:.0f} GB"
                 for i, u, mu, mt in m.get("gpus", [])]
        if "mem" in m:
            parts.append(f"host memory {m['mem'][0]:.0f}/{m['mem'][1]:.0f} GB")
        parts.append(f"load {m['load']:.1f} of {m['cores']} threads")
        out.append("this machine: " + "; ".join(parts))
    return out


def job_row(j):
    s = j.log.s_per_event()
    eta = (j.total - j.done) * s if s and j.total else None
    idle = fmt_dt(j.log_age) if j.log_age is not None and j.state == "running" else "-"
    return (j.stem, os.path.basename(j.outdir), f"{j.done}/{j.total or '?'}",
            f"{s:.1f}" if s else "-", fmt_dt(eta) if eta else "-", idle,
            str(sum(j.log.warnings.values())))


ROW_HEAD = ("job", "dir", "events", "s/event", "ETA", "log idle", "warnings")


def print_text(r, a):
    print("\n".join(summary_lines(r)))
    rows = [j for j in r["jobs"] if a.all or j.state == "running"]
    if rows:
        cells = [(j.stem, j.state) + job_row(j)[1:] for j in rows]
        head = ("job", "state") + ROW_HEAD[1:]
        wid = [max(len(c[i]) for c in cells + [head]) for i in range(len(head))]
        print()
        for c in [head] + cells:
            print("  ".join(x.ljust(w) if i < 3 else x.rjust(w)
                            for i, (x, w) in enumerate(zip(c, wid))))
    if r["problems"]:
        print("\nproblems:")
        for j, why in r["problems"]:
            print(f"  {j.stem if j else '?'} [{j.state if j else '?'}]: {why}")
            if j and j.state == "FAILED" and a.tail:
                for line in list(j.log.tail)[-a.tail:]:
                    print(f"      | {line[:150]}")
    if r["notes"]:
        print("\nnotes:")
        for j, why in r["notes"]:
            print(f"  {j.stem}: {why}")
    if r["warnings"]:
        print("\nmost frequent warnings (all jobs):")
        for k, n in r["warnings"].most_common(a.top_warnings):
            print(f"  {n:>7}  {k}")


def rich_view(r, a):
    from rich.console import Group
    from rich.table import Table
    from rich.text import Text

    parts = [Text("\n".join(summary_lines(r)))]
    rows = [j for j in r["jobs"] if a.all or j.state == "running"]
    if rows:
        t = Table(title="jobs" if a.all else "running jobs", title_justify="left",
                  expand=False, show_edge=False)
        t.add_column("job")
        t.add_column("state")
        for h in ROW_HEAD[1:]:
            t.add_column(h, justify="right")
        style = {"done": "green", "running": "cyan", "FAILED": "bold red", "short": "yellow"}
        for j in rows:
            n, *rest = job_row(j)
            t.add_row(n, Text(j.state, style=style.get(j.state, "")), *rest)
        parts.append(t)
    if r["problems"]:
        t = Table(title="problems", title_justify="left", show_edge=False)
        t.add_column("job", style="bold")
        t.add_column("state")
        t.add_column("what", overflow="fold")
        for j, why in r["problems"]:
            t.add_row(j.stem if j else "?", j.state if j else "?", why)
        parts.append(t)
    if r["notes"]:
        t = Table(title="notes", title_justify="left", show_edge=False)
        t.add_column("job")
        t.add_column("what", overflow="fold")
        for j, why in r["notes"]:
            t.add_row(j.stem, why)
        parts.append(t)
    if r["warnings"]:
        t = Table(title="most frequent warnings", title_justify="left", show_edge=False)
        t.add_column("count", justify="right")
        t.add_column("warning", overflow="fold")
        for k, n in r["warnings"].most_common(a.top_warnings):
            t.add_row(f"{n:,}", k)
        parts.append(t)
    parts.append(Text(f"{time.strftime('%H:%M:%S')}  refresh every {a.watch:g} s, Ctrl-C to "
                      "stop", style="dim"))
    return Group(*parts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="OUTDIRs, or an OUTBASE whose subdirectories "
                    "hold the jobs (globs allowed)")
    ap.add_argument("--njobs", type=int, default=0,
                    help="jobs in the campaign, to count those not started yet")
    ap.add_argument("--watch", type=float, nargs="?", const=10.0, default=None, metavar="S",
                    help="live view, refreshed every S s (default 10)")
    ap.add_argument("--tui", action="store_true",
                    help="the terminal UI (jobmon_tui.py; needs textual): sortable job table, "
                         "a job's details and live log, problems, warnings; refreshed every "
                         "--watch S s (default 10)")
    ap.add_argument("--all", action="store_true", help="list every job, not only running ones")
    ap.add_argument("--stale", type=float, default=900.0, metavar="S",
                    help="without visible driver processes, a job whose log changed within S s "
                         "counts as running (default 900)")
    ap.add_argument("--tail", type=int, default=5, metavar="N",
                    help="last N log lines of each failed job (default 5; 0: none)")
    ap.add_argument("--top-warnings", type=int, default=5, metavar="N",
                    help="the N most frequent warnings (default 5)")
    ap.add_argument("--no-machine", action="store_true",
                    help="no GPU, memory and load of this machine")
    a = ap.parse_args(argv)
    mon = Monitor(a)
    if a.tui:
        try:
            from jobmon_tui import run_tui
        except ImportError as e:
            sys.exit(f"jobmon: --tui needs textual ({e}); pip install textual, or use --watch")
        return run_tui(mon, a.watch or 10.0)
    if a.watch is None:
        mon.scan()
        r = mon.report()
        try:
            print_text(r, a)
            sys.stdout.flush()
        except BrokenPipeError:                      # e.g. | head
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 1 if r["problems"] else 0
    try:
        from rich.live import Live
    except ImportError:
        Live = None
    try:
        if Live is None:
            while True:
                mon.scan()
                print("\033[2J\033[H", end="")
                print_text(mon.report(), a)
                time.sleep(a.watch)
        with Live(auto_refresh=False, screen=False) as live:
            while True:
                mon.scan()
                live.update(rich_view(mon.report(), a), refresh=True)
                time.sleep(a.watch)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
