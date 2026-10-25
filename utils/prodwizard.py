#!/usr/bin/env python3
"""
utils/prodwizard.py -- plan a production: the launch command, its sizing and the checks.

Pick a production preset and a machine; it writes the command that launches the campaign
there (utils/launch_2gpu.sh on the 2 x RTX 3090 box, run_jobs.sh on the GB10 and the Mac,
utils/slurm_prod_array.sh on a cluster) with the settings measured for that machine, and
estimates events/h, duration, disk and memory, each marked measured or estimated, with its
source.  It checks the plan before anything runs: events per job a multiple of the pT-hat
windows x jets, memory, disk and upload rate, the image, the campaign name.  It never starts
jobs: print the plan, or save it as a launch script to read and run.

    utils/prodwizard.py                                   # the wizard (needs textual)
    utils/prodwizard.py --list                            # presets and machines
    utils/prodwizard.py --preset auau_jet --machine 2x3090 --events 2700 --campaign c5 \\
        --out /data/AuAu_c5                               # print the plan
    utils/prodwizard.py --preset oo_hadrons --machine slurm_a40 --events 200000 --gpus 20 \\
        --campaign OO_c3 --save launch_OO_c3.sh           # a script to submit

The presets are utils/prod_presets.json: productions (driver arguments, events per job,
memory per job, disk per event) and machines (launch settings and rates per production, with
their sources).  Update them there after a new benchmark.  Standard library only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import shutil
import sys
import time
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PRESETS = os.path.join(HERE, "prod_presets.json")
CONTAINER_EXAMPLE = "/opt/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape/example"
PROD_DIRS = {"auau": "prod_AuAu_0_10_jet", "oo": "prod_OO_isr"}
LBT_CONTAINER = "/work/lbt_cache/lbt_tables_v1.bin"
UPLOAD_GB_H = 290          # the OSDF upload measured ~80 MB/s (REPORT_SUMMARY_CLUSTER.md)
NODE_BASE_GB = 8           # per machine or node besides the jobs
NAME_OK = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]*$")


def load_presets(path=PRESETS):
    with open(path) as f:
        return json.load(f)


@dataclass
class Options:
    production: str = "auau_jet"
    machine: str = "2x3090"
    events: int = 2700                 # total events asked for
    events_per_job: int = 0            # 0: the preset's
    campaign: str = ""
    out: str = ""                      # OUTBASE (docker), OUTDIR (native), WORK (slurm)
    image: str = ""                    # "" : the preset's pinned image
    first_seed: int = 0                # 0: unique seeds from the registry
    gpus: int = 10                     # slurm: array tasks (one GPU each)
    jobs_per_gpu: int = 0              # 0: the machine's measured setting
    omp: int = 0                       # 0: the machine's
    upload: str = ""                   # docker: osdf:///fno4hic/<upload> while running
    delete: str = "none"               # with upload: none | pair | all
    min_free: str = "100G"             # docker: the disk guard


@dataclass
class Plan:
    commands: list = field(default_factory=list)      # shell lines, in order
    prereq: list = field(default_factory=list)        # once, before the campaign
    estimates: list = field(default_factory=list)     # (label, value, basis)
    checks: list = field(default_factory=list)        # (level, text): ok | warn | error
    notes: list = field(default_factory=list)
    sources: list = field(default_factory=list)
    summary: str = ""
    njobs: int = 0
    events: int = 0

    @property
    def ok(self):
        return not any(lvl == "error" for lvl, _ in self.checks)


def fmt_h(h):
    if h < 1:
        return f"{h * 60:.0f} min"
    if h < 48:
        return f"{h:.1f} h"
    return f"{h / 24:.1f} days"


def fmt_gb(gb):
    return f"{gb / 1000:.2f} TB" if gb >= 1000 else (f"{gb:.0f} GB" if gb >= 10 else f"{gb:.1f} GB")


def rng(lo, hi, f=lambda x: f"{x:,.0f}"):
    return f(lo) if abs(hi - lo) < 1e-9 * max(1, abs(hi)) else f"{f(lo)}-{f(hi)}"


def this_machine():
    """The preset machine the wizard runs on, as far as it can tell (None if none)."""
    if sys.platform == "darwin":
        return "m3max" if "Max" in os.popen("sysctl -n machdep.cpu.brand_string").read() else None
    try:
        out = os.popen("nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null").read()
    except OSError:
        return None
    names = [x.strip() for x in out.splitlines() if x.strip()]
    if len(names) == 2 and all("3090" in n for n in names):
        return "2x3090"
    if names and "GB10" in names[0]:
        return "gb10"
    return None


def cont(args):
    """The driver arguments as a continuation line ('' without any)."""
    return (" \\\n  " + " ".join(map(shlex.quote, args))) if args else ""


def slurm_mem(jobs, prod):
    """--mem per task: the jobs' plan figure + the node base + 12 % page cache, in 4 GB steps."""
    need = (jobs * prod["mem_gb_per_job"] + NODE_BASE_GB) * 1.12
    return int(math.ceil(need * 1.1 / 4) * 4)


def windows(args):
    """K x M of a pT-hat window campaign from its driver arguments (1 without windows)."""
    k = m = 1
    if "--pthat-bins" in args:
        k = len(args[args.index("--pthat-bins") + 1].split(","))
        m = int(args[args.index("--jets-per-bin") + 1]) if "--jets-per-bin" in args else 1
    return k, m


def build(o: Options, presets=None) -> Plan:
    P = presets or load_presets()
    plan = Plan()
    prod = P["productions"][o.production]
    mach = P["machines"][o.machine]
    kind = mach["kind"]
    st = dict(mach.get("settings", {}).get(o.production) or {})
    chk = plan.checks.append
    if not st:
        st = {"jobs_per_gpu": 3 if prod["mem_gb_per_job"] > 10 else 8, "rate": None}
        chk(("warn", f"no measured or documented settings for {o.production} on {o.machine}: "
                     f"{st['jobs_per_gpu']} jobs per GPU assumed (--jobs-per-gpu), rate "
                     "unknown; measure one round first"))

    # -- sizes ---------------------------------------------------------------------------
    k, m = windows(prod["args"])
    mult = prod.get("events_multiple", 1)
    epj = o.events_per_job or prod["events_per_job"]
    if epj % mult:
        chk(("error", f"events per job {epj} is not a multiple of {mult} "
                      f"({k} pT-hat windows x {m} jets per background); use "
                      f"{math.ceil(epj / mult) * mult}"))
    jpg = o.jobs_per_gpu or st.get("jobs_per_gpu", 1)
    gpus = o.gpus if kind == "slurm" else mach.get("gpus", 1)
    if kind == "slurm":
        cpus = st.get("cpus", mach["cpus"])
        omp = o.omp or max(2, cpus // jpg)
    else:
        omp = o.omp or st.get("omp", 3)
    njobs = math.ceil(max(o.events, 1) / epj)
    per_task = math.ceil(njobs / gpus) if kind == "slurm" else njobs
    if kind == "slurm":
        njobs = per_task * gpus
    plan.njobs, plan.events = njobs, njobs * epj
    if plan.events != o.events:
        plan.notes.append(f"{o.events:,} events asked: {njobs} jobs x {epj} events = "
                          f"{plan.events:,}")
    camp = o.campaign or f"{o.production}_{time.strftime('%Y%m%d')}"
    if not NAME_OK.match(camp):
        chk(("error", f"campaign '{camp}': use letters, digits, '.', '_', '-'"))
    image = o.image or P["images"][prod.get("image", "auau")]
    if kind != "native" and "@sha256:" not in image:
        chk(("warn", f"image '{image}' is a moving tag: pin the digest for a campaign "
                     "(README_2stage.md section 1)"))
    if prod.get("image") == "zjet" and kind != "native" and image != P["images"]["zjet"]:
        chk(("warn", "Pb+Pb Z+jet needs an image with TAG_PREFIX support (js-contrib "
                     "2026-10-16 or later); the launchers check it"))

    # -- the driver arguments --------------------------------------------------------------
    config = (CONTAINER_EXAMPLE + "/config") if kind != "native" else "../config"
    args = [a.replace("{CONFIG}", config) for a in prod["args"]]
    tag = prod.get("tag_prefix")

    # -- commands --------------------------------------------------------------------------
    q = shlex.quote
    if kind == "docker2gpu":
        out = o.out or f"/data/{camp}"
        launcher = "utils/launch_2gpu_oo.sh" if prod["prod"] == "oo" else "utils/launch_2gpu.sh"
        env = [f"IMAGE={image}", f"JOBS_PER_GPU={jpg}", f"OMP_THREADS={omp}",
               "MUSIC_CUDA_SYNC=block"] + ([f"TAG_PREFIX={tag}"] if tag else [])
        lopts = []
        if prod.get("lbt"):
            lopts += ["--lbt-cache", LBT_CONTAINER]
        if prod.get("stagger"):
            lopts += ["--stagger", str(prod["stagger"])]
        lopts += ["--min-free", o.min_free or "0"]
        if o.upload:
            lopts += ["--upload", o.upload, "--delete", o.delete]
        plan.commands = [
            f"cd ${{JS_CONTRIB:-{REPO}}}",
            " ".join(env) + " \\\n  " + launcher + " " + " ".join(map(q, lopts)) + " \\\n  "
            + f"{njobs} {epj} {o.first_seed} {q(out)} --campaign {q(camp)}"
            + cont(args)]
        where, watch = f"{out}/gpu0, {out}/gpu1", out
        slots = jpg * gpus
        if njobs % slots:
            plan.notes.append(f"{njobs} jobs on {slots} slots: the last round runs "
                              f"{njobs % slots} jobs; a multiple of {slots} keeps both GPUs "
                              "busy to the end")
        plan.notes.append("run it in tmux; Ctrl-C stops both containers, the same command "
                          "resumes (finished jobs are skipped)")
        if not o.upload and o.delete != "none":
            chk(("warn", "--delete only applies with --upload"))
    elif kind == "native":
        out = o.out or os.path.join("out", camp)
        exports = [f"OMP_NUM_THREADS={omp}", "OMP_WAIT_POLICY=passive"]
        exports += ["KMP_BLOCKTIME=0"] if mach.get("mac") else ["MUSIC_CUDA_SYNC=block"]
        if prod.get("lbt"):
            exports.append("LBT_TABLE_CACHE=$HOME/lbt_cache/lbt_tables_v1.bin")
        if tag:
            exports.append(f"TAG_PREFIX={tag}")
        rj = ["./run_jobs.sh", "-j", str(jpg)] + (["--mps"] if mach.get("mps") else [])
        if prod.get("stagger") and not mach.get("mac"):
            rj += ["--stagger", str(prod["stagger"])]
        rj += ["--campaign", camp, str(njobs), str(epj), str(o.first_seed), out]
        plan.commands = [
            f"conda activate {mach['env']}",
            f"cd ${{JS_CONTRIB:-{REPO}}}/contribs/PyJetscape/example/{PROD_DIRS[prod['prod']]}",
            "export " + " ".join(exports),
            " ".join(map(q, rj)) + cont(args)]
        where = out if os.path.isabs(out) else f"{PROD_DIRS[prod['prod']]}/{out}"
        watch = out if os.path.isabs(out) else os.path.join(
            "contribs/PyJetscape/example", PROD_DIRS[prod["prod"]], out)
        plan.notes.append("run it in tmux; the same command resumes (finished jobs are "
                          "skipped)")
        if o.upload:
            chk(("warn", "upload while running is wired into launch_2gpu.sh only; here run "
                         f"utils/upload_follow.py {out} {o.upload} --follow next to it"))
        if mach.get("mac") and prod["prod"] == "auau":
            plan.notes.append("macOS: run_jobs.sh runs under bash 3.2; --mps is CUDA-only")
    else:                                            # slurm
        cpus = st.get("cpus", mach["cpus"])
        mem = st.get("mem_gb") or slurm_mem(jpg, prod)
        rate_lo = (st.get("rate") or [0, 0])[0]
        hours = per_task * epj / rate_lo * 1.3 if rate_lo else 24
        tlim = min(48, max(1, math.ceil(hours)))
        env = [f"SIF=$PWD/xscape_prod.sif", f"CAMPAIGN={camp}", f"NJOBS={per_task}",
               f"EVENTS={epj}", f"P={jpg}"]
        if prod["prod"] == "oo":
            env.insert(0, "PROD=oo")
        if tag:
            env.append(f"TAG_PREFIX={tag}")
        if o.first_seed:
            env += ["SEED_MODE=ranges", f"SEED_BASE={o.first_seed}"]
        if o.out:
            env.append(f"WORK={o.out}")
        plan.prereq = [f"apptainer pull xscape_prod.sif docker://{image}"]
        plan.commands = [
            " ".join(env) + " \\\n  " + f"EXTRA_ARGS={q(' '.join(args))} \\\n  "
            + f"sbatch --array=0-{gpus - 1} --cpus-per-task={cpus} --mem={mem}G "
              f"--time={tlim}:00:00 utils/slurm_prod_array.sh"]
        where = f"{o.out or '$WORK'}/{camp}/t000 ... t{gpus - 1:03d}"
        watch = f"{o.out or '$WORK'}/{camp}"
        plan.notes.append(f"OMP_NUM_THREADS = cpus-per-task / P = {cpus // jpg} (set by the "
                          f"script); --time {tlim} h = the task's events at the low rate x 1.3")
        plan.notes.append("run from the js-contrib checkout on the cluster; resubmit the same "
                          "command to resume")
        if o.upload:
            chk(("warn", "slurm_prod_array.sh does not upload; run utils/upload_follow.py on a "
                         "login or transfer node"))
        if hours > 48:
            chk(("warn", f"a task needs ~{fmt_h(hours)}: more GPUs (--gpus) or fewer events, or "
                         "a longer --time if the site allows it"))
    plan.commands.append(f"utils/jobmon.py {watch} --tui --njobs {njobs}"
                         "    # watch it (from the js-contrib checkout)")
    plan.estimates.append(("output", where, ""))

    # -- estimates -------------------------------------------------------------------------
    rate = st.get("rate")
    measured = st.get("measured", False) if kind != "slurm" else False
    basis = "measured" if measured else "estimate"
    if rate and kind == "slurm":
        rate = [rate[0] * gpus, rate[1] * gpus]
        basis = f"estimate ({gpus} GPUs at {rng(*st['rate'])} each)"
    disk_gb = plan.events * prod["disk_mb_per_event"] / 1000
    plan.estimates.append(("jobs", f"{njobs} x {epj} events = {plan.events:,} events"
                           + (f" ({gpus} tasks x {per_task} jobs)" if kind == "slurm" else ""), ""))
    plan.estimates.append(("at once", f"{jpg} jobs per GPU x {gpus} GPU(s), "
                           f"{omp} OpenMP threads each", st.get("source", mach.get("source", ""))))
    if rate:
        plan.estimates.append(("rate", f"{rng(*rate)} events/h", basis))
        plan.estimates.append(("duration", rng(plan.events / rate[1], plan.events / rate[0],
                                               fmt_h), basis))
    else:
        plan.estimates.append(("rate", "not known for this machine", "no measurement"))
    plan.estimates.append(("disk", f"{fmt_gb(disk_gb)} ({prod['disk_mb_per_event']:g} MB/event)",
                           prod.get("source", "")))
    if rate:
        gbh = [r * prod["disk_mb_per_event"] / 1000 for r in rate]
        plan.estimates.append(("disk rate", f"{rng(*gbh, lambda x: f'{x:,.1f}' if x < 100 else f'{x:,.0f}')} GB/h", ""))
        if o.upload and gbh[1] > UPLOAD_GB_H:
            chk(("warn", f"writes up to {gbh[1]:,.0f} GB/h, more than the ~{UPLOAD_GB_H} GB/h "
                         "the OSDF upload takes: the backlog grows (a smaller output or "
                         "--delete with --keep-free)"))
    jobs_now = jpg * (1 if kind == "slurm" else gpus)
    mem_have = (st.get("mem_gb") or slurm_mem(jpg, prod)) if kind == "slurm" else mach["mem_gb"]
    if st.get("mem_peak_gb") and kind != "slurm":
        jobs_meas = st["jobs_per_gpu"] * gpus        # the peak scales with the jobs per GPU
        mem_need = st["mem_peak_gb"] * jobs_now / jobs_meas
        mem_basis = (f"measured peak {st['mem_peak_gb']} GB at {jobs_meas} jobs"
                     + ("" if jobs_now == jobs_meas else ", scaled"))
    elif kind == "slurm":
        mem_need = (jobs_now * prod["mem_gb_per_job"] + NODE_BASE_GB) * 1.12
        mem_basis = (f"{prod['mem_gb_per_job']:g} GB per job + {NODE_BASE_GB}, +12 % page "
                     "cache: the plan figure for SLURM")
    else:
        mem_need = jobs_now * prod["vmhwm_gb"] + 6
        mem_basis = f"model: {prod['vmhwm_gb']:g} GB peak per job + 6 GB, no measured peak"
    plan.estimates.append(("memory", f"~{mem_need:.0f} GB of {mem_have} GB"
                           + (" per task" if kind == "slurm" else ""), mem_basis))
    if kind == "slurm" and not st.get("mem_gb"):
        pass                                         # --mem was sized from this estimate
    elif mem_need > 0.95 * mem_have:
        if st.get("mem_peak_gb") or kind == "slurm":
            chk(("error", f"memory: ~{mem_need:.0f} GB needed, {mem_have} GB available: fewer "
                          "jobs per GPU"))
        else:
            chk(("warn", f"memory: the model gives ~{mem_need:.0f} GB of {mem_have} GB (the "
                         "jobs' peaks rarely coincide, more so with --stagger); watch the first "
                         "round, or fewer jobs per GPU"))
    elif mem_need > 0.85 * mem_have:
        chk(("warn", f"memory: ~{mem_need:.0f} GB of {mem_have} GB, little headroom"))

    # -- checks on this machine --------------------------------------------------------------
    here = this_machine()
    if kind != "slurm" and here != o.machine:
        plan.notes.append(f"disk and existing output not checked: this is not the {o.machine} "
                          f"({here or 'unknown machine'})")
    if kind != "slurm" and here == o.machine:
        d = out if os.path.isabs(out) else os.path.join(
            REPO, "contribs/PyJetscape/example", PROD_DIRS[prod["prod"]], out)
        while d and not os.path.exists(d):
            d = os.path.dirname(d)
        if d and d != "/":
            free = shutil.disk_usage(d).free / 1e9
            plan.estimates.append(("free disk", f"{fmt_gb(free)} at {d}", "this machine"))
            if disk_gb > free:
                lvl = "warn" if o.upload and o.delete != "none" else "error"
                chk((lvl, f"disk: {fmt_gb(disk_gb)} to write, {fmt_gb(free)} free at {d}"
                          + (" (upload with --delete frees it as it goes)" if lvl == "warn"
                             else "")))
        marker = os.path.join(out, "gpu0" if kind == "docker2gpu" else "", "run_jobs.campaign")
        if os.path.exists(marker):
            with open(marker) as f:
                old = f.read().strip()
            chk(("warn" if old == camp else "error",
                 f"{out} already holds campaign '{old}'"
                 + (": the command resumes it" if old == camp else
                    ": one campaign per OUTBASE, choose another --out")))
    for x in (prod.get("note"), st.get("note"), mach.get("note")):
        if x:
            plan.notes.append(x)
    for x in (prod.get("source"), st.get("source"), mach.get("source")):
        if x and x not in plan.sources:
            plan.sources.append(x)
    if not any(c[0] in ("warn", "error") for c in plan.checks):
        chk(("ok", "no problems found"))
    plan.summary = (f"{prod['title']} on {mach['title']}: {plan.events:,} events, campaign "
                    f"{camp}")
    return plan


def script(plan: Plan, o: Options) -> str:
    """A launch script with the plan in its header, to read and run (or sbatch from)."""
    lines = ["#!/usr/bin/env bash",
             f"# {plan.summary}",
             f"# Written by utils/prodwizard.py on {time.strftime('%Y-%m-%d %H:%M')} from "
             "utils/prod_presets.json.",
             "#"]
    lines += [f"#   {lab:<10} {val}" + (f"   [{b}]" if b else "") for lab, val, b in plan.estimates]
    if plan.checks:
        lines.append("#")
        lines += [f"#   {lvl.upper():<5} {t}" for lvl, t in plan.checks]
    if plan.notes:
        lines.append("#")
        lines += [f"#   - {n}" for n in plan.notes]
    lines.append("#")
    lines += [f"#   source: {s}" for s in plan.sources]
    if plan.prereq:
        lines += ["#", "# Once, before the campaign:"] + [f"#   {p}" for p in plan.prereq]
    lines += ["", "set -e"]
    body = [c for c in plan.commands if not c.startswith("utils/jobmon.py")]
    if body and body[0].startswith("conda activate"):
        lines += ['eval "$(conda shell.bash hook)"']
    lines += body
    lines += ["", "# watch it:", "# " + plan.commands[-1].split("    #")[0], ""]
    return "\n".join(lines)


def show(plan: Plan):
    print(plan.summary)
    if plan.prereq:
        print("\nonce, before the campaign:")
        for p in plan.prereq:
            print("  " + p)
    print("\ncommands:")
    for c in plan.commands:
        print("  " + c.replace("\n", "\n  "))
    print("\nestimates:")
    for lab, val, basis in plan.estimates:
        print(f"  {lab:<10} {val}" + (f"   [{basis}]" if basis else ""))
    print("\nchecks:")
    for lvl, t in plan.checks:
        print(f"  {lvl.upper():<5} {t}")
    if plan.notes:
        print("\nnotes:")
        for n in plan.notes:
            print(f"  - {n}")
    print("\nsources: " + "; ".join(plan.sources))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    d = Options()
    ap.add_argument("--list", action="store_true", help="the presets and machines")
    ap.add_argument("--preset", dest="production", help="a production preset (--list)")
    ap.add_argument("--machine", help="a machine (--list)")
    ap.add_argument("--events", type=int, default=d.events, help="events in total")
    ap.add_argument("--events-per-job", type=int, default=0, help="default: the preset's")
    ap.add_argument("--campaign", default="", help="the campaign name (file names)")
    ap.add_argument("--out", default="", help="OUTBASE (Docker), OUTDIR (run_jobs.sh) or WORK "
                    "(SLURM)")
    ap.add_argument("--image", default="", help="default: the preset's pinned image")
    ap.add_argument("--first-seed", type=int, default=0, help="0: unique seeds (default)")
    ap.add_argument("--gpus", type=int, default=d.gpus, help="SLURM: array tasks, one GPU each")
    ap.add_argument("--jobs-per-gpu", type=int, default=0, help="default: the machine's")
    ap.add_argument("--omp", type=int, default=0, help="OpenMP threads per job")
    ap.add_argument("--upload", default="", help="Docker: upload to osdf:///fno4hic/NAME")
    ap.add_argument("--delete", default="none", choices=("none", "pair", "all"))
    ap.add_argument("--min-free", default=d.min_free, help="Docker: the disk guard (100G)")
    ap.add_argument("--save", metavar="FILE", help="write a launch script instead of printing")
    a = ap.parse_args(argv)
    P = load_presets()
    if a.list:
        print("presets (--preset):")
        for k, v in P["productions"].items():
            print(f"  {k:<22} {v['title']}")
        print("machines (--machine):")
        for k, v in P["machines"].items():
            print(f"  {k:<22} {v['title']}")
        return 0
    if not a.production and not a.machine:
        try:
            from prodwizard_tui import run_wizard
        except ImportError as e:
            sys.exit(f"prodwizard: the wizard needs textual ({e}); or give --preset and "
                     "--machine (--list)")
        return run_wizard(P)
    for name, key in (("--preset", a.production), ("--machine", a.machine)):
        group = P["productions"] if name == "--preset" else P["machines"]
        if key not in group:
            sys.exit(f"prodwizard: {name} {key!r}: one of {', '.join(group)}")
    o = Options(**{k: v for k, v in vars(a).items() if k in Options.__dataclass_fields__})
    plan = build(o, P)
    if a.save:
        with open(a.save, "w") as f:
            f.write(script(plan, o))
        os.chmod(a.save, 0o755)
        print(f"wrote {a.save} ({'ok' if plan.ok else 'has ERRORS: read its header'})")
    else:
        show(plan)
    return 0 if plan.ok else 1


if __name__ == "__main__":
    sys.exit(main())
