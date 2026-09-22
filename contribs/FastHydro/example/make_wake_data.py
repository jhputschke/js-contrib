#!/usr/bin/env python
"""Produce the two files notebooks/jet_wake.ipynb reads.

    python make_wake_data.py                      # both legs, into <build>/out_wake/
    python make_wake_data.py --device cpu         # slower, bitwise reproducible
    python make_wake_data.py --legs ideal         # just one
    python make_wake_data.py --dry-run            # print what it would do

Two runs of central Au+Au 200 GeV differing in one switch, `transport.mode`:

    wake_ideal.h5    ideal hydro
    wake_visc.h5     Israel-Stewart, eta/s = 0.08

Each file already holds its own jet/no-jet pair (`arr` and `arr_bg`) on one initial condition,
so the notebook needs two files where FNO4d's viscous_vs_ideal.ipynb needs four.

Both runs share `run.seed`, so they start from a bit-identical initial condition.

**The droplets, though, do not come out the same on their own.** The shower responds to the
medium it traverses, so running Matter+LBT twice -- once against an ideal background, once
against a viscous one -- gives two different droplet sets. That is real physics, not a defect,
but it means a live `visc - ideal` mixes the hydrodynamic response to the wake with a
different jet having been produced.

So by default the *first* leg runs the full chain and dumps its droplets, and every other leg
**replays that same droplet set** through its own solver. `visc - ideal` is then the viscosity
alone. Pass `--live` to run every leg end to end instead, which is the more complete physical
statement and the less controlled comparison. The script reports which you got.

Run it from the X-SCAPE build tree, or pass --build: the framework resolves several paths
relative to the working directory.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CONTRIB = os.path.dirname(HERE)

LEGS = {"ideal": "ideal", "visc": "israel_stewart"}

#: where the hotQCD table might already be, in preference order. The config asks for
#: ./eos/hotQCD relative to the working directory; rather than download it again, link
#: whichever copy the machine already has.
EOS_CANDIDATES = (
    os.path.join("EOS", "hotQCD"),                                    # the X-SCAPE build tree
    os.path.join(os.path.expanduser("~"), "FNO4d", "workflow_fastdata", "eos", "hotQCD"),
)
EOS_FILE = "hrg_hotqcd_eos_SMASH_binary.dat"


def find_eos(verbose=True):
    """-> directory holding the SMASH table, or None."""
    for d in EOS_CANDIDATES:
        if os.path.exists(os.path.join(d, EOS_FILE)):
            if verbose:
                print(f"  EoS table: {os.path.join(d, EOS_FILE)}")
            return os.path.abspath(d)
    return None


def link_eos(workdir, src, verbose=True):
    """Put the table where the config expects it (./eos/hotQCD), without copying 1.7 MB."""
    dst_dir = os.path.join(workdir, "eos", "hotQCD")
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, EOS_FILE)
    if not os.path.exists(dst):
        try:
            os.symlink(os.path.join(src, EOS_FILE), dst)
        except OSError:
            shutil.copy2(os.path.join(src, EOS_FILE), dst)
        if verbose:
            print(f"  linked  -> {dst}")
    return dst_dir


def replay_leg(cfg_path, drop_npz, out, transport_mode, overrides, build):
    """One leg from a fixed droplet set: background + jet through this solver, into `out`."""
    sys.path.insert(0, os.path.join(CONTRIB, "python"))
    from fasthydro.config import load_config
    from fasthydro.droplets_io import load_droplets_npz
    from fasthydro.replay import replay_pair

    cwd = os.getcwd()
    os.chdir(build)                       # the EoS path in the YAML is relative
    try:
        cfg = load_config(cfg_path, list(overrides) + [f"transport.mode={transport_mode}"])
        per_event, params, meta = load_droplets_npz(drop_npz)
        ic = meta.get("ic")
        if ic is None or ic.size == 0:
            print(f"     FAILED: {drop_npz} carries no initial condition")
            return 1
        da = per_event[0]
        print(f"     {len(da)} droplets, {da.data[:, 4].sum():.2f} GeV, "
              f"tau_delay = {params.tau_delay} fm/c")
        replay_pair(out, cfg, ic, da, params,
                    meta={"provenance_droplets": os.path.basename(drop_npz)})
    finally:
        os.chdir(cwd)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", default=".",
                    help="X-SCAPE build tree to run in (default: the working directory)")
    ap.add_argument("--out", default="out_wake", help="output directory, relative to --build")
    ap.add_argument("--legs", nargs="+", choices=sorted(LEGS), default=sorted(LEGS))
    ap.add_argument("--events", type=int, default=1)
    ap.add_argument("--device", default=None, help="cpu | cuda | mps (default: from the YAML)")
    ap.add_argument("--dtype", default=None, help="float32 | float64")
    ap.add_argument("--main-xml", default=None,
                    help="default: <build>/../config/jetscape_main.xml")
    ap.add_argument("--set", action="append", default=[], metavar="k.p=v",
                    help="extra dotted override, passed through to every leg")
    ap.add_argument("--live", action="store_true",
                    help="run every leg end to end, letting each shower respond to its own "
                         "medium. Default: run the first leg live and REPLAY its droplets "
                         "through the others, so the comparison isolates the hydro.")
    ap.add_argument("--force", action="store_true", help="regenerate files that already exist")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    build = os.path.abspath(a.build)
    outdir = os.path.join(build, a.out)
    cfg = os.path.join(CONTRIB, "config", "fasthydro_wake.yaml")
    uxml = os.path.join(CONTRIB, "config", "jetscape_user_fasthydro_wake.xml")
    mxml = a.main_xml or os.path.join(build, "..", "config", "jetscape_main.xml")
    driver = os.path.join(HERE, "run_two_stage.py")

    print("=== preflight ===")
    ok = True
    for what, p in (("build tree", build), ("config", cfg), ("user XML", uxml),
                    ("main XML", mxml), ("driver", driver)):
        good = os.path.exists(p)
        ok &= good
        print(f"  {'ok  ' if good else 'MISS'}  {what:11s} {p}")
    eos = find_eos()
    if eos is None:
        ok = False
        print(f"  MISS  EoS table   {EOS_FILE} not found in:")
        for d in EOS_CANDIDATES:
            print(f"                      {d}")
        print("        fetch it with:  bash "
              f"{os.path.join(CONTRIB, 'python', 'fast_data', 'download_hotQCD.sh')} "
              "SMASH_binary <dir>")
    if not ok:
        return 1

    os.makedirs(outdir, exist_ok=True)
    if not a.dry_run:
        link_eos(build, eos)

    overrides = list(a.set)
    if a.device:
        overrides.append(f"run.device={a.device}")
        if a.dtype is None and a.device == "mps":
            overrides.append("run.dtype=float32")
    if a.dtype:
        overrides.append(f"run.dtype={a.dtype}")

    mode = "live" if a.live else "replayed"
    print(f"\n=== generating {len(a.legs)} leg(s) into {outdir}  [{mode} droplets] ===")
    t0 = time.time()
    legs = sorted(a.legs, key=lambda k: k != "ideal")      # the live leg first
    drop_npz = os.path.join(outdir, "wake_droplets.npz")

    for n, leg in enumerate(legs):
        out = os.path.join(outdir, f"wake_{leg}.h5")
        if os.path.exists(out) and not a.force:
            print(f"\n  {leg}: {out} exists, skipping (--force to regenerate)")
            continue

        # every leg after the first is a replay, unless --live
        if n > 0 and not a.live:
            print(f"\n  --- {leg} (transport.mode={LEGS[leg]}, replaying {os.path.basename(drop_npz)}) ---")
            if a.dry_run:
                print(f"     replay_pair -> {out}")
                continue
            if not os.path.exists(drop_npz):
                print(f"     FAILED: {drop_npz} not written by the first leg")
                return 1
            t1 = time.time()
            rc = replay_leg(cfg, drop_npz, out, LEGS[leg], overrides, build)
            if rc:
                return rc
            print(f"     {time.time() - t1:.0f} s -> {out}")
            continue

        cmd = [sys.executable, driver,
               "--config", cfg, "--user-xml", uxml, "--main-xml", os.path.abspath(mxml),
               "--events", str(a.events), "--out", out,
               "--set", f"transport.mode={LEGS[leg]}"]
        if not a.live:
            cmd += ["--dump-droplets", drop_npz]
        for o in overrides:
            cmd += ["--set", o]
        print(f"\n  --- {leg} (transport.mode={LEGS[leg]}) ---")
        if a.dry_run:
            print("     " + " ".join(cmd))
            continue
        log = os.path.join(outdir, f"{leg}.log")
        t1 = time.time()
        with open(log, "w") as fh:
            r = subprocess.run(cmd, cwd=build, stdout=fh, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            print(f"     FAILED (exit {r.returncode}); last lines of {log}:")
            with open(log) as fh:
                for line in fh.readlines()[-15:]:
                    print("       " + line.rstrip())
            return r.returncode
        # echo the lines worth seeing rather than the whole framework banner
        with open(log) as fh:
            for line in fh:
                if any(k in line for k in ("[FastGlauberInitialState] event", "[DropletBridge]",
                                           "droplets fired", "wrote ", "WARNING")):
                    print("     " + line.rstrip())
        print(f"     {time.time() - t1:.0f} s -> {out}")

    if a.dry_run:
        return 0

    print(f"\n=== done in {time.time() - t0:.0f} s ===")
    return check(outdir, a.legs)


def check(outdir, legs):
    """The controls the notebook's section 2 will re-run: same IC, same shower, real wake."""
    import h5py
    import numpy as np

    paths = {k: os.path.join(outdir, f"wake_{k}.h5") for k in legs}
    if not all(os.path.exists(p) for p in paths.values()):
        return 0
    print("=== controls ===")
    F = {k: h5py.File(p, "r") for k, p in paths.items()}
    try:
        for k, f in F.items():
            ic_same = np.array_equal(f["arr"][0, :, :, :, :, 0], f["arr_bg"][0, :, :, :, :, 0])
            dE = float(np.abs(f["arr"][0, 0] - f["arr_bg"][0, 0]).max())
            nd = int(np.diff(f["source/offsets"][:])[0])
            Ed = float(f["source/droplets"][:nd, 4].sum()) if nd else 0.0
            print(f"  {k:6s} IC identical across legs: {ic_same};  {nd} droplets, "
                  f"{Ed:.2f} GeV;  max|e_jet - e_bg| = {dE:.4g} GeV/fm^3;  "
                  f"tau_fo bg/jet = {float(f['tau_freezeout_bg'][0]):.2f}/"
                  f"{float(f['tau_freezeout'][0]):.2f} fm/c")
            if not ic_same or dE <= 0:
                print("        ^ the pair is not usable")
        if len(F) > 1:
            a, b = (F[k] for k in sorted(F))
            same_drops = np.array_equal(a["source/droplets"][:], b["source/droplets"][:])
            differ = not np.array_equal(a["arr"][0], b["arr"][0])
            print(f"  both   same droplet table across legs: {same_drops}  "
                  f"(needed for visc - ideal to be the viscosity)")
            print(f"  both   the two transport settings differ: {differ}")
            if not differ:
                print("        ^ Israel-Stewart is not engaging; the comparison is empty")
    finally:
        for f in F.values():
            f.close()
    print(f"\nNow run:  jupyter lab {os.path.join(CONTRIB, 'notebooks', 'jet_wake.ipynb')}")
    print(f"          (set WAKE_OUT={outdir} if you run it from elsewhere)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
