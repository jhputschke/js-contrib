#!/usr/bin/env python
"""Produce the two files notebooks/jet_wake.ipynb reads.

    python make_wake_data.py                      # both legs, into <build>/out_wake/
    python make_wake_data.py --device cpu         # slower, bitwise reproducible
    python make_wake_data.py --legs ideal         # just one
    python make_wake_data.py --dry-run            # print what it would do
    python make_wake_data.py --medium realistic   # the medium normalized to dN_ch/deta ~ 650
    python make_wake_data.py --medium tune_0_10 --legs visc   # 0-10%, tuned to MUSIC

Two media:

    fno4d      config/fasthydro_wake.yaml            FNO4d's reference medium (the default);
                                                     ~2.5x too dilute for central Au+Au
    realistic  config/fasthydro_wake_realistic.yaml  the medium hadron_wake.ipynb runs on,
                                                     -> <build>/out_wake_realistic/
    tune_0_10  config/AuAu_FastHydro_tune_0_10_wake.yaml  0-10%, tuned to 3D MC-Glauber +
                                                     MUSIC (tau0 0.5, zeta/s 0.12); viscous,
                                                     so --legs visc -> <build>/out_wake_tune_0_10/

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

#: --medium -> (config, default output directory, user XML)
MEDIA = {"fno4d": ("fasthydro_wake.yaml", "out_wake", "jetscape_user_fasthydro_wake.xml"),
         "realistic": ("fasthydro_wake_realistic.yaml", "out_wake_realistic",
                       "jetscape_user_fasthydro_wake.xml"),
         # tau0 = 0.5, so it needs its own XML (<taus>, <tStart>); run it with --legs visc
         "tune_0_10": ("AuAu_FastHydro_tune_0_10_wake.yaml", "out_wake_tune_0_10",
                       "AuAu_FastHydro_tune_0_10_wake.xml")}

#: eos.kind -> the table ensure_eos must find
EOS_TABLES = {"hotqcd_smash": ("hrg_hotqcd_eos_SMASH_binary.dat", "SMASH_binary"),
              "hotqcd": ("hrg_hotqcd_eos_binary.dat", "binary")}

#: The config asks for the hotQCD/SMASH table at ./eos/hotQCD relative to the working
#: directory. Look for a copy already on the machine before fetching one; X-SCAPE ships its own
#: under EOS/hotQCD, and a previous run of this script leaves one in place.
EOS_SUBDIR = os.path.join("eos", "hotQCD")
EOS_FILE = "hrg_hotqcd_eos_SMASH_binary.dat"
EOS_FILETYPE = "SMASH_binary"
RECORD_BYTES = 32           # 4 x float64 per row: e, p, s, T


def _looks_like_a_table(path):
    """A truncated download loads as a short table and quietly changes the physics."""
    try:
        n = os.path.getsize(path)
    except OSError:
        return False
    return n > 0 and n % RECORD_BYTES == 0


def ensure_eos(build, *, eos_dir=None, allow_download=True, verbose=True,
               eos_file=EOS_FILE, filetype=EOS_FILETYPE):
    """Make sure `<build>/eos/hotQCD/<table>` exists. -> its directory, or None.

    `eos_file`/`filetype` pick the table: EOS 91 (`hotqcd_smash`) by default,
    ``("hrg_hotqcd_eos_binary.dat", "binary")`` for EOS 9 (`hotqcd`).

    Order: an explicit --eos-dir, then where this script would have put it, then X-SCAPE's own
    EOS/hotQCD. Failing all of those, download it with fast_data's own fetcher -- plain urllib,
    writing to a .part file and renaming only once the size validates as a whole number of
    32-byte records, so an interrupted fetch cannot leave a half table that silently loads.
    """
    dest = os.path.join(build, EOS_SUBDIR)
    target = os.path.join(dest, eos_file)

    if os.path.exists(target) and _looks_like_a_table(target):
        if verbose:
            print(f"  EoS table: {target}")
        return dest
    if os.path.exists(target):
        print(f"  EoS table: {target} is not a whole number of {RECORD_BYTES} B records "
              f"(truncated); refetching")
        os.remove(target)

    # a copy elsewhere on the machine
    for d in ([eos_dir] if eos_dir else []) + [os.path.join(build, "EOS", "hotQCD")]:
        cand = os.path.join(d, eos_file) if os.path.isdir(d) else d
        if os.path.exists(cand) and _looks_like_a_table(cand):
            os.makedirs(dest, exist_ok=True)
            try:
                os.symlink(os.path.abspath(cand), target)
            except OSError:
                shutil.copy2(cand, target)
            if verbose:
                print(f"  EoS table: {cand}\n             linked -> {target}")
            return dest

    if not allow_download:
        print(f"  MISS  EoS table   {eos_file} not found, and --no-download was given.")
        print(f"        Looked in: {dest}, {os.path.join(build, 'EOS', 'hotQCD')}"
              + (f", {eos_dir}" if eos_dir else ""))
        print( "        Fetch it with:  python -c \"from fast_data.eos import download_hotqcd;"
              f" download_hotqcd('{dest}')\"")
        return None

    print(f"  EoS table not found locally; downloading {eos_file} (~3.2 MB) -> {dest}")
    sys.path.insert(0, os.path.join(CONTRIB, "python"))
    from fast_data.eos import download_hotqcd
    try:
        out = download_hotqcd(dest, filetype=filetype)
    except Exception as exc:
        print(f"  MISS  download failed: {exc}")
        print( "        If this machine has no network, copy the table from any X-SCAPE build")
        print(f"        ({os.path.join('EOS', 'hotQCD', eos_file)}) into {dest},")
        print( "        or pass --eos-dir <dir> pointing at one.")
        return None
    print(f"         got {os.path.getsize(out) / 1e6:.2f} MB, "
          f"{os.path.getsize(out) // RECORD_BYTES} table rows")
    return dest


def _eos_kind(cfg_path):
    """eos.kind of a YAML, without importing fast_data (the preflight runs before sys.path)."""
    import re
    with open(cfg_path) as f:
        text = f.read()
    block = text[text.index("\neos:"):] if "\neos:" in text else ""
    m = re.search(r"^\s+kind:\s*(\w+)", block, re.M)
    return m.group(1) if m else None


def replay_leg(cfg_path, drop_npz, out, transport_mode, overrides, build, force=False):
    """One leg from a fixed droplet set: background + jet through this solver, into `out`."""
    sys.path.insert(0, os.path.join(CONTRIB, "python"))
    from fasthydro.config import load_config
    from fasthydro.droplets_io import load_droplets_npz, showers_from_meta
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
        # --force has to reach here too: the live leg gets it via run_two_stage's
        # --overwrite, and without this the second leg stops on the file the FIRST run of
        # this very script left behind.
        replay_pair(out, cfg, ic, da, params, overwrite=force or None,
                    shower=showers_from_meta(meta, 0),
                    meta={"provenance_droplets": os.path.basename(drop_npz)})
    finally:
        os.chdir(cwd)
    return 0


def _is_locked(path):
    """True if `path` exists and some process holds it open, so HDF5 cannot truncate it.

    Probed by opening the file for writing with an exclusive `flock`, which is what the HDF5
    library itself does -- not by parsing lsof, which is only used afterwards to name the
    culprit.  Absent or unreadable means "not locked": the point is to catch the common case
    cleanly, never to block a run over a failed probe.
    """
    import fcntl

    if not os.path.exists(path):
        return False
    try:
        fd = os.open(path, os.O_RDWR)                # NOT O_TRUNC -- that is the whole point
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    except OSError:
        return True
    finally:
        os.close(fd)


def _holders(paths):
    """A short description of what has these files open, or '' if lsof cannot say."""
    try:
        out = subprocess.run(["lsof", "-F", "cpn", *paths], capture_output=True, text=True,
                             timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return ""
    seen, pid = [], ""
    for line in out.splitlines():
        if line[:1] == "p":
            pid = line[1:]
        elif line[:1] == "c" and (line[1:], pid) not in seen:
            seen.append((line[1:], pid))
    return ", ".join(f"{c} (pid {p})" for c, p in seen)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", default=".",
                    help="X-SCAPE build tree to run in (default: the working directory)")
    ap.add_argument("--medium", choices=sorted(MEDIA), default="fno4d",
                    help="fno4d: fasthydro_wake.yaml, the reference medium (default); realistic: normalized to "
                         "the measured dN_ch/deta, the one hadron_wake.ipynb uses; tune_0_10: "
                         "0-10%%, tuned to MUSIC + 3D Glauber (viscous: use --legs visc)")
    ap.add_argument("--out", default=None,
                    help="output directory, relative to --build (default: out_wake, or "
                         "out_wake_realistic for --medium realistic)")
    ap.add_argument("--legs", nargs="+", choices=sorted(LEGS), default=sorted(LEGS))
    ap.add_argument("--events", type=int, default=1)
    ap.add_argument("--device", default=None, help="cpu | cuda | mps (default: from the YAML)")
    ap.add_argument("--dtype", default=None, help="float32 | float64")
    ap.add_argument("--main-xml", default=None,
                    help="default: <build>/../config/jetscape_main.xml")
    ap.add_argument("--set", action="append", default=[], metavar="k.p=v",
                    help="extra dotted override, passed through to every leg")
    ap.add_argument("--eos-dir", default=None,
                    help="directory holding the hotQCD/SMASH table, if you have one; "
                         "otherwise it is taken from the build tree's EOS/hotQCD or downloaded")
    ap.add_argument("--no-download", action="store_true",
                    help="never fetch the EoS table from the network")
    ap.add_argument("--live", action="store_true",
                    help="run every leg end to end, letting each shower respond to its own "
                         "medium. Default: run the first leg live and REPLAY its droplets "
                         "through the others, so the comparison isolates the hydro.")
    ap.add_argument("--force", action="store_true", help="regenerate files that already exist")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would run; skips the EoS fetch")
    a = ap.parse_args(argv)

    build = os.path.abspath(a.build)
    cfg_name, out_default, uxml_name = MEDIA[a.medium]
    outdir = os.path.join(build, a.out or out_default)
    cfg = os.path.join(CONTRIB, "config", cfg_name)
    uxml = os.path.join(CONTRIB, "config", uxml_name)
    mxml = a.main_xml or os.path.join(build, "..", "config", "jetscape_main.xml")
    driver = os.path.join(HERE, "run_two_stage.py")

    print("=== preflight ===")
    ok = True
    for what, p in (("build tree", build), ("config", cfg), ("user XML", uxml),
                    ("main XML", mxml), ("driver", driver)):
        good = os.path.exists(p)
        ok &= good
        print(f"  {'ok  ' if good else 'MISS'}  {what:11s} {p}")
    if not ok:
        return 1
    eos_kind = _eos_kind(cfg)
    if not a.dry_run and eos_kind in EOS_TABLES:
        eos_file, filetype = EOS_TABLES[eos_kind]
        if ensure_eos(build, eos_dir=a.eos_dir, allow_download=not a.no_download,
                      eos_file=eos_file, filetype=filetype) is None:
            return 1

    os.makedirs(outdir, exist_ok=True)

    held = [p for p in (os.path.join(outdir, f"wake_{leg}.h5") for leg in a.legs)
            if _is_locked(p)]
    if held and not a.dry_run:
        print("\n  BLOCKED: another process has these files open --\n" +
              "".join(f"    {p}\n" for p in held) +
              "  HDF5 refuses to truncate a file that is open elsewhere, even read-only, and\n"
              "  the failure arrives as a 30-line traceback ending in\n"
              "  `BlockingIOError: [Errno 35] ... unable to lock file`.\n"
              "  It is almost always a notebook: a Jupyter/VS Code kernel that opened one of\n"
              "  these keeps the handle for the life of the kernel, and closing the tab is not\n"
              "  enough. Restart the kernel (or `PairBrowser.close()` in it) and re-run.\n"
              "  Worse, HDF5 truncates BEFORE it takes the lock, so the old file is already\n"
              "  destroyed by the time it fails -- which is why this checks first.\n"
              f"  Holding it: {_holders(held) or 'unknown (lsof unavailable)'}")
        return 1

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
            rc = replay_leg(cfg, drop_npz, out, LEGS[leg], overrides, build,
                            force=a.force)
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
        if a.force:
            cmd.append("--overwrite")
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
