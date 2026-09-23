#!/usr/bin/env python
"""Produce the files notebooks/hadron_wake.ipynb reads: both legs, particlized, viscous.

    python make_hadron_wake_data.py                        # into <build>/out_hadron_wake/
    python make_hadron_wake_data.py --events 8 --oversample 2000
    python make_hadron_wake_data.py --hard PythiaGun       # dijets instead of one parton
    python make_hadron_wake_data.py --dry-run

Two runs of central Au+Au 200 GeV, Israel-Stewart hydro (eta/s = 0.08), on the SAME events:

    jet run   IC -> jet (Matter+LBT) -> liquefier -> FastHydro_jet -> iSS   hadrons_jet.npz
    bg  run   IC -> FastHydro_bg -> iSS                                      hadrons_bg.npz

plus jet_events.json / bg_events.json: per event, the IC's sha256, the surface closure, what
the jet deposited, and the shower initiators (the jet axes).  An event's IC depends only on
(run.seed, event index), so event k of one run is event k of the other; the controls at the
end check the hashes.

**The hard process.**  The default is PGun: ONE 60 GeV parton, from the fireball centre, at
phi = 0 and y = 0 in every event (PGun.cc zeroes both the vertex and the angle).  Every
event's wake then sits in the same place and the events stack directly -- the cleanest setting
for identifying a wake.  PythiaGun gives real dijets from Ncoll-sampled vertices; the notebook
then aligns each event on its leading initiator, and the away-side jet's own response overlaps
the near-side jet's diffusion wake.

**Statistics.**  The difference jet - bg sits on the bulk's Poisson noise, which falls as
1/sqrt(oversamples x events) on both legs.  Oversamples are cheap (the surface finder
dominates an event's cost), so buy statistics with --oversample first.  Both legs get the same
number by default: each event has its own background, so there is nothing to reuse.

**Viscous, with an ideal Cooper-Frye.**  The fluid runs Israel-Stewart, but FastHydro stores
pi^{mu nu} and Pi as zero (fast_data's frame carries only e and v), so iSS samples without
delta-f.  The evolution is viscous; the particlization is not.

Run it from the X-SCAPE build tree, or pass --build.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CONTRIB = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(CONTRIB, "python"))

from make_wake_data import ensure_eos  # noqa: E402  (same EoS lookup/fetch)

TRANSPORT = "israel_stewart"
LEGS = ("jet", "bg")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", default=".",
                    help="X-SCAPE build tree to run in (default: the working directory)")
    ap.add_argument("--out", default="out_hadron_wake", help="output directory, relative to --build")
    ap.add_argument("--events", type=int, default=4)
    ap.add_argument("--oversample", type=int, default=1000, help="iSS samples per event, both legs")
    ap.add_argument("--oversample-bg", type=int, default=None,
                    help="override for the background leg only")
    ap.add_argument("--hard", default="PGun", choices=("PGun", "PythiaGun"))
    ap.add_argument("--device", default=None, help="cpu | cuda | mps (default: from the YAML)")
    ap.add_argument("--main-xml", default=None,
                    help="default: <build>/../config/jetscape_main.xml")
    ap.add_argument("--set", action="append", default=[], metavar="k.p=v",
                    help="extra dotted override, passed to both legs")
    ap.add_argument("--eos-dir", default=None)
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--keep-ascii", action="store_true",
                    help="keep the framework's text hadron files next to the .npz")
    ap.add_argument("--force", action="store_true", help="regenerate legs that already exist")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    build = os.path.abspath(a.build)
    outdir = os.path.join(build, a.out)
    cfg = os.path.join(CONTRIB, "config", "fasthydro_particlize.yaml")
    uxml = os.path.join(CONTRIB, "config", "jetscape_user_fasthydro_particlize.xml")
    mxml = os.path.abspath(a.main_xml or os.path.join(build, "..", "config", "jetscape_main.xml"))
    driver = os.path.join(HERE, "run_particlize.py")

    print("=== preflight ===")
    ok = True
    for what, p in (("build tree", build), ("config", cfg), ("user XML", uxml),
                    ("main XML", mxml), ("driver", driver)):
        good = os.path.exists(p)
        ok &= good
        print(f"  {'ok  ' if good else 'MISS'}  {what:11s} {p}")
    if not ok:
        return 1
    if not a.dry_run and ensure_eos(build, eos_dir=a.eos_dir,
                                    allow_download=not a.no_download) is None:
        return 1
    os.makedirs(outdir, exist_ok=True)

    env = dict(os.environ)
    # The surface finder is OpenMP; idle OpenMP threads otherwise spin (see README)
    env.setdefault("OMP_NUM_THREADS", str(min(8, os.cpu_count() or 1)))
    env.setdefault("OMP_WAIT_POLICY", "passive")
    print(f"  OMP_NUM_THREADS={env['OMP_NUM_THREADS']} OMP_WAIT_POLICY={env['OMP_WAIT_POLICY']}")

    overrides = [f"transport.mode={TRANSPORT}"]
    if a.hard == "PGun":
        # PGun puts the parton at the origin whatever the sampler says; say so, which also
        # keeps build_two_stage from warning that the Ncoll vertex is being ignored
        overrides.append("fasthydro.hard_vertex.mode=centre")
    overrides += list(a.set)
    if a.device:
        overrides.append(f"run.device={a.device}")
        if a.device == "mps":
            overrides.append("run.dtype=float32")

    print(f"\n=== {a.events} event(s), hard = {a.hard}, transport = {TRANSPORT} -> {outdir} ===")
    t0 = time.time()
    for leg in LEGS:
        npz = os.path.join(outdir, f"hadrons_{leg}.npz")
        if os.path.exists(npz) and not a.force:
            print(f"\n  {leg}: {npz} exists, skipping (--force to regenerate)")
            continue
        n_os = a.oversample_bg if (leg == "bg" and a.oversample_bg) else a.oversample
        cmd = [sys.executable, driver, "--leg", leg, "--config", cfg, "--user-xml", uxml,
               "--main-xml", mxml, "--events", str(a.events), "--oversample", str(n_os),
               "--out-dir", outdir, "--quiet"]
        if leg == "jet":
            cmd += ["--hard", a.hard]
        for o in overrides:
            cmd += ["--set", o]
        print(f"\n  --- {leg} leg, {n_os} oversamples/event ---")
        if a.dry_run:
            print("     " + " ".join(cmd))
            continue
        log = os.path.join(outdir, f"{leg}.log")
        t1 = time.time()
        with open(log, "w") as fh:
            r = subprocess.run(cmd, cwd=build, env=env, stdout=fh, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            print(f"     FAILED (exit {r.returncode}); last lines of {log}:")
            with open(log) as fh:
                for line in fh.readlines()[-15:]:
                    print("       " + line.rstrip())
            return r.returncode
        with open(log) as fh:
            for line in fh:
                if any(k in line for k in ("[DropletBridge]", "droplets fired", "WARNING",
                                           "not closed", "=== ")):
                    print("     " + line.rstrip())
        txt = os.path.join(outdir, f"{leg}_final_state_hadrons.dat")
        from fasthydro.hadrons import ascii_to_npz
        n_ev = ascii_to_npz(txt, npz)
        mb_txt, mb_npz = os.path.getsize(txt) / 1e6, os.path.getsize(npz) / 1e6
        if not a.keep_ascii:
            os.remove(txt)
        print(f"     {time.time() - t1:.0f} s; {n_ev} event(s) -> {npz} "
              f"({mb_npz:.0f} MB; text was {mb_txt:.0f} MB"
              + (", kept)" if a.keep_ascii else ", removed)"))

    if a.dry_run:
        return 0
    print(f"\n=== done in {time.time() - t0:.0f} s ===")
    return check(outdir)


def check(outdir):
    """The controls the notebook's section 1 re-runs: same ICs, closed surfaces, a deposit."""
    import numpy as np

    from fasthydro.hadrons import load

    meta = {}
    for leg in LEGS:
        with open(os.path.join(outdir, f"{leg}_events.json")) as f:
            meta[leg] = json.load(f)
    H = {leg: load(os.path.join(outdir, f"hadrons_{leg}.npz"), meta[leg]["n_oversample"])
         for leg in LEGS}
    n = min(len(meta["jet"]["events"]), len(meta["bg"]["events"]), H["jet"].n_events,
            H["bg"].n_events)
    print("=== controls ===")
    bad = 0
    for k in range(n):
        ej, eb = meta["jet"]["events"][k], meta["bg"]["events"][k]
        same = ej["ic_sha256"] == eb["ic_sha256"]
        closed = ej["closure"]["closed"] and eb["closure"]["closed"]
        E = (ej.get("deposited_P") or [0.0])[0]
        print(f"  event {k}: IC identical {same}; surfaces closed {closed}; "
              f"{ej.get('n_droplets', 0)} droplets, deposited E = {E:.1f} GeV")
        bad += (not same) or (not closed)

    # the headline number: energy in |y| < 1 and momentum along the jet, summed over events
    mid = lambda h: np.abs(h.y) < 1.0                              # noqa: E731
    ev = list(range(n))
    dE = [H[l].total(mid(H[l]), H[l].E, events=ev) for l in LEGS]
    dPx = [H[l].total(None, H[l].px, events=ev) for l in LEGS]     # PGun jets run along +x
    for label, (j, b) in (("Delta E, |y|<1", dE), ("Delta p_x, all", dPx)):
        m, s = j[0] - b[0], float(np.hypot(j[1], b[1]))
        print(f"  {label:16s} = {m:+8.2f} +- {s:6.2f} GeV per event  ({abs(m) / s:.1f} sigma)")
    Ed = np.mean([(e.get("deposited_P") or [0, 0])[0] for e in meta["jet"]["events"][:n]])
    Px = np.mean([(e.get("deposited_P") or [0, 0])[1] for e in meta["jet"]["events"][:n]])
    print(f"  deposited        : E = {Ed:.2f} GeV, p_x = {Px:.2f} GeV per event")
    if bad:
        print("  ^ some events cannot be paired or did not close; the notebook drops them")
    print(f"\nNow run:  jupyter lab {os.path.join(CONTRIB, 'notebooks', 'hadron_wake.ipynb')}")
    print(f"          (set HADRON_WAKE_OUT={outdir} if you run it from elsewhere)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
