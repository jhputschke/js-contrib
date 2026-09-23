#!/usr/bin/env python
"""Produce the files notebooks/bulk_vs_music.ipynb reads: 0-10% Au+Au, MUSIC vs FastHydro.

    python make_bulk_comparison_data.py                          # into <build>/out_bulk_vs_music/
    python make_bulk_comparison_data.py --events 20 --oversample 200
    python make_bulk_comparison_data.py --models music           # one model only
    python make_bulk_comparison_data.py --music-variants calibrated,matched
    python make_bulk_comparison_data.py --dry-run

The question: how realistic is the initial state that config/fasthydro_wake_realistic.yaml tunes
(fast_data's tilted MC-Glauber, normalized by target_T = 0.39 at b = 0)?  On a SHARED initial
condition FastHydro and MUSIC agree to 1-3% (config/FVvsMUSIC), so running each with its own
initial state isolates the initial state:

    music_calibrated   3D MC-Glauber strings -> MUSIC -> iSS      config/AuAu_MCGlauber_MUSIC_0_10.xml
                       as calibrated: T-dependent eta/s, bulk viscosity, delta-f.  The
                       realistic reference.
    music_matched      the same 3D MC-Glauber, with MUSIC's transport and Cooper-Frye set to
                       FastHydro's: constant eta/s, no bulk, no second-order terms, tau_pi
                       factor from the YAML, no delta-f.  Differs from FastHydro only in the
                       initial state (and in the pre-hydro dynamics that comes with it).
                       Optional: --music-variants calibrated,matched.
    fasthydro          fast_data MC-Glauber -> FastHydro (Israel-Stewart) -> iSS
                       config/fasthydro_particlize.yaml, whose initial state is
                       fasthydro_wake_realistic.yaml's (checked below).  b is sampled on
                       [0, --b-max] with P(b) ~ b, the 0-10% class; the normalization K stays
                       the one calibrated at b = 0, which is exactly the tuning under test.

Every model writes hadrons_<model>.npz (fasthydro.hadrons format) and <model>_events.json.  All
three particlize at T = 0.15 GeV on EOS 9 (UrQMD list), and iSS decays the resonances.

**Centrality.** 3dMCGlauber selects 0-10% on its string count, which is a multiplicity proxy
(EventGenerator.cpp:285-340).  fast_data has only b.  --b-max 4.7 fm is the geometric 0-10%
for sigma_NN = 42 mb (pi b^2 = 0.1 x ~690 fm^2).  The events files record what each model
selected on: b, N_part, N_coll for FastHydro; N_strings for the 3D MC-Glauber.

Run it from the X-SCAPE build tree, or pass --build.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
CONTRIB = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(CONTRIB, "python"))

import make_hadron_wake_data as _mhw  # noqa: E402  (EOS 9 lookup/fetch)

MUSIC_XML = os.path.join(CONTRIB, "config", "AuAu_MCGlauber_MUSIC_0_10.xml")
FH_CFG = os.path.join(CONTRIB, "config", "fasthydro_particlize.yaml")
FH_REALISTIC = os.path.join(CONTRIB, "config", "fasthydro_wake_realistic.yaml")
FH_XML = os.path.join(CONTRIB, "config", "jetscape_user_fasthydro_particlize.xml")
VARIANTS = ("calibrated", "matched")


def _set(root, path, value):
    """Set <a><b>...</b></a> = value, creating what is missing."""
    node = root
    for tag in path.split("/"):
        nxt = node.find(tag)
        if nxt is None:
            nxt = ET.SubElement(node, tag)
        node = nxt
    node.text = str(value)          # no padding: path elements are read verbatim
    return node


def check_same_medium():
    """fasthydro_particlize.yaml must carry fasthydro_wake_realistic.yaml's initial state.

    The two files differ only in output settings (a Cooper-Frye surface needs frames that are
    not zeroed); if anything that shapes the medium drifts apart, this comparison stops being
    about the realistic config.
    """
    import yaml
    a, b = (yaml.safe_load(open(p)) for p in (FH_CFG, FH_REALISTIC))
    bad = [k for k in ("initial_state", "eos", "grid") if a.get(k) != b.get(k)]
    bad += [f"transport.{k}" for k in set(a["transport"]) | set(b["transport"])
            if k != "mode" and a["transport"].get(k) != b["transport"].get(k)]
    bad += [f"time.{k}" for k in ("tau0", "record_dtau", "choose_ntau")
            if a["time"].get(k) != b["time"].get(k)]
    return bad, a


def music_xml(variant, out_sub, a, fh_cfg):
    """The derived user XML for one MUSIC variant; -> (xml path, music_input path)."""
    tree = ET.parse(MUSIC_XML)
    root = tree.getroot()
    _set(root, "nEvents", a.events)
    _set(root, "Random/seed", a.seed)
    _set(root, "outputFilename", os.path.join(out_sub, "music"))
    _set(root, "vlevel", 0)
    _set(root, "SoftParticlization/iSS/number_of_repeated_sampling", a.oversample)

    # MUSIC rewrites its input file in place (EOS_to_use, Include_Bulk_Visc), and iSS reads
    # EOS_to_use back from <iSS_working_path>/music_input to pick its hadron list.  One private
    # copy per variant, which both point at, keeps the build tree's file out of it.
    music_input = os.path.join(out_sub, "music_input")
    template = os.path.join(a.build, "..", "examples", "test_music_files", "music_input")
    if not os.path.exists(template):
        template = os.path.join(a.build, "music_input")
    lines = open(template).read().splitlines()
    if variant == "matched":
        # tau_pi = shear_relax_time_factor * eta/(e+p) comes from the file, not the XML
        f = float(fh_cfg["transport"]["tau_pi_coeff"])
        lines = [f"shear_relax_time_factor {f}" if l.split()[:1] == ["shear_relax_time_factor"]
                 else l for l in lines]
    open(music_input, "w").write("\n".join(lines) + "\n")
    _set(root, "Hydro/MUSIC/MUSIC_input_file", music_input)
    _set(root, "SoftParticlization/iSS/iSS_working_path", out_sub)

    if variant == "matched":
        m = "Hydro/MUSIC/"
        _set(root, m + "shear_viscosity_eta_over_s", fh_cfg["transport"]["eta_over_s"])
        _set(root, m + "T_dependent_Shear_to_S_ratio", 0)     # >0 re-enables its own eta/s(T)
        _set(root, m + "temperature_dependent_bulk_viscosity", 0)
        _set(root, m + "Include_second_order_terms", 0)       # fast_data has none of them
        _set(root, "SoftParticlization/iSS/include_deltaf_shear", 0)
        _set(root, "SoftParticlization/iSS/include_deltaf_bulk", 0)

    rbw = root.find("RootBulkWriter")
    if rbw is not None:
        if a.root_bulk:
            _set(root, "RootBulkWriter/out_file_name", os.path.join(out_sub, "bulk.root"))
        else:
            root.remove(rbw)
    if not a.root_bulk:
        # iSS samples MUSIC's own surface (surface_in_memory); the stored evolution is only
        # for the bulk writer.  Every time step of it is ~65 GB RSS for one central event.
        _set(root, "Hydro/MUSIC/output_evolution_to_memory", 0)
    path = os.path.join(out_sub, "jetscape_user.xml")
    tree.write(path)
    return path, music_input


def parse_music_log(log):
    """Per-event string counts from the MCGlauber wrapper's log ("Produced N strings.").

    N_strings is the quantity 3dMCGlauber's centrality cut is made on.  Its b / Npart / Ncoll
    go only to events_summary.dat, which it rewrites every event, so they are not recoverable
    per event from a runJetscape run.
    """
    import re
    pat = re.compile(r"Produced\s+(\d+)\s+strings")
    rows = []
    with open(log, errors="replace") as fh:
        for line in fh:
            m = pat.search(line)
            if m:
                rows.append({"event": len(rows), "n_strings": int(m[1])})
    return rows


def run_music(variant, a, env, fh_cfg):
    out_sub = os.path.join(a.outdir, f"music_{variant}")
    os.makedirs(out_sub, exist_ok=True)
    xml, _ = music_xml(variant, out_sub, a, fh_cfg)
    cmd = [os.path.join(a.build, "runJetscape"), xml, a.main_xml]
    print(f"\n  --- music_{variant}: {a.events} event(s) x {a.oversample} oversamples ---")
    if a.dry_run:
        print("     " + " ".join(cmd))
        return 0
    log = os.path.join(out_sub, "run.log")
    t1 = time.time()
    with open(log, "w") as fh:
        r = subprocess.run(cmd, cwd=a.build, env=env, stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        return _failed(r.returncode, log)
    txt = os.path.join(out_sub, "music_final_state_hadrons.dat")
    meta = {"model": f"music_{variant}", "n_events": a.events, "n_oversample": a.oversample,
            "T_sw": 0.15, "user_xml": xml, "seed": a.seed, "wall_s": time.time() - t1,
            "events": parse_music_log(log)}
    return _finish(f"music_{variant}", txt, meta, a, t1)


def run_fasthydro(a, env):
    out_sub = os.path.join(a.outdir, "fasthydro")
    os.makedirs(out_sub, exist_ok=True)
    cmd = [sys.executable, os.path.join(HERE, "run_particlize.py"), "--leg", "bg",
           "--config", FH_CFG, "--user-xml", FH_XML, "--main-xml", a.main_xml,
           "--events", str(a.events), "--oversample", str(a.oversample),
           "--out-dir", out_sub, "--quiet",
           "--set", "transport.mode=israel_stewart",
           "--set", f"initial_state.b=[0.0, {a.b_max}]"]
    if a.device:
        cmd += ["--set", f"run.device={a.device}"]
        if a.device == "mps":
            cmd += ["--set", "run.dtype=float32"]
    print(f"\n  --- fasthydro: {a.events} event(s) x {a.oversample} oversamples, "
          f"b in [0, {a.b_max}] fm ---")
    if a.dry_run:
        print("     " + " ".join(cmd))
        return 0
    log = os.path.join(out_sub, "run.log")
    t1 = time.time()
    with open(log, "w") as fh:
        r = subprocess.run(cmd, cwd=a.build, env=env, stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        return _failed(r.returncode, log)
    meta = json.load(open(os.path.join(out_sub, "bg_events.json")))
    meta["model"] = "fasthydro"
    meta["b_range"] = [0.0, a.b_max]
    return _finish("fasthydro", os.path.join(out_sub, "bg_final_state_hadrons.dat"), meta, a, t1)


def _failed(code, log):
    print(f"     FAILED (exit {code}); last lines of {log}:")
    with open(log, errors="replace") as fh:
        for line in fh.readlines()[-20:]:
            print("       " + line.rstrip())
    return code


def _finish(model, txt, meta, a, t1):
    from fasthydro.hadrons import ascii_to_npz
    npz = os.path.join(a.outdir, f"hadrons_{model}.npz")
    n_ev = ascii_to_npz(txt, npz)
    meta["n_events_written"] = n_ev
    with open(os.path.join(a.outdir, f"{model}_events.json"), "w") as f:
        json.dump(meta, f, indent=1)
    mb = os.path.getsize(txt) / 1e6
    if not a.keep_ascii:
        os.remove(txt)
    print(f"     {time.time() - t1:.0f} s; {n_ev} event(s) -> {npz} "
          f"(text was {mb:.0f} MB{', kept' if a.keep_ascii else ', removed'})")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", default=".",
                    help="X-SCAPE build tree to run in (default: the working directory)")
    ap.add_argument("--out", default="out_bulk_vs_music", help="output directory, relative to --build")
    ap.add_argument("--models", default="music,fasthydro", help="comma list of music, fasthydro")
    ap.add_argument("--music-variants", default="calibrated",
                    help="comma list of calibrated, matched (see the docstring)")
    ap.add_argument("--events", type=int, default=10)
    ap.add_argument("--oversample", type=int, default=100, help="iSS samples per event")
    ap.add_argument("--seed", type=int, default=1, help="MUSIC run seed (FastHydro: its YAML's)")
    ap.add_argument("--b-max", type=float, default=4.7,
                    help="FastHydro: b sampled on [0, b_max] fm, P(b) ~ b (0-10%%: 4.7)")
    ap.add_argument("--device", default=None, help="FastHydro: cpu | cuda | mps")
    ap.add_argument("--main-xml", default=None,
                    help="default: <build>/../config/jetscape_main.xml")
    ap.add_argument("--root-bulk", action="store_true",
                    help="keep the XML's RootBulkWriter (the full MUSIC evolution, large)")
    ap.add_argument("--keep-ascii", action="store_true")
    ap.add_argument("--force", action="store_true", help="regenerate models that already exist")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    a.build = os.path.abspath(a.build)
    a.outdir = os.path.join(a.build, a.out)
    a.main_xml = os.path.abspath(a.main_xml or os.path.join(a.build, "..", "config",
                                                            "jetscape_main.xml"))
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    variants = [v.strip() for v in a.music_variants.split(",") if v.strip()]
    for v in variants:
        if v not in VARIANTS:
            ap.error(f"unknown MUSIC variant {v!r}; choose from {VARIANTS}")

    print("=== preflight ===")
    ok = True
    for what, p in (("build tree", a.build), ("runJetscape", os.path.join(a.build, "runJetscape")),
                    ("MUSIC XML", MUSIC_XML), ("FH config", FH_CFG), ("FH XML", FH_XML),
                    ("main XML", a.main_xml)):
        good = os.path.exists(p)
        ok &= good
        print(f"  {'ok  ' if good else 'MISS'}  {what:11s} {p}")
    bad, fh_cfg = check_same_medium()
    print(f"  {'ok  ' if not bad else 'DIFF'}  medium      fasthydro_particlize.yaml == "
          f"fasthydro_wake_realistic.yaml" + (f"  (differ: {bad})" if bad else ""))
    ok &= not bad
    if not ok:
        return 1
    if not a.dry_run and "fasthydro" in models and \
            _mhw.ensure_eos_9(a.build, allow_download=not a.no_download) is None:
        return 1
    os.makedirs(a.outdir, exist_ok=True)

    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", str(min(8, os.cpu_count() or 1)))
    env.setdefault("OMP_WAIT_POLICY", "passive")
    print(f"  OMP_NUM_THREADS={env['OMP_NUM_THREADS']} OMP_WAIT_POLICY={env['OMP_WAIT_POLICY']}")

    jobs = [(f"music_{v}", lambda v=v: run_music(v, a, env, fh_cfg)) for v in variants
            if "music" in models]
    if "fasthydro" in models:
        jobs.append(("fasthydro", lambda: run_fasthydro(a, env)))

    print(f"\n=== {a.events} event(s) x {a.oversample} oversamples -> {a.outdir} ===")
    t0 = time.time()
    for name, job in jobs:
        if os.path.exists(os.path.join(a.outdir, f"hadrons_{name}.npz")) and not a.force:
            print(f"\n  {name}: hadrons_{name}.npz exists, skipping (--force to regenerate)")
            continue
        rc = job()
        if rc:
            return rc
    if a.dry_run:
        return 0
    print(f"\n=== done in {time.time() - t0:.0f} s ===")
    print(f"Now run:  jupyter lab {os.path.join(CONTRIB, 'notebooks', 'bulk_vs_music.ipynb')}")
    print(f"          (set BULK_VS_MUSIC_OUT={a.outdir} if you run it from elsewhere)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
