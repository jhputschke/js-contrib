#!/usr/bin/env python3
"""
example/prod_AuAu_0_10_jet/hadronize.py

Hadronize what run_prod_jet.py --write-particlize stored: iSS on each leg's freeze-out
surface, X-SCAPE's ColorlessHadronization on the final partons (PLAN_particlize_h5.md).
No GPU, no MUSIC: only the stored inputs, the iSS tables and Pythia.

    conda activate js_fno
    python hadronize.py out/AuAu_0_10_jet_seed0001_particlize.h5           # all three tags
    python hadronize.py P.h5 --tags bulk_jet,bulk_bg --oversample 200
    python hadronize.py P.h5 --oversample 200 --oversample-bg auto   # reused bg: N x 200
    python hadronize.py P.h5 --tags jet_frag --n-frag 50
    python hadronize.py P.h5 --use-stored-seeds         # replay a --validate-inline job
    python hadronize.py P.h5 --diagnose-colored         # colour-flow diagnostic only
    python hadronize.py P.h5 --skip-complete            # campaigns: only what is missing
    python hadronize.py P.h5 --keep-bits-p 12 --keep-bits-x 8   # rounded p, x: 58% of the bytes

Output, next to the input (or in --out-dir), one file per tag (jetscape.hadrons_h5):

    <stem>_hadrons_bulk_jet.h5   iSS on surface/jet     bulk + wake        unit = event
    <stem>_hadrons_bulk_bg.h5    iSS on surface/bg      bulk               unit = background
    <stem>_hadrons_jet_frag.h5   Colorless on partons/  jet fragments      unit = event

A jet event is bulk_jet + jet_frag; its background is bulk_bg unit ``events/bg_unit`` of
the particlize file (units/event in the bulk_bg file is the first event that used it).
Oversamples of one event are samples of that event, not new events: average over them.

Seeds.  Every unit gets its own seed, derived from (--seed, tag, unit, sample), and it is
recorded in ``units/seed``: any unit can be regenerated alone.  --use-stored-seeds takes
the seeds a --validate-inline job recorded instead (bulk_jet: iSS, jet_frag: Pythia, one
fragmentation per event), so the result must equal that job's <stem>_inline_*.h5 exactly.

Precision.  p and x are stored as full float32 by default.  --keep-bits-p / --keep-bits-x
round them to that many mantissa bits (max relative error 2**-(bits+1)); the setting is
recorded on the datasets (jetscape.hadrons_h5.hadron_precision).  Choose it once per
campaign: --skip-complete keeps complete files whatever precision they have (with a
warning), and HadronFileReader warns about a campaign of mixed precision.

Settings come from hadronize.xml (the file --validate-inline also uses); the iSS paths are
made absolute and the job's own music_input -- stored in the particlize file -- is written
to iSS's working directory, so iSS reads the same EoS id and flags as it would have in the
job.  Empty surfaces (MUSIC stopped at the grid boundary) give a unit with zero samples,
which drops out of every sample average.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import signal
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROD = os.path.join(os.path.dirname(HERE), "prod_AuAu_0_10")
_spec = importlib.util.spec_from_file_location("_run_prod", os.path.join(PROD, "run_prod.py"))
rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp)                 # also puts PyJetscape/python on sys.path

TAGS = ("bulk_jet", "bulk_bg", "jet_frag")
TAG_INDEX = {t: k for k, t in enumerate(TAGS)}


def _mantissa_bits(text):
    v = int(text)
    if not 1 <= v <= 23:
        raise argparse.ArgumentTypeError(f"must be 1..23 (float32 mantissa bits), got {v}")
    return v


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("particlize", help="a <stem>_particlize.h5 from run_prod_jet.py")
    p.add_argument("--tags", default=",".join(TAGS),
                   help=f"comma-separated subset of {', '.join(TAGS)} (default: all)")
    p.add_argument("--oversample", type=int, default=None,
                   help="iSS samples per surface (default: hadronize.xml's "
                        "number_of_repeated_sampling)")
    p.add_argument("--oversample-bg", default=None, dest="oversample_bg",
                   help="iSS samples per BACKGROUND surface: a number, or 'auto' = "
                        "--oversample x the number of events using that background (the "
                        "optimal split under --reuse N; capped at --oversample-bg-max). "
                        "Default: --oversample")
    p.add_argument("--oversample-bg-max", type=int, default=2000, dest="oversample_bg_max",
                   help="cap for --oversample-bg auto (default 2000, ~6 GB for iSS)")
    p.add_argument("--n-frag", type=int, default=10, dest="n_frag",
                   help="Colorless fragmentations per event (default 10)")
    p.add_argument("--seed", type=int, default=1, help="base seed (default 1)")
    p.add_argument("--events", default=None,
                   help="event range a:b (python slice of the file's events; default all)")
    p.add_argument("--use-stored-seeds", action="store_true", dest="use_stored_seeds",
                   help="replay the seeds a --validate-inline job recorded (bulk_jet and "
                        "jet_frag; one fragmentation per event)")
    p.add_argument("--diagnose-colored", action="store_true", dest="diagnose_colored",
                   help="only run the colour-flow diagnostic (ColoredHadronization on the "
                        "stored partons) and write <stem>_colored_diagnostic.json")
    p.add_argument("--colored-event", type=int, default=None, dest="colored_event",
                   help=argparse.SUPPRESS)   # internal: one event through Colored, in a child
    p.add_argument("--keep-bits-p", type=_mantissa_bits, default=None, dest="keep_bits_p",
                   help="round the momenta p = (E, px, py, pz) to this many float32 mantissa "
                        "bits (1-23; default: full precision). 12 bits: relative error "
                        "<= 1.2e-4. Decide per campaign (see the README)")
    p.add_argument("--keep-bits-x", type=_mantissa_bits, default=None, dest="keep_bits_x",
                   help="round the positions x = (t, x, y, z) likewise; 8 bits: <= 2e-3, "
                        "i.e. <= 0.03 fm at 15 fm (MUSIC's cells are 0.2 fm)")
    p.add_argument("--out-dir", default=None, dest="out_dir",
                   help="output directory (default: next to the input)")
    p.add_argument("--hadronize-xml", default=os.path.join(HERE, "hadronize.xml"),
                   dest="hadronize_xml")
    p.add_argument("--build", default=os.path.join(rp.XSCAPE, "build_gpu"),
                   help="X-SCAPE build tree (default: build_gpu)")
    p.add_argument("--main-xml", default=os.path.join(rp.XSCAPE, "config", "jetscape_main.xml"),
                   dest="main_xml")
    p.add_argument("--workdir", default=None,
                   help="private working directory (default: <out-dir>/work_hadronize/<stem>)")
    p.add_argument("--keep-workdir", action="store_true", dest="keep_workdir")
    p.add_argument("--force", action="store_true", help="overwrite existing outputs")
    p.add_argument("--skip-complete", action="store_true", dest="skip_complete",
                   help="for campaigns: skip tags whose output exists and is complete, redo "
                        "the missing or incomplete ones (exit 0 if nothing is left to do)")
    return p.parse_args(argv)


def unit_seed(base, tag, unit, sample=0):
    """Independent, reproducible 31-bit seed per (base, tag, unit, sample)."""
    s = np.random.SeedSequence([int(base), TAG_INDEX[tag], int(unit), int(sample)])
    return int(s.generate_state(1, dtype=np.uint32)[0] & 0x7FFFFFFF) or 1


def _set_text(parent, tag, value):
    n = parent.find(tag)
    if n is None:
        n = ET.SubElement(parent, tag)
    n.text = str(value)


def background_samples(pf, oversample, spec, cap):
    """-> ({background unit: iSS samples}, [units that hit the cap]).

    ``spec`` None: ``oversample``; a number: that; 'auto': ``oversample`` x the number of
    events in the file that use the background (``events/bg_unit``), at most ``cap``.
    """
    n_bg, _ = pf.bg_units()
    if spec is None:
        return {u: oversample for u in range(n_bg)}, []
    if str(spec).lower() != "auto":
        n = int(spec)
        if n < 1:
            raise ValueError(f"--oversample-bg must be >= 1 or 'auto', got {spec!r}")
        return {u: n for u in range(n_bg)}, []
    uses = np.bincount(np.asarray(pf.events("bg_unit"), dtype=np.int64), minlength=n_bg)
    out, capped = {}, []
    for u in range(n_bg):
        n = oversample * max(int(uses[u]), 1)
        if n > cap:
            capped.append(u)
            n = cap
        out[u] = n
    return out, capped


def write_job_xml(a, path, n_exec, oversample):
    """hadronize.xml with this run's iSS paths, oversample count, seed and event count."""
    tree = ET.parse(a.hadronize_xml)
    root = tree.getroot()
    _set_text(root, "nEvents", max(1, n_exec))
    root.find("Random/seed").text = str(a.seed)
    iss = root.find("SoftParticlization/iSS")
    iss_dir = os.path.join(rp.XSCAPE, "external_packages", "iSS")
    _set_text(iss, "iSS_input_file", os.path.join(iss_dir, "iSS_parameters.dat"))
    _set_text(iss, "iSS_table_path", os.path.join(iss_dir, "iSS_tables"))
    _set_text(iss, "iSS_particle_table_path", os.path.join(iss_dir, "iSS_tables"))
    _set_text(iss, "iSS_working_path", ".")
    _set_text(iss, "number_of_repeated_sampling", oversample)
    tree.write(path)
    return ET.tostring(root, encoding="unicode")


def _complete(path):
    """True if ``path`` is a hadron file that was closed as complete."""
    if not os.path.exists(path):
        return False
    try:
        import h5py
        with h5py.File(path, "r") as f:
            return bool(f.attrs.get("complete", False))
    except OSError:                              # truncated by a kill
        return False


def event_range(spec, n):
    if not spec:
        return list(range(n))
    parts = [int(x) if x else None for x in spec.split(":")]
    return list(range(n))[slice(*parts)]


def colour_stats(partons):
    """Colour-flow numbers of one event's final partons (FINAL_PARTON_COLUMNS rows)."""
    pid, pstat = partons[:, 1].astype(int), partons[:, 2].astype(int)
    col, acol = partons[:, 12].astype(int), partons[:, 13].astype(int)
    qg = (np.abs(pid) <= 6) | (pid == 21)

    def unpaired(sel):
        n = 0
        for sh in np.unique(partons[sel, 0]):
            m = sel & (partons[:, 0] == sh) & qg
            c = [x for x in col[m] if x]
            ac = [x for x in acol[m] if x]
            for x in list(c):
                if x in ac:
                    ac.remove(x)
                    c.remove(x)
            n += len(c) + len(ac)
        return n

    everything = np.ones(len(pid), bool)
    kept = ~np.isin(pstat, (-11, -17))
    return {
        "n_final": int(len(pid)),
        "n_qg": int(qg.sum()),
        "frac_qg_colourless": float((qg & (col == 0) & (acol == 0)).sum() / max(qg.sum(), 1)),
        "unpaired_tags_all": unpaired(everything),
        "unpaired_tags_without_absorbed": unpaired(kept),
        "n_absorbed": int((~kept).sum()),
    }


def _terminate(signum, frame):
    # SIGTERM would kill the process without running the finally blocks, leaving
    # truncated HDF5 files; turn it into an exception so the writers close.
    raise SystemExit(128 + signum)


def main(argv=None):
    signal.signal(signal.SIGTERM, _terminate)
    a = parse_args(argv)
    tags = [t.strip() for t in a.tags.split(",") if t.strip()]
    bad = set(tags) - set(TAGS)
    if bad:
        sys.exit(f"hadronize.py: unknown tag(s) {sorted(bad)}; choose from {TAGS}")
    for k in ("particlize", "hadronize_xml", "build", "main_xml"):
        setattr(a, k, os.path.abspath(getattr(a, k)))
    keep_bits = {"p": a.keep_bits_p, "x": a.keep_bits_x}

    from jetscape import pyjetscape_core as core
    from jetscape.hadrons_h5 import HadronH5Writer, _keep_bits_map, hadron_precision
    from jetscape.particlize_h5 import ParticlizeFile

    pf = ParticlizeFile(a.particlize)
    stem = os.path.basename(a.particlize)
    stem = stem[:-len("_particlize.h5")] if stem.endswith("_particlize.h5") \
        else os.path.splitext(stem)[0]
    out_dir = os.path.abspath(a.out_dir or os.path.dirname(a.particlize))
    os.makedirs(out_dir, exist_ok=True)
    events = event_range(a.events, pf.nevents)
    if a.diagnose_colored and a.colored_event is None:
        return diagnose_colored(a, pf, events, out_dir, stem)
    if a.colored_event is not None:
        tags = []
    for t in tags:
        if t in ("bulk_jet", "bulk_bg") and t.split("_")[1] not in pf.legs:
            sys.exit(f"hadronize.py: {a.particlize} has no {t.split('_')[1]} surface "
                     f"(legs {pf.legs}); drop {t} from --tags")
        if t == "jet_frag" and "partons" not in pf.f:
            sys.exit(f"hadronize.py: {a.particlize} has no partons/")
    if a.use_stored_seeds and any(v is not None for v in keep_bits.values()):
        print("hadronize.py: note -- --keep-bits with --use-stored-seeds: the in-job "
              "(--validate-inline) hadrons are full precision, so compare after rounding "
              "them the same way, or validate without --keep-bits", file=sys.stderr)
    if a.use_stored_seeds:
        for key in ("inline_iss_seed_jet", "inline_pythia_seed"):
            if not pf.has_events(key):
                sys.exit(f"hadronize.py: --use-stored-seeds needs events/{key} (a "
                         "--validate-inline job)")
    xml_root = ET.parse(a.hadronize_xml).getroot()
    oversample = a.oversample or int(
        xml_root.find("SoftParticlization/iSS/number_of_repeated_sampling").text)
    n_frag = 1 if a.use_stored_seeds else a.n_frag
    try:
        bg_samples, bg_capped = background_samples(pf, oversample, a.oversample_bg,
                                                   a.oversample_bg_max)
    except ValueError as err:
        sys.exit(f"hadronize.py: {err}")
    if bg_capped and "bulk_bg" in [t.strip() for t in a.tags.split(",")]:
        print(f"hadronize.py: WARNING -- --oversample-bg auto capped {len(bg_capped)} "
              f"background(s) at --oversample-bg-max {a.oversample_bg_max}", file=sys.stderr)

    outs = {t: os.path.join(out_dir, f"{stem}_hadrons_{t}.h5") for t in tags}
    if a.skip_complete:
        done = [t for t in tags if _complete(outs[t])]
        if done:
            print(f"hadronize.py: {os.path.basename(a.particlize)}: already complete: "
                  f"{', '.join(done)}")
        wanted = _keep_bits_map(keep_bits)
        for t in done:
            have = hadron_precision(outs[t])
            if have != wanted:
                print(f"hadronize.py: WARNING -- {os.path.basename(outs[t])} is complete but "
                      f"was written with precision {have}, not the requested {wanted} (None "
                      "= full float32); kept as it is (--force redoes it)", file=sys.stderr)
        tags = [t for t in tags if t not in done]
        outs = {t: outs[t] for t in tags}
        if not tags:
            return 0
        a.force = True                          # what is left is missing or incomplete

    bg_units = []
    if "bulk_bg" in tags:
        seen = set()
        for e in events:
            u = int(pf.events("bg_unit")[e])
            if u not in seen:
                seen.add(u)
                bg_units.append((u, e))
    n_exec = (len(events) if "bulk_jet" in tags else 0) + len(bg_units)

    for t, path in outs.items():
        if os.path.exists(path) and not a.force:
            sys.exit(f"hadronize.py: {path} exists (use --force)")

    # Private working directory: iSS reads music_input from it, Pythia may write there.
    workdir = os.path.abspath(a.workdir or os.path.join(out_dir, "work_hadronize", stem))
    os.makedirs(workdir, exist_ok=True)
    with open(os.path.join(workdir, "music_input"), "w") as fh:
        fh.write(pf.music_input())
    job_xml = os.path.join(workdir, "hadronize_job.xml")
    xml_text = write_job_xml(a, job_xml, n_exec, oversample)
    main_xml = os.path.join(workdir, "jetscape_main.xml")
    rp._absolute_parent_paths(a.main_xml, main_xml, a.build)
    cwd = os.getcwd()
    os.chdir(workdir)
    print(f"hadronize.py: {a.particlize}\n  {len(events)} event(s), tags {', '.join(tags) or '-'}"
          f", iSS oversample {oversample}"
          + (f" (backgrounds: {sorted(set(bg_samples.values()))})"
             if "bulk_bg" in tags and set(bg_samples.values()) != {oversample} else "")
          + f", {n_frag} fragmentation(s)/event, base seed "
          f"{a.seed}{' (stored seeds)' if a.use_stored_seeds else ''}"
          + (f", p/x rounded to {a.keep_bits_p or 23}/{a.keep_bits_x or 23} mantissa bits"
             if any(v is not None for v in keep_bits.values()) else "")
          + f"\n  workdir {workdir}")

    jetscape = core.JetScapePerEvent()
    jetscape.SetXMLMainFileName(main_xml)
    jetscape.SetXMLUserFileName(job_xml)
    replay = iss = None
    if "bulk_jet" in tags or "bulk_bg" in tags:
        from jetscape.surface_replay import SurfaceReplay
        replay = SurfaceReplay()
        iss = core.create_module("iSS")
        jetscape.Add(replay)
        jetscape.Add(iss)
    jetscape.Init()
    jetscape.ExecInit()
    colorless = None
    if "jet_frag" in tags:
        colorless = core.create_module("ColorlessHadronization")
        colorless.Init()

    if a.colored_event is not None:
        return colored_one_event(core, pf, a.colored_event, cwd)

    common = {"source": os.path.basename(a.particlize),
              "source_uuid": str(pf.attrs.get("file_uuid", "")),
              "hadronize_xml": xml_text, "base_seed": a.seed,
              "stored_seeds": bool(a.use_stored_seeds),
              "music_input": pf.music_input()}
    writers = {}
    for t in tags:
        extra = {}
        if t == "jet_frag":
            n = n_frag
        elif t == "bulk_bg" and a.oversample_bg is not None:
            auto = str(a.oversample_bg).lower() == "auto"
            n = 0 if auto else int(a.oversample_bg)     # 0: per unit, see units/n_samples
            extra = {"oversample_bg": str(a.oversample_bg),
                     "oversample_bg_max": int(a.oversample_bg_max),
                     "oversample_jet": int(oversample)}
        else:
            n = oversample
        writers[t] = HadronH5Writer(outs[t], tag=t, n_samples=n, keep_bits=keep_bits,
                                    attrs=dict(common, **extra, generator=(
                                        "ColorlessHadronization (X-SCAPE)" if t == "jet_frag"
                                        else "iSS (X-SCAPE iSpectraSamplerWrapper)")))

    def run_iss(cells, seed, n_samples):
        if len(cells) == 0:
            return None, seed
        replay.load(cells)
        core.soft_set_number_of_samples(iss, int(n_samples))
        core.soft_set_next_random_seed(iss, int(seed))
        jetscape.ExecPerEvent()
        h = core.soft_hadrons_numpy(iss)
        used = int(core.soft_last_random_seed(iss))
        jetscape.ClearPerEvent()
        if used != int(seed) or replay.n_cells != len(cells):
            # iSS (or the replay) was switched off for this framework event -- e.g. hydro
            # reuse -- and h would be the previous surface's hadrons
            raise RuntimeError(f"iSS did not sample this surface (seed {used} != {seed}, "
                               f"replayed {replay.n_cells} of {len(cells)} cells); check "
                               "<setReuseHydro> false in the hadronization XML")
        if len(h["sample_counts"]) != int(n_samples):
            raise RuntimeError(f"iSS made {len(h['sample_counts'])} samples, "
                               f"{n_samples} were asked for")
        return h, used

    t0 = time.time()
    finished = False
    bg_done = set()
    bg_first = dict(bg_units)
    try:
        for n, e in enumerate(events):
            msg = [f"event {e}"]
            if "bulk_jet" in tags:
                seed = (int(pf.events("inline_iss_seed_jet")[e]) if a.use_stored_seeds
                        else unit_seed(a.seed, "bulk_jet", e))
                cells = pf.surface_unit("jet", e)
                h, used = run_iss(cells, seed, oversample)
                writers["bulk_jet"].append_unit(h if h is not None else [], unit=e, event=e,
                                                seed=used, n_cells=len(cells))
                msg.append(f"bulk_jet {0 if h is None else len(h['pid'])} hadrons")
            if "bulk_bg" in tags:
                u = int(pf.events("bg_unit")[e])
                if u not in bg_done:
                    cells = pf.surface_unit("bg", u)
                    h, used = run_iss(cells, unit_seed(a.seed, "bulk_bg", u), bg_samples[u])
                    writers["bulk_bg"].append_unit(h if h is not None else [], unit=u,
                                                   event=bg_first[u], seed=used,
                                                   n_cells=len(cells))
                    bg_done.add(u)
                    msg.append(f"bulk_bg (unit {u}) {0 if h is None else len(h['pid'])}")
            if "jet_frag" in tags:
                partons = pf.partons(e)
                samples, seeds = [], []
                for s in range(n_frag):
                    seed = (int(pf.events("inline_pythia_seed")[e]) if a.use_stored_seeds
                            else unit_seed(a.seed, "jet_frag", e, s))
                    if len(partons):
                        d = core.hadronize_partons(colorless, partons, seed)
                    else:
                        d = {k: np.zeros((0, 4) if k in ("p", "x") else 0) for k in
                             ("pid", "pstat", "p", "x")}
                    samples.append(d)
                    seeds.append(seed)
                writers["jet_frag"].append_unit(samples, unit=e, event=e, seed=seeds[0],
                                                n_partons=len(partons))
                msg.append(f"jet_frag {sum(len(d['pid']) for d in samples) / n_frag:.1f}"
                           "/sample")
            print("  " + ", ".join(msg), flush=True)
        jetscape.Finish()
        finished = True
    finally:
        # complete only if every event went through: an exception, Ctrl-C or SIGTERM
        # (run_hadronize.py stopping its children) leaves readable, incomplete files that
        # --skip-complete redoes
        for w in writers.values():
            w.close(complete=finished)
        os.chdir(cwd)
    if not a.keep_workdir:
        shutil.rmtree(workdir, ignore_errors=True)
        try:
            os.rmdir(os.path.dirname(workdir))  # work_hadronize/, once it is empty
        except OSError:
            pass
    print(f"hadronize.py: done in {time.time() - t0:.1f} s")
    for t, path in outs.items():
        print(f"  {t:9s} -> {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
    return 0


def colored_one_event(core, pf, e, cwd):
    """Child process of --diagnose-colored: one event through ColoredHadronization."""
    colored = core.create_module("ColoredHadronization")
    colored.Init()
    partons = pf.partons(e)
    d = core.hadronize_partons(colored, partons) if len(partons) else {"pid": []}
    pid = np.asarray(d["pid"])
    partonic = (np.abs(pid) <= 6) | (pid == 21)
    os.chdir(cwd)
    print("COLORED_RESULT " + json.dumps({"n_out": int(len(pid)),
                                          "n_out_partons": int(partonic.sum())}), flush=True)
    return 0


def diagnose_colored(a, pf, events, out_dir, stem):
    """Colour flow of the stored partons, and what ColoredHadronization makes of it.

    ColoredHadronization can abort the process (an assertion on a Pythia output id), so each
    event runs in its own child process; a crash is recorded, not fatal.
    """
    import subprocess

    rows = []
    for e in events:
        st = colour_stats(pf.partons(e))
        st["event"] = int(e)
        cmd = [sys.executable, os.path.abspath(__file__), a.particlize, "--colored-event",
               str(e), "--build", a.build, "--main-xml", a.main_xml,
               "--hadronize-xml", a.hadronize_xml, "--out-dir", out_dir, "--force",
               "--workdir", os.path.join(out_dir, "work_hadronize", f"{stem}_colored_{e}")]
        r = subprocess.run(cmd, capture_output=True, text=True)
        res = [ln for ln in r.stdout.splitlines() if ln.startswith("COLORED_RESULT ")]
        if r.returncode == 0 and res:
            out = json.loads(res[-1].split(" ", 1)[1])
            st.update(out, crashed=False,
                      failed=bool(out["n_out_partons"] > 0 or out["n_out"] == 0))
        else:
            tail = (r.stderr.strip().splitlines() or ["?"])[-1]
            st.update(crashed=True, failed=True, returncode=r.returncode,
                      error=tail[-300:])
        rows.append(st)
        print(f"  event {e}: {st}", flush=True)
    summary = {
        "source": a.particlize,
        "events": rows,
        "mean_frac_qg_colourless": float(np.mean([r["frac_qg_colourless"] for r in rows]))
        if rows else None,
        "failed_fraction": float(np.mean([r["failed"] for r in rows])) if rows else None,
        "crashed_fraction": float(np.mean([r["crashed"] for r in rows])) if rows else None,
    }
    path = os.path.join(out_dir, f"{stem}_colored_diagnostic.json")
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"hadronize.py: colour diagnostic -> {path}: mean colourless q/g fraction "
          f"{summary['mean_frac_qg_colourless']}, ColoredHadronization failed in "
          f"{summary['failed_fraction']} (crashed in {summary['crashed_fraction']}) of events")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
