#!/usr/bin/env python3
"""
example/prod_OO_isr/rehadronize_jets.py

Hadronize the O+O jets of a run_prod_oo_isr.py --store-hadronization-input job again, from
the parton lists its jet_frag file stored (hadronization_input/: the final partons, the beam
remnants from 3dMCGlauber and the colour-balancing parton, exactly as iColoredHadronization
handed them to Pythia).  No 3dMCGlauber, no hydro: only Pythia, with iColoredHadronization's
settings (the jet_frag file's jet_hadronization_xml).

    conda activate js_fno
    python rehadronize_jets.py out/OO_isr_seed0001_hadrons_jet_frag.h5 --check
                                    # the stored seeds: must give the job's jet hadrons
    python rehadronize_jets.py F.h5 --decays off --out F_nodecay.h5
                                    # the primary hadrons (decays off), same seeds
    python rehadronize_jets.py F.h5 --new-seeds 7 --out F_seed7.h5
                                    # another fragmentation of the same partons

--check needs the same Pythia version as the job (the file's pythia_version) and a jet_frag
file written at full precision without a selection (no --keep-bits, --eta-max, --charged);
it compares pid, pstat and p of every event and exits 1 on a difference.

Output (--out): a jet_frag file (jetscape.hadrons_h5) with one sample per event, the seeds
in units/seed, the source file's initiators/ and hadronization_input/, and the settings used
(decays, pythia_version, rehadronized_from).  Positions are zero, as in the job: hadron
positions are a later step (PLAN_smash_ready_hadrons.md 2.2-2.3).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROD = os.path.join(os.path.dirname(HERE), "prod_AuAu_0_10")
_spec = importlib.util.spec_from_file_location("_run_prod", os.path.join(PROD, "run_prod.py"))
rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp)                 # also puts PyJetscape/python on sys.path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("jet_frag", help="a <stem>_hadrons_jet_frag.h5 with hadronization_input/")
    p.add_argument("--check", action="store_true",
                   help="hadronize with the stored seeds and compare with the stored hadrons")
    p.add_argument("--out", default=None, help="write the new jet hadrons to this file")
    p.add_argument("--decays", choices=("on", "off"), default=None,
                   help="Pythia's hadron decays (iColored <weak_decays>; default: as in the "
                        "job). 'off' gives the primary hadrons")
    p.add_argument("--new-seeds", type=int, default=None, dest="new_seeds", metavar="BASE",
                   help="seeds derived from (BASE, event) instead of the stored ones")
    p.add_argument("--events", default=None, help="unit range a:b (default all)")
    p.add_argument("--build", default=os.path.join(rp.XSCAPE, "build_gpu"),
                   help="X-SCAPE build tree (default: build_gpu)")
    p.add_argument("--main-xml", default=os.path.join(rp.XSCAPE, "config", "jetscape_main.xml"),
                   dest="main_xml")
    p.add_argument("--workdir", default=None,
                   help="working directory (default: a temporary one next to --out or the "
                        "input)")
    p.add_argument("--force", action="store_true", help="overwrite --out")
    return p.parse_args(argv)


def new_seed(base, unit):
    """Independent, reproducible 31-bit seed per (base, unit)."""
    s = np.random.SeedSequence([int(base), 2, int(unit)])
    return int(s.generate_state(1, dtype=np.uint32)[0] & 0x7FFFFFFF) or 1


def user_xml(jet_hadronization_xml, decays):
    """A user XML with only the job's effective <JetHadronization>."""
    root = ET.fromstring("<jetscape><enableAutomaticTaskListDetermination>false"
                         "</enableAutomaticTaskListDetermination><nEvents>1</nEvents>"
                         "<vlevel>0</vlevel><Random><seed>1</seed></Random></jetscape>")
    for w in ("JetScapeWriterAscii", "JetScapeWriterAsciiGZ", "JetScapeWriterHepMC",
              "JetScapeWriterFinalStatePartonsAscii", "JetScapeWriterFinalStateHadronsAscii"):
        ET.SubElement(root, w).text = "off"
    jh = ET.fromstring(jet_hadronization_xml)
    for tag in ("reseed_per_event", "store_input"):   # the seed comes with every event
        if jh.find(tag) is not None:
            jh.remove(jh.find(tag))
    if decays is not None:
        w = jh.find("weak_decays")
        if w is None:
            w = ET.SubElement(jh, "weak_decays")
        w.text = decays
    root.append(jh)
    return root


def main(argv=None):
    a = parse_args(argv)
    for k in ("jet_frag", "build", "main_xml"):
        setattr(a, k, os.path.abspath(getattr(a, k)))
    if not a.check and not a.out:
        sys.exit("rehadronize_jets.py: nothing to do: pass --check and/or --out")
    if a.check and (a.decays is not None or a.new_seeds is not None):
        sys.exit("rehadronize_jets.py: --check replays the job: not with --decays / --new-seeds")

    from jetscape import pyjetscape_core as core
    from jetscape.hadrons_h5 import (HadronFile, HadronH5Writer, hadron_precision,
                                     hadron_selection)

    if not getattr(core, "ICOLORED_HAS_INPUT_RECORD", False):
        sys.exit("rehadronize_jets.py: this PyJetscape has no hadronize_input (X-SCAPE "
                 "smash_ready or later)")
    hf = HadronFile(a.jet_frag)
    if hf.tag != "jet_frag" or not hf.has_hadronization_input:
        sys.exit(f"rehadronize_jets.py: {a.jet_frag} is no jet_frag file with "
                 "hadronization_input/ (run_prod_oo_isr.py --store-hadronization-input)")
    jh_xml = str(hf.attrs.get("jet_hadronization_xml", ""))
    if not jh_xml:
        sys.exit(f"rehadronize_jets.py: {a.jet_frag} has no jet_hadronization_xml")
    have_pythia = int(getattr(core, "PYTHIA_VERSION", 0))
    job_pythia = int(hf.attrs.get("pythia_version", 0))
    if job_pythia != have_pythia:
        msg = (f"the job ran Pythia {job_pythia}, this build has {have_pythia}: the same "
               "seeds give other hadrons")
        if a.check:
            sys.exit(f"rehadronize_jets.py: --check: {msg}")
        print(f"rehadronize_jets.py: note -- {msg}", file=sys.stderr)
    if a.check:
        sel, prec = hadron_selection(a.jet_frag), hadron_precision(a.jet_frag)
        if sel["eta_max"] is not None or sel["charged_only"] or prec["p"] is not None:
            sys.exit(f"rehadronize_jets.py: --check needs a jet_frag file at full precision "
                     f"without a selection (this one: {sel}, precision {prec})")
    seeds = hf.units.get("seed")
    if seeds is None and a.new_seeds is None:
        sys.exit("rehadronize_jets.py: the file records no seeds (units/seed): a job before "
                 "the per-event reseed; use --new-seeds")
    units = list(range(hf.n_units))
    if a.events:
        units = units[slice(*[int(x) if x else None for x in a.events.split(":")])]
    out = os.path.abspath(a.out) if a.out else None
    if out and os.path.exists(out) and not a.force:
        sys.exit(f"rehadronize_jets.py: {out} exists (use --force)")

    # A private working directory: Pythia may write there.
    base = os.path.dirname(out or a.jet_frag)
    workdir = os.path.abspath(a.workdir or os.path.join(
        base, "work_rehadronize", os.path.basename(a.jet_frag).replace(".h5", "")))
    os.makedirs(workdir, exist_ok=True)
    job_xml = os.path.join(workdir, "rehadronize.xml")
    ET.ElementTree(user_xml(jh_xml, a.decays)).write(job_xml)
    main_xml = os.path.join(workdir, "jetscape_main.xml")
    rp._absolute_parent_paths(a.main_xml, main_xml, a.build)
    cwd = os.getcwd()
    os.chdir(workdir)

    writer = None
    n_bad = 0
    finished = False
    try:
        jetscape = core.JetScapePerEvent()
        jetscape.SetXMLMainFileName(main_xml)
        jetscape.SetXMLUserFileName(job_xml)
        jetscape.Init()
        module = core.create_module("iColoredHadronization")
        module.Init()
        if out:
            writer = HadronH5Writer(
                out, tag="jet_frag", n_samples=1, initiators=hf.has_initiators,
                hadronization_input=True,
                attrs={**{k: v for k, v in hf.attrs.items()
                          if k in ("prod", "prod_seed", "system", "prod_hard", "eos",
                                   "subtract_hard_momentum", "source", "prod_user_xml")},
                       "module": "iColoredHadronization (rehadronize_jets.py)",
                       "rehadronized_from": os.path.basename(a.jet_frag),
                       "jet_hadronization_xml": ET.tostring(
                           ET.parse(job_xml).getroot().find("JetHadronization"),
                           encoding="unicode"),
                       "decays": a.decays or "as in the job",
                       "seeds": "stored" if a.new_seeds is None else f"new, base {a.new_seeds}",
                       "pythia_version": have_pythia, "decayed": a.decays != "off"})
        for u in units:
            unit = int(hf.units["unit"][u]) if "unit" in hf.units else u
            inp = hf.hadronization_input(unit)
            seed = (int(seeds[u]) if a.new_seeds is None else new_seed(a.new_seeds, unit))
            d = core.hadronize_input(module, inp["pid"], inp["col"], inp["acol"], inp["p"],
                                     inp["mass"], seed)
            if a.check:
                old = hf.sample_event(unit, 0)
                same = (np.array_equal(old["pid"], d["pid"])
                        and np.array_equal(old["pstat"], d["pstat"])
                        and np.array_equal(old["p"], np.asarray(d["p"], dtype=np.float32)))
                n_bad += not same
                print(f"  unit {unit}: {len(d['pid'])} hadrons, seed {seed}: "
                      + ("same as the job" if same else
                         f"DIFFERENT ({len(old['pid'])} in the job)"), flush=True)
            else:
                print(f"  unit {unit}: {len(d['pid'])} hadrons, seed {seed}", flush=True)
            if writer is not None:
                writer.append_unit([d], initiators=hf.initiators(unit), hadronization_input=inp,
                                   unit=unit, event=int(hf.units.get("event", units)[u]),
                                   seed=seed)
        jetscape.Finish()
        finished = True
    finally:
        if writer is not None:
            writer.close(complete=finished)
        hf.close()
        os.chdir(cwd)
        shutil.rmtree(workdir, ignore_errors=True)
        try:
            os.rmdir(os.path.dirname(workdir))
        except OSError:
            pass
    if a.check:
        print(f"rehadronize_jets.py: {len(units) - n_bad} of {len(units)} event(s) "
              f"reproduce the job's jet hadrons" + ("" if not n_bad else " -- MISMATCH"))
    if out:
        print(f"rehadronize_jets.py: -> {out}")
    return 1 if n_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
