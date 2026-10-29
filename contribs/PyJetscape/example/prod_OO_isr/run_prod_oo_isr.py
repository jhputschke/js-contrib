#!/usr/bin/env python3
"""
example/prod_OO_isr/run_prod_oo_isr.py

O+O with initial-state radiation (ISR) and jet energy loss: one MUSIC run per event (no
two-stage hydro), written straight to HDF5 like the Au+Au productions.

    IS (3dMCGlauber O16, positions) -> PythiaIsrGun -> iMATTER (ISR) -> 3dMCGlauber strings
      (the hard partons' momentum subtracted) -> NullPreDynamics -> MUSIC (music4gpu)
      -> Matter + LBT in MUSIC's medium -> iColoredHadronization (partons + beam remnants,
         through the colour flow; the lines that end in the medium closed, <colour_closure>)
      (--matter-only: Matter alone, the default before 2026-10-21)

One call = one job = one seed.  Each job writes, next to each other:

    <stem>.h5                    the hydro evolution on the grid YAML (jetscape.fast_h5_bulk
                                 file, as ../prod_AuAu_0_10's), diag/ per event: pTHat,
                                 sigma, Glauber Npart/Ncoll/Nstrings/b ...
    <stem>_particlize.h5         MUSIC's freeze-out surface (surface/bg) + the final partons
                                 + the shower initiators (jetscape.particlize_h5): iSS
                                 offline with ../prod_AuAu_0_10_jet/hadronize.py --tags bulk_bg
                                 (--no-particlize: not written)
    <stem>_hadrons_bulk_bg.h5    with --bulk-hadrons N: the bulk hadrons, iSS on MUSIC's
                                 surface in the job, N oversamples per event, the settings of
                                 hadronize.xml (the file hadronize.py --tags bulk_bg writes)
    <stem>_hadrons_jet_frag.h5   the jet hadrons of iColoredHadronization, made in the job
                                 (it needs the event's beam remnants), jetscape.hadrons_h5;
                                 with --store-hadronization-input also the parton list
                                 handed to Pythia per event (hadronization_input/), so
                                 rehadronize_jets.py can hadronize the jets again later
    <stem>_hadrons_bulk_bg_primary.h5
                                 with --bulk-hadrons and --keep-primaries N: the undecayed
                                 bulk hadrons of the first N oversamples (for SMASH later)
    <stem>.xml / .json           the job XML and a summary

    conda activate js_fno
    python run_prod_oo_isr.py --events 5 --seed 1                 # pTHat 20-40, b < 3 fm
    python run_prod_oo_isr.py --events 5 --seed 1 --pthat-min 10 --pthat-max 20
    python run_prod_oo_isr.py --events 5 --seed 1 --b-max 5
    python run_prod_oo_isr.py --events 30 --seed 1 --pthat-bins 10-20,20-40,40-60
                                     # event i in window i mod 3, every event complete
    python run_prod_oo_isr.py --events 30 --seed 1 --pthat-bins 10-20,20-30,30-40 \
                              --parton-ymax 0.6       # both hard partons at |y| < 0.6
    python run_prod_oo_isr.py --events 5 --seed 1 --bulk-hadrons 100 --no-particlize
                                     # all hadrons in the job, no surface file
    python run_prod_oo_isr.py --events 5 --seed 1 --store-hadronization-input
                                     # + each event's Pythia input: jets can be redone
    python run_prod_oo_isr.py --events 1 --seed 1 --dry-run       # check the XML/grid only
    ./run_jobs.sh 20 25 0                                         # 20 jobs x 25 events

Jet hadronization seeds.  iColoredHadronization is reseeded every event from the job's
seed (<reseed_per_event> 1) and the seed is recorded (units/seed of the jet_frag file,
jet_pythia_seed of every event record): each event's jet hadrons can be reproduced on their
own.  (Before, every job ran Pythia's hadronization from Pythia's default seed.)  With
--store-hadronization-input the jet_frag file also gets the parton list Pythia received
(final partons, beam remnants from 3dMCGlauber, the colour-balancing parton), the Pythia
version and the effective <JetHadronization> settings; rehadronize_jets.py then hadronizes
the jets again without 3dMCGlauber or the hydro: the same hadrons with the stored seeds
(--check), or with decays off, other seeds, a newer Pythia.  Opt-in: ~2 kB per event.

EOS.  The production XML runs MUSIC with EOS 91 (hotQCD + the SMASH hadron list, the EOS of
the tunes); --eos 9 reruns older EOS 9 campaigns.  With EOS 91 iSS decays the SMASH list
itself (hadronize.xml <decay_smash_list> 1, iSS from smash_ready on).

The hydro is never reused: with ISR the strings, and so the medium, are made from what each
event's hard scattering left, so every event (also with --pthat-bins) runs the whole chain.
X-SCAPE refuses <setReuseHydro> with ISR.

Needs X-SCAPE with <Hard><ISR> in the automatic task list (branch oo_isr_test), built with
3dMCGlauber and MUSIC (music4gpu), and PyJetscape built against it.  iMATTER reads Pythia's
PDF tables next to $PYTHIA8DATA (set by the js_fno env) or under $PYTHIA8DIR.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import platform
import socket
import sys
import time
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
PROD = os.path.join(os.path.dirname(HERE), "prod_AuAu_0_10")
JET_PROD = os.path.join(os.path.dirname(HERE), "prod_AuAu_0_10_jet")

# The Au+Au production's helpers (grid YAML, MUSIC box, seeds, workdir, env checks).
# Loading it also puts PyJetscape/python on sys.path.
_spec = importlib.util.spec_from_file_location("_run_prod", os.path.join(PROD, "run_prod.py"))
rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp)
# the Au+Au jet driver's pTHat-window parser, so --pthat-bins reads the same
_spec_jet = importlib.util.spec_from_file_location("_run_prod_jet",
                                                   os.path.join(JET_PROD, "run_prod_jet.py"))
rj = importlib.util.module_from_spec(_spec_jet)
_spec_jet.loader.exec_module(rj)

USER_XML = os.path.join(rp.CONFIG, "OO_MCGlauber_MUSIC_isr.xml")
GRID_YAML = os.path.join(rp.CONFIG, "grid_oo.yaml")
NAME_BASE = "OO_isr"
HYDRO_ID = "MUSIC"
GUN_ID = "PythiaGun"          # PythiaIsrGun takes its id from <Hard><PythiaGun><name>

#: 3dMCGlauber's per-event record in the working directory (EventGenerator::
#: generate_strings rewrites it every event): event_id Npart Ncoll Nstrings b
GLAUBER_SUMMARY = "events_summary.dat"
#: its strings of the event (MCGlauberGenString always writes event id 0); the header
#: line has the strings' total energy and net pz
GLAUBER_STRINGS = "strings_event_0.dat"
IS_ID = "MCGlauber"


def parse_args() -> argparse.Namespace:
    # no abbreviations: run_jobs.sh must recognise --campaign and the options it sets per job
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--events", type=int, default=10, help="events in this job (default 10)")
    p.add_argument("--grid", default=GRID_YAML,
                   help="output grid YAML of the hydro file (default: ../config/grid_oo.yaml)")
    p.add_argument("--outdir", default=os.path.join(HERE, "out"),
                   help="output directory (default: ./out next to this script)")
    p.add_argument("--out", default=None,
                   help="hydro file name (default: OO_isr[_<campaign>]_seed<NNNN>.h5, or "
                        "OO_isr_<campaign>_<NNNN>.h5 with --seed 0)")
    p.add_argument("--build", default=rp.BUILD,
                   help="X-SCAPE build tree (default: $JETSCAPE_BUILD, else build_gpu)")
    p.add_argument("--main-xml", default=rp.MAIN_XML,
                   dest="main_xml")
    p.add_argument("--user-xml", default=USER_XML, dest="user_xml",
                   help="template user XML (default: ../config/OO_MCGlauber_MUSIC_isr.xml)")
    p.add_argument("--pthat-min", type=float, default=None, dest="pthat_min",
                   help="PythiaGun pTHatMin [GeV] (default: the XML's, 20)")
    p.add_argument("--pthat-max", type=float, default=None, dest="pthat_max",
                   help="PythiaGun pTHatMax [GeV] (default: the XML's, 40)")
    p.add_argument("--pthat-bins", default=None, dest="pthat_bins",
                   help="several pTHat windows in one job, e.g. 10-20,20-40,40-60 [GeV]: "
                        "one Pythia per window, event i uses window i mod K. Every event "
                        "is complete (ISR, strings, MUSIC, Matter): no hydro reuse. Records "
                        "the window per event and each window's cross section. Use a "
                        "multiple of K events for the same number per window")
    p.add_argument("--parton-ymax", type=float, default=None, dest="parton_ymax",
                   help="keep only events whose two outgoing hard partons (Pythia's 2->2, "
                        "before the ISR) pass |y| < YMAX (see --parton-y-mode). Rejected "
                        "events are regenerated before anything else runs; the cross "
                        "sections are Pythia's x kept/tried per window. Records the "
                        "acceptance, and the two hardest partons' y and pT per event. Needs "
                        "X-SCAPE's PythiaIsrGun with <partonYMax>")
    p.add_argument("--parton-y-mode", choices=("both", "any"), default="both",
                   dest="parton_y_mode",
                   help="with --parton-ymax: 'both' hard partons (default) or 'any' of them "
                        "(their pT is equal before the ISR: no 'leading' as in the Au+Au "
                        "driver)")
    p.add_argument("--sqrts", type=float, default=None,
                   help="collision energy per nucleon pair [GeV], for 3dMCGlauber and Pythia "
                        "(default: the XML's, 200). The 3dMCGlauber/MUSIC settings are tuned "
                        "for O+O at 200 GeV")
    p.add_argument("--b-min", type=float, default=None, dest="b_min",
                   help="3dMCGlauber minimum impact parameter [fm] (default: the XML's, 0)")
    p.add_argument("--b-max", type=float, default=None, dest="b_max",
                   help="3dMCGlauber maximum impact parameter [fm] (default: the XML's, 3, "
                        "about 0-20%% of O+O). Inside X-SCAPE b selects the centrality; "
                        "the hard scattering sits at one of the binary collisions")
    p.add_argument("--late-dtau", type=float, default=None, dest="late_dtau", metavar="DT",
                   help="MUSIC's time step from --late-tau on [fm/c] (the XML's "
                        "<late_delta_tau>, 0.05 in the production XML; 0: constant step)")
    p.add_argument("--late-tau", type=float, default=None, dest="late_tau", metavar="TAU",
                   help="when the larger step starts [fm/c] (the XML's <late_tau_switch>)")
    p.add_argument("--bulk-info", choices=("slim", "full"), default="slim", dest="bulk_info",
                   help="MUSIC's framework copy, which Matter reads and the hydro file is "
                        "written from: 'slim' (default) e, s, T, vx, vy, vz per cell; 'full' "
                        "the whole FluidCellInfo")
    p.add_argument("--no-hydro-file", action="store_true", dest="no_hydro_file",
                   help="do not write the hydro evolution (<stem>.h5): MUSIC still runs and "
                        "Matter still reads its medium, nothing of it is stored (hydro_row "
                        "is then -1 for every event; has_medium tells which had one). The "
                        ".json is still written")
    p.add_argument("--no-particlize", action="store_true", dest="no_particlize",
                   help="do not write <stem>_particlize.h5 (MUSIC then builds no surface)")
    p.add_argument("--no-jet-hadrons", action="store_true", dest="no_jet_hadrons",
                   help="do not hadronize the jets in the job (no iColoredHadronization, "
                        "no <stem>_hadrons_jet_frag.h5)")
    p.add_argument("--bulk-hadrons", type=int, default=0, dest="bulk_hadrons", metavar="N",
                   help="also sample the bulk in the job: iSS on MUSIC's surface, N "
                        "oversamples per event -> <stem>_hadrons_bulk_bg.h5 (the file "
                        "hadronize.py --tags bulk_bg writes from the particlize file). "
                        "Default 0: no bulk hadrons in the job")
    p.add_argument("--keep-primaries", type=int, default=0, dest="keep_primaries", metavar="N",
                   help="with --bulk-hadrons: also write the undecayed bulk hadrons of the "
                        "first N oversamples (<stem>_hadrons_bulk_bg_primary.h5; full "
                        "precision, positions, no selection: for SMASH later). Default 0")
    p.add_argument("--store-hadronization-input", action="store_true",
                   dest="store_hadronization_input",
                   help="store each event's Pythia input of the jet hadronization (final "
                        "partons, beam remnants, colour-balancing parton) in the jet_frag "
                        "file (hadronization_input/), so rehadronize_jets.py can redo the "
                        "jet hadrons later (other settings, decays off, positions)")
    p.add_argument("--matter-only", action="store_true", dest="matter_only",
                   help="Matter alone (no <Eloss><Lbt>, Matter's recoil_on 0): the energy "
                        "loss before 2026-10-21, when LBT's partons had no colour. For an "
                        "X-SCAPE without the colour closure, and to extend Matter-only "
                        "campaigns (their jet hadrons are not combinable with Matter + LBT)")
    p.add_argument("--colour-closure", default=None, dest="colour_closure",
                   choices=("reconnect", "reconnect_global", "thermal", "fake", "none"),
                   help="iColored's <colour_closure> for this job (default: the XML's, "
                        "reconnect): how the colour lines that end in the medium are closed")
    p.add_argument("--eos", choices=("9", "91"), default=None,
                   help="MUSIC's EOS (default: the XML's, 91: hotQCD + the SMASH hadron "
                        "list, the tunes' EOS; 9: the UrQMD list, older campaigns)")
    p.add_argument("--hadronize-xml", default=os.path.join(rp.CONFIG, "hadronize.xml"),
                   dest="hadronize_xml",
                   help="iSS settings for --bulk-hadrons (default: "
                        "../config/hadronize.xml, those of hadronize.py)")
    p.add_argument("--keep-bits-p", type=int, default=None, dest="keep_bits_p", metavar="B",
                   help="in-job hadron files: round p to B float32 mantissa bits (1..23; "
                        "default: exact), as hadronize.py --keep-bits-p")
    p.add_argument("--keep-bits-x", type=int, default=None, dest="keep_bits_x", metavar="B",
                   help="in-job hadron files: round x to B mantissa bits (1..23)")
    p.add_argument("--eta-max", type=float, default=None, dest="eta_max",
                   help="in-job hadron files: keep only hadrons with |eta| < ETA_MAX (iSS "
                        "and Pythia still sample everything)")
    p.add_argument("--charged", action="store_true",
                   help="in-job hadron files: keep only charged hadrons")
    p.add_argument("--no-x", action="store_true", dest="no_x",
                   help="in-job hadron files: no positions t, x, y, z")
    p.add_argument("--no-subtraction", action="store_true", dest="no_subtraction",
                   help="REFERENCE run: 3dMCGlauber does not subtract the hard partons' "
                        "momentum (Subtract_hard_momentum 0), so the strings keep the full "
                        "Npart sqrt(s)/2 while the jets (hadronized as usual) also carry the "
                        "hard pair's energy: the energy is counted twice, on purpose, to test "
                        "the effect of the subtraction in small systems.  Same seed = same "
                        "hard events (Pythia), but 3dMCGlauber's geometry agrees only for "
                        "the first event (the subtraction draws from its random stream): "
                        "compare ensembles, not events.  Recorded as "
                        "subtract_hard_momentum = 0 in every file and the .json; keep these "
                        "jobs in their own OUTDIR (or --campaign nosub)")
    p.add_argument("--max-skipped", type=int, default=3, dest="max_skipped", metavar="N",
                   help="an event whose simulation throws (e.g. Matter's 'minus log of "
                        "sudakov negative', ~1 in 400 events) is skipped and recorded; more "
                        "than N skipped events stop the job (default 3)")
    p.add_argument("--dry-run", action="store_true", dest="dry_run",
                   help="check the XML and grid, write the job XML, print the plan; do not run")
    rp.add_seed_args(p, "3dMCGlauber, Pythia, iMATTER, Matter")
    rp.add_workdir_args(p)
    rp.add_h5_args(p)
    return p.parse_args()


def _text(node, path):
    n = node.find(path)
    return None if n is None or n.text is None else n.text.strip()


def _set(parent, tag, value):
    n = parent.find(tag)
    if n is None:
        n = ET.SubElement(parent, tag)
    n.text = f" {value} "
    return n


def job_xml(a, out_h5: str):
    """Check the template, apply this job's settings, write it next to the output."""
    tree = ET.parse(a.user_xml)
    root = tree.getroot()
    problems = []

    _set(root, "nEvents", a.events)
    root.find("Random/seed").text = str(a.seed)
    _set(root, "setReuseHydro", "false")     # every event has its own strings (after ISR)
    if a.matter_only and root.find("Eloss") is not None:
        eloss_node = root.find("Eloss")
        if eloss_node.find("Lbt") is not None:
            eloss_node.remove(eloss_node.find("Lbt"))
        if eloss_node.find("Matter") is not None:
            _set(eloss_node.find("Matter"), "recoil_on", 0)
    _set(root, "nReuseHydro", 1)

    tags = [c.tag for c in root]
    if "IS" not in tags or "Hard" not in tags or tags.index("IS") > tags.index("Hard"):
        problems.append("need <IS> before <Hard> (the task list follows the XML order)")
    mcg = root.find("IS/MCGlauber")
    if mcg is None:
        problems.append("need <IS><MCGlauber>")
    else:
        if _text(mcg, "generateOnlyPositions") != "1":
            problems.append("need <IS><MCGlauber><generateOnlyPositions>1 (ISR)")
        # 1: the hard partons' momentum comes out of the strings (energy-consistent);
        # 0 only as the --no-subtraction reference (energy counted twice, on purpose)
        _set(mcg, "Subtract_hard_momentum", 0 if a.no_subtraction else 1).text = \
            "0" if a.no_subtraction else "1"
        for tag, v in (("sqrts", a.sqrts), ("b_min", a.b_min), ("b_max", a.b_max)):
            if v is not None:
                _set(mcg, tag, f"{v:g}")
    pg = root.find("Hard/PythiaGun")
    if pg is None or root.find("Hard/ISR") is None:
        problems.append("need <Hard><PythiaGun> and <Hard><ISR>")
    else:
        if _text(pg, "name") != GUN_ID:
            problems.append(f"<Hard><PythiaGun><name> must be {GUN_ID}")
        for tag, v in (("pTHatMin", a.pthat_min), ("pTHatMax", a.pthat_max),
                       ("eCM", a.sqrts)):
            if v is not None:
                _set(pg, tag, f"{v:g}")
        if a.pthat_windows:
            _set(pg, "pTHatBins", rj.pthat_bins_text(a.pthat_windows))
        elif pg.find("pTHatBins") is not None:
            pg.remove(pg.find("pTHatBins"))      # a template's windows: only with --pthat-bins
        if a.parton_ymax is not None:
            _set(pg, "partonYMax", f"{a.parton_ymax:g}")
            _set(pg, "partonYMode", a.parton_y_mode)
        else:
            for tag in ("partonYMax", "partonYMode"):   # a template's cut: only with the option
                if pg.find(tag) is not None:
                    pg.remove(pg.find(tag))
        if len(list(root.find("Hard"))) != 2:
            problems.append("<Hard> takes only <PythiaGun> and <ISR>")
    if root.find("Preequilibrium/NullPreDynamics") is None:
        problems.append("need <Preequilibrium><NullPreDynamics> (MUSIC's string source)")
    elif _text(root, "Preequilibrium/evolutionInMemory") != "0":
        problems.append("<Preequilibrium><evolutionInMemory> must be 0")
    hydros = root.findall("Hydro")
    if [_text(h, "MUSIC/name") for h in hydros] != [HYDRO_ID]:
        problems.append(f"need exactly one <Hydro><MUSIC> named {HYDRO_ID}")
    else:
        m = hydros[0].find("MUSIC")
        if _text(m, "output_evolution_to_memory") != "1" or _text(m, "dump_hydro_only") != "0":
            problems.append("MUSIC needs <output_evolution_to_memory>1 and <dump_hydro_only>0: "
                            "Matter reads the framework medium")
        _set(m, "slim_bulk_info", 1 if a.bulk_info == "slim" else 0)
        if a.eos is not None:
            _set(m, "EOS", a.eos)
        if a.late_dtau is not None:
            _set(m, "late_delta_tau", a.late_dtau)
        if a.late_tau is not None:
            _set(m, "late_tau_switch", a.late_tau)
        surf = not a.no_particlize or a.bulk_hadrons > 0
        _set(m, "freeze_out_surface", int(surf))
        _set(m, "skip_surface", int(not surf))
        if _text(root, "Hydro/AddLiquefier") not in (None, "false"):
            problems.append("<Hydro><AddLiquefier> must be false (one MUSIC run, no deposition)")
    eloss = root.find("Eloss")
    if eloss is None:
        problems.append("need <Eloss>")
    # <Eloss><Lbt> needs LBT's colour tags and iColored's <colour_closure>
    # (PLAN_lbt_colour_tags.md); checked against the build in main()
    jh = root.find("JetHadronization")
    if a.no_jet_hadrons:
        if jh is not None:
            root.remove(jh)
    elif jh is None or _text(jh, "name") != "iColored":
        problems.append("need <JetHadronization><name>iColored")
    else:
        # every event reseeds Pythia from the job's seed and records the seed
        _set(jh, "reseed_per_event", 1)
        _set(jh, "store_input", int(a.store_hadronization_input))
        if a.colour_closure:
            _set(jh, "colour_closure", a.colour_closure)
    for tag in ("SoftParticlization", "Afterburner", "Liquefier", "RootBulkWriter",
                "FastRootBulkWriter"):
        if root.find(tag) is not None:
            problems.append(f"remove <{tag}>: iSS runs offline on the stored surface "
                            "(hadronize.py), and there is no deposition")
    if problems:
        sys.exit(f"run_prod_oo_isr.py: {a.user_xml}:\n  " + "\n  ".join(problems))
    if a.bulk_hadrons > 0:
        add_inline_iss(root, a.hadronize_xml, a.bulk_hadrons, a.keep_primaries)

    path = os.path.splitext(out_h5)[0] + ".xml"
    tree.write(path)
    return path, root


def add_inline_iss(root, hadronize_xml, n_samples, keep_primaries=0):
    """--bulk-hadrons: hadronize.xml's <SoftParticlization> (the settings hadronize.py
    runs), sampling MUSIC's surface in the job's own working directory.  It goes last, so
    iSS runs after MUSIC; '../' paths are made absolute with the job XML's."""
    soft = ET.parse(hadronize_xml).getroot().find("SoftParticlization")
    if soft is None or soft.find("iSS") is None:
        sys.exit(f"run_prod_oo_isr.py: {hadronize_xml} has no <SoftParticlization><iSS>")
    soft = copy.deepcopy(soft)
    _set(soft, "hydro_id", HYDRO_ID).text = HYDRO_ID     # compared verbatim with the id
    iss = soft.find("iSS")
    _set(iss, "iSS_working_path", ".").text = "."
    _set(iss, "number_of_repeated_sampling", int(n_samples))
    _set(iss, "keep_primaries", int(keep_primaries))
    root.append(soft)


def jet_hadronization_settings(root, main_xml) -> str:
    """The <JetHadronization> block the job runs: the main XML's with the job XML's tags
    on top (what iColoredHadronization reads), for rehadronize_jets.py."""
    main = ET.parse(main_xml).getroot().find("JetHadronization")
    eff = copy.deepcopy(main) if main is not None else ET.Element("JetHadronization")
    for child in root.find("JetHadronization"):
        old = eff.find(child.tag)
        if old is not None:
            eff.remove(old)
        eff.append(copy.deepcopy(child))
    return ET.tostring(eff, encoding="unicode")


def hadron_selection(a) -> dict:
    """HadronH5Writer keywords for the in-job hadron files (hadronize.py's options)."""
    for k in ("keep_bits_p", "keep_bits_x"):
        v = getattr(a, k)
        if v is not None and not 1 <= v <= 23:
            sys.exit(f"run_prod_oo_isr.py: --{k.replace('_', '-')} must be 1..23")
    if a.eta_max is not None and not a.eta_max > 0:
        sys.exit("run_prod_oo_isr.py: --eta-max must be > 0")
    keep = {"p": a.keep_bits_p, "x": a.keep_bits_x}
    return {"keep_bits": keep if any(v is not None for v in keep.values()) else None,
            "eta_max": a.eta_max, "charged_only": a.charged, "positions": not a.no_x}


def strings_record() -> dict:
    """This event's strings as MUSIC takes them: their energy and pz [GeV], strings plus
    remnant ends, MUSIC's sum (hydro_source_strings.cpp: m [cosh y_i - cosh y] per end
    plus remnant m cosh y), from 3dMCGlauber's strings file.  (Its header's total_energy
    is Npart sqrt(s)/2 by construction, before any subtraction.)"""
    import numpy as np
    try:
        with open(GLAUBER_STRINGS) as fh:
            fh.readline()
            cols = fh.readline().lstrip("#").split()
            rows = np.loadtxt(fh, ndmin=2)
        # exact header tokens: rapidities have no unit ("y_l"), positions do ("y_l[fm]")
        idx = {c: k for k, c in enumerate(cols)}
        m, yl, yr = rows[:, idx["mass[GeV]"]], rows[:, idx["y_l"]], rows[:, idx["y_r"]]
        yli, yri = rows[:, idx["y_l_i"]], rows[:, idx["y_r_i"]]
        rl, rr = rows[:, idx["remnant_l"]], rows[:, idx["remnant_r"]]
        e = m * (np.cosh(yli) + np.cosh(yri) - np.cosh(yl) - np.cosh(yr)) \
            + rl * m * np.cosh(yl) + rr * m * np.cosh(yr)
        pz = m * (np.sinh(yli) + np.sinh(yri) - np.sinh(yl) - np.sinh(yr)) \
            + rl * m * np.sinh(yl) + rr * m * np.sinh(yr)
        return {"strings_E_total": float(e.sum()), "strings_pz_net": float(pz.sum())}
    except (OSError, ValueError, KeyError, IndexError):
        return {}


def isr_record(core, ini) -> dict:
    """What the hard process took from the colliding nucleons this event: the hard-parton
    four-momentum handed over for subtraction (projectile +z, target -z), and the four-
    momenta of the two nucleons' remnants (to the jet hadronization)."""
    if ini is None or not hasattr(core, "isr_hard_momentum"):
        return {}
    r = core.isr_hard_momentum(ini)
    out = {}
    for side in ("proj", "targ"):
        for name, v in zip(("E", "px", "py", "pz"), r[side] or []):
            out[f"isr_sub_{name}_{side}"] = float(v)
        rem = r[f"remnant_{side}"] or []
        if len(rem) == 4:
            out[f"remnant_E_{side}"], out[f"remnant_pz_{side}"] = float(rem[0]), float(rem[3])
    if "isr_sub_E_proj" in out and "isr_sub_E_targ" in out:
        out["isr_sub_E_total"] = out["isr_sub_E_proj"] + out["isr_sub_E_targ"]
    return out


def hard_partons_record(core, gun) -> dict:
    """The two hardest partons the hard process handed over (by pT; before the ISR the
    outgoing pair has equal pT, so 'lead' and 'sub' are just the two): y and pT."""
    if not hasattr(core, "hard_partons_numpy"):
        return {}
    hp = core.hard_partons_numpy(gun)        # pid, px, py, pz, E, pT, y; hardest first
    out = {}
    if len(hp):
        out.update(parton_y_lead=float(hp[0, 6]), parton_pt_lead=float(hp[0, 5]))
    if len(hp) > 1:
        out.update(parton_y_sub=float(hp[1, 6]), parton_pt_sub=float(hp[1, 5]))
    return out


def glauber_record() -> dict:
    """This event's 3dMCGlauber numbers from events_summary.dat in the working directory."""
    try:
        with open(GLAUBER_SUMMARY) as fh:
            rows = [ln.split() for ln in fh if ln.strip() and not ln.startswith("#")]
        _, npart, ncoll, nstrings, b = rows[-1][:5]
        return {"glauber_npart": int(npart), "glauber_ncoll": int(ncoll),
                "glauber_nstrings": int(nstrings), "glauber_b": float(b)}
    except (OSError, IndexError, ValueError):
        return {}


def main() -> int:
    a = parse_args()
    for k in ("build", "outdir", "main_xml", "user_xml", "grid"):
        setattr(a, k, os.path.abspath(getattr(a, k)))
    rp.check_env(a)
    for k in ("events",):
        if getattr(a, k) < 1:
            sys.exit(f"run_prod_oo_isr.py: --{k} must be >= 1")

    a.pthat_windows = None
    if a.pthat_bins is not None:
        if a.pthat_min is not None or a.pthat_max is not None:
            sys.exit("run_prod_oo_isr.py: --pthat-bins replaces --pthat-min/--pthat-max")
        try:
            a.pthat_windows = rj.parse_pthat_bins(a.pthat_bins)
        except ValueError as exc:
            sys.exit(f"run_prod_oo_isr.py: {exc}")
        if any(lo <= 0 for lo, _ in a.pthat_windows):
            sys.exit("run_prod_oo_isr.py: --pthat-bins windows need MIN > 0 (PythiaIsrGun "
                     "runs hard QCD only)")
    if a.parton_ymax is not None and not a.parton_ymax > 0:
        sys.exit(f"run_prod_oo_isr.py: --parton-ymax must be > 0, got {a.parton_ymax:g}")

    grid, max_ntau, grid_text = rp.load_grid_yaml(a.grid)
    os.makedirs(a.outdir, exist_ok=True)
    out_h5 = os.path.join(a.outdir, rp.output_name(a, NAME_BASE))
    rp.resolve_seed(a, out_h5, record=not a.dry_run)
    xml, root = job_xml(a, out_h5)
    box = rp.music_box(root, a.main_xml)
    rp.check_inside(grid, box)
    stem = os.path.splitext(out_h5)[0]
    out_particlize = None if a.no_particlize else stem + "_particlize.h5"
    out_frag = None if a.no_jet_hadrons else stem + "_hadrons_jet_frag.h5"
    out_bulk = stem + "_hadrons_bulk_bg.h5" if a.bulk_hadrons > 0 else None
    if a.pthat_windows and a.events % len(a.pthat_windows):
        print(f"run_prod_oo_isr.py: note -- {a.events} events in {len(a.pthat_windows)} "
              "windows: not the same number per window", file=sys.stderr)
    selection = hadron_selection(a)
    if a.bulk_hadrons < 0:
        sys.exit("run_prod_oo_isr.py: --bulk-hadrons must be >= 0")
    if a.keep_primaries < 0 or (a.keep_primaries and not a.bulk_hadrons):
        sys.exit("run_prod_oo_isr.py: --keep-primaries N needs N >= 0 and --bulk-hadrons")
    if a.store_hadronization_input and a.no_jet_hadrons:
        sys.exit("run_prod_oo_isr.py: --store-hadronization-input needs the jet "
                 "hadronization (drop --no-jet-hadrons)")
    out_bulk_primary = (stem + "_hadrons_bulk_bg_primary.h5"
                        if out_bulk and a.keep_primaries else None)
    a.hadronize_xml = os.path.abspath(a.hadronize_xml)
    eos = (_text(root, "Hydro/MUSIC/EOS") or "").strip()

    mcg = root.find("IS/MCGlauber")
    system = (f"{_text(mcg, 'projectile')}+{_text(mcg, 'target')} "
              f"{float(_text(mcg, 'sqrts')):g} GeV, b in [{_text(mcg, 'b_min')}, "
              f"{_text(mcg, 'b_max')}] fm")
    if a.pthat_windows:
        pthat_desc = ("pTHat windows " + ", ".join(f"{lo:g}-{hi:g}" for lo, hi in a.pthat_windows)
                      + " GeV (event i in window i mod "
                      + f"{len(a.pthat_windows)}, every event complete)")
    else:
        pthat_desc = (f"pTHat {_text(root, 'Hard/PythiaGun/pTHatMin')}-"
                      f"{_text(root, 'Hard/PythiaGun/pTHatMax')} GeV")
    if a.parton_ymax is not None:
        pthat_desc += f", hard partons |y| < {a.parton_ymax:g} ({a.parton_y_mode})"
    hard = (f"PythiaIsrGun {pthat_desc}, ISR (iMATTER) to "
            f"-{_text(root, 'Eloss/maxT')} fm/c")
    # the final-state energy loss, as the job XML has it
    recoil = (_text(root, "Eloss/Matter/recoil_on") or "0").strip()
    eloss = ("Matter + LBT" if root.find("Eloss/Lbt") is not None else "Matter") + \
        f" (Matter recoil_on {recoil})"
    print(f"prod_OO_isr: {a.events} event(s), seed {a.seed}"
          + (" (from OS entropy)" if a.seed_source == "os_entropy" else "")
          + (f", campaign {a.campaign} #{a.index}" if a.campaign else "")
          + f", {system}, {hard}, {eloss} in MUSIC, "
          + ("no jet hadronization" if a.no_jet_hadrons else "iColored jet hadronization")
          + (f", iSS in the job ({a.bulk_hadrons} oversamples)" if out_bulk else "")
          + (", NO SUBTRACTION (reference)" if a.no_subtraction else ""))
    print(f"  build    {a.build}")
    print(f"  job XML  {xml}")
    print(f"  grid     {a.grid}")
    print(rp.describe(grid, max_ntau))
    for path in (None if a.no_hydro_file else out_h5, out_particlize, out_bulk,
                 out_bulk_primary, out_frag):
        if path:
            print(f"  output   {path}")
    if a.dry_run:
        return 0

    import numpy as np
    import jetscape as js
    from jetscape import pyjetscape_core as core
    soft = root.find("SoftParticlization/iSS")
    smash_decays = (soft is not None and (_text(soft, "decay_smash_list") or "0") == "1"
                    and getattr(core, "ISS_DECAYS_SMASH_LIST", False))
    if out_bulk and eos == "91" and not smash_decays:
        # iSS (readindata.cpp) takes the SMASH hadron list for EOS 91; an iSS before
        # smash_ready (or <decay_smash_list> 0) leaves its decays to SMASH
        print("run_prod_oo_isr.py: WARNING -- MUSIC runs EOS 91 (SMASH hadron list) and "
              "this iSS does NOT decay that list (core.ISS_DECAYS_SMASH_LIST False, or "
              "<decay_smash_list> 0): rho, omega, K*, Delta ... stay in the bulk hadrons"
              + ("; with --no-particlize the surfaces are not kept to redo it."
                 if a.no_particlize else "."), file=sys.stderr)
    if (root.find("Eloss/Lbt") is not None and not a.no_jet_hadrons
            and not getattr(core, "ICOLORED_HAS_COLOUR_CLOSURE", False)):
        sys.exit("run_prod_oo_isr.py: <Eloss><Lbt> (the default) needs X-SCAPE with LBT's "
                 "colour tags and iColoredHadronization's <colour_closure> "
                 "(lbt_colored_hadro), and PyJetscape built against it; with an older build "
                 "use --matter-only (Matter alone, as before 2026-10-21) or "
                 "--no-jet-hadrons")
    if not a.no_jet_hadrons and not getattr(core, "ICOLORED_HAS_INPUT_RECORD", False):
        if a.store_hadronization_input:
            sys.exit("run_prod_oo_isr.py: --store-hadronization-input needs X-SCAPE's "
                     "iColoredHadronization with <store_input> (smash_ready) and PyJetscape "
                     "built against it")
        print("run_prod_oo_isr.py: note -- this X-SCAPE's iColoredHadronization cannot "
              "reseed per event: the jet hadrons run from Pythia's default seed",
              file=sys.stderr)
    if a.keep_primaries and not getattr(core, "SOFT_HAS_PRIMARIES", False):
        sys.exit("run_prod_oo_isr.py: --keep-primaries needs X-SCAPE with iSS "
                 "<keep_primaries> (smash_ready) and PyJetscape built against it")
    hadron_list = {"9": "UrQMD", "91": "SMASH"}.get(eos, "other")
    iss_decays = ("none" if not out_bulk else
                  "iSS, Breit-Wigner masses for the unstable daughters of channels closed "
                  "at the pole masses; eta, Sigma0 stable" if hadron_list == "SMASH"
                  and smash_decays else "none" if hadron_list == "SMASH"
                  else "iSS, pole masses")
    from jetscape.fast_h5_bulk import H5BulkWriter
    from jetscape.showers import initiators_from_manager

    provenance = {
        "prod": "PyJetscape/example/prod_OO_isr",
        **rp.seed_provenance(a),
        "prod_host": socket.gethostname(),
        "prod_platform": platform.platform(),
        "prod_build": a.build,
        "prod_user_xml": open(xml).read(),
        "prod_grid_yaml": grid_text,
        "prod_hard": hard,
        "prod_bulk_info": a.bulk_info,
        "system": system,
        "initial_state_kind": "3dMCGlauber_strings_after_ISR",
        "hydro": rp.hydro_label(a.build),
        "jet": f"PythiaIsrGun + iMATTER (ISR); {eloss} in MUSIC's medium (FSR)",
        "eloss": eloss,
        "jet_hadronization": "none" if a.no_jet_hadrons else
                             "iColoredHadronization, in the job (<stem>_hadrons_jet_frag.h5)",
        "bulk_hadronization": f"iSS in the job, {a.bulk_hadrons} oversamples "
                              "(<stem>_hadrons_bulk_bg.h5)" if out_bulk else
                              "none in the job (hadronize.py on the particlize file)",
        "T_fo": float(_text(root, "Hydro/MUSIC/freezeout_temperature") or 0.15),
        "eos": eos,
        "subtract_hard_momentum": 0 if a.no_subtraction else 1,
        **({"parton_ymax": a.parton_ymax, "parton_y_mode": a.parton_y_mode}
           if a.parton_ymax is not None else {}),
    }

    # MUSIC / 3dMCGlauber resolve their input files relative to the working directory:
    # a private one per job (rp.enter_workdir), so concurrent jobs share no file.
    if a.in_build:
        os.chdir(a.build)
        main_xml, workdir = a.main_xml, a.build
    else:
        workdir = rp.job_workdir(a, out_h5)
        main_xml = rp.enter_workdir(a, workdir, xml)
    print(f"  workdir  {workdir}")
    jetscape = js.JetScapePerEvent()
    jetscape.SetXMLMainFileName(main_xml)
    jetscape.SetXMLUserFileName(xml)
    jetscape.Init()
    tasks = {t.GetId(): t for t in jetscape.GetTaskList()}
    for tid in ((HYDRO_ID, GUN_ID) + (() if a.no_jet_hadrons else ("HadronizationManager",))
                + (("iSS",) if out_bulk else ())):
        if tid not in tasks:
            sys.exit(f"run_prod_oo_isr.py: no task {tid!r} (tasks: {sorted(tasks)}); is "
                     "this X-SCAPE built with <Hard><ISR> (branch oo_isr_test)?")
    hydro, gun = tasks[HYDRO_ID], tasks[GUN_ID]
    if "IsrManager" not in {t.GetId() for t in gun.GetTaskList()}:
        sys.exit("run_prod_oo_isr.py: the hard process has no IsrManager: this X-SCAPE "
                 "does not build the ISR chain from <Hard><ISR>")
    mgr = core.JetScapeSignalManager.Instance().GetJetEnergyLossManagerPointer()
    ini = tasks.get(IS_ID)
    if ini is not None and hasattr(core, "isr_hard_momentum"):
        sub = core.isr_hard_momentum(ini)["subtract"]
        want = 0 if a.no_subtraction else 1
        if sub != want:
            sys.exit(f"run_prod_oo_isr.py: {IS_ID} runs Subtract_hard_momentum = {sub}, "
                     f"not {want}")
        if a.no_subtraction:
            print("  ISR      REFERENCE (--no-subtraction): the hard partons' momentum is NOT "
                  "subtracted, the energy is counted twice on purpose; recorded per event "
                  "(isr_sub_*: what would have been subtracted)", file=sys.stderr)
        else:
            print("  ISR      the hard partons' momentum is subtracted from the colliding "
                  "nucleons; recorded per event (isr_sub_*, remnant_*, strings_*)")
    else:
        print("  ISR      this build cannot report the subtracted momentum "
              "(PyJetscape without isr_hard_momentum)", file=sys.stderr)
    windows = None
    if getattr(core, "PYTHIA_ISR_GUN_HAS_PTHAT_BINS", False):
        windows = core.pythia_gun_bins(gun)
        got = [tuple(w) for w in windows["bins"]]
        want = [tuple(w) for w in a.pthat_windows] if a.pthat_windows else got[:1]
        if got != want:
            sys.exit(f"run_prod_oo_isr.py: PythiaIsrGun runs the windows {got}, not {want}")
        if a.pthat_windows:
            print("  pTHat    " + ", ".join(
                f"window {k}: {lo:g}-{hi:g} GeV (Pythia seed {s})"
                for k, ((lo, hi), s) in enumerate(zip(got, windows["seeds"]))))
    elif a.pthat_windows:
        sys.exit("run_prod_oo_isr.py: --pthat-bins needs X-SCAPE's PythiaIsrGun with "
                 "<pTHatBins> (branch oo_isr_test) and PyJetscape built against it")
    if a.parton_ymax is not None:
        if not getattr(core, "PYTHIA_ISR_GUN_HAS_PARTON_Y_CUT", False):
            sys.exit("run_prod_oo_isr.py: --parton-ymax needs X-SCAPE's PythiaIsrGun with "
                     "<partonYMax> (branch oo_hard_sub_compare) and PyJetscape built "
                     "against it")
        if (abs(windows["y_max"] - a.parton_ymax) > 1e-12
                or windows["y_mode"] != a.parton_y_mode):
            sys.exit(f"run_prod_oo_isr.py: PythiaIsrGun cuts |y| < {windows['y_max']:g} "
                     f"({windows.get('y_mode')}), not {a.parton_ymax:g} ({a.parton_y_mode})")
        print(f"  cut      outgoing hard partons, |y| < {a.parton_ymax:g} ({a.parton_y_mode})")

    writer = None
    if not a.no_hydro_file:
        writer = H5BulkWriter(out_h5, grid_mode="framework", out_grid=grid,
                              choose_ntau=max_ntau, compression=a.compression,
                              keep_bits=a.keep_bits, extra_attrs=provenance, verbose=True)
        writer.Init()
    pwriter = None
    if out_particlize:
        from jetscape.particlize_h5 import ParticlizeH5Writer
        pwriter = ParticlizeH5Writer(
            out_particlize, legs=("bg",), bg_id=HYDRO_ID, T_fo=provenance["T_fo"],
            pair_file=None if a.no_hydro_file else out_h5,
            extra_attrs={**{k: v for k, v in provenance.items() if k != "T_fo"},
                         "jet_frag_inline": "" if a.no_jet_hadrons
                         else os.path.basename(out_frag),
                         "bulk_bg_inline": os.path.basename(out_bulk) if out_bulk else ""},
            verbose=True)
        pwriter.attach(jetscape, bg=hydro, manager=mgr)   # reads ./music_input: the job's
    from jetscape.hadrons_h5 import HadronH5Writer
    bwriter = pbwriter = iss = None
    if out_bulk:
        iss = tasks["iSS"]
        core.soft_set_compact_output(iss, True)   # arrays, ~8x less memory per oversample
        bwriter = HadronH5Writer(
            out_bulk, tag="bulk_bg", n_samples=a.bulk_hadrons, **selection,
            attrs={"inline": True, "source": os.path.basename(stem) + ".h5",
                   "generator": "iSS (X-SCAPE iSpectraSamplerWrapper), in the job",
                   "hadronize_xml": ET.tostring(root.find("SoftParticlization"),
                                                encoding="unicode"),
                   "music_input": open("music_input").read(),
                   "hadron_list": hadron_list, "iss_decays": iss_decays,
                   "decayed": iss_decays != "none",
                   **{k: v for k, v in provenance.items()
                      if k in ("prod", "prod_seed", "system", "prod_hard",
                                 "subtract_hard_momentum", "eos", "eloss")}})
        if out_bulk_primary:
            # full precision, positions, no selection: what SMASH needs
            pbwriter = HadronH5Writer(
                out_bulk_primary, tag="bulk_bg_primary",
                n_samples=min(a.keep_primaries, a.bulk_hadrons),
                attrs={**{k: v for k, v in bwriter.f.attrs.items()
                          if k not in ("format", "format_version", "producer", "tag",
                                       "n_samples", "complete", "nunits_written",
                                       "eta_max", "charged_only", "positions",
                                       "p_columns", "x_columns", "units")},
                       "decayed": False, "primaries_of": "bulk_bg",
                       "keep_primaries": int(a.keep_primaries)})
    hwriter = None
    if out_frag:
        hwriter = HadronH5Writer(out_frag, tag="jet_frag", n_samples=1, initiators=True,
                                 hadronization_input=a.store_hadronization_input,
                                 **selection,
                                 attrs={"inline": True, "module": "iColoredHadronization",
                                        "source": os.path.basename(stem) + ".h5",
                                        "pythia_version": int(getattr(core, "PYTHIA_VERSION",
                                                                      0)),
                                        "reseed_per_event": int(getattr(
                                            core, "ICOLORED_HAS_INPUT_RECORD", False)),
                                        "hadronization_input_stored":
                                            bool(a.store_hadronization_input),
                                        "jet_hadronization_xml":
                                            jet_hadronization_settings(root, a.main_xml),
                                        "prod_user_xml": provenance["prod_user_xml"],
                                        **{k: v for k, v in provenance.items()
                                           if k in ("prod", "prod_seed", "system",
                                                    "prod_hard", "subtract_hard_momentum",
                                                    "eos", "eloss")}})

    n = n_hydro = no_medium = 0
    colour_totals = None          # iColored's colour audits added up (the log summary)
    skipped = []                  # (event, error) of events whose simulation threw
    t_job = time.time()
    try:
        jetscape.ExecInit()
        for i in range(jetscape.GetNumberOfEvents()):
            t0 = time.time()
            try:
                jetscape.ExecPerEvent()
            except RuntimeError as exc:
                # A module threw inside the event (C++ exception, e.g. Matter's "minus log
                # of sudakov negative").  The same seed hits it again on a rerun, so skip
                # this event rather than fail the job; nothing of it is written.
                skipped.append((i, str(exc)))
                print(f"prod_OO_isr: event {i + 1}/{a.events} SKIPPED: {exc}",
                      file=sys.stderr)
                try:
                    jetscape.ClearPerEvent()
                except RuntimeError:
                    pass
                if len(skipped) > a.max_skipped:
                    raise RuntimeError(f"{len(skipped)} events skipped (more than "
                                       f"--max-skipped {a.max_skipped}): {skipped}") from exc
                continue
            k = n                         # this event's row in every output file
            h = core.hard_process_info(gun)
            ev = {"pthat": float(h["pthat"]), "sigma_gen": float(h["sigma_gen"]),
                  "sigma_err": float(h["sigma_err"]),
                  "event_weight": float(h["event_weight"]), **glauber_record(),
                  **strings_record(), **isr_record(core, ini),
                  **hard_partons_record(core, gun)}
            if a.pthat_windows:
                ev["pthat_bin"] = int(core.pythia_gun_bins(gun)["active"])
            if not a.no_jet_hadrons and getattr(core, "ICOLORED_HAS_INPUT_RECORD", False):
                ev["jet_pythia_seed"] = int(core.jet_hadronization_last_random_seed(
                    tasks["HadronizationManager"]))
            b = hydro.get_bulk_info()
            medium = bool(b.get_data_size())
            ev["tau0_music"] = float(b.tau_min) if medium else float("nan")
            # explicit: hydro_row is also -1 for every event with --no-hydro-file
            ev["has_medium"] = int(medium)
            row = -1
            if writer is not None and medium:
                before = writer.GetNumberOfEventsWritten()
                writer.Exec()
                if writer.GetNumberOfEventsWritten() > before:
                    row = before
                    writer.write_diag(event=i, **ev)
            if not medium:
                no_medium += 1
            bulk_n = 0
            if bwriter is not None:
                n_cells = int(len(hydro.surface_to_numpy()))
                bulk = core.soft_hadrons_numpy(iss) if n_cells else []
                prim = core.soft_primaries_numpy(iss) if n_cells and pbwriter else None
                iss_seed = int(core.soft_last_random_seed(iss))
                # the event's record (as diag/) in units/ too: hadron-only jobs keep it
                bwriter.append_unit(bulk, unit=k, event=i, seed=iss_seed, n_cells=n_cells,
                                    hydro_row=row, **ev)
                if pbwriter is not None:
                    pbwriter.append_unit(prim if prim is not None else [], unit=k, event=i,
                                         seed=iss_seed, n_cells=n_cells, hydro_row=row, **ev)
                bulk_n = len(bulk["pid"]) if n_cells else 0
                ev["inline_iss_seed_bg"] = iss_seed
            if pwriter is not None:
                pwriter.Exec(k, bg_id=k, hydro_row=row, job_event=i, **ev)
            frag_n = 0
            if hwriter is not None:
                frag = core.hadronization_hadrons_numpy(tasks["HadronizationManager"])
                frag_n = len(frag["pid"])
                hin = (core.hadronization_input_numpy(tasks["HadronizationManager"])
                       if a.store_hadronization_input else None)
                # iColored's colour audit of the event (PLAN_lbt_colour_tags.md, C3)
                audit = ({f"colour_{key}": val for key, val in
                          core.jet_hadronization_colour_audit(
                              tasks["HadronizationManager"]).items()}
                         if getattr(core, "ICOLORED_HAS_COLOUR_AUDIT", False) else {})
                hwriter.append_unit([frag], initiators=initiators_from_manager(mgr),
                                    hadronization_input=hin, unit=k, event=i, hydro_row=row,
                                    seed=ev.get("jet_pythia_seed", 0), **ev, **audit)
            jetscape.ClearPerEvent()
            wall = time.time() - t0
            if pwriter is not None:
                pwriter.write_events(k, wall_s=wall)
            n += 1
            n_hydro += row >= 0
            print(f"prod_OO_isr: event {i + 1}/{a.events} done in {wall:.1f} s, pTHat "
                  f"{ev['pthat']:.1f} GeV"
                  + (f" (window {ev['pthat_bin']})" if "pthat_bin" in ev else "") + ", "
                  f"Npart {ev.get('glauber_npart', '?')}, Ncoll "
                  f"{ev.get('glauber_ncoll', '?')}, {ev.get('glauber_nstrings', '?')} strings"
                  + (f", MUSIC tau0 = {ev['tau0_music']:.3f} fm/c" if medium
                     else ", NO MEDIUM (no string)")
                  + (f", hydro row {row}" if writer is not None else "")
                  + (f", {bulk_n} bulk hadrons ({a.bulk_hadrons} samples)"
                     if bwriter is not None else "")
                  + (f", {frag_n} jet hadrons" if hwriter is not None else ""))
        sigma = core.hard_process_info(gun)
        if windows is not None:
            windows = core.pythia_gun_bins(gun)          # the final cross sections
        if not a.no_jet_hadrons and getattr(core, "ICOLORED_HAS_COLOUR_CLOSURE", False):
            colour_totals = core.jet_hadronization_colour_totals(tasks["HadronizationManager"])
        jetscape.Finish()
    finally:
        if writer is not None:
            writer.Finish()
        if pwriter is not None:
            pwriter.Finish(complete=pwriter.GetNumberOfEventsWritten() + len(skipped)
                           == a.events)
        if hwriter is not None:
            hwriter.close()
        if bwriter is not None:
            bwriter.close()
        if pbwriter is not None:
            pbwriter.close()

    final = {"sigma_gen_mb": float(sigma["sigma_gen"]), "sigma_err_mb": float(sigma["sigma_err"])}
    if a.pthat_windows:
        # per window; sigma_gen_mb above is then the last event's window only
        final = {"pthat_bins": np.asarray(windows["bins"], dtype=np.float64),
                 "pthat_bin_sigma_gen": np.asarray(windows["sigma_gen"], dtype=np.float64),
                 "pthat_bin_sigma_err": np.asarray(windows["sigma_err"], dtype=np.float64),
                 "pthat_bin_n_accepted": np.asarray(windows["n_accepted"], dtype=np.int64),
                 "pthat_bin_seeds": np.asarray(windows["seeds"], dtype=np.int64)}
    if a.parton_ymax is not None:
        # per window; sigma_gen is Pythia's x kept/tried, the raw values count the rejected
        # events
        final.update(
            parton_ymax=a.parton_ymax, parton_y_mode=a.parton_y_mode,
            pthat_bin_n_tried=np.asarray(windows["n_tried"], dtype=np.int64),
            pthat_bin_n_kept=np.asarray(windows["n_kept"], dtype=np.int64),
            pthat_bin_acceptance=np.asarray(windows["acceptance"], dtype=np.float64),
            pthat_bin_sigma_gen_raw=np.asarray(windows["sigma_gen_raw"], dtype=np.float64),
            pthat_bin_sigma_err_raw=np.asarray(windows["sigma_err_raw"], dtype=np.float64))
    import h5py
    for path in (None if writer is None or not n_hydro else out_h5, out_particlize, out_frag,
                 out_bulk, out_bulk_primary):
        if path and os.path.exists(path):
            with h5py.File(path, "a") as f:
                f.attrs.update({**final, "prod_wall_s_total": time.time() - t_job,
                                "events_skipped": len(skipped)})
    summary = {"out": None if writer is None else out_h5, "seed": a.seed, "eos": eos,
               "seed_source": a.seed_source, "campaign": a.campaign, "index": a.index,
               "grid": a.grid, "system": system, "hard": hard, "bulk_info": a.bulk_info,
               "subtract_hard_momentum": 0 if a.no_subtraction else 1,
               "events_requested": a.events, "events_written": n,
               # run_jobs.sh counts a job complete when written + skipped == requested
               "events_skipped": len(skipped),
               "skipped": [{"event": e, "error": m} for e, m in skipped],
               "hydro_events_written": n_hydro, "events_without_medium": no_medium,
               **{k: v.tolist() if hasattr(v, "tolist") else v for k, v in final.items()},
               "wall_s": round(time.time() - t_job, 1)}
    if pwriter is not None:
        summary.update(particlize=out_particlize,
                       particlize_events_written=pwriter.GetNumberOfEventsWritten())
    if out_frag:
        summary["jet_frag"] = out_frag
    if colour_totals is not None:
        summary["jet_colour_totals"] = dict(colour_totals)
    if out_bulk:
        summary.update(bulk_bg=out_bulk, bulk_oversample=a.bulk_hadrons)
    if out_bulk_primary:
        summary.update(bulk_bg_primary=out_bulk_primary, keep_primaries=a.keep_primaries)
    with open(stem + ".json", "w") as f:
        json.dump(summary, f, indent=1)
    print("prod_OO_isr:", json.dumps(summary))
    done = n + len(skipped) == a.events
    rp.leave_workdir(a, workdir, done)
    return 0 if done else 1


if __name__ == "__main__":
    raise SystemExit(main())
