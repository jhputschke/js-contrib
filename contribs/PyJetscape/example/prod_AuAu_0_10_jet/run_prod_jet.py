#!/usr/bin/env python3
"""
example/prod_AuAu_0_10_jet/run_prod_jet.py

prod_AuAu_0_10 with a jet: 0-10% Au+Au 200 GeV, 3D MC-Glauber strings -> the X-SCAPE
two-stage hydro (MUSIC_1 background -> Matter + LBT + CausalLiquefier -> MUSIC_2 with the
jet's deposition), written as background/jet PAIRS straight to FNO4d-schema HDF5 in
FastHydro's layout (jetscape.pair_h5.PairH5Writer): arr = jet leg, arr_bg = background,
source/droplets, shower/, diag/.

One call = one job = one seed = one .h5 file, as in ../prod_AuAu_0_10/run_prod.py, whose
grid YAMLs, grid checks and environment checks this reuses.

    conda activate js_fno
    python run_prod_jet.py --events 10 --seed 1                  # PythiaGun, deposition on
    python run_prod_jet.py --events 1 --seed 1 --no-deposit      # null test: arr == arr_bg
    python run_prod_jet.py --events 10 --seed 1 --hard pgun --pgun-pt 60
    python run_prod_jet.py --events 30 --seed 1 --reuse 3        # one background per 3 jets
    python run_prod_jet.py --events 30 --seed 1 --pthat-bins 20-40,50-70,70-90
                                     # one background per 3 jets, one in each pTHat window
    python run_prod_jet.py --events 30 --seed 1 --pthat-bins 10-20,20-30,30-40 --parton-ymax 0.6
                                     # only events whose hardest parton has |y| < 0.6
    python run_prod_jet.py --events 10 --seed 1 --surface jet    # surface for the jet leg only
    python run_prod_jet.py --events 10 --seed 1 --write-particlize both
                                     # + <stem>_particlize.h5: both surfaces and the final
                                     #   partons, for hadronize.py (PLAN_particlize_h5.md)
    python run_prod_jet.py --events 1 --seed 1 --dry-run         # check, print the plan
    python run_prod_jet.py --events 10 --seed 0 --campaign test  # unique seed from OS entropy,
                                                                 # -> out/AuAu_0_10_jet_test_0001.h5
    ./run_jobs.sh 20 25 0                                        # 20 jobs x 25 events, unique seeds
    ./run_jobs.sh 20 25 1                                        # the same with seeds 1..20

Needs a MUSIC build with the jet source slot (MUSIC cee9460 + X-SCAPE PR #138 on the CPU,
the matching MUSIC4GPU port on the GPU).  Without it MUSIC_2 silently ignores the droplets;
PairH5Writer then warns that the jet leg is bit-identical to the background.
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

# The single-leg production's helpers (grid YAML, MUSIC box, env checks).  Loading it also
# puts PyJetscape/python on sys.path.
_spec = importlib.util.spec_from_file_location("_run_prod", os.path.join(PROD, "run_prod.py"))
rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp)

USER_XML = os.path.join(HERE, "AuAu_MCGlauber_MUSIC_0_10_jet.xml")
GRID_YAML = os.path.join(PROD, "grid_fno.yaml")
BG_ID, JET_ID = "MUSIC_1", "MUSIC_2"

HARD_VERTEX = {
    "pythia": "3dMCGlauber binary-collision points (PythiaGun uses x, y; z = t = 0)",
    "pgun": "origin: PGun samples a vertex and zeroes it (PGun.cc:117-120)",
}


def parse_args() -> argparse.Namespace:
    # no abbreviations: run_jobs.sh must recognise --campaign and the options it sets per job
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--events", type=int, default=10, help="events in this job (default 10)")
    p.add_argument("--grid", default=GRID_YAML,
                   help="output grid YAML (default: ../prod_AuAu_0_10/grid_fno.yaml)")
    p.add_argument("--outdir", default=os.path.join(HERE, "out"),
                   help="output directory (default: ./out next to this script)")
    p.add_argument("--out", default=None,
                   help="output file name (default: AuAu_0_10_jet[_<campaign>]_seed<NNNN>.h5, "
                        "or AuAu_0_10_jet_<campaign>_<NNNN>.h5 with --seed 0)")
    p.add_argument("--build", default=os.path.join(rp.XSCAPE, "build_gpu"),
                   help="X-SCAPE build tree (default: build_gpu)")
    p.add_argument("--main-xml", default=os.path.join(rp.XSCAPE, "config", "jetscape_main.xml"),
                   dest="main_xml")
    p.add_argument("--user-xml", default=USER_XML, dest="user_xml",
                   help="template user XML (default: the one in this folder)")
    p.add_argument("--native", action="store_true",
                   help="write MUSIC's own grid for both legs instead of the YAML's")
    p.add_argument("--hard", choices=("pythia", "pgun"), default="pythia",
                   help="hard process: PythiaGun (default; vertex from the Glauber "
                        "collisions) or PGun (fixed pT, vertex at the origin)")
    p.add_argument("--pthat-min", type=float, default=None, dest="pthat_min",
                   help="PythiaGun pTHatMin [GeV] (default: the XML's)")
    p.add_argument("--pthat-max", type=float, default=None, dest="pthat_max",
                   help="PythiaGun pTHatMax [GeV] (default: the XML's)")
    p.add_argument("--pgun-pt", type=float, default=60.0, dest="pgun_pt",
                   help="PGun parton pT [GeV] (default 60)")
    p.add_argument("--reuse", type=int, default=1,
                   help="jet events per background (setReuseHydro); 1 = a new background "
                        "every event (default)")
    p.add_argument("--pthat-bins", default=None, dest="pthat_bins",
                   help="several PythiaGun pTHat windows in one job, e.g. 20-40,50-70,70-90 "
                        "[GeV]: one Pythia per window, event i uses window i mod K, and every "
                        "background gets --jets-per-bin jets in each window (sets --reuse "
                        "K x jets-per-bin). Records pthat, the window and the event weight "
                        "per event and each window's cross section. Needs X-SCAPE's "
                        "PythiaGun with <pTHatBins>")
    p.add_argument("--jets-per-bin", type=int, default=1, dest="jets_per_bin",
                   help="with --pthat-bins: jets per window per background (default 1)")
    p.add_argument("--parton-ymax", type=float, default=None, dest="parton_ymax",
                   help="PythiaGun: keep only events whose partons handed to the framework "
                        "(status 62 after ISR/MPI; final partons with FSR_on) pass |y| < YMAX "
                        "on their two hardest (see --parton-y-mode). Rejected events are "
                        "regenerated before any shower or hydro; the cross sections are "
                        "Pythia's x kept/tried per window. Records the acceptance and the "
                        "partons' rapidity per event. Needs X-SCAPE's PythiaGun with "
                        "<partonYMax>")
    p.add_argument("--parton-y-mode", choices=("leading", "both", "any"), default="leading",
                   dest="parton_y_mode",
                   help="with --parton-ymax: 'leading' the hardest parton (default), "
                        "'both' the two hardest, 'any' either of the two hardest")
    p.add_argument("--bg-layout", choices=("auto", "full", "shared"), default="auto",
                   dest="bg_layout",
                   help="how arr_bg is stored: 'full' one copy per event; 'shared' each "
                        "background once (arr_bg_store) with arr_bg a virtual dataset over "
                        "it that reads the same; 'auto' (default) shared when a background "
                        "is reused (--reuse > 1, or --pthat-bins), else full")
    p.add_argument("--no-deposit", action="store_true", dest="no_deposit",
                   help="null test: MUSIC_2 without the liquefier, so arr must equal arr_bg")
    p.add_argument("--no-showers", action="store_true", dest="no_showers",
                   help="do not store the parton showers (shower/)")
    p.add_argument("--surface", choices=("none", "bg", "jet", "both"), default="none",
                   help="which legs build MUSIC's freeze-out surface. Nothing in this job "
                        "receives a surface unless --write-particlize stores it (which builds "
                        "its legs anyway), so on its own this only costs ~3 s per MUSIC run; "
                        "a warning says so. 'none' (default) gives a bit-identical evolution")
    p.add_argument("--write-particlize", choices=("none", "jet", "both"), default="none",
                   dest="write_particlize",
                   help="also write <stem>_particlize.h5: the freeze-out surface of the jet "
                        "leg (jet) or of both legs (both, the background once per "
                        "background) plus the final partons, everything hadronize.py needs "
                        "to hadronize the event exactly. Switches on the surface of those "
                        "legs (see --surface). Default none")
    p.add_argument("--validate-inline", action="store_true", dest="validate_inline",
                   help="validation only: also run iSS on the jet leg and Colorless jet "
                        "hadronization inside the job (settings from --hadronize-xml, "
                        "Pythia reseeded per event) and store their hadrons and seeds in "
                        "<stem>_inline_{bulk_jet,jet_frag}.h5, to compare with hadronize.py "
                        "--use-stored-seeds. Needs --write-particlize")
    p.add_argument("--hadronize-xml", default=os.path.join(HERE, "hadronize.xml"),
                   dest="hadronize_xml",
                   help="iSS / JetHadronization settings for --validate-inline (default: "
                        "hadronize.xml next to this script)")
    p.add_argument("--dry-run", action="store_true", dest="dry_run",
                   help="check the XML and grid, write the job XML, print the plan; do not run")
    rp.add_seed_args(p, "3dMCGlauber, Pythia, Matter/LBT")
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


def _set_text(parent, tag, value):
    """As _set, without the padding (for strings compared or used as paths verbatim)."""
    n = _set(parent, tag, value)
    n.text = str(value)
    return n


def music_delta_tau(build: str) -> float:
    """MUSIC's time step from the build tree's music_input."""
    with open(os.path.join(build, "music_input")) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "Delta_Tau":
                return float(parts[1])
    sys.exit(f"run_prod_jet.py: no Delta_Tau in {build}/music_input")


def parse_pthat_bins(text: str) -> list:
    """'20-40,50-70' -> [(20.0, 40.0), (50.0, 70.0)].

    PythiaGun hands the edges to Pythia with one decimal (as for pTHatMin/pTHatMax), so
    finer edges are refused rather than silently rounded.
    """
    bins = []
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        lo_hi = part.split("-")
        try:
            if len(lo_hi) != 2:
                raise ValueError
            lo, hi = float(lo_hi[0]), float(lo_hi[1])
        except ValueError:
            raise ValueError(f"--pthat-bins: {part!r} is not MIN-MAX") from None
        if not 0 <= lo < hi:
            raise ValueError(f"--pthat-bins: {part!r} needs 0 <= MIN < MAX")
        for x in (lo, hi):
            if abs(round(x, 1) - x) > 1e-9:
                raise ValueError(f"--pthat-bins: {x:g} has more than one decimal, which "
                                 "PythiaGun would round")
        bins.append((lo, hi))
    if not bins:
        raise ValueError("--pthat-bins: no window given")
    return bins


def check_pthat_bins(a) -> None:
    """--pthat-bins: parse, check against the other options, set --reuse."""
    a.pthat_windows = None
    if a.pthat_bins is None:
        if a.jets_per_bin != 1:
            raise ValueError("--jets-per-bin needs --pthat-bins")
        return
    if a.hard != "pythia":
        raise ValueError("--pthat-bins is for --hard pythia")
    if a.pthat_min is not None or a.pthat_max is not None:
        raise ValueError("--pthat-bins replaces --pthat-min/--pthat-max")
    if a.reuse != 1:
        raise ValueError("--pthat-bins sets --reuse itself (windows x --jets-per-bin); "
                         "give --jets-per-bin instead")
    if a.jets_per_bin < 1:
        raise ValueError("--jets-per-bin must be >= 1")
    a.pthat_windows = parse_pthat_bins(a.pthat_bins)
    a.reuse = len(a.pthat_windows) * a.jets_per_bin
    if a.events % a.reuse:
        raise ValueError(f"--events {a.events} is not a multiple of {a.reuse} (windows x "
                         "--jets-per-bin): the last background would miss windows")


def check_parton_ycut(a) -> None:
    """--parton-ymax / --parton-y-mode against the other options."""
    if a.parton_ymax is None:
        if a.parton_y_mode != "leading":
            raise ValueError("--parton-y-mode needs --parton-ymax")
        return
    if a.hard != "pythia":
        raise ValueError("--parton-ymax is for --hard pythia")
    if not a.parton_ymax > 0:
        raise ValueError(f"--parton-ymax must be > 0, got {a.parton_ymax:g}")


def pthat_bins_text(windows) -> str:
    """The <pTHatBins> text: 'min max min max ...'."""
    return " ".join(f"{lo:g} {hi:g}" for lo, hi in windows)


def surface_legs(a) -> set:
    """Legs whose MUSIC builds a freeze-out surface: --surface plus --write-particlize."""
    legs = {"none": set(), "bg": {"bg"}, "jet": {"jet"}, "both": {"bg", "jet"}}[a.surface]
    legs |= {"none": set(), "jet": {"jet"}, "both": {"bg", "jet"}}[a.write_particlize]
    return legs


def particlize_legs(a) -> tuple:
    return {"none": (), "jet": ("jet",), "both": ("jet", "bg")}[a.write_particlize]


def add_inline_hadronization(root, hadronize_xml):
    """--validate-inline: iSS on MUSIC_2 and Colorless jet hadronization in this job.

    The blocks come from hadronize.xml, the file hadronize.py uses, so both sides run the
    same settings.  iSS samples the jet leg (hydro_id MUSIC_2) and reads the job's own
    music_input (iSS_working_path "."); Pythia is reseeded every event so each event's jet
    fragments can be reproduced alone.  '../' paths are made absolute with the main XML's.
    """
    src = ET.parse(hadronize_xml).getroot()
    soft, had = src.find("SoftParticlization"), src.find("JetHadronization")
    if soft is None or had is None:
        sys.exit(f"run_prod_jet.py: {hadronize_xml} needs <SoftParticlization> and "
                 "<JetHadronization>")
    soft, had = copy.deepcopy(soft), copy.deepcopy(had)
    _set_text(soft, "hydro_id", JET_ID)          # compared verbatim with the module id
    _set_text(soft.find("iSS"), "iSS_working_path", ".")
    _set(had, "reseed_per_event", 1)
    root.append(had)
    root.append(soft)


def job_xml(a, out_h5: str):
    """Check the template, apply this job's settings, write it next to the output."""
    tree = ET.parse(a.user_xml)
    root = tree.getroot()
    problems = []

    _set(root, "nEvents", a.events)
    root.find("Random/seed").text = str(a.seed)
    _set(root, "setReuseHydro", "true" if a.reuse > 1 else "false")
    _set(root, "nReuseHydro", max(1, a.reuse))

    # Exactly one hard process: the automatic task list would run every block present.
    hard = root.find("Hard")
    if hard is None:
        sys.exit(f"run_prod_jet.py: {a.user_xml} has no <Hard> block")
    for child in list(hard):
        hard.remove(child)
    template = ET.parse(a.user_xml).getroot().find("Hard")
    if a.hard == "pythia":
        pg = template.find("PythiaGun")
        if pg is None:
            sys.exit(f"run_prod_jet.py: {a.user_xml} has no <Hard><PythiaGun>")
        hard.append(pg)
        if a.pthat_min is not None:
            _set(pg, "pTHatMin", a.pthat_min)
        if a.pthat_max is not None:
            _set(pg, "pTHatMax", a.pthat_max)
        if a.pthat_windows:
            _set(pg, "pTHatBins", pthat_bins_text(a.pthat_windows))
        elif pg.find("pTHatBins") is not None:
            pg.remove(pg.find("pTHatBins"))      # a template's windows: only with --pthat-bins
        if a.parton_ymax is not None:
            _set(pg, "partonYMax", f"{a.parton_ymax:g}")
            _set(pg, "partonYMode", a.parton_y_mode)
        else:                                    # a template's cut: only with --parton-ymax
            for tag in ("partonYMax", "partonYMode"):
                if pg.find(tag) is not None:
                    pg.remove(pg.find(tag))
    else:
        pgun = ET.SubElement(hard, "PGun")
        _set(pgun, "name", "PGun")
        _set(pgun, "pT", a.pgun_pt)

    hydros = root.findall("Hydro")
    names = [_text(h, "MUSIC/name") for h in hydros]
    if names != [BG_ID, JET_ID]:
        problems.append(f"need exactly two <Hydro><MUSIC> blocks named {BG_ID}, {JET_ID} "
                        f"(in that order), found {names}")
    else:
        music = hydros[0].find("MUSIC")
        if _text(music, "output_evolution_to_memory") != "1":
            problems.append(f"{BG_ID} needs <output_evolution_to_memory>1")
        if _text(music, "dump_hydro_only") != "0":
            problems.append(f"{BG_ID} needs <dump_hydro_only>0: Matter/LBT read its "
                            "framework medium (MUSIC_2 is switched to 1 in Python)")
        _set(hydros[1], "AddLiquefier", "false" if a.no_deposit else "true")
        # First block: the background leg and the default for every instance; MUSIC_2's
        # own block always gets an explicit value, which overrides it for the jet leg.
        legs = surface_legs(a)
        _set(music, "freeze_out_surface", 1 if "bg" in legs else 0)
        _set(hydros[1].find("MUSIC"), "freeze_out_surface", 1 if "jet" in legs else 0)
    if _text(root, "Preequilibrium/evolutionInMemory") != "0":
        problems.append("<Preequilibrium><evolutionInMemory> must be 0 with NullPreDynamics "
                        "and 3D-Glauber strings (otherwise the medium's tau axis is garbage)")
    if root.find("Eloss") is None or _text(root, "Eloss/AddLiquefier") != "true":
        problems.append("<Eloss> with <AddLiquefier>true is required")
    liq = root.find("Liquefier/CausalLiquefier")
    if liq is None:
        problems.append("<Liquefier><CausalLiquefier> is required")
    else:
        dtau, music_dtau = float(_text(liq, "dtau")), music_delta_tau(a.build)
        if abs(dtau - music_dtau) > 1e-9:
            problems.append(f"<CausalLiquefier><dtau> = {dtau} must equal MUSIC's Delta_Tau "
                            f"= {music_dtau}: the kernel is divided by it and deposited in "
                            "one hydro step")
    for tag in ("SoftParticlization", "JetHadronization", "Afterburner", "RootBulkWriter",
                "FastRootBulkWriter"):
        if root.find(tag) is not None:
            problems.append(f"remove <{tag}>: this job writes the hydro pair only (hadrons "
                            "come from hadronize.py; --validate-inline adds its own blocks)")
    if a.validate_inline and not problems:
        add_inline_hadronization(root, a.hadronize_xml)
    if problems:
        sys.exit(f"run_prod_jet.py: {a.user_xml}:\n  " + "\n  ".join(problems))

    path = os.path.splitext(out_h5)[0] + ".xml"
    tree.write(path)
    return path, root


#: pair-writer diag/ values copied into the particlize file's events/ (energy bookkeeping
#: for the hadron-level balance, and the flags that make an event suspect)
PARTICLIZE_DIAG = ("n_droplets", "E_droplets", "E_droplets_late", "E_droplets_early",
                   "tau0_music", "ntau_jet", "ntau_bg", "frames_identical",
                   "jet_hit_boundary", "bg_hit_boundary", "pthat_bin", "pthat",
                   "event_weight", "parton_y_lead", "parton_pt_lead", "parton_y_sub",
                   "parton_pt_sub")


def open_pthat_bins(a, jetscape):
    """--pthat-bins / --parton-ymax: the PythiaGun task, after checking that it runs
    exactly these windows and this cut.

    A PythiaGun without <pTHatBins> or <partonYMax> would refuse the XML (no default in its
    main XML) or, built otherwise, ignore it, so the build is checked here rather than
    trusted.
    """
    if not a.pthat_windows and a.parton_ymax is None:
        return None
    from jetscape import pyjetscape_core as core

    if not getattr(core, "PYTHIA_GUN_HAS_PTHAT_BINS", False):
        sys.exit("run_prod_jet.py: --pthat-bins / --parton-ymax need X-SCAPE's PythiaGun "
                 "with <pTHatBins> (branch N_ptHat_per_hydro) and PyJetscape built against it")
    if a.parton_ymax is not None and not getattr(core, "PYTHIA_GUN_HAS_PARTON_Y_CUT", False):
        sys.exit("run_prod_jet.py: --parton-ymax needs X-SCAPE's PythiaGun with <partonYMax> "
                 "(branch pyGun_eta_cut) and PyJetscape built against it")
    tasks = {t.GetId(): t for t in jetscape.GetTaskList()}
    gun = tasks.get("PythiaGun")
    if gun is None:
        sys.exit(f"run_prod_jet.py: --pthat-bins / --parton-ymax found no PythiaGun task "
                 f"(tasks: {sorted(tasks)})")
    info = core.pythia_gun_bins(gun)
    got = [tuple(b) for b in info["bins"]]
    if a.pthat_windows and got != [tuple(w) for w in a.pthat_windows]:
        sys.exit(f"run_prod_jet.py: PythiaGun runs the windows {got}, not "
                 f"{a.pthat_windows}")
    if a.parton_ymax is not None and (abs(info["y_max"] - a.parton_ymax) > 1e-12
                                      or info["y_mode"] != a.parton_y_mode):
        sys.exit(f"run_prod_jet.py: PythiaGun cuts |y| < {info['y_max']:g} "
                 f"({info['y_mode']}), not {a.parton_ymax:g} ({a.parton_y_mode})")
    print("  pTHat    " + ", ".join(f"window {k}: {lo:g}-{hi:g} GeV (Pythia seed {s})"
                                    for k, ((lo, hi), s) in enumerate(zip(got,
                                                                          info["seeds"]))))
    if a.parton_ymax is not None:
        print(f"  cut      handed-over partons, two hardest, |y| < {a.parton_ymax:g} "
              f"({a.parton_y_mode})")
    return gun


def pthat_diag(gun) -> dict:
    """The current event's window, pTHat and weight, for diag/ (and events/)."""
    if gun is None:
        return {}
    from jetscape import pyjetscape_core as core

    h = core.hard_process_info(gun)
    d = {"pthat_bin": int(core.pythia_gun_bins(gun)["active"]),
         "pthat": float(h["pthat"]), "event_weight": float(h["event_weight"])}
    if hasattr(core, "hard_partons_numpy"):
        # the partons handed to the framework (what --parton-ymax cuts on), hardest first:
        # pid, px, py, pz, E, pT, y
        hp = core.hard_partons_numpy(gun)
        if len(hp):
            d.update(parton_y_lead=float(hp[0, 6]), parton_pt_lead=float(hp[0, 5]))
            if len(hp) > 1:
                d.update(parton_y_sub=float(hp[1, 6]), parton_pt_sub=float(hp[1, 5]))
    return d


def pthat_windows_final(gun) -> dict:
    """Each window's final Pythia cross section etc., as file attributes."""
    import numpy as np
    from jetscape import pyjetscape_core as core

    info = core.pythia_gun_bins(gun)
    out = {"pthat_bins": np.asarray(info["bins"], dtype=np.float64),
           "pthat_bin_sigma_gen": np.asarray(info["sigma_gen"], dtype=np.float64),
           "pthat_bin_sigma_err": np.asarray(info["sigma_err"], dtype=np.float64),
           "pthat_bin_n_accepted": np.asarray(info["n_accepted"], dtype=np.int64),
           "pthat_bin_seeds": np.asarray(info["seeds"], dtype=np.int64)}
    if info.get("y_max", 0) > 0:
        # sigma_gen is Pythia's x kept/tried; the raw values count the rejected events
        out.update(pthat_bin_n_tried=np.asarray(info["n_tried"], dtype=np.int64),
                   pthat_bin_n_kept=np.asarray(info["n_kept"], dtype=np.int64),
                   pthat_bin_acceptance=np.asarray(info["acceptance"], dtype=np.float64),
                   pthat_bin_sigma_gen_raw=np.asarray(info["sigma_gen_raw"],
                                                      dtype=np.float64),
                   pthat_bin_sigma_err_raw=np.asarray(info["sigma_err_raw"],
                                                      dtype=np.float64))
    return out


class InlineHadrons:
    """--validate-inline: the in-job iSS (jet leg) and Colorless hadrons, with seeds."""

    def __init__(self, iss, hadro_mgr, stem, n_oversample):
        from jetscape.hadrons_h5 import HadronH5Writer

        self.iss, self.mgr = iss, hadro_mgr
        attrs = {"inline": True, "source": os.path.basename(stem) + "_particlize.h5"}
        self.w_bulk = HadronH5Writer(stem + "_inline_bulk_jet.h5", tag="bulk_jet",
                                     n_samples=n_oversample, attrs=attrs)
        self.w_frag = HadronH5Writer(stem + "_inline_jet_frag.h5", tag="jet_frag",
                                     n_samples=1, attrs=attrs)

    def capture(self, idx):
        from jetscape import pyjetscape_core as core

        bulk = core.soft_hadrons_numpy(self.iss)
        iss_seed = int(core.soft_last_random_seed(self.iss))
        self.w_bulk.append_unit(bulk, unit=idx, event=idx, seed=iss_seed)
        frag = core.hadronization_hadrons_numpy(self.mgr)
        py_seed = int(core.jet_hadronization_last_random_seed(self.mgr))
        self.w_frag.append_unit([frag], unit=idx, event=idx, seed=py_seed)
        print(f"  inline: iSS {len(bulk['pid'])} hadrons in {len(bulk['sample_counts'])} "
              f"sample(s) (seed {iss_seed}), Colorless {len(frag['pid'])} hadrons "
              f"(seed {py_seed})")
        return {"inline_iss_seed_jet": iss_seed, "inline_pythia_seed": py_seed}

    def close(self):
        self.w_bulk.close()
        self.w_frag.close()


def open_inline(a, jetscape, stem):
    tasks = {t.GetId(): t for t in jetscape.GetTaskList()}
    if "iSS" not in tasks or "HadronizationManager" not in tasks:
        sys.exit(f"run_prod_jet.py: --validate-inline found no iSS / HadronizationManager "
                 f"task (tasks: {sorted(tasks)}); is this build configured with iSS?")
    n_os = int(_text(ET.parse(a.hadronize_xml).getroot(),
                     "SoftParticlization/iSS/number_of_repeated_sampling") or 1)
    return InlineHadrons(tasks["iSS"], tasks["HadronizationManager"], stem, n_os)


def main() -> int:
    a = parse_args()
    for k in ("build", "outdir", "main_xml", "user_xml", "grid"):
        setattr(a, k, os.path.abspath(getattr(a, k)))
    rp.check_env(a)
    if a.reuse < 1:
        sys.exit("run_prod_jet.py: --reuse must be >= 1")
    try:
        check_pthat_bins(a)
        check_parton_ycut(a)
    except ValueError as exc:
        sys.exit(f"run_prod_jet.py: {exc}")
    if a.validate_inline and a.write_particlize == "none":
        sys.exit("run_prod_jet.py: --validate-inline compares with the stored surfaces and "
                 "partons; add --write-particlize jet (or both)")
    a.hadronize_xml = os.path.abspath(a.hadronize_xml)
    unused = sorted(surface_legs(a) - set(particlize_legs(a)))
    if unused:
        print(f"run_prod_jet.py: WARNING -- --surface {a.surface} builds the freeze-out surface "
              f"of the {' and '.join(unused)} leg(s), but nothing in this job receives it: "
              f"only --write-particlize hands a surface to the framework and stores it. "
              f"That costs ~3 s per MUSIC run of that leg for no output; use "
              f"--write-particlize "
              f"{'both' if 'bg' in unused else 'jet'} to keep it, or drop --surface.",
              file=sys.stderr)

    grid, max_ntau, grid_text = rp.load_grid_yaml(a.grid)
    os.makedirs(a.outdir, exist_ok=True)
    out_h5 = os.path.join(a.outdir, rp.output_name(a, "AuAu_0_10_jet"))
    rp.resolve_seed(a, out_h5, record=not a.dry_run)
    xml, root = job_xml(a, out_h5)
    box = rp.music_box(root, a.main_xml)
    grid_mode = "native" if a.native else "grid"
    if not a.native:
        rp.check_inside(grid, box)

    if a.pthat_windows:
        hard_desc = ("PythiaGun pTHat windows "
                     + ", ".join(f"{lo:g}-{hi:g}" for lo, hi in a.pthat_windows)
                     + f" GeV, {a.jets_per_bin} jet(s) per window per background")
    elif a.hard == "pythia":
        hard_desc = (f"PythiaGun pTHat {_text(root, 'Hard/PythiaGun/pTHatMin')}-"
                     f"{_text(root, 'Hard/PythiaGun/pTHatMax')} GeV")
    else:
        hard_desc = f"PGun pT {a.pgun_pt:g} GeV"
    if a.parton_ymax is not None:
        hard_desc += f", partons |y| < {a.parton_ymax:g} ({a.parton_y_mode})"
    print(f"prod_AuAu_0_10_jet: {a.events} event(s), seed {a.seed}"
          + (" (from OS entropy)" if a.seed_source == "os_entropy" else "")
          + (f", campaign {a.campaign} #{a.index}" if a.campaign else "")
          + f", grid_mode {grid_mode}, "
          f"{hard_desc}, reuse {a.reuse}, deposition {'OFF (null test)' if a.no_deposit else 'on'}, "
          f"freeze-out surface: {'+'.join(sorted(surface_legs(a))) or 'none'}"
          + (f", particlize input: {a.write_particlize}" if particlize_legs(a) else "")
          + (" + inline iSS/Colorless (validation)" if a.validate_inline else ""))
    print(f"  build    {a.build}")
    print(f"  job XML  {xml}")
    print(f"  grid     {a.grid}")
    if a.native:
        print("  grid     MUSIC native: " + ", ".join(
            f"{ax} {b[0]:g} .. {b[1]:g} (n = {b[2]})" for ax, b in box.items())
            + f"; max_ntau = {max_ntau or 'auto'}")
    else:
        print(rp.describe(grid, max_ntau))
    print(f"  output   {out_h5}")
    stem = os.path.splitext(out_h5)[0]
    out_particlize = stem + "_particlize.h5" if particlize_legs(a) else None
    if out_particlize:
        print(f"  output   {out_particlize}")
    if a.dry_run:
        return 0

    import h5py
    import numpy as np
    import jetscape as js
    from jetscape.pair_h5 import PairH5Writer

    provenance = {
        "prod": "PyJetscape/example/prod_AuAu_0_10_jet",
        **rp.seed_provenance(a),
        "prod_host": socket.gethostname(),
        "prod_platform": platform.platform(),
        "prod_build": a.build,
        "prod_user_xml": open(xml).read(),
        "prod_grid_yaml": grid_text,
        "prod_hard": hard_desc,
        "prod_reuse": a.reuse,
        "prod_surface": a.surface,
        "prod_write_particlize": a.write_particlize,
        **({"pthat_bins": np.asarray(a.pthat_windows, dtype=np.float64),
            "pthat_jets_per_bin": a.jets_per_bin} if a.pthat_windows else {}),
        **({"parton_ymax": a.parton_ymax, "parton_y_mode": a.parton_y_mode}
           if a.parton_ymax is not None else {}),
        "system": "AuAu 200 GeV 0-10% (b in [0, 4.7] fm)",
        "initial_state_kind": "3dMCGlauber_strings",
        "hydro": "MUSIC (music4gpu)" if "gpu" in os.path.basename(a.build) else "MUSIC",
        "jet": "Matter + LBT, CausalLiquefier droplets into MUSIC_2",
        "T_fo": 0.15,
    }
    writer = PairH5Writer(
        out_h5, bg_id=BG_ID, jet_id=JET_ID, grid_mode=grid_mode,
        out_grid=None if a.native else grid, choose_ntau=max_ntau,
        store_showers=not a.no_showers, compression=a.compression, keep_bits=a.keep_bits,
        provenance={"hard_vertex": HARD_VERTEX[a.hard],
                    "eos_kind": "hotqcd (MUSIC EOS 9)",
                    "transport_mode": "MUSIC viscous: eta/s(T) and zeta/s(T) "
                                      "parametrization 3, second-order terms"},
        keep_surface=particlize_legs(a), extra_attrs=provenance, verbose=True,
        bg_layout=a.bg_layout, reuse=a.reuse)
    pwriter = None
    if out_particlize:
        from jetscape.particlize_h5 import ParticlizeH5Writer
        pwriter = ParticlizeH5Writer(
            out_particlize, legs=particlize_legs(a), bg_id=BG_ID, jet_id=JET_ID,
            T_fo=provenance["T_fo"], pair_file=out_h5,
            extra_attrs={k: v for k, v in provenance.items() if k != "T_fo"},
            verbose=True)

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
    writer.attach(jetscape)                  # after Init: MUSIC_2 -> dump_hydro_only
    if pwriter is not None:
        pwriter.attach(jetscape)             # reads ./music_input: the job's own
    inline = open_inline(a, jetscape, stem) if a.validate_inline else None
    gun = open_pthat_bins(a, jetscape)

    late = 0
    droplets = e_late = 0.0
    t_job = time.time()
    try:
        jetscape.ExecInit()
        for i in range(jetscape.GetNumberOfEvents()):
            t0 = time.time()
            jetscape.ExecPerEvent()
            idx = writer.Exec(**pthat_diag(gun))    # both legs are live here
            if idx is not None and pwriter is not None:
                d = writer.last_event_diag
                pwriter.Exec(idx, bg_id=d["bg_id"], bg_key=writer.last_bg_key,
                             **{k: d[k] for k in PARTICLIZE_DIAG if k in d})
                if inline is not None:
                    pwriter.write_events(idx, **inline.capture(idx))
            jetscape.ClearPerEvent()
            wall = time.time() - t0
            if idx is None:
                print(f"prod_AuAu_0_10_jet: event {i + 1}/{a.events} SKIPPED (see warning)")
                continue
            writer.write_diag(idx, wall_s=wall)
            d = writer.last_event_diag
            droplets += d["n_droplets"]
            e_late += d["E_droplets_late"]
            msg = (f"prod_AuAu_0_10_jet: event {i + 1}/{a.events} done in {wall:.1f} s, "
                   f"MUSIC tau0 = {d['tau0_music']:.3f} fm/c, jet/bg frames "
                   f"{d['ntau_jet']}/{d['ntau_bg']}, {d['n_droplets']} droplets "
                   f"({d['E_droplets']:.1f} GeV, {d['E_droplets_late']:.1f} after freeze-out), "
                   f"{d['frames_identical']} leading frames identical, bg {d['bg_id']}")
            if not a.native and d["tau0_music"] > grid.tau_min + rp._TOL:
                late += 1
                msg += (f"  WARNING: tau0 after tau.min = {grid.tau_min:g}, so the first "
                        "frame(s) of this event are zeros")
            if gun is not None:
                msg += f", pTHat {d['pthat']:.1f} GeV (window {d['pthat_bin']})"
                if "parton_y_lead" in d:
                    msg += f", hardest parton y {d['parton_y_lead']:+.2f}"
            print(msg)
        windows = None if gun is None else pthat_windows_final(gun)
        jetscape.Finish()
    finally:
        writer.Finish()
        if pwriter is not None:
            pwriter.Finish(complete=pwriter.GetNumberOfEventsWritten()
                           == writer.GetNumberOfEventsWritten())
        if inline is not None:
            inline.close()

    n = writer.GetNumberOfEventsWritten()
    if n:
        with h5py.File(out_h5, "a") as f:
            f.attrs["prod_wall_s_total"] = time.time() - t_job
            if windows is not None:
                f.attrs.update(windows)
        if windows is not None and out_particlize and pwriter.GetNumberOfEventsWritten():
            with h5py.File(out_particlize, "a") as f:
                f.attrs.update(windows)
    summary = {"out": out_h5, "seed": a.seed, "seed_source": a.seed_source,
               "campaign": a.campaign, "index": a.index, "grid": a.grid, "hard": hard_desc,
               "reuse": a.reuse, "deposition": not a.no_deposit, "surface": a.surface,
               "events_requested": a.events, "events_written": n,
               "events_tau0_after_tau_min": late,
               "legs_cut_at_max_ntau": writer.n_clipped,
               "droplets_total": int(droplets), "E_droplets_late_total": round(e_late, 3),
               "wall_s": round(time.time() - t_job, 1)}
    if pwriter is not None:
        summary.update(particlize=out_particlize, particlize_legs=list(particlize_legs(a)),
                       particlize_events_written=pwriter.GetNumberOfEventsWritten())
    if writer.bg_layout == "shared":
        summary["bg_layout"] = "shared"
    if windows is not None:
        if a.parton_ymax is not None:
            summary.update(parton_ymax=a.parton_ymax, parton_y_mode=a.parton_y_mode,
                           pthat_bin_acceptance=[round(float(x), 4) for x in
                                                 windows["pthat_bin_acceptance"]])
        summary.update(pthat_bins=windows["pthat_bins"].tolist(), jets_per_bin=a.jets_per_bin,
                       pthat_bin_sigma_gen_mb=list(windows["pthat_bin_sigma_gen"]),
                       pthat_bin_seeds=[int(x) for x in windows["pthat_bin_seeds"]])
    with open(os.path.splitext(out_h5)[0] + ".json", "w") as f:
        json.dump(summary, f, indent=1)
    print("prod_AuAu_0_10_jet:", json.dumps(summary))
    rp.leave_workdir(a, workdir, n == a.events)
    return 0 if n == a.events else 1


if __name__ == "__main__":
    raise SystemExit(main())
