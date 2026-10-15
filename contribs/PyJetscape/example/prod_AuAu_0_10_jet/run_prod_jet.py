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
    python run_prod_jet.py --events 24 --seed 1 --user-xml ../config/PbPb_0_30_ZJet.xml \
        --grid ../config/grid_PbPb.yaml --pthat-bins 20-30,30-40,40-60,60-80,80-120,120-200 \
        --jets-per-bin 4 --parton-ymax 1.9 --lepton-ptmin 20 --lepton-etamax 2.5 --hadrons-only
                                     # Pb+Pb Z+jet (the XML's PythiaGun has <trigger>Z): the
                                     #   Z's leptons pass through like photons, the Z per
                                     #   event
    python run_prod_jet.py --events 10 --seed 1 --surface jet    # surface for the jet leg only
    python run_prod_jet.py --events 10 --seed 1 --write-particlize both
                                     # + <stem>_particlize.h5: both surfaces and the final
                                     #   partons, for hadronize.py (PLAN_particlize_h5.md)
    python run_prod_jet.py --events 10 --seed 1 --write-particlize both --particlize-only
                                     # only <stem>_particlize.h5 (+ .json): no pair file
    python run_prod_jet.py --events 30 --seed 1 --reuse 3 --hadrons-only
                                     # only hadron files: iSS on both legs and the jet
                                     #   hadronization in the job (PLAN_AuAu_hadrons_only.md)
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

USER_XML = os.path.join(rp.CONFIG, "AuAu_MCGlauber_MUSIC_0_10_jet.xml")
GRID_YAML = os.path.join(rp.CONFIG, "grid_fno.yaml")
HADRONIZE_XML = os.path.join(rp.CONFIG, "hadronize.xml")
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
                   help="output grid YAML (default: ../config/grid_fno.yaml)")
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
                   help="template user XML (default: ../config/"
                        "AuAu_MCGlauber_MUSIC_0_10_jet.xml)")
    p.add_argument("--native", action="store_true",
                   help="write MUSIC's own grid for both legs instead of the YAML's")
    p.add_argument("--edge-e-threshold", type=float, default=None, dest="edge_e_threshold",
                   metavar="E",
                   help="flag a leg whose energy density on the x/y edge of the output grid "
                        "exceeds E [GeV/fm^3] (diag/{bg,jet}_hit_edge). Default: MUSIC's "
                        "freeze-out energy density from the job XML (eps_switch, or "
                        "e(freezeout_temperature) from the build's EOS 9/91 table); 0: no "
                        "flag. diag/{bg,jet}_edge_e_max are recorded either way")
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
                        "on their two hardest (see --parton-y-mode; leptons do not count). "
                        "Rejected events are regenerated before any shower or hydro; the "
                        "cross sections are Pythia's x kept/tried per window. Records the "
                        "acceptance and the partons' rapidity per event. Needs X-SCAPE's "
                        "PythiaGun with <partonYMax>")
    p.add_argument("--parton-y-mode", choices=("leading", "both", "any"), default="leading",
                   dest="parton_y_mode",
                   help="with --parton-ymax: 'leading' the hardest parton (default), "
                        "'both' the two hardest, 'any' either of the two hardest")
    p.add_argument("--trigger", choices=("none", "Z"), default=None,
                   help="PythiaGun's trigger (<trigger>; default: the XML's, else none). 'Z': "
                        "an event needs an e+e- or mu+mu- pair among the particles handed "
                        "over (the Z's decay leptons, which pass through the energy loss like "
                        "photons), and the Z and its leptons are recorded per event (z_pt, "
                        "z_y, z_phi, z_m, lep{1,2}_{pid,pt,eta,phi}). The process is the XML's "
                        "<LinesToRead>. Needs X-SCAPE's PythiaGun with <trigger>")
    p.add_argument("--trigger-ymax", type=float, default=None, dest="trigger_ymax",
                   metavar="Y",
                   help="trigger Z: keep only events with |y_Z| < Y. As all the trigger cuts, "
                        "rejected events are regenerated before any shower or hydro and the "
                        "cross sections are Pythia's x kept/tried (of every cut, with "
                        "--parton-ymax)")
    p.add_argument("--lepton-ptmin", type=float, default=None, dest="lepton_ptmin",
                   metavar="PT", help="trigger Z: both leptons with pT > PT [GeV]")
    p.add_argument("--lepton-etamax", type=float, default=None, dest="lepton_etamax",
                   metavar="ETA", help="trigger Z: both leptons with |eta| < ETA")
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
    p.add_argument("--particlize-only", action="store_true", dest="particlize_only",
                   help="write the particlize file only, no pair file: MUSIC still runs "
                        "both legs, but only the background's evolution is read (to "
                        "recognise a reused one), and no evolution, source/droplets or "
                        "shower/ is stored. The particlize file's events/ then have "
                        "ntau_jet_music (MUSIC steps) instead of ntau_jet, no "
                        "frames_identical, and wall_s. Needs --write-particlize")
    p.add_argument("--music2-restart", action=argparse.BooleanOptionalAction, default=True,
                   dest="music2_restart",
                   help="(default: on) MUSIC_2 starts from MUSIC_1's state at the last "
                        "freeze-out step before the liquefier's tau_delay (no droplet "
                        "deposits earlier) instead of recomputing the identical leading "
                        "steps: no string deposition and ~30 fewer steps per jet event, "
                        "the same output (music4gpu; js-contrib "
                        "docs/Plans/PLAN_music2_restart.md). An event that does not fit "
                        "runs from tau0 and the log says why. Off by itself with "
                        "--write-particlize jet / --surface jet (MUSIC_1 builds no surface) "
                        "and on a build without the restart. --no-music2-restart: every "
                        "jet leg from tau0. MUSIC_RESTART_VERIFY=1 runs every jet leg from "
                        "tau0 and compares its state with the checkpoint instead")
    p.add_argument("--eos", choices=("9", "91"), default=None,
                   help="MUSIC's EOS for both legs (default: the XML's, 91: hotQCD + the SMASH "
                        "hadron list, the tunes' EOS; 9: the UrQMD list, older campaigns). "
                        "Recorded as the files' eos / eos_kind")
    p.add_argument("--late-dtau", type=float, default=None, dest="late_dtau", metavar="DT",
                   help="MUSIC's time step from --late-tau on [fm/c] (the XML's "
                        "<late_delta_tau>, 0.05 in the production XML; 0: constant step)")
    p.add_argument("--late-tau", type=float, default=None, dest="late_tau", metavar="TAU",
                   help="when the larger step starts [fm/c] (the XML's <late_tau_switch>)")
    p.add_argument("--bulk-info", choices=("slim", "full"), default="slim", dest="bulk_info",
                   help="the background's framework copy, which Matter, LBT and the liquefier "
                        "read: 'slim' (default) keeps e, s, T, vx, vy, vz per cell, ~7 GiB "
                        "less memory at 0-10%%, same output; 'full' the whole FluidCellInfo "
                        "(also P; viscous fields and mu's are zero from MUSIC's store anyway), "
                        "for modules that read more (<Hydro><MUSIC><slim_bulk_info>)")
    p.add_argument("--validate-inline", action="store_true", dest="validate_inline",
                   help="validation only: also run iSS on the jet leg and Colorless jet "
                        "hadronization inside the job (settings from --hadronize-xml, "
                        "Pythia reseeded per event) and store their hadrons and seeds in "
                        "<stem>_inline_{bulk_jet,jet_frag}.h5, to compare with hadronize.py "
                        "--use-stored-seeds. Needs --write-particlize")
    p.add_argument("--hadronize-xml", default=HADRONIZE_XML,
                   dest="hadronize_xml",
                   help="iSS / JetHadronization settings for --validate-inline and "
                        "--hadrons-only (default: ../config/hadronize.xml; "
                        "eCMforHadronization is always the job's collision energy)")
    h = p.add_argument_group(
        "hadrons only", "--hadrons-only: iSS on both legs and Colorless on the final partons "
        "in this job, exactly as hadronize.py would from this job's particlize file (the same "
        "code, settings and seeds); writes <stem>_hadrons_{bulk_jet,bulk_bg,jet_frag}.h5 "
        "(+ *_primary.h5) and <stem>.json, nothing else. The options are hadronize.py's")
    h.add_argument("--hadrons-only", action="store_true", dest="hadrons_only",
                   help="write hadron files only: no pair file, no particlize file, no stage 2. "
                        "The jet sample of a seed is the same as without it")
    h.add_argument("--hadronize-seed", type=int, default=None, dest="hadronize_seed",
                   help="base seed of the hadronization (hadronize.py --seed; default 1); "
                        "every unit's seed derives from it and this job's file uuid")
    h.add_argument("--oversample", type=int, default=None,
                   help="iSS samples per jet event (default 1)")
    h.add_argument("--oversample-bg", default=None, dest="oversample_bg",
                   help="iSS samples per background: N, 'auto' (--oversample x the events "
                        "using it) or 'per-pthat-bin' (--oversample x --jets-per-bin). "
                        "Default: per-pthat-bin with --pthat-bins, else auto; with "
                        "--correlated always --oversample (correlated sampling pairs the "
                        "legs only for equal counts)")
    h.add_argument("--oversample-bg-max", type=int, default=None, dest="oversample_bg_max",
                   help="cap for --oversample-bg auto / per-pthat-bin (default 2000)")
    h.add_argument("--n-frag", type=int, default=None, dest="n_frag",
                   help="Colorless fragmentations per event (default 1)")
    h.add_argument("--keep-primaries", default=None, dest="keep_primaries", metavar="N|all",
                   help="iSS's undecayed hadrons of the first N samples of every unit, in "
                        "<stem>_hadrons_bulk_{jet,bg}_primary.h5 (full precision, positions, "
                        "no selection: for SMASH later). Default all; 0: none")
    h.add_argument("--hadron-keep-bits-p", type=int, default=None, dest="hadron_keep_bits_p",
                   help="hadronize.py --keep-bits-p for the hadron files")
    h.add_argument("--hadron-keep-bits-x", type=int, default=None, dest="hadron_keep_bits_x",
                   help="hadronize.py --keep-bits-x for the hadron files")
    h.add_argument("--eta-max", type=float, default=None, dest="eta_max",
                   help="store only hadrons with |eta| < ETA_MAX (hadronize.py --eta-max)")
    h.add_argument("--charged", action="store_true",
                   help="store only charged hadrons (hadronize.py --charged)")
    h.add_argument("--no-x", action="store_true", dest="no_x",
                   help="store no hadron positions (hadronize.py --no-x)")
    h.add_argument("--correlated", action="store_true",
                   help="correlated sampling, jet and background paired (hadronize.py "
                        "--correlated)")
    h.add_argument("--correlated-block", default=None, dest="correlated_block",
                   metavar="DTAU,DX,DETA", help="its block size (hadronize.py)")
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


#: MUSIC's hotQCD tables per <EOS> (music/src/eos_hotQCD.cpp): rows (e, P, s, T), e in
#: GeV/fm^3, T in GeV, float64
EOS_TABLES = {"9": "hrg_hotqcd_eos_binary.dat", "91": "hrg_hotqcd_eos_SMASH_binary.dat"}


def music_eos(root, main_xml: str) -> str:
    """MUSIC's <EOS> as the job runs it: the first <Hydro><MUSIC> block's, else the main
    XML's."""
    for r in (root, ET.parse(main_xml).getroot()):
        v = _text(r, "Hydro/MUSIC/EOS")
        if v:
            return v
    return ""


def freezeout_e(root, main_xml: str, build: str):
    """MUSIC's freeze-out energy density [GeV/fm^3] and where it came from, from the first
    <Hydro><MUSIC> block (the main XML's defaults for tags it lacks).  (None, reason) if it
    cannot be determined here (an EOS without a table lookup below)."""
    main = ET.parse(main_xml).getroot()

    def val(tag):
        for r in (root, main):
            v = _text(r, f"Hydro/MUSIC/{tag}")
            if v:
                return v
        return None

    if val("use_eps_for_freeze_out") == "1":
        return float(val("eps_switch")), "eps_switch"
    eos, T_fo = val("EOS"), float(val("freezeout_temperature"))
    if eos not in EOS_TABLES:
        return None, f"EOS {eos}: no table lookup for it here (pass --edge-e-threshold)"
    import numpy as np
    path = os.path.join(build, "EOS", "hotQCD", EOS_TABLES[eos])
    if not os.path.exists(path):
        return None, f"no EOS table {path} (pass --edge-e-threshold)"
    tab = np.fromfile(path, dtype="<f8").reshape(-1, 4)
    # MUSIC interpolates the table linearly in e, so e(T) is the inverse interpolation
    return (float(np.interp(T_fo, tab[:, 3], tab[:, 0])),
            f"e(T_fo = {T_fo:g} GeV), EOS {eos}")


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


#: --trigger-ymax / --lepton-ptmin / --lepton-etamax -> PythiaGun's XML element
TRIGGER_CUTS = {"trigger_ymax": "triggerYMax", "lepton_ptmin": "leptonPtMin",
                "lepton_etamax": "leptonEtaMax"}


def check_trigger(a) -> None:
    """--trigger and its cuts against the other options (that a cut has trigger Z is
    checked in job_xml, where the template's <trigger> is known)."""
    given = [k for k in TRIGGER_CUTS if getattr(a, k) is not None]
    if a.hard != "pythia" and (given or a.trigger == "Z"):
        raise ValueError("--trigger Z and its cuts are for --hard pythia")
    for k in given:
        if not getattr(a, k) > 0:
            raise ValueError(f"--{k.replace('_', '-')} must be > 0, got {getattr(a, k):g}")


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
    eCMforHadronization is the job's collision energy (jetscape.hadronizer.collision_ecm), as
    hadronize.py sets it from the stored job XML.
    """
    from jetscape import hadronizer as hz

    src = ET.parse(hadronize_xml).getroot()
    soft, had = src.find("SoftParticlization"), src.find("JetHadronization")
    if soft is None or had is None:
        sys.exit(f"run_prod_jet.py: {hadronize_xml} needs <SoftParticlization> and "
                 "<JetHadronization>")
    ecm = hz.collision_ecm(root)
    if ecm:
        hz.set_hadronization_ecm(src, ecm)
    soft, had = copy.deepcopy(soft), copy.deepcopy(had)
    _set_text(soft, "hydro_id", JET_ID)          # compared verbatim with the module id
    _set_text(soft.find("iSS"), "iSS_working_path", ".")
    _set(had, "reseed_per_event", 1)
    root.append(had)
    root.append(soft)


def configure_pythia_gun(a, pg) -> list:
    """This job's settings in a <PythiaGun> block, in place: the pTHat range or windows, the
    parton rapidity cut, the trigger and its cuts (a template's windows and cuts only with
    the options).  Sets ``a.trigger_kind`` and ``a.min_partons`` (the block's <trigger>,
    <minPartons>) -> the problems found.  Also run_prod_vacuum.py's."""
    problems = []
    if a.pthat_min is not None:
        _set(pg, "pTHatMin", a.pthat_min)
    if a.pthat_max is not None:
        _set(pg, "pTHatMax", a.pthat_max)
    if a.pthat_windows:
        _set(pg, "pTHatBins", pthat_bins_text(a.pthat_windows))
    elif pg.find("pTHatBins") is not None:
        pg.remove(pg.find("pTHatBins"))          # a template's windows: only with --pthat-bins
    if a.parton_ymax is not None:
        _set(pg, "partonYMax", f"{a.parton_ymax:g}")
        _set(pg, "partonYMode", a.parton_y_mode)
    else:                                        # a template's cut: only with --parton-ymax
        for tag in ("partonYMax", "partonYMode"):
            if pg.find(tag) is not None:
                pg.remove(pg.find(tag))
    if a.trigger is not None:
        _set_text(pg, "trigger", a.trigger)
    for k, tag in TRIGGER_CUTS.items():
        if getattr(a, k) is not None:
            _set(pg, tag, f"{getattr(a, k):g}")
        elif pg.find(tag) is not None:           # a template's cut: only with the option
            pg.remove(pg.find(tag))
    a.trigger_kind = _text(pg, "trigger") or "none"
    a.min_partons = _text(pg, "minPartons")
    if a.trigger_kind not in ("none", "Z"):
        problems.append(f"<PythiaGun><trigger> is none or Z, not {a.trigger_kind!r}")
    elif a.trigger_kind != "Z" and any(getattr(a, k) is not None for k in TRIGGER_CUTS):
        problems.append("--trigger-ymax / --lepton-ptmin / --lepton-etamax need trigger Z "
                        "(<trigger>Z</trigger> in <PythiaGun>, or --trigger Z)")
    return problems


def job_xml(a, out_h5: str):
    """Check the template, apply this job's settings, write it next to the output."""
    tree = ET.parse(a.user_xml)
    root = tree.getroot()
    problems = []
    a.trigger_kind, a.min_partons = "none", None     # PythiaGun's <trigger>, <minPartons>

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
        problems += configure_pythia_gun(a, pg)
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
        _set(music, "slim_bulk_info", 1 if a.bulk_info == "slim" else 0)
        if a.eos is not None:
            # every MUSIC instance reads the first block; a second block's own <EOS> too
            for h in hydros:
                if h.find("MUSIC") is not None and (h is hydros[0]
                                                    or h.find("MUSIC/EOS") is not None):
                    _set(h.find("MUSIC"), "EOS", a.eos)
        if a.late_dtau is not None:
            _set(music, "late_delta_tau", a.late_dtau)
        if a.late_tau is not None:
            _set(music, "late_tau_switch", a.late_tau)
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
                   "tau0_music", "ntau_jet", "ntau_jet_music", "ntau_bg", "frames_identical",
                   "jet_hit_boundary", "bg_hit_boundary", "jet_hit_edge", "bg_hit_edge",
                   "jet_edge_e_max", "bg_edge_e_max", "jet_edge_e_max_eta",
                   "bg_edge_e_max_eta", "pthat_bin", "pthat",
                   "event_weight", "parton_y_lead", "parton_pt_lead", "parton_y_sub",
                   "parton_pt_sub", "n_coll")


def check_bulk_info(a, jetscape):
    """--bulk-info: the background leg runs it (a build without <slim_bulk_info> would
    ignore the XML tag and keep the full copy)."""
    bg = {t.GetId(): t for t in jetscape.GetTaskList()}.get(BG_ID)
    get = getattr(bg, "get_slim_bulk_info", None)
    if get is None:
        if a.bulk_info == "slim":
            sys.exit("run_prod_jet.py: --bulk-info slim needs X-SCAPE with "
                     "<Hydro><MUSIC><slim_bulk_info> (branch slim_bulk_info) and PyJetscape "
                     "built against it; use --bulk-info full with this build")
        return
    if get() != (a.bulk_info == "slim"):
        sys.exit(f"run_prod_jet.py: {BG_ID} runs slim_bulk_info = {int(get())}, not "
                 f"--bulk-info {a.bulk_info}")
    print(f"  bulk     {BG_ID} -> framework: {a.bulk_info} copy"
          + (" (e, s, T, vx, vy, vz)" if a.bulk_info == "slim" else ""))


def open_pthat_bins(a, jetscape):
    """--pthat-bins / --parton-ymax / trigger Z / <minPartons>: the PythiaGun task, after
    checking that it runs exactly these windows, cuts and trigger.

    A PythiaGun without <pTHatBins>, <partonYMax> or <trigger> would refuse the XML (no
    default in its main XML) or, built otherwise, ignore it, so the build is checked here
    rather than trusted.
    """
    z_gun = a.trigger_kind != "none" or a.min_partons is not None
    if not a.pthat_windows and a.parton_ymax is None and not z_gun:
        return None
    from jetscape import pyjetscape_core as core

    if not getattr(core, "PYTHIA_GUN_HAS_PTHAT_BINS", False):
        sys.exit("run_prod_jet.py: --pthat-bins / --parton-ymax need X-SCAPE's PythiaGun "
                 "with <pTHatBins> (branch N_ptHat_per_hydro) and PyJetscape built against it")
    if a.parton_ymax is not None and not getattr(core, "PYTHIA_GUN_HAS_PARTON_Y_CUT", False):
        sys.exit("run_prod_jet.py: --parton-ymax needs X-SCAPE's PythiaGun with <partonYMax> "
                 "(branch pyGun_eta_cut) and PyJetscape built against it")
    if z_gun and not getattr(core, "PYTHIA_GUN_HAS_Z_TRIGGER", False):
        sys.exit("run_prod_jet.py: <trigger> / <minPartons> need X-SCAPE's PythiaGun with "
                 "the Z trigger (branch zjet) and PyJetscape built against it")
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
    if z_gun:
        want = {"trigger": a.trigger_kind,
                "min_partons": int(a.min_partons) if a.min_partons is not None else 2,
                **{f: getattr(a, k) or 0.0 for k, f in (("trigger_ymax", "trigger_y_max"),
                                                        ("lepton_ptmin", "lepton_pt_min"),
                                                        ("lepton_etamax", "lepton_eta_max"))}}
        got_z = {k: info.get(k) for k in want}
        if any(got_z[k] != v if isinstance(v, (str, int)) else abs(got_z[k] - v) > 1e-12
               for k, v in want.items()):
            sys.exit(f"run_prod_jet.py: PythiaGun runs {got_z}, not {want}")
    print("  pTHat    " + ", ".join(f"window {k}: {lo:g}-{hi:g} GeV (Pythia seed {s})"
                                    for k, ((lo, hi), s) in enumerate(zip(got,
                                                                          info["seeds"]))))
    if a.parton_ymax is not None:
        print(f"  cut      handed-over partons, two hardest, |y| < {a.parton_ymax:g} "
              f"({a.parton_y_mode})")
    if z_gun:
        print(f"  trigger  {trigger_desc(a)}; events need "
              f"{info['min_partons']} parton(s) (leptons do not count)")
    return gun


def trigger_desc(a) -> str:
    """'Z (|y_Z| < 2.4, lepton pT > 20 GeV, |eta| < 2.4)', or 'none'."""
    if a.trigger_kind != "Z":
        return a.trigger_kind
    cuts = ([f"|y_Z| < {a.trigger_ymax:g}"] if a.trigger_ymax is not None else []) \
        + ([f"lepton pT > {a.lepton_ptmin:g} GeV"] if a.lepton_ptmin is not None else []) \
        + ([f"lepton |eta| < {a.lepton_etamax:g}"] if a.lepton_etamax is not None else [])
    return "Z" + (f" ({', '.join(cuts)})" if cuts else "")


def pthat_diag(gun, trigger="none") -> dict:
    """The current event's window, pTHat and weight (and with trigger Z the Z), for diag/
    (and events/)."""
    if gun is None:
        return {}
    import numpy as np
    from jetscape import pyjetscape_core as core

    h = core.hard_process_info(gun)
    d = {"pthat_bin": int(core.pythia_gun_bins(gun)["active"]),
         "pthat": float(h["pthat"]), "event_weight": float(h["event_weight"])}
    if hasattr(core, "hard_partons_numpy"):
        # the partons handed to the framework (what --parton-ymax cuts on), hardest first:
        # pid, px, py, pz, E, pT, y; leptons (a trigger's) are not partons
        hp = core.hard_partons_numpy(gun)
        hp = hp[~np.isin(np.abs(hp[:, 0]), LEPTON_PIDS)]
        if len(hp):
            d.update(parton_y_lead=float(hp[0, 6]), parton_pt_lead=float(hp[0, 5]))
            if len(hp) > 1:
                d.update(parton_y_sub=float(hp[1, 6]), parton_pt_sub=float(hp[1, 5]))
    if trigger == "Z":
        from jetscape.hadrons_h5 import z_record
        d.update(z_record(core.pythia_gun_trigger(gun)["leptons"]))
    return d


def open_initial_state(jetscape):
    """The initial-state task (3dMCGlauber) whose n_coll goes into the event record, or None
    (a build without initial_state_info, or no such task)."""
    from jetscape import pyjetscape_core as core

    if not hasattr(core, "initial_state_info"):
        return None
    for task in jetscape.GetTaskList():
        try:
            core.initial_state_info(task)
        except (ValueError, TypeError):
            continue
        return task
    return None


def geometry_diag(ini) -> dict:
    """The event's n_coll, for diag/ (and events/, units/): a hard process scales with it, so
    it weights the events of a wide centrality class (sampled with P(b) ~ b, one hard process
    each) like the data.  Shared by the jets of a reused background."""
    if ini is None:
        return {}
    from jetscape import pyjetscape_core as core

    n = core.initial_state_info(ini)["n_coll"]
    return {"n_coll": int(n)} if n >= 0 else {}


#: |pid| of the leptons PythiaGun hands over (a Z's decay; Parton::isLepton)
LEPTON_PIDS = (11, 13)


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
    if info.get("y_max", 0) > 0 or info.get("trigger", "none") != "none":
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


#: --hadrons-only options: value options (None when not given) and flags
HADRON_VALUE_OPTS = ("hadronize_seed", "oversample", "oversample_bg", "oversample_bg_max",
                     "n_frag", "keep_primaries", "hadron_keep_bits_p", "hadron_keep_bits_x",
                     "eta_max", "correlated_block")
HADRON_FLAG_OPTS = ("charged", "no_x", "correlated")


def check_hadrons_only(a) -> None:
    """--hadrons-only: its defaults, and what it cannot be combined with (sys.exit)."""
    from jetscape import hadronizer as hz

    if not a.hadrons_only:
        given = ([k for k in HADRON_VALUE_OPTS if getattr(a, k) is not None]
                 + [k for k in HADRON_FLAG_OPTS if getattr(a, k)])
        if given:
            sys.exit("run_prod_jet.py: " + ", ".join("--" + k.replace("_", "-") for k in given)
                     + (" needs" if len(given) == 1 else " need")
                     + " --hadrons-only (or hadronize.py on a particlize file)")
        return
    for bad, what in ((a.validate_inline, "--validate-inline"),
                      (a.particlize_only, "--particlize-only"),
                      (a.write_particlize != "none", "--write-particlize")):
        if bad:
            sys.exit(f"run_prod_jet.py: --hadrons-only writes no particlize file: drop {what}")
    a.write_particlize, a.particlize_only = "both", True   # both surfaces, no pair file
    a.hadronize_seed = 1 if a.hadronize_seed is None else a.hadronize_seed
    a.oversample = 1 if a.oversample is None else a.oversample
    a.n_frag = 1 if a.n_frag is None else a.n_frag
    a.oversample_bg_max = 2000 if a.oversample_bg_max is None else a.oversample_bg_max
    if a.correlated:
        # iSS's correlated sampling pairs sample k of a jet leg with sample k of its
        # background only when both draw the same number of samples: with more background
        # samples the legs come out uncorrelated (hadronize.py refuses it too)
        if a.oversample_bg is not None and str(a.oversample_bg) != str(a.oversample):
            sys.exit(f"run_prod_jet.py: --correlated needs as many background samples as jet "
                     f"samples (--oversample {a.oversample}): drop --oversample-bg "
                     f"{a.oversample_bg}")
        a.oversample_bg = str(a.oversample)
    elif a.oversample_bg is None:
        a.oversample_bg = "per-pthat-bin" if a.pthat_windows else "auto"
    if str(a.oversample_bg).lower() == "per-pthat-bin" and not a.pthat_windows:
        sys.exit("run_prod_jet.py: --oversample-bg per-pthat-bin needs --pthat-bins")
    if a.oversample < 1 or a.n_frag < 1:
        sys.exit("run_prod_jet.py: --oversample and --n-frag must be >= 1")
    try:
        hz.samples_per_background(a.oversample, a.oversample_bg, 1, a.oversample_bg_max)
    except ValueError as err:
        sys.exit(f"run_prod_jet.py: {err}")
    kp = "all" if a.keep_primaries is None else str(a.keep_primaries).lower()
    try:
        a.keep_primaries_n = hz.KEEP_ALL_PRIMARIES if kp == "all" else int(kp)
    except ValueError:
        a.keep_primaries_n = -1
    if a.keep_primaries_n < 0:
        sys.exit(f"run_prod_jet.py: --keep-primaries takes N >= 0 or 'all', got {kp!r}")
    a.keep_primaries = kp
    for k in ("hadron_keep_bits_p", "hadron_keep_bits_x"):
        v = getattr(a, k)
        if v is not None and not 1 <= v <= 23:
            sys.exit(f"run_prod_jet.py: --{k.replace('_', '-')} must be 1..23")
    if a.eta_max is not None and not a.eta_max > 0:
        sys.exit("run_prod_jet.py: --eta-max must be > 0")
    if a.correlated_block is not None and not a.correlated:
        sys.exit("run_prod_jet.py: --correlated-block needs --correlated")


def hadronize_xml_for_job(a):
    """--hadrons-only: hadronize.xml with this job's settings, eCMforHadronization the job's
    collision energy (``a.ecm``) -> (its text, recorded as the hadron files' hadronize_xml,
    and the <SoftParticlization> + <JetHadronization> blocks the modules read, as a
    <jetscape> document)."""
    from jetscape import hadronizer as hz

    root = ET.parse(a.hadronize_xml).getroot()
    soft, had = root.find("SoftParticlization"), root.find("JetHadronization")
    if soft is None or soft.find("iSS") is None or had is None:
        sys.exit(f"run_prod_jet.py: {a.hadronize_xml} needs <SoftParticlization><iSS> and "
                 "<JetHadronization>")
    try:
        hz.configure_hadronize_xml(root, xscape=rp.XSCAPE, seed=a.hadronize_seed, n_events=1,
                                   oversample=a.oversample, keep_primaries=a.keep_primaries_n,
                                   correlated=a.correlated,
                                   correlated_block=a.correlated_block)
    except ValueError as err:
        sys.exit(f"run_prod_jet.py: {err}")
    if a.ecm:
        hz.set_hadronization_ecm(root, a.ecm)
    blocks = ("<jetscape>" + ET.tostring(soft, encoding="unicode")
              + ET.tostring(had, encoding="unicode") + "</jetscape>")
    return ET.tostring(root, encoding="unicode"), blocks


class HadronsOnly:
    """--hadrons-only: the particlize writer's memory sink -> jetscape.hadronizer, per event,
    with hadronize.py's seeds and settings (docs/Plans/PLAN_AuAu_hadrons_only.md)."""

    #: units/ keys the Hadronizer sets itself (not taken from the event record)
    RESERVED = ("unit", "event", "seed", "n_cells", "n_partons", "n_samples")

    def __init__(self, a, stem, pwriter, provenance, xml_text, blocks):
        from jetscape import hadronizer as hz
        from jetscape import pyjetscape_core as core
        from jetscape.hadrons_h5 import PRIMARY_OF

        for cap in ("SOFT_HAS_SAMPLE_CELLS", "HAS_NEXT_TASK_NUMBER"):
            if not getattr(core, cap, False):
                sys.exit(f"run_prod_jet.py: --hadrons-only needs PyJetscape with {cap} "
                         "(X-SCAPE auau_hadrons_only or later)")
        self.a, self.hz = a, hz
        # The modules are made after JetScape.Init and outside its task list; the task
        # counter is set back afterwards, so every later task (the per-event jet-loss
        # clones) gets the ID, and so the random seed, it would have had without them.
        n_task = core.next_task_number()
        core.xml_add_user_blocks(blocks)
        iss = core.create_module("iSS")
        iss.Init()
        colorless = core.create_module("ColorlessHadronization")
        colorless.Init()
        core.set_next_task_number(n_task)

        uuid = str(pwriter.attrs["file_uuid"])
        self.h = hz.Hadronizer(core, iss=iss, colorless=colorless, base_seed=a.hadronize_seed,
                               prod=hz.uuid_key(uuid), common_seeds=a.correlated)
        eos = str(provenance.get("eos", ""))
        decay = hz.iss_decay_attrs(core, eos, ET.fromstring(xml_text), rp.XSCAPE)
        if eos == "91" and not decay["decayed"]:
            print("run_prod_jet.py: WARNING -- EOS 91 and this iSS does not decay the SMASH "
                  "hadron list (<decay_smash_list> 0 in --hadronize-xml?)", file=sys.stderr)
        common = {"source": os.path.basename(stem), "source_uuid": uuid,
                  "hadronize_xml": xml_text, "base_seed": a.hadronize_seed,
                  "stored_seeds": False, "seed_scheme": "production_file",
                  "music_input": pwriter.attrs.get("music_input", ""), "hadrons_only": True,
                  "legs": ["jet", "bg"],
                  **{k: v for k, v in provenance.items()}}
        keep_bits = {"p": a.hadron_keep_bits_p, "x": a.hadron_keep_bits_x}
        per_unit = str(a.oversample_bg).lower() in hz.PER_UNIT_OVERSAMPLE_BG
        self.outs = {}
        for t in hz.TAGS:
            extra = hz.tag_attrs(t, core, decay, common_seeds=a.correlated,
                                 correlated=a.correlated)
            if t == "jet_frag":
                n = a.n_frag
                extra["x_had"] = "none"
            elif t == "bulk_bg":
                n = 0 if per_unit else int(a.oversample_bg)
                extra.update({"oversample_bg": str(a.oversample_bg),
                              "oversample_bg_max": int(a.oversample_bg_max),
                              "oversample_jet": int(a.oversample)})
            else:
                n = a.oversample
            self.outs[t] = f"{stem}_hadrons_{t}.h5"
            ppath = (f"{stem}_hadrons_{PRIMARY_OF[t]}.h5"
                     if t in PRIMARY_OF and a.keep_primaries_n > 0 else None)
            if ppath:
                self.outs[PRIMARY_OF[t]] = ppath
            self.h.open(t, self.outs[t], n_samples=n, attrs=dict(common, **extra),
                        keep_bits=keep_bits, eta_max=a.eta_max, charged_only=a.charged,
                        positions=not a.no_x, initiators=t != "bulk_bg",
                        partons=t == "jet_frag", primary_tag=PRIMARY_OF.get(t),
                        primary_path=ppath, keep_primaries=a.keep_primaries_n)
        self.records = []
        self.seconds = 0.0
        self.n_capped = 0
        if a.pthat_windows:
            self.per_bg = len(a.pthat_windows) * a.jets_per_bin
        else:
            self.per_bg = max(1, a.reuse)
        print(f"  hadrons  in the job: iSS {a.oversample} sample(s) per jet event, "
              f"background {a.oversample_bg} (max {a.oversample_bg_max}), {a.n_frag} "
              f"fragmentation(s), primaries: {a.keep_primaries}, base seed "
              f"{a.hadronize_seed}, EOS {eos} ({decay['hadron_list']} list, decays: "
              f"{decay['iss_decays']})")

    def sink(self, record):
        """The particlize writer's memory sink: keep the event until it is hadronized."""
        self.records.append(record)

    def _background_uses(self, i_loop):
        """Events that will use the background starting at loop event ``i_loop``
        (hadronize.py counts them in the particlize file; the same unless events are
        skipped)."""
        a = self.a
        left = min(self.per_bg, a.events - i_loop)
        if str(a.oversample_bg).lower() == "per-pthat-bin":
            k = len(a.pthat_windows)
            return min(a.jets_per_bin, -(-left // k))
        return left

    def event(self, i_loop):
        """Hadronize the event the sink got last (after ClearPerEvent) -> a log line."""
        import time as _time

        t0 = _time.time()
        a, h, hz = self.a, self.h, self.hz
        rec = self.records.pop()
        e, bg_unit = int(rec["event"]), int(rec["bg_unit"])
        scal = {k: v for k, v in rec["row"].items() if k not in self.RESERVED}
        had, _ = h.bulk("bulk_jet", e, rec["jet"], h.bulk_jet_seed(e, bg_unit), a.oversample,
                        event=e, initiators=rec["initiators"],
                        keep_primaries=a.keep_primaries_n, **scal)
        msg = f"bulk_jet {0 if had is None else len(had['pid'])}"
        if rec["bg"] is not None:
            n, capped = hz.samples_per_background(a.oversample, a.oversample_bg,
                                                  self._background_uses(i_loop),
                                                  a.oversample_bg_max)
            self.n_capped += capped
            bg, _ = h.bulk("bulk_bg", bg_unit, rec["bg"], h.seed("bulk_bg", bg_unit), n,
                           event=e, keep_primaries=a.keep_primaries_n)
            msg += f", bulk_bg (unit {bg_unit}, {n} samples) {0 if bg is None else len(bg['pid'])}"
        frags = h.fragments(e, rec["partons"], [h.seed("jet_frag", e, s)
                                                for s in range(a.n_frag)],
                            initiators=rec["initiators"], **scal)
        dt = _time.time() - t0
        self.seconds += dt
        return (msg + f", jet_frag {sum(len(f['pid']) for f in frags) / a.n_frag:.1f}"
                f"/fragmentation, hadronized in {dt:.1f} s")

    def finish(self, complete, attrs):
        self.h.set_attrs(dict(attrs, prod_hadronize_s_total=round(self.seconds, 2)))
        self.h.close(complete=complete)


def restart_plan(a) -> tuple:
    """Whether MUSIC_2 starts from MUSIC_1's state (--music2-restart, the default), from
    the options and the build alone, so the files record it.  -> (on, why not)."""
    if not a.music2_restart:
        return False, "--no-music2-restart"
    legs = surface_legs(a)
    if "jet" in legs and "bg" not in legs:
        # the jet leg's surface up to the checkpoint would come from MUSIC_1, which
        # builds none: every jet leg would run from tau0 anyway
        return False, (f"{JET_ID} builds the freeze-out surface and {BG_ID} does not "
                       "(--write-particlize jet / --surface jet), so the checkpoint would "
                       "lack it; --write-particlize both or none restarts")
    from jetscape import pyjetscape_core as core
    if not hasattr(core.MpiMusic, "set_restart_from"):
        return False, ("this build has no MpiMusic.set_restart_from (X-SCAPE and "
                       "music4gpu before the restart)")
    return True, ""


def restart_legs(a, jetscape, writer):
    """MUSIC_1 keeps a checkpoint at the liquefier's tau_delay, MUSIC_2 starts from it
    (when restart_plan said so).  -> MUSIC_2, or None."""
    if not a.restart_on:
        print(f"  restart  OFF: {a.restart_why}")
        return None
    tasks = {t.GetId(): t for t in jetscape.GetTaskList()}
    bg, jet = tasks[BG_ID], tasks[JET_ID]
    tau = writer.liquefier_tau_delay
    if not tau:
        sys.exit("run_prod_jet.py: the restart needs the liquefier (its tau_delay sets "
                 "where MUSIC_1 keeps its checkpoint); run with --no-music2-restart")
    if jet.get_freeze_out_surface() and not bg.get_freeze_out_surface():
        sys.exit(f"run_prod_jet.py: {JET_ID} builds the freeze-out surface and {BG_ID} "
                 "does not, which restart_plan did not expect; run with "
                 "--no-music2-restart")
    bg.set_checkpoint_tau(tau)
    jet.set_restart_from(bg)
    print(f"  restart  {JET_ID} starts from {BG_ID}'s state at tau <= {tau:g} fm/c "
          "(the liquefier's tau_delay)"
          + (", MUSIC_RESTART_VERIFY: from tau0, compared"
             if os.environ.get("MUSIC_RESTART_VERIFY") == "1" else ""))
    return jet


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
        check_trigger(a)
    except ValueError as exc:
        sys.exit(f"run_prod_jet.py: {exc}")
    check_hadrons_only(a)          # --hadrons-only: both surfaces, no pair file, a sink
    if a.validate_inline and a.write_particlize == "none":
        sys.exit("run_prod_jet.py: --validate-inline compares with the stored surfaces and "
                 "partons; add --write-particlize jet (or both)")
    if a.particlize_only and a.write_particlize == "none":
        sys.exit("run_prod_jet.py: --particlize-only keeps only the particlize file; add "
                 "--write-particlize jet (or both)")
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
    from jetscape import hadronizer as hz
    a.ecm = hz.collision_ecm(root)     # Colorless's eCMforHadronization: this collision's
    hxml = hadronize_xml_for_job(a) if a.hadrons_only else None
    box = rp.music_box(root, a.main_xml)
    grid_mode = "native" if a.native else "grid"
    if not a.native:
        rp.check_inside(grid, box)
    if a.edge_e_threshold is None:
        a.edge_e_threshold, edge_src = freezeout_e(root, a.main_xml, a.build)
    else:
        edge_src = "--edge-e-threshold"
    if not a.edge_e_threshold:                      # 0 or not determined: no flag
        edge_src = edge_src if a.edge_e_threshold is None else "off (--edge-e-threshold 0)"
        a.edge_e_threshold = None

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
    if a.trigger_kind != "none":
        hard_desc += f", trigger {trigger_desc(a)}"
    print(f"prod_AuAu_0_10_jet: {a.events} event(s), seed {a.seed}"
          + (" (from OS entropy)" if a.seed_source == "os_entropy" else "")
          + (f", campaign {a.campaign} #{a.index}" if a.campaign else "")
          + f", grid_mode {grid_mode}, "
          f"{hard_desc}, reuse {a.reuse}, deposition {'OFF (null test)' if a.no_deposit else 'on'}, "
          f"freeze-out surface: {'+'.join(sorted(surface_legs(a))) or 'none'}"
          + (f", particlize input: {a.write_particlize}" if particlize_legs(a) else "")
          + (" (no pair file)" if a.particlize_only and not a.hadrons_only else "")
          + (" + inline iSS/Colorless (validation)" if a.validate_inline else "")
          + (", HADRONS ONLY (no pair or particlize file)" if a.hadrons_only else ""))
    print(f"  build    {a.build}")
    print(f"  job XML  {xml}")
    print(f"  grid     {a.grid}")
    if a.native:
        print("  grid     MUSIC native: " + ", ".join(
            f"{ax} {b[0]:g} .. {b[1]:g} (n = {b[2]})" for ax, b in box.items())
            + f"; max_ntau = {max_ntau or 'auto'}")
    else:
        print(rp.describe(grid, max_ntau))
    print("  edge     " + (f"flag e > {a.edge_e_threshold:.4f} GeV/fm^3 on the x/y edge of "
                             f"the output grid ({edge_src})" if a.edge_e_threshold is not None
                             else f"no flag: {edge_src}"))
    if not a.particlize_only:
        print(f"  output   {out_h5}")
    stem = os.path.splitext(out_h5)[0]
    out_particlize = (stem + "_particlize.h5"
                      if particlize_legs(a) and not a.hadrons_only else None)
    if out_particlize:
        print(f"  output   {out_particlize}")
    if a.hadrons_only:
        print(f"  output   {stem}_hadrons_{{bulk_jet,bulk_bg,jet_frag}}.h5"
              + (" + *_primary.h5" if a.keep_primaries_n > 0 else ""))
    if a.hadrons_only or a.validate_inline:
        print(f"  hadrons  {a.hadronize_xml}"
              + (f", eCMforHadronization {a.ecm:g} (the collision's)" if a.ecm else ""))
    if a.dry_run:
        return 0

    import h5py
    import numpy as np
    import jetscape as js
    from jetscape.hadrons_h5 import Z_COLUMNS
    from jetscape.pair_h5 import PairH5Writer
    # diag/ values the particlize file's events/ (and the hadron files' units/) get
    particlize_diag = PARTICLIZE_DIAG + (Z_COLUMNS if a.trigger_kind == "Z" else ())
    a.restart_on, a.restart_why = restart_plan(a)

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
        "prod_bulk_info": a.bulk_info,
        **({"prod_particlize_only": True, "prod_jet_leg_read": False}
           if a.particlize_only else {}),
        **({"prod_hadrons_only": True} if a.hadrons_only else {}),
        "prod_music2_restart": a.restart_on,
        **({"pthat_bins": np.asarray(a.pthat_windows, dtype=np.float64),
            "pthat_jets_per_bin": a.jets_per_bin} if a.pthat_windows else {}),
        **({"parton_ymax": a.parton_ymax, "parton_y_mode": a.parton_y_mode}
           if a.parton_ymax is not None else {}),
        **({"trigger": a.trigger_kind,
            **{k: getattr(a, k) for k in TRIGGER_CUTS if getattr(a, k) is not None}}
           if a.trigger_kind != "none" else {}),
        **({"min_partons": int(a.min_partons)} if a.min_partons is not None else {}),
        "system": rp.collision_system(root, a.main_xml),
        "initial_state_kind": "3dMCGlauber_strings",
        "hydro": "MUSIC (music4gpu)" if "gpu" in os.path.basename(a.build) else "MUSIC",
        "jet": "Matter + LBT, CausalLiquefier droplets into MUSIC_2",
        "T_fo": rp.music_freezeout_temperature(root, a.main_xml),
        "eos": music_eos(root, a.main_xml),
    }
    writer = PairH5Writer(
        out_h5, bg_id=BG_ID, jet_id=JET_ID, grid_mode=grid_mode,
        out_grid=None if a.native else grid, choose_ntau=max_ntau,
        store_showers=not a.no_showers, compression=a.compression, keep_bits=a.keep_bits,
        provenance={"hard_vertex": HARD_VERTEX[a.hard],
                    "eos_kind": f"hotqcd (MUSIC EOS {provenance['eos']})",
                    "transport_mode": "MUSIC viscous: eta/s(T) and zeta/s(T) "
                                      "parametrization 3, second-order terms"},
        keep_surface=particlize_legs(a), extra_attrs=provenance, verbose=True,
        bg_layout=a.bg_layout, reuse=a.reuse, store_evolution=not a.particlize_only,
        read_jet=not a.particlize_only, edge_e_threshold=a.edge_e_threshold)
    pwriter = None
    records = []                       # --hadrons-only: the particlize writer's memory sink
    if out_particlize or a.hadrons_only:
        from jetscape.particlize_h5 import ParticlizeH5Writer
        pwriter = ParticlizeH5Writer(
            out_particlize, legs=particlize_legs(a), bg_id=BG_ID, jet_id=JET_ID,
            T_fo=provenance["T_fo"], pair_file=None if a.particlize_only else out_h5,
            extra_attrs={k: v for k, v in provenance.items() if k != "T_fo"},
            verbose=True, sink=records.append if a.hadrons_only else None)

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
    check_bulk_info(a, jetscape)
    writer.attach(jetscape)                  # after Init: MUSIC_2 -> dump_hydro_only
    if pwriter is not None:
        pwriter.attach(jetscape)             # reads ./music_input: the job's own
    jet_leg = restart_legs(a, jetscape, writer)  # after both: their surface settings
    inline = open_inline(a, jetscape, stem) if a.validate_inline else None
    gun = open_pthat_bins(a, jetscape)
    ini = open_initial_state(jetscape)
    hadr = None
    if a.hadrons_only:
        hadr = HadronsOnly(a, stem, pwriter, provenance, *hxml)
        hadr.records = records

    late = 0
    droplets = e_late = 0.0
    t_job = time.time()
    finished = False
    windows = None
    try:
        jetscape.ExecInit()
        for i in range(jetscape.GetNumberOfEvents()):
            t0 = time.time()
            jetscape.ExecPerEvent()
            idx = writer.Exec(**pthat_diag(gun, a.trigger_kind),   # both legs are live
                              **geometry_diag(ini))
            if idx is not None and pwriter is not None:
                d = writer.last_event_diag
                pwriter.Exec(idx, bg_id=d["bg_id"], bg_key=writer.last_bg_key,
                             **{k: d[k] for k in particlize_diag if k in d})
                if inline is not None:
                    pwriter.write_events(idx, **inline.capture(idx))
            jetscape.ClearPerEvent()
            wall = time.time() - t0
            if idx is None:
                print(f"prod_AuAu_0_10_jet: event {i + 1}/{a.events} SKIPPED (see warning)")
                continue
            writer.write_diag(idx, wall_s=wall)
            if a.particlize_only:                # the only per-event record of it
                pwriter.write_events(idx, wall_s=wall)
            hmsg = hadr.event(i) if hadr is not None else None
            d = writer.last_event_diag
            droplets += d["n_droplets"]
            e_late += d["E_droplets_late"]
            frames = (f"jet/bg frames {d['ntau_jet']}/{d['ntau_bg']}" if "ntau_jet" in d
                      else f"jet {d['ntau_jet_music']} MUSIC steps, bg {d['ntau_bg']} frames")
            msg = (f"prod_AuAu_0_10_jet: event {i + 1}/{a.events} done in {wall:.1f} s, "
                   f"MUSIC tau0 = {d['tau0_music']:.3f} fm/c, {frames}, {d['n_droplets']} "
                   f"droplets ({d['E_droplets']:.1f} GeV, {d['E_droplets_late']:.1f} after "
                   f"freeze-out), "
                   + (f"{d['frames_identical']} leading frames identical, "
                      if "frames_identical" in d else "")
                   + f"bg {d['bg_id']}"
                   + (f", N_coll {d['n_coll']}" if "n_coll" in d else ""))
            if not a.native and d["tau0_music"] > grid.tau_min + rp._TOL:
                late += 1
                msg += (f"  WARNING: tau0 after tau.min = {grid.tau_min:g}, so the first "
                        "frame(s) of this event are zeros")
            if jet_leg is not None:
                msg += f", MUSIC_2 {jet_leg.get_restart_status()}"
            if gun is not None:
                msg += f", pTHat {d['pthat']:.1f} GeV (window {d['pthat_bin']})"
                if "parton_y_lead" in d:
                    msg += f", hardest parton y {d['parton_y_lead']:+.2f}"
                if "z_pt" in d:
                    msg += f", Z pT {d['z_pt']:.1f} GeV y {d['z_y']:+.2f}"
            print(msg)
            if hmsg:
                print(f"prod_AuAu_0_10_jet: event {i + 1}/{a.events} hadrons: {hmsg}")
        windows = None if gun is None else pthat_windows_final(gun)
        jetscape.Finish()
        finished = True
    finally:
        writer.Finish()
        if pwriter is not None:
            pwriter.Finish(complete=pwriter.GetNumberOfEventsWritten()
                           == writer.GetNumberOfEventsWritten())
        if inline is not None:
            inline.close()
        if hadr is not None:
            # complete only if every event went through (as hadronize.py's files)
            hadr.finish(complete=finished and pwriter.GetNumberOfEventsWritten()
                        == writer.GetNumberOfEventsWritten(),
                        attrs={**(windows or {}), "prod_wall_s_total": time.time() - t_job,
                               "nevents": pwriter.GetNumberOfEventsWritten(),
                               "complete_job": finished})

    n = writer.GetNumberOfEventsWritten()
    if n and not a.particlize_only:
        with h5py.File(out_h5, "a") as f:
            f.attrs["prod_wall_s_total"] = time.time() - t_job
            if windows is not None:
                f.attrs.update(windows)
    if out_particlize and pwriter.GetNumberOfEventsWritten():
        if windows is not None:
            with h5py.File(out_particlize, "a") as f:
                f.attrs.update(windows)
        if a.particlize_only:
            with h5py.File(out_particlize, "a") as f:
                f.attrs["prod_wall_s_total"] = time.time() - t_job
    summary = {"out": None if a.particlize_only else out_h5, "seed": a.seed,
               "eos": provenance["eos"],
               "seed_source": a.seed_source, "campaign": a.campaign, "index": a.index,
               "grid": a.grid, "hard": hard_desc,
               "reuse": a.reuse, "deposition": not a.no_deposit, "surface": a.surface,
               "bulk_info": a.bulk_info,
               "events_requested": a.events, "events_written": n,
               "events_tau0_after_tau_min": late,
               "legs_cut_at_max_ntau": writer.n_clipped,
               "edge_e_threshold": a.edge_e_threshold,
               "events_hit_edge": writer.n_hit_edge,
               "droplets_total": int(droplets), "E_droplets_late_total": round(e_late, 3),
               "wall_s": round(time.time() - t_job, 1)}
    if out_particlize:
        summary.update(particlize=out_particlize, particlize_legs=list(particlize_legs(a)),
                       particlize_events_written=pwriter.GetNumberOfEventsWritten())
    if a.particlize_only and not a.hadrons_only:
        summary["particlize_only"] = True
    if hadr is not None:
        summary.update(hadrons_only=True, hadron_files=hadr.outs,
                       hadronize_s_total=round(hadr.seconds, 1),
                       hadronize_settings={"oversample": a.oversample,
                                           "oversample_bg": str(a.oversample_bg),
                                           "n_frag": a.n_frag,
                                           "keep_primaries": a.keep_primaries,
                                           "base_seed": a.hadronize_seed,
                                           "correlated": a.correlated},
                       backgrounds_capped=hadr.n_capped)
    if writer.bg_layout == "shared":
        summary["bg_layout"] = "shared"
    if a.trigger_kind != "none":
        summary.update(trigger=a.trigger_kind,
                       **{k: getattr(a, k) for k in TRIGGER_CUTS if getattr(a, k) is not None})
    if a.min_partons is not None:
        summary["min_partons"] = int(a.min_partons)
    if windows is not None:
        if a.parton_ymax is not None:
            summary.update(parton_ymax=a.parton_ymax, parton_y_mode=a.parton_y_mode)
        if "pthat_bin_acceptance" in windows:
            # of every cut (--parton-ymax, the trigger): kept / tried per window
            summary.update(pthat_bin_acceptance=[round(float(x), 4) for x in
                                                 windows["pthat_bin_acceptance"]],
                           pthat_bin_n_tried=[int(x) for x in windows["pthat_bin_n_tried"]])
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
