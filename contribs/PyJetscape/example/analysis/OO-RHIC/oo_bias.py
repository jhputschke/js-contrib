#!/usr/bin/env python3
"""
analysis/OO-RHIC/oo_bias.py -- step 4 of the plan forward (R4b, the full test): the centrality
bias of O+O jet events when the classes are built as the experiments build them, from the
charged-particle multiplicity of full events (hydro + particlization) in the estimators'
acceptance, with the jet in vacuum (no quenching): any deviation from N_coll scaling per class is
then selection bias.

    python oo_bias.py run mb -j 8 --mps           # stage 1: the minimum-bias reference
    python oo_bias.py run jets jets_noveto -j 8 --mps
    python oo_bias.py analyse                     # -> results/oo_bias.json
    jupyter nbconvert --to notebook --execute oo_centrality_bias.ipynb --output results/oo_centrality_bias_executed.ipynb

The bulk screening (plan A of the p+p / O+O bulk tune): minimum-bias stand-ins as `mb`, for p+p
and O+O, with the 3dMCGlauber sets of BULK_SETS (map, pp2407, m3; xsdef on request), 10k events
per sample in 250-event jobs, the samples interleaved so that an early stop leaves them even:

    python oo_bias.py run screen --list                       # the jobs and their commands
    python oo_bias.py run screen -j 8 --mps --omp 3 --limit 8 # a timing test (8 jobs)
    python oo_bias.py run screen -j 8 --mps --omp 3 --shard 0/2   # GPU 0; 1/2 on GPU 1
    python oo_bias.py run screen --events 5000 ...            # 5k per sample; 10k later adds
    python oo_bulk.py analyse --sample oo_pp2407              # O+O against STAR
On the 2 x RTX 3090 box: oo_bias_docker.sh (one container per GPU, the benchmarked settings).

The physics run (selection bias vs quenching; tuning page "Physics run plan", 2026-11-24), one
sub-sample per pTHat window (<sample>_w<lo>_<hi>, continuous 5-80 GeV), into a new output root:

    ./oo_bias_docker.sh --dev /data/OO-RHIC_phys phys1            # mb (~2.5 h); then, after
    ./oo_bias_docker.sh --dev /data/OO-RHIC_phys phys2            # docker rm: the MPI check (~3.5 h)
    ./oo_bias_docker.sh --dev /data/OO-RHIC_phys phys2_soft       # the soft-MPI check (~1.2 h)
    ./oo_bias_docker.sh --dev /data/OO-RHIC_phys phys3            # after gate G2 (~15 h)
    ./oo_bias_docker.sh --dev /data/OO-RHIC_phys phys3_icp        # afterwards: +15k per arm, 7-18 GeV (~6 h)
    ./oo_bias_docker.sh --dev /data/OO-RHIC_phys phys3_ppmed      # 3c: p+p with the jet in its medium (~1.5 h)
    ./oo_bias_docker.sh --dev /data/OO-RHIC_phys phys3_ppref      # 3d, optional: pp_ref +10k per window (~1.7 h)
    ./oo_bias_docker.sh --dev /data/OO-RHIC_phys phys3_icp phys3_hijet   # 3b with 3e: high-pT jets (~6 h together)
A group entry NAME:EVENTS runs the first jobs of NAME only.  Phases 1-2 ran with the old MPI;
phase 3 (oo_vac_*, oo_med_*, pp_ref_*) with soft MPI, under new names and seeds, the p+p
reference from config/pp200_isr_rhictune.xml.  The old-MPI samples that depend on the MPI are
frozen (FROZEN; job_list refuses them).

Samples (O+O 200 GeV, config/OO_MCGlauber_MUSIC_isr_rhictune.xml, b in [0, 10] fm as 3dMCGlauber
samples it, run_prod_oo_isr.py with MUSIC and iSS in the job: --bulk-hadrons 1 (one sample per
event: the physical fluctuations), --vacuum-jet, charged hadrons with |eta| < 5.2 stored):
  mb           the minimum-bias reference: --no-subtraction --hard-pair-string-veto 0, only the
               bulk hadrons counted.  With neither the subtraction nor the veto the ISR chain makes
               the same strings as minimum-bias events at the same N_coll and samples the same
               geometry (closure, centrality_proxy.py: N_coll, N_part, b distributions identical,
               string proxies within 1-2 %), so these are minimum-bias soft events with exactly the
               MUSIC and iSS settings of the jet events (the hard event's hadrons are ignored)
  jets         jet events with the production settings (subtraction, veto on): bulk + the hard
               event's hadrons counted
  jets_noveto  the same with the veto off (--hard-pair-string-veto 0)
  jets_quench  as jets, the same seeds, but without --vacuum-jet: the jet sees the medium (LBT,
               Matter in medium); R_CP(jets_quench) / R_CP(jets) per class is the quenching.
               Stores all hadrons (no --charged), for full jets as sPHENIX's calorimeter jets;
               the other samples store charged hadrons (estimator counts are charged in all)
Jet events in the pTHat windows 9-11, 18-23, 30-40 and 40-55 GeV, each weighted with its N_coll
(the generator puts exactly one hard scattering in every event).

Estimators (charged particles of the event): STAR's EPD 2.1 < |eta| < 5.1 and mid-rapidity |eta|
< 0.5 / < 1 (pT > 0.2 GeV), sPHENIX's MBD 3.51 < |eta| < 4.61 (both sides; >= 1 particle in each
arm is the minimum-bias coincidence).  Classes are percentiles of the minimum-bias multiplicity
(random tie-break); per class the experiment-style R_AA^vac = (f_jet,c / f_MB,c) <N_coll>_MB /
<N_coll>_MB,c, the bias factor <N_coll>_jet,c / <N_coll>_MB,c and R_CP.  Needs X-SCAPE
oo_rhic_run (the 3dMCGlauber switch) and the driver's --vacuum-jet.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C  # noqa: E402
import hstat as H  # noqa: E402

OUT = Path(os.environ.get("OO_RHIC_OO_PROD", Path.home() / "prod" / "OO-RHIC_OO")) / "bias"
OO_XML = C.EXAMPLE / "config" / "OO_MCGlauber_MUSIC_isr_rhictune.xml"
B_MAX = 10.0
JET_WINDOWS = [(9, 11), (18, 23), (30, 40), (40, 55)]
#: name: (events, events per job, windows, extra driver options, count the hard event's hadrons)
SAMPLES = {
    "mb": (50000, 500, [(9, 11)], ["--no-subtraction", "--hard-pair-string-veto", "0"], False),
    "jets": (50000, 500, JET_WINDOWS, [], True),
    "jets_noveto": (30000, 500, JET_WINDOWS, ["--hard-pair-string-veto", "0"], True),
    "jets_quench": (50000, 500, JET_WINDOWS, [], True),
    # validation of the hole/partner treatment (negative hadrons): the jet in the medium with
    # its final partons (holes) and iColored's Pythia input (partners) stored
    "jets_holes": (4500, 450, JET_WINDOWS[1:], ["--store-partons", "--store-hadronization-input"], True),
}
#: jets_quench reuses the seeds of jets (same geometry and hard scatterings, the jet now sees the medium)
SEED_BASE = {"mb": 700000, "jets": 710000, "jets_noveto": 720000, "jets_quench": 710000, "jets_holes": 730000}

# ── the bulk screening (plan A of the p+p / O+O bulk tune on the tuning page) ─────────────
#: 3dMCGlauber sets, as driver options on top of OO_XML.  The screening ran when OO_XML's own set
#: was the JETSCAPE MAP of 2601.17234 with shadowing 0; since 2026-11-24 OO_XML carries the
#: adopted set g_v045_n215, so the MAP is given explicitly (MAP below) and every set below names
#: all the parameters it differs from the XML in -- each sample runs as it did in the screening.
#: The input-file defaults of 3dMCGlauber are already the 2211.16376 method-3 values; they are
#: given explicitly so that the files record them.
MAP = [f"--xml-set=IS/MCGlauber/{k}={v}" for k, v in (
    ("ylossParam4At2", 1.61), ("ylossParam4At4", 1.685), ("ylossParam4At6", 1.685),
    ("ylossParam4At10", 2.95), ("ylossParam4var", 0.682), ("shadowing_factor", 0.0),
    ("BG", 3.96))]
BULK_SETS = {
    "map": MAP,
    # 2407.17443 p+p (Figs. 3-7): method 4, nodes 1.6 / 2.15 / 2.15 / 2.15, variance 0.5,
    # shadowing 0.45, BG 5 (its mcglauber.input); the rest as the MAP
    "pp2407": [f"--xml-set=IS/MCGlauber/{k}={v}" for k, v in (
        ("ylossParam4At2", 1.6), ("ylossParam4At4", 2.15), ("ylossParam4At6", 2.15),
        ("ylossParam4At10", 2.15), ("ylossParam4var", 0.5), ("shadowing_factor", 0.45),
        ("BG", 5.0))],
    # method 3 (2211.16376: A 1.32, alpha1 1.8, alpha2 0.34, variance 0.4), shadowing 0.45,
    # alpha_rem 0.5; the rest (BG, tau_form, hydro) as the MAP
    "m3": MAP + [f"--mcglauber-param={k}={v}" for k, v in (
        ("rapidity_loss_method", 3), ("yloss_param_slope", 1.32), ("yloss_param_alpha1", 1.8),
        ("yloss_param_alpha2", 0.34), ("yloss_param_fluct_var_RHIC", 0.4))]
        + [f"--xml-set=IS/MCGlauber/{k}={v}" for k, v in (
            ("shadowing_factor", 0.45), ("remnant_energy_loss_fraction", 0.5))],
    # X-SCAPE's default set (jetscape_main.xml): not in the default screening
    "xsdef": [f"--xml-set=IS/MCGlauber/{k}={v}" for k, v in (
        ("ylossParam4At2", 1.6), ("ylossParam4At4", 2.15), ("ylossParam4At6", 2.45),
        ("ylossParam4At10", 2.95), ("ylossParam4var", 0.5), ("shadowing_factor", 0.6),
        ("BG", 5.0))],
}
# follow-up of the screening (2026-11-23): pp2407 with a smaller yloss variance; the screening
# found its p+p P(N_ch) ~10 % too wide and its O+O p(N_ch) tail (N_ch > 120) too long
BULK_SETS["pp2407v03"] = [o for o in BULK_SETS["pp2407"] if "ylossParam4var" not in o] + [
    "--xml-set=IS/MCGlauber/ylossParam4var=0.3"]
# grid around it (2026-11-23): variance 0.4 / 0.45 x nodes At4 = At6 = At10 = 2.15 / 2.25; var 0.3
# fixed the p+p P(n) widths but lowered the yield 8-9 % and made the O+O tail too short
for _v in (0.4, 0.45):
    for _n in (2.15, 2.25):
        BULK_SETS[f"g_v{_v:g}_n{_n:g}".replace(".", "")] = [
            o for o in BULK_SETS["pp2407"]
            if not any(k in o for k in ("ylossParam4var", "At4=", "At6=", "At10="))] + [
            f"--xml-set=IS/MCGlauber/ylossParam4var={_v}"] + [
            f"--xml-set=IS/MCGlauber/ylossParam4At{y}={_n}" for y in (4, 6, 10)]
# remnant scan (2026-11-23) on the grid's best set g_v045_n215: the remnant string runs from the
# remnant's rapidity y_rem to y_rem - alpha_rem * y_loss (3dMCGlauber Glauber.cpp, string mode -4),
# so a larger alpha_rem moves remnant energy from |eta| > 3.5 toward mid-rapidity; all sets were
# above PHOBOS's p+p dN/deta beyond |eta| 3.5 and below it at 1.5-3 (XML value 0.536)
REM_BASE = "g_v045_n215"
for _a in (0.4, 0.65, 0.8, 1.0):
    BULK_SETS[f"rem{round(_a * 100):03d}"] = BULK_SETS[REM_BASE] + [
        f"--xml-set=IS/MCGlauber/remnant_energy_loss_fraction={_a}"]
REM_SETS = tuple(s for s in BULK_SETS if s.startswith("rem"))
# Au-going side of d+Au (2026-11-24): g_v045_n215 is 16-57 % high there, more toward peripheral,
# while mid-rapidity and the d side agree.  Shadowing and the y_init = 2 node act only on the
# strings of a nucleon's second and later collisions (which pile up on the Au side; p+p hardly
# changes): shadowing 0.6 (X-SCAPE's default) and 0.8, and At2 1.1 (2407.17443's p+Pb value)
for _k, _o in (("shad060", "shadowing_factor=0.6"), ("shad080", "shadowing_factor=0.8"),
               ("at2_110", "ylossParam4At2=1.1")):
    BULK_SETS[_k] = BULK_SETS[REM_BASE] + [f"--xml-set=IS/MCGlauber/{_o}"]
SHAD_SETS = ("shad060", "shad080", "at2_110")
# formation time (2026-11-24): the RHIC O+O XMLs (from X-SCAPE's BulkFastTest/OO_one_event.xml)
# had tau_form_mean 0.5 and the MAP's 0.424 only in MUSIC's Initial_time_tau_0, which MUSIC
# overwrites with the string source; the Au+Au / LHC XMLs have tau_form_mean 0.424.  All sets
# above ran with 0.5; since the second revision of 2026-11-24 OO_XML has 0.424, so they pin 0.5
# (first, so that a set's own options still win) and run as they did.  tau0424 is the adopted
# set with 0.424, i.e. OO_XML as it is now (dau_tau0424 the d+Au check of the final XML).
TAU_OLD = "--xml-set=IS/MCGlauber/tau_form_mean=0.5"
for _s in BULK_SETS:
    BULK_SETS[_s] = [TAU_OLD] + BULK_SETS[_s]
BULK_SETS["tau0424"] = BULK_SETS[REM_BASE] + ["--xml-set=IS/MCGlauber/tau_form_mean=0.424"]
GRID_SETS = tuple(s for s in BULK_SETS if s.startswith("g_"))
#: the sets run by default ("screen" on the command line): MAP, 2407.17443 p+p, method 3
SCREEN_SETS = ("map", "pp2407", "m3")
#: p+p with the O+O XML's chain (as make_design.py --hydro): one nucleon each, no O16
#: configurations, b <= 3 fm (as the p+p hard tune; the NN overlap ends well before)
PP = ["--xml-set=IS/MCGlauber/projectile=p", "--xml-set=IS/MCGlauber/target=p",
      "--xml-set=IS/MCGlauber/nucleon_configuration_from_file=0", "--b-max", "3"]
#: d+Au with the same chain (dau_bulk.py: the check against the MAP's own calibration data,
#: PHENIX/BRAHMS dN/deta): deuteron (Hulthen) on Au (Woods-Saxon, no configuration file), b < 12 fm
#: (the participants stay near the Au surface, inside MUSIC's +-10 fm box); target moves to -z
DAU = ["--xml-set=IS/MCGlauber/projectile=d", "--xml-set=IS/MCGlauber/target=Au",
       "--xml-set=IS/MCGlauber/nucleon_configuration_from_file=0", "--b-max", "12"]
#: minimum-bias stand-in as `mb` (no subtraction, veto off, bulk hadrons only)
MB_OPTS = ["--no-subtraction", "--hard-pair-string-veto", "0"]
SCREEN_EVENTS, SCREEN_PER_JOB = 10000, 250
for _s, _opts in BULK_SETS.items():
    SAMPLES[f"oo_{_s}"] = (SCREEN_EVENTS, SCREEN_PER_JOB, [(9, 11)], MB_OPTS + _opts, False)
    SAMPLES[f"pp_{_s}"] = (SCREEN_EVENTS, SCREEN_PER_JOB, [(9, 11)], MB_OPTS + PP + _opts, False)
    # every set uses the same seeds per system: the same geometries and hard scatterings as
    # far as the set does not change them, so set-to-set differences are less noisy
    # (jets_holes also starts at 730000: a separate validation sample with other options, run
    # elsewhere; kept as both have run)
    SEED_BASE[f"oo_{_s}"] = 730000
    SEED_BASE[f"pp_{_s}"] = 740000
    SAMPLES[f"dau_{_s}"] = (SCREEN_EVENTS, SCREEN_PER_JOB, [(9, 11)], MB_OPTS + DAU + _opts, False)
    SEED_BASE[f"dau_{_s}"] = 750000
#: name groups on the command line
GROUPS = {"screen": [f"{sys_}_{s}" for s in SCREEN_SETS for sys_ in ("pp", "oo")],
          "var": ["pp_pp2407v03", "oo_pp2407v03"],
          "grid": [f"{sys_}_{s}" for s in GRID_SETS for sys_ in ("pp", "oo")],
          "rem": [f"pp_{s}" for s in REM_SETS],          # step 1: p+p only (the forward shape)
          "shad": [f"{sys_}_{s}" for s in SHAD_SETS for sys_ in ("dau", "oo")],
          "tau": ["pp_tau0424", "oo_tau0424", "dau_tau0424"],
          "bias": ["mb", "jets", "jets_noveto", "jets_quench"],
          "holes": ["jets_holes"]}

#: samples whose jet sees the medium (LBT, Matter in medium); all others run with --vacuum-jet
QUENCHED = {"jets_quench", "jets_holes"}   # + every jets_med_* (below)
#: samples that store all hadrons (full jets as sPHENIX's calorimeter jets); all others --charged.
#: The estimator counts use charged hadrons in either case (_units selects them by species).
ALL_HADRONS = {"jets_quench", "jets_holes"}
EPD, MBD = (2.1, 5.1), (3.51, 4.61)

# ── the physics run (tuning page, "Physics run plan", 2026-11-24) ─────────────────────────
#: Selection bias vs quenching with the tune XMLs.  One sub-sample per pTHat window (the driver
#: puts the same number of events in every window of a job, the plan wants more at low pTHat):
#: <sample>_w<lo>_<hi>, its own directory and seeds, stitched by the windows' cross sections in
#: the analysis.  The windows are continuous, 5-80 GeV, so inclusive hadron, trigger and jet
#: spectra have no gaps.  Run them into a new output root (oo_bias_docker.sh /data/OO-RHIC_phys
#: ...), not over the R4b samples.
#:
#: Two generations of samples (tuning page, "G2, soft MPI"):
#:   phases 1-2 (phys_mb; pp_bulk_*, jets_vac_* ...): the tune XML of 788ca1e, MPI alpha_s 0.152,
#:     pTmin 5 (Pythia: ~1.7 GeV), the p+p ones as the O+O XML with the p+p options (PP).  The
#:     samples whose physics depends on that MPI are FROZEN (below): they are the paper-scheme
#:     (old-MPI) systematic and cannot be run again with today's XML under the same name.
#:   phase 3 (oo_vac_*, oo_med_*, pp_ref_*): the XMLs with soft MPI (pT0Ref 1.48, revision
#:     2026-11-24 (3)), new names and seeds; p+p from config/pp200_isr_rhictune.xml (made from
#:     the O+O XML by rhictune_xml.py; the same job XMLs as the O+O XML with PP, checked).
PHYS_WINDOWS = {(5, 7): 15000, (7, 9): 20000, (9, 11): 20000, (11, 14): 20000, (14, 18): 15000,
                (18, 23): 12500, (23, 30): 10000, (30, 40): 10000, (40, 55): 10000, (55, 80): 10000}
#: the central-enriched pair (STAR's I_CP): triggers come from pTHat 7-18; b restricted so that
#: every event of the 0-20 % class is inside.  B_MAX_CENTRAL is read off phys_mb's b distribution
#: in the EPD classes before the pair runs (environment OO_PHYS_CENTRAL_BMAX until then).
CENTRAL_WINDOWS = {(7, 9): 15000, (9, 11): 15000, (11, 14): 15000, (14, 18): 15000}
#: events added per window and arm by phys3_icp (the central pair's 15k, unrestricted in b)
ICP_EXTRA = 15000
#: events added per window to pp_ref by the optional 3d (phys3_ppref)
PP_EXTRA = 10000
#: 3e (2026-11-25): narrow pTHat windows at the upper ends of the wide high windows.  Inside a
#: window the events follow dsigma/dpTHat, so the 30-40 and 40-55 windows put 82 / 78 % of their
#: events in their lowest 5 GeV and only 558 / 431 (of 10k) at 37.5-40 / 50-55 GeV.  Those few
#: carry a large share of the sPHENIX bins across the window edges (51-60 GeV: 43 % of the yield
#: from 95 jets of the 40-55 window), and with it the error.  The narrow samples overlap the wide
#: windows; oo_analysis_rhic.py splits every window at all window edges and combines the events of
#: each piece (weight sigma_piece / all events in the piece).  Samples <family>_w35_40, _w47_55
#: (family oo_vac, oo_med, pp_ref), seeds 900000 + 1000 i (O+O, vacuum and medium alike) and
#: 910000 + 1000 i (p+p), i = 0, 1.
HIJET_WINDOWS = {(35, 40): 10000, (47, 55): 10000}
B_MAX_CENTRAL = float(os.environ.get("OO_PHYS_CENTRAL_BMAX", "nan"))
PHYS_PER_JOB = 500
STORE = ["--store-partons", "--store-hadronization-input"]      # the holes rule (a3bb4c2)
MPI_OFF = ["--pythia-line", "PartonLevel:MPI = off"]           # appended after the XML's lines
#: soft MPI (tuning page, "G2 input"): Pythia's soft MPI with the reference energy at 200 GeV,
#: pT0 = pT0Ref there; the check after G2 brackets pT0Ref (the response is steep).  Appended after
#: the XML's lines, so they override its MPI settings (since revision (3) the XML has 1.48 itself).
SOFT_PT0REF = (1.42, 1.45, 1.48)
#: the full-chain p+p XML (generated from OO_XML; rhictune_xml.py) and its options: b <= 3 fm
PP_XML = C.EXAMPLE / "config" / "pp200_isr_rhictune.xml"
PP_REF = ["--b-max", "3"]
#: sample -> user XML when not OO_XML
XML_OF = {}
#: samples that ran with the MPI of 788ca1e and depend on it: not to be run again (job_list)
FROZEN = set()


def soft_mpi(pt0ref):
    return [x for v in ("MultipartonInteractions:alphaSvalue = 0.130", "MultipartonInteractions:pTmin = 0.2",
                        "MultipartonInteractions:ecmRef = 200", f"MultipartonInteractions:pT0Ref = {pt0ref:.2f}")
            for x in ("--pythia-line", v)]


def _w(lo, hi):
    return f"w{lo:02d}_{hi:02d}"


_W = [_w(lo, hi) for lo, hi in PHYS_WINDOWS]
_W_UE = _W[:9]                                                  # 5-55 GeV: leading jets 6-40 GeV
_W_VAC2 = ("w18_23", "w30_40")                                  # the O+O MPI check of phase 2
#: phases 1-2 (old MPI; seeds 800000 + 1000 k for O+O, 840000 + 1000 k for p+p, k = window index).
#: Only what ran is defined.  MPI off and the explicit soft-MPI lines do not depend on the XML's
#: MPI, so those stay runnable; MPI on (pp_bulk, jets_vac) and the string switch are frozen.
for _k, ((_lo, _hi), _n) in enumerate(PHYS_WINDOWS.items()):
    _ww = _w(_lo, _hi)
    if _ww in _W_VAC2:
        for _name, _opts in ((f"jets_vac_{_ww}", STORE), (f"jets_vac_mpioff_{_ww}", STORE + MPI_OFF)):
            SAMPLES[_name] = (_n, PHYS_PER_JOB, [(_lo, _hi)], _opts, True)
            SEED_BASE[_name] = 800000 + 1000 * _k
        FROZEN.add(f"jets_vac_{_ww}")
    if _ww in _W_UE:
        for _name, _opts in ((f"pp_bulk_{_ww}", PP + STORE),
                             (f"pp_bulk_mpioff_{_ww}", PP + STORE + MPI_OFF),
                             (f"pp_bulk_switch_{_ww}", PP + STORE + ["--hard-pair-string-veto", "0"]),
                             *((f"pp_bulk_soft{round(100 * _p)}_{_ww}", PP + STORE + soft_mpi(_p))
                               for _p in SOFT_PT0REF)):
            SAMPLES[_name] = (5000, PHYS_PER_JOB, [(_lo, _hi)], _opts, True)
            SEED_BASE[_name] = 840000 + 1000 * _k
        FROZEN |= {f"pp_bulk_{_ww}", f"pp_bulk_switch_{_ww}"}
SAMPLES["phys_mb"] = (50000, PHYS_PER_JOB, [(9, 11)], MB_OPTS, False)    # bulk only: MPI-free
SEED_BASE["phys_mb"] = 860000
#: phase 3 (soft MPI): O+O vacuum and medium use the same seeds per window (870000 + 1000 k; not
#: event-paired: LBT in the task list shifts the random stream, so the events differ and the
#: med/vac errors are independent; GB10 pp_med test 2026-11-24), p+p
#: reference 880000 + 1000 k, the central pair 890000 + 1000 k
for _k, ((_lo, _hi), _n) in enumerate(PHYS_WINDOWS.items()):
    for _name in (f"oo_vac_{_w(_lo, _hi)}", f"oo_med_{_w(_lo, _hi)}"):
        SAMPLES[_name] = (_n, PHYS_PER_JOB, [(_lo, _hi)], STORE, True)
        SEED_BASE[_name] = 870000 + 1000 * _k
    # p+p reference with the jet in vacuum, and (3c) the same events with the jet in p+p's own
    # medium (Matter + LBT where the event has a string and MUSIC ran), the same seeds (not the
    # same events, as for O+O).  With in_vac 0 Matter showers partons outside a medium (T <= T0)
    # down to QS 0.9 GeV, with in_vac 1 (pp_ref, oo_vac, the hard tune) only to Q0 2 GeV: pp_med
    # shares that cutoff with oo_med, pp_med / pp_ref measures it together with p+p quenching
    for _name in (f"pp_ref_{_w(_lo, _hi)}", f"pp_med_{_w(_lo, _hi)}"):
        SAMPLES[_name] = (5000, PHYS_PER_JOB, [(_lo, _hi)], PP_REF + STORE, True)
        SEED_BASE[_name] = 880000 + 1000 * _k
        XML_OF[_name] = PP_XML
for _k, ((_lo, _hi), _n) in enumerate(CENTRAL_WINDOWS.items()):
    for _name in (f"oo_vac_central_{_w(_lo, _hi)}", f"oo_med_central_{_w(_lo, _hi)}"):
        SAMPLES[_name] = (_n, PHYS_PER_JOB, [(_lo, _hi)], STORE + ["--b-max", f"{B_MAX_CENTRAL}"], True)
        SEED_BASE[_name] = 890000 + 1000 * _k
for _i, ((_lo, _hi), _n) in enumerate(HIJET_WINDOWS.items()):
    for _name in (f"oo_vac_{_w(_lo, _hi)}", f"oo_med_{_w(_lo, _hi)}"):
        SAMPLES[_name] = (_n, PHYS_PER_JOB, [(_lo, _hi)], STORE, True)
        SEED_BASE[_name] = 900000 + 1000 * _i
    SAMPLES[f"pp_ref_{_w(_lo, _hi)}"] = (_n, PHYS_PER_JOB, [(_lo, _hi)], PP_REF + STORE, True)
    SEED_BASE[f"pp_ref_{_w(_lo, _hi)}"] = 910000 + 1000 * _i
    XML_OF[f"pp_ref_{_w(_lo, _hi)}"] = PP_XML
GROUPS.update({
    # phase 1: the minimum-bias reference (bulk hadrons only, so valid for either MPI)
    "phys1": ["phys_mb"],
    # phase 2, the MPI check with the old MPI (NAME:EVENTS takes the first jobs of NAME): p+p
    # MPI on, off, and with a string in every jet event; O+O vacuum jets on and off in two
    # windows.  Done; the "on" arms are frozen.
    "phys2": ([f"pp_bulk_{w}:3000" for w in _W_UE] + [f"pp_bulk_mpioff_{w}:3000" for w in _W_UE]
              + [f"pp_bulk_switch_{w}:3000" for w in _W_UE]
              + [f"{s}_{w}:5000" for w in _W_VAC2 for s in ("jets_vac", "jets_vac_mpioff")]),
    # the soft-MPI check after G2: p+p with hydro at three pT0Ref (1.45 added after the first two), 1.5k per window each
    "phys2_soft": [f"pp_bulk_soft{round(100 * p)}_{w}:1500" for w in _W_UE for p in SOFT_PT0REF],
    # phase 3 (soft MPI): vacuum and medium interleaved job by job, then the p+p reference
    "phys3": [f"{s}_{w}" for w in _W for s in ("oo_vac", "oo_med")] + [f"pp_ref_{w}" for w in _W],
    "phys3_central": [f"{s}_central_{_w(lo, hi)}" for lo, hi in CENTRAL_WINDOWS for s in ("oo_vac", "oo_med")],
    # the I_CP add-on after phase 3 (instead of the b-restricted central pair, whose enrichment
    # is only 1.28x): the same samples extended by ICP_EXTRA events in the trigger windows 7-18
    # GeV.  NAME:EVENTS takes the first jobs of NAME's seed list, so the jobs phase 3 ran are
    # skipped and only the new ones (the next seeds, vacuum and medium on the same seeds) run.
    "phys3_icp": [f"{s}_{_w(lo, hi)}:{PHYS_WINDOWS[(lo, hi)] + ICP_EXTRA}" for lo, hi in CENTRAL_WINDOWS
                  for s in ("oo_vac", "oo_med")],
    # 3c: the p+p reference with the jet in p+p's medium (pp_ref's seeds): how much the model
    # quenches in p+p, for a consistent reference (expected small: the medium exists in ~35 % of
    # p+p jet events and is much smaller than O+O's)
    "phys3_ppmed": [f"pp_med_{w}" for w in _W],
    # 3d, OPTIONAL (2026-11-25): pp_ref extended by PP_EXTRA events per window, 5k -> 15k, about
    # O+O's statistics after 3b (12.5-13.5k per window, 28k in 7-18 GeV).  For the precision of what
    # is divided by p+p: R_AA (with 5k p+p events per window against O+O's 12.5k the p+p reference is
    # probably the larger part of the minimum-bias R_AA error), I_AA, and pp_med / pp_ref (3c).  Not
    # R_CP or I_CP (p+p cancels), and not the stitching: the windows the pooled rule leaves out
    # (oo_analysis_rhic.py) are thin in the O+O arms as well, so 3x pp_ref alone recovers only
    # hadrons 15-20 GeV and charged R = 0.2 jets 8-12 GeV.  NAME:EVENTS, as phys3_icp: the 10 jobs
    # phase 3 ran are skipped, the next 20 seeds per window run.  pp_med keeps 5k per window
    # unless extended the same way.
    "phys3_ppref": [f"pp_ref_{w}:{5000 + PP_EXTRA}" for w in _W],
    # 3e (2026-11-25): the narrow high windows (HIJET_WINDOWS), O+O vacuum and medium and p+p; run
    # with 3b (./oo_bias_docker.sh ... phys3_icp phys3_hijet)
    "phys3_hijet": [f"{s}_{_w(lo, hi)}" for lo, hi in HIJET_WINDOWS for s in ("oo_vac", "oo_med", "pp_ref")],
})
#: phase 4 (tuning page "Phase 4", 2026-11-25; OPTIONAL, defined, not run): the vacuum arms in the
#: A+A scheme.  Matter with in_vac 0 showers partons outside a medium down to QS 0.9 GeV (the medium
#: arm's vacuum shower), with in_vac 1 (--vacuum-jet, the hard tune) only to Q0 2 GeV.  The final
#: A+A-scheme recheck (scheme_recheck.ipynb, 294k events per arm in the vacuum chain): jets unchanged
#: (1.00 within 1-2 %), pi0 0.97-0.98 at 2-4 GeV, 0.87-0.95 +- 0.05 at 4-6 GeV, consistent with 1 above;
#: equal chi2 in both schemes.  So the scheme mismatch of oo_med against oo_vac / pp_ref is a
#: systematic of a few percent at most, measured by 3c (pp_med against pp_ref); these arms would remove
#: it for the hadron R_AA and h-h I_CP.  AA_VAC is the full-chain arm of the recheck (aa_vac_h): Matter
#: alone (no LBT, no recoils) and a medium Matter never sees (T0 = hydro_Tc = 100 GeV: no qhat), the
#: medium itself still made (MUSIC, iSS: the classes need the bulk), not the driver's --aa-vacuum (that
#: goes with --vacuum, no medium); the same seeds as oo_vac / pp_ref; 4b extends oo_vac_aa in 7-18 GeV
#: with 3b's seeds.
AA_VAC = ["--matter-only", "--xml-set=Eloss/Matter/T0=100", "--xml-set=Eloss/Matter/hydro_Tc=100"]
AA_SCHEME = set()
for _k, ((_lo, _hi), _n) in enumerate(PHYS_WINDOWS.items()):
    _name = f"oo_vac_aa_{_w(_lo, _hi)}"
    SAMPLES[_name] = (_n, PHYS_PER_JOB, [(_lo, _hi)], STORE + AA_VAC, True)
    SEED_BASE[_name] = 870000 + 1000 * _k
    _name = f"pp_ref_aa_{_w(_lo, _hi)}"
    SAMPLES[_name] = (5000, PHYS_PER_JOB, [(_lo, _hi)], PP_REF + STORE + AA_VAC, True)
    SEED_BASE[_name] = 880000 + 1000 * _k
    XML_OF[_name] = PP_XML
AA_SCHEME |= {n for n in SAMPLES if n.startswith(("oo_vac_aa_", "pp_ref_aa_"))}
GROUPS.update({
    # 4a: the A+A-scheme vacuum arms, O+O and p+p
    "phys4": [f"oo_vac_aa_{w}" for w in _W] + [f"pp_ref_aa_{w}" for w in _W],
    # 4b: oo_vac_aa extended in the trigger windows 7-18 GeV, as phys3_icp
    "phys4_icp": [f"oo_vac_aa_{_w(lo, hi)}:{PHYS_WINDOWS[(lo, hi)] + ICP_EXTRA}" for lo, hi in CENTRAL_WINDOWS],
})
QUENCHED |= {n for n in SAMPLES if n.startswith(("oo_med_", "pp_med_"))}
#: the physics run stores all hadrons (full jets, sPHENIX), the minimum-bias reference too
ALL_HADRONS |= {n for n in SAMPLES if n.startswith(("jets_vac_", "oo_vac_", "oo_med_", "pp_bulk_", "pp_ref_", "pp_med_", "phys_"))}   # incl. *_aa_
#: Au+Au energy-loss calibration (2026-11-25; tuning page "A3 and the proposed runs",
#: auau_bulk_phobos.ipynb): the O+O chain (OO_XML: the RHIC bulk tune, ISR, strings, MUSIC, iSS,
#: Matter + LBT, iColored) with Au + Au, generated nucleon positions, MUSIC grid +-13 fm (no
#: boundary hit in the tests), 0-10 % by b (4.7 fm, geometric), hard partons |y| < 1.0 (either
#: one: no mid-rapidity jet is lost; the driver records the acceptance per window).  Jet in the
#: medium.  B1 auau_med: the current energy-loss setting (Matter QhatParametrizationType 1 = the
#: Type 2 of 2204.01163, alpha_s 0.25, Lbt 0.25).  B2 auau_t3: 2204.01163's default Type 3
#: (type 5, alpha_s 0.3, qhatA 10, qhatB 100, Lbt alpha_s 0.3).  The same seeds (920000 + 1000 k).
#: 2k events per window in jobs of 50 (~15 s per event with 10 jobs per GPU).  Groups auau_b1 /
#: auau_b2 list the windows in the order to run them with --sequential: 23-80 GeV first (the
#: per-parton jet loss), then 9-23 (PHENIX pi0, STAR charged jets), then 5-9 (the low-pT bins).
AUAU = ["--xml-set=IS/MCGlauber/projectile=Au", "--xml-set=IS/MCGlauber/target=Au",
        "--xml-set=IS/MCGlauber/nucleon_configuration_from_file=0",
        "--xml-set=IS/grid_max_x=13", "--xml-set=IS/grid_max_y=13",
        "--b-max", "4.7", "--parton-ymax", "1.0", "--parton-y-mode", "any"]
TYPE3 = ["--xml-set=Eloss/Matter/QhatParametrizationType=5", "--xml-set=Eloss/Matter/alphas=0.3",
         "--xml-set=Eloss/Matter/qhatA=10", "--xml-set=Eloss/Matter/qhatB=100",
         "--xml-set=Eloss/Lbt/alphas=0.3"]
AUAU_EVENTS, AUAU_PER_JOB = 2000, 50
for _k, ((_lo, _hi), _n) in enumerate(PHYS_WINDOWS.items()):
    for _fam, _opts in (("auau_med", []), ("auau_t3", TYPE3)):
        _name = f"{_fam}_{_w(_lo, _hi)}"
        SAMPLES[_name] = (AUAU_EVENTS, AUAU_PER_JOB, [(_lo, _hi)], STORE + AUAU + _opts, True)
        SEED_BASE[_name] = 920000 + 1000 * _k
_AUAU_ORDER = ["w23_30", "w30_40", "w40_55", "w55_80", "w09_11", "w11_14", "w14_18", "w18_23", "w05_07", "w07_09"]
GROUPS.update({"auau_b1": [f"auau_med_{w}" for w in _AUAU_ORDER],
               "auau_b2": [f"auau_t3_{w}" for w in _AUAU_ORDER]})
QUENCHED |= {n for n in SAMPLES if n.startswith(("auau_med_", "auau_t3_"))}
ALL_HADRONS |= {n for n in SAMPLES if n.startswith(("auau_med_", "auau_t3_"))}
#: hadronization tests (2026-11-26, auau_jet_raa.ipynb "Where B1's under-quenching comes from"): B1's job with one
#: change, B1's seeds, windows 18-23 / 23-30 / 30-40 GeV, 500 events each.  T1 (--matter-only) and T2
#: (--colour-closure thermal) ran on the GB10 (~/prod/AuAu-RHIC_hadtests); T3 (--colour-closure fake) here.
HADTEST_WINDOWS = [(18, 23), (23, 30), (30, 40)]
for _k, ((_lo, _hi), _n) in enumerate(PHYS_WINDOWS.items()):
    if (_lo, _hi) not in HADTEST_WINDOWS:
        continue
    _name = f"auau_fake_{_w(_lo, _hi)}"
    SAMPLES[_name] = (500, AUAU_PER_JOB, [(_lo, _hi)], STORE + AUAU + ["--colour-closure", "fake"], True)
    SEED_BASE[_name] = 920000 + 1000 * _k
GROUPS["hadtest_t3"] = [f"auau_fake_{_w(lo, hi)}" for lo, hi in HADTEST_WINDOWS]
QUENCHED |= {n for n in SAMPLES if n.startswith("auau_fake_")}
ALL_HADRONS |= {n for n in SAMPLES if n.startswith("auau_fake_")}
STAR_CLASSES = [(0, 10), (10, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
SPH_CLASSES = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]


def jobs(name, events=None):
    """The jobs of a sample: (name, seed, events, command).  The sample's extra options come
    last, so they win over the defaults here (--b-max for p+p).  events: another total for
    the sample (the first jobs of the same seed list, so a larger total extends a smaller)."""
    total, per_job, windows, extra, _ = SAMPLES[name]
    if "_central_" in name and not np.isfinite(B_MAX_CENTRAL):
        sys.exit(f"oo_bias.py: {name} needs OO_PHYS_CENTRAL_BMAX (b_max of the 0-20 % class, from phys_mb)")
    total = events or total
    per_job -= per_job % len(windows)
    out = []
    for j in range(total // per_job):
        seed = SEED_BASE[name] + j
        cmd = [sys.executable, str(C.DRIVER), "--user-xml", str(XML_OF.get(name, OO_XML)), "--events", str(per_job),
               "--seed", str(seed), "--pthat-bins", ",".join(f"{lo}-{hi}" for lo, hi in windows),
               "--b-min", "0", "--b-max", f"{B_MAX}", *([] if name in QUENCHED or name in AA_SCHEME else ["--vacuum-jet"]), "--bulk-hadrons", "1",
               "--no-particlize", "--no-hydro-file", "--eta-max", "5.2",
               *([] if name in ALL_HADRONS else ["--charged"]),
               "--outdir", str(OUT / name), "--max-skipped", "30", *extra]
        out.append((name, seed, per_job, cmd))
    return out


def done(name, seed, events):
    for js in (OUT / name).glob(f"*_seed{seed:04d}.json"):
        try:
            d = json.loads(js.read_text())
            return d.get("events_written", 0) + d.get("events_skipped", 0) == events
        except (OSError, ValueError):
            return False
    return False


def _split(n, events=None):
    """NAME or NAME:EVENTS (the first jobs of NAME's seed list); --events applies otherwise."""
    if ":" in n:
        name, ev = n.split(":")
        return name, int(ev)
    return n, events


def job_list(names, events=None, shard=(0, 1), limit=None, sequential=False):
    """The pending jobs, the samples interleaved (job 0 of every sample, then job 1, ...), so
    that a run stopped early has about the same number of events in each sample; with
    sequential, sample after sample in the given order (the first samples finish first).  Shard (i, n)
    takes job k of every sample with k % n == i: disjoint shards (one container per GPU, or
    per machine) that never run the same job twice, each with every sample."""
    frozen = sorted({_split(n, events)[0] for n in names} & FROZEN)
    if frozen:
        sys.exit("oo_bias.py: these samples ran with the MPI of 788ca1e and are frozen (the "
                 "tune XML has soft MPI since revision (3); phase 3 runs oo_vac_*, oo_med_*, "
                 "pp_ref_*): " + ", ".join(frozen))
    if any(XML_OF.get(_split(n, events)[0]) == PP_XML for n in names):
        import rhictune_xml
        if rhictune_xml.check(quiet=True):
            sys.exit("oo_bias.py: config/pp200_isr_rhictune.xml is not the O+O XML with p+p nuclei")
    per = [jobs(*_split(n, events)) for n in names]
    i, n = shard
    if sequential:
        mine = [p[k] for p in per for k in range(i, len(p), n)]
    else:
        mine = [p[k] for k in range(i, max(map(len, per), default=0), n) for p in per if k < len(p)]
    todo = [t for t in mine if not done(*t[:3])]
    return todo[:limit] if limit else todo


def run(names, nproc, mps, omp, events=None, shard=(0, 1), limit=None, sequential=False, stagger=0):
    todo = job_list(names, events, shard, limit, sequential)
    print(f"oo_bias.py: {len(todo)} jobs" + (f" (shard {shard[0]}/{shard[1]})" if shard[1] > 1 else "")
          + f", {nproc} at a time, OMP_NUM_THREADS {omp}"
          + f", MUSIC_CUDA_SYNC {os.environ.get('MUSIC_CUDA_SYNC', 'unset')} -> {OUT}", flush=True)
    env = dict(os.environ, OMP_NUM_THREADS=str(omp))
    mps_dir = None
    if mps:
        mps_dir = os.path.join(os.environ.get("XDG_RUNTIME_DIR", "/tmp"), f"xscape-mps.{os.getpid()}")
        os.makedirs(os.path.join(mps_dir, "pipe"), exist_ok=True)
        os.makedirs(os.path.join(mps_dir, "log"), exist_ok=True)
        env.update(CUDA_MPS_PIPE_DIRECTORY=os.path.join(mps_dir, "pipe"),
                   CUDA_MPS_LOG_DIRECTORY=os.path.join(mps_dir, "log"))
        if subprocess.run(["nvidia-cuda-mps-control", "-d"], env=env).returncode != 0:
            sys.exit("oo_bias.py: could not start the CUDA MPS daemon")
    t0 = time.time()

    def one(it):
        idx, (name, seed, events, cmd) = it
        if idx < nproc and stagger:
            time.sleep(idx * stagger)           # the first round starts staggered
        (OUT / name).mkdir(parents=True, exist_ok=True)
        with open(OUT / name / f"run_seed{seed}.log", "w") as log:
            r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                               cwd=str(C.DRIVER.parent))
        return name, seed, r.returncode

    nfail = 0
    try:
        with ThreadPoolExecutor(max_workers=nproc) as ex:
            for k, (name, seed, rc) in enumerate(ex.map(one, enumerate(todo)), 1):
                nfail += rc != 0
                print(f"[{k}/{len(todo)}] {name} seed {seed}: {'ok' if rc == 0 else f'FAILED ({rc})'}"
                      f", {time.time() - t0:.0f} s", flush=True)
    finally:
        if mps_dir:
            subprocess.run(["nvidia-cuda-mps-control"], input=b"quit\n", env=env)
    return 1 if nfail else 0


# ── analysis ────────────────────────────────────────────────────────────────────────────────
_CHARGED = None


def charged_ids():
    """|PDG id| of the charged final-state particles (jetscape.hadrons_h5.CHARGED)."""
    global _CHARGED
    if _CHARGED is None:
        sys.path.insert(0, str(C.EXAMPLE.parent / "python"))
        from jetscape.hadrons_h5 import CHARGED
        _CHARGED = np.asarray(sorted({abs(int(x)) for x in CHARGED}))
    return _CHARGED


def _units(f, n, charged=True, with_pid=False):
    """Momenta (E, px, py, pz) and particles per unit of the first n units.  charged=True keeps the
    charged particles only (files written with --charged hold nothing else; files with all hadrons
    are filtered by species), so the estimator counts mean the same for both kinds of file."""
    so = f["hadrons/sample_offsets"][:]
    uo = f["hadrons/unit_offsets"][: n + 1]
    stop = int(so[uo[-1]])
    p = f["hadrons/p"][:stop].astype(np.float64)
    counts = np.diff(so[uo])
    pid = f["hadrons/pid"][:stop] if (with_pid or (charged and not bool(f.attrs.get("charged_only", False)))) else None
    if charged and not bool(f.attrs.get("charged_only", False)):
        keep = np.isin(np.abs(pid), charged_ids())
        ev = np.repeat(np.arange(n), counts)
        p, pid, counts = p[keep], pid[keep], np.bincount(ev[keep], minlength=n)
    return (p, counts, pid) if with_pid else (p, counts)


def counts_of(p, counts):
    """Per event: charged particles in the estimator windows (the files hold charged hadrons)."""
    E, px, py, pz = p.T
    pt = np.hypot(px, py)
    pabs = np.sqrt(px * px + py * py + pz * pz)
    eta = 0.5 * np.log(np.clip(pabs + pz, 1e-12, None) / np.clip(pabs - pz, 1e-12, None))
    ev = np.repeat(np.arange(len(counts)), counts)
    n = len(counts)
    sel = {"epd": (np.abs(eta) > EPD[0]) & (np.abs(eta) < EPD[1]),
           "mbd_neg": (-eta > MBD[0]) & (-eta < MBD[1]),
           "mbd_pos": (eta > MBD[0]) & (eta < MBD[1]),
           "mid05": (np.abs(eta) < 0.5) & (pt > 0.2),
           "mid10": (np.abs(eta) < 1.0) & (pt > 0.2)}
    return {k: np.bincount(ev[m], minlength=n) for k, m in sel.items()}


def load(name):
    """Per event: N_coll, N_part, b, the pTHat window, has_medium, and the estimator counts
    (bulk, plus the hard event's hadrons where the sample counts them)."""
    import hdf5plugin  # noqa: F401
    import h5py
    with_jet = SAMPLES[name][4]
    rec = {}
    for jf in sorted(glob.glob(str(OUT / name / "*_hadrons_jet_frag.h5"))):
        bf = jf.replace("_hadrons_jet_frag.h5", "_hadrons_bulk_bg.h5")
        if not os.path.exists(jf.replace("_hadrons_jet_frag.h5", ".json")):
            continue  # job still running (the summary is written at the end)
        with h5py.File(jf) as h:
            n = int(h.attrs["nunits_written"])
            u = {k: h[f"units/{k}"][:n] for k in ("glauber_ncoll", "glauber_npart", "glauber_b",
                                                   "pthat_bin", "has_medium", "glauber_nstrings")}
            pj, cj = _units(h, n)
        tot = {k: np.zeros(n, dtype=np.int64) for k in ("epd", "mbd_neg", "mbd_pos", "mid05", "mid10")}
        if os.path.exists(bf):
            with h5py.File(bf) as h:
                if int(h.attrs["nunits_written"]) < n:
                    raise ValueError(f"{bf}: fewer units than {jf}")
                pb, cb = _units(h, n)
            for k, v in counts_of(pb, cb).items():
                tot[k] += v
        if with_jet:
            for k, v in counts_of(pj, cj).items():
                tot[k] += v
        for k, v in {**u, **{f"n_{k}": v for k, v in tot.items()}}.items():
            rec.setdefault(k, []).append(v)
    rec = {k: np.concatenate(v) for k, v in rec.items()}
    if rec:
        rec["n_mbd"] = rec["n_mbd_neg"] + rec["n_mbd_pos"]
    return rec


def class_analysis(mb, jet, est, classes, rng, mask=None):
    """Percentile classes of the minimum-bias multiplicity (all events, random tie-break) applied
    to the N_coll-weighted jet events."""
    nc = len(classes)
    xm = mb[f"n_{est}"] + rng.random(len(mb["glauber_ncoll"]))
    cuts = np.percentile(xm, [100 - hi for _, hi in classes])

    def cls(v):
        c = np.full(len(v), nc - 1)
        for k in range(nc - 1, -1, -1):
            c[v >= cuts[k]] = k
        return c
    cm = cls(xm)
    m = np.ones(len(jet["glauber_ncoll"]), bool) if mask is None else mask
    cj = cls(jet[f"n_{est}"][m] + rng.random(int(m.sum())))
    w = jet["glauber_ncoll"][m].astype(float)
    fmb = np.array([np.mean(cm == k) for k in range(nc)])
    nmb, nmb_e = H.class_mean(cm, mb["glauber_ncoll"], nc)
    fj = np.array([w[cj == k].sum() for k in range(nc)]) / w.sum()
    njet, njet_e = H.class_mean(cj, jet["glauber_ncoll"][m], nc, w)
    # R_AA^vac = (f_jet,c / f_MB,c) <N_coll>_MB / <N_coll>_MB,c with the statistical errors of both
    # samples (hstat.raa_vac); R_CP from the same class sums
    rv = H.raa_vac(cm, mb["glauber_ncoll"], cj, w, nc)
    bias, bias_e = H.ratio(njet, njet_e, nmb, nmb_e)
    out = {"cuts": cuts.tolist(), "frac_mb": fmb.tolist(), "ncoll_mb": nmb.tolist(), "ncoll_mb_err": nmb_e.tolist(),
           "frac_jet": fj.tolist(), "ncoll_jet": njet.tolist(), "ncoll_jet_err": njet_e.tolist(),
           "bias": bias.tolist(), "bias_err": bias_e.tolist(),
           "raa_vac": rv["raa"].tolist(), "raa_err": rv["err"].tolist(), "n_jet": int(m.sum()), "rcp": {}}
    for a, b in ((0, nc - 2), (0, nc - 3), (1, nc - 2)):
        out["rcp"][f"{classes[a][0]}-{classes[a][1]}/{classes[b][0]}-{classes[b][1]}"] = H.rcp(rv, a, b)
    return out


def analyse():
    rng = np.random.default_rng(11)
    mb = load("mb")
    if not mb:
        sys.exit("oo_bias.py: no minimum-bias events yet")
    res = {"windows": JET_WINDOWS, "star_classes": STAR_CLASSES, "sph_classes": SPH_CLASSES,
           "n_mb": int(len(mb["glauber_ncoll"])), "mb": {}, "jets": {}}
    coin_mb = (mb["n_mbd_neg"] > 0) & (mb["n_mbd_pos"] > 0)
    res["mb"] = {"ncoll_mean": float(mb["glauber_ncoll"].mean()),
                 "eps_mbd_coincidence": float(coin_mb.mean()), "eps_mbd_coincidence_err": H.fraction(coin_mb)[1],
                 "has_medium": float(mb["has_medium"].mean()),
                 "mean_counts": {k: float(mb[f"n_{k}"].mean()) for k in ("epd", "mbd", "mid05", "mid10")}}
    print(f"minimum bias: {res['n_mb']} events, <N_coll> {res['mb']['ncoll_mean']:.2f}, MBD coincidence "
          f"{res['mb']['eps_mbd_coincidence']:.3f} (sPHENIX 0.914), with medium {res['mb']['has_medium']:.3f}")
    for name in ("jets", "jets_noveto", "jets_quench"):
        jet = load(name)
        if not jet:
            continue
        w = jet["glauber_ncoll"].astype(float)
        coin = (jet["n_mbd_neg"] > 0) & (jet["n_mbd_pos"] > 0)
        r = {"n": int(len(w)), "eps_mbd_coincidence_weighted": float(np.average(coin, weights=w)),
             "eps_mbd_coincidence_weighted_err": H.fraction(coin, w)[1],
             "has_medium_weighted": float(np.average(jet["has_medium"], weights=w)), "all": {}, "per_window": {}}
        for est, classes in (("epd", STAR_CLASSES), ("mid10", STAR_CLASSES), ("mid05", STAR_CLASSES),
                             ("mbd", SPH_CLASSES), ("epd_sph", SPH_CLASSES)):
            e = "epd" if est == "epd_sph" else est
            r["all"][est] = class_analysis(mb, jet, e, classes, rng)
            for k, wdw in enumerate(JET_WINDOWS):
                r["per_window"].setdefault(f"{wdw[0]}-{wdw[1]}", {})[est] = class_analysis(
                    mb, jet, e, classes, rng, mask=jet["pthat_bin"] == k)
        res["jets"][name] = r
        print(f"{name}: {r['n']} events, MBD coincidence (N_coll-weighted) {r['eps_mbd_coincidence_weighted']:.3f}")
        for est in ("epd", "mid10", "mbd"):
            a = r["all"][est]
            labs = STAR_CLASSES if est != "mbd" else SPH_CLASSES
            print(f"   {est:5s} R_AA^vac " + " ".join(f"{lo}-{hi}% {v:.2f}" for (lo, hi), v in zip(labs, a["raa_vac"])))
        m = r["all"]["mbd"]["raa_vac"]
        rcp = lambda a_, b_: a_ / b_ if b_ > 0 else float("nan")
        print(f"   sPHENIX classes (MBD): R_CP 0-20/60-80 {rcp(m[0], m[3]):.2f}, 0-20/40-60 {rcp(m[0], m[2]):.2f}")
    path = Path(os.environ.get("OO_BIAS_JSON", HERE / "results" / "oo_bias.json"))
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(res, indent=1))
    print("wrote", path)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=("run", "analyse"))
    p.add_argument("names", nargs="*", default=list(SAMPLES))
    p.add_argument("-j", type=int, default=8)
    p.add_argument("--mps", action="store_true", help="share the GPU through a CUDA MPS daemon")
    p.add_argument("--omp", type=int, default=2, help="OMP_NUM_THREADS per job (default 2)")
    p.add_argument("--out", default=None, help="output root (default ~/prod/OO-RHIC_OO/bias)")
    p.add_argument("--shard", default="0/1", metavar="I/N",
                   help="run every N-th job of the list, starting at I (0-based): one container "
                        "per GPU, e.g. --shard 0/2 on GPU 0 and --shard 1/2 on GPU 1")
    p.add_argument("--limit", type=int, default=None, metavar="K",
                   help="run at most K of the pending jobs (a timing test)")
    p.add_argument("--events", type=int, default=None, metavar="N",
                   help="events per sample instead of the sample's total (the first jobs of its "
                        "seed list; a larger N later adds the rest)")
    p.add_argument("--list", action="store_true", help="print the pending jobs and exit")
    p.add_argument("--sequential", action="store_true",
                   help="run the samples one after the other in the given order (default: interleaved)")
    p.add_argument("--stagger", type=float, default=0, metavar="S",
                   help="start the first round of jobs S seconds apart")
    a = p.parse_args(argv)
    global OUT
    if a.out:
        OUT = Path(a.out)
    names = [x for n in a.names for x in GROUPS.get(n, [n])] if a.names else GROUPS["bias"]
    bad = [n for n in names if _split(n)[0] not in SAMPLES]
    if bad:
        sys.exit(f"oo_bias.py: unknown samples {bad}; known {list(SAMPLES)} and the groups "
                 f"{list(GROUPS)}")
    try:
        i, n = (int(x) for x in a.shard.split("/"))
        assert n >= 1 and 0 <= i < n
    except (ValueError, AssertionError):
        sys.exit(f"oo_bias.py: --shard wants I/N with 0 <= I < N, got {a.shard}")
    if a.list:
        for name, seed, ev, cmd in job_list(names, a.events, (i, n), a.limit, a.sequential):
            print(name, seed, ev, " ".join(cmd))
        return 0
    if a.step == "run":
        return run(names, a.j, a.mps, a.omp, a.events, (i, n), a.limit, a.sequential, a.stagger)
    return analyse()


if __name__ == "__main__":
    sys.exit(main())
