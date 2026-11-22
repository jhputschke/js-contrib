"""
analysis/OO-RHIC/config.py -- the settings of the RHIC p+p 200 GeV hard tune, in one place.

The tune follows the plan on the "RHIC p+p and O+O Tuning" page (step 1 and 2): the X-SCAPE
ISR chain in vacuum (3dMCGlauber + PythiaIsrGun + iMATTER + Matter + iColored), run through
prod_OO_isr/run_prod_oo_isr.py --vacuum, pTHat windows from 5 GeV weighted by Pythia's
cross sections, MultipartonInteractions:pTmin = 5 from the XML (Pythia lowers it to 1.7 GeV at
200 GeV: its MPI needs sigma_int > 1.1 sigma_ND, so pT0 and pTmin are reduced at init).  Varied: alpha_s(MZ) of the hard process and the
MPI, the virtuality factor Cv (vir_factor) and Q0, both read by iMATTER and Matter.  Fixed:
Pythia's PDF set = iMATTER's (PDF:pSet 20, NNPDF3.1 NNLO), Lambda_QCD = 0.2 (the XML's),
probStoUD at Pythia's 0.217.  Data in the fit: PHENIX pi0 and STAR inclusive jets R = 0.5;
STAR pi0 and eta/pi0 for validation only.

Everything bulky (events, observables) goes under PROD_DIR; the code, the design, the
HEPData tables and the results of the fit stay in this folder.
"""

from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXAMPLE = HERE.parents[1]                      # contribs/PyJetscape/example
DRIVER = EXAMPLE / "prod_OO_isr" / "run_prod_oo_isr.py"
TEMPLATE_XML = EXAMPLE / "config" / "pp200_isr_vacuum.xml"

#: where the events and the per-point observables go (override with $OO_RHIC_PROD)
PROD_DIR = Path(os.environ.get("OO_RHIC_PROD", Path.home() / "prod" / "OO-RHIC_pp_hard"))

# ── the model ────────────────────────────────────────────────────────────────────────────
#: pTHat windows [GeV].  From 5 GeV: above iMATTER's floor Q0/sqrt(Cv) for every design
#: point (Q0 <= 2, Cv >= 0.16 -> <= 5 GeV).  Up to 80 GeV for STAR's jets to 52 GeV.
PTHAT_WINDOWS = [(5, 7), (7, 9), (9, 11), (11, 14), (14, 18), (18, 23), (23, 30),
                 (30, 40), (40, 55), (55, 80)]
#: fixed Pythia settings of the hard process (alpha_s is added per design point)
FIXED_PYTHIA = ["PDF:pSet = 20"]
#: iColored: pi0 and eta stay undecayed (the data count them); the jets' energy is the same
FIXED_HADRONIZATION = ["111:mayDecay = off", "221:mayDecay = off"]
#: stored hadrons: |eta| < 1.5 covers the jets (|eta| < 0.9, R = 0.5) and their off-axis cones
ETA_MAX = 1.5

# ── the design ───────────────────────────────────────────────────────────────────────────
#: name, range; alpha_s(MZ) goes to SigmaProcess and MultipartonInteractions
PARAMETERS = {"alpha_s": (0.11, 0.20), "Cv": (0.16, 0.50), "Q0": (1.0, 2.0)}
PARAM_LABELS = {"alpha_s": r"$\alpha_s(M_Z)$", "Cv": r"$C_v$", "Q0": r"$Q_0$ [GeV]"}
N_DESIGN = 32
DESIGN_SEED = 20261122
#: events per design point = JOBS_PER_POINT x EVENTS_PER_JOB, spread evenly over the windows
JOBS_PER_POINT = 10
EVENTS_PER_JOB = 30000                         # 3000 per window per job: 300k per point
#: extra events for the pi0 (6-20 GeV comes from these windows): per design point
#: LOW_JOBS_PER_POINT jobs of EVENTS_PER_JOB events over LOW_WINDOWS, in the point's directory
#: (own seeds); observables.py combines the windows of all files of a run by their edges
LOW_WINDOWS = [(5, 7), (7, 9), (9, 11), (11, 14), (14, 18), (18, 23)]
LOW_JOBS_PER_POINT = 6                         # 6 x 30k = 30k more events per low window
#: design jobs that crashed, and the seed that replaced them (the same seed crashes again at
#: the same event).  p030 seed 130001: segfault in X-SCAPE's PartonShower::GetFinalPartons()
#: from IsrManager::Exec() at event 9430 (2026-11-22; one event in ~9.6M); its partial
#: output is kept in PROD_DIR/crashed/
REPLACED_SEEDS = {130001: 130010}
#: validation runs at chosen points (best fit, default): more events
VALIDATION_JOBS = 30

# ── the data ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = HERE / "data"
HEPDATA = {
    "phenix_pi0": ("ins749394", "Figure 1"),                       # 0704.3599, |eta| < 0.35
    "star_jets": ("ins3137518", "Inclusive jet cross section (200 GeV)"),   # 2603.28695
    "star_pi0": ("ins840766", "Figure 22a"),                       # 0912.3838, 0 < eta < 1
    "star_eta_pi0": ("ins840766", "Figure 24a"),
}
#: normalisation uncertainties (fully correlated within a data set)
NORM_UNC = {"phenix_pi0": 0.097, "star_jets": 0.056}
#: observables in the fit, and their pT ranges [GeV] (bins fully inside); from the truncation
#: check (truncation_check.py): pTHat < 5 GeV reaches pi0 only below 5.7 GeV, jets only
#: below 8.2 GeV
FIT = {"phenix_pi0": (6.0, 20.0), "star_jets_eta0": (8.2, 52.0), "star_jets_eta1": (8.2, 52.0)}

# ── jets ─────────────────────────────────────────────────────────────────────────────────
JET_R = 0.5
CONST_PTMIN = 0.2                               # GeV, as STAR's tracks and towers
#: invisible: neutrinos
INVISIBLE = (12, 14, 16)


def point_dir(name: str) -> Path:
    """The events of one design point (or validation run)."""
    return PROD_DIR / "events" / name


def obs_path(name: str) -> Path:
    """The observables of one design point."""
    return PROD_DIR / "obs" / f"{name}.npz"
