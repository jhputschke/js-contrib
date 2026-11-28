"""
analysis/OO-RHIC/_paths.py -- puts analysis/OO-RHIC and its test directories on sys.path, so that
every module imports the others by name wherever it lives (python X.py in any directory, the
notebooks, imports from elsewhere).

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # in a test directory
    import _paths                                                      # noqa: E402,F401

The shared data stay here, in ROOT: results/, data/, design/ (generated, git-ignored) and refs/.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
#: the test directories (each with a README.md); the final O+O analysis stays in ROOT
GROUPS = ("hard_tune", "bulk_tune", "closure_bias", "g2_mpi_check", "oo_followup", "auau_calibration",
          "hadronization")
RESULTS = ROOT / "results"

for _d in reversed([ROOT, *(ROOT / g for g in GROUPS)]):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))
