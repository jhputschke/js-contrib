#!/usr/bin/env python3
"""
examples/python_bulk_root_writer.py

Python equivalent of fno_hydro/root_bulk/bulkRootWriter.cc.

Demonstrates:
  1. Building a JETSCAPE pipeline with Python modules
  2. Using PyBulkRootWriter (a JetScapeModuleBase subclass) to write hydro
     bulk info to a ROOT file via uproot
  3. Reading back and visualising the stored data

Run from the build directory:
    python ../examples/python_bulk_root_writer.py

Prerequisites:
    uproot    (pip install uproot)     — for ROOT output
    matplotlib (pip install matplotlib) — for the quick-look plot
    numpy     (pip install numpy)

Build requirements:
    cmake .. -DUSE_ROOT=ON -DUSE_MUSIC=ON -DUSE_PYTHON=ON
    (same as bulkRootWriter.cc)
"""

from __future__ import annotations

import sys
import os
import time

# ── Add the python/ directory to the search path ─────────────────────────────
# When running from the build directory, the pyjetscape_core.so is installed
# into <repo>/python/jetscape/ by the CMake rule:
#   set_target_properties(pyjetscape_core PROPERTIES
#       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/jetscape")
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "python"))

# ── JETSCAPE Python bindings ──────────────────────────────────────────────────
import python.jetscape as js
from python.jetscape.bulk_root_writer import PyBulkRootWriter

# ── Optional visualisation ────────────────────────────────────────────────────
try:
    import numpy as np
    import matplotlib.pyplot as plt
    _PLT_AVAILABLE = True
except ImportError:
    _PLT_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MAIN_XML = "../config/jetscape_main.xml"
USER_XML = "../fno_hydro/config/jetscape_user_root_bulk_test.xml"
OUTPUT   = "bulk_root_writer_python_test.root"

# User-resolution grid parameters (mirrors bulkRootWriter.cc defaults)
TAU_MAX    = 5.0   # fm/c
D_TAU      = 0.1   # fm/c
D_X        = 0.5   # fm
N_FEATURES = 3     # energy_density, vx, vy


def build_pipeline() -> tuple:
    """
    Build the JETSCAPE pipeline and return (jetscape, bulk_writer).

    The layout mirrors bulkRootWriter.cc::main():
        [ TrentoInitial  →  NullPreDynamics  →  MpiMusic  →  PyBulkRootWriter ]
    """
    # ── Top-level controller ──────────────────────────────────────────────────
    jetscape = js.JetScape()
    jetscape.SetXMLMainFileName(MAIN_XML)
    jetscape.SetXMLUserFileName(USER_XML)

    # ── Physics modules ───────────────────────────────────────────────────────
    trento        = js.create_module("TrentoInitial")
    null_predyn   = js.create_module("NullPreDynamics")
    hydro         = js.create_module("MUSIC")

    # Keep bulk_info alive after JetScapeTask::ClearTasks() so that
    # PyBulkRootWriter.Exec() can still access it.
    # hydro.set_preserve_bulk_info(True) #not needed in this use context!!!

    jetscape.Add(trento)
    jetscape.Add(null_predyn)
    jetscape.Add(hydro)

    # ── Python BulkRootWriter ─────────────────────────────────────────────────
    bulk_writer = PyBulkRootWriter(
        output_name = OUTPUT,
        n_features  = N_FEATURES,
        tau_max     = TAU_MAX,
        d_tau       = D_TAU,
        d_x         = D_X,
        verbose     = True,
    )
    jetscape.Add(bulk_writer)

    return jetscape, hydro, bulk_writer


def show() -> None:
    print("----------------------------------------------------------")
    print("| Python Bulk ROOT Writer  –  JETSCAPE Framework         |")
    print("----------------------------------------------------------")


def run() -> None:
    show()

    js_logger = __import__(
        "python.jetscape.pyjetscape_core", fromlist=["pyjetscape_core"]
    )

    t_wall = time.time()

    jetscape, hydro, bulk_writer = build_pipeline()

    # ── Run simulation ────────────────────────────────────────────────────────
    jetscape.Init()
    jetscape.Exec()
    jetscape.Finish()

    # JetScape::Finish() calls JetScapeTask::FinishTasks() which is a no-op
    # and does NOT propagate Finish() to sub-tasks.  Call it explicitly so
    # the ROOT file is actually written (mirrors ~BulkRootWriter() in the C++).
    # should be fixed in X-SCAPE!!!
    bulk_writer.Finish()

    elapsed = time.time() - t_wall
    print(f"\nFinished in {elapsed:.1f} s")

    # ── Quick-look visualisation ──────────────────────────────────────────────
    if _PLT_AVAILABLE and os.path.exists(OUTPUT):
        _quick_plot(OUTPUT)


def _quick_plot(root_path: str) -> None:
    """
    Read the first event from the ROOT file and plot the energy-density
    map at a few tau slices.
    """
    data = PyBulkRootWriter.read_root(root_path)
    user_res = data["user_res"]      # shape (n_events, nx, ny, ntau, nfeatures)
    gi       = data["grid_info"]

    print(f"\nLoaded {user_res.shape[0]} event(s) from {root_path}")
    print(f"  grid: nx={data['nx']}, ny={data['ny']}, "
          f"ntau={data['ntau']}, nfeatures={data['nfeatures']}")
    print(f"  tau0={gi['tau0']:.2f} fm/c, dtau={gi['dtau']:.2f} fm/c, "
          f"x_min={gi['x_min']:.1f} fm, dx={gi['dx']:.2f} fm")

    ev    = user_res[0]              # (nx, ny, ntau, nfeatures)
    nx    = data["nx"]
    ntau  = data["ntau"]
    x_min = gi["x_min"]
    dx    = gi["dx"]
    tau0  = gi["tau0"]
    dtau  = gi["dtau"]

    # Choose three tau slices
    tau_slices = [1, ntau // 2, ntau - 1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, k in zip(axes, tau_slices):
        tau_val = tau0 + k * dtau
        ed_map  = ev[:, :, k, 0]   # energy density at tau slice k

        x_arr = np.linspace(x_min, x_min + (nx - 1) * dx, nx)
        im = ax.pcolormesh(
            x_arr, x_arr, ed_map.T,
            cmap="hot_r", vmin=0,
        )
        ax.set_title(rf"$ed$ at $\tau = {tau_val:.1f}$ fm/c")
        ax.set_xlabel("x [fm]")
        ax.set_ylabel("y [fm]")
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, label="GeV/fm³")

    fig.suptitle(f"PyBulkRootWriter — {os.path.basename(root_path)} "
                 f"(event 0, energy density)")
    plt.tight_layout()
    plot_file = os.path.splitext(root_path)[0] + "_quick_look.pdf"
    plt.savefig(plot_file, bbox_inches="tight")
    print(f"\nSaved quick-look plot to {plot_file}")


if __name__ == "__main__":
    run()
