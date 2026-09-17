#!/usr/bin/env python3
"""
example/python_fast_bulk_root_writer.py

Run the X-SCAPE C++ FastRootBulkWriter from Python.

FastRootBulkWriter is the hydro-only ROOT dump: it reads MUSIC's native
in-memory evolution store directly and never builds bulk_info.data, so it is
much faster and lighter than PyBulkRootWriter (python_bulk_root_writer.py).
It is a C++ module configured from the XML; Python adds it to the pipeline,
drives the run and reads the output back.

Demonstrates:
  1. XML task list (default): the writer comes from the user XML's
     <FastRootBulkWriter> block and is found in JetScape.GetTaskList().
  2. Manual pipeline (--manual): create_module("FastRootBulkWriter") appended
     after the hydro module.
  3. --check-numpy: per-event loop that also copies MUSIC's native store into
     numpy (MpiMusic.get_native_evolution_numpy) and checks it matches the
     ROOT file bit-for-bit.
  4. Reading the file back (jetscape.fast_root_bulk) and a quick-look plot.

Requirements in the user XML:
  * <Hydro><MUSIC><dump_hydro_only>1  and  <output_evolution_to_memory>1
  * a <FastRootBulkWriter> block (out_file_name, grid_mode, tau_stride, ...)
  * <enableAutomaticTaskListDetermination> true (default) / false (--manual)

Run from the X-SCAPE build directory (MUSIC resolves its input files relative
to the current directory):
    cd X-SCAPE/build_gpu
    python ../external_packages/js-contrib/contribs/PyJetscape/example/python_fast_bulk_root_writer.py \\
        --user ../config/BulkFastTest/OO_one_event_fast.xml

Prerequisites:
    PyJetscape built against a ROOT-enabled X-SCAPE (cmake -DUSE_ROOT=ON)
    uproot, numpy          — reading the output
    matplotlib (optional)  — quick-look plot
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# ── Make sure the python package is importable ────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "python"))

import numpy as np

import jetscape as js
from jetscape.fast_root_bulk import read_fast_root_bulk

try:
    import matplotlib.pyplot as plt
    _PLT_AVAILABLE = True
except ImportError:
    _PLT_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--main", default="../config/jetscape_main.xml",
                   help="Main XML config (default: ../config/jetscape_main.xml).")
    p.add_argument("--user", default="../config/BulkFastTest/OO_one_event_fast.xml",
                   help="User XML config with dump_hydro_only and a "
                        "<FastRootBulkWriter> block.")
    p.add_argument("--events", type=int, default=None,
                   help="Override the number of events from the XML.")
    p.add_argument("--manual", action="store_true",
                   help="Build [initial-state, pre-equilibrium, MUSIC, "
                        "FastRootBulkWriter] in Python. The user XML must have "
                        "enableAutomaticTaskListDetermination = false.")
    p.add_argument("--initial-state", default="TrentoInitial", dest="initial_state",
                   help="Manual mode: initial-state module name.")
    p.add_argument("--preequilibrium", default="NullPreDynamics",
                   help="Manual mode: pre-equilibrium module name.")
    p.add_argument("--check-numpy", action="store_true", dest="check_numpy",
                   help="Per-event loop that also reads MUSIC's native store "
                        "into numpy and compares it with the ROOT file.")
    p.add_argument("--no-plot", action="store_true", dest="no_plot",
                   help="Skip the quick-look plot.")
    return p.parse_args()


def find_writer(jetscape) -> "js.FastRootBulkWriter":
    """Return the FastRootBulkWriter in the (XML-built) top-level task list."""
    for task in jetscape.GetTaskList():
        if isinstance(task, js.FastRootBulkWriter):
            return task
    raise RuntimeError("No FastRootBulkWriter in the task list — add a "
                       "<FastRootBulkWriter> block to the user XML.")


def setup(args, driver_cls):
    """Configure and Init() a driver; return (jetscape, writer)."""
    jetscape = driver_cls()
    jetscape.SetXMLMainFileName(args.main)
    jetscape.SetXMLUserFileName(args.user)

    writer = None
    if args.manual:
        writer = js.create_module("FastRootBulkWriter")
        for mod in (js.create_module(args.initial_state),
                    js.create_module(args.preequilibrium),
                    js.create_module("MUSIC"),
                    writer):                  # must come after the hydro
            jetscape.Add(mod)

    # Init() builds the XML task list (automatic mode) and reads the writer's
    # <FastRootBulkWriter> settings.
    jetscape.Init()
    # Init() sets the event count from <nEvents>, so override it afterwards.
    if args.events is not None:
        jetscape.SetNumberOfEvents(args.events)
    if writer is None:
        writer = find_writer(jetscape)
    return jetscape, writer


def run(args) -> str:
    """Standard run: JetScape.Exec() drives the event loop, writer fills per event."""
    jetscape, writer = setup(args, js.JetScape)
    print(f"FastRootBulkWriter: out_file_name = {writer.GetOutFileName()}, "
          f"grid_mode = {writer.GetGridMode()}, tau_stride = {writer.GetTauStride()}")

    jetscape.Exec()
    jetscape.Finish()                      # propagates Finish() -> file written

    layout = writer.get_event_layout()
    print(f"Wrote {layout['events_written']} event(s); last event shape "
          f"(ntau, nx, ny, neta, nF) = {layout['shape']}")
    return writer.GetOutFileName()


def run_check_numpy(args) -> str:
    """Per-event loop reading the native store into numpy before the writer runs.

    The writer releases MUSIC's native store at the end of its Exec(), so it is
    taken out of the automatic per-event execution (SetActive(False) before
    ExecInit) and called by hand after the numpy copy.  An inactive task is
    also skipped by JetScape.Finish(), so the writer is finished explicitly.
    """
    jetscape, writer = setup(args, js.JetScapePerEvent)
    writer.SetActive(False)
    jetscape.ExecInit()

    first = None                           # keep event 0 only (~1.3 GB native)
    for _ in range(jetscape.GetNumberOfEvents()):
        jetscape.ExecPerEvent()
        hydro = js.JetScapeSignalManager.Instance().GetHydroPointer()
        t0 = time.perf_counter()
        evo = hydro.get_native_evolution_numpy(tau_stride=writer.GetTauStride())
        print(f"[event {jetscape.GetCurrentEvent()}] native store -> numpy "
              f"{evo.shape} in {time.perf_counter() - t0:.2f} s")
        if first is None:
            first = evo
        del evo
        writer.Exec()                      # write this event, release the store
        jetscape.ClearPerEvent()

    jetscape.Finish()
    writer.Finish()

    if writer.GetGridMode() != "native":
        print("--check-numpy compares native-mode output only; skipping check.")
        return writer.GetOutFileName()

    b = read_fast_root_bulk(writer.GetOutFileName(), entry_stop=1)["events"][0]
    same = first.shape == b.shape and np.array_equal(first, b)
    print(f"  event 0: numpy {first.shape} vs ROOT {b.shape} -> "
          f"bit-for-bit equal = {same}")
    return writer.GetOutFileName()


def quick_plot(root_path: str) -> None:
    """Energy density at mid-rapidity for three tau slices of the first event."""
    d = read_fast_root_bulk(root_path, entry_stop=1)
    evo, g = d["events"][0], d["grid"]
    ntau, nx, ny, neta, _ = evo.shape
    print(f"Loaded event 0 from {root_path}: grid_mode = {d['grid_mode']}, "
          f"shape = {evo.shape}")

    ieta = neta // 2
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, k in zip(axes, [ntau // 4, ntau // 2, ntau - 1]):
        im = ax.pcolormesh(g["x"], g["y"], evo[k, :, :, ieta, 0].T,
                           cmap="hot_r", vmin=0)
        ax.set_title(rf"$\epsilon$ at $\tau = {g['tau_min'] + k * g['dtau']:.2f}$ fm/c, "
                     rf"$\eta_s = {g['eta'][ieta]:.2f}$")
        ax.set_xlabel("x [fm]")
        ax.set_ylabel("y [fm]")
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, label="GeV/fm³")

    fig.suptitle(f"FastRootBulkWriter — {os.path.basename(root_path)} (event 0)")
    plt.tight_layout()
    plot_file = os.path.splitext(root_path)[0] + "_quick_look.pdf"
    plt.savefig(plot_file, bbox_inches="tight")
    print(f"Saved quick-look plot to {plot_file}")


def main() -> None:
    args = parse_args()
    if not js.HAS_ROOT:
        sys.exit("PyJetscape was built without ROOT (HAS_ROOT = False), so "
                 "FastRootBulkWriter is not available. Rebuild X-SCAPE with "
                 "-DUSE_ROOT=ON.")

    t_wall = time.time()
    out = run_check_numpy(args) if args.check_numpy else run(args)
    print(f"Finished in {time.time() - t_wall:.1f} s")

    if _PLT_AVAILABLE and not args.no_plot and os.path.exists(out):
        quick_plot(out)


if __name__ == "__main__":
    main()
