#!/usr/bin/env python3
"""
example/python_bulk_h5_writer.py

Write the bulk hydro evolution straight to HDF5 from Python, with no C++ writer involved.

H5BulkWriter is a pure-Python JETSCAPE module (jetscape.fast_h5_bulk).  It reads the hydro
evolution through the existing bindings and writes the FNO4d training schema directly, so
the ROOT file and the root2hdf5 conversion step both disappear.  It also sidesteps ROOT's
1 GB per-object limit, which large native-mode events exceed (README_BulkFast.md section 9).

Three source modes, covering both C++ bulk writer modules:

    --grid-mode native      MUSIC's native store, as-is        (= FastRootBulkWriter native)
    --grid-mode grid        the same, resampled onto a grid    (= FastRootBulkWriter grid)
    --grid-mode framework   the framework AoS, resampled       (= RootBulkWriter)

XML requirements differ by mode and are mutually exclusive:
  * native / grid : <Hydro><MUSIC><dump_hydro_only>1 and <output_evolution_to_memory>1
  * framework     : <output_evolution_to_memory>1 WITHOUT dump_hydro_only
                    (MUSIC then logs "Number of fluid cells received by JETSCAPE: N")

Run from the X-SCAPE build directory (MUSIC resolves its input files relative to the
current directory):

    cd X-SCAPE/build_gpu
    python ../external_packages/js-contrib/contribs/PyJetscape/example/python_bulk_h5_writer.py \
        --user ../config/BulkFastTest/OO_one_event_fast.xml --out hydro_evo.h5

Prerequisites:
    PyJetscape built against X-SCAPE
    h5py, numpy            — writing
    scipy                  — only for --grid-mode grid / framework
    matplotlib (optional)  — quick-look plot
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "python"))

import numpy as np

import jetscape as js
from jetscape.fast_h5_bulk import H5BulkWriter, read_fast_h5_bulk
from jetscape.run_jetscape import run_manual

try:
    import matplotlib.pyplot as plt
    _PLT_AVAILABLE = True
except ImportError:
    _PLT_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--main", default="../config/jetscape_main.xml",
                   help="Main XML config.")
    p.add_argument("--user", default="../config/BulkFastTest/OO_one_event_fast.xml",
                   help="User XML config.")
    p.add_argument("--out", default="hydro_evo.h5", help="Output HDF5 file.")
    p.add_argument("--events", type=int, default=None,
                   help="Override the number of events from the XML.")
    p.add_argument("--grid-mode", default="native", dest="grid_mode",
                   choices=("native", "grid", "framework"),
                   help="Where the cells come from (default: native).")
    p.add_argument("--tau-stride", type=int, default=1, dest="tau_stride",
                   help="native/grid: keep every Nth stored MUSIC step.")
    p.add_argument("--choose-ntau", type=int, default=0, dest="choose_ntau",
                   help="Pin the tau axis so separate jobs produce mergeable files. "
                        "0 (default) grows it to the longest event in this file.")
    # Output grid, for --grid-mode grid / framework.  Same nine keys the C++
    # <FastRootBulkWriter> / <RootBulkWriter> XML blocks take, same meaning: 0 means "use
    # the source grid's value", and the transverse cell counts are derived as
    # 2*int(|min|/d)+1.  If you leave ALL of them at 0 the source grid is used unchanged.
    g = p.add_argument_group("output grid (--grid-mode grid / framework)")
    for key, helptext in (("x_min", "first x cell centre [fm]"),
                          ("dx", "x cell size [fm]"),
                          ("y_min", "first y cell centre [fm]"),
                          ("dy", "y cell size [fm]"),
                          ("eta_min", "first eta cell centre"),
                          ("deta", "eta cell size"),
                          ("tau_min", "first tau frame [fm/c]"),
                          ("dtau", "tau step [fm/c]")):
        g.add_argument(f"--{key.replace('_', '-')}", type=float, default=0.0,
                       dest=key, help=f"{helptext} (0 = source grid)")
    g.add_argument("--ntau", type=int, default=0,
                   help="number of output tau frames (0 = out to the end of the evolution)")
    p.add_argument("--manual", action="store_true",
                   help="Build [initial-state, pre-equilibrium, MUSIC, writer] in Python "
                        "instead of taking the task list from the XML. The user XML must "
                        "then have enableAutomaticTaskListDetermination = false, or the "
                        "XML would build a SECOND pipeline and the writer would read the "
                        "wrong (un-evolved) hydro instance.")
    p.add_argument("--initial-state", default="TrentoInitial", dest="initial_state")
    p.add_argument("--preequilibrium", default="NullPreDynamics")
    p.add_argument("--keep-cpp-writer", action="store_true", dest="keep_cpp_writer",
                   help="Leave the XML's C++ FastRootBulkWriter active (it is deactivated "
                        "by default because it releases MUSIC's store).")
    p.add_argument("--no-plot", action="store_true", dest="no_plot")
    return p.parse_args()


OUT_GRID_KEYS = ("x_min", "dx", "y_min", "dy", "eta_min", "deta", "tau_min", "dtau", "ntau")


def make_writer(args) -> H5BulkWriter:
    out_grid = {k: getattr(args, k) for k in OUT_GRID_KEYS}
    if args.grid_mode != "native" and any(out_grid.values()):
        print(f"output grid: {', '.join(f'{k}={v:g}' for k, v in out_grid.items() if v)}"
              "  (unset keys follow the source grid)")
    return H5BulkWriter(
        out_file_name=args.out,
        grid_mode=args.grid_mode,
        tau_stride=args.tau_stride,
        choose_ntau=args.choose_ntau,
        out_grid=out_grid,
        verbose=True,
    )


def deactivate_cpp_bulk_writers(jetscape, keep: bool) -> None:
    """Take any C++ FastRootBulkWriter in the XML task list out of automatic execution.

    Its Exec() ends with clear_hydro_info_from_memory(), so if it runs first the Python
    writer finds MUSIC's native store already released and skips the event.  They are
    doing the same job here, so the C++ one is switched off unless --keep-cpp-writer.
    An inactive task is also skipped by JetScape.Finish(), which is what we want.
    """
    if not js.HAS_ROOT:
        return
    for task in jetscape.GetTaskList():
        if isinstance(task, js.FastRootBulkWriter):
            if keep:
                print("NOTE: the XML's C++ FastRootBulkWriter is left active; it releases "
                      "MUSIC's store at the end of its Exec(), so the HDF5 writer runs "
                      "first (see example/validate_h5_vs_root.py for the A/B pattern).")
            else:
                task.SetActive(False)
                print("NOTE: deactivated the XML's C++ FastRootBulkWriter "
                      f"({task.GetOutFileName()}) -- it would release MUSIC's store "
                      "before the HDF5 writer sees it. Pass --keep-cpp-writer to keep it.")


def run_xml(args) -> str:
    """Default: the XML builds the task list; Python drives the loop one event at a time.

    The writer is not an XML module, so it cannot be in the XML task list.  Driving the
    loop with JetScapePerEvent lets us call its Exec() between ExecPerEvent() and
    ClearPerEvent(), while the hydro data is still live.  The XML stays the single source
    of the pipeline -- also adding an explicit module list while the XML has
    enableAutomaticTaskListDetermination = true would build the pipeline TWICE, and the
    writer would then read an un-evolved hydro instance.
    """
    writer = make_writer(args)
    jetscape = js.JetScapePerEvent()
    jetscape.SetXMLMainFileName(args.main)
    jetscape.SetXMLUserFileName(args.user)
    jetscape.Init()
    # Init() sets the event count from <nEvents>, so override it afterwards.
    if args.events is not None:
        jetscape.SetNumberOfEvents(args.events)
    deactivate_cpp_bulk_writers(jetscape, args.keep_cpp_writer)

    writer.Init()
    try:
        jetscape.ExecInit()
        for _ in range(jetscape.GetNumberOfEvents()):
            jetscape.ExecPerEvent()
            writer.Exec()                    # hydro data is live here
            jetscape.ClearPerEvent()
        jetscape.Finish()
    finally:
        # JetScape::Finish() calls FinishTasks(), which is a no-op, so sub-task Finish()
        # is never propagated -- close the file by hand.
        writer.Finish()
    return args.out


def run_manual_pipeline(args) -> str:
    """--manual: build the pipeline in Python and let JetScape.Exec() drive the loop.

    Requires enableAutomaticTaskListDetermination = false in the user XML.
    """
    writer = make_writer(args)
    modules = [js.create_module(args.initial_state),
               js.create_module(args.preequilibrium),
               js.create_module("MUSIC"),
               writer]                       # must come after the hydro
    try:
        run_manual(args.main, args.user, modules, n_events=args.events)
    finally:
        writer.Finish()
    return args.out


def summarize(path: str) -> dict:
    d = read_fast_h5_bulk(path, entry_stop=1)
    g, p = d["grid"], d["params"]
    print(f"\n{path}: {int(p['nevents'])} event(s), grid_mode = {d['grid_mode']}")
    print(f"  arr           (nevents, 4, {g['nx']}, {g['ny']}, {g['neta']}, "
          f"{int(p['choose_ntau'])}) float32")
    print(f"  event 0       {d['events'][0].shape}  (ntau, nx, ny, neta, 4)")
    print(f"  tau           {g['tau_min']:.3f} + k*{g['dtau']:.3f} fm/c, "
          f"tau_freezeout = {float(d['tau_freezeout'][0]):.3f}")
    print(f"  features      {d['features']}")
    print("\nFNO4d reads this directly:")
    print("    from loc_libs.read_3d_hdf5 import read_3d_data_hdf5")
    print(f"    read_3d_data_hdf5(['{path}'])")
    return d


def quick_plot(d: dict) -> None:
    if not _PLT_AVAILABLE:
        print("matplotlib not installed — skipping plot.")
        return
    evo, g = d["events"][0], d["grid"]
    ie = evo.shape[3] // 2
    frames = [0, evo.shape[0] // 2, evo.shape[0] - 1]
    fig, axes = plt.subplots(1, len(frames), figsize=(4 * len(frames), 3.6))
    extent = [g["x"][0], g["x"][-1], g["y"][0], g["y"][-1]]
    for ax, k in zip(np.atleast_1d(axes), frames):
        im = ax.imshow(evo[k, :, :, ie, 0].T, origin="lower", extent=extent)
        ax.set_title(f"tau = {g['tau_min'] + k * g['dtau']:.2f} fm/c")
        ax.set_xlabel("x [fm]")
        fig.colorbar(im, ax=ax, label=r"$e$ [GeV/fm$^3$]")
    np.atleast_1d(axes)[0].set_ylabel("y [fm]")
    fig.tight_layout()
    out = "hydro_evo_h5_quicklook.png"
    fig.savefig(out, dpi=120)
    print(f"Wrote {out}")


def main() -> int:
    args = parse_args()
    if not js.HAS_H5PY:
        print("h5py is not installed in this environment (pip install h5py).",
              file=sys.stderr)
        return 1
    path = run_manual_pipeline(args) if args.manual else run_xml(args)
    d = summarize(path)
    if not args.no_plot:
        quick_plot(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
