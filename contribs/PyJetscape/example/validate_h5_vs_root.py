#!/usr/bin/env python3
"""
example/validate_h5_vs_root.py

Run the C++ FastRootBulkWriter and the Python H5BulkWriter on the SAME events and compare.

Both writers see one run, one MUSIC store, one set of seeds, so there is no RNG doubt --
and this needs no change to any C++ code.  It uses the JetScapePerEvent driver: the C++
writer is taken out of automatic execution with SetActive(False), the HDF5 writer runs
first with clear_after_write=False (so it does not release the store), and the C++ writer
is then called by hand, writes, and frees.

What "agreement" means depends on the source mode:

  --grid-mode native   BITWISE.  Both walk the same MUSIC store and apply the same
                       static_cast<float>, so np.array_equal must hold exactly.
  --grid-mode grid     ~1e-7 relative.  Jetscape::real is float (src/framework/RealType.h:26),
  --grid-mode framework  so EvolutionHistory::get() does its own blend in float32 and in a
                       different summation order; this code interpolates in float64 and is
                       the more accurate of the two.  Compare with allclose, never
                       array_equal.

The XML must have a <FastRootBulkWriter> block, plus <dump_hydro_only>1 and
<output_evolution_to_memory>1.

    cd X-SCAPE/build_gpu
    python ../external_packages/js-contrib/contribs/PyJetscape/example/validate_h5_vs_root.py \
        --user ../config/BulkFastTest/OO_one_event_fast.xml
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "python"))

import numpy as np

import jetscape as js
from jetscape.fast_h5_bulk import H5BulkWriter, read_fast_h5_bulk
from jetscape.fast_root_bulk import read_fast_root_bulk

#: FNO4d's `_SCALAR_KEYS` that the ROOT files also carry, for the attribute cross-check
SHARED_KEYS = ("nFeatures", "nx", "ny", "neta", "x_min", "y_min", "eta_min",
               "dx", "dy", "deta", "tau_min", "dtau")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--main", default="../config/jetscape_main.xml")
    p.add_argument("--user", default="../config/BulkFastTest/OO_one_event_fast.xml")
    p.add_argument("--h5", default="validate_h5_vs_root.h5", help="HDF5 output path.")
    p.add_argument("--events", type=int, default=None,
                   help="Override the number of events from the XML.")
    p.add_argument("--grid-mode", default=None, dest="grid_mode",
                   choices=("native", "grid", "framework"),
                   help="Default: whatever the C++ writer's <grid_mode> says.")
    p.add_argument("--rtol", type=float, default=1e-6,
                   help="Tolerance for the non-native modes, on max|delta| / max|ROOT| "
                        "per channel. Expect ~2e-7 (float32 epsilon).")
    return p.parse_args()


def find_root_writer(jetscape):
    for task in jetscape.GetTaskList():
        if isinstance(task, js.FastRootBulkWriter):
            return task
    raise RuntimeError("No FastRootBulkWriter in the task list — add a "
                       "<FastRootBulkWriter> block to the user XML.")


#: the output-grid keys FastRootBulkWriter reads from its own XML block
OUT_GRID_KEYS = ("x_min", "dx", "y_min", "dy", "eta_min", "deta", "tau_min", "dtau")


def read_out_grid(module, tag="FastRootBulkWriter"):
    """Read the C++ writer's output-grid settings back out of the loaded XML.

    JetScapeModuleBase exposes the XML readers, and the XML is a global singleton by the
    time Init() has run, so the Python writer can be handed exactly the same grid.
    """
    grid = {k: module.get_xml_element_double([tag, k], False) for k in OUT_GRID_KEYS}
    grid["ntau"] = module.get_xml_element_int([tag, "ntau"], False)
    return grid


def run_both(args):
    """One run, both writers.  Returns (root_path, h5_path, grid_mode)."""
    jetscape = js.JetScapePerEvent()
    jetscape.SetXMLMainFileName(args.main)
    jetscape.SetXMLUserFileName(args.user)
    jetscape.Init()
    if args.events is not None:
        jetscape.SetNumberOfEvents(args.events)

    root_writer = find_root_writer(jetscape)
    # Out of automatic per-event execution: its Exec() releases MUSIC's native store, and
    # we need to read that store first.  An inactive task is also skipped by
    # JetScape.Finish(), so it is finished by hand below.
    root_writer.SetActive(False)

    grid_mode = args.grid_mode or root_writer.GetGridMode()
    # In grid/framework mode the C++ writer resamples onto the grid in its XML block, so
    # read the same numbers back out or the two would write different grids.
    out_grid = read_out_grid(root_writer) if grid_mode != "native" else None
    if out_grid:
        print(f"output grid from <FastRootBulkWriter>: {out_grid}")
    h5 = H5BulkWriter(out_file_name=args.h5, grid_mode=grid_mode,
                      tau_stride=root_writer.GetTauStride(),
                      out_grid=out_grid,
                      clear_after_write=False, verbose=True)
    h5.Init()

    jetscape.ExecInit()
    for _ in range(jetscape.GetNumberOfEvents()):
        jetscape.ExecPerEvent()
        t0 = time.perf_counter()
        h5.Exec()                       # reads the store, does NOT free it
        dt = time.perf_counter() - t0
        root_writer.Exec()              # writes ROOT, then frees
        print(f"[event {jetscape.GetCurrentEvent()}] HDF5 write {dt:.2f} s")
        jetscape.ClearPerEvent()

    jetscape.Finish()
    root_writer.Finish()
    h5.Finish()
    return root_writer.GetOutFileName(), args.h5, grid_mode


def compare(root_path, h5_path, grid_mode, rtol):
    root = read_fast_root_bulk(root_path)
    h5d = read_fast_h5_bulk(h5_path)
    exact = grid_mode == "native"
    ok = True

    n_root, n_h5 = len(root["events"]), len(h5d["events"])
    print(f"\nevents: ROOT {n_root}, HDF5 {n_h5}")
    if n_root != n_h5:
        print("  FAIL: event counts differ")
        return False

    print(f"\nper-event data ({'bitwise' if exact else f'per-channel max|delta| / '
                                                          f'max|ROOT| <= {rtol:g}'}):")
    for i, (a, b) in enumerate(zip(h5d["events"], root["events"])):
        if a.shape != b.shape:
            print(f"  event {i}: FAIL shape {a.shape} vs {b.shape}")
            ok = False
            continue
        if exact:
            same = np.array_equal(a, b)
            print(f"  event {i}: {a.shape}  {'bit-for-bit equal' if same else 'DIFFERS'}")
        else:
            # Per-element relative tolerance (np.allclose with atol=0) is the wrong test
            # here: the energy density goes to zero in the vacuum cells, so a 1e-20
            # absolute difference there is a huge *relative* one and says nothing.  Scale
            # each channel by its own maximum instead.
            same = True
            print(f"  event {i}: {a.shape}")
            for c, name in enumerate(("energy_density", "vx", "vy", "vz")):
                scale = max(float(np.abs(b[..., c]).max()), 1e-30)
                rel = float(np.abs(a[..., c] - b[..., c]).max()) / scale
                good = rel <= rtol
                same &= good
                print(f"    {name:<15} max|delta| / max|ROOT| = {rel:.3e}  "
                      f"{'ok' if good else 'DIFFERS'}")
        ok &= bool(same)

    print("\nper-event scalars:")
    for name, key in (("ntau_freezeout", "ntau"), ("tau_freezeout", "tau_freezeout")):
        a, b = np.asarray(h5d[key]), np.asarray(root[key])
        same = a.shape == b.shape and np.array_equal(a, b)
        print(f"  {name:<16} {'equal' if same else f'DIFFERS: {a} vs {b}'}")
        ok &= bool(same)

    # Compare against read_fast_root_bulk's CORRECTED grid, not the raw TParameters: in
    # native mode FastRootBulkWriter.cc:89-92 stores the unused user-grid x_min/dx/... (all
    # zero unless the XML sets them) and puts the real values under the *_MUSIC keys.  That
    # is also why root2hdf5/root_to_hdf5.py, which copies the raw keys, produces x_min=0
    # for native-mode files -- H5BulkWriter writes the real grid instead.
    print("\nshared grid attributes (vs read_fast_root_bulk's corrected grid):")
    for k in SHARED_KEYS:
        want, got = root["grid"].get(k, root["params"].get(k)), h5d["params"].get(k)
        if want is None:
            continue
        same = got is not None and np.isclose(float(got), float(want), rtol=1e-6, atol=0)
        print(f"  {k:<16} ROOT {want!r:>12}  HDF5 {got!r:>12}  {'ok' if same else 'DIFFERS'}")
        ok &= bool(same)

    print("\nschema invariants:")
    import h5py
    with h5py.File(h5_path, "r") as f:
        checks = [
            ("attrs['nevents'] == arr.shape[0]",
             int(f.attrs["nevents"]) == f["arr"].shape[0]),
            ("attrs['choose_ntau'] == arr.shape[5]",
             int(f.attrs["choose_ntau"]) == f["arr"].shape[5]),
            ("arr.chunks[-1] == 1 (one tau frame per chunk)", f["arr"].chunks[-1] == 1),
            ("arr dtype float32", f["arr"].dtype == np.float32),
            ("padding past ntau_freezeout is exactly 0.0",
             all(np.all(f["arr"][i, ..., int(n):] == 0.0)
                 for i, n in enumerate(f["ntau_freezeout"][:]))),
            ("/arr layout == root_to_hdf5's moveaxis convention",
             all(np.array_equal(f["arr"][i, ..., :int(n)],
                                np.moveaxis(root["events"][i], (0, 4), (4, 0)))
                 for i, n in enumerate(f["ntau_freezeout"][:])) if exact else True),
        ]
    for label, good in checks:
        print(f"  {'ok  ' if good else 'FAIL'} {label}")
        ok &= bool(good)

    print(f"\n{'PASS' if ok else 'FAIL'}: HDF5 output "
          f"{'matches' if ok else 'does NOT match'} the ROOT path ({grid_mode} mode).")
    return ok


def main() -> int:
    args = parse_args()
    if not js.HAS_ROOT:
        print("PyJetscape was built without ROOT — nothing to compare against.",
              file=sys.stderr)
        return 1
    if not js.HAS_H5PY:
        print("h5py is not installed in this environment.", file=sys.stderr)
        return 1
    root_path, h5_path, grid_mode = run_both(args)
    return 0 if compare(root_path, h5_path, grid_mode, args.rtol) else 1


if __name__ == "__main__":
    raise SystemExit(main())
