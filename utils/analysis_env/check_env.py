#!/usr/bin/env python
"""Check an analysis environment for prod_AuAu_0_10_jet output (setup_analysis_env.sh).

    python check_env.py               # packages, Blosc filter, EoS table, off-screen rendering
    python check_env.py out           # + open the production files in out/
    python check_env.py out --no-render
    python check_env.py --remote-test # + read a public object from GCS and from OSDF

Exits 1 if something the analysis needs is missing or a file doesn't open. Optional parts
(visualization, FastHydro, the EoS table, the compiled pyjetscape_core) are reported only.
"""
from __future__ import annotations

import argparse
import glob
import importlib
import os
import platform
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CONTRIBS = os.path.normpath(os.path.join(HERE, "..", "..", "contribs"))
PYJETSCAPE_PY = os.path.join(CONTRIBS, "PyJetscape", "python")
VISUALIZATION = os.path.join(CONTRIBS, "Visualization")
ROOT_EXPORT = os.path.join(CONTRIBS, "PyJetscape", "example", "prod_AuAu_0_10_jet", "root_export")
if PYJETSCAPE_PY not in sys.path:
    sys.path.insert(0, PYJETSCAPE_PY)

REQUIRED = ["numpy", "h5py", "hdf5plugin", "yaml", "uproot", "awkward", "scipy",
            "matplotlib", "pandas", "fastjet", "vector", "notebook", "ipykernel", "ipywidgets"]
VIZ = ["pyvista", "vtk", "imageio", "imageio_ffmpeg"]
BLOSC = 32001                      # HDF5 filter id of Blosc, the default compression

# a pair file is <stem>.h5; the other .h5 next to it are <stem>_particlize / _hadrons_*
SUFFIXES = ("_particlize.h5", "_hadrons_bulk_jet.h5", "_hadrons_bulk_bg.h5",
            "_hadrons_jet_frag.h5")

failed = []


def ok(msg):
    print(f"  ok    {msg}")


def warn(msg):
    print(f"  --    {msg}")


def fail(msg):
    print(f"  FAIL  {msg}")
    failed.append(msg)


def version(name):
    mod = importlib.import_module(name)
    return getattr(mod, "__version__", getattr(mod, "VTK_VERSION", "?"))


def check_packages():
    print(f"Python {platform.python_version()} ({sys.executable}), {platform.machine()}")
    for name in REQUIRED:
        try:
            ok(f"{name} {version(name)}")
        except Exception as exc:  # noqa: BLE001  (a broken install raises anything)
            fail(f"{name}: {exc}")

    import h5py
    import hdf5plugin  # noqa: F401  (registers the filters)
    if h5py.h5z.filter_avail(BLOSC):
        ok("HDF5 Blosc filter")
    else:
        fail("HDF5 Blosc filter not available: the production files will not open")

    try:
        import jetscape
        from jetscape.hadrons_h5 import HadronFileReader  # noqa: F401
        ok(f"jetscape from {os.path.dirname(jetscape.__file__)}")
        if jetscape.HAS_CORE:
            ok("pyjetscape_core is importable (not needed here)")
        else:
            warn("pyjetscape_core not importable (expected: only running X-SCAPE needs it)")
    except Exception as exc:  # noqa: BLE001
        fail(f"jetscape: {exc}")

    # jet_wake.ipynb and analysis/wake_*.py want the file, the visualization takes either
    eos = os.environ.get("MUSIC_EOS_TABLE", "")
    if os.path.isfile(eos):
        ok(f"MUSIC_EOS_TABLE={eos}")
    else:
        warn("MUSIC_EOS_TABLE is not MUSIC's hrg_hotqcd_eos_binary.dat: no e -> T for the "
             "pair files (visualization: conformal EoS; analysis/wake_*.py: pass --eos)")

    try:
        from importlib.metadata import version as dist_version
        ok(f"FastHydro {dist_version('jetscape-fasthydro')} (optional)")
    except Exception:  # noqa: BLE001
        warn("FastHydro not installed (optional; setup_analysis_env.sh --with-fasthydro)")

    try:
        import fastjet
        import numpy as np
        pj = [fastjet.PseudoJet(float(px), 0.0, 0.0, float(abs(px))) for px in (10.0, 5.0)]
        jets = fastjet.ClusterSequence(pj, fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4))
        np.testing.assert_allclose(jets.inclusive_jets()[0].pt(), 15.0)
        ok("fastjet clusters")
    except Exception as exc:  # noqa: BLE001
        fail(f"fastjet clustering: {exc}")


def check_pyroot():
    """PyROOT is optional: run_h5toROOT.py uses it when it imports, else uproot."""
    try:
        import ROOT
        version = ROOT.gROOT.GetVersion()
    except Exception as exc:  # noqa: BLE001
        warn(f"PyROOT not available ({exc.__class__.__name__}; optional): run_h5toROOT.py "
             "writes with uproot. For ROOT see README.md, 'With ROOT'")
        return
    sys.path.insert(0, ROOT_EXPORT)
    import hadrons_to_root as h2r
    if h2r.have_pyroot():
        ok(f"PyROOT {version} (from {os.path.dirname(os.path.dirname(ROOT.__file__))}): "
           "run_h5toROOT.py writes with ROOT")
    else:
        warn(f"PyROOT {version} imports but root_writers.h does not compile: "
             "run_h5toROOT.py writes with uproot")


def check_viz(render):
    print("Visualization")
    have = True
    for name in VIZ:
        try:
            ok(f"{name} {version(name)}")
        except Exception as exc:  # noqa: BLE001
            warn(f"{name}: {exc}")
            have = False
    if not have:
        warn("the Visualization scripts need requirements_viz.txt")
        return

    if not render:
        return
    # in a subprocess: a missing GL/X stack can abort the process instead of raising
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import hydro_pyvista as hp; "
            "import pyvista as pv; hp._maybe_start_xvfb(off_screen=True); "
            "p = pv.Plotter(off_screen=True, window_size=(64, 64)); p.add_mesh(pv.Sphere()); "
            "img = p.screenshot(return_img=True); p.close(); print(img.shape)")
    try:
        r = subprocess.run([sys.executable, "-c", code, VISUALIZATION], capture_output=True,
                           text=True, timeout=120)
        if r.returncode == 0:
            ok(f"off-screen rendering {r.stdout.strip().splitlines()[-1]}")
        else:
            tail = (r.stderr or r.stdout).strip().splitlines()[-1:] or ["no output"]
            warn(f"off-screen rendering failed ({tail[0]}). Headless Linux: install Xvfb "
                 "(apt install xvfb) or run under xvfb-run")
    except subprocess.TimeoutExpired:
        warn("off-screen rendering timed out")


def check_dir(d):
    print(f"Production files in {d}")
    import h5py
    import hdf5plugin  # noqa: F401

    h5 = sorted(glob.glob(os.path.join(d, "*.h5")))
    pairs = [p for p in h5 if not p.endswith(SUFFIXES)]
    for p in pairs:
        try:
            with h5py.File(p, "r") as f:
                shape = f["arr"].shape
                f["arr"][0, 0, 0, 0, :1]                 # decompress one chunk
                ok(f"{os.path.basename(p)}: arr {shape}, eos_kind "
                   f"{f.attrs.get('eos_kind', '?')!s}")
        except Exception as exc:  # noqa: BLE001
            fail(f"{os.path.basename(p)}: {exc}")
    if not pairs:
        warn("no pair files (<stem>.h5)")

    if glob.glob(os.path.join(d, "*_particlize.h5")):
        try:
            from jetscape.hadrons_h5 import HadronFileReader
            with HadronFileReader(d) as r:
                ok(f"HadronFileReader: {r.n_files} files, {r.n_events} events, tags {r.tags()}")
        except Exception as exc:  # noqa: BLE001
            fail(f"HadronFileReader: {exc}")
    else:
        warn("no *_particlize.h5 (hadron-level files)")

    import uproot
    for p in sorted(glob.glob(os.path.join(d, "*.root"))):
        try:
            with uproot.open(p) as f:
                keys = sorted({k.split(";")[0] for k in f.keys(recursive=False)})
                ok(f"{os.path.basename(p)}: {', '.join(keys[:8])}"
                   + (" ..." if len(keys) > 8 else ""))
        except Exception as exc:  # noqa: BLE001
            fail(f"{os.path.basename(p)}: {exc}")


# public objects, readable without credentials
REMOTE_TESTS = {
    "gcs": ("gcsfs", "gcs://gcp-public-data-landsat", {"token": "anon"}),
    "pelican": ("pelicanfs", "osdf:///ospool/uc-shared/public/OSG-Staff/validation/test.txt", {}),
}


def check_remote(test):
    """The optional remote-storage interfaces (--with-gcs, --with-pelican): packages, the
    fsspec URL schemes they register, and with ``test`` one public read each."""
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as dist_version

    print("Remote storage (optional)")
    have = {}
    for what, dists, schemes, flag in (
            ("gcs", ("gcsfs", "google-cloud-storage"), ("gs", "gcs"), "--with-gcs"),
            ("pelican", ("pelicanfs",), ("osdf", "pelican"), "--with-pelican")):
        try:
            got = ", ".join(f"{d} {dist_version(d)}" for d in dists)
        except PackageNotFoundError:
            warn(f"{' / '.join(dists)} not installed ({flag})")
            continue
        try:
            import fsspec
            for sc in schemes:
                fsspec.get_filesystem_class(sc)
            ok(f"{got}: {', '.join(s + '://' for s in schemes)} for fsspec, h5py and uproot")
            have[what] = True
        except Exception as exc:  # noqa: BLE001
            fail(f"{got} installed, but {schemes} are not fsspec schemes: {exc}")
    if not test:
        return
    import fsspec
    for what, (pkg, url, kw) in REMOTE_TESTS.items():
        if not have.get(what):
            continue
        try:
            if what == "gcs":
                fs = fsspec.filesystem("gcs", **kw)
                n = len(fs.ls(url.split("://", 1)[1]))
                ok(f"{pkg}: listed {url} anonymously ({n} entries)")
            else:
                with fsspec.open(url, "rb") as f:
                    ok(f"{pkg}: read {url} ({f.read(64)!r})")
        except Exception as exc:  # noqa: BLE001
            warn(f"{pkg}: could not read {url} ({exc.__class__.__name__}: "
                 f"{str(exc)[:120]}); network or proxy?")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", nargs="?", help="a production directory to open (e.g. out)")
    ap.add_argument("--remote-test", action="store_true",
                    help="read a public object from GCS and OSDF (network)")
    ap.add_argument("--no-render", action="store_true",
                    help="skip the off-screen rendering test")
    args = ap.parse_args(argv)

    check_packages()
    check_pyroot()
    check_viz(render=not args.no_render)
    check_remote(test=args.remote_test)
    if args.dir:
        check_dir(args.dir)

    if failed:
        print(f"\n{len(failed)} problem(s):")
        for m in failed:
            print(f"  - {m}")
        return 1
    print("\nall good")
    return 0


if __name__ == "__main__":
    sys.exit(main())
