#!/usr/bin/env python
"""
utils/container/image_check.py -- the production image's build-time check (Dockerfile.prod),
also runnable in a running container or any environment with the X-SCAPE build:

    python image_check.py                # the production part only (as a -slim image)
    python image_check.py --analysis     # + the analysis packages and a small analysis
    python image_check.py --analysis --viz

Production part: pyjetscape_core imports (no GPU needed), the HDF5 filters, pyyaml,
pelicanfs, jetscape.paths finds the build and the main XML.  --analysis: the packages of
utils/analysis_env/requirements.txt, then one synthetic event through HadronArray ->
to_awkward -> select -> FastJet -> hist -> an mplhep plot (Agg), and MUSIC's EoS table.
--viz: pyvista renders off screen.  Exits 1 on the first failure.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import tempfile


def check(cond, msg):
    if not cond:
        sys.exit(f"image_check: FAIL: {msg}")
    print(f"image_check: ok    {msg}")


def imports(*names):
    versions = []
    for name in names:
        try:
            mod = importlib.import_module(name)
        except ImportError as exc:
            check(False, f"import {name}: {exc}")
        versions.append(f"{name} {getattr(mod, '__version__', '?')}")
    check(True, ", ".join(versions))


def production():
    import jetscape
    from jetscape import paths
    check(jetscape.HAS_CORE, f"pyjetscape_core imports (HAS_ROOT = {jetscape.HAS_ROOT})")
    imports("h5py", "hdf5plugin", "yaml", "pelicanfs")
    check((paths.build_dir() / "music_input").exists(), f"build {paths.build_dir()}")
    check(paths.main_xml().exists(), f"main XML {paths.main_xml()}")


def analysis():
    imports("awkward", "vector", "fastjet", "particle", "hist", "mplhep", "matplotlib",
            "scipy", "pandas", "uproot", "notebook", "jupyterlab", "ipykernel", "ipywidgets")

    import numpy as np
    import awkward as ak
    import fastjet
    import hist
    import matplotlib
    import matplotlib.pyplot as plt
    import mplhep

    import jetscape as js

    matplotlib.use("Agg")
    rng = np.random.default_rng(1)
    n = 200
    pt, eta, phi = rng.exponential(2.0, n) + 0.2, rng.uniform(-2, 2, n), rng.uniform(-3, 3, n)
    m = np.full(n, 0.13957)
    px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    e = np.sqrt(px**2 + py**2 + pz**2 + m**2)
    pid = rng.choice([211, -211, 111, 2212], n).astype(np.int32)
    had = js.HadronArray(pid, np.zeros(n, np.int32), np.stack([e, px, py, pz], 1)
                         .astype(np.float32), None, m.astype(np.float32))
    arr = had.to_awkward()
    charged = arr[js.select(arr, "charged")]
    check(int(ak.sum(ak.num(charged))) == int(np.sum(pid != 111)), "to_awkward + select")
    jets = fastjet.ClusterSequence(charged, fastjet.JetDefinition(fastjet.antikt_algorithm,
                                                                  0.4)).inclusive_jets()
    check(len(jets) == 1 and len(jets[0]) > 0, f"FastJet: {len(jets[0])} jets")
    h = hist.Hist.new.Reg(20, 0, 10, name="pt").Weight().fill(ak.flatten(charged.pt))
    fig, ax = plt.subplots()
    mplhep.histplot(h, ax=ax, yerr=True)
    with tempfile.TemporaryDirectory() as d:
        fig.savefig(os.path.join(d, "check.png"))
        check(os.path.getsize(os.path.join(d, "check.png")) > 0, "hist + mplhep plot (Agg)")
    eos = os.environ.get("MUSIC_EOS_TABLE", "")
    check(os.path.isfile(eos), f"MUSIC_EOS_TABLE {eos}")


def viz():
    imports("pyvista", "vtk")
    import pyvista as pv
    pv.OFF_SCREEN = True
    p = pv.Plotter(off_screen=True, window_size=(64, 64))
    p.add_mesh(pv.Sphere())
    img = p.screenshot(return_img=True)
    p.close()
    check(img.shape[:2] == (64, 64), f"pyvista {pv.__version__} renders off screen")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis", action="store_true")
    ap.add_argument("--viz", action="store_true")
    a = ap.parse_args()
    production()
    if a.analysis:
        analysis()
    if a.viz:
        viz()
    print("image_check: all ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
