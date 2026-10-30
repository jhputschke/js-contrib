# Contribs and repositories

## The contribs

js-contrib is organised like fastjet-contrib: each contrib lives in its own folder under
[`contribs/`](https://github.com/jhputschke/js-contrib/tree/main/contribs) and is switched on
separately.

| Contrib | What it is | Build | Extra dependencies | Docs on this site |
|---|---|---|---|---|
| [PyJetscape](https://github.com/jhputschke/js-contrib/tree/main/contribs/PyJetscape) | pybind11 bindings (`pyjetscape_core`), the `jetscape` Python package with the Pythonic layer, Python modules, the `PyFNOHydro` trampoline, the production drivers and examples | C++ (`-DUSE_JS_PYJETSCAPE=ON`); already built in the container and the Mac wheel | pybind11; h5py, hdf5plugin, pyyaml; PyTorch optional (`[fno]`) | [Python](../python/index.md), [Run](../run/index.md) |
| [FnoHydro](https://github.com/jhputschke/js-contrib/tree/main/contribs/FnoHydro) | Fourier-neural-operator hydrodynamics as a C++ module through LibTorch, plus the bulk ROOT I/O library | C++ (`-DUSE_JS_FNO_HYDRO=ON`) | libtorch (~2 GB), ROOT | [Live PyTorch FNO hydro](../python/fno.md) |
| [FastHydro](https://github.com/jhputschke/js-contrib/tree/main/contribs/FastHydro) | MC-Glauber initial state and a fast 3+1D Milne finite-volume hydro solver, with the MATTER + LBT → CausalLiquefier jet-deposition workflow and paired background/jet evolutions | pure Python, reuses PyJetscape | PyTorch, scipy, h5py, pyyaml | [FastHydro](../python/fasthydro.md) |
| [Visualization](https://github.com/jhputschke/js-contrib/tree/main/contribs/Visualization) | 3D PyVista rendering of the hydro evolution, resampled from Milne to Cartesian coordinates, with a jet parton-shower overlay | pure Python, reuses PyJetscape | pyvista, scipy, vtk, imageio (`imageio-ffmpeg` for `.mp4`) | [3D visualization](../python/visualization.md) |

FastHydro and Visualization have no CMake build of their own. They read the framework through
the PyJetscape bindings and so need a current `pyjetscape_core`.

## The pieces of a working setup

| Component | Repository / image | Role |
|---|---|---|
| X-SCAPE | [JETSCAPE/X-SCAPE](https://github.com/JETSCAPE/X-SCAPE), branch `contrib` | the framework and the C++ physics modules (MATTER, LBT, liquefier, hadronization, …) |
| MUSIC4GPU | [jhputschke/MUSIC4GPU](https://github.com/jhputschke/MUSIC4GPU), branch `XSCAPE` | MUSIC viscous hydro with CUDA and Metal back ends |
| iSS | [jhputschke/iSS](https://github.com/jhputschke/iSS), branch `smash_ready` | Cooper–Frye particlization (oversampling, SMASH-ready output) |
| 3D MC-Glauber | [jhputschke/3dMCGlauber](https://github.com/jhputschke/3dMCGlauber), branch `cap_hard_quark_draws` | the initial state |
| LBT tables | [JETSCAPE/LBT-tables](https://github.com/JETSCAPE/LBT-tables) | LBT's scattering tables (fetched by `get_lbtTab.sh`, or `xscape-fetch-data lbt` with the wheel) |
| js-contrib | [jhputschke/js-contrib](https://github.com/jhputschke/js-contrib) | this package: bindings, drivers, productions, tools, analysis |
| Production images | [`jhputschke/xscape-prod`](https://hub.docker.com/r/jhputschke/xscape-prod) | everything above, built, for Linux with NVIDIA GPUs (amd64 and arm64) |
| Development images | `jhputschke/xscape-fno4d-dev` | the build environment without sources, for a source build on Linux with NVIDIA GPUs |
| Mac wheel | [jhputschke/xscape-macos](https://github.com/jhputschke/xscape-macos) | everything above, built for Apple Silicon with Metal, installed with uv |
| FNO models | [Zenodo record 16647726](https://doi.org/10.5281/zenodo.16647726) | the trained FNO hydro models and their training data ([Live PyTorch FNO hydro](../python/fno.md#where-the-models-come-from)) |

The X-SCAPE scripts `external_packages/get_*.sh` clone MUSIC4GPU, iSS and 3D MC-Glauber at
pinned commits. When you update X-SCAPE, run them again so the pins take effect. See
[Source build](../install/source.md).

## Versions that belong together

| js-contrib | X-SCAPE | MUSIC4GPU | Mac wheel |
|---|---|---|---|
| `main` | `contrib` (head) | `XSCAPE` (pinned by `get_music4gpu.sh`) | v0.1.0 |

The container images and the Mac wheel are built from one consistent set of commits, so there
is nothing to match by hand. With a source build, keep js-contrib `main` together with X-SCAPE
`contrib`.
