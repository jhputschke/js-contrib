# Environment Setup for js-contrib / PyJetscape

The [`utils/conda_install/`](../utils/conda_install/) directory contains scripts to create and verify the
`js_fno` conda environment, which provides all Python and C++ build
dependencies of the contribs: [PyJetscape](PyJetscape/README.md) with its productions and
analyses, [FastHydro](FastHydro/README.md), [FnoHydro](FnoHydro/README.md) and
[Visualization](Visualization/README.md).

**Containers instead of conda.** On Linux (x86_64 and aarch64) there are images:
development images without sources ([`utils/BuildContainerDev.md`](../utils/BuildContainerDev.md)),
and production images with X-SCAPE and PyJetscape built in
([`utils/BuildContainerProd.md`](../utils/BuildContainerProd.md); running a production with
them: [`docs/README_2stage.md`](../docs/README_2stage.md)). On macOS with Apple Silicon
(MPS), use conda.

**Only analysing production files?** The `js_fno` env and an X-SCAPE build aren't needed. See
[Analysis-only install (no X-SCAPE)](#analysis-only-install-no-x-scape) below.

## Contribs at a glance

| Contrib | What it does |
|---------|--------------|
| [FastHydro](FastHydro/README.md) | MC-Glauber initial state + a fast 3+1D Milne FV hydro solver as JETSCAPE modules, with the Matter+LBT → CausalLiquefier jet-deposition workflow; produces paired background/jet evolutions on an identical IC. Pure Python (no build). |
| [FnoHydro](FnoHydro/README.md) | Neural-network (FNO) surrogate hydrodynamics (C++/LibTorch). |
| [PyJetscape](PyJetscape/README.md) | pybind11 Python bindings for the framework + per-event Python workflow. |
| [Visualization](Visualization/README.md) | 3D PyVista rendering of the hydro medium evolution, resampled from Milne `(τ,x,y,η_s)` to Cartesian lab spacetime `(t,x,y,z)`; emits movies, ParaView VTK series, and an interactive viewer. |

## Why conda instead of pip / system packages?

Conda is the recommended approach for two specific platform situations:

1. **Mac Silicon (Apple M-series, `arm64`)** — MPS (Metal Performance Shaders)
   backend for PyTorch is not available in most container images. Building
   natively on macOS with conda gives you a working MPS-accelerated PyTorch
   and an ARM-native ROOT build, which avoids the Rosetta 2 overhead that
   affects x86_64 Docker images on Apple hardware.

2. **Linux `aarch64` (ARM servers, Raspberry Pi clusters, AWS Graviton)** —
   several JETSCAPE C++ dependencies (Pythia8, HepMC3, FastJet) are not
   reliably available as system packages or pre-built containers for ARM.
   conda-forge ships native `aarch64` builds for all of them.

On standard Linux x86_64 with CUDA, conda is still convenient, but the dev and production
images above are a valid alternative.

---

## Files

The scripts are in [`utils/conda_install/`](../utils/conda_install/); the commands below
run from the js-contrib root.

| Script | Purpose |
|--------|---------|
| `utils/conda_install/test_js_fno_build_env.sh` | **Dry-run check** — verifies every package is reachable *without* installing anything. Run this first. |
| `utils/conda_install/install_js_fno_minimal.sh` | Python / ML stack only (PyTorch, ROOT, numpy, …). Use when X-SCAPE is already built. |
| `utils/conda_install/install_js_fno_build_minimal.sh` | **Full install** — Python stack + JETSCAPE C++ build tools (cmake, Boost, Pythia8, HepMC3, …). Use to build X-SCAPE + js-contrib from source inside the environment. |
| `utils/conda_install/pinned/install_js_fno_pinned.sh` | Exact-version pinned variant of the minimal install (for reproducibility). |
| `utils/conda_install/pinned/install_js_fno_build_pinned.sh` | Exact-version pinned variant of the full build install. |

All four install scripts:
- refuse to replace an existing environment of the same name (`JS_FNO_FORCE=1` replaces
  it);
- take another name from `JS_FNO_ENV_NAME=<name>`, e.g. to try a new install next to a
  working `js_fno`;
- install the pip packages with the new environment's own Python, so an active venv
  elsewhere on `PATH` is left alone;
- end with `pip install -e` of PyJetscape and FastHydro when run from a js-contrib
  checkout (`JS_FNO_NO_EDITABLE=1` skips it).

### What each workflow needs

| Workflow | Needs | Covered by |
|---|---|---|
| Build X-SCAPE + PyJetscape (`pyjetscape_core`) | CMake, compilers, Boost, Pythia8, HDF5, GSL, ROOT (optional), pybind11 | the `*_build_*` scripts |
| Productions `prod_AuAu_0_10`, `prod_AuAu_0_10_jet` (`run_prod*.py`, `run_jobs.sh`) | the build; h5py, hdf5plugin, pyyaml | every script |
| Hadronization `hadronize.py` / `run_hadronize.py` | the build with `-DUSE_ISS=ON` | every script |
| `run_h5toROOT.py` (ROOT export) | uproot, awkward; with PyROOT the smaller files | every script |
| Python analyses (`example/analysis`, notebooks) | numpy, scipy, matplotlib, pandas, Jupyter, fastjet + vector (FastJet notebooks) | every script |
| ROOT analyses (`example/analysis_root`) | ROOT ≥ 6.34 for the default RNTuple files | the unpinned scripts (ROOT 6.34–6.37); the pinned ROOT 6.32 reads only `run_h5toROOT.py --format ttree` files |
| FastHydro | numpy, scipy, h5py; PyTorch for the solver | every script |
| `PyFNOHydro`, FnoHydro, FNO4d training | PyTorch, neuraloperator | every script (FnoHydro's C++ build also needs libtorch: its README) |
| Visualization (`contribs/Visualization`) | pyvista (+ vtk), imageio, imageio-ffmpeg | every script |
| Tests (`contribs/*/tests`, `utils/remote_transfer`) | pytest | every script |
| `utils/remote_transfer` (GCS, OSDF) | nothing from conda: each tool makes its own venv | — |

---

## Step 0 — Run the dry-run test first

Before running any install script, check that every required package is
reachable from your network and conda channels:

```bash
bash utils/conda_install/test_js_fno_build_env.sh
```

The script checks:
- CUDA auto-detection (or CPU/MPS fallback)
- conda / mamba availability
- All conda-forge packages (cmake, ROOT, Boost, Pythia8, HepMC3, FastJet, …)
- PyTorch wheel index (PyPI or CUDA-specific index on download.pytorch.org)
- pip-only packages (neuraloperator, uproot, awkward, pyvista, imageio, imageio-ffmpeg)

It prints a coloured `[PASS]` / `[FAIL]` / `[WARN]` line for each item and a
summary at the end.  A non-zero exit code means at least one check failed.

```bash
# Examples
bash utils/conda_install/test_js_fno_build_env.sh          # auto-detect CUDA
bash utils/conda_install/test_js_fno_build_env.sh none     # force CPU/MPS mode (Mac Silicon)
bash utils/conda_install/test_js_fno_build_env.sh 12.1     # force CUDA 12.1 wheel index
```

Only proceed to installation once you see:

```
All checks passed — safe to run install_js_fno_build_minimal.sh.
```

---

## Step 1 — Full install (build from source)

Use `utils/conda_install/install_js_fno_build_minimal.sh` when you need to compile
X-SCAPE and js-contrib from source inside the conda environment.  This is the
typical case for Mac Silicon and Linux `aarch64`.

```bash
# Mac Silicon / any CPU-only system
bash utils/conda_install/install_js_fno_build_minimal.sh none

# Linux with CUDA (auto-detect)
bash utils/conda_install/install_js_fno_build_minimal.sh

# Linux with a specific CUDA version
bash utils/conda_install/install_js_fno_build_minimal.sh 12.1

# Custom Miniconda location (second argument)
bash utils/conda_install/install_js_fno_build_minimal.sh none /opt/miniconda3
```

What the script installs:

| Layer | Packages |
|-------|---------|
| Python runtime | Python 3.11 |
| C++ build tools | cmake, make, compilers (clang/gcc via conda-forge) |
| JETSCAPE C++ deps | boost-cpp, zlib, hdf5, pythia8, hepmc3, fastjet (scikit-hep's: the C++ library and the Python API), gsl, openmpi |
| ROOT | root ≥ 6.34, < 6.38 (conda-forge, ARM-native on macOS/Linux aarch64). 6.38 writes truncated-float TTree leaves that uproot can't read yet |
| PyTorch | torch, torchvision — CPU/MPS build (pip) or CUDA build (pip from pytorch.org) |
| ML / analysis | numpy, matplotlib, scipy, h5py, hdf5plugin, pyyaml, pandas, seaborn, tqdm, jupyterlab, notebook, ipykernel, ipywidgets, networkx, vector |
| Build and test | pybind11, pytest |
| pip | neuraloperator ≥ 2.0, uproot ≥ 5, awkward ≥ 2, pyvista, imageio, imageio-ffmpeg |
| js-contrib | `pip install -e` of PyJetscape and FastHydro (from a checkout) |

The script auto-installs Miniconda if `conda` is not found, downloads the
correct installer for your OS/architecture, and offers to initialise the shell
integration.  It uses `mamba` automatically if it is available (much faster
solver).

---

## Step 2 — Activate and verify

```bash
conda activate js_fno

# Quick sanity checks
python -c "import torch; print('torch', torch.__version__, '| MPS:', torch.backends.mps.is_available())"
python -c "import ROOT; print('ROOT', ROOT.__version__)"
python -c "import neuralop; print('neuraloperator', neuralop.__version__)"
python -c "import jetscape, fast_data; print('jetscape, HAS_CORE =', jetscape.HAS_CORE)"   # False until built
cmake --version
```

On a Mac Silicon machine with a successful install you should see MPS reported
as available:

```
torch 2.x.x | MPS: True
```

---

## Step 3 — Build X-SCAPE + js-contrib

With the environment active, configure X-SCAPE (branch `contrib`, with js-contrib in
`external_packages/`, see the [PyJetscape README](PyJetscape/README.md#installation)):

```bash
conda activate js_fno

cd /path/to/X-SCAPE
mkdir -p build && cd build

cmake .. \
  -DUSE_MUSIC=ON \
  -DUSE_ISS=ON \
  -DUSE_JS_CONTRIB=ON \
  -DUSE_JS_PYJETSCAPE=ON

make -j$(nproc)
```

CMake finds `Boost`, `Pythia8`, `HepMC3`, `ROOT`, etc. automatically from the
active conda environment because conda puts everything under a single prefix
that is already on `CMAKE_PREFIX_PATH` via `$CONDA_PREFIX`. Only the C++ FnoHydro
(`-DUSE_JS_FNO_HYDRO=ON`) needs PyTorch's CMake files as well:
`-DCMAKE_PREFIX_PATH="$CONDA_PREFIX;$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"`.

---

## Python-only install (ML stack only)

If X-SCAPE is already built by other means (e.g. the official JETSCAPE Docker
image on x86_64) and you only need the Python/ML stack to run `PyFNOHydro`:

```bash
bash utils/conda_install/install_js_fno_minimal.sh none     # CPU/MPS
# or
bash utils/conda_install/install_js_fno_minimal.sh          # auto-detect CUDA
```

This installs the same Python packages as the full install (ROOT, PyTorch, the analysis,
visualization and test packages, pybind11) — but **not** the C++ compiler toolchain or
the JETSCAPE build dependencies.

---

## Analysis-only install (no X-SCAPE)

To analyse and visualize the files of a `prod_AuAu_0_10_jet` production without running
X-SCAPE, use [`utils/analysis_env/`](../utils/analysis_env/README.md) instead. It makes a
plain Python venv (≥ 3.10) with no conda, ROOT, compiler or `pyjetscape_core`. It covers the
pair, particlize and hadron HDF5 files, `run_h5toROOT.py` (uproot), the analysis notebooks,
and `wake_pyvista.py` / `hydro_jet_particles_pyvista.py` on the stored files. It also
downloads MUSIC's hotQCD EoS table.

```bash
./utils/analysis_env/setup_analysis_env.sh           # from the js-contrib root; venv in ~/.venvs/js_analysis
source ~/.venvs/js_analysis/bin/activate
python utils/analysis_env/check_env.py /path/to/out  # opens every file of a production
```

FastHydro is optional (`--with-fasthydro`). For PyROOT and the ROOT macros, make the venv
from a conda env with ROOT and add `--system-site-packages`, or use the conda env in
`utils/analysis_env/environment.yml`. See
[With ROOT](../utils/analysis_env/README.md#with-root). For files in Google Cloud Storage or
a Pelican federation (OSDF), `--with-gcs` and `--with-pelican` add the Python interfaces; see
[Remote files](../utils/analysis_env/README.md#remote-files-google-cloud-storage-and-pelicanosdf).

---

## Pinned-version installs

For exact reproducibility (e.g. paper replication), use the pinned scripts in
`utils/conda_install/pinned/`:

```bash
bash utils/conda_install/pinned/install_js_fno_build_pinned.sh none    # full build, CPU/MPS
bash utils/conda_install/pinned/install_js_fno_pinned.sh               # ML stack only, CUDA auto-detect
```

Pinned scripts specify explicit package versions and are tested against the
environment used for the results in
[Phys. Rev. C 113 (2026) 1, 014904](https://doi.org/10.48550/arXiv.2507.23598). The packages
added since (pybind11, pytest, networkx, notebook, vector, pyvista, imageio; fastjet moved from 3.4.2,
no longer on conda-forge, to 3.5.0.1; gsl 2.7.1 to 2.7) are pinned to versions that resolve
with that environment's Python 3.11 and numpy 1.26 (checked by dry runs of conda and pip
on Linux aarch64/x86_64 and macOS arm64, not by a full install). Its ROOT 6.32
predates the RNTuple format, so read `run_h5toROOT.py` files there in the TTree variant
(`--format ttree`).

---

## Visualization contrib — PyVista dependencies

The [Visualization](Visualization/) contrib renders the hydro medium evolution in 3D.
It is **pure Python** — no CMake build of its own. On stored files (`--file`,
`wake_pyvista.py`, `hydro_jet_particles_pyvista.py`) it needs no X-SCAPE build at all.
Live runs reuse the PyJetscape bindings to read the hydro `EvolutionHistory`, so a
current `pyjetscape_core` build is required.  In particular it uses the
`EvolutionHistory.to_numpy_full()` binding (full `(ntau,nx,ny,neta,nf)` grid) for
genuine 3+1D data; if you pull a fresh tree, rebuild it:

```bash
cmake --build <build-dir> --target pyjetscape_core   # e.g. build_gpu
```

Its Python packages are part of `js_fno` (all four install scripts). For another env, e.g.
the analysis venv ([`utils/analysis_env`](../utils/analysis_env/README.md),
`requirements_viz.txt`) or a dedicated one:

| Package | Use |
|---------|-----|
| pyvista | 3D volume + isosurface rendering, movie and VTK export |
| vtk | rendering backend (installed with pyvista) |
| scipy | `RegularGridInterpolator` for the Milne→Cartesian resampling |
| imageio (+ `imageio-ffmpeg` for `.mp4`) | writes the `.gif` / `.mp4` animation |
| numpy | array handling |

```bash
pip install pyvista imageio imageio-ffmpeg        # vtk is pulled in by pyvista
```

For live runs in an env of its own, the compiled `pyjetscape_core` must match that env's
Python (e.g. built for CPython 3.13 → importable only in a 3.13 env); `js_fno` avoids
that, since X-SCAPE is built with its Python.  Live runs execute from the
X-SCAPE build directory (e.g. `build_gpu`) so MUSIC finds its inputs.  Full usage
and options are in [Visualization/README.md](Visualization/README.md).

---

## Troubleshooting

**`conda: command not found` after install**
: The installer does not modify your shell config automatically.  Run the
  init command printed at the end of the install, e.g.:
  ```bash
  ~/miniconda3/bin/conda init zsh   # or bash
  ```
  Then open a new terminal and retry.

**`mamba: command not found` warning**
: Not an error — the scripts fall back to `conda` automatically.  To speed
  up future solves: `conda install -n base mamba -c conda-forge`.

**ROOT not found by CMake on macOS**
: Make sure the environment is active (`conda activate js_fno`) before
  running cmake.  If cmake still does not find ROOT, add the conda prefix
  explicitly:
  ```bash
  cmake .. -DCMAKE_PREFIX_PATH="${CONDA_PREFIX}"
  ```

**`MPS: False` on Apple Silicon**
: Ensure you installed the **CPU/MPS** PyTorch variant (`none` as the CUDA
  argument).  CUDA PyTorch wheels disable MPS.  Also confirm macOS ≥ 12.3
  and that you are running natively (not under Rosetta 2):
  ```bash
  python -c "import platform; print(platform.machine())"  # should print arm64
  ```

**Segfault on `import jetscape` after `import ROOT`**
: Import `torch` before `jetscape` (and before any ROOT import).  See the
  [PyJetscape README](PyJetscape/README.md#prerequisites) for details.

**`Environment 'js_fno' exists already`**
: The install scripts don't replace an environment silently. Remove it
  (`conda env remove -n js_fno`), replace it with `JS_FNO_FORCE=1`, or install next to it
  with `JS_FNO_ENV_NAME=js_fno_new`.

**The pip packages land in another environment, or `torch` is missing after the install**
: Scripts from before 2026-10-01 ran `conda run -n js_fno pip`, which takes the first
  `pip` on `PATH`, e.g. that of an active venv. The current scripts call the new
  environment's own Python. Re-run them, or install the pip packages with
  `$CONDA_PREFIX/bin/python -m pip install ...` in the activated env.
