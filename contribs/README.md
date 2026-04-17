# Environment Setup for js-contrib / PyJetscape

The `conda_install/` subdirectory contains scripts to create and verify the
`js_fno` conda environment, which provides all Python and C++ build
dependencies needed by the [PyJetscape](PyJetscape/README.md) and
[FnoHydro](FnoHydro/README.md) contribs.

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

On standard Linux x86_64 with CUDA, conda is still convenient but the
official JETSCAPE Docker images are a valid alternative.

---

## Files

| Script | Purpose |
|--------|---------|
| `conda_install/test_js_fno_build_env.sh` | **Dry-run check** — verifies every package is reachable *without* installing anything. Run this first. |
| `conda_install/install_js_fno_minimal.sh` | Python / ML stack only (PyTorch, ROOT, numpy, …). Use when X-SCAPE is already built. |
| `conda_install/install_js_fno_build_minimal.sh` | **Full install** — Python stack + JETSCAPE C++ build tools (cmake, Boost, Pythia8, HepMC3, …). Use to build X-SCAPE + js-contrib from source inside the environment. |
| `conda_install/pinned/install_js_fno_pinned.sh` | Exact-version pinned variant of the minimal install (for reproducibility). |
| `conda_install/pinned/install_js_fno_build_pinned.sh` | Exact-version pinned variant of the full build install. |

---

## Step 0 — Run the dry-run test first

Before running any install script, check that every required package is
reachable from your network and conda channels:

```bash
bash conda_install/test_js_fno_build_env.sh
```

The script checks:
- CUDA auto-detection (or CPU/MPS fallback)
- conda / mamba availability
- All conda-forge packages (cmake, ROOT, Boost, Pythia8, HepMC3, FastJet, …)
- PyTorch wheel index (PyPI or CUDA-specific index on download.pytorch.org)
- pip-only packages (neuraloperator, uproot, awkward)

It prints a coloured `[PASS]` / `[FAIL]` / `[WARN]` line for each item and a
summary at the end.  A non-zero exit code means at least one check failed.

```bash
# Examples
bash conda_install/test_js_fno_build_env.sh          # auto-detect CUDA
bash conda_install/test_js_fno_build_env.sh none     # force CPU/MPS mode (Mac Silicon)
bash conda_install/test_js_fno_build_env.sh 12.1     # force CUDA 12.1 wheel index
```

Only proceed to installation once you see:

```
All checks passed — safe to run install_js_fno_build_minimal.sh.
```

---

## Step 1 — Full install (build from source)

Use `conda_install/install_js_fno_build_minimal.sh` when you need to compile
X-SCAPE and js-contrib from source inside the conda environment.  This is the
typical case for Mac Silicon and Linux `aarch64`.

```bash
# Mac Silicon / any CPU-only system
bash conda_install/install_js_fno_build_minimal.sh none

# Linux with CUDA (auto-detect)
bash conda_install/install_js_fno_build_minimal.sh

# Linux with a specific CUDA version
bash conda_install/install_js_fno_build_minimal.sh 12.1

# Custom Miniconda location (second argument)
bash conda_install/install_js_fno_build_minimal.sh none /opt/miniconda3
```

What the script installs:

| Layer | Packages |
|-------|---------|
| Python runtime | Python 3.11 |
| C++ build tools | cmake, make, compilers (clang/gcc via conda-forge) |
| JETSCAPE C++ deps | boost-cpp, zlib, hdf5, pythia8, hepmc3, fastjet, gsl, openmpi |
| ROOT | root (conda-forge, ARM-native on macOS/Linux aarch64) |
| PyTorch | torch, torchvision — CPU/MPS build (pip) or CUDA build (pip from pytorch.org) |
| ML / analysis | numpy, matplotlib, scipy, h5py, seaborn, tqdm, jupyterlab, ipykernel |
| pip-only | neuraloperator ≥ 2.0, uproot, awkward |

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
python -c "import neuraloperator; print('neuraloperator OK')"
cmake --version
```

On a Mac Silicon machine with a successful install you should see MPS reported
as available:

```
torch 2.x.x | MPS: True
```

---

## Step 3 — Build X-SCAPE + js-contrib

With the environment active, configure CMake pointing to the conda-provided
libraries:

```bash
conda activate js_fno

cd /path/to/X-SCAPE
mkdir -p build && cd build

cmake .. \
  -DUSE_MUSIC=ON \
  -DUSE_ISS=ON \
  -DUSE_JS_CONTRIB=ON \
  -DUSE_JS_PYJETSCAPE=ON \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"

make -j$(nproc)
```

CMake finds `Boost`, `Pythia8`, `HepMC3`, `ROOT`, etc. automatically from the
active conda environment because conda puts everything under a single prefix
that is already on `CMAKE_PREFIX_PATH` via `$CONDA_PREFIX`.

---

## Python-only install (ML stack only)

If X-SCAPE is already built by other means (e.g. the official JETSCAPE Docker
image on x86_64) and you only need the Python/ML stack to run `PyFNOHydro`:

```bash
bash conda_install/install_js_fno_minimal.sh none     # CPU/MPS
# or
bash conda_install/install_js_fno_minimal.sh          # auto-detect CUDA
```

This installs Python, ROOT, PyTorch, numpy, neuraloperator, uproot, and
JupyterLab — but **not** the C++ compiler toolchain or JETSCAPE build
dependencies.

---

## Pinned-version installs

For exact reproducibility (e.g. paper replication), use the pinned scripts in
`conda_install/pinned/`:

```bash
bash conda_install/pinned/install_js_fno_build_pinned.sh none    # full build, CPU/MPS
bash conda_install/pinned/install_js_fno_pinned.sh               # ML stack only, CUDA auto-detect
```

Pinned scripts specify explicit package versions and are tested against the
environment used for the results in
[Phys. Rev. C 113 (2026) 1, 014904](https://doi.org/10.48550/arXiv.2507.23598).

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
  cmake .. -DCMAKE_PREFIX_PATH="${CONDA_PREFIX};$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
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
  [PyJetscape README](../PyJetscape/README.md#prerequisites) for details.
