# Source build (conda)

The source build compiles X-SCAPE (branch `contrib`) with its external packages and js-contrib
inside the tree, in the `js_fno` conda environment. You need it to change C++ (X-SCAPE modules,
MUSIC4GPU, iSS, 3dMCGlauber), the PyJetscape bindings or the `jetscape` package, to use the C++
FNO hydro (FnoHydro, through LibTorch) or the live PyTorch FNO hydro, and for ROOT output. If
you only want to run productions, write scripts and Python modules, or analyse, the
[container](container.md) or the [Mac wheel](mac.md) gets you there without compiling.

**Why conda.** One conda prefix holds every C++ dependency (Boost, Pythia8, HDF5, GSL, HepMC3,
FastJet, ROOT) next to Python, PyTorch, pybind11 and the analysis packages, so CMake finds them
all without system packages. That matters most on Apple Silicon (native ARM builds, PyTorch
with MPS) and on Linux `aarch64` (AWS Graviton, ARM servers, GB10), where several JETSCAPE
dependencies are missing from the usual package managers; conda-forge has native builds of all
of them.

The order of the steps: get the sources, make the environment, fetch the external packages,
configure and build, make `jetscape` importable, run the tests.

## Requirements

=== "Linux"

    - x86_64 or aarch64.
    - For MUSIC on the GPU: an NVIDIA GPU with its driver and the CUDA toolkit (`nvcc`).
      Without them, MUSIC builds and runs on the CPU.
    - git, and conda (the install script offers to install Miniconda if `conda` isn't found).
    - On a machine with an NVIDIA GPU, the [dev container](#the-dev-container-linux-nvidia)
      is an alternative to steps 2 to 4's environment.

=== "macOS"

    - An Apple Silicon Mac. PyTorch from PyPI (the latest release) and MLX need macOS 14 or
      later.
    - Xcode with its Metal toolchain, for MUSIC on the GPU (`-DUSE_METAL=ON`).
    - git, and conda (the install script offers to install Miniconda if `conda` isn't found).

!!! note "The Mac wheel and a source build on one Mac"
    They can coexist, but use them in separate Terminal windows: the source build's conda
    environment and variables (`PYTHONPATH`, `XSCAPE`, ...) must not be active where the wheel's
    environment runs.

## 1. Get X-SCAPE and js-contrib

js-contrib is built inside X-SCAPE's tree, from `external_packages/js-contrib`. X-SCAPE must be
on its `contrib` branch: `get_js_contrib.sh`, the `USE_JS_*` options and the hooks the bindings
use exist only there, not on `main` or in the original JETSCAPE.

```bash
git clone -b contrib https://github.com/JETSCAPE/X-SCAPE.git
cd X-SCAPE/external_packages
./get_js_contrib.sh            # clones https://github.com/jhputschke/js-contrib into js-contrib/
```

## 2. Create the `js_fno` environment

Run the scripts from the js-contrib checkout inside the tree: they finish with an editable
`pip install -e` of PyJetscape and FastHydro from that checkout, so the `jetscape` package you
import is the one you build.

```bash
cd X-SCAPE/external_packages/js-contrib
bash utils/conda_install/test_js_fno_build_env.sh          # dry run: is every package reachable?
```

=== "Linux"

    ```bash
    bash utils/conda_install/install_js_fno_build_minimal.sh        # CUDA detected from nvcc / nvidia-smi
    # bash utils/conda_install/install_js_fno_build_minimal.sh 12.4 # or force a CUDA version
    conda activate js_fno
    ```

=== "macOS"

    ```bash
    bash utils/conda_install/install_js_fno_build_minimal.sh none   # CPU/MPS PyTorch from PyPI
    conda activate js_fno
    ```

The dry run prints `[PASS]`/`[FAIL]`/`[WARN]` per package and ends with
`All checks passed — safe to run install_js_fno_build_minimal.sh.` It takes the same CUDA
argument (`none` on a Mac). The install script then creates the environment (Python 3.11) with:

| layer | packages |
|---|---|
| C++ build | cmake, make, compilers, boost-cpp, zlib, hdf5, pythia8, hepmc3, fastjet, gsl, openmpi |
| ROOT | `root>=6.34,<6.38` from conda-forge |
| PyTorch | torch, torchvision: CUDA wheels from download.pytorch.org, or CPU/MPS wheels with `none` |
| analysis and tests | numpy, scipy, matplotlib, h5py, hdf5plugin, pandas, JupyterLab, ipykernel, vector, pybind11, pytest, ...; with pip: neuraloperator, uproot, awkward, pyvista, imageio |
| MLX | Apple Silicon only |
| js-contrib | `pip install -e` of PyJetscape and FastHydro |

Arguments and settings, the same for all install scripts:

| | |
|---|---|
| 1st argument | the CUDA version for the PyTorch wheels (`12.4`, `13.0`, ...), or `none`; omitted: detected |
| 2nd argument | where to install Miniconda if `conda` isn't found (default `~/miniconda3`) |
| `JS_FNO_ENV_NAME=<name>` | another environment name, e.g. a new install next to a working `js_fno` |
| `JS_FNO_FORCE=1` | replace an existing environment of that name (otherwise the script stops) |
| `JS_FNO_NO_EDITABLE=1` | skip the final editable install of PyJetscape and FastHydro |
| `JS_FNO_NO_MLX=1` | skip MLX on Apple Silicon |

For exact reproducibility there are pinned variants in `utils/conda_install/pinned/`
(`install_js_fno_build_pinned.sh`, Python 3.11.9, torch 2.4.1, ROOT 6.36.14, ...). Details:
[`utils/conda_install/README.md`](https://github.com/jhputschke/js-contrib/blob/main/utils/conda_install/README.md).

## 3. Fetch the external packages

Each external package has a download script in `X-SCAPE/external_packages/`, pinned to a
tested commit.

```bash
cd X-SCAPE/external_packages
./get_music4gpu.sh      # MUSIC with GPU backends (CUDA, Metal), + its hotQCD EoS tables
./get_iSS.sh            # iSS particlization, + its EoS and delta-f tables
./get_3dglauber.sh      # 3dMCGlauber initial state, + LHAPDF
./get_lbtTab.sh         # LBT's scattering tables (1.2 GB), needed at run time
```

| script | for | CMake flag |
|---|---|---|
| `get_music4gpu.sh` | MUSIC on the GPU (MUSIC4GPU, branch `XSCAPE`) | `-DUSE_MUSIC=ON` with `-DUSE_CUDA=ON` or `-DUSE_METAL=ON` |
| `get_music.sh` | MUSIC on the CPU only (the upstream package) | `-DUSE_MUSIC=ON` alone |
| `get_iSS.sh` | iSS (hadronization of the stored surfaces) | `-DUSE_ISS=ON` |
| `get_3dglauber.sh` | 3dMCGlauber (also needed for iColored and iMATTER) | `-DUSE_3DGlauber=ON` |
| `get_lbtTab.sh` | LBT tables | none; run it **before** configuring, so CMake links them into the build directory |
| `get_js_contrib.sh` | js-contrib (step 1) | `-DUSE_JS_CONTRIB=ON` |

!!! tip "After `git pull` of X-SCAPE"
    The `get_*.sh` scripts carry the pinned commit of each package. When a pull changes a pin
    (`git diff` on `external_packages/get_*.sh`), bring that package to the new commit (or
    fetch it again) and rebuild; a pull alone doesn't update the packages already in
    `external_packages/`.

## 4. Configure and build

Build into a directory called **`build_gpu`** at the top of the X-SCAPE tree: the production
scripts (`--build`) and `jetscape.paths` look there by default. With another name, pass
`--build` to the scripts or set `JETSCAPE_BUILD`.

=== "Linux, NVIDIA GPU"

    ```bash
    conda activate js_fno
    cd X-SCAPE
    cmake -S . -B build_gpu -DCMAKE_BUILD_TYPE=Release \
      -DUSE_MUSIC=ON -DUSE_CUDA=ON -DUSE_ISS=ON -DUSE_3DGlauber=ON \
      -DUSE_JS_CONTRIB=ON -DUSE_JS_PYJETSCAPE=ON
    cmake --build build_gpu -j"$(nproc)"
    ```

    MUSIC4GPU compiles for the GPU it finds at configure time (`CMAKE_CUDA_ARCHITECTURES`
    defaults to `native`). On a node without a GPU, give the architectures explicitly, e.g.
    `-DCMAKE_CUDA_ARCHITECTURES=80` for an A100. If nvcc fails to find the CUDA headers with
    conda's compilers, use the system GCC as nvcc's host compiler,
    `export CUDAHOSTCXX=/usr/bin/g++`, as the container builds do.

=== "macOS, Metal"

    ```bash
    conda activate js_fno
    cd X-SCAPE
    cmake -S . -B build_gpu -DCMAKE_BUILD_TYPE=Release \
      -DUSE_MUSIC=ON -DUSE_METAL=ON -DUSE_ISS=ON -DUSE_3DGlauber=ON \
      -DUSE_JS_CONTRIB=ON -DUSE_JS_PYJETSCAPE=ON
    cmake --build build_gpu -j"$(sysctl -n hw.ncpu)"
    ```

=== "CPU only"

    ```bash
    conda activate js_fno
    cd X-SCAPE
    cmake -S . -B build_gpu -DCMAKE_BUILD_TYPE=Release \
      -DUSE_MUSIC=ON -DUSE_ISS=ON -DUSE_3DGlauber=ON \
      -DUSE_JS_CONTRIB=ON -DUSE_JS_PYJETSCAPE=ON
    cmake --build build_gpu -j"$(nproc)"
    ```

    MUSIC without a GPU flag is the CPU MUSIC from `get_music.sh`. Its results differ
    systematically from the GPU path, so don't mix CPU and GPU events in one campaign.

The flags:

| flag | what it does |
|---|---|
| `-DUSE_MUSIC=ON` | builds MUSIC; **required** for the bindings, which include MUSIC's wrapper |
| `-DUSE_CUDA=ON` / `-DUSE_METAL=ON` | MUSIC from MUSIC4GPU on an NVIDIA GPU / on Apple Silicon (mutually exclusive; Metal only on macOS) |
| `-DUSE_ISS=ON` | iSS; needed for hadronization (`hadronize.py`, `run_hadronize.py`) |
| `-DUSE_3DGlauber=ON` | 3dMCGlauber; **required** for the bindings: without it iColoredHadronization and iMATTER are not compiled and `pyjetscape_core` fails to load (`undefined symbol ... iColoredHadronization`) |
| `-DUSE_JS_CONTRIB=ON` | builds js-contrib from `external_packages/js-contrib` |
| `-DUSE_JS_PYJETSCAPE=ON` | the Python bindings `pyjetscape_core` |
| `-DUSE_JS_FNO_HYDRO=ON` | the C++ FnoHydro module (LibTorch, about 2 GB, and ROOT); see below |
| `-DJS_PIP_INSTALL_PYJETSCAPE=ON` | runs `pip install -e` of PyJetscape on every build |
| ROOT | used when the environment has it (`js_fno` does): adds the `FastRootBulkWriter` binding (`jetscape.HAS_ROOT`). Without ROOT everything else builds and `HAS_ROOT` is `False` |

CMake finds Boost, Pythia8, HDF5, ROOT and the rest in the active conda environment by
itself. After a change to the bindings only, rebuild just them:
`cmake --build build_gpu --target pyjetscape_core`.

**FnoHydro (C++ FNO hydro).** Add `-DUSE_JS_FNO_HYDRO=ON` and PyTorch's CMake files to the
prefix path:

```bash
cmake -S . -B build_gpu ... -DUSE_JS_FNO_HYDRO=ON \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX;$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
```

The live PyTorch FNO hydro (`PyFNOHydro`) needs no extra flag, only PyTorch in the Python
environment, which `js_fno` has: [Live PyTorch FNO hydro](../python/fno.md).

## 5. Make `jetscape` importable

The build writes the compiled extension `pyjetscape_core*.so` into
`external_packages/js-contrib/contribs/PyJetscape/python/jetscape/`. The install script of
step 2 already installed that package in editable mode, so with `conda activate js_fno` it
imports from anywhere:

```bash
python -c "import jetscape as js; print(js.HAS_CORE, js.pyjetscape_core.MUSIC_BACKEND)"
# -> True music4gpu-cuda      (Linux)    or    True music4gpu-metal   (macOS)
```

`HAS_CORE` is `True` once the compiled extension loads; `MUSIC_BACKEND` names the MUSIC it was
built against (`music` for the CPU build). In another environment, make the package importable
in one of two ways:

```bash
# editable install (recommended, e.g. for Jupyter)
pip install -e X-SCAPE/external_packages/js-contrib/contribs/PyJetscape
# or, without installing
export PYTHONPATH="/path/to/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape/python:$PYTHONPATH"
```

The compiled core must match that environment's Python version (the one it was built with).

- **`PYTHIA8DATA`:** the compiled core needs it, and `conda activate js_fno` sets it. Without
  it `import jetscape` aborts with
  `Assertion 'InternalHelperPythia.particleData.isParticle(id)' failed`.
- **Import order with PyTorch and ROOT:** both ship their own `libomp`. Import `torch`
  before `jetscape` and ROOT (on macOS, `import jetscape` does this for you); if you see a
  segfault at import, check the order first.

## 6. Run the tests

```bash
cd X-SCAPE
pytest external_packages/js-contrib/contribs/PyJetscape/tests -q
```

Most of the Python tooling is tested without a build; the tests that need the compiled core
are skipped when it doesn't load. FastHydro and Visualization have their own suites in
`contribs/FastHydro/tests` and `contribs/Visualization/tests`.

## The dev container (Linux, NVIDIA)

On Linux with an NVIDIA GPU, the dev image `jhputschke/xscape-fno4d-dev` holds the same build
environment as `js_fno`, ready-made: CUDA, a conda environment (`xscape`, already on `PATH`),
PyTorch and the C++ dependencies, but **no sources**. Mount your X-SCAPE checkout (with its
external packages fetched, steps 1 and 3) and build inside it with the commands of step 4.

| tag | GPUs |
|---|---|
| `cu126` | GH200, H100, A100, RTX 20/30/40xx |
| `cu132` | the above plus Blackwell (GB200, GB10) |

```bash
docker pull jhputschke/xscape-fno4d-dev:cu126
docker run --rm -it --gpus all -v "$PWD/X-SCAPE:/workspace/X-SCAPE" \
  jhputschke/xscape-fno4d-dev:cu126
# inside: cd /workspace/X-SCAPE, then step 4
```

The image starts in `/workspace` and sets `CUDAHOSTCXX` for nvcc. It is published for amd64
and arm64, but has not been tested on aarch64 yet; there, use the conda environment. Image
details: [`utils/BuildContainerDev.md`](https://github.com/jhputschke/js-contrib/blob/main/utils/BuildContainerDev.md).

## Linux aarch64

On ARM servers (AWS Graviton, GH200, GB10) the conda environment is the tested way to build:
conda-forge has native `aarch64` builds of every C++ dependency, and the CUDA 12.x PyTorch
wheels run on a CUDA 13 driver. To run productions without building, the
[production container](container.md) has arm64 images (`cu126`, `cu130`).

## Other ways to build js-contrib

js-contrib can also be built standalone against an X-SCAPE build tree or an installed X-SCAPE
prefix (Paths B and C of the
[js-contrib README](https://github.com/jhputschke/js-contrib/blob/main/README.md#installation));
for the Python bindings that route does not work at present, so use the in-tree build above.
For X-SCAPE checkouts older than the `contrib` integration there is a manual patch recipe
(Path A′ of the same README), kept for legacy checkouts.

## Troubleshooting

| symptom | cause and fix |
|---|---|
| `conda: command not found` after the install | the installer doesn't edit your shell config: run the printed init command, e.g. `~/miniconda3/bin/conda init zsh`, and open a new terminal |
| `Environment 'js_fno' exists already` | remove it (`conda env remove -n js_fno`), replace it with `JS_FNO_FORCE=1`, or install next to it with `JS_FNO_ENV_NAME=js_fno_new` |
| CMake doesn't find ROOT on macOS | activate the environment before `cmake`; if it still fails, add `-DCMAKE_PREFIX_PATH="${CONDA_PREFIX}"` |
| `pyjetscape_core` fails to load: `undefined symbol ... iColoredHadronization` | configure with `-DUSE_3DGlauber=ON` |
| `Assertion 'InternalHelperPythia.particleData.isParticle(id)' failed` on `import jetscape` | `PYTHIA8DATA` not set: `conda activate js_fno` |
| `MPS: False` for PyTorch on Apple Silicon | the environment was made without `none`, or Python runs under Rosetta: `python -c "import platform; print(platform.machine())"` must print `arm64` |
| `JobDir: not found: .../music_input` | the build directory is not `build_gpu`: set `JETSCAPE_BUILD` or pass `--build` |

Step-by-step reference for the environment and the workflows it covers:
[`contribs/README.md`](https://github.com/jhputschke/js-contrib/blob/main/contribs/README.md);
the bindings: [`contribs/PyJetscape/README.md`](https://github.com/jhputschke/js-contrib/blob/main/contribs/PyJetscape/README.md).

## Next: run your first event

With `build_gpu` built and `jetscape` importing, [run your first event](../run/first-event.md).
