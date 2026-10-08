# conda_install — the `js_fno` conda environment

Scripts that create and check the `js_fno` conda env: the Python/ML stack of the contribs
(PyTorch, ROOT, neuraloperator, uproot, …) and, in the `*_build_*` variants, the C++ tools
and libraries to build X-SCAPE and js-contrib. The step-by-step guide (activate, verify,
build X-SCAPE, which workflow needs what) is [`contribs/README.md`](../../contribs/README.md);
this page is the reference for the scripts themselves.

## Why a conda install

**Mainly for macOS on Apple Silicon.** On Linux with an NVIDIA GPU, the Docker images are
the reference environment, with CUDA passed through to the container. Both sets are
published for amd64 and arm64 (aarch64):

| Images | aarch64 status |
|---|---|
| production `xscape-prod` ([`utils/BuildContainerProd.md`](../BuildContainerProd.md)): X-SCAPE and PyJetscape built in | working: `cu130` arm64 tested on the GB10 |
| dev `xscape-fno4d-dev` ([`utils/BuildContainerDev.md`](../BuildContainerDev.md)): the build environment, without sources | built and published, not tested yet on aarch64 |

On a Mac neither helps: Docker runs the containers in a Linux VM, which has no access to the
Apple GPU. Metal, and with it PyTorch's MPS backend and MLX, isn't available inside a
container, so PyTorch there runs on the CPU only. The images are also built for Linux and don't have Metal backends such as
music4gpu's.

A native conda env gives the Mac the GPU-accelerated stack: PyTorch with MPS, MLX, and
ARM-native builds of ROOT and the JETSCAPE C++ dependencies to build X-SCAPE against.

On Linux the env is an alternative to the containers: to work outside Docker, or on aarch64
machines where conda-forge has native builds of every C++ dependency.

## Scripts

| Script | What it does |
|---|---|
| [`test_js_fno_build_env.sh`](test_js_fno_build_env.sh) | Dry run: checks that every package is reachable, installs nothing. Run it first. |
| [`install_js_fno_build_minimal.sh`](install_js_fno_build_minimal.sh) | Full install: Python/ML stack + C++ build tools and JETSCAPE dependencies. Use to build X-SCAPE + js-contrib in the env. |
| [`install_js_fno_minimal.sh`](install_js_fno_minimal.sh) | Python/ML stack only, for an X-SCAPE built elsewhere. |
| [`pinned/install_js_fno_build_pinned.sh`](pinned/install_js_fno_build_pinned.sh) | The full install with exact versions. |
| [`pinned/install_js_fno_pinned.sh`](pinned/install_js_fno_pinned.sh) | The Python/ML stack with exact versions. |

The minimal scripts list top-level packages only, with floors where it matters, and take
the newest releases. The pinned ones fix every main package (Python 3.11.9, torch 2.4.1,
ROOT 6.36.14, …) for reproducibility.

## Quick start

From the js-contrib root:

```bash
bash utils/conda_install/test_js_fno_build_env.sh          # dry run
bash utils/conda_install/install_js_fno_build_minimal.sh   # Linux, CUDA auto-detected
bash utils/conda_install/install_js_fno_build_minimal.sh none   # macOS Apple Silicon (CPU/MPS)
conda activate js_fno
```

## Arguments and settings

All four install scripts take the same arguments:

```
bash <script> [CUDA_VERSION] [CONDA_PREFIX]
```

| Argument | Meaning |
|---|---|
| `CUDA_VERSION` | `12.4`, `13.0`, …: the PyTorch wheel for that CUDA (`cu124`, …) from download.pytorch.org. `none`: CPU/MPS wheels from PyPI; use it on macOS. Omitted: detected from `nvcc`, then `nvidia-smi`, else `none`. |
| `CONDA_PREFIX` | Where to install Miniconda if `conda` isn't found (default `~/miniconda3`); the script asks first. |

| Variable | Effect |
|---|---|
| `JS_FNO_ENV_NAME=<name>` | Name of the env (default `js_fno`), e.g. to try a new install next to a working one. |
| `JS_FNO_FORCE=1` | Replace an existing env of that name; without it the script stops. |
| `JS_FNO_NO_EDITABLE=1` | Skip the final `pip install -e` of PyJetscape and FastHydro. |
| `JS_FNO_NO_MLX=1` | Skip MLX on Apple Silicon. |

`test_js_fno_build_env.sh` takes only `[CUDA_VERSION]`.

## What gets installed

In this order (conda packages from conda-forge; `mamba` is used when available):

| Step | Packages | Scripts |
|---|---|---|
| Env | Python 3.11 | all |
| C++ build | cmake, make, compilers, boost-cpp, zlib, hdf5, pythia8, hepmc3, fastjet, gsl, openmpi | `*_build_*` |
| ROOT | `root>=6.34,<6.38` (pinned: 6.36.14; the PRC 113 014904 environment had 6.32.2, which conda-forge does not build for macOS arm64). 6.38 writes truncated-float TTree leaves uproot can't read yet. | all |
| PyTorch (pip) | torch, torchvision: CUDA wheels, or CPU/MPS with `none` | all |
| Analysis (conda) | numpy, scipy, matplotlib, h5py, hdf5plugin, pandas, seaborn, tqdm, pyyaml, jupyterlab, notebook, ipykernel, metakernel (for ROOT's C++ kernel), ipywidgets, networkx, vector, pybind11, pytest; fastjet in the minimal Python-only script | all |
| pip | neuraloperator ≥ 2.0, uproot ≥ 5, awkward ≥ 2, pyvista, imageio, imageio-ffmpeg | all |
| MLX (pip) | `mlx>=0.30` (pinned: 0.32.3), Apple Silicon only | all |
| js-contrib | `pip install -e` of PyJetscape and FastHydro, when run from a checkout | all |

pip always runs with the new env's own Python, so an active venv elsewhere on `PATH` is
left alone.

## Platform notes

- **macOS, Apple Silicon.** Pass `none`. PyTorch comes from PyPI with the MPS backend; the
  minimal scripts take the newest release, which today needs macOS ≥ 14. MLX is installed as
  well; its wheels also need macOS ≥ 14, and a failed MLX install only warns.
- **Linux aarch64 (e.g. GB10, Graviton).** conda-forge has native builds of all the C++
  dependencies. CUDA 12.x wheels run on a CUDA 13 driver. To run productions, the tested
  production image is the alternative; to build X-SCAPE yourself, use this env (the dev
  image is untested on aarch64, see [above](#why-a-conda-install)).
- **Linux x86_64.** Works the same; the production and dev containers
  ([`utils/BuildContainerProd.md`](../BuildContainerProd.md),
  [`utils/BuildContainerDev.md`](../BuildContainerDev.md)) are the alternative.

## FNO4d

The env doesn't include FNO4d. To train or evaluate with FNO4d in it, run FNO4d's own
installer in the active env (`conda activate js_fno; cd ~/FNO4d; bash install.sh --conda`).
It replaces PyPI `neuraloperator` with FNO4d's 4D fork (needed for `H1Loss(d=4)`) and adds
`loc_libs`, `neuralop_mlx` on Mac, scikit-image and zarr. Details:
[FNO4d in the js_fno environment](../../contribs/README.md#fno4d-in-the-js_fno-environment).

## Related

- [`utils/analysis_env/`](../analysis_env/README.md): a plain venv to analyse production
  files without X-SCAPE, conda or ROOT.
- [Troubleshooting](../../contribs/README.md#troubleshooting) in `contribs/README.md`.
