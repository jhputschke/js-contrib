# js-contrib

Community contributions for the [JETSCAPE](https://github.com/JETSCAPE/JETSCAPE) /
[X-SCAPE](https://github.com/JETSCAPE/X-SCAPE) and C-SCAPE (in development) heavy-ion simulation framework,
analogous to [fastjet-contrib](https://fastjet.hepforge.org/contrib/) for FastJet.

## Available contribs

| Contrib | Description | Extra deps |
|---------|-------------|------------|
| [FastHydro](contribs/FastHydro/) | MC-Glauber initial state + a fast 3+1D Milne FV hydro solver, with the Matter+LBT → CausalLiquefier jet-deposition workflow; paired background/jet evolutions on an identical IC | PyTorch, scipy, h5py, pyyaml; reuses PyJetscape (no build of its own) |
| [FnoHydro](contribs/FnoHydro/) | Neural-network (FNO) hydrodynamics via LibTorch | ROOT, libtorch (~2 GB) |
| [PyJetscape](contribs/PyJetscape/) | pybind11 Python bindings + PyFNOHydro trampoline | pybind11 (pip/conda auto-detected); h5py, hdf5plugin, pyyaml + notebook stack via `pip install -e`; PyTorch optional (`[fno]`, only for PyFNOHydro) |
| [Visualization](contribs/Visualization/) | 3D PyVista visualization of the hydro medium evolution, resampled Milne→Cartesian `(t,x,y,z)`, with a jet parton-shower overlay | pyvista, scipy, vtk, imageio (`imageio-ffmpeg` for `.mp4` output); reuses PyJetscape |

The **FastHydro** and **Visualization** contribs are pure Python (no CMake build of
their own, so neither needs a `USE_JS_*` flag), but both read the framework through the
PyJetscape bindings and so need a current `pyjetscape_core`. Visualization additionally
needs the `EvolutionHistory.to_numpy_full()` binding for genuine 3+1D data, plus a Python
env with PyVista. See
[contribs/README.md](contribs/README.md#visualization-contrib--pyvista-dependencies)
and [contribs/Visualization/README.md](contribs/Visualization/README.md).

### Utilities

[`utils/README.md`](utils/README.md) lists everything in `utils/`: environments, container
images, the SLURM script and the data tools below.

[`utils/h5_inspect.py`](utils/h5_inspect.py) lists every group and dataset of an HDF5 file
with its shape, dtype, storage and attributes. For the js-contrib hydro files
(`xscape/hydro_evolution`, `fast_data/hydro_evolution`, pairs, legacy FNO4d, IC files) it
also summarizes the grid, the evolution-array axes, freeze-out, the ragged `shower/` and
`source/` tables and `diag/`, and flags inconsistencies. It needs only h5py and numpy.

```bash
python utils/h5_inspect.py out/AuAu_0_10_jet_seed0001.h5           # summary + tree
python utils/h5_inspect.py FILE.h5 --stats [--event 0]              # per-feature min/max/mean, NaN/inf
python utils/h5_inspect.py FILE.h5 --attr prod_user_xml             # one attribute in full
```

### Conda environments

[`utils/conda_install/`](utils/conda_install/) creates and checks the `js_fno` conda env: the
Python/ML stack plus, optionally, the C++ build tools for X-SCAPE and js-contrib, minimal or
pinned. See [contribs/README.md](contribs/README.md) for the steps.

### Analysis environment (no X-SCAPE build)

[`utils/analysis_env/`](utils/analysis_env/README.md) makes a Python venv for people who only
analyse the files of a `prod_AuAu_0_10_jet` production. It needs this checkout and a
Python ≥ 3.10, but no X-SCAPE, ROOT, conda or `pyjetscape_core`. It covers the pair,
particlize and hadron HDF5 files, `run_h5toROOT.py` and its ROOT files (uproot), the
analysis notebooks (fastjet), and the Visualization scripts on stored files (pyvista). It
also downloads MUSIC's hotQCD EoS table. FastHydro is optional (`--with-fasthydro`). ROOT is
optional too: made from the Python of a conda env with ROOT and `--system-site-packages`,
the venv uses PyROOT. A single conda env with ROOT, `environment.yml`, also exists. See
[With ROOT](utils/analysis_env/README.md#with-root). For productions in Google Cloud Storage
or a Pelican federation (OSDF), `--with-gcs` and `--with-pelican` add gcsfs /
google-cloud-storage and pelicanfs; see
[Remote files](utils/analysis_env/README.md#remote-files-google-cloud-storage-and-pelicanosdf).

```bash
./utils/analysis_env/setup_analysis_env.sh           # venv in ~/.venvs/js_analysis
source ~/.venvs/js_analysis/bin/activate
python utils/analysis_env/check_env.py /path/to/out  # opens every file of a production
```

### Moving productions: Google Cloud Storage and Pelican/OSDF

[`utils/remote_transfer/`](utils/remote_transfer/README.md) has two command-line tools with
the same commands: `js_gcs.py` for a GCS bucket (default `gs://test_fno`, a
service-account key) and `js_osdf.py` for a Pelican namespace on the OSDF (default
`osdf:///fno4hic`, pelicanfs, a bearer token for writing). They upload and download whole
production directories, single files or patterns, and pick the files by kind:
- `pair`: the hydro pair files;
- `h5`: the particlize and hadron files;
- `root`: the ROOT files of `run_h5toROOT.py`;
- `all`: everything.

Files already there are skipped, unfinished HDF5 files are left out, and every transfer is
checked by size and CRC32C. Each script builds its own Python environment on first use, so
a plain `python3` is all it needs.

```bash
utils/remote_transfer/js_gcs.py upload /data/AuAu_c1 --what root --dry-run   # -> gs://test_fno/AuAu_c1/
utils/remote_transfer/js_osdf.py upload /data/out --what h5 --as AuAu_c1      # -> osdf:///fno4hic/AuAu_c1/
utils/remote_transfer/js_gcs.py download AuAu_c1 --what root --to /scratch
```

[`utils/upload_follow.py`](utils/upload_follow.py) uploads a production to the OSDF **while it
runs** and can delete the verified uploads locally, so a campaign is no longer limited by the
local disk; [`utils/launch_2gpu.sh`](utils/launch_2gpu.sh) runs a campaign on both GPUs of one
machine with it ([`utils/README_launch.md`](utils/README_launch.md)).

### Design notes and benchmarks

[`docs/`](docs/README.md) holds the plans behind the HDF5 writers and the productions, the
GB10 and Apple M3 Max benchmarks, the HDF5 compression study and the container plans.

## Source provenance

The source files in this repository were copied from
[JETSCAPE-FNO](https://github.com/JETSCAPE/JETSCAPE-FNO) at commit
`jhputschke/JETSCAPE-FNO @ PythonTest` (April 2026).

## Compatibility

| js-contrib |     X-SCAPE    |   MUSIC4GPU   |
|------------|----------------|---------------|
| v0.x       | contrib (head) | XSCAPE (head) |

## Installation

All Python packages used by the contribs (examples, notebooks, Visualization, tests) are
listed in [`requirements.txt`](requirements.txt): `pip install -r requirements.txt` in the
environment `pyjetscape_core` is built with. Leaner per-contrib installs:
`pip install -e contribs/PyJetscape` (no torch) and `pip install -e "contribs/FastHydro[solver]"`.
To only analyse production files, without X-SCAPE, use
[`utils/analysis_env/setup_analysis_env.sh`](utils/analysis_env/README.md).

### Path A — via X-SCAPE CMake (recommended)

```bash
# 1. Fetch js-contrib into X-SCAPE
cd X-SCAPE
./external_packages/get_js_contrib.sh

# 2. Rebuild with desired contribs in addition to other physics modules
cd build
cmake .. -DUSE_JS_CONTRIB=ON -DUSE_JS_FNO_HYDRO=ON ...  # or -DUSE_JS_PYJETSCAPE=ON
make -j$(nproc)
```

```bash
# For example to use FNO and new Python Interface, but w/o jet energy loss modudels,
# but they can be attached to use the FNO hydro history for jet quenching
cd build
cmake .. -DUSE_MUSIC=ON -DUSE_ISS=ON -DUSE_PYTHON=ON -USE_ROOT=ON \
          -DUSE_JS_CONTRIB=ON -DUSE_JS_FNO_HYDRO=ON -DUSE_JS_PYJETSCAPE=ON \
          -DCMAKE_PREFIX_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")
make -j$(nproc)
```

After building `pyjetscape_core`, make the `jetscape` Python package importable
in one of two ways:

- **PYTHONPATH** (no install step, works immediately):
  ```bash
  export PYTHONPATH="$PWD/../external_packages/js-contrib/contribs/PyJetscape/python:$PYTHONPATH"
  ```
- **Editable pip install** (recommended for Jupyter / persistent use):
  ```bash
  pip install -e external_packages/js-contrib/contribs/PyJetscape
  ```
  Or let CMake do this automatically on every build by adding
  `-DJS_PIP_INSTALL_PYJETSCAPE=ON` to the `cmake` command above.

> **pybind11 discovery** — CMake first searches `CMAKE_PREFIX_PATH` / `pybind11_DIR`
> (system or conda install), then falls back to asking the active Python interpreter
> (`python -c "import pybind11; print(pybind11.get_cmake_dir())"`), so a plain
> `pip install pybind11` or `conda install -c conda-forge pybind11` is sufficient.
> `Python3` with the `Development.Module` component (CMake ≥ 3.18) is found
> automatically before pybind11 to avoid the `python3_add_library` error.
> If auto-detection still fails, pass the path explicitly:
> ```bash
> -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")
> ```

### Path A′ — manual integration into older X-SCAPE checkouts

If your X-SCAPE checkout predates the js-contrib integration, apply the three
steps below by hand (they mirror exactly what the up-to-date `CMakeLists.txt`,
`cmake/JetScapeConfig.cmake.in`, and `external_packages/get_js_contrib.sh`
already contain).

#### Step 1 — add `external_packages/get_js_contrib.sh`

Create the file with the following content and make it executable:

```bash
#!/usr/bin/env bash
###############################################################################
# Copyright (c) The JETSCAPE Collaboration, 2018
#
# Distributed under the GNU General Public License 3.0 (GPLv3 or later).
# See COPYING for details.
###############################################################################
# Clone js-contrib into external_packages/js-contrib
# Use with: cmake -DUSE_JS_CONTRIB=ON [-DUSE_JS_FNO_HYDRO=ON] [-DUSE_JS_PYJETSCAPE=ON]

folderName="js-contrib"

if [ -d "$folderName" ]; then
  echo "$folderName already exists — skipping clone."
  exit 0
fi

git clone https://github.com/jhputschke/js-contrib.git "$folderName"
```

```bash
chmod +x external_packages/get_js_contrib.sh
```

#### Step 2 — patch the top-level `CMakeLists.txt`

**Block A — option declarations** (add after the `USE_SMASH` block,
immediately before the `# Compile with OpenMP support` comment):

```cmake
# js-contrib extensions. Turn on with 'cmake -DUSE_JS_CONTRIB=ON'.
# Individual contribs are gated by their own sub-options (all OFF by default).
option(USE_JS_CONTRIB "Enable js-contrib extensions" OFF)
if(USE_JS_CONTRIB)
  option(USE_JS_FNO_HYDRO
    "Build FnoHydro contrib (requires libtorch ~2 GB and ROOT)" OFF)
  option(USE_JS_PYJETSCAPE
    "Build PyJetscape pybind11 Python bindings (compile from source)" OFF)
  if(USE_JS_FNO_HYDRO OR USE_JS_PYJETSCAPE)
    message("Enabling js-contrib extensions ...")
  endif()
endif(USE_JS_CONTRIB)
```

**Block B — subdirectory hook** (add after the `if(OPENCL_FOUND AND USE_CLVISC)`
block, before the `if(OPENMP_FOUND)` definition block):

```cmake
if(USE_JS_CONTRIB)
  if(NOT EXISTS "${CMAKE_SOURCE_DIR}/external_packages/js-contrib")
    message(
      FATAL_ERROR
        "Error: js-contrib has not been downloaded in external_packages by ./external_packages/get_js_contrib.sh"
    )
  endif()
  # Pass the per-contrib flags through to the js-contrib CMakeLists
  set(BUILD_FNO_HYDRO ${USE_JS_FNO_HYDRO})
  set(BUILD_PYJETSCAPE ${USE_JS_PYJETSCAPE})
  add_subdirectory(${CMAKE_SOURCE_DIR}/external_packages/js-contrib)
endif(USE_JS_CONTRIB)
```

#### Step 3 — patch `src/CMakeLists.txt`, `cmake/JetScapeConfig.cmake.in`, and the top-level `CMakeLists.txt` (build-tree export)

js-contrib discovers JetScape via a build-tree `export()`. The key constraint is
that `export()` must be called **after** every optional `add_subdirectory()` that
defines an in-tree target that `JetScape` links to (e.g. `music`, `iSS`). Because
`add_subdirectory(./src)` is processed before those optional subdirectories in the
top-level `CMakeLists.txt`, the `export()` call must be placed at the **end of
the top-level `CMakeLists.txt`**, not inside `src/CMakeLists.txt`.

**3a — update `cmake/JetScapeConfig.cmake.in`**

Older checkouts contain only `include(...)`. Replace the entire file with:

```cmake
include("${CMAKE_CURRENT_LIST_DIR}/JetScapeTargets.cmake")

# Provide JetScape_INCLUDE_DIRS so that standalone consumers (e.g. js-contrib
# built with JETSCAPE_DIR pointing at this build directory) can locate the
# X-SCAPE / JETSCAPE framework headers without relying on global
# include_directories() from the parent CMake scope.
set(JetScape_INCLUDE_DIRS
  "@CMAKE_SOURCE_DIR@/src/framework"
  "@CMAKE_SOURCE_DIR@/src/initialstate"
  "@CMAKE_SOURCE_DIR@/src/preequilibrium"
  "@CMAKE_SOURCE_DIR@/src/hydro"
  "@CMAKE_SOURCE_DIR@/src/liquefier"
  "@CMAKE_SOURCE_DIR@/src/jet"
  "@CMAKE_SOURCE_DIR@/src/hadronization"
  "@CMAKE_SOURCE_DIR@/src/afterburner"
  "@CMAKE_SOURCE_DIR@/external_packages"
  "@CMAKE_SOURCE_DIR@/external_packages/gtl/include"
  "@CMAKE_SOURCE_DIR@/external_packages/trento/src"
)
set(JetScape_FOUND TRUE)
```

**3b — Remove** any existing `export(…)` / `configure_file(…JetScapeConfig…)` lines
from `src/CMakeLists.txt`, then **append** the following block at the very end
of the top-level `CMakeLists.txt`:

```cmake
# Build-tree export for js-contrib and other out-of-tree consumers.
# Must live here — AFTER all optional add_subdirectory() calls — so every
# in-tree target already exists when export() is invoked.
set(_js_export_targets JetScape JetScapeThird GTL libtrento Cornelius)
if(${HDF5_FOUND})
  list(APPEND _js_export_targets hydroFromFile)
endif()
if(USE_IPGLASMA)
  list(APPEND _js_export_targets ipglasma_lib)
endif()
if(USE_3DGlauber)
  list(APPEND _js_export_targets 3dMCGlb)
endif()
if(USE_MUSIC)
  list(APPEND _js_export_targets music)
endif()
if(USE_ISS)
  list(APPEND _js_export_targets iSS)
endif()
if(OPENCL_FOUND AND USE_CLVISC)
  list(APPEND _js_export_targets clviscwrapper)
endif()
export(TARGETS ${_js_export_targets} FILE "${CMAKE_BINARY_DIR}/JetScapeTargets.cmake")
configure_file(${CMAKE_SOURCE_DIR}/cmake/JetScapeConfig.cmake.in
               "${CMAKE_BINARY_DIR}/JetScapeConfig.cmake" @ONLY)
```

> **Why not in `src/CMakeLists.txt`?** CMake processes `add_subdirectory(./src)`
> before the `if(USE_MUSIC) add_subdirectory(./external_packages/music) endif()`
> block, so `music`, `iSS`, etc. do not exist yet when `src/CMakeLists.txt` runs.
> Placing `export()` after those blocks avoids the
> *"target X which is not built by this project"* error.

After applying both blocks, run the fetch script and build normally:

```bash
cd external_packages && ./get_js_contrib.sh && cd ../build
cmake .. -DUSE_JS_CONTRIB=ON -DUSE_JS_FNO_HYDRO=ON
make -j$(nproc)
```

### Path B — standalone (fastjet-contrib style)

```bash
git clone https://github.com/jhputschke/js-contrib
cd js-contrib && mkdir build && cd build

cmake .. \
  -DJETSCAPE_DIR=/path/to/X-SCAPE/build \
  -DBUILD_FNO_HYDRO=ON \
  -DBUILD_PYJETSCAPE=ON \
  -DCMAKE_PREFIX_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")

make -j$(nproc)
```

pybind11 is auto-detected from a pip or conda install (see note in Path A above).
If needed, add `-Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")` to the cmake line.

### Path C — standalone against an installed X-SCAPE prefix

If X-SCAPE has been installed with `cmake --install` (≥ X-SCAPE `music4gpu_test`
branch), js-contrib can link against the installed libraries without keeping the
build tree around:

```bash
# 1. Build and install X-SCAPE once
cd X-SCAPE && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/opt/xscape \
         -DUSE_MUSIC=ON -DUSE_ISS=ON -DUSE_3DGlauber=ON
make -j$(nproc)
cmake --install .

# 2. Build js-contrib against the installed prefix
git clone https://github.com/jhputschke/js-contrib
cd js-contrib && mkdir build && cd build

cmake .. \
  -DCMAKE_PREFIX_PATH=/opt/xscape \
  -DBUILD_FNO_HYDRO=ON \
  -DBUILD_PYJETSCAPE=ON \
  -DCMAKE_PREFIX_PATH="$(python -c "import torch; print(torch.utils.cmake_prefix_path)"):/opt/xscape"

make -j$(nproc)
```

`find_package(XSCAPE)` is resolved automatically from
`/opt/xscape/lib/cmake/XSCAPE/`.  `XSCAPE::JetScape` is aliased to the
plain `JetScape` target so no changes to individual contrib `CMakeLists.txt`
files are required.

> **No build tree needed** — you can delete the X-SCAPE build directory after
> `cmake --install` completes.

### Integration modes summary

| Mode | Trigger | JetScape resolved via |
|------|---------|-----------------------|
| Path A — in-tree | `-DUSE_JS_CONTRIB=ON` inside X-SCAPE build | target already in CMake scope |
| Path B — build-tree standalone | `JETSCAPE_DIR=/path/to/xscape/build` | `JetScapeConfig.cmake` in build dir |
| Path C — installed prefix | `CMAKE_PREFIX_PATH=/opt/xscape` | `find_package(XSCAPE)` → `XSCAPE::JetScape` alias |

## Containers

Two sets of Docker images exist, for **Linux with NVIDIA GPUs (CUDA)**, each for amd64 and
arm64 (e.g. GH200, GB10). On HPC clusters, Apptainer/Singularity pulls the same images.

| | images | what | docs |
|---|---|---|---|
| **dev** | `jhputschke/xscape-fno4d-dev:cu126`, `:cu132` | the build and runtime environment (CUDA, conda env, PyTorch) **without sources**: mount your X-SCAPE / js-contrib checkouts and build in it | [`utils/BuildContainerDev.md`](utils/BuildContainerDev.md), [`docs/Plans/PlanContainerDev.md`](docs/Plans/PlanContainerDev.md) |
| **production** | `jhputschke/xscape-prod:cu126`, `:cu124` (amd64 only), `:cu130`; `-gcs` variants | X-SCAPE, MUSIC4GPU, iSS, 3dMCGlauber and PyJetscape **built in**, ready to run `prod_AuAu_0_10_jet`; the same image hadronizes on CPU nodes | [`docs/README_2stage.md`](docs/README_2stage.md) (running a production), [`utils/BuildContainerProd.md`](utils/BuildContainerProd.md) (the images) |

**Not for Apple Silicon with Metal.** Docker on macOS runs containers in a Linux VM without
access to the Metal GPU, so neither MUSIC4GPU's Metal back-end nor PyTorch's MPS backend
works in a container. To use Metal on a Mac, build natively with the conda environment
below. (The arm64 images might run CPU-only on a Mac, e.g. for hadronization; that is not
tested.)

## Native environment (Mac Silicon / Linux aarch64)

A native conda environment is the way to **use Metal on a Mac** (MUSIC4GPU's Metal back-end,
PyTorch's MPS backend) and to develop outside a container. On **Linux `aarch64`** (AWS
Graviton, ARM servers) several JETSCAPE C++ dependencies are absent from most package
managers; the conda environment provides them (with an NVIDIA GPU, the arm64 containers
above are the alternative).

The scripts are in [`utils/conda_install/`](utils/conda_install/). See
[contribs/README.md](contribs/README.md) for the step-by-step setup: dry-run package check,
full build-environment install, and CMake integration.

## XML configuration requirements for new modules

Every JETSCAPE/X-SCAPE module class that reads parameters from XML
(i.e. any class that inherits from `JetScapeModuleBase` and calls
`GetXMLElement` / `GetXMLElementValue`) **must** have a corresponding
default entry in the framework's main XML file (`config/jetscape_main.xml`).

### Why this is required

`JetScape::Init()` calls `CompareElementsFromXML()` →
`recurseToSearch()` (see `src/framework/JetScape.cc`).
This function iterates every element in the *user* XML and looks for a
matching tag in the *main* XML.
If **any** user XML tag has no matching default in the main XML the
framework prints a `JSWARN` and immediately calls **`exit(-1)`** —
the process terminates with no further output.

> **This is not merely a warning.** The check is a hard requirement
> enforced at startup; there is no way to suppress it at runtime.

### What you must do when adding a new module

1. **Add default XML blocks to `config/jetscape_main.xml`** in the
   target X-SCAPE / JETSCAPE checkout, covering every tag your module
   reads via `GetXMLElement*`.
   The defaults should be safe fallback values — they are overridden
   by whatever the user specifies in their user XML.

2. **Place the block in the correct section** of `jetscape_main.xml`
   (within `<IS>`, `<Hard>`, `<Hydro>`, `<SoftParticlization>`, etc.)
   matching the lifecycle stage of the module.

3. **Document the required additions** in your contrib's own `README.md`
   so that users who integrate with an older X-SCAPE checkout know which
   blocks to add by hand.

### Example — FnoHydro

The `FnoHydro` contrib registers two module classes (`FnoHydro` and
`FnoRooIn`).  Both live in the `<Hydro>` section.  The required default
blocks that must be present in `config/jetscape_main.xml` are:

```xml
<FNO>
  <model_file>../fno_hydro/models/default.pt</model_file>
  <centrality-low>0</centrality-low>
  <centrality-high>10</centrality-high>
  <EOS_id_MUSIC>91</EOS_id_MUSIC>
  <n_features>3</n_features>
  <freezeout_temperature>0.136</freezeout_temperature>
  <nx>60</nx>
  <ny>60</ny>
  <ntau>50</ntau>
  <x_min>-15.0</x_min>
  <y_min>-15.0</y_min>
  <dtau>0.1</dtau>
  <neta>1</neta>
  <deta>0</deta>
</FNO>

<FNOROOIN>
  <model_file>../fno_hydro/models/default.pt</model_file>
  <root_file>default.root</root_file>
  <fullHydroIn>0</fullHydroIn>
  <bulkHadroFull>0</bulkHadroFull>
  <QAoutput>0</QAoutput>
  <centrality-low>0</centrality-low>
  <centrality-high>10</centrality-high>
  <EOS_id_MUSIC>91</EOS_id_MUSIC>
  <n_features>3</n_features>
  <tau0>0.5</tau0>
  <freezeout_temperature>0.136</freezeout_temperature>
  <nx>60</nx>
  <ny>60</ny>
  <neta>1</neta>
  <ntau>50</ntau>
  <x_min>-15.0</x_min>
  <y_min>-15.0</y_min>
  <dtau>0.1</dtau>
  <deta>0</deta>
</FNOROOIN>
```

These blocks are already included in the `js-contrib-test` branch of
[JETSCAPE/X-SCAPE](https://github.com/JETSCAPE/X-SCAPE).
Users of older checkouts must add them manually as described in
[Path A′](#path-a--manual-integration-into-older-x-scape-checkouts) above.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
