# js-contrib

Community contributions for the [JETSCAPE](https://github.com/JETSCAPE/JETSCAPE) /
[X-SCAPE](https://github.com/JETSCAPE/X-SCAPE) heavy-ion simulation framework,
analogous to [fastjet-contrib](https://fastjet.hepforge.org/contrib/) for FastJet.

## Available contribs

| Contrib | Description | Extra deps |
|---------|-------------|------------|
| [FnoHydro](contribs/FnoHydro/) | Neural-network (FNO) hydrodynamics via LibTorch | ROOT, libtorch (~2 GB) |
| [PyJetscape](contribs/PyJetscape/) | pybind11 Python bindings + PyFNOHydro trampoline | pybind11 (pip/conda auto-detected), PyTorch |

## Source provenance

The source files in this repository were copied from
[JETSCAPE-FNO](https://github.com/JETSCAPE/JETSCAPE-FNO) at commit
`jhputschke/JETSCAPE-FNO @ PythonTest` (April 2026).
JETSCAPE-FNO remains the upstream source of truth; changes should be made there
and synced here.

## Compatibility

| js-contrib | X-SCAPE | JETSCAPE-FNO branch |
|------------|---------|---------------------|
| v0.1.x     | ≥ main  | PythonTest          |

## Installation

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

If your X-SCAPE checkout predates the js-contrib integration, apply the two
steps below by hand (they mirror exactly what the up-to-date `CMakeLists.txt`
and `external_packages/get_js_contrib.sh` already contain).

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

#### Step 3 — patch `src/CMakeLists.txt` and the top-level `CMakeLists.txt` (build-tree export)

js-contrib discovers JetScape via a build-tree `export()`. The key constraint is
that `export()` must be called **after** every optional `add_subdirectory()` that
defines an in-tree target that `JetScape` links to (e.g. `music`, `iSS`). Because
`add_subdirectory(./src)` is processed before those optional subdirectories in the
top-level `CMakeLists.txt`, the `export()` call must be placed at the **end of
the top-level `CMakeLists.txt`**, not inside `src/CMakeLists.txt`.

**Remove** any existing `export(…)` / `configure_file(…JetScapeConfig…)` lines
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

## Environment setup (Mac Silicon / Linux aarch64)

#### REMARK: Docker/Singularity containers for Linux x86 and arm64 will be provided asap. Mac Silicon containers, once there is MPS provided (maybe for testing purposes a CPU container will be provided soon).

The official JETSCAPE Docker images target Linux x86_64. On **Mac Silicon
(`arm64`)** the MPS (Metal Performance Shaders) PyTorch backend is not
available inside containers, and on **Linux `aarch64`** (AWS Graviton, ARM
servers) several JETSCAPE C++ dependencies are absent from most package
managers and container registries. For both cases the recommended approach is
a native conda environment.

See [contribs/README.md](contribs/README.md) for the step-by-step setup:
dry-run package check, full build-environment install, and CMake integration.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
