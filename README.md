# js-contrib

Community contributions for the [JETSCAPE](https://github.com/JETSCAPE/JETSCAPE) /
[X-SCAPE](https://github.com/JETSCAPE/X-SCAPE) heavy-ion simulation framework,
analogous to [fastjet-contrib](https://fastjet.hepforge.org/contrib/) for FastJet.

## Available contribs

| Contrib | Description | Extra deps |
|---------|-------------|------------|
| [FnoHydro](contribs/FnoHydro/) | Neural-network (FNO) hydrodynamics via LibTorch | ROOT, libtorch (~2 GB) |
| [PyJetscape](contribs/PyJetscape/) | pybind11 Python bindings + PyFNOHydro trampoline | pybind11, PyTorch |

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

# 2. Rebuild with desired contribs
cd build
cmake .. -DUSE_JS_CONTRIB=ON -DUSE_JS_FNO_HYDRO=ON   # or -DUSE_JS_PYJETSCAPE=ON
make -j$(nproc)
```

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
  -DCMAKE_PREFIX_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")

make -j$(nproc)
```

## Environment setup (Mac Silicon / Linux aarch64)

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
