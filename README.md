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
