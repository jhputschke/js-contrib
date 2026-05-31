# Plan: Dev Container for X-SCAPE + music4gpu + js-contrib + FNO4d

## Overview

**Two** development container variants, differing only in the CUDA base image
and target GPU architecture list. Both provide all build and runtime
dependencies but do **not** clone or compile any of the four projects.
The user mounts local source trees and performs configure/build/install themselves.

| Variant | Dockerfile | CUDA base | Arch list | Target GPUs |
|---|---|---|---|---|
| **GH200** (max compat) | `Dockerfile.dev` | `12.6.3-devel-ubuntu24.04` | `75;80;86;89;90` | RTX 20/30/40xx, A100, H100, GH200 |
| **Blackwell** (newest gen) | `Dockerfile.dev.blackwell` | `13.2.1-devel-ubuntu24.04` | `75;80;86;89;90;100` | + GB200, GB10 (sm_100) |

All other phases (conda env, PyTorch, ENV vars, user workflow) are identical.
Both variants build for `linux/amd64` and `linux/arm64` via `docker buildx`.

**Contrast with `PlanContainer.md`**: that plan bakes source + compiled
artifacts into the image. This plan intentionally leaves the image source-free
so developers can work against their own forks and branches.

---

## Decisions

| Parameter | Choice | Reason |
|---|---|---|
| Base image (`Dockerfile.dev`) | `nvidia/cuda:12.6.3-devel-ubuntu24.04` | CUDA 12.6 devel; nvcc supports up to sm_90 (GH200) |
| Base image (`Dockerfile.dev.blackwell`) | `nvidia/cuda:13.2.1-devel-ubuntu24.04` | CUDA 13.2.1 (Apr 2026); first stable release supporting sm_100 (Blackwell) |
| Python | 3.11 via conda-forge | Stable for ROOT/Pythia8; matches production image |
| PyTorch (`Dockerfile.dev`) | 2.12+ via `cu126` pip wheel | Matches CUDA 12.6 base; provides libtorch for FnoHydro cmake |
| PyTorch (`Dockerfile.dev.blackwell`) | 2.12+ via `cu132` pip wheel | Native CUDA 13.2 wheel; no backward-compat layer needed |
| CUDA arch list (`Dockerfile.dev`) | `75;80;86;89;90` | sm_90 = Hopper (H100/GH200); max for CUDA 12.6 |
| CUDA arch list (`Dockerfile.dev.blackwell`) | `75;80;86;89;90;100` | sm_100 = Blackwell (GB200/GB10); requires CUDA ≥ 13.0 |
| ARM64 support | `ARG TARGETARCH` in Phase 2 | Both Dockerfiles build for amd64 + arm64; GH200/GB200 are ARM+NVLink SoC systems |
| Source code | **Not included** — user mounts checkout | Enables iterative dev against personal forks/branches |
| Build artifacts | **Not included** — user builds in-container | CMake cache persists on the volume between runs |
| Conda env name | `xscape` | Consistent with production image |
| Conda prefix | `/opt/miniforge3` | Fixed path so all ENV vars are stable |
| nvcc host compiler | System GCC 13 (`/usr/bin/g++`) | Avoids conda cross-compiler sysroot mismatch; Ubuntu 24.04 GCC 13 is compatible with both CUDA 12.6 and 13.2 |
| Jupyter port | 8888 exposed | Common dev workflow; user forwards with `-p 8888:8888` |

---

## Relevant Files

| File | Purpose |
|---|---|
| `js-contrib/utils/Dockerfile.dev` | GH200-compatible (CUDA 12.6, sm_75–sm_90) |
| `js-contrib/utils/Dockerfile.dev.blackwell` | Blackwell (CUDA 13.2.1, sm_75–sm_100) |
| `X-SCAPE/docker/.devcontainer/devcontainer.json` | **File to create** (VS Code Dev Container config) |
| `js-contrib/contribs/conda_install/install_js_fno_build_minimal.sh` | Reference for all conda/pip packages |
| `X-SCAPE/CMakeLists.txt` | CMake flags verified: `USE_MUSIC`, `USE_CUDA`, `USE_ROOT`, `USE_3DGlauber`, `USE_ISS`, `USE_JS_CONTRIB`, `USE_JS_FNO_HYDRO`, `USE_JS_PYJETSCAPE` |
| `X-SCAPE/external_packages/music4gpu/CMakeLists.txt` | `enable_language(CUDA)`, `find_package(CUDAToolkit)`, `CMAKE_CUDA_STANDARD 17`, `CMAKE_CUDA_ARCHITECTURES` |
| `FNO4d/requirements-base.txt` | Python base dependencies (platform-agnostic) |
| `FNO4d/requirements-linux.txt` | Linux pip requirements (includes base) |
| `FNO4d/requirements-gcs.txt` | GCS cloud storage deps (`gcsfs`, `google-cloud-storage`, `tqdm`) |

---

## Phase 1 — Base Layer

**`Dockerfile.dev`** — up to GH200 (sm_90):

```dockerfile
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04
ARG DEBIAN_FRONTEND=noninteractive
ARG TARGETARCH
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates ninja-build pkg-config \
    libx11-dev libxft-dev libxext-dev libxpm-dev libssl-dev \
  && rm -rf /var/lib/apt/lists/*
```

**`Dockerfile.dev.blackwell`** — GB200/GB10 (sm_100); only the `FROM` line and Phase 3 arch list differ:

```dockerfile
FROM nvidia/cuda:13.2.1-devel-ubuntu24.04
ARG DEBIAN_FRONTEND=noninteractive
ARG TARGETARCH
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates ninja-build pkg-config \
    libx11-dev libxft-dev libxext-dev libxpm-dev libssl-dev \
  && rm -rf /var/lib/apt/lists/*
```

Notes:
- `build-essential` (GCC 13, g++, make) and `/usr/local/cuda/bin` (nvcc) are
  already present in the `nvidia/cuda:*-devel` base — no extra install needed.
- X11 dev headers are required by ROOT even in headless/batch mode.
- Do **not** install cmake, boost, hdf5, gsl, or pythia8 via apt — these come
  from conda-forge to ensure a consistent ABI with all other conda-forge packages.
- `ARG TARGETARCH` is declared here and re-used in Phase 2 to select the correct
  Miniforge installer binary (`x86_64` vs `aarch64`) via `docker buildx`.

---

## Phase 2 — miniforge3 + conda Environment

Install miniforge3 into `/opt/miniforge3`. Create env `xscape` with Python 3.11.
All JETSCAPE C++ build dependencies come from conda-forge in a single solver pass.

```dockerfile
ARG TARGETARCH
RUN ARCH=$([ "$TARGETARCH" = "arm64" ] && echo "aarch64" || echo "x86_64") \
  && wget -q "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${ARCH}.sh" \
       -O /tmp/miniforge.sh \
  && bash /tmp/miniforge.sh -b -p /opt/miniforge3 \
  && rm /tmp/miniforge.sh \
  && /opt/miniforge3/bin/conda config --set auto_activate_base false

RUN /opt/miniforge3/bin/mamba create -n xscape python=3.11 -y -c conda-forge

RUN /opt/miniforge3/bin/mamba install -n xscape -y -c conda-forge \
    cmake make compilers \
    boost-cpp zlib hdf5 gsl \
    pythia8 hepmc3 fastjet \
    root \
    openmpi "pybind11>=2.11" \
    "numpy>=2.0" matplotlib scipy h5py zarr uproot awkward \
    opt-einsum tensorly scikit-image psutil seaborn tqdm \
    ruamel-yaml wandb \
    gcsfs google-cloud-storage \
    jupyterlab ipykernel

RUN /opt/miniforge3/bin/mamba clean -afy
```

Notes:
- The `compilers` package installs an arch-specific conda cross-compiler
  (`x86_64-conda-linux-gnu-g++` on amd64, `aarch64-conda-linux-gnu-g++` on arm64)
  and sets `CXX` to it. Used for all C++ TUs; ensures ABI compatibility with
  conda-forge Pythia8/ROOT/HDF5.
- nvcc uses a **separate** host compiler (`/usr/bin/g++`) — see CUDA Compilation Notes.

---

## Phase 3 — PyTorch + FNO4d Python Stack

**`Dockerfile.dev`** (CUDA 12.6 → GH200):
```dockerfile
RUN /opt/miniforge3/envs/xscape/bin/pip install \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cu126
```

**`Dockerfile.dev.blackwell`** (CUDA 13.2 → GB200/GB10):
```dockerfile
RUN /opt/miniforge3/envs/xscape/bin/pip install \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cu132
```

Both Dockerfiles then install the pip-only packages from `FNO4d/requirements-base.txt`
(all other FNO4d deps are already satisfied by the conda install in Phase 2):
```dockerfile
RUN /opt/miniforge3/envs/xscape/bin/pip install \
    tensorly-torch \
    "zencfg>=0.3.0" configmypy
```

`libtorch` (for FnoHydro cmake `find_package(Torch)`) is provided by the
installed `torch` wheel. Locate it at configure time with:

```bash
python -c "import torch; print(torch.utils.cmake_prefix_path)"
```

`neuraloperator` is **not** installed here — FNO4d vendors its own pinned
fork (`neuraloperator-4d-2.0.0`) which the user installs as an editable
install after mounting the FNO4d source tree (see User Workflow below).
Other FNO4d local libraries (`loc_libs`) follow the same pattern.

---

## Phase 4 — Environment Variables

Set once in the image so cmake `find_package` works without sourcing conda:

```dockerfile
ENV CONDA_ENV=/opt/miniforge3/envs/xscape \
    PATH=/opt/miniforge3/envs/xscape/bin:/opt/miniforge3/bin:$PATH \
    PYTHIA8=/opt/miniforge3/envs/xscape \
    PYTHIA8DIR=/opt/miniforge3/envs/xscape \
    ROOTSYS=/opt/miniforge3/envs/xscape \
    CMAKE_PREFIX_PATH=/opt/miniforge3/envs/xscape \
    CUDAHOSTCXX=/usr/bin/g++ \
    LD_LIBRARY_PATH=/opt/miniforge3/envs/xscape/lib

WORKDIR /workspace
EXPOSE 8888
CMD ["bash"]
```

- `CUDAHOSTCXX=/usr/bin/g++` pins nvcc's host compiler to Ubuntu 24.04 system
  GCC 13, avoiding the conda cross-compiler sysroot mismatch (see CUDA
  Compilation Notes).
- `/workspace` is the default mount point for the user's source tree.
- Port 8888 is for JupyterLab; forward with `docker run -p 8888:8888 ...`.

---

## User Workflow (after starting the container)

The steps below are run **inside** the running container, not during `docker build`.

### 1 — Start the container

```bash
# Mount all four source trees under /workspace
docker run --gpus all -it --rm \
  -v /path/to/X-SCAPE:/workspace/X-SCAPE \
  -v /path/to/FNO4d:/workspace/FNO4d \
  -v /path/to/build:/workspace/build \
  -v /path/to/models:/opt/models \
  -p 8888:8888 \
  xscape-fno4d-dev:latest bash
```

For VS Code, open the folder and use the provided `devcontainer.json` — the
editor handles the volume mounts automatically.

### 2 — Fetch external X-SCAPE packages

```bash
cd /workspace/X-SCAPE/external_packages
bash get_music4gpu.sh   # jhputschke/MUSIC4GPU, branch XSCAPE
bash get_js_contrib.sh  # jhputschke/js-contrib, branch main
bash get_iSS.sh
bash get_3dglauber.sh
bash get_lbtTab.sh      # LBT rate tables (~100 MB)
```

These scripts clone into `external_packages/` relative to X-SCAPE; they do not
require network access beyond the container's Docker network.

### 3 — Configure + Build X-SCAPE

All CMake flag names verified by grepping `X-SCAPE/CMakeLists.txt`.

```bash
TORCH_PREFIX=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")

cmake -S /workspace/X-SCAPE -B /workspace/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${CONDA_ENV};${TORCH_PREFIX}" \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
  -DUSE_MUSIC=ON \
  -DUSE_CUDA=ON  -DUSE_METAL=OFF \
  -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90" \
  -DUSE_3DGlauber=ON \
  -DUSE_ISS=ON \
  -DUSE_JS_CONTRIB=ON \
  -DUSE_JS_FNO_HYDRO=ON \
  -DUSE_JS_PYJETSCAPE=ON \
  -DUSE_ROOT=ON

cmake --build /workspace/build -j$(nproc)
```

Key flags:
- `USE_MUSIC=ON` + `USE_CUDA=ON` → X-SCAPE switches backend to `external_packages/music4gpu`.
- `-DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda` → keeps `find_package(CUDAToolkit)` stable
  when conda prepends PATH.
- `-DCMAKE_CUDA_ARCHITECTURES` → must not be `native` (see CUDA Compilation Notes).
  Use `"75;80;86;89;90"` for `Dockerfile.dev`; use `"75;80;86;89;90;100"` for
  `Dockerfile.dev.blackwell` (sm_100 requires CUDA 13 nvcc).
- `USE_JS_CONTRIB=ON` must precede `USE_JS_FNO_HYDRO` and `USE_JS_PYJETSCAPE`.
- HepMC3 is auto-detected via `find_package(HEPMC)` — no explicit flag needed.

### 4 — FNO4d Editable Installs

Run **after** step 3 (PyJetscape `.so` is produced by the cmake build).

```bash
pip install -e /workspace/FNO4d/loc_libs
pip install -e /workspace/FNO4d/neuraloperator-4d-2.0.0
pip install -e /workspace/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape
```

### 5 — Verify

```bash
# CUDA + PyTorch
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"

# PyJetscape
python -c "import jetscape; print('PyJetscape OK')"

# nvcc
nvcc --version

# X-SCAPE binary
/workspace/build/FinalStateHadrons --help
```

---

## VS Code Dev Container

Create `.devcontainer/devcontainer.json` next to `X-SCAPE/docker/Dockerfile.dev`:

```jsonc
{
  "name": "X-SCAPE + FNO4d Dev",
  "build": {
    "dockerfile": "../docker/Dockerfile.dev",
    "context": ".."
  },
  "runArgs": [
    "--gpus", "all",
    "--shm-size=8g"
  ],
  "mounts": [
    "source=${localWorkspaceFolder},target=/workspace/X-SCAPE,type=bind",
    "source=${localWorkspaceFolder}/../FNO4d,target=/workspace/FNO4d,type=bind,consistency=cached",
    "source=${localWorkspaceFolder}/build_gpu,target=/workspace/build,type=bind",
    "source=${localEnv:HOME}/fno_models,target=/opt/models,type=bind,readonly"
  ],
  "workspaceFolder": "/workspace/X-SCAPE",
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-python.python",
        "ms-vscode.cpptools",
        "ms-vscode.cmake-tools",
        "ms-toolsai.jupyter"
      ],
      "settings": {
        "python.defaultInterpreterPath": "/opt/miniforge3/envs/xscape/bin/python",
        "cmake.configureSettings": {
          "CMAKE_BUILD_TYPE": "Release",
          "CUDA_TOOLKIT_ROOT_DIR": "/usr/local/cuda"
        }
      }
    }
  },
  "postCreateCommand": "echo 'Run: cd /workspace/X-SCAPE/external_packages && bash get_music4gpu.sh && bash get_js_contrib.sh && bash get_iSS.sh && bash get_3dglauber.sh && bash get_lbtTab.sh'"
}
```

Notes:
- `--shm-size=8g` prevents shared-memory exhaustion in PyTorch DataLoader workers.
- The `postCreateCommand` reminds the user to fetch external packages; it does
  not run them automatically (they require network access).
- If the build tree should persist between container rebuilds, point the `build`
  mount at a directory outside the workspace folder.

---

## CUDA Compilation Notes for music4gpu

Verified by inspecting `music4gpu/CMakeLists.txt`, `src/CMakeLists.txt`,
`src/gpu/CUDAPipelines.h`, `src/gpu/music_kernels.cuh`, and `src/advance.h`.

### What nvcc compiles

Three `.cu` files registered via `set_source_files_properties(... LANGUAGE CUDA)`:

| File | Role |
|---|---|
| `src/gpu/GPUGrid_cuda.cu` | GPU memory management; `cudaMallocManaged` (GH200 coherent) vs `cudaMalloc` + pinned staging (discrete GPUs) |
| `src/gpu/music_kernels.cu` | All seven `__global__` hydro kernels — direct CUDA C port of the Metal MSL shaders |
| `src/gpu/CUDAPipelines.cu` | Singleton driver: device selection, stream creation, kernel dispatch, `reduce_max` |

No OpenCL, no Thrust, no cuBLAS. No separable compilation — all TUs compile
independently; no device linking step required.

### nvcc host compiler: MUST use system GCC, not conda cross-compiler

`CUDAHOSTCXX=/usr/bin/g++` is set in Phase 4. The conda `compilers` package
exports `CXX=x86_64-conda-linux-gnu-g++`; if nvcc picks this up as its host
compiler the conda sysroot breaks CUDA system header lookups.

`/usr/bin/g++` is GCC 13 from Ubuntu 24.04 — fully compatible with CUDA 12.6.
C++ TUs still use the conda cross-compiler (via `CXX`) for ABI compatibility
with conda-forge Pythia8/ROOT/HDF5.

### `CMAKE_CUDA_ARCHITECTURES` must not be `native` unless a GPU is present

`music4gpu/CMakeLists.txt` defaults to `native`, which calls
`nvcc --gpu-architecture=native` and requires a physical GPU. If cmake is run
in a CPU-only environment (CI, `docker build`), always pass an explicit list:

**`Dockerfile.dev`** (up to GH200):
```
-DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90"
```
sm_75 (Turing/RTX 20xx), sm_80 (A100), sm_86 (RTX 3090/A40),
sm_89 (RTX 4090/L4), sm_90 (H100/H200/GH200).

**`Dockerfile.dev.blackwell`** (GB200/GB10):
```
-DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90;100"
```
sm_100 = Blackwell (GB200, GB10). Requires CUDA 13+ nvcc — provided by the
`13.2.1-devel-ubuntu24.04` base image.

When running cmake **inside** the container with a GPU attached (e.g. in a VS
Code terminal via `docker run --gpus all`), `native` will also work and produces
a smaller binary optimised for the local GPU.

### Runtime GPU override env vars

| Variable | Effect |
|---|---|
| `MUSIC_FORCE_CPU=1` | Disables GPU dispatch entirely at runtime without rebuild |
| `MUSIC_CUDA_FORCE_COHERENT=1` | Forces unified-memory path (`cudaMallocManaged`) |
| `MUSIC_CUDA_FORCE_DISCRETE=1` | Forces PCIe staging path (default on discrete GPUs) |

---

## Further Considerations

1. **Image size**: ~12–15 GB (no source or build artifacts). ROOT ~2 GB, CUDA
   devel headers ~4 GB, conda packages ~5 GB, PyTorch ~3 GB. Significantly
   smaller than the production image.

2. **Rebuild frequency**: The dev image only needs rebuilding when a dependency
   version is bumped. Source and build changes are entirely on the user's volume.

3. **arm64 + CUDA**: CUDA is functional on arm64 only on NVIDIA-connected ARM
   hosts: Jetson (Orin), GH200 (Grace Hopper, sm_90), GB200/GB10 (Grace
   Blackwell, sm_100). On generic ARM (Graviton, Apple Silicon) the image builds
   fine but `torch.cuda.is_available()` returns `False` and music4gpu falls back
   to CPU. Both Dockerfiles support arm64 via the `ARG TARGETARCH` / `docker
   buildx --platform linux/arm64` path in Phase 2.

4. **Two-compiler strategy for CUDA + conda**:
   - `CXX` → `x86_64-conda-linux-gnu-g++` (conda cross-compiler) — all C++ TUs;
     ensures ABI compatibility with conda-forge Pythia8/ROOT/HDF5.
   - `CUDAHOSTCXX` → `/usr/bin/g++` (system GCC 13) — nvcc host compiler only;
     avoids the conda sysroot breaking CUDA header resolution.
   Both are set as `ENV` in the Dockerfile; cmake and nvcc pick them up automatically.

5. **FnoHydro model files**: Download `.pt` TorchScript files from
   [Zenodo record 16647726](https://zenodo.org/records/16647726) and mount at
   `/opt/models` with `-v /local/models:/opt/models`. The XML config must
   reference `/opt/models/<file>.pt` as the model path.

6. **JupyterLab**: Start inside the container with:
   ```bash
   jupyter lab --ip=0.0.0.0 --no-browser --NotebookApp.token=''
   ```
   Forward port 8888 from `docker run -p 8888:8888` or via VS Code port forwarding.

7. **Pinned environment for reproducibility**: After finalising the dependency
   set, export a lockfile with:
   ```bash
   conda env export -n xscape --no-builds > environment-lock.yml
   ```
   Check this file in alongside the Dockerfiles so builds are reproducible even
   when conda-forge packages update.

8. **PyTorch wheel version**: PyTorch **2.12.0** (stable, May 2026) ships native
   `cu126`, `cu130`, and `cu132` wheels on `https://download.pytorch.org/whl/`.
   The two containers use the best-matched wheel:
   - `Dockerfile.dev` (CUDA 12.6 base) → `cu126`
   - `Dockerfile.dev.blackwell` (CUDA 13.2 base) → `cu132` (native; no backward-compat layer)

   No fallback or backward-compatibility hacks are required.

9. **Build commands** — build and tag each variant:
   ```bash
   # GH200-compatible (amd64 + arm64)
   docker buildx build --platform linux/amd64,linux/arm64 \
     -f X-SCAPE/docker/Dockerfile.dev \
     -t xscape-fno4d-dev:gh200 X-SCAPE/docker/

   # Blackwell (amd64 + arm64)
   docker buildx build --platform linux/amd64,linux/arm64 \
     -f X-SCAPE/docker/Dockerfile.dev.blackwell \
     -t xscape-fno4d-dev:blackwell X-SCAPE/docker/
   ```
