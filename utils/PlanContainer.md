# Plan: Unified Docker Image for X-SCAPE + music4gpu + js-contrib + FNO4d

## Overview

Single-stage CUDA development image based on `nvidia/cuda:12.6.3-devel-ubuntu24.04`
(multi-arch `linux/amd64` + `linux/arm64`). Uses miniforge3/mamba for all physics C++
packages (Pythia8, ROOT, HepMC3, Boost, HDF5, GSL) via conda-forge, then pip for
PyTorch (≥ 2.10, CUDA wheel `cu126`) and the FNO4d ML stack. All four projects are
cloned from their remotes at image build time.

**Included**: core X-SCAPE, music4gpu CUDA, FnoHydro, PyJetscape, iSS, 3D MCGlauber,
ROOT, HepMC3, FNO4d.  
**Excluded**: SMASH, IP-Glasma, CLVisc, Metal backend.

---

## Decisions

| Parameter | Choice | Reason |
|---|---|---|
| Base image | `nvidia/cuda:12.6.3-devel-ubuntu24.04` | CUDA 12.6 devel headers + Ubuntu 24.04; covers `cu126` PyTorch wheel |
| Python | 3.11 via conda-forge | More battle-tested than 3.12 for ROOT/Pythia8 at build time |
| PyTorch | Latest ≥ 2.10 via `cu126` pip wheel | Provides libtorch for FnoHydro C++ cmake integration |
| CUDA architectures | `75;80;86;89;90` (sm_75 → sm_90) | No GPU needed at `docker build`; covers RTX 30/40/A100/H100 |
| Image type | Single-stage dev image | Easy to iterate; compilers + full conda env retained |
| Source strategy | `git clone` at build time | Reproducible; pins to known-good branches |
| FnoHydro model files | Volume mount (`/opt/models`) | Models from Zenodo record 16647726; not baked into image |
| nvcc host compiler | System GCC 13 (`/usr/bin/g++`) | Avoids conda cross-compiler sysroot mismatch with CUDA headers |

---

## Relevant Files

| File | Purpose |
|---|---|
| `X-SCAPE/docker/Dockerfile` | **File to create** |
| `X-SCAPE/CMakeLists.txt` | CMake flags: `USE_MUSIC`, `USE_CUDA`, `USE_ROOT`, `USE_3DGlauber`, `USE_ISS`, `USE_JS_CONTRIB`, `USE_JS_FNO_HYDRO`, `USE_JS_PYJETSCAPE` |
| `X-SCAPE/external_packages/music4gpu/CMakeLists.txt` | Top-level CUDA setup: `enable_language(CUDA)`, `find_package(CUDAToolkit)`, `CMAKE_CUDA_STANDARD 17`, `CMAKE_CUDA_ARCHITECTURES` |
| `X-SCAPE/external_packages/music4gpu/src/CMakeLists.txt` | CUDA source list, `set_source_files_properties(... LANGUAGE CUDA)`, `CUDA::cudart` link, `--use_fast_math`, `-Xcompiler` OpenMP passthrough |
| `X-SCAPE/external_packages/music4gpu/src/gpu/CUDAPipelines.h` | CUDA singleton — uses `void*` for stream handles to avoid CUDA headers in C++ TUs |
| `X-SCAPE/external_packages/music4gpu/src/advance.h` | `#ifdef USE_CUDA` guard includes `CUDAPipelines.h`; `using GPUPipelines = CUDAPipelines` alias |
| `js-contrib/contribs/conda_install/install_js_fno_build_minimal.sh` | Reference for all conda/pip packages |
| `FNO4d/requirements-base.txt` | Python base dependencies |
| `FNO4d/requirements-linux.txt` | Linux-specific (includes base) |
| `FNO4d/install.sh` | CUDA detection + editable install logic |
| `FNO4d/neuraloperator-4d-2.0.0/` | Local PyTorch FNO fork, installed as `pip install -e` |
| `FNO4d/loc_libs/` | Local data I/O library, installed as `pip install -e` |

---

## Phase 1 — Base Layer

```dockerfile
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates ninja-build pkg-config \
    libx11-dev libxft-dev libxext-dev libxpm-dev libssl-dev \
  && rm -rf /var/lib/apt/lists/*
```

Notes:
- The X11 dev headers (`libx11-dev` etc.) are required by ROOT even in headless/batch mode.
- Do **not** install cmake, boost, hdf5, gsl, or pythia8 via apt — these come from
  conda-forge (newer, controlled versions with consistent ABI).
- `build-essential` (GCC 13, g++, make) is already present in the `nvidia/cuda:*-devel` base.
  Ubuntu 24.04 GCC 13 is used as the nvcc host compiler (see CUDA Compilation Notes).
- `/usr/local/cuda/bin` (containing `nvcc`) is already on PATH in the devel base image.

---

## Phase 2 — conda Environment (miniforge3)

Install miniforge3 into `/opt/miniforge3`. Disable `auto_activate_base`.
Create env `xscape` with Python 3.11.

Single `mamba install -n xscape -c conda-forge` RUN layer (for Docker layer-cache efficiency):

```
cmake make compilers
boost-cpp zlib hdf5 gsl
pythia8 hepmc3 fastjet
root
openmpi pybind11>=2.11
numpy>=1.25 matplotlib scipy h5py zarr uproot awkward
opt-einsum tensorly scikit-image psutil seaborn tqdm
jupyterlab ipykernel
```

The `compilers` package installs a conda-specific cross-prefixed GCC
(`x86_64-conda-linux-gnu-g++`) used for all C++ TUs in X-SCAPE (ABI matches
conda-forge physics libraries). nvcc uses a **different** host compiler — see
CUDA Compilation Notes.

Persist ENV vars in the image so cmake `find_package` works without sourcing conda:

```
CONDA_ENV=/opt/miniforge3/envs/xscape
PYTHIA8=/opt/miniforge3/envs/xscape
PYTHIA8DIR=/opt/miniforge3/envs/xscape
ROOTSYS=/opt/miniforge3/envs/xscape
CMAKE_PREFIX_PATH=/opt/miniforge3/envs/xscape
CUDAHOSTCXX=/usr/bin/g++
```

`CUDAHOSTCXX=/usr/bin/g++` pins nvcc's host compiler to Ubuntu 24.04 system GCC 13,
avoiding the conda cross-compiler sysroot mismatch (see CUDA Compilation Notes).

---

## Phase 3 — PyTorch + FNO4d Python Stack

```bash
# Activate env, then:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install ruamel-yaml "zencfg>=0.3.0" configmypy wandb
```

This places libtorch inside the conda env site-packages.
`find_package(Torch)` in `FnoHydro/CMakeLists.txt` locates it via:
```bash
python -c "import torch; print(torch.utils.cmake_prefix_path)"
```

Do this **before** building X-SCAPE so the Torch cmake prefix is available at configure time.

---

## Phase 4 — Clone Source Repos

```bash
# X-SCAPE (branch: music4gpu_test)
git clone --branch music4gpu_test --depth 1 \
  https://github.com/JETSCAPE/X-SCAPE.git /opt/xscape
cd /opt/xscape && git submodule update --init --recursive

# External packages via provided fetch scripts
cd /opt/xscape/external_packages
bash get_music4gpu.sh   # jhputschke/MUSIC4GPU, branch XSCAPE
bash get_js_contrib.sh  # jhputschke/js-contrib, branch main
bash get_iSS.sh         # iSS particle sampler
bash get_3dglauber.sh   # 3D MCGlauber (includes vendored LHAPDF_Lib)
bash get_lbtTab.sh      # LBT rate tables (~100 MB binary data)

# FNO4d (Codeberg)
git clone --depth 1 https://codeberg.org/jhputschke/FNO4d.git /opt/fno4d
pip install -r /opt/fno4d/requirements-linux.txt   # pulls in requirements-base.txt
```

---

## Phase 5 — Build X-SCAPE

All CMake flag names verified by grepping `X-SCAPE/CMakeLists.txt`.

```bash
TORCH_PREFIX=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")

cmake -S /opt/xscape -B /opt/xscape/build \
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

cmake --build /opt/xscape/build -j$(nproc)
```

Notes:
- `USE_MUSIC=ON` + `USE_CUDA=ON` → X-SCAPE CMakeLists switches backend to
  `external_packages/music4gpu` (verified in `X-SCAPE/CMakeLists.txt` lines 101–102).
- `-DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda` makes `find_package(CUDAToolkit REQUIRED)`
  robust even if conda's cmake re-orders PATH.
- HepMC3 is **auto-detected** via `find_package(HEPMC)` — no explicit flag needed;
  conda env provides it and `CMAKE_PREFIX_PATH` covers it.
- Pythia8 is found via the `PYTHIA8` / `PYTHIA8DIR` ENV vars set in Phase 2.
- Expected build time: 20–40 min parallelized.

---

## Phase 6 — FNO4d Editable Installs

Must run **after** Phase 5 (the PyJetscape pybind11 `.so` is produced by the cmake build).

```bash
pip install -e /opt/fno4d/loc_libs
pip install -e /opt/fno4d/neuraloperator-4d-2.0.0
pip install -e /opt/xscape/external_packages/js-contrib/contribs/PyJetscape
```

---

## Phase 7 — Runtime Configuration

Persist in image via `ENV`:

```
XSCAPE_DATA_DIR=/opt/xscape/build
HYDROPROGRAMPATH=/opt/xscape/build
LBT_TABLES_PATH=/opt/xscape/external_packages/LBT-tables
LD_LIBRARY_PATH=/opt/xscape/build:/opt/miniforge3/envs/xscape/lib
PATH=/opt/miniforge3/envs/xscape/bin:/opt/xscape/build:$PATH
```

Volume declarations:

```dockerfile
VOLUME /opt/models   # FnoHydro TorchScript .pt files — download from Zenodo 16647726
VOLUME /work         # user workspace: XML configs, output files
WORKDIR /work
CMD ["bash"]
```

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

No OpenCL, no Thrust, no cuBLAS. No **separable compilation** (`CUDA_SEPARABLE_COMPILATION`
not set) — all TUs compile independently; no device linking step required.

### How nvcc is invoked by cmake

- `CMAKE_CUDA_STANDARD 17` → `nvcc --std=c++17`
- `--use_fast_math` applied to all CUDA TUs via `target_compile_options` generator expression
- OpenMP passed to host compiler via `$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>`
- `CUDA::cudart` linked via `find_package(CUDAToolkit)` → `/usr/local/cuda/lib64/libcudart.so`

### nvcc host compiler: MUST use system GCC, not conda cross-compiler

The conda `compilers` package installs `x86_64-conda-linux-gnu-g++` and exports `CXX`
to it. If nvcc picks up this conda cross-compiler as its host compiler, the
conda-specific sysroot causes CUDA system header lookups to fail.

**Fix**: `CUDAHOSTCXX=/usr/bin/g++` set in the Phase 2 ENV block.

`/usr/bin/g++` is GCC 13 from Ubuntu 24.04. CUDA 12.6 supports GCC ≤ 13 — fully
compatible. The C++ translation units (`advance.cpp`, etc.) continue to use the conda
cross-compiler (via `CXX`) so linking against conda-forge Pythia8/ROOT/HDF5 works.

### `CMAKE_CUDA_ARCHITECTURES` must NOT be `native` in a headless build

`music4gpu/CMakeLists.txt` defaults `CMAKE_CUDA_ARCHITECTURES` to `native`, which
calls `nvcc --gpu-architecture=native` and requires a physical GPU at **build time**.
Inside `docker build` there is no GPU — this will abort.

The cmake invocation in Phase 5 **must** include:
```
-DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90"
```
This cross-compiles PTX + SASS for sm_75 (RTX 20xx), sm_80 (A100), sm_86 (RTX 3090/A40),
sm_89 (RTX 4090/L4), sm_90 (H100/H200) — no GPU needed at build time.

### How C++ TUs consume CUDA interfaces without CUDA headers

`CUDAPipelines.h` stores `cudaStream_t`/`cudaEvent_t` handles as `void*` — intentional
design so `advance.cpp` (compiled by g++) can include it without any CUDA headers.
The `#ifdef USE_CUDA` guard in `src/advance.h` activates the include, and the
`using GPUPipelines = CUDAPipelines` alias makes host dispatch code backend-agnostic.
The `-DUSE_CUDA` define flows from `target_compile_definitions(${libname} PRIVATE USE_CUDA)`
in `src/CMakeLists.txt`.

---

## Build & Verification

```bash
# Build amd64 (add --platform linux/amd64,linux/arm64 for multi-arch with QEMU)
docker buildx build --platform linux/amd64 -t xscape-fno4d:latest X-SCAPE/docker/

# Validate nvcc is accessible and correct version
docker run xscape-fno4d:latest nvcc --version

# 1. CUDA + PyTorch smoke test
docker run --gpus all xscape-fno4d:latest \
  python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"

# 2. music4gpu CUDA kernel
docker run --gpus all xscape-fno4d:latest \
  bash -c "OMP_NUM_THREADS=4 /opt/xscape/build/MUSIChydro input_file"

# 3. X-SCAPE full run
docker run --gpus all \
  -v /path/to/config:/work \
  -v /path/to/models:/opt/models \
  xscape-fno4d:latest \
  /opt/xscape/build/FinalStateHadrons /work/config.xml

# 4. PyJetscape import
docker run xscape-fno4d:latest python -c "import jetscape; print('PyJetscape OK')"

# 5. FNO4d training
docker run --gpus all -v /path/to/data:/work xscape-fno4d:latest \
  python /opt/fno4d/train/test_train.py

# 6. Multi-arch push (requires docker buildx + QEMU binfmt registered)
docker buildx build --platform linux/amd64,linux/arm64 \
  --push -t <registry>/xscape-fno4d:latest X-SCAPE/docker/
```

---

## Further Considerations

1. **Expected image size**: ~20–25 GB (ROOT ~2 GB, CUDA devel headers ~4 GB,
   build artifacts ~3 GB, conda packages ~5 GB, PyTorch ~3 GB).
   Use a `.dockerignore` to exclude `.git/`, `build_*/`, `*.pt` model files from
   the build context (they are cloned inside the container, not COPYed).

2. **arm64 + CUDA**: CUDA is only functional on arm64 for NVIDIA Jetson (Orin) or
   GH200/GB200 (Grace-Blackwell) ARM hosts. On AWS Graviton or other generic ARM,
   the image builds fine but `torch.cuda.is_available()` returns `False` and
   music4gpu falls back to CPU at runtime.

3. **Build-time GPU not required**: `CMAKE_CUDA_ARCHITECTURES="75;80;86;89;90"` is
   set explicitly, so nvcc cross-compiles for all targets without needing a GPU
   present during `docker build`. Fully CI/CD-compatible.

4. **Two-compiler strategy for CUDA + conda**:
   - `CXX` → conda cross-compiler (`x86_64-conda-linux-gnu-g++`) — compiles all C++ TUs;
     ensures ABI compatibility with conda-forge Pythia8/ROOT/HDF5.
   - `CUDAHOSTCXX` → system GCC 13 (`/usr/bin/g++`) — used by nvcc as host compiler;
     avoids the conda sysroot breaking CUDA header resolution.
   Both are set in the Dockerfile `ENV` block; cmake picks them up automatically.

5. **FnoHydro model files**: Download `.pt` TorchScript files from
   [Zenodo record 16647726](https://zenodo.org/records/16647726) and mount at
   runtime with `-v /local/models:/opt/models`. The XML config must reference
   `/opt/models/<file>.pt` as the model path.

6. **Runtime GPU override env vars** (from `music4gpu/PORT_GPU_CUDA.md`):
   - `MUSIC_FORCE_CPU=1` — disable GPU dispatch without rebuild
   - `MUSIC_CUDA_FORCE_COHERENT=1` — force unified-memory path (GH200 emulation on discrete hardware)
   - `MUSIC_CUDA_FORCE_DISCRETE=1` — force PCIe staging path (default on all discrete GPUs)
