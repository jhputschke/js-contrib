# Building and Publishing the Production Container Images

> **Status: plan (2026-09-30).** `Dockerfile.prod`, `Dockerfile.prod.blackwell` and the
> workflow `.github/workflows/docker-prod.yml` described here do not exist yet. The dev
> images are in [`BuildContainerDev.md`](BuildContainerDev.md).

The production images run the two-stage hydro productions of
[`prod_AuAu_0_10_jet`](../contribs/PyJetscape/example/prod_AuAu_0_10_jet/README.md)
(`run_prod_jet.py`, `run_jobs.sh`, `hadronize.py`) on machines other than the GB10: HPC
clusters through Apptainer/Singularity, cloud VMs through Docker. Unlike the dev images,
they hold a **complete, precompiled** X-SCAPE with MUSIC4GPU, so nothing is compiled on the
target machine:

| | dev images | production images |
|---|---|---|
| purpose | develop: sources mounted, built by the user | run productions: read-only, ready to run |
| X-SCAPE + MUSIC4GPU | not included | built in, pinned commits, fat CUDA binary |
| PyTorch, FNO4d, JupyterLab | yes | **no** |
| ROOT | yes | no (to be confirmed, see [Open points](#open-points)) |
| base image shipped | `cuda:*-devel` | `cuda:*-runtime` (multi-stage) |
| size | ~12–15 GB | estimated 4–6 GB (to be measured) |

---

## Images

| Image tag | Dockerfile | CUDA | CPU arch | `CMAKE_CUDA_ARCHITECTURES` | Target GPUs |
|---|---|---|---|---|---|
| `xscape-prod:cu126` | `Dockerfile.prod` | 12.6 | amd64, arm64 | `75-real;80-real;86-real;89-real;90` | RTX 20/30/40xx, A100, A40, L4, H100, GH200 |
| `xscape-prod:cu130` | `Dockerfile.prod.blackwell` | 13.x | amd64, arm64 | `90-real;100-real;120-real;121` | GH200, B200/GB200, RTX 50xx / RTX PRO Blackwell, **GB10** |

Each tag is one multi-arch manifest: `docker pull` and `apptainer pull` pick the image of
the machine's CPU architecture.

**Two independent architectures.** The CPU architecture (amd64 / arm64) decides which image
of the manifest is used. The GPU architectures (`sm_*`) are compiled into each image. The
last entry of each list, without `-real`, also embeds PTX, which the driver compiles for
GPUs newer than the list at the first launch.

**GB10 is compute capability 12.1 (sm_121), not sm_100.** `nvidia-smi` on the GB10 reports
12.1, and `build_gpu` compiles MUSIC4GPU with `-arch=native`, which CMake resolves to
`121-real`. sm_100 code does not run on it (only the embedded compute_100 PTX would, through
JIT compilation at the first launch). [`BuildContainerDev.md`](BuildContainerDev.md),
`Dockerfile.dev.blackwell` and `docs/PlanContainerDev.md` list GB10 as sm_100 and should be
corrected.

**Driver.** The CUDA version in the image must be supported by the host's driver:
`nvidia-smi` shows the highest CUDA version it supports. CUDA 13 needs the R580 driver
series or newer (the GB10 has 580.173); CUDA 12.6 runs on older drivers. Check a cluster
before choosing the tag.

---

## What goes into the image

### Stage 1 — build (not shipped)

Base `nvidia/cuda:12.6.3-devel-ubuntu24.04` (`cu126`) or `nvidia/cuda:13.x-devel-ubuntu24.04`
(`cu130`), then miniforge3 and a conda env `xscape` (conda-forge only):

| layer | packages | why |
|---|---|---|
| build tools | `cmake make compilers "pybind11>=2.11"` | conda's C++ compiler for ABI compatibility with the conda libraries |
| C++ libraries | `boost-cpp zlib hdf5 gsl pythia8` | X-SCAPE, MUSIC4GPU, Pythia8 for PythiaGun |
| Python | `python=3.12 "numpy>=2"` | the interpreter `pyjetscape_core` is built against (as `js_fno`) |
| Python leaves (pip) | `h5py hdf5plugin pyyaml` | everything the production scripts import at run time |

Not included: PyTorch (`jetscape` imports it only optionally), FNO4d, ROOT, HepMC3 (the
current `build_gpu` has none), FastJet, JupyterLab.

Sources at **pinned commits** (build arguments, recorded in the image, see
[Provenance](#provenance)):

| repo | fetched by | pin today |
|---|---|---|
| X-SCAPE (`contrib`) | `git clone` + `git checkout $XSCAPE_REF` | `9509aea2` |
| MUSIC4GPU (`XSCAPE`) | `external_packages/get_music4gpu.sh` (also downloads the hotQCD EoS) | `15ec5e3`, the head of `XSCAPE` and what `build_gpu` is built from; the script pins it since X-SCAPE `9509aea2` |
| iSS (`common_seeds`, fork) | `get_iSS.sh` (+ HRG and δf tables) | `3192982` |
| 3dMCGlauber (`JETSCAPE`) | `get_3dglauber.sh` (+ LHAPDF data) | `71116fe` |
| LBT tables | `get_lbtTab.sh` | (1.2 GB) |
| js-contrib (`main`) | `get_js_contrib.sh` + `git checkout $JS_CONTRIB_REF` | `main` |

Configure and build, into a tree called `build_gpu` so the production scripts' default
`--build` works unchanged:

```bash
export CUDAHOSTCXX=/usr/bin/g++      # nvcc's host compiler: the system GCC, not conda's
cmake -S /opt/X-SCAPE -B /opt/X-SCAPE/build_gpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_CUDA=ON -DUSE_MUSIC=ON -DUSE_ISS=ON -DUSE_3DGlauber=ON \
  -DUSE_JS_CONTRIB=ON -DUSE_JS_PYJETSCAPE=ON -DUSE_JS_FNO_HYDRO=OFF \
  -DUSE_ROOT=OFF \
  -DCMAKE_CUDA_ARCHITECTURES="75-real;80-real;86-real;89-real;90"   # cu130: "90-real;100-real;120-real;121"
cmake --build /opt/X-SCAPE/build_gpu -j"$(nproc)"
```

- **Never `native`** here: MUSIC4GPU's CMake defaults to `CMAKE_CUDA_ARCHITECTURES=native`,
  which needs a GPU at configure time. GitHub's runners and `docker build` have none.
- **Two compilers**, as in the dev images: the conda cross-compiler (`CXX`) for C++, the
  system GCC (`CUDAHOSTCXX`) for nvcc. conda's sysroot breaks nvcc's CUDA header lookup.
- The build compiles every `.cu` file once per listed architecture: expect the MUSIC4GPU part
  to take several times as long as a `native` build.

### Stage 2 — runtime (shipped)

Base `nvidia/cuda:12.6.3-runtime-ubuntu24.04` / `nvidia/cuda:13.x-runtime-ubuntu24.04`
(libcudart, no nvcc, no headers), then:

1. **A runtime conda env** with the same versions as stage 1 minus the build tools:
   `python numpy boost-cpp zlib hdf5 gsl pythia8`, plus `h5py hdf5plugin pyyaml`. Created
   from stage 1's `conda list --explicit` filtered to these packages, so the versions match
   what the libraries were linked against.
2. **The X-SCAPE tree** copied from stage 1 without build intermediates (`*.o`, `CMakeFiles/`,
   test binaries). What the production reads from it:

   | path | what | size |
   |---|---|---|
   | `build_gpu/` libraries | X-SCAPE, MUSIC4GPU, iSS, 3dMCGlauber | (to be measured) |
   | `external_packages/js-contrib/contribs/PyJetscape/python/jetscape/` | `pyjetscape_core` + the HDF5 writers | |
   | `build_gpu/music_input`, `mcglauber.input`, `iSS_parameters.dat` | inputs the scripts copy into each job's directory | small |
   | `build_gpu/EOS/`, `iSS_tables/` | MUSIC's hotQCD EoS; iSS's HRG / δf tables | 6 MB, 250 MB |
   | `build_gpu/LBT-tables/` | LBT (and Matter's heavy-quark recoil) | 1.2 GB |
   | `build_gpu/{tables,eps09,LHAPDF_Lib,nucleusConfigs,data_table}` | Glauber, PDFs, nPDFs, nuclei | |
   | `config/`, the production directories under `contribs/PyJetscape/example/` | main XML, user XMLs, grid YAMLs, scripts | small |

3. **Environment:**

   ```dockerfile
   ENV CONDA_ENV=/opt/miniforge3/envs/xscape \
       PATH=/opt/miniforge3/envs/xscape/bin:$PATH \
       LD_LIBRARY_PATH=/opt/miniforge3/envs/xscape/lib:$LD_LIBRARY_PATH \
       PYTHIA8DATA=/opt/miniforge3/envs/xscape/share/Pythia8/xmldoc \
       XSCAPE=/opt/X-SCAPE
   WORKDIR /opt/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape/example/prod_AuAu_0_10_jet
   ```

   `PYTHIA8DATA` has to be set: without it, importing `pyjetscape_core` can hang
   (`run_prod.py` checks for this).

**The image is read-only, and that fits the scripts.** `run_prod.py` already runs each job in
its own working directory, `OUTDIR/work/<name>`: it copies `music_input` there, links the
read-only assets from the build tree, and points `XSCAPE_DATA_DIR`, `HYDROPROGRAMPATH` and
`LBT_TABLES_PATH` at the build tree. So only `OUTDIR` and its parent, where the seed registry
`seeds_used.tsv` lives, have to be writable: bind the parent (see [Running](#running-the-image)).

### Provenance

The build writes `/opt/X-SCAPE/BUILD_INFO.txt` and OCI labels
(`org.opencontainers.image.revision`, …) with:
- the commit of every repo above, and whether its tree was clean;
- the CUDA version, `CMAKE_CUDA_ARCHITECTURES`, the compilers, the conda package list.

The production files already store the producing build (`prod_build`); in the container that
is `/opt/X-SCAPE/build_gpu`, so the image tag (and digest) must be recorded per campaign,
e.g. in the campaign's `run_jobs.campaign` or the job `.json`. (Planned change to
`run_prod.py`: read `BUILD_INFO.txt` when present and store it as an attribute.)

---

## Option A — GitHub Actions (recommended)

A workflow `.github/workflows/docker-prod.yml`, copied from `docker-dev.yml`: four jobs on
native runners (no QEMU), then one merge per tag.

```
build (cu126-amd64) ─┐
build (cu126-arm64) ─┼─→ merge → jhputschke/xscape-prod:cu126
build (cu130-amd64) ─┤
build (cu130-arm64) ─┴─→ merge → jhputschke/xscape-prod:cu130
```

Differences from the dev workflow:

| | dev | prod |
|---|---|---|
| triggers | push to `main` changing a Dockerfile; manual | manual, with the commits as inputs (`XSCAPE_REF`, `MUSIC4GPU_REF`, `JS_CONTRIB_REF`, …); pushing a `prod-*` tag |
| tags | `cu126`, `cu132` | `cu126`, `cu130`, and one per campaign release, e.g. `cu126-2026.10` (never overwritten) |
| work in the job | installs dependencies | also clones, downloads ~1.5 GB of tables, compiles X-SCAPE with several CUDA architectures |
| timeout | 180 min | 180 min; to be adjusted after the first build |

The secrets are the same (`DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`); see the one-time setup in
[`BuildContainerDev.md`](BuildContainerDev.md#one-time-setup). The `free-disk-space` step is
needed as well.

**Why native runners matter even more here:** the build compiles C++ and CUDA, and under
QEMU emulation nvcc and the Pythia8/X-SCAPE compile take hours.

---

## Option B — Local build on the GB10 (arm64 only)

The GB10 is arm64 with Docker and the NVIDIA container runtime, so it builds and tests the
**arm64** image natively. It cannot build the amd64 image in reasonable time (see Option C).

```bash
cd js-contrib/utils

docker build -t xscape-prod:cu130-arm64 -f Dockerfile.prod.blackwell \
  --build-arg XSCAPE_REF=9509aea2 --build-arg MUSIC4GPU_REF=15ec5e3 \
  --build-arg JS_CONTRIB_REF=main .

docker run --rm --gpus all -v "$PWD/prod:/work" xscape-prod:cu130-arm64 \
  python run_prod_jet.py --events 1 --seed 1 --outdir /work/out
```

This is the place to run the [validation](#validation) before an image is published.

---

## Option C — Local multi-arch build with buildx (slow)

```bash
docker buildx create --use --name multiarch-builder
docker buildx build --platform linux/amd64,linux/arm64 \
  -t yourname/xscape-prod:cu126 -f Dockerfile.prod --push .
```

> **Warning:** on the GB10 (arm64), `linux/amd64` runs under QEMU: compiling X-SCAPE and
> MUSIC4GPU there takes hours and can run out of memory. Use Option A, or add a native amd64
> machine as a remote buildx node (`docker buildx create --append ssh://user@x86host`).

---

## Option D — Build on the cluster with Apptainer

Only where neither Docker Hub nor GitHub can be used. Apptainer builds from a definition file
that starts from the same base image (`Bootstrap: docker`) and runs the same steps:

```bash
apptainer build --fakeroot xscape-prod.sif xscape-prod.def
```

It needs `--fakeroot` (or root) on the build machine, no GPU (thanks to the explicit
architecture list), and has to be done per CPU architecture. It gives up most of what the
precompiled image is for, so treat it as a fallback.

---

## Running the image

### HPC: Apptainer / Singularity

```bash
apptainer pull xscape_prod.sif docker://jhputschke/xscape-prod:cu126

apptainer exec --nv --bind "$SCRATCH/prod:/work" xscape_prod.sif \
  python run_prod_jet.py --events 25 --seed 0 --write-particlize both --outdir /work/out
```

- `--nv` makes the host's NVIDIA driver visible in the container.
- **Bind the parent of `--outdir`,** not `--outdir` itself: the seed registry of `--seed 0`
  is `OUTDIR/../seeds_used.tsv`, and a registry inside the read-only image can't be written.
- **Mixed clusters:** a `.sif` holds one CPU architecture, and `apptainer pull` takes the
  architecture of the machine it runs on. Where the login nodes are x86 but the GPU nodes are
  GH200/GB200 (arm64), pull with `--arch arm64`, or pull on a compute node.
- **Threads:** set `OMP_NUM_THREADS` to the job's cores (see the machine settings in
  [`BENCHMARK_GB10.md`](../docs/BENCHMARK_GB10.md)).
- **Campaigns:** one GPU per job through the scheduler (a SLURM job array: seed 0 per array
  task, `--campaign NAME`), or `run_jobs.sh -j P` inside one allocation. `run_jobs.sh --mps`
  starts CUDA MPS; whether its control binary is reachable inside the container depends on the
  site's Apptainer `--nv` setup. Test it before relying on it.
- A SLURM job-array template, `utils/slurm_prod_array.sh`, is planned next to the Dockerfiles.

### Cloud VMs: Docker

```bash
docker run --rm --gpus all -v "$PWD/prod:/work" jhputschke/xscape-prod:cu126 \
  python run_prod_jet.py --events 25 --seed 0 --outdir /work/out
```

The VM needs the NVIDIA driver and the NVIDIA container toolkit.

### Getting the outputs home

Copy `OUTDIR` to Google Cloud Storage or a Pelican/OSDF namespace, with the command-line tools
(`gcloud storage cp -r`, `pelican object put`) or with the Python interfaces of the analysis
environment ([`analysis_env/README.md`](analysis_env/README.md#remote-files-google-cloud-storage-and-pelicanosdf)).

---

## Validation

Before publishing an image (Option B on the GB10), and once on every new cluster or GPU type:

1. **Starts:** `nvidia-smi`; `python -c "import jetscape; assert jetscape.HAS_CORE"`;
   `cat /opt/X-SCAPE/BUILD_INFO.txt`.
2. **Configuration:** `python run_prod_jet.py --events 1 --seed 1 --dry-run`.
3. **Null test:** `--no-deposit`: the jet leg must equal the background leg exactly
   (`arr == arr_bg`).
4. **Regression against the GB10 baseline:** seed 1 against a native `build_gpu` run at the
   same commits, with the production regression recipe.
   - On the GB10, the image's sm_121 code comes from the same compiler and flags as `native`:
     expect bit-identical output. Any difference is a finding.
   - On other GPU types expect small floating-point differences (`--use_fast_math`, other
     hardware): compare within tolerances, e.g. the energy balance and spectra of the seed-1
     event, not bit for bit.
5. **Throughput:** events per hour for one job and for `-j` jobs per GPU, recorded next to
   the GB10 and M3 Max numbers.

`MUSIC_FORCE_CPU=1` runs MUSIC4GPU on the CPU inside the same image: a slow but useful check
when a GPU result looks wrong.

---

## Open points

- **ROOT:** the pair and particlize files are written from Python with h5py, so
  `USE_ROOT=OFF` should suffice. To be confirmed by building without it and running the
  validation; if something needs it, add `root_base` to both stages (+~2 GB).
- **Versions:** the `get_*.sh` scripts pin what production uses (MUSIC4GPU `15ec5e3` since
  X-SCAPE `9509aea2`), so `MUSIC4GPU_REF` is only needed to build something else. Keep the
  pins and `build_gpu` in step; decide whether X-SCAPE is taken from `contrib` or a release
  tag.
- **Size:** measure the image. If the 1.2 GB of LBT tables matter, they could be fetched once
  per site into a bind-mounted directory instead (`LBT_TABLES_PATH`).
- **Registry:** Docker Hub (`jhputschke/xscape-prod`) as for the dev images; the GitHub
  Container Registry (`ghcr.io`) avoids Docker Hub's pull limits on clusters. For the
  OSPool, images can also be distributed unpacked through CVMFS.
- **Corrections elsewhere:** GB10 as sm_121 (not sm_100) in `BuildContainerDev.md`,
  `Dockerfile.dev.blackwell` and `docs/PlanContainerDev.md`.
