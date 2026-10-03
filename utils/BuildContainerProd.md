# Building and Publishing the Production Container Images

> **Status (2026-10-01).** [`Dockerfile.prod`](Dockerfile.prod) exists, one Dockerfile for
> all CUDA variants (see [Images](#images)).
> - **Published:** the [GitHub workflow](#option-a--github-actions-recommended) has built and
>   pushed `jhputschke/xscape-prod:cu126` and `:cu130` (amd64 + arm64, 2026-09-30, from
>   js-contrib `f0a1a08`) and `:cu124` (amd64, 2026-10-01, from `818fdbf`), all with X-SCAPE
>   `d31946c`.
> - **`cu126` and `cu130` predate two Dockerfile changes:** Pelican/OSDF (`03349ea`) and, in
>   `cu126`, V100 code (sm_70, `818fdbf`). Until the workflow runs again, the published
>   `cu126` and `cu130` have no Pelican and `cu126` doesn't run on a V100; `cu124` has both.
> - **Tested:** a local arm64 `cu130` build on the GB10 (see [Tested](#tested)): production
>   runs and agrees with the native build. The published images, and every amd64 image, are
>   not tested on a machine yet.
>
> The dev images are in [`BuildContainerDev.md`](BuildContainerDev.md): published for amd64
> and arm64, not tested on aarch64.

**For running a production with the images** (Docker, Apptainer, SLURM, hadronization on
CPUs, analysis and ROOT export), see the user guide
[`docs/README_2stage.md`](../docs/README_2stage.md).
This file is about the images themselves.

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
| ROOT | yes | no; `--build-arg WITH_ROOT=1` adds `root_base` (see [ROOT](#root)) |
| base image shipped | `cuda:*-devel` | `cuda:*-base` (multi-stage: libcudart only) |
| size | ~12–15 GB | 4.7 GB (`cu130`, arm64) |

---

## Images

| Image tag | build arguments of `Dockerfile.prod` | CPU arch | Target GPUs |
|---|---|---|---|
| `xscape-prod:cu126` | defaults: `CUDA_VERSION=12.6.3`, `CUDA_ARCHITECTURES="70-real;75-real;80-real;86-real;89-real;90"` | amd64, arm64 | V100, RTX 20/30/40xx, A100, A40, L4, H100, GH200 |
| `xscape-prod:cu130` | `CUDA_VERSION=13.2.1`, `CUDA_ARCHITECTURES="90-real;100-real;120-real;121"` | amd64, arm64 | GH200, B200/GB200, RTX 50xx / RTX PRO Blackwell, **GB10** |
| `xscape-prod:cu124` | `CUDA_VERSION=12.4.1`, `UBUNTU=ubuntu22.04`, architectures as `cu126` | **amd64 only** | as `cu126`, on hosts with R550 drivers (CUDA 12.4) |

One Dockerfile serves all three: the variants differ only in the CUDA base image and the
architecture list, so copies could only drift apart.

Each tag is one multi-arch manifest: `docker pull` and `apptainer pull` pick the image of
the machine's CPU architecture. `cu124` holds only the amd64 image.

**Remote storage.** Every image has Pelican/OSDF: `pelicanfs` for Python (`osdf://`,
`pelican://`) and the `pelican` command-line tool. Google Cloud Storage for Python (`gcsfs`,
`google-cloud-storage`) is optional: `--build-arg WITH_GCS=1`, published as `cu126-gcs`,
`cu130-gcs`, `cu124-gcs`. The `gcloud` CLI is in neither (about 1 GB). See
[Getting the outputs home](#getting-the-outputs-home).

**Why `cu124`.** On a host whose driver supports CUDA 12.4 but not 12.6 (the R550 series),
Docker's NVIDIA runtime refuses to start the `cu126` image (the `nvidia/cuda` images require
their CUDA version, `NVIDIA_REQUIRE_CUDA`). `cu124` is the same build on CUDA 12.4.
- NVIDIA publishes CUDA 12.4 images only up to Ubuntu 22.04, so `cu124` is Ubuntu 22.04 based,
  and nvcc's host compiler is that release's GCC 11 (supported by CUDA 12.4). The conda env,
  and so the C++ compiler (GCC 14) and every library, are the same as in the other variants.
- It is built only when requested (`variants=cu124` or `all`), and for amd64 only.
- **Built and published** on GitHub (2026-10-01); not tested on a machine yet.

**V100 (sm_70) needs `cu126` or `cu124`.** CUDA 13 dropped Volta, so `cu130` has no code
for it. The embedded PTX only runs on GPUs as new as its architecture or newer (compute_90
in `cu126`), never older ones. A build without the GPU's architecture fails on it with "no
kernel image is available for execution on the device". Before MUSIC4GPU stopped on that
error, such a run went on and wrote files with meaningless hydro; discard them.

**Two independent architectures.** The CPU architecture (amd64 / arm64) decides which image
of the manifest is used. The GPU architectures (`sm_*`) are compiled into each image. The
last entry of each list, without `-real`, also embeds PTX, which the driver compiles for
GPUs newer than the list at the first launch.

**GB10 is compute capability 12.1 (sm_121), not sm_100.** `nvidia-smi` on the GB10 reports
12.1, and `build_gpu` compiles MUSIC4GPU with `-arch=native`, which CMake resolves to
`121-real`. sm_100 code does not run on it (only the embedded compute_100 PTX would, through
JIT compilation at the first launch). [`BuildContainerDev.md`](BuildContainerDev.md),
`Dockerfile.dev.blackwell` and `docs/Plans/PlanContainerDev.md` list GB10 as sm_100 and should be
corrected.

**Driver.** The CUDA version in the image must be supported by the host's driver:
`nvidia-smi` shows the highest CUDA version it supports. CUDA 13 needs the R580 driver
series or newer (the GB10 has 580.173); CUDA 12.6 needs R560, CUDA 12.4 R550 (`cu124`).
Check a cluster before choosing the tag.

---

## What goes into the image

### Stage 1 — build (not shipped)

Base `nvidia/cuda:12.6.3-devel-ubuntu24.04` (`cu126`) or `nvidia/cuda:13.x-devel-ubuntu24.04`
(`cu130`), then miniforge3 and a conda env `xscape` (conda-forge only):

| layer | packages | why |
|---|---|---|
| build tools | `"cmake>=3.28,<4" make gcc_linux-<arch>=14 gxx_linux-<arch>=14 "pybind11>=2.11"` | conda's GCC 14 (as for `build_gpu`) for ABI compatibility with the conda libraries; CMake < 4, since some external packages still declare `cmake_minimum_required` < 3.5, which CMake 4 refuses |
| C++ libraries | `boost-cpp zlib hdf5 gsl pythia8` | X-SCAPE, MUSIC4GPU, Pythia8 for PythiaGun |
| Python | `python=3.12 "numpy>=2"` | the interpreter `pyjetscape_core` is built against (as `js_fno`) |
| Python leaves (pip) | `h5py hdf5plugin pyyaml` | everything the production scripts import at run time |

Not included: PyTorch (`jetscape` imports it only optionally), FNO4d, ROOT, HepMC3 (the
current `build_gpu` has none), FastJet, JupyterLab.

Three details the first builds turned up:
- **NVIDIA's CUDA apt repository** (configured in the `nvidia/cuda` images) is dropped before
  `apt-get update`: nothing comes from it, and a mirror sync in progress there failed the build.
- **`pkg-config`** is needed at build time: conda's `h5c++` wrapper, which trento's CMake
  runs, calls it.
- **conda's `lhapdf` is removed** after the solve (`conda remove --force lhapdf`, which keeps
  `pythia8`). Recent conda-forge `pythia8` builds pull in `lhapdf` 6.5 for Pythia's optional
  LHAPDF6 plugin. conda's `-L$PREFIX/lib` then comes first on the link line, and
  3dMCGlauber's `Metropolis.e`, compiled against the LHAPDF 6.2.1 it bundles, links 6.5 and
  fails (`LHAPDF::mkPDF`). The GB10's `js_fno` has no conda `lhapdf` either (its `pythia8`
  build predates the dependency).

Sources at **pinned commits** (build arguments, recorded in the image, see
[Provenance](#provenance)):

| repo | fetched by | pin today |
|---|---|---|
| X-SCAPE (`contrib`) | shallow fetch of `$XSCAPE_REF` (a full 40-character hash, a branch or a tag) | `cbc72639`: with #157/#158 (see [ROOT](#root)), the slim `bulk_info` (#161) that js-contrib `main` needs since `9356bf4`, and the MUSIC4GPU pin below |
| MUSIC4GPU (`XSCAPE`) | `external_packages/get_music4gpu.sh` (also downloads the hotQCD EoS) | `49439c0` (PR #15, the store releases its memory), the head of `XSCAPE` and what `build_gpu` is built from; the script pins it since X-SCAPE `9509aea2` |
| iSS (`common_seeds`, fork) | `get_iSS.sh` (+ HRG and δf tables) | `3192982` |
| 3dMCGlauber (`JETSCAPE`) | `get_3dglauber.sh` (+ LHAPDF data) | `71116fe` |
| LBT tables | `get_lbtTab.sh` | (1.2 GB) |
| js-contrib (`main`) | shallow fetch of `$JS_CONTRIB_REF` | `main` |

Configure and build, into a tree called `build_gpu` so the production scripts' default
`--build` works unchanged:

```bash
export CUDAHOSTCXX=/usr/bin/g++      # nvcc's host compiler: the system GCC, not conda's
cmake -S /opt/X-SCAPE -B /opt/X-SCAPE/build_gpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_CUDA=ON -DUSE_MUSIC=ON -DUSE_ISS=ON -DUSE_3DGlauber=ON \
  -DUSE_JS_CONTRIB=ON -DUSE_JS_PYJETSCAPE=ON -DUSE_JS_FNO_HYDRO=OFF \
  -DCMAKE_CUDA_ARCHITECTURES="70-real;75-real;80-real;86-real;89-real;90"   # cu130: "90-real;100-real;120-real;121"
cmake --build /opt/X-SCAPE/build_gpu -j"$(nproc)"
```

- **Never `native`** here: MUSIC4GPU's CMake defaults to `CMAKE_CUDA_ARCHITECTURES=native`,
  which needs a GPU at configure time. GitHub's runners and `docker build` have none.
- **Two compilers**, as in the dev images: the conda cross-compiler (`CXX`) for C++, the
  system GCC (`CUDAHOSTCXX`) for nvcc. conda's sysroot breaks nvcc's CUDA header lookup.
- The build compiles every `.cu` file once per listed architecture: expect the MUSIC4GPU part
  to take several times as long as a `native` build.

### ROOT

The image has no ROOT: the pair and particlize files are written from Python with h5py, and
nothing in the production path uses ROOT. X-SCAPE doesn't make that straightforward:
- the top level forces `USE_ROOT` on (`set(USE_ROOT ON)` before the `option`, so
  `-DUSE_ROOT=OFF` has no effect), but finds ROOT optionally: without ROOT, `-DUSE_ROOT` and
  the `src/root` include directory are left out;
- `src/CMakeLists.txt` still compiled `src/root/*.cc`, which include ROOT headers, and
  linked `${ROOT_LIBRARIES}`, both under `if(USE_ROOT)`. So a build without ROOT failed.

- `src/framework/JetScape.cc` included `RootBulkWriter.h` and `FastRootBulkWriter.h`, and
  created both writers, unconditionally.

Both are fixed on `contrib` since `d31946c0`: JETSCAPE/X-SCAPE#157 builds `src/root` only
with `USE_ROOT AND ROOT_FOUND`, and #158 puts the two writers in `JetScape.cc` under
`#ifdef USE_ROOT` (an XML asking for one then gets a warning). With ROOT installed nothing
changes. A ROOT-free build of an older `XSCAPE_REF` stops right after fetching the sources,
with a message that names the two PRs.

PyJetscape binds its ROOT writer only under `-DUSE_ROOT`, so without ROOT
`jetscape.HAS_ROOT` is `False` and everything else works. For a ROOT-enabled image,
`--build-arg WITH_ROOT=1` installs conda-forge's `root_base` (ROOT's core libraries, not the
full `root` metapackage) in both stages.

### Stage 2 — runtime (shipped)

Base `nvidia/cuda:12.6.3-base-ubuntu24.04` / `nvidia/cuda:13.2.1-base-ubuntu24.04`: the
CUDA runtime (libcudart) only, which is all MUSIC4GPU links. No nvcc, no headers, no CUDA
math libraries. Then:

1. **A runtime conda env:** stage 1's `conda list --explicit` minus the build tools (CMake,
   make, the compilers, binutils, the sysroot, pybind11), so every library is the exact build
   it was linked against. `h5py`, `hdf5plugin` and `pyyaml` at the versions stage 1 installed.
1. **Remote storage** (stage 2 only; stage 1 doesn't need it):
   - **`pelicanfs`**, from `analysis_env/requirements_pelican.txt`, the analysis
     environment's file, so both environments get the same packages;
   - **the `pelican` CLI**, `/usr/local/bin/pelican`: the release binary of
     `PELICAN_VERSION` (default `7.26.2`), checked against the release's `checksums.txt`;
   - **with `WITH_GCS=1`:** `gcsfs` and `google-cloud-storage`, from
     `analysis_env/requirements_gcs.txt`.

   This layer comes before the X-SCAPE copy, so a new X-SCAPE build reuses it.
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
- the CUDA version, `CMAKE_CUDA_ARCHITECTURES`, the compilers, the conda package list;
- the `pelican` CLI and `pelicanfs` versions, and `WITH_GCS`. The labels `xscape.pelican`
  and `xscape.with_gcs` carry the same, and `/opt/X-SCAPE/pip-freeze-runtime.txt` lists
  every Python package of the runtime env.

The production files already store the producing build (`prod_build`); in the container that
is `/opt/X-SCAPE/build_gpu`, so the image tag (and digest) must be recorded per campaign,
e.g. in the campaign's `run_jobs.campaign` or the job `.json`. (Planned change to
`run_prod.py`: read `BUILD_INFO.txt` when present and store it as an attribute.)

---

## Option A — GitHub Actions (recommended)

The workflow [`.github/workflows/docker-prod.yml`](../.github/workflows/docker-prod.yml)
builds the images on GitHub's native runners (amd64 and arm64, no QEMU) and pushes them to
Docker Hub. It runs **only when started by hand**.

```
setup: resolve the refs to commits, pick the variants and arches
  ├─ build cu126-amd64 ─┐
  ├─ build cu126-arm64 ─┴─ merge → <user>/xscape-prod:cu126, :cu126-<date>-<xscape7>
  ├─ build cu130-amd64 ─┐
  ├─ build cu130-arm64 ─┴─ merge → <user>/xscape-prod:cu130, :cu130-<date>-<xscape7>
  └─ build cu124-amd64 ─── merge → <user>/xscape-prod:cu124, :cu124-<date>-<xscape7>   (on request)
```

### Starting a build

On the website: **Actions** → **Build & push production images** → **Run workflow**, fill in
the inputs, **Run workflow**. Or from the command line:

```bash
gh workflow run docker-prod.yml                          # the latest versions: cu126 + cu130, both arches
gh workflow run docker-prod.yml -f variants=cu124        # CUDA 12.4, amd64 only
gh workflow run docker-prod.yml -f variants=all          # cu126, cu130 and cu124
gh workflow run docker-prod.yml -f variants=cu130 -f platforms=arm64 -f push=false   # test one image
gh workflow run docker-prod.yml -f release_tag=2026.10   # + tags cu126-2026.10, cu130-2026.10
gh workflow run docker-prod.yml -f with_gcs=true         # with GCS: tags cu126-gcs, cu130-gcs, ...
gh run watch                                             # follow the run
```

The run's summary page lists the commits it built and the `docker pull` / `apptainer pull`
commands for the new tags.

### Which versions are built

| input | default | takes |
|---|---|---|
| `xscape_ref` | `contrib` | a branch, a tag or a full commit hash of JETSCAPE/X-SCAPE |
| `music4gpu_ref` | `XSCAPE` | the same for jhputschke/MUSIC4GPU, or `pinned`: the commit X-SCAPE's `get_music4gpu.sh` pins |
| `js_contrib_ref` | `main` | the same for jhputschke/js-contrib |
| `variants` | `cu126+cu130` | `cu126+cu130`, `all` (+ `cu124`), `cu126`, `cu130` or `cu124` (see [Images](#images)) |
| `platforms` | `both` | `amd64`, `arm64` or both; a published tag should have both. `cu124` is built for amd64 only, whatever is chosen (`cu124` with `arm64` stops with an error) |
| `with_gcs` | off | on: `WITH_GCS=1`, Google Cloud Storage for Python; every tag gets `-gcs` after the variant (`cu126-gcs`, `cu126-gcs-<YYYYMMDD>-<xscape7>`), so the images without it keep their tags |
| `release_tag` | empty | an extra tag per variant, e.g. `2026.10` → `cu126-2026.10` |
| `push` | on | off: build only, to test a change of the Dockerfile |

- **By default you get the latest versions:** the heads of X-SCAPE `contrib`, MUSIC4GPU
  `XSCAPE` and js-contrib `main` at the moment the run starts.
- **For specific versions,** pass a tag or a **full 40-character commit hash**; GitHub serves
  single commits only by their full hash. Get it with `git rev-parse <short hash>`, or from
  the commit's page on GitHub. A short hash stops the run with an error saying so.
- **The first job resolves every branch or tag to its commit,** so all builds of a run use the
  same commits, even if someone pushes during the build. The commits are recorded in the run
  summary, in the image labels (`docker inspect`), and in `/opt/X-SCAPE/BUILD_INFO.txt`.
- iSS, 3dMCGlauber and the LBT tables always come at the commits X-SCAPE's `get_*.sh` scripts
  pin, so they follow `xscape_ref`.
- A ROOT-free image needs X-SCAPE `d31946c0` or later (#157/#158). An older `xscape_ref`
  stops right after fetching, with a message.
- js-contrib `main` from `9356bf4` on needs X-SCAPE `011a4679` or later (JETSCAPE/X-SCAPE#161,
  the slim `bulk_info`): against an older `xscape_ref`, `pyjetscape_core` does not compile.
  Build an older X-SCAPE with an older `js_contrib_ref`.

### Tags

| tag | moves? | use |
|---|---|---|
| `cu126`, `cu130`, `cu124` | yes: every run that builds the variant overwrites it | trying out the latest build |
| `cu126-<YYYYMMDD>-<xscape7>` | no | what a campaign should pull: a fixed image |
| `cu126-<release_tag>` | only if the same `release_tag` is given again | a name for a campaign's image |

With `with_gcs`, the same tags with `-gcs` after the variant: `cu126-gcs`,
`cu126-gcs-<YYYYMMDD>-<xscape7>`, `cu126-gcs-<release_tag>`.

Every tag is one multi-arch manifest: the builds push their images untagged, by digest, and a
merge job tags the manifest. Runs are serialized, so two builds never race for the moving
tags.

### One-time setup and costs

- The secrets `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN`, the same as for the dev images: see
  the one-time setup in [`BuildContainerDev.md`](BuildContainerDev.md#one-time-setup). The
  images go to `<DOCKERHUB_USERNAME>/xscape-prod`.
- The workflow file has to be on `main` for the **Run workflow** button to appear.
- The builds free runner disk space first (the `free-disk-space` step, as for the dev images)
  and cache their layers in GitHub's cache, one slot per variant and arch.
- The arm64 runners are billed at twice the minute rate. The GB10, with 20 cores, compiles
  the X-SCAPE part in about 3 min; GitHub's 4-core runners take several times as long. The
  timeout is 4 h; to be adjusted after the first runs.

## Option B — Local build on the GB10 (arm64 only)

The GB10 is arm64 with Docker and the NVIDIA container runtime, so it builds and tests the
**arm64** image natively. It cannot build the amd64 image in reasonable time (see Option C).

```bash
cd js-contrib/utils

docker build -t xscape-prod:cu130-arm64 -f Dockerfile.prod \
  --build-arg CUDA_VERSION=13.2.1 \
  --build-arg CUDA_ARCHITECTURES="90-real;100-real;120-real;121" .

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

PROD=/opt/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape/example/prod_AuAu_0_10_jet
apptainer exec --nv --pwd "$PROD" --bind "$SCRATCH/prod:/work" xscape_prod.sif \
  python run_prod_jet.py --events 25 --seed 0 --write-particlize both --outdir /work/out
```

- `--nv` makes the host's NVIDIA driver visible in the container.
- **`--pwd "$PROD"`:** Docker starts in the image's production directory (its `WORKDIR`).
  Apptainer ignores that and starts in the directory you launched it from, usually your home.
  Without `--pwd`, `python run_prod_jet.py` and `./run_jobs.sh` are not found, and a relative
  OUTDIR such as `./` lands in that directory. The other way is to call the scripts by their
  full path, `$PROD/run_jobs.sh`.
- **OUTDIR's parent must be writable too:** every job records its seed, explicit or drawn, in
  the seed registry `OUTDIR/../seeds_used.tsv`. So bind a directory and put OUTDIR one level
  inside it (`--bind "$SCRATCH/prod:/work"`, OUTDIR `/work/out`), not the bound directory
  itself; everything outside the bound directories is the read-only image.
- **Your home directory:** Apptainer mounts it, but only the directory itself. Its parent
  (e.g. `/home/group/`) is an empty stand-in inside the read-only image. So OUTDIR `~` (or `./`
  started from home) fails with "Read-only file system" on `seeds_used.tsv`; use `~/prod/out`
  instead. `run_jobs.sh` checks this before it starts any job and says what to do.
- **Mixed clusters:** a `.sif` holds one CPU architecture, and `apptainer pull` takes the
  architecture of the machine it runs on. Where the login nodes are x86 but the GPU nodes are
  GH200/GB200 (arm64), pull with `--arch arm64`, or pull on a compute node.
- **Threads:** set `OMP_NUM_THREADS` to the job's cores (see the machine settings in
  [`BENCHMARK_GB10.md`](../docs/BENCHMARK_GB10.md)).
- **Campaigns:** `run_jobs.sh -j P` inside one allocation, and a SLURM job array for many
  GPUs, one GPU per array task ([SLURM](#slurm-one-gpu-per-array-task)). `run_jobs.sh --mps`
  starts CUDA MPS. Inside a container that is not assured: on the GB10 under Docker the MPS
  daemon does not start (see [Several GPUs on one machine](#several-gpus-on-one-machine)),
  so test it on each site before relying on it. For several GPUs in one allocation, see that
  section too.
- **SLURM:** see [SLURM: one GPU per array task](#slurm-one-gpu-per-array-task).

### SLURM: one GPU per array task

[`slurm_prod_array.sh`](slurm_prod_array.sh) runs a campaign as a SLURM job array: each
array task gets one GPU and fills it with `P` production jobs at once (`run_jobs.sh -j P`).

**Why not one SLURM job per production job:** SLURM gives each job whole GPUs. One
production job keeps a GPU only partly busy, because the CPU stages dominate an event
(~40 % on the GB10, less on faster GPUs). Only jobs inside one allocation can share its
GPU. If the site offers GPU sharing (`--gres=shard` or `--gres=mps`), many small SLURM
jobs are an option too.

```bash
apptainer pull xscape_prod.sif docker://jhputschke/xscape-prod:cu126-<YYYYMMDD>-<xscape7>
SIF=$PWD/xscape_prod.sif CAMPAIGN=PbPb sbatch slurm_prod_array.sh     # 10 tasks = 10 GPUs
SIF=$PWD/xscape_prod.sif CAMPAIGN=PbPb sbatch --array=0-19 slurm_prod_array.sh
SIF=$PWD/xscape_prod.sif CAMPAIGN=PbPb BINDS="$HOME/xml:/xml" \
  EXTRA_ARGS="--user-xml /xml/PbPb_0_10.xml" sbatch slurm_prod_array.sh
```

Settings, as environment variables at `sbatch` (defaults in the script). `sbatch` passes
your environment to the job by default (`--export=ALL`); where a site turns that off, use
`sbatch --export=ALL,CAMPAIGN=PbPb,…` or edit the defaults in the script.

| variable | default | meaning |
|---|---|---|
| `SIF` | `$HOME/xscape_prod.sif` | the image |
| `WORK` | `$SCRATCH/xscape_prod` (`$HOME/…` without `$SCRATCH`) | bound to `/work`: writable, the same for all tasks |
| `CAMPAIGN` | `AuAu_0_10_jet` | the campaign; task `t` writes `WORK/CAMPAIGN/t<ttt>/`, named `CAMPAIGN_t<ttt>` |
| `P` | 4 | production jobs at once per GPU |
| `NJOBS`, `EVENTS` | 40, 25 | jobs (files) per task, events per job |
| `SEED_MODE` | `registry` | `registry` or `ranges`, see below |
| `SEED_BASE` | 1 | `ranges`: the first seed of task 0 |
| `USE_MPS` | 0 | 1: `run_jobs.sh --mps` (test it on the site first) |
| `BINDS`, `EXTRA_ARGS` | empty | more binds; arguments for `run_prod_jet.py` |

- **Resources per task:** `#SBATCH --gres=gpu:1 --cpus-per-task=20 --mem=100G --time=24:00:00`
  in the script, for `P=4`. Each production job takes ~5 cores and ~22 GB of host memory,
  so match `P` to what one GPU comes with on the node. (22 GB is the size from before the
  memory fixes in X-SCAPE `cbc72639` / js-contrib `9356bf4`: a job alone now peaks at
  ~9 GB, but a `-j 4` campaign with the fixes is still to be measured; see
  docs/Plans/PLAN_slim_bulk_info.md.) The script sets
  `OMP_NUM_THREADS = cpus-per-task / P` and `OMP_WAIT_POLICY=passive` for the jobs.
- **Time limit:** choose `NJOBS × EVENTS` so a task fits `--time`: ~30–50 s per event and
  job, `P` jobs at once. A task that runs out of time resumes when you submit the same
  command again: it skips its finished jobs.
- **Seeds, `SEED_MODE=registry` (default):** every job draws a new seed (`FIRST_SEED 0`),
  checked against `WORK/CAMPAIGN/seeds_used.tsv`, which all tasks share under a file lock.
  Tasks on different nodes need that lock to hold across nodes. The script reads the mount
  options of `WORK`'s file system and stops on node-local or missing locks (Lustre
  `localflock` or `noflock`; NFS `nolock`, `local_lock=flock` or `local_lock=all`). It also
  stops if a test lock fails.
- **`SEED_MODE=ranges`:** task `t` runs seeds `SEED_BASE + t × NJOBS` to `… + NJOBS − 1`, so
  no lock is needed. Keep the ranges of different campaigns apart yourself, e.g.
  `SEED_BASE=100001` for the next one.
- **MPS:** off by default. Inside containers it is not assured (on the GB10 its daemon does
  not start in a container, see [Several GPUs](#several-gpus-on-one-machine)). Without it the
  jobs take turns on the GPU. If the site runs MPS itself (`--gres=mps`), leave
  `USE_MPS=0`.

Tested on the GB10 without SLURM: the script ran with the SLURM variables set by hand and a
stand-in for `apptainer` that ran the same container through Docker. Two array tasks of
2 × 1-event real jobs (`P=2`) completed and shared one registry (4 different seeds); a
resubmission skipped the finished jobs. The seed ranges, the lock checks (Lustre
`localflock`, NFS `local_lock=flock`, a failing lock) and the setting checks were tested
with stand-ins for `findmnt` and `flock`. **Not tested on a real SLURM cluster.**

### Cloud VMs: Docker

```bash
docker run --rm --gpus all -v "$PWD/prod:/work" jhputschke/xscape-prod:cu126 \
  python run_prod_jet.py --events 25 --seed 0 --outdir /work/out
```

The VM needs the NVIDIA driver and the NVIDIA container toolkit.

### Several GPUs on one machine

music4gpu always runs on the first CUDA device it sees (`cudaSetDevice(0)`), and
`run_jobs.sh` doesn't assign GPUs. So one campaign on a machine with several GPUs puts every
job on GPU 0; the other GPUs sit idle. Nothing fails; you only lose throughput. Run one
campaign per GPU instead, as in
[`prod_AuAu_0_10/README.md`](../contribs/PyJetscape/example/prod_AuAu_0_10/README.md). This
needs only environment variables, so the current images support it. **One container is
enough:**

```bash
# Docker: one container with all GPUs, one campaign per GPU inside it
docker run --rm --gpus all --user "$(id -u):$(id -g)" -v "$PWD/prod:/work" \
  jhputschke/xscape-prod:cu126 bash -c '
  trap "kill \$(jobs -p) 2>/dev/null; wait" INT TERM
  CUDA_VISIBLE_DEVICES=0 ./run_jobs.sh -j 4 --campaign gpu0 20 25 0 /work/out_gpu0 &
  CUDA_VISIBLE_DEVICES=1 ./run_jobs.sh -j 4 --campaign gpu1 20 25 0 /work/out_gpu1 &
  wait'

# Apptainer: the same inside one apptainer exec
PROD=/opt/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape/example/prod_AuAu_0_10_jet
apptainer exec --nv --pwd "$PROD" --bind "$SCRATCH/prod:/work" xscape_prod.sif bash -c '
  trap "kill \$(jobs -p) 2>/dev/null; wait" INT TERM
  CUDA_VISIBLE_DEVICES=0 ./run_jobs.sh -j 4 --campaign gpu0 20 25 0 /work/out_gpu0 &
  CUDA_VISIBLE_DEVICES=1 ./run_jobs.sh -j 4 --campaign gpu1 20 25 0 /work/out_gpu1 &
  wait'
```

- **`trap … INT TERM`** passes `docker stop` or Ctrl-C on to the campaigns, which then stop
  their jobs. Without it, `bash` ignores `docker stop` (as the container's first process it
  has no default signal handling), and the campaigns ignore Ctrl-C (background jobs of a
  script do). `docker stop` then waits its 10 s and kills everything. Either way the
  unfinished jobs rerun when you start the same command again.
- **`wait`** keeps the container alive until both campaigns end. Without it, the container
  exits right after starting them in the background.
- **Seeds:** `/work/out_gpu0` and `/work/out_gpu1` share the seed registry
  `/work/seeds_used.tsv`, so with seed `0` no two jobs draw the same seed, on either GPU.
- **`--campaign gpu0` / `gpu1`** gives the two directories different file names. Otherwise
  both campaigns, started in the same second, are named after the same start time.
- **Threads:** `OMP_NUM_THREADS` ≈ cores / all jobs (here 8), e.g. `docker run -e
  OMP_NUM_THREADS=…`, or set it inside the `bash -c`.
- **No `--mps`.** On the GB10 the MPS daemon does not start inside a container (below).
  And `--mps` in each per-GPU campaign left one GPU unused in a two-GPU test: each daemon
  sees only its campaign's GPU (see the `prod_AuAu_0_10` README). Without MPS, one campaign
  per GPU uses both GPUs (tested 2026-10-01, two-GPU machine, Docker).
- **SLURM with Apptainer:** `apptainer exec` passes the host's `CUDA_VISIBLE_DEVICES` into
  the container. If SLURM set it to e.g. `2,3`, use those numbers (`CUDA_VISIBLE_DEVICES=2`
  and `=3`). On nodes that show a job only its own GPUs it is usually `0,1` anyway.

**Two containers (Docker), one per GPU, work as well.** `--gpus device=N` shows each
container only its GPU, as GPU 0, so no `CUDA_VISIBLE_DEVICES` is needed. `run_jobs.sh` is
the container's first process here and handles `docker stop` itself. Mount the same `/work`
in both so they share the seed registry:

```bash
docker run -d --gpus device=0 --user "$(id -u):$(id -g)" -v "$PWD/prod:/work" jhputschke/xscape-prod:cu126 \
  ./run_jobs.sh -j 4 --campaign gpu0 20 25 0 /work/out_gpu0
docker run -d --gpus device=1 --user "$(id -u):$(id -g)" -v "$PWD/prod:/work" jhputschke/xscape-prod:cu126 \
  ./run_jobs.sh -j 4 --campaign gpu1 20 25 0 /work/out_gpu1
```

**Tested** 2026-09-30 on the GB10 (one GPU, so both campaigns on GPU 0) with the arm64
`cu130` image and a stand-in for `run_prod_jet.py` that only sleeps:
- **Both variants start and run their campaigns.**
- **Stopping:** `docker stop` ends the one-container variant in under 1 s with the `trap`,
  and after 10 s (killed) without it. `docker kill -s INT`, like Ctrl-C, works with the
  `trap` too. The two-container variant stops in about 1 s.
- **MPS in a container fails on the GB10.** `nvidia-cuda-mps-control` is present (the NVIDIA
  runtime mounts it from the host), but `-d` exits with code 33 and writes no log. It fails
  the same way in the plain `nvidia/cuda` base image, as root, with `--privileged`, and with
  `--ipc=host --pid=host`. On the host it starts. `run_jobs.sh --mps` then stops at the start
  with "could not start the MPS daemon", before any job runs. The cause is not known. It
  may be tied to the GB10's integrated GPU with unified CPU–GPU memory, where MPS runs in
  `-force-tegra` mode even on the host.
- **Still to test: MPS in a container on a system without unified memory** (a discrete GPU,
  e.g. an x86 node or an HPC GPU node), with Docker and with Apptainer. Until that is done,
  don't count on `--mps` inside the container. Check on each machine type with
  `nvidia-cuda-mps-control -d` in the container (exit code 0 and a `control.log` in
  `$CUDA_MPS_LOG_DIRECTORY`).

**Not tested yet:**
- **Several real GPUs:** none of these commands has run on a machine with more than one.
- **Real production jobs** in this setup (the test used the sleeping stand-in).
- **Apptainer:** not tested.
- **The seed registry's lock** (`fcntl`) holds on a local disk. Some network filesystems
  (NFS, Lustre) are mounted without lock support; check before many campaigns share one
  registry there.

A `run_jobs.sh --gpus 0,1,…` option that spreads one campaign over the GPUs is on the
js-contrib branch `run_jobs_gpus`, being tested. The images are built from js-contrib
`main`, so to try it in a container, build with `--build-arg JS_CONTRIB_REF=run_jobs_gpus`,
or bind-mount the branch's `contribs/PyJetscape/example/prod_AuAu_0_10/run_jobs.sh` over
the image's.

### A different collision system: your own user XML

The physics comes from the **user XML**, a template that `run_prod_jet.py` fills in (events,
seed, hard process). The image ships `AuAu_MCGlauber_MUSIC_0_10_jet.xml`; for another
system, mount a directory with your own XML and pass it with `--user-xml`. The path is the
one inside the container. Example for a `PbPb_0_10.xml` in `./xml`:

```bash
# Docker
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  -v "$PWD/xml:/xml:ro" -v "$PWD/prod:/work" jhputschke/xscape-prod:cu126 \
  python run_prod_jet.py --user-xml /xml/PbPb_0_10.xml \
    --events 25 --seed 1 --out PbPb_0_10_jet_seed0001.h5 --outdir /work/out

# Apptainer
PROD=/opt/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape/example/prod_AuAu_0_10_jet
apptainer exec --nv --pwd "$PROD" --bind "$PWD/xml:/xml,$SCRATCH/prod:/work" xscape_prod.sif \
  python run_prod_jet.py --user-xml /xml/PbPb_0_10.xml \
    --events 25 --seed 1 --out PbPb_0_10_jet_seed0001.h5 --outdir /work/out

# a campaign: run_jobs.sh passes the arguments after OUTDIR on to run_prod_jet.py
... jhputschke/xscape-prod:cu126 \
  ./run_jobs.sh -j 4 --campaign PbPb 20 25 0 /work/out --user-xml /xml/PbPb_0_10.xml
```

- **`--out`** sets the file name. Without it the files are called `AuAu_0_10_jet_…`, because
  the prefix is fixed in the scripts; in a campaign, `--campaign` at least puts the system
  into the names.
- **Add `--dry-run` first:** it checks the XML and the grid and writes the job XML without
  running an event.
- **A grid of your own** (`--grid /xml/grid_PbPb.yaml`) is mounted the same way.

**What to change for Pb+Pb** in a copy of `AuAu_MCGlauber_MUSIC_0_10_jet.xml`:

| where | Au+Au 200 GeV | Pb+Pb 5.02 TeV |
|---|---|---|
| `<MCGlauber>` `<projectile>`, `<target>` | `Au` | `Pb` |
| `<MCGlauber>` `<sqrts>` (what `MCGlauberWrapper` reads, not `mcglauber.input`) | `200` | `5020` |
| `<MCGlauber>` `<b_max>` for 0–10% | `4.7` fm | ~4.9–5.0 fm (geometric estimate) |
| Pythia's `<eCM>` | `200` | `5020` |
| the grid (`--grid` YAML; MUSIC's grid in the XML) | `grid_fno.yaml` | likely larger and longer: the fireball is bigger and lives longer |

`<cenMin>`/`<cenMax>` are ignored inside X-SCAPE (see the XML's comment); the impact
parameter range selects the centrality.

Tested 2026-09-30 in the `cu130` test image: the mount, the templating and the naming (a copy
of the Au+Au XML with `Pb` nuclei, `--dry-run`; the job XML has `Pb` as projectile and target,
and the output is named `PbPb_0_10_jet_seed0001.h5`). The Pb+Pb physics settings above are
not validated: a first real event will show whether the grid is large enough.

### Machines without an NVIDIA GPU (no CUDA)

The same image runs on machines without an NVIDIA GPU or driver, e.g. an amd64 cluster or
workstation with CPUs only. Start it without GPU access: `docker run` without `--gpus`,
`apptainer exec` without `--nv`. `pyjetscape_core` still imports; the CUDA libraries are
only needed when something calls CUDA.

**Hadronization runs there unchanged.** `hadronize.py` / `run_hadronize.py` use no GPU (iSS
on the stored surfaces, Pythia for the jet partons). So one set of images covers both halves
of a production: the hydro pairs with `--write-particlize both` on GPU nodes, the hadrons
later on CPU nodes, from the stored `*_particlize.h5` files.

```bash
PROD=/opt/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape/example/prod_AuAu_0_10_jet
apptainer exec --pwd "$PROD" --bind "$SCRATCH/prod:/work" xscape_prod.sif \
  python run_hadronize.py /work/out -j 4 --oversample 500 --n-frag 50
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/prod:/work" jhputschke/xscape-prod:cu126 \
  python hadronize.py /work/out/AuAu_0_10_jet_seed0001_particlize.h5 --oversample 500 --n-frag 50
```

**The hydro also runs, on the CPU, but slowly.** MUSIC4GPU finds no CUDA device ("No CUDA
device found: CUDA driver version is insufficient …") and falls back to its CPU path.
- It is about 10× slower: 285 s for one event on the GB10's 20 CPU cores, against ~30 s on
  its GPU.
- The results are not the GPU's: the CPU path's numerics differ (in the test event the jet
  leg froze out after 115 τ steps on the CPU and 105 on the GPU). Don't mix CPU- and
  GPU-produced events in one campaign without comparing them first.
- Set `OMP_WAIT_POLICY=passive` for CPU runs (`-e OMP_WAIT_POLICY=passive`,
  `--env OMP_WAIT_POLICY=passive`): MUSIC warns that with a GCC build idle OpenMP threads
  busy-spin otherwise.
- The fallback message says "Metal init failed, falling back to CPU" although this is a CUDA
  build; only the wording is wrong.

Tested 2026-09-30 with the arm64 `cu130` image on the GB10, started without `--gpus` (no
NVIDIA driver or GPU in the container). The amd64 image builds the same code, but hasn't been
run on a CPU-only amd64 machine yet.
- `hadronize.py` on the regression run's particlize file (2 events, 20 oversamples,
  5 fragmentations): 6.6 s, and the hadron files are **bit-identical** to those of the native
  `build_gpu` (same hadrons, momenta and pids in `bulk_jet`, `bulk_bg` and `jet_frag`).
- `run_prod_jet.py --events 1 --seed 1`: the CPU fallback above, 285 s, a complete pair file.

### Getting the outputs home

Copy `OUTDIR` to Google Cloud Storage or a Pelican/OSDF namespace.
- **From the host,** where `OUTDIR` is bind-mounted: with the command-line tools
  (`gcloud storage cp -r`, `pelican object put`) or the Python interfaces of the analysis
  environment
  ([`analysis_env/README.md`](analysis_env/README.md#remote-files-google-cloud-storage-and-pelicanosdf)).
- **From inside the container,** where the host has no tools, e.g. an OSPool job. On the
  OSPool, HTCondor can also do the transfer itself, with `osdf://` URLs in the submit file.

**Pelican/OSDF**, in every image:

```bash
pelican object put -r /work/out osdf:///NAMESPACE/campaign     # the whole OUTDIR
pelican object put -t /work/token /work/out/FILE.h5 osdf:///NAMESPACE/campaign/FILE.h5
pelican object get osdf:///NAMESPACE/inputs/PbPb_0_10.xml /work/xml/
```

or from Python, through fsspec:

```python
import fsspec
fs = fsspec.filesystem("osdf")
fs.put("/work/out/", "/NAMESPACE/campaign/", recursive=True)
```

**Google Cloud Storage**, in the `-gcs` images, from Python only (no `gcloud`):

```python
import fsspec
fs = fsspec.filesystem("gs")                          # Google's default credentials
fs.put("/work/out/", "BUCKET/campaign/", recursive=True)
```

**Credentials.** Pass them into the container.
- **Pelican:** protected namespaces take a bearer token. Use `-t FILE` for the CLI; for
  pelicanfs, `BEARER_TOKEN_FILE` (e.g. `docker run -e BEARER_TOKEN_FILE=/work/token`), or
  one of the other places listed in the analysis README. On the OSPool, HTCondor's
  credentials are found as well.
- **GCS:** mount a service-account key and point `GOOGLE_APPLICATION_CREDENTIALS` at it
  (`-v key.json:/key.json:ro -e GOOGLE_APPLICATION_CREDENTIALS=/key.json`). On a Google
  Cloud VM, the VM's own service account works without a key.
- **Public data** needs none.

**Tested:** see [Tested](#tested). **Not tested:** writing (`put`) and credentials; only
public reads were tried.

---

## Validation

Before publishing an image (Option B on the GB10), and once on every new cluster or GPU type:

1. **Starts:** `nvidia-smi`; `python -c "import jetscape; assert jetscape.HAS_CORE"`;
   `cat /opt/X-SCAPE/BUILD_INFO.txt`.
2. **Configuration:** `python run_prod_jet.py --events 1 --seed 1 --dry-run`.
3. **Null test:** `--no-deposit`: the jet leg must equal the background leg exactly
   (`arr == arr_bg`).
4. **Regression against the GB10 baseline:** seed 1 against a native `build_gpu` run at the
   same commits, with the production regression recipe
   (`run_prod_jet.py --events 2 --seed 1 --reuse 2 --write-particlize both --seed-registry none`).
   - **Not bit-identical, even on the GB10:** the image's toolchain is not `build_gpu`'s
     (nvcc 13.2 vs 13.0, conda GCC 14.4 vs 14.3, other conda builds), and the results differ
     from the initial state on at floating-point level. Compare within tolerances: see
     [Tested](#tested) for what the GB10 image gives, and treat much larger differences as a
     finding.
   - On other GPU types expect differences of the same kind (`--use_fast_math`, other
     hardware).
5. **Throughput:** events per hour for one job and for `-j` jobs per GPU, recorded next to
   the GB10 and M3 Max numbers.

`MUSIC_FORCE_CPU=1` runs MUSIC4GPU on the CPU inside the same image: a slow but useful check
when a GPU result looks wrong.

### Tested

2026-09-30 on the GB10, arm64 only (amd64 is left to GitHub Actions): `cu130`
(`CUDA_VERSION=13.2.1`, `CUDA_ARCHITECTURES="90-real;100-real;120-real;121"`), X-SCAPE
`d31946c0`, MUSIC4GPU `15ec5e3`, js-contrib `77a23cd`, without ROOT. (The regression run below
used a build from X-SCAPE `8c306762`, which has the same tree as `d31946c0`.)
- **Build:** 3 min 5 s on the GB10's 20 cores with only the conda layers cached (fetching the
  sources and tables, LHAPDF, and compiling X-SCAPE for four GPU architectures). In the image,
  `nvidia-smi` sees the GB10 and `run_prod_jet.py --dry-run` configures the production.
  Image 4.7 GB. The runtime env keeps 73 of the build env's 91 conda packages.
  `pyjetscape_core` imports, `HAS_ROOT = False`.
- **Run:** the regression recipe above in the container (`docker run --gpus all --user …`) and
  natively with `build_gpu`: 61.0 s vs 63.6 s.
- **Comparison, container vs native:**

  | quantity | agreement |
  |---|---|
  | freeze-out steps and τ | identical (jet 105/96, background 95; τ = 11.0/10.1 fm/c) |
  | shower | same topology (178 partons); momenta within 10⁻⁶ GeV; total energy equal to 10⁻⁹ |
  | droplets | same 42; total energy equal to 3×10⁻¹⁰ |
  | hydro e, both legs | first frame within 1.9×10⁻⁹ GeV/fm³; any cell within 6×10⁻⁴ of the frame's maximum; per-frame total within 2.5×10⁻⁶ |
  | wake (jet − background) | within 1.1×10⁻³ of the maximum wake; its total within 10⁻⁴ |
  | freeze-out surfaces | jet 1 999 270 cells in both; background 989 692 vs 989 690 |

**Remote storage**, tested 2026-09-30 on the GB10 (arm64 `cu130`, js-contrib branch
`container_remote_storage`), built as `WITH_GCS=0` and `WITH_GCS=1`:
- **Build:** 3 min 4 s for the default image (stage 1 not cached); the `WITH_GCS=1` build
  then reused everything but the remote-storage layer and took 15 s. Sizes: 4.66 GB before,
  4.74 GB default (+80 MB: the `pelican` CLI, pelicanfs and its dependencies), 4.78 GB with
  GCS.
- **Versions:** `pelican` 7.26.2 (checksum verified), pelicanfs 1.4.1, fsspec 2026.9.0; with
  GCS also gcsfs 2026.8.1 and google-cloud-storage 3.15.1. No package already in the image
  changed version (the conda env lists only the new ones as different).
- **Public reads,** as a user without a home directory (`--user 12345:12345`, `HOME=/`), as
  on HPC: an OSDF object with `pelican object get` and with pelicanfs, and a directory listing
  with pelicanfs. With GCS: a public bucket listed anonymously with gcsfs and with
  `google-cloud-storage`.
- **Production:** `run_prod_jet.py --events 1 --seed 1` in the new default image completes
  (39 s).
- **Not tested:** writing (`pelican object put`, `fs.put`) and credentials (tokens, service
  accounts), and the amd64 images (GitHub Actions).

---

## Open points

- **ROOT:** settled: the image builds and runs without it (see [ROOT](#root), [Tested](#tested)).
  `WITH_ROOT=1` is not tested yet.
- **Validation steps not run yet:** the null test (`--no-deposit`), hadronization inside the
  image *with* a GPU attached (without one it is tested, see
  [Machines without an NVIDIA GPU](#machines-without-an-nvidia-gpu-no-cuda)), throughput with
  several jobs, Apptainer, the amd64 images on an amd64 machine, and
  [several GPUs](#several-gpus-on-one-machine) (one campaign per GPU), and **MPS inside a
  container on a system without unified memory** (discrete GPU). On the GB10 (integrated
  GPU, unified memory) its daemon does not start in a container, so this has to be tested
  on a non-unified system before `--mps` goes into container campaigns.
- **`OMP_WAIT_POLICY=passive`** could be set in the image (`ENV`) instead of per run: MUSIC
  asks for it on the CPU path (the M3 Max campaign settings in the production README set it
  too; the GB10's only set `OMP_NUM_THREADS`).
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
  `Dockerfile.dev.blackwell` and `docs/Plans/PlanContainerDev.md`.
