# Running the two-stage production with the containers

This guide runs `prod_AuAu_0_10_jet` from start to finish **with the production container**,
on a workstation or cloud VM with Docker, or on an HPC cluster with Apptainer and SLURM. The
production has two stages, and the analysis comes after them:

```
 Stage 1, GPU nodes (container)        Stage 2, CPU nodes (same container)    Analysis (local venv)
 ──────────────────────────────        ───────────────────────────────────    ─────────────────────
 run_jobs.sh → run_prod_jet.py         run_hadronize.py → hadronize.py        jetscape.hadrons_h5 readers,
   MUSIC background + jet leg            iSS on the stored surfaces,          notebooks, analysis scripts
   Matter/LBT, liquefier                 Pythia fragmentation of the partons  run_h5toROOT.py → ROOT files
   --write-particlize both                                                     for existing experiment code
        │                                       │
        ▼                                       ▼
 <stem>.h5             hydro pair       <stem>_hadrons_bulk_jet.h5
 <stem>_particlize.h5  surfaces+partons <stem>_hadrons_bulk_bg.h5
                                        <stem>_hadrons_jet_frag.h5
```

- **Stage 1** needs a GPU. It writes the hydro pair (jet leg and background) and, with
  `--write-particlize both`, the **input for hadronization**: both freeze-out surfaces, with
  the full viscous information, and the final partons.
- **Stage 2** needs no GPU. Once the surfaces are stored, the **same container** turns them
  into hadrons on CPU nodes, whenever and wherever convenient, as often as wanted (more
  oversamples, other cuts).
- **Analysis** needs neither X-SCAPE nor the container: a Python venv reads every file and
  converts the hadrons to ROOT for analysis code that expects ROOT.
- **Between them**, the files can travel through Google Cloud Storage or the OSDF with the
  tools in [`utils/remote_transfer`](../utils/remote_transfer/README.md)
  ([§5](#5-moving-the-files)).

The physics and every option of the production are in [`prod_AuAu_0_10_jet/README.md`](../contribs/PyJetscape/example/prod_AuAu_0_10_jet/README.md); the images
themselves (variants, building, testing) in
[`utils/BuildContainerProd.md`](../utils/BuildContainerProd.md).

---

## 1. Requirements

### Images

> **Rebuild pending (status 2026-10-06).** The images on Docker Hub predate several changes
> this guide describes; a rebuild of all three variants from current X-SCAPE `contrib` and
> js-contrib `main` (`gh workflow run docker-prod.yml -f variants=all`, see
> [`utils/BuildContainerProd.md`](../utils/BuildContainerProd.md)) still needs to be done.
> What the published tags contain:
>
> | tag | built | sm_70 (V100) | Pelican | memory fixes | 64×64×32 grid, edge flag |
> |---|---|---|---|---|---|
> | `cu126`, `cu130` (= `…-20260930-d31946c`) | 2026-09-30 | **no** | **no** | **no** (~20 GB per job) | no |
> | `cu124-20260930-d31946c` | 2026-09-30 | yes | yes | **no** | no |
> | `cu124` (= `cu124-20261003-cbc7263`) | 2026-10-03 | yes | yes | yes | no |
>
> Until the rebuild: on V100s or with tight memory, only `cu124-20261003-cbc7263` (amd64)
> matches the requirements below. No image has the new default output grid or the edge flag
> yet ([Output grid](#output-grid)); with an older image, a campaign writes the earlier
> 65 × 65 × 33 grid unless it is given the new one with `--grid`. Update the dated tags in
> this guide after the rebuild.

| tag | CUDA | GPUs | CPU arch | host driver |
|---|---|---|---|---|
| `jhputschke/xscape-prod:cu126` | 12.6 | V100 (sm_70) … H100/GH200 (sm_90), newer through PTX | amd64, arm64 | R560+ |
| `jhputschke/xscape-prod:cu124` | 12.4 | as `cu126` | amd64 | R550+ |
| `jhputschke/xscape-prod:cu130` | 13.2 | GH200 (sm_90) … B200/GB200, RTX 50xx, GB10 (sm_121); **no V100** | amd64, arm64 | R580+ |

- **Which one:** `nvidia-smi` shows the highest CUDA version the driver supports. Use
  `cu126` where it is ≥ 12.6, `cu124` on R550 drivers, `cu130` for Blackwell. There is no
  performance difference to expect between `cu124` and `cu126`.
- **For a campaign, use a dated tag**, e.g. `cu124-20261003-cbc7263`, and the same one on
  every machine: the moving tags (`cu126`) change with every build, and different builds
  can differ in the last digits. Run the whole campaign on GPUs: MUSIC's CPU path differs
  systematically from the GPU path (~0.2% in multiplicity,
  [`MUSIC_CPU_vs_GPU.md`](MUSIC_CPU_vs_GPU.md)), so don't mix the two.
- **`-gcs` tags** (`cu126-gcs`, …; none published yet, the workflow input `with_gcs=true`
  builds them) also have Google Cloud Storage for Python. Every image
  built since Pelican was added (see below; not the published `cu126`/`cu130`) has
  Pelican/OSDF (`pelican` CLI, `pelicanfs`).
- **What's in an image:** `/opt/X-SCAPE/BUILD_INFO.txt` (commits, CUDA, Pelican) and
  `docker inspect` labels. Which commits bring what:

  | feature | needs |
  |---|---|
  | sm_70 (V100), Pelican | an image built after 2026-09-30 19:57 EDT (js-contrib `818fdbf`); the `cu126`/`cu130` builds of 2026-09-30 have neither |
  | memory fixes (~9 GB per job) | X-SCAPE `contrib` `cbc72639` or later |
  | GPU fix for stalled evolutions ([`VacReset_BUG.md`](https://github.com/jhputschke/MUSIC4GPU/blob/XSCAPE/docs/VacReset_BUG.md)) | X-SCAPE `9509aea2` or later (in every published image) |
  | edge flag (`diag/{bg,jet}_hit_edge`) | js-contrib `main` `20109af` or later |
  | 64 × 64 × 32 output grid by default | js-contrib `main` `c8b682d` or later |

### Memory, cores, disk

| per production job (stage 1) | |
|---|---|
| **host memory** | **~9 GB peak** per job alone (images with X-SCAPE `cbc72639` or later; older ones ~20 GB): 8.7 GiB with the pair file, 7.5 GiB with `--particlize-only` (GB10, seed 1; X-SCAPE `contrib` from `cbc72639`, js-contrib `main` from `9356bf4`). It was ~20 GB before the memory fixes and is 14.3 GiB with `--bulk-info full` ([PLAN_slim_bulk_info.md](Plans/PLAN_slim_bulk_info.md)). Most of it is the background's hydro history, kept in memory for Matter/LBT, now as a slim 6-field copy. Several jobs: **4 at once used 29 GB** on the GB10 (66–71 GB before the fixes; [BENCHMARK_GB10.md](BENCHMARK_GB10.md#with-the-memory-fixes-2026-10-03)). **Plan for ~12 GB per job** (~16 GB with `--bulk-info full`). |
| GPU memory | a few hundred MB: a 16 GB V100 is plenty |
| CPU cores | ~5 per job (`OMP_NUM_THREADS=5`) |
| time | ~30–50 s per event and job, depending on GPU and CPU |
| disk | ~285 MB per event (hydro pair) + ~154 MB (`--write-particlize both`), measured on the earlier 65 × 65 × 33 output grid; the 64 × 64 × 32 default has 6% fewer cells per frame |

So **host RAM per GPU usually decides how many jobs share a GPU**: with `-j 4` per GPU, ask
for ~48 GB and ~20 cores per GPU (~12 GB per job; 4 jobs used 29 GB on the GB10), ~64 GB with
`--bulk-info full`, and ~100 GB with images that predate the memory fixes (§1). Lower `-j`
where a node gives less.

| per hadronization process (stage 2) | |
|---|---|
| memory | ~1.4 GB per surface up to ~1000 oversamples |
| time | ~10–14 s per event, both legs, 500 oversamples, 50 fragmentations |
| disk | ~100 MB per leg and event at 500 oversamples (less with the options in §3) |

### Output grid

The hydro pair is stored on the output grid of `../prod_AuAu_0_10/grid_fno.yaml`
(`run_prod_jet.py --grid` for another one). Since js-contrib `c8b682d` (2026-10-06) the default
is **64 × 64 × 32 cells**: x, y −12.2 … 12.2 fm at 0.3875 fm, η_s −4.84 … 4.84 at 0.3125.
Before, it was 65 × 65 × 33 (x, y −10 … 10 fm, all steps 0.3125), as in `fastdata_AuAu200_tune_0_10.h5`.

- **Files of the two grids don't go into one training set.** Keep one grid per campaign, and
  check `prod_grid_yaml` in a file's attributes when combining campaigns.
- **Why the change:** the jet leg's fluid above freeze-out crossed ±10 fm in ~5% of 0–10%
  events; ±12.2 fm holds it. How much of the wake each grid keeps:
  [`Wake_grid_comparison.md`](Wake_grid_comparison.md).
- **With an image that predates the change** (all published images, see above), bind the new
  YAML from a js-contrib checkout (`contribs/PyJetscape/example/prod_AuAu_0_10/grid_fno.yaml`)
  and pass it after OUTDIR: Docker `-v "$PWD/grid_fno.yaml:/grid/grid_fno.yaml:ro"`, Apptainer
  `--bind "$PWD/grid_fno.yaml:/grid/grid_fno.yaml"`, then `… --grid /grid/grid_fno.yaml`.

---

## 2. Stage 1: hydro + jets on GPUs

### The command inside the container: `run_jobs.sh`

Everything runs `run_jobs.sh` from the production folder
`/opt/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape/example/prod_AuAu_0_10_jet`:

```
./run_jobs.sh [-j P] [--mps] [--campaign NAME] NJOBS EVENTS_PER_JOB FIRST_SEED OUTDIR [run_prod_jet.py options]
```

| | |
|---|---|
| `NJOBS` | number of jobs = number of output files |
| `EVENTS_PER_JOB` | events in each job. **Total events = NJOBS × EVENTS_PER_JOB** |
| `-j P` | at most P jobs at the same time (default 1). It doesn't create jobs: `-j 2 1 1 5` runs one job |
| `FIRST_SEED` | `0`: every job draws a new seed, never repeated (a **campaign**; the usual choice). `> 0`: seeds FIRST_SEED, FIRST_SEED+1, … (reproducing, comparing settings) |
| `--campaign NAME` | names the files `AuAu_0_10_jet_NAME_0001.h5`, …; without it, the start time |
| `OUTDIR` | the output directory: **the argument after FIRST_SEED, not `--outdir`**. One campaign per OUTDIR |
| after OUTDIR | options for `run_prod_jet.py`, e.g. `--write-particlize both`, `--reuse 3`, `--user-xml /xml/PbPb.xml` |

- **The seed registry** `seeds_used.tsv` is written **next to** OUTDIR (in its parent), for
  every job. So OUTDIR's parent must be writable: use `/work/out`, not `/work`.
  `run_jobs.sh` checks this before it starts.
- **Restarting:** run the same command again. Finished jobs are skipped; unfinished ones
  are redone.
- **Hadronization input:** add **`--write-particlize both`** to every stage-1 command that
  should later get hadrons.

### Docker (workstation, cloud VM)

```bash
mkdir -p ~/prod
docker pull jhputschke/xscape-prod:cu126-<YYYYMMDD>-<xscape7>
IMG=jhputschke/xscape-prod:cu126-<YYYYMMDD>-<xscape7>

# a test: 2 jobs x 1 event, both at once, seeds 1 and 2
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  -e OMP_NUM_THREADS=4 -e OMP_WAIT_POLICY=passive \
  -v "$HOME/prod:/work" $IMG \
  ./run_jobs.sh -j 2 2 1 1 /work/test --write-particlize both

# a campaign: 20 jobs x 25 events, 4 at a time, new seeds, in the background
docker run -d --name xscape --gpus all --user "$(id -u):$(id -g)" \
  -e OMP_NUM_THREADS=5 -e OMP_WAIT_POLICY=passive \
  -v "$HOME/prod:/work" $IMG \
  ./run_jobs.sh -j 4 --campaign AuAu_a 20 25 0 /work/AuAu_a --write-particlize both
docker logs -f xscape          # progress;  docker stop xscape  stops it cleanly
```

- Docker starts in the production folder, so `./run_jobs.sh` works as is.
- `--user` makes the files yours; `-v` binds a host directory to `/work`.
- **Own XML** (another collision system): bind it and pass `--user-xml`:
  `-v "$PWD/xml:/xml:ro" … ./run_jobs.sh … /work/PbPb --user-xml /xml/PbPb_0_10.xml`
  (see [`prod_AuAu_0_10_jet/README.md`](../contribs/PyJetscape/example/prod_AuAu_0_10_jet/README.md) and `BuildContainerProd.md`, *A different collision system*).

### Apptainer (HPC)

```bash
apptainer pull xscape_prod.sif docker://jhputschke/xscape-prod:cu126-<YYYYMMDD>-<xscape7>
PROD=/opt/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape/example/prod_AuAu_0_10_jet
mkdir -p $SCRATCH/prod

apptainer exec --nv \
  --env OMP_NUM_THREADS=4,OMP_WAIT_POLICY=passive \
  --pwd "$PROD" --bind "$SCRATCH/prod:/work" xscape_prod.sif \
  ./run_jobs.sh -j 2 2 1 1 /work/test --write-particlize both
```

Three things differ from Docker:
- **`--pwd "$PROD"`:** Apptainer ignores the image's working directory and starts where you
  launched it, usually your home. Without `--pwd`, `./run_jobs.sh` is not found and `./`
  means your home.
- **Binding:** the image is read-only; only bound directories (and your home) are writable.
  The bound source directory must exist (`mkdir -p` first).
- **Your home directory** is mounted, but not its parent: OUTDIR `~` puts the seed registry
  into the read-only image ("Read-only file system"). Use `~/prod/out`, or better a bound
  scratch directory.

Environment variables can also come from your shell (`export OMP_NUM_THREADS=4` before
`apptainer exec`), unless `--cleanenv` is used; `--env` always works.

### SLURM: one GPU per array task

[`utils/slurm_prod_array.sh`](../utils/slurm_prod_array.sh) runs a campaign as a job
array: each task gets one GPU and runs `P` jobs on it (`run_jobs.sh -j P`).

```bash
SIF=$PWD/xscape_prod.sif CAMPAIGN=AuAu_a EXTRA_ARGS="--write-particlize both" \
  sbatch utils/slurm_prod_array.sh                    # 10 tasks = 10 GPUs
SIF=$PWD/xscape_prod.sif CAMPAIGN=AuAu_a EXTRA_ARGS="--write-particlize both" \
  NJOBS=2 EVENTS=1 sbatch --array=0-1 --time=1:00:00 utils/slurm_prod_array.sh   # a trial
```

- **Why not one SLURM job per production job:** SLURM gives whole GPUs, and one job keeps
  a GPU only partly busy (most of an event is CPU work). Only jobs in one allocation can
  share a GPU.
- **Size it:** the script asks for `--gres=gpu:1 --cpus-per-task=20 --mem=48G` for `P=4`.
  Match `P`, cores and memory to one GPU's share of your nodes (~5 cores, **~12 GB per
  job**; 4 jobs used 29 GB on the GB10; ask for `--mem=64G` with `--bulk-info full`). It
  sets `OMP_NUM_THREADS = cpus / P` and `OMP_WAIT_POLICY=passive` itself.
- **Output:** `WORK/CAMPAIGN/t000/`, `t001/`, … with one shared seed registry. The script
  checks that the file system's locks work across nodes, or use `SEED_MODE=ranges`.
- **Restarting:** submit the same command again.
- All settings: `BuildContainerProd.md`, *SLURM: one GPU per array task*.

### Several GPUs in one machine or allocation

MUSIC always uses the first GPU it sees. Run **one campaign per GPU**, each with its own
`CUDA_VISIBLE_DEVICES`, OUTDIR and campaign name; the OUTDIRs side by side share the seed
registry:

```bash
# Docker: one container per GPU (each sees its GPU as GPU 0)
for g in 0 1; do
  docker run -d --name xscape_gpu$g --gpus device=$g --user "$(id -u):$(id -g)" \
    -e OMP_NUM_THREADS=5 -e OMP_WAIT_POLICY=passive -v "$HOME/prod:/work" $IMG \
    ./run_jobs.sh -j 4 --campaign AuAu_gpu$g 20 25 0 /work/AuAu/gpu$g --write-particlize both
done

# Apptainer / one container with all GPUs
apptainer exec --nv --env OMP_NUM_THREADS=5,OMP_WAIT_POLICY=passive \
  --pwd "$PROD" --bind "$SCRATCH/prod:/work" xscape_prod.sif bash -c '
  trap "kill \$(jobs -p) 2>/dev/null; wait" INT TERM
  CUDA_VISIBLE_DEVICES=0 ./run_jobs.sh -j 4 --campaign AuAu_gpu0 20 25 0 /work/AuAu/gpu0 --write-particlize both &
  CUDA_VISIBLE_DEVICES=1 ./run_jobs.sh -j 4 --campaign AuAu_gpu1 20 25 0 /work/AuAu/gpu1 --write-particlize both &
  wait'
```

The `trap` passes Ctrl-C and `docker stop` on to both campaigns; `wait` keeps the container
alive until both are done.

**Ready-made for Docker on two GPUs:** [`utils/launch_2gpu.sh`](../utils/launch_2gpu.sh) does the
Docker variant above, with `run_jobs.sh -j 8 --mps` per GPU, a disk guard that stops the
campaign before the disk fills (`--min-free`), and optionally the upload of every finished job
to the OSDF while it runs (`--upload`, below in §5). Measured on 2 × RTX 3090: ~810 events/h
with the c1 settings of §4. Usage, tuning and the measurements:
[`utils/README_launch.md`](../utils/README_launch.md). Under SLURM, if `CUDA_VISIBLE_DEVICES` is already set (e.g. `2,3`),
use those numbers.

**Tested** (2026-10-01): one campaign per GPU on a two-GPU machine with Docker, without MPS:
both GPUs busy. With `--mps` in each campaign only one GPU was used then (see *CUDA MPS*
below); `launch_2gpu.sh` (2026-10-05) runs both GPUs with MPS, one Docker container and one
daemon per GPU.
A `run_jobs.sh --gpus 0,1` option that spreads one campaign over the GPUs is on js-contrib
branch `run_jobs_gpus`, not merged yet; until then, use one campaign per GPU as above.

### CUDA MPS (optional)

`--mps` lets the jobs on one GPU run their kernels side by side instead of taking turns. It
paid off only with several jobs per GPU and a busy GPU (GB10: +2 % at `-j 2`, +12.5 % at
`-j 4`). **Run without it first** and look at the GPU use (`nvidia-smi dmon -s u`): under
~60 % it won't help.

- **Inside containers it is not assured:** on the GB10 the daemon does not start in a
  container. Test with `nvidia-cuda-mps-control -d; echo $?` inside the container.
- **Several GPUs:** one Docker container per GPU (`--gpus device=N`), each with its own
  campaign and `--mps`, works: `launch_2gpu.sh` runs both GPUs of a 2 × RTX 3090 machine that
  way (2026-10-05). Two daemons in **one** container hang the second GPU's jobs, and an
  earlier two-GPU test with `--mps` in each campaign had only one GPU busy (2026-10-01; its
  jobs fell back to the CPU, `No CUDA device found`). So with Apptainer, or one container
  for all GPUs, run without MPS, or start one daemon over all GPUs yourself and run the
  campaigns without `--mps`.
- If the site runs MPS itself (`--gres=mps`), don't add `--mps`.

### Checking a run

In a job log, `OUTDIR/<stem>.log`:

| line | means |
|---|---|
| `[MUSIC-GPU] CUDA device: <GPU name> (cc X.Y, …)` | the hydro runs on that GPU |
| `[MUSIC-GPU] OMP wait policy: passive (max threads N)` | the thread settings arrived |
| `… done in … s` per event | the speed: events/h = jobs × 3600 / s per event |
| `No CUDA device found`, `falling back to CPU` | **no GPU**: the job runs on the CPU, ~7–10× slower, and its results differ systematically from the GPU path's (~0.2%, [`MUSIC_CPU_vs_GPU.md`](MUSIC_CPU_vs_GPU.md)). **Don't add that output to a GPU campaign.** Check `--nv`/`--gpus`, `CUDA_VISIBLE_DEVICES`, MPS |
| `edge     flag e > 0.2342 GeV/fm^3 on the x/y edge of the output grid` (at the start) | the edge flag is on, with MUSIC's freeze-out energy density as threshold (images with js-contrib `20109af` or later) |
| `PairH5Writer: event i: the jet leg has e = … on the transverse edge of the output grid` | fluid above freeze-out reached the edge of the stored box: the stored evolution misses part of it (the surfaces and hadrons don't). Flagged in `diag/{bg,jet}_hit_edge`; counted in the `.json` (`events_hit_edge`). A few per campaign are expected on the earlier ±10 fm grid; frequent ones mean the grid is too small |
| `… the freeze-out surface reached the transverse grid boundary, so MUSIC stopped it early` | MUSIC's own grid (`<IS><grid_max_x>`, ±15 fm) is too small: that event's evolution is truncated (`diag/{bg,jet}_hit_boundary`) |
| `kernel launch '…' failed: no kernel image is available` | the image has no code for this GPU (e.g. a V100 with an image without sm_70, which includes the published `cu126` until the rebuild, §1). **Discard the output** and use an image with sm_70 (`cu124-20261003-cbc7263` for now) |
| `WARNING: NOT in effect … export OMP_WAIT_POLICY=passive` | set `OMP_WAIT_POLICY=passive` before launching |

Each job also writes `<stem>.json` (seed, events written, settings, `events_hit_edge`), and
the campaign ends with `OUTDIR/run_jobs.finished`.

---

## 3. Stage 2: hadronization on CPUs, with the same container

Once a job's `<stem>_particlize.h5` is complete, its hadrons can be made **on any CPU
machine, with the same image, without a GPU**: start it without `--gpus`/`--nv`. iSS samples
the stored surfaces (the full freeze-out information MUSIC handed over, viscous
corrections included); Pythia fragments the stored partons. The result is the same as on
the GPU machine (checked bit for bit against the native build).

```bash
# Docker
docker run --rm --user "$(id -u):$(id -g)" -e OMP_NUM_THREADS=2 \
  -v "$HOME/prod:/work" $IMG \
  ./run_hadronize.py /work/AuAu_a -j 8 --oversample 500 --n-frag 50

# Apptainer
apptainer exec --env OMP_NUM_THREADS=2 --pwd "$PROD" --bind "$SCRATCH/prod:/work" \
  xscape_prod.sif ./run_hadronize.py /work/AuAu_a -j 8 --oversample 500 --n-frag 50
```

`run_hadronize.py` hadronizes every complete particlize file of the given directories (or
files, or globs), `-j` at a time, and skips files whose hadrons are already complete:

| option | |
|---|---|
| `-j P` | processes at once; ~1.4 GB and ~1–2 cores each |
| `--oversample N` | iSS samples per surface (default 100 from `hadronize.xml`) |
| `--n-frag M` | Pythia fragmentations of the partons |
| `--oversample-bg auto` | for `--reuse` runs: a background shared by N events gets N × the samples |
| `--correlated` | correlated jet/background sampling: jet − background ~8× less noisy for the same oversamples |
| `--follow` | keep running while stage 1 is still writing; stops when the campaign is finished |
| `--eta-max X`, `--charged`, `--no-x`, `--keep-bits-p B`, `--keep-bits-x B` | store less: \|η\| cut, charged only, no positions, rounded momenta/positions |
| `--dry-run` | show the plan only |

Output per production file, next to it: `<stem>_hadrons_bulk_jet.h5` (iSS on the jet surface:
bulk + wake), `<stem>_hadrons_bulk_bg.h5` (iSS on the background), `<stem>_hadrons_jet_frag.h5`
(the fragmented partons). Restart with the same command; incomplete outputs are redone. Only
the particlize files are needed: copy just those to the CPU cluster (§5).

**On a SLURM CPU partition:**

```bash
#!/bin/bash
#SBATCH --cpus-per-task=16 --mem=32G --time=12:00:00
PROD=/opt/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape/example/prod_AuAu_0_10_jet
apptainer exec --env OMP_NUM_THREADS=2 --pwd "$PROD" --bind "$SCRATCH/prod:/work" \
  xscape_prod.sif ./run_hadronize.py /work/AuAu_a/t000 /work/AuAu_a/t001 -j 8 \
  --oversample 500 --n-frag 50
```

Three or four hadronization processes keep up with a four-job GPU campaign, so stage 2 can
also run next to stage 1 with `--follow`.

---

## 4. A realistic campaign: pT̂ windows, several jets per background

The examples above give every event its own background. For a real production, as on an
HPC cluster, **share each background among several jets in several pT̂ windows**: the
background leg (MUSIC_1) doesn't depend on the jet, and it costs about as much as the jet
leg. This is what campaign `pth10-40_eta06_c1` did on the GB10 (2026-09-29):

| | c1 |
|---|---|
| pT̂ windows | 10–20, 20–30, 30–40 GeV: `--pthat-bins 10-20,20-30,30-40` (K = 3) |
| jets per window per background | 5: `--jets-per-bin 5` (M = 5), so one background per K × M = **15 events** (`--reuse 15`, set automatically) |
| rapidity cut | hardest parton \|y\| < 0.6: `--parton-ymax 0.6` (events outside are regenerated before any shower or hydro) |
| jobs | 67 × 15 events = **1005 events on 67 backgrounds**, `-j 4 --mps`, seeds from OS entropy |
| time | 179.7 min on one GB10: **336 events/h** (a new background every event: ~190, measured at 50–70 GeV) |
| disk | 170 MB per event (pair, shared `arr_bg`) + 94 MB (particlize): **265 GB** (on the earlier 65 × 65 × 33 grid) |
| checks | energy balance closed, 0 double-counted partons, all legs froze out, no grid-boundary hits |

**The rules.**
- **Events per job must be a multiple of K × M** (here 15): event i uses window i mod K,
  and every background gets exactly M jets in each window. So `EVENTS_PER_JOB` is 15, 30, …
- **Cost per event** ≈ (1 + 1/(K·M)) / 2 of an event with its own background: 0.53 for
  K·M = 15. The GB10 measured 1.8× the throughput.
- **Memory** is no more than without reuse (7.5 GiB peak measured for `--reuse 2`, one job
  alone): one background is held at a time, just longer.
- **Cross sections:** each window is its own Pythia with its own σ, recorded per file;
  combine windows with those weights (§6).
- **Statistics:** the M jets of one window share their background (like `--reuse M` per
  window), and spectra stitched over windows have errors correlated through the
  backgrounds. Per-window results and jet − background are unaffected.
- **The cut** `--parton-ymax 0.6` (mode `leading`) is complete for jets up to
  \|η_jet\| ≈ 0.4–0.5. For inclusive R = 0.4 jets at \|η_jet\| < 0.6, use
  `--parton-ymax 0.7 --parton-y-mode any`. Details: [`prod_AuAu_0_10_jet/README.md`](../contribs/PyJetscape/example/prod_AuAu_0_10_jet/README.md),
  sections *D. Several pTHat windows per background* and *E. Parton rapidity cut*.

### Stage 1

**On one machine** (Docker; Apptainer the same with `--pwd` and `--bind`), the c1 command:

```bash
docker run -d --name c1 --gpus all --user "$(id -u):$(id -g)" \
  -e OMP_NUM_THREADS=5 -e OMP_WAIT_POLICY=passive -v "$HOME/prod:/work" $IMG \
  ./run_jobs.sh -j 4 --campaign pth10-40_eta06_c1 67 15 0 /work/AuAu_0_10_pth10-40_eta06_c1 \
    --pthat-bins 10-20,20-30,30-40 --jets-per-bin 5 --parton-ymax 0.6 --write-particlize both
```

**On an HPC cluster** (SLURM array, one GPU per task, `P = 4` jobs per GPU):

```bash
SIF=$PWD/xscape_prod.sif CAMPAIGN=pth10-40_eta06_c2 NJOBS=40 EVENTS=15 \
  EXTRA_ARGS="--pthat-bins 10-20,20-30,30-40 --jets-per-bin 5 --parton-ymax 0.6 --write-particlize both" \
  sbatch --array=0-19 utils/slurm_prod_array.sh
```

That is 20 tasks × 40 jobs × 15 events = **12 000 events on 800 backgrounds**. Planning
numbers, scaled from c1:

| | per task (40 jobs, 600 events) | campaign (20 tasks) |
|---|---|---|
| GPU time at the GB10's rate (336 events/h) | ~1.8 h | ~36 GPU-hours, ~1.8 h wall on 20 GPUs |
| disk, stage 1 | ~160 GB | ~3.2 TB |
| disk, hadrons (stage 2 below) | ~46 GB | ~0.9 TB |

- **Time limit:** a task's jobs run 4 at a time, ~10–12 min per 15-event job on the GB10.
  Allow 2–3× on slower nodes (`--time` in the script is 24 h). A task that runs out of time
  resumes when submitted again.
- **Size:** per task 4 jobs × ~12 GB host memory and ~20 cores, the script's defaults
  (`--mem=48G`).
- **Smaller trial first:** `NJOBS=2 EVENTS=15 sbatch --array=0-1 --time=2:00:00 …`.

### Stage 2

The background of each window is shared by M = 5 jets, so sample it more:
`--oversample-bg per-pthat-bin` gives each background --oversample × M samples per window
(capped at 2000), the optimal split for per-window results. The c1 plan:

```bash
apptainer exec --env OMP_NUM_THREADS=1 --pwd "$PROD" --bind "$SCRATCH/prod:/work" \
  xscape_prod.sif ./run_hadronize.py "/work/pth10-40_eta06_c2/t*/*_particlize.h5" -j 16 \
  --oversample 400 --oversample-bg per-pthat-bin --n-frag 50 --keep-bits-p 12 --keep-bits-x 8
```

- **The quoted pattern** is expanded by `run_hadronize.py` inside the container (the host
  shell doesn't see `/work`); it must name particlize files, not directories.
- 400 samples per jet event, 5 × 400 = 2000 per background and window, 50 fragmentations.
  `--keep-bits-p 12 --keep-bits-x 8` stores 58 % of the bytes.
- **Time:** ~170–380 s per 15-event file and process (measured on the GB10 for the same
  setup, campaign `pth10-40_eta06_gridnorm`), so `-j 16` on a 16-core node does ~200 files
  in ~1 h. Memory up to ~1.7 GB per process (2000-sample backgrounds): ask for
  ~2 GB per core.
- **Disk:** ~1.15 GB per file, ~77 MB per event (estimate from the same campaign).

### Analysis

```python
# the whole campaign, all tasks: a glob of the particlize files
with HadronFileReader("/path/to/pth10-40_eta06_c2/t*/*_particlize.h5") as r:
    r.pthat_bins                       # (3, 2): the windows
    sigma, err = r.pthat_bin_sigma(1)  # window 20-30 GeV: σ after the cut [mb]
    acc, acc_err = r.pthat_bin_acceptance(1)
    ev = r.pthat_bin_events(1)         # its global events, for events= in hist/total
```

In the ROOT export, `<campaign>_campaign.root` holds per window `sigma_mb` and
`weight_mb = sigma_mb / n_events`, the weight of one event of that window in a
cross-section-weighted sum. For c1 the windows' σ after the cut were 2.3 × 10⁻³,
4.4 × 10⁻⁵ and 2.1 × 10⁻⁶ mb.

---

## 5. Moving the files

The outputs are in the bound directory on the host, so the host's tools can copy them. From
inside the container (e.g. jobs on the OSPool), every image has Pelican:

```bash
pelican object put -r /work/AuAu_a osdf:///NAMESPACE/AuAu_a
pelican object put -t /work/token /work/AuAu_a/FILE.h5 osdf:///NAMESPACE/AuAu_a/FILE.h5
```

The `-gcs` images write to Google Cloud Storage from Python (`fsspec.filesystem("gs")`).
Credentials: `BuildContainerProd.md`, *Getting the outputs home*.

**With the transfer tools.** For whole campaigns,
[`utils/remote_transfer`](../utils/remote_transfer/README.md) has `js_gcs.py` (GCS, default
`gs://test_fno`) and `js_osdf.py` (OSDF, default `osdf:///fno4hic`). Both take whole
directories, single files or patterns, picked by kind with `--what pair|h5|root|all`. They
skip what is already there, so an interrupted transfer finishes when the command is run
again, and they check every file by size and CRC32C.

```bash
utils/remote_transfer/js_osdf.py upload /work/AuAu_a --what h5 --dry-run    # what would go
utils/remote_transfer/js_osdf.py upload /work/AuAu_a --what h5              # -> osdf:///fno4hic/AuAu_a/
utils/remote_transfer/js_gcs.py upload /work/AuAu_a --what pair -j 8        # -> gs://test_fno/AuAu_a/
utils/remote_transfer/js_osdf.py download AuAu_a --what h5 --to /scratch    # on the CPU cluster
```

- **Where they run:** on the host, from a js-contrib checkout. Images built from js-contrib
  after 2026-10-01 have them under
  `/opt/X-SCAPE/external_packages/js-contrib/utils/remote_transfer/`.
- **Their environment:** each script makes its own on first use, which needs PyPI access and
  a writable home directory. In the `-gcs` images `JS_GCS_NO_ENV=1` and `JS_OSDF_NO_ENV=1`
  use the image's packages instead.
- **Credentials:** a service-account key for GCS, a bearer token for writing to OSDF. See
  the tools' README.

**While the campaign runs.** [`utils/upload_follow.py`](../utils/upload_follow.py) uploads each
job to the OSDF once it is complete and, with `--delete pair` (or `all`), deletes the local
copies after a verified upload (size, the manifest's CRC32C, every Nth file downloaded and
compared), so a campaign is not limited by the local disk. It runs on the host, next to
Docker or on a node that sees a SLURM campaign's directories, from a js-contrib checkout:

```bash
utils/upload_follow.py /data/AuAu_a AuAu_a --follow --delete pair
touch /data/AuAu_a/.upload_final       # when the campaign is done: a last pass, then it exits
```

`launch_2gpu.sh --upload REMOTE` starts it with the campaign. Options, checks and the measured
rates (~80 MB/s up, more than the ~57 MB/s a two-GPU campaign writes):
[`utils/README_launch.md`](../utils/README_launch.md#uploading-to-the-osdf-fno4hic).

**What to move where.** The particlize files are **self-contained** (format version 2, from
2026-10-01): surfaces, partons and the shower initiators. So the CPU cluster for stage 2,
and the analysis, need only `<stem>_particlize.h5` and the hadron files next to it, not the
large hydro pair files `<stem>.h5` (~170–285 MB per event), which can stay where the FNO
training uses them. With the transfer tools that is `--what h5` (particlize and hadron
files) for stage 2 and the analysis, `--what pair` for the FNO training, and `--what root`
for the ROOT files of `run_h5toROOT.py`. Particlize files made before carry no initiators; add them once, where
the pair files are next to them, with `add_initiators.py` (in the production folder; in
the container or the venv):

```bash
./add_initiators.py /work/AuAu_a                     # in the container (Docker/Apptainer --pwd)
python contribs/PyJetscape/example/prod_AuAu_0_10_jet/add_initiators.py DIR   # in the venv
```

In images built before this script existed, run it from a js-contrib checkout bound into the
container, or in the venv.

---

## 6. Analysis in a local venv

The analysis doesn't need X-SCAPE, the container, a GPU, conda or ROOT: a Python ≥ 3.10 venv
with this checkout reads every file
([`utils/analysis_env/README.md`](../utils/analysis_env/README.md)):

```bash
git clone https://github.com/jhputschke/js-contrib.git && cd js-contrib
./utils/analysis_env/setup_analysis_env.sh            # venv in ~/.venvs/js_analysis
source ~/.venvs/js_analysis/bin/activate
python utils/analysis_env/check_env.py /path/to/AuAu_a   # opens every file of the directory
```

`--with-pelican` / `--with-gcs` add the remote readers (files read in place over `osdf://`,
`gs://`); `--kernel NAME` registers a Jupyter kernel.

**Reading the hadrons** (`jetscape.hadrons_h5`):

```python
import numpy as np
from jetscape.hadrons_h5 import HadronFileReader

with HadronFileReader("/path/to/AuAu_a") as r:            # every particlize file in it
    print(r.n_files, "files,", r.n_events, "events, tags", r.tags())
    pt, pt_err = r.hist("bulk_bg", "pt", np.linspace(0, 3, 31), mask="charged")
    E_mid, E_err = r.total("bulk_jet", weights="E", mask=lambda ev, info: np.abs(ev.eta) < 1)
    ev = r.jet_event(2, 17)              # global event 2, oversample 17, with its fragments
```

`r.jet_minus_background(...)` gives the wake (jet leg − background) with its error; more in
[`prod_AuAu_0_10_jet/README.md`](../contribs/PyJetscape/example/prod_AuAu_0_10_jet/README.md), *C. Analysing a campaign: `HadronFileReader`*. Ready-made analyses:
[`jet_wake.ipynb`](../contribs/PyJetscape/example/prod_AuAu_0_10_jet/jet_wake.ipynb) (a pair file and its hadrons), and
[`analysis/`](../contribs/PyJetscape/example/analysis/README.md) (`wake_hadrons.py`, `jet_edep_balance_check.py`,
`hadron_distributions.ipynb`, FastJet notebooks). The hydro pair's layout: [`prod_AuAu_0_10_jet/README.md`](../contribs/PyJetscape/example/prod_AuAu_0_10_jet/README.md),
*What is written* and *Reading it*.

### Converting to ROOT for existing analysis code

[`run_h5toROOT.py`](../contribs/PyJetscape/example/prod_AuAu_0_10_jet/run_h5toROOT.py) turns a hadronized campaign into ROOT files, in the same
venv. It writes with uproot, or with ROOT when PyROOT imports (needed only for the
truncated-float options):

```bash
cd contribs/PyJetscape/example/prod_AuAu_0_10_jet                    # in the js-contrib checkout
python run_h5toROOT.py /path/to/AuAu_a -j 4                          # RNTuple, next to the inputs
python run_h5toROOT.py /path/to/AuAu_a -j 4 --format ttree           # TTree, for ROOT < 6.34
python run_h5toROOT.py /path/to/AuAu_a -j 4 --out-dir root --charged --eta-max 1 --no-x
```

Per production file, `<stem>_hadrons.root` with:

| tree / ntuple | one entry per | content |
|---|---|---|
| `bulk_jet`, `bulk_bg`, `jet_frag` | **oversample** (one complete sample of an event, like one MC event) | `event`, `unit`, `sample`, `bg_unit`, `n`, `pid[n]`, `pstat[n]`, `E/px/py/pz[n]` (GeV), `t/x/y/z[n]` (fm) |
| `events` | event | the event's information: pT̂ window and weight, the partons, droplets, the cross section |
| `windows` | pT̂ window | bounds, cross section, counts (`--pthat-bins` runs) |
| `provenance` | — | JSON: every setting, seed, XML and EoS behind the file |

plus `<campaign>_campaign.root` with the cross sections over all files (`windows`:
`sigma_mb`, `weight_mb` per window) and the list of files. An existing event loop reads an
oversample entry like an event: the hadrons of `bulk_jet` (the event with the jet), of
`bulk_bg` (its background) and of `jet_frag` (the jet's own hadrons). Reading in C++ or
with uproot (`uproot.open(f)["bulk_jet"].arrays()`) and the format choices are in
[`root_export/README.md`](../contribs/PyJetscape/example/prod_AuAu_0_10_jet/root_export/README.md).
In C++, [`analysis_root/HadronFileReader.h`](../contribs/PyJetscape/example/analysis_root/README.md)
returns an event's oversample as vectors of hadrons (`r.bkg(e, k)`, `r.bkg_dep(e, k)`,
`r.bkg_dep_frag(e, k)`), and `hadron_distributions.C` there histograms every source and the
wake.

- **Weights:** for windowed runs (`--pthat-bins`), weight each event with its window's
  `weight_mb` from the campaign file.
- **Statistics:** the oversamples of one event, and the events sharing one background
  (`--reuse`), are not independent: estimate errors per event or per background, not per
  entry (`root_export/README.md`).

---

## 7. Troubleshooting

| symptom | cause and fix |
|---|---|
| `./run_jobs.sh: No such file` (Apptainer) | no `--pwd "$PROD"`; or call `$PROD/run_jobs.sh` |
| `--outdir: give OUTDIR as the argument after FIRST_SEED` | OUTDIR is the 4th argument, not an option |
| `The seed registry … can't be written` / `Read-only file system: …/seeds_used.tsv` | OUTDIR's parent isn't writable: put OUTDIR one level inside a bound directory (`/work/out`, `~/prod/out`) |
| only one job runs with `-j 2` | NJOBS is 1: `-j` only limits how many run at once |
| `kernel launch … failed: no kernel image is available` | the image lacks this GPU's architecture: on a V100, an image without sm_70 (built before 2026-09-30 19:57 EDT, including the published `cu126`); use `cu124-20261003-cbc7263` until the rebuild (§1); discard that output |
| `No CUDA device found` / jobs much slower | no GPU visible: `--nv` / `--gpus`, `CUDA_VISIBLE_DEVICES`, or per-campaign MPS on several GPUs (§2) |
| `--mps: could not start the MPS daemon` | MPS doesn't work in this container/site: drop `--mps` |
| jobs killed (OOM) | plan ~12 GB host memory per job (~16 GB with `--bulk-info full`, ~22 GB with builds before the memory fixes): lower `-j` or ask for more `--mem` |
| busy-spinning threads, the libgomp warning | `OMP_WAIT_POLICY=passive` before launching |
| `pelican: command not found` | the published `cu126`/`cu130` predate Pelican: use `cu124`, or move the files from the host (§5) until the rebuild |
| many `hit_edge` warnings | the output grid is too small for this system: a larger x/y range in the grid YAML (§1, *Output grid*) |
| files with different `arr` shapes in one campaign | jobs ran with different grids (an older image's default vs. the new one): keep one grid per campaign |
| a failed job | its `OUTDIR/<stem>.log`; rerun the same command (finished jobs are skipped) |
