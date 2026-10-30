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
  converts the hadrons to ROOT for analysis code that expects ROOT. §6 is the short version;
  [`README_analysis.md`](README_analysis.md) has the environments, the readers for hydro and
  hadrons, FastJet in Python and in ROOT, and the notebooks.
- **Between them**, the files can travel through Google Cloud Storage or the OSDF with the
  tools in [`utils/remote_transfer`](../utils/remote_transfer/README.md)
  ([§5](#5-moving-the-files)).

The physics and every option of the production are in [`prod_AuAu_0_10_jet/README.md`](../contribs/PyJetscape/example/prod_AuAu_0_10_jet/README.md); the images
themselves (variants, building, testing) in
[`utils/BuildContainerProd.md`](../utils/BuildContainerProd.md).

---

## 1. Requirements

### Images

> **Image status (2026-10-14).** All three variants were rebuilt on 2026-10-14 from the merged
> branches (X-SCAPE `contrib` `9e84450`, MUSIC4GPU `XSCAPE` `46dad43`, js-contrib `main`
> `e8997c7`; iSS `98deaff`, 3dMCGlauber `b99d680`; `docker-prod.yml` run 38487241588). They
> have everything this guide describes: the speed-ups of 2026-10-11/12
> ([§2](#what-the-current-image-does-differently-final_optim)), and, new against the image
> before (`cu124-final_optim`), Matter's guard against a qhat lookup before t = 0
> (`8c975d37`) and the iSS leak fix (iSS `98deaff`). Au+Au output is bit-identical to
> `cu124-final_optim`, so its numbers hold: **2,160 events/h** on 2 × RTX 3090. The two fixes
> matter for O+O (ISR, and long jobs with iSS in the job;
> [§8](#8-oo-with-initial-state-radiation-prod_oo_isr)).
>
> | tag (the moving tag = the dated one) | digest | CPU arch | built | sm_70 (V100) | Pelican | memory fixes | 64×64×32 grid, edge flag | CPU fixes, restart | C4, late Δτ, A12 | Matter t < 0 guard, iSS leak fix |
> |---|---|---|---|---|---|---|---|---|---|---|
> | **`cu124` = `cu124-20261013-9e84450`** | `sha256:2cf57f75…` | amd64 | 2026-10-14 | yes | yes | yes | yes | yes | yes | **yes** |
> | **`cu126` = `cu126-20261013-9e84450`** | `sha256:2e870cac…` | amd64, arm64 | 2026-10-14 | yes | yes | yes | yes | yes | yes | **yes** |
> | **`cu130` = `cu130-20261013-9e84450`** | `sha256:76c02874…` | amd64, arm64 | 2026-10-14 | no (CUDA 13) | yes | yes | yes | yes | yes | **yes** |
> | `cu124-final_optim` = `cu124-20261013-9f4b419` | `sha256:91c04102…` | amd64 | 2026-10-13 | yes | yes | yes | yes | yes | yes | no |
> | `cu124-20261003-cbc7263` | | amd64 | 2026-10-03 | yes | yes | yes | no | no | no | no |
> | `cu126-20260930-d31946c`, `cu130-20260930-d31946c` | | amd64, arm64 | 2026-09-30 | **no** | **no** | **no** (~20 GB per job) | no | no | no | no |
>
> Full digests: `cu124` `sha256:2cf57f7556dcfff0289c1c46b9b188b58a3ce8026b2733cc4a17f90bca761b58`,
> `cu126` `sha256:2e870cacbddef12d2f675f206bc54cfb49b7c1f3822e19c487936be93da8fe0f`,
> `cu130` `sha256:76c02874c775dbc00b1d05cd4148bfa239a1e639b115f28208e20cc8970a828a`.
>
> - **Pin the digest** for a campaign (`jhputschke/xscape-prod@sha256:2cf57f75…`), and record it:
>   the files don't say which image made them.
> - **Not run on a machine yet:** the 2026-10-14 images. Their hydro code is that of
>   `cu124-final_optim`, which the 3090 numbers were measured with; the one short run per
>   cluster GPU type ([`REPORT_SUMMARY_CLUSTER.md`](REPORT_SUMMARY_CLUSTER.md)) is still to do.
> - Everything is merged: MUSIC4GPU `final_optim` into `XSCAPE` (PR #22, `46dad43`), X-SCAPE
>   `final_optim` into `contrib` (PR #171, `9e84450`), js-contrib into `main` (PR #60,
>   `e8997c7`). Images built from `contrib` before 2026-10-13 have the CPU fixes and the
>   restart, not C4, the late step or A12.

| tag | CUDA | GPUs | CPU arch | host driver |
|---|---|---|---|---|
| `jhputschke/xscape-prod:cu124` | 12.4 | V100 (sm_70) … H100/GH200 (sm_90), newer through PTX | amd64 | R550+ |
| `jhputschke/xscape-prod:cu126` | 12.6 | as `cu124` | amd64, arm64 | R560+ |
| `jhputschke/xscape-prod:cu130` | 13.2 | GH200 (sm_90) … B200/GB200, RTX 50xx, GB10 (sm_121); **no V100** | amd64, arm64 | R580+ |

- **Which one:** `nvidia-smi` shows the highest CUDA version the driver supports. All three
  are current (box above): `cu124` runs on every driver from R550 on, `cu126` where the driver
  supports ≥ 12.6, `cu130` (R580+) for Blackwell and arm64. There is no performance difference
  to expect between `cu124` and `cu126`.
- **For a campaign, use a dated tag or the digest**, e.g. `cu124-20261013-9e84450` /
  `@sha256:2cf57f75…`, and the same one on every machine: the moving tags (`cu124`) change
  with every build, and different builds can differ in the last digits, or more (the late
  time step changes the medium at 1e-3, [§2](#musics-late-time-step-on-by-default)). Run the
  whole campaign on GPUs: MUSIC's CPU path differs systematically from the GPU path (~0.2% in
  multiplicity, [`MUSIC_CPU_vs_GPU.md`](MUSIC_CPU_vs_GPU.md)), so don't mix the two.
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
  | less host CPU per event, bit-identical (A1–A7, A9 of [`REPORT_CPU_GPU.md`](Plans/REPORT_CPU_GPU.md)): MUSIC's anisotropy diagnostic off, no jet-leg store with `--particlize-only` | X-SCAPE `contrib` `6480dbeb` (MUSIC4GPU `943b089`) and js-contrib `main` `a9dd328` or later |
  | MUSIC_2 from MUSIC_1's state, on by default ([§2](#music_2-from-music_1s-state-on-by-default)) | X-SCAPE `contrib` `1098070c` (#168), js-contrib `main` `eb55c9e` (#57) |
  | the freeze-out without host copies (A10, A11), iSS Part C | X-SCAPE `contrib` `cdb7e437` (#170; MUSIC4GPU `81f38d9`), js-contrib `main` `8e4fba5` |
  | C4 (the solver stall break), the late time step (D1), A12 (Cornelius) ([§2](#what-the-current-image-does-differently-final_optim)) | X-SCAPE `contrib` `9e84450` (PR #171, merged 2026-10-13), MUSIC4GPU `XSCAPE` `46dad43` (PR #22, merged 2026-10-13), js-contrib `main` `e8997c7` (PR #60, merged 2026-10-13); `cu124-final_optim` was built from `9f4b419` / `b972b04` / `6b8ae36`, the 2026-10-14 images from the merge commits |

### Memory, cores, disk

Measured with the current image (`cu124-final_optim`) on 2 × RTX 3090 (Threadripper 3960X,
24 cores, 125 GB), the campaign setting of §4 with `--write-particlize both`, 10 jobs per
GPU ([`REPORT_FINAL_OPTIM.md`](REPORT_FINAL_OPTIM.md),
[`REPORT_SUMMARY_CLUSTER.md`](REPORT_SUMMARY_CLUSTER.md)):

| per production job (stage 1) | |
|---|---|
| **host memory** | **~6–7 GB per job under load** (VmHWM 5.97 GB mean, 6.66 max; 20 jobs peaked at 105 GB of 125) with a shared `LBT_TABLE_CACHE` (+~0.6 GB per job without). A job alone peaks at 6.4–8.7 GB. Most of it is the background's hydro history, kept for Matter/LBT as a slim 6-field copy; it was ~20 GB before the memory fixes (X-SCAPE `cbc72639`) and is 14.3 GiB with `--bulk-info full` ([PLAN_slim_bulk_info.md](Plans/PLAN_slim_bulk_info.md)). **Plan for 7 GB per job** (5 GB with `--particlize-only`, ~12 GB with `--bulk-info full`), plus ~8 GB per node |
| GPU memory | ~1 GB per job (9.6 GB per GPU at 10 jobs), including the restart's ~0.1 GB checkpoint: a 16 GB V100 holds 8–10 jobs |
| CPU | ~35 core-s per event under load (Zen 2). A job keeps ~1 core busy on average, with OpenMP bursts at the string deposition and the freeze-out search: **2–3 threads per job** (`OMP_NUM_THREADS`) |
| GPU | 2.1 GPU-s per event on an RTX 3090 (5.1 with the image before): the GPUs are 63 % busy, the 24 cores set the rate |
| time | ~25 s per event in a job sharing its GPU with 9 others; ~8–15 s per jet event for a job alone |
| disk | **0.243 GB per event** (pair + particlize, one background per 15 events); 0.097 GB with `--particlize-only`. With a new background every event: ~285 + ~154 MB (measured on the earlier 65 × 65 × 33 grid) |

So **cores per GPU now decide how many jobs share a GPU**, and how fast: per job ~2–3
cores and ~7 GB. Lower the number of jobs where a node gives a GPU less, not the threads per
job below 2. With images before `cu124-final_optim` the GPU is the tighter limit, and 4–8
jobs per GPU with ~5 threads each are enough. The memory per job is the same in every image
since the host-memory changes of 2026-10-10 (frame-by-frame reads, the LBT cache); plan
~12 GB per job with older ones.

#### Per-GPU sizing on a cluster

What to ask for per GPU (one SLURM array task), for the Au+Au campaign setting with the
current image. **None of these GPUs has been run.** The table is
[`REPORT_SUMMARY_CLUSTER.md`](REPORT_SUMMARY_CLUSTER.md)'s extrapolation from the 3090 box
and the GB10: GPU time scaled by the GPUs' FP32 rate and memory bandwidth, host time by
~85 Au+Au events/h per Zen 2-class core (newer cores 1.2–1.5× that).

| GPU | jobs per GPU (`P`) | `OMP_NUM_THREADS` | cores per GPU (`--cpus-per-task`) | host memory per GPU (`--mem`) | expected events/h per GPU | limit |
|---|---|---|---|---|---|---|
| (RTX 3090, measured) | 10 | 3 | 12 physical (24 threads) | ~53 GB used | **1,080** (~1,200 in 45-event jobs) | cores |
| **V100** (16 / 32 GB) | 8 | 2 | 12–16 | 64G | 700–950 | GPU |
| **A40** (48 GB) | 10–12 | 2 | 16–24 | 90–100G | 1,100–1,400 | cores at 16, GPU at 24 |
| **A100** (40 / 80 GB) | 10–12 | 2 | 16–24 | 90–100G | 1,000–1,500 | GPU or cores |
| **L40S** (48 GB) | 18–22 | 2 | 24–32 | 135–165G (100–120G with `--particlize-only`) | 2,000–2,900 | cores |

- **`slurm_prod_array.sh`** defaults to the A100/A40 row (`P=10`, `--cpus-per-task=20`,
  `--mem=96G`); for other GPUs pass `P` and the `sbatch` options from the table. The script
  sets `OMP_NUM_THREADS = cpus-per-task / P` (and warns below 2) and `OMP_WAIT_POLICY=passive`.
- **The launch settings of §2 are the script's defaults:** `MUSIC_CUDA_SYNC=block` (the host
  sleeps while it waits for the GPU: ~30 % less CPU per event), LBT's shared table cache
  (`LBT_CACHE`, `/work/lbt_cache/lbt_tables_v1.bin`: −0.6 GB and 6–11 s of start-up per job;
  the first job writes it) and `STAGGER=12` (`run_jobs.sh --stagger`: −11 GB of coinciding
  memory peaks at 20 jobs). MPS stays off (`USE_MPS=0`).
- **Disk is now a limit.** One L40S writes up to ~0.7 TB/h with the pair file, more than the
  ~0.29 TB/h of an OSDF upload. Use `--particlize-only` (~2.5× less) unless the hydro history
  is needed, and node-local scratch with `upload_follow.py` (§5).
- **CUDA MPS:** +5–7 % on the 3090s and +12 % on the GB10, but **never tried on a cluster**;
  leave `USE_MPS=0` until it works on the site ([CUDA MPS](#cuda-mps-optional)). Without it,
  expect ~5–10 % less, and maybe 1–2 more jobs per GPU.
- **Time:** at ~25 s per event in a loaded job, a 45-event job takes ~20 min; a 40-job task at
  `P = 10` ~1.5 h on 3090-class GPUs. Allow 2× on V100s. The script's 24 h is generous.
- **SLURM memory:** the limit counts the page cache of the HDF5 files being written. If tasks
  are killed near it, raise `--mem` by ~10–15 % rather than lowering `P`.
- **Before a large campaign**, on the first node: a 20–30 min sweep of `P` (e.g. 8, 12, 16) at
  the planned `--cpus-per-task`. Record events/h by makespan, `nvidia-smi dmon -s u`, and the
  task's memory peak. If the GPU stays below ~75 % busy while the cores are full, the node is
  host-limited: more cores per GPU pay, more jobs don't.

```bash
# A100/A40 nodes, the script's defaults: 10 jobs per GPU, 2 threads each; 20 tasks = 20 GPUs
SIF=$PWD/xscape_prod.sif CAMPAIGN=pth3_c2 \
  EXTRA_ARGS="--pthat-bins 10-20,20-30,30-40 --jets-per-bin 5 --parton-ymax 0.6 --write-particlize both" \
  sbatch --array=0-19 utils/slurm_prod_array.sh
# V100 nodes: 8 jobs per GPU on 16 cores
SIF=$PWD/xscape_prod.sif CAMPAIGN=pth3_v100 P=8 EXTRA_ARGS="…" \
  sbatch --array=0-19 --cpus-per-task=16 --mem=64G utils/slurm_prod_array.sh
```

| per hadronization process (stage 2) | |
|---|---|
| memory | ~1.4 GB per process up to ~1000 oversamples per surface, 1.7 GB at 2000 |
| time | both legs, 500 oversamples, one core: **10.6 s per event**, 8.8 s with `--correlated` (iSS Part C, in the current image; 14 s before) |
| rate | ~340 events/h per core at 500 oversamples: **~6–7 cores keep up with 2,000 events/h** of stage 1 |
| disk | ~100 MB per leg and event at 500 oversamples (less with the options in §3) |

### Output grid

The hydro pair is stored on the output grid of `../config/grid_fno.yaml`
(`run_prod_jet.py --grid` for another one). Since js-contrib `c8b682d` (2026-10-06) the default
is **64 × 64 × 32 cells**: x, y −12.2 … 12.2 fm at 0.3875 fm, η_s −4.84 … 4.84 at 0.3125.
Before, it was 65 × 65 × 33 (x, y −10 … 10 fm, all steps 0.3125), as in `fastdata_AuAu200_tune_0_10.h5`.

- **Files of the two grids don't go into one training set.** Keep one grid per campaign, and
  check `prod_grid_yaml` in a file's attributes when combining campaigns.
- **Why the change:** the jet leg's fluid above freeze-out crossed ±10 fm in ~5% of 0–10%
  events; ±12.2 fm holds it. How much of the wake each grid keeps:
  [`Wake_grid_comparison.md`](Wake_grid_comparison.md).
- **With an image that predates the change** (the published `cu126`, `cu130` and the older
  `cu124-*` tags, see above), bind the new
  YAML from a js-contrib checkout (`contribs/PyJetscape/example/config/grid_fno.yaml`)
  and pass it after OUTDIR: Docker `-v "$PWD/grid_fno.yaml:/grid/grid_fno.yaml:ro"`, Apptainer
  `--bind "$PWD/grid_fno.yaml:/grid/grid_fno.yaml"`, then `… --grid /grid/grid_fno.yaml`.

---

## 2. Stage 1: hydro + jets on GPUs

### The command inside the container: `run_jobs.sh`

Everything runs `run_jobs.sh` from the production folder
`/opt/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape/example/prod_AuAu_0_10_jet`:

```
./run_jobs.sh [-j P] [--mps] [--stagger S] [--campaign NAME] NJOBS EVENTS_PER_JOB FIRST_SEED OUTDIR [run_prod_jet.py options]
```

| | |
|---|---|
| `NJOBS` | number of jobs = number of output files |
| `EVENTS_PER_JOB` | events in each job. **Total events = NJOBS × EVENTS_PER_JOB** |
| `-j P` | at most P jobs at the same time (default 1). It doesn't create jobs: `-j 2 1 1 5` runs one job |
| `--mps` | the P jobs share the GPU through CUDA MPS ([CUDA MPS](#cuda-mps-optional)) |
| `--stagger S` | the first P jobs start S seconds apart, so their memory peaks don't coincide ([Launch settings](#launch-settings-for-several-jobs-per-gpu)) |
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

### MUSIC_2 from MUSIC_1's state (on by default)

In images with the restart (§1), the jet leg doesn't recompute the background's first steps.
- **Why it works:** up to the first droplet the two legs are bit-identical, and no droplet
  deposits before the liquefier's `tau_delay` (1 fm/c).
- **What happens:** MUSIC_1 keeps a checkpoint of its state there, once per background.
  MUSIC_2 starts from it for every jet on that background.
- **What it saves:** the string deposition and ~30 GPU steps per jet event, ~25 core-s per jet
  event.
- **What doesn't change:** the output, bit for bit.

Details and the checks:
[`prod_AuAu_0_10_jet/README.md`](../contribs/PyJetscape/example/prod_AuAu_0_10_jet/README.md#music_2-from-music_1s-state-default).

- **Turning it off:** `--no-music2-restart` after OUTDIR (`run_jobs.sh … OUTDIR
  --no-music2-restart`), or in `EXTRA_ARGS` for the SLURM script. Older images don't know the
  option ("unrecognized arguments"): they run without the restart anyway.
- **When it switches itself off:**
  - with `--write-particlize jet` (MUSIC_1 then builds no surface, so the checkpoint would lack
    the jet leg's early surface);
  - on builds without it.

  The job says so at the start (`restart  OFF: …`). Use `--write-particlize both` for a
  campaign that should get it.
- **Checking it:** run one job with `MUSIC_RESTART_VERIFY=1` (`-e` / `--env`). Every jet leg
  then runs from τ0, and the log compares its state with the checkpoint (`restart verify at
  step 28: identical to the checkpoint`).

### What the current image does differently (`final_optim`)

`cu124-final_optim` (§1) adds three MUSIC4GPU changes to the CPU fixes and the restart, and
iSS Part C for stage 2 ([`Final_Optims.md`](Plans/Final_Optims.md),
[`OPTIMIZATION_CHANGES.md`](OPTIMIZATION_CHANGES.md)):

| change | what | output | gain |
|---|---|---|---|
| **C4**, the solver stall break | the reconstruction's Newton–Brent solvers stop once the iterate no longer changes in float32; before, lanes below float resolution ran all 60 iterations and held their warps | float rounding: fields ~1e-6, surfaces ±2 cells per leg; the hadron-level wake yield +0.8 % ± 0.2 % | −29 % GPU time per step |
| **D1**, the late time step | Δτ 0.02 → 0.05 fm/c from τ = 3 fm/c (next section) | the medium at 1e-3 | −42 % GPU time per Au+Au event on top |
| **A12**, Cornelius | the freeze-out search without per-element heap allocations, one Cornelius per thread | bit-identical | the search −30 % |
| iSS Part C | the yield loop over packed cell arrays, a tabulated momentum CDF | bit-identical | `hadronize.py` −25 % (§3) |

- **Net:** the Au+Au campaign goes from 1,271 to **2,160 events/h** on 2 × RTX 3090 (+70 %),
  O+O hadrons-only from 10,313 to 13,728 (+33 %). The GPUs are now 63 % busy: the host's cores
  set the rate, so more cores per GPU pay, more jobs per GPU don't (§1).
- **Physics check** (300 Au+Au events on 20 backgrounds, 720 O+O events, same seeds against
  the image before; [`REPORT_FINAL_OPTIM.md`](REPORT_FINAL_OPTIM.md)): C4 accepted, D1
  acceptable with one setting per campaign.
- C4 and A12 have no switch. Only the late step can be turned off.

### MUSIC's late time step (on by default)

From τ = 3 fm/c MUSIC takes steps of 0.05 fm/c instead of 0.02 (D1;
`<late_delta_tau>0.05</late_delta_tau>`, `<late_tau_switch>3.0</late_tau_switch>` in the
production XMLs of `prod_AuAu_0_10_jet` and `prod_OO_isr`).
- **The output format doesn't change.** MUSIC's freeze-out checks (facTau 5 → 2) and stored
  frames (every 5th step → every 2nd) stay every 0.1 fm/c, so the pair and particlize files
  and the FNO format are as before. About half the MUSIC steps: Au+Au background 479 → 272,
  jet leg 501 → 282.
- **Why it is safe:** the stability number Δτ · Σ a/Δx is 0.42 at τ = 3 fm/c with 0.05,
  against 0.37 for the production's own 0.02 at τ0. The jet source and the liquefier follow
  the step (X-SCAPE `d425575e`): their deposition windows tile the τ axis across the switch,
  and the 1/Δτ of the deposit follows it. Without that, every droplet after 3 fm/c would
  deposit 2.5× too much, or be missed.
- **What changes** (against `--late-dtau 0`, same seeds):
  - bit-identical before 3 fm/c; after it the Au+Au medium differs by 1.1e-3 (energy-weighted
    \|Δe\|/e above freeze-out), O+O by 1.2e-4. The GPU-vs-CPU difference is 2e-3;
  - the deposited energy agrees to 1e-4, the wake's energy to +0.3 % ± 0.2 %, its centroid to
    0.03 fm;
  - the cell-level wake δe = e_jet − e_bg differs by ~5 %, almost all of it at scales below
    ~1 fm (0.8 % after 2 fm smoothing): a time discretisation of a source moving at c;
  - hadron level: no change within ±2 % (yield, energy, Δφ shape of the wake);
  - O+O: the freeze-out surface 0.20 % larger, τ_fo one frame later in 11 % of events;
  - the showers differ in about half the events (the medium differs at 1e-3, so Matter and LBT
    sample it differently): other samples, not a bias.
- **One setting per campaign.** Don't mix late-step and constant-step files in one data set:
  for an FNO trained on cell-level wake fields the target differs at that level. A file
  records it in its job XML (`prod_user_xml`, `<stem>.xml`); the check is in
  [`README_analysis.md`](README_analysis.md#production-settings-an-analysis-must-know).
- **Switching it:** after OUTDIR (or in `EXTRA_ARGS`), `--late-dtau 0` for a constant step
  (bit-identical to the same image's constant-step output), `--late-dtau DT` and
  `--late-tau TAU` for another step or switch time. Keep 0.1 fm/c a whole number of steps
  (0.05, 0.025): otherwise the log warns (`late step: the freeze-out check spacing changes`)
  and the frames leave the 0.1 fm/c grid. Older images don't know the options ("unrecognized
  arguments") and run a constant step.
- **With the restart:** the switch (step 130) comes after the checkpoint (step 28), and the
  checkpoint's key records the switch, so the restart works as before
  (`MUSIC_RESTART_VERIFY=1`: identical with the late step).
- **In the log**, for every MUSIC run: `late step: delta_tau 0.02 -> 0.05 fm/c at step 130
  (tau = 3 fm/c), facTau 5 -> 2, a frame every 2 steps`.

### Launch settings for several jobs per GPU

**The quick way: [`utils/prodwizard.py`](../utils/prodwizard.py).** Pick a production
preset and a machine (the 3090 box, the GB10, the M3 Max, or a SLURM cluster with V100, A40,
A100 or L40S). It writes the launch command with the settings below and estimates events/h,
duration, disk and memory, marking each as measured or estimated with its source. It also checks
the plan: events per job a multiple of the pT̂ windows × jets, memory, disk, upload rate, the
image, an OUTBASE that holds another campaign. It never starts jobs. The presets and the measured
settings are in [`utils/prod_presets.json`](../utils/prod_presets.json); update them after a
benchmark.

```bash
utils/prodwizard.py                        # the wizard (textual): a form, the plan beside it
utils/prodwizard.py --list                 # presets and machines
utils/prodwizard.py --preset auau_jet --machine 2x3090 --events 2700 --campaign c5 --out /data/c5
utils/prodwizard.py --preset pbpb_zjet --machine slurm_a40 --events 19200 --gpus 20 \
    --campaign zjet_c10 --save launch_zjet_c10.sh      # a script, the plan in its header
```

The settings measured best on 2 × RTX 3090 (with the current image) and on the GB10 (with
earlier code). None of them changes the output:

| setting | how | why |
|---|---|---|
| jobs per GPU | `run_jobs.sh -j P`: 10 on an RTX 3090 with 12 cores per GPU; others in §1 | the GPU needs several jobs to stay busy; the cores and host memory set the ceiling (~2–3 cores, ~7 GB per job) |
| threads per job | `OMP_NUM_THREADS` ≈ cores per GPU / P, at least 2 | more threads than cores oversubscribe them; keep 2 or more, since the string deposition and the freeze-out search run in OpenMP bursts |
| `OMP_WAIT_POLICY=passive` | `-e` / `--env` | idle OpenMP threads sleep instead of spinning |
| **`MUSIC_CUDA_SYNC=block`** | `-e` / `--env` | the host sleeps while it waits for the GPU: ~30 % less CPU per event at the same throughput. The log says `[MUSIC-GPU] host sync: block` |
| **`LBT_TABLE_CACHE=FILE`** | `-e` / `--env`, a file under a bound directory that exists | all jobs share one read-only copy of LBT's 0.66 GB of tables: −0.6 GB and 6–11 s of start-up per job. The first job writes it (`LBT: wrote the table cache …`); one file per LBT-tables version; not the container's `/dev/shm`. Apptainer/SLURM: [the LBT table cache](#the-lbt-table-cache-with-apptainer-and-slurm) |
| `--stagger S` | `run_jobs.sh` option, before NJOBS | the first P jobs start S s apart, so their memory peaks don't coincide: −11 GB at 20 jobs. S = 12 is the measured setting for 8–10 jobs per GPU, chosen as seconds per event under load / P with the earlier image (~100 s / 8). At today's ~25 s per event a smaller S probably does as well and shortens the ramp (not measured) |
| `--mps` | `run_jobs.sh` option | +5–7 % from 3 jobs per GPU on the 3090s, +12 % on the GB10 ([CUDA MPS](#cuda-mps-optional)) |
| `OPENBLAS_NUM_THREADS=1` | set by the drivers | more threads don't speed up the resampling and cost 8 % CPU spinning |

### Docker (workstation, cloud VM)

```bash
mkdir -p ~/prod/lbt_cache
docker pull jhputschke/xscape-prod:cu124-20261013-9e84450
IMG=jhputschke/xscape-prod@sha256:2cf57f7556dcfff0289c1c46b9b188b58a3ce8026b2733cc4a17f90bca761b58   # = cu124-20261013-9e84450

# a test: 2 jobs x 1 event, both at once, seeds 1 and 2
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  -e OMP_NUM_THREADS=4 -e OMP_WAIT_POLICY=passive \
  -v "$HOME/prod:/work" $IMG \
  ./run_jobs.sh -j 2 2 1 1 /work/test --write-particlize both

# a campaign on one GPU with ~24 cores and 64 GB: 6 jobs at a time (~7 GB and 3 cores each),
# 24 jobs x 45 events with the pT-hat windows of section 4, new seeds, in the background
docker run -d --name xscape --gpus all --user "$(id -u):$(id -g)" \
  -e OMP_NUM_THREADS=3 -e OMP_WAIT_POLICY=passive -e MUSIC_CUDA_SYNC=block \
  -e LBT_TABLE_CACHE=/work/lbt_cache/lbt_tables_v1.bin \
  -v "$HOME/prod:/work" $IMG \
  ./run_jobs.sh -j 6 --mps --stagger 12 --campaign AuAu_a 24 45 0 /work/AuAu_a \
    --pthat-bins 10-20,20-30,30-40 --jets-per-bin 5 --parton-ymax 0.6 --write-particlize both
docker logs -f xscape          # progress;  docker stop xscape  stops it cleanly
```

- Docker starts in the production folder, so `./run_jobs.sh` works as is.
- `--user` makes the files yours; `-v` binds a host directory to `/work`.
- **Own XML** (another collision system): keep it on the host, bind its directory and pass
  the container's path: `-v "$PWD/xml:/xml:ro" … ./run_jobs.sh … /work/PbPb --user-xml
  /xml/PbPb_0_10_jet.xml` ([Your XML from outside the container](#your-xml-and-grid-from-outside-the-container)).
  What to change, with Pb+Pb at 5.02 TeV as the example: [§9](#9-another-collision-system-pbpb-at-the-lhc).
- **O+O with initial-state radiation** (`prod_OO_isr`): another folder of the image, selected
  with `-w` (Apptainer `--pwd`); needs a newer image, see [§8](#8-oo-with-initial-state-radiation-prod_oo_isr).

### Apptainer (HPC)

```bash
apptainer pull xscape_prod.sif docker://jhputschke/xscape-prod:cu124-20261013-9e84450
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

### The LBT table cache with Apptainer and SLURM

`LBT_TABLE_CACHE=FILE` names a file **inside the container**. The first Au+Au job that finds
no usable FILE parses LBT's text tables as usual, then writes FILE and maps it. Every later
job maps FILE read-only, so a node keeps one copy of the 0.66 GB of tables in its page
cache, shared by all its jobs, instead of one copy per job. The results are the same with or
without it. Nothing has to be built, but on a cluster:

- **FILE must be in a bound, writable directory that exists.** The image is read-only, and a
  host path such as `$SCRATCH/...` is not visible inside the container unless bound. With
  `--bind "$SCRATCH/xscape_prod:/work"`, run `mkdir -p $SCRATCH/xscape_prod/lbt_cache` on the
  host and pass `--env LBT_TABLE_CACHE=/work/lbt_cache/lbt_tables_v1.bin`.
  `slurm_prod_array.sh` does both by default (its `WORK` is `$SCRATCH/xscape_prod`).
  If FILE can't be written, the job logs `LBT: could not write the table cache … keeping a
  private copy` and runs as it would without the cache (+0.6 GB).
- **Shared or node-local:**
  - *On the shared file system* (the script's default, `$WORK/lbt_cache/`): one file for
    the whole campaign, written once. Each node reads it once into its page cache. It needs
    only a read-only `mmap` and a `rename`, which Lustre, GPFS and NFS all support. **It has
    not been tried on a cluster yet.**
  - *Node-local* (`BINDS=/tmp:/nodetmp LBT_CACHE=/nodetmp/lbt_tables_v1.bin`): no reads
    from the shared file system. The first job on each node writes the file (6–11 s, 0.66 GB
    of local disk). Where SLURM gives each job its own `/tmp`, every task writes its own
    copy. Use this if the shared file system is slow or short of quota.
- **Make the file once before a large array.** Jobs that start together each find no file.
  Each one parses the tables, writes its own temporary copy (`FILE.tmp.<pid>`) and renames it
  into place. That is safe: no job reads a half-written file, and the copies are identical.
  But 20 tasks starting at once write up to 20 × 0.66 GB to the shared file system. One
  1-event job on a GPU node makes the file first (add your site's `-p`/`-A`):

  ```bash
  WORK=$SCRATCH/xscape_prod; mkdir -p $WORK/lbt_cache
  srun --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=0:20:00 \
    apptainer exec --nv --env LBT_TABLE_CACHE=/work/lbt_cache/lbt_tables_v1.bin \
    --pwd "$PROD" --bind "$WORK:/work" xscape_prod.sif \
    python run_prod_jet.py --events 1 --seed 1 --outdir /work/lbt_cache/init \
      --seed-registry none 2>&1 | grep "LBT:"
  ls -l $WORK/lbt_cache/lbt_tables_v1.bin    # 656,581,952 bytes
  rm -r $WORK/lbt_cache/init                 # the test event, not needed
  ```

  The `grep` shows `LBT: wrote the table cache /work/lbt_cache/lbt_tables_v1.bin`. The
  trial run of the next section (`NJOBS=2 … P=2 sbatch --array=0-1`) makes the file as
  well. Within a task, `--stagger 12` usually lets the first job write the file before
  the second job starts.
- **Check a campaign:** every Au+Au job log has `LBT: tables mapped read-only from
  /work/lbt_cache/lbt_tables_v1.bin`, including the job that wrote the file.
  `grep -L "mapped read-only" OUTDIR/*.log` lists the jobs without the cache.
- **One file per LBT-tables version.** The file's header records the table dimensions and
  the sizes of the table files. A file that doesn't match is ignored (`… does not match
  these tables; reading the text files`) and replaced. Images with the same tables share
  `lbt_tables_v1.bin`. If an image ever comes with other LBT tables, give its file a new name
  (`lbt_tables_v2.bin`). Tables of the same size but other content are not detected, and two
  images using one file would keep rewriting it. Where arm64 and amd64 nodes share a file
  system (NCSA DeltaAI and Delta), keep one file per architecture as well
  (`lbt_tables_v1_arm64.bin`, `…_amd64.bin`): the header would not tell them apart.
- **Memory:** the shared copy is page cache. SLURM counts it once per node, against the task
  whose job read it first. The ~7 GB per job of §1 already assumes the cache.
- **Not used** by O+O (no LBT; the script sets `LBT_CACHE=none`) or by stage 2. Pb+Pb (§9)
  uses the same tables and the same file.

### SLURM: one GPU per array task

[`utils/slurm_prod_array.sh`](../utils/slurm_prod_array.sh) runs a campaign as a job
array: each task gets one GPU and runs `P` jobs on it (`run_jobs.sh -j P`), with the launch
settings above. `PROD=oo` runs `prod_OO_isr` (§8);
[`utils/slurm_hadronize.sh`](../utils/slurm_hadronize.sh) is stage 2 (§3).

```bash
SIF=$PWD/xscape_prod.sif CAMPAIGN=AuAu_a EXTRA_ARGS="--write-particlize both" \
  sbatch utils/slurm_prod_array.sh                    # 10 tasks = 10 GPUs
SIF=$PWD/xscape_prod.sif CAMPAIGN=AuAu_a EXTRA_ARGS="--write-particlize both" \
  NJOBS=2 EVENTS=3 P=2 sbatch --array=0-1 --time=1:00:00 utils/slurm_prod_array.sh   # a trial
```

- **Why not one SLURM job per production job:** SLURM gives whole GPUs, and one job keeps
  a GPU only partly busy (most of an event is CPU work). Only jobs in one allocation can
  share a GPU.
- **Size it:** the defaults, `P=10` with `--gres=gpu:1 --cpus-per-task=20 --mem=96G`, are an
  A100's or A40's share of a node. Match `P`, cores and memory to one GPU's share of yours
  with the table of §1 (~2–3 cores and **~7 GB per job**). The script sets
  `OMP_NUM_THREADS = cpus / P` itself.
- **Launch settings:** `MUSIC_CUDA_SYNC=block`, the LBT cache under `WORK/lbt_cache/` and
  `STAGGER=12` by default; each is a variable (`LBT_CACHE=none`, `STAGGER=0`, `USE_MPS=1`).
  The cache file: [The LBT table cache with Apptainer and SLURM](#the-lbt-table-cache-with-apptainer-and-slurm).
- **After the jobs** each task runs the deposition check of [Checking a run](#checking-a-run)
  on its directory and prints the jobs to redo (`CHECK_DEPOSITION=0` turns it off).
- **Output:** `WORK/CAMPAIGN/t000/`, `t001/`, … with one shared seed registry. The script
  checks that the file system's locks work across nodes, or use `SEED_MODE=ranges`.
- **Restarting:** submit the same command again.
- All settings: `BuildContainerProd.md`, *SLURM: one GPU per array task*.

### Several GPUs in one machine or allocation

MUSIC always uses the first GPU it sees. Run **one campaign per GPU**, each with its own
`CUDA_VISIBLE_DEVICES`, OUTDIR and campaign name; the OUTDIRs side by side share the seed
registry:

```bash
# Docker: one container per GPU (each sees its GPU as GPU 0); 8 jobs per GPU, 3 threads each
for g in 0 1; do
  docker run -d --name xscape_gpu$g --gpus device=$g --user "$(id -u):$(id -g)" \
    -e OMP_NUM_THREADS=3 -e OMP_WAIT_POLICY=passive -e MUSIC_CUDA_SYNC=block \
    -e LBT_TABLE_CACHE=/work/lbt_cache/lbt_tables_v1.bin -v "$HOME/prod:/work" $IMG \
    ./run_jobs.sh -j 8 --stagger 12 --campaign AuAu_gpu$g 24 45 0 /work/AuAu/gpu$g --write-particlize both
done

# Apptainer / one container with all GPUs (no --mps here, see CUDA MPS below)
apptainer exec --nv --env OMP_NUM_THREADS=3,OMP_WAIT_POLICY=passive,MUSIC_CUDA_SYNC=block \
  --env LBT_TABLE_CACHE=/work/lbt_cache/lbt_tables_v1.bin \
  --pwd "$PROD" --bind "$SCRATCH/prod:/work" xscape_prod.sif bash -c '
  trap "kill \$(jobs -p) 2>/dev/null; wait" INT TERM
  CUDA_VISIBLE_DEVICES=0 ./run_jobs.sh -j 8 --stagger 12 --campaign AuAu_gpu0 24 45 0 /work/AuAu/gpu0 --write-particlize both &
  CUDA_VISIBLE_DEVICES=1 ./run_jobs.sh -j 8 --stagger 12 --campaign AuAu_gpu1 24 45 0 /work/AuAu/gpu1 --write-particlize both &
  wait'
```

The `trap` passes Ctrl-C and `docker stop` on to both campaigns; `wait` keeps the container
alive until both are done.

**Ready-made for Docker on two GPUs:** [`utils/launch_2gpu.sh`](../utils/launch_2gpu.sh) does the
Docker variant above with the measured best setup as its default: `run_jobs.sh -j 10 --mps
--stagger 12` per GPU, 3 threads, `MUSIC_CUDA_SYNC=block` and LBT's tables shared through one
cache file. It adds a disk guard that stops the campaign before the disk fills (`--min-free`),
and optionally the upload of every finished job to the OSDF while it runs (`--upload`, below in
§5). Measured on 2 × RTX 3090 with the c1 settings of §4: **2,160 events/h** with the current
image (15-event jobs; ~2,400–2,500 expected in 45-event jobs, where the stagger ramp counts
less). Usage, tuning and the measurements: [`utils/README_launch.md`](../utils/README_launch.md).
Under SLURM, if `CUDA_VISIBLE_DEVICES` is already set (e.g. `2,3`), use those numbers.

```bash
cd js-contrib/utils
# c1 settings of §4, 60 jobs x 45 events (a multiple of 20 jobs), ~1.2 h; pin the image
IMAGE=jhputschke/xscape-prod:cu124-20261013-9e84450 \
nohup ./launch_2gpu.sh 60 45 0 /data/camp_pth3 --campaign pth3 \
    --pthat-bins 10-20,20-30,30-40 --jets-per-bin 5 --parton-ymax 0.6 \
    --write-particlize both > /data/camp_pth3.log 2>&1 &

# an image older than 2026-10-10 (no LBT table cache, no --stagger): 8 jobs per GPU, 4 threads
JOBS_PER_GPU=8 OMP_THREADS=4 ./launch_2gpu.sh --lbt-cache none --stagger 0 64 45 0 /data/camp_old ...
```

**10 jobs per GPU** (the launcher's default) needs an image with the host-memory changes of 2026-10-10 (js-contrib
`593fcc4` or later; `cu124-final_optim` has them). Measured in a same-day comparison, with the
2026-10-10 image:
- **Throughput:** +8.4 % over 8 jobs/GPU, 805 against 743 events/h by makespan.
- **Host memory:** peak 104.5 GB of 125, against 91.7 GB at 8 jobs/GPU.

The launcher options:
- `--lbt-cache FILE` lets all jobs share one read-only copy of LBT's 0.66 GB of tables. If FILE is missing, the launcher makes it first (~1 min).
- `--stagger 12` starts each GPU's first jobs 12 s apart, so their memory peaks don't coincide.

Details: [README_launch.md](../utils/README_launch.md#10-jobs-per-gpu) and
[BENCHMARK_2x3090.md](BENCHMARK_2x3090.md#memory-10-per-gpu).

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
| `[MUSIC-GPU] host sync: block` | `MUSIC_CUDA_SYNC=block` arrived (`auto` without it) |
| `late step: delta_tau 0.02 -> 0.05 fm/c at step 130 (tau = 3 fm/c), facTau 5 -> 2, a frame every 2 steps` | the late time step is on ([§2](#musics-late-time-step-on-by-default)); missing with `--late-dtau 0` and in older images |
| `LBT: wrote the table cache …` / `LBT: … mapped read-only …` | `LBT_TABLE_CACHE` works: the first job wrote the file, the others share it. A cache that doesn't match the tables is ignored with a warning |
| `… done in … s` per event | the speed: events/h = jobs × 3600 / s per event |
| `No CUDA device found`, `falling back to CPU` | **no GPU**: the job runs on the CPU, ~7–10× slower, and its results differ systematically from the GPU path's (~0.2%, [`MUSIC_CPU_vs_GPU.md`](MUSIC_CPU_vs_GPU.md)). **Don't add that output to a GPU campaign.** Check `--nv`/`--gpus`, `CUDA_VISIBLE_DEVICES`, MPS |
| `edge     flag e > 0.2342 GeV/fm^3 on the x/y edge of the output grid` (at the start) | the edge flag is on, with MUSIC's freeze-out energy density as threshold (images with js-contrib `20109af` or later) |
| `PairH5Writer: event i: the jet leg has e = … on the transverse edge of the output grid` | fluid above freeze-out reached the edge of the stored box: the stored evolution misses part of it (the surfaces and hadrons don't). Flagged in `diag/{bg,jet}_hit_edge`; counted in the `.json` (`events_hit_edge`). A few per campaign are expected on the earlier ±10 fm grid; frequent ones mean the grid is too small |
| `… the freeze-out surface reached the transverse grid boundary, so MUSIC stopped it early` | MUSIC's own grid (`<IS><grid_max_x>`, ±15 fm) is too small: that event's evolution is truncated (`diag/{bg,jet}_hit_boundary`) |
| `kernel launch '…' failed: no kernel image is available` | the image has no code for this GPU (e.g. a V100 with an image without sm_70, which included the `cu126` of 2026-09-30, §1). **Discard the output** and use an image with sm_70 (`cu124` or `cu126` of 2026-10-14) |
| `WARNING: NOT in effect … export OMP_WAIT_POLICY=passive` | set `OMP_WAIT_POLICY=passive` before launching |
| `restart  MUSIC_2 starts from MUSIC_1's state at tau <= 1 fm/c` (at the start); per event `…, MUSIC_2 started from the checkpoint at step 28 (tau = 0.96 fm/c)` | the restart is on and used ([§2](#music_2-from-music_1s-state-on-by-default)) |
| `restart  OFF: …` (at the start) | the restart is off, for the reason given: `--no-music2-restart`, `--write-particlize jet`, or a build without it |
| per event `MUSIC_2 run from tau0: <reason>` | this event's jet leg did not fit the checkpoint and ran from τ0, with the same result. Rare; frequent ones point to a setup problem (troubleshooting) |

Each job also writes `<stem>.json` (seed, events written, settings, `events_hit_edge`,
`droplets_total`), and the campaign ends with `OUTDIR/run_jobs.finished`.

**The per-job deposition check.** Once, in ~250 campaign jobs on the 3090 box, all 15 events
of one job lost almost no jet energy (1–5 droplets per event instead of 13–34), with a correct
background and correct initial partons and no warning in the log. The same seed alone gave
the normal result, so the cause is unknown ([`REPORT_FINAL_OPTIM.md`](REPORT_FINAL_OPTIM.md)
§4). Check every campaign for it: flag a job whose droplets per event fall below half the
campaign's median, with [`utils/check_deposition.py`](../utils/check_deposition.py) (standard
library only; on the host, in the venv or in the container; exit code 1 if a job is
flagged). `slurm_prod_array.sh` runs the same check on each task's directory when its jobs
are done.

```bash
utils/check_deposition.py /work/AuAu_a                     # every job summary below it
utils/check_deposition.py /data/c1/gpu0 /data/c1/gpu1      # one campaign in two directories
```

To redo a flagged job, move its files (`<stem>.*`) out of OUTDIR and run the campaign's
command again: `run_jobs.sh` reruns the job (with a new seed in a seed-0 campaign).

**All of it at once, during the run or after: [`utils/jobmon.py`](../utils/jobmon.py).** It
reads the same logs and `.json` files and shows jobs done / running / failed, events, events/h,
the ETA, disk per event, the failed jobs with their last log lines, the deposition check, edge
hits, skipped events and the most frequent warnings. Standard library only; `--watch` uses
`rich` if installed, `--tui` needs `textual` (both in the analysis venv,
[`utils/analysis_env`](../utils/analysis_env/README.md)). For a workshop the TUI also runs in a
browser: `textual serve "utils/jobmon.py /data/c1 --tui"` (package `textual-dev`).

```bash
utils/jobmon.py /data/c1                  # OUTBASE: its gpu0/, gpu1/, ... are taken
utils/jobmon.py /data/c1 --njobs 100      # count the jobs not started yet too
utils/jobmon.py /data/c1 --watch          # live view, every 10 s
utils/jobmon.py /data/c1 --tui            # terminal UI: click a column to sort, f filters,
                                          # a job's details and live log, problems, warnings
utils/jobmon.py /data/c1 --all            # every job; exit code 1 if one failed or is flagged
```

A job counts as running while its driver process is alive (on the machine that runs the
jobs, containers included); on another machine, while its log changed in the last 15 min
(`--stale`). The drivers' `event … done in` lines reach the log in blocks, so a job's
event count jumps by ~10–15 at a time; `PYTHONUNBUFFERED=1` in the jobs' environment avoids it.

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

**On a SLURM CPU partition:** [`utils/slurm_hadronize.sh`](../utils/slurm_hadronize.sh). Task
`t` hadronizes `WORK/CAMPAIGN/t<ttt>/`, the directory of stage-1 task `t`, so the array of
stage 1 covers the campaign; one process per core (`--cpus-per-task=16 --mem=32G` by
default):

```bash
SIF=$PWD/xscape_prod.sif CAMPAIGN=AuAu_a sbatch --array=0-19 utils/slurm_hadronize.sh
SIF=$PWD/xscape_prod.sif CAMPAIGN=AuAu_a \
  HAD_ARGS="--oversample 400 --oversample-bg per-pthat-bin --n-frag 50 --keep-bits-p 12 --keep-bits-x 8" \
  sbatch --array=0-19 utils/slurm_hadronize.sh
```

`HAD_ARGS` (default `--oversample 400 --n-frag 50 --keep-bits-p 12 --keep-bits-x 8`) must be
the same for every task and resubmission of a campaign. `PROD=oo` hadronizes the O+O bulk
(`--tags bulk_bg`); `INPUTS="/work/…"` takes other inputs. All settings:
`BuildContainerProd.md`, *SLURM*.

One process does ~340 events/h at 500 oversamples (10.6 s per event for both legs, 8.8 s
with `--correlated`; iSS Part C), so ~6–7 processes keep up with a stage 1 of 2,000 events/h.
Stage 2 can also run next to stage 1 with `--follow`, but on a host-bound GPU machine its
cores then come out of stage 1's rate.

### Hadrons only: one job, no stage 2 (`--hadrons-only`)

For campaigns with **little or no oversampling** (1–10 samples per jet event) the hadrons
can be made in the GPU job itself, as O+O does: `run_prod_jet.py --hadrons-only` runs iSS on
both legs and the Colorless jet hadronization in the job and writes **only hadron files**,
no pair file, no particlize file, no stage 2
([`PLAN_AuAu_hadrons_only.md`](Plans/PLAN_AuAu_hadrons_only.md)):

```bash
# Docker, inside the image as in §2: 4 jets per background, one sample per jet event
./run_jobs.sh -j 8 --mps 60 45 0 /work/AuAu_ho --reuse 4 --hadrons-only
# SLURM
CAMPAIGN=AuAu_ho EXTRA_ARGS="--pthat-bins 10-20,20-30,30-40 --jets-per-bin 5 \
  --parton-ymax 0.6 --hadrons-only" sbatch --array=0-19 utils/slurm_prod_array.sh
```

Per job: `<stem>_hadrons_{bulk_jet,bulk_bg,jet_frag}.h5`, the undecayed bulk hadrons
`<stem>_hadrons_{bulk_jet,bulk_bg}_primary.h5` (for SMASH later; `--keep-primaries 0`
switches them off) and `<stem>.json`. The jet file also keeps the final partons
(`partons/`), so the jets can be hadronized again later with `hadronize.py` on the jet file
(`--check` replays it). Options are `hadronize.py`'s: `--oversample` (default 1),
`--oversample-bg` (default `auto`, `per-pthat-bin` with `--pthat-bins`), `--n-frag`
(default 1), `--keep-primaries N|all` (default all), `--correlated`, `--eta-max`,
`--charged`, `--no-x`, `--hadron-keep-bits-p/-x`, `--hadronize-seed`.

- **The same hadrons as two stages:** the hadron files equal `hadronize.py`'s on a
  particlize run of the same seed, with the same options (`--seed-uuid` set to the job's
  `source_uuid`), bit for bit, primaries included; the jet sample of a seed does not change.
- **What is lost:** no surfaces are kept, so no new samples or other iSS settings later; the
  bulk can only be re-decayed or sent through SMASH from its primaries. No hydro history
  (no FNO training data). Jet-hadron positions for SMASH need hadronization points the
  files do not have yet (PLAN_smash_ready_hadrons.md 2.2).
- **Cost** (GB10, 4 jobs through MPS, 4 × 30 events of the §4 layout): **~4 % fewer
  events/h** than `--particlize-only` (937 against 977; hadronization ~2 s per event, the
  particlize file not written), **+0.4 GB per job** (6.98 against 6.60 GB), **no stage 2**
  (6.4 core-s per event at 400 samples). The same with 10 samples as with 1. **Disk**:
  0.64 MB per event at the defaults (1 sample, primaries included), 5.4 MB at 10 samples,
  against 92 MB (particlize) + 69 MB (stage-2 hadrons, 400 samples, rounded) = 160 MB per
  event in two stages ([`PLAN_AuAu_hadrons_only.md`](Plans/PLAN_AuAu_hadrons_only.md),
  *Production scale*).
- **Reading:** `HadronFileReader` and `run_h5toROOT.py` take the hadron files alone (the
  event record is in the jet files' `units/`).

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

**With the current image** (`cu124-final_optim`) the same setting runs at **2,160 events/h** on
2 × RTX 3090 (10 jobs per GPU, 15-event jobs) and writes 0.243 GB per event (pair +
particlize, 64 × 64 × 32 grid). c1's GB10 is GPU-bound and not re-measured; the estimate there
is ~750–800 events/h ([`REPORT_SUMMARY_CLUSTER.md`](REPORT_SUMMARY_CLUSTER.md)).

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

**On one machine** (Docker; Apptainer the same with `--pwd` and `--bind`), the c1 setting
with the launch settings of §2. c1 itself ran `-j 4 --mps` with 5 threads per job on the
GB10; size `-j` to the machine (§1):

```bash
docker run -d --name c1 --gpus all --user "$(id -u):$(id -g)" \
  -e OMP_NUM_THREADS=3 -e OMP_WAIT_POLICY=passive -e MUSIC_CUDA_SYNC=block \
  -e LBT_TABLE_CACHE=/work/lbt_cache/lbt_tables_v1.bin -v "$HOME/prod:/work" $IMG \
  ./run_jobs.sh -j 6 --mps --stagger 12 --campaign pth10-40_eta06_c1 67 15 0 \
    /work/AuAu_0_10_pth10-40_eta06_c1 \
    --pthat-bins 10-20,20-30,30-40 --jets-per-bin 5 --parton-ymax 0.6 --write-particlize both
```

**On a two-GPU workstation** (2 × RTX 3090): `utils/launch_2gpu.sh` with 10 jobs per GPU,
the commands in §2 ([Several GPUs](#several-gpus-in-one-machine-or-allocation)).

**On an HPC cluster** (SLURM array, one GPU per task; `P = 10` jobs per GPU on 20 cores, as
for an A100 or A40 in §1):

```bash
SIF=$PWD/xscape_prod.sif CAMPAIGN=pth10-40_eta06_c2 NJOBS=40 EVENTS=15 \
  EXTRA_ARGS="--pthat-bins 10-20,20-30,30-40 --jets-per-bin 5 --parton-ymax 0.6 --write-particlize both" \
  sbatch --array=0-19 utils/slurm_prod_array.sh
```

That is 20 tasks × 40 jobs × 15 events = **12 000 events on 800 backgrounds**. Planning
numbers, at the 3090's measured rate per GPU (1,080 events/h; §1 has the other GPUs):

| | per task (40 jobs, 600 events) | campaign (20 tasks) |
|---|---|---|
| GPU time | ~0.6 h | ~11 GPU-hours, ~0.6 h wall on 20 GPUs |
| disk, stage 1 (0.243 GB per event) | ~146 GB | ~2.9 TB |
| disk, hadrons (stage 2 below) | ~46 GB | ~0.9 TB |

- **Time limit:** a task's jobs run 10 at a time, ~6–7 min per 15-event job at ~25 s per
  event. Allow 2× on V100s (`--time` in the script is 24 h). A task that runs out of time
  resumes when submitted again. Longer jobs (30 or 45 events) amortise the start-up better.
- **Size:** per task 10 jobs × ~7 GB host memory on 20 cores: the script's defaults
  (`--cpus-per-task=20 --mem=96G`, with headroom for SLURM's page-cache accounting).
- **Smaller trial first:** `NJOBS=2 EVENTS=15 P=2 sbatch --array=0-1 --time=2:00:00 …`.

### Stage 2

The background of each window is shared by M = 5 jets, so sample it more:
`--oversample-bg per-pthat-bin` gives each background --oversample × M samples per window
(capped at 2000), the optimal split for per-window results. The c1 plan:

```bash
# one node, the whole campaign
apptainer exec --env OMP_NUM_THREADS=1 --pwd "$PROD" --bind "$SCRATCH/prod:/work" \
  xscape_prod.sif ./run_hadronize.py "/work/pth10-40_eta06_c2/t*/*_particlize.h5" -j 16 \
  --oversample 400 --oversample-bg per-pthat-bin --n-frag 50 --keep-bits-p 12 --keep-bits-x 8

# SLURM: one CPU task per stage-1 task
SIF=$PWD/xscape_prod.sif CAMPAIGN=pth10-40_eta06_c2 \
  HAD_ARGS="--oversample 400 --oversample-bg per-pthat-bin --n-frag 50 --keep-bits-p 12 --keep-bits-x 8" \
  sbatch --array=0-19 utils/slurm_hadronize.sh
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

> **The full analysis guide is [`README_analysis.md`](README_analysis.md):** the
> environments (venv, ROOT, FastJet), reading the hydro, particlize and hadron files, jets
> with FastJet in Python and in C++/ROOT, the notebooks, threads and memory, and
> troubleshooting. This section is the short version.

The analysis doesn't need X-SCAPE, the container, a GPU, conda or ROOT: a Python ≥ 3.10 venv
with this checkout reads every file
([`utils/analysis_env/README.md`](../utils/analysis_env/README.md)):

```bash
git clone https://github.com/jhputschke/js-contrib.git && cd js-contrib
./utils/analysis_env/setup_analysis_env.sh            # venv in ~/.venvs/js_analysis
source ~/.venvs/js_analysis/bin/activate
python utils/analysis_env/check_env.py /path/to/AuAu_a   # opens every file of the directory
```

The venv reads files in place over `osdf://` (pelicanfs, by default; `--no-pelican` skips
it), `--with-gcs` adds `gs://`; `--kernel NAME` registers a Jupyter kernel.

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
| `kernel launch … failed: no kernel image is available` | the image lacks this GPU's architecture: on a V100, an image without sm_70 (built before 2026-09-30 19:57 EDT, including the published `cu126`); use an image of 2026-10-14 or later (`cu124`, `cu126`; §1); discard that output |
| `No CUDA device found` / jobs much slower | no GPU visible: `--nv` / `--gpus`, `CUDA_VISIBLE_DEVICES`, or per-campaign MPS on several GPUs (§2) |
| `--mps: could not start the MPS daemon` | MPS doesn't work in this container/site: drop `--mps` |
| jobs killed (OOM) | plan ~7 GB host memory per job (images from 2026-10-10 on, with `LBT_TABLE_CACHE`; ~12 GB with `--bulk-info full` or older images, ~22 GB before the memory fixes), plus ~8 GB per node: lower `-j`, use `LBT_TABLE_CACHE` and `--stagger`, or ask for more `--mem` |
| busy-spinning threads, the libgomp warning | `OMP_WAIT_POLICY=passive` before launching |
| `pelican: command not found` | the `cu126`/`cu130` of 2026-09-30 predate Pelican: use the images of 2026-10-14 (§1), or move the files from the host (§5) |
| many `hit_edge` warnings | the output grid is too small for this system: a larger x/y range in the grid YAML (§1, *Output grid*) |
| files with different `arr` shapes in one campaign | jobs ran with different grids (an older image's default vs. the new one): keep one grid per campaign |
| `MUSIC_2 run from tau0: the runs differ in <field>` in every event | the two legs' MUSIC settings differ in that field (e.g. a setting changed for one leg only), so MUSIC_2 can't start from MUSIC_1's state. The output is still right, only slower; make them agree, or `--no-music2-restart` |
| `MUSIC_2 run from tau0: the jet source deposits at step k …` | a droplet before the checkpoint (a `tau_delay` smaller than the one the checkpoint was placed at): harmless, that event runs in full |
| `unrecognized arguments: --no-music2-restart` | the image predates the restart and runs without it: drop the option |
| `unrecognized arguments: --late-dtau` | the image predates the late time step and runs a constant step anyway: drop the option |
| `late step: the freeze-out check spacing changes …` (a warning) | `--late-dtau` doesn't divide 0.1 fm/c into whole steps: the frames leave the 0.1 fm/c grid. Use 0.05 or 0.025 |
| one job's events have far fewer droplets than the rest of the campaign | the rare failure of [Checking a run](#checking-a-run): redo that job |
| the GPUs stay ~60 % busy whatever `-j` | expected with the current image: the cores per GPU set the rate (§1). More cores per GPU help, more jobs don't |
| a failed job | its `OUTDIR/<stem>.log`; rerun the same command (finished jobs are skipped) |

## 8. O+O with initial-state radiation (`prod_OO_isr`)

The image also holds [`prod_OO_isr`](../contribs/PyJetscape/example/prod_OO_isr/README.md):
O+O 200 GeV with ISR (PythiaIsrGun + iMATTER), one MUSIC run per event, Matter in the medium,
and the jet hadrons (iColoredHadronization) made in the job; the bulk hadrons in the job
(`--bulk-hadrons N`) or afterwards from the stored surfaces (`hadronize.py --tags bulk_bg`).
There is no two-stage hydro and no pair file.

**Image.** The current images (`cu124`, `cu126`, `cu130`, built 2026-10-14 or later, §1) have
the folder and the two O+O fixes that came after `cu124-final_optim`:
- **Matter's qhat lookup before t = 0** (X-SCAPE `8c975d37`): with ISR, partons start at
  t < 0 and read a garbage qhat (~3,800 lookups per event), which made runs irreproducible
  (events skipped differently from run to run, an occasional crash). Au+Au has no ISR and is
  bit-identical with the fix.
- **iSS's memory leak** (`50226a7c`, iSS `98deaff`: 19,272 leaked blocks per event, matters
  for long jobs with `--bulk-hadrons`).

So use them for O+O campaigns; there is no image to build. The O+O throughput was measured
with `cu124-final_optim`, whose hydro code they share: **13,728 events/h** with hadron files
only on 2 × RTX 3090 (`utils/launch_2gpu_oo.sh`: 8 jobs per GPU, 3 threads,
`MUSIC_CUDA_SYNC=block`, MPS; ±10.5 fm MUSIC grid, the late time step). The images of
2026-09-30 and 2026-10-03 don't have the folder.

**Working directory.** The image starts in `prod_AuAu_0_10_jet`; select the O+O folder with
`-w` (Docker) or `--pwd` (Apptainer), or call its scripts by their full path:

```bash
OO=/opt/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape/example/prod_OO_isr

# Docker: 32 jobs x 60 events, 8 at a time on one GPU, all hadrons made in the jobs,
# nothing else kept (hadron files only)
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  -e OMP_NUM_THREADS=3 -e OMP_WAIT_POLICY=passive -e MUSIC_CUDA_SYNC=block \
  -v "$HOME/prod:/work" -w "$OO" $IMG \
  ./run_jobs.sh -j 8 --mps 32 60 0 /work/OO_a --bulk-hadrons 100 --no-particlize --no-hydro-file

# Apptainer: the same
apptainer exec --nv --env OMP_NUM_THREADS=3,OMP_WAIT_POLICY=passive,MUSIC_CUDA_SYNC=block \
  --pwd "$OO" --bind "$SCRATCH/prod:/work" xscape_prod.sif \
  ./run_jobs.sh -j 8 --mps 32 60 0 /work/OO_a --bulk-hadrons 100 --no-particlize --no-hydro-file

# Stage 2 on CPUs, only if the bulk was left for later (no --bulk-hadrons, particlize kept)
apptainer exec --pwd "$OO" --bind "$SCRATCH/prod:/work" xscape_prod.sif \
  ../prod_AuAu_0_10_jet/run_hadronize.py /work/OO_a -j 8 --tags bulk_bg --oversample 100
```

Everything of §2 applies (output under `/work`, `--user`, the launch settings, MPS, one
container per GPU). An O+O job needs ~1.4 GB of host memory and ~6 core-s per event (with
10 oversamples in the job), so the cores set the rate there too: ~570 events/h per Zen 2
core on the 3090 box. `LBT_TABLE_CACHE` and `--stagger` don't apply (no LBT, small jobs).
The late time step is on in this XML as well (`--late-dtau 0` turns it off); it makes the O+O
freeze-out surface 0.2 % larger, so keep one setting per campaign (§2). Always
`--tags bulk_bg` in stage 2: the jet hadrons were made in the job.
The options, outputs and physics are in the folder's
[README](../contribs/PyJetscape/example/prod_OO_isr/README.md); the analysis of the files in
[`README_OO.md`](README_OO.md). On SLURM: `PROD=oo` for `slurm_prod_array.sh` and
`slurm_hadronize.sh` (§2, §3).

## 9. Another collision system: Pb+Pb at the LHC

The physics of a production is in its **user XML**. `run_prod_jet.py` takes a copy of
`AuAu_MCGlauber_MUSIC_0_10_jet.xml` with `--user-xml`, fills in the job (events, seed, the
hard process and its windows, reuse, the surfaces, `--late-dtau`) and checks it. So another
system or energy is a copy of the XML with the collision changed, bound into the container.
This section goes through Pb+Pb at √s_NN = 5.02 TeV, 0–10 %; the same steps apply to other
nuclei and energies.

### Your XML (and grid) from outside the container

The XML doesn't need to be in the image. Keep it on the host, bind its directory into the
container, and give the drivers the path **as the container sees it**:

| | bind | then |
|---|---|---|
| Docker | `-v "$PWD/xml:/xml:ro"` | `--user-xml /xml/PbPb_0_10_jet.xml` |
| Apptainer | `--bind "$PWD/xml:/xml"`; or nothing for a file under `$HOME`, which Apptainer mounts at its own path | `--user-xml /xml/PbPb_0_10_jet.xml`, or `--user-xml $HOME/xml/PbPb_0_10_jet.xml` |
| SLURM (`slurm_prod_array.sh`, `slurm_hadronize.sh`) | `BINDS="$PWD/xml:/xml"` | in `EXTRA_ARGS` (stage 1) or `HAD_ARGS` (stage 2) |

- **The same for every input file:** `--grid` (the output-grid YAML), `--main-xml` (in place of
  X-SCAPE's `jetscape_main.xml`) and `--hadronize-xml` (rarely needed: one `hadronize.xml` serves
  every system, its `eCMforHadronization` follows the collision, see below).
- **Absolute paths.** The drivers resolve a relative path against their working directory
  inside the container, the production folder of the image, not against the host's.
- **The file is read once, when a job starts.** Each job writes the XML it ran next to its
  output (`<stem>.xml`) and stores it in its files (`prod_user_xml`), so the host copy can
  change later without losing what a file was made with. Editing it during a campaign changes
  the jobs that start afterwards: don't.
- **Check first** with `--dry-run`: it reads the XML and the grid inside the container, checks
  them and writes the job XML, without an event. A path the container can't see ends in
  `FileNotFoundError: [Errno 2] No such file or directory: '/xml/…'`: check the bind.

### What `run_prod_jet.py` needs from a user XML

It refuses an XML that breaks the two-stage setup, before any event:

- exactly two `<Hydro><MUSIC>` blocks, named `MUSIC_1` (the background) and `MUSIC_2` (the jet
  leg), in that order. Every MUSIC instance reads the **first** block, so all MUSIC settings
  go there;
- `MUSIC_1`: `<output_evolution_to_memory>1`, `<dump_hydro_only>0` (Matter, LBT and the
  liquefier read its medium);
- `<Preequilibrium><evolutionInMemory>0`, `<Eloss>` with `<AddLiquefier>true`,
  `<Liquefier><CausalLiquefier>` with `<dtau>` = MUSIC's starting `Delta_Tau` (0.02);
- a `<Hard><PythiaGun>` block (`--pthat-min/max`, `--pthat-bins`, `--parton-ymax` fill it in);
- no `<SoftParticlization>`, `<JetHadronization>`, `<Afterburner>` or writers: hadrons come
  from stage 2;
- an output grid (`--grid` YAML) that lies inside MUSIC's grid, which the XML's `<IS>` block
  sets.

The job XML that ran is written next to every output (`<stem>.xml`) and stored in the files
(`prod_user_xml`); the files' `system` attribute is taken from `<IS><MCGlauber>`
(`Pb+Pb 5020 GeV, b in [0.0, 4.95] fm`; js-contrib `final_optim` after `6b8ae36`, older
versions always wrote the Au+Au string).

### The changes for Pb+Pb at 5.02 TeV

| where | Au+Au 200 GeV | Pb+Pb 5.02 TeV | why |
|---|---|---|---|
| `<IS><MCGlauber><projectile>`, `<target>` | `Au` | `Pb` | |
| `<IS><MCGlauber><sqrts>` | `200` | `5020` | what `MCGlauberWrapper` reads (not `mcglauber.input`) |
| `<IS><MCGlauber><b_max>` | `4.7` fm | `4.95` fm | centrality comes from b inside X-SCAPE (`cenMin`/`cenMax` are ignored): the geometric 0–10 %, π b² = 0.1 × 7.7 b |
| `<IS><grid_max_x>`, `<grid_max_y>` | `15` | `20` | MUSIC's grid (134 × 134 cells at 0.3 fm): the fireball is larger and lives longer |
| `<IS><grid_max_z>` | `6` | `8` | MUSIC's η_s range (80 cells at 0.2): the beam rapidity is 8.5 instead of 5.4 |
| `<Hard><PythiaGun><eCM>` | `200` | `5020` | |
| `<JetHadronization><eCMforHadronization>` (`hadronize.xml`) | `200` | `5020` | Colorless closes an open string with a remnant of √s/6 along the beam: with 200 it would carry 33 GeV instead of 837 GeV. **Automatic** since 2026-10-15: `--hadrons-only`, `--validate-inline` and `hadronize.py` (from the files' `prod_user_xml`) take the job's `<PythiaGun><eCM>`, so the one `hadronize.xml` serves every system |
| the pT̂ windows (`--pthat-bins`) | 10–40 GeV | your choice | the cross section falls more slowly at the LHC |

Made from a js-contrib checkout (the image has the same files under
`/opt/X-SCAPE/external_packages/js-contrib/`):

```bash
mkdir -p xml
T=contribs/PyJetscape/example/config
sed -e 's#<projectile>Au</projectile>#<projectile>Pb</projectile>#' \
    -e 's#<target>Au</target>#<target>Pb</target>#' \
    -e 's#<sqrts>200</sqrts>#<sqrts>5020</sqrts>#' \
    -e 's#<b_max>4.7</b_max>#<b_max>4.95</b_max>#' \
    -e 's#<grid_max_x> 15 </grid_max_x>#<grid_max_x> 20 </grid_max_x>#' \
    -e 's#<grid_max_y> 15 </grid_max_y>#<grid_max_y> 20 </grid_max_y>#' \
    -e 's#<grid_max_z> 6  </grid_max_z>#<grid_max_z> 8  </grid_max_z>#' \
    -e 's#<eCM>200</eCM>#<eCM>5020</eCM>#' \
    $T/AuAu_MCGlauber_MUSIC_0_10_jet.xml > xml/PbPb_0_10_jet.xml
diff $T/AuAu_MCGlauber_MUSIC_0_10_jet.xml xml/PbPb_0_10_jet.xml     # 8 lines must differ
```

**The files as tested** are with all the productions' inputs in
[`contribs/PyJetscape/example/config/`](../contribs/PyJetscape/example/config/) (in the image:
`/opt/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape/example/config/`, no bind
needed): `PbPb_0_10_jet.xml` (the output of this recipe, with a header comment that says so;
EOS 91 since 2026-10-15), the larger output grid `grid_PbPb.yaml` (below) and
`PbPb_0_30_ZJet.xml`, the jet XML at 0–30 % (`b_max` 8.57 fm) with a Z+jet `<PythiaGun>` (Z → ee/μμ, `<trigger>Z`;
[`Plans/PLAN_ZJET_CMS10x.md`](Plans/PLAN_ZJET_CMS10x.md), launchers in `utils/zjet/`). Use them
as they are, or copy them as a start for another system.

**Choices to make, not made here:**

- **The tune.** The 3dMCGlauber string and rapidity-loss parameters, MUSIC's η/s(T) and ζ/s(T),
  T_fo and the Matter/LBT/liquefier settings are those of the Au+Au 200 GeV production. None
  is tuned at the LHC. The rapidity loss is interpolated in the beam rapidity
  (`ylossParam4At2` … `At10`); at y_beam = 8.5 `ylossParam4At6` and `At10` decide it, and
  X-SCAPE's own 5.02 TeV p+Pb XMLs use `At10` = 2.30 where this file has 2.95. Check the bulk
  against data before a production: ALICE measures dN_ch/dη(\|η\| < 0.5) = 1943 ± 54 at 0–5 %
  and 1587 ± 47 at 5–10 % (PRL 116 (2016) 222302); v₂, ⟨pT⟩ and the spectra next.
- **Pythia's MPI.** PythiaGun switches `PartonLevel:MPI = on`, and every parton it hands over
  (status 62, above `softMomentumCutoff`) is showered by Matter/LBT and can deposit. At 5.02
  TeV the hard NN collision brings many more such partons than at 200 GeV (18 showers in the
  test event below, 2–12 at RHIC), on top of the bulk the 3dMCGlauber strings already make.
  `PartonLevel:MPI = off` in `<LinesToRead>` (read after PythiaGun's defaults) leaves the hard
  process and its ISR. Decide before the campaign; the files record it in `prod_user_xml`.
- **The output grid.** The default `grid_fno.yaml` (x, y ±12.2 fm, η_s ±4.84) fits inside the
  larger MUSIC grid, but in the test event the jet leg reached e = 0.233 GeV/fm³ on its x/y
  edge, at the edge flag's threshold (0.234): in a campaign some events will be flagged
  (`diag/{bg,jet}_hit_edge`). A larger box with the same cells, inside MUSIC's grid (3.9
  instead of 2.0 MB per frame; files of the two grids don't go into one training set):

  ```yaml
  # xml/grid_PbPb.yaml (example/config/grid_PbPb.yaml)
  grid:
    x:   {min: -15.30625, max: 15.30625, n: 80}    # 0.3875 fm, as grid_fno.yaml
    y:   {min: -15.30625, max: 15.30625, n: 80}
    eta: {min: -6.09375,  max: 6.09375,  n: 40}    # 0.3125
  tau:
    min: 0.5
    dtau: 0.1
    max_ntau: 0
  ```

- **Names.** `TAG_PREFIX` names the files of another system (default `AuAu_0_10_jet_seed`):
  `TAG_PREFIX=PbPb_0_10_seed ./run_jobs.sh --campaign c1 …` writes `PbPb_0_10_c1_seed0001.*`.
  Letters, digits, `.`, `_`, `-`, ending in `_seed`; one per `OUTDIR`. `launch_2gpu.sh` and
  `slurm_prod_array.sh` pass it into the container; it needs js-contrib of 2026-10-16 or
  later in the image (images of 2026-10-14 ignore it: there, put the system into the campaign,
  `--campaign PbPb_c1` → `AuAu_0_10_jet_PbPb_c1_0001.h5`). For single `run_prod_jet.py` runs,
  `--out`. Details: the [`prod_AuAu_0_10_jet` README](../contribs/PyJetscape/example/prod_AuAu_0_10_jet/README.md#campaigns-with-run_jobssh).

### A first event

Always run `--dry-run` first (it checks the XML and the grid and writes the job XML), then a
few events by hand, and look at the grid flags before a campaign:

```bash
# in the container (Apptainer; Docker with -v "$PWD/xml:/xml:ro" -v "$PWD/prod:/work")
apptainer exec --nv --env MUSIC_CUDA_SYNC=block --pwd "$PROD" \
  --bind "$PWD/xml:/xml,$SCRATCH/prod:/work" xscape_prod.sif \
  python run_prod_jet.py --user-xml /xml/PbPb_0_10_jet.xml --events 3 --seed 1 \
    --write-particlize both --outdir /work/PbPb_test --out PbPb_0_10_jet_seed0001.h5 --dry-run
# ... the same without --dry-run, then stage 2 (eCMforHadronization 5020 from the files)
apptainer exec --pwd "$PROD" --bind "$PWD/xml:/xml,$SCRATCH/prod:/work" xscape_prod.sif \
  ./run_hadronize.py /work/PbPb_test -j 3 --oversample 100 --n-frag 20
```

Measured on the GB10 with these files and, for comparison, the Au+Au XML: X-SCAPE and
MUSIC4GPU `final_optim`, one job alone on all 20 cores, `MUSIC_CUDA_SYNC=block`, the LBT
cache; seed 1, one event, PythiaGun pT̂ 50–70 GeV, `--write-particlize both`, the default
output grid; `hadronize.py` with 100 oversamples:

| | Pb+Pb 5.02 TeV | Au+Au 200 GeV |
|---|---|---|
| wall time of the event (both legs) | 38.9 s | 17.1 s |
| peak host memory of the job | 15.7 GB | 6.1 GB |
| freeze-out τ, background / jet leg | 12.6 / 14.0 fm/c | 10.1 / 12.0 fm/c |
| freeze-out cells per leg | 2.55 M | 1.0 M |
| showers, droplets, deposited energy | 18, 74, 336 GeV | 2, 25, 32 GeV |
| e on the output grid's x/y edge, background / jet leg | 0.141 / 0.233 GeV/fm³ (flag at 0.234) | 0.013 / 0.036 |
| MUSIC grid boundary reached | no | no |
| pair file, particlize file | 321 MB, 395 MB | 245 MB, 155 MB |
| `hadronize.py`, both legs | 10 s, 2.3 GB peak | |
| hadron file per leg | 79 MB (0.79 MB per oversample) | 19 MB |
| dN_ch/dη, \|η\| < 0.5, background | 1534 ± 4 (ALICE 0–10 %: ~1765) | 577 ± 2 |
| ⟨pT⟩ of those charged hadrons | 0.705 GeV | |

One event at an unknown b in 0–4.95 fm is a sanity check, not a validation: the multiplicity
is about right without any LHC tuning.

### Running the campaign

The same commands as for Au+Au, with the XML (and the grid) bound and passed after OUTDIR:

```bash
# Docker
docker run -d --gpus all --user "$(id -u):$(id -g)" \
  -e OMP_NUM_THREADS=3 -e OMP_WAIT_POLICY=passive -e MUSIC_CUDA_SYNC=block \
  -e LBT_TABLE_CACHE=/work/lbt_cache/lbt_tables_v1.bin \
  -v "$PWD/xml:/xml:ro" -v "$HOME/prod:/work" $IMG \
  ./run_jobs.sh -j 4 --stagger 12 --campaign PbPb_c1 20 15 0 /work/PbPb_c1 \
    --user-xml /xml/PbPb_0_10_jet.xml --grid /xml/grid_PbPb.yaml \
    --pthat-bins 40-60,60-80,80-100 --jets-per-bin 5 --parton-ymax 0.6 --write-particlize both

# SLURM, stage 1: ~16 GB per job, so fewer jobs per GPU than for Au+Au
SIF=$PWD/xscape_prod.sif CAMPAIGN=PbPb_c1 P=5 BINDS="$PWD/xml:/xml" \
  EXTRA_ARGS="--user-xml /xml/PbPb_0_10_jet.xml --grid /xml/grid_PbPb.yaml --pthat-bins 40-60,60-80,80-100 --jets-per-bin 5 --parton-ymax 0.6 --write-particlize both" \
  sbatch --array=0-19 --cpus-per-task=20 --mem=96G utils/slurm_prod_array.sh
# SLURM, stage 2 (eCMforHadronization 5020 from the files' job XML)
SIF=$PWD/xscape_prod.sif CAMPAIGN=PbPb_c1 \
  HAD_ARGS="--oversample 200 --n-frag 50 --keep-bits-p 12 --keep-bits-x 8" \
  sbatch --array=0-19 --mem=48G utils/slurm_hadronize.sh
```

- **Memory:** plan ~16 GB per job (15.7 GB peak alone, 2.6× Au+Au's; the background's framework
  copy grows with MUSIC's cells and frames). That is 4–5 jobs on 96 GB. An event costs ~2.3×
  an Au+Au event alone, and the GPU work grows with the cells (2.4×): measure events/h with a
  short sweep of `P` on the first node before choosing it.
- **Disk:** ~0.7 GB per event with one background per event (pair + particlize); a background
  shared by 15 jets saves most of the background's share, as for Au+Au. Hadrons ~4× Au+Au per
  oversample: fewer oversamples (200 instead of 400–500) give the same statistics per event.
- **Stage 2 memory:** ~2.3 GB per process at 100 oversamples, more with
  `--oversample-bg auto`: `--mem` ~3 GB × P.
- **Everything else** (seeds, restart, late time step, checks) is as for Au+Au. Hydro-only
  Pb+Pb (`prod_AuAu_0_10/run_prod.py`) takes the same `<IS>` changes in a copy of its XML.
