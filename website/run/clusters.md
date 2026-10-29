# Clusters, GPUs and monitoring

A production job keeps a GPU only partly busy: much of an event is host work (string
deposition, Matter and LBT, the freeze-out search, writing). So a GPU needs several jobs at
once, and **the cores and host memory per GPU, not the GPU, usually set the rate**. This page
covers running at scale with the production container: the launch settings, SLURM arrays
with Apptainer, several GPUs in one machine, watching a campaign with `jobmon.py`, and what to
do when a run goes wrong. The single-machine commands are on the production pages
([Au+Au jets](auau-jet.md), [O+O](oo-isr.md)).

## Sizing: jobs, cores, memory

For the Au+Au jet production with the current code, per job:

| | |
|---|---|
| host memory | **plan 7 GB per job** (5 GB with `--particlize-only`), plus ~8 GB per node; Pb+Pb ~16 GB |
| cores | ~1 core busy on average, with OpenMP bursts: **2–3 threads per job** |
| GPU memory | ~1 GB per job: a 16 GB V100 holds 8–10 jobs |
| disk | 0.243 GB per event with pair + particlize files |

An O+O job needs ~1.5 GB and, with 10 oversamples in the job, ~6 core-s per event. So set the
number of jobs per GPU from the cores and memory a GPU gets: about cores / 2–3, and about
(memory − 8 GB) / 7 GB. Lower the number of jobs where a node gives a GPU less, not the
threads per job below 2. The measured and extrapolated rates per GPU type (V100, A40, A100,
L40S, RTX 3090, GB10, M3 Max) are in [GPU acceleration](../about/gpu.md).

**Before a large campaign**, on the first node: a 20–30 min sweep of the jobs per GPU (e.g.
8, 12, 16) at the planned cores. Record events/h by makespan, the GPU use
(`nvidia-smi dmon -s u`) and the memory peak. If the GPU stays below ~75 % busy while the
cores are full, more cores per GPU pay, more jobs don't.

## Launch settings

None of these changes the output; together they decide how many jobs fit and how fast they
run.

| setting | how | why |
|---|---|---|
| jobs per GPU | `run_jobs.sh -j P` | the GPU needs several jobs to stay busy (10 on an RTX 3090 with 12 cores per GPU) |
| threads per job | `OMP_NUM_THREADS` ≈ cores per GPU / P, at least 2 | more threads than cores oversubscribe them |
| `OMP_WAIT_POLICY=passive` | environment | idle OpenMP threads sleep instead of spinning |
| `MUSIC_CUDA_SYNC=block` | environment | the host sleeps while it waits for the GPU: ~30 % less CPU per event at the same throughput |
| `LBT_TABLE_CACHE=FILE` | environment; FILE in a bound, writable directory | all jobs map one read-only copy of LBT's 0.66 GB of tables: −0.6 GB and 6–11 s of start-up per job. The first job writes it |
| `--stagger S` | `run_jobs.sh` option | the first P jobs start S s apart, so their memory peaks don't coincide (12 for 8–10 jobs per GPU) |
| `--mps` | `run_jobs.sh` option, CUDA only | the jobs share the GPU through CUDA MPS: +5–12 % where it works |

!!! warning "CUDA MPS in containers"
    MPS is not assured inside containers (on the GB10 the daemon does not start in one) and
    has never been tried on a cluster: run without it first, and test with
    `nvidia-cuda-mps-control -d; echo $?` inside the container. Two daemons in one container
    hang the second GPU's jobs. If the site runs MPS itself (`--gres=mps`), don't add `--mps`.

In Docker pass the variables with `-e`, in Apptainer with `--env A=1,B=2`. On a Mac the same
rule holds without MPS: `OMP_NUM_THREADS=5 OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0
./run_jobs.sh -j 3 …` on a 16-core M3 Max.

## SLURM: one GPU per array task

[`utils/slurm_prod_array.sh`](https://github.com/jhputschke/js-contrib/blob/main/utils/slurm_prod_array.sh)
runs a campaign as a job array in the production container: each task gets one GPU and runs
`P` jobs on it with the launch settings above. SLURM gives whole GPUs, and only jobs in one
allocation can share a GPU, so one SLURM job per production job would leave the GPUs mostly
idle.

**1. Pull the image** once, on a login node (a `.sif` holds one CPU architecture; on x86
login nodes with arm64 GPU nodes such as GH200, add `--arch arm64`):

```bash
apptainer pull xscape_prod.sif docker://jhputschke/xscape-prod:cu126-<date>-<commit>
```

**2. Make the LBT table cache once** with one 1-event job (otherwise 20 tasks starting
together each write their own copy; add your site's `-p`/`-A`):

```bash
PROD=/opt/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape/example/prod_AuAu_0_10_jet
WORK=$SCRATCH/xscape_prod; mkdir -p $WORK/lbt_cache
srun --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=0:20:00 \
  apptainer exec --nv --env LBT_TABLE_CACHE=/work/lbt_cache/lbt_tables_v1.bin \
  --pwd "$PROD" --bind "$WORK:/work" xscape_prod.sif \
  python run_prod_jet.py --events 1 --seed 1 --outdir /work/lbt_cache/init \
    --seed-registry none 2>&1 | grep "LBT:"
rm -r $WORK/lbt_cache/init                 # the test event, not needed
```

The `grep` shows `LBT: wrote the table cache /work/lbt_cache/lbt_tables_v1.bin`. Where arm64
and amd64 nodes share a file system, keep one file per architecture
(`LBT_CACHE=/work/lbt_cache/lbt_tables_v1_arm64.bin`).

**3. A trial, then the campaign** (from a js-contrib checkout):

```bash
# trial: 2 tasks, 2 jobs of 15 events each
SIF=$PWD/xscape_prod.sif CAMPAIGN=pth10-40_eta06_c2 NJOBS=2 EVENTS=15 P=2 \
  EXTRA_ARGS="--pthat-bins 10-20,20-30,30-40 --jets-per-bin 5 --parton-ymax 0.6 --write-particlize both" \
  sbatch --array=0-1 --time=2:00:00 utils/slurm_prod_array.sh

# the campaign: 20 tasks x 40 jobs x 15 events = 12,000 events on 800 backgrounds
SIF=$PWD/xscape_prod.sif CAMPAIGN=pth10-40_eta06_c2 NJOBS=40 EVENTS=15 \
  EXTRA_ARGS="--pthat-bins 10-20,20-30,30-40 --jets-per-bin 5 --parton-ymax 0.6 --write-particlize both" \
  sbatch --array=0-19 utils/slurm_prod_array.sh
```

The defaults, `P=10` jobs per GPU with `--gres=gpu:1 --cpus-per-task=20 --mem=96G`, are an
A100's or A40's share of a node; for other GPUs pass `P` and the `sbatch` options (e.g. V100s:
`P=8 … sbatch --cpus-per-task=16 --mem=64G`). The script sets `OMP_NUM_THREADS = cpus / P`
itself (and warns below 2). At the 3090's measured rate, that 12,000-event campaign is ~11
GPU-hours, ~2.9 TB of stage-1 files and ~0.9 TB of hadrons.

| variable | default | |
|---|---|---|
| `SIF` | `$HOME/xscape_prod.sif` | the image |
| `WORK` | `$SCRATCH/xscape_prod` | bound to `/work`; writable, shared by the tasks |
| `PROD` | `auau` | `oo` runs `prod_OO_isr` |
| `CAMPAIGN`, `NJOBS`, `EVENTS`, `P` | | campaign name, jobs per task, events per job, jobs at once per GPU |
| `EXTRA_ARGS` | | options for the driver (`run_prod_jet.py` / `run_prod_oo_isr.py`) |
| `BINDS` | | more binds, e.g. `"$PWD/xml:/xml"` for your own XML |
| `TAG_PREFIX` | the production's | file names, e.g. `PbPb_0_30_seed` |
| `SEED_MODE` | `registry` | `ranges` with `SEED_BASE`: task t runs seeds `SEED_BASE + t·NJOBS …`, no cross-node lock needed |
| `MUSIC_CUDA_SYNC`, `LBT_CACHE`, `STAGGER`, `USE_MPS` | `block`, `/work/lbt_cache/lbt_tables_v1.bin`, 12, 0 | the launch settings (`LBT_CACHE=none`, `STAGGER=0` to turn off) |
| `CHECK_DEPOSITION` | 1 | after the jobs, flag jobs whose droplets per event fall below half the median |

- **Output:** `WORK/CAMPAIGN/t000/`, `t001/`, … with one shared seed registry
  `WORK/CAMPAIGN/seeds_used.tsv`. The script checks that file locks work across nodes, or use
  `SEED_MODE=ranges`.
- **Restarting:** submit the same command again; each task skips its finished jobs. A task
  that runs out of time resumes.
- **Time:** at ~25 s per event in a loaded job, a 15-event job takes ~6–7 min; allow 2× on
  V100s. Longer jobs (30 or 45 events) amortise the start-up better.
- **Memory:** SLURM counts the page cache of the HDF5 files being written; if tasks are killed
  near the limit, raise `--mem` by ~10–15 % rather than lowering `P`.
- **Disk:** one fast GPU can write more than an OSDF upload takes (an L40S up to ~0.7 TB/h
  with pair files). Use `--particlize-only` unless the hydro history is needed, and
  `upload_follow.py` ([Data on the OSDF](osdf.md#uploading-while-a-production-runs)).
- **O+O:** `PROD=oo CAMPAIGN=OO_c1 EXTRA_ARGS="--bulk-hadrons 10 --no-particlize --no-hydro-file" sbatch --mem=40G utils/slurm_prod_array.sh`.

### Stage 2 on CPU nodes

[`utils/slurm_hadronize.sh`](https://github.com/jhputschke/js-contrib/blob/main/utils/slurm_hadronize.sh)
hadronizes the campaign with the same image, without a GPU: task *t* runs `run_hadronize.py`
on `WORK/CAMPAIGN/t<ttt>/`, the directory of stage-1 task *t*, one process per core.

```bash
SIF=$PWD/xscape_prod.sif CAMPAIGN=pth10-40_eta06_c2 \
  HAD_ARGS="--oversample 400 --oversample-bg per-pthat-bin --n-frag 50 --keep-bits-p 12 --keep-bits-x 8" \
  sbatch --array=0-19 utils/slurm_hadronize.sh
```

- Defaults: `--cpus-per-task=16 --mem=32G`; `HAD_ARGS` defaults to
  `--oversample 400 --n-frag 50 --keep-bits-p 12 --keep-bits-x 8`. **Keep `HAD_ARGS` the same
  for every task and resubmission.**
- `PROD=oo` hadronizes the O+O bulk (`--tags bulk_bg` is added); `INPUTS="/work/…"` takes other
  inputs (directories, particlize files or quoted globs, as paths inside the container).
- Memory ~1.6–1.7 GB per process: ask for ~2 GB per core. One process does ~340 Au+Au
  events/h at 500 oversamples, so ~6–7 cores keep up with 2,000 events/h of stage 1.

## Several GPUs in one machine

MUSIC always uses the first GPU it sees, and `run_jobs.sh` doesn't assign GPUs: on a machine
with several GPUs, run **one campaign per GPU**, each with its own `CUDA_VISIBLE_DEVICES` (or
its own container), OUTDIR and campaign name. OUTDIRs side by side share the seed registry.

**Ready-made for Docker on two GPUs:**
[`utils/launch_2gpu.sh`](https://github.com/jhputschke/js-contrib/blob/main/utils/launch_2gpu.sh)
starts one container per GPU, each running `run_jobs.sh` with MPS, guards the disk and can
upload every finished job to the OSDF while the campaign runs.
[`utils/launch_2gpu_oo.sh`](https://github.com/jhputschke/js-contrib/blob/main/utils/launch_2gpu_oo.sh)
does the same for O+O. Run them from `utils/` of a js-contrib checkout:

```bash
cd js-contrib/utils
# Au+Au, 10 jobs per GPU: shared LBT tables, staggered start, blocking CUDA sync;
# 60 jobs x 45 events (a multiple of 20 jobs)
IMAGE=jhputschke/xscape-prod:cu126 JOBS_PER_GPU=10 OMP_THREADS=3 MUSIC_CUDA_SYNC=block \
nohup ./launch_2gpu.sh --lbt-cache /work/lbt_cache/lbt_tables_v1.bin --stagger 12 \
    60 45 0 /data/camp_pth3 --campaign pth3 \
    --pthat-bins 10-20,20-30,30-40 --jets-per-bin 5 --parton-ymax 0.6 \
    --write-particlize both > /data/camp_pth3.log 2>&1 &

# O+O, hadron files only (defaults: 8 jobs per GPU, 3 threads, MUSIC_CUDA_SYNC=block, MPS)
nohup ./launch_2gpu_oo.sh 64 100 0 /data/OO_had --campaign had1 \
    --bulk-hadrons 100 --no-particlize --no-hydro-file > /data/OO_had.log 2>&1 &
```

```text
./launch_2gpu.sh [options] NJOBS EVENTS_PER_JOB [FIRST_SEED] [OUTBASE] [driver / run_jobs.sh options]
```

- **NJOBS** is the total: GPU0 gets the first half, GPU1 the rest. Output goes to
  `OUTBASE/gpu0/` and `OUTBASE/gpu1/`, with a shared seed registry. With `FIRST_SEED 0` both
  directories hold files of the same names (different seeds): keep the directories.
- **Options before NJOBS:** `--lbt-cache FILE` (made first if missing, ~1 min), `--stagger S`,
  `--min-free SIZE` (the disk guard: stops both containers below SIZE free, default `50G`),
  and the upload options `--upload REMOTE`, `--delete none|pair|all`, `--keep-free SIZE`,
  `--verify-every N` ([Data on the OSDF](osdf.md#uploading-while-a-production-runs)).
- **Environment:** `IMAGE`, `JOBS_PER_GPU` (default 8), `OMP_THREADS` (default 4),
  `WORKDIR` (mounted as `/work`, default `$HOME/prod_test`), `MUSIC_CUDA_SYNC`, `TAG_PREFIX`.
  The defaults predate the current image: set `JOBS_PER_GPU=10 OMP_THREADS=3` as above.
- **Stop** with Ctrl-C or `kill` (both containers and the uploader stop); **resume** with the
  same command.

Measured on 2 × RTX 3090 (24 cores, 125 GB) with the campaign setting: **2,160 events/h** for
Au+Au at 10 jobs per GPU, 105 GB host memory peak; **13,728 events/h** for O+O with hadron
files only. Details: [`utils/README_launch.md`](https://github.com/jhputschke/js-contrib/blob/main/utils/README_launch.md).

**By hand, or with Apptainer** (one container for all GPUs, no `--mps`):

```bash
apptainer exec --nv --env OMP_NUM_THREADS=3,OMP_WAIT_POLICY=passive,MUSIC_CUDA_SYNC=block \
  --env LBT_TABLE_CACHE=/work/lbt_cache/lbt_tables_v1.bin \
  --pwd "$PROD" --bind "$SCRATCH/prod:/work" xscape_prod.sif bash -c '
  trap "kill \$(jobs -p) 2>/dev/null; wait" INT TERM
  CUDA_VISIBLE_DEVICES=0 ./run_jobs.sh -j 8 --stagger 12 --campaign AuAu_gpu0 24 45 0 /work/AuAu/gpu0 --write-particlize both &
  CUDA_VISIBLE_DEVICES=1 ./run_jobs.sh -j 8 --stagger 12 --campaign AuAu_gpu1 24 45 0 /work/AuAu/gpu1 --write-particlize both &
  wait'
```

Under SLURM, if `CUDA_VISIBLE_DEVICES` is already set (e.g. `2,3`), use those numbers.

## Monitoring a campaign

[`utils/jobmon.py`](https://github.com/jhputschke/js-contrib/blob/main/utils/jobmon.py) reads
what the jobs write (their logs, `.json` summaries and the `run_jobs.*` markers) and shows jobs
done / running / failed, events, events/h, the ETA, disk per event, the failed jobs with their
last log lines, the deposition check, edge hits, skipped events and the most frequent
warnings. It works on Docker, SLURM and Mac output alike, during a run or after.

```bash
utils/jobmon.py /data/c1                  # OUTBASE: its gpu0/, gpu1/, ... (or t000/, ...) are taken
utils/jobmon.py /data/c1 --njobs 100      # count the jobs not started yet too
utils/jobmon.py /data/c1 --watch          # live view, every 10 s
utils/jobmon.py /data/c1 --tui            # terminal UI: sort, filter, a job's details and live log
utils/jobmon.py /data/c1 --all            # every job; exit code 1 if one failed or is flagged
```

It needs only the standard library; `--watch` uses `rich` if installed and `--tui` needs
`textual` (both in the [analysis environment](../install/analysis-env.md)). For a workshop,
the TUI runs in a browser: `textual serve "utils/jobmon.py /data/c1 --tui"` (package
`textual-dev`).

- A job counts as running while its driver process is alive (on the machine that runs the
  jobs, containers included); on another machine, while its log changed in the last 15 min
  (`--stale`).
- The drivers' `event … done in` lines reach the log in blocks, so event counts jump by
  ~10–15 at a time; `PYTHONUNBUFFERED=1` in the jobs' environment avoids it. The s per event
  are exact.

**The deposition check.** Once, in ~250 campaign jobs, all events of one job lost almost no
jet energy (1–5 droplets per event instead of 13–34), with a correct background and no
warning in the log; the same seed alone was normal. Check every campaign for it:

```bash
utils/check_deposition.py /work/AuAu_a                     # every job summary below it
utils/check_deposition.py /data/c1/gpu0 /data/c1/gpu1      # one campaign in two directories
```

It flags a job whose droplets per event fall below half the campaign's median (`--fraction`)
and exits 1 if one is flagged. To redo it, move its files (`<stem>.*`) out of OUTDIR and run
the campaign's command again. `slurm_prod_array.sh` runs it on each task's directory.

## Checking a run

In a job log, `OUTDIR/<stem>.log`:

| line | means |
|---|---|
| `[MUSIC-GPU] CUDA device: <GPU name> (cc X.Y, …)` | the hydro runs on that GPU |
| `[MUSIC-GPU] OMP wait policy: passive (max threads N)` | the thread settings arrived |
| `[MUSIC-GPU] host sync: block` | `MUSIC_CUDA_SYNC=block` arrived |
| `LBT: wrote the table cache …` / `LBT: tables mapped read-only from …` | the LBT cache works (`grep -L "mapped read-only" OUTDIR/*.log` lists jobs without it) |
| `late step: delta_tau 0.02 -> 0.05 fm/c at step 130 …` | MUSIC's late time step is on |
| `restart  MUSIC_2 starts from MUSIC_1's state …` / `restart  OFF: …` | the MUSIC_2 restart, on or off with the reason |
| `… done in … s` per event | the speed: events/h = jobs × 3600 / s per event |
| `No CUDA device found`, `falling back to CPU` | **no GPU**: ~7–10× slower and systematically different (~0.2 %). Don't add that output to a GPU campaign |
| `PairH5Writer: event i: the jet leg has e = … on the transverse edge of the output grid` | fluid above freeze-out reached the stored box's edge (`diag/{bg,jet}_hit_edge`); a few per campaign are expected, frequent ones mean the grid is too small |

Each job also writes `<stem>.json` (seed, events written, settings, `events_hit_edge`,
`droplets_total`), and a campaign ends with `OUTDIR/run_jobs.finished`.

## Troubleshooting

| symptom | cause and fix |
|---|---|
| `./run_jobs.sh: No such file` (Apptainer) | no `--pwd "$PROD"`; or call `$PROD/run_jobs.sh` |
| `--outdir: give OUTDIR as the argument after FIRST_SEED` | OUTDIR is the 4th argument of `run_jobs.sh`, not an option |
| `Read-only file system: …/seeds_used.tsv` | OUTDIR's parent isn't writable: put OUTDIR one level inside a bound directory (`/work/out`) |
| only one job runs with `-j 2` | NJOBS is 1: `-j` only limits how many run at once |
| `No CUDA device found` / jobs much slower | no GPU visible: `--nv` / `--gpus`, `CUDA_VISIBLE_DEVICES`, or per-campaign MPS on several GPUs |
| `kernel launch … failed: no kernel image is available` | the image lacks this GPU's architecture (e.g. V100 with a `cu130` image): use `cu124` or `cu126`; discard that output |
| `--mps: could not start the MPS daemon` | MPS doesn't work in this container or site: drop `--mps` |
| jobs killed (OOM) | plan ~7 GB per Au+Au job plus ~8 GB per node: lower `-j`, use `LBT_TABLE_CACHE` and `--stagger`, or ask for more `--mem` |
| busy-spinning threads, the libgomp warning | `OMP_WAIT_POLICY=passive` before launching |
| the GPUs stay ~60 % busy whatever `-j` | expected: the cores per GPU set the rate. More cores per GPU help, more jobs don't |
| one job's events have far fewer droplets than the rest | the rare failure of the deposition check: redo that job |
| `unrecognized arguments: --late-dtau` (or `--no-music2-restart`) | an older image without the option: drop it |
| a failed job | its `OUTDIR/<stem>.log`; rerun the same command (finished jobs are skipped) |

More cases: [`docs/README_2stage.md`, §7](https://github.com/jhputschke/js-contrib/blob/main/docs/README_2stage.md#7-troubleshooting).

## Next steps

- [Data on the OSDF](osdf.md): move the outputs, or upload them while the campaign runs.
- [What a production writes](../analysis/files.md): the files of a finished campaign.
