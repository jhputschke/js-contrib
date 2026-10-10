# prod_AuAu_0_10_jet on an Apple M3 Max: single-job speed-ups and concurrent jobs

Measured 2026-09-25. Questions:
- Do the single-job speed-ups measured on the GB10 ([BENCHMARK_GB10.md](BENCHMARK_GB10.md))
  carry over to Apple silicon (music4gpu on Metal)?
- Does running several jobs at once (`run_jobs.sh -j P`) pay off, and how should it be run?

**Short answer.**
- **The single-job speed-ups carry over, and are larger than on the GB10.** Seed 1 went
  from 38.5 / 44.1 s to **26.1 / 28.7 s per event** (−32 % / −35 %) with the source-fill
  skip and string binning. The GB10 got −17 % from the same changes. Every step gives
  byte-identical output.
- **One job does 132 events/h** (27 s/event), more than the GB10's 118.
- **Concurrent jobs only pay off if the cores are split between them.** With the default
  OpenMP settings, `-j 3` gives 148 events/h and `-j 4` 129, no better than one job. With
  `OMP_NUM_THREADS` = 16 / P and passive OpenMP waiting, `-j 3` gives **213 events/h
  (1.61×)** and `-j 4` **221 (1.67×)**. This is the opposite of the GB10, where splitting
  the threads did not help.
- **Recommendation:**
  ```bash
  conda activate fno_env_mlx
  OMP_NUM_THREADS=5 OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0 ./run_jobs.sh -j 3 NJOBS EVENTS SEED0
  ```
  `-j 4` with `OMP_NUM_THREADS=4` adds ~4 % for ~5 GB more memory (with
  `cuda_blocking_sync` no longer measurable, see the next bullet).
- **`cuda_blocking_sync`** (2026-10-10, [below](#cuda_blocking_sync-on-the-m3-max-2026-10-10)):
  no negative effect on the Mac. One job: −10.5 % per event, −57 % CPU, bit-identical
  output. With `LBT_TABLE_CACHE`, 11.5 s less startup. `-j 3`: 235 → **284 events/h**.
  `-j 4` did not beat `-j 3` on the branch. Short `-j 4` runs vary by ~±15 %, so it is
  open whether the branch changes `-j 4` and whether `--stagger` helps it.
- **`run_jobs.sh` did not run on macOS.** It used bash ≥ 5.1 features, and macOS ships
  bash 3.2; every seed was reported as FAILED without running. Fixed in the same commit as
  this file (see [below](#run_jobssh-on-macos)). `--mps` is CUDA-only; on the Mac it stops
  with "nvidia-cuda-mps-control not found".

## Setup

- **Machine:** Apple M3 Max, 16 cores (12 performance + 4 efficiency), 64 GB unified
  memory. The usual desktop applications were open (editor, browser, sometimes a video
  call): about 14 GB of memory and up to ~1 core.
- **Build:** `build_gpu` (music4gpu, Metal), env `fno_env_mlx`, Homebrew libomp.
- **Code:** X-SCAPE `contrib` `ca5aed17`, MUSIC4GPU `XSCAPE` `566cea6`, js-contrib `main`
  `e834d4c` + the `run_jobs.sh` fix. The single-job steps below were measured on the
  states named in their table.
- **Jobs:** `run_prod_jet.py --events N --seed S` with the defaults: PythiaGun 50–70 GeV,
  `--surface none`, `grid_fno.yaml`, `--reuse 1`.
- **Seed 1 on the Mac is not the GB10's seed 1:** it gives 15 and 9 droplets and 101/100
  and 105/103 jet/background frames. Compare the two machines in percent, not seconds.
- **Numbers:** as in BENCHMARK_GB10.md, *s/event* is the mean of the per-event wall times
  the driver prints (no startup), and *events/h* is Σ over jobs of 3600 / (that job's mean
  s/event).

## Single-job speed-ups

Seed 1, 2 events per run, each state run twice with the states interleaved. The
per-event values are the means of the two runs.

| Build | event 1 | event 2 | Output |
|---|---|---|---|
| `hydro_data_optim` (resample, `bulk_info` copy) | 38.5 s | 44.1 s | — |
| + `cpu_source_fill_optim` (MUSIC4GPU `6d33feb`, X-SCAPE `75b4ce7a`) | 34.5 s | 40.3 s | byte-identical |
| + `string_bin_optim` (MUSIC4GPU `fa6ec5b`) | **26.1 s** | **28.7 s** | byte-identical |
| + `jet_source_dtau` (X-SCAPE `aa1f5d50`), StringFind4 guard (MUSIC4GPU `4f9533b`), per-job working directory (js-contrib `fa36b11`) | 25.5 s | 28.6 s | byte-identical |

The last row is one run against a reference run of the row above it (25.4 / 28.1 s):
no measurable change, as expected.

Where the time goes, from timestamped logs, seconds per event, both legs together:

| Stage | `hydro_data_optim` | + source-fill skip | + string binning |
|---|---|---|---|
| The two string-deposition steps | 16.3 | 15.5 | **5.5** |
| The rest of the evolution | 17.7 | **14.6** | 14.5 |
| `bulk_info` copy | 0.6 | 0.6 | 0.6 |
| After the jet leg (writer: resampling, checks, h5) | 6.5 | 6.5 | 6.5 |

- **String binning saves 10.0 s per event on the Mac**, more than twice the GB10's
  4.2 s (string evaluation 7.9 → 3.7 s per event there).
- **The source-fill skip saves 3.1 s**, in the steps after the strings are deposited.
- **The earlier `hydro_data_optim` changes**, measured the same way:
  - the separable resampling (js-contrib `e3101fd`) cuts 51.3 / 58.3 s to 37.7 / 44.7 s
    per event (−13.6 s), byte-identical;
  - the parallel `bulk_info` fill (X-SCAPE `a80a9932`) cuts the copy from 2.5 s to 0.5 s
    per event.

## Concurrent jobs

`run_jobs.sh -j P` with P jobs of 3 events, seeds 1 … P, all started at the same moment,
each in its own working directory. One run per configuration, so allow about ±5 %.

| Jobs at once | OpenMP settings | s/event per job | events/h | vs 1 job | GPU busy (mean) | GPU idle samples (< 10 %) | cores busy (mean) | memory in use (max) |
|---|---|---|---|---|---|---|---|---|
| 1 | default (12 threads) | 27.2 | 132 | 1.00× | 50 % | 37 % | 5.6 | 32 GB |
| 1 | passive | 26.0 | 138 | 1.05× | 51 % | 35 % | 5.4 | 28 GB |
| 2 | default | 41.5 | 174 | 1.31× | 52 % | 31 % | 9.3 | 36 GB |
| 2 | passive | 42.6 | 169 | 1.28× | 54 % | 29 % | 9.4 | 35 GB |
| 2 | passive, 8 threads | 35.5 | **203** | **1.53×** | 69 % | 18 % | 7.0 | 34 GB |
| 3 | default | 73.6 | 148 | 1.12× | 50 % | 35 % | 11.5 | 43 GB |
| 3 | passive | 81.8 | 132 | 1.00× | 51 % | 31 % | 11.7 | 43 GB |
| 3 | 5 threads | 55.5 | 195 | 1.47× | 71 % | 5 % | 7.2 | 39 GB |
| 3 | passive, 5 threads | 50.7 | **213** | **1.61×** | 70 % | 18 % | 7.5 | 40 GB |
| 4 | passive | 112.8 | 129 | 0.97× | 45 % | 45 % | 12.6 | 48 GB |
| 4 | passive, 4 threads | 65.3 | **221** | **1.67×** | 73 % | 14 % | 9.4 | 45 GB |

- *passive* is `OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0`; *N threads* is
  `OMP_NUM_THREADS=N`. The default is 12 threads per job (MUSIC logs
  `OpenMP: using 12 threads`).
- *Memory in use* is system-wide (active + wired + compressed pages) and includes the
  ~14 GB of the other applications. No run used swap.
- No job failed or hung, although all jobs of a run started at the same moment.

**Why the default settings fail.**
- With 3–4 jobs of 12 threads each, 36–48 threads compete for 16 cores.
- The OpenMP loops (the source fill, the `bulk_info` copy, MUSIC's CPU parts) end in a
  barrier that waits for their slowest thread. With the cores oversubscribed, that is
  often a thread the OS has paused to run another job.
- The cores are then busy (11–13 of 16) but do little useful work, and the GPU stays
  at ~50 %, no busier than with one job.
- **Passive waiting alone does not help** (−3 % at P = 2, −11 % at P = 3): it stops
  idle threads from spinning, but the barriers still wait for descheduled threads.
- **Splitting the cores does help.** With 16 / P threads per job, the GPU rises to
  70–73 % and fewer cores are busy. Passive waiting then adds ~9 % (P = 3: 195 → 213
  events/h).

**How far it goes.**
- `-j 4` is only 4 % above `-j 3`. The GPU reaches ~73 %, so, as on the GB10, it is
  likely becoming the shared bottleneck (not profiled further).
- Memory is not the limit: `-j 4` peaked at 45 GB of 64, including the other
  applications.
- For one job, keep the default thread count (a single job with fewer threads was not
  measured here; on the GB10 it was slower). Passive waiting does not hurt it.

## cuda_blocking_sync on the M3 Max (2026-10-10)

Question: does the branch `cuda_blocking_sync` slow `prod_AuAu_0_10_jet` down on the
Mac, or speed it up? What it changes on the Mac:
- **MUSIC4GPU:** the string source fill computes the per-string constants once per τ step,
  skips strings with a zero envelope, and skips the transverse-flow terms when
  `stringPreEqFlowFactor` = 0. `HydroSourceBase` gets a new virtual method, so libmusic
  and libJetScape must be rebuilt together. `MUSIC_CUDA_SYNC` is CUDA-only and does nothing
  here.
- **X-SCAPE LBT:** the tables live in one block. With `LBT_TABLE_CACHE=<file>` they are
  mapped read-only from that file, which is written on first use.
- **js-contrib:** the jet leg is read one frame at a time; `run_prod.py` sets
  `OPENBLAS_NUM_THREADS=1` unless it is already set; `run_jobs.sh --stagger S`.

**Short answer.**
- **One job:** 10.5 % less time per event, **57 % less CPU**, 1.2 GB less peak RSS
  (0.8 GB less footprint). The output is bit-identical.
- **`LBT_TABLE_CACHE`** cuts the startup from 14.5 to 3.0 s per job, saving 11.5 s rather
  than the ~6.5 s expected, with identical output.
- **`-j 3`** (5 threads, passive): **235 → 284 events/h (+21 %)**; makespan 158.7 →
  123.2 s (−22 %).
- **numpy uses OpenBLAS here, not Accelerate,** so the new one-thread default matters:
  16 OpenBLAS threads make events only 1.7 % faster, for 79 % more CPU.
- **`-j 4` gained nothing on the branch:** by makespan 202, 219 and (staggered) 256
  events/h, against 263 at `-j 3`. Short `-j 4` campaigns vary by ~±15 % between runs of
  the same code (base: 242 and 209), so whether the branch changes `-j 4` is not resolved.
  **`--stagger 10`** at `-j 3` gains nothing.
- **Negative effect on the M3: no.**

**Setup.**
- **Machine:** the same M3 Max. Kept quiet (no video call; editor and terminal open, ~20 GB
  in use when idle). Background daemons (Spotlight, `mediaanalysisd`, `duetexpertd`) were
  active at times. They were logged only in the last two base `-j 4` runs: 2–3 % of a core
  on average, with spikes up to 85 %.
- **Base:** X-SCAPE `contrib` `57e32ce3`, MUSIC4GPU `XSCAPE` `76c4b96`, js-contrib `main`
  `4e22dc6`, `build_gpu` as built for them.
- **Branch:** X-SCAPE `01584cc3`, MUSIC4GPU `c7b75ba`, js-contrib `cd3848e`.
  `cmake --build build_gpu -j 14` with the unchanged `CMakeCache.txt` (Release, Metal,
  ROOT, Unix Makefiles). The dependency tracking recompiled every libmusic source that
  includes `hydro_source_base.h`, `MusicWrapper.cc` and `LBT.cc` in libJetScape, and the
  pybind module (`MpiMusic.get_native_ntau` present). `music_kernels.metallib` was not
  rebuilt (no shader changed) and sits next to the new libmusic.
- **Compiler warnings:** none from the new code. `LBT.cc`'s 11 warnings (unused variables,
  `LBTMutex` destructor) are all on lines from 2018–2020.
- **Single job:** `run_prod_jet.py --events 3 --seed 1 --seed-registry none` under
  `/usr/bin/time -l`, default threads, `LBT_TABLE_CACHE` unset. Base twice, then `new`,
  `blas16`, `new`, `blas16` interleaved; *blas16* is `OPENBLAS_NUM_THREADS=16` on the
  command line. No log contains `MUSIC_METALLIB` (all ran on Metal). *Startup* is `real`
  minus the driver's `wall_s` (imports, framework and LBT initialisation); the log has
  no LBT timestamp.
- **Campaigns:** `bench.sh` from [Reproducing](#reproducing), with one change: an optional
  `EXTRA` before the positional arguments (`$RJ -j $P ${=EXTRA} $P 3 1 $D`). 3 jobs × 3
  events, seeds 1–3, `OMP_NUM_THREADS=5 OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0`. The
  branch runs had `LBT_TABLE_CACHE` exported (the file already existed) and ran
  `run_jobs.sh` under macOS's `/bin/bash` 3.2, with
  `PATH=$CONDA_PREFIX/bin:/bin:/usr/bin:…` (`env bash --version` gives 3.2.57; this Mac
  has no Homebrew bash). The base `-j 4` runs were made later, after switching back to the
  base and rebuilding. The staggered base run used the branch's `run_jobs.sh` (a
  shell-only change) to get `--stagger`, with the base driver.

### One job

Means of two runs, except the cache column (one run, after the run that wrote the
cache).

| | base | branch | Δ | branch, 16 OpenBLAS threads | branch + `LBT_TABLE_CACHE` |
|---|---|---|---|---|---|
| s/event (mean) | 23.9 | **21.4** | **−10.5 %** | 21.1 | 21.1 |
| events 1 / 2 / 3 (s) | 23.0 / 26.6 / 22.2 | 21.5 / 23.3 / 19.5 | | 21.0 / 22.8 / 19.4 | 21.2 / 22.8 / 19.3 |
| wall (`real`) | 87.1 s | 78.8 s | −9.5 % | 77.8 s | **66.3 s** |
| startup | 15.3 s | 14.5 s | −5 % | 14.6 s | **3.0 s** |
| CPU (user + sys) | 328 core-s | **142 core-s** | **−56.6 %** | 255 core-s | 126 core-s |
| of which sys | 27.5 s | 2.9 s | | 31.0 s | 2.6 s |
| max resident set | 8.47 GB | 7.24 GB | −14.4 % | 7.59 GB | 6.58 GB |
| peak memory footprint | 6.80 GB | 6.00 GB | −11.7 % | 5.99 GB | 5.28 GB |

- **Per event, −2.5 s.** The event 1 / 2 / 3 times of the two runs agree within 0.7 s.
  The string source fill and the frame-by-frame read were not timed separately.
- **CPU, −186 core-s per 3-event job, in two parts:**
  - Base → *blas16* (both with 16 OpenBLAS threads): −73 core-s (−22 %), from the source
    fill and the frame-by-frame read.
  - *blas16* → branch default (1 thread): another −113 core-s. Idle OpenBLAS workers spin
    between the resampling's small matrix products; sys time falls from 31 to 3 s.
- **Memory:** 0.8 GB less footprint, as expected from reading the jet leg frame by frame.

**BLAS.** `numpy.show_config()` in `fno_env_mlx` shows numpy 2.3.5 built against OpenBLAS
0.3.30. The conda `blas` package is the `openblas` variant, so `libblas.dylib` links to
`libopenblasp`, and threadpoolctl reports libopenblas with pthreads and 16 threads. It is
not Accelerate.
- **One thread instead of 16:** events are 1.7 % slower (21.4 against 21.1 s; the runs are
  21.3–21.5 and 20.9–21.2 s), and the job uses 44 % less CPU.
- **Campaigns:** with `-j 3` and 5 OpenMP threads per job, 16 OpenBLAS threads per job
  would oversubscribe the cores again (not measured). The default of 1 suits the Mac.

**LBT table cache** (`LBT_TABLE_CACHE=$W/lbt_cache/lbt_tables_v1.bin`, on the internal
SSD):
- **First run:** writes 656,581,952 bytes (`LBT: wrote the table cache …`), then maps the
  file (`LBT: tables mapped read-only from …`) in the same run.
- **Second run:** only the mapping line. Startup falls from 14.5 to 3.0 s, so wall time
  falls 12.5 s (−15.9 %).
- **Memory:** footprint is 0.7 GB lower, because the mapped tables are file-backed pages
  that do not count towards it.
- **Output:** identical to the branch run without the cache.

**Output.** The h5 files were compared dataset by dataset with `numpy.array_equal`
(`h5py` + `hdf5plugin`):
- **Pairs compared:** base vs each of the four branch runs; base vs base; the
  cache-writing and cache-reading runs vs a branch run.
- **35 of the 37 datasets are equal** in every pair. The other two:
  - `diag/wall_s` holds the timings, so it differs.
  - `source/flux` differs in 2–4 of 31 values by ≤ 4.4e-16 (≤ 2 ulp). It does so between
    the two base runs too (the OpenMP reduction).

### Concurrent jobs

| Run | P | threads | `--stagger` | makespan | s/event per job | events/h | events/h from makespan | GPU busy (mean) | GPU idle samples | cores busy | memory in use (max) | above idle |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 3 | 5 | — | 158.7 s | 46.0 | 235 | 204 | 70 % | 19 % | 6.6 | 33.3 GB | 13.1 GB |
| branch | 3 | 5 | — | **123.2 s** | **38.1** | **284** | **263** | 83 % | 6 % | 4.3 | 33.7 GB | 12.7 GB |
| branch | 3 | 5 | 10 s | 137.8 s | 38.1 | 284 | 235 | 81 % | 7 % | 4.0 | 33.1 GB | 13.2 GB |
| base | 4 | 4 | — | 178.3 s | 52.6 | 274 | 242 | 73 % | 17 % | 7.6 | 34.1 GB | 13.7 GB |
| base (repeat) | 4 | 4 | — | 206.4 s | 61.7 | 233 | 209 | 76 % | 13 % | 7.9 | 33.7 GB | 14.6 GB |
| base | 4 | 4 | 15 s | 213.6 s | 48.9 | (295) | (202) | 72 % | 15 % | 6.4 | 32.7 GB | 14.5 GB |
| branch | 4 | 4 | — | 214.4 s | 67.8 | 213 | 201 | 86 % | 6 % | 4.8 | 34.1 GB | 13.2 GB |
| branch (repeat) | 4 | 4 | — | 197.0 s | 62.6 | 230 | 219 | 87 % | 5 % | 5.5 | 34.1 GB | 14.2 GB |
| branch | 4 | 4 | 15 s | 168.9 s | 39.5 | (367) | (256) | 82 % | 4 % | 4.3 | 33.0 GB | 13.5 GB |

- **Columns:** all runs are passive (`OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0`), and the
  branch runs use `LBT_TABLE_CACHE`. *events/h from makespan* is 3P events × 3600 /
  makespan, so it includes startup and the tail. *above idle* is the maximum minus the
  first sample of `mon.log`.
- **Base:** today's base gives 235 events/h at `-j 3`, against 213 in the table above
  (older code).
- **Branch at `-j 3`:** +21 % events/h, −22 % makespan. The GPU is busier (70 → 83 %,
  idle samples 19 → 6 %), and fewer cores are busy (6.6 → 4.3).
- **Memory:** the system-wide peak does not change (~13 GB above idle in every run). This
  metric (active + wired + compressed pages) does not show the per-job savings that
  `/usr/bin/time` shows. At 34 of 64 GB, memory is not the limit on this Mac.
- **`--stagger 10`:** the jobs started at 15:57:53, 15:58:03 and 15:58:13
  (`run_jobs.log`). Per-event times and peak memory are the same as without stagger, but
  the makespan is 14.6 s longer. It is no use on the Mac.
- **`-j 4` with 4 threads:**
  - **Spread:** two runs of the same code differ by ~15 % (base: 242 and 209 events/h by
    makespan; branch: 201 and 219). That is three times the ±5 % assumed above.
  - **Branch vs base:** the branch's two unstaggered runs lie within the base's range, so
    this data cannot tell whether the branch changes `-j 4`.
  - **`-j 4` vs `-j 3`:** on the branch, `-j 4` never beat `-j 3` (by makespan 201, 219
    and, staggered, 256 against 263). On the base, `-j 4` was level with `-j 3` or above it
    (242 and 209 against 204), as in the "+4 %" above. The branch lifted `-j 3` to about
    the rate `-j 4` already reached.
  - **GPU time:** GPU-busy seconds per event (`mon.log`, integrated) overlap: base 11.0,
    12.8 and 12.8, branch 15.3, 13.9 and 11.4. So the data do not show a fourth job using
    the GPU less efficiently.
  - **Memory:** not the limit for `-j 4` on the Mac, so freeing memory per job does not
    help it.
- **`--stagger 15` at `-j 4`:** neither measure is fair here, so the values are in
  parentheses.
  - The per-event times are low because fewer than four jobs run during the first 45 s.
  - The makespan carries the 45 s of start delays.
  - With 3-event jobs, all four jobs run together for only 43 s (branch) and 92 s (base).
  - The staggered branch run was the best branch `-j 4` run (256 by makespan), and the
    staggered base run the worst base run (202). With this spread, that is not evidence
    either way.
- **Open:** deciding `-j 4` and `--stagger` needs longer campaigns, e.g. 8 jobs × 3
  events, where slots refill and the jobs drift out of step as in production. They should
  be base and branch interleaved, with the background daemons logged.
- **Recommendation:**
  ```bash
  export LBT_TABLE_CACHE=/path/on/local/disk/lbt_tables_v1.bin
  OMP_NUM_THREADS=5 OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0 ./run_jobs.sh -j 3 NJOBS EVENTS SEED0
  ```
  Leave `OPENBLAS_NUM_THREADS` at the new default of 1.

**Verdict: negative effect on the M3: no.** The branch is faster per job (−10.5 % per
event, −57 % CPU, −12 s startup with the cache) and per campaign (+21 % at `-j 3`), with
bit-identical output.

## run_jobs.sh on macOS

- **What broke:** `run_jobs.sh` (in `contribs/PyJetscape/example/prod_AuAu_0_10/`, which
  `prod_AuAu_0_10_jet/run_jobs.sh` wraps) kept its running jobs in an associative array (`declare -A`) and
  waited with `wait -n -p`. Both need bash ≥ 5.1. macOS's `/bin/bash` is 3.2, so every
  seed was reported as FAILED and no job ran.
- **Fix:** two indexed arrays (pids and seeds) and a reap that polls the running pids
  once a second with `kill -0`, then collects the exit status with `wait PID`. Only
  bash 3.0 features, so it runs unchanged on Linux.
- **Tested** with a dummy job driver under bash 3.2 (macOS) and bash 5.2: slot refill
  when jobs finish out of order, a failing seed (logged, the others continue, exit code
  1), no extra arguments, and SIGTERM (running seeds stopped, no orphans, exit code
  130). Under bash 5.2 its outcomes match the original script's.
- **Difference:** a finished job is noticed within one second instead of at once,
  negligible against 30–60 s per event.
- **Not tested:** a Linux kernel (the bash 5.2 test ran on macOS).

## Measuring on macOS: pitfalls

- **`DYLD_LIBRARY_PATH` does not survive `/usr/bin/time`.** `/usr/bin/time` is protected
  by System Integrity Protection, so macOS removes `DYLD_*` variables for it and everything
  it starts. To A/B-test builds by pointing `DYLD_LIBRARY_PATH` at saved dylibs, put the
  variable directly on the conda `python`. Log `DYLD_PRINT_LIBRARIES=1` to prove which
  dylib loaded.
- **`libmusic.dylib` needs `music_kernels.metallib` beside it.** Loaded from another
  directory without it, MUSIC falls back to the CPU path (~345 s/event instead of ~27 s)
  and prints `set MUSIC_METALLIB=/path/to/music_kernels.metallib`. Grep the log for
  `MUSIC_METALLIB`.
- **MUSIC4GPU changes that touch `HydroSourceBase`** (e.g. `6d33feb`) change a vtable that
  libmusic and libJetScape share: swap the two dylibs as a pair.
- **GPU utilisation without root:**
  `ioreg -r -d 1 -w 0 -c IOAccelerator | grep -o '"Device Utilization %"=[0-9]*'`.
- **The job logs contain ANSI colour codes**, so `grep` treats them as binary; use
  `grep -a`.

## Reproducing

`bench.sh NAME P [VAR=VALUE …]` runs `run_jobs.sh -j P` with P jobs of 3 events (seeds
1 … P) and samples the GPU, the CPU, memory and swap about every 2.5 s. It leaves
`run_jobs.log`, the job logs, `mon.log` (time, GPU %, CPU %, memory in use in GB, swap)
and `summary.txt` in `NAME/`, and deletes the `.h5` output.

```zsh
#!/bin/zsh
NAME=$1; P=$2; shift 2
RJ=/path/to/js-contrib/contribs/PyJetscape/example/prod_AuAu_0_10_jet/run_jobs.sh
D=$PWD/$NAME; rm -rf $D; mkdir -p $D
source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh; conda activate fno_env_mlx
for kv in "$@"; do export $kv; done
( while :; do
    g=$(ioreg -r -d 1 -w 0 -c IOAccelerator | grep -o '"Device Utilization %"=[0-9]*' | head -1 | cut -d= -f2)
    c=$(top -l 2 -n 0 -s 1 | grep 'CPU usage' | tail -1 | awk '{gsub("%","",$3); gsub("%","",$5); print $3+$5}')
    m=$(vm_stat | awk '/page size/{ps=$8} /Pages active/{a=$3} /Pages wired/{w=$4} /occupied by compressor/{c=$5} END{printf "%.1f", (a+w+c)*ps/2^30}')
    echo "$(date +%s) $g $c $m $(sysctl -n vm.swapusage | awk '{print $6}')"; sleep 1
  done ) > $D/mon.log 2>/dev/null &
M=$!
t0=$(python -c 'import time; print(time.time())')
$RJ -j $P $P 3 1 $D > $D/run_jobs.log 2>&1
t1=$(python -c 'import time; print(time.time())')
kill $M
echo "$NAME P=$P $* makespan=$(python -c "print(round($t1 - $t0, 1))")" | tee $D/summary.txt
rm -f $D/*.h5
```

The sequence that produced the table:

```zsh
PASS="OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0"
./bench.sh p1 1;  ./bench.sh p2 2;  ./bench.sh p3 3
./bench.sh p1_pass 1 ${=PASS};  ./bench.sh p2_pass 2 ${=PASS};  ./bench.sh p3_pass 3 ${=PASS}
./bench.sh p3_pass_t5 3 ${=PASS} OMP_NUM_THREADS=5;  ./bench.sh p4_pass 4 ${=PASS}
./bench.sh p2_pass_t8 2 ${=PASS} OMP_NUM_THREADS=8;  ./bench.sh p4_pass_t4 4 ${=PASS} OMP_NUM_THREADS=4
./bench.sh p3_t5 3 OMP_NUM_THREADS=5
```
