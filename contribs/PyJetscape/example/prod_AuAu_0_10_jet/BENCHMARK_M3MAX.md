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
  `-j 4` with `OMP_NUM_THREADS=4` adds ~4 % for ~5 GB more memory.
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

## run_jobs.sh on macOS

- **What broke:** `run_jobs.sh` (in `../prod_AuAu_0_10/`, which this folder's
  `run_jobs.sh` wraps) kept its running jobs in an associative array (`declare -A`) and
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
