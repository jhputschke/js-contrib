# prod_AuAu_0_10_jet on the GB10: concurrent jobs, OpenMP and GPU

Measured 2026-09-25. Question: does running 2, 3 or 4 jobs at once (`run_jobs.sh -j P`)
pay off, and is there an OpenMP or GPU bottleneck?

**Short answer.** Yes. `-j 3` gives about 2× the throughput of one job, and `-j 4` about
2.3×. Neither OpenMP nor the GPU is the bottleneck. Single-threaded CPU stages are. But
jobs that start at the same moment can **hang forever** at `Initialize MUSIC` (see
[the startup race](#bug-concurrent-jobs-can-hang-at-start)). Until that is fixed,
start concurrent jobs about 20 s apart.

## Setup

- **Machine:** NVIDIA GB10, 20 cores (10 × Cortex-X925 at 3.9 GHz on CPUs 5–9 and 15–19,
  10 × Cortex-A725 at 2.8 GHz on CPUs 0–4 and 10–14), 121 GB unified memory. Idle otherwise.
- **Build:** `build_gpu` (music4gpu, CUDA), env `js_fno`.
- **Jobs:** `run_prod_jet.py --events 3 --seed S` with the defaults: PythiaGun 50–70 GeV,
  `--surface none`, `grid_fno.yaml`, `--reuse 1`.
- **Seeds:** fixed (1, 2, 3, 4), so every configuration does the same work. Seeds 1–3
  were also run alone as the baseline.
- **Sampling:** `nvidia-smi dmon` (GPU SM utilisation, every 1 s), `mpstat` (all cores,
  every 2 s), `free` (memory), and `/usr/bin/time -v` (peak RSS per job).
- **How the numbers are computed:**
  - *s/event* is the mean of the per-event wall times the driver prints, i.e. the steady
    state without startup.
  - *events/h* is Σ over jobs of 3600 / (that job's mean s/event).
  - *vs 1 job* divides by the solo throughput of seeds 1–3.

## Throughput

| Jobs at once | s/event per job | events/h | vs 1 job | GPU busy (mean) | GPU idle samples (< 10 %) | cores busy (mean) | memory used (max) |
|---|---|---|---|---|---|---|---|
| 1 | 53.0 | 68 | 1.00× | 25 % | 64 % | 4.1 | 22 GB |
| 1, `OMP_NUM_THREADS=6` | 65.6 | 55 | 0.81× | 22 % | 66 % | 2.9 | 32 GB |
| 2 | 74.0 | 97 | 1.45× | 44 % | 47 % | 6.8 | 36 GB |
| 2, `OMP_NUM_THREADS=10` | 70.0 | 103 | 1.54× | 39 % | 48 % | 5.6 | 38 GB |
| 3 | 77.4 | 140 | **2.06×** | 53 % | 32 % | 8.6 | 49 GB |
| 3, `OMP_NUM_THREADS=6` | 77.7 | 139 | 2.06× | 50 % | 30 % | 7.0 | 52 GB |
| 4 | 93.6 | 154 | **2.28×** | 57 % | 29 % | 9.6 | 71 GB |

Solo per-event means: seed 1 53.0 s, seed 2 54.5 s, seed 3 52.2 s. Peak RSS per job
is 17–22 GB. The 4-job run includes seed 4, whose solo speed was not measured.

- **Recommendation:** `run_jobs.sh -j 3`. `-j 4` adds about 10 % more if about 71 GB of
  the 121 GB is free. Beyond that the returns are small.
- **OpenMP:** splitting the cores between jobs (`OMP_NUM_THREADS = 20 / P`) doesn't help
  once jobs run in parallel. Leave `OMP_NUM_THREADS` unset. A single job with 6 threads
  is 24 % slower, because the source-term fill (below) uses many cores in short bursts.
- **GPU:** never saturated. Even with 4 jobs it is idle in 29 % of samples.

### Activity of one job alone

In each strip below, one character is one sample.

GPU, sampled every 1 s: `#` means SM > 50 %, `+` means 10–50 %, `.` means < 10 %.

```
.............++########.............##########.....................#.#########.......
.....#############........................#########........########.................
```

CPU, all 20 cores together, sampled every 2 s: `#` means > 50 % busy, `+` means 15–50 %,
`.` means < 15 %.

```
....+#+.......+##+++..........+##+.......+##+++++...........+#+.....+##+++..........
```

Each event has two GPU bursts of about 10 s, one per MUSIC leg, separated by 15–20 s
gaps where the GPU is idle. During those gaps the CPU runs 1–3 threads, with brief
bursts of heavy OpenMP use.

## Where a single job's time goes

This is from a `py-spy record --native --idle` profile of one 2-event job (seed 1)
running alone, main thread only. Seconds per event are corrected for the profiler's
roughly 40 % overhead.

| Stage | ~s/event | Runs on |
|---|---|---|
| MUSIC GPU steps, both legs (`Advance::try_gpu_advance`) | ~8.5 | GPU |
| Resampling onto the output grid (`bulk_sources.resample`, `scipy.ndimage.map_coordinates`) | ~13 | **1 CPU thread** |
| Source term filled on the CPU before each GPU step (`Advance::prefill_hydro_source_on_cpu`: `HydroSourceStrings::get_hydro_energy_source` + `LiquefierBase::get_source`) | ~7 | CPU, OpenMP |
| Copying the background leg into `bulk_info` (`MpiMusic::PassHydroEvolutionHistoryToFramework`) | ~7 | **1 CPU thread** (about 40 % of it is `std::vector<FluidCellInfo>::_M_realloc_insert`) |
| Energy loss (Matter + LBT) | ~4 | 1 CPU thread |
| h5 writing (lzf), `output_momentum_anisotropy_vs_etas`, other | ~4 | CPU |

The GPU work is only about 8.5 s of each 53 s event. Parallel jobs help because each
job's serial CPU stages run on separate cores.

Each job still slows down 1.4–1.8× when others run beside it. This was not pinned
down. Likely causes:
- the GPU switching between processes when their MUSIC steps overlap;
- 10 of the 20 cores being the slower A725;
- the CPU and GPU sharing memory bandwidth, which matters because the resampling, the
  `bulk_info` copy and the hydro stencil all move a lot of memory.

### Cheap single-job improvements

1. **Resampling** (~13 s/event). `resample` uses linear interpolation (`order=1`,
   `prefilter=False`), but calls `map_coordinates` separately for every frame and channel:
   about 1,700 single-threaded calls per event (106 frames × 4 channels × 2 tau
   neighbours × 2 legs), always with the same target points. Computing the 8 interpolation
   indices and weights once per event and applying them as one vectorised gather should
   cut this to a few seconds.
2. **`bulk_info` copy** (~7 s/event). Reserving the `FluidCellInfo` vector up front
   removes the reallocation, about 3 s/event.
3. **`output_momentum_anisotropy_vs_etas`** (~1.5 s/event): diagnostics that nothing in
   this production uses.

### Applied (branches `hydro_data_optim`)

Improvements 1 and 2 are implemented; 3 is on hold. What omitting it would entail is under [README.md, Potential next steps](README.md#potential-next-steps).

- **js-contrib** `e3101fd`: `resample` as three separable matrix-product passes
  (eta, y, x) over all features of a source frame.
- **X-SCAPE** `a80a9932`: `PassHydroEvolutionHistoryToFramework` resizes the store once
  and fills it in an OpenMP loop.

Measured on one job alone, seed 1, 2 events:

| Build | event 1 | event 2 | mean | Output vs baseline |
|---|---|---|---|---|
| baseline | 53.1 s | 60.3 s | 56.7 s | — |
| + `bulk_info` copy | 47.1 s | 58.2 s | 52.7 s | bit-identical |
| + resample | 38.0 s | 46.6 s | **42.3 s (−25 %)** | `arr`: 1 of 1.4 × 10⁸ values differs by 1 ulp; everything else bit-identical |

The concurrency numbers above were measured before these changes.

## Bug: concurrent jobs can hang at start

On the first 3-job attempt, 2 of the 3 jobs stayed at `Initialize MUSIC` at 100 % CPU on one
thread for over 11 minutes. The GPU was idle, and both jobs had `build_gpu/music_input`
open at file position 0.

**What happens:**
- `MpiMusic::InitializeHydro` (`src/hydro/MusicWrapper.cc`) calls
  `update_music_input_parameter(input_file, "EOS_to_use", EOS)` before constructing
  MUSIC. That function reads the **shared** `build_gpu/music_input`, truncates it and
  writes it back.
- A pair job does this twice, once per MUSIC instance.
- A job that opens the file while another job has it truncated reads an empty file.
- MUSIC4GPU's `Util::StringFind4` (`src/util.cpp`) reads lines until it meets
  `EndOfData`. At end of file `getline` returns an empty string, so the loop never ends.

**Why it matters for production:** `run_jobs.sh -j P` launches all P jobs at the same
moment, so the start of a campaign is exactly when this happens. Later jobs start one at
a time, as others finish, so they are much less exposed. The single-leg `-j 2`
measurement in `../prod_AuAu_0_10/README.md` just didn't hit it.

**Workaround used here:** start the jobs 20 s apart. Every run after that completed,
including 4 jobs. `music_input` itself was left intact: it only ever gets the same
content back.

**Possible fixes, from smallest to most thorough:**
- In `update_music_input_parameter`, skip the rewrite when the value is already correct.
  That covers every real campaign, since all jobs use the same EOS.
- Write to a temporary file and `rename()` it over the original, so no job can ever see
  an empty file.
- In `StringFind4`, stop at end of file and return `"empty"` or an error, instead of
  looping.
- Stagger starts in `run_jobs.sh`, as a stopgap.

## Reproducing

Run `bench.sh NAME P OMP EVENTS SEED0`, with `OMP` set to `def` to leave
`OMP_NUM_THREADS` unset. Each run leaves `jobN.log`, `gpu.log`, `cpu.log`, `mem.log` and
`summary.txt` in `NAME/`, and deletes the `.h5` output.

```bash
#!/usr/bin/env bash
NAME=$1; P=$2; OMP=$3; EV=$4; SEED0=$5
D=$PWD/$NAME; rm -rf "$D"; mkdir -p "$D"
PROD=/path/to/js-contrib/contribs/PyJetscape/example/prod_AuAu_0_10_jet/run_prod_jet.py
source ~/miniconda3/etc/profile.d/conda.sh; conda activate js_fno
set -u      # after conda activate: its deactivate hooks trip over unset variables
if [ "$OMP" != def ]; then export OMP_NUM_THREADS=$OMP; else unset OMP_NUM_THREADS; fi

nvidia-smi dmon -s u -d 1 > "$D/gpu.log" 2>&1 & M1=$!
mpstat -P ALL 2           > "$D/cpu.log" 2>&1 & M2=$!
( while :; do echo "$(date +%s) $(free -m | awk '/Mem:/{print $3}')"; sleep 2; done ) \
    > "$D/mem.log" & M3=$!

t0=$(date +%s.%N); pids=()
for ((k = 0; k < P; k++)); do
  s=$((SEED0 + k))
  /usr/bin/time -v python "$PROD" --events "$EV" --seed "$s" --outdir "$D" \
      > "$D/job$s.log" 2>&1 &
  pids+=($!)
  [ $k -lt $((P - 1)) ] && sleep "${STAGGER:-20}"   # avoid the music_input race
done
for p in "${pids[@]}"; do wait "$p"; done
t1=$(date +%s.%N)
kill $M1 $M2 $M3 2>/dev/null
echo "$NAME P=$P OMP=$OMP events=$EV makespan=$(echo "$t1 - $t0" | bc)" | tee "$D/summary.txt"
rm -f "$D"/*.h5
```

The sequence that produced the table:

```bash
./bench.sh p1_s1_def 1 def 3 1;  ./bench.sh p1_s2_def 1 def 3 2;  ./bench.sh p1_s3_def 1 def 3 3
./bench.sh p2_def    2 def 3 1;  ./bench.sh p2_omp10  2 10  3 1
./bench.sh p3_def    3 def 3 1;  ./bench.sh p3_omp6   3 6   3 1
./bench.sh p4_def    4 def 3 1;  ./bench.sh p1_s1_omp6 1 6  3 1
```

`p2_def` ran before the stagger was added and happened not to hang. All the other
concurrent runs used the 20 s stagger.

The profile: `py-spy record --native --idle -r 20 --format raw -o prof.txt -- python
run_prod_jet.py --events 2 --seed 1`. Attaching `py-spy` to a job that is already running
needs root, because `ptrace_scope` is 1.
