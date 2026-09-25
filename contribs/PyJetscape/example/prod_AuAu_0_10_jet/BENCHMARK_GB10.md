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

This is from `py-spy record --native --idle` profiles of one 2-event job (seed 1)
running alone, main thread only, before and after the `hydro_data_optim` changes
(below). Both profiles were classified by the same rules, and each was corrected for its
own profiler overhead (the ratio of the job's event times with and without `py-spy`:
×0.75 before, ×0.82 after).

The rows are seconds per event, **averaged over the job's 2 events and including its
startup**:
- Event 1 of seed 1 has 106 / 95 jet / background frames; event 2 has 126 / 111 and
  runs longer.
- "Everything else" holds the startup before the first event (imports, XML, MUSIC and
  EOS init, Pythia; ~9 s per job), which the driver's per-event times leave out.

The totals are therefore above the per-event times the driver prints. For the per-event
times, see [the table under Applied](#applied-branches-hydro_data_optim).

| Stage | before (s/event) | after (s/event) | Runs on |
|---|---|---|---|
| Source term filled on the CPU before each GPU step (`Advance::prefill_hydro_source_on_cpu`) | 15.0 | 15.3 | CPU, OpenMP — see [The CPU source fill](#the-cpu-source-fill) |
| MUSIC GPU steps, both legs (`Advance::try_gpu_advance`, without the fill) | 9.6 | 10.5 | GPU |
| Resampling onto the output grid (`bulk_sources.resample`) | 14.7 | **3.1** | 1 CPU thread → BLAS |
| Copying the background leg into `bulk_info` (`MpiMusic::PassHydroEvolutionHistoryToFramework`) | 7.7 | **2.5** | 1 CPU thread → OpenMP |
| Energy loss (Matter + LBT) | 3.9 | 3.3 | 1 CPU thread |
| Other MUSIC: evolve loop, frame dump to memory, GPU↔host syncs | 3.3 | 3.5 | CPU |
| Other writer work: native-store read, hashing | 2.4 | 2.6 | CPU |
| h5 writing (h5py, lzf) | 1.1 | 1.1 | 1 CPU thread |
| Momentum-anisotropy output (`output_momentum_anisotropy_vs_etas`) | 1.1 | 1.1 | CPU |
| Everything else (startup share, Pythia, framework) | 4.4 | 4.4 | CPU |
| **Total** | **63.3** | **47.4** | |

The GPU does only about 10 s of each event's work. Parallel jobs help because each
job's CPU stages run on separate cores.

Each job still slows down 1.4–1.8× when others run beside it. This was not pinned
down. Likely causes:
- the GPU switching between processes when their MUSIC steps overlap;
- 10 of the 20 cores being the slower A725, which also leaves the static OpenMP
  loops unevenly loaded;
- the CPU and GPU sharing memory bandwidth, which matters because the source fill, the
  resampling and the hydro stencil all move a lot of memory.

### Cheap single-job improvements

Identified from the first profile ("before" column):

1. **Resampling** (~15 s/event). `resample` used linear interpolation (`order=1`,
   `prefilter=False`), but called `map_coordinates` separately for every frame and
   channel: about 1,700 single-threaded calls per event (106 frames × 4 channels × 2
   tau neighbours × 2 legs), always with the same target points.
2. **`bulk_info` copy** (~8 s/event). One heap allocation and one `push_back` per
   cell for ~10⁸ cells, with the vector reallocating as it grew.
3. **`output_momentum_anisotropy_vs_etas`** (~1 s/event): diagnostics that nothing in
   this production uses.

### Applied (branches `hydro_data_optim`)

Improvements 1 and 2 are implemented; 3 is on hold. What omitting it would entail is under [README.md, Potential next steps](README.md#potential-next-steps).

- **js-contrib** `e3101fd`: `resample` as three separable matrix-product passes
  (eta, y, x) over all features of a source frame.
- **X-SCAPE** `a80a9932`: `PassHydroEvolutionHistoryToFramework` resizes the store once
  and fills it in an OpenMP loop.

Measured on one job alone, seed 1, 2 events: the per-event wall times the driver
prints. The two events are different collisions and live for different times, so
compare along a column, not across:

| Build | event 1 (106 / 95 frames, 25 droplets) | event 2 (126 / 111 frames, 35 droplets) | Output vs baseline |
|---|---|---|---|
| baseline | 53.1 s | 60.3 s | — |
| + `bulk_info` copy | 47.1 s | 58.2 s | bit-identical |
| + resample | **38.0 s (−28 %)** | **46.6 s (−23 %)** | `arr`: 1 of 1.4 × 10⁸ values differs by 1 ulp; everything else bit-identical |

A 1-event run of seed 1 (`python run_prod_jet.py --events 1 --seed 1`, the usual
smoke test) now reports `wall_s` ≈ 38 s in its `.json`, down from ~53 s. Per-event
time scales with the event's lifetime, roughly with its number of frames.

The concurrency numbers above were measured before these changes. The profile table
above has both states.

### What sets an event's runtime

Two things: **the event's lifetime** (its number of hydro time steps) and **its number
of strings**. From the job log of the run above (seed 1, `Delta_Tau` = 0.02 fm/c):

| | event 1 | event 2 | ratio |
|---|---|---|---|
| strings | 1184 | 1591 | 1.34 |
| strings deposited at τ | 0.40, 0.42 only | 0.40, 0.42 only | — |
| background leg ends at τ | 9.96 fm/c | 11.56 fm/c | |
| jet leg ends at τ | 11.06 fm/c | 13.06 fm/c | |
| hydro time steps, both legs | ~1011 | ~1191 | **1.18** |
| droplets | 25 | 35 | 1.4 |
| wall time | 38.0 s | 46.6 s | **1.23** |

- **Lifetime:** almost every stage scales with the number of steps or stored frames:
  the GPU steps, the frame dump, the resampling and the `bulk_info` copy.
- **Strings:** deposited only in the first two steps, but those two steps are
  expensive. Every string is evaluated at every cell, ~0.7 s per source fill, and there
  are ~8 such fills per event (2 steps × 2 Runge–Kutta substeps × 2 legs). That cost,
  ~9 s per event, scales with the number of strings. See
  [The CPU source fill](#the-cpu-source-fill).
- **Split of the 8.6 s difference** (an estimate from the profile, not a separate
  measurement):
  - ~2.5–3 s from the 34 % more strings;
  - most of the rest from the 18 % more steps;
  - a little from the extra droplets and a longer energy-loss stage.
- More strings also mean more energy and a larger, denser fireball that takes longer to
  cool below freeze-out, so the two effects go together.

### The CPU source fill

`Advance::prefill_hydro_source_on_cpu` (MUSIC4GPU `advance.cpp`) was the largest single
stage after the `hydro_data_optim` changes, at ~15 s/event, more than the GPU work
itself. Before a Runge–Kutta substep of either leg it:
- zeroes a 5 × N-cell source buffer (N = 100 × 100 × 60);
- evaluates the string source (and, on the jet leg, the droplet source) at **every**
  cell in an OpenMP loop;
- hands the buffer to the GPU kernel, which adds `qi_source * dt` to T^τμ.

The per-event split before the change:

| Part | ~s/event |
|---|---|
| main thread waiting at the OpenMP barrier | 7.4 |
| `HydroSourceStrings::get_hydro_energy_source` | 5.7 |
| the loop itself (index maths, stores) | 1.3 |
| `LiquefierBase::get_source` (droplets, jet leg only) | 0.9 |

**What was wasteful:**
- **The fill ran at every substep of the whole evolution.** With
  `evolve_QCD_string_mode 4` every string is deposited at the start: the log shows
  `HydroSourceStrings: tau_min = tau_max = 0.424 fm/c`, strings active at τ = 0.40 and
  0.42 fm/c, and none from τ = 0.44 on (the remnant lists end at the same time).
  `flag_add_hydro_source` is nevertheless set once and stays true. So the fill ran at
  all ~500–650 steps of each leg, and at all but the first two it produced zeros.
- **Half of it was waiting.** The static schedule gave each thread an equal share of
  cells. The cost per cell varies by orders of magnitude (cells near strings are
  expensive), and the cores run at two speeds (X925 / A725).

**Applied (branches `cpu_source_fill_optim`):**
- **MUSIC4GPU `6d33feb`:**
  - `HydroSourceBase::has_active_sources_current_tau()`, default `true`.
    `HydroSourceStrings` returns false when its four current-τ lists are empty; every
    `get_hydro_*_source` then returns exactly zero.
  - `try_gpu_advance` evaluates only the sources that are active in the step, and skips
    the fill and the kernel's source add when none is.
  - The loop runs with `schedule(dynamic, 1024)`.
- **X-SCAPE `75b4ce7a`:** `HydroSourceJETSCAPE` implements the same query from the
  liquefier's pruned droplet list. It is false when no droplet can deposit at the step's
  query times, and stays true for a hadronic liquefier or an unprepared window.

Measured on one job alone, seed 1, 2 events, on top of `hydro_data_optim`:

| Build | event 1 | event 2 | source fill (profile) | Output |
|---|---|---|---|---|
| `hydro_data_optim` (js-contrib main, X-SCAPE contrib) | 37.2 s | 45.5 s | 15.3 s/event | — |
| + `cpu_source_fill_optim` | **34.1 s** | **42.5 s** | 10.1 s/event | **byte-identical** (all 30 datasets) |

The fill's split after the change: string evaluation 7.9 s/event, OpenMP overhead
1.3 s, droplets 0.7 s, the loop 0.2 s. The barrier waiting is gone. What remains is
**real work at the two string steps**, which the skip cannot remove, and which the
earlier estimate here ("most of the ~15 s") had missed.

### Next candidate: string deposition

**The cost:** at each of the ~8 string fills per event, every cell loops over every
string, 600,000 cells × 1,200–1,600 strings ≈ 10⁹ string–cell checks, although a string
reaches only the cells within 8 σ_x of it in the transverse plane.

**The fix:** bin the strings once per step by the transverse box they can reach on a
coarse (x, y) grid, and have each cell loop over its bin's strings only.
- A string's transverse position at η is
  `getStringTransverseCoord` = midpoint + `stringTransverseShiftFrac` · (0.5 − η_frac) ·
  (x_l − x_r) / 2, with η_frac in [0, 1]. So its box is midpoint ±
  |`stringTransverseShiftFrac`| · |x_l − x_r| / 4 ± 8 σ_x, and the same in y.
- Keeping them **in list order** keeps the summation order per cell, so the output
  stays bit-identical: a string that fails the distance cut adds nothing.
- This should remove most of the remaining ~9 s/event, and make the cost nearly
  independent of the number of strings.
- It is a change in MUSIC4GPU's `HydroSourceStrings`. Not done yet.

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
