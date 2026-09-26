# prod_AuAu_0_10_jet on the GB10: concurrent jobs, OpenMP and GPU

Measured 2026-09-25. Question: does running 2, 3 or 4 jobs at once (`run_jobs.sh -j P`)
pay off, and is there an OpenMP or GPU bottleneck?

**Short answer.**
- **Before the single-job speed-ups** (first measurement): parallel jobs paid off a lot,
  about 2× the throughput with `-j 3` and 2.3× with `-j 4`, because each job left the GPU
  and most cores idle.
- **After the speed-ups** (re-measured, below): one job alone does 118 events/h (was 68).
  Parallel jobs add much less, about 1.35× with `-j 3` and 1.4× with `-j 4`, because the
  **GPU is now the shared bottleneck**: ~70 % busy with 3–4 jobs.
- **With CUDA MPS** (`run_jobs.sh --mps`) the jobs' kernels share the GPU instead of taking
  turns, and `-j 4` reaches **179 events/h** (+12.5 %), with byte-identical output.
- **With `OMP_NUM_THREADS=5` per job** on top, `-j 4 --mps` reaches **~190 events/h**, the
  best measured. See [Recommended settings](#recommended-settings-on-the-gb10).
- **Absolute throughput at `-j 3`/`-j 4`** went from 140 / 154 to 155 / 159 events/h. The
  speed-ups mostly shorten each job rather than raise the machine's campaign ceiling.
- **The startup hang** (jobs started at the same moment could hang forever at
  `Initialize MUSIC`) is fixed by per-job working directories (see
  [the startup race](#bug-concurrent-jobs-can-hang-at-start)).

## Recommended settings on the GB10

These are measured on this machine (GB10: 20 cores, 10 × X925 + 10 × A725; 121 GB unified
memory; one integrated GPU), for `prod_AuAu_0_10_jet`. **They are deliberately not built into
the scripts:** with another CPU/GPU balance, core count or memory, another `-j` and thread
count can be better. Re-measure there, as described
[below](#finding-the-settings-on-another-machine).

**One job alone:** the defaults. Leave `OMP_NUM_THREADS` unset, so all 20 threads are used;
fewer threads slowed a single job by 24 % in the first measurement.

**A campaign of many jobs:**

```bash
export OMP_NUM_THREADS=5            # ~ cores / jobs at once
./run_jobs.sh -j 4 --mps NJOBS 25 FIRST_SEED
```

| Setting | Value | Effect (4 jobs, MPS; events/h) |
|---|---|---|
| jobs at once | `-j 4` | best measured; `-j 3` 163, `-j 2` 136 (with MPS, default threads) |
| CUDA MPS | `--mps` | 159 → 173–179 |
| OpenMP threads per job | `OMP_NUM_THREADS=5` | 173–179 → **193**, GPU 71–75 % → 81 % busy |
| OpenBLAS threads per job | not needed | `OPENBLAS_NUM_THREADS=4` alone gave 186; combined with `OMP_NUM_THREADS=5` no better (189, 192) |
| `OMP_WAIT_POLICY` | not needed | `passive`: 175, no effect; no cost for one job either (116 vs 118) |
| `KMP_BLOCKTIME` | not applicable | Intel/LLVM `libomp` only; these builds use GNU `libgomp` |

With these settings a campaign runs at **~190 events/h**: 1.6× one job alone (118), and
2.8× the 68 events/h a single job did before the speed-ups. Memory stays at ~66–69 GB of the
121 GB.

**Measurement details:**
- **Runs:** each is 4 jobs × 3 events, seeds 1–4, started at the same moment in their own
  working directories, under a fresh MPS daemon. The rows were measured in sequence on an
  otherwise idle machine.

  | Environment (4 jobs, MPS) | events/h | GPU busy | GPU clock while busy | cores busy |
  |---|---|---|---|---|
  | defaults (3 runs) | 173, 179, 179 | 71–75 % | ~2,400 MHz | ~12 |
  | `OMP_WAIT_POLICY=passive` | 175 | 71 % | 2,396 MHz | 11.8 |
  | `OMP_NUM_THREADS=5` | **193** | **81 %** | 2,397 MHz | 8.0 |
  | `OPENBLAS_NUM_THREADS=4` | 186 | 77 % | 2,395 MHz | 8.9 |
  | `OMP_NUM_THREADS=5` + `OPENBLAS_NUM_THREADS=4` + `passive` (2 runs) | 192, 189 | 78–79 % | ~2,400 MHz | 7.8 |

- **Noise:** the identical default runs spread by about ±2 %, so only the thread limit is
  clearly real (+8–9 %).
- **Why fewer threads help:** by default each job starts 20 OpenMP (and 20 OpenBLAS)
  threads, so with 4 jobs up to 80 of each compete for 20 cores whenever parallel sections
  of different jobs overlap. With 5 per job, fewer cores are busy but the GPU is fed faster.
- **No power throttling:** the GPU clock stays at ~2.4 GHz under every CPU load. The CPU and
  GPU share a power budget on the GB10, but CPU load does not slow the GPU here, so
  blocking instead of spinning CUDA syncs would not help.
- **Not measured:** `-j 5` (~85 GB, the GPU still has ~20 % headroom at `-j 4`), `-j 3` with
  6–7 threads per job, and other thread counts at `-j 4`.

### Finding the settings on another machine

1. **One job alone** gives the baseline events/h, and shows whether the GPU is mostly idle
   (`nvidia-smi dmon -s u`). If it is, parallel jobs will help.
2. **Concurrent jobs:** try `-j 2 … 4` (memory permitting, ~17 GB per job here), with and
   without `--mps`. MPS pays off once the GPU is the shared bottleneck.
3. **Thread limit:** with P jobs at once, try `OMP_NUM_THREADS ≈ cores / P`, and one step
   either side.
4. **Compare runs properly:** use the same seeds and at least 3 events per job, and repeat
   the best and the default settings once each; identical runs differed by ~±2 % here.
   The per-event times (`done in … s`) of all jobs give events/h.

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

### After the speed-ups (current)

Measured with js-contrib `main` + `concurrent_jobs`, X-SCAPE `contrib` (#143, #144, #146),
MUSIC4GPU `XSCAPE` `ca94ed4` + `concurrent_jobs`. Same method as below: 3 events per job,
seeds 1–4, jobs started at the same moment (no stagger), each in its own working directory.

| Jobs at once | s/event per job | events/h | vs 1 job (events) | vs serial (wall time incl. startup) | GPU busy (mean) | GPU idle samples (< 10 %) | cores busy (mean) | memory used (max) |
|---|---|---|---|---|---|---|---|---|
| 1 | 30.6 | **118** | 1.00× | 1.00× | 43 % | 45 % | 5.5 | 22 GB |
| 2 | 54.0 | 133 | 1.17× | 1.21× | 57 % | 36 % | 8.4 | 40 GB |
| 3 | 69.5 | **155** | 1.35× | 1.38× | 67 % | 25 % | 10.4 | 52 GB |
| 4 | 90.5 | **159** | 1.41× | 1.43× | 70 % | 23 % | 11.4 | 67 GB |

Solo per-event means: seed 1 30.6 s, seed 2 32.8 s, seed 3 30.6 s, seed 4 33.5 s.

- **Why the gain shrank: the GPU.**
  - The GPU kernels take ~16 s per event (MUSIC timers: 7.25 ms × ~2,200 substeps, both
    legs). At 159 events/h that is ~2,500 GPU-seconds per hour, ≈ 70 % of the GPU: what
    `nvidia-smi` shows.
  - Kernels from different processes are time-sliced, not overlapped, so a job waits
    while another job's MUSIC step runs.
  - The ceiling at 100 % GPU would be ~225 events/h.
- **Recommendation without MPS: `run_jobs.sh -j 3`** (155 events/h, 52 GB). `-j 4` adds
  only ~3 % for 15 GB more; `-j 2` gives 133. With MPS, see below.
- **Further CPU work** helps single-job latency but hardly the campaign rate. Faster GPU
  kernels would help (profile with `nsys` first).

### With CUDA MPS (`run_jobs.sh --mps`)

**What MPS does:** CUDA MPS (Multi-Process Service) lets the kernels of concurrent jobs
share the GPU instead of being time-sliced. One job alone keeps the SMs only ~44 % busy,
so there is room. Same setup as the table above; the jobs are clients of a user-level MPS
daemon.

| Jobs at once | events/h without MPS | events/h with MPS | gain | GPU busy (MPS) | memory used (max) |
|---|---|---|---|---|---|
| 1 | 118 | 117 | −0.1 % | 44 % | 22 GB |
| 2 | 133 | 136 | +2.2 % | 57 % | 43 GB |
| 3 | 155 | 163 | +5.1 % | 68 % | 51 GB |
| 4 | 159 | **179** | **+12.5 %** | 75 % | 66 GB |

- **Output:** byte-identical with and without MPS (seeds 1–3, 3 events each, all 30
  datasets).
- **Recommendation: `run_jobs.sh -j 4 --mps`** (179 events/h, ~66 GB), and with
  `OMP_NUM_THREADS=5` ~190 events/h (see
  [Recommended settings](#recommended-settings-on-the-gb10)).
- **What `--mps` does:**
  - starts a daemon for the campaign (`nvidia-cuda-mps-control -d`) with its sockets in
    `$MPS_DIR`, default `${XDG_RUNTIME_DIR:-/tmp}/xscape-mps.<pid>`;
  - exports `CUDA_MPS_PIPE_DIRECTORY` / `CUDA_MPS_LOG_DIRECTORY` to the jobs;
  - stops the daemon at the end, also on Ctrl-C, and reports how many clients
    connected.
- **The pitfall it guards against:** MPS's UNIX socket paths are limited to ~107
  characters. A longer pipe directory makes the daemon exit silently, so the script refuses
  to start then.
- **On the GB10,** the MPS server runs in `-force-tegra` mode (integrated GPU), which works
  as expected.

### Before the speed-ups (first measurement)


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

- **Recommendation at the time:** `run_jobs.sh -j 3`; `-j 4` added about 10 %.
  Superseded by the table above.
- **OpenMP:** splitting the cores between jobs (`OMP_NUM_THREADS = 20 / P`) doesn't help
  once jobs run in parallel. Leave `OMP_NUM_THREADS` unset. A single job with 6 threads
  is 24 % slower, because the source-term fill (below) uses many cores in short bursts.
- **GPU:** not saturated then. Even with 4 jobs it was idle in 29 % of samples.

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
| h5 writing (h5py, lzf; Blosc-zstd, now the default, costs the same job time) | 1.1 | 1.1 | 1 CPU thread |
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

### String deposition: binning by transverse reach

**The cost:** at each of the ~8 string fills per event, every cell looped over every
string, 600,000 cells × 1,200–1,600 strings ≈ 10⁹ string–cell checks. Yet the loops
skip a string that is more than 8 σ_x away in x or in y before adding anything.

**Applied (MUSIC4GPU branch `string_bin_optim`, `fa6ec5b`, on top of
`cpu_source_fill_optim`):**
- `prepare_list_for_current_tau_frame` bins the string, remnant and baryon lists by
  the transverse box each entry can reach, one bin per grid cell. Each per-cell loop
  walks only its bin's entries.
- The box:
  - `getStringTransverseCoord` = midpoint + `stringTransverseShiftFrac` ·
    (0.5 − η_frac) · (x_l − x_r) / 2 is linear in the clamped η_frac, so a string's
    (and a baryon's) position spans its values at η_frac = 0 and 1;
  - remnants sit at their fixed (x_perp, y_perp);
  - both get ± 8 σ_x + 10⁻⁶ fm, and the same in y.
- The bins keep **list order**, so every cell adds the same terms in the same order.
- A query outside the grid (+ 1 cell) walks the whole list.
- The three local `n_sigma_skip = 8.` constants now read one shared member, so the
  boxes and the cuts cannot drift apart.

Measured on one job alone, seed 1, 2 events:

| Build | event 1 (1184 strings) | event 2 (1591 strings) | string evaluation in the fill (profile) | Output |
|---|---|---|---|---|
| + `cpu_source_fill_optim` | 34.3 s | 43.0 s | 7.9 s/event | — |
| + `string_bin_optim` | **30.6 s (−11 %)** | **38.0 s (−12 %)** | 3.7 s/event | **byte-identical** (all 30 datasets; MUSIC's logged energy totals and end times identical) |

- **Where the gain is:** the event with more strings gains more.
- **What's left:** the whole fill is now 5.9 s/event (strings 3.7, OpenMP overhead
  1.3, droplets 0.55, loop 0.4), and building the bins costs 0.02 s/event. The
  remaining string time is the strings that really are within 8 σ_x of a cell: the
  erf, exp, cosh and sinh evaluations of the deposition itself.
- **Not checked:** the binning also applies to MUSIC's CPU path (`FirstRKStepT`, e.g.
  `MUSIC_FORCE_CPU=1`), which calls the same functions. It is bit-identical there by
  the same argument, but that path was not rerun.

### Summary of the single-job speed-ups

Seed 1 on the GB10, per event:

| Build | event 1 | event 2 |
|---|---|---|
| start (before `hydro_data_optim`) | 53.1 s | 60.3 s |
| + `hydro_data_optim` (resample, `bulk_info` copy) | 38.0 s | 46.6 s |
| + `cpu_source_fill_optim` (skip empty source fills, dynamic schedule) | 34.1–34.3 s | 42.5–43.0 s |
| + `string_bin_optim` (bin strings by transverse reach) | **30.6 s (−42 %)** | **38.0 s (−37 %)** |

Only the resampling step changes any value, by one float32 ulp in one of 1.4 × 10⁸
entries of `arr`. The other steps are byte-identical.

## Bug: concurrent jobs can hang at start

**Status: fixed** (branches `concurrent_jobs`, see [the fix](#fix-per-job-working-directories)
below). The description is kept for reference.

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

### Fix: per-job working directories

**Not only `music_input`.** Tracing one job (`strace`) showed that `music_input` is not
the only file in the working directory that concurrent jobs share:
- **`music_input`:** rewritten 4× per pair job and opened ~660× (`StringFind4` reopens it
  for every parameter);
- **3dMCGlauber:** writes `strings_event_<N>.dat` and `events_summary.dat`, the same names
  in every job;
- **MUSIC:** writes 20 `momentum_anisotropy` / `eccentricities_evo` / `meanpT_estimators`
  files and `FO_nBvseta.dat`.

**The fix (js-contrib `concurrent_jobs`):** `run_prod.py` and `run_prod_jet.py` no longer
change into the build tree. Each job gets its own directory, `OUTDIR/work/<tag>`, the Python
counterpart of X-SCAPE's `examples/run_in_workdir.sh`:
- a private `music_input`;
- `XSCAPE_DATA_DIR`, `HYDROPROGRAMPATH` and `LBT_TABLES_PATH` pointing at the build tree;
- symlinks for the directories still opened by relative path;
- `../` XML paths made absolute.

It is removed after a successful job. `--in-build` restores the old behaviour.

**Safety net (MUSIC4GPU `concurrent_jobs`):** `StringFind4` now stops with an error
("… ended without an EndOfData line …") when the file ends before `EndOfData`, instead of
looping forever. The old code, given an empty file, hung until killed; the new code exits
at once.

**Validation:**
- `build_gpu/music_input` reset to the CMake template state (EOS 91, bulk 0), which is the
  state that made 2 of 3 jobs hang before.
- Then 4 jobs (seeds 1–4) started at the same moment, with no stagger.
- **All 4 completed.** Each output is **byte-identical** to a serial run of the same seed
  in the build tree.
- **Nothing was written to `build_gpu`,** and its `music_input` was left untouched.

The 20 s stagger is no longer needed.

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
