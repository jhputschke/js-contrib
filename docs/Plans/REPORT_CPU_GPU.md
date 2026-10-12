# CPU or GPU: where O+O and Au+Au events spend their time, and what would raise throughput (2026-10-11)

The O+O diagnostic of [REPORT_OO.md](REPORT_OO.md) ("Where the time goes"), repeated for the two Au+Au
productions: `prod_AuAu_0_10` (hydro only, "normal") and `prod_AuAu_0_10_jet` (the two-stage hydro:
MUSIC_1 background → Matter + LBT + liquefier → MUSIC_2). Then, with the GB10 and M3 Max benchmarks,
which side, CPU or GPU, would raise the event rate on which machine.

## Summary

- **Au+Au is host-dominated per event, like O+O.** One job alone spends 13–19 % of its time on the GPU:
  hydro only 4.95 of 25.8 s per event, the two-stage campaign 6.1 of 45 s, the plain two-stage chain
  11.7 of 75 s (O+O: 1.34 of 8.1 s). CPU per event: 40, 74 and 133 core-s (O+O: 10.6).
- **The largest host items are not O+O's:**
  - **String deposition on the CPU:** ~2 s per fill, 4 fills per MUSIC run, ~25 core-s per run. In the
    two-stage chain MUSIC_2 repeats it for every jet, although its first frames are bit-identical to
    the background's.
  - **The writer's resampling:** 12 s per campaign event. Half of it re-resamples the reused background
    at every event, and 60–70 % of the rest is numpy copying transposed temporaries.
  - **The freeze-out surface for particlize:** ~7 s per MUSIC run, plus a full grid copy.
  - In O+O the same per-cell stages dominate (freeze-out, copies, the diagnostic), on a ±15 fm grid
    where the matter never leaves ±7 fm.
- **On this machine all three are host-limited, but not by much more than the GPUs allow.** With the
  other GPU's jobs running, each job is 43 % (O+O), 20 % (two-stage) and 10 % (hydro only) slower.
  The GPUs are 61–84 % busy, so CPU-side changes can add at most ~1.5× (O+O) and ~1.35× (two-stage
  campaign) before the GPUs saturate.
  - **Hydro only:** 12 jobs per GPU (new, 1,223 events/h, +15 % over 8) already runs at its GPU
    ceiling.
- **The GB10 and the M3 Max are GPU-limited** (83–93 % busy, 4–9 of 16–20 cores busy). The difference
  is the machines' balance:
  - per plain two-stage event a Zen 2 core of the 3960X needs ~2× the core-seconds of a GB10 or M3
    Max core (97 vs 47–48);
  - the RTX 3090 needs less GPU time than the GB10 (10.6 vs ~15 s);
  - this machine has 12 cores per GPU, against 20 and 16.

  So CPU work sets the rate here, and on any machine with few or slow cores per GPU. GPU work sets it
  on the GB10 and the M3 Max, and here too once the CPU work has dropped by about a third.
- **What would raise throughput** (§5, none implemented yet; *update 2026-10-12: A1–A9 are implemented and measured, C1 is tested, and A10/A11 are implemented and measured (O+O +46 %); see §8*):
  - **(A) Cheap fixes with bit-identical output** cut the CPU per event by 19 % (two-stage campaign)
    to 30 % (hydro only):
    - in the writer, recognise a reused background instead of resampling it again, and do the resampling
      as plain matrix products (4× faster, identical output);
    - in MUSIC, skip source fills that only produce zeros, add a switch for the diagnostic, pack into
      pinned memory, and stop the store reallocating.
  - **(A8) Start MUSIC_2 from MUSIC_1's state:** a further −26 core-s and −6 to −13 % GPU per jet event
    (bit-identical expected, to be checked).
  - **(C) Less GPU work** is the only lever on the GB10 and the M3 Max:
    - **O+O on a ±10.5 fm MUSIC grid** halves the cells, and with them most of the GPU and host work
      (XML only, needs a physics check);
    - for Au+Au, skipping vacuum cells or faster kernels.
  - **(B) Strings and freeze-out on the GPU** help where cores are scarce, but must stay cheap in GPU
    time for the GB10 and the M3 Max.
- **Optimized versions** (§6):
  - O+O: grid → diagnostic switch, packing → freeze-out on the GPU.
  - Hydro only: 12 jobs per GPU now; the fixes free cores, GPU work sets the rate.
  - Two-stage: writer fixes → MUSIC_2 from MUSIC_1's state → MUSIC fixes → freeze-out on the GPU.
- **Particlize file only** (`--particlize-only`, §7): on this machine **+15 % events/h** for the
  two-stage campaign (917 against 796 by makespan, 10 jobs per GPU), −11 % CPU per event, 8 GB less
  host memory and 60 % less disk (95 against ~235 MB per event); the GPU work is the same. It still
  re-resamples the background at every event (6 s per jet event) and still stores MUSIC_2's
  evolution (~3 s), although it needs neither (A1, A9).
- **Corrected:** in REPORT_OO.md the "GPU → host copy for Matter" was the grid copy for MUSIC's own
  freeze-out search plus the frame packing, and the "host side of the RK steps" was the string
  deposition.
- **Noticed on the way:** the same O+O seed gave identical bulk hadrons in 5 runs, but different jet
  hadron counts from event 5 on (three different outcomes). Not investigated.

## 1. Per event, one job alone on one RTX 3090

All jobs: image `jhputschke/xscape-prod:cu124` = `cu124-20261011-84df104` (`066094dfcb4f`; X-SCAPE `84df104`,
MUSIC4GPU `bc8543a`, js-contrib `8765281`), GPU0, `OMP_NUM_THREADS=3`, `OMP_WAIT_POLICY=passive`,
`MUSIC_CUDA_SYNC=block`, `LBT_TABLE_CACHE`, `MUSIC_PROFILE=1`, nothing else running.

| | O+O ISR, hadrons only | Au+Au hydro only | Au+Au two-stage, campaign setting | two-stage campaign, particlize file only (§7) | Au+Au two-stage, plain |
|---|---|---|---|---|---|
| job | 6 events, seed 101 | 4 events, seed 1 | 15 events = 1 background × 3 pT̂ bins × 5 jets, particlize both, seed 1 | the same with `--particlize-only` | 3 events, a background every event, particlize both, seed 1 |
| **wall s/event** | 8.1 | 25.8 | 45.2 (event with the background 68.8, jet events 43.5) | 38.0 (59.1, 36.5) | 75.4 |
| **GPU busy s/event** | 1.34 | 4.95 | 6.1 (background event 10.4, jet events 5.7) | 5.9 | 11.7 |
| GPU share of the wall time | 17 % | 19 % | 13 % | 15 % | 16 % |
| **CPU core-s/event** | 10.6 | 40 | 74 | 66 | 133 |
| of which the main thread | 6.6 | 21 | 39 | 32 | 64 |

- **GPU busy** = mean SM % (`nvidia-smi dmon`, 1 s) × wall time, per event window: the time at least one
  MUSIC kernel runs. MUSIC's own timers agree (the RK steps without the source fill: 4.2 s/event for
  hydro only).
- **Everything else is host work, 81–87 % of an event**, as for O+O. The GPU kernels per RK sub-step take
  ~4.2 ms for Au+Au and ~3.4 ms for O+O: the same 100 × 100 × 60 grid (see §5, O+O grid).
- **CPU** = the job's threads' user + sys time (`/proc/<pid>/task/*/stat`), startup included. With 3
  OpenMP threads the two workers add about as much as the main thread: in hydro only ~85 % of it in
  the string deposition, in the two-stage chain about half, the rest in the freeze-out search, the
  droplets and the GPU → host converters.

## 2. Where the host time goes

py-spy `--native --idle`, 10 Hz, main thread, same jobs as above (py-spy adds 5–8 %). *Main* = the main
thread's wall time per event. *Core-s* = CPU time of all threads, attributed second by second: the main
thread's to its stages in that second (none to waiting for the GPU, where it sleeps), the OpenMP
workers' to the stages with parallel regions. Exact for the long stages (string deposition, writer),
approximate for the per-step stages that alternate within a second (freeze-out, copies, packing).

| stage | O+O<br>main s / core-s | Au+Au hydro only<br>main s / core-s | Au+Au two-stage, campaign<br>main s / core-s | two-stage campaign, particlize file only<br>main s / core-s | Au+Au two-stage, plain<br>main s / core-s |
|---|---|---|---|---|---|
| String deposition (CPU, OpenMP) | 0.43 / 0.7 | 10.80 / 26.4 | 11.46 / 27.7 | 11.10 / 25.2 | 22.20 / 51.5 |
| Droplet deposition (CPU, OpenMP) | – | – | 2.55 / 5.4 | 2.48 / 6.1 | 3.17 / 6.8 |
| Freeze-out surface (Cornelius, CPU) | 1.75 / 3.3 | – | 6.91 / 14.0 | 6.45 / 17.7 | 14.90 / 31.0 |
| GPU → host copy of both grids (for Cornelius; hydro only: once at the end) | 0.68 / 1.6 | 0.03 / 0.1 | 1.44 / 2.8 | 1.51 / 4.1 | 4.00 / 9.4 |
| Freeze-out test without a surface (copy + edge scan) | – | 1.12 / 2.1 | – | – | – |
| Waiting for the GPU's RK kernels | 0.93 / 0.0 | 4.15 / 0.0 | 4.57 / 0.0 | 4.74 / 0.0 | 9.10 / 0.0 |
| RK steps, host side (launches) | 0.13 / 0.1 | 0.23 / 0.2 | 0.21 / 0.2 | 0.17 / 0.2 | 0.37 / 0.3 |
| Frames to host memory: GPU pack + copy | 0.63 / 0.6 | 1.78 / 1.6 | 1.58 / 1.5 | 1.55 / 1.5 | 3.20 / 3.2 |
| Frames to host memory: append to the store | 0.33 / 0.3 | 1.58 / 1.7 | 1.25 / 1.2 | 1.37 / 1.3 | 2.37 / 2.4 |
| Eccentricity diagnostic (4× per MUSIC run) | 1.55 / 1.3 | 1.55 / 1.3 | 1.33 / 1.1 | 1.66 / 1.8 | 3.17 / 2.4 |
| MUSIC → framework copy (`bulk_info`) | 0.33 / 0.7 | – | 0.10 / 0.2 | 0.09 / 0.1 | 1.03 / 2.7 |
| MUSIC grids, per-event init, other | 0.28 / 0.5 | 0.25 / 0.7 | 0.35 / 0.8 | 0.30 / 0.5 | 0.93 / 2.0 |
| Writer: read MUSIC frames | – | 2.40 / 2.4 | 1.91 / 1.9 | – | 1.93 / 1.9 |
| Writer: resample to the output grid | – | 1.88 / 2.3 | 12.09 / 12.4 | 6.01 / 6.5 | 8.83 / 8.0 |
| Writer: HDF5 compress + write | 0.02 / 0.0 | 1.18 / 3.0 | 2.59 / 6.1 | 1.17 / 1.2 | 5.43 / 10.5 |
| Writer: other | – | 0.12 / 0.1 | 0.46 / 0.4 | 0.49 / 0.4 | 0.53 / 0.3 |
| Medium lookups for Matter/LBT (`GetHydroInfo`) | 0.72 / 0.8 | – | 0.07 / 0.1 | 0.05 / 0.0 | – |
| Matter, LBT, liquefier | 0.18 / 0.2 | – | 0.15 / 0.1 | 0.13 / 0.1 | 0.53 / 0.5 |
| iSS, Pythia + ISR, 3D MC-Glauber, jet hadronization | 0.60 / 0.7 | 0.18 / 0.2 | 0.09 / 0.0 | 0.07 / 0.1 | 0.27 / 0.1 |
| Startup, other | 0.15 / 0.0 | 0.30 / 0.1 | 0.03 / 0.0 | 0.07 / 0.0 | 0.30 / 0.0 |
| **total** | **8.7 / 11** | **27.5 / 42** | **49.1 / 76** | **39.4 / 67** | **82.3 / 133** |

The O+O and hydro-only columns are from second profiles (`prof_oo_spy10_omp3`, `prof_norm_spy10_omp3`,
with the timeline). The first ones agree within run-to-run spread, except the eccentricity
diagnostic, which varies from event to event: 0.93 s per O+O event in REPORT_OO.md, 1.55 s here.

What the stages are, and what is notable:

- **String deposition** (`Advance::prefill_hydro_source_on_cpu`, OpenMP): before each GPU sub-step that
  has sources, every one of the 600,000 cells evaluates the strings within 8 σ (up to four `erf`, an
  `exp`, `cosh`, `sinh` per string–cell pair), into a buffer the kernel adds. With
  `evolve_QCD_string_mode 4` the strings deposit at τ = 0.40 and 0.42 fm/c only: 4 fills per MUSIC run,
  **~2 s each with 3 threads (~6 core-s)**, for 1,050–1,350 strings. Au+Au has 20–80× more strings than
  O+O (15–60 per event), so this is the largest CPU item of every Au+Au production, and **in the two-stage chain it
  is paid twice per collision: MUSIC_2 deposits exactly the same strings again** (its first 6–13
  frames are bit-identical to the background's; the job log says so per event). In the campaign
  MUSIC_2 runs for every jet, so the identical deposition runs 16 times per background.
- **Extra empty fills (hydro only):** the fill also runs at every step where 3dMCGlauber's
  *electric-charge* string list is non-empty, although the GPU path never deposits charge
  (`turn_on_QS` must be 0). `HydroSourceStrings::has_active_sources_current_tau()` tests all four lists.
  In the 4 events: 0, 64, 54 and 204 extra fills of ~20 ms, ~1.6 s and ~5 core-s per event on average,
  for zeros. In the jet leg the droplets need the fill anyway.
- **Droplet deposition** (jet leg): the liquefier's droplets, at most steps of MUSIC_2: 2.6 s/event.
- **Freeze-out surface** (Cornelius, OpenMP over η slices), with `--write-particlize both` for both legs:
  every 5 steps on the host copy of both grids, **~6–7 s per MUSIC run**, plus the GPU → host copy of
  both grids it needs (1.2–2 s per run; 21 float fields of 600,000 cells, converted to double).
  Hydro only builds no surface and instead tests max(e) < e_fo on the GPU, but then copies the whole
  grid to the host to scan its transverse edge (0.9 s/event).
- **Waiting for the GPU's RK kernels**: the GPU's own work, ~4.2 ms per sub-step. With
  `MUSIC_CUDA_SYNC=block` the main thread sleeps meanwhile: no CPU.
- **Frames to host memory**: every 5 steps a GPU kernel packs the frame (19 MB) into CUDA managed
  memory, which the host then `memcpy`s out page fault by page fault (~13–17 ms per frame, where a
  pinned copy would take ~1–2 ms), and appends it to a `std::vector` that reallocates as it grows.
  2.7–3.4 s per Au+Au MUSIC run together (O+O: 1.0 s).
- **Eccentricity diagnostic** (`output_momentum_anisotropy_vs_etas`, serial, EOS calls per cell): at 4
  fixed steps of every MUSIC run, ~0.25–0.4 s each, written to `.dat` files nobody reads; no switch.
- **The writer** (Python, one thread):
  - *Read*: MUSIC's store → numpy (`get_native_evolution_numpy`, or frame by frame), 1.9–2.8 s per event.
  - *Resample* to the 64 × 64 × 32 output grid (`bulk_sources.resample`): three `np.einsum` passes in
    float64 per frame, **~30–60 ms per frame, 60–70 % of it numpy copying a transposed
    intermediate** (`bmm_einsum` → `reshape`). The same three passes written as plain matrix products
    over contiguous memory take 15 ms per frame against 62 ms, **bit-identical** (checked on 21 frames,
    11 M values; `bench/auau_prof_20261011/resample_test/`).
  - **`PairH5Writer` reads and resamples the background leg at every event, also when it is reused**
    (`_write_event` hashes the resampled background frames to recognise a reused background). In the
    campaign that is 14 extra background resamples per background: ~6 s per jet event.
  - The resampling also gets slower during a long job: in the campaign job, events 2–4 resampled both
    legs in 6–7 s, events 5–15 in 10–16 s. Not explained (the memory state of the process, perhaps
    huge pages for numpy's temporaries no longer granted). The contiguous version does no strided
    copies, so it should be less exposed; not tested in a long job.
  - *HDF5*: Blosc-zstd compression of the output grids, 1.2 s (hydro only) to 5.4 s (plain) per event.
- **Matter, LBT, liquefier, Pythia, 3D MC-Glauber, iSS**: together < 1 % of an Au+Au event.

## 3. Host or GPU: one GPU against two on this machine

As the O+O test S1: the same jobs per GPU on GPU0 alone (GPU1 idle, all 24 cores for one GPU's jobs),
against both GPUs. If the host were not a limit, each GPU would do the same in both.

| production (setting) | jobs per GPU | GPUs | events/h per GPU, makespan (per event) | s/event in a job | core-s/event | GPU busy | threads busy (of 48) |
|---|---|---|---|---|---|---|---|
| O+O, hadrons only (REPORT_OO.md S1, R4) | 8 | 1 | **2,506** | 10.8 | 13.0 | 76 % | |
| | 8 | 2 | **1,753** | 15.5 | 21.6 | 61 % | ~21.5 |
| Au+Au two-stage, campaign (`jetc_j10_gpu0`, `jetc_j10_2gpu`) | 10 | 1 | **488** (554) | 65.1 | 91 | 78 % | 12.8 |
| | 10 | 2 | **398** (464) | 77.8 | 117 | 66 / 67 % | 26.6 |
| Au+Au hydro only (`norm_j8_gpu0`, `norm_j8_2gpu`) | 8 | 1 | **614** (645) | 44.7 | 51 | 82 % | 9.1 |
| | 8 | 2 | **534** (590) | 49.0 | 60 | 72 / 72 % | 18.4 |

- **All three are host-limited on this machine, to different degrees.** When the other GPU's jobs run
  too, each job becomes slower by 43 % (O+O), 20 % (two-stage) and 10 % (hydro only), and every event
  costs 66 %, 28 % and 18 % more CPU. The jobs are the same; what they share is the 24 cores, their
  caches and the memory bandwidth.
- **O+O is the most host-bound** because its host work is mostly streaming 600,000-cell arrays for a
  small fireball (freeze-out, grid copies, packing): bandwidth, not arithmetic.
- **Hydro only at 8 jobs per GPU is limited mainly by the job count:** each job spends ~10 % of its event
  on the GPU, so 8 jobs keep it ~80 % busy at most, and 18 of 48 threads are busy.
  With 12 jobs per GPU (`norm_j12_2gpu`, 2 threads each) it gives **1,223 events/h by makespan
  (1,325 per event), +15 %**, at the same 60 core-s per event, with the GPUs 83–85 % busy, 21 of 48
  threads busy and 66.5 GB of host memory. That is within ~7 % of the GPU ceiling (§4): **hydro only
  is GPU-bound on this machine at 12 jobs per GPU.**
- **The two-stage campaign** (`jetc_j10_2gpu`, today's image): 796 events/h by makespan, 927 per event,
  as R5 on 2026-10-10 (805 / 951) with the previous image. Host memory peaked at 103.8 GB.
- **The GPUs are already 61–84 % busy, so the CPU alone can lift the rate only so far** before the
  GPUs saturate: ~1.5× for O+O, ~1.35× for the two-stage campaign, and for hydro only nothing beyond the
  12-jobs-per-GPU setting (§4).
- For the record, the earlier CPU savings that removed spinning threads rather than serial work did
  little for events/h: `MUSIC_CUDA_SYNC=block` (−30 % CPU) +3.5 % at 8/GPU, one OpenBLAS thread
  (−17 %) +5–7 % (BENCHMARK_2x3090.md). The changes of §5 shorten each job's serial host time, which
  is what sets its event rate, so they should convert better; that is untested.

## 4. All machines: what limits the event rate

The best measured setting of each production per machine. Rates by makespan where measured that way
(per-event rates in parentheses); GB10 and M3 Max numbers from BENCHMARK_GB10.md, BENCHMARK_M3MAX.md
and the O+O README, on code of 2026-10-03 to 10-10. *GPU-s/event* = mean GPU busy × GPUs × 3600 /
(events/h); the *ceiling* is the rate at 90 % GPU busy with that GPU work per event.

| machine | cores per GPU | production (setting) | events/h | GPU busy | cores busy | GPU-s/event | ceiling at 90 % | limit |
|---|---|---|---|---|---|---|---|---|
| 2 × RTX 3090, Threadripper 3960X | 12 Zen 2 (24 threads) | O+O, 8/GPU | 3,505 (3,714) | 61 % | ~21 of 48 threads | 1.24 | ~5,200 | host |
| | | Au+Au hydro only, 12/GPU | 1,223 (1,325) | 84 % | 21 of 48 threads | 4.9 | ~1,310 | GPU (nearly) |
| | | Au+Au two-stage campaign, 10/GPU | 796 (927) | 66 % | 27 of 48 threads | 6.0 | ~1,080 | host |
| | | Au+Au two-stage campaign, particlize file only, 10/GPU | 917 (1,071) | 76 % | 27 of 48 threads | 5.9 | ~1,090 | host |
| | | Au+Au two-stage plain, no particlize, 10/GPU (D1, 2026-10-10) | 497 (588) | 72 % | 23 of 48 threads | 10.4 | ~620 | host |
| GB10 | 20 (10 X925 + 10 A725) | O+O, `-j 8` | (1,620) | 92 % | | 2.0 | reached | GPU |
| | | Au+Au two-stage plain, `-j 6`, no particlize | (205) | 86 % | ~9 of 20 | ~15 | reached | GPU |
| M3 Max | 16 (12 P + 4 E) | Au+Au two-stage plain, `-j 3`, no particlize | 263 (284) | 83 % | 4.3 of 16 | 11.4 | ~285 | GPU |

Per plain two-stage event, one job alone, no particlize:

| | GB10 | M3 Max | 2 × RTX 3090 |
|---|---|---|---|
| CPU | 48 core-s (4 threads; 37 s/event) | 47 core-s (12 threads; 21.4 s/event) | 97 core-s (3 threads; 57.9 s/event) |
| GPU | ~15 s | ~11–15 s | 10.6 s |
| CPU core-s per GPU-s | ~3 | ~3.5 | ~9 |

- **The GB10 and the M3 Max are GPU-limited, this machine is host-limited**, for O+O and for Au+Au
  alike. The code is the same; the balance of the machines is not:
  - a Zen 2 core of the 3960X needs about twice the core-seconds of a GB10 X925 or an M3 Max P-core
    for the same host work;
  - the RTX 3090 does the GPU work ~1.4× faster than the GB10's GPU (10.6 vs ~15 s per plain event);
  - and this machine has 12 cores per GPU, against 20 (GB10) and 16 (M3 Max).
  So per GPU-second this machine has to supply several times more CPU than the others, with fewer
  cores. Under load the CPU per event also grows, by 1.5–2× (contention, §3); on the GB10 and the
  M3 Max 4–9 cores stay busy and the rest idle.
- **Consequences:** CPU work per event decides the rate here and on any machine with few or slow
  cores per GPU (cluster or OSG-style slots with 4–8 cores per GPU). GPU work per event decides it on
  the GB10, the M3 Max and machines like them. And here too it decides the rate once the CPU work
  has shrunk by 25–35 %.
- **The GPU work per event is the same kernels everywhere:** ~4.2 ms per RK sub-step for 600,000
  cells on the RTX 3090, ~7.25 ms on the GB10, whether the event is O+O or Au+Au, because both run on
  the same 100 × 100 × 60 grid.

## 5. What would raise throughput

Savings are core-s per event of one job alone on this machine (§2), for H = hydro only, C = two-stage
campaign, P = two-stage plain, O = O+O. Under load a core-s costs more (contention), so the relative
saving is what carries over. None of these was implemented or run; the resample change was only
timed and checked for identical output on synthetic frames.

**A. Less CPU, nothing added on the GPU, output bit-identical**

| | change | where | saves (core-s/event) | effort |
|---|---|---|---|---|
| A1 | `PairH5Writer`: recognise a reused background without reading and resampling it again (keep the first event's frame hashes and edge maxima per background) | js-contrib `pair_h5.py` | C ~7 | small |
| A2 | `resample`: the three passes as matrix products over contiguous memory (4× faster, identical output on 11 M test values) | js-contrib `bulk_sources.py` | H ~2, C ~4 (after A1), P ~6 | small |
| A3 | `HydroSourceStrings::has_active_sources_current_tau()`: ignore the electric-charge list when charges are not evolved | MUSIC4GPU | H ~5 | tiny |
| A4 | a switch for `output_momentum_anisotropy_vs_etas` (off in production); its `.dat` files are not used | MUSIC4GPU (+ XML) | H 1.8, C 1.1, P 2.4, O ~0.9 | small |
| A5 | pack the stored frame into pinned host memory with an async copy instead of managed memory + `memcpy` | MUSIC4GPU `CUDAPipelines.cu` | H ~1.3, C ~1.2, P ~2.6, O ~0.5 | small |
| A6 | append frames to the store without reallocating (reserve, or one buffer per frame) | MUSIC4GPU `HydroinfoMUSIC` | H ~1.2, C ~0.9, P ~1.9, O ~0.3 | small |
| A7 | the transverse-edge test of the no-surface freeze-out check as a GPU reduction (no full-grid copy) | MUSIC4GPU `evolve.cpp` | H ~0.9 | small |
| A8 | **MUSIC_2 starts from MUSIC_1's state** at the last step before a droplet can deposit, instead of from τ0 (its leading frames are bit-identical to the background's anyway). Removes MUSIC_2's string deposition and its first 30–65 of ~500 steps. A smaller variant keeps MUSIC_1's four string-fill buffers for MUSIC_2 (CPU part only) | MUSIC4GPU + X-SCAPE | C ~26, P ~26; GPU −6 to −13 % of the jet leg | moderate |
| A9 | with `--particlize-only`: don't let MUSIC_2 pack and store its evolution, which nothing reads (a per-instance `output_evolution_to_memory`, switched off by `PairH5Writer` when `read_jet=False`) | MUSIC4GPU + X-SCAPE + js-contrib | particlize-only ~2.8, and the jet leg's ~2 GB store per job | small |

A1–A7 together: H ~12 of 40 core-s (−30 %), C ~14 of 76 (−19 %), P ~13 of 133 (−10 %), O ~2 of 10.6. With A8: C ~40 of 76 (−53 %), P ~39 of 133 (−29 %). A8 should be bit-identical (the
legs agree bit for bit before the first droplet), but that has to be checked when it is built: a
droplet depositing in a string step, for instance, would need the string part kept in double.

**B. Move CPU work to the GPU** (less CPU, a little more GPU time, not bit-identical: GPU math
differs in the last ulp, and the surface's cell order may change)

| | change | saves (core-s/event) | GPU cost (estimate) | effort |
|---|---|---|---|---|
| B1 | string (and droplet) deposition on the GPU, binned as on the CPU | H ~20 (after A3), C ~2 (after A8: once per background) + droplets ~5, P ~26 (after A8) | ~0.1–0.3 s per MUSIC run | large (CUDA and Metal) |
| B2 | freeze-out surface on the GPU, which also removes the grid copy for it | C ~17, P ~40, O ~5 | ~0.1–0.3 s per MUSIC run | large |
| B3 | output-grid frames straight from the GPU (resample inside the pack kernel) instead of pack → store → read → resample | H ~5, C ~5, P ~8 | small | moderate (float64 on the GPU to stay close) |

**C. Less GPU work** (the only lever on the GB10 and the M3 Max, and on this machine once A is done)

| | change | GPU saved | CPU saved | effort |
|---|---|---|---|---|
| C1 | **O+O: MUSIC grid ±10.5 fm instead of ±15 fm** (70 × 70 instead of 100 × 100 transverse cells, 0.49×); the O+O output grid is ±9.84 fm | ~50 % of the kernels, which scale with cells | most per-cell host work (freeze-out, copies, packs, diagnostic) ~40–50 % | XML only, but needs a physics check (edge flags, observables) |
| C2 | A8's late start of MUSIC_2 | 6–13 % of the jet leg | see A8 | see A8 |
| C3 | skip vacuum cells in the kernels (an active box updated every few steps) | Au+Au maybe 20–40 %, O+O more (not measured) | – | large; not bit-identical |
| C4 | faster RK kernels (`gpu_make_delta_qi`, ~60 % of the kernel time: Newton–Brent per cell) | unknown; needs `nsys`/`ncu`, not available here | – | unknown |

**What each machine gets from them**

- **2 × RTX 3090:** A converts into events/h until the GPUs saturate (ceilings in §4). Two-stage
  campaign: A1–A7 ~+20–25 %, with A8 up to the GPU ceiling (~+35 % over today, more with A8's GPU
  saving). O+O: A4–A6 ~+20 %, B2 up to the GPU ceiling (~+50 %), and C1 roughly doubles the ceiling.
  Hydro only is already at its GPU ceiling with 12 jobs per GPU: only C helps.
- **GB10 and M3 Max (GPU-limited):** A and B do not raise events/h; they free cores and memory
  (for `hadronize.py` next to the production) and shorten each job. Events/h needs C: for O+O C1
  (~2× if the kernel time halves with the cells), for the two-stage chain C2 (+3–6 % plain, +6–13 %
  campaign) and C3/C4. B adds a little GPU time, so on these machines it must stay small.
- **Machines with few cores per GPU** (cluster or OSG-style slots with 4–8 cores per GPU) are more
  host-limited than this one: there A and B pay almost one for one.

## 6. Optimized versions per production

The three productions differ enough that each is best served by its own order of changes:

**O+O (`prod_OO_isr`)**: little string work (~50 strings), short events, a small fireball on the
Au+Au-sized grid.
1. **C1, the grid**: ±10.5 fm instead of ±15 fm in `<IS>` (`grid_max_x/y`), if a few hundred events
   show no grid-edge hits (MUSIC's "grid is too small", `hit_grid_boundary`) and unchanged observables
   (dN/dη, v₂, jet hadrons). The single change
   that cuts CPU and GPU work alike, everywhere.
   - In 6 events (seed 101, default outputs, `prof_ooh_clean_omp3`; its `.h5` since deleted, remade by
     `prof_auau.sh ooh clean 3`) the energy density exceeded
     ~0.3 GeV/fm³ (freeze-out) only within |x|, |y| ≲ 4.2 fm, exceeded 0.001 GeV/fm³ only within
     6.4 fm, and was exactly 0 beyond 7 fm at all times.
   - In η it does reach the output grid's edge (1.6 GeV/fm³ at |η| = 4.84), so η stays at ±6.
   - The lower limit for x and y is the ±9.84 fm output grid, which MUSIC's grid must contain: hence
     ±10.5 fm (70 cells).
2. A4, A5, A6 (diagnostic switch, pinned pack, store append): ~−2 s of 8.1 s per event.
3. B2 (freeze-out on the GPU): the largest remaining item, ~1.8 s + 0.7 s for its grid copy.
- This machine: from 3,505 events/h toward the GPU ceiling of ~5,200 (§4) with 2–3; C1 roughly doubles
  the ceiling. GB10: only C1 raises its 1,620 events/h.

**Au+Au hydro only (`prod_AuAu_0_10`)**: no surface, no jet; the strings and the writer are most of
the host time.
1. A3, A2, A4–A7, all small and bit-identical: ~−30 % CPU per event. A3 alone is a one-line test.
2. B1 (strings on the GPU): the largest item, 10–11 s / ~26 core-s per event (~21 after A3).
3. B3 (output-grid frames from the GPU): pack, store, read and resample, ~7.6 s per event today, become
   a kernel and a 2 MB copy per frame.
- This machine: **run it at 12 jobs per GPU** (1,223 events/h by makespan, +15 % over 8, 66.5 GB);
  the GPUs are then 84 % busy, so 1–3 add little events/h here, they free cores (e.g. for
  hadronization). More events/h need less GPU work: C3/C4. On a machine with fewer cores per GPU, 1–3
  are the lever.

**Au+Au two-stage (`prod_AuAu_0_10_jet`), campaign setting**: two MUSIC runs per collision, the second
one per jet.
1. A1 + A2 (js-contrib only, bit-identical): −11 core-s per event, and the writer no longer slows
   down during a job.
2. **A8 (MUSIC_2 from MUSIC_1's state)**: −26 core-s and −6 to −13 % GPU per jet event. Together with 1,
   about half the CPU per event.
3. A4–A6, then B2 (freeze-out on the GPU, both legs; needed for particlize): the next largest item,
   ~17 core-s per event.
- This machine: from 796 events/h today (R5: 805) toward the GPU ceiling of ~1,080, ~1,170 with A8's
  GPU saving (§4). GB10 / M3 Max: A8's GPU part (and C3/C4).
- **If the hydro history is not needed: `--particlize-only` today** (+15 % events/h here, 60 % less
  disk; §7), then A1 and A9, which matter most in that mode.
- The plain mode (a background every event) gains the same in absolute terms, but its GPU ceiling is
  half the campaign's: the campaign's background reuse is already the biggest saving of all.

## 7. Two-stage campaign with the particlize file only

`run_prod_jet.py … --write-particlize both --particlize-only` writes only `<stem>_particlize.h5`: both
legs' freeze-out surfaces (the background's once per background) and the final partons, all that
`hadronize.py` needs. There is no pair file, so no hydro history (`arr`, `arr_bg`), no `source/` and no
`shower/`. Same campaign setting and seeds as `jetc`: the job logs show the same collisions, droplets
and pT̂ event by event.

**One job alone** (`prof_jetq_clean`, `prof_jetq_spy10`; also the extra columns in §1 and §2):

| | pair + particlize (default) | particlize file only | change |
|---|---|---|---|
| wall s/event, 15 events (event with the background / jet events) | 45.2 (68.8 / 43.5) | 38.0 (59.1 / 36.5) | −16 % |
| GPU s/event | 6.1 | 5.9 | the same work |
| CPU core-s/event (main thread) | 74 (39) | 66 (32) | −10 % |
| output per event | ~235 MB | 83 MB (seed 1); 95 MB (20 seeds, below) | −60 % |

**10 jobs per GPU, both GPUs** (`jetq_j10_2gpu` against `jetc_j10_2gpu`: same seeds, settings, stagger):

| | pair + particlize | particlize file only | change |
|---|---|---|---|
| **events/h, makespan (per event)** | 796 (927) | **917 (1,071)** | **+15 %** |
| s/event in a job | 77.8 | 67.4 | −13 % |
| core-s/event | 116.6 | 103.7 | −11 % |
| GPU busy | 66 / 67 % | 75 / 76 % | |
| host memory peak | 103.8 GB | 95.9 GB | −8 GB |
| per-job VmHWM, mean | 6.10 GB | 5.81 GB | |
| disk for the 300 events | ~71 GB | 28.6 GB | −60 % |

All 20 jobs wrote 15 particlize events each and no pair file; no OOM, every log `host sync: block`.

**What it saves**, per jet event (main thread, §2): the jet leg's read (1.9 s), its resampling (~6 s)
and the pair file's compression and write (1.4 s, ~5 core-s).

**What it still does without needing it:**
- **The background is read back and resampled at every event**, only to recognise a reused one
  (`PairH5Writer` with `store_evolution=False` still hashes the resampled background): **6.0 s per jet
  event, 16 % of it**, and 77 % of that is numpy's transposed copy. The same frames took 2.1 s in
  event 1 (the slowdown of §2). A1, or A2 alone, removes most of it.
- **MUSIC_2 still packs and stores its whole evolution** every 5 steps (1.5 + 1.4 s per event),
  although nothing reads the jet leg: `PairH5Writer.attach` sets the jet leg's `dump_hydro_only`
  whatever `read_jet` is, and `output_evolution_to_memory` is one setting for both legs. A9 saves it,
  and the jet leg's in-memory store.
- Everything else is the full output's: strings, freeze-out (the particlize file's content),
  droplets, GPU.

With A1 + A4 + A9 a particlize-only jet event would lose another ~10 s of its 37.6 s (main thread,
py-spy), and with A8 another ~11 s: ~16 s, of which ~6 s on the GPU. Its GPU ceiling on this machine
is ~1,090 events/h (5.9 GPU-s per event at 90 %), 19 % above the 917 measured.

**When to use it:** for any campaign whose hydro history is not needed. On this machine it gives
+15 % events/h, −11 % CPU per event, 8 GB less host memory at the peak and 2.5× less disk and upload
(~24 MB/s at 917 events/h, against ~52 MB/s for the full output). On GPU-limited machines (GB10,
M3 Max) the GPU work is the same, so mainly the disk, upload and memory savings remain.

## 8. Update 2026-10-12: A1–A9, C1, A10 and A11 measured, and what is left

Measured on this machine with `jhputschke/xscape-prod:cu124-music2restart` (`d4acf127186f`, js-contrib `02bed42`, branch `music2_restart`). The details are in [BENCHMARK_2x3090.md](../BENCHMARK_2x3090.md), sections "The CPU fixes A1–A9 and the MUSIC_2 restart" and "O+O campaign".

**Throughput** (events/h by makespan, the launcher settings of each production):

| production | before (§3/§4) | now | change |
|---|---|---|---|
| Au+Au two-stage campaign, 10/GPU | 796 | **1,121** (965 without the restart) | +41 % |
| two-stage campaign, `--particlize-only` | 917 | **1,237** | +35 % |
| Au+Au hydro only, 12/GPU | 1,223 | **1,601** | +31 % |
| O+O, ±15 fm, 8/GPU, hadrons only | 3,505 (3,534 on 2026-10-12) | **3,798** | +7 % |
| O+O, ±15 fm, 8/GPU, default outputs, 60 events per job | (R5 at 12/GPU: 3,657) | **4,006** | |
| O+O, **±10.5 fm** (C1, `--user-xml`), default outputs, 60 events per job | – | **7,282** | +82 % over ±15 fm |
| the same with A10 + A11 (`cu124-freezeoutcopies`, below) | – | **10,629** | +46 % over 7,282 |

- **Outputs are bit-identical** to the old image and between restart and no restart, except `source/flux` (≤ 7 ulp; it differs between two runs of the same image too).
- **C1 passed the physics check:**
  - no grid-edge hit in 2,994 events;
  - the hydro and the freeze-out surface are the same to float32 rounding in 1,020 compared events, a few near-vacuum cells aside;
  - the jet and bulk hadrons agree as well as two runs on the same grid do.
- **The CPU estimates of §5 held.** For one two-stage campaign job alone, A1–A8 were estimated at −40 core-s per event; measured −39.9 (74 → 34 core-s; 45.1 → 22.9 s per event).
- **The fixes also removed GPU work,** which §4 had not expected. GPU-s per event:
  - two-stage campaign 6.00 → 5.29;
  - particlize only 5.94 → 5.16;
  - hydro only 4.94 → 3.95.

  So the GPU limits rose above §4's estimates.

**Distance to the GPU limit now** (90 % busy, from the measured GPU-s per event):

| production | §4/§5 estimate | GPU limit now | measured | distance |
|---|---|---|---|---|
| two-stage campaign | ~1,080; ~1,170 with A8 | ~1,225 | 1,121 | 91 %: ~9 % left |
| two-stage, particlize file only | ~1,090 | ~1,255 | 1,237 | 98 % |
| hydro only | ~1,310 | ~1,640 | 1,601 | 98 % |
| O+O ±15 fm | ~5,200 | ~5,900 | 4,006 | 68 % |
| O+O ±10.5 fm | ~2× ±15 fm | ~10,000 | 7,282 | 73 % |
| O+O ±10.5 fm with A10 + A11 | – | ~11,700 | 10,629 | 91 % |

The Au+Au productions are now GPU-bound or nearly so; only C3/C4 (less GPU work) raise them further. O+O is still host-bound: the GPUs are ~65 % busy, and its events/h follow its core-s per event. (With A10 + A11, below, O+O ±10.5 fm reaches 82 % GPU busy and ~91 % of its limit.)

**Correction to B2: the O+O freeze-out is mostly copying.**
- Profiled inside a running campaign (one job of 16; `bench/oo_grid_20261012/C10_spy`, `C15_spy`), the freeze-out step is 54 % of a job's CPU on ±10.5 fm (5.3 of 9.7 core-s per event) and 67 % on ±15 fm (12.0 of 17.9).
- Of that:

  | share of the freeze-out time | ±10.5 fm | ±15 fm |
  |---|---|---|
  | `store_previous_step_for_freezeout`: copying both grids into the "previous step" buffers | 43 % | 53 % |
  | the GPU → host copy of both grids (21 fields, float → double) | 31 % | 34 % |
  | the scan over the grid | 12 % | 6 % |
  | Cornelius' geometry | 14 % | 7 % |

- In a campaign these copies cost about twice what they do in a job alone (memory bandwidth). That is why §2's single-job table understated them.
- Two new bit-identical changes come before B2:

| | change | where | saves (core-s/event, O+O campaign) | effort |
|---|---|---|---|---|
| A10 | sync the GPU state into two alternating buffer pairs instead of copying the previous step's fields after every search (any later reader of the host grids, e.g. `take_checkpoint`'s ε max, syncs itself) | MUSIC4GPU `evolve.cpp` | ±10.5 fm ~1.2, ±15 fm ~4 | small |
| A11 | copy only ε from the GPU for the scan, and gather the full fields only for the cells of cubes the surface crosses | MUSIC4GPU (CUDA + Metal kernel) | most of the copy's 1.6 (±10.5 fm) | moderate |

- **Together ~−30 % CPU per event for O+O:** from ~7,300 toward the ~10,000 GPU limit (+25–35 %, estimate), with almost no GPU work added.
- **B2 then** saves only the remaining ~1.3 core-s and adds GPU time just as the GPU becomes the limit. It stays useful only where a GPU has few cores.
- **Au+Au** runs the same code, so A10/A11 apply there too, but it is within 2–9 % of its GPU limits.

**Revised order:**
- **O+O:** C1 (XML) → A10 → A11 (A10 + A11 measured below: ~10,600 events/h, ~91 % of the GPU limit); then only GPU savings. B2 only for core-poor machines.
- **Au+Au:** C3/C4 for more events/h on this machine.
- **GB10 / M3 Max:** C1 and C3/C4; A8's GPU part is in.

**Update 2026-10-12: A10 and A11 are implemented** (MUSIC4GPU branch `freezeout_copies`, `2ea5a34`; no X-SCAPE or js-contrib code change).
- **A10:** the checked steps go into two buffers in turn (`FreezeoutSteps`). Nothing is copied from one into the other, and the host arenas are no longer synced for the search.
- **A11:** while the GPU holds the state, only ε comes to the host.
  - The cubes the search visits are found from ε, with Cornelius' own test.
  - The other fields are gathered for their corner cells only: from the GPU state, and from a copy of the previous checked step kept on the GPU.
  - CUDA: kernel `gpu_gather_cells`. Metal: a host gather from the shared buffers (not compiled yet).
- **Not covered:** the boost-invariant search and `output_vorticity` read more than the cube corners. They take full copies into the same two buffers.
- **Switch:** `MUSIC_FREEZEOUT_GATHER=0` gives A10 alone.
- **Checked on the GB10** (one job, OMP 4): O+O seed 101 (4 events, ±15 and ±10.5 fm) and Au+Au two-stage (2 events, `--reuse 2`, particlize both).
  - Hydro, particlize and hadron files are bit-identical to the code before (except `source/flux`, which differs run to run anyway).
  - The same holds with `MUSIC_CUDA_FORCE_DISCRETE=1` (the 3090's memory path), with `MUSIC_FREEZEOUT_GATHER=0`, and with a forced gather failure every third check (the fallback).
  - `MUSIC_RESTART_VERIFY=1` reports the checkpoint identical.
- **Freeze-out host time per check on the GB10** (search + copies, one job alone):

  | | before | A10 | A10 + A11 |
  |---|---|---|---|
  | O+O ±15 fm | 22.2 ms | 10.3 ms | 9.2 ms |
  | O+O ±10.5 fm | 14.3 ms | 8.4 ms | 8.6 ms |
  | Au+Au two-stage | 30.8 ms | 21.0 ms | 20.1 ms |

  - Per O+O job (4 events), CPU drops from 19.5 to 15.6 core-s (±15 fm) and from 15.2 to 13.2 (±10.5 fm).
  - On the GB10, A10 gives most of the saving: with coherent memory the full copy only converts float to double.
  - A11 removes the PCIe transfer and the host memory traffic, which matter on the 3090 under load (§8 above: the copies cost twice as much in a campaign). Its gain there: +5.6 % events/h on top of A10 (below).
- **What remains per check** (O+O ±15 fm): Cornelius' geometry and EOS calls for the ~760 surface elements, ~5.8 ms; the ε scan 1.5 ms; the gather 1.2 ms; the ε copy 0.7 ms.
  - The geometry is now the largest freeze-out item. It was not looked into: per-cube allocations in `Cornelius::find_surface_4d`, five EOS calls per element.

**A10 and A11 measured on the 3090** (2026-10-12, later the same day; `jhputschke/xscape-prod:cu124-freezeoutcopies`, `7785630caf6a`, MUSIC4GPU `2ea5a34`; details in [BENCHMARK_2x3090.md](../BENCHMARK_2x3090.md#oo-a10a11)). O+O ±10.5 fm, 8/GPU, OMP 3, default outputs, 60 events per job, three runs back to back:

| run | events/h | CPU core-s/event | GPU busy | GPU-s/event | GPU limit (90 %) |
|---|---|---|---|---|---|
| old image (control rerun of the 7,282 run) | 7,243 | 10.3 | 66 % | 0.651 | ~9,950 |
| A10 only (`MUSIC_FREEZEOUT_GATHER=0`) | 10,070 (+38 %) | 7.1 | 80 % | 0.572 | ~11,300 |
| **A10 + A11** | **10,629 (+46 %)** | **6.4** | **82 %** | **0.553** | **~11,700** |

- **Better than estimated:** CPU per event fell 38 % (estimate ~30 %), and events/h rose 46 % (estimate +25–35 %), past the old ~10,000 limit. A10 gives most of it; A11 adds +5.6 %.
- **GPU time per event fell 15 % as well.** This is not understood (one guess: the host copies delayed kernel launches). It raises O+O's GPU limit to ~11,700, so the run is at ~91 % of it, like Au+Au.
- **Bit-identical:** the hydro grids (0 of 1.3 × 10¹⁰ values differ) and the freeze-out surfaces in all 540 compared events, for both A10 and A10 + A11. The partons and jet hadrons vary run to run as before, with the same means.
- **Not measured:** ±15 fm, where A10 should save more (~4 core-s per event), and Au+Au, which might gain from the lower GPU time.
- **Consequence for B2:** with O+O now at ~91 % of its GPU limit, B2 has nothing left to gain on this machine.

## Method and files

Everything in `~/prod_test/bench/auau_prof_20261011/`:

- **Single jobs:** `prof_auau.sh WORKLOAD clean|spy10 [OMP]` (workloads `norm`, `jetc`, `jetq`, `jetp`, `jetn`,
  `oo`, `ooh`; run directories `prof_*`). Records the timestamped job log with the `MUSIC_PROFILE=1` timers,
  `nvidia-smi dmon` (GPU0, 1 s), every thread's CPU ticks (1 s) and the job's RSS. py-spy 0.4.2
  (`record --native --idle -r 10`, speedscope format; the first hydro-only profile is raw), started by
  py-spy inside the container (`--cap-add SYS_PTRACE`).
- **Analysis:** `classify_spy.py` sorts each sample by the functions on its stack (first matching rule;
  on the O+O profile of REPORT_OO.md the same rules give its main rows exactly: freeze-out 1.93,
  diagnostic 0.93, medium lookups 0.65, iSS 0.33 s per event) and splits a speedscope
  profile at the job's event ends; `stage_cores.py` adds the threads' CPU per stage; `thread_cpu.py`,
  `gpu_windows.py` (GPU busy per event); `make_table2.py` (§2).
- **Throughput:** `run_auau.sh NAME norm|jetc|jetq J OMP EVENTS` (a copy of `bench/oo_20261011/run_oo.sh`: one
  container per GPU in `$GPUS`, `run_jobs.sh -j J --mps [--stagger]`, passive OpenMP, `block` sync,
  `LBT_TABLE_CACHE`; same monitors and watchdog) and `analyze_auau.py`. One wave per run, seeds from 1.
- **Resampling:** `resample_test/t_resample.py` (the current `resample` on synthetic frames, lazy and whole)
  and `t_resample_fast.py` (the contiguous variant, timing and bitwise comparison).

Caveats:
- One run per setting; earlier repeats on this machine spread by ±2–4 %.
- py-spy slows a job by 5–8 % (the profiled runs took 105–108 % of the clean ones).
- The core-s split by stage is exact for long stages and approximate for those that alternate within
  a second (freeze-out, grid copies, packing), because the threads' CPU is sampled once a second.
- "GPU busy" is the SM utilisation of `nvidia-smi`: the time at least one kernel runs. Under MPS the
  kernels of several jobs can overlap, so 100 % is not a hard ceiling; the GB10 levelled off at
  86–93 %, and §4 uses 90 %.
- Not done: kernel-level GPU profiles (`nsys`/`ncu` are not installed); runs on the GB10 or the M3 Max
  (their numbers are from BENCHMARK_GB10.md and BENCHMARK_M3MAX.md, on older code); a long campaign.
