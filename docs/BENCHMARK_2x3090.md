# prod_AuAu_0_10_jet on 2 × RTX 3090 (Docker): concurrent jobs, OpenMP and MPS

Measured 2026-10-03 with `jhputschke/xscape-prod:cu124` (`b5a36a08c212`, memory fixes included).
Machine: Threadripper 3960X (24 cores / 48 threads), 125 GB RAM, 2 × RTX 3090 (24 GB).

Method as in BENCHMARK_GB10.md: P jobs × 3 events, seeds 1…P, all started at once,
`OMP_WAIT_POLICY=passive`, defaults otherwise (no `--write-particlize`). events/h = Σ over jobs of
3600 / (job's mean s/event). **One container per GPU** (`--gpus device=g`), each running its own
`run_jobs.sh`; two MPS daemons in *one* container hang the second GPU's jobs, but one per
container works. Harness: `bench.sh`, `analyze.py`; raw logs in `bench/<name>/`.

| jobs/GPU (total) | OMP threads | MPS | s/event | events/h | GPU busy | CPU busy (threads of 48) | host mem max | VRAM/GPU |
|---|---|---|---|---|---|---|---|---|
| 1 (1, GPU0 only) | default (48) | – | 32.3 | 112 | 27 % | 10 | 12 GB | 1.0 GB |
| 2 (4) | 12 | no | 47.4 | 305 | 40 % | 16 | 29 GB | 1.8 GB |
| 2 (4) | 12 | yes | 45.9 | 315 | 41 % | 17 | 28 GB | 1.8 GB |
| 3 (6) | 8 | no | 59.2 | 365 | 48 % | 20 | 41 GB | 2.6 GB |
| 3 (6) | 8 | yes | 56.4 | 383 | 48 % | 20 | 39 GB | 2.6 GB |
| 4 (8) | 6 | no (old hand-tuned setup) | 69.7 | 414 | 55 % | 22 | 49 GB | 3.4 GB |
| 4 (8) | 6 | yes (2 runs) | 65.4 | 442, 441 | 55 % | 23 | 48 GB | 3.4 GB |
| 4 (8) | 12 | yes | 65.0 | 444 | | | | |
| 4 (8) | 3 | yes | 76.7 | 377 | | | | |
| 5 (10) | 5 | yes | 77.4 | 466 | 58 % | 25 | 62 GB | 4.2 GB |
| 6 (12) | 4 | yes (2 runs) | 90.9, 87.0 | 476, 498 | 60 % | 27 | 69 GB | 4.9 GB |
| 6 (12) | 8 | yes | 92.1 | 470 | | | | |
| 6 (12) | 2 | yes | 98.7 | 439 | | | | |
| 8 (16) | 3 | yes | 107.7 | **536** | 66 % | 30 | 86 GB | 6.5 GB |
| 8 (16) | 6 | yes | 111.3 | 518 | | | 89 GB | |
| 8 (16) | 6 | no | 119.0 | 485 | | | 87 GB | |
| 10 (20) | 5 | yes | 133.7 | 540 | 70 % | 38 | **105 GB** | 8.1 GB |

Findings:
- **Best: 8 jobs per GPU, `--mps`, 3–6 threads per job: ~520–535 events/h**, +27 % over the
  old setup (4/GPU, 6 threads, no MPS: 414) and 4.7× one job alone. 10/GPU adds nothing (540)
  and needs 105 GB of 125.
- **MPS works in Docker on these discrete GPUs** (unlike the GB10) and gives +5–7 % from 3 jobs/GPU on.
- **Neither GPU nor CPU saturates alone:** GPUs at most ~70 % busy, VRAM ≤ 8 GB of 24. Gains
  flatten once busy threads exceed the 24 physical cores (SMT adds little).
- **Threads:** ~cores/jobs or more is fine (oversubscribing 2× costs nothing); fewer hurts
  (4/GPU with 3 threads −15 %, 6/GPU with 2 threads −7 %).
- **Noise:** identical runs 441/442 and 476/498 (±2–4 %).
- **Memory:** ~4.5 GB host RAM per job + ~12 GB base.
- `--write-particlize both` and a `--pthat-bins` campaign: measured below.
- **Update 2026-10-10:** with MUSIC4GPU `c7b75ba` and `MUSIC_CUDA_SYNC=block`, 8/GPU gives
  553 events/h at −30 % CPU per event, and 10/GPU now pays off (588) but only fits in memory
  without particlize. See [below](#cuda-blocking-sync).
- **Update 2026-10-10 (host memory per job):** with the jet leg read frame by frame,
  `LBT_TABLE_CACHE` and `run_jobs.sh --stagger 12`, **10/GPU with particlize fits** (104.5 GB
  peak in the campaign setting) and gives **+8.4 % events/h** over 8/GPU. See
  [below](#memory-10-per-gpu).
- **Update 2026-10-13 (`cu124-final_optim`: C4 stall break, D1 late Δτ, A12):** the Au+Au
  campaign at 10/GPU runs at **2,160 events/h (+70 % over `cu124-freezeoutcopies`)**. O+O with
  hadron files only runs at **13,728 events/h (+33 %)**. Both are now host-bound. Physics check of
  C4 and D1: [below](#final-optim).

**Recommended launch: [`launch_2gpu.sh`](launch_2gpu.sh)** (usage in [README_launch.md](README_launch.md)).
It starts one container per GPU, each with `run_jobs.sh -j 8 --mps` and `OMP_NUM_THREADS=4`,
splits NJOBS between the GPUs and writes to `OUTBASE/gpu0` and `OUTBASE/gpu1`. OUTBASE can be
under `~/prod_test` (`/work/...`) or on another disk, e.g. `/data/...` (1.6 TB free; tested).
The campaign default for this machine is the `--pthat-bins` setup [below](#suggested-campaign-default);
with the 2026-10-10 image at 10 jobs/GPU, see [below](#memory-10-per-gpu).

## launch_2gpu.sh with `--write-particlize both` (2026-10-04)

`./launch_2gpu.sh 16 3 1 /work/bench/launch_particlize --write-particlize both` (8 jobs/GPU,
MPS, 4 threads; seeds 1-16, as the 8/GPU rows above): all 16 jobs complete, 3 pair + 3
particlize events each. **Host memory max 97.6 GB** of 125 (4 GB idle; ~5.8 GB per job), no swap
growth, VRAM 6.5 GB/GPU. 128.1 s/event, **450 events/h** (vs 518-536 without particlize: the
particlize writing costs ~15 %), GPUs 58 % busy, CPU 35 of 48 threads.

<a id="suggested-campaign-default"></a>
## Suggested campaign default: `--pthat-bins 10-20,20-30,30-40 --jets-per-bin 5 --parton-ymax 0.6 --write-particlize both` (2026-10-04)

`./test_launch.sh bins_j8 16 15 <these args>`, i.e. `launch_2gpu.sh 16 15 1 ...`: 8 jobs/GPU, MPS,
4 threads, seeds 1-16. 3 windows × 5 jets = `--reuse 15`, so EVENTS must be a multiple of 15; here
each job is one background with 15 jet events (5 per window). All 16 jobs complete, 15 pair + 15
particlize events each.

| | particlize, no bins (above) | **bins + reuse 15 + ymax + particlize** |
|---|---|---|
| events per job | 3 | 15 (1 background) |
| s/event (mean of jobs) | 128.1 | **71.3** (first event of a job, with MUSIC_1: ~130 s; later ones 55-75 s) |
| events/h (per-event times) | 450 | **810** (1.8×) |
| events/h (makespan, incl. startup and tail) | 406 | 722 (makespan 1197 s) |
| GPU busy / idle samples | 58 % / 22 % | 61 % / 10-13 % |
| CPU busy (threads of 48) | 35 | 35 |
| **host memory max** | 97.6 GB | **97.2 GB** (no change; swap +54 MB) |
| peak per job (MUSIC log) | | ~7.7 GB |
| VRAM per GPU | 6.5 GB | 6.5 GB |
| disk per event (pair + particlize) | 292 + 167 MB | **162 + 92 MB** (shared `arr_bg`) |

- **Speed:** the background (MUSIC_1) runs once per 15 events instead of every event, so a jet
  event costs ~60 s instead of ~128 s at 16 jobs: **~1.8× the events per hour**.
- **Memory: unchanged.** Reusing the background does not raise the peak (97 GB with 16 jobs, 28 GB
  headroom). 10 jobs/GPU would not fit with particlize (~6 GB/job → ~120 GB). *2026-10-10: it
  does now, with the host-memory changes; see [below](#memory-10-per-gpu).*
- **`--parton-ymax 0.6`** costs nothing visible: rejected events are regenerated in Pythia
  before the shower and hydro (acceptance 0.56 / 0.71 / 1.0 for the three windows, seed 1).
- **Disk:** 61 GB for these 240 events; plan ~0.25 GB per event.
- **Note for reading logs:** with output redirected, the driver's Python lines (`done in`) are
  block-buffered and appear in batches, later than MUSIC's lines; the jobs are not stalled.
- Longer jobs (e.g. 30 or 45 events, 2-3 backgrounds) amortise the ~10 s startup further; the
  makespan rate then approaches the per-event rate (810).

**As a campaign (the default for this machine):**

```bash
nohup ./launch_2gpu.sh 64 45 0 /data/camp_pth3 --campaign pth3 \
    --pthat-bins 10-20,20-30,30-40 --jets-per-bin 5 --parton-ymax 0.6 \
    --write-particlize both > /data/camp_pth3.log 2>&1 &
```

64 jobs × 45 events = 2,880 jet events (192 backgrounds): 4 rounds of 16 jobs, each round
~45 × 65 s ≈ 50 min, so **~3.5 h** and **~730 GB** on `/data`.

## Uploading to the OSDF while producing (2026-10-05)

`remote_transfer/js_osdf.py` (copied from the image) to `osdf:///fno4hic`, from this machine:
upload ~80 MB/s (1.9 GB in 23 s with 4 files at once, 2.0 GB in 25 s with 8: the line or the
origin limits, not the tool), download ~69 MB/s, byte-identical. The campaign default produces
~57 MB/s (810 events/h × 0.254 GB), so uploading keeps up with ~40 % margin; `--particlize-only`
would produce ~21 MB/s. `launch_2gpu.sh --upload REMOTE [--delete none|pair|all] [--keep-free SIZE]`
uploads finished jobs while the campaign runs (`upload_follow.py`); tested with small real
productions: uploads during and after the run, resume, `--delete pair`/`all`, `--keep-free`,
download spot checks, a failed upload (bad token: nothing deleted, exit 1), a failed
verification (not deleted, uploaded again), and stopping the campaign. See README_launch.md.

<a id="cuda-blocking-sync"></a>
## MUSIC4GPU `cuda_blocking_sync`: string deposition and `MUSIC_CUDA_SYNC` (2026-10-09/10)

The two changes described in [BENCHMARK_GB10.md](BENCHMARK_GB10.md) ("CPU per event and the
CUDA sync"): less work per string–cell pair in the string deposition, and the opt-in
`MUSIC_CUDA_SYNC=block|yield|spin|auto`. On the GB10 (GPU-bound) `block` cut CPU per event by
29 % at the same events/h. This machine looked CPU-bound, so the question was whether
throughput rises here.

**Images:**

| | image | X-SCAPE | MUSIC4GPU | js-contrib |
|---|---|---|---|---|
| baseline | `xscape-prod:cu124` of 2026-10-03, `b5a36a08c212` | `cbc7263` | `49439c0` | `384a4a2` |
| new | `xscape-prod:cu124-cuda-sync-test`, `7d15622f5656` (CI run 38020769188) | `57e32ce` | **`c7b75ba`** | `4e22dc6` |

Both: CUDA 12.4.1, gcc 14.4.0, architectures 70–90 (`/opt/X-SCAPE/BUILD_INFO.txt`).

**Setup:** as the 8/GPU rows above. `launch_2gpu.sh` with an image option and `-e
MUSIC_CUDA_SYNC` added (passed only if set, so "unset" really is unset in the container).
16 jobs × 3 events, seeds 1–16, one container per GPU, `run_jobs.sh -j 8 --mps`,
`OMP_NUM_THREADS=4`, no `--write-particlize`, 30 s idle between runs.
- **events/h:** Σ over jobs of 3600 / (job's mean s/event), as above. The makespan rate is
  events × 3600 / makespan.
- **CPU per event:** each container's cgroup `cpu.stat` `usage_usec`, polled every 1 s,
  summed over both containers, divided by events. Startup is included.
- **Host:** `mpstat 2`, `nvidia-smi dmon -s um -d 1`, `free -m` every 2 s.
- **Mode check:** every new-image job log states `[MUSIC-GPU] host sync: auto` (unset) or
  `… block`; all did (16/16, D1 20/20). The baseline prints no such line.

| run | image, `MUSIC_CUDA_SYNC` | events/h | makespan rate | **core-s/event** | s/event [min–max] | GPU0 / GPU1 busy | CPU busy (threads of 48) | host mem max |
|---|---|---|---|---|---|---|---|---|
| A1 | baseline, unset | 535 | 461 | 250.7 | 108.1 [93.8–117.6] | 66 / 65 % | 32.5 | 85.9 GB |
| B1 | new, unset (auto) | 538 | 474 | 220.1 | 107.3 [100.1–114.5] | 67 / 67 % | 29.3 | 102.5 GB |
| C1 | new, `block` | 560 | 482 | 179.0 | 103.4 [83.7–112.4] | 70 / 70 % | 24.4 | 97.6 GB |
| A2 | baseline, unset | 533 | 462 | 252.2 | 108.2 [100.8–117.2] | 66 / 67 % | 32.7 | 86.0 GB |
| C2 | new, `block` | 546 | 473 | 172.6 | 105.9 [93.8–114.5] | 70 / 67 % | 23.2 | 99.5 GB |
| D1 | new, `block`, **10/GPU, 3 threads** (seeds 1–20) | **588** | 497 | 165.0 | 123.3 [101.7–137.6] | 73 / 71 % | 23.3 | **116.1 GB** (9.6 GB left) |

All jobs completed with 3 events; VRAM 6.5 GB/GPU (D1 8.1 GB); no swap growth.

| over A (mean of A1, A2: 534 events/h, 251 core-s/event) | events/h | makespan rate | core-s/event |
|---|---|---|---|
| B: string deposition | +0.7 % | +2.8 % | −12.5 % |
| C: strings + `block` (mean of C1, C2) | **+3.5 %** | +3.4 % | **−30 %** |
| D: + 10 jobs/GPU, 3 threads (one run) | **+10 %** (+6 % over C) | +7.7 % | −34 % |

Repeat spread: A1/A2 0.3 % (events/h) and 0.6 % (core-s/event); C1/C2 2.5 % and 3.6 %.

- **CPU per event falls as on the GB10** (−30 % here, −29 % there). The string change alone
  gives −12.5 %.
- **Throughput rises only a little at 8/GPU:** +3.5 %, above noise (both C runs beat both A
  runs) but small. This machine was less CPU-bound than assumed: the A runs kept only ~33 of
  48 threads busy. With `block` it is ~24, about the physical core count, and the GPUs are
  busy ~70 %.
- **The freed cores pay off with more jobs:** 10/GPU gives 588 events/h, where the baseline
  gave 540 at 10/GPU, no better than 8/GPU. The limit is now host memory (below).
- **Threads per job (not measured with `block`):** a job keeps ~1.7 cores busy on average
  (176 core-s over ~105 s), so it mostly waits on the GPU or runs serial code. With the
  baseline, 3, 4 and 6 threads at 8/GPU were equally good (536, ~528, 518). More threads are
  not expected to help.

**Correctness** (seed 1, `gpu0/AuAu_0_10_jet_seed0001.h5`, every dataset and attribute):
- **`auto` against `block` (B1, C1): identical.** The only difference is `source/flux`: 9 of
  69 values, max |Δ| 1.0e-15, ≤ 9 ulp, the droplet-flux diagnostic that differs between any
  two runs. Of the attributes only `prod_host` differs.
- **Baseline against new image (A1, B1):** the new js-contrib (`4e22dc6`) **changes the default
  output grid** from 65 × 65 × 33 (x, y −10…10 fm, dx 0.3125, η −5…5) to 64 × 64 × 32 (x, y
  ±12.20625 fm, dx 0.3875, η ±4.84375), so `arr` and `arr_bg` cannot be compared directly.
  The MUSIC grid is unchanged. Everything else that depends on the hydro is identical:
  `ntau_freezeout(_bg)`, `tau_freezeout(_bg)`, `shower/*`, `source/droplets`. Only in the new
  image: `diag/{bg,jet}_edge_e_max`, `diag/{bg,jet}_edge_e_max_eta`, `diag/{bg,jet}_hit_edge`
  and the attribute `edge_e_threshold`.
- **New image on the old grid** (one extra job alone, `--grid` with the baseline YAML from
  A1's `prod_grid_yaml`): **`arr` and `arr_bg` are bit-identical to A1**, per event up to
  `ntau_freezeout` and as whole datasets. So the string change leaves the hydro output
  bit-identical on this machine too.

**No FP64 on the GPU:** `libmusic.so`'s SASS for sm_86 (all 10 kernels) has no FP64
instructions (no DADD/DMUL/DFMA, no F2F.F64), against ~4,100 FP32 add/mul/FMA. The same holds
for sm_70/80/89/90. The kernels use `float` with `--use_fast_math`. The `double`s in the GPU
sources are host code: the reduction result and the OpenMP converters between the CPU's
double cells and the GPU's float arrays (`GPUGrid_cuda.cu`). Precision is not a lever on
this card.

### Host memory

**At 16 jobs the new image peaks higher:** 97.6–102.5 GB (B1, C1, C2) against 86 GB (A1, A2).
The per-job peaks are the same (6.4–7.6 GB, job by job). The difference is a short transient
at the end of the first event (t ≈ 135–160 s), when more jobs hit their peak at once,
probably because the faster string fill keeps them in step. Later events overlap less:

| run | peak, end of first event | highest peak later |
|---|---|---|
| A1 / A2 | 85.9 / 86.0 GB | 79.1 / 81.4 GB |
| B1 | 102.5 GB | 88.5 GB |
| C1 / C2 | 97.6 / 99.5 GB | 91.9 / 84.7 GB |
| D1 (20 jobs) | 116.1 GB | 106.4 GB |

In B1 memory climbs from 80 to 103 GB between t = 111 and 143 s and is back at ~65 GB by
t = 160 s. **Starting the jobs a few seconds apart** (about one event time / jobs per GPU,
~12 s) would remove this first spike at a cost of a minute or two per campaign. It would not
remove the later peaks. `run_jobs.sh` has no such option yet.

**Where a job's memory goes** (one job alone on GPU 0, new image, `block`, 4 threads, seed 1,
3 events; VmRSS from `/proc` every 0.5 s, aligned with the job log). The JetScape
`[Info] NNNMB` figure is `ru_maxrss`, the running peak, not the current RSS.

| RSS (GB), max per phase | ev 1 | ev 2 | ev 3 |
|---|---|---|---|
| at event start | 0.9 | 3.2 | 4.4 |
| MUSIC_1 (background) running | 3.3 | 5.4 | 5.4 |
| slim copy to the framework (native store and copy at once) | 4.4 | 5.6 | 5.5 |
| MUSIC_2 (jet leg) running | 5.2 | 5.9 | 5.4 |
| **after MUSIC_2: `PairH5Writer`** | **6.2** | **7.4** | **6.4** |
| the same with `--write-particlize both` | 6.7 | **8.1** | 7.0 |

The peak is always in the writer window of the event with the most jet frames (event 2,
133 frames). All of it is heap (RssFile ≤ 0.18 GB). At the 7.4 GB peak:
- **The background history, 1.6 GB:** `MusicWrapper` keeps one slim copy (float32 e, s, T,
  vx, vy, vz; 67.2 M cells) and frees MUSIC's native store right after copying. Jet energy
  loss reads only this copy (`GetHydroInfo_JETSCAPE` → `bulk_info.get_tz`), and with
  `--reuse`/`--pthat-bins` it serves the whole block. It must stay, and it is already the
  only persistent copy, in its smallest form.
- **MUSIC_2's native store, ~2.3 GB:** kept (`dump_hydro_only`) so the writer can read the
  jet leg.
- **The jet leg as one numpy array, 1.0–1.3 GB:** `event_array(hydro, "native")` →
  `get_native_evolution_numpy()` (`bind_music.cc`) allocates (ntau, 100, 100, 60, 4) float32
  at once, which is then resampled to the output grid and dropped. The background is already
  read frame by frame (`framework_frames`: "~2.4 GB less held at once"). The jet leg is not,
  because the native-store binding has no per-frame accessor.
- **~1 GB base** (Python, libraries, LBT tables ~0.6 GB per job), **plus 1–2 GB left from
  earlier events:** RSS at event start grows 0.9 → 3.2 → 4.4 GB. Part of it is the slim buffer
  keeping its capacity (by design). The rest is probably freed heap that glibc does not
  return (not yet tested).
- **Particlize adds 0.5–0.7 GB** in the same window: the freeze-out surfaces of both legs,
  1.0–1.3 M cells each.

Particlize also costs time: alone 63.2 s/event against 49.4 s without (+28 %); at 16 jobs
the earlier runs measured ~15 %.

### Conclusions and next steps

- **Make `MUSIC_CUDA_SYNC=block` the default.** Output identical, −30 % CPU per event,
  +3.5 % events/h, nothing worse in any run, and it is what lets more jobs per GPU pay.
- **Keep 8 jobs/GPU for particlize campaigns.** 10/GPU gives +10 % without particlize, but
  peaks at 116 GB of 125; with particlize (+0.5–0.7 GB per job at the peak) it would most
  likely not fit. *Superseded 2026-10-10: with items 1, 3 and 4 below, 10/GPU with particlize
  fits; see [below](#memory-10-per-gpu).*
- **Re-check the campaign default's memory** with the new image (particlize at 8/GPU: 97 GB
  with the baseline; the first-event spike could push it toward ~110 GB). *Done 2026-10-10:
  111.2 GB at 8/GPU with this image (R1 [below](#memory-10-per-gpu)).*
- **The throughput lever on this machine is now host memory per job**, then GPU work per
  event; the CPU comes last. In order of value for effort:
  1. **Read the jet leg frame by frame** (a per-frame accessor in `bind_music.cc`, a
     `NativeFrames` reader like `FrameworkFrames`): −1.0–1.3 GB per job at the peak
     (13–17 %), particlize peak ~8.1 → ~6.9 GB. *Done 2026-10-10.*
  2. **Memory left between events:** test `MALLOC_ARENA_MAX=2` / `MALLOC_TRIM_THRESHOLD_`
     (no code change); maybe another ~1 GB per job.
  3. **Share the LBT tables between jobs** (read-only `mmap`): ~0.6 GB per job.
     *Done 2026-10-10 (`LBT_TABLE_CACHE`).*
  4. **A staggered start** in `run_jobs.sh`: removes the first-event spike. *Done 2026-10-10
     (`--stagger S`).*
  5. **More RAM** (the 3960X takes 256 GB): the same effect without code.
  6. **GPU side:** CUDA graphs and fewer host syncs for the ~1500-step loop; start MUSIC_2
     from MUSIC_1's state before the first droplet (its first 6 frames equal the
     background's).
  With 1 alone, 20 jobs with particlize would peak at an estimated ~105–115 GB (likely to
  fit with a stagger, little margin); with 2 as well, comfortably. Per-phase wall-time
  timing of one job (alone and at 16 jobs) would rank 6.
- **Caveat:** the moving tag `xscape-prod:cu124` now points to the new build, whose default
  output grid (64 × 64 × 32) does not load together with files made on the old grid.
  `launch_2gpu.sh` uses `:cu124` by default; pass `--grid` with the old YAML, or pin the
  image, to stay on the old grid.

Raw data on this machine: `~/prod_test/bench/synctest_20261009/` (`REPORT.md`, per-run
monitor logs and job logs, `compare_*.txt`, `memtrace/`), launcher copy
`~/prod_test/launch_2gpu_synctest.sh`.

<a id="memory-10-per-gpu"></a>
## Host memory per job: 10 jobs/GPU with `--write-particlize both` (2026-10-10)

The question: do the host-memory changes from [BENCHMARK_GB10.md](BENCHMARK_GB10.md) ("Host memory per job (2026-10-10)") make 10 jobs/GPU with particlize feasible and worthwhile here? The changes:
- the jet leg read one frame at a time (automatic);
- `LBT_TABLE_CACHE=<file>` (opt-in);
- `run_jobs.sh --stagger S`;
- `OPENBLAS_NUM_THREADS=1` as the drivers' default.

**Images:**

| | image | X-SCAPE | MUSIC4GPU | js-contrib |
|---|---|---|---|---|
| reference | `xscape-prod:cu124-cuda-sync-test`, `7d15622f5656` (as above) | `57e32ce` | `c7b75ba` | `4e22dc6` |
| new | `xscape-prod:cu124-cuda-sync-mem-test`, `aa7568b46742` (CI run 38066958166) | `01584cc` | `c7b75ba` | **`593fcc4`** |

**Setup**
- `launch_2gpu.sh` copy with `--image`, `--stagger S` (passed to `run_jobs.sh` after `--mps`), and `-e MUSIC_CUDA_SYNC -e LBT_TABLE_CACHE`.
- Every run: `--write-particlize both`, `MUSIC_CUDA_SYNC=block`, MPS, seeds from 1, 30 s idle between runs.
- **LBT cache:** host `~/prod_test/lbt_cache/lbt_tables_v1.bin` = `/work/lbt_cache/…` in both containers.
  - Created once before the runs by one job of one event (log: `LBT: wrote the table cache …`; 656,581,952 bytes).
  - Every job of R2–R6 logged `LBT: tables mapped read-only from /work/lbt_cache/lbt_tables_v1.bin`. None wrote it, none warned.
- **Host memory:** used = MemTotal − MemAvailable, from `/proc/meminfo` every 2 s, plus each job's VmRSS/VmHWM.
- **events/h:**
  - per-event convention, as above;
  - **makespan rate** = events × 3600 / makespan. Use this one for staggered runs, whose per-event times look shorter than they are.
- **CPU:** cgroup `cpu.stat` as above.
- **Watchdog:** `docker stop` of the run's containers if MemAvailable < 6 GB. It never fired.
- **OOM:** `docker events` and the kernel's `oom_kill` counter. None in any run.
- **Idle baseline:** 5.7–6.0 GB used before every run.

| run | image | jobs/GPU (total) | OMP | stagger | events/job | **host peak** (min free) | swap | events/h per-event | **events/h makespan** | core-s/event | GPU0 / GPU1 busy | VRAM/GPU | per-job VmHWM, mean [max] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R1 | reference | 8 (16) | 4 | – | 3 | **111.2 GB** (14.5) | +0 | 456 | 404 | 243.4 | 59 / 58 % | 6.5 GB | 7.52 [8.17] GB |
| R2 | new | 8 (16) | 4 | – | 3 | **91.7 GB** (34.0) | +0 | 478 | 432 | 203.0 | 63 / 62 % | 6.5 GB | 5.98 [6.57] GB |
| R3 | new | 10 (20) | 3 | 12 s | 3 | **97.3 GB** (28.3) | +0 | (598) | **458** | 188.1 | 68 / 65 % | 8.1 GB | 5.99 [6.57] GB |
| R4 | new | 10 (20) | 3 | – | 3 | **108.8 GB** (16.9) | +0 | 511 | 468 | 207.9 | 69 / 69 % | 8.0 GB | 6.00 [6.56] GB |
| R5 | new | 10 (20) | 3 | 12 s | 15, campaign¹ | **104.5 GB** (21.1) | +254 MB² | (951) | **805** | 114.8 | 67 / 68 % | 8.1 GB | 6.10 [6.71] GB |
| R6 | new | 8 (16) | 4 | 12 s | 15, campaign¹ | **91.7 GB** (33.9) | +525 MB² | (878) | **743** | 111.6 | 63 / 62 % | 6.5 GB | 6.10 [6.46] GB |

¹ `--pthat-bins 10-20,20-30,30-40 --jets-per-bin 5 --parton-ymax 0.6`: one background per job.

² Not the jobs; see "Swap" below.

All jobs completed: `events_written` and `particlize_events_written` equal the events requested, and every log says `host sync: block`. Per-event rates in parentheses are staggered runs: R3 and R4 have the same makespan rate (458 vs 468) but per-event rates of 598 vs 511.

**When the peaks happen.** The job logs are block-buffered, so event end times were reconstructed backwards from each log's last write and the `done in` times. This agrees with the pair `.h5` being created 6–12 s before event 1's reconstructed end, in the writer window.

| run | jobs on event … at the host peak | peak until all jobs finished event 1 | highest peak after |
|---|---|---|---|
| R1 | 16 × event 1 | 111.2 GB | 104.1 GB |
| R2 | 14 × event 1, 2 × event 2 | 91.7 GB | 86.0 GB |
| R3 (stagger) | 8 × event 1, 12 × event 2 | 97.3 GB | 96.8 GB |
| R4 | 17 × event 1, 3 × event 2 | 108.8 GB | 102.5 GB |
| R5 (stagger, 15 ev) | events 4–7 | 97.9 GB | **104.5 GB** |
| R6 (stagger, 15 ev) | events 9–14 | 81.7 GB | **91.7 GB** |

**Memory saved per job, R1 → R2 (same 16 jobs and seeds):** −19.5 GB at the host peak, **−1.22 GB per job**.
- Per job VmHWM: 7.52 → 5.98 GB; the largest job 8.17 → 6.57 GB.
- 16 jobs with particlize now leave 34 GB free instead of 14.5 GB.
- Also **+5.0 % events/h (+6.8 % makespan) at −16.6 % CPU per event**: one OpenBLAS thread, and no LBT parse.

**The reference image at 8/GPU already peaks at 111.2 GB with particlize** (R1), not the 97.6 GB [above](#suggested-campaign-default). That number is from the 2026-10-03 baseline image; it is the first-event spike [above](#host-memory).

**The stagger, R3 vs R4:** −11.5 GB at the peak (108.8 → 97.3 GB). It removes the first-event spike.
- Makespan rate: 458 vs 468 (−2 %, within noise). With 3-event jobs the 9 × 12 s ramp is a large share; in a campaign it is paid once.
- CPU per event −10 % (188 vs 208 core-s).

**Steady state:** with the stagger the highest peak comes later, when jobs drift into step by chance: R5 104.5 GB (~4.9 GB per job above idle), R6 91.7 GB.

**Swap:** system swap grew in R5 (+254 MB) and R6 (+525 MB). The jobs didn't cause it:
- R6 had at least 33.9 GB available throughout.
- The jobs' own VmSwap, sampled every 5 s in R6, stayed at 0.
- The swapped-out pages belong to idle processes (VS Code servers, gnome-shell, upowerd).
- The kernel (`vm.swappiness` 60) moves them out while ~30–40 GB of freshly written `.h5` data sits in page cache.

**Correctness**
- **R1 vs R2, seed 1:**
  - pair file: 35 datasets identical; only `source/flux` differs (max |Δ| 4.4e-16, 13 of 69 values, ≤ 4 ulp);
  - particlize file: all 33 datasets identical;
  - attributes: only `prod_host` differs.
  - The files record no js-contrib commit (only `format_version`, `prod_build`).
- **R5:** all 20 jobs wrote 15 pair + 15 particlize events, one background each (`diag/bg_id`, `events/bg_id` and `surface/bg/bg_id` all 0; all 300 `done in` lines say `bg 0`).

**Verdict**
- **Feasible:** all jobs complete, no OOM, peak ≤ 110 GB (≥ 15 GB headroom).
  - R3 left 28.3 GB free; the campaign run R5 21.1 GB.
  - Without the stagger it is marginal: R4 left 16.9 GB.
  - R5 covered 22 minutes. A campaign of hours samples more chance alignments, so its peak may creep somewhat higher.
- **Worthwhile:**
  - campaign setting, R5 vs R6: **+8.4 % events/h** (805 vs 743 makespan, 951 vs 878 per event), with +2.9 % CPU per event and GPUs 67–68 % busy instead of 62–63 %;
  - 3-event runs: +6.2 % (R3) and +8.4 % (R4) over R2 by makespan;
  - all well outside the ±3 % noise. For comparison, the 2026-10-04 campaign default measured 810 / 722 events/h.
- **Campaign default for this machine (proposed):**
  - **10 jobs/GPU, `OMP_THREADS=3`, `LBT_TABLE_CACHE` on a bind-mounted host path, `--stagger 12`;**
  - create the cache once per LBT-tables version before the campaign;
  - watch host memory during the first long campaign, and use 9/GPU (not measured) if it peaks above ~112 GB.
  - `utils/launch_2gpu.sh` has both as options since 2026-10-10 (`--lbt-cache FILE` makes the file if it is missing; `--stagger S`), and passes `MUSIC_CUDA_SYNC` if set:

    ```bash
    IMAGE=jhputschke/xscape-prod:cu124-cuda-sync-mem-test JOBS_PER_GPU=10 OMP_THREADS=3 \
    MUSIC_CUDA_SYNC=block nohup ./launch_2gpu.sh --lbt-cache /work/lbt_cache/lbt_tables_v1.bin \
        --stagger 12 60 45 0 /data/camp_pth3 --campaign pth3 \
        --pthat-bins 10-20,20-30,30-40 --jets-per-bin 5 --parton-ymax 0.6 \
        --write-particlize both > /data/camp_pth3.log 2>&1 &
    ```

    60 jobs × 45 events = 2,700 events in 3 rounds of 20 jobs: ~3 h, ~0.6 TB ([README_launch.md](README_launch.md#10-jobs-per-gpu)).

Raw data on this machine: `~/prod_test/bench/memtest_20261010/` (`REPORT.md`, `summary.jsonl`, per-run monitor and job logs, `correctness.txt`); launcher copy `~/prod_test/launch_2gpu_memtest.sh`.

<a id="music2restart"></a>
## The CPU fixes A1–A9 and the MUSIC_2 restart: `cu124-music2restart` (2026-10-12)

The question: how many more events/h do the bit-identical CPU fixes give on this machine at the campaign setting? They are A1–A9 of [REPORT_CPU_GPU.md](REPORT_CPU_GPU.md) §5. A8 is MUSIC_2 starting from MUSIC_1's state at the last freeze-out step before the liquefier's τ_delay. The comparison is with the 10-jobs-per-GPU numbers of 2026-10-11: 796 events/h, and 917 with `--particlize-only`.

**Images:**

| | image | X-SCAPE | MUSIC4GPU | js-contrib |
|---|---|---|---|---|
| old | `xscape-prod:cu124` = `cu124-20261011-84df104`, `066094dfcb4f` | `84df104` | `bc8543a` | `8765281` |
| new | `xscape-prod:cu124-music2restart`, `d4acf127186f` (built 2026-10-12 04:09 UTC) | `f72a42b` | `9550fc9` | `02bed42`: 9 commits that `main` (`a9dd328`) doesn't have yet |

**Setup**
- **Same settings as `launch_2gpu.sh`:** one container per GPU, `run_jobs.sh -j 10 --mps --stagger 12`, `OMP_THREADS=3`, `OMP_WAIT_POLICY=passive`, `MUSIC_CUDA_SYNC=block`, and `LBT_TABLE_CACHE=/work/lbt_cache/lbt_tables_v1.bin`.
  - The runs were started by `run_auau.sh`: the same `docker run` and the same flags as the launcher, plus the monitors of the 2026-10-11 runs.
- **Two-stage runs:** 20 jobs × 15 events (seeds 1–20) at the campaign setting: `--pthat-bins 10-20,20-30,30-40 --jets-per-bin 5 --parton-ymax 0.6 --write-particlize both`, so one background per job.
- **Hydro-only run:** `prod_AuAu_0_10`, 12 jobs/GPU, `OMP_THREADS=2`, 24 jobs × 6 events.
- **Baselines:** `jetc_j10_2gpu`, `jetq_j10_2gpu` and `norm_j12_2gpu` of 2026-10-11 (`bench/auau_prof_20261011/`), with the same seeds and the same settings.
- **events/h by makespan** (events × 3600 / makespan), as for every staggered run; the per-event rate is in parentheses. CPU from cgroup `cpu.stat`, GPU busy from `nvidia-smi dmon`. GPU-s/event = mean SM busy × 2 GPUs × 3600 / events/h.
- **Restart switched on:** every two-stage job logged `restart  MUSIC_2 starts from MUSIC_1's state at tau <= 1 fm/c`, and one `Checkpoint at step 28, tau = 0.96 fm/c: GPU state 96 MB, 76994 surface cells, 6 frames (109 MB)`.
  - All 600 jet legs of the two restart runs (and all 30 of the single jobs) logged `started from the checkpoint at step 28`. None fell back to τ0.
  - `--no-music2-restart` after OUTDIR (run `jetrn`) logged `restart  OFF: --no-music2-restart` in every job.

**Throughput, 2 GPUs:**

| production | image | events/h makespan (per event) | change | s/event in a job | core-s/event | threads busy (of 48) | GPU0 / GPU1 busy | GPU-s/event | host peak | per-job VmHWM, mean [max] | VRAM/GPU |
|---|---|---|---|---|---|---|---|---|---|---|---|
| two-stage campaign, 10/GPU | old | 796 (927) | | 77.8 | 116.6 | 26.6 | 66 / 67 % | 6.00 | 103.8 GB | 6.10 [6.70] GB | 8.1 GB |
| | new, `--no-music2-restart` (A1–A7) | 965 (1,123) | +21 % | 64.2 | 100.5 | 27.6 | 72 / 72 % | 5.40 | 96.0 GB | 5.91 [6.49] GB | 8.1 GB |
| | **new** (A1–A8) | **1,121 (1,299)** | **+41 %** | 55.5 | **62.3** | 20.1 | 83 / 82 % | 5.29 | 100.6 GB | 5.84 [6.44] GB | 9.1 GB |
| two-stage campaign, `--particlize-only`, 10/GPU | old | 917 (1,071) | | 67.4 | 103.7 | 27.1 | 76 / 75 % | 5.94 | 95.9 GB | 5.81 [6.39] GB | 8.1 GB |
| | **new** (A1–A9) | **1,237 (1,442)** | **+35 %** (+55 % over 796) | 50.0 | **50.1** | 17.8 | 89 / 88 % | 5.16 | **74.0 GB** | 4.21 [4.57] GB | 8.9 GB |
| hydro only, 12/GPU | old | 1,223 (1,325) | | 65.4 | 60.3 | 21.2 | 85 / 83 % | 4.94 | 66.5 GB | 4.01 [4.20] GB | 6.5 GB |
| | **new** (A2–A7) | **1,601 (1,743)** | **+31 %** | 49.7 | **40.4** | 18.3 | 90 / 86 % | 3.95 | 57.1 GB | 4.01 [4.18] GB | 6.5 GB |

All jobs completed: 300, 300, 300 and 144 events, none skipped. Every log says `host sync: block`. There was no OOM (`docker events`, kernel `oom_kill`), and the watchdog never fired.

**One job alone on GPU0** (`prof_auau.sh … clean 3`: `MUSIC_PROFILE=1`, OMP 3, seed 1). The old two-stage and hydro jobs were run again today with the old image; the old particlize-only job is the 2026-10-11 one.

| per event | two-stage campaign, 15 events: old → new | particlize file only, 15 events: old → new | hydro only, 4 events: old → new |
|---|---|---|---|
| **wall s** | 45.1 → **22.9** (−49 %) | 38.0 → **18.2** (−52 %) | 25.3 → **16.7** (−34 %) |
| event with the background / jet events | 65.5 / 43.7 → 48.0 / 21.1 | 59.1 / 36.5 → 42.4 / 16.5 | |
| **CPU core-s** (main thread) | 73.8 (39.3) → **33.9** (17.8) | 66.4 (32.1) → **29.4** (13.2) | 39.6 (20.4) → **29.4** (13.0) |
| GPU busy s | 6.07 → 5.19 | 5.87 → 5.04 | 4.63 → 3.70 |

The report's estimate for A1–A8 was −40 core-s per two-stage campaign event (−53 %); measured: −39.9 (−54 %). For hydro only it estimated −12 core-s (A1–A7); measured: −10.2.

**What limits the rate now**
- **Two-stage:** 5.3 GPU-s per event against 6.0 before. 28 of ~500 jet-leg steps are no longer recomputed; the remaining drop comes from A1–A7 (6.00 → 5.40 without the restart).
  - The ceiling at 90 % GPU busy is now ~1,225 events/h (pair + particlize) and ~1,255 (particlize only).
  - The runs reach 91 % and 98 % of that: the two-stage campaign is **nearly GPU-bound** on this machine, and the particlize-only one is.
  - Over half the threads stay idle (18–20 of 48 busy, against 27 before).
- **Hydro only:** 3.95 GPU-s per event (−20 %). The ceiling at 90 % is ~1,640 events/h, and the run reaches 98 % of it.
- **More events/h now need less GPU work per event**, not more jobs: C1 (O+O grid), C3 and C4 of REPORT_CPU_GPU.md §5.
  - 12 jobs/GPU for the two-stage campaign could add at most ~9 % (not measured).
  - Its host peak (100.6 GB at 10/GPU) leaves little room for 12/GPU with the pair file. `--particlize-only` (74.0 GB) does leave room.

**Memory and disk**
- **Particlize file only:** host peak −22 GB (95.9 → 74.0 GB) and per-job VmHWM 5.81 → 4.21 GB. This is A9: MUSIC_2 no longer stores its evolution.
- **Pair + particlize:** peak 103.8 → 100.6 GB, at t = 211 s while the staggered jobs start.
- **GPU memory:** the restart's checkpoint stays on the GPU, 96 MB per job, so VRAM rises 8.1 → 9.1 GB per GPU at 10 jobs (of 24 GB).
- **Swap:** system swap grew by ~100 MB during the two-stage runs, as in R5/R6 above.
- **Disk per 300 events:** unchanged; 66 GiB for pair + particlize, 27 GiB for the particlize file only.

**Correctness** (every dataset and attribute compared bit for bit, NaN = NaN; `compare_h5.py`)
- **Hydro only, old vs new image** (seed 1, 4 events): all 6 datasets are identical.
  - The only other difference is the recorded job XML: `<output_momentum_anisotropy_vs_etas>0` (A4).
- **Two-stage, old vs new image** (seed 1, 15 events, single jobs):
  - all 40 datasets of the particlize file are identical;
  - 45 of the 46 datasets of the pair file are identical. `source/flux` differs in 16 of 330 values, by ≤ 3 ulp (4.4e-16);
  - the job XML differs by the A4 line.
- **Restart vs `--no-music2-restart`** (same image, 300 events, all 20 seeds):
  - all 40 datasets of every particlize file are identical;
  - in the pair files everything is identical except `source/flux`: 1,072 of 6,545 values, ≤ 7 ulp (8.9e-16);
  - attributes differ only in `prod_music2_restart`, `prod_host` and the wall times.
- **`source/flux` differs run to run anyway:** the same image and seed with the restart, as a job alone vs as one of 20 jobs, differ in 29 of 330 values (≤ 4 ulp). R1 vs R2 above showed the same kind of difference (≤ 4 ulp). It is not caused by the changes.
- **Particlize file only vs pair + particlize** (new image, 20 seeds): all 35 datasets the two files share are identical.
  - The particlize-only file lacks, by design (as before), the jet-leg entries that come from the pair writer: `events/ntau_jet`, `frames_identical`, `jet_edge_e_max(_eta)`, `jet_hit_edge` and the `pair_file` attribute.
- **`tests/test_resample.py`** (A2) on this machine's x86 OpenBLAS: 4 passed. That was still open in the js-contrib plan.

**Verdict**
- **Campaign default for this machine:** the same launcher settings, with the new image:

  ```bash
  IMAGE=jhputschke/xscape-prod:cu124-music2restart JOBS_PER_GPU=10 OMP_THREADS=3 \
  MUSIC_CUDA_SYNC=block nohup ./launch_2gpu.sh --lbt-cache /work/lbt_cache/lbt_tables_v1.bin \
      --stagger 12 60 45 0 /data/camp_pth3 --campaign pth3 \
      --pthat-bins 10-20,20-30,30-40 --jets-per-bin 5 --parton-ymax 0.6 \
      --write-particlize both > /data/camp_pth3.log 2>&1 &
  ```

  - **~1,120 events/h (+41 %).** 60 × 45 = 2,700 events then take ~2.4 h instead of ~3.4 h.
  - With `--particlize-only`: **~1,240 events/h** and 22 GB less host memory.
  - The restart is on by default. `--no-music2-restart` after OUTDIR switches it off; that measured 965 events/h.
- **Not covered here:** O+O (A4–A6 apply there too), and a campaign longer than one round of 20 jobs.
  - The new image is ahead of js-contrib `main`. For a campaign, pin it by digest (`jhputschke/xscape-prod@sha256:ce98cc17a4b2115f0ad7f987b1d45c6f18f2273876141e0d3c148f97127592a4`, image ID `d4acf127186f`) or wait for a dated tag.

Raw data on this machine: `~/prod_test/bench/music2restart_20261012/`: `REPORT.md`, `summary.jsonl`, run directories with monitor and job logs, `cmp_*.txt` / `flux_*.txt` / `partonly_common.txt` (comparisons), `chain.sh` and `compare_all.sh` (the order of the runs), `restart_stats.py`, `compare_h5.py`, and `test_resample_3090.log`.

<a id="oo-grid"></a>
## O+O campaign: the standard ±15 fm MUSIC grid vs ±10.5 fm (2026-10-12)

The question: what does an O+O campaign (`prod_OO_isr`) give on this machine with the new image, and with the smaller MUSIC grid that [REPORT_CPU_GPU.md](REPORT_CPU_GPU.md) §5–6 suggests (C1)?
- **C1** is ±10.5 fm instead of ±15 fm in x and y: 70 × 70 instead of 100 × 100 transverse cells, the same 0.3 fm cells and the same η grid.
- The O+O fireball stays within |x|, |y| ≲ 7 fm, and the 64 × 64 × 32 output grid reaches ±9.84 fm.
- The report asked for a physics check before switching: a few hundred events with no grid-edge hits, and unchanged observables.

**Setup**
- **Image:** `xscape-prod:cu124-music2restart` (`d4acf127186f`, as in the section above). One ±15 fm reference run used the old `cu124` (`066094dfcb4f`).
- **Settings as `launch_2gpu_oo.sh`:** one container per GPU, `run_jobs.sh -j J --mps`, `OMP_WAIT_POLICY=passive`, `MUSIC_CUDA_SYNC=block`. One wave, seeds 1…2J, default physics (pT̂ 20–40 GeV, b < 3 fm).
  - Started by `run_oo.sh` (a copy of the 2026-10-11 O+O harness), with the same monitors as the Au+Au runs.
- **The grid:** `--user-xml` with a copy of the image's `OO_MCGlauber_MUSIC_isr.xml`.
  - `…_grid15.xml` is unchanged; `…_grid10p5.xml` differs only in `<grid_max_x>`/`<grid_max_y>` 10.5.
  - MUSIC logged `GPU grid allocated (70x70x60 cells)`, `dx = 0.3 fm`. The hydro files record `X_min_MUSIC = −10.5`, `nX_MUSIC = 70`.
  - The driver's check that the output grid lies inside MUSIC's grid passed.
- **Outputs:**
  - "hadrons only": `--bulk-hadrons 10 --no-particlize --no-hydro-file`, as R1–R4 of 2026-10-11;
  - "default": hydro file, particlize file and jet hadrons, as R5.
- **events/h by makespan**, per-event rate in parentheses. CPU from cgroup `cpu.stat`. GPU-s/event = mean SM busy × 2 × 3600 / events/h.

| run | image | grid | output | jobs/GPU, OMP | events | **events/h makespan** (per event) | s/event in a job | core-s/event | threads busy (of 48) | GPU0 / GPU1 busy | GPU-s/event | host peak | per-job VmHWM, mean [max] | VRAM/GPU |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H15_j8_old | old | ±15 | hadrons only | 8, 3 | 16 × 15 | 3,534 (3,658) | 15.8 | 21.5 | 22.0 | 61 / 61 % | 1.24 | 37.8 GB | 2.44 [2.56] GB | 4.4 GB |
| H15_j8 | new | ±15 | hadrons only | 8, 3 | 16 × 15 | 3,798 (3,940) | 14.6 | 21.7 | 23.3 | 57 / 58 % | 1.09 | 35.4 GB | 2.45 [2.55] GB | 4.4 GB |
| H10_j8 | new | **±10.5** | hadrons only | 8, 3 | 16 × 15 | **6,444** (6,756) | 8.5 | 13.3 | 23.9 | 58 / 57 % | 0.64 | 21.8 GB | 1.37 [1.42] GB | 3.4 GB |
| H15_j12 | new | ±15 | hadrons only | 12, 2 | 24 × 15 (2 skipped) | 3,730 (3,864) | 22.4 | 28.7 | 30.4 | 59 / 58 % | 1.13 | 50.8 GB | 2.44 [2.55] GB | 6.5 GB |
| H10_j12 | new | **±10.5** | hadrons only | 12, 2 | 24 × 15 (2 skipped) | **6,420** (6,664) | 13.0 | 17.1 | 31.1 | 60 / 60 % | 0.67 | 30.1 GB | 1.38 [1.42] GB | 4.9 GB |
| D15_j12 | new | ±15 | default | 12, 2 | 24 × 15 | 4,070 (4,196) | 20.6 | 24.3 | 28.0 | 62 / 62 % | 1.10 | 50.3 GB | 2.42 [2.52] GB | 6.5 GB |
| D15_j12_rep | new | ±15 | default | 12, 2 | 24 × 15 | 4,031 (4,148) | 20.8 | 24.5 | 29.3 | 61 / 62 % | 1.10 | 65.5 GB¹ | 2.42 [2.52] GB | 6.5 GB |
| D10_j12 | new | **±10.5** | default | 12, 2 | 24 × 15 | **7,490** (7,790) | 11.1 | 13.0 | 27.6 | 67 / 66 % | 0.64 | 29.8 GB | 1.35 [1.41] GB | 5.0 GB |
| **L15_j8** (campaign length) | new | ±15 | default | 8, 3 | 16 × 60 (1 skipped) | **4,006** (4,111) | 14.0 | 18.5 | 22.2 | 62 / 61 % | 1.10 | 51.3 GB | 2.50 [2.59] GB | 4.4 GB |
| **L10_j8** (campaign length) | new | **±10.5** | default | 8, 3 | 16 × 60 (5 skipped) | **7,282** (7,593) | 7.6 | 10.2 | 21.5 | 66 / 65 % | 0.65 | 24.7 GB | 1.39 [1.44] GB | 3.4 GB |

¹ The idle baseline before this run was 5.6 GB instead of ~4 GB, and its jobs peaked as in D15_j12: not the jobs.

- **All runs:** all jobs completed, no OOM, the watchdog never fired, and every log says `host sync: block`.
- **Skipped events:** 10 of 4,440, all Matter's "minus log of sudakov negative". The driver skips the event and goes on; that's the known ~1 in 400.
- **Grid-edge hits:** none. `diag/hit_grid_boundary` was 0 in all 2,994 events with a hydro file (1,315 of them on ±10.5 fm).

**Throughput**
- **±10.5 fm grid: +82 % events/h at the campaign length:** 7,282 vs 4,006 (L10/L15, launcher defaults 8/GPU, default outputs).
  - The short runs agree: +70 % hadrons only (6,444 vs 3,798) and +84 % default outputs (7,490 vs 4,070).
  - CPU per event −45 % (10.2 vs 18.5 core-s), GPU per event −41 % (0.65 vs 1.10 s), host memory per job −45 % (1.39 vs 2.50 GB).
- **New image vs old on the ±15 fm grid:** +7 % (3,798 vs 3,534, hadrons only, same day). Against 2026-10-11's R5 (default outputs, 3,657): 4,006–4,070.
- **Still host-bound, now on both grids:**
  - the GPUs are 57–67 % busy, and 12/GPU gives nothing over 8/GPU (6,420 vs 6,444; 3,730 vs 3,798);
  - the busy threads stay at ~22–24 at 8/GPU, the number of physical cores;
  - events/h ≈ 24 cores × 3600 / core-s per event (24 × 3600 / 13.3 = 6,500 vs 6,444 measured).
  - The GPU ceiling at 90 % busy is ~10,000 events/h on the ±10.5 fm grid (~5,900 on ±15). The next step for O+O is B2 (Cornelius on the GPU), as REPORT_CPU_GPU.md §6 says.
- **The 15-event runs' makespans are short:** 134–346 s, and the jobs' startup is a few seconds of that. The 60-event runs (472 and 862 s) are the campaign numbers.

**Output: ~30 MB per event on both grids.** The output grid is the same, and so is the surface (below): 27 GiB per 960 events.
- At 7,300 events/h that is **~220 GB/h**. The 658 GB free on `/` would last ~3 h.
- A campaign with default outputs therefore needs `--upload … --delete all`, or `/data`, as for R5. Hadrons only stays at ~0.15 MB per event.

**Physics check, ±10.5 vs ±15 fm** (same seeds, event by event: `per_event.py`, `odd_events.py`, `odd_detail.py`, `bulk_had.py`)
- **Control, run-to-run noise:** D15_j12 repeated as D15_j12_rep (same image, grid and seeds).
  - In all 360 events the initial state, the hydro output grid and the freeze-out surface are **bit-identical**.
  - The final partons or jet hadrons differ in **266 of 360 events (74 %)**, from event 1–9 of a job on. This is the O+O irreproducibility noted in REPORT_CPU_GPU.md; it lies after the hydro, in Matter/hadronization. The grid comparison below can't be more exact than this.
- **Default outputs, D10_j12 vs D15_j12:** 360 events, 24 jobs.
  - The initial state is identical in every event: Glauber, pT̂, ISR subtraction, string energy.
  - **In 357 events the hydro output grid agrees within 4.1e-6 of each field's maximum, and the freeze-out surface is bit-identical.** The rest is float32 interpolation from a grid with another origin (−10.5 vs −15 fm).
  - 160 of these are identical in everything, including partons and jet hadrons. In the others only the partons or jet hadrons differ, which is fewer than in the control (197 vs 266 events).
- **Campaign runs, L10_j8 vs L15_j8:** 660 events of the 11 jobs that skipped no event in either run. Jobs with a skip are left out because their rows no longer line up.
  - 654 events: hydro within 1.7e-6, surface bit-identical.
- **The other 3 + 6 events** (of 1,020):
  - **Flow velocity only, 4 events** (seed 6 ev 11, seed 7 ev 10, seed 9 ev 42, seed 11 ev 20): it differs by at most 3.4e-4 of its range, in single late frames at the dilute output edge. The surface is bit-identical.
  - **One surface cell, 1 event** (seed 11 ev 23): differs by 3e-8 fm.
  - **Near-vacuum cells, 2 events** (seed 2 ev 12, seed 9 ev 60): the velocity differs by up to 0.6–0.7, but only where e ≤ 0.009 GeV/fm³ (the event maximum is 4.8–8.3). There the velocity of an almost empty cell is ill-conditioned and amplifies the last-bit differences.
    - The energy density differs by at most 4e-4 GeV/fm³ there.
    - The surface has the same cells, with values within 1.5e-4 (relative, per column); τ_freezeout and the surface's extent are the same.
- **Observables, means over the events:** τ_freezeout, surface cells per event, final partons, jet hadrons and the energy on the first output frame agree between the grids well inside one standard error. For example, L10 vs L15:
  - τ_fo 4.357 ± 0.018 fm/c on both;
  - 1.335e5 surface cells on both;
  - jet hadrons 114.7 ± 1.3 vs 114.3 ± 1.3.
- **Bulk hadrons** (iSS in the job, hadrons-only runs, 10 oversamples):
  - bit-identical in 239 of 240 events, H10_j8 vs H15_j8;
  - for comparison, the same grid with another job count (H15_j12 vs H15_j8) gives 227 of 239;
  - 373.44 ± 2.51 hadrons per oversample on both grids.

**Verdict**
- **The ±10.5 fm grid is safe for O+O at these settings (b < 3 fm, 200 GeV), and +82 % events/h.**
  - No grid-edge hit in 2,994 events (1,315 on ±10.5 fm).
  - The hydro and the surface are the same to float32 rounding, apart from a few near-vacuum cells.
  - The jet and bulk hadrons agree as well as two runs of the same grid do.
- **For larger b or another system,** check the fireball extent again: the fireball has to stay well inside ±9.84 fm.
- **Campaign on this machine:** the launcher defaults (8/GPU, OMP 3) with the new image and the grid XML:

  ```bash
  IMAGE=jhputschke/xscape-prod:cu124-music2restart nohup ./launch_2gpu_oo.sh \
      --upload OO_c1 --delete all --min-free 100G 64 100 0 /data/OO_c1 --campaign c1 \
      --user-xml /work/bench/oo_grid_20261012/OO_MCGlauber_MUSIC_isr_grid10p5.xml > /data/OO_c1.log 2>&1 &
  ```

  ~7,300 events/h with default outputs, so 6,400 events in under an hour. The upload then has to keep up with ~220 GB/h (~61 MB/s), against the ~80 MB/s measured on 2026-10-05: just enough, with little margin. `--min-free` stops the campaign if it falls behind. Without the upload, 6,400 events are ~190 GB on `/data`; hadrons only stays at ~0.15 MB per event.
- **Made the default on 2026-10-12, after these runs:** `OO_MCGlauber_MUSIC_isr.xml` (now in `example/config/`) has `grid_max_x/y` 10.5; the ±15 fm grid needs `--user-xml` with a copy.

**Where O+O's CPU goes now** (follow-up, same day). The O+O runs are host-bound, so the question is which CPU work sets their rate.
- **Profile of one job inside a running campaign:** 8/GPU, OMP 3, default outputs, 30 events per job.
  - The containers got SYS_PTRACE, and py-spy ran as root inside the GPU0 container on the seed-1 job, from 30 s to 150 s.
  - The runs gave 7,217 events/h (±10.5 fm) and 4,017 (±15 fm), the same as without the profiler.
  - The profiled job used 9.7 and 17.9 core-s per event, matching the cgroup's 10.3 and 18.5 for the whole run.

| stage, core-s per event (share) | ±10.5 fm | ±15 fm |
|---|---|---|
| **freeze-out: Cornelius search** | 3.65 | 7.95 |
| **freeze-out: GPU → host copy of both grids** (`sync_arena_from_gpu_readonly`) | 1.63 | 4.09 |
| **freeze-out together** | **5.3 (54 %)** | **12.0 (67 %)** |
| writer: HDF5 compress + write / resample to the output grid | 0.76 / 0.64 | 1.28 / 1.04 |
| medium lookups for Matter (`GetHydroInfo`) | 0.99 | 0.59 |
| MUSIC → framework copy (`bulk_info`) | 0.68 | 0.77 |
| string deposition | 0.56 | 0.68 |
| frames to memory (pack), RK host side, per-event init, Matter, other | 0.7 | 1.4 |
| **total** | **9.7** | **17.9** |

- **In the campaign the freeze-out costs about twice what it does in a job alone.**
  - A job alone on the ±10.5 fm grid, default outputs: Cornelius 1.9 + copy 0.7 core-s per event, 6.4 core-s per event in total (`table_oo.md`).
  - These stages stream both grids through memory, and 16 jobs compete for its bandwidth.
- **What the freeze-out time is spent on** (main-thread samples of the profiled job, `fo_split2.py`):

| share of the freeze-out time | ±10.5 fm | ±15 fm |
|---|---|---|
| copying both grids into the "previous step" buffers (`store_previous_step_for_freezeout`: `vector<double>` assignments of all 21 fields) | 43 % | 53 % |
| GPU → host copy of both grids (21 fields, float → double) | 31 % | 34 % |
| the scan over the grid (`getCell`, `isfinite`, indexing) | 12 % | 6 % |
| Cornelius' geometry (surface elements) | 14 % | 7 % |

- **About three quarters of the freeze-out is copying, not surface finding.**
  - Every 5 steps MUSIC copies both grids from the GPU, runs Cornelius on them and on the previous freeze-out step's copies, then copies both grids again into those buffers (`evolve.cpp`, freeze-out block).
  - The next freeze-out step overwrites the host grids from the GPU anyway (`host_curr_fresh_` is reset after every RK step).
- **Cheaper than B2, bit-identical** (not implemented, not measured):
  1. **Double-buffer instead of copying the previous step.** Sync from the GPU into two alternating buffer pairs; the previous freeze-out step is then already in the other pair. ~−1.2 core-s per event on ±10.5 fm, ~−4 on ±15 fm. Small effort.
     - Code that reads the host grids after the freeze-out step must sync them itself. `take_checkpoint` reads `max_energy_density_raw(arena_current)` at exactly that point.
  2. **Copy only ε from the GPU for the scan, and the full fields only for the cells of cubes that the surface crosses** (a gather kernel, CUDA and Metal). Saves most of the 1.6 core-s of the grid copy. Moderate effort.
  - Together: ~−30 % CPU per event. That would take O+O from ~7,300 toward the GPU limit of ~10,000 events/h (+25–35 %, estimate), adding almost no GPU work.
- **B2 after that** (Cornelius on the GPU) would save only the remaining ~1.3 core-s (scan and geometry) and add GPU time just when the GPU becomes the limit: not worth it on this machine. It is still worth it where a GPU has few cores.
- **Au+Au** uses the same freeze-out code, but its productions are within 2–9 % of their GPU limits (section above): at most the two-stage campaign's ~9 %.
- Profiles and scripts: `C10_spy/`, `C15_spy/` (`run_oo_spy.sh`), `prof_*_g15`, `prof_*_g10p5` (`chain3.sh`), `stage_cores.py`, `fo_split2.py`, `make_table_oo.py`.

<a id="oo-a10a11"></a>
**A10 and A11 measured: `cu124-freezeoutcopies`** (2026-10-12, later the same day).
- **Image:** `jhputschke/xscape-prod:cu124-freezeoutcopies`, digest `sha256:73fbbc812ae22217c03b6eea1ed5f3c47540c312146f53aa2a6b5634bb9df3de` (ID `7785630caf6a`).
  - X-SCAPE `1098070`; MUSIC4GPU `2ea5a34` (branch `freezeout_copies`).
  - A10: the freeze-out steps alternate between two buffer slots, so nothing is copied into "previous step" buffers.
  - A11: only ε comes to the host. The full fields are gathered on the GPU for the corner cells of the cubes the surface crosses, from the current state and from a copy of the previous checked step kept on the GPU.
  - `MUSIC_FREEZEOUT_GATHER=0` switches A11 off and keeps A10.
- **Runs:** the L10_j8 setup (±10.5 fm grid, 8 jobs per GPU, OMP 3, 60 events per job, default outputs, seeds 1–16), run back to back. The old image was rerun as a control for machine drift and as the physics reference.

| run (±10.5 fm, 8/GPU, OMP 3, 60 events/job) | events/h (makespan) | vs 7,282 | s/event in the job | CPU core-s/event | busy threads of 48 | GPU busy (SM, mean of 2) | GPU-s/event | GPU limit at 90 % busy |
|---|---|---|---|---|---|---|---|---|
| L10_j8, `cu124-music2restart` (earlier) | 7,282 | — | 7.59 | 10.2 | 21.5 | 65 % | 0.646 | ~10,000 |
| L10_j8_rep, `cu124-music2restart` (control) | 7,243 | −0.5 % | 7.70 | 10.3 | 21.7 | 66 % | 0.651 | ~9,950 |
| F10_j8_nogather, `cu124-freezeoutcopies`, A10 only | 10,070 | **+38 %** | 5.51 | 7.1 | 20.2 | 80 % | 0.572 | ~11,300 |
| **F10_j8, `cu124-freezeoutcopies`, A10 + A11** | **10,629** | **+46 %** | **5.14** | **6.4** | 19.2 | **82 %** | **0.553** | **~11,700** |

- **Larger gains than estimated.**
  - CPU per event fell 38 % (10.2 → 6.4 core-s); the estimate was ~−30 %.
  - Events/h rose 46 %, past the old ~10,000 GPU limit; the estimate was +25–35 %.
  - A10 gives most of it: +38 %, −3.1 core-s per event.
  - A11 adds +5.6 % on top of A10, −0.7 core-s per event.
- **GPU time per event also fell, by 15 %** (0.65 → 0.55 GPU-s). That raised the GPU limit to ~11,700 events/h.
  - A10 alone accounts for 12 points of the 15 (0.65 → 0.57). A11 lowers it a little more (→ 0.55), so its gather kernel and on-GPU copy add no net GPU time.
  - Not understood. One possibility: the host copies competed with the other jobs' host-side work, so kernels were launched later. The measured SM % is all this rests on.
- **Now close to the GPU limit:** the GPUs are 82 % busy and the rate is ~91 % of the 90 % limit (Au+Au: 91–98 %). Freeing more CPU now gives only ~10 % at most on this machine.
- **Correctness** (`per_event.py` / `arr_diff.py` against the control run, the 9 seeds with no skipped event in either run, 540 events each):
  - Hydro output grids (`arr`): 0 of 1.3 × 10¹⁰ values differ, both for A10 + A11 and for A10 only.
  - Freeze-out surfaces: bit-identical in all 540 events, both runs. τ_fo, surface cells and the first frame's ε have the same means.
  - Initial states: identical.
  - Partons and jet hadrons: they differ in 96 % (A10 + A11) and 94 % (A10 only) of the events, after a divergence in event 1–4 of each job. This is O+O's known run-to-run irreproducibility, which leaves hydro and surface untouched: the D15 control showed 74 % at 15 events per job, and once a job diverges every later event differs, so at 60 events per job the share is higher. The parton and jet-hadron means agree within errors.
  - The job logs show only the usual Matter "minus log of sudakov negative" skips (2, 2 and 6 events) and PYTHIA remnant warnings.
- **Not tested:** the ±15 fm grid with the new image. There, A10 should save more (~4 core-s per event by the profile above), so the gain should be larger. Au+Au isn't tested either: it was within 2–9 % of its GPU limits, but the 15 % GPU-time drop seen here may also apply.
- **Campaign command:** as above, with `IMAGE=jhputschke/xscape-prod:cu124-freezeoutcopies`, or `cu124`, which now points to the same image. ~10,600 events/h with default outputs is ~320 GB/h (~89 MB/s), more than the ~80 MB/s upload measured on 2026-10-05. For that rate, write hadrons only, or run fewer jobs.
- Data: `F10_j8/`, `F10_j8_nogather/`, `L10_j8_rep/` (`chain4.sh`; `run_oo.sh` now passes `EXTRA_ENV` to `docker run`), `pe_F10*_vs_L10rep.txt`, `arr_F10*_vs_L10rep.txt`. The runs' `.h5` outputs (144 files, ~81 GiB) were deleted on 2026-10-12 after the comparison; logs, `.json` files, monitor logs and comparison outputs are kept.

Raw data on this machine: `~/prod_test/bench/oo_grid_20261012/`: `REPORT.md`, `summary.jsonl`, run directories, the two XMLs, `pe_*.txt`, `odd_*.txt`, `bulk_hadrons.txt`, `cmp_*.txt`, `chain.sh`, `chain2.sh`, `chain3.sh` (profiles) and `chain4.sh` (A10/A11).

<a id="final-optim"></a>
## `cu124-final_optim`: C4 (solver stall break) and D1 (late Δτ) (2026-10-12/13)

The question: what do the GPU-side changes of `Final_Optims.md` give on this machine, and do C4 and
D1 change the physics? Neither is bit-identical. C4 is MUSIC4GPU's stop at float resolution in the
`delta_qi` solvers. D1 is Δτ 0.02 → 0.05 fm/c from τ = 3 fm/c, with the freeze-out checks and frames
kept at 0.1 fm/c and the liquefier following the step.

**Images:**

| | image | X-SCAPE | MUSIC4GPU | js-contrib |
|---|---|---|---|---|
| old | `xscape-prod:cu124-freezeoutcopies`, `7785630caf6a` (A1–A11) | `1098070` | `2ea5a34` | `d234b1d` |
| new | `xscape-prod:cu124-final_optim` = `cu124`, digest `sha256:91c04102fe465849ba76b8b48730345798e3c878c500044d6e092d593ff48eb5` (built 2026-10-13 01:49 UTC) | `9f4b419` | `b972b04` (C4, D1, A12) | `6b8ae36` |

The new image also has iSS `896e198` (Part C, bit-identical) and 3dMCGlauber `b99d680` (the cap on
hard-quark draws). Its XMLs differ from the old only in `late_delta_tau` 0.05 / `late_tau_switch` 3.0,
and for O+O the ±10.5 fm grid (given to the old image with `--user-xml`).

**Setup**
- **Runs:**
  - **O** = old image;
  - **N0** = new, `--late-dtau 0` (C4 + A12; bit-identical to the stall-break build);
  - **N1** = new, defaults (C4 + D1 + A12).
- **Au+Au:** `run_auau.sh jetc`, the campaign setting with `--write-particlize both`, 10 jobs/GPU,
  OMP 3, `--stagger 12`, LBT cache, 20 jobs × 15 events, seeds 1–20.
- **O+O:** `run_oo.sh`, 8 jobs/GPU, OMP 3, 16 jobs × 60 events, ±10.5 fm. Once with full outputs,
  once with hadron files only (`--bulk-hadrons 10 --no-particlize --no-hydro-file`).
- **Common:** one container per GPU, MPS, `MUSIC_CUDA_SYNC=block`, `OMP_WAIT_POLICY=passive`; the
  monitors and events/h conventions as in the sections above.
- **Quiet machine:** no other user's process went above 20 % CPU (`otherusers.log`, sampled every 10 s).
- **The switch is logged:** every N1 leg says `late step: delta_tau 0.02 -> 0.05 fm/c at step 130
  (tau = 3 fm/c), facTau 5 -> 2, a frame every 2 steps`.

**Throughput, 2 GPUs:**

| run | events/h makespan (per event) | change | s/event in job | core-s/event | threads busy (of 48) | GPU0 / GPU1 busy | GPU-s/event | host peak | per-job VmHWM, mean [max] | VRAM/GPU |
|---|---|---|---|---|---|---|---|---|---|---|
| Au+Au O | 1,271 (1,487) | | 48.5 | 37.6 | 13.8 | 89 / 89 % | 5.06 | 108.1 GB | 5.95 [6.65] GB | 9.7 GB |
| Au+Au N0 | 1,628 (2,025) | +28 % | 35.6 | 34.4 | 16.2 | 81 / 80 % | 3.58 | 106.8 GB | 5.94 [6.45] GB | 9.6 GB |
| **Au+Au N1** | **2,160** (2,846) | **+70 %** | 25.4 | 35.4 | 22.0 | 62 / 63 % | 2.09 | 105.0 GB | 5.97 [6.66] GB | 9.6 GB |
| O+O O, full outputs | 10,435 (10,965) | | 5.3 | 6.5 | 19.3 | 83 / 79 % | 0.558 | 24.2 GB | 1.39 [1.44] GB | 3.6 GB |
| O+O N0, full outputs | 11,949 (12,505) | +15 % | 4.6 | 6.0 | 20.2 | 72 / 70 % | 0.426 | 24.5 GB | 1.40 [1.44] GB | 3.6 GB |
| O+O N1, full outputs | 12,700 (13,384) | +22 % | 4.3 | 5.9 | 21.3 | 61 / 60 % | 0.342 | 24.6 GB | 1.40 [1.44] GB | 3.6 GB |
| O+O O, hadron files only (FH10_j8, 2026-10-12 15:00) | 10,313 (10,924) | | 5.3 | 7.3 | 21.2 | 80 / 77 % | 0.55 | 26.0 GB | 1.43 [1.49] GB | 3.6 GB |
| O+O N0, hadron files only | 12,722 (13,190) | +23 % | 4.4 | 6.1 | 21.8 | 76 / 73 % | 0.421 | 25.3 GB | 1.42 [1.47] GB | 3.6 GB |
| **O+O N1, hadron files only** | **13,728** (14,260) | **+33 %** | 4.0 | 6.1 | 23.5 | 66 / 65 % | 0.344 | 24.9 GB | 1.43 [1.48] GB | 3.6 GB |

All jobs completed: 300 Au+Au events per run, none skipped. The only skips are 2–5 O+O events per
run (Matter's "minus log of sudakov negative"). Every log says `host sync: block`. No OOM, and the
watchdog never fired.

- **GPU time per Au+Au event:** C4 −29 %, D1 another −42 %. These are the GB10 estimates of
  Final_Optims.md (−29 %, −43 %).
- **O+O gains less from D1** (−20 %): its fluid freezes out at τ ≈ 4.4 fm/c, soon after the switch.
- **Host-bound again:**
  - Au+Au N1: 35 core-s/event × 2,160 events/h ≈ 21 of the 24 physical cores, GPUs 63 % busy.
  - O+O hadrons only: ~24 busy threads, ~570 events/h per physical core; the GPU limit is ~18,800.
  - Final_Optims.md predicted the host limit at ~1,400 Au+Au events/h with the pair file. The runs
    beat that because the old image's CPU per event (37.6 core-s) was already far below the 62 of
    `music2restart` (A10/A11).
- **The stagger ramp** (9 × 12 s) is a fifth of the Au+Au N1 makespan (500 s) with 15-event jobs.
  45-event jobs should give ~2,400–2,500 events/h (not measured).
- **A10/A11 for Au+Au** (old image vs `music2restart`, not measured before): 1,121 → 1,271 (+13 %).
- **Host memory unchanged:** 105–108 GB at 20 Au+Au jobs (17–21 GB left). VRAM unchanged.
- **Hadrons only vs full outputs, new image:** hadrons-only is now 8 % faster for O+O (13,728 vs
  12,700) at about the same CPU. With the old image the two were level (10,313 vs 10,435–10,629).
- **Disk:** Au+Au 0.243 GB/event, so ~525 GB/h at 2,160 events/h, more than the ~290 GB/h OSDF upload
  (2026-10-05). Use `--particlize-only` (not re-measured with this image) or local disk. O+O 28 MB/event
  full, 0.13 MB hadrons only.

**Physics check** (details, methods and every number: [REPORT_FINAL_OPTIM.md](REPORT_FINAL_OPTIM.md))

- **C4: float-rounding level.**
  - Au+Au backgrounds and O+O fields: energy-weighted |Δe|/e above e_fo ~1e-6.
  - O+O surfaces +6e-6, τ_fo identical; deposition (same showers) to 1e-6.
  - Wake fields agree to ~1e-3 (L2), with a 2 % tail in a few events.
  - At hadron level the Au+Au wake yield shifts +0.8 % ± 0.2 %: 6e-5 of the jet leg's hadrons, visible
    only because correlated sampling cancels the noise.
- **D1: the medium changes at 1.1e-3** (Au+Au, τ ≥ 3; O+O 1.2e-4), below the GPU-vs-CPU 2e-3. It is
  bit-identical before 3 fm/c.
  - **The liquefier follows the step:** the deposited energy agrees to 1e-4 where the showers agree,
    and the wake carries it (wake / deposit +0.3 % ± 0.2 %).
  - **The wake field changes by ~5 % cell by cell** (L2, same-shower events), almost all of it short-range:
    2.3 % after 0.5 fm smoothing, 0.8 % after 2 fm. Its positive and negative parts, centroid
    (0.03 fm) and width agree at 0.1–0.2 %.
  - **Hadron level** (`hadronize.py --correlated`, seed-matched, errors clustered by background): no
    change within ±2 % in the wake's yield, energy and Δφ shape. Bulk hadrons: multiplicity +7.5e-5,
    ⟨pT⟩ +1e-4.
  - **O+O:** the surface is **+0.20 % larger in every event** and τ_fo one frame later in 11 % of
    events; bulk hadrons +0.05 % ± 0.03 % (all), +0.4 % ± 0.2 % (midrapidity).
  - **Showers diverge** in half the Au+Au events (157/300 the same; GPU vs CPU: 3 of 6), with the
    same average deposit (40.5 vs 40.4 GeV).
- **Verdict:**
  - C4: accept.
  - D1: meets the plan's acceptance criterion. Use one D1 setting per campaign: the cell-level wake
    and the O+O surface shift at the 5 % and 0.2 % levels.
  - The hadron-level wake precision of this check is ±2 % (20 backgrounds). A 100-background
    comparison would be the follow-up.

**A failed job in the old-image run (A_O seed 4).**
- **The symptom:** all 15 events had 1–5 droplets (81 in the job, ~330 otherwise) and 9–18 partons,
  against 55–129. The background hydro and the shower initiators were correct.
- **Not reproducible:** seed 4 run alone with the same old image gives the normal result, identical to N0.
- **Rare:** the only such job among ~250 campaign jobs on this box since 2026-10-04 (a
  `droplets_total` scan of every job's `.json`). Cause unknown; no warning in the log.
- **Recommendation:** campaigns should flag and redo jobs whose `droplets_total / events_written` is
  below half the campaign median. The C4 comparisons above exclude it.

**Campaign defaults for this machine** (the same launcher settings, the new image pinned by digest):

```bash
IMG=jhputschke/xscape-prod@sha256:91c04102fe465849ba76b8b48730345798e3c878c500044d6e092d593ff48eb5
# Au+Au: ~2,160 events/h in 15-event jobs, ~2,400-2,500 expected in 45-event jobs; ~525 GB/h with the pair file
IMAGE=$IMG JOBS_PER_GPU=10 OMP_THREADS=3 MUSIC_CUDA_SYNC=block nohup ./launch_2gpu.sh \
    --lbt-cache /work/lbt_cache/lbt_tables_v1.bin --stagger 12 60 45 0 /data/camp_pth3 --campaign pth3 \
    --pthat-bins 10-20,20-30,30-40 --jets-per-bin 5 --parton-ymax 0.6 \
    --write-particlize both > /data/camp_pth3.log 2>&1 &
# O+O, hadron files only: ~13,700 events/h (the +-10.5 fm grid is the XML default now: no --user-xml)
IMAGE=$IMG nohup ./launch_2gpu_oo.sh 64 100 0 /data/OO_c1 --campaign c1 \
    --bulk-hadrons 10 --no-particlize --no-hydro-file > /data/OO_c1.log 2>&1 &
```

60 × 45 = 2,700 Au+Au events take ~1.1–1.3 h (was ~2.4 h) and ~0.65 TB.

Raw data on this machine: `~/prod_test/bench/final_optim_20261012/`:
- `chain.sh`, `summary.jsonl`, the run directories `A_*`, `F_*`, `FH_*` (`FH_O` → `../oo_grid_20261012/FH10_j8`);
- `rerun4/`: seed 4 alone with each image;
- `cmp/`: `summary_auau.txt`, `pe_*`, `arr_*`, `scale_*`, `had_*`, `leg_*`;
- the scripts `wake_cmp.py`, `cmp_summary.py`, `wake_scale.py`, `had_wake.py`, `had_cmp.py`,
  `leg_spec.py`, `oo_field.py`, `run_cmp.sh`, `run_had.sh`.
