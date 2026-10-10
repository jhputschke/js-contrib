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

**Recommended launch: [`launch_2gpu.sh`](launch_2gpu.sh)** (usage in [README_launch.md](README_launch.md)).
It starts one container per GPU, each with `run_jobs.sh -j 8 --mps` and `OMP_NUM_THREADS=4`,
splits NJOBS between the GPUs and writes to `OUTBASE/gpu0` and `OUTBASE/gpu1`. OUTBASE can be
under `~/prod_test` (`/work/...`) or on another disk, e.g. `/data/...` (1.6 TB free; tested).
The campaign default for this machine is the `--pthat-bins` setup [below](#suggested-campaign-default).

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
  headroom). 10 jobs/GPU would not fit with particlize (~6 GB/job → ~120 GB).
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
  likely not fit.
- **Re-check the campaign default's memory** with the new image (particlize at 8/GPU: 97 GB
  with the baseline; the first-event spike could push it toward ~110 GB).
- **The throughput lever on this machine is now host memory per job**, then GPU work per
  event; the CPU comes last. In order of value for effort:
  1. **Read the jet leg frame by frame** (a per-frame accessor in `bind_music.cc`, a
     `NativeFrames` reader like `FrameworkFrames`): −1.0–1.3 GB per job at the peak
     (13–17 %), particlize peak ~8.1 → ~6.9 GB.
  2. **Memory left between events:** test `MALLOC_ARENA_MAX=2` / `MALLOC_TRIM_THRESHOLD_`
     (no code change); maybe another ~1 GB per job.
  3. **Share the LBT tables between jobs** (read-only `mmap`): ~0.6 GB per job.
  4. **A staggered start** in `run_jobs.sh`: removes the first-event spike.
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
