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
