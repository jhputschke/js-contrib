# Throughput summary (GB10, M3 Max, 2 × RTX 3090) and cluster GPU sizing (updated 2026-10-13)

Two productions:
- **Au+Au two-stage** (`prod_AuAu_0_10_jet`: MUSIC_1 hydro, then MUSIC_2 with jet deposition) at the
  campaign setting: `--pthat-bins 10-20,20-30,30-40 --jets-per-bin 5 --parton-ymax 0.6`, one
  background per 15 jet events.
- **O+O hadron files only** (`prod_OO_isr --bulk-hadrons 10 --no-particlize --no-hydro-file`).

Part 1 is measured. Part 2 is an extrapolation: none of the cluster GPUs has been run.

**Update 2026-10-13: the `final_optim` image.** `jhputschke/xscape-prod:cu124-final_optim`
(= `cu124`, digest `sha256:91c04102fe465849ba76b8b48730345798e3c878c500044d6e092d593ff48eb5`) adds:
- **C4:** the solver stall break, −29 % GPU time per step;
- **D1:** Δτ 0.02 → 0.05 fm/c from τ = 3 fm/c, −42 % more for Au+Au;
- **A12:** Cornelius, bit-identical.

**Update 2026-10-14:** rebuilt from the merged branches as `cu124` / `cu126` / `cu130` =
`cu1xx-20261013-9e84450` (X-SCAPE `9e84450`, MUSIC4GPU `46dad43`, js-contrib `e8997c7`, iSS
`98deaff`), with Matter's guard against a qhat lookup before t = 0 and the iSS leak fix on top;
Au+Au output is bit-identical to `cu124-final_optim`, so the numbers below hold.

On the 3090 box it gives **Au+Au +70 % (1,271 → 2,160 events/h)** and **O+O hadrons only +33 %
(10,313 → 13,728)**. The physics checks are summarised in §3; details in
[REPORT_FINAL_OPTIM.md](REPORT_FINAL_OPTIM.md). The GB10 and M3 Max rows are not re-measured with it.

## 1. Measured: current best settings per machine

events/h by makespan; the per-event rate is in parentheses. Memory is the machine-wide peak of
used memory (idle included), and per job the VmHWM mean [max]. Sources:
[REPORT_FINAL_OPTIM.md](REPORT_FINAL_OPTIM.md), [BENCHMARK_2x3090.md](BENCHMARK_2x3090.md),
`BENCHMARK_GB10.md`, `BENCHMARK_M3MAX.md`, `prod_OO_isr/README.md`, [REPORT_OO.md](REPORT_OO.md).

### Au+Au two-stage

| machine | image / code | setting | events/h (machine) | events/h per GPU | GPU busy | CPU core-s/event | GPU-s/event | host memory peak | per job | VRAM | disk/event |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **2 × RTX 3090**, 3960X 24 c, 125 GB | **`cu124-final_optim`** (A1–A12, C4, D1) | 10/GPU, OMP 3, MPS, `block`, LBT cache, stagger 12; **pair + particlize**; 15-event jobs | **2,160** (2,846) | **1,080** | 63 % | 35.4 | 2.09 | 105.0 GB | 5.97 [6.66] GB | 9.6 GB/GPU | **0.243 GB** |
| | `final_optim`, `--late-dtau 0` (C4, no D1) | same | 1,628 (2,025) | 814 | 81 % | 34.4 | 3.58 | 106.8 GB | 5.94 [6.45] GB | 9.6 GB/GPU | 0.243 GB |
| | `cu124-freezeoutcopies` (A1–A11) | same | 1,271 (1,487) | 636 | 89 % | 37.6 | 5.06 | 108.1 GB | 5.95 [6.65] GB | 9.7 GB/GPU | 0.243 GB |
| | `cu124-music2restart` (A1–A8) | same | 1,121 (1,299) | 560 | 83 % | 62.3 | 5.29 | 100.6 GB | 5.84 [6.44] GB | 9.1 GB/GPU | 0.236 GB |
| | `cu124-music2restart`, **`--particlize-only`** | same | 1,237 (1,442) | 620 | 89 % | 50.1 | 5.16 | 74.0 GB | 4.21 [4.57] GB | 8.9 GB/GPU | **0.097 GB** |
| **GB10**, 20 c, 121 GB unified | js-contrib `music2_restart` (A1–A8) | `-j 4 --mps`, OMP 5, `block`; pair + particlize; no cache, no stagger | **381** | 381 | 89 % | 26.2 | ~7.6 | 31.5 GB | ~6.4 GB¹ | (unified) | 0.24 GB² |
| **M3 Max**, 16 c, 64 GB unified | `cuda_blocking_sync` branch (no A1–A9) | `-j 3`, OMP 5, passive, LBT cache; **plain** (new background every event, pT̂ 50–70, no particlize) | 263 (284) | 263 | 83 % | – | – | 33.7 GB (13 GB above idle) | ~6.0 GB footprint (alone) | (unified) | – |

¹ One job alone at the campaign setting after the memory fixes (6.42 GB).
² The output format is the same everywhere; the size was measured on the 3090.

- **The 3090 box is host-bound again with `final_optim`:** 22 of 48 threads are busy, GPUs 63 %.
  - 35 core-s per event × 2,160 events/h ≈ 21 cores of the 24.
  - The 15-event jobs' stagger ramp (108 s of a 500 s run) pulls the makespan rate down. With 45-event
    jobs expect ~2,400–2,500 events/h (estimate).
- **`--particlize-only` hasn't been measured with `final_optim`.** With the earlier image it gave
  +10 % events/h, 1.6 GB less per job and 0.097 GB per event.
- **Disk per hour:** 3090 box ~525 GB/h (pair + particlize) with `final_optim`, ~0.21 TB/h with
  `--particlize-only` (estimate). The OSDF upload measured ~80 MB/s ≈ 290 GB/h, so with the pair
  file a campaign outruns the upload: use `--particlize-only` or local disk.
- **GB10 and M3 Max** are GPU-bound, so C4 and D1 should help them most. The Final_Optims estimate for
  the GB10 is ~750–800 Au+Au events/h. Neither is measured with the new code.
- The M3 Max row is **not** the campaign setting and predates A1–A9. Its closest GB10 counterpart
  (plain, no particlize) was 205 events/h.

### O+O, hadron files only

| machine | image / grid | output | setting | events/h (machine) | per GPU | GPU busy | core-s/event | GPU-s/event | host peak | per job | VRAM | disk/event |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **2 × RTX 3090** | **`cu124-final_optim`, ±10.5 fm** (XML default) | **hadron files only** (iSS in job, 10 oversamples) | 8/GPU, OMP 3, MPS, `block`, 60 ev/job | **13,728** (14,260) | **6,860** | 65 % | 6.1 | 0.344 | 24.9 GB | 1.43 [1.48] GB | 3.6 GB/GPU | **0.13 MB** |
| | same, `--late-dtau 0` | hadron files only | same | 12,722 (13,190) | 6,360 | 74 % | 6.1 | 0.421 | 25.3 GB | 1.42 [1.47] GB | 3.6 GB/GPU | 0.13 MB |
| | `final_optim` | full outputs, no iSS³ | same | 12,700 (13,384) | 6,350 | 60 % | 5.9 | 0.342 | 24.6 GB | 1.40 [1.44] GB | 3.6 GB/GPU | 28 MB |
| | `cu124-freezeoutcopies` (A10+A11), ±10.5 fm | hadron files only | same | 10,313 (10,924) | 5,160 | 79 % | 7.3 | 0.55 | 26.0 GB | 1.43 [1.49] GB | 3.6 GB/GPU | 0.12 MB |
| | `freezeoutcopies` | full outputs, no iSS³ | same | 10,435 – 10,629 | 5,250 | 81–82 % | 6.4–6.5 | 0.55–0.56 | 24–26 GB | 1.39 [1.44] GB | 3.6 GB/GPU | 28–30 MB |
| | `music2restart`, ±10.5 fm / ±15 fm | hadron files only | 8/GPU, OMP 3, 15 ev/job | 6,444 / 3,798 | 3,220 / 1,900 | 58 % | 13.3 / 21.7 | 0.64 / 1.1 | 21.8 / 35.4 GB | 1.37 / 2.45 GB | 3.4 / 4.4 GB/GPU | 0.15 MB |
| **GB10** | 2026-10-10 code, ±15 fm | hadron files only | `-j 8 --mps`, OMP 2 | **1,620** | 1,620 | 92 % | – | 2.0 | 28.5 GB (~7 idle) | ~2.7 GB | (unified) | 0.15 MB |
| **M3 Max** | – | **not measured** | | – | | | | | | | | |

³ Full outputs are the hydro file, the particlize file (freeze-out surfaces and final partons) and
the jet fragmentation hadrons. No bulk hadrons: they would be sampled later from the particlize
file (`run_hadronize.py`, CPU only). With `final_optim`, hadrons-only is 8 % *faster* than full
outputs at about the same CPU per event (6.1 vs 5.9 core-s). It writes ~200× less; the reason for the
wall-time difference was not examined.

- **The 3090 box is host-bound for O+O:** ~24 busy threads (the physical core count) and GPUs 65 %
  busy. That is ~570 events/h per physical core (13,728 / 24); the GPU limit is ~18,800 events/h
  (0.344 GPU-s per event at 90 % busy).
- **Disk:** hadron files ~1.8 GB/h (0.13 MB/event at 10 oversamples; ~1.2 MB at 100, a linear
  estimate). Full outputs would write ~360 GB/h, more than the ~290 GB/h upload.
- **GB10:** GPU-bound (92 %) on the ±15 fm grid with code older than A1–A12. The ±10.5 fm grid cut GPU
  time per event by 41 % on the 3090, and C4 + D1 by another 38 %, which suggests **~3,500–4,500
  events/h**. Not measured.

## 2. Extrapolated: per-GPU settings for cluster nodes

### Model

The measured quantities per production, from the 3090 box with `final_optim`:

| per event | Au+Au two-stage, pair + particlize | O+O hadrons only, ±10.5 fm |
|---|---|---|
| GPU-s (RTX 3090) | **2.09** (was 5.3) | **0.344** (was 0.55) |
| host core-s (Zen 2 under load; GB10 26 for Au+Au on older code) | **35** (was 62) | **6.1** (was 7.3) |
| s per event in a loaded job | 25 | 4.0 |
| host memory per job, plan with (VmHWM + page cache) | 7 GB (5 GB with `--particlize-only`) | 2 GB |
| VRAM per job | ~1 GB | ~0.45 GB |

**GPU speed relative to the RTX 3090.** MUSIC4GPU runs FP32 only, so FP64 rates don't matter. The
kernels are partly memory-bound: the GB10 has 0.8× the 3090's FP32 issue rate but 0.3× its memory
bandwidth, and it is 1.7× slower per RK step (7.25 vs 4.2 ms). A two-term fit to those two GPUs
(compute + bandwidth) gives:

| GPU | SMs × clock, FP32 lanes/SM | memory BW | GPU time per event vs 3090 (estimate) | sm in `:cu124` |
|---|---|---|---|---|
| V100 (16/32 GB) | 80 × 1.5 GHz, 64 | 0.9 TB/s | **1.5–2.0×** | sm_70 ✓ (not in `cu130`) |
| A40 (48 GB) | 84 × ~1.6 GHz (300 W), 128 | 0.7 TB/s | **1.0–1.2×** | sm_86 ✓ |
| A100 (40/80 GB) | 108 × 1.4 GHz, 64 | 1.6–2.0 TB/s | **0.9–1.4×** | sm_80 ✓ |
| L40S (48 GB) | 142 × ~2.4 GHz, 128; 96 MB L2 | 0.86 TB/s | **0.45–0.65×** | sm_89 ✓ |

`docs/Plans/Final_Optims.md` §6 scales by FP32 peak only (V100 ~0.45, A100 ~0.6, L40S ~1.5–2 of a
3090), which is more pessimistic for V100 and A100. The truth is probably between the two; one short
run per GPU type settles it.

Then, per GPU:
- **GPU ceiling** = 0.9 × 3600 / (GPU-s per event).
- **Host limit** = cores × 3600 / core-s × ~0.85 (headroom for the OpenMP bursts at string deposition
  and for contention). That gives **~85 Au+Au events/h** and **~500 O+O events/h per Zen 2-class
  physical core**. Newer cores (Zen 3/4, Sapphire Rapids) are probably 1.2–1.5× that.
- **Expected rate** = the smaller of the two.
- **Jobs per GPU** (Little's law) ≈ events/h × (s/event in a loaded job) / 3600, with 25–35 s for
  Au+Au and 4–6 s for O+O. Round up by 1–2.
- **Memory** = jobs × per-job plan + ~8 GB per node. SLURM counts the page cache of the HDF5 files
  being written, so add 10–15 %.

**What changed with `final_optim`:**
- The GPU work per event fell 2.4× (Au+Au) and 1.6× (O+O); the CPU work by 1.8× and 1.2×.
- **So cores per GPU now decide the rate on most nodes**, for both productions, and more jobs per GPU
  are needed to keep a GPU busy.
- An Au+Au job keeps ~1 core busy on average (22 threads for 20 jobs); give it 2 threads.

### Au+Au two-stage, campaign setting, pair + particlize (per GPU)

| GPU | GPU ceiling | expected events/h/GPU | limit | jobs/GPU (`-j`) | `OMP_NUM_THREADS` | cores/GPU (`--cpus-per-task`) | host mem/GPU (`--mem`) | VRAM used | disk/GPU-h |
|---|---|---|---|---|---|---|---|---|---|
| (RTX 3090, measured) | ~1,550 | **1,080** (~1,200 in long jobs) | cores (12 per GPU) | 10 | 3 | 12 phys (24 thr) | ~53 GB | 9.6 GB | 260 GB |
| **V100** | 770–1,040 | **700–950** | GPU | 8 | 2 | 12–16 | 64 GB | ~8 GB (fits 16 GB) | 170–230 GB |
| **A40** | 1,300–1,550 | **1,100–1,400** | cores at 16, GPU at 24 | 10–12 | 2 | 16–24 | 90–100 GB | ~11 GB | 270–340 GB |
| **A100** | 1,100–1,700 | **1,000–1,500** | GPU or cores | 10–12 | 2 | 16–24 | 90–100 GB | ~11 GB | 240–365 GB |
| **L40S** | 2,400–3,450 | **2,000–2,900** with 24–32 cores | **cores** | 18–22 | 2 | **24–32** | 135–165 GB (100–120 GB with `--particlize-only`) | ~20 GB | 490–700 GB |

- **Disk and upload are now a limit.** One L40S writes up to ~0.7 TB/h with the pair file. Use
  `--particlize-only` (0.097 GB/event, ~2.5× less) unless the hydro history is needed, and node-local
  scratch with `upload_follow.py`.
- **`--particlize-only`** also saves ~1.6 GB per job and, on the earlier image, gave +10 % events/h.
  Not re-measured with `final_optim`.
- **If a node gives fewer cores per GPU** than in the table, lower `-j`, not the threads per job.
  Expect ~85 events/h per Zen 2-class core.

### O+O hadron files only, ±10.5 fm grid (per GPU)

| GPU | GPU ceiling | expected events/h/GPU | limit | jobs/GPU | OMP | cores/GPU | mem/GPU | VRAM | disk/GPU-h (hadron files) |
|---|---|---|---|---|---|---|---|---|---|
| (RTX 3090, measured) | ~9,400 | **6,860** | cores (12 per GPU) | 8 | 3 | 12 phys | ~13 GB | 3.6 GB | ~1 GB |
| **V100** | 4,700–6,300 | **4,200–5,700** | GPU | 8 | 2 | 12–16 | 24 GB | ~4 GB | <1 GB |
| **A40** | 7,900–9,400 | **7,000–8,500** with 16 cores | cores | 10–12 | 2 | 16 | 32 GB | ~4.5 GB | ~1 GB |
| **A100** | 6,700–10,400 | **6,000–8,800** with 16 cores | GPU or cores | 10–12 | 2 | 16 | 32 GB | ~4.5 GB | ~1 GB |
| **L40S** | 14,700–20,900 | **12,000–17,000** with 24–32 cores | **cores** | 16–20 | 2 | 24–32 | 48 GB | ~8 GB | ~2 GB |

- The **±10.5 fm grid** is now the production XML's default. It was validated for b < 3 fm at 200
  GeV (C1 check).
- **Kernel size:** the ±10.5 fm grid has 294k cells, a small kernel for an A100 or L40S. Several jobs
  under MPS are needed to fill the GPU, so the L40S ceiling is the least certain number here.
- **Memory and disk** are no constraint for hadron files. Full outputs (28 MB/event) would write
  ~0.35–0.5 TB/h per L40S.

### Settings common to all GPUs

- **Image:** `jhputschke/xscape-prod@sha256:2cf57f7556dcfff0289c1c46b9b188b58a3ce8026b2733cc4a17f90bca761b58`
  (`cu124-20261013-9e84450`, = `cu124` of 2026-10-14; `cu126-20261013-9e84450` for R560+ drivers and
  arm64 hosts). sm_70/75/80/86/89/90 + PTX: V100, A40, A100 and L40S. Pin the digest for a
  campaign. `cu130` has no V100 support. The numbers above were measured with `cu124-final_optim`
  (`sha256:91c04102…`), whose Au+Au output is bit-identical; it lacks the Matter t < 0 guard and
  the iSS leak fix, which matter for O+O.
- **One D1 setting per campaign:** the default late Δτ (0.05 from 3 fm/c) or `--late-dtau 0`, not a mix
  of the two in one data set (§3).
- `MUSIC_CUDA_SYNC=block` (−30 % CPU), `OMP_WAIT_POLICY=passive`, and `OPENBLAS_NUM_THREADS=1`
  (the default).
- `LBT_TABLE_CACHE` on node-local disk, one file per LBT-tables version. Don't put it in the
  container's `/dev/shm`. Saves ~0.6 GB and 6–11 s of startup per Au+Au job.
- **Au+Au:** `run_jobs.sh --stagger 12` (−11 GB peak at 20 jobs). Long jobs (45 events, 3 backgrounds)
  amortise startup and the stagger ramp.
- **CUDA MPS:** +5–7 % on the 3090s and +12 % on the GB10, but **never tried on a cluster**. It
  needs a per-job daemon (as `run_jobs.sh --mps` starts it) and a GPU not in exclusive-process mode
  held by another user. Without MPS, expect ~5–10 % less, and maybe 1–2 more jobs per GPU.
- **A per-job sanity check:** flag and redo any Au+Au job whose `droplets_total / events_written`
  (in its `.json`) is below half the campaign median (§3).
- **Before a campaign, on the first node:** a 20–30 min sweep of `-j` (e.g. 8, 12, 16 for Au+Au;
  8, 12, 16 for O+O) at the planned `--cpus-per-task`. Record events/h by makespan,
  `nvidia-smi dmon` SM %, and the cgroup's memory peak. If SM busy stays below ~75 % while the
  cores are full, the node is host-limited: more cores per GPU pays, more jobs doesn't.

## 3. Physics checks of the `final_optim` changes (for completeness)

C4 and D1 are not bit-identical, so they were checked on the 3090 box against the previous image
with the same seeds:
- Au+Au: 300 campaign events, 20 backgrounds, pair + particlize files, then hadronized with correlated,
  seed-matched iSS sampling.
- O+O: 720 aligned events with full outputs, plus the hadrons-only runs.

Details, methods and scripts: [REPORT_FINAL_OPTIM.md](REPORT_FINAL_OPTIM.md),
`bench/final_optim_20261012/`.

| quantity | C4 (stall break) | D1 (late Δτ) | reference: GPU vs CPU |
|---|---|---|---|
| hydro fields, energy-weighted ⟨\|Δe\|/e⟩ above e_fo (Au+Au background, τ ≥ 3) | ~1e-6 | **1.1e-3** (bit-identical before 3 fm/c) | 2.0–2.2e-3 |
| O+O hydro fields, same measure | ~5e-7 | 1.2e-4 (max 6.5e-4) | |
| jet deposition (same showers) | to 1e-6 | to 1e-4: the liquefier follows the step change | |
| wake energy / deposited energy | unchanged (±0.1 %) | +0.3 % ± 0.2 % (all events) | ~1 % |
| wake field δe = e_jet − e_bg, cell by cell (L2, median) | ~1e-3 | **~5 %**; 2.3 % at 0.5 fm smoothing, 0.8 % at 2 fm | 0.1–0.2 % |
| wake's positive / negative parts, centroid, width | unchanged | within 0.1–0.2 %, centroid within 0.03 fm | |
| hadron-level wake (jet − background, charged, \|η\| < 1) | +0.8 % ± 0.2 % in yield (6e-5 of the jet leg) | no change within ±2 % (yield, energy, Δφ shape) | |
| Au+Au bulk hadrons (backgrounds) | 7e-6 | multiplicity +7.5e-5, ⟨pT⟩ +1e-4 | 0.2–0.3 % |
| O+O freeze-out surface | +6e-6 | **+0.20 % ± 0.002 %** (median 0.21 %, max 0.34 %); τ_fo one frame later in 11 % of events | |
| O+O bulk hadrons | −0.02 % ± 0.02 % (all), +0.3 % ± 0.16 % (midrapidity) | +0.05 % ± 0.03 % (all), +0.4 % ± 0.2 % (midrapidity) | 0.2–0.3 % |
| showers the same (Au+Au) | 284 / 300 | 157 / 300 (the medium differs at 1e-3; average deposit unchanged) | 3 of 6 |

- **C4:** float-rounding level, as claimed. Accept.
- **D1 meets the plan's acceptance criterion:** medium differences at or below the GPU-vs-CPU level,
  correct deposition normalisation, no grid-edge hits.
  - It does change the cell-level wake field by ~5 %, mostly at scales below ~1 fm.
  - It makes the O+O surface 0.2 % larger, systematically.
  - So: one D1 setting per campaign. If an FNO is trained on cell-level wake fields, its target
    changes at that level.
  - The hadron-level precision of the check is ±2 % (20 backgrounds). A 100-background run would
    take it to ±0.5–1 %.
- **A rare failure, seen once:** in the previous image's run, one Au+Au job (of ~250 campaign jobs on
  this box since 2026-10-04) lost almost no jet energy in all 15 events.
  - Its background hydro and starting partons were correct.
  - Run alone, the same seed and image gave the normal result.
  - The cause is unknown, and the job log shows no warning. Hence the per-job sanity check above.

## What would firm this up

1. **One short run on each GPU type** (one GPU, Au+Au at `-j 12`, O+O at `-j 8`): it pins down the
   GPU speed factor, now the largest uncertainty together with the core speed.
2. **`--particlize-only` with `final_optim`** on the 3090 box, and a 45-event-job Au+Au campaign
   for the long-job rate.
3. **GB10 and M3 Max with the current images:** both are GPU-bound, so C4 and D1 should give them
   the largest gains. Also the Metal build of C4 (Final_Optims asks for it).
4. **A 100-background Au+Au comparison** with `--late-dtau 0` vs the default, for the hadron-level wake
   at ±0.5–1 %.
