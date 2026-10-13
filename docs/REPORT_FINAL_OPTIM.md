# `cu124-final_optim` on 2 × RTX 3090: throughput, and the physics check of C4 and D1 (2026-10-12/13)

The image: `jhputschke/xscape-prod@sha256:91c04102fe465849ba76b8b48730345798e3c878c500044d6e092d593ff48eb5`
(= `cu124-final_optim` = `cu124`). Its `BUILD_INFO.txt` matches what was announced: X-SCAPE `9f4b419`,
MUSIC4GPU `b972b04`, js-contrib `6b8ae36`, iSS `896e198`, 3dMCGlauber `b99d680`, CUDA 12.4.1,
`sm_70;75;80;86;89;90`. The production XMLs differ from the previous image only in `late_delta_tau`
0.05 / `late_tau_switch` 3.0, and for O+O the ±10.5 fm grid.

Three runs per production, same seeds, so C4 and D1 separate:

| run | image | contains |
|---|---|---|
| **O** | `cu124-freezeoutcopies` (`7785630caf6a`) | A1–A11: the reference |
| **N0** | new, `--late-dtau 0` | + C4 (solver stall break) + A12 (Cornelius, bit-identical) |
| **N1** | new, defaults | + D1 (Δτ 0.02 → 0.05 fm/c from τ = 3 fm/c) |

**Short answer**
- **Au+Au campaign: 1,271 → 2,160 events/h (+70 %).** C4 gives +28 %, D1 another +33 %.
- **O+O: hadron files only 10,313 → 13,728 events/h (+33 %); full outputs 10,435 → 12,700 (+22 %).**
- **GPU time per event:** C4 −29 % (Au+Au), as measured on the GB10. D1 −42 % more (Au+Au) and −20 %
  (O+O, whose events end at τ ≈ 4.4 fm/c). The box is **host-bound again** (GPUs 60–65 % busy).
- **C4 is float-rounding level** for the hydro and the surfaces, as claimed.
- **D1 changes the medium at the 1e-3 level**, below the GPU-vs-CPU difference (2e-3):
  - the jet's deposition is normalised correctly, and the wake carries the same energy;
  - the wake field differs cell by cell by ~5 %, almost all of it at scales below ~1 fm;
  - at hadron level no change is seen at the ~2 % precision of this sample.
  - O+O: the surface is 0.20 % larger and τ_fo shifts by one frame in 11 % of events.
- **Separate finding:** one job of the old-image campaign (A_O seed 4) lost almost no jet energy in
  all 15 events. It doesn't reproduce: the same seed with the same image alone is identical to the
  new image. It is 1 job in ~250 campaign jobs on this box. Details in §4.

Raw data, scripts and logs: `~/prod_test/bench/final_optim_20261012/` (`chain.sh`, `summary.jsonl`,
`cmp/`).

## 1. Throughput (2 × RTX 3090, Threadripper 3960X)

Settings as `launch_2gpu.sh` / `launch_2gpu_oo.sh`: one container per GPU, `run_jobs.sh --mps`,
`MUSIC_CUDA_SYNC=block`, `OMP_WAIT_POLICY=passive`, LBT cache.
- **Au+Au:** 10 jobs/GPU, OMP 3, `--stagger 12`, the campaign setting with `--write-particlize both`,
  20 jobs × 15 events.
- **O+O:** 8 jobs/GPU, OMP 3, 16 jobs × 60 events, ±10.5 fm grid.

events/h by makespan (per-event rate in parentheses). GPU-s/event = mean SM busy × 2 × 3600 / events/h.

| run | events/h | change | s/event in job | core-s/event | threads busy (of 48) | GPU busy | GPU-s/event | host peak | VmHWM/job mean [max] | VRAM/GPU |
|---|---|---|---|---|---|---|---|---|---|---|
| **Au+Au O** | 1,271 (1,487) | | 48.5 | 37.6 | 13.8 | 89 % | 5.06 | 108.1 GB | 5.95 [6.65] GB | 9.7 GB |
| Au+Au N0 | 1,628 (2,025) | +28 % | 35.6 | 34.4 | 16.2 | 81 % | 3.58 | 106.8 GB | 5.94 [6.45] GB | 9.6 GB |
| **Au+Au N1** | **2,160** (2,846) | **+70 %** | 25.4 | 35.4 | 22.0 | 63 % | 2.09 | 105.0 GB | 5.97 [6.66] GB | 9.6 GB |
| O+O O, full outputs | 10,435 (10,965) | | 5.3 | 6.5 | 19.3 | 81 % | 0.558 | 24.2 GB | 1.39 [1.44] GB | 3.6 GB |
| O+O N0, full outputs | 11,949 (12,505) | +15 % | 4.6 | 6.0 | 20.2 | 71 % | 0.426 | 24.5 GB | 1.40 [1.44] GB | 3.6 GB |
| **O+O N1, full outputs** | **12,700** (13,384) | **+22 %** | 4.3 | 5.9 | 21.3 | 60 % | 0.342 | 24.6 GB | 1.40 [1.44] GB | 3.6 GB |
| O+O O, hadron files only (FH10_j8, 2026-10-12 15:00) | 10,313 (10,924) | | 5.3 | 7.3 | 21.2 | 79 % | 0.55 | 26.0 GB | 1.43 [1.49] GB | 3.6 GB |
| O+O N0, hadron files only | 12,722 (13,190) | +23 % | 4.4 | 6.1 | 21.8 | 74 % | 0.421 | 25.3 GB | 1.42 [1.47] GB | 3.6 GB |
| **O+O N1, hadron files only** | **13,728** (14,260) | **+33 %** | 4.0 | 6.1 | 23.5 | 65 % | 0.344 | 24.9 GB | 1.43 [1.48] GB | 3.6 GB |

- **All runs completed.** No OOM, the watchdog never fired, and every log says `host sync: block`.
  The only skipped events are Matter's known "minus log of sudakov negative" in O+O (2–5 per run).
- **No other user's process** went above 20 % CPU during the runs (sampled every 10 s, `otherusers.log`).
- **D1 is on in every N1 leg** (`late step: delta_tau 0.02 -> 0.05 fm/c at step 130 (tau = 3 fm/c),
  facTau 5 -> 2, a frame every 2 steps`) and off in N0.
- **The measured GPU savings match the GB10 estimates:** C4 −29 % (estimate −29 %); D1 −42 % on top
  (estimate −43 %).
- **Now host-bound:**
  - Au+Au N1 keeps 22 of 48 threads busy, against 14 for O.
  - O+O hadrons-only N1 is at ~24 busy threads, the physical core count, i.e. ~95 % of
    24 × 3600 / 6.1 ≈ 14,200 events/h.
- **Disk:** Au+Au 0.243 GB/event (73 GB per 300 events), O+O 28 MB/event full, 0.13 MB hadrons only.
- **Makespan understates the Au+Au N1 rate in these 15-event jobs:** the stagger ramp (108 s) is a
  fifth of the 500 s run. With 45-event jobs expect between 2,160 and 2,850 events/h, probably ~2,400–2,500.
- **The old image (A_O, 1,271)** was never measured for Au+Au before: it is +13 % over the
  `music2restart` image's 1,121 (A10/A11).
- **Host memory** is unchanged by the new image (105–108 GB at 20 Au+Au jobs, 18–21 GB left).

## 2. Au+Au physics check: hydro, deposition, wake

`wake_cmp.py` compares the pair files of two runs file by file on the production output grid
(64 × 64 × 32). It is adapted from the image's `music_cpu_gpu_hydro.py`, which needs `--native`
files. Energies are those of the ideal T^{μν} through τ = const, over fluid with e ≥ 1e-4 GeV/fm³.
`cmp/summary_auau.txt` has the full output.

**Backgrounds** (20, one per job), median / 90 % / max over backgrounds:

| | C4: O vs N0 | D1: N0 vs N1 | GPU vs CPU (MUSIC_CPU_vs_GPU.md) |
|---|---|---|---|
| energy-weighted ⟨\|Δe\|/e⟩ above e_fo, τ < 3 fm/c | 2.6e-7 / 4.4e-7 / 3.6e-6 | **0** (bit-identical) | |
| the same, τ ≥ 3 fm/c | 7.4e-7 / 1.1e-6 / 1.7e-6 | **1.1e-3 / 1.2e-3 / 1.2e-3** | 2.0–2.2e-3 |
| largest single cell \|Δe\|/e | 2.4e-3 / 2.1e-2 / 0.29 | 5.2e-3 / 6.2e-3 / 8.5e-3 | up to 3.9e-2 |
| ⟨\|Δv\|⟩, τ ≥ 3 | 2.2e-7 | 2.0e-4 | |
| fluid energy, max \|ΔE/E\| over τ ≥ 3 | 2.8e-5 / 4.1e-5 / 5.3e-5 | 4.1e-4 / 6.5e-4 / 7.2e-4 | |
| freeze-out frames | all equal | +1 frame (0.1 fm/c) in 2 of 20 | |

C4's largest single-cell values (up to 0.29) were not examined; the energy-weighted measure is
~1e-6, as on the GB10.

**Jet events** (300):

| | C4: O vs N0 | D1: N0 vs N1 |
|---|---|---|
| showers the same (droplets within 1e-2) | 284 / 300 (15 of the 16 others: A_O seed 4, §4) | **157 / 300** |
| deposited energy, same-shower events, max \|ΔE_dep\|/E_dep | 8.7e-7 | 1.0e-4 |
| **wake energy at the last frame / deposited energy**, all events | 0.9854 → 0.9838 | **0.9836 → 0.9864 (+0.0029 ± 0.0016)** |
| the same, same-shower events | 0.9842 → 0.9836 (−0.0007 ± 0.0008) | 0.9762 → 0.9812 (+0.0050 ± 0.0014) |
| same-shower events, \|ΔW\|/W per event, median / 90 % | 0.6 % / 2.0 % | 0.8 % / 3.0 % |

- **Deposition under D1 is normalised correctly.** If the liquefier had kept its starting step, every
  droplet after 3 fm/c would deposit 2.5× too much or be missed. Instead:
  - the deposited energy of same-shower events agrees to 1e-4;
  - the wake carries the deposit (ratio ~0.98 in all runs), to +0.3–0.5 %.
- **Showers diverge under D1 in half the events,** as they did between GPU and CPU (3 of 6 the same
  there). The medium differs at 1e-3, so Matter and LBT sample it differently. They are other
  showers, not a bias: the deposited energy per event is 40.54 vs 40.41 GeV.

**Wake field δe = e_jet − e_bg, cell by cell, same-shower events** (`wake_cmp.py`, `wake_scale.py`;
60 events for the smoothing):

| L2 \|δe_B − δe_A\| / \|δe_A\|, median (90 %) | C4: O vs N0 | D1: N0 vs N1 |
|---|---|---|
| τ = 4 fm/c, no smoothing | 6.8e-4 (2.5e-3) | **3.9e-2 (0.10)** |
| τ = 6, no smoothing | 1.3e-3 (1.9e-2) | **4.5e-2 (0.13)** |
| τ = 6, Gaussian σ = 0.5 fm | 5.4e-4 (1.2e-2) | 2.3e-2 (5.2e-2) |
| τ = 6, σ = 1 fm | 3.6e-4 (7.7e-3) | 1.3e-2 (3.7e-2) |
| τ = 6, σ = 2 fm | 2.6e-4 (4.6e-3) | **7.4e-3 (3.2e-2)** |
| τ = 8, no smoothing / σ = 2 fm | 1.4e-3 / 3.4e-4 | 4.6e-2 / 8.1e-3 |

| D1, same-shower events, B − A | τ = 4 | τ = 6 | τ = 8 |
|---|---|---|---|
| positive part of δe (Σ δe>0 · τ dV), relative | +0.20 % ± 0.15 % | +0.07 % ± 0.11 % | +0.04 % ± 0.08 % |
| negative part (the depletion) | −0.08 % ± 0.12 % | +0.21 % ± 0.09 % | +0.23 % ± 0.09 % |
| centroid shift x, y, η (rms over events) | 0.013, 0.018 fm, 0.007 | 0.030, 0.027 fm, 0.005 | 0.020, 0.025 fm, 0.004 |
| transverse width, relative | −0.13 % ± 0.05 % | −0.06 % ± 0.08 % | −0.02 % ± 0.04 % |

- **D1 does change the wake, but at short wavelengths.** The 4–5 % cell-by-cell difference halves
  with 0.5 fm smoothing and falls to 0.7–0.8 % at 2 fm. Up to 13–38 % of the squared difference sits
  within 1 fm of the wake's peak region, which is 0.1 % of the grid.
- **Integrated, the wake is unchanged at the 0.1–0.2 % level:** its positive and negative parts, its
  centroid (to 0.03 fm) and its width.
- **What this means:** a time discretisation of a source that moves at c and of a sharp front.
  Coarser steps redistribute δe between neighbouring cells; they don't move or reshape the wake.
- **Comparison with GPU vs CPU:** there the same measure gave 0.1–0.2 %, from 2 events on MUSIC's own
  grid. D1's cell-level wake differences are ~20× larger than that; its integrated ones are not.
- **When it matters:** for an FNO trained on cell-level wake fields, D1 is a ~5 % change of the
  target at the grid scale. A campaign should not mix D1 and constant-step files.
- **C4's tails:** C4 has a 90 % tail of 2 % at τ ≥ 6 fm/c: a few events in which float-level
  differences grow. Not examined further.

## 3. Au+Au hadron level

`hadronize.py --correlated --oversample 100 --tags bulk_jet,bulk_bg` on all 60 particlize files.
The N0 and N1 particlize files were given A_O's `file_uuid` (their own kept as `own_file_uuid`), so
all three draw the same iSS random numbers per cell. `had_wake.py` takes jet − background per event
for charged hadrons, 0.2 < pT < 4 GeV:
- **Δφ** to the leading shower initiator, |η| < 1, 16 bins;
- **pT**, |η| < 1, 12 bins;
- **η**, 20 bins;
- **N** and **E**, |η| < 1.

`had_cmp.py` / the clustered analysis give B − A per event. **Errors are clustered by background**:
the 15 events of a job share one background, so there are 20 independent units, not 300. Without
clustering the errors are too small. The first pass did that and showed a spurious 6σ pT shift for D1.

| B − A of jet − background | C4: O vs N0, same showers (284) | D1: N0 vs N1, all (300) | D1: same showers (157) |
|---|---|---|---|
| charged N, \|η\| < 1 (A: ~8.8 per event) | **+0.76 % ± 0.21 %** | +1.9 % ± 2.2 % | +1.1 % ± 2.4 % |
| charged E, \|η\| < 1 | +0.86 % ± 0.23 % | +1.6 % ± 1.4 % | +1.2 % ± 1.5 % |
| Δφ shape, χ² / bins | 48.5 / 16 | **6.3 / 16** | 8.7 / 16 |
| pT, χ² / bins | 49.8 / 12 | 17.7 / 12 | 29.6 / 12 |
| η, χ² / bins | 42.6 / 20 | 25.9 / 20 | 30.7 / 20 |
| largest bin change / signal maximum | 1.0–1.4 % | 5–15 % (within errors) | 7–22 % |

**Each leg alone** (`leg_spec.py`):
- **D1, backgrounds:** multiplicity +7.5e-5, ⟨pT⟩ +1e-4 (relative). The bulk is unchanged.
- **D1's apparent soft-pT shift** in J − B (−0.18 in the lowest bin, +0.34 in the next) comes from
  the backgrounds' fluctuation in those bins (+0.33, −0.37; each 1.5σ, shared by 15 events). It is
  not a jet-leg effect.
- **C4, same showers:** the jet leg gains +0.07 charged hadrons per event out of ~1,180 (6e-5); the
  background changes by 7e-6. It is significant only because correlated sampling cancels nearly
  all noise: 0.8 % of the wake, at 3.6σ, against the ~1 % GPU-vs-CPU agreement of the wake.

**Verdict, hadron level:**
- No D1 effect at the 2 % level in yield, energy and Δφ shape.
- The pT and η shapes have χ² of 18–31 for 12–20 bins. Errors estimated from only 20 clusters make
  χ² unreliable, and the leg-by-leg check accounts for the largest bins.
- **To pin D1 at the 0.5 % level,** ~100 backgrounds are needed, e.g. 100 jobs × 3 events instead of
  20 × 15, with the same three runs.

## 4. A failed job in the old-image campaign (A_O seed 4)

- **The symptom:** in A_O, all 15 events of seed 4 have 1–5 droplets (0–24 GeV deposited, 81 in the
  job) and 9–18 partons. N0, N1, and seed 4 run alone with either image give 13–34 droplets
  (21–66 GeV, 330 in the job) and 55–129 partons.
- **What is the same:** the background hydro (|Δe|/e 6e-7 against N0) and the initiating partons
  (same positions, same pT).
- **Not reproducible:** `rerun4/O`, seed 4 alone with the old image, is identical to N0 in E_dep and
  partons for all 15 events.
- **Rare:** a scan of `droplets_total` in every campaign job on this box since 2026-10-04 (~250 jobs:
  bins_j8, memtest R5/R6, auau_prof, music2restart, these runs) finds no other job below half its run's
  median. It hit all 15 events of one job, so it's per-job state, probably what Matter/LBT read as
  the medium. The log has no warning.
- **Consequences:**
  - The C4 comparisons above are quoted for same-shower events, or without seed 4. With it, A_O's
    wake yield is 6 % low.
  - **For productions:** a cheap guard in `run_jobs.sh` or the campaign checks would catch it. Flag a
    job whose droplets per event (`droplets_total / events_written` in its `.json`) is below half the
    campaign median, and redo it.
  - It has not been seen with the new image (40 jobs here), which says little at this rate.

## 5. O+O physics check (±10.5 fm, full outputs, 12 jobs with no skipped event in either run, 720 events)

`per_event.py` (as in the C1 check) and paired per-event differences:

| | C4: O vs N0 | D1: N0 vs N1 |
|---|---|---|
| initial state | identical in all 720 | identical in all 720 |
| τ_fo | identical in every event | +0.0107 ± 0.0012 fm/c: one frame (0.1 fm/c) later in ~11 % of events |
| freeze-out surface cells | +6e-6 ± 6e-6 (max 0.4 %) | **+0.200 % ± 0.002 %** (median 0.21 %, max 0.34 %) |
| means: τ_fo, cells, partons, jet hadrons, first-frame energy | equal within errors | equal within errors |
| grid-boundary hits | 0 | 0 |
| bulk hadrons (hadron files only, 10 oversamples, paired): all hadrons | −0.015 % ± 0.020 % | +0.048 % ± 0.033 % |
| … dN_ch/dη \|η\| < 1 | +0.30 % ± 0.16 % | +0.43 % ± 0.23 % |
| … ⟨pT⟩ charged | +0.06 % ± 0.12 % | +0.13 % ± 0.15 % |

**Hydro fields** (`oo_field.py`: every 3rd event of the aligned jobs, 240 events; energy-weighted
⟨|Δe|/e⟩ above e_fo, per event the median over frames; median / 90 % / max over events):

| | C4: O vs N0 | D1: N0 vs N1 |
|---|---|---|
| τ < 3 fm/c | 2.5e-7 / 3.7e-7 / 1.3e-3 | 0 (bit-identical) |
| τ ≥ 3 fm/c | 5.0e-7 / 7.1e-7 / 9.5e-4 | 1.2e-4 / 3.5e-4 / 6.5e-4 |
| largest single cell above e_fo | 9.9e-6 / 5.1e-4 / 1.9 | 8.8e-4 / 1.9e-3 / 3.2e-3 |

`arr_diff.py`'s maximum over all cells (`cmp/arr_F_*.txt`, up to 1–1.8 of the field maximum) is set
by the velocities of near-vacuum cells, as in the C1 check. It is not a useful measure here.

- **D1 makes the O+O surface 0.2 % larger, systematically,** and shifts τ_fo by one frame in a tenth
  of the events. The hadron yields follow within errors (+0.05 % total, +0.4 % ± 0.2 % at midrapidity).
  That is inside the 0.2–0.3 % GPU-vs-CPU level, but it is a systematic shift: don't mix D1 and
  constant-step O+O files in one analysis.
- **C4 is invisible in O+O** beyond float rounding.

## 6. Verdict

- **C4 (stall break): accept.**
  - Float-rounding level in hydro and surfaces (~1e-6), as on the GB10.
  - Wake energy and fields unchanged at the 1e-3 level in all but a few chaotic events.
  - One small hadron-level shift: +0.8 % ± 0.2 % of the Au+Au wake yield, 6e-5 of the jet leg.
  - +28 % (Au+Au) and +15–23 % (O+O) events/h.
- **D1 (late Δτ): acceptable for production by the plan's criterion,** with two caveats.
  - **Criterion met:** the medium differs from the constant step at 1.1e-3, below the GPU-vs-CPU
    2e-3; deposition is normalised correctly; the wake's energy, depletion, position and width agree
    at 0.1–0.5 %; hadron-level wake within ~2 %; O+O yields within 0.05–0.4 %.
  - **Caveat 1:** the cell-level wake field changes by ~5 % (0.8 % at 2 fm resolution), and the
    O+O surface by +0.2 %. Each campaign should use one setting throughout. The FNO training target
    changes accordingly.
  - **Caveat 2:** hadron-level precision here is ±2 % (20 backgrounds). A 100-background run would
    take it to ±0.5–1 % if that is needed.
  - +33 % (Au+Au) and +6–8 % (O+O) on top of C4.
- **Campaign defaults for this box with the new image:** unchanged launcher settings, pinned by digest.
  - Au+Au: 10/GPU, OMP 3, stagger 12; host memory ~105 GB. Expect ~2,400 events/h in long jobs.
  - O+O: 8/GPU, OMP 3; ~13,700 events/h hadrons only.
- **Next limit:** now host-bound on 24 cores. More GPUs per host CPU core (or a node with more cores per
  GPU) is what pays now; more jobs per GPU won't.
- **Add the per-job deposition sanity check** of §4 to campaign tooling.

## Files

In `bench/final_optim_20261012/`:
- `chain.sh` and the runs `A_*`, `F_*`, `FH_*`; `FH_O` is a link to `../oo_grid_20261012/FH10_j8`.
- `summary.jsonl`, and the harness copies `run_auau.sh` and `run_oo.sh`.
- `wake_cmp.py`, `cmp_summary.py`, `wake_scale.py`, `had_wake.py`, `had_cmp.py`, `leg_spec.py`,
  `run_cmp.sh`, `run_had.sh`.
- `cmp/`: the comparison outputs; `rerun4/`: seed 4 run alone with each image.
- Disk: ~300 GB (A_*: 73 GB each with hadron files; F_*: 27 GB each). 409 GB were left free on `/`
  after the runs.
