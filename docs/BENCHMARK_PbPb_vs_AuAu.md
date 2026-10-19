# Pb+Pb 5.02 TeV Z+jet against Au+Au 200 GeV jets on the GB10: time, memory, disk

Measured 2026-10-15. Question: how much longer, larger and more memory-hungry is a Pb+Pb
Z+jet job ([PLAN_ZJET_CMS10x.md](Plans/PLAN_ZJET_CMS10x.md)) than an Au+Au jet job, and
why?

**Short answer.**
- **Time: 2.1× longer.** 323 against 693 events/h (2 jobs at once); 37 against 17 core-s
  of CPU per event.
- **Memory: 2.1× more.** 13.8–16.0 GB against 6.8–7.1 GB peak per job.
- **Disk: 1.7× larger.** 9.1 against 5.3 MB per event at 40 samples, from the 1.7× hadron
  multiplicity.
- **The memory comes mostly from MUSIC's larger box.** The same Au+Au job in the Pb+Pb
  box (x, y ±20 fm, η_s ±8 instead of ±15 fm, ±6) needs 10.8 instead of 5.2 GB and 37 %
  more time ([below](#the-box-test)). That is two thirds to three quarters of Pb+Pb's
  extra memory: the job keeps the background's evolution on the whole grid in memory, and
  the grid has 2.4× the cells. The lifetime is not the cause: both systems store about
  100 frames.
- **For the campaigns:** at 14–16 GB a Pb+Pb job needs about twice what
  PLAN_ZJET_CMS10x §6 assumed (8–9 GB), so the `utils/zjet/` launchers' 6 jobs per GPU do
  not fit ([below](#for-the-campaigns)).

## Setup

GB10, 2 jobs × 48 events in parallel per system, `MUSIC_CUDA_SYNC=block`,
`OMP_NUM_THREADS=6`, `/usr/bin/time -v` per job. Code: X-SCAPE `zjet` 38cd11cf, js-contrib
`zjet` 3f85274. Both with the same layout, 24 jets per background (2 backgrounds per job, 4
in all), and the same hadron settings: `--hadrons-only --oversample 40 --n-frag 10
--correlated --charged --eta-max 4 --keep-primaries 0` (seeds 101 and 102).

| | Au+Au 200 GeV 0–10 % | Pb+Pb 5.02 TeV 0–30 % Z+jet |
|---|---|---|
| user XML, grid | `AuAu_MCGlauber_MUSIC_0_10_jet.xml`, `grid_fno.yaml` | `PbPb_0_30_ZJet.xml`, `grid_PbPb.yaml` |
| hard process | QCD 2 → 2, p̂T 10–20, 20–30, 30–40 GeV × 8 jets, `--parton-ymax 0.6` | Z + parton, p̂T 20–30 … 120–200 GeV (6 windows) × 4 jets, `--parton-ymax 1.9`, `--lepton-ptmin 20 --lepton-etamax 2.5` |
| MUSIC grid (`<IS>`) | x, y ±15 fm, η_s ±6 (100 × 100 × 60 = 0.60 M cells) | x, y ±20 fm, η_s ±8 (134 × 134 × 80 = 1.44 M cells) |
| backgrounds' N_coll (3dMCGlauber) | 848, 878, 892, 893 | 443, 487, 772, 835 |

The Pb+Pb jobs are the Z+jet test of
`contribs/PyJetscape/example/analysis/zjet/setup/` (`make_data.sh pbpb`); the Au+Au jobs:

```bash
cd contribs/PyJetscape/example/prod_AuAu_0_10_jet
export MUSIC_CUDA_SYNC=block OMP_NUM_THREADS=6
for s in 101 102; do
  /usr/bin/time -v python run_prod_jet.py --events 48 --seed $s \
    --user-xml ../config/AuAu_MCGlauber_MUSIC_0_10_jet.xml --grid ../config/grid_fno.yaml \
    --seed-registry none --pthat-bins 10-20,20-30,30-40 --jets-per-bin 8 --parton-ymax 0.6 \
    --hadrons-only --oversample 40 --n-frag 10 --correlated \
    --charged --eta-max 4 --keep-primaries 0 --outdir out > auau_seed$s.log 2>&1 &
done
wait
```

## Result

Per system: 2 jobs, 96 events. Wall, CPU and peak memory from `/usr/bin/time -v`; iSS
time and droplets from the job's `.json`; the rest from the hadron files.

| | Au+Au | Pb+Pb | Pb+Pb / Au+Au |
|---|---|---|---|
| wall per job (48 events) | 463–498 s | 910–1071 s | **2.1×** |
| time per event in the event loop | 7.3–7.8 s | 15.5–17.8 s | 2.2× |
| throughput, 2 jobs | 693 events/h | 323 events/h | **2.15× longer** |
| CPU per event | 16.7 core-s | 37.3 core-s | 2.2× |
| peak memory per job | 6.8–7.1 GB | 13.8–16.0 GB | **2.1×** |
| iSS per job | 103–113 s | 153–206 s | 1.7× |
| background frames stored | 96–105 | 97–121 | ~1.05× |
| freeze-out surface per leg | 1.11 M cells | 1.73 M cells | 1.6× |
| charged hadrons per sample (\|η\| < 4) | 4,290 | 7,390 | 1.7× |
| disk per event (40 samples, `x` kept) | 5.30 MB | 9.09 MB | **1.7×** |
| liquefier droplets per event | 22 | 50 | 2.3× |

- **Disk follows the multiplicity**: hadrons per sample and MB per event are both 1.7×.
  `bulk_jet` is 95 % of it: one full event (bulk + wake) per jet event and sample, against
  one per background in `bulk_bg`. These files keep the freeze-out positions at full
  precision; campaigns use `--no-x` and fewer momentum bits.
- **Time and memory grow faster than the multiplicity** (2.1–2.2×): the hydro runs on 2.4×
  the cells, the surfaces are 1.6× larger, and the Z+jet showers at p̂T up to 200 GeV
  deposit 2.3× the droplets.
- The Pb+Pb backgrounds are 0–30 % and have half the Au+Au backgrounds' N_coll in two of
  four cases; the two central ones (772, 835) gave 1.6× the hadrons of the other two.
  **0–10 % Pb+Pb would cost more** than these numbers.

## The box test

Is the memory the box? The same Au+Au job (seed 101, 1 background × 4 jets, p̂T 20–30 GeV,
the hadron settings above, 2 jobs at once), once with its own `<IS>` grid and once with
only `grid_max_x`/`grid_max_y` 15 → 20 and `grid_max_z` 6 → 8, Pb+Pb's box:

| | peak memory | time (job's `wall_s`) | background frames | droplets |
|---|---|---|---|---|
| Au+Au box (100 × 100 × 60) | 5.2 GB | 54.5 s | 101 | 74 |
| Pb+Pb box (134 × 134 × 80) | 10.8 GB | 74.5 s | 101 | 74 |

Same event, same frames and droplets: **the box alone adds 5.6 GB and 37 % time.** With
`output_evolution_to_memory`, the background's evolution stays in memory for the jet's
energy loss to read: MUSIC's own store (8 floats, 32 B per cell and frame) and the slim
copy Matter and LBT read (6 floats, 24 B; [PLAN_slim_bulk_info.md](Plans/PLAN_slim_bulk_info.md)).
Every frame covers the whole grid:

- Au+Au: 0.60 M cells × ~100 frames × 56 B ≈ 3.4 GB;
- Pb+Pb: 1.44 M cells × ~100–120 frames × 56 B ≈ 8–10 GB.

The rest of Pb+Pb's extra memory (~2–3 GB of the 7–9 GB) is the event: MUSIC's working
arrays for the two legs on 2.4× the cells (on the GB10 the GPU shares the system memory;
whether its allocations count in the peak above is not checked), the 1.6× larger surfaces
in C++ and numpy, and the harder jets.

## For the campaigns

- **Jobs per GPU:** `zjet_3090_campaign.sh` and `zjet_slurm_campaign.sh` assumed 8–9 GB per
  job (6 jobs per GPU). At 14–16 GB they now default to 3 per 3090 (6 jobs, ~96 GB of the
  box's 125; 4 per GPU = 112–128 GB only if the box's peaks are lower) and `P=3` per A40
  share (58 GiB). On discrete GPUs the hydro's device arrays are not host memory, so the host
  peak there may be lower than the GB10's: measure it in the benchmark of PLAN_ZJET_CMS10x
  §5a step 10 and set `JOBS_PER_GPU` and `P` from it.
- **Time:** the plan's 1.8–2× of the Au+Au cost per event matches the measured 2.1–2.2×; the
  memory limit is new. With the box's Au+Au rates (`cu124-final_optim`): ~500–720 Pb+Pb
  events/h at 3 jobs per GPU, ~16 h per 9,600-event campaign, 6–8 days for the ten
  (PLAN_ZJET_CMS10x §6, estimates).

**Ideas to cut the memory (not tried):**

- store the background evolution only for the η_s range the jets reach (partons |y| < 1.9:
  |η_s| ≲ 4 instead of ±8 would about halve it), keeping the hydro grid itself (the Pb+Pb
  fluid reaches η_s ≈ ±8);
- store each frame's bounding box of cells above freeze-out instead of the whole grid
  (late frames are mostly empty in x and y);
- store fewer frames or 16-bit floats.

The first two leave the medium the jets see unchanged inside what is stored; the last
changes it and needs a physics check.
