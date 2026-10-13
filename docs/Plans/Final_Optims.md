# Final optimizations: what is left for the O+O and Au+Au jet productions (2026-10-12)

A last look for speed-ups of `prod_OO_isr` and `prod_AuAu_0_10_jet` after A1–A11, the MUSIC_2
restart (A8), the O+O grid (C1) and iSS Part C: in MUSIC4GPU, in what the host still does, and
in how the productions should be run on GPU cluster nodes (WSU Grid, ACCESS). Numbers are from
[REPORT_CPU_GPU.md](REPORT_CPU_GPU.md) §8, [BENCHMARK_2x3090.md](../BENCHMARK_2x3090.md),
[BENCHMARK_GB10.md](../BENCHMARK_GB10.md), and three new measurements made for this note on the
GB10 (§2, §3.2, [Method](#method-and-files)): the first kernel-level profile at the production
grid, the fraction of MUSIC's grid that holds matter, and a check of the compiled kernels.

## Summary

- **Every production is now at or near its GPU limit, on every machine measured.** 2 × RTX 3090:
  Au+Au two-stage campaign 1,121 events/h (91 % of the limit at 90 % GPU busy), particlize-only
  1,237 (98 %), O+O ±10.5 fm 10,629 (91 %). GB10: campaign 381 events/h at 89 % GPU busy. So the
  CPU-side work of the last weeks has done its job: **more events/h now need less GPU work per
  event.** The host comes back as the limit in two cases: on nodes with few or slow cores per
  GPU (§6), and on every machine once the GPU work drops by 2× or more (§7).
- **Where the GPU time goes** (new, GB10, 100 × 100 × 60): 14.05 ms of kernels per step, i.e.
  7.0 ms per RK sub-step. One kernel, `gpu_make_delta_qi` (the KT fluxes of the ideal part with
  12 reconstructions per cell), is **52 %** of it; the viscous `gpu_first_rk_step_w_full` 21 %,
  the source stencil 10 %; packing, gathering and reductions < 1 %. The kernels carry no
  FP64 instructions. Where `delta_qi`'s time went was found by trying (§3 C4): not in its
  arithmetic, but in solver lanes that never converge — the Newton–Brent tolerance of 1e-7 is
  below the float32 ulp of u⁰ > 1.7, so such lanes ran all 60 iterations and held their warps.
  **Stopping at float resolution (MUSIC4GPU branch `reconst_stall`) takes `delta_qi` from 3.65 to
  1.71 ms and the kernels per step from 14.05 to 10.1 ms (−29 %), at float-rounding-level
  differences.** One job alone keeps the GPU busy 50–60 % of a MUSIC run (kernels 14 of 23–29 ms
  per step); the rest of a step is the freeze-out search (4.3 ms per step on average), the
  source fills and ~2–3 ms of launches, syncs and bookkeeping.
- **Three GPU-side changes, together ~5× less GPU time per event (C4 measured, D1 and C3 estimates):**
  - **D1, a larger Δτ after the early phase** (0.02 → 0.05 fm/c from τ ≈ 3 fm/c, frames still every
    0.1 fm/c): −40 to −50 % of the steps. MUSIC has this as `beastMode 2`; it is unusable with a
    jet source only because the liquefier keeps the step width it was given at the start. The
    pruning window already follows a changed step; the liquefier's normalisation does not (a
    3-line fix). Not bit-identical: a physics check like C1's. **Cheapest, and the largest
    certain gain: events/h ×1.5–1.8 on GPU-bound machines with host headroom (GB10, A100 and
    GH200 nodes); the 3090 box only up to its host limit, ~1,400 (+25 %).**
  - **C3, evolve only the region that holds matter.** Measured: in 0–10 % Au+Au, cells with
    e > 10⁻⁴ GeV/fm³ are 17 % of MUSIC's grid at τ = 0.5 fm/c and 38 % at τ = 10 (time mean ~30 %);
    in O+O 3–5 % of the ±15 fm grid, ~10 % of ±10.5 fm. Kernels over a compacted cell list or
    per-η-slice boxes: Au+Au ×0.35–0.45 of the GPU time per step, O+O ×0.15–0.25. Large
    (CUDA and Metal), not bit-identical at the vacuum level only.
  - **C4, `gpu_make_delta_qi` — done (branch `reconst_stall`, measured):** the kernel's time
    was not its redundant arithmetic (precomputing τT^{τμ} per cell, one table index per solver
    iteration, a warm start: all measured, none gained more than 4 %, and the first two are
    not bit-identical because the compiler fuses the stored products differently) but solver
    lanes stalling at float resolution. One extra stop condition per solver: `delta_qi` −53 %,
    **the step −29 %**. Not bit-identical: float-rounding-level differences (energy-weighted
    |Δe|/e ~1e-6, surfaces ±2 elements), to be validated like C1 before production.
- **Host → GPU (B)** matters on core-poor nodes now, and everywhere after D1/C3. What the host
  still does per jet event: the freeze-out search of both legs (~22 ms per check, 75 % of it
  Cornelius' geometry, which allocates per cube: **A12**, bit-identical, small), the droplet
  deposition (~1 s wall per jet leg: **B1′**, a kernel over the active droplets), the pair file's
  compression (6 core-s; `--particlize-only` or `blosc-lz4`), and 3–5 core-s of launches, syncs
  and copies.
- **Structural (E1):** evolving each jet leg only in a moving sub-box around its wake, with the
  background as boundary, is the only route to a jet leg much cheaper than a full hydro
  (×4–6 less GPU per campaign event, estimate). Research-level, 1–2 months, listed for
  completeness (§5).
- **Production environment (§6):** a node is GPU-bound when cores per GPU × core speed exceeds
  about 10 Zen 2 cores per RTX 3090-class GPU for the Au+Au campaign (today's balance), fewer on
  slower GPUs (V100, A100), more on faster ones (H100, GH200, B200). With the known node types:
  A100 and GH200 nodes (Delta, Anvil, DeltaAI, Vista) and the WSU V100 nodes are GPU-bound,
  so D1/C4/C3 set their rate; Bridges-2's 8-GPU nodes (5 cores per V100, ~12 per H100) are
  balanced or host-bound, so there B/A12 and `--particlize-only` pay first. A calibration
  recipe (one job, `MUSIC_PROFILE=1`) and the launch settings per node type are in §6.
- **Recommended order:** D1 → C4 → C3 for the GPU side (all three before B2 on GPU-bound
  machines); A12 and `--particlize-only` any time; B1′ and B2 when a host-bound node is the
  target. Not an option: a smaller Au+Au MUSIC grid — the FNO output grid already reaches
  ±12.4 fm (`grid_fno.yaml`), 2.5 fm inside MUSIC's ±15 fm. Not worth it: FP16,
  a multi-GPU single grid, CUDA graphs, more jobs per GPU on the 3090 box, B3.

## 1. Where things stand

Campaign setting (`--pthat-bins 10-20,20-30,30-40 --jets-per-bin 5 --parton-ymax 0.6
--write-particlize both`), the best launcher setting per machine; O+O with the ±10.5 fm grid.

| production, machine | events/h | GPU-s / event | core-s / event (one job alone → under load) | GPU limit at 90 % busy | of the limit |
|---|---|---|---|---|---|
| Au+Au two-stage campaign, 2 × RTX 3090, 10 jobs/GPU | **1,121** | 5.29 | 33.9 → 62.3 | ~1,225 | 91 % |
| … `--particlize-only` | **1,237** | 5.16 | 29.4 → 50.1 | ~1,255 | 98 % |
| Au+Au hydro only, 12 jobs/GPU | 1,601 | 3.95 | 29.4 → 40.4 | ~1,640 | 98 % |
| O+O ISR ±10.5 fm, 8 jobs/GPU, default outputs (A10 + A11) | **10,629** | 0.553 | ~5 (est.) → 6.4 | ~11,700 | 91 % |
| Au+Au two-stage campaign, GB10, 4 jobs, OMP 5 | **381** | ~7.2 by kernel time, 8.4 by SM busy (this note) | ~26 under load | ~385–450 | 89 % busy |

- GPU-s per event = mean SM busy × GPUs × 3600 / events/h; the GB10 value is 477 jet-leg steps ×
  14.05 ms + the background's 479 steps / 15 = 7.2 s (§2); SM-busy counts the gaps between dependent
  kernels too.
- The Au+Au ±15 fm grid, the O+O ±10.5 fm grid (the XML default since 2026-10-12), Δτ = 0.02 fm/c, frames every 0.1 fm/c, both
  freeze-out surfaces, MUSIC_2 from MUSIC_1's checkpoint at step 28 (τ = 0.96 fm/c).
- `hadronize.py` (stage 2, CPU nodes) is 8.8–10.6 s of one core per event for both legs at 500
  oversamples: 3 cores keep up with 1,100 events/h. It is not a factor.

## 2. What the GPU does per step: the first kernel profile at the production grid

`nsys` works on the GB10 for CUDA kernel tracing (CPU sampling does not: `perf_event_paranoid` 4;
`ncu` needs `NVreg_RestrictProfilingToAdminUsers=0`, see [Method](#method-and-files)). One
two-stage job, 2 events, `--reuse 2`, `--write-particlize both`, `MUSIC_PROFILE=1`, OMP 4,
`MUSIC_CUDA_SYNC=block`, build `build_gpu` (MUSIC4GPU `81f38d9`, A10/A11 in). MUSIC_1 479 steps
(τ 0.424 → 10.0 fm/c), MUSIC_2 501 and 456 steps from step 28. 1,436 steps, 2,872 sub-steps.

| kernel | share of GPU time | per launch (GB10) | what it does | registers / stack / local-memory instr. / MUFU |
|---|---|---|---|---|
| `gpu_make_delta_qi` | **52.0 %** | 3.65 ms | KT fluxes of the ideal part: τT^{τμ} of the cell and 4 neighbours in 3 directions, minmod, 12 reconstructions (Newton–Brent), 12 max-speed evaluations | 95 / 112 B / 82 / 381 |
| `gpu_first_rk_step_w_full` | 21.2 % | 1.49 ms | the 14 W^{μν} + Π relaxation equations | 64 / 168 B / 51 / 84 |
| `gpu_make_w_source_tiled` | 10.5 % | 0.74 ms | the viscous source stencil (shared-memory tile) | 48 / 208 B / 54 / 13 |
| `gpu_finalize_ideal` | 5.6 % | 0.39 ms | e, u from the updated q (+ the source buffer) | 40 / 0 / 0 / 68 |
| `gpu_make_du` | 5.4 % | 0.38 ms | ∂u, θ, a^μ, σ^{μν} | 40 / 0 / 0 / 16 |
| `gpu_make_uwrhs`, `gpu_make_uprhs` | 4.7 % | 0.25 + 0.08 ms | advection of W, Π | 72 / 24 B; 43 / 0 |
| `gpu_pack_evolution_ideal` (every 5 steps) | 0.5 % | 0.37 ms | the stored frame | |
| `gpu_gather_cells` (A11, every 5 steps), reductions | 0.1 % | 0.09 ms | | |
| **sum** | 20.2 s | **14.05 ms per step, 7.0 per sub-step** | | |

- **No FP64:** `cuobjdump --dump-sass` of `libmusic.so` (sm_121) finds no DADD/DMUL/DFMA/F2F.F64 in
  any kernel. The literal-promotion worry of `MUSIC_CPU_vs_GPU.md` does not apply to the current
  code.
- **Local memory:** three kernels keep arrays on the stack (`delta_qi`: the 12 `ReconstResult`s
  and the α-indexed face states; `first_rk_step_w_full`: the 14-component arrays). At 95
  registers `delta_qi` can hold at most ~680 threads per SM (a third of the maximum), and it is
  the kernel that would profit most from latency hiding: its 381 MUFU instructions per cell are
  `logf` (EOS lookups, one per `gpu_P`/`gpu_dPde` call), `sqrtf` and reciprocals.
- **The RTX 3090 does the same step in ~8.4 ms** (4.2 ms per sub-step, REPORT §4); the ratio
  to the GB10, 1.67, is the ratio of the two GPUs' FP32 lanes (10,496 : 6,144), as expected for
  compute-bound kernels.

**The step from the host's side** (MUSIC's timers, the same job, one job alone):

| per step | MUSIC_1 (background) | MUSIC_2 (jet leg) |
|---|---|---|
| `evolve.step_total` | 28.9 ms | 22.4–23.4 ms |
| GPU kernels (above) | 14.05 ms | 14.05 ms |
| freeze-out, per check every 5 steps: Cornelius find / gather / scan / ε copy | 21.7 ms = 16.4 / 2.8 / 1.7 / 0.5 → 4.3 ms per step | 21.6 ms → 4.3 ms per step |
| source fill (`prefill_hydro_source`, OpenMP 4) | 4 string fills × 876 ms → 7.3 ms per step averaged | droplets: 158 of 501 steps × 6.6 ms → 2.1 ms per step |
| frame pack + store (every 5 steps) | 2.2 ms → 0.45 ms per step | 0.45 ms per step |
| launches, syncs, source preparation, bookkeeping (remainder) | ~2.8 ms | ~1.5–2.5 ms |

- Event 2 of the job (a jet event on the reused background): 13.4 s wall, of which MUSIC_2
  10.2 s and its kernels 6.4 s: **the GPU is busy 48 % of a jet event in a job alone**, so 2–3
  jobs per GPU would saturate it if their host work did not collide; in practice 4–8 do.
- These host items are what §4 is about; the kernels are what §3 is about.

## 3. GPU side: less GPU work per event

### D1. A larger time step after the early phase (not bit-identical; physics check)

**Facts.**
- The productions run `beastMode 1` (float, constant Δτ = 0.02 fm/c). MUSIC's `beastMode 2`
  doubles Δτ to 0.04 at step 200 (τ = 4.4 fm/c) and halves `facTau` and the frame cadence
  (`evolve.cpp` ~line 613). The production XML says it is for "soft sector only runs, otherwise
  the change in stepsize will lead to errors".
- The reason is the jet source, not the hydro: `CausalLiquefier` deposits a droplet in the step
  whose window τ ± Δτ/2 contains τ_drop + τ_delay and divides the kernel by Δτ
  (`CausalLiquefier.cc` lines 126–143). It gets Δτ once, at MUSIC's start
  (`MusicWrapper.cc` ~line 443 → `HydroSourceJETSCAPE::set_hydro_dtau`). After a step change the
  window is too narrow (droplets are missed) and the rate is off by the step ratio.
  `HydroSourceJETSCAPE::prepare_list_for_current_tau_frame` already notices a changed step and
  follows it for the pruning window (`MusicWrapper.h` line 82); it only has to pass the new step
  on through `set_hydro_dtau` (which also updates the liquefier). The freeze-out (`DTAU =
  facTau × delta_tau` per check), the checkpoint key (the switch comes after step 28; today the
  key rejects `beastMode 2` past step 200 and would be generalised) and Matter/LBT's lookups
  (the 0.1 fm/c frames) are unaffected.
- **Stability.** For the KT scheme the relevant number is Δτ · Σ_i a/Δx_i with Δx_η = τ Δη and
  a ≤ 1: the production's 0.02 at τ₀ = 0.424 gives 0.37; MUSIC's default `dtaudxRatio` 0.1
  (Δτ = 0.03 at this dx) gives 0.55 at τ₀; `beastMode 2`'s 0.04 at τ = 4.4 gives 0.31; **0.05
  at τ = 3.0 gives 0.42, at τ = 2.4 0.44.** So Δτ = 0.05 from τ ≈ 3 fm/c is no bolder than the
  production's own early-time step. Accuracy is a separate question: the time error is second
  order and far below the spatial one at dx = 0.3 fm, but that has to be shown, not assumed.
- **Why 0.05 and not 0.04:** with `output_evolution_every_N_timesteps` and `facTau` halved from 5
  to 2, Δτ = 0.05 keeps the stored frames and the freeze-out checks at 0.1 fm/c. The pair files,
  the FNO format and the writer are untouched. 0.04 would give 0.08 fm/c frames.

**Design.** Two MUSIC parameters, `late_delta_tau` (0.05) and `late_tau_switch` (3.0 fm/c), in
place of `beastMode 2`'s constants; at the switch, `facTau` and the frame cadence set so that
their product stays 0.1 fm/c; a callback to the sources with the new step (X-SCAPE:
`HydroSourceJETSCAPE` → `set_hydro_dtau`); the checkpoint key records the switch parameters; the
Metal path needs nothing (host-side logic). `run_prod_jet.py` gets `--late-dtau 0.05 --late-tau 3`.

**Gain.** Steps per jet leg (from step 28, freeze-out at 10–11 fm/c): ~477 today → ~250 with the
switch at 3.0 fm/c (−47 %), ~235 at 2.4 (−51 %), ~295 at 4.4 (−38 %). Background 479 → 270.
GPU-s per campaign event ~5.3 → ~2.8 on the 3090, ~7.2 → ~3.8 on the GB10. On GPU-bound machines
**events/h ×1.5–1.8** where the host has headroom: the GB10 from 381 toward ~750–800, Delta's
A100 nodes ×1.8 (§6); the 3090 box only from 1,121 to its host limit of ~1,400 with the pair file
(~1,700 with `--particlize-only`), because its 24 cores are then all busy.

**Validation** (the C1 and GPU-vs-CPU recipes, `music_cpu_gpu_hydro.py` / `_hadrons.py`): the same
events with constant and late Δτ: fields after the switch (|Δe|/e, |Δv|), the surfaces, τ_fo,
dN/dη, ⟨pT⟩, v₂, v₃, the wake per deposited energy and the hadron-level wake with correlated
sampling, over ~100 events; a scan of the switch time (2.4, 3, 4.4) and of Δτ (0.04 with
0.08 fm/c frames as a cross-check). Acceptance: differences at or below the GPU-vs-CPU level
(0.2–0.3 %), no boundary hits, no non-finite resets.

**Effort.** MUSIC4GPU 1–2 days, X-SCAPE ½ day, js-contrib ½ day, validation 2–3 days.

### C3. Evolve only where there is matter (not bit-identical at the vacuum level; large)

**Facts** (from the existing pair files, `extent2.py` in [Method](#method-and-files)):

| | Au+Au 0–10 %, ±15 fm grid (1 event, output box ±10 fm) | O+O, ±15 fm (3 events) | O+O on ±10.5 fm |
|---|---|---|---|
| cells with e > 10⁻⁴ GeV/fm³, share of MUSIC's grid | 17 % at τ = 0.5, 23 % at 4.5, 30 % at 6.5, 38 % at 10 fm/c; time mean 26 % (the box misses the dilute fluid beyond 10 fm late: ~30–35 %) | 2.6–4.0 % at τ₀, 4.7–7.2 % at the end; time mean 3.4–5.1 % | ×2.04: ~7–10 % |
| extent of e > e_fo (0.234 GeV/fm³) | \|x\|, \|y\| ≤ 7.5 fm, \|η\| ≤ 5 → 4.7 fm, 1.6 at the end | \|x\| ≤ 3.6 fm, \|η\| up to 4.8 | |
| extent of e > 10⁻³ GeV/fm³ | reaches the ±10 fm box edge from τ ≈ 5 fm/c (e at the edge up to 0.085 GeV/fm³ at τ = 9.5) | \|x\| ≤ 5.2 fm, \|η\| to the edge | |

So two thirds of the Au+Au grid and ~95 % of the O+O grid are vacuum at every step, and every
kernel visits every cell (the only early exits are the bounds checks).

**Design.** Keep a list of active cells (e > ε_act, say 10⁻⁵ GeV/fm³, plus a halo of 3 cells),
rebuilt every freeze-out check from the ε copy the host already receives (A11) or by a GPU
compaction kernel; the 7 RK kernels, the source add and the pack take the list (1-D launches
over the list; the stencils read neighbours through the full grid as now). Cells outside stay
at their values (vacuum). Per-η-slice bounding boxes are the simpler variant (×1.3 more cells
than the list for a cylinder; for O+O the η range is the full grid). The halo covers the
fluid's motion (≤ 1 cell per 15 steps) and the sound front; the numerical spreading of
vacuum-level changes is below ε_act by construction. The freeze-out scan can be restricted to
the same region (same result, cheaper scan). CUDA and Metal.

**Gain.** Au+Au: GPU time per step ×0.35–0.45 (list) or ×0.45–0.55 (boxes); O+O ×0.15–0.25.
Events/h ×2.2–2.8 (Au+Au) where the GPU is the limit; for O+O the GPU stops being the limit on
any machine (the host's 6.4 core-s per event then sets ~13,500 events/h on the 3090 box).

**Not bit-identical**, but only through the vacuum: the current code evolves vacuum cells with
fluxes at the 10⁻¹² level and the vacuum reset. Acceptance as for C1: fields within float
rounding of the field maximum, surfaces bit-identical in all but a few near-vacuum cells,
observables and the wake unchanged within errors.

**Effort.** 1–2 weeks (both back-ends), plus the validation.

### C4. `gpu_make_delta_qi`: done — the solvers stalled at float resolution (measured)

**What was assumed** (and written in the first version of this note): per cell and sub-step
the kernel evaluates τT^{τμ} 75 times (60 of them the neighbours' values), reconstructs every
face twice and looks P(e) up five times per state, so computing each once should give
−25–35 % per step, bit-identically. **What was measured** (GB10, the 2-event two-stage job of
§2, `nsys`; MUSIC4GPU `README_CUDA.md` Phase 7):

| change | `delta_qi` per launch | bit-identical |
|---|---:|---|
| baseline | 3.65 ms | — |
| q = τT^{τμ} precomputed per cell (75 → 0 evaluations) | 3.51 + 0.16 ms for the new kernel | no: the compiler fuses the stored product differently (FMA contraction), ulp-level |
| P and dP/de from one table index per solver iteration | ~3.5 ms | no, the same way |
| P(e) once per reconstructed state | 3.56 ms | yes |
| Newton started at the previous velocity instead of 0.5 | 3.51 ms | no |
| the CPU's narrow bracket [v₀ ± 0.05] | 5.48 ms (+50 %) | no; the Brent safeguard bisects a narrow bracket |
| **stop when the iterate no longer changes in float** | **1.71 ms (−53 %)** | no, float-rounding level |

The arithmetic was never the cost. Both Newton–Brent solvers break on an absolute step below
1e-7, which is below the float32 ulp of u⁰ above ~1.7 and of v near 1; such lanes ran all 60
iterations and their warps waited. One extra stop condition per solver (`v_root == v_old`,
`u0_root == u0_old`: the root to float resolution) is the whole change, on CUDA and Metal.

**Gain:** kernels per step 14.05 → 10.1 ms on the GB10 (−29 %; `finalize_ideal` profits too), a
MUSIC step 28.9 → 24.3 ms (background) and 23.4 → 19.7 ms (jet leg) in a job alone, a jet
event 13.4 → 11.2 s. The same fraction should apply on every GPU. Events/h where the GPU is
the limit: +40 % (3090 box: the campaign's GPU limit ~1,225 → ~1,700, above its host limit of
~1,400, so there +25 % and then host-bound; GB10 381 → ~500).

**Not bit-identical,** by construction: a stalled lane returns the iterate at which it
stalled, a few ulp from where 60 iterations left it. Over a run (2 Au+Au events, both legs;
2 O+O events) the fields differ at the float-rounding level: energy-weighted |Δe|/e above
e_fo ~1e-6, the largest single-cell |Δe|/e above e_fo 8e-4 at τ ≈ 3–5 fm/c (~50 of 23,000
cells beyond 1e-4, none late), total energy within 2e-6, surfaces ±2 elements per leg, the
in-job jet hadrons identical. ~1000× below the GPU-vs-CPU difference. **Before production:**
the C1-style check (a few hundred events on the 3090 box with the new image: hydro and
surfaces at this level, hadron observables and the wake within errors), and the Metal build
run on the M3 Max.

**Dropped from the list:** the precomputed q and the face-based reconstruction — the first
measured, the second would not be bit-identical either (each cell's reconstruction uses its
own (e, u) as fallback) and the solver is now cheap. `ncu` on the GB10 would still need
`NVreg_RestrictProfilingToAdminUsers=0`.

### The Au+Au MUSIC grid cannot shrink (unlike O+O's)

MUSIC's own grid is set in the XML (`<IS><grid_max_x/y> 15`, `<grid_step> 0.3`: 100 × 100 × 60
cells, as the profiled job logged). The FNO output grid (`prod_AuAu_0_10/grid_fno.yaml`, the
default of `run_prod_jet.py --grid`) does not change it: it is the resampling grid of the stored
frames, and the driver only checks that it lies inside MUSIC's grid. Its cell centres are at
±12.21 fm (64 × 0.3875 fm, cell edges at ±12.4 fm), only 2.5 fm inside MUSIC's ±15 fm boundary,
so it is a lower bound on MUSIC's grid. The dilute fluid (10⁻³–10⁻¹ GeV/fm³) already reaches the old ±10 fm
output edge from τ ≈ 5 fm/c (e up to 0.085 GeV/fm³ there at τ = 9.5), so a MUSIC boundary any
closer to the output edge would show in the stored fields through the clamped stencil at the
boundary. The matter above e_fo stays within 7.5 fm, which is why O+O could go to ±10.5 fm (C1)
and Au+Au cannot: the only way not to pay for Au+Au's vacuum cells is C3.

### Not worth doing on the GPU side

- **FP16/BF16 kernels:** the kernels are compute-bound with FP32 `logf`/`sqrtf`, not bandwidth-bound; accuracy risk for nothing.
- **A single grid across several GPUs:** one event per GPU scales better (README_CUDA, Outlook).
- **CUDA graphs:** 14 launches per step, ~0.1 ms of launch overhead per step; the host's 2–3 ms
  remainder per step is syncs and source preparation, not launches.
- **The FP64 opt-in build** (`MUSIC_CPU_vs_GPU.md`): for validation only; on consumer and L-class
  GPUs (1 : 64) it would be ~10× slower.

## 4. Host → GPU (B), and the bit-identical leftovers (A)

What a campaign jet event still costs the host after A1–A11 (one job alone, 3090: 33.9 core-s
with A1–A8, ~31 with A10/A11; under load ×1.8). Approximate split from §2 and REPORT §2:

| item | core-s per jet event (3090, alone) | change | bit-identical | effort |
|---|---|---|---|---|
| freeze-out search, both legs (~200 checks × 22 ms, OpenMP over η) | 8–12 | **A12:** Cornelius allocates per cube, line and polygon (`cornelius.cpp`: `new double[DIM]` at lines 67, 175–180, 284, 377, 493, 522–531, 578–581, inside the search); per-thread scratch instead. The geometry is 75 % of a check (16.4 of 21.7 ms). | yes | small |
| | | **B2:** the search on the GPU (one thread per crossing cube; the host keeps the element order deterministic). Removes 8–12 core-s and the ε copy. | no (the GPU's cube geometry in float, element order) | large |
| pair-file compression (zstd 3 + shuffle) | ~6 | `--particlize-only` when the history is not needed (already the case for wake analyses); else `blosc-lz4` (README_h5_optim: ~−60 % compression time, +30 % size) | yes | none |
| droplet deposition (jet leg, ~160 fills × 6.6 ms wall on 4 threads) | 4–5 | **B1′:** a kernel over the active droplets (tens per step), binned by reach as on the CPU; the string fills (B1) the same way, but they are amortised over 15 jets in the campaign (~1 core-s per jet event; 14 per hydro-only event) | no (erf/exp on the GPU) | moderate |
| writer: read MUSIC's frames, resample (A2) | ~3 | B3 (resample on the GPU) is not worth it after A2; reading the frames is a copy | – | – |
| launches and `cudaStreamSynchronize` (block), source preparation per step, pack/store, `bulk_info` copy, per-event init | 3–5 | the remainder of §2; an asynchronous freeze-out (the search of check k on the host while the GPU runs steps k+1…k+4; A10's two buffers make it possible) hides host time in a job alone but adds nothing when several jobs share the GPU | yes | moderate |
| strings (background, per 15 jets) | ~1 | B1 | no | large |

On a GPU-bound machine none of these raises events/h; on a host-bound node (§6) they pay one for
one, and after D1/C3 they are the next limit everywhere (§7).

## 5. Structural: the jet leg in a moving sub-box (E1)

The jet leg is a full hydro evolution per jet (~75–80 % of the GPU time per campaign event),
although it differs from the background only where the wake has reached: a region around the jet
path growing at the speed of sound (radius ≈ 0.57 (τ − 1) fm, η extent ≈ ±0.57 ln τ), of order
10–20 % of the fireball's active cells, 2–5 % of the grid.

**Sketch.** The background (MUSIC_1) and the N jet legs of one background step together; each leg
owns a box (the union of its droplets' causal regions plus a margin) and takes its boundary
values from the background's current snapshot each sub-step; outside the box the leg *is* the
background (its surface elements and stored frames are the background's, which the shared
background storage and the correlated sampling already assume). GPU per jet event ≈ (full
grid)/N + box ≈ 0.15–0.25 of today's jet leg: **×4–6 less GPU per campaign event** (estimate), on
top of D1/C3/C4 where it does not overlap with C3 (it overlaps largely: both remove the vacuum;
E1 also removes the unperturbed fireball).

**Costs and risks.** A multi-leg mode in MUSIC (lockstep evolves, boundary exchange, seams of the
freeze-out cubes at the box edge), the driver's event loop (N jets prepared before the hydro
runs), the pair writer (box pasted into the background frame). The box must contain every cell
whose difference from the background exceeds a tolerance; float-level differences spread
faster than sound, so the acceptance is "below a tolerance outside the box", not bit-identity.
1–2 months plus validation. Worth it only if the campaign needs another 3–5× after §3; not
before D1/C3/C4.

## 6. Production environment: which side is the limit on a given node

**The rule.** Per GPU, the host can deliver 3600 · (cores per GPU) · (core speed) / (core-s per
event under load) events/h, the GPU 0.9 · 3600 / (GPU-s per event). Today's Au+Au campaign:
62 core-s per event under load on the 3090 box's Zen 2 cores (34 alone), 5.3 GPU-s on an RTX
3090. **A node is GPU-bound when it has more than ~10 Zen 2-equivalent cores per RTX 3090-class
GPU**; a V100 needs only ~5, an H100 ~20. After D1 the GPU numbers shrink by ~45 %, after D1 +
C3 + C4 by ~85 %: the thresholds then rise to ~20 and ~70 cores per 3090-class GPU, i.e. every
node becomes host-bound and §4 sets the rate.

**GPU speed for these kernels** (FP32, compute-bound; the measured points are the RTX 3090 and
the GB10, the rest scales with FP32 peak rate, ±30 %):

| GPU | relative to RTX 3090 | GPU-s per campaign event today (D1; D1 + C3 + C4) | image |
|---|---|---|---|
| RTX 3090 / A40 | 1.0 (measured) | 5.3 (2.9; 0.8) | cu124 / cu126 |
| GB10 | 0.6 (measured) | 7.5 (4.0; 1.1) | cu130 (sm_121) |
| V100 | ~0.45 | ~12 (6.5; 1.8) | needs sm_70: cu124, or cu126 after the pending rebuild |
| A100 | ~0.6 | ~9 (4.9; 1.3) | cu126 |
| L40S / RTX 6000 Ada | ~1.5–2 | ~3 (1.6; 0.45) | cu126 |
| H100 / GH200 | ~1.8 | ~3 (1.6; 0.45) | cu126 (H100), cu130 arm64 (GH200) |
| B200 / GB200 | ~2+ | ~2.5 | cu130 (sm_100) |

**Node types** (specifications from the ACCESS resource pages and a WSU computing overview; verify
on the machine with `sinfo -o "%P %G %c %m"` before planning; cores are physical):

| node | GPUs, cores per GPU, host memory per GPU | Au+Au campaign limit today: host / GPU (events/h per GPU) | regime now → after D1 → after D1 + C3 + C4 | what to do there |
|---|---|---|---|---|
| WSU Grid, V100 (about ten on the shared queue; the EPYC 7643 nodes have 96 cores) | V100; 24–48 cores per GPU if hosted there (verify) | ~1,600–3,200 / ~270 | GPU → GPU → balanced | D1, C4, C3; 4–6 jobs per GPU |
| Bridges-2 GPU (PSC) | 8 × V100-32 GB, 5 Xeon 6248 cores, 64 GB | ~290 / ~270 | balanced → host → host | `--particlize-only`, A12, B1′, B2; 4–5 jobs per GPU with 1–2 threads; or run 4 GPUs' worth of jobs per node |
| Bridges-2 H100 nodes | 8 × H100-80 GB, ~12 cores per GPU (verify) | ~900 / ~1,100 | host → host | as above; the host sets ~900 per GPU |
| Expanse GPU (SDSC) | 4 × V100-32 GB, 10 cores, 96 GB | ~580 / ~270 | GPU → GPU → host | D1 first; 6–8 jobs per GPU |
| Expanse H100 | 4 × H100, 18 Sapphire Rapids cores (~1.3× Zen 2), 256 GB | ~1,360 / ~1,100 | GPU (barely) → host → host | D1 then B; 8–10 jobs per GPU |
| Delta (NCSA) | 4 × A100-40 GB or 4 × A40, 16 Milan cores (~1.15×), 64 GB; 8 × A100 nodes with 16 cores per GPU | ~1,070 / ~370 (A100), ~610 (A40) | GPU → GPU → host | D1, C4, C3; 8–10 jobs per GPU |
| Anvil GPU (Purdue) | 4 × A100, 32 Milan cores, 64 GB | ~2,100 / ~370 | GPU → GPU → balanced | D1, C4, C3; 8–10 jobs per GPU |
| DeltaAI, Vista (GH200) | 72 Grace cores per H100, 480 GB; coherent memory like the GB10 | ~5,000 / ~1,100 | GPU → GPU → balanced | the GB10 settings carry over (arm64 cu130 image, `MUSIC_CUDA_SYNC=block`, MPS); D1, C4, C3 |

- **Jobs per GPU:** enough that the GPU is busy ~90 % (a job alone keeps it 25–50 % busy:
  4–10 jobs), limited by host memory (5.8 GB per job with the pair file, 4.2 with
  `--particlize-only`, +~2 GB per job in flight) and by cores (OpenMP threads = cores per GPU /
  jobs, at least 2; `OMP_WAIT_POLICY=passive`, `MUSIC_CUDA_SYNC=block`). GPU memory is no limit
  (0.9 GB per job + 0.1 GB checkpoint).
- **MPS** gives +5–7 % and needs a per-user daemon on an exclusively allocated GPU (README_CUDA,
  "Packing several events onto one GPU with CUDA MPS"; works without root in the `Default` compute
  mode). On shared or `Exclusive_Process` GPUs run without it; the loss is the 5–7 %.
- **Images:** V100 needs sm_70 (`cu124`, or `cu126` once rebuilt — the published `cu126`/`cu130`
  of 2026-09-30 lack sm_70, Pelican and the memory fixes, `BuildContainerProd.md`); H100/A100
  `cu126`; GH200, B200 and the GB10 `cu130` (sm_90, 100, 120, 121). Apptainer with `--nv`;
  `slurm_prod_array.sh` for one GPU per task.
- **I/O:** `--particlize-only` writes ~95 MB per event (33 MB/s at 1,250 events/h per node), the
  pair + particlize mode ~235 MB; node-local scratch for the outputs and `LBT_TABLE_CACHE`,
  then `upload_follow.py`/`js_osdf.py`. 45–60 events per job amortise the ~10 s start-up;
  SLURM tasks of several hours.
- **Stage 2 on CPU nodes:** 3–4 cores per 1,000 events/h; no GPU.

## 7. Recommended order

1. **D1 (late Δτ)** — the cheapest change with the largest gain: ×1.5–1.8 on every GPU-bound
   machine; validation with the existing scripts.
2. **C4 (`delta_qi`) — done on MUSIC4GPU `reconst_stall`:** −29 % GPU time per step, measured;
   needs the C1-style validation (float-rounding-level differences) and the Metal run.
3. **C3 (active region)** — the largest and the most work; for O+O it removes the GPU as a limit
   altogether.
4. After 1–3 the host is the limit everywhere (the 3090 box: ~1,400 events/h with the pair file,
   ~1,700 with `--particlize-only`, by its cores, against a GPU limit of ~8,000; the GB10 ~2,700
   against ~3,000): **A12, `--particlize-only`, B1′, B2**, in that order of effort.
5. **E1** only if another 3–5× is needed after that.
- **Core-poor nodes now** (Bridges-2): 4 first (`--particlize-only`, A12, B1′, B2).
- **No grid change for Au+Au:** the output grid already reaches ±12.4 fm of MUSIC's ±15 fm (§3);
  the vacuum cells go only through C3.
- **Hydro-only** (`prod_AuAu_0_10`) profits from D1, C4, C3 the same way; its host side is the
  string deposition (B1).

## Method and files

- **Kernel profile:** `nsys profile --sample=none --cpuctxsw=none --trace=cuda -o nsys_jet python
  run_prod_jet.py --events 2 --seed 1 --reuse 2 --write-particlize both --seed-registry none`
  in `js_fno`, `MUSIC_PROFILE=1 MUSIC_CUDA_SYNC=block OMP_NUM_THREADS=4`; `nsys stats --report
  cuda_gpu_kern_sum`. 2 min on the GB10. `ncu` fails without
  `NVreg_RestrictProfilingToAdminUsers=0` (`RmProfilingAdminOnly: 1` in `/proc/driver/nvidia/params`).
  The job's output and the `.nsys-rep` were written to the session scratch directory
  (temporary).
- **Compiled kernels:** `cuobjdump --dump-resource-usage` and `--dump-sass` on
  `build_gpu/external_packages/music4gpu/src/libmusic.so` (sm_121), counting DADD/DMUL/DFMA/F2F.F64,
  STL/LDL and MUFU per kernel.
- **Matter extent:** `extent2.py` (scratch) over `prod_AuAu_0_10_jet/out/AuAu_0_10_jet_seed0001.h5`
  (one event on the old ±10 fm × ±5 output grid, 95 frames) and `prod_OO_isr/out/OO_isr_seed0001.h5`
  (3 events, ±9.84 fm × ±4.84): cells with e > 10⁻⁴ GeV/fm³ scaled to MUSIC's grid volume.
- **Node specifications:** [ACCESS resource documentation](https://support.access-ci.org/documentation/resources),
  [Purdue Anvil GPU](https://operations.access-ci.org/node/872), and a
  [WSU computing overview](https://conferences.lbl.gov/event/1721/contributions/8815/attachments/5094/4879/WSUComputing.pdf);
  the GPU speed ratios are FP32 peak rates except for the two measured machines.

**Caveats.** One profiled job and one Au+Au event for the extents (the O+O extents are from
3 events); all gains in §3–§5 are estimates from the measured shares, not measurements; the
GPU speed ratios are ±30 %; the per-node numbers follow from them and from the 3090 box's
core-s per event under load, with core-speed factors guessed from the GB10 vs 3960X
comparison (REPORT §4). Calibrate a new node with one job alone (`MUSIC_PROFILE=1`: `evolve.step_total`
and the kernel sum from `nsys`, or MUSIC4GPU's `tests/cuda_perstep_bench.sh`) before choosing
jobs per GPU.
