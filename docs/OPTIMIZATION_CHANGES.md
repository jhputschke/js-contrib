# Optimisation changes in the production chain: bit-identical and not (as of 2026-10-13)

Every speed-up and related fix that went into the O+O and Au+Au productions since the GPU port of
MUSIC, by physics module, with its bit-identity status as recorded in the documents listed at the
end. "Bit-identical" means the output files (hydro frames, surfaces, partons, hadrons) were compared
dataset by dataset against the build before the change and no value differed.

Code state this refers to: branches `final_optim` in MUSIC4GPU (`b972b04`), X-SCAPE (`50226a7c`) and
js-contrib (`074367a`); iSS fork `iSS_optim` (`98deaff`); 3dMCGlauber fork `cap_hard_quark_draws`
(`b99d680`). The production image `cu124-final_optim` (digest `91c04102…`) was built from X-SCAPE
`9f4b419` and predates the last two X-SCAPE commits (Matter guard, iSS leak-fix pin).

## 1. Bit-identical changes

### MUSIC4GPU (hydro)

| item | change | effect |
|---|---|---|
| A3 | `has_active_sources_current_tau()` ignores the charge and baryon lists when those are not evolved | string fills 100 → 12 per hydro-only event |
| A4 | a switch for the momentum-anisotropy diagnostic; off in the production XMLs | −1 to −2 core-s/event |
| A5/A6 | evolution frames packed straight into the store's tail (pinned staging on discrete GPUs, huge pages), no per-frame reallocation | no GPU → host grid copies, 19.8 → 16.4 ms/step host side |
| A7 | the transverse-edge test of the freeze-out check as a GPU reduction, no full-grid copy | freeze-out check 0.93 → 0.016 s |
| A9 | per-instance switch to not store MUSIC_2's evolution with `--particlize-only` | −2.8 s and ~2 GB per particlize-only job |
| A10 | freeze-out steps alternate between two GPU buffer slots; no "previous step" copies | O+O +38 % events/h, −3.1 core-s/event |
| A11 | only ε comes to the host per check; cube corners the surface crosses are gathered on the GPU | +5.6 % on top of A10 (checked: 0 of 1.3e10 O+O output values differ over 540 events) |
| A12 | Cornelius without per-element heap allocations; one instance per OpenMP thread | surface search −30 % |
| port work (May–June 2026) | diagnostics and D2H skips; `freeze_out_surface = 0` for hydro-only; jet source prepared once per step; parallel deterministic surface finder | step 7.30 → 4.77 ms; pair event 184 → 60 s; search 9.6 → 1.2 s/event, identical across thread counts |

### X-SCAPE (framework, jets, liquefier)

| item | change | effect |
|---|---|---|
| A8 | MUSIC_2 restarts from MUSIC_1's GPU checkpoint at the last step before the first droplet instead of from τ0. `MUSIC_RESTART_VERIFY=1` compares the restarted state to a full run bit for bit | two-stage job 337 → 239 s (4 events); all pair/particlize datasets identical, also `--particlize-only` and discrete-GPU mode |
| droplet pruning, per-step source preparation | X-SCAPE #141, `3037be7` | part of the 184 → 60 s above; identical on GPU and CPU |
| per-job working directories, LBT-table path | J#11, J#12, X#147 | concurrent jobs identical to serial runs; heavy-quark kinematics identical to a run with the tables present |
| several pT̂ windows per background | X#152, J#25 (the campaign mode) | background bit-identical; the jet legs are new events by design; 1.8× events/h |

### iSS (particlization)

| item | change | effect |
|---|---|---|
| Part A (A1–A3) | δf coefficients once per cell; thermal-yield cache per exact (T, μ); OpenMP over cells in the yield loop; one leaked hadron object per hadron removed | ~16.5 s fixed cost per surface → ~4 s |
| Part C (C1–C4) | packed per-cell yield inputs and 8-species blocks; tabulated momentum CDF with branchless bisection; surface import without copies; reused inverse-CDF buffer; decay-table lookups by map | both legs at 500 oversamples, 1 thread: 14.2 → 10.6 s (correlated 13.8 → 8.8 s); checked at 1, 5, 20 threads, 71 M values per file |
| C6 | the inverse-CDF running sum in a local (same additions, same order) | removes a store/load per addition that Apple Clang and GCC 15 emitted; neutral on the GB10 (GCC 14), identical hadrons on M3 Max and GB10 |
| leak fix `98deaff` | decay channels and per-sample hadron vectors freed | 19,272 leaked blocks per O+O event → 0; hadrons identical over 100 in-job O+O events and the Au+Au surfaces; iSS time unchanged. Pinned on `final_optim`, not yet in the built image |

### js-contrib / PyJetscape

| item | change | effect |
|---|---|---|
| A1 | a reused background is recognised from the native frames, without resampling | −7 core-s per campaign event (3090) |
| A2 | output-grid resampling as matrix products; tested bit for bit against the einsum reference on x86 OpenBLAS, Arm OpenBLAS and Accelerate | 11 → 3.6 ms per frame (GB10), 30–60 ms → faster on the 3090 |
| C5 | no libtorch import in hadronisation processes | `import jetscape` 0.72 → 0.25 s, −460 MB RSS per process |
| writers | grid-mode bulk writer identical to the legacy writer; in-job vs offline hadronisation identical (1.71 M hadrons) | peak memory 24.6 → 11–15.5 GB (O+O) |

### 3dMCGlauber (initial state)

| item | change | effect |
|---|---|---|
| hard-quark draw cap (`b99d680`, upstream PR chunshen1987/3dMCGlauber#7) | a cap on the draws in `Pick_and_subtract_hard_parton_momentum` | only events that would otherwise hang forever change (~1 in 500–1,300 O+O ISR events); every event that completed before is unchanged |

## 2. Changes that are not bit-identical

| change | module | what changes | validation and verdict |
|---|---|---|---|
| **the GPU port itself** (fp32 on the GPU vs double on the CPU) | MUSIC4GPU | hydro fields 2.0–2.2e-3 (energy-weighted \|Δe\|/e), wake ~1 %, bulk hadrons 0.2–0.3 %; the reference level for everything below | `MUSIC_CPU_vs_GPU.md`, J#47 |
| **C1: O+O MUSIC grid ±15 → ±10.5 fm** (XML default) | MUSIC4GPU via XML | different grid; no boundary hits in 720 events; dN/dη, v₂, jet hadrons unchanged within errors | O+O grid check, `BENCHMARK_2x3090.md`; roughly doubles the O+O GPU ceiling |
| **C4: solver stall break** at float resolution in `gpu_solve_v` / `gpu_solve_u0` (CUDA and Metal) | MUSIC4GPU | float-rounding level: fields ~1e-6, surfaces ±2 elements, wake fields ~1e-3 (L2); Au+Au wake hadron yield +0.8 % ± 0.2 % (6e-5 of the jet leg) | `REPORT_FINAL_OPTIM.md`: **accept**. −29 % GPU time per step; Au+Au +28 %, O+O +15–23 % events/h |
| **D1: Δτ 0.02 → 0.05 fm/c from τ = 3 fm/c**, frames and freeze-out checks still every 0.1 fm/c, the liquefier follows the step | MUSIC4GPU + X-SCAPE + XMLs | bit-identical before 3 fm/c; medium 1.1e-3 (Au+Au, τ ≥ 3), 1.2e-4 (O+O); deposition normalised to 1e-4; cell-level wake ~5 % at sub-fm scales, 0.8 % at 2 fm smoothing; integrated wake (energy, depletion, centroid, width) 0.1–0.5 %; O+O surface +0.20 %, τ_fo one frame later in 11 % of events; hadrons unchanged within ±2 % (20 backgrounds) | `REPORT_FINAL_OPTIM.md`: **acceptable**, one D1 setting per campaign (`--late-dtau 0` turns it off). Au+Au +33 %, O+O +6–8 % on top of C4 |
| **iSS Part B: correlated jet/background sampling** (`hadronize.py --correlated`) | iSS + X-SCAPE + js-contrib | a different sampling algorithm by design: common random numbers per (species, block), decays inline; opt-in, the conventional sampler is unchanged | `PLAN_iSS_optim.md` Part B; ~2× smaller wake error bars at the same oversamples |
| **Matter: no qhat table lookup before t = 0** (`8c975d37`) | X-SCAPE | O+O ISR only: the old code read a garbage qhat for ~3,800 lookups per event (negative table index from ISR partons starting at t < 0) and was itself non-reproducible (run-to-run skips, occasional segfault). Au+Au (no ISR) bit-identical | 100 O+O events, two runs bit-identical, no skip or crash. Not yet in the built image |
| RK2 swap-gate fix; NaN guard with reset | MUSIC4GPU | bug fixes, not optimisations: RK2 had silently been forward Euler (drift 3e-3 → 3e-5); the reset changes a failing event at the 1 % level, the guard alone is bit-identical | `Production_readiness_summary.md` §3 |
| container toolchain (arm64 cu130 image vs native build) | build | hydro within 6e-4 of the frame maximum, wake 1.1e-3, identical freeze-out steps | `Production_readiness_summary.md` §10 |

## 3. Net effect

2 × RTX 3090 box, campaign Au+Au two-stage (pair + particlize), events/h by makespan:

| state | events/h |
|---|---|
| before A1 (2026-10-11) | 796 |
| A1–A8 (`cu124-music2restart`) | 1,121 |
| + A10/A11 (`cu124-freezeoutcopies`) | 1,271 |
| + C4, A12 (`cu124-final_optim`, `--late-dtau 0`) | 1,628 |
| + D1 (`cu124-final_optim`, defaults) | **2,160** |

O+O, hadron files only: 3,798 on the ±15 fm grid → 6,444 (C1) → 10,313 (A10/A11) → 12,722 (C4) →
**13,728** (D1). Both productions are now host-bound on that machine.

Everything in the list is bit-identical except C1, C4 and D1, which are the three changes with a
physics check behind them, and iSS Part B, which is opt-in. The two bug fixes that change output
(Matter t < 0 guard, RK2 gate) correct wrong behaviour rather than approximate it.

## Sources

- `docs/Plans/REPORT_CPU_GPU.md` §5 (the A/B/C item list), `docs/Plans/PLAN_cpu_fixes_A.md`,
  `docs/Plans/PLAN_music2_restart.md` (A8), `docs/Plans/PLAN_iSS_optim.md` (Parts A, B, C, C6),
  `docs/Plans/Final_Optims.md` (C4, D1, A12), `docs/REPORT_FINAL_OPTIM.md` (3090 physics check),
  `docs/BENCHMARK_2x3090.md` (A10/A11, C1, final_optim), `docs/Production_readiness_summary.md`
  (port fixes, writers, containers), `docs/MUSIC_CPU_vs_GPU.md`.
- Commit messages: X-SCAPE `8c975d37` (Matter), iSS `98deaff` (leaks), `b15dd05` (C6);
  3dMCGlauber fork `b99d680`.
