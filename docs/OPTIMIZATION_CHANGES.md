# Optimisation changes in the production chain: bit-identical and not (as of 2026-10-13)

Every speed-up, memory fix and output-changing correction that went into the O+O and Au+Au
productions since the GPU port of MUSIC (May 2026), by physics module, with its bit-identity status.
The list was checked against the git histories of MUSIC4GPU, X-SCAPE, js-contrib, iSS and
3dMCGlauber since 2026-05-01, not only against the plans and reports. "Bit-identical" means the
output files (hydro frames, surfaces, partons, hadrons) were compared dataset by dataset against the
build before the change and no value differed; the validation named is the one recorded in the
commit or document.

Code state this refers to: branches `final_optim` in MUSIC4GPU (`b972b04`), X-SCAPE (`50226a7c`) and
js-contrib (`de5f28c`); iSS fork `iSS_optim` (`98deaff`); 3dMCGlauber fork `cap_hard_quark_draws`
(`b99d680`). The production image `cu124-final_optim` (digest `91c04102…`) was built from X-SCAPE
`9f4b419` and predates the last two X-SCAPE commits (Matter t < 0 guard, iSS leak-fix pin).

Out of scope: changes to what is generated rather than how fast (the pT̂ windows per background,
the parton rapidity cut, the hard-momentum subtraction in O+O ISR, the particlize files and the
hadron-level workflow). They are documented in `Production_readiness_summary.md` §6–8.

## 1. Bit-identical changes

### MUSIC4GPU (hydro)

| when | item | change | effect |
|---|---|---|---|
| May–June | port work | diagnostics and D2H skips; freeze-out GPU sync only on facTau steps; `OMP_WAIT_POLICY=passive` by default | step 7.30 → 4.77 ms (OMP 48); idle threads no longer spin |
| Sep 24 | `freeze_out_surface = 0` | stop without building the surface (hydro-only production, or per instance) | hydro-only event 23.3 → 17.1 s; pair 60.1 → 49.1 s; stop steps identical, also on the CPU path |
| Sep 24 | jet source once per step | the liquefier's droplets prepared once per step, as the main source | with droplet pruning: pair event 184 → 60 s |
| Sep 25 | source fill, empty steps | the CPU source pre-pass is skipped at steps where no source can deposit (`has_active_sources_current_tau`) | source fill 15.3 → 5.9 s/event together with the binning; 30 datasets byte-identical |
| Sep 25 | source fill, string binning | strings, remnants and baryons binned by transverse reach; every cell adds the same terms in the same order | (included above) |
| Sep 26 | parallel surface finder | η-slices in parallel, deterministic; P stored | search 9.6 → 1.2 s/event; identical across runs and thread counts; 7.08 M iSS hadrons identical |
| Oct 2 | store memory released | `clean_hydro_event` swaps the store with an empty vector | −2 GB held per MUSIC instance between events; byte-identical |
| Oct 9 | `MUSIC_CUDA_SYNC=block` | the host blocks instead of spinning in `cudaStreamSynchronize` | −29 % CPU per event at unchanged throughput |
| Oct 9 | string source constants | per-string constants once per step, zero-envelope pairs skipped; no transverse-flow terms when the preflow factor is 0 | source fill 10.8 → 6.4 s (one event, 5 threads); every dataset identical |
| Oct 11 | A3 | the source activity test ignores the charge and baryon lists when those are not evolved | string fills 100 → 12 per hydro-only event |
| Oct 11 | A4 | a switch for the momentum-anisotropy diagnostic; off in the production XMLs | −1 to −2 core-s/event |
| Oct 11 | A5/A6 | frames packed straight into the store's tail (pinned staging on discrete GPUs, huge pages), no per-frame reallocation | no GPU → host grid copies; 19.8 → 16.4 ms/step host side |
| Oct 11 | A7 | the transverse-edge test of the freeze-out check as a GPU reduction | freeze-out check 0.93 → 0.016 s |
| Oct 11 | A9 | per-instance switch to not store MUSIC_2's evolution with `--particlize-only` | −2.8 s and ~2 GB per particlize-only job |
| Oct 12 | A10 | freeze-out steps alternate between two GPU buffer slots; no "previous step" copies | O+O +38 % events/h, −3.1 core-s/event |
| Oct 12 | A11 | only ε comes to the host per check; the cube corners the surface crosses are gathered on the GPU | +5.6 % on top of A10; 0 of 1.3e10 O+O output values differ over 540 events |
| Oct 12 | A12 | Cornelius without per-element heap allocations; one instance per OpenMP thread | surface search −30 % |

### X-SCAPE (framework, jets, liquefier, wrappers)

| when | item | change | effect |
|---|---|---|---|
| Sep 22 | OpenMP build fix | OpenMP flags keyed per language; the framework's pragmas had been dropped on macOS | SurfaceFinder 8.8 → 1.2 s (8 threads); a FastHydro event with iSS 55 → 14.6 s |
| Sep 24 | droplet pruning | the liquefier evaluates only the droplets that can deposit in the current step | pair event 184 → 60 s with the MUSIC4GPU change; identical on GPU and CPU |
| Sep 24 | EOS written before MUSIC is constructed | bug fix: the first instance had run with a stale EOS | all 100 frames identical after the fix |
| Sep 25 | `bulk_info` filled in parallel | the framework copy sized once and filled with an OpenMP loop instead of per-cell `push_back` | −4 s per pair event |
| Sep 25 | active-droplet test | `has_active_sources_current_tau()` from the pruned droplet list, for MUSIC's source skip | byte-identical |
| Sep 26 | LBT tables from LBT's path; per-job working directories (js-contrib) | concurrent jobs no longer race on `music_input`; heavy-quark recoil no longer runs on all-zero tables outside the build tree | 4 simultaneous jobs identical to serial runs; HQ kinematics identical to a run with the tables present |
| Sep 26 | iSS wrapper, hadrons taken once | each sample's hadron vector read once instead of a by-name lookup per hadron | −16 % of hadronize time at 500 oversamples |
| Sep 26 | compact hadron output | opt-in flat arrays from iSS instead of one `Hadron` object each; `SetNumberOfSamples` per surface | ~2.4 → ~0.3 MB per oversample; peak 3.7 → 1.39 GB at 1,000 oversamples |
| Oct 2 | slim `bulk_info` | 6 fields per cell (24 instead of 112 bytes) for Matter, LBT and the liquefier; the old copy freed first | peak memory per jet job ~20 → 8.7 GB; every dataset identical |
| Oct 10 | `LBT_TABLE_CACHE` | the 0.66 GB of LBT tables in one block, mapped read-only and shared between jobs | −0.6 GB and 6–11 s of start-up per Au+Au job; behaviour unchanged without the variable |
| Oct 10 | Matter, outside the light cone | `fillQhatTab` sets qhat = 0 directly where the hydro returned a zero cell anyway | output unchanged; ~500 warnings per O+O event gone |
| Oct 11 | vacuum cell for an empty history | an O+O event without a medium showers in vacuum instead of stopping the job | robustness; ~5 % of 0–80 % O+O events |
| Oct 11 | A8 | MUSIC_2 restarts from MUSIC_1's GPU checkpoint at the last step before the first droplet; `MUSIC_RESTART_VERIFY=1` compares the restarted state to a full run bit for bit | two-stage job 337 → 239 s (4 events); all pair/particlize datasets identical, also `--particlize-only` and discrete-GPU mode |

### iSS (particlization)

| when | item | change | effect |
|---|---|---|---|
| Sep 26 | Part A (A1–A3) | δf coefficients once per cell; thermal-yield cache per exact (T, μ); OpenMP over cells in the yield loop; one leaked hadron object per hadron removed | ~16.5 s fixed cost per surface → ~4 s |
| Oct 12 | Part C (C1–C4) | packed per-cell yield inputs and 8-species blocks; tabulated momentum CDF with branchless bisection; surface import without copies; reused inverse-CDF buffer; decay-table lookups by map | both legs at 500 oversamples, 1 thread: 14.2 → 10.6 s (correlated 13.8 → 8.8 s); checked at 1, 5, 20 threads, 71 M values per file |
| Oct 12 | C6 | the inverse-CDF running sum in a local (same additions, same order) | removes a store/load per addition that Apple Clang and GCC 15 emitted; neutral on the GB10 (GCC 14); identical hadrons on M3 Max and GB10 |
| Oct 12 | leak fix `98deaff` | decay channels and per-sample hadron vectors freed | 19,272 leaked blocks per O+O event → 0; hadrons identical over 100 in-job O+O events and the Au+Au surfaces. Pinned on `final_optim`, not yet in the built image |

### js-contrib / PyJetscape

| when | item | change | effect |
|---|---|---|---|
| Sep 25 | resample as matrix products (first version) | separable matmul passes instead of per-channel `map_coordinates` | part of the GB10 single-job speed-ups |
| Sep 26 | Blosc-zstd default | evolution datasets lzf → Blosc + zstd 3 + byte shuffle; `keep_bits` is an optional lossy rounding, off by default | pair file 400 → 285 MB per event, identical values |
| Sep 26 | hadron writer in whole units | `RaggedGroup.append_many` instead of one append per oversample | −30 % of hadronize time at 500 oversamples; datasets identical |
| Sep 26 | compact hadrons from iSS | `hadronize.py` takes iSS's arrays without copying | 8× less memory per oversample (see X-SCAPE) |
| Sep 27 | shared background | a reused background stored once, `arr_bg` as a virtual dataset | pair file 285 → 170 MB per event with 15 jets per background |
| Sep 29 | reader, shared background binned once | `HadronFileReader` histograms a shared background once per campaign | analysis only; same result as per event |
| Oct 2, Oct 10 | frame-by-frame reads | background and jet legs read one τ frame at a time; one OpenBLAS thread | peak RSS per job 20 → 14 GB, then −0.8 GB more; files identical |
| Oct 11 | A1 | a reused background is recognised from the native frames, without resampling | −7 core-s per campaign event (3090) |
| Oct 11 | A2 | resampling as banded matmuls into reused buffers; tested bit for bit against the einsum reference on x86 OpenBLAS, Arm OpenBLAS and Accelerate | 11 → 3.6 ms per frame (GB10) |
| Oct 12 | C5 | no libtorch import in hadronisation processes | `import jetscape` 0.72 → 0.25 s; −460 MB RSS per process |
| May–Sep | writers | grid-mode bulk writer identical to the legacy writer; in-job vs offline hadronisation identical (1.71 M hadrons) | peak memory 24.6 → 11–15.5 GB (O+O) |

Launch-side settings with no effect on output: CUDA MPS, `run_jobs.sh --stagger`, `OMP_WAIT_POLICY=passive`,
`OPENBLAS_NUM_THREADS=1`.

### 3dMCGlauber (initial state)

| when | item | change | effect |
|---|---|---|---|
| Oct 12 | hard-quark draw cap (`b99d680`, upstream PR chunshen1987/3dMCGlauber#7) | a cap on the draws in `Pick_and_subtract_hard_parton_momentum` | only events that would otherwise hang forever change (~1 in 500–1,300 O+O ISR events); every event that completed before is unchanged |

## 2. Changes that are not bit-identical

### 2a. Speed and precision choices

| when | change | module | what changes | validation and verdict |
|---|---|---|---|---|
| May 26 | **the GPU port itself** (fp32 on the GPU vs double on the CPU; vacuum cells at rest) | MUSIC4GPU | hydro fields 2.0–2.2e-3 (energy-weighted \|Δe\|/e), wake ~1 %, bulk hadrons 0.2–0.3 %, freeze-out volume −0.19 %; the reference level for everything below | `MUSIC_CPU_vs_GPU.md`, J#47, CUDA and Metal |
| May 26 | `--use_fast_math` on the CUDA kernels | MUSIC4GPU | approximate divides and square roots; ε error 2.7e-5 against the exact-math build; 2× on `delta_qi` | part of the port baseline (README_CUDA phase 5); never re-examined separately |
| May 29 | GPU-side packing of the stored frames, GPU-resident state | MUSIC4GPU | the stored P, s, T interpolated from the EOS tables on the GPU: within 8.5e-5; surfaces and velocities identical | ~2× hydro wall time on the 3090; GB10 CPU 106 → 9 s per event |
| Oct 6 | output grid ±12.4 fm at 64 cells, ±5 at 32 (`grid_fno.yaml`) | js-contrib | a different training product (0.3875 fm instead of 0.3125): integrated wake within 1–4 %, shape at ≥ 0.5 fm within ~10 %, lower peaks | `Wake_grid_comparison.md` |
| Oct 12 | **C1: O+O MUSIC grid ±15 → ±10.5 fm** (XML default) | MUSIC4GPU via XML | different grid; no boundary hits in 720 events; dN/dη, v₂, jet hadrons unchanged within errors | O+O grid check, `BENCHMARK_2x3090.md`; roughly doubles the O+O GPU ceiling |
| Oct 12 | **C4: solver stall break** at float resolution in `gpu_solve_v` / `gpu_solve_u0` (CUDA and Metal) | MUSIC4GPU | float-rounding level: fields ~1e-6, surfaces ±2 elements, wake fields ~1e-3 (L2); Au+Au wake hadron yield +0.8 % ± 0.2 % (6e-5 of the jet leg) | `REPORT_FINAL_OPTIM.md`: **accept**. −29 % GPU time per step; Au+Au +28 %, O+O +15–23 % events/h |
| Oct 12 | **D1: Δτ 0.02 → 0.05 fm/c from τ = 3 fm/c**, frames and freeze-out checks still every 0.1 fm/c, the liquefier follows the step | MUSIC4GPU + X-SCAPE + XMLs | bit-identical before 3 fm/c; medium 1.1e-3 (Au+Au, τ ≥ 3), 1.2e-4 (O+O); deposition normalised to 1e-4; cell-level wake ~5 % at sub-fm scales, 0.8 % at 2 fm smoothing; integrated wake 0.1–0.5 %; O+O surface +0.20 %, τ_fo one frame later in 11 % of events; hadrons unchanged within ±2 % (20 backgrounds) | `REPORT_FINAL_OPTIM.md`: **acceptable**, one D1 setting per campaign (`--late-dtau 0` turns it off). Au+Au +33 %, O+O +6–8 % on top of C4 |
| Sep 26 | **iSS Part B: correlated jet/background sampling** (`hadronize.py --correlated`) | iSS + X-SCAPE + js-contrib | a different sampling algorithm by design: common random numbers per (species, block), decays inline; opt-in, the conventional sampler unchanged | `PLAN_iSS_optim.md` Part B; ~2× smaller wake error bars at the same oversamples |
| any build | container toolchain (arm64 cu130 image vs native build) | build | hydro within 6e-4 of the frame maximum, wake 1.1e-3, identical freeze-out steps | `Production_readiness_summary.md` §10 |

### 2b. Corrections that change output

These fix wrong behaviour rather than approximate it. Productions made before them are affected.

| when | change | module | what changes | validation |
|---|---|---|---|---|
| May 26 | RK2 swap-gate fix | MUSIC4GPU | RK2 had silently been forward Euler on the GPU; drift 3e-3 → 3e-5 | identical to the reference `main_gpu`, all benchmark cases |
| Sep 22 | SurfaceFinder flow normalisation | X-SCAPE | u⁰ = √(1+v²) gave u·u = 1 − v⁴ (13 % short at v = 0.7); the last lattice cell in each direction had been dropped | u·u = 1 within 1e-5; 8 tests. Latent until iSS sampled SurfaceFinder cells |
| Sep 28 | **liquefier kernel normalised on MUSIC's grid**, droplets deposited at the hard vertex (X#155) | X-SCAPE | MUSIC had received 0.004–6× each droplet's four-momentum; 40 % of events had their jet source off by > 20 %; vertex droplets were lost | wake ΔP⁰ / droplets: median 0.93 → 0.99, correlation 0.55 → 0.97 (285 events); 9/9 liquefier tests |
| Sep 28 | **no double counting of liquefied partons** (X#154) | X-SCAPE | free-streaming LBT partons absorbed by the liquefier were deposited and hadronised: 13.9 % of the initial jet energy on average (up to 67 %) at pT̂ 10–40 | energy balance closes to 7e-7 GeV; hydro legs, droplets and freeze-out times identical, only the pstat of the affected partons changes; old files repairable with `fix_lbt_double_counting.py` |
| Sep 28 | vacuum guard with reset (VacReset) | MUSIC4GPU | a vacuum cell with u⁰ in the thousands could drag the fluid edge and stall the whole grid; such cells are put at rest. The guard alone is bit-identical; the reset changes a normal event by ≤ 1 % above freeze-out, 0.03 % energy-weighted | the failing event freezes out at the CPU's τ (10.86 fm/c); Metal validated on an M3 Max |
| Oct 12 | **Matter: no qhat table lookup before t = 0** (`8c975d37`) | X-SCAPE | O+O ISR only: a negative table index from partons starting at t < 0 read a garbage qhat (~3,800 lookups per event); the old code was itself non-reproducible (run-to-run skips, occasional segfault). Au+Au (no ISR) bit-identical | 100 O+O events, two runs bit-identical, no skip or crash. Not yet in the built image |

## 3. Net effect

2 × RTX 3090 box, campaign Au+Au two-stage (pair + particlize), events/h by makespan:

| state | events/h |
|---|---|
| before A1 (2026-10-11, with the September changes) | 796 |
| A1–A8 (`cu124-music2restart`) | 1,121 |
| + A10/A11 (`cu124-freezeoutcopies`) | 1,271 |
| + C4, A12 (`cu124-final_optim`, `--late-dtau 0`) | 1,628 |
| + D1 (`cu124-final_optim`, defaults) | **2,160** |

Before the September changes a pair event took 184 s on the GB10 alone (68 events/h); after them
~30–36 s. O+O, hadron files only: 3,798 on the ±15 fm grid → 6,444 (C1) → 10,313 (A10/A11) →
12,722 (C4) → **13,728** (D1). Both productions are now host-bound on that machine.

Everything in §1 is bit-identical. Of §2a, C1, C4 and D1 are the three speed changes with a physics
check behind them; the port, fast-math and GPU packing are the baseline all later comparisons use;
Part B and `keep_bits` are opt-in. The corrections of §2b change physics more than C4 and D1 together,
and productions made before 2026-09-28 need the repair script or a rerun.

## Sources

- `docs/Plans/REPORT_CPU_GPU.md` §5 (the A/B/C item list), `docs/Plans/PLAN_cpu_fixes_A.md`,
  `docs/Plans/PLAN_music2_restart.md` (A8), `docs/Plans/PLAN_iSS_optim.md` (Parts A, B, C, C6),
  `docs/Plans/PLAN_slim_bulk_info.md`, `docs/Plans/Final_Optims.md` (C4, D1, A12),
  `docs/REPORT_FINAL_OPTIM.md` (3090 physics check), `docs/BENCHMARK_2x3090.md` (A10/A11, C1,
  final_optim), `docs/BENCHMARK_GB10.md` (the September single-job speed-ups),
  `docs/Production_readiness_summary.md` §3, §5, §9, §10, `docs/MUSIC_CPU_vs_GPU.md`,
  `docs/Wake_grid_comparison.md`, `docs/README_h5_optim.md`; MUSIC4GPU `README_CUDA.md`.
- Git histories since 2026-05-01: MUSIC4GPU (`6d33feb`, `fa6ec5b`, `9535640`, `c7b75ba`, `5916287`,
  `f8338cc`, `f73ffab`, `5fda4bc`, `2a55f96`), X-SCAPE (`3955cef6`, `88099402`, `a80a9932`,
  `75b4ce7a`, `127ae486`, `00a9e795`, `780f5036`, `dab5f944`, `bc8ceff3`, `01584cc3`, `5900ebd0`,
  `be68bc85`, `8c975d37`), js-contrib (`64d0c2a`, `cec68fd`, `66bc85e`, `ba32ea0`, `86e142e`,
  `9406bc6`, `4941722`), iSS (`01f7cf9`, `3192982`, `896e198`, `b15dd05`, `98deaff`), 3dMCGlauber
  (`b99d680`).
