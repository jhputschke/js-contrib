# Production readiness: jet-wake analysis and FNO training data

A summary of the changes in **MUSIC4GPU**, **X-SCAPE** and **js-contrib** that prepare the
0–10% Au+Au 200 GeV jet + medium production for its two uses:

- **jet-wake analyses**, from hadrons sampled offline on the stored freeze-out surfaces;
- **FNO training data**, the stored hydro evolutions of a background and its jet leg.

For each change it gives the motivation, the benchmarks, the validation and the physics
tests, and ends with an assessment and the open points.

*Status: 2026-10-07. X-SCAPE `contrib` `cbc72639`, MUSIC4GPU `XSCAPE` `49439c0`, iSS fork
`common_seeds` `3192982`, js-contrib `main` `56fac04`. Hardware is the GB10 (20 cores, 121 GB
unified memory) unless noted. PR numbers are given per repository: **M#** MUSIC4GPU, **X#**
X-SCAPE, **J#** js-contrib.*

---

## 1. In short

| | status |
|---|---|
| **Jet-wake analysis** (hadron level) | **Ready for production on GPUs**, from native builds or from images rebuilt from current `main`/`contrib`. The energy deposition is now correct: liquefier kernel normalized (X#155), double counting removed (X#154), vertex droplets kept. The deposited energy reappears in the wake: wake/droplets median 0.99 with correlation 0.97. Correlated sampling makes jet − background 8× cheaper (X#151, J#19). |
| **FNO training data** (hydro evolutions) | **Ready, with stated limits**: new 64 × 64 × 32 output grid (J#45), edge flag (J#44), lossless Blosc-zstd, shared backgrounds. The evolutions carry no viscous fields. Hadrons from the stored history need δf corrections (J#23, J#24). The `ntau_freezeout` convention differs by one from the earlier fast_data files. |
| **Throughput** | Pair event 184 s → ~30–36 s. GB10 campaign 68 → **336 events/h** (c1 settings); **~810 events/h** on 2 × RTX 3090. |
| **Memory** | ~20 GiB → **8.7 GiB** peak per job; 4 jobs in 29 GB instead of 66–69. |
| **Disk** | Pair file 400 MB (lzf) → **170 MB** per event (Blosc-zstd, shared background at 15 jets per background). Particlize file 94–154 MB. Hadrons ~77 MB. |
| **Main open item** | **The production images must be rebuilt.** The published `cu126`/`cu130` lack sm_70, Pelican and the memory fixes. No image has the new grid or the edge flag yet (§10). |

---

## 2. The pipeline and where each part lives

```
3D MC-Glauber strings ──► MUSIC_1 (background) ──► Matter + LBT ──► CausalLiquefier droplets
                                │                                         │
                                │            MUSIC_2 (same strings + droplets = jet leg)
                                ▼                                         ▼
      PairH5Writer: arr_bg, arr, source/droplets, shower/, diag/      → FNO training data
      ParticlizeH5Writer: both freeze-out surfaces + final partons    → hadronize.py (iSS, Colorless)
                                                                         → HadronFileReader / ROOT export
                                                                         → wake analyses
```

| repo | role |
|---|---|
| MUSIC4GPU | GPU port of MUSIC 3.1 (CUDA, Metal; Kokkos on a side branch), fp32, no new physics. Its jet source slot takes the liquefier's droplets. |
| X-SCAPE | Framework: MUSIC wrapper (two legs, memory), liquefier, Matter/LBT, PythiaGun, SoftParticlization/iSS, writers. |
| js-contrib | PyJetscape drivers (`run_prod_jet.py`, `run_jobs.sh`, `hadronize.py`), the HDF5 writers and formats, FastHydro, analysis (readers, ROOT), containers, transfer and launch tools, the docs. |

---

## 3. GPU hydro: MUSIC4GPU

### 3.1 Correctness of the port (May–June 2026)

| change | motivation | validation |
|---|---|---|
| **RK2 swap gate** (M#1, M#4) | `swap_curr_future_gpu()` was gated on the wrong flag, so RK2 silently became forward Euler. This was the true cause of a "~100× accuracy gap" that had been blamed on fp32. | Drift 3e-3 → 3e-5 (64×64×32), bit-identical to the reference `main_gpu`, all benchmark cases pass (M3 Max, RTX 3090). |
| **CUDA uploads** (M#1) | A missing host-to-device upload left the kernels on uninitialised memory (eps_max = 0). Metal's coherent memory masked it. | RTX 3090: max relative error 8.8e-5 (2D), 2.7e-5 (3D). |
| **EOS tables log-sampled** (`2affe85`) | The linear 8192-point tables put the whole QCD crossover on one segment (Δe ≈ 1.18 fm⁻⁴ at e ≈ 0.15). The result was 6% divergence and an abort for EOS 91. | About 1e-4 mid-run, 2.5e-3 late (Metal). The test `eos_gpu_vs_cpu.sh` gives EOS 91 6.4e-4 and EOS 0 4.7e-5. |
| EOS 91 allowed on the GPU (`f93812f`); kernel guards identical to the CPU's; 64-bit surface counts; `MUSIC_FORCE_CPU=1` | A guard silently sent the standard X-SCAPE configuration to the CPU. One binary should be able to run both paths for validation. | — |
| `MUSIC_CUDA_FORCE_COHERENT` (M#2) | Test the GB10's coherent-memory path on a discrete card. | Bit-identical to the discrete path. |

**Precision against the CPU:**
- eps_max: 2.7e-5 to 1e-4 (EOS 0), 6.4e-4 (EOS 91).
- O+O fields: better than 1e-3.
- Packed outputs in cells with T > 0.1 GeV: at most 8.5e-5 (M#6).

### 3.2 Speed (CPU side and production path)

| change | effect | validation |
|---|---|---|
| Diagnostics, D2H skips (M#3, M#5) | step 7.30 → 4.77 ms (OMP 48) | bit-identical |
| GPU-side output packing, parallel memory output, freeze-out sync only on facTau steps, `OMP_WAIT_POLICY=passive` by default (M#6) | ~2× hydro wall time on RTX 3090; GB10 CPU time 106 → ~9 s | surfaces and velocities identical; P/s/T within 8.5e-5 |
| `freeze_out_surface = 0` (M#9, X#142) | hydro-only event 23.3 → 17.1 s; pair 60.1 → 49.1 s | evolutions and stop steps bit-identical, also on the CPU path |
| Source fill: skip empty steps, bin strings by transverse reach (M#10, X#144/#145) | source fill 15.3 → 5.9 s/event; events 37.2/45.5 → 30.6/38.0 s | byte-identical, 30 datasets |
| Jet source prepared once per step (`3037be7`) plus droplet pruning (X#141) | **pair event 184 → 60 s** | bit-identical on GPU and CPU |
| Parallel, deterministic freeze-out surface (M#12) | search 9.6 → 1.2 s/event; both surfaces +5.4 s instead of +13.5 s | bit-identical across runs and thread counts; 7.08 M iSS hadrons bit-identical |

**Kernel speed against the CPU:**
- GB10 vs 20 threads: ~5.2–5.5× per step.
- RTX 3090 vs 48 threads: 4.2×, 9.1× with the diagnostic fix.
- M3 Max vs 12 threads: 5.0×.

On the GB10 the host side dominates the wall time.

### 3.3 Robustness

- **Stalled GPU evolutions fixed (M#13, pinned by X `9509aea2`; [`VacReset_BUG.md`](https://github.com/jhputschke/MUSIC4GPU/blob/XSCAPE/docs/VacReset_BUG.md)).**
  - **Failure:** in about 1 of 40 background runs the GPU grid silently stopped evolving and ran to the maximum time (0 of 600 jet legs). The downstream wake energies came out as −5e4 to −8e4 GeV.
  - **Cause:** fp32 vacuum cells reached u⁰ = 2048 and carried T^ττ of about 1 GeV/fm³. W^μν/Π overflowed, the regulator turned ∞·0 into NaN, and the reverts froze the whole grid.
  - **Fix:**
    - vacuum cells (e < 1e-5 fm⁻⁴, u⁰ > 10) are put at rest;
    - non-finite W/Π are set to 0 and counted, with a warning;
    - `MUSIC_ABORT_ON_NONFINITE=1` stops the run instead.
  - **Validation, CUDA:** the failing event now freezes out at the CPU's τ (10.86 fm/c). The guard alone is bit-identical. With the reset: largest |Δe|/e above freeze-out 1%, background energy −0.03%, wake ±0.6 GeV of about 40.
  - **Validation, Metal:** the same level of change.
  - **Older data:** an affected GPU run shows `Maximum allowed time reached` in its log, and `wake_observables.py` flags it (`*_no_freezeout`). Example: gridnorm file 0002.
- **`StringFind4` stops at EOF** (M#11) instead of spinning forever. It used to hang concurrent jobs that shared `music_input`; the root cause is fixed in js-contrib (J#11).
- **The evolution store releases its memory** (M#15): ~2–2.5 GiB per instance. Byte-identical.
- **Surface pressure initialised** (M#12): it had held garbage, e.g. −1.3e26. Hadrons were unaffected, because iSS recomputes P.
- **Cornelius bounds checks** (M#6): guards against a real latent out-of-bounds write; they never fired in tests.

### 3.4 GPU vs CPU at the level of observables (J#47, `MUSIC_CPU_vs_GPU.md`)

The MUSIC4GPU README called the backends EXPERIMENTAL until a comparison of final-state observables existed. J#47 is that comparison on a small scale: 6 jet events, 3 backgrounds, same seeds, MUSIC's grid, iSS with matched seeds and correlated sampling. Since M#16 (2026-10-07) the MUSIC4GPU README calls the backends **validated against the CPU build** and carries this comparison as `docs/MUSIC_CPU_vs_GPU.md`. CUDA (GB10) and Metal (M3 Max, repeated 2026-10-08 with the same options) are both validated up to hadrons, with the same systematic offsets. **Still missing there:** a real large-scale campaign compared with the CPU build (thousands of events, several centralities and systems).

**Equal within errors:**
- freeze-out times, identical in all 9 MUSIC runs;
- the fields above freeze-out: |Δe|/e 2.1e-3, |ΔT|/T 2.9e-4;
- the momentum anisotropy ε_p;
- ⟨pT⟩, v₂ and v₃;
- the wakes of the events with the same showers: fields 0.1–1%, integrals ~1%.

**Systematic, on the GPU:**
- energy above freeze-out −0.3%, dilute fluid +0.4%;
- freeze-out volume −0.19%;
- **charged hadrons −0.20 ± 0.02%**.

**Consequence:** negligible within one campaign, but a normalization offset between paths. Keep a campaign, and an FNO training set, on one path, and don't add CPU-fallback jobs to a GPU campaign.

---

## 4. The two-stage pair production (X-SCAPE + js-contrib)

| change | motivation | validation / numbers |
|---|---|---|
| **Jet source slot** (MUSIC update X#138; M `b9cc8be`) | MUSIC_2 has to receive the liquefier's droplets on top of the string source. | Null test: jet leg = background in all 95 frames. |
| **`PairH5Writer` + `prod_AuAu_0_10_jet`** (J#7) | One FNO4d-schema file per job with both legs, droplets, showers and diagnostics. A bad leg skips the event instead of writing half a pair. | Null test passes. CPU vs GPU wake: correlation ≥ 0.99999. Hydro-only production reproduced bit for bit. |
| **Grid-boundary flag reset per event** (X#141) | After one hit at MUSIC's grid edge, every later event of that instance stopped at its first freeze-out check. | Flag now per event (`diag/{bg,jet}_hit_boundary`), with a warning. |
| **`EOS_to_use` written before MUSIC is constructed** (X `a1753327`) | The first instance ran with a stale EOS (91 instead of 9): e_fo 0.243 vs 0.234, and the null test failed. | All 100 frames bit-identical after the fix. |
| **`set_hydro_dtau` for every initial condition** (X#146) | String ICs skipped it, so the jet source's time step was 0.1 instead of 0.02 fm/c (pruning window 5× too wide). | Byte-identical; 30.9/38.0 → 29.5/35.6 s. |
| **`bulk_info` sized once, filled in parallel** (X#143); faster resample (J#9) | The framework copy and the resampling dominated the event after the hydro. | Copy stage 7.7 → 2.5 s, resample 14.7 → 3.1 s. Resample: 1 of 1.4e8 values differs by 1 ulp; the rest is byte-identical. |
| **Per-job working directories** (J#11), LBT-tables link (J#12), Matter reads LBT's table path (X#147) | Concurrent jobs raced on `music_input` (2 of 3 hung). Outside the build tree, heavy-quark recoil ran on all-zero tables (garbage c/b kinematics). | 4 simultaneous jobs byte-identical to serial runs. HQ kinematics bit-identical to a run with the tables present. |
| **`--surface` / `freeze_out_surface` switch** (J#8, X#142) | FNO-only runs don't need surfaces. | Bit-identical evolutions; −11 s per pair event. |

---

## 5. Jet–medium physics fixes (X-SCAPE)

These change physics results. Productions made before them are affected (§11).

| fix | problem and size | validation |
|---|---|---|
| **Liquefier kernel normalized on MUSIC's grid** (X#155, recorded by J#29) | MUSIC received 0.004–6× each droplet's four-momentum. 27% of droplets were off by more than 20%, and 40% of events had injected energy off by more than 20%. Vertex droplets (τ = 0) got η = NaN and were lost (56 GeV in 300 events). | 4 new unit tests, 9/9 liquefier tests pass. Production droplets come out exact. **Wake ΔP⁰ / droplets: median 0.93 → 0.99, 16–84% range 0.65–1.10 → 0.91–1.06, correlation 0.55 → 0.97** (285 events). |
| **No double counting of liquefied partons** (X#154) | Free-streaming LBT partons absorbed by the liquefier were deposited **and** hadronized: 13.9% of the initial jet energy on average (up to 67%) at pT̂ 10–40. | Energy balance closes to 7e-7 GeV (15 events). Old files can be repaired with `fix_lbt_double_counting.py`. |
| **SurfaceFinder flow normalization** (X `88099402`) | u⁰ = √(1+v²) gave u·u = 1 − v⁴ (13% short at v = 0.7). Latent until iSS sampled SurfaceFinder cells. | u·u = 1 within 1e-5; 8 tests. |
| **Quieter vertex-conservation warning** (X#142) | 19 warnings for 25 droplets, all ≤ 1.7%. | Output identical. |

**End-to-end energy balance** (J#15, seed 1, before correlated sampling):
- bulk hadrons, jet − background: +24.6 ± 5.6 GeV at |η| < 2, against 25.0 GeV deposited;
- jet fragments: 106.3 ± 5.5 GeV, equal to the surviving partons.

---

## 6. Event generation and campaign design

| change | motivation | numbers / validation |
|---|---|---|
| **Several pT̂ windows per background** (X#152, J#25) | MUSIC_1 doesn't depend on the jet, so one background serves K windows × M jets. | Cost per event ≈ (1 + 1/(K·M))/2. **Campaign c1** (K·M = 15): **336 events/h** vs ~190, 1.8×. Background bit-identical; per-window σ recorded. |
| **Parton rapidity cut** (X#153, J#28) | Only 43–66% of pth10-40 events had their hardest parton at \|y\| < 0.6. | 2.5× more usable 10–20 GeV events per GPU hour. σ = σ_gen × kept/tried, checked to 1.0014. Rejected events are regenerated in ~0.1 ms. |
| **Shared background storage** (J#27) | Under reuse, identical background copies took ~40% of a pair file. | Each background stored once (`arr_bg` is a virtual dataset). ≈ 160 + 148/N MB per event (170 MB at N = 15). Bit-identical reads; FNO4d loader unchanged. |
| **Unique seeds, campaigns, seed registry** (J#22) | FNO4d's `dAu_25ev` was the first 25 events of `dAu_250ev`. Clock seeds duplicated events; Pythia clamps seeds above 9e8. | OS-entropy seeds in 1…9e8, `seeds_used.tsv` under a file lock. 32 concurrent draws all unique. |
| **Independent hadronization seeds per file** (J#21) | Event 0 of every file got the same iSS seed, which correlated files under `--correlated`. | File uuid enters every seed. `--legacy-seeds` reproduces old files bit for bit. |
| **`--particlize-only`** (J#37) | Wake-only campaigns need no pair file. | 25 shared datasets byte-identical; −300 MB/event; peak 7.5 GiB. |

---

## 7. FNO training data

### 7.1 Writers and format

| change | motivation | numbers / validation |
|---|---|---|
| `FastRootBulkWriter` with native and grid modes, `dump_hydro_only` (X#131) | The legacy writer peaked at 24.6 GB (O+O). | Peak 11.0 GB (native) and 15.5 GB (grid). Grid mode bit-identical to the legacy writer. τ-axis fix (X `ab9fe60d`); last-frame fix (X `c99eb7d7`). |
| **`H5BulkWriter`**, pure Python HDF5 (J#4) | ROOT's ~1–2 GB per-object limit: an O+O native event is 1.3–1.9 GB. It also removes a conversion step. | Native output bitwise equal to FastRootBulkWriter; grid output within float32 epsilon. Peak 5.2–6.5 → 3.9 GB. |
| **Blosc-zstd by default, optional `keep_bits`** (J#13) | lzf compressed only 1.1–1.25×. | Pair file 400 → **285 MB lossless**, 159 MB at 12 bits. Wake L2 error: 6.8e-4 at 12 bits, 2e-4 at 14, 6e-5 at 16. 12 bits is fine for the FNO; use 14–16 bits or lossless for precision wake work. |
| FastHydro (J#5, J#6) | Cheap FNO data with physical deposition (Matter+LBT → droplets) instead of test partons. | Agrees with MUSIC to ~0.3% in e on a shared IC. Tuned 0–10%: dN_ch/dη 659 vs MUSIC 667, ⟨pT⟩ within 1–7%. Bulk-viscosity bound vendored from FNO4d. |

### 7.2 Output grid and edge flag

- **Grid** (J#45, `Wake_grid_comparison.md`): **64 × 64 × 32**, x, y ±12.21 fm at 0.3875 fm, η_s ±4.84 at 0.3125. Before: 65 × 65 × 33, ±10 fm at 0.3125. Powers of two suit the FFTs.
- **Why the larger box:** in 3 of 60 events the jet leg's fluid above freeze-out crossed ±10 fm, at τ ≈ 7–14 fm/c, with edge e up to 0.256 GeV/fm³ against e_fo = 0.234. The background never did (edge e ≤ 0.08).
- **Edge flag** (J#44): `diag/{bg,jet}_edge_e_max`, plus `hit_edge` against e_fo, which is taken from the XML and the EoS by default. Cost ~0.01% of an event. Recorded values equal a recomputation from `arr`.
- **What the 0.39 fm grid keeps of the wake** (6 events, offline resampling with the production code):

  | measure | new 0.3875 | old 0.3125 | 0.3 fm shifted (floor) |
  |---|---|---|---|
  | integrated wake | 0.999 [0.94, 1.04] | 1.003 [0.95, 1.11] | 1.000 |
  | shape at ≥ 0.5 fm (error) | 0.096 | 0.109 | 0.083 |
  | peaks | 0.76 | 0.81 | 0.71 |

  The ~30% point-by-point loss is the same on every grid: it comes from cell-scale liquefier structure. The new grid lowers peaks in weak, narrow wakes. If peaks matter, `n = 80` (0.314 fm) is the alternative.

### 7.3 What the stored history can and cannot give (J#23, J#24)

The pair files store e, vx, vy, vz but no π^μν or Π. A study compared hadrons from a surface found in the history against MUSIC's own surface with δf (4 files × 25 events):

| observable | history − MUSIC | main cause |
|---|---|---|
| dN_ch/dη | +0.2% | — |
| p + p̄ | −28.6% | bulk δf |
| ⟨pT⟩ π, K, p | +12%, +12%, +6% | bulk δf |
| v₂, v₃ | +3.3%, +15.4% | shear δf |
| wake | −10% (a scale, r = 0.97) | coarse history; jet-induced freeze-out volume −5% |

**Corrections fitted on half the events (J#24):**
- per-hadron weights close the yields and ⟨pT⟩ to ≤ 0.3%;
- the wake and v₂ are fixed by a scale;
- v₃ needs δf restored at the surface, which is not done.

**Consequence:** an FNO trained on these files predicts the hydro well, but hadronizing its output needs such corrections or stored viscous fields.

---

## 8. Hadron level for the wake analysis

| change | motivation | numbers / validation |
|---|---|---|
| **Particlize files + offline `hadronize.py`** (J#15, X#148) | Hadronize on CPUs, later, as often as needed, from the stored surfaces (with viscous information) and partons. | In-job vs offline bit-identical (1.71 M hadrons). Null test: surfaces identical cell for cell. |
| **iSS speed-up** (X#149, iSS fork `yield_cache`; J#16) | δf recomputed per sample; per-oversample HDF5 appends. | **Both legs, 500 oversamples: 56.5 → 14.6 s** (11.6 s with 5 threads). Bit-identical across 1/5/20 threads. |
| **Compact iSS output** (X#150, J#17) | 2.4 MB per oversample held in memory. | 0.3 MB per oversample; peak 1.4–1.7 GB at 500–2000 oversamples. Bit-identical. |
| **Correlated sampling** (X#151, iSS fork `common_seeds`; J#19) | Jet − background is a small difference of two large, independently sampled numbers. | Philox random numbers keyed by space-time block. **Var(J − B) ×0.12** at \|η\| < 1 (0.09–0.15 per pT bin). Wake energy 35.7 ± 55.7 → 24.1 ± 1.5 GeV. Single legs unchanged (χ²/ndf 0.74). No extra cost. |
| Common-seed sampling errors (J#34) | The wake error was 4–8× too large on c1, and bkg + wake ~√15 too small. | Batch means agree with exact errors within 2%. c1 wake E error ±1.21 → ±0.16 GeV. |
| Background oversampling per window (J#26) | A shared background deserves more samples. | M × `--oversample` per window. |
| Store less (J#18, J#31) | Disk. | 12/8-bit p/x: 58% of the bytes, shifts ≤ 0.14σ. `--charged --no-x`: 39–42%. 837 bit-for-bit checks pass. |
| **Self-contained particlize files** (J#35) | Stage 2 and the analysis shouldn't need the 170–285 MB pair files. | Shower initiators stored; hadronizing without the pair file works. |
| Readers, ROOT export, C++ reader (J#20, J#30, J#33, J#36) | Analyses with the experiments' existing ROOT code. | RNTuple 0.9× the HDF5 size. Macro = Python reader to 1e-12. Shared background binned once. |

**Known limits:**
- ColoredHadronization is unusable with Matter+LBT: 75–91% of partons carry no colour tag.
- Correlated sampling has no local charge conservation and can't be combined with `--oversample-bg`.

---

## 9. Performance and memory (GB10 unless noted)

| stage | before | now |
|---|---|---|
| pair event, one job | 184 s (before X#141) | ~30–36 s (X#141–#146, M#9/#10/#12, J#9) |
| campaign throughput | 68 events/h (1 job) | 118 (1 job) → **197** (`-j 4 --mps`, OMP 5) → 205 (`-j 6`) → **336** (c1: K·M = 15, \|y\| < 0.6) |
| 2 × RTX 3090 (J#49) | — | **~810 events/h** (c1 settings, 16 jobs, MPS per GPU), ~97 GB host |
| peak memory per job | ~20 GiB | **8.7 GiB** pair, 7.5 GiB particlize-only (X#160, M#15, J#38, X#161/J#39 slim `bulk_info`; byte-identical) |
| 4 jobs together | 66–69 GB | **29 GB** (J#42); plan ~12 GB per job |
| `hadronize.py`, both legs, 500 oversamples | 56.5 s, 2.7 GB | ~14 s, 1.6 GB |
| pair file per event | 400 MB (lzf) | 285 MB, 170 MB with a background shared by 15 jets |

The GPU saturates at ~86% busy on the GB10, with a ceiling of ~225 events/h without reuse. Event cost is dominated by the hydro lifetime and the number of strings.

---

## 10. Deployment: containers, launch, transfer

- **Production images** (`utils/Dockerfile.prod`, workflow `docker-prod.yml`): cu126 and cu130 (amd64 + arm64) and cu124 (amd64). Each carries X-SCAPE, MUSIC4GPU, iSS, 3dMCGlauber and PyJetscape, and hadronizes on CPU nodes too.
  **Rebuild pending:**

  | published tag | sm_70 | Pelican | memory fixes | new grid, edge flag |
  |---|---|---|---|---|
  | `cu126`, `cu130` (2026-09-30) | no | no | no | no |
  | `cu124-20260930-d31946c` | yes | yes | no | no |
  | `cu124` = `cu124-20261003-cbc7263` | yes | yes | yes | no |

  A container run on arm64 cu130 agrees with the native build, but not bit for bit (different toolchain): hydro within 6e-4 of the frame maximum, wake 1.1e-3, identical freeze-out steps.
- **Launch:**
  - `run_jobs.sh` (`-j`, `--mps`, campaigns);
  - `slurm_prod_array.sh` (one GPU per task, `--mem=48G` for 4 jobs);
  - **`launch_2gpu.sh`** (J#49): Docker on two GPUs, one MPS daemon per container, disk guard.
- **Transfer:**
  - `js_gcs.py` and `js_osdf.py`: size + CRC32C checks, resumable. The `/fno4hic` issuer bug is fixed in J#43; logins last 15 days.
  - **`upload_follow.py`** (J#49): uploads while a campaign runs and deletes local files only after a verified upload, so a campaign is no longer limited by the local disk (~80 MB/s up against ~57 MB/s produced on the 3090 machine).

---

## 11. Validation and test summary

| kind | what |
|---|---|
| **Bit-identity** (speed and memory changes) | Every performance change was checked byte for byte against a baseline. Exceptions: the resample (1 ulp in 1.4e8 values) and `source/flux` (last bit varies between runs of the old build too). Includes thread counts, in-job vs offline hadronization, shared vs full backgrounds, slim vs full `bulk_info`. |
| **Null tests** | No deposition gives jet leg = background frame for frame and surface cell for cell. Hadron jet − background consistent with zero (+0.85σ in N). |
| **Energy bookkeeping** | Shower/liquefier balance to 7e-7 GeV. Wake/droplets 0.99, correlation 0.97. Hadron-level wake energy matches the deposit within errors. |
| **Cross sections** | Parton-cut σ checked against the uncut run (1.0014); per-window σ recorded. |
| **GPU vs CPU** | Gubser/EOS tests (≤ 6.4e-4), O+O fields (< 1e-3), Au+Au fields (2e-3), observables (§3.4). |
| **Grid** | Wake on 0.3875 vs 0.3125 vs MUSIC's 0.3 fm (§7.2); edge statistics (3 of 60 events beyond ±10 fm). |
| **Model comparisons** | FastHydro vs MUSIC (bulk observables, IC vs transport); FV solver vs MUSIC (0.3% ideal, 1–3% viscous); history vs surface hadrons (§7.3). |
| **Unit tests** | PyJetscape 193 passed / 1 skipped; FastHydro 316; liquefier 9; iSS correlated sampling 98 assertions; transfer tools 33. |

**Productions to treat with care:**
- **Before X#155** (liquefier normalization), X#146 (jet source time step), `a1753327` (EOS) or X#147 (HQ tables): physically affected, and not repairable.
- **Before X#154:** repair with `fix_lbt_double_counting.py`.
- **GPU runs before M#13:** screen for stalled legs (`Maximum allowed time reached`, `*_no_freezeout`).
- **Files on the 65 × 65 × 33 grid:** don't mix with 64 × 64 × 32 files in one training set (check `prod_grid_yaml`).

---

## 12. Open points, by priority

1. **Rebuild the production images** (`gh workflow run docker-prod.yml -f variants=all`). Then update the dated tags in `README_2stage.md` and `README_launch.md` and remove the "rebuild pending" notes.
2. **MUSIC4GPU M#14 (open): CUDA fail-fast.** A GPU without matching kernel code, e.g. a V100 with the published `cu126`, can run "successfully" with fields the GPU never updated. Until it is merged: images with sm_70, and check the logs for `no kernel image`.
3. **FNO data conventions:**
   - `ntau_freezeout` differs by one between fast_data/legacy files and PyJetscape files (`PLAN_consolidate_h5_writer.md`, not implemented);
   - no protection against train/validation overlap in FNO4d;
   - no viscous fields in the pair files (§7.3).
4. **Larger-scale validation still to do:**
   - a long campaign with `--write-particlize both` under the memory fixes;
   - GPU vs CPU with more events and other centralities;
   - the grid study for weak 10–40 GeV wakes and at hadron/FNO level;
   - a full byte comparison of the pair files before and after the edge flag.
5. **Physics caveats:**
   - Holes at large |η_s| can push cells below zero after normalization; MUSIC's reset then adds energy (+23% in 1 of 15 events).
   - The cause of the GPU's 0.2% offset (fp32 vs vacuum reset vs EoS table) is not isolated.
   - The ~10% wake deficit of history-based hadrons is not understood.
6. **Upstreaming:**
   - the iSS fork (`yield_cache`, `common_seeds`) to chunshen1987/iSS;
   - MUSIC4GPU stays pinned to `jhputschke/MUSIC4GPU XSCAPE`;
   - `XSCAPE-2.1.1-RC` and `contrib` need reconciling (#137, #138);
   - the licence question for the GPLv3 liquefier port in FastHydro.
7. **Smaller items:**
   - a MUSIC_2 "count, don't store" mode for particlize-only (~7.5 → ~5 GiB);
   - `run_jobs.sh --gpus` (branch `run_jobs_gpus`);
   - a recompress tool for old lzf files;
   - container validation gaps (amd64 images on hardware, Apptainer, MPS in containers on discrete GPUs, `WITH_ROOT=1`).
8. **Stale statements to fix:**
   - ~~`BuildContainerProd.md`~~: fixed on this branch (status table, Pelican only in images
     built since `03349ea`, no `-gcs` tags published, MUSIC4GPU doesn't stop yet on a GPU
     without code);
   - ~~`README_2stage.md`'s "~100 GB per GPU for `-j 4`"~~: now ~48 GB (fixed on this branch);
   - ~~the jet README's "same peak memory" for `--particlize-only` and its pre-fix memory numbers in section D~~: fixed on this branch.

---

## 13. Where to read more

| topic | document |
|---|---|
| running a production (containers, SLURM, stage 2, analysis) | [`README_2stage.md`](README_2stage.md) |
| the jet production, every option, file layouts | [`prod_AuAu_0_10_jet/README.md`](../contribs/PyJetscape/example/prod_AuAu_0_10_jet/README.md) |
| pair files, particlize files | [`PLAN_pair_h5_music.md`](Plans/PLAN_pair_h5_music.md), [`PLAN_particlize_h5.md`](Plans/PLAN_particlize_h5.md) |
| iSS speed and correlated sampling | [`PLAN_iSS_optim.md`](Plans/PLAN_iSS_optim.md) |
| memory | [`PLAN_slim_bulk_info.md`](Plans/PLAN_slim_bulk_info.md) |
| throughput, MPS, `-j` | [`BENCHMARK_GB10.md`](BENCHMARK_GB10.md), [`BENCHMARK_M3MAX.md`](BENCHMARK_M3MAX.md), [`utils/README_launch.md`](../utils/README_launch.md) |
| HDF5 compression | [`README_h5_optim.md`](README_h5_optim.md) |
| output grid and wake | [`Wake_grid_comparison.md`](Wake_grid_comparison.md) |
| GPU vs CPU | [`MUSIC_CPU_vs_GPU.md`](MUSIC_CPU_vs_GPU.md); MUSIC4GPU [`docs/VacReset_BUG.md`](https://github.com/jhputschke/MUSIC4GPU/blob/XSCAPE/docs/VacReset_BUG.md), [`docs/PORT_GPU.md`](https://github.com/jhputschke/MUSIC4GPU/blob/XSCAPE/docs/PORT_GPU.md), [`README_CUDA.md`](https://github.com/jhputschke/MUSIC4GPU/blob/XSCAPE/README_CUDA.md), [`docs/MUSIC_CPU_vs_GPU.md`](https://github.com/jhputschke/MUSIC4GPU/blob/XSCAPE/docs/MUSIC_CPU_vs_GPU.md) |
| history vs surface hadrons | [`hydro_hist_vs_surface/README.md`](../contribs/PyJetscape/example/hydro_hist_vs_surface/README.md) |
| FastHydro | [`contribs/FastHydro/README.md`](../contribs/FastHydro/README.md) |
| the FNO writer consolidation (not done) | [`PLAN_consolidate_h5_writer.md`](Plans/PLAN_consolidate_h5_writer.md) |
| X-SCAPE fast writers | X-SCAPE `README_BulkFast.md`, `config/FVvsMUSIC/README.md` |
