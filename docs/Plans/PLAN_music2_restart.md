<!-- Design note, written 2026-10-11. Status: implemented on branches music2_restart, checked on
     the GB10 (see "As built" at the end) and on the M3 Max (Metal). -->

# Design: MUSIC_2 starts from MUSIC_1's state (A8 of REPORT_CPU_GPU.md)

## Goal

In the two-stage chain (MUSIC_1 background → Matter + LBT + liquefier → MUSIC_2 with the
droplets) MUSIC_2 recomputes MUSIC_1's evolution step for step until its first droplet
deposits. Those leading steps are bit-identical to the background's: every pair file records
it in `diag/frames_identical`. MUSIC_2 should instead start from MUSIC_1's saved state at the
last step before any droplet can deposit.

Expected saving per jet event, on the 3090 box (report §5):
- **CPU:** −26 core-s. MUSIC_2's four string fills disappear (~6 core-s each), plus its first
  ~30 steps of freeze-out searches, grid copies and packs.
- **GPU:** −6 to −13 % of the jet leg (~30 of ~500 steps).

The output must stay bit-identical.

## What the code says

- **No droplet can deposit before τ_delay.**
  - A CausalLiquefier droplet deposits only in the MUSIC step that contains
    τ_drop + τ_delay (`CausalLiquefier.h`; `HydroSourceJETSCAPE` prunes on exactly that).
    τ_drop ≥ 0, so nothing deposits before τ_delay (`liquefier_tau_delay` = 1.0 fm/c in
    production).
  - In the regression pair file (seed 1, 4 events) the earliest droplets have
    τ_drop = 0.036–0.049 fm/c, so the first deposit is at ~1.04 fm/c in every event.
  - `frames_identical` = 6 output frames (τ = 0.5 … 1.0) in all four events.
  - With τ0 = 0.4 and Δτ = 0.02 that is step 30 of ~500.
  - **Consequence:** one checkpoint per background, at the last freeze-out step before
    τ_delay, serves every jet event. Several checkpoints are not needed.
- **The strings are done long before that.** With `evolve_QCD_string_mode 4` they deposit at
  τ = 0.40 and 0.42 (4 source fills). After that their lists are empty, and after A3 the
  charge lists no longer count.
- **What carries state from one step to the next** (`Evolve::EvolveIt`, `Advance`, `GPUGrid`):
  - **GPU:** `snap_prev` and `snap_curr`. Each is ε, ρ_B, u[4], W^{μν}[14], Π: 21 floats ×
    600,000 cells = 50 MB, so 100 MB for both. `snap_future`, `dwmn`, `qi_out` and the source
    buffer are rewritten every step. The EOS tables are uploaded per run with the same
    parameters.
  - **Host, Evolve:** the step index and τ;
    - the freeze-out bookkeeping: `freezeoutFieldPrev/Curr` (Cornelius' previous check),
      `emax_last_check_` (no-surface mode), `eps_max_cur`, `FO_n{B,Q,S}vsEta_`;
    - the surface cells found so far (the dilute edge has a surface from the first freeze-out
      step);
    - the nonfinite counters, and the host pointer rotation fpPrev/fpCurr/fpNext;
    - `DATA.delta_tau` / `facTau` (changed only by beastMode 2, after step 200).
  - **Host, store:** the frames written so far: one every 5 steps, 19 MB each for 0–10 % Au+Au;
    7 frames by step 30.
  - **Host, Advance:** `gpu_owns_state_`, `host_curr_fresh_` and the other "who holds the
    current state" flags.
- **Lifetimes:**
  - `MUSIC::run_hydro()` makes a new `Evolve` (with its `Advance` and GPU grid) for every run.
    It lives in `evolve_ptr_` until the next run of that instance, so both legs' grids
    coexist already.
  - With `setReuseHydro` MUSIC_1 runs once per background, MUSIC_2 once per jet event. A
    checkpoint owned by MUSIC_1's `MUSIC` object therefore lives exactly as long as it is
    needed.

## Design

### Where the checkpoint is taken

In MUSIC_1's iteration k_c, **right after the freeze-out stage and before `AdvanceRK`**, where
k_c is a freeze-out step (`(it − iFreezeStart) % facTau == 0`).
- At that point the frame of step k_c (if any) is stored and Cornelius has run.
- `freezeoutFieldPrev/Curr` are then exactly the synced host arenas of `snap_prev` /
  `snap_curr` (floats converted to double). They need no storage of their own: MUSIC_2
  rebuilds them from the restored snapshots, as the sync does.
- k_c is the last freeze-out step with τ_{k_c} + Δτ < τ_checkpoint. τ_checkpoint is set by the
  driver, by default τ_delay; MUSIC itself knows nothing about the liquefier.

The checkpoint (new `MusicCheckpoint`, owned by MUSIC_1's `MUSIC` object, replaced at its next
run) holds:

| item | size (0–10 % Au+Au) | where |
|---|---|---|
| `snap_prev`, `snap_curr` | 100 MB | device (CUDA `cudaMalloc`, Metal buffer); a D2D copy, ~0.1 ms |
| it, τ, iFreezeStart, `emax_last_check_`, `eps_max_cur`, `FO_n*vsEta_`, nonfinite counters, fp rotation parity, `delta_tau`, `facTau` | bytes | host |
| surface cells so far (only if MUSIC_1 builds the surface) | ~MB | host |
| the store's frames so far (only if MUSIC_2 stores its evolution) | 7 × 19 MB | host |
| a compatibility key: grid, τ0, Δτ, EOS, viscosity and freeze-out parameters, store cadence, a hash of the string list | bytes | host |

### How MUSIC_2 resumes

`MUSIC::run_hydro_from(const MusicCheckpoint&)`:
1. Builds its own `Evolve` and GPU grid as usual.
2. Checks the compatibility key. On a mismatch it does a full run and logs why.
3. Checks that its own jet source is inactive at every step k < k_c, using the predicate the
   step uses: `prepare_list_for_current_tau_frame(τ_k)`, then
   `has_active_sources_current_tau()`. On the first active step it does a full run and logs
   it. With τ_drop ≥ 0 and τ_checkpoint = τ_delay this never happens, but the check makes
   correctness independent of that argument.
4. Restores the snapshots, the scalars, the surface cells and the frames, and rebuilds the
   freeze-out fields from the snapshots. It sets the Advance flags to "GPU owns the state,
   host stale".
5. Enters the loop at iteration k_c **at `AdvanceRK`**, skipping that iteration's pre-RK work,
   which MUSIC_1 already did. From then on the loop is unchanged.

MUSIC_2's string source still exists and is still asked every step. Its lists are empty from
τ = 0.44 on, exactly as in a full run.

### Wiring

- **MUSIC4GPU:** `MUSIC::set_checkpoint_tau(double)` (MUSIC_1; 0 = off),
  `MUSIC::checkpoint()`, `MUSIC::run_hydro_from(...)`. `GPUPipelines::save_state` /
  `restore_state` for CUDA and Metal. `Evolve::EvolveIt` gets an optional resume state.
- **X-SCAPE:** `MpiMusic::set_checkpoint_tau(double)` on the background, and
  `MpiMusic::set_restart_from(MpiMusic*)` on the jet leg. `EvolveHydro` calls
  `run_hydro_from` when a valid checkpoint exists.
- **js-contrib:**
  - bindings for both;
  - `run_prod_jet.py --music2-restart`, which sets τ_checkpoint from the liquefier's
    τ_delay. It stays off by default until validated on the GB10 and the 3090 box, then
    becomes the default;
  - `PairH5Writer` is unchanged: MUSIC_2's store and `frames_identical` look exactly as
    before.

## Decisions (recommendation first)

1. **Checkpoint on the device, not the host.** 100 MB of GPU memory per job against 100 MB of
   host memory plus a D2H/H2D copy. On the 3090 box (24 GB, 10 jobs per GPU) GPU memory has
   room; host memory peaked at 104 of 128 GB. On the GB10 and the M3 Max memory is unified
   anyway.
2. **The jet leg's leading frames are copied from MUSIC_1's store (MUSIC side).** The other
   option is for `PairH5Writer` to take them from the resampled background, which saves the
   ~130 MB copy but changes MUSIC_2's store (it would start at τ_c) and the writer's frame
   indexing. The copy keeps every consumer unchanged. It is skipped when MUSIC_2 stores no
   evolution (A9, `--particlize-only`).
3. **When MUSIC_1 builds no surface but MUSIC_2 does** (`--write-particlize jet`, where the
   XML switches MUSIC_1's surface off): MUSIC_1 runs Cornelius up to k_c anyway, keeping the
   cells for the checkpoint only and exporting none. That is ~6 searches, ~0.4 s per
   background, against ~0.4 s saved per jet event.
   - The alternative is to restart only when both legs have the same surface setting. That
     drops the `jet` mode.
   - The reverse case (MUSIC_1 builds a surface, MUSIC_2 doesn't) needs only
     `emax_last_check_`, which every checkpoint has.
4. **MUSIC's diagnostics before k_c** (the vs-τ / vs-η_s `.dat` files, off in production) are
   not repeated by MUSIC_2. That is logged, not an error.

## Verification

- **`MUSIC_RESTART_VERIFY=1`:** MUSIC_2 runs from τ0 as today. When it reaches k_c it compares
  its own state with the checkpoint, bit for bit, and reports every difference:
  - the snapshots;
  - the scalars;
  - the surface cells;
  - the store frames.

  This checks the save/restore without changing any output, and can run inside a production
  job.
- **End to end:** `run_prod_jet.py` with and without `--music2-restart`, every dataset of the
  pair and particlize files compared:
  - `--reuse 2`, `--write-particlize both`;
  - `jet`;
  - `--particlize-only`;
  - `MUSIC_CUDA_FORCE_DISCRETE=1`;
  - a few hundred events on the 3090 box.
- **Metal:** compiled and run on the M3 Max before the default changes.

## Steps

1. MUSIC4GPU: `MusicCheckpoint`, save in `EvolveIt`, the verify mode, CUDA save/restore. On
   the GB10, verify mode first, then a resume run compared end to end.
2. MUSIC4GPU: Metal save/restore.
3. X-SCAPE: the two `MpiMusic` setters and the resume path in `EvolveHydro`; `get_music4gpu.sh`
   re-pinned.
4. js-contrib: the bindings, `--music2-restart`, a test with stubs, and timings in this note.
5. One PR per repo, against `XSCAPE`, `contrib` and `main`.

## Open points to settle while building

- Whether any other device buffer carries state across steps (`qi_source_buf`, the RK
  rotation's scratch). The verify mode will show it.
- The equal-τ surface at `it == iFreezeStart` must lie before k_c, so that its cells are in
  the checkpoint. Check iFreezeStart in production (string `tauStart` max, ~0.42 fm/c).
- `hit_grid_boundary` / `reRunHydro` set before k_c. MUSIC_1 would have stopped, so no
  checkpoint exists in that case.

## As built (2026-10-11, GB10)

Commits: MUSIC4GPU `1303079` (`MusicCheckpoint`); X-SCAPE `28d92328` (`MpiMusic` setters) and
`1bff0d15` (pin); js-contrib `run_prod_jet.py --music2-restart`, `PairH5Writer.liquefier_tau_delay`
and the bindings.

**Differences from the design above:**
- **Decision 3 is not built.** With `--write-particlize jet` (MUSIC_2 builds the surface, MUSIC_1
  doesn't), MUSIC_2 refuses the checkpoint ("this run builds the freeze-out surface, the
  checkpoint's run did not") and runs from τ0. The driver doesn't switch the restart on in that
  mode and says so. Running Cornelius up to the checkpoint in MUSIC_1 would let that mode gain
  too; it's left for later.
- **The checkpoint's step** is the last freeze-out step whose RK step ends by τ_delay. In
  production that is step 28, τ = 0.96 fm/c, with τ0 = 0.4, Δτ = 0.02, iFreezeStart 3 and
  facTau 5.
- **What the checkpoint held there:** 96 MB of GPU state, 76,994 surface cells, and 6 frames
  (109 MB).
- **The key:**
  - every `InitData` field that shapes the evolution, as hexfloats;
  - ε_FO;
  - the initial-state source's count, fingerprint (a hash of the string list) and τ range;
  - a hash of the initial arena.

  A mismatch names the first differing field.
- **What `MUSIC_RESTART_VERIFY=1` checks:** that the jet leg, run from τ0, reaches the
  checkpoint's step in the checkpoint's state:
  - the GPU state;
  - the surface;
  - the counters;
  - max e;
  - the frames.

  That the restore is complete is shown by the end-to-end comparison below.

**Checks** (`run_prod_jet.py --events 4 --seed 1 --reuse 2`, compared dataset by dataset with
the same job without the option):

| setting | MUSIC_2 | output | job CPU (user) | wall |
|---|---|---|---|---|
| `MUSIC_RESTART_VERIFY=1`, `--write-particlize both` | 4 × "identical to the checkpoint" | identical | | |
| `--write-particlize both` | 4 × started at step 28 | identical | 337 → 239 s | 99.0 → 92.9 s |
| `--particlize-only` | 4 × started at step 28 | identical | 325 → 229 s | 87.9 → 82.0 s |
| `MUSIC_CUDA_FORCE_DISCRETE=1`, `both` | 4 × started at step 28 | identical | 241 s | 95.4 s |
| `--write-particlize jet` (before the driver guard) | 4 × from τ0, reason logged | identical | 307 → 307 s | |
| `--write-particlize jet` (with the guard) | "restart OFF", reason printed | identical | | |

On this build without the option, hydro-only Au+Au and O+O are identical to the phase 2 build.

**Saving:** ~25 core-s per jet event, as the report estimated; −8 to −10 % wall per jet event.
The GB10 is GPU-limited, so events/h there rise only with the GPU part (28 of ~500 steps).

**Still to do:**
- ~~Metal: compile and run on the M3 Max.~~ Done 2026-10-11, reported by the user. The test
  task of commit `6366159` (`docs/Plans/TEST_M3MAX_music2_restart.md`, removed afterwards)
  passed:
  - the Metal build;
  - the unit tests (including `test_resample.py` with the Mac's numpy and BLAS);
  - the restart and verify-mode checks;
  - the dataset-by-dataset comparison with the code before the PRs.
- ~~The 3090 box: events/h with the amd64 image.~~ Done 2026-10-12 with
  `xscape-prod:cu124-music2restart`, see
  [`BENCHMARK_2x3090.md`](../BENCHMARK_2x3090.md#music2restart):
  - two-stage campaign at 10 jobs/GPU: 796 → 1,121 events/h (+41 %; 965 with
    `--no-music2-restart`); `--particlize-only` 917 → 1,237; hydro only 1,223 → 1,601;
  - every jet leg started at step 28; output identical to the old image and between restart
    and no restart (300 events), except `source/flux` at ≤ 7 ulp, which differs run to run
    anyway.
- **Default on** since the documentation commit (2026-10-11), at the user's request, before
  the M3 Max and 3090 checks. `--no-music2-restart` turns it off. It is off by itself with
  `--write-particlize jet` / `--surface jet` and on builds without the restart, decided from
  the options before the files are opened, so `prod_music2_restart` records what ran. The
  default run (4 events, `--reuse 2`, `both`) restarted in every event; output identical.
- Documented in `prod_AuAu_0_10_jet/README.md` (*MUSIC_2 from MUSIC_1's state*),
  `docs/README_2stage.md` §2, the PyJetscape README, `utils/README_launch.md`, X-SCAPE's README
  and MUSIC4GPU's README.

## runJetscape: the same from XML (2026-10-24, X-SCAPE `music_xml_restart`)

The Python driver sets the restart and the jet leg's storage after `Init()`; X-SCAPE
`music_xml_restart` (c2416c77) makes both XML settings, so `runJetscape` runs a two-stage job
the same way. In the jet leg's own block:

```xml
<Hydro>
  <AddLiquefier> true </AddLiquefier>
  <MUSIC>
    <name>MUSIC_2</name>
    <restart_from>MUSIC_1</restart_from>   <!-- start from MUSIC_1's checkpoint -->
    <dump_hydro_only>1</dump_hydro_only>   <!-- no framework copy of the jet leg -->
    <store_evolution>0</store_evolution>   <!-- and no native frames -->
    <skip_surface>0</skip_surface>         <!-- export the surface (iSS on MUSIC_2) -->
  </MUSIC>
</Hydro>
```

- `dump_hydro_only`, `skip_surface` and the new `store_evolution` are read from the first
  `<Hydro><MUSIC>` block for every instance and overridden in an instance's own block, as
  `freeze_out_surface` already was.
- `restart_from` and `checkpoint_tau` are read only from an instance's own block. The
  instance named in `restart_from` keeps its checkpoint at its `<checkpoint_tau>`, by default
  (0) at `<Liquefier><CausalLiquefier><tau_delay>`, as `restart_legs` does. Main-XML defaults
  `none` / `0` are off.
- The Python setters still override the XML: `run_prod_jet.py` is unchanged (the PyJetscape
  tests touching it: 120 passed).

**Check (GB10, 0-10 % Au+Au, PythiaGun 50-70, seed 4242, 3 events, iSS (20 samples) +
Colorless on MUSIC_2, FinalStateHadrons writer):**

| runJetscape | final-state hadrons | peak RSS | wall |
|---|---|---|---|
| T0: as before (MUSIC_2 keeps its frames) | reference | 15.9 GB | 48.6 s |
| T1: + `dump_hydro_only` 1, `store_evolution` 0, `skip_surface` per leg | identical | 6.3 GB | 45.9 s |
| T2: T1 + `restart_from` MUSIC_1 | identical | 6.5 GB | 42.6 s |

T2 started every jet leg from the checkpoint at step 28 (τ = 0.96 fm/c), as the Python driver.
`run_prod_jet.py --write-particlize both --particlize-only --validate-inline` with the same seed
(iSS 100 samples, plus the particlize file): 49.4 s, 7.0 GB. Its events differ from
runJetscape's from event 1 on, also against T0, so this is not the restart; the same seed does
not give the same events in the two drivers.
