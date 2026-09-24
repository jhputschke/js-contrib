<!-- Archived plan, written 2026-09-24. Status: proposed, not yet implemented.
     Working copy: ~/.claude/plans/provide-a-plan-with-hazy-bubble.md -->

# Plan: two-stage MUSIC pair production (prod_AuAu_0_10 with jet deposition)

## Context

`prod_AuAu_0_10` (PyJetscape) produces 0–10% Au+Au hydro evolutions (3D MC-Glauber strings →
MUSIC on the GPU) straight to FNO4d HDF5. We want the same production **with a jet**: the X-SCAPE
two-stage hydro (MUSIC_1 background → Matter+LBT+CausalLiquefier → MUSIC_2 with the deposition),
written as a **pair** in FastHydro's layout (`arr` = jet leg, `arr_bg` = background,
`source/droplets`, `shower/`) by PyJetscape's writer. The writer keeps its own extras: streaming
frame by frame, a user output grid, and a growable τ axis with repad.

Three things stand in the way today:
1. **MUSIC holds only one source.** With InitialProfile 131, the strings overwrite the liquefier.
   - CPU MUSIC: fixed by MUSIC `cee9460` + X-SCAPE PR #138.
   - music4gpu: has no jet slot yet, so `build_gpu` breaks once #138 is in.
2. **The writer is single-leg.** PyJetscape's writer writes one evolution from the first hydro only.
   The pair layout lives in FastHydro's `PairedH5Writer`, which PyJetscape cannot import (FastHydro
   imports jetscape, so that would be circular).
3. **No production driver** exists for the two-stage MUSIC run.

## Decisions

**Settled with the user (2026-09-24):**
- **Source:** droplets only. `source/droplets` + `offsets` are stored, there is no `source/S`, and
  `has_source=false`.
- **Background reuse:** optional (`--reuse N`), default 1. `arr_bg` is still written per event so
  it stays aligned with `arr`, and a `diag/bg_id` records which background each event used.
- **Hard process:** PythiaGun by default, PGun behind a flag.

**Chosen by me (from exploration):**
- **Freeze-out convention:** PyJetscape's "frames written", stamped as
  `freezeout_convention_id="frames_written"`. This is D1 of `FastHydro/PLAN_consolidate_h5_writer.md`.
- **Same τ length for both legs:** `arr_bg.shape == arr.shape` always. PairBrowser and
  `wake_pyvista.py` require it. The τ axis grows to the longer leg, and the shorter leg's tail is 0.
- **Background read from the framework copy.** Matter/LBT need MUSIC_1's `bulk_info`, and copying
  into it frees the native store. The cells are bit-identical to a native read: both go through
  `get_fluid_cell_with_index`.
- **Deposition leg read from its native store,** via a per-instance `set_dump_hydro_only(True)`
  called after `JetScape.Init()`. Both legs share the first `<Hydro><MUSIC>` block in the XML.
- **API names follow `PLAN_consolidate_h5_writer.md`** (`add_evolution`, `ragged`/`RaggedGroup`,
  the `tau_axis` attribute), so a later consolidation is a drop-in.

## Branches today (checked with git)

**X-SCAPE**
| branch | state |
|---|---|
| `main` (`7c9fc39e`) | no music4gpu support at all |
| `origin/music4gpu_test` (`b42b3e18`, PR #136 merge) | an ancestor of `fasthydro_hadronization` |
| local `music4gpu_test` | `b42b3e18` + 3 unpushed commits, the same patches as `fasthydro_hadronization`'s `f25ba028`, `73996c07`, `c99eb7d7` (`git cherry` marks all three `-`) |
| `fasthydro_hadronization` (`c99eb7d7`) | `origin/music4gpu_test` + 6 commits: soft particlization/SurfaceFinder `88099402`, JetScapeWriter destructor `9b3641b5`, CMake OpenMP `3955cef6`, plus the 3 above |
| `KoKKos-Music-Port` (`09943072`) | older base + 2 commits adding `USE_KOKKOS`; not needed for CUDA |

- Test merges are all clean: `fasthydro_hadronization` → `music4gpu_test`, PR #138 → either
  branch, `KoKKos-Music-Port` → `fasthydro_hadronization`.
- Every branch clones MUSIC4GPU with `-b XSCAPE`, unpinned.
- `build_gpu` is `USE_CUDA=ON`. Its `USE_KOKKOS=OFF` cache entry is left over from a
  `KoKKos-Music-Port` configure.

**MUSIC4GPU**
| branch | state |
|---|---|
| `XSCAPE` (local `a991a36`, origin +1 README commit `4fe539b`) | what X-SCAPE clones |
| `KoKKos-Port` (`a11553f`, **the current checkout**) | `XSCAPE` + 13 Kokkos commits |
| `origin/XSCAPE-KoKKos` | `KoKKos-Port` merged in (PR #8) |

- The Kokkos commits don't touch the CUDA code. In shared files they change only the
  backend-selection block at the top of `src/advance.h`, plus CMake and new `src/gpu/*kokkos*` files.
- So switching the checkout to `XSCAPE` gives the same CUDA physics. The jet-source port touches
  other parts of `advance.h`, so it could be cherry-picked onto a Kokkos branch without
  conflicts if that is ever wanted.

**js-contrib:** `main` (`26bd201`) already contains `fasthydro_hadronization` and `fasthydro`.

## Phase 1: Branch setup (merge fasthydro_hadronization *before*: effectively a fast-forward)

1. **X-SCAPE:** check out `music4gpu_test` and run `git rebase fasthydro_hadronization`.
   - Git drops the 3 unpushed duplicate commits, so `music4gpu_test` = `c99eb7d7`.
   - Pushing is a fast-forward of `origin/music4gpu_test`; no force push.
   - Only unpushed duplicates are rewritten, and a merge commit would work just as well.
   - Afterwards `music4gpu_test` contains all of `fasthydro_hadronization`, so any later
     FastHydro commits merge in cleanly.
2. **X-SCAPE:** merge PR #138 into `music4gpu_test`. Push `music4gpu_test`. That completes the
   consolidation.
3. **X-SCAPE:** create a new branch `pair_h5_music` from the consolidated `music4gpu_test`.
   **Every further X-SCAPE change goes there** (e.g. the `get_music4gpu.sh` pin in 2b), and
   `music4gpu_test` stays as the consolidated base.
4. **MUSIC4GPU:** switch the checkout from `KoKKos-Port` to `XSCAPE` and pull (1 README commit).
   Rebuild `build_gpu` from X-SCAPE `pair_h5_music`.
   - Check that nothing changed: rerun a `prod_AuAu_0_10` seed, 1 event, and compare bit for bit
     with an existing output file.
5. **Leave alone:** X-SCAPE `KoKKos-Music-Port` stays separate, like the MUSIC4GPU Kokkos
   branches; no merges.
6. **js-contrib:** new branch `pair_h5_music` from `main`, with the same name as the X-SCAPE branch.
   Its first commit is this plan, archived as `contribs/PyJetscape/PLAN_pair_h5_music.md` (already
   written, untracked). Keep it up to date as phases land.

## Phase 2: MUSIC source fix (C++)

**2a. MUSIC (CPU).** PR #138 is already in, via Phase 1. Check out `external_packages/music` at
`cee9460` for the CPU build.

**2b. MUSIC4GPU.** Port MUSIC `c8da6a3` to music4gpu.
- **Where:** a branch from `XSCAPE`, e.g. `XSCAPE_jet_source`, merged into `XSCAPE`.
- **Kokkos branches: left alone.** `XSCAPE` and the Kokkos branches (`KoKKos-Port`,
  `XSCAPE-KoKKos`) stay separate, with no merges in either direction (user decision,
  2026-09-24). If a Kokkos build ever needs jets, cherry-pick only the port commit onto it.
  Until then, don't build an X-SCAPE branch carrying PR #138 against a Kokkos branch:
  the wrapper calls `add_hydro_source_terms_from_jet`, which exists only with the port.
- **Pin:** on X-SCAPE `pair_h5_music`, set `get_music4gpu.sh` to that commit, the way
  `get_music.sh` pins MUSIC. Today it clones the branch head, so builds can't be reproduced.
- `src/music.{h,cpp}`: add `hydro_source_terms_from_jet_ptr_` and `add_hydro_source_terms_from_jet()`.
  Pass the pointer to `Evolve` in `run_hydro()` and `prepare_run_hydro_one_time_step()`.
- `src/evolve.{h,cpp}`: add a constructor argument and forward it to `Advance`.
- `src/advance.{h,cpp}`:
  - add `jet_sources_active()` = pointer non-null && `get_number_of_sources() > 0`. Check it per
    step, not in the constructor: in the time-stepped mode `Advance` is built before any droplets
    exist. This also skips the per-cell zero calls in background and hydro-only runs.
  - CPU fallback `FirstRKStepT` (~:803): add the second source, as upstream does.
  - `prefill_hydro_source_on_cpu` (:352): add the jet source's j^μ to each cell's value in `qi_source_buf`.
  - `try_gpu_advance` (:524): run the pre-pass if `flag_add_hydro_source || jet_sources_active()`.
  - `gpu_features_supported` (:344): the QS rejection applies to either source.
- This one change covers CUDA, Metal and Kokkos, because all three read `qi_source_buf`.

**2c. PyJetscape binding.** In `src/bind_music.cc`, bind the existing C++ `set_dump_hydro_only`
and `set_skip_surface` (`MusicWrapper.h:268,275`). Document that they must be called after
`JetScape.Init()`, because `InitializeHydro` rereads the XML then.

**Gate:** a 1-event two-stage run on 131, in both the CPU and GPU builds, gives
max|e_MUSIC_2 − e_MUSIC_1| > 0.

## Phase 3: PyJetscape writer core — `python/jetscape/fno_h5_writer.py`

It stays stdlib + numpy + h5py only, and importable on its own (required by `repad_h5.py` and
`test_h5_bulk.py:574`). Everything below is additive; `H5BulkWriter` files are unchanged except
for new attributes.
- **`add_evolution(name, *, fo_suffix)`:** a second dataset with the same shape, chunks, maxshape
  and compression as `arr`, plus `ntau_freezeout{sfx}` and `tau_freezeout{sfx}`.
  `ensure_capacity` grows every evolution together.
- **`write_frame(i, t, frame, dataset="arr")` and `set_event_meta(..., dataset="arr")`:** only the
  `arr` call advances `nevents_written` and flushes.
- **`ragged(group, offsets_name, fields, *, columns=None, unit="event")` → `RaggedGroup.append(rows)`:**
  - writes as it goes (`maxshape=(None,…)`, gzip-4 + shuffle, ~64k-row chunks).
  - stores column names as `<field>_columns`; fields in a group share one offsets vector.
  - an empty event is a repeated offset, not a missing one.
  - with `unit="event"`, the append count is checked at `set_event_meta`.
- **`write_diag(i, **scalars)`:** one growable `diag/<key>` per key, default NaN, uint64 for
  `*seed`. Written per event, so a crash loses nothing (FastHydro writes at close).
- **Attributes:**
  - `tau_axis` on every dataset with a τ axis.
  - root: `freezeout_convention_id`, `has_source`, and `has_shower` when that group exists.
- **`repad_to`:** find τ datasets by their `tau_axis` attribute. Old files fall back to
  `_TAU_DATASETS` plus `("arr_bg", 5)`, which fixes the repad bug that splits a pair.

## Phase 4: Move the hydro-independent capture into PyJetscape

- **`python/jetscape/showers.py`:** move `fasthydro/showers.py` over unchanged (numpy only,
  duck-typed on the manager). `fasthydro/showers.py` then re-exports from `jetscape.showers`.
- **`python/jetscape/liquefier_io.py`:**
  - `droplets(liq)` → `liq.droplets_numpy()`, an (M,8) array.
  - `liquefier_params(liq)` → `liq.params()` plus `c_diff` and `gamma_relax`.
  - `DROPLET_COLUMNS`.
  - FastHydro's `params_from_liquefier` builds on it.

## Phase 5: `python/jetscape/pair_h5.py` → `PairH5Writer`

A sibling of `H5BulkWriter`, driven by hand per event the way `run_prod.py` drives the single-leg writer.

**Constructor:** `out_file_name, bg_id="MUSIC_1", jet_id="MUSIC_2", grid_mode ("grid"|"native"),
out_grid, tau_stride, choose_ntau, compression, store_showers=True, extra_attrs, force`.

**`attach(jetscape)`, after `Init()`:**
- finds both legs by `GetId()` in `GetTaskList()`.
- requires that the background does not have `get_dump_hydro_only()` set.
- sets `set_dump_hydro_only(True)` on the jet leg and `set_skip_surface(True)` on both.
- `liq = jet.get_liquefier()`; if there is none, the run is in deposition-off (null test) mode.
- takes the shower manager from `JetScapeSignalManager.GetJetEnergyLossManagerPointer()`.

**`Exec()`, per event:**
1. **Read both legs.** `bulk_sources.event_array(bg, "framework", out_spec=grid)` for the
   background and `event_array(jet, grid_mode, out_spec=grid)` for the jet leg. Both go through the
   same `resample`. The spatial grid and `tau_min` must match; otherwise skip the event with a warning.
2. **Write the evolutions.** Frame by frame, jet leg → `arr`, background → `arr_bg`, with
   per-leg `ntau_freezeout[_bg]` and `tau_freezeout[_bg]` (τ end on the source grid).
3. **Write the ragged groups.** `source/droplets` + `offsets`; `shower/partons|vertices|initiators`,
   with the attribute texts copied from `fasthydro/h5_writer.py:233-252`.
4. **Write the diagnostics:**
   - `n_droplets`, `E_droplets`, `E_droplets_late` (energy of droplets after the jet leg froze
     out, which MUSIC's stop drops).
   - `n_showers`, `n_partons`, `tau0_music`, `ntau` per leg, `bg_id`.
   - `frames_identical`: the number of leading frames where `arr == arr_bg` bit for bit. It must
     be at least 1, and it replaces FastHydro's `ic_sha256` check.
5. **Clean up.** `jet.clear_hydro_info_from_memory()`. In deposition-off mode also call
   `liq.ClearTask()`, since no hydro with a liquefier clears the droplets then.

**First event writes the attributes:**
- `attrs_from_grids` + `music_extra_attrs`.
- provenance: `generator="xscape/MUSIC"`, `producer`, `pairing="bg_jet"`, `arr_is`, `arr_bg_is`,
  `source_model="causal_liquefier, point-sampled by MUSIC"`, `source_mode="xscape"`,
  `hard_vertex`, `eos_kind`, `transport_mode`.
- `liquefier_{dtau,tau_delay,time_relax,d_diff,width_delta,c_diff,gamma_relax}`.

**Also:** `Finish()` can be called twice safely, and the writer works as a context manager. No
`eos/` group in v1 (follow-up), so DiffBrowser's `mach_angle` is unavailable.

## Phase 6: Production `example/prod_AuAu_0_10_jet/`

**`AuAu_MCGlauber_MUSIC_0_10_jet.xml`** is derived from the prod XML: same MCGlauber (b_max 4.7),
IS grid, PreEq (NullPreDynamics, `evolutionInMemory 0`) and MUSIC physics.
- First `<Hydro>`: `MUSIC_1` with all MUSIC settings, `dump_hydro_only 0`,
  `output_evolution_to_memory 1`, `output_evolution_every_N_timesteps 5`, `skip_surface 1`.
- `<Hard><PythiaGun>`: eCM 200, pTHat range set in the XML.
- `<Liquefier><CausalLiquefier>`, and `<Eloss>` with Matter + LBT and `<AddLiquefier>true`.
  Parameters taken from `FastHydro/config/AuAu_FastHydro_tune_0_10_wake.xml`.
- Second `<Hydro>`: `<AddLiquefier>true` with `MUSIC_2`.

**`run_prod_jet.py`** keeps the CLI of `run_prod.py` and adds:
- `--hard {pythia,pgun}` with `--pgun-pt` (PGun edits the job XML and warns that the vertex is at
  the origin), `--reuse N`, and `--no-deposit` (sets MUSIC_2 `AddLiquefier false`, for the null test).
- It imports `load_grid_yaml`, `music_box`, `check_inside`, `describe` and `check_env` from
  `../prod_AuAu_0_10/run_prod.py` rather than copying them. The default grid is
  `../prod_AuAu_0_10/grid_fno.yaml`.
- **XML guard:**
  - exactly two `<Hydro>` blocks, named MUSIC_1 and MUSIC_2.
  - the first MUSIC has `dump_hydro_only 0`, and `evolutionInMemory` is 0.
  - `Eloss` and `Liquefier` are present.
  - `SoftParticlization`, `Afterburner`, `RootBulkWriter` and `FastRootBulkWriter` are rejected.
- **Loop:** as in `run_prod.py:262-304`: `Init` → `writer.attach` → `ExecInit` → per event
  `ExecPerEvent` → `writer.Exec` → `ClearPerEvent`.
- **Output:** the provenance of `run_prod.py:243-255` plus `prod_hard` and `prod_reuse`;
  `diag/wall_s`; a JSON summary; the file `AuAu_0_10_jet_seedNNNN.h5`.

**Also:**
- `run_jobs.sh` (prod_AuAu_0_10): take `PROD_SCRIPT` and `TAG_PREFIX` from the environment, with
  today's defaults. The jet folder gets a 3-line wrapper around it.
- A folder `README.md`: layout table, measured time and memory, PairBrowser usage, limitations.

## Phase 7: Small FastHydro follow-ups

- `fasthydro/browse.py` PairBrowser: read `freezeout_convention_id` ("frames_written" →
  `live = ntau_fo`, otherwise `ntau_fo − 1`). Override it in `browse.py`, because
  `fast_data/viz.py` is vendored and never patched.
- Point FastHydro README's "Possible extension: MUSIC two-stage" section at `PairH5Writer`.

## Verification

**Unit tests (no compiled core):**
- PyJetscape `tests/test_h5_bulk.py` stays green.
- New `tests/test_pair_h5.py`:
  - legs of different τ length → same shape, per-leg freeze-out values, zero tails.
  - ragged groups aligned, including an empty event.
  - one diag row per event.
  - `repad_to` grows `arr` and `arr_bg` together, including the fallback for old files.
- FastHydro `test_showers.py` and `test_h5_output.py` stay green.
- New FastHydro test: PairBrowser opens a PyJetscape pair file, `diff == arr − arr_bg`, and `live`
  follows the convention attribute.

**Integration** (`build_gpu`, `conda activate js_fno`, GB10):
1. **Null test:** `--no-deposit`, 1 event → `arr == arr_bg` bit for bit, `frames_identical == ntau`.
2. **Deposition:** 1 event PythiaGun →
   - `n_droplets > 0` and max|Δe| > 0.
   - frames before the first droplet τ are identical.
   - Δe is localized near the droplet positions (PairBrowser plot).
3. **Same background as the hydro-only production:** `arr_bg` equals `prod_AuAu_0_10`'s `arr` for
   the same seed and grid, on event 0.
4. **CPU vs GPU:** small grid, 1 event, same seed; the Δe maps agree to a relative 1e-3. This
   checks the music4gpu port against upstream MUSIC.
5. **Reuse:** `--reuse 3`, 3 events → `bg_id = [0,0,0]`, `arr_bg` identical across events, `arr` differs.
6. **Measure and record** (in the README): wall time per event for MUSIC_1 and MUSIC_2, the CPU
   source pass's share, peak memory. If the source pass dominates, open the follow-up that moves
   it to the GPU.

## Known limitations (documented, not fixed)

- MUSIC stops at freeze-out once the string sources end, so later droplets are dropped
  (`diag/E_droplets_late` shows how much).
- The MUSIC update makes Trento runs (profile 42) stop at freeze-out instead of running to 30 fm.
- No `source/S` and no `eos/` group.
- The liquefier source is computed on the CPU, cells × droplets per step.

## Where the work lands

- **X-SCAPE `music4gpu_test`:** the consolidated base: `fasthydro_hadronization` + PR #138
  (Phase 1). Nothing else is committed there.
- **X-SCAPE `pair_h5_music`** (from the consolidated `music4gpu_test`): every further X-SCAPE
  change, such as the `get_music4gpu.sh` pin. Merge it into `music4gpu_test` when verified; later
  a PR goes `music4gpu_test` → `main`. `fasthydro_hadronization` never needs a separate merge,
  because it is already contained.
- **MUSIC4GPU `XSCAPE`:** the jet-source port (2b). The Kokkos branches stay separate (no merges).
- **js-contrib `pair_h5_music`** (from `main`): the Python work, 2c and Phases 3–7. Merge to
  `main` when verified.
- One commit per phase. Phase 2's gate has to pass before Phases 5 and 6 run on real data.

## Status (2026-09-24, evening)

**Done and committed**
| piece | where | commit |
|---|---|---|
| consolidation | X-SCAPE `music4gpu_test` = `fasthydro_hadronization` (rebased) + PR #138 merge `4f5bdb2a` | pushed |
| MusicWrapper boundary fix | X-SCAPE `pair_h5_music`: clear MUSIC's `reRunHydro` per event, warn, `get_hit_grid_boundary()` | `ca8dd84a`, pushed |
| music4gpu pin | X-SCAPE `pair_h5_music`: `get_music4gpu.sh` checks out MUSIC4GPU `b9cc8be`, and also fetches the EOS 9 table | `3046389a`, pushed |
| music4gpu jet source slot | MUSIC4GPU `b9cc8be` (committed by the user) | pushed |
| droplet pruning | X-SCAPE `896e3d1c` (`LiquefierBase::prepare_active_droplets`, `CausalLiquefier::droplet_may_contribute`, `HydroSourceJETSCAPE::prepare_list_for_current_tau_frame`, pin moved to `3037be7`) and MUSIC4GPU `3037be7` (Evolve calls the hook for the jet source) | local; push MUSIC4GPU `XSCAPE` first, then X-SCAPE `pair_h5_music` |
| writer core, capture, `PairH5Writer`, production, FastHydro | js-contrib `pair_h5_music` | `0c55de0` … `5d8a5c5`, plus this commit |
| bindings | `set_dump_hydro_only`, `set_skip_surface`, `get_hit_grid_boundary` | compiled and used |

**MUSIC4GPU:** the port is `b9cc8be`, and local `XSCAPE` is one commit ahead of
`origin/XSCAPE`. Push it (`git -C external_packages/music4gpu push origin XSCAPE`) so that
`get_music4gpu.sh`'s pin resolves on a fresh clone. The Kokkos branches stay separate.

**Verified on MUSIC** (`build_gpu`, GB10, 3D MC-Glauber, InitialProfile 131)
- **Hydro-only production unchanged.** music4gpu `XSCAPE` + the port reproduces the existing
  `prod_AuAu_0_10` file bit for bit, both with and without PR #138.
- **Null test** (`--no-deposit`, seed 1, full grid): all 95 frames of `arr` and `arr_bg` are identical.
- **Deposition** (seed 1, full grid, PythiaGun 50–70 GeV, 25 droplets):
  - the legs are identical up to τ = 1.0 and separate at τ = 1.1, the first deposit time.
  - the wake starts near the droplets and moves outward.
  - the jet leg lives 106 frames against the background's 95.
- **CPU vs GPU** (`MUSIC_FORCE_CPU=1`, small grid, seed 2): same 19 droplets to 1e-5 fm; the Δe
  maps have correlation ≥ 0.99999 and differ by 1e-4 to 5e-3 of the peak Δe, at the level of the
  background's own CPU/GPU difference (1e-3 to 3e-3).
- **Reuse** (`--reuse 3`): `bg_id` = [0,0,0], `arr_bg` identical, `arr` differs. Without reuse:
  `bg_id` = [0,1,2].
- **Measured cost** (full grid): null test 58 s/event; with deposition 184 s/event before the
  droplet pruning and **60 s/event after it**, with bit-identical output. Peak memory 16 GB.

**Found along the way**
- **Sticky grid-boundary flag.** Once MUSIC's freeze-out surface reached the grid edge, every
  later event of that MUSIC instance was truncated. Fixed in `ca8dd84a`; the writers record
  `diag/{bg,jet}_hit_boundary` (pair files) and `diag/hit_grid_boundary` (single-leg files).
- **Same seed, different event.** The jet XML's extra modules draw from the framework's random
  stream first, so seed N does not reproduce the `prod_AuAu_0_10` event (plan check 3 dropped).
- **Open: one anomalous run in ten.** Once, right after a rebuild, a hydro-only run gave a
  different freeze-out energy density (e_fo 0.2430 vs 0.2341 GeV/fm³ at T = 0.15) and so a
  different evolution. It could not be reproduced in 9 further runs, and the EOS files are
  unchanged. This suggests uninitialized memory in MUSIC's initialization; it needs a separate
  look. The pair null test and `diag/frames_identical` would catch it within a pair.

**Still open**
- CPU-only X-SCAPE build (MUSIC `cee9460`) not built or tested. music4gpu's forced-CPU path
  covers the same code.
- The one-in-ten anomalous run (see above).
- Speed, next: start MUSIC_2 from a full-state MUSIC_1 snapshot taken just before
  `tau_delay` (~4–5 s/event), then the writer's resampling (12.2 s) and the medium-store
  copy (6.2 s). Details in the PyJetscape README, "Future steps".

**Freeze-out surface switch (branches `*surface_off`, 2026-09-24)**
- MUSIC4GPU `XSCAPE_surface_off` `9bdbf92`: `freeze_out_surface = 0` builds no surface and
  stops on max(e) < e_fo, the same step as Cornelius.
- X-SCAPE `pair_h5_music_surface_off` `ff513d63`: `<Hydro><MUSIC><freeze_out_surface>`.
  The first block sets it globally, an instance's own block overrides it, and
  `MpiMusic::set_freeze_out_surface()` overrides both. `get_music4gpu.sh` pins `9bdbf92`.
- js-contrib `pair_h5_music_surface_off`: the binding, `freeze_out_surface 0` in both
  production XMLs, and `run_prod_jet.py --surface {none,bg,jet,both}`.
- Measured, all bit-identical:
  - hydro-only: 23.3 → 17.1 s/event.
  - pair: 60.1 → 49.1 s/event, or 55.3 s with `--surface jet`.
  - the grid-edge flag fires at the same frame, the CPU path matches, and the Python
    setter works.
- Not pushed yet: MUSIC4GPU `XSCAPE_surface_off` first (the pin), then X-SCAPE and
  js-contrib `*_surface_off`.
