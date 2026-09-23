# Soft particlization for FastHydro (framework iSS + SMASH, both legs)

## Context
FastHydro (`external_packages/js-contrib/contribs/FastHydro`) runs a background leg (`FastHydro_bg`) and a jet leg (`FastHydro_jet`, background plus CausalLiquefier droplets from Matter+LBT). Both are 3+1D Milne FV hydro and stop at parton level. The goal is to see the deposited jet energy in hadrons: take iSS (+SMASH) spectra of the jet leg, subtract the background leg, and check the result against the deposited energy.

What is missing:
- FastHydro builds **no freeze-out surface.** `surfaceCellVector_` is never filled; `hydro.py:209` only clears it.
- `pipeline.py:83-86` refuses to run when the XML has a `<SoftParticlization>` block.
- The framework only supports one hydro feeding one iSS feeding one SMASH:
  - `JetScape::SetPointers` wires iSS to the *first* FluidDynamics, which is the background leg (`src/framework/JetScape.cc:1027-1058`).
  - The Afterburner takes its hadrons from the single SoftParticlization pointer (`src/framework/Afterburner.cc:74-86`).
  - The iSS and SMASH XML keys are singletons.
- The time range and grid are too small. The default grid is 25×25×13 at dx=0.5 (about ±6 fm) with τ ≤ 4.6 fm/c, so the fireball is never below T_sw and the surface cannot close. `zero_after_freezeout` would also create a fake surface.
- The SurfaceFinder flow normalization is wrong, and this plan is what would make it matter (details in the next section).
- π^{μν} and Π are stored as zero (`cells.py:16-19`). The first version therefore uses iSS with δf off.

**Principle.** Once FastHydro has filled `bulk_info`, everything downstream is C++: Cornelius via SurfaceFinder, iSS, SMASH and the writers. The work is closing two small framework gaps in C++:
- iSS never *requests* a surface from a `bulk_info`-only hydro.
- iSS cannot choose which hydro to sample.

Plus FastHydro configuration. No surface code is added in Python.

**Design choice.** Instead of making the framework handle several soft/afterburner instances (a big change to singletons), one run particlizes **one chosen leg**:
- **Jet run:** the two-stage pipeline with iSS+SMASH attached to `FastHydro_jet`.
- **Background run:** the same IC, background-only hydro, iSS+SMASH attached to `FastHydro_bg`.

The background is deterministic for a given IC, so it can be oversampled heavily once per IC and reused for every jet event. Start with a fixed IC replayed through `FastFileInitialState` (`initial_state.py:196`).

## The SurfaceFinder normalization bug, and why this plan exposes it
`PrepareASurfaceCell` (`src/framework/SurfaceFinder.cc:508-515`) computes `u0 = sqrt(1 + vx²+vy²+vz²)` and then `u^i = u0·v^i`, and boosts (u0, u0·vz) to Milne.

**Convention.** The framework stores three-velocities in `vx/vy/vz` (`FluidCellInfo` calls them "flow velocity"). MUSIC, freestream-milne and the HydroFromFile reader all follow this. So does FastHydro: `cells.py:3-5` says "Cartesian lab three-velocities", rebuilt the same way as `HydroinfoMUSIC.cpp:316-329`.
- For three-velocities the formula is wrong. It gives u·u = 1 − v⁴ instead of 1: umu is too short by sqrt(1−v⁴), about 13% at v=0.7 and 41% at v=0.9.
- The error depends only on |v|, so it is present in both the 2+1D (`_3D`) and 3+1D (`_4D`) finders. In 2+1D, η=0 makes the Milne boost trivial, but u0 is still wrong.
- If a hydro stored four-velocity components u^x, u^y, u^z instead, the current formula would happen to be right and the fix would break it. **Before relying on either version, confirm which one each producer stores.** For FastHydro it is three-velocities, so the fix below is correct.

**Why the bug is harmless today.** With the stock wiring, nothing uses SurfaceFinder's umu as a four-velocity:
- iSS reads `getSurfaceCellVector`, which has cells only if the hydro wrapper calls `StoreSurfaceCell` (MUSIC does, from its own finder). A hydro that only fills `bulk_info` gives iSS an empty surface.
- HybridHadronization does call SurfaceFinder, but it only uses `umu[i]/umu[0]`, so the normalization cancels.

**Why this plan exposes it.** Step 2 below routes SurfaceFinder cells straight into iSS, the natural thing to do for a hydro without its own finder. iSS would then get umu that is too short, giving spectra that are too hard, yields that are too high and a biased v₂. **So the fix is a prerequisite, not optional.**

**Fix.** `gamma = 1/sqrt(1 − vx²−vy²−vz²)`, `u^{t,x,y,z} = gamma·(1, vx, vy, vz)`, then the existing boost to Milne. Clamp v² < 1 − ε so an interpolated cell cannot give a NaN.

**Two related issues.** Handle both in the same change:
- **Frame of π.** `PrepareASurfaceCell` boosts umu to Milne but copies `pi[i][j]` unchanged. That is harmless while π=0 (MUSIC's history, and FastHydro now). A hydro that stores non-zero Cartesian π would be inconsistent with the Milne umu at η≠0. For now, document that π must be supplied in Milne components, or boost it with the same Λ(η) as umu. If boosting would change current behaviour, add a warning when π≠0 and η≠0 instead.
- **Which vz.** SurfaceFinder assumes vz is the Cartesian lab dz/dt. A producer that stores the Milne longitudinal velocity τu^η/u^τ, as CLVisc's wrapper appears to do, gets a wrong boost at η≠0 on top of the normalization error. Out of scope here, but note it in the SurfaceFinder header comment and flag the CLVisc wrapper (`src/hydro/CLViscWrapper.*`) for a separate check.

**Diagnostic that settles it for any hydro.** On a few SurfaceFinder cells, compute umu[0]² − umu[1]² − umu[2]² − umu[3]². A result of 1 − v⁴ instead of 1 means the error is live for any direct consumer. Turn this into the unit test in Verification step 1, and also run it once on a MUSIC `bulk_info` (`output_evolution_to_memory=1`, no sources) to confirm MUSIC's three-velocity convention.

## Branches
Before any edits, create a new branch `fasthydro_hadronization` in **both** repos:
- **X-SCAPE** (`/Users/du8478/JetScape/X-SCAPE`), currently on `music4gpu_test`. The core C++ changes go here: SurfaceFinder, SoftParticlization, the signal manager, the iSS wrapper and `JetScape.cc`.
- **js-contrib** (`external_packages/js-contrib`). The FastHydro config, pipeline, drivers and README changes go here.

Branch each one from its current HEAD, after confirming the working tree is clean (`git status`).

Then save this plan as `external_packages/js-contrib/contribs/FastHydro/PLAN_hadronization.md`, committed on the js-contrib `fasthydro_hadronization` branch, so the design stays with the module.

## Changes

### 1. Core: fix and extend SurfaceFinder (`src/framework/SurfaceFinder.{h,cc}`)
- Apply the normalization fix and the π/vz notes from the SurfaceFinder bug section above. This must land before step 2 routes cells into iSS.
- Add optional setters for the Cornelius lattice spacing. It is currently hard-coded to dt=0.1, dx=dy=deta=0.2 (`:180-182, 370-373`). The defaults stay the same, so the lattice can match the hydro grid and runs faster. The spacing is set from optional XML keys `SoftParticlization/surface_{dtau,dx,deta}`, passed through the new signal below.

### 2. Core: generic surface for bulk_info-only hydros (C++)
Surface finding and particlization then run entirely in C++ for any FluidDynamics that fills `bulk_info` (FastHydro, PyFNOHydro, HydroFromFile, …).
- **`src/framework/SoftParticlization.h`**:
  - Add a signal `FindHydroHyperSurface(double T, std::vector<SurfaceCellInfo>&)` with a connected flag, modelled on `Hadronization::GetHydroHyperSurface` (`Hadronization.h:103`).
  - Add an XML key, `SoftParticlization/T_sw` (default 0.15).
- **`src/framework/JetScapeSignalManager.cc`**: connect the new signal to `FluidDynamics::FindAConstantTemperatureSurface`, following `:226-236`.
- **`src/hadronization/iSpectraSamplerWrapper.cc` `getSurfCellVector` (:118-162)**: if the vector from `GetHydroHyperSurface` is empty, emit `FindHydroHyperSurface(T_sw, surfVec)` before falling back to reading `surface.dat`.
  - MUSIC is unaffected because its vector is never empty.
  - `skip_surface` runs still get nothing, because `bulk_info` is empty there.
- **`src/framework/JetScape.cc` `SetPointers` (:1027-1031, 1050-1058)**: an optional XML key `<SoftParticlization><hydro_id>` picks the FluidDynamics task by `GetId()` and wires iSS (and therefore the Afterburner) to it. By default it keeps using the first hydro.
  - FastHydro already sets the Ids `FastHydro_bg` / `FastHydro_jet` (`pipeline.py`).
  - Also check that the pointer set here is the one `GetHydroPointer()` returns when the soft signals are connected. The connect calls run inside the same loop, so the chosen hydro must be set before the iSS task is visited, or the connection must be deferred until after the loop.

### 3. FastHydro (`external_packages/js-contrib/contribs/FastHydro/python/fasthydro/`): configuration only
- **`config.py`**: when the XML has `<SoftParticlization>`, validate that the run can produce a closed surface:
  - `output.freezeout: never` (or a T_fo tail below T_sw), `zero_after_freezeout: false`, `stop_at_freezeout: false`;
  - EOS `hotqcd_smash`, so the fluid matches the SMASH/iSS hadron gas;
  - after the run, max T at the last frame < T_sw and transverse boundary cells always < T_sw. This is a warning printed from `EvolveHydro` using the stored array.
  - The open η edges (±2.4) are expected, so analysis stays at |y| < 1.
- **`pipeline.py`**:
  - Replace the `<SoftParticlization>` refusal (:83-86) with the checks above.
  - Append `create_module("iSS")`, optionally `create_module("SMASH")`, and a final-state hadron writer at the end of `build_two_stage`.
  - Add `build_bg_only` (IC → NullPreDynamics → FastHydro_bg → iSS → SMASH) for the background run, so the background run skips the jets and the second leg.
- `hydro.py` is unchanged: its `bulk_info` store is all the C++ side needs.

### 4. Configs and drivers (`FastHydro/config`, `FastHydro/example`)
- **`config/fasthydro_particlize.yaml`**, based on `fasthydro_wake.yaml` (already `hotqcd_smash`):
  - Grid about ±12.5 fm at dx=0.5 (nx=ny=51), η ±2.4, τ from 0.6 to about 14 fm/c.
  - `freezeout: never`.
- **`config/jetscape_user_fasthydro_particlize.xml`** (with `<SoftParticlization><hydro_id>` set to `FastHydro_jet` or `FastHydro_bg`, one XML per leg): the existing wake XML plus the `<SoftParticlization><iSS>` and `<Afterburner><SMASH>` blocks, copied from `config/jetscape_user_3DGlauber_MUSIC_iSS_SMASH_test.xml`:
  - `hydro_mode` 2, `include_deltaf_shear/bulk` 0, SMASH particle table.
  - `number_of_repeated_sampling` large for the background (for example 1000) and moderate for jet events.
  - Start SMASH with `only_decays=1`, then turn on full rescattering.
  - `include_fragmentation_hadrons=0`, because `GatherAfterburnerHadrons` aborts on oversampling otherwise.
- **`example/run_particlize.py --leg {bg,jet}`**: runs one leg for N events on a fixed or replayed IC and writes hadrons plus provenance (IC sha, droplet energy from `hydro.diag`/source accounting, jet axis).
- **`example/delta_spectra.py`**: computes the per-oversample-averaged dN/dy dpT dφ for π/K/p, and ΔN(Δη, Δφ, pT) = ⟨jet⟩ − ⟨bg⟩ around the jet axis. It also computes the ΔE_T balance at mid-rapidity.
- Update the Limitations section of FastHydro `README.md` (:8-11, 490-495).

## Reuse
- `PyFNOHydro` already follows the "store the evolution, then find the surface" pattern (`PyJetscape/python/jetscape/fno_hydro.py:362-363`), as do `FnoHydro.cc:340` and `FnoRooIn.cc:311`.
- `iSpectraSamplerWrapper` already takes the in-memory surface (`iSpectraSamplerWrapper.cc:118-162`) and forces `hydro_mode=2` for non-boost-invariant runs (:67-69). No iSS changes are needed.
- The FVvsMUSIC setup (`config/FVvsMUSIC/`, `run_music_leg.py`) runs MUSIC on the same fast_data IC and serves as the hadron-level reference.

## Verification
1. **SurfaceFinder unit check:**
   - Build a synthetic `EvolutionHistory` with a static fluid and with uniform v = 0.3, 0.7, 0.9 (transverse, longitudinal, and mixed, at η ≠ 0).
   - Assert |u·u − 1| < 1e-6. Before the fix this gives 1 − v⁴.
   - Assert Milne u^τ = γ(cosh η − vz sinh η), and the matching u^η expression.
   - Run the same u·u check on the cells of a real FastHydro background surface and of a MUSIC `bulk_info` surface.
   - Rerun an existing Hybrid hadronization config and confirm its output is unchanged. It only uses umu[i]/umu[0], so it should be.
2. **Background surface:**
   - It closes, per the checks above.
   - The Cooper–Frye energy flux ∫T^{τν}dσ_ν over |η|<1 matches the hydro energy in that slab to within a few %.
3. **Against MUSIC:** on the same IC, compare FastHydro_bg → iSS π/K/p dN/dy, ⟨pT⟩ and v2 with MUSIC + iSS (turn on the surface in the FVvsMUSIC XML: `skip_surface 0`). Expect agreement at the percent level, consistent with the ~0.3% hydro agreement already measured.
4. **Wake:**
   - PGun jets in the jet run give ΔN > 0 along the jet direction and a depletion on the away side / diffusion wake.
   - The mid-rapidity ΔE_T agrees with the deposited droplet energy inside the acceptance.
   - Report the statistical error against the number of oversamples to show when the signal is measurable.
5. Run `pytest` in `FastHydro/tests`, including `test_vendor_intact.py`, since `fast_data` is left untouched.
