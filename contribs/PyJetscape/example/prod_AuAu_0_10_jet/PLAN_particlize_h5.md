<!-- Plan, written 2026-09-26. Status: implemented and validated 2026-09-26 on branches
     surface_to_hadrons (X-SCAPE, js-contrib); see "Status" at the end. -->

# Plan: hadron-level jet pairs from prod_AuAu_0_10_jet (stored surfaces + final partons)

## Context

`prod_AuAu_0_10_jet` writes background/jet hydro pairs (MUSIC_1 → Matter+LBT+CausalLiquefier →
MUSIC_2) to FNO4d HDF5 (`jetscape.pair_h5.PairH5Writer`). FastHydro can already compare the wake
at hadron level: it particlizes each leg with iSS (`FastHydro/PLAN_hadronization.md`,
`fasthydro/particlization.py`, `example/make_hadron_wake_data.py`). We want the same for the real
MUSIC runs, **plus the jet's own fragmentation hadrons**, without tying the hadron workflow to the
FNO files.

**The idea.** The production job stores the *input* to hadronization, not the hadrons:
- each leg's MUSIC freeze-out surface with every field iSS reads;
- the final partons with the fields X-SCAPE's jet hadronization reads.

Hadronization then runs offline, as often as wanted: iSS on each surface, Colorless on the partons.

### Why not the alternatives

| option | problem |
|---|---|
| iSS inside the job | One job has one iSS, wired to one hydro (`<SoftParticlization><hydro_id>`, `JetScape.cc:1020`). The background leg needs a second job, run twice over (~86 s + 2×iSS per event, against 49 s for the pair alone). iSS's hadron list (maybe millions of `shared_ptr<Hadron>`) lives in GPU jobs that already peak at 16 GB. |
| particlize `arr`/`arr_bg` offline | No π^{μν}/Π stored → ideal Cooper–Frye on a viscous evolution. Also: output grid dx 0.3125 vs MUSIC 0.2, \|η\| < 5 vs ±11, a different surface finder, and a possible fake surface where the zero tail starts. |
| second iSS instance in the framework | The singleton refactor `PLAN_hadronization.md` already declined. |

### Why stored surfaces are exact

- The framework's `SurfaceCellInfo` is `Jetscape::real` = **float** (`RealType.h:26`). iSS gets
  exactly these floats (`iSpectraSamplerWrapper.cc:131-165`, 32 values per cell). So a float32
  copy is bit-identical to what iSS would receive inside the job.
- Every MUSIC instance fills its own `surfaceCellVector_` (`surface_in_memory 1`, already set).
  No iSS runs in the job, so both legs' surfaces can be taken from the same event: same IC, no
  twin jobs, no seed matching.

### RNG note (corrects `README.md`, "Seeds")

With `seed ≠ 0` every module gets its own mt19937, seeded from (seed, task number)
(`JetScapeTaskSupport.cc:104-118`) and cached per module (`JetScapeModuleBase.cc:67`).
- 3dMCGlauber is seeded once in `MCGlauberWrapper::InitTask` (`:38`) and after that runs on its
  own generator.
- The README's explanation ("extra modules draw from the framework's random stream first") does
  not match this code. Its observation (seed 1 differs from `prod_AuAu_0_10` at frame 0) is still
  unexplained.
- This plan doesn't depend on it, since both legs come from one job. Resolve it separately.

## Decisions

- **One separate file per job**, `AuAu_0_10_jet_seedNNNN_particlize.h5`, next to the pair file.
  The pair file is unchanged, so the FNO workflow never sees hadron data.
- **Surfaces in float32** (the framework's own precision; exact, see above).
- **Background surface once per `bg_id`** under `--reuse N`. MUSIC_1 isn't rerun, and without
  iSS in the job nothing clears its surface.
- **Colorless is the jet hadronization.** It's the standard JETSCAPE choice for AA with
  Matter+LBT, and it filters by pstat, so absorbed partons aren't hadronized twice. Colored is not
  supported for this setup (see its own section below); col/acol are stored anyway.
- **Offline hadrons are tagged by origin**: `bulk_jet`, `bulk_bg`, `jet_frag`. Jet event =
  `bulk_jet` + `jet_frag`; background = `bulk_bg`.
- **Writer code reused**: `RaggedGroup` from `jetscape/fno_h5_writer.py` (incremental writes,
  crash-safe, resume), as planned in Phase 5 of `FastHydro/PLAN_consolidate_h5_writer.md`.

## Phase 0: measure (1 event, `build_gpu`, seed 1)

1. Run with `--surface both` and `skip_surface 0` (details in Phase 3). Record:
   - cells per leg;
   - MB per leg, raw and gzip/zstd-compressed;
   - wall time against 49.1 s (`--surface none`). The README has 60 s with both surfaces on.
2. Check that MUSIC_2 hands over its surface with `dump_hydro_only` on (set per instance in Python)
   on the GPU build: `getSurfaceCellVector()` must not be empty for either leg.
3. Check every cell: u·u = 1 (MUSIC's own finder; expected fine), and that the charge densities
   and μ columns are all zero (`Include_Rhob 0`).

**Gate:** both legs' surfaces present; the size per event is acceptable next to the pair file.

## Phase 1: bindings (PyJetscape `src/`)

- **Surface as numpy.** `getSurfaceCellVector()` exists (`bind_fluid_dynamics.cc:577`) but
  returns one Python object per cell, too slow for ~10⁵–10⁶ cells. Add
  `FluidDynamics.surface_to_numpy()` → `(N, 32)` float32, with the column order below. Works on
  any `FluidDynamics`, including `MpiMusic`.
- **Surface from numpy** (for the offline side): `PyFluidDynamics.store_surface_from_numpy(arr)`,
  which calls `StoreSurfaceCell` per row.
- **Partons.** Add `color()`, `anti_color()` and `restmass()` to the `Parton` binding
  (`bind_jet.cc:45-54`; accessors in `JetScapeParticles.h:273,356-357`). Add a per-shower
  final-parton accessor on `JetEnergyLossManager`: the exact list `HadronizationManager` receives
  (`JetEnergyLossManager.cc:192`, `GetFinalPartons`). Don't derive it from the shower graph.
- **Hadrons.** `SoftParticlization.get_hadrons_numpy()` (Phase 5 of the consolidation plan), and
  the same for a `Hadronization` module's output. Only the in-job validation runs need these.
- **Colorless offline.** A helper that builds `vector<vector<shared_ptr<Parton>>>` from numpy
  (shower index, pid, pstat, p, x, mass, col, acol) and calls
  `ColorlessHadronization::DoHadronization` directly, after a normal XML `Init` so Pythia gets its
  settings. This skips the framework event loop, where the parton signal is tied to
  `JetEnergyLossManager`.

## Phase 2: writer — `python/jetscape/particlize_h5.py`

`ParticlizeH5Writer`: stdlib + numpy + h5py, testable with stubs like `PairH5Writer`. Called
between `ExecPerEvent()` and `ClearPerEvent()`.

```
<stem>_particlize.h5
  attrs: format="xscape/particlize_input", format_version=1, producer, provenance (as the pair
         file: seed, XML, hard process, reuse), T_fo, EOS id (9), music_input flags iSS needs
         (Include_Bulk_Visc, Include_Rhob, turn_on_baryon_diffusion), pair_file (name + uuid)
  surface/jet/cells      (N, 32) float32   RaggedGroup, one unit per event
  surface/jet/offsets
  surface/bg/cells       (M, 32) float32   one unit per NEW bg_id
  surface/bg/offsets
  surface/bg/bg_id       (n_bg,) int64     first event that used this background
  partons/data           (K, 14) float64   shower, pid, pstat, E, px, py, pz, t, x, y, z,
                                           mass, col, acol
  partons/offsets
  events/  per event:    bg_id, bg_key (from PairH5Writer), ic_hash, n_droplets, E_droplets,
                         E_droplets_late, pythia_seed, iss_seed (optional)
```

- **Surface columns**, iSS's `FO_surf` order: `tau x y eta | ds0 ds1 ds2 ds3 | u0 u1 u2 u3 |
  e T P | nB nQ nS | muB muQ muS | pi00 pi01 pi02 pi03 pi11 pi12 pi13 pi22 pi23 pi33 | Pi`.
  Stored as `cell_columns`, with units and frame (Milne, as MUSIC hands them over) as attributes.
- `nevents_written` + `complete`, and resume that truncates every ragged group to
  `nevents_written`, as `FnoH5Writer`.
- An event whose surface is empty (e.g. MUSIC stopped at the grid boundary) is stored as an empty
  unit and flagged in `events/`. It is never skipped.

## Phase 3: production — `run_prod_jet.py`

- New option `--write-particlize {none,jet,both}`, default `none`.
  - `jet` stores only the jet surface and partons; use it when the background surface is already
    stored for that seed.
  - It implies `--surface` for the same legs, which sets `<freeze_out_surface>` per instance.
  - It sets `<skip_surface>0`. This tag is global; at 1 the hand-off is skipped
    (`MusicWrapper.cc:602`).
- `job_xml()`: keep rejecting `<SoftParticlization>`, `<Afterburner>` and `<JetHadronization>` in
  production. With `--validate-inline` (Phase 5), allow them.
- Create `ParticlizeH5Writer` next to `PairH5Writer`. In the loop:
  `idx = writer.Exec(); pwriter.Exec(idx, bg_key=..., jetscape=...)`.
- Add the particlize file and its event count to the summary `.json`.
- Teach `run_jobs.sh` (via `../prod_AuAu_0_10/run_jobs.sh`) to treat a seed as complete only
  when the particlize file is also `complete`, whenever the option is set.
- Optional, later: make `skip_surface` settable per instance with the `ReadOwnMusicBlockInt`
  pattern, so the leg without a surface stops warning every event.

## Phase 4: offline hadronizer — `example/prod_AuAu_0_10_jet/hadronize.py`

Runs anywhere with a CPU build; no GPU, no MUSIC.

- **Bulk.**
  - A small `SurfaceReplay(PyFluidDynamics)` loads one event's cells with
    `store_surface_from_numpy`, marks itself finished, and fills no `bulk_info`. iSS then takes
    the normal `GetHydroHyperSurface` path, with `<SoftParticlization><iSS>` from a small offline
    XML.
  - One process per leg (`--leg jet|bg`). Events line up by index, and `bg_id` maps each
    background to its events.
  - `music_input` for iSS is written from the file's attributes, as
    `fasthydro.particlization.write_iss_music_input` does (EOS 9: UrQMD list, iSS decays).
- **Jet fragments.**
  - Colorless via the Phase 1 helper on `partons/`, with `<JetHadronization>` settings from the
    offline XML.
  - `eCMforHadronization` = 200, not the main-XML default of 5020.
  - `take_recoil` = 1: recoils (pstat 1) are hadronized; absorbed partons (-11) and holes (-17)
    aren't, because their energy went into MUSIC_2.
  - Beam-remnant hadrons are tagged so analyses can drop them.
- **Output**: `hadrons_<tag>.h5` in the Phase 5 hadron layout:
  - `RaggedGroup("hadrons", "sample_offsets", {pid int32, p float32 (N,4), x float32 (N,4)},
    unit="free")` plus `event_offsets`;
  - attributes `tag`, `n_oversample`, seeds, and the input file + event range.
- **Sample counts are set separately:** `--oversample` for iSS; `--n-frag` for fragmentation
  (Pythia is cheap). For jet finding, combine one fragmentation sample with one iSS oversample.
- **Reader**: `Hadrons.from_h5(path, event=None)` next to `fasthydro/hadrons.py`, with the same
  compound-Poisson errors.
- **Wake notebook**: a hadron section in `jet_wake.ipynb`: ΔN(Δη, Δφ, pT) = ⟨bulk_jet⟩ − ⟨bulk_bg⟩,
  with and without `jet_frag`, and the ΔE_T balance.

## Colored hadronization: not supported for this setup

Found by reading the code; the diagnostic below measures it.
1. **LBT assigns no color tags.** `src/jet/LBT.cc` has no `set_color`/`set_anti_color`;
   `Matter.cc` has 16. Partons produced or re-emitted below Q0 = 2 GeV, and LBT recoils, carry
   col = acol = 0.
2. **The liquefier breaks color lines.** It takes absorbed partons (-11) and holes (-17) out, and
   they carried color tags.
3. **`ColoredHadronization.cc` has no pstat filter** (`:166-189`). It would hadronize absorbed
   partons (counted twice, with MUSIC_2) and holes (as positive energy).
4. **It repairs only one break per shower.** It closes only the first unpaired color and
   anticolor (`:191-245`).
5. **`pythia.next()` isn't checked** (`:311`). A failure is silent. Suspected, not verified: the
   input partons (status 23, final) are then copied out as hadrons.

It would work for **Matter-only** showers (vacuum reference). With LBT it needs physics work
(color assignment in LBT), not just a fix.

**Diagnostic** (offline, once `partons/` holds col/acol), per event:
- the fraction of final partons with col = acol = 0;
- unpaired tags per shower, before and after taking out pstat -11/-17;
- whether `pythia.next()` succeeds when the partons are fed to Colored.

Record it in `hadronize.py --diagnose-colored`. Revisit if a Matter-only campaign is wanted.

## Phase 5: validation

1. **Surface and iSS are exact.** One event with `--validate-inline`: iSS inside the job on
   MUSIC_2 (`hydro_id=MUSIC_2`), its hadrons stored with `get_hadrons_numpy`, and the iSS seed
   recorded. Offline iSS on the stored surface with the same seed must give identical hadrons.
2. **Colorless is exact.** Same event, `<JetHadronization>` Colorless inside the job, Pythia seed
   recorded. Offline fragmentation on `partons/` must be identical. This also shows that the
   stored list equals `GetFinalPartonList`.
3. **Null test.** `--no-deposit`: `surface/jet` = `surface/bg` cell for cell, and
   ⟨bulk_jet⟩ − ⟨bulk_bg⟩ consistent with 0 within the compound-Poisson errors.
4. **Energy balance.** With a jet, ΔE_T(bulk) + E_T(jet_frag) against the jet's initial E_T,
   inside the acceptance. The expected deficit is `E_droplets_late` (energy that reaches neither
   MUSIC_2 nor the fragmentation).
5. **Reuse.** `--reuse 3`: one `surface/bg` unit per `bg_id`, and the right event → bg mapping.
6. `pytest` for `PyJetscape/tests`, with a new `test_particlize_h5.py` (stubs; resume, empty
   surfaces, reuse).

## Out of scope

- **Hybrid hadronization.** Its thermal partons come from the surface (stored), but it also
  queries the live medium (`GetHydroCellSignal`), and it raises the thermal-parton/iSS
  double-counting question.
- **SMASH afterburner.** It needs the SMASH hadron list, i.e. an EOS 91 hydro campaign; the
  production runs EOS 9.
- **Other T_sw.** The surface is at T_fo = 0.15. Other switching temperatures need a hydro rerun,
  or particlization from `arr` (ideal Cooper–Frye).
- **FastHydro.** Porting FastHydro to the same particlize/hadron layout is the natural follow-up;
  its ASCII → `.npz` path would go away.

## Risks

- **Surface size.** Unknown until Phase 0. If it's too large: drop the zero columns (charges, μ;
  recorded as attributes) and compress with Blosc-zstd.
- **Time.** About +11 s per event with both surfaces (README numbers), roughly +22%. With
  `--reuse N`, the background surface costs about 1/N of that.
- **MUSIC_2 surface with `dump_hydro_only`** on the GPU path hasn't been tried (Phase 0.2).
- **Memory.** The in-job surface hand-off copies cells into the framework vector. Small next to
  the 16 GB peak, but check it in Phase 0.

## Status (2026-09-26): implemented and validated

Branches `surface_to_hadrons` in X-SCAPE (from `contrib`) and js-contrib (from `main`).

### What was built

- **X-SCAPE.**
  - `SoftParticlization`: `SetNextRandomSeed` (one-shot) and `GetLastRandomSeed`; iSS takes
    its per-event seed through them (the default draw is unchanged).
  - `ColorlessHadronization`: the same pair, plus `<JetHadronization><reseed_per_event>`
    (default 0 in `config/jetscape_main.xml`, which X-SCAPE requires for every user tag):
    Pythia and the module's own generator are reseeded per event.
- **PyJetscape bindings.**
  - `FluidDynamics.surface_to_numpy()` / `store_surface_from_numpy()` and
    `SURFACE_CELL_COLUMNS`.
  - `JetEnergyLossManager.final_partons_numpy()` and `FINAL_PARTON_COLUMNS`.
  - `Parton.color()` / `anti_color()` / `restmass()`.
  - `bind_hadronization.cc`: `soft_hadrons_numpy`, the iSS seed hooks,
    `hadronization_hadrons_numpy`, `hadronize_partons` (any jet hadronization module on
    stored partons) and `jet_hadronization_last_random_seed`. These are free functions on a
    task, because `create_module()` hands out types pybind11 does not know.
- **Python.**
  - `jetscape/particlize_h5.py` (`ParticlizeH5Writer`, `ParticlizeFile`).
  - `jetscape/hadrons_h5.py` (`HadronH5Writer`, `Hadrons`).
  - `jetscape/surface_replay.py` (`SurfaceReplay`).
  - `PairH5Writer(keep_surface=...)` and its `last_bg_key`.
- **Production.**
  - `run_prod_jet.py --write-particlize {none,jet,both}`, `--validate-inline`,
    `--hadronize-xml`.
  - `run_jobs.sh` counts a seed as complete only when its particlize file is too.
- **Offline.** `hadronize.py` (tags `bulk_jet`, `bulk_bg`, `jet_frag`; `--use-stored-seeds`;
  `--diagnose-colored`) and `hadronize.xml`.
- **Single events.** `Hadrons.sample_event` / `HadronFile.sample_event` (any oversample or
  fragmentation as one event, lazily from disk), and `JetEvents`: bulk + fragments per event
  (`jet_event(event, k)`, origin-tagged) and the event's background (`background_event`,
  reuse-aware).
- **Notebook.** `jet_wake.ipynb` §10.
- **Campaigns.** `HadronFileReader`: all production files of a campaign as one data set
  (global event numbering, per-event accumulation with equal event weights, per-event
  backgrounds under reuse with correlated errors, uuid check). A merge tool to single files
  per tag is noted as an option, not written.
- **Tests.** `tests/test_particlize_h5.py` (15 tests).

### Measured (GB10, seed 1, one 0–10% event)

| | |
|---|---|
| surface cells | 1,001,592 (jet), 989,732 (background); u·u = 1 to 1.4e-6; only the charge and μ columns are zero |
| size | 128 MB per leg raw, 77.6 MB with Blosc-zstd; particlize file 154.5 MB (pair file 285 MB) |
| time | 29.5 s → 43.0 s per event with `both` (+13.5 s): ~12 s for MUSIC's two surfaces and their hand-off, 1.35 s for the write (lz4 0.42 s / 180 MB, gzip 3.8 s / 159 MB). With the parallel surface finder (MUSIC4GPU `5058545`, below): 34.6–35.3 s (+5.4 s) |
| memory | +0.5 GB at the hand-off |
| pair file | byte-identical with and without `--write-particlize` |
| `hadronize.py` | ~20 s per surface at 100 oversamples, ~30 s at 500 |

### Validation (Phase 5)

1. **Surface + iSS are exact.** Two events with `--validate-inline` (100 oversamples) against
   `hadronize.py --use-stored-seeds`: bit-identical, 1,710,050 hadrons.
2. **Colorless is exact.** Same job: bit-identical fragments.
3. **Null test** (`--no-deposit`, seed 2).
   - `surface/jet` = `surface/bg` cell for cell (1,195,143 cells).
   - With 200 oversamples each, jet − bg at |η| < 1 is +0.85σ in charged N and +1.4σ in E_T;
     the Δφ pulls have χ²/ndf = 0.87.
4. **Energy balance** (seed 1, 500 oversamples, 50 fragmentations).
   - The bulk gains 18.1 ± 2.7 GeV at |η| < 1 and 24.6 ± 5.6 GeV at |η| < 2, for 25.0 GeV
     deposited.
   - The fragments carry 106.3 ± 5.5 GeV, exactly the 5 surviving partons' energy.
   - The initial partons had 127.5 GeV (the extra ~4 GeV is what LBT recoils take from the
     medium).
5. **Reuse** (`--reuse 3`, seed 3): one `surface/bg` unit, `bg_unit` = [0, 0, 0], a new jet
   surface every event.
6. **pytest:** 73 passed, 1 skipped.
   - The one failure is `test_surface_finder.py::test_lattice_spacing_is_honoured` (area 41.7
     instead of 32). It needs a SurfaceFinder lattice fix that is not on `contrib`, and this
     branch does not touch SurfaceFinder.

### Colored diagnostic (`hadronize.py --diagnose-colored`, seed 3, 3 events)

75–91% of the final quarks and gluons carry no colour tag (mean 84%), with 2–5 unpaired tags per
event. ColoredHadronization failed in all three events:
- one **aborts the process**: a `JetScapeParticleBase` assertion on a Pythia output id, so
  each event now runs in its own child process;
- in the other two, the input partons come out as "hadrons" (event 1: 86 of 86). That is the
  unchecked `pythia.next()` suspected above.

### Found on the way

- **`--validate-inline` changes the jet sample, not the background.** With seed ≠ 0 every
  module's generator is seeded from its task number (`JetScapeTaskSupport.cc:104-118`). The
  energy-loss clones made at the first event get new task numbers: `JetEnergyLoss`'s copy
  constructor default-constructs its base, which registers a task. So the four modules the
  inline validation adds at Init shift every clone's seed (seed 1: 26 droplets / 33.1 GeV
  instead of 25 / 25.0 GeV), while MCGlauber and MUSIC, created earlier, are unchanged. The
  exactness test compares in-job with offline within the same job, so it is unaffected.
- **The main XML reuses the hydro by default** (`setReuseHydro` true, 10). In an offline
  replay that switches the replay and iSS off for 9 of every 10 framework events, and they
  silently return the previous event's hadrons. `hadronize.xml` turns reuse off, and
  `hadronize.py` refuses a result whose seed iSS did not consume.
- **iSS reads MUSIC's flags from `music_input`.** MusicWrapper rewrites only `EOS_to_use` and
  `Include_Bulk_Visc_Yes_1_No_0` in it, so `Include_Rhob_Yes_1_No_0` keeps the template's 1
  (harmless here: the μ's are zero). The particlize file stores the job's `music_input`
  verbatim, so offline iSS reads exactly what the in-job iSS would.
- **pstat −13** is the four-momentum a vertex failed to conserve
  (`LiquefierBase::check_energy_momentum_conservation`). `add_hydro_sources` leaves it out of
  `p_final`, so it is part of the droplet, and Colorless rightly skips it.

### Deviations from the plan above

- **No resume.** A production job restarts from its first event anyway (`run_jobs.sh` reruns
  incomplete seeds), so the particlize file is rewritten, as the pair file is.
- **Remnant hadrons are not tagged.** Colorless adds a remnant only to close a string (odd
  number of quarks: one; none: two), and Pythia's output does not say which hadrons came from
  its string. It is documented instead; seed 1 needs no remnant.
- **Hadron files also store `pstat`,** so Colorless's negative-parton hadrons (pstat −1) could
  be subtracted if `take_recoil` ever meets unabsorbed holes.
- **`events/` has no separate `ic_hash`:** `bg_key` (the pair writer's hash of the whole
  background leg) identifies the initial condition.

### Surface finder (2026-09-26, MUSIC4GPU `5058545`, merged into `XSCAPE` as PR #12 `6b238c4`, pinned by X-SCAPE `0126403e`)

- **Why it was slow.** MUSIC's 3+1D surface search ran serially: upstream disabled the
  OpenMP loop over η slices (`5025f77`), presumably because with `surface_in_memory` every
  thread pushed into one vector. It is CPU code, run every 5th step on a full host copy of
  the grid; ~57 ms per pass, ~9.6 s per event for both legs (`MUSIC_PROFILE=1`).
- **Change.** The η-slice loop runs in parallel again, each slice into its own vector, the
  slices appended in η order (memory mode only; the file mode stays serial). Search: 1.2 s
  per event (≈8×). A first version reserved the exact size before each append, which made
  every pass re-copy the whole accumulated surface (quadratic, 70% of the time); removed.
  Parallelizing the previous-step copies was slower (bandwidth-bound, thread wake-ups) and
  was reverted.
- **Pressure bug found on the way.** In-memory surface cells never got their pressure, in
  CPU MUSIC and MUSIC4GPU alike (`SurfaceCell::pressure` uninitialised in all three
  in-memory fills), so the stored `P` column was garbage (e.g. −1.3e26) and differed run to
  run. Now set (P/e ≈ 0.17 on the isotherm). iSS overwrites P from its own HRG EoS
  (`regulateEOS = 1`, `MC_sampling` 4), so hadrons were never affected.
- **Checks.** Surfaces bit-identical across runs and thread counts (20, 5), and to the
  serial build in every column but `P`. iSS hadrons from the new surfaces, same seeds:
  bit-identical to those from the serial build (7.08 M hadrons). No-surface runs unchanged
  (29.6 s).
- **Timing** (GB10, one job): both surfaces 43.0 s → 34.6–35.3 s at the default 20 threads.
  With `OMP_NUM_THREADS=5` a single job takes 42.5–43.1 s, as every CPU part runs on 5
  threads; that setting is for `-j 4` campaigns.
- **Not changed:** CPU MUSIC (`external_packages/music`) has the same serial loop and the
  same pressure omission.

