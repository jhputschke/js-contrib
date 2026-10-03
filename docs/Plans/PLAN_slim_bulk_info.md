<!-- Plan, written 2026-09-30. Status: option A done as an option (2026-10-02,
     X-SCAPE and js-contrib branch slim_bulk_info), after three smaller fixes (F1-F3). -->

# Plan: a slim background copy (`bulk_info`) for the jet production

## Status 2026-10-02: where the 20 GB came from, and F1-F3

An RSS trace (0.1 s, tagged with the job's log line; seed 1, 3 events, `--reuse 1`) showed
that ~10 GiB of the peak was memory nothing used any more, not the copy itself:

- `EvolutionHistory::clear_up_evolution_data()` only calls `data.clear()`, which keeps the
  capacity. An event larger than every earlier one (event 2 here: 67.2 M cells after 57.6 M)
  reallocates in `PassHydroEvolutionHistoryToFramework`'s `resize` with the old buffer
  still resident: 6.0 + 7.0 GiB at once, the 20 GiB spike. Smaller events reuse it.
- `HydroinfoMUSIC::clean_hydro_event()` (MUSIC4GPU) also only calls `clear()`: each MUSIC
  instance kept its largest store (~2 GiB) for the rest of the job, MUSIC_2's included.
- The pair writer read the background as `to_numpy_full(5)` plus a 4-field copy: +2.7 GiB.

Fixed, output byte-identical (pair and particlize files, every dataset; `source/flux`
varies in the last bit between runs of the same build anyway):

| | change | where |
|---|---|---|
| F1 | free the old copy before a larger one is allocated | X-SCAPE `MusicWrapper.cc` (branch `free_evolution_memory`) |
| F2 | `clean_hydro_event()` releases the store (swap with an empty vector) | MUSIC4GPU `HydroinfoMUSIC.cpp` (branch `free_evolution_memory`) |
| F3 | the writer reads the background frame by frame (`EvolutionHistory.frame_numpy`, `bulk_sources.FrameworkFrames`) | js-contrib PyJetscape (branch `free_evolution_memory`) |

Measured (GB10, one job alone, peak RSS from `/proc/<pid>/status`, before → after):

| job | peak RSS | wall time |
|---|---|---|
| 3 events, `--write-particlize both` | 19.99 → **14.24 GiB** | 109.2 → 107.6 s |
| 2 events, `--reuse 2`, `--write-particlize both` | 15.65 → **12.24 GiB** | 52.3 → 49.0 s |
| 3 events, `--particlize-only` | 19.94 → **13.02 GiB** | 100.6 → 100.5 s |

The hand-off spike is gone (12.3 GiB there now); the peak is now at the end of the largest
event, MUSIC_2's surface plus the writer, with the framework copy (7 GiB) still held. That
copy is what option A below shrinks, so the numbers in "What is in memory now" are the
pre-fix ones.

## Status 2026-10-02: option A, as a switch

`<Hydro><MUSIC><slim_bulk_info>` (default 0 in `jetscape_main.xml`; MpiMusic
`get/set_slim_bulk_info`) and `run_prod_jet.py --bulk-info {slim,full}` (default `slim`).
What was built, against the phases below:
- `EvolutionHistory`: the slim copy is the existing `data_vector`/`data_info` layout with
  `SlimDataInfo()` = e, s, T, vx, vy, vz. `SetDataInfo` resolves the names once
  (`data_ids`), so `GetFluidCell` neither resolves strings nor allocates per lookup
  (`FromVector` and PyJetscape's `data_info` setter go through it). `get_data_size()`
  counts cells in either layout; `clear_up_evolution_data()` clears both.
- `MpiMusic::PassSlimEvolutionHistoryToFramework`: the same floats the full copy gets, 6
  per cell; frees a smaller buffer before allocating a larger one (as F1). Not with a
  pre-equilibrium evolution in memory (falls back to full, with a warning).
- Guards: `FluidDynamics::GetHydroInfo` accepts either layout;
  `FindSurfaceFromEvolution` refuses the slim copy (`IsSlimCopy()`) with a message, other
  data_vector histories (CLVisc, tests) as before.
- PyJetscape: `to_numpy`, `to_numpy_full`, `frame_numpy` read either layout; a field the
  slim copy lacks raises instead of returning zeros. `tests/test_bulk_info_layouts.py`.

Measured (GB10, one job alone, seed 1), F1-F3 build → `--bulk-info slim`; every dataset of
the pair and particlize files byte-identical (showers, droplets and the jet leg included,
so Matter/LBT saw the same medium; `source/flux` varies in the last bit run to run):

| job | peak RSS | wall time |
|---|---|---|
| 3 events, `--write-particlize both` | 14.24 → **8.71 GiB** | 107.6 → 103.2 s |
| 2 events, `--reuse 2` | 12.24 → **7.54 GiB** | 49.0 → 47.9 s |
| 3 events, `--particlize-only` | 13.02 → **7.51 GiB** | 100.5 → 96.0 s |
| 3 events, `--bulk-info full` | 14.24 → 14.27 GiB | 107.6 → 108.4 s |

From ~20 GiB at the start: 8.7 GiB with the pair file, 7.5 GiB particlize-only. The peak is
now at the end of the largest event (MUSIC_2's surface and the writer); with
`--particlize-only` MUSIC_2's run (its own store, ~2.5 GiB) comes close, which not storing
MUSIC_2's evolution would remove.

`-j` sweep with the merged code (GB10, `--mps`, `OMP_NUM_THREADS ≈ 20/P`, 3 events per job,
no `--write-particlize`; docs/BENCHMARK_GB10.md): `-j 4` 197 events/h in **28.8 GB** for the
whole machine (66–69 GB before), `-j 5` 201 / 33.5 GB, `-j 6` 205 / 36.9 GB, `-j 8` 202 /
52.8 GB, 7.2–8.0 GiB per job. The sizing is now ~12 GB per job (`--mem=48G` for `P=4`).
Not yet run: a long campaign with `--write-particlize both` (one such job alone peaks at
8.7 GiB).

## Context

A `prod_AuAu_0_10_jet` job peaks at **~20 GB of host memory** (17–22 GB measured on the
GB10); the hydro-only `prod_AuAu_0_10` job needs ~4.4 GB. Host RAM per GPU now limits how
many jobs share a GPU (`run_jobs.sh -j P`, `utils/slurm_prod_array.sh`): on cluster nodes
with 64–128 GB per GPU that is 3–5 jobs, while the GPU could take more.

Most of the difference is the background leg (MUSIC_1) kept as X-SCAPE's framework copy,
`bulk_info`, for the whole event, so Matter, LBT and the liquefier can look up the medium.
This plan shrinks that copy to the fields that are read. The goal is **bit-identical
output** with roughly a third less peak memory per job.

## What is in memory now

| | value | source |
|---|---|---|
| peak RSS, one job | 20.1–20.5 GB | `prod_AuAu_0_10_jet/README.md` (seed 1, with and without `--reuse`) |
| between MUSIC runs | ~12.3–12.7 GB | same |
| the peak | a ~1 s spike while the background leg is converted into `bulk_info` | same |
| hydro-only job (no copy) | ~4.4 GB | `prod_AuAu_0_10/README.md` |
| `FluidCellInfo` in `bulk_info.data` | 28 floats = 112 B per cell | `FluidCellInfo.h:85–100`; `bulk_sources._FLUID_CELL_BYTES` |
| MUSIC's own store (`HydroinfoMUSIC::lattice_ideal`) | 8 floats = 32 B per cell: η, s, e, P, T, uˣ, uʸ, u^η | `music4gpu/src/data_struct.h`, `fluidCell_ideal` |
| cells per background | 100 × 100 × 60 grid × ~110 stored frames ≈ 7×10⁷ (to be measured, Phase 0) | MUSIC grid, `output_evolution_every_N_timesteps 5` |

So the copy is ~7.5 GB (more for longer events), MUSIC's store ~2 GB. The spike is both at
once: `MpiMusic::PassHydroEvolutionHistoryToFramework` (`MusicWrapper.cc:808`) fills the
copy from the store, then calls `clear_hydro_info_from_memory()`.

## Who reads the background copy

Checked in the X-SCAPE sources (2026-09-30) and PyJetscape:

| reader | fields | where |
|---|---|---|
| Matter | `temperature`, `entropy_density`, `vx`, `vy`, `vz` | `Matter.cc:496, 778–782, 3904–3908` |
| LBT | `temperature`, `entropy_density`, `vx`, `vy`, `vz` | `LBT.cc:696–700, 800–804, 1062–1065` |
| Liquefier, `filter_partons` | `vx`, `vy`, `vz` (boost to the fluid frame); `temperature` (E < k·T threshold) | `LiquefierBase.cc:179–198` |
| CausalLiquefier | none (droplets from the partons) | `CausalLiquefier.cc` |
| Source term into MUSIC_2 | none (`HydroSourceJETSCAPE` reads the droplets) | `MusicWrapper.h` |
| Pair-file writer, background leg | `energy_density`, `vx`, `vy`, `vz` | `bulk_sources.event_array`, `grid_mode="framework"` → `to_numpy_full` (`bind_evolution.cc:281–305`) |
| `BulkDynamicsManager` (only if in the XML; not in this production) | `energy_density` or `temperature` | `BulkDynamicsManager.cc:672–676` |
| Freeze-out surfaces | none: MUSIC builds them from its own state | `--write-particlize`, `surface/*/cells` |

**How iSS gets a surface** (`iSpectraSamplerWrapper::getSurfCellVector` and the fallback after
it), in this order:
1. MUSIC's own surface, in memory: MUSIC finds it during the evolution from its full
   viscous state, and `MpiMusic::PassHydroSurfaceToFramework` (`MusicWrapper.cc:758`) hands
   it to the framework with π^μν, Π, μ's and charges. `--write-particlize` stores exactly
   these cells, and `hadronize.py` feeds them back to iSS the same way.
2. If the hydro handed over no surface: X-SCAPE's `SurfaceFinder` builds one **from
   `bulk_info`** at T_sw (`FluidDynamics::FindSurfaceFromEvolution`). With MUSIC's ideal store
   that surface already has π^μν = Π = 0; a slim copy would also lose P.
3. Otherwise: `read_in_FO_surface()` reads MUSIC's surface file from iSS's working directory.

Paths 1 and 3 don't touch `bulk_info`; path 2 does.

**Needed: e, s, T, vx, vy, vz: 6 of 28 floats.**

**Not read: P, `qgp_fraction`, μ_B/μ_C/μ_S, π^μν (16 floats), Π.** In this production 21 of
the 28 are zero anyway. MUSIC's store has no viscous or charge fields, and the fill writes
`get_fluid_cell_with_index`'s zeros (`HydroinfoMUSIC.cpp:315–352`, `MusicWrapper.cc:837–846`).
Only P carries information that the slim copy would drop, and nothing reads it.

## Options

### A: slim copy, 6 fields (recommended)

Keep `bulk_info`, but store 6 floats per cell instead of a `FluidCellInfo`.
`EvolutionHistory` already has the storage for it: the flat `data_vector` with a
`data_info` list of field names (`FromVector`, `FluidEvolutionHistory.h:168–178`), and
`GetFluidCell` builds a `FluidCellInfo` from it (`FluidEvolutionHistory.cc:191–297`).

- **Memory:** 24 B per cell, ~1.6 GB instead of ~7.5 GB. Peak ~20 → ~14 GB (store + slim
  copy during the spike), between runs ~12.3 → ~6.5 GB.
- **Output:** expected bit-identical. The 6 fields are the same floats, and the
  interpolation (`EvolutionHistory::get`, `FluidCellInfo` arithmetic) gives the same values
  for them. The other fields come back as 0, as they are now except P.
- **Time:** the fill writes 24 B instead of 112 B per cell, so it should get faster. The
  lookups must not get slower; see A1.

### B: no copy, read MUSIC's store (later, if A is not enough)

Answer the background lookups from MUSIC_1's own store, as `MpiMusic::GetHydroInfo_MUSIC`
already can (`MusicWrapper.cc:866`, unused). The pair writer would read the background from
the store too, as it already does for the jet leg.

- **Memory:** no copy and no spike: the store (~2 GB) stays for the event. Peak ~6–7 GB.
- **Output:** not bit-identical. `HydroinfoMUSIC::get_hydro_info` interpolates differently
  from `EvolutionHistory::get`, so Matter/LBT see slightly different media; this needs a
  physics comparison, not a byte comparison. Its lookup speed is unknown.
- **Also:** MUSIC_1 runs with `<dump_hydro_only>0` today so it fills `bulk_info`; B needs
  its store kept instead, including across `--reuse` groups.

### Not pursued now

Cutting the copy in η or τ: smaller, but it changes what Matter/LBT can see near the edges.

## Phases (option A)

Branches: X-SCAPE `slim_bulk_info` from `contrib`, js-contrib `slim_bulk_info` from `main`.
Don't rebuild `build_gpu` while a production job runs.

**Phase 0: baseline.**
- Cell count per leg: the `Total number of MUSIC fluid cells` line in the job log.
- Memory over time: `/usr/bin/time -v` for the peak, and RSS sampled every 0.5 s to see
  the spike and the plateau.
- Reference outputs: seeds 1–3, 2 events each, `--write-particlize both`, and a
  `--reuse 3` job (the regression recipe: `run_prod_jet.py --events 2 --seed 1 --reuse 2
  --write-particlize both --seed-registry none`).

**Phase 1: `EvolutionHistory` (X-SCAPE `framework/`).**
1. Fast sparse lookups: `GetFluidCell` resolves the entry names by string and allocates a
   `FluidCellInfo` on the heap on every call. Resolve the names to field ids once, when
   `data_info` is set, and fill a stack `FluidCellInfo`. `get` calls `GetFluidCell` 8–16
   times per query, and Matter/LBT query at every step of every parton.
2. `clear_up_evolution_data()` clears only `data`: clear `data_vector` too (it would leak
   the slim copy across events otherwise).
3. `get_data_size()` and the guards that test `data.size()` / `data.empty()`
   (`FluidDynamics.h:549`, `MusicWrapper.cc:610–615`): count records in either mode.
4. A way to start a slim history: a `data_info` setter plus `resize`, without the copy that
   `FromVector` makes.

**Phase 2: the fill (X-SCAPE `hydro/MusicWrapper.cc`).**
- `PassHydroEvolutionHistoryToFramework`: with the switch on, set `data_info` to
  `energy_density, entropy_density, temperature, vx, vy, vz` and fill `data_vector`, 6
  floats per cell, in the same parallel loop. Off: unchanged.
- The switch: a `<slim_bulk_info>` element in `<Hydro><MUSIC>` (default 0), and/or a setter
  for PyJetscape. Off by default: other setups read more (iSS through `bulk_info`, the
  hadronic modules, user code via `to_numpy_full`).
- **Guard for surfaces built from the evolution.** `FluidDynamics::FindSurfaceFromEvolution`
  stops with an error on a slim history: the surface needs P, π^μν and Π. The error says to
  hand over MUSIC's own surface (path 1) or to switch the slim copy off. Where both are
  configured for one hydro (slim copy on, and a `SoftParticlization` that will fall back to
  path 2), warn already at `InitTask`, before the event runs.

**Phase 3: PyJetscape (js-contrib).**
- `to_numpy` / `to_numpy_full` (`bind_evolution.cc`): read either layout. With a slim
  history, return the fields it has; ask for a missing one (e.g. P) → a clear error, not
  zeros.
- `run_prod_jet.py`: turn the switch on for MUSIC_1 (an option, on by default in this
  production once validated). Record it in the `.json`.

**Phase 4: validation.**
- Bit-identical against Phase 0: every dataset of the pair and particlize files (skipping
  `wall_s`, uuids, paths), the showers and the droplets, for all reference jobs.
- Memory: peak and plateau against Phase 0; the target is a peak ≤ 15 GB.
- Time per event: no regression beyond noise (±2 %); the fill should be faster.
- `tests/test_prod_seeds.py` and the PyJetscape tests.
- The guard: a slim history plus `FindSurfaceFromEvolution` stops with the message; a
  full history still builds the surface as before.

**Phase 5: docs and defaults.**
- The memory numbers in `prod_AuAu_0_10_jet/README.md`, `BENCHMARK_GB10.md`, and the
  sizing in `BuildContainerProd.md` (~22 GB per job → the new value).
- `utils/slurm_prod_array.sh`: `--mem` for `P=4`.

## Open questions

- **Other readers in other configurations.** The list above covers this production. The
  switch is off by default, and a slim history should refuse loudly (Phase 3) instead of
  returning zeros where a field is missing.
- **Surfaces from `bulk_info` (path 2).** Never in this production, which uses MUSIC's own
  surface. Elsewhere it is a silent physics risk even today: with MUSIC's ideal store the
  surface lacks π^μν and Π, so iSS samples without viscous corrections and nothing says
  so. The guard above covers the slim copy. A warning for the existing case (path 2 on
  an ideal store) would be worth a separate X-SCAPE change.
- **P.** Nothing here reads it. Adding it later costs 4 B per cell.
- **The jet leg (MUSIC_2)** is read from MUSIC's store (`dump_hydro_only`) and has no
  framework copy; this plan doesn't touch it.
- **Option B** stays open if a peak of ~14 GB still limits `-j`.
