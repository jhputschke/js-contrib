<!-- Plan, written 2026-10-14. Status: not started. Assumes PLAN_smash_ready_hadrons.md Parts 1 and 2 are done. Branches to use: auau_hadrons_only in js-contrib (from main) and in X-SCAPE (from contrib, the iSS wrapper method of step 1). -->

# Plan: Au+Au jet productions that write hadron files only, in one job

## Goal

**One job per output file set**, as O+O does (`--no-particlize --no-hydro-file --bulk-hadrons N`):
`run_prod_jet.py` runs MUSIC, the jets, iSS on both legs and the jet hadronization in the same
process and writes **only hadron files**. No pair file, no particlize file, no intermediate file
of any kind, no stage 2.

**Little or no oversampling:** one sample per jet event (or a few), the background oversampled
by its number of jets at most, a few fragmentations. The hadron files then are small. The
extra CPU in the GPU job is ~8 % of stage 1 and hardly depends on the oversampling (iSS's cost
is mostly fixed per surface, see below), so the oversampling is limited by disk, not CPU.

The files hold everything for analysis now and for later work without the hydro:
- the decayed hadrons (`bulk_jet`, `bulk_bg`, `jet_frag`), as stage 2 writes them;
- the undecayed primaries of the bulk and of the jets, with positions, for SMASH later
  ([`PLAN_smash_ready_hadrons.md`](PLAN_smash_ready_hadrons.md) Parts 1–2; SMASH is its Part 3);
- the final partons with their hadronization points, so the jet hadronization can be redone;
- the event record and the background mapping the readers need.

Assumed done: `PLAN_smash_ready_hadrons.md` Parts 1 and 2 (iSS decays the SMASH list,
`keep_primaries` for bulk and jets, jet-hadron positions, per-group precision).

## Why it pays at low oversampling

| per jet event | stage 1 + stage 2 today (c2) | hadrons only, 1 sample (estimate) |
|---|---|---|
| files kept | pair 0.146 GB + particlize 0.097 GB, then hadrons ~0.077 GB | hadrons + primaries + partons: **~0.5–1 MB** (~0.2 MB per final sample and leg at full precision: 100 MB per leg at 500 samples, README_2stage §1; primaries about as much again) |
| extra CPU in the GPU job | none | iSS ~2.6 s per jet-leg surface + ~2.6 s per background (~0.2 s per event at 15 jets per background), fragmentations ~4 ms each: **~2.8 core-s per event, ~8 %** of stage 1's ~35 core-s (GB10, one thread; ~3.2 s per surface with EOS 91's ~400 species, scaled, not measured) |
| stage 2 | ~9–11 core-s per event on CPU nodes | none |

**iSS's cost per surface is mostly fixed** (PLAN_iSS_optim.md Part C, GB10, one thread, current
iSS): 2.6 s at 1 and at 10 samples, 3.1 s at 100, 5.4 s at 500, 8.2 s at 1000; that is ~2.5 s for
the yields of 321 species in ~1 M cells and the surface import, plus ~5.6 ms per sample. So 100
samples per event cost ~0.5 s more than one, but ~20 MB per leg and event on disk: **the
oversampling is set by disk**. On host-bound nodes (the 3090 box, Delta A40) the ~8 % comes
off events/h; on DeltaAI it fits into the spare cores.

What it costs: no surfaces are kept, so **no new samples or other iSS settings later**; the bulk
can only be re-decayed or sent through SMASH from its stored primaries. Fewer samples per event
also means more events for the same statistics, which stage 1 pays for in GPU time; that is the
trade-off of choosing this mode.

## What exists

- **`hadronize.py`'s per-unit loop** (`hadronize.py:651–716`): `run_iss(cells, seed, n)` loads a
  surface into `jetscape.surface_replay.SurfaceReplay`, sets iSS's sample count and seed
  (`core.soft_set_number_of_samples`, `core.soft_set_next_random_seed`), runs one framework
  event and takes the hadrons (`core.soft_hadrons_numpy`); `core.hadronize_partons(colorless,
  partons, seed)` fragments a parton list, once per fragmentation. Seeds come from
  `unit_seed(base, tag, unit[, sample], prod=…)`; `--common-seeds` gives each jet leg its
  background's seed, `--correlated` addresses iSS's random numbers by cell. It runs in **its own**
  JETSCAPE instance (SurfaceReplay → iSS).
- **`run_prod_jet.py --validate-inline`** (`:346–364`, `:575–604`): iSS as a task in the
  production's own task list with `hydro_id MUSIC_2`, plus Colorless with `reseed_per_event`;
  hadrons written with `HadronH5Writer`. Validation only (it insists on `--write-particlize`),
  the jet leg only, one fragmentation.
- **The particlize writer** (`ParticlizeH5Writer`) already collects, per event, everything stage
  2 needs: both legs' surface cells (the background once per background), the final partons,
  the initiators, the event record (`events/`, with `bg_unit`, the pT̂ windows, weights, the
  edge flags) and `music_input`.
- **O+O's in-job writer** (`run_prod_oo_isr.py:605–630`): the event record per unit in the hadron
  files (`units/`), `music_input` and the settings as attributes; no particlize file.

## Design

**Run `hadronize.py`'s per-unit work inside the production job, on the particlize writer's
in-memory data instead of a file.** Concretely:

1. The per-unit work moves out of `hadronize.py` into a library class
   (`jetscape.hadronizer.Hadronizer`): bulk of one leg (cells, seed, n samples), fragmentations
   of one event (partons, seeds), the writers. `hadronize.py` keeps its options and file handling
   and calls it, unchanged in behaviour.
2. `ParticlizeH5Writer` gets a **memory sink**: per event it hands the same arrays it would write
   (surface cells of the jet leg, of a new background, partons, initiators, the event record) to
   a callback instead of the file. `--write-particlize` stays as it is; the hadrons-only mode
   uses the sink and writes no particlize file.
3. `run_prod_jet.py --hadrons-only` connects the two: per event, iSS on the jet leg, iSS on the
   background when it is new, the fragmentations, all with the seeds `hadronize.py` would use;
   the hadron files and nothing else.

So the in-job hadrons are, by construction, what `hadronize.py` would make from that job's
particlize file with the same settings and seeds, which is also how it is validated.

## Changes

### 1. iSS on given cells inside the production process (1 day)

`hadronize.py` samples a surface by running one event of its own JETSCAPE instance
(SurfaceReplay → iSS). Inside the production there is already a JETSCAPE instance with MUSIC,
the jets and the liquefier, whose `ExecPerEvent` must not run again. And the framework has room
for **one** soft-particlization module only: `<SoftParticlization><hydro_id>` is a single XML
value (`JetScape.cc:1173–1193`), the signal manager keeps one soft-particlization pointer
(`iss_pointer_is_set`, used by the afterburner and the writers too), and iSS gets its cells only
through that signal (`getSurfCellVector` → `GetHydroHyperSurface`,
`iSpectraSamplerWrapper.cc:140, 252`); a module outside the wiring gets none and falls back to
reading a surface file.

**The way: one iSS, called on given cells.**
- X-SCAPE: a public method in `iSpectraSamplerWrapper` that samples a given cell array with a
  given seed and number of samples (the steps of `ExecuteTask`: `music_input`, the cells into
  iSS, the sampling, the compact output and the primaries), ~30–60 lines, `ExecuteTask` itself
  unchanged. The iSS module stays in the task list for its settings and `Init`, switched off for
  the framework's event loop.
- PyJetscape: its binding, `core.soft_sample_cells(iss, cells, seed, n_samples)` → the arrays of
  `core.soft_hadrons_numpy` (and of the primaries).
- `hadronize.py` uses the same call instead of its extra event loop through SurfaceReplay, so
  the job and stage 2 run one code path.

**Not chosen: two iSS tasks, one per leg** (`hydro_id MUSIC_1` and `MUSIC_2`, as
`--validate-inline` does for the jet leg):

| | one iSS on given cells (chosen) | two iSS tasks |
|---|---|---|
| C++ | a new wrapper method and its binding; no framework change | a hydro_id and signal connections per soft-particlization instance in `JetScape.cc` and `JetScapeSignalManager`, shared code everything else uses |
| iSS in memory | once; both legs one after the other, as `hadronize.py` does | twice, in every production job |
| seeds, samples, the background once per background | set per call by the driver, as `hadronize.py` does | injected per task before each framework event; the background task skipped on reused backgrounds (relies on iSS's behaviour on reuse, to verify) |
| equal to stage 2 | by construction (same call, same seeds) | only by testing |
| upstream risk | low: a new method | higher: changes how the framework wires soft particlization |
| downside | a second way into iSS next to `ExecuteTask`, to keep consistent with it | all of the above |

### 2. `jetscape.hadronizer` (1½–2 days)

The per-unit code of `hadronize.py` as a class: `bulk(leg, unit, cells, n_samples)`,
`fragments(event, partons, n_frag)`, `append_*` to the writers, the seed scheme (`unit_seed`,
`--common-seeds`, `--correlated`), the selection and precision options, the primaries groups
(smash_ready 1.6, 2.1), and `partons/` with the hadronization points in the jet file (below).
`hadronize.py` becomes a thin driver; its outputs must stay bit-identical.

### 3. The particlize writer's memory sink (½–1 day)

`ParticlizeH5Writer(sink=callable)`: the same collection and checks, no file. The background's
cells are handed over once per background, with its unit number, as the file stores them.

### 4. `run_prod_jet.py --hadrons-only` (1 day)

- Implies the surfaces of both legs and the memory sink; no pair file, no particlize file
  (`--particlize-only` and `--write-particlize` are refused with it).
- Options as `hadronize.py`: `--oversample` (default 1), `--oversample-bg`
  (`per-pthat-bin` / `auto` / N), `--n-frag` (default 1), `--keep-primaries`,
  `--keep-primaries-samples K`, the selections and precisions, `--correlated`; `--hadronize-xml`
  as now. `--validate-inline` becomes a special case of it (and can stay as an alias).
- **Output, per job:** `<stem>_hadrons_bulk_jet.h5`, `<stem>_hadrons_bulk_bg.h5`,
  `<stem>_hadrons_jet_frag.h5` (each with its `*_primary` groups) and `<stem>.json`. The event
  record goes into `units/` of `jet_frag` and `bulk_jet` (as O+O's jet file), `bg_unit`
  included; the background file's `units/event` is the first event that used it, as today.
  Attributes: `hadrons_only = true`, a `file_uuid` for the job (the seeds' `prod` and the
  readers' consistency check use it, as the particlize file's does today), `music_input`, the
  hadronize settings, the stage-1 provenance.
- **`partons/` in the jet file:** the final partons (`FINAL_PARTON_COLUMNS`) and, as
  `partons/x_had`, their hadronization points, so `hadronize.py --tags jet_frag` can redo the
  jets later from the jet file alone. In the job the point is found against the medium the
  jets saw (the background's `bulk_info`, as Matter/LBT and Hybrid Hadronization look it up);
  smash_ready 2.2's offline variant uses the stored jet-leg surface. The attribute says which.
- `run_jobs.sh`, `slurm_prod_array.sh`, `launch_2gpu.sh`: pass the options through (as for
  O+O); the restart test (`.json` complete) unchanged; `upload_follow.py` follows the hadron
  files.

### 5. Readers (1–1½ days)

`HadronFileReader` today needs `<stem>_particlize.h5` for `events/`, `bg_unit`, the initiators
and the `file_uuid` check (`hadrons_h5.py:556–621`, `:825–835`). For hadrons-only files it takes
them from the jet file's `units/`, `initiators/` and attributes instead: the "no particlize
file" layout that [`PLAN_OO_root_reader.md`](PLAN_OO_root_reader.md) introduces for O+O, here
with two legs and the background mapping. Do both together. `run_h5toROOT.py` follows the
reader. `hadronize.py --tags jet_frag` accepts a hadrons-only jet file as input (its
`partons/`).

### 6. Docs (½ day)

README_2stage: a "hadrons only, one job" setup (Docker, Apptainer, SLURM), the table above, what
is lost; the c-campaign recipes unchanged.

## Validation (1–2 days)

- **Identical to two stages:** 2–3 jobs once with `--hadrons-only` and once with
  `--write-particlize both --particlize-only` followed by `hadronize.py` with the same options
  and seeds: hadron files identical in every group (attributes aside). This replaces what
  `--validate-inline` checks today, for both legs and n fragmentations.
- **`hadronize.py` unchanged:** its outputs bit-identical before and after the refactor (step 2),
  on a c1 file.
- **Redo the jets:** `hadronize.py --tags jet_frag` on a hadrons-only jet file reproduces its jet
  hadrons with the stored seeds.
- **Readers:** a hadrons-only campaign and the same campaign in two stages give the same
  samples through `HadronFileReader` and `run_h5toROOT.py`.
- **Cost:** events/h, CPU and peak host memory per job against the same campaign without
  hadronization, on the 3090 box (10 jobs per GPU) and the GB10. Expected: ~8 % more CPU per
  event (~2.8 core-s), so ~8 % fewer events/h where the cores are the limit; iSS's tables and
  per-cell arrays in every job add memory (a few hundred MB, to measure).

## Effort

| step | days |
|---|---|
| 1 iSS on given cells: wrapper method, binding, `hadronize.py` on it | 1 |
| 2 `jetscape.hadronizer`, `hadronize.py` on top of it | 1½–2 |
| 3 memory sink | ½–1 |
| 4 `--hadrons-only`, outputs, partons, scripts | 1 |
| 5 readers (with PLAN_OO_root_reader's layout) | 1–1½ |
| 6 docs | ½ |
| validation | 1–2 |
| **total** | **6½–9** |

Estimates of focused working days, about ±50 %; not included: PR review, image builds,
production-scale runs.

## Open points

- **The default oversampling:** chosen by disk, since the CPU barely changes up to ~100 samples:
  1–10 per jet event, the background `per-pthat-bin` (M samples for M jets per window), 1–5
  fragmentations? With few samples the jet − background pairing (`--correlated`) matters more,
  not less.
- **Primaries for all samples** (cheap at this oversampling) or only K.
- **Memory per job**, against the ~7 GB per job of the sizing tables. Expected below it:
  - **No shared-table trick as for LBT** (`LBT_TABLE_CACHE`): with EOS 9/91 iSS's static tables
    are a few MB (the hadron list 95 KB / 290 KB, `chosen_particles_*`, `HRGEOS_PST-*` 138 KB,
    the δf coefficients); the one large table, `EOS_tables/HRG4D` (72 MB), is read only for
    EOS 20 (`readindata.cpp:144`).
  - **The cost is the surface being sampled:** `FO_surf` (36 floats and 3 vectors, ~250 B per
    cell with their heap parts) and `FO_surf_LRF` (~170 B per cell), ~0.4–0.5 GB for a ~1 M-cell
    Au+Au surface, plus ~0.1 GB of packed yield arrays and the inverse CDF while sampling
    (estimate from the structs; the whole `hadronize.py` process peaks at 0.90 GB, Python and
    libraries included). Different in every job, so nothing to share; `iSS::clear()` keeps the
    capacity, so it is a steady ~0.5 GB per job, not a growth.
  - **Hadrons-only drops the pair file,** and with it MUSIC_2's native store (~2.3 GB), kept
    today only for the pair writer: a job is like `--particlize-only` (~5 GB per job in the
    sizing tables), so **~5 GB + ~0.5 GB ≈ 5.5 GB**. To measure (validation, *Cost*).
  - If the 0.5 GB matters: free `FO_surf` once `FO_surf_LRF` is built, if nothing reads it later
    (about half, in iSS, to check for bit-identity); release the capacity after each surface
    (`shrink_to_fit`), in case iSS's peak meets the next event's hydro peak (iSS runs at the end
    of an event); hand the cells to iSS without intermediate copies in step 1's call, and drop
    the background's cells once it is sampled.
