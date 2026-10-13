<!-- Plan, written 2026-10-13. Status: not started. Branch to use: oo_reader (from final_optim). -->

# Plan: O+O hadrons in `HadronFileReader`, `run_h5toROOT.py` and a flat ROOT ntuple

## Goal

The O+O ISR production (`prod_OO_isr`) writes its hadrons in the same `jetscape.hadrons_h5`
format as the Au+Au jet production, but the readers and the ROOT export built for Au+Au do not
take them. Today an O+O analysis uses the hand-written loader of
[`README_OO.md`](../README_OO.md) §5 and, for ROOT, a per-file conversion plus an `events` table
written with uproot (§10). The goal:

1. **One Python reader for both productions:** `HadronFileReader` opens an O+O campaign, with
   its event table, cross sections and weights, and streams it in bounded memory.
2. **`run_h5toROOT.py` converts an O+O campaign** into the same ROOT structure as Au+Au (minus
   `bulk_jet`), with a campaign file.
3. **A flat analysis ntuple** in which one entry is one complete O+O event sample (bulk +
   jet hadrons + the event's record and weights), so a ROOT analysis needs no joins.
4. Old files keep working: the O+O campaigns made so far (`hard_sub_compare/sub`, `nosub`,
   any production) are read without being rewritten.

Not a goal: changing the HDF5 format of the hadrons, or X-SCAPE.

## What the code says

**The O+O files** (`run_prod_oo_isr.py`, `HadronH5Writer` calls around lines 605–630):

| file | units | per-unit columns | attributes |
|---|---|---|---|
| `<stem>_hadrons_jet_frag.h5` (always in the job) | one per event, 1 sample | the whole event record (`pthat`, `pthat_bin`, `glauber_*`, `has_medium`, `hydro_row`, `isr_sub_*`, `remnant_*`, `strings_*`, `parton_*`, `sigma_gen`, `event_weight`, `tau0_music`, `inline_iss_seed_bg`), **no `seed`** | `inline`, `source` (`<stem>.h5`), `system`, `subtract_hard_momentum`, `prod_seed`, at the end the per-window `pthat_bin_*`, `events_skipped`; **no `source_uuid`** |
| `<stem>_hadrons_bulk_bg.h5`, in the job (`--bulk-hadrons N`) | one per event, N samples (0 without a medium) | the record + `seed`, `n_cells` | as above, plus `music_input`, `hadronize_xml` |
| `<stem>_hadrons_bulk_bg.h5`, from `hadronize.py --tags bulk_bg` (setups B/C) | one per background = event | `unit`, `event`, `n_samples`, `seed`, `n_cells` only | `source_uuid` of the particlize file |
| `<stem>_particlize.h5` | only setups B/C (not with `--no-particlize`) | `events/` with the record, `bg_unit` | `file_uuid`, the per-window `pthat_bin_*` |

Unit i of the jet file and unit i of the bulk file are the same event (prod_OO_isr README,
*Hadronization*). The usual production setting writes hadron files only: no particlize file.

**Why the Au+Au tools refuse them:**
- `HadronFileReader` (`jetscape/hadrons_h5.py`):
  - `_find_stems` raises `FileNotFoundError` without a `<stem>_particlize.h5`.
  - `_open_stem` takes the event count, `bg_unit`, the windows and σ from the particlize file,
    and refuses a hadron file whose `source_uuid` differs from its `file_uuid` (in-job files
    have none: `'' != uuid`).
  - `TAGS` and the per-event logic assume `bulk_jet` + `bulk_bg` (+ `jet_frag`);
    `jet_event` builds from `bulk_jet`.
- `run_h5toROOT.py`:
  - `event_table` reads `events/` from the particlize file and `units["seed"]` from every tag:
    `KeyError: 'seed'` on the O+O jet file (2026-10-13, `prod_OO_isr/out`).
  - `write_campaign` builds the campaign file through `HadronFileReader`, which refuses them.
- `analysis_root/HadronFileReader.h`: the tags `bulk_jet`, `bulk_bg`, `jet_frag` and the
  `events` table of `run_h5toROOT.py`; the per-event calls are the Au+Au pair's
  (`bkg`, `bkg_dep`, `bkg_dep_frag`).
- `root_export/hadrons_to_root.py` converts any single hadron file (checked on O+O), but
  without the event record.

**Scale.** The README_OO loader holds a campaign in memory: in `hard_sub_compare/sub`
(0–80 %, 10 oversamples) ~185 bulk hadrons per sample, so ~77 B per hadron with the derived
kinematics, ~1.4 GB per 10⁴ events, ~15 GB per 10⁵. At 60 events per job a 10⁵-event
campaign is ~1,700 jobs, ~3,400 hadron files of a few MB (in `sub/`, 75 events without
positions: bulk 3.1 MB, jet 0.6 MB): the cost of opening every file at start matters, the data
volume much less.

## Design

### 1. Production side: self-describing O+O hadron files (small)

In `run_prod_oo_isr.py` (js-contrib only):
- a **job id**: a `job_uuid` attribute (uuid4) on every file the job writes, and on the hadron
  files also `source_uuid` = the particlize file's `file_uuid` when there is one, else the
  `job_uuid`. Pairing then works the same way for both setups;
- **`layout = "single_leg"`** on the hadron files (Au+Au files have no attribute: `pair`);
- **`seed` in the jet units** (the framework's per-event seed if PyJetscape exposes it, else
  −1), so every tag has the same unit columns;
- the per-window `pthat_bin_*` attributes already exist on every file; keep them.

Bit-identical hadrons; only attributes and one unit column are added. `format_version` of
the hadron files stays (readers must not require the new attributes).

### 2. `HadronFileReader`: layout detection and the O+O case

- **Discovery:** `_find_stems` accepts a stem with a `_hadrons_jet_frag.h5` (or
  `_hadrons_bulk_bg.h5`) and no particlize file when the hadron files are `single_leg`
  (attribute, or for old files: `inline` set and `prod` = `PyJetscape/example/prod_OO_isr`).
- **`_open_stem` for `single_leg`:**
  - the event table from the jet file's `units/` (the particlize file's `events/` when it
    exists, cross-checked by count);
  - `bg_unit` = the event's own unit (identity);
  - windows and σ from the hadron files' attributes;
  - the uuid check against `job_uuid` / `source_uuid` when present; for old files pair by stem
    and require equal unit counts in all tags (a clear error otherwise).
- **`TAGS` per layout:** `pair` (`bulk_jet`, `bulk_bg`, `jet_frag`), `single_leg` (`bulk_bg`,
  `jet_frag`). `hist`, `total`, `events=`, `pthat_bin_*` work unchanged.
  `jet_minus_background` raises "no background leg in an O+O campaign".
- **New calls (both layouts where they make sense):**
  - `r.event_table(columns=None)` → dict of per-event arrays: the record, `file`,
    `local_event`, the samples per tag;
  - `r.full_event(e, k)` → bulk sample k + the event's jet hadrons, `origin` 0/1 (O+O's
    counterpart of `jet_event`);
  - `r.weights(kind, sigma_nn=None)`: `"hard"` (σ_k / N_k), `"collision"` (Ncoll σ_k /
    σ_NN / N_k; `sigma_nn` required, 42 mb at 200 GeV), `"ncoll"` (Ncoll), `"none"`;
  - `r.iter_chunks(tag, fields, events=None, files_per_chunk=…)` → flat numpy arrays with
    `event` (global) and `sample`, file by file or N files at a time, so a campaign streams in
    bounded memory; `awkward=True` gives events × samples × hadrons for FastJet's array
    interface.
- **Analysis helpers** (a small `jetscape.oo_analysis` module, numpy only): the
  different-oversample v_n{2} of README_OO §8, the per-event sample-mean histogram with the
  per-event spread as the error, and the multiplicity-estimator centrality classes of
  `hard_sub_compare.ipynb`. They take `iter_chunks` output, so they stream too.
- **Optional, after measuring:** a campaign index (`<campaign>_index.json`: stems, event
  counts, offsets, windows, σ, attributes), written on first open and reused when no file is
  newer, if opening ~3,400 files at start turns out to cost more than a few seconds.

### 3. `run_h5toROOT.py`: O+O support

- The same layout detection (through the generalized reader).
- `event_table`: from the reader's `event_table()`; `seed_*` columns −1 where a tag has
  none; `n_samples_bg`, `n_samples_frag`; the initiators from the jet file's `initiators/`
  (present in O+O jet files).
- Tags default per layout (`bulk_bg,jet_frag` for O+O).
- `write_campaign` unchanged in its tables (`windows` with `sigma_mb`, `weight_mb`; `files`),
  now fed for both layouts. Add `sigma_nn_mb` to the provenance when given (`--sigma-nn`), so
  a per-collision weight is reproducible.
- Output per production file: `bulk_bg`, `jet_frag`, `events`, `windows`, `provenance`, the
  Au+Au structure without `bulk_jet`.

### 4. A flat analysis ntuple (`run_h5toROOT.py --flat`)

One entry = one event sample, everything an analysis loop needs on it:

| column | |
|---|---|
| `event`, `file_index`, `local_event`, `sample`, `n_samples` | the global event, its file and row, the bulk oversample |
| `pthat`, `pthat_bin`, `glauber_npart`, `glauber_ncoll`, `glauber_b`, `has_medium`, `parton_*` | the record (scalars) |
| `w_hard`, `w_coll`, `w_ncoll` | the weights of §2, **already divided by `n_samples`**: a weighted Fill gives per-event-normalised results |
| `n`, `pid[n]`, `E/px/py/pz[n]`, `origin[n]` | bulk sample k and the event's jet hadrons (origin 0/1); `t/x/y/z[n]` unless `--no-x` |

- **Events without a medium** get one entry (`sample` 0, `n_samples` 0 → weights use 1) with
  only the jet hadrons, so they stay in the sample.
- **Size:** the jet hadrons are repeated per sample: ~120 per event against ~185 bulk hadrons
  per sample in 0–80 %, so +6 % per entry at 10 oversamples (measure; more in central events).
- **`--merge N`:** one ROOT file per N production files with a global event index (O+O files
  are small; fewer files are easier to move and chain). The campaign file stays separate.
- Written for O+O first; for Au+Au the same layout would hold bulk_jet + jet_frag per sample
  (the jet event) and is a later option.
- RNTuple by default, `--format ttree` as now; uproot and ROOT writers as in
  `hadrons_to_root.py`.

### 5. C++ (optional)

The flat ntuple makes a reader unnecessary for most analyses (RDataFrame with the weight
column). If wanted: a `single_leg` mode in `HadronFileReader.h` (`bulk(e, k)`, `jet(e)`,
`full_event(e, k)`, `info(e)` with the record and the weights), detected from the
`provenance` layout.

## Validation

- **Bit-for-bit reading:** on `hard_sub_compare/sub` and `nosub` (old files, no new
  attributes) and on a new O+O campaign with §1 (all setups A, B, C), the reader's hadrons
  equal the files' (pid, p, sample offsets), event by event.
- **Observables, three ways:** dN_ch/dη at |η| < 0.5, a charged pT spectrum, the bookkeeping
  identity and v₂{2, different samples}, each from (a) the README_OO loader, (b)
  `HadronFileReader` (`hist`/`total`, `iter_chunks`) and (c) the flat ntuple in RDataFrame and
  uproot: equal to 1e-12 relative. Reference values from the 2026-10-13 test: 2,991 events,
  σ per window 7.59e-4 / 2.23e-5 / 1.41e-6 mb, Ncoll-weighted dN_ch/dη 36.71 ± 0.33,
  v₂{2} 0.0313, bookkeeping 3e-9, 4.9 % without a medium.
- **ROOT export:** entries = samples per tag; `events` = the jet file's units; the campaign
  file's `sigma_mb` = the reader's `pthat_bin_sigma`; a TTree and an RNTuple variant; the uproot
  and ROOT writers.
- **Au+Au unchanged:** the existing tests (`test_particlize_h5.py`, `test_run_h5toROOT.py`,
  `test_hadrons_to_root.py`, `test_analysis_root_reader.py`) pass, and `run_h5toROOT.py` on
  `prod_AuAu_0_10_jet/out` writes the same files as before (compare the ntuples).
- **New tests:** a synthetic O+O fixture (a few events from `HadronH5Writer` with the O+O
  attributes, with and without a particlize file, old and new attributes) in
  `tests/test_oo_reader.py`; O+O cases in `test_run_h5toROOT.py`.
- **Scale:** `iter_chunks` over a 10⁵-event campaign (or the test campaign replicated) within
  a fixed memory budget; the time to open the campaign with and without the index.

## Order and effort (estimates)

| step | effort | depends on |
|---|---|---|
| 2. `HadronFileReader` for `single_leg`, the new calls, the tests | 1–1.5 days | — |
| 1. the attributes in `run_prod_oo_isr.py` | 0.5 day | — (readers must work without it) |
| 3. `run_h5toROOT.py` for O+O | 1 day | 2 |
| 4. the flat ntuple, `--merge` | 0.5–1 day | 2, 3 |
| `jetscape.oo_analysis` helpers | 0.5 day | 2 |
| docs: README_OO §5 and §10, README_analysis | 0.5 day | 2–4 |
| 5. C++ `single_leg` mode | 0.5–1 day | 3; only if asked for |
| the campaign index | 0.5 day | 2; only if the open time measures large |

## Open questions

- **The jet units' `seed`:** does PyJetscape expose the per-event seed of the in-job
  hadronization? If not, −1 and the job's `prod_seed` are what there is.
- **σ_NN for the per-collision weight:** take it from 3dMCGlauber's parameters (the value it
  samples collisions with) rather than a fixed 42 mb, if it is accessible from the job.
- **The flat ntuple for Au+Au:** same layout with bulk_jet + jet_frag (and optionally the
  paired bulk_bg sample) per entry: worth it once the O+O version is in use.
- **Centrality in the files:** store a multiplicity-estimator value per event and sample
  (e.g. charged at 3.1 < |η| < 3.9 from one sample) in the event table, so ROOT users can
  classify without re-reading the hadrons.
