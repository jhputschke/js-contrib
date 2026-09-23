# One FNO4d HDF5 writer for fast_data, FastHydro and PyJetscape

## Context

Three pieces of code write the FNO4d training HDF5 schema (`arr`, `ntau_freezeout`,
`tau_freezeout` + grid attributes):

| | `FNO4d:loc_libs/fast_data/writer.py` | `FastHydro/python/fast_data/writer.py` | `PyJetscape/python/jetscape/fno_h5_writer.py` |
|---|---|---|---|
| Status | upstream original | verbatim vendored copy, sha-checked (`VENDORING.md`, `tests/test_vendor_intact.py`) | **fork**, hand-modified |
| Used by | `fast_data/driver.py:158-224` | `fasthydro/h5_writer.py` `PairedH5Writer` | `jetscape/fast_h5_bulk.py` `H5BulkWriter`, `repad_h5.py` |
| Event axis | pre-allocated, fixed | same | grows per event (`ensure_capacity`) |
| Write unit | whole event (`append_event`) | same | one τ frame (`write_frame` + `set_event_meta`) |
| `arr` chunk | `(1,4,nx,ny,neta,T)`, whole event | same | `(1,4,nx,ny,neta,1)`, one frame |
| `--resume` | yes | yes (unused by `PairedH5Writer`) | no |
| Optional groups | `source/`, `diag/`, `eos/` | same, plus `arr_bg*` and `shower/` added by `PairedH5Writer` | none (dropped) |
| `format` attr | `fast_data/hydro_evolution` | same | `xscape/hydro_evolution` |
| `repad_to` | no (docstring points to PyJetscape) | no | yes |

Paths in this plan: `FNO4d:` = `~/FNO4d`, `X-SCAPE:` = the X-SCAPE root, and
`FastHydro/…` and `PyJetscape/…` are relative to `external_packages/js-contrib/contribs/`.

**The core schema is the same**: channels, grid attributes, zero tail after freeze-out,
`nevents == arr.shape[0]`. FNO4d's loaders read all three outputs. The copies differ in how
they write, and in two places the files actually disagree:

1. **`ntau_freezeout` has two definitions under one name.**
   - Legacy `X-SCAPE:src/root/RootBulkWriter.cc:278-279` sets
     `ntau_freezeout = int((tau_fo - tau_min)/dtau) + 1`. Its own comment says the value
     "ends up being about 2 units too large". The reference data `FNO4d:data/dAu_25ev_mb.h5`
     came from this writer, and fast_data copied its convention:
     `ntau_freezeout = populated frames + 1`, pinned by `FNO4d:tests/test_fast_data_writer.py`
     w3 (`live_tau_lengths == ntau_freezeout - 1`) and w4
     (`tau_freezeout == tau_min + (ntau_freezeout-1)*dtau`).
   - `X-SCAPE:src/root/FastRootBulkWriter.cc:80,133,195` and PyJetscape's `H5BulkWriter`
     (`fast_h5_bulk.py:229`) set `ntau_freezeout = frames written`, with
     `tau_freezeout = tau_min + ntau_native*dtau` on MUSIC's grid.
     `PyJetscape/tests/test_h5_bulk.py:279` pins `live_tau_lengths == ntau_freezeout`.
   - With native grid and stride 1, both give the same `tau_freezeout`, and `ntau_freezeout`
     differs by exactly one. W4 fails on any `H5BulkWriter` file. Readers that derive lifetimes
     from the data (`FNO4d:loc_libs/data/dataset.py:173 live_tau_lengths`) are unaffected.
     Readers that trust the integer (`FNO4d:loc_libs/downsample.py:704`,
     `fast_data/viz.py:142`, `choose_ntau = max(ntau_freezeout)` in `root_to_hdf5.py`) are off
     by one on files from the other convention.
2. **`repad_to` misses datasets it doesn't know about.** `PyJetscape/python/jetscape/fno_h5_writer.py:277`
   hard-codes `_TAU_DATASETS = (arr, source/S, source/P_cart)`. On a FastHydro pair file it grows
   `arr` but not `arr_bg`, so `PairBrowser` (`FastHydro/python/fasthydro/browse.py:31-46`) then
   sees two legs of different τ length. The consistency check does not catch this, because it
   only looks at the listed datasets.

There is also duplication that will get worse. Variable-length per-event data ("ragged": a flat
table + `offsets` + a column-name attribute) is written by three separate hand-written
implementations:
- `source/droplets`, in `writer.py close()`;
- `shower/`, in `FastHydro/python/fasthydro/h5_writer.py:195-253`;
- the replay npz, in `FastHydro/python/fasthydro/droplets_io.py`.

All of them hold the data in memory and write it at `close()`, so a crash loses it even
though `arr` is on disk. A `hadrons/` group is next (see *Phase 5*). Without this refactor it
would become a fourth copy and have to be added to two writers.

**Principle.** One file format, one low-level writer class, in one place. The modules that
produce the data stay separate: `H5BulkWriter` (MUSIC native store, frame streaming,
1–2 GB events), `PairedH5Writer` (background/jet pair, source, shower) and
`fast_data.driver` (standalone generation) do different jobs and should all call the same
writer.

**Where it lives.** Upstream in `FNO4d:loc_libs/fast_data/writer.py`. FNO4d owns the reader
contract, and FastHydro's vendoring rule already says upstream wins. PyJetscape can't import
FastHydro's copy: FastHydro imports `jetscape` (`fasthydro/pipeline.py:109`), so that would be a
circular dependency. PyJetscape therefore gets its own **verbatim** vendored copy with the same
sha check FastHydro uses. That makes two byte-identical copies, both enforced by tests, and no
more forks.

## Decisions to make first

Each decision has a recommendation. Record the choice in this file before starting Phase 1.

- **D1 — freeze-out convention for new files.** Recommend:
  - `ntau_freezeout` = number of populated τ frames (= `live_tau_lengths`);
  - `tau_freezeout` = the physical time the hydro ended, on the source grid;
  - both FastRootBulkWriter and H5BulkWriter already do this, and it is what the name says;
  - bump `format_version` to 2 and add a machine-readable
    `attrs["freezeout_convention_id"] = "frames_written"`;
  - v1 files are read as `"legacy_plus_one"`.
  - Cost: fast_data output changes by −1 in `ntau_freezeout`; FNO4d tests w3/w4 and the
    `viz.py:142` docstring change. Alternative: keep "+1" everywhere and change
    H5BulkWriter/FastRootBulkWriter instead. That keeps the known-bad legacy value alive and
    changes a C++ writer, so it is not recommended.
- **D2 — default `chunk_tau`.** Recommend keeping each producer's current default:
  whole-event for fast_data/FastHydro, 1 for H5BulkWriter. The chunking becomes a writer
  option, not a difference between forks. Revisit after the Phase 4 read benchmark.
- **D3 — `format` string.** Recommend a single `format = "fno4d/hydro_evolution"` with
  `format_version = 2`, plus a separate `producer` attribute: `"fast_data"`,
  `"xscape/H5BulkWriter"` or `"fasthydro/PairedH5Writer"`. The only reader of `format` today is
  `FastHydro/tests/test_h5_output.py:77`. Readers must accept both v1 strings.
- **D4 — event axis.** Recommend always `maxshape=(None, …)`, which costs nothing. Pre-allocate
  when `nevents > 0` is given (fast_data, resume); grow from 0 otherwise (H5BulkWriter). The
  invariant `attrs["nevents"] == arr.shape[0]` holds at every flush in both modes.

## Branches

- FNO4d: new branch `h5_writer_consolidation` from `main`.
- js-contrib: new branch `h5_writer_consolidation` from `fasthydro_hadronization`, which
  carries the FastHydro particlization and `PLAN_hadronization.md`. Merge order: FNO4d first,
  then re-sync, then js-contrib.
- X-SCAPE core: no changes expected. The C++ writers keep their ROOT output.

## Phase 0 — baseline, before any edits

1. Run and record green: `FNO4d: pytest tests/test_fast_data_*.py`,
   `FastHydro: pytest tests/`, `PyJetscape: pytest tests/test_h5_bulk.py`.
2. Write **golden files** into a scratch dir, one per producer. Keep them out of git.
   - fast_data: `python -m fast_data` with a 2-event tiny config (or `write_fno_h5` on
     `gubser.make_event` output).
   - FastHydro: the `written` fixture setup of `tests/test_h5_output.py`, with `_FakeHydro` and
     `_FakeBridge`, plus one real 1-event `example/run_two_stage.py` with showers on.
   - H5BulkWriter: `PyJetscape/example/python_bulk_h5_writer.py` on the smallest MUSIC config.
3. Add `FNO4d:tools/h5_schema_dump.py`. It prints every dataset (shape, dtype, chunks,
   compression, maxshape) and every attribute of a file. Dump the golden files; these dumps
   are the diff target for Phases 2–3.
4. Fill in the freeze-out table below from real files. Do not trust docstrings here: there are
   already conflicting comments. Measure `live_tau_lengths`, `ntau_freezeout` and
   `tau_freezeout − tau_min` for one event per producer.

| producer | populated frames | `ntau_freezeout` | `tau_freezeout − tau_min` |
|---|---|---|---|
| legacy RootBulkWriter → root_to_hdf5 (`dAu_25ev_mb.h5`) | | | |
| fast_data | | | |
| FastHydro pair (`arr` / `arr_bg`) | | | |
| H5BulkWriter native, stride 1 | | | |
| H5BulkWriter native, stride 2 | | | |

## Phase 1 — the merged writer (FNO4d)

File: `FNO4d:loc_libs/fast_data/writer.py`. It must import only stdlib, numpy and h5py at
module level, because PyJetscape vendors it on its own. `eos.write_eos_group` stays a lazy
import inside the `np_eos` branch; `repad_to` moves here.

### 1a. `FnoH5Writer`: union of both APIs, both call patterns keep working

```python
FnoH5Writer(path, attrs, nevents=0, *,
            compression="lzf", chunk_events=1, chunk_tau=None,   # None = whole event (D2)
            growable_tau=True,
            write_source=False, source_compression="gzip",
            write_diagnostics=True, np_eos=None,
            producer="fast_data", extra_attrs=None,
            force=False, resume=False)

.arr, .nevents, .choose_ntau, .growable_tau          # properties (PyJetscape uses them)
.ensure_capacity(nevents=None, choose_ntau=None)     # from PyJetscape
.write_frame(i, t, frame)                            # from PyJetscape
.set_event_meta(i, ntau_fo, tau_fo)                  # from PyJetscape; bumps nevents_written, flushes
.append_event(i, arr_ev, ntau_fo, tau_fo, *, S_ev=None, P_cart=None,
              droplets=None, diag=None)              # fast_data signature, unchanged
.add_evolution(name, *, fo_suffix)                   # NEW: a second arr-shaped dataset (arr_bg)
.ragged(group, offsets_name, fields, *, unit="event", compression="gzip") -> RaggedGroup   # NEW
.close(complete=None)
```

Rules:
- `append_event` grows capacity itself (as the PyJetscape fork does), so a writer opened with
  `nevents=0` accepts `append_event` too.
- `append_event` with `arr_ev.shape[-1] > choose_ntau` and `growable_tau=False` raises. Clipping
  is the caller's decision; H5BulkWriter already clips and warns.
- `set_event_meta` is the only place that advances `nevents_written` and calls `flush()`.
  `append_event` ends by calling it.
- `nevents` and `choose_ntau` attributes are rewritten from the dataset shape at every
  `ensure_capacity` and at `close`.
- Root attributes set at creation: every `SCALAR_KEYS` entry and `choose_ntau`;
  `format="fno4d/hydro_evolution"`, `format_version=2` (D3), `producer`; `feature_names`,
  `units`, `velocity_convention` (one text; merge the two current wordings);
  `freezeout_convention`, `freezeout_convention_id` (D1); `nevents_written`, `complete`,
  `has_source`.
- Every dataset with a τ axis gets an attribute `tau_axis = <int>`: `arr` (5), `arr_bg` (5),
  `source/S` (5), `source/P_cart` (1). This is how `repad_to` finds them (1d).
- `resume=True`: reopen `r+`; check `format_version` and `nevents` as today; reattach
  `arr`, the `add_evolution` datasets and every `RaggedGroup` (1b). Resuming a v1 file is
  refused with a message that says so.

### 1b. `RaggedGroup`: the shared variable-length table

This replaces the hand-written droplet, shower and (future) hadron code. It writes
**incrementally**: every field and the offsets are `maxshape=(None, …)`, resized and written on
each `append`, so a crash loses nothing that `nevents_written` counts.

```python
g = w.ragged("source", "offsets",
             {"droplets": (np.float64, (8,))},
             columns={"droplets": DROPLET_COLUMNS})
g.append(rows={"droplets": d})          # one unit (= one event); rows may be empty
```

- Layout (identical to today's): `<group>/<field>` of shape `(N, *tail)`, and
  `<group>/<offsets_name>` int64 of length `units_written + 1`. Unit `k` is rows
  `offsets[k]:offsets[k+1]`. An empty unit is a repeated offset: empty, not missing.
- `columns=` is stored as `<field>_columns` on the group. This matches today's
  `droplet_columns` and FastHydro's `parton_columns`, `vertex_columns` and
  `initiator_columns`, so no existing reader changes.
- `unit="event"`: `append` must be called exactly once per event, in order. It is checked
  against `nevents_written` at `set_event_meta` time and raises on a skip or a double append.
  `unit="free"`: no such check. Hadron oversamples (Phase 5) use this, with a second
  offsets vector that the owner maintains.
- Several fields share one offsets vector. The hadron table needs `pid` int32 next to `p`
  float32 (N,4): a float column cannot hold PDG codes above 2²⁴ exactly.
- `resume`: truncate every field to `offsets[nevents_written]` and the offsets to
  `nevents_written + 1`, then continue. This fixes today's behaviour, where resume reads all
  droplets back into memory.
- Chunk the rows axis at ~64k rows (tunable), with gzip level 4 + shuffle by default. That is
  today's `source_compression` default.

Port `source/droplets` onto `RaggedGroup` inside `FnoH5Writer`: `write_source=True` creates it.
`diag/` stays as it is: per-event scalars with a varying key set, written at close. It is small
and reconstructable, and it is out of scope here.

### 1c. `add_evolution(name, fo_suffix)`

Creates `name` with the same shape, chunking, maxshape and compression as `arr`, plus
`ntau_freezeout{fo_suffix}` and `tau_freezeout{fo_suffix}`. It is grown by `ensure_capacity`
together with `arr`. `write_frame`, `append_event` and `set_event_meta` take `dataset=name`
(default `"arr"`). This replaces `PairedH5Writer._ensure_bg` (`FastHydro/…/h5_writer.py:120`),
which hand-rolls the same thing and skips `ensure_capacity`.

### 1d. `repad_to` moves upstream and discovers τ datasets

Move it from `PyJetscape/…/fno_h5_writer.py:277-385` unchanged, with one exception: replace
`_TAU_DATASETS` with a scan for datasets carrying a `tau_axis` attribute. For v1 files (no
attributes), fall back to the old list **plus** `arr_bg`. This fixes the `arr_bg` bug from the
Context section.

### 1e. Tests (FNO4d `tests/test_fast_data_writer.py`)

- Keep w1, w2, w5, w6, w7 and w8 unchanged; they must pass as they are.
- Update w3 and w4 to the D1 relation.
- Port the writer tests from `PyJetscape/tests/test_h5_bulk.py` (the ones that don't need the
  compiled extension):
  - `test_writer_matches_the_fno4d_contract` with `chunk_tau=1`;
  - `test_padding_costs_no_disk_space`;
  - `test_pinned_choose_ntau_does_not_grow`;
  - `test_pinning_choose_ntau_generously_is_free`;
  - `test_tau_axis_can_be_repadded_in_place`;
  - all `test_repad_to_*`.
- New tests:
  - `append_event` and `write_frame` produce byte-identical `arr` for the same event;
  - growable (`nevents=0`) and pre-allocated files read identically through `MultiH5Array`;
  - `RaggedGroup` round trip with empty units, multi-field shared offsets, and an int32 field;
  - a crash mid-run (close without `complete`, or `del` without close) leaves ragged data
    consistent with `nevents_written`;
  - resume truncates and continues ragged data correctly;
  - `unit="event"` refuses a skipped or doubled append;
  - `add_evolution` datasets grow with `arr` and are repadded with it;
  - `repad_to` on a v1-style pair file grows `arr_bg`;
  - v1 files (both format strings) still load through `read_3d_data_hdf5`.
- `driver.py`: no call-site change beyond passing `producer="fast_data"`. Run the tiny config
  end to end and diff against the Phase 0 dump. Expected differences: `format`,
  `format_version`, `producer`, `freezeout_convention*`, `ntau_freezeout` −1 (D1), `tau_axis`
  attributes, and maxshape on the event and ragged axes. Nothing else.

Commit on FNO4d; note the commit sha for Phase 2.

## Phase 2 — FastHydro: re-sync and slim `PairedH5Writer`

1. `tools/sync_fast_data.sh ~/FNO4d`; review the `git diff --stat`;
   `pytest tests/test_vendor_intact.py tests/test_fast_data_*.py`.
2. `python/fasthydro/h5_writer.py`:
   - pass `producer="fasthydro/PairedH5Writer"`;
   - replace `_ensure_bg` and the manual `arr_bg` writes with
     `self._bg = w.add_evolution("arr_bg", fo_suffix="_bg")` and
     `w.append_event(i, hyd_bg.arr, …, dataset="arr_bg")`;
   - replace `_stash_shower` and `_write_shower` with three `RaggedGroup`s in `shower/`
     (`partons`/`parton_offsets`, `vertices`/`vertex_offsets`,
     `initiators`/`initiator_offsets`, `unit="event"`), appended in `append()`. Keep every
     `shower/` attribute text (`units`, `coordinates`, `vertex_positions`,
     `endpoint_convention`, `pstat_codes`) and `has_shower`. Write them when the group is
     created, not at close;
   - an event with no shower appends empty rows, as today;
   - keep the IC-sha check and the liquefier attributes unchanged.
3. Optional, same phase: add `resume=` to `PairedH5Writer` and `example/run_two_stage.py`. It
   now comes almost free, because every group is resumable.
4. `droplets_io.py` (the replay npz) stays npz: it deliberately needs neither h5py nor X-SCAPE.
   Only its `_flatten_showers` column order must stay equal to `showers.py`; that is already
   true, so leave it alone.
5. Tests:
   - `tests/test_h5_output.py:77`: accept `format == "fno4d/hydro_evolution"`
     and `producer == "fasthydro/PairedH5Writer"`;
   - `test_freezeout_bookkeeping_for_both_legs`: update to D1;
   - new: `repad_to` on a pair file keeps `arr.shape == arr_bg.shape`, and `PairBrowser` still
     opens it;
   - new: kill mid-run (close with `complete=False` after event 0 of 2), then `PairBrowser`
     reads event 0 including its shower;
   - `tests/test_showers.py:240-290` (writer round trip, per-event slices, empty event, no
     showers) must pass unchanged. They are the contract for the `shower/` layout.
6. Diff the Phase 0 dumps. Expected differences are the Phase 1 list, plus `shower/*`
   maxshape and chunking.

## Phase 3 — PyJetscape: vendor instead of fork

1. Replace `python/jetscape/fno_h5_writer.py` with a verbatim copy of
   `FNO4d:loc_libs/fast_data/writer.py`. Keep the module name, because `repad_h5.py`,
   `fast_h5_bulk.py` and `tests/test_h5_bulk.py` import it. The relative `from .eos import …`
   is only reached with `np_eos`, which H5BulkWriter never passes. Add a test for exactly that,
   so a future upstream edit that makes the import eager fails here and not at runtime.
2. Add, mirroring FastHydro:
   - `python/jetscape/UPSTREAM_fno_h5_writer.json`: commit, sha256, bytes;
   - `tools/sync_fno_h5_writer.sh [path-to-FNO4d]`: a copy of `sync_fast_data.sh` reduced to
     one file;
   - `tests/test_writer_vendor_intact.py`;
   - a `VENDORING.md` section in `README.md`, with the same "no local patches" rule.
3. `python/jetscape/fast_h5_bulk.py`:
   - `_open`: `FnoH5Writer(…, nevents=0, chunk_tau=1, growable_tau=…, producer="xscape/H5BulkWriter", extra_attrs=…)`.
     The `format` string is no longer set here;
   - `Exec`: `set_event_meta(i, n_write, tau_fo)`. `n_write` is already "frames written",
     which is D1. `tau_freezeout` stays `src.tau_min + src.ntau*src.dtau`. Add a comment that
     with `tau_stride > 1` this is the unthinned end time, matching
     `FastRootBulkWriter.cc:133`;
   - no other change: frame streaming, clipping and `clear_after_write` stay as they are.
4. `repad_h5.py`: unchanged. It imports `repad_to` from the vendored module.
5. Tests (`tests/test_h5_bulk.py`):
   - delete the writer-internal tests now ported upstream (Phase 1e); keep the
     resample/grid/framework/bulk-source tests;
   - keep `test_fno4d_loaders_accept_the_file` and `test_h5_tooling_imports_without_the_compiled_extension`;
   - the `arr.chunks[-1] == 1` check stays, but as an H5BulkWriter test, not a writer test;
   - new: an H5BulkWriter file and a fast_data file of the same grid concatenate in
     `MultiH5Array` once repadded to one `choose_ntau`, which was the goal all along.
6. Diff the Phase 0 H5BulkWriter dump. Expected differences: `format`/`producer`/version,
   `freezeout_convention_id`, `tau_axis` attributes. `arr`, `ntau_freezeout` and
   `tau_freezeout` must be bit-identical.

## Phase 4 — cross-checks

- **One reader for all producers.** Point `fast_data.viz.EventBrowser` at each of the three
  golden v2 files, and `fasthydro.browse.PairBrowser` at the pair file. `read_3d_data_hdf5`,
  `MultiH5Array` and `live_tau_lengths` agree with `ntau_freezeout` on all of them (D1).
- **Mixed-version campaign.** A v1 fast_data file and a v2 file of the same grid, repadded
  together, load in `MultiH5Array`. Document in `README_FastData.md` that `ntau_freezeout`
  differs by the convention and that `freezeout_convention_id` is the way to tell.
  Optionally, `downsample.py:704` reads the id and corrects v1 files; that is a separate FNO4d
  commit.
- **Chunking benchmark (D2).** On one real FastHydro event and one MUSIC native event, time
  `MultiH5Array` random τ-window reads (the training access pattern) and file size for
  `chunk_tau ∈ {1, 4, whole}` × `{lzf, gzip-4}`. Record the table here and set the defaults.
- **FV-vs-MUSIC leg** (`X-SCAPE:config/FVvsMUSIC/run_music_leg.py`): it writes `.npz` today,
  so it is not affected. Note it as a candidate to switch to H5BulkWriter later; not in scope.

## Phase 5 — hand-off to the hadron work

This phase is here so that `RaggedGroup` gets designed for hadrons from the start. It is
implemented in the hadron plan, not this one.

- Binding: `SoftParticlization.get_hadrons_numpy()` in `PyJetscape/src/`, next to
  `bind_jet.cc`. It reads the public `Hadron_list_` (`X-SCAPE:src/framework/SoftParticlization.h:144`)
  and returns `pid` int32 (N,), `p` float32 (N,4) `[E,px,py,pz]`, `x` float32 (N,4)
  `[t,x,y,z]`, `mass` float32 (N,) and `sample_counts` int64 (S,).
  - Generic over the SoftParticlization module; no iSS headers needed.
  - Column order follows `FastHydro/python/fasthydro/hadrons.py` (`p = [E, px, py, pz]`).
  - Call it from a module placed after iSS, which runs before `Clear()` empties the list
    (`X-SCAPE:src/hadronization/iSpectraSamplerWrapper.cc:263-271`).
- File layout: `hadrons/` group in the same file as `arr`.
  - One `RaggedGroup("hadrons", "sample_offsets", {pid, p, x, mass}, unit="free")` holds one
    unit per oversample.
  - `hadrons/event_offsets` is a growable int64 vector: event `i` is samples
    `event_offsets[i]:event_offsets[i+1]`.
  - Attributes: `n_oversample`, `T_sw`, `hydro_id`, `units`, `p_columns`, `x_columns`.
  - The second level of offsets is new information: `hadrons.py` currently cannot tell
    oversamples apart, because the ASCII file does not separate them.
- Reader: `Hadrons.from_h5(path, event=None)` in `hadrons.py`. Keep `read_ascii` and
  `ascii_to_npz` for old runs.
- Drop `JetScapeWriterFinalStateHadronsAscii` from `example/run_particlize.py` by default and
  keep it behind a flag.

## Acceptance

- One writer source (`FNO4d:loc_libs/fast_data/writer.py`) and two verbatim copies, each
  enforced by a checksum test. No hand-modified fork remains.
- Every file written by fast_data, FastHydro and H5BulkWriter:
  - is `format="fno4d/hydro_evolution"`, `format_version=2`, with a `producer`;
  - satisfies `live_tau_lengths == ntau_freezeout`;
  - survives `repad_to` with every τ-bearing dataset grown together.
- All variable-length groups (`source/droplets`, `shower/*`) are written per event and
  survive a crash up to `nevents_written`.
- The Phase 0 dumps differ from the Phase 1–3 dumps only in the differences listed per phase.
- All three test suites are green; FNO4d w3/w4 are updated to D1 with a note in the test.

## Risks

- **Copies drift again.** The checksum tests catch local edits, not a stale re-sync. Add the
  upstream commit to both manifests and have a one-line CI or pre-commit check that compares
  the two manifests' `writer.py` sha when both repos are present (skipped otherwise).
- **v1 files already in training campaigns.** The D1 change is only safe because the readers
  that matter derive lifetimes from the data. Grep FNO4d once more for `ntau_freezeout`
  consumers before merging; the Phase 4 note covers `downsample.py`.
- **h5py resize cost.** Growing ragged datasets per event is cheap with chunked storage, but
  check it on a many-small-events fast_data run (1000 events): per-event `flush()` already
  exists, so this is expected to be noise. Measure; don't assume.
- **`resume` + growable axes.** A resumed file opened with a different `nevents` must still be
  refused. Keep the existing check even though the axis could now grow.
