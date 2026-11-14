# Write the files

Store the hadrons of your own runs in the production's file format, and the writers behind the production files. Which file holds what is on [What a production writes](https://jhputschke.github.io/js-contrib/analysis/files/index.md).

## Examples

### Hadrons of your own run

`HadronH5Writer` writes one unit per event; the files it makes are read with everything on [Analyse the files](https://jhputschke.github.io/js-contrib/python/api/analyse/index.md). Give it an absolute path: `JobDir` changes the directory.

```python
import os
import jetscape as js
from jetscape.hadrons_h5 import HadronH5Writer
from jetscape.records import initiator_rows

out = os.path.abspath("my_hadrons_jet_frag.h5")     # absolute: JobDir changes directory
with js.JobDir("out", "oo_workflow.xml") as job, \
     js.per_event(job.main_xml, job.user_xml, n_events=2) as driver, \
     HadronH5Writer(out, tag="jet_frag", n_samples=1, initiators=True) as w:
    had, mgr = driver.hadronization_manager, driver.energy_loss_manager
    n = 0
    for i in driver.events():
        w.append_unit(had.hadron_array().to_dict(), unit=n, event=i,
                      initiators=initiator_rows(mgr.initiator_array()),
                      seed=had.last_random_seed)
        n += 1
```

- **The bulk hadrons** of iSS go into a second writer with `tag="bulk_bg"` and `n_samples=` the number of oversamples: call `driver.soft_particlization.use_compact_output()` before the loop and append `driver.soft_particlization.hadron_array().to_dict()` per event. The complete version is `pythonic/oo_workflow.py` in the examples.
- **Extra keywords** of `append_unit` (`seed=`, `pthat=`, `n_cells=`, ...) are stored per unit, in the file's `units/` table.

### What the productions write

The drivers use the other writers on this page themselves:

| file                        | written by                              | class                |
| --------------------------- | --------------------------------------- | -------------------- |
| `<stem>_particlize.h5`      | `run_prod_jet.py --write-particlize`    | `ParticlizeH5Writer` |
| `<stem>_hadrons_<tag>.h5`   | `hadronize.py` (per unit: `Hadronizer`) | `HadronH5Writer`     |
| `<stem>.h5`, the hydro pair | `run_prod_jet.py`                       | `PairH5Writer`       |

`SurfaceReplay` is a hydro module that hands a stored freeze-out surface to iSS inside a JETSCAPE run, so iSS samples it as it would the live hydro's. `liquefier_io` reads the jet's droplets out of the liquefier during a run.

## Reference

### `jetscape.hadrons_h5.HadronH5Writer(path, *, tag, n_samples, attrs=None, compression=DEFAULT_COMPRESSION, force=True, keep_bits=None, initiators=False, eta_max=None, charged_only=False, positions=True, hadronization_input=False, partons=False, x_had='none', hadron_scatter=False)`

Append units of samples of hadrons (see the module docstring).

`keep_bits` (default None: lossless) rounds `p` and `x` before they are written: an int for both, or a mapping such as `{"p": 12, "x": 8}`. The precision is recorded on each dataset (`hadron_precision`).

`initiators=True` (bulk_jet and jet_frag only) adds the `initiators/` group: every `append_unit` then takes that event's shower initiators, a (K, 11) array in `INITIATOR_COLUMNS` order (None: none).

`eta_max` (default None: all hadrons) keeps only hadrons with |eta| < eta_max (`pseudorapidity` of the stored, i.e. rounded, momenta); `charged_only` only the charged ones (|pid| in CHARGED); the samples stay, with fewer hadrons. `positions=False` writes no `x` (`x` given to `append_unit` is ignored). Recorded as the `eta_max`, `charged_only` and `positions` attributes (`hadron_selection`).

`hadronization_input=True` (jet_frag only) adds the `hadronization_input/` group: every `append_unit` then takes the event's Pythia input, a dict of pid, col, acol, p (K, 4) [E, px, py, pz] and mass (core.hadronization_input_numpy) or a (K, 8) array in `HADRONIZATION_INPUT_COLUMNS` order.

`partons=True` (jet_frag only) adds the `partons/` group: every `append_unit` then takes the event's final partons, a (K, 14) array in `PARTON_COLUMNS` order, so the jets can be hadronized again from this file alone (run_prod_jet.py --hadrons-only). `x_had` is recorded on the group: "none" for the partons' production points only.

`extra_frag` files (the extra hard scatterings) also take `scatterings` (the event's (S, 8) SCATTER_COLUMNS rows, group `scatterings/`) and `initiator_scatter` (each initiator's scattering, `initiators/scatter`) in `append_unit`, and with `hadron_scatter=True` each hadron's scattering (a `scatter` array in every sample, `hadrons/scatter`).

#### `append_unit(samples, initiators=None, hadronization_input=None, partons=None, scatterings=None, initiator_scatter=None, **unit_scalars)`

Append one unit. `samples` is a list of hadron dicts (one per sample, keys pid, pstat, p, x), or one dict with `sample_counts` (as soft_hadrons_numpy returns); `initiators` the event's (K, 11) initiator rows, for a writer made with `initiators=True`; `hadronization_input` the event's Pythia input, for a writer made with `hadronization_input=True`; `partons` the event's final partons, for a writer made with `partons=True`. Returns the unit index.

### `jetscape.hadrons_h5.add_initiators(path, data, offsets, *, force=False)`

Add `initiators/` to an existing bulk_jet or jet_frag file: unit u gets the rows of event `units/event[u]` from `(data, offsets)` (as `pair_initiators` returns). An existing group is kept unless `force`. Returns False if it was kept.

### `jetscape.hadrons_h5.particlize_initiators(path, nevents=None)`

`(data, offsets)` of a particlize file's `initiators/` (format version 2, or added with `add_initiators.py`), or None when the file or the group is missing. With `nevents` the file must hold that many events.

### `jetscape.hadrons_h5.pair_initiators(path, nevents=None)`

`(data, offsets)` of a pair file's `shower/initiators`, or None when the file or the group is missing. With `nevents` the pair file must hold that many events.

### `jetscape.hadrons_h5.z_record(leptons) -> dict`

The `Z_COLUMNS` of one event from its trigger leptons, a (2, 5) array pid, px, py, pz, E (`pyjetscape_core.pythia_gun_trigger`); NaN (pids 0) without a pair.

### `jetscape.hadrons_h5.hadronization_input_rows(d)`

A Pythia input (dict of pid, col, acol, p, mass, or a (K, 8) array) -> (K, 8) float64 rows in `HADRONIZATION_INPUT_COLUMNS` order (exact: ids and colour tags are far below 2\*\*53).

### `jetscape.hadrons_h5.hadronization_input_dict(rows)`

(K, 8) `hadronization_input/data` rows -> the dict core.hadronize_input takes.

### `jetscape.particlize_h5.ParticlizeH5Writer(out_file_name='particlize.h5', *, legs=('jet', 'bg'), store_partons=True, bg_id='MUSIC_1', jet_id='MUSIC_2', compression=DEFAULT_COMPRESSION, music_input=None, T_fo=None, pair_file=None, extra_attrs=None, force=True, verbose=False, sink=None, extra_hard=False)`

Write each event's surfaces and final partons (see the module docstring).

Parameters:

- **`out_file_name`** (`str`, default: `'particlize.h5'` ) – Output path.
- **`legs`** (`iterable of {"jet", "bg"}`, default: `('jet', 'bg')` ) – Which legs' surfaces to store. Their MUSIC instances must build the surface (\<freeze_out_surface>1) and hand it over (skip_surface off, e.g. PairH5Writer(keep_surface=legs)).
- **`store_partons`** (`bool`, default: `True` ) – Write partons/ and initiators/ (both from the energy-loss manager).
- **`bg_id`** (`str`, default: `'MUSIC_1'` ) – Module ids of the two MUSIC instances.
- **`jet_id`** (`str`, default: `'MUSIC_1'` ) – Module ids of the two MUSIC instances.
- **`compression`** (`str or None`, default: `DEFAULT` ) – jetscape.h5_compression spec for the surfaces (default Blosc-zstd: the charge and chemical-potential columns are zero here and cost almost nothing). Partons are gzip'ed (small).
- **`music_input`** (`str`, default: `None` ) – Text of the job's music_input (iSS reads its EoS id and flags from it). By default ./music_input is read at attach (the job's working directory).
- **`T_fo`** (`float`, default: `None` ) – MUSIC's freeze-out temperature, recorded.
- **`pair_file`** (`str`, default: `None` ) – The FNO4d pair file written alongside, recorded (basename).
- **`extra_attrs`** (`mapping`, default: `None` ) – More root attributes (run provenance), written verbatim.
- **`force`** (`bool`, default: `True` ) – Overwrite an existing file.
- **`sink`** (`callable`, default: `None` ) – No file: each event's record goes to sink(record) (see the module docstring).
- **`extra_hard`** (`bool`, default: `False` ) – The events have other hard scatterings (PythiaGun's ): store scatterings/ and initiators/scatter (Exec's scatterings and shower_scatter).

#### `attach(jetscape=None, *, bg=None, jet=None, manager=None)`

Find the MUSIC legs and the energy-loss manager; open the file.

Call after `JetScape.Init()` (and after `PairH5Writer.attach`, which sets the legs' `skip_surface` flags). A stored leg found with `skip_surface` on is switched off here, with a warning.

#### `Exec(i=None, *, bg_id=None, bg_key=None, scatterings=None, shower_scatter=None, **diag)`

Store this event. Returns its index.

`i` is the event's index in the pair file (default: the running count; the two must agree -- events skipped by the pair writer must be skipped here too). `bg_id` names the background: the first event that used it (the pair writer's `diag/bg_id`). A new value stores the background surface; a repeated one (an event reusing its background) only records the reference. Without a pair writer pass `bg_id=i`. Extra keywords go to `events/`. With `extra_hard`: `scatterings` the event's (S, 8) rows (SCATTER_COLUMNS) and `shower_scatter` the hard scattering of each shower (indexed by the shower index; extra_hard_info's parton_scatter).

#### `write_events(i, **scalars)`

Add per-event scalars after `Exec` (e.g. seeds, wall time).

#### `Finish(complete=None)`

Close the file. Idempotent.

### `jetscape.particlize_h5.add_initiators_from_pair(path, pair=None, *, force=False)`

Copy the pair file's `shower/initiators` into the particlize file `path`, so it no longer needs the pair file. `pair` defaults to the file's `pair_file`, next to it.

Returns True if added, False if the file already had `initiators/` (kept unless `force`). Raises if the pair file is missing, holds no initiators, or has another number of events.

### `jetscape.hadronizer`

python/jetscape/hadronizer.py

Hadronizer -- the per-unit work of hadronizing a two-stage hydro event: iSS on a leg's freeze-out cells, ColorlessHadronization on the event's final partons, their seeds, and the hadron files (jetscape.hadrons_h5). One code path for the two places it runs:

```text
example/prod_AuAu_0_10_jet/hadronize.py      afterwards, from a particlize file
example/prod_AuAu_0_10_jet/run_prod_jet.py --hadrons-only
                                             in the production job, from the particlize
                                             writer's memory sink
```

so the job's hadrons are, by construction, what hadronize.py makes of the same cells, partons and seeds (docs/Plans/PLAN_AuAu_hadrons_only.md).

```text
h = Hadronizer(core, iss=iss, colorless=colorless, base_seed=1, prod=production_key)
h.open("bulk_jet", path, n_samples=1, attrs=common, primary_path=ppath)
h.bulk("bulk_jet", e, cells, h.bulk_jet_seed(e, bg_unit), n_samples=1, event=e,
       initiators=ini)
h.fragments(e, partons, [h.seed("jet_frag", e, s) for s in range(n_frag)],
            initiators=ini)
h.extra_fragments(e, extra_partons, n_frag, initiators=extra_ini,
                  initiator_scatter=scat, scatterings=rows)   # extra hard scatterings
h.close(complete=True)
```

With extra hard scatterings (PythiaGun's , docs/Plans/PLAN_multi_hard.md) an event's final partons are split by their shower's scattering (`jetscape.particlize_h5.split_by_scatter`): the trigger's go to `jet_frag` exactly as without extras (same partons, same seeds), the extras' to `extra_frag`, with seeds of their own tag: all together in one Colorless call (mode `groups`) or one call per scattering (`per_scattering`, each hadron's scattering kept).

iSS runs on given cells through `core.soft_sample_cells` (X-SCAPE's `SoftParticlization::SampleCells`): no framework event, so it also works inside a production process whose own event loop must not run again. The iSS and Colorless modules only need `Init()` (after the JETSCAPE instance's `Init` has read the XML).

#### `Hadronizer(core, *, iss=None, colorless=None, base_seed=1, prod=None, common_seeds=False)`

iSS on cells, Colorless on partons, into hadron files (see the module docstring).

Parameters:

- **`core`** (`module`) – jetscape.pyjetscape_core.
- **`iss`** (`module objects or None`, default: `None` ) – Initialised iSS and ColorlessHadronization modules (core.create_module(...), .Init()), for the bulk and the jet tags.
- **`colorless`** (`module objects or None`, default: `None` ) – Initialised iSS and ColorlessHadronization modules (core.create_module(...), .Init()), for the bulk and the jet tags.
- **`base_seed`** (`int`, default: `1` ) – The base of every unit's seed (unit_seed).
- **`prod`** (`int or None`, default: `None` ) – The production file's seed key (uuid_key of its file_uuid); None: the legacy seeds.
- **`common_seeds`** (`bool`, default: `False` ) – bulk_jet of an event gets its background's bulk_bg seed (bulk_jet_seed; hadronize.py --common-seeds / --correlated).

##### `bulk_jet_seed(event, bg_unit)`

`bulk_jet`'s seed: its own, or with `common_seeds` its background's.

##### `open(tag, path, *, n_samples, attrs, primary_tag=None, primary_path=None, keep_primaries=0, **writer_kw)`

A writer for `tag` (and, with `primary_path`, for its undecayed hadrons, `primary_tag`: full precision, positions, no selection, the first `keep_primaries` samples; `KEEP_ALL_PRIMARIES` or more: all). `writer_kw` go to HadronH5Writer (keep_bits, eta_max, charged_only, positions, initiators, hadronization_input, partons).

##### `set_attrs(attrs)`

Add root attributes to every open file (e.g. what is known at the end of a job).

##### `close(complete=True)`

Close every writer (`complete` False leaves readable, incomplete files).

##### `sample(cells, seed, n_samples, *, keep_primaries=0)`

iSS on `cells` -> (hadrons, seed used, primaries or None); (None, seed, None) for no cells.

##### `bulk(tag, unit, cells, seed, n_samples, *, event, initiators=None, keep_primaries=0, **scalars)`

iSS on one surface into `tag`'s file (and its primary file) -> (hadrons or None, seed used).

##### `fragments(unit, partons, seeds, *, event=None, initiators=None, tag='jet_frag', **scalars)`

Colorless on `partons` once per seed into `tag` (default `jet_frag`; and the partons into its `partons/`, if the writer has one) -> the fragmentations.

##### `extra_seeds(unit, n_frag, scatters=None)`

The extras' Colorless seeds of one event: per fragmentation one (`scatters` None, mode groups), or per fragmentation a list, one per scattering in `scatters` (per_scattering).

##### `extra_fragments(unit, partons, n_frag, *, initiators, initiator_scatter, scatterings=None, mode='groups', event=None, **scalars)`

Colorless on the extras' final partons into `extra_frag`, `n_frag` times: all together (`mode` groups) or one call per scattering (per_scattering; every hadron then carries its `scatter`). `partons`/`initiators` are the extras' only (`jetscape.particlize_h5.split_by_scatter`), `initiator_scatter` their scatterings, `scatterings` the event's SCATTER_COLUMNS rows -> the fragmentations.

#### `unit_seed(base, tag, unit, sample=0, prod=None)`

Independent, reproducible 31-bit seed per (base, production file, tag, unit, sample).

`prod` is the production file's key (`uuid_key` of its `file_uuid`); None gives the legacy seeds, the same for every production file.

#### `uuid_key(text)`

A `file_uuid` as the integer seed key (ValueError if it is not a UUID).

#### `samples_per_background(oversample, spec, uses, cap)`

iSS samples for one background -> (n, capped).

`spec` None: `oversample`; a number: that; 'auto' / 'per-pthat-bin': `oversample` x `uses`, the number of events using the background (in the file, or in one pTHat window), at most `cap`.

#### `iss_commit(xscape)`

The iSS checkout an X-SCAPE tree uses (best effort; '' if nothing can tell): git in the tree's external_packages/iSS, else the `external_packages/iSS <sha> ...` line of the tree's BUILD_INFO.txt (written by the production image and the xscape wheel).

#### `iss_decay_attrs(core, eos, xml_root, xscape=None)`

Hadron-file attributes saying which hadron list iSS sampled and how it decayed it.

#### `collision_ecm(job_xml)`

The collision energy sqrt(s_NN) [GeV] of a production job's XML (its text, as the files' `prod_user_xml`, or a parsed root): , else ; None without either (or without the XML).

#### `set_hadronization_ecm(root, ecm)`

Colorless's in a parsed hadronize.xml (`root`) set to `ecm`, the collision's energy, in place: a beam remnant that closes a string carries 1/6 of it, so one hadronize.xml serves every collision system. -> the value the file had (None if it had none). ValueError without .

#### `configure_hadronize_xml(root, *, xscape, seed, n_events, oversample, keep_primaries, correlated=False, correlated_block=None)`

A parsed hadronize.xml (`root`) with a run's settings, in place: the event count, the seed, iSS's paths (absolute, into `xscape`'s external_packages/iSS, or into `xscape` itself for an installed tree, which keeps iSS_parameters.dat and iSS_tables in its data folder; its working directory ".", where the run's music_input is), the oversample count, and the correlated sampling (`correlated_block` "DTAU,DX,DETA"; ValueError if malformed).

#### `tag_attrs(tag, core, decay_attrs, *, common_seeds=False, correlated=False)`

The tag-specific attributes of a hadron file (the generator, decayed, and for the bulk tags the hadron list, its decays and how the two legs were sampled relative to each other; for jet_frag the Pythia version).

### `jetscape.pair_h5`

python/jetscape/pair_h5.py

PairH5Writer -- write an X-SCAPE two-stage MUSIC run as one FNO4d-schema HDF5 file in FastHydro's pair layout: the jet leg (with the CausalLiquefier deposition) as `arr`, the background leg as `arr_bg`, the droplets MUSIC was given as `source/droplets`, and the parton showers as `shower/`.

```text
IS -> Hard -> NullPreDynamics -> MUSIC_1 (background)
   -> JetEnergyLossManager (Matter + LBT, CausalLiquefier)
   -> MUSIC_2 (same IC + the liquefier source)          -> PairH5Writer
```

Where each leg is read from

- **Background (MUSIC_1)** from the framework copy, `bulk_info`. Matter and LBT query the first hydro through `bulk_info`, so it must be filled, and filling it releases MUSIC's native store (`MusicWrapper.cc` `PassHydroEvolutionHistoryToFramework`). The copy is cell for cell the native store: both go through `get_fluid_cell_with_index` and a float copy, so nothing is lost.
- **Jet leg (MUSIC_2)** from its native store, which `attach()` keeps with `set_dump_hydro_only(True)` on that instance only. Every MUSIC instance reads the first `<Hydro><MUSIC>` XML block, so this cannot be set per leg in the XML.

Both legs then go through the same `jetscape.bulk_sources.resample` onto one output grid, so `arr - arr_bg` is the jet's effect and nothing else: before the first droplet deposits the two legs are bit-identical, which `diag/frames_identical` records per event.

Layout (on top of `jetscape.fno_h5_writer.FnoH5Writer`)

- `arr` / `arr_bg` with `ntau_freezeout[_bg]`, `tau_freezeout[_bg]`. The tau axis grows to the longer leg (the jet leg usually outlives the background); the shorter one is exactly zero after its own freeze-out. `arr_bg.shape == arr.shape` always.
- With a reused background (`bg_layout="shared"`, the default for `reuse > 1`) each background is stored once: in `arr_bg_store`, in the row of the first event that used it (the other rows are never written and take no space), with `arr_bg_rows` giving every event's row. `arr_bg` is then a virtual dataset over the store (a row view, `jetscape.fno_h5_writer.write_row_view`): it reads exactly like the full layout, event by event, and root attribute `bg_layout` is `"shared"`. Without reuse the file is written as before (no `bg_layout` attribute).
- A reused background is recognised by a fingerprint of its raw frames and then neither read nor resampled again: its frame hashes and edge maxima are those of the event that first used it (`verify_bg_reuse` checks this). Only `bg_layout="full"` with a file still reads it every event, to store it in every row.
- `source/droplets` `(M, 8)` + `source/offsets`, columns `droplet_columns`. There is no `source/S`: MUSIC evaluates the liquefier kernel per cell and step and never keeps a gridded source, and a re-deposit on the output grid would not be what MUSIC applied. `has_source` is therefore false. With an X-SCAPE whose droplets carry the background fluid where they were made, also `source/fluid` `(M, 4)`, columns `fluid_columns` (vx, vy, vz, e: lab velocity, energy density; e < 0: not looked up), which `<CausalLiquefier><frame>fluid</frame>` uses; older files have no `source/fluid`.
- `shower/` exactly as FastHydro writes it (`jetscape.showers`).
- `diag/`: droplet counts and energies (and how much deposits after the jet leg froze out, which MUSIC never applies), shower counts, MUSIC tau0, per-leg frame counts, `bg_id` (the first event that used this background -- repeats under `setReuseHydro`), `frames_identical`, and per leg the largest energy density on the faces of the output grid (`{bg,jet}_edge_e_max` on the x and y faces, `..._eta` on the eta faces) with, given `edge_e_threshold`, the flag `{bg,jet}_hit_edge`.
- Provenance attributes as in FastHydro's `PairedH5Writer` (`pairing`, `arr_is`, `arr_bg_is`, `source_model`, `liquefier_*` ...), so `fasthydro.browse.PairBrowser`, the wake notebook and `wake_pyvista.py` read the file unchanged.

With `store_evolution=False` no file is written at all: both legs are still read, resampled and hashed exactly as above (a reused background once), so `last_event_diag`, `last_bg_key` and the event count are what a full run gives, for a `jetscape.particlize_h5.ParticlizeH5Writer` that is then the job's only output (`run_prod_jet.py --particlize-only`). `source/` and `shower/` are not kept either. `read_jet=False` on top skips the jet leg's evolution altogether (nothing downstream of a particlize file uses it): its frame count and times then come from MUSIC's grid metadata (`ntau_jet_music`, MUSIC's own steps, instead of `ntau_jet` on the output grid), and `frames_identical` with the two checks built on it (same initial condition, droplets applied) is gone, as are the jet leg's `jet_edge_e_max`, `jet_edge_e_max_eta` and `jet_hit_edge` (`n_hit_edge["jet"]` is None). The background is still read once per run: its hash recognises a reused one. MUSIC_2 then does not store its evolution at all (`set_store_evolution(False)`, music4gpu): its frame count and grid stay in its metadata.

Usage (see example/prod_AuAu_0_10_jet/run_prod_jet.py)

```text
jetscape = js.JetScapePerEvent(); ...; jetscape.Init()
writer = PairH5Writer("pair.h5", grid_mode="grid", out_grid=grid)
writer.attach(jetscape)                 # AFTER Init(): sets dump_hydro_only on MUSIC_2
jetscape.ExecInit()
for i in range(n):
    jetscape.ExecPerEvent()
    idx = writer.Exec()                 # both evolutions are live here
    jetscape.ClearPerEvent()
writer.Finish()
```

The writer is not a framework task: it has to be attached after `Init()` and called between `ExecPerEvent()` and `ClearPerEvent()`, the same pattern run_prod.py uses for `H5BulkWriter`. It needs no compiled extension itself, so it is testable with stubs.

#### `PairH5Writer(out_file_name='pair_evo.h5', *, bg_id='MUSIC_1', jet_id='MUSIC_2', grid_mode='grid', out_grid=None, tau_stride=1, choose_ntau=0, compression=DEFAULT_COMPRESSION, keep_bits=None, store_droplets=True, store_showers=True, keep_surface=(), provenance=None, extra_attrs=None, force=True, verbose=False, bg_layout='auto', reuse=1, store_evolution=True, read_jet=True, edge_e_threshold=None, verify_bg_reuse=False, extra_hard=False)`

Write the background and jet legs of a two-stage MUSIC run as one pair file.

Parameters:

- **`out_file_name`** (`str`, default: `'pair_evo.h5'` ) – Output path.
- **`bg_id`** (`str`, default: `'MUSIC_1'` ) – Module ids of the two MUSIC instances (<name> in the XML).
- **`jet_id`** (`str`, default: `'MUSIC_1'` ) – Module ids of the two MUSIC instances (<name> in the XML).
- **`grid_mode`** (`('grid', 'native')`, default: `"grid"` ) – grid resamples both legs onto out_grid; native writes MUSIC's own grid.
- **`out_grid`** (`mapping or Grid`, default: `None` ) – grid mode only; as for jetscape.fast_h5_bulk.H5BulkWriter.
- **`tau_stride`** (`int`, default: `1` ) – Keep every Nth stored MUSIC step, on both legs.
- **`choose_ntau`** (`int`, default: `0` ) – 0 grows the tau axis to the longest leg of any event; N > 0 pins it (longer legs are clipped, with a warning and a count at Finish()).
- **`compression`** (`(str, mapping or None)`, default: `DEFAULT` ) – jetscape.h5_compression spec for arr and arr_bg (default "blosc-zstd"; "lzf" is the old default). See README_h5_optim.md.
- **`keep_bits`** (`int or None`, default: `None` ) – Round both legs to this many float32 mantissa bits (lossy, relative error \<= 2\*\*-(keep_bits+1)); None (default) is bit-exact. Both legs are rounded the same way, so arr - arr_bg is exactly zero wherever the legs agree.
- **`store_droplets`** (`bool`, default: `True` ) – Write source/droplets and shower/.
- **`store_showers`** (`bool`, default: `True` ) – Write source/droplets and shower/.
- **`keep_surface`** (`iterable of {"jet", "bg"}`, default: `()` ) – Legs whose freeze-out surface MUSIC hands to the framework (skip_surface off), e.g. for jetscape.particlize_h5.ParticlizeH5Writer. The legs not listed get skip_surface on (the default for both). MUSIC must also build the surface for a kept leg (\<freeze_out_surface>1 for that instance).
- **`provenance`** (`mapping`, default: `None` ) – Overrides for the provenance attributes (hard_vertex, eos_kind, transport_mode, source_mode ...).
- **`extra_attrs`** (`mapping`, default: `None` ) – More root attributes, written verbatim (run provenance).
- **`force`** (`bool`, default: `True` ) – Overwrite an existing file.
- **`bg_layout`** (`('auto', 'full', 'shared')`, default: `"auto"` ) – How arr_bg is stored. full: one copy per event, as always. shared: each background once (arr_bg_store, arr_bg_rows), arr_bg a virtual dataset over it that reads the same. auto (default): shared if reuse 1, else full, so files without reuse do not change.
- **`reuse`** (`int`, default: `1` ) – Events per background (setReuseHydro), for bg_layout="auto". Which events share a background is still decided from the data (diag/bg_id), never from this number.
- **`store_evolution`** (`bool`, default: `True` ) – False: write no file (no evolutions, droplets or showers) but run every per-event step and check, so Exec()'s index, last_event_diag and last_bg_key are unchanged -- the bookkeeping a particlize file needs without the pair file.
- **`read_jet`** (`bool`, default: `True` ) – False (needs store_evolution=False): do not read the jet leg's evolution; see the module docstring for what diag then holds instead.
- **`edge_e_threshold`** (`float`, default: `None` ) – Energy density [GeV/fm^3], normally MUSIC's freeze-out value e(T_fo). Every event records per leg the largest energy density on the transverse faces of the OUTPUT grid (the first and last x and y planes, over all stored frames) as diag/{bg,jet}\_edge_e_max, and on the eta faces as ...\_edge_e_max_eta. With a threshold, diag/{bg,jet}\_hit_edge is 1 when the transverse value exceeds it: fluid above freeze-out reached the edge of the stored box, so the stored evolution misses part of it (MUSIC's own, larger grid does not). None: the maxima only, no flag.
- **`verify_bg_reuse`** (`bool`, default: `False` ) – A reused background is recognised by a fingerprint of its raw frames (grid, frame count and three whole frames), and then neither read nor resampled again: its frame hashes and edge maxima come from the event that first used it. Not with bg_layout="full" and a file, which stores the background in every row. True: read and resample it anyway and raise if the fingerprint matched a different background (a check for tests, at the old cost).
- **`extra_hard`** (`bool`, default: `False` ) – The events have other hard scatterings (PythiaGun's ): store each event's scatterings/ (Exec's scatterings, SCATTER_COLUMNS rows).

##### `last_event_diag`

The `diag/` values of the last written event.

##### `last_bg_key`

Hex digest of the last event's whole background leg (the IC's fingerprint).

##### `n_clipped`

Legs cut at a pinned `choose_ntau` so far.

##### `n_hit_edge`

Events so far whose leg exceeded `edge_e_threshold` on the transverse faces of the output grid, as `{"bg": n, "jet": n}` (a reused background counts once per event). `"jet"` is None with `read_jet=False`: not measured.

##### `liquefier_tau_delay`

The liquefier's tau_delay [fm/c] (no droplet deposits earlier), or None without a liquefier. Valid after `attach()`.

##### `bg_layout`

`"full"` (one arr_bg row per event) or `"shared"` (each background once).

##### `deposition`

True if the jet leg has the liquefier, False in the null test, None before attach.

##### `attach(jetscape=None, *, bg=None, jet=None, liquefier=None, manager=None)`

Find the two legs, the liquefier and the shower manager; set per-leg flags.

Call after `JetScape.Init()` (which reads the XML flags this overrides) and before the first event. Anything not passed is looked up: the legs by module id in `jetscape.GetTaskList()`, the liquefier on the jet leg (or, with the jet leg's `<AddLiquefier>` off, on the energy-loss manager's template), the manager through the signal manager.

##### `Exec(*, scatterings=None, **diag)`

Write the current event's pair. Returns its index in the file, or None.

Extra keyword arguments are stored under `diag/` for this event; with `extra_hard`, `scatterings` are the event's hard scatterings (S, 8). Must run between `ExecPerEvent()` and `ClearPerEvent()`.

##### `write_diag(i, **scalars)`

Add per-event scalars after `Exec()` (e.g. the driver's wall time).

##### `Finish()`

Close the file. Idempotent.

#### `shower_group_attrs()`

The `shower/` group attributes FastHydro's `PairedH5Writer` writes.

### `jetscape.surface_replay`

python/jetscape/surface_replay.py

SurfaceReplay -- a FluidDynamics that hands a stored freeze-out surface to the framework, so iSS samples it exactly as it would have sampled the live hydro's.

```text
replay = SurfaceReplay()
iss = create_module("iSS")
jetscape = JetScapePerEvent(); jetscape.Add(replay); jetscape.Add(iss); jetscape.Init()
jetscape.ExecInit()
for cells in surfaces:                       # (N, 32) float32, SURFACE_CELL_COLUMNS
    replay.load(cells)
    jetscape.ExecPerEvent()                  # iSS: GetHydroHyperSurface -> these cells
    hadrons = soft_hadrons_numpy(iss)
    jetscape.ClearPerEvent()
```

There is no initial state, pre-equilibrium or evolution: `Init` is overridden so `FluidDynamics::Init` does not insist on them, and `bulk_info` stays empty. The cells go in through `FluidDynamics::StoreSurfaceCell` -- the same call MUSIC's wrapper makes -- so iSS takes its normal `GetHydroHyperSurface` path.

#### `SurfaceReplay(module_id='SurfaceReplay')`

Bases: `FluidDynamics`

Replay one stored surface per event (call `load` before each event).

##### `load(cells)`

The surface for the next event: (N, 32) float32, columns SURFACE_CELL_COLUMNS.

### `jetscape.liquefier_io`

python/jetscape/liquefier_io.py

Read the droplets and the kernel parameters out of a live C++ `CausalLiquefier`.

The droplets are what the jet deposited into the medium this event: one row per droplet, `(tau, x, y, eta, E, px, py, pz)` -- Milne position, Cartesian momentum in GeV -- exactly as `LiquefierBase::add_hydro_sources` stores them (`LiquefierBase.cc:200-225`). No conversion is applied. A MUSIC jet leg evaluates the same droplets itself, through `HydroSourceJETSCAPE`; these helpers only record them.

Nothing here imports fast_data or torch, so any writer can use it. FastHydro's `liquefier_bridge` wraps the same calls into its own types.

#### `droplets(liq)`

The liquefier's current droplet list as an `(M, 8)` float64 array.

`liq` is a bound `LiquefierBase`/`CausalLiquefier` (`droplets_numpy()`). The list holds this event's droplets until a hydro with the liquefier attached clears it in its `ClearTask`.

#### `droplet_fluid(liq)`

Per droplet, the background fluid where it was made, `(M, 4)` = `(vx, vy, vz, e)`: lab Cartesian velocity and energy density [GeV/fm^3] (`e < 0`: not looked up), or `None` for an X-SCAPE build whose droplets do not carry it (`XSCAPE_DROPLET_FLUID`). `<CausalLiquefier><frame>fluid</frame>` diffuses each droplet in this fluid's rest frame.

#### `liquefier_frame(liq)`

The frame of the causal diffusion: `"bjorken"`, `"lab"` or `"fluid"`. A build with only the two-frame option says it through `bjorken_frame`; one without either is `"lab"`.

#### `droplet_fluxes(liq)`

Per droplet, the sum of its point-sampled kernel on the hydro grid (X-SCAPE `LiquefierBase::get_droplet_flux`), or `None` if this X-SCAPE build does not normalize.

-1 marks a droplet that was never normalized. With `normalize_on_hydro_grid` true, the hydro receives each normalized droplet's four-momentum exactly; without it, `flux` times it.

#### `normalize_on_hydro_grid(liq)`

`<Liquefier><normalize_on_hydro_grid>` of this liquefier, or `None` for an X-SCAPE build without the normalization (every droplet deposits what its sampled kernel sums to).

#### `liquefier_params(liq)`

The causal-diffusion kernel parameters as a dict of floats, keyed by `PARAM_KEYS`.

`liq.params()` gives the five input parameters. The two derived ones come from the object when it exposes them (`CausalLiquefier.c_diff`/`gamma_relax`), otherwise from the C++ formulas `c_diff = sqrt(d_diff/time_relax)` and `gamma_relax = 0.5/time_relax` (`CausalLiquefier.cc:41-42`).
