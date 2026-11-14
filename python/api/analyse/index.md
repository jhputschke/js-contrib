# Analyse the files

Read a production: hadron histograms with per-event errors, the wake, single events for jet finding, the freeze-out surfaces and the hydro pair. All of this needs only the [analysis environment](https://jhputschke.github.io/js-contrib/install/analysis-env/index.md), no X-SCAPE build. The guides are [Hadrons](https://jhputschke.github.io/js-contrib/analysis/hadrons/index.md), [Jets with FastJet](https://jhputschke.github.io/js-contrib/analysis/jets/index.md) and [Hydro and hadrons together](https://jhputschke.github.io/js-contrib/analysis/hydro-hadrons/index.md); the file layouts are on [What a production writes](https://jhputschke.github.io/js-contrib/analysis/files/index.md).

## Examples

### Open a campaign and histogram it

`HadronFileReader` reads all files of a campaign as one data set: a directory (every `*_particlize.h5` with its hadron files), a quoted glob, or a list of stems. `hist` and `total` average each event's samples first and then the events, so every event counts the same and the errors are per event.

```python
import numpy as np
from jetscape.hadrons_h5 import HadronFileReader

with HadronFileReader("/path/to/campaign") as r:
    print(r.n_files, "files,", r.n_events, "events, tags", r.tags())
    pt_bins = np.linspace(0, 3, 31)
    dN, err = r.hist("bulk_bg", "pt", pt_bins, mask="charged")          # charged hadrons per event
    dN_pi, _ = r.hist("bulk_bg", "pt", pt_bins, mask="pi+")             # one species
    h2, e2 = r.hist("bulk_bg", ("pt", "eta"), (pt_bins, np.linspace(-1, 1, 5)), mask="charged")
    n, n_err = r.total("bulk_bg", mask=lambda ev, info: ev.charged & (np.abs(ev.eta) < 0.5))
```

- **`values`, `mask`, `weights`** are names (`pt`, `eta`, `y`, `phi`, `E`, `pid`, `charged`, a species such as `"pi+"`, `"K"`, `"p+pbar"`) or callables `f(ev, info)` with one value per hadron. A tuple of names gives an N-dimensional histogram.

### The tags

The first argument of `hist` and `total` names one of the hadron files of each production file, `<stem>_hadrons_<tag>.h5`. The four tags are `jetscape.hadrons_h5.TAGS`:

| tag          | hadrons                                                                                                                                                           | one unit is                                       | also used by                                             |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- | -------------------------------------------------------- |
| `bulk_jet`   | iSS on the jet leg's surface: the medium with the wake                                                                                                            | an event                                          | `jet_event`, `jet_minus_background`                      |
| `bulk_bg`    | iSS on the background's surface: the medium alone                                                                                                                 | a background, shared by the events that reused it | `background_event`, `jet_minus_background`               |
| `jet_frag`   | the jet's own hadrons, the fragmentation of its surviving partons                                                                                                 | an event                                          | `jet_event`, `jet_minus_background(..., fragments=True)` |
| `extra_frag` | fragments of the event's other hard scatterings; only in campaigns with [extra hard scatterings](https://jhputschke.github.io/js-contrib/run/extra-hard/index.md) | an event                                          | as `jet_frag`; `extras=False` leaves them out            |

`r.tags()` gives the tags every file of the campaign has (`r.tags(i)`: file `i`); a call that needs a missing tag raises. The undecayed iSS hadrons of `--keep-primaries` (tags `bulk_jet_primary`, `bulk_bg_primary`, `PRIMARY_TAGS`) are not read by `HadronFileReader`: open those files with `HadronFile`, `Hadrons.from_h5` or `js.HadronArray.from_h5` ([below](#one-file-or-arrays-for-awkward)). The files' layout is on [What a production writes](https://jhputschke.github.io/js-contrib/analysis/files/#4-hadron-files).

### The wake: jet leg minus its own background

Callables get the event's `info`, which jet-relative observables need. Here Δφ to the event's leading shower initiator:

```python
def lead_phi(info):
    ini = info.initiators()                     # (K, 11): shower, pid, pstat, px, py, pz, E, ...
    px, py = ini[np.argmax(np.hypot(ini[:, 3], ini[:, 4])), 3:5]
    return np.arctan2(py, px)

def dphi(ev, info):                             # in [-pi/2, 3pi/2)
    return np.mod(ev.phi - lead_phi(info) + np.pi / 2, 2 * np.pi) - np.pi / 2

def soft_charged(ev, info):
    return ev.charged & (ev.pt > 0.2) & (ev.pt < 4) & (np.abs(ev.eta) < 1)

with HadronFileReader("/path/to/campaign") as r:
    bins = np.linspace(-np.pi / 2, 3 * np.pi / 2, 17)
    wake, wake_err = r.jet_minus_background(dphi, bins, mask=soft_charged)
    full, _ = r.jet_minus_background(dphi, bins, mask=soft_charged, fragments=True)  # + the jet
```

For files hadronized with `--correlated` the error comes from the per-sample differences, chosen automatically (`r.correlated`). Which error to report: [Hadrons](https://jhputschke.github.io/js-contrib/analysis/hadrons/index.md).

### Single events

For jet finding or anything event by event. An event is one oversample of the bulk plus one fragmentation of the jet's partons:

```python
with HadronFileReader("/path/to/campaign") as r:
    ev = r.jet_event(0, 17)                  # global event 0: oversample 17 + one fragmentation
    E, px, py, pz = ev["p"].T                # p = (E, px, py, pz), x = (t, x, y, z)
    pt, phi = np.hypot(px, py), np.arctan2(py, px)
    frag = ev["origin"] == 1                 # 0 medium (bulk), 1 jet fragment
    bg = r.background_event(0, 17)           # the same oversample of its background
    info = r.event_info(0)                   # stem, seed, bg_unit, pthat, n_coll, ...
    ini = info.initiators()                  # the shower initiators, (K, 11)
    for ev in r.iter_jet_events(0):          # every oversample of event 0
        ...
```

`r.n_oversamples(e)` and `r.n_frag(e)` give the counts; oversample k is paired with fragmentation k mod `n_frag(e)`.

### A subset of events

`events=` restricts any histogram, e.g. to a range of the hard scale:

```python
with HadronFileReader("/path/to/campaign") as r:
    hard = [g for g in r.events() if r.event_info(g).pthat > 20]
    h, e = r.hist("jet_frag", "pt", np.linspace(0, 20, 21), mask="charged", events=hard)
```

Campaigns with several pT̂ windows (`--pthat-bins`) are stitched with each window's cross section: `r.pthat_bin_events(k)`, `r.pthat_bin_sigma(k)`, as on [Hadrons](https://jhputschke.github.io/js-contrib/analysis/hadrons/#5-pt-windows-and-cross-sections).

### Jets with FastJet

One sample of an event as charged PseudoJets, clustered with anti-kT. This is the full event, medium included, without background subtraction; areas and subtraction are on [Jets with FastJet](https://jhputschke.github.io/js-contrib/analysis/jets/index.md).

```python
import fastjet as fj
from jetscape.hadrons_h5 import CHARGED

with HadronFileReader("/path/to/campaign") as r:
    ev = r.jet_event(0, 0)
    E, px, py, pz = ev["p"].astype(np.float64).T
    pt = np.hypot(px, py)
    keep = np.isin(np.abs(ev["pid"]), CHARGED) & (pt > 0.15) & (np.abs(np.arcsinh(pz / pt)) < 1.1)
    parts = [fj.PseudoJet(px[i], py[i], pz[i], np.hypot(pt[i], pz[i])) for i in np.flatnonzero(keep)]
    cs = fj.ClusterSequence(parts, fj.JetDefinition(fj.antikt_algorithm, 0.4))
    jets = fj.sorted_by_pt(cs.inclusive_jets(10.0))         # pT > 10 GeV
```

### One file, or arrays for awkward

```python
from jetscape.hadrons_h5 import HadronFile, Hadrons

stem = "/path/to/campaign/AuAu_0_10_jet_seed0001"
with HadronFile(f"{stem}_hadrons_bulk_jet.h5") as hf:     # one file, nothing loaded
    one = hf.sample_event(0, 17)                          # unit 0, sample 17: pid, pstat, p, x
h = Hadrons.from_h5(f"{stem}_hadrons_bulk_bg.h5")         # a whole file in memory
dn, err = h.hist(h.eta, np.linspace(-1, 1, 5), mask=h.charged)   # averaged per sample
```

`HadronArray` turns a file into awkward arrays of `vector` four-momenta, for selections with awkward and FastJet's array interface:

```python
import awkward as ak
import jetscape as js

bulk = js.HadronArray.from_h5(f"{stem}_hadrons_bulk_bg.h5")
had = bulk.to_awkward(by_unit=True)             # units x samples x hadrons, Momentum4D
charged = had[js.select(had, "charged") & (abs(had.eta) < 0.5)]
n_ch = ak.num(charged, axis=-1)                 # charged multiplicity per unit and sample
mean_pt = ak.mean(charged.pt)
```

### Freeze-out surfaces and final partons

```python
from jetscape.particlize_h5 import ParticlizeFile

with ParticlizeFile(f"{stem}_particlize.h5") as pf:
    cells = pf.surface("jet", 0)          # the jet leg's freeze-out cells of event 0, (N, 32)
    cells_bg = pf.surface("bg", 0)        # the background that event used
    partons = pf.partons(0)               # the final partons, (K, 14)
```

The columns are `SURFACE_COLUMNS` and `FINAL_PARTON_COLUMNS` (below).

### The hydro pair

The jet leg `arr` and its background `arr_bg`, shape `(events, 4, nx, ny, neta, ntau)` with the channels energy density, vx, vy, vz. Read frame by frame, and compare only frames both legs have:

```python
import h5py
import hdf5plugin  # noqa: F401  (the files are Blosc-compressed)

with h5py.File(f"{stem}.h5", "r") as f:
    n = min(f["ntau_freezeout"][0], f["ntau_freezeout_bg"][0])    # frames of event 0 in both legs
    de = f["arr"][0, 0, ..., n // 2] - f["arr_bg"][0, 0, ..., n // 2]   # energy density, jet - bg
    x = f.attrs["x_min"] + f.attrs["dx"] * np.arange(f.attrs["nx"])    # cell centres [fm]
    o = f["source/offsets"]
    droplets = f["source/droplets"][o[0]:o[1]]                     # tau, x, y, eta, E, px, py, pz
```

## Hadron file readers

### `jetscape.hadrons_h5`

python/jetscape/hadrons_h5.py

Hadrons in HDF5, with every sample kept apart: HadronH5Writer writes, Hadrons reads.

One file holds one kind of hadrons (`tag`), e.g. from example/prod_AuAu_0_10_jet/ hadronize.py::

```text
bulk_jet   iSS on the jet leg's surface     (bulk + wake)       unit = event
bulk_bg    iSS on the background's surface  (bulk)              unit = background
jet_frag   ColorlessHadronization on the final partons          unit = event
           (with extra hard scatterings: the trigger's showers only)
extra_frag ColorlessHadronization on the showers of the event's  unit = event
           other hard scatterings (PythiaGun's <extraHard>,
           docs/Plans/PLAN_multi_hard.md); absent without them
```

The full event is bulk_jet + jet_frag + extra_frag: the extras belong to the event, not to the trigger (`JetEvents.jet_event` adds extra_frag when the file exists). A minimum-bias job (run_prod_jet.py --hard none) has no jet_frag.

Layout::

```text
attrs   format, format_version, tag, n_samples (per unit, nominal), source (the
        particlize file) + its file_uuid, generator settings ...; the selection:
        eta_max if only hadrons with |eta| < eta_max were kept (absent: all),
        charged_only (absent: False), positions (absent: True)
hadrons/pid            (N,)   int32
hadrons/pstat          (N,)   int32
hadrons/p              (N, 4) float32   [E, px, py, pz]  GeV
hadrons/x              (N, 4) float32   [t, x, y, z]     fm   (zero for Pythia; absent
                                        if the file was written with positions=False)
hadrons/sample_offsets (S + 1,) int64   sample s is rows sample_offsets[s]:[s+1]
hadrons/unit_offsets   (U + 1,) int64   unit u is samples unit_offsets[u]:[u+1]
units/<key>            (U,)             unit (row in the source group), event (first
                                        event using it), seed, n_cells ...
initiators/data        (K, 11) float64  optional, bulk_jet and jet_frag only: the event's
initiators/offsets     (U + 1,) int64   shower-initiating partons (INITIATOR_COLUMNS), unit
                                        u is rows offsets[u]:[u+1]; copied by hadronize.py
                                        from the pair file's shower/, so jet-relative
                                        analyses do not need the (large) pair file
hadronization_input/data     (K, 8) float64  optional, jet_frag only: the parton list the
hadronization_input/offsets  (U + 1,) int64  jet hadronization handed to Pythia
                                        (HADRONIZATION_INPUT_COLUMNS: final partons, beam
                                        remnants, colour-balancing parton), unit u is rows
                                        offsets[u]:[u+1]; with units/seed, enough to
                                        hadronize the event again (prod_OO_isr's
                                        rehadronize_jets.py)
partons/data           (K, 14) float64  optional, jet_frag only: the event's final partons
partons/offsets        (U + 1,) int64   (particlize_h5.FINAL_PARTON_COLUMNS), what the jet
                                        hadronization was handed, unit u is rows
                                        offsets[u]:[u+1]; written by run_prod_jet.py
                                        --hadrons-only, which keeps no particlize file, so
                                        hadronize.py can redo the jets from the jet file.
                                        Attribute x_had: "none" (the partons' production
                                        points only; hadronization points later)
extra_frag only:
hadrons/scatter        (N,)   int32     with --extra-hadronize per_scattering: the hard
                                        scattering each hadron came from (1, 2, ...)
initiators/scatter     (K,)   int32     each initiator's hard scattering
scatterings/data       (S, 8) float64   the event's hard scatterings (SCATTER_COLUMNS of
scatterings/offsets    (U + 1,) int64   particlize_h5: scatter, pthat, code, x, y,
                                        n_partons, E, pt; the trigger's first, scatter 0)
```

Undecayed hadrons. `bulk_jet_primary` and `bulk_bg_primary` (PRIMARY_TAGS) hold iSS's hadrons as sampled, before any decay, for the first N oversamples of each unit (hadronize.py --keep-primaries N), in their own files with this layout: input for an afterburner (SMASH) later. Sample k of a unit there is the undecayed form of sample k of the same unit in the `bulk_jet` / `bulk_bg` file. They are written at full precision with positions and without a selection (SMASH needs every hadron, its position and its mass from E^2 - p^2). The `decayed` attribute tells the two apart.

`p` and `x` are full float32 unless the writer was given `keep_bits`: then they are rounded to that many mantissa bits (`keep_mantissa_bits` / `max_rel_error` on the dataset, `hadron_precision` reads them back); pid, pstat and the offsets are always exact. Rounding is for campaign storage: e.g. `keep_bits={"p": 12, "x": 8}` keeps 58% of the bytes, with relative errors \<= 1.2e-4 (p) and 2e-3 (x). The inputs (particlize file + `units/seed`) reproduce the exact hadrons at any time.

`eta_max` (writer argument, file attribute) keeps only hadrons with pseudorapidity |eta| < eta_max, for campaign storage too: at |eta| < 2 about half of a 0-10% Au+Au bulk event. Every sample keeps its place (possibly empty), so the per-sample averages inside the cut are unchanged; outside it the file has no hadrons, which is not zero hadrons. `charged_only` keeps only the charged hadrons (CHARGED), `positions=False` leaves out `x` (~40% of the bytes). Readers then give `x` as (N, 0) arrays: nothing to use, and nothing silently zero. `hadron_eta_max` and `hadron_selection` read it back.

Two levels of offsets because an oversample is a sample of the whole event: averages are over samples, and the samples of one unit are not independent events. Errors on sample averages use the compound-Poisson estimate of `fasthydro.hadrons` (`Var = sum w^2 / N_samples^2`), which is exact for iSS's sampling.

Writing is incremental (RaggedGroup): a crash loses nothing already appended.

Single events

Every sample is a complete event of that tag and can be taken alone::

```text
h = Hadrons.from_h5(path);   ev = h.sample_event(unit, k)    # in memory
hf = HadronFile(path);       ev = hf.sample_event(unit, k)   # read from disk, lazily
```

`unit` is the recorded unit (`units/unit`: the event for bulk_jet and jet_frag, the background for bulk_bg); `k` counts that unit's samples from 0. `JetEvents` puts the three tags of one production file together: `jet_event(event, k)` is oversample `k` of the jet leg's bulk plus a fragmentation of the same event's partons, and `background_event(event, k)` is oversample `k` of the background that event used. Oversamples of one event share one fluid: they are independent Cooper-Frye samplings, not independent collisions.

A whole campaign

`HadronFileReader` reads the files of many production files (one per seed) as one data set, the way the hydro files of a campaign are read side by side rather than merged: events get a global index, `jet_event`/`background_event` work across seeds, and `hist`/`total`/`jet_minus_background` accumulate event by event over all files, with the background re-evaluated per event (jet-relative observables) and correct errors when a background is reused.

### `jetscape.hadrons_h5.HadronFileReader(source, *, check_uuid=True)`

The hadron files of a whole campaign, read as one data set.

```text
reader = HadronFileReader("out_had")            # every *_particlize.h5 in there
reader.n_events                                  # events of all seeds
ev = reader.jet_event(123, 17)                   # global event 123, oversample 17
dN, err = reader.hist("bulk_jet", "pt", np.linspace(0, 3, 31), mask="charged")
dN, err = reader.jet_minus_background(dphi_jet, bins, mask=soft_charged)
```

`source` is a directory, a glob pattern (of particlize or hadron files), or a list of stems / particlize paths. Every production file needs its `<stem>_particlize.h5` (event count, the event -> background map, the file uuid) and whichever `<stem>_hadrons_<tag>.h5` hadronize.py made -- or, for run_prod_jet.py --hadrons-only, only its hadron files: the event record is then in the jet files' `units/` (`bg_unit`, `pthat_bin` ...) and the run's settings in their attributes (`hadrons_only`). With `check_uuid` a hadron file whose `source_uuid` is not its particlize file's `file_uuid` (hadrons-only: the jet file's) is refused: files renamed or mixed across runs cannot pair up silently.

**Tags.** The first argument of `hist` and `total` is one of `TAGS`, the four hadron files a production file can have, `<stem>_hadrons_<tag>.h5`:

- `bulk_jet`: iSS on the jet leg's surface, the medium with the wake; one unit per event. Also used by `jet_event` and `jet_minus_background`.
- `bulk_bg`: iSS on the background's surface, the medium alone; one unit per background, shared by the events that reused it. Also used by `background_event` and `jet_minus_background`.
- `jet_frag`: the jet's own hadrons, the fragmentation of its surviving partons; one unit per event. Also in `jet_event` and, with `fragments=True`, `jet_minus_background`.
- `extra_frag`: the fragments of the event's other hard scatterings, only in campaigns with extra hard scatterings; used like `jet_frag` (`extras=False` leaves them out).

`tags()` gives the tags every file of the campaign has, `tags(i)` those of file `i`; a call that needs a missing tag raises. The undecayed iSS hadrons of `--keep-primaries` (`PRIMARY_TAGS`) are not read here: open those files with `HadronFile` or `Hadrons.from_h5`.

**Histograms** are event averages of per-sample means: for every event the samples of its unit are histogrammed and divided by that unit's number of samples, and these per-event means are averaged over the events -- every event weighs the same, even when the files were hadronized with different `--oversample`. Errors are compound-Poisson per event, added over events; `jet_minus_background` of correlated legs (hadronize.py --correlated) takes its error from the per-sample differences instead (`paired`). With common seeds (hadronize.py --common-seeds or --correlated) the jet legs of the events sharing a background draw its random numbers, sample by sample, so they are not independent: bulk_jet's error then comes from the spread of their per-sample sums (the events of one background summed per sample, as `paired` does). `values`, `mask` and `weights` are either a name (an `EventHadrons` attribute, or 'charged', or a species name for `mask`) or a callable `f(ev, info)` returning one value per hadron, where `ev` is an `EventHadrons` and `info` an `EventInfo` -- so jet-relative observables use `info.initiators()`. For an N-d histogram `values` is a tuple of names, or a callable returning a tuple. A background reused by several events counts once for each of them, and its hadrons, counted several times, enter the error as the correlated sum they are. Events whose unit has no samples (an empty surface) are skipped.

`per_event` decides how a reused background is evaluated. Names depend on the hadrons only, so by default (None) a background is binned once and counted for each of its events when `values`, `mask` and `weights` are all names -- n events sharing a background cost one pass, not n. A callable may depend on the event (its `info`), so with one the background is binned once per event with that event's `info`. Pass `per_event=False` for callables that don't use `info` (a lab-frame tuple such as `lambda ev, info: (ev.eta, ev.phi, ev.pt)`) to get the single pass for them too; `True` forces the per-event evaluation. Both give the same result.

#### `pthat_bins`

The campaign's pTHat windows, a (K, 2) array [GeV], or None if its files were made without --pthat-bins. All files must have the same windows.

#### `correlated`

True if every file's bulk_jet and bulk_bg were sampled correlated (hadronize.py --correlated).

#### `duplicate_backgrounds()`

Backgrounds found in more than one production file, as a list of `{"kind", "events"}`: `events` the global index of the first event using that background in each file. `kind` "identical": the same background leg bit for bit (particlize `events/bg_key`, a hash of it on the output grid); "probable": same production seed and the same number of freeze-out cells (`n_cells_bg`, which does not depend on the output grid) but another hash, as for the same collision written on another grid. Both come from campaigns run over the same seeds.

#### `precision(file_index)`

`{tag: {"p": bits, "x": bits}}` of one production file (None = full float32).

#### `eta_max(file_index=None)`

The |eta| cut the hadrons were stored with (hadronize.py --eta-max; None: none). `file_index` None: the campaign's, i.e. the smallest cut of any file and tag; else `{tag: cut}` of that production file.

#### `selection(file_index=None)`

What the hadron files store (hadronize.py --eta-max, --charged, --no-x). `file_index` None: the campaign's, `{"eta_max": the smallest cut or None, "charged_only": True if any file is, "positions": True if every file has them}`; else `{tag: selection}` of that production file.

#### `tags(file_index=None)`

Tags present in every file (or in one file).

#### `locate(event)`

Global event index -> (file index, event inside that production file).

#### `pthat_bin_events(k)`

Global indices of the events in pTHat window `k`, for `events=`.

#### `pthat_bin_sigma(k)`

Window `k`'s cross section [mb] and its error over the whole campaign: the files' Pythia estimates, weighted by their accepted events. To combine windows, weight each window's event average by its cross section.

#### `pthat_bin_acceptance(k)`

Window `k`'s acceptance of the parton rapidity cut (run_prod_jet.py --parton-ymax) over the whole campaign: events kept / tried, and its binomial error. (1.0, 0.0) without a cut. The cross sections (`pthat_bin_sigma`) already include it (sigma_gen = Pythia's x kept/tried per file).

#### `events(selection=None)`

Global event indices: all (None), one int, a slice, or an iterable.

#### `n_samples(tag, event)`

Samples of `tag` for global `event` (bulk_bg: of the background it used).

#### `has_extras(file_index=None)`

True if the production file(s) have extra hard scatterings (extra_frag).

#### `jet_event(event, k, frag_sample=None, fragments=True, extras=True)`

Oversample `k` of global `event`'s jet-leg bulk plus a fragmentation, and the extras' fragments if there are any (see `JetEvents.jet_event`).

#### `hist(tag, values, bins, *, mask=None, weights=None, events=None, per_event=None)`

Sample-averaged histogram of `tag` over `events` (default all) and its error. See the class docstring for `values`, `mask`, `weights` and `per_event`.

#### `total(tag, *, mask=None, weights=None, events=None, per_event=None)`

Sample-averaged sum over hadrons (default: count) and its error.

#### `jet_minus_background(values, bins, *, mask=None, weights=None, events=None, fragments=False, paired=None, per_event=None, extras=True)`

(+ + ) - over the events that have both surfaces, each event against its own background. -> (difference, error).

`fragments`: add the event's fragments: jet_frag, and the extra hard scatterings' extra_frag when the files have it (the full event; `extras=False` leaves them out). A fragment tag no file has is left out (a minimum-bias campaign has no jet_frag).

`paired`: the error of the bulk difference from the spread of the per-sample differences, sample k of the jet leg minus sample k of its background (the events sharing a background summed per sample), instead of the two legs' errors added. With correlated legs (hadronize.py --correlated) only the paired error is right: the added one ignores that the legs' noise cancels and is ~3x too large. None (default): paired if the files were sampled correlated (`correlated`). Paired needs the same number of samples, at least 2, on both legs of an event. `per_event`: see the class docstring.

### `jetscape.hadrons_h5.EventHadrons(h, u)`

All samples of one tag for one event: per-hadron arrays with kinematics.

Attributes: `pid`, `pstat`, `p` [E, px, py, pz], `x` [t, x, y, z], `E`, `px`, `py`, `pz`, `pt`, `eta`, `y`, `phi`, `charged`, `sample` (the sample each hadron belongs to, 0 .. n_samples-1) and `n_samples`.

#### `species(name)`

Boolean mask for a name in SPECIES, 'charged', or 'all'.

### `jetscape.hadrons_h5.EventInfo(reader, g, i, local)`

What is known about one event of a campaign (passed to the callables of `HadronFileReader.hist`).

#### `initiators()`

This event's shower-initiating partons, as a (K, 11) array, columns `shower, pid, pstat, px, py, pz, E, x, y, z, t`: from the hadron files' `initiators/` (hadronize.py copies them there), else from the pair file's `shower/`. With extra hard scatterings these are the trigger's (bulk_jet's and jet_frag's initiators); the extras' are in the extra_frag file.

#### `scatterings()`

This event's hard scatterings, (S, 8) in particlize_h5.SCATTER_COLUMNS order (the trigger's first, scatter 0); None without extra hard scatterings.

### `jetscape.hadrons_h5.HadronFile(path)`

A hadron file opened for random access: single samples are read from disk, so a campaign-sized file is never loaded whole.

```text
with HadronFile("…_hadrons_bulk_jet.h5") as hf:
    ev = hf.sample_event(3, 17)       # oversample 17 of event 3
```

#### `initiators(unit)`

Shower initiators of recorded unit `unit` (an event), (K, 11) in `INITIATOR_COLUMNS` order; None if the file has no `initiators/`.

#### `hadronization_input(unit)`

The Pythia input of recorded unit `unit` (an event) as a dict of pid, col, acol (int32), p (K, 4) [E, px, py, pz] and mass (float64), what core.hadronize_input takes; None if the file has no `hadronization_input/`.

#### `partons(unit)`

The final partons of recorded unit `unit` (an event), (K, 14) in `PARTON_COLUMNS` order; None if the file has no `partons/`.

#### `scatterings(unit)`

The hard scatterings of recorded unit `unit` (an event), (S, 8) in particlize_h5.SCATTER_COLUMNS order; None if the file has no `scatterings/` (extra_frag files have it).

#### `initiator_scatter(unit)`

Each initiator's hard scattering (aligned with `initiators`); zeros for a file without `initiators/scatter`.

#### `sample_event(unit, k)`

Sample `k` of recorded unit `unit`: dict of pid, pstat, p, x (see FIELDS), and `scatter` if the file has per-hadron scatterings (extra_frag, per_scattering).

### `jetscape.hadrons_h5.Hadrons(pid, pstat, p, x, sample_offsets, unit_offsets, attrs=None, units=None)`

All hadrons of one file, with per-unit and per-sample structure and kinematics.

```text
h = Hadrons.from_h5("hadrons_bulk_jet.h5")
dndy, err = h.hist(h.y, np.linspace(-1, 1, 11), mask=h.species("charged"))
```

`hist` / `total` return the average over the selected units of the per-sample quantity, i.e. sum of weights divided by the number of samples in those units, and its compound-Poisson error.

#### `from_h5(path, units=None)`

Read a file (`units`: an index array/slice to keep only those units).

#### `unit_index(unit)`

Position of recorded unit `unit` (`units/unit`) among the loaded units.

#### `n_samples(unit)`

Number of samples (oversamples / fragmentations) of recorded unit `unit`.

#### `sample_event(unit, k)`

Sample `k` of recorded unit `unit` as one event: dict of pid, pstat, p [E, px, py, pz], x [t, x, y, z].

#### `species(name)`

Boolean mask for a name in SPECIES, 'charged', or 'all'.

#### `hist(values, bins, mask=None, weights=None, units=None)`

Sample-averaged histogram over `units` (default all), and its error. `values` may be a tuple for an N-d histogram.

#### `total(mask=None, weights=None, units=None)`

Sample-averaged sum over hadrons (default: count), and its error.

### `jetscape.hadrons_h5.JetEvents(bulk_jet=None, jet_frag=None, bulk_bg=None, particlize=None, extra_frag=None)`

The hadron files of one production file, combined per event.

```text
with JetEvents.from_stem("out/AuAu_0_10_jet_seed0001") as je:
    ev = je.jet_event(0, 17)          # bulk_jet oversample 17 + a fragmentation
    bg = je.background_event(0, 17)   # the same event's background, oversample 17
```

`jet_event` concatenates the bulk hadrons and the fragments and adds an `origin` array (`ORIGIN`: 0 bulk, 1 jet fragment). The fragmentation used for oversample `k` is `frag_sample` if given, else `k mod n_frag` -- with `--n-frag` equal to `--oversample` every oversample gets its own. The background an event used comes from the particlize file (`events/bg_unit`), or for run_prod_jet.py --hadrons-only files from the jet files' `units/bg_unit`, which is what makes `--reuse` work; without either only events that started a background can be looked up.

#### `from_stem(stem)`

The files hadronize.py writes for `<stem>_particlize.h5` (missing ones are skipped).

#### `n_oversamples(event)`

iSS oversamples of `event`'s jet leg (0: its surface was empty).

#### `bg_unit(event)`

The background unit `event` used.

#### `jet_event(event, k, frag_sample=None, fragments=True, extras=True)`

Oversample `k` of `event`'s jet-leg bulk plus one fragmentation of its partons: dict of pid, pstat, p, x and origin (0 bulk, 1 fragment, 2 the fragments of the event's other hard scatterings). With an extra_frag file the extras are part of the event (`extras=False` leaves them out); they use the same fragmentation index.

#### `scatterings(event)`

`event`'s hard scatterings (S, 8), SCATTER_COLUMNS; None without extras.

#### `background_event(event, k)`

Oversample `k` of the background `event` used: dict of pid, pstat, p, x.

#### `iter_jet_events(event, frag_sample=None)`

All oversamples of `event` as jet events (k = 0 .. n_oversamples - 1).

### `jetscape.hadrons_h5.pseudorapidity(p)`

eta of momenta `p` = (N, 4) [E, px, py, pz]; ~+-690 along the beam, 0 for p = 0. The one definition used by the writer's cut and by `Hadrons`.

### `jetscape.hadrons_h5.hadron_precision(f)`

`{"p": bits, "x": bits}` a hadron file (path or open h5py.File) was written with; None means full float32.

### `jetscape.hadrons_h5.hadron_eta_max(f)`

The |eta| cut a hadron file (path or open h5py.File) was written with; None: none.

### `jetscape.hadrons_h5.hadron_selection(f)`

What a hadron file (path or open h5py.File) stores: `{"eta_max": cut or None, "charged_only": bool, "positions": bool}` (hadronize.py --eta-max, --charged, --no-x).

### `jetscape.hadrons_h5.TAGS = ('bulk_jet', 'bulk_bg', 'jet_frag', 'extra_frag')`

The hadron tags, one file each (`<stem>_hadrons_<tag>.h5`): `bulk_jet` (iSS, the jet leg: medium + wake), `bulk_bg` (iSS, the background: medium alone), `jet_frag` (the jet's own hadrons), `extra_frag` (fragments of the other hard scatterings, with extra hard scatterings only). `HadronFileReader` reads these four.

### `jetscape.hadrons_h5.PRIMARY_TAGS = ('bulk_jet_primary', 'bulk_bg_primary')`

iSS's undecayed hadrons (`--keep-primaries`), one file per bulk tag; read them with `HadronFile` or `Hadrons.from_h5`, not `HadronFileReader`.

### `jetscape.hadrons_h5.FIELDS = ('pid', 'pstat', 'p', 'x')`

### `jetscape.hadrons_h5.ORIGIN = {'bulk': 0, 'frag': 1, 'extra': 2}`

### `jetscape.hadrons_h5.CHARGED = (211, 321, 2212, 3222, 3112, 3312, 3334, 11, 13)`

### `jetscape.hadrons_h5.SPECIES = {'pi+': (211,), 'pi-': (-211,), 'K+': (321,), 'K-': (-321,), 'p': (2212,), 'pbar': (-2212,), 'pi': (211, -211), 'K': (321, -321), 'p+pbar': (2212, -2212)}`

## Particlize file reader

### `jetscape.particlize_h5`

python/jetscape/particlize_h5.py

ParticlizeH5Writer -- store everything needed to hadronize a two-stage MUSIC event later, exactly: each leg's freeze-out surface with every field iSS reads, and the final partons the jet hadronization receives. ParticlizeFile reads it back.

```text
IS -> Hard -> MUSIC_1 -> Matter+LBT (+ liquefier) -> MUSIC_2       (production job)
    surface/bg  <- MUSIC_1's surface (once per background, see --reuse)
    surface/jet <- MUSIC_2's surface
    partons/    <- JetEnergyLossManager's final partons
    initiators/ <- the shower-initiating partons (as the pair file's shower/initiators)
                                                -> <stem>_particlize.h5
later, anywhere, as often as wanted (example/prod_AuAu_0_10_jet/hadronize.py):
    iSS on surface/jet -> bulk_jet, iSS on surface/bg -> bulk_bg,
    ColorlessHadronization on partons/ -> jet_frag           -> hadrons_<tag>.h5
```

Why this is exact

- The framework's `SurfaceCellInfo` is `Jetscape::real` = float (`RealType.h`), and iSS copies exactly the 32 fields in `SURFACE_COLUMNS` out of it (`iSpectraSamplerWrapper::getSurfCellVector`). A float32 row is therefore what iSS would have received inside the job, bit for bit.
- `partons/` is `JetEnergyLossManager::GetFinalStatePartons` -- the list the HadronizationManager is handed -- not a reconstruction from the shower graph.
- Both legs come from the same job, so they share the initial condition by construction.

It is a separate file from the FNO4d pair file on purpose: the training data never carries hadronization inputs, and these files can be moved, reprocessed or deleted on their own. From format version 2 it is **self-contained** for hadronization and the jet-relative analyses: `initiators/` holds each event's shower initiators, which before were only in the pair file's `shower/initiators`. `add_initiators_from_pair` adds them to a version-1 file from its pair file (`example/prod_AuAu_0_10_jet/add_initiators.py`).

Layout

::

```text
attrs   format, format_version, producer, surface_columns, parton_columns, units,
        T_fo, music_input (the job's music_input text: iSS reads its EoS id and flags
        from it), pair_file, legs, nevents, nevents_written, complete, provenance ...
surface/jet/cells    (N, 32) float32   one unit per event          (RaggedGroup)
surface/jet/offsets  (nevents + 1,) int64
surface/bg/cells     (M, 32) float32   one unit per NEW background (bg_id changes)
surface/bg/offsets   (n_bg + 1,) int64
surface/bg/bg_id     (n_bg,) int64     the first event that used this background
partons/data         (K, 14) float64   FINAL_PARTON_COLUMNS         (RaggedGroup)
partons/offsets      (nevents + 1,) int64
initiators/data      (J, 11) float64   jetscape.showers.INITIATOR_COLUMNS: shower, pid,
initiators/offsets   (nevents + 1,)    pstat, px, py, pz, E, x, y, z, t  (version >= 2)
events/<key>         (nevents,)        per event: bg_id, bg_unit (row of surface/bg),
                                       bg_key, n_cells_jet, n_cells_bg, n_partons,
                                       n_showers, plus whatever the driver passes
                                       (droplet energies, MUSIC tau0, seeds ...)
```

With the event's other hard scatterings (`extra_hard=True`, PythiaGun's ; docs/Plans/PLAN_multi_hard.md)::

```text
initiators/scatter   (J,) int32        the hard scattering of each initiator (0 the
                                       trigger, 1, 2, ... the extras), so of each shower
scatterings/data     (S, 8) float64    SCATTER_COLUMNS: one row per hard scattering
scatterings/offsets  (nevents + 1,)    (the trigger's first, unless there is none)
```

An event whose surface is empty (e.g. MUSIC stopped at the grid boundary) is an empty unit, never a missing one; `events/n_cells_jet` says so.

Like `jetscape.pair_h5.PairH5Writer` this is not a framework task: attach it after `JetScape.Init()` and call `ParticlizeH5Writer.Exec` between `ExecPerEvent()` and `ClearPerEvent()`. It needs no compiled extension itself (stubs work in tests).

Memory sink (`sink=callable`, no file): the same collection and checks, but each event goes to `sink(record)` instead of the file, as a dict -- `event`, `row` (what `events/` would get; `write_events` adds to it), `jet` (the jet leg's cells), `bg` (a new background's cells, else None), `bg_unit`, `partons`, `initiators`. The root attributes the file would get (`file_uuid`, `music_input` ...) are in `attrs`. `run_prod_jet.py --hadrons-only` hadronizes from it (docs/Plans/PLAN_AuAu_hadrons_only.md). There is no resume: a production job restarts from its first event anyway (`run_jobs.sh` reruns incomplete seeds), so the file is rewritten.

### `jetscape.particlize_h5.ParticlizeFile(path)`

Read a particlize-input file.

```text
with ParticlizeFile("AuAu_0_10_jet_seed0001_particlize.h5") as pf:
    cells = pf.surface("jet", 0)            # (N, 32) float32
    cells_bg = pf.surface("bg", 0)          # the background event 0 used
    partons = pf.partons(0)                 # (K, 14) float64
```

#### `has_initiators`

True if the file holds `initiators/` (format version 2, or added later). A group without a single row (bindings without `get_shower_initiating_partons`) counts as absent, so readers fall back to the pair file: every real event has at least one initiator.

#### `extra_hard`

True if the events have other hard scatterings (`scatterings/`).

#### `events(key)`

`events/<key>` as an array (NaN / '' where an event did not set it).

#### `bg_units()`

Number of stored backgrounds and their bg_id (first event) values.

#### `surface_unit(leg, unit)`

Cells of unit `unit` of `surface/<leg>` (an event for jet, a background for bg).

#### `surface(leg, event)`

Cells of `leg`'s surface for `event` (bg: the background it used).

#### `initiators(event)`

Shower initiators of `event`, (J, 11) in `INITIATOR_COLUMNS` order; None without `initiators/`.

#### `initiators_all()`

`(data, offsets)` of every event, as the pair file's `shower/initiators` and `shower/initiator_offsets`; None without `initiators/`.

#### `initiator_scatter(event)`

The hard scattering of each of `event`'s initiators (0 the trigger), aligned with `initiators`; all 0 for a file without extras.

#### `scatterings(event)`

`event`'s hard scatterings, (S, 8) in SCATTER_COLUMNS order; None for a file without extras.

### `jetscape.particlize_h5.split_by_scatter(partons, initiators, initiator_scatter)`

An event's final partons and initiators split into the triggering scattering's and the extras': -> ((partons, initiators) of scatter 0, (partons, initiators, the initiators' scatter) of scatter > 0). A shower's scattering is its initiator's (`initiator_scatter`, aligned with `initiators`); the rows keep their order and shower index, so the trigger's are exactly what an event without extras holds.

### `jetscape.particlize_h5.SURFACE_COLUMNS = ('tau', 'x', 'y', 'eta', 'ds0', 'ds1', 'ds2', 'ds3', 'u0', 'u1', 'u2', 'u3', 'e', 'T', 'P', 'nB', 'nQ', 'nS', 'muB', 'muQ', 'muS', 'pi00', 'pi01', 'pi02', 'pi03', 'pi11', 'pi12', 'pi13', 'pi22', 'pi23', 'pi33', 'Pi')`

### `jetscape.particlize_h5.FINAL_PARTON_COLUMNS = ('shower', 'pid', 'pstat', 'E', 'px', 'py', 'pz', 't', 'x', 'y', 'z', 'mass', 'col', 'acol')`

### `jetscape.particlize_h5.SCATTER_COLUMNS = ('scatter', 'pthat', 'code', 'x', 'y', 'n_partons', 'E', 'pt')`

## Arrays and awkward views

### `jetscape.hadron_array`

python/jetscape/hadron_array.py

HadronArray: one event's hadrons as one array per field, grouped into samples (iSS oversamples; one sample for jet hadronization), for the Pythonic layer (docs/Plans/PLAN_pyjetscape_pythonic.md, step 1).

It wraps the dict the bindings return (soft_hadrons_numpy, hadronization_hadrons_numpy, hadronize_partons, soft_sample_cells) without copying: with soft_set_compact_output iSS's per-field vectors are already numpy arrays, and a record array would have to interleave them (memory constraint M1). It stores no derived columns (M2): pt, eta, phi and rapidity come from to_awkward() as `vector` Momentum4D records, computed when used. to_hadrons() converts to hadrons_h5.Hadrons when its hist/total are wanted.

```text
pid    (n,)    int32
pstat  (n,)    int32
p      (n, 4)  float32   E, px, py, pz
x      (n, 4)  float32   t, x, y, z    (None: no positions, e.g. a file written without)
mass   (n,)    float32                 (None: not known, e.g. read from a hadron file)
sample_offsets (n_samples + 1,) int64; sample s is rows sample_offsets[s]:[s+1]
```

The same arrays go to a hadron file (to_dict() for HadronH5Writer.append_unit) and come back from one (from_h5: the raw fields only, no derived columns, with the units' sample offsets and their table in `info`).

#### `HadronArray(pid, pstat, p, x, mass, sample_offsets=None, info=None)`

One event's hadrons, flat per field, with sample offsets. numpy only.

##### `sample_counts: np.ndarray`

Hadrons per sample, (n_samples,) int64.

##### `from_dict(d) -> 'HadronArray'`

From a bindings hadron dict; the arrays are taken as they are (no copy). `sample_counts` (one per oversample) becomes `sample_offsets`; without it the hadrons are one sample. Other keys go to `info`.

##### `from_h5(path, units=None) -> 'HadronArray'`

A hadron file (hadrons_h5 format: jet_frag, bulk_bg, ...) as one HadronArray: the stored fields as they are (float32, no derived columns), every sample of every unit in file order. `units`: an index array or slice to read only those units.

info: `unit_offsets` (samples of unit u: unit_offsets\[u\]:[u+1]), `units` (the file's per-unit table: event, seed, event_weight, ...), `attrs` (the file's attributes), `path`. The mass is not stored (None).

##### `to_dict() -> dict`

The bindings' hadron dict (pid, pstat, p, x, mass, sample_counts), the arrays themselves (no copy): what HadronH5Writer.append_unit takes for one unit.

##### `to_hadrons(unit_offsets=None, attrs=None)`

-> hadrons_h5.Hadrons (the file's units after from_h5, else all samples as one unit, unless `unit_offsets` says otherwise), for its hist/total. Hadrons computes its derived columns in float64: about 2.4x the memory of this array.

##### `to_awkward(position=False, charge=False, by_unit=False)`

samples x hadrons as an awkward array of Momentum4D records over these buffers (awkward_views.to_awkward); by_unit (a HadronArray from_h5): units x samples x hadrons.

### `jetscape.awkward_views`

python/jetscape/awkward_views.py

Nested views of the Pythonic layer's flat arrays (docs/Plans/PLAN_pyjetscape_pythonic.md, step 1): to_awkward() and PDG selections. awkward, vector and particle are imported inside the functions, never when jetscape is imported.

```text
to_awkward(HadronArray)          samples x hadrons, Momentum4D records over the same
                                 buffers (no copy; p's columns are strided views)
to_awkward(final partons or      showers x partons (ak.unflatten on the runs of
           initiator records)    `shower`; small tables, a copy)
to_awkward(edge, node or hard    a flat record array
           parton records)
```

Momentum fields are E, px, py, pz, so `vector` gives pt, eta, phi, rapidity, mass when asked (computed, not stored), and FastJet's ClusterSequence takes the result directly: one call clusters every sample. A stored mass (HadronArray.mass, the records' `mass` column: X-SCAPE's rest mass) becomes the field `rest_mass`: vector takes every temporal name (E, mass, m, tau, ...) as a coordinate and, from 1.9 on, refuses a record with two. The position is a nested `position` field (a Vector4D record t, x, y, z): `vector` would read flat t, x, y, z fields as the coordinates of the Momentum4D record. Node records (no momentum) are Vector4D records themselves.

```text
select(arr, "pi+"), select(arr, ["K+", "K-"]), select(arr, "charged")
```

returns a mask of arr's shape. Species are PDG names as `particle` spells them ("pi+", "K-", "p~", "Lambda", "K(S)0") or classes: "charged", "neutral", "hadron", "meson", "baryon", "strange". Each distinct pid is looked up once.

"charged" is the PDG charge != 0, for every pid: resonances included. hadrons_h5.CHARGED (used by Hadrons.charged and the analyses) lists the stable charged hadrons and leptons only; the two agree on decayed hadrons and differ on files with undecayed primaries (iSS ).

#### `pdg_charge(pid) -> np.ndarray`

PDG charge [e] per pid (float64; quarks fractional, unknown pids 0).

#### `to_awkward(obj, position=False, charge=False, by_unit=False)`

The nested awkward view of a HadronArray, a hadron dict or a record table.

position: add the nested `position` (HadronArray; records always carry theirs). charge: add a `charge` field [e] from the PDG (needs `particle`). by_unit: a HadronArray read from a file (from_h5): units x samples x hadrons, over info["unit_offsets"].

#### `select(arr, species)`

Mask of the hadrons (or partons) of `species` in arr: a HadronArray (flat mask), an awkward array with a `pid` field (a mask of its shape), a record array or a pid array. species: a name or class, or a list of them (any of them selects).

### `jetscape.records`

python/jetscape/records.py

Record dtypes for the small per-event tables of the Pythonic layer (docs/Plans/PLAN_pyjetscape_pythonic.md, step 1), and the conversions from the flat float64 arrays the bindings return. numpy only: works without the compiled extension, e.g. on arrays read back from files.

One convention everywhere: momentum (E, px, py, pz), position (t, x, y, z), the order of FourVector.comp(i). The old arrays keep their own column orders; these functions only reorder and retype, bit for bit (ids and statuses are integers stored as float64, so the int32 cast is exact; colour tags are C++ unsigned int, hence uint32: files written before the tag fix hold tags above 2\*\*31).

```text
FINAL_PARTON_DTYPE  final_partons_numpy (FINAL_PARTON_COLUMNS, same order)
INITIATOR_DTYPE     showers.initiators_from_manager (INITIATOR_COLUMNS)
EDGE_DTYPE          PartonShower.to_numpy (src, tgt, pid, pstat, px py pz E, x y z t)
NODE_DTYPE          PartonShower.vertices_to_numpy (id, x, y, z, t)
HARD_PARTON_DTYPE   hard_partons_numpy (pid, px, py, pz, E, pT, y)
```

Hadrons are not records: they stay one array per field (HadronArray), so iSS's arrays are wrapped without a copy.

#### `final_parton_records(rows) -> np.ndarray`

(N, 14) FINAL_PARTON_COLUMNS -> FINAL_PARTON_DTYPE records (same field order).

#### `final_parton_rows(partons) -> np.ndarray`

FINAL_PARTON_DTYPE records (or an (N, 14) array, passed through) -> (N, 14) float64 in FINAL_PARTON_COLUMNS order: what hadronize_partons and the files take.

#### `initiator_records(rows) -> np.ndarray`

(K, 11) INITIATOR_COLUMNS -> INITIATOR_DTYPE records.

#### `initiator_rows(initiators) -> np.ndarray`

INITIATOR_DTYPE records (or a (K, 11) array, passed through) -> (K, 11) float64 in INITIATOR_COLUMNS order: what HadronH5Writer.append_unit(initiators=...) takes.

#### `edge_records(rows) -> np.ndarray`

PartonShower.to_numpy's (n, 12) -> EDGE_DTYPE records.

#### `node_records(rows) -> np.ndarray`

PartonShower.vertices_to_numpy's (n, 5) -> NODE_DTYPE records.

#### `hard_parton_records(rows) -> np.ndarray`

hard_partons_numpy's (n, 7) -> HARD_PARTON_DTYPE records (hardest first, as given).

### `jetscape.particles`

python/jetscape/particles.py

A property view of one particle (docs/Plans/PLAN_pyjetscape_pythonic.md, step 7):

```text
v = view(p)
v.pt, v.eta, v.e, v.momentum      # instead of p.pt(), p.eta(), p.e(), ...
view(p).unwrap() is p
```

The bound classes keep their methods (p.pt()); properties there would replace them and break every script that calls them. Each property is one call into C++, about 1.4 us (PLAN_pyjetscape_complete.md, V4): for many particles use the arrays (final_parton_array(), hadron_array(), to_awkward()), not a loop over views.

#### `ParticleView(particle)`

Read-only properties over a Parton, Hadron or Photon (any JetScapeParticleBase).

##### `momentum`

FourVector, iterates as (E, px, py, pz).

##### `position`

FourVector, iterates as (t, x, y, z).

##### `unwrap()`

The C++ particle itself.

#### `view(particle) -> ParticleView`

A ParticleView of `particle`.

## Shower graphs

### `jetscape.showers`

The parton shower as a space-time graph -- capture, store, and read back for animation.

This module lives in PyJetscape so that any writer can store showers -- the MUSIC pair writer (`jetscape.pair_h5`) as well as FastHydro's. It came from FastHydro, which re-exports it as `fasthydro.showers`; it needs nothing but numpy and duck-types the energy-loss manager. `PairBrowser` below is `fasthydro.browse.PairBrowser`.

The droplets in `source/droplets` are what the shower *gave up*. This is what the shower *was*: every parton, every splitting vertex, with the times at both ends, so the shower can be drawn developing alongside the hydro frames instead of appearing all at once.

What makes it animatable

Each parton carries its own production point `(x, y, z, t)` and its momentum, and `source_id`/`target_id` say which splitting it runs between. So a parton is a *segment in space-time*, not a point: it starts where it was produced and travels in a straight line at `v = p/E` until it splits. `segments()` returns those two endpoints and `shower_at()` on `PairBrowser` draws the partial segment at a given lab time. Final-state partons alone would be endpoints with no history -- a starburst, not a shower.

**The vertices carry no position.** `JetEnergyLoss.cc:414-419` constructs every one of them as `Vertex(0, 0, 0, currentTime)`: the spatial coordinates are hardcoded zero and only `t` is real. Measured on a live run, all 72 vertices of an event sat at the origin while the partons had 18 distinct production points. So the vertex rows are stored for their *times and topology* -- which parton splits into which, and when -- and the geometry is taken from the partons. Building segments from vertex positions instead gives every parton zero length, which looks like a plot bug rather than a data bug, so `segments()` does not touch those columns.

The end time of a parton is the production time of its children, falling back to its target vertex's time when it has none. The framework's first two vertices are default-constructed at `t = 0` (`JetEnergyLoss.cc:306-307`) even when the shower starts later, so the children are the reliable source and the vertex time is only the fallback for final partons. Checked on a live event: propagating the leading parton from its production point along `p/E` to its child's time reproduces the child's stored position to all printed digits.

Fates

`pstat` is the link between this group and `source/droplets`. X-SCAPE's `LiquefierBase` re-stamps partons as it filters them (`src/framework/LiquefierBase.cc:24-26`, `:105-160`):

```text
-11  drop   dropped into the medium below the threshold -- THIS is what became a droplet
-17  neg    negative ("hole") parton, the recoil's partner
-13  miss   missed
 22         photon or lepton (a Z's decay), passed through untouched
101         Matter's hand-off point to LBT
```

`fates()` maps them to names. Colouring segments by fate is what turns the animation from a tangle into a picture of energy leaving the jet.

Coordinates

Vertices come out of the framework in **Cartesian lab** `(x, y, z)` [fm] with `t` [fm/c], while the hydro frames are Milne `tau` and the droplets are `(tau, x, y, eta)`. They are stored raw -- that is the framework's own truth and converting on write would bake in a choice -- and `to_milne()` puts them on the hydro clock when a viewer wants them there.

Layout in the file (all float64, ragged, offsets per event, exactly as `source/droplets`)::

```text
shower/partons          (N, 13)  shower, i_src, i_tgt, pid, pstat, px,py,pz,E, x,y,z,t
shower/vertices         (M,  6)  shower, node_id, x, y, z, t   <- x,y,z are ALWAYS 0
shower/initiators       (K, 11)  shower, pid, pstat, px,py,pz,E, x,y,z,t
shower/parton_offsets   (nev+1,) int64
shower/vertex_offsets   (nev+1,) int64
shower/initiator_offsets(nev+1,) int64
```

`i_src`/`i_tgt` are **row indices into that event's vertex slice**, already offset by shower -- not raw GTL node ids, which restart at 0 in every shower and would collide the moment an event's showers are concatenated. The raw id is kept in the vertex rows for traceability. `PairBrowser.showers(event)` returns the two per-event slices together, so the indices stay valid against each other; slicing one without the other is the way to get this wrong.

#### `ShowerRecord(partons, vertices, initiators)`

One event's showers, flattened and ready to write or draw.

#### `showers_from_manager(mgr)`

Flatten every finished `PartonShower` on a `JetEnergyLossManager` into one record.

Must be called while the showers are alive -- after the manager's Exec and before ClearPerEvent. `DropletBridge.Exec()` is exactly that window, which is why the capture lives there and not in the driver.

Takes any object with `get_showers()` / `get_shower_initiating_partons()`, so it needs no `jetscape` import and stays testable with a stub.

#### `initiators_from_manager(mgr)`

-> (K, 11) float64, one row per shower-initiating parton, `INITIATOR_COLUMNS`.

The rows `showers_from_manager` stores as `shower/initiators`; the particlize writer stores the same rows, so the two files agree by construction. Same window as `showers_from_manager`: after the manager's Exec, before ClearPerEvent.

#### `velocities(partons)`

-> (N, 3) `p/E`. Zero-energy rows (the `miss` placeholders) come back as zero.

#### `first_child(partons, vertices)`

-> (N,) the row index of each parton's first child, or -1 where it has none.

A parton's children are the partons leaving its target vertex. Two cases have none, and both matter:

- **final-state partons**, whose target vertex is a leaf. The framework gives that vertex a time equal to the parton's own production time, so the graph holds no end time at all -- a final parton simply keeps going.
- **negative ("hole") partons**, which `JetEnergyLoss.cc:413-416` attaches *backwards*: the edge runs `new_vertex -> vStart`, so its target is its own parent's source vertex. The partons found there are its siblings, not its children, and they were produced no later than it was. Requiring a strictly later production time is what rejects them -- without it a hole would be drawn running back up its parent's track.

#### `split_times(partons, vertices)`

-> (N,) the lab time each parton splits, NaN where it never does.

#### `segments(partons, vertices)`

-> (start, end, splits): the drawable form of every parton.

`start` and `end` are `(N, 4)` as `(x, y, z, t)`; `splits` is a boolean mask. A parton that splits runs from its own production point to **its child's** production point, both of which are stored data -- no extrapolation. (Propagating the parent along `p/E` instead misses by ~1e-3 fm, because `p_in()` is the momentum at production and the parton loses some of it on the way.) One that does not split -- a final-state parton, or a hole -- gets `end == start` and `splits == False`, because the graph does not say when it stops; `PairBrowser.shower_at` is what decides how far to carry those at a given frame, and there `p/E` is the only thing to go on.

Vertex *positions* are deliberately not used: the framework stores every one of them at the origin (see the module docstring), so a segment built from them would have zero length -- which reads as a plotting bug rather than a data one.

#### `to_milne(xyzt)`

Cartesian `(x, y, z, t)` -> Milne `(tau, x, y, eta)`, to share the hydro's clock.

Space-like points (`|z| > t`, which the framework does produce for partons created before the hydro start time) have no real `tau`; they come back as NaN rather than as a silently wrong number.

#### `fates(partons)`

-> list of fate names, one per parton (see FATES); unknown codes come back as the int.
