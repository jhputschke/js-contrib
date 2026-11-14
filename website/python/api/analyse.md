# Analyse the files

Read a production: hadron histograms with per-event errors, the wake, single events for jet
finding, the freeze-out surfaces and the hydro pair. All of this needs only the
[analysis environment](../../install/analysis-env.md), no X-SCAPE build. The guides are
[Hadrons](../../analysis/hadrons.md), [Jets with FastJet](../../analysis/jets.md) and
[Hydro and hadrons together](../../analysis/hydro-hadrons.md); the file layouts are on
[What a production writes](../../analysis/files.md).

## Examples

### Open a campaign and histogram it

`HadronFileReader` reads all files of a campaign as one data set: a directory (every
`*_particlize.h5` with its hadron files), a quoted glob, or a list of stems. `hist` and `total`
average each event's samples first and then the events, so every event counts the same and the
errors are per event.

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

- **`values`, `mask`, `weights`** are names (`pt`, `eta`, `y`, `phi`, `E`, `pid`, `charged`,
  a species such as `"pi+"`, `"K"`, `"p+pbar"`) or callables `f(ev, info)` with one value per
  hadron. A tuple of names gives an N-dimensional histogram.

### The tags

The first argument of `hist` and `total` names one of the hadron files of each production file,
`<stem>_hadrons_<tag>.h5`. The four tags are `jetscape.hadrons_h5.TAGS`:

| tag | hadrons | one unit is | also used by |
|---|---|---|---|
| `bulk_jet` | iSS on the jet leg's surface: the medium with the wake | an event | `jet_event`, `jet_minus_background` |
| `bulk_bg` | iSS on the background's surface: the medium alone | a background, shared by the events that reused it | `background_event`, `jet_minus_background` |
| `jet_frag` | the jet's own hadrons, the fragmentation of its surviving partons | an event | `jet_event`, `jet_minus_background(..., fragments=True)` |
| `extra_frag` | fragments of the event's other hard scatterings; only in campaigns with [extra hard scatterings](../../run/extra-hard.md) | an event | as `jet_frag`; `extras=False` leaves them out |

`r.tags()` gives the tags every file of the campaign has (`r.tags(i)`: file `i`); a call that
needs a missing tag raises. The undecayed iSS hadrons of `--keep-primaries` (tags
`bulk_jet_primary`, `bulk_bg_primary`, `PRIMARY_TAGS`) are not read by `HadronFileReader`:
open those files with `HadronFile`, `Hadrons.from_h5` or `js.HadronArray.from_h5`
([below](#one-file-or-arrays-for-awkward)). The files' layout is on
[What a production writes](../../analysis/files.md#4-hadron-files).

### The wake: jet leg minus its own background

Callables get the event's `info`, which jet-relative observables need. Here Δφ to the event's
leading shower initiator:

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

For files hadronized with `--correlated` the error comes from the per-sample differences,
chosen automatically (`r.correlated`). Which error to report: [Hadrons](../../analysis/hadrons.md).

### Single events

For jet finding or anything event by event. An event is one oversample of the bulk plus one
fragmentation of the jet's partons:

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

`r.n_oversamples(e)` and `r.n_frag(e)` give the counts; oversample k is paired with
fragmentation k mod `n_frag(e)`.

### A subset of events

`events=` restricts any histogram, e.g. to a range of the hard scale:

```python
with HadronFileReader("/path/to/campaign") as r:
    hard = [g for g in r.events() if r.event_info(g).pthat > 20]
    h, e = r.hist("jet_frag", "pt", np.linspace(0, 20, 21), mask="charged", events=hard)
```

Campaigns with several pT̂ windows (`--pthat-bins`) are stitched with each window's cross
section: `r.pthat_bin_events(k)`, `r.pthat_bin_sigma(k)`, as on
[Hadrons](../../analysis/hadrons.md#5-pt-windows-and-cross-sections).

### Jets with FastJet

One sample of an event as charged PseudoJets, clustered with anti-kT. This is the full event,
medium included, without background subtraction; areas and subtraction are on
[Jets with FastJet](../../analysis/jets.md).

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

`HadronArray` turns a file into awkward arrays of `vector` four-momenta, for selections with
awkward and FastJet's array interface:

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

The jet leg `arr` and its background `arr_bg`, shape `(events, 4, nx, ny, neta, ntau)` with the
channels energy density, vx, vy, vz. Read frame by frame, and compare only frames both legs have:

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

::: jetscape.hadrons_h5
    options:
      members: false

::: jetscape.hadrons_h5.HadronFileReader

::: jetscape.hadrons_h5.EventHadrons

::: jetscape.hadrons_h5.EventInfo

::: jetscape.hadrons_h5.HadronFile

::: jetscape.hadrons_h5.Hadrons

::: jetscape.hadrons_h5.JetEvents

::: jetscape.hadrons_h5.pseudorapidity

::: jetscape.hadrons_h5.hadron_precision

::: jetscape.hadrons_h5.hadron_eta_max

::: jetscape.hadrons_h5.hadron_selection

::: jetscape.hadrons_h5.TAGS

::: jetscape.hadrons_h5.PRIMARY_TAGS

::: jetscape.hadrons_h5.FIELDS

::: jetscape.hadrons_h5.ORIGIN

::: jetscape.hadrons_h5.CHARGED

::: jetscape.hadrons_h5.SPECIES

## Particlize file reader

::: jetscape.particlize_h5
    options:
      members: false

::: jetscape.particlize_h5.ParticlizeFile

::: jetscape.particlize_h5.split_by_scatter

::: jetscape.particlize_h5.SURFACE_COLUMNS

::: jetscape.particlize_h5.FINAL_PARTON_COLUMNS

::: jetscape.particlize_h5.SCATTER_COLUMNS

## Arrays and awkward views

::: jetscape.hadron_array

::: jetscape.awkward_views

::: jetscape.records

::: jetscape.particles

## Shower graphs

::: jetscape.showers
