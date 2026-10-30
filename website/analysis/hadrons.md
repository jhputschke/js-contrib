# Hadrons

This page shows how to read the hadron files of a whole campaign as one data set, fill
histograms with the right per-event statistics, pull out single events, weight pT̂ windows by
their cross sections, and spot bad runs. The tool for almost all of it is `HadronFileReader`
from `jetscape.hadrons_h5`; lower-level readers and the Pythonic `HadronArray` are at the end,
for when you need one file, one sample or awkward arrays. It applies to the Au+Au jet
productions (two stages or hadrons-only jobs) and the Z+jet productions; O+O hadron files made in
the job are read differently ([O+O](oo.md)).

The file layout and what an event is: [What a production writes](files.md#4-hadron-files).

## 1. Open a campaign

```python
import numpy as np
from jetscape.hadrons_h5 import HadronFileReader

r = HadronFileReader("/path/to/campaign")       # or: with HadronFileReader(...) as r:
print(r.n_files, "files,", r.n_events, "events, tags", r.tags())
```

`source` can be:

- **a directory:** every production file in it, i.e. every `*_particlize.h5`, and every
  hadrons-only job (its `*_hadrons_jet_frag.h5`). Subdirectories are **not** searched.
- **a glob pattern** (quoted, of particlize or hadron files): the way to read a campaign spread
  over subdirectories, e.g. the per-GPU output of a multi-GPU run:
  `HadronFileReader("/data/zjet_c00/gpu*/*_hadrons_jet_frag.h5")`.
- **a list** of stems or particlize paths.

Events are numbered globally, in the sorted order of the files: `r.locate(g)` gives
(file index, local event), `r.global_event(file, local)` the reverse, and `r.stems` lists the
files.

**What it checks on open:**

- a hadron file whose recorded `source_uuid` is not its particlize file's `file_uuid` (renamed,
  or from another run) is refused (`check_uuid=False` overrides);
- mixed precisions, η cuts, charged-only or no-position files give warnings: results are then
  right only for what all files hold;
- a background that appears in two production files (two campaigns over the same seeds) gives a
  warning; `r.duplicate_backgrounds()` lists them.

## 2. Histograms, totals and the wake

`hist`, `total` and `jet_minus_background` do the statistics for you. For every event, the
samples of its unit are histogrammed and divided by that unit's number of samples; these
per-event means are then averaged over events. So every event counts the same, even when files
were hadronized with different `--oversample`, and the errors are per event (compound-Poisson
per event, added over events).

```python
def lead(info):
    """phi and pseudorapidity of the event's hardest shower-initiating parton."""
    ini = info.initiators()                     # (K, 11): shower, pid, pstat, px, py, pz, E, ...
    px, py, pz = ini[np.argmax(np.hypot(ini[:, 3], ini[:, 4])), 3:6]
    return np.arctan2(py, px), np.arcsinh(pz / np.hypot(px, py))

def dphi(ev, info):                             # azimuth relative to the leading parton
    return np.mod(ev.phi - lead(info)[0] + np.pi / 2, 2 * np.pi) - np.pi / 2

def soft_charged(ev, info):
    return ev.charged & (ev.pt > 0.2) & (ev.pt < 4) & (np.abs(ev.eta) < 1)

with HadronFileReader("/path/to/campaign") as r:
    # names of EventHadrons attributes: per-event sample means, averaged over events
    pt, pt_err = r.hist("bulk_bg", "pt", np.linspace(0, 3, 31), mask="charged")
    n_mid, n_err = r.total("bulk_jet", mask=lambda ev, info: ev.charged & (np.abs(ev.eta) < 1))

    # the wake: <jet leg> - <its own background>, event by event
    bins = np.linspace(-np.pi / 2, 3 * np.pi / 2, 17)
    wake, wake_err = r.jet_minus_background(dphi, bins, mask=soft_charged)
    full, _ = r.jet_minus_background(dphi, bins, mask=soft_charged, fragments=True)  # + jet_frag
```

- **`values`, `mask`, `weights`** are either names (an attribute of `EventHadrons`: `pid`,
  `pstat`, `p`, `x`, `E`, `px`, `py`, `pz`, `pt`, `eta`, `y`, `phi`, `charged`, `sample`; for
  `mask` also `"charged"` or a species, `"pi+"`, `"K-"`, `"p"`, `"pbar"`, `"pi"`, `"K"`,
  `"p+pbar"`), or callables `f(ev, info)` returning one value per hadron. A tuple of names (or a
  callable returning a tuple) gives an N-dimensional histogram.
- **Callables get the event's `info`** (`EventInfo`: `event`, `stem`, `local_event`, `bg_unit`,
  `seed`, `pthat_bin`, `pthat`, `n_coll`, `z`, `initiators()`), which jet-relative observables
  need: above, Δφ to the event's leading initiator.
- **Names are faster.** With names only, a background shared by several events is binned once
  and counted for each of them. For a callable that does not use `info`, pass
  `per_event=False` for the same single pass; the result is identical.
- **`events=`** restricts any of them to a subset, e.g. one pT̂ window.
- **`jet_minus_background`** uses only events whose jet leg *and* background have samples, each
  against its own background. `fragments=True` adds the jet's own hadrons.

!!! note "Which error"
    For files hadronized with `--correlated`, `jet_minus_background` takes its error from the
    spread of the per-sample differences (`paired=True`, chosen automatically:
    `r.correlated`), the only right error when the legs share hadrons. The default
    compound-Poisson error treats every hadron as independent; resonance daughters in one bin make
    it ~20 % too small for soft multiplicities, which `paired=True` also avoids for independent
    legs with equal sample counts.

## 3. A worked example: multiplicity and spectra

The charged dN/dη of the background and of the jet events, the identified spectra at
\(|y| < 0.5\), and the wake's charged multiplicity, for one campaign:

```python
import numpy as np
from jetscape.hadrons_h5 import HadronFileReader

eta_e = np.linspace(-5, 5, 41)
pt_e = np.linspace(0, 3, 31)
with HadronFileReader("/path/to/campaign") as r:
    # dN_ch/deta, per event (each event: the mean over its oversamples)
    bg, bg_err = r.hist("bulk_bg", "eta", eta_e, mask="charged")
    jet, jet_err = r.hist("bulk_jet", "eta", eta_e, mask="charged")
    dndeta_bg = bg / np.diff(eta_e)

    # identified dN/dpT at |y| < 0.5 (width 1 in y)
    spectra = {}
    for s in ("pi+", "K+", "p"):
        mid_y = lambda ev, info, s=s: ev.species(s) & (np.abs(ev.y) < 0.5)
        h, e = r.hist("bulk_bg", "pt", pt_e, mask=mid_y, per_event=False)
        spectra[s] = (h / np.diff(pt_e), e / np.diff(pt_e))

    # the wake's charged hadrons at |eta| < 1, and the jet's own
    mid = lambda ev, info: ev.charged & (np.abs(ev.eta) < 1)
    wake, wake_err = r.jet_minus_background(None, None, mask=mid, per_event=False)
    n_frag, n_frag_err = r.total("jet_frag", mask=mid, per_event=False)
print("dN_ch/deta at eta = 0:", dndeta_bg[len(dndeta_bg) // 2])
print("wake N_ch(|eta| < 1):", wake[0], "+-", wake_err[0], "| fragments:", n_frag)
```

With `values=None, bins=None`, `jet_minus_background` returns the difference of the totals as a
one-bin array. For a 2D map in the jet frame, pass a callable returning a tuple, e.g.
`lambda ev, info: (ev.eta - lead(info)[1], dphi(ev, info))`, and a tuple of bin edges.

## 4. Single events

```python
with HadronFileReader("/path/to/campaign") as r:
    ev = r.jet_event(0, 17)          # global event 0, oversample 17 + its fragmentation
    ev["pid"], ev["p"], ev["x"]      # p = (E, px, py, pz); ev["origin"]: 0 bulk, 1 fragment
    bg = r.background_event(0, 17)   # oversample 17 of the background event 0 used
    info = r.event_info(0)           # stem, local_event, bg_unit, seed, pthat_bin, initiators()
    n, m = r.n_oversamples(0), r.n_frag(0)
    for ev in r.iter_jet_events(0):  # every oversample of event 0
        ...
```

`jet_event(e, k)` pairs oversample k with fragmentation k mod `n_frag(e)`; `frag_sample=` picks
another, `fragments=False` leaves them out. These are the inputs for jet finding
([Jets with FastJet](jets.md)).

## 5. pT̂ windows and cross sections

Campaigns with several pT̂ windows per background (`--pthat-bins`) have one Pythia, and one
cross section, per window. Each window's histogram is a per-event average, so the windows can
have different event counts; multiply each by its cross section to stitch them:

```python
with HadronFileReader("/path/to/campaign") as r:
    r.pthat_bins                                   # (K, 2): the windows
    edges = np.linspace(0, 40, 41)
    spec, var = np.zeros(40), np.zeros(40)
    for k in range(len(r.pthat_bins)):
        h, e = r.hist("jet_frag", "pt", edges, mask="charged", events=r.pthat_bin_events(k))
        sigma, _ = r.pthat_bin_sigma(k)            # mb, after the parton-y cut
        spec += sigma * h                          # each window: per-event mean x its sigma
        var += (sigma * e) ** 2                    # jet_frag: windows independent
```

`spec` is then dσ/dpT per bin [mb]. `pthat_bin_sigma(k)` combines the files' Pythia estimates,
weighted by their accepted events; `pthat_bin_acceptance(k)` is the parton-rapidity cut's
acceptance. The windows share their backgrounds, so for the bulk tags the errors of different
windows are correlated: the simple sum of variances above is right for `jet_frag` only.

## 6. Bad runs

Two rare failures show up in the data, not in the logs. Check for them before an analysis:

- **A leg that never froze out.** Its background has ~3 times the freeze-out cells and ~11 times
  the hadrons of its jet leg. Compare `n_cells_bg / n_cells_jet` (particlize file's `events/`):
  `wake_hadrons.py`, `hadron_distributions.C` and the C++ `HadronFileReader.h` drop events where
  it is off by more than 20 %. In the job logs, `grep -l "Maximum allowed time reached"
  DIR/*.log` finds such jobs.
- **A job whose jets lost almost no energy** (once in ~250 campaign jobs): its droplets per event
  fall far below the campaign's. `utils/check_deposition.py` reads every job summary and flags
  it (exit code 1):

```bash
python utils/check_deposition.py /path/to/campaign              # every <stem>.json below it
python utils/check_deposition.py /data/c1/gpu0 /data/c1/gpu1    # one campaign, two directories
```

Move a flagged job's files away and rerun the campaign command, which redoes the job.

## 7. Lower-level readers

For one file, one sample, or a whole small file in memory:

```python
from jetscape.hadrons_h5 import HadronFile, JetEvents, Hadrons

with HadronFile("<stem>_hadrons_bulk_jet.h5") as hf:       # random access, nothing loaded
    one = hf.sample_event(0, 17)                            # unit (event) 0, sample 17: pid, pstat, p, x
    ini = hf.initiators(0)                                  # the event's shower initiators
with JetEvents.from_stem("<stem>") as je:                   # the three files + particlize of one job
    ev = je.jet_event(0, 17)                                # bulk_jet sample 17 + a fragmentation

h = Hadrons.from_h5("<stem>_hadrons_bulk_bg.h5")            # a whole file in memory
dn, err = h.hist(h.eta, np.linspace(-1, 1, 5), mask=h.charged)   # per sample
```

`Hadrons` carries `pt`, `eta`, `y`, `phi`, `charged`, `sample`, `unit` as arrays (in float64,
about 2.4 times the file's memory) and `hist` / `total` averaged per sample. `units=` reads only
some units.

### The Pythonic `HadronArray`

`jetscape.HadronArray` holds a hadron file as flat arrays with sample offsets, without derived
columns, and turns it into awkward arrays of `vector` four-momenta on demand, for selections and
FastJet's array interface:

```python
import jetscape as js

bulk = js.HadronArray.from_h5("<stem>_hadrons_bulk_bg.h5")
had = bulk.to_awkward(by_unit=True)            # units x samples x hadrons, Momentum4D records
charged = had[js.select(had, "charged")]
mid = abs(charged.eta) < 0.5
```

It needs no X-SCAPE build and also reads file objects (e.g. a file streamed from the OSDF).
More in [The Pythonic interface](../python/pythonic.md).

## 8. Scripts and notebooks for hadrons

| | what |
|---|---|
| [`analysis/tutorials/hadron_distributions.ipynb`](https://github.com/jhputschke/js-contrib/blob/main/contribs/PyJetscape/example/analysis/tutorials/hadron_distributions.ipynb) | η, φ, pT of background, background + wake, fragments and the wake, in the lab and relative to the jet; checks the correlated-sampling flags (`HADRON_DIR`, `MAX_EVENTS`, `PAIRED`) |
| [`analysis/wake_hadrons.py`](https://github.com/jhputschke/js-contrib/blob/main/contribs/PyJetscape/example/analysis/wake_hadrons.py) + `wake_hadrons.ipynb` | one pass over a campaign: the wake per event in the jet frame, ΔR profiles, identified spectra, totals → `wake_hadrons.h5`; then the figures. See [Hydro and hadrons together](hydro-hadrons.md#4-the-wake-at-hadron-level) |

**Threads and memory.** `HadronFileReader` is one Python process, reading one production file
at a time and only the units it needs: the four events of two seed files with 500 and 100
oversamples (~11 M hadrons) take 2.7 s. A background of 2000 oversamples is ~14 M hadrons,
~1.5 GB with the derived kinematics. For more speed, split a campaign by `events=` over
processes.

**Reading in place from the OSDF.** `HadronFileReader` takes local paths only; download the
campaign first ([Data on the OSDF](../run/osdf.md)). Single files can be read in place with
h5py or `HadronArray` through `fsspec`.

## Next steps

- Cluster the hadrons into jets: [Jets with FastJet](jets.md).
- The same events in the fluid: [Hydro and hadrons together](hydro-hadrons.md).
- The same in ROOT: [ROOT and C++](root.md).
