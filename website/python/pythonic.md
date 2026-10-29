# The Pythonic interface

The Pythonic interface is the layer of PyJetscape you will use most. It runs X-SCAPE events in a
Python loop and hands you each event's partons, hadrons and showers as numpy record arrays,
awkward arrays with four-vector kinematics, or Python objects, ready for numpy, FastJet and
histogramming. Use it for any study that reads what the modules produce: a script of your own, a
notebook, or a small production. It is installed by `import jetscape` and only adds names to the
bound C++ classes, so scripts written against the bare bindings keep working and keep their
output. Where to get it: [Python](index.md#where-you-get-it).

## Quick start

```python
import fastjet
import jetscape as js

with js.per_event("jetscape_main.xml", "my_user.xml") as driver:
    for i in driver.events():
        partons = driver.energy_loss_manager.final_parton_array()   # numpy records
        jet = driver.hadronization_manager.hadron_array()          # HadronArray
        h = jet.to_awkward()                                        # samples x hadrons
        charged = h[js.select(h, "charged") & (abs(h.eta) < 2)]
        jets = fastjet.ClusterSequence(charged, fastjet.JetDefinition(
            fastjet.antikt_algorithm, 0.4)).inclusive_jets(min_pt=5.0)
        print(i, len(partons), "partons;", jets.pt.tolist())
```

The complete worked example is
[`example/pythonic/oo_workflow.ipynb`](https://github.com/jhputschke/js-contrib/blob/main/contribs/PyJetscape/example/pythonic/oo_workflow.ipynb):
three O+O events through the whole chain, the hadrons written to files and read back, spectra
with errors, FastJet. Its simulation half is the script `oo_workflow.py`
(`--xml`, `--events`, `--seed`, `--outdir`), a template for a production of your own.

**Requirements.** The bound classes need the compiled `pyjetscape_core`. Records and
`HadronArray` need only numpy (they also work on files without a build); `to_awkward()` needs
`awkward` and `vector`, `select()` needs `particle`, jets need the `fastjet` Python package, and
the hadron files need `h5py`. All are in the container image, the Mac wheel's environment and the
analysis environment; awkward and vector are imported only when used.

## Conventions

- **Names.** A method ending in `_array()` returns numpy data for the current event
  (`final_parton_array()`, `hadron_array()`, `edge_array()`). A plain noun returns objects
  (`showers`, `nodes`). Properties (`info`, `showers`, `momentum`, `last_random_seed`) take no
  parentheses.
- **Four-vectors.** Momentum is `(E, px, py, pz)`, position `(t, x, y, z)`, everywhere: records,
  `HadronArray.p` / `.x`, the hadron files, `FourVector.comp(i)`. Two older calls keep their own
  order: the bound constructor `FourVector(x, y, z, t)` (for a momentum `FourVector(px, py, pz, E)`)
  and `FourVector.to_list()`.
- **Units.** GeV and fm (fm/c), as in X-SCAPE.
- **Flat first, nested on request.** Every `_array()` is flat numpy plus offsets, the form the
  files use; `to_awkward()` gives the nested view.

## Running events

```python
with js.per_event(main_xml, user_xml) as driver:     # Init() and ExecInit() on entry
    for i in driver.events():                         # ExecPerEvent() ... ClearPerEvent()
        ...                                           # this event's results are in memory
                                                      # Finish() on leaving the block
```

`js.per_event(main_xml=None, user_xml=None, tasks=None, n_events=None)` returns a
`JetScapePerEvent` driver:

- `tasks=[...]` adds a pipeline built by hand, in order; without it the XML's task list is used.
- `n_events` overrides `<nEvents>`.
- `driver.events(n=None, on_error=None)` yields the event index after the event ran and before it
  is cleared. `n` caps the count; `break` clears the current event. With
  `on_error(i, exc)`, an event in which a module throws (for example Matter's rare
  "minus log of sudakov negative") is reported, cleared and skipped, as the productions do;
  without it the exception propagates.

The run's modules are driver properties (`None` when absent):

| property | what |
|---|---|
| `energy_loss_manager` | the final-state `JetEnergyLossManager` (in ISR runs, not the ISR chain's) |
| `hadronization_manager` | the `HadronizationManager` |
| `soft_particlization` | the first `SoftParticlization` (iSS) |
| `hydro` | the hydro module |
| `initial_state` | the initial state |

Anything else: `driver.GetTaskList()` and, for sub-tasks, `task.GetTaskList()`.

**Run in a working directory of its own.** MUSIC, 3dMCGlauber and iSS read inputs from and write
side files to the current directory, and X-SCAPE's XMLs contain paths relative to the build.
`js.JobDir` handles that for a script or notebook anywhere on disk:

```python
with js.JobDir("out", "my_job.xml") as job:      # out/my_job.xml; runs in out/work/my_job
    with js.per_event(job.main_xml, job.user_xml) as driver:
        ...                                       # write your outputs to absolute paths
```

It copies the user XML to `out/`, makes its `../` paths absolute, copies `music_input`, links the
build's tables, changes into `out/work/<name>` and back, and removes that directory after a run
without an exception (`keep=True` keeps it). The build is found through `jetscape.paths`:
`$JETSCAPE_BUILD`, else `<X-SCAPE>/build_gpu`, where X-SCAPE is `$XSCAPE` or the tree the package
sits in. In the container `XSCAPE=/opt/X-SCAPE` is set, so scripts run unchanged in and out of it.

## The results of an event

Each method wraps a function of the bindings, and returns numpy copies or arrays taken over from
C++, so the results stay valid after the event is cleared.

| class | member | returns |
|---|---|---|
| `JetEnergyLossManager` | `showers` | list of `PartonShower`, one per shower |
| | `final_parton_array()` | final-parton records (what hadronization receives) |
| | `initiator_array()` | the partons that started each shower |
| `SoftParticlization` (iSS) | `hadron_array()` | `HadronArray`, one sample per oversample |
| | `use_compact_output(on=True)` | from the next event on, hadrons as arrays (about 8x less memory) |
| | `set_number_of_samples(n)`, `set_next_random_seed(seed)`, `last_random_seed` | oversamples and seeds |
| | `primary_array()` | the undecayed hadrons (iSS `<keep_primaries>`), or `None` |
| `HadronizationManager`, `Hadronization` | `hadron_array()` | `HadronArray` of the jet hadrons |
| | `hadronize(partons, seed=None)` | hadronize stored partons again |
| | `last_random_seed` | the jet hadronization's last seed |
| | `colour_audit()`, `colour_totals()`, `set_colour_closure(mode)` | iColoredHadronization's colour bookkeeping, on builds that have it (`hasattr(had, "colour_audit")`) |
| `HardProcess` | `parton_array()`, `info` | hard partons, hardest first; dict with `pthat`, `sigma_gen`, `sigma_err`, `event_weight` |
| `InitialState` | `info` | dict with `n_coll`, `n_part`, `centrality` (what the module reports) |

!!! note "Compact output is taken once"
    After `use_compact_output()`, `hadron_array()` takes iSS's arrays over without a copy, so call
    it **once per event**. Without compact output it copies and can be called again.

## Record arrays

The small per-event tables are numpy structured arrays (`jetscape.records`): `rec["px"]` instead
of `rec[:, 4]`. Each holds exactly the values of the older column array it reorders.

| dtype | from | fields |
|---|---|---|
| `FINAL_PARTON_DTYPE` | `final_parton_array()` | `shower, pid, pstat, E, px, py, pz, t, x, y, z, mass, col, acol` |
| `INITIATOR_DTYPE` | `initiator_array()` | `shower, pid, pstat, E, px, py, pz, t, x, y, z` |
| `EDGE_DTYPE` | `PartonShower.edge_array()` | `src, tgt, pid, pstat, E, px, py, pz, t, x, y, z` |
| `NODE_DTYPE` | `PartonShower.node_array()` | `id, t, x, y, z` |
| `HARD_PARTON_DTYPE` | `HardProcess.parton_array()` | `pid, E, px, py, pz, pt, y` |

Records work with the usual tools: `rec[rec["pstat"] >= 0]`, `np.unique(rec["shower"])`,
`np.hypot(rec["px"], rec["py"])`, `pandas.DataFrame(rec)`. Converters for arrays from files:
`final_parton_records(rows)`, `final_parton_rows(rec)`, `initiator_rows(rec)`, ... in
`jetscape.records`.

## HadronArray

One event's hadrons (or a whole file's), **one array per field**, grouped into samples: one per
iSS oversample, one for a jet hadronization.

| attribute | | |
|---|---|---|
| `pid`, `pstat` | (n,) int32 | |
| `p` | (n, 4) float32 | E, px, py, pz |
| `x` | (n, 4) float32 or `None` | t, x, y, z |
| `mass` | (n,) float32 or `None` | `None` when read from a file |
| `sample_offsets` | (n_samples + 1,) int64 | sample `s` is rows `sample_offsets[s]:sample_offsets[s+1]` |
| `info` | dict | e.g. the seed; after `from_h5`: `units`, `unit_offsets`, `attrs`, `path` |

`len(h)`, `h.n_samples`, `h.sample_counts`; `h.to_dict()` gives the dict the hadron writer takes;
`h.to_awkward(...)` the nested view; `h.to_hadrons()` a `hadrons_h5.Hadrons` for its histogram
helpers.

**Memory.** `HadronArray` wraps the arrays the bindings return without a copy and stores no
derived columns: pt, eta, phi and rapidity are computed by `to_awkward()` when you ask for them.
`to_hadrons()` computes them all in float64 (about 2.4x the memory); use it for histograms, not to
hold many events.

## Nested views: `to_awkward`

`js.to_awkward(obj, position=False, charge=False, by_unit=False)`, also
`HadronArray.to_awkward(...)`:

| input | result |
|---|---|
| `HadronArray` | samples x hadrons, on the same buffers |
| `HadronArray.from_h5(...)` with `by_unit=True` | units (events) x samples x hadrons |
| final-parton or initiator records | showers x partons |
| edge, node, hard-parton records | a flat array of records |

The records are `vector` `Momentum4D` objects, so kinematics come on demand:

```python
import awkward as ak

h = driver.soft_particlization.hadron_array().to_awkward()
h.pt, h.eta, h.phi, h.rapidity, h.mass          # jagged, the shape of h
mid = h[abs(h.eta) < 0.5]
dndeta = ak.mean(ak.num(mid), axis=0)            # mean over the oversamples
```

- `position=True` adds a nested `h.position` (`Vector4D`: `h.position.t`, `h.position.rho`).
  It is nested on purpose: `vector` would read flat fields named t, x, y, z as the coordinates
  of the momentum.
- `charge=True` adds the PDG charge.
- A stored mass appears as the field `rest_mass`; `h.mass` is `vector`'s, computed from E and p.
- `ak.flatten(a, axis=1)` merges units x samples into samples; `ak.num(a, axis=k)` counts.

## Selecting species: `select`

```python
js.select(h, "pi+")                          # one PDG name
js.select(h, ["K+", "K-"])                   # any of several
js.select(h, "charged")                      # a class
js.select(h, ["p", "p~", "Lambda", "Lambda~"])
```

The result is a boolean mask of the input's shape (jagged for awkward input, so
`h[js.select(h, "charged")]`). Classes: `charged`, `neutral`, `hadron`, `meson`, `baryon`,
`strange`; names as the `particle` package spells them (`pi0`, `K(S)0`, `Sigma+`, `Omega-`, ...).
Each distinct pid is looked up once.

!!! note "`"charged"` and `hadrons_h5.CHARGED`"
    `select(..., "charged")` takes every pid with a nonzero PDG charge, resonances included.
    `hadrons_h5.CHARGED`, used by `Hadrons.charged` and the analysis scripts, lists the stable
    charged hadrons and leptons. They agree on decayed hadrons and differ only on files that keep
    undecayed primaries.

## Hadron files

The productions' hadron files (`jetscape.hadrons_h5`) take a `HadronArray` per unit (event) and
give one back:

```python
from jetscape.hadrons_h5 import HadronH5Writer
from jetscape.records import initiator_rows

with HadronH5Writer("x_hadrons_bulk_bg.h5", tag="bulk_bg", n_samples=20) as bulk_w, \
     HadronH5Writer("x_hadrons_jet_frag.h5", tag="jet_frag", n_samples=1,
                    initiators=True) as jet_w:
    for i in driver.events():
        iss, had = driver.soft_particlization, driver.hadronization_manager
        bulk_w.append_unit(iss.hadron_array().to_dict(), unit=i, event=i,
                           seed=iss.last_random_seed)
        jet_w.append_unit(had.hadron_array().to_dict(), unit=i, event=i,
                          initiators=initiator_rows(driver.energy_loss_manager.initiator_array()))
```

Keywords besides the hadrons (`event=`, `seed=`, `pthat=`, ...) become the file's per-unit table.
Reading:

```python
h = js.HadronArray.from_h5("x_hadrons_bulk_bg.h5")          # every unit, stored fields only
h.info["units"]["seed"]                                      # the per-unit table
a = h.to_awkward(by_unit=True)                               # units x samples x hadrons
first = js.HadronArray.from_h5("x_hadrons_bulk_bg.h5", units=[0])
```

For whole campaigns, with event-averaged histograms and compound-Poisson errors, use
`hadrons_h5.HadronFileReader` and `Hadrons`: [Hadrons](../analysis/hadrons.md).

## Jets with FastJet

FastJet's Python package clusters jagged awkward arrays directly: one call for every event, or
every oversample.

```python
import awkward as ak
import fastjet

jdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)

# the jets of every event (a jet file: events x hadrons)
h = js.HadronArray.from_h5(jet_path).to_awkward()
acc = h[abs(h.eta) < 2]
jets = fastjet.ClusterSequence(acc, jdef).inclusive_jets(min_pt=5.0)   # events x jets
leading = ak.firsts(jets[ak.argsort(jets.pt, ascending=False)])

# charged jets
ch = acc[js.select(acc, "charged")]
ch_jets = fastjet.ClusterSequence(ch, jdef).inclusive_jets(min_pt=3.0)

# every oversample of every event (bulk: units x samples x hadrons)
b = js.HadronArray.from_h5(bulk_path).to_awkward(by_unit=True)
per_sample = ak.flatten(b, axis=1)                                     # samples x hadrons
bulk_jets = fastjet.ClusterSequence(per_sample[abs(per_sample.eta) < 2],
                                    jdef).inclusive_jets(min_pt=5.0)
bulk_jets = ak.unflatten(bulk_jets, ak.num(b, axis=1))                 # units x samples x jets
```

The jagged call gives the same jets as a loop over the samples. Jets are `vector` records too
(`.pt`, `.eta`, `.phi`, `.m`); `constituent_index()` maps them back to hadrons. Areas,
background subtraction and embedding are on [Jets with FastJet](../analysis/jets.md).

## Particles and four-vectors

`FourVector` iterates, indexes and converts in `comp()` order (t, x, y, z) = (E, px, py, pz),
compares by value, and pickles:

```python
import numpy as np
from jetscape import pyjetscape_core as core

v = core.FourVector.momentum(10.0, 1.0, 2.0, 3.0)    # E, px, py, pz
E, px, py, pz = v
np.asarray(v)                                         # array([10., 1., 2., 3.])
v == core.FourVector(1, 2, 3, 10)                     # True (the constructor is x, y, z, t)
```

Because `==` compares values, a `FourVector` is not hashable; use `tuple(v)` as a key.

Particles (`Parton`, `Photon`, `Hadron`) gain `momentum` and `position` (FourVector copies) and
keyword factories that take any 4-sequence:

```python
p = js.make_parton(21, momentum=(50, 50, 0, 0), position=(0.5, 0, 0, 0),
                   stat=0, label=0, color=101, anti_color=102)
g = js.make_photon(momentum=(5, 0, 0, 5))
h = js.make_hadron(211, momentum=p.momentum, position=p.position, stat=1)   # PDG mass
E, px, py, pz = p.momentum
t, x, y, z = p.position
```

The bound methods stay methods (`p.pt()`, `p.eta()`). For properties, wrap a particle:
`from jetscape.particles import view; view(p).pt`. Each is one call into C++ (about 1.4 µs), so
for many particles use the arrays.

!!! warning "A particle's `t()` is its time"
    `x()`, `y()`, `z()`, `t()` of a particle are its position. A parton's virtuality is
    `virtuality()` (C++ `Parton::t()`).

## Parton showers

```python
for shower in driver.energy_loss_manager.showers:
    for parton in shower:                              # the partons (graph edges)
        end = shower.end_node(parton)
        if end.is_one_to_two:
            print(parton.pid(), "splits at t =", end.position.t(),
                  "z =", end.split_z, "kt =", end.split_kt, "dR =", end.split_delta_r)
    edges = shower.edge_array()                        # EDGE_DTYPE records
    nodes = shower.node_array()                        # NODE_DTYPE records
```

`PartonShower` iterates over its partons and has `nodes`, `start_node(p)`, `end_node(p)`,
`edge_array()` and `node_array()`; it has no `len()`, so an empty shower is still true in
`if shower:`. A `ShowerNode` has `id`, `vertex`, `position`, `parents`, `children`, `split_z`,
`split_kt`, `split_delta_r`, `is_one_to_two` and `is_end`. Showers and partons are C++ objects of
the event: use them inside the loop. The record arrays are copies and outlive it.

## Cost and memory

Measured on a GB10:

| | |
|---|---|
| iSS, 972,215 cells, 1000 oversamples (7.0M hadrons), resident memory after the step / peak | dict from the bindings 686 / 916 MB; `HadronArray` 686 / 915 MB (no copy); `to_awkward()` + `.pt` 761 / 915 MB; `Hadrons` 1241 / 1240 MB |
| a Python energy-loss module, 715,438 calls (500 events) | 3.73 s against 3.70 s before the Pythonic layer (+0.9 %) |
| one bound method call (`p.pt()`, `view(p).pt`) | about 1.4 µs |

## Pitfalls

- **Read results inside the event loop.** Call the `_array()` methods inside
  `for i in driver.events():`. Their numpy results outlive the event; C++ objects (showers,
  partons) do not.
- **No surface, no iSS hadrons.** An event without a medium (O+O with no string) has no
  freeze-out surface; check `len(driver.hydro.surface_to_numpy())` before `hadron_array()`, as
  `oo_workflow.py` does.
- **Field names and `vector`.** Do not add fields named t, x, y, z, `tau`, `rho`, `phi`, `eta`,
  ... to a Momentum4D record; `vector` reads them as coordinates. Use nested records.
- **Bound classes are in `pyjetscape_core`.** `import jetscape as js` re-exports only some of
  them (`js.JetScapePerEvent`, `js.JetScapeSignalManager`, `js.FluidDynamics`, ...). For the
  others write `from jetscape import pyjetscape_core as core` and `core.FourVector`,
  `core.JetEnergyLoss`, `core.Parton`.
- **ISR runs** have two energy-loss managers; `driver.energy_loss_manager` is the final-state one.
- **Import `torch` before `jetscape`** when you use PyTorch (see [Python](index.md)).

The full guide, with every member, the guarantees of the layer and a side-by-side table of the
old and new calls:
[`contribs/PyJetscape/PYTHONIC.md`](https://github.com/jhputschke/js-contrib/blob/main/contribs/PyJetscape/PYTHONIC.md).

## Next steps

- [Writing modules in Python](modules.md): the same layer for modules of your own.
- [Hadrons](../analysis/hadrons.md) and [Jets with FastJet](../analysis/jets.md): analysing
  production files.
- [Notebooks](../analysis/notebooks.md): the example notebooks.
