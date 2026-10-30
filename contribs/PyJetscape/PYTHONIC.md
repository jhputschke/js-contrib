# The Pythonic interface of PyJetscape

PyJetscape binds X-SCAPE's C++ classes with their C++ names and argument orders, so a C++
module ports to Python line by line (README, "Writing modules in Python"). The **Pythonic
interface** is a layer on top of that: numpy record arrays instead of column indices,
hadrons that stay one array per field, nested awkward views with `vector` kinematics,
FastJet in one call per event set, an event loop that is a context manager, modules that
declare their XML parameters and return their output.

It is **additive**. Importing `jetscape` installs new methods and properties onto the
bound classes and refuses any name a class already has, so every existing script, module
and file keeps working and keeps its output. You can mix the two styles freely.

Design and validation: js-contrib `docs/Plans/PLAN_pyjetscape_pythonic.md`. A complete
O+O workflow (run, write, read, FastJet): `example/pythonic/oo_workflow.ipynb`.

## Contents

1. [Quick start](#1-quick-start)
2. [Requirements](#2-requirements)
3. [Conventions](#3-conventions)
4. [Running events](#4-running-events)
5. [The results of an event](#5-the-results-of-an-event)
6. [Record arrays](#6-record-arrays)
7. [HadronArray](#7-hadronarray)
8. [Nested views: `to_awkward`](#8-nested-views-to_awkward)
9. [Selecting species: `select`](#9-selecting-species-select)
10. [Hadron files](#10-hadron-files)
11. [Jets with FastJet](#11-jets-with-fastjet)
12. [Particles and four-vectors](#12-particles-and-four-vectors)
13. [Parton showers](#13-parton-showers)
14. [Writing modules](#14-writing-modules)
15. [Type stubs](#15-type-stubs)
16. [What the layer guarantees](#16-what-the-layer-guarantees)
17. [Cost and memory](#17-cost-and-memory)
18. [Pitfalls](#18-pitfalls)
19. [Old and new, side by side](#19-old-and-new-side-by-side)

---

## 1 Quick start

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

`import jetscape` installs the layer. `from jetscape import pyjetscape_core` imports the
package first, so it installs it too: the layer is always there once the bindings are.

## 2 Requirements

| what | needed for |
|---|---|
| PyJetscape built against X-SCAPE (`-DUSE_JS_PYJETSCAPE=ON`; `-DUSE_3DGlauber=ON`, see the README) with pybind11 ≥ 3.0 | everything on the bound classes |
| numpy | records, `HadronArray` (these also work without a build, e.g. on files) |
| `awkward` ≥ 2 and `vector` | `to_awkward()`; imported only when called |
| `particle` (scikit-hep) | `select()`, `to_awkward(charge=True)`; imported only when called |
| `fastjet` (Python package) | jet finding on the awkward views |
| `hist`, `mplhep` | histograms with statistical errors and their plots (the O+O notebook's spectra) |
| `h5py` | `HadronArray.from_h5`, the hadron files |

All are in the `js_fno` environment (`utils/conda_install/install_js_fno_build_minimal.sh`,
`requirements.txt`), and in the production container image (js-contrib
`utils/README_Container_Usage.md`: scripts, notebooks and Jupyter in Docker or Apptainer). Names that depend on the X-SCAPE build (the iColored colour functions,
iSS primaries) are installed only when the build has them (section 5).

## 3 Conventions

**Names.** A method ending in `_array()` returns numpy data, filled once per event
(`final_parton_array()`, `hadron_array()`, `edge_array()`). A plain noun returns objects
(`showers`, `nodes`, `final_partons()`). Properties (`info`, `showers`, `momentum`,
`last_random_seed`, the driver's managers) need no parentheses.

**Four-vectors.** Momentum is `(E, px, py, pz)`, position `(t, x, y, z)`: the order of
`FourVector.comp(i)`, of every record, of `HadronArray.p` / `.x`, and of the hadron files.
Two existing calls keep their own order and are unchanged:

| | order |
|---|---|
| `FourVector(x, y, z, t)` (the bound constructor) | x, y, z, t |
| `FourVector.to_list()` | x, y, z, t |
| `FourVector.comp(i)`, `list(v)`, `v[i]`, `np.asarray(v)` | t, x, y, z |
| `FourVector.momentum(E, px, py, pz)`, `FourVector.position(t, x, y, z)` | as named |
| `p.momentum`, `p.position` of a particle (FourVectors) | unpack as (E, px, py, pz) / (t, x, y, z) |
| records, `HadronArray`, hadron files | E, px, py, pz / t, x, y, z |

**Units.** GeV for momenta and masses, fm and fm/c for positions, as X-SCAPE.

**Flat first, nested on request.** Every `_array()` result is flat numpy plus offsets (or a
`shower` field), the form the files and `hadronize.py` use. `to_awkward()` gives the
nested form (samples x hadrons, showers x partons). There is one nested form, not a second
API.

## 4 Running events

```python
with js.per_event(main_xml, user_xml) as driver:          # Init(), ExecInit()
    for i in driver.events():                              # ExecPerEvent() ... ClearPerEvent()
        ...                                                # this event's results
                                                           # Finish() on leaving the block
```

`js.per_event(main_xml=None, user_xml=None, tasks=None, n_events=None)` makes a
`JetScapePerEvent` (the bound class; its own constructor stays as it is):

- `tasks`: a pipeline built by hand (Mode B), added in order; without it the XML's task list
  is used (Mode A).
- `n_events`: overrides `<nEvents>` (set after `Init()`, which reads the XML).

The driver as a **context manager** runs `Init()`, the event count, and `ExecInit()` on
entry, and `Finish()` on exit, also after an exception.

`driver.events(n=None, on_error=None)` yields the event index once per event, after
`ExecPerEvent()` and before `ClearPerEvent()`: inside the loop body the event's results
are in memory. `n` caps the number of events. Leaving the loop with `break` clears the
current event. `on_error(i, exc)`: when a module throws inside an event (a C++ exception
such as Matter's rare "minus log of sudakov negative"; the same seed hits it again on a
rerun), the callback is called, the event cleared and the loop goes on, as the productions
skip such events; without `on_error` the exception propagates.

**The modules of the run**, as driver properties (`None` when the run has none):

| property | what | found in |
|---|---|---|
| `energy_loss_manager` | the final-state `JetEnergyLossManager` | the signal manager, else the task list; never the ISR chain's `IsrManager` inside the hard process |
| `hadronization_manager` | the `HadronizationManager` | the task list |
| `soft_particlization` | the first `SoftParticlization` (iSS) | the task list |
| `hydro` | the hydro the signal manager holds | the signal manager, else the task list |
| `initial_state` | the initial state | the signal manager, else the task list |

Other modules: `driver.GetTaskList()` (top-level tasks, most derived bound types) and
`task.GetTaskList()` for their sub-tasks, as before.

**Where the run happens: `js.JobDir`.** MUSIC, 3dMCGlauber and iSS read their inputs from,
and write side files to, the current directory, and X-SCAPE's XMLs give some paths relative
to the build (`../external_packages/iSS/...`). The production scripts therefore run every
job in a private working directory; `js.JobDir` (`jetscape.workdir`) does the same for a
script or notebook of one's own, wherever it lives (a copy in a container's `/work`):

```python
with js.JobDir("out", "my_job.xml") as job:     # out/my_job.xml, out/work/my_job (cwd)
    with js.per_event(job.main_xml, job.user_xml) as driver:
        ...                                       # write outputs to absolute paths
```

It copies the user XML to `out/`, makes the `../` paths of both XMLs absolute for the build,
copies `music_input`, links the build's tables, changes into `out/work/<name>` and back on
exit, and removes that directory after a run without an exception (`keep=True` keeps it).
The build is `jetscape.paths.build_dir()`: `$JETSCAPE_BUILD`, else `<X-SCAPE>/build_gpu`,
where the tree is `$XSCAPE` or the one the package is in (`jetscape.paths.xscape_root()`,
`main_xml()`). In the production image `XSCAPE=/opt/X-SCAPE` is set, so scripts run
unchanged in and out of the container: js-contrib `utils/README_Container_Usage.md`.

## 5 The results of an event

Methods on the module classes. Each wraps a function of the bindings, which stays; the
results are numpy copies or arrays taken over from C++, so they stay valid after
`ClearPerEvent()`.

| class | member | returns | wraps |
|---|---|---|---|
| `JetEnergyLossManager` | `showers` | list of `PartonShower`, one per shower | `get_showers()` |
| | `final_parton_array()` | `FINAL_PARTON_DTYPE` records | `final_partons_numpy()` |
| | `initiator_array()` | `INITIATOR_DTYPE` records | `showers.initiators_from_manager` |
| `SoftParticlization` (iSS) | `hadron_array()` | `HadronArray`, one sample per oversample | `soft_hadrons_numpy` |
| | `primary_array()` | `HadronArray` of the undecayed hadrons of the first N oversamples (iSS `<keep_primaries>`), or None | `soft_primaries_numpy` |
| | `use_compact_output(on=True)` | from the next event on, hadrons as arrays (~8x less memory) | `soft_set_compact_output` |
| | `set_number_of_samples(n)` | oversamples from the next event on | `soft_set_number_of_samples` |
| | `set_next_random_seed(seed)` | seed of the next event (one-shot) | `soft_set_next_random_seed` |
| | `last_random_seed` | the seed of the last event | `soft_last_random_seed` |
| | `sample_cells(cells, seed, n_samples)` | `HadronArray` of given (N, 32) surface cells, now | `soft_sample_cells` |
| `HadronizationManager`, `Hadronization` | `hadron_array()` | `HadronArray` (one sample) of the jet hadrons | `hadronization_hadrons_numpy` |
| | `hadronize(partons, seed=None)` | `HadronArray` of stored partons (records or the old (N, 14) array); `info["seed"]`, `info["n_partons_out"]` | `hadronize_partons` |
| | `last_random_seed` | the jet hadronization's last seed | `jet_hadronization_last_random_seed` |
| the same, build with `ICOLORED_HAS_COLOUR_AUDIT` | `colour_audit()` | dict: what iColored received and how it closed the colour lines (holes, partners, joins, ...) | `jet_hadronization_colour_audit` |
| the same, build with `ICOLORED_HAS_COLOUR_CLOSURE` | `colour_totals()`, `set_colour_closure(mode)` | the audits added up over the run; `<colour_closure>` by hand | `jet_hadronization_colour_totals`, `_set_colour_closure` |
| `HardProcess` | `parton_array()` | `HARD_PARTON_DTYPE` records, hardest first | `hard_partons_numpy` |
| | `info` | dict: pthat, sigma_gen, sigma_err, event_weight | `hard_process_info` |
| `InitialState` | `info` | dict: n_coll, n_part, centrality (what the module reports) | `initial_state_info` |

Notes:

- **Compact output.** With `use_compact_output()`, `hadron_array()` takes iSS's arrays over
  without a copy: call it **once per event** (a second call finds them gone). Then
  `primary_array()` (once, after it). Without compact output it copies the C++ hadron list
  and can be called again.
- **`hadronize` on a manager** uses the one hadronization module it holds; with several,
  call it on the module.
- **The colour methods** exist only on builds that have them (X-SCAPE with the iColored
  colour closure, `contrib` since 2026-10-27): `hasattr(had, "colour_audit")` tells.

## 6 Record arrays

The small per-event tables are numpy structured arrays: `rec["px"]` instead of
`rec[:, 4]`. Each equals the array it reorders bit for bit (`jetscape.records`):

**`FINAL_PARTON_DTYPE`**: `final_parton_array()`, `final_parton_records(rows)`

| field | dtype | |
|---|---|---|
| `shower` | int32 | index of the shower (the manager's task list order) |
| `pid`, `pstat` | int32 | PDG id, status |
| `E`, `px`, `py`, `pz` | float64 | GeV |
| `t`, `x`, `y`, `z` | float64 | production point, fm/c and fm |
| `mass` | float64 | GeV |
| `col`, `acol` | uint32 | colour tags (C++ `unsigned int`; files written before the colour-tag fix hold tags above 2³¹) |

**`INITIATOR_DTYPE`** (the partons that started each shower): `shower, pid, pstat, E, px,
py, pz, t, x, y, z`. **`EDGE_DTYPE`** (`PartonShower.edge_array()`): `src, tgt, pid, pstat,
E, px, py, pz, t, x, y, z`, where `src` / `tgt` are vertex ids. **`NODE_DTYPE`**
(`node_array()`): `id, t, x, y, z`. **`HARD_PARTON_DTYPE`** (`HardProcess.parton_array()`):
`pid, E, px, py, pz, pt, y`.

Converters, for arrays from files or older code:

```python
from jetscape.records import (final_parton_records, final_parton_rows,
                              initiator_records, initiator_rows,
                              edge_records, node_records, hard_parton_records)

rec = final_parton_records(rows)        # (N, 14) FINAL_PARTON_COLUMNS -> records
rows = final_parton_rows(rec)           # records -> (N, 14), what the files store
initiator_rows(mgr.initiator_array())   # -> (K, 11), what HadronH5Writer takes
```

Records keep numpy's usual tools: `rec[rec["pstat"] >= 0]`, `np.unique(rec["shower"])`,
`np.hypot(rec["px"], rec["py"])`, `pandas.DataFrame(rec)`.

## 7 HadronArray

One event's hadrons (or a whole file's), **one array per field**, grouped into samples:
one sample per iSS oversample, one for a jet hadronization.

| attribute | | |
|---|---|---|
| `pid`, `pstat` | (n,) int32 | |
| `p` | (n, 4) float32 | E, px, py, pz |
| `x` | (n, 4) float32 or None | t, x, y, z; None when the file has no positions |
| `mass` | (n,) float32 or None | None when read from a file (not stored) |
| `sample_offsets` | (n_samples + 1,) int64 | sample s is rows `sample_offsets[s]:sample_offsets[s+1]` |
| `info` | dict | what came with the hadrons (below) |

| member | |
|---|---|
| `HadronArray.from_dict(d)` | from a bindings hadron dict, without a copy; `sample_counts` becomes `sample_offsets`, other keys go to `info` |
| `HadronArray.from_h5(path, units=None)` | a hadron file (section 10) |
| `len(h)`, `h.n_samples`, `h.sample_counts` | hadrons, samples, hadrons per sample |
| `h.to_dict()` | the bindings' dict (the same arrays), what `HadronH5Writer.append_unit` takes |
| `h.to_awkward(position=False, charge=False, by_unit=False)` | the nested view (section 8) |
| `h.to_hadrons(unit_offsets=None)` | a `hadrons_h5.Hadrons`, for its `hist` / `total` |

`info` holds the seed and `n_partons_out` after `hadronize()`, and after `from_h5`:
`unit_offsets`, `units` (the per-unit table), `attrs` (the file's attributes), `path`.

**Memory.** `HadronArray` wraps the arrays the bindings return, so with compact output iSS's
hadrons are never copied (rule M1: a record array would interleave the fields, a full
copy). It stores no derived columns (M2): pt, eta, phi, rapidity come from `to_awkward()`,
computed when used. `to_hadrons()` computes them all in float64, about 2.4x the memory;
use it for the `Hadrons` histogram helpers, not to hold many events.

## 8 Nested views: `to_awkward`

`js.to_awkward(obj, position=False, charge=False, by_unit=False)` (also
`HadronArray.to_awkward(...)`):

| input | result | memory |
|---|---|---|
| `HadronArray` / hadron dict | samples x hadrons | the same buffers (no copy; `p`'s columns are strided views) |
| `HadronArray.from_h5(...)`, `by_unit=True` | units x samples x hadrons | the same buffers |
| final-parton or initiator records | showers x partons (by the runs of `shower`) | a copy (small) |
| edge, node, hard-parton records | a flat array of records | a copy (small) |

The records are **`vector` `Momentum4D`** (fields E, px, py, pz), so kinematics are
computed when asked, never stored:

```python
h = driver.soft_particlization.hadron_array().to_awkward()
h.pt, h.eta, h.phi, h.rapidity, h.mass            # jagged, same shape as h
mid = h[abs(h.eta) < 0.5]
dndeta = ak.mean(ak.num(mid), axis=0)              # mean over the oversamples
```

- **The position is nested**: `position=True` adds `h.position`, a `Vector4D` record (t, x,
  y, z), so `h.position.t`, `h.position.rho`. It cannot be flat fields: `vector` reads
  fields named t, x, y, z as the coordinates of a Momentum4D record, and `.pt` / `.E` would
  then come from the position. Records always carry theirs (`showers.position.t`). Node
  records, which have no momentum, are `Vector4D` themselves.
- `charge=True` adds a `charge` field [e] from the PDG.
- **A stored mass is `rest_mass`**: `HadronArray.mass` (in-memory hadrons) and the records'
  `mass` column become the field `rest_mass` (X-SCAPE's rest mass); `h.mass` is vector's,
  computed from E and p. Not `mass`: vector takes every temporal name (E, mass, m, tau, ...)
  as a coordinate, and from vector 1.9 on refuses a record with two of them.
- `ak.flatten(a, axis=1)` merges the outer level (units x samples -> samples);
  `ak.num(a, axis=k)` counts at each level.

awkward and vector are imported when `to_awkward` is called, never at `import jetscape`.

## 9 Selecting species: `select`

```python
js.select(h, "pi+")                      # one PDG name, as `particle` spells it
js.select(h, ["K+", "K-"])               # any of several
js.select(h, "charged")                  # a class
js.select(h, ["p", "p~", "Lambda", "Lambda~"])
```

Returns a boolean mask of the input's shape: a flat mask for a `HadronArray`, a record array
or a pid array; a jagged mask for an awkward array (so `h[js.select(h, "charged")]`).
Classes: `charged`, `neutral`, `hadron`, `meson`, `baryon`, `strange`. Names: `pi+`, `pi0`,
`K-`, `K(S)0`, `p`, `p~`, `n`, `Lambda`, `Sigma+`, `Xi-`, `Omega-`, `eta`, `rho(770)0`, ...
(`particle.Particle.from_name`). Each distinct pid is looked up once (`np.unique`), so the
cost does not grow with the number of hadrons.

**`"charged"` versus `hadrons_h5.CHARGED`.** `select(..., "charged")` is every pid with a
nonzero PDG charge, resonances included. `hadrons_h5.CHARGED`, which `Hadrons.charged` and
the existing analyses use, lists the stable charged hadrons and leptons. The two agree on
decayed hadrons. On files with undecayed primaries (iSS `<keep_primaries>`) `select` also
counts the charged resonances (ρ±, K*±, Δ, a1, ...): on one such file 1,114 of 5,434
hadrons, 134 pids. `CHARGED` is unchanged, so existing results stay as they were.

## 10 Hadron files

The hadron files (`hadrons_h5`, the format of the productions and `hadronize.py`) take a
`HadronArray` per unit (event) and give one back:

```python
from jetscape.hadrons_h5 import HadronH5Writer
from jetscape.records import initiator_rows

with HadronH5Writer("x_hadrons_bulk_bg.h5", tag="bulk_bg", n_samples=20) as bulk_w, \
     HadronH5Writer("x_hadrons_jet_frag.h5", tag="jet_frag", n_samples=1,
                    initiators=True) as jet_w:
    for i in driver.events():
        bulk = driver.soft_particlization.hadron_array()
        jet = driver.hadronization_manager.hadron_array()
        bulk_w.append_unit(bulk.to_dict(), unit=i, event=i, seed=iss.last_random_seed)
        jet_w.append_unit(jet.to_dict(), unit=i, event=i,
                          initiators=initiator_rows(mgr.initiator_array()))
```

Keywords besides the hadrons (`event=`, `seed=`, `pthat=`, ...) become the file's per-unit
table. Reading:

```python
h = js.HadronArray.from_h5("x_hadrons_bulk_bg.h5")          # every unit, raw fields
h.info["units"]["seed"]                                      # the per-unit table
h.info["unit_offsets"]                                       # samples of unit u
a = h.to_awkward(by_unit=True)                               # units x samples x hadrons
first = js.HadronArray.from_h5("x_hadrons_bulk_bg.h5", units=[0])
```

`from_h5` reads the stored fields only (float32, no derived columns). For whole campaigns
with event-averaged histograms and compound-Poisson errors, `hadrons_h5.HadronFileReader`
and `Hadrons` stay the tools; `HadronArray.to_hadrons()` bridges to them.

## 11 Jets with FastJet

FastJet's Python package clusters jagged awkward arrays directly: **one call for every
event, or every oversample**.

```python
import fastjet

jdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)

# the jets of every event (a jet file: events x hadrons)
h = js.HadronArray.from_h5(jet_path).to_awkward()
acc = h[abs(h.eta) < 2]
jets = fastjet.ClusterSequence(acc, jdef).inclusive_jets(min_pt=5.0)    # events x jets
jets = jets[ak.argsort(jets.pt, ascending=False)]
leading = ak.firsts(jets)

# charged jets
ch = acc[js.select(acc, "charged")]
ch_jets = fastjet.ClusterSequence(ch, jdef).inclusive_jets(min_pt=3.0)

# every oversample of every event (bulk: units x samples x hadrons)
b = js.HadronArray.from_h5(bulk_path).to_awkward(by_unit=True)
per_sample = ak.flatten(b, axis=1)                         # samples x hadrons
bulk_jets = fastjet.ClusterSequence(per_sample[abs(per_sample.eta) < 2],
                                    jdef).inclusive_jets(min_pt=5.0)
bulk_jets = ak.unflatten(bulk_jets, ak.num(b, axis=1))      # back to units x samples x jets

# a jet embedded in its event's bulk (first oversample)
embedded = ak.concatenate([h, b[:, 0]], axis=1)
```

`ClusterSequence` on a jagged array gives the same jets as a loop over the samples, sample
by sample (checked in `tests/test_pythonic.py`). The jets are `vector` records too (`.pt`,
`.eta`, `.phi`, `.m`). `fastjet.ClusterSequence(...).constituent_index()` maps jets back to
hadrons.

## 12 Particles and four-vectors

**`FourVector`** gains Python protocols in `comp()` order (t, x, y, z) = (E, px, py, pz):

```python
from jetscape import pyjetscape_core as core        # the bound classes; jetscape (js) has the Pythonic helpers

v = core.FourVector.momentum(10.0, 1.0, 2.0, 3.0)  # E, px, py, pz
E, px, py, pz = v                                   # iteration
v[0], v[-1]                                         # 10.0, 3.0
np.asarray(v)                                       # array([10., 1., 2., 3.])
v == core.FourVector(1, 2, 3, 10)                   # True: equal by value
pickle.dumps(v)
```

Because `==` compares values, `FourVector` is not hashable (`hash(v)` raises), as for any
mutable value type.

**Particles** (`Parton`, `Photon`, `Hadron`) get `momentum` and `position`, FourVector
copies that unpack in those orders, and keyword factories that take any 4-sequence or a
FourVector:

```python
p = js.make_parton(21, momentum=(50, 50, 0, 0), position=(0.5, 0, 0, 0),
                   stat=0, label=0, color=101, anti_color=102)
g = js.make_photon(momentum=(5, 0, 0, 5))
h = js.make_hadron(211, momentum=p.momentum, position=p.position, stat=1)   # PDG mass
h = js.make_hadron(2212, (2.0, 0.1, 0.2, 0.3), mass=0.938)
E, px, py, pz = p.momentum
t, x, y, z = p.position
```

Particles keep identity equality (`p == q` only if the same object), so they stay usable in
sets and as dict keys. They pickle (pid, momentum, position, status, label; partons also
colour tags and formation time; hadrons mass, charge, baryon number, strangeness), e.g. for
multiprocessing.

**Property view.** The bound classes keep their methods (`p.pt()`): properties there would
break every script that calls them. `jetscape.particles.view(p)` gives properties instead:

```python
from jetscape.particles import view
v = view(p)
v.pt, v.eta, v.phi, v.rapidity, v.e, v.px, v.mass, v.t, v.momentum
v.unwrap() is p
```

Each property is one call into C++ (about 1.4 µs): for many particles use the arrays.

## 13 Parton showers

```python
for shower in driver.energy_loss_manager.showers:
    for parton in shower:                          # the partons (edges)
        start, end = shower.start_node(parton), shower.end_node(parton)
        if end.is_one_to_two:
            print(parton.pid(), "splits at t =", end.position.t(),
                  "z =", end.split_z, "kt =", end.split_kt, "dR =", end.split_delta_r)
    edges = shower.edge_array()                    # EDGE_DTYPE records
    nodes = shower.node_array()                    # NODE_DTYPE records
```

| `PartonShower` | |
|---|---|
| `iter(shower)` | the partons (`partons()`); no `len()`, so an empty shower stays true in `if shower:` |
| `nodes` | every vertex as a `ShowerNode` |
| `start_node(p)`, `end_node(p)` | the vertices where `p` starts and ends |
| `edge_array()`, `node_array()` | the graph as records |

| `ShowerNode` | |
|---|---|
| `id`, `vertex`, `position` | the vertex id, the C++ `Vertex`, its position (FourVector) |
| `parents`, `children` | the partons that end / start here |
| `split_z`, `split_kt`, `split_delta_r` | the split variables of a 1 → 2 vertex |
| `is_one_to_two`, `is_end` | the vertex kind |

Showers and partons are C++ objects of the event: use them inside the event loop. The
records (`edge_array()`, `final_parton_array()`) are copies and outlive it.

## 14 Writing modules

```python
import jetscape as js
from jetscape import pyjetscape_core as core   # the base classes (JetEnergyLoss, ...) live here

@js.module("CustomModuleHalve", section="Eloss")
class Halve(core.JetEnergyLoss):
    xml_parameters = {"t_split": 1.0, "e_min": 30.0}

    def DoEnergyLoss(self, deltaT, time, Q2, pIn, pOut):
        p = pIn[0]
        if time >= self.t_split and p.e() > self.e_min:
            E, px, py, pz = p.momentum
            return [js.make_parton(p.pid(), (E / 2, px / 2, s * 1.0, pz / 2), p.position)
                    for s in (+1, -1)]
```

and the user XML: `<Eloss><CustomModuleHalve><e_min>25</e_min></CustomModuleHalve></Eloss>`.

**`@js.module(name, section="")`** registers the class in JETSCAPE's module factory
(`register_module`), so the XML can name it: every section takes a module whose element
name contains `CustomModule` (`<JetHadronization>` takes it as `<name>CustomModuleX</name>`).
A module built by hand (`eloss.Add(Halve())`) needs no registration but may have one.

**`xml_parameters`** (`{key: default}`, int, float, str or bool):

1. the defaults become the main-XML default entry `<section><name><key>default</key>...`
   (JetScape refuses a user-XML tag the main XML lacks, so this is needed for the user XML
   to set them);
2. they become class attributes, so a module built by hand has them before `Init`;
3. before the class's own `InitTask` runs (the decorator wraps it), each is read into
   `self.<key>` from `[section, name, key]`: the user XML's value, else the default, with
   the default's type (`bool`: `0`, `false`, `off`, `no` are False). Without an XML (a
   module used outside a run) the defaults stay.

The name is `xml_parameters`, not `params`: `LiquefierBase.params()` is a bound method, and
FastHydro's `DropletBridge` sets `self.params`.

**Returning the output.** Instead of appending to the output lists:

| module | return |
|---|---|
| `DoEnergyLoss(deltaT, time, Q2, pIn, pOut)` | a list of partons: the partons at `time + deltaT` |
| `DoHadronization(shower, hOut, pOut)` | a list of hadrons, or a tuple `(hadrons, partons)` |

`None` (or no return) keeps the behaviour of appending. A returned list goes after what the
output held before the call (an earlier hadronization module's hadrons stay). Returning a
list and also appending raises `ValueError`; returning anything else raises `TypeError`.
For hadronization return a list, not a tuple, for the hadrons alone: a 2-tuple is read as
`(hadrons, partons)`.

**`self.rng`**: a numpy `Generator` for a Python module, seeded from the task's own C++
engine (`GetMt19937Generator()`) on first use:

- reproducible with the run's `<Random><seed>`;
- one stream per task: the energy-loss manager's per-shower `Clone()` gets a new task
  number, so each copy draws its own numbers (the cached generator is tied to the task
  number);
- needs the run's random engines, i.e. a loaded XML (inside a run, or after
  `load_xml`);
- an attribute of your own (`self.rng = np.random.default_rng(1)`) takes precedence;
- C++ modules do not have it: on them `rng` raises `AttributeError`, so nothing (not even
  `hasattr`) draws from their engines.

Everything else about Python modules (ownership, `Clone()`, the lists, the stage table) is in
the README, "Writing modules in Python". `example/python_modules/toy_modules_pythonic.py`
has one module per stage in this style.

## 15 Type stubs

`python/jetscape/pyjetscape_core.pyi` with `py.typed` describes the bindings and the layer,
so editors complete and type-check the code and show the docstrings, and mypy checks
scripts:

```bash
MYPYPATH=contribs/PyJetscape/python mypy my_script.py --follow-imports=silent
```

The stubs follow the build they were made from (names behind build flags: ROOT, the iColored
colour functions, iSS primaries). After a build that changes the bindings, regenerate them
(js-contrib, needs `pip install pybind11-stubgen`):

```bash
python utils/gen_pyjetscape_stubs.py
```

## 16 What the layer guarantees

- **No existing output changes.** `final_partons_numpy`, `PartonShower.to_numpy`,
  `vertices_to_numpy`, the hadron dicts, `hadronize_partons` and the files keep their
  columns and dtypes. A production job run with and without the layer gives identical
  files.
- **New names only.** The installer refuses any name a class already has (own or
  inherited) and any dunder outside a short list (`__iter__`, `__getitem__`, `__array__`,
  `__eq__`, `__ne__`, `__hash__`, `__reduce__`, `__enter__`, `__exit__`).
  `jetscape.pythonic.installed_names()` lists everything it added.
- **Your attributes win.** The layer's properties are non-data descriptors: a module that
  sets `self.rng`, `self.info`, ... keeps its own.
- **Particles stay hashable**, `PartonShower` has no `len()` (an empty shower stays true),
  the bound methods (`p.pt()`) stay methods.

## 17 Cost and memory

GB10 (`docs/Plans/PLAN_pyjetscape_pythonic.md`, V4 and V6):

| | |
|---|---|
| iSS, 972,215 cells, 1000 oversamples (7.0M hadrons): resident memory after the step / peak | dict as returned 686 / 916 MB; `HadronArray` 686 / 915 MB (no copy); `to_awkward()` + `.pt` 761 / 915 MB; `Hadrons` 1241 / 1240 MB |
| a Python energy-loss module, 715,438 `DoEnergyLoss` calls (500 events) | 3.73 s against 3.70 s before the layer (+0.9 %); about 1.4 µs per call into Python |
| a bound method call (`p.pt()`, `view(p).pt`) | about 1.4 µs: use the arrays for many particles |

## 18 Pitfalls

- **Read results inside the event loop.** `ClearPerEvent()` frees the event: call the
  `_array()` methods inside `for i in driver.events():`. Their results (numpy) outlive the
  event; C++ objects (showers, partons, hadron lists) do not.
- **Compact output is taken once.** With `use_compact_output()`, `hadron_array()` takes
  iSS's arrays over: once per event.
- **No surface, no iSS.** An event without a medium (O+O with no string) has no surface;
  check `len(driver.hydro.surface_to_numpy())` before `hadron_array()`, as the notebook does.
- **`vector` and field names.** Do not add fields named t, x, y, z (or `tau`, `rho`, `phi`,
  `eta`, `theta` ...) to a Momentum4D record: `vector` would read them as coordinates. Use
  nested records, as `position`.
- **`FourVector` in sets or as dict keys** raises (unhashable); use `tuple(v)`.
- **ISR runs** have two energy-loss managers: the ISR chain's `IsrManager` inside the hard
  process and the final-state one. `driver.energy_loss_manager` is the final-state one.
- **`self.rng` before the XML** raises "Trying to use JetScapeTaskSupport::
  GetMt19937Generator before initialization": use it in `InitTask` or later.
- **One thread.** Python modules take the GIL in every call.

## 19 Old and new, side by side

| task | bindings | Pythonic layer |
|---|---|---|
| run events | `js = JetScapePerEvent(); js.SetXMLMainFileName(m); ...; js.Init(); for _ in range(js.GetNumberOfEvents()): js.ExecPerEvent(); ...; js.ClearPerEvent(); js.Finish()` | `with js.per_event(m, u) as d: for i in d.events(): ...` |
| the energy-loss manager | `JetScapeSignalManager.Instance().GetJetEnergyLossManagerPointer()` | `d.energy_loss_manager` |
| final partons | `fp = mgr.final_partons_numpy(); fp[:, 4]` (px) | `fp = mgr.final_parton_array(); fp["px"]` |
| soft hadrons | `d = soft_hadrons_numpy(iss); d["p"][:, 1]`, `np.split(..., np.cumsum(d["sample_counts"])[:-1])` | `h = iss.hadron_array().to_awkward(); h.px`, `h.pt` (samples x hadrons) |
| jet hadrons | `hadronization_hadrons_numpy(mgr)` | `had.hadron_array()` |
| hadronize stored partons | `hadronize_partons(module, rows, seed)` | `module.hadronize(records, seed=...)` |
| hard process | `hard_process_info(gun)["pthat"]`, `hard_partons_numpy(gun)` | `gun.info["pthat"]`, `gun.parton_array()` |
| iSS settings | `soft_set_compact_output(iss, True)`, `soft_last_random_seed(iss)` | `iss.use_compact_output()`, `iss.last_random_seed` |
| colour audit | `jet_hadronization_colour_audit(had)` | `had.colour_audit()` |
| a parton | `Parton(0, 21, 0, FourVector(px, py, pz, E), FourVector(x, y, z, t))` | `js.make_parton(21, (E, px, py, pz), (t, x, y, z))` |
| its momentum | `p.e(), p.px(), p.py(), p.pz()` | `E, px, py, pz = p.momentum` |
| a module's parameter | `self.get_xml_element_double(["Eloss", "CustomModuleX", "e_min"], False)` + `register_module(..., defaults="<e_min>30</e_min>")` | `xml_parameters = {"e_min": 30.0}` → `self.e_min` |
| a module's output | `pOut.append(Parton(...))` | `return [js.make_parton(...)]` |
| a module's random numbers | `self.GetMt19937Generator()()`, `.uniform()` | `self.rng.normal(...)` (numpy) |
| shower splits | `ps.split_z(ps.edge_of(p)[2])` | `ps.end_node(p).split_z` |
| a hadron file | `Hadrons.from_h5(path)` (derived columns, float64) | `HadronArray.from_h5(path)` (raw, float32), `.to_awkward(by_unit=True)` |

## Example files

| file | |
|---|---|
| `example/pythonic/oo_workflow.ipynb` | 3 O+O events through the full chain: run, hadron files, read back, awkward, FastJet |
| `example/pythonic/oo_workflow.py` | its run as a script and a template for a production of one's own (`--xml`, `--events`, `--seed`, `--outdir`), with `js.JobDir` |
| `example/pythonic/oo_workflow.xml` | its user XML (made once from the O+O production's template; change the run there) |
| `example/pythonic/oo_analysis.ipynb` | the analysis half of `oo_workflow.ipynb` on its hadron files streamed from the OSDF (`osdf:///fno4hic/test/OO_test`); the analysis env only, no build |
| `example/python_modules/toy_modules_pythonic.py` | one module per stage in the new style |
| `example/python_modules/run_by_hand_pythonic.py` | a hand-built pipeline with the context manager (`--style old|new`) |
| `tests/test_pythonic.py`, `tests/test_pythonic_run.py` | every feature, against the functions it wraps |
