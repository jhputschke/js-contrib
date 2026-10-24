# python_modules — JETSCAPE modules written in Python

Every JETSCAPE stage can be a Python class. You subclass the C++ base class and override the
method JETSCAPE calls, with the C++ name and argument order; the framework then creates, copies,
calls and writes your module as it would a C++ one. This folder has one small module per stage
and runs the same all-Python pipeline two ways.

| file | what it is |
|---|---|
| `toy_modules.py` | `ToyGun`, `ToySplit`, `ToyHadronizer`, `ToySampler`, `ToyAfterburner`, `ToyWriter`: one template per stage, and `register_toy_modules()` |
| `run_from_xml.py` + `user_from_xml.xml` | the modules named in the user XML (Mode A/C) |
| `run_by_hand.py` + `user_by_hand.xml` | the same pipeline built in Python (Mode B) |

The toys are not physics; they show the interface. The design, the validation (Python modules
reproducing C++ ones byte for byte) and the full list of bound classes are in js-contrib
`docs/Plans/PLAN_pyjetscape_complete.md`; the reference summary is the section "Writing modules
in Python" of `contribs/PyJetscape/README.md`.

## Running the examples

X-SCAPE must be built with PyJetscape and `-DUSE_3DGlauber=ON` (see the PyJetscape README); the
compiled `pyjetscape_core` lands in `contribs/PyJetscape/python/jetscape/`, which the scripts put
on `sys.path` themselves.

```bash
conda activate js_fno
cd contribs/PyJetscape/example/python_modules
python run_from_xml.py --outdir /tmp/toy     # modules named in user_from_xml.xml
python run_by_hand.py  --outdir /tmp/toy     # the same pipeline built in Python
```

`--main-xml` points to another main XML (default: the X-SCAPE checkout's
`config/jetscape_main.xml`). Each run makes 3 events and prints the task list and a `SUMMARY`
line:

```
tasks: ['ToyWriter', 'ToyGun', 'JetEnergyLossManager', 'HadronizationManager', 'ToySampler', 'ToyAfterburner', 'JetScapeWriter']
SUMMARY {"afterburner": [7, 7, 7], "writer": [{"Parton": 4, "PartonShower": 2, "Hadron": 4, "sigma_gen": 1.0}, ...], "ascii": [[2, 4, 100.0, 1.0], ...]}
```

- `afterburner`: the hadrons the afterburner holds per event, 3 soft + 4 jet hadrons.
- `writer`: what the Python writer received per event: 2 hard partons and 2 shower-initiating
  partons (4 `Parton`), 2 `PartonShower`, 4 `Hadron`, and the gun's cross section from the event
  header.
- `ascii`: the C++ `JetScapeWriterAscii` file of the run read back with `JetScapeReaderAscii`:
  per event 2 showers, 4 hadrons, their summed energy (100 GeV from the XML, where the gun's
  `<E>` is 50; 80 GeV by hand, where it takes its default of 40), and sigma.

`tests/test_bindings_pipeline.py` runs both scripts and checks these numbers.

## How JETSCAPE calls a module

| when | method | notes |
|---|---|---|
| `JetScape.Init()` | `InitTask()` | once; read the XML parameters here |
| every event | `ExecuteTask()` | gun, sampler, afterburner: the event's work |
| every event, energy loss | `DoEnergyLoss(...)` | once per parton per time step, see below |
| every event, jet hadronization | `DoHadronization(...)` | once per event, all showers at once |
| end of every event | `ClearTask()` | drop per-event state (C++ base classes clear theirs) |
| `JetScape.Finish()` | `FinishTask()` | once; close files |

Override only what you need: a method you leave out does what the C++ base class does (for an
energy-loss or hadronization module, `InitTask` then does nothing). Call the base explicitly when
you extend it, e.g. `js.SoftParticlization.ClearTask(self)`.

## The modules, stage by stage

### Hard process — `ToyGun(js.HardProcess)`

`ExecuteTask()` fills the event with `self.AddParton(js.Parton(...))` (and `AddHadron` for
hadrons from the hard process). `ToyGun` puts two back-to-back gluons of energy `E` at the hard
vertex: the initial state's `sample_binary_collision_point()` → `(t, x, y, z)` when there is one
(`JetScapeSignalManager.Instance().GetInitialStatePointer()`), else the origin. Overriding
`GetSigmaGen`, `GetSigmaErr`, `GetPtHat`, `GetEventWeight` sets what the writers get in the event
header (C++ asks the gun through these). C++ counterparts: `PGun`, `PythiaGun`.

### Energy loss — `ToySplit(js.JetEnergyLoss)`

```python
def DoEnergyLoss(self, deltaT, time, Q2, pIn, pOut): ...
```

- **When:** for every parton of a shower, every time step up to `<Eloss><maxT>`, in steps of
  `<Eloss><deltaT>` [fm/c]. `time` is the end of the step (current time + `deltaT`). `Q2` is
  what the framework passes there: the parton's pT (`JetEnergyLoss.cc`), not a virtuality; the
  virtuality is `pIn[0].virtuality()`.
- **`pIn`:** a list with the parton of this step: the C++ object itself, so `p.set_stat(...)`,
  `p.set_form_time(...)` change it in place. It is valid during the call only; do not keep it.
- **`pOut`:** append the partons at `time`. What the framework does with it:

  | `pOut` after the call | the shower |
  |---|---|
  | empty | the parton (`pIn[0]`, as changed) goes on unchanged into the next step |
  | one parton | it replaces the parton and goes on (no new vertex) |
  | two or more | a vertex at `time`, each parton a new branch that goes on |

  Partons with the liquefier's statuses (droplet −11, miss −13, negative −17) and non-partons
  (photons, leptons) do not go on.
- **Several modules** (e.g. Matter then LBT, or yours after Matter) are called in turn on the
  same lists, as in C++; a module leaves alone a parton another one handles
  (`p.GetController()`, `pOut` already filled).
- **The medium:** `self.GetHydroCell(t, x, y, z)` gives a `FluidCellInfo` (`temperature`,
  `energy_density`, `pressure`, `vx`, `vy`, `vz`, …) or `None` without a hydro.
- **Copies:** the energy-loss manager copies the module once per shower, at the start of every
  event, with `Clone()`, and drops the copies at the end of the event. The default copy runs the
  C++ copy constructor (same id and settings, a new task number in the same order as for a C++
  module) and copies the instance attributes shallowly; `__init__` is not called again, and
  `InitTask` is not called on copies. For attributes that must not be shared between showers
  (lists, objects with state), override `Clone`:

  ```python
  def Clone(self):
      copy = js.JetEnergyLoss.Clone(self)     # the default copy
      copy.history = []                       # one per shower
      return copy
  ```

C++ counterparts: `Matter`, `LBT`.

### Jet hadronization — `ToyHadronizer(js.Hadronization)`

```python
def DoHadronization(self, shower, hOut, pOut): ...
```

Once per event, after the energy loss: `shower` is a list with one list of final partons per
shower (the C++ objects); append `Hadron`s to `hOut` and the partons you leave unhadronized to
`pOut` (both may already hold what an earlier module made). `self.GetHydroHyperSurface(T)` and
`self.GetHydroCell(t, x, y, z)` give the medium. C++ counterparts: `ColorlessHadronization`,
`iColoredHadronization`, `HybridHadronization`.

### Soft particlization — `ToySampler(js.SoftParticlization)`

`ExecuteTask()` stores the event's soft hadrons with `self.set_hadron_list(samples)`: a list of
oversamples, each a list of `Hadron`s. The hydro's freeze-out surface is
`self.GetHydroHyperSurface()`, a list of `SurfaceCellInfo` (`tau`, `x`, `y`, `eta`,
`d3sigma_mu`, `umu`, `temperature`, `energy_density`, …), empty without a hydro. C++
counterpart: iSS.

### Afterburner — `ToyAfterburner(js.Afterburner)`

`ExecuteTask()` takes its input with `self.GatherAfterburnerHadrons()`: per oversample the soft
hadrons, plus the jet hadrons when `<Afterburner><include_fragmentation_hadrons>` is 1 (one
oversample only, then). `GetCurrentHadronList()` returns the result (a list of `Hadron`s). C++
counterpart: SMASH.

### Writer — `ToyWriter(js.JetScapeWriter)`

Implement `GetStatus()` (True when ready) and `Write(obj)`. Per event JETSCAPE calls, in this
order: the tasks fill the event header (`self.GetHeader()`: `GetSigmaGen()`, `GetEventWeight()`,
`GetPtHat()`, `GetVertexX()`, `GetNcoll()`, …), `WriteHeaderToFile()`, then `Write(obj)` for
every object the tasks write, then `WriteEvent()`. What arrives in `Write`:

| task | writes |
|---|---|
| hard process | its hard partons (`Parton`) |
| energy loss, per shower | the shower-initiating `Parton`, then the `PartonShower` |
| jet hadronization | its `Hadron`s |

plus `str` for some tasks; comments go to `WriteComment(s)`. A writer is added with
`JetScape.Add(writer)` before `Init()` (in Mode A too). C++ writers come from
`js.create_module("JetScapeWriterAscii")` + `js.set_writer_output_file(w, name)`;
`js.JetScapeReaderAscii(name)` reads their files back (`Next()`, `GetPartonShowers()`,
`GetHadrons()`, `GetSigmaGen()`).

### The other stages

Not in the toy file, with examples elsewhere:

| stage | base class | override | example |
|---|---|---|---|
| initial state | `InitialState` | `ExecuteTask`, `GetNcoll`, `GetNpart`, `SampleABinaryCollisionPoint` (returns `(t, x, y, z)`) | `example/inspect_bulk_info.py` (`GaussianIC`), FastHydro's `initial_state.py` |
| pre-equilibrium | `PreequilibriumDynamics` | `InitializePreequilibrium`, `EvolvePreequilibrium` (fill with `set_e_numpy`, `set_utau_numpy`, …) | `tests/test_bindings_stages.py` |
| hydro | `FluidDynamics` | `InitializeHydro`, `EvolveHydro` | `python/jetscape/fno_hydro.py` (`PyFNOHydro`), `surface_replay.py`, FastHydro's `hydro.py` |
| liquefier | `LiquefierBase` | `smearing_kernel(tau, x, y, eta, droplet)` → `[j^tau, j^x, j^y, j^eta]` | `tests/test_bindings_stages.py` |
| shower growth | `PartonShowerGenerator` | `DoShower(jloss)` | `tests/test_bindings_pipeline.py` |

## Named in the XML (`run_from_xml.py`)

```python
js.register_module("CustomModuleToyGun", ToyGun, section="Hard", defaults="<E>40</E>")
```

- **The name.** JETSCAPE's XML takes a module whose element name contains `CustomModule` in every
  section: `<Hard><CustomModuleToyGun>`, `<Eloss><CustomModuleToySplit>`,
  `<SoftParticlization><CustomModuleToySampler>`, `<Afterburner><CustomModuleToyAfterburner>`;
  jet hadronization takes it as text, `<JetHadronization><name>CustomModuleToyHadronizer</name>`.
  `register_module` puts the class into JETSCAPE's module factory under that name; the XML (or
  `js.create_module(name)`) then makes the module.
- **The main XML.** JetScape stops (`exit`) on a user-XML tag that does not exist in the main
  XML, parameters included. `JetScape.Init()` therefore adds `<section><name>defaults</name>` to
  the main XML in memory for every registered module: `section` is where it belongs
  (`"Eloss"`, `"Hard"`, …; `""` for none), `defaults` its parameters with their default values
  (XML text). The file on disk is not changed.
- **Parameters.** In `InitTask`, as a C++ module does:
  `self.get_xml_element_double(["Hard", "CustomModuleToyGun", "E"], False)` returns the user XML's
  value, else the default (`get_xml_element_int`, `get_xml_element_text` likewise; with `True`
  a missing element raises).
- `js.registered_modules()` lists the factory (C++ modules of the build and the Python ones),
  `js.unregister_module(name)` removes a Python one.

`user_from_xml.xml` sets the run (3 events, seed 1), switches the C++ ASCII writer on
(`<JetScapeWriterAscii> on`, `<outputFilename>toy_from_xml`, i.e. `toy_from_xml.dat`) and names
the five modules; the gun's `<E>50</E>` overrides the default 40.

## Built by hand (`run_by_hand.py`)

The managers and their containers are made in Python and the modules put inside them, as the XML
would:

```python
eloss_manager = js.JetEnergyLossManager()   # copies `eloss` once per shower
eloss = js.JetEnergyLoss()                  # the per-shower container (deltaT, maxT from the XML)
eloss.Add(ToySplit())                       # the modules, in order (Matter before LBT)
eloss_manager.Add(eloss)

hadronization_manager = js.HadronizationManager()
hadronization = js.Hadronization()          # the container the manager runs
hadronization.Add(ToyHadronizer())
hadronization_manager.Add(hadronization)
```

Then every top-level task goes into the driver in pipeline order (initial state, hard process,
pre-equilibrium, hydro, energy-loss manager, hadronization manager, soft particlization,
afterburner; writers anywhere) before `Init()`. `user_by_hand.xml` switches the automatic task
list off (`<enableAutomaticTaskListDetermination> false`) and only sets the run. Registering the
modules is not needed to build them by hand, but `run_by_hand.py` does it so that `ToyGun` finds
its default `<E>` in the main XML.

## Particles and lists

- `FourVector(x, y, z, t)` (momenta: `FourVector(px, py, pz, E)`); `comp(0)` is t.
- `Parton(label, pid, stat, p, x)` with `p`, `x` FourVectors, or
  `Parton(label, pid, stat, pt, eta, phi, e, x=[t, x, y, z])`; `Hadron(label, pid, stat, p, x[,
  mass[, charge, baryon_number, strangeness]])`.
- Getters and setters as in C++: `pid()`, `pstat()`, `e()`, `px()`, `pt()`, `eta()`, `phi()`,
  `p_in()`, `x_in()`, `set_stat()`, `set_form_time()`, `set_color()`, `reset_momentum()`, ….
- **A particle's `x()`, `y()`, `z()`, `t()` are its position**; a parton's virtuality is
  `virtuality()` (C++ `Parton::t()`), set with `set_t()`.
- Lists C++ passes by reference arrive as Python lists: the inputs are the C++ objects (valid
  during the call), the outputs are copied back into C++ after the call.
- `PartonShower`: `final_partons()`, `children(p)`, `parents(p)`, `production_vertex(p)`,
  `end_vertex(p)`, `split_z(vertex_id)`, … (ids as in `to_numpy()`).

## Random numbers

`self.GetMt19937Generator()` is the module's own engine, as for a C++ module: seeded from
`<Random><seed>` and the module's task number. `g()` is the next 32-bit draw, `g.uniform()` a draw
in [0, 1) as C++'s `ZeroOneDistribution(*gen)`; for numpy, seed it once:
`np.random.default_rng(self.GetMt19937Generator()())`. Every copy made by `Clone()` has its own
task number and so its own engine, as in C++. Runs are then reproducible, and adding a Python
module shifts the task numbers (and seeds) of the tasks made after it, as adding a C++ module
does.

## Checking a module against C++

The way `tests/equivalence_driver.py` does it: run the C++ module and your Python version with
the same seed and compare the output files. For identical random numbers the Python version must
make the same tasks in the same order; a module that holds a C++ module of its own gives it its
own task number back with `js.next_task_number()` / `js.set_next_task_number()` (see `PyMatter`
there). Run each configuration in its own process: JETSCAPE keeps the XML, the seed and the
module registry in process-wide singletons, and stops the process on configuration errors.

## Pitfalls

| symptom | cause |
|---|---|
| `User XML tag <X> is unrecognized` and the run stops | the module (or a parameter tag) is not registered with `section=` / `defaults=` before `JetScape.Init()` |
| a module named in the XML never runs | its name does not contain `CustomModule` |
| a parton's `t()` is not its virtuality | `t()` is the position time; use `virtuality()` |
| state shared between showers | the default `Clone()` copies attributes shallowly; override `Clone` |
| a liquefier attached from Python is gone | C++ keeps only a weak reference (as for its C++ liquefier); keep the Python object alive |
| `undefined symbol ... iColoredHadronization` on import | X-SCAPE built without `-DUSE_3DGlauber=ON` |
| `all 256 slots for Python modules are taken` | more than 256 `register_module` names in one process |

Python modules run in one thread (each call takes the GIL).

## Cost

A call into a Python energy-loss module costs about 1.4 µs on top of the module's own work
(GB10: 715,438 calls in 500 events of a 100 GeV gluon in vacuum, 3.70 s through a Python module
handing each step to C++ Matter against 2.72 s for Matter alone). A module written in Python
costs what its Python code costs on top of that, per parton and time step.
