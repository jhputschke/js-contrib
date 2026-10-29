# Writing modules in Python

Every stage of an X-SCAPE event (hard process, energy loss, hadronization, soft particlization,
afterburner, writer, hydro, ...) can be a Python class. You subclass the C++ base class and
override the method JETSCAPE calls; the framework then creates, copies, calls and writes your
module exactly as it would a C++ one, next to the C++ modules of the run. Use this to prototype a
new physics module, to wrap a model that lives in Python, or to tap the event at a stage the
standard output does not cover. Because the bindings keep the C++ names and argument order, a
working Python module ports to C++ line by line, and a C++ module ports to Python the same way.

Python modules need no compiler: they run with the [production container](../install/container.md)
and the [Mac wheel](../install/mac.md) as they are, against the compiled X-SCAPE inside. The
examples on this page are in `python_modules/` of the copied examples (`xscape-examples DEST`),
or in the repository under
[`contribs/PyJetscape/example/python_modules/`](https://github.com/jhputschke/js-contrib/tree/main/contribs/PyJetscape/example/python_modules).

## How JETSCAPE calls a module

| when | method | |
|---|---|---|
| `Init()` | `InitTask()` | once; read your XML parameters here |
| every event | `ExecuteTask()` | gun, sampler, afterburner: the event's work |
| every event, energy loss | `DoEnergyLoss(deltaT, time, Q2, pIn, pOut)` | once per parton per time step |
| every event, jet hadronization | `DoHadronization(shower, hOut, pOut)` | once per event, all showers at once |
| end of every event | `ClearTask()` | drop per-event state |
| `Finish()` | `FinishTask()` | once; close files |

Override only what you need: a method you leave out does what the C++ base class does. Call the
base explicitly when you extend it, e.g. `core.SoftParticlization.ClearTask(self)`.

## The base classes

All are in `jetscape.pyjetscape_core` (below `core`).

| stage | base class | override | gets / gives |
|---|---|---|---|
| initial state | `InitialState` | `ExecuteTask` (+ `GetNcoll`, `GetNpart`, `SampleABinaryCollisionPoint`) | `set_entropy_density_from_numpy`, `set_num_of_binary_collisions_from_numpy` |
| hard process | `HardProcess` | `ExecuteTask` (+ `GetSigmaGen`, `GetSigmaErr`, `GetPtHat`, `GetEventWeight`) | `self.AddParton(...)`, `self.AddHadron(...)` |
| pre-equilibrium | `PreequilibriumDynamics` | `InitializePreequilibrium`, `EvolvePreequilibrium` | `set_e_numpy`, `set_utau_numpy`, ... |
| hydro | `FluidDynamics` | `InitializeHydro`, `EvolveHydro` | `store_fluid_cells_from_numpy`, the freeze-out surface ([example](fno.md)) |
| energy loss | `JetEnergyLoss` | `DoEnergyLoss(deltaT, time, Q2, pIn, pOut)` | the parton in `pIn`; the partons at `time` out |
| jet hadronization | `Hadronization` | `DoHadronization(shower, hOut, pOut)` | one list of final partons per shower; hadrons out |
| soft particlization | `SoftParticlization` | `ExecuteTask` | `self.GetHydroHyperSurface()`; `self.set_hadron_list([[Hadron, ...], ...])` |
| afterburner | `Afterburner` | `ExecuteTask`, `GetCurrentHadronList` | `self.GatherAfterburnerHadrons()` |
| writer | `JetScapeWriter` | `GetStatus`, `Write(obj)`, `WriteEvent` | every `Parton`, `PartonShower`, `Hadron` the tasks write; `self.GetHeader()` |
| liquefier | `LiquefierBase` | `smearing_kernel(tau, x, y, eta, droplet)` | `[j^tau, j^x, j^y, j^eta]` |
| shower growth | `PartonShowerGenerator` | `DoShower(jloss)` | |

[`toy_modules.py`](https://github.com/jhputschke/js-contrib/blob/main/contribs/PyJetscape/example/python_modules/toy_modules.py)
has one template per stage in the plain binding style (`ToyGun`, `ToySplit`, `ToyHadronizer`,
`ToySampler`, `ToyAfterburner`, `ToyWriter`);
[`toy_modules_pythonic.py`](https://github.com/jhputschke/js-contrib/blob/main/contribs/PyJetscape/example/python_modules/toy_modules_pythonic.py)
has the same in the Pythonic style used below. The toys are not physics; they show the
interface.

## A worked example: an energy-loss module

Three Python modules make a complete jet event: a gun that fires two back-to-back 40 GeV gluons,
an energy-loss module that splits every parton above `e_min` into two halves once
`time >= t_split`, and a hadronizer that turns every final parton into a π⁺. The classes are
those of `toy_modules_pythonic.py`.

```python title="my_modules.py"
import jetscape as js
from jetscape import pyjetscape_core as core


@js.module("CustomModuleToyGunV2", section="Hard")
class ToyGunV2(core.HardProcess):
    """Two back-to-back gluons of E (GeV) along x."""

    xml_parameters = {"E": 40.0}

    def ExecuteTask(self):
        position = (0.0, 0.0, 0.0, 0.0)
        ini = js.JetScapeSignalManager.Instance().GetInitialStatePointer()
        if ini is not None and ini.GetNcoll() > 0:
            position = ini.sample_binary_collision_point()        # (t, x, y, z)
        for sign in (+1, -1):
            self.AddParton(js.make_parton(21, (self.E, sign * self.E, 0.0, 0.0), position))

    def GetSigmaGen(self):
        return 1.0


@js.module("CustomModuleToySplitV2", section="Eloss")
class ToySplitV2(core.JetEnergyLoss):
    """A parton above e_min splits into two halves at t >= t_split."""

    xml_parameters = {"t_split": 1.0, "e_min": 30.0}

    def DoEnergyLoss(self, deltaT, time, Q2, pIn, pOut):
        p = pIn[0]
        if time >= self.t_split and p.e() > self.e_min:
            E, px, py, pz = p.momentum
            return [js.make_parton(p.pid(), (E / 2, px / 2, s * 1.0, pz / 2),
                                   position=p.position) for s in (+1, -1)]


@js.module("CustomModuleToyHadronizerV2")
class ToyHadronizerV2(core.Hadronization):
    """Every final parton becomes a pi+ with its momentum and position."""

    def DoHadronization(self, shower, hOut, pOut):
        return [js.make_hadron(211, p.momentum, p.position, stat=1)
                for partons in shower for p in partons]
```

What the pieces do:

- **`@js.module(name, section)`** registers the class in JETSCAPE's module factory, so an XML can
  name it, and turns `xml_parameters` into XML parameters (next sections). Inside the module they
  are plain attributes: `self.E`, `self.t_split`, `self.e_min`.
- **`DoEnergyLoss` returns the partons at `time`.** Returning `None` (here, below `e_min` or before
  `t_split`) lets the parton go on unchanged; returning two partons makes a splitting vertex.
- **`DoHadronization` returns the hadrons.** `js.make_parton` / `js.make_hadron` take the momentum
  as `(E, px, py, pz)` and the position as `(t, x, y, z)`.

The run builds the pipeline by hand: the energy-loss and hadronization *managers* and their
containers are made in Python and the modules put inside, as the XML would do.

```python title="run_my_modules.py"
import numpy as np
import jetscape as js
from jetscape import paths
from jetscape import pyjetscape_core as core
from my_modules import ToyGunV2, ToySplitV2, ToyHadronizerV2

eloss_manager = core.JetEnergyLossManager()   # copies `eloss` once per shower
eloss = core.JetEnergyLoss()                  # the per-shower container (deltaT, maxT)
eloss.Add(ToySplitV2())                       # modules in order (Matter before LBT)
eloss_manager.Add(eloss)

hadronization_manager = core.HadronizationManager()
hadronization = core.Hadronization()
hadronization.Add(ToyHadronizerV2())
hadronization_manager.Add(hadronization)

tasks = [ToyGunV2(), eloss_manager, hadronization_manager]     # pipeline order
main_xml = paths.xscape_root() / "config" / "jetscape_main.xml"

with js.per_event(main_xml, "my_user.xml", tasks=tasks) as driver:
    for i in driver.events():
        fp = driver.energy_loss_manager.final_parton_array()
        had = driver.hadronization_manager.hadron_array()
        print(f"event {i}: {len(fp)} final partons, E = {fp['E'].sum():.1f} GeV;",
              f"{len(had)} hadrons, pT = {np.hypot(had.p[:, 1], had.p[:, 2]).round(2)}")
```

```xml title="my_user.xml"
<?xml version="1.0"?>
<jetscape>
  <nEvents>3</nEvents>
  <vlevel>0</vlevel>
  <Random><seed>1</seed></Random>
  <enableAutomaticTaskListDetermination> false </enableAutomaticTaskListDetermination>
  <Eloss>
    <deltaT>0.1</deltaT>
    <maxT>2</maxT>
  </Eloss>
</jetscape>
```

```console
$ python run_my_modules.py
event 0: 4 final partons, E = 80.0 GeV; 4 hadrons, pT = [20.02 20.02 20.02 20.02]
event 1: 4 final partons, E = 80.0 GeV; 4 hadrons, pT = [20.02 20.02 20.02 20.02]
event 2: 4 final partons, E = 80.0 GeV; 4 hadrons, pT = [20.02 20.02 20.02 20.02]
```

Each 40 GeV gluon split once into two 20 GeV halves with ±1 GeV along y. Now set a parameter
from the user XML: add

```xml
    <CustomModuleToySplitV2><e_min>50</e_min></CustomModuleToySplitV2>
```

inside `<Eloss>`. The 40 GeV gluons stay below `e_min`, nothing splits, and every event prints
`2 final partons ... 2 hadrons, pT = [40. 40.]`.

### The same modules named in the XML

With the automatic task list (the default), the user XML names the modules, and X-SCAPE builds
the managers itself. Import the module file so the decorators register the classes, then run:

```xml title="my_user_xml.xml"
<?xml version="1.0"?>
<jetscape>
  <nEvents>3</nEvents>
  <vlevel>0</vlevel>
  <Random><seed>1</seed></Random>
  <Hard>
    <CustomModuleToyGunV2><E>50</E></CustomModuleToyGunV2>
  </Hard>
  <Eloss>
    <deltaT>0.1</deltaT>
    <maxT>2</maxT>
    <CustomModuleToySplitV2><e_min>30</e_min></CustomModuleToySplitV2>
  </Eloss>
  <JetHadronization><name>CustomModuleToyHadronizerV2</name></JetHadronization>
</jetscape>
```

```python
import jetscape as js
from jetscape import paths
import my_modules  # noqa: F401  (the decorators register the modules)

main_xml = paths.xscape_root() / "config" / "jetscape_main.xml"
with js.per_event(main_xml, "my_user_xml.xml") as driver:
    print([type(t).__name__ for t in driver.GetTaskList()])
    # ['ToyGunV2', 'JetEnergyLossManager', 'HadronizationManager']
    for i in driver.events():
        fp = driver.energy_loss_manager.final_parton_array()
        print(i, len(fp), fp["E"].sum())          # 4 partons, 100 GeV: E is 50 here
```

C++ and Python modules mix freely in both styles: replace `ToyGunV2` by `<Hard><PythiaGun>`,
or put your energy-loss module after `<Matter>` in `<Eloss>`.

## Named in the XML or built by hand

| | named in the XML | built by hand |
|---|---|---|
| user XML | `<enableAutomaticTaskListDetermination>` true (the default) | false |
| the module | registered (`@js.module` or `register_module`), named in its section | an object you `Add()` |
| managers | made by X-SCAPE | `JetEnergyLossManager` / `JetEnergyLoss`, `HadronizationManager` / `Hadronization` made in Python |
| examples | `run_from_xml.py` + `user_from_xml.xml` | `run_by_hand.py` (plain bindings), `run_by_hand_pythonic.py` (`--style old|new`) |

**Names.** Every XML section accepts a module whose element name contains `CustomModule`:
`<Hard><CustomModuleMyGun>`, `<Eloss><CustomModuleMyLoss>`,
`<SoftParticlization><CustomModuleMySampler>`, `<Afterburner><CustomModuleMyAfterburner>`. Jet
hadronization takes it as text: `<JetHadronization><name>CustomModuleMyHadronizer</name>`. A
module that does not contain `CustomModule` in its name is never created from the XML.

**Every user-XML tag needs a default in the main XML.** At `Init()`, X-SCAPE compares every tag
of the user XML with `config/jetscape_main.xml` and **stops the process** (`exit`) on a tag the
main XML lacks: your module's name and each of its parameters. For a C++ module that means adding
default blocks to `jetscape_main.xml` (js-contrib's
[README](https://github.com/jhputschke/js-contrib/blob/main/README.md#xml-configuration-requirements-for-new-modules)
explains it). A Python module needs no edit: registering it gives the defaults, and `Init()` adds
`<section><name>defaults</name>` to the main XML **in memory** (the file is not changed).

- With `@js.module`, the defaults are `xml_parameters` (int, float, str or bool). Before your
  `InitTask` runs, each is read into `self.<key>` from `[section, name, key]`: the user XML's
  value, else the default, converted to the default's type (`bool`: `0`, `false`, `off`, `no` are
  false).
- With the plain call, give the defaults as XML text and read them yourself in `InitTask`, as a
  C++ module does:

    ```python
    class ToyGun(core.HardProcess):
        def InitTask(self):
            self.e = self.get_xml_element_double(["Hard", "CustomModuleToyGun", "E"], False)

    core.register_module("CustomModuleToyGun", ToyGun, section="Hard", defaults="<E>40</E>")
    ```

  `get_xml_element_int` and `get_xml_element_text` work alike; with `True` a missing element
  raises.
- `core.registered_modules()` lists the factory (the build's C++ modules and the Python ones),
  `core.unregister_module(name)` removes a Python one.

A module built by hand needs no registration, but register it anyway if it reads XML
parameters: without the main-XML default, setting them in the user XML stops the run.

## Energy loss in detail

`DoEnergyLoss(deltaT, time, Q2, pIn, pOut)` is called for every parton of every shower at every
time step up to `<Eloss><maxT>`, in steps of `<Eloss><deltaT>` (fm/c).

- **`time`** is the end of the step. **`Q2`** is what the framework passes: the parton's pT, not
  a virtuality; the virtuality is `pIn[0].virtuality()`.
- **`pIn`** is a list with this step's parton, the C++ object itself: `p.set_stat(...)` changes it
  in place. It is valid during the call only; do not keep it.
- **The output** (appended to `pOut`, or returned as a list):

    | output | the shower |
    |---|---|
    | empty / `None` | the parton (as changed) goes on unchanged into the next step |
    | one parton | it replaces the parton (no new vertex) |
    | two or more | a vertex at `time`; each parton is a new branch |

    Partons with the liquefier's statuses (droplet −11, miss −13, negative −17) and non-partons
    (photons, leptons) do not go on. Return a list *or* append, not both (`ValueError`).
- **Several modules** in `<Eloss>` (Matter then LBT, or yours after Matter) are called in turn on
  the same lists, as in C++. Leave alone a parton another module handles
  (`p.GetController()`, `pOut` already filled).
- **The medium:** `self.GetHydroCell(t, x, y, z)` returns a `FluidCellInfo` (`temperature`,
  `energy_density`, `pressure`, `vx`, `vy`, `vz`, ...), or `None` without a hydro.
- **Copies:** the energy-loss manager copies the module once per shower at the start of each
  event with `Clone()` and drops the copies at the end. The default copy runs the C++ copy
  constructor and copies the instance attributes *shallowly*; `__init__` and `InitTask` are not
  called on copies. Override `Clone` for state that must not be shared between showers:

    ```python
    def Clone(self):
        copy = core.JetEnergyLoss.Clone(self)   # the default copy
        copy.history = []                       # one list per shower
        return copy
    ```

## The other stages

- **Jet hadronization**, `DoHadronization(shower, hOut, pOut)`: once per event, after the energy
  loss. `shower` is a list with one list of final partons per shower. Return (or append to `hOut`)
  the hadrons, or a tuple `(hadrons, partons)` for partons you leave unhadronized; for hadrons
  alone return a list, not a tuple. `self.GetHydroHyperSurface(T)` and `self.GetHydroCell(...)`
  give the medium.
- **Soft particlization**, `ExecuteTask()`: `self.GetHydroHyperSurface()` is the freeze-out
  surface (a list of `SurfaceCellInfo`); store hadrons with `self.set_hadron_list(samples)`, a
  list of oversamples, each a list of `Hadron`s.
- **Afterburner**, `ExecuteTask()` and `GetCurrentHadronList()`: `self.GatherAfterburnerHadrons()`
  gives per oversample the soft hadrons, plus the jet hadrons when
  `<Afterburner><include_fragmentation_hadrons>` is 1.
- **Writer**, `GetStatus()` and `Write(obj)` (+ `WriteEvent()`): per event the header is filled
  (`self.GetHeader().GetSigmaGen()`, ...), then `Write` receives the hard partons, per shower the
  initiating `Parton` and the `PartonShower`, then the jet `Hadron`s. Add a writer with
  `driver.Add(writer)` before `Init()`, also in XML mode. The C++ ASCII writer comes from
  `core.create_module("JetScapeWriterAscii")` + `core.set_writer_output_file(w, name)`, and
  `core.JetScapeReaderAscii(name)` reads its file back.
- **Initial state, pre-equilibrium, hydro, liquefier**: examples in
  `example/inspect_bulk_info.py` (`GaussianIC`), `python/jetscape/fno_hydro.py`
  ([PyFNOHydro](fno.md)), `python/jetscape/surface_replay.py`, FastHydro's `hydro.py` and
  `initial_state.py` ([FastHydro](fasthydro.md)).

## Particles and lists

- `js.make_parton(pid, momentum, position=(0, 0, 0, 0), stat=0, label=0, color=0, anti_color=0)`,
  `js.make_hadron(pid, momentum, position, stat, label, mass=None)` and `js.make_photon(...)`
  take momentum `(E, px, py, pz)` and position `(t, x, y, z)`. The bound constructors keep C++'s
  order: `core.Parton(label, pid, stat, p, x)` with `p = core.FourVector(px, py, pz, E)` and
  `x = core.FourVector(x, y, z, t)`.
- Getters and setters as in C++: `pid()`, `pstat()`, `e()`, `px()`, `pt()`, `eta()`, `p_in()`,
  `x_in()`, `set_stat()`, `set_form_time()`, `set_color()`, ...; and `p.momentum`, `p.position`.
- A particle's `x()`, `y()`, `z()`, `t()` are its **position**; a parton's virtuality is
  `virtuality()`, set with `set_t()`.
- Lists C++ passes by reference arrive as Python lists: the inputs are the C++ objects (valid
  during the call), the outputs are copied back into C++ after the call.

## Random numbers

- **`self.rng`** (Pythonic layer) is a numpy `Generator` seeded from the module's own C++ engine on
  first use: reproducible with `<Random><seed>`, with its own stream for each per-shower copy.
  Use it in `InitTask` or later; before the XML is loaded it raises.
- **`self.GetMt19937Generator()`** is the engine itself, as in C++: `g()` is the next 32-bit
  draw, `g.uniform()` a draw in [0, 1).

The engines are seeded from the seed and the module's task number. Adding a Python module shifts
the task numbers, and so the random streams, of the tasks created after it, exactly as adding a
C++ module does.

## Checking a module against C++

Run the C++ module and your Python version with the same seed and compare the output files, as
[`tests/equivalence_driver.py`](https://github.com/jhputschke/js-contrib/blob/main/contribs/PyJetscape/tests/equivalence_driver.py)
does. For identical random numbers the Python version must create the same tasks in the same
order; a Python module that holds a C++ module of its own gives it its task number back with
`core.next_task_number()` / `core.set_next_task_number()` (see `PyMatter` there). Run each
configuration in its own process: JETSCAPE keeps the XML, the seed and the module registry in
process-wide singletons. Checked this way, a Python module handing each step to C++ Matter gives
byte-identical hadrons to the C++ run, and a Python writer writes the same lines as the C++ one.

## Cost

A call into a Python energy-loss module costs about 1.4 µs on top of the module's own work: on a
GB10, 715,438 calls in 500 events of a 100 GeV gluon in vacuum took 3.70 s through a Python module
that hands each step to C++ Matter, against 2.72 s for Matter alone. A module written in Python
costs what its Python code costs on top of that, per parton and time step. Python modules run in
one thread (each call takes the GIL).

## Pitfalls

| symptom | cause |
|---|---|
| `User XML tag <X> is unrecognized` and the run stops | the module or a parameter is not registered (`@js.module` / `register_module` with `section=` and defaults) before `Init()` |
| a module named in the XML never runs | its name does not contain `CustomModule` |
| `AttributeError: module 'jetscape' has no attribute 'JetEnergyLoss'` | most bound classes are in `jetscape.pyjetscape_core`, not re-exported by `jetscape` |
| a parton's `t()` is not its virtuality | `t()` is the position time; use `virtuality()` |
| state shared between showers | the default `Clone()` copies attributes shallowly; override `Clone` |
| a liquefier attached from Python is gone | C++ keeps only a weak reference; keep the Python object alive |
| `undefined symbol ... iColoredHadronization` on import | X-SCAPE built without `-DUSE_3DGlauber=ON` |
| `all 256 slots for Python modules are taken` | more than 256 registered Python modules in one process |

## Next steps

- [Live PyTorch FNO hydro](fno.md): a complete Python hydro module backed by a neural network.
- [The Pythonic interface](pythonic.md): reading what your module produced.
- The full reference: PyJetscape's
  [README, *Writing modules in Python*](https://github.com/jhputschke/js-contrib/blob/main/contribs/PyJetscape/README.md#writing-modules-in-python)
  and [`example/python_modules/README.md`](https://github.com/jhputschke/js-contrib/blob/main/contribs/PyJetscape/example/python_modules/README.md).
