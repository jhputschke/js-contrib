<!-- Design note, 2026-10-24. Branches `pyjetscape_full` in X-SCAPE (from `contrib` 41bdf8c5)
     and js-contrib (from `main` 03d14fa), worktree ~/JetScape/X-SCAPE-pybind, build
     `build_pybind`. No PR until the physics is vetted (section "Validation"). -->

# Plan: a complete JETSCAPE binding in PyJetscape, so modules can be developed in Python

## Status

- 2026-10-24: plan written; branches and worktree made (`build_pybind` needs
  `-DUSE_3DGlauber=ON`, as `build_gpu`: without it iColoredHadronization is not compiled and
  pyjetscape_core does not load; the old tests then skip their core parts silently).
- **Phase 0 done (2026-10-24):** module classes on `py::smart_holder` (D1); `register_module`,
  `unregister_module`, `registered_modules` (D2, 256 slots); `FourVector`,
  `JetScapeParticleBase`, `Parton` (constructors, setters, colours, formation time,
  `virtuality()`), `Photon`, `Hadron`; `Mt19937` (a task's engine), `random_seed()`,
  `GetMyTaskNumber`; the task hooks (`InitTask`, `ExecuteTask`, …) overridable on
  `JetScapeModuleBase`; `GetTaskAt`/`GetTaskList`/`create_module` return the most derived bound
  type. `JetScape.Init()` adds an empty main-XML default for each registered Python module
  (JetScape exits on a user tag the main XML lacks).
- **Phase 1, energy loss done (2026-10-24):** `JetEnergyLoss` trampoline (`DoEnergyLoss`,
  `Clone`, `InitTask` (no-op default for a module), `ClearTask`, `FinishTask`),
  `GetHydroCell`, the time settings, `DoEnergyLoss`/`Clone` callable on C++ modules. A Python
  module from the XML (`<Eloss><CustomModulePySplit>`) runs, is copied per shower, and its
  output builds the shower. Tests: `tests/test_bindings_core.py`, `test_bindings_eloss.py`;
  suite 281 passed, 2 skipped.
- **Phase 1 done (2026-10-24):** trampolines for `HardProcess` (a Python gun: `ExecuteTask`,
  `AddParton`, the sigma/pTHat/weight virtuals), `PreequilibriumDynamics`
  (`InitializePreequilibrium`, `EvolvePreequilibrium`, `set_<field>_numpy`), `Hadronization`
  (`DoHadronization`, lists in and out; the container and `HadronizationManager` bound),
  `SoftParticlization` (`set_hadron_list`, `GetHydroHyperSurface`), `Afterburner`
  (`GetCurrentHadronList`, `GatherAfterburnerHadrons` through a publicist), `JetScapeWriter`
  (one Python `Write(obj)` for Parton, Vertex, PartonShower, Hadron, str; `JetScapeEventHeader`)
  and `LiquefierBase` (`smearing_kernel` and the deposition hooks). One run with every stage in
  Python (gun, energy loss, jet hadronization, soft particlization, afterburner, writer) passes
  (`tests/test_bindings_stages.py`); suite 289 passed, 2 skipped.
  - Found: with `py::smart_holder` a Python-derived object handed to C++ gets a new shared_ptr
    control block per conversion, so a C++ `weak_ptr` to it (the liquefier in the hydro and the
    energy loss, the signal manager's pointers) expired at once. Those setters now pass
    `py::potentially_slicing_weak_ptr<T>(obj).lock()` (the holder's block): the weak pointer
    lives as long as the Python object, as with JetScape owning its C++ liquefier.
- **Phases 2 and 3 done (2026-10-24):** `IsrManager`, `IsrJet`, `iMATTER`, `IsrShowerPSG` (the ISR
  chain by hand), `PartonShowerGenerator` (subclassable) and `JetEnergyLoss.AddPartonShowerGenerator`,
  `JetScapeReaderAscii(GZ)`; the C++ writers come from `create_module` as `JetScapeWriter`;
  `register_module(name, cls, section=, defaults=)`: the main-XML default entry with the module's
  parameters (JetScape also refuses a user parameter tag the main XML lacks); `run_manual`'s
  stage order covers every stage. `PartonShower`: `final_partons`, `partons`, `children`,
  `parents`, `production_vertex`, `end_vertex`, `edge_of`, the split variables, `new_vertex` /
  `new_parton` (a shower built in Python); `Vertex` constructible. `example/python_modules/`:
  one toy module per stage and the same all-Python pipeline from the XML and by hand, each with
  the C++ ASCII writer read back (`tests/test_bindings_pipeline.py`); suite 296 passed, 2 skipped.
- **Validation (2026-10-24, GB10), against X-SCAPE-regbase (`contrib` 41bdf8c5 + js-contrib
  `main` 03d14fa, `build_gpu_base`), the base of `pyjetscape_full`:**
  - V1, C++ physics unchanged: `run_prod_jet.py --events 2 --seed 1 --reuse 2
    --write-particlize both`: the particlize file identical, the pair file identical except
    `source/flux` (max 2e-16, the 1-ulp liquefier noise two runs of one build show);
    `run_prod_oo_isr.py --events 6 --seed 101 --store-hadronization-input` (ISR, iMATTER,
    iColored): jet hadrons (with the Pythia input) and particlize file identical.
  - V2: `tests/test_bindings_*.py` (core, eloss, stages, pipeline, equivalence).
  - V3, Python modules reproduce C++ ones byte for byte (`tests/test_bindings_equivalence.py`,
    `equivalence_driver.py`: 100 GeV gluon, Matter in vacuum, Colorless, seed 7, 20 events; also
    checked for 500): PGun ported to Python; a Python energy-loss module handing every step to a
    C++ Matter of its own (its inner Matter given the task number a C++ run has there,
    `next_task_number` / `set_next_task_number`; its `Clone` copies the Matter too); a Python
    writer in the C++ final-state format. The control, the inner Matter on a task number of its
    own, differs in 2718 of 2688 + 2688 lines: the comparison can fail.
  - V4, cost: 500 events, 715,438 `DoEnergyLoss` calls: C++ Matter 2.72 s, through the Python
    module 3.70 s, i.e. about 1.4 µs per call for C++ -> Python -> C++ (two runs each).
  - Not done: a Python hadronization through Pythia's Python package against
    ColorlessHadronization (a port of its string building, not a binding question).
- **Phase 5 done (2026-10-24):** PyJetscape README "Writing modules in Python" (the stage table,
  registration and the XML, ownership, copies and seeds, lists, particles, showers and output,
  exactness and cost, limits), the build note `-DUSE_3DGlauber=ON`, Mode B with hadronization;
  `InitialState`'s trampoline extended (task hooks, Npart/Ncoll/entropy/centrality,
  `SampleABinaryCollisionPoint`). Suite 302 passed, 2 skipped.
- Open: phase 4 (module-specific access where the XML is not enough) when a need comes up;
  thread safety (one thread assumed); the PR split (see "Open").

## Goal

Everything a C++ JETSCAPE developer can do should be possible from Python:

1. **Write a module in Python for every stage**: initial state, hard process, pre-equilibrium,
   hydro, jet energy loss, jet hadronization, soft particlization, afterburner, writer, liquefier.
   The framework must create, own, copy and call it like a C++ module, also when the XML builds
   the task list.
2. **Build any pipeline in Python** (Mode B), including the managers the XML builds today
   (energy loss, hadronization, ISR, writers).
3. **Read and change every object a module sees**: partons, hadrons, vertices, showers, fluid
   cells, the hard process, the random seeds.

Not a goal: changing any C++ physics. A run of C++ modules gives the same output with this
build as without it.

## What exists (2026-10-24, js-contrib `main` 03d14fa)

- Any compiled C++ module runs through the XML (Modes A and C) or `create_module(name)`.
- Python subclasses only of `JetScapeModuleBase`, `InitialState` and `FluidDynamics`
  (trampolines with the `std::shared_ptr` holder: the Python object must be kept alive by the
  caller, so C++ can not create or copy one).
- Module-specific bindings: `MpiMusic`, `TrentoInitial`, `CausalLiquefier`, `FastRootBulkWriter`.
- `Parton`, `Vertex`, `PartonShower` read-only (no constructor, no setters); no `FourVector`,
  `Hadron`, `HardProcess` classes; hadrons, hard partons and final partons as numpy.
- `JetEnergyLoss` and `JetEnergyLossManager` constructible, with almost no methods.
- Not bound: hadronization classes and managers, `HardProcess`, `IsrManager`, `Afterburner`,
  `SoftParticlization`, the writers and the reader, the random generators.

## Design decisions

- **D1. `py::smart_holder` for every module class** (pybind11 3.0.3 in `js_fno`), trampolines
  derive from `py::trampoline_self_life_support`. Then a Python module handed to C++ stays alive
  as long as C++ holds it, which the factory (D2) and `Clone()` (D4) need. All classes of one
  hierarchy get the same holder; the conversion is one step for `JetScapeTask` and everything
  bound below it.
- **D2. Python modules in the module factory.** `register_module(name, cls)` puts a creator for
  a Python class into `JetScapeModuleFactory`, so `create_module(name)` and the XML make it. The
  factory's map holds plain function pointers, so the binding keeps a pool of creator slots
  (one per registered name) and reaches the protected map through a subclass: no X-SCAPE
  change. The XML's automatic task list takes a module whose element name contains
  `CustomModule` in every section (`JetScape.cc`), so a Python module meant for the XML is
  registered under such a name. JetScape exits on a user-XML tag the main XML lacks; the main
  XML is parsed only once, so the binding's `JetScape.Init()` opens it first and adds an empty
  default element for each registered Python module (2026-10-24).
- **D3. The Python API keeps the C++ names and argument order** (`DoEnergyLoss(deltaT, time,
  Q2, pIn, pOut)`, `DoHadronization(shower, hOut, pOut)`, `ExecuteTask()`, …), so a C++ module
  ports line by line and the C++ documentation applies. (2026-10-24) Containers passed by
  reference reach Python as plain lists, not opaque types (an opaque `vector<shared_ptr<Parton>>`
  would change what existing functions return): the inputs as references to the C++ objects
  (changed in place, valid during the call only), the outputs as lists the module appends to,
  copied back after the call (`src/pyjs_modules.h`). One exception to the C++ names, kept from
  the first bindings: a particle's `x()`, `y()`, `z()`, `t()` are its position (`jetscape.showers`
  writes the shower initiators with them); the virtuality is `Parton.virtuality()` /
  `set_t()`.
- **D4. `Clone()` of a Python module**: the energy-loss and hadronization managers copy their
  modules per shower / per event through `Clone()`. Default (2026-10-24): a new instance of the
  same Python class made by the base's C++ copy constructor, as a C++ module's implicit copy
  calls it (same id and settings; a new task number registered in the same order, so the
  random seeds of all tasks stay those of a C++ run), then a shallow copy of `__dict__`; the
  Python `__init__` is not called. A module can override `Clone`.
- **D5. Medium access**: a Python energy-loss module gets `GetHydroCell(t, x, y, z)` (its
  `GetHydroCellSignal`, as Matter and LBT use it); the hydro pointer stays available through
  `JetScapeSignalManager`.
- **D6. Random numbers**: Python modules read their task's seed (`JetScapeTaskSupport`, one
  generator per task) and seed their own generator (numpy) from it, so runs stay reproducible
  and independent of other tasks. The C++ generators are not shared with Python.
- **D7. X-SCAPE changes only where a binding can not work around the C++ API**; each is listed
  in its phase and is physics-neutral. Bindings that need a newer X-SCAPE are guarded by a
  macro, as `JETSCAPETASKSUPPORT_HAS_NEXT_TASK_NUMBER` is today.

## Phases

**Phase 0 — foundations.**
- D1 holder conversion; D2 `register_module`, `registered_modules()`.
- `FourVector`; `JetScapeParticleBase` getters (pid, pstat, plabel, p, x, pt, eta, phi,
  rapidity, mass); `Parton` with constructors and every setter the C++ modules use (stat,
  label, momentum, position, virtuality `t`, form time, colours, max/min colour, jet velocity);
  `Hadron` (constructors, charge, baryon number, strangeness, decay width); `Photon`.
- Opaque `PartonList`, `HadronList`, `PartonPtrList`, `HadronPtrList`.
- Seeds: `random_seed()`, a task's seed (D6).

**Phase 1 — a Python module for every stage (trampolines).**
`HardProcess` (a Python gun: `ExecuteTask` + `AddParton`, `AddHadron`, sigma, pTHat, weight),
`PreequilibriumDynamics`, `JetEnergyLoss` modules (D3–D5), `Hadronization` modules,
`SoftParticlization`, `Afterburner` (`GetCurrentHadronList`, the hadrons it receives),
`JetScapeWriter` (`Write(...)` per object type, `WriteEvent`), `LiquefierBase`.

**Phase 2 — pipelines built in Python.** `HadronizationManager` and its `Hadronization` task,
`IsrManager`, the C++ writers (ASCII, gz, HepMC, final-state, Qn) with their file names,
`JetScapeReader`; `create_module` returns the most derived bound type for every class above. A
Mode B example with every stage.

**Phase 3 — every object, not only numpy.** The hard process (partons, hadrons, remnants,
weights), the hadron lists of the hadronization and the afterburner, `PartonShower` navigation
(parents, children, edges ↔ vertices). Keep the numpy accessors (fast paths).

**Phase 4 — module-specific access** where the XML is not enough (to be listed per need, e.g.
PythiaGun's Pythia settings after `Init`, Matter's and LBT's per-parton state).

**Phase 5 — documentation**: a "Writing modules in Python" section in PyJetscape's README, one
example per stage, the API table, the performance note (V4).

## Validation (before any PR)

- **V1. No change to C++ physics.** The production regression recipe (`run_prod_jet.py --events 2
  --seed 1 --reuse 2 --write-particlize both`) and one O+O job (`run_prod_oo_isr.py`), with
  `build_pybind` against `build_gpu_base` / a contrib build: identical outputs (the known 1-ulp
  `source/flux` noise excepted).
- **V2. Unit tests per binding** (pytest): construction, override dispatch, lifetime (a module
  made by the XML or the factory survives its Python creator; `Clone()`), in-place containers.
- **V3. Physics equivalence of the Python path.** Python modules that must reproduce C++ ones
  with the same seeds:
  - a Python energy-loss module that delegates to C++ Matter (and Matter + LBT): showers
    bit-identical to C++ alone (tests `Clone`, the hydro signal, `pIn`/`pOut`, the per-shower
    copies);
  - a Python gun with PGun's fixed kinematics: identical showers downstream;
  - a Python hadronization through Pythia's Python package with ColorlessHadronization's
    settings and seed: identical hadrons, else equal distributions within errors;
  - a Python writer next to the C++ ASCII writer: identical content.
- **V4. Cost.** The time per event of the Python energy-loss path against C++ (the module is
  called per parton per time step), stated in the README.

## Open

- Thread safety: the energy-loss manager's multi-threaded mode is commented out in X-SCAPE;
  Python modules assume one thread (the GIL is taken in every trampoline).
- Whether `pyjetscape_full` should later be split into smaller PRs (foundations, then one per
  stage) for review.
