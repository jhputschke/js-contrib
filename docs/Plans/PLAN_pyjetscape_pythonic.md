<!-- Design note, 2026-10-26 (outline as a claude.ai artifact, "PyJetscape Pythonic Layer");
     implemented 2026-10-27. Branches `pybind11_pin` and `pythonic_api`, both from js-contrib
     `main` 7a97d60 (after #73 pyjetscape_full and #74 lbt_colored_hadro). X-SCAPE unchanged
     (`contrib` 38e90519). Worktree ~/JetScape/X-SCAPE-pybind, build `build_pybind`
     (-DUSE_3DGlauber=ON). -->

# Plan: a Pythonic layer over the PyJetscape bindings

An additive, more Pythonic interface on top of the bindings of #73. The C++-mirroring API
(PLAN_pyjetscape_complete.md, D3) stays as it is, so every existing script, toy module and
the V3 equivalence ports keep working. Most of the layer is pure Python installed onto the
bound classes; only the two module trampolines change in C++. No X-SCAPE change.

## Status

- **2026-10-27, implemented and validated** (GB10, X-SCAPE `contrib` 38e90519 with
  iColored's colour closure, pybind11 3.0.3):
  - Step 0 on `pybind11_pin` (its own PR, first).
  - Steps 1–8 on `pythonic_api`: `python/jetscape/pythonic.py` (installer, module methods,
    protocols, `@module`, `per_event`, `ShowerNode`), `records.py`, `hadron_array.py`,
    `awkward_views.py`, `particles.py`, the trampoline change (`pyjs_modules.h`,
    `bind_jet.cc`, `bind_stages.cc`), `pyjetscape_core.pyi` + `py.typed` with
    `utils/gen_pyjetscape_stubs.py`, `example/python_modules/toy_modules_pythonic.py` and
    `run_by_hand_pythonic.py`, the README section "Pythonic interface";
    tests `tests/test_pythonic.py` (26) and `tests/test_pythonic_run.py` (4).
  - Validation V0–V6 below: all passed.
- **2026-10-27, later: the guide and an O+O workflow.** `contribs/PyJetscape/PYTHONIC.md`
  (guide and reference: conventions, every accessor, records, HadronArray, awkward views,
  `select`, hadron files, FastJet recipes, particles, showers, writing modules, stubs,
  guarantees, cost, pitfalls, old and new side by side);
  `example/pythonic/oo_workflow.ipynb` (3 O+O events through the whole chain with
  `js.per_event`, both hadron files written from `HadronArray`s and read back, awkward,
  FastJet per event and per oversample, a jet embedded in the bulk; stored with its outputs,
  the modules' log removed). For it: `HadronArray.from_h5` / `to_dict()` /
  `to_awkward(by_unit=True)` (`x`, `mass` may be None), `records.initiator_rows`,
  `events(on_error=)`, `ExecInit()` on entering the driver. Found on the way:
  `driver.energy_loss_manager` returned the ISR chain's `IsrManager` (a
  `JetEnergyLossManager` inside the hard process); now the signal manager's final-state
  manager first, and the task-list fallback skips `IsrManager`. Tests: 28 + 5.
- Open: the follow-ups (hist, HepMC3) and the open questions at the end.

## Rules every step is held to

- **Never change an existing function's output.** `final_partons_numpy`,
  `PartonShower.to_numpy`, `vertices_to_numpy`, the hadron dicts and `hadronize_partons`
  keep their column orders and dtypes (the files store `FINAL_PARTON_COLUMNS` in that order).
- **New names only.** `pythonic._add` refuses any attribute a class (or the core module)
  already has, own or inherited; dunders only from an allow-list (`__iter__`, `__getitem__`,
  `__array__`, `__eq__`, `__ne__`, `__hash__`, `__reduce__`, `__enter__`, `__exit__`) and
  never one a bound class defines. Installing the identical value again is a no-op.
- **No `__len__` on `PartonShower`** (an empty shower would become false); `__iter__` only.
- **No `__eq__` on particles** (they stay hashable by identity). `FourVector` gets a value
  `__eq__` and `__hash__ = None` (set explicitly: adding `__eq__` after class creation does
  not clear the hash). A grep of js-contrib found no hashed FourVector.
- **Module parameters are `xml_parameters`**, not `params` (`LiquefierBase.params()` is
  bound; FastHydro's `DropletBridge` sets `self.params`).
- **Properties such as `p.pt` only through a facade** (`jetscape.particles.view`).
- **Properties on the bound classes are non-data descriptors** (`pythonic._Prop`): an
  instance attribute of the same name wins, so a Python module that sets `self.rng = ...`
  (or `self.info`) keeps working. (Added in the implementation.)

Naming: a method ending in `_array()` returns numpy data, filled once per event; a plain
noun (`showers`, `nodes`, `final_partons()`) returns objects. Momentum `(E, px, py, pz)`,
position `(t, x, y, z)`, the order of `FourVector.comp(i)`. Flat first (numpy + offsets, as
the files), nested on request (`to_awkward()`).

## Memory constraints

- **M1 Hadrons stay one array per field.** With `soft_set_compact_output` iSS's vectors are
  moved into numpy without a copy; a record array would interleave them. `HadronArray`
  wraps the dict's arrays; `to_awkward()` builds the nested view over the same buffers
  (awkward keeps the strided `p` columns as views).
- **M2 No eager derived columns.** `HadronArray` stores none; pt, eta, phi, rapidity come
  from `to_awkward()` (`vector` Momentum4D), computed when used. `hadrons_h5.Hadrons` stays
  for the file readers; `HadronArray.to_hadrons()` converts.

## Step 0: pybind11 ≥ 3.0 (branch `pybind11_pin`, own PR)

`main` uses `py::smart_holder` (since #73), which needs pybind11 ≥ 3.0, but every pin
allowed 2.x. `CMakeLists.txt`: `find_package(pybind11 3.0 CONFIG QUIET)` in both lookups,
the FATAL_ERROR names the version; `requirements.txt`, `pyproject.toml`, the three
Dockerfiles: `pybind11>=3.0`; the two conda install scripts: `"pybind11>=3.0"`;
`test_js_fno_build_env.sh` checks `pybind11>=3.0` on conda-forge, not only the name.

## Steps 1–8 (branch `pythonic_api`)

1. **Installer and array schema.** `pythonic.install()` (called by `jetscape/__init__`; a
   direct `import jetscape.pyjetscape_core` runs it too). Record dtypes in `records.py`
   (`FINAL_PARTON_DTYPE`, `INITIATOR_DTYPE`, `EDGE_DTYPE`, `NODE_DTYPE`, and
   `HARD_PARTON_DTYPE`), filled by reordering the existing `*_numpy` outputs.
   `HadronArray` (pid, pstat, p, x, mass, sample_offsets, info; `len`, `n_samples`,
   `sample_counts`, `to_hadrons()`, `to_awkward()`). `awkward_views.py`: `to_awkward(obj,
   position=False, charge=False)` and `select(arr, species)` (PDG names or the classes
   charged, neutral, hadron, meson, baryon, strange; each distinct pid looked up once with
   `particle`). `hadrons_h5.CHARGED` unchanged.
2. **Methods on the module classes**, wrapping the free functions (which stay):

   | class | new | wraps |
   |---|---|---|
   | `JetEnergyLossManager` | `showers`, `final_parton_array()`, `initiator_array()` | `get_showers`, `final_partons_numpy`, `showers.initiators_from_manager` |
   | `SoftParticlization` | `hadron_array()`, `primary_array()`, `use_compact_output()`, `set_number_of_samples()`, `set_next_random_seed()`, `last_random_seed`, `sample_cells()` | `soft_*` |
   | `HadronizationManager`, `Hadronization` | `hadron_array()`, `hadronize(partons, seed=None)`, `last_random_seed` | `hadronization_hadrons_numpy`, `hadronize_partons`, `jet_hadronization_last_random_seed` |
   | the same, with `ICOLORED_HAS_COLOUR_AUDIT` / `_CLOSURE` | `colour_audit()` / `colour_totals()`, `set_colour_closure()` | `jet_hadronization_colour_*` |
   | `HardProcess` | `parton_array()`, `info` | `hard_partons_numpy`, `hard_process_info` |
   | `InitialState` | `info` | `initial_state_info` |

   `hadronize()` converts records to the (N, 14) array in Python and calls
   `hadronize_partons` unchanged; on a manager it uses the one module it holds.
3. **Python protocols and shower nodes.** `FourVector`: iteration, `[i]`, `__array__` in
   `comp()` order, value `==`, `FourVector.momentum(E, px, py, pz)` /
   `.position(t, x, y, z)`. Particles: `momentum`, `position`; `make_parton`,
   `make_photon`, `make_hadron` (keywords, any 4-sequence or FourVector). `PartonShower`:
   `__iter__`, `nodes`, `start_node(p)`, `end_node(p)`, `edge_array()`, `node_array()`.
   `ShowerNode`: `id`, `position`, `parents`, `children`, `split_z`, `split_kt`,
   `split_delta_r`, `is_one_to_two`, `is_end`. Pickling of `FourVector`, `Parton`,
   `Photon`, `Hadron` (`__reduce__` through the factories).
4. **Writing modules.** `@js.module(name, section)` over `register_module`; the main-XML
   default from `xml_parameters` (int, float, str, bool), read into `self.<key>` before the
   class's `InitTask`, and set as class attributes (a module built by hand has them).
   `self.rng`: a numpy Generator seeded from the task's engine, cached with the task number
   (a `Clone()` gets its own stream). C++: `DoEnergyLoss` and `DoHadronization` keep the
   override's return value: `None` as before; a list is the output (appended after what the
   output held); `(hadrons, partons)` for hadronization; returning and appending raises.
5. **The event loop.** `js.per_event(main_xml, user_xml, tasks=, n_events=)` (a factory;
   the bound constructor stays), `with driver:` (Init / Finish), `driver.events(n=None)`
   (between ExecPerEvent and ClearPerEvent), `energy_loss_manager`,
   `hadronization_manager`, `soft_particlization`, `hydro`, `initial_state`.
6. **Type stubs.** `utils/gen_pyjetscape_stubs.py`: pybind11-stubgen on the built module
   (which sees the installed layer), the layer's properties made typed `@property` stubs,
   unresolvable C++ types as `typing.Any`; writes `pyjetscape_core.pyi` and `py.typed`. The
   stubs follow the build they come from (ROOT, iColored flags).
7. **Particle facade.** `jetscape.particles.view(p)`: properties over the bound methods,
   `unwrap()`.
8. **Documentation.** PyJetscape README "Pythonic interface" (after "Writing modules in
   Python"); `example/python_modules/toy_modules_pythonic.py` (every toy in the new style)
   and `run_by_hand_pythonic.py`; this file.

### Where the implementation differs from the outline

- `col` / `acol` are `uint32` (outline: `i4`): C++ tags are `unsigned int`, and files
  written before the tag fix (X-SCAPE #181) hold tags above 2**31.
- The awkward records nest the position as a `position` field (a `Vector4D` t, x, y, z):
  with flat t, x, y, z fields `vector` takes them as the Momentum4D coordinates, so `.pt`
  and `.E` came from the position. Node records (no momentum) are `Vector4D` themselves.
- `records.py` holds the dtypes (outline: unspecified), and `HARD_PARTON_DTYPE` is new.
- `JetScapePerEvent(main_xml=...)` is the factory `js.per_event(...)`: the outline allowed
  a factory or a subclass, and a subclass exported as `jetscape.JetScapePerEvent` would
  have changed an existing name.
- The layer's functions are also installed on `pyjetscape_core` (`js.make_parton` etc.),
  since `toy_modules.py` binds `js` to the core module.
- `self.rng` raises AttributeError on a C++ module (no `__dict__`) instead of drawing from
  its engine, which would change its random stream (even through `hasattr`).
- A returned hadronization list is appended after what `hOut` held (an earlier module's
  hadrons stay); the outline said "it is the output".
- `ShowerNode.parents` / `children` scan the shower's partons with `edge_of` (the id-based
  bindings have no node-to-edges call).

## Validation (GB10, 2026-10-27)

Baseline: X-SCAPE `contrib` 38e90519 + js-contrib `main` 7a97d60, `build_gpu`; new: the
same X-SCAPE + `pythonic_api`, `build_pybind`.

| | check | result |
|---|---|---|
| V0 | configure against pybind11 2.13.6 and against `js_fno` (3.0.3) | before the pin 2.13 was accepted (the build then fails in templates); with it CMake stops with "pybind11 3.0 or newer (py::smart_holder)". With `js_fno` configure and `pyjetscape_core` build as before |
| V1 | `run_prod_jet.py --events 2 --seed 1 --reuse 2 --write-particlize both` | particlize file 35/35 datasets identical; pair file 39/41, the two others `diag/wall_s` and `source/flux` (6.7e-16, the known 1-ulp noise) |
| V2 | PyJetscape, FastHydro, Visualization suites | PyJetscape 345 passed / 2 skipped (baseline 315 / 2 + 30 new), FastHydro 316 / 8, Visualization 46; same skip reasons |
| V3 | every accessor against the function it wraps (3 events, PGun + Matter + Colorless); `hadronize(records)` against `hadronize_partons(rows)` with one seed, Colorless and iColored (with the audit); `to_awkward()` fields and `ak.num` against the flat arrays; FastJet on the jagged view against a loop over samples | all bit for bit (`tests/test_pythonic_run.py`, `test_pythonic.py`) |
| V3 | `select(h, "charged")` against `hadrons_h5.CHARGED` | equal on a decayed bulk file (8,715 hadrons); on a `keep_primaries` file 1,114 of 5,434 hadrons differ, all charged resonances in `select` and not in `CHARGED` (134 pids; most: ρ± 307, K*± 172, Δ 136, a1± 36), none the other way |
| V4 | iSS on one stored surface (972,215 cells), 1000 oversamples (6,996,105 hadrons): resident memory after the step / peak | dict 686.2 / 915.9 MB; `HadronArray` 686.3 / 915.2 MB (shares `p`); `to_awkward()` + `.pt` 761.4 / 915.3 MB (`p` not copied, `pt` float32; peak unchanged); `Hadrons` 1240.9 / 1240.3 MB (+79 B per hadron, peak +35 %) |
| V5 | additivity | the installer refuses clashes (unit tests); `run_by_hand.py`, `run_by_hand_pythonic.py --style old` and `--style new` give the same SUMMARY line; existing suites unchanged (V2) |
| V6 | #73's benchmark (500 events, 715,438 `DoEnergyLoss` calls through PyMatter), 3 runs each | 3.701 s (base) vs 3.734 s (new), +0.9 % (C++-only runs vary by ~1 %); hadrons byte-identical in all runs |

## Follow-ups (after this PR)

- **`hist` for the analysis classes.** `to_hist(values, axes, mask=, weights=)` beside
  `Hadrons.hist`: the unscaled `hist.Hist` (Weight storage) with N_samples in its metadata
  and a helper that scales, so histograms of many files add first and are scaled once. Check:
  scaled equals `hist(...)` bit for bit; two halves added equal the whole.
- **HepMC3 through `pyhepmc`.** `jetscape/hepmc_io.py`: `read_hadrons(path)` → HadronArray
  (status 1, one sample per event, weights and cross section in `info`); `write_hepmc(file,
  path)` for Rivet (one event per oversample, `GenHeavyIon` from the stored initial state;
  decide the oversample weight first). Check that pyhepmc installs on aarch64 first.

## Open questions

- The `_array()` naming rule: keep, or plain nouns for arrays and rename the object
  accessors instead?
- `FourVector` value equality makes it unhashable: keep, or an explicit `isclose()`?
- Installing from Python (no rebuild per change; stubgen sees it, step 6) against binding
  the same API in C++.
- Numpy records as the base with `to_awkward()` on request, or awkward records from the
  start for the small tables? Decide after using both in one analysis.
- Moving `hadronize.py` and the production writers to `HadronArray`: V4 shows the `Hadrons`
  path at +79 B per hadron; not in scope here.
