# FastHydro — MC-Glauber + a fast 3+1D hydro solver, with the jet-deposition workflow

FastHydro runs X-SCAPE's real energy-loss chain — **Matter + LBT → CausalLiquefier →
droplets** — on top of a fast, pure-Python finite-volume Milne hydro solver, and gives back a
**paired background / jet evolution on an identical initial condition**. That pair is what
energy-deposition FNO studies need, and the per-event cost makes large samples practical.

> **Parton level only.** FastHydro computes no Cooper–Frye surface, so there is no soft
> particlization: do not add a `<SoftParticlization>` block, iSS would sample an empty surface
> and produce zero soft hadrons. It produces bulk evolution and jet observables.
> See [Limitations](#limitations).

The solver and the initial state come from [FNO4d](https://github.com/JETSCAPE)'s `fast_data`,
vendored verbatim under `python/fast_data/` — see [VENDORING.md](VENDORING.md). On the same
initial condition it agrees with MUSIC to ~0.3 % relative L2 in energy density at freeze-out.

---

## Contents

| Path | Description |
|---|---|
| `python/fast_data/` | **Vendored, never patched.** MC-Glauber, the FV solver, the EoS readers, the CausalLiquefier port, the HDF5 writer |
| `python/fasthydro/grid.py` | `GridSpec` — the one place fast_data's axes and X-SCAPE's are reconciled |
| `python/fasthydro/initial_state.py` | `FastGlauberInitialState`, `FastFileInitialState` |
| `python/fasthydro/hydro.py` | `FastHydro` — the `FluidDynamics` module |
| `python/fasthydro/cells.py` | solver frame → `FluidCellInfo` features (EoS-derived `T`, `s`, `p`) |
| `python/fasthydro/liquefier_bridge.py` | `DropletBridge` — C++ droplets → the solver's source term |
| `python/fasthydro/droplets_io.py` | droplet npz dump/load (no framework dependency) |
| `python/fasthydro/replay.py` | re-run the jet leg from a dump, without X-SCAPE |
| `python/fasthydro/pipeline.py` | `build_two_stage` |
| `config/` | `jetscape_user_fasthydro.xml`, `fasthydro_twostage.yaml` |
| `example/` | `run_two_stage.py`, `run_replay.py`, `run_hydro_only.py` |
| `tests/` | the vendored `fast_data` suite plus the JETSCAPE-glue gates |

---

## Prerequisites

| Dependency | Notes |
|---|---|
| X-SCAPE with PyJetscape | `cmake -DUSE_JS_CONTRIB=ON -DUSE_JS_PYJETSCAPE=ON`, then `cmake --build $XSCAPE_BUILD --target pyjetscape_core` |
| Python ≥ 3.10 | **the interpreter `pyjetscape_core` was built against** — see the warning below |
| numpy, scipy, h5py, pyyaml | `scipy.special.ive` is the causal-diffusion kernel |
| PyTorch ≥ 2.0 | the solver only; the MC-Glauber and the readers are pure numpy |

> **Interpreter mismatch.** The shipped extension is `cpython-313`, while
> `contribs/conda_install/install_js_fno_minimal.sh` pins `PYTHON_VERSION=3.11`. Following
> that recipe verbatim gives `ImportError: No module named 'jetscape'`. Use the interpreter
> that built `pyjetscape_core`.

FastHydro is **pure Python: it builds nothing and needs no CMake changes.** The only compiled
code it relies on lives in PyJetscape (see [What FastHydro added to PyJetscape](#what-fasthydro-added-to-pyjetscape)).

## Installation

```bash
pip install -e external_packages/js-contrib/contribs/FastHydro
# or, without installing:
export PYTHONPATH=$PWD/external_packages/js-contrib/contribs/FastHydro/python:$PYTHONPATH

conda install -n js_fno -c conda-forge scipy h5py pyyaml     # if the env lacks them
```

The example scripts bootstrap their own paths, so they run from a checkout with no install.

---

## Running

Run from the **X-SCAPE build tree** — the framework resolves several paths relative to `cwd`.

```bash
cd $XSCAPE_BUILD
C=../external_packages/js-contrib/contribs/FastHydro

# 0. smoke test: hydro only, then check GetHydroInfo against the stored grid
python $C/example/run_hydro_only.py --config $C/config/fasthydro_twostage.yaml

# 1. the real thing: background leg, Matter+LBT, jet leg
python $C/example/run_two_stage.py \
    --config   $C/config/fasthydro_twostage.yaml \
    --user-xml $C/config/jetscape_user_fasthydro.xml \
    --main-xml ../config/jetscape_main.xml \
    --events 1 --out out/pair.npz --dump-droplets out/run.droplets.npz

# 2. replay the same shower on different solver settings -- no X-SCAPE, no Matter/LBT
python $C/example/run_replay.py --droplets out/run.droplets.npz \
    --config $C/config/fasthydro_twostage.yaml --check \
    --set transport.mode=israel_stewart
```

`OMP_WAIT_POLICY=passive OMP_NUM_THREADS=8` is worth setting; idle OpenMP threads otherwise
spin and inflate wall time.

## The pipeline

```
FastGlauberInitialState -> PGun -> NullPreDynamics
  -> FastHydro("bg")                                background, no source
  -> JetEnergyLossManager[ JetEnergyLoss(Matter, LBT) ] + CausalLiquefier
  -> DropletBridge                                  droplets -> the jet leg's source
  -> FastHydro("jet")                               same IC, with the source
```

It mirrors `config/jetscape_user_twostagehydro.xml` and
`examples/custom_examples/TwoStagesHydro.cc`, with FastHydro in place of MUSIC.

Four things in it are not free choices:

- **`NullPreDynamics` is mandatory.** `FluidDynamics::Init()` exempts only the module ids
  `"MUSIC"` and `"Brick"` from needing pre-equilibrium and `exit(-1)`s otherwise. (Borrowing
  the id `"MUSIC"` to dodge this would make `Init()` read `<Hydro><MUSIC>` tags and poison
  every downstream `GetId()` check, so FastHydro keeps a descriptive id and accepts the
  cosmetic `Unrecognized hydro module id` warning.)
- **Matter before LBT** — Matter sets the virtuality LBT takes over at `Q0`, and
  `<Eloss><mutex>ON</mutex>` arbitrates the handover.
- **The background leg must come first.** `JetScape::SetPointers()` registers only the *first*
  `FluidDynamics` as the framework's hydro, so Matter, LBT and the liquefier all query the
  background through `GetHydroCellSignal` and the jet leg is invisible to signals. That is
  exactly the two-stage semantics.
- **`hyd_jet.add_a_liquefier(liq)` is bookkeeping, not physics.** FastHydro does not pull
  source terms through `FluidDynamics::get_source_term` the way MUSIC does; it uses
  fast_data's own `CausalLiquefierSource` over the same droplets. Attaching the liquefier is
  what lets `FastHydro.Clear()` find it and empty the droplet list between events. **There is
  no double counting.**

## Configuration

Split deliberately, and `build_two_stage()` refuses to run if the two disagree:

| | |
|---|---|
| `config/jetscape_user_fasthydro.xml` | everything the **framework** reads: `<IS>`, `<Preequilibrium>`, `<Hard>`, `<Eloss>`, `<Liquefier>` |
| `config/fasthydro_twostage.yaml` | everything the **solver** reads: grid, EoS, transport, τ axis, deposit mode, device |

Every XML tag used already exists in `config/jetscape_main.xml`. That is a hard requirement:
`JetScape::Init()` → `CompareElementsFromXML()` → `recurseToSearch()` calls `exit(-1)` for any
user-XML tag with no counterpart there, so a `<FastHydro>` block would mean patching X-SCAPE
core. Hence the YAML.

**`source.mode: conservative` is not a detail.** Point sampling on a cell-centred grid — what
the C++ does on MUSIC's much finer grid — loses the deposit entirely for droplets at large
`|η_d|`, where the causal support can be a fraction of a cell. Fraction of the droplet
momentum that lands (τ_d = 1, deposit at τ = 3, production grid):

| `η_d` | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| point sampling (`xscape`) | 1.000 | 1.009 | **2.561** | **0.000** | **0.000** |
| `conservative` | 1.000 | 1.000 | 0.994 | 1.000 | 1.169 |

Real Matter+LBT showers populate exactly that range.

## Reading the output

`run_two_stage.py` writes `arr` (background) and `arr_jet`, both
`(4, nx, ny, neta, ntau)` float32 `= [e, vx, vy, vz]`, with `e` in GeV/fm³ and `vx,vy,vz`
**Cartesian lab three-velocities**, plus the per-window source `src` in contravariant Milne.
With `--events N > 1` it writes one file per event (`pair_ev0.npz`, `pair_ev1.npz`, …), since
the module objects only ever hold the current event. For an actual dataset use
`fast_data.writer.FnoH5Writer`, which streams one event at a time into the FNO4d HDF5 schema
instead of holding them all in memory.

Two diagnostics are printed and worth reading every run:

- **Droplets outside the τ window.** A droplet fires only if `τ_d + tau_delay` falls inside
  the solver's τ range; Matter and LBT happily produce droplets that deposit after the
  fireball has been evolved. The run reports how much energy that loses — it is routinely
  **20 % or more** with the shipped settings, and the fix is a longer `time.choose_ntau` or a
  smaller `<CausalLiquefier><tau_delay>`.
- **Out-of-grid medium queries.** Partons that leave the fireball see vacuum, which is
  correct; but if the grid is too small this silently removes quenching. `CheckInRange` never
  throws (the throws are commented out in X-SCAPE), so this counter is the only signal.

## Tests

```bash
pytest tests -q                                  # 207 passed, 6 skipped
FAST_DATA_FULL_SELFTEST=1 pytest tests/test_fast_data_selftests.py   # the solver's 10 physics gates
```

Tests needing a built `pyjetscape_core` are marked `needs_xscape` and skip cleanly without it,
so the suite runs on a machine with no X-SCAPE build.

| File | Gate |
|---|---|
| `test_vendor_intact.py` | the vendored tree is byte-identical to `UPSTREAM.json` |
| `test_fast_data_*.py` | the vendored solver still behaves (FNO4d's own suite) |
| `test_adapter_grid.py` | `SetRanges` round-trips to `n`; the axes reproduce `fv.Grid` |
| `test_channels.py` | `T`, `s`, `p` match the EoS; vacuum cells stay finite |
| `test_framework_gates.py` | `GetHydroInfo` reproduces stored nodes; **zero droplets ⇒ the jet leg equals the background bit for bit**; four-momentum conservation to 1e-10; wake linearity |
| `test_cpp_vs_python_kernel.py` | the Python port against the **real C++** kernel, not a transcription |
| `test_replay.py` | the dump round-trips and replays deterministically |

## Limitations

- **No Cooper–Frye surface, no soft particlization.** `JetHadronization` does not need the
  surface and may well work, but that is **unverified** — treat v1 as parton level.
- **Viscous components are stored as zero.** Matter, LBT and `filter_partons` read only
  `temperature`, `entropy_density` and `vx/vy/vz`, so this does not affect the jet chain; it
  would matter for Cooper–Frye, which is absent anyway.
- **The half-cell offset on jet vertices.** `InitialState::CoordFromIdx` maps index `i` to
  `-grid_max + i*step` (MUSIC's axis), while the energy density is cell-centred at
  `-(n-1)/2*step + i*step`. Sampled hard-scattering vertices therefore sit half a cell from
  the matching energy-density node.
- **Bulk viscosity is untested** upstream, and the non-conformal viscous sector has no
  analytic benchmark. See FNO4d's `README_FastData.md`.
- **`stop_at_freezeout: false`** in the shipped YAML, so both legs span the same frames. With
  it on, the two legs can stop at different times and the pair is not comparable.

## What FastHydro added to PyJetscape

All additive; **no X-SCAPE core changes**.

| Addition | Why |
|---|---|
| `PyFluidDynamics::GetHydroInfo` | `FluidDynamics::GetHydroInfo` is a **stub** that never assigns the pointer, and `filter_partons`, Matter and LBT all dereference it unchecked — a null dereference, not merely wrong data. Implemented in C++ (it runs once per parton per timestep) and it **always** allocates |
| boundary snapping in that override | `bulk_info`'s grid metadata is float32, so a `tau_min` of 0.6 comes back as 0.6000000238 — a query at exactly `tau0`, which is where `<Eloss><tStart>` puts Matter, fell *below* `tau_min` and read vacuum. **The entire first frame and every outer cell were invisible.** Queries within 1e-4 of a cell of a boundary are snapped onto it |
| `store_fluid_cells_from_numpy_3d`, `store_fluid_cells_aos_3d` | the existing writer is `(F, nx, ny, ntau)` — 2+1D, no `vz`, correct only for `neta == 1` |
| `bind_liquefier.cc` | `Droplet`, `LiquefierBase`, `CausalLiquefier`, `droplets_numpy()` / `add_droplets_numpy()`, and the kernel functions for the cross-check |
| `JetEnergyLoss` bound; `JetEnergyLossManager` given its base class | neither is in the module factory, and the manager was bound with **no base class**, so it was not a `JetScapeTask` in Python and `js.Add(mgr)` could not convert |
| `load_xml(main, user)` | `CausalLiquefier`'s 0-argument constructor reads XML *in its constructor*, before `JetScape` exists to open it |
| `get_entropy_density_numpy_3d()` | the 2D view is documented as boost-invariant only |
| `set_num_of_binary_collisions_from_numpy()` | without it `SampleABinaryCollisionPoint` only warns and puts **every shower at the fireball centre** |

## Provenance and licence

GPL-3.0-or-later. `python/fast_data/liquefier/` is a port of X-SCAPE's GPLv3
`src/liquefier/CausalLiquefier.cc` and `src/framework/LiquefierBase.cc`; see
[NOTICE](NOTICE), which also records an **open licensing question** — js-contrib has no
top-level LICENSE file — worth settling with the maintainer before publishing.
