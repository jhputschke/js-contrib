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
| `python/fasthydro/h5_writer.py` | `PairedH5Writer` — the FNO4d HDF5 dataset format |
| `python/fasthydro/pipeline.py` | `build_two_stage` |
| `config/` | `jetscape_user_fasthydro.xml`, `fasthydro_twostage.yaml`; `*_wake.*` for the notebook |
| `example/` | `run_two_stage.py`, `run_replay.py`, `run_hydro_only.py`, `make_wake_data.py` |
| `python/fasthydro/browse.py` | `PairBrowser` — read both legs of a pair out of one file |
| `notebooks/jet_wake.ipynb` | the wake analysis: Mach cone, damping, broadening, Mach angle |
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

## Configuration: how the XML and the YAML relate

They are **two schemas for two consumers**, not one split in half:

| | read by | holds |
|---|---|---|
| `config/jetscape_user_fasthydro.xml` | the **JETSCAPE framework** (C++) | `<IS>`, `<Preequilibrium>`, `<Hard>`, `<Eloss>`, `<Liquefier>` |
| `config/fasthydro_twostage.yaml`, top level | the **solver** (`fast_data`) | grid, EoS, transport, τ axis, deposit mode, device |
| the same YAML, `fasthydro:` section | **FastHydro's adapters** | hard-scattering vertices, `bulk_info` storage |

Neither file is generated from the other and neither is loaded by the other. The XML exists
because the framework insists on it; the YAML exists because the framework *cannot* hold
solver settings — `JetScape::Init()` → `CompareElementsFromXML()` calls `exit(-1)` on any
user-XML tag absent from `config/jetscape_main.xml`, so a `<FastHydro>` block would mean
patching X-SCAPE core. The `fasthydro:` section is separate again because `fast_data`'s
validator rejects keys it does not know, and `python/fast_data/` is vendored and never patched.

### The three places they overlap

Overlap happens only where **two different consumers need the same number**.
`build_two_stage()` calls `check_xml_agrees_with_cfg()` and refuses to run on a mismatch, so
the duplication is enforced rather than hoped for.

| quantity | XML | YAML | rule |
|---|---|---|---|
| transverse/longitudinal grid | `<IS><grid_max_*>`, `<grid_step_*>` | `grid:` | must agree; `grid_max = n·d/2` so `GetXSize()` recovers `n` |
| hydro start time | `<Preequilibrium><taus>` | `time.tau0` | must agree |
| energy-loss start | `<Eloss><tStart>` | (none) | must not precede `time.tau0`, or Matter quenches against vacuum |

That is the whole overlap — three quantities, each because two different consumers need the
same number. The liquefier parameters used to be a fourth, mirrored into the YAML and checked;
they are now XML-only (see below).

### The liquefier parameters live in the XML only

They are **not** duplicated. `<Liquefier><CausalLiquefier>` is the single place the five
deposit parameters are set; `build_two_stage()` reads them off the live C++ object and writes
them into `cfg["source"]["params"]`, so the resolved config — and therefore the `config_json`
recorded in the output file — says what actually ran. There is nothing to keep in step, so the
consistency check does not mention them.

Within that block, only some entries reach FastHydro at all:

| | effect |
|---|---|
| `tau_delay`, `time_relax`, `d_diff`, `width_delta` | shape the Python deposit |
| `dtau` | provenance only; its `1/dtau` cancels against the hydro `dtau`, so the Python deposit contains no `dtau` |
| `dx`, `dy`, `deta` | **inert here.** They size the C++ `smearing_kernel`, which FastHydro never calls — the C++ liquefier is used only as a droplet container |

`<Liquefier><threshold_energy_switch>` and `<e_threshold>` are also XML-only and *do* matter:
`filter_partons` reads them live to decide which partons become droplets at all.

### Everything else is disjoint

Deposit numerics (`source.mode`, `renorm`, `n_sub`, `min_in_grid`, …), the EoS, transport,
device and dtype exist only in the YAML. The task list, the hard process, the energy-loss
modules and the liquefier thresholds exist only in the XML. `--set` routes by prefix:
`--set time.choose_ntau=41` goes to `fast_data`, `--set fasthydro.hard_vertex.mode=centre` to
the adapters, and each is validated against its own schema.

### `source.enabled: false` does not mean "no jet"

The `source:` block is in two halves, and the key names do not say which is which:

| keys | read by |
|---|---|
| `mode`, `renorm`, `tau_eval_mode`, `n_sub`, `n_sub_max`, `min_in_grid`, `on_out_of_grid` | **FastHydro** — how a droplet is deposited onto the grid |
| `enabled`, `model`, `partons`, `per_event`, `placement_weight` | **only `fast_data`'s own `generate.py`** — where droplets come from |

`enabled` is the stock generator's switch for synthesising droplets from a `partons:` spec.
FastHydro's droplets come from Matter+LBT, so it must stay `false`; the jet source is on
regardless. `build_two_stage()` refuses to run with it true, because a `partons:` spec here
would be read by nobody — and a silently ignored jet specification gives you a plausible wake
from the wrong jet, which is worse than an error.

**`source.mode: conservative` is not a detail.** Point sampling on a cell-centred grid — what
the C++ does on MUSIC's much finer grid — loses the deposit entirely for droplets at large
`|η_d|`, where the causal support can be a fraction of a cell. Fraction of the droplet
momentum that lands (τ_d = 1, deposit at τ = 3, production grid):

| `η_d` | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| point sampling (`xscape`) | 1.000 | 1.009 | **2.561** | **0.000** | **0.000** |
| `conservative` | 1.000 | 1.000 | 0.994 | 1.000 | 1.169 |

Real Matter+LBT showers populate exactly that range.

## Where hard scatterings happen

`InitialState::SampleABinaryCollisionPoint` draws vertices from a density the initial state
supplies. If none is supplied it **only warns** and returns the origin — so every shower
starts at the fireball centre, which biases any path-length-dependent observable and makes all
events look alike. `fasthydro.hard_vertex.mode` chooses the density:

| mode | what it is |
|---|---|
| `ncoll` *(default)* | binary-collision (T_A T_B) density, each collision Gaussian-smeared by `smear` (0.4 fm, the nucleon width the medium is deposited with). What a hard process actually follows |
| `ncoll_mc` | the same distribution unsmeared — a raw histogram of the MC collision points. Sparse on a coarse grid, so much of the overlap region has zero weight and cannot be drawn at all |
| `npart` | wounded-nucleon density, smeared. Scales like the soft entropy rather than like a hard process; provided for comparison |
| `centre` | set nothing, so every vertex is the origin. The behaviour before the setter existed, kept so it can be reproduced deliberately |

Measured over 6 events at b = 6 fm, the distance of the shower origin from the fireball centre:

| mode | mean \|r\| | max \|r\| |
|---|---|---|
| `ncoll` | 2.61 fm | 4.70 fm |
| `centre` | 0.11 fm | 0.29 fm |

> **`PGun` ignores the vertex.** `src/initialstate/PGun.cc:117-120` samples the point and then
> overwrites `xLoc` with zeros, so with `PGun` every shower starts at (0,0,0) no matter what
> `hard_vertex.mode` says. `PythiaGun` uses it (`PythiaGun.cc:293-297`) and is therefore the
> default here; `build_two_stage()` warns if you pair `PGun` with a non-`centre` mode. This is
> X-SCAPE core behaviour and FastHydro does not patch it.

## Output format

**The dataset format is FNO4d's HDF5 schema** — `fast_data.writer.FnoH5Writer` used
unmodified, so a file written here is interchangeable with one written by fast_data's own
`generate.py` and loads in every existing FNO4d reader with no special case. Give `--out` an
`.h5` name (or leave it to `run.out` in the YAML) and events stream in one at a time, so peak
memory is one pair regardless of how many events you ask for.

```
arr               (N, 4, nx, ny, neta, ntau) f4   the JET leg   [e, vx, vy, vz]
arr_bg            (N, 4, nx, ny, neta, ntau) f4   the background leg
source/S          (N, 4, nx, ny, neta, ntau) f4   what was injected, contravariant Milne
source/droplets   (M, 8) f8                       (tau, x, y, eta, E, px, py, pz)
source/offsets    (N+1,) i8                       per-event slices into droplets
source/P_cart     (N, ntau, 4) f8                 injected four-momentum per frame
ntau_freezeout[_bg], tau_freezeout[_bg]           per leg
diag/                                             per-event scalars, incl. n_droplets,
                                                  n_late, E_in_window, ic_sha256
```

`e` is GeV/fm³ and `vx,vy,vz` are **Cartesian lab three-velocities** (the exact convention is
recorded in the `velocity_convention` attribute). Provenance attributes record `generator`,
`pairing`, `source_mode`, `hard_vertex` and the full resolved config as JSON.

**`arr` is the jet leg, not the background.** That follows fast_data's own convention, stated
in the file's `SOURCE_CONVENTION` attribute: *`arr[...,t]` already contains `S[...,t]` — `arr`
is the single evolution with the source in it.* So `arr` + `source/S` is exactly what an FNO
trained on deposition consumes, identical in shape and meaning to fast_data's `*_jet.yaml`
datasets. `arr_bg` is the extra thing this contribution provides: the same initial condition
with no jet, for a paired difference. Readers that do not know about it ignore it. Both legs
are checked to have evolved the same IC (by sha256) before anything is written — otherwise
`arr - arr_bg` would not be the jet's effect.

**`arr` is not `arr_bg + S`.** The source is injected *into* the evolution, so the fluid keeps
responding after the last droplet has fired and the response spreads well beyond the cells the
source ever touched. In a typical run: at τ = 2.2 fm/c the source is nonzero in 68 cells while
the two legs differ in 848; by τ = 7.0 the source is identically zero everywhere, yet 7186
cells still differ. That propagating difference — the wake — is the physics, and it is why the
file carries two evolutions rather than one evolution plus a source term.

### `.npz` is the convenience format, not the dataset

Giving `--out` a `.npz` name writes one pair (`arr`, `arr_jet`, `src`, `tau`) for a quick
look; with several events it writes `_ev0`, `_ev1`, … and holds them in memory. Use it to
inspect a single event, not to build a training set. `run_replay.py` also reads and writes
`.npz` for droplet dumps, which are small and need no h5py.

### Diagnostics worth reading every run

- **Droplets outside the τ window.** A droplet fires only if `τ_d + tau_delay` falls inside
  the solver's τ range; Matter and LBT happily produce droplets that deposit after the
  fireball has been evolved. The run reports how much energy that loses — routinely **20 % or
  more** with a short τ window — and the fix is a longer `time.choose_ntau` or a smaller
  `<CausalLiquefier><tau_delay>`.
- **Out-of-grid medium queries.** Partons that leave the fireball see vacuum, which is
  correct; but if the grid is too small this silently removes quenching. `CheckInRange` never
  throws in X-SCAPE, so this counter is the only signal.

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
| `test_framework_gates.py` | `GetHydroInfo` reproduces stored nodes; **zero droplets ⇒ the jet leg equals the background bit for bit**; four-momentum conservation to 1e-10; wake linearity; drawn vertices follow the density handed over, and `centre` still pins them to the origin |
| `test_cpp_vs_python_kernel.py` | the Python port against the **real C++** kernel, not a transcription |
| `test_h5_output.py` | the written file is a valid `fast_data/hydro_evolution` dataset: grid attrs, `arr` = jet leg and `arr_bg` = background, per-event droplet slices, provenance, and FNO4d's own reader loads it |
| `test_hard_vertex.py` | each vertex mode does what it claims; participants come out wider than binary collisions; smearing fills the holes a raw histogram leaves |
| `test_replay.py` | the dump round-trips and replays deterministically |

## The wake notebook

`notebooks/jet_wake.ipynb` is adapted from FNO4d's `viscous_vs_ideal.ipynb`, keeping the
wake-relevant analysis: the Mach cone in the $\eta = 0$ plane, wake amplitude / total
disturbance / front width against $\tau$, the Mach-angle check, and freeze-out.

**Generate its inputs with one command**, from the X-SCAPE build tree:

```bash
python ../external_packages/js-contrib/contribs/FastHydro/example/make_wake_data.py
```

About a minute per leg on MPS. `--device cpu` is slower and bitwise reproducible; `--dry-run`
prints what it would do; `--legs ideal` does one. It preflights the build tree and the config
pair, then re-runs the notebook's own controls at the end.

**The EoS table it needs is resolved for you.** It looks for a copy the machine already has —
where a previous run left one, then X-SCAPE's own `EOS/hotQCD`, then anything you point
`--eos-dir` at — and otherwise downloads MUSIC's hotQCD/SMASH table (3.2 MB) with
`fast_data`'s own fetcher: plain urllib, written to a `.part` file and renamed only once the
size validates as a whole number of 32-byte records, so an interrupted fetch cannot leave half
a table that silently loads. A truncated table already in place is detected and refetched.
`--no-download` refuses the network and prints where it looked.

The medium is central Au+Au 200 GeV on a 65×65×33 grid at 0.3125 fm with the hotQCD/SMASH
lattice EoS, $\tau = 0.58 \ldots 11.0$ fm/c — the settings from FNO4d's
`config_AuAu200_central_jet.yaml`, so this is the same medium its Mach-cone study was measured
on. `config/fasthydro_wake.yaml` and `config/jetscape_user_fasthydro_wake.xml` are that pair.

### Two files, where FNO4d needs four

Each file already carries its own jet/no-jet pair, so `PairBrowser` reads both legs out of one:

```python
from fasthydro.browse import PairBrowser
p = PairBrowser("out_wake/wake_ideal.h5")
p.diff(0, k)          # e(jet) - e(background)
p.source_at(0, tau)   # where the jet was, from the droplet table
p.summary()           # controls: IC identical, first differing frame, deposit, freeze-out
```

`PairBrowser` *is* `DiffBrowser`, handed two views of the same file, so `diff`, `source_track`,
`mach_angle` and `blob_radius` behave exactly as they do on an FNO4d pair.

### The control that needs care

**The shower responds to the medium it traverses.** Run Matter+LBT against an ideal background
and against a viscous one and you get *two different droplet sets* — 26 droplets carrying
31.2 GeV against 23 carrying 36.6 GeV, on the shipped configuration. That is real physics, not
a defect, but it means a live `visc − ideal` mixes the hydrodynamic response to the wake with a
different jet having been produced.

So `make_wake_data.py` runs the **first** leg live and **replays its droplets** through the
other solver, which is what the replay path exists for. `visc − ideal` is then the viscosity
alone, and the notebook's §2 checks the two droplet tables really are identical. `--live` gives
the uncontrolled version: the more complete physical statement, the less interpretable
comparison.

### What it shows

On the shipped configuration, with both expectations stated before the numbers are read:

| | |
|---|---|
| wake amplitude, viscous/ideal | **0.647** — damping |
| front width, viscous − ideal | **+4.65 fm** — broadening |
| static-medium Mach half-angle | **≈ 23°** ($c_s \approx 0.39$, lattice EoS) |
| jet extends the medium's life by | **+0.20 fm/c** (ideal), **+0.30** (viscous) |

The Mach angle is 23° rather than the 35° a conformal EoS gives, because the lattice equation
of state is softer near the transition. Anything measured off the maps should exceed it: the
fireball is expanding, and transverse flow at the front opens the cone.

The notebook ships with outputs cleared, following FNO4d's convention.

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
| `sample_binary_collision_point()` | draws one vertex exactly as the hard process does, so the density handed over can be checked directly rather than inferred from droplet positions |
| `load_xml(..., init_random=True)` | also seeds `JetScapeTaskSupport` from `<Random><seed>`; anything drawing random numbers before `JetScape::Init()` (the vertex sampler, for one) otherwise throws |

## Provenance and licence

GPL-3.0-or-later. `python/fast_data/liquefier/` is a port of X-SCAPE's GPLv3
`src/liquefier/CausalLiquefier.cc` and `src/framework/LiquefierBase.cc`; see
[NOTICE](NOTICE), which also records an **open licensing question** — js-contrib has no
top-level LICENSE file — worth settling with the maintainer before publishing.
