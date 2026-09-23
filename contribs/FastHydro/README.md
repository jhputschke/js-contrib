# FastHydro — MC-Glauber + a fast 3+1D hydro solver, with the jet-deposition workflow

FastHydro runs X-SCAPE's real energy-loss chain — **Matter + LBT → CausalLiquefier →
droplets** — on top of a fast, pure-Python finite-volume Milne hydro solver, and gives back a
**paired background / jet evolution on an identical initial condition**. That pair is what
energy-deposition FNO studies need, and the per-event cost makes large samples practical.

> **Hadron level, one leg at a time.** With a `<SoftParticlization>` block, iSS samples the
> chosen leg's freeze-out surface, which X-SCAPE builds in C++ from the stored evolution.
> The jet-induced hadrons are then (jet run) − (background run) on the same events. This
> needs X-SCAPE branch `fasthydro_hadronization`. See
> [Soft particlization](#soft-particlization-hadron-level) and [Limitations](#limitations).

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
| `python/fasthydro/droplets_io.py` | droplet + shower npz dump/load (no framework dependency) |
| `python/fasthydro/showers.py` | the parton shower as a space-time graph: capture, layout, segments |
| `python/fasthydro/replay.py` | re-run the jet leg from a dump, without X-SCAPE |
| `python/fasthydro/h5_writer.py` | `PairedH5Writer` — the FNO4d HDF5 dataset format |
| `python/fasthydro/pipeline.py` | `build_two_stage`, `build_bg_only` |
| `python/fasthydro/particlization.py` | the checks that a leg can give a closed Cooper–Frye surface; iSS's `music_input` |
| `config/` | `jetscape_user_fasthydro.xml`, `fasthydro_twostage.yaml`; `*_wake.*` for the notebook; `*_particlize.*` for hadrons |
| `example/` | `run_two_stage.py`, `run_replay.py`, `run_hydro_only.py`, `make_wake_data.py`, `run_particlize.py`, `delta_spectra.py`, `make_hadron_wake_data.py` |
| `python/fasthydro/browse.py` | `PairBrowser` — read both legs of a pair out of one file |
| `notebooks/jet_wake.ipynb` | the wake analysis: Mach cone, damping, broadening, Mach angle |
| `notebooks/hadron_wake.ipynb` | the wake at hadron level: spectra, ⟨p_T⟩(η), azimuth; the jet-induced excess and depletion, their balance and significance |
| `python/fasthydro/hadrons.py` | hadron files → compact npz; oversample-averaged histograms with compound-Poisson errors |
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
export OMP_WAIT_POLICY=passive OMP_NUM_THREADS=8    # idle OpenMP threads otherwise spin

# 0. smoke test: hydro only, then check GetHydroInfo against the stored grid
python $C/example/run_hydro_only.py --config $C/config/fasthydro_twostage.yaml

# 1. the real thing: background leg, Matter+LBT, jet leg -> one HDF5 dataset
python $C/example/run_two_stage.py \
    --config   $C/config/fasthydro_twostage.yaml \
    --user-xml $C/config/jetscape_user_fasthydro.xml \
    --main-xml ../config/jetscape_main.xml \
    --events 1 --out out/pair.h5 --dump-droplets out/run.droplets.npz

# 2. replay the same shower on different solver settings -- no X-SCAPE, no Matter/LBT.
#    Both legs are replayed, so this writes a normal paired dataset too.
python $C/example/run_replay.py --droplets out/run.droplets.npz \
    --config $C/config/fasthydro_twostage.yaml \
    --set transport.mode=israel_stewart --out out/pair_visc.h5
```

**`--out` dispatches on the extension, and `.h5` is what you want.** It writes FNO4d's HDF5
schema — `arr` (jet leg), `arr_bg` (background), `source/S`, the droplets — streaming one event
at a time, so peak memory is one pair however many events you ask for. A `.npz` name instead
writes a single pair as plain arrays, which is only useful for eyeballing one event.

Omitting `--out` altogether uses `run.out` from the YAML, which is already an `.h5`. The
droplet dump (`--dump-droplets`) stays `.npz`: it is small, and keeping it numpy-only is what
lets the replay path run on a machine with no h5py and no X-SCAPE build.

For the wake notebook's data, one command does both legs:

```bash
python $C/example/make_wake_data.py        # -> out_wake/wake_{ideal,visc}.h5
```

## Soft particlization (hadron level)

FastHydro needs no surface code of its own. It fills `bulk_info`. When iSS finds that the hydro
handed over no surface, it asks the hydro to build one from that stored evolution. The chain is
`SoftParticlization::FindHydroHyperSurface` → `FluidDynamics::FindSurfaceFromEvolution` →
`SurfaceFinder` (Cornelius), all C++ in X-SCAPE core. Any hydro that fills `bulk_info` gets
the same path.

```bash
cd $XSCAPE_BUILD
C=../external_packages/js-contrib/contribs/FastHydro
# jet leg: IC -> Matter+LBT -> liquefier -> FastHydro_jet -> iSS
python $C/example/run_particlize.py --leg jet --events 10 --out-dir out_particlize
# background leg: the SAME events (same seed), no jets; oversample it more
python $C/example/run_particlize.py --leg bg  --events 10 --out-dir out_particlize --oversample 200
python $C/example/delta_spectra.py --out-dir out_particlize --npz out_particlize/delta.npz
```

**One run particlizes one leg.** `<SoftParticlization><hydro_id>` picks `FastHydro_jet` or
`FastHydro_bg`. `run_particlize.py` writes a copy of the XML with the leg set. Only the
soft-particlization signals follow `hydro_id`; Matter, LBT and the liquefier keep querying
the background leg. An event's IC depends only on `(run.seed, event index)`, so event *k*
of the background run is the background of event *k* of the jet run. Both runs record the
IC's sha256, and `delta_spectra.py` refuses to pair events whose hashes differ.

**What makes the surface meaningful.** `check_xml_agrees_with_cfg` refuses a run that
cannot give one:

- `output.zero_after_freezeout` / `stop_at_freezeout` must be off, or the zeroed frames fake a
  surface.
- The EoS must be `hotqcd_smash`, the hadron gas iSS samples (`EOS_to_use 91`).

`EvolveHydro` warns if the fireball is still above T_sw at the last frame or on a transverse
boundary. The η edges are open by construction, so analyse at mid-rapidity. The shipped
`fasthydro_particlize.yaml` is the wake medium: ±10 fm, |η| ≤ 5, τ to 11 fm/c.

**The noise is the background's.** Tens of GeV of wake hadrons sit on top of ~10⁴ GeV of
bulk. iSS's Poisson fluctuations on the (jet − bg) difference fall only as 1/√(oversamples).
`delta_spectra.py` prints each difference with its compound-Poisson error, so you can see
how many oversamples a signal needs.

**Checked.** On the shipped config, one central event:

- The sampled hadrons carry 87% of the hydro's energy at τ₀. The rest leaves through the open
  η edges (T there is 0.159 GeV > T_sw) or falls outside iSS's rapidity window.
- The jet run deposited E = 27.0 GeV and p_T = 7.56 GeV in 23 droplets. The two runs' IC
  hashes agree.

With 200 oversamples per leg, `delta_spectra.py` gives:

| (jet − bg), per oversample | value | significance |
|---|---|---|
| p_T along the deposit, all hadrons | 8.8 ± 2.3 GeV | 3.9σ; deposited: 7.56 GeV |
| p_T along the deposit, \|y\| < 1 | 6.7 ± 1.4 GeV | 4.9σ |
| E, \|y\| < 1 | 17.6 ± 3.2 GeV | 5.5σ |
| N_ch, \|y\| < 1 | 5.8 ± 1.6 | 3.7σ |
| E, all hadrons | 29 ± 56 GeV | not resolved: the bulk's total-energy noise |

- The momentum balance closes within its error. That checks the surface's flow
  normalization as well as the pairing.
- In |y| < 1, the charged p_T excess sits within ~90° of the deposit direction.
- A background event with 200 oversamples takes 15 s wall on 8 threads, against 55 s
  single-threaded. The surface and the hadrons are byte-identical either way. Oversamples
  cost almost nothing; the surface finder dominates, and it scales with the thread count
  (`OMP_NUM_THREADS`).

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

### What the `source:` block is for

It configures **how** a droplet is deposited onto the grid — nothing else. All seven keys in it
are read on every jet run, live and replay alike:

```yaml
source:
  mode: conservative       # the deposit scheme
  renorm: grid
  tau_eval_mode: dep
  n_sub: auto
  n_sub_max: 16
  min_in_grid: 0.99
  on_out_of_grid: warn
```

**Where** droplets come from is not configured here at all: they come from Matter+LBT.
`fast_data`'s schema also carries `enabled`, `model`, `partons`, `per_event` and
`placement_weight` for its own `generate.py`, which synthesises droplets from a parton spec.
FastHydro reads none of them, so the shipped configs leave them at their defaults rather than
listing them as knobs that do nothing. In particular `enabled: false` would not mean "no jet
source" — the jet source is always on — and `build_two_stage()` refuses to run with it true,
because a `partons:` spec would then be silently ignored and you would get a plausible wake
from the wrong jet.

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
shower/partons    (P, 13) f8                      shower, i_src, i_tgt, pid, pstat,
                                                  px,py,pz,E, x,y,z,t
shower/vertices   (V, 6) f8                       shower, node_id, x, y, z, t
shower/initiators (K, 11) f8                      the hard parton, one per shower
shower/{parton,vertex,initiator}_offsets (N+1,) i8
ntau_freezeout[_bg], tau_freezeout[_bg]           per leg
diag/                                             per-event scalars, incl. n_droplets,
                                                  n_late, E_in_window, ic_sha256,
                                                  n_showers, n_partons
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

### The shower itself — `shower/`

`source/droplets` records what the jet **lost**. On its own a file cannot say where the jet
was, what survived it, or how it developed — so the parton shower is stored too, as a
space-time graph. It costs about **9 kB per event** against 5.5 MB for the hydro pair, so it
is written by default; set `fasthydro.store_showers: false` to turn it off.

```python
from fasthydro.browse import PairBrowser
b = PairBrowser("out/pair.h5")
par, ver, ini  = b.showers(0)          # always take the three together — see below
b.parton_fates(0)                      # {'0': 23, 'drop': 23, 'miss': 12, 'neg': 10}
start, tip, alive = b.shower_at(0, 3.0)   # the shower as it stands at t = 3 fm/c
```

`shower_at` is the animation primitive: `alive` masks the partons that exist yet, and `tip` is
the point to draw each one out to. Step `t` over the frames and the shower branches on screen.

Three properties of X-SCAPE's graph are worth knowing, because each one produces a *plausible*
picture when handled naively:

- **Vertices carry no position.** `JetEnergyLoss.cc:414-419` constructs every one of them as
  `Vertex(0,0,0,currentTime)` — only `t` is real. Measured on a live event, all 72 vertices sat
  at the origin while the partons had 18 distinct production points. The geometry is on the
  **partons**; building segments from vertex positions gives every parton zero length.
  `showers.segments` does not touch those columns.
- **Negative ("hole") partons are attached backwards.** `JetEnergyLoss.cc:413-416` runs the
  edge `new_vertex -> vStart`, so a hole's target is its own parent's source vertex and a
  naive child lookup walks it back up the parent's track.
- **A final-state parton has no end time in the graph.** It is still travelling when the
  record stops, so `shower_at` carries it at `p/E`; a parton that split is interpolated along
  its own segment, whose endpoints are both stored data.

`pstat` is what ties this group to `source/droplets`: **−11 `drop`** is a parton the liquefier
absorbed into the medium — those are the ones that became droplets — alongside −17 `neg`,
−13 `miss`, 22 photon and 101 Matter's hand-off to LBT. Colouring segments by fate shows energy
leaving the jet and arriving in the fluid.

Positions here are **Cartesian lab** `(x,y,z,t)`, not the Milne `(τ,x,y,η)` the hydro frames
and droplets use. `b.shower_segments(0, milne=True)` converts; space-like points (`|z| > t`,
which the framework does produce) come back as NaN rather than as a wrong number.

The shower travels with `--dump-droplets`, so a replayed run — the same jet through a different
solver — stays animatable.

### `.npz` is the convenience format, not the dataset

Giving `--out` a `.npz` name writes one pair (`arr`, `arr_jet`, `src`, `tau`) for a quick
look; with several events it writes `_ev0`, `_ev1`, … and holds them all in memory. Use it to
inspect a single event, not to build a training set. From `run_replay.py` a `.npz` gives only
the replayed jet evolution, where an `.h5` replays both legs into a full paired dataset.

Droplet dumps (`--dump-droplets`) are the one place `.npz` is the right answer: they are small,
and staying numpy-only is what lets the replay path run on a machine with no h5py, no X-SCAPE
build and no `pyjetscape_core`.

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
pytest tests -q                                  # 289 passed, 7 skipped
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
wake-relevant analysis: the Mach cone in the $\eta = 0$ plane, the same wake in
$(\eta, \varphi)$ — both about the beam axis and re-centred on the jet — wake amplitude /
total disturbance / front width against $\tau$, the Mach-angle check, and freeze-out.

The $(\eta, \varphi)$ panels are **spatial** angles: $\eta_s$ is the cell's spacetime
rapidity and $\varphi$ its azimuth. Turning them into the momentum-space $(\eta, \varphi)$ a
detector reports would need a Cooper–Frye surface, which FastHydro does not compute. In the
jet-centred panel $\varphi$ is measured about the **jet's own position at that $\tau$**, not
about the beam: the vertex is sampled from the binary-collision density and sat 4.6 fm off
centre here, so a beam-axis $\Delta\varphi$ would be dominated by that offset rather than by
the wake. With the right origin the deposited energy shows up where it should — trailing the
jet at $\Delta\varphi \approx 180°$, with the viscous run's peak visibly damped.

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

## The hadron-wake notebook

`notebooks/hadron_wake.ipynb` asks whether the deposit survives particlization, and what it
looks like in the hadrons. It reads two particlized runs of the same events, both viscous
(Israel–Stewart, η/s = 0.08):

```bash
cd $XSCAPE_BUILD
python ../external_packages/js-contrib/contribs/FastHydro/example/make_hadron_wake_data.py
#   -> out_hadron_wake/hadrons_{jet,bg}.npz, {jet,bg}_events.json   (~4.5 min, 8 threads)
HADRON_WAKE_OUT=$PWD/out_hadron_wake jupyter lab \
    ../external_packages/js-contrib/contribs/FastHydro/notebooks/hadron_wake.ipynb
```

The default is 4 events × 1000 oversamples with **PGun**: one 60 GeV parton from the fireball
centre along +x, so every wake sits in the same place and the events stack. `--hard
PythiaGun` gives dijets, aligned on each event's leading initiator. `--events` and
`--oversample` buy statistics. The hadron text files (~110 MB per event at 1000
oversamples) are converted to `.npz` and removed unless you pass `--keep-ascii`.

The notebook shows:

- the bulk: identified spectra, dN/dy, dN_ch/dη, ⟨p_T⟩(η) per species, and the azimuth;
- the difference jet − bg around the jet axis: the Δφ and (Δη, Δφ) maps with significance,
  the difference in p_T slices, and the excess's spectrum;
- a characterization of the near-side excess and the away-side depletion: yield, energy,
  momentum, width, ⟨p_T⟩ and composition;
- the energy-momentum balance against the deposit, and the statistics needed for 5σ.

**What the default run shows:**

- The momentum balance closes: Δp_T along the jet is 11.74 ± 0.56 GeV per event, against
  12.13 GeV deposited.
- The near-side wake is identified at 12σ in N_ch and 23σ in p_T. It is compact (RMS ≈ 0.5 in
  both Δφ and Δη).
- It is flow-boosted: ⟨p_T⟩ is 1.0 against the bulk's 0.56 GeV, the relative excess grows with
  p_T, and K/π and p/π are enhanced.
- The away-side diffusion wake is not resolved (1.2σ). It needs ~15–20× the statistics.

The Cooper–Frye is ideal (no δf, since π^{μν} is not stored) and there is no SMASH stage;
see the notebook's §6.

## Limitations

- **Soft hadrons only, and ideal Cooper–Frye.** The jet's own surviving partons are not
  hadronized, and `JetHadronization` is still **unverified**. π^{μν} and Π are stored as
  zero, so iSS's δf switches have nothing to act on. That is exact for `transport.mode:
  ideal`, and an approximation for a viscous run.
- **No SMASH in the default build.** With `-DUSE_SMASH=OFF`, an `<Afterburner>` block is
  refused and iSS decays the resonances itself (`Perform_resonance_decays 1`).
- **The surface lattice is coarser than the hydro grid.** The shipped XML uses a Cornelius
  lattice at 2× the hydro spacing (`<surface_dtau>`, `<surface_dx>`, `<surface_deta>`). At
  the hydro's own spacing the finder does ~16× more cubes; at 0, SurfaceFinder's defaults,
  ~60× more. It is threaded only in an X-SCAPE build whose CMake found C++ OpenMP. Before
  the branch's `CMakeLists.txt` fix, macOS builds silently compiled it single-threaded.
  Check with `nm -u <build>/src/CMakeFiles/JetScape.dir/framework/SurfaceFinder.cc.o | grep
  __kmpc_fork_call`.
- **Viscous components are stored as zero.** Matter, LBT and `filter_partons` read only
  `temperature`, `entropy_density` and `vx/vy/vz`, so this does not affect the jet chain.
- **The half-cell offset on jet vertices.** `InitialState::CoordFromIdx` maps index `i` to
  `-grid_max + i*step` (MUSIC's axis), while the energy density is cell-centred at
  `-(n-1)/2*step + i*step`. Sampled hard-scattering vertices therefore sit half a cell from
  the matching energy-density node.
- **Bulk viscosity is untested** upstream, and the non-conformal viscous sector has no
  analytic benchmark. See FNO4d's `README_FastData.md`.
- **`stop_at_freezeout: false`** in the shipped YAML, so both legs span the same frames. With
  it on, the two legs can stop at different times and the pair is not comparable.

## Possible extension: the same writer for a MUSIC two-stage run

`PairedH5Writer` is not tied to the fast solver, and a MUSIC two-stage run could in
principle be written into the same paired file. Two of the three pieces already exist;
this is a sketch of what is left, not a plan of record.

**The droplet and shower capture is already hydro-agnostic.** `params_from_liquefier`,
`droplets_from_liquefier` and `showers_from_manager` read the C++ `CausalLiquefier` and
`JetEnergyLossManager`; neither knows FastHydro exists. X-SCAPE's own
`examples/custom_examples/TwoStagesHydro.cc` already has the shape — one shared
`CausalLiquefier`, `MUSIC_1` added before the energy-loss manager, `MUSIC_2` after it with
the liquefier attached — including the ordering constraint FastHydro documents above, since
only the first `FluidDynamics` is registered as the framework's hydro.

**Pulling `arr` out of MUSIC is already solved.** `jetscape.bulk_sources.event_array(hydro,
grid_mode=…)` returns `(ntau, nx, ny, neta, 4)` from *any* `FluidDynamics` through
`get_bulk_info()`, and `attrs_from_grids` emits the FNO4d scalar keys. That is how
`PyJetscape`'s `H5BulkWriter` already writes MUSIC runs into this schema.

What a MUSIC leg would have to grow is small: an object exposing `.arr`, `.src`, `.diag`,
`.g` and `.ic_sha256` (what `PairedH5Writer.append` reads), a `GridSpec` built from
`bulk_info` rather than from the YAML — `bulk_sources.Grid` already has `from_bulk_info` —
and a config-shaped dict. The writer's only hard uses of the fast_data config are
`GridSpec.from_cfg`, `cfg["eos"]`, `cfg["output"]` and `cfg["run"]`; `source.mode` and
`transport.*` are provenance strings.

Three things would actually bite, and they are the reason this is a sketch:

1. **There is no `source/S` to read out of MUSIC.** `MUSIC::add_hydro_source_terms`
   (`external_packages/MUSIC/src/music.cpp:53`) takes a `HydroSourceBase` pointer and MUSIC
   evaluates it inside its own loop, per cell per substep; nothing accumulates a source
   grid. Either write the pair without it — the schema allows `write_source=False` — or
   re-deposit the same droplets with fast_data's `CausalLiquefierSource`. That
   reconstruction would **not** reproduce what MUSIC applied: MUSIC point-samples where
   conservative mode does sub-cell quadrature, and the measured gap is total, not marginal
   (point sampling loses the whole deposit at `|eta_d| >= 3` — the table under
   [What the `source:` block is for](#what-the-source-block-is-for)). Putting an
   approximation of the source into a training file is a decision, not a detail.
2. **The two legs would not be the same length.** FastHydro evolves both on a fixed tau grid
   with zero tails, so `arr` and `arr_bg` share a shape by construction. MUSIC stops at
   freeze-out and the jet leg outlives the background — 10.78 against 10.58 fm/c on the
   wake event — so the legs would need padding to a common `choose_ntau`.
   `FnoH5Writer` leaves the tau axis extendible and `PyJetscape`'s `repad_h5.py` exists for
   exactly this, but `PairedH5Writer` currently sizes `arr_bg` from the first background
   array it sees and would need to learn about the mismatch.
3. **Memory.** Both evolutions are live at once. `H5BulkWriter`'s own notes put the O+O
   native event at 1.29 GB, so two legs plus their arrays on a native MUSIC grid is where
   this stops being free; `grid_mode="grid"` downsampling is the lever.

Everything else — the `shower/` group, the droplet dump, `PairBrowser`, the wake notebook
and `Visualization/wake_pyvista.py` — reads the file, not the solver, and would work
unchanged on such a pair.

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
