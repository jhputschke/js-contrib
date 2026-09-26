# PyJetscape — Python Bindings for the JETSCAPE Framework

PyJetscape provides [pybind11](https://github.com/pybind11/pybind11) Python
bindings for the JETSCAPE C++ framework, enabling:

- Full Python control of the simulation pipeline (module construction, event
  loops, parameter injection).
- A **trampoline interface** that lets Python classes inherit from
  `FluidDynamics`, `InitialState`, etc., so that a pure-Python module (e.g.
  a PyTorch FNO model) runs as a first-class JETSCAPE module alongside C++
  modules.
- `PyFNOHydro` — a ready-to-use Python hydro module backed by a PyTorch FNO
  model (see [FnoHydro](../FnoHydro/README.md) for the C++ counterpart and
  the trained model files).
- Helper utilities for converting JETSCAPE bulk-info objects to NumPy/PyTorch
  tensors and writing ROOT output from Python with
  [uproot](https://github.com/scikit-hep/uproot5).

The module is derived from the work documented in:

> *Fast prediction of hydrodynamical evolution in ultra-relativistic
> heavy-ion collisions using Fourier Neural Operators*,
> https://doi.org/10.48550/arXiv.2507.23598
> Phys. Rev. C 113 (2026) 1, 014904

Original development repository:
[jhputschke/JETSCAPE-FNO](https://github.com/jhputschke/JETSCAPE-FNO)

---

## Contents

| Path | Description |
|------|-------------|
| `src/pyjetscape_core.cc` | Top-level pybind11 module definition; imports all sub-modules |
| `src/bind_framework.cc` | Bindings for `JetScape`, `JetScapeTask` |
| `src/bind_evolution.cc` | Bindings for `JetEnergyLoss`, `JetEnergyLossManager`, `Hadronization`, `HadronizationManager` |
| `src/bind_initial_state.cc` | Bindings for `InitialState` |
| `src/bind_fluid_dynamics.cc` | `FluidDynamics` trampoline — enables Python subclasses as JETSCAPE hydro modules |
| `src/bind_music.cc` | Bindings for the MUSIC module (incl. native-store numpy export for `dump_hydro_only`) |
| `src/bind_root_bulk_writer.cc` | Binding for the C++ `FastRootBulkWriter` (ROOT builds only, see `HAS_ROOT`) |
| `src/bind_signal_manager.cc` | Bindings for `JetScapeSignalManager` |
| `python/jetscape/__init__.py` | Package entry point; re-exports key symbols from `pyjetscape_core` |
| `python/jetscape/fno_hydro.py` | `PyFNOHydro` — Python FluidDynamics backed by a PyTorch FNO model |
| `python/jetscape/utils.py` | NumPy/PyTorch ↔ JETSCAPE bulk-info conversion helpers |
| `python/jetscape/run_jetscape.py` | High-level simulation drivers: `run_automatic()` (Mode A), `run_manual()` (Mode B), `per_event_loop()` / `run_per_event()` (Mode C) |
| `python/jetscape/bulk_root_writer.py` | Python ROOT bulk-evolution writer via uproot |
| `python/jetscape/fast_root_bulk.py` | Reader for `FastRootBulkWriter` ROOT files (uproot) |
| `python/jetscape/fast_h5_bulk.py` | `H5BulkWriter`: one hydro evolution per event → FNO4d HDF5 |
| `python/jetscape/fno_h5_writer.py` | The FNO4d HDF5 writer both HDF5 writers use (h5py + numpy only), plus `repad_to` |
| `python/jetscape/pair_h5.py` | `PairH5Writer`: a two-stage MUSIC run (background + jet leg) → one pair file in FastHydro's layout |
| `python/jetscape/particlize_h5.py` | `ParticlizeH5Writer` / `ParticlizeFile`: each leg's freeze-out surface (every field iSS reads) + the final partons, to hadronize later |
| `python/jetscape/hadrons_h5.py` | `HadronH5Writer` / `Hadrons`: hadrons with every oversample kept apart, sample averages with compound-Poisson errors |
| `python/jetscape/surface_replay.py` | `SurfaceReplay`: a FluidDynamics that hands a stored surface to iSS |
| `src/bind_hadronization.cc` | iSS / jet-hadronization output as numpy, per-event seeds, `hadronize_partons` on stored partons |
| `python/jetscape/showers.py`, `liquefier_io.py` | Parton-shower graph and liquefier droplet/parameter readers (numpy only) |
| `example/prod_AuAu_0_10/`, `example/prod_AuAu_0_10_jet/` | Productions: 0–10% Au+Au hydro-only, and the same with a jet as background/jet pairs |
| `example/python_fast_bulk_root_writer.py` | Runs the C++ `FastRootBulkWriter` from Python, reads the file back |
| `conda_install/` | Conda environment installation scripts for the `js_fno` environment |
| `pyproject.toml` | Source-only Python package metadata (`name = "pyjetscape"`) |

---

## Prerequisites

| Dependency | Version | Notes |
|------------|---------|-------|
| X-SCAPE or JETSCAPE | ≥ 4.0 | Built and available; see [Path A](#path-a-via-x-scape-cmake) / [Path B](#path-b-standalone-build) |
| CMake | ≥ 3.18 | `FindPython3` with the `Development.Module` component |
| Python | ≥ 3.8 | 3.11 used in the `js_fno` conda environment |
| pybind11 | ≥ 2.11 | Build time only: `pip install pybind11` or conda; CMake stops if it is not found |
| numpy | ≥ 1.21 | |
| h5py | ≥ 3 | HDF5 writers and readers (`fno_h5_writer.py`, `fast_h5_bulk.py`, `pair_h5.py`); sets `jetscape.HAS_H5PY` |
| hdf5plugin | | Blosc filter of the default compression ([README_h5_optim.md](README_h5_optim.md)). Without it the writers fall back to lzf with a warning and Blosc files cannot be read; `import jetscape` registers the filter |
| pyyaml | ≥ 6.0 | Grid YAML of the `prod_AuAu_0_10*` scripts (`run_prod.py`, `run_prod_jet.py`) |
| matplotlib, scipy, pandas, ipywidgets, ipykernel, notebook | | The example notebooks (`check_output.ipynb`, `jet_wake.ipynb`); they need no X-SCAPE build |
| — | — | All of the above are installed by `pip install -e contribs/PyJetscape`. `jetscape.HAS_CORE` reports whether the compiled extension is importable. The HDF5 tooling (`FnoH5Writer`, `grid_attrs`, `repad_to`, `read_fast_h5_bulk`) stays usable without an X-SCAPE build; `H5BulkWriter` is a framework module and raises a clear error without one. |
| PyTorch | ≥ 2.0 | Optional (`pip install -e "contribs/PyJetscape[fno]"`): only `PyFNOHydro` needs it; `pyjetscape_core` does not link libtorch |
| uproot | ≥ 5 | Optional (`[root]`): only `bulk_root_writer.py` and `fast_root_bulk.py` |

> **Important — import order:** when PyTorch is used, `torch` must be imported **before**
> `pyjetscape_core` (i.e., before `import jetscape`).  Both ROOT (loaded by
> the C++ extension) and PyTorch ship their own `libomp`; the one initialised
> second will cause a segfault on some platforms.  All example scripts handle
> this correctly.

---

## Conda Environment Setup

The `js_fno` conda environment contains all Python and build-time dependencies.
Setup scripts are in `conda_install/`:

| Script | Description |
|--------|-------------|
| `install_js_fno_minimal.sh` | Minimal install — top-level packages only, conda resolves dependencies |
| `install_js_fno_pinned.sh` | Fully pinned versions for exact reproducibility |
| `install_js_fno_build_minimal.sh` | Adds C++ build tools (CMake, compilers, ROOT) to the minimal env |
| `install_js_fno_build_pinned.sh` | Pinned versions with build tools |
| `test_js_fno_build_env.sh` | Smoke-test that the environment is correctly configured |

**Quick start (CPU / macOS Apple Silicon):**

```bash
cd contribs/PyJetscape/conda_install
bash install_js_fno_minimal.sh none   # "none" = CPU/MPS, no CUDA
conda activate js_fno
bash test_js_fno_build_env.sh
```

**Quick start (Linux with CUDA):**

```bash
bash install_js_fno_minimal.sh        # auto-detects CUDA version
# or: bash install_js_fno_minimal.sh 12.1   # force a specific CUDA version
conda activate js_fno
bash test_js_fno_build_env.sh
```

After activation, the `js_fno` environment provides `python`, `cmake`,
`pytorch`, `pybind11`, `numpy`, `uproot`, ROOT, and the HDF5/notebook packages
(`h5py`, `hdf5plugin`, `pyyaml`, `scipy`, `matplotlib`, `pandas`, `ipywidgets`, `jupyterlab`).

---

## Installation

### Path A — via X-SCAPE CMake

This is the recommended approach when you are already building X-SCAPE.

**Step 1**: Download js-contrib into X-SCAPE's `external_packages/`:

```bash
cd /path/to/X-SCAPE/external_packages
./get_js_contrib.sh        # clones https://github.com/jhputschke/js-contrib
```

**Step 2**: Activate the conda environment (or ensure prerequisites are in `PATH`):

```bash
conda activate js_fno
```

**Step 3**: Configure X-SCAPE with PyJetscape enabled:

```bash
cd /path/to/X-SCAPE
mkdir -p build && cd build

cmake .. \
  -DUSE_MUSIC=ON \
  -DUSE_ISS=ON \
  -DUSE_JS_CONTRIB=ON \
  -DUSE_JS_PYJETSCAPE=ON \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"

make -j$(nproc) pyjetscape_core
```

The compiled extension `pyjetscape_core.so` (or `.pyd` on Windows) is written
to `contribs/PyJetscape/python/jetscape/` inside the js-contrib source tree.

**Step 4**: Make the package importable — choose one of:

**Option A — PYTHONPATH** (no install, works immediately):

```bash
# Add to your shell configuration (~/.zshrc, ~/.bashrc):
export PYTHONPATH="/path/to/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape/python:$PYTHONPATH"
```

**Option B — editable pip install** (recommended for persistent use, e.g. Jupyter):

```bash
pip install -e /path/to/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape
# add PyTorch for PyFNOHydro and/or uproot for the ROOT bulk readers:
pip install -e "/path/to/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape[fno,root]"
```

This installs the Python dependencies of the HDF5 tooling, the example production scripts
and the notebooks (see [Prerequisites](#prerequisites)); PyTorch stays optional.

Or let CMake do this automatically on every build by adding
`-DJS_PIP_INSTALL_PYJETSCAPE=ON` to the `cmake` command above.

Verify:

```bash
python -c "import jetscape; print(jetscape.__version__)"
```

---

### Path B — Standalone Build

Use this when you have an existing JETSCAPE/X-SCAPE build and want to build
js-contrib independently.

```bash
conda activate js_fno

git clone https://github.com/jhputschke/js-contrib.git
cd js-contrib
mkdir build && cd build

cmake .. \
  -DBUILD_PYJETSCAPE=ON \
  -DJETSCAPE_DIR=/path/to/xscape-build \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"

make -j$(nproc) pyjetscape_core
```

Then make the package importable via PYTHONPATH or `pip install -e` as shown
in [Path A Step 4](#path-a---via-x-scape-cmake).
To have CMake run `pip install -e` automatically, add `-DJS_PIP_INSTALL_PYJETSCAPE=ON`.

---

## Usage

### Importing the Package

```python
# torch MUST be imported before jetscape
import torch
import jetscape
```

### Mode A — XML-Driven Pipeline

All modules are instantiated from the JETSCAPE XML by the C++ factory.  No
Python module injection is needed.  `enableAutomaticTaskListDetermination`
must be `true` in the user XML.

```python
import torch
from jetscape.run_jetscape import run_automatic

js = run_automatic(
    main_xml="config/jetscape_main.xml",
    user_xml="config/jetscape_user_MUSIC.xml",
)
js.Init()
js.Exec()
js.Finish()
```

### Mode B — Python Module Injection

Modules are supplied explicitly as a Python list.  Python trampoline modules
(e.g. `PyFNOHydro`) are fully supported.  Set
`enableAutomaticTaskListDetermination` to `false` in the user XML.

```python
import torch
from jetscape import create_module
from jetscape.fno_hydro import PyFNOHydro, fno_config_from_xml
from jetscape.run_jetscape import run_manual

# --- Configure the FNO grid (must match the trained model) ---
config = fno_config_from_xml("config/jetscape_user_fno.xml")

# --- Build modules ---
ini      = create_module("TrentoInitial")
preeq    = create_module("FreestreamMilne")
fno      = PyFNOHydro("models/traced_JS3.7_10k_3feat_fno_model_cpu_0_10_59bins.pt", config)
jloss_mgr = create_module("JetEnergyLossManager")
jloss     = create_module("JetEnergyLoss")
matter    = create_module("Matter")

jloss.Add(matter)
jloss_mgr.Add(jloss)

# --- Run ---
js = run_manual(
    main_xml="config/jetscape_main.xml",
    user_xml="config/jetscape_user_fno.xml",
    modules=[ini, preeq, fno, jloss_mgr],
)
js.Init()
js.Exec()
js.Finish()
```

### Mode C — Per-Event External Loop

`JetScapePerEvent` exposes the event loop one event at a time, so the loop runs
in Python and the per-event module results can be read *after* each event
executes and *before* its memory is released.  Because the data is still intact
at that point, no `set_preserve_bulk_info()` workaround is needed.  Works with
either XML configuration (automatic or manual task list).

Generator form (`per_event_loop`):

```python
import torch
from jetscape import JetScapeSignalManager
from jetscape.run_jetscape import per_event_loop

for js in per_event_loop("config/jetscape_main.xml",
                         "config/jetscape_user_MUSIC.xml"):
    hydro = JetScapeSignalManager.Instance().GetHydroPointer()
    if hydro is not None:
        info = hydro.get_bulk_info()   # live results for this event
    # ... analyse this event ...
# Finish() is called automatically when the loop ends.
```

Callback form (`run_per_event`):

```python
from jetscape.run_jetscape import run_per_event

def analyse(js):
    hydro = JetScapeSignalManager.Instance().GetHydroPointer()
    ...

run_per_event("config/jetscape_main.xml",
              "config/jetscape_user_MUSIC.xml",
              on_event=analyse)
```

For a manual pipeline, pass `modules=[...]` (validated like `run_manual()`).
`SetStartEvent(n)` / `start_event=n` shifts the global event counter.  Driving
the object directly is also supported:

```python
from jetscape import JetScapePerEvent

js = JetScapePerEvent()
js.SetXMLMainFileName("config/jetscape_main.xml")
js.SetXMLUserFileName("config/jetscape_user_MUSIC.xml")
js.Init()
for _ in range(js.GetNumberOfEvents()):
    js.ExecPerEvent()       # run one event; data stays in memory
    # ... read module data here ...
    js.ClearPerEvent()      # release memory, advance counter
js.Finish()
```

> Note: `Exec()` is disabled on `JetScapePerEvent` (it warns and exits); use the
> `ExecInit` / `ExecPerEvent` / `ClearPerEvent` API instead.

---

## `PyFNOHydro` — Python FNO Hydro Module

`PyFNOHydro` inherits from the `FluidDynamics` trampoline, making it a
drop-in replacement for any C++ hydro module in a JETSCAPE pipeline.

Three model-loading approaches are supported:

```python
from jetscape.fno_hydro import PyFNOHydro

# Approach 1 — JIT-traced .pt (compatible with C++ FnoHydro traced models)
hydro = PyFNOHydro("models/traced.pt", config)

# Approach 2 — Python model class + checkpoint file
from my_model import FNOModel
net = FNOModel(modes=12, width=20)
hydro = PyFNOHydro((net, "checkpoints/epoch50.pt"), config)

# Approach 3 — live in-memory Python model (no serialisation)
hydro = PyFNOHydro(net, config)
```

The `config` dict must specify the FNO grid parameters:

```python
config = dict(
    nx=60, ny=60, ntau=59,
    x_min=-15.0, y_min=-15.0,
    dx=0.5, dy=0.5, dtau=0.1,
    tau0=0.5,
    n_features=3,
    freezeout_temperature=0.136,
    EOS_id_MUSIC=91,
    device="cpu",   # or "cuda", "mps"
)
```

Alternatively, parse the config directly from a JETSCAPE XML file:

```python
from jetscape.fno_hydro import fno_config_from_xml
config = fno_config_from_xml("config/jetscape_user_fno.xml", device="cpu")
```

> For pre-trained model files see the
> [FnoHydro README](../FnoHydro/README.md#obtaining-pre-trained-model-files)
> and https://zenodo.org/records/16647726.

---

## Utility Functions (`utils.py`)

```python
from jetscape.utils import bulk_info_to_numpy, bulk_info_to_tensor, rebin_preeq_to_fno_grid

# Convert JETSCAPE BulkInfo to numpy array
arr = bulk_info_to_numpy(bulk_info)   # shape: (nx, ny, n_features)

# Convert to PyTorch tensor (on the specified device)
t = bulk_info_to_tensor(bulk_info, device="cpu")

# Rebin pre-equilibrium output to FNO grid
fno_input = rebin_preeq_to_fno_grid(preeq_output, config)
```

---

## Python ROOT Bulk Writer (`bulk_root_writer.py`)

`BulkRootWriter` writes the JETSCAPE hydro bulk evolution to a ROOT `TTree`
(via [uproot](https://github.com/scikit-hep/uproot5)) in the same format used
by the C++ `bulkRootWriter` executable, making the output directly usable for
FNO model training.

```python
from jetscape.bulk_root_writer import BulkRootWriter

writer = BulkRootWriter("output_bulk.root")
writer.open()
# ... inside the event loop:
writer.write_event(bulk_info)
writer.close()
```

---

## C++ `FastRootBulkWriter` from Python

`FastRootBulkWriter` is X-SCAPE's hydro-only ROOT dump (see `README_BulkFast.md`
in X-SCAPE). It reads MUSIC's native in-memory store directly and never builds
`bulk_info.data`, so it is much faster and lighter than `PyBulkRootWriter`. It
is a C++ module configured from the XML; the bindings let Python add it to a
pipeline and read its state.

It is compiled only when X-SCAPE is built with `USE_ROOT`. In-tree builds
(Path A) pick this up automatically; `jetscape.HAS_ROOT` tells you at runtime.

The user XML needs:

```xml
<Hydro>
  <MUSIC>
    <output_evolution_to_memory>1</output_evolution_to_memory>
    <dump_hydro_only>1</dump_hydro_only>
    <skip_surface>1</skip_surface>          <!-- optional -->
  </MUSIC>
</Hydro>
<FastRootBulkWriter>
  <out_file_name>hydro_evo_fast.root</out_file_name>
  <grid_mode>native</grid_mode>             <!-- native | grid -->
  <tau_stride>1</tau_stride>
</FastRootBulkWriter>
```

**XML task list** (`enableAutomaticTaskListDetermination = true`): the writer
is created from the `<FastRootBulkWriter>` block. pybind11 returns it as a
`FastRootBulkWriter`, so you can find it in the task list:

```python
import jetscape as js

jetscape = js.JetScape()
jetscape.SetXMLMainFileName("../config/jetscape_main.xml")
jetscape.SetXMLUserFileName("../config/BulkFastTest/OO_one_event_fast.xml")
jetscape.Init()
writer = next(t for t in jetscape.GetTaskList()
              if isinstance(t, js.FastRootBulkWriter))
jetscape.Exec()
jetscape.Finish()        # writes the tree and closes the ROOT file
print(writer.get_event_layout())
```

**Manual pipeline**: add it after the hydro module:

```python
writer = js.create_module("FastRootBulkWriter")
for mod in (js.create_module("TrentoInitial"), js.create_module("NullPreDynamics"),
            js.create_module("MUSIC"), writer):
    jetscape.Add(mod)
```

The file is written and closed in `Finish()`, which `JetScape.Finish()` passes
on to active tasks. If you deactivate the writer (`SetActive(False)`), call
`writer.Finish()` yourself.

Read the output back with `fast_root_bulk.read_fast_root_bulk()`. Events have
different numbers of tau steps, so it returns a list:

```python
from jetscape.fast_root_bulk import read_fast_root_bulk

d   = read_fast_root_bulk("hydro_evo_fast.root", entry_stop=1)
evo = d["events"][0]         # (ntau, nx, ny, neta, 4): energy_density, vx, vy, vz
x   = d["grid"]["x"]         # MUSIC grid in native mode, user grid in grid mode
```

### MUSIC's native store in numpy (no ROOT needed)

With `dump_hydro_only`, `MpiMusic` can also copy its native store straight into
numpy. This path does not need ROOT. The array matches one `FastRootBulkWriter`
native-mode event exactly:

```python
for jse in per_event_loop(main_xml, user_xml):     # user XML without a writer
    hydro = JetScapeSignalManager.Instance().GetHydroPointer()
    evo   = hydro.get_native_evolution_numpy(tau_stride=1)  # (ntau, nx, ny, neta, 4)
    grid  = hydro.get_bulk_info()                         # nx, dx, tau_min, dtau, ...
```

A `FastRootBulkWriter` frees the store at the end of its `Exec()`. If both are
in the same pipeline, read the store before the writer runs (see
`--check-numpy` in the example). Other accessors: `get_dump_hydro_only()`,
`get_skip_surface()`, `get_number_of_fluid_cells()`,
`get_native_fluid_cell(idx)`, `clear_hydro_info_from_memory()`.

### Example

`example/python_fast_bulk_root_writer.py` runs the writer from Python, reads the
file back and saves a quick-look plot. With `--check-numpy` it also checks the
numpy export against the ROOT file. Run it from the X-SCAPE build directory:

```bash
cd X-SCAPE/build_gpu
python ../external_packages/js-contrib/contribs/PyJetscape/example/python_fast_bulk_root_writer.py \
  --user ../config/BulkFastTest/OO_one_event_fast.xml            # XML task list
#  --manual --user <xml with enableAutomaticTaskListDetermination=false>
#  --check-numpy                                                   # numpy == ROOT
```

---

## Python HDF5 Bulk Writer (`fast_h5_bulk.py`)

`H5BulkWriter` is a pure-Python JETSCAPE module that writes the bulk hydro evolution
straight to the FNO4d training HDF5 schema. No C++ writer, no ROOT file, no conversion
step. It covers **both** C++ bulk writer modules through three source modes:

| `grid_mode` | source | C++ equivalent | agreement |
|-------------|--------|----------------|-----------|
| `native` | `MpiMusic.get_native_evolution_numpy()` | `FastRootBulkWriter`, native | **bitwise** |
| `grid` | the same, resampled onto a user grid | `FastRootBulkWriter`, grid | ~1e-7 |
| `framework` | `EvolutionHistory.to_numpy_full()`, resampled | `RootBulkWriter` | ~1e-7 |

Measured on one O+O event: `native` is bit-for-bit equal to `FastRootBulkWriter`; `grid`
agrees with it to 2.6e-7 (e), 6.1e-7 (vx), 6.8e-7 (vy), 3.6e-7 (vz) — float32 epsilon,
since `Jetscape::real` is `float` so the C++ does its own blend in float32. On a matched
event `framework` and `grid` agree **exactly** (0.0 on all four channels), which is what
pins the `to_numpy_full` column mapping on real data.

`framework` mode goes through `get_bulk_info()`, which is bound on the `FluidDynamics`
base class, so it works with **any** hydro module — not just MUSIC.

Why it exists:

* `FastRootBulkWriter` writes one event as a single `std::vector<float>`, so events past
  2^30−2 bytes trip ROOT's `TBufferFile::WriteByteCount` 30-bit length field. The O+O
  native event is 1.29 GB and a 199-step one is 1.91 GB, close to ROOT's hard ~2 GB wall
  (`README_BulkFast.md` §9, "Status: not fixed"). HDF5 has no such limit.
* The FNO training pipeline reads HDF5 and currently gets there via
  `root2hdf5/root_to_hdf5.py`. Writing the target schema directly removes that step and
  the intermediate ROOT file.
* Peak RSS drops from ~5.2–6.5 GB to ~3.9 GB on an O+O native event, because ROOT's
  basket copy of the 1.29 GB event is gone.

```python
from jetscape.fast_h5_bulk import H5BulkWriter

writer = H5BulkWriter(out_file_name="hydro_evo.h5", grid_mode="native")
jetscape.Add(writer)          # must come AFTER the hydro module
jetscape.Init(); jetscape.Exec(); jetscape.Finish()
writer.Finish()               # JetScape::Finish() does not propagate to sub-tasks
```

`Finish()` is idempotent and the writer is a context manager, so `with H5BulkWriter(...)`
is the safe form.

### Output

```
/arr             (nevents, 4, nx, ny, neta, choose_ntau)  float32, Blosc-zstd (README_h5_optim.md)
/ntau_freezeout  (nevents,)  int32
/tau_freezeout   (nevents,)  float32
root attrs: nFeatures nx ny neta choose_ntau nevents            (int64)
            x_min y_min eta_min dx dy deta tau_min tau_min_MUSIC dtau  (float64)
```

FNO4d reads this directly — `read_3d_data_hdf5`, `MultiH5Array` — with no conversion.
Read it back here with `read_fast_h5_bulk(path)`, which returns the same dict shape as
`read_fast_root_bulk` (per event `(ntau, nx, ny, neta, 4)`), so the two are
interchangeable for plotting. The reader does not need the compiled extension.

### Output grid (`grid` and `framework` modes)

Both resampling modes take the **same** nine keys as the C++ `<FastRootBulkWriter>` /
`<RootBulkWriter>` XML blocks, with the same meaning:

```python
H5BulkWriter(grid_mode="framework", out_grid=dict(
    x_min=-10, dx=0.3125, y_min=-10, dy=0.3125,
    eta_min=-5, deta=0.3125, tau_min=0.524, dtau=0.1, ntau=0))
```

* A key that is missing, `None` or `0` falls back to the **source** grid's value.
* Transverse counts are derived as `nx = 2*int(|x_min|/dx) + 1`, copied from
  `FastRootBulkWriter::fill_grid` ([FastRootBulkWriter.cc:182-206](../../../../src/root/FastRootBulkWriter.cc#L182-L206))
  so this reproduces the C++ rather than inventing a second convention. That derivation
  assumes a grid symmetric about the origin, so it does **not** round-trip an even-sized
  source axis: MUSIC's `nx=100, x_min=-15, dx=0.3` comes back as 101 cells spanning
  ±15 rather than -15..+14.7.
* Because that makes a poor default, an **empty spec — or one whose values are all 0 —
  returns the source grid unchanged** instead.
* `ntau=0` means "out to the end of the evolution".

The example exposes all nine as flags (`--x-min --dx --y-min --dy --eta-min --deta
--tau-min --dtau --ntau`).

### Sharing `choose_ntau` across jobs

`arr` is rectangular, so every event in a file is padded out to one `choose_ntau`. By
default the tau axis **grows to the longest event in that file**, which means two jobs on
the same physics routinely disagree:

```
jobA.h5: choose_ntau=14   arr=(2, 4, 8, 8, 2, 14)
jobB.h5: choose_ntau=21   arr=(2, 4, 8, 8, 2, 21)
```

**Nothing pads automatically.** `MultiH5Array` checks
`('nFeatures','nx','ny','neta','choose_ntau')` at construction and refuses rather than
guessing; `read_3d_data_hdf5` does the same:

```
ValueError: Dimension mismatch for 'choose_ntau': jobA.h5 has 14, jobB.h5 has 21
```

The bad part is *when* you find out — at training time, after the jobs have run. Two fixes.

#### Fix 1 (preferred): pin it generously — overestimating is free

```python
H5BulkWriter(..., choose_ntau=200)      # same value for every job in the campaign
```

Because the tau chunk extent is 1, padding chunks are never allocated, so a tau axis far
longer than any event costs nothing on disk. Measured on a 2-event file whose longest
event is 14 frames:

| | shape | on disk |
|---|---|---|
| auto (grows to 14) | `(2, 4, 32, 32, 8, 14)` | 0.04 MB |
| pinned at 200 | `(2, 4, 32, 32, 8, 200)` | 0.04 MB |

The asymmetry is what matters: **over**-estimating is free, **under**-estimating clips real
frames (one warning per event, plus a count at `Finish()`). In auto mode `Finish()` prints
the value to pin for the next run.

#### Fix 2: re-pad existing files in place — `repad_to()`

`repad_to` is a **pure HDF5 utility**: it imports nothing from PyJetscape, needs no
X-SCAPE build, and never loads the compiled extension. `jetscape/__init__.py` degrades
gracefully when the extension is absent (`HAS_CORE` is False, and touching a core name
raises a clear `ImportError`), so this works unchanged on a training machine:

```python
from jetscape import repad_to, FnoH5Writer, read_fast_h5_bulk   # h5py + numpy only
```

`fno_h5_writer.py`, `repad_h5.py` and `h5_compression.py` are also self-contained — copy
the three anywhere and run `python repad_h5.py *.h5` with no package at all.


If the jobs have already run, nothing has to be regenerated. `arr` is created with
`maxshape=(None, 4, nx, ny, neta, None)`, so reconciling a campaign is a metadata resize:

```bash
python -m jetscape.repad_h5 run*/hydro_evo.h5                  # to the largest
python -m jetscape.repad_h5 run*/hydro_evo.h5 --choose-ntau 200
python -m jetscape.repad_h5 run*/hydro_evo.h5 --dry-run
```

```
  run00/hydro_evo.h5: choose_ntau 14 -> 21
  run01/hydro_evo.h5: already at choose_ntau=21
  run02/hydro_evo.h5: choose_ntau 18 -> 21
choose_ntau = 21; 2 of 3 file(s) updated
```

or from Python:

```python
from jetscape import repad_to
target, changed = repad_to(["jobA.h5", "jobB.h5"])     # or choose_ntau=200, dry_run=True
```

No data movement, no size change, and the per-event lifetimes survive — the new region is
unallocated chunks that read back as exactly `0.0`, which is what `live_tau_lengths`
requires:

```
jobA tau 14 -> 21, on disk 0.04 -> 0.04 MB
MultiH5Array now: 4 events, item (4, 32, 32, 8, 21)
live_tau_lengths preserved: [10, 14, 9, 21]
```

It refuses, rather than damaging anything, in three cases: the files disagree on
`(nFeatures, nx, ny, neta)` and could never be concatenated; the target is *smaller* than
some file, since shrinking an HDF5 dataset discards data permanently; or a file's tau axis
has no `maxshape` — FNO4d's own `FnoH5Writer` pre-allocates, so its files must be
rewritten rather than resized. Re-running with the same target is a no-op.

This is the h5-to-h5 counterpart of `root2hdf5/root_to_hdf5.py --global-ntau`.

### Things worth knowing

* **`choose_ntau` is a cross-file contract** — see
  [Sharing `choose_ntau` across jobs](#sharing-choose_ntau-across-jobs) below. It is the one
  real cost of `arr` being rectangular, and the one thing most likely to bite a campaign.
* **The tau chunk extent is 1.** That is what makes the frame-by-frame write cover exactly
  one whole HDF5 chunk (no read-modify-write) and what makes the zero padding of short
  events cost nothing on disk — unwritten chunks are never allocated and read back as the
  fill value, which is exactly `0.0` as `live_tau_lengths` requires.
* **This fixes a bug in the ROOT→HDF5 path for native-mode files.** In native mode
  `FastRootBulkWriter` stores the *unused user-grid* `x_min`/`dx`/… TParameters (all zero
  unless the XML sets them) and puts the real grid under the `*_MUSIC` keys
  (`FastRootBulkWriter.cc:89-92`). `root_to_hdf5.py` copies the raw keys, so it produces
  `x_min=0, dx=0`, which breaks the `cosh_etau75_3d` normalizer and the downsample
  coordinate helpers. `H5BulkWriter` writes the real grid.
* **`tau_freezeout` describes when the hydro ended, not the output grid.** It is taken
  from the source grid (`tau_min_MUSIC + ntau_source*dtau_source`), matching
  `FastRootBulkWriter.cc:133` and `:195`, so on a resampled grid it is *not*
  `tau_min + ntau_freezeout*dtau`. The last written frame is at
  `tau_min + (ntau_freezeout-1)*dtau`.
* **Pick one source for the task list.** The writer is a Python object, so it cannot be in
  the XML task list; it has to be added from Python. If you hand `per_event_loop()` or
  `run_manual()` an explicit module list *while* the user XML still has
  `enableAutomaticTaskListDetermination = true`, the pipeline is built **twice** — MUSIC
  evolves in one instance and `JetScapeSignalManager::GetHydroPointer()` hands the writer
  the other, which reports an empty store. Either let the XML build the list
  (`modules=None`, the example's default, with the loop driven by `JetScapePerEvent`) or
  set that flag to `false` and build it all in Python (`--manual`). The tell-tale in the
  log is `Initialize PreequilibriumDynamics` appearing twice.
* **Events cut short at the grid edge are flagged.** MUSIC stops an event whose freeze-out
  surface reaches the transverse grid boundary (it expects a re-run on a larger grid, which
  X-SCAPE does not do), so that evolution is truncated. With an X-SCAPE build that has
  `MpiMusic.get_hit_grid_boundary()` (X-SCAPE `ca8dd84a`), the writer stores
  `diag/hit_grid_boundary` per event and warns.
* **XML prerequisites differ by mode and are mutually exclusive.** `native`/`grid` need
  `<dump_hydro_only>1` plus `<output_evolution_to_memory>1`; `framework` needs
  `<output_evolution_to_memory>1` *without* `dump_hydro_only`, since it reads
  `bulk_info.data`.

### Examples and tests

```bash
cd X-SCAPE/build_gpu

# write HDF5 directly
python ../external_packages/js-contrib/contribs/PyJetscape/example/python_bulk_h5_writer.py \
  --user ../config/BulkFastTest/OO_one_event_fast.xml --out hydro_evo.h5

# run the C++ ROOT writer and this one on the SAME events and compare
python ../external_packages/js-contrib/contribs/PyJetscape/example/validate_h5_vs_root.py \
  --user ../config/BulkFastTest/OO_one_event_fast.xml
```

`validate_h5_vs_root.py` needs no change to any C++ code: it uses the `JetScapePerEvent`
driver, takes the C++ writer out of automatic execution with `SetActive(False)`, runs the
HDF5 writer first with `clear_after_write=False` so it does not release MUSIC's store, and
then calls the C++ writer by hand.

`tests/test_h5_bulk.py` runs without MUSIC: it checks the interpolation against a literal
transcription of the C++ `EvolutionHistory::get()` and the output against FNO4d's loaders.

```bash
pytest external_packages/js-contrib/contribs/PyJetscape/tests/test_h5_bulk.py -q
```

---

## Background/Jet Pair Writer (`pair_h5.py`)

`PairH5Writer` writes an X-SCAPE **two-stage MUSIC** run as one FNO4d-schema HDF5 file in
FastHydro's pair layout. Each event holds the medium **without** the jet (background leg) and
**with** the jet's energy deposited into it (jet leg). The two legs start from the same
initial condition, so `arr - arr_bg` is the jet's effect and nothing else.

```
IS (e.g. 3dMCGlauber) -> Hard (PythiaGun | PGun) -> NullPreDynamics
   -> MUSIC_1                         background leg                 -> arr_bg
   -> Liquefier + Eloss (Matter+LBT)  energy loss on MUSIC_1's medium -> droplets
   -> MUSIC_2                         same IC + the droplets          -> arr
```

### Requirements

* **MUSIC with a jet source slot**, so that MUSIC_2 gets both the initial-state source
  (e.g. 3D MC-Glauber strings) and the droplets:
  * music4gpu: MUSIC4GPU `XSCAPE` from `3037be7` on; X-SCAPE's `get_music4gpu.sh` pins it.
  * CPU MUSIC: MUSIC `cee9460` (X-SCAPE PR #138). It works, but evaluates every droplet at
    every step (see *Timing* below).
  * Without the slot, MUSIC_2 silently ignores the droplets, and the writer warns that the
    jet leg is identical to the background.
* **X-SCAPE with the pair support** (branch `pair_h5_music`), for the MpiMusic bindings
  `set_dump_hydro_only`, `set_skip_surface` and `get_hit_grid_boundary`, and for the
  per-step droplet pruning. The `<freeze_out_surface>` switch below needs X-SCAPE branch
  `pair_h5_music_surface_off` and MUSIC4GPU branch `XSCAPE_surface_off`.
* **A user XML with:**
  * two `<Hydro><MUSIC>` blocks named `MUSIC_1` and `MUSIC_2`, in that order.
    Every MUSIC instance reads the **first** block, so all MUSIC settings go there. The first
    block needs `<output_evolution_to_memory>1` and `<dump_hydro_only>0`: Matter and LBT read
    MUSIC_1's framework medium.
  * `<Liquefier><CausalLiquefier>` with `<dtau>` equal to MUSIC's `Delta_Tau` (`music_input`).
  * `<Eloss>` with `<AddLiquefier>true`, and `<AddLiquefier>true` in MUSIC_2's `<Hydro>` block.
  * exactly one hard process (the automatic task list runs every `<Hard>` child it finds).
  * with NullPreDynamics and strings, `<Preequilibrium><evolutionInMemory>0`.
  * optionally `<freeze_out_surface>0`, the fast setting for training data (next section).

`example/prod_AuAu_0_10_jet/AuAu_MCGlauber_MUSIC_0_10_jet.xml` is a complete example, and
`run_prod_jet.py` checks all of the above before it runs.

### Usage

The quickest way is the production driver, which mirrors `prod_AuAu_0_10`:

```bash
conda activate js_fno
cd external_packages/js-contrib/contribs/PyJetscape/example/prod_AuAu_0_10_jet
python run_prod_jet.py --events 1 --seed 1 --no-deposit   # null test: arr == arr_bg
python run_prod_jet.py --events 10 --seed 1               # PythiaGun, pTHat 50-70 GeV
python run_prod_jet.py --events 10 --seed 1 --hard pgun --pgun-pt 60
python run_prod_jet.py --events 30 --seed 1 --reuse 3     # one background per 3 jets
./run_jobs.sh -j 2 20 25 1                                # 20 jobs x 25 events
```

See that folder's `README.md` for all options. From Python, the writer is attached after
`Init()` and called between `ExecPerEvent()` and `ClearPerEvent()`, like `H5BulkWriter` in
`run_prod.py`. It is not a framework task:

```python
import jetscape as js
from jetscape.pair_h5 import PairH5Writer
from jetscape.bulk_sources import Grid

jetscape = js.JetScapePerEvent()
jetscape.SetXMLMainFileName(main_xml); jetscape.SetXMLUserFileName(user_xml)
jetscape.Init()

grid = Grid.from_bounds((-10, 10, 65), (-10, 10, 65), (-5, 5, 33), tau_min=0.5, dtau=0.1)
writer = PairH5Writer("pairs.h5", grid_mode="grid", out_grid=grid)   # or grid_mode="native"
writer.attach(jetscape)      # AFTER Init(): switches dump_hydro_only on for MUSIC_2 only

jetscape.ExecInit()
for i in range(jetscape.GetNumberOfEvents()):
    jetscape.ExecPerEvent()
    idx = writer.Exec()      # both legs are in memory here; returns the event index or None
    jetscape.ClearPerEvent()
jetscape.Finish()
writer.Finish()              # idempotent; the writer is also a context manager
```

`attach()` finds the legs by module id (`bg_id="MUSIC_1"`, `jet_id="MUSIC_2"`), the
liquefier, and the shower manager. `writer.write_diag(idx, wall_s=...)` adds per-event
scalars after `Exec()`.

### What is written

On top of the single-leg schema (`arr`, `ntau_freezeout`, `tau_freezeout`, grid attributes):

| | |
|---|---|
| `arr` | the jet leg (MUSIC_2) |
| `arr_bg`, `ntau_freezeout_bg`, `tau_freezeout_bg` | the background leg (MUSIC_1). Always the same shape as `arr`; the τ axis grows to the longer leg, and each leg is exactly 0 after its own freeze-out. |
| `source/droplets` (M, 8), `source/offsets` | the droplets MUSIC_2 was given: `tau, x, y, eta, E, px, py, pz`. Event `i` is rows `offsets[i]:offsets[i+1]`. No `source/S` (`has_source = false`): MUSIC keeps no gridded source. |
| `shower/` | partons, vertices and initiators per event (`jetscape.showers`, as in FastHydro) |
| `diag/` | `n_droplets`, `E_droplets`, `n/E_droplets_late` (deposit after the jet leg froze out), `n/E_droplets_early`, `n_showers`, `n_partons`, `tau0_music`, `ntau_jet`, `ntau_bg`, `bg_id`, `frames_identical`, `bg_hit_boundary`, `jet_hit_boundary` |
| attributes | `pairing = "bg_jet"`, `arr_is`, `arr_bg_is`, `deposition`, `source_model`, `hard_vertex`, `liquefier_*`, `freezeout_convention_id = "frames_written"` |

The ragged tables (`source/`, `shower/`) and `diag/` are written as each event arrives, so a
run killed mid-way loses nothing it has counted. `repad_to` grows `arr` and `arr_bg`
together. Read a pair with FastHydro's `PairBrowser` (`fasthydro.browse.open_pair`), which
follows the file's freeze-out convention, or with plain h5py
(`f["arr"][i] - f["arr_bg"][i]`).

### Freeze-out surface on or off, per leg

MUSIC builds its freeze-out surface on the CPU every 5th step, which cost ~5.6 s per MUSIC
run with the serial search (~0.6 s for the search itself, ~2.7 s per run including the copies
and the hand-off, with the parallel search of MUSIC4GPU `5058545`). Training data does not
need it; only a leg that is later particlized does.
`<Hydro><MUSIC><freeze_out_surface>0` (music4gpu) builds no surface. MUSIC then stops on the
equivalent test, max(e) below the freeze-out energy density in the current and the
previously checked step. That is the same stop step, and the evolution is bit-identical
(checked; see *Timing*). A freeze-out surface reaching the grid edge is still flagged.

| where | effect |
|---|---|
| first `<Hydro><MUSIC>` block | the default for every MUSIC instance (global switch) |
| an instance's own `<Hydro><MUSIC>` block, next to its `<name>` | overrides it for that instance only |
| `MpiMusic.set_freeze_out_surface(bool)` after `Init()` | overrides both for that instance |

Example for later hadronization of the jet leg only: `0` in the first block (MUSIC_1) and
`<freeze_out_surface>1</freeze_out_surface>` in MUSIC_2's block, or
`run_prod_jet.py --surface jet`. `--surface` takes `none` (default), `bg`, `jet` or `both`.
The main XML default is 1; CPU MUSIC ignores the setting and always builds the surface.

### How the two legs are read, and the built-in checks

* **Background from the framework copy** (`bulk_info`): MUSIC_1 has to fill it for Matter and
  LBT, and filling it releases MUSIC's native store. The cells are bit-identical to a
  native read.
* **Jet leg from MUSIC's native store**, via a per-instance `set_dump_hydro_only(True)`.
* **Both legs go through the same resampling** onto one output grid.
* **`diag/frames_identical`** counts the leading bit-identical frames. The writer warns
  when it is 0 (different initial conditions), and when a jet leg that received droplets
  equals the background (MUSIC ignored the liquefier).
* **`diag/{bg,jet}_hit_boundary`** flags a leg MUSIC stopped because its freeze-out surface
  reached the grid edge.
* **`diag/bg_id`** is the first event that used this background; it repeats under
  `--reuse` (`setReuseHydro`).

### Timing (measured)

GB10, `build_gpu` (music4gpu CUDA), one 0–10% Au+Au event (seed 1): 3D MC-Glauber strings
(1184), MUSIC grid 100×100×60, output grid 65×65×17, PythiaGun 50–70 GeV, 25 droplets.

**Per event**

| run | time per event |
|---|---|
| hydro-only (`prod_AuAu_0_10`, one MUSIC run) | ~23–25 s (23.3 s) |
| hydro-only, `freeze_out_surface 0` | **17.1 s**, bit-identical |
| pair, `--no-deposit` (two MUSIC runs + jet shower) | 58 s |
| pair with deposition, before per-step droplet pruning | 184 s |
| pair with deposition (X-SCAPE `896e3d1c` + MUSIC4GPU `3037be7`) | 60 s, bit-identical output, peak memory 16 GB |
| pair with deposition, surface on the jet leg only (`--surface jet`) | 55.3 s, bit-identical |
| **pair with deposition, no surface** (`--surface none`) | **49.1 s**, bit-identical |

The pruning keeps, at each MUSIC step, only the droplets that can deposit in that step: a
`CausalLiquefier` droplet deposits in exactly one step, at `tau_drop + tau_delay`. Without
it, every droplet was evaluated for every cell at every step. That cost 126 s for 25
droplets, and it grows with the number of droplets.

**Breakdown of the 60 s pair event**

| phase | time |
|---|---|
| 3D MC-Glauber strings + Pythia hard process | 0.03 s |
| MUSIC_1 setup (read strings, set up grid) | 0.11 s |
| **MUSIC_1 evolution** (background) | **19.9 s** |
| copy of MUSIC_1's evolution into X-SCAPE's medium store (for Matter/LBT) | 6.2 s |
| Matter + LBT + liquefier | ~0.03 s |
| MUSIC_2 setup | 0.10 s |
| **MUSIC_2 evolution** (with deposition) | **21.5 s** |
| `PairH5Writer`: read both legs, resample onto the output grid, write | 12.2 s |

**Inside each MUSIC evolution** (music4gpu timers, `MUSIC_PROFILE=1`, with a temporary
timer around the CPU source pass)

| part | MUSIC_1 (479 steps) | MUSIC_2 (534 steps) |
|---|---|---|
| string deposition: CPU source pass while strings deposit (first 5 steps) | 3.9 s | 4.0 s |
| CPU source pass, later steps (per-cell calls; MUSIC_2 also the droplets) | 1.3 s | 2.2 s |
| GPU hydro update (kernels + copy back) | 6.7 s | 7.6 s |
| freeze-out surface finding (CPU, every 5th step) | 5.6 s | 5.7 s |
| GPU→host copies for the surface and stored frames | 0.7 s | 0.7 s |
| rest of the step loop | 1.5 s | 1.3 s |
| **total** | **19.8 s** | **21.5 s** |

What the numbers say:
* **Strings cost ~5 s per MUSIC run:** 3.9 s while all strings deposit against all cells
  (10 substeps of ~0.4 s), plus 1.3 s of per-step overhead afterwards. Without the strings a
  run would take ~14.5 s here.
* **The GPU hydro update is only 6.7 s.** The CPU surface finder costs almost as much
  (5.6 s). `skip_surface` does not remove it, but `freeze_out_surface 0` does (see above):
  about 6 s less per MUSIC run.
* **After pruning, the droplets cost ~0.7 s.** The jet leg's longer life (55 more steps) adds
  ~1.5 s.
* **About 18 s of the 60 s is not hydro:** the writer (12.2 s, reading and resampling two
  ~60M-cell legs) and the copy of the background into the medium store (6.2 s). These are
  the largest remaining levers.

### Future steps

**1. Start MUSIC_2 from a MUSIC_1 snapshot (expected saving ~4–5 s per event).** The two
legs are bit-identical until the first droplet deposits: measured, the stored frames match
up to τ = 1.0 fm/c, and the first deposit is at 1.04 fm/c. MUSIC_2 therefore recomputes its
first ~0.6 fm/c, including the ~4 s string deposition and ~30 steps, only to reproduce
MUSIC_1. Continuing MUSIC_2 from MUSIC_1's state just before the first deposit would save
roughly 4–5 s of MUSIC_2's 21.5 s.

* **When to snapshot.** The droplets exist only after MUSIC_1 has finished, because the
  energy loss needs its full evolution. So the snapshot cannot be timed to the actual first
  deposit; it is taken at a time known in advance. Droplet τ is never negative, so nothing
  can deposit before τ = `tau_delay` (1.0 fm/c here), and a snapshot just before it is
  always safe.
* **The bound loses almost nothing.** The earliest droplets have τ ≈ 0.04–0.05 (partons
  near the beam axis, z ≈ t), so the first deposit sits ≈ `tau_delay` + 0.04 (measured
  1.04–1.05 fm/c). Exact timing would save only ~2 more steps.
* **It works with `--reuse`:** every jet on the reused background deposits after
  `tau_delay`.
* **The stored evolution history cannot serve as the snapshot.** It keeps only T, e, s, P
  and the velocity, as float, every 5th step. MUSIC's state also holds the shear tensor
  (14 components) and the bulk pressure, which have their own equations, and the previous
  step's fields for the time derivatives, all in double precision. Restarting from the
  history would reset the viscous stresses and make the legs differ before any deposit.
* **What is needed.** A save/restore of MUSIC's full state (current and previous fields,
  about 0.3 GB on the 100×100×60 grid) at the chosen τ. On the GPU path this includes the
  device's working copy, which stays on the GPU between steps. That means a hook in
  music4gpu and plumbing in X-SCAPE's `MusicWrapper`.
* **Checks.**
  * MUSIC_2 refuses the snapshot if a droplet would deposit before it.
  * `arr` must stay bit-identical to today's full MUSIC_2 run.

**2. Other levers from the timing breakdown.**
* **`PairH5Writer`, 12.2 s:** reading both ~60M-cell legs and resampling them onto the
  output grid. Faster resampling, or resampling only the output frames needed, would help;
  `grid_mode="native"` skips it (larger files).
* **Copy of MUSIC_1 into X-SCAPE's medium store, 6.2 s:** needed by Matter/LBT.
* **Freeze-out surface on the CPU, ~5.6 s per MUSIC run:** done: `freeze_out_surface 0`,
  set per leg (see above), and where a surface is needed the search now runs in parallel
  (MUSIC4GPU `5058545`: ~0.6 s per MUSIC run for the search).
* **CPU source pass after the strings are gone, ~1.3 s per MUSIC run:** it still makes
  one call per cell per substep. Skipping the pass when no string and no droplet is
  active would remove it.
* **Upstream CPU MUSIC:** it lacks the per-step jet-source call (MUSIC4GPU `3037be7`), so
  a CPU-only build still evaluates every droplet at every step.

### Tests

`tests/test_pair_h5.py` runs the writer against stub MUSIC legs (no build needed): the pair
layout, per-leg freeze-out, ragged tables, diagnostics, repad, `bg_id`, and the boundary
flag. FastHydro's `tests/test_music_pair_browser.py` opens such a file with `PairBrowser`.

```bash
pytest external_packages/js-contrib/contribs/PyJetscape/tests/test_pair_h5.py -q
```

---

## Hadron level: surfaces, partons, hadrons (`particlize_h5.py`, `hadrons_h5.py`)

A two-stage run can store what hadronization needs instead of hadrons: each leg's MUSIC
freeze-out surface and the final partons. Hadronizing them later gives exactly what iSS and
`ColorlessHadronization` would have given inside the job (checked bit for bit). The production
driver is `example/prod_AuAu_0_10_jet` (`run_prod_jet.py --write-particlize`, `hadronize.py`,
see its README); the design is in `example/prod_AuAu_0_10_jet/PLAN_particlize_h5.md`. Needs
X-SCAPE branch `surface_to_hadrons` (seed hooks in `SoftParticlization` and
`ColorlessHadronization`, `<JetHadronization><reseed_per_event>`).

**Bindings** (`pyjetscape_core`):

| | |
|---|---|
| `FluidDynamics.surface_to_numpy()` | the surface as `(N, 32)` float32, columns `SURFACE_CELL_COLUMNS` (the `SurfaceCellInfo` fields iSS reads, in its order; lossless, the struct is float) |
| `FluidDynamics.store_surface_from_numpy(cells)` | the reverse, through `StoreSurfaceCell` |
| `JetEnergyLossManager.final_partons_numpy()` | `(N, 14)`, columns `FINAL_PARTON_COLUMNS`: exactly `GetFinalStatePartons`, which the hadronization receives |
| `Parton.color()`, `anti_color()`, `restmass()` | |
| `soft_hadrons_numpy(task)` | a SoftParticlization module's (iSS) hadrons, all oversamples, with `sample_counts` |
| `soft_set_next_random_seed(task, s)`, `soft_last_random_seed(task)` | one-shot iSS seed; the seed the last event used |
| `hadronization_hadrons_numpy(task)` | a HadronizationManager's output hadrons |
| `hadronize_partons(module, partons, seed=None)` | run a jet hadronization module on stored partons; `seed` reseeds ColorlessHadronization first |
| `jet_hadronization_last_random_seed(task)` | ColorlessHadronization's last seed (with `<reseed_per_event>1`) |

These are free functions taking a task because `create_module()` returns modules whose
concrete types are not registered with pybind11.

**Writing** (between `ExecPerEvent()` and `ClearPerEvent()`, after `PairH5Writer.Exec()`):

```python
from jetscape.pair_h5 import PairH5Writer
from jetscape.particlize_h5 import ParticlizeH5Writer

pair = PairH5Writer("pair.h5", keep_surface=("jet", "bg"))   # skip_surface off for both legs
part = ParticlizeH5Writer("pair_particlize.h5", legs=("jet", "bg"), pair_file="pair.h5")
jetscape.Init(); pair.attach(jetscape); part.attach(jetscape)   # attach reads ./music_input
jetscape.ExecInit()
for i in range(n):
    jetscape.ExecPerEvent()
    idx = pair.Exec()
    if idx is not None:
        d = pair.last_event_diag
        part.Exec(idx, bg_id=d["bg_id"], bg_key=pair.last_bg_key)   # bg surface once per bg_id
    jetscape.ClearPerEvent()
pair.Finish(); part.Finish()
```

MUSIC must build the surfaces (`<freeze_out_surface>1` for those instances).

**Replaying** a stored surface into iSS (what `hadronize.py` does; the user XML needs
`<setReuseHydro>false</setReuseHydro>`, or the main XML's reuse switches iSS off for 9 of 10
events):

```python
from jetscape import pyjetscape_core as core
from jetscape.surface_replay import SurfaceReplay
from jetscape.particlize_h5 import ParticlizeFile

replay, iss = SurfaceReplay(), core.create_module("iSS")
jetscape = core.JetScapePerEvent(); jetscape.Add(replay); jetscape.Add(iss)
jetscape.SetXMLMainFileName(main_xml); jetscape.SetXMLUserFileName(user_xml)
jetscape.Init(); jetscape.ExecInit()
with ParticlizeFile("pair_particlize.h5") as pf:
    replay.load(pf.surface("jet", 0))
    core.soft_set_next_random_seed(iss, 12345)
    jetscape.ExecPerEvent()
    hadrons = core.soft_hadrons_numpy(iss)      # pid, pstat, p [E,px,py,pz], x, mass, sample_counts
    jetscape.ClearPerEvent()
```

iSS reads `music_input` from its working directory: write `pf.music_input()` (the job's own,
stored verbatim) there first.

**Reading hadrons:** `Hadrons.from_h5(path, units=None)` gives `pid`, `p`, `pt`, `eta`, `y`,
`phi`, `charged`, the unit and sample of every hadron, and `hist()` / `total()` averaged over
the samples of the selected units with compound-Poisson errors.

**Single events:** every sample (oversample, fragmentation) is a complete event.
`Hadrons.sample_event(unit, k)` (in memory) and `HadronFile(path).sample_event(unit, k)`
(read from disk) return sample `k` of recorded unit `unit` (`units/unit`: the event, or the
background for `bulk_bg`) as a dict of `pid`, `pstat`, `p`, `x`. `JetEvents.from_stem(stem)`
opens a production file's three hadron files plus its particlize file:
`jet_event(event, k)` is bulk_jet oversample `k` plus fragmentation `k mod n_frag` (or
`frag_sample=`) with an `origin` array (0 bulk, 1 fragment), `background_event(event, k)` the
background that event used (reuse-aware through `events/bg_unit`), and
`iter_jet_events(event)` all oversamples. Tests: `tests/test_particlize_h5.py`.

---

## Pipeline Mode Comparison

| Feature | Mode A (XML-driven) | Mode B (manual) | Mode C (per-event) |
|---------|--------------------|--------------------|--------------------|
| `enableAutomaticTaskListDetermination` | `true` | `false` | either |
| Module instantiation | C++ factory from XML | Python list | XML or Python list |
| Python trampoline modules (e.g. `PyFNOHydro`) | Not supported | **Supported** | **Supported** (manual) |
| Event loop driven by | `JetScape::Exec()` | `JetScape::Exec()` | **Python loop** |
| Per-event data accessible before clear | No | No | **Yes** |
| Typical use | Standard JETSCAPE workflow | Research, ML integration | Per-event analysis / streaming |

---

## Example Script

A per-event (Mode C) example is included at `example/per_event_loop.py`. It
runs the event loop in Python and reads the live hydro module for each event.
It supports both the XML task list and a manual pipeline:

```bash
conda activate js_fno
export PYTHONPATH="/path/to/js-contrib/contribs/PyJetscape/python:$PYTHONPATH"

# XML-driven task list (user XML: enableAutomaticTaskListDetermination = true)
python example/per_event_loop.py \
  --main config/jetscape_main.xml \
  --user config/jetscape_user.xml \
  --events 5

# Manual pipeline (user XML: enableAutomaticTaskListDetermination = false)
python example/per_event_loop.py --manual \
  --user config/jetscape_user_MUSIC.xml \
  --initial-state TrentoInitial \
  --preequilibrium NullPreDynamics \
  --hydro-module MUSIC \
  --events 5
```

In `--manual` mode the example builds `[initial-state, pre-equilibrium, hydro]`
via `create_module()` and hands the list to `per_event_loop(..., modules=[...])`;
swap in a Python trampoline module (e.g. `PyFNOHydro`) for the hydro stage as
needed.

A full end-to-end example (Mode B with PyFNOHydro) is included in the
JETSCAPE-FNO repository at
`examples/python_fno_test.py`.  To run it after installing PyJetscape:

```bash
conda activate js_fno
export PYTHONPATH="/path/to/js-contrib/contribs/PyJetscape/python:$PYTHONPATH"

cd /path/to/JETSCAPE-FNO
python examples/python_fno_test.py \
  --model fno_hydro/models/traced_JS3.7_10k_3feat_fno_model_cpu_40_60_59bins.pt \
  --main  config/jetscape_main.xml \
  --user  fno_hydro/config/jetscape_user_root_bulk_test.xml \
  --events 5 \
  --device cpu
```

---

## Troubleshooting

**`ImportError: cannot import name 'pyjetscape_core'`**
: The `.so` is not in `python/jetscape/`.  Rebuild with `make pyjetscape_core`
and confirm `PYTHONPATH` includes `contribs/PyJetscape/python`.

**Segfault on `import jetscape` after `import ROOT`**
: Import `torch` before `jetscape` (and before any ROOT import) to avoid the
dual-OpenMP initialisation crash.  See note in [Prerequisites](#prerequisites).

**`pybind11 not found`**
: CMake will attempt to download pybind11 via `FetchContent`.  Ensure internet
access during the first configure, or install pybind11 manually
(`conda install pybind11` or `pip install pybind11`) and add its prefix to
`CMAKE_PREFIX_PATH`.

**`libtorch_cpu.so: cannot open shared object file`**
: Add the PyTorch library directory to `LD_LIBRARY_PATH`:
```bash
export LD_LIBRARY_PATH="$(python -c 'import torch; import os; print(os.path.dirname(torch.__file__))')/lib:$LD_LIBRARY_PATH"
```

---

## Reference

If you use PyJetscape in scientific work, please cite:

```bibtex
@article{FNOHydro2026,
  title  = {Fast prediction of hydrodynamical evolution in ultra-relativistic
            heavy-ion collisions using Fourier Neural Operators},
  journal= {Phys. Rev. C},
  volume = {113},
  number = {1},
  pages  = {014904},
  year   = {2026},
  doi    = {10.48550/arXiv.2507.23598}
}
```

Please also cite [The JETSCAPE framework](https://arxiv.org/abs/1903.07706)
and [pybind11](https://github.com/pybind11/pybind11).
