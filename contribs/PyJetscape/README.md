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
| `example/python_fast_bulk_root_writer.py` | Runs the C++ `FastRootBulkWriter` from Python, reads the file back |
| `conda_install/` | Conda environment installation scripts for the `js_fno` environment |
| `pyproject.toml` | Source-only Python package metadata (`name = "pyjetscape"`) |

---

## Prerequisites

| Dependency | Version | Notes |
|------------|---------|-------|
| X-SCAPE or JETSCAPE | ≥ 4.0 | Built and available; see [Path A](#path-a-via-x-scape-cmake) / [Path B](#path-b-standalone-build) |
| CMake | ≥ 3.18 | FetchContent support needed for pybind11 auto-download |
| Python | ≥ 3.8 | 3.11 used in the `js_fno` conda environment |
| PyTorch | ≥ 2.0 | Required for `PyFNOHydro`; `pyjetscape_core` itself is pure C++ |
| pybind11 | ≥ 2.11 | Fetched automatically by CMake if not found on system |
| numpy | ≥ 1.21 | |
| uproot | ≥ 5 | Only needed for `bulk_root_writer.py` and `fast_root_bulk.py` |
| h5py | ≥ 3 | Only needed for `fast_h5_bulk.py` (HDF5 bulk writer); sets `jetscape.HAS_H5PY` |
| — | — | `jetscape.HAS_CORE` reports whether the compiled extension is importable. The HDF5 tooling (`FnoH5Writer`, `grid_attrs`, `repad_to`, `read_fast_h5_bulk`) needs only h5py+numpy and stays usable without an X-SCAPE build; `H5BulkWriter` is a framework module and raises a clear error without one. |
| scipy | ≥ 1.9 | Only needed for `fast_h5_bulk.py` in `grid` / `framework` source mode |

> **Important — import order:** `torch` must be imported **before**
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
`pytorch`, `pybind11`, `numpy`, `uproot`, and ROOT.

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
```

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
/arr             (nevents, 4, nx, ny, neta, choose_ntau)  float32, lzf
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

`fno_h5_writer.py` and `repad_h5.py` are also self-contained — copy the pair anywhere and
run `python repad_h5.py *.h5` with no package at all.


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
