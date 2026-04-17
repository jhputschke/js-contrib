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
| `src/bind_music.cc` | Bindings for the MUSIC module |
| `src/bind_signal_manager.cc` | Bindings for `JetScapeSignalManager` |
| `python/jetscape/__init__.py` | Package entry point; re-exports key symbols from `pyjetscape_core` |
| `python/jetscape/fno_hydro.py` | `PyFNOHydro` — Python FluidDynamics backed by a PyTorch FNO model |
| `python/jetscape/utils.py` | NumPy/PyTorch ↔ JETSCAPE bulk-info conversion helpers |
| `python/jetscape/run_jetscape.py` | High-level simulation drivers: `run_automatic()` (Mode A) and `run_manual()` (Mode B) |
| `python/jetscape/bulk_root_writer.py` | Python ROOT bulk-evolution writer via uproot |
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
| uproot | ≥ 5 | Only needed for `bulk_root_writer.py` |

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

**Step 4**: Add the package to `PYTHONPATH`:

```bash
# Add to your shell configuration (~/.zshrc, ~/.bashrc):
export PYTHONPATH="/path/to/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape/python:$PYTHONPATH"
```

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

Then set `PYTHONPATH` as shown in [Path A Step 4](#path-a---via-x-scape-cmake).

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

## Pipeline Mode Comparison

| Feature | Mode A (XML-driven) | Mode B (manual) |
|---------|--------------------|--------------------|
| `enableAutomaticTaskListDetermination` | `true` | `false` |
| Module instantiation | C++ factory from XML | Python list |
| Python trampoline modules (e.g. `PyFNOHydro`) | Not supported | **Supported** |
| Parameter changes without recompile | XML edit | Python dict / XML parse |
| Typical use | Standard JETSCAPE workflow | Research, ML integration |

---

## Example Script

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
