# FnoHydro — Fourier Neural Operator Hydrodynamics for JETSCAPE

FnoHydro is a js-contrib module that replaces the MUSIC hydrodynamics solver in
a JETSCAPE simulation with a pre-trained [Fourier Neural Operator
(FNO)](https://arxiv.org/abs/2010.08895) model.  After training the FNO on
MUSIC-generated QGP evolution data, the model produces hydrodynamic
profiles orders of magnitude faster than MUSIC while maintaining the physical
fidelity needed for downstream hard-probe and hadronisation studies.

The module is derived from the work documented in:

> *Fast prediction of hydrodynamical evolution in ultra-relativistic
> heavy-ion collisions using Fourier Neural Operators*,
> https://doi.org/10.48550/arXiv.2507.23598
> Phys. Rev. C 113 (2026) 1, 014904

Data (ROOT format) and several pre-trained models are available at:
https://zenodo.org/records/16647726

The original development repository is release v0.3 (since published results based on JETSCAPE 3.7.2)
[jhputschke/JETSCAPE-FNO](https://github.com/jhputschke/JETSCAPE-FNO).

---

## Contents

| Path | Description |
|------|-------------|
| `src/FnoHydro.cc/h` | Main FNO hydro module — reads the initial state from pre-equilibrium and drives the FNO prediction |
| `src/FnoRooIn.cc/h` | Variant that reads pre-generated hydro evolution from a ROOT file instead of running a live FNO |
| `src/PGunFno.cc/h` | Particle-gun hard-process helper that seeds events from a ROOT file |
| `src/FnoModuleLinkDef.h` | ROOT dictionary linkdef |
| `root_bulk/` | Library and executables for writing MUSIC bulk evolution to ROOT (`bulkRootWriter`, `bulkRootWriterFull`) |
| `example/` | Analysis and test executables (`fnoHydroTest`, `fnoRooInTest`, `BulkFnoRooIn`, `JetFnoRooIn`, `bulkAna`, `jetAna`) |
| `config/` | Ready-to-use JETSCAPE XML configuration files |
| `models/README.md` | Instructions for obtaining pre-trained `.pt` model files (not stored in git) |

---

## Prerequisites

| Dependency | Version | Notes |
|------------|---------|-------|
| X-SCAPE or JETSCAPE | ≥ 4.0 | Built and available; see [Path A](#path-a-via-x-scape-cmake) / [Path B](#path-b-standalone-build) |
| CMake | ≥ 3.15 | |
| ROOT | ≥ 6.20 | Must be the same ROOT used to build JETSCAPE |
| libtorch (PyTorch C++ runtime) | ≥ 2.0 | ~2 GB download; CPU or CUDA build depending on hardware |
| Python / PyTorch | optional | Only needed for the Python FNO interface ([PyJetscape](../PyJetscape/README.md)) |

> **Note on libtorch size:** The PyTorch C++ libraries are approximately 2 GB.
> The download URL is printed by `python -c "import torch; print(torch.utils.cmake_prefix_path)"`.
> Alternatively, download directly from https://pytorch.org/get-started/locally/ (select "LibTorch").

---

## Obtaining Pre-trained Model Files

Model files (`.pt`, JIT-traced TorchScript) are **not stored in this repository** because of
their size.  Download them from one of the following sources:

```bash
# Option 1 — Zenodo data archive (includes ROOT data + models)
#   https://zenodo.org/records/16647726

# Option 2 — Copy from a local JETSCAPE-FNO checkout
cp /path/to/JETSCAPE-FNO/fno_hydro/models/*.pt  <js-contrib-build>/models/
```

Place or symlink the `.pt` files so that the paths in your XML configuration
file resolve correctly (see [XML Configuration](#xml-configuration) below).

---

## Installation

### Path A — via X-SCAPE CMake

This is the recommended approach when you are already building X-SCAPE.

**Step 1**: Download js-contrib into X-SCAPE's `external_packages/`:

```bash
cd /path/to/X-SCAPE/external_packages
./get_js_contrib.sh        # clones https://github.com/jhputschke/js-contrib
```

**Step 2**: Obtain libtorch and note its prefix path:

```bash
# From an existing PyTorch installation:
python -c "import torch; print(torch.utils.cmake_prefix_path)"
# e.g.  /opt/conda/envs/js_fno/lib/python3.11/site-packages/torch/share/cmake
```

**Step 3**: Configure X-SCAPE with FnoHydro enabled:

```bash
cd /path/to/X-SCAPE
mkdir -p build && cd build

cmake .. \
  -DUSE_MUSIC=ON \
  -DUSE_ISS=ON \
  -DUSE_JS_CONTRIB=ON \
  -DUSE_JS_FNO_HYDRO=ON \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"

make -j$(nproc) FnoModule bulkRootWriter fnoHydroTest fnoRooInTest
```

The shared library `libFnoModule.so` and example executables are placed in the
X-SCAPE build directory.

---

### Path B — Standalone Build

Use this when you have an existing JETSCAPE/X-SCAPE installation and want to
build js-contrib independently.

**Step 1**: Build and install (or locate the build directory of) X-SCAPE or
JETSCAPE so that `JetScapeTargets.cmake` and `JetScapeConfig.cmake` exist
inside the build tree.  If you built X-SCAPE with the js-contrib export patch,
these files are in `<xscape-build>/`.

**Step 2**: Clone js-contrib:

```bash
git clone https://github.com/jhputschke/js-contrib.git
cd js-contrib
mkdir build && cd build
```

**Step 3**: Configure and build:

```bash
cmake .. \
  -DBUILD_FNO_HYDRO=ON \
  -DJETSCAPE_DIR=/path/to/xscape-build \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)');/path/to/root-install"

make -j$(nproc)
```

`JETSCAPE_DIR` can also be set as an environment variable:

```bash
export JETSCAPE_DIR=/path/to/xscape-build
cmake .. -DBUILD_FNO_HYDRO=ON \
         -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
```

---

## XML Configuration

FnoHydro modules are selected in the `<Hydro>` block of the JETSCAPE user XML.
Example configuration files are in `config/`.

### FnoHydro — live FNO prediction

Replace `<MUSIC>` with `<FnoHydro>`:

```xml
<Hydro>
  <FnoHydro>
    <model_file>/path/to/models/traced_JS3.7_10k_3feat_fno_model_cpu_0_10_59bins.pt</model_file>
    <n_features>3</n_features>
    <tau0>0.5</tau0>
    <freezeout_temperature>0.136</freezeout_temperature>
    <nx>60</nx>
    <ny>60</ny>
    <neta>1</neta>
    <ntau>59</ntau>
    <x_min>-15.0</x_min>
    <y_min>-15.0</y_min>
    <dtau>0.1</dtau>
    <EOS_id_MUSIC>91</EOS_id_MUSIC>
  </FnoHydro>
</Hydro>
```

### FnoRooIn — replay hydro evolution from a ROOT file

```xml
<Hydro>
  <FNOROOIN>
    <model_file>/path/to/models/traced_JS3.7_10k_3feat_fno_model_cpu_40_60_59bins.pt</model_file>
    <root_file>/path/to/input_root/sample2K_40_60.root</root_file>
    <fullHydroIn>0</fullHydroIn>
    <bulkHadroFull>1</bulkHadroFull>
    <QAoutput>0</QAoutput>
    <n_features>3</n_features>
    <tau0>0.5</tau0>
    <freezeout_temperature>0.136</freezeout_temperature>
    <nx>60</nx>
    <ny>60</ny>
    <ntau>59</ntau>
    <dtau>0.1</dtau>
    <EOS_id_MUSIC>91</EOS_id_MUSIC>
  </FNOROOIN>
</Hydro>
```

Key parameters:

| Parameter | Description |
|-----------|-------------|
| `model_file` | Path to the JIT-traced TorchScript `.pt` FNO model |
| `root_file` | (FnoRooIn only) ROOT file containing pre-generated MUSIC evolution data |
| `n_features` | Number of hydro features the model was trained on (3: energy density + vx + vy) |
| `nx`, `ny` | Transverse grid size (must match training grid, typically 60×60) |
| `ntau` | Number of time steps (typically 59) |
| `dtau` | Time step size in fm/c (typically 0.1) |
| `tau0` | Initial proper time in fm/c |
| `freezeout_temperature` | Cooper–Frye freeze-out temperature in GeV |
| `EOS_id_MUSIC` | Equation-of-state ID (91 = HotQCD) |
| `fullHydroIn` | (FnoRooIn) `1` = full hydro input from ROOT, `0` = initial state only |
| `bulkHadroFull` | `1` = pass full evolution to iSS, `0` = freeze-out surface only |

---

## Module Classes

### `FnoHydro`

The primary hydro replacement module.  Reads the initial state energy-density
profile from the pre-equilibrium module (e.g. `FreestreamMilne`), regrids it
to the FNO transverse grid, loads the TorchScript model, and runs the FNO
inference to produce the full 4D spacetime evolution (τ, x, y, η).  The
resulting evolution is made available to `iSS` for Cooper–Frye hadronisation.

Registered as `"FnoHydro"` in the JETSCAPE module factory.

### `FnoRooIn`

Reads pre-computed MUSIC evolution from a ROOT file (`TTree` format) instead
of running live FNO inference.  Used for fast replay studies and to validate
the FNO predictions event-by-event against MUSIC truth data.

Registered as `"FNOROOIN"` in the JETSCAPE module factory.

### `PGunFno`

A particle-gun hard-process module whose seeds are drawn from the same ROOT
file as `FnoRooIn`, ensuring event-level consistency between the hard and
soft sectors during replay studies.

---

## Bulk ROOT I/O Library (`root_bulk/`)

The `root_bulk/` subdirectory provides a library and executables for writing
and reading the bulk hydro evolution in ROOT format.  The data layout is:

```
TTree entry per event:
  vector<vector<vector<vector<float>>>>   data[x][y][parameter][tau]
  parameters: { 0: energy density,  1: v_x,  2: v_y }
```

These ROOT files are the input for training FNO models (see the `train_model/`
notebooks in JETSCAPE-FNO) and can be used as input to `FnoRooIn`.

### Executables

| Executable | Description |
|------------|-------------|
| `bulkRootWriter` | Writes minimal 3-feature (ε, vx, vy) evolution from MUSIC |
| `bulkRootWriterFull` | Writes full hydro evolution (more features/grid points) |
| `bulkTest` | Quick read-back validation of a bulk ROOT file |

---

## Example Executables (`example/`)

All examples are built as standalone executables that link against
`libFnoModule.so`.

| Executable | XML to use | Description |
|------------|-----------|-------------|
| `fnoHydroTest` | `config/fno_jet_test_hydro.xml` | Run FnoHydro with a particle-gun hard process |
| `fnoRooInTest` | `config/fno_bulk_test_hydro.xml` | Run FnoRooIn bulk replay (no jets) |
| `BulkFnoRooIn` | `config/fno_bulk_test.xml` | Bulk + FnoRooIn combined run |
| `JetFnoRooIn` | `config/fno_jet_test.xml` | Jet + FnoRooIn combined run |
| `bulkAna` | — | Analyse bulk ROOT output histograms |
| `jetAna` | — | Analyse jet observable ROOT output |
| `simpleReaderAna` | — | Read and inspect JETSCAPE ASCII output |

Run from the build directory, passing the path to a user XML file:

```bash
cd /path/to/build

# Bulk replay (no jets, 2000 events)
./fnoRooInTest ../contribs/FnoHydro/config/fno_bulk_test_hydro.xml

# Jet + FNO hydro (20 000 events, hydro reuse ×10)
./fnoHydroTest ../contribs/FnoHydro/config/fno_jet_test_hydro.xml
```

---

## Data Pipeline Overview

```
MUSIC runs (JETSCAPE-FNO or X-SCAPE)
        ↓
bulkRootWriter  →  ROOT file  (data[x][y][param][tau])
        ↓
train_model.ipynb  (neuraloperator library)  →  FNO model weights
        ↓
torch.jit.trace(model, example_input)  →  traced .pt file
        ↓
FnoHydro / FnoRooIn  (this contrib)  →  replaces MUSIC in production runs
```

For training and model tracing scripts, see the JETSCAPE-FNO repository:
https://github.com/jhputschke/JETSCAPE-FNO/tree/main/fno_hydro/train_model

---

## Reference

If you use FnoHydro in scientific work, please cite:

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

Please also cite [The JETSCAPE framework](https://arxiv.org/abs/1903.07706).
