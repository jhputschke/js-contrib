# ROOT, bulk and FNO files

The hydro evolution itself, cell by cell and frame by frame, as ROOT files or as the HDF5 schema
the FNO models train on, and the PyTorch hydro module that runs such a model inside X-SCAPE. The
guides are [ROOT and C++](../../analysis/root.md), [Live PyTorch FNO hydro](../fno.md) and
[FastHydro](../fasthydro.md).

## Examples

### Read a ROOT bulk file

Files of X-SCAPE's C++ `FastRootBulkWriter` hold one event per entry; the reader returns numpy
arrays and the grid (needs `uproot`):

```python
from jetscape.fast_root_bulk import read_fast_root_bulk

d = read_fast_root_bulk("hydro_evo_fast.root", entry_stop=2)     # the first two events
evo = d["events"][0]              # (ntau, nx, ny, neta, 4) float32
d["features"]                     # ('energy_density', 'vx', 'vy', 'vz')
d["ntau"], d["tau_freezeout"]     # frames written and the end of the hydro, per event
x = d["grid"]["x"]                # cell centres [fm]; also "y", "eta", "dx", "tau_min", "dtau"
```

### The hydro of a live event as numpy

Inside a `per_event` loop ([Run events](run.md)), after the hydro ran:

```python
from jetscape.utils import bulk_info_to_numpy

arr = bulk_info_to_numpy(driver.hydro.get_bulk_info(), n_features=4)
# (ntau, nx, ny, 4): energy density, temperature, vx, vy
```

The full 3+1D evolution, one τ frame at a time, is
`driver.hydro.get_bulk_info().frame_numpy(k)` (`(nx, ny, neta, 4)`: energy density, vx, vy, vz),
listed under `EvolutionHistory` on [C++ bindings](core.md).

## Reading and writing ROOT

::: jetscape.fast_root_bulk

::: jetscape.bulk_root_writer

## The FNO4d HDF5 schema

::: jetscape.fast_h5_bulk

::: jetscape.bulk_sources

::: jetscape.fno_h5_writer

::: jetscape.h5_compression

::: jetscape.repad_h5

## The PyTorch hydro module

::: jetscape.fno_hydro

## Conversions

::: jetscape.utils
