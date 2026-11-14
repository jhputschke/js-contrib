# ROOT, bulk and FNO files

The hydro evolution itself, cell by cell and frame by frame, as ROOT files or as the HDF5 schema the FNO models train on, and the PyTorch hydro module that runs such a model inside X-SCAPE. The guides are [ROOT and C++](https://jhputschke.github.io/js-contrib/analysis/root/index.md), [Live PyTorch FNO hydro](https://jhputschke.github.io/js-contrib/python/fno/index.md) and [FastHydro](https://jhputschke.github.io/js-contrib/python/fasthydro/index.md).

## Examples

### Read a ROOT bulk file

Files of X-SCAPE's C++ `FastRootBulkWriter` hold one event per entry; the reader returns numpy arrays and the grid (needs `uproot`):

```python
from jetscape.fast_root_bulk import read_fast_root_bulk

d = read_fast_root_bulk("hydro_evo_fast.root", entry_stop=2)     # the first two events
evo = d["events"][0]              # (ntau, nx, ny, neta, 4) float32
d["features"]                     # ('energy_density', 'vx', 'vy', 'vz')
d["ntau"], d["tau_freezeout"]     # frames written and the end of the hydro, per event
x = d["grid"]["x"]                # cell centres [fm]; also "y", "eta", "dx", "tau_min", "dtau"
```

### The hydro of a live event as numpy

Inside a `per_event` loop ([Run events](https://jhputschke.github.io/js-contrib/python/api/run/index.md)), after the hydro ran:

```python
from jetscape.utils import bulk_info_to_numpy

arr = bulk_info_to_numpy(driver.hydro.get_bulk_info(), n_features=4)
# (ntau, nx, ny, 4): energy density, temperature, vx, vy
```

The full 3+1D evolution, one τ frame at a time, is `driver.hydro.get_bulk_info().frame_numpy(k)` (`(nx, ny, neta, 4)`: energy density, vx, vy, vz), listed under `EvolutionHistory` on [C++ bindings](https://jhputschke.github.io/js-contrib/python/api/core/index.md).

## Reading and writing ROOT

### `jetscape.fast_root_bulk`

python/jetscape/fast_root_bulk.py

Reader for ROOT files written by the X-SCAPE C++ FastRootBulkWriter.

The writer itself is a C++ module (bound as `jetscape.FastRootBulkWriter` when PyJetscape is built against a ROOT-enabled X-SCAPE, see `HAS_ROOT`). This module only reads its output, so it needs uproot but not ROOT.

File layout written by FastRootBulkWriter

TTree `t`, one entry per event: user_res : vector, flat `[tau][x][y][eta][feature]` features = [energy_density, vx, vy, vz] ntau_freezeout : int — number of tau steps in user_res (varies per event) tau_freezeout : float — tau one step past the last stored MUSIC step File-level TParameters (from the first event): nFeatures, nx, ny, neta, tau_min, dtau, tau_stride, use_vec, x_min, dx, y_min, dy, eta_min, deta, ntau (user grid, grid mode) nX_MUSIC, X_min_MUSIC, dX_MUSIC, ... tau_min_MUSIC, dtau_MUSIC (MUSIC grid) TNamed `grid_mode`: "native" or "grid".

Usage

```text
from jetscape.fast_root_bulk import read_fast_root_bulk

d = read_fast_root_bulk("hydro_evo_fast.root", entry_stop=1)
evo = d["events"][0]              # (ntau, nx, ny, neta, 4) float32
x   = d["grid"]["x"]              # cell coordinates [fm]
```

#### `read_fast_root_bulk(path: str, entry_start: Optional[int] = None, entry_stop: Optional[int] = None) -> dict`

Load a ROOT file written by FastRootBulkWriter.

Native-mode events are large (the full MUSIC grid), so use `entry_start` / `entry_stop` to load only some events.

Parameters:

- **`path`** (`str`) – Path to the .root file.
- **`entry_start`** (`int`, default: `None` ) – Event range to load (uproot semantics; default: all events).
- **`entry_stop`** (`int`, default: `None` ) – Event range to load (uproot semantics; default: all events).

Returns:

- `dict with keys:` – "events" — list of np.ndarray, each (ntau, nx, ny, neta, 4) float32; ntau differs between events "ntau" — np.ndarray of ntau per loaded event "tau_freezeout" — np.ndarray of tau_freezeout per loaded event (None for files written before that branch existed) "grid_mode" — "native" or "grid" "features" — ("energy_density", "vx", "vy", "vz") "grid" — dict: nx, ny, neta, x_min, dx, y_min, dy, eta_min, deta, tau_min, dtau and coordinate arrays x, y, eta (MUSIC grid in native mode, user grid in grid mode; None if the file predates the \_MUSIC parameters) "params" — dict of all TParameters stored in the file
- `Requires uproot.` –

### `jetscape.bulk_root_writer`

python/jetscape/bulk_root_writer.py

PyBulkRootWriter — Python mirror of fno_hydro/root_bulk/bulkRootWriter.cc.

The class inherits from pyjetscape.JetScapeModuleBase (via the PyJetScapeModuleBase C++ trampoline) and can be added to a JETSCAPE pipeline with JetScape.Add(). Its Exec() method is called automatically once per event, exactly like the C++ version.

The bulk info is obtained through JetScapeSignalManager.Instance() .GetHydroPointer(), which mirrors the C++ BulkRootWriter::Exec() call:

```text
auto hydro = JetScapeSignalManager::Instance()->GetHydroPointer();
```

Output format

ROOT TTree written by uproot (uproot >= 4 required). The tree contains one entry per event with branches:

```text
user_res  : float32 array of shape (m_nX, m_nY, m_nT, n_features)
            Features: [energy_density [GeV/fm^3], vx, vy]
nx        : int32   — number of x grid points
ny        : int32   — number of y grid points
ntau      : int32   — number of tau grid points
nfeatures : int32   — number of feature channels
tau0      : float32 — starting proper time [fm/c]
dtau      : float32 — tau step [fm/c]
x_min     : float32 — minimum x [fm]
dx        : float32 — x (and y) step [fm]
```

If uproot is not installed the writer falls back to an .npz archive with the same data stored as individual arrays per event.

Pipeline placement

```text
jetscape.Add(trento)
jetscape.Add(null_predynamics)
jetscape.Add(hydro)          # must come BEFORE PyBulkRootWriter
jetscape.Add(bulk_writer)    # ← here
```

Usage

```text
from python.jetscape.bulk_root_writer import PyBulkRootWriter

bulk_writer = PyBulkRootWriter(
    output_name = "bulk_output.root",
    n_features  = 3,       # energy_density, vx, vy
    tau_max     = 5.0,     # [fm/c]
    d_tau       = 0.1,     # [fm/c]
    d_x         = 0.5,     # [fm]
)
jetscape.Add(bulk_writer)
jetscape.Init()
jetscape.Exec()
jetscape.Finish()      # also calls bulk_writer.Finish(): JetScape::Finish()
                       # finishes every active task (JetScapeTask::FinishTasks)
```

#### `PyBulkRootWriter(output_name: str = 'bulk_root_writer_output.root', n_features: int = 3, tau_max: float = 5.0, d_tau: float = 0.1, d_x: float = 0.5, verbose: bool = False)`

Bases: `JetScapeModuleBase`

Python equivalent of C++ BulkRootWriter.

Hooks into the JETSCAPE pipeline via JetScapeModuleBase. On each call to Exec() it reads the hydro bulk info through JetScapeSignalManager, interpolates it onto a uniform user-defined grid, and accumulates the result. On Finish() the accumulated data are written to a ROOT file (via uproot) or, if uproot is unavailable, to a .npz archive.

Parameters:

- **`output_name`** (`str`, default: `'bulk_root_writer_output.root'` ) – Output file path. Use a .root extension for ROOT output or any other extension for .npz fallback output.
- **`n_features`** (`int`, default: `3` ) – Number of feature channels to store per cell. 3 (default) → [energy_density, vx, vy] 4 → [energy_density, temperature, vx, vy]
- **`tau_max`** (`float`, default: `5.0` ) – Maximum proper time to store [fm/c]. Default: 5.0.
- **`d_tau`** (`float`, default: `0.1` ) – Proper-time step of the output grid [fm/c]. Default: 0.1.
- **`d_x`** (`float`, default: `0.5` ) – Transverse spatial step of the output grid [fm]. Default: 0.5.
- **`verbose`** (`bool`, default: `False` ) – Print per-event timing information. Default: False.

##### `Init() -> None`

Reset per-run state. Called once by JetScape::Init().

##### `Exec() -> None`

Read hydro bulk info for the current event and accumulate it.

Mirrors BulkRootWriter::Exec() in bulkRootWriter.cc:

1. Get hydro pointer via JetScapeSignalManager.
1. Determine user-resolution grid dimensions.
1. Interpolate bInfo.get(tau, x, y, 0) onto the grid.
1. Append to the event list.

##### `Clear() -> None`

Called between events — no per-event cleanup needed here.

##### `Finish() -> None`

Finalise the output file.

For ROOT output the tree has already been filled per event; this method just closes the file handle. For the .npz fallback the accumulated events are written here.

Called automatically by JetScape::Finish(), or manually by the user after jetscape.Exec() completes.

##### `read_root(path: str) -> dict`

Load a ROOT file written by PyBulkRootWriter.

Parameters:

- **`path`** (`str`) – Path to the .root file.

Returns:

- `dict with keys:` – "user_res" — np.ndarray of shape (n_events, nx, ny, ntau, nfeatures) "grid_info" — dict with tau0, dtau, x_min, dx "nx", "ny", "ntau", "nfeatures"
- `Requires uproot.` –

## The FNO4d HDF5 schema

### `jetscape.fast_h5_bulk`

python/jetscape/fast_h5_bulk.py

H5BulkWriter -- a pure-Python JETSCAPE module that writes the bulk hydro evolution straight to the FNO4d training HDF5 schema, covering both C++ bulk writer modules.

It inherits from `pyjetscape.JetScapeModuleBase` (via the PyJetScapeModuleBase C++ trampoline), so it can be added to a pipeline with `JetScape.Add()` and its `Exec()` runs once per event, exactly like the C++ writers. See `jetscape.bulk_sources` for the three source modes and what each one replaces.

Why this exists

- `FastRootBulkWriter` writes one event as a single `std::vector<float>`, so events past 2\*\*30-2 bytes trip ROOT's `TBufferFile::WriteByteCount` 30-bit length field (the O+O native event is 1.29 GB; a 199-step one is 1.91 GB, near ROOT's hard ~2 GB wall). HDF5 has no such limit.
- The FNO training pipeline reads HDF5 and currently gets there by post-processing with `root2hdf5/root_to_hdf5.py`. Writing the target schema directly removes that step and the intermediate ROOT file.

Pipeline placement

```text
jetscape.Add(trento)
jetscape.Add(null_predynamics)
jetscape.Add(hydro)        # must come BEFORE the writer
jetscape.Add(writer)       # <- here
```

Usage

```text
from jetscape.fast_h5_bulk import H5BulkWriter

writer = H5BulkWriter(out_file_name="hydro_evo.h5", grid_mode="native")
jetscape.Add(writer)
jetscape.Init(); jetscape.Exec(); jetscape.Finish()   # also finishes the writer:
                  # JetScape::Finish() calls Finish() on every active task
```

`Finish()` is idempotent, and the writer is a context manager, so the safe form is:

```text
with H5BulkWriter(...) as writer:
    ...
```

Reading it back

```text
from jetscape.fast_h5_bulk import read_fast_h5_bulk
d = read_fast_h5_bulk("hydro_evo.h5", entry_stop=1)
evo = d["events"][0]      # (ntau, nx, ny, neta, 4) float32, as read_fast_root_bulk gives
```

FNO4d's own loaders read the file directly -- `read_3d_data_hdf5` and `MultiH5Array` -- with no conversion.

#### `H5BulkWriter(out_file_name: str = 'hydro_evo.h5', grid_mode: str = 'native', tau_stride: int = 1, choose_ntau: int = 0, out_grid: Optional[dict] = None, compression: Optional[str] = DEFAULT_COMPRESSION, keep_bits: Optional[int] = None, clear_after_write: bool = True, force: bool = True, extra_attrs: Optional[dict] = None, verbose: bool = False)`

Bases: `JetScapeModuleBase`

Write the bulk hydro evolution to an FNO4d-schema HDF5 file.

Parameters:

- **`out_file_name`** (`str`, default: `'hydro_evo.h5'` ) – Output path.
- **`grid_mode`** (`('native', 'grid', 'framework')`, default: `"native"` ) – Where the cells come from; see jetscape.bulk_sources.
- **`tau_stride`** (`int`, default: `1` ) – native/grid only: keep every Nth stored MUSIC step.
- **`choose_ntau`** (`int`, default: `0` ) – 0 (default) grows the tau axis to the longest event in this file. A positive value pins it, which is what makes files from separate jobs mergeable -- every file trained on together must share one choose_ntau or MultiH5Array raises. Longer events are then clipped, with a warning per event and a count at Finish().
- **`out_grid`** (`mapping or Grid`, default: `None` ) – grid/framework only: any of x_min dx y_min dy eta_min deta tau_min dtau ntau. Missing/0 means "use the source grid's value". A Grid (e.g. Grid.from_bounds) is used exactly as given; see resolve_out_grid.
- **`compression`** (`(str, mapping or None)`, default: `DEFAULT` ) – jetscape.h5_compression spec for arr (default "blosc-zstd"; "lzf" is the old default). See README_h5_optim.md.
- **`keep_bits`** (`int or None`, default: `None` ) – Round arr to this many float32 mantissa bits (lossy, relative error \<= 2\*\*-(keep_bits+1)); None (default) is bit-exact.
- **`clear_after_write`** (`bool`, default: `True` ) – Release MUSIC's native store after each event. Set False when another writer runs after this one in the same event and still needs the store.

##### `Init() -> None`

Reset per-run state. Called once by JetScape::Init().

##### `Exec() -> None`

Write the current event. Called once per event by the framework.

##### `write_diag(**scalars) -> None`

Add per-event scalars to `diag/` for the event the last `Exec` wrote (a driver's event record: pTHat, Glauber numbers ...); see `FnoH5Writer.write_diag`.

##### `Finish() -> None`

Close the file. Idempotent; must be called explicitly (see module docstring).

##### `get_event_layout() -> dict`

Layout of the most recently written event.

#### `read_fast_h5_bulk(path: str, entry_start: Optional[int] = None, entry_stop: Optional[int] = None) -> dict`

Load a file written by `H5BulkWriter`.

Returns the same dict shape as `jetscape.fast_root_bulk.read_fast_root_bulk`, so the two are interchangeable for plotting: each event is `(ntau, nx, ny, neta, 4)` float32, trimmed to that event's own `ntau_freezeout`.

Native-mode events are large, so use `entry_start` / `entry_stop` to load a subset. FNO4d's `read_3d_data_hdf5` reads the same file in its own (channel-first) layout.

### `jetscape.bulk_sources`

python/jetscape/bulk_sources.py

Where the bulk hydro cells come from, for the HDF5 bulk writer.

Three source modes cover both C++ bulk writer modules:

=========== ========================================== ============================= grid_mode source C++ equivalent =========== ========================================== ============================= native MpiMusic.get_native_evolution_numpy() FastRootBulkWriter, native grid the same, resampled onto a user grid FastRootBulkWriter, grid framework EvolutionHistory.to_numpy_full(), resampled RootBulkWriter =========== ========================================== =============================

`native` is exact: the numpy export walks the same MUSIC store the C++ writer walks and applies the same `static_cast<float>`, so the output is bit-for-bit identical.

`grid` and `framework` agree with the C++ to ~1e-7 relative, not bitwise. Nothing is lost in the export -- `Jetscape::real` is `float` (src/framework/RealType.h:26), so `FluidCellInfo` is already float32 and `to_numpy_full` is a straight copy. The gap is that `EvolutionHistory::get()` does its own blend in float32 and in a different summation order, while this module interpolates in float64; the numpy answer is the more accurate of the two. Compare with `allclose`, never `array_equal`.

XML prerequisites differ by mode and are mutually exclusive:

- `native` / `grid` need `<Hydro><MUSIC><dump_hydro_only>1`, which keeps MUSIC's native store and skips the framework copy.
- `framework` needs `output_evolution_to_memory=1` *without* `dump_hydro_only`, because it reads `bulk_info.data`. It works with any hydro module, not just MUSIC.

`resample` reproduces `EvolutionHistory::get()` exactly (up to float rounding), including its three non-obvious behaviours:

1. A point outside the source box on *any* axis yields an all-zero cell -- a hard cut, not a clamp (`FluidEvolutionHistory.cc:CheckInRange` -> `get` returns a default cell).
1. The eta range check is skipped when the source is boost invariant, but NOT when `neta == 1` and it is not. In that case `EtaMax() == eta_min`, so every output eta other than exactly `eta_min` maps to zeros. We warn rather than silently produce an empty file.
1. Interpolation is quadrilinear: spatial trilinear at `id_tau` and `id_tau+1`, then linear in tau. `CellIndex` clamps every index to `[0, n-1]`, so the upper edge behaves like scipy's `mode="nearest"`.

#### `Grid(nx: int, ny: int, neta: int, ntau: int, x_min: float, dx: float, y_min: float, dy: float, eta_min: float, deta: float, tau_min: float, dtau: float, boost_invariant: bool = False)`

A uniform (tau, x, y, eta) grid, in the same conventions as EvolutionHistory.

##### `from_bulk_info(b, ntau=None, dtau=None)`

Build from a bound EvolutionHistory (`hydro.get_bulk_info()`).

##### `from_bounds(x, y, eta, tau_min, dtau, ntau=0)`

Build from `(min, max, n)` per spatial axis: n cell centres, min..max inclusive.

The step is `(max - min)/(n - 1)`; `n = 1` needs `min == max` and gets step 0. `ntau = 0` leaves the tau extent to `resolve_out_grid`, which then runs to the end of each event's evolution.

#### `FrameworkFrames(bulk_info, tau_stride=1)`

The framework copy as a read-only `(ntau, nx, ny, neta, 4)` array of `(energy_density, vx, vy, vz)` that fetches one tau frame at a time.

The values are those of `event_array(hydro, "framework")` (the same floats, through `EvolutionHistory.frame_numpy`), but only the last frame read is held: whole, that event is 1.3 GB as `to_numpy_full(5)` plus 1.1 GB as its 4-field copy for a 0-10% Au+Au background. Supports what `resample` and the writers use: `shape`, `len()`, `a[k]` and `a[k, ...slices]` with an integer `k`.

#### `NativeFrames(hydro, tau_stride=1)`

MUSIC's native store as a read-only `(ntau, nx, ny, neta, 4)` array of `(energy_density, vx, vy, vz)` that fetches one tau frame at a time.

The values are those of `get_native_evolution_numpy(tau_stride)` (the same floats, through `get_native_frame_numpy`), but only the last frame read is held: whole, the jet leg of a 0-10% Au+Au event is 1.0-1.3 GB. Same interface as `FrameworkFrames`: `shape`, `len()`, `a[k]` and `a[k, ...slices]`.

#### `resolve_out_grid(src, spec=None)`

Resolve a user output-grid spec against the source grid.

`spec` is a mapping of any of `x_min dx y_min dy eta_min deta tau_min dtau ntau`. A missing key, `None` or `0` means "use the source value", and the transverse cell counts are derived as `2*int(|min|/d) + 1` -- both conventions taken from `FastRootBulkWriter::fill_grid` (src/root/FastRootBulkWriter.cc:182-206) so that this reproduces the C++ grid mode rather than inventing a second convention.

That derivation assumes a grid symmetric about the origin, so it does NOT round-trip an even-sized source axis: MUSIC's nx=100, x_min=-15, dx=0.3 comes back as 101 cells spanning -15..+15 rather than -15..+14.7. That is what the C++ does, and it is what you want when reproducing it -- but it makes a poor default, so an empty spec (or one whose values are all 0/None) returns the source grid unchanged instead.

`spec` may instead be a `Grid` (e.g. from `Grid.from_bounds`), which is taken literally -- any origin, any cell count, and 0 is a value rather than "unset". Its `ntau` is an upper bound: 0 runs to the end of the source's tau range, N > 0 keeps at most the first N frames. Never more than the source covers, so a short event does not gain trailing all-zero frames (which would also inflate its ntau_freezeout).

#### `ntau_to_end(src, tau_min, dtau)`

Output frames from `tau_min` in steps of `dtau` up to the source's last frame.

#### `resample(arr, src, out)`

Resample `(ntau, nx, ny, neta, F)` from grid `src` onto grid `out`.

Reproduces `EvolutionHistory::get()`; see the module docstring for the three behaviours that matter. Works one output tau frame at a time, so peak extra memory is a couple of frames rather than a second copy of the event.

The output points form a tensor-product grid (x by y by eta), so spatial trilinear interpolation factors into three 1-D linear passes, one per axis, each a matrix product over every feature at once. That is the arithmetic of scipy's `map_coordinates(order=1, mode="nearest", prefilter=False)` per channel -- float64, cast to float32 per frame -- with the terms summed in a different order (results agree to about 1 ulp), and it replaces ~1,700 single-threaded `map_coordinates` calls per event with three BLAS-backed products per source frame.

#### `event_array(hydro, grid_mode='native', tau_stride=1, out_spec=None)`

Return `(arr, src_grid, out_grid)` for the current event.

`arr` is `(ntau, nx, ny, neta, 4)` float32 with channels `(energy_density, vx, vy, vz)`, on `out_grid`.

#### `framework_frames(hydro, tau_stride=1)`

`(FrameworkFrames, src_grid)` for the framework copy, or None if this build's `EvolutionHistory` has no `frame_numpy` (then use `event_array`).

#### `native_frames(hydro, tau_stride=1)`

`(NativeFrames, src_grid)` for MUSIC's native store, or None if this build has no `get_native_frame_numpy` (then use `event_array`).

#### `framework_store_bytes(bulk_info)`

Rough size of the framework AoS backing `framework` mode.

#### `attrs_from_grids(src, out)`

The 13 FNO4d `_SCALAR_KEYS` plus `choose_ntau`, for grids `src` -> `out`.

#### `music_extra_attrs(src)`

The source-grid parameters, under the names the ROOT writers use.

Not read by any FNO4d loader, but `read_3d_data_hdf5` returns `dict(hf.attrs)`, so these ride along for free and keep provenance comparable with the ROOT output.

### `jetscape.fno_h5_writer`

python/jetscape/fno_h5_writer.py

The FNO4d training HDF5 schema, written one tau frame at a time.

VENDORED from FNO4d `loc_libs/fast_data/writer.py` (`FnoH5Writer`). FNO4d is not a dependency of PyJetscape and that file is self-contained (stdlib + numpy + h5py), so it is copied rather than imported. The invariants below are pinned on the FNO4d side by `FNO4d/tests/test_fast_data_writer.py` -- check there first if the two ever disagree.

Changes from the FNO4d original:

- The `/source/S`, `/source/P_cart` and `/eos` datasets are not written. They carry fast_data's own gridded source term and equation of state, which a MUSIC run does not produce, and they are optional in the schema (no FNO4d reader touches them).
- `arr` is created with `maxshape=(None, 4, nx, ny, neta, None)` and grown with `resize()`. The original pre-allocates, which requires knowing both `nevents` and `choose_ntau` up front; a MUSIC run knows neither until the events have run.
- The tau chunk extent is configurable and defaults to 1, so a single tau frame can be written without a read-modify-write of the whole event. See `write_frame`.
- A second evolution (`add_evolution`, e.g. `arr_bg` for a background/jet pair), ragged per-event tables (`ragged`, e.g. `source/droplets`, `shower/partons`) and per-event scalars (`write_diag`, `diag/<key>`) can be added. All three are written as the events arrive, so a run killed mid-way loses nothing it has counted. Their names and layout follow FastHydro's pair files, so `fasthydro.browse.PairBrowser` reads them.
- Every dataset with a tau axis carries a `tau_axis` attribute, and `repad_to` finds the datasets to grow by it.
- An evolution shared by several events can be stored once and read per event through a *row view* (`write_row_view`): a virtual dataset whose row i is row `rows[i]` of the stored one, in the same file. `PairH5Writer` uses it for reused backgrounds; readers see an ordinary dataset. `repad_to` grows the store and rebuilds the view.
- `compression` takes an `jetscape.h5_compression` spec and defaults to Blosc-zstd with byte shuffle (lzf before); `keep_bits` optionally rounds every evolution to that many float32 mantissa bits. Both are recorded on each evolution dataset (`compression`, `keep_mantissa_bits`). See README_h5_optim.md.

The method names (`add_evolution`, `ragged`) follow FastHydro's `PLAN_consolidate_h5_writer.md`, so this file can later be replaced by the consolidated upstream writer. One deliberate difference: `ragged` takes the group attributes verbatim (`attrs=`) rather than deriving `<field>_columns` names, because the existing files use `droplet_columns` for `droplets` and `vertex_columns` for `vertices`.

`arr` is the contract every FNO4d loader depends on:

```text
arr             (nevents, 4, nx, ny, neta, choose_ntau) float32
ntau_freezeout  (nevents,) int32
tau_freezeout   (nevents,) float32
```

with root attributes `nFeatures, nx, ny, neta, x_min, y_min, eta_min, dx, dy, deta, tau_min, tau_min_MUSIC, dtau` (FNO4d `loc_libs/read_3d_hdf5.py` `_SCALAR_KEYS`) plus `choose_ntau` and `nevents`.

Three details are load-bearing and were checked against FNO4d's data/dAu_25ev_mb.h5:

- The four channels are (energy_density, vx, vy, vz) with CARTESIAN LAB three-velocities.
- `attrs['nevents']` must equal `arr.shape[0]`: read_3d_data_hdf5 trusts the attribute while MultiH5Array trusts the shape, and they must agree.
- Channel 0 must be exactly zero after freeze-out. loc_libs/data/dataset.py:live_tau_lengths derives each event's lifetime from `abs(ev[0]).max(axis=(0,1,2)) > 0`, not from `ntau_freezeout`, so a non-zero tail makes every window look alive. Frames past the end of an event are never written, and HDF5 returns the fill value (exactly 0.0) for the unallocated chunks, so this holds without an explicit pad.

Anything extra written into the root attributes is returned for free by `read_3d_data_hdf5` (which returns `dict(hf.attrs)`), which is how run provenance travels to the training side.

#### `FnoH5Writer(path, attrs, nevents=0, *, compression=DEFAULT_COMPRESSION, keep_bits=None, chunk_events=1, chunk_tau=1, extra_attrs=None, growable_tau=True, force=False)`

Stream events into an FNO4d-schema file, one tau frame at a time.

`arr` grows along the event axis and the tau axis as events arrive, so neither count has to be known up front. `attrs['nevents']` and `attrs['choose_ntau']` are kept equal to `arr.shape[0]` and `arr.shape[5]` after every write, and the file is flushed per event, so a run killed mid-way still produces a file both `read_3d_data_hdf5` and `MultiH5Array` agree on.

Parameters:

- **`path`** (`str`) – Output file path.
- **`attrs`** (`mapping`) – Must contain every key in SCALAR_KEYS plus choose_ntau (the initial tau extent; the dataset grows past it as needed unless growable_tau=False).
- **`nevents`** (`int`, default: `0` ) – Initial event-axis extent. 0 is fine -- the axis grows per event.
- **`compression`** (`(str, mapping or None)`, default: `DEFAULT` ) – jetscape.h5_compression spec for arr and every added evolution: "blosc-zstd" (default), "blosc-lz4", "lzf", "gzip", None, ... Blosc stores MUSIC evolutions 1.4x smaller than lzf at the same read speed; readers need import hdf5plugin (importing jetscape does it).
- **`keep_bits`** (`int or None`, default: `None` ) – Round the evolutions to this many float32 mantissa bits before compressing: lossy, relative error \<= 2\*\*-(keep_bits+1) (12 bits: 1.2e-4, and files about 2x smaller again). None (default) is bit-exact. Exact zeros stay zero.
- **`chunk_events`** (`int`, default: `1` ) – Chunk extent on the event and tau axes. chunk_tau=1 is what makes a single frame write cover exactly one whole chunk (no read-modify-write) and what makes the zero padding of short events cost nothing on disk.
- **`chunk_tau`** (`int`, default: `1` ) – Chunk extent on the event and tau axes. chunk_tau=1 is what makes a single frame write cover exactly one whole chunk (no read-modify-write) and what makes the zero padding of short events cost nothing on disk.
- **`growable_tau`** (`bool`, default: `True` ) – False pins the tau extent at attrs['choose_ntau'], so every file from a campaign shares one value. Callers must then clip longer events themselves.

##### `evolutions`

Names of the evolution datasets, the primary `"arr"` first.

##### `ensure_capacity(nevents=None, choose_ntau=None)`

Grow the event and/or tau axis. Returns True if anything was resized.

Every evolution grows together, so `arr_bg` always has `arr`'s shape. The tau axis never grows past its creation size when `growable_tau=False`; the caller is responsible for clipping in that case.

##### `add_evolution(name, *, fo_suffix)`

Add a second evolution dataset shaped, chunked and compressed like `arr`.

Creates `name` plus `ntau_freezeout{fo_suffix}` and `tau_freezeout{fo_suffix}`. It grows with `arr` from then on (`ensure_capacity`), so the two always have one shape -- readers that subtract them (`PairBrowser`, `wake_pyvista.py`) rely on that. Events written before the call read back as zeros.

##### `ragged(group, offsets_name, fields, *, attrs=None, unit='event', compression='gzip', chunk_rows=65536)`

Add a ragged per-event table; see `RaggedGroup`.

`fields` maps a dataset name inside `group` to `(dtype, row_shape)`; all of them share the one offsets vector `group/offsets_name`. `attrs` is written verbatim onto the HDF5 group (e.g. `{"droplet_columns": [...]}`).

With `unit="event"` the table must receive exactly one `append` per event, and `set_event_meta` raises if it has not. A table created after some events were written starts with that many empty events, so it stays aligned with `arr`.

##### `write_frame(i, t, frame, dataset='arr')`

Write one tau frame of event `i`. `frame` is (4, nx, ny, neta).

With `chunk_tau=1` this covers exactly one whole chunk, so HDF5 never reads a chunk back to merge.

##### `set_event_meta(i, ntau_fo, tau_fo, dataset='arr')`

Record event `i`'s freeze-out scalars for one evolution.

Only the primary `dataset="arr"` call marks the event written (advances `nevents_written` and flushes), so write the other evolutions first. It also checks that every per-event ragged table received exactly this event.

##### `write_diag(i, **scalars)`

Store one number (or short string) per key for event `i` under `diag/<key>`.

One growable dataset per key. Numbers are float64 with NaN where an event did not set the key, keys ending in `seed` are uint64, strings are variable-length -- the same dtypes as fast_data's diag group, but written per event instead of at close.

##### `append_event(i, arr_ev, ntau_fo, tau_fo, dataset='arr')`

Write event `i` in one call. `arr_ev` is (4, nx, ny, neta, ntau).

Kept for parity with the FNO4d original. It holds the whole event in memory; the streaming path is `write_frame` + `set_event_meta`.

#### `RaggedGroup(group, offsets_name, fields, *, n_empty=0, compression='gzip', chunk_rows=65536, unit='event')`

A variable-length per-event table: row datasets plus one offsets vector.

The layout is the one fast_data uses for `source/droplets` and FastHydro for `shower/`: `group/<field>` has shape `(N, *row_shape)`, `group/<offsets_name>` is int64 of length `units_written + 1`, and unit `k` is rows `offsets[k]:offsets[k+1]` of every field. An empty unit is a repeated offset -- empty, not missing.

Unlike those writers, which collect the rows and write them at close, every `append` resizes and writes at once (`maxshape=(None, ...)`), so a crash loses nothing already appended. Create it with `FnoH5Writer.ragged`.

##### `append(rows=None)`

Append one unit. `rows` maps field -> array of shape (n, \*row_shape).

A missing field, or `rows=None`, is zero rows. All fields share the offsets, so they must all get the same number of rows. Returns that number.

##### `append_many(rows, counts)`

Append several consecutive units in one write.

`rows` is as for `append`, with the units' rows concatenated; `counts` is the number of rows of each unit. The file is the same as from one `append` per unit, but much faster when the units are small compared with a chunk: every append re-reads, extends and recompresses the partly filled last chunk, here each chunk is compressed once. Returns the total number of rows.

#### `grid_attrs(nx, ny, neta, dx, dy, deta, tau_min, dtau, choose_ntau, x_min=None, y_min=None, eta_min=None, tau_min_MUSIC=None)`

The root attribute dict for a cell-centred grid symmetric about the origin.

The mins default to -(n-1)/2\*d, so the stored x_min is the first cell CENTRE (as in the MUSIC-produced files), not a box edge.

#### `write_row_view(f, name, store_name, rows_name)`

(Re)build the virtual dataset `name`: row i is `f[store_name][rows[i]]`.

`rows = f[rows_name]` has one entry per event; rows < 0 are not mapped and read as 0.0, like unwritten events. The view has the store's shape on every axis but the first, which is `len(rows)`. The source is this file (HDF5's `"."`), so the file can be moved or copied whole. Views are metadata only; rebuilding one is cheap, and needed whenever the store or `rows` changes shape.

#### `refresh_row_views(f)`

Rebuild every row view in `f` (after its store was resized). Returns their names.

#### `repad_to(paths, choose_ntau=None, *, dry_run=False, verbose=True)`

Grow the tau axis of several files to one `choose_ntau`, in place.

`arr` is rectangular, so every file trained on together must share one `choose_ntau` or `MultiH5Array` raises `Dimension mismatch`. Files written here have `maxshape=(None, 4, nx, ny, neta, None)`, so reconciling them is a metadata resize: no data moves, the file does not grow (the new region is unallocated chunks that read back as the fill value, exactly 0.0), and per-event lifetimes survive.

This is the h5-to-h5 counterpart of `root_to_hdf5.py --global-ntau`.

Parameters:

- **`paths`** (`str or sequence of str`) – Files to reconcile.
- **`choose_ntau`** (`int`, default: `None` ) – Target tau extent. Default: the largest among paths. Pinning a larger value than any file needs is free, so passing the campaign-wide number is safe.
- **`dry_run`** (`bool`, default: `False` ) – Report what would change without writing.
- **`verbose`** (`bool`, default: `True` ) – Print a line per file.

Returns:

- `(target, changed) : (int, list of str)` – The tau extent applied and the files that needed it.

Raises:

- `ValueError` – If the files disagree on (nFeatures, nx, ny, neta) -- they could never be concatenated; if any file is already LONGER than the target, since shrinking an HDF5 dataset discards data permanently; or if a file's tau axis is not growable (e.g. written by FNO4d's own pre-allocating FnoH5Writer), which needs a rewrite rather than a resize.

### `jetscape.h5_compression`

HDF5 compression for the FNO4d `arr` schema: filter specs, mantissa rounding, plugins.

Self-contained (stdlib + numpy + h5py; `hdf5plugin` optional) so it can be vendored verbatim -- js-contrib carries copies in FastHydro (`fast_data/`, via `tools/sync_fast_data.sh`) and PyJetscape (`jetscape/h5_compression.py`). The measurements behind the defaults are in README_h5_optim.md.

Why not lzf: float32 hydro fields are nearly incompressible for a byte-oriented codec without a shuffle filter. lzf stores MUSIC `arr` at 1.11x (fast_data 1.25x); Blosc-zstd with byte shuffle reaches 1.6x (2.0x) and reads as fast or, with `BLOSC_NTHREADS` and whole-event chunks, twice as fast. Bitshuffle loses on lossless data -- the low mantissa bits are noise -- and after `keep_bits` rounding it helps only the LZ4 family; zstd keeps byte shuffle. Rounding to 12 bits takes the total to 2.8-4x.

Spec strings, as accepted by `h5_filter_kwargs` (and every writer's `compression=`)::

```text
"blosc-zstd"            Blosc + zstd level 3, byte shuffle      (DEFAULT)
"blosc-lz4"             Blosc + LZ4 level 5, byte shuffle       (fastest write)
"blosc-<cname>[:level][+shuffle|+bitshuffle|+noshuffle]"
                        any Blosc codec: blosclz lz4 lz4hc snappy zlib zstd
"lzf"                   h5py's built-in LZF (the old default)
"gzip[:level]"          deflate, level 4 by default, with byte shuffle
None / "none"           uncompressed
```

A mapping (h5py keyword arguments, or an `hdf5plugin` filter object) is passed through unchanged. With `keep_bits` set, an LZ4/blosclz spec without an explicit shuffle switches to bitshuffle (the only case where it was measured to win).

Reading a Blosc file needs the filter registered, i.e. `import hdf5plugin` somewhere in the process before the first read -- importing this module does that. Without `hdf5plugin` a Blosc spec falls back to lzf with a warning, so a writer never fails on a missing optional package.

#### `enable_filters()`

Register the `hdf5plugin` filters (Blosc, ...) with HDF5. True if available.

Idempotent and cheap. Call it (or import this module) before opening a file written with a Blosc spec; h5py otherwise fails with "required filter ... is not registered".

#### `h5_filter_kwargs(spec=DEFAULT, keep_bits=None)`

`(kwargs, label)` for `h5py.Group.create_dataset(..., **kwargs)`.

`label` is a canonical string for the dataset's `compression` attribute (see `tag_dataset`), e.g. `"blosc-zstd:3+shuffle"`. Every label except a `custom:` one is itself a valid spec that gives the same filter back.

#### `round_mantissa(a, keep_bits)`

Round float32 `a` to `keep_bits` explicit mantissa bits (round-half-to-even).

The relative error is at most `2**-(keep_bits+1)` (1.2e-4 for 12 bits). Zeros stay exactly zero -- which the `arr` schema relies on after freeze-out -- and so do the sign, infinities and NaNs. `keep_bits=None` or >= 23 returns `a` unchanged (as float32). The zeroed low bits are what lets bitshuffle compress; lossless data keeps them noisy.

#### `tag_dataset(ds, label, keep_bits=None)`

Record the filter and the rounding on `ds` itself, for provenance and readers.

`compression` is the `h5_filter_kwargs` label; `keep_mantissa_bits` is written only for lossy (rounded) data, so its absence means bit-exact.

#### `dataset_keep_bits(ds)`

The `keep_bits` a dataset was written with (None = lossless).

### `jetscape.repad_h5`

python/jetscape/repad_h5.py

Command-line front end for `jetscape.fno_h5_writer.repad_to`.

`arr` is rectangular, so every HDF5 file trained on together must share one `choose_ntau` or `MultiH5Array` raises `Dimension mismatch`. Files written by `H5BulkWriter` leave the tau axis growable, so reconciling a campaign is a metadata resize: no data moves and the files do not grow.

```text
python -m jetscape.repad_h5 run*/hydro_evo.h5                 # to the largest
python -m jetscape.repad_h5 run*/hydro_evo.h5 --choose-ntau 200
python -m jetscape.repad_h5 run*/hydro_evo.h5 --dry-run
```

This is the h5-to-h5 counterpart of `root2hdf5/root_to_hdf5.py --global-ntau`. It lives in its own module rather than in `fno_h5_writer` so that `python -m` does not re-import a module the package `__init__` has already loaded.

## The PyTorch hydro module

### `jetscape.fno_hydro`

python/jetscape/fno_hydro.py

PyFNOHydro — Python FluidDynamics trampoline that runs a PyTorch FNO model as a first-class JETSCAPE hydro module.

Usage (three model-loading approaches):

```text
# Approach 1: JIT-traced .pt  (compatible with C++ FnoHydro traced models)
hydro = PyFNOHydro("models/traced.pt", config)

# Approach 2: checkpoint + Python class
from my_model import FNOModel
net = FNOModel(...)
hydro = PyFNOHydro((net, "checkpoints/epoch50.pt"), config)

# Approach 3: live Python model (no serialisation)
hydro = PyFNOHydro(net, config)
```

All three produce a PyFNOHydro instance that can be added to a JETSCAPE pipeline with:

```text
js.Add(hydro)     # after Add(preeq) and before Add(jloss_mgr)
```

Config source options

config dict (any approach above): Explicitly pass every grid parameter — the original approach.

fno_config_from_xml(user_xml_path): Parse the JETSCAPE user XML with Python's stdlib before constructing the pipeline. No rebuild required. Use when you want to inspect the config before run_manual() is called.

config=None (reads from JetScapeXML via C++ binding): Pass config=None and the module reads values from the already- loaded JETSCAPE XML inside InitializeHydro() — the most JETSCAPE-native approach, exactly like C++ FnoHydro. Requires the shared library to be rebuilt after the binding addition in bind_framework.cc.

#### `PyFNOHydro(model, config=None, *, device: str = 'cpu')`

Bases: `FluidDynamics`

JETSCAPE FluidDynamics module backed by a PyTorch FNO model.

The model receives initial conditions from the pre-equilibrium module and writes hydrodynamic evolution history back into the C++ EvolutionHistory, including finding the constant-temperature freeze-out surface.

Parameters:

- **`model`** (`str | (Module, str) | Module`) – str — path to a TorchScript .pt file (loaded with torch.jit.load). Compatible with traced models from C++ FnoHydro. (nn.Module, str) — tuple of (model instance, checkpoint path). The checkpoint is loaded with torch.load and load_state_dict. nn.Module — live model used as-is; no loading step.
- **`config`** (`dict | None`, default: `None` ) – When a dict, must contain: nx : int — FNO grid points in x ny : int — FNO grid points in y ntau : int — number of time steps predicted by the model neta : int — eta grid points (1 for boost-invariant) n_features : int — number of model input/output features (3 or 4) x_min : float — minimum x [fm] (e.g. -7.5) y_min : float — minimum y [fm] (e.g. -7.5) dtau : float — tau step [fm/c] deta : float — eta step (0 for boost-invariant) T_freeze : float — freeze-out temperature [GeV] Optional keys in dict: device : str — 'cpu', 'cuda', 'mps' (default: 'cpu') tau_normalise : bool — whether model output is ed\*tau (default True for n_features==3, False for n_features==4) When None, grid parameters are read from the loaded JETSCAPE XML (<Hydro><FNO> block) inside InitializeHydro(), i.e. after js.Init() has opened the XML files. This is the most JETSCAPE- native approach and mirrors what C++ FnoHydro does. Requires that bind_framework.cc has been rebuilt with the get_xml_element\_\* binding additions.
- **`device`** (`str`, default: `'cpu'` ) – PyTorch device string ('cpu', 'cuda', 'mps'). Only used when config is None; when config is a dict, use config.get('device', 'cpu') instead.

##### `InitializeHydro(params)`

Read grid configuration and initialise hydro.

Called by FluidDynamics::Init() after the XML is loaded. If *config* was `None` at construction, grid parameters are read from `<Hydro><FNO>` in the JETSCAPE XML via `get_xml_element_*`.

##### `EvolveHydro()`

1. Fetch pre-equilibrium data as numpy (via get_preeq_pointer()).
1. Rebin to FNO grid → input tensor (1, n_features, nx, ny, ntau).
1. Run model (GIL is still held; heavy compute is in torch C++).
1. Un-normalise output (ed\*tau → ed for n_features==3).
1. Set grid info on bulk_info.
1. Store fluid cells from output numpy array.
1. Mark hydro as FINISHED.
1. Find freeze-out surface.

#### `fno_config_from_xml(xml_path: str, *, device: str = 'cpu') -> dict`

Build an FNO config dict by parsing a JETSCAPE user XML file.

Reads `<Hydro><FNO>` tags using Python's stdlib `xml.etree.ElementTree` — no C++ framework or rebuild needed. Can be called before the pipeline is constructed.

Parameters:

- **`xml_path`** (`str`) – Path to the JETSCAPE user XML file (the one passed as --user).
- **`device`** (`str`, default: `'cpu'` ) – PyTorch device string to embed in the returned dict ('cpu', 'cuda', 'mps').

Returns:

- `dict` – Config dict compatible with PyFNOHydro(model, config).

Raises:

- `ValueError` – If <Hydro><FNO> block or a required tag is missing.

## Conversions

### `jetscape.utils`

python/jetscape/utils.py

Data conversion utilities between JETSCAPE C++ objects and numpy/torch tensors.

Functions:

- **`bulk_info_to_numpy`** – Convert EvolutionHistory.data → np.ndarray shape (ntau, nx, ny, n_features)
- **`bulk_info_to_tensor`** – Convert EvolutionHistory.data → torch.Tensor shape (1, n_features, nx, ny, ntau)
- **`numpy_to_bulk_info`** – Write np.ndarray (n_features, nx, ny, ntau) back into hydro.bulk_info.data via hydro.store_fluid_cells_from_numpy().
- **`tensor_to_bulk_info`** – Write torch.Tensor (1, n_features, nx, ny, ntau) back into hydro.bulk_info.data.
- **`preeq_to_numpy`** – Pack all pre-equilibrium stress-energy fields into a single numpy array shaped (nx_preq * ny_preq, n_preeq_fields).
- **`rebin_preeq_to_fno_grid`** – Nearest-neighbour rebinning from the pre-equilibrium grid to the FNO grid. Returns a numpy array shaped (n_features, nx_fno, ny_fno, 1).
- **`shower_to_networkx`** – Convert a PartonShower to a networkx.DiGraph where nodes are splitting vertices and edges are partons.

#### `bulk_info_to_numpy(bulk_info, n_features: int) -> np.ndarray`

Convert EvolutionHistory.data to a numpy array.

Parameters:

- **`bulk_info`** (`EvolutionHistory`) – The C++ EvolutionHistory object (from FluidDynamics.get_bulk_info()).
- **`n_features`** (`int`) – Number of features per cell to extract. Feature mapping (same as FnoHydro C++ convention): 0: energy_density 1: temperature 2: vx 3: vy

Returns:

- `np.ndarray, shape (ntau, nx, ny, n_features), dtype float32` –

#### `bulk_info_to_numpy_slow(bulk_info, n_features: int) -> np.ndarray`

Pure-Python fallback: convert EvolutionHistory.data to a numpy array.

Identical output to bulk_info_to_numpy() but uses a Python-level triple loop with per-cell C++ boundary crossings — kept for benchmarking and as a reference implementation.

Returns:

- `np.ndarray, shape (ntau, nx, ny, n_features), dtype float32` –

#### `bulk_info_to_tensor(bulk_info, n_features: int, device: str = 'cpu') -> 'torch.Tensor'`

Convert EvolutionHistory.data to a PyTorch tensor.

Returns:

- `(Tensor, shape(1, n_features, nx, ny, ntau))` – FNO convention: batch × features × spatial × time.

#### `numpy_to_bulk_info(arr: np.ndarray, hydro) -> None`

Write a numpy array back into hydro.bulk_info.data.

Parameters:

- **`arr`** (`np.ndarray, shape (n_features, nx, ny, ntau), dtype float32`) – Feature layout: [energy_density, temperature, vx, vy, …] Values must already be in physical units (un-normalised).
- **`hydro`** (`FluidDynamics(PyFluidDynamics)`) – The hydro module whose bulk_info will be populated.

#### `tensor_to_bulk_info(tensor: 'torch.Tensor', hydro) -> None`

Write a PyTorch tensor back into hydro.bulk_info.data.

Parameters:

- **`tensor`** (`(Tensor, shape(1, n_features, nx, ny, ntau))`) – FNO output convention. Values must be in physical units.
- **`hydro`** (`FluidDynamics(PyFluidDynamics)`) – The hydro module whose bulk_info will be populated.

#### `preeq_to_numpy(preeq, field_names: list[str] | None = None) -> np.ndarray`

Pack pre-equilibrium fields into a 2D numpy array.

Parameters:

- **`preeq`** (`PreequilibriumDynamics`) – The C++ pre-equilibrium module (from FluidDynamics.get_preeq_pointer()).
- **`field_names`** (`list of str`, default: `None` ) – Subset of fields to pack. Defaults to all 17 fields in PREEQ_FIELD_NAMES. Each name must correspond to a get_numpy() method on PreequilibriumDynamics.

Returns:

- `np.ndarray, shape (n_cells, n_fields), dtype float64` – n_cells = nx_preq * ny_preq (flattened).

#### `rebin_preeq_to_fno_grid(preeq, ini, nx_fno: int, ny_fno: int, x_min_fno: float, y_min_fno: float, dx_fno: float, dy_fno: float, n_features: int = 4) -> np.ndarray`

Nearest-neighbour rebin from the pre-equilibrium grid to the FNO grid.

Replicates the rebinning logic in FnoHydro::EvolveHydro() (C++).

Parameters:

- **`preeq`** (`PreequilibriumDynamics`) –
- **`ini`** –
- **`nx_fno`** (`int`) –
- **`ny_fno`** (`int`) –
- **`x_min_fno`** (`float (typically negative, e.g. -7.5 fm)`) –
- **`y_min_fno`** (`float (typically negative, e.g. -7.5 fm)`) –
- **`dx_fno`** (`float`) –
- **`dy_fno`** (`float`) –
- **`n_features`** (`int`, default: `4` ) – 3 → [ed, vx, vy]; 4 → [ed, T, vx, vy] For n_features == 3 the input energy density is NOT multiplied by tau here — that normalisation is applied by the Python EvolveHydro() just before feeding the tensor to the model.

Returns:

- `np.ndarray, shape (n_features, nx_fno, ny_fno, 1), dtype float32` – Ready to be repeated along the last axis (ntau times) and fed to the FNO model as shape (n_features, nx_fno, ny_fno, ntau) or unsqueeze(0) → (1, n_features, nx_fno, ny_fno, ntau).

#### `shower_to_networkx(ps)`

Convert a PartonShower to a networkx.DiGraph.

Each **node** is a splitting vertex; each **edge** is a parton propagating between two vertices. Node and edge integer IDs match the GTL node IDs returned by vertices_to_numpy() and to_numpy() so the graph topology is preserved exactly.

Node attributes

x, y, z : float [fm] Space position of the splitting vertex. t : float [fm/c] Time of the splitting vertex.

Edge attributes

pid : int PDG particle ID. pstat : int Particle status code. px, py, pz : float [GeV] 3-momentum components. E : float [GeV] Energy. x, y, z : float [fm] Production position (= source vertex position). t : float [fm/c] Production time (= source vertex time).

Parameters:

- **`ps`** (`PartonShower`) – A C++ PartonShower object, e.g. from JetEnergyLossManager.get_showers().

Returns:

- `DiGraph` –

Examples:

```pycon
>>> from jetscape.utils import shower_to_networkx
>>> sm = JetScapeSignalManager.Instance()
>>> jm = sm.GetJetEnergyLossManagerPointer()
>>> for ps in jm.get_showers():
...     G = shower_to_networkx(ps)
...     print(G.number_of_nodes(), G.number_of_edges())
...     final = [n for n, d in G.out_degree() if d == 0]
...     print("final-state partons:", len(final))
```
