# C++ bindings

`jetscape.pyjetscape_core`: the pybind11 bindings of X-SCAPE, with the C++ class and method names and argument order. Every module base class (`JetEnergyLoss`, `HardProcess`, `FluidDynamics`, ...) can be subclassed in Python; [Writing modules in Python](https://jhputschke.github.io/js-contrib/python/modules/index.md) shows how. The methods and properties of the Pythonic layer ([`jetscape.pythonic`](https://jhputschke.github.io/js-contrib/python/api/run/index.md)) appear here under the class they are installed on.

This page is generated from the type stubs `pyjetscape_core.pyi`, which `utils/gen_pyjetscape_stubs.py` writes from a build (see [Building this site](https://jhputschke.github.io/js-contrib/reference/site/index.md)).

## `jetscape.pyjetscape_core`

pyjetscape_core — Python bindings for the JETSCAPE heavy-ion simulation framework.

Exposes C++ classes and the module factory so that Python code can:

- Drive a full JETSCAPE simulation (JetScape, Init/Exec/Finish)
- Instantiate registered C++ modules by name (create_module)
- Subclass FluidDynamics in Python (PyFNOHydro trampoline)
- Subclass JetScapeModuleBase in Python (PyBulkRootWriter and similar)
- Add the C++ FastRootBulkWriter to a pipeline (ROOT builds, HAS_ROOT)
- Access InitialState and PreequilibriumDynamics data as numpy arrays
- Read EvolutionHistory and SurfaceCellInfo after hydro finishes
- Query globally registered modules via JetScapeSignalManager

### `ERROR: HydroStatus`

### `EVOLVING: HydroStatus`

### `FINAL_PARTON_COLUMNS: tuple = ('shower', 'pid', 'pstat', 'E', 'px', 'py', 'pz', 't', 'x', 'y', 'z', 'mass', 'col', 'acol')`

### `FINISHED: HydroStatus`

### `HAS_GZIP_READER: bool = True`

### `HAS_NEXT_TASK_NUMBER: bool = True`

### `HAS_ROOT: bool = True`

### `ICOLORED_HAS_COLOUR_AUDIT: bool = True`

### `ICOLORED_HAS_COLOUR_CLOSURE: bool = True`

### `ICOLORED_HAS_INPUT_RECORD: bool = True`

### `INITIALIZED: HydroStatus`

### `ISS_DECAYS_SMASH_LIST: bool = True`

### `MUSIC_BACKEND: str = 'music4gpu-cuda'`

### `NOT_START: HydroStatus`

### `PGUN_HAS_POINTS: bool = True`

### `PYTHIA_GUN_HAS_EXTRA_HARD: bool = True`

### `PYTHIA_GUN_HAS_PARTON_Y_CUT: bool = True`

### `PYTHIA_GUN_HAS_PTHAT_BINS: bool = True`

### `PYTHIA_GUN_HAS_Z_TRIGGER: bool = True`

### `PYTHIA_ISR_GUN_HAS_PARTON_Y_CUT: bool = True`

### `PYTHIA_ISR_GUN_HAS_PTHAT_BINS: bool = True`

### `PYTHIA_VERSION: int = 8312`

### `SCATTER_COLUMNS: tuple = ('scatter', 'pthat', 'code', 'x', 'y', 'n_partons', 'E', 'pt')`

### `SOFT_HAS_PRIMARIES: bool = True`

### `SOFT_HAS_SAMPLE_CELLS: bool = True`

### `SURFACE_CELL_COLUMNS: tuple = ('tau', 'x', 'y', 'eta', 'ds0', 'ds1', 'ds2', 'ds3', 'u0', 'u1', 'u2', 'u3', 'e', 'T', 'P', 'nB', 'nQ', 'nS', 'muB', 'muQ', 'muS', 'pi00', 'pi01', 'pi02', 'pi03', 'pi11', 'pi12', 'pi13', 'pi22', 'pi23', 'pi33', 'Pi')`

### `Afterburner()`

Bases: `JetScapeModuleBase`

Hadronic transport after particlization (SMASH in C++).

A Python afterburner subclasses it, takes its input in ExecuteTask() with GatherAfterburnerHadrons() (one list per oversample: the soft hadrons plus the jet hadrons), propagates them, and returns the result from GetCurrentHadronList(). register_module("CustomModuleX", cls) and make the XML use it.

#### `GatherAfterburnerHadrons() -> list`

The hadrons for the transport: per oversample, the soft and the jet hadrons.

#### `GetCurrentHadronList() -> list[Hadron]`

#### `GetFragmentationHadrons() -> list`

#### `GetSoftParticlizationHadrons() -> list`

### `CausalLiquefier()`

Bases: `LiquefierBase`

Causal (telegraph) diffusion of deposited energy-momentum. The default constructor reads from the loaded JETSCAPE XML; the 4-argument one takes the grid directly and needs no XML.

#### `bjorken_frame: bool`

True: the kernel in the frame of the Bjorken flow at the droplet's eta_s (bjorken, the default); False: the lab frame (lab).

#### `c_diff: float`

#### `d_diff: float`

#### `deta: float`

#### `dtau: float`

#### `dx: float`

#### `dy: float`

#### `fluid_e_min: float`

fluid frame: below this energy density [GeV/fm^3] at the droplet, the Bjorken frame.

#### `fluid_v_max: float`

fluid frame: the cap on the fluid's speed relative to the Bjorken flow.

#### `frame: str`

The frame of the causal diffusion: 'bjorken' (the default), 'lab' or 'fluid' (the background fluid's rest frame at the droplet).

#### `gamma_relax: float`

#### `tau_delay: float`

#### `time_relax: float`

#### `width_delta: float`

#### `InitializeParameters() -> None`

Re-read the XML block and recompute c_diff / gamma_relax.

#### `get_peta(px: typing.SupportsFloat | typing.SupportsIndex, pz: typing.SupportsFloat | typing.SupportsIndex, eta: typing.SupportsFloat | typing.SupportsIndex) -> float`

#### `get_ptau(px: typing.SupportsFloat | typing.SupportsIndex, pz: typing.SupportsFloat | typing.SupportsIndex, eta: typing.SupportsFloat | typing.SupportsIndex) -> float`

#### `j_delta(t: typing.SupportsFloat | typing.SupportsIndex, r: typing.SupportsFloat | typing.SupportsIndex) -> float`

#### `j_smooth(t: typing.SupportsFloat | typing.SupportsIndex, r: typing.SupportsFloat | typing.SupportsIndex) -> float`

#### `kernel_frame_velocity(droplet: Droplet) -> typing.Annotated[list[float], 'FixedSize(3)']`

The kernel frame's velocity relative to the Bjorken flow at the droplet (0 for lab and bjorken).

#### `kernel_j(t: typing.SupportsFloat | typing.SupportsIndex, r: typing.SupportsFloat | typing.SupportsIndex) -> float`

#### `kernel_rho(t: typing.SupportsFloat | typing.SupportsIndex, r: typing.SupportsFloat | typing.SupportsIndex) -> float`

#### `params() -> dict`

The five parameters fast_data.liquefier.LiquefierParams needs, read from the live C++ object rather than duplicated on the Python side.

#### `rho_delta(t: typing.SupportsFloat | typing.SupportsIndex, r: typing.SupportsFloat | typing.SupportsIndex) -> float`

#### `rho_smooth(t: typing.SupportsFloat | typing.SupportsIndex, r: typing.SupportsFloat | typing.SupportsIndex) -> float`

#### `set_t_delay(tau_delay: typing.SupportsFloat | typing.SupportsIndex) -> None`

#### `uses_fluid_velocity(droplet: Droplet) -> bool`

Whether this droplet's kernel is boosted with the fluid's velocity.

### `Droplet`

A unit of four-momentum handed from the jet to the medium. Position is Milne (tau, x, y, eta); momentum is Cartesian (E, px, py, pz).

#### `get_efluid() -> float`

The background's energy density [GeV/fm^3] there; < 0: not looked up.

#### `get_pmu() -> typing.Annotated[list[float], 'FixedSize(4)']`

(E, px, py, pz)

#### `get_vfluid() -> typing.Annotated[list[float], 'FixedSize(3)']`

The background fluid's velocity (lab Cartesian vx, vy, vz) where the droplet was made.

#### `get_xmu() -> typing.Annotated[list[float], 'FixedSize(4)']`

(tau, x, y, eta)

#### `set_fluid(v: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], 'FixedSize(3)'], e: typing.SupportsFloat | typing.SupportsIndex) -> None`

### `EvolutionHistory()`

Stores the hydrodynamic evolution history on a (tau,x,y,eta) grid.

Two storage modes: * `data` — vector of FluidCellInfo objects (used by jet quenching) * `data_vector` — flat float32 vector (used by external hydro modules)

Grid parameters: tau_min, dtau, ntau, x_min, dx, nx, y_min, dy, ny, eta_min, deta, neta, boost_invariant, tau_eta_is_tz.

#### `boost_invariant: bool`

#### `data_info: list[str]`

Entry names of one data_vector record (empty: the FluidCellInfo layout). Set through SetDataInfo, which also resolves them.

#### `deta: float`

#### `dtau: float`

#### `dx: float`

#### `dy: float`

#### `eta_min: float`

#### `neta: int`

#### `ntau: int`

#### `nx: int`

#### `ny: int`

#### `tau_eta_is_tz: bool`

#### `tau_min: float`

#### `x_min: float`

#### `y_min: float`

#### `EtaMax() -> float`

#### `EtaMin() -> float`

#### `FromVector(data_: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], data_info_: collections.abc.Sequence[str], tau_min: typing.SupportsFloat | typing.SupportsIndex, dtau: typing.SupportsFloat | typing.SupportsIndex, x_min: typing.SupportsFloat | typing.SupportsIndex, dx: typing.SupportsFloat | typing.SupportsIndex, nx: typing.SupportsInt | typing.SupportsIndex, y_min: typing.SupportsFloat | typing.SupportsIndex, dy: typing.SupportsFloat | typing.SupportsIndex, ny: typing.SupportsInt | typing.SupportsIndex, eta_min: typing.SupportsFloat | typing.SupportsIndex, deta: typing.SupportsFloat | typing.SupportsIndex, neta: typing.SupportsInt | typing.SupportsIndex, tau_eta_is_tz: bool) -> None`

Populate data_vector and grid metadata from a flat float32 vector.

Parameters match EvolutionHistory::FromVector exactly. data\_ must have length ntau * nx * ny * neta * len(data_info\_).

#### `Tau0() -> float`

#### `TauMax() -> float`

#### `XMax() -> float`

#### `XMin() -> float`

#### `YMax() -> float`

#### `YMin() -> float`

#### `cell_index(id_tau: typing.SupportsInt | typing.SupportsIndex, id_x: typing.SupportsInt | typing.SupportsIndex, id_y: typing.SupportsInt | typing.SupportsIndex, id_eta: typing.SupportsInt | typing.SupportsIndex) -> int`

Return flat index for grid position (id_tau, id_x, id_y, id_eta).

#### `clear_up_evolution_data() -> None`

Clear the stored cells (data and data_vector).

#### `data_vector_numpy() -> numpy.typing.NDArray[numpy.float32]`

Zero-copy numpy float32 view of the raw data_vector buffer. Shape: (ntau * nx * ny * neta * n_fields,). The field layout is described by data_info. Valid as long as the EvolutionHistory object is alive.

#### `frame_numpy(id_tau: typing.SupportsInt | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.float32]`

One tau frame of bulk_info.data, shape (nx, ny, neta, 4), float32: [energy_density, vx, vy, vz]. The same values as `to_numpy_full(5)[id_tau][..., [0, 2, 3, 4]]`.

#### `get(tau: typing.SupportsFloat | typing.SupportsIndex, x: typing.SupportsFloat | typing.SupportsIndex, y: typing.SupportsFloat | typing.SupportsIndex, eta: typing.SupportsFloat | typing.SupportsIndex) -> FluidCellInfo`

Return interpolated FluidCellInfo at continuous space-time coordinates (tau, x, y, eta).

Uses trilinear interpolation over the stored grid. Coordinates outside the grid are clamped to the boundary.

Parameters:

- **`tau`** (`SupportsFloat | SupportsIndex`) –
- **`x`** (`SupportsFloat | SupportsIndex`) –
- **`y`** (`SupportsFloat | SupportsIndex`) –
- **`eta`** (`SupportsFloat | SupportsIndex`) –

Returns:

- `FluidCellInfo with interpolated energy_density, temperature,` –
- `vx, vy, vz, etc.` –

#### `get_at_time_step(id_tau: typing.SupportsInt | typing.SupportsIndex, x: typing.SupportsFloat | typing.SupportsIndex, y: typing.SupportsFloat | typing.SupportsIndex, eta: typing.SupportsFloat | typing.SupportsIndex) -> FluidCellInfo`

Return FluidCellInfo at a fixed tau step and continuous (x, y, eta).

Parameters:

- **`id_tau`** (`int — tau grid index`) –
- **`x`** (`SupportsFloat | SupportsIndex`) –
- **`y`** (`SupportsFloat | SupportsIndex`) –
- **`eta`** (`SupportsFloat | SupportsIndex`) –

#### `get_data_size() -> int`

Number of stored cells, in either layout.

#### `get_fluid_cell(id_tau: typing.SupportsInt | typing.SupportsIndex, id_x: typing.SupportsInt | typing.SupportsIndex, id_y: typing.SupportsInt | typing.SupportsIndex, id_eta: typing.SupportsInt | typing.SupportsIndex) -> FluidCellInfo`

Return FluidCellInfo at grid index (id_tau, id_x, id_y, id_eta).

#### `to_numpy(n_features: typing.SupportsInt | typing.SupportsIndex = 4) -> numpy.typing.NDArray[numpy.float32]`

Convert bulk_info.data to a numpy float32 array in a single C++ pass, avoiding per-cell pybind11 overhead.

Parameters:

- **`n_features`** (`int (1–6`, default: `4)` ) – Number of fields to extract per cell. Layout: [energy_density, temperature, vx, vy, entropy_density, pressure]

Returns:

- `np.ndarray, shape (ntau, nx, ny, n_features), dtype float32` –

#### `to_numpy_full(n_features: typing.SupportsInt | typing.SupportsIndex = 5) -> numpy.typing.NDArray[numpy.float32]`

Convert bulk_info.data to a numpy float32 array keeping the full eta (rapidity) axis, in a single C++ pass.

Parameters:

- **`n_features`** (`int (1–6`, default: `5)` ) – Number of fields per cell. Layout: [energy_density, temperature, vx, vy, vz, entropy_density]

Returns:

- `np.ndarray, shape (ntau, nx, ny, neta, n_features), dtype float32` –

#### `uses_data_vector() -> bool`

True if cells are the flat data_vector described by data_info (e.g. MUSIC's slim copy), False for the FluidCellInfo vector.

### `FastRootBulkWriter()`

Bases: `JetScapeModuleBase`

Hydro-only ROOT writer for the MUSIC evolution.

Reads MUSIC's native in-memory store directly and never builds the framework bulk_info.data, so it is much faster and lighter than RootBulkWriter / PyBulkRootWriter.

Registered module name: `"FastRootBulkWriter"` — instantiate via::

```text
writer = create_module("FastRootBulkWriter")
```

or let the XML task list create it from a top-level `<FastRootBulkWriter>` block.

Requirements: * `<Hydro><MUSIC><dump_hydro_only>1` and `<output_evolution_to_memory>1` (MUSIC is the only supported hydro). * Add it to the pipeline AFTER the hydro module. * Settings come from the `<FastRootBulkWriter>` XML block (out_file_name, grid_mode = native|grid, tau_stride, and the user grid for grid mode).

The ROOT file is written and closed by `Finish()` (called by `JetScape.Finish()`). Read it back with `jetscape.fast_root_bulk.read_fast_root_bulk()`.

#### `GetDtau() -> float`

Return the written tau step (includes tau_stride) [fm/c].

#### `GetGridMode() -> str`

Return the grid mode: "native" (MUSIC grid) or "grid" (user grid).

#### `GetNFeatures() -> int`

Return the number of features per cell (4: energy_density, vx, vy, vz).

#### `GetNeta() -> int`

#### `GetNtauWritten() -> int`

Return the number of tau steps in the last written event.

#### `GetNumberOfEventsWritten() -> int`

Return the number of events filled into the tree so far.

#### `GetNx() -> int`

#### `GetNy() -> int`

#### `GetOutFileName() -> str`

Return the output ROOT file name ().

#### `GetTauFreezeout() -> float`

Return tau one step past the last stored MUSIC step [fm/c].

#### `GetTauMin() -> float`

Return the tau of the first written step [fm/c].

#### `GetTauStride() -> int`

Return the native-mode tau stride (every N-th stored step).

#### `IsFileOpen() -> bool`

Return True while the ROOT file is open (between the first event and Finish()).

#### `get_event_layout() -> dict`

Return the writer state as a dict.

Keys

out_file_name, grid_mode, tau_stride, events_written, shape (ntau, nx, ny, neta, nFeatures) of the last written event, tau_min, dtau, tau_freezeout.

### `FluidCellInfo()`

#### `bulk_Pi: float`

Bulk viscous pressure [GeV/fm^3].

#### `energy_density: float`

Local energy density [GeV/fm^3].

#### `entropy_density: float`

Local entropy density [1/fm^3].

#### `mu_B: float`

Net baryon chemical potential [GeV].

#### `mu_C: float`

Net charge chemical potential [GeV].

#### `mu_S: float`

Net strangeness chemical potential [GeV].

#### `pressure: float`

Thermal pressure [GeV/fm^3].

#### `qgp_fraction: float`

QGP fraction (QGP+HRG phase).

#### `temperature: float`

Local temperature [GeV].

#### `vx: float`

Flow velocity vx.

#### `vy: float`

Flow velocity vy.

#### `vz: float`

Flow velocity vz.

### `FluidDynamics()`

Bases: `JetScapeModuleBase`

Base class for hydrodynamics modules.

Subclass this in Python to inject a custom hydro implementation (e.g. an FNO model) into the JETSCAPE pipeline:

```text
class MyHydro(pyjetscape.FluidDynamics):
    def InitializeHydro(self, params):
        ...  # read config
    def EvolveHydro(self):
        ...  # run model, call store_fluid_cells_from_numpy()
```

The trampoline class PyFluidDynamics (transparent to Python) dispatches virtual C++ calls to the Python override automatically.

#### `Clear() -> None`

#### `EvolveHydro() -> None`

Override in Python to run hydro evolution (called from Exec()).

#### `Exec() -> None`

Call EvolveHydro().

#### `FindAConstantTemperatureSurface(T_sw: typing.SupportsFloat | typing.SupportsIndex) -> list[SurfaceCellInfo]`

Find the constant-temperature surface and return list of cells.

#### `GetHydroEndTime() -> float`

#### `GetHydroFreezeOutTemperature() -> float`

#### `GetHydroStartTime() -> float`

#### `GetHydroStatus() -> int`

#### `Init() -> None`

Read XML, retrieve ini/pre_eq pointers, call InitializeHydro().

#### `InitializeHydro(parameter_list: Parameter) -> None`

Override in Python to initialise hydro (called from Init()).

#### `SetHydroStartTime(tau0: typing.SupportsFloat | typing.SupportsIndex) -> None`

Set hydro_tau_0 (uninitialised by the FluidDynamics constructor).

#### `SetId(id: str) -> None`

Set the module identifier string (e.g. 'PyFNOHydro').

#### `add_a_liquefier(liquefier: typing.Any) -> None`

Attach a LiquefierBase (or CausalLiquefier) to this hydro. The hydro keeps a weak reference: keep the liquefier object alive (as JetScape does).

#### `clearSurfaceCellVector() -> None`

Clear the freeze-out surface cell list.

#### `clear_up_evolution_data() -> None`

Clear the FluidCellInfo data in bulk_info.

#### `find_freezeout_surface(T_sw: typing.SupportsFloat | typing.SupportsIndex, dtau: typing.SupportsFloat | typing.SupportsIndex = 0.0, dx: typing.SupportsFloat | typing.SupportsIndex = 0.0, deta: typing.SupportsFloat | typing.SupportsIndex = 0.0) -> None`

Find the iso-temperature freeze-out surface and store it.

Calls clearSurfaceCellVector() + FindAConstantTemperatureSurface() and stores the result in surfaceCellVector\_ so the framework can retrieve it via getSurfaceCellVector(). Requires hydro_status to be FINISHED and bulk_info.data to be populated.

Parameters:

- **`T_sw`** (`float`) – Switch (freeze-out) temperature [GeV].
- **`dtau`** (`float`, default: `0.0` ) – Cornelius lattice spacing (dy = dx). 0 keeps SurfaceFinder's defaults (0.1 fm, 0.2 fm, 0.2).
- **`dx`** (`float`, default: `0.0` ) – Cornelius lattice spacing (dy = dx). 0 keeps SurfaceFinder's defaults (0.1 fm, 0.2 fm, 0.2).
- **`deta`** (`float`, default: `0.0` ) – Cornelius lattice spacing (dy = dx). 0 keeps SurfaceFinder's defaults (0.1 fm, 0.2 fm, 0.2).

#### `getSurfaceCellVector() -> list[SurfaceCellInfo]`

Return a copy of the freeze-out surface cell vector.

#### `get_bulk_info() -> EvolutionHistory`

Return a reference to the EvolutionHistory (read-only).

#### `get_bulk_info_mutable() -> EvolutionHistory`

Return a mutable reference to the EvolutionHistory.

#### `get_hydro_cell(t: typing.SupportsFloat | typing.SupportsIndex, x: typing.SupportsFloat | typing.SupportsIndex, y: typing.SupportsFloat | typing.SupportsIndex, z: typing.SupportsFloat | typing.SupportsIndex) -> FluidCellInfo`

GetHydroInfo() at Cartesian (t, x, y, z), returned by value. Never raises and never returns null: out-of-range gives a zero cell.

#### `get_hydro_status_int() -> int`

Return the raw HydroStatus integer.

#### `get_ini_pointer() -> InitialState`

Return shared_ptr to the InitialState module.

#### `get_liquefier() -> typing.Any`

The attached liquefier, or None.

#### `get_out_of_range_count() -> int`

Number of GetHydroInfo() queries that fell outside the stored grid (and so saw vacuum). CheckInRange never throws, so this is the only signal.

#### `get_preeq_pointer() -> PreequilibriumDynamics`

Return shared_ptr to the PreequilibriumDynamics module.

#### `get_preserve_bulk_info() -> bool`

#### `get_surface_cells() -> list[SurfaceCellInfo]`

Return a mutable reference to the surface cell vector.

#### `reset_out_of_range_count() -> None`

#### `set_hydro_grid_info(tau_min: typing.SupportsFloat | typing.SupportsIndex, dtau: typing.SupportsFloat | typing.SupportsIndex, ntau: typing.SupportsInt | typing.SupportsIndex, x_min: typing.SupportsFloat | typing.SupportsIndex, dx: typing.SupportsFloat | typing.SupportsIndex, nx: typing.SupportsInt | typing.SupportsIndex, y_min: typing.SupportsFloat | typing.SupportsIndex, dy: typing.SupportsFloat | typing.SupportsIndex, ny: typing.SupportsInt | typing.SupportsIndex, eta_min: typing.SupportsFloat | typing.SupportsIndex, deta: typing.SupportsFloat | typing.SupportsIndex, neta: typing.SupportsInt | typing.SupportsIndex, boost_inv: bool = True, tau_eta_is_tz: bool = False) -> None`

Set bulk_info grid metadata. Must be called before store_fluid_cells_from_numpy() in EvolveHydro().

Parameters:

- **`tau_min`** (`(float, float, int)`) –
- **`dtau`** (`(float, float, int)`) –
- **`ntau`** (`(float, float, int)`) –
- **`x_min`** (`SupportsFloat | SupportsIndex`) –
- **`dx`** (`SupportsFloat | SupportsIndex`) –
- **`nx`** (`SupportsFloat | SupportsIndex`) –
- **`y_min`** (`SupportsFloat | SupportsIndex`) –
- **`dy`** (`SupportsFloat | SupportsIndex`) –
- **`ny`** (`SupportsFloat | SupportsIndex`) –
- **`eta_min`** (`(float, float, int)`) –
- **`deta`** (`(float, float, int)`) –
- **`neta`** (`(float, float, int)`) –
- **`boost_inv`** (`bool`, default: `True` ) –
- **`tau_eta_is_tz`** (`bool`, default: `False` ) –

#### `set_hydro_status_finished() -> None`

Mark hydro as FINISHED so that GetHydroInfo() works.

#### `set_hydro_status_initialized() -> None`

Mark hydro as INITIALIZED.

#### `set_preserve_bulk_info(preserve: bool) -> None`

When True, Clear() will not wipe bulk_info.data. Lets Python read bulk_info after js.Exec() returns.

#### `store_fluid_cells_aos_3d(arr: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], names: collections.abc.Sequence[str]) -> None`

Same input as store_fluid_cells_from_numpy_3d(), but fills bulk_info.data (array of FluidCellInfo) instead of data_vector.

#### `store_fluid_cells_from_numpy(arr: typing.Annotated[numpy.typing.ArrayLike, numpy.float32]) -> None`

Populate bulk_info.data from a numpy array.

Parameters:

- **`arr`** (`np.ndarray, shape (n_features, nx, ny, ntau), dtype float32`) – Feature layout (same as FnoHydro C++ convention): n_features == 4: [energy_density, temperature, vx, vy] n_features == 3: [energy_density, vx, vy] Energy density must already be in physical units [GeV/fm^3] (apply inverse tau-normalisation in Python before calling).

Notes

- Call set_hydro_grid_info() before this method.
- The array dimensions (nx, ny, ntau) must match the grid info.

#### `store_fluid_cells_from_numpy_3d(arr: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], names: collections.abc.Sequence[str]) -> None`

Fill bulk_info.data_vector from a (n_features, nx, ny, neta, ntau) float32 array.

`names` must be as long as n_features and each entry must be one that ResolveEntryName() accepts: energy_density, entropy_density, temperature, pressure, qgp_fraction, mu_b, mu_c, mu_s, vx, vy, vz, pi00..pi33, bulk_pi.

Cells are written in EvolutionHistory record order (tau, x, y, eta). data_vector, data_info and data are all cleared first -- clear_up_evolution_data() clears only `data`.

Call set_hydro_grid_info() first; the grid is checked against it.

This is the memory-efficient route: 4 bytes per stored field per cell, against 112 bytes per cell for store_fluid_cells_aos_3d().

#### `store_surface_from_numpy(cells: typing.Annotated[numpy.typing.ArrayLike, numpy.float32]) -> None`

Append cells from an (N, 32) array (columns SURFACE_CELL_COLUMNS) to the surface vector, e.g. to replay a stored surface into iSS.

#### `surface_to_numpy() -> numpy.typing.NDArray[numpy.float32]`

The freeze-out surface as an (N, 32) float32 array, one row per cell, columns SURFACE_CELL_COLUMNS: exactly the fields iSS reads (iSpectraSamplerWrapper::getSurfCellVector), in its order. SurfaceCellInfo is float, so this copy is lossless. Valid after the hydro's Exec, before ClearPerEvent.

### `FourVector`

A four-vector. The constructor takes (x, y, z, t) as C++ FourVector(x_in, y_in, z_in, t_in); momenta are (px, py, pz, E).

#### `Set(x: typing.SupportsFloat | typing.SupportsIndex, y: typing.SupportsFloat | typing.SupportsIndex, z: typing.SupportsFloat | typing.SupportsIndex, t: typing.SupportsFloat | typing.SupportsIndex) -> None`

#### `boost(vx: typing.SupportsFloat | typing.SupportsIndex, vy: typing.SupportsFloat | typing.SupportsIndex, vz: typing.SupportsFloat | typing.SupportsIndex) -> None`

#### `clear() -> None`

#### `comp(i: typing.SupportsInt | typing.SupportsIndex) -> float`

Component i: 0 = t, 1 = x, 2 = y, 3 = z.

#### `eta() -> float`

#### `eta_boost(deta: typing.SupportsFloat | typing.SupportsIndex) -> None`

#### `minus() -> float`

#### `momentum(E: float, px: float, py: float, pz: float) -> 'FourVector'`

FourVector.momentum(E, px, py, pz) (the constructor takes px, py, pz, E).

#### `phi() -> float`

#### `plus() -> float`

#### `position(t: float, x: float, y: float, z: float) -> 'FourVector'`

FourVector.position(t, x, y, z) (the constructor takes x, y, z, t).

#### `rapidity() -> float`

#### `rotate_around_x(theta: typing.SupportsFloat | typing.SupportsIndex) -> None`

#### `rotate_around_y(theta: typing.SupportsFloat | typing.SupportsIndex) -> None`

#### `rotate_around_z(theta: typing.SupportsFloat | typing.SupportsIndex) -> None`

#### `t() -> float`

#### `to_list() -> list[float]`

[x, y, z, t]

#### `x() -> float`

#### `y() -> float`

#### `y_boost(dy: typing.SupportsFloat | typing.SupportsIndex) -> None`

#### `z() -> float`

### `Hadron`

Bases: `JetScapeParticleBase`

A hadron: Hadron(label, id, stat, p, x\[, mass\[, charge, baryon_number, strangeness[, participant]\]\]) with p = FourVector(px, py, pz, E) [GeV] and x = FourVector(x, y, z, t) [fm]. Without a mass the PDG mass is used.

#### `baryon_number() -> int`

#### `charge() -> int`

#### `copy() -> Hadron`

A copy.

#### `decay_width() -> float`

#### `has_no_position() -> bool`

#### `has_valid_momentum() -> bool`

#### `participant() -> bool`

#### `set_baryon_number(baryon_number: typing.SupportsInt | typing.SupportsIndex) -> None`

#### `set_charge(charge: typing.SupportsInt | typing.SupportsIndex) -> None`

#### `set_decay_width(width: typing.SupportsFloat | typing.SupportsIndex) -> None`

#### `set_participant(participant: bool) -> None`

#### `set_strangeness(strangeness: typing.SupportsInt | typing.SupportsIndex) -> None`

#### `strangeness() -> int`

### `HadronArray(pid, pstat, p, x, mass, sample_offsets=None, info=None)`

One event's hadrons, flat per field, with sample offsets. numpy only.

#### `info = dict(info or {})`

#### `mass = None if mass is None else np.asarray(mass)`

#### `n_samples: int`

#### `p = np.asarray(p).reshape(-1, 4)`

#### `pid = np.asarray(pid)`

#### `pstat = np.asarray(pstat)`

#### `sample_counts: np.ndarray`

Hadrons per sample, (n_samples,) int64.

#### `sample_offsets = np.asarray(sample_offsets, dtype=np.int64)`

#### `x = None if x is None else np.asarray(x).reshape(-1, 4)`

#### `from_dict(d) -> 'HadronArray'`

From a bindings hadron dict; the arrays are taken as they are (no copy). `sample_counts` (one per oversample) becomes `sample_offsets`; without it the hadrons are one sample. Other keys go to `info`.

#### `from_h5(path, units=None) -> 'HadronArray'`

A hadron file (hadrons_h5 format: jet_frag, bulk_bg, ...) as one HadronArray: the stored fields as they are (float32, no derived columns), every sample of every unit in file order. `units`: an index array or slice to read only those units.

info: `unit_offsets` (samples of unit u: unit_offsets\[u\]:[u+1]), `units` (the file's per-unit table: event, seed, event_weight, ...), `attrs` (the file's attributes), `path`. The mass is not stored (None).

#### `to_awkward(position=False, charge=False, by_unit=False)`

samples x hadrons as an awkward array of Momentum4D records over these buffers (awkward_views.to_awkward); by_unit (a HadronArray from_h5): units x samples x hadrons.

#### `to_dict() -> dict`

The bindings' hadron dict (pid, pstat, p, x, mass, sample_counts), the arrays themselves (no copy): what HadronH5Writer.append_unit takes for one unit.

#### `to_hadrons(unit_offsets=None, attrs=None)`

-> hadrons_h5.Hadrons (the file's units after from_h5, else all samples as one unit, unless `unit_offsets` says otherwise), for its hist/total. Hadrons computes its derived columns in float64: about 2.4x the memory of this array.

### `Hadronization()`

Bases: `JetScapeModuleBase`

A jet hadronization task: the container a HadronizationManager runs (Add() the modules to it), and the base of every jet hadronization module, C++ (ColorlessHadronization, iColoredHadronization, ...) or Python.

A Python module subclasses it and overrides

```text
def DoHadronization(self, shower, hOut, pOut): ...
```

shower: one list of Partons per shower (the C++ objects); hOut: append the Hadrons; pOut: append the partons not hadronized. GetHydroHyperSurface(T) and GetHydroCell(t, x, y, z) give the medium. register_module( "CustomModuleX", cls) and CustomModuleX make the XML use it.

#### `last_random_seed: int`

ColorlessHadronization's (or iColoredHadronization's) last seed.

#### `DeleteHadrons() -> None`

#### `DeleteRealHadrons() -> None`

#### `DoHadronization(shower: list, hOut: list, pOut: list) -> None`

Hadronize: shower (list of lists of Parton), hOut and pOut (lists, filled with the hadrons and the partons left over).

#### `GetHadrons() -> list`

The hadrons this task made in the event.

#### `GetHydroCell(t: typing.SupportsFloat | typing.SupportsIndex, x: typing.SupportsFloat | typing.SupportsIndex, y: typing.SupportsFloat | typing.SupportsIndex, z: typing.SupportsFloat | typing.SupportsIndex) -> typing.Any`

#### `GetHydroHyperSurface(T: typing.SupportsFloat | typing.SupportsIndex) -> list`

The hydro's constant-temperature surface at T [GeV] as SurfaceCellInfos (empty without a hydro).

#### `GetOutPartons() -> list`

The partons this task left over in the event.

#### `colour_audit() -> dict`

iColoredHadronization: the last event's colour audit (a dict).

#### `colour_totals() -> dict`

iColoredHadronization: the audits added up over the run.

#### `hadron_array() -> HadronArray`

The hadrons this module (or every module of the manager) made this event (hadronization_hadrons_numpy) as a HadronArray (one sample).

#### `hadronize(partons, seed: int | None = None) -> HadronArray`

Hadronize stored final partons: FINAL_PARTON_DTYPE records (final_parton_array()) or the old (N, 14) array (hadronize_partons). -> HadronArray; info holds the seed used and n_partons_out.

#### `set_colour_closure(mode: str) -> None`

iColoredHadronization: by hand (reconnect, reconnect_global, thermal, fake, none).

### `HadronizationManager()`

Bases: `JetScapeModuleBase`

The jet hadronization of an event: it takes the energy-loss manager's final partons and runs its Hadronization tasks (Add() a Hadronization with the modules in it).

#### `last_random_seed: int`

ColorlessHadronization's (or iColoredHadronization's) last seed.

#### `DeleteHadrons() -> None`

#### `DeleteRealHadrons() -> None`

#### `GetHadrons() -> list`

The hadrons of all its tasks in this event.

#### `colour_audit() -> dict`

iColoredHadronization: the last event's colour audit (a dict).

#### `colour_totals() -> dict`

iColoredHadronization: the audits added up over the run.

#### `hadron_array() -> HadronArray`

The hadrons this module (or every module of the manager) made this event (hadronization_hadrons_numpy) as a HadronArray (one sample).

#### `hadronize(partons, seed: int | None = None) -> HadronArray`

Hadronize stored final partons: FINAL_PARTON_DTYPE records (final_parton_array()) or the old (N, 14) array (hadronize_partons). -> HadronArray; info holds the seed used and n_partons_out.

#### `set_colour_closure(mode: str) -> None`

iColoredHadronization: by hand (reconnect, reconnect_global, thermal, fake, none).

### `HardProcess()`

Bases: `JetScapeModuleBase`

The hard process of an event (PythiaGun, PGun, ... in C++).

A Python gun subclasses it and fills the event in ExecuteTask() with AddParton(Parton(...)) (and AddHadron() for hadrons from the hard process); GetSigmaGen, GetSigmaErr, GetPtHat and GetEventWeight can be overridden for the writers. The hard vertex comes from the initial state: JetScapeSignalManager.Instance().GetInitialStatePointer() .sample_binary_collision_point(). register_module("CustomModuleX", cls) makes it available to the XML ().

#### `info: dict`

hard_process_info(): cross section, weight, pTHat, ... of this event.

#### `AddHadron(hadron: Hadron) -> None`

#### `AddParton(parton: Parton) -> None`

#### `AddPartonShower(shower: PartonShower) -> None`

#### `GetEventWeight() -> float`

#### `GetHadronList() -> list`

#### `GetMax_Color() -> float`

#### `GetMax_ColorPerShower() -> float`

#### `GetNHadrons() -> int`

#### `GetNHardPartons() -> int`

#### `GetNPartonShowers() -> int`

#### `GetPartonAt(i: typing.SupportsInt | typing.SupportsIndex) -> Parton`

#### `GetPartonList() -> list`

The hard partons of the event (the objects, not copies).

#### `GetPartonShowerAt(i: typing.SupportsInt | typing.SupportsIndex) -> PartonShower`

#### `GetPtHat() -> float`

#### `GetRemnants() -> list[Parton]`

The beam remnants (copies), with ISR.

#### `GetSigmaErr() -> float`

#### `GetSigmaGen() -> float`

#### `PushRemnants(parton: Parton) -> None`

#### `SetMax_Color(col: typing.SupportsFloat | typing.SupportsIndex) -> None`

#### `parton_array() -> np.ndarray`

The hard partons as HARD_PARTON_DTYPE records (pid, E px py pz, pt, y), hardest first (hard_partons_numpy).

### `HydroStatus(value: typing.SupportsInt | typing.SupportsIndex)`

Members:

NOT_START

INITIALIZED

EVOLVING

FINISHED

ERROR

#### `ERROR: HydroStatus`

#### `EVOLVING: HydroStatus`

#### `FINISHED: HydroStatus`

#### `INITIALIZED: HydroStatus`

#### `NOT_START: HydroStatus`

#### `name: str`

#### `value: int`

### `InitialState()`

Bases: `JetScapeModuleBase`

Interface for initial-state physics modules (e.g. TrentoInitial).

Provides the entropy/binary-collision density grids that seed pre-equilibrium and hydro evolution.

Python subclasses can override Init(), Exec(), and Clear() and populate the IC grid with set_entropy_density_from_numpy() instead of running an external binary.

#### `info: dict`

initial_state_info(): b, Npart, Ncoll, centrality, ... of this event.

#### `GetEntropyDensityDistribution() -> list[float]`

Return the entropy density distribution as a flat list. Storage order: for z { for y { for x } }. Length = GetXSize() * GetYSize() * GetZSize().

#### `GetEventCentrality() -> float`

Event centrality class (-1 if not available).

#### `GetEventId() -> int`

Current event ID.

#### `GetNcoll() -> float`

Number of binary collisions (-1 if not available).

#### `GetNpart() -> float`

Number of participant nucleons (-1 if not available).

#### `GetNumOfBinaryCollisions() -> list[float]`

Return the un-normalised T_A\*T_B binary-collision density as a flat list. Same storage order as GetEntropyDensityDistribution().

#### `GetTotalEntropy() -> float`

Total entropy of the event (-1 if not available).

#### `GetXMax() -> float`

Maximum |x| of the nuclear profile grid [fm].

#### `GetXSize() -> int`

Number of grid points in x: ceil(2\*xmax/dx).

#### `GetXStep() -> float`

Step size dx [fm].

#### `GetYMax() -> float`

Maximum |y| of the nuclear profile grid [fm].

#### `GetYSize() -> int`

Number of grid points in y: ceil(2\*ymax/dy).

#### `GetYStep() -> float`

Step size dy [fm].

#### `GetZMax() -> float`

Maximum |z or eta| of the nuclear profile grid.

#### `GetZSize() -> int`

Number of grid points in z (1 for boost-invariant).

#### `GetZStep() -> float`

Step size dz (or d-eta) in z direction.

#### `SampleABinaryCollisionPoint() -> tuple[float, float, float, float]`

The C++ name of sample_binary_collision_point(): (t, x, y, z) [fm]. A Python initial state overrides it to return its own hard vertices.

#### `SetEventId(event_id: typing.SupportsInt | typing.SupportsIndex) -> None`

#### `SetRanges(xmax: typing.SupportsFloat | typing.SupportsIndex, ymax: typing.SupportsFloat | typing.SupportsIndex, zmax: typing.SupportsFloat | typing.SupportsIndex) -> None`

Set grid extents (xmax, ymax, zmax).

#### `SetSteps(dx: typing.SupportsFloat | typing.SupportsIndex, dy: typing.SupportsFloat | typing.SupportsIndex, dz: typing.SupportsFloat | typing.SupportsIndex) -> None`

Set grid step sizes (dx, dy, dz).

#### `get_entropy_density_numpy() -> numpy.typing.NDArray[numpy.float64]`

Return the entropy density as a 2D numpy array of shape (ny, nx). Only valid for the boost-invariant (2D) case (GetZSize() == 1). Equivalent to: dist = GetEntropyDensityDistribution() arr = np.array(dist).reshape(ny, nx)

#### `get_entropy_density_numpy_3d() -> numpy.typing.NDArray[numpy.float64]`

The initial-state grid as a 3D numpy array of shape (nx, ny, neta).

Storage order is the C-order (nx, ny, neta) ravel, idx = (ny\*neta)\*ix + neta\*iy + ieta which is what set_entropy_density_from_numpy() copies in verbatim and what MUSIC's Initial_profile 42 reads.

NOTE: get_entropy_density_numpy() (the 2D convenience view) instead assumes dist[iy\*nx + ix] and is documented as valid only for GetZSize() == 1. For a square boost-invariant grid the two differ by a transpose, so do not mix them.

#### `sample_binary_collision_point() -> tuple[float, float, float, float]`

Draw one hard-scattering vertex, exactly as the hard process does.

Returns (t, x, y, z) in fm. With no density set this returns the origin (and the framework prints a warning), which is the whole reason set_num_of_binary_collisions_from_numpy() exists.

Coordinates come from CoordFromIdx, i.e. MUSIC's `-grid_max + i*step` axis.

#### `set_entropy_density_from_numpy(arr: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None`

Populate entropy_density_distribution\_ from a numpy array.

Call this from a Python InitialState subclass inside Exec() to inject a custom initial condition (e.g. a Gaussian profile) without running an external program such as trento.

Parameters:

- **`arr`** (`np.ndarray, shape (ny, nx) or (ny*nx,), dtype float64`) – Entropy (or energy) density values in C-contiguous order: boost-invariant storage is for y { for x }, so arr[iy, ix] maps to index iy * nx + ix.

#### `set_num_of_binary_collisions_from_numpy(arr: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None`

Set the Ta\*Tb binary-collision density that hard processes sample vertices from.

Same C-order (nx, ny, neta) ravel as set_entropy_density_from_numpy(). Without it, InitialState::SampleABinaryCollisionPoint() warns and puts every jet at the origin, so all showers start at the fireball centre.

NOTE the half-cell offset: CoordFromIdx() maps index ix to `-grid_max_x + ix*grid_step_x` (InitialState.cc:75-77), i.e. MUSIC's `-n*d/2 + i*d` axis, while the energy-density grid this contribution uses is cell-centred at `-(n-1)*d/2 + i*d`. Sampled vertices therefore sit half a cell from the matching energy-density node.

### `IsrJet()`

Bases: `JetEnergyLoss`

The per-parton task of the IsrManager (an initial-state shower).

### `IsrManager()`

Bases: `JetEnergyLossManager`

The initial-state radiation of the hard partons: an energy-loss manager run backward in time under PythiaIsrGun. As the XML builds it for :

```text
isr = IsrManager(); jet = IsrJet(); imatter = iMATTER()
jet.SetDeltaT(-dt); jet.SetStartT(0); jet.SetMaxT(-maxT); imatter.SetMaxT(-maxT)
jet.AddPartonShowerGenerator(IsrShowerPSG()); jet.Add(imatter)
isr.Add(jet); gun = create_module("PythiaIsrGun"); gun.Add(isr)
gun.Add(create_module("MCGlauberGenString"))
```

### `IsrShowerPSG()`

Bases: `PartonShowerGeneratorDefault`

The initial-state (backward) shower generator of IsrJet.

### `JetEnergyLoss()`

Bases: `JetScapeModuleBase`

An energy-loss task: the per-shower container the manager copies (Add() the algorithm modules to it, Matter before LBT), and the base of every energy-loss module, C++ (Matter, LBT, ...) or Python.

A Python module subclasses it and overrides

```text
def DoEnergyLoss(self, deltaT, time, Q2, pIn, pOut): ...
```

pIn: the parton(s) of this step, the C++ objects (change them in place; valid during the call only). pOut: append the partons at time + deltaT (left empty, the parton goes on unchanged). GetHydroCell(t, x, y, z) gives the medium. The manager copies the module once per shower with Clone(): by default the C++ copy constructor plus a shallow copy of the instance attributes; override Clone() for anything deeper.

register_module("CustomModuleX", cls) makes it available to the XML (...).

#### `AddPartonShowerGenerator(generator: typing.Any) -> None`

Set how this task grows its shower (PartonShowerGeneratorDefault unless set).

#### `AddShowerInitiatingParton(parton: Parton) -> None`

Set the parton this task showers (the manager does this per shower).

#### `Clone() -> JetEnergyLoss`

A copy for one shower (C++ modules: their copy constructor).

#### `DoEnergyLoss(deltaT: typing.SupportsFloat | typing.SupportsIndex, time: typing.SupportsFloat | typing.SupportsIndex, Q2: typing.SupportsFloat | typing.SupportsIndex, pIn: list, pOut: list) -> None`

One step: pIn (list of Parton, changed in place), pOut (list, filled with the partons at time + deltaT).

#### `GetDeltaT() -> float`

#### `GetGetHydroCellSignalConnected() -> bool`

#### `GetHydroCell(t: typing.SupportsFloat | typing.SupportsIndex, x: typing.SupportsFloat | typing.SupportsIndex, y: typing.SupportsFloat | typing.SupportsIndex, z: typing.SupportsFloat | typing.SupportsIndex) -> typing.Any`

The medium at (t, x, y, z) [fm] as a FluidCellInfo, or None without a hydro (the GetHydroCell signal Matter and LBT use).

#### `GetMaxT() -> float`

#### `GetPartonShowerGenerator() -> typing.Any`

#### `GetQhat() -> float`

#### `GetSentInPartonsConnected() -> bool`

#### `GetShower() -> PartonShower`

The finished PartonShower for this task (valid after Exec).

#### `GetShowerInitiatingParton() -> Parton`

The parton this shower started from (the per-shower copies).

#### `GetStartT() -> float`

#### `GetUseIntialPartonShower() -> bool`

#### `SetDeltaT(deltaT: typing.SupportsFloat | typing.SupportsIndex) -> None`

#### `SetMaxT(maxT: typing.SupportsFloat | typing.SupportsIndex) -> None`

#### `SetQhat(qhat: typing.SupportsFloat | typing.SupportsIndex) -> None`

#### `SetStartT(startT: typing.SupportsFloat | typing.SupportsIndex) -> None`

#### `SetUseIntialPartonShower(use: bool) -> None`

Continue a given shower (ISR) instead of starting from one parton.

#### `add_a_liquefier(liquefier: typing.Any) -> None`

Attach the liquefier that collects droplets from this shower. The framework calls liquefier->add_hydro_sources() centrally in DoExecTime(), so Matter and LBT need no changes.

#### `get_liquefier() -> LiquefierBase`

The attached liquefier, or None.

### `JetEnergyLossManager()`

Bases: `JetScapeModuleBase`

Energy-loss manager; one JetEnergyLoss child per shower-initiating parton.

#### `showers: 'list[PartonShower]'`

The PartonShowers of this event, one per shower (get_showers()).

#### `final_parton_array() -> np.ndarray`

The final partons as FINAL_PARTON_DTYPE records (final_partons_numpy). After the manager's Exec, before ClearPerEvent.

#### `final_partons_numpy() -> numpy.typing.NDArray[numpy.float64]`

The final-state partons the hadronization would receive, as an (N, 14) float64 array with columns FINAL_PARTON_COLUMNS (shower index = position in the manager's task list). Valid after the manager's Exec, before ClearPerEvent.

#### `get_shower_initiating_partons() -> list[Parton]`

The hard (shower-initiating) partons, one per shower.

#### `get_showers() -> list[PartonShower]`

List of per-event PartonShowers (one per shower-initiating parton). Valid after the manager's Exec, before ClearPerEvent.

#### `initiator_array() -> np.ndarray`

The shower-initiating partons as INITIATOR_DTYPE records (showers.initiators_from_manager).

#### `number_of_showers() -> int`

### `JetScape()`

Bases: `JetScapeModuleBase`

#### `Exec() -> None`

Run all events.

#### `Finish() -> None`

Finalise output and clean up.

#### `GetNReuseHydro() -> int`

Return the hydro-reuse count.

#### `GetNumberOfEvents() -> int`

Return the configured number of events.

#### `GetReuseHydro() -> bool`

Return whether hydro reuse is enabled.

#### `Init() -> None`

Initialise the full task list (reads XML, wires signals). Python modules registered with register_module() get an empty default entry in the main XML first, so the user XML may name them.

#### `SetNReuseHydro(n: typing.SupportsInt | typing.SupportsIndex) -> None`

Set how many jet events reuse each hydro event.

#### `SetNumberOfEvents(n_events: typing.SupportsInt | typing.SupportsIndex) -> None`

Override the number of events from Python.

#### `SetReuseHydro(reuse: bool) -> None`

Enable or disable hydro reuse across jet events.

#### `SetXMLMainFileName(filename: str) -> None`

#### `SetXMLUserFileName(filename: str) -> None`

### `JetScapeEventHeader`

What the tasks report for an event (CollectHeader): weights, vertex, centrality.

#### `GetEventCentrality() -> float`

#### `GetEventPlaneAngle() -> float`

#### `GetEventWeight() -> float`

#### `GetNcoll() -> float`

#### `GetNpart() -> float`

#### `GetPtHat() -> float`

#### `GetSigmaErr() -> float`

#### `GetSigmaGen() -> float`

#### `GetTotalEntropy() -> float`

#### `GetVertexX() -> float`

#### `GetVertexY() -> float`

#### `GetVertexZ() -> float`

#### `SetEventCentrality(value: typing.SupportsFloat | typing.SupportsIndex) -> None`

#### `SetEventPlaneAngle(value: typing.SupportsFloat | typing.SupportsIndex) -> None`

#### `SetEventWeight(value: typing.SupportsFloat | typing.SupportsIndex) -> None`

#### `SetNcoll(value: typing.SupportsFloat | typing.SupportsIndex) -> None`

#### `SetNpart(value: typing.SupportsFloat | typing.SupportsIndex) -> None`

#### `SetPtHat(value: typing.SupportsFloat | typing.SupportsIndex) -> None`

#### `SetSigmaErr(value: typing.SupportsFloat | typing.SupportsIndex) -> None`

#### `SetSigmaGen(value: typing.SupportsFloat | typing.SupportsIndex) -> None`

#### `SetTotalEntropy(value: typing.SupportsFloat | typing.SupportsIndex) -> None`

#### `SetVertexX(value: typing.SupportsFloat | typing.SupportsIndex) -> None`

#### `SetVertexY(value: typing.SupportsFloat | typing.SupportsIndex) -> None`

#### `SetVertexZ(value: typing.SupportsFloat | typing.SupportsIndex) -> None`

### `JetScapeModuleBase()`

Bases: `JetScapeTask`

Base class for all JETSCAPE physics modules.

Subclass this in Python to create a custom module that can be inserted into a JETSCAPE pipeline via JetScape.Add():

```text
class MyModule(pyjetscape.JetScapeModuleBase):
    def Init(self):
        self.SetId("MyModule")
    def Exec(self):
        sm    = pyjetscape.JetScapeSignalManager.Instance()
        hydro = sm.GetHydroPointer()
        ...
    def Finish(self):
        ...  # called by JetScape.Finish()
```

#### `rng: np.random.Generator`

A numpy Generator seeded from this task's own engine (GetMt19937Generator), made on first use. Cached with the task number: a Clone() (which copies **dict**) gets a new task number and so its own stream. An instance attribute `rng` overrides it. Python modules only: on a C++ module (no **dict**) it raises AttributeError rather than draw from that module's engine.

#### `CalculateTime() -> None`

Time-step loop: before ExecTime().

#### `Clear() -> None`

Clear per-event state (called between events).

#### `ClearTask() -> None`

Per-event clean-up (Clear() calls it).

#### `Exec() -> None`

Execute this module for one event.

#### `ExecTime() -> None`

Time-step loop: the work of one time step.

#### `ExecuteTask() -> None`

The task's work for one event (Exec() calls it).

#### `Finish() -> None`

Finalise the module (called by JetScape::Finish()).

#### `FinishPerEvent() -> None`

End of an event in the time-step loop.

#### `FinishTask() -> None`

End-of-run work (Finish() calls it).

#### `GetCurrentEvent() -> int`

Return the framework-global current event number (advanced by ClearPerEvent() / per-event execution).

#### `GetMt19937Generator() -> Mt19937`

This task's random engine (made on first use; needs the seed from JetScape.Init() or load_xml()).

#### `GetXMLMainFileName() -> str`

Return the main XML configuration file path.

#### `GetXMLUserFileName() -> str`

Return the user XML configuration file path.

#### `IncrementCurrentEvent() -> None`

Increment the framework-global current event number.

#### `Init() -> None`

Initialise this module (reads XML, sets up sub-tasks).

#### `InitPerEvent() -> None`

Start of an event in the time-step loop.

#### `InitTask() -> None`

Per-task initialisation (Init() calls it once).

#### `SetXMLMainFileName(filename: str) -> None`

Set the main (default) XML configuration file path.

#### `SetXMLUserFileName(filename: str) -> None`

Set the user (override) XML configuration file path.

#### `get_xml_element_double(path: collections.abc.Sequence[str], required: bool = True) -> float`

Read a double value from the loaded JETSCAPE XML.

Searches the user XML first, then the main XML.

Parameters:

- **`path`** (`list[str]`) – Sequence of tag names, e.g. ["Hydro", "FNO", "dtau"].
- **`required`** (`bool`, default: `True` ) – If True, raises RuntimeError when the element is not found.

Returns:

- `float` – Parsed double, or 0.0 when not found and not required.

#### `get_xml_element_int(path: collections.abc.Sequence[str], required: bool = True) -> int`

Read an integer value from the loaded JETSCAPE XML.

Searches the user XML first, then the main XML.

Parameters:

- **`path`** (`list[str]`) – Sequence of tag names, e.g. ["Hydro", "FNO", "nx"].
- **`required`** (`bool`, default: `True` ) – If True, raises RuntimeError when the element is not found.

Returns:

- `int` – Parsed integer, or 0 when not found and not required.

#### `get_xml_element_text(path: collections.abc.Sequence[str], required: bool = True) -> str`

Read a text value from the loaded JETSCAPE XML.

Searches the user XML first, then the main XML (same priority as the C++ JetScapeModuleBase::GetXMLElementText helper).

Parameters:

- **`path`** (`list[str]`) – Sequence of tag names, e.g. ["Hydro", "FNO", "model_file"].
- **`required`** (`bool`, default: `True` ) – If True, raises RuntimeError when the element is not found.

Returns:

- `str` – Element text content, or "" when not found and not required.

### `JetScapeParticleBase`

Common base of Parton and Hadron: identity, momentum, position.

#### `momentum: 'FourVector'`

The momentum as a FourVector (a copy); iterates as (E, px, py, pz).

#### `position: 'FourVector'`

The position as a FourVector (a copy); iterates as (t, x, y, z).

#### `GetControlled() -> bool`

#### `GetController() -> str`

#### `SetController(controller: str = '') -> bool`

#### `UnsetController() -> None`

#### `e() -> float`

#### `eta() -> float`

#### `init_jet_v() -> None`

#### `jet_v() -> FourVector`

#### `nu() -> float`

#### `p(i: typing.SupportsInt | typing.SupportsIndex) -> float`

Momentum component i: 0 = E, 1 = px, 2 = py, 3 = pz.

#### `p_in() -> FourVector`

Momentum as FourVector(px, py, pz, E).

#### `phi() -> float`

Azimuth in \[0, 2 pi) (fastjet's phi).

#### `pid() -> int`

#### `pl() -> float`

#### `plabel() -> int`

#### `pstat() -> int`

#### `pt() -> float`

#### `px() -> float`

#### `py() -> float`

#### `pz() -> float`

#### `rap() -> float`

#### `rapidity() -> float`

#### `restmass() -> float`

Rest mass [GeV] (the PDG mass the particle was built with).

#### `set_id(id: typing.SupportsInt | typing.SupportsIndex) -> None`

#### `set_jet_v(v: FourVector) -> None`

#### `set_label(label: typing.SupportsInt | typing.SupportsIndex) -> None`

#### `set_stat(stat: typing.SupportsInt | typing.SupportsIndex) -> None`

#### `set_x(x: FourVector) -> None`

Set the position from a FourVector(x, y, z, t).

#### `t() -> float`

Position time x_in().t() [fm/c]. For a Parton the virtuality is virtuality() (C++ Parton::t()).

#### `time() -> float`

Position time x_in().t().

#### `x() -> float`

#### `x_in() -> FourVector`

Position as FourVector(x, y, z, t).

#### `y() -> float`

#### `z() -> float`

### `JetScapePerEvent()`

Bases: `JetScape`

JetScape driver that runs the event loop one event at a time.

Unlike JetScape (whose Exec() owns the whole loop and clears each event's memory before the next), JetScapePerEvent lets the external Python loop run a single event, inspect the live module results, and then explicitly release the memory:

```text
js = JetScapePerEvent()
js.SetXMLMainFileName(main_xml)
js.SetXMLUserFileName(user_xml)
js.Init()
for _ in range(js.GetNumberOfEvents()):
    js.ExecPerEvent()
    # ... read module data for the current event here ...
    js.ClearPerEvent()
js.Finish()
```

SetStartEvent(n) shifts the (framework-global) event counter so the first event runs as event n (default 0). Exec() is disabled on this class and will warn-and-exit if called.

#### `energy_loss_manager: 'JetEnergyLossManager | None'`

The final-state JetEnergyLossManager (the signal manager's, else the task list's; never the ISR chain's IsrManager inside the hard process), or None.

#### `hadronization_manager: 'HadronizationManager | None'`

The HadronizationManager of the task list, or None.

#### `hydro: 'FluidDynamics | None'`

The hydro the signal manager holds (else the first in the task list), or None.

#### `initial_state: 'InitialState | None'`

The initial state the signal manager holds (else the task list's), or None.

#### `soft_particlization: 'SoftParticlization | None'`

The (first) SoftParticlization module of the task list (e.g. iSS), or None.

#### `ClearPerEvent() -> None`

Release the per-event module memory and advance the event counter. Call after reading the data for the current event.

#### `ExecInit() -> None`

One-time pre-loop setup (collect writers, CheckExec, snapshot active flags). Idempotent; ExecPerEvent() also calls it lazily.

#### `ExecPerEvent() -> None`

Run a single event, leaving all module data in memory for inspection. The event number is taken from the global counter.

#### `GetStartEvent() -> int`

Return the configured starting event number.

#### `SetStartEvent(start_event: typing.SupportsInt | typing.SupportsIndex) -> None`

Set the starting event number (default 0). Must be called before the first ExecPerEvent().

#### `events(n: int | None = None, on_error: 'Callable[[int, Exception], None] | None' = None) -> 'Iterator[int]'`

Yield the event index once per event, after ExecPerEvent() and before ClearPerEvent(), so the event's results can be read; n: at most n events (default GetNumberOfEvents()). Stops cleanly on break (the event is cleared); Finish() is the context manager's.

```text
on_error: when a module throws inside an event (a C++ exception, e.g. Matter's "minus
log of sudakov negative"; the same seed hits it again on a rerun), call
on_error(i, exc), clear the event and go on with the next one, as the productions
skip such events.  Without it the exception propagates.
```

### `JetScapeReaderAscii(file_name: str)`

Read a JetScapeWriterAscii file: while not r.Finished(): r.Next(); r.GetPartonShowers(), r.GetHadrons(), r.GetSigmaGen(), ...

#### `Close() -> None`

#### `Finished() -> bool`

True at the end of the file.

#### `GetCurrentEvent() -> int`

#### `GetCurrentNumberOfPartonShowers() -> int`

#### `GetEventPlaneAngle() -> float`

#### `GetEventWeight() -> float`

#### `GetHadrons() -> list`

#### `GetPartonShowers() -> list[PartonShower]`

#### `GetSigmaErr() -> float`

#### `GetSigmaGen() -> float`

#### `GetVertexX() -> float`

#### `GetVertexY() -> float`

#### `GetVertexZ() -> float`

#### `Next() -> None`

Read the next event.

### `JetScapeReaderAsciiGZ(file_name: str)`

As JetScapeReaderAscii, for .gz files.

#### `Close() -> None`

#### `Finished() -> bool`

True at the end of the file.

#### `GetCurrentEvent() -> int`

#### `GetCurrentNumberOfPartonShowers() -> int`

#### `GetEventPlaneAngle() -> float`

#### `GetEventWeight() -> float`

#### `GetHadrons() -> list`

#### `GetPartonShowers() -> list[PartonShower]`

#### `GetSigmaErr() -> float`

#### `GetSigmaGen() -> float`

#### `GetVertexX() -> float`

#### `GetVertexY() -> float`

#### `GetVertexZ() -> float`

#### `Next() -> None`

Read the next event.

### `JetScapeSignalManager`

Singleton signal/pointer manager for the JETSCAPE framework.

Provides global read access to all registered top-level physics modules so that any module (including Python modules added to the pipeline via JetScape.Add()) can look up its peers by type without holding an explicit reference.

The singleton is populated automatically by JetScape::Init() for all modules registered with JetScape.Add(). Python code that creates a custom module (e.g. PyBulkRootWriter) can therefore access any framework module from inside its Exec() method:

```text
from pyjetscape import JetScapeSignalManager
sm    = JetScapeSignalManager.Instance()
hydro = sm.GetHydroPointer()   # FluidDynamics or None
if hydro is not None:
    info = hydro.get_bulk_info()
```

Notes

- The object returned by Instance() is owned by the C++ singleton; do not store it across events or after Finish().
- All Get\*Pointer() methods lock the stored weak_ptr and return the shared_ptr. They return None if the module was never registered.

#### `GetHydroPointer() -> FluidDynamics`

Return the registered FluidDynamics module, or None if not set.

#### `GetInitialStatePointer() -> InitialState`

Return the registered InitialState module, or None if not set.

#### `GetJetEnergyLossManagerPointer() -> JetEnergyLossManager`

Return the registered JetEnergyLossManager (holds the per-event parton showers), or None if not set.

#### `GetNumberOfEdensitySignals() -> int`

Return the number of connected energy-density signals.

#### `GetNumberOfGetHydroCellSignals() -> int`

Return the number of connected GetHydroCell signals.

#### `GetNumberOfJetSignals() -> int`

Return the number of connected jet signals.

#### `GetPreEquilibriumPointer() -> PreequilibriumDynamics`

Return the registered PreequilibriumDynamics module, or None if not set.

#### `Instance() -> JetScapeSignalManager`

Return the singleton JetScapeSignalManager instance.

#### `SetHydroPointer(hydro: typing.Any) -> None`

Register a FluidDynamics module with the signal manager.

#### `SetInitialStatePointer(ini: typing.Any) -> None`

Register an InitialState module with the signal manager.

#### `SetPreEquilibriumPointer(preeq: typing.Any) -> None`

Register a PreequilibriumDynamics module with the signal manager.

### `JetScapeTask`

#### `Add(task: JetScapeTask) -> None`

Add a sub-module/task to this task's list.

#### `ClearTaskList() -> None`

Remove all sub-tasks.

#### `GetActive() -> bool`

Return whether this task is active.

#### `GetId() -> str`

Return the string identifier of this task.

#### `GetMyTaskNumber() -> int`

The task's number (fixes its random generator's seed).

#### `GetNumberOfTasks() -> int`

Return the number of sub-tasks currently registered.

#### `GetTaskAt(i: typing.SupportsInt | typing.SupportsIndex) -> typing.Any`

Return the sub-task at index i (as its most derived bound type).

#### `GetTaskList() -> list`

Return the list of sub-tasks as Python objects (each as its most derived bound type).

#### `SetActive(active: bool) -> None`

Enable or disable this task.

#### `SetId(id: str) -> None`

Set the string identifier for this task.

### `JetScapeWriter()`

Bases: `JetScapeModuleBase`

An output writer (ASCII, HepMC, ... in C++).

A Python writer subclasses it, implements GetStatus() (True when ready), and overrides Write(obj): it gets every Parton, Vertex, PartonShower, Hadron and str the tasks write in an event, after WriteHeaderToFile() (GetHeader() then holds the event header) and before WriteEvent(). Add it to the pipeline with JetScape.Add(writer).

#### `Close() -> None`

#### `GetHeader() -> JetScapeEventHeader`

#### `GetOutputFileName() -> str`

#### `GetStatus() -> bool`

#### `Open() -> None`

#### `SetOutputFileName(file_name: str) -> None`

#### `WriteComment(s: str) -> None`

#### `WriteEvent() -> None`

#### `WriteHeaderToFile() -> None`

#### `WriteWhiteSpace(s: str) -> None`

### `LiquefierBase()`

Turns what the jet loses to the medium (droplets) into a hydro source. A Python liquefier subclasses it and overrides smearing_kernel(tau, x, y, eta, droplet) -> [j^tau, j^x, j^y, j^eta] (and the deposition hooks as needed).

#### `COLUMNS: tuple = ('tau', 'x', 'y', 'eta', 'E', 'px', 'py', 'pz')`

Column order of droplets_numpy(): (tau, x, y, eta, E, px, py, pz).

#### `FLUID_COLUMNS: tuple = ('vx', 'vy', 'vz', 'e')`

Column order of droplet_fluid_numpy(): (vx, vy, vz, e).

#### `ClearTask() -> None`

Empty the droplet list. FluidDynamics::ClearTask() is normally what calls this; a Python hydro that overrides Clear() must call it itself, or droplets accumulate across events.

#### `add_a_droplet(droplet: Droplet) -> None`

#### `add_droplets_numpy(arr: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> int`

Append droplets from an (M, 8) array in the droplets_numpy() column order, or (M, 12) with the fluid's (vx, vy, vz, e) (droplet_fluid_numpy()) after them; with 8 columns the fluid is unknown (the fluid frame then uses the Bjorken frame). Used by the replay path.

#### `deposit_half_width() -> float`

#### `deposit_time(droplet: Droplet) -> float`

#### `droplet_fluid_numpy() -> numpy.typing.NDArray[numpy.float64]`

Per droplet (droplets_numpy() order), the background fluid where it was made, (M, 4) = (vx, vy, vz, e): lab Cartesian velocity and energy density [GeV/fm^3]; e < 0: not looked up.

#### `droplet_fluxes_numpy() -> numpy.typing.NDArray[numpy.float64]`

Per droplet (droplets_numpy() order): the sum of its point-sampled kernel on the hydro grid, i.e. the fraction of its four-momentum the hydro would receive without normalization; -1 if it was never normalized (no grid, switched off, or it never reached a hydro step). With normalization on, the hydro receives exactly the droplet (flux > 0, or flux 0: deposited into one cell).

#### `droplet_may_contribute(droplet: Droplet, tau_lo: typing.SupportsFloat | typing.SupportsIndex, tau_hi: typing.SupportsFloat | typing.SupportsIndex) -> bool`

#### `droplets_numpy() -> numpy.typing.NDArray[numpy.float64]`

The droplet list as an (M, 8) float64 array.

Columns are (tau, x, y, eta, E, px, py, pz) -- exactly fast_data.liquefier.droplets.COLUMNS, so the result is a DropletArray payload with no conversion.

float64 is emitted even though Jetscape::real is float: the values have already been rounded to float, and keeping them wide avoids a second rounding.

#### `get_GetHydroCellSignalConnected() -> bool`

True once the framework has wired this liquefier to a hydro. filter_partons() needs that signal to boost partons into the fluid rest frame.

#### `get_a_droplet(idx: typing.SupportsInt | typing.SupportsIndex) -> Droplet`

#### `get_drop_stat() -> int`

pstat of a parton absorbed into the fluid (-11).

#### `get_dropletlist_size() -> int`

#### `get_dropletlist_total_energy() -> float`

#### `get_miss_stat() -> int`

pstat of the energy-momentum-conservation filler parton (-13).

#### `get_neg_stat() -> int`

pstat of a negative / back-reaction parton (-17).

#### `get_normalize_on_hydro_grid() -> bool`

Whether each droplet is normalized on the hydro grid it is sampled on (, default on).

#### `get_source(tau: typing.SupportsFloat | typing.SupportsIndex, x: typing.SupportsFloat | typing.SupportsIndex, y: typing.SupportsFloat | typing.SupportsIndex, eta: typing.SupportsFloat | typing.SupportsIndex) -> typing.Annotated[list[float], 'FixedSize(4)']`

Summed source current j^mu [GeV/fm^4] at a Milne point, causality-gated. This is the reference the pure-Python kernel is checked against.

#### `normalizable_on_grid() -> bool`

#### `normalization_summary() -> str`

#### `query_time(tau: typing.SupportsFloat | typing.SupportsIndex, dtau: typing.SupportsFloat | typing.SupportsIndex, k: typing.SupportsInt | typing.SupportsIndex) -> float`

#### `set_hydro_step(dtau: typing.SupportsFloat | typing.SupportsIndex) -> None`

#### `set_late_hydro_step(dtau_late: typing.SupportsFloat | typing.SupportsIndex, tau_switch: typing.SupportsFloat | typing.SupportsIndex) -> None`

#### `set_normalize_on_hydro_grid(on: bool) -> None`

#### `smearing_kernel(tau: typing.SupportsFloat | typing.SupportsIndex, x: typing.SupportsFloat | typing.SupportsIndex, y: typing.SupportsFloat | typing.SupportsIndex, eta: typing.SupportsFloat | typing.SupportsIndex, droplet: Droplet) -> list[float]`

### `MpiMusic()`

Bases: `FluidDynamics`

MUSIC viscous relativistic hydrodynamics module.

Registered module name: `"MUSIC"` — instantiate via::

```text
hydro = create_module("MUSIC")
```

Important setup before calling Init(): * The JetScape XML must have `<Hydro><MUSIC>` settings with `output_evolution_to_memory: 1` for bulk_info to be populated. * Call `hydro.set_preserve_bulk_info(True)` so that bulk_info survives the ClearTasks() call at the end of each event.

After Exec() completes, `hydro.get_bulk_info()` returns an EvolutionHistory with all stored fluid cells.

#### `GetHydroFreezeOutTemperature() -> float`

Return the freeze-out temperature set in the XML [GeV].

#### `clear_hydro_info_from_memory() -> None`

Release MUSIC's native evolution store. FastRootBulkWriter does this after writing each event.

#### `get_checkpoint_tau() -> float`

tau [fm/c] by which this instance keeps a checkpoint of its state in every evolution (0: off).

#### `get_dump_hydro_only() -> bool`

Return whether is enabled (native store kept, bulk_info.data not built).

#### `get_freeze_out_surface() -> bool`

Whether MUSIC builds the freeze-out surface for this instance.

#### `get_hit_grid_boundary() -> bool`

True if the last evolution stopped because the freeze-out surface reached the transverse grid boundary (MUSIC's reRunHydro): the stored evolution is truncated. Reset at the start of every evolution.

#### `get_native_evolution_numpy(tau_stride: typing.SupportsInt | typing.SupportsIndex = 1) -> numpy.typing.NDArray[numpy.float32]`

Copy MUSIC's native evolution store into a numpy array in one C++ pass (GIL released), without building bulk_info.data.

The values and layout are exactly what FastRootBulkWriter writes in `native` mode, so `arr.ravel()` equals one `user_res` entry of its ROOT output.

Requires `<Hydro><MUSIC><dump_hydro_only>1`. If a FastRootBulkWriter is in the pipeline it releases the store at the end of its Exec(), so read the store before that runs.

Parameters:

- **`tau_stride`** (`int`, default: `1` ) – Keep every N-th stored tau step.

Returns:

- `np.ndarray, shape (ntau, nx, ny, neta, 4), dtype float32` – Features: [energy_density, vx, vy, vz]. Grid spacing and origin are in get_bulk_info() (dtau is multiplied by tau_stride).

#### `get_native_fluid_cell(idx: typing.SupportsInt | typing.SupportsIndex) -> FluidCellInfo`

Return one cell of MUSIC's native store as a FluidCellInfo.

`idx` is the flat tau-major (tau, x, y, eta) index, i.e. `((it*nx + ix)*ny + iy)*neta + ieta` with the grid sizes from `get_bulk_info()`.

#### `get_native_frame_numpy(it: typing.SupportsInt | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.float32]`

One tau step of MUSIC's native store, (nx, ny, neta, 4) float32.

#### `get_native_ntau() -> int`

Number of tau steps in MUSIC's native store (0 when empty).

#### `get_number_of_fluid_cells() -> int`

Return the number of cells in MUSIC's native evolution store (0 before InitializeHydro() or after the store was released).

#### `get_preserve_bulk_info() -> bool`

Return whether bulk_info preservation is enabled.

#### `get_restart_status() -> str`

What the last evolution did with set_restart_from ('' if not set).

#### `get_skip_surface() -> bool`

Return whether is enabled (freeze-out surface not exported to the framework).

#### `get_slim_bulk_info() -> bool`

Return whether is on: bulk_info gets a slim copy (e, s, T, vx, vy, vz) instead of full FluidCellInfo cells.

#### `get_store_evolution() -> bool`

Whether MUSIC stores this instance's evolution frames in memory.

#### `set_checkpoint_tau(tau: typing.SupportsFloat | typing.SupportsIndex) -> None`

Keep a checkpoint of this instance's state in every evolution, at the last freeze-out step whose RK step ends by tau [fm/c] (0: off), for another instance to start from (set_restart_from). music4gpu; CPU MUSIC ignores it.

#### `set_dump_hydro_only(dump: bool) -> None`

Keep this instance's native evolution store and skip building bulk_info.data. Call it AFTER JetScape.Init(): InitializeHydro() resets the flag from the XML. Needs output_evolution_to_memory = 1. Nothing may query this instance's medium (energy loss reads the first hydro only).

#### `set_freeze_out_surface(build: bool) -> None`

Build the freeze-out surface (True) or not (False) for this instance, from its next evolution on. Without a surface MUSIC stops on the equivalent max(e) < e_fo test at the same step (music4gpu; CPU MUSIC always builds it). The XML default is in the first block, overridden per instance in its own block.

#### `set_preserve_bulk_info(preserve: bool) -> None`

When True, bulk_info.data is retained after Clear() so that it can be inspected from Python after the event finishes. Set this BEFORE calling Init().

#### `set_restart_from(other: MpiMusic) -> None`

Start every evolution of this instance from `other`'s latest checkpoint when it fits (same parameters, initial arena and initial-state source, no droplet before its step), from tau0 otherwise: the output is the same either way. None: always from tau0. MUSIC_RESTART_VERIFY=1 runs from tau0 and compares the state with the checkpoint at its step. music4gpu; CPU MUSIC ignores it.

#### `set_skip_surface(skip: bool) -> None`

Do not export this instance's freeze-out surface to the framework. Call it AFTER JetScape.Init(), like set_dump_hydro_only.

#### `set_slim_bulk_info(slim: bool) -> None`

Override for this instance. Call it AFTER JetScape.Init() (which reads the XML flag).

#### `set_store_evolution(store: bool) -> None`

Store this instance's evolution frames in MUSIC's in-memory store (True, default) or not (False), from its next evolution on. False keeps the grid metadata and the frame count (bulk_info, get_native_ntau) but no cells: for an instance whose evolution nobody reads. music4gpu; CPU MUSIC ignores it.

### `Mt19937`

A task's std::mt19937 engine (JetScapeModuleBase.GetMt19937Generator()).

#### `discard(n: typing.SupportsInt | typing.SupportsIndex) -> None`

#### `seed(seed: typing.SupportsInt | typing.SupportsIndex) -> None`

#### `uniform() -> float`

A draw in \[0, 1), as the C++ modules' ZeroOneDistribution(\*gen).

### `Parameter()`

#### `hydro_input_filename: str`

### `Parton`

Bases: `JetScapeParticleBase`

A parton: Parton(label, id, stat, p, x) with p = FourVector(px, py, pz, E) [GeV] and x = FourVector(x, y, z, t) [fm]; or Parton(label, id, stat, pt, eta, phi, e, x=None) with x a list (t, x, y, z) as C++ double[4].

x(), y(), z(), t() are the position; the virtuality is virtuality() (C++ Parton::t()), set with set_t().

#### `anti_color() -> int`

Anti-colour tag (0 = none).

#### `color() -> int`

Colour tag (0 = none).

#### `copy() -> Parton`

A copy.

#### `edgeid() -> int`

#### `form_time() -> float`

#### `isLepton(pid: typing.SupportsInt | typing.SupportsIndex) -> bool`

#### `isNonParton(pid: typing.SupportsInt | typing.SupportsIndex) -> bool`

#### `isPhoton(pid: typing.SupportsInt | typing.SupportsIndex) -> bool`

#### `max_color() -> int`

#### `mean_form_time() -> float`

#### `min_anti_color() -> int`

#### `min_color() -> int`

#### `reset_p(px: typing.SupportsFloat | typing.SupportsIndex, py: typing.SupportsFloat | typing.SupportsIndex, pz: typing.SupportsFloat | typing.SupportsIndex) -> None`

Set the 3-momentum, keeping E.

#### `set_anti_color(acol: typing.SupportsInt | typing.SupportsIndex) -> None`

#### `set_color(col: typing.SupportsInt | typing.SupportsIndex) -> None`

#### `set_form_time(form_time: typing.SupportsFloat | typing.SupportsIndex) -> None`

#### `set_max_color(col: typing.SupportsInt | typing.SupportsIndex) -> None`

#### `set_mean_form_time() -> None`

#### `set_min_anti_color(acol: typing.SupportsInt | typing.SupportsIndex) -> None`

#### `set_min_color(col: typing.SupportsInt | typing.SupportsIndex) -> None`

#### `set_t(t: typing.SupportsFloat | typing.SupportsIndex) -> None`

C++ Parton::set_t(): set the virtuality (rescales the momentum).

#### `virtuality() -> float`

C++ Parton::t(): the virtuality [GeV^2].

### `PartonShower()`

Parton shower as a directed graph (vertices = nodes, partons = edges); ids as in to_numpy() / vertices_to_numpy().

#### `nodes: 'list[ShowerNode]'`

Every vertex as a ShowerNode, in vertices_to_numpy order.

#### `children(parton: Parton) -> list`

The partons coming out of the vertex where this parton ends.

#### `edge_array() -> np.ndarray`

The partons as EDGE_DTYPE records (src, tgt, pid, pstat, E px py pz, t x y z).

#### `edge_of(parton: Parton) -> tuple[int, int, int]`

(edge id, source vertex id, target vertex id) of a parton of this shower.

#### `end_node(parton: 'Parton') -> 'ShowerNode'`

The ShowerNode where `parton` ends.

#### `end_vertex(parton: Parton) -> Vertex`

#### `final_partons() -> list`

The partons at the end of the shower.

#### `is_end_vertex(vertex_id: typing.SupportsInt | typing.SupportsIndex) -> bool`

#### `is_one_to_two(vertex_id: typing.SupportsInt | typing.SupportsIndex) -> bool`

#### `new_parton(source_id: typing.SupportsInt | typing.SupportsIndex, target_id: typing.SupportsInt | typing.SupportsIndex, parton: Parton) -> int`

Add a parton from vertex source_id to vertex target_id.

#### `new_vertex(vertex: Vertex) -> int`

Add a vertex; returns its id.

#### `node_array() -> np.ndarray`

The vertices as NODE_DTYPE records (id, t, x, y, z).

#### `number_of_partons() -> int`

#### `number_of_vertices() -> int`

#### `parents(parton: Parton) -> list`

The partons going into the vertex where this parton starts.

#### `parton_at(i: typing.SupportsInt | typing.SupportsIndex) -> Parton`

Parton on the i-th edge.

#### `partons() -> list`

Every parton (edge) of the shower.

#### `production_vertex(parton: Parton) -> Vertex`

#### `split_delta_r(vertex_id: typing.SupportsInt | typing.SupportsIndex) -> float`

#### `split_kt(vertex_id: typing.SupportsInt | typing.SupportsIndex) -> float`

#### `split_z(vertex_id: typing.SupportsInt | typing.SupportsIndex) -> float`

z of the 1 -> 2 split at vertex id.

#### `start_node(parton: 'Parton') -> 'ShowerNode'`

The ShowerNode where `parton` starts.

#### `to_numpy() -> numpy.typing.NDArray[numpy.float64]`

Flatten the shower's edges to a numpy float64 array, one row per parton, shape (n_partons, 12):

[source_id, target_id, pid, pstat, px, py, pz, E, x, y, z, t]

Momentum (px,py,pz,E) in GeV; position (x,y,z) in fm; t in fm/c. (x,y,z,t) is the parton's production position (source vertex); use source_id / target_id to correlate with vertices_to_numpy().

#### `vertex_at(i: typing.SupportsInt | typing.SupportsIndex) -> Vertex`

Vertex on the i-th node.

#### `vertex_by_id(id: typing.SupportsInt | typing.SupportsIndex) -> Vertex`

#### `vertex_id(vertex: Vertex) -> int`

#### `vertices_to_numpy() -> numpy.typing.NDArray[numpy.float64]`

Flatten the shower's nodes to a numpy float64 array, one row per splitting vertex, shape (n_vertices, 5):

[node_id, x, y, z, t]

Position (x,y,z) in fm; t in fm/c. node_id matches the source_id / target_id columns in to_numpy(), allowing the two arrays to be joined into a directed graph (see jetscape.utils.shower_to_networkx).

### `PartonShowerGenerator()`

How a JetEnergyLoss task grows its shower (DoShower(jloss), and the per-time-step hooks). Subclass it in Python and hand it to JetEnergyLoss.AddPartonShowerGenerator().

#### `DoCalculateTime(jloss: JetEnergyLoss) -> None`

#### `DoExecTime(jloss: JetEnergyLoss) -> None`

#### `DoFinishPerEvent(jloss: JetEnergyLoss) -> None`

#### `DoInitPerEvent(jloss: JetEnergyLoss) -> None`

#### `DoShower(jloss: JetEnergyLoss) -> None`

### `PartonShowerGeneratorDefault()`

Bases: `PartonShowerGenerator`

The final-state shower generator every JetEnergyLoss task uses by default.

### `Photon(label: typing.SupportsInt | typing.SupportsIndex, id: typing.SupportsInt | typing.SupportsIndex, stat: typing.SupportsInt | typing.SupportsIndex, p: FourVector, x: FourVector)`

Bases: `Parton`

A photon from a shower: Photon(label, id, stat, p, x) as Parton.

### `PreequilibriumDynamics()`

Bases: `JetScapeModuleBase`

Interface for pre-equilibrium dynamics modules. All stress-energy-tensor fields (e\_, P\_, ux\_, uy\_, ueta\_, pi??*, bulk_Pi*) are public std::vector members — directly accessible as zero-copy numpy arrays via the get\_\*\_numpy() methods.

#### `preequilibrium_tau_0_: float`

#### `preequilibrium_tau_max_: float`

#### `EvolvePreequilibrium() -> None`

Every event, from ExecuteTask().

#### `GetPreequilibriumEndTime() -> float`

Proper time tau at the end of pre-equilibrium [fm/c].

#### `GetPreequilibriumEvodtau() -> float`

#### `GetPreequilibriumStartTime() -> float`

Proper time tau_0 at the start of pre-equilibrium [fm/c].

#### `GetPreequilibriumStatus() -> int`

0 NOT_STARTED, 1 INIT, 2 DONE, 3 ERR.

#### `InitializePreequilibrium() -> None`

Once, from Init() (after tau0 and taus are read).

#### `SetPreequilibriumStatus(status: typing.SupportsInt | typing.SupportsIndex) -> None`

#### `clear_evolution_data() -> None`

#### `get_P_numpy() -> numpy.typing.NDArray[numpy.float64]`

Pressure [GeV/fm^3], shape (nx\*ny,).

#### `get_bulk_Pi_numpy() -> numpy.typing.NDArray[numpy.float64]`

Bulk pressure Pi [GeV/fm^3], shape (nx\*ny,).

#### `get_e_numpy() -> numpy.typing.NDArray[numpy.float64]`

Energy density [GeV/fm^3], shape (nx\*ny,).

#### `get_fluid_cell_with_index(idx: typing.SupportsInt | typing.SupportsIndex) -> typing.Any`

#### `get_ini_pointer() -> InitialState`

The initial state (set by Init()).

#### `get_ntau() -> int`

#### `get_number_of_fluid_cells() -> int`

#### `get_pi00_numpy() -> numpy.typing.NDArray[numpy.float64]`

pi^{00} [GeV/fm^3], shape (nx\*ny,).

#### `get_pi01_numpy() -> numpy.typing.NDArray[numpy.float64]`

pi^{01}, shape (nx\*ny,).

#### `get_pi02_numpy() -> numpy.typing.NDArray[numpy.float64]`

pi^{02}, shape (nx\*ny,).

#### `get_pi03_numpy() -> numpy.typing.NDArray[numpy.float64]`

pi^{03}, shape (nx\*ny,).

#### `get_pi11_numpy() -> numpy.typing.NDArray[numpy.float64]`

pi^{11}, shape (nx\*ny,).

#### `get_pi12_numpy() -> numpy.typing.NDArray[numpy.float64]`

pi^{12}, shape (nx\*ny,).

#### `get_pi13_numpy() -> numpy.typing.NDArray[numpy.float64]`

pi^{13}, shape (nx\*ny,).

#### `get_pi22_numpy() -> numpy.typing.NDArray[numpy.float64]`

pi^{22}, shape (nx\*ny,).

#### `get_pi23_numpy() -> numpy.typing.NDArray[numpy.float64]`

pi^{23}, shape (nx\*ny,).

#### `get_pi33_numpy() -> numpy.typing.NDArray[numpy.float64]`

pi^{33}, shape (nx\*ny,).

#### `get_ueta_numpy() -> numpy.typing.NDArray[numpy.float64]`

u^eta, shape (nx\*ny,).

#### `get_utau_numpy() -> numpy.typing.NDArray[numpy.float64]`

u^tau, shape (nx\*ny,).

#### `get_ux_numpy() -> numpy.typing.NDArray[numpy.float64]`

u^x, shape (nx\*ny,).

#### `get_uy_numpy() -> numpy.typing.NDArray[numpy.float64]`

u^y, shape (nx\*ny,).

#### `set_P_numpy(values: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None`

Set the field (a copy, flattened).

#### `set_bulk_Pi_numpy(values: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None`

Set the field (a copy, flattened).

#### `set_e_numpy(values: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None`

Set the field (a copy, flattened).

#### `set_pi00_numpy(values: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None`

Set the field (a copy, flattened).

#### `set_pi01_numpy(values: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None`

Set the field (a copy, flattened).

#### `set_pi02_numpy(values: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None`

Set the field (a copy, flattened).

#### `set_pi03_numpy(values: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None`

Set the field (a copy, flattened).

#### `set_pi11_numpy(values: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None`

Set the field (a copy, flattened).

#### `set_pi12_numpy(values: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None`

Set the field (a copy, flattened).

#### `set_pi13_numpy(values: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None`

Set the field (a copy, flattened).

#### `set_pi22_numpy(values: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None`

Set the field (a copy, flattened).

#### `set_pi23_numpy(values: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None`

Set the field (a copy, flattened).

#### `set_pi33_numpy(values: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None`

Set the field (a copy, flattened).

#### `set_ueta_numpy(values: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None`

Set the field (a copy, flattened).

#### `set_utau_numpy(values: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None`

Set the field (a copy, flattened).

#### `set_ux_numpy(values: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None`

Set the field (a copy, flattened).

#### `set_uy_numpy(values: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None`

Set the field (a copy, flattened).

### `ShowerNode(shower, node_id: int)`

A vertex of a PartonShower by id, with its position, the partons in and out and the split variables (PartonShower's id-based methods).

#### `children: list`

The partons that start here.

#### `id = int(node_id)`

#### `is_end: bool`

#### `is_one_to_two: bool`

#### `parents: list`

The partons that end here.

#### `position: 'core.FourVector'`

FourVector (iterates as t, x, y, z).

#### `shower = shower`

#### `split_delta_r: float`

#### `split_kt: float`

#### `split_z: float`

#### `vertex: 'core.Vertex'`

### `SoftParticlization()`

Bases: `JetScapeModuleBase`

Particlization of the hydro's freeze-out surface (iSS in C++).

A Python sampler subclasses it, gets the surface in ExecuteTask() with GetHydroHyperSurface() (or FindHydroHyperSurface() from the stored evolution) and stores the event's hadrons with set_hadron_list(samples), one list of Hadrons per oversample. register_module("CustomModuleX", cls) and make the XML use it.

#### `last_random_seed: int`

The seed used for the last event (soft_last_random_seed).

#### `ClearHydroHyperSurface() -> None`

#### `GetHydroHyperSurface() -> list`

The hydro's freeze-out surface (its own, as MUSIC gives it) as SurfaceCellInfos; empty without one.

#### `GetLastRandomSeed() -> int`

#### `SetNextRandomSeed(seed: typing.SupportsInt | typing.SupportsIndex) -> None`

#### `SetNumberOfSamples(n: typing.SupportsInt | typing.SupportsIndex) -> bool`

#### `get_hadron_list() -> list`

The event's hadrons: one list of Hadrons per oversample.

#### `hadron_array() -> HadronArray`

This event's hadrons (soft_hadrons_numpy) as a HadronArray, one sample per oversample. With use_compact_output() the arrays are taken over without a copy: call it once per event.

#### `primary_array() -> 'HadronArray | None'`

The undecayed hadrons of the first N oversamples (soft_primaries_numpy) as a HadronArray; None if none were kept. Once per event, after hadron_array().

#### `sample_cells(cells, seed: int, n_samples: int) -> HadronArray`

Sample the given (N, 32) surface cells now (soft_sample_cells) -> HadronArray.

#### `set_hadron_list(samples: list) -> None`

Store the event's hadrons: a list of oversamples, each a list of Hadrons.

#### `set_next_random_seed(seed: int) -> None`

Seed for the next event's sampling, one-shot (soft_set_next_random_seed).

#### `set_number_of_samples(n: int) -> None`

Oversamples per event from the next event on (soft_set_number_of_samples).

#### `use_compact_output(on: bool = True) -> None`

Hand the hadrons over as arrays from the next event on (soft_set_compact_output).

### `SurfaceCellInfo()`

#### `bulk_Pi: float`

Bulk viscous pressure [GeV/fm^3].

#### `d3sigma_mu: list[float]`

Surface normal vector d^3 sigma_mu (4-vector).

#### `energy_density: float`

Energy density on the surface [GeV/fm^3].

#### `entropy_density: float`

Entropy density on the surface [1/fm^3].

#### `eta: float`

Space-time rapidity.

#### `mu_B: float`

#### `mu_Q: float`

#### `mu_S: float`

#### `pi: list[float]`

Shear stress tensor pi (10 independent components).

#### `pressure: float`

Pressure on the surface [GeV/fm^3].

#### `qgp_fraction: float`

#### `tau: float`

Proper time tau [fm/c].

#### `temperature: float`

Temperature on the surface [GeV].

#### `umu: list[float]`

4-velocity u^mu.

#### `x: float`

Transverse x [fm].

#### `y: float`

Transverse y [fm].

### `TrentoInitial()`

Bases: `InitialState`

TRENTo initial-condition module.

Registered module name: `"TrentoInitial"` — instantiate via::

```text
ini = create_module("TrentoInitial")
```

After Exec() completes: * `ini.get_entropy_density_numpy()` returns the entropy-density grid as a 2D numpy array. * `ini.get_event_info()` returns a dict of per-event geometric quantities (impact parameter, Npart, Ncoll, eccentricities …).

#### `GetEventCentrality() -> float`

Return the event centrality percentile [0–100].

#### `get_eccentricity(order: typing.SupportsInt | typing.SupportsIndex) -> float`

Return the n-th order participant-plane eccentricity epsilon_n.

#### `get_event_centrality() -> float`

Return the event centrality percentile [0–100].

#### `get_event_info() -> dict`

Return per-event geometric information as a Python dict.

Keys

impact_parameter, num_participant, num_binary_collisions, total_entropy, normalization, event_centrality, xmid, ymid, eccentricities (dict {order: epsilon_n}), participant_planes (dict {order: psi_n}).

#### `get_impact_parameter() -> float`

Return the impact parameter b [fm].

#### `get_num_binary_collisions() -> float`

Return the number of binary collisions (Ncoll).

#### `get_num_participant() -> float`

Return the number of wounded nucleons (Npart).

#### `get_participant_plane_angle(order: typing.SupportsInt | typing.SupportsIndex) -> float`

Return the n-th order participant-plane angle psi_n [rad].

#### `get_total_entropy() -> float`

Return the total entropy of the event.

### `Vertex`

A splitting vertex: space-time position (x,y,z) [fm], t [fm/c]. Vertex(x, y, z, t) or Vertex(FourVector).

#### `set_location(x: FourVector) -> None`

#### `t() -> float`

#### `x() -> float`

#### `x_in() -> FourVector`

#### `y() -> float`

#### `z() -> float`

### `iMATTER()`

Bases: `JetEnergyLoss`

MATTER's initial-state (space-like) shower.

### `create_module(name: str) -> typing.Any`

Create a registered JETSCAPE C++ module by name.

Parameters:

- **`name`** (`str`) – Registered module name, e.g. "TrentoInitial", "FreestreamMilne", "FnoHydro", "JetEnergyLossManager", "Matter", …

Returns:

- `JetScapeModuleBase` – A shared_ptr-managed instance of the requested module.

Raises:

- `ValueError` – If the name is not found in the module registry.

### `extra_hard_info(task: JetScapeTask) -> dict`

PythiaGun's : `on`, `no_trigger`, `pthat_min` [GeV], the warm-up cross section `sigma` +- `sigma_err` and `sigma_nn` [mb], `warmup`, `fixed_n` (< 0: Poisson), `pythia_seed`, `rng_seed`; the current event's `n_extra` and Poisson mean `mu` (N_coll sigma/sigma_nn), `scatterings` (S, 8) in SCATTER_COLUMNS (scatter, pthat, code, x, y, n_partons, E, pt; the trigger's first, scatter 0, unless no_trigger) and `parton_scatter` (int, one per hard parton in the framework's order, which is the shower index). Call between ExecPerEvent() and ClearPerEvent().

### `hadronization_hadrons_numpy(task: JetScapeTask) -> dict`

Output hadrons of a HadronizationManager (all its Hadronization tasks) or of one Hadronization task, as a dict of arrays. Call between ExecPerEvent() and ClearPerEvent().

### `hadronization_input_numpy(module: JetScapeTask) -> dict`

iColoredHadronization with 1: the parton list the last event handed to Pythia (final partons, beam remnants, the colour-balancing parton), as a dict of pid, col, acol (int32), p (N, 4) float64 [E, px, py, pz] and mass float64. Empty without . Accepts the module, its parent or the HadronizationManager.

### `hadronize_input(module: JetScapeTask, pid: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], col: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], acol: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], p: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], mass: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], seed: typing.Any = None) -> dict`

Hadronize a parton list stored with hadronization_input_numpy again with iColoredHadronization (create_module('iColoredHadronization') after a JetScape Init() that loaded the job's XML, then module.Init()). With the stored `seed` and the same Pythia version and settings the hadrons are those of the job. Returns the hadron dict plus the seed used.

### `hadronize_partons(module: JetScapeTask, partons: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], seed: typing.Any = None) -> dict`

Hadronize stored final partons (an (N, 14) array, FINAL_PARTON_COLUMNS) with a jet hadronization module, e.g. create_module('ColorlessHadronization') after a JetScape Init() that loaded the XML and after module.Init(). With `seed`, the ColorlessHadronization generators are reseeded with it first (exact replay of an event hadronized with 1). iColoredHadronization (ICOLORED_HAS_COLOUR_CLOSURE) takes the partons in their stored order, without beam remnants, through its pstat policy and (jet_hadronization_colour_audit afterwards). Returns the hadron dict plus n_partons_out and the seed used.

### `hard_partons_numpy(task: JetScapeTask) -> numpy.typing.NDArray[numpy.float64]`

The partons the hard process handed to the framework this event (for PythiaGun: status 62 after ISR/MPI, or the final partons with FSR_on), as an (n, 8) array pid, px, py, pz, E, pT, y, scatter, hardest first (scatter: the hard scattering, 0 the event's triggering one, 1, 2, ... the extras of PythiaGun's ). For PythiaGun the scatter-0 rows are what cuts on. Call between ExecPerEvent() and ClearPerEvent().

### `hard_process_info(task: JetScapeTask) -> dict`

The current event's hard process: `pthat` [GeV], `sigma_gen` and `sigma_err` (the generator's running cross-section estimate, mb; for PythiaGun with several pTHat windows, the current event's window) and `event_weight`. Call between ExecPerEvent() and ClearPerEvent().

### `initial_state_info(task: JetScapeTask) -> dict`

The current event's geometry from its initial-state module: `n_coll` (binary collisions), `n_part` and `centrality`; -1 where the module does not provide it (3dMCGlauber gives n_coll only). A hard process scales with n_coll, so events of a wide centrality class sampled minimum-bias-like (P(b) ~ b, one hard process each) are weighted by it to match data. Call between ExecPerEvent() and ClearPerEvent().

### `isr_hard_momentum(task: JetScapeTask) -> dict`

With initial-state radiation (PythiaIsrGun + iMATTER + 3dMCGlauber): this event's hard-parton four-momentum (E, px, py, pz) [GeV] handed to the initial state for subtraction from the projectile (`proj`, +z) and target (`targ`, -z) nucleon of the hard collision; `subtract` (1: subtracted, 0: 3dMCGlauber zeroes it first, -1: no such setting); and the four-momenta of those two nucleons' remnants (`remnant_proj`, `remnant_targ`), which iColoredHadronization hadronizes. Empty lists from a module without them. Call between ExecPerEvent() and ClearPerEvent().

### `jet_hadronization_colour_audit(module: JetScapeTask) -> dict`

iColoredHadronization: the colour audit of the last event (PLAN_lbt_colour_tags.md, C3): partons and remnants handed to Pythia, holes (pstat -1, dropped) and their energy, other negative pstat skipped, unpaired colour / anticolour tags, duplicate tags and how many were relabelled. Accepts the module, its parent or the HadronizationManager.

### `jet_hadronization_colour_totals(module: JetScapeTask) -> dict`

iColoredHadronization: the colour audits added up over the run (events hadronized, failed = thrown, events with open lines, holes and their energy, open ends, joins, partners, soft end partons, retries, duplicates); what the run's log summary prints.

### `jet_hadronization_last_random_seed(module: JetScapeTask) -> int`

The seed ColorlessHadronization (or iColoredHadronization) used for the last event (0 if it was not reseeded). Accepts the module itself, its Hadronization parent or the HadronizationManager.

### `jet_hadronization_set_colour_closure(module: JetScapeTask, mode: str) -> str`

iColoredHadronization: set the closure of open colour lines (reconnect, reconnect_global, thermal, fake, none) instead of the XML's ; returns it.

### `load_xml(main_xml: str, user_xml: str = '', init_random: bool = True) -> bool`

Open the main (and optionally user) XML into the JetScapeXML singleton now.

Call this BEFORE constructing any C++ module that reads XML in its constructor -- CausalLiquefier is the one that matters. JetScape.Init() does the same thing, but that is too late for modules built beforehand. Calling it twice is harmless.

With init_random=True (the default) it also seeds JetScapeTaskSupport from , so code that draws random numbers before JetScape.Init() -- such as InitialState.sample_binary_collision_point() -- works.

Returns True if the main file parsed.

### `make_hadron(pid: int, momentum, position=(0.0, 0.0, 0.0, 0.0), stat: int = 0, label: int = 0, mass: float | None = None) -> 'core.Hadron'`

A Hadron by keywords: momentum (E, px, py, pz), position (t, x, y, z) as any 4-sequence or a FourVector; without a mass the PDG mass is used (C++ Hadron).

### `make_parton(pid: int, momentum, position=(0.0, 0.0, 0.0, 0.0), stat: int = 0, label: int = 0, *, color: int = 0, anti_color: int = 0, cls=None) -> 'core.Parton'`

A Parton by keywords: momentum (E, px, py, pz) and position (t, x, y, z) as any 4-sequence or a FourVector; colour tags optional. cls=Photon for a photon.

### `make_photon(momentum, position=(0.0, 0.0, 0.0, 0.0), stat: int = 0, label: int = 0) -> 'core.Photon'`

A Photon (pid 22) by keywords, as make_parton.

### `module(name: str, section: str = '') -> 'Callable[[type], type]'`

Class decorator: register a Python module under `name` (register_module), with a main-XML default built from the class's `xml_parameters` ({key: default}, int, float, str or bool). The class's InitTask first sets self. from [section, name, key] in the XML; the defaults are also class attributes, so a module built by hand has them before Init. Use a CustomModule... name for modules the XML names ().

### `next_task_number() -> int`

The task ID the next task made will get (JetScapeTaskSupport).

### `per_event(main_xml: str | None = None, user_xml: str | None = None, tasks: Sequence | None = None, n_events: int | None = None) -> 'core.JetScapePerEvent'`

A JetScapePerEvent driver by keywords: the XML files, a by-hand pipeline (tasks added in order; else the XML's task list) and the number of events.

```text
with js.per_event(main_xml, user_xml) as driver:
    for _ in driver.events():
        fp = driver.energy_loss_manager.final_parton_array()
```

The bound constructor JetScapePerEvent() stays as it is.

### `pgun_points(task: JetScapeTask) -> dict`

PGun's (pT, rapidity) points (, else the one of /), the current event's point (i mod K), phi and parID.

### `pythia_gun_bins(task: JetScapeTask) -> dict`

PythiaGun's or PythiaIsrGun's pTHat windows (; one window, pTHatMin-pTHatMax, without it): `n_bins`, `bins` [(min, max)], `active` (the current event's window: event i uses i mod n_bins), and per window `seeds` (Pythia's Random:seed), `sigma_gen`, `sigma_err` [mb] and `n_accepted` so far. After Init(): the windows and seeds; after the last event: the final cross sections. With PYTHIA_GUN_HAS_PARTON_Y_CUT also `y_max` (0 = no cut), `y_mode` and per window `n_tried` / `n_kept` (events reaching / passing the cut), `acceptance` and `sigma_gen_raw` / `sigma_err_raw` (Pythia's own, which counts the rejected events; `sigma_gen` is it times the acceptance). With PYTHIA_GUN_HAS_Z_TRIGGER also `min_partons`, `trigger` ('none' or 'Z') and the Z acceptance `trigger_y_max`, `lepton_pt_min`, `lepton_eta_max` (0 = no cut); with a trigger the per-window counts and the acceptance are those of both cuts.

### `pythia_gun_trigger(task: JetScapeTask) -> dict`

PythiaGun's `trigger` ('none' or 'Z') and the current event's trigger `leptons`: the Z's opposite-charge, same-flavour pair as handed to the framework (the pair with the largest invariant mass if there are several), an (n, 5) array pid, px, py, pz, E, hardest first; n = 2 with trigger Z, else 0. Call between ExecPerEvent() and ClearPerEvent().

### `random_seed() -> int`

The run's as JetScapeTaskSupport read it (0 before JetScape.Init() or load_xml(); a time-based seed when the XML says 0).

### `register_module(name: str, factory: typing.Any, section: str = '', defaults: str = '') -> None`

Register a Python module class (or any callable returning a module) in JETSCAPE's module factory under `name`.

create_module(name) then makes one, and so does the XML's automatic task list: every section takes a module whose element name contains `CustomModule` (e.g. `<Eloss><CustomModulePyBrick>`), so register modules meant for the XML under such a name. C++ owns what it makes; the Python object lives as long as C++ holds it.

JetScape refuses a user-XML tag the main XML lacks, so JetScape.Init() adds a default entry for each registered module to the main XML (in memory): under `section` (e.g. "Eloss"; "" = the root) with the children in `defaults` (XML text, e.g. "40"). A module reads its parameters with get_xml_element\_\*([section, name, ...]): the user XML's value, else this default.

Registering a name again replaces the Python factory; a C++ module's name is refused.

### `registered_modules(python_only: bool = False) -> list[str]`

The names in JETSCAPE's module factory (C++ modules of this build and the Python ones registered); python_only=True lists only the latter.

### `select(arr, species)`

Mask of the hadrons (or partons) of `species` in arr: a HadronArray (flat mask), an awkward array with a `pid` field (a mask of its shape), a record array or a pid array. species: a name or class, or a list of them (any of them selects).

### `set_next_task_number(n: typing.SupportsInt | typing.SupportsIndex) -> None`

Set the task ID the next task gets: back to next_task_number()'s value before modules made outside the physics, so later tasks keep their random seeds.

### `set_writer_output_file(writer: JetScapeModuleBase, file_name: str) -> None`

Set the output file of a writer created with create_module(), e.g. create_module("JetScapeWriterFinalStateHadronsAscii"). Call before JetScape.Init(), which opens the file.

### `soft_hadrons_numpy(task: JetScapeTask) -> dict`

All hadrons a SoftParticlization module (iSS) produced this event: dict of pid, pstat, p [E,px,py,pz], x [t,x,y,z], mass, and sample_counts (one entry per oversample; the rows are the samples concatenated in order). Call between ExecPerEvent() and ClearPerEvent().

### `soft_last_random_seed(task: JetScapeTask) -> int`

The seed the SoftParticlization module used for the last event.

### `soft_primaries_numpy(task: JetScapeTask) -> typing.Any`

The undecayed (primary) hadrons of the first N oversamples this event, as sampled before any decay (iSS N, with soft_set_compact_output): the soft_hadrons_numpy dict, sample_counts with N entries. None if none were kept. Takes them over without copying: call it once per event, after soft_hadrons_numpy.

### `soft_sample_cells(task: JetScapeTask, cells: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], seed: typing.SupportsInt | typing.SupportsIndex, n_samples: typing.SupportsInt | typing.SupportsIndex) -> dict`

Sample the given surface cells now, outside the framework's event loop: an (N, 32) float32 array (SURFACE_CELL_COLUMNS, e.g. a particlize file's surface unit), this seed and n_samples oversamples. Returns soft_hadrons_numpy's dict (one entry of sample_counts per oversample; none for no cells); with iSS the undecayed hadrons follow from soft_primaries_numpy. Equal to replaying the cells through the event loop with soft_set_next_random_seed. The module must be initialised (Init); it need not be in a task list.

### `soft_set_compact_output(task: JetScapeTask, on: bool = True) -> None`

From the next event on, hand the hadrons over as flat arrays instead of one Hadron object each: the same soft_hadrons_numpy result with ~8x less memory. soft_hadrons_numpy then takes the arrays over without copying, so call it once per event. The framework's Hadron_list\_ stays empty (no bulk hadrons for writers or afterburners). iSS: ExecuteTask only.

### `soft_set_next_random_seed(task: JetScapeTask, seed: typing.SupportsInt | typing.SupportsIndex) -> None`

Use `seed` for the next event's sampling (one-shot) instead of a draw from the module's generator.

### `soft_set_number_of_samples(task: JetScapeTask, n: typing.SupportsInt | typing.SupportsIndex) -> None`

Samples (oversamples) per event from the next event on (iSS: number_of_repeated_sampling). Call after Init().

### `to_awkward(obj, position=False, charge=False, by_unit=False)`

The nested awkward view of a HadronArray, a hadron dict or a record table.

position: add the nested `position` (HadronArray; records always carry theirs). charge: add a `charge` field [e] from the PDG (needs `particle`). by_unit: a HadronArray read from a file (from_h5): units x samples x hadrons, over info["unit_offsets"].

### `unregister_module(name: str) -> bool`

Remove a Python module registered with register_module(); False if the name was not one.

### `xml_add_user_blocks(text: str) -> int`

Append the children of `<jetscape>` in `text` to the open user XML, in memory.

After JetScape.Init(), so the automatic task list never sees them: modules made afterwards (create_module + Init) read their settings from them, while the event loop keeps exactly the tasks the file had. A block the user XML has already is refused. Returns the number of blocks added.
