# Plan: PyVista hydro-medium evolution visualizer (Milne → Cartesian)

## Context

X-SCAPE now has a per-event Python workflow (`per_event_loop`) and a pybind11
interface that exposes the live hydro `EvolutionHistory` via
`JetScapeSignalManager`. We want to *visualize* the bulk medium evolution in 3D
with PyVista. The hydro lives on a **Milne** grid `(τ, x, y, η_s)`, but to set up
the **next phase** — overlaying parton-shower propagation, which happens in
Cartesian `(t, x, y, z)` — the medium must be shown in **Cartesian lab spacetime**.

Goal: a new script in the (currently empty) `js-contrib` `Visualization/`
directory that, for each event of a per-event run, resamples the Milne hydro to a
Cartesian `(x,y,z)` volume at a sequence of **lab times `t`**, and renders it with
PyVista (energy-density volume + freeze-out temperature isosurface), producing a
movie, a ParaView-loadable VTK time series, and/or an interactive window. The code
is structured so the parton-overlay phase can hook straight in.

Decisions confirmed with the user:
- **Time axis:** lab-time `t` Cartesian animation (medium expands/recedes in z).
- **Default view:** energy-density **volume** + temperature **isosurface** (freeze-out shell).
- **Outputs:** movie (mp4/gif) **and** VTK `.vti`+`.pvd` **and** interactive — all CLI-selectable.
- **Velocity glyphs:** optional `--velocity` flag (off by default), boosted to lab frame.

Runs in the existing conda env **`fno_pyvista_env`** (verified: pyvista 0.47.3,
scipy 1.16.3, vtk 9.6.1, imageio 2.37.3, numpy 2.3.5, Python 3.13 — matches the
compiled `pyjetscape_core.cpython-313` module).

## Files

- **Create** `external_packages/js-contrib/contribs/Visualization/hydro_pyvista.py` — the visualizer.
- **Create** `external_packages/js-contrib/contribs/Visualization/README.md` — short usage + env + next-phase hook note.

No C++ changes and **no rebuild** required.

## What to reuse (do not reinvent)

- Per-event driver: `per_event_loop(main_xml, user_xml, modules=None, n_events=…, start_event=…)`
  in `external_packages/js-contrib/contribs/PyJetscape/python/jetscape/run_jetscape.py:255`.
  Yields the `JetScapePerEvent` driver once per event with live data in memory.
- Hydro access pattern (mirror exactly): `JetScapeSignalManager.Instance().GetHydroPointer().get_bulk_info()`
  — see `bulk_root_writer.py:175` and `example/per_event_loop.py:117`.
- Fast extraction: `bulk_info_to_numpy(bulk, n_features)` → `bulk.to_numpy(4)` →
  `(ntau, nx, ny, nf)`, layout `[energy_density, temperature, vx, vy]`, one C++ pass
  (`python/jetscape/utils.py:41`, `src/bind_evolution.cc:229`).
- Grid metadata accessors on the bulk object: `tau_min, dtau, x_min, dx, y_min, dy,
  ntau, nx, ny, eta_min, deta, neta, boost_invariant` (all bound, def_readwrite;
  see `src/bind_evolution.cc:130`).
- CLI/skeleton conventions (argparse, `sys.path` setup, `--manual` pipeline,
  GaussianIC fallback, `--load`): copy from `example/inspect_bulk_info.py` and
  `example/per_event_loop.py`.
- Milne→Cartesian math + the boost-invariant velocity boost are already in C++
  `get_tz()` (`src/framework/FluidEvolutionHistory.cc:397`); `get_tz` is **not**
  bound to Python, so we replicate it **vectorized in numpy**.

## Architecture of `hydro_pyvista.py`

`sys.path` setup: insert `…/contribs/PyJetscape/python` so `import jetscape` works
(note: this script's parent is `Visualization/`, not `PyJetscape/`, so compute the
PyJetscape path explicitly rather than the `_REPO_ROOT` trick used in the examples).

### 1. Data acquisition
- `iter_events(args)` — generator. Wraps `per_event_loop(...)`; for each yielded
  `js`: `hydro = JetScapeSignalManager.Instance().GetHydroPointer()`,
  `bulk = hydro.get_bulk_info()`, `arr = bulk_info_to_numpy(bulk, 4)`, and read the
  metadata dict. Yields `(event_id, arr, meta)`. Supports `--manual` pipeline
  (build `[ini, preeq, hydro]` via `create_module`, like per_event_loop.py).
- `load_milne(path)` — for `--load`: read a saved `.npz` (`arr` + `grid_info`),
  yield a single `(0, arr, meta)`. Pair with `--save-milne PATH` to dump from a run
  for fast render iteration without re-running hydro.

### 2. Milne → Cartesian resampling (vectorized numpy/scipy)
- `build_interpolator(arr, meta)` → `scipy.interpolate.RegularGridInterpolator`
  over axes `(τ, x, y)` with `values=arr` (shape `(ntau,nx,ny,4)`; RGI carries the
  trailing feature axis), `method="linear"`, `bounds_error=False`, `fill_value=0`.
- `cartesian_frame(interp, t, cart_grid, meta, want_velocity)` for one lab time `t`:
  - Cartesian axes: transverse `x,y` reuse the hydro grid (`x_min,dx,nx`,`y_min,dy,ny`);
    `z = linspace(-z_max, z_max, nz)` (display choice — see §4).
  - Per `z`: `tau = sqrt(t² − z²)` (only where `t² > z²`), `eta = 0.5*ln((t+z)/(t−z))`.
    `tau` depends on `z` only, constant over the x–y plane → build query points
    `(tau_z, X, Y)` of shape `(nz, nx, ny, 3)`, call `interp` once → `(nz,nx,ny,4)`,
    transpose to `(nx,ny,nz,4)`.
  - Mask cells where `|z| ≥ t` **or** `tau < tau_min` **or** `tau > tau_max` → 0.
  - Fields: `e = [...,0]`, `T = [...,1]`. If `want_velocity`: take `vx,vy` from
    `[...,2:4]`, set `vz = z/t`, `gammaL = 1/sqrt(1−vz²)`, divide `vx,vy` by `gammaL`
    (exactly mirroring `get_tz`'s boost-invariant branch).
  - Returns a dict `{e, T, [vx, vy, vz]}` of `(nx,ny,nz)` arrays.

  *Boost-invariance note:* `to_numpy` returns the `η_s = 0` slice, and the current
  MUSIC setup is 2+1D boost-invariant (`boost_invariant == True`), so the field is
  η-independent and the (τ,x,y) interpolator is exact. Guard at startup: if
  `not boost_invariant` (neta > 1), print a clear warning that the script currently
  uses the central η slice, and leave a documented extension point to instead loop
  `bulk.get_fluid_cell(k,i,j,id_eta)` / `bulk.get(τ,x,y,η)` over the full η grid.

### 3. PyVista rendering
- `make_image_data(fields, cart_grid)` → `pv.ImageData(dimensions=(nx,ny,nz),
  spacing=(dx,dy,dz), origin=(x_min,y_min,-z_max))`; attach `point_data["e"]`,
  `point_data["T"]`, and (if velocity) `point_data["v"]` as an `(N,3)` array.
  Flatten with Fortran order to match VTK point ordering.
- `render_event(event_id, frames, cart_grid, args, overlay=None)`:
  - `pv.Plotter(off_screen=not args.interactive)`. On non-darwin headless, attempt
    `pv.start_xvfb()` in a try/except (covers the GB10/Linux case).
  - Compute a **global** `clim` for `e` across all frames so the color/opacity scale
    is stable through the animation.
  - If movie: `plotter.open_movie(path)` (`.mp4`) or `plotter.open_gif(path)`.
  - Per frame `t`: rebuild/overwrite the grid scalars; `add_volume(grid,
    scalars="e", cmap="inferno", opacity="sigmoid", clim=clim)`; overlay
    `grid.contour([T_fo], scalars="T")` as a translucent isosurface (default
    `T_fo=0.155` GeV, `--freeze-temp`); optional `grid.glyph(...)` arrows when
    `--velocity`. Add a corner text actor `event N, t = … fm/c`. Call
    `overlay(plotter, t)` if provided (parton-overlay hook). `plotter.write_frame()`
    for movie; manage per-frame actors via stored refs + `remove_actor`.
  - If `--vtk-dir`: `grid.save(f"{dir}/event{eid}_t{frame:03d}.vti")` per frame and
    write a `.pvd` collection (small XML: `<DataSet timestep=… file=…/>` per frame)
    so ParaView opens the whole series with real time values.
  - If `--interactive`: after building, `plotter.show()`.
- `overlay=None` parameter is the **parton-shower extensibility hook**: the next
  phase passes a callback that draws `pv.PolyData` lines/points for parton tracks at
  matching lab time `t`.

### 4. CLI (argparse)
- Run/source: `--main` (default `…/X-SCAPE/config/jetscape_main.xml`), `--user`
  (default the MUSIC bulk config used by inspect_bulk_info:
  `…/PyJetscape/fno_hydro/config/jetscape_user_root_bulk_test.xml`), `--events`,
  `--start-event`, `--manual` + `--initial-state/--preequilibrium/--hydro-module`,
  `--load`, `--save-milne`.
- Grid/time: `--nz` (default 64), `--z-max` (default `0.8 * t_max`), `--nt`
  (default = `ntau`), `--t-min` (default `tau_min`), `--t-max` (default `tau_max`).
- Render: `--field {e,T,both}` (default `both`), `--freeze-temp` (default 0.155,
  repeatable), `--velocity`, `--cmap` (default inferno).
- Output: `--movie PATH` (`.mp4`/`.gif` by extension), `--vtk-dir DIR`,
  `--interactive`, `--outdir` (default `hydro_pyvista_out/`). At least one of
  movie/vtk/interactive required.
- `main()`: for each `(event_id, arr, meta)` from `iter_events`/`load_milne`, build
  the interpolator, compute frames over `t`, call `render_event` for the selected
  outputs.

### README.md
Short: purpose, `conda activate fno_pyvista_env`, example commands (live per-event
run, `--load` fast path, headless movie), the Milne→Cartesian convention, the
boost-invariant caveat, and a one-paragraph "Phase 2: parton overlay" note pointing
at the `overlay` hook.

## Verification

```bash
conda activate fno_pyvista_env
cd external_packages/js-contrib/contribs/Visualization

# A. Fastest smoke test of the renderer (no hydro run): dump once, then render.
#    First produce a small npz from a 1-event run:
python hydro_pyvista.py --events 1 --save-milne /tmp/milne.npz --vtk-dir /tmp/vti_out --nt 8 --nz 32
#    Then re-render purely from the dump (proves Milne→Cartesian + PyVista path):
python hydro_pyvista.py --load /tmp/milne.npz --movie /tmp/hydro.gif --field both

# B. Full live per-event path, headless movie + ParaView series:
python hydro_pyvista.py --events 2 --movie hydro_pyvista_out/evt.mp4 \
    --vtk-dir hydro_pyvista_out/vti --field both --velocity

# C. Interactive (needs a display):
python hydro_pyvista.py --load /tmp/milne.npz --interactive
```

Checks:
1. `iter_events` prints `hydro bulk_info available` per event and a non-empty
   `(ntau,nx,ny,4)` array (reuse the metadata print from inspect_bulk_info).
2. The Cartesian frame at `t = tau_min` is a central z-band; at `t > tau_max` it is
   a hollow receding shell (sanity-confirms `tau = sqrt(t²−z²)` masking).
3. `/tmp/hydro.gif` / `.mp4` is written and shows the fireball growing then
   thinning along z; the `T=0.155` shell shrinks to nothing at freeze-out.
4. `.vti` files + `.pvd` open in ParaView as a time series with correct `t` values.
5. Energy-density and temperature ranges in the corner text match the bulk_info
   summary numbers.

## Notes / caveats
- Current MUSIC runs are 2+1D boost-invariant → the (τ,x,y) interpolator is exact;
  the script warns and falls back to the central η slice for any 3+1D input, with a
  documented loop-based extension point.
- `get_tz` exists in C++ but is unbound; replicating it in numpy avoids a rebuild.
  (Optional future cleanup: add a `get_tz`/`to_numpy_eta` binding for native 3+1D.)
- z extent is a display parameter for boost-invariant data (no intrinsic η range);
  `--z-max`/`--nz` control it.
