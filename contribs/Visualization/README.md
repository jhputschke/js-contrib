# Visualization

3D visualization of the X-SCAPE hydro medium evolution with
[PyVista](https://pyvista.org), built on the per-event Python workflow and the
`JetScapeSignalManager` interface (the same access path used by
`PyJetscape/python/jetscape/bulk_root_writer.py`).

The hydro is evolved in **Milne** coordinates `(τ, x, y, η_s)`, but this tool
resamples it into **Cartesian lab spacetime** `(t, x, y, z)` so that the next
phase — overlaying **parton-shower propagation**, which is naturally Cartesian —
shares one coordinate system with the medium.

## Files

- [`hydro_pyvista.py`](hydro_pyvista.py) — the visualizer (data acquisition →
  Milne→Cartesian resampling → PyVista rendering).
- [`PlanVisualization.md`](PlanVisualization.md) — the design plan.

## Environment

```bash
conda activate fno_pyvista_env      # pyvista, scipy, vtk, imageio, numpy
```

The compiled `pyjetscape_core` module must match this env's Python (built for
CPython 3.13). Live hydro runs execute from the X-SCAPE build directory
(default `<repo>/build_gpu`) so MUSIC can find `music_input`, `EOS/`, and the
tables — the script `chdir`'s there for you.

## Usage

```bash
cd external_packages/js-contrib/contribs/Visualization

# Live per-event run (BulkFastTest/OO_one_event.xml drives MUSIC with
# output_evolution_to_memory=1) → movie + ParaView time series:
python hydro_pyvista.py --events 1 \
    --movie evt.gif --vtk-dir vti

# Fast render-only iteration: dump the Milne array once (needs the build dir),
# then re-render from the dump (no MUSIC, no build dir):
python hydro_pyvista.py --events 1 --save-milne /tmp/milne.npz
python hydro_pyvista.py --load /tmp/milne.npz --movie /tmp/hydro.gif --field both

# Interactive window with a lab-time slider:
python hydro_pyvista.py --load /tmp/milne.npz --interactive
```

Key options: `--field {e,T,both}`, `--freeze-temp 0.155` (isosurface, repeatable),
`--velocity` (lab-frame flow arrows), `--nz/--nt/--z-max/--t-min/--t-max` (grid &
lab-time sampling), `--main/--user/--workdir` (run config), `--manual` (explicit
pipeline). At least one of `--movie`, `--vtk-dir`, `--interactive` is produced
(defaults to `evolution.gif`).

## How the Milne → Cartesian mapping works

For each lab time `t`, the script sweeps `z` and maps back to Milne, mirroring C++
`EvolutionHistory::get_tz()`:

```
τ      = sqrt(t² − z²)            # only inside the light cone, |z| < t
η_s    = 0.5 · ln((t + z)/(t − z))
fields = bulk(τ, x, y, η_s)      # 3+1D: full η dependence
```

Cells outside the light cone or outside the stored `τ ∈ [τ_min, τ_max]` are zero,
so a frame at `t = τ_min` is a central `z`-sheet and a frame at `t > τ_max` is a
hollow, receding shell — the medium expanding along the beam axis. Sampling uses a
single vectorized `scipy.interpolate.RegularGridInterpolator`.

### Boost-invariant vs 3+1D — both supported

The script auto-detects the data kind from `bulk.boost_invariant` / `neta`:

- **3+1D** (`boost_invariant=false`, `neta>1`, e.g. `OO_one_event.xml`): the full
  rapidity grid is pulled via the `EvolutionHistory.to_numpy_full(n_features)`
  binding (shape `(ntau, nx, ny, neta, nf)`, layout `[e, T, vx, vy, vz, s]`) and
  interpolated in **4D** `(τ, x, y, η_s)`, so different `z` correctly sample
  different rapidity slices. Velocity uses the stored lab-frame `(vx, vy, vz)`.
  `to_numpy_full` was added in
  `PyJetscape/src/bind_evolution.cc` (rebuild `pyjetscape_core` if you pull a
  fresh tree: `cmake --build build_gpu --target pyjetscape_core`).
- **2+1D / boost-invariant**: the fast `to_numpy()` transverse slice is exact;
  interpolation is **3D** `(τ, x, y)` and the lab-frame flow is synthesized
  (`v_z = z/t`, then `v_x, v_y /= γ_L`).

The full 3+1D grid can be ~GB in memory; `--save-milne` compresses it well (the
vacuum rapidity tails are mostly zero — a 643 MB grid → ~37 MB `.npz`).

## Next phase: parton-shower overlay

`render_event(..., overlay=callback)` accepts an `overlay(plotter, t)` callback
invoked at each lab time `t`. A future parton visualizer passes a callback that
adds `pyvista.PolyData` lines/points for the parton tracks at time `t` — they land
in the same Cartesian `(t, x, y, z)` frame as the medium volume. The `.vti`+`.pvd`
ParaView export reuses the identical grid, so the overlay can also be done in
ParaView.
```
