# Visualization

3D visualization of the X-SCAPE hydro medium evolution with
[PyVista](https://pyvista.org), built on the per-event Python workflow and the
`JetScapeSignalManager` interface (the same access path used by
`PyJetscape/python/jetscape/bulk_root_writer.py`).

The hydro is evolved in **Milne** coordinates `(τ, x, y, η_s)`, but this tool
resamples it into **Cartesian lab spacetime** `(t, x, y, z)`. The companion
[`hydro_jet_pyvista.py`](hydro_jet_pyvista.py) overlays the **jet parton shower**
in that same `(t,x,y,z)` frame.

## Files

- [`hydro_pyvista.py`](hydro_pyvista.py) — the medium visualizer (data acquisition →
  Milne→Cartesian resampling → PyVista rendering).
- [`hydro_jet_pyvista.py`](hydro_jet_pyvista.py) — medium **plus the jet parton
  shower** as accumulating arrows (see [Jet overlay](#jet-overlay)).
- [`config/`](config/) — bundled example MUSIC configs (`OO_one_event.xml`,
  `OO_one_event_jet.xml`).
- [`PlanVisualization.md`](PlanVisualization.md) — the design plan.

## Environment

```bash
conda activate fno_pyvista_env      # pyvista, scipy, vtk, imageio, numpy
pip install imageio-ffmpeg          # only needed for .mp4 output (.gif works without)
```

The compiled `pyjetscape_core` module must match this env's Python (built for
CPython 3.13). Live hydro runs execute from the X-SCAPE build directory
(default `<repo>/build_gpu`) so MUSIC can find `music_input`, `EOS/`, and the
tables — the script `chdir`'s there for you.

### Movie output & playback speed

`--movie out.gif` writes a GIF (always available); `--movie out.mp4` writes an MP4
(smaller, smoother — needs the `imageio-ffmpeg` package above, otherwise it falls
back to `.gif`). Control the speed with `--framerate` (frames/sec, default 6, applies
to both) or the more intuitive `--frame-duration SECONDS` (seconds each frame is
shown, e.g. `--frame-duration 0.5` for 2 fps to follow the evolution closely).
`--vtk-dir DIR` instead writes a `.vti`+`.pvd` time series for ParaView.

## Usage

```bash
cd external_packages/js-contrib/contribs/Visualization

# Live per-event run (the bundled config/OO_one_event.xml drives MUSIC with
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
`--velocity` (lab-frame flow arrows), `--nxy/--nz` (display grid resolution; the
Cartesian box is sampled finer than the hydro grid via interpolation, so higher =
smoother and slower — default 96³), `--nt/--z-max/--t-min/--t-max` (lab-time
sampling & z extent), `--xy-max` (transverse half-extent in fm; default auto-crops
to the fireball + 2 fm so it isn't lost in the ~30 fm hydro grid), `--cmap`,
`--main/--user/--workdir` (run config), `--manual` (explicit pipeline). At least one
of `--movie`, `--vtk-dir`, `--interactive` is produced (defaults to `evolution.gif`).

Smoothness note: the render resolution is decoupled from the hydro grid. The Milne→
Cartesian resampling is linear, so increasing `--nxy/--nz` gives a smoother volume
without changing the physics; lower them (e.g. `--nxy 48 --nz 48`) for quick previews.

Longitudinal anti-aliasing (`--z-oversample`, default 3): the medium begins at
`τ_min` — a razor-thin, very hot surface — and in the Cartesian lab frame that sharp
`τ = √(t²−z²) = τ_min` surface sweeps the discrete z-grid, so a plain uniform sampling
makes the captured peak **flicker** frame-to-frame (it aliases as the surface falls on
vs between grid points). The fix is **non-uniform z sampling**: extra z-samples that
are *uniform in τ* — hence dense exactly at the τ_min light-cone edge and always
including τ_min itself — are added and bin-pooled onto the z-grid (max for the
intensities e, T; mean for the velocity), so the hot edge is captured consistently
every frame. This cut the hot-phase peak's variation from ~11 % to ~0.3 % in testing
(full dynamic range kept), at ~3–5× the resampling cost; set `--z-oversample 1` to
disable. (Plain uniform supersampling or capping the colour scale do **not** fix it —
`e(τ)` is intrinsically steep near `τ_min`, so only resolving that surface works.)

Performance: the Milne→Cartesian resampling is the dominant cost and is parallelised
across lab-time frames with threads (`--jobs`, default `min(cores, 8)`; scipy releases
the GIL so it scales ~linearly — e.g. 50 frames at 96³ drop from ~15 s to ~3 s). The
GPU volume rendering is sequential. Each run prints a `resampled in … s` / `rendered …
frames in … s` breakdown so you can see where time goes and tune `--nxy/--nz/--nt/--jobs`.

The scene uses a dark-grey gradient background, an x/y/z orientation triad, a
labelled bounding box in fm, an upper-left event/lab-time read-out, and an
energy-density colour bar in `GeV/fm³` (held at a fixed scale across the animation).

Camera: the heavy-ion convention — beam axis **z runs left↔right** (horizontal),
with the transverse **x–y** plane tilted toward the viewer so its evolution is
visible. Tune with `--azimuth` (default 35°, tilts in the transverse x) and
`--elevation` (default 20°, looks down on the x–y plane).

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

## Jet overlay

[`hydro_jet_pyvista.py`](hydro_jet_pyvista.py) renders the medium **and** the jet
parton shower together. It reuses everything in `hydro_pyvista.py` via the
`render_event(..., overlay=callback)` hook — the medium is identical; the overlay
adds the partons as arrows in the same Cartesian `(t,x,y,z)` frame, accumulating
over lab time (earlier partons are not removed, so the shower tree builds up).

```bash
# Live: run MUSIC + jets (config/OO_one_event_jet.xml) and render medium + shower
python hydro_jet_pyvista.py --events 1 --movie evt_jet.gif

# Non-live: medium from a dump + shower from the ASCII writer output
python hydro_jet_pyvista.py --load hydro.npz --jet-ascii test_out.dat --movie evt_jet.gif
```

How the shower is obtained:

- **Live (default):** read from the framework via pybind11 bindings at the
  per-event yield point — the same place the hydro is read, no file parsing:
  ```python
  jm = JetScapeSignalManager.Instance().GetJetEnergyLossManagerPointer()
  for ps in jm.get_showers():        # one PartonShower per shower-initiating parton
      edges = ps.to_numpy()          # (n_partons, 12)
      # [source_id, target_id, pid, pstat, px, py, pz, E, x, y, z, t]
  ```
  These bindings (`Parton`, `Vertex`, `PartonShower`, `JetEnergyLossManager`,
  `JetScapeSignalManager.GetJetEnergyLossManagerPointer`) live in
  `PyJetscape/src/bind_jet.cc`; rebuild `pyjetscape_core` after pulling. The full
  shower is only complete *after* the `JetEnergyLossManager` has executed (its
  child `JetEnergyLoss` copies each hold a finished shower).
- **Non-live (`--load`):** the shower is parsed from the `JetScapeWriterAscii`
  output (`--jet-ascii`, default `<workdir>/test_out.dat`). This path also serves
  to verify the bindings (it produces the identical graph).

Each parton is a graph edge between two space-time vertices; only those endpoints
are recorded, so a parton's position at an intermediate lab time `t` is the
straight-line (constant-velocity) interpolation between its start and end vertex.

Partons are colour-coded by **pT** (default colormap `cool`, contrasting the
inferno medium) on a fixed scale whose maximum is the hardest parton's pT — shown
as a second colour bar (left), separate from the energy-density bar (right).

Jet options: `--jet-radius`/`--jet-width` (arrow width in fm, default 0.04),
`--jet-cmap` (pT colormap), `--jet-color` (one fixed colour instead of the pT
scale), `--jet-min-energy` (drop soft partons), `--jet-ascii`. By default the
animation runs from **t=0** (when the jets are born at the hard vertex) to the
medium lifetime `τ_max`: the jets first evolve **in vacuum**, then the medium
appears once it forms (`t ≥ τ₀`, below which the hydro resampler returns an empty
volume).

The **medium box matches the medium-only script exactly** — it is *not* widened
for the jets, so the medium evolution renders identically (widening `z` would
otherwise reveal the early-proper-time medium near the light-cone edge at large
`|z|`, and the last frame would not look cooled). The jets are instead kept in
frame by an invisible camera anchor and may extend past the box. Use `--t-max` to
follow the jets further out (the forward, high-energy partons escape to large
`z`), `--t-min` to start later, or `--z-max`/`--xy-max` to enlarge the box.
```
