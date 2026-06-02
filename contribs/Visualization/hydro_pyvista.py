"""
contribs/Visualization/hydro_pyvista.py

3D PyVista visualization of the X-SCAPE hydro medium evolution, resampled from
Milne (tau, x, y, eta_s) coordinates into Cartesian lab spacetime (t, x, y, z).

Why Cartesian?  The hydro is evolved on a constant proper-time grid, but parton
showers (the next visualization phase) propagate in Cartesian (t, x, y, z).  This
script resamples each event's evolution history onto a Cartesian (x, y, z) volume
at a sequence of *lab times* t, so the medium and — later — parton trajectories
share one coordinate system.  The transform mirrors C++ EvolutionHistory::get_tz()
(src/framework/FluidEvolutionHistory.cc):

    tau = sqrt(t^2 - z^2),   eta_s = 0.5 * ln((t + z) / (t - z))

and, for boost-invariant data, boosts the transverse flow to the lab frame
(vz = z/t, then vx, vy /= gamma_L).

Data is pulled live from the per-event Python workflow exactly like
example/per_event_loop.py and python/jetscape/bulk_root_writer.py:

    hydro = JetScapeSignalManager.Instance().GetHydroPointer()
    bulk  = hydro.get_bulk_info()                 # EvolutionHistory
    # 3+1D (boost_invariant=false, neta>1):
    arr   = bulk.to_numpy_full(nf)                # (ntau, nx, ny, neta, nf)
                                                  # [e, T, vx, vy, vz, s]
    # 2+1D / boost-invariant:
    arr   = bulk_info_to_numpy(bulk, nf)          # (ntau, nx, ny, nf)
                                                  # [e, T, vx, vy]

3+1D data is interpolated in 4D (tau, x, y, eta_s); boost-invariant data in 3D
(tau, x, y).  The kind is auto-detected from bulk.boost_invariant / neta.

Environment
-----------
    conda activate fno_pyvista_env

Run the hydro from the X-SCAPE build directory (default: <repo>/build_gpu) so
MUSIC finds music_input / EOS / tables.  The script chdir's there for you.

Usage
-----
  # Live per-event run (OO_one_event.xml drives MUSIC with evolution-in-memory),
  # write a movie + a ParaView time series:
  python hydro_pyvista.py --events 1 \
      --movie hydro_pyvista_out/evt.gif --vtk-dir hydro_pyvista_out/vti

  # Fast iteration on the rendering only: dump once, re-render from the dump
  # (no MUSIC / no build dir needed):
  python hydro_pyvista.py --events 1 --save-milne /tmp/milne.npz
  python hydro_pyvista.py --load /tmp/milne.npz --movie /tmp/hydro.gif

  # Interactive window with a time slider:
  python hydro_pyvista.py --load /tmp/milne.npz --interactive

Next phase (parton overlay)
---------------------------
render_event() accepts an ``overlay`` callback ``overlay(plotter, t)`` invoked for
each lab time t.  A future parton-shower visualizer passes a callback that adds
pyvista.PolyData lines/points for the parton tracks at time t — they land in the
same Cartesian frame as the medium.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# Suppress duplicate-OpenMP abort on macOS (vtk/torch may both ship libomp).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ── Make the `jetscape` package importable (it lives in a sibling contrib) ─────
_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))            # …/contribs/Visualization
_CONTRIBS   = os.path.dirname(_THIS_DIR)                            # …/contribs
_PYJETSCAPE = os.path.join(_CONTRIBS, "PyJetscape")
_REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.dirname(_CONTRIBS)))  # …/X-SCAPE
sys.path.insert(0, os.path.join(_PYJETSCAPE, "python"))

# Feature layout returned by bulk_info_to_numpy / EvolutionHistory.to_numpy.
FEAT_E, FEAT_T, FEAT_VX, FEAT_VY = 0, 1, 2, 3

# Scene aesthetics.
DARK_GREY  = (0.16, 0.17, 0.19)   # plot background
PANEL_GREY = (0.10, 0.11, 0.13)   # bottom of the gradient background
E_UNITS    = "energy density  [GeV/fm³]"
# Opacity transfer function for the energy-density volume: ramps up quickly so the
# warm/dilute medium stays visible (plain "sigmoid" makes most cells transparent).
VOLUME_OPACITY = [0.0, 0.22, 0.45, 0.65, 0.82, 0.95]


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ── Source ────────────────────────────────────────────────────────────────
    p.add_argument("--main",
                   default=os.path.join(_REPO_ROOT, "config", "jetscape_main.xml"),
                   help="Main XML config (default: <repo>/config/jetscape_main.xml).")
    p.add_argument("--user",
                   default=os.path.join(_REPO_ROOT, "config", "BulkFastTest",
                                        "OO_one_event.xml"),
                   help="User XML config (default: BulkFastTest/OO_one_event.xml, "
                        "which runs MUSIC with output_evolution_to_memory=1).")
    p.add_argument("--workdir",
                   default=os.path.join(_REPO_ROOT, "build_gpu"),
                   help="Directory to run from so MUSIC finds music_input/EOS/tables "
                        "(default: <repo>/build_gpu). Ignored with --load.")
    p.add_argument("--events", type=int, default=1,
                   help="Number of events to process (default: 1).")
    p.add_argument("--start-event", type=int, default=0, dest="start_event")
    p.add_argument("--load", default=None,
                   help="Render from a .npz dumped by --save-milne (skips the run).")
    p.add_argument("--save-milne", default=None, dest="save_milne",
                   help="Dump the Milne (ntau,nx,ny,4) array + grid metadata to .npz.")
    # ── Manual (Mode B) pipeline ──────────────────────────────────────────────
    p.add_argument("--manual", action="store_true",
                   help="Build the pipeline explicitly instead of from the XML task "
                        "list (user XML must set enableAutomaticTaskListDetermination "
                        "= false).")
    p.add_argument("--initial-state", default="TrentoInitial", dest="initial_state")
    p.add_argument("--preequilibrium", default="NullPreDynamics")
    p.add_argument("--hydro-module", default="MUSIC", dest="hydro_module")
    # ── Cartesian grid / lab-time sampling ────────────────────────────────────
    p.add_argument("--nxy", type=int, default=96,
                   help="Transverse (x,y) display grid points per axis (default 96). "
                        "Sampled finer than the hydro grid via interpolation, so "
                        "higher = smoother (and slower).")
    p.add_argument("--nz", type=int, default=96, help="z grid points (default 96).")
    p.add_argument("--z-max", type=float, default=None, dest="z_max",
                   help="z half-extent in fm (default 0.8*t_max).")
    p.add_argument("--xy-max", type=float, default=None, dest="xy_max",
                   help="Transverse half-extent in fm to display "
                        "(default: auto-crop to the medium + 2 fm margin).")
    p.add_argument("--nt", type=int, default=None,
                   help="Number of lab-time frames (default = ntau).")
    p.add_argument("--t-min", type=float, default=None, dest="t_min",
                   help="First lab time in fm/c (default = tau_min).")
    p.add_argument("--t-max", type=float, default=None, dest="t_max",
                   help="Last lab time in fm/c (default = tau_max).")
    # ── Rendering ─────────────────────────────────────────────────────────────
    p.add_argument("--field", choices=["e", "T", "both"], default="both",
                   help="e: energy-density volume; T: temperature isosurface; "
                        "both (default).")
    p.add_argument("--freeze-temp", type=float, nargs="+", default=[0.155],
                   dest="freeze_temp",
                   help="Temperature isosurface value(s) in GeV (default 0.155).")
    p.add_argument("--cmap", default="inferno", help="Colormap (default inferno).")
    p.add_argument("--velocity", action="store_true",
                   help="Add lab-frame flow-velocity arrow glyphs.")
    p.add_argument("--azimuth", type=float, default=35.0,
                   help="Camera azimuth in deg from the beam-side view (default 35). "
                        "Beam axis z is horizontal; azimuth tilts in the transverse x.")
    p.add_argument("--elevation", type=float, default=20.0,
                   help="Camera elevation in deg (default 20); looks down on the x–y plane.")
    # ── Output (at least one required) ────────────────────────────────────────
    p.add_argument("--movie", default=None,
                   help="Animation path; .gif (default-friendly) or .mp4.")
    p.add_argument("--vtk-dir", default=None, dest="vtk_dir",
                   help="Write one .vti per frame + a .pvd collection for ParaView.")
    p.add_argument("--interactive", action="store_true",
                   help="Open a live window with a time-frame slider.")
    p.add_argument("--outdir", default="hydro_pyvista_out",
                   help="Base output directory (default hydro_pyvista_out/).")
    p.add_argument("--framerate", type=int, default=10,
                   help="Movie frames per second (default 10).")
    p.add_argument("--jobs", "-j", type=int, default=min(os.cpu_count() or 1, 8),
                   help="Threads for the Milne→Cartesian resampling (the dominant "
                        "cost). scipy releases the GIL so threads scale ~linearly; "
                        f"default min(cores,8). Use 1 to disable.")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Data acquisition
# ──────────────────────────────────────────────────────────────────────────────

def _read_meta(bulk) -> dict:
    """Grid metadata from an EvolutionHistory (all def_readwrite bound)."""
    return {
        "tau_min": float(bulk.tau_min), "dtau": float(bulk.dtau),
        "x_min":   float(bulk.x_min),   "dx":   float(bulk.dx),
        "y_min":   float(bulk.y_min),   "dy":   float(bulk.dy),
        "eta_min": float(bulk.eta_min), "deta": float(bulk.deta),
        "ntau": int(bulk.ntau), "nx": int(bulk.nx),
        "ny":   int(bulk.ny),   "neta": int(bulk.neta),
        "boost_invariant": bool(bulk.boost_invariant),
    }


def _build_manual_pipeline(args) -> list:
    """Explicit [initial-state, pre-equilibrium, hydro] pipeline (Mode B)."""
    from jetscape import create_module
    ini   = create_module(args.initial_state)
    preeq = create_module(args.preequilibrium)
    hydro = create_module(args.hydro_module)
    return [ini, preeq, hydro]


def iter_events(args):
    """Yield (event_id, milne_arr, meta) for each event of a per-event run.

    Mirrors example/per_event_loop.py: the hydro module is read at the yield
    point (after ExecPerEvent, before ClearPerEvent) so the evolution history is
    still live and no set_preserve_bulk_info() is needed.
    """
    from jetscape import JetScapeSignalManager
    from jetscape.run_jetscape import per_event_loop
    from jetscape.utils import bulk_info_to_numpy

    modules = _build_manual_pipeline(args) if args.manual else None
    if modules is not None:
        print("Manual pipeline: " + " -> ".join(m.GetId() for m in modules))

    for js in per_event_loop(args.main, args.user, modules=modules,
                             n_events=args.events, start_event=args.start_event):
        event_id = js.GetCurrentEvent()
        hydro = JetScapeSignalManager.Instance().GetHydroPointer()
        if hydro is None:
            print(f"[event {event_id}] no hydro module registered — skipping.")
            continue
        bulk = hydro.get_bulk_info()
        meta = _read_meta(bulk)
        # 3+1D data needs the full eta axis (different z map to different eta_s);
        # to_numpy() only exposes id_eta=0.  Boost-invariant data is exact with
        # the fast (tau,x,y) transverse slice.
        full_3d = (not meta["boost_invariant"]) and meta["neta"] > 1
        try:
            if full_3d:
                nf = 5 if args.velocity else 2        # [e, T, vx, vy, vz]
                arr = bulk.to_numpy_full(nf)          # (ntau, nx, ny, neta, nf)
            else:
                nf = 4 if args.velocity else 2        # [e, T, vx, vy]
                arr = bulk_info_to_numpy(bulk, nf)    # (ntau, nx, ny, nf)
        except Exception as exc:                       # empty data, etc.
            print(f"[event {event_id}] bulk_info unavailable ({exc}); make sure the "
                  "hydro stores evolution in memory (output_evolution_to_memory=1).")
            continue
        meta["has_eta"] = (arr.ndim == 5)

        tau_max = meta["tau_min"] + (meta["ntau"] - 1) * meta["dtau"]
        kind = "3+1D full-eta" if meta["has_eta"] else "2+1D / boost-invariant"
        print(f"[event {event_id}] bulk_info {arr.shape}  [{kind}]  "
              f"tau in [{meta['tau_min']:.3f}, {tau_max:.3f}] fm/c  "
              f"x in [{meta['x_min']:.1f}, {meta['x_min']+(meta['nx']-1)*meta['dx']:.1f}] fm  "
              f"eta in [{meta['eta_min']:.1f}, {meta['eta_min']+(meta['neta']-1)*meta['deta']:.1f}]")

        if args.save_milne:
            _save_milne(args.save_milne, arr, meta, event_id, args.events)
        yield event_id, arr, meta


def _save_milne(path: str, arr: np.ndarray, meta: dict,
                event_id: int, n_events: int) -> None:
    base, ext = os.path.splitext(path)
    out = path if ext == ".npz" else base + ".npz"
    if n_events > 1:                                  # keep per-event dumps distinct
        out = f"{os.path.splitext(out)[0]}_evt{event_id}.npz"
    np.savez_compressed(out, arr=arr.astype(np.float32),
                        meta=np.array([json.dumps(meta)]))
    print(f"  saved Milne dump -> {out}")


def load_milne(path: str):
    """Yield a single (0, milne_arr, meta) from a --save-milne dump."""
    d = np.load(path, allow_pickle=True)
    arr  = d["arr"]
    meta = json.loads(str(d["meta"][0]))
    meta["has_eta"] = (arr.ndim == 5)
    kind = "3+1D full-eta" if meta["has_eta"] else "2+1D / boost-invariant"
    print(f"Loaded {path}: arr {arr.shape} [{kind}]")
    yield 0, arr, meta


# ──────────────────────────────────────────────────────────────────────────────
# Milne -> Cartesian resampling (vectorized)
# ──────────────────────────────────────────────────────────────────────────────

def build_interpolator(arr: np.ndarray, meta: dict):
    """RegularGridInterpolator over (tau, x, y[, eta]); the feature axis rides along.

    4-D ``arr`` (ntau, nx, ny, nf) → interpolate over (tau, x, y) [boost-invariant].
    5-D ``arr`` (ntau, nx, ny, neta, nf) → interpolate over (tau, x, y, eta) [3+1D].
    The array is kept float32 (the 3+1D grid can be ~GB; do not upcast).
    """
    from scipy.interpolate import RegularGridInterpolator
    taus = meta["tau_min"] + meta["dtau"] * np.arange(arr.shape[0])
    xs   = meta["x_min"]   + meta["dx"]   * np.arange(meta["nx"])
    ys   = meta["y_min"]   + meta["dy"]   * np.arange(meta["ny"])
    if arr.ndim == 5:
        etas = meta["eta_min"] + meta["deta"] * np.arange(arr.shape[3])
        grid_axes = (taus, xs, ys, etas)
    else:
        grid_axes = (taus, xs, ys)
    return RegularGridInterpolator(
        grid_axes, arr, method="linear", bounds_error=False, fill_value=0.0)


def medium_xy_max(arr: np.ndarray, meta: dict, frac: float = 0.02,
                  margin: float = 2.0) -> float:
    """Transverse half-extent (fm) covering the energy-bearing region + margin.

    The OO/AA fireball is only a few fm across while the hydro grid spans ~30 fm,
    so showing the whole grid buries the medium.  This finds the |x|,|y| where the
    energy density exceeds ``frac`` of its peak and adds a margin.
    """
    e = arr[..., FEAT_E]                                   # (ntau,nx,ny[,neta])
    other = tuple(i for i in range(e.ndim) if i not in (1, 2))
    e_xy = e.max(axis=other)                               # (nx, ny)
    if e_xy.max() <= 0:
        return min(abs(meta["x_min"]), abs(meta["y_min"]))
    xi, yi = np.where(e_xy > frac * e_xy.max())
    xs = meta["x_min"] + meta["dx"] * np.arange(meta["nx"])
    ys = meta["y_min"] + meta["dy"] * np.arange(meta["ny"])
    reach = max(np.abs(xs[xi]).max(), np.abs(ys[yi]).max()) + margin
    grid_reach = min(abs(meta["x_min"]), meta["x_min"] + (meta["nx"] - 1) * meta["dx"])
    return float(min(reach, abs(grid_reach)))


def cartesian_axes(meta: dict, args, t_max: float, xy_max=None):
    """Cartesian sampling axes, with display resolution decoupled from the hydro grid.

    The transverse axes use args.nxy points (and z uses args.nz) so the volume can
    be sampled finer than the hydro spacing — the Milne→Cartesian interpolation is
    linear, so a finer display grid renders smoothly instead of blocky.
    """
    if xy_max is None:
        x_lo, x_hi = meta["x_min"], meta["x_min"] + (meta["nx"] - 1) * meta["dx"]
        y_lo, y_hi = meta["y_min"], meta["y_min"] + (meta["ny"] - 1) * meta["dy"]
    else:
        x_lo, x_hi = -xy_max, xy_max
        y_lo, y_hi = -xy_max, xy_max
    xs = np.linspace(x_lo, x_hi, args.nxy)
    ys = np.linspace(y_lo, y_hi, args.nxy)
    z_max = args.z_max if args.z_max is not None else 0.8 * t_max
    zs = np.linspace(-z_max, z_max, args.nz)
    return xs, ys, zs


def cartesian_frame(interp, t: float, axes, meta: dict, want_velocity: bool) -> dict:
    """Resample the Milne fields onto the Cartesian (x, y, z) box at lab time t.

    Returns dict of (nx, ny, nz) float32 arrays: {"e", "T"[, "v"(...,3)]}.
    Cells outside the light cone (|z| >= t) or outside [tau_min, tau_max] are 0.

    For 3+1D data (meta["has_eta"]) the query carries eta_s = 0.5 ln((t+z)/(t-z))
    so different z sample different rapidity slices; velocity is read directly.
    For boost-invariant data the field is eta-independent and the transverse flow
    is boosted to the lab frame (vz = z/t, vx,vy /= gamma_L), mirroring get_tz().
    """
    xs, ys, zs = axes
    nx, ny, nz = len(xs), len(ys), len(zs)
    has_eta = bool(meta.get("has_eta", False))
    tau_min = meta["tau_min"]
    tau_max = meta["tau_min"] + (meta["ntau"] - 1) * meta["dtau"]

    # tau, eta depend on z only (constant across the x-y plane at fixed t).
    inside = t * t > zs * zs
    tau_z = np.zeros(nz)
    tau_z[inside] = np.sqrt(t * t - zs[inside] ** 2)

    TAU = np.broadcast_to(tau_z[:, None, None], (nz, nx, ny))
    X   = np.broadcast_to(xs[None, :, None],    (nz, nx, ny))
    Y   = np.broadcast_to(ys[None, None, :],    (nz, nx, ny))
    if has_eta:
        eta_z = np.zeros(nz)
        zin = zs[inside]
        eta_z[inside] = 0.5 * np.log((t + zin) / (t - zin))
        ETA = np.broadcast_to(eta_z[:, None, None], (nz, nx, ny))
        pts = np.stack([TAU, X, Y, ETA], axis=-1).reshape(-1, 4)
    else:
        pts = np.stack([TAU, X, Y], axis=-1).reshape(-1, 3)
    vals = interp(pts).reshape(nz, nx, ny, -1)          # (nz,nx,ny,nf)
    vals = np.transpose(vals, (1, 2, 0, 3))             # (nx,ny,nz,nf)

    valid = inside & (tau_z >= tau_min) & (tau_z <= tau_max)   # (nz,)
    mask = valid[None, None, :]                                # broadcast to (nx,ny,nz)

    fields = {
        "e": np.ascontiguousarray(vals[..., FEAT_E] * mask, dtype=np.float32),
        "T": np.ascontiguousarray(vals[..., FEAT_T] * mask, dtype=np.float32),
    }
    if want_velocity:
        nf = vals.shape[-1]
        if has_eta and nf >= 5:
            # 3+1D: vx, vy, vz are the stored lab-frame flow components.
            vx = vals[..., 2] * mask
            vy = vals[..., 3] * mask
            vz = vals[..., 4] * mask
            fields["v"] = np.ascontiguousarray(
                np.stack([vx, vy, vz], axis=-1), dtype=np.float32)
        elif (not has_eta) and nf > FEAT_VY:
            # Boost-invariant: synthesize vz and boost transverse flow (get_tz).
            vx = vals[..., FEAT_VX]
            vy = vals[..., FEAT_VY]
            vz = np.broadcast_to(zs[None, None, :] / t, (nx, ny, nz))
            gammaL = 1.0 / np.sqrt(np.clip(1.0 - vz ** 2, 1e-6, None))
            vx = (vx / gammaL) * mask
            vy = (vy / gammaL) * mask
            vz = vz * mask
            fields["v"] = np.ascontiguousarray(
                np.stack([vx, vy, vz], axis=-1), dtype=np.float32)
    return fields


# ──────────────────────────────────────────────────────────────────────────────
# PyVista rendering
# ──────────────────────────────────────────────────────────────────────────────

def resample_frames(interp, ts, axes, meta, want_velocity, jobs: int):
    """Resample all lab-time frames, in parallel across threads.

    Frames are independent and the dominant cost is the scipy interpolation,
    which releases the GIL — so a thread pool scales nearly linearly with no
    array copying (the big Milne grid is shared, not pickled).
    """
    def one(t):
        return cartesian_frame(interp, float(t), axes, meta, want_velocity)
    if jobs <= 1 or len(ts) <= 1:
        return [one(t) for t in ts]
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        return list(ex.map(one, ts))      # ex.map preserves input order


def make_image_data(fields: dict, axes):
    """Wrap the Cartesian fields in a pyvista.ImageData (VTK Fortran ordering)."""
    import pyvista as pv
    xs, ys, zs = axes
    nx, ny, nz = len(xs), len(ys), len(zs)
    grid = pv.ImageData()
    grid.dimensions = (nx, ny, nz)
    grid.spacing = (float(xs[1] - xs[0]) if nx > 1 else 1.0,
                    float(ys[1] - ys[0]) if ny > 1 else 1.0,
                    float(zs[1] - zs[0]) if nz > 1 else 1.0)
    grid.origin = (float(xs[0]), float(ys[0]), float(zs[0]))
    grid.point_data["e"] = fields["e"].ravel(order="F")
    grid.point_data["T"] = fields["T"].ravel(order="F")
    if "v" in fields:
        v = fields["v"]
        grid.point_data["v"] = np.column_stack([
            v[..., 0].ravel(order="F"),
            v[..., 1].ravel(order="F"),
            v[..., 2].ravel(order="F"),
        ])
    return grid


def _maybe_start_xvfb(off_screen: bool) -> None:
    """On headless Linux (e.g. GB10) start a virtual framebuffer for VTK."""
    if off_screen and sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        try:
            import pyvista as pv
            pv.start_xvfb()
        except Exception as exc:
            print(f"  [!] start_xvfb failed ({exc}); off-screen render may fail.")


def _scene_bounds(axes):
    """(xmin, xmax, ymin, ymax, zmin, zmax) of the Cartesian sampling box [fm]."""
    xs, ys, zs = axes
    return (float(xs[0]), float(xs[-1]),
            float(ys[0]), float(ys[-1]),
            float(zs[0]), float(zs[-1]))


def _decorate_scene(plotter, bounds) -> None:
    """Background, orientation triad, and a labelled x/y/z bounding box (fm).

    Called once per frame (after plotter.clear()).  ``bounds`` is passed
    explicitly so the labelled box is stable even on near-empty frames.
    """
    plotter.set_background(DARK_GREY, top=PANEL_GREY)
    # Corner orientation triad (x=red, y=green, z=blue arrows with labels).
    plotter.add_axes(xlabel="x", ylabel="y", zlabel="z",
                     line_width=3, labels_off=False, color="white")
    # Labelled bounding box with physical units.
    plotter.show_grid(
        bounds=bounds,
        xtitle="x  [fm]", ytitle="y  [fm]", ztitle="z  [fm]",
        color="white", grid="back", location="outer", ticks="both",
        font_size=10,
    )


def _beam_camera(plotter, azimuth: float, elevation: float) -> None:
    """Heavy-ion convention: beam axis z runs left↔right (horizontal), the
    transverse x–y plane is tilted toward the viewer so its evolution is visible.

    Starts from a side view (look along +x, y up → z horizontal, y vertical) and
    rotates by ``azimuth`` (reveals the transverse x depth) and ``elevation``
    (look down on the x–y plane).  reset_camera() then fits the box, preserving
    the orientation.
    """
    plotter.camera_position = [(1.0, 0.0, 0.0),    # position (refit below)
                               (0.0, 0.0, 0.0),    # focal point
                               (0.0, 1.0, 0.0)]    # view up = +y
    plotter.camera.azimuth = azimuth
    plotter.camera.elevation = elevation
    plotter.reset_camera()


def _add_frame_actors(plotter, grid, args, clim, label: str, overlay, t: float):
    """Add the volume / isosurface / glyph actors for one frame (named for reuse)."""
    import pyvista as pv  # noqa: F401  (ensures pyvista is importable here)
    if args.field in ("e", "both") and clim[1] > 0:
        plotter.add_volume(
            grid, scalars="e", cmap=args.cmap, opacity=VOLUME_OPACITY, clim=clim,
            name="vol", reset_camera=False,
            scalar_bar_args=dict(
                title=E_UNITS, color="white",
                title_font_size=16, label_font_size=13, n_labels=5,
                fmt="%.2f", vertical=True,
                position_x=0.88, position_y=0.12, width=0.05, height=0.6,
            ),
        )
    if args.field in ("T", "both"):
        contour = grid.contour(args.freeze_temp, scalars="T")
        if contour.n_points:
            plotter.add_mesh(contour, color="deepskyblue", opacity=0.30, name="cont",
                             smooth_shading=True, reset_camera=False,
                             show_scalar_bar=False)
    if args.velocity and "v" in grid.point_data:
        active = grid.threshold(max(1e-3, 0.02 * clim[1]), scalars="e")
        if active.n_points:
            active.set_active_vectors("v")
            glyphs = active.glyph(orient="v", scale=False, factor=0.6, tolerance=0.04)
            plotter.add_mesh(glyphs, color="white", opacity=0.7, name="vel",
                             reset_camera=False, show_scalar_bar=False)
    # Time / event read-out, upper-left.
    plotter.add_text(label, name="label", position="upper_left",
                     font_size=14, color="white", shadow=True)
    if overlay is not None:
        overlay(plotter, t)


def _frame_label(event_id, t: float) -> str:
    return f"event {event_id}\nt = {t:6.2f} fm/c"


def render_event(event_id, arr, meta, args, overlay=None) -> None:
    """Build per-lab-time Cartesian frames and emit movie / VTK / interactive."""
    import pyvista as pv

    tau_max = meta["tau_min"] + (meta["ntau"] - 1) * meta["dtau"]
    t_min = args.t_min if args.t_min is not None else meta["tau_min"]
    t_max = args.t_max if args.t_max is not None else tau_max
    nt    = args.nt if args.nt is not None else min(meta["ntau"], 50)
    ts    = np.linspace(t_min, t_max, max(1, nt))

    xy_max = args.xy_max if args.xy_max is not None else medium_xy_max(arr, meta)
    axes   = cartesian_axes(meta, args, t_max, xy_max)
    interp = build_interpolator(arr, meta)

    jobs = max(1, min(args.jobs, len(ts)))
    print(f"  resampling {len(ts)} lab-time frames onto a "
          f"({len(axes[0])}, {len(axes[1])}, {args.nz}) Cartesian grid "
          f"(x,y in ±{xy_max:.1f} fm, t in [{t_min:.2f}, {t_max:.2f}] fm/c) "
          f"using {jobs} thread(s) ...")
    t0 = time.perf_counter()
    frames = resample_frames(interp, ts, axes, meta, args.velocity, jobs)
    print(f"  resampled in {time.perf_counter() - t0:.1f} s")
    e_max = max((f["e"].max() for f in frames), default=0.0)
    clim  = (0.0, float(e_max) if e_max > 0 else 1.0)
    print(f"  energy-density max = {e_max:.4g} GeV/fm^3")

    os.makedirs(args.outdir, exist_ok=True)

    # ── VTK time series (.vti + .pvd) ─────────────────────────────────────────
    if args.vtk_dir:
        vdir = args.vtk_dir if os.path.isabs(args.vtk_dir) \
            else os.path.join(args.outdir, args.vtk_dir)
        os.makedirs(vdir, exist_ok=True)
        entries = []
        for fi, (t, fields) in enumerate(zip(ts, frames)):
            grid = make_image_data(fields, axes)
            fn = f"event{event_id}_t{fi:03d}.vti"
            grid.save(os.path.join(vdir, fn))
            entries.append((float(t), fn))
        _write_pvd(os.path.join(vdir, f"event{event_id}.pvd"), entries)
        print(f"  wrote {len(entries)} .vti + event{event_id}.pvd -> {vdir}")

    # ── Movie ─────────────────────────────────────────────────────────────────
    if args.movie:
        movie = args.movie if os.path.isabs(args.movie) \
            else os.path.join(args.outdir, args.movie)
        _maybe_start_xvfb(off_screen=True)
        plotter = pv.Plotter(off_screen=True, window_size=(1000, 800))
        if movie.lower().endswith(".mp4"):
            try:
                plotter.open_movie(movie, framerate=args.framerate)
            except Exception as exc:
                movie = os.path.splitext(movie)[0] + ".gif"
                print(f"  [!] mp4 unavailable ({exc}); falling back to {movie}")
                plotter.open_gif(movie)
        else:
            plotter.open_gif(movie)
        scene_bounds = _scene_bounds(axes)
        cam = None
        t0 = time.perf_counter()
        for fi, (t, fields) in enumerate(zip(ts, frames)):
            grid = make_image_data(fields, axes)
            plotter.clear()
            _decorate_scene(plotter, scene_bounds)
            _add_frame_actors(plotter, grid, args, clim,
                              _frame_label(event_id, t), overlay, float(t))
            if cam is None:
                _beam_camera(plotter, args.azimuth, args.elevation)
                cam = plotter.camera_position
            else:
                plotter.camera_position = cam
            plotter.write_frame()
        plotter.close()
        print(f"  wrote movie -> {movie}  "
              f"(rendered {len(ts)} frames in {time.perf_counter() - t0:.1f} s)")

    # ── Interactive (time slider) ─────────────────────────────────────────────
    if args.interactive:
        _show_interactive(frames, axes, ts, args, event_id, clim, overlay)


def _write_pvd(pvd_path: str, entries) -> None:
    lines = ['<?xml version="1.0"?>',
             '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
             '  <Collection>']
    for t, fn in entries:
        lines.append(f'    <DataSet timestep="{t}" group="" part="0" file="{fn}"/>')
    lines += ['  </Collection>', '</VTKFile>', '']
    with open(pvd_path, "w") as fh:
        fh.write("\n".join(lines))


def _clear_frame_actors(plotter) -> None:
    """Remove only the per-frame actors (and the energy colour bar), leaving the
    static scene (background, axes, box) and any interactor widgets intact.

    NOTE: do NOT call plotter.clear() inside a slider callback — clearing the
    widget's own representation crashes the VTK interactor (segfault).
    """
    for name in ("vol", "cont", "vel"):
        plotter.remove_actor(name, reset_camera=False)
    try:
        plotter.remove_scalar_bar(title=E_UNITS, render=False)
    except Exception:
        pass  # no energy bar present yet (e.g. --field T)


def _show_interactive(frames, axes, ts, args, event_id, clim, overlay) -> None:
    """On-screen viewer with a lab-time slider (static scene set up once)."""
    import pyvista as pv
    grids  = [make_image_data(f, axes) for f in frames]
    bounds = _scene_bounds(axes)
    plotter = pv.Plotter(window_size=(1000, 800))
    _decorate_scene(plotter, bounds)            # background/axes/box: added ONCE

    def render_idx(value):
        idx = max(0, min(len(grids) - 1, int(round(value))))
        _clear_frame_actors(plotter)
        _add_frame_actors(plotter, grids[idx], args, clim,
                          _frame_label(event_id, ts[idx]), overlay, float(ts[idx]))

    plotter.add_slider_widget(render_idx, [0, len(grids) - 1],
                              value=len(grids) // 2, title="frame index", fmt="%.0f")
    render_idx(len(grids) // 2)
    _beam_camera(plotter, args.azimuth, args.elevation)
    plotter.show()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if not (args.movie or args.vtk_dir or args.interactive):
        # Default to a GIF so a bare invocation still produces something.
        args.movie = "evolution.gif"
        print("No output selected; defaulting to --movie evolution.gif")

    if args.load:
        source = load_milne(args.load)
    else:
        # Resolve XML paths before chdir, then run from the build dir so MUSIC
        # finds music_input / EOS / tables.
        args.main = os.path.abspath(args.main)
        args.user = os.path.abspath(args.user)
        if args.workdir and os.path.isdir(args.workdir):
            print(f"Running from workdir: {args.workdir}")
            os.chdir(args.workdir)
        elif args.workdir:
            print(f"[!] workdir {args.workdir} not found; running from {os.getcwd()}")
        # outdir is relative to the launch dir, not the build dir.
        if not os.path.isabs(args.outdir):
            args.outdir = os.path.join(_THIS_DIR, args.outdir)
        source = iter_events(args)

    n = 0
    for event_id, arr, meta in source:
        print(f"=== rendering event {event_id} ===")
        render_event(event_id, arr, meta, args)
        n += 1
    print(f"Done. Rendered {n} event(s). Outputs in {os.path.abspath(args.outdir)}/")


if __name__ == "__main__":
    main()
