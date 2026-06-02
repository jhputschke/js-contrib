"""
contribs/Visualization/hydro_jet_pyvista.py

Hydro medium evolution (Milne→Cartesian) WITH the jet parton shower overlaid,
in PyVista.  Builds on hydro_pyvista.py: the medium is rendered exactly as there
(energy-density volume + freeze-out isosurface in lab spacetime), and the parton
shower is drawn as accumulating arrows in the same Cartesian (t,x,y,z) frame.

Where the shower comes from
---------------------------
The shower is read LIVE from the framework via the pybind11 bindings (no output
file parsing), at the per-event yield point — the same place the hydro is read:

    sm = JetScapeSignalManager.Instance()
    jm = sm.GetJetEnergyLossManagerPointer()      # JetEnergyLossManager
    for ps in jm.get_showers():                   # one per shower-initiating parton
        edges = ps.to_numpy()                     # (n_partons, 12)
        # [source_id, target_id, pid, pstat, px, py, pz, E, x, y, z, t]

The shower graph: nodes (vertices) carry a time, edges (partons) carry a
space-time position (x,y,z,t in fm / fm/c) and momentum.  Only the recorded
vertex points exist, so a parton's position at an intermediate lab time t is the
straight-line interpolation between its start and end vertex (i.e. motion at the
parton velocity).  As lab time advances the shower is NOT cleared — earlier
partons persist, so the full tree accumulates.

Non-live mode
-------------
With --load <hydro.npz> the medium is read from a dump (as in hydro_pyvista) and
the shower is read from the JetScape ASCII writer output (--jet-ascii, default
<workdir>/test_out.dat).  This is also the path used to verify the bindings.

Usage
-----
    conda activate fno_pyvista_env

    # Live: run MUSIC + jets (config/OO_one_event_jet.xml) and render medium+jets
    python hydro_jet_pyvista.py --events 1 --movie evt_jet.gif

    # Fast re-render from a prior run's hydro dump + ascii shower:
    python hydro_jet_pyvista.py --load hydro.npz --jet-ascii test_out.dat \
        --movie evt_jet.gif

Jet options: --jet-radius, --jet-color, --jet-cmap (colour by energy),
--jet-min-energy, --jet-ascii.
"""

from __future__ import annotations

import os
import re
import sys

import numpy as np

# Make this directory importable so we can reuse hydro_pyvista.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

import hydro_pyvista as hp   # noqa: E402  (also sets up the jetscape import path)

# Column layout of PartonShower.to_numpy() and parse_shower_ascii().
C_SRC, C_TGT, C_PID, C_PSTAT, C_PX, C_PY, C_PZ, C_E, C_X, C_Y, C_Z, C_T = range(12)


# ──────────────────────────────────────────────────────────────────────────────
# Shower acquisition
# ──────────────────────────────────────────────────────────────────────────────

def showers_live():
    """Per-event shower edge arrays from the live framework (bindings).

    Returns a list of (n_partons, 12) numpy arrays, or [] if no jets.
    Must be called at the per-event yield point (before ClearPerEvent).
    """
    from jetscape import JetScapeSignalManager
    jm = JetScapeSignalManager.Instance().GetJetEnergyLossManagerPointer()
    if jm is None:
        return []
    return [ps.to_numpy() for ps in jm.get_showers()]


def parse_shower_ascii(path: str):
    """Parse a JetScapeWriterAscii file into per-event shower edge arrays.

    Returns a list (indexed by event) of lists of (n_partons, 12) arrays in the
    same column layout as PartonShower.to_numpy().  The ASCII momentum is stored
    as (pt, eta, phi, E); it is converted to (px, py, pz, E) here.
    """
    edge_re = re.compile(r"\[(\d+)\]=>\[(\d+)\]\s+P\s+(.*)")
    events, cur_event, cur_shower = [], None, None
    with open(path) as fh:
        for line in fh:
            if re.match(r"^\s*\d+\s+Event", line):
                cur_event = []
                events.append(cur_event)
                cur_shower = None
                continue
            if "Parton Shower in JetScape format" in line:
                if cur_event is None:
                    cur_event = []
                    events.append(cur_event)
                cur_shower = []
                cur_event.append(cur_shower)
                continue
            m = edge_re.match(line)
            if m and cur_shower is not None:
                src, tgt = int(m.group(1)), int(m.group(2))
                f = [float(v) for v in m.group(3).split()]
                # plabel pid pstat pt eta phi E x y z t
                _, pid, pstat, pt, eta, phi, E, x, y, z, t = f[:11]
                px = pt * np.cos(phi)
                py = pt * np.sin(phi)
                pz = pt * np.sinh(eta)
                cur_shower.append([src, tgt, pid, pstat, px, py, pz, E, x, y, z, t])
    return [[np.asarray(s, dtype=float).reshape(-1, 12) for s in ev] for ev in events]


# ──────────────────────────────────────────────────────────────────────────────
# Shower → drawable parton segments
# ──────────────────────────────────────────────────────────────────────────────

def _segments_from_edges(edges: np.ndarray, min_energy: float):
    """One shower's edge array → straight-line parton segments.

    Each parton edge a→b is a segment from vertex a's position@t_a to vertex b's
    position@t_b.  The recorded (x,y,z,t) on an edge is its *endpoint* (vertex b),
    so vertex positions are taken from the edges; the start of a segment is the
    parent edge's endpoint.  The degenerate root edge (zero length, t0=t1) is
    dropped.
    """
    if edges is None or len(edges) == 0:
        return None
    src = edges[:, C_SRC].astype(int)
    tgt = edges[:, C_TGT].astype(int)
    pos = edges[:, C_X:C_Z + 1]
    tim = edges[:, C_T]
    mom = edges[:, C_PX:C_PZ + 1]
    E   = edges[:, C_E]

    vpos, vtime = {}, {}
    for i in range(len(edges)):                       # vertex b ← edge a→b endpoint
        vpos[tgt[i]] = pos[i]
        vtime[tgt[i]] = tim[i]
    for i in range(len(edges)):                       # roots (source w/o incoming edge)
        if src[i] not in vpos:
            vpos[src[i]] = pos[i]
            vtime[src[i]] = tim[i]

    starts, ends, t0, t1, energy, dirs = [], [], [], [], [], []
    for i in range(len(edges)):
        if E[i] < min_energy:
            continue
        a, b = src[i], tgt[i]
        s, e = vpos[a], vpos[b]
        ta, tb = vtime[a], vtime[b]
        seg = e - s
        seg_len = float(np.linalg.norm(seg))
        if seg_len < 1e-6 and abs(tb - ta) < 1e-9:
            continue                                  # degenerate root parton
        p = mom[i]
        pn = float(np.linalg.norm(p))
        if pn > 1e-9:
            d = p / pn
        elif seg_len > 1e-9:
            d = seg / seg_len
        else:
            d = np.array([0.0, 0.0, 1.0])
        starts.append(s); ends.append(e); t0.append(ta); t1.append(tb)
        energy.append(E[i]); dirs.append(d)

    if not starts:
        return None
    return dict(starts=np.asarray(starts), ends=np.asarray(ends),
                t0=np.asarray(t0), t1=np.asarray(t1),
                energy=np.asarray(energy), dirs=np.asarray(dirs))


def build_segments(shower_arrays, min_energy: float = 0.0):
    """Combine all showers of an event into one segment bundle (or None)."""
    parts = [_segments_from_edges(a, min_energy) for a in shower_arrays]
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    return {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}


# ──────────────────────────────────────────────────────────────────────────────
# Overlay: accumulating parton arrows at lab time t
# ──────────────────────────────────────────────────────────────────────────────

def _add_jet_actors(plotter, starts, tips, dirs, energy, args) -> None:
    import pyvista as pv
    plotter.remove_actor("jets", reset_camera=False)
    plotter.remove_actor("jet_heads", reset_camera=False)
    m = len(starts)
    if m == 0:
        return
    # Shafts: one line per parton (start → current tip), thickened to tubes.
    pts = np.empty((2 * m, 3), dtype=float)
    pts[0::2] = starts
    pts[1::2] = tips
    lines = np.empty((m, 3), dtype=np.int64)
    lines[:, 0] = 2
    lines[:, 1] = np.arange(0, 2 * m, 2)
    lines[:, 2] = np.arange(1, 2 * m, 2)
    poly = pv.PolyData(pts, lines=lines.ravel())
    poly["energy"] = np.repeat(energy, 2)
    shafts = poly.tube(radius=args.jet_radius, n_sides=8)

    head_geom = pv.Cone(radius=args.jet_radius * 3.2, height=args.jet_radius * 8.0,
                        resolution=12)
    heads = pv.PolyData(np.asarray(tips, dtype=float))
    heads["dir"] = np.asarray(dirs, dtype=float)
    heads["energy"] = np.asarray(energy, dtype=float)
    glyphs = heads.glyph(orient="dir", scale=False, factor=1.0, geom=head_geom)

    common = dict(reset_camera=False, show_scalar_bar=False)
    if args.jet_cmap:
        clim = (max(0.1, float(np.min(energy))), float(np.max(energy)))
        plotter.add_mesh(shafts, scalars="energy", cmap=args.jet_cmap, clim=clim,
                         log_scale=True, name="jets", **common)
        plotter.add_mesh(glyphs, scalars="energy", cmap=args.jet_cmap, clim=clim,
                         log_scale=True, name="jet_heads", **common)
    else:
        plotter.add_mesh(shafts, color=args.jet_color, name="jets", **common)
        plotter.add_mesh(glyphs, color=args.jet_color, name="jet_heads", **common)


def make_jet_overlay(seg, args):
    """Return overlay(plotter, t) drawing the shower accumulated up to lab time t."""
    if seg is None:
        return None
    starts, ends = seg["starts"], seg["ends"]
    t0, t1, dirs, energy = seg["t0"], seg["t1"], seg["dirs"], seg["energy"]
    span = np.where(t1 > t0, t1 - t0, 1.0)

    def overlay(plotter, t):
        active = t >= t0
        if not active.any():
            plotter.remove_actor("jets", reset_camera=False)
            plotter.remove_actor("jet_heads", reset_camera=False)
            return
        frac = np.clip((t - t0) / span, 0.0, 1.0)        # straight-line in time
        tips = starts + (ends - starts) * frac[:, None]
        _add_jet_actors(plotter, starts[active], tips[active],
                        dirs[active], energy[active], args)

    return overlay


# ──────────────────────────────────────────────────────────────────────────────
# CLI + main
# ──────────────────────────────────────────────────────────────────────────────

def build_parser():
    p = hp.build_parser()
    p.description = __doc__
    # Default to the jet config (Hard + Eloss/Matter); the base script defaults
    # to the hydro-only OO_one_event.xml which produces no showers.
    p.set_defaults(user=os.path.join(_THIS_DIR, "config", "OO_one_event_jet.xml"))
    g = p.add_argument_group("jet overlay")
    g.add_argument("--jet-ascii", default=None, dest="jet_ascii",
                   help="JetScape ASCII shower file for --load mode "
                        "(default <workdir>/test_out.dat).")
    g.add_argument("--jet-radius", type=float, default=0.06, dest="jet_radius",
                   help="Parton tube radius in fm (default 0.06).")
    g.add_argument("--jet-color", default="#1cf5a0", dest="jet_color",
                   help="Parton colour when not colouring by energy (default green).")
    g.add_argument("--jet-cmap", default=None, dest="jet_cmap",
                   help="Colour partons by energy with this colormap (e.g. cool).")
    g.add_argument("--jet-min-energy", type=float, default=0.0, dest="jet_min_energy",
                   help="Drop partons below this energy in GeV (default 0).")
    return p


def _extent_for(args, arr, meta, seg):
    """Pick a Cartesian display box + lab-time span covering medium AND jets.

    By default the animation spans the medium lifetime (τ_max); the box is sized
    to where partons are *at that final time* (not their far-future endpoints), so
    one forward high-energy parton doesn't blow the box up.  Extend with --t-max
    to watch the jets fly out further (the box grows to match).
    """
    tau_max = meta["tau_min"] + (meta["ntau"] - 1) * meta["dtau"]
    med_xy = hp.medium_xy_max(arr, meta)
    t_max = args._user_t_max if args._user_t_max is not None else tau_max
    jx = jz = 0.0
    if seg is not None:
        span = np.where(seg["t1"] > seg["t0"], seg["t1"] - seg["t0"], 1.0)
        frac = np.clip((t_max - seg["t0"]) / span, 0.0, 1.0)
        tips = seg["starts"] + (seg["ends"] - seg["starts"]) * frac[:, None]
        active = t_max >= seg["t0"]
        if active.any():
            ta = tips[active]
            jx = float(np.abs(ta[:, :2]).max())
            jz = float(np.abs(ta[:, 2]).max())
    xy_max = args._user_xy_max if args._user_xy_max is not None else max(med_xy, jx) + 1.0
    z_max  = args._user_z_max  if args._user_z_max  is not None else max(0.8 * tau_max, jz) + 1.0
    return float(xy_max), float(z_max), float(t_max)


def _render(event_id, arr, meta, shower_arrays, args):
    seg = build_segments(shower_arrays, args.jet_min_energy)
    n_part = 0 if seg is None else len(seg["t0"])
    print(f"  jets: {len(shower_arrays)} shower(s), {n_part} drawable parton segments")
    args.xy_max, args.z_max, args.t_max = _extent_for(args, arr, meta, seg)
    overlay = make_jet_overlay(seg, args)
    hp.render_event(event_id, arr, meta, args, overlay=overlay)


def main() -> None:
    args = build_parser().parse_args()
    # Remember which extents the user fixed (so jet auto-extent doesn't override).
    args._user_xy_max, args._user_z_max, args._user_t_max = \
        args.xy_max, args.z_max, args.t_max

    if not (args.movie or args.vtk_dir or args.interactive):
        args.movie = "evolution_jet.gif"
        print("No output selected; defaulting to --movie evolution_jet.gif")

    if args.load:
        # Non-live: hydro from npz, shower from ASCII writer output.
        jet_ascii = args.jet_ascii or os.path.join(args.workdir, "test_out.dat")
        ascii_events = []
        if os.path.exists(jet_ascii):
            ascii_events = parse_shower_ascii(jet_ascii)
            print(f"Loaded shower ASCII {jet_ascii}: {len(ascii_events)} event(s)")
        else:
            print(f"[!] no shower file at {jet_ascii}; rendering medium only.")
        if not os.path.isabs(args.outdir):
            args.outdir = os.path.join(_THIS_DIR, args.outdir)
        for ev_i, (event_id, arr, meta) in enumerate(hp.load_milne(args.load)):
            showers = ascii_events[ev_i] if ev_i < len(ascii_events) else []
            _render(event_id, arr, meta, showers, args)
    else:
        # Live: run the sim; read hydro and shower at the per-event yield point.
        args.main = os.path.abspath(args.main)
        args.user = os.path.abspath(args.user)
        if args.workdir and os.path.isdir(args.workdir):
            print(f"Running from workdir: {args.workdir}")
            os.chdir(args.workdir)
        if not os.path.isabs(args.outdir):
            args.outdir = os.path.join(_THIS_DIR, args.outdir)
        for event_id, arr, meta in hp.iter_events(args):
            showers = showers_live()        # live shower for THIS event
            _render(event_id, arr, meta, showers, args)

    print(f"Done. Outputs in {os.path.abspath(args.outdir)}/")


if __name__ == "__main__":
    main()
