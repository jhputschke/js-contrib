"""
contribs/Visualization/hydro_jet_particles_pyvista.py

Medium + deposit, the parton shower, and then the HADRONS: one production file of
PyJetscape's `example/prod_AuAu_0_10_jet`, from the collision to after freeze-out.

    t = 0              the jet partons are born at the hard vertex and run in vacuum
    t >= tau_0         the medium forms (the jet leg `arr` = background + deposit)
    t >= t_emit        each bulk hadron appears where iSS emitted it and free-streams
    t >= --frag-time   the partons are replaced by their fragmentation hadrons
                       (with --formation-tau0: each hadron at its own formation time)

No wake difference: this is the single jet-leg panel of `wake_pyvista.py` (`--panels jet`),
followed past freeze-out.  The background leg (`arr_bg`, `*_hadrons_bulk_bg.h5`) is not
used.

Where the data comes from
-------------------------
All from the files `run_prod_jet.py` + `hadronize.py` leave next to each other:

    <stem>.h5                        arr (medium + deposit) and shower/   (the pair file)
    <stem>_hadrons_bulk_jet.h5       iSS on the jet leg's surface: bulk + wake hadrons
    <stem>_hadrons_jet_frag.h5       ColorlessHadronization of the final partons

Both hadron files are OVERSAMPLED: every sample is a complete, independent Cooper-Frye
sampling (resp. fragmentation) of the same event.  One sample of each is drawn -- chosen at
random unless `--sample` / `--frag-sample` fix it; the run prints which one it took, and
`--rng-seed` makes the choice reproducible.

What the hadron positions mean
------------------------------
* **Bulk hadrons** carry iSS's space-time point `x = (t, x, y, z)`: the freeze-out cell for
  a directly emitted hadron, the decay vertex for a resonance daughter.  A hadron is drawn
  from that time on, at `x + v (t - t_emit)` with `v = p/E`.  Only final-state hadrons are
  stored, so a resonance is invisible between its emission and its decay, and daughters of
  long-lived parents (weak and electromagnetic decays -- iSS puts them at t ~ 1e10 fm/c) are
  produced after the last frame and never drawn; the run reports how many.
* **Jet hadrons** have NO space-time position: Pythia's string fragmentation writes x = 0.
  They are drawn on straight lines from the event's hard vertex (`initiators/`),
  `x_hard + v t`, and only from `--frag-time` on, when the shower overlay is switched off.
  At |v| ~ 1 that puts them about where the leading partons are, but it is a picture, not a
  transport: the deflections in the medium are not in it.  The default `--frag-time` is the
  last vertex of the shower, i.e. the moment the shower stops changing.

Formation times (`--formation-tau0`)
------------------------------------
By default every jet hadron appears at the one `--frag-time`.  That is how the event was
MADE -- X-SCAPE hadronizes once, after the whole shower, and ColorlessHadronization hands
Pythia momenta and colours only -- but not how hadrons form: in the inside-outside picture
a hadron forms at a lab time `t_form ~ tau0 E/m`, so soft fragments appear early and the
leading ones last (a 10 GeV pion at tau0 = 1 fm/c: ~70 fm/c).  With `--formation-tau0 TAU0`
each jet hadron appears at

    t_on = max(--frag-time, t_vertex + TAU0 * E / m)

on its straight line from the hard vertex.  `--frag-time` stays a floor (hadrons form after
the shower is over); `--frag-time 0` removes it.  `m` is the species mass, at least the pion
mass, so photons and leptons from Pythia's decays count as pions (their parents' formation
is what matters and is not in the file).  The file has no link from a hadron to the partons
it came from -- Pythia's strings mix them -- so no parton can be switched off when "its"
hadron forms: `--parton-off formed` (the default in this mode) keeps the whole shower on
until the last drawn hadron has formed, `--parton-off frag` switches it off at --frag-time.

Hadrons with `t_emit` far beyond the displayed range are also far along the beam (t =
tau cosh eta_s); `--hadron-eta-max` keeps the picture to mid-rapidity if the forward ones
distract.

Usage
-----
    conda activate fno_pyvista_env
    cd external_packages/js-contrib/contribs/Visualization

    python hydro_jet_particles_pyvista.py \
        --file ../PyJetscape/example/prod_AuAu_0_10_jet/out/AuAu_0_10_jet_seed0001.h5 \
        --nt 60 --movie jet_particles.mp4

    # a fixed sample, mid-rapidity hadrons only, interactive
    python hydro_jet_particles_pyvista.py --file ...seed0001.h5 --sample 17 \
        --hadron-eta-max 1.5 --interactive
"""

from __future__ import annotations

import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# wake_pyvista sets up the FastHydro path (shower reconstruction) and imports the other two;
# hydro_pyvista puts PyJetscape/python on sys.path (jetscape.hadrons_h5).
import wake_pyvista as wp                                             # noqa: E402
import hydro_jet_pyvista as hjp                                       # noqa: E402
import hydro_pyvista as hp                                            # noqa: E402

#: bulk_jet = iSS on the jet leg (background + deposit); jet_frag = the jet's own hadrons
TAGS = ("bulk_jet", "jet_frag")

#: INITIATOR_COLUMNS (jetscape.showers): shower, pid, pstat, px, py, pz, E, x, y, z, t
_INI_X = slice(7, 10)
_INI_T = 10

#: masses [GeV] by |pid| for the formation time; others from the four-momentum.  Stored p
#: may be mantissa-rounded (keep_bits), which makes E^2 - p^2 useless at high E, hence a table.
MASS = {211: 0.13957, 111: 0.13498, 321: 0.49368, 311: 0.49761, 310: 0.49761,
        130: 0.49761, 221: 0.54786, 2212: 0.93827, 2112: 0.93957, 3122: 1.11568,
        3222: 1.18937, 3212: 1.19264, 3112: 1.19745, 3322: 1.31486, 3312: 1.32171,
        3334: 1.67245}
#: floor of the formation-time mass: photons and leptons (decay products) count as pions
M_PI = 0.13957

#: iSS places the decay of a long-lived parent (weak, electromagnetic) at t ~ 1e10 fm/c;
#: only used to say how many hadrons that hides
LATE_DECAY_T = 1.0e5


# ──────────────────────────────────────────────────────────────────────────────
# Reading the hadrons
# ──────────────────────────────────────────────────────────────────────────────

def hadron_stem(pair_path):
    """`<stem>.h5` -> `<stem>`, the prefix hadronize.py gives the hadron files."""
    base = str(pair_path)
    return base[:-3] if base.endswith(".h5") else base


def _pick(n, fixed, rng, what):
    if n == 0:
        return None
    if fixed is None:
        return int(rng.integers(n))
    if not 0 <= fixed < n:
        raise SystemExit(f"{what} {fixed} out of range (the event has {n})")
    return int(fixed)


def hard_vertex(pair_path, frag_file, event):
    """-> (x, y, z, t) of the hard vertex, from the initiators (all share it).

    The hadron files carry a copy of the pair file's `shower/initiators` (hadronize.py since
    js-contrib #20); older ones do not, and then the pair file is read.  (0, 0, 0, 0) if
    neither has any.
    """
    ini = frag_file.initiators(event) if frag_file is not None else None
    if ini is None or not len(ini):
        import h5py
        with h5py.File(str(pair_path), "r") as f:
            if "shower/initiators" in f:
                o = f["shower/initiator_offsets"][:]
                ini = f["shower/initiators"][o[event]:o[event + 1]]
    if ini is None or not len(ini):
        return np.zeros(3), 0.0
    ini = np.asarray(ini, float)
    return ini[0, _INI_X], float(ini[0, _INI_T])


def load_hadrons(pair_path, event=0, stem=None, sample=None, frag_sample=None,
                 rng_seed=None):
    """-> dict with one sample of the jet leg's bulk hadrons and one fragmentation.

    Keys: `bulk`, `frag` (each None or a dict of pid, pstat, p [E,px,py,pz], x [t,x,y,z]),
    `k`, `n_k` (bulk sample taken, of how many), `j`, `n_j` (fragmentation), `vertex`,
    `t_vertex` (hard vertex), `paths`.
    """
    from jetscape.hadrons_h5 import HadronFile

    stem = stem or hadron_stem(pair_path)
    paths = {tag: f"{stem}_hadrons_{tag}.h5" for tag in TAGS}
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if len(missing) == len(paths):
        raise SystemExit(
            f"no hadron files next to {pair_path} (looked for {missing[0]} ...). Make them "
            f"with prod_AuAu_0_10_jet/hadronize.py, or point --hadron-stem at them.")
    for p in missing:
        print(f"  [!] {p} not found; drawing without those hadrons")

    rng = np.random.default_rng(rng_seed)
    out = dict(bulk=None, frag=None, k=None, n_k=0, j=None, n_j=0, paths=paths)
    files = {tag: HadronFile(p) for tag, p in paths.items() if os.path.exists(p)}
    try:
        if "bulk_jet" in files:
            hf = files["bulk_jet"]
            out["n_k"] = hf.n_samples(event)
            out["k"] = _pick(out["n_k"], sample, rng, "--sample")
            if out["k"] is not None:
                out["bulk"] = hf.sample_event(event, out["k"])
        frag = files.get("jet_frag")
        if frag is not None:
            out["n_j"] = frag.n_samples(event)
            out["j"] = _pick(out["n_j"], frag_sample, rng, "--frag-sample")
            if out["j"] is not None:
                out["frag"] = frag.sample_event(event, out["j"])
        out["vertex"], out["t_vertex"] = hard_vertex(pair_path, frag, event)
    finally:
        for hf in files.values():
            hf.close()
    return out


def select(h, eta_max=None, pt_min=0.0):
    """Boolean mask of the hadrons of `h` worth drawing.

    Drops negative-status entries (a hole's fragments are subtracted in an analysis, not
    particles one could see), and applies the optional pseudorapidity / pT cuts.
    """
    if h is None:
        return None
    p = np.asarray(h["p"], float)
    keep = np.asarray(h["pstat"]) >= 0
    pt = np.hypot(p[:, 1], p[:, 2])
    if pt_min > 0:
        keep &= pt >= pt_min
    if eta_max is not None:
        pabs = np.linalg.norm(p[:, 1:], axis=1)
        eta = 0.5 * np.log(np.maximum(pabs + p[:, 3], 1e-12) /
                           np.maximum(pabs - p[:, 3], 1e-12))
        keep &= np.abs(eta) <= eta_max
    return keep


# ──────────────────────────────────────────────────────────────────────────────
# Overlay
# ──────────────────────────────────────────────────────────────────────────────

class FreeStreaming:
    """Straight-line hadron tracks: at lab time t the hadron is at `x0 + v (t - t0)`,
    and is drawn only once `t >= t_on` (its emission, or the fragmentation time)."""

    def __init__(self, p, x0, t0, t_on, pT):
        p = np.asarray(p, float)
        E = np.maximum(p[:, 0], 1e-9)
        self.v = p[:, 1:] / E[:, None]
        self.x0 = np.asarray(x0, float)
        self.t0 = np.asarray(t0, float)
        self.t_on = np.asarray(t_on, float)
        self.pT = np.asarray(pT, float)

    def __len__(self):
        return len(self.t0)

    def at(self, t):
        on = t >= self.t_on
        pos = self.x0[on] + self.v[on] * (t - self.t0[on])[:, None]
        return pos, self.pT[on]


def bulk_tracks(h, keep):
    x = np.asarray(h["x"], float)[keep]
    p = np.asarray(h["p"], float)[keep]
    return FreeStreaming(p, x[:, 1:], x[:, 0], x[:, 0], np.hypot(p[:, 1], p[:, 2]))


def hadron_mass(pid, p):
    """Mass [GeV] for the formation time: MASS by |pid|, else sqrt(E^2 - p^2); never below
    the pion mass."""
    pid = np.abs(np.asarray(pid, int))
    p = np.asarray(p, float)
    inv = np.sqrt(np.maximum(p[:, 0] ** 2 - (p[:, 1:] ** 2).sum(axis=1), 0.0))
    m = np.array([MASS.get(int(i), np.nan) for i in pid], float)
    m = np.where(np.isnan(m), inv, m)
    return np.maximum(m, M_PI)


def formation_times(pid, p, t_vertex, tau0, t_floor):
    """Lab time each jet hadron forms: max(t_floor, t_vertex + tau0 * E/m)."""
    E = np.asarray(p, float)[:, 0]
    return np.maximum(float(t_floor), float(t_vertex) + float(tau0) * E / hadron_mass(pid, p))


def frag_tracks(h, keep, vertex, t_vertex, t_frag, tau0=None):
    """Jet hadrons on straight lines from the hard vertex; they appear at `t_frag`, or with
    `tau0` at their own `formation_times` (t_frag is then the floor)."""
    p = np.asarray(h["p"], float)[keep]
    n = len(p)
    t_on = (np.full(n, float(t_frag)) if tau0 is None else
            formation_times(np.asarray(h["pid"])[keep], p, t_vertex, tau0, t_frag))
    return FreeStreaming(p, np.tile(vertex, (n, 1)), np.full(n, t_vertex), t_on,
                         np.hypot(p[:, 1], p[:, 2]))


def parton_off_time(mode, t_frag, frag):
    """When the shower arrows go: at `t_frag` ("frag"), or once the last drawn jet hadron
    has formed ("formed"; t_frag if there are none)."""
    if mode == "formed" and frag is not None and len(frag):
        return float(max(t_frag, frag.t_on.max()))
    return float(t_frag)


def _pt_bar_args():
    a = hjp._jet_bar_args()
    a["title"] = "jet pT  [GeV]"          # partons, then the fragments, on one scale
    return a


def _add_pt_bar(plotter, max_pT, cmap):
    import pyvista as pv
    proxy = pv.PolyData(np.zeros((2, 3)))
    proxy["pT"] = np.array([0.0, max_pT], dtype=float)
    plotter.add_mesh(proxy, scalars="pT", cmap=cmap, clim=(0.0, max_pT), opacity=0.0,
                     name="jet_pt_bar", reset_camera=False, show_scalar_bar=True,
                     scalar_bar_args=_pt_bar_args())


def _draw_points(plotter, name, pos, **kw):
    import pyvista as pv
    if not len(pos):
        plotter.remove_actor(name, reset_camera=False)
        return
    plotter.add_mesh(pv.PolyData(np.ascontiguousarray(pos)), name=name,
                     render_points_as_spheres=True, reset_camera=False,
                     show_scalar_bar=False, **kw)


def make_overlay(seg, bulk, frag, args, max_pT, t_partons_off, t_anchor, note, box):
    """overlay(plotter, t): the shower until `t_partons_off`, the jet hadrons from their
    appearance time on (`frag.t_on`), the bulk hadrons
    from their emission on.  `note` is the static part of the lower-left read-out, `box`
    the medium box's (xmin, xmax, ymin, ymax, zmin, zmax)."""
    import pyvista as pv

    partons = (hjp.make_jet_overlay(seg, args, max_pT, t_anchor, colorbar=False)
               if seg is not None else None)
    state = {"bar": False}
    n_bulk = len(bulk) if bulk is not None else 0

    def overlay(plotter, t):
        if not state["bar"] and not args.jet_color and (partons or frag is not None):
            _add_pt_bar(plotter, max_pT, args.jet_cmap)
            state["bar"] = True

        if partons is not None and t < t_partons_off:
            partons(plotter, t)
        else:
            plotter.remove_actor("jets", reset_camera=False)
            plotter.remove_actor("jet_heads", reset_camera=False)

        shown_b = 0
        if bulk is not None:
            pos, _ = bulk.at(t)
            shown_b = len(pos)
            _draw_points(plotter, "bulk_hadrons", pos, color=args.bulk_color,
                         point_size=args.bulk_size, opacity=args.bulk_opacity)
        shown_f = 0
        if frag is not None:
            pos, pT = frag.at(t)
            shown_f = len(pos)
            if shown_f:
                cloud = pv.PolyData(np.ascontiguousarray(pos))
                cloud["pT"] = pT
                kw = (dict(color=args.jet_color) if args.jet_color else
                      dict(scalars="pT", cmap=args.jet_cmap, clim=(0.0, max_pT)))
                plotter.add_mesh(cloud, name="jet_hadrons", point_size=args.frag_size,
                                 render_points_as_spheres=True, reset_camera=False,
                                 show_scalar_bar=False, **kw)
            else:
                plotter.remove_actor("jet_hadrons", reset_camera=False)

        text = (f"{note}\nbulk hadrons shown {shown_b} / {n_bulk}"
                + (f",  jet hadrons {shown_f} / {len(frag)}" if frag is not None else ""))
        plotter.add_text(text, name="hadron_note", position="lower_left", font_size=10,
                         color="#c8d6e0", shadow=False)
        # Every add_mesh re-fits show_grid's labelled box to the bounds of ALL actors, and
        # the hadrons fly far out of the medium box -- so without this the box grows frame
        # by frame and reads as a zoom.  The overlay runs last in a frame, so pin it here.
        axes_actor = getattr(plotter.renderer, "cube_axes_actor", None)
        if axes_actor is not None:
            axes_actor.update_bounds(box)

    return overlay


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_parser():
    p = hjp.build_parser()
    p.description = __doc__
    g = p.add_argument_group("production file and hadrons")
    g.add_argument("--file", required=True,
                   help="the pair file <stem>.h5 of a prod_AuAu_0_10_jet production; the "
                        "hadron files <stem>_hadrons_{bulk_jet,jet_frag}.h5 are found next "
                        "to it")
    g.add_argument("--hadron-stem", default=None, dest="hadron_stem",
                   help="prefix of the hadron files if they are not next to --file")
    g.add_argument("--event", type=int, default=0, help="event in the file (default 0)")
    g.add_argument("--sample", type=int, default=None,
                   help="bulk_jet oversample to draw (default: a random one)")
    g.add_argument("--frag-sample", type=int, default=None, dest="frag_sample",
                   help="jet_frag fragmentation to draw (default: a random one)")
    g.add_argument("--rng-seed", type=int, default=None, dest="rng_seed",
                   help="seed for the random sample choice (default: fresh each run; the "
                        "run prints what it took)")
    g.add_argument("--eos-table", default=None, dest="eos_table",
                   help="MUSIC hotQCD table (file or EOS/hotQCD directory) for the "
                        "temperature; by default $MUSIC_EOS_TABLE, then "
                        "<prod_build>/EOS/hotQCD")
    g.add_argument("--frag-time", type=float, default=None, dest="frag_time",
                   help="lab time [fm/c] at which the partons are replaced by the jet "
                        "hadrons (default: the shower's last vertex)")
    g.add_argument("--formation-tau0", type=float, default=None, dest="formation_tau0",
                   metavar="TAU0",
                   help="give each jet hadron its own formation time "
                        "max(--frag-time, t_vertex + TAU0*E/m), TAU0 in fm/c (e.g. 1); "
                        "default off: all appear at --frag-time")
    g.add_argument("--parton-off", choices=("auto", "frag", "formed"), default="auto",
                   dest="parton_off",
                   help="when the shower arrows are removed: at --frag-time ('frag'), or "
                        "once the last drawn jet hadron has formed ('formed'). 'auto' "
                        "(default): 'formed' with --formation-tau0, else 'frag'")
    g.add_argument("--stream-time", type=float, default=6.0, dest="stream_time",
                   help="how long to follow the hadrons after both the medium and the "
                        "shower have ended, in fm/c (default 6); --t-max overrides")
    g.add_argument("--hadron-eta-max", type=float, default=None, dest="hadron_eta_max",
                   help="draw only hadrons with |pseudorapidity| below this")
    g.add_argument("--hadron-pt-min", type=float, default=0.0, dest="hadron_pt_min",
                   help="draw only hadrons above this pT in GeV (default 0)")
    g.add_argument("--bulk-color", default="#e6f2ff", dest="bulk_color",
                   help="colour of the bulk hadrons (default a pale blue-white)")
    g.add_argument("--bulk-size", type=float, default=4.0, dest="bulk_size",
                   help="bulk hadron point size in pixels (default 4)")
    g.add_argument("--bulk-opacity", type=float, default=0.85, dest="bulk_opacity")
    g.add_argument("--frag-size", type=float, default=10.0, dest="frag_size",
                   help="jet hadron point size in pixels (default 10), coloured by pT on "
                        "the partons' scale")
    g.add_argument("--no-jet", action="store_true", dest="no_jet",
                   help="do not draw the parton shower (the jet hadrons stay)")
    g.add_argument("--no-bulk", action="store_true", dest="no_bulk",
                   help="do not draw the bulk hadrons")
    g.add_argument("--no-frag", action="store_true", dest="no_frag",
                   help="do not draw the jet hadrons")
    p.set_defaults(movie="jet_particles.mp4", t_min=0.0)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not (args.movie or args.vtk_dir or args.interactive):
        args.movie = "jet_particles.mp4"
    if not os.path.isabs(args.outdir):
        args.outdir = os.path.join(_THIS_DIR, args.outdir)

    print(f"=== {args.file}  event {args.event} ===")
    panels, meta, attrs = wp.load_pair(args.file, args.event, ("jet",),
                                       eos_table=args.eos_table)
    arr = panels["jet"]
    tau_max = meta["tau_min"] + (meta["ntau"] - 1) * meta["dtau"]
    print(f"  medium + deposit (arr): grid {meta['nx']}x{meta['ny']}x{meta['neta']}"
          f"x{meta['ntau']}, tau {meta['tau_min']:.2f}..{tau_max:.2f} fm/c")
    print(f"  temperature from {meta['eos']}")

    seg = None if args.no_jet else wp.load_shower(args.file, args.event,
                                                  args.jet_min_energy)
    if seg is not None:
        print(f"  shower: {len(seg['starts'])} partons, max pT {seg['pT'].max():.1f} GeV")
    elif not args.no_jet:
        print("  [!] no shower/ in the pair file; drawing without partons")

    had = load_hadrons(args.file, args.event, args.hadron_stem, args.sample,
                       args.frag_sample, args.rng_seed)

    # When the partons turn into hadrons: the last time the shower changes.
    t_shower = None
    if seg is None:
        seg_all = wp.load_shower(args.file, args.event, 0.0)
    else:
        seg_all = seg
    if seg_all is not None:
        t_shower = float(max(seg_all["t0"].max(), seg_all["t1"].max()))
    t_frag = args.frag_time if args.frag_time is not None else (t_shower or tau_max)
    if args.t_max is None:
        args.t_max = max(t_frag, tau_max) + args.stream_time
    # The medium box stays the one hydro_jet_pyvista uses (z tied to the hydro's life, not
    # to the particles'): a z range widened to the hadrons' reach would show the thin
    # early-proper-time medium near the light cone, and the hadrons may leave the box.
    if args.z_max is None:
        args.z_max = 0.8 * tau_max

    bulk = frag = None
    if had["bulk"] is not None and not args.no_bulk:
        keep = select(had["bulk"], args.hadron_eta_max, args.hadron_pt_min)
        bulk = bulk_tracks(had["bulk"], keep)
        late = bulk.t_on > args.t_max
        decays = int((bulk.t_on > LATE_DECAY_T).sum())
        print(f"  bulk_jet sample {had['k']} of {had['n_k']}: {len(bulk)} hadrons drawn "
              f"(of {len(keep)}); {int(late.sum())} are produced after t = "
              f"{args.t_max:.1f} fm/c and never appear: {decays} decay products of "
              f"long-lived parents (t > {LATE_DECAY_T:.0e} fm/c), the rest emitted far "
              f"along the beam (t = tau cosh eta_s)")
    elif had["n_k"] == 0 and not args.no_bulk:
        print("  [!] no bulk_jet samples for this event (empty surface?)")
    if had["frag"] is not None and not args.no_frag:
        keep = select(had["frag"], args.hadron_eta_max, args.hadron_pt_min)
        frag = frag_tracks(had["frag"], keep, had["vertex"], had["t_vertex"], t_frag,
                           args.formation_tau0)
        neg = int((np.asarray(had["frag"]["pstat"]) < 0).sum())
        when = (f"shown from t = {t_frag:.2f} fm/c" if args.formation_tau0 is None else
                f"formation time t_vertex + {args.formation_tau0:g}*E/m, not before "
                f"{t_frag:.2f} fm/c")
        print(f"  jet_frag sample {had['j']} of {had['n_j']}: {len(frag)} hadrons drawn"
              + (f" ({neg} negative-status not drawn)" if neg else "")
              + f", from the hard vertex ({', '.join(f'{v:.2f}' for v in had['vertex'])})"
              f" fm, {when}")
        if args.formation_tau0 is not None and len(frag):
            late = int((frag.t_on > args.t_max).sum())
            print(f"    formation times {frag.t_on.min():.1f} .. median "
                  f"{np.median(frag.t_on):.1f} .. {frag.t_on.max():.1f} fm/c; {late} form "
                  f"after t = {args.t_max:.1f} fm/c"
                  + (f" (--t-max {np.ceil(frag.t_on.max()):.0f} shows all)" if late else ""))
    if args.sample is None or args.frag_sample is None:
        print(f"  (random sample choice; reproduce with --sample {had['k']} "
              f"--frag-sample {had['j']})")

    max_pT = 1.0
    if seg is not None and len(seg["pT"]):
        max_pT = float(seg["pT"].max())
    elif frag is not None and len(frag):
        max_pT = float(frag.pT.max())

    lead = wp.leading_parton(args.file, args.event)
    note = f"medium + deposit;  bulk sample {had['k']}/{had['n_k']}"
    if had["j"] is not None:
        note += f",  fragmentation {had['j']}/{had['n_j']}"
    if lead:
        note = f"leading parton pT = {lead[0]:.1f} GeV\n" + note
    mode = args.parton_off
    if mode == "auto":
        mode = "formed" if args.formation_tau0 is not None else "frag"
    t_off = parton_off_time(mode, t_frag, frag)
    if args.formation_tau0 is None:
        note += f"\npartons -> jet hadrons at t = {t_frag:.1f} fm/c"
    else:
        note += (f"\njet hadrons form at max({t_frag:.1f}, t_vertex + "
                 f"{args.formation_tau0:g} fm/c * E/m); partons off at t = {t_off:.1f}")
    # the same box hp.render_event will build, for make_overlay to hold the grid to
    xy_max = args.xy_max if args.xy_max is not None else hp.medium_xy_max(arr, meta)
    box = hp._scene_bounds(hp.cartesian_axes(meta, args, args.t_max, xy_max))
    # The camera frames the jets where the shower ends, whatever the mode: framing them at
    # t_off (up to ~100 fm/c with formation times) would shrink the medium to a dot.
    t_anchor = min(t_shower if t_shower is not None else tau_max, args.t_max)
    overlay = make_overlay(seg, bulk, frag, args, max_pT, t_off, t_anchor, note, box)

    hp.render_event(args.event, arr, meta, args, overlay=overlay)
    print(f"Done. Outputs in {os.path.abspath(args.outdir)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
