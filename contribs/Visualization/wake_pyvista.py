"""
contribs/Visualization/wake_pyvista.py

The jet wake, side by side, from ONE paired HDF5 file (FastHydro, or a PyJetscape MUSIC pair):

    ┌────────────────────┬────────────────────┬────────────────────┐
    │  medium,no deposit │  medium + deposit  │  the wake          │
    │  arr_bg            │  arr               │  arr - arr_bg      │
    │  + shower          │  + the SAME shower │  + the SAME shower │
    └────────────────────┴────────────────────┴────────────────────┘

The left panel is **not** a no-jet scenario.  There is exactly one shower in the run and
it is drawn, unchanged, in all three panels: a real Matter+LBT shower, already quenched.
What the left panel leaves out is only the medium's *back-reaction* to the energy that
shower gave up.  So left-to-middle adds the response, not the jet.

That the left panel is the medium the shower actually traversed is not incidental -- it
is the mechanism.  `JetScape::SetPointers()` registers only the FIRST FluidDynamics in the
task list as the framework's hydro, and FastHydro puts the background leg there, so Matter
and LBT query `arr_bg` through `GetHydroCellSignal` and the jet leg is invisible to them.
Measured on a run: 119831 medium queries against the background leg, 0 against the jet leg.

The coupling is therefore **one-way**.  The shower is quenched by the undisturbed medium,
its droplets are deposited into the second leg, and nothing feeds the wake back into the
shower -- which is what makes `arr - arr_bg` a clean linear response rather than a mixture
of two different jets, and is also its limitation.

Neither of the first two panels shows the wake: it is a per-mille disturbance on a
28 GeV/fm^3 background, and they are deliberately drawn on ONE colour scale so that is
visible as a fact rather than hidden by rescaling.  The right panel is the subtraction,
where the wake is all that is left.

The three views share a camera, so they rotate and zoom together.

Where the data comes from
-------------------------
A FastHydro pair (`js-contrib/contribs/FastHydro`), which already holds everything needed:

    arr                the jet leg   (source included, per fast_data's SOURCE_CONVENTION)
    arr_bg             the background leg, same initial condition, no jet
    eos/{e,p,T}_tab    the equation of state, so temperature is exact rather than assumed
    shower/partons     the parton shower as a space-time graph
    shower/vertices    its splitting times
    source/droplets    what the shower gave up to the medium

Nothing is run here: this reads a finished file.  Produce one with

    python FastHydro/example/make_wake_data.py --force

A MUSIC pair from PyJetscape's `PairH5Writer` (e.g. `PyJetscape/example/prod_AuAu_0_10_jet/
run_prod_jet.py`) has the same layout -- arr = MUSIC_2 with the CausalLiquefier source,
arr_bg = MUSIC_1 -- but no eos/ group: it names the EoS in `eos_kind` and the build in
`prod_build`, and `resolve_temperature` loads MUSIC's own hotQCD table from there (or from
--eos-table).  Such a file can hold a dijet, and every shower in it is drawn.

Either way, only frames where BOTH legs are still live are shown (`live_frames`): the
jet leg outlives its background, and past the background's freeze-out the difference
would be the jet leg's whole medium rather than a wake.

Coordinates
-----------
FastHydro stores Milne `(tau, x, y, eta_s)`; `hydro_pyvista` resamples to Cartesian lab
`(t, x, y, z)`, which is the frame the shower is already in, so the overlay needs no
conversion.  Resampling happens once per panel, so a three-panel render costs three times
a single one -- keep `--nt` modest while iterating.

Usage
-----
    conda activate fno_pyvista_env

    python wake_pyvista.py --file ../../../build_gpu/out_wake/wake_ideal.h5 \
        --movie wake.mp4

    # just the subtraction, more frames, no shower
    python wake_pyvista.py --file wake_ideal.h5 --panels diff --nt 80 --no-jet

    # interactive, with a lab-time slider
    python wake_pyvista.py --file wake_ideal.h5 --interactive
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

try:    # registers Blosc, the default compression of pair files (README_h5_optim.md)
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# FastHydro's `showers` module is pure numpy and is the reference implementation of the
# space-time reconstruction (it is the one with the gates); reuse it rather than keeping a
# second copy of a reconstruction that has three easy ways to be subtly wrong.
_FASTHYDRO = os.path.join(os.path.dirname(_THIS_DIR), "FastHydro", "python")
if os.path.isdir(_FASTHYDRO) and _FASTHYDRO not in sys.path:
    sys.path.insert(0, _FASTHYDRO)

import hydro_jet_pyvista as hjp                                       # noqa: E402
import hydro_pyvista as hp                                            # noqa: E402

#: feature layout hydro_pyvista expects, which is NOT FastHydro's (e, vx, vy, vz)
FEATURES = ("e", "T", "vx", "vy", "vz")

#: Text sizes. A three-panel window is not three times a one-panel window with three
#: times the text: the viewports shrink and the fonts do not, so everything hydro_pyvista
#: sizes for a full window comes out a third too small here.
BAR_TITLE_FONT = 16
BAR_LABEL_FONT = 14
AXIS_FONT = 16
LABEL_FONT = 16
NOTE_FONT = 12

#: Per-panel viewport, width x height. Both divisible by 16 so ffmpeg's macro_block_size
#: does not resize and blur the frames. Wider than a third of hydro_pyvista's 1008 because
#: the axis titles and bar titles need the room at these font sizes.
PANEL_W, PANEL_H = 640, 864

#: The camera is fitted to the medium box, which puts the outer axis titles ("y [fm]")
#: just outside the viewport; pulling back this much brings them in without shrinking the
#: fireball noticeably.
CAMERA_ZOOM = 0.82

PANELS = {
    "bg":      ("medium, no deposit", "arr_bg"),
    "jet":     ("medium + deposit",   "arr"),
    "diff":    ("the wake",           "arr - arr_bg"),
    "reldiff": ("relative wake",      "(arr - arr_bg) / arr_bg"),
}

#: panels whose field is SIGNED: diverging colour map, symmetric limits, their own bar
SIGNED = ("diff", "reldiff")

#: printed under every figure, because the panel titles alone invite the wrong reading
SHOWER_NOTE = ("same quenched shower in all panels -- it traversed arr_bg "
               "(one-way coupling)")


# ──────────────────────────────────────────────────────────────────────────────
# Reading a pair
# ──────────────────────────────────────────────────────────────────────────────

def _meta_from_attrs(a, ntau):
    """hydro_pyvista's meta dict, from the file's root attributes.

    The names line up with EvolutionHistory's because FastHydro writes the FNO4d schema,
    whose scalar keys were themselves taken from MUSIC-produced files.
    """
    return {
        "tau_min": float(a["tau_min"]), "dtau": float(a["dtau"]),
        "x_min": float(a["x_min"]), "dx": float(a["dx"]),
        "y_min": float(a["y_min"]), "dy": float(a["dy"]),
        "eta_min": float(a["eta_min"]), "deta": float(a["deta"]),
        "ntau": int(ntau), "nx": int(a["nx"]), "ny": int(a["ny"]),
        "neta": int(a["neta"]),
        "boost_invariant": False,
        "has_eta": True,
    }


#: MUSIC's hotQCD tables (src/eos_hotQCD.cpp): rows of float64 (e, P, s, T), e in GeV/fm^3,
#: T in GeV.  EOS 91 is the SMASH hadron list, EOS 9 the UrQMD one.
_MUSIC_HOTQCD = {9: "hrg_hotqcd_eos_binary.dat", 91: "hrg_hotqcd_eos_SMASH_binary.dat"}


def _attr_str(a, key, default=""):
    v = a.get(key, default)
    return v.decode() if isinstance(v, bytes) else str(v)


def _conformal(dof):
    a_sb = np.pi ** 2 / 30.0 * dof / (0.1973269804 ** 3)      # GeV/fm^3 per GeV^4
    return lambda e: ((np.maximum(e, 0.0) / a_sb) ** 0.25).astype(np.float32)


def _music_table(eos_id, attrs, eos_table=None):
    """Path of MUSIC's hotQCD table for `eos_id`, or None.

    Looked for, in order, at --eos-table and $MUSIC_EOS_TABLE (each a file or a directory
    holding it), then under the producing build's EOS/hotQCD, which PyJetscape's
    productions record as the `prod_build` attribute.
    """
    fname = _MUSIC_HOTQCD[eos_id]
    cands = [eos_table, os.environ.get("MUSIC_EOS_TABLE")]
    build = _attr_str(attrs, "prod_build")
    if build:
        cands += [os.path.join(build, "EOS", "hotQCD"), os.path.join(build, "eos", "hotQCD")]
    for c in cands:
        if not c:
            continue
        path = os.path.join(c, fname) if os.path.isdir(c) else c
        if os.path.isfile(path):
            return path
    return None


def resolve_temperature(f, eos_table=None):
    """-> (T(e) callable, one-line description) for a pair file, from whatever it carries.

    The arrays hold (e, vx, vy, vz) -- no temperature -- while hydro_pyvista wants it for
    the freeze-out isosurface, and T(e) near T_fo depends strongly on the EoS: at
    T = 0.15 GeV a conformal gas (dof 42.25) has e = 0.92 GeV/fm^3, the hotQCD table 0.23.
    So the EoS is taken from the file, in this order:

    1. an `eos/` group with a table (`e_tab`, `T_tab`): FastHydro writes one, so the file
       is self-contained;
    2. an `eos/` group of kind "ideal": conformal with the group's `dof`;
    3. the `eos_kind` attribute naming MUSIC's hotQCD EoS (PyJetscape's MUSIC pairs
       write "hotqcd (MUSIC EOS 9)"): MUSIC's own table, found by `_music_table`;
    4. otherwise a conformal gas with dof 42.25, and a warning, since a silently wrong
       isosurface is worse than a coarse one.
    """
    if "eos/e_tab" in f and "eos/T_tab" in f:
        e_tab, T_tab = f["eos/e_tab"][:], f["eos/T_tab"][:]
        order = np.argsort(e_tab)
        e_tab, T_tab = e_tab[order], T_tab[order]
        name = _attr_str(f["eos"].attrs, "name", "table")
        return (lambda e: np.interp(e, e_tab, T_tab).astype(np.float32),
                f"the file's eos/ table ({name})")
    if "eos" in f and _attr_str(f["eos"].attrs, "kind") == "ideal":
        dof = float(f["eos"].attrs.get("dof", 42.25))
        return _conformal(dof), f"conformal, dof = {dof:g} (the file's eos/ group)"

    kind = _attr_str(f.attrs, "eos_kind").lower()
    if kind.startswith("hotqcd"):
        eos_id = 91 if "smash" in kind or "91" in kind else 9
        path = _music_table(eos_id, f.attrs, eos_table)
        if path is not None:
            tab = np.fromfile(path, dtype="<f8").reshape(-1, 4)
            tab = tab[np.argsort(tab[:, 0])]
            e_tab, T_tab = tab[:, 0], tab[:, 3]
            return (lambda e: np.interp(e, e_tab, T_tab).astype(np.float32),
                    f"MUSIC EOS {eos_id} table {path}")
        print(f"  [!] eos_kind is '{_attr_str(f.attrs, 'eos_kind')}' but MUSIC's "
              f"{_MUSIC_HOTQCD[eos_id]} was not found; pass --eos-table (file or "
              f"MUSIC's EOS/hotQCD directory) or set MUSIC_EOS_TABLE.")

    dof = 42.25                                    # fast_data's conformal default
    print("  [!] no EoS table for this file; temperature from a conformal EoS "
          f"(dof={dof}). The freeze-out isosurface is indicative only.")
    return _conformal(dof), f"conformal, dof = {dof:g} (fallback)"


def _temperature(e, f, eos_table=None):
    """T [GeV] on the energy-density grid; see `resolve_temperature`."""
    return resolve_temperature(f, eos_table)[0](e)


def live_frames(f, event, ntau, panels=("bg", "jet", "diff")):
    """-> (frames to show, frames the jet leg wrote, frames the background wrote).

    Past a leg's freeze-out its frames are zero-padded.  The two legs freeze out at
    different times -- the deposit reheats the jet leg, by 1.1 fm/c on a 0-10% Au+Au MUSIC
    pair -- so past the background's freeze-out `arr - arr_bg` is no longer a wake but the
    jet leg's whole medium, and the background panel is empty.  Every panel that uses the
    background therefore stops at the last frame where BOTH legs are live; a jet-only
    render runs to the jet leg's own end.

    `ntau_freezeout` has two conventions: PyJetscape writes the number of frames written
    and stamps `freezeout_convention_id = "frames_written"`; fast_data/FastHydro files
    carry no id and store that number plus one.  A file without the counts (a bare
    fixture) shows every frame.
    """
    off = 0 if _attr_str(f.attrs, "freezeout_convention_id") == "frames_written" else 1

    def count(key):
        if key not in f:
            return None
        return max(int(f[key][event]) - off, 1)

    n_jet = count("ntau_freezeout") or ntau
    n_bg = count("ntau_freezeout_bg") or n_jet
    n_jet, n_bg = min(n_jet, ntau), min(n_bg, ntau)
    uses_bg = any(p in panels for p in ("bg", "diff", "reldiff"))
    return (min(n_jet, n_bg) if uses_bg else n_jet), n_jet, n_bg


def relative_floor(e_bg, frac=0.1, absolute=0.0):
    """-> (ntau,1,1,1) density below which `de/e` is not worth forming, per FRAME.

    Per frame, not globally, because the fireball cools by two orders of magnitude over
    a run: the peak is 28 GeV/fm^3 at tau = 0.6 and 0.72 by tau = 6.3.  A single global
    floor set high enough to be meaningful early blanks the whole panel after mid
    evolution; one set low enough to keep it alive late is no floor at all when it
    matters.  A fraction of each frame's own peak tracks "where the medium is still
    dense NOW", which is the question the ratio is asking.

    `absolute` is a hard GeV/fm^3 floor applied on top, for "only where e > 1" cuts.
    """
    peak = np.asarray(e_bg, np.float32).reshape(len(e_bg), -1).max(axis=1)
    return np.maximum(float(frac) * peak, float(absolute))[:, None, None, None]


def load_pair(path, event=0, panels=("bg", "jet", "diff"),
              rel_floor=0.1, rel_floor_abs=0.0, eos_table=None):
    """-> (dict of panel name -> (ntau,nx,ny,neta,5) array, meta, attrs).

    The arrays are built in hydro_pyvista's feature order (e, T, vx, vy, vz), and hold
    only the frames `live_frames` allows.  `meta` also records `ntau_jet`, `ntau_bg`
    (frames each leg wrote), `ntau_file` and `eos` (where the temperature came from).
    """
    import h5py

    out, attrs = {}, {}
    with h5py.File(str(path), "r") as f:
        if "arr_bg" not in f:
            raise SystemExit(
                f"{path} has no 'arr_bg', so it is not a paired file -- there is no "
                f"no-jet reference to subtract. Produce one with "
                f"FastHydro/example/make_wake_data.py, or use hydro_pyvista.py for a "
                f"single evolution.")
        nev = int(f["arr"].shape[0])
        if not 0 <= event < nev:
            raise SystemExit(f"event {event} out of range (file holds {nev})")

        attrs = {k: f.attrs[k] for k in f.attrs}
        ntau_file = int(f["arr"].shape[-1])
        n, n_jet, n_bg = live_frames(f, event, ntau_file, panels)
        # (4, nx, ny, neta, ntau) -> (ntau, nx, ny, neta, 4)
        jet = np.asarray(f["arr"][event, ..., :n], np.float32).transpose(4, 1, 2, 3, 0)
        bg = np.asarray(f["arr_bg"][event, ..., :n], np.float32).transpose(4, 1, 2, 3, 0)
        meta = _meta_from_attrs(f.attrs, jet.shape[0])
        T_of_e, eos_desc = resolve_temperature(f, eos_table)
        meta.update(ntau_file=ntau_file, ntau_jet=n_jet, ntau_bg=n_bg, eos=eos_desc)

        def assemble(e, v_from):
            a = np.empty(e.shape + (5,), np.float32)
            a[..., 0] = e
            a[..., 1] = T_of_e(e)
            a[..., 2:5] = v_from[..., 1:4]
            return a

        if "bg" in panels:
            out["bg"] = assemble(bg[..., 0], bg)
        if "jet" in panels:
            out["jet"] = assemble(jet[..., 0], jet)
        if "diff" in panels:
            # Only the energy density is differenced. A difference of temperatures or of
            # flow velocities is not a physical field, and the isosurface/glyphs on this
            # panel would be meaningless -- so it carries the JET leg's T and v, and the
            # renderer switches both off for this panel by default.
            d = assemble(jet[..., 0] - bg[..., 0], jet)
            d[..., 1] = T_of_e(jet[..., 0])
            out["diff"] = d
        if "reldiff" in panels:
            # de/e, masked where the background is too thin for the ratio to mean
            # anything. Masked cells are set to 0, not NaN: the volume mapper renders
            # NaN as a hole in the data rather than as "no wake here".
            #
            # The mask is applied HERE, on the Milne grid, and not after resampling --
            # the Cartesian resampler interpolates, and interpolating across the mask
            # edge would smear the dilute-tail values it exists to exclude back in.
            e, de = bg[..., 0], jet[..., 0] - bg[..., 0]
            floor = relative_floor(e, rel_floor, rel_floor_abs)
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.where(e > floor, de / np.maximum(e, 1e-30), 0.0)
            rd = assemble(np.asarray(r, np.float32), jet)
            rd[..., 1] = T_of_e(jet[..., 0])
            out["reldiff"] = rd
    return out, meta, attrs


def load_shower(path, event=0, min_energy=0.0):
    """-> `make_jet_overlay`'s segment dict, from the file's `shower/` group, or None.

    Built on `fasthydro.showers.segments`, so it inherits the three corrections that
    module documents: vertex positions are ignored (X-SCAPE stores them all at the
    origin), backwards hole edges are not mistaken for splittings, and a parton with no
    splitting is free-streamed rather than frozen.
    """
    import h5py

    from fasthydro.showers import ABSORBED, segments, velocities

    with h5py.File(str(path), "r") as f:
        if "shower/partons" not in f:
            return None
        po = f["shower/parton_offsets"][:]
        vo = f["shower/vertex_offsets"][:]
        par = f["shower/partons"][po[event]:po[event + 1]]
        ver = f["shower/vertices"][vo[event]:vo[event + 1]]
    if not len(par):
        return None

    start, end, splits = segments(par, ver)
    # |E|, not E: a negative ("hole") parton carries negative energy by construction, so a
    # plain `E >= min_energy` silently drops every one of them even at the default 0.
    keep = np.abs(par[:, 8]) >= float(min_energy)
    if not keep.any():
        return None
    par, start, end, splits = par[keep], start[keep], end[keep], splits[keep]

    p = par[:, 5:8]
    norm = np.linalg.norm(p, axis=1, keepdims=True)
    dirs = np.where(norm > 1e-9, p / np.where(norm > 1e-9, norm, 1.0), [0.0, 0.0, 1.0])

    # A parton that never splits keeps going -- UNLESS the liquefier took it: pstat -11
    # `drop` is a parton absorbed into the medium, and its energy is in source/droplets
    # from that moment. Free-streaming it onward would draw the jet twice, once as a
    # parton and once as the wake it turned into.
    absorbed = np.isin(par[:, 4].astype(int), ABSORBED)
    return dict(
        starts=start[:, :3], ends=end[:, :3],
        t0=start[:, 3], t1=np.where(splits, end[:, 3], start[:, 3]),
        energy=par[:, 8], pT=np.hypot(p[:, 0], p[:, 1]),
        dirs=dirs, vel=velocities(par),
        is_leaf=~splits & ~absorbed)


def leading_parton(path, event=0):
    """-> (pT, E, pid) of the hardest SHOWER-INITIATING parton, or None.

    The initiators are what JetScape handed to JetEnergyLoss, i.e. the hard partons
    before any quenching -- so this is "the leading parton" in the sense a jet analysis
    means it.  Deliberately not the maximum over `shower/partons`, which is the same
    parton a step later, after Matter has already taken some of its energy: on the wake
    event that reads 50.9 GeV against the initiator's 54.6 GeV.
    """
    import h5py

    with h5py.File(str(path), "r") as f:
        if "shower/initiators" not in f:
            return None
        o = f["shower/initiator_offsets"][:]
        ini = f["shower/initiators"][o[event]:o[event + 1]]
    if not len(ini):
        return None
    pT = np.hypot(ini[:, 3], ini[:, 4])            # px, py
    i = int(np.argmax(pT))
    return float(pT[i]), float(ini[i, 6]), int(ini[i, 1])


# ──────────────────────────────────────────────────────────────────────────────
# Rendering
# ──────────────────────────────────────────────────────────────────────────────

def _clim(frames_by_panel, name, pct=99.9):
    """Colour limits: [0, max] for a leg, symmetric and PERCENTILE-scaled for the difference.

    The difference is not scaled to its maximum on purpose.  Measured on a central Au+Au
    event, max|de| over the whole evolution is 2.36 GeV/fm^3 -- but that is one spike in
    one frame at tau = 1.6, where the first droplets land.  The wake that follows it runs
    at 0.1-0.3, so a max-scaled colour map renders the entire thing the script exists to
    show at a few percent of full scale, i.e. invisible.  The 99.9th percentile of the
    non-zero cells clips that one deposit spike and leaves the wake legible; `--diff-clim`
    overrides it when the saturation matters more than the contrast.
    """
    fr = frames_by_panel[name]
    if name not in SIGNED:
        m = max((f["e"].max() for f in fr), default=0.0)
        return (0.0, float(m) if m > 0 else 1.0)

    vals = np.concatenate([np.abs(f["e"])[np.abs(f["e"]) > 0].ravel() for f in fr]) \
        if fr else np.zeros(0, np.float32)
    if not vals.size:
        return (-1.0, 1.0)
    m = float(np.percentile(vals, pct))
    return (-m, m) if m > 0 else (-float(vals.max()), float(vals.max()))


#: opacity ramp for the signed difference: transparent at zero, opaque at both extremes,
#: so a negative (depleted) region is as visible as the positive wake front.
DIVERGING_OPACITY = [0.95, 0.7, 0.35, 0.0, 0.35, 0.7, 0.95]


def _panel_args(args, name):
    """A shallow copy of `args` with the per-panel overrides `_add_frame_actors` reads."""
    import copy

    a = copy.copy(args)
    if name in SIGNED:
        a.cmap = args.diff_cmap
        # T and v belong to the jet leg on this panel, not to the difference; drawing
        # them here would put a freeze-out surface around a field that has none.
        a.field = "e"
        a.velocity = False
    return a


def render(panel_arrays, meta, args, seg=None, event_id=0, lead=None):
    """Resample every panel and emit the movie / VTK series / interactive window."""
    import pyvista as pv

    names = [n for n in args.panels if n in panel_arrays]
    tau_max = meta["tau_min"] + (meta["ntau"] - 1) * meta["dtau"]
    t_min = args.t_min if args.t_min is not None else meta["tau_min"]
    t_max = args.t_max if args.t_max is not None else tau_max
    nt = args.nt if args.nt is not None else min(meta["ntau"], 50)
    ts = np.linspace(t_min, t_max, max(1, nt))

    ref = panel_arrays.get("jet", panel_arrays[names[0]])
    xy_max = args.xy_max if args.xy_max is not None else hp.medium_xy_max(ref, meta)
    axes = hp.cartesian_axes(meta, args, t_max, xy_max)

    print(f"  resampling {len(ts)} lab-time frames x {len(names)} panel(s) onto a "
          f"({len(axes[0])}, {len(axes[1])}, {args.nz}) Cartesian grid "
          f"(x,y in ±{xy_max:.1f} fm, t in [{t_min:.2f}, {t_max:.2f}] fm/c) ...")
    frames, t0 = {}, time.perf_counter()
    for n in names:
        frames[n] = hp.resample_frames(
            hp.build_interpolator(panel_arrays[n], meta), ts, axes, meta,
            args.velocity and n not in SIGNED, max(1, min(args.jobs, len(ts))),
            oversample_z=int(getattr(args, "z_oversample", 1)))
        print(f"    {n:5s} done")
    print(f"  resampled in {time.perf_counter() - t0:.1f} s")

    max_pT = 1.0
    if seg is not None and len(seg["pT"]):
        max_pT = float(seg["pT"].max())

    clims = {n: _clim(frames, n, args.diff_pct) for n in names}
    for key, override in (("diff", args.diff_clim), ("reldiff", args.rel_clim)):
        if override is not None and key in clims:
            clims[key] = (-abs(override), abs(override))
    # The two legs MUST share a colour scale. The whole claim of the figure is that the
    # left and middle panels look alike and the wake only shows up in the subtraction;
    # two independently-scaled panels would make that claim untestable by eye.
    legs = [n for n in names if n not in SIGNED]
    if len(legs) > 1:
        top = max(clims[n][1] for n in legs)
        for n in legs:
            clims[n] = (0.0, top)
    # Each signed panel carries its own bar (their units differ); the legs share one,
    # drawn on the rightmost of them.
    bar_on = {"panels": {n for n in names if n in SIGNED} | ({legs[-1]} if legs else set()),
              "max_pT": max_pT,
              "lead": lead}
    for n in names:
        extra = ""
        if n in SIGNED:
            peak = max((np.abs(f["e"]).max() for f in frames[n]), default=0.0)
            if peak > clims[n][1]:
                flag = "--rel-clim" if n == "reldiff" else "--diff-clim"
                extra = (f"   (peak {peak:.3g}, saturated above "
                         f"{clims[n][1]:.3g}; {flag} to change)")
        unit = "" if n == "reldiff" else " GeV/fm^3"
        print(f"  {PANELS[n][0]:18s} ({PANELS[n][1]:>23s})  clim = "
              f"[{clims[n][0]:.4g}, {clims[n][1]:.4g}]{unit}{extra}")

    # A per-panel actor-name suffix; the pT bar is added once by _add_pt_bar instead
    # (hjp's own bar geometry is tuned for a full-window single panel).
    overlays = {n: (hjp.make_jet_overlay(seg, args, max_pT, t_max, suffix="_" + n,
                                         colorbar=False)
                    if seg is not None else None) for n in names}

    os.makedirs(args.outdir, exist_ok=True)
    if args.interactive:
        _interactive(names, frames, axes, ts, args, clims, overlays, event_id, bar_on)
        return
    _movie(names, frames, axes, ts, args, clims, overlays, event_id, bar_on)


def _bar_args(title, fmt="%.2f", x=0.84):
    """A vertical bar inside the panel's own viewport.

    Scalar-bar positions are fractions of the VIEWPORT, so the single-panel defaults --
    tuned against a full window -- land on the axis labels here, and a title centred on a
    bar near an edge is clipped rather than wrapped. Hence explicit geometry, a shorter
    title, and a smaller title font than hydro_pyvista uses.
    """
    return dict(title=title, color="white", title_font_size=BAR_TITLE_FONT,
                label_font_size=BAR_LABEL_FONT, n_labels=5, fmt=fmt, vertical=True,
                position_x=x, position_y=0.16, width=0.055, height=0.58)


#: shorter than hydro_pyvista's E_UNITS, which is clipped at a third of the window width
E_TITLE = "e [GeV/fm3]"
DIFF_TITLE = "de [GeV/fm3]"
REL_TITLE = "de/e"          # dimensionless: 1.0 means the wake doubled the local density
PT_TITLE = "parton pT [GeV]"


def _add_pt_bar(plotter, max_pT, cmap, name="pt_bar"):
    """The parton-pT bar, at the right of the first panel -- same place as every other bar.

    hjp puts it on the LEFT to clear the energy bar, which works in a full-width window.
    Here the left edge is where show_grid draws the y tick labels and the "y [fm]" title,
    so the bar lands on top of them; the right edge is clear because each panel carries at
    most one bar. Drawn here rather than through hjp._add_jet_colorbar for the same reason
    its geometry is set locally: that one is tuned for a single full-window panel.
    """
    import pyvista as pv

    proxy = pv.PolyData(np.zeros((2, 3)))
    proxy["pT"] = np.array([0.0, max_pT], dtype=float)
    plotter.add_mesh(proxy, scalars="pT", cmap=cmap, clim=(0.0, max_pT), opacity=0.0,
                     name=name, reset_camera=False, show_scalar_bar=True,
                     scalar_bar_args=_bar_args(PT_TITLE, "%.0f"))


def _draw(plotter, names, frames, axes, args, clims, overlays, ti, t, event_id, bar_on):
    """One frame, all panels.

    Deliberately not `hp._add_frame_actors`: that draws one panel's worth of defaults --
    its own colour bar geometry, its own actor names, its own single opacity ramp -- and
    the difference panel needs a signed ramp while the two legs need to share one bar.
    """
    for k, n in enumerate(names):
        plotter.subplot(0, k)
        grid = hp.make_image_data(frames[n][ti], axes)
        pa = _panel_args(args, n)
        show_bar = n in bar_on["panels"]

        if clims[n][1] > clims[n][0]:
            kw = dict(scalars="e", cmap=pa.cmap, clim=clims[n], name="vol_" + n,
                      reset_camera=False, show_scalar_bar=show_bar,
                      opacity=(DIVERGING_OPACITY if n in SIGNED else hp.VOLUME_OPACITY))
            if show_bar:
                title, fmt = {"diff": (DIFF_TITLE, "%.2g"),
                              "reldiff": (REL_TITLE, "%.2g")}.get(n, (E_TITLE, "%.2f"))
                kw["scalar_bar_args"] = _bar_args(title, fmt)
            plotter.add_volume(grid, **kw)

        if pa.field in ("T", "both"):
            contour = grid.contour(pa.freeze_temp, scalars="T")
            if contour.n_points:
                plotter.add_mesh(contour, color="deepskyblue", opacity=0.30,
                                 name="cont_" + n, smooth_shading=True,
                                 reset_camera=False, show_scalar_bar=False)
        if pa.velocity and "v" in grid.point_data:
            active = grid.threshold(max(1e-3, 0.02 * clims[n][1]), scalars="e")
            if active.n_points:
                active.set_active_vectors("v")
                plotter.add_mesh(active.glyph(orient="v", scale=False, factor=0.6,
                                              tolerance=0.04),
                                 color="white", opacity=0.7, name="vel_" + n,
                                 reset_camera=False, show_scalar_bar=False)

        label = f"{PANELS[n][0]}   ({PANELS[n][1]})\n"
        if k == 0:
            label += hp._frame_label(event_id, t)
            if bar_on.get("lead"):
                pT, E, _pid = bar_on["lead"]
                label += f"\nleading parton  pT = {pT:.1f} GeV,  E = {E:.1f} GeV"
        else:
            label += f"t = {t:6.2f} fm/c"
        plotter.add_text(label, name="label_" + n, position="upper_left",
                         font_size=LABEL_FONT, color="white", shadow=True)
        if k == 0 and any(o is not None for o in overlays.values()):
            # Without this the titles read as "a run with no jet" vs "a run with a jet",
            # which is not what the panels are: there is one shower and it is in all three.
            plotter.add_text(SHOWER_NOTE, name="shower_note", position="lower_left",
                             font_size=NOTE_FONT, color="#9fb6c4", shadow=False)
        if overlays[n] is not None:
            overlays[n](plotter, t)


def _movie(names, frames, axes, ts, args, clims, overlays, event_id, bar_on):
    import pyvista as pv

    movie = (args.movie if os.path.isabs(args.movie)
             else os.path.join(args.outdir, args.movie))
    hp._maybe_start_xvfb(off_screen=True)
    # width divisible by 16 per panel keeps ffmpeg from resizing (macro_block_size)
    plotter = pv.Plotter(off_screen=True, shape=(1, len(names)),
                         window_size=(PANEL_W * len(names), PANEL_H), border=False)
    fps = (1.0 / args.frame_duration) if args.frame_duration else float(args.framerate)
    if movie.lower().endswith(".mp4"):
        try:
            plotter.open_movie(movie, framerate=max(1, int(round(fps))))
        except Exception as exc:                                        # noqa: BLE001
            movie = os.path.splitext(movie)[0] + ".gif"
            print(f"  [!] mp4 unavailable ({exc}); falling back to {movie}")
            plotter.open_gif(movie, fps=fps)
    else:
        plotter.open_gif(movie, fps=fps)

    bounds = hp._scene_bounds(axes)
    for k in range(len(names)):
        plotter.subplot(0, k)
        hp._decorate_scene(plotter, bounds, font_size=AXIS_FONT)
        hp._beam_camera(plotter, args.azimuth, args.elevation)
        plotter.camera.zoom(CAMERA_ZOOM)
    plotter.link_views()                       # one camera: the panels stay comparable
    if overlays[names[0]] is not None and not args.jet_color:
        plotter.subplot(0, 0)
        _add_pt_bar(plotter, bar_on['max_pT'], args.jet_cmap)

    for ti, t in enumerate(ts):
        _draw(plotter, names, frames, axes, args, clims, overlays, ti, t, event_id,
              bar_on)
        plotter.write_frame()
        print(f"\r  frame {ti + 1}/{len(ts)}  t = {t:6.2f} fm/c", end="", flush=True)
    print()
    plotter.close()
    print(f"  wrote {movie}")


def _interactive(names, frames, axes, ts, args, clims, overlays, event_id, bar_on):
    import pyvista as pv

    plotter = pv.Plotter(shape=(1, len(names)),
                         window_size=(PANEL_W * len(names), PANEL_H), border=False)
    bounds = hp._scene_bounds(axes)
    for k in range(len(names)):
        plotter.subplot(0, k)
        hp._decorate_scene(plotter, bounds, font_size=AXIS_FONT)
        hp._beam_camera(plotter, args.azimuth, args.elevation)
        plotter.camera.zoom(CAMERA_ZOOM)
    plotter.link_views()
    if overlays[names[0]] is not None and not args.jet_color:
        plotter.subplot(0, 0)
        _add_pt_bar(plotter, bar_on['max_pT'], args.jet_cmap)

    state = {"i": 0}

    def on_slide(value):
        i = int(np.clip(np.searchsorted(ts, value), 0, len(ts) - 1))
        if i == state["i"]:
            return
        state["i"] = i
        _draw(plotter, names, frames, axes, args, clims, overlays, i, ts[i],
              event_id, bar_on)

    _draw(plotter, names, frames, axes, args, clims, overlays, 0, ts[0], event_id,
          bar_on)
    plotter.subplot(0, 0)
    plotter.add_slider_widget(on_slide, [float(ts[0]), float(ts[-1])],
                              value=float(ts[0]), title="t  [fm/c]",
                              style="modern", fmt="%.2f")
    plotter.show()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_parser():
    p = hjp.build_parser()
    p.description = __doc__
    g = p.add_argument_group("wake panels")
    g.add_argument("--file", required=True,
                   help="a paired HDF5 file, jet leg `arr` + background `arr_bg` "
                        "(FastHydro, or PyJetscape's MUSIC pairs)")
    g.add_argument("--eos-table", default=None, dest="eos_table",
                   help="MUSIC hotQCD table (file, or MUSIC's EOS/hotQCD directory) for a "
                        "file that names the EoS in `eos_kind` but carries no eos/ group; "
                        "by default $MUSIC_EOS_TABLE, then <prod_build>/EOS/hotQCD")
    g.add_argument("--event", type=int, default=0)
    g.add_argument("--panels", default="bg,jet,diff",
                   help="comma-separated subset of bg,jet,diff,reldiff in display order "
                        "(default bg,jet,diff). 'reldiff' is de/e, floored by "
                        "--rel-floor; it is off by default because the floor is a "
                        "judgement call that the absolute difference does not need")
    g.add_argument("--diff-cmap", default="coolwarm", dest="diff_cmap",
                   help="diverging colormap for the difference panel (default coolwarm)")
    g.add_argument("--diff-clim", type=float, default=None, dest="diff_clim",
                   help="symmetric colour limit for the difference panel in GeV/fm^3; "
                        "by default the 99.9th percentile of the non-zero cells, which "
                        "clips the initial deposit spike so the wake stays visible")
    g.add_argument("--diff-pct", type=float, default=99.9, dest="diff_pct",
                   help="percentile used for that automatic limit (default 99.9); "
                        "applies to both signed panels")
    g.add_argument("--rel-clim", type=float, default=None, dest="rel_clim",
                   help="symmetric colour limit for the relative panel (dimensionless)")
    g.add_argument("--rel-floor", type=float, default=0.1, dest="rel_floor",
                   help="on the 'reldiff' panel, blank cells whose background density is "
                        "below this fraction of THAT FRAME's peak (default 0.1). de/e is "
                        "meaningless where e -> 0: unfloored it peaks at 3.0 in a cell "
                        "with e = 0.067 GeV/fm^3. 0 disables the floor")
    g.add_argument("--rel-floor-abs", type=float, default=0.0, dest="rel_floor_abs",
                   help="additional hard floor in GeV/fm^3 for the 'reldiff' panel "
                        "(default 0); the effective floor is the larger of the two")
    g.add_argument("--no-jet", action="store_true", dest="no_jet",
                   help="do not overlay the parton shower")
    p.set_defaults(movie="wake.mp4", t_min=0.0)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.panels = [s.strip() for s in args.panels.split(",") if s.strip()]
    bad = [n for n in args.panels if n not in PANELS]
    if bad:
        raise SystemExit(f"unknown panel(s) {bad}; choose from {sorted(PANELS)}")

    print(f"=== {args.file}  event {args.event} ===")
    panel_arrays, meta, attrs = load_pair(args.file, args.event, tuple(args.panels),
                                          rel_floor=args.rel_floor,
                                          rel_floor_abs=args.rel_floor_abs,
                                          eos_table=args.eos_table)
    if "reldiff" in args.panels:
        print(f"  relative panel: de/e where e_bg > max({args.rel_floor:g} x the frame "
              f"peak, {args.rel_floor_abs:g} GeV/fm^3); elsewhere drawn as zero")
    print(f"  grid {meta['nx']}x{meta['ny']}x{meta['neta']}x{meta['ntau']}, "
          f"tau {meta['tau_min']:.2f}.."
          f"{meta['tau_min'] + (meta['ntau'] - 1) * meta['dtau']:.2f} fm/c")
    if meta["ntau"] < max(meta["ntau_jet"], meta["ntau_bg"]):
        print(f"  frames: jet leg {meta['ntau_jet']}, background {meta['ntau_bg']}; showing "
              f"the {meta['ntau']} where both are live (past the background's freeze-out "
              f"arr - arr_bg would be the jet leg's whole medium)")
    print(f"  temperature from {meta['eos']}")
    for k in ("generator", "source_mode", "transport_mode", "hard_vertex"):
        if k in attrs:
            print(f"  {k:15s} {attrs[k]}")

    seg = None
    if not args.no_jet:
        seg = load_shower(args.file, args.event, args.jet_min_energy)
        if seg is None:
            print("  [!] no shower/ group in this file (or no parton passed "
                  "--jet-min-energy); rendering the medium only. Regenerate with a "
                  "FastHydro build that stores showers to get the overlay.")
        else:
            print(f"  shower: {len(seg['starts'])} partons, "
                  f"{int((~seg['is_leaf']).sum())} ending in a splitting or the medium, "
                  f"max pT {seg['pT'].max():.1f} GeV")

    render(panel_arrays, meta, args, seg=seg, event_id=args.event,
           lead=leading_parton(args.file, args.event))
    return 0


if __name__ == "__main__":
    sys.exit(main())
