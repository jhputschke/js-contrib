#!/usr/bin/env python3
"""
example/hydro_hist_vs_surface/make_surfaces.py

Freeze-out surfaces for the hydro-history vs full-surface study (README.md), written as
particlize files that ``prod_AuAu_0_10_jet/hadronize.py`` hadronizes unchanged.

    python make_surfaces.py hist      DATA/*_particlize.h5 --out-dir DATA/hist_vs_surface/hist
    python make_surfaces.py ref_ideal DATA/*_particlize.h5 --out-dir DATA/hist_vs_surface/ref_ideal
    python make_surfaces.py ref_no_bulk  ...      # Pi = 0 only
    python make_surfaces.py ref_no_shear ...      # pi^{mu nu} = 0 only

The inputs are the production's reference particlize files (``run_prod_jet.py
--write-particlize both``): MUSIC's own freeze-out surfaces with every field iSS reads,
including pi^{mu nu} and Pi.  Each variant replaces the surfaces and keeps everything else
(final partons, events/, music_input, file_uuid):

    hist        the surface a hadronization would get from the SAVED HYDRO HISTORY alone:
                the pair file's arr (jet leg) and arr_bg (background), i.e. e, vx, vy, vz on
                the output grid (0.3125 fm, 0.3125 in eta, 0.1 fm/c), no viscous fields.
                T and P from MUSIC's EoS table (EOS 9, hotQCD), then X-SCAPE's SurfaceFinder
                (Cornelius, FluidDynamics::FindSurfaceFromEvolution) at T_sw, then
                pi^{mu nu} = Pi = 0: ideal Cooper-Frye.
    ref_ideal   MUSIC's surface with pi^{mu nu} = Pi = 0: the same surface as the reference,
                without the viscous (delta f) corrections.  hist - ref_ideal is the effect of
                the coarse history alone, ref_ideal - ref that of delta f alone.
    ref_no_bulk, ref_no_shear
                MUSIC's surface with only Pi = 0, or only pi^{mu nu} = 0: which of the two
                viscous corrections makes the difference.

Why the file_uuid is kept.  hadronize.py derives every unit's iSS seed from (--seed, the
particlize file's file_uuid, tag, unit).  With the reference's uuid, the variants draw the
same random numbers as the reference, and with --correlated (random numbers addressed by
the space-time cell block) they give the same hadrons wherever their surfaces agree: the
difference variant - reference is then much less noisy than either.  --new-uuid switches
this off (independent samples).

The hist surface, in detail:
  * Every event's jet leg (arr[e]) and every NEW background (arr_bg of events/bg_id) are
    handed to a FluidDynamics (store_fluid_cells_from_numpy_3d: temperature, energy
    density, pressure, vx, vy, vz) and SurfaceFinder runs on the Cornelius lattice
    --lattice (default: the output grid itself).  Frames after the leg's freeze-out
    (ntau_freezeout) are zero and are dropped, except one: the surface finder then covers
    exactly the frames the leg was written for.
  * e and P of each cell are recomputed from its interpolated T through the EoS (the
    SurfaceFinder interpolates e and T separately, which leaves e off the T_sw isotherm by
    up to 60%); the charges and chemical potentials, which the finder leaves uninitialised,
    are set to 0.
  * --cap (default on) adds MUSIC's Do_FreezeOut_lowtemp elements at the first frame: every
    grid point with 0.05 GeV/fm^3 < e < e(T_sw) freezes out on a tau = const element
    d^3sigma = (dx dy deta, 0, 0, 0) (MUSIC's evolve.cpp FreezeOut_equal_tau_Surface_XY).
    MUSIC does this at its first freeze-out step; the history starts later (tau_min of the
    output grid), so the hist cap sits there.
  * Diagnostics per event go to events/hist_* (see the README): the largest T on the
    transverse and eta edges of the grid (an open surface there), and in the last kept frame.

Run with the environment pyjetscape_core was built in (js_fno on the GB10).  The surface
finder is OpenMP-parallel: ~35 s per leg and event on the GB10's 20 cores, so one process
at a time with all threads is the fastest.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid as uuidlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PYJETSCAPE = os.path.abspath(os.path.join(HERE, "..", ".."))
XSCAPE = os.path.abspath(os.path.join(PYJETSCAPE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(PYJETSCAPE, "python"))

VARIANTS = ("hist", "ref_ideal", "ref_no_bulk", "ref_no_shear")
DEFAULT_EOS = os.path.join(XSCAPE, "build_gpu", "EOS", "hotQCD", "hrg_hotqcd_eos_binary.dat")
#: MUSIC's lower cut of the low-temperature freeze-out (evolve.cpp: epsFO_low = 0.05/hbarc)
E_CAP_LOW = 0.05

COL = None           # column name -> index, filled from SURFACE_COLUMNS
SHEAR = ("pi00", "pi01", "pi02", "pi03", "pi11", "pi12", "pi13", "pi22", "pi23", "pi33")
BULK = ("Pi",)
VISCOUS = SHEAR + BULK
#: the columns each reference variant sets to zero
ZEROED = {"ref_ideal": VISCOUS, "ref_no_bulk": BULK, "ref_no_shear": SHEAR}
CHARGES = ("nB", "nQ", "nS", "muB", "muQ", "muS")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("variant", choices=VARIANTS)
    p.add_argument("inputs", nargs="+",
                   help="reference *_particlize.h5 files, or directories holding them")
    p.add_argument("--out-dir", required=True, dest="out_dir")
    p.add_argument("--events", default=None,
                   help="event range a:b (python slice; default all). The backgrounds "
                        "these events use are written as well")
    p.add_argument("--T-sw", type=float, default=None, dest="T_sw",
                   help="switching temperature [GeV] (default: the file's T_fo, 0.15)")
    p.add_argument("--lattice", default=None, metavar="DTAU,DX,DETA",
                   help="Cornelius lattice for hist (default: the output grid's spacing)")
    p.add_argument("--no-cap", action="store_false", dest="cap",
                   help="hist: no low-temperature cap at the first frame (see above)")
    p.add_argument("--eos", default=DEFAULT_EOS,
                   help=f"MUSIC's EOS 9 table (default {DEFAULT_EOS})")
    p.add_argument("--new-uuid", action="store_true", dest="new_uuid",
                   help="give the output a new file_uuid: hadronize.py then samples it "
                        "independently of the reference (default: the reference's uuid, "
                        "same seeds)")
    p.add_argument("--force", action="store_true", help="overwrite existing outputs")
    p.add_argument("--skip-complete", action="store_true", dest="skip_complete",
                   help="skip inputs whose output exists and is complete")
    return p.parse_args(argv)


# ── EoS ─────────────────────────────────────────────────────────────────────────
class Eos:
    """MUSIC's hotQCD table (EOS 9): rows (e, P, s, T), e and P in GeV/fm^3, T in GeV,
    interpolated linearly as MUSIC's own table lookup does (uniform e steps)."""

    def __init__(self, path):
        if not os.path.exists(path):
            sys.exit(f"make_surfaces.py: EoS table {path} not found (--eos). It is the file "
                     "MUSIC reads for EOS 9: <build>/EOS/hotQCD/hrg_hotqcd_eos_binary.dat")
        tab = np.fromfile(path, dtype="<f8")
        if tab.size == 0 or tab.size % 4:
            sys.exit(f"make_surfaces.py: {path} is not a MUSIC hotQCD table")
        self.path = path
        self.e, self.P, self.s, self.T = tab.reshape(-1, 4).T
        if np.any(np.diff(self.e) <= 0) or np.any(np.diff(self.T) <= 0):
            sys.exit(f"make_surfaces.py: {path}: e and T must increase along the table")

    def T_of_e(self, e):
        T = np.interp(e, self.e, self.T)
        return np.where(e > 0, T, 0.0)

    def P_of_e(self, e):
        # below the first point (e ~ 4e-4) P/e is held; above the last, P is extrapolated
        # linearly -- neither occurs at T_sw
        P = np.interp(e, self.e, self.P)
        low = e < self.e[0]
        return np.where(low, np.clip(e, 0, None) * self.P[0] / self.e[0], P)

    def e_of_T(self, T):
        return np.interp(T, self.T, self.e)


# ── surfaces ────────────────────────────────────────────────────────────────────
def milne_u(vx, vy, vz, eta):
    """u^mu in Milne components (u^tau, u^x, u^y, tau u^eta), as SurfaceFinder builds it."""
    v2 = np.minimum(vx * vx + vy * vy + vz * vz, 1 - 1e-10)
    g = 1 / np.sqrt(1 - v2)
    ch, sh = np.cosh(eta), np.sinh(eta)
    return np.stack([g * (ch - vz * sh), g * vx, g * vy, g * (vz * ch - sh)], axis=-1)


def hist_surface(core, leg_arr, ntau_fo, grid, eos, T_sw, lattice, cap):
    """Surface of one leg of the history.  leg_arr: (4, nx, ny, neta, ntau) float32
    (energy_density, vx, vy, vz).  -> (cells (N, 32) float32, diagnostics dict)."""
    nt = min(leg_arr.shape[-1], int(ntau_fo) + 1)
    a = leg_arr[..., :nt]
    e = a[0].astype(np.float64)
    T = eos.T_of_e(e)
    fields = np.stack([T, e, eos.P_of_e(e), a[1], a[2], a[3]]).astype(np.float32)
    nx, ny, neta, _ = e.shape
    fd = core.FluidDynamics()
    fd.set_hydro_grid_info(tau_min=grid["tau_min"], dtau=grid["dtau"], ntau=nt,
                           x_min=grid["x_min"], dx=grid["dx"], nx=nx,
                           y_min=grid["y_min"], dy=grid["dy"], ny=ny,
                           eta_min=grid["eta_min"], deta=grid["deta"], neta=neta,
                           boost_inv=False, tau_eta_is_tz=False)
    fd.store_fluid_cells_from_numpy_3d(np.ascontiguousarray(fields),
                                       ["temperature", "energy_density", "pressure",
                                        "vx", "vy", "vz"])
    del fields
    fd.set_hydro_status_finished()
    fd.find_freezeout_surface(T_sw, *lattice)
    cells = np.asarray(fd.surface_to_numpy(), dtype=np.float32).reshape(-1, 32)
    del fd
    # thermodynamics on the T_sw isotherm, charges zero, ideal
    Tc = cells[:, COL["T"]].astype(np.float64)
    ec = eos.e_of_T(Tc)
    cells[:, COL["e"]] = ec
    cells[:, COL["P"]] = eos.P_of_e(ec)
    for name in CHARGES + VISCOUS:
        cells[:, COL[name]] = 0.0

    n_cap = 0
    if cap:
        e_sw = float(eos.e_of_T(T_sw))
        # MUSIC loops over grid points but the last one in each direction
        e0 = e[:-1, :-1, :-1, 0]
        m = (e0 > E_CAP_LOW) & (e0 < e_sw)
        ix, iy, ie = np.nonzero(m)
        n_cap = len(ix)
        if n_cap:
            x = grid["x_min"] + grid["dx"] * ix
            y = grid["y_min"] + grid["dy"] * iy
            eta = grid["eta_min"] + grid["deta"] * ie
            ecap = e0[m]
            c = np.zeros((n_cap, 32), dtype=np.float32)
            c[:, COL["tau"]] = grid["tau_min"]
            c[:, COL["x"]], c[:, COL["y"]], c[:, COL["eta"]] = x, y, eta
            c[:, COL["ds0"]] = grid["dx"] * grid["dy"] * grid["deta"]
            u = milne_u(a[1][ix, iy, ie, 0], a[2][ix, iy, ie, 0], a[3][ix, iy, ie, 0], eta)
            c[:, COL["u0"]:COL["u3"] + 1] = u
            c[:, COL["e"]] = ecap
            c[:, COL["T"]] = eos.T_of_e(ecap)
            c[:, COL["P"]] = eos.P_of_e(ecap)
            cells = np.concatenate([c, cells])

    diag = {
        "hist_ntau_used": int(nt),
        "hist_n_cap": int(n_cap),
        "hist_T_max_last_frame": float(T[..., -1].max()),
        "hist_T_max_xy_edge": float(max(T[[0, -1]].max(), T[:, [0, -1]].max())),
        "hist_T_max_eta_edge": float(T[:, :, [0, -1]].max()),
        "hist_T_max_first_frame": float(T[..., 0].max()),
    }
    return cells, diag


def zeroed_copy(cells, names):
    c = np.array(cells, dtype=np.float32, copy=True)
    for name in names:
        c[:, COL[name]] = 0.0
    return c


# ── files ───────────────────────────────────────────────────────────────────────
def find_inputs(inputs):
    import glob
    out = []
    for p in inputs:
        if os.path.isdir(p):
            out += sorted(glob.glob(os.path.join(p, "*_particlize.h5")))
        else:
            out += sorted(glob.glob(p)) or [p]
    return [os.path.abspath(p) for p in out]


def _complete(path):
    import h5py
    try:
        with h5py.File(path, "r") as f:
            return bool(f.attrs.get("complete", False))
    except OSError:
        return False


def event_range(spec, n):
    if not spec:
        return list(range(n))
    parts = [int(x) if x else None for x in spec.split(":")]
    return list(range(n))[slice(*parts)]


def process(a, src, core, eos):
    import h5py

    from jetscape.h5_compression import h5_filter_kwargs
    from jetscape.particlize_h5 import ParticlizeFile

    out = os.path.join(a.out_dir, os.path.basename(src))
    if os.path.exists(out):
        if a.skip_complete and _complete(out):
            print(f"make_surfaces.py: {os.path.basename(out)} complete, skipped")
            return out
        if not (a.force or a.skip_complete):
            sys.exit(f"make_surfaces.py: {out} exists (--force or --skip-complete)")
    pf = ParticlizeFile(src)
    if tuple(pf.legs) != ("jet", "bg") and set(pf.legs) != {"jet", "bg"}:
        sys.exit(f"make_surfaces.py: {src} has legs {pf.legs}; the study needs both")
    events = event_range(a.events, pf.nevents)
    bg_unit = np.asarray(pf.events("bg_unit"), dtype=np.int64)
    n_bg, bg_id = pf.bg_units()
    units = sorted({int(bg_unit[e]) for e in events})
    T_sw = float(a.T_sw if a.T_sw is not None else pf.attrs.get("T_fo", 0.15))

    pair = pair_path = None
    grid = lattice = None
    if a.variant == "hist":
        pair_path = os.path.join(os.path.dirname(src), str(pf.attrs.get("pair_file", "")))
        if not os.path.isfile(pair_path):
            sys.exit(f"make_surfaces.py: pair file {pair_path} of {src} not found")
        pair = h5py.File(pair_path, "r")
        A = pair.attrs
        grid = {k: float(A[k]) for k in ("tau_min", "dtau", "x_min", "dx", "y_min", "dy",
                                         "eta_min", "deta")}
        lattice = ((grid["dtau"], grid["dx"], grid["deta"]) if not a.lattice
                   else tuple(float(v) for v in a.lattice.split(",")))
        if len(lattice) != 3:
            sys.exit("make_surfaces.py: --lattice wants DTAU,DX,DETA")
        if list(pair["arr"].attrs.get("feature_names", A.get("feature_names", []))) [:4] \
                != ["energy_density", "vx", "vy", "vz"]:
            sys.exit(f"make_surfaces.py: {pair_path}: features are not e, vx, vy, vz")

    settings = {"variant": a.variant, "reference": src, "T_sw": T_sw,
                "events": [int(e) for e in events], "bg_units": units}
    if a.variant != "hist":
        settings["zeroed_columns"] = list(ZEROED[a.variant])
    else:
        settings.update({"pair_file": pair_path, "lattice": list(lattice), "cap": bool(a.cap),
                         "cap_e_low": E_CAP_LOW, "eos": eos.path, "grid": grid})

    tmp = out + ".part"
    kw, label = h5_filter_kwargs("blosc-zstd")
    t_start = time.time()
    with h5py.File(tmp, "w") as fo:
        for k, v in pf.attrs.items():
            fo.attrs[k] = v
        fo.attrs["complete"] = False
        # hadronize.py and HadronFileReader find the pair file (initiators) through this
        ref_pair = os.path.join(os.path.dirname(src), str(pf.attrs.get("pair_file", "")))
        fo.attrs["pair_file"] = ref_pair
        fo.attrs["hvs_variant"] = a.variant
        fo.attrs["hvs_reference"] = src
        fo.attrs["hvs_reference_uuid"] = str(pf.attrs.get("file_uuid", ""))
        fo.attrs["hvs_settings"] = json.dumps(settings)
        fo.attrs["hvs_producer"] = "js-contrib PyJetscape/example/hydro_hist_vs_surface/make_surfaces.py"
        if a.new_uuid:
            fo.attrs["file_uuid"] = str(uuidlib.uuid4())
        # an --events subset is a file of those events: renumber nothing, keep them all but
        # write empty units for the rest, so event and unit numbers stay the reference's
        pf.f.copy(pf.f["partons"], fo, "partons")
        pf.f.copy(pf.f["events"], fo, "events")
        ev_g = fo["events"]
        diag_rows = {}

        for leg, n_units in (("jet", pf.nevents), ("bg", n_bg)):
            g = fo.create_group(f"surface/{leg}")
            for k, v in pf.f[f"surface/{leg}"].attrs.items():
                g.attrs[k] = v
            ds = g.create_dataset("cells", (0, 32), maxshape=(None, 32), dtype=np.float32,
                                  chunks=(65536, 32), **kw)
            ds.attrs["compression"] = label
            for k, v in pf.f[f"surface/{leg}/cells"].attrs.items():
                if k != "compression":
                    ds.attrs[k] = v
            offsets = [0]
            wanted = set(events) if leg == "jet" else set(units)
            for u in range(n_units):
                if u not in wanted:
                    offsets.append(offsets[-1])
                    continue
                t0 = time.time()
                if a.variant != "hist":
                    cells, diag = zeroed_copy(pf.surface_unit(leg, u), ZEROED[a.variant]), {}
                else:
                    e_src = u if leg == "jet" else int(bg_id[u])
                    name, nfo = ("arr", "ntau_freezeout") if leg == "jet" else \
                        ("arr_bg", "ntau_freezeout_bg")
                    cells, diag = hist_surface(core, pair[name][e_src],
                                               int(pair[nfo][e_src]), grid, eos, T_sw,
                                               lattice, a.cap)
                n0 = ds.shape[0]
                ds.resize((n0 + len(cells), 32))
                ds[n0:] = cells
                offsets.append(n0 + len(cells))
                ref_n = int(pf.f[f"surface/{leg}/offsets"][u + 1]
                            - pf.f[f"surface/{leg}/offsets"][u])
                for key, v in diag.items():
                    diag_rows.setdefault(f"{key}_{leg}", {})[u] = v
                print(f"  {os.path.basename(src)} {leg} unit {u}: {len(cells)} cells "
                      f"(reference {ref_n}) in {time.time() - t0:.1f} s"
                      + (f", cap {diag['hist_n_cap']}, T_max edges xy/eta "
                         f"{diag['hist_T_max_xy_edge']:.3f}/{diag['hist_T_max_eta_edge']:.3f}"
                         if diag else ""), flush=True)
                fo.flush()
            g.create_dataset("offsets", data=np.asarray(offsets, dtype=np.int64))
            if leg == "bg":
                g.create_dataset("bg_id", data=pf.f["surface/bg/bg_id"][:])
            n_cells = np.diff(offsets)
            key = "n_cells_jet" if leg == "jet" else "n_cells_bg"
            if key in ev_g:
                per_event = n_cells if leg == "jet" else n_cells[bg_unit[:pf.nevents]]
                ev_g[key][: pf.nevents] = per_event
            if pf.has_events(key):
                ev_g.create_dataset(f"hvs_ref_{key}", data=pf.events(key))

        # per-event diagnostics (bg ones by the event's background)
        for key, per_unit in diag_rows.items():
            col = np.full(pf.nevents, np.nan)
            for e in range(pf.nevents):
                u = e if key.endswith("_jet") else int(bg_unit[e])
                if u in per_unit:
                    col[e] = per_unit[u]
            ev_g.create_dataset(key, data=col)
        fo.attrs["hvs_wall_s"] = time.time() - t_start
        fo.attrs["complete"] = True
    pf.close()
    if pair is not None:
        pair.close()
    os.replace(tmp, out)
    print(f"make_surfaces.py: {a.variant} -> {out} ({os.path.getsize(out) / 1e9:.2f} GB, "
          f"{time.time() - t_start:.0f} s)", flush=True)
    return out


def main(argv=None):
    global COL
    a = parse_args(argv)
    from jetscape.particlize_h5 import SURFACE_COLUMNS
    COL = {n: i for i, n in enumerate(SURFACE_COLUMNS)}
    srcs = find_inputs(a.inputs)
    if not srcs:
        sys.exit("make_surfaces.py: no *_particlize.h5 inputs")
    a.out_dir = os.path.abspath(a.out_dir)
    if any(os.path.dirname(s) == a.out_dir for s in srcs):
        sys.exit("make_surfaces.py: --out-dir must not be the reference directory (the "
                 "outputs keep the reference file names)")
    os.makedirs(a.out_dir, exist_ok=True)
    core = eos = None
    if a.variant == "hist":
        from jetscape import pyjetscape_core as core
        eos = Eos(a.eos)
        if tuple(core.SURFACE_CELL_COLUMNS) != tuple(COL):
            sys.exit("make_surfaces.py: pyjetscape_core's surface columns differ from "
                     "jetscape.particlize_h5's; rebuild the core")
    for s in srcs:
        process(a, s, core, eos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
