"""
python/jetscape/particlize_h5.py

ParticlizeH5Writer -- store everything needed to hadronize a two-stage MUSIC event later,
exactly: each leg's freeze-out surface with every field iSS reads, and the final partons
the jet hadronization receives.  ParticlizeFile reads it back.

    IS -> Hard -> MUSIC_1 -> Matter+LBT (+ liquefier) -> MUSIC_2       (production job)
        surface/bg  <- MUSIC_1's surface (once per background, see --reuse)
        surface/jet <- MUSIC_2's surface
        partons/    <- JetEnergyLossManager's final partons
                                                    -> <stem>_particlize.h5
    later, anywhere, as often as wanted (example/prod_AuAu_0_10_jet/hadronize.py):
        iSS on surface/jet -> bulk_jet, iSS on surface/bg -> bulk_bg,
        ColorlessHadronization on partons/ -> jet_frag           -> hadrons_<tag>.h5

Why this is exact
-----------------
* The framework's ``SurfaceCellInfo`` is ``Jetscape::real`` = float (``RealType.h``), and
  iSS copies exactly the 32 fields in :data:`SURFACE_COLUMNS` out of it
  (``iSpectraSamplerWrapper::getSurfCellVector``).  A float32 row is therefore what iSS
  would have received inside the job, bit for bit.
* ``partons/`` is ``JetEnergyLossManager::GetFinalStatePartons`` -- the list the
  HadronizationManager is handed -- not a reconstruction from the shower graph.
* Both legs come from the same job, so they share the initial condition by construction.

It is a separate file from the FNO4d pair file on purpose: the training data never carries
hadronization inputs, and these files can be moved, reprocessed or deleted on their own.

Layout
------
::

    attrs   format, format_version, producer, surface_columns, parton_columns, units,
            T_fo, music_input (the job's music_input text: iSS reads its EoS id and flags
            from it), pair_file, legs, nevents, nevents_written, complete, provenance ...
    surface/jet/cells    (N, 32) float32   one unit per event          (RaggedGroup)
    surface/jet/offsets  (nevents + 1,) int64
    surface/bg/cells     (M, 32) float32   one unit per NEW background (bg_id changes)
    surface/bg/offsets   (n_bg + 1,) int64
    surface/bg/bg_id     (n_bg,) int64     the first event that used this background
    partons/data         (K, 14) float64   FINAL_PARTON_COLUMNS         (RaggedGroup)
    partons/offsets      (nevents + 1,) int64
    events/<key>         (nevents,)        per event: bg_id, bg_unit (row of surface/bg),
                                           bg_key, n_cells_jet, n_cells_bg, n_partons,
                                           n_showers, plus whatever the driver passes
                                           (droplet energies, MUSIC tau0, seeds ...)

An event whose surface is empty (e.g. MUSIC stopped at the grid boundary) is an empty unit,
never a missing one; ``events/n_cells_jet`` says so.

Like :class:`jetscape.pair_h5.PairH5Writer` this is not a framework task: attach it after
``JetScape.Init()`` and call :meth:`ParticlizeH5Writer.Exec` between ``ExecPerEvent()`` and
``ClearPerEvent()``.  It needs no compiled extension itself (stubs work in tests).
There is no resume: a production job restarts from its first event anyway
(``run_jobs.sh`` reruns incomplete seeds), so the file is rewritten.
"""

from __future__ import annotations

import os
import uuid
import warnings

import numpy as np

from .fno_h5_writer import RaggedGroup
from .h5_compression import DEFAULT as DEFAULT_COMPRESSION

__all__ = ["FORMAT", "FORMAT_VERSION", "SURFACE_COLUMNS", "FINAL_PARTON_COLUMNS",
           "ParticlizeH5Writer", "ParticlizeFile"]

FORMAT = "xscape/particlize_input"
FORMAT_VERSION = 1

#: columns of a surface row: the fields of SurfaceCellInfo that iSS reads, in its FO_surf
#: order.  Mirrors pyjetscape_core.SURFACE_CELL_COLUMNS (checked when the core is present).
SURFACE_COLUMNS = (
    "tau", "x", "y", "eta", "ds0", "ds1", "ds2", "ds3", "u0", "u1", "u2", "u3",
    "e", "T", "P", "nB", "nQ", "nS", "muB", "muQ", "muS",
    "pi00", "pi01", "pi02", "pi03", "pi11", "pi12", "pi13", "pi22", "pi23", "pi33", "Pi")

#: columns of a final-parton row; mirrors pyjetscape_core.FINAL_PARTON_COLUMNS
FINAL_PARTON_COLUMNS = ("shower", "pid", "pstat", "E", "px", "py", "pz",
                        "t", "x", "y", "z", "mass", "col", "acol")

SURFACE_UNITS = (
    "Milne, as MUSIC hands them to the framework (MpiMusic::PassHydroSurfaceToFramework): "
    "tau [fm/c], x, y [fm], eta; ds_mu = d^3 sigma_mu (covariant, tau-weighted as MUSIC "
    "writes it) [fm^3]; u^mu contravariant; e, P [GeV/fm^3], T [GeV]; nB, nQ, nS [1/fm^3]; "
    "muB, muQ, muS [GeV]; pi^{mu nu} (10 independent components), Pi [GeV/fm^3].")
PARTON_UNITS = ("E, px, py, pz [GeV]; t [fm/c], x, y, z [fm] (Cartesian lab); mass [GeV] "
                "(rest mass); col, acol colour tags (0 = none; LBT assigns none); shower = "
                "index of the JetEnergyLoss task (one per shower-initiating parton)")
PSTAT_NOTE = ("pstat as the liquefier left it: 0 shower parton, 1 recoil, -1 hole (not "
              "absorbed), -11 absorbed into the medium below threshold, -17 hole absorbed, "
              "-13 the four-momentum a vertex did not conserve (LiquefierBase::"
              "check_energy_momentum_conservation; it goes into the droplet), 22 photon. "
              "ColorlessHadronization takes 0, 1 (take_recoil), 22 and -1, so energy that "
              "went into MUSIC_2 is never hadronized twice.")

class ParticlizeH5Writer:
    """Write each event's surfaces and final partons (see the module docstring).

    Parameters
    ----------
    out_file_name : str
        Output path.
    legs : iterable of {"jet", "bg"}
        Which legs' surfaces to store.  Their MUSIC instances must build the surface
        (``<freeze_out_surface>1``) and hand it over (``skip_surface`` off, e.g.
        ``PairH5Writer(keep_surface=legs)``).
    store_partons : bool
        Write ``partons/``.
    bg_id, jet_id : str
        Module ids of the two MUSIC instances.
    compression : str or None
        :mod:`jetscape.h5_compression` spec for the surfaces (default Blosc-zstd: the
        charge and chemical-potential columns are zero here and cost almost nothing).
        Partons are gzip'ed (small).
    music_input : str, optional
        Text of the job's ``music_input`` (iSS reads its EoS id and flags from it).  By
        default ``./music_input`` is read at :meth:`attach` (the job's working directory).
    T_fo : float, optional
        MUSIC's freeze-out temperature, recorded.
    pair_file : str, optional
        The FNO4d pair file written alongside, recorded (basename).
    extra_attrs : mapping, optional
        More root attributes (run provenance), written verbatim.
    force : bool
        Overwrite an existing file.
    """

    def __init__(self, out_file_name="particlize.h5", *, legs=("jet", "bg"),
                 store_partons=True, bg_id="MUSIC_1", jet_id="MUSIC_2",
                 compression=DEFAULT_COMPRESSION, music_input=None, T_fo=None,
                 pair_file=None, extra_attrs=None, force=True, verbose=False):
        self._path = str(out_file_name)
        self._legs = tuple(leg for leg in ("jet", "bg") if leg in set(legs))
        unknown = set(legs) - {"jet", "bg"}
        if unknown:
            raise ValueError(f"legs takes 'jet' and/or 'bg', got {sorted(unknown)}")
        self._store_partons = bool(store_partons)
        self._bg_id, self._jet_id = str(bg_id), str(jet_id)
        self._compression = compression
        self._music_input = music_input
        self._T_fo = T_fo
        self._pair_file = pair_file
        self._extra = dict(extra_attrs or {})
        self._force = bool(force)
        self._verbose = bool(verbose)

        self._bg = self._jet = self._mgr = None
        self.f = None
        self._surf = {}
        self._bg_ids = None
        self._partons = None
        self._events = None
        self._i = 0
        self._last_bg_id = None
        self._bg_unit = -1
        self._n_cells_bg = 0
        self._n_empty = 0

    # ── wiring ──────────────────────────────────────────────────────────────────
    def attach(self, jetscape=None, *, bg=None, jet=None, manager=None):
        """Find the MUSIC legs and the energy-loss manager; open the file.

        Call after ``JetScape.Init()`` (and after ``PairH5Writer.attach``, which sets the
        legs' ``skip_surface`` flags).  A stored leg found with ``skip_surface`` on is
        switched off here, with a warning.
        """
        need = {"jet": jet, "bg": bg}
        if any(need[leg] is None for leg in self._legs):
            if jetscape is None:
                raise ValueError("attach: pass the JetScape object or the stored legs")
            tasks = {t.GetId(): t for t in jetscape.GetTaskList()}
            jet = tasks.get(self._jet_id) if jet is None else jet
            bg = tasks.get(self._bg_id) if bg is None else bg
        for leg, obj, mid in (("jet", jet, self._jet_id), ("bg", bg, self._bg_id)):
            if leg in self._legs and obj is None:
                raise RuntimeError(f"attach: no task with id {mid!r} for the {leg} leg")
            if leg in self._legs and _get_flag(obj, "get_skip_surface"):
                warnings.warn(f"ParticlizeH5Writer: {mid} had skip_surface on, so its "
                              "surface would never reach the framework; switching it off.",
                              RuntimeWarning, stacklevel=2)
                obj.set_skip_surface(False)
        if self._store_partons and manager is None:
            try:
                from .pyjetscape_core import JetScapeSignalManager
                manager = JetScapeSignalManager.Instance().GetJetEnergyLossManagerPointer()
            except ImportError:                       # pragma: no cover - stubs in tests
                manager = None
            if manager is None:
                warnings.warn("ParticlizeH5Writer: no JetEnergyLossManager; partons/ will "
                              "be empty.", RuntimeWarning, stacklevel=2)
        self._jet, self._bg, self._mgr = jet, bg, manager
        if self._music_input is None and os.path.exists("music_input"):
            with open("music_input") as fh:
                self._music_input = fh.read()
        self._open()
        print(f"ParticlizeH5Writer: surfaces of {', '.join(self._legs) or 'no leg'}"
              f"{' + final partons' if self._store_partons else ''} -> {self._path}")

    def _open(self):
        import h5py

        if os.path.exists(self._path) and not self._force:
            raise FileExistsError(f"{self._path} exists (pass force=True)")
        os.makedirs(os.path.dirname(os.path.abspath(self._path)) or ".", exist_ok=True)
        self.f = h5py.File(self._path, "w")
        a = self.f.attrs
        a["format"], a["format_version"] = FORMAT, FORMAT_VERSION
        a["producer"] = "js-contrib/contribs/PyJetscape (jetscape.particlize_h5)"
        a["file_uuid"] = str(uuid.uuid4())
        a["legs"] = list(self._legs)
        a["jet_id"], a["bg_id"] = self._jet_id, self._bg_id
        a["surface_columns"] = list(SURFACE_COLUMNS)
        a["surface_units"] = SURFACE_UNITS
        a["parton_columns"] = list(FINAL_PARTON_COLUMNS)
        a["parton_units"] = PARTON_UNITS
        a["pstat_convention"] = PSTAT_NOTE
        a["music_input"] = self._music_input or ""
        if self._T_fo is not None:
            a["T_fo"] = float(self._T_fo)
        if self._pair_file:
            a["pair_file"] = os.path.basename(str(self._pair_file))
        a["nevents"] = a["nevents_written"] = 0
        a["complete"] = False
        for k, v in self._extra.items():
            a[k] = v

        ncol = len(SURFACE_COLUMNS)
        for leg in self._legs:
            g = self.f.require_group(f"surface/{leg}")
            self._surf[leg] = RaggedGroup(g, "offsets", {"cells": (np.float32, (ncol,))},
                                          compression=self._compression, chunk_rows=65536,
                                          unit="event" if leg == "jet" else "free")
            g.attrs["cell_columns"] = list(SURFACE_COLUMNS)
            g.attrs["hydro_id"] = self._jet_id if leg == "jet" else self._bg_id
            g.attrs["unit"] = ("event" if leg == "jet" else
                               "background: one unit per new bg_id (see bg_id, "
                               "events/bg_unit)")
        if "bg" in self._legs:
            self._bg_ids = self.f["surface/bg"].create_dataset(
                "bg_id", (0,), maxshape=(None,), dtype=np.int64, chunks=(1024,))
        if self._store_partons:
            g = self.f.require_group("partons")
            self._partons = RaggedGroup(g, "offsets", {
                "data": (np.float64, (len(FINAL_PARTON_COLUMNS),))},
                compression="gzip", chunk_rows=16384)
            g.attrs["data_columns"] = list(FINAL_PARTON_COLUMNS)
        self._events = _ScalarTable(self.f.require_group("events"))

    # ── per event ───────────────────────────────────────────────────────────────
    def Exec(self, i=None, *, bg_id=None, bg_key=None, **diag):
        """Store this event.  Returns its index.

        ``i`` is the event's index in the pair file (default: the running count; the two
        must agree -- events skipped by the pair writer must be skipped here too).
        ``bg_id`` names the background: the first event that used it (the pair writer's
        ``diag/bg_id``).  A new value stores the background surface; a repeated one (an
        event reusing its background) only records the reference.  Without a pair writer
        pass ``bg_id=i``.  Extra keywords go to ``events/``.
        """
        if self.f is None:
            raise RuntimeError("ParticlizeH5Writer: call attach() after JetScape.Init()")
        k = self._i
        if i is not None and int(i) != k:
            raise ValueError(f"ParticlizeH5Writer: event index {i} but {k} events written; "
                             "skip the same events as the pair writer")
        bg_id = k if bg_id is None else int(bg_id)
        row = {"event": k, "bg_id": bg_id}

        if "jet" in self._legs:
            cells = _surface(self._jet)
            self._surf["jet"].append({"cells": cells})
            row["n_cells_jet"] = len(cells)
            if not len(cells):
                self._n_empty += 1
                warnings.warn(f"ParticlizeH5Writer: event {k}: {self._jet_id} handed over "
                              "an empty surface (stored as an empty unit)",
                              RuntimeWarning, stacklevel=2)
        if "bg" in self._legs:
            if bg_id != self._last_bg_id:
                cells = _surface(self._bg)
                self._surf["bg"].append({"cells": cells})
                n = self._bg_ids.shape[0]
                self._bg_ids.resize((n + 1,))
                self._bg_ids[n] = bg_id
                self._bg_unit, self._last_bg_id = n, bg_id
                self._n_cells_bg = len(cells)
                if not len(cells):
                    warnings.warn(f"ParticlizeH5Writer: event {k}: {self._bg_id} handed "
                                  "over an empty surface for a new background",
                                  RuntimeWarning, stacklevel=2)
            row["bg_unit"] = self._bg_unit
            row["n_cells_bg"] = self._n_cells_bg
        if self._store_partons:
            rows = (np.zeros((0, len(FINAL_PARTON_COLUMNS))) if self._mgr is None
                    else np.asarray(self._mgr.final_partons_numpy(), dtype=np.float64))
            self._partons.append({"data": rows})
            row["n_partons"] = len(rows)
            row["n_showers"] = int(len(np.unique(rows[:, 0]))) if len(rows) else 0
        if bg_key is not None:
            row["bg_key"] = str(bg_key)
        row.update(diag)
        self._events.append(k, row)

        self._i += 1
        self.f.attrs["nevents"] = self.f.attrs["nevents_written"] = self._i
        self.f.flush()
        if self._verbose:
            parts = [f"{leg} {row.get('n_cells_' + leg, 0)} cells" for leg in self._legs]
            if self._store_partons:
                parts.append(f"{row['n_partons']} final partons")
            print(f"ParticlizeH5Writer: event {k}: " + ", ".join(parts)
                  + (f", bg_unit {row['bg_unit']}" if "bg_unit" in row else ""))
        return k

    def write_events(self, i, **scalars):
        """Add per-event scalars after :meth:`Exec` (e.g. seeds, wall time)."""
        if self.f is not None:
            self._events.set(i, scalars)

    def Finish(self, complete=None):
        """Close the file.  Idempotent."""
        if self.f is None:
            return
        self.f.attrs["complete"] = bool(True if complete is None else complete)
        self.f.close()
        self.f = None
        print(f"ParticlizeH5Writer: wrote {self._i} event(s) to {self._path}"
              + (f" ({self._n_empty} with an empty jet surface)" if self._n_empty else ""))

    # ── accessors ───────────────────────────────────────────────────────────────
    def GetNumberOfEventsWritten(self):
        return self._i

    @property
    def path(self):
        return self._path


class ParticlizeFile:
    """Read a particlize-input file.

        with ParticlizeFile("AuAu_0_10_jet_seed0001_particlize.h5") as pf:
            cells = pf.surface("jet", 0)            # (N, 32) float32
            cells_bg = pf.surface("bg", 0)          # the background event 0 used
            partons = pf.partons(0)                 # (K, 14) float64
    """

    def __init__(self, path):
        import h5py

        from . import h5_compression  # noqa: F401  (registers the Blosc filter)

        self.path = str(path)
        self.f = h5py.File(self.path, "r")
        if self.f.attrs.get("format") != FORMAT:
            raise ValueError(f"{self.path}: not a {FORMAT} file "
                             f"(format={self.f.attrs.get('format')!r})")
        self.attrs = dict(self.f.attrs)
        self.legs = tuple(str(x) for x in self.attrs.get("legs", ()))
        self.nevents = int(self.attrs.get("nevents_written", 0))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        if self.f is not None:
            self.f.close()
            self.f = None

    def events(self, key):
        """``events/<key>`` as an array (NaN / '' where an event did not set it)."""
        return self.f[f"events/{key}"][: self.nevents]

    def has_events(self, key):
        return f"events/{key}" in self.f

    def bg_units(self):
        """Number of stored backgrounds and their bg_id (first event) values."""
        if "surface/bg/bg_id" not in self.f:
            return 0, np.zeros(0, dtype=np.int64)
        ids = self.f["surface/bg/bg_id"][:]
        return len(ids), ids

    def surface_unit(self, leg, unit):
        """Cells of unit ``unit`` of ``surface/<leg>`` (an event for jet, a background
        for bg)."""
        g = self.f[f"surface/{leg}"]
        off = g["offsets"]
        return g["cells"][int(off[unit]):int(off[unit + 1])]

    def surface(self, leg, event):
        """Cells of ``leg``'s surface for ``event`` (bg: the background it used)."""
        if leg == "bg":
            return self.surface_unit("bg", int(self.events("bg_unit")[event]))
        return self.surface_unit(leg, event)

    def partons(self, event):
        g = self.f["partons"]
        off = g["offsets"]
        return g["data"][int(off[event]):int(off[event + 1])]

    def music_input(self):
        return str(self.attrs.get("music_input", ""))


# ── helpers ─────────────────────────────────────────────────────────────────────
def _get_flag(obj, name):
    get = getattr(obj, name, None)
    return bool(get()) if get is not None else False


def _surface(hydro):
    a = np.asarray(hydro.surface_to_numpy(), dtype=np.float32)
    if a.ndim != 2 or (a.size and a.shape[1] != len(SURFACE_COLUMNS)):
        raise ValueError(f"surface_to_numpy returned shape {a.shape}, expected "
                         f"(N, {len(SURFACE_COLUMNS)})")
    return a.reshape(-1, len(SURFACE_COLUMNS))


class _ScalarTable:
    """Per-event scalars under one group, one growable dataset per key.

    Written as it goes (crash-safe); a key first seen at event k is back-filled with its
    missing value (NaN, -1 or '') for events 0..k-1.
    """

    def __init__(self, group):
        self.g = group
        self.n = 0

    @staticmethod
    def _kind(v):
        if isinstance(v, (bool, np.bool_)):
            return np.int8, 0
        if isinstance(v, (int, np.integer)):
            return (np.uint64, 0) if int(v) > np.iinfo(np.int64).max else (np.int64, -1)
        if isinstance(v, (float, np.floating)):
            return np.float64, np.nan
        return h5py_str(), b""

    def _ensure(self, key, v):
        if key not in self.g:
            dtype, fill = self._kind(v)
            ds = self.g.create_dataset(key, (self.n,), maxshape=(None,), dtype=dtype,
                                       chunks=(1024,), fillvalue=fill)
            return ds
        return self.g[key]

    def append(self, i, row):
        self.n = i + 1
        for key in self.g:
            ds = self.g[key]
            if ds.shape[0] < self.n:
                ds.resize((self.n,))
        for key, v in row.items():
            ds = self._ensure(key, v)
            if ds.shape[0] < self.n:
                ds.resize((self.n,))
            ds[i] = _cast(v, ds.dtype)

    def set(self, i, row):
        for key, v in row.items():
            ds = self._ensure(key, v)
            if ds.shape[0] < self.n:
                ds.resize((self.n,))
            ds[i] = _cast(v, ds.dtype)


def h5py_str():
    import h5py
    return h5py.string_dtype()


def _cast(v, dtype):
    if dtype.kind in ("O", "S", "U"):
        return str(v)
    return v
