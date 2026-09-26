"""
python/jetscape/pair_h5.py

PairH5Writer -- write an X-SCAPE two-stage MUSIC run as one FNO4d-schema HDF5 file in
FastHydro's pair layout: the jet leg (with the CausalLiquefier deposition) as ``arr``, the
background leg as ``arr_bg``, the droplets MUSIC was given as ``source/droplets``, and the
parton showers as ``shower/``.

    IS -> Hard -> NullPreDynamics -> MUSIC_1 (background)
       -> JetEnergyLossManager (Matter + LBT, CausalLiquefier)
       -> MUSIC_2 (same IC + the liquefier source)          -> PairH5Writer

Where each leg is read from
---------------------------
* **Background (MUSIC_1)** from the framework copy, ``bulk_info``.  Matter and LBT query the
  first hydro through ``bulk_info``, so it must be filled, and filling it releases MUSIC's
  native store (``MusicWrapper.cc`` ``PassHydroEvolutionHistoryToFramework``).  The copy is
  cell for cell the native store: both go through ``get_fluid_cell_with_index`` and a float
  copy, so nothing is lost.
* **Jet leg (MUSIC_2)** from its native store, which ``attach()`` keeps with
  ``set_dump_hydro_only(True)`` on that instance only.  Every MUSIC instance reads the first
  ``<Hydro><MUSIC>`` XML block, so this cannot be set per leg in the XML.

Both legs then go through the same :func:`jetscape.bulk_sources.resample` onto one output
grid, so ``arr - arr_bg`` is the jet's effect and nothing else: before the first droplet
deposits the two legs are bit-identical, which ``diag/frames_identical`` records per event.

Layout (on top of :class:`jetscape.fno_h5_writer.FnoH5Writer`)
--------------------------------------------------------------
* ``arr`` / ``arr_bg`` with ``ntau_freezeout[_bg]``, ``tau_freezeout[_bg]``.  The tau axis
  grows to the longer leg (the jet leg usually outlives the background); the shorter one is
  exactly zero after its own freeze-out.  ``arr_bg.shape == arr.shape`` always.
* ``source/droplets`` ``(M, 8)`` + ``source/offsets``, columns ``droplet_columns``.  There is no
  ``source/S``: MUSIC evaluates the liquefier kernel per cell and step and never keeps a
  gridded source, and a re-deposit on the output grid would not be what MUSIC applied.
  ``has_source`` is therefore false.
* ``shower/`` exactly as FastHydro writes it (``jetscape.showers``).
* ``diag/``: droplet counts and energies (and how much deposits after the jet leg froze out,
  which MUSIC never applies), shower counts, MUSIC tau0, per-leg frame counts, ``bg_id``
  (the first event that used this background -- repeats under ``setReuseHydro``),
  ``frames_identical``.
* Provenance attributes as in FastHydro's ``PairedH5Writer`` (``pairing``, ``arr_is``,
  ``arr_bg_is``, ``source_model``, ``liquefier_*`` ...), so ``fasthydro.browse.PairBrowser``,
  the wake notebook and ``wake_pyvista.py`` read the file unchanged.

Usage (see example/prod_AuAu_0_10_jet/run_prod_jet.py)
-----------------------------------------------------
    jetscape = js.JetScapePerEvent(); ...; jetscape.Init()
    writer = PairH5Writer("pair.h5", grid_mode="grid", out_grid=grid)
    writer.attach(jetscape)                 # AFTER Init(): sets dump_hydro_only on MUSIC_2
    jetscape.ExecInit()
    for i in range(n):
        jetscape.ExecPerEvent()
        idx = writer.Exec()                 # both evolutions are live here
        jetscape.ClearPerEvent()
    writer.Finish()

The writer is not a framework task: it has to be attached after ``Init()`` and called
between ``ExecPerEvent()`` and ``ClearPerEvent()``, the same pattern run_prod.py uses for
``H5BulkWriter``.  It needs no compiled extension itself, so it is testable with stubs.
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import replace
from typing import Optional

import numpy as np

from .bulk_sources import (Grid, attrs_from_grids, event_array, music_extra_attrs,
                           resample, resolve_out_grid)
from .fno_h5_writer import FnoH5Writer
from .h5_compression import DEFAULT as DEFAULT_COMPRESSION
from .liquefier_io import DROPLET_COLUMNS, droplets, liquefier_params
from .showers import (FATES, INITIATOR_COLUMNS, PARTON_COLUMNS, VERTEX_COLUMNS,
                      showers_from_manager)

__all__ = ["PairH5Writer", "shower_group_attrs"]

#: relative tolerance for "the two legs are on the same grid"; bulk_info is float32
_GRID_RTOL = 1e-5


def shower_group_attrs():
    """The ``shower/`` group attributes FastHydro's ``PairedH5Writer`` writes."""
    return {
        "parton_columns": list(PARTON_COLUMNS),
        "vertex_columns": list(VERTEX_COLUMNS),
        "initiator_columns": list(INITIATOR_COLUMNS),
        "units": "p in GeV; x, y, z in fm; t in fm/c",
        "coordinates": (
            "CARTESIAN LAB (x, y, z) and lab time t -- NOT the Milne (tau, x, y, eta) the "
            "hydro frames and source/droplets use. Convert with jetscape.showers.to_milne "
            "(also fasthydro.showers.to_milne): tau = sqrt(t^2 - z^2), eta = atanh(z/t)."),
        "vertex_positions": (
            "ZERO BY CONSTRUCTION. X-SCAPE builds every vertex as Vertex(0,0,0,currentTime) "
            "(JetEnergyLoss.cc:414-419), so only the `t` column is real. The geometry is on "
            "the partons, which carry their own production point; use "
            "jetscape.showers.segments, which propagates each parton along p/E rather than "
            "reading vertex positions."),
        "endpoint_convention": (
            "i_src and i_tgt are ROW INDICES into this event's slice of `vertices` "
            "(vertex_offsets[i] : vertex_offsets[i+1]), already offset across the event's "
            "showers. They are not raw GTL node ids, which restart at 0 per shower; the raw "
            "id is kept as the vertex `node_id` column."),
        "pstat_codes": [f"{k}: {v}" for k, v in sorted(FATES.items())],
    }


class PairH5Writer:
    """Write the background and jet legs of a two-stage MUSIC run as one pair file.

    Parameters
    ----------
    out_file_name : str
        Output path.
    bg_id, jet_id : str
        Module ids of the two MUSIC instances (``<name>`` in the XML).
    grid_mode : {"grid", "native"}
        ``grid`` resamples both legs onto ``out_grid``; ``native`` writes MUSIC's own grid.
    out_grid : mapping or bulk_sources.Grid, optional
        ``grid`` mode only; as for :class:`jetscape.fast_h5_bulk.H5BulkWriter`.
    tau_stride : int
        Keep every Nth stored MUSIC step, on both legs.
    choose_ntau : int
        0 grows the tau axis to the longest leg of any event; N > 0 pins it (longer legs
        are clipped, with a warning and a count at Finish()).
    compression : str, mapping or None
        :mod:`jetscape.h5_compression` spec for ``arr`` and ``arr_bg`` (default
        ``"blosc-zstd"``; ``"lzf"`` is the old default).  See README_h5_optim.md.
    keep_bits : int or None
        Round both legs to this many float32 mantissa bits (lossy, relative error
        <= ``2**-(keep_bits+1)``); None (default) is bit-exact.  Both legs are rounded the
        same way, so ``arr - arr_bg`` is exactly zero wherever the legs agree.
    store_droplets, store_showers : bool
        Write ``source/droplets`` and ``shower/``.
    provenance : mapping, optional
        Overrides for the provenance attributes (``hard_vertex``, ``eos_kind``,
        ``transport_mode``, ``source_mode`` ...).
    extra_attrs : mapping, optional
        More root attributes, written verbatim (run provenance).
    force : bool
        Overwrite an existing file.
    """

    def __init__(self, out_file_name="pair_evo.h5", *, bg_id="MUSIC_1", jet_id="MUSIC_2",
                 grid_mode="grid", out_grid=None, tau_stride=1, choose_ntau=0,
                 compression=DEFAULT_COMPRESSION, keep_bits=None, store_droplets=True,
                 store_showers=True, provenance=None, extra_attrs=None, force=True,
                 verbose=False):
        if grid_mode not in ("grid", "native"):
            raise ValueError(f"grid_mode must be 'grid' or 'native', got {grid_mode!r}")
        self._out_file_name = str(out_file_name)
        self._bg_id, self._jet_id = str(bg_id), str(jet_id)
        self._grid_mode = grid_mode
        self._out_grid = out_grid if isinstance(out_grid, Grid) else dict(out_grid or {})
        self._tau_stride = max(1, int(tau_stride))
        self._choose_ntau = max(0, int(choose_ntau))
        self._compression = compression
        self._keep_bits = keep_bits
        self._store_droplets = bool(store_droplets)
        self._store_showers = bool(store_showers)
        self._provenance = dict(provenance or {})
        self._extra_attrs = dict(extra_attrs or {})
        self._force = bool(force)
        self._verbose = bool(verbose)

        self._bg = self._jet = self._liq = self._mgr = None
        self._deposition = None                 # True: MUSIC_2 has the liquefier
        self._tau_delay = 0.0
        self._w: Optional[FnoH5Writer] = None
        self._g_drop = self._g_sh = None
        self._i = 0
        self._n_clipped = 0
        self._bg_hash = None
        self._bg_first = 0
        self._last = {}

    # ── wiring ──────────────────────────────────────────────────────────────────
    def attach(self, jetscape=None, *, bg=None, jet=None, liquefier=None, manager=None):
        """Find the two legs, the liquefier and the shower manager; set per-leg flags.

        Call after ``JetScape.Init()`` (which reads the XML flags this overrides) and
        before the first event.  Anything not passed is looked up: the legs by module id
        in ``jetscape.GetTaskList()``, the liquefier on the jet leg (or, with the jet leg's
        ``<AddLiquefier>`` off, on the energy-loss manager's template), the manager through
        the signal manager.
        """
        if bg is None or jet is None:
            if jetscape is None:
                raise ValueError("attach: pass the JetScape object or both legs")
            tasks = {t.GetId(): t for t in jetscape.GetTaskList()}
            bg = tasks.get(self._bg_id) if bg is None else bg
            jet = tasks.get(self._jet_id) if jet is None else jet
            missing = [n for n, t in ((self._bg_id, bg), (self._jet_id, jet)) if t is None]
            if missing:
                raise RuntimeError(
                    f"attach: no task with id {missing} (task ids: {sorted(tasks)}). Name "
                    f"the two MUSIC blocks <name>{self._bg_id}</name> and "
                    f"<name>{self._jet_id}</name>.")
        if bg.get_dump_hydro_only():
            raise RuntimeError(
                f"attach: {self._bg_id} has dump_hydro_only set, so it will not fill "
                "bulk_info and Matter/LBT would see no medium. Set "
                "<Hydro><MUSIC><dump_hydro_only>0 (the jet leg is switched here).")
        jet.set_dump_hydro_only(True)
        for leg in (bg, jet):
            leg.set_skip_surface(True)

        if manager is None:
            try:
                from .pyjetscape_core import JetScapeSignalManager
                manager = JetScapeSignalManager.Instance().GetJetEnergyLossManagerPointer()
            except ImportError:                       # pragma: no cover - stubs in tests
                manager = None

        own = jet.get_liquefier() if hasattr(jet, "get_liquefier") else None
        if liquefier is None:
            liquefier = own
        if liquefier is None and manager is not None:
            for task in manager.GetTaskList():
                get = getattr(task, "get_liquefier", None)
                if get is not None and get() is not None:
                    liquefier = get()
                    break
        self._deposition = own is not None
        if liquefier is None and self._store_droplets:
            warnings.warn("PairH5Writer: no liquefier found; source/droplets will be empty.",
                          RuntimeWarning, stacklevel=2)

        self._bg, self._jet, self._liq, self._mgr = bg, jet, liquefier, manager
        self._tau_delay = (float(liquefier_params(liquefier)["tau_delay"])
                           if liquefier is not None else 0.0)
        mode ="on" if self._deposition else "OFF (null test: the jet leg has no liquefier)"
        print(f"PairH5Writer: {self._bg_id} -> arr_bg (framework copy), {self._jet_id} -> "
              f"arr (native store), deposition {mode}, output -> {self._out_file_name}")

    # ── per event ───────────────────────────────────────────────────────────────
    def Exec(self, **diag):
        """Write the current event's pair.  Returns its index in the file, or None.

        Extra keyword arguments are stored under ``diag/`` for this event.  Must run
        between ``ExecPerEvent()`` and ``ClearPerEvent()``.
        """
        if self._bg is None:
            raise RuntimeError("PairH5Writer: call attach() after JetScape.Init() first")
        try:
            return self._write_event(diag)
        finally:
            # The jet leg's native store and (in the null test) the droplets are not
            # released by anything else.
            self._jet.clear_hydro_info_from_memory()
            if not self._deposition and self._liq is not None:
                self._liq.ClearTask()

    def _write_event(self, diag):
        i = self._i
        try:
            src_bg = Grid.from_bulk_info(self._bg.get_bulk_info())
            src_jet = Grid.from_bulk_info(self._jet.get_bulk_info())
        except Exception as exc:                      # noqa: BLE001 - never kill the run
            warnings.warn(f"PairH5Writer: skipping event {i}: no grid ({exc})",
                          RuntimeWarning, stacklevel=3)
            return None
        mismatch = _grid_mismatch(src_bg, src_jet)
        if mismatch:
            warnings.warn(f"PairH5Writer: skipping event {i}: the legs are on different "
                          f"grids ({mismatch})", RuntimeWarning, stacklevel=3)
            return None

        # Read and resample both legs before writing anything, so a leg that cannot be read
        # skips the whole event instead of leaving half a pair behind.  Only one leg's
        # source-grid array is alive at a time; the output frames of both are kept.
        legs = {}
        for name, hydro, framework in (("jet", self._jet, False),
                                       ("background", self._bg, True)):
            try:
                arr, src = self._read(hydro, framework=framework)
            except Exception as exc:                  # noqa: BLE001
                warnings.warn(f"PairH5Writer: skipping event {i}: {name} leg: {exc}",
                              RuntimeWarning, stacklevel=3)
                return None
            if arr.shape[0] == 0:
                warnings.warn(f"PairH5Writer: skipping event {i}: the {name} leg has no "
                              "frames", RuntimeWarning, stacklevel=3)
                return None
            out = self._out(src)
            legs[name] = (self._resampled(arr, src, out), src, out)
            del arr
        jet_frames, jet_src, jet_out = legs["jet"]
        bg_frames, bg_src, bg_out = legs.pop("background")
        del legs

        if self._w is None:
            self._open(bg_src, bg_out)
        elif (bg_out.nx, bg_out.ny, bg_out.neta) != tuple(self._w.arr.shape[2:5]):
            warnings.warn(f"PairH5Writer: skipping event {i}: grid "
                          f"{(bg_out.nx, bg_out.ny, bg_out.neta)} differs from the file's "
                          f"{tuple(self._w.arr.shape[2:5])}", RuntimeWarning, stacklevel=3)
            return None

        # Background first, then the jet leg; frame hashes carry the comparison across.
        n_bg = self._clip(i, "background", bg_frames.shape[0])
        self._w.ensure_capacity(nevents=i + 1, choose_ntau=n_bg)
        bg_hashes = []
        for t in range(n_bg):
            frame = np.ascontiguousarray(bg_frames[t].transpose(3, 0, 1, 2))
            self._w.write_frame(i, t, frame, dataset="arr_bg")
            bg_hashes.append(hashlib.blake2b(frame.tobytes(), digest_size=16).digest())
        del bg_frames
        tau_fo_bg = bg_src.tau_min + bg_src.ntau * bg_src.dtau
        self._w.set_event_meta(i, n_bg, tau_fo_bg, dataset="arr_bg")

        # A new background run?  Hash the whole leg: on MUSIC's native grid the first
        # frame is at MUSIC's tau0, before any string deposits, and all zero.
        bg_key = hashlib.blake2b(b"".join(bg_hashes), digest_size=16).digest()
        if bg_key != self._bg_hash:
            self._bg_hash, self._bg_first = bg_key, i

        n_jet = self._clip(i, "jet", jet_frames.shape[0])
        self._w.ensure_capacity(nevents=i + 1, choose_ntau=n_jet)
        identical, same = 0, True
        for t in range(n_jet):
            frame = np.ascontiguousarray(jet_frames[t].transpose(3, 0, 1, 2))
            self._w.write_frame(i, t, frame)
            if same and t < n_bg and hashlib.blake2b(
                    frame.tobytes(), digest_size=16).digest() == bg_hashes[t]:
                identical += 1
            else:
                same = False
        del jet_frames
        tau_fo_jet = jet_src.tau_min + jet_src.ntau * jet_src.dtau

        d = self._liq_rows(tau_fo_jet, jet_src.tau_min)
        if self._g_drop is not None:
            self._g_drop.append({"droplets": d.pop("_rows")})
        else:
            d.pop("_rows")
        d.update(self._shower_rows())
        d.update(tau0_music=float(jet_src.tau_min), ntau_jet=n_jet, ntau_bg=n_bg,
                 bg_id=self._bg_first, frames_identical=identical)
        for leg, hydro in (("bg", self._bg), ("jet", self._jet)):
            hit = _hit_grid_boundary(hydro)
            if hit is not None:
                d[f"{leg}_hit_boundary"] = int(hit)
        d.update(diag)
        self._w.write_diag(i, **d)
        self._w.set_event_meta(i, n_jet, tau_fo_jet)          # primary last: marks written
        self._i += 1
        self._last = d
        self._check_pair(i, d, n_bg, n_jet)
        if self._verbose:
            print(f"PairH5Writer: event {i}: jet {n_jet} / bg {n_bg} frames, "
                  f"{d['n_droplets']} droplets ({d['E_droplets']:.2f} GeV), "
                  f"{identical} leading frames identical, bg_id {self._bg_first}")
        return i

    def write_diag(self, i, **scalars):
        """Add per-event scalars after ``Exec()`` (e.g. the driver's wall time)."""
        if self._w is not None:
            self._w.write_diag(i, **scalars)

    def Finish(self):
        """Close the file.  Idempotent."""
        if self._w is None:
            return
        n, t = self._w.nevents, self._w.choose_ntau
        self._w.close()
        self._w = None
        print(f"PairH5Writer: wrote {n} pair(s) to {self._out_file_name} (choose_ntau = {t})")
        if self._n_clipped:
            print(f"PairH5Writer: WARNING -- {self._n_clipped} leg(s) were clipped to "
                  f"choose_ntau={t}.")

    # ── accessors ───────────────────────────────────────────────────────────────
    def GetOutFileName(self):
        return self._out_file_name

    def GetNumberOfEventsWritten(self):
        return self._i

    @property
    def last_event_diag(self):
        """The ``diag/`` values of the last written event."""
        return dict(self._last)

    @property
    def n_clipped(self):
        """Legs cut at a pinned ``choose_ntau`` so far."""
        return self._n_clipped

    @property
    def deposition(self):
        """True if the jet leg has the liquefier, False in the null test, None before attach."""
        return self._deposition

    # ── internals ───────────────────────────────────────────────────────────────
    def _read(self, hydro, framework):
        """(ntau, nx, ny, neta, 4) on the leg's source grid, and that grid (strided)."""
        if framework:
            arr, src, _ = event_array(hydro, "framework")
            if self._tau_stride > 1:              # framework mode has no stride of its own
                arr = arr[::self._tau_stride]
                src = replace(src, ntau=arr.shape[0], dtau=src.dtau * self._tau_stride)
        else:
            arr, src, _ = event_array(hydro, "native", tau_stride=self._tau_stride)
        return arr, src

    def _out(self, src):
        return src if self._grid_mode == "native" else resolve_out_grid(src, self._out_grid)

    @staticmethod
    def _resampled(arr, src, out):
        return arr if out is src else resample(arr, src, out)

    def _clip(self, i, leg, n):
        if self._w.growable_tau or n <= self._w.choose_ntau:
            return n
        self._n_clipped += 1
        warnings.warn(f"PairH5Writer: event {i}: {leg} leg has {n} frames but choose_ntau "
                      f"is pinned at {self._w.choose_ntau}; clipping.",
                      RuntimeWarning, stacklevel=4)
        return self._w.choose_ntau

    def _liq_rows(self, tau_fo_jet, tau0):
        """This event's droplets plus their counts and energies.

        A droplet deposits at ``tau + tau_delay`` (``CausalLiquefier.cc:117``); one that
        deposits after the jet leg stopped, or before it started, never reaches MUSIC.
        """
        rows = np.zeros((0, len(DROPLET_COLUMNS)))
        if self._liq is not None:
            rows = droplets(self._liq)
        e = rows[:, 4]
        d = {"_rows": rows, "n_droplets": int(len(rows)), "E_droplets": float(e.sum())}
        if self._liq is not None and len(rows):
            t_dep = rows[:, 0] + self._tau_delay
            late, early = t_dep > tau_fo_jet, t_dep < tau0
            d.update(n_droplets_late=int(late.sum()), E_droplets_late=float(e[late].sum()),
                     n_droplets_early=int(early.sum()),
                     E_droplets_early=float(e[early].sum()))
        else:
            d.update(n_droplets_late=0, E_droplets_late=0.0, n_droplets_early=0,
                     E_droplets_early=0.0)
        return d

    def _shower_rows(self):
        if self._g_sh is None:
            return {}
        rec = None
        if self._mgr is not None:
            try:
                rec = showers_from_manager(self._mgr)
            except Exception as exc:                  # noqa: BLE001 - keep the hydro pair
                warnings.warn(f"PairH5Writer: shower capture failed ({exc}); storing an "
                              "empty shower for this event", RuntimeWarning, stacklevel=4)
        for name, g in self._g_sh.items():
            g.append(None if rec is None else {name: getattr(rec, name)})
        if rec is None:
            return {"n_showers": 0, "n_partons": 0}
        return {"n_showers": int(rec.n_showers), "n_partons": int(len(rec.partons))}

    def _open(self, src, out):
        attrs = attrs_from_grids(src, out)
        growable = self._choose_ntau <= 0
        attrs["choose_ntau"] = max(1, out.ntau) if growable else self._choose_ntau
        extra = music_extra_attrs(src)
        extra.update(xscape_grid_mode=self._grid_mode, xscape_tau_stride=self._tau_stride,
                     xscape_writer="jetscape.pair_h5.PairH5Writer")
        extra.update(self._provenance_attrs())
        if self._liq is not None:
            extra.update({f"liquefier_{k}": v for k, v in liquefier_params(self._liq).items()})
        extra.update(self._extra_attrs)

        self._w = FnoH5Writer(self._out_file_name, attrs, nevents=0,
                              compression=self._compression, keep_bits=self._keep_bits,
                              chunk_events=1, chunk_tau=1,
                              growable_tau=growable, extra_attrs=extra, force=self._force)
        self._w.add_evolution("arr_bg", fo_suffix="_bg")
        if self._store_droplets:
            self._g_drop = self._w.ragged(
                "source", "offsets", {"droplets": (np.float64, (len(DROPLET_COLUMNS),))},
                attrs={"droplet_columns": list(DROPLET_COLUMNS),
                       "units": "tau in fm/c; x, y in fm; eta dimensionless; E, p in GeV",
                       "convention": (
                           "Droplets from X-SCAPE's CausalLiquefier, exactly as the jet leg "
                           "(arr) received them: Milne position (tau, x, y, eta), Cartesian "
                           "momentum (E, px, py, pz). Each deposits at tau + "
                           "liquefier_tau_delay. There is no source/S: MUSIC evaluates the "
                           "kernel per cell and step and keeps no gridded source.")})
        if self._store_showers:
            widths = {"partons": len(PARTON_COLUMNS), "vertices": len(VERTEX_COLUMNS),
                      "initiators": len(INITIATOR_COLUMNS)}
            offsets = {"partons": "parton_offsets", "vertices": "vertex_offsets",
                       "initiators": "initiator_offsets"}
            self._g_sh = {name: self._w.ragged("shower", offsets[name],
                                               {name: (np.float64, (w,))})
                          for name, w in widths.items()}
            for k, v in shower_group_attrs().items():
                self._w.f["shower"].attrs[k] = v
            self._w.f.attrs["has_shower"] = True

    def _provenance_attrs(self):
        p = {
            "generator": "xscape/MUSIC",
            "producer": "js-contrib/contribs/PyJetscape (jetscape.pair_h5.PairH5Writer)",
            "pairing": "bg_jet",
            "arr_is": (f"jet leg ({self._jet_id}): MUSIC with the CausalLiquefier source on "
                       "the background's initial condition"),
            "arr_bg_is": (f"background leg ({self._bg_id}): identical initial condition, no "
                          "jet source"),
            "source_model": ("causal_liquefier (droplets from X-SCAPE Matter+LBT), "
                             "point-sampled by MUSIC at its cell centres"),
            "source_mode": "xscape",
            "deposition": "on" if self._deposition else "off",
            "hard_vertex": "unknown",
            "eos_kind": "unknown",
            "transport_mode": "MUSIC",
        }
        p.update(self._provenance)
        return p

    def _check_pair(self, i, d, n_bg, n_jet):
        for leg in ("bg", "jet"):
            if d.get(f"{leg}_hit_boundary"):
                warnings.warn(
                    f"PairH5Writer: event {i}: the {leg} leg's freeze-out surface reached "
                    f"the transverse grid boundary, so MUSIC stopped it early and its "
                    f"evolution is truncated (diag/{leg}_hit_boundary). Enlarge "
                    f"<IS><grid_max_x>/<grid_max_y>.",
                    RuntimeWarning, stacklevel=4)
        if d["frames_identical"] == 0:
            warnings.warn(
                f"PairH5Writer: event {i}: the legs differ already in the first frame -- "
                "they did not start from the same initial condition, so arr - arr_bg is not "
                "the jet's effect.", RuntimeWarning, stacklevel=4)
        applied = d["E_droplets"] - d["E_droplets_late"] - d["E_droplets_early"]
        if (self._deposition and applied > 0 and n_bg == n_jet
                and d["frames_identical"] == n_jet):
            warnings.warn(
                f"PairH5Writer: event {i}: {applied:.2f} GeV of droplets should have "
                "reached the jet leg, but it is bit-identical to the background. MUSIC "
                "ignored the liquefier -- check that the MUSIC build has the jet source slot "
                "(music4gpu needs the add_hydro_source_terms_from_jet port).",
                RuntimeWarning, stacklevel=4)


def _hit_grid_boundary(hydro):
    """MpiMusic.get_hit_grid_boundary() if this build binds it, else None."""
    get = getattr(hydro, "get_hit_grid_boundary", None)
    return None if get is None else bool(get())


def _grid_mismatch(a, b):
    """'' if the two source grids agree spatially and in tau origin and step."""
    bad = []
    for k in ("nx", "ny", "neta"):
        if getattr(a, k) != getattr(b, k):
            bad.append(f"{k} {getattr(a, k)} vs {getattr(b, k)}")
    for k in ("x_min", "dx", "y_min", "dy", "eta_min", "deta", "tau_min", "dtau"):
        va, vb = getattr(a, k), getattr(b, k)
        if not np.isclose(va, vb, rtol=_GRID_RTOL, atol=1e-6):
            bad.append(f"{k} {va:g} vs {vb:g}")
    return ", ".join(bad)
