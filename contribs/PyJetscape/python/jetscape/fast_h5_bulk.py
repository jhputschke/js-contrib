"""
python/jetscape/fast_h5_bulk.py

H5BulkWriter -- a pure-Python JETSCAPE module that writes the bulk hydro evolution straight
to the FNO4d training HDF5 schema, covering both C++ bulk writer modules.

It inherits from ``pyjetscape.JetScapeModuleBase`` (via the PyJetScapeModuleBase C++
trampoline), so it can be added to a pipeline with ``JetScape.Add()`` and its ``Exec()``
runs once per event, exactly like the C++ writers.  See :mod:`jetscape.bulk_sources` for the
three source modes and what each one replaces.

Why this exists
---------------
* ``FastRootBulkWriter`` writes one event as a single ``std::vector<float>``, so events past
  2**30-2 bytes trip ROOT's ``TBufferFile::WriteByteCount`` 30-bit length field (the O+O
  native event is 1.29 GB; a 199-step one is 1.91 GB, near ROOT's hard ~2 GB wall).  HDF5
  has no such limit.
* The FNO training pipeline reads HDF5 and currently gets there by post-processing with
  ``root2hdf5/root_to_hdf5.py``.  Writing the target schema directly removes that step and
  the intermediate ROOT file.

Pipeline placement
------------------
    jetscape.Add(trento)
    jetscape.Add(null_predynamics)
    jetscape.Add(hydro)        # must come BEFORE the writer
    jetscape.Add(writer)       # <- here

Usage
-----
    from jetscape.fast_h5_bulk import H5BulkWriter

    writer = H5BulkWriter(out_file_name="hydro_evo.h5", grid_mode="native")
    jetscape.Add(writer)
    jetscape.Init(); jetscape.Exec(); jetscape.Finish()
    writer.Finish()   # must be called explicitly -- JetScape::Finish() calls FinishTasks(),
                      # which is a no-op; sub-task Finish() is never propagated.

``Finish()`` is idempotent, and the writer is a context manager, so the safe form is:

    with H5BulkWriter(...) as writer:
        ...

Reading it back
---------------
    from jetscape.fast_h5_bulk import read_fast_h5_bulk
    d = read_fast_h5_bulk("hydro_evo.h5", entry_stop=1)
    evo = d["events"][0]      # (ntau, nx, ny, neta, 4) float32, as read_fast_root_bulk gives

FNO4d's own loaders read the file directly -- ``read_3d_data_hdf5`` and ``MultiH5Array`` --
with no conversion.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

try:
    import h5py  # noqa: F401
    _H5PY_AVAILABLE = True
except ImportError:
    _H5PY_AVAILABLE = False

# The compiled core is needed to WRITE (H5BulkWriter is a framework module) but not to
# READ, so read_fast_h5_bulk works on a training machine with no X-SCAPE build -- the same
# split fast_root_bulk.py makes for uproot.
try:
    from .pyjetscape_core import JetScapeModuleBase, JetScapeSignalManager
    _CORE_AVAILABLE = True
except ImportError:                                   # pragma: no cover - reader-only install
    _CORE_AVAILABLE = False
    JetScapeModuleBase, JetScapeSignalManager = object, None

from .bulk_sources import (GRID_MODES, Grid, attrs_from_grids, event_array,
                           framework_store_bytes, music_extra_attrs)
from .fno_h5_writer import CHANNELS, FnoH5Writer

__all__ = ["H5BulkWriter", "read_fast_h5_bulk", "FEATURES"]

FEATURES = CHANNELS

#: warn above this much framework AoS in `framework` mode (FluidCellInfo is 112 B/cell)
_FRAMEWORK_WARN_BYTES = 4 << 30


class H5BulkWriter(JetScapeModuleBase):
    """Write the bulk hydro evolution to an FNO4d-schema HDF5 file.

    Parameters
    ----------
    out_file_name : str
        Output path.
    grid_mode : {"native", "grid", "framework"}
        Where the cells come from; see :mod:`jetscape.bulk_sources`.
    tau_stride : int
        ``native``/``grid`` only: keep every Nth stored MUSIC step.
    choose_ntau : int
        0 (default) grows the tau axis to the longest event in this file.  A positive value
        pins it, which is what makes files from separate jobs mergeable -- every file
        trained on together must share one ``choose_ntau`` or ``MultiH5Array`` raises.
        Longer events are then clipped, with a warning per event and a count at Finish().
    out_grid : mapping or bulk_sources.Grid, optional
        ``grid``/``framework`` only: any of ``x_min dx y_min dy eta_min deta tau_min dtau
        ntau``.  Missing/0 means "use the source grid's value".  A ``Grid`` (e.g.
        ``Grid.from_bounds``) is used exactly as given; see ``resolve_out_grid``.
    compression : str or None
        h5py compression for ``arr``.  ``"lzf"`` matches the existing reference files.
    clear_after_write : bool
        Release MUSIC's native store after each event.  Set False when another writer runs
        after this one in the same event and still needs the store.
    """

    def __init__(
        self,
        out_file_name: str = "hydro_evo.h5",
        grid_mode: str = "native",
        tau_stride: int = 1,
        choose_ntau: int = 0,
        out_grid: Optional[dict] = None,
        compression: Optional[str] = "lzf",
        clear_after_write: bool = True,
        force: bool = True,
        extra_attrs: Optional[dict] = None,
        verbose: bool = False,
    ) -> None:
        if not _CORE_AVAILABLE:
            raise ImportError(
                "H5BulkWriter needs the compiled pyjetscape_core extension; only "
                "read_fast_h5_bulk() is available without it.")
        if not _H5PY_AVAILABLE:
            raise ImportError("h5py is required by H5BulkWriter. Install with: pip install h5py")
        if grid_mode not in GRID_MODES:
            raise ValueError(f"grid_mode must be one of {GRID_MODES}, got {grid_mode!r}")

        super().__init__()
        self.SetId("H5BulkWriter")

        self._out_file_name = out_file_name
        self._grid_mode = grid_mode
        self._tau_stride = max(1, int(tau_stride))
        self._choose_ntau = max(0, int(choose_ntau))
        self._out_grid = out_grid if isinstance(out_grid, Grid) else dict(out_grid or {})
        self._compression = compression
        self._clear_after_write = bool(clear_after_write)
        self._force = bool(force)
        self._extra_attrs = dict(extra_attrs or {})
        self._verbose = bool(verbose)

        self._w: Optional[FnoH5Writer] = None
        self._i = 0
        self._n_clipped = 0
        self._src = None
        self._out = None
        self._warned_framework_size = False

    # ── JETSCAPE interface ──────────────────────────────────────────────────────
    def Init(self) -> None:
        """Reset per-run state.  Called once by JetScape::Init()."""
        if self._w is not None:
            self._w.close()
            self._w = None
        self._i = 0
        self._n_clipped = 0
        self._src = self._out = None
        print(f"H5BulkWriter: output -> {self._out_file_name}, grid_mode = "
              f"{self._grid_mode}, tau_stride = {self._tau_stride}, "
              f"choose_ntau = {self._choose_ntau or 'auto'}")

    def Exec(self) -> None:
        """Write the current event.  Called once per event by the framework."""
        hydro = JetScapeSignalManager.Instance().GetHydroPointer()
        if hydro is None:
            warnings.warn("H5BulkWriter: no hydro pointer found, skipping event.",
                          RuntimeWarning, stacklevel=2)
            return

        if self._grid_mode == "framework":
            self._warn_framework_size(hydro)

        try:
            arr, src, out = event_array(hydro, self._grid_mode, self._tau_stride,
                                        self._out_grid)
        except Exception as exc:                      # noqa: BLE001 - never kill the run
            warnings.warn(f"H5BulkWriter: skipping event {self._i}: {exc}",
                          RuntimeWarning, stacklevel=2)
            return

        ntau = int(arr.shape[0])
        if ntau == 0 or arr.size == 0:
            warnings.warn(f"H5BulkWriter: event {self._i} produced no tau frames, skipping.",
                          RuntimeWarning, stacklevel=2)
            return

        if self._w is None:
            self._open(src, out, ntau)
        elif (out.nx, out.ny, out.neta) != self._w.arr.shape[2:5]:
            # The file's spatial extent is fixed by event 0.  Without this the write would
            # fail with an opaque h5py broadcast error part-way through a long run.
            warnings.warn(
                f"H5BulkWriter: event {self._i} has grid "
                f"{(out.nx, out.ny, out.neta)} but the file was opened with "
                f"{tuple(self._w.arr.shape[2:5])}; skipping.",
                RuntimeWarning, stacklevel=2)
            return
        self._src, self._out = src, out

        n_write = ntau
        if not self._w.growable_tau and ntau > self._w.choose_ntau:
            n_write = self._w.choose_ntau
            self._n_clipped += 1
            warnings.warn(
                f"H5BulkWriter: event {self._i} has {ntau} tau frames but choose_ntau is "
                f"pinned at {n_write}; CLIPPING {ntau - n_write} frames of training data.",
                RuntimeWarning, stacklevel=2)

        i = self._i
        self._w.ensure_capacity(nevents=i + 1, choose_ntau=n_write)
        for t in range(n_write):
            # (nx, ny, neta, 4) -> (4, nx, ny, neta): with chunk_tau=1 this lands on exactly
            # one whole HDF5 chunk, so the frame-by-frame write costs no read-modify-write.
            self._w.write_frame(i, t, np.ascontiguousarray(arr[t].transpose(3, 0, 1, 2)))
        # tau_freezeout describes when the HYDRO ended, so it comes from the source grid,
        # not the output one -- FastRootBulkWriter.cc:133 (native) and :195 (grid) both use
        # tau_min + ntau_native*dtau off MUSIC's grid regardless of the output sampling.
        # (With tau_stride > 1 the C++ uses the unthinned count, so the two can differ by
        # up to one thinned step.)
        # MUSIC stops an event whose freeze-out surface reaches the grid edge; record
        # it, because the stored evolution is then truncated.
        get_hit = getattr(hydro, "get_hit_grid_boundary", None)
        if get_hit is not None:
            hit = bool(get_hit())
            self._w.write_diag(i, hit_grid_boundary=int(hit))
            if hit:
                warnings.warn(
                    f"H5BulkWriter: event {i}: the freeze-out surface reached the "
                    "transverse grid boundary, so MUSIC stopped the evolution early "
                    "(diag/hit_grid_boundary). Enlarge <IS><grid_max_x>/<grid_max_y>.",
                    RuntimeWarning, stacklevel=2)
        self._w.set_event_meta(i, n_write, src.tau_min + src.ntau * src.dtau)
        self._i += 1

        del arr
        if self._verbose:
            print(f"H5BulkWriter: event {i} -> {n_write} tau frames "
                  f"({out.nx}x{out.ny}x{out.neta}), tau_freezeout = "
                  f"{src.tau_min + src.ntau * src.dtau:.3f} fm/c")

        if self._clear_after_write and self._grid_mode != "framework":
            if hasattr(hydro, "clear_hydro_info_from_memory"):
                hydro.clear_hydro_info_from_memory()

    def Clear(self) -> None:
        pass

    def Finish(self) -> None:
        """Close the file.  Idempotent; must be called explicitly (see module docstring)."""
        if self._w is None:
            return
        n, t = self._w.nevents, self._w.choose_ntau
        self._w.close()
        self._w = None
        print(f"H5BulkWriter: wrote {n} event(s) to {self._out_file_name} "
              f"(choose_ntau = {t})")
        if self._choose_ntau <= 0 and n:
            print(f"H5BulkWriter: pass choose_ntau={t} to make further runs mergeable "
                  "with this file (MultiH5Array requires one shared value).")
        if self._n_clipped:
            print(f"H5BulkWriter: WARNING -- {self._n_clipped} event(s) were clipped to "
                  f"choose_ntau={t}.")

    # ── accessors (parity with the C++ FastRootBulkWriter bindings) ─────────────
    def GetOutFileName(self) -> str:
        return self._out_file_name

    def GetGridMode(self) -> str:
        return self._grid_mode

    def GetTauStride(self) -> int:
        return self._tau_stride

    def IsFileOpen(self) -> bool:
        return self._w is not None

    def GetNumberOfEventsWritten(self) -> int:
        return self._i

    def get_event_layout(self) -> dict:
        """Layout of the most recently written event."""
        o = self._out
        return {
            "out_file_name": self._out_file_name,
            "grid_mode": self._grid_mode,
            "tau_stride": self._tau_stride,
            "events_written": self._i,
            "choose_ntau": self._w.choose_ntau if self._w else 0,
            "shape": None if o is None else (len(FEATURES), o.nx, o.ny, o.neta),
            "tau_min": None if o is None else o.tau_min,
            "dtau": None if o is None else o.dtau,
        }

    # ── internals ───────────────────────────────────────────────────────────────
    def _open(self, src, out, ntau):
        attrs = attrs_from_grids(src, out)
        growable = self._choose_ntau <= 0
        attrs["choose_ntau"] = ntau if growable else self._choose_ntau

        extra = music_extra_attrs(src)
        extra.update(xscape_grid_mode=self._grid_mode,
                     xscape_tau_stride=self._tau_stride,
                     xscape_writer="jetscape.fast_h5_bulk.H5BulkWriter")
        extra.update(self._extra_attrs)

        self._w = FnoH5Writer(
            self._out_file_name, attrs, nevents=0,
            compression=self._compression, chunk_events=1, chunk_tau=1,
            growable_tau=growable, extra_attrs=extra, force=self._force)

    def _warn_framework_size(self, hydro):
        if self._warned_framework_size:
            return
        self._warned_framework_size = True
        try:
            nbytes = framework_store_bytes(hydro.get_bulk_info())
        except Exception:                             # noqa: BLE001
            return
        if nbytes > _FRAMEWORK_WARN_BYTES:
            warnings.warn(
                f"H5BulkWriter: grid_mode='framework' holds the framework AoS in memory "
                f"(~{nbytes / (1 << 30):.1f} GB at this grid). Use grid_mode='grid' to read "
                "MUSIC's native store instead.", RuntimeWarning, stacklevel=3)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.Finish()
        return False

    def __del__(self):
        try:
            self.Finish()
        except Exception:                             # noqa: BLE001 - interpreter teardown
            pass


# ─────────────────────────────────────────────────────────────────────── reader
def read_fast_h5_bulk(
    path: str,
    entry_start: Optional[int] = None,
    entry_stop: Optional[int] = None,
) -> dict:
    """Load a file written by :class:`H5BulkWriter`.

    Returns the same dict shape as ``jetscape.fast_root_bulk.read_fast_root_bulk``, so the
    two are interchangeable for plotting: each event is ``(ntau, nx, ny, neta, 4)`` float32,
    trimmed to that event's own ``ntau_freezeout``.

    Native-mode events are large, so use ``entry_start`` / ``entry_stop`` to load a subset.
    FNO4d's ``read_3d_data_hdf5`` reads the same file in its own (channel-first) layout.
    """
    if not _H5PY_AVAILABLE:
        raise ImportError("h5py is required to read these files. Install with: pip install h5py")
    import h5py

    with h5py.File(path, "r") as f:
        params = {k: _scalar(v) for k, v in f.attrs.items()}
        sl = slice(entry_start, entry_stop)
        ntau = np.asarray(f["ntau_freezeout"][sl])
        tau_f = np.asarray(f["tau_freezeout"][sl])
        # stored (nev, 4, nx, ny, neta, T) -> per event (ntau, nx, ny, neta, 4)
        # f["arr"][i, ..., :n] is (4, nx, ny, neta, ntau); the ROOT reader hands out
        # (ntau, nx, ny, neta, 4), so feature -> last and tau -> first.
        events = [np.ascontiguousarray(
                      np.moveaxis(f["arr"][i, ..., :int(n)], (0, 4), (4, 0)))
                  for i, n in zip(range(*sl.indices(f["arr"].shape[0])), ntau)]

    grid = {k: params.get(k) for k in
            ("nx", "ny", "neta", "x_min", "dx", "y_min", "dy",
             "eta_min", "deta", "tau_min", "dtau")}
    for axis in ("x", "y", "eta"):
        lo, step = grid[f"{axis}_min"], grid[f"d{axis}"]
        n = grid[{"x": "nx", "y": "ny", "eta": "neta"}[axis]]
        grid[axis] = None if lo is None else lo + step * np.arange(n)

    return {
        "events": events,
        "ntau": ntau,
        "tau_freezeout": tau_f,
        "grid_mode": params.get("xscape_grid_mode", "native"),
        "features": FEATURES,
        "grid": grid,
        "params": params,
    }


def _scalar(v):
    """h5py gives back 0-d arrays and bytes; hand out plain Python where it is one value."""
    if isinstance(v, np.ndarray) and v.ndim == 0:
        v = v[()]
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.generic):
        return v.item()
    return v
