"""
python/jetscape/fno_h5_writer.py

The FNO4d training HDF5 schema, written one tau frame at a time.

VENDORED from FNO4d ``loc_libs/fast_data/writer.py`` (``FnoH5Writer``).  FNO4d is not a
dependency of PyJetscape and that file is self-contained (stdlib + numpy + h5py), so it is
copied rather than imported.  The invariants below are pinned on the FNO4d side by
``FNO4d/tests/test_fast_data_writer.py`` -- check there first if the two ever disagree.

Changes from the FNO4d original:

* The ``/source``, ``/diag`` and ``/eos`` groups are dropped.  They carry fast_data's own
  droplet source term and equation of state, which a MUSIC run does not produce, and they
  are optional in the schema (no FNO4d reader touches them).
* ``arr`` is created with ``maxshape=(None, 4, nx, ny, neta, None)`` and grown with
  ``resize()``.  The original pre-allocates, which requires knowing both ``nevents`` and
  ``choose_ntau`` up front; a MUSIC run knows neither until the events have run.
* The tau chunk extent is configurable and defaults to 1, so a single tau frame can be
  written without a read-modify-write of the whole event.  See ``write_frame``.

``arr`` is the contract every FNO4d loader depends on:

    arr             (nevents, 4, nx, ny, neta, choose_ntau) float32
    ntau_freezeout  (nevents,) int32
    tau_freezeout   (nevents,) float32

with root attributes ``nFeatures, nx, ny, neta, x_min, y_min, eta_min, dx, dy, deta,
tau_min, tau_min_MUSIC, dtau`` (FNO4d ``loc_libs/read_3d_hdf5.py`` ``_SCALAR_KEYS``) plus
``choose_ntau`` and ``nevents``.

Three details are load-bearing and were checked against FNO4d's data/dAu_25ev_mb.h5:

* The four channels are (energy_density, vx, vy, vz) with CARTESIAN LAB three-velocities.
* ``attrs['nevents']`` must equal ``arr.shape[0]``: read_3d_data_hdf5 trusts the attribute
  while MultiH5Array trusts the shape, and they must agree.
* Channel 0 must be exactly zero after freeze-out.  loc_libs/data/dataset.py:live_tau_lengths
  derives each event's lifetime from ``abs(ev[0]).max(axis=(0,1,2)) > 0``, not from
  ``ntau_freezeout``, so a non-zero tail makes every window look alive.  Frames past the end
  of an event are never written, and HDF5 returns the fill value (exactly 0.0) for the
  unallocated chunks, so this holds without an explicit pad.

Anything extra written into the root attributes is returned for free by
``read_3d_data_hdf5`` (which returns ``dict(hf.attrs)``), which is how run provenance travels
to the training side.
"""

from __future__ import annotations

import gc
import os

import numpy as np

__all__ = ["FnoH5Writer", "grid_attrs", "repad_to", "FORMAT", "FORMAT_VERSION",
           "SCALAR_KEYS", "CHANNELS"]

FORMAT = "xscape/hydro_evolution"
FORMAT_VERSION = 1

#: mirrors FNO4d loc_libs/read_3d_hdf5.py `_SCALAR_KEYS`
SCALAR_KEYS = ("nFeatures", "nx", "ny", "neta", "x_min", "y_min", "eta_min",
               "dx", "dy", "deta", "tau_min", "tau_min_MUSIC", "dtau")

CHANNELS = ("energy_density", "vx", "vy", "vz")

VELOCITY_CONVENTION = (
    "Cartesian LAB three-velocities, as produced by X-SCAPE HydroinfoMUSIC: "
    "utau = sqrt(1+ux^2+uy^2+ueta^2) with ueta the ORTHONORMAL Milne component (tau*u^eta); "
    "ut = utau*cosh(eta)+ueta*sinh(eta); uz = utau*sinh(eta)+ueta*cosh(eta); "
    "v = (ux,uy,uz)/ut."
)
FREEZEOUT_CONVENTION = (
    "ntau_freezeout = number of tau frames actually written for this event, so the last "
    "written frame is at tau_min + (ntau_freezeout-1)*dtau. tau_freezeout is when the "
    "HYDRO ended, taken from the SOURCE grid (tau_min_MUSIC + ntau_source*dtau_source) as "
    "X-SCAPE FastRootBulkWriter does, so on a resampled output grid it is NOT "
    "tau_min + ntau_freezeout*dtau. Every channel is exactly zero from frame "
    "ntau_freezeout onward (those chunks are never allocated, so HDF5 returns the fill "
    "value)."
)


def grid_attrs(nx, ny, neta, dx, dy, deta, tau_min, dtau, choose_ntau,
               x_min=None, y_min=None, eta_min=None, tau_min_MUSIC=None):
    """The root attribute dict for a cell-centred grid symmetric about the origin.

    The mins default to -(n-1)/2*d, so the stored x_min is the first cell CENTRE (as in the
    MUSIC-produced files), not a box edge.
    """
    half = lambda n, d: -0.5 * (n - 1) * d
    return {
        "nFeatures": 4,
        "nx": int(nx), "ny": int(ny), "neta": int(neta),
        "dx": float(dx), "dy": float(dy), "deta": float(deta),
        "x_min": float(half(nx, dx) if x_min is None else x_min),
        "y_min": float(half(ny, dy) if y_min is None else y_min),
        "eta_min": float(half(neta, deta) if eta_min is None else eta_min),
        "tau_min": float(tau_min),
        "tau_min_MUSIC": float(tau_min if tau_min_MUSIC is None else tau_min_MUSIC),
        "dtau": float(dtau),
        "choose_ntau": int(choose_ntau),
    }


class FnoH5Writer:
    """Stream events into an FNO4d-schema file, one tau frame at a time.

    ``arr`` grows along the event axis and the tau axis as events arrive, so neither count
    has to be known up front.  ``attrs['nevents']`` and ``attrs['choose_ntau']`` are kept
    equal to ``arr.shape[0]`` and ``arr.shape[5]`` after every write, and the file is
    flushed per event, so a run killed mid-way still produces a file both
    ``read_3d_data_hdf5`` and ``MultiH5Array`` agree on.

    Parameters
    ----------
    path : str
        Output file path.
    attrs : mapping
        Must contain every key in :data:`SCALAR_KEYS` plus ``choose_ntau`` (the initial tau
        extent; the dataset grows past it as needed unless ``growable_tau=False``).
    nevents : int
        Initial event-axis extent.  0 is fine -- the axis grows per event.
    compression : str or None
        h5py compression for ``arr``.  ``"lzf"`` (default) matches the existing reference
        files and decompresses 2-3x faster than gzip on the training side.
    chunk_events, chunk_tau : int
        Chunk extent on the event and tau axes.  ``chunk_tau=1`` is what makes a single
        frame write cover exactly one whole chunk (no read-modify-write) and what makes the
        zero padding of short events cost nothing on disk.
    growable_tau : bool
        False pins the tau extent at ``attrs['choose_ntau']``, so every file from a campaign
        shares one value.  Callers must then clip longer events themselves.
    """

    def __init__(self, path, attrs, nevents=0, *, compression="lzf",
                 chunk_events=1, chunk_tau=1, extra_attrs=None,
                 growable_tau=True, force=False):
        import h5py

        self.path = str(path)
        self._closed = False
        self.growable_tau = bool(growable_tau)

        if os.path.exists(self.path) and not force:
            raise FileExistsError(f"{self.path} already exists (pass force=True)")
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)

        nx, ny, nz = int(attrs["nx"]), int(attrs["ny"]), int(attrs["neta"])
        t0 = max(1, int(attrs["choose_ntau"]))
        n0 = max(0, int(nevents))

        self.f = h5py.File(self.path, "w")
        a = self.f.attrs
        for k in SCALAR_KEYS:
            a[k] = attrs[k]
        a["choose_ntau"] = t0
        a["nevents"] = n0                       # must equal arr.shape[0]
        a["format"], a["format_version"] = FORMAT, FORMAT_VERSION
        a["feature_names"] = list(CHANNELS)
        a["units"] = "energy_density: GeV/fm^3; v: dimensionless (Cartesian lab); lengths: fm"
        a["velocity_convention"] = VELOCITY_CONVENTION
        a["freezeout_convention"] = FREEZEOUT_CONVENTION
        a["nevents_written"], a["complete"] = 0, False
        for k, v in (extra_attrs or {}).items():
            a[k] = v

        ce = max(1, int(chunk_events))
        ct = max(1, int(chunk_tau))
        self.arr = self.f.create_dataset(
            "arr", (n0, 4, nx, ny, nz, t0), dtype=np.float32,
            maxshape=(None, 4, nx, ny, nz, None if self.growable_tau else t0),
            chunks=(ce, 4, nx, ny, nz, min(ct, t0)),
            compression=compression)
        # Contiguous datasets cannot be extended, so these are chunked (invisible
        # downstream -- no FNO4d reader inspects dataset layout).
        self.ntau_fo = self.f.create_dataset(
            "ntau_freezeout", (n0,), maxshape=(None,), dtype=np.int32, chunks=(256,))
        self.tau_fo = self.f.create_dataset(
            "tau_freezeout", (n0,), maxshape=(None,), dtype=np.float32, chunks=(256,))

    # ------------------------------------------------------------------ capacity
    @property
    def nevents(self):
        return int(self.arr.shape[0])

    @property
    def choose_ntau(self):
        return int(self.arr.shape[5])

    def ensure_capacity(self, nevents=None, choose_ntau=None):
        """Grow the event and/or tau axis.  Returns True if anything was resized.

        The tau axis never grows past its creation size when ``growable_tau=False``; the
        caller is responsible for clipping in that case.
        """
        self._check_open()
        changed = False
        if nevents is not None and int(nevents) > self.arr.shape[0]:
            n = int(nevents)
            self.arr.resize(n, axis=0)
            self.ntau_fo.resize((n,))
            self.tau_fo.resize((n,))
            changed = True
        if (choose_ntau is not None and self.growable_tau
                and int(choose_ntau) > self.arr.shape[5]):
            self.arr.resize(int(choose_ntau), axis=5)
            changed = True
        if changed:
            self.f.attrs["nevents"] = self.arr.shape[0]
            self.f.attrs["choose_ntau"] = self.arr.shape[5]
        return changed

    # ------------------------------------------------------------------ writing
    def write_frame(self, i, t, frame):
        """Write one tau frame of event `i`.  `frame` is (4, nx, ny, neta).

        With ``chunk_tau=1`` this covers exactly one whole chunk, so HDF5 never reads a
        chunk back to merge.
        """
        self._check_open()
        self.arr[i, :, :, :, :, t] = np.asarray(frame, dtype=np.float32)

    def set_event_meta(self, i, ntau_fo, tau_fo):
        """Record event `i`'s freeze-out scalars and mark it written."""
        self._check_open()
        self.ntau_fo[i] = int(ntau_fo)
        self.tau_fo[i] = float(tau_fo)
        self.f.attrs["nevents_written"] = max(
            int(self.f.attrs.get("nevents_written", 0)), int(i) + 1)
        self.f.flush()
        gc.collect()

    def append_event(self, i, arr_ev, ntau_fo, tau_fo):
        """Write event `i` in one call.  `arr_ev` is (4, nx, ny, neta, ntau).

        Kept for parity with the FNO4d original.  It holds the whole event in memory; the
        streaming path is ``write_frame`` + ``set_event_meta``.
        """
        self._check_open()
        arr_ev = np.asarray(arr_ev, dtype=np.float32)
        self.ensure_capacity(nevents=i + 1, choose_ntau=arr_ev.shape[-1])
        self.arr[i, :, :, :, :, :arr_ev.shape[-1]] = arr_ev
        self.set_event_meta(i, ntau_fo, tau_fo)

    # ------------------------------------------------------------------ teardown
    def close(self, complete=None):
        if self._closed:
            return
        n_written = int(self.f.attrs.get("nevents_written", 0))
        self.f.attrs["nevents"] = self.arr.shape[0]
        self.f.attrs["choose_ntau"] = self.arr.shape[5]
        self.f.attrs["complete"] = bool(
            n_written >= self.arr.shape[0] if complete is None else complete)
        self.f.close()
        self._closed = True

    @property
    def closed(self):
        return self._closed

    def _check_open(self):
        if self._closed:
            raise RuntimeError("writer is closed")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# ───────────────────────────────────────────── reconciling files across jobs
#: datasets that carry the tau axis, as (path, axis).  "arr" is required; the fast_data
#: /source group is optional but must be grown with it or the file goes inconsistent.
_TAU_DATASETS = (("arr", 5), ("source/S", 5), ("source/P_cart", 1))


def repad_to(paths, choose_ntau=None, *, dry_run=False, verbose=True):
    """Grow the tau axis of several files to one ``choose_ntau``, in place.

    ``arr`` is rectangular, so every file trained on together must share one
    ``choose_ntau`` or ``MultiH5Array`` raises ``Dimension mismatch``.  Files written here
    have ``maxshape=(None, 4, nx, ny, neta, None)``, so reconciling them is a metadata
    resize: no data moves, the file does not grow (the new region is unallocated chunks
    that read back as the fill value, exactly 0.0), and per-event lifetimes survive.

    This is the h5-to-h5 counterpart of ``root_to_hdf5.py --global-ntau``.

    Parameters
    ----------
    paths : str or sequence of str
        Files to reconcile.
    choose_ntau : int, optional
        Target tau extent.  Default: the largest among ``paths``.  Pinning a *larger*
        value than any file needs is free, so passing the campaign-wide number is safe.
    dry_run : bool
        Report what would change without writing.
    verbose : bool
        Print a line per file.

    Returns
    -------
    (target, changed) : (int, list of str)
        The tau extent applied and the files that needed it.

    Raises
    ------
    ValueError
        If the files disagree on (nFeatures, nx, ny, neta) -- they could never be
        concatenated; if any file is already LONGER than the target, since shrinking an
        HDF5 dataset discards data permanently; or if a file's tau axis is not growable
        (e.g. written by FNO4d's own pre-allocating ``FnoH5Writer``), which needs a
        rewrite rather than a resize.
    """
    import h5py

    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    paths = [str(p) for p in paths]
    if not paths:
        raise ValueError("repad_to: no files given")

    info = {}
    for path in paths:
        with h5py.File(path, "r") as f:
            ds = f["arr"]
            present = [(name, ax) for name, ax in _TAU_DATASETS if name in f]
            extents = {name: int(f[name].shape[ax]) for name, ax in present}
            if len(set(extents.values())) > 1:
                raise ValueError(
                    f"repad_to: {path} is already inconsistent -- its tau-bearing "
                    f"datasets disagree: {extents}")
            info[path] = {
                "extent": int(ds.shape[5]),
                "dims": tuple(int(n) for n in ds.shape[1:5]),
                "growable": all(f[name].maxshape[ax] is None for name, ax in present),
                "attr": int(f.attrs["choose_ntau"]),
                "datasets": present,
            }

    dims = {v["dims"] for v in info.values()}
    if len(dims) > 1:
        raise ValueError(
            "repad_to: files disagree on (nFeatures, nx, ny, neta) and could never be "
            "concatenated: "
            + ", ".join(f"{p}={v['dims']}" for p, v in info.items()))

    target = max(v["extent"] for v in info.values()) if not choose_ntau else int(choose_ntau)

    longer = {p: v["extent"] for p, v in info.items() if v["extent"] > target}
    if longer:
        raise ValueError(
            f"repad_to: choose_ntau={target} is smaller than "
            + ", ".join(f"{p} ({n})" for p, n in longer.items())
            + ". Shrinking an HDF5 dataset discards data permanently; pass "
              f"choose_ntau >= {max(longer.values())} instead.")

    stuck = [p for p, v in info.items() if v["extent"] < target and not v["growable"]]
    if stuck:
        raise ValueError(
            "repad_to: the tau axis is not growable (no maxshape) in "
            + ", ".join(stuck)
            + " -- it predates the extendible-tau change, so it must be rewritten "
              "rather than resized.")

    changed = []
    for path, v in info.items():
        if v["extent"] == target and v["attr"] == target:
            if verbose:
                print(f"  {path}: already at choose_ntau={target}")
            continue
        changed.append(path)
        if verbose:
            print(f"  {path}: choose_ntau {v['extent']} -> {target}"
                  + (" (dry run)" if dry_run else ""))
        if dry_run:
            continue
        with h5py.File(path, "r+") as f:
            for name, ax in v["datasets"]:
                if f[name].shape[ax] < target:
                    f[name].resize(target, axis=ax)
            f.attrs["choose_ntau"] = target
    return target, changed
