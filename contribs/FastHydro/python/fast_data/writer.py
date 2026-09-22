"""The FNO4d training HDF5 schema, written one event at a time.

`arr` is the contract every FNO4d loader depends on:

    arr             (nevents, 4, nx, ny, neta, ntau) float32, chunks (1,4,nx,ny,neta,ntau), lzf
    ntau_freezeout  (nevents,) int32
    tau_freezeout   (nevents,) float32

with root attributes `nFeatures, nx, ny, neta, x_min, y_min, eta_min, dx, dy, deta, tau_min,
tau_min_MUSIC, dtau` (loc_libs/read_3d_hdf5.py `_SCALAR_KEYS`) plus `choose_ntau` and `nevents`.

Three details are load-bearing and were checked against data/dAu_25ev_mb.h5:

* The four channels are (energy_density, vx, vy, vz) with CARTESIAN LAB three-velocities.
* `attrs['nevents']` must equal `arr.shape[0]`: read_3d_data_hdf5 trusts the attribute while
  MultiH5Array trusts the shape, and they must agree.
* Channel 0 must be exactly zero after freeze-out.  loc_libs/data/dataset.py:live_tau_lengths
  derives each event's lifetime from `abs(ev[0]).max(axis=(0,1,2)) > 0`, not from
  `ntau_freezeout`, so a non-zero tail makes every window look alive.  Measured on the real data:
  `ntau_freezeout == (non-zero frames) + 1` and `tau_freezeout == tau_min + (ntau_freezeout-1)*dtau`.

Anything extra written into the root attributes is returned for free by `read_3d_data_hdf5`
(which returns `dict(hf.attrs)`), which is how fast_data provenance travels to the training side.
"""

from __future__ import annotations

import gc
import os

import numpy as np

__all__ = ["FnoH5Writer", "write_fno_h5", "FORMAT", "FORMAT_VERSION", "SCALAR_KEYS", "CHANNELS"]

FORMAT = "fast_data/hydro_evolution"
FORMAT_VERSION = 1

#: mirrors loc_libs/read_3d_hdf5.py `_SCALAR_KEYS`
SCALAR_KEYS = ("nFeatures", "nx", "ny", "neta", "x_min", "y_min", "eta_min",
               "dx", "dy", "deta", "tau_min", "tau_min_MUSIC", "dtau")

CHANNELS = ("energy_density", "vx", "vy", "vz")

SOURCE_CONVENTION = (
    "S[...,t] = sum over hydro substeps in (tau[t-1], tau[t]] of Delta(tau*T^{tau nu}), "
    "nu ordering (tau, x, y, eta), CONTRAVARIANT Milne components, mostly-plus metric "
    "g = diag(-1, 1, 1, tau^2).  S[...,0] = 0 because frame 0 is the initial condition.  "
    "arr[...,t] ALREADY CONTAINS S[...,t] -- arr is the single evolution with the source in it."
)
SOURCE_UNITS = (
    "per cell, already integrated over the record interval: NOT divided by dtau and NOT "
    "multiplied by the cell volume dV = dx*dy*deta.  Same units as the conserved state "
    "q = tau*T^{tau nu}: GeV/fm^2 for nu in (tau,x,y), GeV/fm^3 for nu = eta."
)


def grid_attrs(nx, ny, neta, dx, dy, deta, tau_min, dtau, choose_ntau,
               x_min=None, y_min=None, eta_min=None, tau_min_MUSIC=None):
    """The root attribute dict for a cell-centred grid symmetric about the origin.

    The mins default to -(n-1)/2*d, matching both fv.Grid and glauber.centres, so the stored
    x_min is the first cell CENTRE (as in the MUSIC-produced files), not a box edge.
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
        # No MUSIC stage here, so the two coincide; kept because it is in _SCALAR_KEYS.
        "tau_min_MUSIC": float(tau_min if tau_min_MUSIC is None else tau_min_MUSIC),
        "dtau": float(dtau),
        "choose_ntau": int(choose_ntau),
    }


class FnoH5Writer:
    """Stream events into an FNO4d-schema file.

    The datasets are created at full `nevents` up front so the file is structurally valid from
    the first flush; `nevents_written` and `complete` track progress for --resume.  An incomplete
    file loads, but its unwritten tail events are all zeros -- do not train on one.
    """

    def __init__(self, path, attrs, nevents, *, compression="lzf", source_compression="gzip",
                 chunk_events=1, write_source=False, write_diagnostics=True,
                 extra_attrs=None, np_eos=None, force=False, resume=False):
        import h5py

        self.path, self.nevents = str(path), int(nevents)
        self.write_source = bool(write_source)
        self.write_diagnostics = bool(write_diagnostics)
        self._droplets, self._offsets, self._diag = [], [0], []
        self._closed = False

        exists = os.path.exists(self.path)
        if exists and not (force or resume):
            raise FileExistsError(f"{self.path} already exists (pass force=True or resume=True)")
        if resume and not exists:
            resume = False
        self.resuming = resume

        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
        if resume:
            self.f = h5py.File(self.path, "r+")
            self.start = int(self.f.attrs.get("nevents_written", 0))
            if int(self.f.attrs["nevents"]) != self.nevents:
                raise ValueError(f"cannot resume: file holds {int(self.f.attrs['nevents'])} events, "
                                 f"config asks for {self.nevents}")
            self.arr = self.f["arr"]
            self.S = self.f["source/S"] if "source/S" in self.f else None
            self._offsets = list(self.f["source/offsets"][:]) if "source/offsets" in self.f else [0]
            if "source/droplets" in self.f and len(self.f["source/droplets"]):
                self._droplets = [self.f["source/droplets"][:]]
            return

        nx, ny, nz, T = (attrs["nx"], attrs["ny"], attrs["neta"], attrs["choose_ntau"])
        self.start = 0
        self.f = h5py.File(self.path, "w")
        a = self.f.attrs
        for k in SCALAR_KEYS:
            a[k] = attrs[k]
        a["choose_ntau"] = int(attrs["choose_ntau"])
        a["nevents"] = self.nevents                 # must equal arr.shape[0]
        a["format"], a["format_version"] = FORMAT, FORMAT_VERSION
        a["feature_names"] = list(CHANNELS)
        a["units"] = "energy_density: GeV/fm^3; v: dimensionless (Cartesian lab); lengths: fm"
        a["velocity_convention"] = (
            "Cartesian LAB three-velocities, matching X-SCAPE FastRootBulkWriter / "
            "HydroinfoMUSIC: utau = sqrt(1+ux^2+uy^2+ueta^2) with ueta the ORTHONORMAL Milne "
            "component (tau*u^eta); ut = utau*cosh(eta)+ueta*sinh(eta); "
            "uz = utau*sinh(eta)+ueta*cosh(eta); v = (ux,uy,uz)/ut."
        )
        a["freezeout_convention"] = (
            "ntau_freezeout = 1 + (last frame index with max-over-cells T >= T_fo), or 1 if never; "
            "tau_freezeout = tau_min + (ntau_freezeout-1)*dtau; every channel is exactly zero "
            "from frame ntau_freezeout onward."
        )
        a["nevents_written"], a["complete"] = 0, False
        a["has_source"] = self.write_source
        for k, v in (extra_attrs or {}).items():
            a[k] = v

        ce = max(1, int(chunk_events))
        # The tau axis is left extendible so files from separate jobs can be reconciled
        # to a common choose_ntau afterwards, in place, without a rewrite (see
        # X-SCAPE js-contrib PyJetscape `python -m jetscape.repad_h5`).  Growing it costs
        # nothing: the added region is unallocated chunks that read back as the fill
        # value, exactly 0.0, which is what live_tau_lengths requires.  The event axis
        # stays fixed -- this writer pre-allocates it on purpose.
        self.arr = self.f.create_dataset(
            "arr", (self.nevents, 4, nx, ny, nz, T), dtype=np.float32,
            maxshape=(self.nevents, 4, nx, ny, nz, None),
            chunks=(ce, 4, nx, ny, nz, T), compression=compression)
        self.f.create_dataset("ntau_freezeout", (self.nevents,), dtype=np.int32)
        self.f.create_dataset("tau_freezeout", (self.nevents,), dtype=np.float32)

        self.S = None
        if self.write_source:
            g = self.f.create_group("source")
            kw = {"compression": source_compression}
            if source_compression == "gzip":
                kw.update(compression_opts=4, shuffle=True)
            self.S = g.create_dataset("S", (self.nevents, 4, nx, ny, nz, T), dtype=np.float32,
                                      maxshape=(self.nevents, 4, nx, ny, nz, None),
                                      chunks=(ce, 4, nx, ny, nz, T), **kw)
            g.create_dataset("P_cart", (self.nevents, T, 4), dtype=np.float64,
                             maxshape=(self.nevents, None, 4), chunks=(1, T, 4))
            g.attrs["convention"] = SOURCE_CONVENTION
            g.attrs["units"] = SOURCE_UNITS
            g.attrs["channels"] = ["S_tau", "S_x", "S_y", "S_eta"]

        if np_eos is not None:
            from .eos import write_eos_group
            write_eos_group(self.f, np_eos)

    # ------------------------------------------------------------------ writing
    def append_event(self, i, arr_ev, ntau_fo, tau_fo, *, S_ev=None, P_cart=None,
                     droplets=None, diag=None):
        """Write event `i`.  `arr_ev` is (4, nx, ny, neta, ntau) and already tail-zeroed."""
        if self._closed:
            raise RuntimeError("writer is closed")
        self.arr[i] = np.asarray(arr_ev, dtype=np.float32)
        self.f["ntau_freezeout"][i] = int(ntau_fo)
        self.f["tau_freezeout"][i] = float(tau_fo)
        if self.S is not None and S_ev is not None:
            self.S[i] = np.asarray(S_ev, dtype=np.float32)
        if self.S is not None and P_cart is not None:
            self.f["source/P_cart"][i] = np.asarray(P_cart, dtype=np.float64)
        if self.write_source:
            d = np.zeros((0, 8)) if droplets is None else np.asarray(droplets, dtype=np.float64)
            self._droplets.append(d)
            self._offsets.append(self._offsets[-1] + len(d))
        if self.write_diagnostics and diag is not None:
            self._diag.append(dict(diag))
        self.f.attrs["nevents_written"] = int(i) + 1
        self.f.flush()
        gc.collect()

    def close(self, complete=None):
        if self._closed:
            return
        if self.write_source and "source" in self.f:
            g = self.f["source"]
            data = (np.concatenate(self._droplets, 0) if self._droplets
                    else np.zeros((0, 8), dtype=np.float64))
            for name, val in (("droplets", data),
                              ("offsets", np.asarray(self._offsets, dtype=np.int64))):
                if name in g:
                    del g[name]
                g.create_dataset(name, data=val)
            g.attrs["droplet_columns"] = ["tau", "x", "y", "eta", "E", "px", "py", "pz"]
        if self.write_diagnostics and self._diag:
            g = self.f.require_group("diag")
            for k in sorted({k for d in self._diag for k in d}):
                vals = [d.get(k, np.nan) for d in self._diag]
                if k in g:
                    del g[k]
                # seeds are uint64 and overflow int64, which numpy would silently widen to float
                dtype = np.uint64 if k.endswith("seed") else None
                try:
                    g.create_dataset(k, data=np.asarray(vals, dtype=dtype))
                except (TypeError, ValueError, OverflowError):
                    g.create_dataset(k, data=np.asarray([str(v) for v in vals], dtype="S64"))
        n_written = int(self.f.attrs.get("nevents_written", 0))
        self.f.attrs["complete"] = bool(n_written >= self.nevents if complete is None else complete)
        self.f.close()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def write_fno_h5(path, events, attrs, *, ntau_freezeout=None, tau_freezeout=None,
                 extra_attrs=None, compression="lzf", force=True, **kw):
    """One-shot convenience writer: `events` is a sequence of (4,nx,ny,neta,ntau) arrays, or of
    (arr, ntau_fo, tau_fo) tuples as produced by gubser.make_event."""
    evs, nfo, tfo = [], [], []
    for k, ev in enumerate(events):
        if isinstance(ev, tuple):
            a, n, t = ev
        else:
            a = ev
            n = attrs["choose_ntau"] if ntau_freezeout is None else ntau_freezeout[k]
            t = (attrs["tau_min"] + (n - 1) * attrs["dtau"]) if tau_freezeout is None else tau_freezeout[k]
        evs.append(a); nfo.append(n); tfo.append(t)
    with FnoH5Writer(path, attrs, len(evs), compression=compression, force=force,
                     extra_attrs=extra_attrs, **kw) as w:
        for i, (a, n, t) in enumerate(zip(evs, nfo, tfo)):
            w.append_event(i, a, n, t)
    return path
