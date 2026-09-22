"""Write the paired run in FNO4d's HDF5 schema -- the format the training pipeline reads.

The `.npz` files the example drivers can also emit are a convenience for poking at one event;
**this** is the dataset format.  It is `fast_data.writer.FnoH5Writer` used unmodified, so a
file written here is byte-compatible with one written by fast_data's own `generate.py` and
loads in every existing FNO4d reader without a special case.

Which leg is `arr`
------------------
fast_data's convention (writer.SOURCE_CONVENTION) is that

    arr[...,t] ALREADY CONTAINS S[...,t] -- arr is the single evolution with the source in it.

So `arr` is the **jet** leg and `source/S` is what was injected into it.  That is what an FNO
trained on deposition wants, and it is the same convention as fast_data's own `*_jet.yaml`
datasets, so the two are directly interchangeable.

The background leg has no place in that schema, so it goes in alongside as `arr_bg`, with its
own freeze-out bookkeeping.  Readers that do not know about it ignore it; readers that do get
an exactly-paired no-jet reference on an identical initial condition, which is the thing this
whole contribution exists to produce.
"""

from __future__ import annotations

import numpy as np

__all__ = ["PairedH5Writer"]

#: datasets this adds on top of the fast_data schema
EXTRA = ("arr_bg", "ntau_freezeout_bg", "tau_freezeout_bg")


class PairedH5Writer:
    """Streaming writer for (background, jet) pairs. One event at a time, like FnoH5Writer.

        with PairedH5Writer("out.h5", cfg, nevents) as w:
            w.append(i, hyd_bg, hyd_jet, bridge)
    """

    def __init__(self, path, cfg, nevents, *, compression=None, source_compression=None):
        import h5py

        from fast_data.writer import FnoH5Writer, grid_attrs

        from .grid import GridSpec

        self.path = str(path)
        self.cfg = cfg
        self.g = g = GridSpec.from_cfg(cfg)
        self.nevents = int(nevents)

        attrs = grid_attrs(g.nx, g.ny, g.neta, g.dx, g.dy, g.deta,
                           tau_min=g.tau0, dtau=g.record_dtau, choose_ntau=g.ntau)

        out = cfg["output"]
        self._w = FnoH5Writer(
            self.path, attrs, self.nevents,
            compression=compression or out["compression"],
            source_compression=source_compression or out["source_compression"],
            write_source=True, write_diagnostics=out["write_diagnostics"],
            # FnoH5Writer copies only SCALAR_KEYS out of `attrs`; anything else has to go
            # through extra_attrs or it is silently dropped.
            extra_attrs={**dict(out["extra_attrs"] or {}), **self._provenance()})
        self._h5py = h5py
        self._bg = None                      # created lazily, on the first event
        self._closed = False

    # ------------------------------------------------------------------ provenance
    def _provenance(self):
        import json

        cfgj = {k: v for k, v in self.cfg.items() if k != "fasthydro"}
        return {
            "generator": "fasthydro",
            "producer": "js-contrib/contribs/FastHydro",
            "pairing": "bg_jet",
            "arr_is": "jet leg (source included, per fast_data SOURCE_CONVENTION)",
            "arr_bg_is": "background leg, identical initial condition, no source",
            "source_model": "causal_liquefier (droplets from X-SCAPE Matter+LBT)",
            "source_mode": str(self.cfg["source"]["mode"]),
            "hard_vertex": str((self.cfg.get("fasthydro") or {})
                               .get("hard_vertex", {}).get("mode", "?")),
            "config_json": json.dumps(cfgj, sort_keys=True, default=str),
            "fasthydro_json": json.dumps(self.cfg.get("fasthydro", {}), sort_keys=True,
                                         default=str),
        }

    # ------------------------------------------------------------------ writing
    def _ensure_bg(self, arr_bg):
        if self._bg is not None:
            return
        f = self._w.f
        out = self.cfg["output"]
        self._bg = f.create_dataset(
            "arr_bg", shape=(self.nevents,) + tuple(arr_bg.shape), dtype="f4",
            chunks=(1,) + tuple(arr_bg.shape),
            maxshape=(self.nevents,) + arr_bg.shape[:-1] + (None,),   # tau extendible, as arr
            compression=out["compression"])
        f.create_dataset("ntau_freezeout_bg", shape=(self.nevents,), dtype="i4")
        f.create_dataset("tau_freezeout_bg", shape=(self.nevents,), dtype="f4")

    def append(self, i, hyd_bg, hyd_jet, bridge=None):
        """Write one event. `hyd_*` are FastHydro instances that have run."""
        if self._closed:
            raise RuntimeError("writer is closed")
        if hyd_jet.arr is None:
            raise ValueError("the jet leg has no arr; construct FastHydro with keep_arr=True")

        # The pair is only worth anything if both legs evolved the SAME initial condition:
        # arr - arr_bg is read as the jet's effect, so a different IC underneath would look
        # like an enormous wake. They share an `ic` object by construction, but construction
        # is exactly what a future edit changes, so check rather than assume.
        if (hyd_bg.ic_sha256 and hyd_jet.ic_sha256
                and hyd_bg.ic_sha256 != hyd_jet.ic_sha256):
            raise ValueError(
                f"event {i}: the two legs ran different initial conditions "
                f"(background {hyd_bg.ic_sha256[:12]}, jet {hyd_jet.ic_sha256[:12]}). "
                f"arr - arr_bg would not be the jet's effect. Both FastHydro instances must "
                f"be given the same ic= object.")

        d = dict(hyd_jet.diag or {})
        droplets = None
        if bridge is not None and bridge.droplets is not None and len(bridge.droplets):
            droplets = bridge.droplets.data
            d["n_droplets"] = int(len(bridge.droplets))
            d["E_droplets"] = float(bridge.droplets.data[:, 4].sum())
            for k in ("n_late", "n_early", "E_in_window"):
                if hasattr(bridge, k):
                    d[k] = getattr(bridge, k)
        d["ic_sha256"] = hyd_jet.ic_sha256 or ""          # identical for both legs; asserted above

        self._w.append_event(
            i, hyd_jet.arr, d.get("ntau_freezeout", self.g.ntau),
            d.get("tau_freezeout", float("nan")),
            S_ev=hyd_jet.src, P_cart=d.get("P_cart"),
            droplets=droplets, diag=_scalars(d))

        if hyd_bg.arr is not None:
            self._ensure_bg(hyd_bg.arr)
            self._bg[i] = np.asarray(hyd_bg.arr, dtype=np.float32)
            bd = hyd_bg.diag or {}
            self._w.f["ntau_freezeout_bg"][i] = int(bd.get("ntau_freezeout", self.g.ntau))
            self._w.f["tau_freezeout_bg"][i] = float(bd.get("tau_freezeout", float("nan")))
            self._w.f.flush()

    def close(self, complete=None):
        if not self._closed:
            self._w.close(complete=complete)
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close(complete=exc[0] is None)


def _scalars(d):
    """The diagnostics FnoH5Writer can store: one number (or short string) per event."""
    out = {}
    for k, v in d.items():
        if k == "P_cart":                       # already written as its own dataset
            continue
        if isinstance(v, (bool, int, float, np.integer, np.floating)):
            out[k] = v
        elif isinstance(v, str):
            out[k] = v
        elif isinstance(v, np.ndarray) and v.size == 1:
            out[k] = v.reshape(-1)[0]
    return out
