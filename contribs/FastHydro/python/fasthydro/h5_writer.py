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

The `shower/` group is added the same way and for the same reason: `source/droplets` records
what the jet *lost*, and on its own a file cannot say where the jet was or what survived.  See
`fasthydro/showers.py` for the layout.  It is ragged, accumulated here and flushed at close,
because `FnoH5Writer` is vendored verbatim and knows only about droplets.
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

    def __init__(self, path, cfg, nevents, *, compression=None, source_compression=None,
                 keep_bits=None, overwrite=None):
        import h5py

        from fast_data.eos import resolve_eos
        from fast_data.writer import FnoH5Writer, grid_attrs

        from .grid import GridSpec

        self.path = str(path)
        self.cfg = cfg
        self.g = g = GridSpec.from_cfg(cfg)
        self.nevents = int(nevents)

        self._np_eos = None
        if cfg["eos"].get("store_table", True):
            try:
                self._np_eos = resolve_eos(cfg["eos"])[0]
            except Exception as exc:                      # a missing table must not lose a run
                print(f"[PairedH5Writer] EoS table not stored ({exc}); "
                      f"sound_speed()/mach_angle() will be unavailable", flush=True)

        attrs = grid_attrs(g.nx, g.ny, g.neta, g.dx, g.dy, g.deta,
                           tau_min=g.tau0, dtau=g.record_dtau, choose_ntau=g.ntau)

        out = cfg["output"]
        self._w = FnoH5Writer(
            self.path, attrs, self.nevents,
            compression=compression or out["compression"],
            source_compression=source_compression or out["source_compression"],
            # the same rounding for both legs, so arr - arr_bg stays exact where they agree
            keep_bits=keep_bits if keep_bits is not None else out.get("keep_bits"),
            write_source=True, write_diagnostics=out["write_diagnostics"],
            # FnoH5Writer refuses to clobber an existing file unless told to. Honour
            # run.overwrite so a second run does not stop on a name it chose itself.
            force=bool(cfg["run"].get("overwrite", False) if overwrite is None else overwrite),
            # Without the EoS group, viz's sound_speed() -- and so mach_angle() -- returns
            # None for anything but a conformal EoS, and glauber.load_eos() cannot rebuild
            # the table. write_eos_group makes the file self-contained.
            np_eos=self._np_eos,
            # FnoH5Writer copies only SCALAR_KEYS out of `attrs`; anything else has to go
            # through extra_attrs or it is silently dropped.
            extra_attrs={**dict(out["extra_attrs"] or {}), **self._provenance()})
        self._h5py = h5py
        self._bg = None                      # created lazily, on the first event
        self._closed = False
        # shower/ is ragged like source/droplets: accumulate, write once at close.
        self._sh = {"partons": [], "vertices": [], "initiators": []}
        self._sh_off = {"partons": [0], "vertices": [0], "initiators": [0]}
        self._sh_any = False

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
            # viz.sound_speed() falls back on this when there is no table
            "eos_kind": str(self.cfg["eos"]["kind"]),
            "transport_mode": str(self.cfg["transport"]["mode"]),
            "eta_over_s": float(self.cfg["transport"]["eta_over_s"]),
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
        from fast_data.h5_compression import tag_dataset

        f = self._w.f
        # arr's exact filter (an explicit `compression=` argument included), not the config's
        kw, label = self._w.arr_filter_kwargs, self._w.arr_filter_label
        self._bg = f.create_dataset(
            "arr_bg", shape=(self.nevents,) + tuple(arr_bg.shape), dtype="f4",
            chunks=(1,) + tuple(arr_bg.shape),
            maxshape=(self.nevents,) + arr_bg.shape[:-1] + (None,),   # tau extendible, as arr
            **kw)
        tag_dataset(self._bg, label, self._w.keep_bits)
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

        # viz's source_track()/blob_radius() read these off the file and otherwise fall back
        # to hardcoded defaults (tau_delay 1.0, c_diff 0.894), which would put the jet's track
        # and the deposition blob in the wrong place for any run not using them. fast_data's
        # own driver never writes them; write them here, from the live C++ liquefier.
        if bridge is not None and getattr(bridge, "params", None) is not None:
            a = self._w.f.attrs
            if "liquefier_tau_delay" not in a:
                q = bridge.params
                a["liquefier_dtau"] = float(q.dtau)
                a["liquefier_tau_delay"] = float(q.tau_delay)
                a["liquefier_time_relax"] = float(q.time_relax)
                a["liquefier_d_diff"] = float(q.d_diff)
                a["liquefier_width_delta"] = float(q.width_delta)
                a["liquefier_c_diff"] = float(q.c_diff)
                a["liquefier_gamma_relax"] = float(q.gamma_relax)

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
        d.update(self._stash_shower(getattr(bridge, "shower", None)))

        self._w.append_event(
            i, hyd_jet.arr, d.get("ntau_freezeout", self.g.ntau),
            d.get("tau_freezeout", float("nan")),
            S_ev=hyd_jet.src, P_cart=d.get("P_cart"),
            droplets=droplets, diag=_scalars(d))

        if hyd_bg.arr is not None:
            from fast_data.h5_compression import round_mantissa

            self._ensure_bg(hyd_bg.arr)
            self._bg[i] = round_mantissa(hyd_bg.arr, self._w.keep_bits)
            bd = hyd_bg.diag or {}
            self._w.f["ntau_freezeout_bg"][i] = int(bd.get("ntau_freezeout", self.g.ntau))
            self._w.f["tau_freezeout_bg"][i] = float(bd.get("tau_freezeout", float("nan")))
            self._w.f.flush()

    # ------------------------------------------------------------------ shower/
    def _stash_shower(self, rec):
        """Queue one event's shower graph; -> the diagnostics it contributes.

        Every event appends an offset even when there is no shower, so `parton_offsets` stays
        one entry longer than the event count and `offsets[i]:offsets[i+1]` is always the
        right slice -- an event with no shower is an empty one, not a missing one.
        """
        from .showers import PARTON_COLUMNS, VERTEX_COLUMNS, INITIATOR_COLUMNS

        widths = {"partons": len(PARTON_COLUMNS), "vertices": len(VERTEX_COLUMNS),
                  "initiators": len(INITIATOR_COLUMNS)}
        for name, w in widths.items():
            a = (np.zeros((0, w)) if rec is None
                 else np.asarray(getattr(rec, name), dtype=np.float64).reshape(-1, w))
            self._sh[name].append(a)
            self._sh_off[name].append(self._sh_off[name][-1] + len(a))
        if rec is None:
            return {}
        self._sh_any = True
        return {"n_showers": int(rec.n_showers), "n_partons": int(len(rec.partons))}

    def _write_shower(self):
        from .showers import FATES, INITIATOR_COLUMNS, PARTON_COLUMNS, VERTEX_COLUMNS

        if not self._sh_any:
            return
        g = self._w.f.require_group("shower")
        spec = {"partons": (len(PARTON_COLUMNS), "parton_offsets"),
                "vertices": (len(VERTEX_COLUMNS), "vertex_offsets"),
                "initiators": (len(INITIATOR_COLUMNS), "initiator_offsets")}
        for name, (w, off_name) in spec.items():
            rows = self._sh[name]
            data = np.concatenate(rows, 0) if rows else np.zeros((0, w))
            for key, val in ((name, data),
                             (off_name, np.asarray(self._sh_off[name], dtype=np.int64))):
                if key in g:
                    del g[key]
                g.create_dataset(key, data=val)
        g.attrs["parton_columns"] = list(PARTON_COLUMNS)
        g.attrs["vertex_columns"] = list(VERTEX_COLUMNS)
        g.attrs["initiator_columns"] = list(INITIATOR_COLUMNS)
        g.attrs["units"] = "p in GeV; x, y, z in fm; t in fm/c"
        g.attrs["coordinates"] = (
            "CARTESIAN LAB (x, y, z) and lab time t -- NOT the Milne (tau, x, y, eta) the "
            "hydro frames and source/droplets use. Convert with fasthydro.showers.to_milne: "
            "tau = sqrt(t^2 - z^2), eta = atanh(z/t).")
        g.attrs["vertex_positions"] = (
            "ZERO BY CONSTRUCTION. X-SCAPE builds every vertex as Vertex(0,0,0,currentTime) "
            "(JetEnergyLoss.cc:414-419), so only the `t` column is real. The geometry is on "
            "the partons, which carry their own production point; use "
            "fasthydro.showers.segments, which propagates each parton along p/E rather than "
            "reading vertex positions.")
        g.attrs["endpoint_convention"] = (
            "i_src and i_tgt are ROW INDICES into this event's slice of `vertices` "
            "(vertex_offsets[i] : vertex_offsets[i+1]), already offset across the event's "
            "showers. They are not raw GTL node ids, which restart at 0 per shower; the raw "
            "id is kept as the vertex `node_id` column.")
        g.attrs["pstat_codes"] = [f"{k}: {v}" for k, v in sorted(FATES.items())]
        self._w.f.attrs["has_shower"] = True

    def close(self, complete=None):
        if not self._closed:
            self._write_shower()
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
