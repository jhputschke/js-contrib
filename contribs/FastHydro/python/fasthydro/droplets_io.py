"""Droplet dump/load for the replay path.

Deliberately free of any `jetscape` import: replay drives fast_data directly and must work on
a machine with no X-SCAPE build.  `DropletBridge` cannot live here because it subclasses a
bound C++ type at import time.
"""

from __future__ import annotations

import numpy as np

__all__ = ["save_droplets_npz", "load_droplets_npz", "showers_from_meta"]

#: keys `_flatten_showers` writes, so `load_droplets_npz` can keep them out of `meta`
_SHOWER_KEYS = ("shower_partons", "shower_vertices", "shower_initiators",
                "shower_parton_offsets", "shower_vertex_offsets",
                "shower_initiator_offsets")


def _flatten_showers(history):
    """Per-event `ShowerRecord`s -> flat blocks + offsets, or empty blocks if there are none.

    Kept here rather than in `showers.py` so the npz side needs nothing but numpy; the widths
    are read from `showers` only when there is something to write.
    """
    if not history:
        return {}
    from .showers import INITIATOR_COLUMNS, PARTON_COLUMNS, VERTEX_COLUMNS

    out = {}
    for name, cols, off in (("partons", PARTON_COLUMNS, "parton_offsets"),
                            ("vertices", VERTEX_COLUMNS, "vertex_offsets"),
                            ("initiators", INITIATOR_COLUMNS, "initiator_offsets")):
        blocks = [np.asarray(getattr(r, name), dtype=np.float64).reshape(-1, len(cols))
                  for r in history]
        out["shower_" + name] = (np.concatenate(blocks, 0) if blocks
                                 else np.zeros((0, len(cols))))
        out["shower_" + off] = np.cumsum([0] + [len(b) for b in blocks]).astype(np.int64)
    return out


def showers_from_meta(meta, event=0):
    """-> `ShowerRecord` for one event out of `load_droplets_npz`'s `meta`, or None."""
    from .showers import ShowerRecord

    if "shower_partons" not in meta:
        return None
    parts = []
    for name, off in (("partons", "parton_offsets"), ("vertices", "vertex_offsets"),
                      ("initiators", "initiator_offsets")):
        o = meta["shower_" + off]
        parts.append(meta["shower_" + name][o[event]:o[event + 1]])
    return ShowerRecord(*parts)



def save_droplets_npz(path, bridge, *, cfg_sha256="", ic_sha256=(), seeds=(),
                      arr_sha256=(), ic=None, extra=None):
    """Everything a replay run needs, numpy only -- no h5py, no X-SCAPE.

    `ic` is the (nx, ny, neta) float64 initial energy density.  Storing it is what makes
    replay reproduce the original bit for bit: regenerating the Glauber event from its seed
    would have to reproduce every RNG draw exactly, across numpy versions.  It costs
    ~1 MB at 65x65x33, which is nothing beside the evolution it replaces.
    """
    hist = bridge.history
    data = (np.concatenate([d.data for d in hist], axis=0)
            if hist else np.zeros((0, 8), np.float64))
    offsets = np.cumsum([0] + [len(d) for d in hist]).astype(np.int64)
    flags = (np.concatenate([d.flags for d in hist], axis=0)
             if hist else np.zeros((0,), np.uint32))
    p = bridge.params
    g = bridge.hydro_jet.g

    # The shower graph travels with the droplets so a replayed file can still be animated.
    # Stored as three flat blocks plus offsets, same shape as the h5 `shower/` group; absent
    # (zero-row) when the run did not capture it.
    sh = _flatten_showers(getattr(bridge, "shower_history", None))

    payload = dict(
        droplets=data, offsets=offsets, flags=flags, **sh,
        columns=np.array(["tau", "x", "y", "eta", "E", "px", "py", "pz"]),
        params=np.array([p.dtau, p.tau_delay, p.time_relax, p.d_diff, p.width_delta]),
        grid=np.array([g.nx, g.ny, g.neta, g.dx, g.dy, g.deta], dtype=np.float64),
        tau_grid=g.tau_grid(),
        cfg_sha256=np.array(cfg_sha256), ic_sha256=np.array(list(ic_sha256)),
        seeds=np.array(list(seeds), dtype=np.int64),
        arr_sha256=np.array(list(arr_sha256)),
        ic=(np.ascontiguousarray(ic, dtype=np.float64) if ic is not None
            else np.zeros((0, 0, 0), np.float64)),
    )
    payload.update(extra or {})
    np.savez(str(path), **payload)
    return str(path)


def load_droplets_npz(path):
    """-> (list[DropletArray] per event, LiquefierParams, meta dict)."""
    from fast_data.liquefier import LiquefierParams
    from fast_data.liquefier.droplets import DropletArray

    z = np.load(str(path), allow_pickle=False)
    cols = [str(c) for c in z["columns"]]
    want = ["tau", "x", "y", "eta", "E", "px", "py", "pz"]
    if cols != want:
        raise ValueError(f"{path}: droplet columns {cols} != {want}")
    data, offs, flags = z["droplets"], z["offsets"], z["flags"]
    per_event = [
        DropletArray(data[offs[i]:offs[i + 1]],
                     np.array([0, offs[i + 1] - offs[i]], dtype=np.int64),
                     flags=flags[offs[i]:offs[i + 1]])
        for i in range(len(offs) - 1)
    ]
    pr = z["params"]
    params = LiquefierParams(dtau=float(pr[0]), tau_delay=float(pr[1]),
                             time_relax=float(pr[2]), d_diff=float(pr[3]),
                             width_delta=float(pr[4]))
    meta = {k: z[k] for k in z.files if k not in ("droplets", "offsets", "flags",
                                                  "columns", "params")}
    return per_event, params, meta
