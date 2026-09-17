"""
python/jetscape/fast_root_bulk.py

Reader for ROOT files written by the X-SCAPE C++ FastRootBulkWriter.

The writer itself is a C++ module (bound as ``jetscape.FastRootBulkWriter``
when PyJetscape is built against a ROOT-enabled X-SCAPE, see ``HAS_ROOT``).
This module only reads its output, so it needs uproot but not ROOT.

File layout written by FastRootBulkWriter
-----------------------------------------
TTree ``t``, one entry per event:
    user_res        : vector<float>, flat [tau][x][y][eta][feature]
                      features = [energy_density, vx, vy, vz]
    ntau_freezeout  : int   — number of tau steps in user_res (varies per event)
    tau_freezeout   : float — tau one step past the last stored MUSIC step
File-level TParameters (from the first event):
    nFeatures, nx, ny, neta, tau_min, dtau, tau_stride, use_vec,
    x_min, dx, y_min, dy, eta_min, deta, ntau        (user grid, grid mode)
    nX_MUSIC, X_min_MUSIC, dX_MUSIC, ... tau_min_MUSIC, dtau_MUSIC (MUSIC grid)
TNamed ``grid_mode``: "native" or "grid".

Usage
-----
    from jetscape.fast_root_bulk import read_fast_root_bulk

    d = read_fast_root_bulk("hydro_evo_fast.root", entry_stop=1)
    evo = d["events"][0]              # (ntau, nx, ny, neta, 4) float32
    x   = d["grid"]["x"]              # cell coordinates [fm]
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import uproot
    _UPROOT_AVAILABLE = True
except ImportError:
    _UPROOT_AVAILABLE = False


FEATURES = ("energy_density", "vx", "vy", "vz")

_INT_PARAMS = ("nFeatures", "nx", "ny", "neta", "ntau", "tau_stride", "use_vec",
               "nX_MUSIC", "nY_MUSIC", "neta_MUSIC")
_FLOAT_PARAMS = ("x_min", "dx", "y_min", "dy", "eta_min", "deta",
                 "tau_min", "dtau",
                 "X_min_MUSIC", "dX_MUSIC", "Y_min_MUSIC", "dY_MUSIC",
                 "eta_min_MUSIC", "deta_MUSIC", "tau_min_MUSIC", "dtau_MUSIC")


def read_fast_root_bulk(
    path: str,
    entry_start: Optional[int] = None,
    entry_stop: Optional[int] = None,
) -> dict:
    """
    Load a ROOT file written by FastRootBulkWriter.

    Native-mode events are large (the full MUSIC grid), so use
    ``entry_start`` / ``entry_stop`` to load only some events.

    Parameters
    ----------
    path : str
        Path to the .root file.
    entry_start, entry_stop : int, optional
        Event range to load (uproot semantics; default: all events).

    Returns
    -------
    dict with keys:
        "events"        — list of np.ndarray, each (ntau, nx, ny, neta, 4)
                          float32; ntau differs between events
        "ntau"          — np.ndarray of ntau per loaded event
        "tau_freezeout" — np.ndarray of tau_freezeout per loaded event
                          (None for files written before that branch existed)
        "grid_mode"     — "native" or "grid"
        "features"      — ("energy_density", "vx", "vy", "vz")
        "grid"          — dict: nx, ny, neta, x_min, dx, y_min, dy, eta_min,
                          deta, tau_min, dtau and coordinate arrays x, y, eta
                          (MUSIC grid in native mode, user grid in grid mode;
                          None if the file predates the _MUSIC parameters)
        "params"        — dict of all TParameters stored in the file

    Requires uproot.
    """
    if not _UPROOT_AVAILABLE:
        raise ImportError("uproot is required to read ROOT files.  "
                          "Install with: pip install uproot")

    with uproot.open(path) as f:
        params = {}
        for key in _INT_PARAMS + _FLOAT_PARAMS:
            if key in f:
                val = f[key].member("fVal")
                params[key] = int(val) if key in _INT_PARAMS else float(val)
        grid_mode = f["grid_mode"].member("fTitle") if "grid_mode" in f else "native"

        tree = f["t"]
        kw = dict(entry_start=entry_start, entry_stop=entry_stop, library="np")
        raw   = tree["user_res"].array(**kw)
        ntau  = tree["ntau_freezeout"].array(**kw)
        # not written by FastRootBulkWriter versions before the tau_freezeout branch
        tau_f = (tree["tau_freezeout"].array(**kw)
                 if "tau_freezeout" in tree else None)

    nx, ny, neta = params["nx"], params["ny"], params["neta"]
    nf = params.get("nFeatures", len(FEATURES))
    events = [np.asarray(r, dtype=np.float32).reshape(int(nt), nx, ny, neta, nf)
              for r, nt in zip(raw, ntau)]

    # Native mode writes MUSIC's grid; its x_min/dx/... parameters are the
    # (unused) user-grid values, so take the coordinates from the _MUSIC keys.
    # Files written before those keys existed get None coordinates.
    keys = ("x_min", "dx", "y_min", "dy", "eta_min", "deta")
    if grid_mode == "native":
        src = ("X_min_MUSIC", "dX_MUSIC", "Y_min_MUSIC", "dY_MUSIC",
               "eta_min_MUSIC", "deta_MUSIC")
    else:
        src = keys
    grid = {k: params.get(s) for k, s in zip(keys, src)}
    grid.update(nx=nx, ny=ny, neta=neta,
                tau_min=params["tau_min"], dtau=params["dtau"])
    for axis, n in (("x", nx), ("y", ny), ("eta", neta)):
        lo, step = grid[f"{axis}_min"], grid[f"d{axis}"]
        grid[axis] = None if lo is None else lo + step * np.arange(n)

    return {
        "events":        events,
        "ntau":          np.asarray(ntau),
        "tau_freezeout": None if tau_f is None else np.asarray(tau_f),
        "grid_mode":     grid_mode,
        "features":      FEATURES,
        "grid":          grid,
        "params":        params,
    }
