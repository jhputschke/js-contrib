"""
python/jetscape/records.py

Record dtypes for the small per-event tables of the Pythonic layer
(docs/Plans/PLAN_pyjetscape_pythonic.md, step 1), and the conversions from the
flat float64 arrays the bindings return.  numpy only: works without the compiled
extension, e.g. on arrays read back from files.

One convention everywhere: momentum (E, px, py, pz), position (t, x, y, z), the
order of FourVector.comp(i).  The old arrays keep their own column orders; these
functions only reorder and retype, bit for bit (ids and statuses are integers
stored as float64, so the int32 cast is exact; colour tags are C++ unsigned int, hence
uint32: files written before the tag fix hold tags above 2**31).

    FINAL_PARTON_DTYPE  final_partons_numpy (FINAL_PARTON_COLUMNS, same order)
    INITIATOR_DTYPE     showers.initiators_from_manager (INITIATOR_COLUMNS)
    EDGE_DTYPE          PartonShower.to_numpy (src, tgt, pid, pstat, px py pz E, x y z t)
    NODE_DTYPE          PartonShower.vertices_to_numpy (id, x, y, z, t)
    HARD_PARTON_DTYPE   hard_partons_numpy (pid, px, py, pz, E, pT, y)

Hadrons are not records: they stay one array per field (HadronArray), so iSS's
arrays are wrapped without a copy.
"""

from __future__ import annotations

import numpy as np

__all__ = ["FINAL_PARTON_DTYPE", "INITIATOR_DTYPE", "EDGE_DTYPE", "NODE_DTYPE",
           "HARD_PARTON_DTYPE", "final_parton_records", "final_parton_rows",
           "initiator_records", "initiator_rows", "edge_records", "node_records",
           "hard_parton_records"]

_MOMENTUM = [("E", "f8"), ("px", "f8"), ("py", "f8"), ("pz", "f8")]
_POSITION = [("t", "f8"), ("x", "f8"), ("y", "f8"), ("z", "f8")]

FINAL_PARTON_DTYPE = np.dtype([("shower", "i4"), ("pid", "i4"), ("pstat", "i4"),
                               *_MOMENTUM, *_POSITION, ("mass", "f8"),
                               ("col", "u4"), ("acol", "u4")])
INITIATOR_DTYPE = np.dtype([("shower", "i4"), ("pid", "i4"), ("pstat", "i4"),
                            *_MOMENTUM, *_POSITION])
EDGE_DTYPE = np.dtype([("src", "i4"), ("tgt", "i4"), ("pid", "i4"), ("pstat", "i4"),
                       *_MOMENTUM, *_POSITION])
NODE_DTYPE = np.dtype([("id", "i4"), *_POSITION])
HARD_PARTON_DTYPE = np.dtype([("pid", "i4"), *_MOMENTUM, ("pt", "f8"), ("y", "f8")])

# record field <- column of the flat array
_FINAL_PARTON_SRC = {"shower": 0, "pid": 1, "pstat": 2, "E": 3, "px": 4, "py": 5, "pz": 6,
                     "t": 7, "x": 8, "y": 9, "z": 10, "mass": 11, "col": 12, "acol": 13}
_INITIATOR_SRC = {"shower": 0, "pid": 1, "pstat": 2, "px": 3, "py": 4, "pz": 5, "E": 6,
                  "x": 7, "y": 8, "z": 9, "t": 10}
_EDGE_SRC = {"src": 0, "tgt": 1, "pid": 2, "pstat": 3, "px": 4, "py": 5, "pz": 6, "E": 7,
             "x": 8, "y": 9, "z": 10, "t": 11}
_NODE_SRC = {"id": 0, "x": 1, "y": 2, "z": 3, "t": 4}
_HARD_PARTON_SRC = {"pid": 0, "px": 1, "py": 2, "pz": 3, "E": 4, "pt": 5, "y": 6}


def _records(rows, dtype, src):
    rows = np.asarray(rows, dtype=np.float64)
    ncol = max(src.values()) + 1
    rows = rows.reshape(-1, ncol) if rows.size == 0 else rows
    if rows.ndim != 2 or rows.shape[1] < ncol:
        raise ValueError(f"need an (N, {ncol}) array, got shape {rows.shape}")
    out = np.empty(len(rows), dtype=dtype)
    for name in dtype.names:
        out[name] = rows[:, src[name]]
    return out


def final_parton_records(rows) -> np.ndarray:
    """(N, 14) FINAL_PARTON_COLUMNS -> FINAL_PARTON_DTYPE records (same field order)."""
    return _records(rows, FINAL_PARTON_DTYPE, _FINAL_PARTON_SRC)


def final_parton_rows(partons) -> np.ndarray:
    """FINAL_PARTON_DTYPE records (or an (N, 14) array, passed through) -> (N, 14) float64
    in FINAL_PARTON_COLUMNS order: what hadronize_partons and the files take."""
    a = np.asarray(partons)
    if a.dtype.names is None:
        return np.ascontiguousarray(a, dtype=np.float64)
    missing = [n for n in _FINAL_PARTON_SRC if n not in a.dtype.names]
    if missing:
        raise ValueError(f"final parton records lack the fields {missing}")
    out = np.empty((len(a), len(_FINAL_PARTON_SRC)), dtype=np.float64)
    for name, col in _FINAL_PARTON_SRC.items():
        out[:, col] = a[name]
    return out


def initiator_records(rows) -> np.ndarray:
    """(K, 11) INITIATOR_COLUMNS -> INITIATOR_DTYPE records."""
    return _records(rows, INITIATOR_DTYPE, _INITIATOR_SRC)


def initiator_rows(initiators) -> np.ndarray:
    """INITIATOR_DTYPE records (or a (K, 11) array, passed through) -> (K, 11) float64 in
    INITIATOR_COLUMNS order: what HadronH5Writer.append_unit(initiators=...) takes."""
    a = np.asarray(initiators)
    if a.dtype.names is None:
        return np.ascontiguousarray(a, dtype=np.float64).reshape(-1, len(_INITIATOR_SRC))
    out = np.empty((len(a), len(_INITIATOR_SRC)), dtype=np.float64)
    for name, col in _INITIATOR_SRC.items():
        out[:, col] = a[name]
    return out


def edge_records(rows) -> np.ndarray:
    """PartonShower.to_numpy's (n, 12) -> EDGE_DTYPE records."""
    return _records(rows, EDGE_DTYPE, _EDGE_SRC)


def node_records(rows) -> np.ndarray:
    """PartonShower.vertices_to_numpy's (n, 5) -> NODE_DTYPE records."""
    return _records(rows, NODE_DTYPE, _NODE_SRC)


def hard_parton_records(rows) -> np.ndarray:
    """hard_partons_numpy's (n, 7) -> HARD_PARTON_DTYPE records (hardest first, as given)."""
    return _records(rows, HARD_PARTON_DTYPE, _HARD_PARTON_SRC)
