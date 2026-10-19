"""The bookkeeping records of the map steps' outputs, written next to their data so that every
output stands alone (``weights.file_record``, ``weights.campaign_ids``): ``file_info`` (one
entry: the production file's name, events, hydro backgrounds and pTHat windows) and
``run_info`` (one entry: the map step's settings, e.g. the jet radius).  As ROOT trees
(uproot here, TTree branches of the same names in analysis_root/common/AnaCommon.h) or as
``fi_*`` / ``ri_*`` arrays of an ``.npz`` (the Python pools)."""
import numpy as np

#: the per-window columns of a file record
WINDOW_KEYS = ("pthat_lo", "pthat_hi", "sigma_gen", "sigma_err", "n_accepted", "n_stored")


def file_info_tree(rec):
    """A file record -> the dict uproot writes as the one-entry tree ``file_info``: name,
    n_events, vacuum, the backgrounds ``units`` and the windows ``win_<key>`` (jagged, one
    row)."""
    import awkward as ak
    out = {"name": np.array([rec["name"]]),
           "n_events": np.array([rec["n_events"]], np.int64),
           "vacuum": np.array([int(rec["vacuum"])], np.int32),
           "units": ak.Array([np.asarray(rec["units"], np.int64)])}
    out["win"] = ak.zip({k: ak.Array([np.asarray(rec[k], np.int64 if k == "n_stored"
                                                 else np.float64)]) for k in WINDOW_KEYS})
    return out


def _flat(arrays):
    """A tree's arrays with records flattened (win -> win_pthat_lo, ...): uproot's RNTuple
    record, or the C++ TTree's win_pthat_lo branches as they are."""
    out = {}
    for k in arrays.fields:
        x = arrays[k]
        if x.fields:
            for sub in x.fields:
                out[f"{k}_{sub}"] = x[sub]
        else:
            out[k] = x
    return out


def read_file_info(f):
    """The file record of an open uproot file (its ``file_info`` tree) -> dict as
    ``weights.file_record``; None if the file has none (made before 2026-10-19)."""
    import awkward as ak
    if "file_info" not in f:
        return None
    t = _flat(f["file_info"].arrays())
    name = t["name"][0]
    rec = {"name": name if isinstance(name, str) else bytes(name).decode(),
           "n_events": int(t["n_events"][0]), "vacuum": bool(t["vacuum"][0]),
           "units": np.asarray(ak.to_numpy(t["units"][0]), np.int64)}
    for k in WINDOW_KEYS:
        rec[k] = np.asarray(ak.to_numpy(t[f"win_{k}"][0]),
                            np.int64 if k == "n_stored" else np.float64)
    return rec


def run_info_tree(info):
    """A dict of scalar settings -> the dict uproot writes as the one-entry tree
    ``run_info``."""
    return {k: np.array([v], np.int64 if isinstance(v, (int, np.integer)) and
                        not isinstance(v, bool) else np.float64) for k, v in info.items()}


def read_run_info(f):
    """The settings of an open uproot file (its ``run_info`` tree) -> dict; {} without."""
    import awkward as ak
    if "run_info" not in f:
        return {}
    t = f["run_info"].arrays()
    out = {}
    for k in t.fields:
        v = ak.to_numpy(t[k])
        out[k] = int(v[0]) if np.issubdtype(v.dtype, np.integer) else float(v[0])
    return out


def file_info_npz(rec):
    """A file record -> ``fi_*`` arrays for an .npz."""
    out = {"fi_name": np.array(rec["name"]), "fi_n_events": np.array(rec["n_events"], np.int64),
           "fi_vacuum": np.array(int(rec["vacuum"]), np.int32),
           "fi_units": np.asarray(rec["units"], np.int64)}
    for k in WINDOW_KEYS:
        out[f"fi_{k}"] = np.asarray(rec[k], np.int64 if k == "n_stored" else np.float64)
    return out


def read_file_info_npz(d):
    """The file record of an .npz's ``fi_*`` arrays (a dict); None without."""
    if "fi_name" not in d:
        return None
    rec = {"name": str(d["fi_name"]), "n_events": int(d["fi_n_events"]),
           "vacuum": bool(d["fi_vacuum"]), "units": np.asarray(d["fi_units"], np.int64)}
    for k in WINDOW_KEYS:
        rec[k] = np.asarray(d[f"fi_{k}"])
    return rec
