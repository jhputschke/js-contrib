"""Event weights of a windowed production: every event weighs sigma_k / N_k (its pTHat
window's cross section over the events in that window), times its N_coll in Pb+Pb (the geometry
is drawn with P(b) ~ b and every event gets one hard process, while its rate scales with
N_coll).  The event table is ``zjet_wake.event_table``'s: per event ``pthat_bin``,
``sigma_bin`` and ``n_coll``."""
import numpy as np


def event_weights(table, sel, ncoll=True):
    """Per-event weights sigma_k / N_k (x n_coll), normalized to mean 1 over ``sel``; 0
    outside it.  N_k counts all events of window k (the selection is physics, not
    sampling).  Without windows every event has sigma 1; without n_coll it is left out."""
    t = table
    w = np.ones(len(sel))
    if np.any(t["pthat_bin"] >= 0):
        nk = np.bincount(np.clip(t["pthat_bin"], 0, None), minlength=1).astype(float)
        w = t["sigma_bin"] / nk[np.clip(t["pthat_bin"], 0, None)]
    if ncoll and np.all(np.isfinite(t["n_coll"][sel])):
        w = w * t["n_coll"]
    w = np.where(sel, w, 0.0)
    if sel.any():
        w = w / w[sel].mean()
    return w


def effective_events(w, sel):
    """Kish's effective number of events of weights ``w`` over ``sel``."""
    x = w[sel]
    return float(x.sum() ** 2 / np.sum(x ** 2)) if len(x) else 0.0


def weight_fn(w):
    """weights(ev, info): every hadron of an event gets the event's weight ``w[event]``."""
    return lambda ev, info: np.full(len(ev), w[info.event])


# ── outputs that combine across campaigns: ids and weights at load time ───────────────
# Every map-step output (and pool) of one production file carries that file's bookkeeping
# (``file_record``): its name, its number of events, its hydro backgrounds and, per pTHat
# window, the Pythia cross section and accepted events and the events stored.  The ids and
# weights of a set of such outputs, from any number of campaigns, then follow from these
# records alone, by the rules of ``HadronFileReader`` (the global event index, the dense
# background id over (file, background), the window cross sections weighted by accepted events):
# a campaign is mapped once and combined with others at load time.

def file_record(reader, file_index):
    """The bookkeeping of one production file of a ``HadronFileReader`` -> dict: name (the
    file's stem), n_events (all its events), vacuum, units (its hydro backgrounds, sorted;
    vacuum: every event its own, the local events), and per pTHat window (empty without):
    pthat_lo, pthat_hi, sigma_gen [mb], sigma_err, n_accepted, n_stored (its events of the
    window, before any cut)."""
    import os
    f = reader._files[file_index]
    lo, hi = int(reader._offsets[file_index]), int(reader._offsets[file_index + 1])
    infos = [reader.event_info(g) for g in range(lo, hi)]
    vacuum = "bulk_jet" not in reader.tags(file_index)
    units = np.unique(np.array([i.local_event if vacuum else i.bg_unit for i in infos],
                               np.int64))
    rec = {"name": os.path.basename(f["stem"]), "n_events": hi - lo, "vacuum": vacuum,
           "units": units}
    bins = reader.pthat_bins
    if bins is None:
        nw = 0
        win = {k: np.zeros(0) for k in ("pthat_lo", "pthat_hi", "sigma_gen", "sigma_err",
                                        "n_accepted")}
    else:
        nw = len(bins)
        if f["sigma_gen"] is None or f["n_accepted"] is None:
            raise ValueError(f"{f['stem']}: no cross sections (job not finished?)")
        win = {"pthat_lo": np.array([b[0] for b in bins], float),
               "pthat_hi": np.array([b[1] for b in bins], float),
               "sigma_gen": np.asarray(f["sigma_gen"], float),
               "sigma_err": np.asarray(f["sigma_err"], float),
               "n_accepted": np.asarray(f["n_accepted"], float)}
    k = np.array([-1 if i.pthat_bin is None else i.pthat_bin for i in infos], np.int64)
    rec.update(win)
    rec["n_stored"] = np.bincount(k[k >= 0], minlength=nw).astype(np.int64)[:nw]
    return rec


def combine_windows(records):
    """The windows of a set of file records -> (sigma_k [mb], sigma_err_k, N_k): the files'
    cross sections weighted by their accepted events (``HadronFileReader.pthat_bin_sigma``,
    the same expression and order, so the same bits) and N_k their stored events.  Files with
    other windows than the first: ValueError.  No windows: three empty arrays."""
    records = sorted(records, key=lambda r: r["name"])          # the reader's file order
    if not records or not len(records[0]["pthat_lo"]):
        if any(len(r["pthat_lo"]) for r in records):
            raise ValueError("some files have pTHat windows, others none")
        return np.zeros(0), np.zeros(0), np.zeros(0, np.int64)
    r0 = records[0]
    for r in records:
        if not (np.array_equal(r["pthat_lo"], r0["pthat_lo"])
                and np.array_equal(r["pthat_hi"], r0["pthat_hi"])):
            raise ValueError(f"{r['name']} has other pTHat windows than {r0['name']}")
    nw = len(r0["pthat_lo"])
    sig, err, nk = np.zeros(nw), np.zeros(nw), np.zeros(nw, np.int64)
    for k in range(nw):
        s = np.asarray([r["sigma_gen"][k] for r in records], np.float64)
        e = np.asarray([r["sigma_err"][k] for r in records], np.float64)
        n = np.asarray([r["n_accepted"][k] for r in records], np.float64)
        if n.sum() == 0:
            sig[k] = err[k] = np.nan
        else:
            sig[k] = float(np.sum(n * s) / n.sum())
            err[k] = float(np.sqrt(np.sum((n * e) ** 2)) / n.sum())
        nk[k] = sum(int(r["n_stored"][k]) for r in records)
    return sig, err, nk


def window_weights(records):
    """weight_mb per window, sigma_k / N_k over a set of file records (NaN where N_k = 0)."""
    sig, _, nk = combine_windows(records)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(nk > 0, sig / np.maximum(nk, 1), np.nan)


def campaign_ids(records):
    """The global ids of a set of file records, ordered by file name (the reader's order for a
    campaign's directory) -> dict: order (record indices in that order), file (each record's
    file index), event_offset (each record's first global event), units (each record's
    {unit: dense background id}, the ids numbered over (file, unit) in sorted order as
    ``numpy.unique`` over the reader's events).  Duplicate file names: ValueError."""
    names = [r["name"] for r in records]
    if len(set(names)) != len(names):
        dup = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"file names occur twice: {dup[:3]} (one map output per production "
                         "file; campaigns need distinct names, TAG_PREFIX)")
    order = sorted(range(len(records)), key=lambda i: names[i])
    file_ = np.zeros(len(records), np.int64)
    ev_off = np.zeros(len(records), np.int64)
    units = [None] * len(records)
    n_ev = n_bg = 0
    for f, i in enumerate(order):
        file_[i], ev_off[i] = f, n_ev
        u = np.asarray(records[i]["units"], np.int64)
        units[i] = {int(x): n_bg + j for j, x in enumerate(u)}
        n_ev += int(records[i]["n_events"])
        n_bg += len(u)
    return {"order": order, "file": file_, "event_offset": ev_off, "units": units}
