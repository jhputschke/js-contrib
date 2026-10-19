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
