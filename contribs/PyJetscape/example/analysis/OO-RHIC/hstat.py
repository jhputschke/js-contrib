"""
analysis/OO-RHIC/hstat.py -- statistical uncertainties for the OO-RHIC analyses and notebooks.

Every model event has one hydro and one iSS sample (no oversampling), so events are independent
and the statistical error follows from the events themselves:

  histograms of events          hist Weight storage: sum w and sum w^2 per bin (correct for the
                                N_coll / pTHat importance weights)            -> counts(), density()
  unweighted means per bin       hist Mean storage: mean and the variance of the mean  -> profile()
  weighted means per bin         sum w^2 (y - <y>)^2 / (sum w)^2  (hist's WeightedMean treats
                                weights as frequencies and underestimates this) -> profile(w=...)
  histograms of particles        per-event histograms averaged over events (event-to-event spread):
                                particles of one event are correlated, sqrt(N) is too small
                                                                               -> event_mean()
  class fractions                binomial (unweighted) / weighted                -> fraction()

All functions return (value, error) numpy arrays; empty bins give nan.
"""

from __future__ import annotations

import hist
import numpy as np


def counts(x, edges, w=None):
    """Histogram of x (one entry per event) with Poisson / sum-w^2 errors."""
    h = hist.Hist(hist.axis.Variable(np.asarray(edges, float)), storage=hist.storage.Weight())
    h.fill(np.asarray(x, float), weight=None if w is None else np.asarray(w, float))
    return h.values(), np.sqrt(h.variances())


def density(x, edges, w=None, area_mask=None):
    """Self-normalised density (unit area over the bins in area_mask, default all) with errors.
    The normalisation is treated as exact (its error is small against a single bin's)."""
    v, e = counts(x, edges, w)
    width = np.diff(np.asarray(edges, float))
    m = np.ones(len(v), bool) if area_mask is None else np.asarray(area_mask, bool)
    area = max(float(v[m].sum()), 1e-300)
    return v / area / width, e / area / width


def profile(x, y, edges, w=None):
    """Mean of y in bins of x and its standard error (weighted with importance weights w)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    edges = np.asarray(edges, float)
    if w is None:
        h = hist.Hist(hist.axis.Variable(edges), storage=hist.storage.Mean())
        h.fill(x, sample=y)
        n = h.counts()
        val = np.where(n > 0, h.values(), np.nan)
        err = np.where(n > 1, np.sqrt(np.maximum(h.variances(), 0)), np.nan)
        return val, err
    w = np.asarray(w, float)
    i = np.digitize(x, edges) - 1
    ok = (i >= 0) & (i < len(edges) - 1)
    i, y, w = i[ok], y[ok], w[ok]
    nb = len(edges) - 1
    sw = np.bincount(i, w, nb)
    mean = np.divide(np.bincount(i, w * y, nb), sw, out=np.full(nb, np.nan), where=sw > 0)
    d = y - mean[i]
    var = np.divide(np.bincount(i, w * w * d * d, nb), sw ** 2, out=np.full(nb, np.nan), where=sw > 0)
    return mean, np.sqrt(var)


def class_mean(cls, y, ncls, w=None):
    """Mean of y per class index 0 .. ncls-1 and its standard error."""
    return profile(np.asarray(cls, float), y, np.arange(ncls + 1) - 0.5, w)


def event_mean(per_event, w=None):
    """Column means of an (events x bins) array of per-event histograms and their standard errors
    (the event-to-event spread; the right error for particle distributions)."""
    a = np.asarray(per_event, float)
    nev, nb = a.shape
    return profile(np.tile(np.arange(nb), nev), a.ravel(), np.arange(nb + 1) - 0.5,
                   None if w is None else np.repeat(np.asarray(w, float), nb))


def fraction(sel, w=None):
    """Fraction of (weighted) events with sel true and its error."""
    sel = np.asarray(sel, bool)
    if w is None:
        n = len(sel)
        f = sel.mean()
        return float(f), float(np.sqrt(max(f * (1 - f), 0) / max(n, 1)))
    w = np.asarray(w, float)
    W = w.sum()
    f = w[sel].sum() / W
    var = (np.sum(w[sel] ** 2) * (1 - f) ** 2 + np.sum(w[~sel] ** 2) * f ** 2) / W ** 2
    return float(f), float(np.sqrt(var))


def ratio(a, ea, b, eb):
    """a/b with uncorrelated errors."""
    a, ea, b, eb = (np.asarray(v, float) for v in (a, ea, b, eb))
    r = np.divide(a, b, out=np.full(np.broadcast(a, b).shape, np.nan), where=b != 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.abs(r) * np.sqrt(np.where(a != 0, (ea / a) ** 2, 0) + np.where(b != 0, (eb / b) ** 2, 0))
    return r, e


def class_sums(cls, ncls, w=None):
    """Sum of w and of w^2 per class index 0 .. ncls-1 (hist Weight storage)."""
    h = hist.Hist(hist.axis.Integer(0, ncls), storage=hist.storage.Weight())
    h.fill(np.asarray(cls, int), weight=None if w is None else np.asarray(w, float))
    return h.values(), h.variances()


def _wfrac_var(W, W2):
    f = W / W.sum()
    return f, (W2 * (1 - f) ** 2 + (W2.sum() - W2) * f ** 2) / W.sum() ** 2


def raa_vac(cls_mb, ncoll_mb, cls_jet, w_jet, ncls):
    """Experiment-style R_AA^vac per class for N_coll-weighted jet events:
        (f_jet,c / f_MB,c) <N_coll>_MB / <N_coll>_MB,c = g_jet,c / g_MB,c,
    g = the N_coll-weighted fraction of the sample in class c (jets: their weights w_jet; minimum
    bias: N_coll).  The jet and minimum-bias samples are independent, each g has the
    weighted-fraction error.  Returns a dict (raa, err and the class sums for rcp())."""
    Wm, W2m = class_sums(cls_mb, ncls, ncoll_mb)
    Wj, W2j = class_sums(cls_jet, ncls, w_jet)
    gm, vm = _wfrac_var(Wm, W2m)
    gj, vj = _wfrac_var(Wj, W2j)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(gm > 0, gj / gm, np.nan)
        e = np.abs(r) * np.sqrt(np.where(gj > 0, vj / gj ** 2, np.nan) + np.where(gm > 0, vm / gm ** 2, np.nan))
    return {"raa": r, "err": e, "Wm": Wm, "W2m": W2m, "Wj": Wj, "W2j": W2j}


def rcp(rv, a, b):
    """R_CP = R_AA^vac(a) / R_AA^vac(b) from raa_vac()'s class sums; the sample totals cancel and
    the classes are disjoint, so the four sums are independent."""
    Wj, W2j, Wm, W2m = rv["Wj"], rv["W2j"], rv["Wm"], rv["W2m"]
    if min(Wj[a], Wj[b], Wm[a], Wm[b]) <= 0:
        return float("nan"), float("nan")
    r = (Wj[a] / Wj[b]) / (Wm[a] / Wm[b])
    rel = np.sqrt(W2j[a] / Wj[a] ** 2 + W2j[b] / Wj[b] ** 2 + W2m[a] / Wm[a] ** 2 + W2m[b] / Wm[b] ** 2)
    return float(r), float(r * rel)


# --- plotting -----------------------------------------------------------------------------------

def band(ax, x, y, e, color=None, alpha=0.25, **kw):
    """Line with a +-1 sigma band; returns the line."""
    (ln,) = ax.plot(x, y, color=color, **kw)
    y, e = np.asarray(y, float), np.asarray(e, float)
    ax.fill_between(x, y - e, y + e, color=ln.get_color(), alpha=alpha, lw=0)
    return ln


def stairs(ax, y, e, edges, color=None, alpha=0.25, **kw):
    """Step histogram with a +-1 sigma band per bin; returns the line."""
    y, e = np.asarray(y, float), np.asarray(e, float)
    ln = ax.stairs(y, edges, color=color, **kw)
    ax.stairs(y + e, edges, baseline=y - e, fill=True, color=ln.get_edgecolor(), alpha=alpha, lw=0)
    return ln


def points(ax, x, y, e, fmt="o-", **kw):
    """Markers with vertical error bars."""
    kw.setdefault("capsize", 2)
    kw.setdefault("ms", 3)
    return ax.errorbar(x, y, yerr=e, fmt=fmt, **kw)


def step(ax, x, edges, w=None, density=True, **kw):
    """ax.hist(..., histtype="step") with a +-1 sigma band (density=True: unit area)."""
    v, e = density_(x, edges, w) if density else counts(x, edges, w)
    return stairs(ax, v, e, edges, **kw)


density_ = density
