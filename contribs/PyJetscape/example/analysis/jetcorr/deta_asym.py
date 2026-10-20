"""The eta-separated dijet-hadron analysis of CMS HIN-25-012 (arXiv:2602.19431), trigger-free:
classes, reflection, the per-trigger near-side yields, the mixed-event acceptance, the
sideband, R_asym - R_sym and its integral, the Yang-Wang lab-eta form (arXiv:2508.04194), and
a jackknife over blocks of hydro backgrounds.  dijet/deta.py runs it with the Z as jet1 (the
Z+jet campaigns' pools and jets); the dijet stage gives it jet1 instead.

The inputs are per trigger event i: its weight w_i, the axis' rapidity y_i (jet1's, the Z's),
and its hadron contents on the fine Delta y grid per side of the axis, ``C_i`` of shape
(N_PT, 2, N_DY) (Delta y = eta - y_i in 0.1 over +-4.8, side 0 the near side |Delta phi| <
pi/2: zjet_mixing.side_hist), or in lab eta per side, ``L_i`` (N_PT, 2, N_ETA)
(zjet_mixing.lab_side_hist).  A trigger's oversamples can fall into different classes (jet2
moves with the fragmentation and, in Pb+Pb, with the background): ``event_classes`` turns
per-oversample classes into per-event fractions f_ic and a reflection per class (the majority
of its oversamples), and every class sum weighs the event's contents by w_i f_ic.

CMS's steps (2602.19431): reflect so that eta_jet1 > eta_jet2 (jet2 and its wake at negative
Delta eta); per trigger and unit Delta eta on the near side; the pair acceptance corrected
with mixed events (jet1 of one event with the tracks of others, normalized to unity where the
acceptance is full: here the triggers' rapidities folded with a pooled lab density,
``mixed_event``, reflected with their contents); the level of 1.5 < Delta eta < 2.5 subtracted per class; R_asym - R_sym for
the large-gap classes; their integral over -2.4 < Delta eta < -0.6.  CMS's HEPData record
(ins3123406) bins Delta eta in 0.2 with edges at odd tenths, and its Fig. 3 values are the
plain sums of Fig. 2's over the 10 bins centred -2.4 ... -0.6 (edges -2.5 ... -0.5):
``cms_bins``, ``integral``.
"""
import warnings

import numpy as np

#: the fine Delta y grid (zjet_mixing.DY_FINE) and lab eta grid (zjet_mixing.ETA)
DY = np.linspace(-4.8, 4.8, 97)
DYC = 0.5 * (DY[1:] + DY[:-1])
N_DY = len(DYC)
ETA = np.linspace(-2.4, 2.4, 49)
ETAC = 0.5 * (ETA[1:] + ETA[:-1])
WIDTH = 0.1
#: CMS's gap classes |Delta eta_jet1,jet2| (the first: R_sym)
GAPS = ((0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0))
SIDEBAND = (1.5, 2.5)
SIGNAL = (-2.4, -0.6)            # CMS's words; its bins: centres -2.4 ... -0.6
SIGNAL_BINS = (-2.5, -0.5)       # the edges of those 10 bins of 0.2
TRACK_ETA = 2.4                  # CMS's |eta_ch| < 2.4


def _nanmean(x, axis):
    """np.nanmean without the warning for all-NaN slices (bins outside the acceptance)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(x, axis=axis)


# ── classes ──────────────────────────────────────────────────────────────────────
def gap_class(y1, eta2, gaps=GAPS):
    """CMS's gap class of every oversample: the index of |y1 - eta2| in ``gaps`` (-1 outside,
    or no jet2: eta2 NaN), and its reflection (eta2 > y1: Delta eta -> -Delta eta, so that
    jet2 sits at negative Delta eta)."""
    y1, eta2 = np.asarray(y1, float), np.asarray(eta2, float)
    g = np.abs(y1 - eta2)
    cls = np.full(len(g), -1, np.int64)
    for c, (lo, hi) in enumerate(gaps):
        cls[(g >= lo) & (g < hi)] = c
    return cls, np.nan_to_num(eta2 - y1, nan=0.0) > 0


def jet_eta_class(eta2, edges=(0.0, 0.8, 1.6)):
    """Yang-Wang's classes by jet2's lab |eta| (2508.04194: |eta_jet| < 0.8 against 0.8-1.6,
    their jets |eta| < 1.6) and the reflection that puts jet2 at eta > 0 (eta_h -> -eta_h)."""
    eta2 = np.asarray(eta2, float)
    a = np.abs(eta2)
    cls = np.full(len(a), -1, np.int64)
    for c in range(len(edges) - 1):
        cls[(a >= edges[c]) & (a < edges[c + 1])] = c
    return cls, np.nan_to_num(eta2, nan=0.0) < 0


def hemisphere_class(y1, eta2):
    """Hemisphere classes (2508.04194 Sec. III): 0 jet1 and jet2 in the same hemisphere
    (y1 eta2 > 0), 1 opposite; reflection to jet2 at eta > 0."""
    y1, eta2 = np.asarray(y1, float), np.asarray(eta2, float)
    cls = np.where(np.isnan(eta2), -1, np.where(y1 * eta2 > 0, 0, 1)).astype(np.int64)
    return cls, np.nan_to_num(eta2, nan=0.0) < 0


def event_classes(ev_of_row, n_events, n_rows_of_event, cls, refl, n_classes):
    """Per-oversample classes (rows: their event ``ev_of_row``, class ``cls`` (-1 none) and
    reflection ``refl``) -> per event (f (n_events, n_classes): the fraction of its
    ``n_rows_of_event`` oversamples in each class; reflect (n_events, n_classes): the
    majority of the class's oversamples, ties to no reflection)."""
    ev = np.asarray(ev_of_row, np.int64)
    cls = np.asarray(cls, np.int64)
    refl = np.asarray(refl, bool)
    f = np.zeros((n_events, n_classes))
    nr = np.zeros((n_events, n_classes))
    ok = cls >= 0
    np.add.at(f, (ev[ok], cls[ok]), 1.0)
    np.add.at(nr, (ev[ok], cls[ok]), refl[ok].astype(float))
    reflect = nr > 0.5 * f
    n = np.asarray(n_rows_of_event, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(n[:, None] > 0, f / n[:, None], 0.0)
    return f, reflect


# ── per-event sums, the inputs of everything below (linear: blocks subtract) ─────────
def flip(x):
    """Reflection Delta y -> -Delta y (or eta -> -eta): the grids are symmetric."""
    return x[..., ::-1]


def class_sums(C, w, f, reflect):
    """sum_i w_i f_ic C_i (reflected where ``reflect``) and sum_i w_i f_ic per class ->
    (num (n_classes, *C.shape[1:]), den (n_classes,)).  C: (n, ...) per event."""
    C = np.asarray(C, float)
    wf = np.asarray(w, float)[:, None] * f
    a, b = wf * ~reflect, wf * reflect
    num = np.tensordot(a.T, C, axes=1) + np.tensordot(b.T, flip(C), axes=1)
    return num, wf.sum(axis=0)


def shift_density(rho, y):
    """A lab-eta density rho (N_PT, N_ETA) seen from axes at rapidities y (n,): every lab bin
    centre moved by -y into its Delta y bin (zjet_mixing.ZKernel's rule) -> (n, N_PT, N_DY)."""
    y = np.asarray(y, float)
    j = np.clip(np.searchsorted(DY, ETAC[None, :] - y[:, None], side="right") - 1, 0,
                len(DYC) - 1)                                     # (n, N_ETA)
    out = np.zeros((len(y), rho.shape[0], len(DYC)))
    rows = np.repeat(np.arange(len(y)), len(ETAC))
    for p in range(rho.shape[0]):
        np.add.at(out[:, p], (rows, j.ravel()), np.tile(rho[p], len(y)))
    return out


def mixed_event(y1, rho):
    """Per trigger the mixed-event pair density of its axis at y1: the lab density rho (N_PT,
    N_ETA) shifted (``shift_density``) -> (n, N_PT, N_DY).  A reflected trigger's is the same
    flipped (Delta y -> -Delta y, as its contents), so ``class_sums`` of it is the class's
    mixed event.  rho: a pooled density of other events' tracks (CMS's mixed events), or ones
    inside |eta| < 2.4 (the pure geometric pair acceptance, ``flat_density``)."""
    return shift_density(rho, y1)


def flat_density(n_pt):
    """Ones in every lab-eta bin (|eta| < 2.4): the geometric pair acceptance."""
    return np.ones((n_pt, len(ETAC)))


def full_acceptance(ymax):
    """The Delta y bins where every axis with |y| < ymax sees full track acceptance."""
    return np.abs(DYC) < TRACK_ETA - ymax - 1e-9


# ── the observable ───────────────────────────────────────────────────────────────
def corrected(num, den, me, norm_bins, side=0, with_acceptance=False):
    """Per class the per-trigger near-side yield per unit Delta y, divided by the mixed
    event normalized to unity over ``norm_bins`` (where the acceptance is full) -> (n_classes,
    N_PT, N_DY); NaN where the mixed event is empty.  num: class_sums' (n_classes, N_PT, 2,
    N_DY); me: the class sums of mixed_event (n_classes, N_PT, N_DY).  ``with_acceptance``:
    also the normalized mixed event (the pair acceptance, the sideband's weights)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        y = num[:, :, side] / den[:, None, None] / WIDTH
        m = me / me[..., norm_bins].mean(axis=-1, keepdims=True)
        yc = np.where(m > 1e-6, y / m, np.nan)
    return (yc, m) if with_acceptance else yc


def window_mean(y, lo, hi, weights=None):
    """The mean of y (..., N_DY) over the bins with centres in (lo, hi), weighted by
    ``weights`` (the pair acceptance: each bin as its pairs, as a fit of a constant weighs
    Poisson bins; None: plain)."""
    m = (DYC > lo) & (DYC < hi)
    if weights is None:
        return _nanmean(y[..., m], axis=-1)
    x, w = y[..., m], np.where(np.isfinite(y[..., m]), weights[..., m], 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nansum(x * w, axis=-1) / w.sum(axis=-1)


def asym_minus_sym(yc, sideband=SIDEBAND, subtract=True, acceptance=None):
    """R_asym - R_sym: every class's corrected yield minus its sideband level (``subtract``;
    the mean weighted by ``acceptance``, corrected's), the large-gap classes minus the first
    (R_sym) -> (dif (n_classes - 1, N_PT, N_DY), level (n_classes, N_PT))."""
    level = window_mean(yc, *sideband, weights=acceptance)
    r = yc - level[..., None] if subtract else yc
    return r[1:] - r[:1], level


def integral(x, lo=SIGNAL_BINS[0], hi=SIGNAL_BINS[1]):
    """The integral over lo < Delta y < hi (bins with centres inside, times 0.1): -2.5 ...
    -0.5 is the range of CMS's 10 bins centred -2.4 ... -0.6 (its Fig. 3 lists the plain sum
    of Fig. 2's values over them, i.e. 5x this integral if Fig. 2 is per unit Delta eta)."""
    m = (DYC > lo) & (DYC < hi)
    return np.nansum(x[..., m], axis=-1) * WIDTH


def rebin(x, lo=-2.5, hi=2.1, width=0.2):
    """Fine bins (0.1) averaged into bins of ``width`` from ``lo`` to ``hi`` (default: CMS
    HIN-25-012's 23 bins of 0.2, edges at odd tenths) -> (values (..., n), centres)."""
    k = int(round(width / WIDTH))
    i0 = int(round((lo - DY[0]) / WIDTH))
    n = int(round((hi - lo) / width))
    blocks = x[..., i0:i0 + n * k].reshape(*x.shape[:-1], n, k)
    return _nanmean(blocks, axis=-1), lo + width * (np.arange(n) + 0.5)


# ── the Yang-Wang form: lab eta on the trigger side ──────────────────────────────
def lab_yields(num, den, side=0):
    """Per class the per-trigger dN/deta on ``side`` (0: the trigger's side, |Delta phi| <
    pi/2) -> (n_classes, N_PT, N_ETA).  num: class_sums of the L contents."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return num[:, :, side] / den[:, None, None] / WIDTH


# ── errors ───────────────────────────────────────────────────────────────────────
def blocks_of(groups, n_blocks):
    """The distinct ``groups`` (hydro backgrounds) in ``n_blocks`` contiguous blocks -> per
    event its block index."""
    u, inv = np.unique(np.asarray(groups), return_inverse=True)
    b = np.zeros(len(u), np.int64)
    for i, part in enumerate(np.array_split(np.arange(len(u)), min(n_blocks, len(u)))):
        b[part] = i
    return b[inv.ravel()]


def jackknife(fn, block, n_blocks=None):
    """The delete-one-block jackknife of ``fn(keep)`` (keep: a boolean mask over events) ->
    (value, error).  fn returns an array; NaN-safe."""
    block = np.asarray(block)
    ids = np.unique(block)
    full = np.asarray(fn(np.ones(len(block), bool)), float)
    if len(ids) < 2:
        return full, np.full_like(full, np.nan)
    reps = np.array([np.asarray(fn(block != b), float) for b in ids])
    g = len(ids)
    err = np.sqrt((g - 1) / g * np.nansum((reps - _nanmean(reps, axis=0)) ** 2, axis=0))
    return full, err
