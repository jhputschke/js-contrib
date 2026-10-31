"""Minimal flow and bulk observables of a production: Nch, [p_T], v_n{2} with an eta gap, and
the paired difference jet leg - background (docs/Plans/PLAN_multi_hard.md, D15 and §3.10).

    from flow import Cuts, event_table, class_averages, select
    with HadronFileReader("out_had") as reader:
        t = event_table(reader, Cuts())                      # one row per event
    res = class_averages(t, select(t, n_coll_percent=(0, 10)))
    res["full/v2"], res["delta_full/dv2"]                    # (value, jackknife error)

**Compositions.**  Every event is evaluated as

    full        bulk_jet + jet_frag + extra_frag   (the whole event; D16)
    no_extras   bulk_jet + jet_frag                (the extras' fragments left out)
    bulk_jet    bulk_jet                           (iSS on the jet leg: bulk + all wakes)
    bulk_bg     bulk_bg                            (iSS on the jet-free background)

and each of the first three against the background, ``delta_<composition>`` = composition -
bulk_bg.  A minimum-bias file (run_prod_jet.py --hard none) has no jet_frag, a file without
extra hard scatterings no extra_frag: the missing tags contribute nothing (full = no_extras
without extras).  The fragments of oversample k are fragmentation k mod n_frag, as
:meth:`HadronFileReader.jet_event` picks them.

**Per sample** (one oversample of one composition) in the charged-hadron window (|pid| in
``Cuts.pids``, default ``hadrons_h5.CHARGED``; pt_min < p_T < pt_max):

    N         hadrons with |eta| < eta_max
    S         sum of their p_T;   [p_T] = S / N   (undefined for N = 0)
    Q_n^A     sum of exp(i n phi) over -flow_eta_max < eta < -eta_gap / 2   (M_A hadrons)
    Q_n^B     the same over eta_gap / 2 < eta < flow_eta_max               (M_B hadrons)

Q-vectors add across the tags of a composition sample by sample (bulk_jet sample k + jet_frag
sample k mod n_frag + extra_frag sample k mod n_extra_frag); the pair sums do not, so every
composition is formed before any product is taken.

**Per event** (the mean over its K oversamples, every oversample a separate Cooper-Frye
sampling of the same fluid, never mixed with another sample in a pair):

    nch       mean_k N_k
    pt        mean_k [p_T]_k        (samples with N_k > 0; n_pt of them)
    pt_var    var_k [p_T]_k, ddof 1 (NaN for n_pt < 2): the sampling noise of one sample
    c_n       mean_k Re(Q_n^A Q_n^B*)          the pair sum of the two-subevent correlator
    w         mean_k M_A M_B                   its number of pairs

so each oversample is treated as a physical event (pairs within one sample: the decays'
non-flow is in, as in data), and an event with more oversamples is not weighted more.

**Per class** (:func:`class_averages`, events weigh the same):

    nch, dnch_deta    <nch>, and <nch> / (2 eta_max)
    pt                <pt> over the events
    sigma_pt          the spread over events of the sample-averaged pt (includes the noise
                      var_k / K of each event's average; this is NOT what one measures)
    sigma_pt_dyn      sqrt(sigma_pt^2 - <pt_var / n_pt>): the event-by-event fluctuation of the
                      fluid's [p_T], the oversampling noise removed (NaN with K = 1)
    sigma_pt_one      sqrt(sigma_pt_dyn^2 + <pt_var>): the spread with one sample per event,
                      i.e. what an experiment sees (its own noise included)
    c2, c3            <<2>>_n = sum_e c_n / sum_e w  (the multiplicity-weighted two-subevent
                      correlator; no self pairs since A and B do not overlap)
    v2, v3            v_n{2, |Delta eta| > eta_gap} = sqrt(<<2>>_n) (NaN if <<2>>_n <= 0)

and per ``delta_<composition>`` over the events that have both legs:

    dnch, dpt         <nch - nch_bg>, <pt - pt_bg>
    dc2, dc3          <<2>>_n(composition) - <<2>>_n(bulk_bg): the change of the correlator
    dv2, dv3          v_n{2}(composition) - v_n{2}(bulk_bg)
    rdnch             dnch / <nch_bg>

**The flow vector's kick** (per ``delta_<composition>``; the paired samples k of the composition
and of bulk_bg, as hadronize.py --correlated pairs them, the first min(K, K_bg) of each):
with q_n^X = Q_n^X / M_X the per-hadron flow vector of subevent X and Delta q^X = q^X(composition)
- q^X(bulk_bg) per sample,

    kick2_n   mean_k Re(Delta q^A Delta q^B*): <|Delta V_n|^2>, the squared change of the event's
              flow vector (size and direction).  A and B have independent sampling noise, so
              it is unbiased by the oversampling (as v_n{2} is by the gap)
    align_n   mean_k Re(q_bg^A Delta q^B* + Delta q^A q_bg^B*): 2 <V . Delta V>, the change
              along the existing flow (dilution and push; the random part averages out)
    xc_n, cc_n, bb_n  mean_k of Re(q^A q_bg^B*) (symmetrised), Re(q^A q^B*), Re(q_bg^A q_bg^B*)

so that the per-hadron <<2>> changes by exactly align_n + kick2_n.  Per class: kick2{n}
(the mean), kick{n} = sqrt(kick2{n}), rkick{n} = kick{n} / v_n{2} of bulk_bg (the kick
relative to the flow), align{n}, and r{n} = <xc> / sqrt(<cc> <bb>), the correlation of the
flow vectors with and without (1: untouched; 1 - r_n ~ kick2 / (2 v_n^2) for a random kick).
Every event weighs the same here (per-hadron vectors), unlike the multiplicity-weighted c_n.

dc_n is the primary quantity (additive, defined when v_n{2} is not); dv_n is what the plan's
V5 compares.  Both are differences of class averages over the same events, so the errors
(from the jackknife over the same blocks) keep the correlation of the two legs: with
hadronize.py --correlated, sample k of bulk_jet and of its bulk_bg share their random numbers
and the difference is low-noise; without it the legs are independent samplings and the error
says so.  The legs may have different K: only sample averages are differenced.

**Errors** are delete-one-block jackknife errors.  The unit is the hydro background, not the
event: events that share a background (run_prod_jet.py --reuse) share their bulk and are left
out together; with --reuse 1 the unit is the event.  More than ``max_blocks`` backgrounds are
grouped into that many contiguous blocks.  A background reused by several events counts once
per event, as the readers count it.

**Selections** (:func:`select`): a list of events, an n_coll range, or n_coll percentiles
(the events ordered by N_coll, largest first: a centrality-like proxy from the Glauber
collision, not multiplicity classes).

Not here (a later project, D15): v_n{4} and cumulants, rho(v_2^2, [p_T]), symmetry planes,
multiplicity-based centrality, pT-differential v_n, event weights of --pthat-bins campaigns
(every event weighs the same).
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass, field

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
try:
    import jetscape  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "python"))
from jetscape.hadrons_h5 import CHARGED, ORIGIN, pseudorapidity  # noqa: E402

#: the compositions of an event: name -> ORIGIN codes of jet_event() it adds up
JET_COMPOSITIONS = {"full": ("bulk", "frag", "extra"), "no_extras": ("bulk", "frag"),
                    "bulk_jet": ("bulk",)}
BG = "bulk_bg"
COMPOSITIONS = tuple(JET_COMPOSITIONS) + (BG,)
DELTAS = tuple(f"delta_{c}" for c in JET_COMPOSITIONS)

#: per-event columns of every composition (``<composition>.<column>`` in the table)
EVENT_COLUMNS = ("K", "nch", "pt", "pt_var", "n_pt", "c", "w")
#: per-event columns of every paired difference (``delta_<composition>.<column>``; one value
#: per harmonic): the flow vector's kick (module docstring)
KICK_COLUMNS = ("kick2", "align", "xc", "cc", "bb")


@dataclass(frozen=True)
class Cuts:
    """The acceptance.  ``flow_eta_max`` None: ``eta_max`` (the subevents are then
    eta_gap/2 < |eta| < eta_max, e.g. 0.5 < |eta| < 0.8 for the defaults)."""

    eta_max: float = 0.8
    pt_min: float = 0.2
    pt_max: float = 3.0
    eta_gap: float = 1.0
    flow_eta_max: float | None = None
    harmonics: tuple = (2, 3)
    pids: tuple = field(default=CHARGED)

    def __post_init__(self):
        if not self.eta_max > 0 or not self.pt_max > self.pt_min >= 0:
            raise ValueError(f"bad window: |eta| < {self.eta_max}, "
                             f"{self.pt_min} < p_T < {self.pt_max}")
        if not self.flow_eta > self.eta_gap / 2:
            raise ValueError(f"empty subevents: eta_gap / 2 = {self.eta_gap / 2} >= "
                             f"flow_eta_max = {self.flow_eta}")
        if not self.harmonics or any(int(n) < 1 for n in self.harmonics):
            raise ValueError(f"harmonics must be >= 1, got {self.harmonics}")

    @property
    def flow_eta(self):
        return self.eta_max if self.flow_eta_max is None else float(self.flow_eta_max)

    @property
    def eta_reach(self):
        """The largest |eta| any observable uses."""
        return max(self.eta_max, self.flow_eta)

    def as_dict(self):
        return {"eta_max": self.eta_max, "pt_min": self.pt_min, "pt_max": self.pt_max,
                "eta_gap": self.eta_gap, "flow_eta_max": self.flow_eta,
                "harmonics": [int(n) for n in self.harmonics],
                "pids": [int(p) for p in self.pids]}


# ── per sample ──────────────────────────────────────────────────────────────────
def sample_sums(p, pid, cuts, group=None, n_groups=1):
    """The additive sums of hadrons ``p`` (N, 4) [E, px, py, pz], ``pid`` (N,), split by
    ``group`` (N,) ints in 0 .. n_groups-1 (None: one group) -> dict of arrays: N, S
    (n_groups,), MA, MB (n_groups,), QA, QB (n_groups, n_harmonics) complex."""
    p = np.asarray(p, dtype=np.float64).reshape(-1, 4)
    pid = np.asarray(pid).reshape(-1)
    g = (np.zeros(len(pid), np.int64) if group is None
         else np.asarray(group, dtype=np.int64).reshape(-1))
    pt = np.hypot(p[:, 1], p[:, 2])
    eta = pseudorapidity(p)
    phi = np.arctan2(p[:, 2], p[:, 1])
    sel = np.isin(np.abs(pid), cuts.pids) & (pt > cuts.pt_min) & (pt < cuts.pt_max)
    win = sel & (np.abs(eta) < cuts.eta_max)
    half, fe = cuts.eta_gap / 2, cuts.flow_eta
    a = sel & (eta < -half) & (eta > -fe)
    b = sel & (eta > half) & (eta < fe)

    def count(m, w=None):
        return np.bincount(g[m], weights=None if w is None else w[m], minlength=n_groups)

    out = {"N": count(win).astype(np.float64), "S": count(win, pt),
           "MA": count(a).astype(np.float64), "MB": count(b).astype(np.float64)}
    for key, m in (("QA", a), ("QB", b)):
        q = np.zeros((n_groups, len(cuts.harmonics)), np.complex128)
        for j, n in enumerate(cuts.harmonics):
            q[:, j] = count(m, np.cos(n * phi)) + 1j * count(m, np.sin(n * phi))
        out[key] = q
    return out


def _stack(sums):
    """A list of per-sample sums (one group each) -> dict of arrays with a leading K axis."""
    return {k: np.stack([s[k][0] for s in sums]) for k in sums[0]} if sums else None


def event_values(s, n_harmonics):
    """The per-event columns (EVENT_COLUMNS) from per-sample sums ``s`` (dict of arrays with
    a leading axis of K samples; None or K = 0: an event without samples, all NaN)."""
    K = 0 if s is None else len(s["N"])
    if K == 0:
        nan = np.full(n_harmonics, np.nan)
        return {"K": 0, "nch": np.nan, "pt": np.nan, "pt_var": np.nan, "n_pt": 0,
                "c": nan, "w": np.nan}
    N, S = s["N"], s["S"]
    ok = N > 0
    mpt = S[ok] / N[ok]
    c = np.real(s["QA"] * np.conj(s["QB"])).mean(axis=0)
    return {"K": K, "nch": float(N.mean()), "pt": float(mpt.mean()) if ok.any() else np.nan,
            "pt_var": float(mpt.var(ddof=1)) if ok.sum() > 1 else np.nan,
            "n_pt": int(ok.sum()), "c": c, "w": float((s["MA"] * s["MB"]).mean())}


def kick_values(s, sb, n_harmonics):
    """The flow vector's kick of a composition against bulk_bg (KICK_COLUMNS, module
    docstring) from their per-sample sums ``s`` and ``sb``, sample k against sample k (the
    first min(K, K_bg)); NaN without a pair whose four subevents all have hadrons."""
    nan = {c: np.full(n_harmonics, np.nan) for c in KICK_COLUMNS}
    if s is None or sb is None:
        return nan
    K = min(len(s["N"]), len(sb["N"]))
    ok = ((s["MA"][:K] > 0) & (s["MB"][:K] > 0) & (sb["MA"][:K] > 0) & (sb["MB"][:K] > 0))
    if not ok.any():
        return nan

    def q(x, key, m):
        return x[key][:K][ok] / x[m][:K][ok, None]
    qa, qb, ra, rb = q(s, "QA", "MA"), q(s, "QB", "MB"), q(sb, "QA", "MA"), q(sb, "QB", "MB")
    da, db = qa - ra, qb - rb
    return {"kick2": np.real(da * np.conj(db)).mean(axis=0),
            "align": np.real(ra * np.conj(db) + da * np.conj(rb)).mean(axis=0),
            "xc": 0.5 * np.real(qa * np.conj(rb) + ra * np.conj(qb)).mean(axis=0),
            "cc": np.real(qa * np.conj(qb)).mean(axis=0),
            "bb": np.real(ra * np.conj(rb)).mean(axis=0)}


# ── per event ───────────────────────────────────────────────────────────────────
def _event_record(reader, e):
    """N_coll, the extras (count, energy, p_T) and the hydro background of event ``e``."""
    info = reader.event_info(e)
    sc = info.scatterings()
    ex = (np.zeros((0, 8)) if sc is None else np.asarray(sc).reshape(-1, 8))
    ex = ex[ex[:, 0] > 0]                     # scatter 0 is the trigger (absent: min. bias)
    return {"event": e, "file_index": info.file_index, "local_event": info.local_event,
            "bg_unit": -1 if info.bg_unit is None else info.bg_unit,
            "n_coll": np.nan if info.n_coll is None else float(info.n_coll),
            "n_extra": len(ex), "extra_E": float(ex[:, 6].sum()),
            "extra_pt": float(ex[:, 7].sum())}


def event_table(reader, cuts=None, events=None, progress=None):
    """One row per event of ``reader`` (a jetscape.hadrons_h5.HadronFileReader) -> dict of
    arrays: the event record (event, file_index, local_event, bg_unit (-1: unknown), n_coll
    (NaN: unknown), n_extra, extra_E, extra_pt: the extras' count and summed E and p_T from
    the extra_frag file's scatterings, 0 without), per composition the EVENT_COLUMNS as
    ``<composition>.<column>`` (``.c`` is (n_events, n_harmonics)), ``group`` (the jackknife
    unit), ``harmonics`` and ``cuts``.  ``events``: global event indices (default all).
    ``progress``: a callable(i, n) or None.  A background reused by several events is read
    once while they follow each other."""
    cuts = cuts or Cuts()
    stored = reader.eta_max()
    if stored is not None and stored < cuts.eta_reach:
        raise ValueError(f"the hadron files keep only |eta| < {stored} (hadronize.py "
                         f"--eta-max), the cuts reach |eta| = {cuts.eta_reach}")
    nh = len(cuts.harmonics)
    origins = [ORIGIN[o] for o in ("bulk", "frag", "extra")]
    g_of = np.full(max(origins) + 1, -1, np.int64)
    g_of[origins] = np.arange(len(origins))
    g = reader.events(events)
    rows = []
    bg_cache = {}
    for i, e in enumerate(g):
        rec = _event_record(reader, int(e))
        K = reader.n_samples("bulk_jet", e)
        per = []
        for k in range(K):
            ev = reader.jet_event(int(e), k)
            per.append(sample_sums(ev["p"], ev["pid"], cuts, g_of[ev["origin"]],
                                   len(origins)))
        stacks = {}
        for name, parts in JET_COMPOSITIONS.items():
            idx = [origins.index(ORIGIN[o]) for o in parts]
            s = ({key: np.stack([sum(x[key][j] for j in idx) for x in per]) for key in per[0]}
                 if per else None)
            stacks[name] = s
            rec.update({f"{name}.{c}": v for c, v in event_values(s, nh).items()})
        key = (rec["file_index"], rec["bg_unit"] if rec["bg_unit"] >= 0 else -1 - int(e))
        if key not in bg_cache:
            if len(bg_cache) > 8:
                bg_cache.clear()
            has_bg = "bulk_bg" in reader.tags(rec["file_index"])
            Kb = reader.n_samples("bulk_bg", e) if has_bg else 0
            sb = _stack([sample_sums(ev["p"], ev["pid"], cuts)
                         for ev in (reader.background_event(int(e), k) for k in range(Kb))])
            bg_cache[key] = (event_values(sb, nh), sb)
        rec.update({f"{BG}.{c}": v for c, v in bg_cache[key][0].items()})
        for name, s in stacks.items():
            rec.update({f"delta_{name}.{c}": v
                        for c, v in kick_values(s, bg_cache[key][1], nh).items()})
        rows.append(rec)
        if progress is not None:
            progress(i + 1, len(g))
    t = {k: np.array([r[k] for r in rows]) for k in (rows[0] if rows else _empty_keys())}
    for c in COMPOSITIONS:
        if f"{c}.c" in t:
            t[f"{c}.c"] = np.asarray(t[f"{c}.c"], dtype=np.float64).reshape(len(rows), nh)
    for d in DELTAS:
        for c in KICK_COLUMNS:
            if f"{d}.{c}" in t:
                t[f"{d}.{c}"] = np.asarray(t[f"{d}.{c}"], dtype=np.float64).reshape(len(rows),
                                                                                   nh)
    t["harmonics"] = np.asarray(cuts.harmonics, dtype=np.int64)
    t["cuts"] = cuts
    t["group"] = _groups(t)
    return t


def _empty_keys():
    return (["event", "file_index", "local_event", "bg_unit", "n_coll", "n_extra", "extra_E",
             "extra_pt"] + [f"{c}.{x}" for c in COMPOSITIONS for x in EVENT_COLUMNS]
            + [f"{d}.{x}" for d in DELTAS for x in KICK_COLUMNS])


def _groups(t):
    """The jackknife unit of each event: its hydro background (file, bg_unit); an event
    without a recorded background is its own unit."""
    n = len(t["event"])
    if n == 0:
        return np.zeros(0, np.int64)
    bg = np.asarray(t["bg_unit"], dtype=np.int64)
    key = np.where(bg >= 0, bg, -1 - np.asarray(t["local_event"], dtype=np.int64))
    pairs = np.stack([np.asarray(t["file_index"], dtype=np.int64), key], axis=1)
    return np.unique(pairs, axis=0, return_inverse=True)[1].ravel()


# ── selections ──────────────────────────────────────────────────────────────────
def select(t, events=None, n_coll=None, n_coll_percent=None, mask=None):
    """A boolean mask over the table's events: ``events`` (global indices), ``n_coll`` =
    (lo, hi) with lo <= N_coll < hi (None for an open end), ``n_coll_percent`` = (lo, hi):
    the events from lo % to hi % when ordered by N_coll, largest first (0-10: the 10 %
    with the most binary collisions), among those passing the other cuts.  Events
    without N_coll fail the N_coll cuts."""
    n = len(t["event"])
    m = np.ones(n, bool) if mask is None else np.asarray(mask, bool).copy()
    if events is not None:
        m &= np.isin(t["event"], np.asarray(list(events), dtype=np.int64))
    nc = np.asarray(t["n_coll"], dtype=np.float64)
    if n_coll is not None:
        lo, hi = n_coll
        with np.errstate(invalid="ignore"):
            m &= np.isfinite(nc)
            if lo is not None:
                m &= nc >= lo
            if hi is not None:
                m &= nc < hi
    if n_coll_percent is not None:
        lo, hi = n_coll_percent
        idx = np.flatnonzero(m & np.isfinite(nc))
        order = idx[np.argsort(-nc[idx], kind="stable")]
        pct = 100.0 * (np.arange(len(order)) + 0.5) / max(len(order), 1)
        keep = np.zeros(n, bool)
        keep[order[(pct >= lo) & (pct < hi)]] = True
        m &= keep
    return m


# ── class averages ──────────────────────────────────────────────────────────────
def _comp_stats(t, c, keep, cuts):
    harm = [int(n) for n in t["harmonics"]]
    ok = keep & (t[f"{c}.K"] > 0)
    out = {}
    nch = t[f"{c}.nch"][ok]
    out["nch"] = nch.mean() if len(nch) else np.nan
    out["dnch_deta"] = out["nch"] / (2 * cuts.eta_max)
    pt = t[f"{c}.pt"][ok]
    pt_ok = np.isfinite(pt)
    pt = pt[pt_ok]
    out["pt"] = pt.mean() if len(pt) else np.nan
    var = pt.var(ddof=1) if len(pt) > 1 else np.nan
    out["sigma_pt"] = np.sqrt(var)
    pv = t[f"{c}.pt_var"][ok][pt_ok]
    npt = t[f"{c}.n_pt"][ok][pt_ok]
    have = np.isfinite(pv)
    if have.any() and have.all():
        dyn = var - np.mean(pv / npt)
        out["var_pt_dyn"] = dyn
        out["sigma_pt_dyn"] = np.sqrt(dyn) if dyn > 0 else np.nan
        one = dyn + np.mean(pv)
        out["sigma_pt_one"] = np.sqrt(one) if one > 0 else np.nan
    else:
        out["var_pt_dyn"] = out["sigma_pt_dyn"] = out["sigma_pt_one"] = np.nan
    w = t[f"{c}.w"][ok].sum()
    cc = t[f"{c}.c"][ok].sum(axis=0)
    for j, n in enumerate(harm):
        cn = cc[j] / w if w > 0 else np.nan
        out[f"c{n}"] = cn
        out[f"v{n}"] = np.sqrt(cn) if cn > 0 else np.nan
    return out


def _stats(t, keep, cuts):
    """All class quantities on the events ``keep`` -> flat dict "<name>/<quantity>"."""
    out = {}
    for c in COMPOSITIONS:
        out.update({f"{c}/{k}": v for k, v in _comp_stats(t, c, keep, cuts).items()})
    harm = [int(n) for n in t["harmonics"]]
    for c in JET_COMPOSITIONS:
        both = keep & (t[f"{c}.K"] > 0) & (t[f"{BG}.K"] > 0)
        a, b = _comp_stats(t, c, both, cuts), _comp_stats(t, BG, both, cuts)
        d = {}
        dn = t[f"{c}.nch"][both] - t[f"{BG}.nch"][both]
        d["dnch"] = dn.mean() if len(dn) else np.nan
        d["rdnch"] = d["dnch"] / b["nch"] if b["nch"] > 0 else np.nan
        dp = t[f"{c}.pt"][both] - t[f"{BG}.pt"][both]
        dp = dp[np.isfinite(dp)]
        d["dpt"] = dp.mean() if len(dp) else np.nan
        for n in harm:
            d[f"dc{n}"] = a[f"c{n}"] - b[f"c{n}"]
            d[f"dv{n}"] = a[f"v{n}"] - b[f"v{n}"]
        if f"delta_{c}.kick2" in t:                 # the flow vector's kick
            kk = {x: t[f"delta_{c}.{x}"][both] for x in KICK_COLUMNS}
            fin = np.all([np.isfinite(v).all(axis=1) for v in kk.values()], axis=0)
            for j, n in enumerate(harm):
                m = {x: v[fin, j].mean() if fin.any() else np.nan for x, v in kk.items()}
                d[f"kick2{n}"] = m["kick2"]
                d[f"kick{n}"] = np.sqrt(m["kick2"]) if m["kick2"] > 0 else np.nan
                d[f"rkick{n}"] = (d[f"kick{n}"] / b[f"v{n}"] if b[f"v{n}"] > 0
                                  else np.nan)
                d[f"align{n}"] = m["align"]
                den = m["cc"] * m["bb"]
                d[f"r{n}"] = m["xc"] / np.sqrt(den) if den > 0 else np.nan
        d["n_events"] = float(both.sum())
        out.update({f"delta_{c}/{k}": v for k, v in d.items()})
    for c in COMPOSITIONS:
        out[f"{c}/n_events"] = float((keep & (t[f"{c}.K"] > 0)).sum())
    return out


def blocks_of(groups, max_blocks):
    """The distinct jackknife units ``groups`` in at most ``max_blocks`` contiguous blocks ->
    per event its block index."""
    u, inv = np.unique(np.asarray(groups), return_inverse=True)
    b = np.zeros(len(u), np.int64)
    for i, part in enumerate(np.array_split(np.arange(len(u)), min(max_blocks, len(u)))):
        b[part] = i
    return b[inv.ravel()]


def class_averages(t, mask=None, max_blocks=100):
    """The class quantities (module docstring) of the events ``mask`` (default all) ->
    dict "<composition>/<quantity>" -> (value, jackknife error), plus "n_blocks".
    n_events entries count the events used; their error is 0."""
    cuts = t["cuts"]
    n = len(t["event"])
    keep = np.ones(n, bool) if mask is None else np.asarray(mask, bool)
    full = _stats(t, keep, cuts)
    names = list(full)
    val = np.array([full[k] for k in names], dtype=np.float64)
    sel = np.flatnonzero(keep)
    block = blocks_of(t["group"][sel], max_blocks) if len(sel) else np.zeros(0, np.int64)
    ids = np.unique(block)
    if len(ids) < 2:
        err = np.full(len(names), np.nan)
    else:
        reps = []
        for b in ids:
            k = keep.copy()
            k[sel[block == b]] = False
            r = _stats(t, k, cuts)
            reps.append([r[x] for x in names])
        reps = np.asarray(reps, dtype=np.float64)
        G = len(ids)
        with warnings.catch_warnings():         # all-NaN quantities stay NaN
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(reps, axis=0)
            err = np.sqrt((G - 1) / G * np.nansum((reps - mean) ** 2, axis=0))
        err[~np.isfinite(val)] = np.nan
    out = {k: (float(v), float(e)) for k, v, e in zip(names, val, err)}
    for k in names:
        if k.endswith("/n_events"):
            out[k] = (full[k], 0.0)
    out["n_blocks"] = (float(len(ids)), 0.0)
    return out


def nested(res):
    """class_averages() as {name: {quantity: [value, error]}} (for JSON)."""
    out = {}
    for k, (v, e) in res.items():
        a, _, b = k.partition("/")
        if b:
            out.setdefault(a, {})[b] = [v, e]
        else:
            out[a] = [v, e]
    return out


def table_arrays(t):
    """The event table as plain arrays (np.savez): the cuts as a JSON string."""
    import json

    out = {k.replace(".", "__"): np.asarray(v) for k, v in t.items() if k != "cuts"}
    out["cuts"] = np.asarray(json.dumps(t["cuts"].as_dict()))
    return out
