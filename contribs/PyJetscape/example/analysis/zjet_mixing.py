"""Event mixing for the Z-hadron correlations: CMS 2507.09307's S - B on the hadron files of
a Z+jet production (``zjet_test/mix.py`` runs it on the test and on campaign directories).

CMS measures Delta<N_ch> = S - B per Z: S the pairs of the Z with the tracks of its own
event, B those with the tracks of other Z events (mixed events).  The model also knows every
event's Z-correlated yield exactly, the paired wake <bulk_jet> - <bulk_bg> plus the jet
fragments (``zjet_wake``, ``zjet_test/compare.py``: the "factorized" result).  Mixing the
full events instead is the experiment's way: a closure test of it, and experiment-like
errors.  One mixed pair is exactly

    S_i - B_ij = (wake_i + frag_i) - (wake_j + frag_j)|Z_i + (bulk_bg_i - bulk_bg_j)|Z_i

(bulk_jet = bulk_bg + wake): the factorized result plus the bulk term, which averages to zero
when the bulk does not know the Z (3dMCGlauber ``Subtract_hard_momentum`` 0, the Z's direction
independent of the event plane) and carries the bulk's fluctuations into the errors.

Modes
-----
``all`` (:func:`all_partners`)
    Every selected Z event is a partner (``rule="all"``), the pairs weighted w_i w_j.  B then
    factorizes: the pooled lab-frame track density, shifted into each Z's frame
    (:class:`ZKernel`), with no noise of its own.  With the bulk off (``content="zc"``: the
    Z-correlated yield instead of the full event) and compare.py's pair rules (Delta phi
    every pair, ``rule="all"``; Delta y all but the self pairs, ``rule="self"``;
    ``phi_flat``) it is compare.py's result, to rounding.
``cms10`` (:func:`explicit_partners` on all Z's)
    CMS's procedure on every Z: 10 partner events from 10 different backgrounds, never the
    Z's own, drawn with probability ~ w; each partner one recorded collision (one oversample,
    ``Pool.k_one``).  S is the event's 40-sample mean.
``strict`` (:func:`strict_set` + :func:`explicit_partners`)
    A one-to-one emulation of the measurement: one Z per background, drawn ~ w inside it and
    weighted by the background's total weight W_b, so that every Z has its own fluid as in
    data; one oversample for S; 10 partners from 10 other backgrounds of that set, one
    oversample each; the error from the spread over the Z's, the experiment's statistical
    error.  For comparing the model's errors with CMS's.

Partners and errors
-------------------
The 24 events of a background share its fluid (and with --correlated its random numbers,
sample by sample): a partner from the Z's own background carries the same flow pattern and
cancels part of S's fluctuations, which no experiment can do.  So the explicit modes never
take a partner from the Z's own background, and take their partners from as many different
backgrounds.  That exclusion has a price with weighted events: the backgrounds' weights
W_b (N_coll) grow with their multiplicity N_b, and leaving a Z's own background out of its
partners shifts B by a flat level, sum_b W_b^2 (N_b - <N>) / (W^2 - sum W_b^2) for the pooled
pairs (:func:`exclusion_bias`), of order 1/(number of backgrounds): 3 per radian at 1-2 GeV
with the test's 4 backgrounds, ~10^-3 of the error with a campaign's 4,000 (estimated; data have no
weights, so no such term).  The pooled B has no noise and the own background is 1/n_bg of
it, so the ``all`` mode keeps every partner (``rule="all"``: S - B integrates to zero
exactly, the self pairs put 1/N_eff of the Z's own yield into B).  Errors: a
delete-one-block jackknife over blocks of backgrounds (the independent units); ``strict``
also the spread over its Z's.

Binning
-------
S is binned per track in the Z frame (compare.py's grid: track p_T classes, Delta y 0.4,
Delta phi pi/10).  The ``all`` mode's B shifts binned lab-frame densities (eta 0.1, phi 6
degrees) by bin centre minus y_Z (phi_Z), compare.py's convention: a smearing of +-0.05 in
Delta y and +-3 degrees in Delta phi of a smooth density.  The explicit modes shift the
partners' tracks themselves.
"""

from __future__ import annotations

import os
import warnings

import numpy as np

import zjet_wake as zw

#: the grid (zjet_test/accumulate.py's): track p_T classes, the Z frame, the lab frame
PT = np.asarray(zw.PT_CLASSES, float)
DETA = np.linspace(-4.8, 4.8, 25)                 # Delta y = eta - y_Z, 0.4
DPHI = np.linspace(-np.pi / 2, 3 * np.pi / 2, 21)  # CMS: pi / 10
ETA = np.linspace(-2.4, 2.4, 49)                  # lab eta, 0.1
PHI = np.linspace(-np.pi, np.pi, 61)              # lab phi, 6 degrees
N_PT, N_DY, N_DPHI = len(PT) - 1, len(DETA) - 1, len(DPHI) - 1
N_ETA, N_PHI = len(ETA) - 1, len(PHI) - 1
CETA = 0.5 * (ETA[1:] + ETA[:-1])
CPHI = 0.5 * (PHI[1:] + PHI[:-1])
#: what an event contributes: the full event, or its Z-correlated yield ("bulk off")
CONTENTS = ("full", "zc")

_CD = 0.5 * (DPHI[1:] + DPHI[:-1])
#: the Delta phi bin at -Delta phi, and the Z side |Delta phi| < pi/2 (compare.Model's)
MIRROR = np.array([int(np.argmin(np.abs(x - _CD)))
                   for x in (-_CD + np.pi / 2) % (2 * np.pi) - np.pi / 2])
ZSIDE = np.abs(_CD) < np.pi / 2


def _index(v, edges):
    """Bin of every value (-1 outside), numpy's convention: the last edge inclusive (as the
    reader's histograms)."""
    v = np.asarray(v, float)
    j = np.searchsorted(edges, v, side="right") - 1
    j[v == edges[-1]] = len(edges) - 2
    j[(j < 0) | (j > len(edges) - 2)] = -1
    return j


def z_hist(ipt, eta, phi, y_z, phi_z, sample=None, n_samples=1):
    """Tracks (p_T class ``ipt``, eta, phi) in a Z's frame -> counts (N_PT, N_DY, N_DPHI);
    with ``sample`` (each track's sample) per sample, (n_samples, N_PT, N_DY, N_DPHI)."""
    idy = _index(np.asarray(eta, float) - y_z, DETA)
    idp = _index(zw.wrap_dphi(np.asarray(phi, float) - phi_z), DPHI)
    ok = (idy >= 0) & (idp >= 0)
    flat = (np.asarray(ipt, np.int64)[ok] * N_DY + idy[ok]) * N_DPHI + idp[ok]
    nb = N_PT * N_DY * N_DPHI
    if sample is None:
        return np.bincount(flat, minlength=nb).reshape(N_PT, N_DY, N_DPHI).astype(float)
    h = np.bincount(np.asarray(sample, np.int64)[ok] * nb + flat, minlength=n_samples * nb)
    return h.reshape(n_samples, N_PT, N_DY, N_DPHI).astype(float)


def lab_hist(ipt, eta, phi):
    """Tracks in the lab frame -> counts (N_PT, N_ETA, N_PHI)."""
    ie, ip = _index(eta, ETA), _index(phi, PHI)
    ok = (ie >= 0) & (ip >= 0)
    flat = (np.asarray(ipt, np.int64)[ok] * N_ETA + ie[ok]) * N_PHI + ip[ok]
    return np.bincount(flat, minlength=N_PT * N_ETA * N_PHI).reshape(
        N_PT, N_ETA, N_PHI).astype(float)


def observables(D):
    """S - B per Z, (..., N_PT, N_DY, N_DPHI) -> CMS's (dphi (..., N_PT, N_DPHI) per radian,
    all Delta y; dy (..., N_PT, N_DY) per unit Delta y, Z side): +- merged and scaled by
    0.5, as compare.Model."""
    hp = D.sum(axis=-2)
    dphi = 0.5 * (hp + hp[..., MIRROR]) / np.diff(DPHI)
    s = D[..., ZSIDE].sum(axis=-1)
    dy = 0.5 * (s + s[..., ::-1]) / np.diff(DETA)
    return dphi, dy


# ── the map step: one production file's pool ─────────────────────────────────────
def _tracks(ev, info, mask):
    """The tracks of an EventHadrons: (p_T class, eta, phi, sample) inside the classes."""
    m = mask(ev, info)
    ipt = _index(ev.pt[m], PT)
    k = ipt >= 0
    return ipt[k], ev.eta[m][k], ev.phi[m][k], ev.sample[m][k]


def file_pool(stem, local_events, global_events, weights, backgrounds, *, seed=0,
              eta_max=2.4):
    """The pool of one production file (the map step), for its selected events
    ``local_events`` (their campaign indices ``global_events``, weights ``weights``,
    background ids ``backgrounds``) -> dict of arrays, :meth:`Pool.from_parts` merges them.

    Per event: S in its Z frame for the full event (the per-sample mean, ``S_full``; its
    number of samples ``n_samples``) and one oversample ``k_one`` (``S_one``; the
    fragmentation ``k_one mod n_frag`` as ``JetEvents.jet_event``), and that oversample's
    tracks (the partner content of the explicit modes).  ``k_one`` is
    ``zjet_wake.key_index(event_key(seed, the file's name, the event in the file), n)``: the
    same when files are added to a campaign, and in analysis_root/zjet's C++ pools.  Per
    background, summed over its selected events: w S, w^2 S and the lab-frame density w rho, for the full event and the Z-correlated yield (bulk_jet -
    bulk_bg, per sample, + fragments).  Vacuum files (jet_frag only): the event is its own
    background and the two contents are the same, stored once (``Pool`` aliases them)."""
    from jetscape.hadrons_h5 import EventHadrons, HadronFile, HadronFileReader, Hadrons

    r = HadronFileReader([stem])
    tags = r.tags(0)
    bulk = "bulk_jet" in tags and "bulk_bg" in tags
    mask = zw.track_mask(eta_max)
    local_events = np.asarray(local_events, np.int64)
    infos = [r.event_info(int(e)) for e in local_events]

    def load(tag, units):
        path = f"{stem}_hadrons_{tag}.h5"
        with HadronFile(path) as hf:
            pos = sorted({hf.unit_index(int(u)) for u in units})
        return Hadrons.from_h5(path, units=np.array(pos, dtype=np.int64))

    shp_z, shp_l = (N_PT, N_DY, N_DPHI), (N_PT, N_ETA, N_PHI)
    out = {k: [] for k in ("event", "bg", "w", "y", "phi", "k_one", "n_samples", "S_full",
                           "S_one", "trk_n", "trk_pt", "trk_eta", "trk_phi")}
    per_bg = {}
    contents = CONTENTS if bulk else ("full",)
    name = os.path.basename(stem)
    if len(local_events) == 0:
        return _pack(out, per_bg, contents)
    hfr = load("jet_frag", local_events)
    if bulk:
        hj = load("bulk_jet", local_events)
        hb = load("bulk_bg", {i.bg_unit for i in infos})
    bg_cache = {}
    for e, g, w, b, info in zip(local_events, global_events, weights, backgrounds, infos):
        z = info.z
        y, ph = float(z["z_y"]), float(z["z_phi"])
        F = EventHadrons(hfr, hfr.unit_index(int(e)))
        nf = F.n_samples
        tf = _tracks(F, info, mask)
        if bulk:
            J = EventHadrons(hj, hj.unit_index(int(e)))
            K = J.n_samples
            if info.bg_unit not in bg_cache:
                B = EventHadrons(hb, hb.unit_index(info.bg_unit))
                tb = _tracks(B, info, mask)        # the Z leptons are not in the bulk
                bg_cache[info.bg_unit] = (B.n_samples, tb, lab_hist(*tb[:3]))
            Kb, tb, Rb = bg_cache[info.bg_unit]
            if K == 0 or nf == 0 or Kb != K:
                warnings.warn(f"{stem} event {e}: {K} jet / {Kb} background samples, {nf} "
                              "fragmentations: left out", RuntimeWarning, stacklevel=2)
                continue
            tj = _tracks(J, info, mask)
        elif nf == 0:
            continue
        n_s = K if bulk else nf
        k = zw.key_index(zw.event_key(seed, name, int(e)), n_s)
        SF, RF = z_hist(*tf[:3], y, ph) / nf, lab_hist(*tf[:3]) / nf
        fk = tf[3] == (k % nf)
        if bulk:
            SJ, RJ = z_hist(*tj[:3], y, ph) / K, lab_hist(*tj[:3]) / K
            SB, RB = z_hist(*tb[:3], y, ph) / K, Rb / K
            S_full, S_zc = SJ + SF, SJ - SB + SF
            R_full, R_zc = RJ + RF, RJ - RB + RF
            jk = tj[3] == k
            one = tuple(np.concatenate([a[jk], c[fk]]) for a, c in zip(tj[:3], tf[:3]))
        else:
            S_full = S_zc = SF
            R_full = R_zc = RF
            one = tuple(a[fk] for a in tf[:3])
        S_one = z_hist(*one, y, ph)
        for key, v in (("event", g), ("bg", b), ("w", w), ("y", y), ("phi", ph),
                       ("k_one", k), ("n_samples", n_s), ("S_full", S_full), ("S_one", S_one),
                       ("trk_n", len(one[0])), ("trk_pt", one[0]), ("trk_eta", one[1]),
                       ("trk_phi", one[2])):
            out[key].append(v)
        s = per_bg.setdefault(int(b), {"W": 0.0, "W2": 0.0,
                                       **{f"{q}_{c}": np.zeros(shp_z) for q in ("SW", "T")
                                          for c in contents},
                                       **{f"R_{c}": np.zeros(shp_l) for c in contents}})
        s["W"] += w
        s["W2"] += w * w
        for c, S_, R_ in (("full", S_full, R_full), ("zc", S_zc, R_zc))[:len(contents)]:
            s[f"SW_{c}"] += w * S_
            s[f"T_{c}"] += w * w * S_
            s[f"R_{c}"] += w * R_
    return _pack(out, per_bg, contents)


def _pack(out, per_bg, contents):
    shp_z, shp_l = (N_PT, N_DY, N_DPHI), (N_PT, N_ETA, N_PHI)
    n = len(out["event"])
    d = {"event": np.asarray(out["event"], np.int64), "bg": np.asarray(out["bg"], np.int64),
         "w": np.asarray(out["w"], float), "y": np.asarray(out["y"], float),
         "phi": np.asarray(out["phi"], float), "k_one": np.asarray(out["k_one"], np.int64),
         "n_samples": np.asarray(out["n_samples"], np.int64),
         "S_full": np.asarray(out["S_full"], np.float32).reshape(n, *shp_z),
         "S_one": np.asarray(out["S_one"], np.float32).reshape(n, *shp_z),
         "trk_n": np.asarray(out["trk_n"], np.int64),
         "trk_pt": np.concatenate(out["trk_pt"]).astype(np.int8) if n else np.zeros(0, np.int8),
         "trk_eta": (np.concatenate(out["trk_eta"]).astype(np.float32) if n
                     else np.zeros(0, np.float32)),
         "trk_phi": (np.concatenate(out["trk_phi"]).astype(np.float32) if n
                     else np.zeros(0, np.float32))}
    ids = sorted(per_bg)
    d["bg_id"] = np.asarray(ids, np.int64)
    d["bg_W"] = np.array([per_bg[b]["W"] for b in ids], float)
    d["bg_W2"] = np.array([per_bg[b]["W2"] for b in ids], float)
    for c in contents:
        for q, shp in (("SW", shp_z), ("T", shp_z), ("R", shp_l)):
            d[f"bg_{q}_{c}"] = (np.array([per_bg[b][f"{q}_{c}"] for b in ids]) if ids
                                else np.zeros((0, *shp)))
    return d


def read_root_pool(path):
    """A pool written by analysis_root/zjet/zhadron_pools (one ROOT file per production file)
    -> the dict of :func:`file_pool`.  Trees ``pool_events`` (event, bg, w, y, phi, k_one,
    n_samples, S_full[N_PT*N_DY*N_DPHI], S_one[...], trk_pt/eta/phi[n]) and ``pool_bg``
    (bg_id, W, W2, SW_<c>, T_<c>, R_<c> for the contents it has)."""
    import awkward as ak
    import uproot

    shp_z, shp_l = (N_PT, N_DY, N_DPHI), (N_PT, N_ETA, N_PHI)
    with uproot.open(path) as f:
        ev = f["pool_events"].arrays(library="ak")
        bg = f["pool_bg"].arrays(library="ak")
    n = len(ev)

    def col(a, k, dt):
        return np.asarray(ak.to_numpy(a[k]), dt)

    d = {"event": col(ev, "event", np.int64), "bg": col(ev, "bg", np.int64),
         "w": col(ev, "w", float), "y": col(ev, "y", float), "phi": col(ev, "phi", float),
         "k_one": col(ev, "k_one", np.int64), "n_samples": col(ev, "n_samples", np.int64),
         "S_full": col(ev, "S_full", np.float32).reshape(n, *shp_z),
         "S_one": col(ev, "S_one", np.float32).reshape(n, *shp_z),
         "trk_n": np.asarray(ak.to_numpy(ak.num(ev["trk_pt"])), np.int64),
         "trk_pt": np.asarray(ak.to_numpy(ak.flatten(ev["trk_pt"])), np.int8),
         "trk_eta": np.asarray(ak.to_numpy(ak.flatten(ev["trk_eta"])), np.float32),
         "trk_phi": np.asarray(ak.to_numpy(ak.flatten(ev["trk_phi"])), np.float32),
         "bg_id": col(bg, "bg_id", np.int64), "bg_W": col(bg, "W", float),
         "bg_W2": col(bg, "W2", float)}
    nb = len(bg)
    for c in CONTENTS:
        if f"SW_{c}" in bg.fields:
            for q, shp in (("SW", shp_z), ("T", shp_z), ("R", shp_l)):
                d[f"bg_{q}_{c}"] = col(bg, f"{q}_{c}", float).reshape(nb, *shp)
    return d


# ── the reduce step's input ─────────────────────────────────────────────────────
class Pool:
    """The per-file pools of a campaign merged: per selected event (``event``, ``bg``,
    ``w``, ``y``, ``phi``, ``k_one``, ``S_full``, ``S_one``, its one-sample tracks
    ``tracks(i)``) and per background (``bg_W``, ``bg_W2``, ``bg_SW[c]``, ``bg_T[c]``,
    ``bg_R[c]`` for c in CONTENTS).  ``zb``: each event's background as an index into the
    per-background arrays.  Vacuum pools store one content: "zc" is then "full"."""

    def __init__(self, d):
        self.__dict__.update({k: v for k, v in d.items() if not k.startswith("bg_")})
        order = np.argsort(d["bg_id"], kind="stable")
        self.bg_id = d["bg_id"][order]
        if len(np.unique(self.bg_id)) != len(self.bg_id):
            raise ValueError("a background occurs in more than one pool part")
        self.bg_W, self.bg_W2 = d["bg_W"][order], d["bg_W2"][order]

        def per_bg(q):
            full = d[f"bg_{q}_full"][order]
            zc = d[f"bg_{q}_zc"][order] if f"bg_{q}_zc" in d else full
            return {"full": full, "zc": zc}

        self.bg_SW, self.bg_T, self.bg_R = per_bg("SW"), per_bg("T"), per_bg("R")
        self.zb = np.searchsorted(self.bg_id, self.bg)
        self.trk_off = np.concatenate([[0], np.cumsum(self.trk_n)])

    @classmethod
    def from_parts(cls, parts):
        parts = [dict(p) for p in parts]
        keys = set(parts[0])
        if any(set(p) != keys for p in parts):
            raise ValueError("pool parts with different contents (vacuum and Pb+Pb mixed?)")
        return cls({k: np.concatenate([p[k] for p in parts]) for k in parts[0]})

    @classmethod
    def load(cls, paths):
        """The pools of ``paths``: ``.npz`` (``file_pool``) or ``.root``
        (analysis_root/zjet's ``zhadron_pools``, :func:`read_root_pool`)."""
        parts = []
        for p in paths:
            if str(p).endswith(".root"):
                parts.append(read_root_pool(p))
                continue
            with np.load(p) as f:
                parts.append({k: f[k] for k in f.files})
        return cls.from_parts(parts)

    @property
    def n(self):
        return len(self.event)

    @property
    def n_bg(self):
        return len(self.bg_id)

    def tracks(self, i):
        a, b = int(self.trk_off[i]), int(self.trk_off[i + 1])
        return self.trk_pt[a:b], self.trk_eta[a:b], self.trk_phi[a:b]

    def blocks(self, n):
        """The backgrounds in ``n`` contiguous blocks (file order) for the jackknife."""
        return [b for b in np.array_split(np.arange(self.n_bg), min(n, self.n_bg)) if len(b)]


def n_eff(w):
    """Kish's effective number of a weighted sample."""
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / np.sum(w * w)) if len(w) else 0.0


# ── the all-partner mixing ───────────────────────────────────────────────────────
def _cuts(values, period=None):
    v = np.round(np.ravel(values) if period is None else np.mod(np.ravel(values), period), 9)
    return np.unique(v)


# the shift of a lab bin into the Z frame changes only where y_Z (phi_Z) crosses a lab bin
# centre minus a Z-frame bin edge: between those cuts every Z shifts the same way
_YCUT = _cuts(CETA[:, None] - DETA[None, :])
_PCUT = _cuts(CPHI[:, None] - DPHI[None, :], 2 * np.pi)


def _dy_of(y):
    """Delta y bin of every lab eta bin for a Z at rapidity y (compare.Model's rule)."""
    return np.clip(np.searchsorted(DETA, CETA - y) - 1, 0, N_DY - 1)


def _dphi_of(phi):
    return np.clip(np.searchsorted(DPHI, zw.wrap_dphi(CPHI - phi), side="right") - 1,
                   0, N_DPHI - 1)


def _y_cell(y):
    return np.searchsorted(_YCUT, np.asarray(y, float))


def _y_rep(c):
    if c == 0:
        return _YCUT[0] - 1.0
    if c == len(_YCUT):
        return _YCUT[-1] + 1.0
    return 0.5 * (_YCUT[c - 1] + _YCUT[c])


def _phi_cell(phi):
    c = np.searchsorted(_PCUT, np.mod(np.asarray(phi, float), 2 * np.pi))
    return np.where(c == len(_PCUT), 0, c)


def _phi_rep(c):
    if c == 0:
        return np.mod(0.5 * (_PCUT[-1] + _PCUT[0] + 2 * np.pi), 2 * np.pi)
    return 0.5 * (_PCUT[c - 1] + _PCUT[c])


def _onehot(idx, n):
    m = np.zeros((n, len(idx)))
    m[idx, np.arange(len(idx))] = 1.0
    return m


#: per phi cell, the lab phi bin -> Delta phi bin map as a matrix (N_DPHI, N_PHI)
_P_PHI = np.stack([_onehot(_dphi_of(_phi_rep(c)), N_DPHI) for c in range(len(_PCUT))])


class ZKernel:
    """K(rho) = sum_i w_i shift_i(rho) for a set of Z's (y_i, phi_i, w_i): a lab-frame
    density rho (N_PT, N_ETA, N_PHI) seen from every Z (each lab bin centre moved by -y_i,
    -phi_i into its Z-frame bin), summed -> (N_PT, N_DY, N_DPHI).  The Z's are summed in the
    cells inside which they all shift alike, so it costs the same for 10 or 10^5 Z's and
    equals the Z-by-Z sum."""

    def __init__(self, y, phi, w):
        y, phi, w = (np.asarray(a, float) for a in (y, phi, w))
        cy, inv = np.unique(_y_cell(y), return_inverse=True)
        wc = np.zeros((len(cy), len(_PCUT)))
        np.add.at(wc, (inv.ravel(), _phi_cell(phi)), w)
        self.wy = wc.sum(axis=1)
        self.py = (np.stack([_onehot(_dy_of(_y_rep(c)), N_DY) for c in cy]) if len(cy)
                   else np.zeros((0, N_DY, N_ETA)))
        self.a = np.einsum("cf,fal->cal", wc, _P_PHI)

    def __call__(self, rho, phi_flat=False):
        """``phi_flat``: rho integrated over phi and spread evenly over Delta phi (B flat in
        Delta phi by construction, compare.py)."""
        if not len(self.wy):
            return np.zeros((N_PT, N_DY, N_DPHI))
        if phi_flat:
            out = np.einsum("c,cdk,pk->pd", self.wy, self.py, rho.sum(axis=-1)) / N_DPHI
            return np.repeat(out[:, :, None], N_DPHI, axis=2)
        return np.einsum("cdk,pkl,cal->pda", self.py, rho, self.a, optimize=True)


def shift_sum(rho, y, phi, w, phi_flat=False):
    """The Z-by-Z sum ZKernel replaces (for the tests)."""
    out = np.zeros((N_PT, N_DY, N_DPHI))
    for yi, pi, wi in zip(y, phi, w):
        dy, dp = _dy_of(yi), _dphi_of(pi)
        for p in range(N_PT):
            if phi_flat:
                out[p] += wi * np.bincount(dy, rho[p].sum(-1), N_DY)[:, None] / N_DPHI
            else:
                flat = (dy[:, None] * N_DPHI + dp[None, :]).ravel()
                out[p] += wi * np.bincount(flat, rho[p].ravel(), N_DY * N_DPHI).reshape(
                    N_DY, N_DPHI)
    return out


RULES = ("all", "self", "background")


def all_partners(pool, content="full", rule="all", phi_flat=False, blocks=()):
    """S - B per Z, (N_PT, N_DY, N_DPHI), every selected event a partner except by ``rule``:
    "all" none excluded (B is then the pooled density, self pairs included), "self" the Z's
    own event (its exact self pair w_i^2 S_i, compare.py's rule), "background" the Z's whole
    background (biased by :func:`exclusion_bias`).  Pairs weigh w_i w_j:

        B = sum_i w_i sum_{j partner of i} w_j shift_i(rho_j) / sum_i w_i sum_j w_j.

    ``blocks``: lists of background indices; for each, the result without those
    backgrounds (their Z's and their partners) for the jackknife.  -> (D, [D_L])."""
    if rule not in RULES:
        raise ValueError(f"rule {rule!r}: one of {RULES}")
    SW, T, R = pool.bg_SW[content], pool.bg_T[content], pool.bg_R[content]
    Wb, W2b = pool.bg_W, pool.bg_W2
    k_tot = ZKernel(pool.y, pool.phi, pool.w)
    r_tot = R.sum(axis=0)
    kk_tot = k_tot(r_tot, phi_flat)
    sw_tot, t_tot = SW.sum(axis=0), T.sum(axis=0)
    if rule == "background":
        # sum_{i in b} w_i shift_i(rho_b): the pairs inside each background, taken out
        members = np.split(np.argsort(pool.zb, kind="stable"),
                           np.cumsum(np.bincount(pool.zb, minlength=pool.n_bg))[:-1])
        q = np.array([ZKernel(pool.y[m], pool.phi[m], pool.w[m])(R[b], phi_flat)
                      for b, m in enumerate(members)])
        q_tot = q.sum(axis=0)

    def flat(t):
        return np.repeat(t.sum(axis=-1, keepdims=True) / N_DPHI, N_DPHI, axis=-1)

    def result(drop):
        drop = np.asarray(drop, np.int64)
        W = Wb.sum() - Wb[drop].sum()
        S = (sw_tot - SW[drop].sum(axis=0)) / W
        kk = kk_tot
        if len(drop):
            inl = np.isin(pool.zb, drop)
            k_l = ZKernel(pool.y[inl], pool.phi[inl], pool.w[inl])
            r_l = R[drop].sum(axis=0)
            kk = kk_tot - k_tot(r_l, phi_flat) - k_l(r_tot, phi_flat) + k_l(r_l, phi_flat)
        if rule == "all":
            num, den = kk, W * W
        elif rule == "self":
            t = t_tot - T[drop].sum(axis=0)
            num = kk - (flat(t) if phi_flat else t)
            den = W * W - (W2b.sum() - W2b[drop].sum())
        else:
            num = kk - (q_tot - q[drop].sum(axis=0))
            den = W * W - (np.sum(Wb ** 2) - np.sum(Wb[drop] ** 2))
        if den <= 0:
            raise ValueError("no partner pairs: the rule leaves no partners (a single "
                             "background?)")
        return S - num / den

    return result(()), [result(b) for b in blocks]


def exclusion_bias(pool, content="full"):
    """The flat level (per Z, per radian of Delta phi, all Delta y) by which leaving the
    Z's own background out of its partners shifts S - B, from the backgrounds' weights W_b
    and multiplicities N_b (tracks per Z of ``content``, per p_T class) -> dict "pooled"
    (:func:`all_partners` rule "background", pairs weighted w_i w_j: sum_b W_b^2 (N_b - <N>)
    / (W^2 - sum W_b^2)) and "explicit" (partners drawn ~ W_c from the other backgrounds,
    S - B weighted per Z: sum_b W_b (N_b - <N>) / (W - W_b)), each (N_PT,)."""
    Wb = pool.bg_W
    W = Wb.sum()
    Nb = pool.bg_R[content].sum(axis=(2, 3)) / Wb[:, None]
    dev = Nb - (Wb[:, None] * Nb).sum(axis=0) / W
    pooled = (Wb[:, None] ** 2 * dev).sum(axis=0) / (W * W - np.sum(Wb ** 2))
    explicit = (Wb[:, None] * dev / (W - Wb)[:, None]).sum(axis=0)
    return {"pooled": pooled / (2 * np.pi), "explicit": explicit / (2 * np.pi)}


# ── the noise budget per Z ───────────────────────────────────────────────────────
def _shift_one(rho, y, phi):
    """shift_i(rho) for one Z (ZKernel's sum for a single Z)."""
    flat = (_dy_of(y)[:, None] * N_DPHI + _dphi_of(phi)[None, :]).ravel()
    return np.stack([np.bincount(flat, rho[p].ravel(), N_DY * N_DPHI)
                     for p in range(N_PT)]).reshape(N_PT, N_DY, N_DPHI)


#: the pieces of :func:`per_z_noise`
NOISE_PIECES = ("pooled", "shape", "poisson")


def per_z_noise(pool):
    """The pieces of S - B's noise per Z, as CMS's observables per Z -> {piece: (dphi (n,
    N_PT, N_DPHI), dy (n, N_PT, N_DY))}:

    "pooled"   S_full,i - B_i, B the pooled density (every event a partner, no noise of its
               own): the fluid's fluctuations (its multiplicity and flow, seen from the Z) plus
               the Poisson noise of the n_samples mean;
    "shape"    the same with B scaled to the event's own multiplicity, B_i n_i / <n> per p_T
               class: without each event's flat level;
    "poisson"  S_one,i - S_full,i: the Poisson noise of one oversample (iSS sampling and
               fragmentation).

    Their weighted spreads over the Z's go into :func:`strict_noise`."""
    W = pool.w.sum()
    r = pool.bg_R["full"].sum(axis=0)
    nbar = r.sum(axis=(1, 2)) / W                   # tracks per Z, per p_T class
    out = {k: ([], []) for k in NOISE_PIECES}
    for i in range(pool.n):
        S = pool.S_full[i].astype(float)
        B = _shift_one(r, pool.y[i], pool.phi[i]) / W
        scale = np.divide(S.sum(axis=(1, 2)), nbar, out=np.ones(N_PT), where=nbar > 0)
        for key, D in (("pooled", S - B), ("shape", S - B * scale[:, None, None]),
                       ("poisson", pool.S_one[i].astype(float) - S)):
            dphi, dy = observables(D)
            out[key][0].append(dphi)
            out[key][1].append(dy)
    return {k: (np.array(a), np.array(b)) for k, (a, b) in out.items()}


def strict_noise(sd, n_samples, n_mix=10, efficiency=1.0):
    """The strict emulation's noise per Z, predicted from :func:`per_z_noise`'s weighted
    spreads ``sd`` ({piece: array}): the fluid term sqrt(pooled^2 - poisson^2 / n_samples)
    (the pooled S is an n_samples mean; with fewer fragmentations than iSS samples the
    fragments' part is underestimated), then S from one oversample and ``n_mix`` partners
    like it, sqrt((fluid^2 + poisson^2 / efficiency) (1 + 1 / n_mix)); ``efficiency`` scales
    the Poisson part as efficiency-corrected tracks do.  -> {"fluid", "fluid_shape", "full"
    (with each event's flat level), "shape" (without it), "poisson" (Poisson only)}."""
    p2 = sd["poisson"] ** 2
    fluid = np.sqrt(np.clip(sd["pooled"] ** 2 - p2 / n_samples, 0, None))
    fluid_s = np.sqrt(np.clip(sd["shape"] ** 2 - p2 / n_samples, 0, None))
    f = 1.0 + 1.0 / n_mix
    return {"fluid": fluid, "fluid_shape": fluid_s,
            "full": np.sqrt((fluid ** 2 + p2 / efficiency) * f),
            "shape": np.sqrt((fluid_s ** 2 + p2 / efficiency) * f),
            "poisson": np.sqrt(p2 / efficiency * f)}


# ── the explicit (CMS-style) mixing ──────────────────────────────────────────────
def strict_set(pool, seed=0):
    """One selected event per background, drawn with probability w_i / W_b inside it, and
    weighted by W_b: every Z its own fluid, as in data, and an unbiased estimate of the
    w-weighted average.  -> (pool indices, weights)."""
    rng = np.random.default_rng([int(seed), 0x5717C7])
    order = np.argsort(pool.zb, kind="stable")
    members = np.split(order, np.cumsum(np.bincount(pool.zb, minlength=pool.n_bg))[:-1])
    pick = np.array([rng.choice(m, p=pool.w[m] / pool.w[m].sum()) for m in members])
    return pick, pool.bg_W.copy()


def explicit_partners(pool, z_index, partner_index, partner_weight, *, n_mix=10, s="full",
                      seed=0):
    """CMS-style mixing: every Z (pool indices ``z_index``) with ``n_mix`` partner events
    from as many different backgrounds, never its own.  The backgrounds are drawn without
    replacement with probability ~ the summed ``partner_weight`` of their candidates
    (``partner_index``), then one candidate ~ its weight; a partner contributes its one
    recorded oversample (``Pool.tracks``).  ``s``: "full" (S the 40-sample mean) or "one" (S
    one oversample, as one recorded collision).  With fewer backgrounds available than
    ``n_mix``, all of them (a warning).  -> dict dphi (n, N_PT, N_DPHI), dy (n, N_PT, N_DY)
    per Z, n_mix (n,) the partners used, partners (list of their pool indices)."""
    rng = np.random.default_rng([int(seed), 0xC3510])
    partner_index = np.asarray(partner_index, np.int64)
    partner_weight = np.asarray(partner_weight, float)
    cb = pool.zb[partner_index]
    bgs, inv = np.unique(cb, return_inverse=True)
    inv = inv.ravel()
    cand = [partner_index[inv == k] for k in range(len(bgs))]
    cw = [partner_weight[inv == k] for k in range(len(bgs))]
    wsum = np.array([c.sum() for c in cw])
    S_all = pool.S_full if s == "full" else pool.S_one
    n = len(z_index)
    dphi = np.zeros((n, N_PT, N_DPHI))
    dy = np.zeros((n, N_PT, N_DY))
    used = np.zeros(n, np.int64)
    partners = []
    short = 0
    for a, i in enumerate(z_index):
        own = np.flatnonzero(bgs == pool.zb[i])
        p = wsum.copy()
        p[own] = 0.0
        avail = int(np.count_nonzero(p))
        m = min(n_mix, avail)
        short += m < n_mix
        chosen = []
        if m:
            p = p / p.sum()
            while len(chosen) < m:                     # successive sampling ~ p
                for d in rng.choice(len(bgs), size=2 * m, p=p):
                    if d not in chosen:
                        chosen.append(int(d))
                        if len(chosen) == m:
                            break
        js = [int(rng.choice(cand[d], p=cw[d] / cw[d].sum())) for d in chosen]
        partners.append(js)
        used[a] = m
        if m:
            trk = [pool.tracks(j) for j in js]
            B = z_hist(np.concatenate([t[0] for t in trk]), np.concatenate([t[1] for t in trk]),
                       np.concatenate([t[2] for t in trk]), pool.y[i], pool.phi[i]) / m
        else:
            B = np.full((N_PT, N_DY, N_DPHI), np.nan)
        dphi[a], dy[a] = observables(S_all[i].astype(float) - B)
    if short:
        warnings.warn(f"{short} of {n} Z's had fewer than {n_mix} other backgrounds to mix "
                      "with: all of them used", RuntimeWarning, stacklevel=2)
    return {"dphi": dphi, "dy": dy, "n_mix": used, "partners": partners}


def weighted_mean(x, w, groups=None, blocks=()):
    """The weighted mean over the first axis of ``x`` -> (mean, spread error, jackknife
    error over ``blocks``: lists of ``groups`` values left out together; None without)."""
    w = np.asarray(w, float)
    ok = np.all(np.isfinite(x.reshape(len(x), -1)), axis=1)
    x, w = x[ok], w[ok]
    sw = w.sum()
    mean = np.tensordot(w, x, axes=1) / sw
    spread = np.sqrt(np.tensordot(w * w, (x - mean) ** 2, axes=1)) / sw
    jk = None
    if len(blocks):
        g = np.asarray(groups)[ok]
        reps = []
        for b in blocks:
            keep = ~np.isin(g, b)
            reps.append(np.tensordot(w[keep], x[keep], axes=1) / w[keep].sum())
        jk = jackknife(np.array(reps))
    return mean, spread, jk


def jackknife(reps):
    """Delete-one-block jackknife error of replicates (G, ...)."""
    reps = np.asarray(reps, float)
    G = len(reps)
    return np.sqrt((G - 1) / G * np.sum((reps - reps.mean(axis=0)) ** 2, axis=0))
