"""Event mixing for the Z-hadron correlations: CMS 2507.09307's S - B on the hadron files of
a Z+jet production (``mix.py`` runs it on the test and on campaign directories).

CMS measures Delta<N_ch> = S - B per Z: S the pairs of the Z with the tracks of its own
event, B those with the tracks of other Z events (mixed events).  The model also knows every
event's Z-correlated yield exactly, the paired wake <bulk_jet> - <bulk_bg> plus the jet
fragments (``zjet_wake``, ``test/compare.py``: the "factorized" result).  Mixing the
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

Extra hard scatterings
----------------------
With an extra_frag file (docs/Plans/PLAN_multi_hard.md) the extras belong to the event, not to
the trigger: the full event (S_full, S_one and its tracks, Y_full, Y_one, L_full, the lab
density R_full) is bulk_jet + jet_frag + extra_frag, the extras' fragmentation k mod their
count with oversample k.  The Z-correlated yield (S_zc, Y_zc, L_zc, R_zc) stays the trigger's,
bulk_jet - bulk_bg + jet_frag, so B, the bulk alone (Y_full - Y_zc), becomes bulk_bg +
extra_frag, per event (the extras are drawn per jet event).  The pools then carry
``bgw_R_extra`` (the extras' lab density, summed per background and window), which
:func:`assemble` takes out of R_zc.  ``extras=False`` (mix.py --no-extras) builds the pools as
for files without extras.

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
import sys
import warnings

import numpy as np

import zjet_wake as zw

#: the pools' layout: 4 adds the per-event keys, the dijet proxy's contents (Y_*, L_*), the
#: file record (fi_*) and the unweighted per-(background, window) sums (bgw_*, bgr_*) in
#: place of the weighted per-background sums of 3
POOL_VERSION = 4
#: the grid (test/accumulate.py's): track p_T classes, the Z frame, the lab frame
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
#: the dijet proxy's per-event grid (docs/Plans/PLAN_dijet_deta_wake.md A3, D5): Delta y =
#: eta - y_Z in 0.1, from which CMS HIN-25-012's 0.2 bins (edges at odd tenths), its sideband
#: and signal ranges and the 0.4 grid above all rebin exactly; per side of the Z (0: the Z
#: side |Delta phi| < pi/2, the DPHI bins of ZSIDE; 1: the away side)
DY_FINE = np.linspace(-4.8, 4.8, 97)
N_DYF, N_SIDE = len(DY_FINE) - 1, 2
#: the per-event contents of the dijet proxy (Y_*: Z frame, DY_FINE; L_*: lab eta, ETA)
Y_CONTENTS = ("full", "one", "zc", "frag")
L_CONTENTS = ("full", "zc", "frag")
#: the grid's edges by name: every pool stores them (``grid_<name>``; the C++ pools in the tree
#: ``pool_grid``), and :func:`check_grid` refuses pools binned on other edges
GRID = {"pt": PT, "dy": DETA, "dphi": DPHI, "eta": ETA, "phi": PHI, "dyf": DY_FINE}
#: the edges of the pools written before the edges were stored (versions 3 and 4 until
#: 2026-10-29), fixed here: should GRID change, those pools are refused, not misread
GRID_UNSTORED = {"pt": np.array([1.0, 2.0, 4.0, 10.0]), "dy": np.linspace(-4.8, 4.8, 25),
                 "dphi": np.linspace(-np.pi / 2, 3 * np.pi / 2, 21),
                 "eta": np.linspace(-2.4, 2.4, 49), "phi": np.linspace(-np.pi, np.pi, 61),
                 "dyf": np.linspace(-4.8, 4.8, 97)}

_CD = 0.5 * (DPHI[1:] + DPHI[:-1])
#: the Delta phi bin at -Delta phi, and the Z side |Delta phi| < pi/2 (cms2507.Model's)
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


def _side(phi, phi_z):
    """0 on the Z side (|Delta phi| < pi/2: the DPHI bins of ZSIDE), 1 on the away side."""
    idp = _index(zw.wrap_dphi(np.asarray(phi, float) - phi_z), DPHI)
    return (idp >= N_DPHI // 2).astype(np.int64)


def side_hist(ipt, eta, phi, y_z, phi_z):
    """Tracks in a Z's frame -> counts (N_PT, N_SIDE, N_DYF): Delta y = eta - y_Z in 0.1, per
    side of the Z (the dijet proxy's per-event grid)."""
    idy = _index(np.asarray(eta, float) - y_z, DY_FINE)
    ok = idy >= 0
    flat = ((np.asarray(ipt, np.int64)[ok] * N_SIDE + _side(np.asarray(phi)[ok], phi_z))
            * N_DYF + idy[ok])
    return np.bincount(flat, minlength=N_PT * N_SIDE * N_DYF).reshape(
        N_PT, N_SIDE, N_DYF).astype(float)


def lab_side_hist(ipt, eta, phi, phi_z):
    """Tracks -> counts (N_PT, N_SIDE, N_ETA): lab eta (0.1) per side of the Z (the Yang-Wang
    form of the dijet proxy)."""
    ie = _index(eta, ETA)
    ok = ie >= 0
    flat = ((np.asarray(ipt, np.int64)[ok] * N_SIDE + _side(np.asarray(phi)[ok], phi_z))
            * N_ETA + ie[ok])
    return np.bincount(flat, minlength=N_PT * N_SIDE * N_ETA).reshape(
        N_PT, N_SIDE, N_ETA).astype(float)


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
              eta_max=2.4, record=None, extras=True):
    """The pool of one production file (the map step), for its selected events
    ``local_events`` (their campaign indices ``global_events``, weights ``weights``,
    background ids ``backgrounds``: those of the directory it was built from, kept for the
    record) -> dict of arrays; :meth:`Pool.load` combines the pools of any set of files.

    Per event: S in its Z frame for the full event (the per-sample mean, ``S_full``; its
    number of samples ``n_samples``) and one oversample ``k_one`` (``S_one``; the
    fragmentation ``k_one mod n_frag`` as ``JetEvents.jet_event``), and that oversample's
    tracks (the partner content of the explicit modes).  ``k_one`` is
    ``zjet_wake.key_index(event_key(seed, the file's name, the event in the file), n)``: the
    same when files are added to a campaign, and in analysis_root/zjet's C++ pools.  The dijet
    proxy's contents (PLAN_dijet_deta_wake A3) on the fine grid, per side of the Z: ``Y_full``,
    ``Y_one``, ``Y_zc`` (the Z-correlated yield, bulk_jet - bulk_bg per sample + fragments),
    ``Y_frag`` (the fragments), and the lab-eta densities ``L_full``, ``L_zc``, ``L_frag``.
    Its keys: ``local_event``, ``pthat_bin``, ``n_coll``, ``bg_unit``, and the file's record
    (``fi_*``, ``jetcorr.mapio``; ``record``: ``jetcorr.weights.file_record``'s, else made
    here).  Per (hydro background, pTHat window), unweighted, over its selected events
    (``bgw_*``): their number, N_coll, the sums of S and of the lab density for the full event
    and of S for the Z-correlated yield; per background its bulk_bg lab density per sample
    (``bgr_*``).  The weights (window cross section / events x N_coll) factorize per window and
    background, so :meth:`Pool.load` rebuilds the weighted sums of any combination exactly
    (to rounding).  Vacuum files (jet_frag only): the event is its own background and the
    contents are the same, stored once (``Pool`` aliases them).  With extra hard scatterings
    (an extra_frag file and ``extras``) the full event and its one oversample include the
    extras' fragments, the Z-correlated contents do not, and ``bgw_R_extra`` holds the extras'
    lab density per (background, window) (module docstring)."""
    from jetscape.hadrons_h5 import EventHadrons, HadronFile, HadronFileReader, Hadrons

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from jetcorr import weights as jw

    r = HadronFileReader([stem])
    tags = r.tags(0)
    bulk = "bulk_jet" in tags and "bulk_bg" in tags
    xtra = bool(extras) and bulk and "extra_frag" in tags      # extra hard scatterings
    mask = zw.track_mask(eta_max)
    local_events = np.asarray(local_events, np.int64)
    infos = [r.event_info(int(e)) for e in local_events]
    rec = record if record is not None else jw.file_record(r, 0)

    def load(tag, units):
        path = f"{stem}_hadrons_{tag}.h5"
        with HadronFile(path) as hf:
            pos = sorted({hf.unit_index(int(u)) for u in units})
        return Hadrons.from_h5(path, units=np.array(pos, dtype=np.int64))

    out = {k: [] for k in ("event", "bg", "w", "y", "phi", "k_one", "n_samples", "S_full",
                           "S_one", "trk_n", "trk_pt", "trk_eta", "trk_phi", "local_event",
                           "pthat_bin", "n_coll", "bg_unit")}
    ycont = Y_CONTENTS if bulk else ("full", "one")
    lcont = L_CONTENTS if bulk else ("full",)
    for c in ycont:
        out[f"Y_{c}"] = []
    for c in lcont:
        out[f"L_{c}"] = []
    per_bgw, per_bgr = {}, {}
    contents = CONTENTS if bulk else ("full",)
    name = os.path.basename(stem)
    if len(local_events) == 0:
        return _pack(out, per_bgw, per_bgr, contents, ycont, lcont, rec, xtra)
    hfr = load("jet_frag", local_events)
    if bulk:
        hj = load("bulk_jet", local_events)
        hb = load("bulk_bg", {i.bg_unit for i in infos})
    if xtra:
        hx = load("extra_frag", local_events)
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
        nx = 0
        if xtra:                                   # the extras' fragments
            X = EventHadrons(hx, hx.unit_index(int(e)))
            nx = X.n_samples
            tx = _tracks(X, info, mask)
        n_s = K if bulk else nf
        k = zw.key_index(zw.event_key(seed, name, int(e)), n_s)
        SF, RF = z_hist(*tf[:3], y, ph) / nf, lab_hist(*tf[:3]) / nf
        YF, LF = side_hist(*tf[:3], y, ph) / nf, lab_side_hist(*tf[:3], ph) / nf
        fk = tf[3] == (k % nf)
        if bulk:
            SJ, RJ = z_hist(*tj[:3], y, ph) / K, lab_hist(*tj[:3]) / K
            SB, RB = z_hist(*tb[:3], y, ph) / K, Rb / K
            S_full, S_zc = SJ + SF, SJ - SB + SF
            R_full = RJ + RF
            YJ, YB = side_hist(*tj[:3], y, ph) / K, side_hist(*tb[:3], y, ph) / K
            LJ, LB = lab_side_hist(*tj[:3], ph) / K, lab_side_hist(*tb[:3], ph) / K
            Y = {"full": YJ + YF, "zc": YJ - YB + YF, "frag": YF}
            L = {"full": LJ + LF, "zc": LJ - LB + LF, "frag": LF}
            jk = tj[3] == k
            one = tuple(np.concatenate([a[jk], c[fk]]) for a, c in zip(tj[:3], tf[:3]))
            unit = info.bg_unit
            RX = np.zeros((N_PT, N_ETA, N_PHI))
            if nx:                                 # the extras: the full event, not zc
                SX, RX = z_hist(*tx[:3], y, ph) / nx, lab_hist(*tx[:3]) / nx
                S_full, R_full = SJ + SF + SX, RJ + RF + RX
                Y["full"] = YJ + YF + side_hist(*tx[:3], y, ph) / nx
                L["full"] = LJ + LF + lab_side_hist(*tx[:3], ph) / nx
                xk = tx[3] == (k % nx)
                one = tuple(np.concatenate([o, x[xk]]) for o, x in zip(one, tx[:3]))
        else:
            S_full = S_zc = SF
            R_full = RF
            Y, L = {"full": YF}, {"full": LF}
            one = tuple(a[fk] for a in tf[:3])
            unit = int(e)
        S_one = z_hist(*one, y, ph)
        Y["one"] = side_hist(*one, y, ph)
        kw = -1 if info.pthat_bin is None else int(info.pthat_bin)
        nc = np.nan if info.n_coll is None else float(info.n_coll)
        for key, v in (("event", g), ("bg", b), ("w", w), ("y", y), ("phi", ph),
                       ("k_one", k), ("n_samples", n_s), ("S_full", S_full), ("S_one", S_one),
                       ("trk_n", len(one[0])), ("trk_pt", one[0]), ("trk_eta", one[1]),
                       ("trk_phi", one[2]), ("local_event", int(e)), ("pthat_bin", kw),
                       ("n_coll", nc), ("bg_unit", unit)):
            out[key].append(v)
        for c in ycont:
            out[f"Y_{c}"].append(Y[c])
        for c in lcont:
            out[f"L_{c}"].append(L[c])
        s = per_bgw.setdefault((int(unit), kw), {
            "n": 0, "ncoll": nc, **{f"S_{c}": np.zeros((N_PT, N_DY, N_DPHI)) for c in contents},
            "R_full": np.zeros((N_PT, N_ETA, N_PHI)),
            **({"R_extra": np.zeros((N_PT, N_ETA, N_PHI))} if xtra else {})})
        if not (s["ncoll"] == nc or (np.isnan(s["ncoll"]) and np.isnan(nc))):
            raise ValueError(f"{stem}: N_coll differs inside background {unit}")
        s["n"] += 1
        s["S_full"] += S_full
        if bulk:
            s["S_zc"] += S_zc
            per_bgr.setdefault(int(unit), RB)
        s["R_full"] += R_full
        if xtra:
            s["R_extra"] += RX
    return _pack(out, per_bgw, per_bgr, contents, ycont, lcont, rec, xtra)


def _pack(out, per_bgw, per_bgr, contents, ycont, lcont, rec, xtra=False):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from jetcorr import mapio
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
                     else np.zeros(0, np.float32)),
         "local_event": np.asarray(out["local_event"], np.int64),
         "pthat_bin": np.asarray(out["pthat_bin"], np.int64),
         "n_coll": np.asarray(out["n_coll"], float),
         "bg_unit": np.asarray(out["bg_unit"], np.int64)}
    for c in ycont:
        d[f"Y_{c}"] = np.asarray(out[f"Y_{c}"], np.float32).reshape(n, N_PT, N_SIDE, N_DYF)
    for c in lcont:
        d[f"L_{c}"] = np.asarray(out[f"L_{c}"], np.float32).reshape(n, N_PT, N_SIDE, N_ETA)
    keys = sorted(per_bgw)
    d["bgw_unit"] = np.asarray([u for u, _ in keys], np.int64)
    d["bgw_k"] = np.asarray([k for _, k in keys], np.int64)
    d["bgw_n"] = np.asarray([per_bgw[x]["n"] for x in keys], np.int64)
    d["bgw_ncoll"] = np.asarray([per_bgw[x]["ncoll"] for x in keys], float)
    for c in contents:
        d[f"bgw_S_{c}"] = (np.array([per_bgw[x][f"S_{c}"] for x in keys]) if keys
                           else np.zeros((0, *shp_z)))
    d["bgw_R_full"] = (np.array([per_bgw[x]["R_full"] for x in keys]) if keys
                       else np.zeros((0, *shp_l)))
    if xtra:                                       # the extras' lab density (assemble)
        d["bgw_R_extra"] = (np.array([per_bgw[x]["R_extra"] for x in keys]) if keys
                            else np.zeros((0, *shp_l)))
    if "zc" in contents:
        units = sorted(per_bgr)
        d["bgr_unit"] = np.asarray(units, np.int64)
        d["bgr_RB"] = (np.array([per_bgr[u] for u in units]) if units
                       else np.zeros((0, *shp_l)))
    d.update(mapio.file_info_npz(rec))
    d.update(grid_arrays())
    d["pool_version"] = np.array(POOL_VERSION, np.int64)
    return d


def grid_arrays():
    """:data:`GRID` as a pool's ``grid_*`` arrays."""
    return {f"grid_{k}": np.asarray(e, float) for k, e in GRID.items()}


def grid_tag(grid=None):
    """A short digest of the edges (:data:`GRID`, or ``grid``): mix.py's pool cache key."""
    import hashlib

    g = GRID if grid is None else grid
    data = b"".join(k.encode() + np.asarray(g[k], "<f8").tobytes() for k in sorted(g))
    return hashlib.sha1(data).hexdigest()[:16]


def pool_grid(part):
    """The edges a pool part (the dict of :func:`file_pool` or :func:`read_root_pool`) was
    binned on: its ``grid_*``, else :data:`GRID_UNSTORED`'s (written before they were stored)."""
    return {k: np.asarray(part[f"grid_{k}"], float) if f"grid_{k}" in part else e
            for k, e in GRID_UNSTORED.items()}


def check_grid(parts):
    """ValueError unless every pool part was binned on :data:`GRID`, the edges equal to 10^-12
    (the C++ map step's linspace may differ from numpy's in the last bit)."""
    for p in parts:
        g = pool_grid(p)
        bad = [k for k, e in GRID.items()
               if g[k].shape != e.shape or not np.allclose(g[k], e, rtol=0, atol=1e-12)]
        if bad:
            name = str(p["fi_name"]) if "fi_name" in p else "a pool part"
            raise ValueError(f"{name}: binned on other edges than zjet_mixing's "
                             f"({', '.join(bad)}): rebuild its pools (mix.py --rebuild, or "
                             "zhadron_pools)")


def read_root_pool(path):
    """A pool written by analysis_root/zjet/zhadron_pools (one ROOT file per production file)
    -> the dict of :func:`file_pool`.  Trees ``pool_events`` (per event the arrays of
    file_pool, S_*/Y_*/L_* flattened), ``pool_bgw`` (per background and window: unit, k, n,
    ncoll, S_<c>, R_full), ``pool_bgr`` (unit, RB; Pb+Pb), ``file_info``, ``pool_grid`` (one
    entry, the edges of :data:`GRID`; not in older pools); pools of version 3 have ``pool_bg``
    (bg_id, W, W2, SW_<c>, T_<c>, R_<c>) instead.  With extra hard scatterings pool_bgw also
    has R_extra (``bgw_R_extra``)."""
    import awkward as ak
    import uproot

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from jetcorr import mapio
    shp_z, shp_l = (N_PT, N_DY, N_DPHI), (N_PT, N_ETA, N_PHI)
    with uproot.open(path) as f:
        ev = f["pool_events"].arrays(library="ak")
        v4 = "pool_bgw" in f
        bg = f["pool_bgw" if v4 else "pool_bg"].arrays(library="ak")
        bgr = f["pool_bgr"].arrays(library="ak") if "pool_bgr" in f else None
        grid = f["pool_grid"].arrays(library="np") if "pool_grid" in f else None
        rec = mapio.read_file_info(f)
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
         "trk_phi": np.asarray(ak.to_numpy(ak.flatten(ev["trk_phi"])), np.float32)}
    nb = len(bg)
    if not v4:
        d.update({"bg_id": col(bg, "bg_id", np.int64), "bg_W": col(bg, "W", float),
                  "bg_W2": col(bg, "W2", float)})
        for c in CONTENTS:
            if f"SW_{c}" in bg.fields:
                for q, shp in (("SW", shp_z), ("T", shp_z), ("R", shp_l)):
                    d[f"bg_{q}_{c}"] = col(bg, f"{q}_{c}", float).reshape(nb, *shp)
        return d
    for k, dt in (("local_event", np.int64), ("pthat_bin", np.int64), ("n_coll", float),
                  ("bg_unit", np.int64)):
        d[k] = col(ev, k, dt)
    for c in Y_CONTENTS:
        if f"Y_{c}" in ev.fields:
            d[f"Y_{c}"] = col(ev, f"Y_{c}", np.float32).reshape(n, N_PT, N_SIDE, N_DYF)
    for c in L_CONTENTS:
        if f"L_{c}" in ev.fields:
            d[f"L_{c}"] = col(ev, f"L_{c}", np.float32).reshape(n, N_PT, N_SIDE, N_ETA)
    for k, dt in (("unit", np.int64), ("k", np.int64), ("n", np.int64), ("ncoll", float)):
        d[f"bgw_{k}"] = col(bg, k, dt)
    for c in CONTENTS:
        if f"S_{c}" in bg.fields:
            d[f"bgw_S_{c}"] = col(bg, f"S_{c}", float).reshape(nb, *shp_z)
    d["bgw_R_full"] = col(bg, "R_full", float).reshape(nb, *shp_l)
    if "R_extra" in bg.fields:
        d["bgw_R_extra"] = col(bg, "R_extra", float).reshape(nb, *shp_l)
    if bgr is not None:
        d["bgr_unit"] = col(bgr, "unit", np.int64)
        d["bgr_RB"] = col(bgr, "RB", float).reshape(len(bgr), *shp_l)
    d.update(mapio.file_info_npz(rec))
    if grid is not None:
        d.update({f"grid_{k}": np.asarray(grid[k][0], float) for k in GRID})
    d["pool_version"] = np.array(POOL_VERSION, np.int64)
    return d


def assemble(parts):
    """Pool parts of version 4 (one per production file, from one or several campaigns, built
    together or apart) -> the dict :class:`Pool` takes, for the whole set: per event its
    global ``event``, background ``bg``, ``file`` (index into ``file_names``) and weight ``w``
    (window cross section / events x N_coll, normalized to mean 1), all from the parts' file
    records (``jetcorr.weights.campaign_ids``, ``window_weights``: the rules of
    ``HadronFileReader``, so a campaign alone gets the ids and weights of mix.py's direct
    build); per background W, W2 and the weighted sums SW, T, R rebuilt from the unweighted
    per-(background, window) sums (the same to rounding); with extra hard scatterings R_zc
    without the extras' lab density (``bgw_R_extra``).  Parts binned on other edges than
    :data:`GRID`: ValueError (:func:`check_grid`)."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from jetcorr import mapio
    from jetcorr import weights as jw

    check_grid(parts)
    recs = [mapio.read_file_info_npz(p) for p in parts]
    ids = jw.campaign_ids(recs)
    wmb = jw.window_weights(recs)
    order = ids["order"]
    parts = [parts[i] for i in order]
    recs = [recs[i] for i in order]
    per = [(int(ids["event_offset"][i]), ids["units"][i]) for i in order]
    _same_contents(parts)
    bulk = "bgw_S_zc" in parts[0]
    xtra = "bgw_R_extra" in parts[0]               # extra hard scatterings

    def base(k):
        k = np.asarray(k, np.int64)
        if not len(wmb):
            return np.ones(len(k))
        return np.where(k >= 0, wmb[np.clip(k, 0, None)], np.nan)

    nc_all = np.concatenate([p["n_coll"] for p in parts])
    use_nc = bool(len(nc_all)) and bool(np.all(np.isfinite(nc_all)))
    w0 = [base(p["pthat_bin"]) * (p["n_coll"] if use_nc else 1.0) for p in parts]
    w0_all = np.concatenate(w0)
    norm = w0_all.mean() if len(w0_all) else 1.0
    per_event = [k for k in parts[0]
                 if not k.startswith(("bgw_", "bgr_", "fi_", "pool_", "grid_"))
                 and k not in ("trk_pt", "trk_eta", "trk_phi")]
    d = {k: np.concatenate([p[k] for p in parts]) for k in per_event}
    for k in ("trk_pt", "trk_eta", "trk_phi"):
        d[k] = np.concatenate([p[k] for p in parts])
    d["event"] = np.concatenate([off + p["local_event"] for (off, _), p in zip(per, parts)])
    d["bg"] = np.concatenate([np.array([u[int(x)] for x in p["bg_unit"]], np.int64)
                              for (_, u), p in zip(per, parts)])
    d["file"] = np.concatenate([np.full(len(p["local_event"]), f, np.int64)
                                for f, p in enumerate(parts)])
    d["w"] = w0_all / norm
    d["file_names"] = np.array([r["name"] for r in recs])
    d.update(grid_arrays())
    # per background, from the per-(background, window) rows
    contents = CONTENTS if bulk else ("full",)
    sums = {}
    for (_, u), p in zip(per, parts):
        rb = dict(zip(p["bgr_unit"].tolist(), p["bgr_RB"])) if bulk else {}
        for r_ in range(len(p["bgw_unit"])):
            b = u[int(p["bgw_unit"][r_])]
            nc = p["bgw_ncoll"][r_] if use_nc else 1.0
            wk = base([p["bgw_k"][r_]])[0] * nc / norm
            n = float(p["bgw_n"][r_])
            s = sums.setdefault(b, {"W": 0.0, "W2": 0.0, "RB": rb.get(int(p["bgw_unit"][r_])),
                                    **{f"{q}_{c}": 0.0 for q in ("SW", "T") for c in contents},
                                    "R_full": 0.0, "R_extra": 0.0})
            s["W"] += n * wk
            s["W2"] += n * wk * wk
            for c in contents:
                S_ = p[f"bgw_S_{c}"][r_]
                s[f"SW_{c}"] = s[f"SW_{c}"] + wk * S_
                s[f"T_{c}"] = s[f"T_{c}"] + wk * wk * S_
            s["R_full"] = s["R_full"] + wk * p["bgw_R_full"][r_]
            if xtra:
                s["R_extra"] = s["R_extra"] + wk * p["bgw_R_extra"][r_]
    bids = sorted(sums)
    d["bg_id"] = np.asarray(bids, np.int64)
    d["bg_W"] = np.array([sums[b]["W"] for b in bids], float)
    d["bg_W2"] = np.array([sums[b]["W2"] for b in bids], float)
    shp = {"SW": (N_PT, N_DY, N_DPHI), "T": (N_PT, N_DY, N_DPHI), "R": (N_PT, N_ETA, N_PHI)}
    for c in contents:
        for q in ("SW", "T"):
            d[f"bg_{q}_{c}"] = (np.array([sums[b][f"{q}_{c}"] for b in bids]) if bids
                                else np.zeros((0, *shp[q])))
    d["bg_R_full"] = (np.array([sums[b]["R_full"] for b in bids]) if bids
                      else np.zeros((0, *shp["R"])))
    if bulk:
        d["bg_R_zc"] = (np.array([sums[b]["R_full"] - sums[b]["W"] * sums[b]["RB"]
                                  - (sums[b]["R_extra"] if xtra else 0.0)
                                  for b in bids]) if bids else np.zeros((0, *shp["R"])))
    return d


def _same_contents(parts):
    """ValueError unless the parts hold the same arrays (the edges aside: older parts have
    none, :func:`pool_grid`)."""
    keys = [{k for k in p if not k.startswith("grid_")} for p in parts]
    if any(k != keys[0] for k in keys):
        raise ValueError("pool parts with different contents (vacuum and Pb+Pb mixed? with "
                         "and without extra hard scatterings?)")


# ── the reduce step's input ─────────────────────────────────────────────────────
class Pool:
    """The per-file pools of a campaign merged: per selected event (``event``, ``bg``,
    ``w``, ``y``, ``phi``, ``k_one``, ``S_full``, ``S_one``, its one-sample tracks
    ``tracks(i)``) and per background (``bg_W``, ``bg_W2``, ``bg_SW[c]``, ``bg_T[c]``,
    ``bg_R[c]`` for c in CONTENTS).  ``zb``: each event's background as an index into the
    per-background arrays.  Vacuum pools store one content: "zc" is then "full".  Binned on
    :data:`GRID` (:func:`check_grid`); :meth:`to_hist` gives the contents as hist.Hist."""

    def __init__(self, d):
        check_grid([d])
        self.__dict__.update({k: v for k, v in d.items() if not k.startswith(("bg_", "grid_"))})
        for pre, conts in (("Y", Y_CONTENTS), ("L", L_CONTENTS)):    # vacuum: one content
            if f"{pre}_full" in d:
                for c in conts:
                    if f"{pre}_{c}" not in d:
                        setattr(self, f"{pre}_{c}", d[f"{pre}_full"])
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
        check_grid(parts)
        _same_contents(parts)
        d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]
             if not k.startswith("grid_")}
        return cls({**d, **grid_arrays()})

    @classmethod
    def load(cls, paths):
        """The pools of ``paths`` (several campaigns' alike): ``.npz`` (``file_pool``) or
        ``.root`` (analysis_root/zjet's ``zhadron_pools``, :func:`read_root_pool`).  Version 4
        (the file records): the ids and weights of the whole set (:func:`assemble`); version
        3: as stored, which holds only for the set it was built on.  Pools binned on other
        edges than :data:`GRID`: ValueError (:func:`check_grid`)."""
        parts = []
        for p in paths:
            if str(p).endswith(".root"):
                parts.append(read_root_pool(p))
                continue
            with np.load(p) as f:
                parts.append({k: f[k] for k in f.files})
        v4 = ["fi_name" in p for p in parts]
        if all(v4):
            return cls(assemble(parts))
        if any(v4):
            raise ValueError("pools of version 3 and 4 mixed: rebuild the older ones")
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

    def rotate_backgrounds(self, seed):
        """Rotate every background by its own random multiple of the lab phi bin (6 degrees),
        in place: its Z's (``phi``), their tracks (``trk_phi``) and its lab densities
        (``bg_R``) together, an exact symmetry of each event; the Z-frame histograms do not
        change.  3dMCGlauber puts the impact parameter along x in every event, so the
        backgrounds' event planes all lie near phi = 0 (zjet_c00: 84 % within pi/8 at 1-2
        GeV): their means are unaffected (phi_Z is uniform), but partners sharing the Z's
        event plane cancel part of its flow modulation, which mixing in data cannot.  After
        the rotation the planes are random.  ``seed``: anything ``np.random.default_rng``
        takes; one draw per background in ``bg_id`` order.  Returns the angles [rad]."""
        rng = np.random.default_rng(seed)
        k = np.array([int(rng.integers(N_PHI)) for _ in self.bg_id], np.int64)
        dphi = k * (2 * np.pi / N_PHI)
        wrap = lambda p: np.mod(p + np.pi, 2 * np.pi) - np.pi
        ev = dphi[self.zb]
        self.phi = wrap(self.phi + ev)
        self.trk_phi = wrap(self.trk_phi + np.repeat(ev, self.trk_n)).astype(self.trk_phi.dtype)
        same = self.bg_R["zc"] is self.bg_R["full"]
        for c in ("full",) if same else ("full", "zc"):
            r = self.bg_R[c]
            self.bg_R[c] = np.stack([np.roll(r[b], k[b], axis=-1) for b in range(len(k))])
        if same:
            self.bg_R["zc"] = self.bg_R["full"]
        return dphi

    def to_hist(self, name, per=None, events=None):
        """A content of the pool as a ``hist.Hist`` on :data:`GRID` (axes by name: ``pt``, the
        track p_T class in GeV, then ``dy`` and ``dphi``, ``eta`` and ``phi``, or ``side`` and
        ``dy``), for notebooks and mplhep; a copy, the mixing works on the arrays.

        ``name``: "S_full", "S_zc", "S_one" (the Z frame), "R_full", "R_zc" (the lab-frame
        density), "Y_<c>" (the dijet proxy's Delta y in 0.1 per side of the Z), "L_<c>" (its lab
        eta per side).  ``per``: None, the mean per Z over all Z's, sum_i w_i X_i / sum_i w_i;
        "background", the same over the Z's of each background (axis ``background``: ``bg_id``);
        "event", each event's own X_i (axis ``event``: its global id; ``events``, pool indices,
        default all; not for S_zc and R_*, stored per background only).  As the pool stands:
        after :meth:`rotate_backgrounds` its rotated phi.

        Storage Weight with the variance NaN: the tracks of an event are correlated (its
        oversamples, its fluid), so sqrt(sum w^2) is not their error (the jackknife over
        backgrounds is, ``all_partners``' blocks); mplhep then draws no error bars."""
        kind, _, c = name.partition("_")
        if kind not in _HIST_AXES or not c:
            raise ValueError(f"{name!r}: S_*, R_*, Y_* or L_*")
        if per not in (None, "background", "event"):
            raise ValueError(f"per {per!r}: None, 'background' or 'event'")
        if events is not None and per != "event":
            raise ValueError("events: for per='event' only")
        if kind == "R" or name == "S_zc":
            if per == "event":
                raise ValueError(f"{name} is stored per background only, not per event")
            store = self.bg_R if kind == "R" else self.bg_SW
            if c not in store:
                raise ValueError(f"{name}: no such content ({', '.join(store)})")
            X, W = store[c], self.bg_W
        elif name == "S_full" and per != "event":
            X, W = self.bg_SW["full"], self.bg_W
        else:
            if not hasattr(self, name):
                raise ValueError(f"{name}: not in this pool (Y_*, L_*: pools of version 4)")
            A = getattr(self, name)
            if per == "event":
                idx = np.arange(self.n) if events is None else np.asarray(events, np.int64)
                return _weight_hist([_category("event", self.event[idx])]
                                    + _grid_axes(_HIST_AXES[kind]), np.asarray(A[idx], float),
                                    name)
            X = np.asarray(A, float)
            wx = self.w.reshape(-1, *[1] * (X.ndim - 1)) * X     # per background, weighted
            X = np.zeros((self.n_bg, *X.shape[1:]))
            np.add.at(X, self.zb, wx)
            W = self.bg_W
        axes = _grid_axes(_HIST_AXES[kind])
        if per == "background":
            return _weight_hist([_category("background", self.bg_id)] + axes,
                                X / W.reshape(-1, *[1] * (X.ndim - 1)), name)
        return _weight_hist(axes, X.sum(axis=0) / W.sum(), name)


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


def partner_map(pool, partners, i):
    """The tracks of the ``partners`` (pool indices; each its one recorded oversample) in the
    frame of Z ``i``, summed -> (N_PT, N_DY, N_DPHI)."""
    trk = [pool.tracks(j) for j in partners]
    return z_hist(*(np.concatenate([t[q] for t in trk]) for q in range(3)), pool.y[i],
                  pool.phi[i])


def explicit_partners(pool, z_index, partner_index, partner_weight, *, n_mix=10, s="full",
                      seed=0, maps=False):
    """CMS-style mixing: every Z (pool indices ``z_index``) with ``n_mix`` partner events
    from as many different backgrounds, never its own.  The backgrounds are drawn without
    replacement with probability ~ the summed ``partner_weight`` of their candidates
    (``partner_index``), then one candidate ~ its weight; a partner contributes its one
    recorded oversample (``Pool.tracks``).  ``s``: "full" (S the 40-sample mean) or "one" (S
    one oversample, as one recorded collision).  With fewer backgrounds available than
    ``n_mix``, all of them (a warning).  -> dict dphi (n, N_PT, N_DPHI), dy (n, N_PT, N_DY)
    per Z, n_mix (n,) the partners used, partners (list of their pool indices); ``maps``:
    also B (n, N_PT, N_DY, N_DPHI), each Z's B map (for :func:`explicit_replicates`)."""
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
    B_maps = np.zeros((n, N_PT, N_DY, N_DPHI)) if maps else None
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
        B = partner_map(pool, js, i) / m if m else np.full((N_PT, N_DY, N_DPHI), np.nan)
        if maps:
            B_maps[a] = B
        dphi[a], dy[a] = observables(S_all[i].astype(float) - B)
    if short:
        warnings.warn(f"{short} of {n} Z's had fewer than {n_mix} other backgrounds to mix "
                      "with: all of them used", RuntimeWarning, stacklevel=2)
    out = {"dphi": dphi, "dy": dy, "n_mix": used, "partners": partners}
    if maps:
        out["B"] = B_maps
    return out


def explicit_replicates(pool, z_index, w, res, S, blocks=()):
    """The weighted mean S - B map of explicit mixing and its delete-block jackknife
    replicates, the block left out of S and B together.  ``res``: :func:`explicit_partners`'
    result for the Z's ``z_index`` (pool indices) with weights ``w``; ``S`` their S maps (n,
    N_PT, N_DY, N_DPHI); ``blocks``: lists of background indices.

    S is the mean over the Z's (weights w_i), B the mean over all (Z, partner) pairs, each
    weighing w_i / m_i (m_i the Z's partners): the same as sum_i w_i B_i / sum_i w_i.  A
    block leaves out its Z's (with their pairs) and the pairs with its events as partners,
    and both means are renormalized over what is left.  The partners come from the same
    sample as the Z's, so a fluid's fluctuations (its multiplicity above all) enter S and B
    together and largely cancel in the mean; leaving a block out of S alone, each Z's S - B
    fixed (:func:`weighted_mean` on the per-Z observables), counts them anyway: on
    zjet_c00-01 8x the scatter of disjoint sub-campaigns at 1-2 GeV, 5x at 2-4, right at
    4-10 GeV (mix.py's check 5).  Re-averaging each Z's B over its partners left instead
    weighs those up by m / (m - 1) and over-corrects by 1/(m - 1) of the block's
    fluctuation.  The observables and the Delta y rebinning are linear: the replicate maps
    give theirs.  ``res`` with ``B`` (``maps=True``) saves recomputing the B maps.
    -> (mean (N_PT, N_DY, N_DPHI), [replicate per block])."""
    z_index = np.asarray(z_index, np.int64)
    w = np.asarray(w, float)
    partners = res["partners"]
    S = np.asarray(S, float)
    B = res.get("B")
    if B is None:
        B = np.stack([partner_map(pool, js, i) / len(js) if len(js)
                      else np.full(S.shape[1:], np.nan) for i, js in zip(z_index, partners)])
    ok = np.all(np.isfinite(B.reshape(len(B), -1)), axis=1)    # Z's with partners
    s_num, s_den = np.tensordot(w[ok], S[ok], axes=1), w[ok].sum()
    b_num, b_den = np.tensordot(w[ok], B[ok], axes=1), w[ok].sum()
    zb = pool.zb[z_index]
    pbg = np.full((len(z_index), max((len(js) for js in partners), default=0)), -1, np.int64)
    for a, js in enumerate(partners):
        pbg[a, :len(js)] = pool.zb[np.asarray(js, np.int64)]
    m = (pbg >= 0).sum(axis=1)
    reps = []
    for blk in blocks:
        own = np.isin(zb, blk) & ok
        sn = s_num - np.tensordot(w[own], S[own], axes=1)
        bn = b_num - np.tensordot(w[own], B[own], axes=1)
        sd = bd = s_den - w[own].sum()
        hit = np.isin(pbg, blk) & (pbg >= 0)
        for a in np.flatnonzero(hit.any(axis=1) & ok & ~own):
            drop = [partners[a][q] for q in np.flatnonzero(hit[a])]
            pw = w[a] / m[a]                               # each pair's weight
            bn = bn - pw * partner_map(pool, drop, z_index[a])
            bd = bd - pw * len(drop)
        reps.append(sn / sd - bn / bd)
    return s_num / s_den - b_num / b_den, reps


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


# ── histograms (hist): the pools' contents and the results, for notebooks and mplhep ──────
#: the axes of each kind of content (Pool.to_hist)
_HIST_AXES = {"S": ("pt", "dy", "dphi"), "R": ("pt", "eta", "phi"), "Y": ("pt", "side", "dyf"),
              "L": ("pt", "side", "eta")}
_HIST_LABELS = {"pt": "track $p_T$ class [GeV]", "dy": r"$\Delta y = \eta - y_Z$",
                "dphi": r"$\Delta\phi = \phi - \phi_Z$", "eta": r"$\eta$", "phi": r"$\phi$",
                "dyf": r"$\Delta y = \eta - y_Z$"}


def _grid_axes(names, edges=None):
    """hist axes of :data:`GRID` (``edges`` replaces some), no flow bins: ``dyf`` is named
    ``dy``, ``side`` is 0 on the Z side (|Delta phi| < pi/2), 1 on the away side."""
    import hist

    edges = {**GRID, **(edges or {})}
    out = []
    for n in names:
        if n == "side":
            out.append(hist.axis.Integer(0, N_SIDE, name="side", label="0: Z side, 1: away side",
                                         underflow=False, overflow=False))
        else:
            out.append(hist.axis.Variable(edges[n], name="dy" if n == "dyf" else n,
                                          label=_HIST_LABELS[n], underflow=False,
                                          overflow=False))
    return out


def _category(name, ids):
    import hist

    return hist.axis.IntCategory([int(x) for x in ids], name=name, overflow=False)


def _weight_hist(axes, value, label, error=None):
    """A Weight-storage hist.Hist with these values and the variance error^2, NaN (unknown)
    without ``error``."""
    import hist

    h = hist.Hist(*axes, storage=hist.storage.Weight(), label=label)
    v = h.view()
    v.value = value
    v.variance = np.nan if error is None else np.asarray(error, float) ** 2
    return h


def _rebin_y(dy, k):
    from cms2507 import rebin_y

    return rebin_y(dy, np.zeros_like(dy), k)[0]


def zframe_hist(value, error=None, label=None):
    """A Z-frame map (N_PT, N_DY, N_DPHI), e.g. S, B or S - B per Z averaged over the Z's, as a
    ``hist.Hist`` (axes ``pt``, ``dy``, ``dphi``), its variance error^2 (NaN without
    ``error``).  Its projections are CMS's observables before the +- merging
    (:func:`observables`): sum over ``dy`` for Delta phi, over the Z side's ``dphi`` bins
    (|Delta phi| < pi/2, the first N_DPHI / 2) for Delta y; their errors from replicates."""
    return _weight_hist(_grid_axes(_HIST_AXES["S"]), np.asarray(value, float),
                        label or "Z frame, per Z", error)


def obs_hist(var, value, error=None, k=None, label=None):
    """CMS's observable of S - B as a ``hist.Hist`` (axes ``pt`` and ``dphi`` or ``dy``), its
    variance error^2 (the jackknife's or the spread's; NaN without ``error``): ``var`` "phi",
    (N_PT, N_DPHI) per radian, or "y", (N_PT, N_DY) per unit Delta y on DETA, or with ``k``
    cms2507.rebin_y's |Delta y| < 4 in bins of 0.4 k (mix.py's results: ``obs_hist("y",
    *R["all"]["y"], k=2)``).

    hist adds the variances of what it sums as independent: right for Pb+Pb - pp
    (``h_pbpb + h_pp * -1``: Weight storage has no ``-``), wrong for neighbouring bins
    (rebinning) and for two results on the same Z's (modes, the bulk on and off): those from
    the replicates (:func:`observables_hist` on the difference, ``mix.py``'s checks)."""
    value = np.asarray(value, float)
    if var == "phi":
        name, edges = "dphi", DPHI
    elif var == "y":
        name, edges = "dy", DETA if k is None else DETA[2:23][::k]
    else:
        raise ValueError(f"var {var!r}: 'phi' or 'y'")
    if value.shape != (N_PT, len(edges) - 1):
        raise ValueError(f"{var}: shape {value.shape}, the edges want {(N_PT, len(edges) - 1)}")
    return _weight_hist(_grid_axes(("pt", name), {name: edges}), value,
                        label or f"S - B, d<N_ch>/d{name}", error)


def observables_hist(D, reps=(), k=None, label=None):
    """S - B per Z (:func:`all_partners`' result, (N_PT, N_DY, N_DPHI)) and its jackknife
    replicates -> {"phi": Hist, "y": Hist}: CMS's observables (:func:`observables`), Delta y
    rebinned with ``k`` before the jackknife (as mix.py), the variance the jackknife's (NaN
    with fewer than two replicates).  For a difference of two results on the same Z's pass
    D1 - D2 and the replicates' differences."""
    dphi, dy = observables(D)
    rp = [observables(r) for r in reps]
    if k is not None:
        dy = _rebin_y(dy, k)
    ep = ey = None
    if len(rp) > 1:
        ep = jackknife([r[0] for r in rp])
        ry = np.array([r[1] for r in rp])
        ey = jackknife(ry if k is None else _rebin_y(ry, k))
    return {"phi": obs_hist("phi", dphi, ep, label=label),
            "y": obs_hist("y", dy, ey, k, label=label)}
