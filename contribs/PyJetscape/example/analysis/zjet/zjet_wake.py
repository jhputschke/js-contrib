"""Z-hadron wake from the hadron files of a Z+jet production (``zjet_wake.ipynb``).

The files are those of ``run_prod_jet.py`` with a Z+jet user XML (PythiaGun ``<trigger>Z``,
``example/config/PbPb_0_30_ZJet.xml``): ``--hadrons-only`` jobs, or ``--write-particlize``
jobs hadronized by ``hadronize.py``.  ``HadronFileReader`` reads both, and gives each event's
Z (``EventInfo.z``, ``jetscape.hadrons_h5.Z_COLUMNS``) and N_coll (``EventInfo.n_coll``).

The Z frame
-----------
Hadron angles are measured from the Z, as CMS (2507.09307, 2103.04377) and ATLAS
(2008.09811) measure them: ``dphi = phi_h - phi_Z`` in [-pi/2, 3pi/2) (the recoiling jet
near pi) and ``deta = eta_h - y_Z`` (the hadron's pseudorapidity against the Z's rapidity,
CMS's Delta y_ch,Z up to the sign).  The Z is untouched by the medium, so its direction is
the jet's before any quenching, the reference the wake studies use.

What is drawn
-------------
* **wake**: the paired truth <bulk_jet> - <bulk_bg>, each event's jet leg against its own
  background (correlated sampling: sample k against sample k), per Z.  No mixed events: the
  background is the same fireball without the jet's deposition.
* **jet**: the fragments (``jet_frag``) without the Z's two leptons, which pass the medium
  and Colorless untouched (:func:`not_z_lepton`).
* **total** = wake + jet, the medium-modified Z-tagged yield; with a vacuum reference
  (``run_prod_vacuum.py``) the analogue of CMS's PbPb - pp.

Weights
-------
Every event weighs ``sigma_k / N_k`` (its pTHat window's cross section over the events in
that window) times its ``n_coll``: the geometry is drawn with P(b) ~ b and every event gets
one Z, while Z production scales with N_coll.  :func:`event_weights` (``jetcorr.weights``)
normalizes them to mean 1 over the selected events, so ``HadronFileReader``'s event averages
become weighted averages (it divides by the number of events).  The angles and the portable
random choices are ``jetcorr.keys``'; both are re-exported here.
"""

from __future__ import annotations

import os
import sys

import numpy as np

try:
    import jetcorr  # noqa: F401
except ImportError:                                   # example/analysis not on sys.path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jetcorr.keys import (event_key, fnv1a64, key_index, key_uniform,  # noqa: E402,F401
                          splitmix64, wrap_dphi)
from jetcorr.weights import effective_events, event_weights, weight_fn  # noqa: E402,F401

#: |pid| of a Z's decay leptons
LEPTON_PIDS = (11, 13)

#: the selection of CMS 2507.09307 (Z-hadron Delta phi / Delta y), the direct wake measurement
CMS_2507 = dict(pt=(40.0, 350.0), ymax=2.4, mass=(60.0, 120.0), lep_pt=20.0, lep_eta=2.4)
#: CMS 2103.04377 (Z-hadron, p_T^Z > 30)
CMS_2103 = dict(pt=(30.0, np.inf), ymax=None, mass=(60.0, 120.0), lep_pt=20.0, lep_eta=2.4)


# ── the event record ──────────────────────────────────────────────────────────────
def refuse_minimum_bias(reader, who):
    """SystemExit if a production file of ``reader`` has no jet_frag: a minimum-bias job
    (run_prod_jet.py --hard none, docs/Plans/PLAN_multi_hard.md) has no trigger, so no Z."""
    if any("jet_frag" not in reader.tags(i) for i in range(reader.n_files)):
        raise SystemExit(f"{who}: files without jet_frag (minimum bias: no trigger) have no Z")


def event_table(reader):
    """-> dict of per-event arrays over the campaign: event, z_pt, z_y, z_phi, z_m,
    lep{1,2}_{pid,pt,eta,phi} (NaN / 0 without a Z), n_coll (NaN if not recorded),
    pthat_bin (-1 without windows), sigma_bin [mb] (the window's cross section)."""
    from jetscape.hadrons_h5 import Z_COLUMNS

    n = reader.n_events
    cols = {k: np.full(n, np.nan) for k in Z_COLUMNS}
    n_coll = np.full(n, np.nan)
    pbin = np.full(n, -1, dtype=np.int64)
    for g in range(n):
        info = reader.event_info(g)
        if info.z:
            for k in Z_COLUMNS:
                cols[k][g] = info.z[k]
        if info.n_coll is not None:
            n_coll[g] = info.n_coll
        if info.pthat_bin is not None:
            pbin[g] = info.pthat_bin
    sigma = np.ones(n)
    if reader.pthat_bins is not None:
        sig = np.array([reader.pthat_bin_sigma(k)[0] for k in range(len(reader.pthat_bins))])
        sigma = np.where(pbin >= 0, sig[np.clip(pbin, 0, None)], np.nan)
    return dict(event=np.arange(n), n_coll=n_coll, pthat_bin=pbin, sigma_bin=sigma, **cols)


def lepton_p3(pt, eta, phi):
    """(px, py, pz) of massless leptons from the record's pt, eta, phi."""
    pt, eta, phi = (np.asarray(x, float) for x in (pt, eta, phi))
    return np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)], axis=-1)


def select(table, pt=(40.0, 350.0), ymax=2.4, mass=(60.0, 120.0), lep_pt=20.0, lep_eta=2.4,
           flavour=None):
    """Boolean mask of the events passing an analysis's Z selection (``CMS_2507``, ...):
    p_T^Z in ``pt``, |y_Z| < ``ymax`` (None: no cut), m in ``mass``, both leptons above
    ``lep_pt`` and inside |eta| < ``lep_eta``; ``flavour`` 11 or 13 for one channel."""
    t = table
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(t["z_pt"]) & (t["z_pt"] > pt[0]) & (t["z_pt"] < pt[1])
        ok &= (t["z_m"] > mass[0]) & (t["z_m"] < mass[1])
        if ymax is not None:
            ok &= np.abs(t["z_y"]) < ymax
        for i in (1, 2):
            ok &= (t[f"lep{i}_pt"] > lep_pt) & (np.abs(t[f"lep{i}_eta"]) < lep_eta)
        if flavour is not None:
            ok &= np.abs(t["lep1_pid"]) == flavour
    return ok


# ── per-hadron callables for HadronFileReader ─────────────────────────────────────
def z_frame(ev, info):
    """(deta, dphi) of every hadron in its event's Z frame."""
    z = info.z
    return ev.eta - z["z_y"], wrap_dphi(ev.phi - z["z_phi"])


def not_z_lepton(ev, info, rtol=1e-3):
    """False for the Z's two leptons (same pid and momentum as lep1 / lep2), True for the
    rest; decay leptons of hadrons stay."""
    out = np.ones(len(ev), bool)
    z = info.z
    if not z:
        return out
    p3 = np.stack([ev.px, ev.py, ev.pz], axis=-1)
    for i in (1, 2):
        lp = lepton_p3(z[f"lep{i}_pt"], z[f"lep{i}_eta"], z[f"lep{i}_phi"])
        tol = rtol * max(float(np.linalg.norm(lp)), 1.0)
        out &= ~((ev.pid == int(z[f"lep{i}_pid"]))
                 & (np.linalg.norm(p3 - lp, axis=1) < tol))
    return out


def track_mask(eta_max=2.4, charged=True):
    """Tracks with |eta| < eta_max (charged by default), never the Z's leptons."""
    def mask(ev, info):
        m = np.abs(ev.eta) < eta_max
        if charged:
            m &= ev.charged
        return m & not_z_lepton(ev, info)
    return mask


# ── the binned Z-frame yields ─────────────────────────────────────────────────────
#: the track p_T classes of CMS 2507.09307 [GeV]
PT_CLASSES = (1.0, 2.0, 4.0, 10.0)


def pt_z_frame(ev, info):
    """(p_T, deta, dphi) of every hadron, its Z frame binned per track p_T class."""
    deta, dphi = z_frame(ev, info)
    return ev.pt, deta, dphi


def z_frame_yields(reader, events, w, deta_bins, dphi_bins, pt_edges=PT_CLASSES,
                   eta_max=2.4, fragments=True):
    """Per-Z yields in (track p_T class, Delta eta, Delta phi), one pass per source ->
    dict ``wake`` (the paired <bulk_jet> - <bulk_bg>) and ``jet`` (jet_frag without the Z's
    leptons, if ``fragments``), each (mean, error) of shape (n_pt, n_eta, n_phi): counts per
    Z per bin, not divided by the bin area.  ``events``: global indices; ``w``:
    :func:`event_weights`."""
    bins = (np.asarray(pt_edges, float), np.asarray(deta_bins), np.asarray(dphi_bins))
    kw = dict(mask=track_mask(eta_max), weights=weight_fn(w), events=events, per_event=True)
    out = {"wake": reader.jet_minus_background(pt_z_frame, bins, **kw)}
    if fragments:
        out["jet"] = reader.hist("jet_frag", pt_z_frame, bins, **kw)
    return out


def vacuum_yields(reader, events, w, deta_bins, dphi_bins, pt_edges=PT_CLASSES,
                  eta_max=2.4):
    """The same for a vacuum reference (run_prod_vacuum.py: jet_frag only)."""
    bins = (np.asarray(pt_edges, float), np.asarray(deta_bins), np.asarray(dphi_bins))
    return reader.hist("jet_frag", pt_z_frame, bins, mask=track_mask(eta_max),
                       weights=weight_fn(w), events=events, per_event=True)


def project(h, err, axis, bins_kept=None):
    """Sum a 2-d (Delta eta, Delta phi) yield over ``axis`` (0: eta -> a Delta phi
    profile, 1: phi -> a Delta eta profile), over ``bins_kept`` of that axis (a boolean
    mask; default all), with the errors added in quadrature (bins of one event are
    correlated, so this is a lower bound)."""
    h, err = np.asarray(h), np.asarray(err)
    if bins_kept is not None:
        sl = [slice(None)] * 2
        sl[axis] = np.asarray(bins_kept)
        h, err = h[tuple(sl)], err[tuple(sl)]
    return h.sum(axis), np.sqrt((err ** 2).sum(axis))


def cms_dphi(h, err, dphi_bins):
    """A Delta phi yield (counts per Z per bin, last axis over ``dphi_bins`` from -pi/2)
    in CMS's convention (2507.09307): +/-Delta phi merged and halved (the mean of the bins
    at Delta phi and -Delta phi), per radian, minus its Delta phi average (what the
    mixed-event subtraction removes over all Delta y). -> (value, error); the average's
    error is left out."""
    edges = np.asarray(dphi_bins, float)
    c = 0.5 * (edges[1:] + edges[:-1])
    neg = (-c + np.pi / 2) % (2 * np.pi) - np.pi / 2
    mirror = np.array([int(np.argmin(np.abs(x - c))) for x in neg])
    if not np.allclose(c[mirror], neg, atol=1e-9):
        raise ValueError("dphi_bins must be symmetric under dphi -> -dphi (an even number "
                         "of equal bins from -pi/2)")
    width = np.diff(edges)
    h, err = np.asarray(h, float), np.asarray(err, float)
    hs = 0.5 * (h + h[..., mirror]) / width
    self_ = mirror == np.arange(len(c))
    es = np.where(self_, err, 0.5 * np.hypot(err, err[..., mirror])) / width
    return hs - np.average(hs, axis=-1, weights=width)[..., None], es


# ── hydro: one event's energy-density wake in the transverse plane ────────────────
def pair_grid(f):
    """(x, y, eta_s, tau) axes of a pair file (``PairH5Writer``: x, y, eta_s from their
    minima to minus their minima, tau from tau_min in steps of dtau)."""
    a = f.attrs
    x = np.linspace(a["x_min"], -a["x_min"], int(a["nx"]))
    y = np.linspace(a["y_min"], -a["y_min"], int(a["ny"]))
    eta = np.linspace(a["eta_min"], -a["eta_min"], int(a["neta"]))
    tau = a["tau_min"] + a["dtau"] * np.arange(f["arr"].shape[-1])
    return x, y, eta, tau


def transverse_wake(path, event, taus, eta_window=0.5):
    """-> (x, y, {tau: de(x, y)}, z, vertex, phi_lead): the jet leg minus its background in
    GeV/fm^3, averaged over |eta_s - y_Z| < ``eta_window``, at the frames nearest to
    ``taus``; the event's Z record (diag/), the hard vertex and the azimuth of the hardest
    initiator that is not a lepton (initiators)."""
    import h5py

    import jetscape.h5_compression  # noqa: F401  (the pair files' Blosc filter)

    with h5py.File(str(path), "r") as f:
        x, y, eta, tau = pair_grid(f)
        z = {k: float(f[f"diag/{k}"][event])
             for k in ("z_pt", "z_y", "z_phi", "z_m") if f"diag/{k}" in f}
        sl = np.flatnonzero(np.abs(eta - z.get("z_y", 0.0)) < eta_window)
        if not len(sl):
            sl = [int(np.argmin(np.abs(eta - z.get("z_y", 0.0))))]
        n_live = int(min(f["ntau_freezeout"][event], f["ntau_freezeout_bg"][event]))
        out = {}
        for t in taus:
            k = int(np.argmin(np.abs(tau[:n_live] - t)))
            e = f["arr"][event, 0, :, :, sl[0]:sl[-1] + 1, k]
            eb = f["arr_bg"][event, 0, :, :, sl[0]:sl[-1] + 1, k]
            out[float(tau[k])] = (e - eb).mean(axis=-1)
        vertex, phi_lead = np.zeros(2), None
        if "shower/initiators" in f:
            o = f["shower/initiator_offsets"][:]
            ini = f["shower/initiators"][o[event]:o[event + 1]]
            if len(ini):
                vertex = ini[0, 7:9]
                par = ini[~np.isin(np.abs(ini[:, 1]).astype(int), LEPTON_PIDS)]
                if len(par):
                    lead = par[np.argmax(np.hypot(par[:, 3], par[:, 4]))]
                    phi_lead = float(np.arctan2(lead[4], lead[3]))
    return x, y, out, z, vertex, phi_lead
