#!/usr/bin/env python3
"""
analysis/OO-RHIC/oo_analysis_rhic.py -- the O+O physics run at RHIC (phase 3): what is selection
bias, what is quenching.  Reads the phase-3 families of oo_bias.py and writes
results/oo_analysis_rhic.json for oo_analysis_rhic.ipynb.

    python oo_analysis_rhic.py                                   # /data/OO-RHIC_phys, all families
    python oo_analysis_rhic.py --out /data/OO-RHIC_phys_smoke --mb-root /data/OO-RHIC_phys   # a test root
    python oo_analysis_rhic.py --cached                          # reuse results/oo_analysis_rhic_<family>.npz

Samples (oo_bias.py, phase 3; tuning page "Phase 3: the runs as decided"):
  oo_vac  O+O, jet in vacuum (--vacuum-jet; the medium is made)      -> selection only
  oo_med  O+O, jet in the medium (Matter + LBT), the same seeds         -> selection x quenching
  pp_ref  p+p (config/pp200_isr_rhictune.xml), jet in vacuum            -> the R_AA reference
  pp_med  p+p, jet in p+p's own medium (3c)                             -> consistent reference, QS check
  phys_mb O+O minimum bias (bulk hadrons only)                          -> class edges, <N_coll> per class
The same seeds do not give the same events (LBT shifts the random stream): every ratio of two
samples is statistical, with independent errors.

Classes: percentiles of the minimum-bias estimator (larger = more central), the jet events sorted by
their own estimator (bulk + hard-event charged hadrons, as oo_bias.load).  Estimators: STAR's EPD
(2.1 < |eta| < 5.1; STAR's classes), sPHENIX's MBD (3.51 < |eta| < 4.61; sPHENIX's classes),
mid-rapidity |eta| < 1 (pT > 0.2; STAR's classes; contains the jet), and the impact parameter
(truth: no selection by construction).

Weights.  An O+O event of window k stands for N_coll x sigma_k / N_k of hard scatterings per
minimum-bias event (sigma_NN cancels), a p+p event for sigma_k / N_k.  Per class c of width f_c:
    R_AA^c(x) = sum_{ev in c} N_coll w h(x) / (f_c <N_coll>_c^MB  sum_pp w h(x))
    selection  = R_AA^c of oo_vac (1 = no bias; the b classes check the normalisation)
    quenching  Q^c = R_AA^c(oo_med) / R_AA^c(oo_vac)   (the same classes, the normalisations cancel)
    R_CP = R_AA^a / R_AA^b.
Stitching: a window enters a bin where every compared family has at least MIN_ENTRIES raw entries
there, pooled over the classes; one choice per observable and bin, the same for every class,
estimator and family (numerator and denominator).  The rule keeps out a low window's rare entries
with their large weight; pooling keeps in a window that carries a real share of the yield but is
thin in a sparse class (that class just gets the larger error, from the sum of squared weights).
Until 2026-11-25 the minimum over the classes decided, which dropped windows carrying up to half of
a bin's yield (the 7-9 GeV window, a quarter of the 7-30 GeV triggers; the 5-7 GeV window at 5-6 GeV).
Diagnostics per bin: the share of each family's yield in the windows left out (dropped_share,
flagged above DROP_FLAG), and the share of the lowest generated window (lowest_window_share): above
INCOMPLETE_SHARE the bin is marked incomplete (complete = False), since pTHat below the lowest
window is not generated and would contribute too (the charged hadrons below ~7 GeV).
Overlapping windows (3e, oo_bias.py HIJET_WINDOWS: narrow 35-40 and 47-55 GeV samples inside the
wide 30-40 and 40-55 windows): every window is cut at all window edges into pieces; each piece's
cross section is the event-count-weighted mean of the covering windows' estimates sigma_k n_kj / N_k,
and each of its events gets sigma_piece / (all events in the piece) (stitch_pieces).  Without
overlaps the weights are exactly sigma_k / N_k (closure checked to 1e-15 on the full phase 3, also
with extra split points, --split-at); unbiased in a synthetic test with a falling spectrum.  The
stitching rule and the class sums keep the wide windows as groups.
Histograms: hist (boost-histogram) per window x class x bin, Weight storage (sum w, sum w^2 per
event and bin) and the raw entries; filled from per-event arrays, as the classes are assigned later.

Observables (on the hard event's hadrons, the jet_frag file, as oo_bias_jets.py: no bulk, so no
background subtraction; the medium response (recoils) is included):
  charged hadrons |eta| < 1 (HAD_EDGES); recoils included, holes NOT subtracted: hadron-level
    observables (this spectrum and the h-h I_CP) need prod_OO_isr/negative_hadrons.py's negative
    hadrons (pstat -1, counted with weight -1); not yet run on the phase-3 medium files.  Holes are
    soft, so the effect should be small above ~4 GeV, but it is not quantified
  sPHENIX full jets anti-kT R = 0.4, |eta_jet| < 0.7, visible particles pT > 0.2 (SPH_EDGES), with
    the medium's holes and their colour partners subtracted at parton level: the four-momenta of
    the holes (partons/, pstat -1) and of the partner copies (hadronization_input/, the hole's
    three-momentum) inside dR < R of the jet axis are subtracted and pT recomputed (iColored
    hadronizes recoil + partner, the balance is recoil - hole); sph_jet_raw without it
  STAR charged jets R = 0.2 and 0.5, |eta_jet| < 1 - R (CH_EDGES), the holes + partners in the
    cone subtracted with the charged fraction 2/3 of their transverse momentum (CHARGED_FRAC; parton
    four-momenta do not separate into charged and neutral); ch_jet_R*_raw without
  STAR I_CP (arXiv:2604.13935): one charged trigger 7 < pT < 30 GeV, |eta| < 1.5, chosen at random
    if several;
    h-h: associated charged 3 < pT < 7 GeV, |eta| < 1.5 (hard event + bulk), near |dphi| < 0.8,
         recoil |dphi - pi| < 1, flat background from 0.8 < |dphi| < 1.6 (STAR's v_n = 0 variation);
    h+jet: recoil charged jets R = 0.2, 0.5 (hard-event charged particles pT > 0.2, |eta| < 1.5),
         |eta_jet| < 1.5 - R, |dphi - pi| < pi/4, 8 < pT,jet < 19 GeV (no rho A, no mixed events:
         the hard event alone); the holes + partners in the cone subtracted with the charged
         fraction 2/3 of their transverse momentum (CHARGED_FRAC; hjet_R*_raw without); the
         trigger hadron is fine as it is; the h-h associated hadrons are not subtracted (hadron level:
         needs negative_hadrons.py, as the charged-hadron spectrum);
    per-trigger yields per class, I_CP = 0-10 % (and 10-20, 20-40) over 40-60 % in the EPD.
Model results, not vetted physics results.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import hist
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _paths  # noqa: E402,F401  (analysis/OO-RHIC and its test directories on sys.path)
import oo_bias as B  # noqa: E402
import oo_bias_jets as J  # noqa: E402

FAMILIES = ("oo_vac", "oo_med", "pp_ref", "pp_med")
OO_FAMILIES, PP_FAMILIES = ("oo_vac", "oo_med"), ("pp_ref", "pp_med")
MB_SAMPLE = "phys_mb"
MIN_ENTRIES = 10
DROP_FLAG = 0.02            # a bin whose dropped windows carry more than this share of a family's yield is flagged
INCOMPLETE_SHARE = 0.02     # a bin whose lowest generated window carries more than this share is incomplete
INVISIBLE = (12, 14, 16)
CONST_PTMIN = 0.2
HAD_ETA = 1.0
HAD_EDGES = np.array([2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 30.0])
SPH_R, SPH_ETA = 0.4, 0.7
SPH_EDGES = np.array([10, 12, 15, 19, 24, 30, 36, 43, 51, 60.0])   # above 19 GeV: sPHENIX's bins (sPH-CONF-JET-2026-02)
CH_RS = (0.2, 0.5)
CH_EDGES = np.array([5, 8, 10, 12, 15, 19, 25, 30, 40.0])
TRIG = (7.0, 30.0)
TRIG_ETA = 1.5
ASSOC = (3.0, 7.0)
NEAR, RECOIL, BG = 0.8, 1.0, (0.8, 1.6)
HJET_DPHI = np.pi / 4
HJET_EDGES = np.array([5, 6, 8, 10, 12, 15, 19.0])                # STAR's recoil-jet bins (HEPData)
HJET_INT = (8.0, 19.0)                                            # STAR's integrated I_CP range
STAR_CL, SPH_CL = B.STAR_CLASSES, B.SPH_CLASSES
ESTIMATORS = {"epd": ("n_epd", STAR_CL), "mbd": ("n_mbd", SPH_CL), "mid10": ("n_mid10", STAR_CL),
              "b": ("b", SPH_CL)}
ICP_REF = (40, 60)
#: STAR's values from the paper's text (arXiv:2604.13935 v2): 0-10 % / 40-60 % in the EPD, (value,
#: stat, syst); the full tables are read from refs/star_oo_icp (HEPData ins3145466, load_data).
STAR_ICP = {"hh_recoil_0_10": (0.836, 0.021, 0.032), "hjet_R0.5_0_10": (0.806, 0.037, 0.025),
            "hh_recoil_slope_per_pct": (-0.020, 0.012, None), "hjet_R0.5_slope": (-0.019, 0.014, None),
            "hjet_R0.2_slope": (-0.008, 0.020, None)}


# ── per file ──────────────────────────────────────────────────────────────────────────────────
def _kin(p):
    E, px, py, pz = p.T
    pt = np.hypot(px, py)
    pa = np.sqrt(px * px + py * py + pz * pz)
    eta = 0.5 * np.log(np.clip(pa + pz, 1e-12, None) / np.clip(pa - pz, 1e-12, None))
    return pt, eta, np.arctan2(py, px)


def _jets(p, ev, n, R, eta_part, ptmin=1.0, full=False):
    """anti-kT jets per event of the particles p (rows) with event index ev: (event, pt, eta, phi),
    with full=True also (px, py)."""
    import awkward as ak
    import fastjet
    cnt = np.bincount(ev, minlength=n)
    parts = ak.zip({"px": ak.unflatten(p[:, 1], cnt), "py": ak.unflatten(p[:, 2], cnt),
                    "pz": ak.unflatten(p[:, 3], cnt), "E": ak.unflatten(p[:, 0], cnt)})
    js = fastjet.ClusterSequence(parts, fastjet.JetDefinition(fastjet.antikt_algorithm, R)).inclusive_jets(min_pt=ptmin)
    nj = ak.to_numpy(ak.num(js))
    if not nj.sum():
        z = np.zeros(0)
        return (z.astype(int), z, z, z, z, z) if full else (z.astype(int), z, z, z)
    jx, jy, jz, jE = (ak.to_numpy(ak.flatten(js[k])) for k in ("px", "py", "pz", "E"))
    pt, eta, phi = _kin(np.stack([jE, jx, jy, jz], axis=1))
    if full:
        return np.repeat(np.arange(n), nj), pt, eta, phi, jx, jy
    return np.repeat(np.arange(n), nj), pt, eta, phi


def hard_partons(h, n):
    """Per event the outgoing partons of the hard scattering (initiators/, pstat 1000: the showers'
    initiators after ISR): rows (event, pid, eta, phi)."""
    if "initiators" not in h:
        return np.zeros((0, 4))
    off = h["initiators/offsets"][: n + 1]
    d = h["initiators/data"][: off[-1]]
    ev = np.repeat(np.arange(n), np.diff(off))
    m = d[:, 2] == 1000
    px, py, pz = d[m, 3], d[m, 4], d[m, 5]
    pt = np.hypot(px, py)
    return np.stack([ev[m], d[m, 1], np.arcsinh(pz / np.maximum(pt, 1e-9)), np.arctan2(py, px)], axis=1)


def jet_flavour(je, jeta, jphi, hard, R):
    """Per jet 1 (quark or antiquark) or 2 (gluon) from the nearest hard outgoing parton of its event
    within dR < R; 0 when none is that close."""
    fl = np.zeros(len(je), int)
    if not len(je) or not len(hard):
        return fl
    for i in range(len(je)):
        hp = hard[hard[:, 0] == je[i]]
        if not len(hp):
            continue
        dr = np.hypot(hp[:, 2] - jeta[i], _dphi(hp[:, 3], jphi[i]))
        k = np.argmin(dr)
        if dr[k] < R:
            fl[i] = 2 if int(hp[k, 1]) == 21 else 1
    return fl


def negative_partons(h, n):
    """Per event the medium's holes (partons/, pstat -1) and the copies of their colour partners
    (hadronization_input/ entries with the hole's three-momentum, as negative_hadrons.py finds
    them): rows (event, E, px, py, pz).  iColored hadronizes the recoils and the partners but drops
    the holes, so the jet's hadrons hold recoil + partner; subtracting hole + partner in the cone
    leaves recoil - hole (JETSCAPE's balance)."""
    if "partons/data" not in h:
        return np.zeros((0, 5))
    po = h["partons/offsets"][: n + 1]
    pdat = h["partons/data"][: po[-1]]
    hole = pdat[:, 2] == -1
    if not hole.any():
        return np.zeros((0, 5))
    pev = np.repeat(np.arange(n), np.diff(po))
    rows = [np.column_stack([pev[hole], pdat[hole][:, 3:7]])]
    if "hadronization_input/data" in h:
        ho = h["hadronization_input/offsets"][: n + 1]
        hdat = h["hadronization_input/data"][: ho[-1]]
        for e in np.unique(pev[hole]):
            hp = hdat[ho[e]:ho[e + 1], 3:7]                            # E, px, py, pz
            for hrow in pdat[(pev == e) & hole]:
                j = np.flatnonzero(np.abs(hp[:, 1:4] - hrow[4:7]).sum(1) < 1e-6)
                if len(j):
                    rows.append(np.concatenate([[e], hp[j[0]]])[None, :])
    return np.concatenate(rows)


#: the charged fraction of a hole's / partner's momentum for charged jets (isospin-symmetric
#: hadronization: about 2/3 of the energy goes into charged hadrons)
CHARGED_FRAC = 2.0 / 3.0


def subtract_negatives(je, jeta, jphi, jx, jy, neg, R, frac=1.0):
    """pT of each jet after subtracting frac x the transverse momenta of the event's negative
    partons (holes + partner copies) inside dR < R of the jet axis."""
    sx, sy = np.zeros(len(je)), np.zeros(len(je))
    if len(neg) and len(je):
        nev = neg[:, 0].astype(int)
        _, neta, nphi = _kin(neg[:, 1:5])
        for i in np.flatnonzero(np.isin(je, nev)):
            q = nev == je[i]
            inside = np.hypot(neta[q] - jeta[i], _dphi(nphi[q], jphi[i])) < R
            sx[i] = neg[q][inside, 2].sum()
            sy[i] = neg[q][inside, 3].sum()
    return np.hypot(jx - frac * sx, jy - frac * sy)


def _hist(ev, x, edges, n, w=None):
    out = np.zeros((n, len(edges) - 1))
    b = np.digitize(x, edges) - 1
    ok = (b >= 0) & (b < len(edges) - 1)
    np.add.at(out, (ev[ok], b[ok]), 1.0 if w is None else w[ok])
    return out


def _dphi(a, b):
    return np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)


def samples_of(so, uo, n):
    """Fragmentation oversamples (--n-frag) of the first n units: samples per unit, the unit of each
    sample, particles per sample, and per particle its sample and whether that is its unit's first."""
    nsu = np.diff(uo[: n + 1]).astype(int)
    ns = int(uo[n])
    unit_of = np.repeat(np.arange(n), nsu)
    cs = np.diff(so[: ns + 1]).astype(int)
    samp = np.repeat(np.arange(ns), cs)
    return nsu, ns, unit_of, cs, samp, samp == np.asarray(uo)[unit_of[samp]]


def rows_to_samples(unit, nsu, uo):
    """Rows tagged with a unit index, copied to every sample of that unit: (row index, sample index)."""
    unit = np.asarray(unit, int)
    reps = nsu[unit]
    idx = np.repeat(np.arange(len(unit)), reps)
    k_in = np.arange(len(idx)) - np.repeat(np.cumsum(reps) - reps, reps)
    return idx, np.asarray(uo)[unit[idx]] + k_in


def samples_to_units(v, unit_of, cs, n):
    """Per-sample observable rows averaged into their unit's row.  Samples without any hadron (Pythia
    could not hadronize them with any seed: rehadronize_jets.py, rehadronize_icolored.py) are left out
    of the average; a unit with no hadronized sample contributes zero."""
    good = cs > 0
    ng = np.bincount(unit_of[good], minlength=n).astype(float)
    w = np.where(good, 1.0 / np.maximum(ng[unit_of], 1.0), 0.0)
    r = np.zeros((n,) + v.shape[1:])
    np.add.at(r, unit_of, v * w.reshape((-1,) + (1,) * (v.ndim - 1)))
    return r


#: re-hadronized samples (rehadronize_colorless.py): the root they were read from (--out) and the root of
#: the original files (--est-root), whose centrality estimators and bulk are used, so that every event
#: keeps its class and its h-h background whatever the jet hadronization
EST_ROOT = None
OUT_ROOT = None


def file_obs(path):
    """Per event of one jet_frag file (and its bulk_bg file): units, estimators and observables.  Holes:
    iColored files carry none among the hadrons (subtracted at parton level, negative_partons);
    Colorless files (rehadronize_colorless.py, run_prod_jet_single.py) carry them as pstat -1 hadrons,
    subtracted at hadron level (JETSCAPE's): signed spectra and h-h counts, jets of the positive hadrons
    minus the negative ones in the cone, triggers from the positive hadrons only.
    Fragmentation oversamples (--n-frag of the rehadronize tools): every sample is a sub-event (jets
    clustered and a trigger chosen per sample; the event's holes, partners, hard partons and bulk
    copied to each), and the samples' observables are averaged into the event's row (samples_to_units),
    as auau_raa.py: the errors stay per event.  The centrality estimators are the event's: bulk +
    the first sample's charged hadrons.  One sample per event: as before, bit for bit."""
    import h5py
    import hdf5plugin  # noqa: F401
    orig = path if EST_ROOT is None else str(Path(EST_ROOT) / Path(path).relative_to(OUT_ROOT))
    bf = orig.replace("_hadrons_jet_frag.h5", "_hadrons_bulk_bg.h5")
    with h5py.File(path) as h:
        n = int(h.attrs["nunits_written"])
        u = {k: h[f"units/{k}"][:n] for k in ("glauber_ncoll", "glauber_b", "pthat_bin", "has_medium", "seed", "pthat")}
        sig = np.asarray(h.attrs["pthat_bin_sigma_gen"], float)
        nacc = np.asarray(h.attrs["pthat_bin_n_accepted"], float)
        bins = [tuple(map(float, w)) for w in np.asarray(h.attrs["pthat_bins"]).reshape(-1, 2)]
        pa, ca, pid = B._units(h, n, charged=False, with_pid=True)
        so, uo = h["hadrons/sample_offsets"][:], h["hadrons/unit_offsets"][: n + 1]
        pst = h["hadrons/pstat"][: int(so[uo[-1]])] if "hadrons/pstat" in h else np.zeros(len(pa), int)
        neg = negative_partons(h, n)
        hard = hard_partons(h, n)
    nsu, ns, unit_of, cs, samp, first = samples_of(so, uo, n)
    try:
        with h5py.File(bf) as h:
            pb, cb = B._units(h, n)                                   # bulk, charged only
    except OSError:
        pb, cb = np.zeros((0, 4)), np.zeros(n, int)
    ev_u = np.repeat(np.arange(n), ca)                                # unit of each hadron
    ev_b = np.repeat(np.arange(n), cb)
    ch = np.isin(np.abs(pid), B.charged_ids())
    hneg = pst == -1
    hadron_holes = bool(hneg.any())
    pos = ~hneg
    sgn = np.where(hneg, -1.0, 1.0)
    # estimators: bulk + the hard event's charged hadrons (as oo_bias.load); of the original file with EST_ROOT
    est = {k: v.astype(float) for k, v in B.counts_of(pb, cb).items()}
    if orig != path:
        with h5py.File(orig) as h:
            po_, co_, pido_ = B._units(h, n, charged=False, with_pid=True)
            fo_ = samples_of(h["hadrons/sample_offsets"][:], h["hadrons/unit_offsets"][: n + 1], n)[5]
        cho = np.isin(np.abs(pido_), B.charged_ids()) & fo_
        evo = np.repeat(np.arange(n), co_)
        for k, v in B.counts_of(po_[cho], np.bincount(evo[cho], minlength=n)).items():
            est[k] += v
    else:
        m = ch & pos & first
        for k, v in B.counts_of(pa[m], np.bincount(ev_u[m], minlength=n)).items():
            est[k] += v
    # from here on the events are the samples (one per unit without oversamples)
    n_units, n = n, ns
    ev_a = samp
    unit_keys = {"neg_n": np.bincount(neg[:, 0].astype(int), minlength=n_units).astype(float)
                 if len(neg) else np.zeros(n_units)}
    if ns != n_units:                       # the unit's holes + partners, hard partons and bulk in each sample
        if len(neg):
            i, sj = rows_to_samples(neg[:, 0], nsu, uo)
            neg = neg[i]
            neg[:, 0] = sj
        if len(hard):
            i, sj = rows_to_samples(hard[:, 0], nsu, uo)
            hard = hard[i]
            hard[:, 0] = sj
        if len(pb):                         # only the h-h associated range is needed per sample
            ptb_, etab_, _ = _kin(pb)
            mb_ = (ptb_ > ASSOC[0]) & (ptb_ < ASSOC[1]) & (np.abs(etab_) < TRIG_ETA)
            i, ev_b = rows_to_samples(ev_b[mb_], nsu, uo)
            pb = pb[mb_][i]
    est["mbd"] = est["mbd_neg"] + est["mbd_pos"]
    pt, eta, phi = _kin(pa)
    vis = ~np.isin(np.abs(pid), INVISIBLE) & (pt > CONST_PTMIN)
    per_unit = {"ncoll": u["glauber_ncoll"].astype(float), "b": u["glauber_b"].astype(float),
                "win": u["pthat_bin"].astype(int), "has_medium": u["has_medium"].astype(int),
                "pthat": u["pthat"].astype(float),
                "n_epd": est["epd"], "n_mbd": est["mbd"], "n_mid10": est["mid10"]}
    out = {}
    # inclusive charged hadrons
    m = ch & (np.abs(eta) < HAD_ETA)
    out["had"] = _hist(ev_a[m], pt[m], HAD_EDGES, n, sgn[m] if hadron_holes else None)
    def neg_rows(sel):
        return np.column_stack([ev_a[sel], pa[sel]]) if sel.any() else np.zeros((0, 5))
    # sPHENIX full jets, with the holes and partner copies in the cone subtracted at parton level (iColored)
    # or the hole hadrons at hadron level (Colorless)
    m = vis & (np.abs(eta) < SPH_ETA + SPH_R + 0.1)
    je, jpt, jeta, jphi, jx, jy = _jets(pa[m & pos], ev_a[m & pos], n, SPH_R, None, ptmin=SPH_EDGES[0], full=True)
    jpt_sub = subtract_negatives(je, jeta, jphi, jx, jy, neg_rows(m & hneg) if hadron_holes else neg, SPH_R)
    k = np.abs(jeta) < SPH_ETA
    out["sph_jet"] = _hist(je[k], jpt_sub[k], SPH_EDGES, n)
    out["sph_jet_raw"] = _hist(je[k], jpt[k], SPH_EDGES, n)
    fl = jet_flavour(je[k], jeta[k], jphi[k], hard, SPH_R)
    out["sph_jet_q"] = _hist(je[k][fl == 1], jpt_sub[k][fl == 1], SPH_EDGES, n)
    out["sph_jet_g"] = _hist(je[k][fl == 2], jpt_sub[k][fl == 2], SPH_EDGES, n)
    out["neg_pt_in_jets"] = np.bincount(je[k], weights=(jpt - jpt_sub)[k], minlength=n) if len(je) else np.zeros(n)
    # STAR charged jets (inclusive) and the charged particles for the h+jet
    mc = ch & (pt > CONST_PTMIN) & (np.abs(eta) < TRIG_ETA)
    cjets = {}
    for R in CH_RS:
        je, jpt, jeta, jphi, jx, jy = _jets(pa[mc & pos], ev_a[mc & pos], n, R, None, ptmin=CH_EDGES[0] * 0.5, full=True)
        # the holes + partners subtracted with the charged fraction 2/3 (inclusive and recoil jets); with hole
        # hadrons the charged ones, exactly
        jpt_sub = (subtract_negatives(je, jeta, jphi, jx, jy, neg_rows(mc & hneg), R) if hadron_holes
                   else subtract_negatives(je, jeta, jphi, jx, jy, neg, R, CHARGED_FRAC))
        cjets[R] = (je, jpt, jeta, jphi, jpt_sub)
        k = np.abs(jeta) < 1.0 - R
        out[f"ch_jet_R{R}"] = _hist(je[k], jpt_sub[k], CH_EDGES, n)
        out[f"ch_jet_R{R}_raw"] = _hist(je[k], jpt[k], CH_EDGES, n)
    # one random trigger per event (seeded per event: reproducible)
    tm = ch & pos & (pt > TRIG[0]) & (pt < TRIG[1]) & (np.abs(eta) < TRIG_ETA)
    tphi = np.full(n, np.nan)
    tidx = np.flatnonzero(tm)
    if len(tidx):
        tev = ev_a[tidx]
        rng = np.random.default_rng(int(u["seed"][0]) if n else 0)
        order = rng.permutation(len(tidx))
        first = {}
        for j in order:                                               # a random trigger per event
            first.setdefault(int(tev[j]), int(tidx[j]))
        sel = np.array(list(first.values()), int)
        tphi[ev_a[sel]] = phi[sel]
        trig_row = np.full(n, -1)
        trig_row[ev_a[sel]] = sel
    else:
        trig_row = np.full(n, -1)
    out["ntrig"] = np.isfinite(tphi).astype(float)
    # h-h: associated from the hard event and the bulk, the trigger itself excluded
    pbt, etab, phib = _kin(pb) if len(pb) else (np.zeros(0),) * 3
    a_ev = np.concatenate([ev_a, ev_b])
    a_pt = np.concatenate([pt, pbt])
    a_eta = np.concatenate([eta, etab])
    a_phi = np.concatenate([phi, phib])
    a_ch = np.concatenate([ch, np.ones(len(pb), bool)])
    a_sg = np.concatenate([sgn, np.ones(len(pb))])                    # hole hadrons count -1
    a_row = np.concatenate([np.arange(len(pa)), np.full(len(pb), -2)])
    am = a_ch & (a_pt > ASSOC[0]) & (a_pt < ASSOC[1]) & (np.abs(a_eta) < TRIG_ETA) & np.isfinite(tphi[a_ev]) \
        & (a_row != trig_row[a_ev])
    d = _dphi(a_phi[am], tphi[a_ev[am]])
    ev_am = a_ev[am]
    sg_am = a_sg[am]
    out["hh_near"] = np.bincount(ev_am[d < NEAR], weights=sg_am[d < NEAR], minlength=n).astype(float)
    out["hh_recoil"] = np.bincount(ev_am[d > np.pi - RECOIL], weights=sg_am[d > np.pi - RECOIL], minlength=n).astype(float)
    bgm = (d > BG[0]) & (d < BG[1])
    out["hh_bg"] = np.bincount(ev_am[bgm], weights=sg_am[bgm], minlength=n).astype(float)
    # h+jet: recoil charged jets
    for R in CH_RS:
        je, jpt, jeta, jphi, jpt_sub = cjets[R]
        k = np.isfinite(tphi[je]) & (np.abs(jeta) < TRIG_ETA - R) if len(je) else np.zeros(0, bool)
        if len(je):
            k &= np.abs(_dphi(jphi, tphi[je]) - np.pi) < HJET_DPHI
        out[f"hjet_R{R}"] = _hist(je[k], jpt_sub[k], HJET_EDGES, n)
        out[f"hjet_R{R}_raw"] = _hist(je[k], jpt[k], HJET_EDGES, n)
    if ns != n_units:                                                 # samples -> events
        out = {k: samples_to_units(v, unit_of, cs, n_units) for k, v in out.items()}
    return bins, sig, nacc, {**per_unit, **out, **unit_keys}


# ── per family ────────────────────────────────────────────────────────────────────────────────
#: extra split points in pTHat (GeV) for the closure test of the piecewise stitching (no effect on
#: the weights without overlapping samples; set by --split-at)
SPLIT_AT = ()


def stitch_pieces(wins, win, pthat, sigma, extra=()):
    """Weights for samples whose pTHat windows may overlap (3e: narrow windows inside wide ones).
    Every window is cut at all window edges (and at extra), into pieces j.  A window k with N_k
    events estimates the piece's cross section as sigma_k n_kj / N_k (n_kj of its events fall in j);
    the windows covering j are combined weighted by n_kj, and each event in j gets
    sigma_j / n_j (n_j = all events in j).  Without overlaps every event keeps sigma_k / N_k exactly.
    Returns the weights, the piece index per event, the pieces, and per event the rule group: the
    widest window containing its piece (for the stitching rule and the class sums, as before)."""
    wins = [tuple(map(float, w)) for w in wins]
    cuts = sorted({x for w in wins for x in w} | {float(x) for x in extra
                                                    if any(lo < x < hi for lo, hi in wins)})
    pieces = [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if any(lo <= a and b <= hi for lo, hi in wins)]
    pl = np.array([a for a, _ in pieces])
    j = np.clip(np.searchsorted(pl, pthat, side="right") - 1, 0, len(pieces) - 1)
    # an event's piece must lie inside its own window (pTHat exactly on an edge)
    lo_w = np.array([wins[k][0] for k in win]); hi_w = np.array([wins[k][1] for k in win])
    j = np.where(np.array([pieces[x][0] for x in j]) < lo_w, j + 1, j)
    j = np.where(np.array([pieces[x][1] for x in j]) > hi_w, j - 1, j)
    nk = np.bincount(win, minlength=len(wins)).astype(float)
    nkj = np.zeros((len(wins), len(pieces)))
    np.add.at(nkj, (win, j), 1.0)
    est = sigma[:, None] * nkj / np.maximum(nk, 1)[:, None]          # window k's estimate of sigma_j
    nj = nkj.sum(0)
    sig_j = np.where(nj > 0, (nkj * est).sum(0) / np.maximum(nj, 1), 0.0)
    w = (sig_j / np.maximum(nj, 1))[j]
    wide = [k for k, (lo, hi) in enumerate(wins)
            if not any((l2 <= lo and hi <= h2) and (l2, h2) != (lo, hi) for l2, h2 in wins)]
    group_of_piece = np.array([next(g for g in wide if wins[g][0] <= a and b <= wins[g][1]) for a, b in pieces])
    return w, j, pieces, group_of_piece[j], [wins[g] for g in wide], wide


def load_family(fam, j):
    """All finished files of the family's window sub-samples, stitched: per-event arrays with the
    rule-group (window) index and the weight; overlapping narrow windows (3e) combined piecewise
    (stitch_pieces)."""
    try:
        comps, windows = J.components(fam)
    except SystemExit:
        return None
    if PHASE3_ONLY:
        hij = {f"{s}_{B._w(lo, hi)}" for lo, hi in getattr(B, "HIJET_WINDOWS", {}) for s in FAMILIES}
        comps = [(nm, w) for nm, w in comps if WITH_3E or nm not in hij]
        seeds = {nm: {s for _, s, _, _ in B.jobs(nm)} for nm, _ in comps}
        files = [f for nm, _ in comps for f in J.finished_files(nm)
                 if int(f.rsplit("_seed", 1)[1].split("_")[0]) in seeds[nm]]
    else:
        files = [f for name, _ in comps for f in J.finished_files(name)]
    if not files:
        return None
    with ProcessPoolExecutor(max_workers=j) as ex:
        res = list(ex.map(file_obs, files))
    wl = sorted({w for bins, *_ in res for w in bins})
    sig_n, nacc = np.zeros(len(wl)), np.zeros(len(wl))
    parts = []
    for bins, sig, na, o in res:
        g = np.array([wl.index(w) for w in bins])
        sig_n[g] += sig * na
        nacc[g] += na
        o = dict(o)
        o["win"] = g[o["win"]]
        parts.append(o)
    ev = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    sigma = sig_n / np.maximum(nacc, 1)
    w, piece, pieces, group, groups, wide = stitch_pieces(wl, ev["win"], ev["pthat"], sigma, SPLIT_AT)
    ev["w"] = w
    ev["piece"] = piece
    ev["pieces"] = np.asarray(pieces)
    pos = np.full(len(wl), -1)
    pos[wide] = np.arange(len(wide))
    ev["win"] = pos[group]                                              # index into the rule groups
    ev["windows"] = np.asarray(groups)
    ev["files"] = np.array([len(files)])
    return ev


def mb_reference():
    rec = B.load(MB_SAMPLE)
    return {"n_epd": rec["n_epd"].astype(float), "n_mbd": rec["n_mbd"].astype(float),
            "n_mid10": rec["n_mid10"].astype(float), "b": rec["glauber_b"].astype(float),
            "ncoll": rec["glauber_ncoll"].astype(float)}


def classify(mb, ev, est, rng):
    key, cls_def = ESTIMATORS[est]
    if key == "b":
        cm, cj = J.classes(-mb["b"], -ev["b"], cls_def, rng, jitter=False)
    else:
        cm, cj = J.classes(mb[key], ev[key], cls_def, rng)
    return cm, cj


# ── observables ──────────────────────────────────────────────────────────────────────────────
HIST_OBS = ("had", "sph_jet", "sph_jet_raw", "ch_jet_R0.2", "ch_jet_R0.5", "ch_jet_R0.2_raw", "ch_jet_R0.5_raw",
            "sph_jet_q", "sph_jet_g")
#: --phase3: only the jobs of phase 3 (each sample's defined size, no NAME:EVENTS extensions such as
#: 3b, no 3e samples), so that results stay comparable while later add-ons are still arriving
PHASE3_ONLY = False
#: with PHASE3_ONLY: keep the 3e samples (HIJET_WINDOWS) too (--with-3e), still without the 3b extensions
WITH_3E = False


def global_windows(fams):
    """The union of the families' pTHat windows and, per family, its window index -> global index
    (families can have different windows finished)."""
    wl = sorted({tuple(map(float, w)) for ev in fams.values() for w in ev["windows"]})
    return wl, {f: np.array([wl.index(tuple(map(float, w))) for w in ev["windows"]]) for f, ev in fams.items()}


def edges_of(key):
    return HAD_EDGES if key == "had" else SPH_EDGES if key.startswith("sph_jet") else CH_EDGES


def class_hists(ev, cj, nc, key, oo, wmap=None, nw=None):
    """hist histograms window x class x bin of observable key: the weighted one (Weight storage; one
    entry per event and bin with weight w x count, so the variance is the sum over events of (w x
    count)^2) and the raw entries.  Windows mapped to a global list with wmap; weights N_coll x
    sigma_k / N_k for O+O (oo), sigma_k / N_k for p+p."""
    h = ev[key]
    W = ev["w"] * (ev["ncoll"] if oo else 1.0)
    win = ev["win"] if wmap is None else wmap[ev["win"]]
    nw = len(ev["windows"]) if nw is None else nw
    edges = edges_of(key)
    axes = (hist.axis.Integer(0, nw, name="window", label="pTHat window", underflow=False, overflow=False),
            hist.axis.Integer(0, nc, name="cls", label="class", underflow=False, overflow=False),
            hist.axis.Variable(edges, name="x", label=key, underflow=False, overflow=False))
    hw = hist.Hist(*axes, storage=hist.storage.Weight())
    hn = hist.Hist(*axes, storage=hist.storage.Double())
    e, b = np.nonzero(h)
    xc = 0.5 * (edges[b] + edges[b + 1])
    hw.fill(window=win[e], cls=np.asarray(cj)[e], x=xc, weight=W[e] * h[e, b])
    hn.fill(window=win[e], cls=np.asarray(cj)[e], x=xc, weight=h[e, b])
    return hw, hn


def class_hist(ev, cj, nc, key, oo, wmap=None, nw=None):
    """(S, S2, N) arrays window x class x bin: the weighted sums, the sums of squared weights and the
    raw entries, from class_hists."""
    hw, hn = class_hists(ev, cj, nc, key, oo, wmap, nw)
    return hw.values(), hw.variances(), hn.values()


def window_use(pooled):
    """The stitching rule: window k enters bin b where every family's raw entries, pooled over the
    classes, reach MIN_ENTRIES (pooled: arrays window x bin, one per family)."""
    use = np.ones(pooled[0].shape, bool)
    for N in pooled:
        use &= N >= MIN_ENTRIES
    return use


def stitch_diag(yields, use):
    """Per family the share of its yield (window x bin, pooled over the classes) in the windows left
    out, and the largest share of the lowest generated window (index 0) over the families."""
    drop, low = {}, np.zeros(use.shape[1:] if use.ndim > 1 else ())
    for f, Y in yields.items():
        tot = Y.sum(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            drop[f] = np.where(tot > 0, np.where(use, 0, Y).sum(0) / tot, 0.0)
            low = np.maximum(low, np.where(tot > 0, Y[0] / tot, 0.0))
    flag = np.zeros_like(low, bool)
    for d in drop.values():
        flag |= d > DROP_FLAG
    return {"dropped_share": {f: np.round(d, 4).tolist() for f, d in drop.items()},
            "lowest_window_share": np.round(low, 4).tolist(), "complete": (low <= INCOMPLETE_SHARE).tolist(),
            "flag_dropped": flag.tolist()}


def stitched_raa(fams, pp, mb, est, key, rng):
    """R_AA per class for the O+O families in fams against the p+p family pp, the same windows
    (the min-entries rule over all classes of all O+O families and the p+p total)."""
    _, cls_def = ESTIMATORS[est]
    nc = len(cls_def)
    width = np.array([(hi - lo) / 100.0 for lo, hi in cls_def])
    wl, wmap = global_windows(fams)
    sums = {}
    for f, ev in fams.items():
        cm, cj = classify(mb, ev, est, rng)
        sums[f] = class_hist(ev, cj, nc, key, True, wmap[f], len(wl))
    # <N_coll> per class of minimum bias
    cm, _ = classify(mb, fams[next(iter(fams))], est, rng)
    nmb = np.array([mb["ncoll"][cm == c].mean() if (cm == c).any() else np.nan for c in range(nc)])
    nbins = next(iter(sums.values()))[0].shape[2]
    pooled = [N.sum(1) for S, S2, N in sums.values()]                 # window x bin, all classes
    yields = {f: S.sum(1) for f, (S, S2, N) in sums.items()}
    if pp is None:
        # no p+p yet: unit reference -> "raa" is the N_coll-scaled yield per class; R_CP and
        # med/vac are unaffected (the reference cancels), R_AA itself is not available
        use = window_use(pooled)
        ppS, ppE2 = np.ones(nbins), np.zeros(nbins)
    else:
        Sp, S2p, Np = class_hist(pp, np.zeros(len(pp["w"]), int), 1, key, False)
        # windows: the O+O families' list mapped onto p+p's (a window p+p lacks is left out)
        pl = [tuple(map(float, w)) for w in pp["windows"]]
        pidx = np.array([pl.index(w) if w in pl else -1 for w in wl])
        have = (pidx >= 0)[:, None]
        k = np.clip(pidx, 0, None)
        use = window_use(pooled + [np.where(have, Np[k, 0, :], 0)])
        yields["pp_ref"] = np.where(have, Sp[k, 0, :], 0)
        ppS = np.where(use, Sp[k, 0, :], 0).sum(0)
        ppE2 = np.where(use, S2p[k, 0, :], 0).sum(0)
    out = {"classes": cls_def, "ncoll_mb": nmb.tolist(), "pp": ppS.tolist(), "windows_used": use.sum(0).tolist(),
           "pp_missing": pp is None, "stitch": stitch_diag(yields, use)}
    for f, (S, S2, N) in sums.items():
        A = np.where(use[:, None, :], S, 0).sum(0)                    # (class, bin)
        A2 = np.where(use[:, None, :], S2, 0).sum(0)
        denom = width[:, None] * nmb[:, None] * ppS[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            raa = A / denom
            err = raa * np.sqrt(A2 / A ** 2 + ppE2[None, :] / ppS[None, :] ** 2)
            # 0-100 %: all classes, <N_coll> of all minimum bias
            Aall, A2all = A.sum(0), A2.sum(0)
            raa_mb = Aall / (mb["ncoll"].mean() * ppS)
            err_mb = raa_mb * np.sqrt(A2all / Aall ** 2 + ppE2 / ppS ** 2)
        out[f] = {"raa": raa.tolist(), "err": err.tolist(), "raa_mb": raa_mb.tolist(), "err_mb": err_mb.tolist(),
                  "raw": np.where(use[:, None, :], N, 0).sum(0).tolist()}
    return out


def ratio(a, ea, b, eb):
    a, ea, b, eb = map(np.asarray, (a, ea, b, eb))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = a / b
        return r, np.abs(r) * np.sqrt((ea / a) ** 2 + (eb / b) ** 2)


def per_trigger(ev, sel, oo):
    """Per-trigger yields (near, recoil h-h with flat background; h+jet per bin and 8-19 GeV) of the
    events sel, weights N_coll x sigma_k / N_k; errors by linearising the ratio of sums."""
    W = ev["w"] * (ev["ncoll"] if oo else 1.0)
    W = W[sel]
    t = ev["ntrig"][sel]
    T = (W * t).sum()
    bg = ev["hh_bg"][sel] / (2 * (BG[1] - BG[0]))                    # per radian, both sides
    out = {"n_trig_raw": float(t.sum()), "T": float(T)}

    def yld(x):
        if T <= 0:
            return float("nan"), float("nan")
        r = (W * x).sum() / T
        e = np.sqrt(((W * (x - r * t)) ** 2).sum()) / T
        return float(r), float(e)
    out["near"] = yld(ev["hh_near"][sel] - bg * 2 * NEAR)
    out["recoil"] = yld(ev["hh_recoil"][sel] - bg * 2 * RECOIL)
    for R in CH_RS:
        for tag in ("", "_raw"):
            h = ev[f"hjet_R{R}{tag}"][sel]
            out[f"hjet_R{R}{tag}"] = [yld(h[:, i]) for i in range(h.shape[1])]
            inr = (HJET_EDGES[:-1] >= HJET_INT[0]) & (HJET_EDGES[1:] <= HJET_INT[1])
            out[f"hjet_R{R}{tag}_8_19"] = yld(h[:, inr].sum(1))
    return out


def icp(fams, pp, mb, rng):
    """I_CP in STAR's EPD classes (each class over 40-60 %) and I_AA (0-100 % O+O over p+p).
    Triggers from a pTHat window count only where that window has at least MIN_ENTRIES triggers in
    every O+O family and in p+p, pooled over the classes (the stitching rule; a low window's rare
    7-30 GeV trigger would otherwise carry a huge weight)."""
    _, cls_def = ESTIMATORS["epd"]
    ref = cls_def.index(ICP_REF)
    res = {"classes": cls_def, "ref": list(ICP_REF)}
    cls = {f: classify(mb, ev, "epd", rng)[1] for f, ev in fams.items()}
    wl, wmap = global_windows(fams)
    pooled, tw = [], {}
    for f, ev in fams.items():
        pooled.append(np.bincount(wmap[f][ev["win"]], weights=ev["ntrig"], minlength=len(wl)))
        tw[f] = np.bincount(wmap[f][ev["win"]], weights=ev["w"] * ev["ncoll"] * ev["ntrig"], minlength=len(wl))
    if pp is not None:
        pl = [tuple(map(float, w)) for w in pp["windows"]]
        npp = np.bincount(pp["win"], weights=pp["ntrig"], minlength=len(pl))
        wpp = np.bincount(pp["win"], weights=pp["w"] * pp["ntrig"], minlength=len(pl))
        pooled.append(np.array([npp[pl.index(w)] if w in pl else 0 for w in wl]))
        tw["pp_ref"] = np.array([wpp[pl.index(w)] if w in pl else 0 for w in wl])
    ok = window_use(pooled)
    if pp is not None:
        pp_ok = np.array([w in [x for x, k in zip(wl, ok) if k] for w in pl])
    res["windows_used"] = [list(map(float, w)) for w, k in zip(wl, ok) if k]
    res["stitch"] = stitch_diag(tw, ok)
    res["stitch"]["trigger_share_per_window"] = {f: np.round(t / max(t.sum(), 1e-300), 4).tolist() for f, t in tw.items()}
    res["stitch"]["windows"] = [list(map(float, w)) for w in wl]
    ppy = per_trigger(pp, pp_ok[pp["win"]], False) if pp is not None else None
    res["pp"] = ppy
    for f, ev in fams.items():
        cj = cls[f]
        inw = ok[wmap[f][ev["win"]]]
        ys = [per_trigger(ev, (cj == c) & inw, True) for c in range(len(cls_def))]
        yall = per_trigger(ev, inw, True)
        r = {"yields": ys, "mb": yall, "icp": {}, "iaa": {}}
        for obs in ("near", "recoil", "hjet_R0.2_8_19", "hjet_R0.5_8_19", "hjet_R0.2_raw_8_19", "hjet_R0.5_raw_8_19"):
            r["icp"][obs] = [list(map(float, ratio(ys[c][obs][0], ys[c][obs][1], ys[ref][obs][0], ys[ref][obs][1])))
                             for c in range(len(cls_def))]
            if ppy is not None:
                r["iaa"][obs] = list(map(float, ratio(yall[obs][0], yall[obs][1], ppy[obs][0], ppy[obs][1])))
        # recoil-jet I_CP per jet-pT bin (STAR's Fig. 6: 0-10 % over 40-60 %)
        for R in CH_RS:
            for tag in ("", "_raw"):
                k = f"hjet_R{R}{tag}"
                r["icp_vs_pt_" + k] = [list(map(float, ratio(ys[0][k][i][0], ys[0][k][i][1], ys[ref][k][i][0], ys[ref][k][i][1])))
                                       for i in range(len(HJET_EDGES) - 1)]
        res[f] = r
    return res


def load_data():
    """STAR's HEPData tables (refs/star_oo_icp, ins3145466: Figs. 3, 5, 6) and sPHENIX's preliminary
    jet R_CP / R_AA (refs/sphenix_oo_jet, extracted from the note's vector figures)."""
    import yaml
    out = {}
    d = HERE / "refs" / "star_oo_icp"
    if (d / "figure3_1.yaml").exists():
        def tab(name):
            t = yaml.safe_load((d / name).read_text())
            iv = t["independent_variables"][0]["values"]
            cols = []
            for dv in t["dependent_variables"]:
                q = {x["name"]: x.get("value") for x in dv.get("qualifiers", [])}
                vals = [(v["value"], *[e["symerror"] for e in v["errors"]]) for v in dv["values"]]
                cols.append({"name": dv["header"]["name"], "qual": q, "values": vals})
            return {"x": [v.get("value") or [v["low"], v["high"]] for v in iv], "cols": cols}
        out["star"] = {"icp_hh_vs_ea": tab("figure3_1.yaml"), "icp_hjet_vs_ea": tab("figure3_2.yaml"),
                       "icp_hjet_vs_pt": tab("figure6.yaml"), "hjet_spectra_R0.2": tab("figure5_1.yaml"),
                       "hjet_spectra_R0.5": tab("figure5_2.yaml"),
                       "source": "STAR, arXiv:2604.13935, HEPData ins3145466 v1 (values: value, stat, syst)"}
    f = HERE / "refs" / "sphenix_oo_jet" / "sphenix_oo_jet_prelim.json"
    if f.exists():
        out["sphenix"] = json.loads(f.read_text())
    return out


def pp_ratio(num, den, key):
    """Inclusive spectrum ratio of two p+p families (pp_med / pp_ref), the same stitching rule."""
    Sa, S2a, Na = class_hist(num, np.zeros(len(num["w"]), int), 1, key, False)
    Sb, S2b, Nb = class_hist(den, np.zeros(len(den["w"]), int), 1, key, False)
    la, lb = [tuple(w) for w in num["windows"]], [tuple(w) for w in den["windows"]]
    common = [w for w in la if w in lb]
    ia, ib = [la.index(w) for w in common], [lb.index(w) for w in common]
    use = window_use([Na[ia, 0, :], Nb[ib, 0, :]])
    A, A2 = np.where(use, Sa[ia, 0, :], 0).sum(0), np.where(use, S2a[ia, 0, :], 0).sum(0)
    Bs, B2 = np.where(use, Sb[ib, 0, :], 0).sum(0), np.where(use, S2b[ib, 0, :], 0).sum(0)
    r, e = ratio(A, np.sqrt(A2), Bs, np.sqrt(B2))
    return {"ratio": r.tolist(), "err": e.tolist(), "windows_used": use.sum(0).tolist(),
            "stitch": stitch_diag({"pp_med": Sa[ia, 0, :], "pp_ref": Sb[ib, 0, :]}, use)}


def qs_check(pp_ref, pp_med):
    """Matter's vacuum cutoff (QS 0.9 with in_vac 0 vs Q0 2 with in_vac 1): pp_med's events without a
    medium over pp_ref's, per window (inclusive hadrons and full jets, weights sigma_k / N_k)."""
    out = {}
    for key in ("had", "sph_jet", "ch_jet_R0.5"):
        rows = []
        for k, w in enumerate(pp_ref["windows"]):
            if tuple(w) not in [tuple(x) for x in pp_med["windows"]]:
                continue
            km = [tuple(x) for x in pp_med["windows"]].index(tuple(w))
            a = (pp_med["win"] == km) & (pp_med["has_medium"] == 0)
            b = (pp_ref["win"] == k) & (pp_ref["has_medium"] == 0)
            if a.sum() < 50 or b.sum() < 50:
                continue
            ha, hb = pp_med[key][a].mean(0), pp_ref[key][b].mean(0)
            ea, eb = pp_med[key][a].std(0) / np.sqrt(a.sum()), pp_ref[key][b].std(0) / np.sqrt(b.sum())
            r, e = ratio(ha, ea, hb, eb)
            tot, etot = ratio(pp_med[key][a].sum(1).mean(), pp_med[key][a].sum(1).std() / np.sqrt(a.sum()),
                              pp_ref[key][b].sum(1).mean(), pp_ref[key][b].sum(1).std() / np.sqrt(b.sum()))
            rows.append({"window": list(map(float, w)), "n_med": int(a.sum()), "n_ref": int(b.sum()),
                         "ratio": r.tolist(), "err": e.tolist(), "total": float(tot), "total_err": float(etot)})
        out[key] = rows
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/data/OO-RHIC_phys", help="root of the phase-3 families")
    ap.add_argument("--mb-root", default=None, help="root holding phys_mb (default --out)")
    ap.add_argument("-j", type=int, default=16)
    ap.add_argument("--cached", action="store_true")
    ap.add_argument("--phase3", action="store_true", help="only the jobs of phase 3 (no 3b extensions, no 3e "
                    "samples); caches and the default JSON get the suffix _phase3")
    ap.add_argument("--with-3e", action="store_true", help="with --phase3: also the 3e samples (narrow high "
                    "pTHat windows); outputs with the suffix _phase3e")
    ap.add_argument("--split-at", default="", help="extra pTHat split points, comma-separated (closure test of the "
                    "piecewise stitching; re-reads the files and writes no cache)")
    ap.add_argument("--json", default=str(HERE / "results" / "oo_analysis_rhic.json"))
    ap.add_argument("--est-root", default=None, dest="est_root",
                    help="re-hadronized samples under --out: take the centrality estimators and the bulk from "
                         "the original files under this root (every event keeps its class)")
    ap.add_argument("--tag", default="", help="suffix of the caches and of the default --json (e.g. colorless)")
    a = ap.parse_args(argv)
    global SPLIT_AT, PHASE3_ONLY, WITH_3E, EST_ROOT, OUT_ROOT
    EST_ROOT = a.est_root
    OUT_ROOT = str(Path(a.out).resolve()) if a.est_root else None
    SPLIT_AT = tuple(float(x) for x in a.split_at.split(",") if x)
    PHASE3_ONLY = a.phase3
    WITH_3E = a.phase3 and a.with_3e
    sfx = (("_phase3e" if WITH_3E else "_phase3") if a.phase3 else "") + (f"_{a.tag}" if a.tag else "")
    if (a.phase3 or a.tag) and a.json == str(HERE / "results" / "oo_analysis_rhic.json"):
        a.json = str(HERE / "results" / f"oo_analysis_rhic{sfx}.json")
    if SPLIT_AT:
        a.cached = False
    (HERE / "results").mkdir(exist_ok=True)
    fams = {}
    for f in FAMILIES:
        cache = HERE / "results" / f"oo_analysis_rhic_{f}{sfx}.npz"
        if a.cached and cache.exists():
            ev = dict(np.load(cache))
            if str(ev.get("root", "")) == str(Path(a.out).resolve()):  # only a cache of this root
                fams[f] = ev
                continue
            print(f"oo_analysis_rhic.py: {cache.name} is of another root ({ev.get('root', '?')}), re-reading")
        B.OUT = Path(a.out)
        ev = load_family(f, a.j)
        if ev is None:
            print(f"oo_analysis_rhic.py: no finished files for {f} under {a.out}")
            cache.unlink(missing_ok=True)                                # no stale cache of another root
            continue
        ev["root"] = np.array(str(Path(a.out).resolve()))
        if not SPLIT_AT:
            np.savez(cache, **ev)
        fams[f] = ev
        print(f"{f}: {len(ev['w'])} events, {int(ev['files'][0])} files, windows "
              + " ".join(f"{lo:g}-{hi:g}" for lo, hi in ev["windows"]))
    B.OUT = Path(a.mb_root or a.out)
    mb = mb_reference()
    print(f"{MB_SAMPLE}: {len(mb['b'])} events")
    oo = {f: fams[f] for f in OO_FAMILIES if f in fams}
    res = {"inputs": {"out": a.out, "mb_root": str(B.OUT),
                      "events": {f: int(len(ev["w"])) for f, ev in fams.items()}, "mb_events": int(len(mb["b"]))},
           "edges": {"had": HAD_EDGES.tolist(), "sph_jet": SPH_EDGES.tolist(), "ch_jet": CH_EDGES.tolist(),
                     "hjet": HJET_EDGES.tolist()},
           "star_icp_data": STAR_ICP, "data": load_data(), "raa": {}, "icp": {}}
    rng = np.random.default_rng(11)
    res["inputs"]["pp_missing"] = "pp_ref" not in fams
    if oo:
        for est in ESTIMATORS:
            res["raa"][est] = {key: stitched_raa(oo, fams.get("pp_ref"), mb, est, key, rng) for key in HIST_OBS}
        res["icp"] = icp(oo, fams.get("pp_ref"), mb, rng)
    if "pp_med" in fams and "pp_ref" in fams:
        res["qs_check"] = qs_check(fams["pp_ref"], fams["pp_med"])
        res["pp_med_over_ref"] = {key: pp_ratio(fams["pp_med"], fams["pp_ref"], key) for key in HIST_OBS}
    Path(a.json).write_text(json.dumps(res, indent=1, default=float))
    print(f"wrote {a.json}")
    # a short summary
    if "epd" in res["raa"]:
        for f in OO_FAMILIES:
            if f not in res["raa"]["epd"]["had"]:
                continue
            r = res["raa"]["epd"]["had"][f]
            print(f"{f} charged-hadron " + ("N_coll-scaled yield (no p+p yet)" if res["inputs"]["pp_missing"] else "R_AA")
                  + " (EPD), classes x pT bins:")
            for c, (lo, hi) in enumerate(res["raa"]["epd"]["had"]["classes"]):
                print(f"  {lo:3d}-{hi:<3d}% " + " ".join(f"{x:5.2f}" for x in r["raa"][c]))
    if "mbd" in res["raa"]:
        print("stitching (pooled rule; the same for every estimator): per bin the largest dropped share over "
              f"the families, '*' = incomplete (lowest window > {INCOMPLETE_SHARE:.0%})")
        for key in ("had", "sph_jet", "ch_jet_R0.5"):
            st = res["raa"]["mbd"][key]["stitch"]
            d = np.max(np.array(list(st["dropped_share"].values())), axis=0)
            print(f"  {key:12s} " + " ".join(f"{x:.3f}{'*' if not c else ' '}" for x, c in zip(d, st["complete"])))
    if res["icp"] and "stitch" in res["icp"]:
        st = res["icp"]["stitch"]
        print("  triggers: windows used " + " ".join(f"{lo:g}-{hi:g}" for lo, hi in res["icp"]["windows_used"])
              + "; dropped share " + ", ".join(f"{f} {d:.3f}" for f, d in st["dropped_share"].items()))
    if res["icp"]:
        for f in OO_FAMILIES:
            if f in res["icp"]:
                ic = res["icp"][f]["icp"]
                print(f"{f} I_CP 0-10/40-60 (EPD): h-h near {ic['near'][0][0]:.3f}+-{ic['near'][0][1]:.3f}, "
                      f"recoil {ic['recoil'][0][0]:.3f}+-{ic['recoil'][0][1]:.3f}, "
                      f"h+jet R0.5 {ic['hjet_R0.5_8_19'][0][0]:.3f}+-{ic['hjet_R0.5_8_19'][0][1]:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
