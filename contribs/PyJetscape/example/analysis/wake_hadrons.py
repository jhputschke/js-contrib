#!/usr/bin/env python3
"""Hadron-level jet and wake observables from the hadron files of jet productions.

Goes once through the ``<stem>_hadrons_{bulk_jet,bulk_bg,jet_frag}.h5`` that
``hadronize.py`` wrote for each production file and writes per-event histograms into one
HDF5 file. ``wake_hadrons.ipynb`` only reads that file (and ``wake_observables.h5`` for
the parton- and hydro-level side), so a new figure never needs another pass over the
hadrons.

The three sources
-----------------
* ``jet``  : iSS on the jet leg's surface (MUSIC_2, with the liquefier source): bulk + wake;
* ``bg``   : iSS on that event's background surface (MUSIC_1, no source);
* ``frag`` : ColorlessHadronization of the surviving partons.

The wake at hadron level is <jet> - <bg>, each an average over its oversamples. The full
medium-modified jet is frag + wake. Oversamples of one event share one fluid, so per event
everything is a sample mean, and its error comes from the spread of the per-sample sums
(right for the fragments, whose samples conserve the partons' energy, and for decay
daughters; Poisson counting would not be).

With common seeds (``hadronize.py --common-seeds``, implied by ``--correlated``) the jet
legs of the events sharing a background draw its random numbers, sample by sample. With
``--correlated`` they and the background then share most of their hadrons. So the jet leg
of one event is not independent of the jet legs of the others, nor of its background, and
neither is the wake. For such files the jet leg and the wake (jet - bg) are also stored as
means over ``--batches`` M blocks of aligned samples (block b of the jet leg with block b of
its background, and of every other event of that background). Any sum over events, bins
and weights then has the variance var_b(sum over the events of a background) / M, added
over backgrounds. Files sampled independently get no batches, and nothing else changes.

Jet frame
---------
Angles are measured from the event's **leading initiator** (the highest-pT parton that
starts a shower, ``initiators/`` of the hadron files): d_eta = eta - eta_L (pseudorapidity),
d_phi = phi - phi_L in [-pi/2, 3pi/2), dR = sqrt(d_eta^2 + d_phi^2) with d_phi in
[-pi, pi]. The background is binned in each event's own frame. It has its own flow
harmonics, so rotating it would not do.

Per event and source (``hist/<name>/<source>/mean`` and ``/var``, var = variance of the
mean):

* ``jetframe`` (d_eta, d_phi, pT; N, sum pT) of charged hadrons;
* ``dR``       (group, dR, pT; N, sum pT, sum E), group = charged, visible (all but
  neutrinos);
* ``spectra``  (region, species, pT; N) at |y| < 1, region = near (|d_phi| < pi/2), away;
* ``totals``   (group, |eta| band, pT; N, E, pT, pT cos d_phi, pT sin d_phi, pz, p.n), group
  = all (every particle, neutrinos included: the energy balance), charged; n = the leading
  initiator's direction. The last |eta| band (> 5) holds ColorlessHadronization's beam
  remnants (E = sqrt(s)/6 each, pT ~ 0.3 GeV).

With common seeds also ``hist/<name>/{jet,wake}/batch``, shape (event, quantity, batch,
...): the batch means of the jet leg and of jet - bg (NaN for events without them,
``events/batched`` = False). The difference is taken before rounding to ``--batch-bits``
mantissa bits (default 10: relative error < 2**-11 = 5e-4 per value). Rounding each leg
instead would not do: where the legs share most hadrons, the difference is much smaller
than either leg, and their rounding errors would swamp its noise.

Stored hadrons
--------------
``hadronize.py --eta-max X`` and ``--charged`` store only part of what iSS and Pythia made.
Each event records the cut of its files (``events/eta_max``, inf without one;
``events/charged_only``), the file its tightest one (attributes ``eta_max``,
``charged_only``), and the script warns. With a cut, "all" means "all inside the cut": the
|eta| bands beyond it are empty, the fragments lose ColorlessHadronization's beam remnants,
and the jet frame is cut at |d_eta| ~ X - |eta_L|.

Bad runs
--------
A leg that never froze out (MUSIC ran to its maximum time, as the background of job 0002
of ``AuAu_0_10_pth10-40_eta06_gridnorm``) has a surface that is not a freeze-out surface:
its hadrons are meaningless, and the wake of every event that uses it with them. Each event
gets the checks below before anything is loaded, and events that fail are left out (NaN
rows, ``usable`` = False) unless ``--keep-flagged``:

* ``*_hot_last`` : the same test as ``wake_observables.py``, from the pair file: the hottest
  cell of the leg's last stored frame is above 1.5 e_fo;
* ``mult_ratio`` / ``cells_ratio`` : hadron level, the background's multiplicity per sample
  (and freeze-out cells) over the jet leg's. The wake changes them by < 1%. A ratio outside
  [1/1.2, 1.2] means one surface is not like the other;
* ``log_max_time`` : "Maximum allowed time reached" in the job's ``<stem>.log`` (reported
  only, the log does not say which leg);
* ``wo_flag``    : the flags of ``wake_observables.h5`` if it is found (cross-check; the
  script warns when they disagree).

Usage::

    python wake_hadrons.py DIR [...] -o DIR/wake_hadrons.h5 [-j 8] [--wake-obs FILE]
                           [--batches M] [--batch-bits B]
"""

import argparse
import glob
import os
import sys
import time
from multiprocessing import Pool

import h5py
import numpy as np

try:  # Blosc, the default compression of the hadron files
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

from wake_observables import EoS, find_eos


def _h5_compression():
    """jetscape/h5_compression.py by its path: importing the jetscape package would load
    pyjetscape_core, which this script does not need (the module is self-contained)."""
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python",
                        "jetscape", "h5_compression.py")
    spec = importlib.util.spec_from_file_location("_js_h5_compression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_H5C = _h5_compression()

FORMAT = "js-contrib/wake_hadrons"
VERSION = 2                 # 2: batch means for common-seed files
SOURCES = ("jet", "bg", "frag")
TAG_OF = {"jet": "bulk_jet", "bg": "bulk_bg", "frag": "jet_frag"}
HOT_FACTOR = 1.5            # as wake_observables.py: last frame hotter than 1.5 e_fo
RATIO_MAX = 1.2             # bg / jet multiplicity and cell count outside [1/1.2, 1.2]
NEUTRINOS = (12, 14, 16)
# |pid| of the charged particles after iSS's decays, as jetscape.hadrons_h5.CHARGED (not
# imported: the jetscape package loads pyjetscape_core, which is not needed here)
CHARGED = (211, 321, 2212, 3222, 3112, 3312, 3334, 11, 13)

PT_EDGES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0, 20.0, 1e9])
DETA_EDGES = np.linspace(-4.0, 4.0, 17)
DPHI_EDGES = np.linspace(-np.pi / 2, 3 * np.pi / 2, 25)
DR_EDGES = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5,
                     3.0])
ETA_BANDS = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 1e9])       # |eta|
SPEC_PT_EDGES = np.r_[np.arange(0.0, 3.0, 0.2), 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0,
                      20.0]
Y_SPECTRA = 1.0
SPECIES = {"pi+": (211,), "pi-": (-211,), "K+": (321,), "K-": (-321,), "p": (2212,),
           "pbar": (-2212,), "Lambda+bar": (3122, -3122), "charged": None, "visible": None}
GROUPS_DR = ("charged", "visible")
GROUPS_TOT = ("all", "charged")
REGIONS = ("near", "away")
Q_JF = ("N", "pT")
Q_DR = ("N", "pT", "E")
Q_SP = ("N",)
Q_TOT = ("N", "E", "pT", "pT_cos", "pT_sin", "pz", "p_n")

SHAPES = {
    "jetframe": (len(DETA_EDGES) - 1, len(DPHI_EDGES) - 1, len(PT_EDGES) - 1),
    "dR": (len(GROUPS_DR), len(DR_EDGES) - 1, len(PT_EDGES) - 1),
    "spectra": (len(REGIONS), len(SPECIES), len(SPEC_PT_EDGES) - 1),
    "totals": (len(GROUPS_TOT), len(ETA_BANDS) - 1, len(PT_EDGES) - 1),
}
QUANTITIES = {"jetframe": Q_JF, "dR": Q_DR, "spectra": Q_SP, "totals": Q_TOT}


# ── hadrons of one unit ─────────────────────────────────────────────────────────
class Unit:
    """All samples of one unit of a hadron file, with the frame-independent kinematics."""

    def __init__(self, f, u):
        g = f["hadrons"]
        uo, so = g["unit_offsets"], g["sample_offsets"]
        s0, s1 = int(uo[u]), int(uo[u + 1])
        a, b = int(so[s0]), int(so[s1])
        self.K = s1 - s0
        self.sample = np.repeat(np.arange(self.K), np.diff(so[s0:s1 + 1]))
        self.pid = g["pid"][a:b]
        p = g["p"][a:b].astype(np.float64)
        self.E, self.px, self.py, self.pz = p.T
        self.pt = np.hypot(self.px, self.py)
        pabs = np.sqrt(self.pt ** 2 + self.pz ** 2)
        self.eta = 0.5 * np.log(np.clip(pabs + self.pz, 1e-300, None)
                                / np.clip(pabs - self.pz, 1e-300, None))
        self.y = 0.5 * np.log(np.clip(self.E + self.pz, 1e-300, None)
                              / np.clip(self.E - self.pz, 1e-300, None))
        self.phi = np.arctan2(self.py, self.px)
        apid = np.abs(self.pid)
        self.charged = np.isin(apid, CHARGED)
        self.visible = ~np.isin(apid, NEUTRINOS)
        self.ipt = np.searchsorted(PT_EDGES, self.pt, side="right") - 1
        self.ispt = np.searchsorted(SPEC_PT_EDGES, self.pt, side="right") - 1
        self.iband = np.searchsorted(ETA_BANDS, np.abs(self.eta), side="right") - 1
        self.mid_y = np.abs(self.y) < Y_SPECTRA
        self.species = {k: (self.charged if k == "charged" else self.visible if k == "visible"
                            else np.isin(self.pid, v)) for k, v in SPECIES.items()}


def sample_means(u, rows, idx, nbins, weights, batches=0):
    """Per bin: mean over u's samples of the per-sample sum of each weight, and the variance
    of that mean (from the spread of the per-sample sums). ``rows``: the hadrons (indices
    into u), ``idx``: their flat bins. With ``batches`` M (a divisor of u.K) also the means
    over M blocks of consecutive samples. -> (nq, nbins) twice, (nq, M, nbins) or None."""
    K = u.K
    key = u.sample[rows] * nbins + idx
    mean, var = np.zeros((len(weights), nbins)), np.zeros((len(weights), nbins))
    batch = np.zeros((len(weights), batches, nbins)) if batches else None
    for q, w in enumerate(weights):
        S = np.bincount(key, weights=w, minlength=K * nbins).reshape(K, nbins)
        mean[q] = S.mean(0)
        var[q] = S.var(0, ddof=1) / K if K > 1 else 0.0
        if batches:
            batch[q] = S.reshape(batches, K // batches, nbins).mean(1)
    return mean, var, batch


def bin_unit(u, ax, batches=0):
    """All histograms of one unit in the frame ``ax`` (phi_L, eta_L, n3). -> {name: (mean,
    var, batch)} with shapes (nq,) + SHAPES[name] and (nq, batches) + SHAPES[name] (None
    without ``batches``)."""
    phi_L, eta_L, n3 = ax
    dphi = np.mod(u.phi - phi_L + np.pi / 2, 2 * np.pi) - np.pi / 2       # [-pi/2, 3pi/2)
    dphi_s = np.where(dphi > np.pi, dphi - 2 * np.pi, dphi)               # [-pi, pi]
    deta = u.eta - eta_L
    dR = np.hypot(deta, dphi_s)
    ptok = (u.ipt >= 0) & (u.ipt < len(PT_EDGES) - 1)
    out = {}

    # (d_eta, d_phi, pT) of charged hadrons
    ie = np.searchsorted(DETA_EDGES, deta, side="right") - 1
    ip = np.minimum(np.searchsorted(DPHI_EDGES, dphi, side="right") - 1, len(DPHI_EDGES) - 2)
    nE, nP, nT = SHAPES["jetframe"]
    s = u.charged & ptok & (ie >= 0) & (ie < nE)
    idx = (ie[s] * nP + ip[s]) * nT + u.ipt[s]
    out["jetframe"] = sample_means(u, np.flatnonzero(s), idx, nE * nP * nT,
                                   (np.ones(s.sum()), u.pt[s]), batches)

    # (group, dR, pT)
    ir = np.searchsorted(DR_EDGES, dR, side="right") - 1
    nG, nR, nT = SHAPES["dR"]
    parts = []
    for gi, grp in enumerate(GROUPS_DR):
        s = getattr(u, grp) & ptok & (ir >= 0) & (ir < nR)
        parts.append((s, (gi * nR + ir[s]) * nT + u.ipt[s]))
    out["dR"] = _groups(u, parts, nG * nR * nT, lambda s: (np.ones(s.sum()), u.pt[s], u.E[s]),
                        batches)

    # (region, species, pT) at |y| < 1
    nRg, nS, nT = SHAPES["spectra"]
    sok = u.mid_y & (u.ispt >= 0) & (u.ispt < nT)
    away = (np.abs(dphi_s) >= np.pi / 2).astype(np.int64)
    parts = []
    for si, name in enumerate(SPECIES):
        s = sok & u.species[name]
        parts.append((s, (away[s] * nS + si) * nT + u.ispt[s]))
    out["spectra"] = _groups(u, parts, nRg * nS * nT, lambda s: (np.ones(s.sum()),), batches)

    # (group, |eta| band, pT) totals
    nG, nB, nT = SHAPES["totals"]
    pn = u.px * n3[0] + u.py * n3[1] + u.pz * n3[2]
    parts = []
    for gi, grp in enumerate(GROUPS_TOT):
        s = (np.ones(len(u.pid), bool) if grp == "all" else u.charged) & ptok
        parts.append((s, (gi * nB + u.iband[s]) * nT + u.ipt[s]))
    out["totals"] = _groups(u, parts, nG * nB * nT, lambda s: (
        np.ones(s.sum()), u.E[s], u.pt[s], u.pt[s] * np.cos(dphi[s]),
        u.pt[s] * np.sin(dphi[s]), u.pz[s], pn[s]), batches)
    nq = {k: len(QUANTITIES[k]) for k in out}
    return {k: (m.reshape((nq[k],) + SHAPES[k]), v.reshape((nq[k],) + SHAPES[k]),
                None if b is None else b.reshape((nq[k], batches) + SHAPES[k]))
            for k, (m, v, b) in out.items()}


def _groups(u, parts, nbins, weights, batches=0):
    """sample_means over (selection, flat index) parts that fill disjoint bins; one hadron
    may enter several parts (e.g. charged and visible). ``weights(sel)`` -> the weights."""
    rows = np.concatenate([np.flatnonzero(s) for s, _ in parts])
    idx = np.concatenate([i for _, i in parts])
    ws = [np.concatenate(w) for w in zip(*(weights(s) for s, _ in parts))]
    return sample_means(u, rows, idx, nbins, ws, batches)


# ── the axis ────────────────────────────────────────────────────────────────────
def axis_of(ini):
    """(phi, eta, n3) of the highest-pT initiator, and a record of the two leading ones.
    ini columns: shower, pid, pstat, px, py, pz, E, x, y, z, t."""
    pt = np.hypot(ini[:, 3], ini[:, 4])
    o = np.argsort(-pt)
    L = ini[o[0]]
    p3 = L[3:6]
    phi, eta = float(np.arctan2(L[4], L[3])), float(np.arcsinh(L[5] / pt[o[0]]))
    rec = dict(lead_shower=int(L[0]), lead_pid=int(L[1]), lead_pt=float(pt[o[0]]),
               lead_E=float(L[6]), lead_eta=eta, lead_phi=phi,
               n_init=len(ini), E_ini=float(ini[:, 6].sum()),
               pT_ini_par=float((pt * np.cos(np.arctan2(ini[:, 4], ini[:, 3]) - phi)).sum()))
    if len(o) > 1:
        S = ini[o[1]]
        rec.update(sub_pid=int(S[1]), sub_pt=float(pt[o[1]]),
                   sub_eta=float(np.arcsinh(S[5] / pt[o[1]])),
                   sub_dphi=float(np.mod(np.arctan2(S[4], S[3]) - phi + np.pi / 2, 2 * np.pi)
                                  - np.pi / 2))
    else:
        rec.update(sub_pid=0, sub_pt=np.nan, sub_eta=np.nan, sub_dphi=np.nan)
    return (phi, eta, p3 / np.linalg.norm(p3)), rec


# ── bad-run checks ──────────────────────────────────────────────────────────────
def last_frame_hot(pair, n_ev, eos_path):
    """Per event (jet, bg): the hottest cell of each leg's last stored frame [GeV/fm^3] and
    e_fo, from the pair file -- the test of wake_observables.py. None without a pair file."""
    if not pair or not os.path.exists(pair):
        return None
    eos = EoS(eos_path)
    with h5py.File(pair, "r") as f:
        e_fo = float(np.interp(float(f.attrs.get("T_fo", 0.15)), eos.T, eos.e))
        shared = "arr_bg_rows" in f
        rows = f["arr_bg_rows"][:] if shared else np.arange(n_ev)
        bg = f["arr_bg_store"] if shared else f["arr_bg"]
        ntj, ntb = f["ntau_freezeout"][:], f["ntau_freezeout_bg"][:]
        ej, eb, cache = np.zeros(n_ev), np.zeros(n_ev), {}
        for ev in range(n_ev):
            ej[ev] = float(f["arr"][ev, 0, :, :, :, int(ntj[ev]) - 1].max())
            key = (int(rows[ev]), int(ntb[ev]))
            if key not in cache:
                cache[key] = float(bg[key[0], 0, :, :, :, key[1] - 1].max())
            eb[ev] = cache[key]
    return ej, eb, e_fo


def wake_obs_flags(path):
    """{(pair file basename, ev): no-freeze-out flag} from a wake_observables.h5."""
    if not path or not os.path.exists(path):
        return {}
    with h5py.File(path, "r") as f:
        files = [os.path.basename(s.decode()) for s in f["files"][:]]
        E = f["events"]
        bad = np.zeros(len(E["ev"]), bool)
        for c in ("jet_no_freezeout", "bg_no_freezeout"):
            if c in E:
                bad |= E[c][:].astype(bool)
        return {(files[i], int(e)): bool(b) for i, e, b in zip(E["file"][:], E["ev"][:], bad)}


# ── one production file ─────────────────────────────────────────────────────────
def process_stem(job):
    fi, stem, eos_path, wo, max_events, keep_flagged, batches = job
    t0 = time.time()
    part = h5py.File(f"{stem}_particlize.h5", "r")
    hf = {s: h5py.File(f"{stem}_hadrons_{TAG_OF[s]}.h5", "r") for s in SOURCES}
    uuid = str(part.attrs.get("file_uuid", ""))
    for s, f in hf.items():
        if str(f.attrs.get("source_uuid", "")) != uuid:
            raise ValueError(f"{f.filename}: made from another particlize file than {stem}")
    # common seeds: the jet legs of a background draw its random numbers (batches below)
    common = bool(batches) and any(bool(hf["jet"].attrs.get(k, False))
                                   for k in ("common_seeds", "correlated_sampling"))
    no_batch = []
    # what hadronize.py kept (--eta-max, --charged): the tightest of the three files
    cuts = [hf[s].attrs.get("eta_max") for s in SOURCES]
    eta_max = min((float(c) for c in cuts if c is not None), default=np.inf)
    charged_only = any(bool(hf[s].attrs.get("charged_only", False)) for s in SOURCES)
    pev = {k: part["events"][k][:] for k in part["events"]}
    n_ev = len(pev["bg_unit"])
    if max_events:
        n_ev = min(n_ev, max_events)
    sig = np.atleast_1d(part.attrs.get("pthat_bin_sigma_gen", [np.nan])).astype(float)
    pair = os.path.join(os.path.dirname(stem), str(part.attrs.get("pair_file", "")))
    part.close()

    def unit_pos(f):
        ids = f["units/unit"][:] if "units" in f and "unit" in f["units"] else None
        return {int(u): k for k, u in enumerate(ids)} if ids is not None else None

    def per_sample(f, pos):
        uo, so = f["hadrons/unit_offsets"][:], f["hadrons/sample_offsets"][:]
        n = np.diff(uo)
        mult = np.array([(so[uo[k + 1]] - so[uo[k]]) / max(n[k], 1) for k in range(len(n))])
        return n, mult

    pos = {s: unit_pos(hf[s]) for s in SOURCES}
    nsmp, mult = {}, {}
    for s in SOURCES:
        nsmp[s], mult[s] = per_sample(hf[s], pos[s])
    cells_bg = hf["bg"]["units/n_cells"][:] if "n_cells" in hf["bg"]["units"] else None

    def upos(s, u):
        return int(u) if pos[s] is None else pos[s].get(int(u))

    hot = last_frame_hot(pair, n_ev, eos_path)
    log = f"{stem}.log"
    log_max = bool(os.path.exists(log) and "Maximum allowed time reached" in open(
        log, errors="replace").read())
    init_f = hf["jet"] if "initiators" in hf["jet"] else hf["frag"]
    ioff, idata = init_f["initiators/offsets"][:], init_f["initiators/data"][:]
    ipos = unit_pos(init_f)

    rows, hists = [], []
    bg_cache = (None, None)
    for ev in range(n_ev):
        bu = int(pev["bg_unit"][ev])
        pj, pb, pf = upos("jet", ev), upos("bg", bu), upos("frag", ev)
        kj = int(nsmp["jet"][pj]) if pj is not None else 0
        kb = int(nsmp["bg"][pb]) if pb is not None else 0
        kf = int(nsmp["frag"][pf]) if pf is not None else 0
        mj = mult["jet"][pj] if kj else np.nan
        mb = mult["bg"][pb] if kb else np.nan
        mr = mb / mj if kj and kb else np.nan
        cr = (float(pev["n_cells_bg"][ev]) / max(float(pev["n_cells_jet"][ev]), 1.0)
              if "n_cells_bg" in pev else np.nan)
        rec = dict(ev=ev, bg_unit=bu, n_jet=kj, n_bg=kb, n_frag=kf, mult_jet=mj, mult_bg=mb,
                   mult_ratio=mr, cells_ratio=cr,
                   pthat=float(pev["pthat"][ev]) if "pthat" in pev else np.nan,
                   pthat_bin=int(pev["pthat_bin"][ev]) if "pthat_bin" in pev else 0,
                   event_weight=float(pev["event_weight"][ev]) if "event_weight" in pev else 1.0,
                   E_droplets=float(pev["E_droplets"][ev]) if "E_droplets" in pev else np.nan,
                   ntau_jet=int(pev.get("ntau_jet", [-1] * (ev + 1))[ev]),
                   ntau_bg=int(pev.get("ntau_bg", [-1] * (ev + 1))[ev]),
                   log_max_time=log_max, eta_max=eta_max, charged_only=charged_only)
        rec["sigma_bin"] = float(sig[rec["pthat_bin"]]) if rec["pthat_bin"] < len(sig) else np.nan
        if hot is not None:
            ej, eb, e_fo = hot
            rec.update(e_last_jet=ej[ev], e_last_bg=eb[ev], e_fo=e_fo,
                       jet_hot_last=bool(ej[ev] > HOT_FACTOR * e_fo),
                       bg_hot_last=bool(eb[ev] > HOT_FACTOR * e_fo))
        else:
            rec.update(e_last_jet=np.nan, e_last_bg=np.nan, e_fo=np.nan, jet_hot_last=False,
                       bg_hot_last=False)
        off_ratio = lambda r: bool(np.isfinite(r) and not 1 / RATIO_MAX <= r <= RATIO_MAX)
        rec["mult_off"] = off_ratio(mr)
        rec["cells_off"] = off_ratio(cr)
        w = wo.get((os.path.basename(pair), ev))
        rec["wo_flag"] = -1 if w is None else int(w)
        rec["flagged"] = bool(rec["jet_hot_last"] or rec["bg_hot_last"] or rec["mult_off"]
                              or rec["cells_off"] or rec["wo_flag"] == 1)
        rec["usable"] = bool(kj > 0 and kb > 0 and not rec["flagged"])
        ip = ev if ipos is None else ipos.get(ev)
        ini = idata[int(ioff[ip]):int(ioff[ip + 1])] if ip is not None else np.zeros((0, 11))
        if len(ini):
            ax, arec = axis_of(ini)
        else:
            ax, arec = None, {}
            rec["usable"] = False
        rec.update(arec)

        H = None
        nb = 0
        if ax is not None and kj > 0 and kb > 0 and (rec["usable"] or keep_flagged):
            if common:
                if kj == kb and kj % batches == 0:
                    nb = batches
                else:
                    no_batch.append(ev)
            H = {}
            H["jet"] = bin_unit(Unit(hf["jet"], pj), ax, nb)
            if bg_cache[0] != pb:
                bg_cache = (pb, Unit(hf["bg"], pb))
            H["bg"] = bin_unit(bg_cache[1], ax, nb)
            H["frag"] = bin_unit(Unit(hf["frag"], pf), ax) if kf > 0 else None
            if nb:          # keep the jet leg's and the wake's batches, not the background's
                H["wake"] = {k: (None, None, (H["jet"][k][2] - H["bg"][k][2]).astype(np.float32))
                             for k in SHAPES}
                for s in ("jet", "bg"):
                    H[s] = {k: (m, v, b.astype(np.float32) if s == "jet" else None)
                            for k, (m, v, b) in H[s].items()}
        rec["processed"] = H is not None
        rec["common_seeds"] = common
        rec["batched"] = nb > 0
        rows.append(rec)
        hists.append(H)
    for f in hf.values():
        f.close()
    if no_batch:
        print(f"WARNING: {os.path.basename(stem)}: common seeds, but {len(no_batch)} event(s) "
              f"(e.g. {no_batch[:3]}) have unequal jet / background samples or a count not "
              f"divisible by --batches {batches}: no batches, their errors are taken as "
              "independent", file=sys.stderr)
    return fi, rows, hists, time.time() - t0


# ── driver ──────────────────────────────────────────────────────────────────────
def find_stems(paths):
    stems = []
    for a in paths:
        cands = (sorted(glob.glob(os.path.join(a, "*_particlize.h5"))) if os.path.isdir(a)
                 else [a])
        for p in cands:
            s = p[:-len("_particlize.h5")] if p.endswith("_particlize.h5") else p
            if all(os.path.exists(f"{s}_hadrons_{TAG_OF[t]}.h5") for t in SOURCES):
                stems.append(s)
            else:
                print(f"skipping {s}: not all three hadron files", file=sys.stderr)
    return stems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+", help="production directories or *_particlize.h5")
    ap.add_argument("-o", "--out", default="wake_hadrons.h5")
    ap.add_argument("-j", "--jobs", type=int, default=4, help="production files in parallel")
    ap.add_argument("--eos", default="", help="hotQCD EoS table (default: from prod_build)")
    ap.add_argument("--wake-obs", default=None,
                    help="wake_observables.h5 to cross-check the flags against (default: "
                         "<dir>/wake_observables.h5 of each directory, if there)")
    ap.add_argument("--keep-flagged", action="store_true",
                    help="histogram flagged events too (their 'usable' stays False)")
    ap.add_argument("--max-events", type=int, default=0, help="per file, for tests")
    ap.add_argument("--batches", type=int, default=10,
                    help="batch means per event for common-seed files (a divisor of the "
                         "samples per leg; default 10, 0: none)")
    ap.add_argument("--batch-bits", type=int, default=10, dest="batch_bits",
                    help="mantissa bits kept of the batch means (default 10: relative error "
                         "< 5e-4; 23: lossless float32)")
    args = ap.parse_args(argv)
    if args.batches < 0 or args.batches == 1:
        ap.error("--batches must be 0 or >= 2")
    if not 1 <= args.batch_bits <= 23:
        ap.error("--batch-bits must be 1..23")

    stems = find_stems(args.paths)
    if not stems:
        sys.exit("no production file with all three hadron files found")
    wo = {}
    for p in ([args.wake_obs] if args.wake_obs else
              [os.path.join(a, "wake_observables.h5") for a in args.paths if os.path.isdir(a)]):
        wo.update(wake_obs_flags(p))
    with h5py.File(f"{stems[0]}_particlize.h5", "r") as pf:
        pair = os.path.join(os.path.dirname(stems[0]), str(pf.attrs.get("pair_file", "")))
    attrs = h5py.File(pair, "r").attrs if os.path.exists(pair) else {}
    eos_path = find_eos(dict(attrs), args.eos) if attrs else ""

    jobs = [(i, s, eos_path, wo, args.max_events, args.keep_flagged, args.batches)
            for i, s in enumerate(stems)]
    results = [None] * len(stems)
    t0 = time.time()
    with Pool(max(1, min(args.jobs, len(stems)))) as pool:
        for r in pool.imap_unordered(process_stem, jobs):
            results[r[0]] = r
            n_ok = sum(x["processed"] for x in r[1])
            bad = [x["ev"] for x in r[1] if x["flagged"]]
            print(f"[{sum(x is not None for x in results)}/{len(stems)}] "
                  f"{os.path.basename(stems[r[0]])}: {n_ok}/{len(r[1])} events"
                  + (f", FLAGGED {len(bad)}" if bad else "") + f", {r[3]:.0f} s", flush=True)

    rows, hists, ev_file = [], [], []
    for fi, rr, hh, _ in results:
        for r, h in zip(rr, hh):
            rows.append(r)
            hists.append(h)
            ev_file.append(fi)
    n = len(rows)
    eta_cut = min((r["eta_max"] for r in rows), default=np.inf)
    charged_only = any(r["charged_only"] for r in rows)
    with h5py.File(args.out, "w") as out:
        out.attrs.update(dict(
            format=FORMAT, version=VERSION, created=time.strftime("%Y-%m-%d %H:%M:%S"),
            pt_edges=PT_EDGES, deta_edges=DETA_EDGES, dphi_edges=DPHI_EDGES, dR_edges=DR_EDGES,
            eta_bands=ETA_BANDS, spec_pt_edges=SPEC_PT_EDGES, y_spectra=Y_SPECTRA,
            species=list(SPECIES), groups_dR=list(GROUPS_DR), groups_tot=list(GROUPS_TOT),
            regions=list(REGIONS), hot_factor=HOT_FACTOR, ratio_max=RATIO_MAX,
            eta_max=eta_cut, charged_only=charged_only,
            batches=args.batches if any(r["batched"] for r in rows) else 0, doc=__doc__))
        out.create_dataset("files", data=np.array(stems, dtype="S"))
        ge = out.create_group("events")
        keys = sorted({k for r in rows for k in r})
        for k in keys:
            ge.create_dataset(k, data=np.array([r.get(k, np.nan) for r in rows]))
        ge.create_dataset("file", data=np.array(ev_file, np.int64))
        gh = out.create_group("hist")
        for name, shape in SHAPES.items():
            g = gh.create_group(name)
            g.attrs["quantities"] = list(QUANTITIES[name])
            full = (n, len(QUANTITIES[name])) + shape
            for s in SOURCES:
                for j, part in enumerate(("mean", "var")):
                    a = np.full(full, np.nan, np.float32)
                    for i, h in enumerate(hists):
                        if h is None:
                            continue
                        if h[s] is None:          # no fragments: the jet was absorbed whole
                            a[i] = 0.0
                        else:
                            a[i] = h[s][name][j]
                    g.create_dataset(f"{s}/{part}", data=a, compression="gzip",
                                     compression_opts=4, chunks=(1,) + full[1:])
            for s in ("jet", "wake") if out.attrs["batches"] else ():
                fb = full[:2] + (args.batches,) + shape
                a = np.full(fb, np.nan, np.float32)
                for i, h in enumerate(hists):
                    if h is not None and h.get(s) is not None and h[s][name][2] is not None:
                        a[i] = h[s][name][2]
                ds = g.create_dataset(f"{s}/batch", data=_H5C.round_mantissa(a, args.batch_bits),
                                      compression="gzip", compression_opts=4, shuffle=True,
                                      chunks=(1,) + fb[1:])
                _H5C.tag_dataset(ds, "gzip:4+shuffle", args.batch_bits)
    n_flag = sum(r["flagged"] for r in rows)
    kept = []
    if np.isfinite(eta_cut):
        kept.append(f"|eta| < {eta_cut:g}")
    if charged_only:
        kept.append("charged hadrons")
    if kept:
        print(f"WARNING: the hadron files keep only {' and '.join(kept)} (hadronize.py "
              "--eta-max / --charged): every histogram holds only those", file=sys.stderr)
    print(f"\n{n} events, {sum(r['processed'] for r in rows)} histogrammed, {n_flag} flagged "
          f"-> {args.out}  [{time.time() - t0:.0f} s]")
    for fi, s in enumerate(stems):
        rr = [r for r, f in zip(rows, ev_file) if f == fi]
        why = {k for r in rr if r["flagged"] for k in ("jet_hot_last", "bg_hot_last", "mult_off",
                                                        "cells_off") if r[k]}
        if why or any(r["log_max_time"] for r in rr):
            print(f"  {os.path.basename(s)}: flagged {sum(r['flagged'] for r in rr)}/{len(rr)} "
                  f"({', '.join(sorted(why)) or 'none'}); log 'Maximum allowed time': "
                  f"{rr[0]['log_max_time']}")
    dis = [r for r in rows if r["wo_flag"] >= 0 and bool(r["wo_flag"]) != (
        r["jet_hot_last"] or r["bg_hot_last"])]
    if dis:
        print(f"WARNING: {len(dis)} events where wake_observables.h5 and the pair-file check "
              "disagree")


if __name__ == "__main__":
    main()
