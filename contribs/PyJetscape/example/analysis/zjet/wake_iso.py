#!/usr/bin/env python3
"""The wake and the fragments of Z+jet events, each on its own, in (Delta eta, Delta phi) around
the Z and around the recoiling parton: no mixed event, no mean subtraction
(docs/Plans/PLAN_wake_isolated.md, step 1).

    python wake_iso.py map --src <dir> [<dir> ...] --out <run>.npz [--jobs 8] [--rec-eta 1.0]
        [--zpt LO HI] [--rec-pt LO HI]
    python wake_iso.py check --src <dir> [--file 0]      # the Z frame against zjet_wake

The contents, per Z (CMS 2507.09307's Z selection, zjet_wake.CMS_2507; every event weighted by
zjet_wake.event_weights, its window's cross section / events x N_coll):

  wake  <bulk_jet> - <bulk_bg>: each event's jet leg minus its own background, each the mean
        over its samples (correlated sampling: the bulk cancels in the mean, and so does most
        of its sampling noise, sample k against sample k)
  frag  <jet_frag> without the Z's two leptons

Vacuum (pp) files have the fragments only.  The frames, Delta phi in [-pi/2, 3pi/2):

  Z     d_eta = eta - y_Z,     d_phi = phi - phi_Z    (the recoil near pi)    all Z's
  ZJ    the same, for the Z's whose recoil has |eta_rec| < --rec-eta
  J     d_eta = eta - eta_rec, d_phi = phi - phi_rec  (the Z near pi)         as ZJ

The recoil is the event's hardest initiator that is a quark or a gluon.  Tracks: charged,
|eta| < 2.4, never the Z's leptons, in the p_T classes PT_EDGES.  Per frame and content: the
map (p_T class, d_eta, d_phi), the exact harmonics sum cos(n d_phi), n = 0 .. NH-1, per (p_T
class, d_eta bin), and the p_T spectra (SPEC_EDGES) toward the Z (d_phi on the Z's half) and
toward the recoil.  Per content, over every stored hadron (the files' |eta| < eta_max, all
species, any p_T): TOTALS.  Per event its deposited droplet energy (units/E_droplets).

Around the recoil, for the momentum balance (the J frame's Z's): every stored hadron in Delta R
rings (RING_EDGES), per p_T class RING_PT, its count, sum p_T and sum p_T cos(phi - phi_rec)
(``ring_<content>``; maps made before them have none).

Everything is stored as weighted sums per group: a hydro background in Pb+Pb, a block of
--block events in vacuum.  The per-Z means are the summed contents over the summed weights, and
a jackknife over groups follows from the stored sums (:class:`Maps`, :func:`jackknife`).  Runs
with the same seeds have the same groups and events and pair group by group
(``wake_scan/compare_iso.py``)."""
import argparse
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                        # jetcorr
try:
    import jetscape  # noqa: F401
except ImportError:                                              # the source tree
    sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "python"))
import zjet_wake as zw                                           # noqa: E402
from jetcorr.keys import wrap_dphi                               # noqa: E402

VERSION = 1
#: track p_T classes [GeV]: 0.5-1 (the model only), then CMS 2507.09307's
PT_EDGES = np.array([0.5, 1.0, 2.0, 4.0, 10.0])
PT_LABELS = [f"{a:g} < p_T^ch < {b:g} GeV" for a, b in zip(PT_EDGES[:-1], PT_EDGES[1:])]
DETA = np.linspace(-4.8, 4.8, 49)                                # 0.2
DPHI = np.linspace(-np.pi / 2, 3 * np.pi / 2, 33)                # pi / 16
NH = 5                                                           # cos(n d_phi), n = 0 .. 4
SPEC_EDGES = np.concatenate([np.arange(0.0, 3.01, 0.25), [3.5, 4, 5, 6, 8, 10, 15, 20]])
TOTALS = ("N", "E", "pT", "pT cos dphi_Z", "pT cos dphi_rec")
FRAMES = ("Z", "ZJ", "J")
#: rings in Delta R around the recoil (the J frame's Z's), for the momentum balance: every stored
#: hadron (any species, |eta| < the files' eta_max), per p_T class RING_PT, RING_Q per ring
RING_EDGES = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.7, 2.0, 2.5, 3.0,
                       3.5, 4.0, 6.0])
RING_PT = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 10.0, 1.0e4])
RING_Q = ("N", "pT", "pT cos dphi_rec")
TRACK_ETA = 2.4
N_PT, N_ETA, N_PHI, N_SPEC = len(PT_EDGES) - 1, len(DETA) - 1, len(DPHI) - 1, len(SPEC_EDGES) - 1
D_ETA, D_PHI = DETA[1] - DETA[0], DPHI[1] - DPHI[0]
ETA_C = 0.5 * (DETA[1:] + DETA[:-1])
PHI_C = 0.5 * (DPHI[1:] + DPHI[:-1])


# ── one event ────────────────────────────────────────────────────────────────────
def recoil(ini):
    """The hardest initiator that is a quark or gluon -> (pt, eta, phi), or None."""
    pid = np.abs(ini[:, 1]).astype(int) if len(ini) else np.zeros(0, int)
    par = ini[(pid <= 6) | (pid == 21)]
    if not len(par):
        return None
    pt = np.hypot(par[:, 3], par[:, 4])
    k = int(np.argmax(pt))
    return float(pt[k]), float(np.arcsinh(par[k, 5] / pt[k])), float(np.arctan2(par[k, 4],
                                                                                 par[k, 3]))


def _bin(x, edges):
    """Bin index of every value, numpy.histogramdd's rule (the last edge belongs to the last
    bin); -1 outside."""
    i = np.searchsorted(edges, x, side="right") - 1
    i[x == edges[-1]] = len(edges) - 2
    i[(i < 0) | (i >= len(edges) - 1)] = -1
    return i


def frame_sums(pt, deta, dphi, toward_z):
    """One set of tracks in one frame -> map (N_PT, N_ETA, N_PHI), harmonics (N_PT, N_ETA, NH)
    and spectra (2, N_SPEC): counts; harm[..., n] = sum cos(n d_phi) (exact, not from the
    map's bins); spec[0] the tracks with ``toward_z``, spec[1] the others (any d_eta)."""
    ip, ie, iq = _bin(pt, PT_EDGES), _bin(deta, DETA), _bin(dphi, DPHI)
    ok = (ip >= 0) & (ie >= 0) & (iq >= 0)
    cell = ip[ok] * N_ETA + ie[ok]
    m = np.bincount(cell * N_PHI + iq[ok], minlength=N_PT * N_ETA * N_PHI)
    d = dphi[ok]
    h = np.stack([np.bincount(cell, weights=np.cos(n * d), minlength=N_PT * N_ETA)
                  for n in range(NH)], axis=-1)
    js = _bin(pt, SPEC_EDGES)
    okp = js >= 0
    s = np.bincount((~toward_z[okp]).astype(np.int64) * N_SPEC + js[okp], minlength=2 * N_SPEC)
    return (m.reshape(N_PT, N_ETA, N_PHI).astype(float), h.reshape(N_PT, N_ETA, NH),
            s.reshape(2, N_SPEC).astype(float))


def ring_sums(pt, eta, phi, rec):
    """Every hadron in Delta R rings around the recoil -> (len(RING_PT) - 1, len(RING_EDGES) - 1,
    len(RING_Q)): count, sum p_T, sum p_T cos(phi - phi_rec) per (p_T class, ring)."""
    dphi = wrap_dphi(phi - rec[2], -np.pi)
    dr = np.hypot(eta - rec[1], dphi)
    ip, ir = _bin(pt, RING_PT), _bin(dr, RING_EDGES)
    ok = (ip >= 0) & (ir >= 0)
    n_p, n_r = len(RING_PT) - 1, len(RING_EDGES) - 1
    cell = ip[ok] * n_r + ir[ok]
    p, c = pt[ok], np.cos(dphi[ok])
    return np.stack([np.bincount(cell, weights=x, minlength=n_p * n_r)
                     for x in (np.ones(len(p)), p, p * c)], axis=-1).reshape(n_p, n_r, -1)


def leg_sums(ev, z, rec, in_j, keep=None):
    """One leg (an EventHadrons) of one event, summed over its samples (not divided) -> dict
    frame -> (map, harm, spec) (frames ZJ and J only if ``in_j``), the totals (TOTALS) and the
    rings around the recoil (:func:`ring_sums`, None unless ``in_j``).  ``z``: (y_Z, phi_Z);
    ``rec``: (pt, eta, phi) of the recoil; ``keep``: the hadrons that count (default all; the
    fragments leave the Z's leptons out)."""
    keep = np.ones(len(ev), bool) if keep is None else keep
    pt, eta, phi = ev.pt[keep], ev.eta[keep], ev.phi[keep]
    trk = ev.charged[keep] & (np.abs(eta) < TRACK_ETA)
    t_pt, t_eta, t_phi = pt[trk], eta[trk], phi[trk]
    dz = wrap_dphi(t_phi - z[1])
    out = {"Z": frame_sums(t_pt, t_eta - z[0], dz, dz < np.pi / 2)}
    if in_j:
        out["ZJ"] = out["Z"]
        dj = wrap_dphi(t_phi - rec[2])
        out["J"] = frame_sums(t_pt, t_eta - rec[1], dj, dj >= np.pi / 2)
    tot = np.array([len(pt), ev.E[keep].sum(), pt.sum(), (pt * np.cos(phi - z[1])).sum(),
                    (pt * np.cos(phi - rec[2])).sum()])
    return out, tot, (ring_sums(pt, eta, phi, rec) if in_j else None)


def _empty():
    d = {"W": 0.0, "W2": 0.0, "WJ": 0.0, "N": 0, "NJ": 0, "edrop": 0.0}
    return d


def _add(acc, key, x):
    if key in acc:
        acc[key] += x
    else:
        acc[key] = np.array(x, dtype=float)


def file_job(stem, local_events, weights, rec_eta, block, rec_pt=None):
    """The sums of one production file's selected events (the map step) -> dict: ``groups``
    (key -> dict of sums), ``events`` (per-event table), ``vacuum``."""
    import h5py

    import jetscape.h5_compression  # noqa: F401  (Blosc)
    from jetscape.hadrons_h5 import EventHadrons, HadronFile, HadronFileReader, Hadrons

    r = HadronFileReader([stem])
    tags = r.tags(0)
    vacuum = not ("bulk_jet" in tags and "bulk_bg" in tags)
    local_events = np.asarray(local_events, np.int64)
    infos = [r.event_info(int(e)) for e in local_events]

    def load(tag, units):
        path = f"{stem}_hadrons_{tag}.h5"
        with HadronFile(path) as hf:
            pos = sorted({hf.unit_index(int(u)) for u in units})
        return Hadrons.from_h5(path, units=np.array(pos, dtype=np.int64))

    with h5py.File(f"{stem}_hadrons_jet_frag.h5", "r") as f:
        u = f["units"]
        edrop = (dict(zip(u["unit"][:].tolist(), u["E_droplets"][:].tolist()))
                 if "E_droplets" in u else {})
    hf_ = load("jet_frag", local_events)
    if not vacuum:
        hj = load("bulk_jet", local_events)
        hb = load("bulk_bg", {i.bg_unit for i in infos})
    groups = {}
    cols = ("seed", "local_event", "group", "w", "z_pt", "z_y", "z_phi", "rec_pt", "rec_eta",
            "rec_phi", "in_j", "e_drop")
    events = {c: [] for c in cols}
    for e, w, info in zip(local_events, weights, infos):
        zr = info.z
        z = (float(zr["z_y"]), float(zr["z_phi"]))
        rec = recoil(info.initiators())
        if rec is None:
            warnings.warn(f"{stem} event {e}: no parton initiator: left out", RuntimeWarning,
                          stacklevel=2)
            continue
        if rec_pt is not None and not rec_pt[0] <= rec[0] < rec_pt[1]:
            continue
        in_j = abs(rec[1]) < rec_eta
        F = EventHadrons(hf_, hf_.unit_index(int(e)))
        nf = F.n_samples
        if vacuum:
            if nf == 0:
                continue
            legs = {"frag": [(F, 1.0 / nf, zw.not_z_lepton(F, info))]}
            key = f"{info.seed}:{int(e) // block}"
        else:
            J = EventHadrons(hj, hj.unit_index(int(e)))
            B = EventHadrons(hb, hb.unit_index(info.bg_unit))
            K, Kb = J.n_samples, B.n_samples
            if K == 0 or nf == 0 or Kb != K:
                warnings.warn(f"{stem} event {e}: {K} jet / {Kb} background samples, {nf} "
                              "fragmentations: left out", RuntimeWarning, stacklevel=2)
                continue
            # the bulk has no Z leptons: every hadron counts
            legs = {"wake": [(J, 1.0 / K, None), (B, -1.0 / Kb, None)],
                    "frag": [(F, 1.0 / nf, zw.not_z_lepton(F, info))]}
            key = f"{info.seed}:{info.bg_unit}"
        acc = groups.setdefault(key, _empty())
        acc["W"] += w
        acc["W2"] += w * w
        acc["N"] += 1
        if in_j:
            acc["WJ"] += w
            acc["NJ"] += 1
        de = float(edrop.get(int(e), 0.0))
        acc["edrop"] += w * de
        for content, parts in legs.items():
            for leg, scale, keep in parts:
                fr, tot, ring = leg_sums(leg, z, rec, in_j, keep)
                _add(acc, f"tot_{content}", w * scale * tot)
                if ring is not None:
                    _add(acc, f"ring_{content}", w * scale * ring)
                for frame, (m, h, s) in fr.items():
                    _add(acc, f"map_{content}_{frame}", w * scale * m)
                    _add(acc, f"harm_{content}_{frame}", w * scale * h)
                    _add(acc, f"spec_{content}_{frame}", w * scale * s)
        for c, v in zip(cols, (info.seed, int(e), key, w, zr["z_pt"], z[0], z[1], rec[0],
                               rec[1], rec[2], in_j, de)):
            events[c].append(v)
    return {"groups": groups, "events": events, "vacuum": vacuum}


def _job(args):
    t0 = time.time()
    out = file_job(*args)
    return os.path.basename(args[0]), len(out["events"]["w"]), time.time() - t0, out


def stems_of(srcs):
    """The production stems of directories (every production file in them) or stems."""
    from jetscape.hadrons_h5 import HadronFileReader

    out = []
    for s in srcs:
        out += list(HadronFileReader(s).stems) if os.path.isdir(s) else [s]
    return out


def build(srcs, out, jobs=4, rec_eta=1.0, block=60, rebuild=False, zpt=None, rec_pt=None):
    """The map step of the production files in ``srcs`` (directories or stems) -> ``out``
    (.npz), cached: rebuilt when the files or the settings changed.  ``zpt``: (lo, hi) of p_T^Z
    instead of CMS 2507's 40-350 GeV; ``rec_pt``: (lo, hi) of the recoil's p_T (default any)."""
    from jetscape.hadrons_h5 import HadronFileReader

    stems = stems_of(srcs)
    r = HadronFileReader(stems)
    t = zw.event_table(r)
    cuts = dict(zw.CMS_2507, **({"pt": tuple(zpt)} if zpt else {}))
    sel = zw.select(t, **cuts)
    w = zw.event_weights(t, sel)
    loc = np.array([r.locate(g) for g in range(r.n_events)], np.int64).reshape(-1, 2)
    meta = {"version": VERSION, "sources": [os.path.abspath(s) for s in srcs],
            "selection": "CMS_2507", "zpt": list(cuts["pt"]),
            "rec_pt": list(rec_pt) if rec_pt else None, "rec_eta": rec_eta, "block": block,
            "n_events": int(r.n_events), "n_selected": int(sel.sum()),
            "files": [[os.path.basename(s), os.path.getmtime(f"{s}_hadrons_jet_frag.h5")]
                      for s in r.stems],
            "pt_edges": PT_EDGES.tolist(), "deta": DETA.tolist(), "dphi": DPHI.tolist(),
            "spec_edges": SPEC_EDGES.tolist(), "totals": list(TOTALS), "track_eta": TRACK_ETA,
            "ring_edges": RING_EDGES.tolist(), "ring_pt": RING_PT.tolist(), "ring_q": list(RING_Q),
            "eta_max": r.eta_max()}
    if not rebuild and os.path.exists(out):
        with np.load(out) as d:
            old = json.loads(str(d["meta"]))
        old.pop("vacuum", None)
        if old == json.loads(json.dumps(meta)):
            print(f"{out}: cached", flush=True)
            return out
    t0 = time.time()
    print(f"map: {len(r.stems)} files, {int(sel.sum())} selected events, {jobs} jobs",
          flush=True)
    args = []
    for i, stem in enumerate(r.stems):
        g = np.flatnonzero(sel & (loc[:, 0] == i))
        args.append((stem, loc[g, 1], w[g], rec_eta, block, rec_pt))
    results = []
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for name, n, dt, res in ex.map(_job, args):
            print(f"  {name}: {n} events, {dt:.0f} s", flush=True)
            results.append(res)
    vac = {res["vacuum"] for res in results}
    if len(vac) != 1:
        raise SystemExit("map: Pb+Pb and vacuum files mixed")
    meta["vacuum"] = vac.pop()
    groups = {}
    for res in results:
        for k, v in res["groups"].items():
            if k in groups:
                raise SystemExit(f"map: group {k} in two files (seeds must differ per file)")
            groups[k] = v
    keys = sorted(groups)
    arrays = {"meta": np.array(json.dumps(meta)), "group": np.array(keys)}
    names = sorted({n for v in groups.values() for n in v})
    for n in names:
        arrays[n] = np.array([groups[k][n] for k in keys])
    ev = {c: np.concatenate([np.asarray(res["events"][c]) for res in results])
          for c in results[0]["events"]}
    order = np.lexsort((ev["local_event"], ev["seed"]))
    for c, v in ev.items():
        arrays[f"ev_{c}"] = v[order]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez_compressed(out, **arrays)
    print(f"-> {out} ({len(keys)} groups, {len(order)} Z's, {time.time() - t0:.0f} s)",
          flush=True)
    return out


# ── reading: per-Z means and the jackknife ──────────────────────────────────────────
class Maps:
    """One run's stored sums (``wake_iso.py map``)."""

    def __init__(self, path):
        with np.load(path, allow_pickle=False) as d:
            self.meta = json.loads(str(d["meta"]))
            self.groups = d["group"].astype(str)
            self.sums = {k: d[k] for k in d.files
                         if k.startswith(("map_", "harm_", "spec_", "tot_", "ring_"))}
            self.W = {"Z": d["W"], "ZJ": d["WJ"], "J": d["WJ"]}
            self.W2, self.edrop = d["W2"], d["edrop"]
            self.N, self.NJ = d["N"], d["NJ"]
            self.events = {k[3:]: d[k] for k in d.files if k.startswith("ev_")}
        self.path = path
        self.vacuum = bool(self.meta["vacuum"])
        if self.meta["version"] != VERSION:
            raise ValueError(f"{path}: version {self.meta['version']}, need {VERSION}")

    @property
    def n_z(self):
        return int(self.N.sum())

    @property
    def n_eff(self):
        return float(self.W["Z"].sum() ** 2 / self.W2.sum())

    def blocks(self, n):
        g = np.arange(len(self.groups))
        return [b for b in np.array_split(g, min(n, len(g))) if len(b)]

    def same_events(self, other):
        """True if ``other`` has the same groups, Z events and weights."""
        a, b = self.events, other.events
        return (np.array_equal(self.groups, other.groups)
                and all(np.array_equal(a[k], b[k]) for k in ("seed", "local_event", "group",
                                                             "in_j"))
                and all(np.allclose(a[k], b[k], rtol=1e-12, atol=0)
                        for k in ("w", "z_pt", "z_y", "z_phi")))


class View:
    """The per-Z means of a run over a subset of its groups (``keep``, default all)."""

    def __init__(self, maps, keep=None):
        self.m, self.keep = maps, keep

    def _sel(self, x):
        return x if self.keep is None else x[self.keep]

    def mean(self, key):
        """Per-Z mean of a stored content: ``map_<content>_<frame>`` etc. (per Z per bin),
        ``tot_<content>``, ``edrop``."""
        frame = (key.rsplit("_", 1)[-1] if key.startswith(("map_", "harm_", "spec_")) else
                 "J" if key.startswith("ring_") else "Z")
        S = self.m.edrop if key == "edrop" else self.m.sums[key]
        return self._sel(S).sum(0) / self._sel(self.m.W[frame]).sum()


def jackknife(fn, sets, n_blocks=100):
    """A quantity ``fn(views)`` of several runs and its jackknife error.  ``sets``: lists of
    Maps; the runs of one set share their groups (the same backgrounds: paired, they are left
    out together), different sets are independent (their variances add); ``views`` is the flat
    list of the runs' :class:`View`s in that order.  -> (value, error)."""
    flat = [m for s in sets for m in s]
    full = [View(m) for m in flat]
    value = np.asarray(fn(full), float)
    var = np.zeros_like(value)
    pos = 0
    for s in sets:
        for m in s[1:]:
            if not np.array_equal(m.groups, s[0].groups):
                raise ValueError(f"{m.path} and {s[0].path}: other groups, not paired")
        blocks = s[0].blocks(n_blocks)
        G = len(blocks)
        if G >= 2:
            reps = []
            for b in blocks:
                keep = np.ones(len(s[0].groups), bool)
                keep[b] = False
                views = list(full)
                for j in range(len(s)):
                    views[pos + j] = View(flat[pos + j], keep)
                reps.append(np.asarray(fn(views), float))
            r = np.array(reps)
            var = var + (G - 1) / G * ((r - r.mean(0)) ** 2).sum(0)
        pos += len(s)
    return value, np.sqrt(var)


# ── the quantities read off the maps (PLAN_wake_isolated.md §1) ─────────────────────
def zside_phi(frame):
    """The d_phi bins toward the Z: |d_phi| < pi/2 in the Z frames, |d_phi - pi| < pi/2 in J."""
    return np.abs(PHI_C) < np.pi / 2 if frame != "J" else np.abs(PHI_C - np.pi) < np.pi / 2


def phi_proj(v, content, frame, eta_max=None):
    """dN/dd_phi per Z (N_PT, N_PHI), over |d_eta| < eta_max (default all)."""
    m = v.mean(f"map_{content}_{frame}")
    if eta_max is not None:
        m = m[:, np.abs(ETA_C) < eta_max]
    return m.sum(1) / D_PHI


def eta_proj(v, content, frame, side="z"):
    """dN/dd_eta per Z (N_PT, N_ETA) on one half in d_phi: ``z`` toward the Z, ``rec``
    toward the recoil."""
    m = v.mean(f"map_{content}_{frame}")
    zs = zside_phi(frame)
    return m[:, :, zs if side == "z" else ~zs].sum(-1) / D_ETA


def harmonics(v, content, frame, eta_max=None):
    """c_n = sum cos(n d_phi) per Z, (N_PT, NH), over |d_eta| < eta_max (default all)."""
    h = v.mean(f"harm_{content}_{frame}")
    if eta_max is not None:
        h = h[:, np.abs(ETA_C) < eta_max]
    return h.sum(1)


def dip(v, content, frame, eta_max=1.0, inner=np.pi / 8, outer=(np.pi / 4, np.pi / 2)):
    """The density toward the Z at |d_phi - phi_Z| < inner minus that at outer[0] < |d_phi -
    phi_Z| < outer[1], per Z per rad per unit d_eta, at |d_eta| < eta_max -> (N_PT,), and the
    two densities (N_PT, 2)."""
    m = v.mean(f"map_{content}_{frame}")[:, np.abs(ETA_C) < eta_max]
    a = np.abs(wrap_dphi(PHI_C - (0.0 if frame != "J" else np.pi), -np.pi))
    i, o = a < inner, (a > outer[0]) & (a < outer[1])
    span = 2 * eta_max
    rho = np.stack([m[:, :, i].sum((1, 2)) / (i.sum() * D_PHI * span),
                    m[:, :, o].sum((1, 2)) / (o.sum() * D_PHI * span)], axis=-1)
    return rho[:, 0] - rho[:, 1], rho


def core_wings(v, content, core=0.4, edge=1.4):
    """Around the recoil (J frame, its half |d_phi| < pi/2): the density per Z per unit d_eta
    at |d_eta| < core and at core < |d_eta| < edge -> (N_PT, 2)."""
    m = v.mean(f"map_{content}_J")[:, :, np.abs(PHI_C) < np.pi / 2].sum(-1)
    a = np.abs(ETA_C)
    c, w = a < core, (a > core) & (a < edge)
    return np.stack([m[:, c].sum(1) / (c.sum() * D_ETA), m[:, w].sum(1) / (w.sum() * D_ETA)],
                    axis=-1)


def spectra(v, content, frame="Z"):
    """dN/dp_T per Z toward the Z and toward the recoil -> (2, N_SPEC)."""
    return v.mean(f"spec_{content}_{frame}") / np.diff(SPEC_EDGES)


# ── the check against zjet_wake ──────────────────────────────────────────────────────
def check(src, file_index=0, jobs=1):
    """The Z frame maps of one file against zjet_wake.z_frame_yields (the reader's paired
    difference): the same events and weights, the same bins -> max |difference|."""
    from jetscape.hadrons_h5 import HadronFileReader

    stem = stems_of([src])[file_index]
    scratch = os.environ.get("TMPDIR") or os.path.dirname(os.path.abspath(stem))
    out = os.path.join(scratch, f".wake_iso_check_{os.getpid()}.npz")
    try:
        build([stem], out, jobs=jobs, rebuild=True)
        v = View(Maps(out))
    finally:
        if os.path.exists(out):
            os.remove(out)
    r = HadronFileReader([stem])
    t = zw.event_table(r)
    sel = zw.select(t, **zw.CMS_2507)
    w = zw.event_weights(t, sel)
    if v.m.vacuum:
        y = {"jet": zw.vacuum_yields(r, np.flatnonzero(sel), w, DETA, DPHI, pt_edges=PT_EDGES)}
    else:
        y = zw.z_frame_yields(r, np.flatnonzero(sel), w, DETA, DPHI, pt_edges=PT_EDGES)
    worst = 0.0
    for content, ref in (("wake", y.get("wake")), ("frag", y.get("jet"))):
        if ref is None:
            continue
        mine = v.mean(f"map_{content}_Z")
        d = float(np.abs(mine - ref[0]).max())
        scale = float(np.abs(ref[0]).max())
        print(f"{os.path.basename(stem)} {content}: max |wake_iso - zjet_wake| = {d:.3g} "
              f"(largest bin {scale:.3g}, total {ref[0].sum():+.4f} vs {mine.sum():+.4f})")
        worst = max(worst, d / max(scale, 1e-300))
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("map", help="the map step -> one .npz")
    m.add_argument("--src", nargs="+", required=True, help="production directories (one run)")
    m.add_argument("--out", required=True)
    m.add_argument("--jobs", type=int, default=4)
    m.add_argument("--rec-eta", type=float, default=1.0,
                   help="|eta_rec| cut of the frames ZJ and J (default 1: |d_eta| < 1.4 in "
                        "the track acceptance)")
    m.add_argument("--block", type=int, default=60,
                   help="vacuum: events per group (the jackknife unit)")
    m.add_argument("--zpt", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                   help="p_T^Z window instead of CMS 2507's 40-350 GeV")
    m.add_argument("--rec-pt", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                   help="the recoil parton's p_T window (default any)")
    m.add_argument("--rebuild", action="store_true")
    c = sub.add_parser("check", help="the Z frame of one file against zjet_wake.z_frame_yields")
    c.add_argument("--src", required=True)
    c.add_argument("--file", type=int, default=0)
    c.add_argument("--tol", type=float, default=1e-9, help="relative to the largest bin")
    a = ap.parse_args()
    if a.cmd == "map":
        build(a.src, a.out, a.jobs, a.rec_eta, a.block, a.rebuild, a.zpt, a.rec_pt)
    else:
        worst = check(a.src, a.file)
        print(f"check: {'OK' if worst <= a.tol else 'FAILED'} (worst {worst:.2g} of the largest "
              f"bin, tolerance {a.tol:g})")
        sys.exit(0 if worst <= a.tol else 1)


if __name__ == "__main__":
    main()
