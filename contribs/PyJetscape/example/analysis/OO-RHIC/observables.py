#!/usr/bin/env python3
"""
analysis/OO-RHIC/observables.py -- the model's observables of one run (a design point or a
validation run), in the binning of the data.

    python observables.py p000                    # PROD_DIR/events/p000 -> PROD_DIR/obs/p000.npz
    python observables.py --design -j 8           # the design points p000 ... (design.json)
    python observables.py --all -j 8              # every run under PROD_DIR/events
    python observables.py trunc --dir DIR         # any directory of jet_frag files

Per pTHat window k the counts are kept separately (``<obs>_w`` and ``<obs>_w2``: the sum of
the per-hadron or per-jet weights and of their squares, events of window k only), with the
window's cross section sigma_k [mb] and number of events N_k.  The observable is
    O = sum_k sigma_k / N_k * w_k / (bin width ...)
so windows can be dropped or their share inspected (truncation_check.py).

Definitions (data in data/data.json):
    phenix_pi0      E d3sigma/dp3 = 1/(2 pi pT) d2sigma/(dpT dy), pi0 with |y| < 0.35 [mb GeV^-2]
    star_pi0        the same, 0 < y < 1
    star_eta_pi0    eta / pi0 yields, 0 < y < 1, same pT bins
    star_jets_eta0  d2sigma/(dpT deta) [pb/GeV], anti-kT R = 0.5 on the visible final particles
    star_jets_eta1  (pT > 0.2 GeV, |eta| < 1.5), jets with |eta| < 0.5 / 0.5 < |eta| < 0.9,
                    at least one charged constituent, pT corrected for the underlying event
                    with STAR's off-axis cones: two cones of radius R at the jet's eta and
                    phi +- pi/2, dpT = A_jet (rho_1 + rho_2) / 2 with A_jet = pi R^2.
The pi0 and eta are undecayed in the model (config.FIXED_HADRONIZATION).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import config as C

CHARGED = None          # filled lazily from jetscape.hadrons_h5.CHARGED


def _charged_ids():
    global CHARGED
    if CHARGED is None:
        sys.path.insert(0, str(C.EXAMPLE.parent / "python"))
        from jetscape.hadrons_h5 import CHARGED as CH
        CHARGED = np.asarray(sorted(set(abs(int(x)) for x in CH)))
    return CHARGED


def load_data():
    return json.loads((C.DATA_DIR / "data.json").read_text())


def _read_units(f, n):
    """pid, p and hadrons per unit of the first n units (all samples of a unit together)."""
    so = f["hadrons/sample_offsets"][:]
    uo = f["hadrons/unit_offsets"][: n + 1]
    stop = int(so[uo[-1]])
    return (f["hadrons/pid"][:stop], f["hadrons/p"][:stop].astype(np.float64),
            np.diff(so[uo]))


def read_file(path, with_bulk=True):
    """One jet_frag file: hadrons (pid, E, px, py, pz), counts per event, window per event,
    and the file's per-window cross sections and accepted events.  The jet file has one
    sample per unit.  With a <stem>_hadrons_bulk_bg.h5 next to it (the p+p hydro runs,
    --bulk-hadrons 1) and with_bulk, each event's bulk hadrons are added to its jet hadrons
    (unit i of both files is event i; events without a medium have no bulk sample)."""
    import hdf5plugin  # noqa: F401
    import h5py
    with h5py.File(path, "r") as f:
        n = int(f.attrs["nunits_written"])
        pid, p, counts = _read_units(f, n)
        win = f["units/pthat_bin"][:n].astype(np.int64)
        sig = np.asarray(f.attrs["pthat_bin_sigma_gen"], dtype=np.float64)
        nacc = np.asarray(f.attrs["pthat_bin_n_accepted"], dtype=np.int64)
        bins = np.asarray(f.attrs["pthat_bins"], dtype=np.float64)
    bulk = str(path).replace("_hadrons_jet_frag.h5", "_hadrons_bulk_bg.h5")
    n_bulk = None
    if with_bulk and bulk != str(path) and os.path.exists(bulk):
        with h5py.File(bulk, "r") as f:
            if int(f.attrs["nunits_written"]) < n:
                raise ValueError(f"{bulk}: fewer units than {path}")
            bpid, bp, bcounts = _read_units(f, n)
        ev = np.concatenate([np.repeat(np.arange(n), counts), np.repeat(np.arange(n), bcounts)])
        order = np.argsort(ev, kind="stable")              # per event: jet, then bulk
        pid = np.concatenate([pid, bpid])[order]
        p = np.concatenate([p, bp])[order]
        n_bulk = bcounts
        counts = counts + bcounts
    return dict(pid=pid, p=p, counts=counts, win=win, sigma=sig, n_acc=nacc, bins=bins,
                n_bulk=n_bulk)


def _kin(p):
    E, px, py, pz = p.T
    pt = np.hypot(px, py)
    y = 0.5 * np.log(np.clip(E + pz, 1e-12, None) / np.clip(E - pz, 1e-12, None))
    pabs = np.sqrt(px * px + py * py + pz * pz)
    eta = 0.5 * np.log(np.clip(pabs + pz, 1e-12, None) / np.clip(pabs - pz, 1e-12, None))
    phi = np.arctan2(py, px)
    return pt, y, eta, phi


def _edges(d):
    e = np.asarray(d["edges"], dtype=np.float64)
    return np.append(e[:, 0], e[-1, 1])


def _hist_by_window(x, w, win, nwin, edges):
    """sum of w and w^2 per (window, bin)."""
    out = np.zeros((nwin, len(edges) - 1))
    out2 = np.zeros_like(out)
    idx = np.searchsorted(edges, x, side="right") - 1
    ok = (idx >= 0) & (idx < len(edges) - 1)
    np.add.at(out, (win[ok], idx[ok]), w[ok])
    np.add.at(out2, (win[ok], idx[ok]), w[ok] ** 2)
    return out, out2


def hadron_obs(fd, data, acc):
    """pi0 and eta spectra of one file into acc."""
    pid, p, counts, win = fd["pid"], fd["p"], fd["counts"], fd["win"]
    hwin = np.repeat(win, counts)
    pt, y, _, _ = _kin(p)
    nwin = len(fd["sigma"])
    for name, sel, dy in (("phenix_pi0", (pid == 111) & (np.abs(y) < 0.35), 0.7),
                          ("star_pi0", (pid == 111) & (y > 0) & (y < 1), 1.0)):
        e = _edges(data[name])
        w = 1.0 / (2 * np.pi * pt[sel] * dy)
        a, a2 = _hist_by_window(pt[sel], w, hwin[sel], nwin, e)
        acc[name + "_w"] += a
        acc[name + "_w2"] += a2
    e = _edges(data["star_eta_pi0"])
    for name, code in (("star_eta_pi0_num", 221), ("star_eta_pi0_den", 111)):
        sel = (pid == code) & (y > 0) & (y < 1)
        a, a2 = _hist_by_window(pt[sel], np.ones(sel.sum()), hwin[sel], nwin, e)
        acc[name + "_w"] += a
        acc[name + "_w2"] += a2


def jet_obs(fd, data, acc, chunk=20000):
    """STAR jets of one file into acc (chunks of events; fastjet on awkward arrays, the
    charged-constituent requirement and the off-axis cones vectorised with numpy)."""
    import awkward as ak
    import fastjet

    charged = _charged_ids()
    pid, p, counts, win = fd["pid"], fd["p"], fd["counts"], fd["win"]
    pt, _, eta, phi = _kin(p)
    keep = ~np.isin(np.abs(pid), C.INVISIBLE) & (pt > C.CONST_PTMIN)
    ev = np.repeat(np.arange(len(counts)), counts)[keep]
    P, PT, ETA, PHI = p[keep], pt[keep], eta[keep], phi[keep]
    CH = np.isin(np.abs(pid[keep]), charged)
    cnt = np.bincount(ev, minlength=len(counts))
    off = np.concatenate([[0], np.cumsum(cnt)])
    jdef = fastjet.JetDefinition(fastjet.antikt_algorithm, C.JET_R)
    nwin = len(fd["sigma"])
    e = _edges(data["star_jets_eta0"])
    R2 = C.JET_R ** 2
    area = np.pi * R2
    for c0 in range(0, len(counts), chunk):
        c1 = min(c0 + chunk, len(counts))
        s0, s1 = off[c0], off[c1]
        cc = cnt[c0:c1]
        parts = ak.zip({"px": ak.unflatten(P[s0:s1, 1], cc), "py": ak.unflatten(P[s0:s1, 2], cc),
                        "pz": ak.unflatten(P[s0:s1, 3], cc), "E": ak.unflatten(P[s0:s1, 0], cc)})
        cs = fastjet.ClusterSequence(parts, jdef)
        jets = cs.inclusive_jets(min_pt=4.0)
        cons = cs.constituent_index(min_pt=4.0)
        njet = ak.to_numpy(ak.num(jets))
        if njet.sum() == 0:
            continue
        jpx, jpy, jpz = (ak.to_numpy(ak.flatten(jets[k])) for k in ("px", "py", "pz"))
        jev = np.repeat(np.arange(c0, c1), njet)              # the jet's event (absolute)
        jpt = np.hypot(jpx, jpy)
        jp = np.sqrt(jpx ** 2 + jpy ** 2 + jpz ** 2)
        jeta = 0.5 * np.log(np.clip(jp + jpz, 1e-12, None) / np.clip(jp - jpz, 1e-12, None))
        jphi = np.arctan2(jpy, jpx)
        nj = len(jpt)
        # at least one charged constituent: constituent indices are per event
        ncon = ak.to_numpy(ak.flatten(ak.num(cons, axis=2)))
        cidx = ak.to_numpy(ak.flatten(cons, axis=None)).astype(np.int64)
        cjet = np.repeat(np.arange(nj), ncon)
        has_ch = np.bincount(cjet, weights=CH[off[jev[cjet]] + cidx], minlength=nj) > 0
        # off-axis cones: every particle of the jet's event against both cones
        npart = cnt[jev]
        pj = np.repeat(np.arange(nj), npart)
        pidx = off[jev[pj]] + (np.arange(npart.sum()) - np.repeat(np.cumsum(npart) - npart, npart))
        deta2 = (ETA[pidx] - jeta[pj]) ** 2
        rho = np.zeros(nj)
        for sign in (+1.0, -1.0):
            dphi = np.mod(PHI[pidx] - jphi[pj] - sign * np.pi / 2 + np.pi, 2 * np.pi) - np.pi
            inc = (deta2 + dphi ** 2) < R2
            rho += np.bincount(pj[inc], weights=PT[pidx[inc]], minlength=nj) / area / 2.0
        ptc = jpt - area * rho
        jw = win[jev]
        for name, lo, hi in (("star_jets_eta0", 0.0, 0.5), ("star_jets_eta1", 0.5, 0.9)):
            sel = has_ch & (np.abs(jeta) >= lo) & (np.abs(jeta) < hi)
            deta = 2 * (hi - lo)
            a, a2 = _hist_by_window(ptc[sel], np.full(sel.sum(), 1.0 / deta), jw[sel], nwin, e)
            acc[name + "_w"] += a
            acc[name + "_w2"] += a2


OBS_HAD = ("phenix_pi0", "star_pi0", "star_eta_pi0_num", "star_eta_pi0_den")
OBS_JET = ("star_jets_eta0", "star_jets_eta1")


def compute(files, data):
    """Per-window sums over the files of one run and the windows' sigma and N.  Files may
    have different windows (the design jobs and the extra low-window jobs): the run's windows
    are the union, matched by their edges; each window's cross section is the files'
    estimates weighted by their accepted events."""
    fds = [(path, read_file(path)) for path in files]
    edges = sorted({tuple(w) for _, fd in fds for w in fd["bins"].tolist()})
    nwin = len(edges)
    acc = {}
    for name in OBS_HAD + OBS_JET:
        key = "star_eta_pi0" if name.startswith("star_eta_pi0") else name
        nb = len(data[key]["edges"])
        acc[name + "_w"] = np.zeros((nwin, nb))
        acc[name + "_w2"] = np.zeros((nwin, nb))
    sig_w, nacc_tot, n_ev = np.zeros(nwin), np.zeros(nwin), np.zeros(nwin)
    for path, fd in fds:
        gidx = np.array([edges.index(tuple(w)) for w in fd["bins"].tolist()])
        nloc = len(fd["sigma"])
        part = {k: np.zeros((nloc, v.shape[1])) for k, v in acc.items()}
        hadron_obs(fd, data, part)
        jet_obs(fd, data, part)
        for k in acc:
            acc[k][gidx] += part[k]
        sig_w[gidx] += fd["n_acc"] * fd["sigma"]
        nacc_tot[gidx] += fd["n_acc"]
        n_ev[gidx] += np.bincount(fd["win"], minlength=nloc)
    sigma = sig_w / np.maximum(nacc_tot, 1)
    return dict(acc, sigma=sigma, n_events=n_ev, pthat_bins=np.asarray(edges, dtype=np.float64),
                n_files=len(files))


def combine(o, name, data, windows=None):
    """The observable [data units] and its statistical error from the per-window sums
    (windows: indices to include, default all)."""
    k = np.arange(len(o["sigma"])) if windows is None else np.asarray(windows)
    scale = o["sigma"][k] / np.maximum(o["n_events"][k], 1)          # mb per event
    if name == "star_eta_pi0":
        num = (scale[:, None] * o["star_eta_pi0_num_w"][k]).sum(0)
        den = (scale[:, None] * o["star_eta_pi0_den_w"][k]).sum(0)
        nv = (scale[:, None] ** 2 * o["star_eta_pi0_num_w2"][k]).sum(0)
        dv = (scale[:, None] ** 2 * o["star_eta_pi0_den_w2"][k]).sum(0)
        r = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
        err = r * np.sqrt(np.divide(nv, num ** 2, out=np.zeros_like(num), where=num > 0)
                          + np.divide(dv, den ** 2, out=np.zeros_like(den), where=den > 0))
        return r, err
    e = np.asarray(data[name]["edges"], dtype=np.float64)
    dpt = e[:, 1] - e[:, 0]
    val = (scale[:, None] * o[name + "_w"][k]).sum(0) / dpt
    err = np.sqrt((scale[:, None] ** 2 * o[name + "_w2"][k]).sum(0)) / dpt
    if data[name]["kind"] == "jet_d2sigma":                           # mb -> pb
        val, err = val * 1e9, err * 1e9
    return val, err


def run_files(run):
    d = C.point_dir(run)
    return sorted(glob.glob(str(d / "*_hadrons_jet_frag.h5")))


def process(run, files=None, out=None):
    data = load_data()
    files = files or run_files(run)
    if not files:
        raise FileNotFoundError(f"no jet_frag files for {run}")
    o = compute(files, data)
    out = Path(out or C.obs_path(run))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **o)
    return out, int(o["n_events"].sum())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="*", help="run names under PROD_DIR/events")
    p.add_argument("--design", action="store_true", help="the design points (design.json)")
    p.add_argument("--all", action="store_true", help="every run under PROD_DIR/events")
    p.add_argument("--dir", default=None, help="read this directory instead (one run)")
    p.add_argument("--force", action="store_true", help="recompute existing .npz")
    p.add_argument("-j", type=int, default=1, help="runs in parallel")
    a = p.parse_args(argv)
    if a.dir:
        if len(a.runs) != 1:
            sys.exit("observables.py: --dir needs exactly one run name (the output's name)")
        files = sorted(glob.glob(os.path.join(a.dir, "*_hadrons_jet_frag.h5")))
        out, n = process(a.runs[0], files)
        print(f"{a.runs[0]}: {n} events -> {out}")
        return 0
    runs = list(a.runs)
    if a.design:
        import make_design as D
        runs = list(D.load()["points"]) + runs
    if a.all:
        runs = sorted(x.name for x in (C.PROD_DIR / "events").iterdir() if x.is_dir())
    todo = [r for r in runs if a.force or not C.obs_path(r).exists()]
    with ProcessPoolExecutor(max_workers=max(1, a.j)) as ex:
        for r, (out, n) in zip(todo, ex.map(process, todo)):
            print(f"{r}: {n} events -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
