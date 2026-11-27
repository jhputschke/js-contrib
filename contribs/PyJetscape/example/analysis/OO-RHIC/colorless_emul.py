#!/usr/bin/env python3
"""
analysis/OO-RHIC/colorless_emul.py -- X-SCAPE's ColorlessHadronization re-implemented in Python on standalone
Pythia (8.312), for ablations of its steps (icolored_diag.py):

  string ends   quarks paired by smallest dR (rapidity, phi) in record order; an unpaired last quark gets a fake
                remnant quark (pT 0.28 GeV, pz = +-sqrt(s)/6, random sign); no quark at all: two fakes;
  gluons        each gluon to the string whose two ends are closest on average in dR, then chained from the
                first end by nearest dR;
  close pairs   every pair of partons with |p_i - p_j| < 4 Lambda_QCD (0.8 GeV): the one with the larger |pz|
                gets pz += sign * 4 Lambda_QCD (ColorlessHadronization.cc: 'too close partons');
  Pythia        massive partons (m0), E from p and m0, string fragmentation, decays tau0 < 10 mm.

Only the positive partons (pstat 0, 1000 relabelled 0, recoils 1, photons 22) are emulated: the hole hadrons
(pstat -1) do not enter the high-pT yields.

    python colorless_emul.py -j 16          # -> ~/prod/AuAu-RHIC_hadtests/diag/colorless_emul.json

Variants: full (as X-SCAPE), noshift (no close-pair shift), norecoil (recoils dropped), jetonly (no recoils, no
shift).  Model results, not vetted.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import icolored_diag as I

P_FAKE = 200.0 / 6.0
SHIFT = 4 * 0.2
VARIANTS = ("full", "noshift", "norecoil", "jetonly")
_PY = None


def pythia():
    global _PY
    if _PY is None:
        import pythia8
        _PY = pythia8.Pythia("", False)
        for s in ("ProcessLevel:all = off", "PartonLevel:FSR=off", "HadronLevel:Decay = on",
                  "ParticleDecays:limitTau0 = on", "ParticleDecays:tau0Max = 10.0", "Print:quiet = on",
                  "Check:event = off"):
            _PY.readString(s)
        _PY.init()
    return _PY


def yphi(p):   # p: (N, 4) E px py pz
    y = 0.5 * np.log(np.maximum(p[:, 0] + p[:, 3], 1e-300) / np.maximum(p[:, 0] - p[:, 3], 1e-300))
    return y, np.arctan2(p[:, 2], p[:, 1])


def dR(y1, f1, y2, f2):
    d = np.abs(f1 - f2)
    d = np.where(d > np.pi, 2 * np.pi - d, d)
    return np.hypot(y1 - y2, d)


def colorless_strings(pid, p, rng, shift=True):
    """pid (N,), p (N, 4: E px py pz) -> pid, col, acol, p of the partons handed to Pythia."""
    pid = list(pid.astype(int)); p = [np.array(r, float) for r in p]
    isq = [i for i, q in enumerate(pid) if abs(q) <= 6]
    nstr = max(int(len(isq) / 2 + 0.6), 1)
    done = [0] * len(pid)
    sign = 1.0 if rng.random() > 0.5 else -1.0
    def fake(sz):
        q = np.array([0.0, 0.2, 0.2, sz * P_FAKE]); q[0] = np.sqrt((q[1:] ** 2).sum())
        pid.append(1); p.append(q); done.append(1)
        return len(pid) - 1
    one, two = [], []
    if not isq:
        a = fake(sign); b = fake(-sign); isq += [a, b]; one.append(a); two.append(b)
    k = 0
    while k < len(isq):
        iq = isq[k]
        if not done[iq]:
            done[iq] = 1; one.append(iq)
            cand = [j for j in isq if j != iq and not done[j]]
            if cand:
                P = np.array([p[j] for j in cand]); y, f = yphi(P)
                y0, f0 = yphi(p[iq][None])
                j = cand[int(np.argmin(dR(y0[0], f0[0], y, f)))]
                done[j] = 1; two.append(j)
            else:
                j = fake(sign); isq.append(j); two.append(j)
        k += 1
    nstr = len(one)
    P = np.array(p); y, f = yphi(P)
    my = np.zeros(len(pid), int)
    gl = [i for i, q in enumerate(pid) if q == 21]
    for g in gl:
        d = [(dR(y[g], f[g], y[one[s]], f[one[s]]) + dR(y[g], f[g], y[two[s]], f[two[s]])) / 2 for s in range(nstr)]
        my[g] = int(np.argmin(d))
    col = [0] * len(pid); acol = [0] * len(pid); lab = 102
    for s in range(nstr):
        t = one[s]
        if pid[t] > 0:
            col[t] = lab
        else:
            acol[t] = lab
        lab += 1; link = t
        while True:
            cand = [g for g in gl if not done[g] and my[g] == s]
            if not cand:
                break
            g = cand[int(np.argmin(dR(y[link], f[link], y[cand], f[cand])))]
            done[g] = 1
            if col[link] == lab - 1:
                col[g], acol[g] = lab, lab - 1
            else:
                col[g], acol[g] = lab - 1, lab
            lab += 1; link = g
        if col[link] == lab - 1:
            col[two[s]], acol[two[s]] = 0, lab - 1
        else:
            col[two[s]], acol[two[s]] = lab - 1, 0
    for q in isq:
        pid[q] = abs(pid[q]) if col[q] != 0 else -abs(pid[q])
    if shift:   # ColorlessHadronization.cc's close-pair loop (the outer parton's values are read once)
        for i in range(len(pid)):
            pi = p[i].copy()
            for j in range(i + 1, len(pid)):
                pj = p[j]
                if np.sqrt(((pi[1:] - pj[1:]) ** 2).sum()) < SHIFT:
                    if abs(pi[3]) >= abs(pj[3]):
                        s_ = 1.0 if pi[3] >= 0 else -1.0
                        q = pi.copy(); q[3] += s_ * SHIFT; p[i] = q
                    else:
                        s_ = 1.0 if pj[3] >= 0 else -1.0
                        q = pj.copy(); q[3] += s_ * SHIFT; p[j] = q
    return np.array(pid), np.array(col), np.array(acol), np.array(p)


def one_file(args):
    fi, variants, seed = args
    ch = I.charged_ids()
    d = I.read(fi)
    py = pythia(); ev = py.event
    rng = np.random.default_rng(seed)
    out = {v: {"n8": 0, "E_eta": np.zeros(6), "pt1": 0.0, "events": 0, "fail": 0} for v in variants}
    bins = [0, 1, 2, 3, 4, 6, 20]
    for u in range(d["n"]):
        par = d["par"][d["po"][u]:d["po"][u + 1]]
        st = par[:, 2].astype(int)
        for v in variants:
            keep = np.isin(st, (0, 1000, 22)) | ((st == 1) & (v in ("full", "noshift")))
            q = par[keep]
            isphot = q[:, 1] == 22
            pid, col, acol, p = colorless_strings(q[~isphot, 1], q[~isphot][:, [3, 4, 5, 6]], rng,
                                                  shift=v in ("full", "norecoil"))
            ev.reset()
            for k in range(len(pid)):
                m = py.particleData.m0(int(pid[k]))
                e = np.sqrt((p[k, 1:] ** 2).sum() + m * m)
                ev.append(int(pid[k]), 23, int(col[k]), int(acol[k]), p[k, 1], p[k, 2], p[k, 3], e, m)
            for r in q[isphot]:
                ev.append(22, 23, 0, 0, r[4], r[5], r[6], r[3], 0.0)
            py.rndm.init(int(rng.integers(1, 900000000)))
            o = out[v]; o["events"] += 1
            if not py.next():
                o["fail"] += 1
                continue
            H = np.array([[ev[i].e(), ev[i].px(), ev[i].py(), ev[i].pz(), ev[i].id()] for i in range(ev.size()) if ev[i].isFinal()])
            pt = np.hypot(H[:, 1], H[:, 2]); eta = np.arcsinh(H[:, 3] / np.maximum(pt, 1e-12))
            chg = np.isin(np.abs(H[:, 4]).astype(int), ch)
            o["n8"] += int(((pt > 8) & (np.abs(eta) < 1) & chg).sum())
            o["pt1"] += float(pt[np.abs(eta) < 1].sum())
            o["E_eta"] += np.histogram(np.abs(eta), bins, weights=H[:, 0])[0]
    return out


SAMPLES = {"vacuum": (f"{I.HT}/from3090/pp_ref_w{{w}}", 10), "T1": (f"{I.HT}/auau_matteronly_w{{w}}", 10),
           "B1": (f"{I.HT}/from3090/auau_med_w{{w}}", 40)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-j", type=int, default=16)
    ap.add_argument("--windows", default="18_23,23_30,30_40")
    ap.add_argument("--out", default=f"{I.HT}/diag/colorless_emul.json")
    a = ap.parse_args(argv)
    res = {}
    with ProcessPoolExecutor(a.j) as ex:
        for s, (g, nf) in SAMPLES.items():
            for w in a.windows.split(","):
                fs = sorted(glob.glob(f"{g.format(w=w)}/*_hadrons_jet_frag.h5"))[:nf]
                rs = list(ex.map(one_file, [(f, VARIANTS, 1000 + k) for k, f in enumerate(fs)]))
                tot = {}
                for v in VARIANTS:
                    n = sum(r[v]["events"] for r in rs)
                    tot[v] = {"events": n, "fail": sum(r[v]["fail"] for r in rs),
                              "n8": sum(r[v]["n8"] for r in rs) / n, "pt1": sum(r[v]["pt1"] for r in rs) / n,
                              "E_eta": (sum(r[v]["E_eta"] for r in rs) / n).round(1).tolist()}
                res[f"{s}_w{w}"] = tot
                print(f"{s}_w{w}: " + "  ".join(f"{v} n8 {tot[v]['n8']:.3f}" for v in VARIANTS), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("->", a.out)


if __name__ == "__main__":
    raise SystemExit(main())
