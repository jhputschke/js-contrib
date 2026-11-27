#!/usr/bin/env python3
"""
analysis/OO-RHIC/icolored_diag.py -- why iColored makes ~2x the high-pT hadrons of Colorless from the
same partons in the medium (auau_jet_raa.ipynb, chain_diag.py), while the two agree in vacuum.

iColored is replayed event by event in standalone Pythia (the pythia8 module of the job's version,
8.312) from the stored hadronization_input/ (exactly the partons, colours and seeds iColored handed to
Pythia; the replay reproduces the stored hadrons), so that Pythia's event record can be traced: every
high-pT hadron back to its primary hadron and its string (colour singlet), and every string parton
back to its input row and from there to partons/ (shower, pstat) or to iColored's own additions
(beam remnants, closure partners on a hole's momentum, soft fake ends).  Colorless is read from the
rehadronized files (prod_OO_isr/rehadronize_colorless.py) of the same partons.

    python icolored_diag.py -j 16              # -> OUT/icolored_diag.npz, .json (OUT = --out)

Per sample (vacuum pp_ref, T1 MATTER only, B1 MATTER+LBT) and window (18-23, 23-30, 30-40 GeV):
 1. bookkeeping per event: energy and |eta|<1 pT of the input classes (jet partons pstat 0, recoils
    pstat 1, holes, remnants, partners, fakes) and of the hadrons, iColored vs Colorless;
 2. z = pT(hadron) / pT(hardest final parton, pstat >= 0, within dR < 0.4) for charged hadrons
    pT > 8 GeV, |eta| < 1, iColored vs Colorless;
 3. per such iColored hadron: its string's partons by class, number of showers, invariant mass,
    summed pT, the hardest string parton, and the string partons' summed pT within dR < 0.4 of
    the hadron; whether it came from a ministring (Pythia status 81/82).
Model results, not vetted.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HT = os.path.expanduser("~/prod/AuAu-RHIC_hadtests")
PH = os.path.expanduser("~/prod/AuAu-RHIC_phys")
WINDOWS = ("18_23", "23_30", "30_40")
PT3C = os.path.expanduser("~/prod/OO-RHIC_ppmed_test")
SAMPLES = {   # name: (iColored files with hadronization_input/, Colorless files of the same partons, windows)
    "vacuum": (f"{HT}/from3090/pp_ref_w{{w}}", f"{PH}/rehad_colorless/pp_ref_w{{w}}", WINDOWS),
    "T1": (f"{HT}/auau_matteronly_w{{w}}", f"{PH}/hadtests/rehad/auau_matteronly_w{{w}}", WINDOWS),
    "B1": (f"{HT}/from3090/auau_med_w{{w}}", f"{PH}/rehad_colorless/auau_med_w{{w}}", WINDOWS),
    # 3c p+p (seeds 880000 / 887000, 500 events): pp_ref = the tune's scheme (Matter in_vac 1, Q0 2),
    # pp_med = the A+A scheme (in_vac 0, QS 0.9 outside the medium, LBT inside), split by has_medium
    "pp_tune": (f"{PT3C}/pp_ref_w{{w}}", f"{HT}/diag/rehad_colorless/pp_ref_w{{w}}", ("05_07", "30_40")),
    "pp_aa": (f"{PT3C}/pp_med_w{{w}}", f"{HT}/diag/rehad_colorless/pp_med_w{{w}}", ("05_07", "30_40")),
}
CLASSES = ("jet", "recoil", "remnant", "partner", "fake", "unknown")   # input-row classes
PT_MIN, ETA_MAX, DR = 8.0, 1.0, 0.4
ZBINS = np.linspace(0, 2, 41)
CHARGED = None
_PY = None


def charged_ids():
    global CHARGED
    if CHARGED is None:
        import pythia8
        p = pythia8.Pythia("", False)
        ids = [i for i in range(1, 6000) if p.particleData.isParticle(i)
               and p.particleData.charge(i) != 0 and p.particleData.isHadron(i)] + [11, 13]
        CHARGED = np.array(sorted(set(ids)))
    return CHARGED


def pythia():
    global _PY
    if _PY is None:
        import pythia8
        _PY = pythia8.Pythia("", False)
        for s in ("ProcessLevel:all = off", "PartonLevel:FSR=off", "HadronLevel:Decay = on",
                  "ParticleDecays:limitTau0 = on", "ParticleDecays:tau0Max = 10.0", "Print:quiet = on"):
            _PY.readString(s)
        _PY.init()
    return _PY


def kin(p):   # p: (N, 4) E px py pz
    pt = np.hypot(p[:, 1], p[:, 2])
    eta = np.arcsinh(p[:, 3] / np.maximum(pt, 1e-12))
    phi = np.arctan2(p[:, 2], p[:, 1])
    return pt, eta, phi


def dr(eta1, phi1, eta2, phi2):
    d = np.abs(phi1 - phi2)
    d = np.where(d > np.pi, 2 * np.pi - d, d)
    return np.hypot(eta1 - eta2, d)


def zref(hpt, heta, hphi, ppt, peta, pphi):
    """pT(hadron) / pT(hardest parton within DR) (inf if none)."""
    out = np.full(len(hpt), np.inf)
    for k in range(len(hpt)):
        m = dr(heta[k], hphi[k], peta, pphi) < DR
        if m.any():
            out[k] = hpt[k] / ppt[m].max()
    return out


def read(path):
    import h5py
    import hdf5plugin  # noqa: F401
    h = h5py.File(path)
    n = int(h.attrs["nunits_written"])
    d = {"eta_max": float(h.attrs.get("eta_max", 1e9)), "n": n, "po": h["partons/offsets"][: n + 1], "par": h["partons/data"][:],
         "so": h["hadrons/sample_offsets"][:], "uo": h["hadrons/unit_offsets"][: n + 1],
         "hp": h["hadrons/p"][:], "hid": h["hadrons/pid"][:],
         "hst": h["hadrons/pstat"][:] if "hadrons/pstat" in h else None}
    if "hadronization_input" in h:
        d["io"] = h["hadronization_input/offsets"][: n + 1]
        d["inp"] = h["hadronization_input/data"][:]
        d["seed"] = h["units/seed"][:n]
    d["has_medium"] = h["units/has_medium"][:n] if "units/has_medium" in h else np.zeros(n, int)
    h.close()
    return d


def hadrons_of(d, u):
    a = d["uo"][u]
    s0, s1 = d["so"][a], d["so"][a + 1]
    return (d["hp"][s0:s1].astype(float), d["hid"][s0:s1],
            None if d["hst"] is None else d["hst"][s0:s1])


def one(args):
    """One pair of files (iColored with input, Colorless): per-event bookkeeping and per-hadron rows."""
    fi, fc = args
    ch = charged_ids()
    di = read(fi)
    dc = read(fc) if os.path.exists(fc) else None
    py = pythia()
    ev = py.event
    book, rows, zc, reproduced = [], [], [], 0
    for u in range(di["n"]):
        par = di["par"][di["po"][u]:di["po"][u + 1]]   # shower pid pstat E px py pz t x y z mass col acol
        inp = di["inp"][di["io"][u]:di["io"][u + 1]]   # pid col acol E px py pz m
        # input-row classes
        key = {tuple(np.round(r[4:7], 6)): (int(r[0]), int(r[2])) for r in par if r[2] >= 0}
        hole = {tuple(np.round(r[4:7], 6)) for r in par if r[2] == -1}
        cls = np.empty(len(inp), int)
        shw = np.full(len(inp), -1)
        for i, r in enumerate(inp):
            k = tuple(np.round(r[4:7], 6))
            if k in key:
                shw[i], st = key[k]
                cls[i] = 0 if st in (0, 1000) else (1 if st == 1 else 5)   # 1000: PythiaIsrGun hard partons + daughters
            elif r[4] == 0 and r[5] == 0 and abs(r[3] - 1) < 1e-9:
                cls[i] = 4
            elif k in hole:
                cls[i] = 3
            else:
                cls[i] = 2
        # replay
        ev.reset()
        for r in inp:
            ev.append(int(r[0]), 23, int(r[1]), int(r[2]), r[4], r[5], r[6], r[3], r[7])
        py.rndm.init(int(di["seed"][u]) % 900000000 + 1)
        if not py.next():
            continue
        N = ev.size()
        st = np.array([ev[i].status() for i in range(N)])
        m1 = np.array([ev[i].mother1() for i in range(N)])
        m2 = np.array([ev[i].mother2() for i in range(N)])
        P = np.array([[ev[i].e(), ev[i].px(), ev[i].py(), ev[i].pz()] for i in range(N)])
        ids = np.array([ev[i].id() for i in range(N)])
        etas = np.array([ev[i].eta() for i in range(N)])
        fin = np.array([ev[i].isFinal() for i in range(N)]) & (np.abs(etas) <= 20)
        stored = fin & (np.abs(etas) < di["eta_max"])
        hp_s, hid_s, _ = hadrons_of(di, u)
        reproduced += int(stored.sum() == len(hid_s) and np.array_equal(ids[stored], hid_s))
        # bookkeeping
        ppt, peta, pphi = kin(par[:, 3:7])
        ipt, ieta, _ = kin(inp[:, 3:7])
        b = {"has_medium": int(di["has_medium"][u])}
        b.update({"E_in_" + c: float(inp[cls == k, 3].sum()) for k, c in enumerate(CLASSES)})
        b.update({"pt1_in_" + c: float(ipt[(cls == k) & (np.abs(ieta) < 1)].sum()) for k, c in enumerate(CLASSES)})
        b["E_holes"] = float(par[par[:, 2] == -1, 3].sum())
        b["pt1_holes"] = float(ppt[(par[:, 2] == -1) & (np.abs(peta) < 1)].sum())
        hpt, heta, hphi = kin(P[fin])
        hch = np.isin(np.abs(ids[fin]), ch)
        b["E_had_ic"] = float(P[fin, 0].sum())
        b["pt1_had_ic"] = float(hpt[np.abs(heta) < 1].sum())
        b["n8_ic"] = int(((hpt > PT_MIN) & (np.abs(heta) < ETA_MAX) & hch).sum())
        pos = par[:, 2] >= 0
        if dc is not None:
            cp, cid, cst = hadrons_of(dc, u)
            cpt, ceta, cphi = kin(cp)
            pc = cst >= 0
            cch = np.isin(np.abs(cid), ch)
            b["E_had_cl"] = float(cp[pc, 0].sum())
            b["E_had_cl_neg"] = float(cp[~pc, 0].sum())
            b["pt1_had_cl"] = float(cpt[pc & (np.abs(ceta) < 1)].sum())
            b["pt1_had_cl_neg"] = float(cpt[~pc & (np.abs(ceta) < 1)].sum())
            sel = pc & cch & (cpt > PT_MIN) & (np.abs(ceta) < ETA_MAX)
            b["n8_cl"] = int(sel.sum())
            zc.extend(zref(cpt[sel], ceta[sel], cphi[sel], ppt[pos], peta[pos], pphi[pos]).tolist())
        book.append(b)
        # per high-pT iColored hadron: its string
        idx = np.where(fin)[0][(hpt > PT_MIN) & (np.abs(heta) < ETA_MAX) & hch]
        for i in idx:
            j = i
            while abs(st[j]) >= 90 and m1[j] > 0:     # decays back to the primary hadron
                j = m1[j]
            prim = st[j]
            lo, hi = m1[j], max(m2[j], m1[j])
            orig = set()
            for q in range(lo, hi + 1):            # string partons (copies) -> input rows
                k = q
                while abs(st[k]) != 23 and 0 < m1[k] != k:
                    k = m1[k]
                if 1 <= k <= len(inp):
                    orig.add(k - 1)
            orig = np.array(sorted(orig), int)
            h_pt, h_eta, h_phi = kin(P[i][None])
            if len(orig):
                sp = inp[orig, 3:7]
                spt, seta, sphi = kin(sp)
                tot = sp.sum(0)
                mass = float(np.sqrt(max(tot[0] ** 2 - (tot[1:] ** 2).sum(), 0)))
                near = dr(h_eta[0], h_phi[0], seta, sphi) < DR
                c = cls[orig]
                nshow = len(set(shw[orig][(c == 0) | (c == 1)]))
            else:
                spt = np.zeros(0); near = np.zeros(0, bool); c = np.zeros(0, int); mass = 0.0; nshow = 0
            zr = zref(h_pt, h_eta, h_phi, ppt[pos], peta[pos], pphi[pos])[0]
            rows.append([int(di["has_medium"][u]), h_pt[0], zr, prim, len(orig), mass, spt.sum(), spt.max() if len(spt) else 0,
                         spt[near].sum(), spt[near].max() if near.any() else 0, nshow]
                        + [int((c == k).sum()) for k in range(len(CLASSES))]
                        + [float(spt[c == k].sum()) for k in range(len(CLASSES))])
    return book, rows, zc, reproduced, di["n"]


ROWCOLS = (["has_medium", "pt", "z", "prim_status", "n_str_partons", "str_mass", "str_sumpt", "str_maxpt",
            "str_sumpt_near", "str_maxpt_near", "n_showers"]
           + [f"n_{c}" for c in CLASSES] + [f"sumpt_{c}" for c in CLASSES])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-j", type=int, default=16)
    ap.add_argument("--out", default=f"{HT}/diag")
    ap.add_argument("--max-files", type=int, default=None, help="per sample and window")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    arrays, summary = {}, {}
    with ProcessPoolExecutor(max_workers=a.j) as ex:
        for s, (gi, gc, wins) in SAMPLES.items():
            for w in wins:
                fs = sorted(glob.glob(f"{gi.format(w=w)}/*_hadrons_jet_frag.h5"))[: a.max_files]
                pairs = [(f, os.path.join(gc.format(w=w), os.path.basename(f))) for f in fs]
                res = list(ex.map(one, pairs))
                book = [b for r in res for b in r[0]]
                rows = np.array([x for r in res for x in r[1]], float).reshape(-1, len(ROWCOLS))
                zc = np.array([x for r in res for x in r[2]], float)
                nev = len(book)
                tag = f"{s}_w{w}"
                arrays[f"{tag}_rows"] = rows
                arrays[f"{tag}_zcl"] = zc
                keys = book[0].keys() if book else []
                arrays[f"{tag}_book"] = np.array([[b.get(k, np.nan) for k in keys] for b in book])
                summary[tag] = {"files": len(fs), "events": nev,
                                "replay_reproduced": int(sum(r[3] for r in res)),
                                "book_keys": list(keys),
                                **{f"mean_{k}": float(np.nanmean([b.get(k, np.nan) for b in book])) for k in keys}}
                print(tag, len(fs), "files", nev, "events, reproduced", summary[tag]["replay_reproduced"],
                      f"n8 ic/cl {summary[tag].get('mean_n8_ic', 0):.3f}/{summary[tag].get('mean_n8_cl', 0):.3f}",
                      flush=True)
    np.savez_compressed(f"{a.out}/icolored_diag.npz", rowcols=np.array(ROWCOLS), **arrays)
    Path(f"{a.out}/icolored_diag.json").write_text(json.dumps(summary, indent=1))
    print("->", a.out)


if __name__ == "__main__":
    raise SystemExit(main())
