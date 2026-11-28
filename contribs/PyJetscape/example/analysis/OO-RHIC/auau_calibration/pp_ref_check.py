#!/usr/bin/env python3
"""
analysis/OO-RHIC/auau_calibration/pp_ref_check.py -- the p+p reference of the Au+Au R_AA (auau_raa.py) against the
measured p+p spectra and against Pythia 8, for auau_jet_raa.ipynb ("The p+p reference"):

    python pp_ref_check.py -j 6          # -> results/pp_ref_check.json

Samples (hard event only, the same code for all):
- pp_ref: the full chain in p+p (pp200_isr_rhictune.xml; strings, MUSIC where a string exists, the jet
  in vacuum: vacuum_jet 1, Matter in_vac 1 to Q0 2 GeV, no LBT, iColored), /data/OO-RHIC_phys/pp_ref_w*,
  the jet_frag files (no bulk, no parton |y| cut: the inclusive spectra the data measure);
- pythia8_default (Monash) and pythia8_detroit (STAR's PYTHIA 8 tune), refs/pythia_ref.py, the same p-hat
  windows, /data/OO-RHIC_pp_refs (whole events, so their jets include Pythia's underlying event).
STAR's charged-jet R_AA reference is PYTHIA 6 Perugia 2012 with PARP(90) = 0.213 (2006.00582 Sec. IX),
which is not available here; Detroit and Monash are PYTHIA 8 stand-ins for it.
Decays: iColored (X-SCAPE iColoredHadronization.cc, no <weak_decays> in the XML) runs Pythia with
HadronLevel:Decay on, ParticleDecays:limitTau0 on, tau0Max 10 mm: pi0, eta and all resonances decay, the
weakly decaying K0S, Lambda, Sigma+-, Xi, Omega do not (the data's "primary" particles: STAR subtracts the
weak-decay feed-down from its pi+- and h+-).  pythia8_* (pythia_ref.py's default) decays everything except
pi0 and eta; pythia8_*_ic uses iColored's settings (--line 111:mayDecay = on, 221:mayDecay = on,
ParticleDecays:limitTau0 = on, ParticleDecays:tau0Max = 10.0) and is the like-for-like comparison.
- pp19_ic: JETSCAPE's PP19 p+p tune (arXiv:1910.05481; the p+p baseline of 2204.01163): PythiaGun with
  Pythia's ISR and MPI (8.312 defaults, FSR off), partons to Matter in vacuum (Q0 1 GeV, vir_factor 0.25),
  Colorless hadronization with the same decays (refs/pp19_200.xml, refs/hadronize_pp19_icdecays.xml,
  prod_AuAu_0_10_jet/run_prod_vacuum.py --n-frag 1; /data/OO-RHIC_pp_refs/pp19_icdecays).

Observables, as cross sections [mb]:
- pi_star: (pi+ + pi-)/2, |y| < 0.5, E d3sigma/dp3 (STAR, 1110.0579's p+p, sigma_NSD 30 mb);
- pi_phenix: (pi+ + pi-)/2, |y| < 0.35, E d3sigma/dp3 (PHENIX pi0, data/data.json);
- h_star: (h+ + h-)/2, |eta| < 0.5, E d3sigma/dp3 (STAR, nucl-ex/0305015's p+p NSD);
- chjet_R*: charged jets anti-kT R = 0.2, 0.3, 0.4, constituents pT > 0.2 GeV |eta| < 1, |eta_jet| < 1 - R,
  d2sigma/dpT deta, all jets and with a 5 GeV leading charged hadron (_lead5); _uesub: the jet pT minus
  the mean charged pT in the two cones of radius R at phi +- pi/2 (the underlying event; perp_R* / njet_R*
  is its mean per jet).
The p-hat windows start at 5 GeV, so hadrons below ~6 GeV miss part of their cross section. Errors from
the summed squared weights.  Model results, not vetted.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import _paths  # noqa: E402,F401  (analysis/OO-RHIC and its test directories on sys.path)
ROOT = _paths.ROOT                           # analysis/OO-RHIC: results/, refs/, data/
import auau_raa as R  # noqa: E402
import oo_analysis_rhic as A  # noqa: E402
import oo_bias as B  # noqa: E402

SAMPLES = {"pp_ref": ("/data/OO-RHIC_phys", "pp_ref_w*/*_hadrons_jet_frag.h5"),
           "pythia8_default": ("/data/OO-RHIC_pp_refs", "pythia8_default/*_hadrons_jet_frag.h5"),
           "pythia8_detroit": ("/data/OO-RHIC_pp_refs", "pythia8_detroit/*_hadrons_jet_frag.h5"),
           "pythia8_default_ic": ("/data/OO-RHIC_pp_refs", "pythia8_default_icdecays/*_hadrons_jet_frag.h5"),
           "pythia8_detroit_ic": ("/data/OO-RHIC_pp_refs", "pythia8_detroit_icdecays/*_hadrons_jet_frag.h5"),
           "pp19_ic": ("/data/OO-RHIC_pp_refs", "pp19_icdecays/*_hadrons_jet_frag.h5")}
PI_STAR = np.array([5, 5.5, 6, 6.5, 7, 8, 10, 12, 15.0])
PI_PHX = np.array([5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 12, 14, 16, 18, 20.0])
H_STAR = np.array([5.1, 6, 7, 8, 9, 10, 12.0])
CHJ = R.CHJ_EDGES


def files(root, pattern):
    return sorted(f for f in glob.glob(f"{root}/{pattern}")
                  if os.path.exists(f.replace("_hadrons_jet_frag.h5", ".json")) or os.path.exists(f.replace(".h5", ".json")))


def whist(ev, x, w, edges, n):
    """Per event: sum of w in the bins of x."""
    out = np.zeros((n, len(edges) - 1))
    b = np.digitize(x, edges) - 1
    ok = (b >= 0) & (b < len(edges) - 1)
    np.add.at(out, (ev[ok], b[ok]), w[ok])
    return out


def one_file(path):
    import h5py
    import hdf5plugin  # noqa: F401
    with h5py.File(path) as h:
        n = int(h.attrs["nunits_written"])
        sig = np.asarray(h.attrs["pthat_bin_sigma_gen"], float)
        nacc = np.asarray(h.attrs["pthat_bin_n_accepted"], float)
        bins = [tuple(map(float, w)) for w in np.asarray(h.attrs["pthat_bins"]).reshape(-1, 2)]
        win = h["units/pthat_bin"][:n].astype(int)
        pthat = h["units/pthat"][:n].astype(float)
        pa, ca, pid = B._units(h, n, charged=False, with_pid=True)
    ev = np.repeat(np.arange(n), ca)
    pt, eta, phi = A._kin(pa)
    E_, pz_ = pa[:, 0], pa[:, 3]
    y = 0.5 * np.log(np.clip(E_ + pz_, 1e-12, None) / np.clip(E_ - pz_, 1e-12, None))
    ch = np.isin(np.abs(pid), B.charged_ids())
    inv = 1.0 / (2 * np.pi * np.maximum(pt, 1e-9))
    out = {}
    m = (np.abs(pid) == 211) & (np.abs(y) < 0.5)
    out["pi_star"] = whist(ev[m], pt[m], inv[m] / 2 / 1.0, PI_STAR, n)
    m = (np.abs(pid) == 211) & (np.abs(y) < 0.35)
    out["pi_phenix"] = whist(ev[m], pt[m], inv[m] / 2 / 0.7, PI_PHX, n)
    m = ch & (np.abs(eta) < 0.5)
    out["h_star"] = whist(ev[m], pt[m], inv[m] / 2 / 1.0, H_STAR, n)
    mc = ch & (pt > A.CONST_PTMIN) & (np.abs(eta) < 1.0)
    for Rj in (0.2, 0.3, 0.4):
        je, jpt, jeta, jphi = A._jets(pa[mc], ev[mc], n, Rj, None, ptmin=CHJ[0] * 0.5)[:4]
        k = np.abs(jeta) < 1.0 - Rj
        deta = 2 * (1.0 - Rj)
        out[f"chjet_R{Rj}"] = whist(je[k], jpt[k], np.full(k.sum(), 1.0 / deta), CHJ, n)
        # underlying event: charged pT in the two cones at phi +- pi/2 (same eta), averaged, per jet
        perp = np.zeros(len(je))
        ec, ptc, etac, phic = ev[mc], pt[mc], eta[mc], phi[mc]
        st = np.searchsorted(ec, np.arange(n + 1))
        for i in np.nonzero(k)[0]:
            s0, s1 = st[je[i]], st[je[i] + 1]
            for sgn in (1, -1):
                dphi = A._dphi(phic[s0:s1], jphi[i] + sgn * np.pi / 2)
                perp[i] += 0.5 * ptc[s0:s1][np.hypot(etac[s0:s1] - jeta[i], dphi) < Rj].sum()
        out[f"chjet_R{Rj}_uesub"] = whist(je[k], (jpt - perp)[k], np.full(k.sum(), 1.0 / deta), CHJ, n)
        out[f"perp_R{Rj}"] = whist(je[k], jpt[k], perp[k], CHJ, n)          # sum of perp-cone pT per jet bin
        out[f"njet_R{Rj}"] = whist(je[k], jpt[k], np.ones(k.sum()), CHJ, n)
        lk = R.lead_ok(je, jeta, jphi, ev[mc], pt[mc], eta[mc], phi[mc], Rj)
        out[f"chjet_R{Rj}_lead5"] = whist(je[k & lk], jpt[k & lk], np.full((k & lk).sum(), 1.0 / deta), CHJ, n)
    return bins, sig, nacc, win, pthat, out


EDGES = {"pi_star": PI_STAR, "pi_phenix": PI_PHX, "h_star": H_STAR}


def collect(fs, j):
    with ProcessPoolExecutor(max_workers=j) as ex:
        res = list(ex.map(one_file, fs))
    wl = sorted({w for r in res for w in r[0]})
    sig_n, nacc = np.zeros(len(wl)), np.zeros(len(wl))
    win, pth, H = [], [], {}
    for bins, sig, na, w, p, o in res:
        g = np.array([wl.index(x) for x in bins])
        sig_n[g] += sig * na; nacc[g] += na
        win.append(g[w]); pth.append(p)
        for k, v in o.items():
            H.setdefault(k, []).append(v)
    win, pth = np.concatenate(win), np.concatenate(pth)
    w, *_ = A.stitch_pieces(wl, win, pth, sig_n / np.maximum(nacc, 1))
    spec = {}
    for k, v in H.items():
        e = EDGES.get(k, CHJ)
        X = w[:, None] * np.concatenate(v)
        spec[k] = {"edges": e.tolist(), "value": (X.sum(0) / np.diff(e)).tolist(),
                   "err": (np.sqrt((X ** 2).sum(0)) / np.diff(e)).tolist()}
    return {"events": len(win), "files": len(fs), "windows": wl, "spectra": spec}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-j", type=int, default=6)
    a = ap.parse_args(argv)
    out = {"units": "E d3sigma/dp3 [mb GeV^-2] for pi_*, h_star; d2sigma/dpT deta [mb/GeV] for chjet_*", "samples": {}}
    for name, (root, pat) in SAMPLES.items():
        fs = files(root, pat)
        if not fs:
            print(f"{name}: no files")
            continue
        out["samples"][name] = collect(fs, a.j)
        print(f"{name}: {out['samples'][name]['events']} events, {len(fs)} files")
    fn = ROOT / "results" / "pp_ref_check.json"
    fn.write_text(json.dumps(out))
    print("->", fn)


if __name__ == "__main__":
    sys.exit(main())
