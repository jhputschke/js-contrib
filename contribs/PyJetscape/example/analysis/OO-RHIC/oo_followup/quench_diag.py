#!/usr/bin/env python3
"""
analysis/OO-RHIC/oo_followup/quench_diag.py -- why the model's quenching is small, and where the selection bias
comes from (follow-up A1, A2, B4 of the tuning page, 2026-11-25):

    python quench_diag.py --out /data/OO-RHIC_phys -j 12 eloss   # A1 + A2 -> results/quench_diag_eloss.npz
    python quench_diag.py --out /data/OO-RHIC_phys -j 12 bias    # B4      -> results/quench_diag_bias.npz
    python quench_diag.py summary                                # tables from both

A1, per hard outgoing parton (initiators/, pstat 1000: the final-state showers' initiators after
ISR; pT > 10 GeV, |eta| < 0.5), in oo_med and oo_vac:
  - the energy the jet gives to the medium: sum E of the recoils (partons/, pstat 1) minus sum E of
    the holes (pstat -1) in the parton's hemisphere (|dphi| < pi/2) and inside the cone (dR < R).
    Each scattering is jet + thermal parton (hole) -> jet' + recoil, so recoil - hole is the jet's
    loss (the files conserve energy: initiators = final partons + recoils - holes per event).
    The shower index cannot be used: LBT's outgoing shower partons carry pstat 0, the ISR's index
    space;
  - the matched R = 0.4 jet (the hardest jet within dR < 0.4 of the parton), as pT_jet / pT_parton:
      parton level, shower + ISR partons only (no medium response);
      parton level with the recoils (pstat 1) and the holes (pstat -1) subtracted in the cone;
      hadron level as in oo_analysis_rhic.py (visible hadrons pT > 0.2, holes + partner copies
      subtracted), and without the subtraction.
  The jet-level loss at fixed parton pT, 1 - <r_med>/<r_vac>, is free of the spectrum's slope.
A2, per parton and event: has_medium, MUSIC's tau0, the medium interactions near the jet (holes in
the cone: number, first and last time t), the production point, and the freeze-out time of the
bulk at mid-rapidity (bulk_bg hadrons |eta_s| < 1: tau = sqrt(t^2 - z^2)).
B4, per event of oo_vac (and phys_mb as the reference): the forward estimators (n_mbd, n_epd from
oo_analysis_rhic.py's cache; the string energy per MBD side), the energy the hard process takes
from each nucleus (isr_sub_E_proj / _targ) and x_proj, x_targ of the leading pair.

Weights: sigma_k / N_k per window (stitch_pieces). Centrality by impact parameter, the edges from
phys_mb. Model results, not vetted.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import _paths  # noqa: E402,F401  (analysis/OO-RHIC and its test directories on sys.path)
ROOT = _paths.ROOT                           # analysis/OO-RHIC: results/, refs/, data/
import oo_analysis_rhic as A  # noqa: E402
import oo_bias as B  # noqa: E402
import oo_bias_jets as J  # noqa: E402

R = 0.4
PT_MIN, ETA_MAX = 10.0, 0.5
SQRTS = 200.0


def _files(fam):
    comps, _ = J.components(fam)
    return [f for nm, _ in comps for f in J.finished_files(nm)]


def _match(je, jpt, jeta, jphi, ev, eta, phi):
    """pT of the hardest jet within dR < R of each parton (0 if none)."""
    out = np.zeros(len(ev))
    for i in range(len(ev)):
        q = je == ev[i]
        if not q.any():
            continue
        dr = np.hypot(jeta[q] - eta[i], A._dphi(jphi[q], phi[i]))
        k = dr < R
        if k.any():
            out[i] = jpt[q][k].max()
    return out


def eloss_file(path):
    import h5py
    import hdf5plugin  # noqa: F401
    bf = path.replace("_jet_frag.h5", "_bulk_bg.h5")
    with h5py.File(path) as h:
        n = int(h.attrs["nunits_written"])
        sig = np.asarray(h.attrs["pthat_bin_sigma_gen"], float)
        nacc = np.asarray(h.attrs["pthat_bin_n_accepted"], float)
        bins = [tuple(map(float, w)) for w in np.asarray(h.attrs["pthat_bins"]).reshape(-1, 2)]
        u = {k: h[f"units/{k}"][:n] for k in ("glauber_b", "pthat_bin", "pthat", "has_medium", "tau0_music")}
        io = h["initiators/offsets"][: n + 1]; ini = h["initiators/data"][: io[-1]]
        po = h["partons/offsets"][: n + 1]; par = h["partons/data"][: po[-1]]
        pa, ca, pid = B._units(h, n, charged=False, with_pid=True)
        neg = A.negative_partons(h, n)
    # freeze-out time of the bulk at mid-rapidity, per event
    tauf = np.full(n, np.nan)
    try:
        with h5py.File(bf) as hb:
            uo = hb["hadrons/unit_offsets"][: n + 1]
            x = hb["hadrons/x"][: uo[-1]].astype(float)
        evb = np.repeat(np.arange(n), np.diff(uo))
        t, z = x[:, 0], x[:, 3]
        ok = np.abs(z) < np.tanh(1.0) * np.abs(t)
        tau = np.sqrt(np.maximum(t ** 2 - z ** 2, 0))
        if ok.any():
            for e in np.unique(evb[ok]):
                tauf[e] = np.median(tau[ok & (evb == e)])
    except (OSError, KeyError):
        pass
    evi = np.repeat(np.arange(n), np.diff(io))
    evp = np.repeat(np.arange(n), np.diff(po))
    m = ini[:, 2] == 1000
    ipt = np.hypot(ini[:, 3], ini[:, 4])
    ieta = np.arcsinh(ini[:, 5] / np.maximum(ipt, 1e-9))
    sel = m & (ipt > PT_MIN) & (np.abs(ieta) < ETA_MAX)
    I = ini[sel]; iev = evi[sel]; ipt, ieta = ipt[sel], ieta[sel]
    iphi = np.arctan2(I[:, 4], I[:, 3])
    # energy given to the medium: recoils - holes in the parton's hemisphere and cone
    med = np.isin(par[:, 2], (1, -1))
    sgn = np.where(par[:, 2] == 1, 1.0, -1.0)
    mpt = np.hypot(par[:, 4], par[:, 5])
    meta = np.arcsinh(par[:, 6] / np.maximum(mpt, 1e-9)); mphi = np.arctan2(par[:, 5], par[:, 4])
    e_hemi, e_cone = np.zeros(len(I)), np.zeros(len(I))
    for k in range(len(I)):
        q = med & (evp == iev[k])
        if not q.any():
            continue
        dphi = np.abs(A._dphi(mphi[q], iphi[k]))
        hemi = dphi < np.pi / 2
        cone = np.hypot(meta[q] - ieta[k], dphi) < R
        e_hemi[k] = (sgn[q] * par[q, 3])[hemi].sum()
        e_cone[k] = (sgn[q] * par[q, 3])[cone].sum()
    # parton-level jets: shower (1000) + ISR (0) partons; with the recoils (1) and the holes (-1)
    p4 = par[:, 3:7]
    base = np.isin(par[:, 2], (0, 1000))
    je, jpt, jeta, jphi = A._jets(p4[base], evp[base], n, R, None, ptmin=3.0)
    r_par = _match(je, jpt, jeta, jphi, iev, ieta, iphi)
    wr = base | (par[:, 2] == 1)
    je, jpt, jeta, jphi, jx, jy = A._jets(p4[wr], evp[wr], n, R, None, ptmin=3.0, full=True)
    holes = par[:, 2] == -1
    hneg = np.column_stack([evp[holes], p4[holes]]) if holes.any() else np.zeros((0, 5))
    jsub = A.subtract_negatives(je, jeta, jphi, jx, jy, hneg, R)
    r_par_resp = _match(je, jsub, jeta, jphi, iev, ieta, iphi)
    # hadron-level jets as in oo_analysis_rhic.py
    ev_a = np.repeat(np.arange(n), ca)
    pt, eta, _ = A._kin(pa)
    vis = ~np.isin(np.abs(pid), A.INVISIBLE) & (pt > A.CONST_PTMIN) & (np.abs(eta) < 1.6)
    je, jpt, jeta, jphi, jx, jy = A._jets(pa[vis], ev_a[vis], n, R, None, ptmin=3.0, full=True)
    jsub = A.subtract_negatives(je, jeta, jphi, jx, jy, neg, R)
    r_had = _match(je, jsub, jeta, jphi, iev, ieta, iphi)
    r_had_raw = _match(je, jpt, jeta, jphi, iev, ieta, iphi)
    # medium interactions near the jet: holes within dR < R of the parton
    nh, t0h, t1h, eh = np.zeros(len(I)), np.full(len(I), np.nan), np.full(len(I), np.nan), np.zeros(len(I))
    if holes.any():
        hp = par[holes]; hev = evp[holes]
        hpt = np.hypot(hp[:, 4], hp[:, 5]); heta = np.arcsinh(hp[:, 6] / np.maximum(hpt, 1e-9)); hphi = np.arctan2(hp[:, 5], hp[:, 4])
        for k in range(len(I)):
            q = hev == iev[k]
            if not q.any():
                continue
            ins = np.hypot(heta[q] - ieta[k], A._dphi(hphi[q], iphi[k])) < R
            if ins.any():
                nh[k] = ins.sum(); eh[k] = hp[q][ins, 3].sum()
                t0h[k] = hp[q][ins, 7].min(); t1h[k] = hp[q][ins, 7].max()
    out = {"ev": iev, "pid": I[:, 1], "pt": ipt, "E": I[:, 6], "eta": ieta, "x0": I[:, 7], "y0": I[:, 8],
           "phi": iphi, "e_med_hemi": e_hemi, "e_med_cone": e_cone, "r_par": r_par / ipt, "r_par_resp": r_par_resp / ipt,
           "r_had": r_had / ipt, "r_had_raw": r_had_raw / ipt, "n_holes": nh, "E_holes": eh,
           "t_first": t0h, "t_last": t1h}
    evd = {"b": u["glauber_b"].astype(float), "win": u["pthat_bin"].astype(int), "pthat": u["pthat"].astype(float),
           "has_medium": u["has_medium"].astype(int), "tau0": u["tau0_music"].astype(float), "tauf": tauf}
    return bins, sig, nacc, n, evd, out


def run_eloss(j):
    res = {}
    for fam in ("oo_med", "oo_vac"):
        files = _files(fam)
        with ProcessPoolExecutor(max_workers=j) as ex:
            rr = list(ex.map(eloss_file, files))
        wl = sorted({w for r in rr for w in r[0]})
        sig_n, nacc = np.zeros(len(wl)), np.zeros(len(wl))
        evs, ins, off = [], [], 0
        for bins, sig, na, n, evd, o in rr:
            g = np.array([wl.index(w) for w in bins])
            sig_n[g] += sig * na; nacc[g] += na
            evd = dict(evd); evd["win"] = g[evd["win"]]; evs.append(evd)
            o = dict(o); o["ev"] = o["ev"] + off; ins.append(o); off += n
        ev = {k: np.concatenate([e[k] for e in evs]) for k in evs[0]}
        w, *_ = A.stitch_pieces(wl, ev["win"], ev["pthat"], sig_n / np.maximum(nacc, 1))
        ip = {k: np.concatenate([o[k] for o in ins]) for k in ins[0]}
        for k, v in ev.items():
            res[f"{fam}/ev_{k}"] = v
        res[f"{fam}/ev_w"] = w
        for k, v in ip.items():
            res[f"{fam}/{k}"] = v
        print(f"{fam}: {len(files)} files, {len(ev['b'])} events, {len(ip['pt'])} partons (pT > {PT_MIN:g}, |eta| < {ETA_MAX:g})")
    mb = B.load(A.MB_SAMPLE)
    res["mb_b"] = np.asarray(mb["glauber_b"], float)
    np.savez_compressed(ROOT / "results" / "quench_diag_eloss.npz", **res)
    print("-> results/quench_diag_eloss.npz")


def bias_file(path):
    import h5py
    import hdf5plugin  # noqa: F401
    keys = ("glauber_b", "glauber_ncoll", "pthat", "parton_pt_lead", "parton_pt_sub", "parton_y_lead", "parton_y_sub",
            "isr_sub_E_proj", "isr_sub_E_targ", "isr_sub_pz_proj", "isr_sub_pz_targ", "strings_eloc_mbd_pos",
            "strings_eloc_mbd_neg", "strings_eloc_epd", "remnant_E_proj", "remnant_E_targ", "has_medium")
    with h5py.File(path) as h:
        n = int(h.attrs["nunits_written"])
        return {k: h[f"units/{k}"][:n].astype(float) for k in keys}


def mb_units(j):
    root = Path(B.OUT) / A.MB_SAMPLE
    files = sorted(str(f) for f in root.glob("*_bulk_bg.h5"))
    keys = ("glauber_b", "glauber_ncoll", "strings_eloc_mbd_pos", "strings_eloc_mbd_neg", "strings_eloc_epd")

    def one(f):
        import h5py
        import hdf5plugin  # noqa: F401
        with h5py.File(f) as h:
            n = int(h.attrs["nunits_written"])
            return {k: h[f"units/{k}"][:n].astype(float) for k in keys if f"units/{k}" in h}
    rr = [one(f) for f in files]
    return {k: np.concatenate([r[k] for r in rr if k in r]) for k in keys}


def run_bias(j):
    res = {}
    for fam in ("oo_vac", "oo_med"):
        files = _files(fam)
        with ProcessPoolExecutor(max_workers=j) as ex:
            rr = list(ex.map(bias_file, files))
        ev = {k: np.concatenate([r[k] for r in rr]) for k in rr[0]}
        cache = np.load(ROOT / "results" / f"oo_analysis_rhic_{fam}.npz")
        if len(cache["b"]) != len(ev["glauber_b"]) or not np.allclose(cache["b"], ev["glauber_b"]):
            raise SystemExit(f"{fam}: event order differs from the cache; rerun oo_analysis_rhic.py")
        for k in ("n_mbd", "n_epd", "n_mid10", "w", "win"):
            ev[k] = cache[k]
        for k, v in ev.items():
            res[f"{fam}/{k}"] = v
        print(f"{fam}: {len(files)} files, {len(ev['pthat'])} events")
    mb = B.load(A.MB_SAMPLE)
    for k in ("glauber_b", "glauber_ncoll", "n_mbd", "n_epd", "n_mid10"):
        res[f"mb/{k}"] = np.asarray(mb[k], float)
    try:
        mu = mb_units(j)
        if len(mu["glauber_b"]) == len(res["mb/glauber_b"]) and np.allclose(mu["glauber_b"], res["mb/glauber_b"]):
            for k in ("strings_eloc_mbd_pos", "strings_eloc_mbd_neg", "strings_eloc_epd"):
                res[f"mb/{k}"] = mu[k]
        else:
            print("phys_mb units: order differs from oo_bias.load; string energies of mb not used")
    except (OSError, KeyError, ValueError) as e:
        print("phys_mb units not read:", e)
    np.savez_compressed(ROOT / "results" / "quench_diag_bias.npz", **res)
    print("-> results/quench_diag_bias.npz")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", choices=("eloss", "bias", "both"))
    ap.add_argument("--out", default="/data/OO-RHIC_phys")
    ap.add_argument("-j", type=int, default=8)
    a = ap.parse_args(argv)
    B.OUT = Path(a.out)
    (ROOT / "results").mkdir(exist_ok=True)
    if a.part in ("eloss", "both"):
        run_eloss(a.j)
    if a.part in ("bias", "both"):
        run_bias(a.j)
    return 0


if __name__ == "__main__":
    sys.exit(main())
