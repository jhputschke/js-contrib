#!/usr/bin/env python3
"""
analysis/OO-RHIC/truncation_check.py -- how much of each low-pT data bin comes from below the
tune's lowest pTHat window (5 GeV), from the run 'trunc' (windows 4-5, 5-7, ..., 14-18 GeV at
the centre of the parameter box: alpha_s 0.155, Cv 0.33, Q0 1.5, iMATTER floor 2.6 GeV).

    python make_design.py --add trunc 0.155 0.33 1.5 --jobs 10 --windows 4-5,5-7,7-9,9-11,11-14,14-18
    python run_runs.py trunc -j 10 && python observables.py trunc
    python truncation_check.py          # -> results/truncation.json and a table

Two parts.  (1) The share of the 4-5 GeV window per data bin.  It is zero above 4.5 GeV
(pi0) and in all jet bins, but 30k events bound it only loosely there.  (2) The kinematic
reach: the leading pi0 and the leading jet (R = 0.5) of every event relative to the event's
own pTHat.  Result of the run of 2026-11-22 (30k events): pi0 / pTHat never above 1.14, so
pTHat < 5 GeV reaches pi0 only below 5.7 GeV; jet / pTHat above 1.2 in 0.52 % and above 1.4 in
0.047 % of the events, maximum 1.59: pTHat < 5 GeV reaches 6.9-8.2 GeV jets at the few-%
level (the 4-5 GeV window's cross section is 2.4 times the 5-7 GeV one), 8.2 GeV and up
not.  Hence config.FIT: pi0 from 6 GeV, jets from 8.2 GeV.
"""

from __future__ import annotations

import json
import sys

import numpy as np

import config as C
import observables as O


def reach():
    """Leading pi0 and leading jet pT over the event's pTHat, all events of 'trunc'."""
    import glob
    import awkward as ak
    import fastjet
    import hdf5plugin  # noqa: F401
    import h5py
    rp, rj = [], []
    for path in sorted(glob.glob(str(C.point_dir("trunc") / "*_jet_frag.h5"))):
        fd = O.read_file(path)
        with h5py.File(path) as f:
            pth = f["units/pthat"][:len(fd["win"])]
        pt, _, _, _ = O._kin(fd["p"])
        ev = np.repeat(np.arange(len(fd["counts"])), fd["counts"])
        m = fd["pid"] == 111
        lead = np.zeros(len(pth))
        np.maximum.at(lead, ev[m], pt[m])
        rp.append(lead / pth)
        vis = ~np.isin(np.abs(fd["pid"]), C.INVISIBLE) & (pt > C.CONST_PTMIN)
        cc = np.bincount(ev[vis], minlength=len(pth))
        P = fd["p"][vis]
        parts = ak.zip({k: ak.unflatten(P[:, i], cc)
                        for k, i in (("px", 1), ("py", 2), ("pz", 3), ("E", 0))})
        js = fastjet.ClusterSequence(parts, fastjet.JetDefinition(
            fastjet.antikt_algorithm, C.JET_R)).inclusive_jets(min_pt=1.0)
        rj.append(ak.to_numpy(ak.fill_none(ak.max(np.hypot(js.px, js.py), axis=1), 0.0)) / pth)
    res = {}
    for name, r in (("pi0", np.concatenate(rp)), ("jet", np.concatenate(rj))):
        res[name] = {"events": int(len(r)), "frac_gt_1.0": float(np.mean(r > 1.0)),
                     "frac_gt_1.2": float(np.mean(r > 1.2)),
                     "frac_gt_1.4": float(np.mean(r > 1.4)), "max": float(r.max())}
        print(f"leading {name} / pTHat: >1.0 {100 * res[name]['frac_gt_1.0']:.2f} %, "
              f">1.2 {100 * res[name]['frac_gt_1.2']:.3f} %, >1.4 "
              f"{100 * res[name]['frac_gt_1.4']:.3f} %, max {res[name]['max']:.2f}")
    return res


def main():
    data = O.load_data()
    o = np.load(C.obs_path("trunc"))
    out = {}
    print("bin [GeV]        share of pTHat 4-5 GeV   (stat. error)")
    for name, pmax in (("phenix_pi0", 12.0), ("star_pi0", 12.0), ("star_jets_eta0", 20.0),
                       ("star_jets_eta1", 20.0)):
        tot, etot = O.combine(o, name, data)
        low, elow = O.combine(o, name, data, windows=[0])
        e = np.asarray(data[name]["edges"])
        rows = []
        print(name)
        for (lo, hi), t, l, el in zip(e, tot, low, elow):
            if lo < 3.0 or hi > pmax or t <= 0:
                continue
            share, err = l / t, el / t
            rows.append([lo, hi, share, err])
            print(f"  {lo:5.1f}-{hi:5.1f}    {100 * share:6.2f} %   ({100 * err:.2f} %)")
        out[name] = rows
    out["reach"] = reach()
    (C.HERE / "results").mkdir(exist_ok=True)
    (C.HERE / "results" / "truncation.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
