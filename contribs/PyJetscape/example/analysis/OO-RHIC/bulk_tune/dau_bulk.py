#!/usr/bin/env python3
"""
analysis/OO-RHIC/bulk_tune/dau_bulk.py -- d+Au 200 GeV minimum-bias bulk against PHENIX and BRAHMS: the check
that a bulk set tuned on p+p and O+O does not break the calibration of the JETSCAPE MAP
(2601.17234), which used d+Au (PHENIX dN/deta, 0-20 %) and Au+Au.

    python dau_bulk.py fetch                          # HEPData tables -> refs/dau_200/
    python dau_bulk.py analyse [--out ROOT] [SAMPLES]  # default: every dau_* sample -> results/dau_bulk.json

Model events (oo_bias.py, samples dau_<set>): the O+O XML's chain with projectile d (Hulthen) and
target Au (Woods-Saxon), b < 12 fm, the minimum-bias stand-in (no subtraction, hard-pair veto
off), MUSIC + iSS with one sample per event, charged hadrons with |eta| < 5.2; only the bulk (iSS)
hadrons are counted, as for O+O `mb` and pp_bulk.py.  3dMCGlauber's target moves toward -z, so
the Au-going side is eta < 0, as in PHENIX's and BRAHMS's tables.

Data (HEPData):
  PHENIX, arXiv:1807.11928, HEPData ins1684475 Fig. 2b: charged dN/deta for |eta| < 2.6 in 0-5,
      5-10, 10-20, 20-40, 40-60, 60-88 %; centrality from the charge in the Au-going BBC
      (-3.9 < eta < -3.1), as percentiles of the inelastic cross section (the classes end at 88 %,
      the trigger's share); the table gives systematic errors only
  BRAHMS, HEPData ins643085 Table 2 ("Centrality dependence of charged-particle pseudorapidity
      distributions from d+Au collisions at 200 GeV"): charged dN/deta for -2.0 < eta < 4.2 (the
      Au side negative, as its larger yields show), minimum bias and 0-30 / 30-60 / 60-80 %; only the
      minimum-bias series is compared (no estimator emulated): the d-going side reaches the
      deuteron's remnants, the region where the model's p+p is above PHOBOS
  STAR, nucl-ex/0408016, HEPData ins656934 Table 2: charged dN/deta for |eta| < 1 (TPC, pT
      extrapolated to 0) in minimum bias and 0-20 / 20-40 / 40-100 %; centrality from the charged
      multiplicity in the Au-going FTPC (-3.8 < eta < -2.8), as percentiles of the total d+Au cross
      section (the minimum-bias trigger, a ZDC-Au neutron, sees 95 +- 3 %); total errors
  STAR FTPC, refs/dau_200/star_dau_ftpc.json (extract_star_ftpc_dau.py, read from a ROOT figure
      of the STAR d+Au FTPC analysis, 2004; the figure is not in the repository): the same classes at 3.0 <= |eta| <= 3.5 on both sides, i.e. in
      the Au fragmentation region and on the deuteron side where its remnants land.  For these points
      the classes come from the charged multiplicity at mid-rapidity, |eta| < 0.5 (as in that analysis;
      no autocorrelation with the FTPCs), not from the FTPC-Au.  The figure's
      40-100 % and minimum-bias points are 13 % and 9 % above the published |eta| < 1 values, so every
      class is scaled by published / figure at |eta| < 1 (0-20 and 20-40 %: 1.007 / 1.004).  Compared as
      the mean over the 10 points per side; errors as drawn (bars hidden by the marker: <= 0.40)
Model classes: percentiles of the model's charged multiplicity in the Au-going BBC acceptance
among all its events (each has >= 1 NN collision; no diffraction), random tie-break; PHENIX's
classes are percentiles of the full inelastic cross section, so no trigger normalisation enters.
--class-frac F tests that: the model's events as a fraction F of the cross section the classes
refer to (F < 1: classes among the model's highest-F events, as STAR O+O; F > 1: the model
misses part of it) -> results/dau_bulk_f<F>.json.  2026-11-24: F < 1 steepens the peripheral
rise of g_v045_n215; F ~ 1.05-1.10 flattens mid-rapidity and the d side, the central
Au-side excess (~1.15-1.2 vs PHENIX) stays for every F.
Model results, not vetted physics results; statistical errors only.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import _paths  # noqa: E402,F401  (analysis/OO-RHIC and its test directories on sys.path)
ROOT = _paths.ROOT                           # analysis/OO-RHIC: results/, refs/, data/
import oo_bias as B  # noqa: E402

REFS = ROOT / "refs" / "dau_200"
TABLES = {"phenix": ("ins1684475", "Figure 2b"), "brahms": ("ins643085", "Table 2"),
          "star": ("ins656934", "Table 2")}
OUT_JSON = ROOT / "results" / "dau_bulk.json"
ETA_EDGES = np.round(np.arange(-5.0, 5.0001, 0.1), 3)
BBC_S = (-3.9, -3.1)                    # PHENIX's Au-going BBC
PHENIX_CLASSES = [(0, 5), (5, 10), (10, 20), (20, 40), (40, 60), (60, 88)]
FTPC_AU = (-3.8, -2.8)                  # STAR's Au-going FTPC (centrality of the |eta| < 1 points)
MID_CLS = 0.5                           # |eta| < 0.5: centrality of STAR's FTPC dN/deta points
STAR_CLASSES = [(0, 20), (20, 40), (40, 100)]


# ── data ──────────────────────────────────────────────────────────────────────────────────
def _path(key):
    rec, t = TABLES[key]
    return REFS / f"{key}_{rec}_{t.replace(' ', '')}.yaml"


def fetch():
    REFS.mkdir(parents=True, exist_ok=True)
    for key, (rec, t) in TABLES.items():
        url = f"https://www.hepdata.net/download/table/{rec}/{t.replace(' ', '%20')}/yaml"
        subprocess.run(["curl", "-sfL", url, "-o", str(_path(key))], check=True)
        print("wrote", _path(key))
    return 0


def data():
    import yaml
    out = {}
    d = yaml.safe_load(_path("phenix").read_text())
    iv = d["independent_variables"][0]["values"]
    lo, hi = np.array([v["low"] for v in iv]), np.array([v["high"] for v in iv])
    ph = {"eta_lo": lo.tolist(), "eta_hi": hi.tolist(), "classes": {}}
    for dv in d["dependent_variables"]:
        c = dv["qualifiers"][0]["value"]
        ph["classes"][c] = {"y": [float(v["value"]) for v in dv["values"]],
                            "sys": [float(sum(e.get("symerror", 0) ** 2 for e in v.get("errors", [])) ** 0.5)
                                    for v in dv["values"]]}
    out["phenix"] = ph
    d = yaml.safe_load(_path("brahms").read_text())
    eta = [float(v["value"]) for v in d["independent_variables"][0]["values"]]
    br = {"eta": eta, "classes": {}}
    for dv in d["dependent_variables"]:
        c = str(dv["qualifiers"][0]["value"])
        br["classes"][c] = {"y": [float(v["value"]) for v in dv["values"]],
                            "err": [float(sum(e.get("symerror", 0) ** 2 for e in v.get("errors", [])) ** 0.5)
                                    for v in dv["values"]]}
    out["brahms"] = br
    d = yaml.safe_load(_path("star").read_text())
    iv = d["independent_variables"][0]["values"]
    st = {"eta_lo": [float(v["low"]) for v in iv], "eta_hi": [float(v["high"]) for v in iv], "classes": {}}
    for dv in d["dependent_variables"]:
        q = dv["qualifiers"][0]["value"]
        lab = "minbias" if "minimum bias" in q else q.split()[-1]
        st["classes"][lab] = {"y": [float(v["value"]) for v in dv["values"]],
                              "err": [float(sum(e.get("symerror", 0) ** 2 for e in v.get("errors", [])) ** 0.5)
                                      for v in dv["values"]]}
    out["star"] = st
    fp = REFS / "star_dau_ftpc.json"
    if fp.exists():
        out["star_ftpc"] = json.loads(fp.read_text())
    return out


# ── model ─────────────────────────────────────────────────────────────────────────────────
def load(name):
    """Per event: eta histogram (0.1 bins) of the bulk charged particles, Au-going BBC count."""
    import hdf5plugin  # noqa: F401
    import h5py
    hs, bbc, ftpc, mid = [], [], [], []
    for jf in sorted(glob.glob(str(B.OUT / name / "*_hadrons_jet_frag.h5"))):
        if not os.path.exists(jf.replace("_hadrons_jet_frag.h5", ".json")):
            continue                                   # still running
        bf = jf.replace("_hadrons_jet_frag.h5", "_hadrons_bulk_bg.h5")
        if not os.path.exists(bf):
            continue
        with h5py.File(jf) as h:
            n = int(h.attrs["nunits_written"])
        with h5py.File(bf) as h:
            p, cnt = B._units(h, n)
        E, px, py, pz = p.T
        pa = np.sqrt(px * px + py * py + pz * pz)
        eta = 0.5 * np.log(np.clip(pa + pz, 1e-12, None) / np.clip(pa - pz, 1e-12, None))
        ev = np.repeat(np.arange(n), cnt)
        hh = np.zeros((n, len(ETA_EDGES) - 1))
        ib = np.digitize(eta, ETA_EDGES) - 1
        ok = (ib >= 0) & (ib < len(ETA_EDGES) - 1)
        np.add.at(hh, (ev[ok], ib[ok]), 1)
        hs.append(hh)
        bbc.append(np.bincount(ev[(eta > BBC_S[0]) & (eta < BBC_S[1])], minlength=n))
        ftpc.append(np.bincount(ev[(eta > FTPC_AU[0]) & (eta < FTPC_AU[1])], minlength=n))
        mid.append(np.bincount(ev[np.abs(eta) < MID_CLS], minlength=n))
    if not hs:
        return None, None, None, None
    return np.concatenate(hs), np.concatenate(bbc), np.concatenate(ftpc), np.concatenate(mid)


CLASS_FRAC = 1.0                        # --class-frac


def classes(x, cls, seed=1):
    """Class index per event from percentiles of x (highest first), random tie-break; the
    model's events are a fraction CLASS_FRAC of the cross section the classes refer to."""
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(x)), -x))          # descending x, ties at random
    rank = np.empty(len(x)); rank[order] = (np.arange(len(x)) + 0.5) / len(x) * 100 / CLASS_FRAC
    c = np.full(len(x), -1)
    for i, (a, b) in enumerate(cls):
        c[(rank >= a) & (rank < b)] = i
    return c


def mean_in(h, mask, lo, hi):
    """Mean dN/deta per event over [lo, hi) and its standard error (from the 0.1 bins)."""
    a, b = np.searchsorted(ETA_EDGES, lo + 1e-6) - 1, np.searchsorted(ETA_EDGES, hi - 1e-6)
    v = h[mask][:, a:b].sum(1) / (ETA_EDGES[b] - ETA_EDGES[a])
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))


def analyse(names):
    D = data()
    ph, br, st = D["phenix"], D["brahms"], D["star"]
    res = {"note": "model, not vetted; see the module docstring", "class_frac": CLASS_FRAC, "samples": {}}
    out_json = OUT_JSON if CLASS_FRAC == 1.0 else OUT_JSON.with_name(f"dau_bulk_f{CLASS_FRAC:g}.json")
    for name in names:
        h, bbc, ftpc, nmid = load(name)
        if h is None:
            print(f"{name}: no finished jobs")
            continue
        o = {"n_events": int(len(h)), "classes": {}}
        c = classes(bbc.astype(float), PHENIX_CLASSES)
        for i, (a, b) in enumerate(PHENIX_CLASSES):
            lab = f"{a}-{b}%"
            m = c == i
            d = ph["classes"][lab]
            mod, err = zip(*(mean_in(h, m, lo, hi) for lo, hi in zip(ph["eta_lo"], ph["eta_hi"])))
            y, s = np.array(d["y"]), np.array(d["sys"])
            r = np.array(mod) / y
            eta = 0.5 * (np.array(ph["eta_lo"]) + np.array(ph["eta_hi"]))
            sides = {"au": eta < -1, "mid": np.abs(eta) < 0.5, "d": eta > 1}
            o["classes"][lab] = {"n": int(m.sum()), "eta": eta.tolist(), "model": list(mod), "model_err": list(err),
                                 "data": y.tolist(), "sys": s.tolist(), "ratio": r.tolist(),
                                 **{f"ratio_{k}": float(r[v].mean()) for k, v in sides.items()},
                                 **{f"data_sys_rel_{k}": float((s[v] / y[v]).mean()) for k, v in sides.items()}}
        # 0-20 % as PHENIX's three classes weighted by their widths (the calibration's class)
        w = {"0-5%": 5, "5-10%": 5, "10-20%": 10}
        for k in ("au", "mid", "d"):
            o[f"ratio_0_20_{k}"] = sum(w[l] * o["classes"][l][f"ratio_{k}"] for l in w) / 20
        # BRAHMS minimum bias, d side to eta 4.2: all model events
        allm = np.ones(len(h), bool)
        be = np.array(br["eta"])
        mb = br["classes"]["minbias"]
        mod = [mean_in(h, allm, e - 0.1, e + 0.1) for e in be]
        o["brahms_minbias"] = {"eta": be.tolist(), "model": [m_ for m_, _ in mod], "model_err": [e_ for _, e_ in mod],
                               "data": mb["y"], "err": mb["err"],
                               "ratio": (np.array([m_ for m_, _ in mod]) / np.array(mb["y"])).tolist()}
        # STAR, |eta| < 1, classes from the Au-going FTPC, and minimum bias (all model events)
        cs = classes(ftpc.astype(float), STAR_CLASSES, seed=2)
        cmid = classes(nmid.astype(float), STAR_CLASSES, seed=3)      # classes of the FTPC points
        o["star"] = {}
        for lab, m in [(f"{a}-{b}%", cs == i) for i, (a, b) in enumerate(STAR_CLASSES)] + [("minbias", np.ones(len(h), bool))]:
            d = st["classes"][lab]
            mod, err = zip(*(mean_in(h, m, lo, hi) for lo, hi in zip(st["eta_lo"], st["eta_hi"])))
            y, e = np.array(d["y"]), np.array(d["err"])
            eta = 0.5 * (np.array(st["eta_lo"]) + np.array(st["eta_hi"]))
            r = np.array(mod) / y
            o["star"][lab] = {"n": int(m.sum()), "eta": eta.tolist(), "model": list(mod), "model_err": list(err),
                              "data": y.tolist(), "err": e.tolist(), "ratio": r.tolist(),
                              "ratio_au": float(r[eta < -0.5].mean()), "ratio_mid": float(r[np.abs(eta) < 0.5].mean()),
                              "ratio_d": float(r[eta > 0.5].mean()), "err_rel": float((e / y).mean()),
                              "asym_model": float(np.mean(np.array(mod)[eta < -0.5]) / np.mean(np.array(mod)[eta > 0.5])),
                              "asym_data": float(y[eta < -0.5].mean() / y[eta > 0.5].mean())}
            if "star_ftpc" in D:                              # the FTPC points, both sides
                F = D["star_ftpc"]; nrm = F["norm_to_published"][lab]
                mf = np.ones(len(h), bool) if lab == "minbias" else cmid == [f"{a}-{b}%" for a, b in STAR_CLASSES].index(lab)
                for side, sel in (("ftpc_au", lambda e: e < -2), ("ftpc_d", lambda e: e > 2)):
                    pts = [q for q in F["classes"][lab] if sel(q["eta"])]
                    fe = np.array([q["eta"] for q in pts]); fy = nrm * np.array([q["y"] for q in pts])
                    fr = nrm * np.array([q["err"] if q["err"] is not None else q["err_max"] for q in pts])
                    lo_, hi_ = (-3.5, -3.0) if side == "ftpc_au" else (3.0, 3.5)
                    mm, me = mean_in(h, mf, lo_, hi_)
                    o["star"][lab][side] = {"eta": fe.tolist(), "data": fy.tolist(), "err": fr.tolist(),
                                            "model_mean": mm, "model_err": me, "data_mean": float(fy.mean()),
                                            "data_err_rel": float((fr / fy).mean()), "ratio": mm / float(fy.mean()),
                                            "model": [mean_in(h, mf, e - 0.05, e + 0.05)[0] for e in fe],
                                            "class_estimator": "|eta| < 0.5"}
        res["samples"][name] = o
        print(f"{name}: {len(h)} events")
        for lab, d in o["classes"].items():
            print(f"   {lab:7s} n {d['n']:5d}  model/PHENIX Au side {d['ratio_au']:.2f}, |eta|<0.5 {d['ratio_mid']:.2f}, "
                  f"d side {d['ratio_d']:.2f}   (data sys {100 * d['data_sys_rel_mid']:.0f} %)")
        print(f"   0-20 %: Au side {o['ratio_0_20_au']:.2f}, mid {o['ratio_0_20_mid']:.2f}, d side {o['ratio_0_20_d']:.2f}")
        for lab, d in o["star"].items():
            if "ftpc_au" in d:
                print(f"   STAR {lab:8s} FTPC  model/STAR Au side (-3.5..-3) {d['ftpc_au']['ratio']:.2f}, d side (3..3.5) "
                      f"{d['ftpc_d']['ratio']:.2f}  (data err {100 * d['ftpc_au']['data_err_rel']:.0f} / {100 * d['ftpc_d']['data_err_rel']:.0f} %)")
            print(f"   STAR {lab:8s} n {d['n']:5d}  model/STAR -1<eta<-0.5 {d['ratio_au']:.2f}, |eta|<0.5 {d['ratio_mid']:.2f}, "
                  f"0.5<eta<1 {d['ratio_d']:.2f}  (data err {100 * d['err_rel']:.0f} %); Au/d asymmetry model {d['asym_model']:.3f}, data {d['asym_data']:.3f}")
        rb = np.array(o["brahms_minbias"]["ratio"])
        print("   BRAHMS min. bias model/data at eta " + ", ".join(
            f"{e:.1f}: {x:.2f}" for e, x in zip(be, rb) if any(abs(e - t) < 0.15 for t in (-2.0, 0.0, 2.0, 3.0, 3.6, 4.2))))
    out_json.parent.mkdir(exist_ok=True)
    if out_json.exists():                                  # keep samples analysed earlier
        old = json.loads(out_json.read_text())
        old.get("samples", {}).update(res["samples"])
        res["samples"] = old.get("samples", res["samples"])
    out_json.write_text(json.dumps(res, indent=1))
    print("wrote", out_json)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["fetch", "analyse"])
    p.add_argument("samples", nargs="*")
    p.add_argument("--out", default=None, help="output root of oo_bias.py (default its OUT)")
    p.add_argument("--class-frac", type=float, default=1.0, dest="class_frac",
                   help="the model's events as this fraction of the classes' cross section (default 1)")
    a = p.parse_intermixed_args(argv)
    global CLASS_FRAC
    CLASS_FRAC = a.class_frac
    if a.out:
        B.OUT = Path(a.out)
    if a.cmd == "fetch":
        return fetch()
    names = a.samples or sorted(Path(d).name for d in glob.glob(str(B.OUT / "dau_*")))
    return analyse(names)


if __name__ == "__main__":
    sys.exit(main())
