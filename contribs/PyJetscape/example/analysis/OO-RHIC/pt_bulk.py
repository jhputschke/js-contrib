#!/usr/bin/env python3
"""
analysis/OO-RHIC/pt_bulk.py -- identified pT spectra and <pT> of the bulk (iSS) hadrons of the
oo_bias.py minimum-bias samples (p+p, d+Au, O+O) against STAR at 200 GeV: a check of the hydro side
of the chain (MUSIC, freeze-out), which the bulk tune (yields and dN/deta) did not touch.

    python pt_bulk.py fetch                          # HEPData tables -> refs/pt_200/
    python pt_bulk.py analyse [--out ROOT] [SETS]     # default sets g_v045_n215 tau0424 map
                                                      # -> results/pt_bulk.json

Data
  STAR TOF, nucl-ex/0309012 (PLB 616 (2005) 8), HEPData ins628232 Tables 2a-c: pi+-, K+-, p, pbar
      invariant yields 1/N d2N/(2 pi pT dpT dy), -0.5 < y < 0 (Au-going side negative), pT 0.3-3 GeV
      (kaons to 1.7); NSD p+p (BBC coincidence, 3.3 < |eta| < 5) and d+Au minimum bias and 0-20 /
      20-40 / 40-100 % from the Au-going FTPC multiplicity (-3.8 < eta < -2.8).  Pions corrected for
      weak-decay feed-down, p and pbar inclusive (hyperon decays ~20 % of the yield).  Errors stat and
      syst (bin-to-bin ~8 %); normalisation 14 % (p+p), 10 % (d+Au).
  STAR dE/dx, arXiv:0808.2041 (PRC 79 034909), HEPData ins793126 Fig. 18a-c: d+Au the same classes,
      |y| < 0.1, pT 0.2-0.7 (pi), 0.2-0.75 (K), 0.35-1.1 GeV (p); stat + point-to-point syst.  Its
      Tables VII (<pT>, extrapolated) and VIII (dN/dy) for p+p, d+Au and Au+Au 200 GeV are not on
      HEPData: refs/pt_200/star_0808_2041_tables.json (transcribed).
  STAR p+p multiplicity classes, nucl-ex/0606028 (PRD 74 032006), HEPData ins719969 Fig. 1 (left) and
      Fig. 8: charged spectra 1/n_ch 1/pT dn/dpT (unit integral after extrapolation) and <pT> by direct
      integration in classes of the observed multiplicity n_hat (uncorrected tracks with pT > 0.2 GeV,
      |eta| < 0.5; corrected n'_ch = 1.35 n_hat, i.e. ~74 % efficiency): n_hat = 1..8, 9+10, 11+12.

Model, per sample (charged hadrons of iSS, |eta| < 5.2, one sample per event; no weak decays, so
  Sigma+-, Xi, Omega are stable and the protons exclude hyperon feed-down, unlike STAR's):
  p+p: STAR's NSD trigger as >= 1 charged particle in 3.3 < eta < 5 and in -5 < eta < -3.3.
  d+Au: classes from percentiles of the charged multiplicity in -3.8 < eta < -2.8 among all model
      events (as dau_bulk.py; the Au nucleus goes to -z, so the Au-going side is eta < 0 as in STAR).
  p+p n_hat: the charged count with pT > 0.2 GeV, |eta| < 0.5, thinned binomially with 1/1.35.
  O+O (no data): STAR's flow-paper classes, N_ch with 0.2 < pT < 2 GeV, |eta| < 1.5, as fractions
      of the model's highest f (results/centrality_norm.json flat_f, default 0.86).
  Only the bulk hadrons: the minimum-bias stand-in's one hard scattering (9-11 GeV) is left out, and
  real minimum bias has (mini)jets the bulk does not contain.  So the model has no hard component:
  compare spectra below ~1.5-2 GeV and expect the p+p <pT>(n_ch) rise, which STAR attributes to
  minijets, to be missing.  Freeze-out at T = 0.15 GeV without a hadronic afterburner: no
  hadronic-phase radial push (expect proton <pT> low).
Model results, not vetted physics results; statistical errors only.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import urllib.parse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oo_bias as B  # noqa: E402

REFS = HERE / "refs" / "pt_200"
TABLES = {
    "tof_pi": ("ins628232", "Table 2a: Invariant yield of $\\pi^{+}$ and $\\pi^{-}$"),
    "tof_k": ("ins628232", "Table 2b: Invariant yield of $K^{+}$ and $K^{-}$"),
    "tof_p": ("ins628232", "Table 2c: Invariant yield of p and $\\bar{p}$"),
    "dedx_pi": ("ins793126", "Figure 18a - pions"),
    "dedx_k": ("ins793126", "Figure 18b - kaons"),
    "dedx_p": ("ins793126", "Figure 18c - protons"),
    "ppmult_pt": ("ins719969", "Figure 1 (left panel)"),
    "ppmult_meanpt": ("ins719969", "Figure 8"),
}
STAR_TAB = REFS / "star_0808_2041_tables.json"
OUT_JSON = HERE / "results" / "pt_bulk.json"
CN_JSON = HERE / "results" / "centrality_norm.json"

SPECIES = ["pi+", "pi-", "K+", "K-", "p", "pbar"]
PDG = {211: 0, -211: 1, 321: 2, -321: 3, 2212: 4, -2212: 5}
MASS = np.array([0.13957, 0.13957, 0.49368, 0.49368, 0.93827, 0.93827])
BBC = (3.3, 5.0)
FTPC_AU = (-3.8, -2.8)
DAU_CLASSES = {"0-20%": (0, 20), "20-40%": (20, 40), "40-100%": (40, 100)}
OO_CLASSES = [(0, 10), (10, 20), (20, 40), (40, 60), (60, 80)]
NHAT_EFF = 1 / 1.35
NHAT_CLASSES = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 10), (11, 12)]
TOF_Y = (-0.5, 0.0)
DEDX_Y = (-0.1, 0.1)
MEAN_Y = (-0.5, 0.5)                    # model <pT> and dN/dy (STAR's |y| < 0.1; flat at mid-rapidity)


# ── data ──────────────────────────────────────────────────────────────────────────────────
def _path(key):
    return REFS / f"{key}_{TABLES[key][0]}.yaml"


def fetch():
    REFS.mkdir(parents=True, exist_ok=True)
    for key, (rec, t) in TABLES.items():
        url = f"https://www.hepdata.net/download/table/{rec}/{urllib.parse.quote(t)}/yaml"
        subprocess.run(["curl", "-sfL", "--retry", "5", "--retry-all-errors", url, "-o", str(_path(key))], check=True)
        print("wrote", _path(key))
    return 0


def _species(label):
    s = label.replace("$", "").replace("\\", "").replace("{", "").replace("}", "").replace(" ", "")
    for key, sp in (("pi^+", "pi+"), ("pi^-", "pi-"), ("K^+", "K+"), ("K^-", "K-"),
                    ("barp", "pbar"), ("pbar", "pbar"), ("proton", "p")):
        if key in s:
            return sp
    if s.startswith("p,") or s == "p":
        return "p"
    raise ValueError(label)


def _err(v, labels=None):
    return float(sum(e.get("symerror", 0) ** 2 for e in v.get("errors", [])
                     if labels is None or e.get("label") in labels) ** 0.5)


def _edges(c):
    c = np.asarray(c, float)
    mid = 0.5 * (c[1:] + c[:-1])
    return np.concatenate([[c[0] - (mid[0] - c[0])], mid, [c[-1] + (c[-1] - mid[-1])]])


def data():
    import yaml
    out = {"tof": {}, "dedx": {}}
    for key in ("tof_pi", "tof_k", "tof_p"):
        d = yaml.safe_load(_path(key).read_text())
        pt = [float(v["value"]) for v in d["independent_variables"][0]["values"]]
        for dv in d["dependent_variables"]:
            name = dv["header"]["name"]
            sp = _species(name.split(",")[0])
            cls = name.split(",", 1)[1].replace("$", "").replace("\\%", "%").replace("\\", "").strip()
            cls = {"p+p NSD": "pp", "d+Au minbias": "minbias"}.get(cls, cls.replace("d+Au ", ""))
            ok = [(x, v) for x, v in zip(pt, dv["values"]) if v["value"] != "-"]
            out["tof"].setdefault(cls, {})[sp] = {
                "pt": [x for x, _ in ok], "y": [float(v["value"]) for _, v in ok],
                "stat": [_err(v, ("stat",)) for _, v in ok],
                "sys": [_err(v, ("syst",)) for _, v in ok]}
    for key in ("dedx_pi", "dedx_k", "dedx_p"):
        d = yaml.safe_load(_path(key).read_text())
        pt = [float(v["value"]) for v in d["independent_variables"][0]["values"]]
        for dv in d["dependent_variables"]:
            q = dv["qualifiers"][0]["value"]
            sp = _species(q.split("(")[0])
            cls = q.split("(")[1].rstrip(")").strip()
            cls = "minbias" if cls == "min-bias" else cls
            ok = [(x, v) for x, v in zip(pt, dv["values"]) if v["value"] != "-"]
            out["dedx"].setdefault(cls, {})[sp] = {"pt": [x for x, _ in ok], "y": [float(v["value"]) for _, v in ok],
                                                   "err": [_err(v) for _, v in ok]}
    d = yaml.safe_load(_path("ppmult_pt").read_text())
    iv = d["independent_variables"][0]["values"]
    lo, hi = [float(v["low"]) for v in iv], [float(v["high"]) for v in iv]
    pm = {"pt_lo": lo, "pt_hi": hi, "classes": {}}
    for dv, (a, b) in zip(d["dependent_variables"], NHAT_CLASSES):
        pm["classes"][f"{a}-{b}"] = {"y": [float(v["value"]) if v["value"] not in ("", "-") else np.nan for v in dv["values"]],
                                     "err": [_err(v) if v["value"] not in ("", "-") else np.nan for v in dv["values"]]}
    d = yaml.safe_load(_path("ppmult_meanpt").read_text())
    nh = [float(v["value"]) for v in d["independent_variables"][0]["values"]]
    pm["meanpt"] = {"nhat": nh}
    for dv in d["dependent_variables"]:
        lab = dv["header"]["name"].split(",", 1)[1].strip()
        pm["meanpt"][lab] = {"y": [float(v["value"]) if v["value"] not in ("", "-") else np.nan for v in dv["values"]],
                             "err": [_err(v) if v["value"] not in ("", "-") else np.nan for v in dv["values"]]}
    out["ppmult"] = pm
    out["star_tab"] = json.loads(STAR_TAB.read_text())
    return out


# ── model ─────────────────────────────────────────────────────────────────────────────────
def load(name):
    """Particle arrays (event, pT, y, eta, species index or -1) and per-event estimator counts."""
    import hdf5plugin  # noqa: F401
    import h5py
    ev_total, per_job = B.SAMPLES[name][0], B.SAMPLES[name][1]
    done = len(glob.glob(str(B.OUT / name / "*_hadrons_jet_frag.h5").replace("_hadrons_jet_frag.h5", ".json")))
    if done < ev_total // per_job:
        print(f"{name}: {done} of {ev_total // per_job} jobs finished, skipped")
        return None
    parts, ev_off = [], 0
    est = {k: [] for k in ("bbc_pos", "bbc_neg", "ftpc_au", "flow", "mid05")}
    for jf in sorted(glob.glob(str(B.OUT / name / "*_hadrons_jet_frag.h5"))):
        if not os.path.exists(jf.replace("_hadrons_jet_frag.h5", ".json")):
            continue                                   # still running
        bf = jf.replace("_hadrons_jet_frag.h5", "_hadrons_bulk_bg.h5")
        if not os.path.exists(bf):
            continue
        with h5py.File(jf) as h:
            n = int(h.attrs["nunits_written"])
        with h5py.File(bf) as h:
            p, cnt, pid = B._units(h, n, with_pid=True)
        E, px, py, pz = p.T
        pt = np.hypot(px, py)
        pa = np.sqrt(px * px + py * py + pz * pz)
        eta = 0.5 * np.log(np.clip(pa + pz, 1e-12, None) / np.clip(pa - pz, 1e-12, None))
        y = 0.5 * np.log(np.clip(E + pz, 1e-12, None) / np.clip(E - pz, 1e-12, None))
        sp = np.full(len(pid), -1)
        for code, i in PDG.items():
            sp[pid == code] = i
        ev = np.repeat(np.arange(n), cnt)
        parts.append((ev + ev_off, pt, y, eta, sp))
        for k, m in (("bbc_pos", (eta > BBC[0]) & (eta < BBC[1])), ("bbc_neg", (-eta > BBC[0]) & (-eta < BBC[1])),
                     ("ftpc_au", (eta > FTPC_AU[0]) & (eta < FTPC_AU[1])),
                     ("flow", (np.abs(eta) < 1.5) & (pt > 0.2) & (pt < 2.0)),
                     ("mid05", (np.abs(eta) < 0.5) & (pt > 0.2))):
            est[k].append(np.bincount(ev[m], minlength=n))
        ev_off += n
    if not parts:
        return None
    P = {k: np.concatenate([x[i] for x in parts]) for i, k in enumerate(("ev", "pt", "y", "eta", "sp"))}
    P["n_events"] = ev_off
    P.update({k: np.concatenate(v) for k, v in est.items()})
    return P


def percentile_classes(x, classes, frac=1.0, seed=1):
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(x)), -x))
    rank = np.empty(len(x)); rank[order] = (np.arange(len(x)) + 0.5) / len(x) * 100 / frac
    return [(rank >= a) & (rank < b) for a, b in classes]


def spectrum(P, evmask, sp, yr, edges):
    """Invariant yield 1/N d2N/(2 pi pT dpT dy) in the given pT bins, with its statistical error."""
    nev = int(evmask.sum())
    m = evmask[P["ev"]] & (P["sp"] == sp) & (P["y"] > yr[0]) & (P["y"] < yr[1])
    w = 1.0 / (2 * np.pi * P["pt"][m])
    s, _ = np.histogram(P["pt"][m], edges, weights=w)
    s2, _ = np.histogram(P["pt"][m], edges, weights=w * w)
    norm = nev * np.diff(edges) * (yr[1] - yr[0])
    return s / norm, np.sqrt(s2) / norm


def mean_pt(P, evmask, sp, yr=MEAN_Y):
    m = evmask[P["ev"]] & (P["sp"] == sp) & (P["y"] > yr[0]) & (P["y"] < yr[1])
    v = P["pt"][m]
    nev = int(evmask.sum())
    dndy = len(v) / nev / (yr[1] - yr[0])
    return (float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))), dndy,
            float(np.sqrt(len(v)) / nev / (yr[1] - yr[0]))) if len(v) > 1 else (np.nan, np.nan, dndy, np.nan)


def system_of(name):
    return name.split("_", 1)[0]


def analyse(sets, D):
    res = {"note": "model, not vetted; see the module docstring", "species": SPECIES, "samples": {}}
    CN = json.loads(CN_JSON.read_text())["samples"] if CN_JSON.exists() else {}
    for s in sets:
        for sysn in ("pp", "dau", "oo"):
            name = f"{sysn}_{s}"
            P = load(name)
            if P is None:
                continue
            N = P["n_events"]
            o = {"system": sysn, "n_events": int(N), "classes": {}}
            if sysn == "pp":
                cls = {"pp": (P["bbc_pos"] > 0) & (P["bbc_neg"] > 0)}
            elif sysn == "dau":
                ms = percentile_classes(P["ftpc_au"].astype(float), list(DAU_CLASSES.values()))
                cls = {"minbias": np.ones(N, bool), **dict(zip(DAU_CLASSES, ms))}
            else:
                f = CN.get(name, {}).get("flat_f", 0.86)
                ms = percentile_classes(P["flow"].astype(float), OO_CLASSES, frac=f)
                cls = {"minbias": np.ones(N, bool), **{f"{a}-{b}%": m for (a, b), m in zip(OO_CLASSES, ms)}}
                o["flat_f"] = f
            for lab, em in cls.items():
                c = {"n": int(em.sum()), "meanpt": {}, "dndy": {}, "tof": {}, "dedx": {}}
                for i, sp in enumerate(SPECIES):
                    mp, mpe, dn, dne = mean_pt(P, em, i)
                    c["meanpt"][sp] = [mp, mpe]
                    c["dndy"][sp] = [dn, dne]
                    for key, yr in (("tof", TOF_Y), ("dedx", DEDX_Y)):
                        d = D[key].get(lab if sysn != "pp" else "pp", {}).get(sp) if sysn != "oo" else None
                        if d is None or (key == "dedx" and sysn == "pp"):
                            continue
                        e = _edges(d["pt"])
                        mv, me = spectrum(P, em, i, yr, e)
                        dy = np.asarray(d["y"])
                        c[key][sp] = {"pt": d["pt"], "model": mv.tolist(), "model_err": me.tolist(),
                                      "data": d["y"], "ratio": (mv / dy).tolist(),
                                      "data_err": (np.hypot(d["stat"], d["sys"]) if "stat" in d else np.asarray(d["err"])).tolist()}
                # charged multiplicity density at |eta| < 0.5 (all charged, all pT) for <pT> vs dNch/deta
                m = em[P["ev"]] & (np.abs(P["eta"]) < 0.5)
                c["dnch_deta"] = float(m.sum() / max(em.sum(), 1))
                c["dnch_dy_6"] = float(sum(c["dndy"][sp][0] for sp in SPECIES))
                o["classes"][lab] = c
            if sysn == "pp":
                # n_hat classes: uncorrected tracks with pT > 0.2, |eta| < 0.5 (binomial efficiency)
                rng = np.random.default_rng(7)
                nhat = rng.binomial(P["mid05"], NHAT_EFF)
                nsd = cls["pp"]
                pmd = D["ppmult"]
                lo, hi = np.asarray(pmd["pt_lo"]), np.asarray(pmd["pt_hi"])
                edges = np.concatenate([lo, hi[-1:]])
                o["nhat"] = {}
                for a, b in NHAT_CLASSES:
                    em = nsd & (nhat >= a) & (nhat <= b)
                    m = em[P["ev"]] & (np.abs(P["eta"]) < 0.5)
                    v = P["pt"][m]
                    h, _ = np.histogram(v, edges, weights=1.0 / v)
                    norm = len(v) * np.diff(edges)          # 1/n_ch 1/pT dn/dpT, unit integral over all pT
                    o["nhat"][f"{a}-{b}"] = {"n": int(em.sum()), "meanpt": [float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))],
                                             "frac_events": float(em.sum() / max(nsd.sum(), 1)),
                                             "spec": (h / norm).tolist()}
            res["samples"][name] = o
            print(summary(name, o, D))
    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(res))
    print("wrote", OUT_JSON)
    return res


def _tabkey(sysn, lab):
    return "pp" if sysn == "pp" else f"dAu {lab}" if sysn == "dau" else None


def summary(name, o, D):
    T = D["star_tab"]; idx = {sp: T["species"].index(sp) for sp in SPECIES}
    lines = [f"{name}: {o['n_events']} events"]
    for lab, c in o["classes"].items():
        tk = _tabkey(o["system"], lab)
        mp = " ".join(f"{sp} {c['meanpt'][sp][0]:.3f}" + (f"/{T['meanpt'][tk][idx[sp]][0]:.3f}" if tk in T["meanpt"] else "")
                      for sp in ("pi-", "K-", "pbar", "p"))
        dn = " ".join(f"{sp} {c['dndy'][sp][0]:.3f}" + (f"/{T['dndy'][tk][idx[sp]][0]:.3f}" if tk in T["dndy"] else "")
                      for sp in ("pi-", "K-", "pbar", "p"))
        lines.append(f"   {lab:8s} n {c['n']:5d}  <pT> model/STAR: {mp}")
        lines.append(f"   {'':8s}          dN/dy model/STAR: {dn};  dNch/deta(|eta|<0.5) {c['dnch_deta']:.2f}")
        for key in ("tof", "dedx"):
            if c[key]:
                r = " ".join(f"{sp} {np.nanmean(np.asarray(v['ratio'])[np.asarray(v['pt']) < 1.0]):.2f}"
                             f"/{np.nanmean(np.asarray(v['ratio'])[(np.asarray(v['pt']) >= 1.0) & (np.asarray(v['pt']) < 2.0)]) if any((np.asarray(v['pt']) >= 1.0) & (np.asarray(v['pt']) < 2.0)) else np.nan:.2f}"
                             for sp, v in c[key].items())
                lines.append(f"   {'':8s}          STAR {key} model/data pT<1 / 1-2 GeV: {r}")
    if "nhat" in o:
        di = [x for x in D["ppmult"]["meanpt"]["direct integration"]["y"] if np.isfinite(x)]
        lines.append("   p+p <pT>(n_hat) model / STAR direct integration: " + "  ".join(
            f"{k} {v['meanpt'][0]:.3f}/{di[i]:.3f}" if i < len(di) else f"{k} {v['meanpt'][0]:.3f}"
            for i, (k, v) in enumerate(o["nhat"].items())))
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("cmd", choices=["fetch", "analyse"])
    p.add_argument("sets", nargs="*", default=["g_v045_n215", "tau0424", "map"])
    p.add_argument("--out", default=None, help="output root of oo_bias.py (default its OUT)")
    a = p.parse_args(argv)
    if a.cmd == "fetch":
        return fetch()
    if a.out:
        B.OUT = Path(a.out)
    analyse(a.sets, data())
    return 0


if __name__ == "__main__":
    sys.exit(main())
