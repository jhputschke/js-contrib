#!/usr/bin/env python3
"""
analysis/OO-RHIC/pp_bulk.py -- p+p 200 GeV minimum-bias bulk against UA5 (plan A(1) of the p+p /
O+O bulk tune): the single-collision yield of each 3dMCGlauber set, from the p+p screening
samples of oo_bias.py (pp_map, pp_pp2407, pp_m3, ...).

    python pp_bulk.py fetch                         # UA5 tables from HEPData -> refs/ua5_200/
    python pp_bulk.py analyse [--out ROOT] [SAMPLES]  # -> results/pp_bulk.json

Model events (oo_bias.py): the O+O XML's chain with projectile and target p, b < 3 fm, the
minimum-bias stand-in (no subtraction, hard-pair veto off), MUSIC + iSS with one sample per
event, charged hadrons with |eta| < 5.2.  Only the bulk (iSS) hadrons are counted, as for O+O
`mb`: not the hard event's hadrons (with --no-subtraction the hard pair's strings and remnants
are in the fluid as well, so iColored's copy of them would be counted twice).  The beam remnants
are in the bulk: 3dMCGlauber turns every wounded nucleon's remnant into a remnant string
(rapidity loss x remnant_energy_loss_fraction) that MUSIC deposits, so |eta| > 3 is a model
prediction.  Not modelled: diffraction (UA5 inelastic, PHOBOS), and hydro + Cooper-Frye is least
certain at forward rapidity in p+p, where the energy density is low.  Every event has one NN collision; an event whose strings make no medium has no bulk.

Data (HEPData, p-pbar at 200 GeV; at mid-rapidity p+p and p-pbar agree within a few %):
  UA5, Z. Phys. C 33 (1986) 1, ins233599: dN/deta NSD (Table 1) and inelastic (Table 2),
      |eta| < 4.9 in bins of 0.25
  UA5, Z. Phys. C 43 (1989) 357, ins267179: NSD P(n) in |eta| < 0.5, 1.5, 3, 5 (Tables 3-6),
      full phase space (Table 1), <n> and dispersion (Table 11)
Other RHIC p+p data (refs/rhic_pp200/rhic_pp200.json, extract_rhic_pp200.py; none on HEPData):
  PHOBOS, arXiv:1011.1940 Fig. 28a: inelastic dN/deta for |eta| <= 5.3 (read from the vector figure,
      90 % C.L. systematic band); compared up to |eta| = 4.9 (the model's histogram ends at 5)
  STAR, arXiv:0808.2041 Table II: NSD dN/deta at |eta| < 0.5 = 2.98 +- 0.34 (BBC coincidence)
  PHENIX, arXiv:1509.06727 Sec. VII: minimum-bias p+p dN/deta at mid-rapidity = 2.38 +- 0.17
NSD in the model: >= 1 charged particle in each of 2 < eta < 5.2 and -5.2 < eta < -2 (UA5's
hodoscopes: 2 < |eta| < 5.6; the stored acceptance ends at 5.2); STAR-like NSD: the same in
3.3 < |eta| < 5.0 (STAR's BBC).  The model has no diffraction, so its "inelastic" is close to
non-diffractive; PHOBOS's inelastic sample includes single diffraction.

Model results, not vetted physics results; statistical errors only (events are independent).
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
sys.path.insert(0, str(HERE))
import oo_bias as B  # noqa: E402

REFS = HERE / "refs" / "ua5_200"
RHIC = HERE / "refs" / "rhic_pp200" / "rhic_pp200.json"
TABLES = {"ins233599": ["Table 1", "Table 2"],
          "ins267179": ["Table 1", "Table 3", "Table 4", "Table 5", "Table 6", "Table 11"]}
OUT_JSON = Path(os.environ.get("PP_BULK_JSON", HERE / "results" / "pp_bulk.json"))
ETA_EDGES = np.arange(-5.0, 5.0001, 0.25)
WINDOWS = (0.5, 1.5, 3.0, 5.0)
NSD_ARM = (2.0, 5.2)
STAR_BBC = (3.3, 5.0)


# ── data ──────────────────────────────────────────────────────────────────────────────────
def fetch():
    REFS.mkdir(parents=True, exist_ok=True)
    for rec, tabs in TABLES.items():
        for t in tabs:
            path = REFS / f"{rec}_{t.replace(' ', '')}.yaml"
            url = f"https://www.hepdata.net/download/table/{rec}/{t.replace(' ', '%20')}/yaml"
            subprocess.run(["curl", "-sfL", url, "-o", str(path)], check=True)
            print("wrote", path)
    return 0


def _yaml(rec, t):
    import yaml
    return yaml.safe_load((REFS / f"{rec}_{t.replace(' ', '')}.yaml").read_text())


def _series_200(doc):
    """(x, y, err) of the sqrt(s) = 200 GeV series of a HEPData table."""
    xs = [v.get("value", 0.5 * (v.get("low", 0) + v.get("high", 0)))
          for v in doc["independent_variables"][0]["values"]]
    for dv in doc["dependent_variables"]:
        if any(q["name"].startswith("SQRT(S)") and float(q["value"]) == 200.0 for q in dv["qualifiers"]):
            keep = [i for i, v in enumerate(dv["values"]) if v["value"] not in ("-", "", None)]
            y = np.array([float(dv["values"][i]["value"]) for i in keep])
            e = np.array([np.sqrt(sum(float(er.get("symerror", 0)) ** 2
                                      for er in dv["values"][i].get("errors", []))) for i in keep])
            return np.array(xs, float)[keep], y, e
    raise ValueError("no 200 GeV series")


def data():
    d = {}
    for key, t in (("dndeta_nsd", "Table 1"), ("dndeta_inel", "Table 2")):
        x, y, e = _series_200(_yaml("ins233599", t))
        d[key] = {"eta": x.tolist(), "y": y.tolist(), "err": e.tolist()}
    for w, t in zip(WINDOWS, ("Table 3", "Table 4", "Table 5", "Table 6")):
        x, y, e = _series_200(_yaml("ins267179", t))
        d[f"pn_{w:g}"] = {"n": x.tolist(), "p": y.tolist(), "err": e.tolist()}
    x, y, e = _series_200(_yaml("ins267179", "Table 1"))
    d["pn_full"] = {"n": x.tolist(), "p": y.tolist(), "err": e.tolist()}
    if RHIC.exists():
        d["rhic"] = json.loads(RHIC.read_text())
    return d


def phobos_sym(ph):
    """PHOBOS inelastic dN/deta symmetrised: the mean of the two eta halves at |eta| (they are
    separate measurements), the band mean / 1.645 as 1 sigma, and half the halves' difference."""
    eta, y, s = map(np.asarray, (ph["eta"], ph["y"], ph["sys"]))
    out = {"eta": [], "y": [], "sys": [], "half_diff": []}
    for a in sorted({round(abs(v), 1) for v in eta}):
        m = np.isclose(np.abs(eta), a, atol=0.02)
        out["eta"].append(a); out["y"].append(float(y[m].mean())); out["sys"].append(float(s[m].mean()))
        out["half_diff"].append(float(0.5 * (y[m].max() - y[m].min())))
    return {k: np.array(v) for k, v in out.items()}


# ── model ─────────────────────────────────────────────────────────────────────────────────
def load(name):
    """Per event: eta histogram of the bulk charged particles, counts in the windows, NSD
    flag, has_medium."""
    import hdf5plugin  # noqa: F401
    import h5py
    rec = {"h": [], "has_medium": [], "nsd": [], "nsd_star": [], **{f"n_{w:g}": [] for w in WINDOWS}}
    for jf in sorted(glob.glob(str(B.OUT / name / "*_hadrons_jet_frag.h5"))):
        if not os.path.exists(jf.replace("_hadrons_jet_frag.h5", ".json")):
            continue                                   # still running
        bf = jf.replace("_hadrons_jet_frag.h5", "_hadrons_bulk_bg.h5")
        with h5py.File(jf) as h:
            n = int(h.attrs["nunits_written"])
            med = h["units/has_medium"][:n]
        if not os.path.exists(bf):
            continue
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
        rec["h"].append(hh)
        rec["has_medium"].append(med)
        for w in WINDOWS:
            rec[f"n_{w:g}"].append(np.bincount(ev[np.abs(eta) < w], minlength=n))
        pos = np.bincount(ev[(eta > NSD_ARM[0]) & (eta < NSD_ARM[1])], minlength=n) > 0
        neg = np.bincount(ev[(-eta > NSD_ARM[0]) & (-eta < NSD_ARM[1])], minlength=n) > 0
        rec["nsd"].append(pos & neg)
        ps = np.bincount(ev[(eta > STAR_BBC[0]) & (eta < STAR_BBC[1])], minlength=n) > 0
        ns = np.bincount(ev[(-eta > STAR_BBC[0]) & (-eta < STAR_BBC[1])], minlength=n) > 0
        rec["nsd_star"].append(ps & ns)
    return {k: np.concatenate(v) for k, v in rec.items()} if rec["h"] else {}


def dndeta(h, mask):
    """Symmetrised dN/deta in |eta| bins of 0.25 (centres 0.125 ...) and its standard error."""
    hm = h[mask] / np.diff(ETA_EDGES)
    nb = hm.shape[1] // 2
    sym = 0.5 * (hm[:, nb:] + hm[:, :nb][:, ::-1])
    return sym.mean(0), sym.std(0, ddof=1) / np.sqrt(len(sym))


def analyse(names):
    D = data()
    ua5_eta = np.array(D["dndeta_nsd"]["eta"])
    res = {"data": D, "samples": {}, "note": "bulk (iSS) hadrons only; NSD = >= 1 charged in "
           "2 < |eta| < 5.2 on both sides; model, not vetted"}
    for name in names:
        r = load(name)
        if not r:
            print(f"{name}: no finished jobs")
            continue
        n = len(r["nsd"])
        out = {"n_events": n, "frac_has_medium": float(r["has_medium"].mean()),
               "frac_nsd": float(r["nsd"].mean())}
        for sel, m in (("inel", np.ones(n, bool)), ("nsd", r["nsd"])):
            y, e = dndeta(r["h"], m)
            cen = 0.5 * (ETA_EDGES[:-1] + ETA_EDGES[1:])
            cen = cen[len(cen) // 2:]
            o = {"eta": cen.tolist(), "y": y.tolist(), "err": e.tolist(), "n": int(m.sum())}
            ref = D["dndeta_nsd" if sel == "nsd" else "dndeta_inel"]
            ry, re_ = np.array(ref["y"]), np.array(ref["err"])
            my = np.interp(ref["eta"], cen, y)
            o["ratio_to_ua5"] = (my / ry).tolist()
            o["mid05"] = float(np.mean(y[:2]))
            o["mid05_err"] = float(np.sqrt(np.sum(e[:2] ** 2)) / 2)
            o["ua5_mid05"] = float(np.mean(ry[:2]))
            out[f"dndeta_{sel}"] = o
            if sel == "nsd":
                for w in WINDOWS:
                    k = r[f"n_{w:g}"][m]
                    pn = D[f"pn_{w:g}"]
                    xn, pd = np.array(pn["n"]), np.array(pn["p"])
                    pm = np.array([np.mean(k == int(round(v))) for v in xn])
                    out[f"pn_{w:g}"] = {"mean": float(k.mean()), "mean_err": float(k.std(ddof=1) / np.sqrt(len(k))),
                                        "disp": float(k.std(ddof=1)),
                                        "ua5_mean": float(np.sum(xn * pd) / np.sum(pd)),
                                        "ua5_disp": float(np.sqrt(np.sum(pd * xn ** 2) / np.sum(pd)
                                                                  - (np.sum(xn * pd) / np.sum(pd)) ** 2)),
                                        "n": xn.tolist(), "p_model": pm.tolist()}
        if "rhic" in D:
            out["rhic"] = rhic_compare(D["rhic"], r, out)
        res["samples"][name] = out
        dn, di = out["dndeta_nsd"], out["dndeta_inel"]
        print(f"{name}: {n} events, with medium {out['frac_has_medium']:.3f}, NSD {out['frac_nsd']:.3f}")
        print(f"   dN/deta |eta|<0.5  NSD {dn['mid05']:.2f} +- {dn['mid05_err']:.2f} (UA5 {dn['ua5_mid05']:.2f})"
              f",  inel {di['mid05']:.2f} (UA5 {di['ua5_mid05']:.2f})")
        print("   NSD model/UA5 at |eta| = 0.1 / 1.1 / 2.1 / 3.1 / 4.1: " + " / ".join(
            f"{np.interp(x, ua5_eta, dn['ratio_to_ua5']):.2f}" for x in (0.125, 1.125, 2.125, 3.125, 4.125)))
        print("   NSD P(n): <n> model/UA5 " + ", ".join(
            f"|eta|<{w:g} {out[f'pn_{w:g}']['mean']:.2f}/{out[f'pn_{w:g}']['ua5_mean']:.2f}" for w in WINDOWS)
              + "; dispersion/mean " + ", ".join(
            f"{out[f'pn_{w:g}']['disp'] / out[f'pn_{w:g}']['mean']:.2f}/"
            f"{out[f'pn_{w:g}']['ua5_disp'] / out[f'pn_{w:g}']['ua5_mean']:.2f}" for w in WINDOWS))
        if "rhic" in out:
            rr = out["rhic"]
            print(f"   PHOBOS inel |eta|<1: model {rr['phobos']['mid1_model']:.2f} / data {rr['phobos']['mid1_data']:.2f};"
                  " model/PHOBOS at |eta| 0.1 / 1.1 / 2.1 / 3.1 / 4.1 / 4.9: " + " / ".join(
                      f"{np.interp(x, rr['phobos']['eta'], rr['phobos']['ratio']):.2f}" for x in (0.1, 1.1, 2.1, 3.1, 4.1, 4.9))
                  + "; shape (norm. |eta|<1) at 3.1 / 4.1 / 4.9: " + " / ".join(
                      f"{np.interp(x, rr['phobos']['eta'], rr['phobos']['shape_ratio']):.2f}" for x in (3.1, 4.1, 4.9)))
            print(f"   STAR NSD |eta|<0.5: model (STAR-like NSD, frac {rr['star']['frac_nsd_star']:.3f}) "
                  f"{rr['star']['model']:.2f} / data {rr['star']['data']:.2f} +- {rr['star']['err']:.2f};  "
                  f"PHENIX MB: model inel {rr['phenix']['model_inel']:.2f}, NSD {rr['phenix']['model_nsd']:.2f} / data "
                  f"{rr['phenix']['data']:.2f} +- {rr['phenix']['err']:.2f}")
    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(res, indent=1))
    print("wrote", OUT_JSON)
    return 0


def rhic_compare(R, r, out):
    """Model against PHOBOS (inelastic dN/deta, absolute and shape normalised at |eta| < 1), STAR
    (NSD with STAR's BBC coincidence, |eta| < 0.5) and PHENIX (mid-rapidity, |eta| < 0.5 here
    for its 0.35: the model's bins are 0.25 wide)."""
    ph = phobos_sym(R["phobos_dndeta"])
    keep = ph["eta"] < 5.0                                   # the model histogram ends at |eta| = 5
    di = out["dndeta_inel"]
    cen, my, me = np.array(di["eta"]), np.array(di["y"]), np.array(di["err"])
    m_at = np.interp(ph["eta"][keep], cen, my)
    me_at = np.interp(ph["eta"][keep], cen, me)
    mid_d = float(ph["y"][ph["eta"] < 1].mean())
    mid_m = float(my[cen < 1].mean())
    ns = r["nsd_star"]
    ys, es = dndeta(r["h"], ns)
    phx = R["phenix_mb"]; st = R["star_nsd"]
    return {
        "phobos": {"eta": ph["eta"][keep].tolist(), "data": ph["y"][keep].tolist(), "sys": ph["sys"][keep].tolist(),
                   "half_diff": ph["half_diff"][keep].tolist(), "model": m_at.tolist(), "model_err": me_at.tolist(),
                   "ratio": (m_at / ph["y"][keep]).tolist(),
                   "shape_ratio": ((m_at / mid_m) / (ph["y"][keep] / mid_d)).tolist(),
                   "mid1_model": mid_m, "mid1_data": mid_d, "norm": "inelastic"},
        "star": {"model": float(np.mean(ys[:2])), "model_err": float(np.sqrt(np.sum(es[:2] ** 2)) / 2),
                 "frac_nsd_star": float(ns.mean()), "data": st["dndeta"], "err": st["err"], "norm": st["norm"]},
        "phenix": {"model_inel": di["mid05"], "model_nsd": out["dndeta_nsd"]["mid05"], "data": phx["dndeta"],
                   "err": phx["err"], "norm": phx["norm"]},
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=("fetch", "analyse"))
    p.add_argument("names", nargs="*", default=None)
    p.add_argument("--out", default=None, help="output root of oo_bias.py (default its OUT)")
    a = p.parse_intermixed_args(argv)
    if a.out:
        B.OUT = Path(a.out)
    if a.step == "fetch":
        return fetch()
    names = a.names or [f"pp_{s}" for s in B.BULK_SETS]
    return analyse(names)


if __name__ == "__main__":
    sys.exit(main())
