#!/usr/bin/env python3
"""
analysis/OO-RHIC/g2_check.py -- gate G2 of the physics run (tuning page, "Physics run plan"): the
MPI checks of phase 2, with the frozen hard + bulk tune (config/OO_MCGlauber_MUSIC_isr_rhictune.xml,
js-contrib 788ca1e).  Figures and the reading: g2_mpi_check.ipynb.

    python g2_check.py [--out /data/OO-RHIC_phys] [-j 16] [--cached]   # -> results/g2_check.json

Part 1, the p+p underlying event in the production chain (strings, MUSIC + iSS, hard-momentum
subtraction, oo_bias.py PP options), against STAR (PRD 101, 052004; ue_check.py's definitions:
leading charged+neutral anti-kT R = 0.6 jet in |eta| < 0.4, charged pT > 0.2 GeV, |eta| < 1):
  pp_bulk_w*          MPI as in the tune (pTmin 5 GeV, Pythia lowers it to ~1.7 GeV at 200 GeV)
  pp_bulk_mpioff_w*   PartonLevel:MPI = off, the same seeds
  pp_bulk_switch_w*   MPI on, --hard-pair-string-veto 0 (a string in every jet event, U3b)
  nine pTHat windows 5-55 GeV (one sub-sample each, stitched with sigma_k / N_k); leading-jet bins
  below ~9-11 GeV are incomplete (no pTHat < 5 GeV).
The minimum-bias level in the same acceptance comes from pp_tau0424 (the p+p minimum-bias
stand-in with the final XML, /data/OO-RHIC_bulk): charged pT > 0.2 GeV, |eta| < 1, per event and
d eta d phi, for all events and for NSD events (>= 1 charged particle in 3.3 < eta < 5 and in
-5 < eta < -3.3, pt_bulk.py's trigger); that sample's NSD dN/deta agrees with STAR's (pp_bulk.py).
The hard tune alone (vacuum chain, no strings or hydro; results/ue_check.json "ref_tune") is
carried along as the reference without soft sector.

Part 2, the O+O vacuum bias with MPI on and off: jets_vac_w18_23 / _w30_40 (MPI on) and
jets_vac_mpioff_* (off, the same seeds), classes from phys_mb (oo_bias.class_analysis: EPD and
|eta| < 1 in STAR's classes, MBD in sPHENIX's), N_coll-weighted; R_AA^vac, R_CP, bias factor,
the MBD coincidence of jet events and <N_EPD> vs N_coll (the mechanism).  The arms start from the
same seeds but do not share geometries event by event (MPI off changes the random-number stream;
the job checks it: same_geometry_fraction), so on - off has independent errors.

Model results, not vetted physics results.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hstat as H  # noqa: E402
import oo_bias as B  # noqa: E402
import ue_check as U  # noqa: E402

UE_VARIANTS = {"pp_bulk": "MPI on (tune)", "pp_bulk_mpioff": "MPI off", "pp_bulk_switch": "MPI on, string veto 0"}
VAC_WINDOWS = ("w18_23", "w30_40")
VAC_ARMS = {"jets_vac": "MPI on (tune)", "jets_vac_mpioff": "MPI off"}
MB_PP = ("/data/OO-RHIC_bulk", "pp_tau0424")
BBC = (3.3, 5.0)
#: leading-jet bins (ue_check.LEAD_EDGES) fully covered by pTHat >= 5 GeV
COMPLETE_FROM = 11.0
XML = B.C.EXAMPLE / "config" / "OO_MCGlauber_MUSIC_isr_rhictune.xml"


# ── settings: what the samples ran with ──────────────────────────────────────────────────────
#: the XML commit phases 1-2 ran with (the MPI of the tune before revision (3), soft MPI)
G2_XML_COMMIT = "788ca1e"


def settings():
    """The hard and bulk tune values of the XML the G2 samples ran with (commit G2_XML_COMMIT,
    read from git, not assumed; the file itself has soft MPI since revision (3))."""
    rel = XML.relative_to(B.C.EXAMPLE.parent.parent.parent)
    try:
        x = subprocess.run(["git", "show", f"{G2_XML_COMMIT}:{rel}"], cwd=HERE, capture_output=True,
                           text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        sys.exit(f"g2_check.py: cannot read {rel} at {G2_XML_COMMIT} from git")
    x_nc = re.sub(r"<!--.*?-->", "", x, flags=re.S)

    def tag(name, block=None):
        s = x_nc
        if block:
            m = re.search(rf"<{block}>(.*?)</{block}>", s, re.S)
            s = m.group(1) if m else ""
        m = re.search(rf"<{name}>\s*([^<]*?)\s*</{name}>", s)
        return m.group(1) if m else None
    lines = re.search(r"<LinesToRead>(.*?)</LinesToRead>", x_nc, re.S)
    py = {}
    for ln in (lines.group(1).splitlines() if lines else []):
        if "=" in ln:
            k, v = (s.strip() for s in ln.split("=", 1))
            py[k] = v
    hard = {"Pythia (PythiaIsrGun)": py,
            "Matter Q0 [GeV]": tag("Q0", "Matter"), "Matter vir_factor (C_v)": tag("vir_factor", "Matter"),
            "Matter alphas (medium)": tag("alphas", "Matter"), "LBT alphas (medium)": tag("alphas", "Lbt")}
    bulk = {k: tag(k) for k in ("ylossParam4At2", "ylossParam4At4", "ylossParam4At6", "ylossParam4At10",
                                "ylossParam4var", "shadowing_factor", "BG", "remnant_energy_loss_fraction",
                                "tau_form_mean", "freezeout_temperature")}
    commit = subprocess.run(["git", "log", "-1", "--format=%h %ad %s", "--date=short", G2_XML_COMMIT],
                            cwd=HERE, capture_output=True, text=True).stdout.strip()
    return {"xml": str(rel), "xml_commit": commit, "xml_modified_since": False, "hard": hard, "bulk": bulk}


def job_settings(out, sample):
    """Per-sample options on top of the XML, from one finished job: its summary (xml_set, the
    string veto, subtraction, vacuum jet) and its exact XML (the Pythia lines, MPI on / off)."""
    fs = sorted(glob.glob(str(out / sample / "OO_isr_seed*.json")))
    if not fs:
        return None
    d = json.loads(Path(fs[0]).read_text())
    xml = Path(fs[0]).with_suffix(".xml")
    lines = []
    if xml.exists():
        m = re.search(r"<PythiaGun>.*?<LinesToRead>(.*?)</LinesToRead>", re.sub(r"<!--.*?-->", "", xml.read_text(), flags=re.S), re.S)
        lines = [ln.strip() for ln in (m.group(1).splitlines() if m else []) if ln.strip()]
    return {"xml_set": d.get("xml_set"), "hard_pair_string_veto": d.get("hard_pair_string_veto"),
            "subtract_hard_momentum": d.get("subtract_hard_momentum"), "vacuum_jet": d.get("vacuum_jet"),
            "pythia_lines": lines, "mpi_off": any(re.sub(r"\s", "", ln) == "PartonLevel:MPI=off" for ln in lines)}


def sample_status(out, pattern):
    rows = {}
    for d in sorted(glob.glob(str(out / pattern))):
        js = glob.glob(d + "/OO_isr_seed*.json")
        ev = sum(json.loads(Path(f).read_text()).get("events_written", 0) for f in js)
        rows[Path(d).name] = {"jobs_done": len(js), "events": int(ev)}
    return rows


# ── part 1: p+p underlying event ─────────────────────────────────────────────────────────────
def ue_files(out, variant):
    fs = sorted(glob.glob(str(out / f"{variant}_w[0-9]*" / "*_hadrons_jet_frag.h5")))
    return [f for f in fs if os.path.exists(f.replace("_hadrons_jet_frag.h5", ".json"))]


def ue_sums(files, j):
    """ue_check.run_sums over an explicit file list (the window sub-samples of one variant)."""
    with ProcessPoolExecutor(max_workers=j) as ex:
        res = list(ex.map(U.file_sums, files))
    edges = sorted({w for r in res for w in r[0]})
    nwin, nb = len(edges), len(U.LEAD_EDGES) - 1
    acc = {k: np.zeros((nwin, nb)) for k in res[0][4]}
    sig_w, nacc, nev = np.zeros(nwin), np.zeros(nwin), np.zeros(nwin)
    for bins, sig, na, ne, o in res:
        g = np.array([edges.index(w) for w in bins])
        for k in acc:
            acc[k][g] += o[k]
        sig_w[g] += na * sig
        nacc[g] += na
        nev[g] += ne
    acc.update(sigma=sig_w / np.maximum(nacc, 1), n_events=nev, pthat_bins=np.asarray(edges, float))
    return acc


def ue_rows(acc):
    d = U.densities(acc)
    rows = []
    for i in range(len(U.LEAD_EDGES) - 1):
        r = {k: float(d[k][0][i]) for k in ("n_toward", "n_transverse", "n_away", "n_trans05", "n_transmax",
                                            "n_transmin", "meanpt_transverse", "has_bulk")}
        r |= {f"{k}_err": float(d[k][1][i]) for k in U.QUANT if d[k][1] is not None}
        r["events"] = int(d["n_events"][i])
        rows.append(r)
    # share of each leading-jet bin from the lowest window (5-7): how much the missing pTHat < 5
    # would matter is bounded by it
    low = [i for i, w in enumerate(acc["pthat_bins"].tolist()) if w[0] < 7]
    rest = [i for i in range(len(acc["sigma"])) if i not in low]
    W_all, W_rest = U.densities(acc)["weight"], U.densities(acc, rest)["weight"]
    share = 1 - W_rest / np.where(W_all > 0, W_all, np.nan)
    return rows, np.nan_to_num(share).tolist(), acc["n_events"].tolist(), acc["pthat_bins"].tolist()


def mb_level(root, sample):
    """Charged pT > 0.2 (and > 0.5) GeV, |eta| < 1 per event and per d eta d phi in the p+p
    minimum-bias stand-in (bulk hadrons only, as oo_bias's mb), all events and NSD."""
    import h5py
    import hdf5plugin  # noqa: F401
    n02, n05, nsd = [], [], []
    for bf in sorted(glob.glob(f"{root}/{sample}/*_hadrons_bulk_bg.h5")):
        if not os.path.exists(bf.replace("_hadrons_bulk_bg.h5", ".json")):
            continue
        with h5py.File(bf) as h:
            n = int(h.attrs["nunits_written"])
            p, c = B._units(h, n)
        E, px, py, pz = p.T
        pt = np.hypot(px, py)
        pa = np.sqrt(px * px + py * py + pz * pz)
        eta = 0.5 * np.log(np.clip(pa + pz, 1e-12, None) / np.clip(pa - pz, 1e-12, None))
        ev = np.repeat(np.arange(n), c)
        mid = np.abs(eta) < 1.0
        n02.append(np.bincount(ev[mid & (pt > 0.2)], minlength=n))
        n05.append(np.bincount(ev[mid & (pt > 0.5)], minlength=n))
        pos = np.bincount(ev[(eta > BBC[0]) & (eta < BBC[1])], minlength=n)
        neg = np.bincount(ev[(-eta > BBC[0]) & (-eta < BBC[1])], minlength=n)
        nsd.append((pos > 0) & (neg > 0))
    if not n02:
        return None
    n02, n05, nsd = (np.concatenate(v) for v in (n02, n05, nsd))
    area = 2 * 1.0 * 2 * np.pi
    out = {"sample": f"{root}/{sample}", "events": int(len(n02)), "frac_nsd": float(nsd.mean())}
    for lab, m in (("all", np.ones(len(n02), bool)), ("nsd", nsd)):
        for k, x in (("pt02", n02), ("pt05", n05)):
            out[f"{lab}_{k}"] = float(x[m].mean() / area)
            out[f"{lab}_{k}_err"] = float(x[m].std() / np.sqrt(m.sum()) / area)
    return out


def ue_part(out, j, cached):
    star = U.load_star()
    res = {"star": {k: (v if k == "edges" else {kk: np.asarray(vv).tolist() for kk, vv in v.items()})
                    for k, v in star.items()},
           "complete_from_gev": COMPLETE_FROM, "variants": {}}
    for v, lab in UE_VARIANTS.items():
        npz = HERE / "results" / f"ue_phys_{v}.npz"
        files = ue_files(out, v)
        if cached and npz.exists():
            acc = dict(np.load(npz))
        elif files:
            acc = ue_sums(files, j)
            np.savez(npz, **acc)
        else:
            print(f"g2_check.py: no finished files for {v}")
            continue
        rows, share, nev, bins = ue_rows(acc)
        res["variants"][v] = {"label": lab, "files": len(files), "rows": rows, "low_window_share": share,
                              "events_per_window": nev, "pthat_bins": bins,
                              "job": job_settings(out, f"{v}_w09_11")}
        t = [r["n_transverse"] for r in rows]
        print(f"{v:16s} transverse " + " ".join(f"{x:.3f}" for x in t))
    print("STAR             transverse " + " ".join(f"{x:.3f}" for x in star["transverse"]["value"]))
    try:
        ref = json.loads((HERE / "results" / "ue_check.json").read_text())["runs"]["ref_tune"]
        res["hard_only"] = {"label": "hard tune alone (vacuum chain, ref_tune)", "rows": ref}
    except (OSError, KeyError, ValueError):
        res["hard_only"] = None
    res["mb_level"] = mb_level(*MB_PP)
    try:
        pb = json.loads((HERE / "results" / "pp_bulk.json").read_text())
        m = pb["samples"][MB_PP[1]]["dndeta_nsd"]
        s = pb["data"]["dndeta_nsd"]
        res["mb_level"]["dndeta_nsd_eta05"] = {"model": float(np.mean(m["y"][:2])), "star": float(np.mean(s["y"][:2])),
                                               "star_err": float(np.mean(s["err"][:2]))}
    except (OSError, KeyError, ValueError, TypeError):
        pass
    # summary numbers per variant over the complete leading-jet bins
    lo = np.asarray(U.LEAD_EDGES[:-1])
    comp = lo >= COMPLETE_FROM
    sv = np.asarray(star["transverse"]["value"])
    se = np.hypot(star["transverse"]["stat"], star["transverse"]["sys"])
    mbl = res["mb_level"]["nsd_pt02"] if res["mb_level"] else np.nan
    for v, r in res["variants"].items():
        t = np.array([x["n_transverse"] for x in r["rows"]])
        te = np.array([x["n_transverse_err"] for x in r["rows"]])
        ratio = t / sv
        r["summary"] = {"ratio_to_star_complete_bins": ratio[comp].tolist(),
                        "mean_ratio_to_star": float(np.mean(ratio[comp])),
                        "chi2_complete_bins": float(np.sum(((t - sv) / np.hypot(te, se))[comp] ** 2)),
                        "n_bins": int(comp.sum()),
                        "trend_first_over_last": float(t[comp][0] / t[comp][-1]),
                        "star_trend_first_over_last": float(sv[comp][0] / sv[comp][-1]),
                        "over_mb_nsd": (t[comp] / mbl).tolist()}
    res["star_over_mb_nsd"] = (sv[comp] / mbl).tolist()
    return res


# ── part 2: O+O vacuum bias, MPI on / off ───────────────────────────────────────────────────
def vac_part(out):
    B.OUT = out
    mb = B.load("phys_mb")
    if not mb:
        return None
    res = {"mb": {"n": int(len(mb["glauber_ncoll"])), "ncoll_mean": float(mb["glauber_ncoll"].mean()),
                  "eps_mbd_coincidence": float(((mb["n_mbd_neg"] > 0) & (mb["n_mbd_pos"] > 0)).mean())},
           "star_classes": B.STAR_CLASSES, "sph_classes": B.SPH_CLASSES, "windows": {}}
    nc_edges = np.array([0.5, 1.5, 2.5, 4.5, 7.5, 11.5, 16.5, 22.5, 30.5, 40.5, 60.5])
    v, e = H.profile(mb["glauber_ncoll"], mb["n_epd"], nc_edges)
    res["epd_vs_ncoll"] = {"edges": nc_edges.tolist(), "mb": v.tolist(), "mb_err": e.tolist()}
    for w in VAC_WINDOWS:
        R = {}
        loaded = {}
        for arm in VAC_ARMS:
            name = f"{arm}_{w}"
            if not (out / name).exists():
                continue
            jet = B.load(name)
            if not jet:
                continue
            loaded[arm] = jet
            wt = jet["glauber_ncoll"].astype(float)
            coin = (jet["n_mbd_neg"] > 0) & (jet["n_mbd_pos"] > 0)
            r = {"n": int(len(wt)), "eps_mbd_coincidence_weighted": float(np.average(coin, weights=wt)),
                 "eps_mbd_coincidence_weighted_err": H.fraction(coin, wt)[1], "est": {},
                 "job": job_settings(out, name)}
            for est, classes in (("epd", B.STAR_CLASSES), ("mid10", B.STAR_CLASSES), ("mbd", B.SPH_CLASSES)):
                # the same random tie-break for both arms
                r["est"][est] = B.class_analysis(mb, jet, est, classes, np.random.default_rng(11))
            v, e = H.profile(jet["glauber_ncoll"], jet["n_epd"], nc_edges)
            r["epd_vs_ncoll"], r["epd_vs_ncoll_err"] = v.tolist(), e.tolist()
            R[arm] = r
        if len(loaded) == 2:
            a, b = loaded.values()
            n = min(len(a["glauber_b"]), len(b["glauber_b"]))
            R["same_geometry_fraction"] = float(np.mean(np.isclose(a["glauber_b"][:n], b["glauber_b"][:n])))
            R["paired_events"] = n
            # on - off per R_CP (independent errors: the arms do not share geometries)
            d = {}
            for est in ("epd", "mid10", "mbd"):
                for key, (x, ex) in R["jets_vac"]["est"][est]["rcp"].items():
                    y, ey = R["jets_vac_mpioff"]["est"][est]["rcp"][key]
                    d[f"{est} {key}"] = (x - y, float(np.hypot(ex, ey)))
            R["rcp_on_minus_off"] = d
        res["windows"][w] = R
        for arm, r in R.items():
            if isinstance(r, dict) and "est" in r:
                m = r["est"]["mbd"]["rcp"]["0-20/60-80"]
                p = r["est"]["epd"]["rcp"]["0-10/60-80"]
                print(f"{w} {arm:16s} n {r['n']:6d}  R_CP EPD 0-10/60-80 {p[0]:.3f}+-{p[1]:.3f}  "
                      f"MBD 0-20/60-80 {m[0]:.3f}+-{m[1]:.3f}")
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="/data/OO-RHIC_phys", help="output root of the physics run")
    p.add_argument("-j", type=int, default=16)
    p.add_argument("--cached", action="store_true", help="reuse results/ue_phys_<variant>.npz")
    p.add_argument("--skip", choices=("ue", "vac"), default=None)
    a = p.parse_args(argv)
    out = Path(a.out)
    (HERE / "results").mkdir(exist_ok=True)
    res = {"settings": settings(),
           "status": {"ue": sample_status(out, "pp_bulk*_w*"), "vac": sample_status(out, "jets_vac*_w*"),
                      "mb": sample_status(out, "phys_mb")},
           "note": "model results, not vetted physics results"}
    res["ue"] = None if a.skip == "ue" else ue_part(out, a.j, a.cached)
    res["vac"] = None if a.skip == "vac" else vac_part(out)
    path = HERE / "results" / "g2_check.json"
    path.write_text(json.dumps(res, indent=1, default=float))
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
