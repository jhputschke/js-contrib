#!/usr/bin/env python3
"""
analysis/OO-RHIC/centrality_proxy.py -- step R4b, part 1 of the plan forward (tuning page):
a cheap, CPU-only first look at the centrality bias from the hard-momentum subtraction in O+O,
before any hydro.

The proxy: 3dMCGlauber's strings as MUSIC deposits them (run_prod_oo_isr.strings_eloc:
constant local energy per unit eta_s between the string ends plus the remnant ends' rest
energy), summed in the eta_s windows of STAR's centrality estimators: |eta| < 0.5 and < 1
(mid-rapidity multiplicity) and 2.1 < |eta| < 5.1 (EPD).  Without hydro and particlization this
stands in for the multiplicity those detectors would count; the jet's own hadrons are not in it
(for the forward windows that is a small part, R4; at mid-rapidity it is the autocorrelation part
of the bias, which this proxy leaves out).

Samples (all O+O 200 GeV, b in [0, 10] fm as 3dMCGlauber samples it, the O+O tune XML):
  mb          minimum bias: 3dMCGlauber alone (generateOnlyPositions 0, no hard scattering,
              strings made normally) -- defines the percentile classes, as experiments do
  jet_sub     jet events (run_prod_oo_isr.py --vacuum, ISR chain, hard-momentum subtraction on)
  jet_nosub   the same with --no-subtraction (the strings keep the full nucleon momentum; the
              hard pair's string veto of 3dMCGlauber 09e0a7e still applies)
Jet events in pTHat windows 9-11, 18-23, 30-40 and 40-55 GeV; each event weighted with its N_coll
(the probability of a hard scattering is proportional to N_coll, the generator puts exactly one
in every event).

Per class c of the minimum-bias proxy distribution (0-10, 10-20, 20-40, 40-60, 60-80, 80-100 %):
the bias factor <N_coll>_jet,c / <N_coll>_MB,c, which is the experiment-style R_AA of a jet
yield unchanged per hard scattering (R4: within ~2 %).

    python centrality_proxy.py run -j 20      # PROD_DIR/centrality_proxy/<sample>
    python centrality_proxy.py analyse        # -> results/centrality_proxy.json
    python centrality_proxy.py smear          # NBD fluctuations -> results/centrality_proxy_smear.json
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C  # noqa: E402
import hstat as H  # noqa: E402

#: v2 (2026-11-22): + sPHENIX MBD windows and the jet events' charged hadrons to |eta| < 5 (the
#: first round, EPD and mid-rapidity only, is in centrality_proxy/ itself)
OUT = C.PROD_DIR / "centrality_proxy" / "v2"
OO_XML = C.EXAMPLE / "config" / "OO_MCGlauber_MUSIC_isr_rhictune.xml"
B_MAX = 10.0
JET_WINDOWS = [(9, 11), (18, 23), (30, 40), (40, 55)]
JET_JOBS, JET_EVENTS = 10, 20000          # per jet sample: 200k events
MB_JOBS, MB_EVENTS = 10, 20000            # 200k minimum-bias events
CLASSES = [(0, 10), (10, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
COIN_THR = (0.0, 0.5, 1.0)                 # GeV of string local energy for an MBD arm to fire
ESTIMATORS = ("epd", "mid05", "mid10", "mbd")
#: jet samples; *_noveto with 3dMCGlauber's hard-pair string veto off (--hard-pair-string-veto 0,
#: needs X-SCAPE oo_rhic_run): the closure of the method (jet_nosub_noveto: strings as minimum
#: bias at fixed N_coll -> R_AA^vac = 1) and the subtraction alone (jet_sub_noveto)
JET_SAMPLES = ("jet_sub", "jet_nosub", "jet_sub_noveto", "jet_nosub_noveto", "jet_sub_mpioff",
               "jet_sub_mpi14")
#: variants of jet_sub on its seeds: extra Pythia lines (MPI pre-look, 2026-11-24)
VARIANT_LINES = {"_mpioff": ["PartonLevel:MPI = off"],
                 # soft MPI at the U1 setting that gives STAR's p+p UE level (vacuum chain)
                 "_mpi14": ["MultipartonInteractions:alphaSvalue = 0.130",
                            "MultipartonInteractions:pTmin = 0.2",
                            "MultipartonInteractions:ecmRef = 200",
                            "MultipartonInteractions:pT0Ref = 1.4"]}
#: where analyse/smear write (default results/; another directory keeps earlier results)
RES_DIR = Path(os.environ.get("CENTRALITY_PROXY_RESULTS", str(HERE / "results")))
ARMS = ("mbd_neg", "mbd_pos")
MBD_ETA = (3.51, 4.61)


def driver_module():
    spec = importlib.util.spec_from_file_location("_oo_isr", str(C.DRIVER))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def mb_xml(path):
    """The tune XML reduced to 3dMCGlauber alone, making its own strings."""
    tree = ET.parse(OO_XML)
    root = tree.getroot()
    for tag in ("Hard", "Preequilibrium", "Hydro", "Eloss", "JetHadronization",
                "SoftParticlization", "Afterburner"):
        for el in root.findall(tag):
            root.remove(el)
    mcg = root.find("IS/MCGlauber")
    mcg.find("generateOnlyPositions").text = "0"
    mcg.find("b_min").text = "0.0"
    mcg.find("b_max").text = f"{B_MAX}"
    tree.write(path)


def mb_job(seed, events, outdir):
    """One minimum-bias job: 3dMCGlauber alone, per event b, N_part, N_coll, strings and the
    local-energy proxy -> outdir/mb_seed<seed>.npz."""
    Z = driver_module()
    rp = Z.rp
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    xml = outdir / f"mb_seed{seed}.xml"
    mb_xml(xml)
    tree = ET.parse(xml)
    tree.getroot().find("Random/seed").text = str(seed)
    tree.getroot().find("nEvents").text = str(events)
    tree.write(xml)
    a = argparse.Namespace(build=rp.BUILD, main_xml=rp.MAIN_XML)
    workdir = outdir / "work" / f"mb_seed{seed}"
    main_xml = rp.enter_workdir(a, str(workdir), str(xml))
    import jetscape as js
    rec = {k: [] for k in ("b", "npart", "ncoll", "nstrings",
                           *[f"eloc_{e}" for e in ESTIMATORS + ARMS])}
    with js.per_event(main_xml, str(xml), n_events=events) as drv:
        for _ in drv.events():
            ini = drv.initial_state
            g = Z.glauber_record()
            s = ini.string_array() if hasattr(ini, "string_array") else None
            el = Z.strings_eloc(s)
            rec["b"].append(g.get("glauber_b", np.nan))
            rec["npart"].append(g.get("glauber_npart", -1))
            rec["ncoll"].append(g.get("glauber_ncoll", -1))
            rec["nstrings"].append(g.get("glauber_nstrings", -1))
            for e in ESTIMATORS + ARMS:
                rec[f"eloc_{e}"].append(el[f"strings_eloc_{e}"])
    np.savez(outdir / f"mb_seed{seed}.npz", **{k: np.asarray(v) for k, v in rec.items()})
    return 0


def jet_cmd(name, seed, xml):
    cmd = [sys.executable, str(C.DRIVER), "--vacuum", "--user-xml", str(xml),
           "--events", str(JET_EVENTS - JET_EVENTS % len(JET_WINDOWS)), "--seed", str(seed),
           "--pthat-bins", ",".join(f"{lo}-{hi}" for lo, hi in JET_WINDOWS),
           "--eloss-threads", "0", "--eta-max", "5.0", "--charged",
           "--b-min", "0", "--b-max", f"{B_MAX}", "--outdir", str(OUT / name),
           "--max-skipped", "30"]
    if name.startswith("jet_nosub"):
        cmd.append("--no-subtraction")
    if name.endswith("_noveto"):              # 3dMCGlauber's hard-pair string veto off
        cmd += ["--hard-pair-string-veto", "0"]
    for suffix, lines in VARIANT_LINES.items():
        if name.endswith(suffix):
            for line in lines:
                cmd += ["--pythia-line", line]
    return cmd


def run(nproc, names):
    OUT.mkdir(parents=True, exist_ok=True)
    xml = C.PROD_DIR / "closure" / "xml" / "OO.xml"
    if not xml.exists():
        xml.parent.mkdir(parents=True, exist_ok=True)
        xml.write_text(OO_XML.read_text())
    todo = []
    for k, name in enumerate(JET_SAMPLES):
        if name not in names:
            continue
        base = next((name[:-len(s)] for s in VARIANT_LINES if name.endswith(s)), name)
        kk = JET_SAMPLES.index(base)              # a variant runs on its base sample's seeds
        for j in range(JET_JOBS):
            seed = 990000 + 1000 * kk + j
            if not list((OUT / name).glob(f"*_seed{seed:04d}.json")):
                todo.append((name, seed, jet_cmd(name, seed, xml), C.DRIVER.parent))
    if "mb" in names:
        for j in range(MB_JOBS):
            seed = 995000 + j
            if not (OUT / "mb" / f"mb_seed{seed}.npz").exists():
                todo.append(("mb", seed, [sys.executable, str(Path(__file__).resolve()), "mb-job",
                                          str(seed), str(MB_EVENTS), str(OUT / "mb")], HERE))
    print(f"centrality_proxy.py: {len(todo)} jobs, {nproc} at a time -> {OUT}", flush=True)
    env = dict(os.environ, OMP_NUM_THREADS="1")
    t0 = time.time()

    def one(t):
        name, seed, cmd, cwd = t
        (OUT / name).mkdir(parents=True, exist_ok=True)
        with open(OUT / name / f"run_seed{seed}.log", "w") as log:
            r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(cwd))
        return name, seed, r.returncode

    nfail = 0
    with ThreadPoolExecutor(max_workers=nproc) as ex:
        for name, seed, rc in ex.map(one, todo):
            nfail += rc != 0
            print(f"{name} {seed}: {'ok' if rc == 0 else f'FAILED ({rc})'}, {time.time() - t0:.0f} s",
                  flush=True)
    return 1 if nfail else 0


def load_jet(name):
    import hdf5plugin  # noqa: F401
    import h5py
    keys = ("glauber_b", "glauber_ncoll", "glauber_npart", "glauber_nstrings", "pthat_bin",
            "strings_E_total", "isr_sub_E_total",
            *[f"strings_eloc_{e}" for e in ESTIMATORS + ARMS])
    out = {k: [] for k in keys + ("nch_mbd_neg", "nch_mbd_pos", "nch_epd")}
    import observables as O
    files = sorted(glob.glob(str(OUT / name / "*_hadrons_jet_frag.h5")))
    if not files:                              # sample not run: empty arrays
        return {k: np.zeros(0) for k in out}
    for f in files:
        with h5py.File(f) as h:
            n = int(h.attrs["nunits_written"])
            for k in keys:
                out[k].append(h[f"units/{k}"][:n] if f"units/{k}" in h else np.full(n, np.nan))
        fd = O.read_file(f)                    # charged hadrons of the hard event, |eta| < 5
        _, _, eta, _ = O._kin(fd["p"])
        ev = np.repeat(np.arange(len(fd["counts"])), fd["counts"])
        for arm, sgn in (("nch_mbd_neg", -1), ("nch_mbd_pos", 1)):
            m = (sgn * eta > MBD_ETA[0]) & (sgn * eta < MBD_ETA[1])
            out[arm].append(np.bincount(ev[m], minlength=len(fd["counts"])))
        m = (np.abs(eta) > 2.1) & (np.abs(eta) < 5.0)   # EPD (stored hadrons end at |eta| 5)
        out["nch_epd"].append(np.bincount(ev[m], minlength=len(fd["counts"])))
    return {k: np.concatenate(v) for k, v in out.items()}


def analyse():
    mb = {}
    for f in sorted((OUT / "mb").glob("mb_seed*.npz")):
        d = np.load(f)
        for k in d.files:
            mb.setdefault(k, []).append(d[k])
    mb = {k: np.concatenate(v) for k, v in mb.items()}
    ok = mb["ncoll"] > 0
    mb = {k: v[ok] for k, v in mb.items()}
    res = {"windows": JET_WINDOWS, "classes": CLASSES, "b_max": B_MAX,
           "n_mb": int(ok.sum()), "mb": {}, "jet": {}}
    edges, mb_cls = {}, {}
    for e in ESTIMATORS:
        x = mb[f"eloc_{e}"]
        # class c = percentile range of the minimum-bias proxy, most active first
        cuts = np.percentile(x, [100 - hi for _, hi in CLASSES])
        edges[e] = cuts
        cls = classify(x, cuts)
        nm, nme = H.class_mean(cls, mb["ncoll"], len(CLASSES))
        res["mb"][e] = {"cuts": cuts.tolist(), "ncoll": nm.tolist(), "ncoll_err": nme.tolist(),
                        "frac": [float(np.mean(cls == c)) for c in range(len(CLASSES))]}
        mb_cls[e] = cls
    res["mb_mean"] = {k: float(np.mean(mb[k])) for k in ("b", "npart", "ncoll", "nstrings")}
    for name in JET_SAMPLES:
        d = load_jet(name)
        if not len(d["glauber_ncoll"]):
            continue
        w = d["glauber_ncoll"].astype(float)             # hard-scattering probability
        r = {"n": int(len(w)), "per_window": {}}
        for k, wdw in enumerate(JET_WINDOWS):
            m = d["pthat_bin"] == k
            rw = {}
            for e in ESTIMATORS:
                cls = classify(d[f"strings_eloc_{e}"][m], edges[e])
                ww = w[m]
                frac_jet = np.array([ww[cls == c].sum() for c in range(len(CLASSES))]) / ww.sum()
                ncoll_jet, ncoll_jet_e = H.class_mean(cls, d["glauber_ncoll"][m], len(CLASSES), ww)
                ncoll_mb = np.asarray(res["mb"][e]["ncoll"])
                # experiment-style R_AA^vac per class: jet yield per MB event in c / (<Ncoll>_MB,c
                # x yield per binary collision); with N_coll-weighted sampling:
                # = (frac_jet,c / frac_MB,c) * <Ncoll>_MB / <Ncoll>_MB,c  (N_coll-weighted jets)
                rv = H.raa_vac(mb_cls[e], mb["ncoll"], cls, ww, len(CLASSES))
                bias, bias_e = H.ratio(ncoll_jet, ncoll_jet_e, ncoll_mb, res["mb"][e]["ncoll_err"])
                rw[e] = {"frac_jet": frac_jet.tolist(), "ncoll_jet": ncoll_jet.tolist(),
                         "ncoll_jet_err": ncoll_jet_e.tolist(), "bias": bias.tolist(), "bias_err": bias_e.tolist(),
                         "raa_vac": rv["raa"].tolist(), "raa_err": rv["err"].tolist(),
                         "mean_eloc_jet": float(np.average(d[f"strings_eloc_{e}"][m], weights=ww))}
            r["per_window"][f"{wdw[0]}-{wdw[1]}"] = rw
        res["jet"][name] = r
    # sPHENIX's minimum-bias condition, >= 1 hit in each MBD arm: in the proxy an arm fires
    # if strings deposit local energy above THR in it (minimum bias), or that or a charged
    # hadron of the hard event is in it (jet events)
    coin = {"thresholds_GeV": list(COIN_THR), "mb": {}, "jet": {}}
    nbins = [(1, 1), (2, 2), (3, 4), (5, 8), (9, 1000)]
    for thr in COIN_THR:
        fire = (mb["eloc_mbd_neg"] > thr) & (mb["eloc_mbd_pos"] > thr)
        bn = [H.fraction(fire[(mb["ncoll"] >= lo) & (mb["ncoll"] <= hi)]) for lo, hi in nbins]
        coin["mb"][str(thr)] = {"all": float(fire.mean()), "all_err": H.fraction(fire)[1],
                                "by_ncoll": [v for v, _ in bn], "by_ncoll_err": [e for _, e in bn]}
    for name in JET_SAMPLES:
        d = load_jet(name)
        if not len(d["glauber_ncoll"]):
            continue
        w = d["glauber_ncoll"].astype(float)
        coin["jet"][name] = {}
        for thr in COIN_THR:
            arms = [(d[f"strings_eloc_{a}"] > thr) | (d[f"nch_{a}"] > 0) for a in ARMS]
            strings_only = (d["strings_eloc_mbd_neg"] > thr) & (d["strings_eloc_mbd_pos"] > thr)
            fire = arms[0] & arms[1]
            bn = [H.fraction(fire[(d["glauber_ncoll"] >= lo) & (d["glauber_ncoll"] <= hi)]) for lo, hi in nbins]
            coin["jet"][name][str(thr)] = {
                "weighted": float(np.average(fire, weights=w)), "weighted_err": H.fraction(fire, w)[1],
                "by_ncoll_err": [e for _, e in bn],
                "weighted_strings_only": float(np.average(strings_only, weights=w)),
                "by_ncoll": [float(fire[(d["glauber_ncoll"] >= lo) & (d["glauber_ncoll"] <= hi)].mean())
                             for lo, hi in nbins],
                "by_ncoll_strings_only": [float(strings_only[(d["glauber_ncoll"] >= lo)
                                                             & (d["glauber_ncoll"] <= hi)].mean())
                                          for lo, hi in nbins]}
    coin["ncoll_bins"] = nbins
    res["mbd_coincidence"] = coin
    for thr in COIN_THR:
        print(f"MBD coincidence (arm threshold {thr} GeV): minimum bias {coin['mb'][str(thr)]['all']:.3f}; "
              + ", ".join(f"{n} N_coll-weighted {c[str(thr)]['weighted']:.4f} (strings only "
                          f"{c[str(thr)]['weighted_strings_only']:.3f})" for n, c in coin["jet"].items()))
    path = RES_DIR / "centrality_proxy.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=1))
    lab = [f"{lo}-{hi}%" for lo, hi in CLASSES]
    print(f"minimum bias: {res['n_mb']} events, <Ncoll> {res['mb_mean']['ncoll']:.2f}")
    for name, r in res["jet"].items():
        for wkey, rw in r["per_window"].items():
            for e in ("epd", "mid10", "mbd"):
                print(f"{name:9s} pTHat {wkey:6s} {e:5s} R_AA^vac: "
                      + " ".join(f"{l} {v:.2f}" for l, v in zip(lab, rw[e]["raa_vac"])))
    print("wrote", path)
    return 0


# ── step 1 of the plan forward: fluctuations (negative-binomial smearing of the proxy) ─────────
SMEAR_MU = (0.3, 0.6, 1.0, 2.0)            # charged particles per GeV of string local energy
SMEAR_KAPPA = (0.1, 0.3, 1.0, 3.0, 1e6)    # NBD shape per GeV (1e6: Poisson)
SPH_CLASSES = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]


def nbd(rng, energy, mu, kappa):
    """Counts per event and arm: NBD with mean mu * E and shape kappa * E (a sum of independent
    NBDs per GeV, as Glauber + NBD per ancestor); E = 0 gives 0."""
    e = np.asarray(energy, dtype=float)
    out = np.zeros(len(e), dtype=np.int64)
    m = e > 0
    r = kappa * e[m]
    out[m] = rng.negative_binomial(r, r / (r + mu * e[m]))
    return out


def smear():
    """Classes from smeared counts in the MBD (each arm smeared, the sum classifies; >= 1 count
    in each arm is the minimum-bias coincidence) and in the EPD.  Jet events add their own
    charged hadrons in the MBD arms and the EPD (no smearing, they are particles).  Without the
    subtraction the hard pair's remnant is counted twice (in the strings and as iColored's
    remnant hadrons, as --no-subtraction does on purpose), so that reference is not meaningful
    here; the subtraction case is consistent.  Per (mu, kappa): the
    minimum-bias coincidence efficiency (sPHENIX: 0.914), the experiment-style R_AA^vac in
    sPHENIX's classes for the jet events (subtraction on and off, N_coll-weighted) and R_CP;
    the minimum-bias classes are built from minimum-bias events passing the coincidence and
    anchored to all events, as with a trigger efficiency."""
    rng = np.random.default_rng(7)
    mb = {}
    for f in sorted((OUT / "mb").glob("mb_seed*.npz")):
        d = np.load(f)
        for k in d.files:
            mb.setdefault(k, []).append(d[k])
    mb = {k: np.concatenate(v) for k, v in mb.items()}
    mb = {k: v[mb["ncoll"] > 0] for k, v in mb.items()}
    jets = {n: load_jet(n) for n in JET_SAMPLES if list((OUT / n).glob("*_hadrons_jet_frag.h5"))}
    res = {"mu": SMEAR_MU, "kappa": SMEAR_KAPPA, "classes": SPH_CLASSES, "grid": []}
    nc = len(SPH_CLASSES)

    def classes_from(counts_mb, counts):
        # percentiles of all minimum-bias events (the Glauber + NBD anchoring), random tie-break
        xm = counts_mb + rng.random(len(counts_mb))
        cuts = np.percentile(xm, [100 - hi for _, hi in SPH_CLASSES])
        def cls(v):
            c = np.full(len(v), nc - 1)
            for k in range(nc - 1, -1, -1):
                c[v >= cuts[k]] = k
            return c
        return cls(xm), cls(counts + rng.random(len(counts)))

    for mu in SMEAR_MU:
        for kappa in SMEAR_KAPPA:
            row = {"mu": mu, "kappa": kappa}
            n_neg = nbd(rng, mb["eloc_mbd_neg"], mu, kappa)
            n_pos = nbd(rng, mb["eloc_mbd_pos"], mu, kappa)
            row["eps_trig_mb"] = float(np.mean((n_neg > 0) & (n_pos > 0)))
            n_epd_mb = nbd(rng, mb["eloc_epd"], mu, kappa)
            for est in ("mbd", "epd"):
                cnt_mb = n_neg + n_pos if est == "mbd" else n_epd_mb
                r_est = {}
                for name, d in jets.items():
                    w = d["glauber_ncoll"].astype(float)
                    jn = nbd(rng, d["strings_eloc_mbd_neg"], mu, kappa) + d["nch_mbd_neg"]
                    jp = nbd(rng, d["strings_eloc_mbd_pos"], mu, kappa) + d["nch_mbd_pos"]
                    cnt_j = jn + jp if est == "mbd" else nbd(rng, d["strings_eloc_epd"], mu, kappa) + d["nch_epd"]
                    cm, cj = classes_from(cnt_mb, cnt_j)
                    nmb = np.array([mb["ncoll"][cm == k].mean() for k in range(nc)])
                    rv = H.raa_vac(cm, mb["ncoll"], cj, w, nc)
                    (r03, e03), (r02, e02) = H.rcp(rv, 0, 3), H.rcp(rv, 0, 2)
                    r_est[name] = {"raa_vac": rv["raa"].tolist(), "raa_err": rv["err"].tolist(),
                                   "ncoll_mb": nmb.tolist(),
                                   "rcp_0_20_60_80": r03, "rcp_0_20_60_80_err": e03,
                                   "rcp_0_20_40_60": r02, "rcp_0_20_40_60_err": e02,
                                   "eps_coinc_jet": float(np.average((jn > 0) & (jp > 0), weights=w))}
                row[est] = r_est
            res["grid"].append(row)
            m = row["mbd"]["jet_sub"]
            print(f"mu {mu:3.1f}/GeV kappa {kappa:7.1f}/GeV: eps_trig(MB) {row['eps_trig_mb']:.3f} | MBD classes, "
                  f"subtraction: R_AA^vac " + " ".join(f"{v:.2f}" for v in m["raa_vac"][:4])
                  + f"  R_CP 0-20/60-80 {m['rcp_0_20_60_80']:.2f}, 0-20/40-60 {m['rcp_0_20_40_60']:.2f}"
                  + (f" | no sub {row['mbd']['jet_nosub']['rcp_0_20_60_80']:.2f}" if "jet_nosub" in row["mbd"] else "")
                  + f" | EPD sub {row['epd']['jet_sub']['rcp_0_20_60_80']:.2f}", flush=True)
    path = RES_DIR / "centrality_proxy_smear.json"
    path.write_text(json.dumps(res, indent=1))
    print("wrote", path)
    return 0


def classify(x, cuts):
    """cuts: ascending proxy values at 100-hi percentiles (most active class first)."""
    # class 0 = highest proxy: x >= cuts[0]; class k: cuts[k] <= x < cuts[k-1]
    c = np.full(len(x), len(CLASSES) - 1)
    for k in range(len(CLASSES) - 1, -1, -1):
        c[x >= cuts[k]] = k
    return c


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "mb-job":
        return mb_job(int(argv[1]), int(argv[2]), argv[3])
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=("run", "analyse", "smear"))
    p.add_argument("names", nargs="*", default=["mb", *JET_SAMPLES])
    p.add_argument("-j", type=int, default=os.cpu_count())
    a = p.parse_args(argv)
    if a.step == "smear":
        return smear()
    return run(a.j, a.names) if a.step == "run" else analyse()


if __name__ == "__main__":
    sys.exit(main())
