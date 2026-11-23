#!/usr/bin/env python3
"""
analysis/OO-RHIC/closure_oo.py -- steps R3 and R4 of the plan forward (tuning page, #forward):

  R3  the p+p reference for O+O, in exactly the O+O configuration:
      config/OO_MCGlauber_MUSIC_isr_rhictune.xml (the tune's Pythia lines: pSet 20, alpha_s
      0.152 for the hard process and the MPI, MPI pTmin 5; Matter Q0 2, vir_factor 0.25) with
      3dMCGlauber's projectile and target set to p, run with --vacuum.  Compared with the data
      (chi2 as in the tune) and with the tune's runs.
  R4  the vacuum closure: the same XML for O+O with --vacuum (no hydro, no LBT, Matter in
      vacuum) in impact-parameter classes; per hard scattering the O+O spectra against p+p.

    python closure_oo.py run -j 20         # all samples (resumable), PROD_DIR/closure/<name>
    python closure_oo.py run pp OO_b0-2    # some of them
    python closure_oo.py analyse -j 6      # -> PROD_DIR/obs/closure_<name>.npz, results/closure_oo*.{json,npz}
    jupyter nbconvert --to notebook --execute closure_oo.ipynb --output results/closure_oo_executed.ipynb

Samples (all --vacuum, the tune's pTHat windows config.PTHAT_WINDOWS, pi0 and eta undecayed,
|eta| < config.ETA_MAX stored; seeds 980000 + 1000 * sample + job, pp_low 979000 + job):
  pp          30 jobs x 30k events (900k)
  OO_b<lo-hi> 10 jobs x 30k events (300k) per class: b in [0,2], [2,4], [4,6], [6,10] fm
  pp_low      R3's pi0: 60 jobs x 30k events (1.8M) in config.LOW_WINDOWS (5-23 GeV), analysed
              together with pp (obs closure_pp_all) and in 4 independent subsets (closure_pp_split<i>)
and, for the eta distributions of the hard event and the remnants, without the eta cut and with
iColored's input stored: <name>_eta, pTHat 9-11 and 23-30 GeV, 20k events, for pp and each class.

What the generator does (and so what R4 can and cannot see): every event has exactly one hard
scattering, from Pythia for a free p+p pair (Beams:idA/idB = 2212 in PythiaIsrGun), put at one
of the event's binary collisions; its momentum is subtracted from the two nucleons, their
remnants (what 3dMCGlauber leaves of them) go to iColored.  N_coll scaling is therefore built in,
and neither isospin nor energy conservation can veto the hard scattering.  R4 measures what is
left: whether the O+O environment (the nucleons' other collisions, their remnants, the colour
flow into iColored) changes the hard yield per scattering, and how often the bookkeeping fails.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C  # noqa: E402

OO_XML = C.EXAMPLE / "config" / "OO_MCGlauber_MUSIC_isr_rhictune.xml"
TUNE_POINT = {"alpha_s": 0.152, "Cv": 0.25, "Q0": 2.0}         # what that XML carries
B_CLASSES = [(0, 2), (2, 4), (4, 6), (6, 10)]
EVENTS_PER_JOB = 30000
ETA_WINDOWS = [(9, 11), (23, 30)]
ETA_EVENTS = 20000
CH_EDGES = np.array([2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 17, 20, 25, 30], dtype=float)
CLOSURE = C.PROD_DIR / "closure"
NSPLIT = 4
XML_DIR = CLOSURE / "xml"


def samples():
    """name -> (system, b range or None, jobs, windows, events per job, eta sample?)"""
    out = {"pp": ("pp", None, 30, C.PTHAT_WINDOWS, EVENTS_PER_JOB, False)}
    for lo, hi in B_CLASSES:
        out[f"OO_b{lo}-{hi}"] = ("OO", (lo, hi), 10, C.PTHAT_WINDOWS, EVENTS_PER_JOB, False)
    for name in list(out):
        sysn, b, _, _, _, _ = out[name]
        out[name + "_eta"] = (sysn, b, 1, ETA_WINDOWS, ETA_EVENTS, True)
    # R3: more p+p events in the low windows for the pi0 (appended last: the seeds of the
    # samples above depend on their position)
    out["pp_low"] = ("pp", None, 60, C.LOW_WINDOWS, EVENTS_PER_JOB, False)
    return out


def write_xmls():
    """The O+O tune XML as is, and its p+p version (projectile/target p, no nucleus file)."""
    XML_DIR.mkdir(parents=True, exist_ok=True)
    oo = XML_DIR / "OO.xml"
    oo.write_text(OO_XML.read_text())
    tree = ET.parse(OO_XML)
    mcg = tree.getroot().find("IS/MCGlauber")
    mcg.find("projectile").text = "p"
    mcg.find("target").text = "p"
    mcg.find("nucleon_configuration_from_file").text = "0"
    pp = XML_DIR / "pp.xml"
    tree.write(pp)
    return {"OO": oo, "pp": pp}


def wtext(windows):
    return ",".join(f"{lo:g}-{hi:g}" for lo, hi in windows)


def jobs(names):
    xmls = write_xmls()
    allm = samples()
    out = []
    for name in names:
        sysn, b, njobs, windows, events, eta = allm[name]
        k = list(allm).index(name)
        events -= events % len(windows)
        for j in range(njobs):
            seed = 979000 + j if name == "pp_low" else 980000 + 1000 * k + j
            cmd = [sys.executable, str(C.DRIVER), "--vacuum", "--user-xml", str(xmls[sysn]),
                   "--events", str(events), "--seed", str(seed),
                   "--pthat-bins", wtext(windows), "--eloss-threads", "0",
                   "--outdir", str(CLOSURE / name), "--max-skipped", "30"]
            if not eta:
                cmd += ["--eta-max", str(C.ETA_MAX)]
            else:
                cmd += ["--store-hadronization-input"]
            if b is not None:
                cmd += ["--b-min", str(b[0]), "--b-max", str(b[1])]
            for line in C.FIXED_HADRONIZATION:
                cmd += ["--hadronization-line", line]
            out.append((name, seed, events, cmd))
    return out


def done(name, seed, events):
    for js in (CLOSURE / name).glob(f"*_seed{seed:04d}.json"):
        try:
            d = json.loads(js.read_text())
            return d.get("events_written", 0) + d.get("events_skipped", 0) == events
        except (OSError, ValueError):
            return False
    return False


def run(names, nproc):
    todo = [t for t in jobs(names) if not done(*t[:3])]
    print(f"closure_oo.py: {len(todo)} jobs to run, {nproc} at a time -> {CLOSURE}", flush=True)
    env = dict(os.environ, OMP_NUM_THREADS="1")
    t0 = time.time()

    def one(t):
        name, seed, events, cmd = t
        d = CLOSURE / name
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"run_seed{seed}.log", "w") as log:
            r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                               cwd=str(C.DRIVER.parent))
        return name, seed, r.returncode

    n_fail = 0
    with ThreadPoolExecutor(max_workers=nproc) as ex:
        for k, (name, seed, rc) in enumerate(ex.map(one, todo), 1):
            n_fail += rc != 0
            if rc or k % 10 == 0 or k == len(todo):
                print(f"[{k}/{len(todo)}] {name} seed {seed}: {'ok' if rc == 0 else f'FAILED ({rc})'}"
                      f", {time.time() - t0:.0f} s", flush=True)
    return 1 if n_fail else 0


# ── analysis ─────────────────────────────────────────────────────────────────────────────
UNIT_KEYS = ("glauber_b", "glauber_ncoll", "glauber_npart", "glauber_nstrings", "pthat",
             "pthat_bin", "parton_pt_lead", "parton_y_lead", "remnant_E_proj", "remnant_E_targ",
             "remnant_pz_proj", "remnant_pz_targ", "isr_sub_E_total", "strings_E_total")


def charged_and_events(name):
    """Charged-hadron d2sigma/(dpT deta), |eta| < 1, per window sums; the per-event units;
    the jobs' skipped events and 3dMCGlauber's 'Can not find x > x_hard' count."""
    import hdf5plugin  # noqa: F401
    import h5py
    import observables as O
    charged = O._charged_ids()
    files = sorted(glob.glob(str(CLOSURE / name / "*_hadrons_jet_frag.h5")))
    nwin = len(C.PTHAT_WINDOWS)
    w = np.zeros((nwin, len(CH_EDGES) - 1))
    w2 = np.zeros_like(w)
    sig_w, nacc, nev = np.zeros(nwin), np.zeros(nwin), np.zeros(nwin)
    units = {k: [] for k in UNIT_KEYS}
    for f in files:
        fd = O.read_file(f)
        pt, _, eta, _ = O._kin(fd["p"])
        hw = np.repeat(fd["win"], fd["counts"])
        sel = np.isin(np.abs(fd["pid"]), charged) & (np.abs(eta) < 1.0)
        a, a2 = O._hist_by_window(pt[sel], np.full(sel.sum(), 0.5), hw[sel], nwin, CH_EDGES)
        w += a
        w2 += a2
        sig_w += fd["n_acc"] * fd["sigma"]
        nacc += fd["n_acc"]
        nev += np.bincount(fd["win"], minlength=nwin)
        with h5py.File(f) as h:
            n = int(h.attrs["nunits_written"])
            for k in UNIT_KEYS:
                units[k].append(h[f"units/{k}"][:n] if f"units/{k}" in h else np.full(n, np.nan))
    sigma = sig_w / np.maximum(nacc, 1)
    scale = sigma / np.maximum(nev, 1)
    dpt = np.diff(CH_EDGES)
    ch = (scale[:, None] * w).sum(0) / dpt
    ch_err = np.sqrt((scale[:, None] ** 2 * w2).sum(0)) / dpt
    units = {k: np.concatenate(v) if v else np.array([]) for k, v in units.items()}
    skipped, nx, req = 0, 0, 0
    for js in (CLOSURE / name).glob("*_seed*.json"):
        d = json.loads(js.read_text())
        skipped += d.get("events_skipped", 0)
        req += d.get("events_written", 0) + d.get("events_skipped", 0)
    for log in (CLOSURE / name).glob("run_seed*.log"):
        with open(log, errors="replace") as fh:
            nx += sum(1 for ln in fh if "Can not find x > x_hard" in ln)
    return dict(ch=ch, ch_err=ch_err, sigma=sigma, n_events=nev, units=units,
                skipped=skipped, requested=req, xhard_msgs=nx, n_files=len(files))


def eta_sample(name):
    """dN/deta (all charged, all pT) of the hard event per window; parton y of iColored's
    input; per-event histograms, so the errors are the event-to-event spread (hstat.event_mean)."""
    import hdf5plugin  # noqa: F401
    import h5py
    import hstat as H
    import observables as O
    charged = O._charged_ids()
    edges = np.arange(-8, 8.0001, 0.25)
    nb = len(edges) - 1
    out = {}
    files = sorted(glob.glob(str(CLOSURE / name / "*_hadrons_jet_frag.h5")))

    def per_event(x, ev, n):
        ib = np.digitize(x, edges) - 1
        ok = (ib >= 0) & (ib < nb)
        a = np.zeros((n, nb))
        np.add.at(a, (ev[ok], ib[ok]), 1.0)
        return a
    for k, wdw in enumerate(ETA_WINDOWS):
        hs, ys = [], []
        for f in files:
            fd = O.read_file(f)
            m_ev = fd["win"] == k
            nev = len(fd["counts"])
            ev = np.repeat(np.arange(nev), fd["counts"])
            _, _, eta, _ = O._kin(fd["p"])
            sel = np.isin(np.abs(fd["pid"]), charged)
            hs.append(per_event(eta[sel], ev[sel], nev)[m_ev])
            with h5py.File(f) as h:
                d = h["hadronization_input/data"][:]
                off = h["hadronization_input/offsets"][:]
            pev = np.repeat(np.arange(len(off) - 1), np.diff(off))
            E, pz = d[:, 3], d[:, 6]
            y = 0.5 * np.log(np.maximum(E + pz, 1e-9) / np.maximum(E - pz, 1e-9))
            ys.append(per_event(y, pev, len(off) - 1)[m_ev[:len(off) - 1]])
        hs, ys = np.concatenate(hs), np.concatenate(ys)
        (dn, dn_e), (py, py_e) = H.event_mean(hs), H.event_mean(ys)
        bw = np.diff(edges)
        out[wtext([wdw])] = {"dndeta": (dn / bw).tolist(), "dndeta_err": (dn_e / bw).tolist(),
                             "parton_dndy": (py / bw).tolist(), "parton_dndy_err": (py_e / bw).tolist(),
                             "n_events": int(len(hs))}
    return {"edges": edges.tolist(), **out}


def pp_combined(low_files):
    """R3 with the low-window events: the p+p run and pp_low together (observables.compute
    matches the windows by their edges), and NSPLIT independent subsets (files round-robin)
    for the run-to-run scatter of the chi2."""
    import observables as O
    base = sorted(glob.glob(str(CLOSURE / "pp" / "*_hadrons_jet_frag.h5")))
    obs = C.PROD_DIR / "obs"
    O.process("closure_pp_all", base + low_files, obs / "closure_pp_all.npz")
    for i in range(NSPLIT):
        O.process(f"closure_pp_split{i}", base[i::NSPLIT] + low_files[i::NSPLIT],
                  obs / f"closure_pp_split{i}.npz")
    return {"n_files_base": len(base), "n_files_low": len(low_files), "n_split": NSPLIT}


def analyse_one(name):
    import observables as O
    files = sorted(glob.glob(str(CLOSURE / name / "*_hadrons_jet_frag.h5")))
    if not files:
        return name, None
    if name.endswith("_eta"):
        return name, {"eta": eta_sample(name)}
    if name == "pp_low":
        return name, {"combined": pp_combined(files)}
    out, n = O.process(f"closure_{name}", files, C.PROD_DIR / "obs" / f"closure_{name}.npz")
    return name, charged_and_events(name)


def analyse(names, nproc):
    res = {}
    with ProcessPoolExecutor(max_workers=nproc) as ex:
        for name, r in ex.map(analyse_one, names):
            if r is None:
                print(f"{name}: no files")
                continue
            res[name] = r
            print(f"{name}: done", flush=True)
    summary = {"tune_point": TUNE_POINT, "xml": str(OO_XML), "b_classes": B_CLASSES,
               "ch_edges": CH_EDGES.tolist(), "pthat_windows": [list(w) for w in C.PTHAT_WINDOWS],
               "samples": {}}
    arrays = {}
    rng = np.random.default_rng(1)
    for name, r in res.items():
        if "eta" in r:
            summary["samples"][name] = {"eta": r["eta"]}
            continue
        if "combined" in r:
            summary["samples"][name] = r["combined"]
            continue
        u = r["units"]
        s = {"ch": r["ch"].tolist(), "ch_err": r["ch_err"].tolist(), "sigma": r["sigma"].tolist(),
             "n_events": r["n_events"].tolist(), "skipped": r["skipped"],
             "requested": r["requested"], "xhard_msgs": r["xhard_msgs"], "n_files": r["n_files"]}
        for k in ("glauber_b", "glauber_ncoll", "glauber_npart", "glauber_nstrings",
                  "remnant_E_proj", "remnant_E_targ", "isr_sub_E_total", "strings_E_total"):
            if len(u[k]):
                s["mean_" + k] = float(np.nanmean(u[k]))
        summary["samples"][name] = s
        n = len(u["pthat"])
        pick = rng.choice(n, size=min(60000, n), replace=False) if n else []
        for k in UNIT_KEYS:
            arrays[f"{name}__{k}"] = np.asarray(u[k])[pick].astype(np.float32)
    path = HERE / "results" / "closure_oo.json"
    path.parent.mkdir(exist_ok=True)
    if path.exists():                                   # keep samples analysed earlier
        old = json.loads(path.read_text())
        summary["samples"] = {**old.get("samples", {}), **summary["samples"]}
    path.write_text(json.dumps(summary))
    npz = HERE / "results" / "closure_oo_units.npz"
    if npz.exists():
        old = dict(np.load(npz))
        old.update(arrays)
        arrays = old
    np.savez(npz, **arrays)
    print("wrote", path, npz)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=("run", "analyse"))
    p.add_argument("names", nargs="*", default=None)
    p.add_argument("-j", type=int, default=os.cpu_count())
    a = p.parse_args(argv)
    names = a.names or list(samples())
    bad = [n for n in names if n not in samples()]
    if bad:
        sys.exit(f"closure_oo.py: unknown samples {bad}; known {list(samples())}")
    return run(names, a.j) if a.step == "run" else analyse(names, a.j)


if __name__ == "__main__":
    sys.exit(main())
