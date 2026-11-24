#!/usr/bin/env python3
"""
analysis/OO-RHIC/oo_bias.py -- step 4 of the plan forward (R4b, the full test): the centrality
bias of O+O jet events when the classes are built as the experiments build them, from the
charged-particle multiplicity of full events (hydro + particlization) in the estimators'
acceptance, with the jet in vacuum (no quenching): any deviation from N_coll scaling per class is
then selection bias.

    python oo_bias.py run mb -j 8 --mps           # stage 1: the minimum-bias reference
    python oo_bias.py run jets jets_noveto -j 8 --mps
    python oo_bias.py analyse                     # -> results/oo_bias.json
    jupyter nbconvert --to notebook --execute oo_centrality_bias.ipynb --output results/oo_centrality_bias_executed.ipynb

The bulk screening (plan A of the p+p / O+O bulk tune): minimum-bias stand-ins as `mb`, for p+p
and O+O, with the 3dMCGlauber sets of BULK_SETS (map, pp2407, m3; xsdef on request), 10k events
per sample in 250-event jobs, the samples interleaved so that an early stop leaves them even:

    python oo_bias.py run screen --list                       # the jobs and their commands
    python oo_bias.py run screen -j 8 --mps --omp 3 --limit 8 # a timing test (8 jobs)
    python oo_bias.py run screen -j 8 --mps --omp 3 --shard 0/2   # GPU 0; 1/2 on GPU 1
    python oo_bias.py run screen --events 5000 ...            # 5k per sample; 10k later adds
    python oo_bulk.py analyse --sample oo_pp2407              # O+O against STAR
On the 2 x RTX 3090 box: oo_bias_docker.sh (one container per GPU, the benchmarked settings).

Samples (O+O 200 GeV, config/OO_MCGlauber_MUSIC_isr_rhictune.xml, b in [0, 10] fm as 3dMCGlauber
samples it, run_prod_oo_isr.py with MUSIC and iSS in the job: --bulk-hadrons 1 (one sample per
event: the physical fluctuations), --vacuum-jet, charged hadrons with |eta| < 5.2 stored):
  mb           the minimum-bias reference: --no-subtraction --hard-pair-string-veto 0, only the
               bulk hadrons counted.  With neither the subtraction nor the veto the ISR chain makes
               the same strings as minimum-bias events at the same N_coll and samples the same
               geometry (closure, centrality_proxy.py: N_coll, N_part, b distributions identical,
               string proxies within 1-2 %), so these are minimum-bias soft events with exactly the
               MUSIC and iSS settings of the jet events (the hard event's hadrons are ignored)
  jets         jet events with the production settings (subtraction, veto on): bulk + the hard
               event's hadrons counted
  jets_noveto  the same with the veto off (--hard-pair-string-veto 0)
  jets_quench  as jets, the same seeds, but without --vacuum-jet: the jet sees the medium (LBT,
               Matter in medium); R_CP(jets_quench) / R_CP(jets) per class is the quenching.
               Stores all hadrons (no --charged), for full jets as sPHENIX's calorimeter jets;
               the other samples store charged hadrons (estimator counts are charged in all)
Jet events in the pTHat windows 9-11, 18-23, 30-40 and 40-55 GeV, each weighted with its N_coll
(the generator puts exactly one hard scattering in every event).

Estimators (charged particles of the event): STAR's EPD 2.1 < |eta| < 5.1 and mid-rapidity |eta|
< 0.5 / < 1 (pT > 0.2 GeV), sPHENIX's MBD 3.51 < |eta| < 4.61 (both sides; >= 1 particle in each
arm is the minimum-bias coincidence).  Classes are percentiles of the minimum-bias multiplicity
(random tie-break); per class the experiment-style R_AA^vac = (f_jet,c / f_MB,c) <N_coll>_MB /
<N_coll>_MB,c, the bias factor <N_coll>_jet,c / <N_coll>_MB,c and R_CP.  Needs X-SCAPE
oo_rhic_run (the 3dMCGlauber switch) and the driver's --vacuum-jet.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C  # noqa: E402
import hstat as H  # noqa: E402

OUT = Path(os.environ.get("OO_RHIC_OO_PROD", Path.home() / "prod" / "OO-RHIC_OO")) / "bias"
OO_XML = C.EXAMPLE / "config" / "OO_MCGlauber_MUSIC_isr_rhictune.xml"
B_MAX = 10.0
JET_WINDOWS = [(9, 11), (18, 23), (30, 40), (40, 55)]
#: name: (events, events per job, windows, extra driver options, count the hard event's hadrons)
SAMPLES = {
    "mb": (50000, 500, [(9, 11)], ["--no-subtraction", "--hard-pair-string-veto", "0"], False),
    "jets": (50000, 500, JET_WINDOWS, [], True),
    "jets_noveto": (30000, 500, JET_WINDOWS, ["--hard-pair-string-veto", "0"], True),
    "jets_quench": (50000, 500, JET_WINDOWS, [], True),
    # validation of the hole/partner treatment (negative hadrons): the jet in the medium with
    # its final partons (holes) and iColored's Pythia input (partners) stored
    "jets_holes": (4500, 450, JET_WINDOWS[1:], ["--store-partons", "--store-hadronization-input"], True),
}
#: jets_quench reuses the seeds of jets (same geometry and hard scatterings, the jet now sees the medium)
SEED_BASE = {"mb": 700000, "jets": 710000, "jets_noveto": 720000, "jets_quench": 710000, "jets_holes": 730000}

# ── the bulk screening (plan A of the p+p / O+O bulk tune on the tuning page) ─────────────
#: 3dMCGlauber sets, as driver options on top of OO_XML.  The screening ran when OO_XML's own set
#: was the JETSCAPE MAP of 2601.17234 with shadowing 0; since 2026-11-24 OO_XML carries the
#: adopted set g_v045_n215, so the MAP is given explicitly (MAP below) and every set below names
#: all the parameters it differs from the XML in -- each sample runs as it did in the screening.
#: The input-file defaults of 3dMCGlauber are already the 2211.16376 method-3 values; they are
#: given explicitly so that the files record them.
MAP = [f"--xml-set=IS/MCGlauber/{k}={v}" for k, v in (
    ("ylossParam4At2", 1.61), ("ylossParam4At4", 1.685), ("ylossParam4At6", 1.685),
    ("ylossParam4At10", 2.95), ("ylossParam4var", 0.682), ("shadowing_factor", 0.0),
    ("BG", 3.96))]
BULK_SETS = {
    "map": MAP,
    # 2407.17443 p+p (Figs. 3-7): method 4, nodes 1.6 / 2.15 / 2.15 / 2.15, variance 0.5,
    # shadowing 0.45, BG 5 (its mcglauber.input); the rest as the MAP
    "pp2407": [f"--xml-set=IS/MCGlauber/{k}={v}" for k, v in (
        ("ylossParam4At2", 1.6), ("ylossParam4At4", 2.15), ("ylossParam4At6", 2.15),
        ("ylossParam4At10", 2.15), ("ylossParam4var", 0.5), ("shadowing_factor", 0.45),
        ("BG", 5.0))],
    # method 3 (2211.16376: A 1.32, alpha1 1.8, alpha2 0.34, variance 0.4), shadowing 0.45,
    # alpha_rem 0.5; the rest (BG, tau_form, hydro) as the MAP
    "m3": MAP + [f"--mcglauber-param={k}={v}" for k, v in (
        ("rapidity_loss_method", 3), ("yloss_param_slope", 1.32), ("yloss_param_alpha1", 1.8),
        ("yloss_param_alpha2", 0.34), ("yloss_param_fluct_var_RHIC", 0.4))]
        + [f"--xml-set=IS/MCGlauber/{k}={v}" for k, v in (
            ("shadowing_factor", 0.45), ("remnant_energy_loss_fraction", 0.5))],
    # X-SCAPE's default set (jetscape_main.xml): not in the default screening
    "xsdef": [f"--xml-set=IS/MCGlauber/{k}={v}" for k, v in (
        ("ylossParam4At2", 1.6), ("ylossParam4At4", 2.15), ("ylossParam4At6", 2.45),
        ("ylossParam4At10", 2.95), ("ylossParam4var", 0.5), ("shadowing_factor", 0.6),
        ("BG", 5.0))],
}
# follow-up of the screening (2026-11-23): pp2407 with a smaller yloss variance; the screening
# found its p+p P(N_ch) ~10 % too wide and its O+O p(N_ch) tail (N_ch > 120) too long
BULK_SETS["pp2407v03"] = [o for o in BULK_SETS["pp2407"] if "ylossParam4var" not in o] + [
    "--xml-set=IS/MCGlauber/ylossParam4var=0.3"]
# grid around it (2026-11-23): variance 0.4 / 0.45 x nodes At4 = At6 = At10 = 2.15 / 2.25; var 0.3
# fixed the p+p P(n) widths but lowered the yield 8-9 % and made the O+O tail too short
for _v in (0.4, 0.45):
    for _n in (2.15, 2.25):
        BULK_SETS[f"g_v{_v:g}_n{_n:g}".replace(".", "")] = [
            o for o in BULK_SETS["pp2407"]
            if not any(k in o for k in ("ylossParam4var", "At4=", "At6=", "At10="))] + [
            f"--xml-set=IS/MCGlauber/ylossParam4var={_v}"] + [
            f"--xml-set=IS/MCGlauber/ylossParam4At{y}={_n}" for y in (4, 6, 10)]
# remnant scan (2026-11-23) on the grid's best set g_v045_n215: the remnant string runs from the
# remnant's rapidity y_rem to y_rem - alpha_rem * y_loss (3dMCGlauber Glauber.cpp, string mode -4),
# so a larger alpha_rem moves remnant energy from |eta| > 3.5 toward mid-rapidity; all sets were
# above PHOBOS's p+p dN/deta beyond |eta| 3.5 and below it at 1.5-3 (XML value 0.536)
REM_BASE = "g_v045_n215"
for _a in (0.4, 0.65, 0.8, 1.0):
    BULK_SETS[f"rem{round(_a * 100):03d}"] = BULK_SETS[REM_BASE] + [
        f"--xml-set=IS/MCGlauber/remnant_energy_loss_fraction={_a}"]
REM_SETS = tuple(s for s in BULK_SETS if s.startswith("rem"))
GRID_SETS = tuple(s for s in BULK_SETS if s.startswith("g_"))
#: the sets run by default ("screen" on the command line): MAP, 2407.17443 p+p, method 3
SCREEN_SETS = ("map", "pp2407", "m3")
#: p+p with the O+O XML's chain (as make_design.py --hydro): one nucleon each, no O16
#: configurations, b <= 3 fm (as the p+p hard tune; the NN overlap ends well before)
PP = ["--xml-set=IS/MCGlauber/projectile=p", "--xml-set=IS/MCGlauber/target=p",
      "--xml-set=IS/MCGlauber/nucleon_configuration_from_file=0", "--b-max", "3"]
#: d+Au with the same chain (dau_bulk.py: the check against the MAP's own calibration data,
#: PHENIX/BRAHMS dN/deta): deuteron (Hulthen) on Au (Woods-Saxon, no configuration file), b < 12 fm
#: (the participants stay near the Au surface, inside MUSIC's +-10 fm box); target moves to -z
DAU = ["--xml-set=IS/MCGlauber/projectile=d", "--xml-set=IS/MCGlauber/target=Au",
       "--xml-set=IS/MCGlauber/nucleon_configuration_from_file=0", "--b-max", "12"]
#: minimum-bias stand-in as `mb` (no subtraction, veto off, bulk hadrons only)
MB_OPTS = ["--no-subtraction", "--hard-pair-string-veto", "0"]
SCREEN_EVENTS, SCREEN_PER_JOB = 10000, 250
for _s, _opts in BULK_SETS.items():
    SAMPLES[f"oo_{_s}"] = (SCREEN_EVENTS, SCREEN_PER_JOB, [(9, 11)], MB_OPTS + _opts, False)
    SAMPLES[f"pp_{_s}"] = (SCREEN_EVENTS, SCREEN_PER_JOB, [(9, 11)], MB_OPTS + PP + _opts, False)
    # every set uses the same seeds per system: the same geometries and hard scatterings as
    # far as the set does not change them, so set-to-set differences are less noisy
    # (jets_holes also starts at 730000: a separate validation sample with other options, run
    # elsewhere; kept as both have run)
    SEED_BASE[f"oo_{_s}"] = 730000
    SEED_BASE[f"pp_{_s}"] = 740000
    SAMPLES[f"dau_{_s}"] = (SCREEN_EVENTS, SCREEN_PER_JOB, [(9, 11)], MB_OPTS + DAU + _opts, False)
    SEED_BASE[f"dau_{_s}"] = 750000
#: name groups on the command line
GROUPS = {"screen": [f"{sys_}_{s}" for s in SCREEN_SETS for sys_ in ("pp", "oo")],
          "var": ["pp_pp2407v03", "oo_pp2407v03"],
          "grid": [f"{sys_}_{s}" for s in GRID_SETS for sys_ in ("pp", "oo")],
          "rem": [f"pp_{s}" for s in REM_SETS],          # step 1: p+p only (the forward shape)
          "bias": ["mb", "jets", "jets_noveto", "jets_quench"],
          "holes": ["jets_holes"]}

#: samples whose jet sees the medium (LBT, Matter in medium); all others run with --vacuum-jet
QUENCHED = {"jets_quench", "jets_holes"}
#: samples that store all hadrons (full jets as sPHENIX's calorimeter jets); all others --charged.
#: The estimator counts use charged hadrons in either case (_units selects them by species).
ALL_HADRONS = {"jets_quench", "jets_holes"}
EPD, MBD = (2.1, 5.1), (3.51, 4.61)
STAR_CLASSES = [(0, 10), (10, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
SPH_CLASSES = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]


def jobs(name, events=None):
    """The jobs of a sample: (name, seed, events, command).  The sample's extra options come
    last, so they win over the defaults here (--b-max for p+p).  events: another total for
    the sample (the first jobs of the same seed list, so a larger total extends a smaller)."""
    total, per_job, windows, extra, _ = SAMPLES[name]
    total = events or total
    per_job -= per_job % len(windows)
    out = []
    for j in range(total // per_job):
        seed = SEED_BASE[name] + j
        cmd = [sys.executable, str(C.DRIVER), "--user-xml", str(OO_XML), "--events", str(per_job),
               "--seed", str(seed), "--pthat-bins", ",".join(f"{lo}-{hi}" for lo, hi in windows),
               "--b-min", "0", "--b-max", f"{B_MAX}", *([] if name in QUENCHED else ["--vacuum-jet"]), "--bulk-hadrons", "1",
               "--no-particlize", "--no-hydro-file", "--eta-max", "5.2",
               *([] if name in ALL_HADRONS else ["--charged"]),
               "--outdir", str(OUT / name), "--max-skipped", "30", *extra]
        out.append((name, seed, per_job, cmd))
    return out


def done(name, seed, events):
    for js in (OUT / name).glob(f"*_seed{seed:04d}.json"):
        try:
            d = json.loads(js.read_text())
            return d.get("events_written", 0) + d.get("events_skipped", 0) == events
        except (OSError, ValueError):
            return False
    return False


def job_list(names, events=None, shard=(0, 1), limit=None):
    """The pending jobs, the samples interleaved (job 0 of every sample, then job 1, ...), so
    that a run stopped early has about the same number of events in each sample.  Shard (i, n)
    takes job k of every sample with k % n == i: disjoint shards (one container per GPU, or
    per machine) that never run the same job twice, each with every sample."""
    per = [jobs(n, events) for n in names]
    i, n = shard
    mine = [p[k] for k in range(i, max(map(len, per), default=0), n) for p in per if k < len(p)]
    todo = [t for t in mine if not done(*t[:3])]
    return todo[:limit] if limit else todo


def run(names, nproc, mps, omp, events=None, shard=(0, 1), limit=None):
    todo = job_list(names, events, shard, limit)
    print(f"oo_bias.py: {len(todo)} jobs" + (f" (shard {shard[0]}/{shard[1]})" if shard[1] > 1 else "")
          + f", {nproc} at a time, OMP_NUM_THREADS {omp}"
          + f", MUSIC_CUDA_SYNC {os.environ.get('MUSIC_CUDA_SYNC', 'unset')} -> {OUT}", flush=True)
    env = dict(os.environ, OMP_NUM_THREADS=str(omp))
    mps_dir = None
    if mps:
        mps_dir = os.path.join(os.environ.get("XDG_RUNTIME_DIR", "/tmp"), f"xscape-mps.{os.getpid()}")
        os.makedirs(os.path.join(mps_dir, "pipe"), exist_ok=True)
        os.makedirs(os.path.join(mps_dir, "log"), exist_ok=True)
        env.update(CUDA_MPS_PIPE_DIRECTORY=os.path.join(mps_dir, "pipe"),
                   CUDA_MPS_LOG_DIRECTORY=os.path.join(mps_dir, "log"))
        if subprocess.run(["nvidia-cuda-mps-control", "-d"], env=env).returncode != 0:
            sys.exit("oo_bias.py: could not start the CUDA MPS daemon")
    t0 = time.time()

    def one(t):
        name, seed, events, cmd = t
        (OUT / name).mkdir(parents=True, exist_ok=True)
        with open(OUT / name / f"run_seed{seed}.log", "w") as log:
            r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                               cwd=str(C.DRIVER.parent))
        return name, seed, r.returncode

    nfail = 0
    try:
        with ThreadPoolExecutor(max_workers=nproc) as ex:
            for k, (name, seed, rc) in enumerate(ex.map(one, todo), 1):
                nfail += rc != 0
                print(f"[{k}/{len(todo)}] {name} seed {seed}: {'ok' if rc == 0 else f'FAILED ({rc})'}"
                      f", {time.time() - t0:.0f} s", flush=True)
    finally:
        if mps_dir:
            subprocess.run(["nvidia-cuda-mps-control"], input=b"quit\n", env=env)
    return 1 if nfail else 0


# ── analysis ────────────────────────────────────────────────────────────────────────────────
_CHARGED = None


def charged_ids():
    """|PDG id| of the charged final-state particles (jetscape.hadrons_h5.CHARGED)."""
    global _CHARGED
    if _CHARGED is None:
        sys.path.insert(0, str(C.EXAMPLE.parent / "python"))
        from jetscape.hadrons_h5 import CHARGED
        _CHARGED = np.asarray(sorted({abs(int(x)) for x in CHARGED}))
    return _CHARGED


def _units(f, n, charged=True, with_pid=False):
    """Momenta (E, px, py, pz) and particles per unit of the first n units.  charged=True keeps the
    charged particles only (files written with --charged hold nothing else; files with all hadrons
    are filtered by species), so the estimator counts mean the same for both kinds of file."""
    so = f["hadrons/sample_offsets"][:]
    uo = f["hadrons/unit_offsets"][: n + 1]
    stop = int(so[uo[-1]])
    p = f["hadrons/p"][:stop].astype(np.float64)
    counts = np.diff(so[uo])
    pid = f["hadrons/pid"][:stop] if (with_pid or (charged and not bool(f.attrs.get("charged_only", False)))) else None
    if charged and not bool(f.attrs.get("charged_only", False)):
        keep = np.isin(np.abs(pid), charged_ids())
        ev = np.repeat(np.arange(n), counts)
        p, pid, counts = p[keep], pid[keep], np.bincount(ev[keep], minlength=n)
    return (p, counts, pid) if with_pid else (p, counts)


def counts_of(p, counts):
    """Per event: charged particles in the estimator windows (the files hold charged hadrons)."""
    E, px, py, pz = p.T
    pt = np.hypot(px, py)
    pabs = np.sqrt(px * px + py * py + pz * pz)
    eta = 0.5 * np.log(np.clip(pabs + pz, 1e-12, None) / np.clip(pabs - pz, 1e-12, None))
    ev = np.repeat(np.arange(len(counts)), counts)
    n = len(counts)
    sel = {"epd": (np.abs(eta) > EPD[0]) & (np.abs(eta) < EPD[1]),
           "mbd_neg": (-eta > MBD[0]) & (-eta < MBD[1]),
           "mbd_pos": (eta > MBD[0]) & (eta < MBD[1]),
           "mid05": (np.abs(eta) < 0.5) & (pt > 0.2),
           "mid10": (np.abs(eta) < 1.0) & (pt > 0.2)}
    return {k: np.bincount(ev[m], minlength=n) for k, m in sel.items()}


def load(name):
    """Per event: N_coll, N_part, b, the pTHat window, has_medium, and the estimator counts
    (bulk, plus the hard event's hadrons where the sample counts them)."""
    import hdf5plugin  # noqa: F401
    import h5py
    with_jet = SAMPLES[name][4]
    rec = {}
    for jf in sorted(glob.glob(str(OUT / name / "*_hadrons_jet_frag.h5"))):
        bf = jf.replace("_hadrons_jet_frag.h5", "_hadrons_bulk_bg.h5")
        if not os.path.exists(jf.replace("_hadrons_jet_frag.h5", ".json")):
            continue  # job still running (the summary is written at the end)
        with h5py.File(jf) as h:
            n = int(h.attrs["nunits_written"])
            u = {k: h[f"units/{k}"][:n] for k in ("glauber_ncoll", "glauber_npart", "glauber_b",
                                                   "pthat_bin", "has_medium", "glauber_nstrings")}
            pj, cj = _units(h, n)
        tot = {k: np.zeros(n, dtype=np.int64) for k in ("epd", "mbd_neg", "mbd_pos", "mid05", "mid10")}
        if os.path.exists(bf):
            with h5py.File(bf) as h:
                if int(h.attrs["nunits_written"]) < n:
                    raise ValueError(f"{bf}: fewer units than {jf}")
                pb, cb = _units(h, n)
            for k, v in counts_of(pb, cb).items():
                tot[k] += v
        if with_jet:
            for k, v in counts_of(pj, cj).items():
                tot[k] += v
        for k, v in {**u, **{f"n_{k}": v for k, v in tot.items()}}.items():
            rec.setdefault(k, []).append(v)
    rec = {k: np.concatenate(v) for k, v in rec.items()}
    if rec:
        rec["n_mbd"] = rec["n_mbd_neg"] + rec["n_mbd_pos"]
    return rec


def class_analysis(mb, jet, est, classes, rng, mask=None):
    """Percentile classes of the minimum-bias multiplicity (all events, random tie-break) applied
    to the N_coll-weighted jet events."""
    nc = len(classes)
    xm = mb[f"n_{est}"] + rng.random(len(mb["glauber_ncoll"]))
    cuts = np.percentile(xm, [100 - hi for _, hi in classes])

    def cls(v):
        c = np.full(len(v), nc - 1)
        for k in range(nc - 1, -1, -1):
            c[v >= cuts[k]] = k
        return c
    cm = cls(xm)
    m = np.ones(len(jet["glauber_ncoll"]), bool) if mask is None else mask
    cj = cls(jet[f"n_{est}"][m] + rng.random(int(m.sum())))
    w = jet["glauber_ncoll"][m].astype(float)
    fmb = np.array([np.mean(cm == k) for k in range(nc)])
    nmb, nmb_e = H.class_mean(cm, mb["glauber_ncoll"], nc)
    fj = np.array([w[cj == k].sum() for k in range(nc)]) / w.sum()
    njet, njet_e = H.class_mean(cj, jet["glauber_ncoll"][m], nc, w)
    # R_AA^vac = (f_jet,c / f_MB,c) <N_coll>_MB / <N_coll>_MB,c with the statistical errors of both
    # samples (hstat.raa_vac); R_CP from the same class sums
    rv = H.raa_vac(cm, mb["glauber_ncoll"], cj, w, nc)
    bias, bias_e = H.ratio(njet, njet_e, nmb, nmb_e)
    out = {"cuts": cuts.tolist(), "frac_mb": fmb.tolist(), "ncoll_mb": nmb.tolist(), "ncoll_mb_err": nmb_e.tolist(),
           "frac_jet": fj.tolist(), "ncoll_jet": njet.tolist(), "ncoll_jet_err": njet_e.tolist(),
           "bias": bias.tolist(), "bias_err": bias_e.tolist(),
           "raa_vac": rv["raa"].tolist(), "raa_err": rv["err"].tolist(), "n_jet": int(m.sum()), "rcp": {}}
    for a, b in ((0, nc - 2), (0, nc - 3), (1, nc - 2)):
        out["rcp"][f"{classes[a][0]}-{classes[a][1]}/{classes[b][0]}-{classes[b][1]}"] = H.rcp(rv, a, b)
    return out


def analyse():
    rng = np.random.default_rng(11)
    mb = load("mb")
    if not mb:
        sys.exit("oo_bias.py: no minimum-bias events yet")
    res = {"windows": JET_WINDOWS, "star_classes": STAR_CLASSES, "sph_classes": SPH_CLASSES,
           "n_mb": int(len(mb["glauber_ncoll"])), "mb": {}, "jets": {}}
    coin_mb = (mb["n_mbd_neg"] > 0) & (mb["n_mbd_pos"] > 0)
    res["mb"] = {"ncoll_mean": float(mb["glauber_ncoll"].mean()),
                 "eps_mbd_coincidence": float(coin_mb.mean()), "eps_mbd_coincidence_err": H.fraction(coin_mb)[1],
                 "has_medium": float(mb["has_medium"].mean()),
                 "mean_counts": {k: float(mb[f"n_{k}"].mean()) for k in ("epd", "mbd", "mid05", "mid10")}}
    print(f"minimum bias: {res['n_mb']} events, <N_coll> {res['mb']['ncoll_mean']:.2f}, MBD coincidence "
          f"{res['mb']['eps_mbd_coincidence']:.3f} (sPHENIX 0.914), with medium {res['mb']['has_medium']:.3f}")
    for name in ("jets", "jets_noveto", "jets_quench"):
        jet = load(name)
        if not jet:
            continue
        w = jet["glauber_ncoll"].astype(float)
        coin = (jet["n_mbd_neg"] > 0) & (jet["n_mbd_pos"] > 0)
        r = {"n": int(len(w)), "eps_mbd_coincidence_weighted": float(np.average(coin, weights=w)),
             "eps_mbd_coincidence_weighted_err": H.fraction(coin, w)[1],
             "has_medium_weighted": float(np.average(jet["has_medium"], weights=w)), "all": {}, "per_window": {}}
        for est, classes in (("epd", STAR_CLASSES), ("mid10", STAR_CLASSES), ("mid05", STAR_CLASSES),
                             ("mbd", SPH_CLASSES), ("epd_sph", SPH_CLASSES)):
            e = "epd" if est == "epd_sph" else est
            r["all"][est] = class_analysis(mb, jet, e, classes, rng)
            for k, wdw in enumerate(JET_WINDOWS):
                r["per_window"].setdefault(f"{wdw[0]}-{wdw[1]}", {})[est] = class_analysis(
                    mb, jet, e, classes, rng, mask=jet["pthat_bin"] == k)
        res["jets"][name] = r
        print(f"{name}: {r['n']} events, MBD coincidence (N_coll-weighted) {r['eps_mbd_coincidence_weighted']:.3f}")
        for est in ("epd", "mid10", "mbd"):
            a = r["all"][est]
            labs = STAR_CLASSES if est != "mbd" else SPH_CLASSES
            print(f"   {est:5s} R_AA^vac " + " ".join(f"{lo}-{hi}% {v:.2f}" for (lo, hi), v in zip(labs, a["raa_vac"])))
        m = r["all"]["mbd"]["raa_vac"]
        rcp = lambda a_, b_: a_ / b_ if b_ > 0 else float("nan")
        print(f"   sPHENIX classes (MBD): R_CP 0-20/60-80 {rcp(m[0], m[3]):.2f}, 0-20/40-60 {rcp(m[0], m[2]):.2f}")
    path = Path(os.environ.get("OO_BIAS_JSON", HERE / "results" / "oo_bias.json"))
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(res, indent=1))
    print("wrote", path)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=("run", "analyse"))
    p.add_argument("names", nargs="*", default=list(SAMPLES))
    p.add_argument("-j", type=int, default=8)
    p.add_argument("--mps", action="store_true", help="share the GPU through a CUDA MPS daemon")
    p.add_argument("--omp", type=int, default=2, help="OMP_NUM_THREADS per job (default 2)")
    p.add_argument("--out", default=None, help="output root (default ~/prod/OO-RHIC_OO/bias)")
    p.add_argument("--shard", default="0/1", metavar="I/N",
                   help="run every N-th job of the list, starting at I (0-based): one container "
                        "per GPU, e.g. --shard 0/2 on GPU 0 and --shard 1/2 on GPU 1")
    p.add_argument("--limit", type=int, default=None, metavar="K",
                   help="run at most K of the pending jobs (a timing test)")
    p.add_argument("--events", type=int, default=None, metavar="N",
                   help="events per sample instead of the sample's total (the first jobs of its "
                        "seed list; a larger N later adds the rest)")
    p.add_argument("--list", action="store_true", help="print the pending jobs and exit")
    a = p.parse_args(argv)
    global OUT
    if a.out:
        OUT = Path(a.out)
    names = [x for n in a.names for x in GROUPS.get(n, [n])] if a.names else GROUPS["bias"]
    bad = [n for n in names if n not in SAMPLES]
    if bad:
        sys.exit(f"oo_bias.py: unknown samples {bad}; known {list(SAMPLES)} and the groups "
                 f"{list(GROUPS)}")
    try:
        i, n = (int(x) for x in a.shard.split("/"))
        assert n >= 1 and 0 <= i < n
    except (ValueError, AssertionError):
        sys.exit(f"oo_bias.py: --shard wants I/N with 0 <= I < N, got {a.shard}")
    if a.list:
        for name, seed, ev, cmd in job_list(names, a.events, (i, n), a.limit):
            print(name, seed, ev, " ".join(cmd))
        return 0
    if a.step == "run":
        return run(names, a.j, a.mps, a.omp, a.events, (i, n), a.limit)
    return analyse()


if __name__ == "__main__":
    sys.exit(main())
