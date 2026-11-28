#!/usr/bin/env python3
"""
analysis/OO-RHIC/hard_tune/dndeta.py -- the charged-particle pseudorapidity distribution over the full eta
range at fixed pTHat, for all p+p models and the MPI variants of the X-SCAPE chain: where in eta
the particles are made (the X-SCAPE remnants are one per initiator, Pythia's a shared beam
remnant; tuning page, "Full soft MPI").

    python dndeta.py run -j 20          # the samples below, under PROD_DIR/diag/eta/<window>/<name>
    python dndeta.py run best pp19      # some of them
    python dndeta.py analyse            # -> results/dndeta.json (the notebook)
    python diag_strings.py PROD/diag/eta/9-11/best_nompi PROD/diag/eta/9-11/best \\
        PROD/diag/eta/9-11/best_mpi4 PROD/diag/eta/9-11/best_mpidef   # partons, topology

Samples (EVENTS per window, all particles stored, pi0 and eta undecayed as everywhere):
  X-SCAPE best point (design.json "best") in vacuum with the tune's MPI (pTmin 5, lowered to
  ~1.7 GeV by Pythia at 200 GeV), pTmin 4, MPI off, and Pythia's full MPI (pTmin 0.2); the
  parton input of iColored stored (--store-hadronization-input) for diag_strings.py;
  JETSCAPE PP19 (refs/run_refs.py's setup); standalone Pythia 8 default, Detroit, default without
  colour reconnection, and default with the tune's alpha_s (also the MPI's) and PDF, CR on/off.
Resumable: a sample whose .json reports all events written is skipped.

Analysis: per event, charged particles (|pid| in jetscape.hadrons_h5.CHARGED) in eta bins of
0.25 over |eta| < 8, all pT and pT > 0.2 GeV; mean over events and its standard error.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import _paths  # noqa: E402,F401  (analysis/OO-RHIC and its test directories on sys.path)
ROOT = _paths.ROOT                           # analysis/OO-RHIC: results/, refs/, data/
import config as C  # noqa: E402

EVENTS = 3000
WINDOWS = [(9, 11), (23, 30)]
EDGES = np.arange(-8, 8.0001, 0.25)
TUNE_COUPLING = ["PDF:pSet = 20", "SigmaProcess:alphaSvalue = 0.153",
                 "MultipartonInteractions:alphaSvalue = 0.153"]
#: name: (generator, label, extra Pythia lines)
SAMPLES = {
    "best_nompi": ("xscape", "X-SCAPE best, MPI off", ["PartonLevel:MPI = off"]),
    "best": ("xscape", "X-SCAPE best (tune MPI: pTmin 5 → ~1.7)", []),
    "best_mpi4": ("xscape", "X-SCAPE best, MPI pTmin 4", ["MultipartonInteractions:pTmin = 4.0"]),
    "best_mpidef": ("xscape", "X-SCAPE best, full MPI (pTmin 0.2)",
                    ["MultipartonInteractions:pTmin = 0.2"]),
    "pp19": ("pp19", "JETSCAPE PP19", []),
    "pythia8_default": ("pythia:default", "Pythia 8 default (Monash)", []),
    "pythia8_detroit": ("pythia:detroit", "Pythia 8 Detroit (STAR)", []),
    "pythia8_nocr": ("pythia:nocr", "Pythia 8 default, CR off", []),
    "pythia8_tunecpl": ("pythia:default", "Pythia 8, tune α_s/PDF (also MPI)", TUNE_COUPLING),
    "pythia8_tunecpl_nocr": ("pythia:nocr", "Pythia 8, tune α_s/PDF, CR off", TUNE_COUPLING),
}


def wtext(w):
    return f"{w[0]:g}-{w[1]:g}"


def sample_dir(name, w):
    return C.PROD_DIR / "diag" / "eta" / wtext(w) / name


def job(name, w, seed):
    gen, _, lines = SAMPLES[name]
    out = sample_dir(name, w)
    if gen == "xscape":
        import make_design as D
        import run_runs as R
        prm = D.load()["extra"]["best"]["params"]
        cmd, _, _ = R.command("best", prm, seed, [w], lines, events=EVENTS)
        i = cmd.index("--eta-max")
        del cmd[i:i + 2]                                   # all particles
        cmd[cmd.index("--outdir") + 1] = str(out)
        cmd.append("--store-hadronization-input")
        cwd = C.DRIVER.parent
    elif gen == "pp19":
        vac = C.EXAMPLE / "prod_AuAu_0_10_jet" / "run_prod_vacuum.py"
        cmd = [sys.executable, str(vac), "--events", str(EVENTS), "--seed", str(seed),
               "--user-xml", str(ROOT / "refs" / "pp19_200.xml"),
               "--hadronize-xml", str(ROOT / "refs" / "hadronize_pp19.xml"),
               "--pthat-bins", wtext(w), "--n-frag", "1", "--outdir", str(out),
               "--seed-registry", "none"]
        cwd = vac.parent
    else:
        cmd = [sys.executable, str(ROOT / "refs" / "pythia_ref.py"), "--tune", gen.split(":")[1],
               "--events", str(EVENTS), "--seed", str(seed), "--windows", wtext(w),
               "--eta-max", "0", "--out", str(out / f"{name}_seed{seed}_hadrons_jet_frag.h5")]
        for line in lines:
            cmd += ["--line", line]
        cwd = HERE
    return cmd, out, cwd


def done(out):
    for js in out.glob("*.json"):
        try:
            if json.loads(js.read_text()).get("events_written", 0) == EVENTS:
                return True
        except (OSError, ValueError):
            pass
    return False


def run(names, nproc):
    todo = []
    for name in names:
        i = list(SAMPLES).index(name)                      # seeds independent of the selection
        for k, w in enumerate(WINDOWS):
            cmd, out, cwd = job(name, w, 4200 + 10 * i + k)
            if not done(out):
                todo.append((name, w, cmd, out, cwd))
    print(f"dndeta.py: {len(todo)} samples to run, {nproc} at a time", flush=True)
    env = dict(os.environ, OMP_NUM_THREADS="1")
    t0 = time.time()

    def one(item):
        name, w, cmd, out, cwd = item
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "run.log", "w") as log:
            r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(cwd))
        return name, w, r.returncode

    n_fail = 0
    with ThreadPoolExecutor(max_workers=nproc) as ex:
        for name, w, rc in ex.map(one, todo):
            n_fail += rc != 0
            print(f"{name} {wtext(w)}: {'ok' if rc == 0 else f'FAILED (rc {rc})'}, "
                  f"{time.time() - t0:.0f} s", flush=True)
    return 1 if n_fail else 0


def analyse(names):
    import observables as O
    charged = O._charged_ids()
    nb = len(EDGES) - 1
    out = {"edges": EDGES.tolist(), "events": EVENTS, "windows": [list(w) for w in WINDOWS],
           "samples": {}}
    for name in names:
        res = {"label": SAMPLES[name][1]}
        for w in WINDOWS:
            files = sorted(sample_dir(name, w).glob("*_hadrons_jet_frag.h5"))
            if not files:
                continue
            h = {"all": [], "pt02": []}
            for f in files:
                d = O.read_file(str(f))
                pt, _, eta, _ = O._kin(d["p"])
                ev = np.repeat(np.arange(len(d["counts"])), d["counts"])
                ch = np.isin(np.abs(d["pid"]), charged)
                ib = np.digitize(eta, EDGES) - 1
                ok = ch & (ib >= 0) & (ib < nb)
                n_ev = len(d["counts"])
                for key, m in (("all", ok), ("pt02", ok & (pt > 0.2))):
                    h[key].append(np.bincount(ev[m] * nb + ib[m],
                                              minlength=n_ev * nb).reshape(n_ev, nb))
            r = {"n_events": int(sum(len(x) for x in h["all"]))}
            for key in h:
                a = np.concatenate(h[key]) / np.diff(EDGES)
                r[key] = a.mean(0).tolist()
                r[key + "_sem"] = (a.std(0) / np.sqrt(len(a))).tolist()
            res[wtext(w)] = r
            c = 0.5 * (EDGES[:-1] + EDGES[1:])
            mid = np.abs(c) < 1
            print(f"{name:22s} {wtext(w):6s} {r['n_events']:5d} ev  dN/deta(|eta|<1) "
                  f"{np.mean(np.asarray(r['all'])[mid]):5.2f}  total "
                  f"{np.sum(np.asarray(r['all']) * np.diff(EDGES)):5.1f}")
        out["samples"][name] = res
    path = ROOT / "results" / "dndeta.json"
    path.parent.mkdir(exist_ok=True)
    if path.exists():                       # keep samples analysed earlier
        old = json.loads(path.read_text())
        if old.get("edges") == out["edges"]:
            out["samples"] = {**old.get("samples", {}), **out["samples"]}
    path.write_text(json.dumps(out))
    print("wrote", path)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=("run", "analyse"))
    p.add_argument("names", nargs="*", default=list(SAMPLES))
    p.add_argument("-j", type=int, default=os.cpu_count())
    a = p.parse_args(argv)
    bad = [n for n in a.names if n not in SAMPLES]
    if bad:
        sys.exit(f"dndeta.py: unknown samples {bad}; known: {list(SAMPLES)}")
    return run(a.names, a.j) if a.step == "run" else analyse(a.names)


if __name__ == "__main__":
    sys.exit(main())
