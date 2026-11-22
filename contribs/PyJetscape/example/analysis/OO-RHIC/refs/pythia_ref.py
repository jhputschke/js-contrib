#!/usr/bin/env python3
"""
analysis/OO-RHIC/refs/pythia_ref.py -- standalone Pythia 8 p+p 200 GeV reference events in the
layout of the tune's hadron files, so that observables.py and ue_check.py read them unchanged.

    python refs/pythia_ref.py --tune default --events 30000 --seed 960000 --out FILE.h5
    python refs/pythia_ref.py --tune detroit ...

Tunes:
  default   Pythia 8 as it comes (8.312: Monash 2013, NNPDF2.3 QCD+QED LO, ISR, FSR, MPI,
            colour reconnection, hadronization and decays on)
  nocr      diagnostic: default with ColourReconnection:reconnect = off
  detroit   STAR's Detroit tune (Aguilar et al., PRD 105, 016011 (2022)), the settings listed
            on the tuning page: MPI pT0Ref 1.40 at ecmRef 200, ecmPow 0.135, bProfile 2,
            coreRadius 0.56, coreFraction 0.78, ColourReconnection:range 5.4, PDF:pSet 17

Common: HardQCD:all in the pTHat windows of config.PTHAT_WINDOWS (one Pythia per window, the
same number of events in each; event i in window i mod K), pi0 and eta undecayed
(config.FIXED_HADRONIZATION's convention), final particles with |eta| < config.ETA_MAX stored.
Options for diagnostics (dndeta.py): --windows 9-11 (other windows), --eta-max 0 (all
particles), --line "Pythia setting" (added after the tune's lines, repeatable).

File layout (the subset observables.read_file and ue_check.py use): hadrons/pid, hadrons/p
(E, px, py, pz), hadrons/sample_offsets and hadrons/unit_offsets (one sample per event),
units/pthat, units/pthat_bin; attributes nunits_written, pthat_bins, pthat_bin_sigma_gen [mb]
(Pythia's sigmaGen per window), pthat_bin_n_accepted, generator, tune.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

TUNES = {
    "default": [],
    "detroit": ["MultipartonInteractions:ecmRef = 200",
                "MultipartonInteractions:pT0Ref = 1.40",
                "MultipartonInteractions:ecmPow = 0.135",
                "MultipartonInteractions:bProfile = 2",
                "MultipartonInteractions:coreRadius = 0.56",
                "MultipartonInteractions:coreFraction = 0.78",
                "ColourReconnection:range = 5.4",
                "PDF:pSet = 17"],
    # diagnostic: Pythia default without colour reconnection (how much CR lowers the UE)
    "nocr": ["ColourReconnection:reconnect = off"],
}


def make_pythia(window, seed, tune, extra=()):
    import pythia8
    py = pythia8.Pythia("", False)
    for line in (["Beams:eCM = 200", "HardQCD:all = on",
                  f"PhaseSpace:pTHatMin = {window[0]}", f"PhaseSpace:pTHatMax = {window[1]}",
                  "Random:setSeed = on", f"Random:seed = {seed}", "Next:numberCount = 0",
                  "111:mayDecay = off", "221:mayDecay = off"] + TUNES[tune] + list(extra)):
        if not py.readString(line):
            raise ValueError(f"Pythia rejected {line!r}")
    if not py.init():
        raise RuntimeError(f"Pythia init failed for window {window}")
    return py


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tune", choices=sorted(TUNES), default="default")
    p.add_argument("--events", type=int, default=3000)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--windows", default=None,
                   help="pTHat windows, e.g. 9-11,23-30 (default config.PTHAT_WINDOWS)")
    p.add_argument("--eta-max", type=float, default=C.ETA_MAX, dest="eta_max",
                   help=f"stored |eta| range (default {C.ETA_MAX}; 0: all particles)")
    p.add_argument("--line", action="append", default=[],
                   help="extra Pythia setting after the tune's (repeatable)")
    a = p.parse_args(argv)
    windows = (C.PTHAT_WINDOWS if a.windows is None else
               [tuple(float(x) for x in w.split("-")) for w in a.windows.split(",")])
    eta_max = a.eta_max if a.eta_max > 0 else np.inf
    K = len(windows)
    if a.events % K:
        sys.exit(f"pythia_ref.py: --events must be a multiple of the {K} windows")
    t0 = time.time()
    pys = [make_pythia(w, (a.seed * 100 + k) % 900000000 + 1, a.tune, a.line)
           for k, w in enumerate(windows)]
    pid, mom, offs, pth, win = [], [], [0], [], []
    n_fail = 0
    for i in range(a.events):
        k = i % K
        py = pys[k]
        while not py.next():
            n_fail += 1
        ev = py.event
        rows = [(q.id(), q.e(), q.px(), q.py(), q.pz()) for q in ev
                if q.isFinal() and abs(q.eta()) < eta_max]
        if rows:
            r = np.asarray(rows, dtype=np.float64)
            pid.append(r[:, 0].astype(np.int32))
            mom.append(r[:, 1:].astype(np.float32))
        offs.append(offs[-1] + len(rows))
        pth.append(py.infoPython().pTHat())
        win.append(k)
    sig = np.array([q.infoPython().sigmaGen() for q in pys])
    nacc = np.array([q.infoPython().nAccepted() for q in pys], dtype=np.int64)
    import h5py
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as f:
        g = f.create_group("hadrons")
        g.create_dataset("pid", data=np.concatenate(pid), compression="gzip")
        g.create_dataset("p", data=np.concatenate(mom), compression="gzip")
        g.create_dataset("sample_offsets", data=np.asarray(offs, dtype=np.int64))
        g.create_dataset("unit_offsets", data=np.arange(a.events + 1, dtype=np.int64))
        u = f.create_group("units")
        u.create_dataset("pthat", data=np.asarray(pth))
        u.create_dataset("pthat_bin", data=np.asarray(win, dtype=np.int32))
        f.attrs["nunits_written"] = a.events
        f.attrs["pthat_bins"] = np.asarray(windows, dtype=np.float64)
        f.attrs["pthat_bin_sigma_gen"] = sig
        f.attrs["pthat_bin_n_accepted"] = nacc
        f.attrs["generator"] = "Pythia 8 standalone (analysis/OO-RHIC/refs/pythia_ref.py)"
        f.attrs["tune"] = a.tune
        f.attrs["tune_lines"] = json.dumps(TUNES[a.tune] + a.line)
        f.attrs["eta_max"] = a.eta_max
        f.attrs["seed"] = a.seed
    summary = {"out": str(out), "tune": a.tune, "seed": a.seed, "events_written": a.events,
               "events_skipped": 0, "next_failures": n_fail, "sigma_gen_mb": sig.tolist(),
               "wall_s": round(time.time() - t0, 1)}
    out.with_suffix("").with_suffix(".json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
