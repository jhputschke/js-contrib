#!/usr/bin/env python
"""Particlize one FastHydro leg with iSS: the hadron-level half of a wake measurement.

    # jet leg: IC -> jets (Matter+LBT) -> liquefier -> FastHydro_jet -> iSS
    python run_particlize.py --leg jet --events 10 --out-dir out_particlize
    # background leg: the SAME events (same seed), no jets, many more oversamples
    python run_particlize.py --leg bg  --events 10 --out-dir out_particlize --oversample 200

    python delta_spectra.py --out-dir out_particlize          # (jet - bg) at hadron level

Each event's initial condition depends only on (run.seed, event index), not on the jets, so
event k of the bg run is the background of event k of the jet run.  Both runs record the IC's
sha256 per event in <leg>_events.json, and delta_spectra.py refuses to pair events whose
hashes differ.

The surface is not built here, or in Python at all.  iSS finds the chosen leg handed over none
and builds one from its stored evolution in C++ (see fasthydro/particlization.py).

Run it from the X-SCAPE build tree: the framework and iSS resolve paths relative to it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import xml.etree.ElementTree as ET

import _bootstrap  # noqa: F401  (puts PyJetscape + FastHydro on sys.path)

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CFG = os.path.join(os.path.dirname(_HERE), "config")


def derived_xml(user_xml, leg, oversample, path):
    """A copy of ``user_xml`` with <SoftParticlization><hydro_id> (and the oversampling) set.

    The C++ side reads hydro_id from the XML itself (JetScape::SetPointers), so the leg has to
    be in the file, not just in Python.
    """
    tree = ET.parse(user_xml)
    sp = tree.getroot().find("SoftParticlization")
    if sp is None:
        raise SystemExit(f"{user_xml} has no <SoftParticlization> block")
    hid = sp.find("hydro_id")
    if hid is None:
        hid = ET.SubElement(sp, "hydro_id")
    hid.text = f"FastHydro_{leg}"
    if oversample is not None:
        n = sp.find("iSS/number_of_repeated_sampling")
        if n is None:
            raise SystemExit(f"{user_xml} has no <SoftParticlization><iSS>"
                             f"<number_of_repeated_sampling>")
        n.text = str(int(oversample))
    tree.write(path)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--leg", choices=("jet", "bg"), required=True)
    ap.add_argument("--config", default=os.path.join(_CFG, "fasthydro_particlize.yaml"))
    ap.add_argument("--user-xml",
                    default=os.path.join(_CFG, "jetscape_user_fasthydro_particlize.xml"))
    ap.add_argument("--main-xml", default="../config/jetscape_main.xml")
    ap.add_argument("--events", type=int, default=None)
    ap.add_argument("--oversample", type=int, default=None,
                    help="iSS samples per event (default: the XML's number_of_repeated_sampling)")
    ap.add_argument("--out-dir", default="out_particlize")
    ap.add_argument("--hard", default="PythiaGun",
                    help="jet leg only. PGun zeroes the sampled vertex (PGun.cc:117-120).")
    ap.add_argument("--set", action="append", default=[], metavar="k.p=v",
                    help="dotted override of the YAML, repeatable")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    from fasthydro.config import load_config
    from fasthydro.pipeline import build_bg_only, build_two_stage
    from jetscape.run_jetscape import run_manual

    os.makedirs(a.out_dir, exist_ok=True)
    cfg = load_config(a.config, a.set)
    nev = a.events if a.events is not None else int(cfg["run"]["nevents"])
    user_xml = derived_xml(a.user_xml, a.leg, a.oversample,
                           os.path.join(a.out_dir, f"jetscape_user_{a.leg}.xml"))
    hadron_file = os.path.join(a.out_dir, f"{a.leg}_final_state_hadrons.dat")

    common = dict(user_xml=user_xml, main_xml=a.main_xml, verbose=not a.quiet,
                  keep_bg_arr=False, hadron_file=hadron_file)
    if a.leg == "jet":
        modules, parts = build_two_stage(cfg, hard=(a.hard or None), **common)
    else:
        modules, parts = build_bg_only(cfg, **common)
    hyd = parts[f"hyd_{a.leg}"]
    soft = parts["particlization"]

    # One record per event, taken in the sampled leg's Clear() -- the only moment the event's
    # hydro results still exist. The last event's Clear() fires inside Finish().
    events = []
    _orig_clear = hyd.Clear

    def _record():
        if hyd.ic_sha256 is not None:
            rec = {"event": len(events), "ic_sha256": hyd.ic_sha256,
                   "tau_freezeout": hyd.diag.get("tau_freezeout"),
                   "closure": hyd.closure}
            P = hyd.diag.get("P_cart")
            if P is not None:          # the four-momentum the source deposited, [E, px, py, pz]
                rec["deposited_P"] = np.asarray(P, dtype=float).sum(axis=0).tolist()
            events.append(rec)
            hyd.ic_sha256 = None       # so a repeated Clear() does not record the event twice
        _orig_clear()

    hyd.Clear = _record

    t0 = time.time()
    run_manual(a.main_xml, user_xml, modules, n_events=nev)
    wall = time.time() - t0

    meta = {
        "leg": a.leg, "n_events": nev, "n_oversample": soft["n_oversample"],
        "T_sw": soft["T_sw"], "hadron_file": os.path.abspath(hadron_file),
        "user_xml": os.path.abspath(user_xml), "config": os.path.abspath(a.config),
        "overrides": a.set, "seed": int(cfg["run"]["seed"]), "wall_s": wall,
        "events": events,
    }
    out_json = os.path.join(a.out_dir, f"{a.leg}_events.json")
    with open(out_json, "w") as f:
        json.dump(meta, f, indent=1)

    open_events = [e["event"] for e in events if e["closure"] and not e["closure"]["closed"]]
    print(f"\n=== {a.leg} leg: {len(events)} event(s), {soft['n_oversample']} oversamples "
          f"each, {wall:.0f} s ===")
    print(f"hadrons    : {hadron_file}")
    print(f"provenance : {out_json}")
    if open_events:
        print(f"WARNING: the surface did not close in event(s) {open_events}; see 'closure'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
