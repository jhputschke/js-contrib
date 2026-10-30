#!/usr/bin/env python3
"""
example/pythonic/oo_workflow.py

The simulation half of oo_workflow.ipynb as a script, and a template for a production of
one's own: a user XML through the whole X-SCAPE chain with the Pythonic layer, the bulk and
jet hadrons of every event into two hadron files (hadrons_h5 format).  It runs from any
directory -- a copy in a container's /work as well as here -- and finds the tree and the
build through jetscape.paths ($XSCAPE, $JETSCAPE_BUILD).

    python oo_workflow.py                              # oo_workflow.xml: 3 O+O events, seed 7
    python oo_workflow.py --events 10 --seed 3 --outdir out_oo_s3
    python oo_workflow.py --xml my_job.xml --outdir out_mine

Outputs in OUTDIR: <name>.xml (the job's XML, as run), <name>_hadrons_bulk_bg.h5 (iSS, one
sample per oversample; only with SoftParticlization in the XML),
<name>_hadrons_jet_frag.h5 (the jet hadrons and their initiators).  Read them back with
js.HadronArray.from_h5(...) -- oo_workflow.ipynb, oo_analysis.ipynb, PYTHONIC.md.

To make it yours: change the XML, and the per-event block below (what is stored).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import xml.etree.ElementTree as ET
from contextlib import nullcontext
from pathlib import Path

HERE = Path(__file__).resolve().parent
try:
    import jetscape as js
except ImportError:                                  # a source tree, without PYTHONPATH
    sys.path.insert(0, str(HERE.parents[1] / "python"))
    import jetscape as js
import numpy as np  # noqa: E402
from jetscape import paths  # noqa: E402
from jetscape import pyjetscape_core as core  # noqa: E402
from jetscape.hadrons_h5 import HadronH5Writer  # noqa: E402
from jetscape.records import initiator_rows  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--xml", default=str(HERE / "oo_workflow.xml"),
                   help="user XML (default: oo_workflow.xml next to this script)")
    p.add_argument("--events", type=int, default=None, help="events (default: the XML's)")
    p.add_argument("--seed", type=int, default=None, help="framework seed (default: the XML's)")
    p.add_argument("--outdir", default="out_oo",
                   help="output directory (default: ./out_oo in the current directory)")
    p.add_argument("--name", default=None, help="file stem (default: the XML's name)")
    p.add_argument("--build", default=None,
                   help="build tree (default: $JETSCAPE_BUILD, else $XSCAPE/build_gpu)")
    p.add_argument("--keep-workdir", action="store_true", dest="keep_workdir",
                   help="keep OUTDIR/work/<name> (the modules' side files)")
    return p.parse_args()


def glauber_record() -> dict:
    """This event's 3dMCGlauber numbers, from events_summary.dat in the working directory."""
    if not os.path.exists("events_summary.dat"):       # not a 3dMCGlauber run
        return {}
    with open("events_summary.dat") as fh:
        rows = [ln.split() for ln in fh if ln.strip() and not ln.startswith("#")]
    _, npart, ncoll, nstrings, b = rows[-1][:5]
    return {"glauber_npart": int(npart), "glauber_ncoll": int(ncoll),
            "glauber_nstrings": int(nstrings), "glauber_b": float(b)}


def main() -> int:
    a = parse_args()
    outdir = Path(a.outdir).resolve()
    name = a.name or Path(a.xml).stem
    outdir.mkdir(parents=True, exist_ok=True)

    # the job's XML in OUTDIR, with --events/--seed applied (JobDir then uses it in place)
    tree = ET.parse(a.xml)
    root = tree.getroot()
    if a.events is not None:
        root.find("nEvents").text = f" {a.events} "
    if a.seed is not None:
        root.find("Random/seed").text = str(a.seed)
    job_xml = outdir / f"{name}.xml"
    tree.write(job_xml)
    iss_node = root.find("SoftParticlization/iSS/number_of_repeated_sampling")
    n_oversample = int(iss_node.text) if iss_node is not None else 1

    stem = str(outdir / name)
    bulk_path, jet_path = stem + "_hadrons_bulk_bg.h5", stem + "_hadrons_jet_frag.h5"
    build = Path(a.build).resolve() if a.build else paths.build_dir()
    print(f"oo_workflow: {job_xml}\n  build {build}\n  out   {outdir}")

    skipped, n_done, t0 = [], 0, time.time()
    with js.JobDir(outdir, job_xml, build=build, keep=a.keep_workdir) as job, \
         js.per_event(job.main_xml, job.user_xml) as driver:
        gun = next((t for t in driver.GetTaskList() if isinstance(t, core.HardProcess)), None)
        iss, had = driver.soft_particlization, driver.hadronization_manager
        mgr, hydro = driver.energy_loss_manager, driver.hydro
        if iss is not None:
            iss.use_compact_output()                   # iSS hands its hadrons over as arrays
        with (HadronH5Writer(bulk_path, tag="bulk_bg", n_samples=n_oversample)
              if iss is not None else nullcontext()) as bulk_w, \
             HadronH5Writer(jet_path, tag="jet_frag", n_samples=1, initiators=True) as jet_w:
            for i in driver.events(on_error=lambda i, exc: skipped.append((i, str(exc)))):
                # ── per event: what is stored ──────────────────────────────────────
                record = glauber_record()
                if gun is not None:
                    info = gun.info
                    record.update(pthat=info["pthat"], sigma_gen=info["sigma_gen"],
                                  event_weight=info["event_weight"])
                n_cells = len(hydro.surface_to_numpy()) if hydro is not None else 0
                bulk = iss.hadron_array() if bulk_w is not None and n_cells else None
                if bulk is not None:
                    bulk_w.append_unit(bulk.to_dict(), unit=n_done, event=i,
                                       seed=iss.last_random_seed, n_cells=n_cells, **record)
                jet = had.hadron_array()
                audit = had.colour_audit() if hasattr(had, "colour_audit") else {}
                initiators = initiator_rows(mgr.initiator_array()) if mgr is not None else None
                jet_w.append_unit(jet.to_dict(), initiators=initiators, unit=n_done, event=i,
                                  seed=had.last_random_seed, **record,
                                  **{f"colour_{k}": v for k, v in audit.items()})
                n_done += 1
                fp = mgr.final_parton_array() if mgr is not None else np.zeros(0)
                print(f"event {i}: {len(fp)} final partons, {n_cells} surface cells, "
                      f"{0 if bulk is None else len(bulk)} bulk hadrons, {len(jet)} jet "
                      f"hadrons, {time.time() - t0:.1f} s")

    print(f"oo_workflow: {n_done} event(s) written, skipped {skipped or 'none'}")
    print(f"  {jet_path}" + (f"\n  {bulk_path}" if iss is not None else ""))
    return 0 if n_done else 1


if __name__ == "__main__":
    raise SystemExit(main())
