#!/usr/bin/env python3
"""
example/python_modules/run_by_hand.py

The pipeline of run_from_xml.py built in Python (Mode B): the toy modules of
toy_modules.py placed in the managers by hand, a Python writer and the C++ ASCII writer
(create_module("JetScapeWriterAscii")) added to the task list. user_by_hand.xml only sets
the run (events, seed, the energy-loss time steps) and switches the automatic task list
off. The ASCII file is read back with JetScapeReaderAscii at the end.

    conda activate js_fno
    python run_by_hand.py [--outdir DIR]
"""

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_from_xml import XSCAPE, read_back
from toy_modules import (
    ToyAfterburner,
    ToyGun,
    ToyHadronizer,
    ToySampler,
    ToySplit,
    ToyWriter,
    js,
    register_toy_modules,
)


def build_pipeline(ascii_file):
    """The top-level tasks in pipeline order, each manager with its modules inside."""
    eloss_manager = js.JetEnergyLossManager()   # one JetEnergyLoss copy per shower
    eloss = js.JetEnergyLoss()                   # the per-shower container
    eloss.Add(ToySplit())                        # copied per shower with Clone()
    eloss_manager.Add(eloss)

    hadronization_manager = js.HadronizationManager()
    hadronization = js.Hadronization()           # the container the manager runs
    hadronization.Add(ToyHadronizer())
    hadronization_manager.Add(hadronization)

    ascii_writer = js.create_module("JetScapeWriterAscii")
    js.set_writer_output_file(ascii_writer, ascii_file)
    return [ToyGun(), eloss_manager, hadronization_manager, ToySampler(), ToyAfterburner(),
            ToyWriter(), ascii_writer]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", default=".", help="where the ASCII file goes")
    p.add_argument("--main-xml", default=str(XSCAPE / "config" / "jetscape_main.xml"))
    a = p.parse_args(argv)
    os.makedirs(a.outdir, exist_ok=True)
    os.chdir(a.outdir)

    # not needed to make the modules, but JetScape.Init() then gives ToyGun its default
    # <Hard><CustomModuleToyGun><E> in the main XML
    register_toy_modules()
    tasks = build_pipeline("toy_by_hand.dat")
    driver = js.JetScapePerEvent()
    driver.SetXMLMainFileName(a.main_xml)
    driver.SetXMLUserFileName(str(HERE / "user_by_hand.xml"))
    for task in tasks:
        driver.Add(task)
    driver.Init()
    print("tasks:", [type(t).__name__ for t in driver.GetTaskList()])
    afterburner = next(t for t in tasks if isinstance(t, ToyAfterburner))
    writer = next(t for t in tasks if isinstance(t, ToyWriter))
    summary = {"afterburner": [], "writer": None, "ascii": None}
    for _ in range(driver.GetNumberOfEvents()):
        driver.ExecPerEvent()
        summary["afterburner"].append(len(afterburner.GetCurrentHadronList()))
        driver.ClearPerEvent()
    driver.Finish()
    summary["writer"] = writer.events
    summary["ascii"] = read_back("toy_by_hand.dat")
    print("SUMMARY " + json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
