#!/usr/bin/env python3
"""
example/python_modules/run_from_xml.py

Every stage a Python module (toy_modules.py), named in the user XML
(user_from_xml.xml: <Hard><CustomModuleToyGun>, <Eloss><CustomModuleToySplit>, ...), with
the C++ ASCII writer of the XML and a Python writer next to it. The ASCII file is read
back with JetScapeReaderAscii at the end.

    conda activate js_fno
    python run_from_xml.py [--outdir DIR]
"""

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from toy_modules import (
    ToyAfterburner,
    ToyWriter,
    js,
    register_toy_modules,
)

XSCAPE = HERE.parents[5]


def read_back(path):
    """(showers, hadrons, sum of the hadrons' E, sigma_gen) per event of a
    JetScapeWriterAscii file"""
    reader = js.JetScapeReaderAscii(str(path))
    events = []
    while not reader.Finished():
        reader.Next()
        if reader.GetCurrentNumberOfPartonShowers() or reader.GetHadrons():
            hadrons = reader.GetHadrons()
            events.append((reader.GetCurrentNumberOfPartonShowers(), len(hadrons),
                           round(sum(h.e() for h in hadrons), 6), reader.GetSigmaGen()))
    reader.Close()
    return events


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", default=".", help="where the ASCII file goes")
    p.add_argument("--main-xml", default=str(XSCAPE / "config" / "jetscape_main.xml"))
    a = p.parse_args(argv)
    os.makedirs(a.outdir, exist_ok=True)
    os.chdir(a.outdir)

    register_toy_modules()
    driver = js.JetScapePerEvent()
    driver.SetXMLMainFileName(a.main_xml)
    driver.SetXMLUserFileName(str(HERE / "user_from_xml.xml"))
    writer = ToyWriter()
    driver.Add(writer)
    driver.Init()
    print("tasks:", [type(t).__name__ for t in driver.GetTaskList()])
    afterburner = next(t for t in driver.GetTaskList() if isinstance(t, ToyAfterburner))
    summary = {"afterburner": [], "writer": None, "ascii": None}
    for _ in range(driver.GetNumberOfEvents()):
        driver.ExecPerEvent()
        summary["afterburner"].append(len(afterburner.GetCurrentHadronList()))
        driver.ClearPerEvent()
    driver.Finish()
    summary["writer"] = writer.events
    summary["ascii"] = read_back("toy_from_xml.dat")
    print("SUMMARY " + json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
