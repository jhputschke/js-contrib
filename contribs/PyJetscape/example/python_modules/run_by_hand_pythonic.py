#!/usr/bin/env python3
"""
example/python_modules/run_by_hand_pythonic.py

run_by_hand.py with the Pythonic layer's driver: js.per_event(...) as a context manager
(Init on entry, Finish on exit) and driver.events() for the event loop.  --style old
runs the toys of toy_modules.py, --style new those of toy_modules_pythonic.py; both print
the SUMMARY line run_by_hand.py prints.

    conda activate js_fno
    python run_by_hand_pythonic.py [--style old|new] [--outdir DIR]
"""

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_from_xml import XSCAPE, read_back  # noqa: E402
from toy_modules import ToyAfterburner, ToyWriter, js, register_toy_modules  # noqa: E402


def build_pipeline(style, ascii_file):
    """The top-level tasks in pipeline order, each manager with its modules inside."""
    if style == "new":
        import toy_modules_pythonic as toys

        gun, split, hadronizer, sampler = (toys.ToyGunV2(), toys.ToySplitV2(),
                                           toys.ToyHadronizerV2(), toys.ToySamplerV2())
    else:
        import toy_modules as toys

        register_toy_modules()          # gives ToyGun its <Hard><CustomModuleToyGun><E>
        gun, split, hadronizer, sampler = (toys.ToyGun(), toys.ToySplit(),
                                           toys.ToyHadronizer(), toys.ToySampler())
    eloss_manager = js.JetEnergyLossManager()
    eloss = js.JetEnergyLoss()
    eloss.Add(split)
    eloss_manager.Add(eloss)
    hadronization_manager = js.HadronizationManager()
    hadronization = js.Hadronization()
    hadronization.Add(hadronizer)
    hadronization_manager.Add(hadronization)
    ascii_writer = js.create_module("JetScapeWriterAscii")
    js.set_writer_output_file(ascii_writer, ascii_file)
    return [gun, eloss_manager, hadronization_manager, sampler, ToyAfterburner(),
            ToyWriter(), ascii_writer]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--style", choices=("old", "new"), default="new")
    p.add_argument("--outdir", default=".", help="where the ASCII file goes")
    p.add_argument("--main-xml", default=str(XSCAPE / "config" / "jetscape_main.xml"))
    a = p.parse_args(argv)
    os.makedirs(a.outdir, exist_ok=True)
    os.chdir(a.outdir)

    tasks = build_pipeline(a.style, "toy_by_hand.dat")
    afterburner = next(t for t in tasks if isinstance(t, ToyAfterburner))
    writer = next(t for t in tasks if isinstance(t, ToyWriter))
    summary = {"afterburner": [], "writer": None, "ascii": None}
    with js.per_event(a.main_xml, HERE / "user_by_hand.xml", tasks=tasks) as driver:
        print("tasks:", [type(t).__name__ for t in driver.GetTaskList()])
        for _ in driver.events():
            summary["afterburner"].append(len(afterburner.GetCurrentHadronList()))
    summary["writer"] = writer.events
    summary["ascii"] = read_back("toy_by_hand.dat")
    print("SUMMARY " + json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
