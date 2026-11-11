#!/usr/bin/env python3
"""
run_notebooks.py CONTRIB -- execute a contrib's tutorials, as CI and authors do.

    python tutorials/run_notebooks.py pyjetscape                 # check: run, keep nothing
    python tutorials/run_notebooks.py pyjetscape --inplace       # store the outputs (website)
    python tutorials/run_notebooks.py pyjetscape --only 01 06

Runs every *ready* tutorial of tutorials.json whose needs this environment meets (the same
test as 01_check_setup.ipynb) and skips the others, saying why.  A notebook fails when a
cell raises, so the "check your result" asserts make a run fail.  The working directory of
each notebook is its folder, as in Jupyter.  Needs nbclient and ipykernel.

Store outputs (--inplace) from a neutral setup, such as the production container, so that the
website shows no paths of your own computer.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="run_notebooks.py", description=__doc__.split("\n\n")[0])
    p.add_argument("contrib", help="the folder below tutorials/, e.g. pyjetscape")
    p.add_argument("--inplace", action="store_true", help="write the outputs into the notebooks")
    p.add_argument("--only", nargs="*", help="tutorial ids to run (default: all ready ones)")
    p.add_argument("--timeout", type=int, default=1800, help="seconds per cell")
    a = p.parse_args(argv)

    folder = HERE / a.contrib
    sys.path.insert(0, str(folder))
    from _common import tutorial_env as te           # noqa: E402

    import nbclient
    import nbformat

    env = te.setup()
    tuts = json.loads((folder / "tutorials.json").read_text())["tutorials"]
    failed = []
    for t in tuts:
        if a.only and t["id"] not in a.only:
            continue
        if t["status"] != "ready":
            continue
        ok, missing = te.meets(t["needs"], env.caps)
        if not ok:
            print(f"skip {t['file']}: needs {', '.join(missing)}")
            continue
        path = folder / t["file"]
        nb = nbformat.read(path, as_version=4)
        t0 = time.time()
        try:
            nbclient.NotebookClient(nb, timeout=a.timeout, kernel_name="python3",
                                    resources={"metadata": {"path": str(folder)}}).execute()
        except Exception as e:                       # noqa: BLE001
            print(f"FAIL {t['file']} ({time.time() - t0:.0f} s): {type(e).__name__}: "
                  f"{str(e).strip().splitlines()[-1] if str(e).strip() else ''}")
            failed.append(t["file"])
            continue
        print(f"ok   {t['file']} ({time.time() - t0:.0f} s)")
        if a.inplace:
            nbformat.write(nb, path)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
