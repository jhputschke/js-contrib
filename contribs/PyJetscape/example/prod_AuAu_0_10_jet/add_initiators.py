#!/usr/bin/env python3
"""
example/prod_AuAu_0_10_jet/add_initiators.py

Make particlize files made before 2026-10-01 (format version 1) self-contained: copy each
production file's shower initiators from its pair file (``shower/initiators``) into its
``<stem>_particlize.h5`` (``initiators/``).  Afterwards hadronize.py, HadronFileReader's
``info.initiators()`` and run_h5toROOT.py need only the particlize and hadron files, not the
pair file with the hydro.  Version-2 particlize files have them already.

    ./add_initiators.py out_had                         # every particlize file in it
    ./add_initiators.py "out/t*/*_particlize.h5"        # a glob (quoted: expanded here)
    ./add_initiators.py out_had --force                 # replace existing initiators/

INPUTS are directories (their ``*_particlize.h5``), particlize files, or globs.  The pair file
is the one the particlize file names (``pair_file``), next to it.  Runs in the container and
in the analysis venv (no X-SCAPE needed).  Files that already have initiators are kept.  The
exit code is 1 if a file failed (pair file missing, another run, no shower/ group).
"""
import argparse
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))


def scan(inputs):
    found = set()
    for item in inputs:
        if os.path.isdir(item):
            found.update(glob.glob(os.path.join(item, "*_particlize.h5")))
        elif any(c in item for c in "*?["):
            found.update(p for p in glob.glob(item) if p.endswith("_particlize.h5"))
        elif os.path.exists(item):
            found.add(item)
        else:
            print(f"add_initiators.py: {item}: not found", file=sys.stderr)
    return sorted(found)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="directories, particlize files or globs")
    ap.add_argument("--force", action="store_true", help="replace existing initiators/")
    a = ap.parse_args(argv)

    from jetscape.particlize_h5 import add_initiators_from_pair

    paths = scan(a.inputs)
    added = kept = failed = 0
    for p in paths:
        try:
            if add_initiators_from_pair(p, force=a.force):
                added += 1
                print(f"{p}: added")
            else:
                kept += 1
                print(f"{p}: has initiators/, kept")
        except (OSError, ValueError) as exc:
            failed += 1
            print(f"{p}: FAILED: {exc}", file=sys.stderr)
    print(f"add_initiators.py: {len(paths)} file(s): {added} added, {kept} kept, "
          f"{failed} failed")
    return 1 if failed or not paths else 0


if __name__ == "__main__":
    raise SystemExit(main())
