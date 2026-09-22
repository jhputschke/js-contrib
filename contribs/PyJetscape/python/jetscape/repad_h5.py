"""
python/jetscape/repad_h5.py

Command-line front end for :func:`jetscape.fno_h5_writer.repad_to`.

``arr`` is rectangular, so every HDF5 file trained on together must share one
``choose_ntau`` or ``MultiH5Array`` raises ``Dimension mismatch``.  Files written by
``H5BulkWriter`` leave the tau axis growable, so reconciling a campaign is a metadata
resize: no data moves and the files do not grow.

    python -m jetscape.repad_h5 run*/hydro_evo.h5                 # to the largest
    python -m jetscape.repad_h5 run*/hydro_evo.h5 --choose-ntau 200
    python -m jetscape.repad_h5 run*/hydro_evo.h5 --dry-run

This is the h5-to-h5 counterpart of ``root2hdf5/root_to_hdf5.py --global-ntau``.  It lives
in its own module rather than in ``fno_h5_writer`` so that ``python -m`` does not re-import
a module the package ``__init__`` has already loaded.
"""

from __future__ import annotations

import argparse
import sys

try:
    from .fno_h5_writer import repad_to
except ImportError:  # run as a plain script: `python repad_h5.py ...`, no package needed
    from fno_h5_writer import repad_to

__all__ = ["main"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m jetscape.repad_h5",
        description="Grow the tau axis of FNO4d-schema HDF5 files to a common "
                    "choose_ntau, in place, so MultiH5Array will concatenate them.")
    ap.add_argument("paths", nargs="+", help="HDF5 files to reconcile")
    ap.add_argument("--choose-ntau", type=int, default=None, dest="choose_ntau",
                    help="target tau extent (default: the largest among the files). "
                         "Pinning larger than needed costs nothing on disk.")
    ap.add_argument("--dry-run", action="store_true", dest="dry_run",
                    help="report what would change without writing")
    args = ap.parse_args(argv)

    try:
        target, changed = repad_to(args.paths, args.choose_ntau, dry_run=args.dry_run)
    except (ValueError, OSError, KeyError) as exc:
        # These are all "your files don't line up" cases; a stack trace helps nobody.
        print(f"error: {exc}", file=sys.stderr)
        return 1
    verb = "would be updated" if args.dry_run else "updated"
    print(f"choose_ntau = {target}; {len(changed)} of {len(args.paths)} file(s) {verb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
