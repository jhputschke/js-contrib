#!/usr/bin/env python3
"""
utils/check_deposition.py -- flag Au+Au jet jobs whose jets deposited far less than the rest.

Once, in ~250 campaign jobs on the 2 x RTX 3090 box (2026-10-12), all 15 events of one job lost
almost no jet energy: 1-5 droplets per event instead of 13-34, with a correct background, the
correct initial partons and no warning in the log; the same seed run alone was normal
(docs/REPORT_FINAL_OPTIM.md section 4).  This reads every job summary (<stem>.json) of the
given directories and flags a job whose droplets per event (droplets_total / events_written)
fall below a fraction (default 0.5) of the campaign's median.

    utils/check_deposition.py /work/AuAu_c1                 # every *.json below it
    utils/check_deposition.py /data/c1/gpu0 /data/c1/gpu1   # one campaign over two directories
    utils/check_deposition.py /work/AuAu_c1 --fraction 0.6

Reports only, and exits 1 if a job is flagged.  To redo one: move its files (<stem>.*) out of
its OUTDIR and run the campaign's command again; run_jobs.sh reruns the job (with a new seed in
a FIRST_SEED 0 campaign).  Standard library only: runs on the host, in the venv or in the
container.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys


def rows(dirs):
    out = []
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "**", "*.json"), recursive=True)):
            try:
                with open(p) as f:
                    j = json.load(f)
            except (OSError, ValueError):
                continue
            if (isinstance(j, dict) and j.get("droplets_total") is not None
                    and j.get("events_written")):
                out.append((j["droplets_total"] / j["events_written"], j["events_written"], p))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", help="campaign directories (searched recursively)")
    ap.add_argument("--fraction", type=float, default=0.5,
                    help="flag jobs below this fraction of the median (default 0.5)")
    a = ap.parse_args(argv)
    r = rows(a.dirs)
    if len(r) < 3:
        print(f"check_deposition: {len(r)} finished Au+Au job(s) found, too few to compare")
        return 0
    med = statistics.median(x for x, _, _ in r)
    bad = [x for x in sorted(r) if x[0] < a.fraction * med]
    print(f"check_deposition: {len(r)} jobs, {sum(n for _, n, _ in r)} events, median "
          f"{med:.1f} droplets per event; {len(bad)} below {a.fraction:g} x the median")
    for x, n, p in bad:
        print(f"  REDO {p}: {x:.1f} droplets per event ({n} events)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
