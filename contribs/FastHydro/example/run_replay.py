#!/usr/bin/env python
"""Re-run the jet leg from a droplet dump.

    python run_replay.py --droplets out/run.droplets.npz \
                         --config ../config/fasthydro_twostage.yaml --check

Needs neither a JETSCAPE pipeline nor pyjetscape_core: replay drives fast_data directly.
Use it to try a different grid, EoS or transport on the same shower without paying for
Matter+LBT again.  With --check it asserts the replay reproduces the recorded run.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--droplets", required=True)
    ap.add_argument("--config", required=True, help="fast_data YAML")
    ap.add_argument("--event", type=int, default=0)
    ap.add_argument("--out", default=None, help="npz for the replayed evolution")
    ap.add_argument("--check", action="store_true",
                    help="assert the replay reproduces the recorded arr_jet")
    ap.add_argument("--set", action="append", default=[], metavar="k.p=v")
    a = ap.parse_args(argv)

    import _bootstrap_replay  # noqa: F401
    from fast_data.config import apply_overrides, load_config
    from fasthydro.droplets_io import load_droplets_npz
    from fasthydro.replay import replay_event, verify_replay

    cfg = load_config(a.config)
    if a.set:
        cfg = apply_overrides(cfg, a.set)

    per_event, params, meta = load_droplets_npz(a.droplets)
    if not 0 <= a.event < len(per_event):
        raise SystemExit(f"event {a.event} out of range (dump has {len(per_event)})")
    da = per_event[a.event]

    ic = meta.get("ic")
    if ic is None or ic.size == 0:
        raise SystemExit(
            f"{a.droplets} carries no initial condition, so a replay cannot reproduce the "
            "original. Re-run run_two_stage.py with --dump-droplets (it stores the IC).")

    print(f"replaying event {a.event}: {len(da)} droplets, "
          f"E={da.data[:, 4].sum():.3f} GeV, params={params}")
    arr, src, diag = replay_event(cfg, ic, da, params)
    print(f"  {diag.get('n_steps', '?')} steps, "
          f"tau_fo={diag.get('tau_freezeout', float('nan')):.3f} fm/c")

    if a.check:
        ref = meta.get("arr_sha256")
        sha = str(ref[a.event]) if ref is not None and len(ref) > a.event else None
        exact = cfg["run"]["device"] == "cpu" and cfg["run"]["dtype"] == "float64"
        ok, msg = verify_replay(arr, reference_sha256=sha, exact=exact)
        print(f"  check: {msg}")
        if not ok:
            if not exact:
                print("  (not on CPU/float64, so bitwise reproduction is not expected)")
            return 1

    if a.out:
        np.savez_compressed(a.out, arr_jet=arr,
                            src=(src if src is not None else np.zeros(0, np.float32)))
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
