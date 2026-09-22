#!/usr/bin/env python
"""Smoke test: initial state + FastHydro, no energy loss.

    python run_hydro_only.py --config ../config/fasthydro_twostage.yaml

Runs the solver through the framework and then checks the one thing everything downstream
depends on: that GetHydroInfo() gives back what was stored.  If this fails, Matter and LBT
are quenching against the wrong medium and nothing further is worth looking at.
"""
from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--samples", type=int, default=5000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="k.p=v")
    a = ap.parse_args(argv)

    from fast_data.config import apply_overrides, load_config
    from fasthydro.hydro import FastHydro
    from fasthydro.initial_state import FastGlauberInitialState

    cfg = load_config(a.config)
    if a.set:
        cfg = apply_overrides(cfg, a.set)

    ini = FastGlauberInitialState(cfg)
    ini.Exec()
    h = FastHydro(cfg, stage=1, ic=ini)
    h.InitializeHydro(None)
    h.EvolveHydro()

    g = h.g
    print("\n=== GetHydroInfo vs the stored grid ===")
    # The metric is |de| / e_max, not |de| / e.  Per-cell relative error is meaningless in the
    # dilute tail, where e sits on the 1e-6 vacuum floor and a float32 coordinate rounding of
    # 1e-7 fm moves the interpolation weight enough to matter in ratio and not at all in
    # substance.  This is the same convention README_fv_vs_music uses, and for the same reason.
    rng = np.random.default_rng(0)
    h.reset_out_of_range_count()
    e_max = float(h.arr[0].max())
    worst_abs, worst_at, n = 0.0, None, 0
    worst_interior = 0.0
    for _ in range(a.samples):
        it = int(rng.integers(0, g.ntau))
        ix, iy, ie = (int(rng.integers(0, g.nx)), int(rng.integers(0, g.ny)),
                      int(rng.integers(0, g.neta)))
        tau = g.tau0 + it * g.record_dtau
        eta = g.eta_min + ie * g.deta
        c = h.get_hydro_cell(tau * np.cosh(eta), g.x_min + ix * g.dx,
                             g.y_min + iy * g.dy, tau * np.sinh(eta))
        want = float(h.arr[0, ix, iy, ie, it])
        if want <= 0:
            continue
        n += 1
        d = abs(c.energy_density - want)
        if d > worst_abs:
            worst_abs, worst_at = d, (ix, iy, ie, it)
        on_edge = (ix in (0, g.nx - 1) or iy in (0, g.ny - 1)
                   or ie in (0, g.neta - 1) or it in (0, g.ntau - 1))
        if not on_edge:
            worst_interior = max(worst_interior, d)
    print(f"  {n} fluid nodes sampled, e_max = {e_max:.4g} GeV/fm^3")
    print(f"  worst |de|          : {worst_abs:.3e} GeV/fm^3 at {worst_at}")
    print(f"  worst |de| / e_max  : {worst_abs / max(e_max, 1e-30):.3e}")
    print(f"  interior nodes only : {worst_interior / max(e_max, 1e-30):.3e}")
    print(f"  out-of-grid queries : {h.get_out_of_range_count()}")
    ok = worst_abs / max(e_max, 1e-30) < 1e-5
    print("  => " + ("PASS" if ok else "FAIL (tolerance 1e-5 of the peak, float32 store)"))

    if a.out:
        np.savez_compressed(a.out, arr=h.arr, tau=g.tau_grid())
        print(f"wrote {a.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
