#!/usr/bin/env python3
"""
example/prod_AuAu_0_10_jet/wake_grid_compare.py

How much of the jet's wake survives the output grid: a pair file written on MUSIC's own grid
(``run_prod_jet.py --native``) is resampled offline onto other grids with the same
:func:`jetscape.bulk_sources.resample` that ``PairH5Writer`` uses, so the hydro is identical
and only the sampling differs.  See docs/Wake_grid_comparison.md for the method and results.

    python run_prod_jet.py --native --events 6 --reuse 2 --seed 7 --outdir OUT --seed-registry none
    python wake_grid_compare.py OUT/AuAu_0_10_jet_seed0007.h5 [--per-event]

Per frame with both legs alive, for the wake de = e_jet - e_bg:

    net, pos, neg   sum of de (its positive, negative part) * tau * dx dy deta over the
                    common region |x|, |y| <= 10 fm, |eta_s| <= 4.84375, grid / native
    peak            the largest de in that region, grid / native
    err             relative L2 error of de after the grid is resampled back onto MUSIC's
                    points (what the grid lost), in the region
    err_s           the same after a Gaussian smoothing of both (sigma 0.5 fm in x, y and
                    0.3 in eta): the error at the scales of 0.5 fm and up
    bg              the background's sum of e * tau * dV, grid / native

Needs the environment PyJetscape was built in (js_fno on the GB10) for jetscape and scipy;
~1 GB of memory per leg and event at 0-10%.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))

from jetscape.bulk_sources import Grid, resample  # noqa: E402

#: name -> (x and y axis, eta axis) as (min, max, n) cell centres
GRIDS = {
    "new 0.3875": ((-12.20625, 12.20625, 64), (-4.84375, 4.84375, 32)),   # grid_fno.yaml
    "old 0.3125": ((-10.0, 10.0, 65), (-5.0, 5.0, 33)),                   # before Oct 2026
    # MUSIC's own spacing, shifted by half a cell: the floor of interpolating at all
    "0.3 shifted": ((-12.15, 12.15, 82), (-4.9, 4.9, 50)),
}
#: the region every grid covers, for the integrals and errors
XR, ER = 10.0, 4.84375
#: Gaussian sigma in native cells (tau, x, y, eta): 0.5 fm in x, y and 0.3 in eta
SMOOTH = (0, 0.5 / 0.3, 0.5 / 0.3, 0.3 / 0.2)
#: frames shown with --per-event [fm/c]
TAUS = (2.0, 4.0, 6.0, 8.0)
KEYS = ("net", "pos", "neg", "peak", "err", "err_s", "bg")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pair_file", help="pair file written with run_prod_jet.py --native")
    p.add_argument("--per-event", action="store_true", dest="per_event",
                   help="also print every event at tau = 2, 4, 6, 8 fm/c")
    return p.parse_args()


def native_grid(attrs):
    if attrs.get("xscape_grid_mode") != "native":
        sys.exit("wake_grid_compare.py: the file must be on MUSIC's grid "
                 "(run_prod_jet.py --native)")
    axes = [(attrs[f"{a}_min"], attrs[f"{a}_min"] + attrs[f"d{a}"] * (attrs[f"n{a}"] - 1),
             int(attrs[f"n{a}"])) for a in ("x", "y", "eta")]
    return Grid.from_bounds(*axes, tau_min=float(attrs["tau_min"]),
                            dtau=float(attrs["dtau"]), ntau=1)


def region(g):
    x, y, e = g.axis("x"), g.axis("y"), g.axis("eta")
    return ((np.abs(x)[:, None, None] <= XR + 1e-9) & (np.abs(y)[None, :, None] <= XR + 1e-9)
            & (np.abs(e)[None, None, :] <= ER + 1e-9))


def integrals(d, g, m, taus):
    """Sum over the region of d * tau * dx dy deta per frame: (net, positive, negative)."""
    dv = g.dx * g.dy * g.deta
    dm = d * m[None]
    net = dm.sum((1, 2, 3)) * taus * dv
    pos = np.clip(dm, 0, None).sum((1, 2, 3)) * taus * dv
    return net, pos, net - pos


def rel_l2(a, b, m):
    return np.sqrt((((a - b) * m) ** 2).sum((1, 2, 3))
                   / np.maximum(((b * m) ** 2).sum((1, 2, 3)), 1e-30))


def compare_event(f, i, nat):
    """Per-frame measures of event i: {grid: {key: array over frames}}, taus, native wake."""
    from scipy.ndimage import gaussian_filter

    n = int(min(f["ntau_freezeout"][i], f["ntau_freezeout_bg"][i]))      # both legs alive
    ej = np.ascontiguousarray(np.moveaxis(f["arr"][i, 0, ..., :n], -1, 0))
    eb = np.ascontiguousarray(np.moveaxis(f["arr_bg"][i, 0, ..., :n], -1, 0))
    src = replace(nat, ntau=n)
    taus = src.tau_min + src.dtau * np.arange(n)
    d = ej - eb
    del ej
    m = region(src)
    w_nat = integrals(d, src, m, taus)
    bg_nat = integrals(eb, src, m, taus)[0]
    peak_nat = (d * m).max((1, 2, 3))
    d_smooth = gaussian_filter(d, SMOOTH)
    out = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        for name, (xs, es) in GRIDS.items():
            g = Grid.from_bounds(xs, xs, es, tau_min=src.tau_min, dtau=src.dtau, ntau=n)
            d_g = resample(d[..., None], src, g)
            m_g = region(g)
            w = integrals(d_g[..., 0], g, m_g, taus)
            back = resample(d_g, g, src)[..., 0]
            out[name] = {
                "net": w[0] / w_nat[0], "pos": w[1] / w_nat[1], "neg": w[2] / w_nat[2],
                "peak": (d_g[..., 0] * m_g).max((1, 2, 3)) / peak_nat,
                "err": rel_l2(back, d, m),
                "err_s": rel_l2(gaussian_filter(back, SMOOTH), d_smooth, m),
                "bg": integrals(resample(eb[..., None], src, g)[..., 0], g, m_g, taus)[0]
                / bg_nat,
            }
    return out, taus, w_nat


def main():
    a = parse_args()
    import hdf5plugin  # noqa: F401  (Blosc-compressed pair files)

    with h5py.File(a.pair_file, "r") as f:
        nat = native_grid(f.attrs)
        print(f"{a.pair_file}: {f['arr'].shape[0]} events, MUSIC grid dx {nat.dx:.3f} fm, "
              f"deta {nat.deta:.3f}")
        summary = {name: {k: [] for k in KEYS} for name in GRIDS}
        for i in range(f["arr"].shape[0]):
            out, taus, w_nat = compare_event(f, i, nat)
            # a real wake with both parts, so every ratio is finite
            sel = (taus >= 2.0 - 1e-9) & (w_nat[1] >= 1.0) & (w_nat[2] <= -0.5)
            for name in GRIDS:
                for k in KEYS:
                    summary[name][k].append(out[name][k][sel])
            if not a.per_event:
                continue
            print(f"\nevent {i}: {len(taus)} frames with both legs, E_droplets "
                  f"{f['diag/E_droplets'][i]:.1f} GeV")
            print("  tau  wake net/pos/neg [GeV]   " + "   ".join(
                f"{name}: net pos neg peak err" for name in GRIDS))
            for t in TAUS:
                k = int(round((t - taus[0]) / (taus[1] - taus[0])))
                if k >= len(taus):
                    continue
                row = f"  {t:3.0f}  {w_nat[0][k]:6.2f} {w_nat[1][k]:6.2f} {w_nat[2][k]:6.2f}   "
                row += "   ".join(" ".join(f"{out[name][key][k]:.2f}" for key in
                                           ("net", "pos", "neg", "peak", "err"))
                                  for name in GRIDS)
                print(row)

    print("\nall events, frames with tau >= 2 fm/c, wake pos >= 1 GeV and neg <= -0.5 GeV: "
          "median [5%, 95%]")
    for name in GRIDS:
        row = f"  {name:12s}"
        for k in KEYS:
            v = np.concatenate(summary[name][k])
            row += f"  {k} {np.median(v):.3f} [{np.percentile(v, 5):.3f}, {np.percentile(v, 95):.3f}]"
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
