#!/usr/bin/env python3
"""
example/prod_AuAu_0_10_jet/paired_noise.py

How noisy is jet - background, sample by sample?  Compares two hadronize.py outputs of the
same particlize file: one with independent seeds (the default) and one where the legs are
meant to be correlated (--common-seeds, or later correlated sampling in iSS;
PLAN_iSS_optim.md, Part B).

    python paired_noise.py out/ out_common/ --stem AuAu_0_10_jet_seed0001

For each bin of a few charged-hadron observables it prints, over the oversamples k of
the first unit of each file (the first event and its background):

    <J>        mean jet-leg count per sample
    <J-B>      mean difference, independent and correlated (the signal: must agree)
    var        variance of the per-sample difference J_k - B_k, independent and correlated
    ratio      var correlated / var independent: the gain (1 = none, 0 = all noise gone)
    rho        correlation of J_k and B_k in the correlated pair

With N samples a variance is known to ~sqrt(2/N) (6% for 500), so a ratio of 0.9-1.1 is no
gain.  Both pairs need the same number of samples on both legs.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "python"))

from jetscape.hadrons_h5 import Hadrons  # noqa: E402

# name -> (variable, bin edges, restrict to |eta| < 1)
OBSERVABLES = {
    "charged, |eta| < 1": ("eta", np.array([-1.0, 1.0]), True),
    "charged pT [GeV], |eta| < 1": ("pt", np.array([0, 0.5, 1, 1.5, 2, 3, 5]), True),
    "charged eta": ("eta", np.linspace(-4, 4, 9), False),
    "charged phi, |eta| < 1": ("phi", np.linspace(-np.pi, np.pi, 9), True),
}


def per_sample_counts(h, var, bins, midrap):
    """(samples, bins) counts of charged hadrons (``h`` holds one unit)."""
    m = h.charged.copy()
    if midrap:
        m &= np.abs(h.eta) < 1
    ns = int(h.samples_per_unit[0]) if h.n_units else 0
    nb = len(bins) - 1
    b = np.digitize(getattr(h, var)[m], bins) - 1
    ok = (b >= 0) & (b < nb)
    idx = h.sample[m][ok] * nb + b[ok]
    return np.bincount(idx, minlength=ns * nb).reshape(ns, nb).astype(float)


def pair_stats(directory, stem):
    jet = Hadrons.from_h5(os.path.join(directory, f"{stem}_hadrons_bulk_jet.h5"), units=[0])
    bg = Hadrons.from_h5(os.path.join(directory, f"{stem}_hadrons_bulk_bg.h5"), units=[0])
    out = {}
    for name, (var, bins, midrap) in OBSERVABLES.items():
        J = per_sample_counts(jet, var, bins, midrap)
        B = per_sample_counts(bg, var, bins, midrap)
        if J.shape != B.shape:
            sys.exit(f"paired_noise.py: {directory}: {J.shape[0]} jet vs {B.shape[0]} "
                     "background samples; both legs need the same number")
        D = J - B
        rho = np.array([np.corrcoef(J[:, i], B[:, i])[0, 1] for i in range(J.shape[1])])
        out[name] = {"J": J.mean(0), "D": D.mean(0), "var": D.var(0, ddof=1), "rho": rho,
                     "n": J.shape[0]}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("independent", help="hadronize.py output directory, independent seeds")
    p.add_argument("correlated", help="the same with correlated legs (e.g. --common-seeds)")
    p.add_argument("--stem", default="AuAu_0_10_jet_seed0001")
    a = p.parse_args(argv)
    ind, cor = pair_stats(a.independent, a.stem), pair_stats(a.correlated, a.stem)
    n = next(iter(cor.values()))["n"]
    print(f"{a.stem}: {n} samples, first unit; variance known to ~{np.sqrt(2 / (n - 1)):.0%}")
    for name, (var, bins, _) in OBSERVABLES.items():
        i, c = ind[name], cor[name]
        print(f"\n{name}")
        print(f"{'bin':>15} {'<J>':>8} {'<J-B> ind':>10} {'<J-B> cor':>10} {'var ind':>9} "
              f"{'var cor':>9} {'ratio':>6} {'rho':>6}")
        for k in range(len(bins) - 1):
            print(f"{bins[k]:7.2f}-{bins[k + 1]:<7.2f} {i['J'][k]:8.1f} {i['D'][k]:10.2f} "
                  f"{c['D'][k]:10.2f} {i['var'][k]:9.1f} {c['var'][k]:9.1f} "
                  f"{c['var'][k] / i['var'][k]:6.3f} {c['rho'][k]:6.3f}")


if __name__ == "__main__":
    main()
