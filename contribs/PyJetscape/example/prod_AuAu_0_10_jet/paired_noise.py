#!/usr/bin/env python3
"""
example/prod_AuAu_0_10_jet/paired_noise.py

How noisy is jet - background, sample by sample, and is each leg still the same physics?
Compares two hadronize.py outputs of the same particlize file: one with independent seeds
(the default) and one with correlated legs (--correlated, or --common-seeds;
PLAN_iSS_optim.md, Part B).

    python paired_noise.py out/ out_correlated/ --stem AuAu_0_10_jet_seed0001

Part 1, jet - background.  For each bin of a few charged-hadron observables, over the
oversamples k of the first unit of each file (the first event and its background):

    <J>        mean jet-leg count per sample
    <J-B>      mean difference, independent and correlated (the signal: must agree)
    var        variance of the per-sample difference J_k - B_k, independent and correlated
    ratio      var correlated / var independent: the gain (1 = none, 0 = all noise gone);
               the same precision takes `ratio` times the oversamples
    rho        correlation of J_k and B_k in the correlated pair

Part 2, each leg alone.  Correlated sampling must not change a leg's physics: per bin the
mean of each leg in the two runs (z = difference / its error) and the ratio of the
per-sample variances (1 within ~sqrt(4/N)), plus identified yields and sum cos 2phi.

With N samples a variance is known to ~sqrt(2/N) (6% for 500).  Both runs need the same
number of samples on both legs.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "python"))

from jetscape.hadrons_h5 import SPECIES, Hadrons  # noqa: E402

# name -> (variable, bin edges, restrict to |eta| < 1): charged-hadron counts per bin
OBSERVABLES = {
    "charged, |eta| < 1": ("eta", np.array([-1.0, 1.0]), True),
    "charged pT [GeV], |eta| < 1": ("pt", np.array([0, 0.5, 1, 1.5, 2, 3, 5]), True),
    "charged eta": ("eta", np.linspace(-4, 4, 9), False),
    "charged phi, |eta| < 1": ("phi", np.linspace(-np.pi, np.pi, 9), True),
}
LEG_SPECIES = ("pi+", "pi-", "K+", "K-", "p", "pbar")


def per_sample(h, mask, bin_index, nbins, weights=None):
    """(samples, bins) sums over the hadrons in ``mask`` (``h`` holds one unit)."""
    ns = int(h.samples_per_unit[0]) if h.n_units else 0
    b = bin_index[mask]
    ok = (b >= 0) & (b < nbins)
    idx = h.sample[mask][ok] * nbins + b[ok]
    w = None if weights is None else weights[mask][ok]
    return np.bincount(idx, weights=w, minlength=ns * nbins).reshape(ns, nbins)


def binned(h, var, bins, midrap):
    m = h.charged.copy()
    if midrap:
        m &= np.abs(h.eta) < 1
    return per_sample(h, m, np.digitize(getattr(h, var), bins) - 1, len(bins) - 1)


def leg_observables(h):
    """name -> (samples, bins) for part 2: the binned counts, yields and sum cos 2phi."""
    out = {n: binned(h, v, b, mid) for n, (v, b, mid) in OBSERVABLES.items()}
    mid = np.abs(h.eta) < 1
    zero = np.zeros(len(h.pid), dtype=int)
    out["identified yields, |eta| < 1"] = np.concatenate(
        [per_sample(h, mid & np.isin(h.pid, SPECIES[s]), zero, 1) for s in LEG_SPECIES], 1)
    m = h.charged & mid
    out["charged sum cos 2phi, sin 2phi, |eta| < 1"] = np.concatenate(
        [per_sample(h, m, zero, 1, np.cos(2 * h.phi)),
         per_sample(h, m, zero, 1, np.sin(2 * h.phi))], 1)
    return out


LEG_LABELS = {"identified yields, |eta| < 1": list(LEG_SPECIES),
              "charged sum cos 2phi, sin 2phi, |eta| < 1": ["cos 2phi", "sin 2phi"]}


def load(directory, stem):
    legs = {}
    for leg in ("jet", "bg"):
        h = Hadrons.from_h5(os.path.join(directory, f"{stem}_hadrons_bulk_{leg}.h5"),
                            units=[0])
        legs[leg] = leg_observables(h)
    n_jet = next(iter(legs["jet"].values())).shape[0]
    n_bg = next(iter(legs["bg"].values())).shape[0]
    if n_jet != n_bg:
        sys.exit(f"paired_noise.py: {directory}: {n_jet} jet vs {n_bg} background samples; "
                 "both legs need the same number")
    return legs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("independent", help="hadronize.py output directory, independent seeds")
    p.add_argument("correlated", help="the same with correlated legs (--correlated)")
    p.add_argument("--stem", default="AuAu_0_10_jet_seed0001")
    a = p.parse_args(argv)
    ind, cor = load(a.independent, a.stem), load(a.correlated, a.stem)
    n = next(iter(cor["jet"].values())).shape[0]
    print(f"{a.stem}: {n} samples, first unit; a variance is known to "
          f"~{np.sqrt(2 / (n - 1)):.0%}")

    print("\n=== 1. jet - background, per sample")
    for name, (var, bins, _) in OBSERVABLES.items():
        Ji, Bi, Jc, Bc = (r[leg][name] for r in (ind, cor) for leg in ("jet", "bg"))
        Di, Dc = Ji - Bi, Jc - Bc
        vi, vc = Di.var(0, ddof=1), Dc.var(0, ddof=1)
        print(f"\n{name}")
        print(f"{'bin':>15} {'<J>':>8} {'<J-B> ind':>10} {'<J-B> cor':>10} {'var ind':>9} "
              f"{'var cor':>9} {'ratio':>6} {'rho':>6}")
        for k in range(len(bins) - 1):
            rho = np.corrcoef(Jc[:, k], Bc[:, k])[0, 1]
            print(f"{bins[k]:7.2f}-{bins[k + 1]:<7.2f} {Ji[:, k].mean():8.1f} "
                  f"{Di[:, k].mean():10.2f} {Dc[:, k].mean():10.2f} {vi[k]:9.1f} "
                  f"{vc[k]:9.1f} {vc[k] / vi[k]:6.3f} {rho:6.3f}")

    print("\n=== 2. each leg alone: correlated vs independent sampling")
    print("z = (mean correlated - mean independent) / error; var ratio = correlated / "
          f"independent (1 within ~{np.sqrt(4 / (n - 1)):.0%})")
    zs = []
    for name in ind["jet"]:
        labels = LEG_LABELS.get(name)
        if labels is None:
            bins = OBSERVABLES[name][1]
            labels = [f"{bins[k]:.2f}-{bins[k + 1]:.2f}" for k in range(len(bins) - 1)]
        print(f"\n{name}")
        print(f"{'bin':>15} {'jet <X> ind':>12} {'z':>6} {'var ratio':>9} "
              f"{'bg <X> ind':>12} {'z':>6} {'var ratio':>9}")
        for k, label in enumerate(labels):
            row = f"{label:>15}"
            for leg in ("jet", "bg"):
                xi, xc = ind[leg][name][:, k], cor[leg][name][:, k]
                err = np.sqrt(xi.var(ddof=1) / len(xi) + xc.var(ddof=1) / len(xc))
                z = (xc.mean() - xi.mean()) / err if err > 0 else 0.0
                zs.append(z)
                ratio = xc.var(ddof=1) / xi.var(ddof=1) if xi.var() > 0 else float("nan")
                row += f" {xi.mean():12.2f} {z:6.2f} {ratio:9.3f}"
            print(row)
    zs = np.array(zs)
    print(f"\n{len(zs)} leg means: chi2/ndf = {np.sum(zs ** 2) / len(zs):.2f}, "
          f"max |z| = {np.max(np.abs(zs)):.2f} (bins overlap, so they are not independent)")


if __name__ == "__main__":
    main()
