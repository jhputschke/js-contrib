#!/usr/bin/env python3
"""
analysis/OO-RHIC/conditional.py -- the tune's posterior with one parameter held fixed (emulator
only, no new runs): e.g. Q0 = 2 GeV for O+O with Matter + LBT, where Q0 is also the Matter -> LBT
handover of the AA tune.

    python conditional.py                     # Q0 = 2.0 -> results/conditional_Q0_2.json
    python conditional.py --fix Q0=1.5
    python conditional.py --fix Q0=2.0 --fix Cv=0.25

Uses tune.py's emulator (results/emulator.pkl, with its LOO-scaled std) and likelihood; emcee
over the free parameters inside config.PARAMETERS. Reports MAP, median, 68 / 95 % ranges,
chi2 (emulated) at the MAP, and the valley alpha_s(Cv) at the fixed values.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys

import numpy as np

import config as C
import observables as O
import tune as T


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fix", action="append", default=None, help="NAME=VALUE (repeatable)")
    p.add_argument("--walkers", type=int, default=32)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--burn", type=int, default=800)
    p.add_argument("--seed", type=int, default=2)
    a = p.parse_args(argv)
    fix = dict(f.split("=") for f in (a.fix or ["Q0=2.0"]))
    fix = {k: float(v) for k, v in fix.items()}
    names = list(T.NAMES)
    if any(k not in names for k in fix):
        sys.exit(f"conditional.py: unknown parameter in {fix}; known {names}")
    free = [k for k in names if k not in fix]
    ifree = [names.index(k) for k in free]
    lo, hi = T.LO[ifree], T.HI[ifree]

    sys.modules["__main__"].Emulator = T.Emulator          # unpickle tune.py's class
    emu = pickle.load(open(T.RES / "emulator.pkl", "rb"))["emulator"]
    like = T.Likelihood(O.load_data(), emu)

    def full(xf):
        xf = np.atleast_2d(xf)
        X = np.empty((len(xf), len(names)))
        for k, v in fix.items():
            X[:, names.index(k)] = v
        X[:, ifree] = xf
        return X

    def log_post(xf):
        xf = np.atleast_2d(xf)
        out = np.full(len(xf), -np.inf)
        ok = np.all((xf >= lo) & (xf <= hi), axis=1)
        if ok.any():
            out[ok] = like.loglike_many(full(xf[ok]))
        return out

    import emcee
    from scipy.optimize import minimize
    rng = np.random.default_rng(a.seed)
    # start near the unconditional median, projected on the free parameters
    med = json.loads((T.RES / "summary.json").read_text())["median"]
    x0 = np.array([med[k] for k in free])
    p0 = np.clip(x0 + 0.02 * (hi - lo) * rng.standard_normal((a.walkers, len(free))),
                 lo + 1e-6, hi - 1e-6)
    s = emcee.EnsembleSampler(a.walkers, len(free), log_post, vectorize=True)
    s.run_mcmc(p0, a.steps, progress=False)
    chain = s.get_chain(discard=a.burn, flat=True)
    lp = s.get_log_prob(discard=a.burn, flat=True)
    xmax = chain[np.argmax(lp)]
    opt = minimize(lambda x: -log_post(x)[0], xmax, method="Nelder-Mead",
                   options={"xatol": 1e-5, "fatol": 1e-4, "maxiter": 4000})
    xmap = opt.x if np.isfinite(opt.fun) and -opt.fun >= lp.max() else xmax
    r, v = like.residual(full(xmap)[0])
    chi2 = like.chi2(r, v)
    q = np.percentile(chain, [2.5, 16, 50, 84, 97.5], axis=0)
    out = {"fixed": fix, "free": free, "n_bins": like.n, "ndof": like.n - len(free),
           "map": dict(zip(free, xmap.tolist())), "median": dict(zip(free, q[2].tolist())),
           "ci68": {k: [q[1, i], q[3, i]] for i, k in enumerate(free)},
           "ci95": {k: [q[0, i], q[4, i]] for i, k in enumerate(free)},
           "chi2_map_emulated": chi2,
           "acceptance": float(np.mean(s.acceptance_fraction))}
    if "alpha_s" in free and "Cv" in free:              # the valley at the fixed values
        ia, ic = free.index("alpha_s"), free.index("Cv")
        edges = np.linspace(lo[ic], hi[ic], 8)
        valley = []
        for c0, c1 in zip(edges[:-1], edges[1:]):
            m = (chain[:, ic] >= c0) & (chain[:, ic] < c1)
            if m.sum() > 50:
                valley.append([0.5 * (c0 + c1), float(np.median(chain[m, ia])),
                               float(np.std(chain[m, ia])), int(m.sum())])
        out["valley_alpha_s_vs_Cv"] = valley
        out["valley_fit"] = np.polyfit([v[0] for v in valley], [v[1] for v in valley], 1).tolist()
    tag = "_".join(f"{k}_{v:g}" for k, v in fix.items())
    path = T.RES / f"conditional_{tag}.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("fixed", "map", "median", "ci68", "ci95",
                                          "chi2_map_emulated", "ndof")}, indent=1))
    if "valley_alpha_s_vs_Cv" in out:
        print("valley alpha_s(Cv):", " ".join(f"{c:.2f}:{m:.4f}" for c, m, _, _ in
                                               out["valley_alpha_s_vs_Cv"]),
              "| fit alpha_s = %.4f + %.4f (Cv - 0.16)" % (
                  out["valley_fit"][1] + out["valley_fit"][0] * 0.16, out["valley_fit"][0]))
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
