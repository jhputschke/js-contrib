#!/usr/bin/env python3
"""
analysis/OO-RHIC/tune.py -- the Bayesian fit of the p+p 200 GeV hard tune, and the chi-square
cross-check on the simulated design points.

    python tune.py                       # results/summary.json, chains.npz, emulator.pkl, ...
    python tune.py --steps 4000 --walkers 48

Inputs: data/data.json, design/design.json, PROD_DIR/obs/p*.npz (observables.py).

Method.
  * Fit vector: the bins of config.FIT (PHENIX pi0 6-20 GeV, STAR jets R = 0.5 8.2-52 GeV in
    both eta ranges), in log space: r_i = ln(model_i / data_i).
  * Emulator: one Gaussian process per bin (sklearn; constant x anisotropic RBF + white
    noise, the design points' statistical error as known noise), inputs scaled to the unit
    cube; every bin of every observable is emulated (also the validation ones), the fit uses
    the config.FIT bins.  Leave-one-out check over the design points: the emulator's std is
    scaled by the factor that gives the LOO pulls (emulator std (+) the left-out point's
    statistical error) an rms of 1 (never below 1).
  * Likelihood: C = diag((stat^2 + sys^2) / data^2) + diag(emulator variance) +
    norm^2 * 1 1^T per data set (PHENIX 9.7 %, STAR 5.6 %, fully correlated).  The systematic
    errors are taken as uncorrelated between bins (the tables give no correlations).
  * Posterior: uniform priors on config.PARAMETERS, emcee (affine-invariant ensemble).
    MAP by maximising the posterior from the best sample (scipy).
  * Cross-check: chi^2 of every simulated design point with the same covariance (the model's
    statistical error instead of the emulator's), no emulator.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings

import numpy as np

import config as C
import make_design as D
import observables as O

RES = C.HERE / "results"
ALL_OBS = ("phenix_pi0", "star_jets_eta0", "star_jets_eta1", "star_pi0", "star_eta_pi0")
NAMES = list(C.PARAMETERS)
LO = np.array([C.PARAMETERS[k][0] for k in NAMES])
HI = np.array([C.PARAMETERS[k][1] for k in NAMES])


def to_unit(x):
    return (np.asarray(x) - LO) / (HI - LO)


def fit_mask(name, data):
    e = np.asarray(data[name]["edges"])
    if name not in C.FIT:
        return np.zeros(len(e), bool)
    lo, hi = C.FIT[name]
    return (e[:, 0] >= lo - 1e-9) & (e[:, 1] <= hi + 1e-9)


def load_design_obs(data):
    """Design parameters (n, 3) and per observable the model values and errors (n, nbins)."""
    design = D.load()
    names = list(design["points"])
    X = np.array([[design["points"][n]["params"][k] for k in NAMES] for n in names])
    vals = {o: [] for o in ALL_OBS}
    errs = {o: [] for o in ALL_OBS}
    for n in names:
        ob = np.load(C.obs_path(n))
        for o in ALL_OBS:
            v, e = O.combine(ob, o, data)
            vals[o].append(v)
            errs[o].append(e)
    return names, X, {o: np.array(v) for o, v in vals.items()}, \
        {o: np.array(e) for o, e in errs.items()}


class Emulator:
    """Independent GPs on ln(observable) per bin; bins with a non-positive value at any
    design point are not emulated (NaN)."""

    def __init__(self, X, vals, errs):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
        self.u = to_unit(X)
        self.scale = 1.0          # emulator std factor, from the leave-one-out check
        self.models = {}
        for o, V in vals.items():
            E = errs[o]
            ms = []
            for b in range(V.shape[1]):
                v, e = V[:, b], E[:, b]
                if not np.all(v > 0) or not np.all(np.isfinite(v)):
                    ms.append(None)
                    continue
                y = np.log(v)
                noise = np.clip((e / v) ** 2, 1e-8, None)
                mu, sd = y.mean(), y.std() or 1.0
                kern = (ConstantKernel(1.0, (1e-3, 1e3))
                        * RBF(length_scale=np.full(len(NAMES), 0.5),
                              length_scale_bounds=(0.05, 20.0))
                        + WhiteKernel(1e-4, (1e-8, 1e-1)))
                gp = GaussianProcessRegressor(kern, alpha=noise / sd ** 2, normalize_y=False,
                                              n_restarts_optimizer=3, random_state=0)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    gp.fit(self.u, (y - mu) / sd)
                ms.append((gp, mu, sd))
            self.models[o] = ms

    def predict(self, x, obs, return_std=True):
        """ln(observable) at parameter points x (m, 3): mean and std (m, nbins)."""
        u = to_unit(np.atleast_2d(x))
        ms = self.models[obs]
        mean = np.full((len(u), len(ms)), np.nan)
        std = np.full_like(mean, np.nan)
        for b, m in enumerate(ms):
            if m is None:
                continue
            gp, mu, sd = m
            # the predictive std without the white-noise term: the GP's own uncertainty
            k = gp.kernel_
            mtr, s = gp.predict(u, return_std=True)
            white = getattr(k, "k2", None)
            wv = white.noise_level if white is not None and hasattr(white, "noise_level") else 0
            s = np.sqrt(np.clip(s ** 2 - wv, 0, None))
            mean[:, b] = mu + sd * mtr
            std[:, b] = self.scale * sd * s
        return (mean, std) if return_std else mean


class Likelihood:
    def __init__(self, data, emu):
        self.emu = emu
        self.blocks = []
        for o in C.FIT:
            m = fit_mask(o, data)
            d = np.asarray(data[o]["value"])[m]
            st = np.asarray(data[o]["stat"])[m]
            sy = np.asarray(data[o]["sys"])[m]
            ok = np.array([emu.models[o][b] is not None for b in np.where(m)[0]])
            if not ok.all():
                raise ValueError(f"{o}: fit bins without an emulator (zero model values)")
            self.blocks.append((o, m, np.log(d), (st ** 2 + sy ** 2) / d ** 2,
                                data[o]["norm"] or 0.0))
        self.n = sum(int(b[1].sum()) for b in self.blocks)

    def cov_data(self):
        covs = []
        for o, m, ld, rel2, norm in self.blocks:
            covs.append(np.diag(rel2) + norm ** 2 * np.ones((len(ld), len(ld))))
        from scipy.linalg import block_diag
        return block_diag(*covs)

    def residual(self, x):
        r, v = [], []
        for o, m, ld, rel2, norm in self.blocks:
            mean, std = self.emu.predict(x, o)
            r.append(mean[0, m] - ld)
            v.append(std[0, m] ** 2)
        return np.concatenate(r), np.concatenate(v)

    def loglike(self, x):
        r, v = self.residual(x)
        Cm = self.cov_data() + np.diag(v)
        L = np.linalg.cholesky(Cm)
        z = np.linalg.solve(L, r)
        return -0.5 * z @ z - np.log(np.diag(L)).sum()

    def chi2(self, r, extra_var):
        Cm = self.cov_data() + np.diag(extra_var)
        return float(r @ np.linalg.solve(Cm, r))


    def loglike_many(self, X):
        """loglike for several points at once (one GP call per bin for all of them)."""
        X = np.atleast_2d(X)
        rs, vs = [], []
        for o, m, ld, rel2, norm in self.blocks:
            mean, std = self.emu.predict(X, o)
            rs.append(mean[:, m] - ld)
            vs.append(std[:, m] ** 2)
        R, V = np.concatenate(rs, axis=1), np.concatenate(vs, axis=1)
        Cd = self.cov_data()
        out = np.empty(len(X))
        for i in range(len(X)):
            L = np.linalg.cholesky(Cd + np.diag(V[i]))
            z = np.linalg.solve(L, R[i])
            out[i] = -0.5 * z @ z - np.log(np.diag(L)).sum()
        return out


def log_post(x, like):
    """Vectorised (emcee vectorize=True): x is (walkers, ndim)."""
    x = np.atleast_2d(x)
    out = np.full(len(x), -np.inf)
    inside = np.all((x >= LO) & (x <= HI), axis=1)
    if inside.any():
        out[inside] = like.loglike_many(x[inside])
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--walkers", type=int, default=32)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--burn", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args(argv)
    import emcee
    from scipy.optimize import minimize

    RES.mkdir(exist_ok=True)
    data = O.load_data()
    names, X, vals, errs = load_design_obs(data)
    print(f"{len(names)} design points, observables {ALL_OBS}")
    emu = Emulator(X, vals, errs)
    like = Likelihood(data, emu)
    print(f"fit: {like.n} bins")

    # leave-one-out check of the emulator on the fit bins
    loo = []
    for i in range(len(names)):
        keep = np.arange(len(names)) != i
        e_i = Emulator(X[keep], {o: vals[o][keep] for o in C.FIT},
                       {o: errs[o][keep] for o in C.FIT})
        for o in C.FIT:
            m = fit_mask(o, data)
            mean, std = e_i.predict(X[i], o)
            true = np.log(vals[o][i, m])
            loo.append(np.stack([mean[0, m] - true, std[0, m],
                                 errs[o][i, m] / vals[o][i, m]], axis=1))
    loo = np.concatenate(loo)

    def pulls(f):
        return loo[:, 0] / np.sqrt((f * loo[:, 1]) ** 2 + loo[:, 2] ** 2)

    loo_flat = pulls(1.0)
    from scipy.optimize import brentq
    rms = lambda f: np.sqrt(np.mean(pulls(f) ** 2)) - 1.0
    emu.scale = brentq(rms, 1.0, 50.0) if rms(1.0) > 0 else 1.0
    print(f"emulator LOO: pull mean {loo_flat.mean():+.2f}, rms {loo_flat.std():.2f} "
          f"({len(loo_flat)} predictions); emulator std scaled by {emu.scale:.2f} "
          f"(pull rms then {np.sqrt(np.mean(pulls(emu.scale) ** 2)):.2f})")

    # chi^2 of the design points, simulated values (no emulator)
    chi2_design = []
    for i, n in enumerate(names):
        r, v = [], []
        for o, m, ld, rel2, norm in like.blocks:
            r.append(np.log(vals[o][i, m]) - ld)
            v.append((errs[o][i, m] / vals[o][i, m]) ** 2)
        chi2_design.append((n, dict(zip(NAMES, X[i].tolist())),
                            like.chi2(np.concatenate(r), np.concatenate(v))))
    chi2_design.sort(key=lambda t: t[2])
    print("best design points (chi2 / %d bins):" % like.n)
    for n, prm, c in chi2_design[:5]:
        print(f"  {n}: {prm}  chi2 = {c:.1f}")

    # MCMC
    rng = np.random.default_rng(a.seed)
    best0 = np.array(list(chi2_design[0][1].values()))
    p0 = np.clip(best0 + 0.02 * (HI - LO) * rng.standard_normal((a.walkers, len(NAMES))),
                 LO + 1e-6, HI - 1e-6)
    sampler = emcee.EnsembleSampler(a.walkers, len(NAMES), log_post, args=(like,),
                                    vectorize=True)
    sampler.run_mcmc(p0, a.steps, progress=False)
    chain = sampler.get_chain(discard=a.burn, flat=True)
    lp = sampler.get_log_prob(discard=a.burn, flat=True)
    try:
        tau = sampler.get_autocorr_time(discard=a.burn, quiet=True).tolist()
    except Exception:
        tau = None
    acc = float(np.mean(sampler.acceptance_fraction))
    xmax = chain[np.argmax(lp)]
    opt = minimize(lambda x: -log_post(x, like)[0], xmax, method="Nelder-Mead",
                   options={"xatol": 1e-5, "fatol": 1e-4, "maxiter": 4000})
    xmap = opt.x if np.isfinite(opt.fun) and -opt.fun >= lp.max() else xmax
    r, v = like.residual(xmap)
    chi2_map = like.chi2(r, v)
    q = np.percentile(chain, [2.5, 16, 50, 84, 97.5], axis=0)
    summary = {
        "parameters": NAMES, "ranges": {k: list(C.PARAMETERS[k]) for k in NAMES},
        "fit": {k: list(v) for k, v in C.FIT.items()}, "n_bins": like.n,
        "map": dict(zip(NAMES, xmap.tolist())),
        "median": dict(zip(NAMES, q[2].tolist())),
        "ci68": {k: [q[1, i], q[3, i]] for i, k in enumerate(NAMES)},
        "ci95": {k: [q[0, i], q[4, i]] for i, k in enumerate(NAMES)},
        "correlation": np.corrcoef(chain.T).tolist(),
        "chi2_map_emulated": chi2_map, "ndof": like.n - len(NAMES),
        "mcmc": {"walkers": a.walkers, "steps": a.steps, "burn": a.burn,
                 "acceptance": acc, "autocorr_time": tau},
        "emulator_loo": {"pull_mean": float(loo_flat.mean()), "pull_rms": float(loo_flat.std()),
                         "n": int(len(loo_flat)), "std_scale": float(emu.scale)},
        "design_chi2_best": [{"point": n, "params": prm, "chi2": c}
                             for n, prm, c in chi2_design[:10]],
    }
    (RES / "summary.json").write_text(json.dumps(summary, indent=1))
    (RES / "design_chi2.json").write_text(json.dumps(
        [{"point": n, "params": prm, "chi2": c} for n, prm, c in chi2_design], indent=1))
    np.savez(RES / "chains.npz", chain=chain, log_prob=lp, names=np.array(NAMES))
    # posterior predictive of every bin: 400 posterior samples
    pick = chain[rng.choice(len(chain), size=min(400, len(chain)), replace=False)]
    pred = {}
    for o in ALL_OBS:
        mean, std = emu.predict(pick, o)
        pred[o] = mean
        pred[o + "_map"], pred[o + "_map_std"] = emu.predict(xmap, o)
    np.savez(RES / "posterior_predictive.npz", samples=pick, **pred)
    with open(RES / "emulator.pkl", "wb") as f:
        pickle.dump({"emulator": emu, "names": names, "X": X}, f)
    print(f"MAP: {summary['map']}")
    print("68% intervals: " + ", ".join(f"{k} [{q[1, i]:.3g}, {q[3, i]:.3g}]"
                                        for i, k in enumerate(NAMES)))
    print(f"chi2(MAP, emulated) = {chi2_map:.1f} / {summary['ndof']} dof; "
          f"acceptance {acc:.2f}; tau {tau}")
    print(f"-> {RES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
