#!/usr/bin/env python3
"""
analysis/OO-RHIC/hard_recheck.py -- two runs of the hard tune's observables compared bin by bin
(a paired check: same seeds, one setting changed), and each run's chi2 against the fit bins.

    python hard_recheck.py [NEW OLD]      # default ref_tune ref_mapbulk; observables.py first

Made for the recheck of 2026-11-24: the tune's reference point (alpha_s 0.15, vir_factor 0.25,
Q0 2) with the new 3dMCGlauber block of config/pp200_isr_vacuum.xml (the RHIC bulk tune) and with
the old one (the JETSCAPE MAP; make_design.py --template with the file of git a3bb4c2,
--seeds-of ref_tune).  Per observable the ratio new/old in the fit bins (or all bins of a
validation observable) with the largest pull, and chi2 with tune.py's data covariance (stat+sys
per bin, normalisation fully correlated) plus the model's statistical errors.  Result: all
ratios within 1.2 sigma, chi2 35.4 (new) vs 33.8 (old) over 35 bins: the hard tune stands.
Model results, not vetted physics results.
"""
import sys
import numpy as np
from scipy.linalg import block_diag
sys.path.insert(0, ".")
import config as C
import observables as O
import tune as T

data = O.load_data()
runs = tuple(sys.argv[1:3]) if len(sys.argv) > 2 else ("ref_tune", "ref_mapbulk")
ob = {r: np.load(C.obs_path(r)) for r in runs}
chi = {r: [] for r in runs}
print(f"{'observable':16s} {'bins':>4s}  new/old per fit bin (pT)  -> mean, max |pull|")
for o in T.ALL_OBS:
    m = T.fit_mask(o, data)
    e = np.asarray(data[o]["edges"])
    v = {r: O.combine(ob[r], o, data) for r in runs}
    (a, ea), (b, eb) = v[runs[0]], v[runs[1]]
    ratio = a / b
    rerr = ratio * np.sqrt((ea / a) ** 2 + (eb / b) ** 2)
    pull = (ratio - 1) / rerr
    sel = m if m.any() else np.isfinite(ratio)
    print(f"{o:16s} {int(sel.sum()):4d}  " + " ".join(f"{x:.3f}" for x in ratio[sel])
          + f"  -> {np.nanmean(ratio[sel]):.3f}, {np.nanmax(np.abs(pull[sel])):.1f} sigma")
    if o in C.FIT:
        d = np.asarray(data[o]["value"])[m]
        rel2 = (np.asarray(data[o]["stat"])[m] ** 2 + np.asarray(data[o]["sys"])[m] ** 2) / d ** 2
        norm = data[o]["norm"] or 0.0
        for r in runs:
            val, err = v[r]
            res = np.log(val[m]) - np.log(d)
            cov = np.diag(rel2 + (err[m] / val[m]) ** 2) + norm ** 2 * np.ones((m.sum(), m.sum()))
            chi[r].append((o, int(m.sum()), float(res @ np.linalg.solve(cov, res))))
print()
for r in runs:
    tot = sum(c for _, _, c in chi[r]); n = sum(k for _, k, _ in chi[r])
    print(f"{r:12s} chi2 vs data {tot:6.1f} / {n} bins   " + ", ".join(f"{o} {c:.1f}/{k}" for o, k, c in chi[r]))
