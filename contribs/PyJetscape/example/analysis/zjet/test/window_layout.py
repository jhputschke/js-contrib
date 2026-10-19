#!/usr/bin/env python3
"""Window layouts for the Z-hadron selections, from a vacuum reference: every event carries its
pTHat, so a sub-range [a, b) of the generated windows has sigma = sum_k sigma_k x (fraction of
window k's events in [a, b)) and its own passing fraction.  Equal events per window (as
--jets-per-bin), N events in all; N_eff = (sum sigma f)^2 / sum(sigma^2 f / n)."""
import itertools
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))       # jetcorr
sys.path.insert(0, os.path.dirname(HERE))                        # zjet_wake.py, cms2507.py
try:
    import jetscape  # noqa: F401
except ImportError:                                              # the source tree
    sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "..", "python"))
import numpy as np
import zjet_wake as zw
from jetcorr import runs
from jetscape.hadrons_h5 import HadronFileReader
ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--vacuum", default=runs.data("zjet_test", "vacuum_nocut"),
                help="a vacuum reference whose windows cover 20-200 GeV (make_data.sh vacuum_nocut)")
a = ap.parse_args()
r = HadronFileReader(a.vacuum); t = zw.event_table(r)
pth = np.array([r.event_info(g).pthat for g in range(r.n_events)])
k = t["pthat_bin"]
sig_k = np.array([r.pthat_bin_sigma(i)[0] for i in range(len(r.pthat_bins))]) * 1e9   # pb
nk = np.bincount(k, minlength=len(sig_k))
dens = sig_k[k] / nk[k]                              # sigma per event (pb), the sample's weight
lep = lambda eta: (np.abs(t["lep1_eta"]) < eta) & (np.abs(t["lep2_eta"]) < eta)
S2507 = (t["z_pt"] > 40) & (t["z_pt"] < 350) & (np.abs(t["z_y"]) < 2.4) & (t["z_m"] > 60) & (t["z_m"] < 120) & lep(2.4)
S2103 = (t["z_pt"] > 30) & (t["z_m"] > 60) & (t["z_m"] < 120) & lep(2.4)
SATL = (t["z_pt"] > 30) & (t["z_m"] > 76) & (t["z_m"] < 106) & lep(2.5)
N, KISH = 96000, 0.848
def neff(edges, sel):
    num = den = 0.0
    n = N / (len(edges) - 1)
    for a, b in zip(edges[:-1], edges[1:]):
        inw = (pth >= a) & (pth < b)
        sW = dens[inw].sum()
        if sW <= 0: return 0.0
        f = dens[inw & sel].sum() / sW
        num += sW * f; den += sW**2 * f / n
    return num**2 / den * KISH
def report(name, edges):
    print(f"{name:34s} {'-'.join(f'{e:g}' for e in edges):32s} 2507 {neff(edges, S2507):7.0f}  2103 {neff(edges, S2103):7.0f}  ATLAS>30 {neff(edges, SATL):7.0f}")
print(f"{r.n_events} vacuum events; N_eff (x N_coll Kish {KISH}) for {N} events, equal per window")
report("current", [20, 30, 40, 60, 80, 120, 200])
cands = [20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 100, 120, 150, 200]
best = {}
for K in (5, 6):
    for inner in itertools.combinations(cands[1:-1], K - 1):
        e = [20, *inner, 200]
        for nm, s in (("2507", S2507), ("2103", S2103)):
            v = neff(e, s)
            if v > best.get((K, nm), (0, None))[0]:
                best[(K, nm)] = (v, e)
for (K, nm), (v, e) in sorted(best.items()):
    report(f"best {K} windows for {nm}", e)
for e in ([20, 30, 40, 50, 60, 80, 200], [20, 30, 35, 40, 50, 60, 200], [20, 30, 40, 50, 60, 80, 120], [25, 30, 40, 50, 60, 80, 200]):
    report("candidate", e)
