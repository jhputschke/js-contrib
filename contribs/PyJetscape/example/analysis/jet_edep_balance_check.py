#!/usr/bin/env python3
"""Jet energy balance: initial partons = surviving partons + deposited droplets.

Reads the shower graph (``shower/partons``) and the droplet table (``source/droplets``) of
PairH5Writer files from ``run_prod_jet.py`` and checks, per event, the three identities of
§11 in ``prod_AuAu_0_10_jet/jet_wake.ipynb``:

(a) the graph closes:  E_ini + E(-17) = E(0, 1, 22) - E(-1) + E(-11) + E(-13)
(b) the droplet table = E(-11) + E(-13) - E(-17) + free-streaming droplets
(c) E_ini = E_surviving + sum E_droplet - E_double

pstat as the liquefier leaves it (jetscape.particlize_h5.PSTAT_NOTE): 0 shower parton,
1 recoil, 22 photon, -1 hole not absorbed (counts negative), -11 absorbed, -13 missing
four-momentum of a vertex, -17 absorbed hole (enters its droplet with -E).

E_double are surviving partons that are also a droplet: LBT free-streams a parton
(``pOutTemp.size() == 1`` in JetEnergyLoss::DoExecTime), the liquefier drops it below
e_threshold and deposits its whole four-momentum, but no graph edge is written in that branch,
so the parton keeps pstat 1 and is hadronized as well.  They are found as table droplets whose
four-momentum equals a surviving leaf parton's.

Usage::

    python jet_edep_balance_check.py FILE.h5 [FILE.h5 ...] [--per-event]
"""

import argparse
import sys

import h5py
import numpy as np

try:  # Blosc, the default compression since h5_optim
    import hdf5plugin  # noqa: F401
except ImportError:
    pass


def balance(f, ev):
    """Energy bookkeeping of event ``ev`` of an open pair file, as a dict of GeV sums."""
    g = f["shower"]
    a, b = g["parton_offsets"][ev:ev + 2]
    P = g["partons"][a:b]            # shower, i_src, i_tgt, pid, pstat, px, py, pz, E, x, y, z, t
    a, b = g["initiator_offsets"][ev:ev + 2]
    I = g["initiators"][a:b]         # shower, pid, pstat, px, py, pz, E, x, y, z, t
    a, b = f["source/offsets"][ev:ev + 2]
    D = f["source/droplets"][a:b]    # tau, x, y, eta, E, px, py, pz

    src, tgt, st = P[:, 1].astype(int), P[:, 2].astype(int), P[:, 4]
    E, pmu = P[:, 8], P[:, [8, 5, 6, 7]]
    hole = st == -17                 # an edge from its own root INTO the vertex
    leaf = ~np.isin(tgt, src) & ~hole
    surv = leaf & np.isin(st, (0, 1, 22, -1))
    signed = np.where(st == -1, -E, E)

    double = np.zeros(len(P), bool)
    for d in D:
        same = np.all(np.isclose(pmu, d[4:8], rtol=1e-5, atol=1e-5), axis=1)
        hit = np.flatnonzero(surv & (st != -1) & same)
        double[hit[:1]] = True

    return dict(E_ini=I[:, 6].sum(), E_surv=signed[surv].sum(), E_drop=D[:, 4].sum(),
                E_11=E[leaf & (st == -11)].sum(), E_13=E[leaf & (st == -13)].sum(),
                E_17=E[hole].sum(), E_double=E[double].sum(), n_double=int(double.sum()))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="+", help="PairH5Writer files (with a shower/ group)")
    ap.add_argument("--per-event", action="store_true", help="print one line per event")
    args = ap.parse_args(argv)

    rows = []
    for path in args.files:
        with h5py.File(path, "r") as f:
            if "shower" not in f:
                print(f"skip {path}: no shower/ group", file=sys.stderr)
                continue
            for ev in range(len(f["ntau_freezeout"])):
                r = balance(f, ev)
                rows.append(r)
                if args.per_event:
                    naive = r["E_surv"] + r["E_drop"] - r["E_ini"]
                    print(f"{path} ev {ev:3d}: E_ini {r['E_ini']:8.3f}  surv+drop-ini "
                          f"{naive:+8.3f}  double {r['n_double']:2d} ({r['E_double']:7.3f} GeV)  "
                          f"residual {naive - r['E_double']:+.1e}")
    if not rows:
        sys.exit("no events with a shower graph")

    R = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    graph = R["E_surv"] + R["E_11"] + R["E_13"] - R["E_17"] - R["E_ini"]
    table = R["E_11"] + R["E_13"] - R["E_17"] + R["E_double"] - R["E_drop"]
    naive = R["E_surv"] + R["E_drop"] - R["E_ini"]
    fixed = naive - R["E_double"]
    print(f"{len(rows)} events, <E_ini> = {R['E_ini'].mean():.1f} GeV")
    print(f"(a) graph closes           max |resid| {np.abs(graph).max():.1e} GeV")
    print(f"(b) droplet table matches  max |resid| {np.abs(table).max():.1e} GeV")
    print(f"(c) surv + drop - ini      mean {naive.mean():+.3f} GeV "
          f"({np.mean(naive / R['E_ini']):+.1%}), max |.| {np.abs(naive).max():.3f} GeV")
    print(f"    minus double-counted   max |resid| {np.abs(fixed).max():.1e} GeV")
    print(f"double-counted partons: {R['n_double'].sum()} in {(R['n_double'] > 0).sum()} of "
          f"{len(rows)} events, E_double / E_ini mean {np.mean(R['E_double'] / R['E_ini']):.1%}, "
          f"max {np.max(R['E_double'] / R['E_ini']):.1%}")


if __name__ == "__main__":
    main()
