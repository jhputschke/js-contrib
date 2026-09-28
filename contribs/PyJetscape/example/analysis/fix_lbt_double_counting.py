#!/usr/bin/env python3
"""Patch the pstat of partons that runs before the LBT/liquefier fix counted twice.

Before X-SCAPE's fix (branch fix_lbt_liquifer_double_counting, JetEnergyLoss::DoExecTime),
a parton LBT free-streamed and the liquefier then absorbed was deposited in full as a
droplet but kept pstat 0/1 in the shower graph. It stayed a final-state parton and was
hadronized as well. The hydro, the droplets and the surfaces of those runs are correct: a
rerun with the fixed build and the same seeds is bit-identical except for the pstat of these
partons, which the fix sets to -11. This script makes the same change in the stored files.

A double-counted parton is a surviving (pstat 0, 1, 22) final parton whose four-momentum
equals a droplet of the same event, as in jet_edep_balance_check.py. It is set to -11 in

  <stem>.h5             shower/partons   (the shower graph; column 4 = pstat)
  <stem>_particlize.h5  partons/data     (the final partons hadronize.py fragments; column 2)

so ColorlessHadronization no longer takes it. Both files must pick the same partons per
event, otherwise neither is written. The changed rows and their old pstat are stored next to
each table (lbt_fix_rows, lbt_fix_pstat_before) and the file gets the attribute
lbt_double_count_fix, so a file is patched only once and --revert undoes it. After patching,
each event must satisfy E_initial = E_surviving + E_droplets (graph) to 1e-4 GeV.

Only the parton tables are rewritten (a few kB per file); the hydro and surfaces are not
touched. Without --apply nothing is written.

Usage::

    python fix_lbt_double_counting.py DIR_OR_PAIR_H5 [...]            # report only
    python fix_lbt_double_counting.py DIR_OR_PAIR_H5 [...] --apply    # patch in place
    python fix_lbt_double_counting.py DIR_OR_PAIR_H5 [...] --revert   # undo a patch

then re-run hadronize.py on the _particlize.h5 files (jet_frag changes; bulk_jet and
bulk_bg come from the surfaces and do not).
"""

import argparse
import glob
import os
import re
import sys

import h5py
import numpy as np

try:  # Blosc, the default compression since h5_optim
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

ATTR = "lbt_double_count_fix"
NOTE = ("pstat 0/1 -> -11 for final partons LBT free-streamed and the liquefier absorbed "
        "(deposited in full as a droplet, but left in the final state before the X-SCAPE "
        "fix fix_lbt_liquifer_double_counting); rows and old pstat in lbt_fix_rows / "
        "lbt_fix_pstat_before")
DROP_STAT = -11
SURVIVING = (0, 1, 22)
TOL = dict(rtol=1e-5, atol=1e-5)   # droplets are stored from float32

# (group, table, pstat column, four-momentum columns E, px, py, pz)
GRAPH = ("shower", "partons", 4, [8, 5, 6, 7])
FINAL = ("partons", "data", 2, [3, 4, 5, 6])


def matches(rows, pcols, drops):
    """Indices into ``rows`` whose four-momentum equals a droplet's (at most one per droplet)."""
    hit = set()
    for d in drops:
        same = np.flatnonzero(np.all(np.isclose(rows[:, pcols], d[4:8], **TOL), axis=1))
        same = [k for k in same if k not in hit]
        if same:
            hit.add(same[0])
    return np.array(sorted(hit), dtype=np.int64)


def find_double(pair, part):
    """Global row indices to set to -11 in the graph and in the final partons, per file."""
    doff, drops = pair["source/offsets"][:], pair["source/droplets"][:]
    goff, graph = pair["shower/parton_offsets"][:], pair["shower/partons"][:]
    n_ev = len(pair["ntau_freezeout"])
    if part is not None:
        foff, final = part["partons/offsets"][:], part["partons/data"][:]
        if len(foff) - 1 != n_ev:
            raise ValueError(f"{n_ev} events in the pair file, {len(foff) - 1} in particlize")

    g_rows, f_rows, per_event = [], [], []
    for ev in range(n_ev):
        D = drops[doff[ev]:doff[ev + 1]]
        a = goff[ev]
        Q = graph[goff[ev]:goff[ev + 1]]
        src, tgt, st = Q[:, 1].astype(int), Q[:, 2].astype(int), Q[:, 4]
        leaf_surv = np.flatnonzero(~np.isin(tgt, src) & np.isin(st, SURVIVING))
        g = leaf_surv[matches(Q[leaf_surv], GRAPH[3], D)]
        g_rows.append(a + g)
        per_event.append(len(g))
        if part is not None:
            b = foff[ev]
            F = final[foff[ev]:foff[ev + 1]]
            surv = np.flatnonzero(np.isin(F[:, 2], SURVIVING))
            fr = surv[matches(F[surv], FINAL[3], D)]
            if len(fr) != len(g) or not np.allclose(
                    np.sort(F[fr][:, FINAL[3]], axis=0), np.sort(Q[g][:, GRAPH[3]], axis=0)):
                raise ValueError(f"event {ev}: graph and particlize disagree "
                                 f"({len(g)} vs {len(fr)} partons)")
            f_rows.append(b + fr)
    cat = lambda r: np.concatenate(r) if r else np.zeros(0, np.int64)
    return cat(g_rows), (cat(f_rows) if part is not None else None), np.array(per_event)


def check_balance(pair):
    """max |E_ini - E_surviving - E_droplets| over events, from the (patched) graph."""
    worst = 0.0
    for ev in range(len(pair["ntau_freezeout"])):
        g = pair["shower"]
        a, b = g["parton_offsets"][ev:ev + 2]
        Q = g["partons"][a:b]
        a, b = g["initiator_offsets"][ev:ev + 2]
        e_ini = g["initiators"][a:b, 6].sum()
        a, b = pair["source/offsets"][ev:ev + 2]
        e_drop = pair["source/droplets"][a:b, 4].sum()
        src, tgt, st = Q[:, 1].astype(int), Q[:, 2].astype(int), Q[:, 4]
        leaf = ~np.isin(tgt, src) & (st != -17)
        e_surv = Q[leaf & np.isin(st, SURVIVING), 8].sum() - Q[leaf & (st == -1), 8].sum()
        worst = max(worst, abs(e_ini - e_surv - e_drop))
    return worst


def patch(f, spec, rows):
    grp, table, col, _ = spec
    ds = f[grp][table]
    before = ds[:, col][rows] if len(rows) else np.zeros(0)
    for r in rows:                       # a few hundred rows at most
        ds[r, col] = DROP_STAT
    f[grp].create_dataset("lbt_fix_rows", data=rows)
    f[grp].create_dataset("lbt_fix_pstat_before", data=before)
    f.attrs[ATTR] = NOTE


def revert(f, spec):
    grp, table, col, _ = spec
    ds = f[grp][table]
    for r, p in zip(f[grp]["lbt_fix_rows"][:], f[grp]["lbt_fix_pstat_before"][:]):
        ds[r, col] = p
    del f[grp]["lbt_fix_rows"], f[grp]["lbt_fix_pstat_before"]
    del f.attrs[ATTR]


def pair_files(args):
    out = []
    for a in args:
        if os.path.isdir(a):
            out += sorted(p for p in glob.glob(os.path.join(a, "*.h5"))
                          if not re.search(r"_(particlize|hadrons_\w+)\.h5$", p))
        else:
            out.append(a)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+", help="directories or pair .h5 files")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="patch the files in place")
    mode.add_argument("--revert", action="store_true", help="undo an earlier --apply")
    args = ap.parse_args(argv)

    mode_w = args.apply or args.revert
    tot_rows = tot_ev = tot_hit = 0
    failed = []
    for path in pair_files(args.paths):
        ppath = re.sub(r"\.h5$", "_particlize.h5", path)
        has_part = os.path.exists(ppath)
        name = os.path.basename(path)
        try:
            with h5py.File(path, "r+" if mode_w else "r") as pair, \
                    (h5py.File(ppath, "r+" if mode_w else "r") if has_part
                     else _Null()) as part:
                if "shower" not in pair:
                    print(f"{name}: no shower graph, skipped")
                    continue
                patched = ATTR in pair.attrs
                if args.revert:
                    if not patched:
                        print(f"{name}: not patched, nothing to revert")
                        continue
                    revert(pair, GRAPH)
                    if part is not None and ATTR in part.attrs:
                        revert(part, FINAL)
                    print(f"{name}: reverted")
                    continue
                if patched:
                    print(f"{name}: already patched ({len(pair['shower/lbt_fix_rows'])} "
                          f"partons), max |balance| {check_balance(pair):.1e} GeV")
                    continue
                g_rows, f_rows, per_ev = find_double(pair, part)
                tot_rows += len(g_rows)
                tot_ev += len(per_ev)
                tot_hit += int((per_ev > 0).sum())
                what = (f"{len(g_rows)} partons in {(per_ev > 0).sum()} of {len(per_ev)} "
                        f"events" + ("" if has_part else "; no _particlize.h5"))
                if not args.apply:
                    print(f"{name}: would set {what} to -11")
                    continue
                patch(pair, GRAPH, g_rows)
                if part is not None:
                    patch(part, FINAL, f_rows)
                worst = check_balance(pair)
                if worst > 1e-4:
                    raise RuntimeError(f"balance still off by {worst:.3g} GeV after patching")
                print(f"{name}: set {what} to -11; max |balance| {worst:.1e} GeV")
        except Exception as exc:  # report and go on with the other files
            failed.append(name)
            print(f"{name}: FAILED, {exc}", file=sys.stderr)

    if not args.revert and tot_ev:
        verb = "set" if args.apply else "would set"
        print(f"\n{verb} {tot_rows} partons in {tot_hit} of {tot_ev} events to -11"
              + ("" if args.apply else "  (dry run: add --apply to write)"))
    if failed:
        sys.exit(f"{len(failed)} file(s) failed: {', '.join(failed)}")


class _Null:
    """Stands in for a missing _particlize.h5 in the with-statement."""
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    main()
