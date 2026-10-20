#!/usr/bin/env python3
"""Two directories of map outputs or pools (*.root) compared tree by tree and column by column,
for the files in both (B may hold a few remapped files of A): every column of A must be in B
and equal bit for bit (NaN = NaN); B may have more columns and trees (a newer map step's).
For regressions of the map steps against older outputs (docs/Plans/PLAN_dijet_deta_wake.md
A4: the map step at R = 0.3 against runs/zjet_c00/data/jets/).

    python cmp_outputs.py <old dir> <new dir> [--files '*.root']

Exits 1 on a difference, or if no file is in both."""
import argparse
import glob
import os
import sys

import numpy as np

#: uproot's counters of jagged records (not data)
COUNTERS = ("njet", "nnw", "nemb", "nrc")


def flat(arr):
    out = {}
    for k in arr.fields:
        x = arr[k]
        if x.fields:
            for s in x.fields:
                out[f"{k}_{s}"] = x[s]
        else:
            out[k] = x
    return out


def same(x, y):
    import awkward as ak
    if len(x) != len(y):
        return False
    if x.ndim > 1 and not np.array_equal(ak.to_numpy(ak.num(x, axis=1)),
                                         ak.to_numpy(ak.num(y, axis=1))):
        return False
    a, b = ak.to_numpy(ak.ravel(x)), ak.to_numpy(ak.ravel(y))
    if a.dtype.kind in "USO" or b.dtype.kind in "USO":
        return bool(np.array_equal(a, b))
    return bool(np.all((a == b) | (np.isnan(a.astype(float)) & np.isnan(b.astype(float)))))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--files", default="*.root", help="a glob of file names in <old>")
    a = ap.parse_args()
    import uproot
    bad = 0
    paths = [p for p in sorted(glob.glob(os.path.join(a.old, a.files)))
             if os.path.exists(os.path.join(a.new, os.path.basename(p)))]
    if not paths:
        sys.exit(f"no {a.files} in both {a.old} and {a.new}")
    print(f"{len(paths)} file(s) in both")
    for pa in paths:
        pb = os.path.join(a.new, os.path.basename(pa))
        with uproot.open(pa) as fa, uproot.open(pb) as fb:
            ta = sorted({k.split(";")[0] for k in fa.keys()})
            tb = {k.split(";")[0] for k in fb.keys()}
            for t in ta:
                if t not in tb:
                    print(f"  MISSING tree {t}")
                    bad += 1
                    continue
                A, B = flat(fa[t].arrays()), flat(fb[t].arrays())
                diff = [k for k in A if k in B and not same(A[k], B[k])]
                miss = [k for k in A if k not in B and k not in COUNTERS]
                bad += len(diff) + len(miss)
                extra = sorted(set(B) - set(A))
                print(f"{os.path.basename(pa)} {t}: {len(A)} columns"
                      + (f"; DIFFERENT: {diff}" if diff else "")
                      + (f"; MISSING: {miss}" if miss else "")
                      + (f"; new: {extra}" if extra else ""))
            new_trees = sorted(tb - set(ta))
            if new_trees:
                print(f"  new trees: {new_trees}")
    print("IDENTICAL" if not bad else f"{bad} difference(s)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
