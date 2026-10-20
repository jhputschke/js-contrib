#!/usr/bin/env python3
"""Campaigns mapped apart and combined at load time give what one mapping of all their files
gives (docs/Plans/PLAN_dijet_deta_wake.md A0/A4): the Python jet map step and the pools, run
on every campaign alone and on all of them together, then loaded.

    python combine_check.py --campaign <c00 dir or quoted glob> --campaign <c01 ...> \\
        --out <scratch dir> [--samples 4] [--R 0.3] [--jobs 4]

Checks, bit for bit: zjet_jets.load_jets / load_qa_jets of the separate outputs against
those of the joint one (every column; the ids and weight_mb rebuilt from the file records),
and the joint outputs' stored event, file, bg and weight_mb (the reader's, over all files)
against the rebuilt ones; zjet_mixing.Pool.load of the separate pools against the joint ones
(every array), and the rebuilt weights against the joint build's stored w.  Exits 1 on a
difference.  The joint run reads the files through symbolic links in <out>/all."""
import argparse
import glob
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ZJET = os.path.dirname(HERE)
sys.path.insert(0, ZJET)
sys.path.insert(0, os.path.dirname(ZJET))                        # jetcorr
try:
    import jetscape  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.join(ZJET, "..", "..", "..", "python"))
BAD = []


def report(ok, what):
    print(f"  {'ok  ' if ok else 'FAIL'} {what}")
    if not ok:
        BAD.append(what)


def stems(src):
    from jetscape.hadrons_h5 import HadronFileReader
    return list(HadronFileReader(src).stems)


def same(x, y):
    import awkward as ak
    if isinstance(x, ak.Array) or isinstance(y, ak.Array):
        if len(x) != len(y):
            return False
        if x.ndim > 1 and not np.array_equal(ak.to_numpy(ak.num(x, axis=1)),
                                             ak.to_numpy(ak.num(y, axis=1))):
            return False
        x, y = ak.to_numpy(ak.ravel(x)), ak.to_numpy(ak.ravel(y))
    x, y = np.asarray(x), np.asarray(y)
    if x.shape != y.shape:
        return False
    if x.dtype.kind == "f":
        return bool(np.array_equal(x, y, equal_nan=True))
    return bool(np.array_equal(x, y))


def compare(what, a, b):
    keys = sorted(set(a) | set(b))
    bad = [k for k in keys if k not in a or k not in b or not same(a[k], b[k])]
    report(not bad, f"{what}: {len(keys)} columns identical" + (f" (not: {bad})" if bad else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--campaign", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--R", type=float, default=0.3)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    import mix
    import uproot
    import zjet_jets as zj
    import zjet_mixing as zm

    os.makedirs(a.out, exist_ok=True)
    alld = os.path.join(a.out, "all")
    os.makedirs(alld, exist_ok=True)
    for src in a.campaign:
        for st in stems(src):
            for p in glob.glob(st + "*"):
                q = os.path.join(alld, os.path.basename(p))
                if not os.path.lexists(q):
                    os.symlink(os.path.abspath(p), q)
    runs_ = [(f"c{i}", src) for i, src in enumerate(a.campaign)] + [("all", alld)]
    mapper = os.path.join(ZJET, "zjet_jets_map.py")
    for name, src in runs_:
        out = os.path.join(a.out, f"jets_{name}")
        n = len(stems(src))
        procs = [subprocess.Popen([sys.executable, mapper, src, out, "--samples", str(a.samples),
                                   "--R", str(a.R), "--files", f"{i}:{i + 1}"],
                                  stdout=subprocess.DEVNULL) for i in range(n)]
        if any(p.wait() for p in procs):
            raise SystemExit(f"map step failed on {src}")
        mix.build_pools(src, os.path.join(a.out, f"mix_{name}"), a.jobs, 1, True)
    sep = [os.path.join(a.out, f"jets_c{i}") for i in range(len(a.campaign))]
    joint = os.path.join(a.out, "jets_all")
    print("jets: the campaigns' outputs loaded together against the joint mapping")
    for what, A, B in zip(("jet_events", "jets", "bg_jets"), zj.load_jets(sep),
                          zj.load_jets(joint)):
        compare(what, A, B)
    for what, A, B in zip(("qa_jets", "qa_bg_jets"), zj.load_qa_jets(sep),
                          zj.load_qa_jets(joint)):
        compare(what, A, B)
    E, _, _ = zj.load_jets(joint)
    stored = {k: [] for k in ("event", "file", "bg", "weight_mb")}
    for p in zj.map_files(joint):
        with uproot.open(p) as f:
            t = f["jet_events"].arrays(library="np")
        for k in stored:
            stored[k].append(t[k])
    for k in stored:
        report(same(np.concatenate(stored[k]), E[k]),
               f"joint mapping: stored {k} = rebuilt (the reader's over all files)")
    print("pools: the campaigns' pools loaded together against the joint build")
    sp = sorted(sum((glob.glob(os.path.join(a.out, f"mix_c{i}", "*.npz"))
                     for i in range(len(a.campaign))), []))
    jp = sorted(glob.glob(os.path.join(a.out, "mix_all", "*.npz")))
    A, B = zm.Pool.load(sp), zm.Pool.load(jp)
    keys = sorted(k for k in vars(B) if isinstance(getattr(B, k), np.ndarray))
    bad = [k for k in keys if not same(getattr(A, k), getattr(B, k))]
    report(not bad, f"Pool: {len(keys)} arrays identical" + (f" (not: {bad})" if bad else ""))
    for c in zm.CONTENTS:
        for q in ("bg_SW", "bg_T", "bg_R"):
            report(same(getattr(A, q)[c], getattr(B, q)[c]), f"Pool.{q}[{c}] identical")
    w_build = np.concatenate([np.load(p)["w"] for p in jp])
    report(same(w_build, B.w), "joint build: stored w = rebuilt w")
    print(f"{len(BAD)} difference(s)")
    sys.exit(1 if BAD else 0)


if __name__ == "__main__":
    main()
