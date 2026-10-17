#!/usr/bin/env python3
"""The C++ map steps (example/analysis_root/zjet) against the Python ones, on the same events.

    python check_cxx.py pools <python pool dir> <C++ pool dir>     # mix.py's / zhadron_pools'
    python check_cxx.py jets  <python jets dir> <C++ jets dir>     # zjet_jets_map.py's / zjets'
    python check_cxx.py qa    <python qa dir> <C++ qa dir>         # qa.py map's / zqa's

pools: every array of zjet_mixing's pool (per event S_full, S_one, the one oversample's
tracks, weights, Z, k_one; per background W, W2, SW, T, rho for both contents).  jets: every
(event, sample) entry's jets, embedding pairs and random cones, and every background
sample's jets (bg_jets).  Integers and the events
must be equal; floats agree to float32 / float64 rounding (the C++ and numpy logarithms can
differ in the last bit, which can move a track across a bin edge: such moves are counted).
Exits 1 on a difference beyond that."""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
try:
    import jetscape  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "python"))

BAD = []


def report(ok, what):
    print(f"  {'ok  ' if ok else 'FAIL'} {what}")
    if not ok:
        BAD.append(what)


def close(a, b, what, rtol):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        report(False, f"{what}: shapes {a.shape} / {b.shape}")
        return
    d = np.abs(a - b)
    tol = rtol * np.maximum(np.abs(a), np.abs(b)) + 1e-12
    n = int(np.count_nonzero(d > tol))
    report(n == 0, f"{what}: max |diff| {d.max() if d.size else 0:.2e}"
                   + (f", {n} of {d.size} beyond {rtol:g}" if n else ""))


def pools(py_dir, cxx_dir):
    import zjet_mixing as zm
    for py in sorted(glob.glob(os.path.join(py_dir, "*.npz"))):
        stem = os.path.basename(py)[:-4]
        cx = os.path.join(cxx_dir, stem + "_pool.root")
        print(stem)
        if not os.path.exists(cx):
            report(False, "C++ pool exists")
            continue
        with np.load(py) as f:
            a = {k: f[k] for k in f.files}
        b = zm.read_root_pool(cx)
        report(set(a) == set(b), f"same arrays ({len(a)})"
               + ("" if set(a) == set(b) else f": {sorted(set(a) ^ set(b))}"))
        for k in ("event", "bg", "k_one", "n_samples", "trk_n", "trk_pt", "bg_id"):
            report(np.array_equal(a[k], b[k]), f"{k} equal")
        for k in ("w", "y", "phi", "bg_W", "bg_W2"):
            close(a[k], b[k], k, 1e-12)
        for k in ("S_full", "S_one", "trk_eta", "trk_phi"):
            close(a[k], b[k], k, 2e-7)
        for k in sorted(x for x in a if x.startswith(("bg_SW", "bg_T", "bg_R"))):
            if k in b:
                close(a[k], b[k], k, 1e-12)


def jets(py_dir, cxx_dir):
    import awkward as ak
    from zjet_jets import load_jets
    Ea, Ja, Ba = load_jets(py_dir)
    Eb, Jb, Bb = load_jets(cxx_dir)
    for what, a, b in (("jet_events", Ea, Eb), ("jets", Ja, Jb), ("bg_jets", Ba, Bb)):
        report(set(a) == set(b), f"{what}: the same columns ({len(a)})"
               + ("" if set(a) == set(b) else f": {sorted(set(a) ^ set(b))}"))
        for k in sorted(set(a) & set(b)):
            x, y = a[k], b[k]
            if isinstance(x, ak.Array):
                if not ak.all(ak.num(x) == ak.num(y)):
                    report(False, f"{what}.{k}: entries per row")
                    continue
                x, y = ak.to_numpy(ak.flatten(x)), ak.to_numpy(ak.flatten(y))
            x, y = np.asarray(x), np.asarray(y)
            if len(x) != len(y):
                report(False, f"{what}.{k}: {len(x)} / {len(y)} values")
            elif x.dtype.kind in "iu":
                report(np.array_equal(x, y), f"{what}.{k} equal ({len(x)})")
            else:
                nan = np.isnan(x)
                if not np.array_equal(nan, np.isnan(y)):
                    report(False, f"{what}.{k}: NaN pattern")
                    continue
                close(x[~nan], y[~nan], f"{what}.{k} ({len(x)})", 1e-9)


def qa(py_dir, cxx_dir):
    from qa import load
    Ea, Ha = load(py_dir)
    Eb, Hb = load(cxx_dir)
    for what, a, b in (("qa_events", Ea, Eb), ("qa_hist", Ha, Hb)):
        report(set(a) == set(b), f"{what}: the same columns ({len(a)})"
               + ("" if set(a) == set(b) else f": {sorted(set(a) ^ set(b))}"))
        for k in sorted(set(a) & set(b)):
            x, y = np.asarray(a[k]), np.asarray(b[k])
            if x.shape != y.shape:
                report(False, f"{what}.{k}: shapes {x.shape} / {y.shape}")
            elif x.dtype.kind in "iu" and y.dtype.kind in "iu":
                report(np.array_equal(x, y), f"{what}.{k} equal")
            else:
                close(x, y, f"{what}.{k}", 1e-9)


def main():
    modes = {"pools": pools, "jets": jets, "qa": qa}
    if len(sys.argv) != 4 or sys.argv[1] not in modes:
        print(__doc__)
        sys.exit(2)
    modes[sys.argv[1]](sys.argv[2], sys.argv[3])
    print(f"{len(BAD)} difference(s)")
    sys.exit(1 if BAD else 0)


if __name__ == "__main__":
    main()
