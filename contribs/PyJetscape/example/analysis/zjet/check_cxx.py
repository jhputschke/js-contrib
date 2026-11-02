#!/usr/bin/env python3
"""The C++ map steps (example/analysis_root/zjet) against the Python ones, on the same events.

    python check_cxx.py pools <python pool dir> <C++ pool dir>     # mix.py's / zhadron_pools'
    python check_cxx.py jets  <python jets dir> <C++ jets dir>     # zjet_jets_map.py's / zjets'
    python check_cxx.py qa    <python qa dir> <C++ qa dir>         # qa.py map's / zqa's

pools: every array of zjet_mixing's pool (per event S_full, S_one, the one oversample's
tracks, weights, Z, k_one, the keys and the dijet proxy's Y_*, L_*; per background and window
the unweighted sums, per background bulk_bg's density, the file record, the grid's edges;
version 3: per background W, W2, SW, T, rho for both contents).  jets: every
(event, sample) entry's jets, embedding pairs, random cones and Z cone, every background
sample's jets (bg_jets), the jet QA trees (qa_jets, qa_bg_jets), and the outputs' records
(file_info, run_info).  With extra hard scatterings also the pools' bgw_R_extra, the jets'
bgx_* (B_own: bulk_bg + extra_frag per sample), run_info's extras and the QA's extra_frag
histograms and n_extra, nch*_extra (every array and column is compared, and both sides must
have the same ones: mapped with and without extras do not pass).  Integers and the events
must be equal; floats agree to float32 / float64 rounding (the C++ and numpy logarithms can
differ in the last bit, which can move a track across a bin edge: such moves are counted).
Exits 1 on a difference beyond that."""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                         # zjet_mixing
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
        ints = ("event", "bg", "k_one", "n_samples", "trk_n", "trk_pt", "bg_id", "local_event",
                "pthat_bin", "bg_unit", "bgw_unit", "bgw_k", "bgw_n", "bgr_unit", "fi_n_events",
                "fi_vacuum", "fi_units", "fi_n_stored", "pool_version")
        for k in ints:
            if k in a:
                report(np.array_equal(a[k], b[k]), f"{k} equal")
        for k in ("fi_name", "fi_pthat_lo", "fi_pthat_hi", "fi_sigma_gen", "fi_sigma_err",
                  "fi_n_accepted"):
            if k in a:
                report(np.array_equal(a[k], b[k]), f"{k} equal")
        for k in sorted(x for x in a if x.startswith("grid_")):     # the edges, to 10^-12
            close(a[k], b[k], k, 0.0)
        for k in ("n_coll", "bgw_ncoll"):
            if k in a:
                report(np.array_equal(a[k], b[k], equal_nan=True), f"{k} equal")
        for k in ("w", "y", "phi", "bg_W", "bg_W2"):
            if k in a:
                close(a[k], b[k], k, 1e-12)
        for k in sorted(x for x in a if x.startswith(("S_", "Y_", "L_")) or x in ("trk_eta",
                                                                                  "trk_phi")):
            close(a[k], b[k], k, 2e-7)
        for k in sorted(x for x in a if x.startswith(("bg_SW", "bg_T", "bg_R", "bgw_S",
                                                      "bgw_R", "bgr_RB"))):
            if k in b:
                close(a[k], b[k], k, 1e-12)


def records(py_dir, cxx_dir, suffix):
    """The file_info and run_info records of every output pair (jetcorr.mapio)."""
    import uproot
    sys.path.insert(0, os.path.dirname(HERE))
    from jetcorr import mapio
    for pa in sorted(glob.glob(os.path.join(py_dir, f"*{suffix}"))):
        pb = os.path.join(cxx_dir, os.path.basename(pa))
        if not os.path.exists(pb):
            report(False, f"{os.path.basename(pa)}: no C++ output")
            continue
        with uproot.open(pa) as fa, uproot.open(pb) as fb:
            ra, rb = mapio.read_file_info(fa), mapio.read_file_info(fb)
            ia, ib = mapio.read_run_info(fa), mapio.read_run_info(fb)
        if ra is None or rb is None:
            report(ra is None and rb is None, f"{os.path.basename(pa)}: file_info in both")
            continue
        same = all(np.array_equal(np.asarray(ra[k]), np.asarray(rb[k])) for k in ra)
        report(same, f"{os.path.basename(pa)}: file_info equal ({ra['name']}, "
                     f"{ra['n_events']} events, {len(ra['units'])} units)")
        report(ia == ib, f"{os.path.basename(pa)}: run_info equal {ia}"
               + ("" if ia == ib else f" / {ib}"))


def jets(py_dir, cxx_dir):
    import awkward as ak
    from zjet_jets import load_jets, load_qa_jets
    records(py_dir, cxx_dir, "_jets.root")
    Ea, Ja, Ba = load_jets(py_dir)
    Eb, Jb, Bb = load_jets(cxx_dir)
    Qa, QBa = load_qa_jets(py_dir)
    Qb, QBb = load_qa_jets(cxx_dir)
    for what, a, b in (("jet_events", Ea, Eb), ("jets", Ja, Jb), ("bg_jets", Ba, Bb),
                       ("qa_jets", Qa, Qb), ("qa_bg_jets", QBa, QBb)):
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
