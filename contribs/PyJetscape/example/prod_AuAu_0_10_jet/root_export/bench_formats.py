#!/usr/bin/env python3
"""
example/prod_AuAu_0_10_jet/root_export/bench_formats.py

Which ROOT format for the hadrons?  Converts one hadron file with every writer of
hadrons_to_root.py and measures size, write time and read time, against the HDF5 file:

    python bench_formats.py RUN_hadrons_bulk_jet.h5 --out-dir /scratch/root_bench

Variants (one entry per sample in all of them, ZSTD level 5 unless --compression):

    uproot-ttree     TTree written by uproot, float32
    uproot-rntuple   RNTuple written by uproot, float32 (no ROOT needed to write)
    ttree            TTree written by ROOT, float32
    ttree-f16        TTree written by ROOT, Float16_t: p with --bits-p, x with --bits-x
    rntuple          RNTuple, float32
    rntuple-trunc    RNTuple, Real32Trunc: p with --bits-p, x with --bits-x

each with and without the positions (t, x, y, z).  Reading computes the same observable in
every case (the number and the summed pT of charged hadrons at |eta| < 1, from pid, px, py,
pz only): in C++ (read_bench.C, compiled; TTree loop, RNTupleReader views, RDataFrame), with
uproot + awkward, and from the HDF5 file with h5py + numpy.  All must agree.  The files are
read right after they are written, i.e. from the page cache: the times are CPU (decompression
and deserialization), not disk.

Writes <out-dir>/bench_formats.json and prints a markdown table.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hadrons_to_root as h2r  # noqa: E402

VARIANTS = {
    "uproot-ttree": dict(writer="uproot", fmt="ttree", trunc=False),
    "uproot-rntuple": dict(writer="uproot", fmt="rntuple", trunc=False),
    "ttree": dict(writer="root", fmt="ttree", trunc=False),
    "ttree-f16": dict(writer="root", fmt="ttree", trunc=True),
    "rntuple": dict(writer="root", fmt="rntuple", trunc=False),
    "rntuple-trunc": dict(writer="root", fmt="rntuple", trunc=True),
}
CHARGED = (211, 321, 2212, 3222, 3112, 3312, 3334, 11, 13)


def observable(pid, px, py, pz):
    """(n, sum pT) of charged hadrons at |eta| < 1, as read_bench.C computes it."""
    px, py, pz = (np.asarray(v, dtype=np.float64) for v in (px, py, pz))
    pt = np.hypot(px, py)
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = np.arcsinh(pz / pt)
    m = np.isin(np.abs(pid), CHARGED) & (np.abs(eta) < 1.0)
    return int(m.sum()), float(pt[m].sum())


def read_h5(path):
    import h5py

    from jetscape import h5_compression  # noqa: F401

    t0 = time.time()
    with h5py.File(path, "r") as f:
        pid, p = f["hadrons/pid"][:], f["hadrons/p"][:]
    t_io = time.time() - t0
    n, s = observable(pid, p[:, 1], p[:, 2], p[:, 3])
    return n, s, time.time() - t0, t_io


def read_uproot(path, name):
    import awkward as ak
    import uproot

    t0 = time.time()
    with uproot.open(path) as f:
        arr = f[name].arrays(["pid", "px", "py", "pz"])
    t_io = time.time() - t0
    n, s = observable(*(ak.to_numpy(ak.flatten(arr[k])) for k in ("pid", "px", "py", "pz")))
    return n, s, time.time() - t0, t_io


def read_cpp(path, name, mode, build_dir):
    macro = os.path.join(HERE, "read_bench.C")
    cmd = ["root", "-l", "-b", "-q", "-e", f'gSystem->SetBuildDir("{build_dir}", true);',
           f'{macro}+("{path}", "{name}", "{mode}")']
    r = subprocess.run(cmd, capture_output=True, text=True)
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
    if r.returncode != 0 or not line:
        raise RuntimeError(f"read_bench.C {mode} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    _, _, entries, n, s, sec = line[-1].split()
    return int(n), float(s), float(sec), int(entries)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("hadrons", help="a <stem>_hadrons_<tag>.h5")
    p.add_argument("--out-dir", required=True, dest="out_dir")
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--bits-p", type=int, default=12, dest="bits_p")
    p.add_argument("--bits-x", type=int, default=8, dest="bits_x")
    p.add_argument("--compression", type=int, default=505)
    p.add_argument("--only-with-x", action="store_true", dest="only_with_x")
    a = p.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    build_dir = os.path.join(a.out_dir, "aclic")
    os.makedirs(build_dir, exist_ok=True)

    h2r._root()                                 # compile root_writers.h outside the timing
    t0 = time.time()
    cols = h2r.load_columns(a.hadrons, with_x=True)
    t_load = time.time() - t0
    tag = cols["tag"]
    n_had, n_samp = len(cols["pid"]), len(cols["soff"]) - 1
    print(f"{a.hadrons}: {tag}, {n_samp} samples, {n_had} hadrons, "
          f"{os.path.getsize(a.hadrons) / 1e6:.1f} MB; loaded in {t_load:.1f} s", flush=True)
    rows = []
    n_ref, s_ref, t_h5, t_h5_io = read_h5(a.hadrons)
    rows.append(dict(variant="hdf5 (input)", with_x=True, size=os.path.getsize(a.hadrons),
                     reads={"h5py+numpy": t_h5}, read_io={"h5py+numpy": t_h5_io}))
    print(f"  h5py read: {t_h5:.1f} s (I/O {t_h5_io:.1f} s): n = {n_ref}, sum pT = {s_ref:.6e}",
          flush=True)

    ok = True
    for with_x in ((True,) if a.only_with_x else (True, False)):
        c = dict(cols, with_x=with_x)
        for name in [v.strip() for v in a.variants.split(",") if v.strip()]:
            v = VARIANTS[name]
            out = os.path.join(a.out_dir, f"{tag}_{name}{'' if with_x else '_nox'}.root")
            t1 = time.time()
            if v["writer"] == "uproot":
                h2r.write_uproot(out, c, fmt=v["fmt"], level=a.compression % 100)
            else:
                h2r.write_root(out, c, fmt=v["fmt"], bits_p=a.bits_p if v["trunc"] else 0,
                               bits_x=a.bits_x if v["trunc"] else 0,
                               compression=a.compression)
            t_write = time.time() - t1
            row = dict(variant=name, with_x=with_x, size=os.path.getsize(out), write=t_write,
                       reads={}, read_io={})
            modes = ("ttree", "rdf") if v["fmt"] == "ttree" else ("rntuple", "rdf")
            for mode in modes:
                n, s, sec, entries = read_cpp(out, tag, mode, build_dir)
                row["reads"][f"C++ {mode}"] = sec
                good = n == n_ref and abs(s - s_ref) <= 1e-10 * abs(s_ref) and entries == n_samp
                ok &= good
                row.setdefault("agree", True)
                row["agree"] &= good
            try:
                n, s, sec, t_io = read_uproot(out, tag)
                row["reads"]["uproot+awkward"] = sec
                row["read_io"]["uproot+awkward"] = t_io
                good = n == n_ref and abs(s - s_ref) <= 1e-10 * abs(s_ref)
                ok &= good
                row["agree"] &= good
            except Exception as err:                   # noqa: BLE001 - e.g. RNTuple support
                row["reads"]["uproot+awkward"] = f"failed: {type(err).__name__}: {err}"[:200]
            rows.append(row)
            print(f"  {name:14s} x={'yes' if with_x else 'no '} {row['size'] / 1e6:8.1f} MB, "
                  f"write {t_write:6.1f} s, reads {row['reads']}, agree {row['agree']}",
                  flush=True)

    ref = rows[0]["size"]
    print("\n| variant | x | size [MB] | vs HDF5 | bytes/hadron | write [s] | read [s] | agree |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        reads = ", ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}"
                          for k, v in r["reads"].items())
        print(f"| {r['variant']} | {'yes' if r['with_x'] else 'no'} | {r['size'] / 1e6:.1f} | "
              f"{r['size'] / ref:.2f} | {r['size'] / n_had:.1f} | "
              f"{r.get('write', float('nan')):.1f} | {reads} | {r.get('agree', '')} |")
    with open(os.path.join(a.out_dir, "bench_formats.json"), "w") as fh:
        json.dump(dict(input=a.hadrons, tag=tag, n_hadrons=n_had, n_samples=n_samp,
                       observable=dict(n=n_ref, sum_pt=s_ref), bits_p=a.bits_p,
                       bits_x=a.bits_x, compression=a.compression, rows=rows), fh, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
