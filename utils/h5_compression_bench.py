#!/usr/bin/env python3
"""Benchmark HDF5 filters on a real evolution dataset: ratio, write and read speed.

Reads the first N events of one dataset (``arr`` by default) into memory, rewrites them
with each filter using the file's own chunk shape and the writers' access pattern (one
tau frame per write for tau-sliced chunks, one event otherwise), reads them back, and
checks the round trip.  The numbers in contribs/PyJetscape/README_h5_optim.md come from
this script.

    python utils/h5_compression_bench.py FILE.h5 [--events 2] [--key arr_bg]
    BLOSC_NTHREADS=8 python utils/h5_compression_bench.py FILE.h5

Reads come from the page cache (the test file was just written), so they measure
decompression plus HDF5's copy into the output array, not the disk.  Needs h5py,
hdf5plugin and numpy; nothing from this repository.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time

import numpy as np

try:
    import h5py
    import hdf5plugin
except ImportError as exc:  # pragma: no cover
    sys.exit(f"h5_compression_bench.py: needs h5py and hdf5plugin ({exc})")

B = hdf5plugin.Blosc

#: label -> h5py keyword arguments
FILTERS = {
    "none": {},
    "lzf": dict(compression="lzf"),
    "gzip4+shuffle": dict(compression="gzip", compression_opts=4, shuffle=True),
    "blosc-lz4+shuffle": dict(B(cname="lz4", clevel=5, shuffle=B.SHUFFLE)),
    "blosc-lz4+bitshuffle": dict(B(cname="lz4", clevel=5, shuffle=B.BITSHUFFLE)),
    "blosc-lz4:9+shuffle": dict(B(cname="lz4", clevel=9, shuffle=B.SHUFFLE)),
    "blosc-zstd:3+shuffle": dict(B(cname="zstd", clevel=3, shuffle=B.SHUFFLE)),
    "blosc-zstd:3+bitshuffle": dict(B(cname="zstd", clevel=3, shuffle=B.BITSHUFFLE)),
}
#: the ones worth repeating on mantissa-rounded data
ROUNDED = ("blosc-lz4+shuffle", "blosc-lz4+bitshuffle", "blosc-zstd:3+shuffle",
           "blosc-zstd:3+bitshuffle")


def round_mantissa(a, keep_bits):
    """float32 -> keep_bits mantissa bits, round-half-to-even (as h5_compression.py)."""
    drop = 23 - keep_bits
    bits = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    lsb = (bits >> np.uint32(drop)) & np.uint32(1)
    r = (bits + (np.uint32((1 << (drop - 1)) - 1) + lsb)) & np.uint32(
        (0xFFFFFFFF << drop) & 0xFFFFFFFF)
    return np.where(np.isfinite(a), r.view(np.float32), a).astype(np.float32)


def run(label, kw, data, chunks, tmpdir):
    fn = os.path.join(tmpdir, "bench.h5")
    t0 = time.perf_counter()
    with h5py.File(fn, "w") as f:
        d = f.create_dataset("arr", data.shape, np.float32, chunks=chunks, **kw)
        for i in range(data.shape[0]):
            if chunks[-1] > 1:                 # whole-event chunks (fast_data, FastHydro)
                d[i] = data[i]
            else:                              # tau-sliced chunks (PyJetscape writers)
                for t in range(data.shape[-1]):
                    d[i, ..., t] = data[i, ..., t]
    t_write = time.perf_counter() - t0
    size = os.path.getsize(fn)
    t0 = time.perf_counter()
    with h5py.File(fn) as f:
        back = np.empty_like(data)
        f["arr"].read_direct(back)
    t_read = time.perf_counter() - t0
    exact = np.array_equal(back.view(np.uint32), data.view(np.uint32))
    os.remove(fn)
    mb = data.nbytes / 1e6
    print(f"  {label:32s} ratio {data.nbytes / size:5.2f}  {size / 1e6:8.1f} MB  "
          f"write {mb / t_write:6.0f} MB/s  read {mb / t_read:6.0f} MB/s"
          f"{'' if exact else '  MISMATCH'}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("file")
    ap.add_argument("--key", default="arr", help="dataset to benchmark (default: arr)")
    ap.add_argument("--events", type=int, default=2, help="events to load (default: 2)")
    ap.add_argument("--keep-bits", type=int, default=12, dest="keep_bits",
                    help="mantissa bits for the rounded pass; 0 skips it (default: 12)")
    ap.add_argument("--only", nargs="*", default=None, metavar="LABEL",
                    help=f"subset of filters (default: all of {', '.join(FILTERS)})")
    ap.add_argument("--tmpdir", default=None,
                    help="where to write the test file (default: next to FILE)")
    a = ap.parse_args(argv)

    with h5py.File(a.file) as f:
        ds = f[a.key]
        data = ds[:a.events]
        chunks = (1,) + tuple(ds.chunks[1:]) if ds.chunks else (1,) + data.shape[1:]
    nz = [round(float((data[:, c] == 0).mean()), 3) for c in range(data.shape[1])]
    print(f"{os.path.basename(a.file)}[{a.key}]  {data.shape[0]} events  "
          f"{data.nbytes / 1e6:.0f} MB raw  chunks {chunks}  "
          f"BLOSC_NTHREADS={os.environ.get('BLOSC_NTHREADS', 'unset')}  "
          f"exact-zero fraction per channel {nz}")

    tmpdir = a.tmpdir or os.path.dirname(os.path.abspath(a.file))
    with tempfile.TemporaryDirectory(dir=tmpdir) as td:
        for label in (a.only or FILTERS):
            run(label, FILTERS[label], data, chunks, td)
        if a.keep_bits:
            r = round_mantissa(data, a.keep_bits)
            live = data != 0
            err = np.max(np.abs(r[live] - data[live]) / np.abs(data[live])) if live.any() else 0
            print(f"-- rounded to {a.keep_bits} mantissa bits (max relative error {err:.2e})")
            for label in ROUNDED:
                if a.only is None or label in a.only:
                    run(f"keep_bits={a.keep_bits} {label}", FILTERS[label], r, chunks, td)
    return 0


if __name__ == "__main__":
    sys.exit(main())
