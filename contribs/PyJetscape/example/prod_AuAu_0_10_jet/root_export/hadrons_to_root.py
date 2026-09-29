#!/usr/bin/env python3
"""
example/prod_AuAu_0_10_jet/root_export/hadrons_to_root.py

One hadron file of hadronize.py (``<stem>_hadrons_<tag>.h5``) -> a ROOT file, for analyses in
ROOT.  Prototype of the converter compared in bench_formats.py (see README.md here).

    python hadrons_to_root.py RUN_hadrons_bulk_jet.h5 -o RUN_bulk_jet.root           # RNTuple
    python hadrons_to_root.py RUN_hadrons_bulk_jet.h5 -o RUN_bulk_jet.root --format ttree
                                                        # for ROOT < 6.34
    python hadrons_to_root.py RUN_hadrons_bulk_jet.h5 -o out.root --no-x \\
           --eta-max 1 --charged                        # a small analysis export
    python hadrons_to_root.py RUN_hadrons_bulk_jet.h5 -o out.root --writer uproot  # no ROOT

One entry per sample (oversample), named after the tag (``bulk_jet``, ``bulk_bg``,
``jet_frag``):

    event, unit, sample, bg_unit     int       event (first event using the unit), unit,
                                               sample k within the unit, the event's
                                               background unit (bulk_bg: the unit itself)
    n                                int       hadrons in the sample (TTree counter)
    pid, pstat                       int[n]
    E, px, py, pz                    float[n]  GeV
    t, x, y, z                       float[n]  fm, unless --no-x

Oversamples of one event share one fluid, and the events of a background share it: they
are not independent events.  Errors have to be taken per event (``event``) or, for
--correlated hadron files, per sample pair (``bg_unit``, ``sample``).

Writers (--writer): ``root`` through PyROOT (root_writers.h; TTree or RNTuple, optionally
with truncated floats: Float16_t / Real32Trunc), or ``uproot`` (TTree or RNTuple, float32;
no ROOT needed).  ``auto`` (default) takes ROOT when PyROOT imports and uproot otherwise:
the files are the same for every reader, but uproot's RNTuple is 16-24% larger (its floats
are not byte-split) and ~1.2x slower to read in ROOT than ROOT's (see README.md).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "python"))

#: TTree / RNTuple fields per hadron, in root_writers.h's order
P_FIELDS = ("E", "px", "py", "pz")
X_FIELDS = ("t", "x", "y", "z")


def load_columns(path, *, with_x=True, eta_max=None, charged=False, particlize=None):
    """A hadron file as flat columns: per sample (``soff``, ``event``, ``unit``, ``sample``,
    ``bg_unit``) and per hadron (``pid``, ``pstat``, ``E`` ... ``pz``, ``t`` ... ``z``).
    ``eta_max`` / ``charged`` drop hadrons; every sample keeps its entry.  ``bg_unit`` comes
    from ``particlize`` (default: the file the hadron file names as its ``source``, next to
    it)."""
    import h5py

    from jetscape import h5_compression  # noqa: F401  (Blosc filter)
    from jetscape.hadrons_h5 import CHARGED, pseudorapidity

    with h5py.File(path, "r") as f:
        tag = str(f.attrs["tag"])
        g = f["hadrons"]
        so = g["sample_offsets"][:].astype(np.int64)
        uo = g["unit_offsets"][:].astype(np.int64)
        pid, pstat, p = g["pid"][:], g["pstat"][:], g["p"][:]
        x = g["x"][:] if with_x else None
        units = {k: f["units"][k][:] for k in f["units"]}
        source = str(f.attrs.get("source", ""))
    n_units = len(uo) - 1
    sample_unit = np.repeat(np.arange(n_units), np.diff(uo))
    cols = {"tag": tag,
            "unit": units["unit"].astype(np.int32)[sample_unit],
            "event": units["event"].astype(np.int32)[sample_unit],
            "sample": (np.arange(len(so) - 1) - uo[sample_unit]).astype(np.int32)}
    if tag == "bulk_bg":
        cols["bg_unit"] = cols["unit"].copy()
    else:
        cols["bg_unit"] = np.full(len(so) - 1, -1, np.int32)
        part = particlize or os.path.join(os.path.dirname(os.path.abspath(path)), source)
        if (particlize or source) and os.path.exists(part):
            with h5py.File(part, "r") as pf:
                if "events/bg_unit" in pf:
                    cols["bg_unit"] = pf["events/bg_unit"][:].astype(np.int32)[cols["event"]]
    keep = None
    if eta_max is not None:
        keep = np.abs(pseudorapidity(p)) < eta_max
    if charged:
        ch = np.isin(np.abs(pid), CHARGED)
        keep = ch if keep is None else keep & ch
    if keep is not None:
        sample = np.repeat(np.arange(len(so) - 1), np.diff(so))
        counts = np.bincount(sample[keep], minlength=len(so) - 1)
        so = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        pid, pstat, p = pid[keep], pstat[keep], p[keep]
        x = x[keep] if x is not None else None
    cols["soff"] = so
    cols["pid"] = np.ascontiguousarray(pid, dtype=np.int32)
    cols["pstat"] = np.ascontiguousarray(pstat, dtype=np.int32)
    for k, name in enumerate(P_FIELDS):
        cols[name] = np.ascontiguousarray(p[:, k], dtype=np.float32)
    cols["with_x"] = x is not None
    for k, name in enumerate(X_FIELDS):
        cols[name] = (np.ascontiguousarray(x[:, k], dtype=np.float32) if x is not None
                      else np.zeros(1, np.float32))
    return cols


_DECLARED = False


def _root():
    global _DECLARED
    import ROOT

    if not _DECLARED:
        if not ROOT.gInterpreter.Declare(f'#include "{os.path.join(HERE, "root_writers.h")}"'):
            raise RuntimeError("root_writers.h did not compile")
        _DECLARED = True
    return ROOT


def write_root(out, cols, *, fmt="ttree", bits_p=0, bits_x=0, compression=505, update=False):
    """Write with ROOT (root_writers.h); ``update``: add to the existing file ``out``.
    -> seconds spent in ROOT."""
    ROOT = _root()
    c = ROOT.hadrons_root.make_columns(
        len(cols["soff"]) - 1, cols["soff"], cols["event"], cols["unit"], cols["sample"],
        cols["bg_unit"], cols["pid"], cols["pstat"], *(cols[k] for k in P_FIELDS),
        *(cols[k] for k in X_FIELDS), int(cols["with_x"]))
    fn = {"ttree": ROOT.hadrons_root.write_ttree,
          "rntuple": ROOT.hadrons_root.write_rntuple}[fmt]
    return float(fn(str(out), cols["tag"], int(compression), int(bits_p), int(bits_x), c,
                    int(bool(update))))


#: uproot writes one TTree basket / one RNTuple cluster per extend: samples per extend
UPROOT_CHUNK = {"ttree": 20, "rntuple": 500}


def write_uproot(out, cols, *, fmt="ttree", level=5, chunk_samples=None):
    """Write a TTree or an RNTuple with uproot (ZSTD ``level``; one TTree basket or RNTuple
    cluster per ``chunk_samples`` samples, default UPROOT_CHUNK). -> seconds."""
    import uproot

    t0 = time.time()
    with uproot.recreate(str(out), compression=uproot.ZSTD(level)) as fo:
        write_uproot_into(fo, cols, fmt=fmt, chunk_samples=chunk_samples)
    return time.time() - t0


def write_uproot_into(fo, cols, *, fmt="ttree", chunk_samples=None):
    """Add the tag's tree / ntuple to an open uproot file ``fo``."""
    import awkward as ak

    so = cols["soff"]
    chunk = chunk_samples or UPROOT_CHUNK[fmt]
    fields = ["pid", "pstat", *P_FIELDS] + (list(X_FIELDS) if cols["with_x"] else [])
    scalars = ("event", "unit", "sample", "bg_unit")
    obj = None
    for s0 in range(0, len(so) - 1, chunk):
        s1 = min(s0 + chunk, len(so) - 1)
        a, b = int(so[s0]), int(so[s1])
        counts = np.diff(so[s0:s1 + 1])
        jagged = {k: ak.unflatten(cols[k][a:b], counts) for k in fields}
        if fmt == "rntuple":
            data = ak.zip({**{k: cols[k][s0:s1] for k in scalars}, **jagged}, depth_limit=1)
            if obj is None:
                obj = fo.mkrntuple(cols["tag"], data)
            else:
                obj.extend(data)
            continue
        data = {k: cols[k][s0:s1] for k in scalars}
        data["hadron"] = ak.zip(jagged)
        if obj is None:
            obj = fo.mktree(cols["tag"], {k: v.dtype for k, v in data.items()
                                          if k != "hadron"} | {"hadron": data["hadron"].type},
                            counter_name=lambda counted: "n",
                            field_name=lambda outer, inner: inner)
        obj.extend(data)


def have_pyroot():
    """True if PyROOT imports and can compile the writers."""
    try:
        _root()
    except Exception:                                  # noqa: BLE001 - ImportError and more
        return False
    return True


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("hadrons", help="a <stem>_hadrons_<tag>.h5 from hadronize.py")
    p.add_argument("-o", "--out", required=True, help="output .root file")
    p.add_argument("--format", choices=("rntuple", "ttree"), default="rntuple",
                   help="rntuple (default: smallest, fastest; ROOT >= 6.34 to read) or ttree "
                        "(any ROOT)")
    p.add_argument("--writer", choices=("auto", "root", "uproot"), default="auto",
                   help="auto (default): root if PyROOT imports, else uproot; root (PyROOT: "
                        "the smallest files, truncated floats); uproot (no ROOT needed, "
                        "float32 only)")
    p.add_argument("--bits-p", type=int, default=0, dest="bits_p",
                   help="mantissa bits for E, px, py, pz (Float16_t / Real32Trunc; 0: float)")
    p.add_argument("--bits-x", type=int, default=0, dest="bits_x",
                   help="mantissa bits for t, x, y, z (0: float)")
    p.add_argument("--no-x", action="store_true", dest="no_x", help="leave out t, x, y, z")
    p.add_argument("--eta-max", type=float, default=None, dest="eta_max",
                   help="only hadrons with |eta| < ETA_MAX")
    p.add_argument("--charged", action="store_true", help="only charged hadrons")
    p.add_argument("--compression", type=int, default=505,
                   help="ROOT compression setting, algorithm*100 + level (default 505: ZSTD 5)")
    a = p.parse_args(argv)
    writer = a.writer
    if writer == "auto":
        writer = "root" if have_pyroot() else "uproot"
        if writer == "uproot":
            print("hadrons_to_root.py: note -- PyROOT not available, writing with uproot: "
                  "readable by the same tools, but "
                  + ("16-24% larger and ~1.2x slower to read in ROOT than ROOT's RNTuple"
                     if a.format == "rntuple" else "~5% larger than ROOT's TTree"),
                  file=sys.stderr)
    if writer == "uproot" and (a.bits_p or a.bits_x):
        p.error("truncated floats (--bits-p/--bits-x) need the ROOT writer (PyROOT)")
    t0 = time.time()
    cols = load_columns(a.hadrons, with_x=not a.no_x, eta_max=a.eta_max, charged=a.charged)
    t_load = time.time() - t0
    if writer == "uproot":
        t_write = write_uproot(a.out, cols, fmt=a.format, level=a.compression % 100)
    else:
        t_write = write_root(a.out, cols, fmt=a.format, bits_p=a.bits_p, bits_x=a.bits_x,
                             compression=a.compression)
    print(f"hadrons_to_root.py: {a.hadrons} -> {a.out} ({cols['tag']}, {a.format} by {writer}, "
          f"{len(cols['soff']) - 1} samples, {len(cols['pid'])} hadrons, "
          f"{os.path.getsize(a.out) / 1e6:.1f} MB); read {t_load:.1f} s, write {t_write:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
