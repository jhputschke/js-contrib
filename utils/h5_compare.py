#!/usr/bin/env python3
"""Compare two HDF5 files (or two directories of them): which datasets and attributes differ.

The check after a change that should not alter the physics output: run the same job (same
seed) before and after, and compare the files.  By default the comparison is **bit-exact**:
a dataset is the same only if shape, dtype and every byte agree.  For each dataset that
differs it prints how many values differ and by how much; for each attribute, both values.

Usage::

    python h5_compare.py A.h5 B.h5                       # datasets and attributes that differ
    python h5_compare.py dirA dirB                       # every *.h5 of the same name in both
    python h5_compare.py A.h5 dirB                       # A.h5 against dirB/A.h5
    python h5_compare.py A.h5 B.h5 --ignore-stamps       # skip wall times, UUIDs, host, dates
    python h5_compare.py A.h5 B.h5 --ignore 'diag/*' --ignore '@prod_user_xml'
    python h5_compare.py A.h5 B.h5 --no-attrs            # datasets only
    python h5_compare.py A.h5 B.h5 --rtol 1e-6           # floats may differ by 1e-6 (relative)
    python h5_compare.py A.h5 B.h5 -q && echo same       # exit code only

Names: a dataset is its path (``diag/wall_s``), an attribute is ``PATH@NAME`` (``@nevents``
for a file attribute, ``shower@units`` for one of an object).  ``--ignore`` takes shell
patterns on these names (``*`` also matches ``/``) and can be given several times.

``--ignore-stamps`` skips what differs between any two runs of the same job: the wall times
(``*wall_s*``), the UUIDs (``file_uuid``, ``source_uuid``, ...), ``prod_host`` and the
``created`` / ``updated`` dates.  Two runs that agree with it produced the same physics.

With ``--rtol`` / ``--atol`` a float dataset whose values all agree within
``|a - b| <= atol + rtol * |b|`` (NaN equal to NaN) is reported as "within tolerance" and does
not count as a difference.  Integer, string and compound datasets are always compared exactly.

Exit code: 0 the same, 1 differences, 2 a file is missing or unreadable.

Needs only h5py and numpy (hdf5plugin for Blosc-compressed files).  Large datasets are read
block by block along their first axis (about ``--block-mb`` per file at a time, and at least
one entry of that axis), so native evolution files do not have to fit in memory.
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import sys

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover
    sys.exit("h5_compare.py: needs h5py (pip install h5py, or activate the js_fno / fno_env env)")

try:    # registers Blosc & co.; without it, reading a Blosc-compressed dataset fails
    import hdf5plugin  # noqa: F401
except ImportError:  # pragma: no cover
    pass

#: --ignore-stamps: what differs between any two runs of the same job
STAMPS = ("*wall_s*", "*@*uuid*", "*@prod_host", "*@created", "*@updated")
SUFFIXES = (".h5", ".hdf5")
NUMERIC = "fiubc"


# ── one file's content ───────────────────────────────────────────────────────────────────
def _walk(h):
    """-> (datasets {path: Dataset}, attributes {PATH@NAME: value}, groups {path})."""
    dsets, attrs, groups = {}, {}, set()

    def visit(name, obj):
        for k in obj.attrs:
            attrs[f"{name}@{k}"] = obj.attrs[k]
        if isinstance(obj, h5py.Dataset):
            dsets[name] = obj
        else:
            groups.add(name)

    h.visititems(visit)
    for k in h.attrs:
        attrs[f"@{k}"] = h.attrs[k]
    return dsets, attrs, groups


def _ignored(name, patterns):
    return any(fnmatch.fnmatchcase(name, p) for p in patterns)


# ── comparing values ─────────────────────────────────────────────────────────────────────
def _same_bytes(x, y):
    """Bit-exact for fixed-size dtypes; variable-length (object) values element by element."""
    if x.dtype.kind == "O" or x.dtype.hasobject:
        return x.tolist() == y.tolist()
    if x.flags.c_contiguous and y.flags.c_contiguous and x.dtype.itemsize:
        return np.array_equal(x.reshape(-1).view(np.uint8), y.reshape(-1).view(np.uint8))
    return x.tobytes() == y.tobytes()


def _blocks(ds, block_bytes):
    """Slices along the first axis, each of at most block_bytes (at least one row)."""
    if ds.shape == () or ds.shape[0] == 0 or ds.dtype.kind == "O":
        yield ()
        return
    row = max(ds.dtype.itemsize * int(np.prod(ds.shape[1:], dtype=np.int64)), 1)
    step = max(int(block_bytes // row), 1)
    for i in range(0, ds.shape[0], step):
        yield slice(i, min(i + step, ds.shape[0]))


class _Diff:
    """What accumulates over the blocks of one dataset pair."""

    def __init__(self):
        self.n = 0              # values that differ (NaN == NaN)
        self.bits_only = 0      # blocks whose bytes differ but whose values compare equal
        self.max_abs = 0.0
        self.max_rel = 0.0
        self.outside = 0        # values outside the tolerance
        self.first = None       # index of the first differing value

    def add(self, x, y, offset, rtol, atol):
        numeric = x.dtype.kind in NUMERIC and y.dtype.kind in NUMERIC
        if numeric and x.dtype.kind in "fc":
            ne = (x != y) & ~(np.isnan(x) & np.isnan(y))
        elif x.dtype.kind == "O":
            ne = np.array([a != b for a, b in zip(x.ravel().tolist(), y.ravel().tolist())],
                          dtype=bool).reshape(x.shape)
        elif x.dtype.names:     # compound: row by row
            ne = x != y
        else:
            ne = x != y
        ne = np.asarray(ne)
        n = int(np.count_nonzero(ne))
        if n == 0:
            self.bits_only += 1     # e.g. -0.0 against 0.0, or NaNs with another payload
            return
        if self.first is None:
            idx = [int(i) for i in np.argwhere(ne)[0]] if ne.ndim else []
            if idx and offset:
                idx[0] += offset
            self.first = tuple(idx)
        self.n += n
        if not numeric:
            self.outside += n
            return
        a = x[ne].astype(np.complex128 if "c" in (x.dtype.kind, y.dtype.kind) else np.float64)
        b = y[ne].astype(a.dtype)
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            d = np.abs(a - b)
            d = np.where(np.isnan(d), np.inf, d)        # NaN against a number, inf against -inf
            scale = np.maximum(np.abs(a), np.abs(b))
            rel = np.where(scale > 0, d / scale, 0.0)
            rel = np.where(np.isnan(rel), np.inf, rel)
            self.max_abs = max(self.max_abs, float(d.max()))
            self.max_rel = max(self.max_rel, float(rel.max()))
            if x.dtype.kind in "fc" and (rtol > 0 or atol > 0):
                self.outside += int(np.count_nonzero(~(d <= atol + rtol * np.abs(b))))
            else:
                self.outside += n


def _compare_dataset(da, db, block_bytes, rtol, atol):
    """-> None if the same, else (kind, text): kind 'differ' or 'tolerance'."""
    if da.shape != db.shape:
        return "differ", f"shape {da.shape} | {db.shape}"
    numeric = da.dtype.kind in NUMERIC and db.dtype.kind in NUMERIC
    dtype_note = ""
    if da.dtype != db.dtype:
        if not numeric:
            return "differ", f"dtype {da.dtype} | {db.dtype}"
        dtype_note = f"dtype {da.dtype} | {db.dtype}"
    size = int(np.prod(da.shape, dtype=np.int64))
    diff = _Diff()
    for sl in _blocks(da, block_bytes):
        x, y = np.asarray(da[sl]), np.asarray(db[sl])
        if not dtype_note and _same_bytes(x, y):
            continue
        diff.add(x, y, sl.start if isinstance(sl, slice) else 0, rtol, atol)
    if diff.n == 0:
        if dtype_note:
            return "differ", f"{dtype_note}; values equal"
        if diff.bits_only:
            return "differ", "bit patterns differ, values equal (-0.0 or NaN payloads)"
        return None
    text = f"{diff.n} of {size} values differ"
    if numeric:
        text += f", max |a-b| {diff.max_abs:.3g}, max rel {diff.max_rel:.3g}"
    if diff.first:
        text += f", first at [{', '.join(map(str, diff.first))}]"
    if dtype_note:
        return "differ", f"{dtype_note}; {text}"
    if diff.outside == 0:
        return "tolerance", text
    if rtol > 0 or atol > 0:
        text += f"; {diff.outside} outside the tolerance"
    return "differ", text


def _short(v, width=40):
    if isinstance(v, bytes):
        v = v.decode(errors="replace")
    if isinstance(v, np.ndarray) and v.size > 8:
        return f"array{v.shape} {v.dtype}"
    s = " ".join(str(v).split())
    return s if len(s) <= width else f"{s[:width - 12]}... ({len(s)} chars)"


def _same_attr(a, b):
    x, y = np.asarray(a), np.asarray(b)
    if x.shape != y.shape:
        return False
    if x.dtype.kind in "USO" and y.dtype.kind in "USO":     # strings: the text, not its storage
        return x.tolist() == y.tolist()
    return x.dtype == y.dtype and _same_bytes(x, y)


# ── one pair of files ────────────────────────────────────────────────────────────────────
def compare_files(path_a, path_b, *, ignore=(), attrs=True, rtol=0.0, atol=0.0,
                  block_mb=256, out=print, verbose=False):
    """Compare two HDF5 files and print the differences.  -> number of differences."""
    with h5py.File(path_a, "r") as ha, h5py.File(path_b, "r") as hb:
        da, aa, ga = _walk(ha)
        db, ab, gb = _walk(hb)
        skipped, lines, n_diff, n_tol = [], [], 0, 0

        names = sorted(set(da) | set(db))
        width = max([len(n) for n in names] + [len(n) for n in set(aa) | set(ab)] + [1])
        width = min(width, 44)
        same = 0
        for name in names:
            if _ignored(name, ignore):
                skipped.append(name)
            elif name not in db:
                kind = "a group in B" if name in gb else "only in A"
                lines.append(f"  {name:<{width}}  {kind}")
                n_diff += 1
            elif name not in da:
                kind = "a group in A" if name in ga else "only in B"
                lines.append(f"  {name:<{width}}  {kind}")
                n_diff += 1
            else:
                res = _compare_dataset(da[name], db[name], block_mb * 2 ** 20, rtol, atol)
                if res is None:
                    same += 1
                elif res[0] == "tolerance":
                    n_tol += 1
                    lines.append(f"  {name:<{width}}  within tolerance: {res[1]}")
                else:
                    n_diff += 1
                    lines.append(f"  {name:<{width}}  {res[1]}")
        lonely = [n for n in names if (n in da) != (n in db)]
        for name in sorted((ga ^ gb) - set(da) - set(db)):
            # a group whose datasets are already listed as "only in" needs no line of its own
            if _ignored(name, ignore) or any(n.startswith(name + "/") for n in lonely):
                continue
            lines.append(f"  {name + '/':<{width}}  group only in {'A' if name in ga else 'B'}")
            n_diff += 1
        tol = f", {n_tol} within tolerance" if n_tol else ""
        n_ds_diff = n_diff
        out(f"datasets:   {len(da)} in A, {len(db)} in B; {same} identical{tol}, "
            f"{n_ds_diff} differ")
        for line in lines:
            out(line)

        n_attr_diff = 0
        if attrs:
            lines, same = [], 0
            for name in sorted(set(aa) | set(ab)):
                if _ignored(name, ignore):
                    skipped.append(name)
                elif name not in ab:
                    lines.append(f"  {name:<{width}}  only in A: {_short(aa[name])}")
                elif name not in aa:
                    lines.append(f"  {name:<{width}}  only in B: {_short(ab[name])}")
                elif _same_attr(aa[name], ab[name]):
                    same += 1
                else:
                    lines.append(f"  {name:<{width}}  {_short(aa[name])} | {_short(ab[name])}")
            n_attr_diff = len(lines)
            out(f"attributes: {len(aa)} in A, {len(ab)} in B; {same} identical, "
                f"{n_attr_diff} differ")
            for line in lines:
                out(line)
        if skipped:
            out(f"ignored:    {len(skipped)}" + (": " + ", ".join(skipped) if verbose else
                                                 " (-v lists them)"))
    return n_ds_diff + n_attr_diff


# ── pairing files ────────────────────────────────────────────────────────────────────────
def _h5_files(top):
    found = {}
    for root, dirs, files in os.walk(top):
        dirs.sort()
        for f in sorted(files):
            if f.endswith(SUFFIXES):
                full = os.path.join(root, f)
                found[os.path.relpath(full, top)] = full
    return found


def _pairs(a, b):
    """-> (pairs [(label, path_a, path_b)], files only on one side [(label, side)])."""
    if os.path.isdir(a) and os.path.isdir(b):
        fa, fb = _h5_files(a), _h5_files(b)
        both = sorted(set(fa) & set(fb))
        lonely = [(n, "A" if n in fa else "B") for n in sorted(set(fa) ^ set(fb))]
        return [(n, fa[n], fb[n]) for n in both], lonely
    if os.path.isdir(a) != os.path.isdir(b):        # a file against the same name in a directory
        if os.path.isdir(a):
            a = os.path.join(a, os.path.basename(b))
        else:
            b = os.path.join(b, os.path.basename(a))
    return [(os.path.basename(a), a, b)], []


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog="Exit code: 0 the same, 1 differences, 2 a file is missing or unreadable.")
    p.add_argument("a", metavar="A", help="HDF5 file or directory")
    p.add_argument("b", metavar="B", help="HDF5 file or directory")
    p.add_argument("--ignore", action="append", default=[], metavar="PATTERN",
                   help="skip the datasets (PATH) and attributes (PATH@NAME, @NAME for the "
                        "file's) matching this shell pattern; can be repeated")
    p.add_argument("--ignore-stamps", action="store_true",
                   help="skip what differs between any two runs: " + ", ".join(STAMPS))
    p.add_argument("--no-attrs", action="store_true", help="compare the datasets only")
    p.add_argument("--rtol", type=float, default=0.0,
                   help="float datasets: relative tolerance (default 0: bit-exact)")
    p.add_argument("--atol", type=float, default=0.0,
                   help="float datasets: absolute tolerance (default 0)")
    p.add_argument("--block-mb", type=int, default=256, metavar="MB",
                   help="read about this much of a dataset at a time (default 256)")
    p.add_argument("-v", "--verbose", action="store_true", help="list the ignored names")
    p.add_argument("-q", "--quiet", action="store_true",
                   help="print only the result line of each pair")
    a = p.parse_args(argv)
    if a.rtol < 0 or a.atol < 0:
        p.error("--rtol and --atol must be >= 0")

    ignore = list(a.ignore) + (list(STAMPS) if a.ignore_stamps else [])
    for path in (a.a, a.b):
        if not os.path.exists(path):
            print(f"h5_compare.py: no such file or directory: {path}", file=sys.stderr)
            return 2
    pairs, lonely = _pairs(a.a, a.b)
    if not pairs and not lonely:
        print(f"h5_compare.py: no {' / '.join(SUFFIXES)} files in {a.a} and {a.b}",
              file=sys.stderr)
        return 2

    quiet = (lambda *_: None) if a.quiet else print
    differing, trouble = 0, False
    for label, pa, pb in pairs:
        quiet(f"A: {pa}\nB: {pb}")
        try:
            n = compare_files(pa, pb, ignore=ignore, attrs=not a.no_attrs, rtol=a.rtol,
                              atol=a.atol, block_mb=a.block_mb, out=quiet, verbose=a.verbose)
        except OSError as e:
            print(f"h5_compare.py: {label}: {e}", file=sys.stderr)
            trouble = True
            continue
        differing += n > 0
        print(f"{label}: " + ("the same" if n == 0 else f"DIFFERENT ({n} differences)"))
        quiet("")
    for label, side in lonely:
        print(f"{label}: only in {side}")
    if len(pairs) + len(lonely) > 1:
        print(f"{len(pairs)} pairs compared: {len(pairs) - differing} the same, "
              f"{differing} different" + (f"; {len(lonely)} files without a partner"
                                          if lonely else ""))
    if trouble:
        return 2
    return 1 if differing or lonely else 0


if __name__ == "__main__":
    sys.exit(main())
