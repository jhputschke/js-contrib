#!/usr/bin/env python3
"""Inspect an HDF5 file: every group and dataset with its shape, dtype, storage and
attributes, plus a schema summary for the hydro formats written in js-contrib.

Works on any HDF5 file.  The files js-contrib writes also get a summary section that
decodes their conventions:

* ``xscape/hydro_evolution`` -- PyJetscape ``H5BulkWriter`` (fast_h5_bulk), ``FnoH5Writer``
  and the ``PairH5Writer`` background/jet pairs (``prod_AuAu_0_10``, ``prod_AuAu_0_10_jet``).
* ``fast_data/hydro_evolution`` -- FastHydro / fast_data files, with ``source/S`` and the
  ``arr_bg`` wake pairs.
* legacy FNO4d files without a ``format`` attribute (``arr`` of shape
  ``(nevents, 4, nx, ny, neta, ntau)`` plus grid attributes), and FV-MUSIC IC files (``e0``).

The summary lists the output grid (and MUSIC's native grid, when recorded), the meaning of
each axis of the evolution arrays, per-event freeze-out, the ragged ``*offsets`` groups
(``shower/``, ``source/droplets``) with their column names and rows per event, and the
``diag/`` vectors.  It also flags inconsistencies, e.g. ``attrs['nevents'] != arr.shape[0]``.

Usage::

    python h5_inspect.py FILE.h5 [FILE2.h5 ...]
    python h5_inspect.py FILE.h5 --stats             # per-channel min/max/mean, NaN/inf, zeros
    python h5_inspect.py FILE.h5 --stats --event 0   # statistics of one event only
    python h5_inspect.py FILE.h5 --attr prod_user_xml   # print one attribute in full
    python h5_inspect.py FILE.h5 --attr shower@units    # attribute of an object: PATH@NAME
    python h5_inspect.py FILE.h5 --full              # long attributes and all siblings in full
    python h5_inspect.py FILE.h5 --no-tree           # summary only
    python h5_inspect.py FILE.h5 --json > meta.json  # machine-readable structure + attributes

Needs only h5py and numpy.  Nothing is read from the large datasets unless ``--stats`` is
given; the statistics are accumulated block by block (at most ``--block-mb`` at a time), so
native MUSIC files do not have to fit in memory.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover
    sys.exit("h5_inspect.py: needs h5py (pip install h5py, or activate the js_fno / fno_env env)")

try:    # registers Blosc & co.; without it, reading a Blosc-compressed dataset fails
    import hdf5plugin  # noqa: F401
except ImportError:  # pragma: no cover
    pass

#: registered HDF5 filter ids (https://github.com/HDFGroup/hdf5_plugins), for plugin
#: filters h5py does not name itself (its ds.compression is None for them)
_FILTER_NAMES = {32001: "blosc", 32004: "lz4", 32008: "bitshuffle", 32015: "zstd",
                 32026: "blosc2", 307: "bzip2", 32013: "zfp", 32017: "sz", 32024: "sz3"}


# ─────────────────────────────────────────────────────────────── formatting helpers

def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


def fmt_num(v) -> str:
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.7g}"
    return str(v)


def _decode(v):
    if isinstance(v, bytes):
        try:
            return v.decode()
        except UnicodeDecodeError:
            return repr(v)
    return v


def describe_text(s: str, width: int, full: bool) -> str:
    """One-line description of a (possibly long) string attribute."""
    if full or (len(s) <= width and "\n" not in s):
        return repr(s) if not full else s
    stripped = s.lstrip()
    nlines = s.count("\n") + 1
    if stripped[:1] in "{[":
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                keys = ", ".join(list(obj)[:12]) + (", ..." if len(obj) > 12 else "")
                return f"<JSON object, {len(s)} chars; keys: {keys}>"
            return f"<JSON {type(obj).__name__} of {len(obj)}, {len(s)} chars>"
        except ValueError:
            pass
    if stripped.startswith("<"):
        m = re.match(r"<\s*([\w:.-]+)", stripped)
        root = m.group(1) if m else "?"
        return f"<XML <{root}>, {nlines} lines, {len(s)} chars>"
    first = next((ln.strip() for ln in s.splitlines() if ln.strip()), "")
    if len(first) > width - 30:
        first = first[: max(10, width - 33)] + "..."
    if nlines > 1:
        return f"{first!r} ... <{nlines} lines, {len(s)} chars>"
    return f"{first!r} <{len(s)} chars>"


def describe_value(v, width: int = 100, full: bool = False) -> str:
    """Attribute / small-dataset value -> one line."""
    v = _decode(v)
    if isinstance(v, str):
        return describe_text(v, width, full)
    if isinstance(v, np.ndarray):
        if v.dtype.kind in "OSU":
            items = [_decode(x) for x in v.ravel()]
            items = [x if isinstance(x, str) else str(x) for x in items]
            if full or (v.size <= 24 and sum(len(x) for x in items) < 4 * width):
                if v.ndim <= 1:
                    return "[" + ", ".join(repr(x) for x in items) + "]"
                return f"{v.shape} " + repr(np.array(items).reshape(v.shape).tolist())
            return f"<{v.dtype} array {v.shape}; first: {items[0][:40]!r}>"
        if v.dtype.names:
            return f"<compound {v.shape} fields {list(v.dtype.names)}>"
        if full or v.size <= 12:
            if v.ndim == 0:
                return fmt_num(v.item())
            return "[" + ", ".join(fmt_num(x) for x in v.ravel()) + "]" + (
                "" if v.ndim <= 1 else f" (shape {v.shape})")
        finite = v[np.isfinite(v)] if v.dtype.kind == "f" else v
        rng = (f"{fmt_num(finite.min())} .. {fmt_num(finite.max())}" if finite.size else "all non-finite")
        return f"<{v.dtype} array {v.shape}; {rng}>"
    if isinstance(v, (np.generic, int, float, bool)):
        return fmt_num(v)
    if isinstance(v, h5py.Reference):
        return "<object reference>"
    return repr(v)


def dtype_str(dt: np.dtype) -> str:
    s = h5py.check_string_dtype(dt)
    if s is not None:
        return f"str({s.encoding}{', vlen' if s.length is None else f', {s.length}'})"
    vl = h5py.check_vlen_dtype(dt)
    if vl is not None:
        return f"vlen<{vl}>"
    if dt.names:
        return "compound{" + ", ".join(f"{n}:{dt.fields[n][0]}" for n in dt.names) + "}"
    return str(dt)


def compression_str(ds: h5py.Dataset) -> str:
    parts = []
    if ds.compression:
        c = ds.compression
        if ds.compression_opts is not None:
            c += f"({ds.compression_opts})"
        parts.append(c)
    else:
        # plugin filter: prefer the writer's own label (fast_data.h5_compression)
        label = ds.attrs.get("compression")
        plugins = [_FILTER_NAMES.get(int(k), f"filter{k}") for k in ds._filters
                   if str(k).isdigit()]
        if plugins:
            parts.append(label if isinstance(label, str) else "+".join(plugins))
    if "keep_mantissa_bits" in ds.attrs:
        parts.append(f"keep_bits={int(ds.attrs['keep_mantissa_bits'])}")
    if ds.shuffle:
        parts.append("shuffle")
    if ds.fletcher32:
        parts.append("fletcher32")
    if ds.scaleoffset is not None:
        parts.append(f"scaleoffset({ds.scaleoffset})")
    return "+".join(parts)


def storage_size(ds: h5py.Dataset) -> int:
    try:
        return int(ds.id.get_storage_size())
    except Exception:
        return 0


def logical_size(ds: h5py.Dataset) -> int:
    try:
        return int(np.prod(ds.shape, dtype=np.int64)) * ds.dtype.itemsize if ds.shape else ds.dtype.itemsize
    except Exception:
        return 0


# ─────────────────────────────────────────────────────────────── tree

_NUMBERED = re.compile(r"^(.*?)(\d+)$")


def _signature(obj) -> tuple:
    if isinstance(obj, h5py.Dataset):
        return ("D", obj.shape, str(obj.dtype))
    if isinstance(obj, h5py.Group):
        return ("G", tuple(sorted(obj.keys())))
    return ("?",)


def _collapse_runs(names: list[str], group: h5py.Group, min_run: int):
    """Group numbered siblings (Frame_0000, Frame_0001, ...) into runs.

    Returns a list of (name, None) for ordinary children and (first_name, run_names) for
    runs of at least ``min_run`` members sharing a prefix and digit width.
    """
    runs: dict[tuple, list[str]] = {}
    order: list = []
    for n in names:
        m = _NUMBERED.match(n)
        key = (m.group(1), len(m.group(2))) if m else None
        if key is None:
            order.append((n, None))
            continue
        if key not in runs:
            runs[key] = []
            order.append(key)
        runs[key].append(n)
    out = []
    for item in order:
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], int):
            members = sorted(runs[item], key=lambda n: int(_NUMBERED.match(n).group(2)))
            nums = [int(_NUMBERED.match(n).group(2)) for n in members]
            consecutive = nums == list(range(nums[0], nums[0] + len(nums)))
            if len(members) >= min_run and consecutive:  # not e.g. Pi00, Pi01, ..., Pi33
                out.append((members[0], members))
            else:
                out.extend((n, None) for n in members)
        else:
            out.append(item)
    return out


class TreePrinter:
    def __init__(self, width=100, full=False, collapse=4, show_attrs=True, out=sys.stdout):
        self.width = width
        self.full = full
        self.collapse = 0 if full else collapse
        self.show_attrs = show_attrs
        self.out = out

    def p(self, s=""):
        print(s, file=self.out)

    def attrs(self, obj, indent):
        if not self.show_attrs:
            return
        try:
            items = list(obj.attrs.items())
        except Exception as e:  # unreadable attribute type
            self.p(f"{indent}  @<attributes unreadable: {e}>")
            return
        if not items:
            return
        kw = min(max(len(k) for k, _ in items), 28)
        for k, v in items:
            self.p(f"{indent}  @{k:<{kw}} = {describe_value(v, self.width, self.full)}")

    def dataset_line(self, name, ds):
        shape = "scalar" if ds.shape == () else "(" + ", ".join(str(s) for s in ds.shape) + ")"
        bits = [f"{shape} {dtype_str(ds.dtype)}"]
        ls, ss = logical_size(ds), storage_size(ds)
        if ls >= 1024:
            size = human_bytes(ls)
            if ss and ss != ls:
                size += f" -> {human_bytes(ss)} on disk ({ls / ss:.1f}x)" if ss else ""
            bits.append(size)
        if ds.chunks:
            bits.append("chunks (" + ", ".join(str(c) for c in ds.chunks) + ")")
        c = compression_str(ds)
        if c:
            bits.append(c)
        if ds.maxshape and ds.maxshape != ds.shape:
            bits.append("maxshape (" + ", ".join("inf" if m is None else str(m)
                                                  for m in ds.maxshape) + ")")
        line = f"{name}  " + " | ".join(bits)
        # tiny datasets: show the values inline (offsets, versions, scalars)
        if ds.shape == () or (0 < ds.size <= 8 and ds.dtype.kind in "biufSUO?"):
            try:
                line += "  = " + describe_value(ds[()], self.width, False)
            except Exception:
                pass
        return line

    def walk(self, group: h5py.Group, indent=""):
        names = list(group.keys())
        entries = _collapse_runs(names, group, self.collapse) if self.collapse else [(n, None) for n in names]
        for name, run in entries:
            link = group.get(name, getlink=True)
            if isinstance(link, h5py.SoftLink):
                self.p(f"{indent}{name} -> soft link to {link.path}")
                continue
            if isinstance(link, h5py.ExternalLink):
                self.p(f"{indent}{name} -> external link {link.filename}:{link.path}")
                continue
            try:
                obj = group[name]
            except Exception as e:
                self.p(f"{indent}{name}  <unreadable: {e}>")
                continue
            if isinstance(obj, h5py.Dataset):
                self.p(indent + self.dataset_line(name, obj))
                self.attrs(obj, indent)
            elif isinstance(obj, h5py.Group):
                self.p(f"{indent}{name}/  ({len(obj)} members)")
                self.attrs(obj, indent)
                self.walk(obj, indent + "    ")
            else:
                self.p(f"{indent}{name}  <{type(obj).__name__}>")
            if run:
                last = group[run[-1]]
                same = _signature(last) == _signature(obj)
                self.p(f"{indent}... + {len(run) - 1} more like {name!r} ({run[1]} .. {run[-1]}), "
                       + ("same structure" if same else
                          f"structure differs, last: {_signature(last)[:2]}")
                       + "   [--full shows all]")


# ─────────────────────────────────────────────────────────────── schema summary

_AXES = {  # label: (n, min, step) attribute names
    "x": ("nx", "x_min", "dx"),
    "y": ("ny", "y_min", "dy"),
    "eta": ("neta", "eta_min", "deta"),
    "tau": ("choose_ntau", "tau_min", "dtau"),
}
_AXES_MUSIC = {
    "x": ("nX_MUSIC", "X_min_MUSIC", "dX_MUSIC"),
    "y": ("nY_MUSIC", "Y_min_MUSIC", "dY_MUSIC"),
    "eta": ("neta_MUSIC", "eta_min_MUSIC", "deta_MUSIC"),
    "tau": ("ntau_MUSIC", "tau_min_MUSIC", "dtau_MUSIC"),
}
_EVOLUTION_AXES = ("event", "feature", "x", "y", "eta", "tau")
_IC_AXES = ("event", "x", "y", "eta")


def _a(attrs, k, default=None):
    if not k or k not in attrs:
        return default
    v = _decode(attrs[k])
    if isinstance(v, np.ndarray) and v.shape == (1,):
        v = v[0]
    if isinstance(v, np.generic):
        v = v.item()
    return v


def detect_format(f: h5py.File) -> str:
    a = f.attrs
    fmt = _a(a, "format")
    if fmt:
        ver = _a(a, "format_version")
        s = f"{fmt}" + (f" v{ver}" if ver is not None else "")
        extra = [x for x in (_a(a, "generator"), _a(a, "xscape_writer") or _a(a, "producer")) if x]
        if _a(a, "pairing"):
            extra.append(f"pairing={_a(a, 'pairing')}")
        return s + (f"  ({'; '.join(map(str, extra))})" if extra else "")
    if "arr" in f and isinstance(f["arr"], h5py.Dataset) and f["arr"].ndim == 6:
        return "FNO4d hydro evolution, legacy (no 'format' attribute)"
    if "e0" in f and isinstance(f["e0"], h5py.Dataset):
        return "initial condition file (e0: event, x, y, eta), e.g. FastHydro/FV-MUSIC IC"
    if "Event" in f and isinstance(f["Event"], h5py.Group) and any(k.startswith("Frame_") for k in f["Event"]):
        return "JETSCAPE/MUSIC 2+1D hydro frames (Event/Frame_NNNN)"
    if {"init", "event", "particle"} <= set(f.keys()):
        return "Pythia 8 LHEF-HDF5 (LHAHDF5) event file"
    if any(k.startswith("event_") for k in f.keys()):
        return "per-event groups (event_N), e.g. TRENTo/initial-state output"
    return "generic HDF5 (no known schema)"


def _axis_line(label, attrs, keys, n_override=None):
    kn, kmin, kd = keys
    n = n_override if n_override is not None else _a(attrs, kn)
    lo, d = _a(attrs, kmin), _a(attrs, kd)
    if n is None and lo is None and d is None:
        return None
    s = f"  {label:<4} n = {n if n is not None else '?':<5}"
    if lo is not None and d is not None and n is not None:
        s += f" {lo:9.5g} .. {lo + (int(n) - 1) * d:<9.5g} step {d:.6g}"
    else:
        s += "".join(f"  {lab} {v:.6g}" for lab, v in (("min", lo), ("step", d)) if v is not None)
    return s


def _ragged_pairs(g: h5py.Group):
    """Yield (offsets_name, data_name, columns) for every ragged table in group g."""
    names = [k for k in g if isinstance(g[k], h5py.Dataset)]
    for on in names:
        if not on.endswith("offsets"):
            continue
        stem = on[: -len("offsets")].rstrip("_")
        cands = []
        if stem:  # parton_offsets -> partons, vertex_offsets -> vertices, ...
            forms = {stem, stem + "s", stem + "es", re.sub("ex$", "ices", stem),
                     re.sub("y$", "ies", stem)}
            cands = [n for n in names if n != on and n in forms]
        else:  # plain "offsets": the other 1D/2D tables with more than one row
            cands = [n for n in names if n != on and not n.endswith("offsets")
                     and g[n].ndim in (1, 2) and g[n].shape[0] != g[on].shape[0]]
        for dn in cands[:1] if stem else cands:
            if g[dn].shape[:1] == () or g[dn].ndim > 2:
                continue
            col_key = None
            for k in g.attrs:
                if not k.endswith("_columns"):
                    continue
                base = k[: -len("_columns")]
                if dn.startswith(base) or (stem and stem.startswith(base)):
                    col_key = k
                    break
            cols = list(map(_decode, g.attrs[col_key])) if col_key else None
            yield on, dn, cols


def summary(f: h5py.File, max_events: int, out=sys.stdout):
    p = lambda s="": print(s, file=out)
    a = f.attrs
    warnings = []
    p(f"format   {detect_format(f)}")

    # ── events
    evo = [n for n in ("arr", "arr_bg", "source/S") if n in f and isinstance(f[n], h5py.Dataset)]
    lead = f["arr"] if "arr" in evo else (f["e0"] if "e0" in f else None)
    n_attr, n_written, complete = _a(a, "nevents"), _a(a, "nevents_written"), _a(a, "complete")
    if lead is not None or n_attr is not None:
        s = f"events   {lead.shape[0] if lead is not None else '?'} in the file"
        if n_attr is not None:
            s += f"; attrs nevents = {n_attr}"
        if n_written is not None:
            s += f", nevents_written = {n_written}"
        if complete is not None:
            s += f", complete = {bool(complete)}"
        p(s)
        if lead is not None and n_attr is not None and int(n_attr) != lead.shape[0]:
            warnings.append(f"attrs['nevents'] = {n_attr} but {lead.name} has {lead.shape[0]} events "
                            "(read_3d_data_hdf5 trusts the attribute)")
        if complete is not None and not bool(complete):
            warnings.append("complete = False: the writer did not finish (crash or still running)")

    # ── physics descriptors worth seeing at a glance
    keys = ["system", "initial_state_kind", "hydro", "solver", "transport_mode", "eos_kind",
            "eos_desc", "T_fo", "eta_over_s", "jet", "source_model", "source_mode", "deposition",
            "hard_vertex", "prod_hard", "prod_seed", "master_seed", "device", "prod_host",
            "prod_platform", "xscape_grid_mode"]
    shown = [(k, _a(a, k)) for k in keys if k in a]
    if shown:
        p("physics")
        for k, v in shown:
            p(f"  {k:<20} {describe_value(v, 90)}")

    # ── grid
    if lead is not None and lead.ndim in (4, 6) and any(k in a for k in ("nx", "dx", "x_min")):
        p("grid (output)")
        for label, keys_ in _AXES.items():
            n_over = None
            if label == "tau" and "arr" in evo:
                n_over = f["arr"].shape[5]
            line = _axis_line(label, a, keys_, n_over)
            if line:
                p(line)
        # consistency of the spatial shape with the attributes
        sp = lead.shape[2:5] if lead.ndim == 6 else lead.shape[1:4]
        want = tuple(_a(a, k) for k in ("nx", "ny", "neta"))
        if None not in want and tuple(int(w) for w in want) != tuple(sp):
            warnings.append(f"{lead.name} spatial shape {sp} != attrs (nx, ny, neta) = {want}")
        if "arr" in evo and _a(a, "choose_ntau") is not None and int(_a(a, "choose_ntau")) != f["arr"].shape[5]:
            warnings.append(f"attrs['choose_ntau'] = {_a(a, 'choose_ntau')} but arr has "
                            f"{f['arr'].shape[5]} tau frames")
    if any(k in a for k in _AXES_MUSIC["x"]):
        p("grid (MUSIC native)")
        for label, keys_ in _AXES_MUSIC.items():
            line = _axis_line(label, a, keys_)
            if line:
                p(line)

    # ── evolution arrays
    feats = _a(a, "feature_names")
    feats = [str(_decode(x)) for x in feats] if feats is not None else None
    if evo or lead is not None:
        p("arrays")
    for n in evo:
        ds = f[n]
        axes = _EVOLUTION_AXES if ds.ndim == 6 else tuple(f"dim{i}" for i in range(ds.ndim))
        chans = None
        if n == "source/S" and "channels" in f["source"].attrs:
            chans = [str(_decode(x)) for x in f["source"].attrs["channels"]]
        elif ds.ndim == 6:
            chans = feats or ["energy_density", "vx", "vy", "vz"][: ds.shape[1]]
        dims = ", ".join(f"{ax}={s}" for ax, s in zip(axes, ds.shape))
        p(f"  {n:<9} ({dims})")
        if chans:
            p(f"            features: {', '.join(chans)}"
              + ("" if feats or n == "source/S" else "   (assumed: no feature_names attr)"))
        ta = _a(ds.attrs, "tau_axis")
        if ta is not None and int(ta) != 5:
            warnings.append(f"{n}: tau_axis = {ta}, not the last axis")
        if ds.ndim == 6 and feats and ds.shape[1] != len(feats) and n != "source/S":
            warnings.append(f"{n} has {ds.shape[1]} features but feature_names lists {len(feats)}")
        extra = _a(a, "arr_is" if n == "arr" else "arr_bg_is" if n == "arr_bg" else "")
        if extra:
            p(f"            is: {describe_value(extra, 90)}")
    if "e0" in f and isinstance(f["e0"], h5py.Dataset) and f["e0"].ndim == 4:
        p(f"  e0        ({', '.join(f'{ax}={s}' for ax, s in zip(_IC_AXES, f['e0'].shape))})")

    # ── freeze-out per event
    for suf in ("", "_bg"):
        nt = f.get(f"ntau_freezeout{suf}")
        tt = f.get(f"tau_freezeout{suf}")
        if not isinstance(nt, h5py.Dataset):
            continue
        ntv = nt[()]
        s = (f"freeze-out{' (bg leg)' if suf else ''}: ntau_freezeout{suf} "
             f"{ntv.min()} .. {ntv.max()}")
        if isinstance(tt, h5py.Dataset):
            ttv = tt[()]
            s += f", tau_freezeout{suf} {ttv.min():.4g} .. {ttv.max():.4g} fm/c"
        p(s)
        conv = _a(a, "freezeout_convention_id") or _a(a, "freezeout_rule")
        if conv and not suf:
            p(f"  convention: {conv}")
        if "arr" in evo and ntv.size and ntv.max() > f["arr"].shape[5]:
            warnings.append(f"ntau_freezeout{suf} max {ntv.max()} > tau frames {f['arr'].shape[5]}")

    # ── ragged per-event tables
    ragged = []
    f.visititems(lambda name, obj: ragged.append(obj) if isinstance(obj, h5py.Group) else None)
    ragged.insert(0, f)
    lines = []
    for g in ragged:
        try:
            pairs = list(_ragged_pairs(g))
        except Exception:
            continue
        for on, dn, cols in pairs:
            off = g[on][()]
            data = g[dn]
            if off.ndim != 1 or off.size < 1:
                continue
            per = np.diff(off)
            path = (g.name.rstrip("/") + "/" + dn).lstrip("/")
            s = (f"  {path:<22} {data.shape[0]:>8} rows x {data.shape[1] if data.ndim == 2 else 1:<3}"
                 f" over {off.size - 1} events")
            if per.size:
                s += f"; per event {per.min()} .. {per.max()} (mean {per.mean():.1f})"
            lines.append(s)
            if cols:
                lines.append(f"  {'':<22} columns: {', '.join(map(str, cols))}")
            if off.size and off[-1] != data.shape[0]:
                warnings.append(f"{g.name}/{on}[-1] = {off[-1]} but {dn} has {data.shape[0]} rows")
            if lead is not None and off.size - 1 != lead.shape[0]:
                warnings.append(f"{path}: {off.size - 1} events in {on}, {lead.shape[0]} in {lead.name}")
    if lines:
        p("ragged tables (event i = rows offsets[i]:offsets[i+1])")
        for ln in lines:
            p(ln)

    # ── diag vectors
    for gname in ("diag",):
        g = f.get(gname)
        if not isinstance(g, h5py.Group):
            continue
        vecs = {k: g[k] for k in g if isinstance(g[k], h5py.Dataset) and g[k].ndim == 1}
        if not vecs:
            continue
        n = max(v.shape[0] for v in vecs.values())
        p(f"{gname}/ ({len(vecs)} per-event vectors, {n} events)")
        if n <= max_events:
            kw = max(len(k) for k in vecs)
            for k, ds in vecs.items():
                vals = [describe_value(x, 30) if ds.dtype.kind in "SUO" else fmt_num(x) for x in ds[()]]
                p(f"  {k:<{kw}}  " + "  ".join(vals))
        else:
            for k, ds in vecs.items():
                v = ds[()]
                if v.dtype.kind in "biuf":
                    p(f"  {k:<24} min {fmt_num(v.min())}  mean {fmt_num(v.mean())}  max {fmt_num(v.max())}")
                elif v.dtype.kind == "b":
                    p(f"  {k:<24} {int(v.sum())}/{v.size} true")
                else:
                    p(f"  {k:<24} {v.dtype} e.g. {describe_value(v[0], 40)}")

    if warnings:
        p("WARNINGS")
        for w in warnings:
            p(f"  ! {w}")
    return warnings


# ─────────────────────────────────────────────────────────────── statistics

def _blocks(ds: h5py.Dataset, event, block_bytes: int):
    """Yield (feature-axis-first numpy blocks) covering ds, at most ~block_bytes each.

    For 6D evolution arrays the block is sliced along the event axis and then along tau.
    """
    shape = ds.shape
    if ds.ndim == 0:
        yield np.asarray(ds[()])
        return
    ev = range(shape[0]) if event is None else [event]
    per_event = int(np.prod(shape[1:], dtype=np.int64)) * ds.dtype.itemsize
    if ds.ndim >= 2 and per_event > block_bytes:
        per_last = max(1, per_event // shape[-1])
        step = max(1, block_bytes // per_last)
        for e in ev:
            for t0 in range(0, shape[-1], step):
                yield ds[e, ..., t0:t0 + step][None]
    else:
        step = max(1, block_bytes // max(1, per_event))
        idx = list(ev)
        for i in range(0, len(idx), step):
            sl = idx[i:i + step]
            yield ds[sl[0]:sl[-1] + 1]


def _ragged_index(f: h5py.File) -> tuple[dict, set]:
    """-> ({data path: (offsets array, columns)}, {offsets paths}) over the whole file."""
    groups = [f]
    f.visititems(lambda n, o: groups.append(o) if isinstance(o, h5py.Group) else None)
    data, offs = {}, set()
    for g in groups:
        try:
            pairs = list(_ragged_pairs(g))
        except Exception:
            continue
        for on, dn, cols in pairs:
            off = g[on][()]
            if off.ndim == 1 and off.size >= 1:
                data[g[dn].name] = (off, cols)
                offs.add(g[on].name)
    return data, offs


def _accumulate(blocks, nch, chan_axis):
    acc = [dict(n=0, mn=np.inf, mx=-np.inf, s=0.0, nan=0, inf=0, zero=0) for _ in range(nch)]
    for blk in blocks:
        blk = np.asarray(blk)
        for c in range(nch):
            x = np.take(blk, c, axis=chan_axis) if chan_axis is not None else blk
            r = acc[c]
            r["n"] += x.size
            good = x
            if x.dtype.kind == "f":
                bad = ~np.isfinite(x)
                if bad.any():
                    r["nan"] += int(np.isnan(x).sum())
                    r["inf"] += int(np.isinf(x).sum())
                    good = x[~bad]
            if good.size:
                r["mn"] = min(r["mn"], float(good.min()))
                r["mx"] = max(r["mx"], float(good.max()))
                r["s"] += float(good.sum(dtype=np.float64))
            r["zero"] += int(np.count_nonzero(x == 0))
    return acc


def stats(f: h5py.File, event, block_mb: int, out=sys.stdout):
    """min/max/mean/zeros/non-finite per dataset; per feature for 6D evolution arrays and
    per column for ragged tables.  With ``event``: per-event datasets are sliced to that
    event and ragged tables to its rows; other datasets are skipped."""
    p = lambda s="": print(s, file=out)
    block = block_mb << 20
    feats = _a(f.attrs, "feature_names")
    feats = [str(_decode(x)) for x in feats] if feats is not None else None
    lead = f.get("arr") if isinstance(f.get("arr"), h5py.Dataset) else f.get("e0")
    nev = lead.shape[0] if isinstance(lead, h5py.Dataset) and lead.ndim else None
    ragged, offsets = _ragged_index(f)
    targets = []
    f.visititems(lambda n, o: targets.append(o) if isinstance(o, h5py.Dataset)
                 and o.dtype.kind in "biuf" and o.size > 0 and not o.dtype.names else None)
    p(f"statistics{'' if event is None else f' (event {event})'}")
    for ds in targets:
        if ds.name in offsets:
            continue
        names, chan_axis = [""], None
        if ds.name in ragged:
            off, cols = ragged[ds.name]
            if event is not None and event + 1 >= off.size:
                continue
            r0, r1 = (0, ds.shape[0]) if event is None else (int(off[event]), int(off[event + 1]))
            if ds.ndim == 2:
                names = cols if cols and len(cols) == ds.shape[1] else [f"c{i}" for i in range(ds.shape[1])]
                chan_axis = 1
            rows = max(1, block // max(1, ds.dtype.itemsize * int(np.prod(ds.shape[1:], dtype=np.int64))))
            blocks = (ds[i:min(i + rows, r1)] for i in range(r0, r1, rows))
        else:
            if event is not None and (nev is None or ds.ndim == 0 or ds.shape[0] != nev or event >= nev):
                continue
            if ds.ndim == 6:
                nch = ds.shape[1]
                if ds.name == "/source/S" and "channels" in f["source"].attrs:
                    names = [str(_decode(x)) for x in f["source"].attrs["channels"]]
                elif feats and len(feats) == nch:
                    names = feats
                elif nch == 4:  # legacy FNO4d files carry no feature_names
                    names = ["energy_density", "vx", "vy", "vz"]
                else:
                    names = [f"ch{i}" for i in range(nch)]
                chan_axis = 1
            blocks = _blocks(ds, event, block)
        acc = _accumulate(blocks, len(names), chan_axis)
        for c, r in enumerate(acc):
            if r["n"] == 0:
                continue
            label = ds.name.lstrip("/") + (f"[{names[c]}]" if chan_axis is not None else "")
            good_n = r["n"] - r["nan"] - r["inf"]
            mean = r["s"] / good_n if good_n else float("nan")
            pct = 100.0 * r["zero"] / r["n"]
            zeros = ">99.9" if (r["zero"] < r["n"] and pct >= 99.95) else f"{pct:5.1f}"
            if good_n:
                s = (f"  {label:<32} min {r['mn']:<11.5g} max {r['mx']:<11.5g} mean {mean:<11.5g}"
                     f" zeros {zeros}%")
            else:
                s = f"  {label:<32} {'all values non-finite':<55}"
            if r["nan"] or r["inf"]:
                s += f"  NaN {r['nan']}  inf {r['inf']}  <-- non-finite values"
            p(s)


# ─────────────────────────────────────────────────────────────── JSON dump

def to_json(f: h5py.File) -> dict:
    def attrs_of(obj):
        out = {}
        for k, v in obj.attrs.items():
            v = _decode(v)
            if isinstance(v, np.ndarray):
                v = [(_decode(x) if isinstance(x, bytes) else x) for x in v.tolist()]
            elif isinstance(v, np.generic):
                v = v.item()
            out[k] = v
        return out

    def node(obj):
        if isinstance(obj, h5py.Dataset):
            return {"kind": "dataset", "shape": list(obj.shape), "dtype": dtype_str(obj.dtype),
                    "chunks": list(obj.chunks) if obj.chunks else None,
                    "maxshape": [None if m is None else m for m in obj.maxshape] if obj.maxshape else None,
                    "compression": compression_str(obj) or None,
                    "bytes": logical_size(obj), "bytes_on_disk": storage_size(obj),
                    "attrs": attrs_of(obj)}
        members = {}
        for k in obj:
            link = obj.get(k, getlink=True)
            if isinstance(link, h5py.SoftLink):
                members[k] = {"kind": "softlink", "path": link.path}
            elif isinstance(link, h5py.ExternalLink):
                members[k] = {"kind": "externallink", "file": link.filename, "path": link.path}
            else:
                try:
                    members[k] = node(obj[k])
                except Exception as e:
                    members[k] = {"kind": "unreadable", "error": str(e)}
        return {"kind": "group", "attrs": attrs_of(obj), "members": members}

    d = node(f)
    d["file"] = os.path.abspath(f.filename)
    d["format"] = detect_format(f)
    return d


# ─────────────────────────────────────────────────────────────── main

def print_attr(f: h5py.File, spec: str) -> int:
    path, _, name = spec.rpartition("@")
    obj = f[path] if path else f
    if name not in obj.attrs:
        avail = ", ".join(obj.attrs.keys())
        print(f"h5_inspect.py: no attribute {name!r} on {obj.name}. Available: {avail}", file=sys.stderr)
        return 1
    v = _decode(obj.attrs[name])
    if isinstance(v, np.ndarray):
        v = "\n".join(str(_decode(x)) for x in v.ravel()) if v.dtype.kind in "OSU" else np.array2string(
            v, threshold=sys.maxsize)
    print(v)
    return 0


def inspect_file(path, args) -> int:
    try:
        f = h5py.File(path, "r")
    except OSError as e:
        print(f"h5_inspect.py: cannot open {path}: {e}", file=sys.stderr)
        return 1
    with f:
        if args.attr:
            return print_attr(f, args.attr)
        if args.json:
            json.dump(to_json(f), sys.stdout, indent=1, default=str)
            print()
            return 0
        size = os.path.getsize(path)
        bar = "=" * min(100, max(40, len(path) + 6))
        print(bar)
        print(f"file     {os.path.abspath(path)}  ({human_bytes(size)})")
        rc = 0
        if not args.no_summary:
            w = summary(f, args.max_events)
            rc = 2 if (w and args.strict) else 0
        if not args.no_tree:
            print("-" * len(bar))
            print("/  (root attributes)" if f.attrs else "/")
            tp = TreePrinter(width=args.width, full=args.full, collapse=args.collapse,
                             show_attrs=not args.no_attrs)
            tp.attrs(f, "")
            tp.walk(f, "")
        if args.stats:
            print("-" * len(bar))
            stats(f, args.event, args.block_mb)
        return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Show every HDF5 entry with its dimensions, storage and attributes, plus a "
                    "schema summary for the js-contrib hydro formats.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("Usage::")[1])
    ap.add_argument("files", nargs="+", help="HDF5 file(s)")
    ap.add_argument("--stats", action="store_true",
                    help="read numeric datasets and print min/max/mean, zero fraction, NaN/inf "
                         "(per feature for the 6D evolution arrays)")
    ap.add_argument("--event", type=int, default=None, help="with --stats: only this event")
    ap.add_argument("--block-mb", type=int, default=256, help="--stats read block size (MB)")
    ap.add_argument("--attr", metavar="[PATH@]NAME", help="print one attribute in full and exit")
    ap.add_argument("--full", action="store_true",
                    help="print long attributes in full and do not collapse numbered siblings")
    ap.add_argument("--width", type=int, default=100, help="truncate attribute text at this width")
    ap.add_argument("--collapse", type=int, default=4,
                    help="collapse runs of at least N numbered siblings (0 = never)")
    ap.add_argument("--max-events", type=int, default=12,
                    help="print diag/ per event up to this many events, else min/mean/max")
    ap.add_argument("--no-tree", action="store_true", help="summary only")
    ap.add_argument("--no-summary", action="store_true", help="tree only")
    ap.add_argument("--no-attrs", action="store_true", help="tree without attributes")
    ap.add_argument("--json", action="store_true", help="dump structure and attributes as JSON")
    ap.add_argument("--strict", action="store_true", help="exit 2 if the summary has warnings")
    args = ap.parse_args(argv)
    rc = 0
    for path in args.files:
        rc = max(rc, inspect_file(path, args))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
