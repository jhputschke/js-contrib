"""HDF5 compression for the FNO4d ``arr`` schema: filter specs, mantissa rounding, plugins.

Self-contained (stdlib + numpy + h5py; ``hdf5plugin`` optional) so it can be vendored
verbatim -- js-contrib carries copies in FastHydro (``fast_data/``, via
``tools/sync_fast_data.sh``) and PyJetscape (``jetscape/h5_compression.py``).  The
measurements behind the defaults are in README_h5_optim.md.

Why not lzf: float32 hydro fields are nearly incompressible for a byte-oriented codec
without a shuffle filter.  lzf stores MUSIC ``arr`` at 1.11x (fast_data 1.25x); Blosc-zstd
with byte shuffle reaches 1.6x (2.0x) and reads as fast or, with ``BLOSC_NTHREADS`` and
whole-event chunks, twice as fast.  Bitshuffle loses on lossless data -- the low mantissa
bits are noise -- and after ``keep_bits`` rounding it helps only the LZ4 family; zstd keeps
byte shuffle.  Rounding to 12 bits takes the total to 2.8-4x.

Spec strings, as accepted by :func:`h5_filter_kwargs` (and every writer's
``compression=``)::

    "blosc-zstd"            Blosc + zstd level 3, byte shuffle      (DEFAULT)
    "blosc-lz4"             Blosc + LZ4 level 5, byte shuffle       (fastest write)
    "blosc-<cname>[:level][+shuffle|+bitshuffle|+noshuffle]"
                            any Blosc codec: blosclz lz4 lz4hc snappy zlib zstd
    "lzf"                   h5py's built-in LZF (the old default)
    "gzip[:level]"          deflate, level 4 by default, with byte shuffle
    None / "none"           uncompressed

A mapping (h5py keyword arguments, or an ``hdf5plugin`` filter object) is passed through
unchanged.  With ``keep_bits`` set, an LZ4/blosclz spec without an explicit shuffle switches
to bitshuffle (the only case where it was measured to win).

Reading a Blosc file needs the filter registered, i.e. ``import hdf5plugin`` somewhere in
the process before the first read -- importing this module does that.  Without
``hdf5plugin`` a Blosc spec falls back to lzf with a warning, so a writer never fails on
a missing optional package.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping

import numpy as np

__all__ = ["DEFAULT", "HAVE_HDF5PLUGIN", "enable_filters", "h5_filter_kwargs",
           "round_mantissa", "tag_dataset", "dataset_keep_bits"]

#: the recommended lossless default (README_h5_optim.md)
DEFAULT = "blosc-zstd"

_BLOSC_CODECS = ("blosclz", "lz4", "lz4hc", "snappy", "zlib", "zstd")
_BLOSC_LEVEL = {"zstd": 3, "zlib": 4}          # everything else: 5
#: codecs that compress rounded data better with bitshuffle than with byte shuffle
_BITSHUFFLE_WHEN_ROUNDED = ("blosclz", "lz4", "lz4hc")
_FLOAT32_MANTISSA = 23


def enable_filters():
    """Register the ``hdf5plugin`` filters (Blosc, ...) with HDF5.  True if available.

    Idempotent and cheap.  Call it (or import this module) before opening a file written
    with a Blosc spec; h5py otherwise fails with "required filter ... is not registered".
    """
    try:
        import hdf5plugin  # noqa: F401
    except ImportError:
        return False
    return True


HAVE_HDF5PLUGIN = enable_filters()


def _parse_blosc(spec, keep_bits):
    body, _, shuf = spec[len("blosc-"):].partition("+")
    cname, _, level = body.partition(":")
    if cname not in _BLOSC_CODECS:
        raise ValueError(f"compression {spec!r}: unknown Blosc codec {cname!r} "
                         f"(one of {', '.join(_BLOSC_CODECS)})")
    clevel = int(level) if level else _BLOSC_LEVEL.get(cname, 5)
    if not shuf:
        rounded = keep_bits is not None and int(keep_bits) < _FLOAT32_MANTISSA
        shuf = "bitshuffle" if rounded and cname in _BITSHUFFLE_WHEN_ROUNDED else "shuffle"
    if shuf not in ("shuffle", "bitshuffle", "noshuffle"):
        raise ValueError(f"compression {spec!r}: unknown shuffle {shuf!r} "
                         "(use +shuffle, +bitshuffle or +noshuffle)")
    return cname, clevel, shuf


def h5_filter_kwargs(spec=DEFAULT, keep_bits=None):
    """``(kwargs, label)`` for ``h5py.Group.create_dataset(..., **kwargs)``.

    ``label`` is a canonical string for the dataset's ``compression`` attribute (see
    :func:`tag_dataset`), e.g. ``"blosc-zstd:3+shuffle"``.  Every label except a
    ``custom:`` one is itself a valid spec that gives the same filter back.
    """
    if spec is None or (isinstance(spec, str) and spec.lower() in ("", "none")):
        return {}, "none"
    if isinstance(spec, Mapping):
        return dict(spec), f"custom:{dict(spec).get('compression', '?')}"
    if not isinstance(spec, str):
        raise TypeError(f"compression must be a str, a mapping or None, got {type(spec)}")
    s = spec.lower()
    if s == "lzf":
        return {"compression": "lzf"}, "lzf"
    if s == "gzip" or s.startswith("gzip:"):
        level = int(s.partition(":")[2].partition("+")[0] or 4)   # "gzip:4+shuffle" too
        return {"compression": "gzip", "compression_opts": level, "shuffle": True}, \
            f"gzip:{level}+shuffle"
    if s.startswith("blosc-"):
        cname, clevel, shuf = _parse_blosc(s, keep_bits)
        if not HAVE_HDF5PLUGIN:
            warnings.warn(f"compression {spec!r} needs the hdf5plugin package, which is not "
                          "installed; writing lzf instead (pip install hdf5plugin)",
                          RuntimeWarning, stacklevel=2)
            return {"compression": "lzf"}, "lzf"
        import hdf5plugin

        B = hdf5plugin.Blosc
        shuffle = {"shuffle": B.SHUFFLE, "bitshuffle": B.BITSHUFFLE,
                   "noshuffle": B.NOSHUFFLE}[shuf]
        flt = B(cname=cname, clevel=clevel, shuffle=shuffle)
        return dict(flt), f"blosc-{cname}:{clevel}+{shuf}"
    raise ValueError(f"unknown compression {spec!r}; use 'blosc-zstd', 'blosc-lz4', "
                     "'blosc-<cname>[:level][+shuffle|+bitshuffle]', 'lzf', 'gzip[:level]' "
                     "or None")


def round_mantissa(a, keep_bits):
    """Round float32 ``a`` to ``keep_bits`` explicit mantissa bits (round-half-to-even).

    The relative error is at most ``2**-(keep_bits+1)`` (1.2e-4 for 12 bits).  Zeros stay
    exactly zero -- which the ``arr`` schema relies on after freeze-out -- and so do the
    sign, infinities and NaNs.  ``keep_bits=None`` or >= 23 returns ``a`` unchanged (as
    float32).  The zeroed low bits are what lets bitshuffle compress; lossless data keeps
    them noisy.
    """
    a = np.asarray(a, dtype=np.float32)
    if keep_bits is None or int(keep_bits) >= _FLOAT32_MANTISSA:
        return a
    k = int(keep_bits)
    if k < 1:
        raise ValueError(f"keep_bits must be >= 1, got {keep_bits}")
    drop = _FLOAT32_MANTISSA - k
    bits = np.ascontiguousarray(a).view(np.uint32)
    lsb = (bits >> np.uint32(drop)) & np.uint32(1)             # ties go to the even value
    rounded = (bits + (np.uint32((1 << (drop - 1)) - 1) + lsb)) & np.uint32(
        (0xFFFFFFFF << drop) & 0xFFFFFFFF)
    out = np.where(np.isfinite(a), rounded.view(np.float32), a)
    return out.astype(np.float32, copy=False)


def tag_dataset(ds, label, keep_bits=None):
    """Record the filter and the rounding on ``ds`` itself, for provenance and readers.

    ``compression`` is the :func:`h5_filter_kwargs` label; ``keep_mantissa_bits`` is
    written only for lossy (rounded) data, so its absence means bit-exact.
    """
    ds.attrs["compression"] = label
    if keep_bits is not None and int(keep_bits) < _FLOAT32_MANTISSA:
        ds.attrs["keep_mantissa_bits"] = int(keep_bits)
        ds.attrs["max_rel_error"] = float(2.0 ** -(int(keep_bits) + 1))


def dataset_keep_bits(ds):
    """The ``keep_bits`` a dataset was written with (None = lossless)."""
    v = ds.attrs.get("keep_mantissa_bits")
    return None if v is None else int(v)
