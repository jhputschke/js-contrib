"""Equation-of-state resolution for fast_data.

One config block picks ONE equation of state, and `resolve_eos` returns it twice: as the numpy
object the MC-Glauber initial state needs (entropy -> energy conversion, temperature diagnostics)
and as the torch object the FV solver needs.  The torch one is always built from the numpy one via
`glauber.to_fv_eos`, so the two cannot drift apart -- a drift would silently change the initial
temperature relative to the evolution.

Kinds
-----
    conformal      ideal massless gas, p = e/3, e = (pi^2/30) dof T^4 / (hbar c)^3
    hotqcd         MUSIC's lattice table, EOS id 9   (hrg_hotqcd_eos_binary.dat)
    hotqcd_smash   MUSIC's lattice table, EOS id 91  (hrg_hotqcd_eos_SMASH_binary.dat)
    music_check    MUSIC's own check_EoS_<id>_PST.dat text dump

The `dof` default is 47.5 = 2*8 + (7/8)*4*3*2.5, the Nf = 2.5 Stefan-Boltzmann counting that the
hotQCD table asymptotes to at high T.  Switching `conformal -> hotqcd` therefore changes the data
as little as possible.  MUSIC's own ideal-gas EOS 0 uses 42.25 instead; set `dof: 42.25` for that.
"""

from __future__ import annotations

import os

import numpy as np

from . import glauber
from .glauber import IdealGasEoS, TableEoS, _MUSIC_FILES

__all__ = ["resolve_eos", "download_hotqcd", "write_eos_group", "read_eos_group",
           "eos_descriptor", "DEFAULT_DOF"]

DEFAULT_DOF = 47.5

#: MUSIC fetches these from the same Bitbucket repo; see EOS/download_hotQCD.sh.
_BITBUCKET = ("https://api.bitbucket.org/2.0/repositories/"
              "wayne_state_nuclear_theory/hotqcd/src/main/{fname}")

_KIND_TO_ID = {"hotqcd": 9, "hotqcd_smash": 91}
_RECORD_BYTES = 32          # 4 x float64 per row: e, p, s, T


def download_hotqcd(dest_dir, filetype="binary", timeout=300, force=False):
    """Fetch MUSIC's hotQCD table into `dest_dir` and return the file path.

    Deliberately plain urllib rather than a curl subprocess: it works the same on every platform,
    and it can be monkeypatched in tests.  Writes to a .part file and renames only after the size
    validates, so an interrupted download cannot leave a half table that silently loads.
    """
    import urllib.request

    fname = f"hrg_hotqcd_eos_{filetype}.dat"
    if fname not in _MUSIC_FILES.values():
        raise ValueError(f"unknown hotQCD filetype '{filetype}'; "
                         f"expected one of {sorted(f.split('_eos_')[1][:-4] for f in _MUSIC_FILES.values())}")
    os.makedirs(dest_dir, exist_ok=True)
    out = os.path.join(dest_dir, fname)
    if os.path.exists(out) and not force:
        return out

    url, part = _BITBUCKET.format(fname=fname), out + ".part"
    with urllib.request.urlopen(url, timeout=timeout) as r, open(part, "wb") as fh:
        while chunk := r.read(1 << 20):
            fh.write(chunk)
    size = os.path.getsize(part)
    if size == 0 or size % _RECORD_BYTES:
        os.remove(part)
        raise RuntimeError(f"downloaded {fname} is {size} B, not a whole number of "
                           f"{_RECORD_BYTES} B records -- the fetch was truncated or is not the table")
    os.replace(part, out)
    return out


def _resolve_table_path(kind, path, download):
    """The .dat file for `kind`, downloading it when allowed.  Raises with the exact remedy."""
    fname = _MUSIC_FILES[_KIND_TO_ID[kind]]
    if path is None:
        path = os.path.join("eos", "hotQCD")
    cand = os.path.join(path, fname) if os.path.isdir(path) else path
    if os.path.exists(cand):
        return cand
    if download:
        return download_hotqcd(path if os.path.isdir(path) or not path.endswith(".dat")
                               else os.path.dirname(path),
                               filetype=fname.split("_eos_")[1][:-4])
    raise FileNotFoundError(
        f"hotQCD table not found: {cand}\n"
        f"Fetch it with one of:\n"
        f"  python -c \"from fast_data.eos import download_hotqcd; download_hotqcd('{path}')\"\n"
        f"  bash loc_libs/fast_data/download_hotQCD.sh {fname.split('_eos_')[1][:-4]} {path}\n"
        f"or set eos.download: true in the config.")


def resolve_eos(cfg, device=None, dtype=None, log=None):
    """-> (np_eos, fv_eos) for the `eos:` config block.

    `np_eos` drives the Glauber initial state; `fv_eos` is the same EoS as a solver object, with
    its tables materialised on `device`/`dtype` (required on MPS, which has no float64).
    """
    from . import fv as fv_mod          # deferred: importing fv pulls in torch

    kind = str(cfg.get("kind", "conformal")).lower()
    if kind in ("conformal", "ideal", "0"):
        np_eos = IdealGasEoS(dof=float(cfg.get("dof", DEFAULT_DOF)))
    elif kind in _KIND_TO_ID:
        src = _resolve_table_path(kind, cfg.get("path"), bool(cfg.get("download", False)))
        np_eos = TableEoS.from_music_binary(src, n_points=int(cfg.get("n_points", 1500)))
        np_eos.source_path = os.path.abspath(src)
    elif kind == "music_check":
        src = cfg.get("path")
        if not src or not os.path.exists(src):
            raise FileNotFoundError(f"eos.kind=music_check needs eos.path to a check_EoS_*_PST.dat file (got {src!r})")
        np_eos = TableEoS.from_music_check(src, n_points=int(cfg.get("n_points", 1500)))
        np_eos.source_path = os.path.abspath(src)
    else:
        raise ValueError(f"unknown eos.kind '{kind}'; expected conformal | hotqcd | hotqcd_smash | music_check")

    fv_eos = glauber.to_fv_eos(np_eos, fv_mod, device=device, dtype=dtype)
    if log is not None:
        log(f"eos: {eos_descriptor(np_eos)}")
    return np_eos, fv_eos


def eos_descriptor(np_eos):
    """A one-line human description, with a sample point so a run log pins the calibration down."""
    e1 = np.asarray(1.0)
    if isinstance(np_eos, IdealGasEoS):
        return f"conformal dof={np_eos.dof} (e=1 GeV/fm^3 -> T={float(np_eos.T(e1)):.4f} GeV)"
    return (f"table '{np_eos.name}' (e=1 GeV/fm^3 -> T={float(np_eos.T(e1)):.4f} GeV, "
            f"p/e={float(np_eos.p(e1)):.4f})")


def write_eos_group(h5file, np_eos):
    """Write an `eos/` group byte-compatible with glauber.save_events.

    Consequence: glauber.load_eos() works unchanged on our output files, so a consumer rebuilds
    the exact EoS (TableEoS.from_tables does not resample) without needing the original .dat.
    """
    g = h5file.create_group("eos")
    g.attrs["name"] = np_eos.name
    g.attrs["music_eos_id"] = glauber._eos_id(np_eos)
    if isinstance(np_eos, IdealGasEoS):
        g.attrs["kind"], g.attrs["dof"] = "ideal", np_eos.dof
    else:
        g.attrs["kind"], g.attrs["n_ext"] = "table", np_eos.n_ext
        g.attrs["e_raw_range"] = np.asarray(np_eos.e_raw_range)
        for k in ("e_tab", "p_tab", "T_tab"):
            g.create_dataset(k, data=getattr(np_eos, k))
        if getattr(np_eos, "source_path", None):
            g.attrs["source_path"] = np_eos.source_path
    return g


def read_eos_group(path):
    """The EoS stored in a fast_data (or stage-1 Glauber) file."""
    return glauber.load_eos(path)
