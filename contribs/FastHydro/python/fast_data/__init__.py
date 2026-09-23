"""fast_data — MC-Glauber initial state + structure-preserving Milne hydro -> FNO4d HDF5.

A standalone, pure-Python replacement for the X-SCAPE/MUSIC route when what you need is a lot of
3+1D hydro evolutions in the FNO4d training format, optionally carrying jet energy-momentum
deposition as a hydro source term.  No C++ build and no X-SCAPE runtime are involved.

    workflow_fastdata/generate.py --config config.yaml

Modules
-------
    fv          the finite-volume 3+1D Milne solver (torch)
    glauber     tilted 3D MC-Glauber initial conditions + the MUSIC EoS readers (numpy)
    gubser      semi-analytic viscous Gubser flow, an independent check of the writer (numpy)
    eos         which EoS a config asks for, and how it is fetched and recorded
    config      YAML schema, defaults, validation, dotted overrides
    partons     YAML parton specs -> droplets
    liquefier   the pure-Python port of X-SCAPE's CausalLiquefier jet source
    evolve      the memory-streaming stepping loop
    convert     conserved state -> the (e, vx, vy, vz) training channels
    writer      the FNO4d `arr` HDF5 schema
    viz         lazy browsing of an output file; EventBrowser, and DiffBrowser for
                jet-minus-no-jet wakes (matplotlib)

`fv`, `evolve`, `convert`, `source`, `liquefier` and `gubser` are imported lazily, because
`loc_libs` does not depend on torch and has pure-numpy consumers -- `import fast_data` must stay
cheap.  See PLAN_fast_data.md and README_FastData.md.

Provenance: promoted from addition_drafts/FastData/, which stays as the frozen record of what was
originally validated.  See README_FastData.md for what changed.
"""

from __future__ import annotations

import importlib

from .config import DEFAULTS, apply_overrides, load_config, validate_config
from .eos import download_hotqcd, eos_descriptor, read_eos_group, resolve_eos, write_eos_group
from .writer import FnoH5Writer, write_fno_h5

_LAZY = {
    "fv": ".fv",
    "glauber": ".glauber",
    "gubser": ".gubser",
    "viz": ".viz",
    "evolve": ".evolve",
    "convert": ".convert",
    "partons": ".partons",
    "liquefier": ".liquefier",
}


def __getattr__(name):                       # PEP 562: keep torch off the import path
    if name in _LAZY:
        mod = importlib.import_module(_LAZY[name], __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(_LAZY))


__all__ = [
    "DEFAULTS", "apply_overrides", "load_config", "validate_config",
    "resolve_eos", "download_hotqcd", "write_eos_group", "read_eos_group", "eos_descriptor",
    "FnoH5Writer", "write_fno_h5",
    *_LAZY,
]
