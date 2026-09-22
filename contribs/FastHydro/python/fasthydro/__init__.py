"""fasthydro -- JETSCAPE adapters for the vendored `fast_data` MC-Glauber + FV hydro.

This package is the *only* place FastHydro-specific code lives.  `fast_data` next to it is a
verbatim vendored copy and is never patched (see ../../VENDORING.md).

    FastGlauberInitialState   InitialState   -> tilted 3D MC-Glauber
    FastHydro                 FluidDynamics  -> the structure-preserving Milne FV solver
    DropletBridge             module         -> C++ CausalLiquefier droplets -> the solver's source
    build_two_stage           pipeline       -> background leg, Matter+LBT, jet leg

Only `grid` and the pure-numpy helpers import at module load; everything that needs torch or
`pyjetscape_core` is imported lazily, so `import fasthydro` stays cheap and works on a machine
with no X-SCAPE build (for replay and analysis).
"""

from __future__ import annotations

import importlib
import os
import pathlib

__all__ = [
    "GridSpec",
    "FastGlauberInitialState",
    "FastFileInitialState",
    "FastHydro",
    "DropletBridge",
    "build_two_stage",
    "check_vendored_fast_data",
]


def check_vendored_fast_data(strict: bool = True) -> str:
    """Verify the importable `fast_data` is the copy vendored beside this package.

    `fast_data` is a top-level package name that also exists in FNO4d.  If both are on
    ``sys.path`` the first one wins *silently*, which would mean running a different solver
    than the one this contribution pins and tests.  Set
    ``FASTHYDRO_ALLOW_EXTERNAL_FAST_DATA=1`` to allow it deliberately.

    Returns the resolved path of the imported package.
    """
    import fast_data

    got = pathlib.Path(fast_data.__file__).resolve().parent
    want = (pathlib.Path(__file__).resolve().parent.parent / "fast_data").resolve()
    if got != want and strict and not os.environ.get("FASTHYDRO_ALLOW_EXTERNAL_FAST_DATA"):
        raise ImportError(
            "fasthydro imported a foreign `fast_data`.\n"
            f"  imported: {got}\n"
            f"  expected: {want}\n"
            "`fast_data` is a top-level name that also exists in FNO4d, so whichever comes "
            "first on sys.path wins silently. Put this contribution's python/ directory "
            "first, or set FASTHYDRO_ALLOW_EXTERNAL_FAST_DATA=1 to allow it on purpose."
        )
    return str(got)


_LAZY = {
    "GridSpec": ".grid",
    "FastGlauberInitialState": ".initial_state",
    "FastFileInitialState": ".initial_state",
    "FastHydro": ".hydro",
    "DropletBridge": ".liquefier_bridge",
    "build_two_stage": ".pipeline",
}


def __getattr__(name):              # PEP 562 -- keep torch and pyjetscape_core off the import path
    if name in _LAZY:
        mod = importlib.import_module(_LAZY[name], __name__)
        obj = getattr(mod, name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(_LAZY))
