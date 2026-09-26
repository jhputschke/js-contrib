"""
python/jetscape/__init__.py

Convenience re-exports so user code can write:

    import pyjetscape
    js = pyjetscape.JetScape()

instead of:

    from python.jetscape import pyjetscape_core as pyjetscape
"""

# NOTE: do not reorder these three imports.  `fno_hydro` must be imported before the bulk
# `pyjetscape_core` import or the process aborts with "OMP: Error #15: Initializing
# libomp.dylib, but found libomp.dylib already initialized" -- ROOT (loaded by the
# extension) and the scientific stack each ship a libomp.  Verified: moving `.utils` or
# `.fno_hydro` after the core import fails deterministically.
try:
    from .fno_hydro import fno_config_from_xml  # noqa: F401
    from .utils import shower_to_networkx  # noqa: F401

    from .pyjetscape_core import (  # noqa: F401
    # Framework
        JetScapeTask,
        JetScapeModuleBase,
        JetScape,
        JetScapePerEvent,
        create_module,
        # Signal manager — global access to all registered pipeline modules
        JetScapeSignalManager,
        # Evolution history
        FluidCellInfo,
        SurfaceCellInfo,
        EvolutionHistory,
        # Physics modules — base classes
        InitialState,
        PreequilibriumDynamics,
        FluidDynamics,
        # Physics modules — concrete C++ implementations
        MpiMusic,
        TrentoInitial,
        # Hydro status enum and Parameter struct
        HydroStatus,
        Parameter,
        # True when built against a ROOT-enabled X-SCAPE
        HAS_ROOT,
    )

    HAS_CORE = True
except ImportError:  # pragma: no cover - reader-only / training-side install
    HAS_CORE = False
    HAS_ROOT = False

# C++ ROOT writers — only compiled into X-SCAPE with USE_ROOT
if HAS_CORE and HAS_ROOT:
    from .pyjetscape_core import FastRootBulkWriter  # noqa: F401


# ── HDF5 tooling — h5py and numpy only, no X-SCAPE build required ────────────
# This is why the block above is survivable: these are used on training machines that
# have no extension.  Only H5BulkWriter (a framework module) needs it, and it raises a
# clear ImportError when constructed without one.
try:
    import h5py as _h5py  # noqa: F401

    HAS_H5PY = True
except ImportError:  # pragma: no cover
    HAS_H5PY = False

if HAS_H5PY:
    # registers the hdf5plugin filters (Blosc, the default compression), so every reader
    # that imports jetscape opens the files; README_h5_optim.md
    from . import h5_compression  # noqa: F401
    from .h5_compression import HAVE_HDF5PLUGIN, round_mantissa  # noqa: F401
    from .fno_h5_writer import FnoH5Writer, grid_attrs, repad_to  # noqa: F401
    from .fast_h5_bulk import H5BulkWriter, read_fast_h5_bulk  # noqa: F401
    from .pair_h5 import PairH5Writer  # noqa: F401
    # hadron level: stored surfaces + final partons, and hadron files (PLAN_particlize_h5.md)
    from .particlize_h5 import ParticlizeFile, ParticlizeH5Writer  # noqa: F401
    from .hadrons_h5 import HadronFile, HadronH5Writer, Hadrons, JetEvents  # noqa: F401


#: names that exist only with the compiled extension, for the message below
_CORE_NAMES = frozenset({
    "JetScapeTask", "JetScapeModuleBase", "JetScape", "JetScapePerEvent", "create_module",
    "JetScapeSignalManager", "FluidCellInfo", "SurfaceCellInfo", "EvolutionHistory",
    "InitialState", "PreequilibriumDynamics", "FluidDynamics", "MpiMusic", "TrentoInitial",
    "HydroStatus", "Parameter", "FastRootBulkWriter", "fno_config_from_xml",
    "shower_to_networkx",
})


def __getattr__(name):
    """Turn a missing extension into a clear message, not a bare AttributeError."""
    if name in _CORE_NAMES:
        raise ImportError(
            f"jetscape.{name} needs the compiled pyjetscape_core extension, which is not "
            "importable here (no X-SCAPE build in this environment). The HDF5 tooling -- "
            "FnoH5Writer, grid_attrs, repad_to, read_fast_h5_bulk, h5_compression -- works "
            "without it.")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
