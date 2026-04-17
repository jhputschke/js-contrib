"""
python/jetscape/__init__.py

Convenience re-exports so user code can write:

    import pyjetscape
    js = pyjetscape.JetScape()

instead of:

    from python.jetscape import pyjetscape_core as pyjetscape
"""

from .fno_hydro import fno_config_from_xml  # noqa: F401

from .pyjetscape_core import (  # noqa: F401
    # Framework
    JetScapeTask,
    JetScapeModuleBase,
    JetScape,
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
)
