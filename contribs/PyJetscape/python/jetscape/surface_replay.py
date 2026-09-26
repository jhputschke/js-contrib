"""
python/jetscape/surface_replay.py

SurfaceReplay -- a FluidDynamics that hands a stored freeze-out surface to the framework,
so iSS samples it exactly as it would have sampled the live hydro's.

    replay = SurfaceReplay()
    iss = create_module("iSS")
    jetscape = JetScapePerEvent(); jetscape.Add(replay); jetscape.Add(iss); jetscape.Init()
    jetscape.ExecInit()
    for cells in surfaces:                       # (N, 32) float32, SURFACE_CELL_COLUMNS
        replay.load(cells)
        jetscape.ExecPerEvent()                  # iSS: GetHydroHyperSurface -> these cells
        hadrons = soft_hadrons_numpy(iss)
        jetscape.ClearPerEvent()

There is no initial state, pre-equilibrium or evolution: ``Init`` is overridden so
``FluidDynamics::Init`` does not insist on them, and ``bulk_info`` stays empty.  The cells
go in through ``FluidDynamics::StoreSurfaceCell`` -- the same call MUSIC's wrapper makes --
so iSS takes its normal ``GetHydroHyperSurface`` path.
"""

from __future__ import annotations

import numpy as np

from .particlize_h5 import SURFACE_COLUMNS

try:
    from .pyjetscape_core import FluidDynamics
except ImportError:                                   # pragma: no cover - no core build
    FluidDynamics = object

__all__ = ["SurfaceReplay"]


class SurfaceReplay(FluidDynamics):
    """Replay one stored surface per event (call :meth:`load` before each event)."""

    def __init__(self, module_id="SurfaceReplay"):
        super().__init__()
        self.SetId(module_id)
        self._next = None
        self.n_cells = 0

    def load(self, cells):
        """The surface for the next event: (N, 32) float32, columns SURFACE_CELL_COLUMNS."""
        a = np.ascontiguousarray(cells, dtype=np.float32)
        if a.ndim != 2 or a.shape[1] != len(SURFACE_COLUMNS):
            raise ValueError(f"SurfaceReplay.load: need (N, {len(SURFACE_COLUMNS)}), got "
                             f"{a.shape}")
        self._next = a

    # ── framework hooks ─────────────────────────────────────────────────────────
    def Init(self):
        # Deliberately not FluidDynamics::Init: that exits without an initial state
        # and (for an unknown module id) a pre-equilibrium module.  Nothing else of it
        # is needed to hand over a surface.
        pass

    def InitializeHydro(self, params):
        pass

    def EvolveHydro(self):
        self.clearSurfaceCellVector()
        if self._next is None:
            raise RuntimeError("SurfaceReplay: no surface loaded for this event "
                               "(call load(cells) before ExecPerEvent)")
        self.store_surface_from_numpy(self._next)
        self.n_cells = len(self._next)
        self._next = None
        self.set_hydro_status_finished()
