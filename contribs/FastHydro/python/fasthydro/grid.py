"""Grid geometry: the one place fast_data's axes and X-SCAPE's axes are reconciled.

Axis conventions, and why they are what they are
------------------------------------------------
fast_data's `fv.Grid` is cell-centred and symmetric about zero:

    coord(i) = (i - (n-1)/2) * d          (fv.py Grid.__post_init__)

X-SCAPE's `EvolutionHistory` imposes *no* convention of its own -- it is simply

    EtaCoord(i) = eta_min + i * deta      (FluidEvolutionHistory.h:390)

so FastHydro keeps fast_data's symmetric axis end to end and declares it through
``eta_min = -(neta-1)/2 * deta``.  MUSIC's ``eta_i = i*deta - neta*deta/2`` is MUSIC's own,
reached only through ``Initial_profile 42``; it differs by half a cell.  That form appears
here in exactly one place -- :meth:`GridSpec.is_ranges`, used only so that
``InitialState::GetXSize()`` recovers ``nx`` -- so the conversion has a single home.
See `config/FVvsMUSIC/run_music_leg.py:325-330` for the same offset documented on the
MUSIC side.

`InitialState::GetXSize()` is ``ceil(2*grid_max/grid_step)``, so ``grid_max = n*d/2`` is what
makes the framework agree with an ``n``-point array.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class GridSpec:
    """Spatial + temporal grid shared by the initial state, the solver and `bulk_info`."""

    nx: int
    ny: int
    neta: int
    dx: float
    dy: float
    deta: float
    tau0: float
    record_dtau: float
    ntau: int

    # ---------------------------------------------------------------- fast_data cell centres
    @property
    def x_min(self) -> float:
        return -0.5 * (self.nx - 1) * self.dx

    @property
    def y_min(self) -> float:
        return -0.5 * (self.ny - 1) * self.dy

    @property
    def eta_min(self) -> float:
        return -0.5 * (self.neta - 1) * self.deta

    @property
    def tau_max(self) -> float:
        return self.tau0 + (self.ntau - 1) * self.record_dtau

    @property
    def n_cells(self) -> int:
        return self.nx * self.ny * self.neta * self.ntau

    # ---------------------------------------------------------------- framework hand-over
    def is_ranges(self) -> tuple[float, float, float]:
        """``SetRanges`` arguments so that ``GetXSize()/GetYSize()/GetZSize()`` give n.

        This is the *only* place the ``n*d/2`` half-cell form appears.
        """
        return (self.nx * self.dx / 2.0, self.ny * self.dy / 2.0, self.neta * self.deta / 2.0)

    def is_steps(self) -> tuple[float, float, float]:
        return (self.dx, self.dy, self.deta)

    def expected_is_sizes(self) -> tuple[int, int, int]:
        """What ``GetXSize()`` etc. will return for :meth:`is_ranges` -- assert against this."""
        rmax, steps = self.is_ranges(), self.is_steps()
        return tuple(int(math.ceil(2.0 * r / s)) for r, s in zip(rmax, steps))  # type: ignore[return-value]

    def tau_grid(self):
        import numpy as np

        return self.tau0 + self.record_dtau * np.arange(self.ntau, dtype=np.float64)

    def to_fv_grid(self, device=None, dtype=None):
        import torch

        from fast_data import fv

        return fv.Grid(self.nx, self.ny, self.neta, self.dx, self.dy, self.deta,
                       device=torch.device(device or "cpu"),
                       dtype=dtype or torch.float64)

    # ---------------------------------------------------------------- construction
    @classmethod
    def from_cfg(cls, cfg) -> "GridSpec":
        """Build from a resolved `fast_data` config dict."""
        g, t = cfg["grid"], cfg["time"]
        tau0 = float(t["tau0"])
        record_dtau = float(t["record_dtau"])
        ntau = int(t.get("choose_ntau") or 0)
        if ntau <= 0:
            tau_end = float(t["tau_end"])
            ntau = int(round((tau_end - tau0) / record_dtau)) + 1
        return cls(int(g["nx"]), int(g["ny"]), int(g["neta"]),
                   float(g["dx"]), float(g["dy"]), float(g["deta"]),
                   tau0, record_dtau, ntau)

    def describe(self) -> str:
        return (f"{self.nx}x{self.ny}x{self.neta} cells "
                f"(dx={self.dx:g} dy={self.dy:g} deta={self.deta:g} fm), "
                f"x in [{self.x_min:g}, {-self.x_min:g}], eta in [{self.eta_min:g}, {-self.eta_min:g}]; "
                f"tau {self.tau0:g}..{self.tau_max:g} fm/c in {self.ntau} frames "
                f"(dtau={self.record_dtau:g})")
