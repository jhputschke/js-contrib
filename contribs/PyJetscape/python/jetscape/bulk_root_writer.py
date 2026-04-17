"""
python/jetscape/bulk_root_writer.py

PyBulkRootWriter — Python mirror of fno_hydro/root_bulk/bulkRootWriter.cc.

The class inherits from pyjetscape.JetScapeModuleBase (via the
PyJetScapeModuleBase C++ trampoline) and can be added to a JETSCAPE pipeline
with JetScape.Add().  Its Exec() method is called automatically once per
event, exactly like the C++ version.

The bulk info is obtained through JetScapeSignalManager.Instance()
.GetHydroPointer(), which mirrors the C++ BulkRootWriter::Exec() call:

    auto hydro = JetScapeSignalManager::Instance()->GetHydroPointer();

Output format
-------------
ROOT TTree written by uproot (uproot >= 4 required).  The tree contains
one entry per event with branches:

    user_res  : float32 array of shape (m_nX, m_nY, m_nT, n_features)
                Features: [energy_density [GeV/fm^3], vx, vy]
    nx        : int32   — number of x grid points
    ny        : int32   — number of y grid points
    ntau      : int32   — number of tau grid points
    nfeatures : int32   — number of feature channels
    tau0      : float32 — starting proper time [fm/c]
    dtau      : float32 — tau step [fm/c]
    x_min     : float32 — minimum x [fm]
    dx        : float32 — x (and y) step [fm]

If uproot is not installed the writer falls back to an .npz archive with the
same data stored as individual arrays per event.

Pipeline placement
------------------
    jetscape.Add(trento)
    jetscape.Add(null_predynamics)
    jetscape.Add(hydro)          # must come BEFORE PyBulkRootWriter
    jetscape.Add(bulk_writer)    # ← here

Usage
-----
    from python.jetscape.bulk_root_writer import PyBulkRootWriter

    bulk_writer = PyBulkRootWriter(
        output_name = "bulk_output.root",
        n_features  = 3,       # energy_density, vx, vy
        tau_max     = 5.0,     # [fm/c]
        d_tau       = 0.1,     # [fm/c]
        d_x         = 0.5,     # [fm]
    )
    jetscape.Add(bulk_writer)
    jetscape.Init()
    jetscape.Exec()
    jetscape.Finish()
    bulk_writer.Finish()   # must be called explicitly — JetScape::Finish()
                           # calls FinishTasks() which is a no-op; sub-task
                           # Finish() is never propagated by the framework.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np

# ── Optional dependencies ────────────────────────────────────────────────────
try:
    import uproot
    _UPROOT_AVAILABLE = True
except ImportError:
    _UPROOT_AVAILABLE = False

try:
    import awkward as ak
    _AWKWARD_AVAILABLE = True
except ImportError:
    _AWKWARD_AVAILABLE = False

from .pyjetscape_core import JetScapeModuleBase, JetScapeSignalManager


class PyBulkRootWriter(JetScapeModuleBase):
    """
    Python equivalent of C++ BulkRootWriter.

    Hooks into the JETSCAPE pipeline via JetScapeModuleBase.  On each call
    to Exec() it reads the hydro bulk info through JetScapeSignalManager,
    interpolates it onto a uniform user-defined grid, and accumulates the
    result.  On Finish() the accumulated data are written to a ROOT file
    (via uproot) or, if uproot is unavailable, to a .npz archive.

    Parameters
    ----------
    output_name : str
        Output file path.  Use a ``.root`` extension for ROOT output or any
        other extension for .npz fallback output.
    n_features : int
        Number of feature channels to store per cell.
        3 (default) → [energy_density, vx, vy]
        4            → [energy_density, temperature, vx, vy]
    tau_max : float
        Maximum proper time to store [fm/c].  Default: 5.0.
    d_tau : float
        Proper-time step of the output grid [fm/c].  Default: 0.1.
    d_x : float
        Transverse spatial step of the output grid [fm].  Default: 0.5.
    verbose : bool
        Print per-event timing information.  Default: False.
    """

    def __init__(
        self,
        output_name: str = "bulk_root_writer_output.root",
        n_features: int = 3,
        tau_max: float = 5.0,
        d_tau: float = 0.1,
        d_x: float = 0.5,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.SetId("PyBulkRootWriter")

        self._output_name = output_name
        self._n_features  = n_features
        self._tau_max     = tau_max
        self._d_tau       = d_tau
        self._d_x         = d_x
        self._verbose     = verbose

        # Populated on first Exec()
        self._init_branch: bool = False
        self._m_nX: int = 0
        self._m_nY: int = 0
        self._m_nT: int = 0
        self._tau0:  float = 0.0
        self._x_min: float = 0.0
        self._dx_hydro: float = 0.0

        # Open ROOT file handle (set on first Exec() once grid dims are known)
        self._root_file = None
        self._event_count: int = 0

        # Per-event data accumulator used only for the .npz fallback path
        self._events: list[np.ndarray] = []

    # ── JETSCAPE interface ─────────────────────────────────────────────────────

    def Init(self) -> None:
        """Reset per-run state.  Called once by JetScape::Init()."""
        if self._root_file is not None:
            self._root_file.close()
            self._root_file = None
        self._events = []
        self._init_branch = False
        self._event_count = 0
        print(f"PyBulkRootWriter: output → {self._output_name}, "
              f"n_features = {self._n_features}")

    def Exec(self) -> None:
        """
        Read hydro bulk info for the current event and accumulate it.

        Mirrors BulkRootWriter::Exec() in bulkRootWriter.cc:
          1. Get hydro pointer via JetScapeSignalManager.
          2. Determine user-resolution grid dimensions.
          3. Interpolate bInfo.get(tau, x, y, 0) onto the grid.
          4. Append to the event list.
        """
        t_start = time.perf_counter()

        sm    = JetScapeSignalManager.Instance()
        hydro = sm.GetHydroPointer()
        if hydro is None:
            print("PyBulkRootWriter::Exec() — WARNING: no hydro pointer, "
                  "skipping event.")
            return

        b_info = hydro.get_bulk_info()

        tau0  = b_info.Tau0()
        x_min = b_info.XMin()
        x_max = b_info.XMax()   # same for y (symmetric grid)
        dx    = b_info.dx

        # User-resolution grid dimensions (mirrors bulkRootWriter.cc)
        m_nT = int(self._tau_max / self._d_tau)
        m_nX = int((x_max - x_min + dx) / self._d_x)
        m_nY = m_nX

        # Record grid info from first event (matches C++ !initBranch guard)
        if not self._init_branch:
            self._tau0     = tau0
            self._x_min    = x_min
            self._dx_hydro = dx
            self._m_nT     = m_nT
            self._m_nX     = m_nX
            self._m_nY     = m_nY
            self._init_branch = True
            if self._verbose:
                print(f"  InitBranch: tau0={tau0:.3f} fm/c, "
                      f"x ∈ [{x_min:.1f}, {x_max:.1f}] fm, "
                      f"grid = ({m_nX}, {m_nY}, {m_nT})")
            # Open the ROOT file and create the tree now that dimensions are known
            if _UPROOT_AVAILABLE and self._output_name.endswith(".root"):
                flat_size = m_nX * m_nY * m_nT * self._n_features
                self._root_file = uproot.recreate(self._output_name)
                self._root_file.mktree("t", {
                    "user_res":  np.dtype(("float32", (flat_size,))),
                    "nx":        "int32",
                    "ny":        "int32",
                    "ntau":      "int32",
                    "nfeatures": "int32",
                    "tau0":      "float32",
                    "dtau":      "float32",
                    "x_min":     "float32",
                    "dx":        "float32",
                })

        # ── Build user-resolution array [x][y][tau][features] ─────────────────
        data = np.zeros((m_nX, m_nY, m_nT, self._n_features), dtype=np.float32)

        for k in range(m_nT):
            tau_in = tau0 + k * self._d_tau
            for i in range(m_nX):
                x_in = x_min + i * self._d_x
                for j in range(m_nY):
                    y_in = x_min + j * self._d_x
                    cell = b_info.get(tau_in, x_in, y_in, 0.0)
                    data[i, j, k, 0] = cell.energy_density
                    if self._n_features == 3:
                        data[i, j, k, 1] = cell.vx
                        data[i, j, k, 2] = cell.vy
                    elif self._n_features >= 4:
                        data[i, j, k, 1] = cell.temperature
                        data[i, j, k, 2] = cell.vx
                        data[i, j, k, 3] = cell.vy

        if self._root_file is not None:
            # Write this event directly into the open tree
            self._root_file["t"].extend({
                "user_res":  data.reshape(1, -1),
                "nx":        np.array([m_nX],           dtype=np.int32),
                "ny":        np.array([m_nY],           dtype=np.int32),
                "ntau":      np.array([m_nT],           dtype=np.int32),
                "nfeatures": np.array([self._n_features], dtype=np.int32),
                "tau0":      np.array([self._tau0],     dtype=np.float32),
                "dtau":      np.array([self._d_tau],    dtype=np.float32),
                "x_min":     np.array([self._x_min],   dtype=np.float32),
                "dx":        np.array([self._d_x],     dtype=np.float32),
            })
            self._event_count += 1
        else:
            # npz fallback: accumulate and write at Finish()
            self._events.append(data)

        if self._verbose:
            elapsed = time.perf_counter() - t_start
            count = self._event_count if self._root_file is not None else len(self._events)
            print(f"  PyBulkRootWriter: event {count} "
                  f"extracted in {elapsed:.2f} s")

    def Clear(self) -> None:
        """Called between events — no per-event cleanup needed here."""
        pass

    def Finish(self) -> None:
        """
        Finalise the output file.

        For ROOT output the tree has already been filled per event; this method
        just closes the file handle.  For the .npz fallback the accumulated
        events are written here.

        Called automatically by JetScape::Finish(), or manually by the user
        after jetscape.Exec() completes.
        """
        if self._root_file is not None:
            self._root_file.close()
            self._root_file = None
            if self._event_count == 0:
                print("PyBulkRootWriter::Finish() — no events were written.")
            else:
                print(f"PyBulkRootWriter: wrote {self._event_count} event(s) to "
                      f"{self._output_name}")
                print(f"  user_res shape per event: "
                      f"({self._m_nX}, {self._m_nY}, {self._m_nT}, {self._n_features})")
            return

        # ── npz fallback ───────────────────────────────────────────────────────
        if not self._events:
            print("PyBulkRootWriter::Finish() — no events to write.")
            return

        if not _UPROOT_AVAILABLE and self._output_name.endswith(".root"):
            fallback = os.path.splitext(self._output_name)[0] + ".npz"
            print(f"PyBulkRootWriter: uproot not available, "
                  f"falling back to {fallback}")
            self._output_name = fallback
        self._write_npz()

    # ── Output helpers ─────────────────────────────────────────────────────────


    def _write_npz(self) -> None:
        """
        Fallback: write events to a NumPy .npz archive.

        Arrays saved:
            user_res  — shape (n_events, m_nX, m_nY, m_nT, n_features)
            grid_info — 1D float32 array [tau0, dtau, x_min, dx, nx, ny, ntau, nf]
        """
        stacked = np.stack(self._events, axis=0)
        m_nX, m_nY, m_nT, nF = self._events[0].shape

        grid_info = np.array(
            [self._tau0, self._d_tau, self._x_min, self._d_x,
             m_nX, m_nY, m_nT, nF],
            dtype=np.float32,
        )

        np.savez_compressed(
            self._output_name,
            user_res=stacked,
            grid_info=grid_info,
        )
        print(f"PyBulkRootWriter: wrote {len(self._events)} event(s) to "
              f"{self._output_name}.npz")
        print(f"  user_res shape: {stacked.shape}")

    # ── Convenience: read back a written ROOT file ─────────────────────────────

    @staticmethod
    def read_root(path: str) -> dict:
        """
        Load a ROOT file written by PyBulkRootWriter.

        Parameters
        ----------
        path : str
            Path to the .root file.

        Returns
        -------
        dict with keys:
            "user_res"  — np.ndarray of shape (n_events, nx, ny, ntau, nfeatures)
            "grid_info" — dict with tau0, dtau, x_min, dx
            "nx", "ny", "ntau", "nfeatures"

        Requires uproot.
        """
        if not _UPROOT_AVAILABLE:
            raise ImportError("uproot is required to read ROOT files.  "
                              "Install with: pip install uproot")

        with uproot.open(path) as f:
            tree = f["t"]
            nx        = int(tree["nx"].array(library="np")[0])
            ny        = int(tree["ny"].array(library="np")[0])
            ntau      = int(tree["ntau"].array(library="np")[0])
            nfeatures = int(tree["nfeatures"].array(library="np")[0])
            tau0      = float(tree["tau0"].array(library="np")[0])
            dtau      = float(tree["dtau"].array(library="np")[0])
            x_min     = float(tree["x_min"].array(library="np")[0])
            dx        = float(tree["dx"].array(library="np")[0])
            raw       = tree["user_res"].array(library="np")

        flat_size = nx * ny * ntau * nfeatures
        n_events  = raw.shape[0]
        user_res  = raw.reshape(n_events, nx, ny, ntau, nfeatures)

        return {
            "user_res":  user_res,
            "nx":        nx,
            "ny":        ny,
            "ntau":      ntau,
            "nfeatures": nfeatures,
            "grid_info": {
                "tau0":  tau0,
                "dtau":  dtau,
                "x_min": x_min,
                "dx":    dx,
            },
        }
