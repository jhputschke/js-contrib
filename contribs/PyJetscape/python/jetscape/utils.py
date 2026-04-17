"""
python/jetscape/utils.py

Data conversion utilities between JETSCAPE C++ objects and numpy/torch tensors.

Functions
---------
bulk_info_to_numpy(bulk_info, n_features)
    Convert EvolutionHistory.data  →  np.ndarray  shape (ntau, nx, ny, n_features)

bulk_info_to_tensor(bulk_info, n_features, device='cpu')
    Convert EvolutionHistory.data  →  torch.Tensor  shape (1, n_features, nx, ny, ntau)

numpy_to_bulk_info(arr, hydro, n_features)
    Write  np.ndarray (n_features, nx, ny, ntau) back into hydro.bulk_info.data
    via hydro.store_fluid_cells_from_numpy().

tensor_to_bulk_info(tensor, hydro, n_features)
    Write  torch.Tensor (1, n_features, nx, ny, ntau) back into hydro.bulk_info.data.

preeq_to_numpy(preeq, ini)
    Pack all pre-equilibrium stress-energy fields into a single numpy array
    shaped (nx_preq * ny_preq, n_preeq_fields).

rebin_preeq_to_fno_grid(preeq_arr, preeq_grid, fno_grid)
    Nearest-neighbour rebinning from the pre-equilibrium grid to the FNO grid.
    Returns a numpy array shaped (n_features, nx_fno, ny_fno, 1).
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    # Avoid hard import of torch at module-load time; only needed at call time.
    import torch

# ─── EvolutionHistory ↔ numpy ────────────────────────────────────────────────

def bulk_info_to_numpy(bulk_info, n_features: int) -> np.ndarray:
    """
    Convert EvolutionHistory.data to a numpy array.

    Parameters
    ----------
    bulk_info : EvolutionHistory
        The C++ EvolutionHistory object (from FluidDynamics.get_bulk_info()).
    n_features : int
        Number of features per cell to extract.
        Feature mapping (same as FnoHydro C++ convention):
          0: energy_density
          1: temperature
          2: vx
          3: vy

    Returns
    -------
    np.ndarray, shape (ntau, nx, ny, n_features), dtype float32
    """
    return bulk_info.to_numpy(n_features)


def bulk_info_to_numpy_slow(bulk_info, n_features: int) -> np.ndarray:
    """
    Pure-Python fallback: convert EvolutionHistory.data to a numpy array.

    Identical output to bulk_info_to_numpy() but uses a Python-level triple
    loop with per-cell C++ boundary crossings — kept for benchmarking and
    as a reference implementation.

    Returns
    -------
    np.ndarray, shape (ntau, nx, ny, n_features), dtype float32
    """
    ntau = bulk_info.ntau
    nx   = bulk_info.nx
    ny   = bulk_info.ny
    n    = bulk_info.get_data_size()
    if n == 0:
        raise ValueError("bulk_info.data is empty — run EvolveHydro() first.")
    if n != ntau * nx * ny:
        raise ValueError(
            f"bulk_info size {n} does not match ntau*nx*ny = {ntau*nx*ny}. "
            "Make sure the grid metadata matches the stored data."
        )

    arr = np.zeros((ntau, nx, ny, n_features), dtype=np.float32)
    _GETTERS = [
        lambda c: c.energy_density,
        lambda c: c.temperature,
        lambda c: c.vx,
        lambda c: c.vy,
        lambda c: c.entropy_density,
        lambda c: c.pressure,
    ]
    getters = _GETTERS[:n_features]

    for k in range(ntau):
        for i in range(nx):
            for j in range(ny):
                cell = bulk_info.get_fluid_cell(k, i, j, 0)
                for f, g in enumerate(getters):
                    arr[k, i, j, f] = g(cell)
    return arr


def bulk_info_to_tensor(bulk_info, n_features: int,
                        device: str = "cpu") -> "torch.Tensor":
    """
    Convert EvolutionHistory.data to a PyTorch tensor.

    Returns
    -------
    torch.Tensor, shape (1, n_features, nx, ny, ntau)
        FNO convention: batch × features × spatial × time.
    """
    import torch
    arr = bulk_info_to_numpy(bulk_info, n_features)          # (ntau, nx, ny, nf)
    t   = torch.from_numpy(arr.transpose(3, 1, 2, 0))        # (nf, nx, ny, ntau)
    return t.unsqueeze(0).to(device)                          # (1, nf, nx, ny, ntau)


def numpy_to_bulk_info(arr: np.ndarray, hydro) -> None:
    """
    Write a numpy array back into hydro.bulk_info.data.

    Parameters
    ----------
    arr : np.ndarray, shape (n_features, nx, ny, ntau), dtype float32
        Feature layout: [energy_density, temperature, vx, vy, …]
        Values must already be in physical units (un-normalised).
    hydro : FluidDynamics (PyFluidDynamics)
        The hydro module whose bulk_info will be populated.
    """
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    hydro.store_fluid_cells_from_numpy(arr)


def tensor_to_bulk_info(tensor: "torch.Tensor", hydro) -> None:
    """
    Write a PyTorch tensor back into hydro.bulk_info.data.

    Parameters
    ----------
    tensor : torch.Tensor, shape (1, n_features, nx, ny, ntau)
        FNO output convention.  Values must be in physical units.
    hydro : FluidDynamics (PyFluidDynamics)
        The hydro module whose bulk_info will be populated.
    """
    arr = tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
    # arr shape: (n_features, nx, ny, ntau) — matches store_fluid_cells_from_numpy
    numpy_to_bulk_info(arr, hydro)


# ─── Pre-equilibrium helpers ──────────────────────────────────────────────────

_PREEQ_FIELD_NAMES = [
    "e", "P", "utau", "ux", "uy", "ueta",
    "pi00", "pi01", "pi02", "pi03",
    "pi11", "pi12", "pi13",
    "pi22", "pi23",
    "pi33",
    "bulk_Pi",
]


def preeq_to_numpy(preeq, field_names: list[str] | None = None) -> np.ndarray:
    """
    Pack pre-equilibrium fields into a 2D numpy array.

    Parameters
    ----------
    preeq : PreequilibriumDynamics
        The C++ pre-equilibrium module (from FluidDynamics.get_preeq_pointer()).
    field_names : list of str, optional
        Subset of fields to pack.  Defaults to all 17 fields in
        _PREEQ_FIELD_NAMES.  Each name must correspond to a
        get_{name}_numpy() method on PreequilibriumDynamics.

    Returns
    -------
    np.ndarray, shape (n_cells, n_fields), dtype float64
        n_cells = nx_preq * ny_preq (flattened).
    """
    if field_names is None:
        field_names = _PREEQ_FIELD_NAMES
    cols = [getattr(preeq, f"get_{name}_numpy")() for name in field_names]
    return np.column_stack(cols).astype(np.float64)


def rebin_preeq_to_fno_grid(
    preeq,
    ini,
    nx_fno: int,
    ny_fno: int,
    x_min_fno: float,
    y_min_fno: float,
    dx_fno: float,
    dy_fno: float,
    n_features: int = 4,
) -> np.ndarray:
    """
    Nearest-neighbour rebin from the pre-equilibrium grid to the FNO grid.

    Replicates the rebinning logic in FnoHydro::EvolveHydro() (C++).

    Parameters
    ----------
    preeq : PreequilibriumDynamics
    ini   : InitialState
    nx_fno, ny_fno : int
    x_min_fno, y_min_fno : float  (typically negative, e.g. -7.5 fm)
    dx_fno, dy_fno : float
    n_features : int
        3 → [ed, vx, vy]; 4 → [ed, T, vx, vy]
        For n_features == 3 the input energy density is NOT multiplied by tau
        here — that normalisation is applied by the Python EvolveHydro() just
        before feeding the tensor to the model.

    Returns
    -------
    np.ndarray, shape (n_features, nx_fno, ny_fno, 1), dtype float32
        Ready to be repeated along the last axis (ntau times) and fed to the
        FNO model as shape (n_features, nx_fno, ny_fno, ntau) or
        unsqueeze(0) → (1, n_features, nx_fno, ny_fno, ntau).
    """
    dx_preq  = ini.GetXStep()
    dy_preq  = ini.GetYStep()
    x_min_preq = -ini.GetXMax()
    y_min_preq = -ini.GetYMax()
    nx_preq  = ini.GetXSize()
    ny_preq  = ini.GetYSize()

    e_arr  = np.array(preeq.get_e_numpy(),  dtype=np.float32)
    ux_arr = np.array(preeq.get_ux_numpy(), dtype=np.float32)
    uy_arr = np.array(preeq.get_uy_numpy(), dtype=np.float32)

    out = np.zeros((n_features, nx_fno, ny_fno, 1), dtype=np.float32)

    for i in range(nx_fno):
        for j in range(ny_fno):
            x_in = x_min_fno + i * dx_fno
            y_in = y_min_fno + j * dy_fno

            ix = int((x_in - x_min_preq) / dx_preq)
            iy = int((y_in - y_min_preq) / dy_preq)
            ix = max(0, min(nx_preq - 1, ix))
            iy = max(0, min(ny_preq - 1, iy))
            idx = ix * ny_preq + iy

            ed = float(e_arr[idx])
            vx = float(ux_arr[idx])
            vy = float(uy_arr[idx])

            if n_features == 4:
                # Placeholder temperature (to be filled from EOS in Python)
                out[0, i, j, 0] = ed
                out[1, i, j, 0] = 0.0  # caller fills T via EOS
                out[2, i, j, 0] = vx
                out[3, i, j, 0] = vy
            elif n_features == 3:
                out[0, i, j, 0] = ed   # NOT tau-normalised; caller does that
                out[1, i, j, 0] = vx
                out[2, i, j, 0] = vy
            else:
                raise ValueError(
                    f"n_features={n_features} not supported. Use 3 or 4."
                )

    return out
