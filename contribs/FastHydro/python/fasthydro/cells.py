"""Solver frame -> the feature array `bulk_info` stores.

`evolve_event` gives `(4, nx, ny, neta, ntau)` float32 = ``[e, vx, vy, vz]`` where ``e`` is
GeV/fm^3 and ``vx, vy, vz`` are **Cartesian lab three-velocities** (`fast_data.convert`
rebuilds u^tau from the spatial parts exactly as `HydroinfoMUSIC.cpp:316-329` does).
`FluidCellInfo` wants the same quantities in the same units, so those four are a straight copy.

Two more fields the jet chain actually reads are not in the solver output and are computed
here from the same EoS the solver used:

    temperature      [GeV]    eos.T(e)          Matter.cc:495,777 ; LBT.cc:702,806 ;
                                                LiquefierBase::filter_partons
    entropy_density  [1/fm^3] (e + p) / T       LBT.cc:703,807 ; Matter QhatParametrizationType 4
    pressure         [GeV/fm^3] eos.p(e)        cheap, filled for completeness

`pi^{mu nu}` and `bulk_Pi` are left ZERO.  A grep of `src/jet/Matter.cc`, `src/jet/LBT.cc` and
`src/framework/LiquefierBase.cc` shows they read only temperature, entropy_density and
vx/vy/vz, so this has no effect on the jet chain.  It would matter for Cooper-Frye, which this
contribution does not provide at all.

The field names below must be exactly those `ResolveEntryName`
(`src/framework/FluidEvolutionHistory.cc:40-71`) accepts.
"""

from __future__ import annotations

import numpy as np

#: every name `ResolveEntryName` accepts, for validation with a useful error message
LEGAL_FIELDS = (
    "energy_density", "entropy_density", "temperature", "pressure", "qgp_fraction",
    "mu_b", "mu_c", "mu_s", "vx", "vy", "vz",
    "pi00", "pi01", "pi02", "pi03", "pi11", "pi12", "pi13", "pi22", "pi23", "pi33",
    "bulk_pi",
)

#: what FastHydro stores by default.  7 fields x 4 B x 13.9 M cells = 390 MB at 65x65x33x100,
#: against 1.56 GB for the array-of-FluidCellInfo route (112 B/cell).
DEFAULT_FIELDS = (
    "energy_density", "temperature", "entropy_density", "pressure", "vx", "vy", "vz",
)

#: below this temperature a cell is vacuum: do not divide by it
T_FLOOR = 1e-6


def validate_fields(fields) -> tuple[str, ...]:
    fields = tuple(fields)
    bad = [f for f in fields if f not in LEGAL_FIELDS]
    if bad:
        raise ValueError(
            f"unknown bulk_info field(s) {bad}. "
            f"ResolveEntryName accepts only: {', '.join(LEGAL_FIELDS)}"
        )
    if not fields:
        raise ValueError("at least one field must be stored")
    for required in ("energy_density", "temperature"):
        if required not in fields:
            raise ValueError(f"{required!r} is required by the jet energy-loss modules")
    return fields


def frame_to_cells(arr, eos, fields=DEFAULT_FIELDS, *, device=None, dtype=None):
    """``arr`` (4, nx, ny, neta, ntau) float32 [e, vx, vy, vz] -> (F, ...) float32.

    Parameters
    ----------
    arr : np.ndarray
        The solver output buffer, as written by `fast_data.evolve.evolve_event`.
    eos : ConformalEoS | TabulatedEoS
        The *same* EoS object the solver ran with -- not a re-read table.
    fields : sequence of str
        Names from :data:`LEGAL_FIELDS`; see :data:`DEFAULT_FIELDS`.
    """
    import torch

    fields = validate_fields(fields)
    if arr.ndim != 5 or arr.shape[0] != 4:
        raise ValueError(f"expected (4, nx, ny, neta, ntau), got {arr.shape}")

    e_np, vx, vy, vz = arr[0], arr[1], arr[2], arr[3]

    # Evaluate the EoS in the solver's own precision, then narrow to float32 once.
    e = torch.as_tensor(e_np, device=device or getattr(eos, "device", None),
                        dtype=dtype or getattr(eos, "dtype", torch.float64))
    e_pos = e.clamp_min(0.0)
    need_T = "temperature" in fields or "entropy_density" in fields
    need_p = "pressure" in fields or "entropy_density" in fields
    T = eos.T(e_pos) if need_T else None
    p = eos.p(e_pos) if need_p else None

    if "entropy_density" in fields:
        # mu_B = 0  =>  s = (e + p) / T.  Vacuum cells have no entropy, and T -> 0 there.
        s = torch.where(T > T_FLOOR, (e_pos + p) / T.clamp_min(T_FLOOR),
                        torch.zeros_like(e_pos))
        s = torch.where(e_pos > 0, s, torch.zeros_like(s))
    else:
        s = None

    def np32(t):
        return np.ascontiguousarray(t.detach().cpu().numpy(), dtype=np.float32)

    source = {
        "energy_density": lambda: np.ascontiguousarray(e_np, dtype=np.float32),
        "temperature": lambda: np32(T),
        "entropy_density": lambda: np32(s),
        "pressure": lambda: np32(p),
        "vx": lambda: np.ascontiguousarray(vx, dtype=np.float32),
        "vy": lambda: np.ascontiguousarray(vy, dtype=np.float32),
        "vz": lambda: np.ascontiguousarray(vz, dtype=np.float32),
    }

    out = np.empty((len(fields),) + arr.shape[1:], dtype=np.float32)
    for i, name in enumerate(fields):
        make = source.get(name)
        out[i] = make() if make is not None else 0.0   # pi**, bulk_pi, mu_*, qgp_fraction
    return out
