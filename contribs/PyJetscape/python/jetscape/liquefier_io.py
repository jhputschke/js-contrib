"""
python/jetscape/liquefier_io.py

Read the droplets and the kernel parameters out of a live C++ ``CausalLiquefier``.

The droplets are what the jet deposited into the medium this event: one row per droplet,
``(tau, x, y, eta, E, px, py, pz)`` -- Milne position, Cartesian momentum in GeV --
exactly as ``LiquefierBase::add_hydro_sources`` stores them (``LiquefierBase.cc:200-225``).
No conversion is applied.  A MUSIC jet leg evaluates the same droplets itself, through
``HydroSourceJETSCAPE``; these helpers only record them.

Nothing here imports fast_data or torch, so any writer can use it.  FastHydro's
``liquefier_bridge`` wraps the same calls into its own types.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["DROPLET_COLUMNS", "PARAM_KEYS", "droplets", "liquefier_params"]

#: columns of one droplet row; the name and order fast_data stores as ``droplet_columns``
DROPLET_COLUMNS = ("tau", "x", "y", "eta", "E", "px", "py", "pz")

#: keys of :func:`liquefier_params`, in the order FastHydro writes ``liquefier_<key>`` attrs
PARAM_KEYS = ("dtau", "tau_delay", "time_relax", "d_diff", "width_delta", "c_diff",
              "gamma_relax")


def droplets(liq):
    """The liquefier's current droplet list as an ``(M, 8)`` float64 array.

    ``liq`` is a bound ``LiquefierBase``/``CausalLiquefier`` (``droplets_numpy()``).  The list
    holds this event's droplets until a hydro with the liquefier attached clears it in its
    ``ClearTask``.
    """
    return np.asarray(liq.droplets_numpy(), dtype=np.float64).reshape(-1, len(DROPLET_COLUMNS))


def liquefier_params(liq):
    """The causal-diffusion kernel parameters as a dict of floats, keyed by :data:`PARAM_KEYS`.

    ``liq.params()`` gives the five input parameters.  The two derived ones come from the
    object when it exposes them (``CausalLiquefier.c_diff``/``gamma_relax``), otherwise from
    the C++ formulas ``c_diff = sqrt(d_diff/time_relax)`` and ``gamma_relax =
    0.5/time_relax`` (``CausalLiquefier.cc:41-42``).
    """
    p = {k: float(v) for k, v in dict(liq.params()).items()}
    c_diff = getattr(liq, "c_diff", None)
    gamma_relax = getattr(liq, "gamma_relax", None)
    p["c_diff"] = float(math.sqrt(p["d_diff"] / p["time_relax"]) if c_diff is None else c_diff)
    p["gamma_relax"] = float(0.5 / p["time_relax"] if gamma_relax is None else gamma_relax)
    return {k: p[k] for k in PARAM_KEYS}
