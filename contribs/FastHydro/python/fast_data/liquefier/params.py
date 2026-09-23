"""Parameters of the causal (telegraph) diffusion kernel, and the constants derived from them.

Ported from X-SCAPE `src/liquefier/CausalLiquefier.cc:27-43`; the defaults are the ones in
`config/jetscape_main.xml:426-435`.

A note that saves a reader an hour: of the five XML parameters, **only `time_relax`, `d_diff` and
`width_delta` affect the physics here**.  The C++ also reads `dx`, `dy`, `deta` but never uses them
in any formula, and its `dtau` enters only as a `1/dtau` normalisation that cancels against the
`* dtau_step` in the hydro update (see `source.py`).  `dtau` is kept so it can be recorded in the
output attributes and so `mode="xscape"` can reproduce the C++ point-sampling exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

__all__ = ["LiquefierParams", "XSCAPE_DEFAULTS"]

#: config/jetscape_main.xml:426-435
XSCAPE_DEFAULTS = dict(dtau=0.02, tau_delay=2.0, time_relax=0.1, d_diff=0.08, width_delta=0.1)


@dataclass(frozen=True)
class LiquefierParams:
    """Causal-diffusion parameters, all in fm (``dtau``, ``tau_delay``, ``time_relax``,
    ``d_diff``, ``width_delta``)."""

    dtau: float = 0.02
    tau_delay: float = 2.0
    time_relax: float = 0.1
    d_diff: float = 0.08
    width_delta: float = 0.1

    def __post_init__(self):
        if self.time_relax <= 0:
            raise ValueError(f"time_relax must be > 0 (got {self.time_relax})")
        if self.d_diff < 0:
            raise ValueError(f"d_diff must be >= 0 (got {self.d_diff})")
        if self.width_delta <= 0:
            raise ValueError(f"width_delta must be > 0 (got {self.width_delta})")
        if self.c_diff > 1.0:
            raise ValueError(
                f"c_diff = sqrt(d_diff/time_relax) = {self.c_diff:.4f} > 1 is superluminal; "
                f"the C++ only warns, but it would break the causality guarantees this port "
                f"relies on (the ds^2 >= 0 cut becomes non-redundant).")

    # ---- derived constants (CausalLiquefier.cc:41-42) ----
    @property
    def c_diff(self) -> float:
        """Signal velocity in units of c; sqrt(0.8) = 0.894427190999916 with the defaults."""
        return math.sqrt(self.d_diff / self.time_relax)

    @property
    def gamma_relax(self) -> float:
        """Relaxation rate in 1/fm; 5.0 with the defaults."""
        return 0.5 / self.time_relax

    def deposit_tau(self, tau_droplet):
        """The tau at which a droplet produced at `tau_droplet` deposits."""
        return tau_droplet + self.tau_delay

    def replace(self, **kw) -> "LiquefierParams":
        return replace(self, **kw)

    @classmethod
    def from_config(cls, cfg) -> "LiquefierParams":
        known = {f: cfg[f] for f in ("dtau", "tau_delay", "time_relax", "d_diff", "width_delta")
                 if f in (cfg or {})}
        unknown = set(cfg or {}) - {"dtau", "tau_delay", "time_relax", "d_diff", "width_delta"}
        if unknown:
            raise ValueError(f"unknown liquefier parameter(s): {sorted(unknown)}")
        return cls(**{k: float(v) for k, v in known.items()})

    def to_attrs(self, prefix="liquefier_") -> dict:
        """Flat dict for HDF5 root attributes, including the derived constants."""
        return {
            f"{prefix}dtau": self.dtau,
            f"{prefix}tau_delay": self.tau_delay,
            f"{prefix}time_relax": self.time_relax,
            f"{prefix}d_diff": self.d_diff,
            f"{prefix}width_delta": self.width_delta,
            f"{prefix}c_diff": self.c_diff,
            f"{prefix}gamma_relax": self.gamma_relax,
        }

    def __str__(self):
        return (f"LiquefierParams(tau_delay={self.tau_delay}, time_relax={self.time_relax}, "
                f"d_diff={self.d_diff}, width_delta={self.width_delta} fm "
                f"-> c_diff={self.c_diff:.6f}, gamma_relax={self.gamma_relax:.4f}/fm)")
