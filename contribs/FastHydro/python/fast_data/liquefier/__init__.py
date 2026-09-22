"""Pure-Python port of X-SCAPE's CausalLiquefier jet source.

No GSL, no X-SCAPE runtime: the kernel is the analytic Green's function of causal (telegraph)
diffusion, reimplemented from `src/liquefier/CausalLiquefier.cc` and `src/framework/LiquefierBase.cc`
in numpy, and validated against the C++ unit tests in `examples/unittests/causal_liquifier.cc`.

    params    parameter block and derived constants (c_diff, gamma_relax)
    kernel    the Green's function itself
    droplets  droplet container, parton -> droplet, YAML placement
    deposit   grid weights and the Milne basis conversion
    source    CausalLiquefierSource, duck-typed to the FV solver's source hook

`source` imports torch lazily (only inside `step`), so importing this package does not.
"""

from .deposit import DepositPatch, droplet_weights, milne_dq_from_cartesian, support_box
from .droplets import (COLUMNS, DropletArray, DropletFlags, flag_names,
                       parton_to_droplet, partons_from_config, sample_value)
from .params import XSCAPE_DEFAULTS, LiquefierParams
from .source import CausalLiquefierSource

__all__ = [
    "LiquefierParams", "XSCAPE_DEFAULTS",
    "DropletArray", "DropletFlags", "COLUMNS", "flag_names",
    "parton_to_droplet", "partons_from_config", "sample_value",
    "DepositPatch", "droplet_weights", "milne_dq_from_cartesian", "support_box",
    "CausalLiquefierSource",
]
