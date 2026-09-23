"""FastHydro's own config block, layered on top of `fast_data`'s.

`fast_data.config` validates strictly: any key not in its DEFAULTS raises, naming it.  That is
a good property and it is not ours to change -- `python/fast_data/` is vendored and never
patched (VENDORING.md).  But FastHydro has settings fast_data knows nothing about, because
they are about the JETSCAPE side: where hard scatterings go, what to store in `bulk_info`.

So the YAML carries one extra top-level section, `fasthydro:`, which is popped here before the
rest is handed to `fast_data.config.load_config`, validated against its own defaults with the
same unknown-key strictness, and attached back onto the returned dict as ``cfg["fasthydro"]``.
One file for the user, one schema each.

    cfg = fasthydro.config.load_config("fasthydro_twostage.yaml")
    cfg["grid"]["nx"]                       # fast_data's
    cfg["fasthydro"]["hard_vertex"]["mode"] # ours
"""

from __future__ import annotations

import copy

__all__ = ["DEFAULTS", "SECTION", "load_config", "resolve", "validate"]

#: the top-level YAML key this module owns
SECTION = "fasthydro"

DEFAULTS = {
    # Where hard scatterings happen.  See fasthydro/hard_vertex.py.
    "hard_vertex": {
        "mode": "ncoll",      # ncoll | ncoll_mc | npart | centre
        "smear": 0.4,         # fm; the nucleon width glauber deposits energy with
    },
    "hydro": {
        # FreestreamMilne-style pre-equilibrium carries flow and viscous stress that
        # fv.initial_state_from_energy cannot represent (it starts from u=(1,0,0,0), pi=Pi=0).
        # FastHydro refuses rather than discard it silently; set this to proceed anyway.
        "accept_preeq_flow_loss": False,
        # "vector" = bulk_info.data_vector, 4 B per stored field per cell.
        # "aos"    = bulk_info.data, 112 B per cell regardless. 390 MB vs 1.56 GB at 65x65x33x100.
        "store": "vector",
        # null -> fasthydro.cells.DEFAULT_FIELDS
        "store_fields": None,
    },
    # The parton shower itself -- every parton and splitting vertex, written to `shower/`.
    # Default on: measured at 9.4 kB/event against 5.5 MB for the hydro pair, and without it
    # a file records only what the jet LOST (source/droplets), never where the jet was.
    "store_showers": True,
}


def _merge(defaults, user, path=()):
    """Recursive merge with unknown-key rejection, mirroring fast_data.config."""
    from fast_data.config import ConfigError

    out = copy.deepcopy(defaults)
    for k, v in (user or {}).items():
        here = path + (k,)
        if k not in defaults:
            raise ConfigError(
                f"unknown config key '{SECTION}.{'.'.join(here)}'; "
                f"known keys here: {sorted(defaults)}")
        if isinstance(defaults[k], dict) and isinstance(v, dict):
            out[k] = _merge(defaults[k], v, here)
        else:
            out[k] = v
    return out


def validate(block):
    from fast_data.config import ConfigError

    from .cells import LEGAL_FIELDS, validate_fields
    from .hard_vertex import MODES

    hv = block["hard_vertex"]
    if hv["mode"] not in MODES:
        raise ConfigError(
            f"unknown {SECTION}.hard_vertex.mode {hv['mode']!r}. Choose one of:\n" +
            "\n".join(f"  {k:10s} {v}" for k, v in MODES.items()))
    if hv["mode"] in ("ncoll", "npart") and float(hv["smear"]) <= 0:
        raise ConfigError(
            f"{SECTION}.hard_vertex.smear must be > 0 for mode={hv['mode']!r} "
            f"(got {hv['smear']}); use mode: ncoll_mc for an unsmeared histogram")

    hyd = block["hydro"]
    if hyd["store"] not in ("vector", "aos"):
        raise ConfigError(f"{SECTION}.hydro.store must be 'vector' or 'aos' "
                          f"(got {hyd['store']!r})")
    if hyd["store_fields"] is not None:
        try:
            validate_fields(hyd["store_fields"])
        except ValueError as exc:
            raise ConfigError(f"{SECTION}.hydro.store_fields: {exc}") from None
    return block


def resolve(user_block):
    """Merge a user `fasthydro:` mapping onto DEFAULTS and validate it."""
    return validate(_merge(DEFAULTS, user_block or {}))


def load_config(path, overrides=None):
    """Load a YAML carrying both schemas.

    The `fasthydro:` section is split off before `fast_data` sees it, so fast_data's
    unknown-key check stays as strict as it is meant to be.

    `overrides` are dotted `key.path=value` strings; those starting with `fasthydro.` are
    applied to our block, the rest go to fast_data.
    """
    import yaml

    from fast_data.config import DEFAULTS as FD_DEFAULTS
    from fast_data.config import _merge as fd_merge
    from fast_data.config import apply_overrides, resolve_config

    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    mine = raw.pop(SECTION, {})

    ours = [o for o in (overrides or []) if o.split("=", 1)[0].startswith(SECTION + ".")]
    theirs = [o for o in (overrides or []) if o not in ours]

    # fast_data.load_config() reads the file itself, so replicate its body on `raw`:
    # merge onto DEFAULTS (which is also its unknown-key check), then resolve+validate.
    cfg = fd_merge(FD_DEFAULTS, raw)
    if theirs:
        cfg = apply_overrides(cfg, theirs)
    cfg = resolve_config(cfg)

    mine = _merge(DEFAULTS, mine)
    for o in ours:
        key, _, val = o.partition("=")
        _set_dotted(mine, key.split(".")[1:], val)
    cfg[SECTION] = validate(mine)
    return cfg


def _set_dotted(d, parts, value):
    from fast_data.config import ConfigError

    node = d
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            raise ConfigError(f"override path '{SECTION}.{'.'.join(parts)}' does not exist")
        node = node[p]
    leaf = parts[-1]
    if leaf not in node:
        raise ConfigError(f"override path '{SECTION}.{'.'.join(parts)}' does not exist")
    cur = node[leaf]
    if isinstance(cur, bool):
        node[leaf] = str(value).strip().lower() in ("1", "true", "yes", "on")
    elif isinstance(cur, (int, float)) and cur is not None:
        node[leaf] = type(cur)(value)
    else:
        node[leaf] = value
