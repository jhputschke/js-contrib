"""YAML schema, defaults, validation and dotted overrides for fast_data.

Plain PyYAML, matching every other setup in this repository (`workflow_torch/train.py:50-154`):
`--config FILE` -> `yaml.safe_load` -> a resolved dict.  OmegaConf is proposed in
PLAN_OmegaConf.md but nothing in the repo uses it yet, and fast_data should not be the first.

Two deliberate strictnesses, because a data-generation run is expensive and a typo that silently
does nothing is worse than a crash:

* every key is checked against DEFAULTS and an unknown one raises, naming it;
* `--set a.b.c=1` raises if `a.b.c` is not already a known key, so `--set run.nevent=4` fails
  instead of quietly adding a dead entry.
"""

from __future__ import annotations

import copy
import math

__all__ = ["DEFAULTS", "load_config", "validate_config", "apply_overrides", "resolve_config",
           "derived_tau_grid", "ConfigError"]


class ConfigError(ValueError):
    """A problem in the configuration, raised before any compute happens."""


DEFAULTS = {
    "run": {
        "odir": "./out",
        "out": "./out/fastdata.h5",
        "overwrite": False,
        "nevents": 2,
        "batch_size": 1,
        "seed": 20260919,
        "device": "cpu",            # cpu | cuda | mps  (mps implies float32)
        "dtype": "float64",         # solver precision; `arr` is always float32 on disk
        "log": True,
        "progress_every": 1,
        "compile": False,           # torch.compile primitive_recovery (dynamic shapes).  Worth
                                    # ~2.7x/event at steady state but NOT bit-identical, and it
                                    # only pays from ~8 events cold / ~2 warm.  See
                                    # fv.enable_compiled_recovery.
    },
    "grid": {"nx": 65, "ny": 65, "neta": 33, "dx": 0.3125, "dy": 0.3125, "deta": 0.3125},
    "time": {
        "tau0": 0.58,
        "record_dtau": 0.1,
        "choose_ntau": 39,
        "tau_end": None,            # give this OR choose_ntau
        "cfl": 0.4,
        "hydro_dtau": None,         # None -> adaptive CFL; required for source.mode: xscape
        "dtau_max": None,
    },
    "initial_state": {
        "kind": "mc_glauber",       # mc_glauber | from_file | smooth
        "cache": None,              # optional IC .h5: written if absent, reused if present
        "proj": "d",
        "targ": "Au",
        "sigma_nn_mb": 42.0,
        "b": None,                  # None = min-bias; 5.0 = fixed; [lo, hi] = range
        "b_max": None,
        "require_collision": True,
        "quantity": "entropy",      # entropy (K*profile is s) | energy (K*profile is e)
        "K": None,                  # exactly one of K / target_T / target_e / target_s
        "target_T": 0.40,
        "target_e": None,
        "target_s": None,
        "calib_b": 0.0,
        "calib_events": None,
        "e_floor": 1.0e-6,
        "file": None,               # kind: from_file
        "start_event": 0,
        "eos_from_file": True,
        "profile": {"w": 0.4, "alpha": 0.145, "eta0": 1.5, "sig_eta": 1.3, "eta_m": 3.36,
                    "gamma_k": None, "string_fluct": False, "edge": 0.3},
        "smooth": {"e0": 15.0, "R": 3.0, "a2": 0.0, "psi2": 0.0, "a3": 0.0, "psi3": 0.0,
                   "eta_flat": 2.0, "sigma_eta": 1.0},
    },
    "eos": {
        "kind": "conformal",        # conformal | hotqcd | hotqcd_smash | music_check
        "dof": 47.5,
        "path": "./eos/hotQCD",
        "download": False,
        "store_table": True,
        "n_points": 1500,
    },
    "transport": {
        "mode": "ideal",            # ideal | israel_stewart
        "eta_over_s": 0.08,
        "zeta_over_s": 0.0,
        "tau_pi_coeff": 5.0,
        "delta_pipi": 4.0 / 3.0,
        "delta_PiPi": 2.0 / 3.0,
        "tau_min": 0.02,
        "pi_rho_max": 1.0,          # rescale pi above this |pi|/(e+p); null disables (see fv.regulate_pi)
        "pi_e_min": 1e-3,           # GeV/fm^3; freeze pi below this e and in capped cells; null disables
        "pi_advection": "centred",  # centred | upwind; the stencil pi is advected with (see fv._upwind_advect)
    },
    "source": {
        "enabled": False,
        "model": "causal_liquefier",
        "mode": "conservative",     # conservative | xscape
        "renorm": "grid",           # grid | extended
        "tau_eval_mode": "dep",     # dep | slice | step
        "n_sub": "auto",
        "n_sub_max": 16,
        "min_in_grid": 0.99,
        "on_out_of_grid": "warn",   # warn | drop | raise
        "per_event": "resample",    # resample | same
        "placement_weight": "uniform",   # uniform | energy_weighted
        "params": {"dtau": 0.02, "tau_delay": 2.0, "time_relax": 0.1,
                   "d_diff": 0.08, "width_delta": 0.1},
        "partons": None,            # a mapping (one parton) or a list of mappings
    },
    "output": {
        "compression": "lzf",
        "source_compression": "gzip",
        "chunk_events": 1,
        "T_fo": 0.150,
        "freezeout": "max_T",       # max_T | central_T | never
        "zero_after_freezeout": True,
        "stop_at_freezeout": True,
        "write_source": True,
        "write_diagnostics": True,
        "extra_attrs": {},
    },
}

#: sections whose contents are free-form and therefore not key-checked
_FREEFORM = {("output", "extra_attrs"), ("source", "partons")}


def _merge(base, over, path=()):
    """Recursive merge with unknown-key rejection."""
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        here = path + (k,)
        if k not in base:
            loc = ".".join(path) or "(top level)"
            raise ConfigError(f"unknown config key '{'.'.join(here)}' in {loc}; "
                              f"known keys there: {sorted(base)}")
        if isinstance(base[k], dict) and isinstance(v, dict) and here not in _FREEFORM:
            out[k] = _merge(base[k], v, here)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path, overrides=None):
    """Read a YAML file, merge it onto DEFAULTS, apply `--set` overrides, and validate."""
    import yaml

    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = _merge(DEFAULTS, raw)
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    return resolve_config(cfg)


def apply_overrides(cfg, overrides):
    """Apply `a.b.c=VALUE` strings.  VALUE is parsed as YAML; the path must already exist."""
    import yaml

    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ConfigError(f"override '{item}' is not of the form key.path=value")
        key, val = item.split("=", 1)
        parts = key.strip().split(".")
        node = cfg
        for pi in parts[:-1]:
            if not isinstance(node, dict) or pi not in node:
                raise ConfigError(f"override path '{key}' does not exist in the config")
            node = node[pi]
        if not isinstance(node, dict) or parts[-1] not in node:
            raise ConfigError(f"override path '{key}' does not exist in the config "
                              f"(did you mean one of {sorted(node) if isinstance(node, dict) else '?'}?)")
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def derived_tau_grid(cfg):
    """The exact list of record times.  Frame 0 is the initial condition."""
    t = cfg["time"]
    n = t["choose_ntau"]
    if t.get("tau_end") is not None:
        n_from_end = 1 + int(round((t["tau_end"] - t["tau0"]) / t["record_dtau"]))
        if n is not None and n != n_from_end:
            raise ConfigError(
                f"time.tau_end={t['tau_end']} implies choose_ntau={n_from_end}, "
                f"but choose_ntau={n} was given; set one or make them agree")
        n = n_from_end
    if not n or n < 2:
        raise ConfigError("time needs choose_ntau >= 2 (or a tau_end that implies it)")
    return [t["tau0"] + k * t["record_dtau"] for k in range(n)]


def resolve_config(cfg):
    """Fill derived fields and validate.  Returns the config; raises ConfigError on a problem."""
    taus = derived_tau_grid(cfg)
    cfg["time"]["choose_ntau"] = len(taus)
    cfg["time"]["tau_end"] = taus[-1]
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    run, grid, t = cfg["run"], cfg["grid"], cfg["time"]
    ini, src, out = cfg["initial_state"], cfg["source"], cfg["output"]

    if run["device"] == "mps" and run["dtype"] != "float32":
        raise ConfigError("run.device=mps has no float64; set run.dtype: float32 "
                          "(verified accurate: float32 matches float64 to ~1e-6 on the "
                          "conserved state over a full evolution)")
    if run["dtype"] not in ("float32", "float64"):
        raise ConfigError(f"run.dtype must be float32 or float64 (got {run['dtype']!r})")
    if run["batch_size"] > 1 and src["enabled"]:
        raise ConfigError("run.batch_size > 1 is not supported with a jet source "
                          "(freeze-out, droplets and diagnostics are all per event); use "
                          "--shard for parallelism instead")
    for k in ("nx", "ny", "neta"):
        if grid[k] < 2:
            raise ConfigError(f"grid.{k} must be >= 2")

    if ini["kind"] == "mc_glauber":
        norms = [k for k in ("K", "target_T", "target_e", "target_s") if ini.get(k) is not None]
        if len(norms) != 1:
            raise ConfigError(
                f"initial_state needs exactly one of K / target_T / target_e / target_s "
                f"(got {norms or 'none'}); they are mutually exclusive normalisations")
    elif ini["kind"] == "from_file":
        if not ini.get("file"):
            raise ConfigError("initial_state.kind=from_file needs initial_state.file")
    elif ini["kind"] != "smooth":
        raise ConfigError(f"unknown initial_state.kind {ini['kind']!r}")

    if cfg["transport"]["mode"] not in ("ideal", "israel_stewart"):
        raise ConfigError(f"unknown transport.mode {cfg['transport']['mode']!r}")
    if cfg["transport"]["pi_advection"] not in ("centred", "upwind"):
        raise ConfigError(f"unknown transport.pi_advection {cfg['transport']['pi_advection']!r}")
    if out["freezeout"] not in ("max_T", "central_T", "never"):
        raise ConfigError(f"unknown output.freezeout {out['freezeout']!r}")

    if src["enabled"]:
        if src["model"] != "causal_liquefier":
            raise ConfigError(f"unknown source.model {src['model']!r}")
        if src["mode"] not in ("conservative", "xscape"):
            raise ConfigError(f"unknown source.mode {src['mode']!r}")
        if src["partons"] is None:
            raise ConfigError("source.enabled is true but source.partons is empty")
        # The xscape mode snaps the kernel to the hydro node, as the C++ does, so it needs the
        # fixed step the C++ assumes.  The default (conservative) evaluates at tau_dep and is
        # provably step-size independent, so it imposes nothing.
        if src["mode"] == "xscape":
            hd, ld = t["hydro_dtau"], src["params"]["dtau"]
            if hd is None:
                raise ConfigError("source.mode=xscape reproduces the C++ point sampling on the "
                                  "hydro node grid, so it needs a fixed time.hydro_dtau "
                                  f"(set it to source.params.dtau = {ld})")
            if abs(hd - ld) > 1e-12:
                raise ConfigError(f"source.mode=xscape needs time.hydro_dtau ({hd}) == "
                                  f"source.params.dtau ({ld})")
            ratio = t["record_dtau"] / hd
            if abs(ratio - round(ratio)) > 1e-9:
                raise ConfigError(f"time.record_dtau ({t['record_dtau']}) must be an integer "
                                  f"multiple of time.hydro_dtau ({hd})")
            # The CFL bound is tightest at tau0, because the eta light speed is 1/tau.
            cfl_max = t["cfl"] * min(grid["dx"], grid["dy"], t["tau0"] * grid["deta"])
            if hd > cfl_max:
                raise ConfigError(
                    f"time.hydro_dtau ({hd}) exceeds the CFL bound at tau0, "
                    f"{t['cfl']}*min(dx, dy, tau0*deta) = {cfl_max:.5f}.  rollout would silently "
                    f"shorten the step and desynchronise the single-step deposit window.")
    return cfg
