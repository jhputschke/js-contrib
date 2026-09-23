"""Bound fast_data's bulk pressure: Pi/p in [lo, hi] after every viscous step.  Opt-in.

    fasthydro:
      hydro:
        bulk_clamp: [-0.9, 0.3]      # null (the default) leaves fast_data untouched

Why this exists.  fast_data's bulk sector had never run (zeta/s = 0 in every shipped config)
until the 0-10% tune (config/AuAu_FastHydro_tune_0_10.yaml) needed zeta/s = 0.12.  With any
zeta/s > 0, about one central Au+Au event in five diverges, at tau ~ 4.5 fm/c:

  - `fv.viscous_step` freezes Pi in cells below `transport.pi_e_min` or with a capped velocity,
    and a frozen cell keeps its last Pi;
  - the pressure of that dilute corona keeps falling under the stale Pi.  Measured: Pi/p
    drifts from -0.3 to -5.8 over 2 fm/c, p + Pi < 0, and the cell goes non-finite;
  - the regulator bounds |Pi| by pi_rho_max * (e + p), which near T_c is 4-7 p, so it does not
    stop this.  The primitive recovery is validated for Pi/p in [-0.9, +0.3] (fv.py:324).

Raising `pi_e_min` does not help: freezing is what makes Pi stale.  So this clamps Pi to the
recovery's validated range after every viscous step.  In the evolving fluid Pi/p stays above
-0.66 at zeta/s = 0.12, so only the corona, far below T_sw, is touched.

`python/fast_data/` is vendored and never patched (VENDORING.md), and `fv.strang_step` looks
up `viscous_step` as a module global.  So this wraps that global, process-wide and
idempotently.  It is a stopgap: the fix belongs upstream in FNO4d's fast_data
(`fv.viscous_step`), after which this module and the setting can go.
"""

from __future__ import annotations

__all__ = ["install", "installed_bounds", "from_cfg"]

_ORIG = None          # fast_data.fv.viscous_step before any wrapping
_BOUNDS = None        # the (lo, hi) currently installed, or None


def installed_bounds():
    return _BOUNDS


def install(bounds):
    """Wrap fast_data.fv.viscous_step to bound Pi/p by ``bounds`` = (lo, hi); None unwraps.

    Idempotent.  The wrap is process-wide, so every solver in the process gets it -- which is
    what a two-leg run on one config wants.  -> the bounds now installed.
    """
    global _ORIG, _BOUNDS
    import torch
    from fast_data import fv

    if _ORIG is None:
        _ORIG = fv.viscous_step
    if bounds is None:
        fv.viscous_step = _ORIG
        _BOUNDS = None
        return None
    lo, hi = (float(b) for b in bounds)
    if not lo < 0.0 < hi:
        raise ValueError(f"bulk_clamp must bracket 0, got {bounds}")

    def viscous_step(pi10, Pi, prim, *a, **kw):
        pi_new, Pi_new = _ORIG(pi10, Pi, prim, *a, **kw)
        p = prim["p"].unsqueeze(1)
        return pi_new, torch.maximum(torch.minimum(Pi_new, hi * p), lo * p)

    viscous_step.bulk_clamp = (lo, hi)
    fv.viscous_step = viscous_step
    _BOUNDS = (lo, hi)
    return _BOUNDS


def from_cfg(cfg, verbose=True):
    """Install (or remove) the clamp a resolved config asks for, and warn when bulk runs
    without one.  Call wherever a fast_data Transport is built from ``cfg``."""
    bounds = ((cfg.get("fasthydro") or {}).get("hydro") or {}).get("bulk_clamp")
    tr = cfg.get("transport", {})
    viscous = str(tr.get("mode", "ideal")).lower() in ("israel_stewart", "viscous", "is")
    if viscous and float(tr.get("zeta_over_s") or 0.0) > 0.0 and bounds is None and verbose:
        print("[fasthydro] WARNING: transport.zeta_over_s > 0 without fasthydro.hydro.bulk_clamp;"
              " fast_data's frozen corona cells can drive p + Pi < 0 and diverge "
              "(see fasthydro/bulk_regulator.py)", flush=True)
    return install(bounds)
