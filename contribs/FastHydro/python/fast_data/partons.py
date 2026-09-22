"""Turning the `source:` config block into this event's droplets.

Two things live here rather than in `liquefier/droplets.py`, because they need the event:

* **`tau: auto`.**  d+Au freeze-out runs 1.98-4.08 fm while the default `tau_delay` is 2.0 fm, so
  a naively placed parton very often deposits at or after freeze-out, into a medium that is no
  longer there.  `auto` samples the production time so the DEPOSIT lands inside
  ``[tau0 + margin_lo, tau_fo - margin_hi]``, using this event's own freeze-out time.
* **fireball-weighted placement**, which samples (x, y, eta) from the initial energy density so a
  test parton starts inside the medium rather than in vacuum.

Seeding: the parton stream is deliberately INDEPENDENT of the initial-state stream
(``SeedSequence(master).spawn(2)``).  Editing the `source:` block therefore cannot perturb the
Glauber sampling, so a with-source and a without-source dataset stay matched event for event and
can be differenced, and `glauber.regenerate_event` keeps working.
"""

from __future__ import annotations

import numpy as np

from .liquefier import DropletArray, DropletFlags, LiquefierParams, partons_from_config

__all__ = ["seed_streams", "build_droplets", "auto_tau_window"]


def seed_streams(master_seed, nevents):
    """-> (ic_seeds, source_seeds), each (nevents,) uint64.

    `ic_seeds` is generated exactly as `glauber.save_events` does, so a stage-1 IC file and an
    inline run with the same master seed produce the same events.
    """
    ss_ic, ss_src = np.random.SeedSequence(int(master_seed)).spawn(2)
    return (ss_ic.generate_state(int(nevents), dtype=np.uint64),
            ss_src.generate_state(int(nevents), dtype=np.uint64))


def auto_tau_window(tau0, tau_fo, params: LiquefierParams, margin_lo=0.2, margin_hi=0.5,
                    tau_prod_min=0.05):
    """The DEPOSIT window for `tau: auto`, or None if this event has no usable one.

    Two constraints, and the second is easy to miss: the deposit has to land inside the fireball
    lifetime, AND the production time it implies, ``tau_dep - tau_delay``, has to be a positive
    proper time.  With the X-SCAPE default tau_delay = 2.0 fm and a d+Au event that freezes out
    at 1.98 fm there is no such window at all -- every deposit inside the fireball would need a
    parton produced before the collision.  That is a real physical conflict, not a rounding
    detail, so it is reported rather than quietly clipped.
    """
    lo = max(tau0 + margin_lo, params.tau_delay + tau_prod_min)
    hi = tau_fo - margin_hi
    return (lo, hi) if hi > lo else None


def _weighted_positions(e, grid, rng, n, e_min=0.0):
    """Sample n (x, y, eta) points with probability proportional to the energy density."""
    w = np.asarray(e, dtype=np.float64).ravel().copy()
    w[w < e_min] = 0.0
    if not np.any(w):
        return None
    idx = rng.choice(w.size, size=n, p=w / w.sum())
    nx, ny, nz = np.asarray(e).shape
    ix, iy, ie = np.unravel_index(idx, (nx, ny, nz))
    gx = np.asarray(grid.x.detach().cpu(), dtype=np.float64)
    gy = np.asarray(grid.y.detach().cpu(), dtype=np.float64)
    ge = np.asarray(grid.eta.detach().cpu(), dtype=np.float64)
    # jitter inside the chosen cell so the deposit is not locked to cell centres
    return np.column_stack([
        gx[ix] + (rng.random(n) - 0.5) * float(grid.dx),
        gy[iy] + (rng.random(n) - 0.5) * float(grid.dy),
        ge[ie] + (rng.random(n) - 0.5) * float(grid.deta),
    ])


def build_droplets(cfg_source, rng, params, *, tau0, tau_fo=None, e_initial=None, grid=None,
                   log=None):
    """The `source:` block -> a single-event DropletArray, with flags already set.

    `tau_fo` (this event's freeze-out time) enables `tau: auto` and the AFTER_FREEZEOUT flag.
    `e_initial` + `grid` enable `placement_weight: energy_weighted`.
    """
    spec = cfg_source.get("partons")
    if spec is None:
        return DropletArray.empty(1)

    window = None
    if tau_fo is not None:
        window = auto_tau_window(tau0, float(tau_fo), params)
        if window is None and _wants_auto(spec):
            raise ValueError(
                f"source uses `tau: auto`, but this event freezes out at tau_fo={tau_fo:.3f} fm "
                f"and tau_delay={params.tau_delay} fm leaves no deposit window after tau0="
                f"{tau0:.3f}.  Lower source.params.tau_delay, or give an explicit tau.")

    drops = partons_from_config(spec, rng, params, tau_window=window)
    if len(drops) == 0:
        return drops

    if cfg_source.get("placement_weight", "uniform") == "energy_weighted":
        if e_initial is None or grid is None:
            raise ValueError("source.placement_weight=energy_weighted needs the initial energy "
                             "density and the grid (available only for a generated IC)")
        pos = _weighted_positions(e_initial, grid, rng, len(drops))
        if pos is None:
            raise ValueError("source.placement_weight=energy_weighted: the initial energy density "
                             "is empty everywhere")
        drops.data[:, 1:4] = pos

    # --- flags: recorded, never used to silently drop anything -------------------------
    tau_dep = drops.tau_dep(params)
    drops.flags |= np.where(tau_dep < tau0, DropletFlags.BEFORE_TAU0, 0).astype(np.uint32)
    if tau_fo is not None:
        drops.flags |= np.where(tau_dep >= float(tau_fo),
                                DropletFlags.AFTER_FREEZEOUT, 0).astype(np.uint32)
    if grid is not None:
        gx = np.asarray(grid.x.detach().cpu()); gy = np.asarray(grid.y.detach().cpu())
        ge = np.asarray(grid.eta.detach().cpu())
        outside = ((drops.data[:, 1] < gx.min()) | (drops.data[:, 1] > gx.max())
                   | (drops.data[:, 2] < gy.min()) | (drops.data[:, 2] > gy.max())
                   | (drops.data[:, 3] < ge.min()) | (drops.data[:, 3] > ge.max()))
        drops.flags |= np.where(outside, DropletFlags.OUT_OF_GRID, 0).astype(np.uint32)

    if log is not None:
        n_after = int((drops.flags & DropletFlags.AFTER_FREEZEOUT).astype(bool).sum())
        n_before = int((drops.flags & DropletFlags.BEFORE_TAU0).astype(bool).sum())
        if n_after or n_before:
            log(f"  WARNING: {n_after} droplet(s) deposit at or after freeze-out, "
                f"{n_before} before tau0; they are flagged, not dropped")
    return drops


def _wants_auto(spec):
    entries = [spec] if isinstance(spec, dict) else list(spec or [])
    return any(isinstance(e, dict) and e.get("tau", None) == "auto" for e in entries)
