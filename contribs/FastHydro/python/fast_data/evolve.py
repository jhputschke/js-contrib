"""The stepping loop, streaming each record straight into the float32 output buffers.

`fv.rollout` stacks every record of `q` (and of `src`) in solver precision and holds them all: at
65x65x33 float64 with 39 frames that is ~180 MB for each, and it scales with `choose_ntau`.  Since
the output is float32 and only needs (e, vx, vy, vz), converting each record as it is produced and
dropping the float64 state costs one pass and bounds the memory at two frame buffers.

The loop is otherwise a line-for-line copy of `rollout` -- same clipping of the step to the next
record boundary (including the same absorption of a residual step into the one before it, see
`_substeps_per_frame`), same `strang_step` call, same `src` accumulation between records -- so the
two agree record for record, and `rollout` stays available for interactive use and parity checks.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from . import fv as _fv
from .convert import freezeout_index, q_to_fno_frame, zero_after_freezeout

__all__ = ["evolve_event", "EvolutionDiverged"]

#: float32 cannot hold more than this, and a training file must never contain inf or NaN
_F32_MAX = float(np.finfo(np.float32).max)

#: One definition, shared with `fv.rollout`, which clips to its record boundary the same way
_ABSORB = _fv.DTAU_ABSORB

#: below this fraction of the nominal step, a substep is not a small step, it is an amplifier
_STUB = 1e-3


class EvolutionDiverged(RuntimeError):
    """The solver produced a frame that cannot be written as valid training data."""


def _substeps_per_frame(tau_grid, nominal):
    """Refuse a record axis that a fixed substep does not divide.  Returns the substep count.

    A fixed substep that does not divide the record spacing leaves a residual step at every
    frame boundary, and `fv.strang_step` forms d_tau u as a difference of two Newton-recovered
    velocities *divided by the step*: at a residual of 1e-9 fm/c that is recovery noise
    amplified by 1e7.  It goes into the shear source, and the Israel-Stewart run dies a few
    frames later -- looking like an IS instability, since the ideal sector never consumes
    d_tau u and so never notices.

    Residuals below `_ABSORB` of the nominal step are float representation crumbs (a tau axis
    that has been through float32 anywhere carries them) and the loop absorbs those.  Anything
    larger was asked for, and is refused here rather than quietly turned into noise.
    """
    counts = []
    for k in range(1, len(tau_grid)):
        rec = tau_grid[k] - tau_grid[k - 1]
        ratio = rec / nominal
        n = round(ratio)
        if abs(ratio - n) > _ABSORB:
            what = ("it is larger than that spacing, so every step would be clipped to it"
                    if ratio < 1.0 else
                    f"every frame would end on a residual step of "
                    f"{rec - int(ratio) * nominal:.3g} fm/c")
            raise ValueError(
                f"a fixed substep of {nominal!r} fm/c does not divide the record spacing "
                f"tau_grid[{k}] - tau_grid[{k - 1}] = {rec!r} fm/c: the ratio is {ratio!r}, not "
                f"an integer, so {what}.  d_tau u is a difference of recovered velocities "
                f"divided by the step, which is what that destroys.  Snap the tau grid, or pass "
                f"hydro_dtau=None for adaptive CFL.")
        counts.append(int(n))
    return counts


def _diagnosis(has_source, dtau=None, nominal=None):
    """The things that actually cause a divergence, named for whichever applies.

    The substep branch goes first and is not a guess: a stub step is a mechanism, and the
    freeze-out text below is confidently wrong about it.  Acting on that text -- shortening
    tau_end -- makes the symptom disappear without touching the cause, which is the worst
    possible outcome.
    """
    if dtau is not None and nominal is not None and 0.0 < dtau < _STUB * nominal:
        return (f"The last substep before this frame was {dtau:.1e} fm/c against a nominal "
                f"{nominal:.4g} -- a record grid that does not divide the substep leaves a "
                f"residual step, and d_tau u is a difference of recovered velocities divided by "
                f"it, which destroys the Israel-Stewart source.  Snap the tau grid, or pass "
                f"hydro_dtau=None for adaptive CFL.")
    if has_source:
        return ("The usual cause is a jet deposit landing in a nearly empty cell, where the "
                "Landau match has no subluminal solution. Reduce the parton energy, place it "
                "inside the fireball (source.placement_weight: energy_weighted), or refine the "
                "grid.")
    return ("With no source attached this is almost always a fireball that was evolved far past "
            "freeze-out, where the grid is essentially vacuum and the scheme is not stable. Keep "
            "output.stop_at_freezeout on (those frames are zeroed anyway), or shorten "
            "time.tau_end.")


def _check_frame(frame, k, tau, has_source=False, dtau=None, nominal=None):
    """Refuse to write a frame that is not physical.

    Writing NaN or a superluminal velocity into a training file is worse than failing: the file
    loads, trains, and quietly poisons the model.

    `dtau`/`nominal` are the last substep taken before this frame and the step that was asked
    for; they only steer the diagnosis text.
    """
    why = _diagnosis(has_source, dtau, nominal)
    if not np.isfinite(frame).all():
        n = int((~np.isfinite(frame)).sum())
        raise EvolutionDiverged(
            f"frame {k} (tau = {tau:.3f}) has {n} non-finite value(s). {why}")
    vmax = float(np.abs(frame[1:4]).max())
    if vmax > 1.0:
        raise EvolutionDiverged(
            f"frame {k} (tau = {tau:.3f}) has |v| = {vmax:.4f} > 1: the energy-momentum tensor "
            f"went spacelike. {why}")
    if float(np.abs(frame).max()) > _F32_MAX:
        raise EvolutionDiverged(
            f"frame {k} (tau = {tau:.3f}) exceeds the float32 range and cannot be stored. {why}")


def evolve_event(q, pi, Pi, tau_grid, grid, eos, *, transport=None, source=None,
                 cfl=0.4, hydro_dtau=None, dtau_max=None, slope_fn=None, speed="local",
                 out=None, src_out=None, T_fo=0.150, freezeout="max_T",
                 zero_tail=True, stop_at_freezeout=True, progress=None):
    """Evolve one event over `tau_grid` (frame 0 is the initial state) into float32 buffers.

    out     : (4, nx, ny, neta, ntau) float32 -- [e, vx, vy, vz]
    src_out : the same shape, or None -- Delta q accumulated per record, contravariant Milne

    A fixed `hydro_dtau` must divide every record spacing in `tau_grid`, and that is checked
    against the floats themselves before the first step rather than assumed from the config --
    an axis read back out of a float32 store does not arrive exact.  Crumb-sized residuals are
    absorbed into the preceding step; anything larger is a ValueError.  Both are
    `_substeps_per_frame`, which says why the residual matters.

    `stop_at_freezeout` ends the loop once the fireball has cooled below T_fo everywhere (and,
    with a source, once every droplet has fired).  The frames past freeze-out are zeroed in any
    case, so continuing to step is pure waste -- and it is not harmless waste: a fireball evolved
    far past freeze-out is essentially vacuum on the grid, and the scheme becomes unstable there.
    A central Au+Au event freezes out around tau = 4 fm and diverges around tau = 9 if it is
    made to keep going.

    Returns a diagnostics dict, including the freeze-out index the writer needs.
    """
    slope_fn = slope_fn or _fv.minmod_slope
    tau_grid = [float(t) for t in tau_grid]
    T = len(tau_grid)
    nx, ny, nz = grid.nx, grid.ny, grid.neta
    if out is None:
        out = np.zeros((4, nx, ny, nz, T), dtype=np.float32)
    if src_out is None and source is not None:
        src_out = np.zeros((4, nx, ny, nz, T), dtype=np.float32)

    t0 = time.perf_counter()
    tau = tau_grid[0]
    dudtau = None
    T_max = np.zeros(T)
    P_cart = np.zeros((T, 4))
    n_steps = 0
    dtau_seen = []
    last_step = (None, None)          # (substep taken, substep asked for) -- for _diagnosis
    stopped_early = None

    if hydro_dtau is not None and T > 1:
        # On the actual floats, not on the config: derived_tau_grid builds an exact axis, but
        # an axis read back out of a file -- MUSIC's float32 native store, a downsampled
        # dataset, an `f4` attribute -- does not arrive exact, and that is the direction this
        # keeps going.  Cheap, once, and it names both numbers.
        _substeps_per_frame(tau_grid, min(hydro_dtau, dtau_max) if dtau_max else hydro_dtau)

    has_source = source is not None
    frame, temp = q_to_fno_frame(q, pi, Pi, tau, grid, eos)
    f0 = frame[0].detach().cpu().numpy()
    _check_frame(f0, 0, tau, has_source)
    out[..., 0] = f0.astype(np.float32)
    T_max[0] = float(temp.max())

    src_acc = torch.zeros_like(q) if source is not None else None
    P_acc = np.zeros(4)

    for k in range(1, T):
        tau_rec = tau_grid[k]
        while tau < tau_rec - 1e-12:
            nominal = hydro_dtau if hydro_dtau is not None else grid.max_dtau(tau, cfl)
            if dtau_max is not None:
                nominal = min(nominal, dtau_max)
            rem = tau_rec - tau
            dtau = min(nominal, rem)
            if 0.0 < rem - dtau < _ABSORB * nominal:
                # The record boundary is one crumb past the end of this step.  Take one
                # imperceptibly long step instead of leaving a ~0 one to be taken next
                # iteration: d_tau u is a difference of recovered velocities divided by the
                # step, so a residual is not a small step, it is an amplifier.  The overshoot
                # is bounded by _ABSORB of the step, i.e. nothing -- including against the CFL
                # bound on the adaptive path.
                dtau = rem
            q, pi, Pi, dudtau = _fv.strang_step(q, pi, Pi, tau, dtau, grid, eos, transport,
                                                slope_fn, source, dudtau, speed)
            if source is not None and source.last_dq is not None:
                src_acc = src_acc + source.last_dq
                P_acc += _fv.cartesian_four_momentum(source.last_dq, tau + dtau, grid)[0] \
                    .detach().cpu().numpy()
                if hasattr(source, "frame_index"):
                    source.frame_index[source.fired & (source.frame_index < 0)] = k
            tau += dtau
            n_steps += 1
            dtau_seen.append(dtau)
            last_step = (dtau, nominal)

        frame, temp = q_to_fno_frame(q, pi, Pi, tau, grid, eos)
        f = frame[0].detach().cpu().numpy()
        _check_frame(f, k, tau, has_source, *last_step)
        out[..., k] = f.astype(np.float32)
        T_max[k] = float(temp.max()) if freezeout != "central_T" else \
            float(temp[0, nx // 2, ny // 2, nz // 2])
        if src_out is not None:
            src_out[..., k] = src_acc[0].detach().cpu().numpy().astype(np.float32)
            P_cart[k] = P_acc
            src_acc = torch.zeros_like(q)
            P_acc = np.zeros(4)
        if progress is not None:
            progress(k, T, tau)

        if stop_at_freezeout and T_max[k] < T_fo and freezeout != "never":
            # Everything from here on is zeroed anyway.  Wait for any droplet still to come,
            # since a deposit can reheat the medium after the background has frozen out.
            pending = bool(has_source and hasattr(source, "fired") and not source.fired.all())
            if not pending:
                stopped_early = k
                break

    if source is not None and hasattr(source, "finalize"):
        source.finalize(tau_end=tau_grid[-1])

    ntau_fo = T if freezeout == "never" else freezeout_index(T_max, T_fo)
    tau_fo = tau_grid[min(ntau_fo, T) - 1]
    if zero_tail and freezeout != "never":
        zero_after_freezeout(out, ntau_fo, src_out)

    vmax = float(np.abs(out[1:4, ..., :max(ntau_fo, 1)]).max()) if out.size else 0.0
    return {
        "ntau_freezeout": int(ntau_fo),
        "tau_freezeout": float(tau_fo),
        "frozen_out": bool(ntau_fo < T),
        "T_max": T_max,
        "P_cart": P_cart,
        "n_steps": int(n_steps),
        "dtau_min": float(min(dtau_seen)) if dtau_seen else 0.0,
        "dtau_max": float(max(dtau_seen)) if dtau_seen else 0.0,
        "v_max": vmax,
        "stopped_early_at": stopped_early,
        "frames_skipped": (T - 1 - stopped_early) if stopped_early is not None else 0,
        "wall_s": time.perf_counter() - t0,
        "arr": out,
        "src": src_out,
    }
