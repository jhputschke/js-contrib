"""Conserved hydro state -> the FNO4d training channels, plus the freeze-out policy.

The single subtlety, and the one worth testing hardest: the FV solver carries the CONTRAVARIANT
Milne component u^eta (units 1/fm), while MUSIC -- and therefore the training data -- is written
in terms of the ORTHONORMAL Milne component ueta = tau * u^eta, from which the Cartesian lab
three-velocities follow.  Getting the factor of tau wrong yields data that is finite, smooth and
has |v| < 1 everywhere, and is simply wrong; `tests/test_fast_data_convert.py` pins it down with a
fluid at rest in the lab at eta != 0, which must give v = 0 exactly.

Reference (X-SCAPE src/root/FastRootBulkWriter.cc:146-148 via
external_packages/music4gpu/src/HydroinfoMUSIC.cpp:316-329):

    utau = sqrt(1 + ux^2 + uy^2 + ueta^2)
    uz   = utau*sinh(eta) + ueta*cosh(eta)
    ut   = utau*cosh(eta) + ueta*sinh(eta)
    vx, vy, vz = ux/ut, uy/ut, uz/ut
"""

from __future__ import annotations

import numpy as np
import torch

from . import fv as _fv

__all__ = ["q_to_fno_frame", "freezeout_index", "zero_after_freezeout", "cartesian_P_of_source"]


def q_to_fno_frame(q, pi, Pi, tau, grid, eos, prim=None):
    """-> (frame (B,4,X,Y,Z) = [e, vx, vy, vz], T (B,X,Y,Z)).

    `frame` is in the FNO4d channel order with Cartesian LAB three-velocities; `T` is the
    temperature in GeV, used for the freeze-out test.
    """
    if prim is None:
        prim = _fv.primitive_recovery(q, pi, Pi, tau, eos)
    e = prim["e"]
    ux, uy = prim["u_x"], prim["u_y"]
    ueta = tau * prim["u_eta"]                 # contravariant u^eta -> orthonormal
    # Recompute utau from the spatial components rather than reusing prim["u_tau"], exactly as
    # HydroinfoMUSIC.cpp:320 does.  They agree to round-off in normal cells, but when the Landau
    # match hits its velocity cap (V_CAP) the cached u_tau is no longer consistent with the
    # spatial parts, and the ut below can then collapse towards zero and hand back |v| >> 1.
    # Rebuilding it makes the four-velocity internally consistent by construction, so
    # ut >= |ueta|(cosh - |sinh|) > 0 and |v| <= 1 always.
    utau = torch.sqrt(1.0 + ux * ux + uy * uy + ueta * ueta)

    eta = grid.eta.view(1, 1, 1, -1)           # (1,1,1,Z), broadcasting over (B,X,Y,Z)
    ch, sh = torch.cosh(eta), torch.sinh(eta)
    ut = utau * ch + ueta * sh
    uz = utau * sh + ueta * ch
    frame = torch.stack([e, ux / ut, uy / ut, uz / ut], dim=1)
    return frame, prim["T"]


def freezeout_index(T_max_per_frame, T_fo=0.150):
    """`ntau_freezeout` in the FNO4d convention, measured on data/dAu_25ev_mb.h5.

    Let `k_last` be the last frame whose peak temperature is still >= T_fo, so frames
    0..k_last are alive.  The reference files satisfy, for all 25 events,

        ntau_freezeout == (number of non-zero frames) + 1 == k_last + 2
        tau_freezeout  == tau_min + (ntau_freezeout - 1) * dtau

    i.e. `tau_freezeout` is the tau of the FIRST zeroed frame, and zeroing starts at index
    `ntau_freezeout - 1`.  Clamped to the number of frames when the event never cools below
    T_fo (then nothing is zeroed); 1 when it is already cold at tau0.

    The reduction over cells is a MAX rather than the central cell: a tilted Glauber fireball's
    hot spot is not at the origin, and `ntau_freezeout` in the reference data is a whole-fireball
    lifetime.
    """
    T = np.asarray(T_max_per_frame)
    above = np.nonzero(T >= float(T_fo))[0]
    if not above.size:
        return 1
    return int(min(above[-1] + 2, len(T)))


def zero_after_freezeout(arr, ntau_fo, *extra):
    """Zero every channel from frame `ntau_fo` onward, in place, on `arr` and any `extra` arrays.

    The solver does not stop at freeze-out -- it keeps evolving a cold fluid -- so this is a
    deliberate post-process that makes the file look like MUSIC output.  It is what
    loc_libs/data/dataset.py:live_tau_lengths reads, since that thresholds channel 0 rather than
    consulting `ntau_freezeout`.
    """
    # Frames from ntau_fo - 1 onward are dead: ntau_freezeout counts the live frames plus the
    # first dead one, which is the frame tau_freezeout points at.  When the event never cooled
    # below T_fo within the window there IS no dead frame (ntau_fo was clamped to the frame
    # count), and zeroing one would throw away good data -- so do nothing and let the caller
    # warn that the tau range was too short.
    T_frames = arr.shape[-1]
    if int(ntau_fo) >= T_frames:
        return arr
    n = max(int(ntau_fo) - 1, 0)
    if n < T_frames:
        arr[..., n:] = 0.0
        for a in extra:
            if a is not None:
                a[..., n:] = 0.0
    return arr


def cartesian_P_of_source(S_frame, tau, grid):
    """Total Cartesian (P^t, P^x, P^y, P^z) in GeV carried by one source frame.

    `S_frame` is (4,X,Y,Z) of Delta q = Delta(tau*T^{tau nu}) in contravariant Milne, so with
    dV = dx*dy*deta (the same weight fv.cartesian_four_momentum uses):

        P^t = sum dV (cosh(eta) S^tau + tau sinh(eta) S^eta)
        P^z = sum dV (sinh(eta) S^tau + tau cosh(eta) S^eta)
    """
    S = np.asarray(S_frame, dtype=np.float64)
    eta = np.asarray(grid.eta.detach().cpu(), dtype=np.float64).reshape(1, 1, -1)
    ch, sh = np.cosh(eta), np.sinh(eta)
    dV = float(grid.dx) * float(grid.dy) * float(grid.deta)
    Pt = np.sum(ch * S[0] + tau * sh * S[3]) * dV
    Pz = np.sum(sh * S[0] + tau * ch * S[3]) * dV
    return np.array([Pt, np.sum(S[1]) * dV, np.sum(S[2]) * dV, Pz], dtype=np.float64)
