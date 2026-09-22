"""`CausalLiquefierSource` -- the object the FV solver calls.

It is duck-typed to the hook in `structure_preserving_hydro_fv.strang_step`:

    q_new, prim_old = ideal_step(...)
    if source is not None:
        q_new = q_new + source.step(tau, dtau, grid, prim_old)

so `step` returns Delta q directly, in the units of ``q = tau*T^{tau nu}``, contravariant Milne.
`rollout` additionally reads `.last_dq` to accumulate its `src` channel between records.

Why there is no dtau in the deposit
-----------------------------------
The C++ fires a droplet in the single hydro step whose window brackets ``tau_d + tau_delay``, with
the kernel carrying ``1/dtau_liquefier``; MUSIC then does ``q += tau * j * dtau_hydro``.  With the
two dtau equal they cancel identically, so the deposited four-momentum is

    Delta q^nu = tau_eval * <K_tau> * P^nu_orth / dV          (eta component additionally / tau)

with no dtau at all.  That is what makes an ADAPTIVE solver step harmless: the steps tile
[tau0, tau_end) and the firing window is half-open, so every droplet fires exactly once, and the
amount it deposits does not depend on how long that step happened to be.  `params.dtau` therefore
does not enter the physics here; it is kept for provenance and for `mode="xscape"`.

The kernel is evaluated at ``tau_eval = tau_dep`` by default, which also makes the deposit SHAPE
independent of the step sequence -- two runs with different CFL numbers produce identical
deposits.  `tau_eval_mode="step"` snaps to the hydro node as the C++ does.
"""

from __future__ import annotations

import warnings

import numpy as np

from .deposit import droplet_weights, milne_dq_from_cartesian
from .droplets import DropletArray, DropletFlags, flag_names
from .params import LiquefierParams

__all__ = ["CausalLiquefierSource"]


class CausalLiquefierSource:
    """Deposit a set of droplets into a hydro run via X-SCAPE's causal-diffusion kernel."""

    #: the deposit is per event; batching would need per-event droplet lists
    supports_batch = False

    def __init__(self, droplets, params: LiquefierParams, *, mode="conservative",
                 renorm="grid", tau_eval_mode="dep", n_sub="auto", n_sub_max=24,
                 shell="widen", min_in_grid=0.99, on_out_of_grid="warn",
                 device=None, dtype=None, B=1):
        if isinstance(droplets, DropletArray):
            self.droplets = droplets
        else:
            arr = np.atleast_2d(np.asarray(droplets, dtype=np.float64)).reshape(-1, 8)
            self.droplets = DropletArray(arr, np.array([0, len(arr)], dtype=np.int64))
        self.p = params
        self.mode, self.renorm, self.tau_eval_mode = mode, renorm, tau_eval_mode
        self.n_sub, self.n_sub_max, self.shell = n_sub, n_sub_max, shell
        self.min_in_grid, self.on_out_of_grid = min_in_grid, on_out_of_grid
        self.device, self.dtype, self.B = device, dtype, int(B)

        self._patches = {}
        self.last_dq = None
        self.reset()

    # ------------------------------------------------------------------ lifecycle
    def reset(self):
        """Forget which droplets have fired and all bookkeeping; keep the patch cache."""
        m = len(self.droplets)
        self.fired = np.zeros(m, dtype=bool)
        self.tau_fire = np.full(m, np.nan)
        self.tau_eval = np.full(m, np.nan)
        self.n_raw = np.full(m, np.nan)
        self.in_grid = np.full(m, np.nan)
        self.frame_index = np.full(m, -1, dtype=np.int32)
        self.flags = self.droplets.flags.copy()
        self.P_injected = np.zeros(4)
        self.P_requested = np.zeros(4)
        self.last_dq = None
        return self

    def clone(self):
        """A fresh source over the same droplets, sharing the (immutable) patch cache."""
        out = CausalLiquefierSource(
            self.droplets, self.p, mode=self.mode, renorm=self.renorm,
            tau_eval_mode=self.tau_eval_mode, n_sub=self.n_sub, n_sub_max=self.n_sub_max,
            shell=self.shell, min_in_grid=self.min_in_grid,
            on_out_of_grid=self.on_out_of_grid, device=self.device, dtype=self.dtype, B=self.B)
        out._patches = self._patches
        return out

    # ------------------------------------------------------------------ the hook
    def tau_dep(self):
        return self.droplets.tau_dep(self.p)

    def _patch(self, i, tau_eval, grid):
        key = (i, round(float(tau_eval), 12), id(grid))
        pt = self._patches.get(key)
        if pt is None:
            pt = droplet_weights(self.droplets.data[i], tau_eval, grid, self.p,
                                 mode=self.mode, n_sub=self.n_sub, n_sub_max=self.n_sub_max,
                                 shell=self.shell, renorm=self.renorm,
                                 min_in_grid=self.min_in_grid)
            self._patches[key] = pt
        return pt

    def step(self, tau, dtau, grid, prim=None):
        """-> Delta q (B,4,X,Y,Z) for the step [tau, tau+dtau).  Also sets `last_dq`."""
        import torch

        dev = self.device or grid.device
        dt = self.dtype or grid.dtype
        tau_dep = self.tau_dep()
        # half-open window: every tau_dep falls in exactly one step of the tiling
        firing = np.nonzero((~self.fired) & (tau_dep >= tau) & (tau_dep < tau + dtau))[0]
        if firing.size == 0:
            self.last_dq = None
            return torch.zeros(self.B, 4, grid.nx, grid.ny, grid.neta, device=dev, dtype=dt)

        tau_q = float(tau + dtau)          # q lives on the tau + dtau surface after ideal_step
        dq = torch.zeros(self.B, 4, grid.nx, grid.ny, grid.neta, device=dev, dtype=dt)
        for i in firing:
            if self.tau_eval_mode == "dep":
                te = float(tau_dep[i])
            elif self.tau_eval_mode == "slice":
                te = tau_q
            elif self.tau_eval_mode == "step":
                te = float(tau)
            else:
                raise ValueError(f"unknown tau_eval_mode {self.tau_eval_mode!r}")

            pt = self._patch(int(i), te, grid)
            self.fired[i] = True
            self.tau_fire[i], self.tau_eval[i] = float(tau), te
            self.n_raw[i], self.in_grid[i] = pt.n_raw, pt.in_grid_fraction
            self.flags[i] |= np.uint32(pt.flags)
            P = self.droplets.data[i, 4:8]
            self.P_requested += P
            if pt.is_empty:
                self._complain(i, pt)
                continue
            if (pt.flags & DropletFlags.OUT_OF_GRID) and self.on_out_of_grid == "drop":
                self._complain(i, pt)
                continue
            self._complain(i, pt)

            block = milne_dq_from_cartesian(pt, P, tau_q, grid)
            dq[:, :, pt.ix, pt.iy, pt.ie] += torch.as_tensor(block, device=dev, dtype=dt)
            self.P_injected += P * pt.in_grid_fraction

            if prim is not None:
                self._flag_vacuum(i, pt, P, prim, tau_q, grid)

        self.last_dq = dq
        return dq

    # ------------------------------------------------------------------ diagnostics
    def _complain(self, i, pt):
        if not (pt.flags & DropletFlags.OUT_OF_GRID):
            return
        msg = (f"droplet {i} deposits only {pt.in_grid_fraction:.3f} of its momentum inside the "
               f"grid (tau_dep={self.tau_dep()[i]:.3f}, eta_d={self.droplets.data[i, 3]:.2f})")
        if self.on_out_of_grid == "raise":
            raise ValueError(msg)
        if self.on_out_of_grid == "warn":
            warnings.warn(msg, RuntimeWarning, stacklevel=3)

    def _flag_vacuum(self, i, pt, P, prim, tau_q, grid):
        """Flag a deposit landing where there is essentially no medium: the Landau match then
        drives |v| -> 1 and the result is not trustworthy."""
        try:
            e = prim["e"]
            k = int(np.argmax(pt.w.sum(axis=(0, 1))))
            ie = pt.ie.start + k if pt.ie.start is not None else k
            e_local = float(e[..., ie].abs().max()) if hasattr(e, "abs") else float(np.max(np.abs(e)))
        except Exception:
            return
        dV = float(grid.dx) * float(grid.dy) * float(grid.deta)
        de = abs(float(P[0])) / max(tau_q * dV * max(pt.n_cells, 1), 1e-30)
        if e_local < 1e-3 or de > 10.0 * max(e_local, 1e-12):
            self.flags[i] |= DropletFlags.VACUUM

    def finalize(self, tau_end=None):
        """Mark droplets that never fired.  Call once the rollout is done."""
        never = ~self.fired
        if tau_end is not None:
            never &= self.tau_dep() >= tau_end
        self.flags[never] |= DropletFlags.NEVER_FIRED
        return self

    def report(self):
        """A dict of everything worth recording about this event's deposits."""
        return {
            "n_droplets": int(len(self.droplets)),
            "n_fired": int(self.fired.sum()),
            "n_never_fired": int((~self.fired).sum()),
            "n_flagged": int((self.flags != 0).sum()),
            "P_cart_requested": self.P_requested.copy(),
            "P_cart_injected": self.P_injected.copy(),
            "E_injected": float(self.P_injected[0]),
            # all-NaN when nothing fired; report the neutral value rather than warn
            "in_grid_min": (float(np.nanmin(self.in_grid))
                            if np.any(~np.isnan(self.in_grid)) else 1.0),
            "n_raw_max_dev": (float(np.nanmax(np.abs(self.n_raw - 1.0)))
                              if np.any(~np.isnan(self.n_raw)) else 0.0),
        }

    def droplet_table(self):
        """Per-droplet diagnostics, as columns for the HDF5 `source/` group."""
        return {
            "tau_dep": self.tau_dep(),
            "tau_fire": self.tau_fire,
            "tau_eval": self.tau_eval,
            "n_raw": self.n_raw,
            "in_grid": self.in_grid,
            "frame_index": self.frame_index,
            "flags": self.flags,
        }

    def flag_summary(self):
        """{flag name: count} over the droplets, for the run log."""
        out = {}
        for f in self.flags:
            for name in flag_names(f):
                out[name] = out.get(name, 0) + 1
        return out
