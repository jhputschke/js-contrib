"""`FastHydro` -- the structure-preserving 3+1D Milne FV solver as a JETSCAPE hydro module.

The solver itself (`fast_data.fv` / `fast_data.evolve`) is used unmodified.  This class only
moves data across the framework boundary:

    IC (GeV/fm^3)  ->  fv.initial_state_from_energy  ->  evolve_event  ->  bulk_info

Two things here are load-bearing and easy to get wrong.

**Clear().**  `set_preserve_bulk_info(True)` looks like the right way to keep the evolution
readable after `Exec()`, but `PyFluidDynamics::Clear()` early-returns when it is set and so
skips `FluidDynamics::ClearTask()` -- which is the *only* code in the framework that empties
the liquefier droplet list (`FluidDynamics.cc:175-181`).  Droplets would then accumulate from
one event to the next, which looks like a physics result rather than a bug.  So the flag stays
off and `Clear()` is defined here instead, doing the one thing that matters.

**boost_invariant.**  It must be False.  `EvolutionHistory::get_tz` rewrites ``vz = z/t`` for a
boost-invariant history (`FluidEvolutionHistory.cc:405-410`), which would silently replace the
solver's longitudinal flow.  The member has no default initialiser, so it is set explicitly.

The module id must not be "MUSIC" or "Brick": `FluidDynamics::Init()` special-cases those two
to skip pre-equilibrium and `exit(-1)`s for anything else with no pre-eq module attached
(`FluidDynamics.cc:117-124`).  So a FastHydro pipeline always carries `NullPreDynamics`, and
the id stays descriptive -- borrowing "MUSIC" would make `Init()` read `<Hydro><MUSIC>` tags
and poison every downstream `GetId()` check.
"""

from __future__ import annotations

import numpy as np

from .cells import DEFAULT_FIELDS, frame_to_cells, validate_fields
from .grid import GridSpec

__all__ = ["FastHydro"]


def _base():
    from jetscape.pyjetscape_core import FluidDynamics
    return FluidDynamics


class FastHydro(_base()):
    """A `FluidDynamics` module that evolves one event with the fast FV solver.

    Parameters
    ----------
    cfg : dict
        A resolved `fast_data` config.
    stage : int
        1 = background leg, 2 = jet leg.  Only used for the default module id and for logs.
    ic : object, optional
        The initial-state module instance.  Passing it makes the jet leg replay *exactly* the
        background leg's IC (checked by sha256) rather than re-deriving it.
    source : optional
        A `CausalLiquefierSource`, or None.  Usually set later by `DropletBridge`.
    store : {"vector", "aos"}
        ``vector`` (default) writes `bulk_info.data_vector`: 4 bytes per stored field per
        cell.  ``aos`` writes `bulk_info.data`: 112 bytes per cell regardless.  At
        65x65x33x100 that is 390 MB against 1.56 GB.
    """

    def __init__(self, cfg, *, stage=1, module_id=None, ic=None, source=None,
                 device=None, dtype=None, store=None, fields=None,
                 keep_arr=True, verbose=True):
        super().__init__()
        self.cfg = cfg
        # fasthydro-only settings; see fasthydro/config.py for why they are a separate block
        _fh = (cfg.get("fasthydro") or {}).get("hydro") or {}
        store = store if store is not None else _fh.get("store", "vector")
        fields = fields if fields is not None else (_fh.get("store_fields") or DEFAULT_FIELDS)
        self.stage = int(stage)
        self.SetId(module_id or f"FastHydro_{stage}")
        self.ic = ic
        self._source = source
        self.store = store
        self.fields = validate_fields(fields)
        self.keep_arr = bool(keep_arr)
        self.verbose = bool(verbose)
        self._device = device or cfg["run"]["device"]
        self._dtype_name = dtype or cfg["run"]["dtype"]

        # Results, filled by EvolveHydro()
        self.arr = None          # (4, nx, ny, neta, ntau) float32 [e, vx, vy, vz]
        self.src = None          # the same shape, or None
        self.diag = {}
        self.ic_sha256 = None

        if store not in ("vector", "aos"):
            raise ValueError(f"store must be 'vector' or 'aos', got {store!r}")
        # NOT set_preserve_bulk_info(True) -- see the module docstring.

    # ------------------------------------------------------------------ source
    def set_source(self, source):
        """Attach (or clear, with None) the jet source for this leg."""
        self._source = source
        return self

    # ------------------------------------------------------------------ framework hooks
    def InitializeHydro(self, params):
        import torch

        from fast_data import fv
        from fast_data.eos import resolve_eos

        self.g = GridSpec.from_cfg(self.cfg)
        self._torch_dtype = getattr(torch, self._dtype_name)
        if str(self._device) == "mps" and self._torch_dtype is torch.float64:
            raise ValueError("device 'mps' has no float64; set run.dtype: float32")

        self._np_eos, self._eos = resolve_eos(
            self.cfg["eos"], device=self._device, dtype=self._torch_dtype)
        self._fvgrid = self.g.to_fv_grid(self._device, self._torch_dtype)

        tr = self.cfg["transport"]
        if str(tr["mode"]).lower() in ("israel_stewart", "viscous", "is"):
            self._transport = fv.Transport(
                eta_over_s=tr["eta_over_s"], zeta_over_s=tr["zeta_over_s"],
                tau_pi_coeff=tr["tau_pi_coeff"], delta_pipi=tr["delta_pipi"],
                delta_PiPi=tr["delta_PiPi"], tau_min=tr["tau_min"],
                pi_rho_max=tr["pi_rho_max"], pi_e_min=tr["pi_e_min"],
                pi_advection=tr["pi_advection"])
        else:
            self._transport = None

        self._tau_grid = self.g.tau_grid()
        # hydro_tau_0 has no default initialiser in the FluidDynamics constructor.
        self.SetHydroStartTime(float(self.g.tau0))
        if self.verbose:
            print(f"[{self.GetId()}] {self.g.describe()}", flush=True)
            print(f"[{self.GetId()}] eos={self.cfg['eos']['kind']} "
                  f"transport={tr['mode']} device={self._device}/{self._dtype_name} "
                  f"store={self.store} fields={len(self.fields)}", flush=True)

    def EvolveHydro(self):
        import torch

        from fast_data import evolve, fv

        e0 = self._read_ic()
        self.ic_sha256 = __import__("hashlib").sha256(
            np.ascontiguousarray(e0, dtype=np.float64).tobytes()).hexdigest()

        q = fv.initial_state_from_energy(
            torch.as_tensor(e0, device=self._fvgrid.device,
                            dtype=self._fvgrid.dtype)[None], self.g.tau0, self._eos)

        # The viscous sector is switched on by pi/Pi EXISTING, not by transport alone:
        # strang_step does `viscous = transport is not None and pi is not None`, so passing
        # pi=None runs ideal however transport is configured -- silently. fast_data's own
        # driver allocates them the same way (driver.py:177-178).
        pi = self._fvgrid.zeros(1, 10) if self._transport is not None else None
        Pi = self._fvgrid.zeros(1, 1) if self._transport is not None else None

        src = None
        if self._source is not None:
            # A fresh accounting per event; the patch cache is shared by clone().
            src = self._source.clone().reset()

        t = self.cfg["time"]
        o = self.cfg["output"]
        shape = (4, self.g.nx, self.g.ny, self.g.neta, self.g.ntau)
        out = np.zeros(shape, dtype=np.float32)
        src_out = np.zeros(shape, dtype=np.float32) if src is not None else None

        self.diag = evolve.evolve_event(
            q, pi, Pi, self._tau_grid, self._fvgrid, self._eos,
            transport=self._transport, source=src,
            cfl=t["cfl"], hydro_dtau=t["hydro_dtau"], dtau_max=t["dtau_max"],
            out=out, src_out=src_out,
            T_fo=o["T_fo"], freezeout=o["freezeout"],
            zero_tail=o["zero_after_freezeout"], stop_at_freezeout=o["stop_at_freezeout"])
        self.diag["ic_sha256"] = self.ic_sha256
        self._live_source = src

        # ── hand the evolution to the framework ────────────────────────────────
        self.set_hydro_grid_info(
            tau_min=float(self.g.tau0), dtau=float(self.g.record_dtau), ntau=int(self.g.ntau),
            x_min=float(self.g.x_min), dx=float(self.g.dx), nx=int(self.g.nx),
            y_min=float(self.g.y_min), dy=float(self.g.dy), ny=int(self.g.ny),
            eta_min=float(self.g.eta_min), deta=float(self.g.deta), neta=int(self.g.neta),
            boost_inv=False,        # get_tz would otherwise overwrite vz with z/t
            tau_eta_is_tz=False)    # the store is Milne

        cells = frame_to_cells(out, self._eos, self.fields)
        if self.store == "vector":
            self.store_fluid_cells_from_numpy_3d(cells, list(self.fields))
        else:
            self.store_fluid_cells_aos_3d(cells, list(self.fields))
        self.set_hydro_status_finished()
        self.reset_out_of_range_count()

        self.arr = out if self.keep_arr else None
        self.src = src_out
        if self.verbose:
            d = self.diag
            print(f"[{self.GetId()}] {d.get('n_steps', '?')} steps, "
                  f"tau_fo={d.get('tau_freezeout', float('nan')):.3f} fm/c "
                  f"(frame {d.get('ntau_freezeout', '?')}), "
                  f"wall={d.get('wall_s', float('nan')):.1f}s"
                  + (f", droplets fired={int(np.sum(src.fired))}/{len(src.droplets)}"
                     if src is not None else ""), flush=True)

    def Clear(self):
        # Deliberately not chaining to FluidDynamics::Clear(): that calls
        # clear_up_evolution_data() and the writer still needs bulk_info.  But DO reproduce
        # the one other thing ClearTask() does -- emptying the droplet list, which nothing
        # else in the framework does (FluidDynamics.cc:175-181).
        self.clearSurfaceCellVector()
        liq = self.get_liquefier()
        if liq is not None:
            liq.ClearTask()

    # ------------------------------------------------------------------ the IC
    def _read_ic(self) -> np.ndarray:
        """(nx, ny, neta) float64 energy density [GeV/fm^3].

        Route 1 (the `ic=` object) is what lets the jet leg replay the background leg's event
        exactly.  Route 2 goes through pre-equilibrium, the framework-native path.  Route 3 is
        the raw initial state.
        """
        g = self.g
        want = (g.nx, g.ny, g.neta)

        if self.ic is not None and getattr(self.ic, "e0", None) is not None:
            e0 = np.ascontiguousarray(self.ic.e0, dtype=np.float64)
            if e0.shape != want:
                raise ValueError(f"IC object grid {e0.shape} != hydro grid {want}")
            return e0

        preeq = self.get_preeq_pointer()
        if preeq is not None:
            e = np.asarray(preeq.get_e_numpy(), dtype=np.float64)
            if e.size == g.nx * g.ny * g.neta:
                self._check_preeq_flow(preeq)
                return np.ascontiguousarray(e.reshape(want))
            if e.size:
                raise ValueError(
                    f"pre-equilibrium returned {e.size} cells, expected "
                    f"nx*ny*neta = {g.nx * g.ny * g.neta} for grid {want}")

        ini = self.get_ini_pointer()
        if ini is not None:
            e = np.asarray(ini.get_entropy_density_numpy_3d(), dtype=np.float64)
            if e.shape == want:
                return np.ascontiguousarray(e)
            raise ValueError(f"initial state grid {e.shape} != hydro grid {want}")

        raise RuntimeError(
            f"{self.GetId()}: no initial condition. Pass ic=<InitialState> or put an "
            "InitialState + PreequilibriumDynamics ahead of this module in the pipeline.")

    def _check_preeq_flow(self, preeq):
        """Refuse to silently discard pre-equilibrium flow and viscous stress.

        `fv.initial_state_from_energy` starts from ``u = (1,0,0,0)``, ``pi = Pi = 0``, which is
        exactly what `NullPreDynamics` produces -- but not what FreestreamMilne produces.
        """
        if ((self.cfg.get("fasthydro") or {}).get("hydro") or {}).get("accept_preeq_flow_loss"):
            return
        worst, name = 0.0, None
        for field in ("ux", "uy", "ueta", "pi00", "pi01", "pi02", "pi11", "pi12", "pi22"):
            getter = getattr(preeq, f"get_{field}_numpy", None)
            if getter is None:
                continue
            v = np.asarray(getter())
            if v.size:
                m = float(np.max(np.abs(v)))
                if m > worst:
                    worst, name = m, field
        if worst > 1e-8:
            raise ValueError(
                f"{self.GetId()}: pre-equilibrium carries non-trivial flow/viscous stress "
                f"(max |{name}| = {worst:.3e}), but the FV solver is initialised from energy "
                "density alone with u = (1,0,0,0), pi = Pi = 0 -- that information would be "
                "silently discarded. Use NullPreDynamics, or set "
                "fasthydro.hydro.accept_preeq_flow_loss: true to proceed anyway.")
