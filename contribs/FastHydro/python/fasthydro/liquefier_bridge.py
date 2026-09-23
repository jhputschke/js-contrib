"""`DropletBridge` -- move droplets from the C++ liquefier into the solver's source term.

It is a pure-Python `JetScapeModuleBase` sitting in the task list between the energy-loss
manager and the jet-leg hydro, following `jetscape.fast_h5_bulk.H5BulkWriter`: no CMake, no
factory registration, just `jetscape.Add(bridge)`.

    ... -> FastHydro("bg") -> JetEnergyLossManager -> DropletBridge -> FastHydro("jet")

That position is the whole point: it runs after the showers are finished and the liquefier is
full, and before the jet leg evolves, in every event -- including under `run_manual`, where
there is no per-event Python yield point at all.

No conversion happens to the droplet rows.  `LiquefierBase::add_hydro_sources` writes
``xmu = (tau, x, y, eta)`` and ``pmu = (E, px, py, pz)`` (`LiquefierBase.cc:200-225`), which is
exactly `fast_data.liquefier.droplets.COLUMNS`.  The (M,8) array from `droplets_numpy()` is a
`DropletArray` payload as it stands.

Deposit mode
------------
The source is built with ``mode="conservative"`` by default and that is not a detail.  Point
sampling on a cell-centred grid -- what the C++ does on MUSIC's much finer grid -- loses the
deposit entirely for droplets at large ``|eta_d|``, because the causal support there can be a
fraction of a cell.  Measured on the production grid (tau_d = 1, deposit at tau = 3), the
fraction of the droplet momentum that lands:

    eta_d          0      1      2      3      4
    point (xscape) 1.000  1.009  2.561  0.000  0.000
    conservative   1.000  1.000  0.994  1.000  1.169

Real Matter+LBT showers populate exactly that range.  See README_FastData.md.
"""

from __future__ import annotations

import numpy as np

# save/load live in droplets_io so that replay -- which needs no framework -- can import them
# without dragging in pyjetscape_core (DropletBridge subclasses a bound C++ type at import
# time).  Re-exported here for convenience.
from .droplets_io import load_droplets_npz, save_droplets_npz  # noqa: F401

__all__ = ["DropletBridge", "droplets_from_liquefier", "params_from_liquefier",
           "save_droplets_npz", "load_droplets_npz"]


def _base():
    from jetscape.pyjetscape_core import JetScapeModuleBase
    return JetScapeModuleBase


def params_from_liquefier(liq):
    """`LiquefierParams` read from the live C++ object rather than duplicated in YAML."""
    from fast_data.liquefier import LiquefierParams
    p = liq.params()
    return LiquefierParams(dtau=p["dtau"], tau_delay=p["tau_delay"],
                           time_relax=p["time_relax"], d_diff=p["d_diff"],
                           width_delta=p["width_delta"])


def droplets_from_liquefier(liq):
    """The C++ droplet list as a `DropletArray` (single event). No conversion; see above."""
    from fast_data.liquefier.droplets import DropletArray
    d = np.asarray(liq.droplets_numpy(), dtype=np.float64).reshape(-1, 8)
    return DropletArray(d, np.array([0, len(d)], dtype=np.int64))


class DropletBridge(_base()):
    """Read the liquefier after energy loss; hand a source to the jet-leg hydro."""

    def __init__(self, liquefier, hydro_jet, cfg, *, manager=None, verbose=True):
        super().__init__()
        self.SetId("FastHydroDropletBridge")
        self.liq = liquefier
        self.hydro_jet = hydro_jet
        self.cfg = cfg
        self.verbose = bool(verbose)
        self.params = None
        self.droplets = None          # DropletArray of the current event
        self.history = []             # one DropletArray per event, for the npz dump
        # The energy-loss manager, for capturing the shower graph. This task's position is
        # the ONLY window where the showers exist and are finished: the manager builds them
        # in its own Exec and drops them in ClearPerEvent.
        self.manager = manager
        self.shower = None            # ShowerRecord of the current event
        self.shower_history = []

    def Init(self):
        self.params = params_from_liquefier(self.liq)

    def Exec(self):
        from fast_data.liquefier import CausalLiquefierSource

        if self.params is None:                      # Init() may not have been called
            self.params = params_from_liquefier(self.liq)

        self._capture_shower()

        da = droplets_from_liquefier(self.liq)
        self.droplets = da
        self.history.append(da)

        if len(da) == 0:
            # No deposition: the jet leg must then reproduce the background leg bit for bit.
            self.hydro_jet.set_source(None)
            if self.verbose:
                print("[DropletBridge] 0 droplets -- jet leg is a pure background repeat",
                      flush=True)
            return

        s = self.cfg["source"]
        self.hydro_jet.set_source(CausalLiquefierSource(
            da, self.params,
            mode=s["mode"], renorm=s["renorm"], tau_eval_mode=s["tau_eval_mode"],
            n_sub=s["n_sub"], n_sub_max=s["n_sub_max"],
            min_in_grid=s["min_in_grid"], on_out_of_grid=s["on_out_of_grid"],
            device=self.cfg["run"]["device"],
            dtype=getattr(__import__("torch"), self.cfg["run"]["dtype"])))

        self._report_window(da)

    def _capture_shower(self):
        """Flatten this event's parton showers, if there is a manager and it is wanted.

        A failure here must not lose the event: the shower group is a bonus alongside the
        hydro, and a binding that predates `get_showers()` should degrade to "no shower
        group", not to "no run".
        """
        from .showers import empty_record, showers_from_manager

        if self.manager is None or not (self.cfg.get("fasthydro") or {}).get(
                "store_showers", True):
            return
        try:
            rec = showers_from_manager(self.manager)
        except Exception as exc:                                  # noqa: BLE001
            print(f"[DropletBridge] WARNING: could not capture the parton shower ({exc}); "
                  f"the file will have no shower/ group", flush=True)
            rec = empty_record()
        self.shower = rec
        self.shower_history.append(rec)
        if self.verbose and len(rec.partons):
            print(f"[DropletBridge] {rec.n_showers} shower(s), {len(rec.partons)} partons, "
                  f"{len(rec.vertices)} vertices", flush=True)

    def _report_window(self, da):
        """Say what fraction of the deposited momentum the hydro window can actually take.

        A droplet fires only if ``tau_d + tau_delay`` falls inside the solver's tau range.
        Matter and LBT happily produce droplets that deposit after the fireball has been
        evolved, and those are simply never injected -- so the jet leg would quietly carry
        less momentum than the shower gave up.  That must be visible, not silent.
        """
        g = self.hydro_jet.g
        tau_dep = da.tau_dep(self.params)
        P = da.data[:, 4:8].sum(axis=0)
        late = tau_dep > g.tau_max
        early = tau_dep < g.tau0
        self.n_late, self.n_early = int(late.sum()), int(early.sum())
        self.E_in_window = float(da.data[~(late | early), 4].sum())

        if self.verbose:
            print(f"[DropletBridge] {len(da)} droplets, E={P[0]:.3f} GeV, "
                  f"p=({P[1]:.3f},{P[2]:.3f},{P[3]:.3f}), "
                  f"tau_dep {tau_dep.min():.2f}..{tau_dep.max():.2f} fm/c "
                  f"(mode={self.cfg['source']['mode']})", flush=True)
        if self.n_late or self.n_early:
            lost = float(da.data[late | early, 4].sum())
            print(f"[DropletBridge] WARNING: {self.n_late} droplet(s) deposit after "
                  f"tau_max={g.tau_max:.2f} and {self.n_early} before tau0={g.tau0:.2f} "
                  f"fm/c -- {lost:.3f} of {P[0]:.3f} GeV ({100*lost/max(P[0], 1e-9):.1f}%) "
                  f"is never injected. Extend time.choose_ntau, or lower "
                  f"<CausalLiquefier><tau_delay>.", flush=True)

    def Clear(self):
        # The droplet list itself is cleared by FastHydro.Clear() -> liq.ClearTask(), which is
        # the only place that happens. Keep `history` for the dump.
        pass
