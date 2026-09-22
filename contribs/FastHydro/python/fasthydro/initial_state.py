"""JETSCAPE `InitialState` modules backed by fast_data's tilted 3D MC-Glauber.

How the initial condition crosses into the framework
----------------------------------------------------
``set_entropy_density_from_numpy`` is a misnomer.  The container it fills is simply *the* IC
grid; `NullPreDynamics::EvolvePreequilibrium` (`src/preequilibrium/NullPreDynamics.cc:36-56`)
copies `ini->GetEntropyDensityDistribution()` straight into a variable it calls
``energy_density`` and sets ``P = e/3``, ``u = (1,0,0,0)``, ``pi = Pi = 0``.  So the array
travels as **energy density in GeV/fm^3** with no ``s_factor``, no ``hbarc`` and no entropy
conversion.  The working precedent is `config/FVvsMUSIC/run_music_leg.py:105`, which passes
`glauber`'s ``e0`` through unchanged and annotates it in exactly those words.

The array is a C-order ``(nx, ny, neta)`` ravel, i.e. ``idx = (ny*neta)*ix + neta*iy + ieta``,
which is also how MUSIC's ``Initial_profile 42`` reads it -- so the same IC can be handed to a
MUSIC leg for a cross-check without any reindexing.

`Init()` is deliberately *not* overridden.  `bind_framework.cc` binds ``Init`` to
``&JetScapeModuleBase::Init``, which dispatches virtually, so a Python ``Init`` calling
``super().Init()`` would re-enter itself.  Leaving it to C++ also means ``InitialState::Init()``
reads the ``<IS><grid_*>`` tags, which ``Exec()`` then cross-checks against the YAML.
"""

from __future__ import annotations

import hashlib

import numpy as np

from .grid import GridSpec

__all__ = ["FastGlauberInitialState", "FastFileInitialState"]


def _base():
    from jetscape.pyjetscape_core import InitialState
    return InitialState


def _sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()


class _FastInitialStateMixin:
    """Shared hand-over to the framework; subclasses only provide :meth:`_make_event`."""

    def _setup(self, cfg, *, verbose=True):
        self.cfg = cfg
        self.g = GridSpec.from_cfg(cfg)
        self.verbose = verbose
        self.e0 = None            # (nx, ny, neta) float64, GeV/fm^3
        self.meta = {}
        self.ic_sha256 = None
        self.event_index = 0
        self._np_eos = None

    # ------------------------------------------------------------------ EoS
    @property
    def np_eos(self):
        """The numpy-side EoS, resolved once (the solver gets the torch twin separately)."""
        if self._np_eos is None:
            from fast_data.eos import resolve_eos
            resolved = resolve_eos(self.cfg["eos"])
            self._np_eos = resolved[0] if isinstance(resolved, tuple) else resolved
        return self._np_eos

    # ------------------------------------------------------------------ framework hooks
    def Exec(self):
        made = self._make_event()
        e0, meta = made[0], made[1]
        ev = made[2] if len(made) > 2 else None
        coll = None if ev is None else ev.get("coll")   # (Ncoll, 2) transverse vertices
        e0 = np.ascontiguousarray(e0, dtype=np.float64)
        g = self.g
        if e0.shape != (g.nx, g.ny, g.neta):
            raise ValueError(f"IC array {e0.shape} != configured grid {(g.nx, g.ny, g.neta)}")

        self.SetRanges(*g.is_ranges())
        self.SetSteps(*g.is_steps())
        got = (self.GetXSize(), self.GetYSize(), self.GetZSize())
        if got != (g.nx, g.ny, g.neta):
            raise ValueError(
                f"InitialState grid {got} != IC grid {(g.nx, g.ny, g.neta)}. "
                f"SetRanges({g.is_ranges()}) / SetSteps({g.is_steps()}) did not round-trip; "
                "GetXSize() is ceil(2*grid_max/grid_step)."
            )

        # ENERGY density, GeV/fm^3 -- see the module docstring.
        self.set_entropy_density_from_numpy(e0)
        self._set_binary_collisions(coll)

        self.e0, self.meta, self.ic_sha256 = e0, meta, _sha256(e0)
        if self.verbose:
            ijk = np.unravel_index(int(np.argmax(e0)), e0.shape)
            print(f"[{type(self).__name__}] event {self.event_index}: "
                  f"e_max={e0.max():.4f} GeV/fm^3 at {ijk}, "
                  f"Npart={meta.get('npart', '?')} Ncoll={meta.get('ncoll', '?')} "
                  f"b={meta.get('b', float('nan')):.3f} fm  ic={self.ic_sha256[:12]}",
                  flush=True)
        self.event_index += 1

    def _set_binary_collisions(self, coll):
        """Hand the Ta*Tb density to the framework so hard processes get real vertices.

        Without this, `InitialState::SampleABinaryCollisionPoint` only warns and puts every
        shower at (0,0,0) -- i.e. every jet starts at the fireball centre, which biases any
        path-length-dependent observable.

        The density is a transverse histogram of the binary-collision positions, broadcast
        over eta (the sampler only uses x and y; it returns z = 0).

        Note the half-cell offset: `CoordFromIdx` maps ix to ``-grid_max_x + ix*dx``, MUSIC's
        axis, while the energy density here is cell-centred at ``-(nx-1)/2*dx + ix*dx``.
        The histogram below is binned on the *sampler's* axis so a vertex lands where its
        density says, accepting that this is half a cell from the matching e-node.
        """
        g = self.g
        if coll is None or len(coll) == 0:
            self.ncoll_density = None
            return
        xr, yr, _ = g.is_ranges()
        edges_x = -xr + g.dx * (np.arange(g.nx + 1) - 0.5)
        edges_y = -yr + g.dy * (np.arange(g.ny + 1) - 0.5)
        h2, _, _ = np.histogram2d(np.asarray(coll)[:, 0], np.asarray(coll)[:, 1],
                                  bins=[edges_x, edges_y])
        dens = np.repeat(h2[:, :, None], g.neta, axis=2)
        if dens.sum() <= 0:
            self.ncoll_density = None
            return
        self.ncoll_density = np.ascontiguousarray(dens, dtype=np.float64)
        self.set_num_of_binary_collisions_from_numpy(self.ncoll_density)

    def Clear(self):
        # Keep e0: the jet leg replays the identical IC from this object.
        pass


class FastGlauberInitialState(_FastInitialStateMixin, _base()):
    """Tilted 3D MC-Glauber (`fast_data.glauber.make_event`) as a JETSCAPE InitialState."""

    def __init__(self, cfg, *, seed=None, verbose=True):
        super().__init__()
        self._setup(cfg, verbose=verbose)
        self.SetId("FastGlauberInitialState")
        self.master_seed = int(cfg["run"]["seed"] if seed is None else seed)
        self._K = None

    def _seed_for_event(self) -> int:
        # Independent, reproducible per-event stream; SeedSequence avoids the correlations a
        # bare seed + counter can show.
        return int(np.random.SeedSequence([self.master_seed, self.event_index])
                   .generate_state(1, dtype=np.uint32)[0])

    def _calibrated_K(self):
        if self._K is None:
            from fast_data import glauber
            ini = self.cfg["initial_state"]
            if ini.get("K") is not None:
                self._K = float(ini["K"])
            else:
                # calibrate_K evaluates the ansatz at the origin only, so it takes no grid
                # and no tau0 -- just the profile kwargs.
                if ini.get("target_T") is not None:
                    kind, target = "T", ini["target_T"]
                elif ini.get("target_e") is not None:
                    kind, target = "e", ini["target_e"]
                elif ini.get("target_s") is not None:
                    kind, target = "s", ini["target_s"]
                else:
                    raise ValueError(
                        "initial_state needs one of K, target_T, target_e or target_s")
                kw = dict(ini["profile"])
                for drop in ("gamma_k", "string_fluct", "edge"):
                    kw.pop(drop, None)          # calibrate_K strips these itself; be explicit
                self._K = float(glauber.calibrate_K(
                    target, kind=kind, quantity=ini["quantity"], eos=self.np_eos,
                    proj=ini["proj"], targ=ini["targ"], b=ini["calib_b"],
                    sigma_nn_mb=ini["sigma_nn_mb"],
                    n_events=int(ini["calib_events"] or 20),
                    seed=self.master_seed, **kw))
            if self.verbose:
                print(f"[FastGlauberInitialState] K = {self._K:.6g}", flush=True)
        return self._K

    def _make_event(self):
        from fast_data import glauber
        ini = self.cfg["initial_state"]
        return glauber.make_event(
            return_event=True,          # we also want ev["coll"] for the jet vertices
            proj=ini["proj"], targ=ini["targ"], b=ini["b"], b_max=ini["b_max"],
            sigma_nn_mb=ini["sigma_nn_mb"],
            grid=(self.g.nx, self.g.ny, self.g.neta),
            spacing=(self.g.dx, self.g.dy, self.g.deta),
            seed=self._seed_for_event(), require_collision=ini["require_collision"],
            eos=self.np_eos,
            # everything below goes through make_event's **kw into energy_density_3d;
            # tau0 is NOT one of them -- the IC is a tau0-independent profile.
            K=self._calibrated_K(), quantity=ini["quantity"], e_floor=ini["e_floor"],
            **ini["profile"])


class FastFileInitialState(_FastInitialStateMixin, _base()):
    """Replay ICs from a stage-1 `glauber.save_events` HDF5.

    This is what makes a replay run reproduce the original bit for bit across processes: the
    jet leg reads the very array the background leg used, rather than regenerating it.
    """

    def __init__(self, cfg, path, *, start_event=0, verbose=True):
        super().__init__()
        self._setup(cfg, verbose=verbose)
        self.SetId("FastFileInitialState")
        self.path = str(path)
        self.event_index = int(start_event)

    def _make_event(self):
        from fast_data import glauber
        e, meta = glauber.load_events(self.path, slice(self.event_index, self.event_index + 1))
        if e.shape[0] == 0:
            raise IndexError(f"{self.path} has no event {self.event_index}")
        scalar = {k: (v[0] if isinstance(v, np.ndarray) and v.size == 1 else v)
                  for k, v in meta.items()}
        return e[0], scalar
