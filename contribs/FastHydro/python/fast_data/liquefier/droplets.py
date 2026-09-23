"""Droplets: the objects the liquefier deposits, and how a YAML parton spec becomes them.

A droplet is a row ``(tau, x, y, eta, E, px, py, pz)``: the position is Milne, the momentum is
Cartesian, exactly as X-SCAPE's `Droplet` stores them (`LiquefierBase.h:40-84`).

**How many droplets one parton makes, in X-SCAPE.**  This is the thing that is easy to get wrong.
`LiquefierBase::add_hydro_sources` (`LiquefierBase.cc:157`) builds **one** droplet per call, and
it carries only ``dP = sum(p_in) - sum(kept p_out)`` -- the four-momentum lost in that call --
placed at the midpoint of the step's endpoints, and only when the loss exceeds
`hydro_source_abs_err`.  It is called from `JetEnergyLoss::DoExecTime`
(`JetEnergyLoss.cc:377`), which runs **once per parton per shower time step**, and that step is
``Eloss/deltaT = 0.1`` fm (`jetscape_main.xml:290`) out to ``maxT = 20`` fm.

So a single hard parton in X-SCAPE produces *tens* of droplets strung along its path -- a source
that moves -- which is why one jet there radiates a Mach cone.  One droplet never does; a point
deposit radiates a spherical blast wave.

`parton_to_droplet` below is the OTHER limit: one parton, one droplet, its whole four-momentum at
once.  That is X-SCAPE's full-absorption case, where `filter_partons` (`LiquefierBase.cc:92`)
marks a parton whose local-rest-frame energy has fallen below ``e_threshold = 2.0`` GeV as
`drop_stat` and the medium swallows it whole.  For a *traversing* jet use `trajectory:`, which
reconstructs the per-time-step loop; ``ds: 0.1`` matches X-SCAPE's `deltaT` exactly.

A replay path that ingests real X-SCAPE droplets would use the same array unchanged.

The YAML grammar, per key:

    x: 2.5                    explicit
    x: [-5.0, 5.0]            uniform on [lo, hi]
    x: {uniform: [lo, hi]}    the same, spelled out
    x: {normal: [mu, sigma]}  Gaussian
    x: {choice: [a, b, c]}    discrete uniform

`partons` itself is either one mapping or a list of mappings, so "a single parton" and "a list of
partons" are both natural.  An entry may instead carry a ``trajectory:`` block, which expands into
a *train* of droplets along a straight lightlike path -- a parton losing energy as it traverses the
medium.  That distinction is physical, not cosmetic: one droplet is a point explosion and radiates
a spherical blast wave, whereas a supersonic source moving through the fluid is what builds a Mach
cone.  See `_trajectory_rows`.  Momentum comes as Cartesian ``E, px, py, pz`` or as massless-polar
``E, phi, rapidity`` (pT = E/cosh y, which gives E^2 = px^2+py^2+pz^2 by construction); mixing the
two spellings is an error rather than a guess.

Flags are recorded, never used to silently drop a droplet: the caller decides.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["COLUMNS", "DropletFlags", "DropletArray", "parton_to_droplet",
           "partons_from_config", "sample_value", "flag_names", "trajectory_rows"]

COLUMNS = ("tau", "x", "y", "eta", "E", "px", "py", "pz")

_POS_KEYS = ("x", "y", "eta", "tau", "tau_dep")
_MOM_CART = ("E", "px", "py", "pz")
_MOM_POLAR = ("E", "phi", "rapidity")
_META_KEYS = ("n", "label", "back_to_back", "E_partner", "mass", "trajectory")
_ALL_KEYS = set(_POS_KEYS) | set(_MOM_CART) | set(_MOM_POLAR) | set(_META_KEYS)


class DropletFlags:
    """Bitmask recorded per droplet.  Nothing here drops a droplet by itself."""
    BEFORE_TAU0 = 1          #: deposits before the hydro starts
    AFTER_FREEZEOUT = 2      #: deposits at or after this event's freeze-out
    OUT_OF_GRID = 4          #: a significant part of the kernel falls outside the grid
    SUBCELL = 8              #: the whole causal support fits inside one cell -> CIC fallback
    SHELL_DOMINATED = 16     #: the thin wave front carries most of the momentum
    HOLE = 32                #: negative energy (a recoil hole)
    VACUUM = 64              #: deposited into a cell with essentially no medium
    NEVER_FIRED = 128        #: tau_dep is past the end of the evolution
    UNDER_RESOLVED = 256     #: the grid cannot resolve the deposit shape; the total
                             #: is still exact (renormalised), but the shape is not


_FLAG_NAMES = [(v, k) for k, v in vars(DropletFlags).items() if isinstance(v, int) and not k.startswith("_")]


def flag_names(flags):
    """Human-readable names for a flag bitmask, for logs and diagnostics."""
    return [name for bit, name in sorted(_FLAG_NAMES) if int(flags) & bit]


@dataclass
class DropletArray:
    """(M,8) droplets plus per-event offsets, matching the HDF5 layout."""

    data: np.ndarray                 # (M, 8) float64
    offsets: np.ndarray              # (N+1,) int64
    flags: np.ndarray = None         # (M,) uint32
    labels: tuple = ()               # (M,) str, free-form

    def __post_init__(self):
        self.data = np.atleast_2d(np.asarray(self.data, dtype=np.float64)).reshape(-1, 8)
        self.offsets = np.asarray(self.offsets, dtype=np.int64)
        if self.flags is None:
            self.flags = np.zeros(len(self.data), dtype=np.uint32)
        else:
            self.flags = np.asarray(self.flags, dtype=np.uint32)
        if self.offsets[-1] != len(self.data):
            raise ValueError(f"offsets[-1]={self.offsets[-1]} != {len(self.data)} droplets")

    def __len__(self):
        return len(self.data)

    @property
    def nevents(self):
        return len(self.offsets) - 1

    def tau_dep(self, params):
        """The tau at which each droplet deposits."""
        return self.data[:, 0] + params.tau_delay

    def event(self, i):
        """The droplets of event `i` as a fresh single-event DropletArray."""
        a, b = int(self.offsets[i]), int(self.offsets[i + 1])
        return DropletArray(self.data[a:b], np.array([0, b - a], dtype=np.int64),
                            self.flags[a:b], self.labels[a:b] if self.labels else ())

    def as_float32(self):
        """Round-trip through float32, which is what Jetscape::real does on ingest."""
        return DropletArray(self.data.astype(np.float32).astype(np.float64),
                            self.offsets, self.flags, self.labels)

    def validate(self, allow_spacelike=True):
        """Flag holes; check the momentum is not spacelike unless explicitly allowed."""
        d = self.data
        self.flags |= np.where(d[:, 4] < 0, DropletFlags.HOLE, 0).astype(np.uint32)
        m2 = d[:, 4] ** 2 - (d[:, 5] ** 2 + d[:, 6] ** 2 + d[:, 7] ** 2)
        bad = m2 < -1e-6 * np.maximum(d[:, 4] ** 2, 1.0)
        if bad.any() and not allow_spacelike:
            raise ValueError(f"{int(bad.sum())} droplet(s) have spacelike momentum (E^2 < p^2)")
        if not np.all(np.isfinite(d)):
            raise ValueError("droplet array contains non-finite entries")
        if np.any(d[:, 0] <= 0):
            raise ValueError("droplet tau must be > 0 (it is a Milne proper time)")
        return self

    @classmethod
    def empty(cls, nevents=1):
        return cls(np.zeros((0, 8)), np.zeros(nevents + 1, dtype=np.int64))

    @classmethod
    def concatenate(cls, per_event):
        """Stack per-event DropletArrays (or (M,8) arrays) into one, building the offsets."""
        arrs = [np.atleast_2d(np.asarray(getattr(a, "data", a), dtype=np.float64)).reshape(-1, 8)
                for a in per_event]
        flags = [np.asarray(getattr(a, "flags", np.zeros(len(x), np.uint32)), dtype=np.uint32)
                 for a, x in zip(per_event, arrs)]
        labels = tuple(l for a in per_event for l in (getattr(a, "labels", ()) or
                                                      ("",) * len(getattr(a, "data", a))))
        offs = np.cumsum([0] + [len(a) for a in arrs]).astype(np.int64)
        data = np.concatenate(arrs, 0) if arrs else np.zeros((0, 8))
        fl = np.concatenate(flags, 0) if flags else np.zeros(0, np.uint32)
        return cls(data, offs, fl, labels)


# --------------------------------------------------------------------------- sampling

def sample_value(spec, rng, key="value"):
    """Resolve one YAML field: a scalar is explicit, a 2-list is a uniform range."""
    if spec is None:
        raise ValueError(f"'{key}' is required")
    if isinstance(spec, (int, float, np.floating, np.integer)) and not isinstance(spec, bool):
        return float(spec)
    if isinstance(spec, (list, tuple)):
        if len(spec) != 2:
            raise ValueError(f"'{key}': a list means a uniform range [lo, hi] and must have 2 "
                             f"entries (got {len(spec)}: {spec!r}); use {{choice: [...]}} for a "
                             f"discrete draw")
        lo, hi = float(spec[0]), float(spec[1])
        if hi < lo:
            raise ValueError(f"'{key}': range [lo, hi] has hi < lo ({spec!r})")
        return float(rng.uniform(lo, hi))
    if isinstance(spec, dict):
        if len(spec) != 1:
            raise ValueError(f"'{key}': a distribution mapping takes exactly one of "
                             f"uniform/normal/choice (got {sorted(spec)})")
        (kind, arg), = spec.items()
        if kind == "uniform":
            return float(rng.uniform(float(arg[0]), float(arg[1])))
        if kind == "normal":
            return float(rng.normal(float(arg[0]), float(arg[1])))
        if kind == "choice":
            return float(rng.choice(np.asarray(arg, dtype=np.float64)))
        raise ValueError(f"'{key}': unknown distribution {kind!r}; expected uniform|normal|choice")
    raise ValueError(f"'{key}': cannot interpret {spec!r} as a scalar, a [lo, hi] range or a "
                     f"{{uniform|normal|choice: ...}} mapping")


def parton_to_droplet(spec, rng, params, *, tau_window=None):
    """One resolved parton spec -> one droplet row (8,) plus its label.

    `tau_window` is (tau_lo, tau_hi) for the DEPOSIT time, used by `tau: auto`.
    """
    unknown = set(spec) - _ALL_KEYS
    if unknown:
        raise ValueError(f"unknown parton key(s) {sorted(unknown)}; known: {sorted(_ALL_KEYS)}")

    has_cart = any(k in spec for k in ("px", "py", "pz"))
    has_polar = any(k in spec for k in ("phi", "rapidity"))
    if has_cart and has_polar:
        raise ValueError("give the momentum either as (E, px, py, pz) or as "
                         "(E, phi, rapidity), not a mix of both")
    if "E" not in spec:
        raise ValueError("a parton needs 'E' (the energy transferred to the medium, GeV)")

    E = sample_value(spec["E"], rng, "E")
    if has_polar:
        phi = sample_value(spec.get("phi", 0.0), rng, "phi")
        yrap = sample_value(spec.get("rapidity", 0.0), rng, "rapidity")
        pT = E / np.cosh(yrap)                 # massless: E = pT cosh(y)
        px, py, pz = pT * np.cos(phi), pT * np.sin(phi), pT * np.sinh(yrap)
    else:
        px = sample_value(spec.get("px", 0.0), rng, "px")
        py = sample_value(spec.get("py", 0.0), rng, "py")
        pz = sample_value(spec.get("pz", 0.0), rng, "pz")

    x = sample_value(spec.get("x", 0.0), rng, "x")
    y = sample_value(spec.get("y", 0.0), rng, "y")
    eta = sample_value(spec.get("eta", 0.0), rng, "eta")

    # tau may be given directly, as a deposit time, or as `auto` (sample inside the fireball).
    if "tau_dep" in spec:
        tau = sample_value(spec["tau_dep"], rng, "tau_dep") - params.tau_delay
    elif spec.get("tau", None) == "auto":
        if tau_window is None:
            raise ValueError("tau: auto needs a deposit window; the caller must supply the "
                             "event's tau0 and freeze-out time")
        lo, hi = tau_window
        if hi <= lo:
            raise ValueError(f"tau: auto has an empty deposit window [{lo:.3f}, {hi:.3f}] -- the "
                             f"event freezes out before tau_delay={params.tau_delay} elapses; "
                             f"lower source.params.tau_delay or use an explicit tau")
        tau = float(rng.uniform(lo, hi)) - params.tau_delay
    else:
        tau = sample_value(spec.get("tau", 0.6), rng, "tau")

    return np.array([tau, x, y, eta, E, px, py, pz], dtype=np.float64), str(spec.get("label", ""))


_TRAJ_KEYS = ("x", "y", "eta", "tau", "phi", "rapidity", "dEdx", "E", "ds", "n", "length")


def _resolve_trajectory(traj, rng):
    """Draw every trajectory field once, so both legs of a back-to-back pair share a vertex."""
    unknown = set(traj) - set(_TRAJ_KEYS)
    if unknown:
        raise ValueError(f"unknown trajectory key(s) {sorted(unknown)}; "
                         f"known: {sorted(_TRAJ_KEYS)}")
    if traj.get("tau", None) == "auto":
        raise ValueError("trajectory.tau is the production time of the leading parton and must "
                         "be explicit; `auto` places a single droplet, not a path")
    r = {k: sample_value(traj.get(k, d), rng, f"trajectory.{k}")
         for k, d in (("x", 0.0), ("y", 0.0), ("eta", 0.0), ("tau", 0.6),
                      ("phi", 0.0), ("rapidity", 0.0), ("ds", 0.25))}
    if r["ds"] <= 0:
        raise ValueError(f"trajectory.ds must be > 0 (got {r['ds']})")
    if r["tau"] <= 0:
        raise ValueError(f"trajectory.tau must be > 0 (got {r['tau']})")

    if ("n" in traj) == ("length" in traj):
        raise ValueError("a trajectory needs exactly one of 'n' (number of deposits) or "
                         "'length' (path length in fm); ds sets the spacing")
    if "n" in traj:
        r["n"] = int(round(sample_value(traj["n"], rng, "trajectory.n")))
    else:
        r["n"] = max(int(round(sample_value(traj["length"], rng, "trajectory.length") / r["ds"])), 1)
    if r["n"] < 1:
        raise ValueError(f"a trajectory needs at least one deposit (got n={r['n']})")

    if ("dEdx" in traj) == ("E" in traj):
        raise ValueError("a trajectory needs exactly one of 'dEdx' (GeV per fm of path) or "
                         "'E' (total GeV, split evenly over the deposits)")
    if "dEdx" in traj:
        r["dE"] = sample_value(traj["dEdx"], rng, "trajectory.dEdx") * r["ds"]
    else:
        r["dE"] = sample_value(traj["E"], rng, "trajectory.E") / r["n"]
    return r


def trajectory_rows(res, flip=False):
    """A resolved trajectory -> (n, 8) droplet rows along a straight lightlike path.

    The parton is produced at Milne ``(tau, x, y, eta)`` and travels at the speed of light in the
    direction ``(phi, rapidity)``.  Deposit k sits where the parton was after ``k*ds`` of lab time,
    which for a massless parton is also ``k*ds`` of path length, and carries the lightlike
    four-momentum ``dE * (1, v)`` it lost there -- energy AND the forward momentum kick, which is
    what drives the wake.  `flip` gives the recoil partner: the same vertex, the opposite
    direction (``phi + pi``, ``rapidity -> -rapidity``).

    Each row is a PRODUCTION point; the liquefier deposits it ``tau_delay`` later at that same
    spatial point, so the source pattern trails the parton but still sweeps at the speed of light.
    """
    phi = res["phi"] + (np.pi if flip else 0.0)
    yrap = -res["rapidity"] if flip else res["rapidity"]
    vx = np.cos(phi) / np.cosh(yrap)
    vy = np.sin(phi) / np.cosh(yrap)
    vz = np.tanh(yrap)                                  # |v| = 1 exactly

    t0 = res["tau"] * np.cosh(res["eta"])
    z0 = res["tau"] * np.sinh(res["eta"])
    s = np.arange(res["n"], dtype=np.float64) * res["ds"]

    t, z = t0 + s, z0 + vz * s
    tau = np.sqrt(np.maximum(t * t - z * z, 1e-12))
    eta = np.arctanh(np.clip(z / t, -1 + 1e-15, 1 - 1e-15))
    dE = np.full(res["n"], res["dE"])
    return np.column_stack([tau, res["x"] + vx * s, res["y"] + vy * s, eta,
                            dE, dE * vx, dE * vy, dE * vz])


def partons_from_config(spec, rng, params, *, tau_window=None):
    """The `source.partons` block -> a single-event DropletArray.

    `spec` is one mapping or a list of mappings.  `n:` repeats an entry, each copy drawing its own
    random values; `back_to_back: true` adds a partner at phi + pi with the same position.
    """
    if spec is None:
        return DropletArray.empty(1)
    entries = [spec] if isinstance(spec, dict) else list(spec)
    rows, labels = [], []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"each parton must be a mapping, got {type(entry).__name__}: {entry!r}")
        if "trajectory" in entry:
            stray = set(entry) - {"trajectory", "label", "back_to_back", "n"}
            if stray:
                raise ValueError(f"a trajectory entry takes only label/back_to_back/n besides "
                                 f"'trajectory:'; move {sorted(stray)} inside the trajectory block")
            label = str(entry.get("label", ""))
            for _ in range(int(entry.get("n", 1))):
                res = _resolve_trajectory(entry["trajectory"], rng)
                legs = [trajectory_rows(res)]
                if entry.get("back_to_back", False):
                    legs.append(trajectory_rows(res, flip=True))
                for leg, suffix in zip(legs, ("", "_away")):
                    rows.extend(leg)
                    labels.extend([(label + suffix) if label else suffix.lstrip("_")] * len(leg))
            continue

        n_spec = entry.get("n", 1)
        n = int(sample_value(n_spec, rng, "n")) if not isinstance(n_spec, int) else n_spec
        if n < 0:
            raise ValueError(f"parton 'n' must be >= 0 (got {n})")
        for _ in range(n):
            row, label = parton_to_droplet(entry, rng, params, tau_window=tau_window)
            rows.append(row)
            labels.append(label)
            if entry.get("back_to_back", False):
                away = row.copy()
                away[5:8] *= -1.0                       # flip the momentum, keep the position
                if "E_partner" in entry:
                    Ep = sample_value(entry["E_partner"], rng, "E_partner")
                    scale = Ep / row[4] if row[4] != 0 else 1.0
                    away[4] = Ep
                    away[5:8] *= scale
                rows.append(away)
                labels.append((label + "_away") if label else "away")
    if not rows:
        return DropletArray.empty(1)
    data = np.stack(rows, 0)
    return DropletArray(data, np.array([0, len(data)], dtype=np.int64),
                        labels=tuple(labels)).validate()
