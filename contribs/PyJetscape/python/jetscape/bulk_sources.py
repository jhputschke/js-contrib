"""
python/jetscape/bulk_sources.py

Where the bulk hydro cells come from, for the HDF5 bulk writer.

Three source modes cover both C++ bulk writer modules:

===========  ==========================================  =============================
grid_mode    source                                      C++ equivalent
===========  ==========================================  =============================
native       MpiMusic.get_native_evolution_numpy()       FastRootBulkWriter, native
grid         the same, resampled onto a user grid        FastRootBulkWriter, grid
framework    EvolutionHistory.to_numpy_full(), resampled RootBulkWriter
===========  ==========================================  =============================

``native`` is exact: the numpy export walks the same MUSIC store the C++ writer walks and
applies the same ``static_cast<float>``, so the output is bit-for-bit identical.

``grid`` and ``framework`` agree with the C++ to ~1e-7 relative, not bitwise.  Nothing is
lost in the export -- ``Jetscape::real`` is ``float`` (src/framework/RealType.h:26), so
``FluidCellInfo`` is already float32 and ``to_numpy_full`` is a straight copy.  The gap is
that ``EvolutionHistory::get()`` does its own blend in float32 and in a different summation
order, while this module interpolates in float64; the numpy answer is the more accurate of
the two.  Compare with ``allclose``, never ``array_equal``.

XML prerequisites differ by mode and are mutually exclusive:

* ``native`` / ``grid``  need ``<Hydro><MUSIC><dump_hydro_only>1``, which keeps MUSIC's
  native store and skips the framework copy.
* ``framework``          needs ``output_evolution_to_memory=1`` *without* ``dump_hydro_only``,
  because it reads ``bulk_info.data``.  It works with any hydro module, not just MUSIC.

:func:`resample` reproduces ``EvolutionHistory::get()`` exactly (up to float rounding),
including its three non-obvious behaviours:

1. A point outside the source box on *any* axis yields an all-zero cell -- a hard cut, not
   a clamp (``FluidEvolutionHistory.cc:CheckInRange`` -> ``get`` returns a default cell).
2. The eta range check is skipped when the source is boost invariant, but NOT when
   ``neta == 1`` and it is not.  In that case ``EtaMax() == eta_min``, so every output eta
   other than exactly ``eta_min`` maps to zeros.  We warn rather than silently produce an
   empty file.
3. Interpolation is quadrilinear: spatial trilinear at ``id_tau`` and ``id_tau+1``, then
   linear in tau.  ``CellIndex`` clamps every index to ``[0, n-1]``, so the upper edge
   behaves like scipy's ``mode="nearest"``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace

import numpy as np

__all__ = ["Grid", "GRID_MODES", "event_array", "resample", "ntau_to_end",
           "resolve_out_grid", "attrs_from_grids", "music_extra_attrs"]

GRID_MODES = ("native", "grid", "framework")

#: bytes per FluidCellInfo in the framework AoS, used only for the memory warning
_FLUID_CELL_BYTES = 112


@dataclass(frozen=True)
class Grid:
    """A uniform (tau, x, y, eta) grid, in the same conventions as EvolutionHistory."""

    nx: int
    ny: int
    neta: int
    ntau: int
    x_min: float
    dx: float
    y_min: float
    dy: float
    eta_min: float
    deta: float
    tau_min: float
    dtau: float
    boost_invariant: bool = False

    @classmethod
    def from_bulk_info(cls, b, ntau=None, dtau=None):
        """Build from a bound EvolutionHistory (``hydro.get_bulk_info()``)."""
        return cls(
            nx=int(b.nx), ny=int(b.ny), neta=max(1, int(b.neta)),
            ntau=int(b.ntau if ntau is None else ntau),
            x_min=float(b.x_min), dx=float(b.dx),
            y_min=float(b.y_min), dy=float(b.dy),
            eta_min=float(b.eta_min), deta=float(b.deta),
            tau_min=float(b.tau_min),
            dtau=float(b.dtau if dtau is None else dtau),
            boost_invariant=bool(b.boost_invariant),
        )

    @classmethod
    def from_bounds(cls, x, y, eta, tau_min, dtau, ntau=0):
        """Build from ``(min, max, n)`` per spatial axis: n cell centres, min..max inclusive.

        The step is ``(max - min)/(n - 1)``; ``n = 1`` needs ``min == max`` and gets step 0.
        ``ntau = 0`` leaves the tau extent to :func:`resolve_out_grid`, which then runs to
        the end of each event's evolution.
        """
        def axis(name, spec):
            lo, hi, n = float(spec[0]), float(spec[1]), int(spec[2])
            if n < 1:
                raise ValueError(f"{name}: n must be >= 1, got {n}")
            if n == 1:
                if hi != lo:
                    raise ValueError(f"{name}: n = 1 needs min == max, got {lo}..{hi}")
                return lo, 0.0, 1
            if hi <= lo:
                raise ValueError(f"{name}: max must be > min, got {lo}..{hi}")
            return lo, (hi - lo) / (n - 1), n

        x_min, dx, nx = axis("x", x)
        y_min, dy, ny = axis("y", y)
        eta_min, deta, neta = axis("eta", eta)
        if dtau <= 0:
            raise ValueError(f"dtau must be > 0, got {dtau}")
        return cls(nx=nx, ny=ny, neta=neta, ntau=max(0, int(ntau)),
                   x_min=x_min, dx=dx, y_min=y_min, dy=dy,
                   eta_min=eta_min, deta=deta,
                   tau_min=float(tau_min), dtau=float(dtau))

    @property
    def tau_max(self):
        return self.tau_min + (self.ntau - 1) * self.dtau

    def axis(self, name):
        n = {"x": self.nx, "y": self.ny, "eta": self.neta, "tau": self.ntau}[name]
        lo = getattr(self, f"{name}_min")
        d = getattr(self, "dtau" if name == "tau" else f"d{name}")
        return lo + d * np.arange(n)


# --------------------------------------------------------------------- out grid
def resolve_out_grid(src, spec=None):
    """Resolve a user output-grid spec against the source grid.

    ``spec`` is a mapping of any of ``x_min dx y_min dy eta_min deta tau_min dtau ntau``.
    A missing key, ``None`` or ``0`` means "use the source value", and the transverse cell
    counts are derived as ``2*int(|min|/d) + 1`` -- both conventions taken from
    ``FastRootBulkWriter::fill_grid`` (src/root/FastRootBulkWriter.cc:182-206) so that this
    reproduces the C++ grid mode rather than inventing a second convention.

    That derivation assumes a grid symmetric about the origin, so it does NOT round-trip an
    even-sized source axis: MUSIC's nx=100, x_min=-15, dx=0.3 comes back as 101 cells
    spanning -15..+15 rather than -15..+14.7.  That is what the C++ does, and it is what you
    want when reproducing it -- but it makes a poor default, so an empty spec (or one whose
    values are all 0/None) returns the source grid unchanged instead.

    ``spec`` may instead be a :class:`Grid` (e.g. from :meth:`Grid.from_bounds`), which is
    taken literally -- any origin, any cell count, and 0 is a value rather than "unset".
    Its ``ntau`` is an upper bound: 0 runs to the end of the source's tau range, N > 0 keeps
    at most the first N frames.  Never more than the source covers, so a short event does
    not gain trailing all-zero frames (which would also inflate its ntau_freezeout).
    """
    if isinstance(spec, Grid):
        ntau = ntau_to_end(src, spec.tau_min, spec.dtau)
        if spec.ntau > 0:
            ntau = min(ntau, spec.ntau)
        return replace(spec, ntau=ntau, boost_invariant=src.boost_invariant)

    spec = dict(spec or {})
    if not any(spec.get(k) for k in
               ("x_min", "dx", "y_min", "dy", "eta_min", "deta", "tau_min", "dtau", "ntau")):
        return src

    def pick(key, fallback):
        v = spec.get(key)
        return fallback if v is None or v == 0 else float(v)

    dx = pick("dx", src.dx)
    dy = pick("dy", src.dy)
    deta = pick("deta", src.deta)
    dtau = pick("dtau", src.dtau)
    x_min = pick("x_min", src.x_min)
    y_min = pick("y_min", src.y_min)
    eta_min = pick("eta_min", src.eta_min)
    tau_min = pick("tau_min", src.tau_min)

    nx = 2 * int(abs(x_min) / dx) + 1
    ny = 2 * int(abs(y_min) / dy) + 1
    neta = 2 * int(abs(eta_min) / deta) + 1 if deta else 1

    ntau = int(spec.get("ntau") or 0)
    if ntau <= 0:
        ntau = ntau_to_end(src, tau_min, dtau)

    return Grid(nx=nx, ny=ny, neta=neta, ntau=ntau,
                x_min=x_min, dx=dx, y_min=y_min, dy=dy,
                eta_min=eta_min, deta=deta, tau_min=tau_min, dtau=dtau,
                boost_invariant=src.boost_invariant)


def ntau_to_end(src, tau_min, dtau):
    """Output frames from ``tau_min`` in steps of ``dtau`` up to the source's last frame."""
    if not dtau:
        return 0
    # Same 1e-6 slack as resample(): the source grid is float32, so an output frame
    # landing exactly on the last stored step can divide out at N-1e-7 and be dropped.
    # FastRootBulkWriter.cc applies the same slack.
    return max(int((src.tau_max - tau_min) / dtau + 1e-6) + 1, 0)


# -------------------------------------------------------------------- resampling
def _linear_weights(f, n):
    """1-D linear interpolation on sample positions ``f`` over ``n`` source points.

    -> ``(i0, i1, w)`` with ``value = (1 - w) * v[i0] + w * v[i1]``.  Both indices are
    clamped to ``[0, n-1]``, which is ``CellIndex`` in the C++ and scipy's
    ``map_coordinates(order=1, mode="nearest")``.  ``w`` is float64.
    """
    f = np.asarray(f, dtype=np.float64)
    fl = np.floor(f)
    w = f - fl
    i0 = fl.astype(np.intp)
    return np.clip(i0, 0, n - 1), np.clip(i0 + 1, 0, n - 1), w


def _weight_matrix(i0, i1, w, n):
    """Dense ``(len(w), n)`` matrix of the 1-D linear weights, so a pass is one matmul.

    Each row has ``1 - w`` at ``i0`` and ``w`` at ``i1`` (their sum where the two clamp
    onto the same point) and exact zeros elsewhere, which add nothing to the product.
    """
    m = np.zeros((len(w), n))
    rows = np.arange(len(w))
    np.add.at(m, (rows, i0), 1.0 - w)
    np.add.at(m, (rows, i1), w)
    return m


def resample(arr, src, out):
    """Resample ``(ntau, nx, ny, neta, F)`` from grid ``src`` onto grid ``out``.

    Reproduces ``EvolutionHistory::get()``; see the module docstring for the three
    behaviours that matter.  Works one output tau frame at a time, so peak extra memory is
    a couple of frames rather than a second copy of the event.

    The output points form a tensor-product grid (x by y by eta), so spatial trilinear
    interpolation factors into three 1-D linear passes, one per axis, each a matrix
    product over every feature at once.  That is the arithmetic of scipy's
    ``map_coordinates(order=1, mode="nearest", prefilter=False)`` per channel -- float64,
    cast to float32 per frame -- with the terms summed in a different order (results
    agree to about 1 ulp), and it replaces ~1,700 single-threaded ``map_coordinates``
    calls per event with three BLAS-backed products per source frame.
    """
    n_feat = arr.shape[-1]
    ntau_src = arr.shape[0]
    # Range tests run in index units, so a point that should land exactly on the last
    # sample can come out at n-1+4e-16 and be dropped.  Without this slack the final tau
    # frame of every event silently zeroes.  1e-6 of a cell is far below anything physical.
    tol = 1e-6

    fx = (out.axis("x") - src.x_min) / src.dx
    fy = (out.axis("y") - src.y_min) / src.dy

    # The eta axis collapses when the source has no eta structure: GetAtTimeStep forces
    # id_eta=0 for boost-invariant data and GetFluidCell forces it for neta<=1.
    eta_is_flat = src.boost_invariant or src.neta <= 1
    if eta_is_flat:
        feta = np.zeros(out.neta)
        eta_ok = np.ones(out.neta, dtype=bool)
        if src.neta <= 1 and not src.boost_invariant and out.neta > 1:
            warnings.warn(
                "source grid has neta=%d but is not flagged boost_invariant: the C++ "
                "EvolutionHistory::get() would zero every output eta except eta_min. "
                "Treating eta as flat instead; output will differ from RootBulkWriter."
                % src.neta, RuntimeWarning, stacklevel=2)
    else:
        feta = (out.axis("eta") - src.eta_min) / src.deta
        eta_ok = (feta >= -tol) & (feta <= src.neta - 1 + tol)

    x_ok = (fx >= -tol) & (fx <= src.nx - 1 + tol)
    y_ok = (fy >= -tol) & (fy <= src.ny - 1 + tol)
    mask = (x_ok[:, None, None] & y_ok[None, :, None] & eta_ok[None, None, :])
    mask = mask[..., None].astype(np.float32)          # broadcast over features

    # Only the source rows that some output point touches take part.  Cropping to them
    # keeps the float64 intermediates and the weight matrices small.
    def axis_matrix(f, n):
        i0, i1, w = _linear_weights(f, n)
        lo, hi = int(min(i0.min(), i1.min())), int(max(i0.max(), i1.max())) + 1
        return slice(lo, hi), _weight_matrix(i0 - lo, i1 - lo, w, hi - lo)

    sx, mx = axis_matrix(fx, src.nx)
    sy, my = axis_matrix(fy, src.ny)
    se, me = axis_matrix(feta, max(int(src.neta), 1))

    def spatial(k):
        """Source tau step k -> (out.nx, out.ny, out.neta, F) float32, unmasked."""
        a = np.asarray(arr[k, sx, sy, se, :], dtype=np.float64)
        a = np.einsum("ke,xyef->xykf", me, a, optimize=True)   # eta first: shrinks most
        a = np.einsum("jy,xykf->xjkf", my, a, optimize=True)
        a = np.einsum("ix,xjkf->ijkf", mx, a, optimize=True)
        return a.astype(np.float32)          # map_coordinates' output dtype is the input's

    result = np.zeros((out.ntau, out.nx, out.ny, out.neta, n_feat), dtype=np.float32)
    cache = {}

    def frame(k):
        """Spatially interpolated source tau step k, 2-deep LRU (consecutive output frames
        share a source step whenever the output tau step is finer than the source's)."""
        if k not in cache:
            if len(cache) >= 2:
                cache.pop(next(iter(cache)))
            cache[k] = spatial(k)
        return cache[k]

    for j in range(out.ntau):
        tau = out.tau_min + j * out.dtau
        ft = (tau - src.tau_min) / src.dtau if src.dtau else 0.0
        if ft < -tol or ft > ntau_src - 1 + tol:
            continue                                   # CheckInRange -> zero cell
        ft = min(max(ft, 0.0), ntau_src - 1.0)
        k0 = min(int(ft), ntau_src - 1)                # C++ truncates, then CellIndex clamps
        k1 = min(k0 + 1, ntau_src - 1)
        w = ft - k0
        v = frame(k0)
        if w and k1 != k0:
            v = (1.0 - w) * v + w * frame(k1)          # float32, as before
        result[j] = v * mask

    return result


# ------------------------------------------------------------------ event access
def event_array(hydro, grid_mode="native", tau_stride=1, out_spec=None):
    """Return ``(arr, src_grid, out_grid)`` for the current event.

    ``arr`` is ``(ntau, nx, ny, neta, 4)`` float32 with channels
    ``(energy_density, vx, vy, vz)``, on ``out_grid``.
    """
    if grid_mode not in GRID_MODES:
        raise ValueError(f"grid_mode must be one of {GRID_MODES}, got {grid_mode!r}")
    if hydro is None:
        raise RuntimeError("no hydro pointer -- add the writer AFTER the hydro module")

    if grid_mode == "framework":
        b = hydro.get_bulk_info()
        # to_numpy_full's layout is [e, temperature, vx, vy, vz]; taking [:4] would
        # silently store temperature as vx.
        raw = b.to_numpy_full(n_features=5)
        arr = np.ascontiguousarray(raw[..., [0, 2, 3, 4]])
        del raw
        src = Grid.from_bulk_info(b, ntau=arr.shape[0])
    else:
        if not hasattr(hydro, "get_native_evolution_numpy"):
            raise RuntimeError(
                f"grid_mode={grid_mode!r} needs MUSIC's native store; this hydro module "
                "does not expose get_native_evolution_numpy(). Use grid_mode='framework'.")
        if hasattr(hydro, "get_dump_hydro_only") and not hydro.get_dump_hydro_only():
            raise RuntimeError(
                f"grid_mode={grid_mode!r} requires <Hydro><MUSIC><dump_hydro_only>1 so "
                "MUSIC keeps its native evolution store.")
        arr = hydro.get_native_evolution_numpy(tau_stride=int(tau_stride))
        b = hydro.get_bulk_info()
        # Thinning in tau widens the effective step, as in FastRootBulkWriter.cc:127.
        src = Grid.from_bulk_info(b, ntau=arr.shape[0],
                                  dtau=float(b.dtau) * int(tau_stride))

    if grid_mode == "native":
        return arr, src, src

    out = resolve_out_grid(src, out_spec)
    return resample(arr, src, out), src, out


def framework_store_bytes(bulk_info):
    """Rough size of the framework AoS backing ``framework`` mode."""
    n = max(1, int(bulk_info.nx)) * max(1, int(bulk_info.ny)) * max(1, int(bulk_info.neta))
    return n * max(1, int(bulk_info.ntau)) * _FLUID_CELL_BYTES


# ----------------------------------------------------------------- attribute maps
def attrs_from_grids(src, out):
    """The 13 FNO4d `_SCALAR_KEYS` plus ``choose_ntau``, for grids ``src`` -> ``out``."""
    return {
        "nFeatures": 4,
        "nx": int(out.nx), "ny": int(out.ny), "neta": int(out.neta),
        "x_min": float(out.x_min), "dx": float(out.dx),
        "y_min": float(out.y_min), "dy": float(out.dy),
        "eta_min": float(out.eta_min), "deta": float(out.deta),
        "tau_min": float(out.tau_min), "dtau": float(out.dtau),
        # The source grid's origin, kept because it is in _SCALAR_KEYS.  Equal to tau_min
        # in native mode; they differ once an output tau grid is imposed.
        "tau_min_MUSIC": float(src.tau_min),
        "choose_ntau": int(out.ntau),
    }


def music_extra_attrs(src):
    """The source-grid parameters, under the names the ROOT writers use.

    Not read by any FNO4d loader, but ``read_3d_data_hdf5`` returns ``dict(hf.attrs)``, so
    these ride along for free and keep provenance comparable with the ROOT output.
    """
    return {
        "nX_MUSIC": int(src.nx), "dX_MUSIC": float(src.dx),
        "X_min_MUSIC": float(src.x_min),
        "nY_MUSIC": int(src.ny), "dY_MUSIC": float(src.dy),
        "Y_min_MUSIC": float(src.y_min),
        "neta_MUSIC": int(src.neta), "deta_MUSIC": float(src.deta),
        "eta_min_MUSIC": float(src.eta_min),
        "dtau_MUSIC": float(src.dtau), "ntau_MUSIC": int(src.ntau),
        "boost_invariant": bool(src.boost_invariant),
    }
