"""Where hard scatterings happen: the density `SampleABinaryCollisionPoint` draws from.

`InitialState::SampleABinaryCollisionPoint` (`InitialState.cc:80-97`) builds a
`std::discrete_distribution` over `num_of_binary_collisions_` and returns the cell centre of
whatever index it draws.  If that vector is empty it only prints a warning and puts the vertex
at the origin -- so **every shower starts at the fireball centre**, which biases any
path-length-dependent observable and quietly makes all events look alike.

Hard processes scale with the binary-collision density T_A T_B, so `ncoll` is the physical
default.  The others are there to make the comparison, not because they are equally right.

The grid
--------
The density must be expressed on the axis the *sampler* uses, which is MUSIC's

    CoordFromIdx: x_i = -grid_max_x + i*grid_step_x            (InitialState.cc:75-77)

and not the cell-centred axis the energy density lives on.  The two differ by half a cell.
`node_axes()` below returns the sampler's, and the smeared modes evaluate the Gaussian
directly at those nodes, so a drawn vertex lands exactly where its density said it would.
The residual half-cell offset against the matching energy-density node is inherent to the
framework and is documented in the README.
"""

from __future__ import annotations

import numpy as np

__all__ = ["MODES", "node_axes", "binary_collision_density"]

#: how to distribute hard-scattering vertices
MODES = {
    "ncoll": "binary-collision (T_A T_B) density, each collision Gaussian-smeared -- the "
             "physical choice for a hard process, and the default",
    "ncoll_mc": "raw histogram of the MC binary-collision points; the same distribution as "
                "'ncoll' but unsmeared, so it is sparse and noisy on a coarse grid",
    "npart": "wounded-nucleon (participant) density, Gaussian-smeared; scales like the soft "
             "entropy rather than like a hard process. For comparison",
    "centre": "do not set a density at all: the framework warns and puts every vertex at "
              "(0,0,0). This is what happened before the setter existed -- kept so the old "
              "behaviour can be reproduced deliberately",
}


def node_axes(g):
    """The (x, y) node coordinates `CoordFromIdx` will report, given `GridSpec` `g`."""
    xr, yr, _ = g.is_ranges()
    return (-xr + g.dx * np.arange(g.nx), -yr + g.dy * np.arange(g.ny))


def _smeared(points, g, width):
    """Sum of 2D Gaussians of width `width` at `points`, evaluated on the sampler's nodes."""
    xs, ys = node_axes(g)
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(p) == 0:
        return np.zeros((g.nx, g.ny))
    inv = 1.0 / (2.0 * width * width)
    gx = np.exp(-((xs[None, :] - p[:, 0:1]) ** 2) * inv)          # (N, nx)
    gy = np.exp(-((ys[None, :] - p[:, 1:2]) ** 2) * inv)          # (N, ny)
    return np.einsum("ni,nj->ij", gx, gy)


def _histogram(points, g):
    """Raw MC histogram, binned so each bin is centred on a sampler node."""
    xs, ys = node_axes(g)
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(p) == 0:
        return np.zeros((g.nx, g.ny))
    ex = np.concatenate([xs - 0.5 * g.dx, [xs[-1] + 0.5 * g.dx]])
    ey = np.concatenate([ys - 0.5 * g.dy, [ys[-1] + 0.5 * g.dy]])
    h, _, _ = np.histogram2d(p[:, 0], p[:, 1], bins=[ex, ey])
    return h


def binary_collision_density(ev, g, mode="ncoll", width=0.4):
    """-> (nx, ny, neta) float64 density, or None when nothing should be set.

    Parameters
    ----------
    ev : dict
        A `fast_data.glauber` event: ``coll`` (binary-collision positions),
        ``plus`` / ``minus`` (participants).
    g : GridSpec
    mode : str
        One of :data:`MODES`.
    width : float
        Gaussian width in fm for the smeared modes.  The default matches the nucleon width
        `glauber.tilted_profile` deposits energy with, so the vertices and the medium they
        traverse are smeared consistently.

    Returns None for ``centre``, and also when the event has no points to draw from -- in
    both cases the caller must leave `num_of_binary_collisions_` empty.
    """
    if mode not in MODES:
        raise ValueError(
            f"unknown hard_vertex.mode {mode!r}. Choose one of:\n" +
            "\n".join(f"  {k:10s} {v}" for k, v in MODES.items()))
    if mode == "centre":
        return None
    if width <= 0 and mode in ("ncoll", "npart"):
        raise ValueError(f"hard_vertex.smear must be > 0 for mode={mode!r} (got {width})")

    if ev is None:
        return None
    if mode == "npart":
        parts = [np.asarray(ev[k]).reshape(-1, 2) for k in ("plus", "minus")
                 if ev.get(k) is not None and len(ev[k])]
        pts = np.concatenate(parts, axis=0) if parts else np.zeros((0, 2))
    else:
        pts = np.asarray(ev.get("coll") if ev.get("coll") is not None
                         else np.zeros((0, 2))).reshape(-1, 2)
    if len(pts) == 0:
        return None

    dens2d = _histogram(pts, g) if mode == "ncoll_mc" else _smeared(pts, g, width)
    if not np.isfinite(dens2d).all() or dens2d.sum() <= 0:
        return None

    # The sampler only uses x and y (it returns z = 0), so broadcasting over eta leaves the
    # (x, y) marginal unchanged and keeps the array the shape the framework expects.
    return np.ascontiguousarray(np.repeat(dens2d[:, :, None], g.neta, axis=2))
