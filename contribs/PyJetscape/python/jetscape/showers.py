"""The parton shower as a space-time graph -- capture, store, and read back for animation.

This module lives in PyJetscape so that any writer can store showers -- the MUSIC pair
writer (`jetscape.pair_h5`) as well as FastHydro's.  It came from FastHydro, which re-exports
it as `fasthydro.showers`; it needs nothing but numpy and duck-types the energy-loss manager.
`PairBrowser` below is `fasthydro.browse.PairBrowser`.

The droplets in `source/droplets` are what the shower *gave up*.  This is what the shower
*was*: every parton, every splitting vertex, with the times at both ends, so the shower can be
drawn developing alongside the hydro frames instead of appearing all at once.

What makes it animatable
------------------------
Each parton carries its own production point ``(x, y, z, t)`` and its momentum, and
``source_id``/``target_id`` say which splitting it runs between.  So a parton is a *segment in
space-time*, not a point: it starts where it was produced and travels in a straight line at
``v = p/E`` until it splits.  `segments()` returns those two endpoints and `shower_at()` on
`PairBrowser` draws the partial segment at a given lab time.  Final-state partons alone would
be endpoints with no history -- a starburst, not a shower.

**The vertices carry no position.**  `JetEnergyLoss.cc:414-419` constructs every one of them
as ``Vertex(0, 0, 0, currentTime)``: the spatial coordinates are hardcoded zero and only ``t``
is real.  Measured on a live run, all 72 vertices of an event sat at the origin while the
partons had 18 distinct production points.  So the vertex rows are stored for their *times and
topology* -- which parton splits into which, and when -- and the geometry is taken from the
partons.  Building segments from vertex positions instead gives every parton zero length, which
looks like a plot bug rather than a data bug, so `segments()` does not touch those columns.

The end time of a parton is the production time of its children, falling back to its target
vertex's time when it has none.  The framework's first two vertices are default-constructed at
``t = 0`` (`JetEnergyLoss.cc:306-307`) even when the shower starts later, so the children are
the reliable source and the vertex time is only the fallback for final partons.  Checked on a
live event: propagating the leading parton from its production point along ``p/E`` to its
child's time reproduces the child's stored position to all printed digits.

Fates
-----
``pstat`` is the link between this group and `source/droplets`.  X-SCAPE's `LiquefierBase`
re-stamps partons as it filters them (`src/framework/LiquefierBase.cc:24-26`, `:105-160`):

    -11  drop   dropped into the medium below the threshold -- THIS is what became a droplet
    -17  neg    negative ("hole") parton, the recoil's partner
    -13  miss   missed
     22         photon, passed through untouched
    101         Matter's hand-off point to LBT

`fates()` maps them to names.  Colouring segments by fate is what turns the animation from a
tangle into a picture of energy leaving the jet.

Coordinates
-----------
Vertices come out of the framework in **Cartesian lab** ``(x, y, z)`` [fm] with ``t`` [fm/c],
while the hydro frames are Milne ``tau`` and the droplets are ``(tau, x, y, eta)``.  They are
stored raw -- that is the framework's own truth and converting on write would bake in a choice
-- and `to_milne()` puts them on the hydro clock when a viewer wants them there.

Layout in the file (all float64, ragged, offsets per event, exactly as `source/droplets`)::

    shower/partons          (N, 13)  shower, i_src, i_tgt, pid, pstat, px,py,pz,E, x,y,z,t
    shower/vertices         (M,  6)  shower, node_id, x, y, z, t   <- x,y,z are ALWAYS 0
    shower/initiators       (K, 11)  shower, pid, pstat, px,py,pz,E, x,y,z,t
    shower/parton_offsets   (nev+1,) int64
    shower/vertex_offsets   (nev+1,) int64
    shower/initiator_offsets(nev+1,) int64

``i_src``/``i_tgt`` are **row indices into that event's vertex slice**, already offset by
shower -- not raw GTL node ids, which restart at 0 in every shower and would collide the
moment an event's showers are concatenated.  The raw id is kept in the vertex rows for
traceability.  `PairBrowser.showers(event)` returns the two per-event slices together, so the
indices stay valid against each other; slicing one without the other is the way to get this
wrong.
"""

from __future__ import annotations

import numpy as np

__all__ = ["PARTON_COLUMNS", "VERTEX_COLUMNS", "INITIATOR_COLUMNS", "FATES", "ABSORBED",
           "ShowerRecord", "showers_from_manager", "empty_record",
           "segments", "first_child", "split_times", "velocities", "to_milne", "fates"]

PARTON_COLUMNS = ("shower", "i_src", "i_tgt", "pid", "pstat",
                  "px", "py", "pz", "E", "x", "y", "z", "t")
VERTEX_COLUMNS = ("shower", "node_id", "x", "y", "z", "t")
INITIATOR_COLUMNS = ("shower", "pid", "pstat", "px", "py", "pz", "E", "x", "y", "z", "t")

#: LiquefierBase's status codes (LiquefierBase.cc:24-26) plus the two the jet modules set
FATES = {-11: "drop", -17: "neg", -13: "miss", 22: "photon", 101: "matter_handoff"}

#: fates where the parton left the parton list for good -- `drop` went into the medium and
#: became a droplet, `miss` was removed. They stop at their production point; everything else
#: that does not split is still travelling when the graph runs out. Used by
#: `PairBrowser.shower_at` to decide which non-splitting partons keep moving.
ABSORBED = (-11, -13)


class ShowerRecord:
    """One event's showers, flattened and ready to write or draw."""

    __slots__ = ("partons", "vertices", "initiators")

    def __init__(self, partons, vertices, initiators):
        self.partons = np.asarray(partons, dtype=np.float64).reshape(-1, len(PARTON_COLUMNS))
        self.vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, len(VERTEX_COLUMNS))
        self.initiators = np.asarray(initiators, dtype=np.float64).reshape(
            -1, len(INITIATOR_COLUMNS))

    def __len__(self):
        return len(self.partons)

    @property
    def n_showers(self):
        return int(len(self.initiators)) or (
            0 if not len(self.partons) else int(self.partons[:, 0].max()) + 1)

    def __repr__(self):
        return (f"<ShowerRecord {self.n_showers} shower(s), {len(self.partons)} partons, "
                f"{len(self.vertices)} vertices>")


def empty_record():
    return ShowerRecord(np.zeros((0, len(PARTON_COLUMNS))),
                        np.zeros((0, len(VERTEX_COLUMNS))),
                        np.zeros((0, len(INITIATOR_COLUMNS))))


def showers_from_manager(mgr):
    """Flatten every finished `PartonShower` on a `JetEnergyLossManager` into one record.

    Must be called while the showers are alive -- after the manager's Exec and before
    ClearPerEvent.  `DropletBridge.Exec()` is exactly that window, which is why the capture
    lives there and not in the driver.

    Takes any object with `get_showers()` / `get_shower_initiating_partons()`, so it needs no
    `jetscape` import and stays testable with a stub.
    """
    partons, vertices, inits = [], [], []
    v_base = 0                                  # rows of `vertices` already emitted

    for ish, sh in enumerate(mgr.get_showers()):
        v = np.asarray(sh.vertices_to_numpy(), dtype=np.float64).reshape(-1, 5)
        e = np.asarray(sh.to_numpy(), dtype=np.float64).reshape(-1, 12)

        # GTL node ids are not promised to be the row order they come out in, and after a
        # deletion they are not even contiguous -- so build the map instead of assuming
        # id == row. Getting this wrong would draw every segment from the wrong vertex.
        row_of = {int(nid): i for i, nid in enumerate(v[:, 0])}

        vertices.append(np.column_stack([np.full(len(v), ish, dtype=np.float64), v]))

        if len(e):
            miss = [int(i) for i in np.concatenate([e[:, 0], e[:, 1]])
                    if int(i) not in row_of]
            if miss:
                raise ValueError(
                    f"shower {ish}: parton endpoints reference node id(s) {sorted(set(miss))} "
                    f"that vertices_to_numpy() did not return; the graph is inconsistent")
            i_src = np.array([v_base + row_of[int(i)] for i in e[:, 0]], dtype=np.float64)
            i_tgt = np.array([v_base + row_of[int(i)] for i in e[:, 1]], dtype=np.float64)
            partons.append(np.column_stack([
                np.full(len(e), ish, dtype=np.float64), i_src, i_tgt, e[:, 2:]]))
        v_base += len(v)

    for ish, p in enumerate(_initiators(mgr)):
        if p is None:
            continue
        inits.append([ish, p.pid(), p.pstat(), p.px(), p.py(), p.pz(), p.e(),
                      p.x(), p.y(), p.z(), p.t()])

    cat = lambda rows, w: (np.concatenate(rows, 0) if rows else np.zeros((0, w)))
    return ShowerRecord(cat(partons, len(PARTON_COLUMNS)),
                        cat(vertices, len(VERTEX_COLUMNS)),
                        np.asarray(inits, dtype=np.float64).reshape(-1,
                                                                    len(INITIATOR_COLUMNS)))


def _initiators(mgr):
    """The hard partons, one per shower -- absent on older bindings, which is not fatal."""
    try:
        return mgr.get_shower_initiating_partons()
    except AttributeError:
        return []


# ---------------------------------------------------------------------------- reading
def velocities(partons):
    """-> (N, 3) ``p/E``.  Zero-energy rows (the `miss` placeholders) come back as zero."""
    p = np.asarray(partons).reshape(-1, len(PARTON_COLUMNS))
    E = p[:, 8:9]
    return np.where(E > 0, p[:, 5:8] / np.where(E > 0, E, 1.0), 0.0)


def first_child(partons, vertices):
    """-> (N,) the row index of each parton's first child, or -1 where it has none.

    A parton's children are the partons leaving its target vertex.  Two cases have none, and
    both matter:

    * **final-state partons**, whose target vertex is a leaf.  The framework gives that vertex
      a time equal to the parton's own production time, so the graph holds no end time at all
      -- a final parton simply keeps going.
    * **negative ("hole") partons**, which `JetEnergyLoss.cc:413-416` attaches *backwards*:
      the edge runs ``new_vertex -> vStart``, so its target is its own parent's source vertex.
      The partons found there are its siblings, not its children, and they were produced no
      later than it was.  Requiring a strictly later production time is what rejects them --
      without it a hole would be drawn running back up its parent's track.
    """
    p = np.asarray(partons).reshape(-1, len(PARTON_COLUMNS))
    v = np.asarray(vertices).reshape(-1, len(VERTEX_COLUMNS))
    if not len(p) or not len(v):
        return np.full(len(p), -1, dtype=np.int64)
    i_src, i_tgt, t0 = p[:, 1].astype(int), p[:, 2].astype(int), p[:, 12]

    out = np.full(len(v), -1, dtype=np.int64)           # earliest departure from each vertex
    for i in np.argsort(t0, kind="stable")[::-1]:       # descending: the earliest writes last
        out[i_src[i]] = i

    kid = out[i_tgt]
    return np.where((kid >= 0) & (t0[np.maximum(kid, 0)] > t0), kid, -1)


def split_times(partons, vertices):
    """-> (N,) the lab time each parton splits, NaN where it never does."""
    p = np.asarray(partons).reshape(-1, len(PARTON_COLUMNS))
    kid = first_child(p, vertices)
    return np.where(kid >= 0, p[np.maximum(kid, 0), 12], np.nan)


def segments(partons, vertices):
    """-> (start, end, splits): the drawable form of every parton.

    `start` and `end` are ``(N, 4)`` as ``(x, y, z, t)``; `splits` is a boolean mask.  A parton
    that splits runs from its own production point to **its child's** production point, both
    of which are stored data -- no extrapolation.  (Propagating the parent along ``p/E``
    instead misses by ~1e-3 fm, because `p_in()` is the momentum at production and the parton
    loses some of it on the way.)  One that does not split -- a final-state parton, or a hole
    -- gets ``end == start`` and ``splits == False``, because the graph does not say when it
    stops; `PairBrowser.shower_at` is what decides how far to carry those at a given frame,
    and there ``p/E`` is the only thing to go on.

    Vertex *positions* are deliberately not used: the framework stores every one of them at
    the origin (see the module docstring), so a segment built from them would have zero length
    -- which reads as a plotting bug rather than a data one.
    """
    p = np.asarray(partons).reshape(-1, len(PARTON_COLUMNS))
    v = np.asarray(vertices).reshape(-1, len(VERTEX_COLUMNS))
    if not len(p):
        z = np.zeros((0, 4))
        return z, z, np.zeros(0, bool)
    if len(v) and (p[:, 1:3].max() >= len(v) or p[:, 1:3].min() < 0):
        raise IndexError(
            "parton endpoint indices fall outside the vertex block -- partons and vertices "
            "must come from the same event (PairBrowser.showers(event) returns the matching "
            "pair; slicing one of the two on its own is what breaks this)")

    start = np.column_stack([p[:, 9:12], p[:, 12]])
    kid = first_child(p, v)
    splits = kid >= 0
    end = np.where(splits[:, None], p[np.maximum(kid, 0)][:, 9:13], start)
    return start, end, splits


def to_milne(xyzt):
    """Cartesian ``(x, y, z, t)`` -> Milne ``(tau, x, y, eta)``, to share the hydro's clock.

    Space-like points (``|z| > t``, which the framework does produce for partons created
    before the hydro start time) have no real ``tau``; they come back as NaN rather than as a
    silently wrong number.
    """
    a = np.asarray(xyzt, dtype=np.float64).reshape(-1, 4)
    x, y, z, t = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    t2z2 = t * t - z * z
    tau = np.where(t2z2 > 0, np.sqrt(np.abs(t2z2)), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = np.where(t2z2 > 0, 0.5 * np.log(np.clip((t + z) / (t - z), 1e-300, None)),
                       np.nan)
    return np.column_stack([tau, x, y, eta])


def fates(partons):
    """-> list of fate names, one per parton (see FATES); unknown codes come back as the int."""
    p = np.asarray(partons).reshape(-1, len(PARTON_COLUMNS))
    return [FATES.get(int(s), str(int(s))) for s in p[:, 4]]
