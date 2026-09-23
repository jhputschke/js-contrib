"""Gates for the `shower/` group: capture, layout, and the space-time reconstruction.

The graph X-SCAPE hands over has three properties that are easy to get wrong and that produce
a *plausible* picture when you do, so each one has a gate here:

  * vertices carry no position (all at the origin), so geometry must come from the partons;
  * negative "hole" partons are attached with the edge reversed, so a naive child lookup walks
    back up the parent's track;
  * a final-state parton has no end time in the graph at all.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from fasthydro import showers as sh                                      # noqa: E402

P, V, I = (len(sh.PARTON_COLUMNS), len(sh.VERTEX_COLUMNS), len(sh.INITIATOR_COLUMNS))


# --------------------------------------------------------------------------- stubs
class _Shower:
    """Stands in for a bound PartonShower: the two numpy views are its whole interface."""

    def __init__(self, edges, nodes):
        self._e, self._n = np.asarray(edges, float), np.asarray(nodes, float)

    def to_numpy(self):
        return self._e

    def vertices_to_numpy(self):
        return self._n


class _Parton:
    def __init__(self, row):
        self._r = list(row)

    pid = property(lambda s: s._r[0])
    pstat = property(lambda s: s._r[1])
    px = property(lambda s: s._r[2])
    py = property(lambda s: s._r[3])
    pz = property(lambda s: s._r[4])
    e = property(lambda s: s._r[5])
    x = property(lambda s: s._r[6])
    y = property(lambda s: s._r[7])
    z = property(lambda s: s._r[8])
    t = property(lambda s: s._r[9])

    def __getattribute__(self, k):
        v = object.__getattribute__(self, k)
        return (lambda: v) if k in ("pid", "pstat", "px", "py", "pz", "e",
                                    "x", "y", "z", "t") else v


class _Mgr:
    def __init__(self, showers, inits=()):
        self._s, self._i = showers, [_Parton(r) for r in inits]

    def get_showers(self):
        return self._s

    def get_shower_initiating_partons(self):
        return self._i


def _linear_shower(n=3, t0=0.0, dt=1.0, v=(1.0, 0.0, 0.0), start=(0.0, 0.0, 0.0)):
    """A chain of `n` partons, each produced where the previous one reached. No branching."""
    nodes, edges, x = [], [], np.array(start, float)
    for k in range(n + 1):
        nodes.append([k, 0.0, 0.0, 0.0, t0 + k * dt])          # positions zero, as X-SCAPE does
    for k in range(n):
        t = t0 + k * dt
        edges.append([k, k + 1, 21, 0, v[0], v[1], v[2], 1.0, x[0], x[1], x[2], t])
        x = x + np.array(v) * dt
    return _Shower(edges, nodes)


# --------------------------------------------------------------------------- capture
def test_flattening_keeps_every_parton_and_vertex():
    rec = sh.showers_from_manager(_Mgr([_linear_shower(3), _linear_shower(2)]))
    assert rec.partons.shape == (5, P) and rec.vertices.shape == (7, V)
    assert rec.n_showers == 2
    assert list(rec.partons[:, 0]) == [0, 0, 0, 1, 1]


def test_endpoint_indices_are_offset_across_showers():
    """The killer bug: GTL node ids restart at 0 per shower, so concatenating without an
    offset points every second shower's partons at the first shower's vertices."""
    rec = sh.showers_from_manager(_Mgr([_linear_shower(2), _linear_shower(2)]))
    second = rec.partons[rec.partons[:, 0] == 1]
    assert second[:, 1].min() >= 3, "second shower still indexes the first shower's vertices"
    start, end, _ = sh.segments(rec.partons, rec.vertices)
    assert np.isfinite(start).all() and np.isfinite(end).all()


def test_node_ids_are_mapped_not_assumed():
    """GTL ids need not be the row order; assuming id == row draws from the wrong vertex."""
    nodes = [[7, 0, 0, 0, 0.0], [3, 0, 0, 0, 1.0], [9, 0, 0, 0, 2.0]]
    edges = [[7, 3, 21, 0, 1, 0, 0, 1, 0, 0, 0, 0.0],
             [3, 9, 21, 0, 1, 0, 0, 1, 1, 0, 0, 1.0]]
    rec = sh.showers_from_manager(_Mgr([_Shower(edges, nodes)]))
    assert list(rec.partons[:, 1]) == [0, 1] and list(rec.partons[:, 2]) == [1, 2]
    assert list(rec.vertices[:, 1]) == [7, 3, 9], "raw node_id must survive for traceability"


def test_a_dangling_endpoint_is_refused_not_silently_drawn():
    nodes = [[0, 0, 0, 0, 0.0]]
    edges = [[0, 5, 21, 0, 1, 0, 0, 1, 0, 0, 0, 0.0]]     # node 5 does not exist
    with pytest.raises(ValueError, match="inconsistent"):
        sh.showers_from_manager(_Mgr([_Shower(edges, nodes)]))


def test_initiators_are_captured():
    rec = sh.showers_from_manager(_Mgr([_linear_shower(2)],
                                       inits=[[21, 0, 10, 0, 0, 10, 1, 2, 3, 0.4]]))
    assert rec.initiators.shape == (1, I)
    assert list(rec.initiators[0]) == [0, 21, 0, 10, 0, 0, 10, 1, 2, 3, 0.4]


def test_an_empty_manager_gives_an_empty_record():
    rec = sh.showers_from_manager(_Mgr([]))
    assert len(rec) == 0 and rec.n_showers == 0
    a, b, s = sh.segments(rec.partons, rec.vertices)
    assert a.shape == (0, 4) and b.shape == (0, 4) and s.shape == (0,)


# --------------------------------------------------------------------------- geometry
def test_segments_ignore_vertex_positions():
    """X-SCAPE builds every Vertex at (0,0,0) (JetEnergyLoss.cc:414-419). Reading geometry
    from them gives every parton zero length -- which looks like a plotting bug."""
    rec = sh.showers_from_manager(_Mgr([_linear_shower(3, v=(1.0, 0.0, 0.0))]))
    assert (rec.vertices[:, 2:5] == 0).all(), "the stub must reproduce X-SCAPE's convention"
    start, end, splits = sh.segments(rec.partons, rec.vertices)
    assert splits[:2].all() and not splits[2], "the last parton of a chain cannot split"
    assert np.allclose(np.linalg.norm((end - start)[:2, :3], axis=1), 1.0)


def test_a_splitting_parton_ends_exactly_on_its_child():
    """Not near it: both points are stored data, so the segment must be exact. Propagating the
    parent along p/E instead misses by ~1e-3 fm on a real shower, because p_in() is the
    momentum at production and the parton loses some of it on the way."""
    rec = sh.showers_from_manager(_Mgr([_linear_shower(3)]))
    start, end, splits = sh.segments(rec.partons, rec.vertices)
    kid = sh.first_child(rec.partons, rec.vertices)
    for i in np.where(splits)[0]:
        assert np.array_equal(end[i], rec.partons[kid[i], 9:13])


def test_a_backwards_hole_edge_is_not_mistaken_for_a_split():
    """LiquefierBase marks holes pstat -17 and JetEnergyLoss.cc:413-416 attaches them with the
    edge REVERSED (new_vertex -> vStart). Its 'children' are its parent's siblings, produced
    no later than it was; without the strictly-later test the hole is drawn running back up
    the parent's track."""
    nodes = [[0, 0, 0, 0, 0.0], [1, 0, 0, 0, 1.0], [2, 0, 0, 0, 1.0], [3, 0, 0, 0, 1.0]]
    edges = [[0, 1, 21, 0, 1, 0, 0, 1, 0, 0, 0, 0.0],      # parent 0 -> 1
             [1, 2, 21, 0, 1, 0, 0, 1, 1, 0, 0, 1.0],      # child, produced later
             [3, 1, 21, -17, -1, 0, 0, 1, 5, 5, 5, 1.0]]   # hole, edge points BACK into 1
    rec = sh.showers_from_manager(_Mgr([_Shower(edges, nodes)]))
    start, end, splits = sh.segments(rec.partons, rec.vertices)
    assert sh.fates(rec.partons)[2] == "neg"
    assert not splits[2], "the hole's target is its parent's vertex, not a splitting point"
    assert np.array_equal(end[2], start[2])


def test_split_times_agree_with_first_child():
    rec = sh.showers_from_manager(_Mgr([_linear_shower(3)]))
    kid = sh.first_child(rec.partons, rec.vertices)
    t = sh.split_times(rec.partons, rec.vertices)
    assert np.isnan(t[kid < 0]).all()
    assert np.allclose(t[kid >= 0], rec.partons[kid[kid >= 0], 12])


def test_milne_conversion_and_its_space_like_points():
    m = sh.to_milne([[0, 0, 0, 2.0],          # tau = 2, eta = 0
                     [0, 0, 1.0, 2.0],        # tau = sqrt(3)
                     [0, 0, 5.0, 1.0]])       # |z| > t: no real tau
    assert np.isclose(m[0, 0], 2.0) and np.isclose(m[0, 3], 0.0)
    assert np.isclose(m[1, 0], np.sqrt(3.0))
    assert np.isclose(m[1, 3], 0.5 * np.log(3.0))
    assert np.isnan(m[2, 0]) and np.isnan(m[2, 3]), "space-like must be NaN, not a wrong number"


def test_fates_name_the_liquefier_codes():
    p = np.zeros((4, P))
    p[:, 4] = [-11, -17, -13, 0]
    assert sh.fates(p) == ["drop", "neg", "miss", "0"]


def test_velocities_survive_a_zero_energy_row():
    p = np.zeros((2, P))
    p[0, 5:9] = [1.0, 0.0, 0.0, 2.0]
    v = sh.velocities(p)
    assert np.allclose(v[0], [0.5, 0, 0]) and np.allclose(v[1], 0.0)
    assert np.isfinite(v).all(), "E=0 (the `miss` placeholders) must not produce NaN"


def test_mismatched_partons_and_vertices_are_refused():
    rec = sh.showers_from_manager(_Mgr([_linear_shower(3)]))
    with pytest.raises(IndexError, match="same event"):
        sh.segments(rec.partons, rec.vertices[:1])


# --------------------------------------------------------------------------- round trip
def _cfg():
    from fasthydro.config import load_config
    return load_config(os.path.join(os.path.dirname(__file__), "..", "config",
                                    "fasthydro_twostage.yaml"), [])


class _Leg:
    def __init__(self, g, ntau):
        self.arr = np.zeros((4, g.nx, g.ny, g.neta, ntau), np.float32)
        self.src, self.diag, self.g, self.ic_sha256 = None, {}, g, "abc"


class _Bridge:
    def __init__(self, rec):
        self.droplets, self.params, self.shower = None, None, rec


def _write(tmp_path, recs):
    from fasthydro.grid import GridSpec
    from fasthydro.h5_writer import PairedH5Writer

    cfg = _cfg()
    cfg["eos"]["store_table"] = False
    g = GridSpec.from_cfg(cfg)
    p = str(tmp_path / "pair.h5")
    with PairedH5Writer(p, cfg, len(recs), overwrite=True) as w:
        for i, r in enumerate(recs):
            w.append(i, _Leg(g, g.ntau), _Leg(g, g.ntau), _Bridge(r))
    return p


def test_the_group_round_trips_through_the_writer_and_browser(tmp_path):
    pytest.importorskip("h5py")
    from fasthydro.browse import PairBrowser

    rec = sh.showers_from_manager(_Mgr([_linear_shower(3), _linear_shower(2)],
                                       inits=[[21, 0, 1, 0, 0, 1, 0, 0, 0, 0.0]]))
    with PairBrowser(_write(tmp_path, [rec])) as b:
        assert b.has_shower
        par, ver, ini = b.showers(0)
        assert np.array_equal(par, rec.partons) and np.array_equal(ver, rec.vertices)
        assert np.array_equal(ini, rec.initiators)
        assert b.summary(0)["n_partons"] == 5


def test_per_event_slices_do_not_bleed_into_each_other(tmp_path):
    """With one offsets array per block, an off-by-one hands event 1 event 0's partons while
    still looking structurally fine."""
    pytest.importorskip("h5py")
    from fasthydro.browse import PairBrowser

    recs = [sh.showers_from_manager(_Mgr([_linear_shower(n)])) for n in (2, 4, 3)]
    with PairBrowser(_write(tmp_path, recs)) as b:
        for i, r in enumerate(recs):
            par, ver, _ = b.showers(i)
            assert np.array_equal(par, r.partons), f"event {i} got the wrong partons"
            assert np.array_equal(ver, r.vertices)
            # the indices must still resolve WITHIN the slice
            assert par[:, 1:3].max() < len(ver)


def test_an_event_with_no_shower_is_empty_not_missing(tmp_path):
    pytest.importorskip("h5py")
    from fasthydro.browse import PairBrowser

    recs = [sh.showers_from_manager(_Mgr([_linear_shower(3)])), sh.empty_record(),
            sh.showers_from_manager(_Mgr([_linear_shower(2)]))]
    with PairBrowser(_write(tmp_path, recs)) as b:
        assert len(b.showers(1)[0]) == 0
        assert len(b.showers(2)[0]) == 2, "the empty event must still advance the offsets"


def test_a_file_without_showers_reads_back_as_none(tmp_path):
    pytest.importorskip("h5py")
    from fasthydro.browse import PairBrowser

    with PairBrowser(_write(tmp_path, [None])) as b:
        assert not b.has_shower
        assert b.showers(0) is None and b.shower_at(0, 1.0) is None
        assert b.summary(0)["n_partons"] is None


def test_shower_at_respects_each_parton_s_fate():
    rec = sh.showers_from_manager(_Mgr([_linear_shower(2, v=(1.0, 0.0, 0.0))]))
    # make the trailing parton one the liquefier absorbed
    rec.partons[-1, 4] = -11
    start, end, splits = sh.segments(rec.partons, rec.vertices)
    assert not splits[-1]
    assert sh.fates(rec.partons)[-1] == "drop"


def test_the_npz_carries_the_shower_for_replay(tmp_path):
    """A replayed file must stay animatable, or 'same jet, different solver' loses the jet."""
    from fasthydro.droplets_io import _flatten_showers, showers_from_meta

    recs = [sh.showers_from_manager(_Mgr([_linear_shower(3)])),
            sh.showers_from_manager(_Mgr([_linear_shower(2)]))]
    meta = _flatten_showers(recs)
    for i, r in enumerate(recs):
        got = showers_from_meta(meta, i)
        assert np.array_equal(got.partons, r.partons)
        assert np.array_equal(got.vertices, r.vertices)
    assert showers_from_meta({}, 0) is None


# --------------------------------------------------------------------------- the bridge
def _bare_bridge(manager, store=True):
    """A DropletBridge with only the fields _capture_shower touches.

    Built with __new__ so the test needs no CausalLiquefier and no running hydro: the capture
    is deliberately independent of both.
    """
    from fasthydro.liquefier_bridge import DropletBridge

    b = DropletBridge.__new__(DropletBridge)
    b.manager, b.verbose = manager, False
    b.cfg = {"fasthydro": {"store_showers": store}}
    b.shower, b.shower_history = None, []
    return b


def test_the_bridge_captures_the_shower():
    b = _bare_bridge(_Mgr([_linear_shower(3)]))
    b._capture_shower()
    assert len(b.shower.partons) == 3 and len(b.shower_history) == 1


def test_store_showers_false_captures_nothing():
    b = _bare_bridge(_Mgr([_linear_shower(3)]), store=False)
    b._capture_shower()
    assert b.shower is None and b.shower_history == []


def test_a_broken_manager_costs_the_group_not_the_run(capsys):
    """The shower rides alongside the hydro; a binding that cannot produce it must degrade to
    'no shower group', never to a lost event."""
    class _Broken:
        def get_showers(self):
            raise RuntimeError("no such binding")

    b = _bare_bridge(_Broken())
    b.verbose = True
    b._capture_shower()
    assert len(b.shower) == 0
    assert "WARNING" in capsys.readouterr().out


def test_no_manager_means_no_capture():
    b = _bare_bridge(None)
    b._capture_shower()
    assert b.shower is None
