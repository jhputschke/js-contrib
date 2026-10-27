"""
The Pythonic layer (docs/Plans/PLAN_pyjetscape_pythonic.md), the parts that need no XML:
the installer's refusals, the record arrays against the arrays they reorder, HadronArray
and its awkward view, PDG selections, 4-vectors, particles, pickling and shower nodes.
The parts that run JETSCAPE are in test_pythonic_run.py (in subprocesses).
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
# a pyjetscape_core that is there but does not load is an error, not a skip
core = pytest.importorskip("jetscape.pyjetscape_core", exc_type=ModuleNotFoundError)

import jetscape  # noqa: E402
from jetscape import pythonic  # noqa: E402
from jetscape.hadron_array import HadronArray  # noqa: E402
from jetscape.records import (EDGE_DTYPE, FINAL_PARTON_DTYPE, INITIATOR_DTYPE,  # noqa: E402
                              NODE_DTYPE, edge_records, final_parton_records,
                              final_parton_rows, hard_parton_records, initiator_records,
                              node_records)

rng = np.random.default_rng(7)


# ------------------------------------------------------------------------------ installer

def test_the_installer_refuses_a_name_a_class_has():
    with pytest.raises(AttributeError, match="only adds names"):
        pythonic._add(core.FourVector, "x", lambda self: 0)
    with pytest.raises(AttributeError, match="only adds names"):
        pythonic._add(core.Parton, "pt", pythonic._Prop(lambda self: 0))   # inherited


def test_the_installer_refuses_dunders_outside_its_list_and_existing_ones():
    with pytest.raises(AttributeError, match="not a dunder"):
        pythonic._add(core.FourVector, "__len__", lambda self: 4)
    with pytest.raises(AttributeError, match="exists"):
        pythonic._add(core.FourVector, "__iter__", lambda self: iter(()))   # a second one


def test_install_is_idempotent_and_the_layer_is_on_the_core_module():
    pythonic.install()
    assert core.make_parton is jetscape.make_parton
    assert core.to_awkward is jetscape.to_awkward
    names = pythonic.installed_names()
    assert "PartonShower.__iter__" in names and "PartonShower.__len__" not in names
    assert not any(n.startswith(("Parton.__eq__", "Hadron.__eq__")) for n in names)


def test_particles_keep_identity_equality_and_hash():
    p = core.make_parton(21, (10, 0, 0, 10))
    q = core.make_parton(21, (10, 0, 0, 10))
    assert p != q and hash(p) != hash(q) and len({p, q}) == 2


def test_an_empty_shower_stays_truthy():
    assert bool(core.PartonShower())          # no __len__


# ------------------------------------------------------------------------------ records

def _final_rows(n):
    a = rng.normal(size=(n, 14))
    a[:, 0] = np.repeat(np.arange(3), n // 3 + 1)[:n]          # shower
    a[:, 1] = rng.choice([21, 1, -2, 22], n)                   # pid
    a[:, 2] = rng.integers(-1, 2, n)                           # pstat
    a[:, 12] = rng.integers(0, 4_000_000_000, n)               # col: above 2**31 too
    a[:, 13] = rng.integers(0, 4_000_000_000, n)
    return a


def test_final_parton_records_reorder_bit_for_bit():
    a = _final_rows(50)
    r = final_parton_records(a)
    assert r.dtype == FINAL_PARTON_DTYPE
    cols = ("shower", "pid", "pstat", "E", "px", "py", "pz", "t", "x", "y", "z", "mass",
            "col", "acol")
    for i, name in enumerate(cols):
        assert np.array_equal(r[name].astype(np.float64), a[:, i]), name
    assert np.array_equal(final_parton_rows(r), a)              # and back
    assert final_parton_rows(a) is not None and np.array_equal(final_parton_rows(a), a)


def test_initiator_edge_node_and_hard_records():
    ini = rng.normal(size=(4, 11))
    ini[:, :3] = rng.integers(0, 5, (4, 3))
    r = initiator_records(ini)
    assert r.dtype == INITIATOR_DTYPE
    assert np.array_equal(r["E"], ini[:, 6]) and np.array_equal(r["t"], ini[:, 10])
    assert np.array_equal(r["px"], ini[:, 3]) and np.array_equal(r["x"], ini[:, 7])
    hard = rng.normal(size=(2, 7))
    hard[:, 0] = [21, -1]
    h = hard_parton_records(hard)
    assert np.array_equal(h["E"], hard[:, 4]) and np.array_equal(h["y"], hard[:, 6])
    assert len(final_parton_records(np.zeros((0, 14)))) == 0


def _shower():
    ps = core.PartonShower()
    v = [ps.new_vertex(core.Vertex(0.1 * k, 0.0, 0.0, float(k))) for k in range(4)]
    mother = core.make_parton(21, (40, 40, 0, 0))
    d1 = core.make_parton(21, (30, 30, 1, 0), position=(1, 0.1, 0, 0))
    d2 = core.make_parton(21, (10, 10, -1, 0), position=(1, 0.1, 0, 0))
    ps.new_parton(v[0], v[1], mother)
    ps.new_parton(v[1], v[2], d1)
    ps.new_parton(v[1], v[3], d2)
    return ps, v, mother, d1, d2


def test_edge_and_node_arrays_against_to_numpy():
    ps, *_ = _shower()
    e, raw = ps.edge_array(), ps.to_numpy()
    assert e.dtype == EDGE_DTYPE
    for name, col in (("src", 0), ("tgt", 1), ("pid", 2), ("pstat", 3), ("px", 4),
                      ("py", 5), ("pz", 6), ("E", 7), ("x", 8), ("y", 9), ("z", 10),
                      ("t", 11)):
        assert np.array_equal(e[name].astype(np.float64), raw[:, col]), name
    n, rawv = ps.node_array(), ps.vertices_to_numpy()
    assert n.dtype == NODE_DTYPE
    assert np.array_equal(n["id"], rawv[:, 0]) and np.array_equal(n["t"], rawv[:, 4])
    assert np.array_equal(edge_records(raw), e) and np.array_equal(node_records(rawv), n)


# ------------------------------------------------------------------------------ hadrons

def _hadron_dict(counts=(3, 0, 4)):
    n = sum(counts)
    p = np.zeros((n, 4), np.float32)
    p[:, 1:] = rng.normal(0, 1, (n, 3))
    p[:, 0] = np.linalg.norm(p[:, 1:], axis=1) + 0.14
    return {"pid": rng.choice([211, -211, 111, 2212, 3122, 22], n).astype(np.int32),
            "pstat": np.zeros(n, np.int32), "p": p,
            "x": rng.normal(size=(n, 4)).astype(np.float32),
            "mass": np.full(n, 0.14, np.float32),
            "sample_counts": np.array(counts, np.int64)}


def test_hadron_array_wraps_the_dict_without_a_copy():
    d = _hadron_dict()
    h = HadronArray.from_dict(d)
    assert len(h) == 7 and h.n_samples == 3
    assert list(h.sample_offsets) == [0, 3, 3, 7]
    for k in ("pid", "pstat", "p", "x", "mass"):
        assert np.shares_memory(getattr(h, k), d[k]), k
    one = HadronArray.from_dict({k: v for k, v in d.items() if k != "sample_counts"} |
                                {"seed": 5, "n_partons_out": 0})
    assert one.n_samples == 1 and one.info == {"seed": 5, "n_partons_out": 0}


def test_hadron_array_to_hadrons():
    h = HadronArray.from_dict(_hadron_dict())
    old = h.to_hadrons()
    assert np.array_equal(old.pid, h.pid) and np.array_equal(old.sample_offsets,
                                                             h.sample_offsets)


def test_to_awkward_is_samples_by_hadrons_over_the_same_buffers():
    ak = pytest.importorskip("awkward")
    pytest.importorskip("vector")
    d = _hadron_dict()
    h = HadronArray.from_dict(d)
    a = h.to_awkward(position=True)
    assert ak.num(a).tolist() == list(d["sample_counts"])
    for i, f in enumerate(("E", "px", "py", "pz")):
        assert np.array_equal(ak.to_numpy(ak.flatten(a[f])), d["p"][:, i])
        assert np.shares_memory(np.asarray(a.layout.content.content(f).data), d["p"])
    for i, f in enumerate(("t", "x", "y", "z")):
        assert np.array_equal(ak.to_numpy(ak.flatten(a.position[f])), d["x"][:, i])
    pt = ak.to_numpy(ak.flatten(a.pt))
    assert np.allclose(pt, np.hypot(d["p"][:, 1], d["p"][:, 2]))
    assert np.array_equal(ak.to_numpy(ak.flatten(a.pid)), d["pid"])


def test_to_awkward_of_final_partons_is_showers_by_partons():
    ak = pytest.importorskip("awkward")
    pytest.importorskip("vector")
    r = final_parton_records(_final_rows(10))
    a = jetscape.to_awkward(r)
    assert ak.num(a).tolist() == [4, 4, 2]
    assert np.array_equal(ak.to_numpy(ak.flatten(a.E)), r["E"])
    assert np.array_equal(ak.to_numpy(ak.flatten(a.position.t)), r["t"])
    assert np.allclose(ak.to_numpy(ak.flatten(a.pt)), np.hypot(r["px"], r["py"]))
    flat = jetscape.to_awkward(np.zeros(3, EDGE_DTYPE))
    assert len(flat) == 3 and flat.ndim == 1


def test_fastjet_on_the_awkward_view_equals_a_loop_over_samples():
    ak = pytest.importorskip("awkward")
    fastjet = pytest.importorskip("fastjet")
    d = _hadron_dict((40, 25, 60))
    a = HadronArray.from_dict(d).to_awkward()
    jdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
    jets = fastjet.ClusterSequence(a, jdef).inclusive_jets()
    offsets = np.r_[0, np.cumsum(d["sample_counts"])]
    for s in range(3):
        pj = [fastjet.PseudoJet(float(q[1]), float(q[2]), float(q[3]), float(q[0]))
              for q in d["p"][offsets[s]:offsets[s + 1]]]
        loop = sorted(j.pt() for j in fastjet.ClusterSequence(pj, jdef).inclusive_jets())
        assert np.allclose(sorted(ak.to_numpy(jets[s].pt)), loop)


def test_select_by_name_class_and_the_charged_list():
    pytest.importorskip("particle")
    from jetscape.hadrons_h5 import CHARGED

    pid = np.array([211, -211, 111, 2212, -2212, 3122, 22, 11, 321, 310, 113, 2224])
    h = HadronArray(pid, np.zeros(12), np.zeros((12, 4)), np.zeros((12, 4)), np.zeros(12))
    assert list(pid[jetscape.select(h, "pi+")]) == [211]
    assert list(pid[jetscape.select(h, ["K+", "p~"])]) == [-2212, 321]
    assert set(pid[jetscape.select(h, "baryon")]) == {2212, -2212, 3122, 2224}
    assert set(pid[jetscape.select(h, "strange")]) == {3122, 321, 310}
    charged = jetscape.select(pid, "charged")
    stable = ~np.isin(pid, [113, 2224])            # resonances: in "charged", not CHARGED
    assert np.array_equal(charged[stable], np.isin(np.abs(pid), CHARGED)[stable])
    assert set(pid[charged & ~np.isin(np.abs(pid), CHARGED)]) == {2224}
    with pytest.raises(ValueError, match="neither a PDG name"):
        jetscape.select(pid, "pion")


def test_select_on_awkward_keeps_the_shape():
    ak = pytest.importorskip("awkward")
    pytest.importorskip("particle")
    a = HadronArray.from_dict(_hadron_dict()).to_awkward(charge=True)
    m = jetscape.select(a, "charged")
    assert ak.num(m).tolist() == ak.num(a).tolist()
    assert np.array_equal(ak.to_numpy(ak.flatten(m)),
                          ak.to_numpy(ak.flatten(a.charge)) != 0)


# ------------------------------------------------------------------------------ 4-vectors

def test_fourvector_protocols_in_comp_order():
    v = core.FourVector.momentum(10.0, 1.0, 2.0, 3.0)
    assert list(v) == [10.0, 1.0, 2.0, 3.0] == [v.comp(i) for i in range(4)]
    assert v[0] == 10.0 and v[-1] == 3.0
    assert np.array_equal(np.asarray(v), [10.0, 1.0, 2.0, 3.0])
    assert v.to_list() == [1.0, 2.0, 3.0, 10.0]            # unchanged: (x, y, z, t)
    assert core.FourVector.position(0.5, 1, 2, 3).t() == 0.5
    assert v == core.FourVector(1, 2, 3, 10) and v != core.FourVector(1, 2, 3, 11)
    with pytest.raises(TypeError):
        hash(v)
    with pytest.raises(IndexError):
        v[4]
    assert pickle.loads(pickle.dumps(v)) == v


def test_particles_by_keywords_momentum_and_position():
    p = core.make_parton(21, (10, 1, 2, 3), (0.5, 0.1, 0.2, 0.3), stat=2, label=4,
                         color=101, anti_color=102)
    E, px, py, pz = p.momentum
    assert (E, px, py, pz) == (p.e(), p.px(), p.py(), p.pz()) == (10, 1, 2, 3)
    assert tuple(p.position) == (0.5, 0.1, 0.2, 0.3) == (p.t(), p.x(), p.y(), p.z())
    assert (p.pstat(), p.plabel(), p.color(), p.anti_color()) == (2, 4, 101, 102)
    h = core.make_hadron(2212, core.FourVector.momentum(2.0, 0.1, 0.2, 0.3), mass=0.938)
    assert h.restmass() == pytest.approx(0.938) and h.pid() == 2212
    assert core.make_parton(21, p.momentum).momentum == p.momentum      # a FourVector in


def test_pickling_particles():
    p = core.make_parton(1, (10, 1, 2, 3), (0.5, 0, 0, 0), color=7)
    p.set_form_time(0.25)
    q = pickle.loads(pickle.dumps(p))
    assert type(q) is core.Parton and q.momentum == p.momentum and q.position == p.position
    assert (q.pid(), q.color(), q.form_time()) == (1, 7, 0.25)
    assert type(pickle.loads(pickle.dumps(core.make_photon((5, 0, 0, 5))))) is core.Photon
    h = core.make_hadron(211, (1.0, 0.3, 0.2, 0.1), (1, 2, 3, 4), stat=1)
    g = pickle.loads(pickle.dumps(h))
    assert (g.pid(), g.pstat(), g.restmass()) == (211, 1, h.restmass())
    assert g.momentum == h.momentum and g.position == h.position


# ------------------------------------------------------------------------------ showers

def test_shower_nodes_against_the_id_based_calls():
    ps, v, mother, d1, d2 = _shower()
    assert [p.e() for p in ps] == [p.e() for p in ps.partons()]
    split = ps.end_node(mother)
    assert split == ps.start_node(d1) == ps.start_node(d2)
    assert split.id == ps.edge_of(mother)[2]
    assert {id(p) for p in split.children} == {id(d1), id(d2)}
    assert [id(p) for p in split.parents] == [id(mother)]
    assert split.split_z == ps.split_z(split.id)
    assert split.split_kt == ps.split_kt(split.id)
    assert split.split_delta_r == ps.split_delta_r(split.id)
    assert split.is_one_to_two == bool(ps.is_one_to_two(split.id)) is True
    assert ps.end_node(d1).is_end and not split.is_end
    assert tuple(split.position) == (1.0, 0.1, 0.0, 0.0)
    assert sorted(n.id for n in ps.nodes) == sorted(v)


# ------------------------------------------------------------------------------ modules

def test_module_decorator_declares_defaults_and_registers():
    @core.module("CustomModulePyTestDefaults", section="Eloss")
    class M(core.JetEnergyLoss):
        xml_parameters = {"t_split": 1.0, "e_min": 30, "label": "x", "on": True}

    try:
        assert (M.t_split, M.e_min, M.label, M.on) == (1.0, 30, "x", True)
        assert "CustomModulePyTestDefaults" in core.registered_modules(True)
        assert M.module_section == "Eloss"
    finally:
        core.unregister_module("CustomModulePyTestDefaults")


def test_module_decorator_refuses_other_parameter_types():
    with pytest.raises(TypeError, match="int, float, str or bool"):
        @core.module("CustomModulePyTestBad")
        class M(core.JetEnergyLoss):
            xml_parameters = {"list": [1, 2]}


def test_rng_is_for_python_modules_only():
    assert not hasattr(core.create_module("Matter"), "rng")     # no draw from its engine

    class Mine(core.JetEnergyLoss):
        def __init__(self):
            super().__init__()
            self.rng = "own"
    assert Mine().rng == "own"                                  # an attribute wins


# ------------------------------------------------------------------------------ returns

class _ReturnSplit(core.JetEnergyLoss):
    def DoEnergyLoss(self, deltaT, time, Q2, pIn, pOut):
        p = pIn[0]
        if time >= 1.0:
            E, px, py, pz = p.momentum
            return [core.make_parton(p.pid(), (E / 2, px / 2, s * 1.0, pz / 2),
                                     p.position) for s in (+1, -1)]


def test_a_returned_list_is_the_energy_loss_output():
    m, pin, pout = _ReturnSplit(), [core.make_parton(21, (50, 50, 0, 0))], []
    core.JetEnergyLoss.DoEnergyLoss(m, 0.1, 0.5, 0.0, pin, pout)       # returns None
    assert pout == []
    core.JetEnergyLoss.DoEnergyLoss(m, 0.1, 1.0, 0.0, pin, pout)       # through C++
    assert [(q.e(), q.py()) for q in pout] == [(25.0, 1.0), (25.0, -1.0)]


def test_returning_and_appending_is_refused():
    class Both(core.JetEnergyLoss):
        def DoEnergyLoss(self, deltaT, time, Q2, pIn, pOut):
            pOut.append(pIn[0])
            return [pIn[0]]

    class Wrong(core.JetEnergyLoss):
        def DoEnergyLoss(self, deltaT, time, Q2, pIn, pOut):
            return 3

    pin = [core.make_parton(21, (50, 50, 0, 0))]
    with pytest.raises(ValueError, match="also appended"):
        core.JetEnergyLoss.DoEnergyLoss(Both(), 0.1, 1.0, 0.0, pin, [])
    with pytest.raises(TypeError, match="None or a list"):
        core.JetEnergyLoss.DoEnergyLoss(Wrong(), 0.1, 1.0, 0.0, pin, [])


def test_hadronization_returns_hadrons_or_hadrons_and_partons():
    class Pions(core.Hadronization):
        def DoHadronization(self, shower, hOut, pOut):
            return [core.make_hadron(211, p.momentum, p.position) for s in shower for p in s]

    class Both(core.Hadronization):
        def DoHadronization(self, shower, hOut, pOut):
            return ([core.make_hadron(111, p.momentum) for s in shower for p in s],
                    [s[0] for s in shower])

    shower = [[core.make_parton(21, (10, 10, 0, 0)), core.make_parton(1, (5, 0, 5, 0))]]
    earlier = core.make_hadron(22, (1, 1, 0, 0))
    h, p = [earlier], []
    core.Hadronization.DoHadronization(Pions(), shower, h, p)
    assert [x.pid() for x in h] == [22, 211, 211] and p == []     # after what was there
    h, p = [], []
    core.Hadronization.DoHadronization(Both(), shower, h, p)
    assert [x.pid() for x in h] == [111, 111] and [x.pid() for x in p] == [21]


# ------------------------------------------------------------------------------ facade

def test_the_particle_view():
    from jetscape.particles import view

    p = core.make_parton(21, (10, 3, 4, 5), (0.5, 1, 2, 3), stat=2)
    v = view(p)
    assert v.unwrap() is p
    assert (v.pid, v.pstat, v.e, v.px, v.py, v.pz) == (21, 2, 10, 3, 4, 5)
    assert (v.pt, v.eta, v.phi, v.rapidity) == (p.pt(), p.eta(), p.phi(), p.rapidity())
    assert (v.t, v.x, v.y, v.z) == (0.5, 1, 2, 3)
    assert v.momentum == p.momentum and v.position == p.position
    assert callable(p.pt)                                  # the core class is unchanged
    h = core.make_hadron(2212, (2, 0.1, 0.2, 0.3), mass=0.938)
    assert view(h).mass == pytest.approx(0.938)
