"""Gates for the hadron side of hydro_jet_particles_pyvista.py.

Nothing renders (see test_wake_pyvista.py).  What is gated is what would be silently wrong
on screen: which oversample was drawn, where a hadron is at lab time t, when it appears, and
where the jet hadrons -- which carry no position of their own -- are made to start.
"""

import os
import sys

import numpy as np
import pytest

_VIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _VIZ not in sys.path:
    sys.path.insert(0, _VIZ)

h5py = pytest.importorskip("h5py")
hjpp = pytest.importorskip("hydro_jet_particles_pyvista")
hh5 = pytest.importorskip("jetscape.hadrons_h5")

VERTEX = (1.5, -2.0, 0.25, 0.0)          # x, y, z, t of the hard vertex


def _hadron(E, px, py, pz, x=(0.0, 0.0, 0.0, 0.0), pstat=0, pid=211):
    return pid, pstat, (E, px, py, pz), x


def _sample(rows):
    pid, pstat, p, x = zip(*rows)
    return dict(pid=np.array(pid, np.int32), pstat=np.array(pstat, np.int32),
                p=np.array(p, np.float32), x=np.array(x, np.float32))


def _initiators():
    ini = np.zeros((2, 11))
    ini[:, 7:11] = VERTEX
    ini[0, 3:7] = (10.0, 0.0, 0.0, 10.0)
    ini[1, 3:7] = (-10.0, 0.0, 0.0, 10.0)
    return ini


def _write(stem, n_bulk=4, n_frag=3, initiators=True):
    """bulk sample k holds one hadron with E = 1 + k, emitted at t = 2 at (1, 0, 0) and
    moving along +x at 0.5; fragmentation j one hadron with E = 10 + j along +y at c."""
    bulk = [_sample([_hadron(1.0 + k, 0.5 * (1.0 + k), 0.0, 0.0, x=(2.0, 1.0, 0.0, 0.0))])
            for k in range(n_bulk)]
    frag = [_sample([_hadron(10.0 + j, 0.0, 10.0 + j, 0.0),
                     _hadron(1.0, 0.0, 0.5, 0.0, pstat=-1)])      # a hole's fragment
            for j in range(n_frag)]
    with hh5.HadronH5Writer(f"{stem}_hadrons_bulk_jet.h5", tag="bulk_jet",
                            n_samples=n_bulk) as w:
        w.append_unit(bulk, unit=0, event=0)
    with hh5.HadronH5Writer(f"{stem}_hadrons_jet_frag.h5", tag="jet_frag",
                            n_samples=n_frag, initiators=initiators) as w:
        w.append_unit(frag, initiators=_initiators() if initiators else None,
                      unit=0, event=0)
    return f"{stem}.h5"


def test_fixed_sample_is_the_one_drawn(tmp_path):
    pair = _write(str(tmp_path / "prod"))
    had = hjpp.load_hadrons(pair, 0, sample=2, frag_sample=1)
    assert (had["k"], had["n_k"], had["j"], had["n_j"]) == (2, 4, 1, 3)
    assert had["bulk"]["p"][0, 0] == pytest.approx(3.0)
    assert had["frag"]["p"][0, 0] == pytest.approx(11.0)


def test_random_sample_is_reproducible_and_in_range(tmp_path):
    pair = _write(str(tmp_path / "prod"), n_bulk=50, n_frag=20)
    a = hjpp.load_hadrons(pair, 0, rng_seed=3)
    b = hjpp.load_hadrons(pair, 0, rng_seed=3)
    assert (a["k"], a["j"]) == (b["k"], b["j"])
    assert 0 <= a["k"] < 50 and 0 <= a["j"] < 20
    assert a["bulk"]["p"][0, 0] == pytest.approx(1.0 + a["k"])
    picks = {hjpp.load_hadrons(pair, 0, rng_seed=s)["k"] for s in range(20)}
    assert len(picks) > 1                               # it is not always sample 0


def test_out_of_range_sample_is_refused(tmp_path):
    pair = _write(str(tmp_path / "prod"))
    with pytest.raises(SystemExit):
        hjpp.load_hadrons(pair, 0, sample=4)


def test_no_hadron_files_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        hjpp.load_hadrons(str(tmp_path / "nothing.h5"), 0)


def test_hard_vertex_from_the_hadron_file_else_the_pair_file(tmp_path):
    pair = _write(str(tmp_path / "a"))
    had = hjpp.load_hadrons(pair, 0, sample=0, frag_sample=0)
    np.testing.assert_allclose(had["vertex"], VERTEX[:3])

    # an older hadron file without initiators/: the pair file's shower/ is used
    pair = _write(str(tmp_path / "b"), initiators=False)
    with h5py.File(pair, "w") as f:
        f.create_dataset("shower/initiators", data=_initiators())
        f.create_dataset("shower/initiator_offsets", data=np.array([0, 2]))
    had = hjpp.load_hadrons(pair, 0, sample=0, frag_sample=0)
    np.testing.assert_allclose(had["vertex"], VERTEX[:3])


def test_select_drops_holes_and_applies_the_cuts():
    h = _sample([_hadron(5.0, 3.0, 0.0, 0.0),                 # eta 0, pT 3
                 _hadron(5.0, 0.3, 0.0, 4.0),                 # eta ~ 3.3, pT 0.3
                 _hadron(5.0, 3.0, 0.0, 0.0, pstat=-1)])      # a hole
    assert hjpp.select(h).tolist() == [True, True, False]
    assert hjpp.select(h, eta_max=1.0).tolist() == [True, False, False]
    assert hjpp.select(h, pt_min=1.0).tolist() == [True, False, False]


def test_bulk_hadron_appears_at_emission_and_free_streams(tmp_path):
    pair = _write(str(tmp_path / "prod"))
    had = hjpp.load_hadrons(pair, 0, sample=0, frag_sample=0)   # E = 1, px = 0.5: v = 0.5
    tr = hjpp.bulk_tracks(had["bulk"], hjpp.select(had["bulk"]))
    pos, _ = tr.at(1.9)
    assert len(pos) == 0                                        # not yet emitted
    pos, _ = tr.at(2.0)
    np.testing.assert_allclose(pos, [[1.0, 0.0, 0.0]], atol=1e-6)
    pos, _ = tr.at(6.0)
    np.testing.assert_allclose(pos, [[1.0 + 0.5 * 4.0, 0.0, 0.0]], atol=1e-6)


def test_jet_hadrons_start_at_the_hard_vertex_and_appear_at_frag_time(tmp_path):
    pair = _write(str(tmp_path / "prod"))
    had = hjpp.load_hadrons(pair, 0, sample=0, frag_sample=0)
    keep = hjpp.select(had["frag"])
    assert keep.sum() == 1                                      # the hole is not drawn
    tr = hjpp.frag_tracks(had["frag"], keep, had["vertex"], had["t_vertex"], t_frag=5.0)
    assert len(tr.at(4.99)[0]) == 0
    pos, pT = tr.at(8.0)
    # E = 10, py = 10: along +y at c, from the vertex at t = 0
    np.testing.assert_allclose(pos, [[VERTEX[0], VERTEX[1] + 8.0, VERTEX[2]]], atol=1e-5)
    np.testing.assert_allclose(pT, [10.0])


# ── formation times (--formation-tau0) ────────────────────────────────────────────
def test_formation_time_is_tau0_E_over_m_with_the_floor():
    m_pi = hjpp.MASS[211]
    p = np.array([[1.0, 0.0, 1.0, 0.0],          # soft pion: E/m ~ 7
                  [10.0, 0.0, 10.0, 0.0],        # hard pion: E/m ~ 72
                  [5.0, 0.0, 4.9, 0.0]])         # proton: E/m ~ 5.3
    pid = np.array([211, -211, 2212])
    t = hjpp.formation_times(pid, p, t_vertex=0.5, tau0=1.0, t_floor=0.0)
    np.testing.assert_allclose(t, 0.5 + p[:, 0] / np.array([m_pi, m_pi, hjpp.MASS[2212]]))
    # E/m, not E, orders them: the 5 GeV proton forms before the 1 GeV pion
    assert t[2] < t[0] < t[1]
    t = hjpp.formation_times(pid, p, t_vertex=0.5, tau0=1.0, t_floor=10.0)
    assert t[0] == t[2] == 10.0 and t[1] > 10.0   # the floor holds the early ones


def test_photons_and_unknown_species_count_as_at_least_a_pion():
    p = np.array([[2.0, 0.0, 2.0, 0.0],                  # a photon: m = 0
                  [3.0, 0.0, 0.0, np.sqrt(9.0 - 1.5 ** 2)]])   # unknown pid, m = 1.5
    m = hjpp.hadron_mass(np.array([22, 9999999]), p)
    np.testing.assert_allclose(m, [hjpp.M_PI, 1.5], rtol=1e-9)


def test_formation_mode_on_the_tracks_and_the_parton_switch(tmp_path):
    pair = _write(str(tmp_path / "prod"))
    had = hjpp.load_hadrons(pair, 0, sample=0, frag_sample=0)
    keep = hjpp.select(had["frag"])
    tr = hjpp.frag_tracks(had["frag"], keep, had["vertex"], had["t_vertex"], t_frag=5.0,
                          tau0=1.0)
    t_on = 10.0 / hjpp.MASS[211]                  # E = 10 pion from the vertex at t = 0
    assert tr.t_on[0] == pytest.approx(t_on)
    assert len(tr.at(t_on - 0.01)[0]) == 0
    pos, _ = tr.at(t_on + 1.0)                    # on its straight line from the vertex
    np.testing.assert_allclose(pos[0, 1], VERTEX[1] + t_on + 1.0, atol=1e-5)

    assert hjpp.parton_off_time("frag", 5.0, tr) == 5.0
    assert hjpp.parton_off_time("formed", 5.0, tr) == pytest.approx(t_on)
    assert hjpp.parton_off_time("formed", 5.0, None) == 5.0
