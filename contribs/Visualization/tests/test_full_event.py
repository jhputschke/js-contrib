"""Gates for full_event_pyvista.py.

Nothing renders.  What is gated is what would be silently wrong on screen: the synthetic
nuclei (count, hard core, participants, motion), which colour class a hadron lands in,
which jet leg a jet hadron is given, the participant-plane alignment, and that no medium
is drawn before it forms -- in particular not mirrored at negative times.
"""

import os
import sys

import numpy as np
import pytest

_VIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _VIZ not in sys.path:
    sys.path.insert(0, _VIZ)

fe = pytest.importorskip("full_event_pyvista")
hp = pytest.importorskip("hydro_pyvista")


# ── synthetic nuclei ──────────────────────────────────────────────────────────

def test_nucleus_has_A_nucleons_a_hard_core_and_its_radius():
    A, R, a = fe.NUCLEI["Au"]
    pos = fe.sample_nucleus(A, R, a, np.random.default_rng(1), d_min=0.4)
    assert pos.shape == (A, 3)
    assert np.allclose(pos.mean(axis=0), 0.0, atol=1e-12)
    d2 = np.sum((pos[:, None] - pos[None]) ** 2, axis=2)
    assert np.sqrt(d2[~np.eye(A, dtype=bool)].min()) >= 0.4 - 1e-12
    # Woods-Saxon <r^2> = (3/5) R^2 + (7/5) pi^2 a^2  ->  ~5.3 fm rms for Au
    rms = np.sqrt(np.mean(np.sum(pos ** 2, axis=1)))
    assert rms == pytest.approx(np.sqrt(0.6 * R ** 2 + 1.4 * np.pi ** 2 * a ** 2), rel=0.06)


def test_same_seed_same_nucleus():
    a = fe.sample_nucleus(16, 2.6, 0.5, np.random.default_rng(5))
    b = fe.sample_nucleus(16, 2.6, 0.5, np.random.default_rng(5))
    assert np.array_equal(a, b)


def test_participants_head_on_vs_far_apart():
    rng = np.random.default_rng(2)
    A, R, a = fe.NUCLEI["Au"]
    p, t = fe.sample_nucleus(A, R, a, rng), fe.sample_nucleus(A, R, a, rng)
    _, _, wp0, wt0 = fe.collide(p, t, 0.0, 42.0)
    assert wp0.sum() + wt0.sum() > 300                 # central Au+Au: N_part ~ 340-380
    _, _, wpf, wtf = fe.collide(p, t, 30.0, 42.0)
    assert wpf.sum() + wtf.sum() == 0


def test_participant_is_within_the_black_disk_distance():
    proj = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    targ = np.array([[1.0, 0.0, 3.0]])                  # z does not matter
    _, _, wp, wt = fe.collide(proj, targ, 0.0, 42.0)   # d = sqrt(4.2/pi) = 1.156 fm
    assert wp.tolist() == [True, False] and wt.tolist() == [True]


def test_impact_parameter_is_along_phi_b():
    one = np.zeros((1, 3))
    p, t, _, _ = fe.collide(one, one, 4.0, 42.0, phi_b=np.pi / 2)
    assert np.allclose(p[0], [0.0, 2.0, 0.0], atol=1e-12)
    assert np.allclose(t[0], [0.0, -2.0, 0.0], atol=1e-12)


def test_nuclei_approach_meet_at_zero_and_participants_fade():
    proj = np.array([[0.0, 0.0, 6.0], [3.0, 0.0, 0.0]])
    targ = np.array([[0.0, 0.0, -6.0]])
    n = fe.Nuclei(proj, targ, np.array([True, False]), np.array([True]),
                  gamma=100.0, gamma_vis=10.0, t_fade=2.0)
    s = n.at(-5.0)
    assert s["proj_part"][0][0, 2] == pytest.approx(0.6 - 5.0 * n.beta)   # flattened by 10
    assert s["targ_part"][0][0, 2] == pytest.approx(-0.6 + 5.0 * n.beta)
    assert s["proj_part"][1] == 1.0
    assert n.at(1.0)["proj_part"][1] == pytest.approx(0.5)
    assert n.at(3.0)["targ_part"][1] == 0.0
    assert n.at(3.0)["proj_spec"][1] == 1.0                              # spectators fly on
    assert n.n_part == 2


def test_sigma_nn_and_b_range():
    assert fe.sigma_nn(200.0) == pytest.approx(42.0, abs=1.5)
    assert fe.sigma_nn(5020.0) == pytest.approx(68.0, abs=1.5)
    lo, hi = fe.b_range(0.0, 10.0, 6.38, 42.0)
    assert lo == 0.0 and 4.0 < hi < 5.0


def test_collision_setup_from_the_user_xml():
    xml = ("<jetscape><IS><MCGlauber><projectile> Pb </projectile><sqrts>5020</sqrts>"
           "<cenMin>20</cenMin><cenMax>30</cenMax></MCGlauber></IS></jetscape>")
    assert fe.collision_setup({"prod_user_xml": xml}) == ("Pb", 5020.0, 20.0, 30.0)
    assert fe.collision_setup({}) == ("Au", 200.0, 0.0, 10.0)


def test_psi2_is_the_short_axis_of_the_almond():
    meta = dict(x_min=-12.0, dx=0.25, y_min=-12.0, dy=0.25, eta_min=-1.0, deta=0.5)
    x = meta["x_min"] + meta["dx"] * np.arange(97)
    X, Y = np.meshgrid(x, x, indexing="ij")
    for ang in (0.0, np.pi / 6, -np.pi / 4):
        # almond elongated perpendicular to `ang` (the impact parameter direction)
        u = (X - 1.0) * np.cos(ang) + (Y + 0.5) * np.sin(ang)
        v = -(X - 1.0) * np.sin(ang) + (Y + 0.5) * np.cos(ang)
        e = np.exp(-u ** 2 / 2.0 - v ** 2 / 8.0)
        arr = np.zeros((1, 97, 97, 5, 5))
        arr[0, ..., hp.FEAT_E] = e[:, :, None]
        xc, yc, psi = fe.medium_geometry(arr, meta)
        assert (xc, yc) == pytest.approx((1.0, -0.5), abs=1e-3)
        d = np.angle(np.exp(2j * (psi - ang))) / 2
        assert abs(d) < 1e-3


# ── colour classes ────────────────────────────────────────────────────────────

def test_species_classes():
    pid = [211, -211, 111, 321, -321, 310, 130, 2212, -2212, 221, 3212, 113, 2112, 3122,
           22, 11]
    want = [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
    assert fe.species_class(pid).tolist() == want


def test_jet_hadrons_go_to_the_nearest_leg():
    ini = np.array([[50.0, 30.0, 0.0, 40.0],         # E, px, py, pz: eta ~ 1.1, phi 0
                    [30.0, -30.0, 0.0, 0.0]])        # eta 0, phi pi
    had = np.array([[5.0, 3.0, 0.3, 4.0],            # near leg 0
                    [3.0, -2.9, -0.2, 0.1],          # near leg 1 (across the phi wrap)
                    [2.0, 0.0, 1.9, -0.3]])          # between them: phi pi/2
    assert fe.assign_legs(had, ini).tolist()[:2] == [0, 1]
    assert fe.assign_legs(had, ini, dr_max=0.5).tolist() == [0, 1, -1]
    assert fe.assign_legs(had[:0], ini).shape == (0,)
    assert fe.assign_legs(had, ini[:0]).tolist() == [-1, -1, -1]


def test_tracks_switch_on_and_stream():
    p = np.array([[2.0, 1.0, 0.0, 0.0]])
    tr = fe.Tracks(p, np.array([[1.0, 0.0, 0.0]]), np.array([2.0]), np.array([2.0]),
                   np.array([211]), leg=[1])
    on, pos = tr.state(1.0)
    assert not on.any() and pos.shape == (0, 3)
    on, pos = tr.state(4.0)
    assert on.all() and np.allclose(pos, [[2.0, 0.0, 0.0]])
    assert tr.cls.tolist() == [0] and tr.leg.tolist() == [1]


# ── no medium before tau_0 ────────────────────────────────────────────────────

def test_no_medium_before_it_forms_and_none_mirrored_at_negative_t():
    meta = dict(tau_min=0.5, dtau=0.5, ntau=6, x_min=-2.0, dx=1.0, nx=5, y_min=-2.0,
                dy=1.0, ny=5, eta_min=-2.0, deta=1.0, neta=5, has_eta=True)
    arr = np.ones((6, 5, 5, 5, 5), np.float32)
    interp = hp.build_interpolator(arr, meta)
    axes = (np.linspace(-2, 2, 5), np.linspace(-2, 2, 5), np.linspace(-2, 2, 9))
    for t in (-2.0, -0.6, 0.0, 0.3):
        f = hp.cartesian_frame(interp, t, axes, meta, want_velocity=True)
        assert f["e"].max() == 0.0 and f["v"].shape == (5, 5, 9, 3)
    assert hp.cartesian_frame(interp, 2.0, axes, meta, False)["e"].max() > 0.0
