"""Gates for CausalLiquefierSource and the YAML parton interface."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

torch = pytest.importorskip("torch")

from fast_data import fv                                                  # noqa: E402
from fast_data.liquefier import (CausalLiquefierSource, DropletFlags,     # noqa: E402
                                 LiquefierParams, partons_from_config)

P = LiquefierParams(tau_delay=0.5)


def _grid():
    return fv.Grid(25, 25, 17, 0.5, 0.5, 0.4, dtype=torch.float64)


def test_s1_every_droplet_fires_exactly_once_under_adaptive_steps():
    """The firing window is half-open and the steps tile [tau0, tau_end), so a droplet cannot
    be missed or double-counted however the solver chooses its steps."""
    g = _grid()
    rng = np.random.default_rng(1)
    for _ in range(20):
        drops = np.column_stack([rng.uniform(0.3, 1.2, 6), rng.uniform(-2, 2, 6),
                                 rng.uniform(-2, 2, 6), rng.uniform(-1, 1, 6),
                                 rng.uniform(1, 10, 6), np.zeros(6), np.zeros(6), np.zeros(6)])
        src = CausalLiquefierSource(drops, P)
        tau = 0.6
        while tau < 3.0:
            dt = float(rng.uniform(0.01, 0.2))
            src.step(tau, dt, g, None)
            tau += dt
        assert src.fired.sum() == (src.tau_dep() < tau).sum()


def test_s2_the_deposit_does_not_depend_on_the_solver_step_size():
    """The C++'s 1/dtau cancels against MUSIC's *dtau, so the deposit contains no dtau at all.
    That is what makes an adaptive CFL step safe, and it is worth a hard gate."""
    g = _grid()
    drop = np.array([[0.6, 0.0, 0.5, 0.3, 10.0, 6.0, 2.0, 3.0]])
    for dtau in (0.02, 0.05, 0.137, 0.31):
        src = CausalLiquefierSource(drop, P)
        tau, P4 = 0.6, np.zeros(4)
        while tau < 2.0:
            dq = src.step(tau, dtau, g, None)
            if src.last_dq is not None:
                # convert on the surface the deposit was expressed on
                P4 += fv.cartesian_four_momentum(dq, tau + dtau, g)[0].numpy()
            tau += dtau
        assert np.allclose(P4, drop[0, 4:8], atol=1e-12), f"dtau={dtau}: {P4}"


def test_s3_zero_droplets_is_bit_identical_to_no_source():
    g = _grid()
    eos = fv.ConformalEoS(47.5)
    q0 = fv.initial_state_from_energy(
        torch.full((1, g.nx, g.ny, g.neta), 5.0, dtype=torch.float64), 0.6, eos)
    a = fv.rollout(q0.clone(), None, None, 0.6, 1.2, g, eos, record_dtau=0.2)
    b = fv.rollout(q0.clone(), None, None, 0.6, 1.2, g, eos, record_dtau=0.2,
                   source=CausalLiquefierSource(np.zeros((0, 8)), P))
    assert torch.equal(a["q"], b["q"])


def test_s4_wake_is_linear_in_the_deposited_energy():
    g = _grid()
    eos = fv.ConformalEoS(47.5)
    q0 = fv.initial_state_from_energy(
        torch.full((1, g.nx, g.ny, g.neta), 5.0, dtype=torch.float64), 0.6, eos)
    base = fv.rollout(q0.clone(), None, None, 0.6, 1.6, g, eos, record_dtau=0.2)
    drop = np.array([[0.6, 0.0, 0.0, 0.0, 8.0, 3.0, -1.0, 2.0]])
    w = []
    for s in (1.0, 2.0):
        d = drop.copy()
        d[0, 4:8] *= s
        o = fv.rollout(q0.clone(), None, None, 0.6, 1.6, g, eos, record_dtau=0.2,
                       source=CausalLiquefierSource(d, P))
        w.append(float((o["q"][-1] - base["q"][-1]).abs().sum()))
    assert 1.9 < w[1] / w[0] < 2.1


def test_s5_bookkeeping_reports_exactly_what_was_injected():
    g = _grid()
    eos = fv.ConformalEoS(47.5)
    q0 = fv.initial_state_from_energy(
        torch.full((1, g.nx, g.ny, g.neta), 5.0, dtype=torch.float64), 0.6, eos)
    drop = np.array([[0.6, 0.0, 0.0, 0.0, 8.0, 3.0, -1.0, 2.0]])
    src = CausalLiquefierSource(drop, P)
    fv.rollout(q0, None, None, 0.6, 1.6, g, eos, record_dtau=0.2, source=src)
    r = src.report()
    assert r["n_fired"] == 1 and r["n_never_fired"] == 0
    assert np.allclose(r["P_cart_injected"], drop[0, 4:8], atol=1e-12)


def test_s6_a_droplet_past_the_end_is_flagged_not_silently_dropped():
    g = _grid()
    drop = np.array([[5.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0]])    # deposits at tau = 5.5
    src = CausalLiquefierSource(drop, P)
    tau = 0.6
    while tau < 2.0:
        src.step(tau, 0.1, g, None)
        tau += 0.1
    src.finalize(tau_end=2.0)
    assert src.report()["n_never_fired"] == 1
    assert src.flags[0] & DropletFlags.NEVER_FIRED


# ---------------------------------------------------------------- the YAML parton grammar

def test_p1_single_mapping_and_list_are_both_accepted():
    rng = np.random.default_rng(0)
    one = partons_from_config(dict(x=1.0, y=0.0, eta=0.0, tau=0.6,
                                   E=10.0, px=10.0, py=0.0, pz=0.0), rng, P)
    assert len(one) == 1
    many = partons_from_config(
        [dict(x=1.0, y=0.0, eta=0.0, tau=0.6, E=10.0, px=10.0, py=0.0, pz=0.0),
         dict(x=-1.0, y=0.0, eta=0.0, tau=0.6, E=10.0, px=-10.0, py=0.0, pz=0.0)], rng, P)
    assert len(many) == 2
    assert many.data[0, 5] == -many.data[1, 5]


def test_p2_explicit_scalars_land_exactly_and_ranges_land_inside():
    rng = np.random.default_rng(0)
    d = partons_from_config(dict(x=2.5, y=-1.25, eta=0.75, tau=0.8,
                                 E=13.0, px=1.0, py=2.0, pz=3.0), rng, P)
    assert d.data[0, :4].tolist() == [0.8, 2.5, -1.25, 0.75]
    d = partons_from_config(dict(n=200, x=[-5, 5], y=[-2, 2], eta=[-1, 1], tau=[0.6, 1.5],
                                 E=[5, 9], phi=[0, 6.283], rapidity=[-1, 1]), rng, P)
    assert len(d) == 200
    assert (d.data[:, 1] >= -5).all() and (d.data[:, 1] <= 5).all()
    assert (d.data[:, 4] >= 5).all() and (d.data[:, 4] <= 9).all()


def test_p3_massless_polar_momentum_is_exactly_lightlike():
    rng = np.random.default_rng(0)
    d = partons_from_config(dict(n=50, x=0.0, y=0.0, eta=0.0, tau=0.6,
                                 E=[5, 30], phi=[0, 6.283], rapidity=[-2, 2]), rng, P)
    E, px, py, pz = d.data[:, 4], d.data[:, 5], d.data[:, 6], d.data[:, 7]
    assert np.allclose(E ** 2 - (px ** 2 + py ** 2 + pz ** 2), 0.0, atol=1e-10)


def test_p4_back_to_back_adds_the_recoil_partner():
    rng = np.random.default_rng(0)
    d = partons_from_config(dict(x=0.0, y=0.0, eta=0.0, tau=0.6, E=20.0,
                                 phi=0.0, rapidity=0.0, back_to_back=True), rng, P)
    assert len(d) == 2
    assert np.allclose(d.data[0, 5:8], -d.data[1, 5:8])
    assert np.allclose(d.data[0, :4], d.data[1, :4])       # same production point


def test_p5_distribution_mappings_and_multiplicity():
    rng = np.random.default_rng(0)
    d = partons_from_config(dict(n=4, x={"normal": [0.0, 1.0]}, y={"choice": [-1.0, 1.0]},
                                 eta=0.0, tau=0.6, E=10.0, px=10.0, py=0.0, pz=0.0), rng, P)
    assert len(d) == 4
    assert set(np.unique(d.data[:, 2])) <= {-1.0, 1.0}


def test_p6_malformed_specs_raise_naming_the_key():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="x"):
        partons_from_config(dict(x=[1, 2, 3], y=0, eta=0, tau=0.6, E=1, px=1, py=0, pz=0), rng, P)
    with pytest.raises(ValueError, match="mix"):
        partons_from_config(dict(x=0, y=0, eta=0, tau=0.6, E=1, px=1, phi=0.0), rng, P)
    with pytest.raises(ValueError, match="unknown"):
        partons_from_config(dict(x=0, y=0, eta=0, tau=0.6, E=1, px=1, py=0, pz=0, oops=1), rng, P)
    with pytest.raises(ValueError, match="E"):
        partons_from_config(dict(x=0, y=0, eta=0, tau=0.6), rng, P)


def test_p7_holes_are_supported_and_flagged():
    rng = np.random.default_rng(0)
    d = partons_from_config(dict(x=0.0, y=0.0, eta=0.0, tau=0.6,
                                 E=-5.0, px=0.0, py=0.0, pz=0.0), rng, P)
    assert d.data[0, 4] == -5.0
    assert d.flags[0] & DropletFlags.HOLE


# ------------------------------------------------------------------ trajectories (jet wakes)
#
# A single droplet is a point explosion and radiates a spherical blast wave.  A Mach cone needs a
# source that MOVES supersonically, i.e. a train of deposits along a path -- which is what
# `trajectory:` builds.  These gates pin the kinematics of that expansion.

def _traj(**kw):
    base = dict(x=-4.0, y=0.0, eta=0.0, tau=0.58, phi=0.0, rapidity=0.0,
                dEdx=2.0, ds=0.25, length=8.0)
    base.update(kw)
    return {"trajectory": base}


def test_trajectory_deposits_are_lightlike_and_sum_to_dEdx_times_length():
    rng = np.random.default_rng(0)
    d = partons_from_config(_traj(), rng, LiquefierParams(dtau=0.02))
    a = d.data
    assert len(a) == 32                                    # length / ds
    assert a[:, 4].sum() == pytest.approx(2.0 * 8.0)       # dEdx * length
    # every deposit is a lightlike four-momentum: the energy AND the forward kick
    assert np.abs(a[:, 4] ** 2 - (a[:, 5:8] ** 2).sum(1)).max() < 1e-20


def test_trajectory_E_form_splits_the_total_evenly():
    rng = np.random.default_rng(0)
    t = _traj(); t["trajectory"].pop("dEdx"); t["trajectory"]["E"] = 16.0
    d = partons_from_config(t, rng, LiquefierParams(dtau=0.02))
    assert d.data[:, 4].sum() == pytest.approx(16.0)
    assert np.ptp(d.data[:, 4]) == pytest.approx(0.0)


def test_trajectory_path_travels_at_the_speed_of_light():
    """Positions must be the Milne image of a straight lightlike lab path -- not tau + k*ds."""
    rng = np.random.default_rng(0)
    t = _traj(eta=0.7, rapidity=1.2); t["trajectory"].pop("length"); t["trajectory"]["n"] = 6
    d = partons_from_config(t, rng, LiquefierParams(dtau=0.02))
    a = d.data
    lab_t = a[:, 0] * np.cosh(a[:, 3])
    lab_z = a[:, 0] * np.sinh(a[:, 3])
    assert np.allclose(np.diff(lab_t), 0.25)                     # ds of lab time per deposit
    assert np.allclose(np.diff(lab_z) / np.diff(lab_t), np.tanh(1.2))
    assert np.allclose(np.diff(a[:, 1]) / np.diff(lab_t), 1.0 / np.cosh(1.2))
    v = a[:, 5:8] / a[:, 4:5]
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)


def test_trajectory_back_to_back_mirrors_the_momentum_from_one_vertex():
    rng = np.random.default_rng(0)
    t = _traj(rapidity=0.8); entry = {**t, "back_to_back": True, "label": "dijet"}
    d = partons_from_config(entry, rng, LiquefierParams(dtau=0.02))
    n = len(d) // 2
    near, away = d.data[:n], d.data[n:]
    assert near[0, 0] == pytest.approx(away[0, 0])               # same production vertex
    assert near[0, 1:4] == pytest.approx(away[0, 1:4])
    assert near[0, 5:8] == pytest.approx(-away[0, 5:8])          # opposite momentum
    assert near[-1, 1] > near[0, 1] and away[-1, 1] < away[0, 1]  # and they separate
    assert d.labels[0] == "dijet" and d.labels[-1] == "dijet_away"


@pytest.mark.parametrize("bad, match", [
    (dict(n=4), "exactly one of 'n'"),                      # n AND length
    (dict(E=10.0), "exactly one of 'dEdx'"),                # dEdx AND E
    (dict(ds=0.0), "ds must be > 0"),
    (dict(tau="auto"), "must be explicit"),
    (dict(bogus=1.0), "unknown trajectory key"),
])
def test_trajectory_rejects_ambiguous_specs(bad, match):
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match=match):
        partons_from_config(_traj(**bad), rng, LiquefierParams(dtau=0.02))


def test_trajectory_entry_rejects_stray_top_level_keys():
    """`x:` beside `trajectory:` would be silently ignored, so it has to be an error."""
    rng = np.random.default_rng(0)
    entry = {**_traj(), "x": 2.0}
    with pytest.raises(ValueError, match="move"):
        partons_from_config(entry, rng, LiquefierParams(dtau=0.02))


def test_trajectory_length_is_rounded_to_whole_deposits():
    rng = np.random.default_rng(0)
    d = partons_from_config(_traj(ds=0.3, length=1.0), rng, LiquefierParams(dtau=0.02))
    assert len(d) == 3                                      # round(1.0 / 0.3)
