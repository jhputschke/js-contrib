"""`transport.Pi_p_bounds`: the bulk pressure is held to Pi/p in [lo, hi] after every step.

Bulk viscosity first ran nonzero on 0-10% Au+Au, and without a bound events diverged (within a
couple of events at zeta/s = 0.04, ~1 in 30 at 0.12): a corona cell frozen below `pi_e_min`
keeps its Pi while its p falls, Pi/p runs to -6, p + Pi < 0.  `pi_rho_max * (e + p)` does not
catch it -- for a conformal fluid that bound is 4 p.  These pin the fix: a frozen, stale Pi
is brought back to [-0.9, 0.3] p; the bound is exactly inert at zeta = 0; and the config
accepts and rejects what it should.  ~5 s, no data.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

torch = pytest.importorskip("torch")

from fast_data import fv as _fv                                      # noqa: E402
from fast_data.config import DEFAULTS, ConfigError, load_config      # noqa: E402
from fast_data.evolve import evolve_event                            # noqa: E402

DT = torch.float64
DEV = torch.device("cpu")
TAU0 = 0.58


def _fireball(nx=10, ny=10, neta=6, dx=0.5, deta=0.5, e_max=30.0):
    grid = _fv.Grid(nx, ny, neta, dx, dx, deta, device=DEV, dtype=DT)
    X, Y, E = grid.mesh()
    e = e_max * torch.exp(-((X - 0.3) ** 2 + (Y + 0.2) ** 2) / 6.0 - (E / 2.5) ** 2) + 1e-3
    e = e.expand(1, 1, nx, ny, neta).squeeze(1).contiguous()
    eos = _fv.ConformalEoS(dof=42.25)
    q = _fv.initial_state_from_energy(e, TAU0, eos)
    pi = torch.zeros(1, 10, nx, ny, neta, dtype=DT, device=DEV)
    Pi = torch.zeros(1, 1, nx, ny, neta, dtype=DT, device=DEV)
    return q, pi, Pi, grid, eos


def _stale_corona_step(bounds):
    """One viscous step on a fluid whose every cell is frozen with a stale Pi = -3 p.

    pi_e_min above every e freezes all cells, which is the corona's situation: Pi is held,
    and only the bounds act on it.  -> Pi/p after the step.
    """
    q, pi, Pi, grid, eos = _fireball()
    prim = _fv.primitive_recovery(q, pi, Pi, TAU0, eos)
    stale = (-3.0 * prim["p"]).unsqueeze(1)
    tr = _fv.Transport(eta_over_s=0.08, zeta_over_s=0.12, pi_e_min=1e6, Pi_p_bounds=bounds)
    _, Pi_new = _fv.viscous_step(pi[:, :10], stale, prim, None, TAU0, 0.01, grid, eos, tr)
    return Pi_new[:, 0] / prim["p"]


def test_stale_frozen_pi_is_bounded():
    r = _stale_corona_step((-0.9, 0.3))
    assert float(r.min()) == pytest.approx(-0.9)
    assert float(r.max()) <= 0.3 + 1e-12


def test_without_bounds_the_stale_pi_survives():
    # the failure mode itself: pi_rho_max (e + p) = 4 p for a conformal fluid lets -3 p through
    r = _stale_corona_step(None)
    assert float(r.min()) == pytest.approx(-3.0)


def test_inert_at_zero_zeta():
    """Pi is identically zero at zeta = 0, so the bound must not change a single bit."""
    q, pi, Pi, grid, eos = _fireball()
    taus = [TAU0 + 0.1 * k for k in range(5)]
    arrs = []
    for bounds in ((-0.9, 0.3), None):
        tr = _fv.Transport(eta_over_s=0.08, zeta_over_s=0.0, Pi_p_bounds=bounds)
        arrs.append(evolve_event(q.clone(), pi.clone(), Pi.clone(), taus, grid, eos,
                                 transport=tr, freezeout="never", zero_tail=False,
                                 stop_at_freezeout=False)["arr"])
    assert (arrs[0] == arrs[1]).all()


def test_config_default_and_validation(tmp_path):
    assert DEFAULTS["transport"]["Pi_p_bounds"] == [-0.9, 0.3]
    cfg = tmp_path / "c.yaml"
    for bad in ("[0.1, 0.3]", "[-0.9]", "'nonsense'"):
        cfg.write_text(f"transport:\n  Pi_p_bounds: {bad}\n")
        with pytest.raises(ConfigError):
            load_config(str(cfg))
    cfg.write_text("transport:\n  Pi_p_bounds: null\n")
    assert load_config(str(cfg))["transport"]["Pi_p_bounds"] is None
