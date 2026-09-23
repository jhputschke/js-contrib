"""`evolve_event`'s time axis: a fixed substep must not leave a residual step behind.

The physics is in one line of `fv.strang_step`::

    dudtau = (prim_final["u"] - prim["u"]) / dtau

a difference of two *Newton-recovered* velocities divided by the step.  A residual step of
1e-9 fm/c against a nominal 0.02 therefore amplifies recovery noise by 1e7, feeds it to the
Israel-Stewart shear source, and kills the run a few frames later -- while the ideal sector,
which never consumes ``dudtau``, sails through.  That is what made it look like an IS
instability for as long as it did (``RESULTS_fv_vs_music.md``).

These tests pin the two halves of the fix: crumb-sized residuals are absorbed into the step
before them, and anything bigger than a crumb is refused before the first step rather than
turned into noise.  ~20 s, no data files.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

torch = pytest.importorskip("torch")

from fast_data import fv as _fv                                      # noqa: E402
from fast_data.evolve import (EvolutionDiverged, _diagnosis,         # noqa: E402
                              _substeps_per_frame, evolve_event)

DT = torch.float64
DEV = torch.device("cpu")


def _fireball(nx=10, ny=10, neta=6, dx=0.5, deta=0.5, tau0=0.58, e_max=30.0):
    """A smooth off-centre blob -- enough of a fluid to have a d_tau u worth differencing."""
    grid = _fv.Grid(nx, ny, neta, dx, dx, deta, device=DEV, dtype=DT)
    X, Y, E = grid.mesh()                    # (B, C, X, Y, Z) broadcasting axes
    e = e_max * torch.exp(-((X - 0.3) ** 2 + (Y + 0.2) ** 2) / 6.0 - (E / 2.5) ** 2) + 1e-3
    e = e.expand(1, 1, nx, ny, neta).squeeze(1).contiguous()          # -> (B, X, Y, Z)
    eos = _fv.ConformalEoS(dof=42.25)
    q = _fv.initial_state_from_energy(e, tau0, eos)
    pi = torch.zeros(1, 10, nx, ny, neta, dtype=DT, device=DEV)
    Pi = torch.zeros(1, 1, nx, ny, neta, dtype=DT, device=DEV)
    return q, pi, Pi, grid, eos


def _f32_axis(tau0=0.58, dtau=0.1, n=11):
    """The axis MUSIC's native store hands back: float32 crumbs, promoted to float64.

    ``0.5799999833106995 + k*0.10000000149011612``.  Note ``round(x, 9)`` does NOT clean this
    up -- ``round(0.5799999833106995, 9) == 0.579999983`` -- which is how a "tau grid snapped
    so it divides exactly" row once appeared to rule the crumb out when it had not.
    """
    t0, dt = float(np.float32(tau0)), float(np.float32(dtau))
    return [t0 + k * dt for k in range(n)], dt


# ------------------------------------------------------------------ the residual step itself


def test_f32_axis_takes_no_stub_substep():
    """The invariant: every substep is the one that was asked for, to within `_ABSORB`.

    Before the fix this run's smallest substep was ~1.5e-9 fm/c -- at every one of the ten
    frame boundaries -- against a nominal 0.02.
    """
    tau_grid, _ = _f32_axis()
    q, pi, Pi, grid, eos = _fireball()
    diag = evolve_event(q, pi, Pi, tau_grid, grid, eos,
                        transport=_fv.Transport(eta_over_s=0.08, zeta_over_s=0.0),
                        hydro_dtau=0.02, freezeout="never", zero_tail=False,
                        stop_at_freezeout=False)
    assert np.isfinite(diag["arr"]).all()
    assert diag["dtau_min"] >= 0.02 * (1.0 - _fv.DTAU_ABSORB)
    assert diag["dtau_max"] <= 0.02 * (1.0 + _fv.DTAU_ABSORB)
    assert diag["n_steps"] == 5 * (len(tau_grid) - 1)     # exactly 5 per frame, none left over


def test_exact_axis_is_unchanged_by_the_absorption():
    """A config-built axis already divides exactly, so the fix must be a no-op there."""
    q, pi, Pi, grid, eos = _fireball()
    exact = [0.58 + k * 0.1 for k in range(11)]
    a = evolve_event(q, pi, Pi, exact, grid, eos, hydro_dtau=0.02,
                     freezeout="never", zero_tail=False, stop_at_freezeout=False)
    poisoned, _ = _f32_axis()
    b = evolve_event(q, pi, Pi, poisoned, grid, eos, hydro_dtau=0.02,
                     freezeout="never", zero_tail=False, stop_at_freezeout=False)
    assert a["n_steps"] == b["n_steps"] == 50
    # Same number of steps of the same size: the two runs differ only by the crumb itself.
    assert np.allclose(a["arr"], b["arr"], rtol=1e-6, atol=0)


def test_ideal_sector_is_indifferent():
    """Ideal never consumes d_tau u, so it survived the bug -- and must survive the fix."""
    tau_grid, _ = _f32_axis(n=6)
    q, pi, Pi, grid, eos = _fireball()
    d = evolve_event(q, pi, Pi, tau_grid, grid, eos, hydro_dtau=0.02,
                     freezeout="never", zero_tail=False, stop_at_freezeout=False)
    assert np.isfinite(d["arr"]).all() and d["v_max"] < 1.0


# ------------------------------------------------------------------ the up-front axis check


def test_non_dividing_substep_is_refused_before_the_first_step():
    """0.03 into 0.1 leaves a third of a step over -- too big to absorb, so refuse it."""
    q, pi, Pi, grid, eos = _fireball()
    with pytest.raises(ValueError) as e:
        evolve_event(q, pi, Pi, [0.58 + k * 0.1 for k in range(4)], grid, eos, hydro_dtau=0.03,
                     freezeout="never", stop_at_freezeout=False)
    msg = str(e.value)
    assert "0.03" in msg and "d_tau u" in msg            # it names both numbers and the reason
    assert "hydro_dtau=None" in msg                      # and the way out


def test_substeps_per_frame_accepts_a_crumb_and_counts():
    tau_grid, _ = _f32_axis(n=4)
    assert _substeps_per_frame(tau_grid, 0.02) == [5, 5, 5]
    assert _substeps_per_frame([1.0, 1.5, 2.0], 0.1) == [5, 5]
    with pytest.raises(ValueError):
        _substeps_per_frame([1.0, 1.5, 1.77], 0.1)       # the second interval is the bad one


def test_adaptive_cfl_needs_no_check_and_leaves_no_stub():
    """hydro_dtau=None was never the bug, and is not made to pay for it either."""
    tau_grid, _ = _f32_axis(n=5)
    q, pi, Pi, grid, eos = _fireball()
    d = evolve_event(q, pi, Pi, tau_grid, grid, eos,
                     transport=_fv.Transport(eta_over_s=0.08, zeta_over_s=0.0),
                     hydro_dtau=None, freezeout="never", zero_tail=False,
                     stop_at_freezeout=False)
    assert np.isfinite(d["arr"]).all()
    assert d["dtau_min"] > 1e-3                          # no crumb-sized step anywhere


# ------------------------------------------------------------------ parity with fv.rollout


def test_evolve_event_matches_rollout_record_for_record():
    """The module docstring calls the loop a line-for-line copy of `rollout`.  Hold it to that.

    Both now absorb a residual step into the one before it, so this also pins the two copies of
    that rule together: change one and this fails.
    """
    from fast_data.convert import q_to_fno_frame
    q, pi, Pi, grid, eos = _fireball()
    rec, n = 0.1, 6
    tau_grid = [0.58 + k * rec for k in range(n)]
    tr = _fv.Transport(eta_over_s=0.08, zeta_over_s=0.0)

    a = evolve_event(q, pi, Pi, tau_grid, grid, eos, transport=tr, freezeout="never",
                     zero_tail=False, stop_at_freezeout=False)["arr"]
    out = _fv.rollout(q, pi, Pi, tau_grid[0], tau_grid[-1], grid, eos, transport=tr,
                      record_dtau=rec)
    assert len(out["tau"]) == n
    for k in range(n):
        f, _ = q_to_fno_frame(out["q"][k], out["pi"][k], out["Pi"][k], out["tau"][k], grid, eos)
        got = f[0].detach().cpu().numpy().astype(np.float32)
        assert np.array_equal(got, a[..., k]), f"frame {k} differs"


# ------------------------------------------------------------------ what the failure says


def test_diagnosis_blames_the_substep_not_freezeout():
    """The freeze-out text is confidently wrong here, and acting on it hides the cause."""
    stub = _diagnosis(False, dtau=1.0e-9, nominal=0.02)
    assert "substep" in stub and "1.0e-09" in stub and "0.02" in stub
    assert "freeze-out" not in stub
    # and it still says the right thing when the stepping was uniform
    assert "freeze-out" in _diagnosis(False, dtau=0.02, nominal=0.02)
    assert "Landau match" in _diagnosis(True, dtau=0.02, nominal=0.02)


def test_check_frame_still_refuses_bad_data():
    from fast_data.evolve import _check_frame
    bad = np.zeros((4, 2, 2, 2), dtype=np.float64)
    bad[1] = 2.0                                          # |v| > 1
    with pytest.raises(EvolutionDiverged, match="spacelike"):
        _check_frame(bad, 3, 1.58)
    bad[1] = np.nan
    with pytest.raises(EvolutionDiverged, match="non-finite"):
        _check_frame(bad, 3, 1.58)
