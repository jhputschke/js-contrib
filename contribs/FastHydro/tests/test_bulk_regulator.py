"""The opt-in bound on fast_data's bulk pressure (fasthydro/bulk_regulator.py).

With zeta/s > 0, fast_data's frozen corona cells keep a stale Pi while p falls under it, so
p + Pi < 0 and the event diverges.  The regulator clamps Pi/p after every viscous step.  These
check that it clamps exactly that, leaves pi alone, is off unless asked for, and that the
config accepts and rejects what it should.
"""

import pytest
import torch

from fast_data import fv
from fast_data.config import ConfigError

from fasthydro import bulk_regulator
from fasthydro.config import DEFAULTS as FH_DEFAULTS
from fasthydro.config import resolve


@pytest.fixture
def stub(monkeypatch):
    """A fake fv.viscous_step returning a fixed (pi, Pi); module state restored afterwards."""
    pi = torch.randn(1, 10, 2, 2, 3)
    Pi = torch.tensor([-2.0, -0.95, -0.5, 0.0, 0.2, 0.5]).reshape(1, 1, 2, 1, 3).repeat(1, 1, 1, 2, 1)

    def fake(pi10, Pi_in, prim, *a, **kw):
        return pi.clone(), Pi.clone()

    monkeypatch.setattr(fv, "viscous_step", fake)
    monkeypatch.setattr(bulk_regulator, "_ORIG", None)
    monkeypatch.setattr(bulk_regulator, "_BOUNDS", None)
    return fake, pi, Pi


def _prim(p=1.0):
    return {"p": torch.full((1, 2, 2, 3), p)}


def test_clamps_pi_over_p_and_leaves_shear(stub):
    fake, pi, Pi = stub
    assert bulk_regulator.install((-0.9, 0.3)) == (-0.9, 0.3)
    pi_new, Pi_new = fv.viscous_step(None, None, _prim(1.0))   # Pi runs -2 .. 0.5: both bounds bite
    ratio = Pi_new / 1.0
    assert float(ratio.min()) == pytest.approx(-0.9)
    assert float(ratio.max()) == pytest.approx(0.3)
    inside = (Pi > -0.9) & (Pi < 0.3)
    assert torch.equal(Pi_new[inside], Pi[inside])          # untouched where already in range
    assert torch.equal(pi_new, pi)                          # shear is not the regulator's business


def test_idempotent_and_removable(stub):
    fake, _, _ = stub
    bulk_regulator.install((-0.9, 0.3))
    bulk_regulator.install((-0.8, 0.2))                     # re-wraps the ORIGINAL, not the wrapper
    assert fv.viscous_step.bulk_clamp == (-0.8, 0.2)
    _, Pi_new = fv.viscous_step(None, None, _prim(1.0))
    assert float(Pi_new.min()) == pytest.approx(-0.8)
    assert bulk_regulator.install(None) is None
    assert fv.viscous_step is fake


def test_off_by_default(stub):
    fake, _, _ = stub
    cfg = {"transport": {"mode": "israel_stewart", "zeta_over_s": 0.0},
           "fasthydro": resolve({})}
    assert FH_DEFAULTS["hydro"]["bulk_clamp"] is None
    assert bulk_regulator.from_cfg(cfg) is None
    assert fv.viscous_step is fake


def test_warns_on_bulk_without_clamp(stub, capsys):
    cfg = {"transport": {"mode": "israel_stewart", "zeta_over_s": 0.12},
           "fasthydro": resolve({})}
    bulk_regulator.from_cfg(cfg)
    assert "bulk_clamp" in capsys.readouterr().out


def test_config_parses_and_validates():
    assert resolve({"hydro": {"bulk_clamp": [-0.9, 0.3]}})["hydro"]["bulk_clamp"] == [-0.9, 0.3]
    assert resolve({"hydro": {"bulk_clamp": "[-0.9, 0.3]"}})["hydro"]["bulk_clamp"] == [-0.9, 0.3]
    for bad in ([0.1, 0.3], [-0.9], "nonsense"):
        with pytest.raises(ConfigError):
            resolve({"hydro": {"bulk_clamp": bad}})
