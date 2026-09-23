"""Run the promoted drafts' own physics self-tests, so a regression in the solver or the Glauber
initial state is caught by physics rather than by a diff against the frozen drafts."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

torch = pytest.importorskip("torch")

from fast_data import fv, glauber                      # noqa: E402


@pytest.fixture(autouse=True)
def _restore_default_dtype():
    """Put `torch.set_default_dtype` back the way we found it.

    The default dtype is process-wide, and pytest runs the whole suite in one process, so a
    gate that leaves it at float64 is not a local choice -- it silently re-types every model
    built after it.  `test_windowing.py` and `test_workflow_model.py` are the ones that
    noticed: twelve of their tests failed in a full-suite run and passed in isolation, which
    is the signature of leaked global state rather than of a bug in them.

    `fv.run_selftests` already saves and restores for exactly this reason; this is the same
    guarantee for the gates this file calls directly, and it covers any added later.
    """
    previous = torch.get_default_dtype()
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("name", ["recovery", "telescoping", "eta_sector"])
def test_cheap_solver_selftests(name):
    """The three fastest of the solver's fourteen gates: Landau round trip, exact telescoping
    conservation, and the eta sector with u^eta != 0."""
    torch.set_default_dtype(torch.float64)   # restored by _restore_default_dtype
    dev, dt = torch.device("cpu"), torch.float64
    fn = {"recovery": fv.test_recovery_roundtrip,
          "telescoping": fv.test_telescoping,
          "eta_sector": fv.test_uniform_cartesian_flow}[name]
    assert fn(dev, dt)


@pytest.mark.skipif("FAST_DATA_FULL_SELFTEST" not in os.environ,
                    reason="set FAST_DATA_FULL_SELFTEST=1 for the full suite (Gubser + wake, ~1 min)")
def test_full_solver_selftests():
    assert fv.run_selftests("cpu")


def test_glauber_selftest(capsys):
    glauber._selftest()
    out = capsys.readouterr().out
    assert "Glauber" in out
    assert "regenerated from stored seeds: max |difference| = 0.0e+00" in out


def test_gubser_writes_the_same_schema(tmp_path):
    """gubser.py writes through the shared writer, so its round trip is an independent check
    that the schema is right."""
    from fast_data import gubser
    assert gubser.selftest(verbose=False, tmpdir=tmp_path)
