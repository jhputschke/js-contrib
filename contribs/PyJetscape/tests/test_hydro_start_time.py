"""
tests/test_hydro_start_time.py

FluidDynamics' hydro start time is what Matter and LBT use as their tStart (the start of the
medium interaction, GetHydroTau0Signal), so it must be defined from the moment a hydro module
exists: 0 until the module sets it (MpiMusic: MUSIC's tau0 after each run, X-SCAPE branch
tstart_hydro_tau0).  Before, the members were left uninitialised, and a module made in memory
another one had freed inherited that one's value.  Needs the compiled extension.
"""

import pytest

core = pytest.importorskip("jetscape.pyjetscape_core")


def test_a_new_hydro_module_starts_at_zero():
    for _ in range(5):
        m = core.create_module("MUSIC")
        assert m.GetHydroStartTime() == 0.0
        assert m.GetHydroEndTime() == 0.0
        # dirty the memory this module leaves behind for the next one (the base class's
        # setter: no MUSIC object exists before Init)
        core.FluidDynamics.SetHydroStartTime(m, 3.7)
        assert m.GetHydroStartTime() == pytest.approx(3.7)
        del m
