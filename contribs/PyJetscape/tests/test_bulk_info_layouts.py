"""
tests/test_bulk_info_layouts.py

EvolutionHistory in the flat data_vector layout that MUSIC's slim copy uses
(<slim_bulk_info>1: e, s, T, vx, vy, vz per cell): the exports read it, a field it lacks
is an error rather than zeros, and lookups return the stored values.  Needs the compiled
extension.
"""

import numpy as np
import pytest

core = pytest.importorskip("jetscape.pyjetscape_core")

SLIM = ["energy_density", "entropy_density", "temperature", "vx", "vy", "vz"]
NTAU, NX, NY, NETA = 3, 4, 5, 2


def _history(names=SLIM):
    rng = np.random.default_rng(7)
    rec = rng.uniform(0.1, 1.0, (NTAU, NX, NY, NETA, len(names))).astype(np.float32)
    h = core.EvolutionHistory()
    h.FromVector(rec.ravel().tolist(), names, 0.4, 0.2, -1.5, 1.0, NX, -2.0, 1.0, NY,
                 -0.5, 1.0, NETA, False)
    h.boost_invariant = False
    return h, {n: rec[..., i] for i, n in enumerate(names)}


def test_slim_history_exports_its_fields():
    h, f = _history()
    assert h.uses_data_vector() and h.data_info == SLIM
    assert h.get_data_size() == NTAU * NX * NY * NETA
    full = h.to_numpy_full(6)                  # e, T, vx, vy, vz, s
    for i, n in enumerate(["energy_density", "temperature", "vx", "vy", "vz",
                           "entropy_density"]):
        np.testing.assert_array_equal(full[..., i], f[n])
    frame = h.frame_numpy(1)
    for i, n in enumerate(["energy_density", "vx", "vy", "vz"]):
        np.testing.assert_array_equal(frame[..., i], f[n][1])
    np.testing.assert_array_equal(h.to_numpy(5)[..., 4], f["entropy_density"][:, :, :, 0])


def test_a_field_the_slim_copy_lacks_is_an_error():
    h, _ = _history()
    with pytest.raises(RuntimeError, match="no 'pressure'"):
        h.to_numpy(6)


def test_lookups_on_grid_nodes_return_the_stored_values():
    h, f = _history()
    k, i, j, l = 1, 2, 3, 1
    cell = h.get_fluid_cell(k, i, j, l)               # exact: no interpolation
    for n in SLIM:
        assert getattr(cell, n) == f[n][k, i, j, l], n
    assert cell.pressure == 0 and cell.bulk_Pi == 0    # not held: zero, as documented
    # interpolated at the node: the float coordinate arithmetic leaves ~1 ulp
    c = h.get(0.4 + 0.2 * k, -1.5 + i, -2.0 + j, -0.5 + l)
    for n in SLIM:
        assert getattr(c, n) == pytest.approx(float(f[n][k, i, j, l]), rel=1e-6), n


def test_data_info_set_from_python_is_resolved():
    h, f = _history(["temperature", "energy_density", "entropy_density", "vx", "vy", "vz"])
    h.data_info = ["temperature", "energy_density", "entropy_density", "vx", "vy", "vz"]
    c = h.get_fluid_cell(0, 1, 1, 0)
    assert c.temperature == f["temperature"][0, 1, 1, 0]
    assert c.energy_density == f["energy_density"][0, 1, 1, 0]


def test_clear_empties_either_layout():
    h, _ = _history()
    h.clear_up_evolution_data()
    assert h.get_data_size() == 0
    with pytest.raises(RuntimeError, match="empty"):
        h.frame_numpy(0)


def test_a_surface_from_the_slim_copy_is_refused():
    """FindSurfaceFromEvolution needs P, pi, Pi; the slim copy would give it zeros."""
    fd = core.FluidDynamics()
    fd.set_hydro_grid_info(tau_min=0.6, dtau=0.1, ntau=4, x_min=-1.0, dx=0.5, nx=5,
                           y_min=-1.0, dy=0.5, ny=5, eta_min=-0.5, deta=0.5, neta=3,
                           boost_inv=False, tau_eta_is_tz=False)
    arr = np.full((len(SLIM), 5, 5, 3, 4), 0.2, dtype=np.float32)
    fd.store_fluid_cells_from_numpy_3d(arr, SLIM)
    fd.set_hydro_status_finished()
    with pytest.raises(RuntimeError, match="slim_bulk_info"):
        fd.find_freezeout_surface(0.15, 0.0, 0.0, 0.0)
