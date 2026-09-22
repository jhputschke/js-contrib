"""The solver frame -> FluidCellInfo feature array.

Matter and LBT read `temperature` and `entropy_density`, which the solver does not carry --
they are derived here from the EoS.  Getting that silently wrong would change quenching
without changing anything visible in the stored energy density.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fast_data import fv                                            # noqa: E402
from fasthydro.cells import (DEFAULT_FIELDS, LEGAL_FIELDS,          # noqa: E402
                             frame_to_cells, validate_fields)


def _frame(nx=5, ny=7, neta=3, ntau=4, seed=0):
    rng = np.random.default_rng(seed)
    arr = np.zeros((4, nx, ny, neta, ntau), np.float32)
    arr[0] = rng.uniform(0.0, 30.0, (nx, ny, neta, ntau))
    arr[0, 0, 0, 0, 0] = 0.0          # a vacuum cell: must not divide by T
    arr[1:] = rng.uniform(-0.6, 0.6, (3, nx, ny, neta, ntau))
    return arr


def test_default_fields_are_all_legal():
    assert set(DEFAULT_FIELDS) <= set(LEGAL_FIELDS)
    assert validate_fields(DEFAULT_FIELDS) == tuple(DEFAULT_FIELDS)


def test_unknown_field_is_rejected_by_name():
    with pytest.raises(ValueError, match="nonsense"):
        validate_fields(("energy_density", "temperature", "nonsense"))


@pytest.mark.parametrize("missing", ["temperature", "energy_density"])
def test_fields_the_jet_chain_needs_cannot_be_dropped(missing):
    fields = tuple(f for f in DEFAULT_FIELDS if f != missing)
    with pytest.raises(ValueError, match=missing):
        validate_fields(fields)


def test_e_and_velocities_are_copied_bit_for_bit():
    arr = _frame()
    cells = frame_to_cells(arr, fv.ConformalEoS(dof=42.25))
    i = {n: k for k, n in enumerate(DEFAULT_FIELDS)}
    assert np.array_equal(cells[i["energy_density"]], arr[0])
    assert np.array_equal(cells[i["vx"]], arr[1])
    assert np.array_equal(cells[i["vy"]], arr[2])
    assert np.array_equal(cells[i["vz"]], arr[3])


def test_thermodynamics_match_the_eos():
    arr = _frame()
    eos = fv.ConformalEoS(dof=42.25)
    cells = frame_to_cells(arr, eos)
    i = {n: k for k, n in enumerate(DEFAULT_FIELDS)}
    e = torch.as_tensor(arr[0], dtype=torch.float64).clamp_min(0.0)
    assert np.allclose(cells[i["temperature"]], eos.T(e).numpy().astype(np.float32))
    assert np.allclose(cells[i["pressure"]], eos.p(e).numpy().astype(np.float32))
    # mu_B = 0  =>  s = (e + p)/T
    s = ((e + eos.p(e)) / eos.T(e).clamp_min(1e-6)).numpy().astype(np.float32)
    ok = arr[0] > 0
    assert np.allclose(cells[i["entropy_density"]][ok], s[ok])


def test_vacuum_cells_are_finite_and_zero():
    arr = _frame()
    cells = frame_to_cells(arr, fv.ConformalEoS(dof=42.25))
    i = {n: k for k, n in enumerate(DEFAULT_FIELDS)}
    assert np.isfinite(cells).all(), "NaN/inf leaked from the T -> 0 limit"
    assert cells[i["entropy_density"]][0, 0, 0, 0] == 0.0
    assert cells[i["temperature"]][0, 0, 0, 0] == 0.0


def test_unsupported_fields_are_stored_as_zero():
    """pi** and bulk_Pi are legal names but the solver does not export them (yet)."""
    arr = _frame()
    fields = DEFAULT_FIELDS + ("pi00", "bulk_pi")
    cells = frame_to_cells(arr, fv.ConformalEoS(dof=42.25), fields)
    assert cells.shape[0] == len(fields)
    assert np.all(cells[-1] == 0) and np.all(cells[-2] == 0)


def test_shape_is_checked():
    with pytest.raises(ValueError, match="expected"):
        frame_to_cells(np.zeros((3, 4, 4, 2, 2), np.float32), fv.ConformalEoS())
