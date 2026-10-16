"""zjet_wake.cms_dphi: CMS's Delta phi convention (2507.09307) on a binned yield."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "python"))
sys.path.insert(0, str(HERE.parent / "example" / "analysis"))

import zjet_wake as zw  # noqa: E402

EDGES = np.linspace(-np.pi / 2, 3 * np.pi / 2, 21)          # CMS: pi / 10
C = 0.5 * (EDGES[1:] + EDGES[:-1])
W = np.diff(EDGES)


def test_flat_yield_is_zero_and_shape_survives():
    v, e = zw.cms_dphi(np.full(20, 3.0), np.full(20, 0.4), EDGES)
    assert np.allclose(v, 0.0)
    assert np.allclose(e, 0.5 * np.hypot(0.4, 0.4) / W)
    # a peak at pi stays a peak at pi, the average goes
    shape = 1.0 + np.cos(C - np.pi)
    v, _ = zw.cms_dphi(shape * W, np.zeros(20), EDGES)
    assert np.allclose(v, np.cos(C - np.pi), atol=1e-12)


def test_merges_plus_and_minus_dphi():
    h = np.zeros(20)
    h[5] = 2.0                                     # the bin at +pi/20 only
    v, _ = zw.cms_dphi(h, np.zeros(20), EDGES)
    i_neg = int(np.argmin(np.abs(C + C[5])))       # its mirror at -pi/20
    assert np.isclose(v[5], v[i_neg])
    assert np.isclose(v[5] - v[0], 1.0 / W[5])     # half the count in each, per radian


def test_leading_axes_and_bad_bins():
    h = np.random.default_rng(1).random((3, 20))
    v, e = zw.cms_dphi(h, np.ones((3, 20)), EDGES)
    assert v.shape == e.shape == (3, 20)
    assert np.allclose(v.mean(axis=-1), 0.0)
    with pytest.raises(ValueError):
        zw.cms_dphi(np.ones(5), np.ones(5), np.linspace(-np.pi / 2, 3 * np.pi / 2, 6))
