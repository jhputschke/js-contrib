"""jetcorr.deta_asym (CMS HIN-25-012's dijet-hadron observable): classes and reflection, the
mixed-event correction (a bulk that every event shares comes out flat, so R_asym - R_sym is
zero), CMS's 0.2 bins and the integral, the jackknife; and the file records of
jetcorr.weights that combine campaigns."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "example" / "analysis"))           # jetcorr

from jetcorr import deta_asym as da  # noqa: E402
from jetcorr import weights as jw  # noqa: E402


def test_gap_classes_and_reflection():
    cls, refl = da.gap_class([0.0, 0.0, 0.5, 0.2, 0.0], [0.3, -0.7, 1.9, np.nan, 2.2])
    assert cls.tolist() == [0, 1, 2, -1, -1]
    assert refl.tolist() == [True, False, True, False, True]


def test_event_classes_fractions_and_majority():
    # event 0: 3 rows in class 1 (2 reflected), 1 row without jet2; event 1: 2 rows class 0
    ev = np.array([0, 0, 0, 0, 1, 1])
    cls = np.array([1, 1, 1, -1, 0, 0])
    refl = np.array([True, True, False, False, False, True])
    f, r = da.event_classes(ev, 2, np.array([4, 2]), cls, refl, 2)
    assert np.allclose(f, [[0.0, 0.75], [1.0, 0.0]])
    assert r[0, 1] and not r[1, 0]                     # 2 of 3; a tie is no reflection


def test_class_sums_reflect_by_flipping():
    rng = np.random.default_rng(1)
    C = rng.uniform(0, 1, (3, 2, 2, da.N_DY))
    f = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    r = np.array([[False, False], [True, False], [False, True]])
    w = np.array([1.0, 2.0, 3.0])
    num, den = da.class_sums(C, w, f, r)
    assert np.allclose(num[0], C[0] + 1.0 * C[1][..., ::-1])
    assert np.allclose(num[1], 1.0 * C[1] + 3.0 * C[2][..., ::-1])
    assert np.allclose(den, [2.0, 4.0])


def bulk_events(n, rho, rng):
    """Events whose only content is a shared lab density rho (N_PT, N_ETA), seen from axes at
    random rapidities |y| < 1, binned as the pools bin their tracks."""
    y = rng.uniform(-1, 1, n)
    C = np.zeros((n, rho.shape[0], 2, da.N_DY))
    C[:, :, 0] = da.shift_density(rho, y)                # the near side
    return y, C


def test_a_shared_bulk_is_flat_after_the_mixed_event():
    rng = np.random.default_rng(2)
    rho = np.stack([1.0 + 0.3 * np.cos(da.ETAC), 0.5 + 0.1 * da.ETAC ** 2])   # not flat in eta
    y, C = bulk_events(400, rho, rng)
    eta2 = y + rng.uniform(-2, 2, len(y))                 # jet2 anywhere: every class filled
    cls, refl = da.gap_class(y, eta2)
    f, r = da.event_classes(np.arange(len(y)), len(y), np.ones(len(y)), cls, refl,
                            len(da.GAPS))
    w = rng.uniform(0.5, 2.0, len(y))
    num, den = da.class_sums(C, w, f, r)
    me, _ = da.class_sums(da.mixed_event(y, rho), w, f, r)
    yc = da.corrected(num, den, me, da.full_acceptance(1.0))
    inside = np.abs(da.DYC) < 3.3                         # every class has acceptance here
    for c in range(len(da.GAPS)):
        for p in range(2):
            v = yc[c, p, inside]
            assert np.allclose(v, v[0], rtol=1e-12)       # flat: S / ME is constant
    dif, _ = da.asym_minus_sym(yc)
    assert np.allclose(np.nan_to_num(dif[..., inside]), 0.0, atol=1e-12)
    yc2, acc = da.corrected(num, den, me, da.full_acceptance(1.0), with_acceptance=True)
    dif2, _ = da.asym_minus_sym(yc2, acceptance=acc)       # the acceptance-weighted level
    assert np.allclose(np.nan_to_num(dif2[..., inside]), 0.0, atol=1e-12)


def test_a_centrality_dependent_bulk_needs_the_matched_mixed_event():
    # two centrality classes with different multiplicities and eta shapes; the central events
    # sit preferentially in the large-gap classes: one pooled density leaves a residual in
    # R_asym - R_sym, the density of each event's own class removes it exactly
    rng = np.random.default_rng(4)
    rho = np.stack([np.stack([1.0 + 0.0 * da.ETAC]),                      # flat, peripheral
                    np.stack([3.0 * (1.0 - 0.05 * da.ETAC ** 2)])])         # narrower, central
    n = 800
    y = rng.uniform(-1, 1, n)
    cent = rng.uniform(0, 1, n) < 0.5
    gap = np.where(cent, rng.uniform(1.0, 2.0, n), rng.uniform(0.0, 1.0, n))
    eta2 = y + np.where(rng.uniform(0, 1, n) < 0.5, -gap, gap)
    k = cent.astype(np.int64)
    C = np.zeros((n, 1, 2, da.N_DY))
    for c in (0, 1):
        C[k == c, :, 0] = da.shift_density(rho[c], y[k == c])
    cls, refl = da.gap_class(y, eta2)
    f, r = da.event_classes(np.arange(n), n, np.ones(n), cls, refl, len(da.GAPS))
    w = np.ones(n)
    num, den = da.class_sums(C, w, f, r)
    inside = np.abs(da.DYC) < 3.3

    def dif(K):
        me, _ = da.class_sums(K, w, f, r)
        yc, acc = da.corrected(num, den, me, da.full_acceptance(1.0), with_acceptance=True)
        return da.asym_minus_sym(yc, acceptance=acc)[0][..., inside]
    pooled = rho[0] * (k == 0).sum() + rho[1] * (k == 1).sum()
    assert np.nanmax(np.abs(dif(da.mixed_event(y, pooled)))) > 1e-3       # the mismatch
    assert np.allclose(np.nan_to_num(dif(da.mixed_event_matched(y, rho, k))), 0.0, atol=1e-12)
    assert np.allclose(da.mixed_event_matched(y, rho[:1], np.zeros(n, np.int64)),
                       da.mixed_event(y, rho[0]))                         # one class
    q = da.quantile_classes(np.arange(100.0), 4)
    assert np.bincount(q).tolist() == [25, 25, 25, 25]
    assert not da.quantile_classes([3.0, 1.0], 1).any()


def test_a_wake_survives_and_the_geometric_acceptance_is_exact():
    rng = np.random.default_rng(3)
    y = rng.uniform(-1, 1, 200)
    # a dip of depth 1 per unit Delta y at Delta y in (-1.3, -0.7): inside every trigger's
    # acceptance (|Delta y| < 2.4 - 1)
    C = np.zeros((len(y), 1, 2, da.N_DY))
    dip = (da.DYC > -1.3) & (da.DYC < -0.7)
    C[:, 0, 0, dip] = -1.0 * da.WIDTH
    f = np.ones((len(y), 1))
    r = np.zeros((len(y), 1), bool)
    num, den = da.class_sums(C, np.ones(len(y)), f, r)
    me, _ = da.class_sums(da.mixed_event(y, da.flat_density(1)), np.ones(len(y)), f, r)
    yc = da.corrected(num, den, me, da.full_acceptance(1.0))
    assert np.allclose(yc[0, 0, dip], -1.0)
    assert np.allclose(np.nan_to_num(yc[0, 0, ~dip]), 0.0)
    assert np.isclose(da.integral(yc[0, 0], -1.3, -0.7), -0.6)


def test_cms_bins_and_integral():
    x = np.arange(da.N_DY, dtype=float)
    v, c = da.rebin(x)
    assert len(c) == 23 and np.isclose(c[0], -2.4) and np.isclose(c[-1], 2.0)
    assert np.isclose(v[0], 0.5 * (x[23] + x[24]))       # [-2.5, -2.3] = fine bins 23, 24
    ones = np.ones(da.N_DY)
    assert np.isclose(da.integral(ones), 2.0)             # -2.5 ... -0.5
    v4, c4 = da.rebin(x, -4.8, 4.8, 0.4)                  # the Z-hadron grid
    assert len(c4) == 24 and np.isclose(v4[0], x[:4].mean())


def test_jackknife_of_a_mean():
    rng = np.random.default_rng(4)
    x = rng.normal(0, 1, 1000)
    block = np.repeat(np.arange(100), 10)
    v, e = da.jackknife(lambda keep: np.array([x[keep].mean()]), block)
    assert np.isclose(v[0], x.mean())
    assert 0.5 < e[0] / (x.std() / np.sqrt(len(x))) < 1.5


def record(name, n_events, units, n_acc, sig, n_stored):
    nw = len(n_acc)
    return {"name": name, "n_events": n_events, "vacuum": False, "units": np.array(units),
            "pthat_lo": np.arange(nw, dtype=float), "pthat_hi": np.arange(1, nw + 1, dtype=float),
            "sigma_gen": np.array(sig, float), "sigma_err": 0.1 * np.array(sig, float),
            "n_accepted": np.array(n_acc, float), "n_stored": np.array(n_stored)}


def test_campaign_ids_and_windows_follow_the_file_names():
    a = record("c01_seed0004", 48, [0, 1], [10, 20], [2.0, 1.0], [24, 24])
    b = record("c00_seed0001", 24, [0], [30, 10], [4.0, 3.0], [12, 12])
    ids = jw.campaign_ids([a, b])
    assert ids["order"] == [1, 0]                         # c00 first, as the reader orders
    assert ids["event_offset"].tolist() == [24, 0]
    assert ids["units"][1] == {0: 0} and ids["units"][0] == {0: 1, 1: 2}
    sig, err, nk = jw.combine_windows([a, b])
    assert np.allclose(sig, [(30 * 4 + 10 * 2) / 40, (10 * 3 + 20 * 1) / 30])
    assert nk.tolist() == [36, 36]
    assert np.allclose(jw.window_weights([a, b]), sig / 36)
