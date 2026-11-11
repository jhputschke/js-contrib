"""The liquefier's frame in fast_data: LiquefierParams.frame = "bjorken" (X-SCAPE's default,
the kernel in the frame of the Bjorken flow at the droplet's eta_s) or "lab"
(<CausalLiquefier><frame>lab</frame>, the liquefier's paper and X-SCAPE before 2026-11).

* the Bjorken-frame kernel is the eta_s = 0 kernel shifted by eta_d;
* the deposit (droplet_weights) of a forward droplet then looks like a midrapidity one;
* the parameters carry the frame through config, attributes and the C++ bridge;
* against the C++ (needs an X-SCAPE with the option).
"""

import numpy as np
import pytest

from fast_data.liquefier import LiquefierParams
from fast_data.liquefier import kernel as K

LAB = LiquefierParams(tau_delay=1.0, frame="lab")
BJORKEN = LiquefierParams(tau_delay=1.0)


def _points(n=400, seed=7):
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.2, 1.2, n), rng.uniform(-1.2, 1.2, n), rng.uniform(-0.7, 0.7, n)


# ── parameters ──────────────────────────────────────────────────────────────────────────
def test_frame_is_validated_and_defaults_to_bjorken():
    assert LiquefierParams().frame == "bjorken"
    with pytest.raises(ValueError):
        LiquefierParams(frame="comoving")


def test_frame_round_trips_through_attrs_and_config():
    for p in (LAB, BJORKEN):
        a = p.to_attrs()
        assert a["liquefier_frame"] == p.frame
        assert LiquefierParams.from_attrs(a) == p
        assert LiquefierParams.from_config({"tau_delay": 1.0, "frame": p.frame}) == p
    # a file from before the option: the lab frame, although the default is now bjorken
    old = {k: v for k, v in BJORKEN.to_attrs().items() if k != "liquefier_frame"}
    assert LiquefierParams.from_attrs(old).frame == "lab"


def test_eta_shift():
    assert LAB.eta_shift(2.4) == 0.0
    assert BJORKEN.eta_shift(2.4) == 2.4
    assert BJORKEN.eta_shift(float("nan")) == 0.0       # a droplet at tau = 0


# ── the kernel ──────────────────────────────────────────────────────────────────────────
def test_bjorken_kernel_is_the_midrapidity_kernel_shifted():
    tau = 2.0
    x, y, de = _points()
    ref = K.k_tau(tau, x, y, de, (1.0, 0.03, -0.05, 0.0), LAB)
    assert np.count_nonzero(ref) > 50
    for eta_d in (1.0, 2.4, -3.1):
        got = K.k_tau(tau, x, y, eta_d + de, (1.0, 0.03, -0.05, eta_d), BJORKEN)
        np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-12 * np.abs(ref).max())


def test_both_frames_agree_at_midrapidity():
    x, y, e = _points(100, seed=9)
    drop = (1.0, 0.03, -0.05, 0.0)
    np.testing.assert_array_equal(K.k_tau(2.0, x, y, e, drop, BJORKEN),
                                  K.k_tau(2.0, x, y, e, drop, LAB))


def test_jmu_keeps_the_lab_momentum_in_milne_components():
    """The frame changes the scalar kernel only: jmu / K_tau is p^tau, p^x, p^y, p^eta of the
    droplet's lab momentum at the cell's eta, in both frames."""
    drop = np.array([1.0, 0.0, 0.0, 2.4, 5.0, 0.4, -0.3, 4.9])
    tau = drop[0] + BJORKEN.tau_delay
    x, y, de = _points(200, seed=11)
    eta = drop[3] + de
    j = K.smearing_kernel_jmu(tau, x, y, eta, drop, BJORKEN)
    kt = K.k_tau(tau, x, y, eta, drop, BJORKEN) / BJORKEN.dtau
    m = np.abs(kt) > 1e-6 * np.abs(kt).max()
    E, px, py, pz = drop[4:]
    np.testing.assert_allclose(j[m, 0] / kt[m], E * np.cosh(eta[m]) - pz * np.sinh(eta[m]), rtol=1e-9)
    np.testing.assert_allclose(j[m, 3] / kt[m], pz * np.cosh(eta[m]) - E * np.sinh(eta[m]), rtol=1e-9)
    np.testing.assert_allclose(j[m, 1] / kt[m], px, rtol=1e-9)


# ── the deposit on a grid ───────────────────────────────────────────────────────────────
def test_forward_deposit_is_midrapidity_like_in_the_bjorken_frame():
    torch = pytest.importorskip("torch")
    from fast_data import fv
    from fast_data.liquefier.deposit import droplet_weights

    g = fv.Grid(41, 41, 33, 0.3125, 0.3125, 0.3125, dtype=torch.float64)
    tau_eval = 2.0
    mid = droplet_weights(np.array([1.0, 0.03, -0.05, 0.0, 1.0, 0, 0, 0]), tau_eval, g, LAB)
    for p, eta_d in ((BJORKEN, 2.4), (BJORKEN, 1.25)):
        d = np.array([1.0, 0.03, -0.05, eta_d, 1.0, 0, 0, 0])
        w = droplet_weights(d, tau_eval, g, p)
        assert w.eta_centroid == pytest.approx(eta_d, abs=0.05)
        assert w.n_raw == pytest.approx(mid.n_raw, rel=0.05)
        assert w.n_cells == pytest.approx(mid.n_cells, rel=0.35)
    lab = droplet_weights(np.array([1.0, 0.03, -0.05, 2.4, 1.0, 0, 0, 0]), tau_eval, g, LAB)
    assert lab.eta_centroid < 2.0                      # the lab frame: displaced backward


# ── against the C++ ─────────────────────────────────────────────────────────────────────
@pytest.mark.needs_xscape
def test_bjorken_frame_matches_the_cpp():
    core = pytest.importorskip("jetscape.pyjetscape_core")
    from fasthydro.liquefier_bridge import params_from_liquefier
    cpp = core.CausalLiquefier(0.02, 0.3, 0.3, 0.2)
    if not hasattr(cpp, "bjorken_frame"):
        pytest.skip("X-SCAPE without <CausalLiquefier><frame>")
    assert cpp.bjorken_frame is True                   # the default
    p = params_from_liquefier(cpp)
    assert p.frame == "bjorken"
    drop = np.array([[1.0, 0.03, -0.05, 2.1, 10.0, 1.0, -2.0, 9.6]])
    cpp.ClearTask()
    cpp.add_droplets_numpy(drop)
    tau = 1.0 + p.tau_delay
    x, y, de = _points(60, seed=13)
    n_nonzero = 0
    for xi, yi, ei in zip(x, y, drop[0, 3] + de):
        c = np.asarray(cpp.get_source(tau, xi, yi, ei), dtype=np.float64)
        q = np.asarray(K.get_source_xscape(tau, xi, yi, ei, drop, p, emulate_float32=True),
                       dtype=np.float64).ravel()
        scale = max(1e-9, np.abs(c).max())
        assert np.max(np.abs(c - q)) / scale < 2e-5, (xi, yi, ei, c, q)
        n_nonzero += int(np.abs(c).max() > 0)
    assert n_nonzero > 5
    cpp.ClearTask()


@pytest.mark.needs_xscape
def test_fluid_frame_is_refused_not_mapped():
    """X-SCAPE's <frame>fluid</frame> has no fast_data counterpart yet: the bridge refuses it
    rather than treating it as lab (bjorken_frame is False for it)."""
    core = pytest.importorskip("jetscape.pyjetscape_core")
    from fasthydro.liquefier_bridge import params_from_liquefier
    cpp = core.CausalLiquefier(0.02, 0.3, 0.3, 0.2)
    if not hasattr(cpp, "frame"):
        pytest.skip("X-SCAPE without the fluid frame")
    cpp.frame = "fluid"
    with pytest.raises(NotImplementedError):
        params_from_liquefier(cpp)
    cpp.frame = "lab"
    assert params_from_liquefier(cpp).frame == "lab"
