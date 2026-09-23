"""The gates that need a built pyjetscape_core.

These cover the failure modes that are invisible downstream:

  * GetHydroInfo was a STUB that never assigned the pointer -- Matter, LBT and
    LiquefierBase::filter_partons all dereference it unchecked, so the old behaviour was a
    null dereference, and a wrong-but-non-null one would silently change quenching.
  * store_fluid_cells_from_numpy is 2+1D only; the 3+1D writer must use EvolutionHistory's
    own record order or the fireball comes back transposed and still looks plausible.
  * boost_invariant must be False or get_tz() overwrites vz with z/t.
  * zero droplets must reproduce the background bit for bit, or no jet/no-jet pair means
    anything.
"""

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.needs_xscape

from fast_data.config import DEFAULTS                                      # noqa: E402
from fast_data.liquefier import CausalLiquefierSource, LiquefierParams     # noqa: E402
from fast_data.liquefier.droplets import DropletArray                      # noqa: E402


def _cfg(**over):
    c = copy.deepcopy(DEFAULTS)
    c["grid"].update(nx=17, ny=13, neta=7, dx=0.6, dy=0.5, deta=0.4)   # asymmetric on purpose
    c["time"].update(tau0=0.6, record_dtau=0.2, choose_ntau=9)
    c["initial_state"].update(proj="Au", targ="Au", b=2.0, K=None,
                              target_T=0.45, calib_events=3)
    c["eos"].update(kind="conformal", dof=42.25)
    c["run"].update(device="cpu", dtype="float64", seed=99)
    c["output"].update(stop_at_freezeout=False)
    # the fasthydro-only block; normally added by fasthydro.config.load_config
    from fasthydro.config import resolve as _fh_resolve
    c["fasthydro"] = _fh_resolve({})
    for k, v in over.items():
        c[k].update(v)
    return c


@pytest.fixture(scope="module")
def evolved():
    from fasthydro.hydro import FastHydro
    from fasthydro.initial_state import FastGlauberInitialState

    cfg = _cfg()
    ini = FastGlauberInitialState(cfg, seed=99, verbose=False)
    ini.Exec()
    h = FastHydro(cfg, stage=1, ic=ini, verbose=False)
    h.InitializeHydro(None)
    h.EvolveHydro()
    return cfg, ini, h


# ── the initial condition ────────────────────────────────────────────────────

def test_ic_round_trips_through_the_framework(evolved):
    cfg, ini, h = evolved
    got = (ini.GetXSize(), ini.GetYSize(), ini.GetZSize())
    assert got == (h.g.nx, h.g.ny, h.g.neta)
    back = ini.get_entropy_density_numpy_3d()
    assert back.shape == ini.e0.shape
    assert np.array_equal(back, ini.e0), "IC was reindexed crossing the boundary"


def test_binary_collision_density_is_set(evolved):
    """Without it SampleABinaryCollisionPoint puts every shower at the fireball centre."""
    _, ini, _ = evolved
    assert ini.hard_vertex_mode == "ncoll"
    assert ini.ncoll_density is not None
    assert ini.ncoll_density.sum() > 0


def _sample(ini, n=1500):
    return np.array([ini.sample_binary_collision_point()[1:3] for _ in range(n)])


def test_drawn_vertices_follow_the_density_we_handed_over(evolved):
    """The end of the chain: what the framework actually draws must be our density, not the
    origin. Compared against the density's own moments, within the sampling error."""
    from fasthydro.hard_vertex import node_axes

    _, ini, _ = evolved
    pts = _sample(ini)
    xs, ys = node_axes(ini.g)
    w = ini.ncoll_density[:, :, 0] / ini.ncoll_density[:, :, 0].sum()
    ex = float((w.sum(1) * xs).sum())
    sx = float(np.sqrt((w.sum(1) * (xs - ex) ** 2).sum()))
    tol = 5 * sx / np.sqrt(len(pts))            # 5 sigma on the mean
    assert pts[:, 0].mean() == pytest.approx(ex, abs=tol)
    assert pts[:, 0].std() == pytest.approx(sx, rel=0.15)
    assert len(np.unique(pts, axis=0)) > 50, "vertices are not actually spread out"


def test_centre_mode_reproduces_the_old_behaviour():
    """Every vertex at the origin -- what happened before the setter existed. Kept as a
    deliberate option, so it must keep working."""
    from fasthydro.initial_state import FastGlauberInitialState

    cfg = _cfg()
    cfg["fasthydro"] = {"hard_vertex": {"mode": "centre", "smear": 0.4},
                        "hydro": {"accept_preeq_flow_loss": False, "store": "vector",
                                  "store_fields": None}}
    ini = FastGlauberInitialState(cfg, seed=99, verbose=False)
    ini.Exec()
    assert ini.ncoll_density is None
    pts = _sample(ini, 200)
    assert np.array_equal(pts, np.zeros_like(pts))


# ── the evolution store ──────────────────────────────────────────────────────

def test_grid_metadata_is_declared_explicitly(evolved):
    _, _, h = evolved
    bi = h.get_bulk_info()
    assert (bi.nx, bi.ny, bi.neta, bi.ntau) == (h.g.nx, h.g.ny, h.g.neta, h.g.ntau)
    # both are uninitialised in the C++ constructor, so they must be set, not inherited
    assert bi.boost_invariant is False, "get_tz() would rewrite vz = z/t"
    assert bi.tau_eta_is_tz is False, "the store is Milne"


def test_hydro_reports_finished(evolved):
    _, _, h = evolved
    assert h.get_hydro_status_int() == 3        # HydroStatus::FINISHED


@pytest.mark.parametrize("it", [0, 2, 4])
def test_gethydroinfo_reproduces_stored_nodes(evolved, it):
    """At an exact grid node the interpolation is the identity -- so any mismatch is an
    index-order bug, not interpolation error. The grid is asymmetric, so a transpose shows."""
    _, _, h = evolved
    g = h.g
    ix, iy, ie = map(int, np.unravel_index(
        h.arr[0, :, :, :, it].argmax(), (g.nx, g.ny, g.neta)))
    tau = g.tau0 + it * g.record_dtau
    x, y, eta = g.x_min + ix * g.dx, g.y_min + iy * g.dy, g.eta_min + ie * g.deta
    c = h.get_hydro_cell(tau * np.cosh(eta), x, y, tau * np.sinh(eta))
    for name, got, want in (("e", c.energy_density, h.arr[0, ix, iy, ie, it]),
                            ("vx", c.vx, h.arr[1, ix, iy, ie, it]),
                            ("vy", c.vy, h.arr[2, ix, iy, ie, it]),
                            ("vz", c.vz, h.arr[3, ix, iy, ie, it])):
        assert got == pytest.approx(float(want), rel=2e-6, abs=2e-6), name
    assert c.temperature > 0 and c.entropy_density > 0 and c.pressure > 0


def test_vz_is_the_solvers_not_z_over_t(evolved):
    """The discriminating check for the boost-invariant rewrite in get_tz()."""
    _, _, h = evolved
    g, it = h.g, 1
    ix, iy = g.nx // 2, g.ny // 2
    ie = g.neta - 2                       # away from eta = 0, where z/t and vz differ
    tau = g.tau0 + it * g.record_dtau
    eta = g.eta_min + ie * g.deta
    c = h.get_hydro_cell(tau * np.cosh(eta), g.x_min + ix * g.dx, g.y_min + iy * g.dy,
                         tau * np.sinh(eta))
    stored = float(h.arr[3, ix, iy, ie, it])
    assert c.vz == pytest.approx(stored, rel=2e-6, abs=2e-6)
    assert abs(stored - np.tanh(eta)) > 1e-5, "test point cannot tell vz from z/t; pick another"


def test_the_first_frame_is_reachable(evolved):
    """Regression: bulk_info's grid metadata is float32 (Jetscape::real is float), so a
    tau_min of 0.6 comes back as 0.6000000238.  A query at exactly tau0 was then below
    tau_min, CheckInRange failed and get() answered vacuum -- making the entire first frame
    invisible.  Matter starts at <Eloss><tStart> = tau0, so partons quenched against nothing
    for their first step, silently."""
    _, _, h = evolved
    g = h.g
    assert h.get_bulk_info().tau_min > g.tau0, (
        "float32 no longer rounds tau_min up on this grid; pick a tau0 where it does, "
        "otherwise this test proves nothing")
    ix, iy, ie = g.nx // 2, g.ny // 2, g.neta // 2
    eta = g.eta_min + ie * g.deta
    c = h.get_hydro_cell(g.tau0 * np.cosh(eta), g.x_min + ix * g.dx,
                         g.y_min + iy * g.dy, g.tau0 * np.sinh(eta))
    assert c.energy_density == pytest.approx(float(h.arr[0, ix, iy, ie, 0]),
                                             rel=2e-6, abs=2e-6)


@pytest.mark.parametrize("axis", ["x", "y", "eta"])
def test_outermost_cell_on_each_axis_is_reachable(evolved, axis):
    """Same float32 boundary problem on the spatial axes: the edge cells must not read vacuum."""
    _, _, h = evolved
    g, it = h.g, 1
    tau = g.tau0 + it * g.record_dtau
    idx = {"x": [0, g.ny // 2, g.neta // 2],
           "y": [g.nx // 2, 0, g.neta // 2],
           "eta": [g.nx // 2, g.ny // 2, 0]}[axis]
    ix, iy, ie = idx
    eta = g.eta_min + ie * g.deta
    c = h.get_hydro_cell(tau * np.cosh(eta), g.x_min + ix * g.dx,
                         g.y_min + iy * g.dy, tau * np.sinh(eta))
    assert c.energy_density == pytest.approx(float(h.arr[0, ix, iy, ie, it]),
                                             rel=2e-6, abs=2e-6)


def test_out_of_grid_returns_vacuum_and_is_counted(evolved):
    """CheckInRange never throws (the throws are commented out), so the count is the only
    signal that Matter/LBT sampled outside the fireball."""
    _, _, h = evolved
    g = h.g
    h.reset_out_of_range_count()
    far = 10 * abs(g.x_min)
    c = h.get_hydro_cell(g.tau0 + 0.1, far, far, 0.0)
    assert c.energy_density == 0.0 and c.temperature == 0.0
    assert h.get_out_of_range_count() >= 1


# ── the source ───────────────────────────────────────────────────────────────

def _run(cfg, ini, source, tag):
    from fasthydro.hydro import FastHydro
    h = FastHydro(cfg, stage=1, module_id=tag, ic=ini, source=source, verbose=False)
    h.InitializeHydro(None)
    h.EvolveHydro()
    return h


def test_zero_droplets_reproduce_the_background_bitwise(evolved):
    """The single most important regression test: if this drifts, no jet/no-jet pair from
    this contribution means anything."""
    cfg, ini, bg = evolved
    empty = DropletArray(np.zeros((0, 8)), np.array([0, 0], dtype=np.int64))
    src = CausalLiquefierSource(empty, LiquefierParams(), mode="conservative",
                                dtype=torch.float64)
    z = _run(cfg, ini, src, "zero")
    assert np.array_equal(z.arr, bg.arr)


def test_injected_four_momentum_equals_the_droplets(evolved):
    cfg, ini, bg = evolved
    p = LiquefierParams(tau_delay=0.6)
    drops = np.array([[0.6, 0.6, -0.5, 0.0, 8.0, 2.0, 1.0, 0.5],
                      [0.8, -1.2, 1.0, 0.8, 5.0, -1.0, 0.5, -1.5]])
    da = DropletArray(drops, np.array([0, 2], dtype=np.int64))
    jet = _run(cfg, ini, CausalLiquefierSource(da, p, mode="conservative",
                                               dtype=torch.float64), "jet")
    s = jet._live_source
    assert s.fired.all(), "a droplet inside the window never fired"
    rel = np.max(np.abs(s.P_injected - s.P_requested)) / max(1e-12, np.max(np.abs(s.P_requested)))
    assert rel < 1e-10, f"four-momentum not conserved: {rel:.3e}"
    assert np.abs(jet.arr[0] - bg.arr[0]).max() > 0, "the wake never reached the evolution"


def test_wake_grows_with_deposited_energy(evolved):
    """Linear response: double the momentum, roughly double the energy difference."""
    cfg, ini, bg = evolved
    p = LiquefierParams(tau_delay=0.6)
    base = np.array([[0.6, 0.0, 0.0, 0.0, 4.0, 1.0, 0.0, 0.0]])

    def wake(scale):
        d = base.copy()
        d[:, 4:8] *= scale
        da = DropletArray(d, np.array([0, 1], dtype=np.int64))
        h = _run(cfg, ini, CausalLiquefierSource(da, p, mode="conservative",
                                                 dtype=torch.float64), f"s{scale}")
        return float(np.abs(h.arr[0] - bg.arr[0]).sum())

    w1, w2 = wake(1.0), wake(2.0)
    assert 1.8 < w2 / w1 < 2.2, f"not linear: {w2 / w1:.3f}"
