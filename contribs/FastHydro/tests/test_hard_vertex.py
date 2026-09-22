"""Where hard scatterings are placed.

With no density set, `SampleABinaryCollisionPoint` only warns and puts every shower at the
origin -- every jet starting at the fireball centre, which biases any path-length-dependent
observable and makes all events look alike. These check that each mode does what it claims,
including that `centre` still reproduces that old behaviour on purpose.
"""

import numpy as np
import pytest

from fasthydro.config import DEFAULTS as FH_DEFAULTS
from fasthydro.config import resolve
from fasthydro.grid import GridSpec
from fasthydro.hard_vertex import MODES, binary_collision_density, node_axes

G = GridSpec(nx=25, ny=21, neta=7, dx=0.5, dy=0.6, deta=0.4,
             tau0=0.6, record_dtau=0.2, ntau=5)


def _event(seed=0, n_coll=400, spread=2.0, offset=(0.8, -0.4)):
    rng = np.random.default_rng(seed)
    coll = rng.normal(offset, spread, size=(n_coll, 2))
    part = rng.normal(offset, spread * 1.4, size=(2 * n_coll, 2))
    return {"coll": coll, "plus": part[:n_coll], "minus": part[n_coll:]}


def _moments(dens, g):
    xs, ys = node_axes(g)
    w = dens[:, :, 0] / dens[:, :, 0].sum()
    mx = float((w.sum(1) * xs).sum())
    my = float((w.sum(0) * ys).sum())
    sx = float(np.sqrt((w.sum(1) * (xs - mx) ** 2).sum()))
    return mx, my, sx


# ── the grid the sampler uses ────────────────────────────────────────────────

def test_node_axes_are_the_samplers_not_the_cell_centred_ones():
    """CoordFromIdx uses -grid_max + i*step (MUSIC's axis); the energy density is
    cell-centred. Half a cell apart, and the density must live on the former."""
    xs, ys = node_axes(G)
    xr, yr, _ = G.is_ranges()
    assert xs[0] == pytest.approx(-xr)
    assert np.allclose(np.diff(xs), G.dx)
    assert len(xs) == G.nx and len(ys) == G.ny
    assert xs[0] - G.x_min == pytest.approx(-0.5 * G.dx)      # exactly half a cell


# ── the modes ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("mode", ["ncoll", "ncoll_mc", "npart"])
def test_shape_and_positivity(mode):
    d = binary_collision_density(_event(), G, mode=mode)
    assert d.shape == (G.nx, G.ny, G.neta)
    assert d.dtype == np.float64
    assert np.isfinite(d).all() and d.sum() > 0 and (d >= 0).all()


@pytest.mark.parametrize("mode", ["ncoll", "ncoll_mc", "npart"])
def test_eta_broadcast_leaves_the_xy_marginal_alone(mode):
    """The sampler only uses x and y (it returns z = 0), so every eta slice must be equal."""
    d = binary_collision_density(_event(), G, mode=mode)
    for k in range(1, G.neta):
        assert np.array_equal(d[:, :, k], d[:, :, 0])


def test_centre_sets_nothing():
    assert binary_collision_density(_event(), G, mode="centre") is None


def test_no_event_or_no_points_sets_nothing():
    assert binary_collision_density(None, G, mode="ncoll") is None
    empty = {"coll": np.zeros((0, 2)), "plus": np.zeros((0, 2)), "minus": np.zeros((0, 2))}
    assert binary_collision_density(empty, G, mode="ncoll") is None


def test_smeared_and_raw_describe_the_same_distribution():
    """ncoll and ncoll_mc are the same physics; smearing only fills in the grid."""
    ev = _event(seed=3, n_coll=3000)
    sm = binary_collision_density(ev, G, mode="ncoll")
    mc = binary_collision_density(ev, G, mode="ncoll_mc")
    msx, msy, ssx = _moments(sm, G)
    mmx, mmy, smx = _moments(mc, G)
    assert msx == pytest.approx(mmx, abs=0.1)
    assert msy == pytest.approx(mmy, abs=0.1)
    # smearing widens by roughly the smear width in quadrature, not more
    assert smx < ssx < np.hypot(smx, 0.4) + 0.05


def test_raw_histogram_leaves_holes_that_smearing_fills():
    """This is what `ncoll` buys over `ncoll_mc`. A cell with a zero count cannot be drawn at
    all, so at realistic Ncoll the raw histogram makes much of the overlap region unreachable
    and the reachable part lumpy."""
    ev = _event(seed=7, n_coll=400)          # a realistic mid-central Ncoll
    sm = binary_collision_density(ev, G, mode="ncoll")[:, :, 0]
    mc = binary_collision_density(ev, G, mode="ncoll_mc")[:, :, 0]
    # restrict to where the distribution actually has support, so the comparison is fair
    core = sm > 1e-3 * sm.max()
    holes_mc = int((mc[core] == 0).sum())
    holes_sm = int((sm[core] == 0).sum())
    assert holes_sm == 0
    assert holes_mc > 0.2 * core.sum(), (holes_mc, int(core.sum()))


def test_participants_are_wider_than_binary_collisions():
    """Physics check: binary collisions concentrate in the overlap core, participants do not."""
    ev = _event(seed=5, n_coll=3000)
    _, _, s_coll = _moments(binary_collision_density(ev, G, mode="ncoll"), G)
    _, _, s_part = _moments(binary_collision_density(ev, G, mode="npart"), G)
    assert s_part > s_coll


def test_density_tracks_the_event_offset():
    off = (1.5, -1.0)
    d = binary_collision_density(_event(offset=off, n_coll=4000), G, mode="ncoll")
    mx, my, _ = _moments(d, G)
    assert mx == pytest.approx(off[0], abs=0.25)
    assert my == pytest.approx(off[1], abs=0.25)


# ── configuration ────────────────────────────────────────────────────────────

def test_default_mode_is_the_physical_one():
    assert FH_DEFAULTS["hard_vertex"]["mode"] == "ncoll"
    assert set(MODES) == {"ncoll", "ncoll_mc", "npart", "centre"}


def test_unknown_mode_is_rejected_and_lists_the_choices():
    with pytest.raises(ValueError) as e:
        binary_collision_density(_event(), G, mode="nonsense")
    for m in MODES:
        assert m in str(e.value)


@pytest.mark.parametrize("mode", ["ncoll", "npart"])
def test_non_positive_smear_is_rejected(mode):
    with pytest.raises(ValueError, match="smear"):
        binary_collision_density(_event(), G, mode=mode, width=0.0)


def test_config_block_validates():
    from fast_data.config import ConfigError

    assert resolve({})["hard_vertex"]["mode"] == "ncoll"
    assert resolve({"hard_vertex": {"mode": "centre"}})["hard_vertex"]["mode"] == "centre"
    with pytest.raises(ConfigError, match="hard_vertex.mode"):
        resolve({"hard_vertex": {"mode": "nope"}})
    with pytest.raises(ConfigError, match="smear"):
        resolve({"hard_vertex": {"smear": 0}})
    with pytest.raises(ConfigError, match="unknown config key"):
        resolve({"hard_vertex": {"typo": 1}})
    with pytest.raises(ConfigError, match="store"):
        resolve({"hydro": {"store": "weird"}})


# ── XML / YAML consistency ───────────────────────────────────────────────────

def test_shipped_xml_and_yaml_agree():
    """The two files are independent schemas, but four quantities appear in both and
    build_two_stage() refuses to run if they disagree. The shipped pair must pass."""
    import pathlib

    from fasthydro.config import load_config
    from fasthydro.pipeline import check_xml_agrees_with_cfg

    root = pathlib.Path(__file__).resolve().parent.parent
    cfg = load_config(root / "config" / "fasthydro_twostage.yaml")
    check_xml_agrees_with_cfg(str(root / "config" / "jetscape_user_fasthydro.xml"), cfg)


@pytest.mark.parametrize("mutate,expect", [
    (lambda c: c["source"]["params"].__setitem__("tau_delay", 99.0), "tau_delay"),
    (lambda c: c["grid"].__setitem__("dx", 0.77), "grid_step_x"),
    (lambda c: c["time"].__setitem__("tau0", 9.0), "taus"),
])
def test_disagreement_is_caught(mutate, expect):
    """Each overlapping quantity must be checked, including source.params -- which is never
    read, so a drift there would otherwise be a silent no-op."""
    import pathlib

    from fasthydro.config import load_config
    from fasthydro.pipeline import check_xml_agrees_with_cfg

    root = pathlib.Path(__file__).resolve().parent.parent
    cfg = load_config(root / "config" / "fasthydro_twostage.yaml")
    mutate(cfg)
    with pytest.raises(ValueError, match=expect):
        check_xml_agrees_with_cfg(str(root / "config" / "jetscape_user_fasthydro.xml"), cfg)
