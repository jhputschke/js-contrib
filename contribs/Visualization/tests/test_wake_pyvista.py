"""Gates for the FastHydro-pair reader and the shower adapter behind wake_pyvista.py.

Nothing here renders: PyVista needs a GPU context and would make the suite unrunnable in
CI.  What is gated is the part that can be silently wrong on a screen -- which array became
which panel, which parton got which segment, and whether the difference is scaled so the
wake is actually visible.
"""

import os
import sys

import numpy as np
import pytest

_VIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FASTHYDRO = os.path.join(os.path.dirname(_VIZ), "FastHydro", "python")
for _p in (_VIZ, _FASTHYDRO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

h5py = pytest.importorskip("h5py")

wp = pytest.importorskip("wake_pyvista")                                 # noqa: E402
from fasthydro import showers as fsh                                     # noqa: E402

NX = NY = 6
NZ, NT = 4, 5


def _write_pair(path, *, with_shower=True, with_eos=True, nev=1):
    """A minimal file with FastHydro's layout: the schema, not the physics."""
    rng = np.random.default_rng(0)
    bg = rng.random((nev, 4, NX, NY, NZ, NT)).astype(np.float32)
    jet = bg.copy()
    jet[:, 0, 2, 2, 2, 3] += 5.0                     # one cell of wake, in a known place
    with h5py.File(path, "w") as f:
        f.create_dataset("arr", data=jet)
        f.create_dataset("arr_bg", data=bg)
        for k, v in dict(nx=NX, ny=NY, neta=NZ, dx=0.5, dy=0.5, deta=0.4,
                         x_min=-1.25, y_min=-1.25, eta_min=-0.6,
                         tau_min=0.6, dtau=0.2, choose_ntau=NT, nevents=nev).items():
            f.attrs[k] = v
        if with_eos:
            # linear in e on purpose: _temperature interpolates the table, so a curved
            # T(e) would make this a test of np.interp's accuracy instead of a test that
            # the file's own table is the thing being read.
            e = np.linspace(0.0, 50.0, 64)
            f.create_dataset("eos/e_tab", data=e)
            f.create_dataset("eos/T_tab", data=0.01 * e + 0.05)
        if with_shower:
            # two partons in a chain: 0 -> 1 -> 2, moving along +x at c
            par = np.array([
                [0, 0, 1, 21, 0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0, 1, 2, 21, 0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]], float)
            ver = np.array([[0, 0, 0, 0, 0, 0.0], [0, 1, 0, 0, 0, 0.0],
                            [0, 2, 0, 0, 0, 1.0]], float)
            f.create_dataset("shower/partons", data=par)
            f.create_dataset("shower/vertices", data=ver)
            f.create_dataset("shower/initiators",
                             data=np.zeros((1, len(fsh.INITIATOR_COLUMNS))))
            f.create_dataset("shower/parton_offsets", data=np.array([0, 2] + [2] * (nev - 1)))
            f.create_dataset("shower/vertex_offsets", data=np.array([0, 3] + [3] * (nev - 1)))
            f.create_dataset("shower/initiator_offsets",
                             data=np.array([0, 1] + [1] * (nev - 1)))
    return str(path)


# --------------------------------------------------------------------------- loading
def test_the_panels_are_the_right_arrays(tmp_path):
    """Swapping arr and arr_bg flips the sign of every wake and nothing else looks wrong,
    so the mapping is gated rather than eyeballed."""
    p = _write_pair(tmp_path / "p.h5")
    pa, meta, _ = wp.load_pair(p, 0)
    assert set(pa) == {"bg", "jet", "diff"}
    with h5py.File(p) as f:
        jet = f["arr"][0].transpose(4, 1, 2, 3, 0)
        bg = f["arr_bg"][0].transpose(4, 1, 2, 3, 0)
    assert np.array_equal(pa["jet"][..., 0], jet[..., 0])
    assert np.array_equal(pa["bg"][..., 0], bg[..., 0])
    assert np.allclose(pa["diff"][..., 0], jet[..., 0] - bg[..., 0])
    # the one seeded cell, at (tau=3, x=2, y=2, eta=2)
    assert pytest.approx(5.0, abs=1e-5) == pa["diff"][3, 2, 2, 2, 0]


def test_the_milne_axes_survive_the_transpose(tmp_path):
    """arr is (4,nx,ny,neta,ntau); the renderer wants (ntau,nx,ny,neta,nf). A wrong
    permutation still yields a valid-looking box on a cubic grid -- so use a grid where
    no two axes match."""
    p = _write_pair(tmp_path / "p.h5")
    pa, meta, _ = wp.load_pair(p, 0)
    assert pa["jet"].shape == (NT, NX, NY, NZ, 5)
    assert (meta["nx"], meta["ny"], meta["neta"], meta["ntau"]) == (NX, NY, NZ, NT)
    assert meta["has_eta"] and not meta["boost_invariant"]


def test_temperature_comes_from_the_files_own_eos(tmp_path):
    p = _write_pair(tmp_path / "p.h5")
    pa, _, _ = wp.load_pair(p, 0, ("jet",))
    e = pa["jet"][..., 0]
    assert np.allclose(pa["jet"][..., 1], 0.01 * e + 0.05, rtol=1e-5, atol=1e-6)


def test_a_file_with_no_eos_still_loads(tmp_path, capsys):
    p = _write_pair(tmp_path / "p.h5", with_eos=False)
    pa, _, _ = wp.load_pair(p, 0, ("jet",))
    assert np.isfinite(pa["jet"][..., 1]).all()
    assert "conformal" in capsys.readouterr().out


def test_the_diff_panel_keeps_the_jet_legs_temperature(tmp_path):
    """A difference of temperatures is not a field; an isosurface drawn on one would be
    a contour of nothing."""
    p = _write_pair(tmp_path / "p.h5")
    pa, _, _ = wp.load_pair(p, 0)
    assert np.array_equal(pa["diff"][..., 1], pa["jet"][..., 1])
    assert (pa["diff"][..., 1] >= 0).all()


def test_a_single_evolution_is_refused_with_a_reason(tmp_path):
    p = str(tmp_path / "solo.h5")
    with h5py.File(p, "w") as f:
        f.create_dataset("arr", data=np.zeros((1, 4, NX, NY, NZ, NT), np.float32))
    with pytest.raises(SystemExit, match="arr_bg"):
        wp.load_pair(p, 0)


def test_an_out_of_range_event_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="out of range"):
        wp.load_pair(_write_pair(tmp_path / "p.h5"), 7)


# --------------------------------------------------------------------------- shower
def test_the_shower_adapter_produces_drawable_segments(tmp_path):
    seg = wp.load_shower(_write_pair(tmp_path / "p.h5"), 0)
    assert seg is not None
    for k in ("starts", "ends", "t0", "t1", "dirs", "pT", "vel", "is_leaf", "energy"):
        assert k in seg, f"make_jet_overlay needs {k}"
    assert seg["starts"].shape == (2, 3) and seg["ends"].shape == (2, 3)
    # parton 0 splits at t=1 where parton 1 is produced; parton 1 never splits
    assert np.allclose(seg["starts"][0], [0, 0, 0]) and np.allclose(seg["ends"][0], [1, 0, 0])
    assert seg["t0"][0] == 0.0 and seg["t1"][0] == 1.0
    assert bool(seg["is_leaf"][1]) and not bool(seg["is_leaf"][0])


def test_an_absorbed_parton_does_not_free_stream(tmp_path):
    """pstat -11 is a parton the liquefier put INTO the medium; its energy is in
    source/droplets from that moment, so streaming it onward draws the jet twice."""
    p = _write_pair(tmp_path / "p.h5")
    with h5py.File(p, "r+") as f:
        f["shower/partons"][1, 4] = -11
    seg = wp.load_shower(p, 0)
    assert not bool(seg["is_leaf"][1])


def test_negative_energy_holes_are_not_dropped_by_the_default_filter(tmp_path):
    """Holes carry negative energy, so a plain `E >= min_energy` removes every one of
    them even at the default 0."""
    p = _write_pair(tmp_path / "p.h5")
    with h5py.File(p, "r+") as f:
        f["shower/partons"][1, 8] = -1.0
        f["shower/partons"][1, 4] = -17
    seg = wp.load_shower(p, 0, min_energy=0.0)
    assert len(seg["starts"]) == 2, "the hole was silently filtered out"


def test_no_shower_group_is_not_an_error(tmp_path):
    assert wp.load_shower(_write_pair(tmp_path / "p.h5", with_shower=False), 0) is None


# --------------------------------------------------------------------------- scaling
def _frames(vals):
    return {"diff": [{"e": np.asarray(v, np.float32)} for v in vals]}


def test_the_difference_is_percentile_scaled_not_max_scaled():
    """Measured on a real event: max|de| is 2.36 in ONE frame at tau=1.6 where the first
    droplets land, while the wake that follows runs at 0.1-0.3. Scaling to the max renders
    the wake at a few percent of full scale -- invisible, which is the whole figure."""
    spike = np.full(100_000, 0.2, np.float32)
    spike[0] = 100.0                       # the tau ~ 1.6 deposit, one cell in one frame
    lo, hi = wp._clim(_frames([spike]), "diff")
    assert lo == -hi
    assert hi < 1.0, "a single spike must not set the scale"
    assert pytest.approx(0.2, abs=0.02) == hi


def test_the_difference_scale_is_symmetric():
    """A depleted region is as physical as the wake front; an asymmetric map would show
    only one of them."""
    lo, hi = wp._clim(_frames([np.array([-3.0, 1.0], np.float32)]), "diff")
    assert hi > 0 and lo == -hi


def test_a_leg_is_scaled_from_zero_to_its_max():
    lo, hi = wp._clim({"jet": [{"e": np.array([0.0, 4.0], np.float32)}]}, "jet")
    assert (lo, hi) == (0.0, 4.0)


def test_an_all_zero_difference_does_not_produce_a_degenerate_scale():
    lo, hi = wp._clim(_frames([np.zeros(10, np.float32)]), "diff")
    assert lo < hi, "a zero clim makes add_volume raise rather than draw nothing"


# --------------------------------------------------------------------------- labelling
def test_the_panel_titles_do_not_claim_a_jetless_run():
    """There is ONE shower and it is in all three panels, quenched by arr_bg. Titling the
    left panel "no jet" reads as a jet/no-jet comparison, which is not what is drawn: the
    difference between the first two panels is the medium's back-reaction, not the jet.
    JetScape::SetPointers registers only the FIRST FluidDynamics as the framework's hydro,
    so Matter and LBT query the background leg and never see the jet leg at all."""
    titles = " ".join(t for t, _ in wp.PANELS.values()).lower()
    assert "no jet" not in titles and "with jet" not in titles
    assert wp.PANELS["bg"][1] == "arr_bg" and wp.PANELS["jet"][1] == "arr"
    assert "same quenched shower" in wp.SHOWER_NOTE
    assert "one-way" in wp.SHOWER_NOTE
