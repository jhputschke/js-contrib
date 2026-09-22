"""Gates for the visualisation layer and the notebook it backs.

Rendering is checked under the Agg backend, so this needs no display.  The point is not that the
pictures look nice -- it is that EventBrowser reads the schema correctly, stays lazy, and that
every figure the notebook can ask for actually builds.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
h5py = pytest.importorskip("h5py")

import matplotlib.pyplot as plt          # noqa: E402

from fast_data import viz, writer        # noqa: E402

NX = NY = 10
NZ, NT, NEV = 6, 9, 2


@pytest.fixture
def sample(tmp_path):
    """A two-event file with a source group, shaped like a real run."""
    attrs = writer.grid_attrs(NX, NY, NZ, 0.5, 0.5, 0.5, tau_min=0.6, dtau=0.1, choose_ntau=NT)
    path = str(tmp_path / "viz.h5")
    rng = np.random.default_rng(0)
    drops = np.array([[0.6, 0.5, -0.5, 0.3, 12.0, 4.0, 1.0, 2.0]])
    with writer.FnoH5Writer(path, attrs, NEV, force=True, write_source=True,
                            extra_attrs={"generator": "fast_data", "proj": "Au", "targ": "Au",
                                         "liquefier_tau_delay": 0.5}) as w:
        for i in range(NEV):
            a = np.zeros((4, NX, NY, NZ, NT), dtype=np.float32)
            nfo = 6 + i
            for t in range(nfo - 1):
                a[0, ..., t] = rng.random((NX, NY, NZ)) * (5.0 - 0.5 * t) + 0.1
                a[1:4, ..., t] = (rng.random((3, NX, NY, NZ)) - 0.5) * 0.6
            S = np.zeros_like(a)
            S[0, ..., 2] = 1.0
            w.append_event(i, a, nfo, 0.6 + (nfo - 1) * 0.1, S_ev=S,
                           P_cart=np.ones((NT, 4)), droplets=drops,
                           diag={"b": 7.0 + i, "npart": 100 - 10 * i, "ncoll": 180})
    return path


def test_browser_reads_the_schema(sample):
    with viz.EventBrowser(sample) as br:
        assert (br.nevents, br.ntau) == (NEV, NT)
        assert (br.nx, br.ny, br.neta) == (NX, NY, NZ)
        # the axes must be reconstructed from the attributes, not guessed
        assert br.x[0] == pytest.approx(-0.5 * (NX - 1) * 0.5)
        assert br.tau[0] == pytest.approx(0.6)
        assert br.tau[1] - br.tau[0] == pytest.approx(0.1)
        assert br.has_source
        # live == ntau_freezeout - 1, the measured MUSIC convention
        assert [br.live(i) for i in range(NEV)] == [int(n) - 1 for n in br.ntau_fo]
        assert "Au+Au" in viz._title(br, 0) and "b = 7.00 fm" in viz._title(br, 0)


def test_browser_is_lazy(sample):
    """One frame at a time: a multi-GiB file has to be browsable without loading it."""
    with viz.EventBrowser(sample) as br:
        f = br.frame(0, 3)
        assert f.shape == (4, NX, NY, NZ)
        assert br.frame(0, 3, channel=0).shape == (NX, NY, NZ)
        assert br.source_frame(0, 2, 0).shape == (NX, NY, NZ)
        assert br.droplets(0).shape == (1, 8)


def test_summary_and_describe(sample):
    with viz.EventBrowser(sample) as br:
        s = br.summary()
        for k in ("event", "ntau_freezeout", "tau_freezeout", "b", "npart"):
            assert k in s and len(s[k]) == NEV
        text = br.describe()
        assert "2 events" in text and "tau" in text


@pytest.mark.parametrize("channel", [0, 1, 3])
def test_plot_slice_builds(sample, channel):
    with viz.EventBrowser(sample) as br:
        fig = br.plot_slice(event=0, itau=2, ieta=NZ // 2, channel=channel)
        assert len(fig.axes) >= 3
        plt.close(fig)


def test_plot_slice_handles_the_edges(sample):
    """Frame 0, the last frame, and a frame past freeze-out must all render."""
    with viz.EventBrowser(sample) as br:
        for itau in (0, NT - 1, int(br.ntau_fo[0])):
            fig = br.plot_slice(event=0, itau=itau, ieta=0)
            plt.close(fig)
        fig = br.plot_slice(event=0, itau=2, ieta=NZ // 2, show_source=True)
        plt.close(fig)
        fig = br.plot_slice(event=0, itau=2, ieta=NZ // 2, log=True)
        plt.close(fig)


def test_flow_mask_hides_vacuum_cells():
    """Arrows in near-vacuum cells say nothing about the fluid -- |v| there runs to the solver's
    cap -- and drawing them buries the ones that mean something."""
    e = np.array([[10.0, 1.0], [1e-9, 0.0]])
    keep = viz.flow_mask(e, floor=1e-3)
    assert keep.tolist() == [[True, True], [False, False]]
    assert viz.flow_mask(e, floor=0.5).tolist() == [[True, False], [False, False]]
    assert viz.flow_mask(np.zeros((2, 2))).sum() == 0        # an all-zero frame masks everything
    assert viz.flow_mask(np.array([])).size == 0             # and an empty one does not divide by 0


def test_quiver_is_drawn_and_respects_the_floor(sample):
    with viz.EventBrowser(sample) as br:
        fig = br.plot_slice(event=0, itau=2, quiver_floor=1e-3)
        assert [c for c in fig.axes[0].collections if hasattr(c, "U")], "no quiver was drawn"
        plt.close(fig)
        plt.close(br.plot_slice(event=0, itau=2, quiver=False))


def test_plot_evolution_and_initial_state(sample):
    with viz.EventBrowser(sample) as br:
        fig = br.plot_evolution(event=1, n_frames=4)
        plt.close(fig)
        fig = br.plot_initial_state(event=0)
        assert len(fig.axes) >= 2
        plt.close(fig)


def test_plot_initial_state_does_not_call_show_under_agg(sample):
    """plot_initial_state used to call plt.show() unconditionally, which warns under a
    non-interactive backend and draws twice in a notebook."""
    import warnings
    with viz.EventBrowser(sample) as br:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fig = br.plot_initial_state(event=0)
            plt.close(fig)
        assert not [w for w in caught if "non-interactive" in str(w.message)]


def test_browser_opens_a_file_without_a_source_group(tmp_path):
    attrs = writer.grid_attrs(NX, NY, NZ, 0.5, 0.5, 0.5, tau_min=0.6, dtau=0.1, choose_ntau=NT)
    path = str(tmp_path / "nosrc.h5")
    with writer.FnoH5Writer(path, attrs, 1, force=True, write_source=False) as w:
        a = np.zeros((4, NX, NY, NZ, NT), dtype=np.float32)
        a[0, ..., :4] = 1.0
        w.append_event(0, a, 5, 1.0)
    with viz.EventBrowser(path) as br:
        assert not br.has_source
        assert br.droplets(0) is None
        assert br.source_frame(0, 1) is None
        plt.close(br.plot_slice(event=0, itau=1))


# ------------------------------------------------------------------ DiffBrowser / Mach cone

def _pair(tmp_path, *, wedge_deg=None, ref_live=6, jet_live=7, dx=0.5):
    """A (jet, ref) pair on one grid.  `wedge_deg` paints a synthetic cone of that half-angle.

    The wedge is the geometry `front_angle` has to recover: a ridge leaving the source at
    +-wedge_deg from the jet axis, on a background the two files share exactly.
    """
    n = 41
    attrs = writer.grid_attrs(n, n, 5, dx, dx, dx, tau_min=0.6, dtau=0.1, choose_ntau=NT)
    x = attrs["x_min"] + np.arange(n) * dx
    drops = np.array([[0.6 + 0.1 * k, -4.0 + 0.5 * k, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
                      for k in range(8)])          # a +x trajectory, 8 deposits
    out = []
    for tag, live, jet in (("jet", jet_live, True), ("ref", ref_live, False)):
        path = str(tmp_path / f"{tag}.h5")
        with writer.FnoH5Writer(path, attrs, 1, force=True, write_source=jet,
                                extra_attrs={"liquefier_tau_delay": 0.1,
                                             "liquefier_c_diff": 0.9}) as w:
            a = np.zeros((4, n, n, 5, NT), dtype=np.float32)
            for t in range(live):
                a[0, ..., t] = 3.0                         # a flat, identical background
                if jet and wedge_deg is not None:
                    src = -4.0 + 0.5 * (t - 1)             # where the source is at frame t
                    k = 1.0 / np.tan(np.radians(wedge_deg))
                    for iy, yv in enumerate(x):
                        if yv <= 0 or yv > 5.0:
                            continue
                        ix = int(np.argmin(np.abs(x - (src - k * yv))))
                        a[0, ix, iy, :, t] += 1.0          # the ridge, upper half only
            w.append_event(0, a, live + 1, 0.6 + live * 0.1,
                           S_ev=(np.zeros_like(a) if jet else None),
                           P_cart=(np.ones((NT, 4)) if jet else None),
                           droplets=(drops if jet else None))
        out.append(path)
    return out


def test_diff_of_a_file_with_itself_is_zero(sample):
    with viz.DiffBrowser(sample, sample) as d:
        assert d.ic_matches is True
        for t in range(d.ntau):
            assert np.all(d.diff(0, t) == 0.0)


def test_live_is_the_overlap_not_the_union(tmp_path):
    """Past the reference's freeze-out the difference is not a wake, so it must be excluded."""
    jet, ref = _pair(tmp_path, ref_live=5, jet_live=8)
    with viz.DiffBrowser(jet, ref, check_ic=False) as d:
        assert d.jet.live(0) == 8 and d.ref.live(0) == 5
        assert d.live(0) == 5


def test_source_track_is_ordered_and_delayed(tmp_path):
    jet, ref = _pair(tmp_path)
    with viz.DiffBrowser(jet, ref, check_ic=False) as d:
        t = d.source_track(0)
        assert len(t) == 8
        assert np.all(np.diff(t[:, 0]) > 0)
        # tau_dep = tau_production + tau_delay, read from the file's own attribute
        assert t[0, 0] == pytest.approx(0.6 + 0.1)
        assert d.jet_direction(0) == pytest.approx([1.0, 0.0, 0.0])


def test_source_at_interpolates_and_clamps(tmp_path):
    jet, ref = _pair(tmp_path)
    with viz.DiffBrowser(jet, ref, check_ic=False) as d:
        t = d.source_track(0)
        mid = d.source_at(0, 0.5 * (t[0, 0] + t[1, 0]))
        assert mid[0] == pytest.approx(0.5 * (t[0, 1] + t[1, 1]))
        # outside the deposit window the marker parks, it does not extrapolate
        assert d.source_at(0, 99.0)[0] == pytest.approx(t[-1, 1])
        assert d.source_at(0, 0.0)[0] == pytest.approx(t[0, 1])
        assert d.source_active(0, 0.0) is False
        assert d.source_active(0, t[3, 0]) is True


@pytest.mark.parametrize("wedge", [30.0, 45.0, 60.0])
def test_front_angle_recovers_a_synthetic_wedge(tmp_path, wedge):
    """The angle measurement is the quantitative claim, so it is checked against known geometry."""
    jet, ref = _pair(tmp_path, wedge_deg=wedge)
    with viz.DiffBrowser(jet, ref, check_ic=False) as d:
        got = d.front_angle(0, 4, ieta=2)
        assert got == pytest.approx(wedge, abs=4.0)


def test_grid_mismatch_is_refused(tmp_path):
    jet, _ = _pair(tmp_path)
    other, _ = _pair(tmp_path / "b", dx=0.25)
    with pytest.raises(ValueError, match="disagree"):
        viz.DiffBrowser(jet, other, check_ic=False)


def test_plot_mach_builds(tmp_path):
    jet, ref = _pair(tmp_path, wedge_deg=40.0)
    with viz.DiffBrowser(jet, ref, check_ic=False) as d:
        fig = d.plot_mach(0, itau=4)
        assert len(fig.axes) >= 3
        plt.close(fig)


def test_colour_scale_ignores_the_deposit_cell(tmp_path):
    """One saturated cell at the source must not set the scale, or the wake is invisible."""
    jet, ref = _pair(tmp_path, wedge_deg=40.0)
    with viz.DiffBrowser(jet, ref, check_ic=False) as d:
        de = d.diff(0, 4)
        pos = d.source_at(0, d.tau[4])
        ix = int(np.argmin(np.abs(d.x - pos[0])))
        iy = int(np.argmin(np.abs(d.y - pos[1])))
        de[ix, iy, 2] = 1000.0
        assert viz._diff_scale(d, de, 2, pos, 99.5) < 10.0


# ------------------------------------------------------------------ lazy-read performance
#
# `arr` is chunked one whole event per chunk, so a single-frame read decompresses the entire
# event unless the chunk stays cached.  Two things have to hold, and each was broken once:
# the cache must be big enough for a chunk, and the dataset handle must stay open, because
# HDF5 keeps the chunk cache on the dataset and frees it when the last handle closes.
# Getting either wrong costs a factor of 110 per frame and is invisible except as slowness.

def test_chunk_cache_is_sized_to_one_chunk(sample):
    with viz.EventBrowser(sample) as br:
        ds = br._f["arr"]
        chunk_bytes = int(np.prod(ds.chunks)) * ds.dtype.itemsize
        assert br.cache_bytes >= chunk_bytes
        _, nbytes, _ = ds.id.get_access_plist().get_chunk_cache()
        assert nbytes >= chunk_bytes           # and HDF5 actually accepted it


def test_dataset_handle_is_held_open_across_reads(sample):
    with viz.EventBrowser(sample) as br:
        ds = br._arr
        for k in range(br.ntau):
            br.frame(0, k, 0)
        assert br._arr is ds and ds.id.valid    # never re-opened, so the cache survives


def test_a_chunk_too_large_to_cache_is_not_cached(tmp_path):
    """The cache is bounded: an enormous chunk must fall back, not allocate gigabytes."""
    jet, _ = _pair(tmp_path)
    with viz.EventBrowser(jet, max_cache_bytes=1024) as br:
        assert br.cache_bytes == 0
        assert br.frame(0, 0, 0).shape == (41, 41, 5)   # and still reads correctly


# ------------------------------------------------------------------ colour scaling
#
# A fireball cools by ~100x between tau0 and freeze-out, so one linear scale pinned to the peak
# renders every late frame as black.  These pin the scaling rules rather than the pictures.

def _norms(fig):
    """The norms of the heatmaps only -- a scatter's ScalarMappable carries an empty one.

    Colour bars are drawn as meshes too and share their mappable's norm object, so counting
    DISTINCT norm objects is the robust way to ask how many scales a figure really uses.
    """
    from matplotlib.collections import QuadMesh

    return [m.norm for ax in fig.axes for m in ax.collections
            if isinstance(m, QuadMesh) and m.norm is not None and m.norm.vmax is not None]


def test_evolution_is_logarithmic_by_default(sample):
    from matplotlib.colors import LogNorm
    with viz.EventBrowser(sample) as br:
        fig = br.plot_evolution(event=0, n_frames=4)
        norms = [n for n in _norms(fig) if isinstance(n, LogNorm)]
        assert norms, "the filmstrip and the eta-tau map must default to a log scale"
        for n in norms:                       # bounded below, never down to the vacuum floor
            assert n.vmin > 0 and n.vmax / n.vmin == pytest.approx(1e3, rel=0.01)
        plt.close(fig)


def test_evolution_shares_one_scale_across_the_strip(sample):
    """Shared, or the frames stop being comparable and the cooling is invisible."""
    with viz.EventBrowser(sample) as br:
        # two scales in the whole figure: one shared by the strip, one for the eta-tau map,
        # which plots max over x,y and legitimately differs
        fig = br.plot_evolution(event=0, n_frames=4)
        assert len({id(n) for n in _norms(fig)}) == 2
        plt.close(fig)

        fig = br.plot_evolution(event=0, n_frames=4, per_frame=True)
        assert len({id(n) for n in _norms(fig)}) > 2      # one per panel instead
        plt.close(fig)


def test_evolution_linear_still_available(sample):
    from matplotlib.colors import LogNorm
    with viz.EventBrowser(sample) as br:
        fig = br.plot_evolution(event=0, n_frames=3, log=False)
        assert not any(isinstance(n, LogNorm) for n in _norms(fig))
        plt.close(fig)


def test_log_norm_floor_is_relative_not_the_vacuum_floor(sample):
    """The raw minimum is 1e-6; spanning to it would leave the fireball in the top sliver."""
    a = np.array([1e-6, 1e-3, 1.0, 30.0])
    n = viz._log_norm(a, dynamic_range=1e4)
    assert n.vmax == pytest.approx(30.0)
    assert n.vmin == pytest.approx(30.0 / 1e4)
    assert viz._log_norm(np.zeros(5)) is None


def test_slice_log_applies_to_both_heatmaps(sample):
    """`log` reached only the x-y panel once; the x-eta panel stayed linear beside it."""
    from matplotlib.colors import LogNorm
    with viz.EventBrowser(sample) as br:
        fig = br.plot_slice(event=0, itau=1, ieta=2, log=True, quiver=False)
        assert sum(isinstance(n, LogNorm) for n in _norms(fig)) >= 2
        plt.close(fig)


# ------------------------------------------------------------------ the notebook itself

NOTEBOOK = os.path.join(os.path.dirname(__file__), "..", "workflow_fastdata",
                        "explore_fastdata.ipynb")


def _nb():
    import json
    if not os.path.exists(NOTEBOOK):
        pytest.skip("explore_fastdata.ipynb not present")
    with open(NOTEBOOK) as fh:
        return json.load(fh)


def _code_cells(nb):
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_notebook_never_calls_plt_show():
    """`plt.show()` inside an ipywidgets callback is what drew every figure three times.

    ipywidgets runs `show_inline_matplotlib_plots()` after the callback AND before the next
    update, so a figure left open in pyplot's registry is drawn by each of those as well as by
    `plt.show()` itself.  The notebook displays and then closes instead, which makes all three
    no-ops.  Classic Jupyter happens to hide the bug; VS Code does not.
    """
    for src in _code_cells(_nb()):
        body = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
        body = body.replace('"""', "\x00").split("\x00")           # drop docstrings
        code = "".join(body[::2])
        assert "plt.show()" not in code, f"plt.show() in:\n{src}"


def test_notebook_closes_every_figure_it_draws():
    cells = _code_cells(_nb())
    assert any("def render(fig)" in c and "plt.close(fig)" in c for c in cells), \
        "the notebook must define a render() helper that closes the figure it displays"
    for src in cells:
        for call in ("plot_slice(", "plot_evolution(", "plot_initial_state(", "plot_mach("):
            if call in src:
                assert "render(" in src, f"{call} not routed through render() in:\n{src}"


def test_notebook_is_committed_without_outputs():
    """Outputs make the diff unreadable and can carry a multi-MB PNG per figure."""
    nb = _nb()
    for c in nb["cells"]:
        assert not c.get("outputs"), "notebook committed with outputs"
        assert c.get("execution_count") is None
