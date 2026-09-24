"""PairBrowser on a pair file written by PyJetscape's PairH5Writer (MUSIC two-stage layout).

The writer is driven with stub MUSIC legs, so this needs neither MUSIC nor the compiled
extension -- only that `jetscape.pair_h5` imports.  What it pins is the contract between the
two packages: the MUSIC pair file opens in PairBrowser, `diff` is `arr - arr_bg`, `live()`
follows the file's "frames_written" freeze-out convention, and the shower group reads back.
"""

from __future__ import annotations

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
pair_h5 = pytest.importorskip("jetscape.pair_h5")

NX, NY, NETA = 5, 4, 3


class _Bulk:
    def __init__(self, evo):
        self.evo = evo
        self.ntau, self.nx, self.ny, self.neta = evo.shape[:4]
        self.dx = self.dy = 0.5
        self.deta = 0.4
        self.x_min = -0.5 * (self.nx - 1) * self.dx
        self.y_min = -0.5 * (self.ny - 1) * self.dy
        self.eta_min = -0.5 * (self.neta - 1) * self.deta
        self.tau_min, self.dtau, self.boost_invariant = 0.4, 0.1, False

    def to_numpy_full(self, n_features=5):
        e, v = self.evo[..., :1], self.evo[..., 1:]
        return np.concatenate([e, np.zeros_like(e), v], axis=-1)


class _Music:
    def __init__(self, name, evo, liq=None):
        self.name, self.evo, self.liq, self.dump = name, evo, liq, False

    def GetId(self):
        return self.name

    def get_bulk_info(self):
        return _Bulk(self.evo)

    def get_dump_hydro_only(self):
        return self.dump

    def set_dump_hydro_only(self, v):
        self.dump = v

    def set_skip_surface(self, v):
        pass

    def get_native_evolution_numpy(self, tau_stride=1):
        return self.evo[::tau_stride]

    def clear_hydro_info_from_memory(self):
        pass

    def get_liquefier(self):
        return self.liq


class _Liq:
    c_diff, gamma_relax = 0.894427190999916, 5.0

    def droplets_numpy(self):
        return np.array([[0.6, 0.0, 0.0, 0.0, 3.0, 1.0, 0.0, 0.0]])

    def params(self):
        return {"dtau": 0.02, "tau_delay": 0.2, "time_relax": 0.1, "d_diff": 0.08,
                "width_delta": 0.1}

    def ClearTask(self):
        pass


class _Shower:
    def vertices_to_numpy(self):
        return np.array([[0, 0, 0, 0, 0.0], [1, 0, 0, 0, 1.0]])

    def to_numpy(self):
        return np.array([[0, 1, 21, 0, 1, 0, 0, 10, 0, 0, 0, 0.5]], dtype=float)


class _Manager:
    def get_showers(self):
        return [_Shower()]

    def get_shower_initiating_partons(self):
        return []

    def GetTaskList(self):
        return []


class _JS:
    def __init__(self, *tasks):
        self.tasks = tasks

    def GetTaskList(self):
        return list(self.tasks)


def _evo(ntau):
    e = np.zeros((ntau, NX, NY, NETA, 4), np.float32)
    e[..., 0] = 1.0 + np.arange(ntau, dtype=np.float32)[:, None, None, None]
    e[..., 1:] = 0.1
    return e


@pytest.fixture
def music_pair(tmp_path):
    bg = _Music("MUSIC_1", _evo(4))
    jet_evo = _evo(6)
    jet_evo[2:, 2, 2, 1, 0] += 0.5                  # the jet from frame 2 on
    jet = _Music("MUSIC_2", jet_evo, liq=_Liq())
    path = tmp_path / "music_pair.h5"
    w = pair_h5.PairH5Writer(path, grid_mode="native",
                             provenance={"hard_vertex": "3dMCGlauber"})
    w.attach(_JS(bg, jet), manager=_Manager())
    w.Exec()
    w.Finish()
    return path


def test_pair_browser_opens_a_music_pair_file(music_pair):
    from fasthydro.browse import PairBrowser

    with PairBrowser(music_pair) as p:
        assert p.ic_matches                         # frame 0 identical
        # frames_written: 6 jet frames, 4 background frames, both live for 4
        assert p.jet.live(0) == 6 and p.bg.live(0) == 4 and p.live(0) == 4
        with h5py.File(music_pair, "r") as f:
            for k in range(4):
                np.testing.assert_allclose(p.diff(0, k), f["arr"][0, 0, ..., k].astype(float)
                                           - f["arr_bg"][0, 0, ..., k])
        assert p.diff(0, 1).max() == 0 and p.diff(0, 3).max() == pytest.approx(0.5)
        assert p.has_shower
        partons, vertices, _ = p.showers(0)
        assert partons.shape == (1, 13) and vertices.shape == (2, 6)
        s = p.summary(0)
        assert s["hard_vertex"] == "3dMCGlauber"


def test_fast_data_files_keep_the_legacy_convention(tmp_path):
    """A file without freezeout_convention_id is read as ntau_freezeout - 1, as before."""
    from fasthydro.browse import _freezeout_offset

    p = tmp_path / "legacy.h5"
    with h5py.File(p, "w") as f:
        f.attrs["x"] = 1
    with h5py.File(p, "r") as f:
        assert _freezeout_offset(f) == 1
    with h5py.File(p, "a") as f:
        f.attrs["freezeout_convention_id"] = "frames_written"
    with h5py.File(p, "r") as f:
        assert _freezeout_offset(f) == 0
