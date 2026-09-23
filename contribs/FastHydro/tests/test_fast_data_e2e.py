"""End-to-end: run generate.py in-process on a tiny grid and check the file it produces.

Needs no data files and no network.  ~10 s.
"""

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "python"))
sys.path.insert(0, os.path.join(_HERE, "..", "workflow_fastdata"))

pytest.importorskip("torch")
pytest.importorskip("h5py")
yaml = pytest.importorskip("yaml")

# generate.py, data.dataset and read_3d_hdf5 live in FNO4d (workflow_fastdata/ and
# loc_libs/), NOT in fast_data, so they are not vendored here -- FastHydro drives the solver
# through JETSCAPE instead of through generate.py.  The test is kept verbatim so that a
# checkout with FNO4d on PYTHONPATH still runs it; otherwise it skips.
generate = pytest.importorskip(
    "generate", reason="FNO4d's workflow_fastdata/generate.py is not vendored by FastHydro")
live_tau_lengths = pytest.importorskip("data.dataset").live_tau_lengths      # noqa: E402
_r3 = pytest.importorskip("read_3d_hdf5")                                    # noqa: E402
MultiH5Array, read_3d_data_hdf5 = _r3.MultiH5Array, _r3.read_3d_data_hdf5

TINY = {
    "run": {"nevents": 2, "seed": 4242, "device": "cpu", "dtype": "float64",
            "log": False, "overwrite": True},
    "grid": {"nx": 12, "ny": 12, "neta": 8, "dx": 0.6, "dy": 0.6, "deta": 0.5},
    "time": {"tau0": 0.6, "record_dtau": 0.15, "choose_ntau": 8},
    "initial_state": {"proj": "d", "targ": "Au", "target_T": 0.28, "calib_events": 8},
    "eos": {"kind": "conformal", "dof": 47.5},
    "output": {"T_fo": 0.150},
}


def _cfg(tmp_path, out, **over):
    import copy
    d = copy.deepcopy(TINY)
    for k, v in over.items():
        d.setdefault(k, {})
        d[k].update(v) if isinstance(v, dict) else d.__setitem__(k, v)
    d["run"]["out"] = str(tmp_path / out)
    d["run"]["odir"] = str(tmp_path)
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(d))
    return str(p), d["run"]["out"]


def test_e2e_background_only(tmp_path):
    cfg, out = _cfg(tmp_path, "bg.h5")
    assert generate.main(["--config", cfg]) == 0

    d = read_3d_data_hdf5([out])
    assert d["arr"].shape == (2, 4, 12, 12, 8, 8)
    assert np.isfinite(d["arr"]).all()
    assert (d["arr"][:, 0] >= 0).all(), "energy density must not go negative"
    assert np.abs(d["arr"][:, 1:4]).max() <= 1.0, "|v| must stay subluminal"
    lt = live_tau_lengths(MultiH5Array([out]))
    for live, n in zip(lt, d["ntau_freezeout"]):
        assert live == n - 1 or n >= d["choose_ntau"]
    assert d["generator"] == "fast_data"


def test_e2e_is_reproducible_from_the_seed(tmp_path):
    c1, o1 = _cfg(tmp_path, "a.h5")
    generate.main(["--config", c1])
    c2, o2 = _cfg(tmp_path, "b.h5")
    generate.main(["--config", c2])
    a = read_3d_data_hdf5([o1])["arr"]
    b = read_3d_data_hdf5([o2])["arr"]
    assert np.array_equal(a, b), "the same seed must reproduce the dataset exactly"

    c3, o3 = _cfg(tmp_path, "c.h5", run={"seed": 777})
    generate.main(["--config", c3])
    assert not np.array_equal(a, read_3d_data_hdf5([o3])["arr"])


def test_e2e_with_a_jet_source_conserves_what_it_injected(tmp_path):
    import h5py
    # A smooth, dense fireball rather than a dilute d+Au, so the test measures the source
    # bookkeeping rather than the solver's behaviour in near-vacuum cells (which is what
    # test_e2e_a_vacuum_deposit_fails_loudly covers).
    src = {"enabled": True, "params": {"tau_delay": 0.5},
           "partons": {"label": "trigger", "x": 0.0, "y": 0.0, "eta": 0.0, "tau": 0.6,
                       "E": 2.0, "phi": 0.0, "rapidity": 0.0, "back_to_back": True}}
    cfg, out = _cfg(tmp_path, "jet.h5", source=src,
                    initial_state={"kind": "smooth", "K": 1.0, "target_T": None,
                                   "smooth": {"e0": 40.0, "R": 4.0, "eta_flat": 2.0,
                                              "sigma_eta": 1.0}})
    assert generate.main(["--config", cfg]) == 0

    with h5py.File(out) as f:
        assert f.attrs["has_source"] and f.attrs["source_model"] == "causal_liquefier"
        S, P4 = f["source/S"][:], f["source/P_cart"][:]
        drops, offs = f["source/droplets"][:], f["source/offsets"][:]
        assert (S[..., 0] == 0).all(), "frame 0 is the IC"
        assert offs[-1] == len(drops)
        for i in range(2):
            want = drops[offs[i]:offs[i + 1], 4:8].sum(axis=0)
            assert np.allclose(P4[i].sum(axis=0), want, atol=1e-9), \
                "the injected four-momentum must equal the droplets'"
            assert f["diag/src_n_fired"][i] == 2
            assert f["diag/src_E_injected"][i] == pytest.approx(4.0, abs=1e-9)


def test_e2e_the_source_stream_does_not_perturb_the_initial_state(tmp_path):
    """Adding partons must leave the backgrounds event-for-event identical, so a with-source and
    a without-source dataset can be differenced."""
    import h5py
    c1, o1 = _cfg(tmp_path, "nosrc.h5")
    generate.main(["--config", c1])
    src = {"enabled": True, "params": {"tau_delay": 0.5},
           "placement_weight": "energy_weighted",      # keep the deposit inside the medium
           "partons": {"x": 0.0, "y": 0.0, "eta": 0.0, "tau": 0.6,
                       "E": 0.5, "px": 0.5, "py": 0.0, "pz": 0.0}}
    c2, o2 = _cfg(tmp_path, "withsrc.h5", source=src)
    generate.main(["--config", c2])
    with h5py.File(o1) as a, h5py.File(o2) as b:
        assert np.array_equal(a["diag/seed"][:], b["diag/seed"][:])
        assert np.array_equal(a["diag/b"][:], b["diag/b"][:])
        assert np.array_equal(a["diag/e_max"][:], b["diag/e_max"][:])
        assert not np.array_equal(a["arr"][:], b["arr"][:])   # but the evolution does differ


def test_e2e_dry_run_writes_nothing(tmp_path):
    cfg, out = _cfg(tmp_path, "dry.h5")
    assert generate.main(["--config", cfg, "--dry-run"]) == 0
    assert not os.path.exists(out)


def test_e2e_shards_are_independently_valid(tmp_path):
    cfg, out = _cfg(tmp_path, "sh.h5", run={"nevents": 4})
    for i in range(2):
        assert generate.main(["--config", cfg, "--shard", f"{i}/2"]) == 0
    stem = out[:-3]
    files = [f"{stem}.shard{i:02d}.h5" for i in range(2)]
    a = MultiH5Array(files)
    assert len(a) == 4
    for p in files:
        d = read_3d_data_hdf5([p])
        assert d["nevents"] == d["arr"].shape[0] == 2


def test_e2e_the_frame_guard_rejects_unphysical_data():
    """A training file full of NaN loads, trains and poisons a model without announcing itself,
    so a frame that cannot be valid training data must stop the run.

    Tested directly on the guard rather than by contriving a divergence end to end: since
    evolution now stops at freeze-out, the scenario that used to blow up (a fireball evolved far
    past freeze-out, where the grid is essentially vacuum) no longer arises, and a test that
    depends on a solver instability is not one worth keeping.
    """
    from fast_data.evolve import EvolutionDiverged, _check_frame

    good = np.ones((4, 2, 2, 2), dtype=np.float32) * 0.5
    _check_frame(good, 3, 1.2)                       # must not raise

    nan = good.copy(); nan[0, 0, 0, 0] = np.nan
    with pytest.raises(EvolutionDiverged, match="non-finite"):
        _check_frame(nan, 3, 1.2)

    fast = good.copy(); fast[1, 0, 0, 0] = 1.5
    with pytest.raises(EvolutionDiverged, match="spacelike"):
        _check_frame(fast, 3, 1.2)

    # float64 here on purpose: the solver hands the guard its own dtype, and in a float32 run
    # anything past the float32 range is already inf and caught by the check above.  The range
    # check is what catches a float64 run whose values could not be stored.
    huge = np.ones((4, 2, 2, 2), dtype=np.float64) * 0.5
    huge[0, 0, 0, 0] = 1e39
    with pytest.raises(EvolutionDiverged, match="float32"):
        _check_frame(huge, 3, 1.2)

    # the diagnosis names the cause that applies, rather than always blaming a jet
    with pytest.raises(EvolutionDiverged, match="freeze-out"):
        _check_frame(nan, 3, 1.2, has_source=False)
    with pytest.raises(EvolutionDiverged, match="jet deposit"):
        _check_frame(nan, 3, 1.2, has_source=True)


def test_e2e_evolution_stops_once_the_fireball_has_frozen_out(tmp_path):
    """Frames past freeze-out are zeroed anyway, and continuing to step a near-vacuum grid is
    where the scheme goes unstable.  Stopping must not change the frames that are kept."""
    import h5py
    cfg, out = _cfg(tmp_path, "stop.h5", time={"choose_ntau": 30}, output={"T_fo": 0.150})
    assert generate.main(["--config", cfg]) == 0
    with h5py.File(out) as f:
        arr, nfo = f["arr"][:], f["ntau_freezeout"][:]
        assert np.isfinite(arr).all()
        for i, n in enumerate(nfo):
            assert n < 30, "this config should freeze out well inside the tau range"
            assert (arr[i, ..., n - 1:] == 0).all(), "the tail must be zero"
            assert np.abs(arr[i, 0, ..., :n - 1]).max() > 0, "the kept frames must hold data"
