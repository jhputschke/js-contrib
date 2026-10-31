"""
tests/test_extras_analyses.py

The Z+jet analyses with extra hard scatterings (docs/Plans/PLAN_multi_hard.md §3.6) on the
synthetic production files of extras_campaign.py, one with an extra_frag file and its twin
without: the extras belong to the event, not to the trigger.  Per analysis (the Z-hadron pools
of zjet_mixing.file_pool and mix.build_pools, the QA map step qa.py, wake_iso's map step, the
jet map step zjet_jets_map.py): the full event takes the extras' fragments, the trigger's
truth (the Z-correlated yield, the fragments, the background, the reference jets) does not
change, and with the switch off (extras=False / --no-extras) the outputs are those of the
files without extras, array for array.

    pytest tests/test_extras_analyses.py -q
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent / "example" / "analysis"
sys.path.insert(0, str(HERE.parent / "python"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ANALYSIS))                                   # jetcorr
sys.path.insert(0, str(ANALYSIS / "zjet"))

h5py = pytest.importorskip("h5py")

import extras_campaign as xc  # noqa: E402

N_X = xc.N_HARD + xc.N_SOFT        # the extras' charged tracks per fragmentation (|eta| < 2.4)


@pytest.fixture(scope="module")
def prods(tmp_path_factory):
    """The same production file with ("x") and without ("n") extra hard scatterings, and a
    minimum-bias job ("m") -> {kind: stem}."""
    base = tmp_path_factory.mktemp("extras")
    return {"x": xc.zjet_files(base / "x", extras=True),
            "n": xc.zjet_files(base / "n", extras=False),
            "m": xc.zjet_files(base / "m", minbias=True)}


def _same(a, b, skip=()):
    assert set(a) == set(b), sorted(set(a) ^ set(b))
    for k in a:
        if k not in skip:
            x, y = np.asarray(a[k]), np.asarray(b[k])
            assert np.array_equal(x, y, equal_nan=x.dtype.kind == "f"), k


# ── the Z-hadron pools ───────────────────────────────────────────────────────────
def test_pools_full_event_takes_the_extras(prods):
    import zjet_mixing as zm

    def pool(stem, extras=True):
        return zm.file_pool(stem, [0, 1], [0, 1], [1.0, 1.0], [0, 0], seed=1, extras=extras)

    on, off, no = pool(prods["x"]), pool(prods["x"], False), pool(prods["n"])
    _same(off, no)                                      # switched off: as without extras
    assert "bgw_R_extra" in on and "bgw_R_extra" not in off
    for k in ("Y_zc", "L_zc", "Y_frag", "L_frag", "bgw_S_zc", "bgr_RB", "k_one", "w"):
        assert np.array_equal(on[k], off[k]), k         # the trigger's truth
    for k in ("S_full", "Y_full", "L_full", "S_one", "Y_one"):
        d = on[k].astype(float) - off[k]
        assert np.allclose(d.reshape(2, -1).sum(1), N_X), k     # 8 tracks per event
    assert np.array_equal(on["trk_n"] - off["trk_n"], [N_X, N_X])
    assert np.isclose(on["bgw_R_extra"].sum(), 2 * N_X)
    assert np.allclose(on["bgw_R_full"] - off["bgw_R_full"], on["bgw_R_extra"])
    # B, the bulk alone (Y_full - Y_zc), is bulk_bg + extra_frag
    b_on = on["Y_full"].astype(float) - on["Y_zc"]
    b_off = off["Y_full"].astype(float) - off["Y_zc"]
    assert np.allclose((b_on - b_off).reshape(2, -1).sum(1), N_X)
    # assemble: R_zc without the extras, the full density with them
    p_on, p_off = zm.Pool(zm.assemble([on])), zm.Pool(zm.assemble([off]))
    assert np.allclose(p_on.bg_R["zc"], p_off.bg_R["zc"], rtol=0, atol=1e-12)
    assert np.isclose(p_on.bg_R["full"].sum() - p_off.bg_R["full"].sum(),
                      p_on.bg_W.sum() * N_X)
    with pytest.raises(ValueError, match="extra hard scatterings"):
        zm.assemble([on, {**off, "fi_name": np.array("run_B")}])


def test_pools_of_mix_record_the_extras(prods, tmp_path):
    import json

    import mix
    src = os.path.dirname(prods["x"])
    on = mix.build_pools(src, str(tmp_path / "on"), 1, 1, True)
    off = mix.build_pools(src, str(tmp_path / "off"), 1, 1, True, extras=False)
    no = mix.build_pools(os.path.dirname(prods["n"]), str(tmp_path / "no"), 1, 1, True)
    with open(tmp_path / "on" / "manifest.json") as f:
        assert json.load(f)["extras"][0][0] == "run_A"
    for d in ("off", "no"):
        with open(tmp_path / d / "manifest.json") as f:
            assert "extras" not in json.load(f)
    with np.load(off[0]) as a, np.load(no[0]) as b:
        _same(dict(a), dict(b))
    with np.load(on[0]) as a:
        assert "bgw_R_extra" in a.files


# ── the QA map step ──────────────────────────────────────────────────────────────
def test_qa_full_multiplicity_takes_the_extras(prods, tmp_path):
    import qa

    def run(stem, name, extras=True):
        out = tmp_path / f"{name}.npz"
        qa.qa_file(stem, 0, [0, 1], [0, 1], np.ones(2), np.full(2, -1), [0, 0], str(out),
                   extras)
        with np.load(out) as f:
            return dict(f)

    on, off, no = run(prods["x"], "on"), run(prods["x"], "off", False), run(prods["n"], "no")
    _same(off, no)
    extra = [k for k in on if k not in off]
    assert set(extra) == ({f"h_extra_frag_{v}_{c}_{w}" for v in qa.VARS for c in qa.CHARGES
                           for w in qa.WEIGHTS} | {f"s_extra_frag_{w}" for w in qa.WEIGHTS}
                          | {f"ev_{k}" for k in qa.EXTRA_COLUMNS})
    assert np.array_equal(on["ev_n_extra"], [2, 2])
    assert np.allclose(on["ev_nch24_extra"], N_X) and np.allclose(on["ev_nch05_extra"], 3)
    assert np.allclose(on["ev_nch24_full"] - off["ev_nch24_full"], N_X)
    for k in ("ev_nch24_bg", "ev_nch24_frag", "h_bulk_jet_pt_all_n", "h_jet_frag_pt_all_n"):
        assert np.array_equal(on[k], off[k]), k
    assert on["h_extra_frag_pt_charged_n"].sum() == pytest.approx(2 * N_X)


# ── wake_iso: the measured event's noise ─────────────────────────────────────────
def test_wake_iso_noise_takes_the_extras(prods):
    import wake_iso as wi

    def job(stem, extras=True):
        return wi.file_job(stem, [0, 1], [1.0, 1.0], 1.0, 60, extras=extras)

    on, off, no = job(prods["x"]), job(prods["x"], False), job(prods["n"])
    (key, g_on), = on["groups"].items()
    g_off, g_no = off["groups"][key], no["groups"][key]
    _same(g_off, g_no)
    for k in g_on:                    # the contents stay the trigger's; the noise has extras
        same = np.array_equal(g_on[k], g_off[k])
        assert same == (not k.startswith(("nze_", "nzr_", "nzc_"))), k
    assert g_on["N"] == g_on["NJ"] == 2


# ── the jet map step ─────────────────────────────────────────────────────────────
def _map(src, out, *args):
    env = dict(os.environ, PYTHONPATH=str(HERE.parent / "python"))
    return subprocess.run([sys.executable, str(ANALYSIS / "zjet" / "zjet_jets_map.py"), src,
                           str(out), "--samples", "2", *args], capture_output=True, text=True,
                          env=env)


def test_jets_map_full_event_and_b_take_the_extras(prods, tmp_path):
    pytest.importorskip("fastjet")
    uproot = pytest.importorskip("uproot")
    ak = pytest.importorskip("awkward")
    runs = {"on": (prods["x"], ()), "off": (prods["x"], ("--no-extras",)),
            "no": (prods["n"], ())}
    data = {}
    for name, (stem, args) in runs.items():
        res = _map(os.path.dirname(stem), tmp_path / name, *args)
        assert res.returncode == 0, res.stderr
        with uproot.open(tmp_path / name / "run_A_jets.root") as f:
            data[name] = {t: f[t].arrays() for t in ("jet_events", "jets", "bg_jets",
                                                     "qa_jets", "run_info")}
    for t in data["off"]:                               # switched off: as without extras
        assert data["off"][t].fields == data["no"][t].fields, t
        assert ak.to_list(data["off"][t]) == ak.to_list(data["no"][t]), t
    on, off = data["on"], data["off"]
    assert int(on["run_info"]["extras"][0]) == 1 and "extras" not in off["run_info"].fields
    assert "bgx" in on["jets"].fields and "bgx" not in off["jets"].fields
    # the trigger's reference jets and the bulk's own jets do not change
    assert ak.to_list(on["jets"]["emb"]["ref_pt"]) == ak.to_list(off["jets"]["emb"]["ref_pt"])
    assert ak.to_list(on["bg_jets"]) == ak.to_list(off["bg_jets"])
    # the extras' jet (5 x 9 GeV) is in the full event, in B (bgx) and in the no-wake twin
    for col in ("jet", "bgx", "nw"):
        assert ak.all(ak.sum(on["jets"][col]["pt"] > 30, axis=1) == 1), col
    for col in ("jet", "nw"):
        assert ak.all(ak.sum(off["jets"][col]["pt"] > 30, axis=1) == 0), col
    assert ak.all(ak.sum(on["bg_jets"]["jet"]["pt"] > 30, axis=1) == 0)


def test_z_analyses_refuse_minimum_bias(prods, tmp_path):
    """A minimum-bias job has no trigger (no jet_frag, no Z): refused with a clear message."""
    import mix
    import wake_iso as wi
    import zjet_wake as zw
    from jetscape.hadrons_h5 import HadronFileReader
    src = os.path.dirname(prods["m"])
    zw.refuse_minimum_bias(HadronFileReader(os.path.dirname(prods["x"])), "x")   # passes
    with pytest.raises(SystemExit, match="minimum bias"):
        mix.build_pools(src, str(tmp_path / "pools"), 1, 1, True)
    with pytest.raises(SystemExit, match="minimum bias"):
        wi.build([src], str(tmp_path / "wi.npz"), jobs=1)
    pytest.importorskip("fastjet")
    res = _map(src, tmp_path / "m")
    assert res.returncode != 0 and "minimum bias" in res.stderr
