"""
tests/test_run_h5toROOT.py

example/prod_AuAu_0_10_jet/run_h5toROOT.py on a small --pthat-bins campaign (two production
files, two windows): one ROOT file per production file with the hadron ntuples, the event
and window tables and the provenance, and the campaign file with the combined cross
sections.  Read back with uproot; the ROOT writer is skipped without PyROOT.

    pytest tests/test_run_h5toROOT.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "python"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "example" / "prod_AuAu_0_10_jet"))

h5py = pytest.importorskip("h5py")
uproot = pytest.importorskip("uproot")
ak = pytest.importorskip("awkward")

import run_h5toROOT as r2r  # noqa: E402
from test_particlize_h5 import PHI_JET, _seed_files  # noqa: E402

BINS = np.array([[20.0, 40.0], [50.0, 70.0]])
SIGMA = {"A": [2.0, 0.2], "B": [4.0, 0.4]}
N_ACC = {"A": [10, 30], "B": [30, 10]}


@pytest.fixture()
def campaign(tmp_path):
    src = tmp_path / "prod"
    src.mkdir()
    for name in ("A", "B"):
        _seed_files(src, name, (0, 0),                  # one background, one jet per window
                    events_extra=[{"pthat_bin": 0, "pthat": 25.0},
                                  {"pthat_bin": 1, "pthat": 55.0}],
                    attrs_extra={"pthat_bins": BINS, "pthat_bin_sigma_gen": SIGMA[name],
                                 "pthat_bin_sigma_err": [0.1, 0.01],
                                 "pthat_bin_n_accepted": N_ACC[name],
                                 "prod_campaign": "tiny"})
    return src


def _writers():
    out = ["uproot"]
    try:
        import hadrons_to_root as h2r
        if h2r.have_pyroot():
            out.append("root")
    except ImportError:
        pass
    return out


@pytest.mark.parametrize("fmt", ["rntuple", "ttree"])
@pytest.mark.parametrize("writer", _writers())
def test_campaign_to_root(campaign, tmp_path, fmt, writer, capsys):
    out = tmp_path / "root"
    assert r2r.main([str(campaign), "--out-dir", str(out), "--format", fmt,
                     "--writer", writer]) == 0
    for name in ("A", "B"):
        with uproot.open(out / f"run_{name}_hadrons.root") as f:
            assert {k.split(";")[0] for k in f.keys()} >= {"events", "windows", "provenance",
                                                             "bulk_jet", "bulk_bg", "jet_frag"}
            ev = f["events"].arrays()
            assert list(ev["event"]) == [0, 1] and list(ev["pthat_bin"]) == [0, 1]
            assert list(ev["pthat"]) == [25.0, 55.0] and list(ev["bg_unit"]) == [0, 0]
            assert np.allclose(ev["sigma_file_mb"], SIGMA[name])
            assert list(ev["n_samples_jet"]) == [2, 2] and list(ev["n_samples_bg"]) == [2, 2]
            assert list(ev["n_samples_frag"]) == [1, 1]
            assert np.allclose([x[0] for x in ev["ini_px"]],
                               [np.cos(PHI_JET[(name, e)]) for e in (0, 1)])
            win = f["windows"].arrays()
            assert np.allclose(win["sigma_mb"], SIGMA[name])
            assert list(win["n_accepted"]) == N_ACC[name]
            assert f["bulk_jet"].num_entries == 4 and f["bulk_bg"].num_entries == 2
            assert f["jet_frag"].num_entries == 2
            jet = f["bulk_jet"].arrays()
            assert list(jet["event"]) == [0, 0, 1, 1] and list(jet["sample"]) == [0, 1, 0, 1]
            assert list(jet["bg_unit"]) == [0, 0, 0, 0]
            assert np.allclose(ak.flatten(jet["E"]), [1, 1, 1, 2, 2, 2, 11, 11, 11, 12, 12, 12])
            prov = json.loads(str(f["provenance"]))
            assert prov["settings"]["format"] == fmt and prov["settings"]["writer"] == writer
            assert np.allclose(prov["particlize"]["pthat_bins"], BINS)
            assert set(prov["hadrons"]) == {"bulk_jet", "bulk_bg", "jet_frag"}
    with uproot.open(out / "tiny_campaign.root") as f:
        win = f["windows"].arrays()
        s0 = (10 * 2.0 + 30 * 4.0) / 40
        s1 = (30 * 0.2 + 10 * 0.4) / 40
        assert np.allclose(win["sigma_mb"], [s0, s1])
        assert list(win["n_events"]) == [2, 2]
        assert np.allclose(win["weight_mb"], [s0 / 2, s1 / 2])
        files = f["files"].arrays()
        assert list(files["event_offset"]) == [0, 2] and list(files["n_events"]) == [2, 2]
        prov = json.loads(str(f["provenance"]))
        assert [x["stem"] for x in prov["files"]] == ["run_A", "run_B"]
        assert prov["campaigns"] == ["tiny"]


def test_restart_skip_and_incomplete_inputs(campaign, tmp_path, capsys):
    out = tmp_path / "root"
    with h5py.File(campaign / "run_B_hadrons_jet_frag.h5", "a") as f:
        f.attrs["complete"] = False                    # still being hadronized
    assert r2r.main([str(campaign), "--out-dir", str(out), "--writer", "uproot"]) == 0
    text = capsys.readouterr().out
    assert "1 to convert" in text and "not ready:  run_B (hadrons missing or incomplete: " \
                                     "jet_frag)" in text
    assert not (out / "run_B_hadrons.root").exists()
    with uproot.open(out / "tiny_campaign.root") as f:
        assert list(f["files"].arrays()["n_events"]) == [2]
    # again: A is kept; with other settings a warning, and the file is kept as it is
    before = (out / "run_A_hadrons.root").stat().st_mtime_ns
    assert r2r.main([str(campaign), "--out-dir", str(out), "--writer", "uproot",
                     "--no-x"]) == 0
    captured = capsys.readouterr()
    assert "1 already converted" in captured.out and "other settings" in captured.err
    assert (out / "run_A_hadrons.root").stat().st_mtime_ns == before
    # B complete now: only B is converted, the campaign file covers both
    with h5py.File(campaign / "run_B_hadrons_jet_frag.h5", "a") as f:
        f.attrs["complete"] = True
    assert r2r.main([str(campaign), "--out-dir", str(out), "--writer", "uproot"]) == 0
    assert "1 to convert, 1 already converted" in capsys.readouterr().out
    with uproot.open(out / "tiny_campaign.root") as f:
        assert list(f["files"].arrays()["n_events"]) == [2, 2]
    assert not list(out.glob("*.part"))


def test_dry_run_writes_nothing(campaign, tmp_path, capsys):
    out = tmp_path / "root"
    assert r2r.main([str(campaign), "--out-dir", str(out), "--dry-run"]) == 0
    assert "convert:    run_A" in capsys.readouterr().out
    assert not out.exists()
