"""
tests/test_analysis_root_reader.py

example/analysis_root/HadronFileReader.h (C++, through PyROOT) on the small --pthat-bins
campaign of test_run_h5toROOT.py, converted by run_h5toROOT.py: every event, oversample
and source (bkg, bkg_dep, frag, bkg_dep_frag) gives the hadrons that
jetscape.hadrons_h5.HadronFileReader reads from the HDF5 files, for RNTuple and TTree,
both writers, truncated floats and files without positions.  Skipped without PyROOT.

    pytest tests/test_analysis_root_reader.py -q
"""

from __future__ import annotations

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
ROOT = pytest.importorskip("ROOT")

import run_h5toROOT as r2r  # noqa: E402
from jetscape.hadrons_h5 import HadronFileReader  # noqa: E402
from test_run_h5toROOT import SIGMA, _writers, campaign  # noqa: E402,F401

HEADER = HERE.parent / "example" / "analysis_root" / "HadronFileReader.h"


@pytest.fixture(scope="module")
def cxx():
    assert ROOT.gInterpreter.Declare(f'#include "{HEADER}"')   # (#pragma once)
    # cppyy can't pass Python callables as a Filter: set one from C++
    ROOT.gInterpreter.Declare("""
        inline void test_frag_only(hadrons_root::HadronFileReader &r) {
          r.set_filter([](const hadrons_root::Hadron &h) {
            return h.origin == hadrons_root::kFrag;
          });
        }""")
    return ROOT.hadrons_root


def _arrays(hadrons, positions):
    cols = ["pid", "pstat", "E", "px", "py", "pz", "origin"]
    if positions:
        cols += ["t", "x", "y", "z"]
    return {c: np.array([getattr(h, c) for h in hadrons]) for c in cols}


def _expect(ev, origin=None):
    p, x = np.asarray(ev["p"], float), np.asarray(ev["x"], float)
    out = {"pid": ev["pid"], "pstat": ev["pstat"], "E": p[:, 0], "px": p[:, 1],
           "py": p[:, 2], "pz": p[:, 3],
           "origin": ev["origin"] if origin is None else np.full(len(ev["pid"]), origin)}
    if x.shape[1]:
        out.update(t=x[:, 0], x=x[:, 1], y=x[:, 2], z=x[:, 3])
    return out


def _same(got, want, rtol):
    assert len(got["pid"]) == len(want["pid"])
    for c, v in got.items():
        if c in want:
            assert np.allclose(v, want[c], rtol=rtol, atol=rtol), c


CASES = [(fmt, writer, {}) for fmt in ("rntuple", "ttree") for writer in _writers()]
if "root" in _writers():
    CASES += [("ttree", "root", {"bits": 12}), ("rntuple", "root", {"bits": 12})]
CASES += [("rntuple", "uproot", {"no_x": True})]


@pytest.mark.parametrize("fmt,writer,opt", CASES)
def test_reader_matches_h5(campaign, tmp_path, cxx, fmt, writer, opt):  # noqa: F811
    out = tmp_path / "root"
    argv = [str(campaign), "--out-dir", str(out), "--format", fmt, "--writer", writer]
    if opt.get("bits"):
        argv += ["--bits-p", str(opt["bits"]), "--bits-x", str(opt["bits"])]
    if opt.get("no_x"):
        argv += ["--no-x"]
    assert r2r.main(argv) == 0
    rtol = 2.0 ** -opt.get("bits", 23)

    h5 = HadronFileReader(str(campaign))
    r = cxx.HadronFileReader(str(out), True)               # with positions
    assert r.n_files() == 2 and r.n_events() == h5.n_events == 4
    assert r.n_windows() == 2
    with uproot.open(out / "tiny_campaign.root") as f:
        win = f["windows"].arrays()
    weight = win["weight_mb"]
    assert np.allclose([r.weight_mb(k) for k in range(2)], weight)
    assert np.allclose([r.sigma_err_mb(k) for k in range(2)], win["sigma_err_mb"])
    assert r.sigma_source().startswith("campaign file")
    assert list(r.select_events()) == [0, 1, 2, 3] and list(r.select_events(1)) == [1, 3]

    for e in range(h5.n_events):
        info, want = r.info(e), h5.event_info(e)
        assert (info.event, info.file_index, info.local_event) == (e, want.file_index,
                                                                   want.local_event)
        assert info.bg_unit == want.bg_unit and info.pthat_bin == want.pthat_bin
        assert info.pthat == want.pthat and not info.flagged()
        assert info.weight_mb == pytest.approx(weight[want.pthat_bin])
        sigma_file = SIGMA["AB"[want.file_index]][want.pthat_bin]
        assert info.sigma_file_mb == pytest.approx(sigma_file)
        ini = want.initiators()
        assert info.initiators.size() == len(ini)
        assert np.allclose([[p.px, p.py, p.pz, p.E] for p in info.initiators], ini[:, 3:7])
        lead = ini[np.argmax(np.hypot(ini[:, 3], ini[:, 4]))]
        assert info.leading().px == pytest.approx(lead[3])
        assert (r.n_oversamples(e), r.n_bg_samples(e), r.n_frag(e)) == (
            h5.n_oversamples(e), h5.n_samples("bulk_bg", e), h5.n_frag(e))
        pos = not opt.get("no_x")
        for k in range(r.n_oversamples(e)):
            _same(_arrays(r.bkg(e, k), pos), _expect(h5.background_event(e, k), 0), rtol)
            _same(_arrays(r.bkg_dep(e, k), pos),
                  _expect(h5.jet_event(e, k, fragments=False)), rtol)
            _same(_arrays(r.bkg_dep_frag(e, k), pos), _expect(h5.jet_event(e, k)), rtol)
            _same(_arrays(r.jet_event(e, k, 0), pos), _expect(h5.jet_event(e, k, 0)), rtol)
        for j in range(r.n_frag(e)):
            frag = _expect(h5.jet_event(e, 0, frag_sample=j))
            keep = frag["origin"] == 1
            _same(_arrays(r.frag(e, j), pos), {c: v[keep] for c, v in frag.items()}, rtol)
        if not pos:
            assert all(h.t == 0 and h.x == 0 for h in r.bkg_dep(e, 0))
    with pytest.raises(Exception, match="out of range"):
        r.info(4)
    with pytest.raises(Exception, match="no bulk_jet sample 9"):
        r.bkg_dep(0, 9)


def test_filter_and_kinematics(campaign, tmp_path, cxx):  # noqa: F811
    out = tmp_path / "root"
    assert r2r.main([str(campaign), "--out-dir", str(out), "--writer", "uproot"]) == 0
    r = cxx.HadronFileReader(str(out / "run_A_hadrons.root"))
    assert r.n_events() == 2 and r.campaign_file().endswith("tiny_campaign.root")
    h5 = HadronFileReader(str(campaign / "run_A_particlize.h5"))
    for e in range(2):
        hs = r.bkg_dep_frag(e, 1)
        ev = h5.jet_event(e, 1)
        p = np.asarray(ev["p"], float)
        pt = np.hypot(p[:, 1], p[:, 2])
        assert np.allclose([h.pt() for h in hs], pt)
        assert np.allclose([h.phi() for h in hs], np.arctan2(p[:, 2], p[:, 1]))
        pabs = np.sqrt((p[:, 1:] ** 2).sum(1))
        assert np.allclose([h.eta() for h in hs], np.arctanh(p[:, 3] / pabs))
        assert np.allclose([h.p4().E() for h in hs], p[:, 0])
        ROOT.test_frag_only(r)
        assert len(r.bkg_dep_frag(e, 1)) == int((ev["origin"] == 1).sum())
        r.set_charged_eta(0)
        charged = np.isin(np.abs(ev["pid"]), (211, 321, 2212, 3222, 3112, 3312, 3334, 11, 13))
        assert len(r.bkg_dep_frag(e, 1)) == int(charged.sum())
        r.clear_filter()
        assert len(r.bkg_dep_frag(e, 1)) == len(ev["pid"])


def test_cross_sections_without_campaign_file(campaign, tmp_path, cxx):  # noqa: F811
    """No *_campaign.root: the files' windows tables, combined as the campaign file is;
    weight_mb over the events of the files read."""
    out = tmp_path / "root"
    assert r2r.main([str(campaign), "--out-dir", str(out), "--writer", "uproot",
                     "--format", "ttree"]) == 0
    with uproot.open(out / "tiny_campaign.root") as f:
        win = f["windows"].arrays()
    h5 = HadronFileReader(str(campaign))
    (out / "tiny_campaign.root").unlink()
    r = cxx.HadronFileReader(str(out))
    assert r.campaign_file() == "" and "windows tables of the 2 file(s)" in r.sigma_source()
    assert r.n_windows() == 2
    for k in range(2):
        sig, err = h5.pthat_bin_sigma(k)
        assert r.sigma_mb(k) == pytest.approx(sig) == pytest.approx(win["sigma_mb"][k])
        assert r.sigma_err_mb(k) == pytest.approx(err)
        assert r.weight_mb(k) == pytest.approx(win["weight_mb"][k])
        assert r.n_window_events(k) == 2
        assert (r.pthat_lo(k), r.pthat_hi(k)) == tuple(win[c][k] for c in ("pthat_lo",
                                                                            "pthat_hi"))
    assert [r.info(e).weight_mb for e in range(4)] == pytest.approx(
        [win["weight_mb"][b] for b in (0, 1, 0, 1)])
    # one file: its own cross sections, over its own events
    one = cxx.HadronFileReader(str(out / "run_B_hadrons.root"))
    assert [one.sigma_mb(k) for k in range(2)] == pytest.approx(SIGMA["B"])
    assert [one.weight_mb(k) for k in range(2)] == pytest.approx(SIGMA["B"])  # 1 event each
    # the campaign file with all files gives the same; with another campaign's windows, an
    # error
    assert r2r.main([str(campaign), "--out-dir", str(out), "--writer", "uproot",
                     "--format", "ttree"]) == 0
    with_camp = cxx.HadronFileReader(str(out))
    assert [with_camp.sigma_mb(k) for k in range(2)] == pytest.approx(
        [r.sigma_mb(k) for k in range(2)])
    with uproot.recreate(tmp_path / "other_campaign.root") as f:
        f["windows"] = {"pthat_lo": np.array([1.0, 2.0]), "pthat_hi": np.array([2.0, 3.0]),
                        "sigma_mb": np.array([1.0, 1.0]), "weight_mb": np.array([1.0, 1.0])}
    with pytest.raises(Exception, match="other pTHat windows"):
        cxx.HadronFileReader(str(out), False, str(tmp_path / "other_campaign.root"))
