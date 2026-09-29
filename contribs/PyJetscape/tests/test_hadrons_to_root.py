"""
tests/test_hadrons_to_root.py

example/prod_AuAu_0_10_jet/root_export/hadrons_to_root.py: a hadron file -> ROOT, one entry
per sample.  Read back with uproot; the PyROOT writers are skipped without ROOT.

    pytest tests/test_hadrons_to_root.py -q
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

h5py = pytest.importorskip("h5py")
uproot = pytest.importorskip("uproot")
ak = pytest.importorskip("awkward")

from jetscape.hadrons_h5 import HadronH5Writer  # noqa: E402

EXPORT = Path(__file__).resolve().parents[1] / "example" / "prod_AuAu_0_10_jet" / "root_export"


def _h2r():
    spec = importlib.util.spec_from_file_location("hadrons_to_root",
                                                  EXPORT / "hadrons_to_root.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sample(n, rng, pid=211):
    pt, eta = rng.uniform(0.2, 2.0, n), rng.uniform(-3, 3, n)
    phi = rng.uniform(-np.pi, np.pi, n)
    px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    return {"pid": np.full(n, pid, np.int32), "pstat": np.zeros(n, np.int32),
            "p": np.stack([np.sqrt(px ** 2 + py ** 2 + pz ** 2 + 0.019), px, py, pz],
                          axis=1).astype(np.float32),
            "x": rng.normal(0, 5, (n, 4)).astype(np.float32)}


@pytest.fixture()
def hadron_file(tmp_path):
    """bulk_jet with 2 events (units 3 and 5), 3 samples each, one of them empty, p and x
    rounded to 12 / 8 bits (as for a campaign), and a particlize file next to it giving
    their backgrounds (bg_unit 0 and 1)."""
    rng = np.random.default_rng(7)
    path = tmp_path / "run_hadrons_bulk_jet.h5"
    samples = [[_sample(4, rng), _sample(0, rng), _sample(2, rng, pid=-321)],
               [_sample(3, rng), _sample(5, rng), _sample(1, rng, pid=22)]]
    with HadronH5Writer(path, tag="bulk_jet", n_samples=3, keep_bits={"p": 12, "x": 8},
                        attrs={"source": "run_particlize.h5"}) as w:
        w.append_unit(samples[0], unit=3, event=0, seed=1)
        w.append_unit(samples[1], unit=5, event=1, seed=2)
    with h5py.File(tmp_path / "run_particlize.h5", "w") as f:
        f["events/bg_unit"] = np.array([0, 1], np.int32)
    return path, samples


def _check(path, samples, ref, with_x=True):
    """The ROOT file holds one entry per sample with the h5 file's values (``ref``: its
    load_columns)."""
    t = uproot.open(path)["bulk_jet"]
    assert t.num_entries == 6
    a = t.arrays()
    assert list(a["event"]) == [0, 0, 0, 1, 1, 1]
    assert list(a["unit"]) == [3, 3, 3, 5, 5, 5]
    assert list(a["sample"]) == [0, 1, 2, 0, 1, 2]
    assert list(a["bg_unit"]) == [0, 0, 0, 1, 1, 1]
    flat = [s for ev in samples for s in ev]
    assert [len(x) for x in a["pid"]] == [len(s["pid"]) for s in flat]
    assert np.array_equal(ak.to_numpy(ak.flatten(a["pid"])), ref["pid"])
    for name in ("E", "px", "py", "pz") + (("t", "x", "y", "z") if with_x else ()):
        assert np.array_equal(ak.to_numpy(ak.flatten(a[name])), ref[name]), name
    assert ("x" in t.keys()) == with_x


@pytest.mark.parametrize("fmt", ["ttree", "rntuple"])
def test_uproot_writer(hadron_file, tmp_path, fmt):
    h2r = _h2r()
    path, samples = hadron_file
    cols = h2r.load_columns(path)
    h2r.write_uproot(tmp_path / "u.root", cols, fmt=fmt, chunk_samples=4)   # 2 chunks
    _check(tmp_path / "u.root", samples, cols)
    nox = h2r.load_columns(path, with_x=False)
    h2r.write_uproot(tmp_path / "nox.root", nox, fmt=fmt)
    _check(tmp_path / "nox.root", samples, nox, with_x=False)


def test_cli_auto_writer_falls_back_to_uproot(hadron_file, tmp_path, monkeypatch, capsys):
    h2r = _h2r()
    path, samples = hadron_file
    monkeypatch.setattr(h2r, "have_pyroot", lambda: False)
    assert h2r.main([str(path), "-o", str(tmp_path / "a.root")]) == 0
    captured = capsys.readouterr()
    assert "PyROOT not available" in captured.err and "rntuple by uproot" in captured.out
    _check(tmp_path / "a.root", samples, h2r.load_columns(path))
    with pytest.raises(SystemExit):                  # truncated floats need ROOT
        h2r.main([str(path), "-o", str(tmp_path / "b.root"), "--bits-p", "12"])


def test_selection_keeps_every_sample(hadron_file):
    h2r = _h2r()
    path, samples = hadron_file
    cols = h2r.load_columns(path, with_x=False, eta_max=1.0, charged=True)
    assert len(cols["soff"]) == 7 and not cols["with_x"]
    flat = [s for ev in samples for s in ev]
    want = []
    for s in flat:
        pt = np.hypot(s["p"][:, 1], s["p"][:, 2])
        eta = np.arcsinh(s["p"][:, 3] / pt)
        want.append(int(((np.abs(eta) < 1) & (s["pid"] != 22)).sum()))
    assert list(np.diff(cols["soff"])) == want
    assert np.all(cols["pid"] != 22)


@pytest.mark.parametrize("fmt,bits_p,bits_x", [("ttree", 0, 0), ("ttree", 12, 8),
                                               ("rntuple", 0, 0), ("rntuple", 12, 8)])
def test_root_writers(hadron_file, tmp_path, fmt, bits_p, bits_x):
    """Float and truncated storage (Float16_t / Real32Trunc) both give back the h5 values
    exactly: those are already rounded to 12 / 8 bits."""
    pytest.importorskip("ROOT")
    h2r = _h2r()
    path, samples = hadron_file
    out = tmp_path / f"{fmt}{bits_p}.root"
    cols = h2r.load_columns(path)
    h2r.write_root(out, cols, fmt=fmt, bits_p=bits_p, bits_x=bits_x)
    _check(out, samples, cols)
