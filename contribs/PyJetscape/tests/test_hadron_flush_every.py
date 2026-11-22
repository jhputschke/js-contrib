"""
tests/test_hadron_flush_every.py

HadronH5Writer(flush_every=N): the units are committed N at a time (one write per ragged
group and per units/ key, one flush), for cheap events such as p+p in vacuum
(run_prod_oo_isr.py --vacuum).  The file must be the one written unit by unit, with
initiators, hadronization input, empty units and a key that first appears late; and a
writer that is never closed (a crashed job) must leave the committed units readable.
No compiled extension needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import hdf5plugin  # noqa: E402,F401
import h5py  # noqa: E402

from jetscape.hadrons_h5 import HadronFile, HadronH5Writer  # noqa: E402
from jetscape.showers import INITIATOR_COLUMNS  # noqa: E402


def _unit(rng, k):
    n = 0 if k % 5 == 3 else int(rng.integers(1, 40))       # some empty units
    had = {"pid": rng.choice([211, -211, 111, 321, 2212], n).astype(np.int32),
           "pstat": np.zeros(n, np.int32),
           "p": rng.normal(size=(n, 4)).astype(np.float32),
           "x": rng.normal(size=(n, 4)).astype(np.float32)}
    ini = rng.normal(size=(int(rng.integers(2, 6)), len(INITIATOR_COLUMNS)))
    m = int(rng.integers(3, 9))
    hin = {"pid": rng.choice([1, 2, 21], m), "col": rng.integers(0, 600, m),
           "acol": rng.integers(0, 600, m), "p": rng.normal(size=(m, 4)),
           "mass": np.zeros(m)}
    scal = {"pthat": float(rng.uniform(5, 60)), "seed": int(rng.integers(1, 2**31)),
            "glauber_npart": 2}
    if k >= 4:                                             # a key first seen at unit 4
        scal["colour_open_ends"] = int(k % 3)
    return had, ini, hin, scal


def _write(path, n_units, flush_every, close=True):
    rng = np.random.default_rng(7)
    w = HadronH5Writer(path, tag="jet_frag", n_samples=1, initiators=True,
                       hadronization_input=True, flush_every=flush_every)
    for k in range(n_units):
        had, ini, hin, scal = _unit(rng, k)
        w.append_unit([had], initiators=ini, hadronization_input=hin, unit=k, event=k, **scal)
    if close:
        w.close()
    return w


def _datasets(path):
    out = {}
    with h5py.File(path) as f:
        f.visititems(lambda name, obj: out.__setitem__(name, obj[()])
                     if isinstance(obj, h5py.Dataset) else None)
        attrs = dict(f.attrs)
    return out, attrs


def test_batched_file_equals_unit_by_unit(tmp_path):
    a, b = tmp_path / "a.h5", tmp_path / "b.h5"
    _write(a, 23, 1)
    _write(b, 23, 7)                     # 23 = 3 commits of 7 + 2 at close
    da, aa = _datasets(a)
    db, ab = _datasets(b)
    assert da.keys() == db.keys()
    for k in da:
        assert np.array_equal(da[k], db[k]), k
    assert aa["nunits_written"] == ab["nunits_written"] == 23
    assert bool(ab["complete"])
    with HadronFile(b) as f:
        assert f.n_units == 23
        assert np.isnan(f.units["colour_open_ends"][:4].astype(float)).all() or \
            (f.units["colour_open_ends"][:4] == -1).all()


def test_unclosed_writer_keeps_committed_units(tmp_path):
    p = tmp_path / "crash.h5"
    w = _write(p, 17, 5, close=False)    # 3 commits (15 units), 2 units held
    w.f.flush()
    with h5py.File(p) as f:
        assert int(f.attrs["nunits_written"]) == 15
        assert not bool(f.attrs["complete"])
        assert f["hadrons/unit_offsets"].shape[0] == 16
        assert f["units/pthat"].shape[0] == 15
        assert f["initiators/offsets"].shape[0] == 16
    w.f.close()
    ref = tmp_path / "ref.h5"
    _write(ref, 15, 1)
    with HadronFile(p) as f, HadronFile(ref) as g:
        assert f.n_units == g.n_units == 15
        for u in range(15):
            a, b = f.sample_event(u, 0), g.sample_event(u, 0)
            assert np.array_equal(a["pid"], b["pid"]) and np.array_equal(a["p"], b["p"])
