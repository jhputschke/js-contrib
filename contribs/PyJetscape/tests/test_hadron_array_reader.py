"""HadronFileReader.hadron_array, the row reads of from_h5(units=...) and the
to_awkward options charged / massless / pion_mass.

Run:  python -m pytest contribs/PyJetscape/tests/test_hadron_array_reader.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_particlize_h5 as tp  # noqa: E402  (the synthetic two-seed campaign)
from jetscape.hadrons_h5 import (  # noqa: E402
    CHARGED, HadronFile, HadronFileReader, HadronH5Writer, Hadrons, _read_rows)


# ── reading only the selected rows ───────────────────────────────────────────────────
def test_read_rows_equals_indexing_the_whole_dataset(tmp_path):
    import h5py

    rng = np.random.default_rng(7)
    ref = rng.normal(size=(5000, 4)).astype(np.float32)
    with h5py.File(tmp_path / "d.h5", "w") as f:
        f.create_dataset("a", data=ref, chunks=(256, 4))
    with h5py.File(tmp_path / "d.h5", "r") as f:
        d = f["a"]
        cases = [([], []), ([10], [10]), ([4000, 10, 4000], [4100, 20, 4100]),
                 ([0], [5000]), ([300, 0], [2900, 100])]        # last two: whole read
        for _ in range(20):
            s = rng.integers(0, 5000, rng.integers(1, 12))
            cases.append((s, np.minimum(s + rng.integers(0, 300, len(s)), 5000)))
        for starts, stops in cases:
            want = (np.concatenate([ref[a:b] for a, b in zip(starts, stops)])
                    if len(starts) else ref[:0])
            got = _read_rows(d, starts, stops)
            assert got.dtype == ref.dtype and np.array_equal(got, want)


def _file(tmp_path, n_units=6):
    rng = np.random.default_rng(3)
    path = str(tmp_path / "h_bulk_bg.h5")
    with HadronH5Writer(path, tag="bulk_bg", n_samples=3) as w:
        for u in range(n_units):
            w.append_unit([tp._hads(rng.uniform(-3, 3, rng.integers(0, 6)), 211, u + 1)
                           for _ in range(int(rng.integers(0, 4)))], unit=10 + u, seed=u)
    return path


@pytest.mark.parametrize("units", [None, [4, 1], [2, 2, 0], slice(1, 5, 2), 3, []])
def test_from_h5_units_reads_the_same_hadrons(tmp_path, units):
    from jetscape import HadronArray

    path = _file(tmp_path)
    full = HadronArray.from_h5(path)
    uo, so = full.info["unit_offsets"], full.sample_offsets
    sel = np.atleast_1d(np.arange(len(uo) - 1)[units]) if units is not None \
        else np.arange(len(uo) - 1)
    rows = [np.arange(so[uo[u]], so[uo[u + 1]]) for u in sel]
    idx = np.concatenate(rows) if rows else np.zeros(0, int)
    counts = [np.diff(so[uo[u]:uo[u + 1] + 1]) for u in sel]
    for got in (HadronArray.from_h5(path, units=units), Hadrons.from_h5(path, units=units)):
        assert np.array_equal(got.pid, full.pid[idx])
        assert np.array_equal(got.p, full.p[idx]) and np.array_equal(got.x, full.x[idx])
        assert np.array_equal(np.diff(got.sample_offsets),
                              np.concatenate(counts) if counts else np.zeros(0, int))
    a = HadronArray.from_h5(path, units=units)
    assert np.array_equal(np.diff(a.info["unit_offsets"]), [len(c) for c in counts])
    assert np.array_equal(a.info["units"]["unit"], 10 + sel)


# ── HadronFileReader.hadron_array ─────────────────────────────────────────────────────
def test_hadron_array_events_samples_and_backgrounds(tmp_path):
    a, b = tp._two_seeds(tmp_path)
    with HadronFileReader(str(tmp_path)) as r:
        arr = r.hadron_array("bulk_jet")                  # 4 events x 2 samples x 3
        assert np.array_equal(arr.info["units"]["event"], [0, 1, 2, 3])
        assert np.array_equal(np.diff(arr.info["unit_offsets"]), [2, 2, 2, 2])
        assert len(arr) == 24 and arr.p.dtype == np.float32
        for e in range(4):                                 # sample k = jet_event's bulk
            for k in range(2):
                s = 2 * e + k
                lo, hi = arr.sample_offsets[s], arr.sample_offsets[s + 1]
                ev = r.jet_event(e, k, fragments=False)
                assert np.array_equal(arr.p[lo:hi], ev["p"])

        one = r.hadron_array("bulk_jet", events=3, samples=1)
        assert np.array_equal(one.p, r.jet_event(3, 1, fragments=False)["p"])
        assert list(one.info["samples"]) == [1]
        assert one.info["units"]["file"][0] == 1 and one.info["units"]["local_event"][0] == 1

        # bulk_bg: events 0 and 1 (seed A) share background 0, each gets its copy
        bg = r.hadron_array("bulk_bg", events=[1, 0], samples=slice(0, 1))
        assert np.array_equal(bg.info["units"]["event"], [0, 1])
        assert np.array_equal(bg.info["units"]["file_unit"], [0, 0])
        assert np.array_equal(bg.p, np.concatenate([r.background_event(0, 0)["p"]] * 2))

        aw = r.hadron_array("jet_frag", events=slice(0, 3)).to_awkward(by_unit=True)
        assert str(aw.type).startswith("3 * var * var * Momentum4D")


def test_hadron_array_masks_like_hist(tmp_path):
    tp._two_seeds(tmp_path)
    with HadronFileReader(str(tmp_path)) as r:
        assert len(r.hadron_array("bulk_bg", mask="pi-")) == 4 * 2 * 2
        assert len(r.hadron_array("bulk_bg", mask="pi+")) == 0
        none = r.hadron_array("bulk_jet", events=0, mask=lambda ev, info: ev.pt > 5)
        assert len(none) == 0 and none.n_samples == 2      # empty samples stay

        def near_jet(ev, info):                            # depends on the event
            ini = info.initiators()[0]
            d = np.mod(ev.phi - np.arctan2(ini[4], ini[3]) + np.pi, 2 * np.pi) - np.pi
            return np.abs(d) < 0.3

        # the shared background (phi 0.5) is near event A0's jet (0.3), not A1's (2.0)
        bg = r.hadron_array("bulk_bg", events=[0, 1], mask=near_jet)
        assert np.array_equal(np.diff(bg.sample_offsets), [2, 2, 0, 0])
        n, _ = r.total("bulk_bg", mask=near_jet, events=[0, 1])
        assert n == pytest.approx(len(bg) / bg.n_samples)


def test_hadron_array_empty_surfaces_and_bad_samples(tmp_path):
    tp._two_seeds(tmp_path, empty_jet_b=(1,))              # event 3: no bulk_jet samples
    with HadronFileReader(str(tmp_path)) as r:
        arr = r.hadron_array("bulk_jet", samples=0)
        assert np.array_equal(np.diff(arr.info["unit_offsets"]), [1, 1, 1, 0])
        with pytest.raises(IndexError):
            r.hadron_array("bulk_jet", events=0, samples=2)
        assert r.hadron_array("bulk_jet", events=0, samples=-1).info["samples"][0] == 1
        with pytest.raises(ValueError):
            r.hadron_array("bulk", events=0)
        nox = r.hadron_array("jet_frag", events=0, positions=False)
        assert nox.x is None and len(nox) == 1


def test_hadron_array_reads_only_what_it_needs(tmp_path, monkeypatch):
    """The reader reads the selected rows only, not whole datasets."""
    import jetscape.hadrons_h5 as hh

    tp._two_seeds(tmp_path)
    calls = []
    orig = hh._read_rows
    monkeypatch.setattr(hh, "_read_rows",
                        lambda d, s, e: calls.append(int(np.sum(np.subtract(e, s))))
                        or orig(d, s, e))
    with HadronFileReader(str(tmp_path)) as r:
        r.hadron_array("bulk_jet", events=1, samples=0)
    assert calls and max(calls) == 6                       # event 1's two samples of 3


# ── to_awkward: charged, massless, pion_mass ──────────────────────────────────────────
def _mixed():
    from jetscape import HadronArray

    pid = np.array([211, -321, 22, 111, 2212, 11, 130], np.int32)
    p = np.zeros((7, 4), np.float32)
    p[:, 1], p[:, 2], p[:, 3] = 0.3, -0.4, 1.2
    p[:, 0] = 2.0                                          # stored E: not the massless E
    return HadronArray.from_dict({"pid": pid, "pstat": np.zeros(7, np.int32), "p": p,
                                  "sample_counts": [3, 4]})


def test_to_awkward_charged_massless_and_pion_mass():
    pytest.importorskip("awkward")
    pytest.importorskip("vector")
    import awkward as ak

    from jetscape.awkward_views import M_PION

    h = _mixed()
    want_charged = np.isin(np.abs(h.pid), CHARGED)
    a = h.to_awkward(charged=True)
    assert np.array_equal(ak.flatten(a.charged).to_numpy(), want_charged)
    assert np.array_equal(ak.flatten(a.E).to_numpy(), h.p[:, 0])      # stored E kept
    pp = np.sum(h.p[0, 1:].astype(np.float64) ** 2)
    m0 = h.to_awkward(massless=True)
    assert np.allclose(ak.flatten(m0.E).to_numpy(), np.sqrt(pp), rtol=0, atol=1e-12)
    e0 = ak.flatten(m0.E).to_numpy()
    assert np.allclose(e0 ** 2 - pp, 0, atol=1e-12)        # float64; vector's .mass
                                                            # rounds px, py, pz in float32
    mp = h.to_awkward(pion_mass=True)
    m = np.where(want_charged, M_PION, 0.0)
    assert np.allclose(ak.flatten(mp.E).to_numpy(), np.sqrt(pp + m ** 2), rtol=0, atol=1e-12)
    assert np.allclose(ak.flatten(mp.E).to_numpy() ** 2 - pp, m ** 2, atol=1e-12)
    with pytest.raises(ValueError):
        h.to_awkward(massless=True, pion_mass=True)


def test_to_awkward_by_unit_on_a_file(tmp_path):
    pytest.importorskip("awkward")
    pytest.importorskip("vector")
    from jetscape import HadronArray

    path = _file(tmp_path)
    with HadronFile(path) as hf:
        per_unit = np.diff(hf.unit_offsets)
    a = HadronArray.from_h5(path)
    assert len(a.to_awkward()) == per_unit.sum()                       # samples in a row
    assert len(a.to_awkward(by_unit=True)) == len(per_unit)            # units x samples


def test_arrays_from_files_are_native_and_cluster_with_fastjet(tmp_path):
    """An HDF5 dataset's dtype is marked '<', and so is an array allocated with it; fastjet
    3.5's array interface fails on that (its correct_byteorder returns a dtype), so the
    reads hand out native ('=') arrays, as h5py's own reads are."""
    from jetscape import HadronArray

    tp._two_seeds(tmp_path)
    with HadronFileReader(str(tmp_path)) as r:
        arrays = [r.hadron_array("bulk_jet"), HadronArray.from_h5(r.stems[0] +
                                                                   "_hadrons_bulk_jet.h5")]
    for a in arrays:
        for x in (a.p, a.pid, a.sample_offsets, a.info["unit_offsets"]):
            assert x.dtype.byteorder in ("=", "|")
    fj = pytest.importorskip("fastjet")
    pytest.importorskip("vector")
    jd = fj.JetDefinition(fj.antikt_algorithm, 0.4)
    for a in arrays:
        for kw in ({}, {"massless": True}, {"pion_mass": True}):
            jets = fj.ClusterSequence(a.to_awkward(**kw), jd).inclusive_jets()
            assert len(jets) == a.n_samples
