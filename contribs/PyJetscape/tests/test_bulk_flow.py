"""example/analysis/bulk (flow.py, bulk_obs.py): Nch, [p_T], sigma([p_T]) and v_n{2} with an
eta gap on synthetic hadron files (run_prod_jet.py --hadrons-only layout, written with
HadronH5Writer): an imposed v_2, v_3 and a known [p_T] and its event-by-event spread are
recovered within the jackknife errors; the paired difference of identical legs is zero; the
fragments enter the compositions as hadronize.py's reader convention says (fragmentation
k mod n_frag); files without jet_frag (minimum bias) or extra_frag work."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "example" / "analysis" / "bulk"))

import bulk_obs  # noqa: E402
import flow  # noqa: E402
from jetscape.hadrons_h5 import HadronFileReader, HadronH5Writer  # noqa: E402


def _hadrons(pt, eta, phi, pid):
    pt, eta, phi = (np.asarray(v, np.float64) for v in (pt, eta, phi))
    p = np.stack([pt * np.cosh(eta), pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)], 1)
    return {"pid": np.asarray(pid, np.int32), "pstat": np.zeros(len(pt), np.int32),
            "p": p.astype(np.float32), "x": np.zeros((len(pt), 4), np.float32)}


def _flow_sample(rng, M, v2, v3, psi2, psi3, scale, eta_max=2.5):
    """M hadrons, dN/dphi ~ 1 + 2 v2 cos 2(phi - psi2) + 2 v3 cos 3(phi - psi3), eta uniform
    in |eta| < eta_max, p_T = scale x U(0.5, 1.5) (mean: scale); 20 % neutral (pid 111) and
    a few charged at p_T = 5 GeV, both to be ignored."""
    fmax = 1 + 2 * v2 + 2 * v3
    phi = np.zeros(0)
    while len(phi) < M:
        x = rng.uniform(-np.pi, np.pi, 2 * M)
        f = 1 + 2 * v2 * np.cos(2 * (x - psi2)) + 2 * v3 * np.cos(3 * (x - psi3))
        phi = np.concatenate([phi, x[rng.uniform(0, fmax, len(x)) < f]])
    phi = phi[:M]
    eta = rng.uniform(-eta_max, eta_max, M)
    pt = scale * rng.uniform(0.5, 1.5, M)
    pid = np.where(rng.uniform(size=M) < 0.2, 111, rng.choice([211, -211, 321, 2212], M))
    hard = 5
    return _hadrons(np.concatenate([pt, np.full(hard, 5.0)]),
                    np.concatenate([eta, np.zeros(hard)]),
                    np.concatenate([phi, np.zeros(hard)]),
                    np.concatenate([pid, np.full(hard, 211)]))


def _write_job(stem, bulk, bg=None, bg_unit=None, jet_frag=None, extra_frag=None,
               n_coll=None, eta_max=None):
    """A run_prod_jet.py --hadrons-only production file: ``bulk`` per event a list of
    samples (bulk_jet), ``bg`` {background unit: samples}, ``bg_unit`` per event (default:
    its own), ``jet_frag`` / ``extra_frag`` per event a list of samples (None: no file)."""
    n = len(bulk)
    bg_unit = list(range(n)) if bg_unit is None else list(bg_unit)
    n_coll = [100 + e for e in range(n)] if n_coll is None else list(n_coll)
    attrs = {"hadrons_only": True, "source_uuid": f"uuid-{Path(stem).name}", "prod_seed": 1}
    record = [{"bg_unit": bg_unit[e], "n_coll": n_coll[e]} for e in range(n)]
    with HadronH5Writer(f"{stem}_hadrons_bulk_jet.h5", tag="bulk_jet",
                        n_samples=len(bulk[0]), attrs=attrs, eta_max=eta_max) as w:
        for e, samples in enumerate(bulk):
            w.append_unit(samples, unit=e, event=e, seed=e, **record[e])
    if bg is not None:
        with HadronH5Writer(f"{stem}_hadrons_bulk_bg.h5", tag="bulk_bg", n_samples=0,
                            attrs=attrs, eta_max=eta_max) as w:
            for u, samples in bg.items():
                w.append_unit(samples, unit=u, event=bg_unit.index(u), seed=u)
    if jet_frag is not None:
        with HadronH5Writer(f"{stem}_hadrons_jet_frag.h5", tag="jet_frag",
                            n_samples=len(jet_frag[0]), attrs=attrs) as w:
            for e, samples in enumerate(jet_frag):
                w.append_unit(samples, unit=e, event=e, seed=e, **record[e])
    if extra_frag is not None:
        with HadronH5Writer(f"{stem}_hadrons_extra_frag.h5", tag="extra_frag",
                            n_samples=len(extra_frag[0]), attrs=attrs) as w:
            for e, samples in enumerate(extra_frag):
                scat = np.array([[1, 12.0, 111, 0, 0, 2, 40.0, 24.0],
                                 [2, 11.0, 111, 0, 0, 2, 30.0, 22.0]])
                w.append_unit(samples, unit=e, event=e, seed=e, scatterings=scat)
    return stem


V2, V3, SIGMA_SCALE = 0.08, 0.04, 0.01


@pytest.fixture(scope="module")
def flow_job(tmp_path_factory):
    """200 events x 4 oversamples with v2, v3 about random planes and an event scale of p_T
    (mean 0.8, spread SIGMA_SCALE); bulk_bg identical to bulk_jet."""
    rng = np.random.default_rng(12345)
    d = tmp_path_factory.mktemp("flow")
    n_ev, K, M = 200, 4, 1000
    scales = rng.normal(0.8, SIGMA_SCALE, n_ev)
    bulk = []
    for e in range(n_ev):
        psi2, psi3 = rng.uniform(-np.pi, np.pi, 2)
        bulk.append([_flow_sample(rng, M, V2, V3, psi2, psi3, scales[e]) for _ in range(K)])
    _write_job(d / "AA_seed0001", bulk, bg={e: bulk[e] for e in range(n_ev)})
    return d, scales


@pytest.fixture(scope="module")
def flow_table(flow_job):
    d, _ = flow_job
    with HadronFileReader(str(d)) as r:
        return flow.event_table(r, flow.Cuts(flow_eta_max=2.5))


def test_imposed_flow_and_mean_pt_are_recovered(flow_job, flow_table):
    _, scales = flow_job
    res = flow.class_averages(flow_table)
    for c in flow.COMPOSITIONS:
        for name, true in (("v2", V2), ("v3", V3)):
            v, e = res[f"{c}/{name}"]
            assert 0 < e < 0.1 * true
            assert abs(v - true) < 4 * e, (c, name, v, e)
        v, e = res[f"{c}/pt"]
        assert abs(v - scales.mean()) < 4 * e and e < 0.002
        # N in |eta| < 0.8 at 0.2 < p_T < 3: 80 % charged of 1000 hadrons in |eta| < 2.5
        v, e = res[f"{c}/nch"]
        assert abs(v - 1000 * 0.8 * 0.8 / 2.5) < 5
    assert res["n_blocks"][0] == 100 and res["full/n_events"][0] == 200


def test_sigma_pt_noise_is_removed(flow_job, flow_table):
    _, scales = flow_job
    res = flow.class_averages(flow_table)
    true = scales.std(ddof=1)
    dyn, e = res["full/sigma_pt_dyn"]
    raw, e_raw = res["full/sigma_pt"]
    one, _ = res["full/sigma_pt_one"]
    assert abs(dyn - true) < 4 * e, (dyn, e, true)
    assert raw - true > 3 * e_raw              # the oversample average still has noise
    assert one > raw > dyn


def test_default_cuts_give_the_same_flow_with_larger_errors(flow_job, flow_table):
    d, _ = flow_job
    with HadronFileReader(str(d)) as r:
        t = flow.event_table(r, flow.Cuts(), events=range(100))
    res = flow.class_averages(t)
    wide = flow.class_averages(flow_table)
    v, e = res["full/v2"]
    assert abs(v - V2) < 4 * e and e > wide["full/v2"][1]


def test_identical_legs_give_a_zero_difference(flow_table):
    res = flow.class_averages(flow_table)
    for dname in flow.DELTAS:
        for q in ("dnch", "rdnch", "dpt", "dc2", "dc3", "dv2", "dv3"):
            assert res[f"{dname}/{q}"] == (0.0, 0.0), (dname, q)


def test_select_by_events_and_ncoll(flow_table):
    t = flow_table
    assert flow.select(t, events=[3, 5]).sum() == 2
    m = flow.select(t, n_coll=(150, None))
    assert m.sum() == 150 and t["n_coll"][m].min() == 150
    top = flow.select(t, n_coll_percent=(0, 10))
    assert top.sum() == 20 and t["n_coll"][top].min() == 280
    assert (flow.select(t, n_coll_percent=(10, 100)) & top).sum() == 0
    res = flow.class_averages(t, top)
    assert res["full/n_events"][0] == 20 and res["n_blocks"][0] == 20


def _flat(n, eta=0.0, pt=1.0, phi=0.0, pid=211):
    return _hadrons(np.full(n, pt), np.full(n, eta), np.full(n, phi), np.full(n, pid))


def _fragment_job(d, jet=True, extra=True, bg=True, reuse=False):
    """4 events x 4 oversamples of 50 bulk hadrons; jet_frag with 2 fragmentations of 3 and
    4 hadrons in the window (+1 at eta = 3), extra_frag with 2 of 5 and 7; bulk_bg identical
    to bulk_jet (``reuse``: events 0 and 1 share background 0)."""
    rng = np.random.default_rng(7)
    bulk = [[_hadrons(rng.uniform(0.3, 2, 50), rng.uniform(-1, 1, 50),
                      rng.uniform(-np.pi, np.pi, 50), np.full(50, 211)) for _ in range(4)]
            for _ in range(4)]
    bg_unit = [0, 0, 2, 3] if reuse else None
    backgrounds = {u: bulk[u] for u in sorted(set(bg_unit or range(4)))} if bg else None
    frag = [[_concat(_flat(3 + j, pt=2.0), _flat(1, eta=3.0)) for j in range(2)]
            for _ in range(4)]
    ext = [[_flat(5 + 2 * j, eta=0.1, pt=1.5) for j in range(2)] for _ in range(4)]
    return _write_job(d / "S_seed0001", bulk, bg=backgrounds, bg_unit=bg_unit,
                      jet_frag=frag if jet else None, extra_frag=ext if extra else None)


def _concat(*evs):
    return {k: np.concatenate([e[k] for e in evs]) for k in evs[0]}


def _table(d, **kw):
    with HadronFileReader(str(d)) as r:
        return flow.event_table(r, **kw)


def test_fragments_enter_the_compositions(tmp_path):
    _fragment_job(tmp_path)
    t = _table(tmp_path)
    # fragmentation k mod 2: 3, 4, 3, 4 jet fragments; 5, 7, 5, 7 extras
    assert np.allclose(t["no_extras.nch"] - t["bulk_jet.nch"], 3.5)
    assert np.allclose(t["full.nch"] - t["bulk_jet.nch"], 9.5)
    assert np.array_equal(t["n_extra"], [2] * 4) and np.allclose(t["extra_E"], 70.0)
    res = flow.class_averages(t)
    assert res["delta_full/dnch"][0] == pytest.approx(9.5)
    assert res["delta_no_extras/dnch"][0] == pytest.approx(3.5)
    assert res["delta_bulk_jet/dnch"] == (0.0, 0.0)
    assert res["delta_full/dpt"][0] > res["delta_no_extras/dpt"][0] > 0
    assert res["delta_full/dc2"][0] == 0.0          # no fragments at |eta| > 0.5
    # with a small gap the extras (eta = 0.1) are in subevent B: the Q-vectors of the
    # tags add per sample before the pair sum, as for the whole event taken at once
    cuts = flow.Cuts(eta_gap=0.1, flow_eta_max=0.8)
    t2 = _table(tmp_path, cuts=cuts)
    assert np.all(t2["full.c"] != t2["bulk_jet.c"])
    with HadronFileReader(str(tmp_path)) as r:
        for e in range(4):
            c = np.zeros(2)
            for k in range(4):
                p = r.jet_event(e, k)["p"].astype(np.float64)
                pt, phi = np.hypot(p[:, 1], p[:, 2]), np.arctan2(p[:, 2], p[:, 1])
                eta = np.arcsinh(p[:, 3] / pt)
                inside = (pt > 0.2) & (pt < 3)
                a = inside & (eta < -0.05) & (eta > -0.8)
                b = inside & (eta > 0.05) & (eta < 0.8)
                for j, n in enumerate((2, 3)):
                    c[j] += np.real(np.exp(1j * n * phi[a]).sum()
                                    * np.conj(np.exp(1j * n * phi[b]).sum())) / 4
            assert np.allclose(t2["full.c"][e], c, rtol=1e-9, atol=1e-9)


def test_minimum_bias_files_without_jet_frag(tmp_path):
    _fragment_job(tmp_path, jet=False)
    t = _table(tmp_path)
    assert np.array_equal(t["no_extras.nch"], t["bulk_jet.nch"])
    assert np.allclose(t["full.nch"] - t["bulk_jet.nch"], 6.0)


def test_files_without_extra_frag(tmp_path):
    _fragment_job(tmp_path, extra=False)
    t = _table(tmp_path)
    for col in ("nch", "pt", "c", "w"):
        assert np.array_equal(t[f"full.{col}"], t[f"no_extras.{col}"])
    assert np.array_equal(t["n_extra"], [0] * 4) and np.all(t["extra_E"] == 0)


def test_files_without_bulk_bg(tmp_path):
    _fragment_job(tmp_path, bg=False)
    res = flow.class_averages(_table(tmp_path))
    assert res["bulk_bg/n_events"][0] == 0 and res["delta_full/n_events"][0] == 0
    assert np.isnan(res["delta_full/dnch"][0]) and res["full/n_events"][0] == 4


def test_a_reused_background_is_one_jackknife_unit(tmp_path):
    _fragment_job(tmp_path, reuse=True)
    t = _table(tmp_path)
    assert t["group"][0] == t["group"][1] and len(np.unique(t["group"])) == 3
    assert t["bulk_bg.nch"][1] == t["bulk_bg.nch"][0]
    res = flow.class_averages(t)
    assert res["n_blocks"][0] == 3 and res["bulk_bg/n_events"][0] == 4


def test_cuts_beyond_the_stored_eta_range_are_refused(tmp_path):
    bulk = [[_flat(5, eta=0.2)] * 2]
    _write_job(tmp_path / "C_seed0001", bulk, bg={0: bulk[0]}, eta_max=0.5)
    with HadronFileReader(str(tmp_path)) as r:
        with pytest.raises(ValueError, match="keep only"):
            flow.event_table(r)
        assert flow.event_table(r, flow.Cuts(eta_max=0.5, eta_gap=0.4))["full.nch"][0] == 5
    with pytest.raises(ValueError, match="empty subevents"):
        flow.Cuts(eta_gap=2.0)


def test_cli_writes_json_npz_and_the_event_table(tmp_path, capsys):
    _fragment_job(tmp_path)
    out = tmp_path / "res.json"
    table = tmp_path / "events.npz"
    bulk_obs.main([str(tmp_path), "--out", str(out), "--table", str(table), "--quiet",
                   "--ncoll-classes", "0", "50", "100"])
    text = capsys.readouterr().out
    assert "delta_full" in text and "Ncoll 0-50 %" in text
    doc = json.loads(out.read_text())
    assert set(doc["classes"]) == {"all", "Ncoll 0-50 %", "Ncoll 50-100 %"}
    assert doc["classes"]["all"]["delta_full"]["dnch"][0] == pytest.approx(9.5)
    assert doc["classes"]["Ncoll 0-50 %"]["full"]["n_events"][0] == 2
    z = np.load(table)
    assert len(z["full__nch"]) == 4 and json.loads(str(z["cuts"]))["eta_max"] == 0.8
    npz = tmp_path / "res.npz"
    bulk_obs.main([str(tmp_path / "S_seed0001_hadrons_bulk_jet.h5"), "--out", str(npz),
                   "--quiet", "--events", "0:2"])
    z = np.load(npz)
    assert z["all|full/n_events"][0] == 2
