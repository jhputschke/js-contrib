"""wake_iso: the binning against numpy.histogramdd, the frames and harmonics of one leg, the
recoil choice, and the per-Z means, paired differences and jackknife of stored group sums."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "python"))
sys.path.insert(0, str(HERE.parent / "example" / "analysis"))           # jetcorr
sys.path.insert(0, str(HERE.parent / "example" / "analysis" / "zjet"))

import wake_iso as wi  # noqa: E402


def test_bin_follows_histogramdd():
    rng = np.random.default_rng(0)
    edges = wi.DETA
    x = np.concatenate([rng.uniform(-5.5, 5.5, 5000), edges, [edges[-1], edges[0] - 1e-9]])
    i = wi._bin(x, edges)
    h = np.histogram(x, bins=edges)[0]
    assert np.array_equal(np.bincount(i[i >= 0], minlength=len(edges) - 1), h)
    assert i[-1] == -1 and i[-2] == len(edges) - 2


def test_frame_sums_harmonics_and_sides():
    rng = np.random.default_rng(1)
    n = 4000
    pt = rng.uniform(0.2, 12.0, n)
    deta = rng.uniform(-5.0, 5.0, n)
    dphi = rng.uniform(-np.pi / 2, 3 * np.pi / 2, n)
    toward = dphi < np.pi / 2
    m, h, s = wi.frame_sums(pt, deta, dphi, toward)
    assert np.allclose(m.sum(-1), h[..., 0])                      # c_0 is the count
    ip, ie = wi._bin(pt, wi.PT_EDGES), wi._bin(deta, wi.DETA)
    ok = (ip >= 0) & (ie >= 0)
    for n_ in range(1, wi.NH):                                    # exact, not binned
        ref = np.zeros((wi.N_PT, wi.N_ETA))
        np.add.at(ref, (ip[ok], ie[ok]), np.cos(n_ * dphi[ok]))
        assert np.allclose(h[..., n_], ref)
    js = wi._bin(pt, wi.SPEC_EDGES)
    assert s[0].sum() == np.sum((js >= 0) & toward)
    assert s[1].sum() == np.sum((js >= 0) & ~toward)


def test_flat_tracks_have_no_harmonics_dip_or_width():
    """One track on every bin centre: c_n = 0 for n >= 1 (to rounding), no dip, the same
    density in the core and the wings."""
    e, p = np.meshgrid(wi.ETA_C, wi.PHI_C, indexing="ij")
    m, h, _ = wi.frame_sums(np.full(e.size, 1.5), e.ravel(), p.ravel(), np.zeros(e.size, bool))
    assert np.allclose(h[..., 1:].sum(1), 0.0, atol=1e-9)
    sums = {}
    for c in ("wake", "frag"):
        for f in wi.FRAMES:
            sums[f"map_{c}_{f}"], sums[f"harm_{c}_{f}"] = m[None], h[None]
            sums[f"spec_{c}_{f}"] = np.zeros((1, 2, wi.N_SPEC))
        sums[f"tot_{c}"] = np.zeros((1, len(wi.TOTALS)))
    v = wi.View(_maps(_pack(sums, np.ones(1))))
    for frame in ("Z", "J"):
        dip, rho = wi.dip(v, "wake", frame)
        assert np.allclose(dip[1], 0.0) and rho[1, 0] > 0
    cw = wi.core_wings(v, "wake")
    assert np.allclose(cw[1, 0], cw[1, 1])


def test_leg_sums_frames_and_totals():
    """The Z frame and the recoil (J) frame of two tracks, away from bin edges; ZJ is Z; only
    |eta| < 2.4 charged tracks are binned, every hadron counts in the totals."""
    z = (0.3, 1.0)                                       # y_Z, phi_Z
    rec = (40.0, -0.5, 1.0 + np.pi + 0.1)                 # pt, eta, phi of the recoil
    ev = _Ev(pt=[1.5, 1.5, 3.0], eta=[0.35, -0.45, 2.5], phi=[1.05, 1.0 + np.pi + 0.05, 0.0],
             E=[2.0, 2.0, 20.0], charged=[True, True, True])
    out, tot = wi.leg_sums(ev, z, rec, in_j=True)
    mz, hz, sz = out["Z"]
    mj, _, sj = out["J"]
    assert out["ZJ"] is out["Z"]
    assert mz.sum() == 2 and mz[1, 24, 8] == 1 and mz[1, 20, 24] == 1   # deta 0.05 / -0.75
    assert mj.sum() == 2 and mj[1, 28, 23] == 1 and mj[1, 24, 7] == 1   # dphi pi - 0.05 / -0.05
    assert np.allclose(hz[1].sum(0)[1], 0.0, atol=1e-12)               # cos x + cos(pi + x)
    assert sz[0].sum() == 1 and sz[1].sum() == 1                        # toward Z / recoil
    assert sj[0].sum() == 1 and sj[1].sum() == 1
    phi = np.asarray(ev.phi)
    pt = np.asarray(ev.pt)
    assert np.allclose(tot, [3, 24.0, 6.0, (pt * np.cos(phi - z[1])).sum(),
                             (pt * np.cos(phi - rec[2])).sum()])


def test_recoil_is_the_hardest_parton():
    ini = np.zeros((3, 11))
    ini[:, 1] = [13, 21, 2]                         # a muon (hardest), a gluon, a u quark
    ini[:, 3:6] = [[90.0, 0, 5.0], [0, 30.0, 10.0], [10.0, 0, 0.0]]
    pt, eta, phi = wi.recoil(ini)
    assert pt == pytest.approx(30.0) and phi == pytest.approx(np.pi / 2)
    assert eta == pytest.approx(np.arcsinh(10.0 / 30.0))
    assert wi.recoil(ini[:1]) is None


def test_means_paired_differences_and_jackknife():
    rng = np.random.default_rng(2)
    G = 6
    a, W = _random_sums(rng, G), rng.uniform(0.5, 2.0, G)
    b = {k: v.copy() for k, v in a.items()}
    b["map_wake_Z"] = a["map_wake_Z"] + rng.normal(0, 0.1, a["map_wake_Z"].shape)
    ma, mb = _maps(_pack(a, W)), _maps(_pack(b, W))
    v = wi.View(ma)
    assert np.allclose(v.mean("map_wake_Z"), a["map_wake_Z"].sum(0) / W.sum())
    # paired: the difference of identical runs is 0 with 0 error
    d, e = wi.jackknife(lambda V: V[0].mean("map_frag_Z") - V[1].mean("map_frag_Z"),
                        [[ma, mb]])
    assert np.allclose(d, 0) and np.allclose(e, 0)
    # the leave-one-group-out jackknife by hand
    f = lambda V: V[0].mean("map_wake_Z").sum()                  # noqa: E731
    val, err = wi.jackknife(f, [[ma]])
    reps = np.array([np.delete(a["map_wake_Z"], g, 0).sum() / np.delete(W, g).sum()
                     for g in range(G)])
    assert val == pytest.approx(a["map_wake_Z"].sum() / W.sum())
    assert err == pytest.approx(np.sqrt((G - 1) / G * ((reps - reps.mean()) ** 2).sum()))
    # independent sets add in quadrature
    _, e1 = wi.jackknife(lambda V: V[0].mean("map_wake_Z").sum(), [[ma]])
    _, e2 = wi.jackknife(lambda V: V[0].mean("map_wake_Z").sum(), [[mb]])
    _, e12 = wi.jackknife(lambda V: V[0].mean("map_wake_Z").sum()
                          - V[1].mean("map_wake_Z").sum(), [[ma], [mb]])
    assert e12 == pytest.approx(np.hypot(e1, e2))
    with pytest.raises(ValueError):
        other = _pack(a, W)
        other["group"] = np.array([f"9:{g}" for g in range(G)])
        wi.jackknife(lambda V: 0.0, [[ma, _maps(other)]])


# ── helpers ──────────────────────────────────────────────────────────────────────
class _Ev:
    """The EventHadrons attributes leg_sums uses."""

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, np.asarray(v))

    def __len__(self):
        return len(self.pt)


def _random_sums(rng, G):
    d = {}
    for c in ("wake", "frag"):
        for f in wi.FRAMES:
            d[f"map_{c}_{f}"] = rng.uniform(0, 1, (G, wi.N_PT, wi.N_ETA, wi.N_PHI))
            d[f"harm_{c}_{f}"] = rng.uniform(-1, 1, (G, wi.N_PT, wi.N_ETA, wi.NH))
            d[f"spec_{c}_{f}"] = rng.uniform(0, 1, (G, 2, wi.N_SPEC))
        d[f"tot_{c}"] = rng.uniform(0, 1, (G, len(wi.TOTALS)))
    return d


def _pack(sums, W):
    G = len(W)
    d = {"meta": np.array(json.dumps({"version": wi.VERSION, "vacuum": False, "rec_eta": 1.0})),
         "group": np.array([f"1:{g}" for g in range(G)]), "W": W, "WJ": W, "W2": W ** 2,
         "edrop": np.zeros(G), "N": np.ones(G, int), "NJ": np.ones(G, int)}
    d.update(sums)
    return d


def _maps(d):
    """A Maps from a dict of arrays (through a temporary .npz)."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "maps.npz")
        np.savez(path, **d)
        return wi.Maps(path)
