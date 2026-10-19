"""zjet_mixing: the all-partner kernel and pair rules against brute-force pair sums, the
jackknife replicates, and the explicit (CMS-style, strict) partner draws, on synthetic pools."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "python"))
sys.path.insert(0, str(HERE.parent / "example" / "analysis"))

import zjet_mixing as zm  # noqa: E402

SZ, SL = (zm.N_PT, zm.N_DY, zm.N_DPHI), (zm.N_PT, zm.N_ETA, zm.N_PHI)


def synthetic(n_bg=4, per_bg=(3, 1, 4, 2), seed=0):
    """A pool with random per-event S, rho and tracks -> (Pool, per-event S, per-event rho)."""
    rng = np.random.default_rng(seed)
    bg = np.repeat(np.arange(n_bg) * 7 + 3, per_bg[:n_bg])          # ids need not be dense
    n = len(bg)
    w = rng.uniform(0.2, 3.0, n)
    S = rng.uniform(0, 2, (n, *SZ))
    rho = rng.uniform(0, 1, (n, *SL))
    trk_n = rng.integers(5, 30, n)
    m = int(trk_n.sum())
    d = {"event": np.arange(n), "bg": bg, "w": w, "y": rng.uniform(-2.39, 2.39, n),
         "phi": rng.uniform(-np.pi, np.pi, n), "k_one": np.zeros(n, np.int64),
         "n_samples": np.full(n, 40),
         "S_full": S.astype(np.float32), "S_one": S.astype(np.float32), "trk_n": trk_n,
         "trk_pt": rng.integers(0, zm.N_PT, m).astype(np.int8),
         "trk_eta": rng.uniform(-2.4, 2.4, m).astype(np.float32),
         "trk_phi": rng.uniform(-np.pi, np.pi, m).astype(np.float32)}
    ids = np.unique(bg)
    d["bg_id"] = ids
    d["bg_W"] = np.array([w[bg == b].sum() for b in ids])
    d["bg_W2"] = np.array([(w[bg == b] ** 2).sum() for b in ids])
    for c in zm.CONTENTS:
        d[f"bg_SW_{c}"] = np.array([(w[bg == b, None, None, None] * S[bg == b]).sum(0)
                                    for b in ids])
        d[f"bg_T_{c}"] = np.array([(w[bg == b, None, None, None] ** 2 * S[bg == b]).sum(0)
                                   for b in ids])
        d[f"bg_R_{c}"] = np.array([(w[bg == b, None, None, None] * rho[bg == b]).sum(0)
                                   for b in ids])
    return d, S, rho


def brute(d, S, rho, rule, phi_flat):
    """sum_i w_i (S_i - B_i) / sum w with pair weights w_i w_j, by explicit pairs."""
    w, bg = d["w"], d["bg"]
    num, den = np.zeros(SZ), 0.0
    for i in range(len(w)):
        for j in range(len(w)):
            if rule == "self" and j == i or rule == "background" and bg[j] == bg[i]:
                continue
            x = zm.shift_sum(rho[j], [d["y"][i]], [d["phi"][i]], [1.0], phi_flat)
            num += w[i] * w[j] * x
            den += w[i] * w[j]
    s = (w[:, None, None, None] * S).sum(0) / w.sum()
    if rule == "self":
        # compare.py's convention: the pooled density includes the self pairs (binned,
        # shift_i(rho_i)) and the exact self pairs w_i^2 S_i are subtracted; on real data
        # the two differ only by the binning
        t = (w[:, None, None, None] ** 2 * S).sum(0)
        if phi_flat:
            t = np.repeat(t.sum(-1, keepdims=True) / zm.N_DPHI, zm.N_DPHI, -1)
        sp = sum(zm.shift_sum(rho[i], [d["y"][i]], [d["phi"][i]], [w[i] ** 2], phi_flat)
                 for i in range(len(w)))
        num = num + sp - t
    return s - num / den


def subset(d, keep_bg):
    """The pool without the backgrounds not in ``keep_bg`` (ids)."""
    ev = np.isin(d["bg"], keep_bg)
    bk = np.isin(d["bg_id"], keep_bg)
    out = {}
    for k, v in d.items():
        if k.startswith("bg_"):
            out[k] = v[bk]
        elif k.startswith("trk_") and k != "trk_n":
            off = np.concatenate([[0], np.cumsum(d["trk_n"])])
            out[k] = np.concatenate([v[off[i]:off[i + 1]] for i in np.flatnonzero(ev)])
        else:
            out[k] = v[ev]
    return out


@pytest.mark.parametrize("phi_flat", [False, True])
def test_kernel_equals_z_by_z_shifts(phi_flat):
    rng = np.random.default_rng(3)
    y = rng.uniform(-2.4, 2.4, 200)
    phi = rng.uniform(-np.pi, np.pi, 200)
    w = rng.uniform(0.1, 2.0, 200)
    rho = rng.uniform(0, 1, SL)
    k = zm.ZKernel(y, phi, w)(rho, phi_flat)
    assert np.allclose(k, zm.shift_sum(rho, y, phi, w, phi_flat), rtol=1e-12, atol=1e-12)


def test_kernel_conserves_tracks():
    """Every lab bin lands in some Z-frame bin: the totals agree."""
    rng = np.random.default_rng(4)
    rho = rng.uniform(0, 1, SL)
    out = zm.ZKernel([0.3, -1.7], [0.2, 2.9], [1.0, 2.0])(rho)
    assert np.allclose(out.sum(axis=(1, 2)), 3.0 * rho.sum(axis=(1, 2)))


@pytest.mark.parametrize("rule", zm.RULES)
@pytest.mark.parametrize("phi_flat", [False, True])
def test_all_partners_equals_pair_sums(rule, phi_flat):
    d, S, rho = synthetic()
    D, _ = zm.all_partners(zm.Pool(d), "full", rule, phi_flat)
    assert np.allclose(D, brute(d, S, rho, rule, phi_flat), rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("rule", zm.RULES)
def test_jackknife_replicates_are_the_reduced_pools(rule):
    d, _, _ = synthetic()
    pool = zm.Pool(d)
    blocks = pool.blocks(3)
    _, reps = zm.all_partners(pool, "zc", rule, False, blocks)
    for b, rep in zip(blocks, reps):
        keep = np.setdiff1d(pool.bg_id, pool.bg_id[b])
        ref, _ = zm.all_partners(zm.Pool(subset(d, keep)), "zc", rule, False)
        assert np.allclose(rep, ref, rtol=1e-10, atol=1e-10)


def test_exclusion_bias_is_the_rule_difference():
    """Leaving the own background out shifts S - B by exclusion_bias's flat level."""
    d, _, _ = synthetic(seed=2)
    pool = zm.Pool(d)
    D_bg, _ = zm.all_partners(pool, "full", "background")
    D_all, _ = zm.all_partners(pool, "full", "all")
    level = (zm.observables(D_bg)[0] - zm.observables(D_all)[0]).mean(axis=-1)
    assert np.allclose(level, zm.exclusion_bias(pool)["pooled"], rtol=1e-10)


def test_single_background_has_no_partners():
    d, _, _ = synthetic(n_bg=1, per_bg=(4,))
    with pytest.raises(ValueError, match="no partner pairs"):
        zm.all_partners(zm.Pool(d), "full", "background")


def test_explicit_partners_unique_backgrounds_never_own():
    d, _, _ = synthetic(n_bg=4, per_bg=(3, 1, 4, 2))
    pool = zm.Pool(d)
    with pytest.warns(RuntimeWarning, match="fewer than 10"):
        res = zm.explicit_partners(pool, np.arange(pool.n), np.arange(pool.n), pool.w,
                                   n_mix=10, seed=5)
    for i, js in enumerate(res["partners"]):
        b = pool.zb[js]
        assert pool.zb[i] not in b
        assert len(set(b)) == len(b) == 3            # all the other three backgrounds
    res = zm.explicit_partners(pool, np.arange(pool.n), np.arange(pool.n), pool.w, n_mix=2,
                               seed=5)
    assert all(len(set(pool.zb[js])) == 2 for js in res["partners"])


def test_explicit_partner_draw_follows_weights():
    """Backgrounds ~ their summed weight, then events ~ w inside: one partner per Z."""
    d, _, _ = synthetic(n_bg=4, per_bg=(3, 1, 4, 2), seed=1)
    pool = zm.Pool(d)
    z = np.flatnonzero(pool.zb == 0)[:1]                 # one Z of background 0
    counts = np.zeros(pool.n)
    reps = 4000
    res = zm.explicit_partners(pool, np.repeat(z, reps), np.arange(pool.n), pool.w, n_mix=1,
                               seed=2)
    for js in res["partners"]:
        counts[js[0]] += 1
    other = pool.zb != 0
    expect = np.where(other, pool.w, 0) / pool.w[other].sum()
    assert np.allclose(counts / reps, expect, atol=0.03)


def test_explicit_mixing_value():
    """S - B with given partners: the partners' one-sample tracks in the Z's frame."""
    d, _, _ = synthetic()
    pool = zm.Pool(d)
    res = zm.explicit_partners(pool, [0], np.arange(pool.n), pool.w, n_mix=2, seed=7)
    js = res["partners"][0]
    B = sum(zm.z_hist(*pool.tracks(j), pool.y[0], pool.phi[0]) for j in js) / 2
    dphi, dy = zm.observables(pool.S_full[0].astype(float) - B)
    assert np.allclose(res["dphi"][0], dphi) and np.allclose(res["dy"][0], dy)


def test_strict_set_one_per_background():
    d, _, _ = synthetic()
    pool = zm.Pool(d)
    pick, W = zm.strict_set(pool, seed=3)
    assert len(pick) == pool.n_bg
    assert np.array_equal(pool.zb[pick], np.arange(pool.n_bg))
    assert np.allclose(W, pool.bg_W)


def test_observables_flat_is_flat():
    D = np.ones(SZ)
    dphi, dy = zm.observables(D)
    assert np.allclose(dphi, zm.N_DY / np.diff(zm.DPHI))
    assert np.allclose(dy, zm.ZSIDE.sum() / np.diff(zm.DETA))


def test_vacuum_pool_aliases_one_content():
    """Vacuum pools store one content; Pool serves it as both, and refuses mixed parts."""
    d, _, _ = synthetic()
    one = {k: v for k, v in d.items() if not k.endswith("_zc")}
    pool = zm.Pool(one)
    assert pool.bg_R["zc"] is pool.bg_R["full"]
    D_full, _ = zm.all_partners(pool, "full", "self")
    D_zc, _ = zm.all_partners(pool, "zc", "self")
    assert np.array_equal(D_full, D_zc)
    with pytest.raises(ValueError, match="different contents"):
        zm.Pool.from_parts([d, one])


def test_per_z_noise_pieces():
    """pooled: each Z's flat level (n_i - <n>) / 2 pi; shape: none; poisson: S_one - S_full."""
    d, S, rho = synthetic()
    d["S_one"] = (S + 0.5).astype(np.float32)
    pool = zm.Pool(d)
    nz = zm.per_z_noise(pool)
    W = pool.w.sum()
    nbar = pool.bg_R["full"].sum(axis=0).sum(axis=(1, 2)) / W
    n_i = pool.S_full.astype(float).sum(axis=(2, 3))
    assert np.allclose(nz["pooled"][0].mean(axis=-1), (n_i - nbar) / (2 * np.pi))
    assert np.allclose(nz["shape"][0].mean(axis=-1), 0.0, atol=1e-10)
    dphi, dy = zm.observables(np.full(SZ, 0.5))
    assert np.allclose(nz["poisson"][0], dphi, atol=1e-6)
    assert np.allclose(nz["poisson"][1], dy, atol=1e-6)


def test_strict_noise_algebra():
    sd = {"pooled": np.array([3.0]), "shape": np.array([2.0]), "poisson": np.array([4.0])}
    out = zm.strict_noise(sd, n_samples=16, n_mix=10, efficiency=0.5)
    fluid2 = 9.0 - 16.0 / 16
    assert np.allclose(out["fluid"], np.sqrt(fluid2))
    assert np.allclose(out["full"], np.sqrt((fluid2 + 32.0) * 1.1))
    assert np.allclose(out["shape"], np.sqrt((4.0 - 1.0 + 32.0) * 1.1))
    assert np.allclose(out["poisson"], np.sqrt(32.0 * 1.1))


def test_rotate_backgrounds_is_a_symmetry_of_each_background():
    """One random multiple of 6 degrees per background, applied to its Z's, their tracks and
    its lab densities: every background seen from its own Z's does not change; the same
    seed gives the same angles; a vacuum pool keeps its one content."""
    d, _, _ = synthetic()
    pool, rot = zm.Pool(d), zm.Pool(d)
    dphi = rot.rotate_backgrounds(11)
    step = 2 * np.pi / zm.N_PHI
    assert np.allclose(dphi / step, np.round(dphi / step)) and len(dphi) == pool.n_bg
    assert np.allclose(np.mod(rot.phi - pool.phi - dphi[pool.zb] + np.pi, 2 * np.pi), np.pi)
    ev = np.repeat(dphi[pool.zb], pool.trk_n)
    assert np.allclose(np.mod(rot.trk_phi - pool.trk_phi - ev + np.pi, 2 * np.pi), np.pi,
                       atol=1e-5)
    assert np.array_equal(rot.trk_eta, pool.trk_eta) and np.array_equal(rot.S_full, pool.S_full)
    for c in zm.CONTENTS:
        for i in range(pool.n):
            b = pool.zb[i]
            assert np.allclose(zm.shift_sum(rot.bg_R[c][b], [rot.y[i]], [rot.phi[i]], [1.0]),
                               zm.shift_sum(pool.bg_R[c][b], [pool.y[i]], [pool.phi[i]], [1.0]))
    assert np.array_equal(zm.Pool(d).rotate_backgrounds(11), dphi)
    vac = zm.Pool({k: v for k, v in d.items() if not k.endswith("_zc")})
    vac.rotate_backgrounds(11)
    assert vac.bg_R["zc"] is vac.bg_R["full"]
