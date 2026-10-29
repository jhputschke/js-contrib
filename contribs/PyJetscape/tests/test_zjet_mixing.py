"""zjet_mixing: the all-partner kernel and pair rules against brute-force pair sums, the
jackknife replicates, and the explicit (CMS-style, strict) partner draws, on synthetic pools."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "python"))
sys.path.insert(0, str(HERE.parent / "example" / "analysis"))           # jetcorr
sys.path.insert(0, str(HERE.parent / "example" / "analysis" / "zjet"))

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


# ── the grid's edges in the pools ───────────────────────────────────────────────
def vacuum_part(name, n=3, seed=0, grid=True):
    """A version-4 pool part of a vacuum file with ``n`` events (each its own background)."""
    from jetcorr import mapio

    rng = np.random.default_rng(seed)
    S = rng.uniform(0, 2, (n, *SZ)).astype(np.float32)
    m = 4 * n
    rec = {"name": name, "n_events": n, "vacuum": True, "units": np.arange(n),
           "n_stored": np.zeros(0, np.int64),
           **{k: np.zeros(0) for k in mapio.WINDOW_KEYS if k != "n_stored"}}
    d = {"event": np.arange(n), "bg": np.arange(n), "w": np.ones(n),
         "y": rng.uniform(-2, 2, n), "phi": rng.uniform(-np.pi, np.pi, n),
         "k_one": np.zeros(n, np.int64), "n_samples": np.full(n, 10), "S_full": S, "S_one": S,
         "trk_n": np.full(n, 4), "trk_pt": np.zeros(m, np.int8),
         "trk_eta": rng.uniform(-2, 2, m).astype(np.float32),
         "trk_phi": rng.uniform(-np.pi, np.pi, m).astype(np.float32),
         "local_event": np.arange(n), "pthat_bin": np.full(n, -1), "n_coll": np.full(n, np.nan),
         "bg_unit": np.arange(n), "bgw_unit": np.arange(n), "bgw_k": np.full(n, -1),
         "bgw_n": np.ones(n, np.int64), "bgw_ncoll": np.full(n, np.nan),
         "bgw_S_full": S.astype(float), "bgw_R_full": rng.uniform(0, 1, (n, *SL)),
         **mapio.file_info_npz(rec), "pool_version": np.array(zm.POOL_VERSION)}
    if grid:
        d.update(zm.grid_arrays())
    return d


def test_unstored_grid_is_the_grid():
    """The pools written before the edges were stored were binned on today's GRID, and
    mix.py's cache key of their manifests is today's."""
    for k, e in zm.GRID.items():
        assert np.array_equal(zm.GRID_UNSTORED[k], e)
    assert zm.grid_tag(zm.GRID_UNSTORED) == zm.grid_tag()


def test_assemble_takes_parts_with_and_without_edges():
    a, b = vacuum_part("vac_a", seed=1), vacuum_part("vac_b", seed=2)
    ref = zm.assemble([a, b])
    old = {k: v for k, v in b.items() if not k.startswith("grid_")}
    mixed = zm.assemble([a, old])
    assert set(mixed) == set(ref)
    for k in ref:
        assert np.array_equal(mixed[k], ref[k], equal_nan=ref[k].dtype.kind == "f"), k
    pool = zm.Pool(mixed)
    assert pool.n == 6 and not any(k.startswith("grid_") for k in vars(pool))


@pytest.mark.parametrize("key, shift", [("dy", 0.1), ("pt", 0.5), ("dyf", 1e-9)])
def test_other_edges_are_refused(key, shift, tmp_path):
    a, b = vacuum_part("vac_a", seed=1), vacuum_part("vac_b", seed=2)
    b[f"grid_{key}"] = b[f"grid_{key}"] + shift
    with pytest.raises(ValueError, match=f"vac_b: binned on other edges.*{key}"):
        zm.assemble([a, b])
    for name, part in (("a", a), ("b", b)):
        np.savez(tmp_path / f"{name}.npz", **part)
    with pytest.raises(ValueError, match="other edges"):
        zm.Pool.load([tmp_path / "a.npz", tmp_path / "b.npz"])
    with pytest.raises(ValueError, match="other edges"):
        zm.Pool({**synthetic()[0], f"grid_{key}": b[f"grid_{key}"]})


def test_edges_to_rounding_are_accepted():
    """The C++ map step's linspace may differ from numpy's in the last bit."""
    b = vacuum_part("vac_b")
    b["grid_dphi"] = np.nextafter(b["grid_dphi"], np.inf)
    zm.check_grid([b])


def test_load_combines_parts_with_and_without_edges(tmp_path):
    a, b = vacuum_part("vac_a", seed=1), vacuum_part("vac_b", seed=2)
    np.savez(tmp_path / "a.npz", **a)
    np.savez(tmp_path / "b.npz", **{k: v for k, v in b.items() if not k.startswith("grid_")})
    pool = zm.Pool.load([tmp_path / "a.npz", tmp_path / "b.npz"])
    ref = zm.Pool(zm.assemble([a, b]))
    assert np.array_equal(pool.S_full, ref.S_full) and np.array_equal(pool.w, ref.w)


# ── hist: the pools' contents and the results ───────────────────────────────────
def test_to_hist_contents():
    hist = pytest.importorskip("hist")
    d, S, rho = synthetic()
    d["Y_full"] = np.random.default_rng(9).uniform(0, 1, (len(S), zm.N_PT, zm.N_SIDE, zm.N_DYF))
    pool = zm.Pool(d)
    w, W = pool.w, pool.w.sum()
    h = pool.to_hist("S_full")
    assert [a.name for a in h.axes] == ["pt", "dy", "dphi"]
    for a, k in zip(h.axes, ("pt", "dy", "dphi")):
        assert np.array_equal(a.edges, zm.GRID[k])
    assert np.allclose(h.values(), np.tensordot(w, S, axes=1) / W)
    assert np.all(np.isnan(h.variances()))
    assert np.allclose(h[{"pt": hist.loc(1.5)}].values(), h.values()[0])     # 1-2 GeV
    hb = pool.to_hist("S_zc", per="background")
    assert list(hb.axes["background"]) == list(pool.bg_id)
    assert np.allclose(hb.values(), pool.bg_SW["zc"] / pool.bg_W[:, None, None, None])
    ho = pool.to_hist("S_one")                                # from the per-event arrays
    assert np.allclose(ho.values(), np.tensordot(w, pool.S_one.astype(float), axes=1) / W)
    he = pool.to_hist("S_one", per="event", events=[2, 0])
    assert list(he.axes["event"]) == [pool.event[2], pool.event[0]]
    assert np.array_equal(he.values(), pool.S_one[[2, 0]].astype(float))
    hr = pool.to_hist("R_full", per="background")
    assert [a.name for a in hr.axes] == ["background", "pt", "eta", "phi"]
    assert np.allclose(hr.values(), pool.bg_R["full"] / pool.bg_W[:, None, None, None])
    hy = pool.to_hist("Y_full", per="background")
    assert [a.name for a in hy.axes] == ["background", "pt", "side", "dy"]
    yb = np.array([(w[pool.zb == b, None, None, None] * d["Y_full"][pool.zb == b]).sum(0)
                   / w[pool.zb == b].sum() for b in range(pool.n_bg)])
    assert np.allclose(hy.values(), yb)
    for args, kw in ((("R_full",), {"per": "event"}), (("S_zc",), {"per": "event"}),
                     (("X_full",), {}), (("L_full",), {}), (("S_one",), {"events": [0]})):
        with pytest.raises(ValueError):
            pool.to_hist(*args, **kw)


def test_observables_hist_jackknife():
    pytest.importorskip("hist")
    from cms2507 import rebin_y

    d, _, _ = synthetic()
    pool = zm.Pool(d)
    D, reps = zm.all_partners(pool, "zc", "all", False, pool.blocks(4))
    hs = zm.observables_hist(D, reps, k=2)
    dphi, dy = zm.observables(D)
    rp = np.array([zm.observables(r)[0] for r in reps])
    ry = np.array([zm.observables(r)[1] for r in reps])
    ry = rebin_y(ry, np.zeros_like(ry), 2)[0]
    assert np.allclose(hs["phi"].values(), dphi)
    assert np.allclose(hs["phi"].variances(), zm.jackknife(rp) ** 2)
    assert np.array_equal(hs["y"].axes["dy"].edges, zm.DETA[2:23:2])           # |Δy| < 4, 0.8
    assert np.allclose(hs["y"].axes["dy"].edges, np.linspace(-4, 4, 11))
    assert np.allclose(hs["y"].values(), rebin_y(dy, np.zeros_like(dy), 2)[0])
    assert np.allclose(hs["y"].variances(), zm.jackknife(ry) ** 2)
    one = zm.observables_hist(D, reps[:1])                 # too few replicates: no error
    assert one["y"].axes["dy"].size == zm.N_DY and np.all(np.isnan(one["y"].variances()))


def test_obs_hist_shapes_and_independent_difference():
    pytest.importorskip("hist")
    rng = np.random.default_rng(5)
    a, b = rng.normal(size=(zm.N_PT, zm.N_DPHI)), rng.normal(size=(zm.N_PT, zm.N_DPHI))
    ha, hb = zm.obs_hist("phi", a, np.full_like(a, 0.3)), zm.obs_hist("phi", b, 0.4)
    diff = ha + hb * -1                    # Pb+Pb - pp, independent (Weight storage: no -)
    assert np.allclose(diff.values(), a - b) and np.allclose(diff.variances(), 0.25)
    with pytest.raises(ValueError, match="shape"):
        zm.obs_hist("y", np.zeros((zm.N_PT, zm.N_DY)), k=2)
    with pytest.raises(ValueError):
        zm.obs_hist("eta", a)


def test_zframe_hist_projections_are_the_observables():
    """Summed over dy, or over the Z side's dphi bins, then +- merged: observables()."""
    hist = pytest.importorskip("hist")
    D = np.random.default_rng(6).normal(size=SZ)
    h = zm.zframe_hist(D, np.full(SZ, 0.1))
    assert [a.name for a in h.axes] == ["pt", "dy", "dphi"]
    assert np.allclose(h.variances(), 0.01)
    p = h[{"dy": sum}].values() / np.diff(zm.DPHI)
    y = h[{"dphi": slice(0, zm.N_DPHI // 2, sum)}].values() / np.diff(zm.DETA)
    dphi, dy = zm.observables(D)
    assert np.allclose(0.5 * (p + p[:, zm.MIRROR]), dphi)
    assert np.allclose(0.5 * (y + y[:, ::-1]), dy)
    assert h[{"dphi": hist.loc(np.pi)}].shape == (zm.N_PT, zm.N_DY)


@pytest.mark.parametrize("n_mix", [1, 2])
def test_explicit_replicates_leave_blocks_out_of_s_and_b(n_mix):
    """Each replicate, by brute force: S the mean over the Z's off the block; B the mean over
    the (Z, partner) pairs with neither on the block, each weighing w_i / m_i.  The mean is
    the weighted mean of explicit_partners' per-Z observables."""
    d, _, _ = synthetic(n_bg=4, per_bg=(3, 2, 4, 3), seed=4)
    pool = zm.Pool(d)
    z = np.arange(pool.n)
    res = zm.explicit_partners(pool, z, z, pool.w, n_mix=n_mix, seed=3)
    S = pool.S_full.astype(float)
    blocks = pool.blocks(4)                                  # one background each
    mean, reps = zm.explicit_replicates(pool, z, pool.w, res, S, blocks)
    for got, per_z in zip(zm.observables(mean), (res["dphi"], res["dy"])):
        assert np.allclose(got, zm.weighted_mean(per_z, pool.w)[0])
    res_maps = zm.explicit_partners(pool, z, z, pool.w, n_mix=n_mix, seed=3, maps=True)
    assert res_maps["partners"] == res["partners"]
    _, reps_maps = zm.explicit_replicates(pool, z, pool.w, res_maps, S, blocks)
    assert np.allclose(reps_maps, reps)
    for blk, rep in zip(blocks, reps):
        sn, sd, bn, bd = np.zeros(SZ), 0.0, np.zeros(SZ), 0.0
        for i in z:
            if pool.zb[i] in blk:
                continue
            sn, sd = sn + pool.w[i] * S[i], sd + pool.w[i]
            js = res["partners"][i]
            for j in js:
                if pool.zb[j] not in blk:
                    pw = pool.w[i] / len(js)
                    bn = bn + pw * zm.z_hist(*pool.tracks(j), pool.y[i], pool.phi[i])
                    bd += pw
        assert np.allclose(rep, sn / sd - bn / bd)
