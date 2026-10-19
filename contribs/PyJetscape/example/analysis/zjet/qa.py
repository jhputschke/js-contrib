#!/usr/bin/env python3
"""QA of a Z+jet production: the Z's, their leptons and weights, the hadrons of every source and
the total charged multiplicity.  Step 0 of ../ANA_Zhadron.md and ../ANA_Zjet.md.

    python qa.py map "<dir or glob of *_hadrons_jet_frag.h5>" <out dir> [--jobs 4] [--files i:j]
    python qa.py plot --pbpb <map out> [--pp <map out>] [--outdir figures] [--tag _c00]

The map step reads every production file once (Python on the HDF5 files; the same in C++ on
the ROOT files: ../../analysis_root/zjet/zqa) and writes ``<stem>_qa.npz`` (C++:
``<stem>_qa.root``) with

* a per-event table: the event, its file and hydro background, p̂T window, N_coll, the
  weight w_s = sigma_k / N_k (window cross section over the window's events in the whole
  directory), the Z and its leptons, and the total **charged multiplicity** at |eta| < 0.5 and
  < 2.4 (all p_T): of the full event (bulk_jet + jet_frag; mean and variance over the
  oversamples), of the background (bulk_bg) and of the fragments, and how many of the Z's
  leptons were found among the fragments (2 expected; they are removed from every hadron
  histogram, as in the analyses);
* per-source hadron histograms (bulk_jet, bulk_bg, jet_frag; vacuum: jet_frag), p_T, eta, phi
  and the species, charged and all particles, per event (the mean over its oversamples)
  summed unweighted (``n``) and with w_s x N_coll (``sc``).

The plot step sums the files of a directory (either format) and draws figures/qa_*{tag}.png
and qa_summary{tag}.txt: the Z (p_T, y, phi, mass, the leptons' p_T and eta by flavour) for all
events, CMS 2507.09307's and CMS 1702.01060's selections, unweighted, sigma-weighted and
sigma x N_coll-weighted; p_T^Z per p̂T window, N_coll, the weights and N_eff; the hadrons per
source, the wake (bulk_jet - bulk_bg) and the full event; the charged multiplicity (its
distribution, dN_ch/deta at |eta| < 0.5 against N_coll, the oversamples' spread against
Poisson); the species; pp next to Pb+Pb."""
import argparse
import glob
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                        # jetcorr
try:
    import jetscape  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "python"))
import zjet_wake as zw  # noqa: E402
from jetcorr import runs  # noqa: E402

SOURCES = ("bulk_jet", "bulk_bg", "jet_frag")
VARS = ("pt", "eta", "phi")
CHARGES = ("charged", "all")
WEIGHTS = ("n", "sc")


def linspace(a, b, n):
    return np.linspace(a, b, n)


#: the bins (the same arrays in ZJetCommon.h's qa section)
EDGES = {
    "pt": np.concatenate([linspace(0, 2, 21)[:-1], linspace(2, 5, 13)[:-1],
                          linspace(5, 10, 11)[:-1], linspace(10, 30, 11)[:-1],
                          np.array([30.0, 40.0, 60.0, 100.0])]),
    "eta": linspace(-2.6, 2.6, 53),
    "phi": linspace(-math.pi, math.pi, 37),
}
#: |pid| of the species histogram (the last bin: everything else)
SPECIES = (211, 321, 2212, 3222, 3112, 3312, 3334, 11, 13, 22, 111, 130, 310, 311, 2112, 3122,
           3212, 3322, 221)
SPECIES_NAMES = ("π±", "K±", "p", "Σ+", "Σ−", "Ξ−", "Ω−", "e", "μ", "γ", "π0", "K0L", "K0S", "K0",
                 "n", "Λ", "Σ0", "Ξ0", "η", "other")
EVENT_COLUMNS = ("event", "file", "bg", "pthat_bin", "n_coll", "w_s", "n_samples",
                 "z_pt", "z_y", "z_phi", "z_m", "lep1_pid", "lep1_pt", "lep1_eta", "lep1_phi",
                 "lep2_pid", "lep2_pt", "lep2_eta", "lep2_phi",
                 "nch05_full", "nch24_full", "nch24_full_var", "nch05_bg", "nch24_bg",
                 "nch05_frag", "nch24_frag", "nzlep", "nzlep_min")


def _index(v, edges):
    """numpy's searchsorted(side='right') - 1, the last edge inclusive, -1 outside (as
    zjet_mixing._index and ZJetCommon.h's bin_index)."""
    v = np.asarray(v, float)
    j = np.searchsorted(edges, v, side="right") - 1
    j[v == edges[-1]] = len(edges) - 2
    j[(j < 0) | (j > len(edges) - 2)] = -1
    return j


def species_index(pid):
    a = np.abs(np.asarray(pid))
    out = np.full(len(a), len(SPECIES), np.int64)
    for i, s in enumerate(SPECIES):
        out[a == s] = i
    return out


def source_hists(ev, keep, n_samples):
    """Per-event mean histograms of an EventHadrons: {(var, charge): counts / n_samples},
    'species': counts / n_samples (all particles)."""
    out = {}
    for ch in CHARGES:
        m = keep & (ev.charged if ch == "charged" else np.ones(len(keep), bool))
        for var in VARS:
            vals = getattr(ev, var)[m]
            j = _index(vals, EDGES[var])
            ok = j >= 0
            out[(var, ch)] = np.bincount(j[ok], minlength=len(EDGES[var]) - 1) / n_samples
    out["species"] = np.bincount(species_index(ev.pid[keep]),
                                 minlength=len(SPECIES) + 1) / n_samples
    return out


def nch_per_sample(ev, keep, n_samples, eta_max):
    m = keep & ev.charged & (np.abs(ev.eta) < eta_max)
    return np.bincount(ev.sample[m], minlength=n_samples).astype(float)


def qa_file(stem, file_index, local_events, global_events, w_s, n_coll, bgid, out):
    """The map step for one production file -> ``out`` (.npz)."""
    from jetscape.hadrons_h5 import EventHadrons, HadronFileReader, Hadrons
    t0 = time.time()
    r = HadronFileReader([stem])
    bulk = "bulk_jet" in r.tags(0)
    h = {t: Hadrons.from_h5(f"{stem}_hadrons_{t}.h5")
         for t in (SOURCES if bulk else ("jet_frag",))}
    hist = {f"h_{s}_{v}_{c}_{w}": np.zeros(len(EDGES[v]) - 1)
            for s in (SOURCES if bulk else ("jet_frag",)) for v in VARS for c in CHARGES
            for w in WEIGHTS}
    hist.update({f"s_{s}_{w}": np.zeros(len(SPECIES) + 1)
                 for s in (SOURCES if bulk else ("jet_frag",)) for w in WEIGHTS})
    cols = {k: [] for k in EVENT_COLUMNS}
    bg_cache = {}
    for e, g, ws, nc, b in zip(local_events, global_events, w_s, n_coll, bgid):
        info = r.event_info(int(e))
        z = info.z
        F = EventHadrons(h["jet_frag"], h["jet_frag"].unit_index(int(e)))
        nf = F.n_samples
        if nf == 0:
            continue
        keep_f = zw.not_z_lepton(F, info)
        zlep = np.bincount(F.sample[~keep_f], minlength=nf)
        per = {"jet_frag": (F, keep_f, nf)}
        if bulk:
            J = EventHadrons(h["bulk_jet"], h["bulk_jet"].unit_index(int(e)))
            K = J.n_samples
            if K == 0:
                continue
            per["bulk_jet"] = (J, np.ones(len(J), bool), K)
            if info.bg_unit not in bg_cache:
                B = EventHadrons(h["bulk_bg"], h["bulk_bg"].unit_index(info.bg_unit))
                kb = np.ones(len(B), bool)
                bg_cache[info.bg_unit] = (source_hists(B, kb, B.n_samples),
                                          nch_per_sample(B, kb, B.n_samples, 0.5).mean(),
                                          nch_per_sample(B, kb, B.n_samples, 2.4).mean())
        w_sc = ws * (nc if nc >= 0 else 1.0)
        for src, (ev, keep, n) in per.items():
            sh = source_hists(ev, keep, n)
            for (v, c), x in ((k, x) for k, x in sh.items() if k != "species"):
                hist[f"h_{src}_{v}_{c}_n"] += x
                hist[f"h_{src}_{v}_{c}_sc"] += w_sc * x
            hist[f"s_{src}_n"] += sh["species"]
            hist[f"s_{src}_sc"] += w_sc * sh["species"]
        nf05, nf24 = (nch_per_sample(F, keep_f, nf, eta) for eta in (0.5, 2.4))
        if bulk:
            bh, b05, b24 = bg_cache[info.bg_unit]
            for (v, c), x in ((k, x) for k, x in bh.items() if k != "species"):
                hist[f"h_bulk_bg_{v}_{c}_n"] += x
                hist[f"h_bulk_bg_{v}_{c}_sc"] += w_sc * x
            hist["s_bulk_bg_n"] += bh["species"]
            hist["s_bulk_bg_sc"] += w_sc * bh["species"]
            J, _, K = per["bulk_jet"]
            kj = np.ones(len(J), bool)
            full05 = nch_per_sample(J, kj, K, 0.5) + nf05[np.arange(K) % nf]
            full24 = nch_per_sample(J, kj, K, 2.4) + nf24[np.arange(K) % nf]
            ns = K
        else:
            b05 = b24 = 0.0
            full05, full24, ns = nf05, nf24, nf
        row = {"event": g, "file": file_index, "bg": b,
               "pthat_bin": -1 if info.pthat_bin is None else info.pthat_bin,
               "n_coll": nc, "w_s": ws, "n_samples": ns,
               **{k: z[k] for k in EVENT_COLUMNS[7:19]},
               "nch05_full": full05.mean(), "nch24_full": full24.mean(),
               "nch24_full_var": full24.var(ddof=1) if len(full24) > 1 else 0.0,
               "nch05_bg": b05, "nch24_bg": b24,
               "nch05_frag": nf05.mean(), "nch24_frag": nf24.mean(),
               "nzlep": zlep.mean(), "nzlep_min": zlep.min()}
        for k in EVENT_COLUMNS:
            cols[k].append(row[k])
    d = {f"ev_{k}": np.asarray(v, np.int64 if k in ("event", "file", "bg", "pthat_bin",
                                                     "n_coll", "n_samples", "lep1_pid",
                                                     "lep2_pid") else float)
         for k, v in cols.items()}
    d.update(hist)
    np.savez_compressed(out, **d)
    return os.path.basename(stem), len(cols["event"]), time.time() - t0


def cmd_map(a):
    from jetscape.hadrons_h5 import HadronFileReader
    r = HadronFileReader(a.source)
    n = r.n_events
    infos = [r.event_info(g) for g in range(n)]
    bulk = "bulk_jet" in r.tags()
    pbin = np.array([-1 if i.pthat_bin is None else i.pthat_bin for i in infos])
    if r.pthat_bins is not None:
        sig = np.array([r.pthat_bin_sigma(k)[0] for k in range(len(r.pthat_bins))])
        nk = np.bincount(pbin[pbin >= 0], minlength=len(sig))
        k = np.clip(pbin, 0, None)
        w_s = np.where(pbin >= 0, sig[k] / np.maximum(nk[k], 1), np.nan)
    else:
        w_s = np.ones(n)
    n_coll = np.array([-1 if i.n_coll is None else i.n_coll for i in infos])
    loc = np.array([(i.file_index, i.bg_unit if bulk else i.local_event) for i in infos])
    _, bgid = np.unique(loc, axis=0, return_inverse=True)
    bgid = bgid.ravel()
    has_z = np.array([bool(i.z) and math.isfinite(i.z["z_pt"]) for i in infos])
    lo, hi = 0, r.n_files
    if a.files:
        i, _, j = a.files.partition(":")
        lo, hi = max(0, int(i)), min(hi, int(j) if j else int(i) + 1)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = []
        for fi in range(lo, hi):
            stem = r.stems[fi]
            g = np.flatnonzero(has_z & (loc[:, 0] == fi))
            local = g - int(r._offsets[fi])
            futs.append(ex.submit(qa_file, stem, fi, local, g, w_s[g], n_coll[g], bgid[g],
                                  os.path.join(a.out, f"{os.path.basename(stem)}_qa.npz")))
        for f in futs:
            name, ne, dt = f.result()
            print(f"  {name}: {ne} events, {dt:.0f} s", flush=True)
    print(f"qa map: {hi - lo} files, {time.time() - t0:.0f} s")


# ── the plot step ────────────────────────────────────────────────────────────────
def load(directory):
    """Sum the map step's files of a directory (.npz or C++'s .root) -> (events dict,
    histograms dict)."""
    paths = sorted(glob.glob(os.path.join(directory, "*_qa.npz")))
    roots = sorted(glob.glob(os.path.join(directory, "*_qa.root")))
    parts = []
    for p in paths:
        with np.load(p) as f:
            parts.append({k: f[k] for k in f.files})
    if roots:
        import awkward as ak
        import uproot
        for p in roots:
            with uproot.open(p) as f:
                ev = f["qa_events"].arrays()
                hs = f["qa_hist"].arrays()
                d = {f"ev_{k}": ak.to_numpy(ev[k]) for k in ev.fields}
                d.update({k: ak.to_numpy(hs[k])[0] for k in hs.fields})
                parts.append(d)
    if not parts:
        raise FileNotFoundError(f"no *_qa.npz or *_qa.root in {directory}")
    E = {k[3:]: np.concatenate([p[k] for p in parts]) for k in parts[0] if k.startswith("ev_")}
    H = {k: sum(p[k] for p in parts) for k in parts[0] if not k.startswith("ev_")}
    return E, H


def z_selections(E):
    import zjet_jets as zj
    t = {k: E[k] for k in ("z_pt", "z_y", "z_m", "lep1_pt", "lep1_eta", "lep2_pt", "lep2_eta",
                           "lep1_pid")}
    return {"all Z's": np.isfinite(E["z_pt"]), "CMS 2507": zw.select(t, **zw.CMS_2507),
            "CMS 1702": zj.z_selection(E)}


def event_weights_qa(E):
    ws = np.nan_to_num(E["w_s"], nan=0.0)
    nc = np.where(E["n_coll"] >= 0, E["n_coll"], 1)
    return {"unweighted": np.ones(len(ws)), "σ": ws, "σ × N_coll": ws * nc}


def n_eff(w):
    return float(w.sum() ** 2 / np.sum(w * w)) if len(w) and np.sum(w * w) > 0 else 0.0


def cmd_plot(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from jetcorr import figdata

    os.makedirs(a.outdir, exist_ok=True)
    samples = [("Pb+Pb", a.pbpb)] + ([("pp", a.pp)] if a.pp else [])
    data = {name: load(d) for name, d in samples}
    lines = []
    zvars = (("z_pt", np.linspace(0, 250, 51), "p_T^Z [GeV]"),
             ("z_y", np.linspace(-3, 3, 31), "y_Z"),
             ("z_phi", np.linspace(-np.pi, np.pi, 33), "φ_Z"),
             ("z_m", np.linspace(60, 120, 31), "m_ℓℓ [GeV]"))
    for name, (E, H) in data.items():
        sels, wts = z_selections(E), event_weights_qa(E)
        lines.append(f"{name}: {len(E['event'])} events with a Z")
        for sn, m in sels.items():
            lines.append(f"  {sn:9s}: {int(m.sum()):6d} Z's, N_eff (σ) {n_eff(wts['σ'][m]):8.0f}, "
                         f"N_eff (σ × N_coll) {n_eff(wts['σ × N_coll'][m]):8.0f}")
        # the Z and its leptons
        fig, ax = figdata.subplots(6, 3, figsize=(15, 22))
        for col, (wn, w) in enumerate(wts.items()):
            for row, (var, edges, lab) in enumerate(zvars):
                for sn, m in sels.items():
                    ax[row, col].hist(E[var][m], edges, weights=w[m], histtype="step", lw=1.4,
                                      density=col > 0, label=sn)
                ax[row, col].set(xlabel=lab, ylabel="Z's" if col == 0 else "normalized",
                                 title=f"{lab.split(' ')[0]}, {wn}")
            for row, var, edges, lab in ((4, "pt", np.linspace(0, 200, 41), "lepton p_T [GeV]"),
                                         (5, "eta", np.linspace(-3, 3, 31), "lepton η")):
                for sn, m in sels.items():
                    for fl, nm in ((11, "e"), (13, "μ")):
                        mf = m & (np.abs(E["lep1_pid"]) == fl)
                        vals = np.concatenate([E[f"lep{i}_{var}"][mf] for i in (1, 2)])
                        ax[row, col].hist(vals, edges, weights=np.concatenate([w[mf], w[mf]]),
                                          histtype="step", lw=1.2, density=col > 0,
                                          ls="-" if fl == 13 else "--", label=f"{nm}, {sn}")
                ax[row, col].set(xlabel=lab, title=f"{lab.split(' [')[0]}, {wn}")
        for x in ax.ravel():
            x.legend(fontsize=7)
        ax[0, 0].set_yscale("log")
        fig.suptitle(f"QA, {name}: the Z and its leptons (columns: unweighted, σ_k/N_k, "
                     f"σ_k/N_k × N_coll)", y=0.995)
        fig.tight_layout()
        figdata.savefig(fig, os.path.join(a.outdir, f"qa_z_{name.replace('+', '')}{a.tag}.png"),
                        dpi=90)
        plt.close(fig)
        # windows, N_coll, weights
        fig, ax = figdata.subplots(1, 4, figsize=(20, 4))
        for k in np.unique(E["pthat_bin"][E["pthat_bin"] >= 0]):
            m = E["pthat_bin"] == k
            ax[0].hist(E["z_pt"][m], zvars[0][1], histtype="step", label=f"window {k}")
        ax[0].set(xlabel="p_T^Z [GeV]", ylabel="Z's", title="p_T^Z per p̂T window", yscale="log")
        ax[0].legend(fontsize=7)
        ax[1].hist(E["z_pt"], zvars[0][1], weights=wts["σ"], color="0.6")
        ax[1].set(xlabel="p_T^Z [GeV]", title="σ-weighted (all windows)", yscale="log")
        if np.any(E["n_coll"] >= 0):
            _, first = np.unique(E["bg"], return_index=True)
            ax[2].hist(E["n_coll"][first], 30, color="0.6")
            ax[2].set(xlabel="N_coll", ylabel="hydro backgrounds", title="N_coll per background")
        w = wts["σ × N_coll"][sels["CMS 2507"]]
        if len(w) and w.max() > 0:
            ax[3].hist(w / w.mean(), np.logspace(-3, 2, 51), color="0.6")
            ax[3].set(xscale="log", xlabel="w / ⟨w⟩", title=f"σ × N_coll weights, CMS 2507 "
                      f"(N_eff {n_eff(w):.0f} of {len(w)})")
        fig.tight_layout()
        tagged = f"qa_weights_{name.replace('+', '')}{a.tag}.png"
        figdata.savefig(fig, os.path.join(a.outdir, tagged), dpi=90)
        plt.close(fig)
        # the Z leptons among the fragments
        bad = int(np.sum(E["nzlep_min"] != 2))
        lines.append(f"  Z leptons found among the fragments: mean {E['nzlep'].mean():.3f} per "
                     f"fragmentation; events with a fragmentation of != 2: {bad}")
    # the hadrons: per charge a bulk-scale and a jet-scale column
    fig, ax = figdata.subplots(3, 4, figsize=(24, 13))
    for col0, ch in ((0, "charged"), (2, "all")):
        for row, var in enumerate(VARS):
            edges = EDGES[var]
            width = np.diff(edges)
            for name, (E, H) in data.items():
                wsum = event_weights_qa(E)["σ × N_coll"].sum()
                get = lambda s: H.get(f"h_{s}_{var}_{ch}_sc")
                ls = "--" if name == "pp" else "-"
                big, small = [], []
                if name == "Pb+Pb" and get("bulk_jet") is not None:
                    j, b, f_ = get("bulk_jet"), get("bulk_bg"), get("jet_frag")
                    big = [("background (bulk_bg)", b), ("background + wake (bulk_jet)", j),
                           ("full event (bulk_jet + fragments)", j + f_)]
                    small = [("fragments", f_), ("wake (bulk_jet − bulk_bg)", j - b)]
                elif get("jet_frag") is not None:
                    small = [("fragments", get("jet_frag"))]
                for c, curves in ((col0, big), (col0 + 1, small)):
                    for lab, x in curves:
                        ax[row, c].stairs(x / wsum / width, edges, lw=1.4, ls=ls,
                                          label=f"{name}: {lab}")
            unit = {"pt": "p_T [GeV]", "eta": "η", "phi": "φ"}[var]
            for c, scale in ((col0, "the bulk"), (col0 + 1, "the jet: fragments, wake, pp")):
                ax[row, c].set(xlabel=unit, ylabel=f"dN/d{var} per Z event",
                               title=f"{ch}, {var}: {scale}")
                ax[row, c].axhline(0, color="0.6", lw=0.5)
            if var == "pt":
                ax[row, col0].set(xscale="log", yscale="log")
                ax[row, col0 + 1].set_xscale("log")
                ax[row, col0 + 1].set_yscale("symlog", linthresh=1e-2)
    for x in ax.ravel():
        x.legend(fontsize=6)
    fig.suptitle("QA: hadrons per source, per Z event (σ × N_coll-weighted; the Z's leptons "
                 "removed)", y=0.995)
    fig.tight_layout()
    figdata.savefig(fig, os.path.join(a.outdir, f"qa_hadrons{a.tag}.png"), dpi=90)
    plt.close(fig)
    # the total charged multiplicity
    fig, ax = figdata.subplots(1, 5, figsize=(26, 4.3))
    for name, (E, H) in data.items():
        w = event_weights_qa(E)
        panel = ax[0] if name == "Pb+Pb" else ax[1]
        lo, hi = np.percentile(E["nch24_full"], [0.05, 99.95])
        span = max(hi - lo, 10)
        for wn in ("unweighted", "σ × N_coll"):
            panel.hist(E["nch24_full"], np.linspace(max(lo - 0.1 * span, 0), hi + 0.1 * span, 60),
                       weights=w[wn], density=True, histtype="step", lw=1.4,
                       ls="--" if wn == "unweighted" else "-", label=f"{name}, {wn}")
        panel.set(xlabel="N_ch (|η| < 2.4, all p_T), full event", ylabel="normalized",
                  title=f"{name}: total charged multiplicity")
        mean24 = np.average(E["nch24_full"], weights=w["σ × N_coll"])
        mean05 = np.average(E["nch05_full"], weights=w["σ × N_coll"])
        lines.append(f"{name}: total charged multiplicity of the full event (σ × N_coll-weighted): "
                     f"<N_ch(|η| < 2.4)> {mean24:.1f}, dN_ch/dη(|η| < 0.5) {mean05:.1f}; "
                     f"background {np.average(E['nch05_bg'], weights=w['σ × N_coll']):.1f}, "
                     f"fragments {np.average(E['nch05_frag'], weights=w['σ × N_coll']):.2f}")
        if name == "Pb+Pb" and np.any(E["n_coll"] >= 0):
            ax[2].plot(E["n_coll"], E["nch05_full"], ".", ms=3, alpha=0.5, label="full event")
            ax[2].plot(E["n_coll"], E["nch05_bg"], ".", ms=3, alpha=0.5, label="background")
            ax[2].set(xlabel="N_coll", ylabel="dN_ch/dη (|η| < 0.5)",
                      title="Pb+Pb: charged multiplicity against N_coll")
            ratio = E["nch24_full_var"] / np.maximum(E["nch24_full"], 1e-9)
            ax[3].hist(ratio, np.linspace(0, max(3, np.percentile(ratio, 99.5)), 50), color="0.6")
            ax[3].axvline(1, color="k", lw=0.8)
            ax[3].set(xlabel="var / mean of N_ch(|η| < 2.4) over the oversamples",
                      title="Pb+Pb: the oversamples' spread (1: Poisson)")
            lines.append(f"Pb+Pb: the oversamples' var / mean of N_ch(|η| < 2.4): median "
                         f"{np.median(ratio):.3f}")
        top = max(np.percentile(E["nch24_frag"], 99.9) * 1.1, 5)
        ax[4].hist(E["nch24_frag"], np.linspace(0, top, 50),
                   weights=w["σ × N_coll"], density=True, histtype="step", lw=1.4, label=name)
    ax[4].set(xlabel="N_ch (|η| < 2.4) of the fragments",
              title="the jet's (fragments') charged multiplicity")
    for x in ax:
        if x.get_legend_handles_labels()[0]:
            x.legend(fontsize=7)
    fig.tight_layout()
    figdata.savefig(fig, os.path.join(a.outdir, f"qa_multiplicity{a.tag}.png"), dpi=90)
    plt.close(fig)
    # the species
    fig, ax = figdata.subplots(figsize=(13, 4.5))
    xs = np.arange(len(SPECIES_NAMES))
    bars = [(f"{n}: {s}", H[f"s_{s}_sc"]) for n, (E, H) in data.items()
            for s in (SOURCES if n == "Pb+Pb" else ("jet_frag",)) if f"s_{s}_sc" in H]
    for i, (lab, x) in enumerate(bars):
        ax.bar(xs + (i - len(bars) / 2) * 0.8 / len(bars), x / max(x.sum(), 1e-300),
               0.8 / len(bars), label=lab)
        lines.append(f"species {lab}: " + ", ".join(f"{n} {v:.3f}" for n, v in
                                                   zip(SPECIES_NAMES, x / max(x.sum(), 1e-300))))
    ax.set_xticks(xs, SPECIES_NAMES)
    ax.set(yscale="log", ylabel="fraction of the particles", title="QA: species per source")
    ax.legend(fontsize=7)
    fig.tight_layout()
    figdata.savefig(fig, os.path.join(a.outdir, f"qa_species{a.tag}.png"), dpi=90)
    plt.close(fig)
    text = "\n".join(lines) + "\n"
    with open(os.path.join(a.outdir, f"qa_summary{a.tag}.txt"), "w") as f:
        f.write(text)
    print(text, end="")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("map", help="the map step on the HDF5 files")
    m.add_argument("source", help="a directory or a quoted glob of hadron files")
    m.add_argument("out")
    m.add_argument("--jobs", type=int, default=2)
    m.add_argument("--files", default=None, help="i:j, only production files i .. j-1")
    p = sub.add_parser("plot", help="the figures and the summary")
    p.add_argument("--pbpb", required=True, help="the map step's Pb+Pb outputs (.npz or .root)")
    p.add_argument("--pp", default=None, help="the same for the vacuum reference")
    p.add_argument("--run", default="zjet_setup", help="runs/<run>/figures is the default --outdir (jetcorr.runs)")
    p.add_argument("--outdir", default=None)
    p.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.cmd == "plot":
        a.outdir = a.outdir or runs.figures(a.run)
    (cmd_map if a.cmd == "map" else cmd_plot)(a)


if __name__ == "__main__":
    main()
