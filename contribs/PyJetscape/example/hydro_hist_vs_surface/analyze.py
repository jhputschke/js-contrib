#!/usr/bin/env python3
"""
example/hydro_hist_vs_surface/analyze.py

Summarize the reference and every variant (make_surfaces.py + hadronize.py) into one small
``summary.npz``: per event and leg, the hadron observables and the surface properties that
hydro_hist_vs_surface.ipynb compares.  Reading the hadron files takes minutes; the notebook
then only loads this file.

    python analyze.py --ref DATA --variants-dir DATA/hist_vs_surface \\
                      --variants ref ref_ideal ref_no_bulk ref_no_shear hist \\
                      --out DATA/hist_vs_surface/summary.npz

``ref`` is DATA itself (MUSIC's surfaces, full viscous corrections); every other name is a
directory under --variants-dir.  All variants must hold the same production files (stems)
and events.

What is stored (keys ``had/<variant>/<leg>/<name>``, leg ``jet`` = bulk_jet, ``bg`` =
bulk_bg of the event's background; first axis = global event, every value is the mean over
that event's oversamples):

    dndeta      charged dN/deta over ETA_EDGES
    dndy        dN/dy at |y| < 0.5 for SPECIES_IDS (pi+, pi-, K+, K-, p, pbar)
    sumpt       sum of pT at |y| < 0.5 for SPECIES_IDS (<pT> = sumpt / dndy)
    ptspec      dN/dpT at |y| < 0.5, (species pi, K, p) x PT_EDGES
    ptspec_ch   charged dN/dpT at |eta| < 1 over PTCH_EDGES
    Q           charged |eta| < 1, sum of exp(i n phi) for n = 1..NMAX, per PTV_EDGES bin
                and (last column) integrated over 0.2 < pT < 3 GeV; complex
    NQ          the number of hadrons in each of those bins
    QA, NQA     Q and NQ (integrated bin only, n = 1..NMAX) of the first half of the
                oversamples, as sums (not per-sample means): with Q they give each event's
                sampling noise of v_n{2} (half A against half B = all - A)
    E_eta1      energy of all hadrons at |eta| < 1 [GeV]
    N_all, E_all  all hadrons, all rapidities: number and energy
    dphi_jet    soft charged (pT < 4 GeV, |eta - y_jet| < 1): counts in DPHI_EDGES of
                phi - phi_jet (jet axis = the hardest shower initiator)
    dphi_jet_pt the same, pT-weighted
    pt_near     soft charged with |dphi| < 1 and |eta - y_jet| < 1: dN/dpT over PTCH_EDGES
    n_samples   oversamples of the unit

and ``surf/<variant>/<leg>/<name>`` for the variants with their own surface geometry (ref
and hist; the ref_* variants share ref's):

    n_cells, V (= sum of d^3sigma_mu u^mu), V_neg (its negative part),
    dVdtau over TAU_EDGES, dVdeta over ETAS_EDGES (each cell spread over its eta width),
    dVdur: V in bins of the transverse four-velocity u_T = sqrt(u_x^2 + u_y^2) over
    UR_EDGES (|eta_s| < 1), mean_ur (V-weighted, |eta_s| < 1),
    Pi_mean, pipi_mean: V-weighted means of Pi and sqrt(pi:pi) [GeV/fm^3] (|eta_s| < 1)

plus ``jet/phi``, ``jet/y`` (the jet axis per event), the bin edges, and ``meta`` (JSON:
stems, settings per variant, hadronization settings found in the hadron files).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PYJETSCAPE = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(PYJETSCAPE, "python"))

ETA_EDGES = np.linspace(-6, 6, 49)
SPECIES_IDS = (211, -211, 321, -321, 2212, -2212)
SPECIES_GROUPS = ((211, -211), (321, -321), (2212, -2212))     # ptspec rows: pi, K, p
PT_EDGES = np.linspace(0, 3, 31)
PTCH_EDGES = np.linspace(0, 4, 41)
PTV_EDGES = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0])
NMAX = 4
DPHI_EDGES = np.linspace(-np.pi, np.pi, 25)
TAU_EDGES = np.linspace(0, 16, 33)       # 0.5 fm/c: whole multiples of both surfaces' 0.1 fm/c steps
ETAS_EDGES = np.linspace(-6, 6, 25)
UR_EDGES = np.linspace(0, 1.5, 31)

#: the hadron-file attributes that must agree between variants for a fair comparison
MATCH_ATTRS = ("base_seed", "seed_scheme", "correlated_sampling", "common_seeds", "n_samples")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref", required=True, help="the production directory (reference)")
    p.add_argument("--variants-dir", required=True, dest="variants_dir")
    p.add_argument("--variants", nargs="+",
                   default=["ref", "ref_ideal", "ref_no_bulk", "ref_no_shear", "hist"])
    p.add_argument("--out", required=True)
    p.add_argument("--events", default=None,
                   help="event range a:b of every production file (python slice; default all)")
    p.add_argument("--no-surface", action="store_false", dest="surface",
                   help="skip the surface summaries (the slow part: ~4 GB per file)")
    return p.parse_args(argv)


def variant_dir(a, v):
    return os.path.abspath(a.ref if v == "ref" else os.path.join(a.variants_dir, v))


def stems_of(d):
    import glob
    return sorted(os.path.basename(p)[:-len("_particlize.h5")]
                  for p in glob.glob(os.path.join(d, "*_particlize.h5")))


def event_range(spec, n):
    if not spec:
        return list(range(n))
    parts = [int(x) if x else None for x in spec.split(":")]
    return list(range(n))[slice(*parts)]


# ── hadrons ─────────────────────────────────────────────────────────────────────
def jet_axis(ini):
    """phi and rapidity of the hardest shower initiator ((K, 11) INITIATOR_COLUMNS)."""
    if ini is None or len(ini) == 0:
        return np.nan, np.nan
    k = np.argmax(np.hypot(ini[:, 3], ini[:, 4]))
    px, py, pz, E = ini[k, 3:7]
    return float(np.arctan2(py, px)), float(0.5 * np.log((E + pz) / (E - pz)))


def unit_observables(h, u, phi_j, y_j):
    """Per-sample means of one unit of a Hadrons object."""
    from jetscape.hadrons_h5 import EventHadrons

    ev = EventHadrons(h, u)
    ns = max(ev.n_samples, 1)
    out = {"n_samples": ev.n_samples}
    ch = ev.charged
    out["dndeta"] = np.histogram(ev.eta[ch], ETA_EDGES)[0] / ns
    mid = np.abs(ev.y) < 0.5
    out["dndy"] = np.array([np.sum(mid & (ev.pid == s)) for s in SPECIES_IDS]) / ns
    out["sumpt"] = np.array([ev.pt[mid & (ev.pid == s)].sum() for s in SPECIES_IDS]) / ns
    out["ptspec"] = np.stack([np.histogram(ev.pt[mid & np.isin(ev.pid, g)], PT_EDGES)[0]
                              for g in SPECIES_GROUPS]) / ns / np.diff(PT_EDGES)
    e1 = np.abs(ev.eta) < 1
    out["ptspec_ch"] = np.histogram(ev.pt[ch & e1], PTCH_EDGES)[0] / ns / np.diff(PTCH_EDGES)
    # flow vectors: per pT bin and integrated (0.2 < pT < 3)
    sel = ch & e1
    pt, phi = ev.pt[sel], ev.phi[sel]
    ib = np.searchsorted(PTV_EDGES, pt, side="right") - 1
    nb = len(PTV_EDGES) - 1
    inside = (ib >= 0) & (ib < nb)
    Q = np.zeros((NMAX, nb + 1), dtype=np.complex128)
    NQ = np.zeros(nb + 1)
    NQ[:nb] = np.bincount(ib[inside], minlength=nb)
    NQ[nb] = inside.sum()
    for n in range(1, NMAX + 1):
        z = np.exp(1j * n * phi[inside])
        Q[n - 1, :nb] = (np.bincount(ib[inside], weights=z.real, minlength=nb)
                         + 1j * np.bincount(ib[inside], weights=z.imag, minlength=nb))
        Q[n - 1, nb] = z.sum()
    out["Q"], out["NQ"] = Q / ns, NQ / ns
    half = inside & (ev.sample[sel] < ns // 2)
    out["QA"] = np.array([np.exp(1j * n * phi[half]).sum() for n in range(1, NMAX + 1)])
    out["NQA"] = float(half.sum())
    out["E_eta1"] = ev.E[e1].sum() / ns
    out["N_all"] = len(ev) / ns
    out["E_all"] = ev.E.sum() / ns
    # jet-relative
    if np.isfinite(phi_j):
        dphi = np.mod(ev.phi - phi_j + np.pi, 2 * np.pi) - np.pi
        soft = ch & (ev.pt < 4) & (np.abs(ev.eta - y_j) < 1)
        out["dphi_jet"] = np.histogram(dphi[soft], DPHI_EDGES)[0] / ns
        out["dphi_jet_pt"] = np.histogram(dphi[soft], DPHI_EDGES, weights=ev.pt[soft])[0] / ns
        near = soft & (np.abs(dphi) < 1)
        out["pt_near"] = np.histogram(ev.pt[near], PTCH_EDGES)[0] / ns / np.diff(PTCH_EDGES)
    else:
        out["dphi_jet"] = np.full(len(DPHI_EDGES) - 1, np.nan)
        out["dphi_jet_pt"] = np.full(len(DPHI_EDGES) - 1, np.nan)
        out["pt_near"] = np.full(len(PTCH_EDGES) - 1, np.nan)
    return out


def hadron_attrs(path):
    import h5py
    with h5py.File(path, "r") as f:
        return {k: (v.item() if hasattr(v, "item") else v) for k, v in f.attrs.items()
                if k in MATCH_ATTRS or k in ("oversample_bg", "source_uuid")}


# ── surfaces ────────────────────────────────────────────────────────────────────
def surface_observables(c, deta):
    """Volume and flow summaries of one surface (N, 32) (SURFACE_COLUMNS)."""
    tau, eta = c[:, 0].astype(np.float64), c[:, 3].astype(np.float64)
    ds, u = c[:, 4:8].astype(np.float64), c[:, 8:12].astype(np.float64)
    dV = tau * (ds[:, 0] * u[:, 0] + ds[:, 1] * u[:, 1] + ds[:, 2] * u[:, 2]) + ds[:, 3] * u[:, 3]
    out = {"n_cells": len(c), "V": dV.sum(), "V_neg": dV[dV < 0].sum()}
    out["dVdtau"] = np.histogram(tau, TAU_EDGES, weights=dV)[0] / np.diff(TAU_EDGES)
    K = 8                                     # spread each cell over its eta width
    sub = (np.arange(K) + 0.5) / K - 0.5
    out["dVdeta"] = (np.histogram((eta[:, None] + deta * sub[None, :]).ravel(), ETAS_EDGES,
                                  weights=np.repeat(dV / K, K))[0] / np.diff(ETAS_EDGES))
    mid = np.abs(eta) < 1
    ur = np.hypot(u[:, 1], u[:, 2])
    out["dVdur"] = np.histogram(ur[mid], UR_EDGES, weights=dV[mid])[0]
    w = dV[mid]
    out["mean_ur"] = float((ur[mid] * w).sum() / w.sum())
    # viscous fields, raw [GeV/fm^3]: the pressure column of MUSIC's in-memory surface is
    # not usable (PLAN_particlize_h5.md), so the notebook divides by e + P at T_sw from the EoS
    pi = c[mid, 21:31].astype(np.float64)
    p00, p01, p02, p03, p11, p12, p13, p22, p23, p33 = pi.T
    pipi = (p00 ** 2 + p11 ** 2 + p22 ** 2 + p33 ** 2
            - 2 * (p01 ** 2 + p02 ** 2 + p03 ** 2) + 2 * (p12 ** 2 + p13 ** 2 + p23 ** 2))
    out["Pi_mean"] = float((c[mid, 31] * w).sum() / w.sum())
    out["pipi_mean"] = float((np.sqrt(np.clip(pipi, 0, None)) * w).sum() / w.sum())
    return out


def main(argv=None):
    a = parse_args(argv)
    from jetscape.hadrons_h5 import Hadrons, HadronFile
    from jetscape.particlize_h5 import ParticlizeFile

    stems = stems_of(variant_dir(a, "ref"))
    for v in a.variants:
        have = stems_of(variant_dir(a, v))
        if have != stems:
            sys.exit(f"analyze.py: {v} has production files {have}, the reference {stems}")
    # events and backgrounds per file, jet axes (from the reference's bulk_jet initiators)
    evs, bg_units = [], []
    for s in stems:
        with ParticlizeFile(os.path.join(variant_dir(a, "ref"), f"{s}_particlize.h5")) as pf:
            evs.append(event_range(a.events, pf.nevents))
            bg_units.append(np.asarray(pf.events("bg_unit"), dtype=np.int64))
    nev = [len(x) for x in evs]
    G = int(sum(nev))
    phi_j, y_j = np.full(G, np.nan), np.full(G, np.nan)
    g = 0
    for s, el in zip(stems, evs):
        with HadronFile(os.path.join(variant_dir(a, "ref"), f"{s}_hadrons_bulk_jet.h5")) as hf:
            for e in el:
                phi_j[g], y_j[g] = jet_axis(hf.initiators(e))
                g += 1

    res = {"jet/phi": phi_j, "jet/y": y_j,
           "event/file": np.repeat(np.arange(len(stems)), nev),
           "event/local": np.concatenate([np.asarray(x, dtype=np.int64) for x in evs])}
    meta = {"stems": stems, "nevents": nev, "events": a.events, "variants": {},
            "hadron_attrs": {}}
    t0 = time.time()
    for v in a.variants:
        d = variant_dir(a, v)
        vmeta = {"dir": d}
        if v != "ref":
            import h5py
            with h5py.File(os.path.join(d, f"{stems[0]}_particlize.h5"), "r") as f:
                vmeta["settings"] = json.loads(f.attrs.get("hvs_settings", "{}"))
        meta["variants"][v] = vmeta
        for leg, tag in (("jet", "bulk_jet"), ("bg", "bulk_bg")):
            rows = []
            attrs = []
            g = 0
            for fi, (s, el) in enumerate(zip(stems, evs)):
                path = os.path.join(d, f"{s}_hadrons_{tag}.h5")
                if not os.path.exists(path):
                    sys.exit(f"analyze.py: {path} missing (run hadronize.py first)")
                attrs.append(hadron_attrs(path))
                h = Hadrons.from_h5(path)
                for e in el:
                    unit = e if leg == "jet" else int(bg_units[fi][e])
                    rows.append(unit_observables(h, h.unit_index(unit), phi_j[g], y_j[g]))
                    g += 1
                del h
            for k in rows[0]:
                res[f"had/{v}/{leg}/{k}"] = np.stack([np.asarray(r[k]) for r in rows])
            meta["hadron_attrs"][f"{v}/{tag}"] = attrs
            print(f"analyze.py: hadrons {v}/{leg}: {len(rows)} events "
                  f"({time.time() - t0:.0f} s)", flush=True)

    # the comparison is only fair if the variants were hadronized like the reference
    ref_attrs = meta["hadron_attrs"]["ref/bulk_jet"][0]
    for key, lst in meta["hadron_attrs"].items():
        for at in lst:
            diff = {k: (at.get(k), ref_attrs.get(k)) for k in MATCH_ATTRS
                    if at.get(k) != ref_attrs.get(k)}
            if diff:
                print(f"analyze.py: WARNING -- {key} hadronized differently from the "
                      f"reference: {diff}", file=sys.stderr)
                break

    if a.surface:
        import h5py
        for v in [x for x in a.variants if x in ("ref", "hist")]:
            d = variant_dir(a, v)
            for leg in ("jet", "bg"):
                rows = []
                for fi, (s, el) in enumerate(zip(stems, evs)):
                    with ParticlizeFile(os.path.join(d, f"{s}_particlize.h5")) as pf:
                        if v == "hist":
                            deta = json.loads(pf.attrs["hvs_settings"])["lattice"][2]
                        else:
                            with h5py.File(os.path.join(d, str(pf.attrs["pair_file"])),
                                           "r") as pr:
                                deta = float(pr.attrs["deta_MUSIC"])
                        for e in el:
                            unit = e if leg == "jet" else int(bg_units[fi][e])
                            rows.append(surface_observables(pf.surface_unit(leg, unit), deta))
                for k in rows[0]:
                    res[f"surf/{v}/{leg}/{k}"] = np.array([r[k] for r in rows])
                print(f"analyze.py: surfaces {v}/{leg} ({time.time() - t0:.0f} s)", flush=True)

    edges = {"ETA_EDGES": ETA_EDGES, "PT_EDGES": PT_EDGES, "PTCH_EDGES": PTCH_EDGES,
             "PTV_EDGES": PTV_EDGES, "DPHI_EDGES": DPHI_EDGES, "TAU_EDGES": TAU_EDGES,
             "ETAS_EDGES": ETAS_EDGES, "UR_EDGES": UR_EDGES,
             "SPECIES_IDS": np.array(SPECIES_IDS)}
    res.update({f"edges/{k}": v for k, v in edges.items()})
    res["meta"] = np.array(json.dumps(meta, default=str))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, **res)
    print(f"analyze.py: -> {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB, "
          f"{time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
