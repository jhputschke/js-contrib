#!/usr/bin/env python3
"""
example/prod_AuAu_0_10_jet/music_cpu_gpu_hadrons.py

Freeze-out surfaces and iSS hadrons of MUSIC on the CPU against MUSIC4GPU, for the same
events (the two run_prod_jet.py --write-particlize jobs of music_cpu_gpu_hydro.py).  See
docs/MUSIC_CPU_vs_GPU.md.

    # 1. give the CPU particlize file the GPU file's file_uuid: hadronize.py derives every
    #    unit's iSS seed from it, so both paths then draw the same random numbers
    python music_cpu_gpu_hadrons.py --match-seeds OUT/gpu OUT/cpu
    # 2. hadronize both the same way; --correlated addresses iSS's random numbers by the cell,
    #    so the two give the same hadrons wherever their surfaces agree
    python hadronize.py OUT/gpu/<stem>_particlize.h5 --tags bulk_jet,bulk_bg --oversample 200 --correlated
    python hadronize.py OUT/cpu/<stem>_particlize.h5 --tags bulk_jet,bulk_bg --oversample 200 --correlated
    # 3. compare
    python music_cpu_gpu_hadrons.py OUT/gpu OUT/cpu

Surfaces: the summaries of hydro_hist_vs_surface/analyze.py (cells, volume V = sum dsigma.u,
its negative part, the V-weighted transverse flow and viscous stresses at |eta_s| < 1).

Hadrons: per oversample (sample k of one path and sample k of the other are paired), the
charged multiplicity, identified dN/dy and <pT> at |y| < 0.5, and the total energy; the paired
difference and its standard error.  v2, v3: charged, |eta| < 1, 0.2 < pT < 3 GeV, from all
oversamples of the event (they share its event plane), |Q_n| / N; the error of their
difference from the two halves of the oversamples.

Needs the environment PyJetscape was built in (js_fno on the GB10).
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))

SPECIES = {211: "pi+", 321: "K+", 2212: "p"}


def analyze_module():
    path = os.path.join(HERE, "..", "hydro_hist_vs_surface", "analyze.py")
    spec = importlib.util.spec_from_file_location("hvs_analyze", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stem_of(d, stem):
    if stem:
        return stem
    found = sorted(os.path.basename(p)[:-len("_particlize.h5")]
                   for p in glob.glob(os.path.join(d, "*_particlize.h5")))
    if len(found) != 1:
        sys.exit(f"music_cpu_gpu_hadrons.py: {d} has {len(found)} particlize files; pass --stem")
    return found[0]


def match_seeds(gd, cd, stem):
    with h5py.File(os.path.join(gd, f"{stem}_particlize.h5"), "r") as g:
        uuid = g.attrs["file_uuid"]
    with h5py.File(os.path.join(cd, f"{stem}_particlize.h5"), "a") as c:
        if "cpu_original_file_uuid" not in c.attrs:
            c.attrs["cpu_original_file_uuid"] = c.attrs["file_uuid"]
        c.attrs["file_uuid"] = uuid
    print(f"music_cpu_gpu_hadrons.py: {cd}/{stem}_particlize.h5 now has file_uuid {uuid} "
          "(its own kept as cpu_original_file_uuid)")


def units(d, stem):
    from jetscape.particlize_h5 import ParticlizeFile
    with ParticlizeFile(os.path.join(d, f"{stem}_particlize.h5")) as pf:
        return (list(range(pf.nevents)), sorted(set(int(x) for x in pf.events("bg_unit"))),
                str(pf.attrs.get("file_uuid", "")))


def surfaces(an, d, stem, leg, us):
    from jetscape.particlize_h5 import ParticlizeFile
    with h5py.File(os.path.join(d, f"{stem}.h5"), "r") as pr:
        deta = float(pr.attrs["deta_MUSIC"])
    with ParticlizeFile(os.path.join(d, f"{stem}_particlize.h5")) as pf:
        return [an.surface_observables(pf.surface_unit(leg, u), deta) for u in us]


def per_sample(ev):
    ns, s, ch = ev.n_samples, ev.sample, ev.charged
    out = {"dNch/deta |eta|<0.5": np.bincount(s[ch & (np.abs(ev.eta) < 0.5)], minlength=ns),
           "Nch |eta|<5": np.bincount(s[ch & (np.abs(ev.eta) < 5)], minlength=ns),
           "E all": np.bincount(s, weights=ev.E, minlength=ns)}
    midy = np.abs(ev.y) < 0.5
    for pid, name in SPECIES.items():
        m = midy & (ev.pid == pid)
        n = np.bincount(s[m], minlength=ns).astype(float)
        out[f"dN/dy {name}"] = n
        out[f"<pT> {name}"] = np.bincount(s[m], weights=ev.pt[m], minlength=ns) / np.maximum(n, 1)
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def flow(ev, half=None):
    sel = ev.charged & (np.abs(ev.eta) < 1) & (ev.pt > 0.2) & (ev.pt < 3.0)
    if half is not None:
        first = ev.sample < ev.n_samples // 2
        sel &= first if half == 0 else ~first
    phi = ev.phi[sel]
    return {f"v{n}": float(np.abs(np.exp(1j * n * phi).sum()) / max(len(phi), 1))
            for n in (2, 3)}


def compare_hadrons(gd, cd, stem, tag, us):
    from jetscape.hadrons_h5 import EventHadrons, Hadrons
    hg = Hadrons.from_h5(os.path.join(gd, f"{stem}_hadrons_{tag}.h5"))
    hc = Hadrons.from_h5(os.path.join(cd, f"{stem}_hadrons_{tag}.h5"))
    rows = []
    for u in us:
        eg, ec = EventHadrons(hg, hg.unit_index(u)), EventHadrons(hc, hc.unit_index(u))
        if eg.n_samples != ec.n_samples:
            sys.exit(f"music_cpu_gpu_hadrons.py: {tag} unit {u}: {eg.n_samples} vs "
                     f"{ec.n_samples} oversamples")
        pg, pc = per_sample(eg), per_sample(ec)
        row = {}
        for k in pg:
            d, m = pg[k] - pc[k], pc[k].mean()
            row[k] = (m, d.mean() / m, d.std(ddof=1) / np.sqrt(len(d)) / m,
                      pc[k].std(ddof=1) / np.sqrt(len(d)) / m)
        fg, fc = flow(eg), flow(ec)
        ha = [flow(eg, 0), flow(ec, 0), flow(eg, 1), flow(ec, 1)]
        for n in ("v2", "v3"):
            err = abs((ha[0][n] - ha[1][n]) - (ha[2][n] - ha[3][n])) / 2
            row[n] = (fc[n], fg[n] - fc[n], err, abs(ha[0][n] - ha[2][n]) / 2)
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("gpu_dir")
    p.add_argument("cpu_dir")
    p.add_argument("--stem", default=None, help="production file stem (default: the only one)")
    p.add_argument("--match-seeds", action="store_true", dest="match_seeds",
                   help="copy the GPU particlize file's file_uuid into the CPU one and exit")
    a = p.parse_args()
    stem = stem_of(a.gpu_dir, a.stem)
    if a.match_seeds:
        match_seeds(a.gpu_dir, a.cpu_dir, stem)
        return 0
    import hdf5plugin  # noqa: F401

    evs, bgus, ug = units(a.gpu_dir, stem)
    evc, bguc, uc = units(a.cpu_dir, stem)
    if (evs, bgus) != (evc, bguc):
        sys.exit("music_cpu_gpu_hadrons.py: the two particlize files have different events")
    if ug != uc:
        print("music_cpu_gpu_hadrons.py: WARNING -- different file_uuid, so different iSS "
              "seeds: run --match-seeds before hadronize.py for a paired comparison",
              file=sys.stderr)
    an = analyze_module()

    print("=== freeze-out surfaces: GPU/CPU - 1 per unit (CPU value)")
    for leg, us in (("bg", bgus), ("jet", evs)):
        sg, sc = surfaces(an, a.gpu_dir, stem, leg, us), surfaces(an, a.cpu_dir, stem, leg, us)
        for k in ("n_cells", "V", "V_neg", "mean_ur", "Pi_mean", "pipi_mean"):
            print(f"  {leg:3s} {k:9s} " + "  ".join(f"{g[k] / c[k] - 1:+.2e} ({c[k]:.4g})"
                                                    for g, c in zip(sg, sc)))
        print(f"  {leg:3s} dV/dtau   sum|G-C|/sum|C| " + "  ".join(
            f"{np.abs(g['dVdtau'] - c['dVdtau']).sum() / np.abs(c['dVdtau']).sum():.2e}"
            for g, c in zip(sg, sc)))

    print("\n=== hadrons: GPU - CPU, paired per oversample")
    for tag, us in (("bulk_bg", bgus), ("bulk_jet", evs)):
        rows = compare_hadrons(a.gpu_dir, a.cpu_dir, stem, tag, us)
        print(f"\n  -- {tag} ({len(us)} units; n_samples from the files)")
        print("  observable            CPU value per unit  |  (G-C)/C +- paired error per unit"
              "  (v2, v3: G-C absolute)  [independent sampling error]")
        for k in rows[0]:
            fmt = (lambda r: f"{r[k][1]:+.1e}+-{r[k][2]:.0e}")
            print(f"  {k:20s}  " + " ".join(f"{r[k][0]:.4g}" for r in rows) + "  |  "
                  + " ".join(fmt(r) for r in rows)
                  + f"  [{np.mean([r[k][3] for r in rows]):.0e}]")
        print("  mean over units: " + ", ".join(
            f"{k} {np.mean([r[k][1] for r in rows]):+.1e}" for k in rows[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
