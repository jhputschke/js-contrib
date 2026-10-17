#!/usr/bin/env python3
"""pp (vacuum) Z+jet against CMS 1702.01060 (HEPData ins1512112): anti-k_T R = 0.3 particle-level
jets from all final particles except neutrinos and the Z's leptons, smeared with CMS's resolution
(Appendix A): sigma = sqrt(C^2 + S^2/pT + N^2/pT^2), C 0.061; pp S 0.95 N 0.001 (vs the
unsmeared pp table), PbPb 0-30 % S 1.24 N 8.08 (vs the 'smeared pp' tables, smeared to PbPb's
resolution).  Z: opposite-sign pair, 70-110 GeV, muons |eta| < 2.4, electrons |eta| < 2.5 outside
1.44-1.57 (both p_T > 20 from the generator); jets p_T > 30, |eta| < 1.6, dR(lepton) > 0.4.
Events weigh window sigma / N, every fragmentation sample counts 1/n_frag."""
import glob
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                        # zjet_wake.py
sys.path.insert(0, HERE)                                         # compare.py
try:
    import jetscape  # noqa: F401
except ImportError:                                              # the source tree
    sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "python"))
import numpy as np, yaml, fastjet
import jetscape.h5_compression  # noqa: F401
import h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figdata

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--vacuum", default=os.path.join(HERE, "data", "vacuum_all"),
                help="run_prod_vacuum.py output with all particles (make_data.sh vacuum_all)")
ap.add_argument("--hepdata", default=os.path.join(HERE, "data", "hepdata_zjet", "HEPData-ins1512112-v1-yaml"))
ap.add_argument("--outdir", default=os.path.join(HERE, "figures"))
ap.add_argument("--limit", type=int, default=None, help="only the first N selected Z's (a test)")
a = ap.parse_args()
VAC, HEP, OUT = a.vacuum, a.hepdata, a.outdir
os.makedirs(OUT, exist_ok=True)
C = 0.061
RES = {"pp": (0.95, 0.001), "PbPb 0-30 %": (1.24, 8.08)}
NU = (12, 14, 16)
PHI_EDGES = np.array([0, np.pi/5, 2*np.pi/5, 3*np.pi/5, 7*np.pi/10, 4*np.pi/5, 9*np.pi/10, 19*np.pi/20, np.pi])
X_EDGES = np.linspace(0, 2, 11)
ZPT_BINS = [(40, 50), (50, 60), (60, 80), (80, 120)]
rng = np.random.default_rng(7)
jdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.3)

f = h5py.File(glob.glob(f"{VAC}/*_hadrons_jet_frag.h5")[0], "r")
u = f["units"]
summ = __import__("json").load(open(glob.glob(f"{VAC}/*.json")[0]))
sig = np.array(summ["pthat_bin_sigma_gen_mb"]) * 1e9
pbin = u["pthat_bin"][:]
nk = np.bincount(pbin, minlength=len(sig))
w_ev = sig[pbin] / nk[pbin]
Z = {k: u[k][:] for k in ("z_pt", "z_y", "z_phi", "z_m", "lep1_pid", "lep1_pt", "lep1_eta", "lep1_phi",
                          "lep2_pid", "lep2_pt", "lep2_eta", "lep2_phi")}
def lep_ok(pid, eta):
    a = np.abs(eta)
    return np.where(np.abs(pid) == 13, a < 2.4, (a < 2.5) & ~((a > 1.44) & (a < 1.57)))
zsel = (Z["z_m"] > 70) & (Z["z_m"] < 110) & (Z["z_pt"] > 40) & lep_ok(Z["lep1_pid"], Z["lep1_eta"]) & lep_ok(Z["lep2_pid"], Z["lep2_eta"])
P, PID = f["hadrons/p"], f["hadrons/pid"]
so, uo = f["hadrons/sample_offsets"][:], f["hadrons/unit_offsets"][:]

def dphi(a, b):
    d = np.abs(a - b) % (2 * np.pi)
    return np.where(d > np.pi, 2 * np.pi - d, d)

# per Z: list over samples of jets (pt_gen, eta, phi)
acc = {k: dict(phi=np.zeros(len(PHI_EDGES) - 1), x=np.zeros(len(X_EDGES) - 1), nz=0.0,
               rjz=np.zeros(len(ZPT_BINS)), nzb=np.zeros(len(ZPT_BINS)), xsum=np.zeros(len(ZPT_BINS)),
               x2=np.zeros(len(ZPT_BINS)), npair=np.zeros(len(ZPT_BINS))) for k in RES}
n_used = 0
for e in np.flatnonzero(zsel)[:a.limit]:
    s0, s1 = int(uo[e]), int(uo[e + 1])
    nfr = s1 - s0
    lep = [(Z[f"lep{i}_eta"][e], Z[f"lep{i}_phi"][e]) for i in (1, 2)]
    zpt, zphi = Z["z_pt"][e], Z["z_phi"][e]
    w = w_ev[e] / nfr
    for s in range(s0, s1):
        p = P[so[s]:so[s + 1]].astype(np.float64); pid = PID[so[s]:so[s + 1]]
        pt = np.hypot(p[:, 1], p[:, 2]); eta = np.arcsinh(p[:, 3] / np.maximum(pt, 1e-9)); phi = np.arctan2(p[:, 2], p[:, 1])
        keep = ~np.isin(np.abs(pid), NU)
        for le, lp in lep:                      # the Z's leptons: the closest charged lepton to each
            cand = np.flatnonzero(np.isin(np.abs(pid), (11, 13)) & keep)
            if len(cand):
                dr = np.hypot(eta[cand] - le, dphi(phi[cand], lp))
                if dr.min() < 1e-3: keep[cand[np.argmin(dr)]] = False
        pj = [fastjet.PseudoJet(*map(float, (q[1], q[2], q[3], q[0]))) for q in p[keep]]
        jets = [j for j in fastjet.sorted_by_pt(fastjet.ClusterSequence(pj, jdef).inclusive_jets(10.0))
                if abs(j.eta()) < 1.6 and all(np.hypot(j.eta() - le, dphi(j.phi_std(), lp)) > 0.4 for le, lp in lep)]
        for key, (S, N) in RES.items():
            a = acc[key]
            a["nz"] += w * (zpt > 60)
            ib = [i for i, (lo, hi) in enumerate(ZPT_BINS) if lo <= zpt < hi]
            if ib: a["nzb"][ib[0]] += w
            for j in jets:
                pg = j.pt()
                pr = pg * rng.normal(1.0, np.sqrt(C**2 + S**2 / pg + N**2 / pg**2))
                if pr < 30: continue
                dp = dphi(j.phi_std(), zphi); x = pr / zpt
                if zpt > 60:
                    a["phi"] += w * np.histogram([dp], PHI_EDGES)[0]
                    if dp > 7 * np.pi / 8: a["x"] += w * np.histogram([x], X_EDGES)[0]
                if ib and dp > 7 * np.pi / 8:
                    a["rjz"][ib[0]] += w; a["xsum"][ib[0]] += w * x; a["x2"][ib[0]] += w * x * x; a["npair"][ib[0]] += w
    n_used += 1

def cms(i):
    d = yaml.safe_load(open(os.path.join(HEP, f"data{i}.yaml")))
    v = d["dependent_variables"][0]["values"]
    y = np.array([float(b["value"]) for b in v]); e = np.array([np.hypot(*[float(x["symerror"]) for x in b["errors"]]) for b in v])
    return y, e
out = [f"{n_used} selected Z's (p_T^Z > 40, CMS 1702.01060 lepton cuts) from {os.path.basename(VAC)}"]
fig, ax = figdata.subplots(1, 4, figsize=(19, 4.3))
phic = 0.5 * (PHI_EDGES[1:] + PHI_EDGES[:-1]); xc = 0.5 * (X_EDGES[1:] + X_EDGES[:-1]); zc = [45, 55, 70, 100]
for key, sty in (("pp", "s"), ("PbPb 0-30 %", "o")):
    a = acc[key]
    phi = a["phi"] / a["nz"] / np.diff(PHI_EDGES); x = a["x"] / a["nz"] / np.diff(X_EDGES)
    rjz = a["rjz"] / a["nzb"]; mx = a["xsum"] / a["npair"]
    tag = "model, pp-smeared" if key == "pp" else "model, PbPb-smeared"
    ax[0].plot(phic, phi, sty + "--", label=tag); ax[1].plot(xc, x, sty + "--", label=tag)
    ax[2].plot(zc, mx, sty + "--", label=tag); ax[3].plot(zc, rjz, sty + "--", label=tag)
    out.append(f"{tag:22s} R_jZ(p_T^Z > 60, dphi > 7pi/8) {a['x'].sum() / a['nz']:.3f};  <x_jZ> {np.round(mx, 3).tolist()};  R_jZ {np.round(rjz, 3).tolist()}")
    a.update(phi_d=phi, x_d=x, rjz_d=rjz, mx_d=mx)
for i, axi, xv, lab in ((1, 0, phic, "CMS smeared pp"), (3, 1, xc, "CMS smeared pp"), (9, 1, xc, "CMS pp, unsmeared"),
                        (5, 2, zc, "CMS smeared pp"), (7, 3, zc, "CMS smeared pp")):
    y, e = cms(i)
    ax[axi].errorbar(np.array(xv) + (0.02 if i == 9 else 0), y, e, fmt="o" if i != 9 else "D", color="k" if i != 9 else "C2", ms=4, label=lab)
    m = acc["PbPb 0-30 %"] if i != 9 else acc["pp"]
    mv = {1: m["phi_d"], 3: m["x_d"], 9: m["x_d"], 5: m["mx_d"], 7: m["rjz_d"]}[i]
    chi = np.sum((mv - y) ** 2 / e ** 2)
    out.append(f"  vs data{i} ({lab}): chi2 {chi:.1f}/{len(y)} (CMS errors only)")
for a_, t in zip(ax, ("Δφ_jZ, p_T^Z > 60", "x_jZ, p_T^Z > 60, Δφ > 7π/8", "⟨x_jZ⟩ vs p_T^Z", "R_jZ vs p_T^Z")):
    a_.set_title(t); a_.legend(fontsize=7)
ax[0].set_yscale("log")
fig.tight_layout(); figdata.savefig(fig, os.path.join(OUT, "zjet_pp_vs_cms1702.png"), dpi=100)
print("\n".join(out)); open(os.path.join(OUT, "zjet_pp_summary.txt"), "w").write("\n".join(out) + "\n")
