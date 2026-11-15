import sys, time
OUT = sys.argv[1]
import numpy as np
import fastjet as fj
from jetscape.hadrons_h5 import CHARGED, HadronFileReader

ETA_MAX, R = 1.5, 0.4
JET_DEF = fj.JetDefinition(fj.antikt_algorithm, R)
KT_DEF = fj.JetDefinition(fj.kt_algorithm, R)
AREA_DEF = fj.AreaDefinition(fj.active_area_explicit_ghosts, fj.GhostedAreaSpec(ETA_MAX, 1, 0.005))
CHARGED_PID = np.array(CHARGED)

def pseudojets(ev, charged=True, pt_min=0.15):
    """Massless PseudoJets (E = |p|) of one sample; user_index = row in the sample."""
    E, px, py, pz = np.asarray(ev["p"], dtype=np.float64).T
    pt = np.hypot(px, py)
    eta = np.arcsinh(pz / np.maximum(pt, 1e-12))
    keep = (np.abs(eta) < ETA_MAX) & (pt > pt_min)
    if charged:
        keep &= np.isin(np.abs(ev["pid"]), CHARGED_PID)
    out = []
    for i in np.flatnonzero(keep):
        pj = fj.PseudoJet(px[i], py[i], pz[i], np.sqrt(pt[i] ** 2 + pz[i] ** 2))
        pj.set_user_index(int(i))
        out.append(pj)
    return out

t0 = time.time()
r = HadronFileReader(OUT)
rho_sel = (~fj.SelectorNHardest(2)) * fj.SelectorAbsRapMax(ETA_MAX - R)
jet_sel = fj.SelectorAbsRapMax(ETA_MAX - R) * fj.SelectorPtMin(10.0)
pt_sub = []
for e in r.events():
    for k in range(min(r.n_oversamples(e), 20)):
        parts = pseudojets(r.jet_event(e, k))              # bulk (bkg + wake) + fragments
        cs = fj.ClusterSequenceArea(parts, JET_DEF, AREA_DEF)
        bge = fj.JetMedianBackgroundEstimator(rho_sel, KT_DEF, AREA_DEF)
        bge.set_use_area_4vector(False)                    # rho = median pT/A, scalar area
        bge.set_particles(parts)
        rho = bge.rho()
        for jet in fj.sorted_by_pt(jet_sel(cs.inclusive_jets())):
            pt_sub.append(jet.pt() - rho * jet.area())
print(len(pt_sub), "jets", np.round(sorted(pt_sub)[-5:], 1), "rho", rho, time.time() - t0)

# --- array interface ---
import awkward as ak, vector
vector.register_awkward()

def soft_charged(ev, info):
    return ev.charged & (abs(ev.eta) < ETA_MAX) & (ev.pt > 0.15)

t0 = time.time()
arr = r.hadron_array("bulk_jet", events=0, samples=slice(0, 100), mask=soft_charged)
ev = arr.to_awkward(massless=True)
cs = fj.ClusterSequence(ev, JET_DEF)                       # all 100 samples in one call
jets = cs.inclusive_jets(min_pt=10.0)
cons = cs.constituent_index(min_pt=10.0)
print(ak.num(jets)[:10].tolist(), ak.max(jets.pt), time.time() - t0)

# the same with several events: events x samples x jets
arr = r.hadron_array("jet_frag", events=range(5), mask=soft_charged)
jets = fj.ClusterSequence(arr.to_awkward(by_unit=True, pion_mass=True), JET_DEF).inclusive_jets(min_pt=10.0)
print(jets.type, arr.info["units"]["event"].tolist())
