import sys, time
CAMPAIGN = sys.argv[1]
t0 = time.time()
# ---- block 1: load a campaign
import glob
import h5py, hdf5plugin, numpy as np
from jetscape.hadrons_h5 import CHARGED, pseudorapidity

def read(path):
    """units/ and the hadrons of one file; 'event' = the unit's position, 'sample' within it."""
    with h5py.File(path, "r") as f:
        units = {k: f["units"][k][:] for k in f["units"]}
        attrs = dict(f.attrs)
        so, uo = f["hadrons/sample_offsets"][:], f["hadrons/unit_offsets"][:]
        p, pid = f["hadrons/p"][:].astype(np.float64), f["hadrons/pid"][:]
    n_had = np.diff(so)
    unit_of_sample = np.repeat(np.arange(len(uo) - 1), np.diff(uo))
    h = {"event": np.repeat(unit_of_sample, n_had),
         "sample": np.repeat(np.arange(len(n_had)) - uo[:-1][unit_of_sample], n_had),
         "pid": pid, "E": p[:, 0], "px": p[:, 1], "py": p[:, 2], "pz": p[:, 3]}
    return units, attrs, h

def load(campaign):
    """Every finished job (its .json) of a campaign: one event table, bulk and jet hadrons."""
    stems = sorted(p[:-5] for p in glob.glob(f"{campaign}/**/*.json", recursive=True))
    ev, bulk, jet, n0 = [], [], [], 0
    for i, stem in enumerate(stems):
        u, attrs, hj = read(f"{stem}_hadrons_jet_frag.h5")      # the event record: always here
        ub, _, hb = read(f"{stem}_hadrons_bulk_bg.h5")          # in the job or hadronize.py
        assert len(ub["unit"]) == len(u["unit"])                # unit i = event i in both
        u["n_samples_jet"] = u.pop("n_samples")
        u["n_samples_bg"] = ub["n_samples"]                     # 0: no medium
        u["file"] = np.full(len(u["unit"]), i)
        for k in ("pthat_bin_sigma_gen", "pthat_bin_n_accepted"):   # per window, this file
            u[k] = np.tile(np.atleast_1d(attrs.get(k, np.nan)), (len(u["unit"]), 1))
        hj["event"] += n0
        hb["event"] += n0
        ev.append(u), jet.append(hj), bulk.append(hb)
        n0 += len(u["unit"])
    cat = lambda parts: {k: np.concatenate([x[k] for x in parts]) for k in parts[0]}
    ev, bulk, jet = cat(ev), cat(bulk), cat(jet)
    for h in (bulk, jet):
        h["pt"] = np.hypot(h["px"], h["py"])
        h["eta"] = pseudorapidity(np.stack([h["E"], h["px"], h["py"], h["pz"]], axis=1))
        h["phi"] = np.arctan2(h["py"], h["px"])
        h["charged"] = np.isin(np.abs(h["pid"]), CHARGED)
    return ev, bulk, jet

ev, bulk, jet = load(CAMPAIGN)
n_ev = len(ev["unit"])
print(n_ev, "events,", len(bulk["E"]), "bulk hadrons,", len(jet["E"]), "jet hadrons", f"{time.time()-t0:.1f}s")

# ---- block 2: weights
first = np.unique(ev["file"], return_index=True)[1]          # one row per file
sig, n_acc = ev["pthat_bin_sigma_gen"][first], ev["pthat_bin_n_accepted"][first]
sigma_k = (n_acc * sig).sum(0) / n_acc.sum(0)                 # mb per window, as HadronFileReader
k = ev["pthat_bin"]
K = len(sigma_k)
N_k = np.bincount(k, minlength=K)
w_hard = sigma_k[k] / N_k[k]                     # per hard event: sum over events = sigma [mb]
SIGMA_NN = 42.0                                  # mb, inelastic NN at 200 GeV
w_coll = ev["glauber_ncoll"] * sigma_k[k] / SIGMA_NN / N_k[k]   # per O+O collision in the b range
print("sigma_k", sigma_k, "N_k", N_k, "sum w_hard", w_hard.sum(), "per-collision yield", w_coll.sum())

# ---- block 3: checks
E_beam = ev["glauber_npart"] * 200.0 / 2
E_sum = ev["strings_E_total"] + ev["isr_sub_E_total"] + ev["remnant_E_proj"] + ev["remnant_E_targ"]
med = ev["has_medium"] == 1                                  # no medium: strings_E_total is NaN
print("bookkeeping max |dE|/E", np.max(np.abs(E_sum[med] - E_beam[med]) / E_beam[med]),
      "| no medium", 1 - med.mean(), "| bulk samples", np.unique(ev["n_samples_bg"]))

# ---- block 4: dN_ch/deta per event (bulk: mean over its samples) + jet hadrons
n_samp = ev["n_samples_bg"]
mid_b = bulk["charged"] & (np.abs(bulk["eta"]) < 0.5)
mid_j = jet["charged"] & (np.abs(jet["eta"]) < 0.5)
dn_bulk = np.bincount(bulk["event"][mid_b], minlength=n_ev) / np.maximum(n_samp, 1)
dn_jet = np.bincount(jet["event"][mid_j], minlength=n_ev)
dndeta = dn_bulk + dn_jet                                # |eta| < 0.5, so /1
w = ev["glauber_ncoll"]                                  # the geometry of hard events
print("dNch/deta (Ncoll-weighted)", np.average(dndeta, weights=w), "unweighted", dndeta.mean())

# ---- block 5: v2{2} from pairs of different oversamples
def vn2(h, n, n_samples, mask):
    m = mask & (n_samples[h["event"]] >= 2)
    e, s, q = h["event"][m], h["sample"][m], np.exp(1j * n * h["phi"][m])
    key = e * 100000 + s                                  # (event, sample)
    uk, inv = np.unique(key, return_inverse=True)
    Qs = np.bincount(inv, weights=q.real) + 1j * np.bincount(inv, weights=q.imag)
    Ms = np.bincount(inv).astype(float)
    ev_of = uk // 100000
    Q = np.bincount(ev_of, weights=Qs.real) + 1j * np.bincount(ev_of, weights=Qs.imag)
    M = np.bincount(ev_of, weights=Ms)
    QQ = np.bincount(ev_of, weights=np.abs(Qs) ** 2)
    MM = np.bincount(ev_of, weights=Ms ** 2)
    num, den = np.abs(Q) ** 2 - QQ, M ** 2 - MM           # pairs from different samples only
    ok = den > 0
    return np.sqrt(num[ok].sum() / den[ok].sum()), num[ok] / den[ok], np.flatnonzero(ok)

sel = bulk["charged"] & (np.abs(bulk["eta"]) < 1) & (bulk["pt"] > 0.2) & (bulk["pt"] < 3)
v2, per_ev, evs = vn2(bulk, 2, n_samp, sel)
print("v2{2, different samples}", v2, "events", len(evs))

# ---- block 6: jets with FastJet: bulk sample 0 + the jet hadrons
import fastjet as fj
JET_DEF = fj.JetDefinition(fj.antikt_algorithm, 0.4)
def pjs(h, m):
    pt = h["pt"][m]
    return [fj.PseudoJet(a, b, c, np.hypot(d, c)) for a, b, c, d in
            zip(h["px"][m], h["py"][m], h["pz"][m], pt)]
acc_b = bulk["charged"] & (np.abs(bulk["eta"]) < 1.5) & (bulk["pt"] > 0.15) & (bulk["sample"] == 0)
acc_j = jet["charged"] & (np.abs(jet["eta"]) < 1.5) & (jet["pt"] > 0.15)
lead = []
order_b, order_j = np.argsort(bulk["event"], kind="stable"), np.argsort(jet["event"], kind="stable")
cut_b = np.searchsorted(bulk["event"][order_b], np.arange(n_ev + 1))
cut_j = np.searchsorted(jet["event"][order_j], np.arange(n_ev + 1))
for e in range(min(n_ev, 300)):
    ib, ij = order_b[cut_b[e]:cut_b[e + 1]], order_j[cut_j[e]:cut_j[e + 1]]
    parts = pjs({k2: v[ib] for k2, v in bulk.items()}, acc_b[ib]) + pjs({k2: v[ij] for k2, v in jet.items()}, acc_j[ij])
    jets = fj.sorted_by_pt(fj.SelectorAbsRapMax(1.1)(fj.ClusterSequence(parts, JET_DEF).inclusive_jets(5.0)))
    lead.append(jets[0].pt() if jets else 0.0)
print("leading jet pT mean", np.mean(lead), f"{time.time()-t0:.1f}s")

# ---- block 7: dphi to the leading initiator
from jetscape.hadrons_h5 import HadronFile
stems = sorted(p[:-5] for p in glob.glob(f"{CAMPAIGN}/**/*.json", recursive=True))
with HadronFile(f"{stems[0]}_hadrons_jet_frag.h5") as hf:
    ini = hf.initiators(int(ev["unit"][0]))       # (K, 11): shower, pid, pstat, px, py, pz, E, x, y, z, t
    print("initiators", ini.shape, np.hypot(ini[:, 3], ini[:, 4]).max())
