# Analysing the O+O productions (`prod_OO_isr`)

This guide is for the analysis of O+O 200 GeV events with initial-state radiation from
[`prod_OO_isr`](../contribs/PyJetscape/example/prod_OO_isr/README.md): what an event is, what
the files hold, how to read and weight them, and recipes for the bulk, flow and jets, in Python
and in ROOT. Running the production is in the folder's README and
[`README_2stage.md`](README_2stage.md) §8; the environments, threads and FastJet installation
are in [`README_analysis.md`](README_analysis.md) §2.

**Contents**
1. [What an O+O event is](#1-what-an-oo-event-is)
2. [The files](#2-the-files)
3. [The per-event record](#3-the-per-event-record)
4. [Settings to check before combining files](#4-settings-to-check-before-combining-files)
5. [Reading a campaign](#5-reading-a-campaign)
6. [Weights and statistics](#6-weights-and-statistics)
7. [Checks](#7-checks)
8. [Observables](#8-observables)
9. [The hydro and particlize files](#9-the-hydro-and-particlize-files)
10. [ROOT](#10-root)
11. [Notebooks and references](#11-notebooks-and-references)
12. [Troubleshooting](#12-troubleshooting)

The code blocks marked *(tested)* ran on 2026-10-13 in `js_fno` on the campaign
`prod_OO_isr/hard_sub_compare/sub/`: 40 jobs × 75 events = 2,991 events in three pT̂ windows,
b < 5.6 fm, hadron files only, 10 bulk oversamples, no positions. The scripts behind them,
and `run_doc_blocks.py` to re-run them, are in [`analysis_snippets/`](analysis_snippets/README.md).

---

## 1. What an O+O event is

```
IS: 3dMCGlauber (O16, positions only)
 -> Hard: PythiaIsrGun at one binary collision      the hard scattering, pT-hat windows
          -> IsrManager / iMATTER                    ISR, backward in time
          -> MCGlauberGenString                      strings from what the hard pair left
 -> MUSIC                                            ONE hydro per event (no two-stage pair)
 -> Eloss: Matter + LBT in MUSIC's medium          the final-state shower
 -> JetHadronization: iColored                       jet hadrons, in the job
 -> SoftParticlization: iSS                          bulk hadrons, in the job or later
```

Differences from the Au+Au jet production that matter for an analysis:

- **One MUSIC run per event, no background leg.** There is no `jet − background` pair: the
  medium response to the jet is part of the bulk and cannot be separated from it.
- **Every event is hard-triggered**: exactly one hard scattering, at a binary collision drawn
  uniformly. The geometry is sampled as in minimum-bias collisions within the b range (P(b) ∝ b,
  Npart ≥ 2), but in nature the hard-process rate grows with Ncoll. Weight with Ncoll for the
  geometry of hard events (§6).
- **The hard partons' momentum is taken from the strings** (`Subtract_hard_momentum` 1, the
  default): the bulk loses what the hard pair and its remnants carry. That biases a
  multiplicity-based centrality towards peripheral, more so at small Npart
  ([`hard_sub_compare`](../contribs/PyJetscape/example/prod_OO_isr/hard_sub_compare/README.md)).
- **The jet hadrons are the whole hard nucleon pair:** the dijet showered by Matter, the ISR
  emissions, Pythia's MPI partons and the pair's beam remnants, hadronized together by
  iColored. Their pstat is mostly 1000 (PythiaIsrGun's mark, which Matter's daughters keep);
  that is not "unshowered".
- **Energy loss: Matter + LBT** (since 2026-10-21; Matter alone before, `--matter-only`).
  The partons carry the colour flow of every scattering off the medium, the holes are not
  hadronized, and iColored closes the colour lines that end in the medium
  (`<colour_closure>`, default `reconnect`); the jet file's `units/colour_*` and the job
  summary's `jet_colour_totals` record it per event and per job
  ([`prod_OO_isr/README.md`](../contribs/PyJetscape/example/prod_OO_isr/README.md), "Jet
  hadronization colour checks"; X-SCAPE `README_ColourFlow.md`).
- **Events without a medium:** when the hard collision is the only one, the pair makes no
  string. There is no evolution, Matter showers in vacuum, the bulk unit is empty
  (`has_medium` 0). In 0–80 % (b < 5.6 fm) that is ~5 % of the events, mostly with Ncoll = 1.
- **Centrality comes from b**, not from `cenMin`/`cenMax` (ignored inside X-SCAPE). For O+O at
  200 GeV with this XML's 3dMCGlauber settings (100k standalone events, σ_inel = 1.55 b) the
  b percentiles are 10 % 1.91, 20 % 2.71, 40 % 3.84, 60 % 4.71, 80 % 5.59 fm. The production's
  default b < 3 fm is about 0–24 %.

## 2. The files

Each job writes, next to each other (stem `OO_isr_<campaign>_NNNN` or `OO_isr_seedNNNN`):

| file | holds | when | size per event |
|---|---|---|---|
| `<stem>_hadrons_jet_frag.h5` | the jet hadrons, one sample per event; `units/` with the whole event record (§3); `initiators/` | always (unless `--no-jet-hadrons`) | small |
| `<stem>_hadrons_bulk_bg.h5` | the bulk (iSS), N oversamples per event | `--bulk-hadrons N` in the job, or later `hadronize.py --tags bulk_bg` | ~13 kB per oversample (10: 0.13 MB) |
| `<stem>_particlize.h5` | MUSIC's freeze-out surface (one unit per event), the final partons, the initiators | default; not with `--no-particlize` | ~10 MB |
| `<stem>.h5` | MUSIC's evolution (e, vx, vy, vz) on `grid_oo.yaml`, `diag/` | default; not with `--no-hydro-file` | ~17 MB |
| `<stem>.json`, `.xml`, `.log` | the job summary (cross sections, skipped events), the job XML, the log | always | |

The usual production setting is **hadron files only** (`--bulk-hadrons 10 --no-particlize
--no-hydro-file`): 13,728 events/h on 2 × RTX 3090, 0.13 MB per event. The bulk can then not be
resampled later; keep the particlize files (setup B or C of the folder's README) when more
oversamples or other iSS settings may be needed.

**Downloading:** `utils/remote_transfer/js_osdf.py download <campaign> --what h5 --to DIR`
(public, no login; [`README_analysis.md`](README_analysis.md#getting-the-files-the-osdf-and-gcs)).
`--what h5` brings the hadron files, and the particlize files where there are any.

**Unit i of the jet file and unit i of the bulk file are the same event.** The hadron files
made in the job have no `source_uuid`, so `HadronFileReader` and `run_h5toROOT.py` (made for
the Au+Au pairs) do not read them; §5 and §10 have the O+O readers.

## 3. The per-event record

`units/` of the jet file (and of an in-job bulk file), `diag/` of the hydro file and `events/`
of the particlize file carry the same columns:

| column | meaning |
|---|---|
| `unit`, `event` | the unit (row), the event's number in the job (`events_skipped` leave gaps in `event`) |
| `pthat`, `pthat_bin` | the hard scattering's pT̂ and its window (`--pthat-bins`) |
| `sigma_gen`, `sigma_err` | Pythia's running cross-section estimate at that event [mb]; use the per-window values of the file attributes (§6) |
| `event_weight` | Pythia's event weight: 1 (unweighted generation) |
| `glauber_b`, `glauber_npart`, `glauber_ncoll`, `glauber_nstrings` | the geometry |
| `has_medium`, `hydro_row` | 1 if the event has a medium; its row in `<stem>.h5` (−1: none, or `--no-hydro-file`) |
| `isr_sub_{E,px,py,pz}_{proj,targ}`, `isr_sub_E_total` | what the ISR handed over for subtraction from the projectile (+z) and target (−z) nucleon |
| `remnant_E_{proj,targ}`, `remnant_pz_{proj,targ}` | the hard pair's remnants, which go to the jet hadronization |
| `strings_E_total`, `strings_pz_net` | what MUSIC's strings carry (NaN without a medium) |
| `parton_y_lead`, `parton_pt_lead`, `parton_y_sub`, `parton_pt_sub` | the two hardest partons of the hard process after the ISR |
| `tau0_music` | MUSIC's start time |
| `n_samples`, `seed`, `n_cells` | per file: the unit's samples, its seed, the surface cells (bulk file) |

Per event, exactly: `strings_E_total + isr_sub_E_total + remnant_E_proj + remnant_E_targ =
Npart √s_NN / 2` (with the subtraction; checked in §7).

## 4. Settings to check before combining files

| setting | why | where |
|---|---|---|
| `subtract_hard_momentum` | 1 (default) or 0 (`--no-subtraction`, a reference that counts the hard pair's energy twice) | attribute of every file; `.json` |
| the b range, √s | the `system` attribute, e.g. `O+O 200 GeV, b in [0.0, 3.0] fm` | attribute; job XML |
| MUSIC's late time step | on by default since `cu124-final_optim`: the freeze-out surface is 0.20 % larger and the bulk +0.05 % (+0.4 % ± 0.2 % at \|η\| < 1) against a constant step; one setting per campaign | `<late_delta_tau>` in the job XML (`prod_user_xml` of the hydro or particlize file, `<stem>.xml`) |
| MUSIC's grid | ±10.5 fm since 2026-10-12 (±15 fm before): the same physics within float rounding, no boundary hits | `<IS><grid_max_x>` in the job XML |
| Matter's lookup before t = 0 | X-SCAPE `8c975d37` (in the images of 2026-10-14, `cu1xx-20261013-9e84450`; not in `cu124-final_optim`): without it Matter read a garbage qhat for the ISR partons, which skipped events differently from run to run | the image or build; `events_skipped` in the attributes and `.json` |
| pT̂ windows, parton rapidity cut | the cross section per window; `--parton-ymax` keeps events with both hard partons at \|y\| < Y (before the ISR) | `pthat_bins`, `pthat_bin_sigma_gen`, `pthat_bin_acceptance`, `parton_ymax` attributes |
| bulk oversamples, hadron selection | `n_samples`; `--eta-max`, `--charged`, `--no-x`, `--keep-bits-*` | `n_samples`, `eta_max`, `charged_only`, `positions` attributes |
| jet colour tags (X-SCAPE `lbt_colored_hadro`) | before it, most events had duplicate colour tags (strung silently) and a failed fragmentation wrote the partons out as hadrons (`hard_sub_compare/` `sub/` seeds 0103, 0129); since, the jet hadrons differ: don't combine jobs from before and after. The jobs since have `units/colour_*` in the jet file; `<colour_closure>` must be the same across a campaign | `units/colour_partons` present or not; `<colour_closure>` in the job XML (`jet_hadronization_xml` attribute) |
| energy loss | Matter + LBT (the default since 2026-10-21) or Matter alone (`--matter-only`, and every job before): different campaigns | the `eloss` attribute (absent before 2026-10-21: Matter alone); `<Eloss><Lbt>` in the job XML |

```python
# the settings of every job of a campaign
import glob, h5py, hdf5plugin
for p in sorted(glob.glob("/path/to/OO_c1/**/*_hadrons_jet_frag.h5", recursive=True)):
    with h5py.File(p, "r") as f:
        A = f.attrs
        windows = A["pthat_bins"].tolist() if "pthat_bins" in A else "none"
        print(p.rsplit("/", 1)[-1], "|", A["system"], "| subtraction", A["subtract_hard_momentum"],
              "| skipped", A.get("events_skipped"), "| windows", windows)
```

The late step is recorded only in the job XML, which hadron-only jobs keep next to the files
(`<stem>.xml`): `grep late_delta_tau OO_c1/*.xml`.

## 5. Reading a campaign

The hadron files of an O+O campaign are small (0.13 MB per event at 10 oversamples), so a whole
campaign fits in memory as flat arrays: one event table and one array per hadron quantity, with
the global event index and the sample of each hadron *(tested)*:

```python
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

ev, bulk, jet = load("/path/to/OO_c1")
n_ev = len(ev["unit"])          # the test campaign: 2,991 events, 5.5 M bulk and 0.35 M jet hadrons, 1 s
```

- **Only finished jobs** (those with a `.json`) are read, so this also works while a
  campaign runs.
- **Without positions** (`--no-x`) the files have no `hadrons/x`; the loader doesn't read it.
- **Lower-level readers** from `jetscape.hadrons_h5`: `Hadrons.from_h5(path)` (one file in
  memory, with `pt`, `eta`, `phi`, `charged`, `unit`, `sample` and `hist`/`total` per sample)
  and `HadronFile(path).sample_event(unit, k)` (one sample from disk).

## 6. Weights and statistics

**Cross sections.** Each pT̂ window is its own Pythia; each file records per window
`pthat_bin_sigma_gen` [mb] (after the parton-rapidity cut: Pythia's σ × kept/tried) and
`pthat_bin_n_accepted`. Combine the files weighted by their accepted events, as
`HadronFileReader.pthat_bin_sigma` does for Au+Au *(tested)*:

```python
first = np.unique(ev["file"], return_index=True)[1]          # one row per file
sig, n_acc = ev["pthat_bin_sigma_gen"][first], ev["pthat_bin_n_accepted"][first]
sigma_k = (n_acc * sig).sum(0) / n_acc.sum(0)                 # mb per window
k = ev["pthat_bin"]
N_k = np.bincount(k, minlength=len(sigma_k))                  # events per window

w_hard = sigma_k[k] / N_k[k]             # sum over events = sigma of the hard process [mb]
SIGMA_NN = 42.0                          # mb, inelastic NN at 200 GeV
w_coll = ev["glauber_ncoll"] * sigma_k[k] / SIGMA_NN / N_k[k]   # per O+O collision in the b range
```

| weight | normalises to | use |
|---|---|---|
| none (per event, per window) | per hard event of that window | shapes and ratios within a window |
| `w_hard` | dσ/dX of the hard process [mb], summed over windows | spectra stitched over windows |
| `w_coll` | the yield per O+O collision in the b range, assuming the hard rate ∝ Ncoll σ_k/σ_NN^inel | comparisons with measured per-event yields; ratios to a p+p reference |
| `ev["glauber_ncoll"]` alone | the geometry of hard events (averages, profiles) | ⟨Npart⟩, multiplicities "for events with a jet" |

The Ncoll weight is needed because the events are hard-triggered with one hard scattering
whatever the geometry: unweighted, peripheral events are over-represented compared with where
hard scatterings happen. In the test campaign (0–80 %) it moves the bulk's dN_ch/dη at
\|η\| < 0.5 from 28.3 (unweighted) to 36.7 (Ncoll-weighted).

**Statistics.**
- **Events are independent:** there is no shared background, and every event runs the whole
  chain. Different windows are independent samples.
- **The bulk oversamples of one event sample one fluid:** average each event over its samples
  (`n_samples_bg`), then over events, and take errors from the spread of the per-event values.
- **The jet hadrons are one sample per event.** Combining them with each bulk oversample makes
  N correlated copies of the event: weight each by 1/N.
- `n_samples_bg` is 0 for events without a medium: their bulk is empty, not missing. Keep them
  (they are part of the sample) unless the analysis needs a medium.

## 7. Checks

*(tested)*

```python
E_beam = ev["glauber_npart"] * 200.0 / 2                     # Npart sqrt(s_NN) / 2
E_sum = (ev["strings_E_total"] + ev["isr_sub_E_total"]
         + ev["remnant_E_proj"] + ev["remnant_E_targ"])
med = ev["has_medium"] == 1                                  # no medium: strings_E_total is NaN
print(np.max(np.abs(E_sum[med] - E_beam[med]) / E_beam[med]))   # test campaign: 3e-9
print("without a medium:", 1 - med.mean())                   # 4.9 % in 0-80 %
```

- **Skipped events:** `events_skipped` in every file's attributes and the `.json` (with the
  error). A few per campaign were Matter's "minus log of sudakov negative" before X-SCAPE
  `8c975d37`; with it there should be none.
- **The b range:** `glauber_b` must stay inside the `system` attribute's range.
- **Windows:** `np.bincount(ev["pthat_bin"])` should be about equal (event i uses window
  i mod K).

## 8. Observables

### Multiplicity, spectra, ⟨pT⟩

Per event: the bulk's sample mean plus the event's jet hadrons *(tested)*:

```python
n_samp = ev["n_samples_bg"]
mid_b = bulk["charged"] & (np.abs(bulk["eta"]) < 0.5)
mid_j = jet["charged"] & (np.abs(jet["eta"]) < 0.5)
dn_bulk = np.bincount(bulk["event"][mid_b], minlength=n_ev) / np.maximum(n_samp, 1)
dn_jet = np.bincount(jet["event"][mid_j], minlength=n_ev)
dndeta = dn_bulk + dn_jet                                    # |eta| < 0.5: width 1
w = ev["glauber_ncoll"]
mean = np.average(dndeta, weights=w)                         # test campaign: 36.7
err = np.sqrt(np.sum(w**2 * (dndeta - mean)**2)) / w.sum()
```

A pT spectrum is the same with `np.histogram(..., weights=1/n_samp[event])` per hadron and the
event weights on top; `hard_sub_compare.ipynb` (`spectrum()`) has a complete, weighted
version with errors from the per-event spread.

### Centrality from the multiplicity

An experiment classifies events by a multiplicity, not by b. `hard_sub_compare.ipynb` §2 shows
how: an estimator from charged hadrons at 3.1 < \|η\| < 3.9 (like RHIC's beam-beam counters)
from **one** bulk sample (one collision's fluctuations) plus the jet hadrons; class boundaries
from a reference without a hard scattering's effect (the not-subtracted bulk, one sample per
event, unweighted); then each event classified by what an experiment would count. The
subtraction makes events with a hard scattering look more peripheral, the more so at small
Npart: compare classes defined this way, not by b, with data.

### Flow from the oversamples

Pairs of hadrons from **different** oversamples of one event correlate only through the fluid
(its flow and geometry): no resonance-decay or momentum-conservation pairs within one sample.
That gives v_n{2} of the events' own fluids, without that non-flow *(tested)*:

```python
def vn2(h, n, n_samples, mask):
    """v_n{2} from pairs of hadrons in different oversamples of the same event."""
    m = mask & (n_samples[h["event"]] >= 2)
    e, s, q = h["event"][m], h["sample"][m], np.exp(1j * n * h["phi"][m])
    key = e * 100000 + s                                     # (event, sample)
    uk, inv = np.unique(key, return_inverse=True)
    Qs = np.bincount(inv, weights=q.real) + 1j * np.bincount(inv, weights=q.imag)
    Ms = np.bincount(inv).astype(float)
    ev_of = uk // 100000
    Q = np.bincount(ev_of, weights=Qs.real) + 1j * np.bincount(ev_of, weights=Qs.imag)
    M = np.bincount(ev_of, weights=Ms)
    num = np.abs(Q)**2 - np.bincount(ev_of, weights=np.abs(Qs)**2)   # sum over a != b of Q_a Q_b*
    den = M**2 - np.bincount(ev_of, weights=Ms**2)                    # number of such pairs
    ok = den > 0
    return np.sqrt(num[ok].sum() / den[ok].sum())

sel = bulk["charged"] & (np.abs(bulk["eta"]) < 1) & (bulk["pt"] > 0.2) & (bulk["pt"] < 3)
v2 = vn2(bulk, 2, ev["n_samples_bg"], sel)       # test campaign (0-80 %, unweighted): 0.031
```

- The pairs are weighted by their number; for an Ncoll-weighted average, weight `num` and
  `den` per event.
- Ten oversamples give ~45 sample pairs per event, enough for v₂; v₃ needs more oversamples.
- A pseudorapidity gap between the two hadrons (as in data) is a mask on each sample's Q.

### Jets with FastJet

The full event at hadron level is one bulk oversample plus the jet hadrons. Clustering bulk
sample 0 with them *(tested)*:

```python
import fastjet as fj
JET_DEF = fj.JetDefinition(fj.antikt_algorithm, 0.4)

def pseudojets(h, idx):        # massless, as in README_analysis section 6
    return [fj.PseudoJet(px, py, pz, np.hypot(pt, pz))
            for px, py, pz, pt in zip(h["px"][idx], h["py"][idx], h["pz"][idx], h["pt"][idx])]

acc_b = bulk["charged"] & (np.abs(bulk["eta"]) < 1.5) & (bulk["pt"] > 0.15) & (bulk["sample"] == 0)
acc_j = jet["charged"] & (np.abs(jet["eta"]) < 1.5) & (jet["pt"] > 0.15)
by_b = np.split(np.flatnonzero(acc_b), np.searchsorted(bulk["event"][acc_b], np.arange(1, n_ev)))
by_j = np.split(np.flatnonzero(acc_j), np.searchsorted(jet["event"][acc_j], np.arange(1, n_ev)))
lead_pt = np.zeros(n_ev)
for e in range(n_ev):
    parts = pseudojets(bulk, by_b[e]) + pseudojets(jet, by_j[e])
    jets = fj.sorted_by_pt(fj.SelectorAbsRapMax(1.1)(
        fj.ClusterSequence(parts, JET_DEF).inclusive_jets(5.0)))
    lead_pt[e] = jets[0].pt() if jets else 0.0
```

(`bulk["event"]` is sorted within the loaded arrays, so `np.split` at the event boundaries
gives each event's hadrons.)

- **Background:** the O+O bulk is small; ρA subtraction works as in `jet_fastjet.ipynb` (areas
  with explicit ghosts, ρ from kT jets).
- **Using all oversamples:** cluster bulk sample k plus the jet hadrons for every k, weight
  each by 1/N: the jet hadrons are the same in all of them.
- **Relative to the hard partons:** the initiators of event (file i, unit u) are in the jet
  file: `HadronFile(path).initiators(u)` → `(K, 11)`, columns `shower, pid, pstat, px, py, pz,
  E, x, y, z, t`; the two hardest hard-process partons after the ISR are also in
  `parton_{y,pt}_{lead,sub}`.
- **A p+p reference** with the same chain (ISR, Matter in vacuum, iColored) is not part of the
  production. A user XML with `p` nuclei would give it (no string, so no medium, in most p+p
  events); untested.

## 9. The hydro and particlize files

With the default outputs (`--bulk-hadrons` optional, no `--no-hydro-file`):

- **`<stem>.h5`:** `arr` `(nevents, 4, 64, 64, 32, ntau)` on `grid_oo.yaml` (x, y ±9.84 fm and
  η_s ±4.84 at 0.3125; MUSIC's own grid ±10.5 fm × ±6), τ = 0.5 + k · 0.1 fm/c, `diag/` with
  the record of §3 and `hit_grid_boundary`. Only events with a medium have a row: an event's row
  is its `hydro_row`. Reading it is as for the Au+Au files
  ([`README_analysis.md`](README_analysis.md) §3); `oo_isr_overview.ipynb` draws e at three τ
  with the freeze-out contour and the dijet.
- **`<stem>_particlize.h5`:** `surface/bg` (one unit per event, the 32 iSS columns), the final
  partons and the initiators; `ParticlizeFile` reads it. It is the input for the bulk later:

```bash
# in the container or an X-SCAPE build; the jet hadrons were made in the job
../prod_AuAu_0_10_jet/run_hadronize.py /work/OO_c1 -j 8 --tags bulk_bg --oversample 100
# SLURM, one CPU task per stage-1 task directory
SIF=$PWD/xscape_prod.sif PROD=oo CAMPAIGN=OO_c1 sbatch --array=0-19 utils/slurm_hadronize.sh
```

`hadronize.py` refuses `jet_frag` on these files: Colorless would drop the hard partons
(pstat 1000) and overwrite the in-job jet hadrons.

## 10. ROOT

`run_h5toROOT.py` and `HadronFileReader.h` expect the Au+Au pair layout (it stops on O+O with
`KeyError: 'seed'`). For O+O, convert each tag with the single-file converter and write the
event record next to it *(tested)*:

```python
import glob, subprocess, sys
import h5py, hdf5plugin, numpy as np, uproot

CONV = "contribs/PyJetscape/example/prod_AuAu_0_10_jet/root_export/hadrons_to_root.py"
for jf in sorted(glob.glob("/path/to/OO_c1/**/*_hadrons_jet_frag.h5", recursive=True)):
    stem = jf[: -len("_hadrons_jet_frag.h5")]
    for tag in ("bulk_bg", "jet_frag"):                     # one ntuple per tag, one entry per sample
        subprocess.run([sys.executable, CONV, f"{stem}_hadrons_{tag}.h5",
                        "-o", f"{stem}_{tag}.root", "--no-x"], check=True)
    with h5py.File(jf, "r") as f:                           # the event record (jet file)
        ev = {k: f["units"][k][:] for k in f["units"]}
        win = {k: np.atleast_1d(f.attrs[k]) for k in ("pthat_bin_sigma_gen", "pthat_bin_n_accepted")
               if k in f.attrs}
    with h5py.File(f"{stem}_hadrons_bulk_bg.h5", "r") as f:  # unit i = event i in both files
        ev["n_samples_bg"] = f["units/n_samples"][:]
    ev["n_samples_jet"] = ev.pop("n_samples")
    with uproot.recreate(f"{stem}_events.root") as out:
        out["events"] = {k: v.astype(np.float64 if v.dtype.kind == "f" else np.int64)
                         for k, v in ev.items()}
        if win:
            out["windows"] = {k: v.astype(np.float64) for k, v in win.items()}
```

Per job: `<stem>_bulk_bg.root` and `<stem>_jet_frag.root` (RNTuple by default, `--format
ttree` for ROOT < 6.34; one entry per sample: `event`, `unit`, `sample`, `pid[n]`, `pstat[n]`,
`E/px/py/pz[n]`), and `<stem>_events.root` (TTrees `events`, one entry per unit in order, and
`windows`). The hadron entries' `unit` is the `events` row's `unit`.

**In RDataFrame**, a charged dN/dη of the bulk, each event the mean over its oversamples,
Ncoll-weighted *(tested: 22.54 at η = 0.1 for one file, the same as from the HDF5 file)*:

```cpp
// oo_dndeta.C:  root -l -b -q 'oo_dndeta.C+("/path/OO_c1/OO_isr_c1_0001")'
#include <ROOT/RDataFrame.hxx>
#include <ROOT/RVec.hxx>
#include <TFile.h>
#include <algorithm>
#include <cmath>
using ROOT::RVecD; using ROOT::RVecF; using ROOT::RVecI;

void oo_dndeta(const char *stem, const char *out = "oo_dndeta.root") {
  ROOT::RDataFrame ev("events", std::string(stem) + "_events.root");
  auto unit = *ev.Take<Long64_t>("unit");
  auto ncoll = *ev.Take<Long64_t>("glauber_ncoll");
  auto nsamp = *ev.Take<Long64_t>("n_samples_bg");
  std::vector<double> w(1 + *std::max_element(unit.begin(), unit.end()), 0.0);
  double wsum = 0;
  for (size_t i = 0; i < unit.size(); ++i)
    if (nsamp[i] > 0) { w[unit[i]] = ncoll[i] / double(nsamp[i]); wsum += ncoll[i]; }

  const std::vector<int> charged{211, 321, 2212, 3222, 3112, 3312, 3334, 11, 13};
  ROOT::RDataFrame bulk("bulk_bg", std::string(stem) + "_bulk_bg.root");
  auto h = bulk
    .Define("eta_ch", [&](const RVecI &pid, const RVecF &px, const RVecF &py, const RVecF &pz) {
        RVecD eta;
        for (size_t i = 0; i < pid.size(); ++i)
          if (std::find(charged.begin(), charged.end(), std::abs(pid[i])) != charged.end())
            eta.push_back(std::asinh(pz[i] / std::hypot(px[i], py[i])));
        return eta;
      }, {"pid", "px", "py", "pz"})
    .Define("w", [&](const RVecD &eta, int u) { return RVecD(eta.size(), w[u] / wsum); },
            {"eta_ch", "unit"})
    .Histo1D({"dndeta", ";#eta;dN_{ch}/d#eta per event", 48, -6, 6}, "eta_ch", "w");
  h->Scale(1.0, "width");
  TFile f(out, "RECREATE");
  h->Write();
}
```

For a whole campaign, loop over the stems (or chain the files) and sum the weights over all
events. FastJet in C++ takes the hadrons of an entry as in [`README_analysis.md`](README_analysis.md)
§9 (`fastjet::PseudoJet(px, py, pz, |p|)`).

## 11. Notebooks and references

| | what |
|---|---|
| [`prod_OO_isr/oo_isr_overview.ipynb`](../contribs/PyJetscape/example/prod_OO_isr/oo_isr_overview.ipynb) | one job: the medium at three τ with the freeze-out contour and the dijet; η, φ, pT and Δφ-to-the-leading-parton of bulk and jet hadrons (`OO_STEM`, `XSCAPE_BUILD` for the EoS) |
| [`prod_OO_isr/hard_sub_compare/hard_sub_compare.ipynb`](../contribs/PyJetscape/example/prod_OO_isr/hard_sub_compare/README.md) | two campaigns with and without the hard-momentum subtraction: the centrality bias from a multiplicity estimator, bulk and jet-hadron dN/dη, dE/dη and spectra, the energy vs the parton pT, a table per window; Ncoll-weighted (`HSC_DIR`) |
| [`prod_OO_isr/README.md`](../contribs/PyJetscape/example/prod_OO_isr/README.md) | running it, the options, the outputs, physics notes and limits |
| [`README_2stage.md`](README_2stage.md) §8 | the container, the launch settings, SLURM (`PROD=oo`) |
| [`REPORT_FINAL_OPTIM.md`](REPORT_FINAL_OPTIM.md) §5 | the O+O check of the late time step and the solver stall break (720 events) |
| [`REPORT_SUMMARY_CLUSTER.md`](REPORT_SUMMARY_CLUSTER.md) | O+O throughput per machine and per cluster GPU |

## 12. Troubleshooting

| symptom | cause and fix |
|---|---|
| `HadronFileReader`: `… was made from particlize file uuid '' …` | hadrons made in the job have no `source_uuid`: read them as in §5 |
| `run_h5toROOT.py`: `KeyError: 'seed'` | it is for the Au+Au pairs: §10 |
| `hadronize.py` refuses `jet_frag` | the O+O jet hadrons are made in the job: `--tags bulk_bg` |
| NaN in `strings_E_total` | an event without a medium (`has_medium` 0) |
| empty bulk unit (`n_samples` 0) | an event without a medium: keep it, its bulk is empty |
| a campaign's events per window differ much | skipped events (`events_skipped`), or jobs with a number of events that is not a multiple of the windows |
| bulk differences at the 0.2 % level between campaigns | late-step vs constant-step files, or MUSIC's grid (±15 vs ±10.5 fm, float rounding only): §4 |
| `OSError: Can't (synchronously) read data …` | `import hdf5plugin` before reading |
