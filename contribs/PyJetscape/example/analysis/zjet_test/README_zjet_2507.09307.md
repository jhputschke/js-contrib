# CMS 2507.09307 (Z–hadron Δφ / Δy): every cut, and its setting in the framework

[arXiv:2507.09307](https://arxiv.org/abs/2507.09307) (CMS-HIN-23-006, PLB 2025, 140120;
HEPData [ins2945894](https://www.hepdata.net/record/ins2945894)): the Δφ and Δy correlations of
charged hadrons with a Z boson in Pb+Pb and pp at 5.02 TeV. Below, for every cut, what the
**production** does (the job XML `../../config/PbPb_0_30_ZJet.xml` and the arguments of the
`utils/zjet/` launchers) and what the **analysis** applies offline (`../zjet_wake.py`,
`compare.py` here). ✓ = exact match. State of 2026-10-16 (the campaign set-up of
`utils/zjet/README.md`; this folder's test data were made before it, with the earlier windows and
the parton cut |y| < 1.9).

## 1. Collision system, data sample, centrality

| CMS 2507.09307 | Production | Analysis |
|---|---|---|
| Pb+Pb 2018, 1.67 nb⁻¹, √s_NN = 5.02 TeV | `<MCGlauber>` Pb+Pb, `sqrts 5020`; `<PythiaGun><eCM> 5020` | – |
| pp 2017, 301 pb⁻¹ (the reference) | `run_prod_vacuum.py`: the same PythiaGun and cuts, Matter in vacuum, Colorless, no medium | the vacuum files, the same selection and observable ✓ |
| Centrality from the HF energy percentiles: 0–30, 30–50, 50–90 % (HEPData also 0–90 %) | **0–30 % only, cut on b:** `b_max 8.57` fm (π b² = 0.3 × 7.7 b; the model's own σ_inel is 7.89 b, so it is its 0–29.3 %, 8.68 fm would be exact). X-SCAPE ignores `cenMin`/`cenMax`. Classes ordered by activity (string count, N_part) share 96–97 % of their events with it (`centrality_check.py`) | every event weighted by its `n_coll` (`zjet_wake.event_weights`), since Z production scales with N_coll; 30–50 and 50–90 % are not produced |
| Event selection: non-collision events removed; pp one primary vertex within 25 cm along the beam (pileup); Pb+Pb pileup < 0.01 | not applicable: one clean collision per event, at the Glauber vertex | – |
| Trigger: one muon p_T > 12 GeV, \|η\| < 2.4 | implied by the lepton cuts below | – |

## 2. The Z

| CMS 2507.09307 | Production | Analysis |
|---|---|---|
| Z → μ⁺μ⁻ only | Z → ee and μμ (`23:onIfAny = 11 13`) | `select(..., flavour=13)` for the μμ channel alone; the default keeps both (the same physics, twice the statistics) |
| γ* → μμ included in the signal | **pure Z** (`WeakZ0:gmZmode = 2`): no γ*, no interference | small inside 60–120 GeV; not modelled |
| Both muons p_T > 20 GeV | `--lepton-ptmin 20` (`<trigger>Z`, `<leptonPtMin>`): an event failing it is regenerated in PythiaGun before any shower or hydro | `lep_pt=20` ✓ |
| Muons \|η\| < 2.4 | `--lepton-etamax 2.5` (looser, so that ATLAS's and the electrons' 2.5 stay possible) | `lep_eta=2.4` ✓ |
| Opposite charge, 60 < m_μμ < 120 GeV | the Z's decay pair; no mass cut in the generator | `mass=(60, 120)` ✓ |
| 40 < p_T^Z < 350 GeV | no p_T^Z cut; **p̂T windows 20–30, 30–35, 35–45, 45–55, 55–70, 70–200 GeV** (4 jets per window per background) hold ~99 % of the p_T^Z > 40 cross section; p̂T > 200 GeV (~0.8 %) is not produced, so Z's at 200–350 GeV are under-represented | `pt=(40, 350)` ✓ (with that caveat) |
| \|y_Z\| < 2.4 | no Z rapidity cut | `ymax=2.4` ✓ |
| No requirement on the recoiling jet | **no parton cut** (`--parton-ymax` is not passed; it removed 18–19 % of these Z's, `parton_cut_check.py`) | none ✓ |
| The highest-p_T Z if there are several (0.2 % of the events) | one Z per event | – |
| Muon efficiency, trigger and acceptance corrections | not needed: particle-level truth | – |

The analysis cuts of this table are `zjet_wake.CMS_2507` = `dict(pt=(40, 350), ymax=2.4,
mass=(60, 120), lep_pt=20, lep_eta=2.4)`, used by `zjet_wake.select`.

## 3. The tracks

| CMS 2507.09307 | Production | Analysis |
|---|---|---|
| Tracks p_T^ch > 1 GeV | all final particles are stored (no `--charged`), from p_T = 0 | the track p_T classes `zjet_wake.PT_CLASSES = (1, 2, 4, 10)` ✓ |
| \|η^ch\| < 2.4 | stored \|η\| < 2.6 (`--eta-max 2.6`) | `zjet_wake.track_mask(2.4)`: \|η\| < 2.4 and charged ✓ |
| Track classes 1–2, 2–4, 4–10 GeV | – | `PT_CLASSES` ✓ |
| Tracks within ΔR < 0.0025 of a Z muon removed | the Z's two leptons are stored in `jet_frag` with their exact momenta | `zjet_wake.not_z_lepton`: removes exactly the two leptons (pid and momentum); stricter than the cone, which would also take a hadron next to a muon (negligible) |
| Tracks corrected to primary charged particles (efficiency, acceptance, fakes) | **particle level** (the "Read first" note of `docs/Plans/PLAN_ZJET_CMS10x.md`): strong and electromagnetic decays done; **η and Σ⁰ stable in the bulk** (iSS, SMASH list); **no weak decays** anywhere (Σ±, Ξ⁻, Ω⁻ are ~10 % of the bulk's charged particles above 1 GeV); e/μ from heavy-flavour decays in the jets (Pythia, τ₀ < 10 mm/c) | every charged final particle counts: the primary-particle definition except η in the bulk; charged hyperons counted as stable primaries |

## 4. The observable

| CMS 2507.09307 | Production | Analysis (`compare.py`, `Model.observables`; `zjet_wake.cms_dphi`) |
|---|---|---|
| Δφ_ch,Z = φ_Z − φ_ch | – | `zjet_wake.z_frame`: φ_h − φ_Z, the opposite sign, which the symmetrization removes ✓ |
| Δy_ch,Z = y_Z − η_ch (the Z's rapidity, the track's pseudorapidity) | – | η_h − y_Z ✓ (the same sign remark) |
| Δ⟨N_ch⟩ = S − B per Z: S the same-event pairs, B mixed events (the Z of one event with the tracks of another Z event of the sample, 10 mixings, no further matching) | S of the model is its **truth Z-correlated yield**: the paired wake ⟨`bulk_jet`⟩ − ⟨`bulk_bg`⟩ (each event against its own background, `--correlated`) + the jet fragments; the uncorrelated bulk cancels exactly. 40 iSS samples and 40 fragmentations per event | **Δφ:** B is flat in Δφ, so S − B = S minus its Δφ average. **Δy:** B = half (the Z side) of event i's y_Z folded with event j's lab-frame η distribution, over the pairs i ≠ j, the analogue of the mixing. **The experiment-like analysis**: `mix.py` (`../zjet_mixing.py`): the full events mixed with other Z events of the sample, every event a partner, CMS's 10 partners from 10 other hydro backgrounds, and a strict emulation (one Z per background, one oversample) for comparing errors with CMS's |
| ±Δφ (±Δy) merged and scaled by 0.5, then reflected for display | – | `0.5 (h + h[mirror])`, `cms_dphi` ✓ |
| Δy on the Z side only, \|Δφ\| < π/2 | – | `zside = \|Δφ\| < π/2` ✓ |
| Bins: Δφ π/10 over [−π/2, 3π/2); Δy 0.8 (Pb+Pb) / 0.4 (pp) over [−4, 4] | – | the same bins (Δy in 0.4, merged to 0.8 for Pb+Pb) ✓ |
| No unfolding (bins much wider than the resolution); errors stat ⊕ syst | – | errors per event ⊕ a jackknife over the hydro backgrounds (the events of one background are correlated); χ² over the 10 (Δφ) or 5 (Δy, Pb+Pb) independent bins with CMS stat ⊕ syst |

## 5. Not modelled, and why it matters little

- **The detector** (tracking efficiency and fakes, muon efficiency, trigger, vertexing): CMS
  corrects for it, so particle-level truth is the right comparison.
- **The particle definition** (§3 and the plan's "Read first" note): the η decay products of the
  bulk are missing (~1–3 % of its charged pions, mostly at low p_T); weak decays are not done, as
  in the primary-particle definition.
- **γ*** and the **hard process**: pure Z; Pythia's leading-order Z + parton with ISR and MPI
  (CMS's reference generator: MadGraph5_aMC@NLO).
- **Centrality**: on b, 0–30 % only, with N_coll weights, not HF-energy classes.
- **p_T^Z above ~200 GeV**: under-represented (p̂T up to 200 GeV).

## 6. How to run it

The comparison of this folder (`README.md`): `accumulate.py` (the histograms of the selection
above) and `compare.py` (the observable of §4 against the HEPData tables) on `data/`. The
campaigns' files are read the same way (`HadronFileReader` on a campaign directory).
