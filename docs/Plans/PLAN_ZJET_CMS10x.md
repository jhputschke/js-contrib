# Z+jet in Pb+Pb 5.02 TeV beyond CMS's statistical precision (plan, 2026-10-15; layout 2026-10-16)

Goal: reproduce the CMS Z+jet and Z–hadron measurements with the full chain (Pythia Z+jet with
Glauber vertices, MUSIC4GPU hydro with the wake, iSS, jet hadronization), analysed **with the experiments'
own background subtraction** on full sampled events, beyond their statistical precision in every
p_T^Z bin. First aimed at 10× everywhere; **set up for the Z–hadron measurements** (2026-10-16:
no parton cut, windows for the most effective Z's), the 96k events (§2) give 5.2× CMS 2507.09307's
and 3.5× 2103.04377's precision by Z count (more with the 40 samples per event, ~50× with the
paired truth), and 9.9× for CMS Z+jet's p_T^Z > 60 (9–12× per bin above 60, 5–7× in 40–60).
Companion of `PLAN_CAMPAIGN_1M.md` (same settings, same tools).

## Status (2026-10-16)

Production code (§5a steps 1–7) on the `zjet` branches of X-SCAPE (off `contrib`) and js-contrib
(off `main`), tested on the GB10. **Image before the PRs** (changed order, 2026-10-16): the test
image `jhputschke/xscape-prod:cu124-zjet` is built from the `zjet` branches (`docker-prod.yml`
with the new `moving_tag=false`, so the moving `cu124` stays on the production image) and tested
on the RTX box with `utils/zjet/zjet_3090_test.sh` (step 10) before the PRs; benchmark and
campaigns not yet. What changed against the plan below, found while implementing:

- **The Z's leptons are not status 62.** Pythia attaches the resonance decay after the beam
  remnants: the Z is status −62 and its leptons are final with status **23**. PythiaGun takes
  every final-state e/μ (`isFinal()`), not the status-62 ones (step 1).
- **The leptons go the photons' way everywhere**, which is more than Matter and LBT: the
  `Parton` class refused every pid but quarks, gluons and photons, and `JetEnergyLoss` and
  `LiquefierBase` special-case photons. One helper `Parton::isNonParton` (γ, e, μ) now serves all
  of them; Colorless needs no change (they are colourless and come out in `jet_frag`) (step 3).
- **The rapidity cut and the Z acceptance are C++** (steps 4, 6): `--parton-ymax` only writes
  `<partonYMax>`, PythiaGun cuts in its event loop. The lepton skip of the leading parton and the
  new `<trigger>Z` with `<triggerYMax>`, `<leptonPtMin>`, `<leptonEtaMax>` are there, counted
  in the same tried/kept as `<partonYMax>` (σ = sigmaGen × the acceptance of both);
  `run_prod_jet.py --trigger-ymax / --lepton-ptmin / --lepton-etamax` write them.
  `<minPartons>` (default 2) counts quarks, gluons and photons, not leptons.
- **MPI at 5.02 TeV:** PythiaGun hands over **~15 partons above 2 GeV per event** (standalone
  Pythia, all four windows), not "several"; each is showered and can deposit. The same holds for
  QCD dijets at 5.02 TeV, so the benchmark measures it.
- **Acceptances** (the generator's, both leptons p_T > 20, |η| < 2.5; vacuum reference,
  2026-10-16): 0.53 / 0.57 / 0.57 / 0.60 / 0.62 / 0.61 for the six windows 20–30 … 70–200,
  σ × acceptance 46.6 / 14.6 / 18.6 / 11.4 / 9.82 / 11.3 pb (112.3 pb in all); every paper's own
  cuts come on top, offline (§2). (With the first windows and the parton cut: 0.39–0.60.)
- **Z–hadron focus (2026-10-16): no parton cut, other windows.** The leading-parton cut |y| < 1.9
  (for CMS Z+jet's jets in |η| < 1.6) removed 18–19 % of CMS 2507.09307's Z's, those whose recoil
  goes forward (jet and wake at large |Δy|); the Z–hadron measurements have no jet requirement.
  Without it the pp reference matches CMS better (vacuum, 4,000 selected Z's: Δφ χ² 35 → 20, Δy
  17 → 12 over the three p_T classes, 30 independent bins each; the 2–10 GeV jet-side peak was too
  high with the cut, so part of the pp mismatch of the GB10 test was the cut). Jet observables
  select |η_jet| < 1.6 offline. With equal events per window, the 6-window layout from 20 to 200 GeV
  with the most effective 2507.09307 Z's is **20–30, 30–35, 35–45, 45–55, 55–70, 70–200**
  (reweighting a vacuum sample by its events' p̂T, then confirmed with a vacuum run of these
  windows): 41.6k effective Z's against 32.3k with the first windows (both without the cut), +25 %
  for 2103.04377 and ATLAS. Launcher arguments only: no image rebuild.
- **Pb+Pb inputs were stale** (step 7): `PbPb_0_10_jet.xml` still had EOS 9 and
  `hadronize_5020.xml` predated smash_ready (no `decay_smash_list`, `keep_primaries`). The jet XML
  is remade with README_2stage §9's recipe from today's Au+Au files (EOS 91);
  `PbPb_0_30_ZJet.xml` is it with the Z+jet block. `hadronize_5020.xml` is gone: Colorless's
  `eCMforHadronization` (the beam remnant) is now the job's `<PythiaGun><eCM>` everywhere
  (`--hadrons-only`, `--validate-inline`, `hadronize.py` from the files' `prod_user_xml`, the
  vacuum driver), so one `hadronize.xml` serves every system. The launchers must pass
  `--grid grid_PbPb.yaml` (edge flags); the templates did not.
- **0–30 %, the CMS class** (all three CMS papers, checked 2026-10-15; only ATLAS has 0–10 %):
  `example/config/PbPb_0_30_ZJet.xml` (`b_max` 8.57 fm, π b² = 0.3 × 7.7 b) replaces the 0–10 %
  Z+jet XML. Checked 2026-10-16 with 200,000 minimum-bias 3dMCGlauber events
  (`analysis/zjet_test/centrality_check.py`): this model's own σ_inel is 7.89 b (3dMCGlauber, σ_NN 67.6 mb), not the 7.7 b behind b_max, so b < 8.57 fm is its 0–29.3 % (exactly 0–30 %: 8.68 fm); classes defined by event activity (the string count, which is 3dMCGlauber's own centrality cut, N_part, or a two-component N_part/N_coll mix) agree with the b class for 96–97 % of their events, with the same ⟨N_coll⟩ (1,030) and N_coll-weighted ⟨b⟩ (4.9 fm); X-SCAPE ignores `cenMin`/`cenMax`,
  so b is the only cut. The geometry is drawn with P(b) ∝ b and every event gets one Z, but Z production
  scales with N_coll, so every event is weighted by its `n_coll`, which `run_prod_jet.py` now
  records per event (`diag/`, `events/`, `units/n_coll`; `HadronFileReader`'s
  `EventInfo.n_coll`). Centrality is geometric (b), not CMS's HF-energy percentiles.
- **The selection, decided 2026-10-15** (§2–§3 and §5b updated to it on 2026-10-16):
  both **decay leptons p_T > 20 GeV, |η| < 2.5** (the lepton cuts of the three wake papers:
  CMS 2103.04377, CMS 2507.09307, ATLAS 2008.09811; CMS Z+jet's 10–20 GeV muons are not
  generated); **no Z rapidity cut** in the generator and no p_T^Z cut (each paper's |y_Z|, p_T^Z
  and mass window offline from `units/z_*`); **p̂T windows 20–30, 30–40, 40–60, 60–80, 80–120,
  120–200** (changed 2026-10-16, bullet above) — from 40 GeV on, the windows held only 79 % of the p_T^Z > 40 and 53 % of the
  p_T^Z > 30 cross section (p_T^Z ≠ p̂T event by event), from 20 GeV on 99.1 % and 97 %
  (standalone Pythia with the final selection; per-window table in `utils/zjet/README.md`); ATLAS's 15–30 GeV p_T^Z bin stays out (the Z + parton matrix element is
  not reliable at p̂T ≲ 20); **4 jets per window**: 24 events per background, 96-event jobs, 100
  jobs, 9,600 events per campaign, 96k in ten (16k per window, 400 backgrounds per campaign).
- **Correlated sampling needs equal counts** (measured 2026-10-15): with `--oversample-bg
  per-pthat-bin` (160 background samples against 40 per event) iSS sampled the background
  uncorrelated with the jets, so the launchers no longer pass it; with `--correlated`,
  `run_prod_jet.py --hadrons-only` now gives the background `--oversample` samples and
  refuses any other `--oversample-bg` (as `hadronize.py` does). Paired wake errors are ~1.8×
  smaller and the background's iSS time 4× lower.
- **All the productions' XMLs and grid YAMLs** are in `contribs/PyJetscape/example/config/` since
  2026-10-15 (`docs/PbPb_5020/` is gone): one user XML per setup, the shared `hadronize.xml`.
- **Cost measured on the GB10** (2026-10-15, [BENCHMARK_PbPb_vs_AuAu.md](../BENCHMARK_PbPb_vs_AuAu.md)):
  against an Au+Au jet job with the same layout and hadron settings, a Pb+Pb Z+jet job takes
  **2.1× the time** (323 against 693 events/h, 37 against 17 core-s per event), **2.1× the
  memory (14–16 GB per job**; §6 assumed 8–9) and 1.7× the disk. The memory is mostly MUSIC's
  larger box: the same Au+Au job in the Pb+Pb box needs 10.8 instead of 5.2 GB. The launchers
  now default to 3 jobs per 3090 and `P=3` per A40, and §6 is redone with these numbers;
  **all to be confirmed on the RTX box** (step 10).
- The templates are now `utils/zjet/` (launchers read the image's own `example/config/` files;
  the 3090 smoke test is `./zjet_3090_campaign.sh 0 10`).
- **Step 8 done:** `prod_AuAu_0_10_jet/run_prod_vacuum.py` takes the user XML's PythiaGun (through
  `run_prod_jet.configure_pythia_gun`, so the windows, cuts and trigger are the medium job's) and
  Matter with `<in_vac> 1`; Colorless as `--hadrons-only` (the same `jetscape.hadronizer` and seed
  scheme), eCMforHadronization from the PythiaGun's eCM. ~0.02 s per Z+jet event with 4
  fragmentations (GB10): 10k events in minutes. Per shower Matter conserves the energy exactly;
  the 3-momentum moves by a median 1.2 % (its virtuality kinematics, the same code as in Pb+Pb).

## 1. The measurements to reproduce

| | CMS Z+jet, PRL 119 082301 ([1702.01060](https://arxiv.org/abs/1702.01060)) | CMS Z–hadron ([2103.04377](https://arxiv.org/abs/2103.04377)) | CMS Z–hadron Δφ/Δy ([2507.09307](https://arxiv.org/abs/2507.09307)) | ATLAS Z–hadron I_AA ([2008.09811](https://arxiv.org/abs/2008.09811)) |
|---|---|---|---|---|
| data | 2015, 404 μb⁻¹ PbPb, 27.4 pb⁻¹ pp | 2018, 1.7 nb⁻¹ / 320 pb⁻¹ | 2018, 1.67 nb⁻¹ / 301 pb⁻¹ | 2018, 1.7 nb⁻¹ / 260 pb⁻¹ |
| centrality | 0–30 % | 0–30, 30–50, 50–70, 70–90 % | 0–30, 30–50, 50–90 % | 0–10, 10–30, 30–80 % |
| Z | ee, μμ; 70–110 GeV; **p_T^Z > 60** (Δφ, x_jZ); 40–120 for ⟨x_jZ⟩, R_jZ bins | ee, μμ; 60–120; **p_T^Z > 30** | μμ; 60–120; **40 < p_T^Z < 350** | ee, μμ; 76–106; **p_T^Z > 15** (15–30, 30–60, > 60) |
| jet / hadrons | anti-k_T **R = 0.3**, **p_T > 30**, **\|η\| < 1.6**, ΔR(ℓ) > 0.4, **Δφ_jZ > 7π/8** | tracks p_T > 1, \|η\| < 2.4 | p_T 1–2, 2–4, 4–10; \|η\| < 2.4; Δφ, Δy | p_T > 1, \|η\| < 2.5, **Δφ > 3π/4** |
| background | UE-subtracted jets | mixed MB events matched in HF energy | mixed Z events, 10 mixings, S − B | 40–160 MB events per Z, centrality to 0.1 %, **event plane to π/16** |
| statistics | **232 Z's** (p_T^Z > 60), ~170 pairs after Δφ cut | ~5,000 Z's | ~2,000 Z's | ~7,500 Z's |
| observables | Δφ_jZ, x_jZ = p_T^jet/p_T^Z, ⟨x_jZ⟩(p_T^Z), R_jZ(p_T^Z) | Δφ_hZ, ξ_T, p_T spectra per Z | Δφ, Δy per Z in p_T bins | Y(p_T^ch) per Z, I_AA |

All four are statistics-limited (statistical > systematic uncertainties throughout). CMS's Z's with
p_T^Z > 60 split roughly as 150 (60–80), 65 (80–120), 20 (> 120), and ~500 / 200 in the 40–50 / 50–60
R_jZ bins (from the ~p_T⁻⁴·⁵ spectrum).

## 2. Statistics: generated events → selected Z's (recomputed 2026-10-16)

**Layout:** 10 campaigns × 9,600 = **96,000 events**: 6 p̂T windows (20–30, 30–35, 35–45, 45–55,
55–70, 70–200 GeV) × 16,000, on **4,000 backgrounds** (24 events each: 6 windows × 4 jets), 1,000
jobs of 96 events. Every generated event already has both leptons p_T > 20 GeV, |η| < 2.5
(regenerated in PythiaGun before any shower or hydro); no parton cut. Per window, from the vacuum
reference with these settings (1,000 events per window, 2026-10-16):

| p̂T window [GeV] | σ × generator acceptance | generator acceptance | events (10 campaigns) | events passing 2507.09307 | share of the σ-weighted 2507.09307 sample |
|---|---|---|---|---|---|
| 20–30 | 46.6 pb | 0.53 | 16,000 | 4.7 % | 4.9 % |
| 30–35 | 14.6 pb | 0.57 | 16,000 | 19 % | 6.3 % |
| 35–45 | 18.6 pb | 0.57 | 16,000 | 52 % | 21.9 % |
| 45–55 | 11.4 pb | 0.60 | 16,000 | 86 % | 22.1 % |
| 55–70 | 9.82 pb | 0.62 | 16,000 | 95 % | 20.9 % |
| 70–200 | 11.3 pb | 0.61 | 16,000 | 95 % | 24.0 % |

Events weigh σ_window / N_window × N_coll; a weighted sample counts as N_eff = (Σw)² / Σw². The
2507.09307 sample is spread evenly over the windows, so the window weights cost only 13 %; the
N_coll weights another 15 % (Kish factor 0.85 for P(b) ∝ b, b < 8.57 fm; minimum-bias 3dMCGlauber,
`analysis/zjet_test/centrality_check.py`). Each paper's cuts offline (mass window, |y_Z|, lepton
|η| < 2.4 for CMS), 96k events:

| selection | Z's selected | N_eff (window weights) | N_eff (× N_coll) | CMS, 0–30 % | √(N_eff / N_CMS) |
|---|---|---|---|---|---|
| **2507.09307: 40 < p_T^Z < 350, \|y_Z\| < 2.4** | 56,300 | 49,000 | **41,600** | ~1,550 (78 % of ~2,000) | **5.2** |
| **2103.04377: p_T^Z > 30** | 75,000 | 57,100 | **48,400** | ~3,900 (78 % of ~5,000) | **3.5** |
| ATLAS 2008.09811 (76–106 GeV): 30–60 | 48,000 | 36,200 | 30,700 | (of ~7,500 in 0–80 %, not split) | – |
| > 60 | 27,900 | 27,400 | 23,200 | | – |
| 1702.01060 (70–110 GeV): p_T^Z 40–50 | 15,400 | 12,800 | 10,900 | ~500 | 4.7 |
| 50–60 | 12,900 | 11,600 | 9,900 | ~200 | 7.0 |
| 60–80 | 15,100 | 14,700 | 12,500 | ~150 | 9.1 |
| 80–120 | 8,800 | 8,700 | 7,400 | ~65 | 10.7 |
| > 120 | 3,400 | 3,400 | 2,900 | ~20 | 12.0 |
| > 60 (Δφ_jZ, x_jZ) | 27,300 | 26,900 | 22,800 | 232 | 9.9 |

(78 %: the share of all Z's in 0–30 % for N_coll scaling. CMS 2507.09307 uses μμ only; our
counts hold ee + μμ. The jet rows count Z's; without the parton cut ~18 % of them have the jet
outside |η| < 1.6, as in data.)

- **Z–hadron the experiments' way** (mixed events): **5.2× (2507.09307) and 3.5× (2103.04377)
  by Z count alone.** CMS's per-Z variance is mostly the uncorrelated bulk; every event here has 40
  independent samples of it, worth up to √40 more where the bulk dominates, down to the
  event-by-event fluid fluctuation (not measured; the first mixed-event analysis will).
- **Z–hadron truth** (the paired wake, correlated sampling): the GB10 test (`analysis/zjet_test/`,
  34 effective events, errors incl. a jackknife over its 4 backgrounds) already had half of CMS's
  errors at Δφ ≈ 0 for 1–2 GeV; scaled to 42k effective events, ~50× (extrapolated).
- **Jet observables** (Δφ_jZ, x_jZ, ⟨x_jZ⟩, R_jZ): ~√(N_eff / N_CMS), 9.9× for p_T^Z > 60, 9–12×
  per bin above 60, 5–7× in 40–60. (The first layout, 20–30 … 120–200 with the cut, gave 13–22×
  above 80 and 4–6× in 40–60, but 4.7× / 3.2× for the Z–hadron measurements.)
- **One campaign:** 4.2k effective 2507.09307 Z's (2.7× CMS's count), 2.3k with p_T^Z > 60 (10×
  CMS Z+jet's 232). The smoke test (10 jobs, 960 events): ~420 and ~230, about one CMS Z+jet
  equivalent at p_T^Z > 60 and a quarter of 2507.09307's.
- **More statistics where it pays:** 10× for 2507.09307 by Z count alone would take ~3.7× the
  events (~36 campaigns); the 40 samples per event may already give it (above).
- ATLAS's 15–30 GeV p_T^Z bin is not covered (p̂T from 20 GeV; the Z + parton matrix element is
  not reliable below). p̂T > 200 GeV (~0.8 % of p_T^Z > 40) is not produced.

## 3. Production setting

- **System:** Pb+Pb 5.02 TeV, `example/config/PbPb_*` inputs, 0–30 % (the CMS class; ATLAS has
  0–10 %): `b_max` 8.57 fm in `PbPb_0_30_ZJet.xml`, events weighted by `n_coll` (Status above).
- **Hard process (Pythia inside PythiaGun; no X-SCAPE ISR module):** `HardQCD:all = off`,
  `WeakBosonAndParton:qqbar2gmZg = on`, `WeakBosonAndParton:qg2gmZq = on`, `WeakZ0:gmZmode = 2` (pure Z;
  CMS 70–110 GeV window), `23:onMode = off`, `23:onIfAny = 11 13`; vertex from the Glauber collisions
  (the `--hard pythia` default); the lines go into the XML's `<LinesToRead>` (exists). **Pythia does
  the ISR, with PythiaGun's defaults unchanged: `PartonLevel:ISR = on`, `PartonLevel:MPI = on`,
  FSR off (Matter showers), hadronization off** — the settings of the Au+Au campaign. PythiaGun hands
  over every status-62 quark, gluon or photon with p_T > `softMomentumCutoff` = 2 GeV (hard process,
  ISR and MPI alike), all at the Glauber vertex. So p_T^Z ≠ p̂T event by event as in data, and the
  semi-hard MPI partons of the Z-producing nucleon–nucleon collision are showered too (several per
  event at 5.02 TeV, ~0 at 200 GeV): they are physical (the UE in Z events exceeds MB, which ATLAS
  subtracts in pp but which survives mixed-event subtraction in Pb+Pb), and they deposit a few GeV
  next to the jet's 20–40 GeV. MPI off would be a cleaner "one jet, one wake" sample but not the
  campaign's setting and not what data contain. PythiaIsrGun (the O+O ISR production) is not used.
  **PythiaGun keeps an event only with ≥ 2 accepted partons:** with the Z dropped (id 23 fails the
  parton filter) the hard process gives one, so with MPI off the sample would be biased toward hard
  ISR emissions; with MPI on the MPI partons nearly always satisfy it. Relaxed to ≥ 1 for Z+jet
  (`<minPartons>1` in `PbPb_0_30_ZJet.xml`, done).
- **Layout:** `--pthat-bins 20-30,30-35,35-45,45-55,55-70,70-200 --jets-per-bin 4` (24 events
  per background), no `--parton-ymax` (2026-10-16; first `20-30,30-40,40-60,60-80,80-120,120-200` with
  `--parton-ymax 1.9`), 96 events per job (4 backgrounds) → 100 jobs and 9,600
  events per campaign, 1,000 jobs and 96k events (4,000 backgrounds) in ten (2026-10-15; first
  planned: 4 windows from 40 GeV × 5 jets, 100-event jobs, 100k on 5,000 backgrounds).
- **Output:** `--hadrons-only --oversample 40 --n-frag 40 --correlated
  --charged --eta-max 2.6 --no-x --hadron-keep-bits-p 12 --hadron-keep-bits-x 8 --keep-primaries 0`.
  Disk: 9.1 MB per event in the GB10 test (|η| < 4, positions kept at full precision, 10
  fragmentations; 1.7× Au+Au); with these arguments (|η| < 2.6, no positions, 12-bit momenta,
  40 fragmentations) **1.6 MB per event, 10.3 bytes per hadron** (measured on the GB10, 2026-10-16:
  one 24-event job, a background with N_coll 443, 3,650 charged hadrons per sample; 14 s per event
  and 10.6 GB alone) → ~2 MB per event over 0–30 % → **~0.2 TB** for the 96k events (3.84 M
  sampled full events).
  A 5k-event subset with `--write-particlize both` if hydro-level Z-jet wakes are wanted (~2 TB).
- **pp reference** (for I_AA, x_jZ^pp, R_jZ^pp): the same Pythia Z+jet with vacuum showers and
  Colorless hadronization, no hydro, no medium (10k events, minutes). A vacuum mode of
  `run_prod_jet.py` does not exist; it is a small driver (Pythia → Matter vacuum → Colorless).
- **Cost:** measured on the GB10 (Status, §6): 2.1× the Au+Au time and memory per event, so
  memory sets the jobs per GPU → ~500–720 events/h on the 3090 box (**6–8 days** for 96k events,
  ~16 h per campaign) or ~250–400 per A40 (~240–380 A40-hours); estimates, to be confirmed on
  the box.

## 4. Analysis chain (experiment-like)

Each oversample k of an event is one full Pb+Pb event: `bulk_jet` sample k (medium + wake) + `jet_frag`
fragmentation k, with the Z four-momentum from the event record.

**Z+jet (CMS 1702.01060):**
1. anti-k_T R = 0.3 on all charged + neutral hadrons of the sample (|η| < 2.6 stored; CMS uses
   calorimeter jets), **UE subtraction**: the ρA-median k_T method of `README_analysis` §6 adapted to
   R = 0.3, or CMS's iterative subtraction; validate ⟨δp_T⟩ = 0 by embedding into untouched `bulk_bg`
   samples (the existing embedding test).
2. jets with p_T > 30, |η| < 1.6, ΔR(jet, lepton) > 0.4 (leptons from the Z decay are in the record),
   pair with the Z: Δφ_jZ, x_jZ for Δφ_jZ > 7π/8, ⟨x_jZ⟩ and R_jZ in p_T^Z bins 40–50, 50–60, 60–80, 80–120.
3. errors: samples of one event share its shower and fluid; average over samples first, then over
   events, errors per event (clustered by background for the shared-background correlation).
4. the same on the pp reference; and on the *paired* medium-free jet (the jet's own partons with
   `--correlated` background subtracted) for the subtraction bias.

**Z–hadron (CMS 2103.04377, 2507.09307; ATLAS 2008.09811):**
1. S: hadrons with p_T > 1 GeV, |η| < 2.4 (2.5), per Z: Δφ, Δy, ξ_T, p_T spectra, Δφ > 3π/4 for I_AA.
2. B: mixed events — other backgrounds' `bulk_bg` samples matched in N_ch (the HF-energy match) and in
   Ψ₂ (ATLAS: π/16), 10–160 mixings; or other samples of the same background (ideal matching, to
   measure what the matching costs).
3. S − B per Z, ⟨N_ch⟩ subtracted (CMS 2507.09307), I_AA against the pp reference (ATLAS).
4. truth on the same events: paired `bulk_jet` − `bulk_bg` with `--correlated`; the difference to the
   mixed-event result is the method's bias on the wake, the hole and the Δy dip.

The Z leptons are in `jet_frag` with pid ±11/±13 (and pass `--charged`): every track and jet
analysis removes them first (the event record's `lep{1,2}_*` identify them).

Scripts to write: mixed-event builder with N_ch/Ψ₂ matching, R = 0.3 UE-subtracted jet finder on the
hadron files, Z-pairing. Existing: `HadronFileReader`, `jet_minus_background`, ρA subtraction,
embedding, `wake_hadrons.py`.

## 5a. Production-side code changes, in order (what has to exist before job 1)

What exists and is reused: `<LinesToRead>` in the user XML (the Pythia process lines are
configuration); PythiaGun's photon path (id 22 is accepted, Matter rejects it and passes it to the
final list, `Matter.cc:300–313`); the `initiators/` record of every PythiaGun parton per event in the
hadron and particlize files (`INITIATOR_COLUMNS`); `--user-xml`, `--pthat-bins`, `--jets-per-bin`,
`--parton-ymax`, `--hadrons-only`; `example/config/PbPb_0_10_jet.xml` (README_2stage §9).

| step | file | change | check |
|---|---|---|---|
| **1. Carry the Z as its two leptons** | X-SCAPE `src/initialstate/PythiaGun.cc` (`Exec`, the status-62 filter) | accept `\|id\| == 11, 13` next to 21/22 and quarks; count them in `p62` but **not** against the `softMomentumCutoff` (they are not partons) | a 1-event dry run prints two leptons among the initiators |
| **2. ≥ 2-parton requirement** | same, `if (p62.size() < 2) continue;` | `< 1`, or an XML `<minPartons>` (default 2, 1 for Z+jet) | an event with one hard parton and no MPI/ISR parton above 2 GeV is kept |
| **3. Leptons through the energy loss untouched** | X-SCAPE `src/jet/Matter.cc:301` and `src/jet/LBT.cc:337` (both `pid == photonid`: set status 22, push to `pOut`, return) | one helper `is_non_parton(pid)` for {22, 11, 13} used by both guards and by step 1's filter. **No change in the hadronization:** `ColorlessHadronization.cc:229` already passes every status-22 particle into the hadron list, so the leptons come out in `jet_frag` with their pid and four-momentum, exactly as prompt photons do today (γ-jet already works end-to-end in the current image) | lepton four-momenta identical in and out; no droplets from leptons; two leptons per event in `jet_frag` |
| **4. Leptons are not partons downstream** | js-contrib `run_prod_jet.py`: the `--parton-ymax` / `--parton-y-mode leading` selection (`check_parton_ycut`, the leading-parton choice); the liquefier's parton list; `hadronizer` (`fragments`) | skip `\|id\| ∈ {11, 13, 22}` when choosing the leading parton, when building the droplet sources and when fragmenting (Colorless must not see leptons) | `droplets_total` and `E_dep` unchanged against a run where the leptons are removed by hand |
| **5. The Z in the event record** | js-contrib `run_prod_jet.py` (the per-event record → `units/` and `events/`), `hadrons_h5.py` | add `z_pt, z_y, z_phi, z_m` and `lep_*` columns computed from the two lepton initiators (pdg ±11/±13 pair with the largest invariant mass if several); attribute `trigger = "Z"` | `HadronFileReader` returns the Z per event; `z_pt` ≈ p_T of the recoil system |
| **6. Z acceptance pre-selection** | js-contrib `run_prod_jet.py`, next to `--parton-ymax` (events outside are regenerated before any shower or hydro) | `--trigger-ymax Y` (\|y_Z\| < Y), `--lepton-ptmin`, `--lepton-etamax`. Decided: no y_Z cut, leptons 20 GeV / 2.5 (every paper's cuts offline, §2) | the regeneration counter in the `.json`; the acceptance 0.53–0.62 per window (no parton cut) |
| **7. Pb+Pb Z+jet user XML** | `example/config/PbPb_0_30_ZJet.xml` from `example/config/PbPb_0_10_jet.xml` | the Z+jet `<PythiaGun>` block (+ `<trigger>Z`); `eCM 5020`; grid and η range covering \|η\| < 2.6 (hadrons) and \|η_jet\| < 1.9 | `--dry-run`; the edge flags of a 15-event test job |
| **8. Vacuum (pp) driver** | js-contrib `prod_AuAu_0_10_jet/run_prod_vacuum.py` (new, ~150 lines) | PythiaGun → Matter in vacuum (`<Matter><in_vac>1`) → Colorless, no hydro, same hadron files (`jet_frag` + `initiators/` + Z columns) | R_jZ, ⟨x_jZ⟩ of 10k events against CMS pp |
| **9. Image (mandatory)** | before the PRs the test image `jhputschke/xscape-prod:cu124-zjet` (`docker-prod.yml` from X-SCAPE `zjet` and js-contrib `zjet`, MUSIC4GPU XSCAPE; `moving_tag=false`); after the PRs the same from `contrib` / `main` (smash_ready and auau_hadrons_only are merged there) | **`--hadrons-only`, the Z trigger and the `TAG_PREFIX` naming exist only in such an image**, and cannot be added from outside: X-SCAPE C++ (`SoftParticlization::SampleCells`, `SetNextTaskNumber`, PythiaGun's trigger), rebuilt PyJetscape bindings and the `smash_ready` iSS, not just Python. One image, digest pinned, for all ten campaigns | `zjet_3090_test.sh info`: `BUILD_INFO.txt`, `--help` shows `--trigger`, the dry run, the `TAG_PREFIX` checks |
| **10. Benchmark** | 3090 box, the XML of 7, QCD dijets or Z+jet at the six windows, 24 events/job (one background), 3 and 4 jobs/GPU (GB10: 14–16 GB per job), `--hadrons-only` | events/h, VmHWM, droplets per event, edge flags | confirms or replaces the GB10-based `JOBS_PER_GPU` / `P` (3) and the time estimate of §6 |

Steps 1–3 are X-SCAPE (one PR), 4–6 js-contrib (one PR), 7 configuration, 8 optional for the
first campaign, 9–10 before campaign 1. Nothing here touches MUSIC, iSS or the hydro writers, so the
Au+Au outputs stay bit-identical (check with one Au+Au seed).

## 5b. Campaign scheme: 10 × 9,600 events

The 96k events run as **ten identical campaigns of 9,600 events** (`zjet_c00` … `zjet_c09`), each
6 windows × 1,600 events on 400 backgrounds, 100 jobs × 96 events (4 backgrounds × 24; the
2026-10-15 layout, Status above; first planned as 4 windows × 2,500 on 500 backgrounds). A single
CMS Z+jet equivalent at p_T^Z > 60 (~1,000 events, §2) is one campaign with `NJOBS=1` per task: the smoke test. Campaigns differ only in
their seed range (`SEED_BASE = 1 + 1000 × c`), so any subset combines, and a bad campaign is redone
alone. Launchers in `utils/zjet/` (the step-7 XML is `example/config/PbPb_0_30_ZJet.xml`):

| file | what |
|---|---|
| `zjet_slurm_campaign.sh` | Delta A40: `C=3 sbatch zjet_slurm_campaign.sh` → one campaign on 10 A40s (array 0–9, 10 jobs × 96 events per task, `P=3`, 16 cores, 58 GB); ~3–4.5 h per task at ~250–400 events/h (estimate) |
| `zjet_3090_campaign.sh` | the 3090 box: `IMAGE=… ./zjet_3090_campaign.sh 3` → the same campaign via `launch_2gpu.sh`, 3 jobs/GPU, OMP 6; ~14–19 h (estimate) |
| `zjet_3090_test.sh` | the box test before campaign 0: `info`, `bench 3` / `bench 4`, `report`, `smoke` |
| `README.md` | the per-campaign checklist (dry run, `check_deposition.py`, upload) |

The launchers run from a checkout of js-contrib (they use its `launch_2gpu.sh` /
`slurm_prod_array.sh`) and set `TAG_PREFIX=PbPb_0_30_seed`: the files are
`PbPb_0_30_zjet_cNN_seedNNNN.*` (box) and `PbPb_0_30_zjet_cNN_tTTT_seedNNNN.*` (Delta), not
`AuAu_0_10_jet_*` (`run_jobs.sh` passes every job its name as `--out`; js-contrib 2026-10-16).
Both launchers pass the same `run_prod_jet.py` arguments (`$D` = the image's
`contribs/PyJetscape/example/config`): `--user-xml $D/PbPb_0_30_ZJet.xml
--grid $D/grid_PbPb.yaml --pthat-bins 20-30,30-35,35-45,45-55,55-70,70-200 --jets-per-bin 4
--lepton-ptmin 20 --lepton-etamax 2.5 --hadrons-only
--oversample 40 --n-frag 40 --correlated --charged --eta-max 2.6 --no-x
--hadron-keep-bits-p 12 --hadron-keep-bits-x 8 --keep-primaries 0`.
**The container decides which mode runs.** With the step-9 image, `--hadrons-only` as above. With any
current image (no `--hadrons-only`, but the Z+jet steps 1–6 would also be missing, so this is only the
benchmark/QCD-dijet case): `--write-particlize both --particlize-only` in its place and stage 2
afterwards with the same hadronization arguments; the campaign scheme is unchanged, the disk is
~100 MB/event instead of ~7 MB until stage 2 and deletion.

## 5c. Analysis-side prerequisites (not production)

| item | what |
|---|---|
| mixed-event builder | `bulk_bg` samples of other backgrounds matched in N_ch (HF energy) and Ψ₂ (ATLAS: π/16), 10–160 mixings; or other samples of the same background (ideal matching) |
| R = 0.3 UE-subtracted jets on the hadron files | ρA-median k_T method (`README_analysis` §6) at R = 0.3, or a CMS-like iterative subtraction; embedding validation ⟨δp_T⟩ = 0 |
| Z pairing and observables | Δφ_jZ, x_jZ, ⟨x_jZ⟩(p_T^Z), R_jZ; Δφ_hZ, Δy, ξ_T, I_AA; errors per event, clustered by background |
| sanity | R_jZ and ⟨x_jZ⟩ of the pp reference against CMS pp; Δφ_jZ shape (CMS: KS p > 0.4 between pp and PbPb) |

## 6. Time: 2 × RTX 3090 box alone vs Delta A40s

**Estimated from two measurements, to be confirmed on the RTX box** (§5a step 10): the
Pb+Pb-to-Au+Au ratios measured on the GB10, applied to the Au+Au rates measured on the box.

### Inputs

| input | value | source |
|---|---|---|
| Pb+Pb Z+jet / Au+Au jets, same layout and hadron settings, 2 jobs at once | **2.1× the wall time, 2.2× the CPU** (37 / 17 core-s per event), **2.1× the memory** (14–16 / 6.8–7.1 GB per job), 1.7× the disk | GB10, 2026-10-15, [BENCHMARK_PbPb_vs_AuAu.md](../BENCHMARK_PbPb_vs_AuAu.md) |
| why the memory | MUSIC's box: x, y ±20 fm, η_s ±8 = 2.4× Au+Au's cells, the background's evolution kept on the whole grid; the same Au+Au job in the Pb+Pb box: 5.2 → 10.8 GB | same |
| Pb+Pb job on the GB10 | 19–22 s per event (wall, 96-event average incl. start and hadronization), ~1.8 cores busy per job | same |
| Au+Au on the box (`cu124-final_optim`, 20 jobs, campaign setting with particlize) | 25.4 s per event and job, 35.4 core-s, 2.09 GPU-s per event; VmHWM 6.0 [6.7] GB; ~1 GB VRAM per job; host-bound (21 of 24 cores), 2,160 events/h | [BENCHMARK_2x3090.md](../BENCHMARK_2x3090.md) |
| box against GB10, CPU per Au+Au event | ~1.5–1.9× (box ≈ 32 core-s without the pair-file compression, with hadronization; GB10 16.7) | the two above |

### Pb+Pb on the box

- **Per event:** ~55–70 core-s (2.2 × 32 at the box's speed; the first estimate here said 60–70),
  ~5 GPU-s (2.09 × 2.4 cells), ~2.4 GB VRAM per job.
- **Per job:** ~30–42 s per event (the GB10's 19–22 s × 1.5–1.9); a 96-event job takes ~50–67 min.
- **Memory sets the jobs** (14–16 GB each, the box 125 GB with ~10 GB for the rest):

| limit | jobs or events/h |
|---|---|
| host memory | **6 jobs (3 per GPU)**, ~96 GB; 8 (4 per GPU, 112–128 GB) only if the box's peaks are ≤ ~13.5 GB |
| CPU (21 of 24 cores, as Au+Au) | ~1,100–1,350 events/h: not reached by 6–8 jobs (~11–15 cores busy) |
| GPU (~65 % busy, as Au+Au) | ~950 events/h: not reached by 6 jobs (~45 % busy) |
| **→ events/h** | **6 × 3600 / 30–42 s ≈ 500–720** (central ~600); 4 per GPU: ~650–900 |

**A campaign** (9,600 events = 100 jobs × 96, 50 per GPU, 3 at a time = 17 rounds of one job):
**~14–19 h**, so one campaign (§2: 2.7× CMS 2507.09307's Z count, 10× CMS Z+jet's p_T^Z > 60) is an
overnight-plus run. The
smoke test (10 jobs, ~1,000 events) takes ~2 h.

### Totals

| | 2 × RTX 3090 box | Delta A40 (not measured; EPYC cores ~ the box's or faster) |
|---|---|---|
| jobs per GPU | 3 (memory) | `P=3` (58 GB share / ~16 GB) |
| events/h | **~500–720** (4 per GPU: ~650–900) | ~250–400 per A40 (3 × 3600 / 27–42 s) |
| one campaign (9,600) | **~14–19 h** | ~3–4.5 h per task (10 jobs: 4 rounds at `P=3`; 9 or 12 jobs per task would avoid the 1-job round) |
| 96k events (10 campaigns) | **~140–190 h ≈ 6–8 days** (4 per GPU: ~5–6 days) | ~240–380 A40-hours |
| wall on 8 / 32 / 100 A40 | – | ~30–48 h / ~8–12 h / ~2.5–4 h (+ queue) |
| cost | none; no queue | ~16–25k credits at the A100 rate (less with the A40 discount) |
| disk | ~0.2 TB (§3), fits `/data` (1.6 TB free) | same, no upload issue |
| pp reference (10k, no hydro) | < 1 h | – |
| stage 2 | none (hadrons-only) | none |

### Is the RTX box still feasible?

**Yes: about a week of continuous running for all ten campaigns** (6–8 days; the first
estimate, with 8–9 GB per job, said 4–6), and ~16 h per campaign, so campaigns can be run one
by one between other work and combined later. Delta does it in 1–2 days on 2 nodes or half a
day on 8+. The box stays the simpler choice unless the result is wanted within days or the box
is busy with phases 1–2 of `PLAN_CAMPAIGN_1M.md`.

Ways to shorten it:

- **4 jobs per GPU**, if the box's peaks are lower than the GB10's (on discrete GPUs the hydro's
  arrays are device memory; Au+Au peaks the same on both machines): ~5–6 days.
- **Less memory per job** (BENCHMARK_PbPb_vs_AuAu.md, not tried): keeping the background's
  evolution only where the jets go (|η_s| ≲ 4 for partons with |y| < 1.9) or only above freeze-out
  could bring a job toward ~8–10 GB, 5–6 per GPU, and the box toward its GPU/CPU limit of
  ~950–1,100 events/h: ~4 days.
- **The GB10 alongside** (323 events/h measured with 2 jobs): ~4–5 days together, if both run the
  same commits (the campaigns' seeds are disjoint, so their files combine).

**To be confirmed on the RTX box** (§5a step 10): VmHWM per job and events/h by makespan at 3 and
4 jobs per GPU with the campaign arguments; then `JOBS_PER_GPU`, `P` and these tables.

## 7. Summary

96k Pb+Pb 5.02 TeV 0–30 % Z+jet events in ten campaigns of 9,600 (6 p̂T windows × 16k, 4,000
backgrounds), hadrons-only at 40 samples (3.8 M full events, ~0.2 TB), ~240–380 A40-hours or
6–8 days on the 3090 box alone, ~16 h per campaign (§6, estimates from the GB10 and box
measurements, to be confirmed on the box); 10k pp-reference events. Set up for the Z–hadron
measurements (no parton cut, windows 20–30, 30–35, 35–45, 45–55, 55–70, 70–200), it gives (§2)
5.2× CMS 2507.09307's and 3.5× 2103.04377's statistical precision by Z count for the Z–hadron
Δφ/Δy/I_AA with the experiments' mixed-event subtraction (more with the 40 samples per event), the
paired truth on the same events (~50×), and 9.9× for x_jZ, ⟨x_jZ⟩, R_jZ at p_T^Z > 60 (9–12× per
bin above 60, 5–7× in 40–60). The production
side has the Z record and needs the RTX-box benchmark; the analysis side needs the mixed-event and
R = 0.3 subtraction scripts.
