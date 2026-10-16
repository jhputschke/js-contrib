# Z+jet in Pb+Pb 5.02 TeV at 10× CMS's statistical precision (plan, 2026-10-15)

Goal: reproduce the CMS Z+jet and Z–hadron measurements with the full chain (Pythia Z+jet with
Glauber vertices, MUSIC4GPU hydro with the wake, iSS, jet hadronization), analysed **with the experiments'
own background subtraction** on full sampled events, at 10× their statistical precision in every
p_T^Z bin. Companion of `PLAN_CAMPAIGN_1M.md` (same settings, same tools).

## Status (2026-10-15)

Production code (§5a steps 1–7) on the `zjet` branches of X-SCAPE (off `contrib`) and js-contrib
(off `main`), tested on the GB10; image, benchmark and campaigns not yet. What changed against
the plan below, found while implementing:

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
- **Acceptances** (standalone Pythia, 4,000 events per window): Z (|y_Z| < 2.4, leptons
  p_T > 20, |η| < 2.4) 0.56 / 0.59 / 0.62 / 0.66 for the four windows; with the leading parton
  |y| < 1.9 as well 0.45 / 0.50 / 0.53 / 0.61.
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
  Z+jet XML. The geometry is drawn with P(b) ∝ b and every event gets one Z, but Z production
  scales with N_coll, so every event is weighted by its `n_coll`, which `run_prod_jet.py` now
  records per event (`diag/`, `events/`, `units/n_coll`; `HadronFileReader`'s
  `EventInfo.n_coll`). Centrality is geometric (b), not CMS's HF-energy percentiles.
- **The selection, decided 2026-10-15** (supersedes §2–§3 and §5b's numbers):
  both **decay leptons p_T > 20 GeV, |η| < 2.5** (the lepton cuts of the three wake papers:
  CMS 2103.04377, CMS 2507.09307, ATLAS 2008.09811; CMS Z+jet's 10–20 GeV muons are not
  generated); **no Z rapidity cut** in the generator and no p_T^Z cut (each paper's |y_Z|, p_T^Z
  and mass window offline from `units/z_*`); **p̂T windows 20–30, 30–40, 40–60, 60–80, 80–120,
  120–200** — from 40 GeV on, the windows held only 79 % of the p_T^Z > 40 and 53 % of the
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

## 2. Statistics: generated events → accepted pairs

10× precision = 100× the accepted sample per bin. Per generated event with p̂T in a window: Z in the
lepton acceptance (|y_Z| ≲ 2.4, lepton p_T) ~65 %; partner parton within |η| < 1.6 + R is free
(`--parton-ymax 1.9` regenerates events outside before any shower or hydro); jet above 30 GeV and
Δφ_jZ > 7π/8 ~60–70 % (R_jZ is itself a result; all events count for it). **≈ 0.45 accepted pairs per
generated event.**

| p̂T window (GeV) | CMS Z's (est.) | 100× (Z's) | events generated | Z's in acceptance | precision vs CMS |
|---|---|---|---|---|---|
| 40–60 | ~700 | 70k | 25k | ~16k | ~5× (10× needs +50k in this window) |
| 60–80 | ~150 | 15k | 25k | ~16k | ~10× |
| 80–120 | ~65 | 6.5k | 25k | ~16k | ~16× |
| 120–200 | ~20 | 2k | 25k | ~16k | ~30× |
| (30–40, optional) | – | – | 25k | – | for the p_T^Z > 30 Z–hadron selection |

**Baseline: 4 windows × 25k = 100k events on 5,000 backgrounds**, ≥ 10× CMS for everything above 60 GeV.
Add 50k in 40–60 if the R_jZ turn-on matters; add the 30–40 window for 2103.04377's selection.

Z–hadron side: 100× CMS's 2,000–5,000 Z's is 2–5 × 10⁵ sample-events of uncorrelated S − B. At 40 iSS
samples per event with the fluid-level floor (×2), 10–25k events with p_T^Z > 40 — inside the 100k.
Jet-level observables (x_jZ, R_jZ) are dominated by event-to-event shower and energy-loss variation,
which only events average; oversampling contributes the UE realizations for the jet reconstruction.

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
  ISR emissions; with MPI on the MPI partons nearly always satisfy it. Relax it to ≥ 1 for Z+jet anyway.
- **Layout:** `--pthat-bins 20-30,30-40,40-60,60-80,80-120,120-200 --jets-per-bin 4` (24 events
  per background), `--parton-ymax 1.9`, 96 events per job (4 backgrounds) → 1,000 jobs (2026-10-15;
  first planned: 4 windows from 40 GeV × 5 jets, 100-event jobs).
- **Output:** `--hadrons-only --oversample 40 --n-frag 40 --correlated
  --charged --eta-max 2.6 --no-x --hadron-keep-bits-p 12 --hadron-keep-bits-x 8 --keep-primaries 0`.
  Central Pb+Pb has ~2× the Au+Au multiplicity: ~7 MB/event → **~0.7 TB** for 4 M sampled full events.
  A 5k-event subset with `--write-particlize both` if hydro-level Z-jet wakes are wanted (~2 TB).
- **pp reference** (for I_AA, x_jZ^pp, R_jZ^pp): the same Pythia Z+jet with vacuum showers and
  Colorless hadronization, no hydro, no medium (10k events, minutes). A vacuum mode of
  `run_prod_jet.py` does not exist; it is a small driver (Pythia → Matter vacuum → Colorless).
- **Cost:** measured on the GB10 (Status, §6): 2.1× the Au+Au time and memory per event, so
  memory sets the jobs per GPU → ~250–350 events/h per A40 → **~300–400 A40-hours**; 5–7 days
  on the 3090 box (estimates, to be confirmed there).

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
| **6. Z acceptance pre-selection** | js-contrib `run_prod_jet.py`, next to `--parton-ymax` (events outside are regenerated before any shower or hydro) | `--trigger-ymax Y` (\|y_Z\| < Y), `--lepton-ptmin`, `--lepton-etamax`; CMS: 2.4 / 20 / 2.4 (ee 2.5) | the regeneration counter in the `.json`; the efficiency ≈ 0.65 for p̂T 60–80 |
| **7. Pb+Pb Z+jet user XML** | `example/config/PbPb_0_30_ZJet.xml` from `example/config/PbPb_0_10_jet.xml` | the Z+jet `<PythiaGun>` block (+ `<trigger>Z`); `eCM 5020`; grid and η range covering \|η\| < 2.6 (hadrons) and \|η_jet\| < 1.9 | `--dry-run`; the edge flags of a 15-event test job |
| **8. Vacuum (pp) driver** | js-contrib `prod_AuAu_0_10_jet/run_prod_vacuum.py` (new, ~150 lines) | PythiaGun → Matter in vacuum (`<Matter><in_vac>1`) → Colorless, no hydro, same hadron files (`jet_frag` + `initiators/` + Z columns) | R_jZ, ⟨x_jZ⟩ of 10k events against CMS pp |
| **9. Image (mandatory)** | a new `xscape-prod` build (CI `workflow_dispatch`, as the test images were built from branch heads) from **X-SCAPE `auau_hadrons_only`** (+ steps 1–3), **js-contrib `auau_hadrons_only`** (+ steps 4–6), **iSS `smash_ready`**, MUSIC4GPU main | **`--hadrons-only` exists only in that image.** The current production images (`cu124-20261013-9e84450` and earlier) do not have it, and it cannot be added from outside: it needs X-SCAPE C++ (`SoftParticlization::SampleCells`, `SetNextTaskNumber`), rebuilt PyJetscape bindings (`soft_sample_cells`, `set_next_task_number`, `xml_add_user_blocks`) and the `smash_ready` iSS, not just Python. One image, digest pinned, for all ten campaigns and the Au+Au phases 2–3 | `BUILD_INFO.txt` lists the four commits; `run_prod_jet.py --help` shows `--hadrons-only`; the 4-event identity check of `PLAN_AuAu_hadrons_only.md` (hadrons-only ≡ particlize-only + `hadronize.py --seed-uuid`) passes in the image |
| **10. Benchmark** | 3090 box, the XML of 7, QCD dijets or Z+jet at the six windows, 24 events/job (one background), 3 and 4 jobs/GPU (GB10: 14–16 GB per job), `--hadrons-only` | events/h, VmHWM, droplets per event, edge flags | confirms or replaces the GB10-based `JOBS_PER_GPU` / `P` (3) and the time estimate of §6 |

Steps 1–3 are X-SCAPE (one PR), 4–6 js-contrib (one PR), 7 configuration, 8 optional for the
first campaign, 9–10 before campaign 1. Nothing here touches MUSIC, iSS or the hydro writers, so the
Au+Au outputs stay bit-identical (check with one Au+Au seed).

## 5b. Campaign scheme: 10 × 9,600 events

The 96k events run as **ten identical campaigns of 9,600 events** (`zjet_c00` … `zjet_c09`), each
6 windows × 1,600 events on 400 backgrounds, 100 jobs × 96 events (4 backgrounds × 24; the
2026-10-15 layout, Status above; first planned as 4 windows × 2,500 on 500 backgrounds). A single
CMS-equivalent (~1,000 events) is one campaign with `NJOBS=1` per task: the smoke test. Campaigns differ only in
their seed range (`SEED_BASE = 1 + 1000 × c`), so any subset combines, and a bad campaign is redone
alone. Launchers in `utils/zjet/` (the step-7 XML is `example/config/PbPb_0_30_ZJet.xml`):

| file | what |
|---|---|
| `zjet_slurm_campaign.sh` | Delta A40: `C=3 sbatch zjet_slurm_campaign.sh` → one campaign on 10 A40s (array 0–9, 10 jobs × 96 events per task, `P=3`, 16 cores, 58 GB); ~3–4 h per task at ~250–350 events/h (estimate) |
| `zjet_3090_campaign.sh` | the 3090 box: `IMAGE=… ./zjet_3090_campaign.sh 3` → the same campaign via `launch_2gpu.sh`, 3 jobs/GPU, OMP 6; ~12–16 h (estimate) |
| `README.md` | the per-campaign checklist (dry run, `check_deposition.py`, upload) |

Both launchers pass the same `run_prod_jet.py` arguments (`$D` = the image's
`contribs/PyJetscape/example/config`): `--user-xml $D/PbPb_0_30_ZJet.xml
--grid $D/grid_PbPb.yaml --pthat-bins 20-30,30-40,40-60,60-80,80-120,120-200 --jets-per-bin 4
--parton-ymax 1.9 --lepton-ptmin 20 --lepton-etamax 2.5 --hadrons-only
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

**Measured on the GB10** (2026-10-15, [BENCHMARK_PbPb_vs_AuAu.md](../BENCHMARK_PbPb_vs_AuAu.md)):
Pb+Pb 5.02 TeV 0–30 % Z+jet against Au+Au 200 GeV 0–10 % jets, the same layout (24 jets per
background) and hadron settings, 2 jobs at once: **2.1× the time** (37 against 17 core-s per
event, 323 against 693 events/h), **14–16 GB per job** (Au+Au 6.8–7.1) and 1.7× the disk. The
memory is mostly MUSIC's larger box: x, y ±20 fm and η_s ±8 have 2.4× Au+Au's cells, and the
background's evolution is kept in memory on the whole grid; the same Au+Au job in the Pb+Pb box
needs 10.8 instead of 5.2 GB. The first estimate here (1.8–2× the Au+Au event, ~8–9 GB per job,
from multiplicity, lifetime and grid) got the time right and half the memory. **Host memory, not
cores or the GPU, now sets the jobs per GPU:**

| | 2 × RTX 3090 box | Delta A40 |
|---|---|---|
| jobs per GPU | 3 (6 × ~16 GB ≈ 96 GB of 125; 4 if the box's peaks are lower) | 3 (58 GB share / ~16 GB) |
| limit | host memory | host memory (the cores would allow more) |
| events/h | **~600–850** (estimate) | **~250–350 per A40** (estimate) |
| 96k events (10 campaigns) | **~115–160 h ≈ 5–7 days** | ~300–400 A40-hours |
| wall on 8 A40 (2 nodes) | – | ~35–50 h |
| wall on 32 A40 (8 nodes) | – | ~9–12 h (+ queue) |
| wall on 100 A40 (25 nodes) | – | ~3–4 h (+ queue) |
| cost | none; no queue | ~20–27k credits at the A100 rate (less with the A40 discount) |
| disk | 0.7 TB, fits `/data` (1.6 TB free) | 0.7 TB, no upload issue |
| pp reference (10k, no hydro) | < 1 h | – |
| stage 2 | none (hadrons-only) | none |

The rates assume ~25–35 s per event and job at 3 jobs per 3090 (the GB10's 19–22 s with 2 jobs,
plus the sharing) and ~30–45 s on an A40. **To be confirmed on the RTX box** (§5a step 10, the
launchers' defaults follow these numbers): VmHWM per job at 3 and 4 jobs per GPU (on discrete GPUs
the hydro's arrays are device memory, so the host peak may be lower than the GB10's; Au+Au peaks
the same on both), events/h by makespan; then `JOBS_PER_GPU`, `P` and this table.

**The 3090 box alone does the whole production in about a week of continuous running**, against
~2 days on 2 Delta nodes or half a day on 8+. For a 96k-event study the box is still the simpler
choice; Delta wins if the result is wanted within days or the box is busy with phases 1–2 of
`PLAN_CAMPAIGN_1M.md`.

**Memory is the lever:** keeping the background's evolution only where the jets go (|η_s| ≲ 4 for
partons with |y| < 1.9) or only above freeze-out could bring a job back toward Au+Au's and double
the jobs per GPU (BENCHMARK_PbPb_vs_AuAu.md, not tried).

## 7. Summary

100k Pb+Pb 5.02 TeV Z+jet events (4 p̂T windows × 25k, 5,000 backgrounds), hadrons-only at 40 samples
(4 M full events, ~0.7 TB), ~300–400 A40-hours or 5–7 days on the 3090 box alone (§6, estimates from the GB10
measurement, to be confirmed on the box); 10k pp-reference events. Gives ≥ 10× CMS's statistical
precision in every p_T^Z bin above 60 GeV for x_jZ, ⟨x_jZ⟩, R_jZ, and ≥ 10× for the Z–hadron Δφ/Δy/I_AA
with the experiments' mixed-event subtraction, with the paired truth on the same events. The production
side has the Z record and needs the RTX-box benchmark; the analysis side needs the mixed-event and
R = 0.3 subtraction scripts.
