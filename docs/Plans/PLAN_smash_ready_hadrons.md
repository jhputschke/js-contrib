<!-- Plan, written 2026-10-14. Status: not started. Branches to use: smash_ready in the iSS fork (from iSS_optim 98deaff), X-SCAPE (from contrib) and js-contrib (from main). -->

# Plan: EOS 91 hadrons, and undecayed primaries for a later SMASH afterburner (Au+Au and O+O)

## Goal

1. **EOS 91 in production.** MUSIC with EOS 91 (hotQCD + the SMASH hadron list) and iSS
   decaying that list itself, so an EOS 91 campaign gives decayed hadrons like EOS 9 does today.
2. **Bulk primaries.** iSS also keeps the undecayed hadrons (with positions), so SMASH can run on
   them later, without a new hydro or particlization.
3. **Jet primaries with positions.** The jet hadronization keeps its undecayed hadrons as well,
   and gives every jet hadron a space-time position, so the jet hadrons can go into SMASH too.

For **Au+Au and O+O**, in every place where the hadronization runs (table in
[Where it applies](#where-it-applies)). Part 1 is the bulk (goals 1 and 2), Part 2 the jet hadrons
(goal 3).

Not in this plan: running SMASH (the converter, the runs, the analysis), and a hadrons-only
mode for Au+Au ([Follow-ups](#follow-ups)).

## What the code does today

**MUSIC.** EOS 9 and 91 use the same code (`EOS_hotQCD`, `music4gpu/src/eos_hotQCD.cpp:44–49`),
only the table differs. Both are in the image (`get_music4gpu.sh` downloads `binary` and
`SMASH_binary`). All production XMLs run 9:
`prod_AuAu_0_10_jet/AuAu_MCGlauber_MUSIC_0_10_jet.xml:207`,
`prod_AuAu_0_10/AuAu_MCGlauber_MUSIC_0_10_fast.xml:176`,
`prod_OO_isr/OO_MCGlauber_MUSIC_isr.xml:182`, `docs/PbPb_5020/PbPb_0_10_jet.xml:213`.

The two tables, from the build:

| T [GeV] | e₉₁/e₉ | P₉₁/P₉ | c_s² 9 / 91 |
|---|---|---|---|
| 0.12 | 1.016 | 1.009 | 0.183 / 0.181 |
| 0.15 (T_fo) | 1.038 (e_fo = 0.234 vs 0.243 GeV/fm³) | 1.022 | 0.156 / 0.154 |
| 0.17 | 1.058 | 1.037 | 0.150 / 0.147 |
| 0.18 | 1.049 | 1.043 | 0.163 / 0.175 |
| 0.3–0.5 | 1.012–1.013 | 1.013–1.018 | the same |

`run_prod_jet.py:235` already takes e(T_fo) and the edge flag from the right table.

**iSS follows MUSIC's EOS.** The particlize file stores `music_input` verbatim; `hadronize.py`
writes it into iSS's working directory, and iSS reads `EOS_to_use` from it
(`iSS/src/readindata.cpp:93`). EOS 9 forces the UrQMD list, 91 the SMASH list, HRG table and δf
tables (`readindata.cpp:134–139`). The tables for both ship with iSS (`pdg-SMASH.dat`,
`chosen_particles_SMASH.dat`, `HRGEOS_PST-SMASH.dat`, `deltaf_tables/smash`, `smash_box`).
A hydro of one EOS can't be particlized with the other list (−4 % energy at the surface with
91 + UrQMD).

**With EOS 91 iSS does not decay.** The SMASH list sets the afterburner type to SMASH, and the
decays are skipped for it (`iSS/src/FSSW.cpp:403`, and `:1549–1550` in the correlated sampling).
`hadronize.py:569` and `run_prod_oo_isr.py` (js-contrib `cc10ae7`) warn about it.

**The decayer can't do the SMASH list as it is** (`iSS/src/particle_decay.cpp`):
- It decays at the pole mass (the Breit–Wigner sampling at `:314` is commented out) and calls
  `exit(1)` for a closed channel (`:308–312`). **215 channels of `pdg-SMASH.dat` are closed at the
  pole mass**, among them ω → πρ (SMASH's form of ω → 3π: three charge channels, 90 % of ω decays),
  η′ → ησ (66 %),
  f₂ → ρρ, K₁ → Kω. SMASH handles them with mass-dependent widths.
- `perform_decays` (`:237–283`) handles 2- and 3-body channels only.
- **The SMASH list has η and Σ⁰ stable**; the UrQMD list decays them (η → γγ, 3π;
  Σ⁰ → Λγ). The charm and bottom states listed as stable are not in
  `chosen_particles_SMASH.dat`, so iSS never samples them.

**A loss in today's EOS 9 production.** The UrQMD list marks its 45 closed channels with
`Npart = -2` (b₁(1235) → ηρ 10 %, N(1520) → Nρ 15 %, K₁(1270) → Kω 11 %, N(1535) → N(1440)π,
…). `perform_decays` handles neither branch for them, so the mother is dropped with no
daughters: those resonances vanish silently, with their energy and charge. How many hadrons this
costs per event has not been measured.

**The jet hadronization keeps only final hadrons, at x = 0.** `ColorlessHadronization.cc:509`
(Au+Au) and `iColoredHadronization.cc:268` (O+O) take `isFinal()` from Pythia's record, with an
empty `FourVector x` or `x[4] = {0,0,0,0}`. Pythia keeps the whole history, though: the string
fragmentation's hadrons have status 81–89 and turn negative when they decay in the same
`pythia.next()`, so the primaries are there.

**How the bulk hadrons reach the files.** iSS → `iSpectraSamplerWrapper` (flat arrays, all
`pstat 11`, `iSpectraSamplerWrapper.cc:452`) → `core.soft_hadrons_numpy` →
`jetscape.hadrons_h5.HadronH5Writer`. `hadronize.py` (Au+Au stage 2, O+O from kept particlize
files) and `run_prod_oo_isr.py` (O+O in the job, `--bulk-hadrons`) both write with that class.

**Where the jet hadronization can run:**
- **Au+Au:** in stage 2 from the particlize file's `partons/` (`FINAL_PARTON_COLUMNS`, with t,
  x, y, z, col, acol; `particlize_h5.py:94`). The colorless module makes its own remnants. So it
  can be redone at any time on kept particlize files.
- **O+O:** only in the job. iColoredHadronization needs the beam remnants from 3dMCGlauber
  (`iColoredHadronization.cc:155–179`) and the ISR colour connections, which exist only during
  the job (`prod_OO_isr/README.md:195`). Today only `remnant_E` and `remnant_pz` are recorded per
  event.

**How JETSCAPE gives fragmentation hadrons positions.** Only `HybridHadronization` does
(JETSCAPE `main` since v3.6, `roch/fragmentation_hadrons_afterburner`; the same code in X-SCAPE
`contrib`):
- each final parton starts at its production point (`x_in`, `HybridHadronization.cc:701–704`);
- inside the medium it free-streams until T ≤ `hydro_Tc` (asking the hydro every `Eloss/deltaT`);
  outside it stays where it is, plus at least `part_prop` (`:885–957`). So the hadronization
  point is the later of production point and T_c crossing;
- recombination hadrons get their constituents' position; string hadrons are traced back to
  their string's partons with `StringFragmentation:TraceColours` and placed on the gluon, on the
  segment between two partons (spread along it), or at a junction's centre
  (`set_spacetime_for_pythia_hadrons`, `:9476–10100`).

X-SCAPE branch `cparker/eA-update` (`ColoredHadronizationSinglet`, e+A) does it more simply: the
partons' `x_in` become Pythia's production vertices (`vProd`) and the hadrons take Pythia's
`vProd` (`b21c45b3`); hadrons Pythia places "at infinity" are moved inside SMASH's range
(`225b5064`). It is not merged; the hadron shifting there is specific to e+A.

## Decisions before starting

1. **Which setup were the model parameters calibrated with?** X-SCAPE's default
   `jetscape_main.xml:549` uses EOS 91, and 3D Glauber + MUSIC calibrations usually run SMASH as
   afterburner. If our parameter sets come from such a calibration, the missing hadronic stage
   matters more than 9 vs 91, and this plan is the way back to it. If they were tuned with EOS 9
   and no afterburner, EOS 91 changes the tuned setup.
2. **Fix the EOS 9 loss too?** The Breit–Wigner daughter masses (1.2) also open the UrQMD
   list's closed channels. That changes EOS 9 output (not bit-identical) and needs a version
   attribute, so files from before and after are told apart. Measure the loss first (1.0).
3. **The jet-hadron position model** (2.3): the string's mean position (simple), or Hybrid
   Hadronization's colour tracing (more exact, larger). And whether a hadron formation time is
   added (2.4).

## Part 1: the bulk (iSS)

### 1.0 Measure the EOS 9 loss (½ day)

Count, per event, the hadrons iSS drops through the `-2` channels (a counter in
`perform_decays`, or a Python re-run of the decay chain on one file's primaries). Report the
dropped energy per event and per species. Decides decision 2.

### 1.1 iSS: decays for the SMASH list (½ day)

Decouple "which list" from "who decays": `FSSW.cpp:403` and `:1549–1550` decay whenever
`perform_decays` is on and **no SMASH runs** (a parameter, e.g. `afterburner_runs`, default 0
in our XMLs; X-SCAPE's `SmashWrapper` path sets it to 1). The list stays the one the EOS forces.

### 1.2 iSS: Breit–Wigner masses for unstable daughters (1–2 days)

In `perform_two_body_decay` / `perform_three_body_decay`: when the pole masses close the channel,
sample the unstable daughters' masses from a Breit–Wigner with m₁ + m₂ (+ m₃) ≤ M
(`sample_breit_wigner`, `:501`, already there), and keep the sampled mass in the daughter, so its
own decay uses it. Stable daughters keep their mass. A channel that stays closed after the
sampling (a stable-only channel below threshold) is skipped and another one drawn, never
`exit(1)` and never a silent drop. `Npart = -2` is read as a 2-body channel.

The random numbers: the correlated sampling addresses its random numbers by cell
(`FSSW.cpp`, Philox); the extra draws must come from the same per-primary stream, so the jet leg
and the background still decay a shared primary identically.

### 1.3 Our copy of `pdg-SMASH.dat`: η and Σ⁰ decays (½ day)

Add their channels from `pdg-urqmd_v3.3+.dat`, so the final stable set is the same as with
EOS 9 (γ, π, K, p, n, Λ, Σ±, Ξ, Ω; π⁰ stays stable as today, see
[`PLAN_analysis_decays.md`](PLAN_analysis_decays.md)). Record the table's checksum in the
hadron files.

### 1.4 iSS: `keep_primaries` (½ day)

A parameter (`keep_primaries`, default 0). Both decay paths, `decay_one_primary` (correlated
sampling) and `perform_resonance_feed_down`, copy each primary into a second list before
decaying it. No new random numbers, so the final hadrons are bit-identical with and without it.

### 1.5 Wrapper and binding (½ day)

`iSpectraSamplerWrapper` hands the second list over next to the final one (flat arrays, as the
compact output does); PyJetscape gets `core.soft_primaries_numpy` and a capability flag
(`core.ISS_DECAYS_SMASH_LIST`), which the warnings use.

### 1.6 Writers, precision, attributes (1 day)

- `hadronize.py` and `run_prod_oo_isr.py` write the primaries as separate groups,
  `bulk_bg_primary` and `bulk_jet_primary`, with the layout of the final groups (`pid`, `pstat`,
  `p`, `x`, sample and unit offsets). Existing readers keep reading only the final groups.
- **Precision per group.** SMASH needs the primaries' positions and full float32 momenta: the
  mass comes from E² − p², and with 12 mantissa bits a 5 GeV pion's m² is ~25 % off.
  `--keep-bits-p`, `--keep-bits-x` and `--no-x` then apply to the final groups only, and the
  primaries keep full precision (or get their own options).
- Attributes: `decayed` (final groups: true; primaries: false), `hadron_list` (`SMASH` /
  `UrQMD`), the iSS commit and table checksum, `decayer` (`iSS, Breit–Wigner daughters`).
- An option for the setting: `hadronize.py --keep-primaries`, `run_prod_oo_isr.py
  --keep-primaries` (writes them for the in-job bulk), passed into the iSS block of
  `hadronize.xml`.

### 1.7 EOS 91 in the XMLs and the warnings (2 h)

- An `--eos 9|91` option in `run_prod_jet.py` and `run_prod_oo_isr.py` that sets `<EOS>` in the
  job XML (the default stays 9), recorded in the files; the XML comments about EOS 91 updated.
- The warnings in `hadronize.py:569` and `run_prod_oo_isr.py` only for a build without
  `core.ISS_DECAYS_SMASH_LIST`.

### 1.8 Validation (1–2 days)

- **Decays:** energy, momentum and charge conserved per primary through its chain; no
  `exit`, no dropped mother; the closed-channel draws distributed as the Breit–Wigner.
- **Primaries → finals:** decaying a file's stored primaries with the same per-primary random
  stream reproduces its final hadrons exactly (a test script on one job of each system).
- **Bit-identity where nothing should change:** EOS 9 with `keep_primaries 0` and without the
  Breit–Wigner change (if decision 2 is "no") identical to today's hadrons
  (the baseline-vs-new comparison of earlier bit-identical changes, e.g.
  [`PLAN_iSS_optim.md`](PLAN_iSS_optim.md) Part C).
- **Physics:** EOS 91 against EOS 9 on the same seeds: dN/dη, identified π/K/p yields and
  spectra, ⟨p_T⟩, the wake observables; EOS 9 before and after the Breit–Wigner change.
- **Speed:** iSS samples ~400 instead of ~321 species and decays longer chains; measure stage-2
  time and memory per event.

## Part 2: the jet hadrons

### 2.1 `keep_primaries` in both jet modules (½–1 day)

`ColorlessHadronization` and `iColoredHadronization`: with `keep_primaries` on, a second loop
over Pythia's record collects |status| 81–89 (the primary hadrons, decayed or not) into a
second list. `pythia_decays` can stay on: the final hadrons are unchanged. Through the
HadronizationManager and the binding (`core.hadronization_primaries_numpy`), written as
`jet_frag_primary`, one per fragmentation sample (Au+Au `--n-frag`, O+O one), so a converter can
pair fragmentation sample k with bulk oversample k.

### 2.2 The partons' hadronization point (1–2 days)

The rule of Hybrid Hadronization: **the later of the production point and the T_c crossing on
the parton's straight path**, plus at least `part_prop`. The same for the negative partons.
- **O+O (in the job):** with the hydro, as Hybrid Hadronization does (`GetHydroCellSignal` every
  `Eloss/deltaT`), at T_fo = 0.15 GeV.
- **Au+Au (stage 2):** no hydro, but the jet leg's freeze-out surface is in the particlize file.
  The crossing is where the straight path leaves the region inside the surface; a parton that
  starts outside stays. (A cell-wise lookup on the stored surface, done once per parton in
  Python before the partons go to Pythia.)
- Stored per parton next to its production point (O+O: in `hadronization_input/`, 2.5; Au+Au:
  computed in stage 2, no new stage-1 data).

### 2.3 Hadron positions from the strings (1 day simple; 3–5 days with colour tracing)

- **Simple:** each string hadron gets the momentum-weighted mean position of its string's
  partons (after 2.2), its time likewise. Strings are known: the colorless module builds them
  itself, iColored hands Pythia a colour-connected system.
- **As Hybrid Hadronization:** `StringFragmentation:TraceColours = on`, each hadron traced to one
  gluon, a segment (positions spread along it) or a junction; ported from
  `set_spacetime_for_pythia_hadrons`. A hadron that can't be traced gets the string mean and is
  counted, never (0, 0, 0, 0) silently.
- Primaries and final hadrons both get positions; a decay product takes its mother's position
  (Pythia's decay vertices are tiny at tau0Max = 10 mm/c, or use Pythia's `vProd` offsets).
- The files record the model (`jet_positions`: `none` | `string_mean` | `colour_trace`).

### 2.4 Formation time (decision only)

Placing a hadron where its partons are lets SMASH interact with it at once; a fast hadron
forms much later in the string picture. Either a formation time per hadron in the file
(e.g. from the string's breakup, or E/m² scaled), or left to SMASH's settings. Ask Hendrik Roch
what JETSCAPE's rescattering study does.

### 2.5 O+O: what is needed to redo the jet hadronization later (1 day)

`iColoredHadronization` dumps the complete parton list it gives Pythia (final partons, ISR
partons, beam remnants: pid, four-momentum, col, acol, t, x, y, z, the hadronization point of
2.2) into a ragged group `hadronization_input/` of the jet file (the layout of `partons/`). The
Pythia seed is already recorded (`reseed_per_event`). Then the O+O jet hadronization can be redone
offline with other settings (positions, decays off, another model), which today is impossible.
Size per event: tens of KB (estimate; next to 0.13 MB of hadrons per event in hadron-only runs).
Needs an offline driver that feeds the stored list to Pythia the way iColored does.

### 2.6 Validation (1–2 days)

- `keep_primaries` on/off: the final jet hadrons bit-identical.
- Redoing the O+O jet hadronization from `hadronization_input/` with the stored seed
  reproduces the in-job jet hadrons exactly.
- Positions: every hadron has one; the distributions of hadronization points and hadron
  positions against the freeze-out surface (inside / on / outside), per p_T bin; no hadron at
  "infinity" (the e+A branch's problem).

## Where it applies

| | Au+Au | O+O |
|---|---|---|
| **EOS 91** | stage-1 XML (`--eos 91`) | stage-1 XML (`--eos 91`) |
| **Part 1, bulk** | stage 2 (`hadronize.py`), on kept particlize files | in the job (`--bulk-hadrons`) or stage 2 on kept particlize files |
| **Part 2, jet primaries** | stage 2 (`hadronize.py`, colorless) | **in the job only** (iColored), or offline from `hadronization_input/` once 2.5 exists |
| **Part 2, positions** | stage 2, against the stored jet-leg surface | in the job, against the hydro |
| redo later | everything, as long as the particlize files are kept | the bulk only with particlize files or with the bulk primaries; the jet hadrons only with 2.5 |

**O+O with hadron files only** (`--no-particlize --no-hydro-file`) keeps nothing to redo the
bulk: with EOS 91, either keep the particlize files (~10 MB per event instead of 0.13 MB) or
write the bulk primaries in the job (1.6). And the jet hadrons need 2.5 to be redone at all.

## Repositories, branches, images

| repo | branch (from) | parts |
|---|---|---|
| iSS fork `jhputschke/iSS` | `smash_ready` (from `iSS_optim` 98deaff) | 1.1–1.4 |
| X-SCAPE | `smash_ready` (from `contrib`) | the wrapper (1.5), `get_iSS.sh` pin, both jet modules (2.1–2.3, 2.5) |
| js-contrib | `smash_ready` (from `main`) | binding, `hadronize.py`, `run_prod_*.py`, `hadrons_h5`, tests, docs |

The iSS branch must be pushed before the X-SCAPE pin works for a fresh clone. Production images
need a rebuild (CI for amd64, see `utils/BuildContainerProd.md`).

## Effort

| part | days |
|---|---|
| 1.0 measure the EOS 9 loss | ½ |
| 1.1–1.4 iSS | 2½–3½ |
| 1.5–1.7 wrapper, binding, writers, options | 2 |
| 1.8 validation | 1–2 |
| **Part 1** | **6–8** |
| 2.1 jet primaries | ½–1 |
| 2.2 hadronization points | 1–2 |
| 2.3 positions (simple / colour tracing) | 1 / 3–5 |
| 2.5 O+O `hadronization_input/` + offline driver | 1–2 |
| 2.6 validation | 1–2 |
| **Part 2** | **4½–12** |

## Follow-ups

- **SMASH on the stored primaries** (its own plan): a converter to SMASH's OSCAR2013 List
  input (one SMASH event per oversample, jet sample k with bulk sample k), SMASH 3.2.2 as pinned
  by `get_smash.sh`, and the way back to the hadron format. Check that every pid in the files
  exists in SMASH's `particles.txt` with the same mass (`pdg-SMASH.dat` must match the version),
  and how its List mode treats hadrons emitted at different times (back-propagation with
  formation times). Time SMASH per central Au+Au event before planning a campaign.
- **Au+Au's correlated sampling after SMASH** (expected, not measured): SMASH's rescattering
  diverges between the jet leg and the background as soon as they differ, so most of the
  noise cancellation of the jet − background subtraction is probably lost. O+O has one leg and
  is not affected.
- **A hadrons-only mode for Au+Au**, as O+O has (`run_prod_jet.py --validate-inline` does half
  of it): only if disk or upload becomes the limit; it moves stage 2's CPU into the GPU job.
- **Analysis-level decays** ([`PLAN_analysis_decays.md`](PLAN_analysis_decays.md)) work on the
  final groups as before; the stable set stays the same with 1.3.

## Open points

- Decisions 1–3 above.
- Hybrid Hadronization's `hydro_Tc` default against our T_fo = 0.15 GeV.
- How `ColoredHadronizationSinglet` (`cparker/eA-update`) gets hadron vertices out of Pythia
  without `Fragmentation:setVertices`, and why some come out at infinity (Cameron Parker,
  erickolbusz).
- Whether X-SCAPE upstream wants 1.1–1.2 and 2.1–2.3 (they change shared modules): keep them
  behind switches that default to today's behaviour.
