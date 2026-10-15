<!-- Plan, written 2026-10-14. Status: not started. Branches to use: smash_ready in the iSS fork (from iSS_optim 98deaff), X-SCAPE (from contrib) and js-contrib (from main). -->

# Plan: EOS 91 hadrons, undecayed primaries, and SMASH as a later afterburner (Au+Au and O+O)

## Goal

1. **EOS 91 in production.** MUSIC with EOS 91 (hotQCD + the SMASH hadron list) and iSS
   decaying that list itself, so an EOS 91 campaign gives decayed hadrons like EOS 9 does today.
2. **Bulk primaries.** iSS also keeps the undecayed hadrons (with positions), so SMASH can run on
   them later, without a new hydro or particlization.
3. **Jet primaries with positions.** The jet hadronization keeps its undecayed hadrons as well,
   and gives every jet hadron a space-time position, so the jet hadrons can go into SMASH too.
4. **The infrastructure to run SMASH** on the stored primaries, as a separate step on CPU nodes
   after the production, never inside the GPU jobs (too much CPU for them). Lower priority than
   Parts 1 and 2, and done after them.

For **Au+Au and O+O**, in every place where the hadronization runs (table in
[Where it applies](#where-it-applies)). Part 1 is the bulk (goals 1 and 2), Part 2 the jet hadrons
(goal 3), Part 3 the SMASH step (goal 4); Part 3 adds to Part 1, it does not replace it.

Not in this plan: SMASH campaigns and their analysis, and a hadrons-only mode for Au+Au
([Follow-ups](#follow-ups)).

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
3. **The jet-hadron formation time** (2.4): Pythia's Lund space-time picture of the string
   breakups (`Fragmentation:setVertices`) on top of the positions of 2.3, or no formation time.

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

### 2.3 Hadron positions from the strings: Pythia's own (½–1 day)

**Pythia already does it.** With `PartonVertex:setVertex = on`, `HadronLevel::next` calls
`PartonVertex::vertexHadrons` after each string fragmentation (Pythia 8.316 source,
`HadronLevel.cc:261`, `PartonVertex.cc:196–330`; the image has 8.312, which has the same
settings):
- **qq̄ strings and closed gluon loops:** the primary hadrons are taken in their order along
  the string; each one's energy midpoint is located between the two partons whose energies span
  it (a gluon counts half to each side), and the hadron gets the **energy-weighted interpolation
  of those two partons' vertices** (`vProdAdd`);
- **junction systems:** the same along each leg, the legs identified by their status (85, 86);
- **a string that collapses to one hadron:** the midpoint of its end partons;
- no random numbers are drawn, so the hadrons' momenta should stay exactly the same (to check).

That is Hybrid Hadronization's "spread along the string segments", done by Pythia from its own
string record: no colour tracing of our own (`set_spacetime_for_pythia_hadrons`, ~600 lines),
and nothing different for colorless and iColored.

What the two modules need:
- each input parton's `vProd` = its hadronization point of 2.2 (fm → mm, `Pythia8::FM2MM`);
  today `ColorlessHadronization` sets only the beam remnants' `vProd(0,0,0,0)` (`:180`) and
  iColored none. The remnants: at their partner's hadronization point, or the event's hard
  vertex (a choice to record);
- `PartonVertex:setVertex = on`: iColored can take it through `<LinesToRead>` (`:91–96`);
  Colorless has no such hook, so a switch (`<JetHadronization><hadron_vertices>`, default off) in
  both;
- the hadron's `vProd()` (mm → fm) into the output's `x` instead of the zeros (Colorless `:324`, `:513`,
  iColored `:273`), for the primaries and the final hadrons.

Decay products get Pythia's decay vertices (`vDec` of the mother). In fm these are far away for
anything not strongly decaying: π⁰ cτ ≈ 25 nm ≈ 2.5·10⁷ fm, D mesons ~0.1 mm ≈ 10¹¹ fm. That is
most likely the e+A branch's "hadron formed at infinity" with decays on (not verified for its
configuration). For SMASH only the primaries' positions matter (they decay in SMASH); the final
hadrons' positions are bookkeeping and can store the decay vertex or the primary's position (a
recorded choice). The files record the model (`jet_positions`: `none` | `pythia_parton_vertex`
| `+lund_breakup`, 2.4).

### 2.4 Formation time: Pythia's string breakup points (decision; ½ day)

`Fragmentation:setVertices = on` adds the Lund-model space-time of the string breakups
(`StringFragmentation::setHadronVertices`, `StringFragmentation.cc:1109–1590`): from the string's
momenta and tension (`HadronVertex:kappa` = 1 GeV/fm) each breakup point gets a space-time
position relative to the string's origin, the hadron is produced at the middle of its two
breakups (`HadronVertex:mode` 0), with a Gaussian transverse smearing (`xySmear` 0.5 fm). A fast
hadron then forms far out and late, roughly in proportion to its energy: a formation time from
the model itself, added on top of the parton vertices of 2.3 (both add to `vProd`).

Caveats:
- the smearing draws random numbers, and a hadron with |τ| > `HadronVertex:maxTau` (40 fm) makes
  Pythia redo the fragmentation: the hadrons themselves then differ from a run without it, not
  only their positions;
- the breakups are measured from the string's origin, here the hadronization point of 2.2: the
  string is taken to form there;
- whether JETSCAPE's rescattering study uses a formation time, and which: ask Hendrik Roch.

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
- Positions: every primary has one; `PartonVertex:setVertex` on/off gives the same momenta
  (no random numbers in `vertexHadrons`); the distributions of hadronization points and hadron
  positions against the freeze-out surface (inside / on / outside), per p_T bin; with 2.4, the
  formation times against E/κ, and how often `maxTau` makes Pythia retry.

## Part 3: SMASH as afterburner (a separate CPU step, later)

The production keeps running as before (decayed hadrons from Parts 1 and 2, plus the
primaries). SMASH runs afterwards, on CPU nodes, on the stored primaries; its hadrons go into
files of their own. Lower priority than Parts 1 and 2.

### Two ways to run SMASH, and which one

| | standalone SMASH (List modus) | X-SCAPE's `SmashWrapper` (`-DUSE_SMASH=ON`) |
|---|---|---|
| input | the stored primaries, converted to SMASH's OSCAR2013 list input | iSS's hadrons in the same JETSCAPE job (stage 2 on kept particlize files), or a new "hadron replay" module for stored primaries |
| Pythia | SMASH 3.2.2 needs **exactly Pythia 8.315** (`find_package(Pythia 8.315 EXACT)`); standalone it brings its own, in its own environment | all of X-SCAPE has to link the same 8.315: PythiaGun, PythiaIsrGun and both jet hadronizations change (the images have **8.312**), so the jet production is no longer bit-identical to the existing campaigns |
| production images | unchanged | SMASH in the build and runtime stages, both architectures |
| code to write | converter there and back, driver | the compact iSS output leaves `Hadron_list_` empty, which `SmashWrapper` reads (`GatherAfterburnerHadrons`): read the compact arrays (or switch them off, ~8× the memory); a binding for the afterburner's hadrons (status 27, `SmashWrapper.cc:263–275`); the job XML |
| jet hadrons | in the same list as the bulk sample | only with one bulk oversample (`Afterburner.cc`, `include_fragmentation_hadrons`) |

**Standalone is the way for this plan:** it is a separate step anyway, it leaves the production
images and their Pythia alone, and it works the same for files from both systems. The
`SmashWrapper` route stays possible for a one-job stage 2 later, if the Pythia change is
accepted for other reasons.

### 3.0 Measure first (½–1 day)

SMASH 3.2.2 standalone on a few converted samples: one central Au+Au background, one jet leg,
one O+O event. Time and memory per SMASH event, at the planned `end_time`. This decides how many
oversamples get a SMASH run (all 100–200, or 10–20) and how many CPU-hours a campaign needs.
Also on the same samples, one thread each:
- `smash-fast` (default settings), for its cell-list partner re-search (bit-identical) and
  with `Lazy_Propagation` (not bit-identical); see 3.4;
- `Delta_Time` 0.05 against 0.5 fm/c, and `End_Time` 1000 against ~200 fm/c.

### 3.1 The SMASH environment (1 day)

SMASH 3.2.2 with Pythia 8.315, Eigen3 and GSL, as a CPU-only image (`xscape-smash`, both
architectures, built by CI like the production images) and as a conda environment for the
workstations. The SMASH and Pythia versions go into every output file.

### 3.2 Converter: hadron files → SMASH input (1 day)

- One SMASH event per bulk oversample: `bulk_bg_primary` (O+O, and the Au+Au background) or
  `bulk_jet_primary` (the Au+Au jet leg), OSCAR2013 (`t x y z mass E px py pz pdg id charge`).
- Every pid checked against SMASH's `particles.txt` (`pdg-SMASH.dat` must belong to the same SMASH
  version), the mass taken from the full-precision momenta (1.6) or set to the pole mass.
- After Part 2: jet sample k added to bulk sample k, each particle's origin kept in the id
  range, so the output can be split again.

### 3.3 Driver and SLURM script (2 days)

- `run_afterburner.py FILE|DIR [--samples N] [--end-time T] [--only-decays] [-j J]`: converts,
  runs SMASH per sample (CPU parallel over samples), reads SMASH's output back and writes
  `<stem>_hadrons_smash.h5` with groups `bulk_bg_smash`, `bulk_jet_smash` (and later the jet
  hadrons) in the `hadrons_h5` layout, attributes: SMASH and Pythia version, the config, the
  seed per sample (derived from the file's seed and the sample index, so a rerun is identical).
- `utils/slurm_afterburner.sh`: one CPU task per stage-1 task directory, like
  `slurm_hadronize.sh`; resumable (skips finished files).
- `HadronFileReader` reads the new groups (½ day of it).

### 3.4 SMASH settings (½ day)

**What JETSCAPE and X-SCAPE run.** `external_packages/smash/smash_config.yaml` (the same in
JETSCAPE `main`): List modus, `Time_Step_Mode: Fixed`; `SmashWrapper` sets `End_Time` and
`Delta_Time` from the XML, which defaults to **`end_time` 1000 fm/c and `Delta_Time` 0.05 fm/c**
(`jetscape_main.xml`), i.e. 20,000 time steps per event. Everything else is SMASH's default:
one ensemble, no potentials, **`Strings: true`** (the default for every modus but Box). In
JETSCAPE's [Default-tunes](https://github.com/JETSCAPE/Default-tunes), SMASH is in two
configurations only, both with these defaults: arXiv:2011.01430 (`<SMASH>` with just its name)
and arXiv:2510.06996 (`end_time` 1000, `only_decays` 0); the Pb+Pb and p+Pb tunes have no
afterburner.

Ours: the same physics settings (strings on, no potentials), `end_time` and `Delta_Time`
chosen from 3.0's timing, the List-modus options for hadrons emitted at different times (as
far as known, SMASH propagates them back and gives them a formation time; check for 3.2.2), the
output (final particles only), `only_decays` as a switch.

**What `smash-fast` (`jhputschke/smash-fast`) can do for these settings:**

| its speed-up | applies to the afterburner? |
|---|---|
| ensembles in parallel (5.5–6.5× at 8 threads), strings in parallel | **no**: List modus refuses more than one ensemble (`listmodus.cc:92`) |
| cell-parallel finding with one ensemble (3.9× at 8 threads) | **no with `Strings: true`**, which switches it off (`experiment.h:2729`). `Strings: false` would be a physics change here (high-√s hadron pairs, jet hadrons), unlike in its low-energy use; and performing stays serial |
| mean field, GPU kernels | **no**: no potentials in the afterburner |
| cell-list partner re-search (default, bit-identical, single thread) | **yes**: every non-stochastic collision run; it removes the O(N²) term of performing, 1.25–1.83× on its box tests (100–400 test particles); for thousands of hadrons in a central Au+Au event probably more, not measured |
| `Lazy_Propagation` (opt-in, not bit-identical) | **yes**: 1.34–1.66× more on its box tests |

With one ensemble and strings on, more OpenMP threads make `smash-fast` *slower* (a Pythia
instance per thread, nothing to split). So: **one thread per SMASH process, and the samples in
parallel as separate processes**, which is what 3.3 does. `smash-fast` is SMASH 3.3 with
Pythia 8.316 (standalone, so no conflict), and its verification suite has no List-modus case
yet: add one before using it.

### 3.5 Validation (2 days)

- **`only_decays` against iSS's decays (Part 1):** the same primaries decayed by SMASH and by
  iSS; identified yields and spectra agree within SMASH's mass-dependent widths. A cross-check of
  Part 1.2.
- Energy, momentum and charge per event before and after SMASH; the same seed gives the same
  output.
- Full SMASH against none: the expected changes (baryon–antibaryon annihilation, resonance
  regeneration, p_T broadening of protons), O+O and Au+Au.
- For Au+Au: how much of the jet − background noise cancellation survives (see
  [Follow-ups](#follow-ups)), on one campaign's worth of samples.

## Where it applies

| | Au+Au | O+O |
|---|---|---|
| **EOS 91** | stage-1 XML (`--eos 91`) | stage-1 XML (`--eos 91`) |
| **Part 1, bulk** | stage 2 (`hadronize.py`), on kept particlize files | in the job (`--bulk-hadrons`) or stage 2 on kept particlize files |
| **Part 2, jet primaries** | stage 2 (`hadronize.py`, colorless) | **in the job only** (iColored), or offline from `hadronization_input/` once 2.5 exists |
| **Part 2, positions** | stage 2, against the stored jet-leg surface | in the job, against the hydro |
| **Part 3, SMASH** | separate CPU step after stage 2, on the primaries | separate CPU step after the job (in-job primaries) or after stage 2 |
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
| js-contrib | later branch (Part 3) | converter, `run_afterburner.py`, `slurm_afterburner.sh`, the `xscape-smash` image and its CI |

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
| 2.3 positions (Pythia's parton vertices) | ½–1 |
| 2.4 formation time (Pythia's breakups, if chosen) | ½ |
| 2.5 O+O `hadronization_input/` + offline driver | 1–2 |
| 2.6 validation | 1–2 |
| **Part 2** | **4½–8½** |
| 3.0 SMASH timing | ½–1 |
| 3.1 SMASH environment and image | 1 |
| 3.2 converter | 1 |
| 3.3 driver, SLURM script, reader | 2 |
| 3.4 SMASH settings | ½ |
| 3.5 validation | 2 |
| **Part 3** | **7–7½** |

The numbers are estimates of focused working days (writing, building, a small test run,
fixing), about ±50 %. Not included: waiting on decisions, PR review, CI image builds,
production-scale validation runs.

## Follow-ups

- **SMASH campaigns and their analysis**, with the infrastructure of Part 3.
- **A faster SMASH for the afterburner:** of `smash-fast`'s speed-ups only the single-thread
  ones apply (3.4). A List-modus verification case, then its time against SMASH 3.2.2 in 3.0.
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
- `ColoredHadronizationSinglet` (`cparker/eA-update`) sets the partons' `vProd` but neither
  `PartonVertex:setVertex` nor `Fragmentation:setVertices` (none found on the branch), so its
  primary hadrons should come out at (0,0,0,0) and the "infinity" ones from decay vertices: ask
  Cameron Parker / erickolbusz whether that is what they see.
- Whether conda-forge has Pythia 8.315 (for 3.1), or it is built from source in the image.
- Whether X-SCAPE upstream wants 1.1–1.2 and 2.1–2.3 (they change shared modules): keep them
  behind switches that default to today's behaviour.
