<!-- Plan, written 2026-10-14. Status (2026-10-15): Part 1 and the minimal Part 2 (O+O: the jet hadronization's input stored, opt-in) implemented on branches smash_ready in the iSS fork (from iSS_optim 98deaff), X-SCAPE (from contrib) and js-contrib (from main); see "Implementation status". Parts 2.1-2.4, 2.6 and 3 not started. -->

# Plan: EOS 91 hadrons, undecayed primaries, and SMASH as a later afterburner (Au+Au and O+O)

## Goal

1. **EOS 91 in production** (the tunes' EOS; decision 1). MUSIC with EOS 91 (hotQCD + the SMASH
   hadron list) and iSS decaying that list itself, so an EOS 91 campaign gives decayed hadrons
   like EOS 9 does today.
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

## Implementation status (2026-10-15)

Done on the `smash_ready` branches; scope set on 2026-10-15: Part 1, and of Part 2 only what
is needed to redo the O+O jet hadronization later (positions left for later), opt-in for O+O.

| item | where | state |
|---|---|---|
| 1.1 iSS decays the SMASH list | iSS `FSSW` (`decay_smash_list`, default 0: as before), X-SCAPE wrapper + `jetscape_main.xml` default, `hadronize.xml` sets 1 | done |
| 1.2 Breit–Wigner daughter masses, re-drawn closed channels | iSS `particle_decay` (`smash_list_`: SMASH list only, UrQMD/PDG lists unchanged) | done |
| 1.0 measure the EOS 9 loss, 1.3 η/Σ⁰ decays | — | dropped (decisions 2 and 4) |
| 1.4 `keep_primaries` | iSS `FSSW` (first N samples, both sampling paths) | done |
| 1.5 wrapper, binding | `SoftParticlization` compact primaries, `core.soft_primaries_numpy`, `ISS_DECAYS_SMASH_LIST`, `SOFT_HAS_PRIMARIES`, `PYTHIA_VERSION` | done |
| 1.6 writers | `hadrons_h5` tags `bulk_jet_primary` / `bulk_bg_primary` (own files), `hadronize.py` / `run_prod_oo_isr.py --keep-primaries N`; attributes `decayed`, `hadron_list`, `iss_decays`, `iss_commit` | done |
| 1.7 EOS 91 | the three production XMLs, `--eos` in `run_prod_jet.py` and `run_prod_oo_isr.py`, `eos` / `eos_kind` recorded, `wake_observables.py`, `make_surfaces.py`, `music_cpu_gpu_hydro.py` take the file's EOS table | done |
| 1.8 validation | see 1.8 | done, except speed at production scale |
| 2.5 minimal: O+O input stored | iColored `<reseed_per_event>`, `<store_input>`, `HadronizeInput`; `core.hadronization_input_numpy` / `hadronize_input`; `run_prod_oo_isr.py --store-hadronization-input` (opt-in; reseed always on), `hadronization_input/` group, `rehadronize_jets.py` | done |
| Pythia version in the jet files | `pythia_version` attribute (hadronize.py jet_frag, O+O jet_frag) | done |
| Au+Au replay check | `hadronize.py --check [JET_FRAG]`: re-makes a jet_frag file from `partons/` with the seeds and settings it records and compares (Au+Au needs no stored input: Colorless runs from the particlize file's partons) | done |

## What the code does today

**MUSIC.** EOS 9 and 91 use the same code (`EOS_hotQCD`, `music4gpu/src/eos_hotQCD.cpp:44–49`),
only the table differs. Both are in the image (`get_music4gpu.sh` downloads `binary` and
`SMASH_binary`). All production XMLs run 9:
`config/AuAu_MCGlauber_MUSIC_0_10_jet.xml:207`,
`config/AuAu_MCGlauber_MUSIC_0_10_fast.xml:176`,
`config/OO_MCGlauber_MUSIC_isr.xml:182`, `config/PbPb_0_10_jet.xml:213` (all in `example/config/` since 2026-10-15).

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

All taken; 2–4 on 2026-10-15.

1. **Decided (2026-10-14): the tunes are EOS 91, and the production campaigns use EOS 91.**
   Today's EOS 9 production differs from the tuned setup in the EOS (the table above) and in
   the hadron list. Consequences:
   - EOS 91 becomes the default of the production XMLs (1.7), not an option.
   - Part 1 (1.1–1.3, 1.8) is needed before an EOS 91 campaign gives decayed bulk hadrons.
     Au+Au stage 1 has no iSS (unless `--validate-inline`), so it can switch to EOS 91 as soon
     as 1.7 is in; its stage 2 waits for Part 1. O+O with in-job bulk hadrons waits for Part 1,
     or keeps its particlize files until then.
   - EOS 9 primaries can never go into SMASH (UrQMD species), so only EOS 91 campaigns can.
   - Still open: whether the tunes also ran SMASH as afterburner. If yes, Part 3 is part of
     the tuned setup too, not only a later option.
2. **Decided: EOS 9 stays bit-identical, and its loss is not measured.** The decay fixes
   (1.2) apply to the SMASH list only. (A by-product of the 1.8 checks: in one Au+Au file, the
   EOS 9 decays lose ~2.5 % of a sample's energy, and charges are not conserved, through the
   UrQMD list's `-2` and 4-body channels.)
3. **The jet-hadron formation time** (2.4): open, but no longer on the production's path: the
   minimal 2.5 lets 2.3–2.4 run in an offline redo of the jet hadronization.
4. **Decided: η and Σ⁰ stay stable** (SMASH's stable set), so 1.3 is dropped: with EOS 91 the
   final hadrons include η and Σ⁰, which EOS 9 decays.
5. **Decided: `--keep-primaries N`**, the first N oversamples, default 0; and the O+O driver
   always reseeds the jet hadronization per event (not bit-identical for O+O jet hadrons).

## Part 1: the bulk (iSS)

### 1.0 Measure the EOS 9 loss (½ day) — dropped (decision 2)

Count, per event, the hadrons iSS drops through the `-2` channels (a counter in
`perform_decays`, or a Python re-run of the decay chain on one file's primaries). Report the
dropped energy per event and per species. Decides decision 2.

### 1.1 iSS: decays for the SMASH list (½ day) — done

Decouple "which list" from "who decays": `FSSW.cpp:403` and `:1549–1550` decay whenever
`perform_decays` is on and **no SMASH runs**. The list stays the one the EOS forces.
Implemented the other way round, so nothing changes for anyone who does not ask: iSS
parameter `decay_smash_list` (default 0: the SMASH list is left to SMASH, as before; X-SCAPE
`<SoftParticlization><iSS><decay_smash_list>`, default 0 in `jetscape_main.xml`), set to 1 in
`hadronize.xml`.

### 1.2 iSS: Breit–Wigner masses for unstable daughters (1–2 days) — done (SMASH list only)

As implemented (`particle_decay.cpp`): a channel open at the pole masses decays as before. One
the pole masses close gets, for each unstable daughter with a width, a mass from the
(non-relativistic, constant-width) Breit–Wigner truncated to [its lowest decay mass, M minus
the others' lowest masses], drawn again until they fit; the lowest decay mass is the lightest
channel's sum, recursively. A channel closed even at the lowest masses is replaced by a draw
among the open ones (by branching ratio); a mother with none stays undecayed (counted and
reported, never `exit`, never dropped; it does not happen with the shipped table, where every
species has an open channel). The `-2` and 4-body channels of the UrQMD list are untouched
(decision 2). SMASH itself uses mass-dependent widths, so far off-shell daughters are more
suppressed there.

In `perform_two_body_decay` / `perform_three_body_decay`: when the pole masses close the channel,
sample the unstable daughters' masses from a Breit–Wigner with m₁ + m₂ (+ m₃) ≤ M
(`sample_breit_wigner`, `:501`, already there), and keep the sampled mass in the daughter, so its
own decay uses it. Stable daughters keep their mass. A channel that stays closed after the
sampling (a stable-only channel below threshold) is skipped and another one drawn, never
`exit(1)` and never a silent drop. `Npart = -2` is read as a 2-body channel.

The random numbers: the correlated sampling addresses its random numbers by cell
(`FSSW.cpp`, Philox); the extra draws must come from the same per-primary stream, so the jet leg
and the background still decay a shared primary identically.

### 1.3 Our copy of `pdg-SMASH.dat`: η and Σ⁰ decays (½ day) — dropped (decision 4)

Add their channels from `pdg-urqmd_v3.3+.dat`, so the final stable set is the same as with
EOS 9 (γ, π, K, p, n, Λ, Σ±, Ξ, Ω; π⁰ stays stable as today, see
[`PLAN_analysis_decays.md`](PLAN_analysis_decays.md)). Record the table's checksum in the
hadron files.

### 1.4 iSS: `keep_primaries` (½ day) — done

A parameter (`keep_primaries` = N, default 0: the first N samples). Both decay paths, `decay_one_primary` (correlated
sampling) and `perform_resonance_feed_down`, copy each primary into a second list before
decaying it. No new random numbers, so the final hadrons are bit-identical with and without it.

### 1.5 Wrapper and binding (½ day) — done

`iSpectraSamplerWrapper` hands the second list over next to the final one (flat arrays, as the
compact output does); PyJetscape gets `core.soft_primaries_numpy` and a capability flag
(`core.ISS_DECAYS_SMASH_LIST`), which the warnings use.

### 1.6 Writers, precision, attributes (1 day) — done

- `hadronize.py` and `run_prod_oo_isr.py` write the primaries as separate **files** (one tag
  per file, as the hadron format has it), tags `bulk_bg_primary` and `bulk_jet_primary`, with
  the layout of the final files (`pid`, `pstat`, `p`, `x`, sample and unit offsets). Existing
  readers never see them.
- **Precision per group.** SMASH needs the primaries' positions and full float32 momenta: the
  mass comes from E² − p², and with 12 mantissa bits a 5 GeV pion's m² is ~25 % off.
  `--keep-bits-p`, `--keep-bits-x` and `--no-x` then apply to the final groups only, and the
  primaries keep full precision (or get their own options).
- Attributes: `decayed` (final groups: true; primaries: false), `hadron_list` (`SMASH` /
  `UrQMD`), `iss_commit`, `iss_decays` (`iSS, Breit-Wigner masses ...` / `iSS, pole masses`).
- An option for the setting: `hadronize.py --keep-primaries`, `run_prod_oo_isr.py
  --keep-primaries` (writes them for the in-job bulk), passed into the iSS block of
  `hadronize.xml`.

### 1.7 EOS 91 in the XMLs, the scripts and the warnings (½ day) — done

- `<EOS>91</EOS>` in the production XMLs (`prod_AuAu_0_10_jet`, `prod_AuAu_0_10`, `prod_OO_isr`;
  decision 1), and an `--eos 9|91` option in `run_prod_jet.py` and `run_prod_oo_isr.py` for
  reruns of EOS 9 campaigns; the XML comments about EOS 91 updated.
- **Places that assume EOS 9 today:**
  - `run_prod_jet.py:785` writes `eos_kind = "hotqcd (MUSIC EOS 9)"` whatever the XML says: take
    it from `<EOS>` (and record the EOS number as its own attribute);
  - `analysis/wake_observables.py:184` (`find_eos`) falls back to the EOS 9 table
    (`hrg_hotqcd_eos_binary.dat`): choose the table by the file's EOS, as `run_prod_jet.py:235`
    does, and refuse a mismatch with an explicit `--eos` table;
  - the defaults of `hydro_hist_vs_surface/make_surfaces.py:79` and
    `prod_AuAu_0_10_jet/music_cpu_gpu_hydro.py:42` (validation scripts; the same change);
  - [`PLAN_analysis_decays.md`](PLAN_analysis_decays.md) reads the UrQMD table: with EOS 91 the
    decay table is `pdg-SMASH.dat`, where η and Σ⁰ are stable (decision 4). Not changed yet:
    that plan is not implemented.
- The warnings in `hadronize.py:569` and `run_prod_oo_isr.py:457` only for a build without
  `core.ISS_DECAYS_SMASH_LIST`.

### 1.8 Validation (1–2 days) — done 2026-10-15 (GB10; 2-event Au+Au and O+O jobs)

- **Decays** (a standalone test of `particle_decay` with the SMASH list, 200 000 decays each of
  ω, η′, f₂, K₁ in flight): four-momentum conserved to 2·10⁻⁷ GeV, the sampled daughter masses
  always below M, no `exit`, no undecayed mother; ω → πρ takes its 90 %, and the ρ masses
  follow the truncated Breit–Wigner (KS distance 0.0036, 95 % bound 0.0055); f₂ → ρρ (two
  unstable daughters) works.
- **Primaries → finals** (instead of a replay of the decays): per kept sample, the summed
  four-momentum of the decayed hadrons equals the primaries' to 1–5·10⁻⁵ (float32 storage)
  and Q, B, S agree exactly; Au+Au jet leg and background, plain and `--correlated`, and O+O.
  Only stable species remain (15: γ, π, K, N, Λ, Σ, Ξ, Ω, η, Σ⁰).
- **Bit-identity:** EOS 9 hadronize.py output (all three tags, plain and `--correlated`)
  identical to the build before; EOS 9 and EOS 91 decayed hadrons identical with and without
  `--keep-primaries`; O+O EOS 9 in-job bulk identical. The O+O jet hadrons differ, as they
  must (per-event reseed).
- **O+O replay:** `rehadronize_jets.py --check` reproduces the job's jet hadrons exactly from
  the stored input and seeds; `--decays off` gives the primary hadrons.
- **Physics comparison EOS 91 vs EOS 9:** not needed (2026-10-15).
- **Speed:** not measured at production scale (the 2-event tests: 3.3 s with EOS 91 against
  3.0 s with EOS 9 for 5 oversamples per surface).

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

**Pythia versions.** The parton vertices came in 8.230 (Oct 2017, for rope hadronization;
rewritten in 8.303, Sep 2020). Between the image's 8.312 and 8.316 `PartonVertex.h` and the
vertex documentation are unchanged apart from the copyright line, and the update notes of
8.313–8.316 list no change to them: 2.3 works with 8.312.

**Beyond Pythia's intended use.** Pythia's docs describe the parton vertices as transverse
(MPI) offsets "still assumed to occur at t = z = 0", inherited by the hadrons with "an
interpolation … in case of strings stretched between partons from different vertices".
`vertexHadrons` interpolates all four components of `vProd`, so our hadronization points with
t, z ≠ 0 work arithmetically, but outside the documented use: the validation checks that
primaries land between their partons' hadronization points, not at t = 0.

**Compared with JETSCAPE's Hybrid Hadronization** (`HybridHadronization.cc`, the same in
JETSCAPE `main` and X-SCAPE), which does not use Pythia's vertex settings
(`PartonVertex:setVertex` is commented out, `:479`) and places the hadrons itself after Pythia,
by colour tracing:

| | Hybrid Hadronization | this plan (Pythia's own) |
|---|---|---|
| partons' hadronization point | later of production point and T_c crossing, at least `part_prop` | the same rule (2.2) |
| who places the hadrons | HH after Pythia (`set_spacetime_for_pythia_hadrons`, ~600 lines) | Pythia during the fragmentation, from the partons' `vProd` |
| along a string segment | **evenly spaced** by order in the list, at 1/(n+1) … n/(n+1) between the two partons (`:10060–10090`) | **energy-weighted**: each hadron's energy midpoint along the whole string, gluons counted half to each side |
| one gluon / one-hadron string / junction | the gluon's position / the average / the legs' centre | covered by the same interpolation / the midpoint / along each leg |
| untraceable topology | (0,0,0,0) with a warning | Pythia logs an error and leaves it unshifted: also at 0 |
| decay products | the string's position (traced back through the mothers) | Pythia's decay vertex; the primaries' positions are stored, which is what SMASH uses |
| formation time | none; optional `had_prop` (default 0) moves every hadron by a fixed proper time, adding γ·`had_prop` to the position but only `had_prop` to the time (`:1203–1215`), which looks inconsistent | optional: Lund breakup space-time (2.4) |
| recombination hadrons | constituents' boosted centre-of-mass position (`pos_lab`) | none: strings only (Colorless, iColored), as in production today |

The same coarse picture (hadrons between their string's partons' hadronization points), with
differences at the fm level along a string.

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
- **needs Pythia ≥ 8.315:** 8.315 (May 2025) fixed "a rare bug in
  `StringFragmentation::setHadronVertices` that caused a segmentation fault when the string
  endpoints are c or b quarks" and added a quark-mass offset correction. Our jets have c and b
  quarks (heavy flavour in Pythia's hard QCD, in Matter and LBT), so with the image's 8.312 a
  job can rarely crash. 8.315 is also what SMASH 3.2.2 needs (Part 3): choosing 2.4 means the
  Pythia upgrade of the production. (The string-model vertices came in 8.235, Mar 2018;
  `maxSmear` in 8.244, `maxTau` in 8.303.)
- whether JETSCAPE's rescattering study uses a formation time (Pythia's, or HH's `had_prop`),
  and which: ask Hendrik Roch.

### 2.5 O+O: what is needed to redo the jet hadronization later (1 day) — minimal part done

Implemented (2026-10-15), positions left out: `--store-hadronization-input` (opt-in) stores
`pythia.event` before `pythia.next()` (pid, col, acol, E, px, py, pz, m; float64) as
`hadronization_input/` of the jet_frag file, with the per-event seed (`units/seed`, always on),
`pythia_version` and the effective `<JetHadronization>` block (`jet_hadronization_xml`);
`rehadronize_jets.py` (prod_OO_isr) is the offline driver (`--check`, `--decays off`,
`--new-seeds`). Still to do for positions: the partons' hadronization points (2.2), stored
next to the input.

The original plan:

`iColoredHadronization` dumps the complete parton list it gives Pythia (final partons, ISR
partons, beam remnants: pid, four-momentum, col, acol, t, x, y, z, the hadronization point of
2.2) into a ragged group `hadronization_input/` of the jet file (the layout of `partons/`); most
simply `pythia.event` just before `pythia.next()`, which also has the remnants' momenta as
iColored builds them from 3dMCGlauber and the colour-balancing parton. **The seed is not there
yet:** iColored sets no `Random:seed` (not in the module, not in `OO_MCGlauber_MUSIC_isr.xml`),
so every job starts from Pythia's default seed and the stream runs on through the job's events.
Exact replay needs a per-event reseed as in Colorless (`reseed_per_event`, `:125`), recorded in
the file; it changes O+O's jet hadrons (not bit-identical). Then the O+O jet hadronization can be redone
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

### 2.7 What a later jet hadronization can and cannot change (no code; ½ h in the docs)

With the final partons kept (Au+Au: the particlize file's `partons/`, or `partons/x_had` in a
hadrons-only jet file, `PLAN_AuAu_hadrons_only.md`; O+O: `hadronization_input/`, 2.5), the jet
hadronization can be redone at any time with:
- another Pythia version or tune (e.g. ≥ 8.315 for 2.4), decays off, primaries for SMASH;
- positions (2.2–2.3), the Lund formation time (2.4), other numbers of fragmentations or seeds.

It **cannot** change:
- **anything before the final partons:** Matter/LBT, the liquefier's absorption (`pstat` −11,
  −17), recoils; the partons are fixed;
- **the colour flow of Au+Au jets:** LBT assigns no colour tags (`PLAN_particlize_h5.md:179`),
  so the stored `col`/`acol` are incomplete for Matter+LBT showers, which is why Au+Au uses
  Colorless (`hadronize.xml:21`). A colour-based hadronization would have to start in LBT;
- **medium-dependent hadronization without the surfaces:** Hybrid Hadronization samples thermal
  partons from the hydro's hypersurface (`ThermalPartonSampler`, `HybridHadronization.cc:727–830`)
  and moves partons to T_c with the hydro's temperatures. Offline it needs the stored surfaces
  (a SurfaceReplay-like path, and the surface-based variant of 2.2), and our surface is at
  T_fo = 0.15 GeV, not necessarily at HH's `hydro_Tc`. Not possible from hadrons-only Au+Au
  files or O+O runs with `--no-particlize`.

**Reproducing the original** needs the same seed (recorded, `units/seed`) and the **same Pythia
version**: 2.1's writer records Pythia's version (`PYTHIA_VERSION_INTEGER` of the build, e.g. 8312) in every
jet hadron file, next to the seeds.

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
| **EOS 91** | stage-1 XML (the default, 1.7) | job XML (the default, 1.7) |
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
| 1.5–1.7 wrapper, binding, writers, options | 2½ |
| 1.8 validation | 1–2 |
| **Part 1** | **6½–8½** |
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

- Decision 3 above, and whether the EOS 91 tunes ran SMASH (decision 1).
- Hybrid Hadronization's `hydro_Tc` is 0.16 GeV in X-SCAPE (`jetscape_main.xml:349`); 2.2 uses
  our T_fo = 0.15 GeV, so the jet hadrons start on the bulk's surface (recorded in the files).
- **Decision 3 can move out of the production:** the Au+Au jet hadronization is redone from the
  stored partons anyway, and O+O's once 2.5 exists. Then the production needs only 2.2 (O+O: in
  the job, with the hydro) and 2.5; 2.1, 2.3 and 2.4 run in the offline redo, with Pythia ≥ 8.315.
- **Negative hadrons:** Au+Au stage 2 runs `take_recoil 1` (`hadronize.xml:78`), so Colorless
  turns holes into negative hadrons (`ColorlessHadronization.cc:519`). SMASH can't propagate
  negative-weight particles: keep them out of SMASH and subtract them afterwards, unscattered?
  Ask Hendrik Roch with the formation-time question.
- **Jet sample k with bulk sample k:** `--n-frag` is 10 by default against 100–200 bulk
  oversamples, and the reader pairs `k mod n_frag`, so the same jet hadrons would go into 10–20
  SMASH events. Either n_frag = the number of samples that get SMASH, or accept the correlation.
- **How many oversamples keep primaries:** full float32 with positions roughly doubles the bulk
  hadron storage per kept sample. `--keep-primaries N` exists (1.6); the campaign value of N
  comes from 3.0's timing (how many samples get SMASH).
- **Au+Au correlated sampling after SMASH** (Follow-ups) before Parts 2–3 for Au+Au: the
  `--keep-primaries` files of a few EOS 91 events can go through standalone SMASH (one jet leg
  and its background) now, together with 3.0's timing.
- **Jet and bulk coordinates:** the partons' `x_in` and the hydro/iSS positions in the same
  frame (origin, t against τ₀), for both systems; a check in 2.6.
- **The surface crossing of 2.2 (Au+Au):** the last crossing of a straight worldline with a
  set of discrete surface cells is not an inside/outside lookup; it needs an algorithm and a
  tolerance, and is the riskiest estimate of Part 2.
- **The SMASH version of iSS's `pdg-SMASH.dat`:** EOS 91's table, iSS's sampling and SMASH
  3.2.2's `particles.txt` should have the same species and masses (3.2's pid check).
- `ColoredHadronizationSinglet` (`cparker/eA-update`) sets the partons' `vProd` but neither
  `PartonVertex:setVertex` nor `Fragmentation:setVertices` (none found on the branch), so its
  primary hadrons should come out at (0,0,0,0) and the "infinity" ones from decay vertices: ask
  Cameron Parker / erickolbusz whether that is what they see.
- Whether conda-forge has Pythia 8.315 (for 3.1), or it is built from source in the image.
- Whether X-SCAPE upstream wants 1.1–1.2 and 2.1–2.3 (they change shared modules): kept
  behind switches that default to today's behaviour (`decay_smash_list`, `keep_primaries`,
  `<reseed_per_event>` and `<store_input>` for iColored, all 0 by default).
- **FastHydro's particlization** (`contribs/FastHydro`) still refuses `hotqcd_smash` (EOS 91)
  without an afterburner (`check_xml_agrees_with_cfg`); with `decay_smash_list` it could allow
  it. Not changed.
