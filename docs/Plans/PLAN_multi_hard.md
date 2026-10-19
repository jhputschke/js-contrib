# PLAN: extra hard scatterings per event (stage 1)

2026-10-19. Status: **decided, not implemented.** Today every event has exactly one hard
scattering: PythiaGun at one binary collision of 3dMCGlauber, with the MPI partons of that one
nucleon–nucleon collision. A Pb+Pb collision has many more: N_coll × σ(p̂_T > p₀)/σ_NN of them.
Stage 1 adds the extra **hard** scatterings (p̂_T above p₀ = 15 GeV by default), as more showers
in the existing two-stage chain. They give the model the jet background its Pb+Pb events lack
today: the fake/combinatorial jet2 of the dijet proxy (`PLAN_dijet_deta_wake.md` D2), the B of
Z+jet (`zjet/ANA_Zjet.md`: "the model's background has no other hard scatterings"), CMS-like
mixed events, and the hard tail of the 4–10 GeV underlying event. Stage 2 (semi-hard minijets,
p̂_T ~ 3–10 GeV, a change of the bulk model with string subtraction and a retune) is a later,
separate plan (§6).

## 1. Numbers

LO Pythia 8.3 at 5.02 TeV (HardQCD, ISR on, FSR and MPI off), per 0–30 % Pb+Pb event with
⟨N_coll⟩ = 1030 and σ_NN = 67.6 mb (3dMCGlauber's PDG fit, `Glauber.cpp:352-357`):

| p₀ | σ(p̂_T > p₀) | scatterings per event | partons > 2 GeV each | extra showers per event |
|---|---|---|---|---|
| 20 GeV | 0.173 mb | 2.6 | 5.0 | ~13 |
| **15 GeV (default)** | **0.525 mb** | **8.0** | **4.8** | **~38** |
| 10 GeV | 2.32 mb | 35 | 4.4 | ~160 |
| 7 GeV | 8.06 mb | 120 | 4.1 | ~510 |
| 5 GeV | 24.2 mb | 370 | 3.7 | ~1,400 |

0–10 %: × 1.55. Today an event has 2–18 showers (the trigger's Pythia event with its MPI). At
p₀ = 15 the extras add ~2–3× that; their energy is a few per cent of the event's E_T.

## 2. Decisions (2026-10-19)

| | question | decision |
|---|---|---|
| D1 | goal | staged: **stage 1 hard extras** (this plan), stage 2 minijets later |
| D2 | threshold | **p₀ an option, default 15 GeV** (10 and 20 for comparisons and cheap runs) |
| D3 | placement | **A: in the jet leg only.** MUSIC_1 = strings (as today), MUSIC_2 = strings + the droplets of the trigger and the extras. bulk_jet − bulk_bg is the response to all scatterings; no third hydro leg |
| D4 | energy budget | **no subtraction** from the strings (as for the trigger today); the extras' summed energy recorded per event |
| D5 | sampling | **N ~ Poisson(N_coll σ/σ_NN)** per event (= an independent draw per binary collision), each extra at its own binary collision drawn uniformly (∝ T_A T_B, the trigger's rule); its own Pythia instance: HardQCD above p₀, ISR on, FSR off (Matter showers), MPI off; **σ from a warm-up** at Init (recorded); `--extra-n N` fixes N for tests |
| D6 | reuse | **per jet event**: every jet event draws its own extras from its background's N_coll and collision list |
| D7 | hadronization | **two groups when extras are on**: the trigger (with its MPI partons) alone, the extras together in a second Colorless call; **per scattering** as an option. Extras off: unchanged |
| D8 | output | the extras' hadrons in a **new tag `extra_frag`**; `jet_frag` stays the trigger's alone. Per-event and per-scattering records; the scattering index on the shower initiators |
| D9 | validation | §4: off = bit-identical; on = the trigger's `jet_frag` and `bulk_bg` bit-identical to extras off; sampling, energy, physics, cost |

The trigger keeps everything it has: its window, cuts (`--trigger Z`, `--parton-ymax`), σ_k,
`pthat`, `pthat_bin`, `event_weight`, and the event weight σ_k/N_k × N_coll. The extras are
unbiased: one forced trigger plus Poisson extras is the right event for windows with μ_k =
N_coll σ_k/σ_NN ≪ 1 (the Z; dijet windows above ~50 GeV: μ(p̂_T > 100) ≈ 0.003); for low jet
windows (10–30 GeV, μ_k ~ 2–30) it is an approximation, documented.

## 3. Implementation

Branches `multi_hard` in X-SCAPE (off `contrib`) and js-contrib (off `main`). Everything is
behind an option: **off by default, and off is bit-identical to today** (§4 V1).

### 3.1 Generator (X-SCAPE `src/initialstate/PythiaGun.{cc,h}`)

The framework keeps one HardProcess pointer (`JetScape.cc:1217`, the last wins; a second
`<Hard>` would read the same XML block and seed), so the extras live inside PythiaGun.

- XML `<PythiaGun><extraHard>`: `<on>` (0), `<pTHatMin>` (15), `<sigmaNN>` (default: the PDG fit
  of 3dMCGlauber at `<eCM>`, `Glauber::compute_NN_inelastic_cross_section`), `<warmup>` (20000
  events), `<fixedN>` (−1: Poisson), `<LinesToRead>` (extra Pythia settings).
- A dedicated Pythia instance (the `extraPythia_` pattern of the windows, `PythiaGun.cc:80-91`)
  with its own seed (splitmix of the job seed with its own salt, `:61-78`), so the trigger's
  generation is untouched: HardQCD:all, `PhaseSpace:pTHatMin` = p₀, PartonLevel:ISR on, FSR
  off, MPI off, HadronLevel off.
- Init: `<warmup>` parton-level events → σ_extra and its error (Pythia's sigmaGen/sigmaErr),
  kept fixed for the job and recorded.
- ExecuteTask, after the trigger is accepted (`:515-639`) and its partons added (`:666-691`):
  μ = `ini->GetNcoll()` · σ_extra/σ_NN; N ~ Poisson(μ) from the extras' Pythia RNG (or
  `<fixedN>`); for each: `py_extra.next()`, a vertex from `ini->SampleABinaryCollisionPoint`
  (x, y; z = t = 0 as the trigger), the status-62 partons above `softMomentumCutoff` as for the
  trigger, colour tags offset per scattering, `plabel` = the scattering index (0 the trigger,
  1… the extras), appended **after** the trigger's partons.
- Per-event getters for the writers: n_extra, μ, the summed energy and p_T of the extras'
  handed-over partons, and per scattering (index, p̂_T, Pythia process code, x, y, number of
  partons). `CollectHeader` stays the trigger's.
- Binding (js-contrib `src/bind_hard_process.cc`): `extra_hard_info(gun)` → those per-event
  and per-scattering arrays; `hard_partons_numpy` gains the label column.

### 3.2 Energy loss, liquefier, MUSIC_2

No structural change: `JetEnergyLossManager` clones one JetEnergyLoss per parton
(`JetEnergyLossManager.cc:96-185`), all deposit into the one liquefier list, MUSIC_2 sums the
droplets, and the MUSIC_2 restart stays valid (checkpoint at `tau_delay`).

- **Order:** the trigger's partons first (3.1), so its showers run first and draw the same
  random numbers as without extras (the clones share their module's engine; production seeds
  are nonzero, so 3dMCGlauber's engine, which draws the extras' vertices, is a different one).
- **Risk to check early (V2):** if the clones' task numbers come from a global counter, more
  clones per event would shift the trigger's engine from the second event on. Fix if it shows:
  seed the clones from (event, shower index), or shower the extras in a second manager after
  the trigger's.
- **Cost:** MUSIC_2's jet-source gather (all cells × active droplets, on the CPU even in the GPU
  build, `music4gpu advance.cpp:491-560`) grows with the droplets: ~2–3× at p₀ = 15. Measured in
  V6; the per-droplet scatter (or a GPU kernel) is not part of stage 1 unless V6 says so.

### 3.3 Hadronization (js-contrib `src/bind_hadronization.cc`, `python/jetscape/hadronizer.py`)

Colorless itself is unchanged; the binding calls it per group.

- The scattering of every shower from its initiator's `plabel` (the final partons carry the
  shower index, `FINAL_PARTON_COLUMNS`).
- Mode `groups` (default with extras): one `DoHadronization` call with the trigger's showers →
  `jet_frag`; one with all the extras' showers → `extra_frag`. Mode `per_scattering`: one call
  per extra (all into `extra_frag`, with a per-hadron scattering index). Each group draws its
  own beam remnants where it needs them (forward hadrons).
- Seeds: the trigger's group keeps today's `unit_seed(base, prod, "jet_frag", event, s)`; the
  extras' use the tag `extra_frag`, so `jet_frag` is bit-identical to extras off.
- `n_frag` fragmentations for both tags.

### 3.4 Production driver and files (js-contrib `example/prod_AuAu_0_10_jet/`)

- `run_prod_jet.py`: `--extra-hard` (off), `--extra-pthat-min 15`, `--extra-n N`,
  `--extra-hadronize groups|per_scattering`; written into `<PythiaGun><extraHard>` by
  `configure_pythia_gun`. `run_prod_vacuum.py` rejects them (pp: one collision, Pythia's MPI).
- Per event (pair `diag/`, particlize `events/`, hadrons-only `units/`): `n_extra`, `extra_mu`,
  `extra_E`, `extra_pt` (`PARTICLIZE_DIAG`).
- Per scattering: a ragged group `scatterings/` (offsets + data; columns scatter, role, pthat,
  code, x, y, n_partons) in the pair, particlize and `jet_frag` hadron files.
- `INITIATOR_COLUMNS` gains `scatter` (appended last: readers that index the known columns keep
  working); root attributes `extra_hard`, `extra_pthat_min`, `extra_sigma`,
  `extra_sigma_err`, `extra_sigma_nn`, `extra_seed`, `extra_warmup`, `extra_hadronize`.
- Hadron files: `<stem>_hadrons_extra_frag.h5` (hadronize.py and `--hadrons-only`), with
  `initiators/` of the extras.

### 3.5 Readers (js-contrib)

- `python/jetscape/hadrons_h5.py`: `TAGS` + `extra_frag` (optional tag: files without it read
  as today); `JetEvents.jet_event` (the full event = bulk_jet k + jet_frag fk + extra_frag fk);
  `EventInfo.scatterings()`; the sample checks and seeds.
- C++ `example/analysis_root/HadronFileReader.h`: the tag in the enum, `bkg_dep_frag` adds
  extra_frag, a new `extra(e, j)`; `run_h5toROOT.py` converts it;
  `prod_AuAu_0_10_jet/root_export/check_h5_vs_root.py` checks it.

### 3.6 Analyses (js-contrib `example/analysis/`, Python and C++ twins)

The rule: **extra_frag belongs to the event, not to the trigger.**

- Full event = bulk_jet + jet_frag + extra_frag: `zjet/zjet_jets_map.py` / `zjets.cxx`,
  `zjet/zjet_mixing.py` (`file_pool`: S_full, Y_full, L_full) / `zhadron_pools.cxx`,
  `zjet/qa.py` / `zqa.cxx` (a per-source histogram for extra_frag).
- The trigger's truth stays the trigger's fragments: S_zc / Y_zc = bulk_jet − bulk_bg +
  jet_frag (the paired wake contains the extras' wakes under D3, uncorrelated with the trigger);
  B (bulk alone) becomes bulk_bg + extra_frag.
- Z+jet B_own: per event, the jets of bulk_bg k + extra_frag fk (a per-event collection in the
  map steps, as `nw`), since the extras are per jet event (D6); `bg_jets` per background stays
  for the bulk alone.
- `dijet/deta.py` takes the contents from the pools; the fake jet2 per class then counts the
  extras' jets.
- `check_cxx.py` covers the new collections and arrays.

### 3.7 Documents

`docs/README_2stage.md` (a section on extra scatterings), `prod_AuAu_0_10_jet/README.md`
(options, files), `docs/README_analysis.md` (extra_frag, the full event), `zjet/ANA_*.md` and
`dijet/ANA_Deta.md` (what changes with extras on), the config XMLs (`<extraHard>`, off).

## 4. Validation

| | check | how |
|---|---|---|
| V1 | **off = today** | the prod regression recipe (Au+Au, 2 events, reuse 2, `--write-particlize both`) and one Pb+Pb Z+jet event, every dataset bit-identical to `contrib`/`main` |
| V2 | **on: the trigger untouched** | the same jobs with `--extra-hard`: `jet_frag` and `bulk_bg` bit-identical to extras off; `bulk_jet` differs; `extra_frag` appears; checked on several events (the task-number risk of 3.2) |
| V3 | sampling | n_extra against Poisson(μ) over many events; μ ∝ N_coll; the warm-up σ against Pythia's long-run 0.525 mb at 15 GeV; the vertices against the binary-collision density; the extras' p̂_T spectrum against Pythia's inclusive one |
| V4 | energy | `extra_E` per event; `E_droplets` growing with the extras' energy loss; the liquefier's normalization (`source/flux`), `E_droplets_early`/`late` |
| V5 | physics | jets > 30 GeV (R = 0.4) in bulk_bg + extra_frag against N_coll-scaled pp inclusive jets; the dijet proxy's fake jet2 per class with extras; Z+jet B_own with extras against CMS 1702.01060's B; the 4–10 GeV underlying event in the Z–hadron noise budget |
| V6 | cost | time and memory per event at p₀ = 10, 15, 20 GeV on the GB10 (MUSIC_2's source gather, the showers, the warm-up), against today |

Unit tests: the Poisson/vertex/colour bookkeeping (C++ via the binding on a small Glauber
event), the two-group hadronization (trigger group = today's call), the readers with and
without `extra_frag`, the analysis contents with a synthetic extra_frag.

## 5. Order of work

| step | what | where |
|---|---|---|
| 1 | generator, binding, driver options and records; V1, V3 | X-SCAPE PythiaGun; js-contrib bind_hard_process, run_prod_jet, pair/particlize writers |
| 2 | hadronization in groups, `extra_frag` in the hadron files; V2 | bind_hadronization, hadronizer, hadronize.py, HadronsOnly |
| 3 | readers (Python, C++, converter) | hadrons_h5, HadronFileReader.h, run_h5toROOT |
| 4 | analyses (Python + C++ twins, check_cxx) | zjet map steps, pools, QA, dijet |
| 5 | V4–V6 on the GB10 (a few Pb+Pb Z+jet jobs, p₀ = 10/15/20) | |
| 6 | documents; PRs (one per repo, against `contrib` / `main`); then an image and the RTX box | |

## 6. Stage 2 (later): semi-hard minijets

p₀ ~ 3–7 GeV, hundreds to thousands per event: their energy is part of what the strings carry
today, so stage 2 changes the bulk model. Open there: the separation scale; subtracting the
scatterings' energy from the strings (3dMCGlauber's one-collision `Pick_and_subtract`,
`Glauber.cpp:426-564`, generalized to a list, in the two-step string generation); placement
(a third leg, or minijets as an initial-state source in both legs); the liquefier threshold
for vacuum partons (`threshold_energy_switch` 1 liquefies below 2 GeV without a temperature
check); MUSIC_2's source gather as a per-droplet scatter or a GPU kernel; a retune of the
bulk observables.
