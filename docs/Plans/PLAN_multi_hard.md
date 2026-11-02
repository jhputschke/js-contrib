# PLAN: extra hard scatterings per event (stage 1)

2026-10-19, extended 2026-10-30 (D10–D17). Status: **implemented 2026-10-30** (§7). Today every
event has exactly one hard
scattering: PythiaGun at one binary collision of 3dMCGlauber, with the MPI partons of that one
nucleon–nucleon collision. A Pb+Pb collision has many more: N_coll × σ(p̂_T > p₀)/σ_NN of them.
Stage 1 adds the extra **hard** scatterings (p̂_T above p₀ = 10–15 GeV, per system), as more showers
in the existing two-stage chain. They give the model the jet background its Pb+Pb events lack
today: the fake/combinatorial jet2 of the dijet proxy (`PLAN_dijet_deta_wake.md` D2), the B of
Z+jet (`zjet/ANA_Zjet.md`: "the model's background has no other hard scatterings"), CMS-like
mixed events, and the hard tail of the 4–10 GeV underlying event. They also enter the **bulk
observables** (Nch, [p_T], v_n; review P7 of 2026-10-30), so the extras are provided for **all
analyses and systems**: Au+Au and Pb+Pb, the new LHC small systems O+O and Ne+Ne (§3.9), and a
**minimum-bias mode** without a forced trigger (§3.8). Stage 2 (semi-hard minijets,
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

Per binary collision and energy (same settings, 20,000 events each, 2026-10-30; σ_NN 42 mb at
200 GeV, 67.6 mb at 5.02 TeV, ~68 mb at 5.36 TeV):

| √s_NN | σ(p̂_T > 10) | σ(p̂_T > 15) | extras per binary collision, p₀ = 10 / 15 |
|---|---|---|---|
| 200 GeV | 0.0058 mb | 0.00054 mb | 1.4·10⁻⁴ / 1.3·10⁻⁵ |
| 5.02 TeV | 2.31 mb | 0.52 mb | 0.034 / 0.0077 |
| 5.36 TeV | 2.54 mb | 0.58 mb | 0.037 / 0.0085 |

So Au+Au 0–10 % at 200 GeV gets μ ≈ 0.13 at p₀ = 10, and O+O at 200 GeV (the ISR production)
μ ≈ 0.007: at RHIC the extras barely matter. Central O+O / Ne+Ne at 5.36 TeV get ~1.9 at 10 GeV
and ~0.4 at 15 GeV (for N_coll ≈ 50, a rough guess, not computed; V3 records the real N_coll).

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

Extended 2026-10-30 for the bulk observables and the small systems (D2 and D5 changed):

| | question | decision |
|---|---|---|
| D2′ | threshold | **per system, in each XML**: p₀ = **10 GeV** for Au+Au, O+O, Ne+Ne (few extras), **15 GeV** for Pb+Pb; `--extra-pthat-min` overrides |
| D5′ | random numbers | the extras use **their own engines** throughout: the Pythia instance, the Poisson draw, the vertex choice, and the **showers** (the extras' Matter/LBT clones get an engine seeded from (job seed, event, scattering)). The trigger's showers then draw exactly what they draw without extras, from every event on (today the clones share the module's engine across events, `JetScapeModuleBase.cc:67-73`, and 3dMCGlauber's `SampleABinaryCollisionPoint` draws from the wrapper's engine, `MCGlauberWrapper.cc:269-276`) |
| D10 | small systems | **O+O and Ne+Ne at 5.36 TeV through the two-stage chain** (`run_prod_jet.py`, liquefier, MUSIC_2), new XMLs (§3.9). The 200 GeV O+O ISR chain (one MUSIC, no deposition) gets no extras |
| D11 | small-system bulk | the production **Pb+Pb LHC set** (3dMCGlauber + MUSIC of the 2601.17234 family) with ¹⁶O / PGCM ²⁰Ne and eCM 5360, documented as untuned for small systems |
| D12 | small-system centrality | **impact-parameter presets** (0–10 % default, minimum bias as an option), as the other XMLs; multiplicity classes come with the full flow tools (later project) |
| D13 | minimum bias | **a mode with no forced trigger** in `run_prod_jet.py` (`--hard none --extra-hard`): an event = strings + Poisson extras (N may be 0), weight 1; the jet leg holds the extras, so bulk_jet − bulk_bg is their effect on Nch, [p_T], v_n. Events with N = 0 skip MUSIC_2 and are flagged |
| D14 | legs | **two legs** (D3) now; a third leg (strings + extras) that isolates the trigger's wake is a later option, and the files are laid out so it adds a tag without changing the others |
| D15 | flow tools | **a minimal set in this project**: per event Nch, [p_T], v_n{2} with an η gap (n = 2, 3), and the paired Δ (bulk_jet − bulk_bg); enough for V5. v_n{4}, ρ(v₂², [p_T]), symmetry planes and multiplicity centrality are a later project |
| D16 | readers | **extra_frag included by default** when present: full event = bulk_jet + jet_frag + extra_frag, B = bulk_bg + extra_frag; a switch to exclude it. Files without extras read as today |
| D17 | energy | **D4 kept** for all systems: no subtraction; extra_E and the extras' fraction of the event's E_T recorded (larger in central O+O / Ne+Ne at 10 GeV) |

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

- XML `<PythiaGun><extraHard>`: `<on>` (0), `<pTHatMin>` (15; the production XMLs set their
  system's p₀, D2′), `<sigmaNN>` (default: the PDG fit
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
  `<fixedN>`); for each: `py_extra.next()`, a vertex drawn uniformly from the binary-collision
  list **with the extras' engine** (D5′: a new `InitialState` accessor for the list, or
  `SampleABinaryCollisionPoint` with an engine argument; 3dMCGlauber's engine is not touched)
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

- **Order:** the trigger's partons first (3.1), so its showers run first.
- **Own engines for the extras' showers (D5′).** Checked 2026-10-30: a module's engine is made
  once from its task number (`JetScapeModuleBase::GetMt19937Generator`, `JetScapeTaskSupport.cc:91-116`)
  and the per-parton clones share it across events, so extras showered from the same engine
  would shift the trigger's showers from the second event on. Fix in X-SCAPE: a
  `SetMt19937Generator` on `JetScapeModuleBase`; `JetEnergyLoss` gives the clones of a parton
  with `plabel` > 0 (Matter, LBT and their sub-clones) an engine seeded from (job seed, event,
  scattering index) by the same splitmix as the gun; the trigger's clones keep the module's
  engine. Off (no extras): nothing changes.
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

- `run_prod_jet.py`: `--extra-hard` (off), `--extra-pthat-min P` (default: the XML's),
  `--extra-n N`, `--extra-hadronize groups|per_scattering`; written into
  `<PythiaGun><extraHard>` by `configure_pythia_gun`. `run_prod_vacuum.py` rejects them (pp:
  one collision, Pythia's MPI).
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
  as today); `JetEvents.jet_event` (the full event = bulk_jet k + jet_frag fk + extra_frag fk,
  **included by default when present**, `extras=False` to leave it out, D16);
  `EventInfo.scatterings()`; the sample checks and seeds. A minimum-bias file (D13) has no
  `jet_frag`: the full event is bulk_jet + extra_frag.
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
(options, files, the minimum-bias mode, the small systems), `docs/README_analysis.md`
(extra_frag, the full event, the flow tools), `zjet/ANA_*.md` and `dijet/ANA_Deta.md` (what
changes with extras on), the config XMLs (`<extraHard>`, off, with the system's p₀).

### 3.8 Minimum-bias mode (D13)

- `run_prod_jet.py --hard none --extra-hard`: PythiaGun skips the trigger (a `<trigger>off`
  / `<noTrigger>` switch in `<PythiaGun>`, so σ_k windows, `pthat`, `pthat_bin` are absent and
  `event_weight` = 1) and hands over only the extras; `--extra-hard` is required with
  `--hard none`. `--pthat-bins`, `--trigger`, `--parton-ymax` are rejected.
- The two legs as usual; an event with N = 0 has no droplets: MUSIC_2 is skipped and bulk_jet
  is written as bulk_bg's copy with the flag `no_extras` (or a reference, decided in step 1 by
  what the writers allow), so the per-event Δ is exactly 0 there.
- The files carry `mode = "minbias"`; the readers and the flow tools (§3.10) accept them; the
  jet analyses refuse them (no trigger).
- The centrality range comes from the XML's b preset (D12), `--bmin/--bmax` if the driver has
  them, else the XML.

### 3.9 Small systems at the LHC (D10–D12)

- New user XMLs in `example/config/`: `OO_5360_jet.xml` and `NeNe_5360_jet.xml`, copies of
  `PbPb_0_10_jet.xml` (the production Pb+Pb set, D11) with 3dMCGlauber's ¹⁶O (α-cluster
  configurations, as the O+O ISR XML) and PGCM ²⁰Ne (`Nucleus.cpp:507-528`), eCM 5360,
  σ_NN at 5.36 TeV, `<extraHard><pTHatMin>` 10 (D2′), b presets for 0–10 % and minimum bias
  (D12; the b edges from a 3dMCGlauber minimum-bias sample, recorded in the XML header).
- The MUSIC grid and `grid_*.yaml` sized for the small fireball (as `grid_oo.yaml`), the
  liquefier unchanged.
- A short note in `prod_AuAu_0_10_jet/README.md`: untuned for small systems (no LHC O+O / Ne+Ne
  tune; compare dNch/dη and v₂{2} with ALICE/CMS/ATLAS O+O and Ne+Ne before quoting numbers).

### 3.10 Minimal flow and bulk tools (D15)

- `example/analysis/bulk/` (Python; a C++ twin only if the ROOT path needs it): per event
  Nch and [p_T] in |η| < 0.8 (and a configurable window), v_n{2} for n = 2, 3 from Q-vectors with
  an η gap (two sub-events, |Δη| > 1 by default), for bulk_jet (+ jet_frag + extra_frag, the full
  event), bulk_bg, and the paired Δ = full − bulk_bg.
- Class averages: ⟨Nch⟩, ⟨[p_T]⟩, σ([p_T]), v_n{2} with jackknife errors; works on Pb+Pb,
  Au+Au, O+O, Ne+Ne and minimum-bias files.
- Tests on synthetic events with an imposed v₂ and a known [p_T].

## 4. Validation

| | check | how |
|---|---|---|
| V1 | **off = today** | the prod regression recipe (Au+Au, 2 events, reuse 2, `--write-particlize both`) and one Pb+Pb Z+jet event, every dataset bit-identical to `contrib`/`main` |
| V2 | **on: the trigger untouched** | the same jobs with `--extra-hard`: `jet_frag` and `bulk_bg` bit-identical to extras off **on every event** (the own engines of D5′); `bulk_jet` differs; `extra_frag` appears |
| V3 | sampling | n_extra against Poisson(μ) over many events; μ ∝ N_coll; the warm-up σ against the table of §1 (5.02 / 5.36 TeV, 10 / 15 GeV); the vertices against the binary-collision density; the extras' p̂_T spectrum against Pythia's inclusive one; the same for O+O / Ne+Ne and the minimum-bias mode |
| V4 | energy | `extra_E` per event; `E_droplets` growing with the extras' energy loss; the liquefier's normalization (`source/flux`), `E_droplets_early`/`late` |
| V5 | physics | jets > 30 GeV (R = 0.4) in bulk_bg + extra_frag against N_coll-scaled pp inclusive jets; the dijet proxy's fake jet2 per class with extras; Z+jet B_own with extras against CMS 1702.01060's B; the 4–10 GeV underlying event in the Z–hadron noise budget; **bulk (§3.10)**: ΔNch, Δ[p_T], Δv_n{2} from the minimum-bias pairs in Pb+Pb, O+O, Ne+Ne against extra_E, and the full-event v_n{2} with and without extras |
| V6 | cost | time and memory per event at p₀ = 10, 15, 20 GeV on the GB10 (MUSIC_2's source gather, the showers, the warm-up), against today; Pb+Pb at 10 GeV decides whether the source gather needs work in this project |
| V7 | small systems | O+O / Ne+Ne 5.36 TeV, a few events each: the chain runs, dNch/dη and ⟨p_T⟩ at mid-rapidity sane against the published O+O / Ne+Ne values (untuned, D11), N_coll and μ recorded |

Unit tests: the Poisson/vertex/colour bookkeeping (C++ via the binding on a small Glauber
event), the two-group hadronization (trigger group = today's call), the readers with and
without `extra_frag`, the analysis contents with a synthetic extra_frag.

## 5. Order of work

| step | what | where |
|---|---|---|
| 1 | generator (extras, own engines for the Poisson draw and the vertices, the no-trigger switch), the shower engines (D5′), binding, driver options and records; V1, V3 | X-SCAPE PythiaGun, InitialState/MCGlauberWrapper, JetScapeModuleBase, JetEnergyLoss; js-contrib bind_hard_process, run_prod_jet, pair/particlize writers |
| 2 | hadronization in groups, `extra_frag` in the hadron files; V2 | bind_hadronization, hadronizer, hadronize.py, HadronsOnly |
| 3 | minimum-bias mode (§3.8) and the O+O / Ne+Ne 5.36 TeV XMLs (§3.9); V7 | run_prod_jet, config/ |
| 4 | readers (Python, C++, converter), extra_frag by default | hadrons_h5, HadronFileReader.h, run_h5toROOT |
| 5 | analyses (Python + C++ twins, check_cxx) and the minimal flow tools (§3.10) | zjet map steps, pools, QA, dijet, analysis/bulk |
| 6 | V4–V6 on the GB10 (a few Pb+Pb Z+jet jobs at p₀ = 10/15/20; minimum-bias Pb+Pb, O+O, Ne+Ne) | |
| 7 | documents; PRs (one per repo, against `contrib` / `main`); then an image and the RTX box | |

## 6. Stage 2 (later): semi-hard minijets

p₀ ~ 3–7 GeV, hundreds to thousands per event: their energy is part of what the strings carry
today, so stage 2 changes the bulk model. Open there: the separation scale; subtracting the
scatterings' energy from the strings (3dMCGlauber's one-collision `Pick_and_subtract`,
`Glauber.cpp:426-564`, generalized to a list, in the two-step string generation); placement
(a third leg, or minijets as an initial-state source in both legs); the liquefier threshold
for vacuum partons (`threshold_energy_switch` 1 liquefies below 2 GeV without a temperature
check); MUSIC_2's source gather as a per-droplet scatter or a GPU kernel; a retune of the
bulk observables.

## 7. Status and validation (2026-10-30)

Implemented on `multi_hard` in X-SCAPE (off `contrib` c7bcbe01) and js-contrib (off `main`
5afbc7e). What differs from §3:

- **Random streams (D5′), three leaks found and closed**, each checked by V2: (1) the clones
  shared their module's engine; (2) Matter's `ran0` kept its state in function statics shared
  by every Matter instance; (3) every clone registers a task number from the global counter,
  and a module seeds its engine from it, so more clones in one event shifted the trigger's
  engines from the next event on. X-SCAPE: `JetScapeModuleBase::UseOwnStreams(scatter, seed)`
  (an engine; Matter overrides it for its `ran0` state, which is now a per-instance handle,
  shared by default as before), called by `JetEnergyLossManager::SeparateExtraEngines` for
  the extras' clones, and their copies take task numbers from a range of their own (from
  2^28) with the counter put back. Colour tags are counted per scattering
  (`HardProcess::NextColorTag(scatter)`), the trigger's floor from its own partons.
- **N = 0 without a trigger:** the energy loss is skipped (`JetEnergyLossManager` would stop
  the run without a parton); MUSIC_2 still runs (it restarts from MUSIC_1's checkpoint, so it
  is cheap), the event is not flagged otherwise than by `n_extra` = 0 (§3.8 said "skipped").
- **Vertices:** `InitialState::SampleABinaryCollisionPointWith(rng, ...)` (MCGlauberWrapper:
  uniform over its binary-collision list) instead of a list accessor.
- **Files:** `initiators/` keeps its 11 columns; the scattering of each initiator is a field
  of its own, `initiators/scatter` (particlize and extra_frag files), so files without extras
  are unchanged. `hard_partons_numpy` has the 8th column `scatter`.
- **O+O / Ne+Ne:** 3dMCGlauber's name for neon is `Ne20`. The b edges from the 0–10 % quantile
  of b in 80,000 minimum-bias events (the centrality tool's σ_inel line assumes every b < 10 fm
  collides, which fails for small nuclei).

| | result |
|---|---|
| V1 | extras off: every dataset of the pair and particlize files bit-identical to `contrib`/`main` (Au+Au 2 events reuse 2; Pb+Pb Z+jet 1 event; `source/flux` to 1e-12) |
| V2 | `--extra-n 3`/`4`: the trigger's final partons (14 columns), initiators and the background surface bit-identical to extras off in every event (3 events reuse 1; 2 events reuse 2); hadron level (Pb+Pb Z+jet): `jet_frag` and `bulk_bg` identical, `bulk_jet` differs |
| V3 | IS + PythiaGun only, 2,000 Pb+Pb 0–30 % (p₀ 15) and 3,000 O+O 0–10 % (p₀ 10) events: ⟨N⟩ − ⟨μ⟩ = +0.07 ± 0.06 / −0.02 ± 0.02, (N − μ)/√μ rms 1.01 / 1.00, P(N = k) χ² 22.6/22 / 2.5/7; μ/N_coll = σ/σ_NN exactly; warm-up σ 0.524 mb (5.02 TeV, > 15), 2.52 mb (5.36 TeV, > 10), as §1; fraction above 20 (Pb+Pb) / 15 GeV (O+O) 0.330 / 0.236 against Pythia's σ ratios 0.331 / 0.228; vertices: ⟨r⟩ 3.00 fm against the triggers' N_coll-weighted 2.98 ± 0.03 |
| V4 | `extra_E`, `extra_pt` per event (E includes the extras' forward ISR partons: ~550 GeV per extra at 5 TeV, ΣpT ~60 GeV); droplets grow with the extras (below) |
| V5 | the bulk tools (`example/analysis/bulk/`) and the analyses' extras (`extra_frag` in the full event, B = bulk_bg + extra_frag) are in place; the physics comparisons need a campaign |
| V6 | Pb+Pb 0–10 %, 3 events, GB10, `--hadrons-only`: 32.1 s per event without extras; p₀ 20 / 15 / 10 GeV (μ ≈ 5 / 15 / 66): +8 / +13 / +42 %; droplets 45 → 195 / 464 / 1,300; peak memory 18.98 → 19.16 GB. MUSIC_2's source gather needs no work for stage 1 (table in `docs/README_2stage.md` §10) |
| V7 | O+O / Ne+Ne 0–10 % at 5.36 TeV, 2 events each: the chain runs, dN_ch/dη(\|η\| < 0.5) ≈ 136 / 131, ⟨pT⟩ ≈ 0.59 / 0.65 GeV, N_coll 34–49, μ 1.3–1.8 (untuned; ATLAS 2606.20257 for the comparison) |
