# The event's other hard scatterings, and the bulk and flow analysis

This guide covers the production option `--extra-hard` (each event's other hard
scatterings), its minimum-bias mode, the O+O and Ne+Ne inputs at 5.36 TeV, the files, how
analyses read the extras, and the bulk and flow tools in
[`example/analysis/bulk/`](../contribs/PyJetscape/example/analysis/bulk/README.md). The
design decisions and the full validation are in
[`Plans/PLAN_multi_hard.md`](Plans/PLAN_multi_hard.md). Running productions in general is in
[`README_2stage.md`](README_2stage.md), reading the files in
[`README_analysis.md`](README_analysis.md).

Needs X-SCAPE with PythiaGun `<extraHard>` (JETSCAPE/X-SCAPE#187, branch `multi_hard`) and
js-contrib from jhputschke/js-contrib#83 (branch `multi_hard`).

**Contents**
1. [What it is](#1-what-it-is)
2. [Running it](#2-running-it)
3. [The XML settings](#3-the-xml-settings)
4. [What the production does](#4-what-the-production-does)
5. [The files](#5-the-files)
6. [Reading the extras in analyses](#6-reading-the-extras-in-analyses)
7. [The bulk and flow analysis](#7-the-bulk-and-flow-analysis)
8. [Recipes](#8-recipes)
9. [Validation, cost and limits](#9-validation-cost-and-limits)

---

## 1. What it is

Without the option, every event of `run_prod_jet.py` has exactly one hard scattering: the
trigger, a PythiaGun event at one binary collision of the 3dMCGlauber event, together with
the MPI partons of that one nucleon–nucleon collision. A nucleus–nucleus collision has many
more: N_coll × σ(p̂T > p₀)/σ_NN of them. `--extra-hard` adds these **extras** above a
threshold p₀. Their fragments are the jet background a Pb+Pb event otherwise lacks: the
fake jets, B for Z+jet, and the hard tail of the underlying event. Their wakes are part of the
bulk: they change Nch, [pT] and vₙ, which is why the minimum-bias mode and the bulk tools
exist.

Per binary collision (LO Pythia 8.3, HardQCD, ISR on, MPI and FSR off):

| √s_NN | σ(p̂T > 10 GeV) | σ(p̂T > 15 GeV) | extras per binary collision, p₀ = 10 / 15 GeV |
|---|---|---|---|
| 200 GeV | 0.0058 mb | 0.00054 mb | 1.4·10⁻⁴ / 1.3·10⁻⁵ |
| 5.02 TeV | 2.31 mb | 0.52 mb | 0.034 / 0.0077 |
| 5.36 TeV | 2.54 mb | 0.58 mb | 0.037 / 0.0085 |

Per event:

| system | ⟨N_coll⟩ | p₀ (XML default) | extras per event |
|---|---|---|---|
| Au+Au 200 GeV, 0–10 % | ~960 | 10 GeV | ~0.13 (at RHIC they barely matter) |
| Pb+Pb 5.02 TeV, 0–10 % | ~1,800 | 15 GeV | ~15 (~66 at 10 GeV) |
| Pb+Pb 5.02 TeV, 0–30 % | 1,030 | 15 GeV | ~8 (~35 at 10 GeV) |
| O+O 5.36 TeV, 0–10 % | 42 | 10 GeV | ~1.5 |
| Ne+Ne 5.36 TeV, 0–10 % | 51 | 10 GeV | ~1.9 |

Each extra hands over ~4.5–4.8 partons (its two hard partons and the ISR partons above
2 GeV), Σp_T ~40–60 GeV.

**The model in short:** stage 1 of the plan; semi-hard minijets (p₀ ~ 3–7 GeV), whose energy
would have to come out of the strings, are a later stage.

- The extras shower in the **jet leg** (MUSIC_2) with the trigger. The background leg
  (MUSIC_1, `bulk_bg`) stays the strings alone, so `bulk_jet − bulk_bg` is the response to all
  scatterings. A third leg that isolates the trigger's wake is not built yet (D14).
- Their energy is **not subtracted** from the strings, as for the trigger (D4, D17).
- One trigger plus Poisson extras is the right event when the trigger window is rare
  (μ_k = N_coll σ_k/σ_NN ≪ 1: the Z, dijets above ~50 GeV). For low trigger windows
  (10–30 GeV) it is an approximation.

---

## 2. Running it

The options of `run_prod_jet.py` (`example/prod_AuAu_0_10_jet/`):

| option | |
|---|---|
| `--extra-hard` | switch the extras on. Off (default): the files are bit-identical to before |
| `--extra-pthat-min P0` | their threshold [GeV]; default: the XML's `<extraHard><pTHatMin>` |
| `--extra-n N` | exactly N extras per event instead of Poisson (tests) |
| `--extra-warmup N` | Pythia events for their cross section at Init (default 20,000, ~1 s at the LHC) |
| `--extra-hadronize {groups,per_scattering}` | `--hadrons-only`: all extras' showers in one Colorless call (default), or one call per scattering, with each hadron's scattering kept |
| `--hard none` | no trigger: the **minimum-bias** event, the bulk and its extras (N may be 0), weight 1. Needs `--extra-hard`; refuses `--pthat-*`, `--parton-ymax`, `--trigger`. No `jet_frag` |

Everything else works as without extras: `--reuse`, `--pthat-bins`, `--trigger Z`,
`--hadrons-only`, `--write-particlize`, `--correlated`, `run_jobs.sh`, the container
launchers. `run_prod_vacuum.py` (pp) ignores `<extraHard>`: pp has one collision and Pythia's
own MPI.

```bash
cd example/prod_AuAu_0_10_jet
# Au+Au 0-10 % with extras above 10 GeV (the XML's)
python run_prod_jet.py --events 10 --seed 1 --extra-hard --hadrons-only

# Pb+Pb 0-30 % Z+jet with extras above 15 GeV
python run_prod_jet.py --events 24 --seed 1 --user-xml ../config/PbPb_0_30_ZJet.xml \
    --grid ../config/grid_PbPb.yaml --pthat-bins 20-30,30-35,35-45,45-55,55-70,70-200 \
    --jets-per-bin 4 --lepton-ptmin 20 --lepton-etamax 2.5 --extra-hard --hadrons-only

# O+O / Ne+Ne 0-10 % at 5.36 TeV, with a trigger and extras
python run_prod_jet.py --events 10 --seed 1 --user-xml ../config/OO_5360_jet.xml \
    --grid ../config/grid_oo.yaml --extra-hard --hadrons-only
python run_prod_jet.py --events 10 --seed 1 --user-xml ../config/NeNe_5360_jet.xml \
    --grid ../config/grid_oo.yaml --extra-hard --hadrons-only

# minimum bias (no trigger), for the bulk observables: correlated legs, several samples
python run_prod_jet.py --events 30 --seed 1 --user-xml ../config/OO_5360_jet.xml \
    --grid ../config/grid_oo.yaml --hard none --extra-hard --hadrons-only \
    --oversample 10 --correlated

# a two-stage campaign: particlize files, then hadronize.py (which handles the extras itself)
python run_prod_jet.py --events 10 --seed 1 --extra-hard --write-particlize both --particlize-only
python hadronize.py out/AuAu_0_10_jet_seed0001_particlize.h5 --oversample 20 --n-frag 10
```

The job prints the extras' settings once, and their number per event:

```
  extras   pTHat > 15 GeV: sigma = 0.524 +- 0.002 mb (20000 warm-up events, Pythia seed 234669607), sigma_NN 67.57 mb, N ~ Poisson(N_coll sigma/sigma_NN)
prod_AuAu_0_10_jet: event 1/1 done in 40.5 s, ..., N_coll 1294, ..., 10 extra scattering(s) (mu 10.04, 3342 GeV)
```

The energy in that line, `extra_E`, includes the extras' forward ISR partons, which carry
most of it. `extra_pt` is the better measure of what they bring to mid-rapidity.

**File names.** The driver's default name is `AuAu_0_10_jet[_<campaign>]_seed<NNNN>`, whatever
the XML, so the O+O, Ne+Ne and Pb+Pb runs above are written as `AuAu_0_10_jet_*`. Name them
yourself, as for Pb+Pb ([`README_2stage.md`](README_2stage.md) §9): `--out` for a single job,
`TAG_PREFIX` for a campaign through `run_jobs.sh` (it must end in `_seed`; one prefix per
OUTDIR, which is how a campaign resumes):

```bash
python run_prod_jet.py --events 10 --seed 1 --user-xml ../config/OO_5360_jet.xml \
    --grid ../config/grid_oo.yaml --extra-hard --hadrons-only --out OO_5360_seed0001.h5
TAG_PREFIX=OO_5360_seed ./run_jobs.sh --campaign mb1 20 30 1 out_oo_mb1 \
    --user-xml ../config/OO_5360_jet.xml --grid ../config/grid_oo.yaml \
    --hard none --extra-hard --hadrons-only --oversample 10 --correlated
                                                  # -> OO_5360_mb1_seed0001.* ...
TAG_PREFIX=NeNe_5360_seed ./run_jobs.sh --campaign mb1 20 30 1 out_nene_mb1 \
    --user-xml ../config/NeNe_5360_jet.xml --grid ../config/grid_oo.yaml \
    --hard none --extra-hard --hadrons-only --oversample 10 --correlated
```

`launch_2gpu.sh` and `slurm_prod_array.sh` pass `TAG_PREFIX` into the container.

**The O+O and Ne+Ne XMLs** (`example/config/OO_5360_jet.xml`, `NeNe_5360_jet.xml`) are copies
of `PbPb_0_10_jet.xml`: 3dMCGlauber's ¹⁶O / PGCM ²⁰Ne configurations (`Ne20` is
3dMCGlauber's name for neon), √s_NN = 5.36 TeV, MUSIC's transverse grid ±12 fm, b < 2.09 /
2.41 fm for 0–10 %. Those edges are the 10 % quantile of b in 80,000 minimum-bias 3dMCGlauber
events; set `b_max` to 20 for minimum bias. The bulk is **untuned** for small systems. Two
test events give dN_ch/dη(|η| < 0.5) ≈ 136 (O+O) and 131 (Ne+Ne), ⟨pT⟩ ≈ 0.59 / 0.65 GeV:
compare with ATLAS (arXiv:2606.20257) and ALICE before quoting numbers.

---

## 3. The XML settings

`<Hard><PythiaGun><extraHard>` (defaults in X-SCAPE's `config/jetscape_main.xml`; the driver
writes `on`, `noTrigger`, and `pTHatMin` / `fixedN` / `warmup` when given):

| tag | default | |
|---|---|---|
| `<on>` | 0 | 1: the extras on |
| `<noTrigger>` | 0 | 1: no triggering scattering, the extras alone (`--hard none`). Refuses `<pTHatBins>`, `<partonYMax>`, `<trigger>` |
| `<pTHatMin>` | 15 | their p̂T threshold [GeV]. The production XMLs set their system's: 10 (`AuAu_MCGlauber_MUSIC_0_10_jet.xml`, `OO_5360_jet.xml`, `NeNe_5360_jet.xml`), 15 (`PbPb_0_10_jet.xml`, `PbPb_0_30_ZJet.xml`) |
| `<sigmaNN>` | empty | σ_NN^inel [mb]; empty: 3dMCGlauber's parametrisation at `<eCM>`, the one its N_coll is made with (40.8 mb at 200 GeV, 67.6 at 5.02 TeV, 68.3 at 5.36 TeV) |
| `<warmup>` | 20000 | Pythia events at Init for σ(p̂T > p₀) (Pythia's `sigmaGen`, error `sigmaErr`); kept fixed for the job |
| `<fixedN>` | −1 | ≥ 0: exactly that many per event (tests); < 0: Poisson |
| `<LinesToRead>` | empty | more Pythia settings for the extras' instance only (not the trigger's `<LinesToRead>`: that may hold a Z process) |

The extras' Pythia: `HardQCD:all = on`, `PhaseSpace:pTHatMin` = p₀, ISR on, MPI off, FSR as
`<FSR_on>` (off in production: Matter showers), hadron level off, the trigger's `<eCM>` and
`<softMomentumCutoff>`.

**Choosing p₀.** Lower thresholds put more of the hard tail into the bulk and cost more
(Pb+Pb 0–10 %, §9): +13 % per event at 15 GeV, +42 % at 10 GeV. Below ~10 GeV the extras'
energy becomes a sizeable part of what the strings already carry. That is stage 2 (minijets
with subtraction and a retune), not this option.

---

## 4. What the production does

Per jet event (also per jet event under `--reuse`: every jet event draws its own extras from
its background's N_coll and binary collisions):

1. PythiaGun makes the trigger exactly as without extras: the same Pythia stream, windows,
   cuts, vertex and partons.
2. μ = N_coll σ/σ_NN; N ~ Poisson(μ) from the extras' own random engine (or `<fixedN>`).
3. Each extra: one event of the extras' Pythia instance (own seed), a vertex drawn uniformly
   from the background's binary collisions (`InitialState::SampleABinaryCollisionPointWith`,
   with the extras' engine), its status-62 partons above `softMomentumCutoff` handed over
   after the trigger's, with `plabel` = its scattering index (1, 2, …) and colour tags offset
   by 100000 × index.
4. Matter + LBT shower all partons in MUSIC_1's medium; the liquefier collects all droplets;
   MUSIC_2 (the jet leg) gets them all. MUSIC_1 is unchanged.
5. Hadronization: the final partons are split by their shower's scattering. The trigger's go
   to `jet_frag`, the extras' to `extra_frag`.

**The trigger is untouched.** Everything random the extras do runs on streams of their own:

- their Pythia instance, the Poisson draw and the vertices;
- their shower clones' engines, and Matter's `ran0` state, per extra parton
  (`JetScapeModuleBase::UseOwnStreams`, called by `JetEnergyLossManager`);
- the task numbers of their energy-loss copies, which come from a range of their own (modules
  seed their engines from the task number, so more copies would otherwise shift the
  trigger's engines from the next event on);
- their colour tags, counted per scattering (`HardProcess::NextColorTag(scatter)`);
- their hadronization seeds (tag `extra_frag`).

So with the same seed, a job with extras has the same trigger showers, `jet_frag` and
`bulk_bg` as a job without them, bit for bit, in every event. Only `bulk_jet` (their wakes)
and the new records differ. Comparing the two is a paired measurement of what the extras do.

**Minimum bias** (`--hard none`): step 1 is skipped. An event with N = 0 has no hard parton:
the energy loss is skipped, MUSIC_2 still runs (restarted from MUSIC_1's checkpoint, so
cheap), and `bulk_jet` differs from `bulk_bg` only by sampling.

**Hadronization of the extras** (`--extra-hadronize`, `hadronize.py --extra-hadronize`):

- `groups` (default): all extras' showers of an event in one Colorless call, `--n-frag` times.
- `per_scattering`: one Colorless call per scattering (each closes its own strings, with
  beam remnants where needed); every hadron keeps its scattering (`hadrons/scatter`).

---

## 5. The files

| where | what |
|---|---|
| per event: pair `diag/`, particlize `events/`, hadron `units/` | `n_extra`, `extra_mu` (= N_coll σ/σ_NN), `extra_E`, `extra_pt` (summed E and scalar pT of the extras' handed-over partons) |
| `scatterings/data` + `offsets` (pair, particlize, `extra_frag`) | one row per scattering, columns `scatter, pthat, code, x, y, n_partons, E, pt` (the trigger's first, scatter 0; none with `--hard none`); `code` is Pythia's process code, x, y [fm] the binary collision |
| `initiators/scatter` (particlize, `extra_frag`) | each shower initiator's scattering, aligned with `initiators/data` |
| `<stem>_hadrons_extra_frag.h5` | the extras' fragments: the layout of `jet_frag` (unit = event, `--n-frag` samples), plus `scatterings/`, `initiators/` (the extras'), `partons/` (hadrons-only jobs), `hadrons/scatter` (per_scattering) |
| `<stem>_hadrons_jet_frag.h5` | the trigger's fragments alone (absent with `--hard none`) |
| attributes | `extra_hard`, `extra_no_trigger`, `extra_pthat_min`, `extra_sigma`, `extra_sigma_err`, `extra_sigma_nn`, `extra_warmup`, `extra_fixed_n`, `extra_seed`, `extra_rng_seed`; `prod_extra_hard`, `prod_extra_pthat_min`, `prod_extra_hadronize`, `prod_extra_no_trigger` (`prod_extra_n`); `extra_hadronize` on extra_frag |
| `<stem>.json` | `extra_hard`, `extra_pthat_min`, `extra_sigma(_err)`, `extra_sigma_nn`, `n_extra_total` |
| `pyjetscape_core.hard_partons_numpy` | 8th column `scatter`; `extra_hard_info(gun)` gives all of the above per event |

Files without extras have none of these: their layout is unchanged.

---

## 6. Reading the extras in analyses

**The extras belong to the event, not to the trigger.** By default the full event is
`bulk_jet + jet_frag + extra_frag`, and "the event without its trigger" (B) is
`bulk_bg + extra_frag`. The trigger's truth (e.g. Z–hadron S_zc, Y_zc = bulk_jet − bulk_bg +
jet_frag) stays the trigger's. Every place that builds the full event has a switch to leave
the extras out. A campaign without `extra_frag` reads exactly as before.

| | with the extras (default) | without |
|---|---|---|
| Python `HadronFileReader` / `JetEvents` | `jet_event(e, k)` (`origin` 0 bulk, 1 jet_frag, 2 extra_frag); `jet_minus_background(..., fragments=True)`; `EventInfo.scatterings()`; `has_extras()` | `extras=False` |
| `hist("extra_frag", ...)` | the extras' fragments alone | |
| C++ `HadronFileReader.h` | `jet_event`, `bkg_dep_frag` add them; `extra(e, j)`; `bkg_extra(e, k)` = B; `info(e).scatterings`, `extra_initiators`, `n_extra` | `set_extras(false)`, or `bkg_dep_frag(e, k, j, false)` |
| `run_h5toROOT.py` | converts `extra_frag` when present; events columns `n_samples_extra`, `seed_extra`, `scat_*`, `xini_*` | |
| Z+jet map step `zjet_jets_map.py` / `zjets` | S = bulk_jet + jet_frag + extra_frag; B per event = bulk_bg + extra_frag (collection `bgx`) | `--no-extras` / `--extras 0` |
| Z–hadron pools `zjet_mixing.py`, `mix.py` / `zhadron_pools` | the full-event contents with the extras; B = bulk_bg + extra_frag (`R_extra`) | `--no-extras` / `--extras 0` |
| QA `qa.py map` / `zqa` | `extra_frag` histograms, `nch*_extra`, `n_extra` | `--no-extras` / `--extras 0` |
| jet QA `zjet_jets_qa.py` | B_own / B_mixed from `qa_jets`' `bgx_*` (every jet of the QA samples' B), as `zjet_jets.py` | via the map steps |
| `wake_iso.py map` | the measured event's noise with the extras; the contents stay trigger truth | `--no-extras` |
| `dijet/deta.py` | jet2 and the pools take the extras from the map steps (fake jet2 includes the extras' jets) | via the map steps |
| `hadron_distributions.C` | a source `extra`; `full` includes it | `extras=false` |

Minimum-bias files are refused by the analyses that need a trigger (`zjet_jets_map.py`,
`mix.py`, `qa.py`, `wake_iso.py`). Map a campaign one way: pools with and without extras
refuse to combine. Details per analysis: [`zjet/ANA_Zjet.md`](../contribs/PyJetscape/example/analysis/zjet/ANA_Zjet.md),
[`ANA_Zhadron.md`](../contribs/PyJetscape/example/analysis/zjet/ANA_Zhadron.md),
[`dijet/ANA_Deta.md`](../contribs/PyJetscape/example/analysis/dijet/ANA_Deta.md).

```python
from jetscape.hadrons_h5 import HadronFileReader
import numpy as np

r = HadronFileReader("out_had")                 # every production file in the folder
r.has_extras()                                  # True: the campaign has extra_frag
info = r.event_info(0)
sc = info.scatterings()                         # (S, 8): scatter, pthat, code, x, y, n, E, pt
print(sc[sc[:, 0] > 0, 1])                      # the extras' pTHat
ev = r.jet_event(0, 0)                          # bulk_jet + jet_frag + extra_frag
extras_only = ev["origin"] == 2
bins = np.linspace(0, 20, 41)
full, err = r.jet_minus_background("pt", bins, mask="charged", fragments=True)
trig, err = r.jet_minus_background("pt", bins, mask="charged", fragments=True, extras=False)
```

---

## 7. The bulk and flow analysis

[`example/analysis/bulk/`](../contribs/PyJetscape/example/analysis/bulk/README.md): per event
Nch, [pT] and two-particle vₙ with an η gap, for the full event, its parts and the paired
difference to the jet-free background; class averages with jackknife errors. It works on
every production with hadron files: Au+Au, Pb+Pb, O+O, Ne+Ne, with or without a trigger,
with or without extras.

### What is computed

Charged hadrons (`hadrons_h5.CHARGED`), 0.2 < pT < 3 GeV by default.

Each event is evaluated for four **compositions**: `full` = bulk_jet + jet_frag + extra_frag,
`no_extras` = bulk_jet + jet_frag, `bulk_jet` (the bulk with every wake), and `bulk_bg` (the
jet-free twin). A missing part is empty: without extras `full` = `no_extras`; minimum bias
has no jet_frag.

| per event | definition |
|---|---|
| `nch` | hadrons with \|η\| < η_max (0.8), the mean over the oversamples |
| `pt` | [pT] = ΣpT / N in that window, per oversample, then the mean over the oversamples |
| `pt_var` | the variance of [pT] over the oversamples (one sample's noise) |
| `c`, `w` | the two-subevent pair sum Re(Qₙᴬ Qₙᴮ*) and its pair count Mᴬ Mᴮ, means over the oversamples; A: −η_flow < η < −Δη/2, B: Δη/2 < η < η_flow (default 0.5 < \|η\| < 0.8, \|Δη\| > 1) |

| per class | definition |
|---|---|
| `nch`, `dnch_deta` | ⟨nch⟩ and ⟨nch⟩ / (2 η_max) |
| `pt` | ⟨[pT]⟩ |
| `sigma_pt` | the spread of the events' [pT] (still contains each average's sampling noise) |
| `sigma_pt_dyn` | √(sigma_pt² − ⟨pt_var / K⟩): the fluid's event-by-event [pT] fluctuation, the sampling noise removed with the K oversamples (NaN with one) |
| `sigma_pt_one` | √(sigma_pt_dyn² + ⟨pt_var⟩): the spread with one sample per event, as an experiment sees it |
| `c2`, `c3` | ⟨⟨2⟩⟩ₙ = Σ c / Σ w (multiplicity-weighted) |
| `v2`, `v3` | vₙ{2, \|Δη\| > 1} = √⟨⟨2⟩⟩ₙ (NaN if ⟨⟨2⟩⟩ₙ ≤ 0) |

| paired difference `delta_full`, `delta_no_extras`, `delta_bulk_jet` (composition − bulk_bg) | |
|---|---|
| `dnch`, `rdnch` | ⟨nch − nch_bg⟩, and that over ⟨nch_bg⟩ |
| `dpt` | ⟨[pT] − [pT]_bg⟩ |
| `dc2`, `dc3` | ⟨⟨2⟩⟩ₙ − ⟨⟨2⟩⟩ₙ,bg: the change of the correlator; it adds up, and it exists where vₙ{2} does not |
| `dv2`, `dv3` | vₙ{2} − vₙ{2}_bg |

**The flow vector's kick.** The extras also move each event's flow vector in a random
direction. Per paired difference the tools give `kick{n}` = √⟨|ΔVₙ|²⟩, measured as
⟨Re(ΔqₙᴬΔqₙᴮ*)⟩ across the η gap so the sampling noise drops out, `rkick{n}` (relative to
vₙ{2}), `align{n}` = 2⟨V·ΔV⟩ (the change along the existing flow: dilution or push) and `r{n}`,
the correlation of the flow vectors with and without the extras. The per-hadron ⟨⟨2⟩⟩ changes by
exactly align + kick², so the random part and the systematic part come apart. On the 60-event
Pb+Pb 0–10 % pilot (p₀ = 15 GeV): kick₂² = (3.1 ± 3.0)·10⁻⁶ (|ΔV₂| ≈ 0.002, ~4 % of v₂),
kick₃² consistent with 0, r₂ = 0.9991 ± 0.0009.

What a difference means depends on the production:

- **Minimum bias** (`--hard none --extra-hard`): `delta_bulk_jet` is the extras' wakes,
  `delta_full` their wakes and fragments.
- **Triggered with extras:** the trigger's and the extras' together. Compare with the same
  seeds run without `--extra-hard` (the trigger and the background are identical) to separate
  them.
- **Without extras:** the trigger's wake (`delta_bulk_jet`) and fragments (`delta_full`).

Each event also records N_coll, the number of extras and their summed E and pT (from
`extra_frag`'s `scatterings/`).

### Conventions

- **Oversamples** are independent Cooper-Frye samplings of the same fluid, not independent
  collisions. Each is treated as one physical event: pairs and [pT] are formed within one
  sample, so decay non-flow is included, as in data. The per-event values are means over the
  oversamples, and every event weighs the same whatever its number of oversamples. The legs
  may have different numbers of oversamples, since only per-event means are differenced.
- **Paired difference:** both legs are averaged over the same events and differenced in
  every jackknife replica, so the errors keep the legs' correlation. With correlated sampling
  (`--correlated`) sample k of `bulk_jet` and of its `bulk_bg` share their random numbers and
  Δ is low-noise. Without it the legs are independent samplings, and a Δ of a few per mille
  needs many samples.
- **Errors:** delete-one-block jackknife over **backgrounds**, not events. Events sharing a
  background (`--reuse`) are left out together; with `--reuse 1` the unit is the event. More
  than `--max-blocks` (100) units are grouped into contiguous blocks.
- **Classes:** a list of events, an N_coll range, or N_coll percentiles (largest first). This
  is a centrality proxy from the Glauber collision, not multiplicity classes. Multiplicity
  classes matter in small systems and are a later project.
- **Weights:** every event weighs the same. That is right for minimum-bias and single-window
  files; `--pthat-bins` campaigns would need the window weights (not done).
- Hadron files written with an `--eta-max` smaller than the cuts reach are refused.

### Command line

```bash
cd example/analysis/bulk
python bulk_obs.py out_had/                                  # every production file in it
python bulk_obs.py "OUTBASE/gpu*/*_hadrons_bulk_jet.h5" --ncoll-classes 0 10 30 50 100 \
       --out bulk.json --table bulk_events.npz
python bulk_obs.py out_had/ --eta-max 0.5 --flow-eta-max 2.4 --eta-gap 2 --harmonics 2 3 4
```

| option | default | |
|---|---|---|
| `source ...` | | directories, quoted globs or files (several allowed) |
| `--eta-max` | 0.8 | \|η\| window of Nch and [pT] |
| `--pt-min`, `--pt-max` | 0.2, 3.0 | GeV |
| `--eta-gap` | 1.0 | Δη between the two subevents |
| `--flow-eta-max` | `--eta-max` | outer edge of the subevents |
| `--harmonics` | 2 3 | n of vₙ{2} |
| `--events` | all | global events, `0:100` or `3,5,9` |
| `--ncoll-min`, `--ncoll-max` | | an N_coll range |
| `--ncoll-classes` | | N_coll percentile edges, e.g. `0 10 30 50 100` |
| `--max-blocks` | 100 | jackknife blocks (at most) |
| `--out` | | the class results, `.json` or `.npz` |
| `--table` | | the per-event table, `.npz` |
| `--quiet` | | no table on the screen |

The output on the 3-event minimum-bias O+O test job (one oversample, uncorrelated, so σ_dyn
is NaN and the vₙ are noise; it shows the layout, not physics):

```
1 production file(s), 3 events, tags bulk_jet, bulk_bg, extra_frag
  no jet_frag in (some) files (minimum bias): full = bulk_jet + extra_frag
  bulk_jet / bulk_bg sampled correlated: False
  oversamples per event: bulk_jet [1], bulk_bg [1];  N_coll 36-43;  extras per event 2.33, extra_E 927 GeV

== all: 3 events, 3 jackknife blocks
                       <Nch>            dNch/deta          <[pT]> GeV        sigma([pT])   ...   v2{2}
full                201.7 +- 26          126 +- 16       0.6656 +- 0.019   0.03209 +- 0.017 ...  0.08748 +- 0.029
bulk_bg             197.7 +- 19        123.5 +- 12       0.6524 +- 0.019   0.03325 +- 0.024 ...  nan
                        dNch           dNch/Nch_bg        d<[pT]> GeV        dc2 = d<<2>>  ...
delta_full              4 +- 15      0.02024 +- 0.078   0.01324 +- 0.0087  0.01226 +- 0.0038 ...
```

`--out bulk.json` holds `cuts`, `sources`, `tags`, `extras`, `correlated` and `classes`: per
class, per composition and per difference, every quantity as (value, jackknife error).

### Python

```python
import sys
sys.path.insert(0, "example/analysis/bulk")
from flow import Cuts, event_table, select, class_averages
from jetscape.hadrons_h5 import HadronFileReader

with HadronFileReader("out_had") as r:
    t = event_table(r, Cuts(eta_max=0.8, pt_min=0.2, pt_max=3.0, eta_gap=1.0))
res = class_averages(t, select(t, n_coll_percent=(0, 10)))
res["full/v2"], res["bulk_bg/v2"]               # (value, jackknife error)
res["delta_full/dnch"], res["delta_bulk_jet/dc2"]
```

### Not yet (a later project)

vₙ{4} and higher cumulants, ρ(v₂², [pT]), symmetry planes, multiplicity-based centrality,
pT-differential vₙ(pT), the experiments' ⟨ΔpT ΔpT⟩ correlator, pairs across oversamples (the
fluid's vₙ without decay non-flow), event weights, and a C++ twin for the ROOT path.

---

## 8. Recipes

**What the extras do to the bulk (V5).** Minimum bias, correlated legs, a few oversamples,
then the paired differences per N_coll class:

```bash
cd example/prod_AuAu_0_10_jet
TAG_PREFIX=PbPb_0_10_seed ./run_jobs.sh -j 4 --campaign mb_x 20 30 0 out_mb_x \
    --user-xml ../config/PbPb_0_10_jet.xml --grid ../config/grid_PbPb.yaml \
    --hard none --extra-hard --hadrons-only --oversample 10 --correlated
cd ../analysis/bulk
python bulk_obs.py ../../prod_AuAu_0_10_jet/out_mb_x --ncoll-classes 0 50 100 --out mb_x.json
# delta_bulk_jet/dc2, dpt, rdnch: the extras' wakes;  delta_full: + their fragments
```

Scale the threshold to see the dependence on p₀ (`--extra-pthat-min 20`, `15`, `10`, same
seeds: the backgrounds are the same). Compare with the linear estimate the review proposed:
the single-trigger response per deposited energy (`E_droplets`), times the number of
scatterings.

**The same events with and without extras.** Run the same seeds twice, once with
`--extra-hard`. `jet_frag` and `bulk_bg` are identical between the two, so any difference in
an observable built from them is zero by construction. Differences in `bulk_jet` and the full
event are the extras' effect, event by event.

**Small systems.** Run `OO_5360_jet.xml` / `NeNe_5360_jet.xml` with a trigger (jets, Z) or
minimum bias, at 0–10 % or with `b_max` 20, and the same tools. Check dN_ch/dη and v₂{2}
against the LHC O+O / Ne+Ne data first (§2).

**Z+jet with the extras' background.** The Pb+Pb 0–30 % Z+jet campaign as before, plus
`--extra-hard`; the map steps then use B = bulk_bg + extra_frag (`zjet/ANA_Zjet.md`).

---

## 9. Validation, cost and limits

Checked on the GB10, 2026-10-30 (`PLAN_multi_hard.md` §7):

| | |
|---|---|
| off = before | every dataset of the pair and particlize files bit-identical to `contrib` / `main` (Au+Au 0–10 %, 2 events, reuse 2; Pb+Pb 0–30 % Z+jet) |
| the trigger untouched | with 3–4 extras per event the trigger's final partons (all 14 columns, colour tags included), its initiators and the background bit-identical to extras off in every event, with and without `--reuse`; after hadronization `jet_frag` and `bulk_bg` identical |
| sampling | 2,000 Pb+Pb and 3,000 O+O events (IS + PythiaGun only): N follows Poisson(μ) (χ² 22.6/22 bins), μ/N_coll = σ/σ_NN, σ(p̂T > 15 GeV) = 0.524 mb at 5.02 TeV, the p̂T tail matches Pythia's σ ratios (0.330 vs 0.331), the vertices follow the binary collisions |
| analyses | Python and C++ twins agree (`check_cxx`); outputs of files without extras unchanged; the bulk tools recover an imposed v₂ = 0.08 (0.0802 ± 0.0011), v₃ = 0.04, [pT] and its dynamical spread on synthetic events |

Cost (Pb+Pb 0–10 %, 3 events, the same backgrounds, one job alone, `--hadrons-only`):

| | extras per event | droplets | s per event | peak memory |
|---|---|---|---|---|
| extras off | — | 45 | 32.1 | 18.98 GB |
| p₀ = 20 GeV | ~5 | 195 | 34.5 (+8 %) | 19.04 GB |
| p₀ = 15 GeV | ~15 | 464 | 36.4 (+13 %) | 19.08 GB |
| p₀ = 10 GeV | ~66 | 1,300 | 45.6 (+42 %) | 19.16 GB |

**Limits and open points**

- Two legs only: `bulk_jet − bulk_bg` contains the extras' wakes as well as the trigger's.
  They are uncorrelated with the trigger and average out in trigger-relative correlations, but
  add noise. A third leg is a later option (D14).
- No energy subtraction from the strings (D4); a stage-2 topic.
- The O+O / Ne+Ne bulk is untuned; the 0–10 % b edges are geometric, not multiplicity classes.
- The per-hadron `hadrons/scatter` is not exported to ROOT; event weights for `--pthat-bins`
  are not applied in the bulk tools.
- The Z+jet B_mixed takes the extras from other Z events on the nearest backgrounds (the
  extras are drawn per jet event): a choice worth a second look.
- Not from this option: iSS returns a few non-finite hadron momenta on Pb+Pb surfaces (~10 per
  115,000 hadrons). Hadron files written with `--eta-max` never store them; without a cut they
  crash FastJet.
- File names do not follow the XML: without `--out` or `TAG_PREFIX` every system is written
  as `AuAu_0_10_jet_*` (§2).

**References:** O+O / Ne+Ne charged-particle dn/dη and ⟨pT⟩: ATLAS,
[arXiv:2606.20257](https://www.alphaxiv.org/abs/2606.20257); ALICE light-ion overview,
[arXiv:2604.07184](https://arxiv.org/abs/2604.07184). Physics motivation for the bulk side
(minijets and vₙ, v₀(pT), ρ(v₂², [pT])): the review's item P7 and `PLAN_multi_hard.md`.
