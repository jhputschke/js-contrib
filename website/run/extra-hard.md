# Extra hard scatterings, small systems and minimum bias

Without extra options, every event of the jet production has exactly one hard scattering: the
trigger. A nucleus–nucleus collision has many more, N_coll × σ(p̂T > p₀)/σ_NN of them, and at
the LHC they matter: their fragments are the jet background a Pb+Pb event otherwise lacks (the
fake jets, the B of Z+jet, the hard tail of the underlying event), and their wakes change the
bulk (Nch, [pT], vₙ). `run_prod_jet.py --extra-hard` adds these **extras** above a threshold
p₀ to every event. The same option, without a trigger, gives **minimum-bias** events, and two
new inputs run **O+O and Ne+Ne at 5.36 TeV**. Everything on [Au+Au jets](auau-jet.md) applies;
this page adds what is new. The analysis side is on [Bulk and flow](../analysis/bulk.md) and
in the [Hadrons](../analysis/hadrons.md#the-events-other-hard-scatterings) page.

!!! warning "Work in progress"
    The analysis scripts and notebooks for these files (bulk and flow, v₂(p_T) against data)
    are work in progress and need further vetting; they are meant mainly as examples of how to
    use the simulation outputs. See the [note on the analysis section](../analysis/index.md).

!!! note "Needs the current code"
    The option needs X-SCAPE with PythiaGun `<extraHard>` and the current js-contrib. Check
    an image first: `python run_prod_jet.py --help` must list `--extra-hard`.
    On a Mac, the [xscape wheel](../install/mac.md) has it from 0.2.0. A 0.1.0 environment
    stops with `User XML tag <extraHard> is unrecognized`, also for the XMLs that carry the
    block switched off (`AuAu_MCGlauber_MUSIC_0_10_jet.xml`, `PbPb_0_10_jet.xml`,
    `PbPb_0_30_ZJet.xml`): install the current release next to it.

## How many extras

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
2 GeV), Σp_T ~40–60 GeV. The physics, and what the model leaves out, is in
[The physics](../about/physics.md#15-the-events-other-hard-scatterings).

## How an event is made

```text
IS (3dMCGlauber) -> Hard (PythiaGun: the trigger, then N ~ Poisson(N_coll sigma/sigma_NN) extras)
   -> MUSIC_1                    background leg: the strings alone           -> bulk_bg
   -> Liquefier + Eloss          Matter + LBT shower the trigger and the extras -> droplets
   -> MUSIC_2                    the same strings + all droplets             -> bulk_jet
   -> hadronization              trigger's partons -> jet_frag, the extras' -> extra_frag
```

- Each extra is one event of a Pythia instance of its own (HardQCD above p₀), at a binary
  collision drawn from the background's collisions; its partons are handed over after the
  trigger's, labelled with their scattering (1, 2, …).
- **The trigger is untouched.** Everything random the extras do (their Pythia, the Poisson
  draw, the vertices, their shower engines, their colour tags and hadronization seeds) runs on
  streams of their own. With the same seed, a job with extras has the same trigger showers,
  `jet_frag` and `bulk_bg` as a job without, bit for bit, in every event; only `bulk_jet`
  (their wakes) and the new records differ. Running the same seeds twice, with and without
  `--extra-hard`, is a paired measurement of what the extras do.
- With `--reuse` every jet event draws its own extras from its background's N_coll and
  binary collisions.
- Their energy is **not** subtracted from the strings, as for the trigger.

## Running it

| option of `run_prod_jet.py` | |
|---|---|
| `--extra-hard` | switch the extras on. Off (default): the files are bit-identical to before |
| `--extra-pthat-min P0` | their threshold [GeV]; default: the XML's `<extraHard><pTHatMin>` |
| `--extra-n N` | exactly N extras per event instead of Poisson (tests) |
| `--extra-warmup N` | Pythia events for their cross section at Init (default 20,000, ~1 s at the LHC) |
| `--extra-hadronize {groups,per_scattering}` | with `--hadrons-only`: all extras' showers in one Colorless call (default), or one call per scattering, with each hadron's scattering kept |
| `--hard none` | no trigger: the **minimum-bias** event, the bulk and its extras (N may be 0), weight 1. Needs `--extra-hard`; refuses `--pthat-*`, `--parton-ymax`, `--trigger`. No `jet_frag` |

Everything else works as without extras: `--reuse`, `--pthat-bins`, `--trigger Z`,
`--hadrons-only`, `--write-particlize`, `--correlated`, `run_jobs.sh` and the container
launchers. `run_prod_vacuum.py` (pp) ignores the extras: pp has one collision and Pythia's own
MPI.

```bash
cd prod_AuAu_0_10_jet                            # in your copy of the examples
# Au+Au 0-10 % with extras above 10 GeV (the XML's)
python run_prod_jet.py --events 10 --seed 1 --extra-hard --hadrons-only

# Pb+Pb 0-30 % Z+jet with extras above 15 GeV
python run_prod_jet.py --events 24 --seed 1 --user-xml ../config/PbPb_0_30_ZJet.xml \
    --grid ../config/grid_PbPb.yaml --pthat-bins 20-30,30-35,35-45,45-55,55-70,70-200 \
    --jets-per-bin 4 --lepton-ptmin 20 --lepton-etamax 2.5 --extra-hard --hadrons-only

# a two-stage campaign: particlize files, then hadronize.py (which handles the extras itself)
python run_prod_jet.py --events 10 --seed 1 --extra-hard --write-particlize both --particlize-only
python hadronize.py out/AuAu_0_10_jet_seed0001_particlize.h5 --oversample 20 --n-frag 10
```

The job prints the extras' settings once, and their number per event:

```text
  extras   pTHat > 15 GeV: sigma = 0.524 +- 0.002 mb (20000 warm-up events, Pythia seed 234669607), sigma_NN 67.57 mb, N ~ Poisson(N_coll sigma/sigma_NN)
prod_AuAu_0_10_jet: event 1/1 done in 40.5 s, ..., N_coll 1294, ..., 10 extra scattering(s) (mu 10.04, 3342 GeV)
```

The energy in that line includes the extras' forward ISR partons, which carry most of it; the
summed p_T (`extra_pt` in the files) is the better measure of what they bring to mid-rapidity.

### Choosing p₀

Lower thresholds put more of the hard tail into the bulk and cost more. Below ~10 GeV the
extras' energy becomes a sizeable part of what the strings already carry; that needs minijets
with energy subtraction and a retuned bulk, which this option does not do. The production
XMLs set 10 GeV for Au+Au, O+O and Ne+Ne, 15 GeV for Pb+Pb.

Cost on the GB10 (Pb+Pb 0–10 %, 3 events, the same backgrounds, one job alone,
`--hadrons-only`; hadronization ~5 s per event either way):

| | extras per event | droplets | s per event | peak memory |
|---|---|---|---|---|
| extras off | — | 45 | 32.1 | 18.98 GB |
| p₀ = 20 GeV | ~5 | 195 | 34.5 (+8 %) | 19.04 GB |
| p₀ = 15 GeV (Pb+Pb default) | ~15 | 464 | 36.4 (+13 %) | 19.08 GB |
| p₀ = 10 GeV | ~66 | 1,300 | 45.6 (+42 %) | 19.16 GB |

The time grows with the showers and with MUSIC_2's jet source (all cells × the active
droplets, on the CPU); the warm-up for σ adds ~1 s per job. Memory hardly changes.

### The XML settings

The driver writes the settings into `<Hard><PythiaGun><extraHard>` of the job XML; set them
there yourself to change a default for a whole campaign. The defaults are in X-SCAPE's
`config/jetscape_main.xml`.

| tag | default | |
|---|---|---|
| `<on>` | 0 | 1: the extras on (`--extra-hard`) |
| `<noTrigger>` | 0 | 1: no triggering scattering, the extras alone (`--hard none`). Refuses `<pTHatBins>`, `<partonYMax>`, `<trigger>` |
| `<pTHatMin>` | 15 | their p̂T threshold [GeV] (`--extra-pthat-min`). The production XMLs set their system's: 10 (`AuAu_MCGlauber_MUSIC_0_10_jet.xml`, `OO_5360_jet.xml`, `NeNe_5360_jet.xml`), 15 (`PbPb_0_10_jet.xml`, `PbPb_0_30_ZJet.xml`) |
| `<sigmaNN>` | empty | σ_NN^inel [mb]; empty: 3dMCGlauber's parametrization at `<eCM>`, the one its N_coll is made with (40.8 mb at 200 GeV, 67.6 at 5.02 TeV, 68.3 at 5.36 TeV) |
| `<warmup>` | 20000 | Pythia events at Init for σ(p̂T > p₀) (`--extra-warmup`); kept fixed for the job |
| `<fixedN>` | −1 | ≥ 0: exactly that many per event (`--extra-n`, tests); < 0: Poisson |
| `<LinesToRead>` | empty | more Pythia settings for the extras' instance only (not the trigger's `<LinesToRead>`, which may hold a Z process) |

The extras' Pythia runs `HardQCD:all = on` with `PhaseSpace:pTHatMin` = p₀, ISR on, MPI off,
FSR as `<FSR_on>` (off in production: Matter showers), hadron level off, and the trigger's
`<eCM>` and `<softMomentumCutoff>`.

## Minimum bias

`--hard none --extra-hard` makes events without a trigger: the bulk and its extras, N of them
from Poisson (N may be 0), every event with weight 1. An event with N = 0 has no hard parton:
the energy loss is skipped, MUSIC_2 still runs (restarted from MUSIC_1's checkpoint, so it is
cheap), and `bulk_jet` differs from `bulk_bg` only by sampling. `bulk_jet − bulk_bg` is then
the extras' wakes. These are the events for the bulk observables of
[Bulk and flow](../analysis/bulk.md); use correlated legs and several samples so that the
paired differences are small-noise:

```bash
TAG_PREFIX=PbPb_0_10_seed ./run_jobs.sh -j 4 --campaign mb_x 20 30 0 out_mb_x \
    --user-xml ../config/PbPb_0_10_jet.xml --grid ../config/grid_PbPb.yaml \
    --hard none --extra-hard --hadrons-only --oversample 10 --correlated
```

For the 0–10 % XMLs this is minimum bias *within* 0–10 %; set `b_max` to 20 in a copy of the
XML for all impact parameters. Scale the threshold with `--extra-pthat-min 20`, `15`, `10` on
the same seeds (the backgrounds are the same) to see the dependence on p₀.

## O+O and Ne+Ne at 5.36 TeV

`example/config/OO_5360_jet.xml` and `NeNe_5360_jet.xml` (in the image's `example/config/`)
are copies of `PbPb_0_10_jet.xml` for the LHC's light-ion run:

- 3dMCGlauber's ¹⁶O and PGCM ²⁰Ne configurations (`Ne20` is 3dMCGlauber's name for neon),
  √s_NN = 5.36 TeV, MUSIC's transverse grid ±12 fm, the output grid `grid_oo.yaml`;
- 0–10 %: b < 2.09 fm (O+O) and 2.41 fm (Ne+Ne), the 10 % quantile of b in 80,000
  minimum-bias 3dMCGlauber events, ⟨N_coll⟩ 42 / 51; set `b_max` to 20 for minimum bias;
- `<extraHard><pTHatMin>` 10 GeV.

!!! warning "Not refitted for the LHC or for small systems"
    The bulk settings are the Pb+Pb file's: the RHIC best fit of the JETSCAPE 3D calibration
    ([arXiv:2601.17234](https://arxiv.org/abs/2601.17234)), without the paper's Pb+Pb refit and without an afterburner
    ([The physics](../about/physics.md#7-hydrodynamics-music)). Two test events give dN_ch/dη(|η| < 0.5) ≈ 136
    (O+O) and 131 (Ne+Ne), ⟨pT⟩ ≈ 0.59 / 0.65 GeV. Compare dN_ch/dη and v₂{2} with ATLAS
    ([arXiv:2606.20257](https://arxiv.org/abs/2606.20257)) and ALICE
    ([arXiv:2604.07184](https://arxiv.org/abs/2604.07184)) before quoting numbers. The 0–10 %
    edges are geometric, not multiplicity classes.

```bash
# 0-10 % with a jet trigger and extras
python run_prod_jet.py --events 10 --seed 1 --user-xml ../config/OO_5360_jet.xml \
    --grid ../config/grid_oo.yaml --extra-hard --hadrons-only --out OO_5360_seed0001.h5
python run_prod_jet.py --events 10 --seed 1 --user-xml ../config/NeNe_5360_jet.xml \
    --grid ../config/grid_oo.yaml --extra-hard --hadrons-only --out NeNe_5360_seed0001.h5

# minimum-bias campaigns, correlated legs, 10 samples per event
TAG_PREFIX=OO_5360_seed ./run_jobs.sh --campaign mb1 20 30 1 out_oo_mb1 \
    --user-xml ../config/OO_5360_jet.xml --grid ../config/grid_oo.yaml \
    --hard none --extra-hard --hadrons-only --oversample 10 --correlated
                                                  # -> OO_5360_mb1_seed0001.* ...
TAG_PREFIX=NeNe_5360_seed ./run_jobs.sh --campaign mb1 20 30 1 out_nene_mb1 \
    --user-xml ../config/NeNe_5360_jet.xml --grid ../config/grid_oo.yaml \
    --hard none --extra-hard --hadrons-only --oversample 10 --correlated
```

These run through the jet production's driver, so they are not the
[O+O with ISR](oo-isr.md) production at 200 GeV: no initial-state shower module, the
two-leg pair, and the Au+Au jet production's files.

!!! tip "Name the files"
    The driver's default name is `AuAu_0_10_jet[_<campaign>]_seed<NNNN>`, whatever the XML, so
    without a name the O+O, Ne+Ne and Pb+Pb runs above are all written as `AuAu_0_10_jet_*`.
    Use `--out` for a single job and `TAG_PREFIX` for a campaign through `run_jobs.sh` (it must
    end in `_seed`; one prefix per OUTDIR). `launch_2gpu.sh` and `slurm_prod_array.sh` pass
    `TAG_PREFIX` into the container.

## What the files hold

- **`<stem>_hadrons_extra_frag.h5`:** the extras' fragments, with the layout of `jet_frag`
  (unit = event, `--n-frag` samples), plus `scatterings/` and the extras' `initiators/`.
  `jet_frag` stays the trigger's alone and is absent with `--hard none`.
- **`scatterings/`** (pair, particlize and `extra_frag` files): one row per scattering, the
  trigger's first: `scatter, pthat, code, x, y, n_partons, E, pt`.
- **Per event** (`diag/`, `events/`, `units/`): `n_extra`, `extra_mu` (= N_coll σ/σ_NN),
  `extra_E`, `extra_pt`.
- **Attributes and the `.json`:** the threshold, σ and its error, σ_NN, the warm-up and the
  seeds of the extras (`extra_*`, `prod_extra_*`).

Files without extras have none of these; their layout is unchanged. Full layout:
[What a production writes](../analysis/files.md#the-events-other-hard-scatterings).

## Next steps

- [Hadrons](../analysis/hadrons.md#the-events-other-hard-scatterings): the extras in
  `HadronFileReader`, and "the event without its trigger".
- [Jets with FastJet](../analysis/jets.md): the Z+jet and dijet analyses with the extras' B.
- [Bulk and flow](../analysis/bulk.md): Nch, [pT] and vₙ{2}, and what the extras do to them.
- Full details and the validation:
  [`docs/README_extra_hard.md`](https://github.com/jhputschke/js-contrib/blob/main/docs/README_extra_hard.md)
  and the design notes
  [`docs/Plans/PLAN_multi_hard.md`](https://github.com/jhputschke/js-contrib/blob/main/docs/Plans/PLAN_multi_hard.md).
