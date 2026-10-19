# Z+jet analyses: the campaign workflow

The analyses of the Pb+Pb 5.02 TeV 0–30 % Z+jet campaigns
([`docs/Plans/PLAN_ZJET_CMS10x.md`](../../../../../docs/Plans/PLAN_ZJET_CMS10x.md)): Z–hadron
correlations against CMS 2507.09307 and Z+jet against CMS 1702.01060, from the hadron files of
`run_prod_jet.py --hadrons-only` and the pp reference of `run_prod_vacuum.py`. A campaign step by
step: [`ANA_Zhadron.md`](ANA_Zhadron.md) (QA, the pools, event mixing, the ROOT route) and
[`ANA_Zjet.md`](ANA_Zjet.md) (the jets).

## Layout

```
example/analysis/
  jetcorr/            what zjet/ and dijet/ share: run paths (runs.py), figure data (figdata.py, replot.py)
  zjet/               this folder: the workflow
    setup/            the first 96-event test and the campaign set-up checks (setup/README.md)
  dijet/              the η-separated dijet analysis, first with the Z as jet1 (gap_count.py)
  runs/<run>/         data/ and figures/ of one campaign or test set (git-ignored)
example/analysis_root/zjet/   the C++ map steps (zqa, zhadron_pools, zjets; README.md there)
```

## The workflow

| step | Python (HDF5) | C++ (ROOT, route B) | |
|---|---|---|---|
| 0. QA | `qa.py map` → `qa.py plot` | `zqa` → `qa.py plot` | the Z's, weights, hadrons per source, multiplicity |
| 1. map: Z–hadron pools | `mix.py` (builds them) | `zhadron_pools` | one pool per production file |
| 1. map: jets | `zjet_jets_map.py` | `zjets` | one `<stem>_jets.root` per production file |
| 2. final: Z–hadron | `mix.py` | `mix.py --pbpb-pools` | the four mixing modes against CMS 2507.09307 |
| 2. final: Z+jet | `zjet_jets.py`, `zjet_jets_qa.py` | the same, on `zjets`' output | against CMS 1702.01060; the jet QA |
| check | `check_cxx.py pools / jets / qa` | | C++ = Python, 0 differences |

Route B converts the hadron files once (`../../prod_AuAu_0_10_jet/run_h5toROOT.py`).

| file | |
|---|---|
| `qa.py` | QA of a Z+jet production (step 0 of both ANA documents): `map` (per file: a table per Z event, per-source hadron histograms; the twin of `../../analysis_root/zjet/zqa`) and `plot` → `figures/qa_z_*`, `qa_weights_*`, `qa_hadrons*`, `qa_multiplicity*` (the total charged multiplicity), `qa_species*`, `qa_summary*.txt` |
| `zjet_jets_map.py`, `zjet_jets.py` | Z+jet against CMS 1702.01060: the map step in Python (anti-k_T R = 0.3, ρA-subtracted in Pb+Pb, embedding, random cones, the backgrounds' `bulk_bg` jets, per jet mass, constituents and leading p_T, the QA trees with all jets of one oversample; the twin of `../../analysis_root/zjet/zjets`) and the shared final stage, three steps: the fragments' jets (the reference), S − B with the own background's jets, S − B mixed as CMS (Z and jet cuts, the optional CMS smearing in four variants, HEPData, χ², jackknife), and pp against CMS's pp tables at pp's and at PbPb's resolution → `figures/zjet_jets_pp*.png`, `zjet_jets_step{1,2,3}*.png`, `zjet_jets_embedding*.png`, `zjet_jets_summary*.txt` |
| `zjet_jets_qa.py` | the jet QA of the Z+jet analysis, from either map step's QA trees: the ρA subtraction (p_T,raw, ρA, the negative tail, the fake fraction B/S), area and ρ (vs N_coll, η, dN_ch/dη; ρ of the Z event against its background), random cones and the embedding's JES/JER, geometry and the lepton cut, constituents, B_mixed against B_own → `figures/qa_jets_{subtraction,area_rho,response,geometry,constituents,mixing}*.png`, `qa_jets_summary*.txt` |
| `mix.py` | event mixing (`zjet_mixing.py`, [below](#event-mixing-mixpy)): pools of `runs/<run>/data/pbpb` and `vacuum` (cached in `data/mix/`) → step 1, the reference (factorized, with pp; `setup/compare.py`'s figure): `figures/mixing_reference_d{phi,y}.png`; all four steps: `mixing_dphi.png`, `mixing_dy.png`; the errors: `mixing_errors_d{phi,y}.png`; `mixing_summary.txt` (~10 s on the test) |
| `check_cxx.py` | the C++ map steps (`../../analysis_root/zjet`) against the Python ones: `pools`, `jets` (with the jet QA trees) and `qa` |
| `README_zjet_2507.09307.md` | every cut of CMS 2507.09307 (Z–hadron Δφ / Δy) and its setting in the production and the analysis |
| `zjet_wake.py`, `zjet_mixing.py`, `cms2507.py` | the library: the Z record, selection, weights and Z frame; the pools and the mixing modes; CMS 2507.09307's HEPData and observables (the factorized result) |
| `zjet_wake.ipynb` | the Z–hadron wake notebook (stored without outputs; runs on `setup/`'s test data) |

## Runs: where the data and figures go

Every analysis writes outside the code, into `../runs/<run>/` (git-ignored: the hadron files are
large and the results not public yet). `--run <name>` sets the defaults of `qa.py plot`,
`mix.py`, `zjet_jets.py` and `zjet_jets_qa.py`: figures to `runs/<name>/figures/`, and for
`mix.py` its inputs and pool cache under `runs/<name>/data/` (`pbpb/`, `vacuum/`, `mix/`).
The default run is `zjet_setup`; explicit `--outdir`, `--pools`, `--pbpb`, … still win. The ANA
documents keep their outputs in `$ANA` (e.g. `ANA=$EX/analysis/runs/zjet_c00`).

| run | |
|---|---|
| `zjet_setup` | the test's and the set-up checks' data (`setup/make_data.sh`), HEPData records, figures, `MIXING.md`, `RESULTS.md` |
| `zjet_c00-01` | campaigns c00 + c01 on the GB10: pools, QA, figures |

## Figure data

Every PNG below comes with `<figure>.npz` and `<figure>.root`: what it draws, to restyle it
(`python ../jetcorr/replot.py <figure>.npz --style my.mplstyle`, or `replot.draw(..., edit=…)`)
or to use its numbers elsewhere without rerunning (`../jetcorr/figdata.py`'s header has the layout;
`JS_FIGDATA=0` writes the PNGs only, `JS_FIGDATA_ROOT=0` skips the ROOT files; ROOT via
PyROOT, else uproot with the graphs as small TTrees).

## Event mixing (`mix.py`)

`mix.py` (with `zjet_mixing.py`, 2026-10-16; all details, the numbers and the plan for the
campaigns in the git-ignored `runs/zjet_setup/figures/MIXING.md`) forms S − B the experiment's way, from the full
events (`bulk_jet` + fragments, the bulk included), next to the factorized result above. One
mixed pair is exactly the factorized pair plus the bulk term (bulk_bg_i − bulk_bg_j) seen from
Z_i, which averages to zero here (the bulk does not know the Z: `Subtract_hard_momentum` 0) and
carries the bulk's fluctuations. The map step writes one pool per production file
(`runs/<run>/data/mix/<sample>/<stem>.npz`, compressed, cached and rebuilt when the files change; per event its S in
its Z frame for the 40-sample mean and for one oversample, and that oversample's tracks; per
background the summed lab-frame densities), ~2 s per test file; the modes take seconds.

| mode | S | B | error |
|---|---|---|---|
| factorized | Z-correlated yield (paired wake + fragments) | compare.py's: Δφ S's own average, Δy the folded η density over the pairs i ≠ j | jackknife over blocks of backgrounds |
| all | full event, 40-sample mean | every selected event, pairs weighted w_i w_j: the pooled density shifted into each Z's frame, no noise of its own | jackknife |
| cms10 | full event, 40-sample mean | CMS's procedure: 10 partners from 10 other backgrounds, drawn ~ w, one oversample each (one recorded collision) | jackknife |
| strict | one Z per background (drawn ~ w inside it, weighted by the background's total W_b): every Z its own fluid, as in data; one oversample | 10 partners from 10 other backgrounds of that set, one oversample each | the spread over the Z's (the experiment's statistical error) |

**The backgrounds are rotated** (2026-10-17, default): 3dMCGlauber puts the impact parameter
along x in every event, so all hydro backgrounds have their event plane near φ = 0 (on the
first campaign 84 % within ±π/8). The means do not care (φ_Z is uniform), but partners sharing
the Z's plane cancel part of its flow modulation in the mixed modes' noise, which mixing in data
cannot. `mix.py` rotates every Pb+Pb background by its own random multiple of 6° when it loads
the pools (`zjet_mixing.Pool.rotate_backgrounds`; the Z's, tracks and lab densities together, an
exact symmetry); `--no-rotate` keeps them as produced, `--rotate-seed` sets the angles. The
pools on disk are not rotated. Details: [`ANA_Zhadron.md`](ANA_Zhadron.md#the-measurement-s-b-and-the-ladder) (*The backgrounds' event planes*).

The checks, first in `runs/<run>/figures/mixing_summary.txt`:

1. **factorized = setup/compare.py** to 10⁻¹⁵ (Pb+Pb) and 2 × 10⁻¹⁴ (pp) relative: the pools, the
   shifts and the weights.
2. **The own-background rule with the bulk off** agrees with (1) within its errors.
3. **Closure**: the bulk term (all − all with the bulk off) against 0. On the test only a rough
   check: with a 4-block jackknife the χ² per bin has mean ≈ 3 (F(1, 3)) instead of 1.
4. **The exclusion bias**: leaving a Z's own background out of its partners shifts S − B by a
   flat level Σ W_b² (N_b − ⟨N⟩) / (W² − Σ W_b²) (`zjet_mixing.exclusion_bias`), because the
   backgrounds' weights (N_coll) grow with their multiplicity. Measured = predicted to all
   digits; of order 1/(number of backgrounds): 3 per radian at 1–2 GeV with the test's 4, ~10⁻³
   of the errors with a campaign's 4,000 (estimated). Data have no weights, so no such term. So
   `all` keeps every partner (S − B then integrates to zero exactly); `cms10` and `strict`
   exclude the own background as CMS does and print the bias.
5. **The strict spread error** against the scatter of 10 disjoint subsamples (partners inside
   each): pp pull widths 0.84–1.09 over 6 seeds and 10 or 20 subsamples: right to ~15 %. Pb+Pb
   needs ≥ 220 backgrounds, i.e. a campaign.
6. **The noise budget per Z**, from all Z's (`zjet_mixing.per_z_noise`, `strict_noise`): the
   spreads of the pooled mode, of the same with B scaled to each event's own multiplicity (no
   flat level), and of one oversample's Poisson noise; from them the strict error predicted at
   CMS's Z count with and without each event's flat multiplicity level and Poisson only
   (`--efficiency` scales the Poisson part as efficiency-corrected tracks do), against CMS's
   statistical errors (drawn in `mixing_errors_d*.png`). On the test the level is ~90 %
   of the per-Z variance at 1–4 GeV; which variant matches CMS's errors needs a campaign.

**The strict emulation against CMS's errors**: its error × √(N_eff / N_CMS) (`--n-cms`, default
1,550, the plan's estimate for 0–30 %) against CMS's statistical errors (HEPData), bin by bin
(`mixing_errors_d*.png`, the ratio in the summary). On the test the strict set is 4 Z's
with 3 partners each: the comparison is meaningless here and is meant for the campaigns (400 Z's
per campaign, 4,000 for all ten). Caveats: CMS's tracks are efficiency-corrected (their Poisson
part is larger than at perfect efficiency), and N_CMS is an estimate.

How the pools relate to the usual loop over the other events' tracks (S as per-Z Δy/Δφ
histograms, B from a few partners as their tracks, B from all partners as their summed
density): [`ANA_Zhadron.md`](ANA_Zhadron.md#appendix-the-pools-and-the-naive-mixing-loop).

On a campaign (the vacuum reference with the campaign's windows and storage):

```bash
python mix.py --pbpb <campaign dir> --vacuum ../runs/zjet_setup/data/vacuum_all --pools <pool dir> \
  --hists '' --jobs 8 --tag _c01
```
