# analysis_root/zjet — the Z+jet analyses in ROOT

The map steps of the two Z+jet analyses in C++, on the ROOT files of
[`run_h5toROOT.py`](../../prod_AuAu_0_10_jet/run_h5toROOT.py):

- **Z–hadron** (CMS 2507.09307): `zhadron_pools` writes, per production file, the pool of
  [`zjet_mixing.file_pool`](../../analysis/zjet/zjet_mixing.py) (per Z its same-event yield in the Z
  frame and one oversample's tracks; per hydro background the summed densities).
- **Z+jet** (CMS 1702.01060): `zjets` clusters every oversample of every Z event with FastJet,
  ρA-subtracted in Pb+Pb, with an embedding check and random cones.

With **extra hard scatterings** (an `extra_frag` ntuple) all three map steps take them into the
full event by default and keep the trigger's truth as it is; `zjets` also writes B's jets per
oversample (`bgx_*`: `bulk_bg` + the event's extras, the final stage's B_own and B_mixed),
`zhadron_pools` the extras' lab density (`pool_bgw` `R_extra`), `zqa` their histograms and
multiplicity. `--extras 0` leaves them out: the outputs of files without extras
([`../../analysis/zjet/ANA_Zjet.md`](../../analysis/zjet/ANA_Zjet.md),
[`ANA_Zhadron.md`](../../analysis/zjet/ANA_Zhadron.md), *Extra hard scatterings*).

The **final stage is shared** with the Python map steps: `../../analysis/zjet/mix.py`
(event mixing, HEPData, χ², jackknife, figures) and `zjet_jets.py` (selection, the optional
CMS smearing, HEPData, χ², jackknife, figures). Every map step has a Python twin
(`zjet_mixing.file_pool`, `zjet/zjet_jets_map.py`), and the two agree value for value
(*Checked*, below).

| file | |
|---|---|
| `ZJetCommon.h` | the C++ mirror of `zjet_wake.py`: Z selection, track mask, Z frame and its grid |
| `../common/AnaCommon.h`, `../common/JetMap.h` | shared with later analyses (the mirror of `../../analysis/jetcorr`): numpy's binning conventions, the portable random choices (`event_key`, `key_index`, `key_uniform`), event weights and background ids, the command line; the jet finding (anti-k_T, ρA with `StripRho`, the jet columns) |
| `zhadron_pools.cxx` | the Z–hadron map step → `<out>/<stem>_pool.root` |
| `zjets.cxx` | the Z+jet map step → `<out>/<stem>_jets.root` |
| `zqa.cxx` | the QA map step (the twin of `zjet/qa.py map`) → `<out>/<stem>_qa.root`: a table per Z event (the Z, weights, N_coll, the total charged multiplicity per source) and per-source hadron histograms; drawn by `qa.py plot` |
| `Makefile` | `make` with `root-config` and `fastjet-config` on `PATH`; `make FASTJET_PY=<the fastjet wheel's package directory>` builds against a pip `fastjet` wheel instead (below) |

A campaign from the RTX box (`gpu0/`, `gpu1/`), step by step in Python and in ROOT:
[`../../analysis/zjet/ANA_Zhadron.md`](../../analysis/zjet/ANA_Zhadron.md),
[`../../analysis/zjet/ANA_Zjet.md`](../../analysis/zjet/ANA_Zjet.md).

## The chain

```bash
conda activate js_fno                         # ROOT 6.36, FastJet 3.5.1 (headers and python)
cd example/analysis_root/zjet && make
# or, with ROOT and a venv with the pip fastjet wheel (rhicML: ~/root, js_analysis):
#   make FASTJET_PY=$(python -c 'import fastjet, os; print(os.path.dirname(fastjet.__file__))')
#   the C++ map steps then use the Python map steps' own FastJet (3.5.2 there, against the
#   system's 3.4.2 in /usr/local) and can link fjcontrib (the wheel's libfastjetcontribfragile).
#   The wheel's fastjet-config has its build paths, so the Makefile sets the flags itself.
#   Checked 2026-10-18 on one c00 file, --samples 2: check_cxx.py jets, 0 differences.
# 1. HDF5 -> ROOT (vacuum files: jet_frag only, found by themselves)
P=../../prod_AuAu_0_10_jet; D=../../analysis/runs/zjet_test/data
python $P/run_h5toROOT.py $D/pbpb_campaign -j 4 --out-dir $D/root/pbpb_campaign \
    --campaign-file $D/root/pbpb_campaign/pbpb_campaign.root
python $P/root_export/check_h5_vs_root.py $D/pbpb_campaign $D/root/pbpb_campaign \
    --campaign $D/root/pbpb_campaign/pbpb_campaign.root      # every hadron, column, sigma
# 2. the map steps (--files i:j for one slice of the files; the weights read all of them)
./zhadron_pools $D/root/pbpb_campaign $D/mix_cxx/pbpb [--seed 1] [--eta-max 2.4]
./zjets $D/root/pbpb_campaign $D/jets_cxx/pbpb_campaign [--R 0.3] [--samples N] [--cones 4] \
    [--ptmin 5] [--qa-samples 1] [--emb-samples N] [--rho-m 1]
# 3. the final stage (Python)
cd ../../analysis/zjet
python mix.py --pbpb-pools data/mix_cxx/pbpb --vacuum-pools data/mix_cxx/vacuum --tag _cxx
python zjet_jets_qa.py --pbpb data/jets_cxx/pbpb_campaign --pp data/jets_cxx/vacuum_all
python zjet_jets.py --pbpb data/jets_cxx/pbpb_campaign --pp data/jets_cxx/vacuum_all
# the C++ map steps against the Python ones
python check_cxx.py pools data/mix/pbpb data/mix_cxx/pbpb
python check_cxx.py jets data/jets_py/vacuum_all data/jets_cxx/vacuum_all
```

The final stage's figures come with `<figure>.root` (one directory per panel: TH1D, TH2D,
TGraphAsymmErrors, TGraph, named after the curves) and `<figure>.npz`, so the plotted numbers
can be restyled in ROOT or Python without rerunning (`../../analysis/jetcorr/figdata.py`,
`replot.py`).

## `zhadron_pools`

For the events passing CMS 2507.09307's Z selection (`zjet::CMS_2507` = `zjet_wake.CMS_2507`),
weighted by window cross section / events × N_coll (normalized to mean 1, `event_weights`):

- `pool_events`, one entry per Z: `event`, `bg` (the hydro background, a dense id over the
  input directory), `w`, `y`, `phi`, `k_one`, `n_samples`, `S_full[1440]` and `S_one[1440]`
  (p_T class × Δy × Δφ, float), the one oversample's tracks `trk_pt` (class), `trk_eta`,
  `trk_phi`; its keys `local_event`, `pthat_bin`, `n_coll`, `bg_unit`; and the dijet proxy's
  contents (`docs/Plans/PLAN_dijet_deta_wake.md` A3) on a 0.1 Δy grid per side of the Z
  (p_T class × side × Δy, `[576]`): `Y_full`, `Y_one`, `Y_zc`, `Y_frag`, and in lab η per side
  (`[288]`): `L_full`, `L_zc`, `L_frag` (vacuum: `Y_full`, `Y_one`, `L_full`);
- `pool_bgw`, one entry per hydro background and pTHat window, unweighted: `unit`, `k`, `n`
  (its selected events), `ncoll`, `S_full[1440]`, `S_zc[1440]` (the Z-correlated yield
  bulk_jet − bulk_bg + jet_frag; vacuum files: `full` only), `R_full[8640]` (lab η × φ);
  `pool_bgr`, one per background: `unit`, `RB[8640]` (bulk_bg's lab density per sample);
- `file_info`, the production file's record (`../common/AnaCommon.h` `FileRecord`);
- `pool_grid`, one entry: the grid's edges `pt[4]`, `dy[25]`, `dphi[21]`, `eta[49]`, `phi[61]`,
  `dyf[97]` (`zjet_mixing.GRID`; since 2026-10-29: `assemble` refuses pools binned otherwise, to
  10⁻¹², as this linspace may differ from numpy's in the last bit).

Pool version 4 (2026-10-18): the weights factorize per window and background, so
`zjet_mixing.assemble` (`Pool.load`) rebuilds `w`, the ids and the weighted per-background sums
of any set of pools, from several campaigns built apart; version 3 (`pool_bg`: `W`, `W2`,
`SW_<c>`, `T_<c>`, `R_<c>`) still loads, for its own campaign.

The oversample `k_one` is `key_index(event_key(seed, stem, event in the file), n)`, the same
number in C++ and Python and whatever other files a campaign has. `zjet_mixing.read_root_pool`
reads the files; `mix.py --pbpb-pools/--vacuum-pools` runs every mixing mode on them.

## `zjets`

Events with a Z of 60 < m < 120 GeV and p_T^Z > 30 GeV (the final stage applies the paper's
cuts); for each oversample k (≤ `--samples`): bulk_jet k + jet_frag (k mod n_frag) in Pb+Pb, the
fragmentation k in vacuum; all final particles but neutrinos and the Z's two leptons.

- **Jets**: anti-k_T R = `--R` (0.3), p_T,raw > 1 GeV, p_T > `--ptmin` (5 GeV) after the
  subtraction, |η| < 2; per jet also its mass, the number of its real (non-ghost) constituents
  and the leading one's p_T.
- **Speed** (2026-10-19): FastJet's background estimator no longer computes ρ_m, which it
  does by default from every k_T jet's constituents (a quarter of the time; ρ unchanged bit for
  bit); `--rho-m 1` turns it back on and stores ρ_m per jet (`<prefix>_rho_m`, for jet-mass
  subtraction; ρ unchanged, ~1.6× the time with `--emb-samples 4`). `--emb-samples N` embeds only the first N oversamples (`emb_*`, `nw_*`,
  `zcone_nw_*`; the jets, the Z cone and the random cones of every oversample unchanged): one
  48-event Pb+Pb file at R = 0.4, 40 oversamples, 180 s before, 121 s now, 65 s with
  `--emb-samples 4`. `zjet_jets.py` takes step 1 and the no-wake rows from the embedded
  oversamples only.
- **Records** (2026-10-18): `file_info` (the production file's name, events, hydro backgrounds
  and pTHat windows) and `run_info` (R, ghost area, `--ptmin`, `--cones`, `--seed`, ...): every
  output stands alone, and `zjet_jets.load_jets` rebuilds the ids and `weight_mb` of any set of
  outputs (several campaigns mapped apart); the stored `event`, `file`, `bg`, `weight_mb` are
  the input directory's. `jet_events` also has `bg_unit`.
- **Subtraction** (Pb+Pb): active areas with explicit ghosts on a fixed grid (|y| < 2.6, area
  0.01, no scatter, so that C++ and Python find the same areas); ρ(jet) the median p_T / area of
  the k_T R = 0.4 jets with |y| < 2.2 in the strip |y − y_jet| < 1; p_T = p_T,raw − ρA.
- **Embedding** (Pb+Pb): the fragments alone (reference jets, p_T > 10 GeV) and embedded in
  `bulk_bg` sample k, which has no wake; each reference jet with the nearest embedded jet
  (ΔR < R/2, else NaN): the subtraction's offset and resolution without medium response. The
  embedded event's jets above `--ptmin` are stored too (`nw_*`): the Z event without its wake,
  for the final stage's "no wake" variant and the wake's effect (`ANA_Zjet.md` A8).
- **Random cones** (Pb+Pb): `--cones` per oversample in `bulk_bg` at |η| < 1.6, positions from
  `key_uniform`; δp_T = Σp_T(ΔR < R) − ρπR²: the model's background fluctuations, to set
  against CMS's N term.
- **The Z cone**: the same at (y_Z, φ_Z) in the full event (`zcone_sum`, `zcone_rho`,
  `zcone_dpt`) and in the embedded event without the wake (`zcone_nw_*`; vacuum: ρ = 0, no
  `nw`): jet1's p_T variant of the dijet proxy (`../../analysis/dijet/deta.py --jet1 zcone`).
- **Background jets** (Pb+Pb): every hydro background's `bulk_bg` samples clustered and
  subtracted the same way, the model's minimum-bias events: the final stage's B (the jets not
  from the Z's interaction), from the Z's own background (step 2) or mixed from the backgrounds
  closest in N_coll (step 3, as CMS's 40 minimum-bias events).
- **Jet QA**: the first `--qa-samples` (1) oversamples of every Z event and background again,
  with all their jets (no cut on the subtracted p_T: its negative tail), for
  `zjet/zjet_jets_qa.py`; a few hundred kB per campaign file.

Output: `jet_events` (per event: `event`, `local_event`, `file`, `bg`, `pthat_bin`, `weight_mb`
= σ_k / N_k, `n_coll`, `n_samples`, the Z and its leptons) and `jets` (per event and oversample:
`jet_{pt_raw, pt, eta, phi, area, rho, m, nconst, lead_pt}`, the same `nw_*` for the embedded
event (no wake), `emb_{ref_pt, ref_eta, ref_phi,
pt, dr}`, `rc_{eta, phi, sum, rho, dpt}`), `bg_jets` (per background and `bulk_bg` sample:
`bg`, `bg_unit`, `sample`, `n_coll`, the same `jet_*`), and for the QA `qa_jets` (`event`,
`sample`, `jet_*`) and `qa_bg_jets` (as `bg_jets`). `zjet/zjet_jets.py`
reads both layouts (the Python map
step's groups come back from uproot as records).

The final stage is a ladder of three steps: (1) the fragments' jets alone with CMS's theory
smearing (the model-only reference); (2) S − B with the Z's own background's jets (the model's
twin); (3) S − B mixed from other backgrounds of similar N_coll (CMS's procedure). The CMS
smearing is optional: `none`, `detector` (C and S of PbPb), `full` (C, S, N) and `match` (C, S
and √(N² − σ_RC²), σ_RC the random cones' width), all four shown in steps 2 and 3
([`../../analysis/zjet/ANA_Zjet.md`](../../analysis/zjet/ANA_Zjet.md)).

## Checked (2026-10-16, the GB10)

- **The converter**: `check_h5_vs_root.py` on the campaign-format job (`data/pbpb_campaign`,
  24 events × 40 oversamples, all particles, 12-bit momenta, no positions) and on
  `data/vacuum_all` (12,000 events): every entry and hadron (pid, pstat, E, p) bit for bit,
  every column of the event record (the Z, its leptons and N_coll among them), the windows and
  the campaign cross sections. Conversion 2–3 s per file; 51 MB for the 24 events (HDF5 62 MB),
  154 MB for the 12,000 pp events.
- **`zhadron_pools`** against `zjet_mixing.file_pool` (`data/pbpb`, `data/vacuum`): events,
  backgrounds, `k_one`, the tracks, `S_full` and `S_one` identical; the background sums to
  10⁻¹³ (the weights' normalization). `mix.py` on the C++ pools prints the same summary as on
  the Python ones, apart from the last digit of check (1). 0.5–0.8 s per test file.
  The pools of both map steps are as produced; `mix.py` rotates every Pb+Pb background's event
  plane when it loads them (the default since 2026-10-17, `--no-rotate` to keep them:
  [`ANA_Zhadron.md`](../../analysis/zjet/ANA_Zhadron.md#the-measurement-s-b-and-the-ladder), *The
  backgrounds' event planes*), the same angles for both, so `check_cxx.py pools` and the
  summaries' comparison are unaffected.
- **`zjets`** against `zjet_jets_map.py`: pp (`vacuum_all`, 96,980 oversamples, 213,682
  jets) identical; Pb+Pb (`pbpb_campaign`, 4 oversamples of 19 Z events: 2,085 jets, 61
  embedding pairs, 304 random cones) to 10⁻¹⁴, areas and ρ included; the masses, constituent
  counts, leading p_T and the QA trees (2,457 jets, 127 background jets) identical
  (2026-10-17, with the jet QA's columns; the jets and the final stage's summary as before).
- **`zqa`** against `qa.py map` on `data/pbpb`, `data/vacuum`, `data/pbpb_campaign`, `data/vacuum_all`:
  the per-event tables and every histogram identical (`check_cxx.py qa`: 0 differences), the plot
  step's summaries line for line.
- **The final stage** on the C++ jets reproduces the earlier pp comparison (`zjet_pp_jets.py`):
  R_jZ 0.30, 0.42, 0.54, 0.68 in the four p_T^Z classes; its pp figure (`zjet_jets_pp*.png`)
  agrees with `zjet_pp_vs_cms1702.png` within the smearing's random numbers.
- **Speed**: `zjets` 0.12 s per Pb+Pb oversample (six clusterings with areas), 3 s for the
  12,000 pp events. The Python twin is about as fast for jets (FastJet does the work in both);
  the C++ map steps are an example of the ROOT workflow rather than a speed-up. A campaign's
  ~6,000 Z events × 40 oversamples: ~8 CPU hours per campaign for `zjets` (estimated), less
  with `--samples`.
