# ANA_Zhadron — the Z–hadron (wake) analysis of a campaign, step by step

CMS 2507.09307 (Z–hadron Δφ and Δy, Pb+Pb 5.02 TeV 0–30 % and pp; HEPData ins2945894) on the
first campaign, `zjet_c00`: 9,600 Pb+Pb events from the 2 × RTX 3090 box, in two GPU
subdirectories. Two routes to the same final stage: **A**, Python on the HDF5 files;
**B**, ROOT/C++ map step on the converted files, then the same Python final stage. The
analysis itself is a ladder of four steps, from the model-only ideal (step 1, the reference) to
a one-to-one emulation of the measurement (step 4). The Z+jet analysis of the same events is
[ANA_Zjet.md](ANA_Zjet.md); the method in detail:
[`zjet_test/README.md`](zjet_test/README.md#event-mixing-mixpy), `zjet_mixing.py`,
[`../analysis_root/zjet/README.md`](../analysis_root/zjet/README.md).

**Results are not public yet:** write the outputs (`--outdir`) outside the repository, as
below; the repository's `zjet_test/figures/` is git-ignored for the same reason.

## 0. What you have

```
/data/zjet_c00/                  # OUT of utils/zjet/zjet_3090_campaign.sh 0
  gpu0/PbPb_0_30_zjet_c00_seed0001 ... seed0050.*       # launch_2gpu.sh: GPU0 the first 50 seeds
  gpu1/PbPb_0_30_zjet_c00_seed0051 ... seed0100.*       #             GPU1 the next 50
        <stem>_hadrons_bulk_jet.h5, _hadrons_bulk_bg.h5, _hadrons_jet_frag.h5, <stem>.json, ...
/data/zjet_pp/vacuum_jet_zjet_pp_seed900001_hadrons_jet_frag.h5   # the pp reference
```

100 jobs × 96 events = 4 hydro backgrounds × 6 p̂T windows × 4 Z's per job, 400 backgrounds;
~35 GB. The pp reference is `run_prod_vacuum.py --events 12000 --seed 900001 --campaign zjet_pp
… --n-frag 40 --eta-max 2.6 --no-x` (the command in [`utils/zjet/README.md`](../../../../utils/zjet/README.md#pp-reference)),
minutes on any CPU. What the stored hadrons are (η and Σ⁰ stable in the bulk, no weak decays):
the "Read first" note of [`docs/Plans/PLAN_ZJET_CMS10x.md`](../../../../docs/Plans/PLAN_ZJET_CMS10x.md).

**The two subdirectories.** The readers do not look below the directory they are given
(`../README.md`, *Multi-GPU output*): give Python a quoted glob over `gpu*/`, and convert both
subdirectories into one ROOT directory for C++. GPU0 has the first block of seeds and the seeds
are zero-padded in the names, so both readers see the files in the same order.

## The measurement: S, B, and the ladder

CMS 2507.09307 (§5, Eqs. 1–2) measures, per Z, the charged tracks around it in the Z's frame,
Δφ = φ_Z − φ_ch and Δy = y_Z − η_ch (the model uses φ_ch − φ_Z and η_ch − y_Z; the folding
below removes the sign):

- **S**, the same-event yield: the Z of an event with the tracks of **its own** event;
- **B**, the mixed-event yield: the Z of an event with the tracks of **other** Z events, 10 per Z
  in CMS, without further matching ("the impact … was found to be negligible");
- **Δ⟨N_ch⟩ = S − B**, averaged over the Z's. The uncorrelated bulk cancels on average, what is
  left is the Z-correlated yield: the jet's fragments (Δφ ≈ π), the medium response, the wake's
  depletion near the Z (Δφ ≈ 0, Δy ≈ 0). ±Δφ (±Δy) are merged and scaled by 0.5; Δy is taken on
  the Z side, |Δφ| < π/2. B is flat in Δφ, so in Δφ S − B is S around its mean (it integrates to
  zero); in Δy B carries the acceptance (|η_ch| < 2.4 shifted by y_Z).

Tracks: charged, |η| < 2.4, p_T classes 1–2, 2–4, 4–10 GeV, the Z's two leptons removed (CMS: a
cone of 0.0025 around them). Z's: CMS 2507.09307's cuts (`zjet_wake.CMS_2507`: 40 < p_T^Z <
350 GeV, |y_Z| < 2.4, 60 < m < 120 GeV, leptons p_T > 20 GeV, |η| < 2.4; ee and μμ, CMS μμ only).
Every Z weighs window cross section / events × N_coll (CMS: every Z 1), normalized to mean 1.

**Step 1 is the model-only ideal.** Every event has a twin without the jet (`bulk_bg`, the same
fluid without the jet's energy, sampled with the same random numbers), so its Z-correlated yield
is known exactly: S of the jet leg minus S of its background, sample by sample, plus the
fragments; B is then noise-free. That is the **reference**: the model's prediction of the
quantity CMS measures, as precise as the model allows. CMS's S − B is an estimator of that same
Z-correlated yield, not another observable, so step 1 is compared with the data directly. Whether
that is legitimate is what **step 2** decides: it applies the experiment's method (mixing, the
full events) to the same model events. If its closure holds (the bulk term consistent with zero),
mixing and step 1 give the same answer on average and the data comparison belongs on step 1,
which has the smallest errors and no estimator noise. If it fails, something in the model
correlates the bulk with the Z, CMS's S − B is then not the Z-correlated yield alone, and the
like-for-like comparison with data is step 2 or 3 (the data went through mixing too); the
difference to step 1 is then the method's bias (CMS's own MC closure found it negligible). Steps
3 and 4 are not closer to the data's answer: they estimate the same quantity with CMS's partner
noise and an experiment's statistics, and step 4 exists to compare the errors. Each step adds one
ingredient, its difference to the step before is what that ingredient costs:

| step | mode | S (per Z) | B (per Z) | adds | its difference to the step before |
|---|---|---|---|---|---|
| **1** | factorized | the Z-correlated yield: (`bulk_jet` − `bulk_bg`) + fragments, mean over the 40 oversamples | Δφ: S's own Δφ average; Δy: the pooled η density of the Z-correlated yields folded with each y_Z, pairs i ≠ j, weights w_i w_j | the reference; pp against CMS pp first | – |
| **2** | all | the full event (`bulk_jet` + fragments), mean over 40 | the pooled full-event density (η, φ) of every selected event, shifted into each Z's frame (`ZKernel`): no noise of its own | the bulk and its fluctuations in S | the **closure**: zero within the bulk's noise, CMS's method tested on the model (check 3) |
| **3** | cms10 | full event, mean over 40 | 10 partner events from 10 different hydro backgrounds, never the Z's own, drawn ∝ weight; each one recorded oversample (`k_one`) | CMS's procedure on every Z | partner noise and a small exclusion bias (check 4) |
| **4** | strict | one Z per background (drawn ∝ w, weighted by the background's Σw), **one** oversample | 10 partners from 10 other backgrounds **of that set**, one oversample each | an experiment's statistics: every Z its own fluid | the **errors**, compared with CMS's statistical errors (checks 5, 6); its central values are not the point |
| (pp) | – | the fragments (Pythia's Z + jet with MPI, Matter in vacuum) | as step 1 (no fluid) | – | the reference for Pb+Pb − pp in every step |

Why steps 3 and 4 never take a partner from the Z's own background: its 24 events share the
fluid (and the random numbers), and such a partner would cancel part of S's fluctuations, which
no experiment can do. The price, with weighted backgrounds, is a small flat bias of order
1/(number of backgrounds) (check 4); step 2 keeps every partner because its B has no noise anyway.

**The settings** (`mix.py`; one run makes all four steps):

| option | default | what it changes |
|---|---|---|
| `--n-mix` | 10 | partners per Z in steps 3 and 4 (CMS: 10); more partners, less B noise |
| `--blocks` | 100 | jackknife blocks of hydro backgrounds (4 per block for 400 backgrounds) |
| `--seed` | 1 | the strict set, the partner draws, and each event's recorded oversample `k_one` |
| `--n-cms` | 1550 | CMS's 0–30 % Z count (78 % of "about 2000"), for scaling the strict errors |
| `--strict-split` | 10 | disjoint subsamples for check (5) |
| `--efficiency` | 1 | the Poisson part of the predicted strict errors / ε (CMS's tracks are efficiency-corrected) |
| `--hists` | the test's | `''` on a campaign (check 1 compares with the 96-event test's `compare.py`) |

In the code (`zjet_mixing.all_partners`): the pair rule `all` (every pair), `self` (i ≠ j,
compare.py's Δy rule), `background` (no pair inside a background, check 2); `phi_flat` (B
flat in Δφ by construction, compare.py's); the contents `full` and `zc` (the Z-correlated
yield, "bulk off").

## The pools: what the map step keeps

The analysis runs in two stages. A **map step** reads the hadrons of every production file once
and writes a **pool**, a per-file summary from which the **reduce step** (`mix.py`) computes all
four steps, the checks and the jackknife without reading the hadrons again. Two map steps make
the same pools:

```
route A:  *_hadrons_*.h5  --mix.py (map + reduce)--------------------------------------->  figures
route B:  *_hadrons_*.h5  --run_h5toROOT.py-->  *_hadrons.root  --zhadron_pools-->  *_pool.root
                                                                          --mix.py --pbpb-pools (reduce)-->  figures
```

| pools made by | format | read by |
|---|---|---|
| `mix.py`'s own map step (route A, Python, from the HDF5 files; `zjet_mixing.file_pool`) | compressed `.npz` in `--pools/<sample>/`, with a manifest | `mix.py` (cached; rebuilt when the files, the selection or the seed change) |
| `zhadron_pools` (route B, C++, from the ROOT hadron files) | `.root` (TTrees `pool_events`, `pool_bg`) | `mix.py --pbpb-pools / --vacuum-pools` (`zjet_mixing.read_root_pool`) |

The content is the same (`check_cxx.py pools`, array by array); `mix.py` never reads the ROOT
hadron files itself. Pb+Pb and pp can come from different routes, but the checks assume one.

**Per selected Z** (CMS 2507.09307's Z selection):

| field | shape | what |
|---|---|---|
| `event`, `bg` | – | the global event index, its hydro background (dense id over the campaign) |
| `w` | – | the weight: window σ / events × N_coll, mean 1 over the selected Z's |
| `y`, `phi` | – | the Z's y_Z and φ_Z |
| `k_one`, `n_samples` | – | the event's one "recorded" oversample (`zjet_wake.key_index(event_key(seed, file name, event in the file), n)`), the number of oversamples |
| `S_full` | 3 × 24 × 20 | the full event's tracks (`bulk_jet` + fragments) in the Z frame: p_T class × Δy (0.4) × Δφ (π/10), mean over all oversamples |
| `S_one` | 3 × 24 × 20 | the same for oversample `k_one` alone: one collision |
| `trk_pt`, `trk_eta`, `trk_phi` | n tracks | that oversample's tracks in the lab frame (p_T class, η, φ): what this event contributes as a mixing partner |

**Per hydro background** (summed over its selected Z's):

| field | shape | what |
|---|---|---|
| `W`, `W2` | – | Σw, Σw² |
| `SW_full`, `SW_zc` | 3 × 24 × 20 | Σ w·S in the Z frame: the full event; the Z-correlated yield (`bulk_jet` − `bulk_bg` sample by sample, + fragments) |
| `T_full`, `T_zc` | 3 × 24 × 20 | Σ w²·S, the self pairs' terms |
| `R_full`, `R_zc` | 3 × 48 × 60 | Σ w·ρ, the lab-frame track density: p_T class × η (0.1) × φ (6°) |

Tracks are always charged, |η| < 2.4, p_T 1–10 GeV, without the Z's two leptons. Vacuum pools
store one content (full = Z-correlated).

**Which step uses what:**
- step 1 (factorized): the `_zc` sums: S from `SW_zc`, B from `R_zc` (its η marginal folded with
  each y_Z), the self pairs removed with `T_zc`;
- step 2 (all): `SW_full` and `R_full` (η and φ), shifted into each Z's frame; against the `_zc`
  version, the closure;
- step 3 (cms10): each Z's `S_full` with 10 partners' `trk_*`, shifted into its frame;
- step 4 (strict): `S_one` with partners' `trk_*`; the noise budget (check 6): `S_full`,
  `S_one`, `R_full`;
- the jackknife: whole backgrounds left out, which the per-background sums make cheap.

Only one oversample's tracks are kept per event (all 40 would be ~40× the size); the densities
and the S histograms average over all 40. Size: ~0.4–0.6 MB per test file (compressed), ~1 MB
per campaign file and ~100 MB per campaign (estimated), against the campaign's ~35 GB of hadrons.

## 1. Set up (once)

```bash
git clone -b zjet https://github.com/jhputschke/js-contrib ~/js-contrib-zjet
REPO=~/js-contrib-zjet; EX=$REPO/contribs/PyJetscape/example
DATA=/data/zjet_c00; PP=/data/zjet_pp; ANA=/data/ana_c00       # ANA: outputs, private
mkdir -p $ANA/logs
```

- **Python**: numpy, h5py, PyYAML, matplotlib, hdf5plugin (the files' Blosc filter), and for
  route B uproot and awkward: the venv of [`utils/analysis_env`](../../../../utils/analysis_env/README.md),
  or the conda env `js_fno` on the GB10. The scripts find `jetscape` in the checkout.
- **ROOT ≥ 6.34** (route B; RNTuple) with `root-config` on `PATH`.
- **CMS's HEPData record** (once, into `zjet_test/data/hepdata/`, git-ignored):
  `cd $EX/analysis/zjet_test && ./make_data.sh hepdata`.

## 2. Check the data

```bash
python $REPO/utils/check_deposition.py $DATA          # every <stem>.json below it; exit 1: flagged
ls $DATA/gpu*/*_hadrons_jet_frag.h5 | wc -l           # 100
python - <<EOF
import sys; sys.path.insert(0, "$EX/../python")
from jetscape.hadrons_h5 import HadronFileReader
r = HadronFileReader("$DATA/gpu*/*_hadrons_jet_frag.h5")
print(r.n_files, "files,", r.n_events, "events, correlated", r.correlated)
print("windows", r.pthat_bins, "duplicate backgrounds", r.duplicate_backgrounds())
EOF
```

Expect 100 files, 9,600 events, `correlated True`, the six windows 20–30 … 70–200 GeV and no
duplicate backgrounds. A job flagged by `check_deposition.py` (no jet deposition, seen once
in ~250 Au+Au jobs): move its files away and rerun the campaign's command, which redoes it.

## QA first (both analyses): `qa.py`, `zqa`

The same events for both papers; one QA pass before either analysis. The map step reads every
production file once and writes per file a table per Z event and per-source hadron histograms;
the plot step draws them. Route A (HDF5) and route B (ROOT) give the same numbers
(`check_cxx.py qa`).

```bash
cd $EX/analysis/zjet_test
# route A
python qa.py map "$DATA/gpu*/*_hadrons_jet_frag.h5" $ANA/qa_py/pbpb --jobs 8
python qa.py map "$PP/*_hadrons_jet_frag.h5" $ANA/qa_py/pp
python qa.py plot --pbpb $ANA/qa_py/pbpb --pp $ANA/qa_py/pp --outdir $ANA/figures --tag _c00
# route B (after the conversion, B1)
$EX/analysis_root/zjet/zqa $ANA/root/pbpb $ANA/qa_cxx/pbpb [--files i:j]
$EX/analysis_root/zjet/zqa $ANA/root/pp $ANA/qa_cxx/pp
python qa.py plot --pbpb $ANA/qa_cxx/pbpb --pp $ANA/qa_cxx/pp --outdir $ANA/figures --tag _c00_cxx
python check_cxx.py qa $ANA/qa_py/pbpb $ANA/qa_cxx/pbpb | tail -1          # 0 difference(s)
```

| figure | shows |
|---|---|
| `qa_z_PbPb_c00.png`, `qa_z_pp_c00.png` | 6 × 3: p_T^Z, y_Z, φ_Z, m_ℓℓ, the leptons' p_T and η (e dashed, μ solid); columns unweighted, σ_k/N_k-weighted, σ_k/N_k × N_coll-weighted (normalized); every panel for all Z's, CMS 2507.09307's and CMS 1702.01060's selections |
| `qa_weights_PbPb_c00.png`, `qa_weights_pp_c00.png` | p_T^Z per p̂T window, the σ-weighted p_T^Z, N_coll per hydro background, the σ × N_coll weights of CMS 2507's Z's (w / ⟨w⟩) with N_eff |
| `qa_hadrons_c00.png` | 3 × 4: p_T, η, φ per Z event (σ × N_coll-weighted), charged and all particles, each on the bulk's scale (background, background + wake, full event) and on the jet's (Pb+Pb fragments, the wake bulk_jet − bulk_bg, pp fragments); the Z's leptons removed |
| `qa_multiplicity_c00.png` | the **total charged multiplicity** N_ch (\|η\| < 2.4, all p_T) of the full event per Z event, Pb+Pb and pp, unweighted and σ × N_coll-weighted; dN_ch/dη (\|η\| < 0.5) against N_coll (full event, background); the oversamples' var / mean of N_ch (1: Poisson; resonance decays make it ~1.5–2); the fragments' N_ch, Pb+Pb against pp |
| `qa_species_c00.png` | the species fractions per source (bulk π⁰, η, Σ⁰, K⁰ are stable in iSS's list, the fragments' decayed) |
| `qa_summary_c00.txt` | Z counts and N_eff per selection and weighting, the Z leptons found among the fragments (2 per fragmentation expected; they are removed), the mean multiplicities, the oversamples' spread, the species |

Look for: the windows tiling p_T^Z smoothly, N_coll spanning 0–30 % (not a few values), the
weights' range (N_eff), exactly 2 Z leptons per fragmentation, a smooth bulk η (|η| < 2.6 stored)
and φ (flat on average over many backgrounds), dN_ch/dη rising with N_coll.

## A. Python on the HDF5 files

**A1. The pools and the four steps** (map: one process per production file; reduce: every step):

```bash
cd $EX/analysis/zjet_test
python mix.py --pbpb "$DATA/gpu*/*_hadrons_jet_frag.h5" --vacuum "$PP/*_hadrons_jet_frag.h5" \
    --pools $ANA/pools --hists '' --jobs 8 --outdir $ANA/figures --tag _c00 \
    > $ANA/logs/mix_c00.log 2>&1
# step 4's predicted errors with the Poisson part of CMS's efficiency-corrected tracks
python mix.py --pbpb "$DATA/gpu*/*_hadrons_jet_frag.h5" --vacuum "$PP/*_hadrons_jet_frag.h5" \
    --pools $ANA/pools --hists '' --efficiency 0.65 --outdir $ANA/figures --tag _c00_eff065
```

- The pools land in `$ANA/pools/pbpb` and `$ANA/pools/vacuum` (compressed `.npz`, a manifest);
  the second command reuses them. They are rebuilt by themselves when the files, the selection
  or the seed change (add a campaign: everything is rebuilt, the weights change).
- `--hists ''`: check (1) compares with the 96-event test's `compare.py` and is skipped here.
- Cost (estimated from the test): the map a few seconds per file and ~2–3 GB per process, so
  `--jobs 8` takes minutes and ~20 GB; the reduce a few minutes.

Then read `$ANA/figures/mixing_summary_c00.txt` and the figures **in the order of the steps**:

**A2. Step 1, the reference** — `mixing_reference_dphi_c00.png`, `mixing_reference_dy_c00.png`,
the summary's "Step 1" block: the model's prediction against CMS, to be read together with
step 2's closure (A3), which decides whether it is the comparison with data.
1. **pp first**: the model's pp against CMS's pp (top row). It tests the Z + jet itself (Pythia
   with ISR and MPI, Matter in vacuum, the hadronization) without any medium; Pb+Pb − pp cancels
   much of what it gets wrong, not all.
2. **Pb+Pb 0–30 %** (middle row): the jet side (Δφ ≈ π, quenched fragments plus medium response)
   and the Z side (Δφ ≈ 0, Δy ≈ 0: the wake's depletion at 1–2 GeV).
3. **Pb+Pb − pp** (bottom row): the medium's modification, the paper's headline observable.

**A3. Step 2, the closure** — `mixing_dphi_c00.png`, `mixing_dy_c00.png`: blue (all) against red
(factorized); the summary's checks (2) and (3). Expect the bulk term (all − all with the bulk
off) consistent with zero (χ²/ndf ≈ 1 with 100 blocks) and blue's errors larger than red's at
1–2 GeV (the bulk's fluctuations), equal at 4–10 GeV. A significant bulk term would mean the bulk
correlates with the Z: then CMS's S − B is not the Z-correlated yield alone.

**A4. Step 3, CMS's procedure** — green (cms10) in the same figures; check (4): the exclusion
bias, predicted = measured, ≲ 0.03 per radian at 1–2 GeV with 400 backgrounds (an estimate).
cms10 agrees with steps 1–2 within its larger errors (10 partners' noise).

**A5. Step 4, the errors** — orange (strict) and `mixing_errors_dphi_c00.png`,
`mixing_errors_dy_c00.png`; checks (5) and (6).
- (5): the strict error formula, pull widths ≈ 1 now also for Pb+Pb (400 Z's, ≥ 220 needed).
- (6): the noise budget per Z from 400 fluids (the test had 4): the flat multiplicity level's
  share, the fluid term, and the predicted strict errors at CMS's Z count with each event's flat
  level, without it, and Poisson only, at `--efficiency` 1 and 0.65 (the `_c00_eff065` run).
- The "ratio" column: the strict error at CMS's count against CMS's statistical error. Which
  prediction matches tells how CMS's errors were made, or how realistic the model's fluctuations
  are; at 4–10 GeV the model's errors can come out small, its background having no minijets from
  the other binary collisions.

**A6. Optional: maps and profiles.** `zjet_wake.ipynb` (in this folder) draws the (Δη, Δφ)
maps, the p_T^Z dependence and one event's wake: point its data cell at the same glob
(`HadronFileReader("$DATA/gpu*/*_hadrons_jet_frag.h5")`) and the vacuum file.

## B. ROOT: the C++ map step, the same final stage

**B1. Convert to ROOT** (once; [ANA_Zjet.md](ANA_Zjet.md) uses the same files):

```bash
P=$EX/prod_AuAu_0_10_jet
python $P/run_h5toROOT.py $DATA/gpu0 $DATA/gpu1 -j 8 --out-dir $ANA/root/pbpb \
    --campaign-file $ANA/root/pbpb/zjet_c00_campaign.root > $ANA/logs/h5toroot.log 2>&1
python $P/run_h5toROOT.py $PP --out-dir $ANA/root/pp \
    --campaign-file $ANA/root/pp/zjet_pp_campaign.root            # vacuum: jet_frag alone
python $P/root_export/check_h5_vs_root.py "$DATA/gpu*/*_hadrons_jet_frag.h5" $ANA/root/pbpb \
    --campaign $ANA/root/pbpb/zjet_c00_campaign.root | tail -1     # "0 difference(s)"
```

Both GPU directories go into one `--out-dir`. ~80 B per hadron of the largest tag while a file
is converted (~2 GB for a 96-event job), a few seconds per file; ~0.8 × the HDF5 size (RNTuple).
`check_h5_vs_root.py` compares every hadron, event column and cross section (it reads every
file: some minutes).

**B2. Build** (`root-config` and `fastjet-config` on `PATH`): `cd $EX/analysis_root/zjet && make`.

**B3. The pools in C++**, in slices (`--files i:j`; every process reads all event tables for
the weights, so the slices combine):

```bash
cd $EX/analysis_root/zjet
for i in 0 25 50 75; do
  ./zhadron_pools $ANA/root/pbpb $ANA/pools_cxx/pbpb --files $i:$((i + 25)) \
      > $ANA/logs/pools_cxx_$i.log 2>&1 &
done; wait
./zhadron_pools $ANA/root/pp $ANA/pools_cxx/pp
```

~1 s per test file, ~0.5–1.6 GB per process (measured on the test); the outputs are
`<stem>_pool.root`, written as `.part` and renamed when complete (a rerun redoes its slice).

**B4. The final stage on the C++ pools** (all four steps, read as in A2–A5):

```bash
cd $EX/analysis/zjet_test
python mix.py --pbpb-pools $ANA/pools_cxx/pbpb --vacuum-pools $ANA/pools_cxx/pp --hists '' \
    --outdir $ANA/figures --tag _c00_cxx
```

**B5. C++ against Python** (after A1):

```bash
python check_cxx.py pools $ANA/pools/pbpb $ANA/pools_cxx/pbpb | tail -1   # 0 difference(s)
python check_cxx.py pools $ANA/pools/vacuum $ANA/pools_cxx/pp | tail -1
diff $ANA/figures/mixing_summary_c00.txt $ANA/figures/mixing_summary_c00_cxx.txt
```

The pools agree array by array (background sums to ~10⁻¹³) and the summaries line by line,
apart from rounding in the last digits. A difference in the event numbers means the two readers
saw the files in another order (`../README.md`, *Multi-GPU output*).

## Outputs of every step

| step | writes | figures |
|---|---|---|
| 2 | nothing (stdout) | – |
| QA | `$ANA/qa_py/{pbpb,pp}/<stem>_qa.npz` (route B: `qa_cxx/…/<stem>_qa.root`), `qa_summary_c00.txt` | `qa_z_*`, `qa_weights_*`, `qa_hadrons_*`, `qa_multiplicity_*`, `qa_species_*` |
| A1 (map) | `$ANA/pools/pbpb/<stem>.npz` (100), `$ANA/pools/vacuum/<stem>.npz`, `manifest.json` each | – |
| A1 (reduce) | `$ANA/figures/mixing_summary_c00.txt` | step 1: `mixing_reference_dphi_c00.png`, `mixing_reference_dy_c00.png`; steps 1–4: `mixing_dphi_c00.png`, `mixing_dy_c00.png`; step 4: `mixing_errors_dphi_c00.png`, `mixing_errors_dy_c00.png` |
| A1, second run | `mixing_summary_c00_eff065.txt` | the same six with `_c00_eff065` (only the predicted strict errors change) |
| A6 | – | in the notebook: (Δη, Δφ) maps, profiles, p_T^Z dependence |
| B1 | `$ANA/root/pbpb/<stem>_hadrons.root` (100), `zjet_c00_campaign.root`; `$ANA/root/pp/…` | – |
| B3 | `$ANA/pools_cxx/pbpb/<stem>_pool.root` (100), `$ANA/pools_cxx/pp/…` | – |
| B4 | `mixing_summary_c00_cxx.txt` | the six figures with `_c00_cxx`: equal to A1's |
| B5 | stdout | – |

(`zjet_test/compare.py`'s `zjet_vs_cms2507_*.png` are step 1 on the 96-event test set; on a
campaign `mix.py`'s `mixing_reference_d*` are the same figure, check (1) shows the equality.)

## The final figures and how they compare with data

All against CMS 2507.09307's HEPData (ins2945894, in `zjet_test/data/hepdata/`): the Pb+Pb
0–30 % and pp tables per track p_T class ("Δφ_ch,Z in PbPb 0-30%", "Δφ_ch,Z spectra in pp",
"Δy_ch,Z in PbPb 0-30%", "Δy_ch,Z spectra in pp"), with their statistical and systematic errors,
drawn as stat ⊕ syst (black). In every figure the columns are the track p_T classes 1–2, 2–4,
4–10 GeV; y is d⟨ΔN_ch⟩/dΔφ (per Z per radian) or d⟨ΔN_ch⟩/dΔy (per Z per unit), after the ±
merging and the factor 0.5; x is Δφ_ch,Z from −π/2 to 3π/2 in 20 bins of π/10 (CMS's binning and
reflected range), or Δy_ch,Z on the Z side (|Δφ| < π/2) over |Δy| < 4 in bins of 0.8 (Pb+Pb) and
0.4 (pp), as CMS. Pb+Pb − pp: the model's Pb+Pb minus its pp, against CMS's PbPb minus pp (errors
in quadrature; CMS's pp rebinned to 0.8 in Δy). The summary gives the value at the first bin
past 0 (Δφ ≈ π/20, Δy 0–0.8: next to the Z, the wake's depletion) and χ²/ndf over the
independent bins (10 in Δφ, 5 in Pb+Pb Δy, 10 in pp Δy) with the model's errors ⊕ CMS's stat ⊕
syst, bin-to-bin correlations ignored as CMS does.

**Step 1: `mixing_reference_dphi_c00.png`, `mixing_reference_dy_c00.png`** — 3 rows: pp,
Pb+Pb 0–30 %, Pb+Pb − pp; the factorized model (red) against CMS: the model's prediction of
CMS's observable, the comparison with data once step 2's closure holds (else steps 2–3, which
went through the same mixing as the data).

**Steps 1–4: `mixing_dphi_c00.png`, `mixing_dy_c00.png`** — 2 rows: Pb+Pb 0–30 % and
Pb+Pb − pp; the four steps side by side (factorized red, all blue, cms10 green, strict orange)
with their errors (jackknife over the backgrounds; strict: the spread over its Z's). How each
experimental ingredient moves the result (it should not, within the errors, if the closure
holds) and widens the errors.

**Step 4: `mixing_errors_dphi_c00.png`, `mixing_errors_dy_c00.png`** — 1 × 3, Pb+Pb only, log
scale: the statistical error per bin of every step, the strict error scaled to CMS's Z count (×
√(N_eff / 1550), orange diamonds), the three predictions of check (6) at CMS's count (purple:
with each event's flat multiplicity level dashed, without it dash-dotted, Poisson only dotted),
and CMS's statistical error (black, HEPData's stat column). The question: are the model's
event-by-event fluctuations those of the data, and which error definition did CMS use.

**`mixing_summary_c00.txt`** — the step-1 block (pp, Pb+Pb, Pb+Pb − pp against CMS), the checks
(1)–(6), the table of every step against CMS (Pb+Pb and Pb+Pb − pp), the errors table and the
noise budget per Z.

## 3. Afterwards

- Keep `$ANA/figures/` private; note in the campaign log the image digest, the commit of
  this checkout (`git -C $REPO rev-parse --short HEAD`) and the flagged jobs.
- With more campaigns: one glob over all of them (`"/data/zjet_c0*/gpu*/*_hadrons_jet_frag.h5"`),
  a fresh `--pools` directory (the weights change), `--blocks 100` still.
- The plan for the campaigns' results, and what to look at in which order:
  `zjet_test/figures/MIXING.md` §8 in the GB10's checkout (git-ignored), in short: closure,
  exclusion bias and strict pulls on the first campaign; the strict error comparison and the
  error definition with all ten.

## Appendix: the pools and the naive mixing loop

Event mixing is usually written as a loop over the other events' tracks:

```
for each Z event i:
    for each track t of event i:                       # S
        fill S(η_t − y_i, φ_t − φ_i)
    for each of M partner events j:                    # B
        for each track t of event j:
            fill B(η_t − y_i, φ_t − φ_i), weight 1/M
ΔN = (S − B) / N_Z
```

The pools (*The pools: what the map step keeps*) store each piece of that loop at the point
where it no longer depends on the pairing.

1. **S, the own tracks.** S needs only event i's tracks and its own Z, so the map step bins it
   once: `S_full` and `S_one` are the per-Z Δy/Δφ distributions, already in the Z's frame.
   `S_full` averages the 40 oversamples (the loop over all 40 tracks lists, divided by 40),
   `S_one` takes one of them.
2. **B from a few partners (steps 3, 4) is the naive loop.** A partner's Δy/Δφ histogram cannot be
   stored in advance, it depends on the Z it is paired with (y_i, φ_i). So the pool keeps each
   event's tracks in the lab frame (`trk_*`, one oversample), and the reduce step
   (`zjet_mixing.explicit_partners`) runs the inner loop literally: for Z i, its 10 partners j,
   every track, η − y_i and φ − φ_i.
3. **B from all partners (step 2) is the same loop, summed first.** Pairing Z i with every track
   of every other event gives

   ```
   B_i(Δy, Δφ) = Σ_j Σ_{t∈j} δ(Δy − (η_t − y_i)) δ(Δφ − (φ_t − φ_i)) = ρ(η = Δy + y_i, φ = Δφ + φ_i),
   ```

   ρ the summed lab-frame (η, φ) density of all partner tracks. Instead of looping over the tracks
   of thousands of events per Z, the pool stores ρ (`R_full`, summed per background) and every Z
   gets ρ shifted by its own (y_i, φ_i) (`ZKernel`, which even groups the Z's that shift alike).
   The result is the naive loop's up to the binning of ρ (η 0.1, φ 6°: ±0.05 in Δy, ±3° in Δφ).
   The cost: ~5,600 Z's × 5,600 events × ~1,600 tracks ≈ 5 × 10¹⁰ track pairs per oversample for
   a campaign, against one shift of a 3 × 48 × 60 histogram per Z.
4. **Step 1 (factorized) is the same with what tracks cannot hold.** Its content is the
   Z-correlated yield, `bulk_jet` − `bulk_bg` sample by sample plus the fragments: a difference
   of histograms that can be negative, so only histograms work: `SW_zc` for S, `R_zc` for B (flat
   in Δφ, in Δy the η density folded with each y_Z).

**The per-background sums** make the variations of the loop subtractions: the exclusion "not
the Z's own background" is the total minus that background's sum; the jackknife leaves a block
of backgrounds out without a new loop; step 1's i ≠ j rule removes the self pairs with the w²
terms (`T_*`).

**Weights.** The weighted naive loop fills S with w_i and B with w_i·w_j (normalized); the pools
carry w per Z and Σw per background for that.

In short: S is stored as the histograms usually filled, B from a few partners as the partners'
tracks, B from all partners as their summed density.
