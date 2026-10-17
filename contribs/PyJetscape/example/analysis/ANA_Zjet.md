# ANA_Zjet — the Z+jet analysis of a campaign, step by step

CMS 1702.01060 (Z+jet in Pb+Pb 5.02 TeV 0–30 % and pp: Δφ_jZ, x_jZ, ⟨x_jZ⟩, R_jZ; HEPData
ins1512112) on the first campaign, `zjet_c00`: 9,600 Pb+Pb events from the 2 × RTX 3090 box, in
two GPU subdirectories. Two routes to the same final stage: **A**, the Python map step on the
HDF5 files; **B**, the C++ map step (FastJet) on the converted files. Both find the same jets
(*B5*); the final stage `zjet_test/zjet_jets.py` is shared. The Z–hadron analysis of the same
events, and the shared steps 0–2 in detail: [ANA_Zhadron.md](ANA_Zhadron.md); the jet map step
in detail: [`../analysis_root/zjet/README.md`](../analysis_root/zjet/README.md).

**Results are not public yet:** write the outputs (`--outdir`) outside the repository.

## 0.–2. Data, set-up, checks

As in [ANA_Zhadron.md](ANA_Zhadron.md) §0–2, with these variables:

```bash
REPO=~/js-contrib-zjet; EX=$REPO/contribs/PyJetscape/example        # js-contrib zjet >= f741862
DATA=/data/zjet_c00     # gpu0/ (seeds 1-50), gpu1/ (51-100): PbPb_0_30_zjet_c00_seedNNNN_hadrons_*.h5
PP=/data/zjet_pp        # vacuum_jet_zjet_pp_seed900001_hadrons_jet_frag.h5 (12,000 events, 40 fragmentations)
ANA=/data/ana_c00; mkdir -p $ANA/logs
cd $EX/analysis/zjet_test && ./make_data.sh hepdata_zjet      # CMS 1702.01060's HEPData, once
```

In addition to ANA_Zhadron's Python packages: **FastJet** for Python (`pip install fastjet`,
3.5) for route A, uproot and awkward for the final stage, and for route B the C++ FastJet
(`fastjet-config` on `PATH`, e.g. conda-forge `fastjet`) with ROOT ≥ 6.34. The jets use all
particles: the campaign files store them all (|η| < 2.6), the 96-event test set of
`zjet_test/data/pbpb` holds charged ones only.

## What the map step does (both routes)

Per Z event (60 < m < 120 GeV, p_T^Z > 30 GeV; the paper's cuts come in the final stage) and per
oversample k (all 40 by default, `--samples N` fewer): the whole event, `bulk_jet` k +
`jet_frag` (k mod 40), all final particles but neutrinos and the Z's two leptons;

- **jets**: anti-k_T R = 0.3, |η| < 2, p_T > 5 GeV after the subtraction;
- **ρA subtraction**: explicit ghosts on a fixed grid (|y| < 2.6, area 0.01), ρ(jet) the
  median p_T / area of the k_T R = 0.4 jets with |y| < 2.2 in the strip |y − y_jet| < 1;
- **embedding**: the fragments alone (reference jets, p_T > 10 GeV) and embedded into
  `bulk_bg` sample k, which has no wake: the subtraction's offset and resolution;
- **random cones**: 4 per oversample in `bulk_bg` (|η| < 1.6): the background's fluctuations,
  to set against CMS's N term;
- pp: the fragmentation k alone, no subtraction, no embedding or cones.

## A. Python on the HDF5 files

**A1. The map step**, in 8 slices of the 100 files (`--files i:j`; the weights come from all
files, so the slices combine):

```bash
cd $EX/analysis/zjet_test
for i in 0 13 26 39 52 65 78 91; do
  python zjet_jets_map.py "$DATA/gpu*/*_hadrons_jet_frag.h5" $ANA/jets_py/pbpb \
      --files $i:$((i + 13)) > $ANA/logs/jets_py_$i.log 2>&1 &
done; wait
python zjet_jets_map.py "$PP/*_hadrons_jet_frag.h5" $ANA/jets_py/pp > $ANA/logs/jets_py_pp.log 2>&1
```

- Cost (estimated from the test): ~0.13 s per Pb+Pb oversample, ~75 Z events × 40 oversamples
  per file, so ~6–7 min per file and ~1.5 h with 8 processes; ~2–3 GB per process (a file's
  hadrons in memory). `--samples 10` makes it 4× faster with 10 oversamples per event.
- pp: ~1 min (the test's 9,700 Z events × 10 fragmentations took 15 s).
- Output: one `<stem>_jets.root` per production file (trees `jet_events`, `jets`).

**A2. The final stage**:

```bash
python zjet_jets.py --pbpb $ANA/jets_py/pbpb --pp $ANA/jets_py/pp \
    --outdir $ANA/figures --tag _c00 --blocks 100
```

CMS 1702.01060's cuts: 70 < m < 110 GeV, p_T^Z > 40 GeV, muons |η| < 2.4, electrons |η| < 2.5
outside 1.44–1.57; jets |η| < 1.6, ΔR > 0.4 from the leptons, p_T > 30 GeV after the smearing.
Weights: window cross section / events × N_coll (Pb+Pb), every oversample 1/40. Errors: a
jackknife over 100 blocks of the 400 hydro backgrounds (pp: of events). Seconds.

**A3. Read `$ANA/figures/zjet_jets_summary_c00.txt`** and `zjet_jets_c00.png`,
`zjet_jets_embedding_c00.png`:

- **The Z counts** (pp, Pb+Pb with CMS 1702.01060's cuts) and the number of backgrounds.
- **Random cones**: the mean of δp_T (≈ 0 if ρ is unbiased) and its width σ_RC against CMS's
  PbPb N = 8.08 GeV: how the model's background (hydro hadrons, no minijets from the other
  binary collisions) fluctuates against the data's, over 400 backgrounds.
- **Embedding**: the matching efficiency and ⟨p_T,emb − p_T⟩ and its spread per reference-p_T
  class. An offset that grows with p_T is a subtraction bias to understand before reading
  x_jZ.
- **The comparisons**: pp smeared to PbPb's resolution against CMS's smeared-pp tables (data1, 3,
  5, 7), pp with pp's resolution against the unsmeared pp x_jZ (data9), and Pb+Pb in four
  variants against the PbPb tables (data2, 4, 6, 8). The CMS smearing is optional:
  `none` (the subtracted particle-level jets), `detector` (C and S of PbPb: the model's own
  fluctuations stand for N), `full` (C, S and N: double counts them), `match` (C, S and
  √(N² − σ_RC²)). With σ_RC ≥ N, `match` equals `detector`.

## B. ROOT: the C++ map step, the same final stage

**B1. Convert to ROOT** (once, shared with ANA_Zhadron §B1):

```bash
P=$EX/prod_AuAu_0_10_jet
python $P/run_h5toROOT.py $DATA/gpu0 $DATA/gpu1 -j 8 --out-dir $ANA/root/pbpb \
    --campaign-file $ANA/root/pbpb/zjet_c00_campaign.root
python $P/run_h5toROOT.py $PP --out-dir $ANA/root/pp --campaign-file $ANA/root/pp/zjet_pp_campaign.root
python $P/root_export/check_h5_vs_root.py "$DATA/gpu*/*_hadrons_jet_frag.h5" $ANA/root/pbpb \
    --campaign $ANA/root/pbpb/zjet_c00_campaign.root | tail -1     # "0 difference(s)"
```

**B2. Build**: `cd $EX/analysis_root/zjet && make` (ROOT and FastJet via `root-config`,
`fastjet-config`).

**B3. The map step in C++**, in slices:

```bash
cd $EX/analysis_root/zjet
for i in 0 13 26 39 52 65 78 91; do
  ./zjets $ANA/root/pbpb $ANA/jets_cxx/pbpb --files $i:$((i + 13)) > $ANA/logs/zjets_$i.log 2>&1 &
done; wait
./zjets $ANA/root/pp $ANA/jets_cxx/pp
```

0.12 s per Pb+Pb oversample (measured on the test), ~0.35 GB per process; about as fast as the
Python route (FastJet clusters in both), the pp in seconds. Options: `--samples N`, `--cones 4`,
`--ptmin 5`, `--seed 1`.

**B4. The final stage on the C++ jets**:

```bash
cd $EX/analysis/zjet_test
python zjet_jets.py --pbpb $ANA/jets_cxx/pbpb --pp $ANA/jets_cxx/pp \
    --outdir $ANA/figures --tag _c00_cxx --blocks 100
```

**B5. C++ against Python**: route A's and B's outputs side by side, or a quick check on two
files with 4 oversamples:

```bash
python zjet_jets_map.py "$DATA/gpu*/*_hadrons_jet_frag.h5" $ANA/jets_py_s4 --files 0:2 --samples 4
$EX/analysis_root/zjet/zjets $ANA/root/pbpb $ANA/jets_cxx_s4 --files 0:2 --samples 4
python check_cxx.py jets $ANA/jets_py_s4 $ANA/jets_cxx_s4 | tail -1     # 0 difference(s)
python check_cxx.py jets $ANA/jets_py/pp $ANA/jets_cxx/pp | tail -1
```

Every jet, area, ρ, embedding pair and random cone agrees to ~10⁻¹⁴ (on the test: 213,682 pp
jets identical, Pb+Pb to 10⁻¹⁴). With the same jets the two summaries of A2 and B4 agree too:
the smearing's random numbers are drawn in the same order from `--seed`.

## 3. Afterwards

- Keep `$ANA/figures/` private; note the image digest and this checkout's commit.
- More campaigns: one glob (`"/data/zjet_c0*/gpu*/*_hadrons_jet_frag.h5"`) for route A, all
  campaigns converted into one ROOT directory for route B.
- Open in the method: a CMS-like iterative subtraction next to ρA; the model's minijet-less
  background (the random cones tell how much it matters for R = 0.3);
  [`docs/Plans/PLAN_ZJET_CMS10x.md`](../../../../docs/Plans/PLAN_ZJET_CMS10x.md) §5c.
