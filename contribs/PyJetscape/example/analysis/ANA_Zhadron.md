# ANA_Zhadron — the Z–hadron (wake) analysis of a campaign, step by step

CMS 2507.09307 (Z–hadron Δφ and Δy, Pb+Pb 5.02 TeV 0–30 % and pp; HEPData ins2945894) on the
first campaign, `zjet_c00`: 9,600 Pb+Pb events from the 2 × RTX 3090 box, in two GPU
subdirectories. Two routes to the same final stage: **A**, Python on the HDF5 files;
**B**, ROOT/C++ map step on the converted files, then the same Python final stage. The
Z+jet analysis of the same events is [ANA_Zjet.md](ANA_Zjet.md); the method in detail:
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

## 1. Set up (once)

```bash
git clone -b zjet https://github.com/jhputschke/js-contrib ~/js-contrib-zjet   # >= f741862
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

## A. Python on the HDF5 files

**A1. Pools and mixing** (map: one process per production file; reduce: every mode):

```bash
cd $EX/analysis/zjet_test
python mix.py --pbpb "$DATA/gpu*/*_hadrons_jet_frag.h5" --vacuum "$PP/*_hadrons_jet_frag.h5" \
    --pools $ANA/pools --hists '' --jobs 8 --outdir $ANA/figures --tag _c00 \
    > $ANA/logs/mix_c00.log 2>&1
# the same with the Poisson part of the predicted strict errors for CMS's corrected tracks
python mix.py --pbpb "$DATA/gpu*/*_hadrons_jet_frag.h5" --vacuum "$PP/*_hadrons_jet_frag.h5" \
    --pools $ANA/pools --hists '' --efficiency 0.65 --outdir $ANA/figures --tag _c00_eff065
```

- The pools land in `$ANA/pools/pbpb` and `$ANA/pools/vacuum` (compressed `.npz`, a manifest);
  the second command reuses them. They are rebuilt by themselves when the files, the selection
  or the seed change (add a campaign: everything is rebuilt, the weights change).
- `--hists ''`: check (1) compares with the 96-event test's `compare.py` and is skipped here.
- Cost (estimated from the test): the map a few seconds per file and ~2–3 GB per process, so
  `--jobs 8` takes minutes and ~20 GB; the reduce a few minutes.
- Options: `--n-mix` (10, CMS's partners), `--blocks` (100 jackknife blocks of backgrounds, 4
  backgrounds each here), `--n-cms` (1,550, CMS's estimated 0–30 % Z count), `--strict-split`
  (10), `--seed` (1).

**A2. Read `$ANA/figures/mixing_summary_c00.txt`** (the checks first):

| check | expect on one campaign |
|---|---|
| (2) own-background rule, bulk off − factorized | within the errors |
| (3) closure: the bulk term against 0 | χ²/ndf ≈ 1 with 100 blocks (a real test now; the test sample's 4 blocks could not do it) |
| (4) exclusion bias | predicted = measured; ≲ 0.03 per radian at 1–2 GeV (O(1 / 400 backgrounds)) |
| (5) strict spread error against 10 subsamples | pull widths ≈ 1 for Pb+Pb now (≥ 220 backgrounds), as for pp |
| (6) noise budget per Z | the flat multiplicity level's share and the fluid term from 400 fluids (the test had 4) |

Then the modes against CMS (Pb+Pb, Pb+Pb − pp) and the errors:

- **factorized** is the physics result (the paired wake + fragments; jackknife over 100 blocks);
  **all** is the experiment's method on the same events, its difference from factorized the
  closure; **cms10** CMS's procedure literally; **strict** the one-to-one emulation (400 Z's,
  one per background).
- **The error comparison** (`mixing_errors_d{phi,y}_c00.png`, the "ratio" column): the strict
  error at CMS's Z count against CMS's statistical error, and the three predictions of check (6)
  (with each event's flat level, without it, Poisson only), with `--efficiency 1` and 0.65.
  Which one matches tells how CMS's errors were made, or how realistic the model's fluctuations
  are; at 4–10 GeV the model's errors can come out small, its background having no minijets
  from the other binary collisions.
- Figures: `mixing_dphi_c00.png`, `mixing_dy_c00.png`. What every mode and check means:
  `zjet_test/README.md`, *Event mixing*.

**A3. Optional: maps and profiles.** `zjet_wake.ipynb` (in this folder) draws the (Δη, Δφ)
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

**B4. The final stage on the C++ pools**:

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

## 3. Afterwards

- Keep `$ANA/figures/` private; note in the campaign log the image digest, the commit of
  this checkout (`git -C $REPO rev-parse --short HEAD`) and the flagged jobs.
- With more campaigns: one glob over all of them (`"/data/zjet_c0*/gpu*/*_hadrons_jet_frag.h5"`),
  a fresh `--pools` directory (the weights change), `--blocks 100` still.
- The plan for the campaigns' results, and what to look at in which order:
  `zjet_test/figures/MIXING.md` §8 in the GB10's checkout (git-ignored), in short: closure,
  exclusion bias and strict pulls on the first campaign; the strict error comparison and the
  error definition with all ten.
