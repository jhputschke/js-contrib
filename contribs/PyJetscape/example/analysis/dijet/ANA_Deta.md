# ANA_Deta — CMS's η-separated dijet wake with the Z as jet1, step by step

CMS HIN-25-012 ([arXiv:2602.19431](https://arxiv.org/abs/2602.19431), HEPData ins3123406)
observed the diffusion wake by comparing dijet–hadron correlations for dijets with a large
rapidity gap (R_asym) and a small one (R_sym). The Z+jet campaigns carry the same analysis
with the Z as jet1: the plan is [`docs/Plans/PLAN_dijet_deta_wake.md`](../../../../../docs/Plans/PLAN_dijet_deta_wake.md)
(its decisions in §6), the library [`../jetcorr/deta_asym.py`](../jetcorr/deta_asym.py), the
final stage [`deta.py`](deta.py). The Z+jet steps it builds on: [ANA_Zhadron.md](../zjet/ANA_Zhadron.md),
[ANA_Zjet.md](../zjet/ANA_Zjet.md).

**Results are not public yet:** the outputs go into `runs/` (git-ignored).

## The measurement and the model's ladder

CMS: anti-k_T R = 0.4; jet1 > 130 GeV, |η| < 1; jet2 > 50 GeV, |η| < 2, Δφ > 5π/6; gap classes
|Δη_jet1,jet2| < 0.5 (R_sym), 0.5–1, 1–1.5, 1.5–2 (R_asym); charged tracks 1–2 and 2–4 GeV in
|η| < 2.4 against jet1's axis, per dijet, every event reflected so that η_jet1 > η_jet2 (jet2
and its wake at negative Δη); pair acceptance from mixed events; near side |Δφ| < π/2
projected on Δη; the level of 1.5 < Δη < 2.5 subtracted per class; R_asym − R_sym, Pb+Pb − pp,
and its integral over −2.4 < Δη < −0.6 (CMS's 10 bins of 0.2 centred there: −2.5 … −0.5).

Here (the plan's §6): jet1 = the Z of the pools (2507.09307's selection), |y_Z| < 1, p_T^Z >
60 GeV; jet2 = the leading R = 0.4 jet above 30 GeV, Δφ > 5π/6, below jet1 (x_jZ < 1). Per
event the fraction of its 40 oversamples in each class weighs its contents:

| | content | role |
|---|---|---|
| T | wake + fragments (`Y_zc`) | the truth: no bulk, geometric acceptance |
| W | the wake (`Y_zc − Y_frag`) | what is wake |
| F | the full event, 40 samples (`Y_full`) | experiment-like: mixed event and sideband |
| O | one oversample (`Y_one`), its own jets | the strict emulation |
| N | `bulk_bg` + the same fragments | closure: must be flat |
| B | the bulk alone | closure: what the bulk leaks |
| pp | the vacuum events | the baseline (F, O) |

## 0. Set-up (once)

```bash
EX=<repo>/contribs/PyJetscape/example        # conda env js_fno (rhicML: the analysis env)
A=$EX/analysis; RUNS=$A/runs
C=00                                         # the campaign, then 01, ...
PB=/data/zjet_c$C                            # its hadron files (the GPU subdirectories: a glob)
PP=/data/zjet_pp                             # GB10: $RUNS/zjet_test/data/vacuum_all
cd $EX/analysis_root/zjet && make FASTJET_PY=$(python -c 'import fastjet, os; print(os.path.dirname(fastjet.__file__))')
```

The C++ map steps against the Python wheel's FastJet (plan A4: one FastJet for both routes).
Old c00 outputs from before the layout move: `runs/zjet_c00/{data,figures}` (`zjet/README.md`).

## 1. The map step at R = 0.4, once per campaign (plan A5)

Every output carries its file's record (`file_info`) and the settings (`run_info`): a new
campaign maps only itself, and combinations are lists of runs at load time.

```bash
# Python on the HDF5 files, one process per --files slice (~0.3 s per Pb+Pb oversample)
N=$(ls $PB/*_hadrons_jet_frag.h5 | wc -l)
for i in $(seq 0 $((N - 1))); do
  python $A/zjet/zjet_jets_map.py "$PB" $RUNS/zjet_c$C/data/jets_r04 --R 0.4 --files $i:$((i + 1)) &
done; wait
# or C++ on the converted files (route B, ANA_Zjet.md): the same jets
python $EX/prod_AuAu_0_10_jet/run_h5toROOT.py "$PB" -j 8 --out-dir $RUNS/zjet_c$C/data/root
$EX/analysis_root/zjet/zjets $RUNS/zjet_c$C/data/root $RUNS/zjet_c$C/data/jets_r04 --R 0.4 --files 0:10
# pp (once)
python $A/zjet/zjet_jets_map.py $PP $RUNS/zjet_pp/data/jets_r04 --R 0.4
```

## 2. The pools (version 4), once per campaign

```bash
python $A/zjet/mix.py --pbpb "$PB" --pools $RUNS/zjet_c$C/data/mix --vacuum $PP --jobs 16 --pools-only
#   -> runs/zjet_c$C/data/mix/pbpb/*.npz (and mix/vacuum for the pp, once)
# C++: $EX/analysis_root/zjet/zhadron_pools $RUNS/zjet_c$C/data/root $RUNS/zjet_c$C/data/mix_cxx/pbpb
```

## 3. Checks (plan A4)

```bash
# campaigns mapped apart and loaded together = one mapping (the first files of c00 and c01)
python $A/zjet/test/combine_check.py --campaign "/data/zjet_c00/gpu0/*seed000[12]_*" \
    --campaign "/data/zjet_c01/gpu0/*seed010[12]_*" --out <scratch>/combine --samples 2
# the C++ map steps against the Python ones at R = 0.4 (one file, --samples 2)
python $A/zjet/zjet_jets_map.py "$PB" <scratch>/py --R 0.4 --samples 2 --files 0:1
$EX/analysis_root/zjet/zjets $RUNS/zjet_c$C/data/root <scratch>/cxx --R 0.4 --samples 2 --files 0:1
python $A/zjet/check_cxx.py jets <scratch>/py <scratch>/cxx
```

## 4. The jets at R = 0.4: QA and counts

```bash
python $A/zjet/zjet_jets_qa.py --pbpb $RUNS/zjet_c$C/data/jets_r04 --pp $RUNS/zjet_pp/data/jets_r04 \
    --run zjet_c$C --tag _c${C}_r04                 # rho A at R = 0.4, sigma_RC, JES/JER, fakes
python $A/dijet/gap_count.py $RUNS/zjet_c$C/data/jets_r04 $RUNS/zjet_pp/data/jets_r04
```

`deta.py` prints the **fake jet2 per class** (the cuts on the Z's own `bulk_bg`, the same
oversample) relative to the full events' jet2: the input of D2's revisit (60/30 against 60/40).

## 5. The analysis

```bash
D="--pools $RUNS/zjet_c$C/data/mix/pbpb --vacuum-pools $RUNS/zjet_c$C/data/mix/vacuum \
   --jets $RUNS/zjet_c$C/data/jets_r04 --pp-jets $RUNS/zjet_pp/data/jets_r04 --run zjet_c$C"
python $A/dijet/deta.py $D --tag _c$C                          # the baseline
python $A/dijet/deta.py $D --tag _c${C}_noxcut --no-xcut       # the variants (B4.7)
python $A/dijet/deta.py $D --tag _c${C}_zcone --jet1 zcone
python $A/dijet/deta.py $D --tag _c${C}_b2b --jet2-pick b2b
python $A/dijet/deta.py $D --tag _c${C}_6040 --jet2 40
python $A/dijet/deta.py $D --tag _c${C}_10050 --zpt 100 --jet2 50
python $A/dijet/deta.py $D --tag _c${C}_10050c --zpt 100 --jet2 50 --central   # the CoLBT-like cut
```

**A combination** (e.g. c00 + c01): every option takes several directories, figures to the
combination's run:

```bash
python $A/dijet/deta.py --pools $RUNS/zjet_c00/data/mix/pbpb $RUNS/zjet_c01/data/mix/pbpb \
    --vacuum-pools $RUNS/zjet_c00/data/mix/vacuum --jets $RUNS/zjet_c00/data/jets_r04 \
    $RUNS/zjet_c01/data/jets_r04 --pp-jets $RUNS/zjet_pp/data/jets_r04 --run zjet_c00-01 --tag _c00-01
```

## Outputs

`runs/<run>/figures/`, each PNG with its `.npz`/`.root` (`../jetcorr/figdata.py`):

| file | shows |
|---|---|
| `deta_asym<tag>.png` | R_asym − R_sym per gap class (columns) at 1–2 and 2–4 GeV (rows): Pb+Pb F and T, pp, Pb+Pb − pp, in CMS's 0.2 bins; the signal range shaded |
| `deta_closure<tag>.png` | N, B, W, O and F − T |
| `deta_yw<tag>.png` | the Yang–Wang form: the Z-side dN/dη, \|η_jet2\| 0.8–1.6 minus < 0.8, and other minus same hemisphere |
| `deta_integrated<tag>.png` | the integral over −2.5 < Δy < −0.5 per class (CMS's Fig. 3 quantity) |
| `deta_summary<tag>.txt` | the selection, the effective Z's per class, the fake jet2, the integrals of every content, the closure χ², the noise per effective Z (R4), the Yang–Wang differences |

## Reading the closure (plan B4)

1. **Class sum:** the classes summed (unreflected, Z side) equal `S_full`'s Z side to rounding
   (float32 storage: ~10⁻⁸): the bookkeeping.
2. **N flat** and 3. **B flat:** χ² per bin ≈ 1 against zero. A residual in B is the bulk's
   dN/dη leaking through the mixed event (a shared bulk is exactly flat: `tests/test_deta_asym.py`).
4. **F = T:** the experiment-like path closes on the truth.
5. **The sideband's share:** T's level in 1.5 < Δy < 2.5 per class, what CMS's subtraction
   removes of the signal.
6. **pp flat:** CMS's pp is flat within its errors; the model's pp at high statistics can show
   a small class dependence of the underlying event, which Pb+Pb − pp removes.

The jackknife needs many backgrounds: with a few (the GB10's mini campaign: 8) the errors and
χ² are not meaningful, only the mechanics.

## The GB10 test (the mini campaign, 2026-10-18)

Four campaign-format Pb+Pb jobs of 48 events (seeds 102–105, two backgrounds each), two
pseudo-campaigns `runs/zjet_mini_c00`, `runs/zjet_mini_c01` (`PbPb_0_30_zjet_c0N_seedNNNN`),
the pp of `runs/zjet_test/data/vacuum_all`: for the code's checks, not for results.
