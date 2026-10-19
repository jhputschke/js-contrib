# ANA_Zjet — the Z+jet analysis of a campaign, step by step

CMS 1702.01060 (Z+jet in Pb+Pb 5.02 TeV 0–30 % and pp: Δφ_jZ, x_jZ, ⟨x_jZ⟩, R_jZ; HEPData
ins1512112) on the first campaign, `zjet_c00`: 9,600 Pb+Pb events from the 2 × RTX 3090 box, in
two GPU subdirectories. Two routes to the same final stage: **A**, the Python map step on the
HDF5 files; **B**, the C++ map step (FastJet) on the converted files. Both find the same jets
(*B5*); the final stage `zjet_jets.py` is shared. The Pb+Pb analysis is a ladder of
three steps, from the model-only ideal (step 1) to CMS's procedure (step 3). The Z–hadron
analysis of the same events, and the shared steps 0–2 in detail: [ANA_Zhadron.md](ANA_Zhadron.md);
the jet map step in detail: [`../analysis_root/zjet/README.md`](../../analysis_root/zjet/README.md).

**Results are not public yet:** write the outputs (`--outdir`) outside the repository.

## The measurement: S, B, and the ladder

CMS 1702.01060 pairs every Z (70 < m < 110 GeV, p_T^Z > 40 GeV; muons p_T > 10 GeV, |η| < 2.4,
electrons p_T > 20 GeV, |η| < 2.5 outside 1.44–1.57) with every jet of its event (anti-k_T R =
0.3, p_T > 30 GeV, |η| < 1.6, ΔR > 0.4 from the leptons), and measures, per Z:

- **Δφ_jZ** = |φ_jet − φ_Z|: (1/N_Z) dN_jZ/dΔφ_jZ for p_T^Z > 60 GeV;
- **x_jZ** = p_T^jet / p_T^Z: (1/N_Z) dN_jZ/dx_jZ for p_T^Z > 60 GeV and Δφ_jZ > 7π/8;
- **⟨x_jZ⟩** and **R_jZ** (the jet partners per Z, Δφ_jZ > 7π/8) in p_T^Z classes 40–50,
  50–60, 60–80, 80–120 GeV.

**S and B in CMS.** S are the Z–jet pairs of the Z's own event. In Pb+Pb some of those jets do
not come from the Z's parton-parton interaction: background fluctuations reconstructed as jets,
and other hard scatterings of the collision. CMS estimates this **B** by pairing each Z with the
jets of 40 minimum-bias events of the same centrality and vertex, and subtracts its jet spectrum
from the raw one (p. 3–4); in pp no B. The UE is subtracted from every jet (an iterative method),
the jets are calibrated but not unfolded: CMS gives the resolution σ(p_T)/p_T = √(C² + S²/p_T +
N²/p_T²) (Appendix A: C 0.061; pp S 0.95, N 0.001; PbPb 0–30 % S 1.24, N 8.08, N the
background's fluctuations), smears its pp data to PbPb's resolution for the comparison (the
"smeared pp" tables), and asks theory to smear generator-level jets without background with the
full PbPb σ.

**The ladder** (Pb+Pb; pp is the same in every step: the fragments' jets, no background):

| step | jets of the Z event (S) | subtracted (B) | adds | model only? |
|---|---|---|---|---|
| **1** | the fragments alone (`jet_frag`: the quenched jet, without bulk and without the medium response), the embedding's reference jets | – (no background, so no fake jets) | the reference: CMS's recipe for theory, the fragments smeared with the full σ_PbPb (and unsmeared) | yes |
| **2** | the whole event (bulk with the jet's wake + fragments), ρA-subtracted | the jets of the Z's **own** background (`bulk_bg`: the same fluid without the jet, sample k against sample k) | the background in the jets, the medium response, and fake jets subtracted with the model's twin | yes |
| **3** | the whole event, ρA-subtracted | the jets of 40 `bulk_bg` samples of the 20 **other** backgrounds closest in N_coll | B as CMS makes it: minimum-bias events of the same centrality | no: CMS's procedure |

The model's minimum-bias events are the `bulk_bg` samples: the fluid without any hard process.
Its background has no other hard scatterings (the only one is the Z + jet with Pythia's MPI), so
B holds the fake jets of the bulk's fluctuations only. Steps 2 and 3 come in four smearing
variants (CMS's smearing being optional): `none`, `detector` (C, S of PbPb: the model's own
fluctuations stand for N), `full` (C, S, N: counts them twice), `match` (C, S and
√(N² − σ_RC²), σ_RC the random cones' width, so that the total is CMS's N). Step 2 − step 1 is
what the background and the medium response do to the jets; step 3 − step 2 is what mixing
instead of the twin does to B.

**Which step is compared with the data.** Unlike the Z–hadron analysis (where step 1 is the
model's exact version of the measured S − B), here **step 1 is a pure model quantity**: jets
without background and without the medium response, which no measurement sees. The model's
version of the measured observable is **steps 2 and 3**: their jets carry the background and the
medium response as the data's do, ρA-subtracted, with the fakes removed by B. Step 3 is CMS's
procedure (B from minimum-bias events of the same centrality) and the like-for-like comparison;
step 2 is the same with the ideal B, and step 3 − step 2 tests CMS's B on the model. Step 1 is
the reference that shows what the background, its subtraction and the response add (and is what
CMS's recipe for theory approximates).

**The backgrounds' event planes are all near φ = 0** (3dMCGlauber's impact parameter along x in
every event; on `zjet_c00` 84 % of the 400 backgrounds within ±π/8). In data they are random. The
means of steps 2 and 3 are unaffected (φ_Z is uniform), but step 3's mixed backgrounds share the
Z's plane, so their fake jets (more in plane) line up with the Z event's own in Δφ_jZ and cancel
part of its fluctuations, which CMS's minimum-bias events cannot: step 3's errors would come out
too small. So `zjet_jets.py` (and `zjet_jets_qa.py`'s B_mixed, with the same angles) rotates every
background by its own random angle by default, as the Z–hadron mixing does (`mix.py`,
[ANA_Zhadron.md](ANA_Zhadron.md#the-measurement-s-b-and-the-ladder), *The backgrounds' event
planes*). Rotating a background with its Z events, their jets, its own jets and random cones is an
exact symmetry, so only B_mixed changes: a mixed jet of background b seen from a Z of background b'
turns by angle(b) − angle(b'). Steps 1 and 2 and pp do not depend on it; `--no-rotate` reproduces
the outputs from before 2026-10-17 (checked: every figure array and summary line).

**The settings** (`zjet_jets.py`): `--blocks` (100, jackknife blocks of the Z's backgrounds),
`--n-mixed` (40, step 3's `bulk_bg` samples per Z, CMS's 40 minimum-bias events), `--n-nearest`
(20, the other backgrounds closest in N_coll to draw them from), `--seed` (7, the smearing's and
the draws' random numbers), `--no-rotate` (keep the mixed backgrounds as produced),
`--rotate-seed` (the rotations; default from `--seed`; `zjet_jets_qa.py` takes the same). The map step: `--samples` (all 40 oversamples), `--cones` (4),
`--ptmin` (5 GeV), `--seed` (1).

## 0.–2. Data, set-up, checks

As in [ANA_Zhadron.md](ANA_Zhadron.md) §0–2, with these variables:

```bash
REPO=~/js-contrib-zjet; EX=$REPO/contribs/PyJetscape/example
DATA=/data/zjet_c00     # gpu0/ (seeds 1-50), gpu1/ (51-100): PbPb_0_30_zjet_c00_seedNNNN_hadrons_*.h5
PP=/data/zjet_pp        # vacuum_jet_zjet_pp_seed900001_hadrons_jet_frag.h5 (12,000 events, 40 fragmentations)
# PP=$EX/analysis/runs/zjet_setup/data/vacuum_all   # on the GB10: 12,000 events, 10 fragmentations (ANA_Zhadron §0)
ANA=/data/ana_c00; mkdir -p $ANA/logs
cd $EX/analysis/zjet/setup && ./make_data.sh hepdata_zjet      # CMS 1702.01060's HEPData, once
```

In addition to ANA_Zhadron's Python packages: **FastJet** for Python (`pip install fastjet`,
3.5) for route A, uproot and awkward for the final stage, and for route B the C++ FastJet
(`fastjet-config` on `PATH`, e.g. conda-forge `fastjet`) with ROOT ≥ 6.34. The jets use all
particles: the campaign files store them all (|η| < 2.6), the 96-event test set of
`../runs/zjet_setup/data/pbpb` holds charged ones only.

## QA first

The same QA pass as for the Z–hadron analysis: [ANA_Zhadron.md](ANA_Zhadron.md#qa-first-both-analyses-qapy-zqa)
(`qa.py map`, `zqa`, `qa.py plot`: the Z, its leptons and weights, the hadrons per source, the total
charged multiplicity, the species).

## What the map step does (both routes)

Per Z event (60 < m < 120 GeV, p_T^Z > 30 GeV; the paper's cuts come in the final stage) and per
oversample k (all 40 by default, `--samples N` fewer): the whole event, `bulk_jet` k +
`jet_frag` (k mod 40), all final particles but neutrinos and the Z's two leptons;

- **jets** (S): anti-k_T R = 0.3, |η| < 2, p_T,raw > 1 GeV and p_T > 5 GeV after the
  subtraction; per jet p_T,raw, p_T, η, φ, area, ρ, the mass, the number of its (non-ghost)
  constituents and the leading one's p_T;
- **ρA subtraction**: explicit ghosts on a fixed grid (|y| < 2.6, area 0.01), ρ(jet) the
  median p_T / area of the k_T R = 0.4 jets with |y| < 2.2 in the strip |y − y_jet| < 1;
- **embedding**: the fragments alone (reference jets, p_T > 10 GeV: step 1's jets) and embedded
  into `bulk_bg` sample k, which has no wake: the subtraction's offset and resolution. The
  embedded event's jets above 5 GeV are stored as well (`nw_*`): **the Z event without its
  wake** (the same fragments, fluid sample and random numbers), for the final stage's
  "no wake" variant (A8);
- **random cones**: 4 per oversample in `bulk_bg` (|η| < 1.6): the background's fluctuations,
  to set against CMS's N;
- per hydro background and `bulk_bg` sample: its **jets**, subtracted the same way (tree
  `bg_jets`, with the background's N_coll): steps 2 and 3's B;
- **the jet QA** (trees `qa_jets`, `qa_bg_jets`): the first `--qa-samples` (default 1)
  oversamples of every Z event and background with **all** their jets, without the cut on the
  subtracted p_T (the negative tail; a few hundred kB per file);
- pp: the fragmentation k alone, no subtraction, no embedding, cones or background jets.

## A. Python on the HDF5 files

**A1. The map step**, in 8 slices of the 100 files (`--files i:j`; the weights come from all
files, so the slices combine):

```bash
cd $EX/analysis/zjet
for i in 0 13 26 39 52 65 78 91; do
  python zjet_jets_map.py "$DATA/gpu*/*_hadrons_jet_frag.h5" $ANA/jets_py/pbpb \
      --files $i:$((i + 13)) > $ANA/logs/jets_py_$i.log 2>&1 &
done; wait
python zjet_jets_map.py "$PP/*_hadrons_jet_frag.h5" $ANA/jets_py/pp > $ANA/logs/jets_py_pp.log 2>&1
```

- Cost (measured on `zjet_c00`, the RTX box): ~75 Z events × 40 oversamples per file plus 4 ×
  40 background samples; ~0.37 s per Pb+Pb oversample in one process. With 16 processes 23 min
  per file and 2.8 h for the 100 files, 92 GB at peak (~5.3 GB per process). Since 2026-10-17
  ~1.5× faster with the same outputs, bit for bit (both map steps): ρ(jet) once per strip
  content, no ρ for jets that cannot pass the p_T cut, the background's particles and ρ kept
  per background and sample, the random cones' particles preselected. `--samples 10` makes it
  another 4× faster.
- pp: ~4 min for 12,000 events × 40 fragmentations (`zjet_pp`).

**A2. The final stage**:

```bash
python zjet_jets.py --pbpb $ANA/jets_py/pbpb --pp $ANA/jets_py/pp \
    --outdir $ANA/figures --tag _c00 --blocks 100
```

Weights: window cross section / events × N_coll (Pb+Pb), every oversample 1/40. Errors: a
jackknife over 100 blocks of the 400 hydro backgrounds (pp: of events). Seconds to a minute.

Then read `$ANA/figures/zjet_jets_summary_c00.txt` and the figures **in the order of the steps**:

**A3. The jets' diagnostics first** — the jet QA (seconds; `--hadron-qa`: the hadron QA's
output of §QA first, for ρ against dN_ch/dη; without it that line is left out):

```bash
python zjet_jets_qa.py --pbpb $ANA/jets_py/pbpb --pp $ANA/jets_py/pp \
    --hadron-qa $ANA/qa_py/pbpb --outdir $ANA/figures --tag _c00
```

Its six figures and `qa_jets_summary_c00.txt` (what each shows and what to look for:
*The jet QA figures* below), then `zjet_jets_embedding_c00.png` and the final stage's random-cone,
embedding and B lines:
- the random cones' mean δp_T (≈ 0 if ρ is unbiased) and width σ_RC against CMS's N = 8.08 GeV:
  how the model's background (hydro hadrons, no minijets from the other binary collisions)
  fluctuates against the data's, over 400 backgrounds; it sets the `match` smearing;
- the embedding per reference-p_T class: the matching efficiency, ⟨p_T,emb − p_T⟩ and its spread.
  An offset that grows with p_T is a subtraction bias to understand before reading x_jZ;
- the B lines: background jets per Z (R_jZ of B) for B_own and B_mixed against R_jZ of S alone:
  how much of S is fake.

**A4. pp, the baseline** — `zjet_jets_pp_c00.png`, the summary's "pp against CMS" block. Before
any Pb+Pb step: does the model's vacuum Z+jet describe CMS's pp? Every Pb+Pb conclusion rests on
it, and it is the one Z+jet comparison without a background. Two comparisons, each with the
smearing CMS's table has:
1. **pp at PbPb's resolution** (blue) against CMS's pp smeared to PbPb (`data1`, `data3`,
   `data5`, `data7`, black), all four observables: the pp the Pb+Pb tables are compared with,
   the top row of every step figure;
2. **pp at pp's resolution** (green) against CMS's pp x_jZ (`data9`, green diamonds), the only
   pp table at pp's own resolution.

The unsmeared model (grey, particle level, no table) shows what the smearing does: mostly x_jZ
and ⟨x_jZ⟩ at low p_T^Z, where the resolution is widest. The lower row: model / CMS, the bands
CMS's errors (stat ⊕ syst). A pp disagreement is the generator's (Pythia, the Z's and the jets'
p_T), not the medium's, and carries over into every Pb+Pb step.

**A5. Step 1, the reference** (a model quantity, not the measured observable) —
`zjet_jets_step1_c00.png`, the summary's "Step 1" block: the fragments' jets with CMS's theory
smearing (and unsmeared) against CMS's PbPb tables (bottom row): the jet's energy loss alone,
without background or medium response. The top row repeats A4's pp at PbPb's resolution.

**A6. Step 2** — `zjet_jets_step2_c00.png`: S − B_own in the four smearing variants. Against step 1:
what the background, its subtraction and the medium response (the wake's hadrons inside the
cone) do to the jets.

**A7. Step 3, the comparison with data** — `zjet_jets_step3_c00.png`: S − B_mixed, CMS's
procedure, against CMS's PbPb tables. Against step 2: whether mixing other backgrounds of similar
N_coll reproduces the twin's B (it should, within the errors, if centrality is the only thing
that matters for the fakes); if it does, step 2 is the same comparison with smaller errors.

**A8. The wake's effect** — `zjet_jets_wake_c00.png` and the summary's "Without the wake" and
"The wake's effect" blocks. Steps 2 and 3 are repeated, unsmeared, on the embedded event
(`bulk_bg` k + the same fragments; `nw_*`), the Z event without its wake. It shares everything
else with step 2: the fragments, the background's fluid sample and random numbers (correlated
sampling), the ρA subtraction, B, the fluctuations' threshold feed. So **step 2 − step 2
without the wake is what the wake does to Δφ_jZ, x_jZ, ⟨x_jZ⟩ and R_jZ**, with the jackknife of
the difference (the two share their events: far smaller errors than either). Step 2 − step 1
is not: it also contains the background's fluctuations and the subtraction. Top row: CMS,
step 2, step 2 without the wake, step 1 (all unsmeared); bottom row: the difference.

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
Python route (FastJet clusters in both), the pp in seconds.

**B4. The final stage on the C++ jets** (pp and all three steps, read as in A3–A7):

```bash
cd $EX/analysis/zjet
python zjet_jets_qa.py --pbpb $ANA/jets_cxx/pbpb --pp $ANA/jets_cxx/pp \
    --hadron-qa $ANA/qa_cxx/pbpb --outdir $ANA/figures --tag _c00_cxx
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

Every jet, area, ρ, embedding pair, random cone and background jet agrees to ~10⁻¹⁴ (on the
test: 213,682 pp jets identical, Pb+Pb to 10⁻¹⁴). With the same jets the two summaries of A2
and B4 agree too: the smearing's and the draws' random numbers come in the same order from
`--seed`.

## Outputs of every step

| step | writes | figures |
|---|---|---|
| A1 / B3 (map) | `$ANA/jets_py/pbpb/<stem>_jets.root` (100), `$ANA/jets_py/pp/<stem>_jets.root` (or `jets_cxx/`): trees `jet_events` (per Z event: the Z, its leptons, weight σ_k/N_k, N_coll, background, samples used), `jets` (per event and oversample: jets, embedding pairs, random cones), `bg_jets` (per background and `bulk_bg` sample: its jets, N_coll), `qa_jets`, `qa_bg_jets` (the first oversample's jets without the p_T cut) | – |
| A3 / B4 (jet QA) | `$ANA/figures/qa_jets_summary_c00.txt` (`_c00_cxx`) | `qa_jets_subtraction_c00.png`, `qa_jets_area_rho_c00.png`, `qa_jets_response_c00.png`, `qa_jets_geometry_c00.png`, `qa_jets_constituents_c00.png`, `qa_jets_mixing_c00.png` |
| A2 / B4 (final) | `$ANA/figures/zjet_jets_summary_c00.txt` (`_c00_cxx`) | `zjet_jets_pp_c00.png`, `zjet_jets_step1_c00.png`, `zjet_jets_step2_c00.png`, `zjet_jets_step3_c00.png`, `zjet_jets_wake_c00.png`, `zjet_jets_embedding_c00.png` |
| B1 | `$ANA/root/…` (shared with ANA_Zhadron) | – |
| B5 | stdout | – |

Every figure also comes with its `.npz` and `.root` figure data (ANA_Zhadron.md, *Outputs*:
restyle with `../jetcorr/replot.py` without rerunning anything).

## The jet QA figures

`zjet_jets_qa.py`, before the final stage: the jet reconstruction, not the physics. Jets per Z
with the Z's weights; S the Z events' jets, B_own the jets of the Z's own background, B_mixed
those of the 20 backgrounds nearest in N_coll (step 3's), pp the vacuum's; "Z+jet cuts" =
|η| < 1.6, ΔR > 0.4 from the Z's leptons, p_T > 30 GeV, without smearing. All from the QA trees
(one oversample, all jets) except the random cones and the embedding (every oversample).

| figure | panels | what to look for |
|---|---|---|
| `qa_jets_subtraction_c00.png` | p_T,raw; ρA; p_T = p_T,raw − ρA with its negative tail; p_T,raw against ρA (S); the fake fraction B/S against p_T, inclusive and at Δφ_jZ > 7π/8, for B_own and B_mixed; S − B_own against pp | about half the jets below 0 after subtracting the median (ρ is a median); a B/S still large at 30 GeV says how much of steps 2 and 3 is subtraction; S − B_own above ~30 GeV approaching the shape of pp's spectrum, below it by the quenching |
| `qa_jets_area_rho_c00.png` | A/πR² in p_T classes; ⟨A⟩/πR² against p_T; ρ per sample (Z events, `bulk_bg`); ρ against N_coll per background; ρ against η; ρ(Z event) − ρ(its own `bulk_bg`) | hard jets at A ≈ πR² (soft ones spread lower); ρ rising with N_coll, and its ratio to dN_ch/dη (summary) the same for every background; ρ(η) follows dN/dη of the hydro; ρ(Z event) − ρ(bulk_bg) ≈ 0: the wake and the fragments do not move the median (if they did, step 2 would subtract differently from data) |
| `qa_jets_response_c00.png` | random cones' δp_T (log: the tails) against Gaussians of σ_RC and of CMS's N; σ_RC and ⟨δp_T⟩ against N_coll and η; the embedding's (p_T,emb − p_T)/p_T (JES, JER) and matching efficiency against p_T, ΔR of the pairs | ⟨δp_T⟩ ≈ 0; σ_RC against CMS's 8.08 GeV (it sets the `match` smearing) and its growth with N_coll; non-Gaussian tails (the summary's fraction beyond 3σ, Gauss 0.27 %); JES → 0 and efficiency → 1 above 30 GeV |
| `qa_jets_geometry_c00.png` | η, Δφ to the Z, φ (after the cuts); jets per Z; ΔR(jet, nearest Z lepton) before the 0.4 cut; jet against Z p_T (S) | flat η and φ within the acceptance; B flat in Δφ, S and pp peaked at π; no pile-up of jets at ΔR → 0 from the leptons (they are removed from the particles; what the cut removes is the leptons' radiation, the summary gives the rate) |
| `qa_jets_constituents_c00.png` | constituents (after the cuts) and against p_T; p_T,lead / p_T,raw; m / p_T,raw (unsubtracted) | Pb+Pb jets hold the background's tens of soft particles (pp: about ten); fakes have a small leading fraction; the mass is the unsubtracted one (background-dominated in Pb+Pb), a QA quantity only |
| `qa_jets_mixing_c00.png` | ΔN_coll of the mixed backgrounds; their ⟨ρ⟩ − the own ρ; B_mixed / B_own against p_T | small ΔN_coll and Δρ ≈ 0, and the ratio ≈ 1: then step 3's B is step 2's, as the ladder assumes (needs ≥ 2 backgrounds: empty on one) |

`qa_jets_summary_c00.txt`: per set the jets per Z (all, below 0, above 30 GeV), the fake
fractions at 30–40 and 50–70 GeV, ρ per background and against dN_ch/dη, ρ(Z event) −
ρ(background) with its error, the random cones (mean, σ_RC, tails), the embedding's JES, JER and
efficiency, the lepton cut's rate, the constituents, the mixing's ΔN_coll and Δρ. Only σ_RC has a
number from the paper to compare with (CMS's N); the rest are the model's own checks.

## The final figures and how they compare with data

Against CMS 1702.01060's HEPData (ins1512112, `../runs/zjet_setup/data/hepdata_zjet/`), errors: CMS's
stat ⊕ syst (black) and the model's jackknife.

**`zjet_jets_pp_c00.png`, pp** — 2 rows × 4 columns: Δφ_jZ (log scale), x_jZ, ⟨x_jZ⟩ vs p_T^Z,
R_jZ vs p_T^Z (top), model / CMS with CMS's errors as bands (bottom). The model's vacuum jets
three ways: smeared with PbPb's resolution against the smeared-pp tables `data1`, `data3`,
`data5`, `data7`; smeared with pp's resolution (S 0.95, N 0.001) against `data9` (x_jZ, pp
unsmeared by CMS's definition: at pp's own resolution); unsmeared, for reference. The pp
selection, jets, weights and errors (a jackknife over blocks of events) are those of the Pb+Pb
steps. `setup/zjet_pp_jets.py` makes the same comparison from the HDF5 files with its own
jet finding (`figures/zjet_pp_vs_cms1702.png`): an independent cross-check, agreeing within the
smearing's random numbers.

**`zjet_jets_step1_c00.png`, `_step2_`, `_step3_`** — the same layout for every step, 2 rows × 4
columns: Δφ_jZ (log scale), x_jZ, ⟨x_jZ⟩ vs p_T^Z, R_jZ vs p_T^Z.
- **Top row, pp** (the same in every step, the blue points of the pp figure): the model's pp jets smeared with PbPb's resolution (the
  model is at particle level, so the whole σ_PbPb, which is where CMS's pp data smeared by
  √(σ_PbPb² − σ_pp²) end up) against CMS's "smeared pp" tables `data1` (Δφ_jZ), `data3` (x_jZ),
  `data5` (⟨x_jZ⟩), `data7` (R_jZ).
- **Bottom row, Pb+Pb 0–30 %**, against CMS's PbPb tables `data2`, `data4`, `data6`, `data8`:
  step 1 the fragments' jets (theory smearing, unsmeared: the reference, a model quantity);
  steps 2 and 3 S − B in the four smearing variants (the model's version of the measured
  observable; step 3 with CMS's B).
- Bins: Δφ_jZ edges 0, π/5, 2π/5, 3π/5, 7π/10, 4π/5, 9π/10, 19π/20, π; x_jZ 10 bins of 0.2 up to
  2; the four p_T^Z classes. In the summary, per observable and variant: the model values and
  χ²/ndf against the table (bins with zero CMS error left out: CMS's empty x_jZ bins, 0 ± 0). After subtracting B a class can come
  out empty or negative: ⟨x_jZ⟩ is then not given (NaN).

**`zjet_jets_embedding_c00.png`** — left: the random cones' δp_T = Σp_T − ρπR² in `bulk_bg`
(R = 0.3, |η| < 1.6), with a Gaussian of CMS's N = 8.08 GeV: the model's background fluctuations
against the data's. Right: embedded − reference p_T against the reference p_T: the ρA
subtraction's offset and resolution.

**`zjet_jets_summary_c00.txt`** — the Z counts after CMS's cuts (pp, Pb+Pb) and the number of
backgrounds, the B lines, the random cones, the embedding, the pp block (χ² against `data1`,
`data3`, `data5`, `data7` at PbPb's resolution and `data9` at pp's, the unsmeared values), and per
step the χ² lines of Pb+Pb. All χ² use the model's jackknife and CMS's errors in quadrature.

**Caveats of the comparison**
- **The muons**: CMS takes muons from p_T > 10 GeV, the production requires p_T > 20 GeV for both
  leptons (`--lepton-ptmin 20`, the Z–hadron selection's); the Z sample lacks the dimuons with
  one muon at 10–20 GeV. Per-Z observables are hardly affected; the Z counts are not CMS's.
- **Step 1 is not quite CMS's theory recipe**: the fragments' jets have no medium response (in
  the model it is in the bulk), and only the reference jets above 10 GeV are stored (the theory
  smearing can lift a 10-GeV jet above 30 GeV only rarely).
- **B**: the model's background has no other hard scatterings, so B holds fluctuations only; CMS's
  minimum-bias events also have minijets.
- Particle level against data corrected to particle level except the resolution; φ smearing
  neglected (CMS: negligible); the model's hadron definition: the "Read first" note of the plan.

## 3. Afterwards

- Keep `$ANA/figures/` private; note the image digest and this checkout's commit.
- More campaigns: one glob (`"/data/zjet_c0*/gpu*/*_hadrons_jet_frag.h5"`) for route A, all
  campaigns converted into one ROOT directory for route B.
- Open in the method: a CMS-like iterative subtraction next to ρA; the model's minijet-less
  background (the random cones tell how much it matters for R = 0.3);
  [`docs/Plans/PLAN_ZJET_CMS10x.md`](../../../../../docs/Plans/PLAN_ZJET_CMS10x.md) §5c.
