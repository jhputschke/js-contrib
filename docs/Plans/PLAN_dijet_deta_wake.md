# PLAN: CMS's η-separated dijet wake analysis (HIN-25-012), first on the Z+jet campaigns with the Z as jet1

2026-10-18. Why: CMS observed the diffusion wake with dijets separated in pseudorapidity
([arXiv:2602.19431](https://arxiv.org/abs/2602.19431), CMS-HIN-25-012, > 5σ at 1–2 GeV in
0–30 %). A dijet production with the full chain (a hydro run with the wake per event) is
expensive, so the analysis is first built, tested and closed on the Z+jet campaigns that already
exist, with the Z standing in for jet1. That is more than a prototype: Yang & Wang
([arXiv:2508.04194](https://arxiv.org/abs/2508.04194)) proposed the same rapidity asymmetry for
γ/Z+jet and computed it with CoLBT-hydro, so the Z version is a result in its own right. The Z
test also measures the noise per event that sets the size of the later dijet production.

> **Status 2026-10-18** (js-contrib branch `dijet_deta`): Phase A (A0–A3) and Phase B's code
> are implemented in Python and C++ (`jetcorr/{weights,mapio,deta_asym}.py`, `dijet/deta.py`,
> `dijet/ANA_Deta.md`, the map steps' `--R`, `file_info`/`run_info` and Z cone, pools version
> 4); the decisions of §6 are taken (D1–D8). Checked on the GB10 (A4): both map steps at R = 0.3
> unchanged in every old column, `zjet_jets.py`'s summary identical, `mix.py`'s identical but
> for rounding (10⁻¹⁵), `check_cxx.py` jets / pools / qa 0 differences (jets also at R = 0.4),
> and on a four-job mini campaign (two pseudo-campaigns) campaigns mapped and pooled apart =
> together (`zjet/test/combine_check.py`). **To do on rhicML:** A5 (the R = 0.4 map steps and
> v4 pools of c00, c01 and the pp), A4's checks on c00/c01, then B5 (`ANA_Deta.md`), and D2's
> revisit with the fake jet2 at R = 0.4. Seen on the GB10's stand-in pp (`vacuum_all`, ~3,800
> Z's with |y_Z| < 1): a near-uniform offset of R_asym − R_sym at 1–2 GeV in some classes
> after the sideband subtraction (a class-dependent η shape of the pp underlying event; a bulk
> the events share comes out exactly flat, `tests/test_deta_asym.py`); to be looked at with the
> campaigns' pp reference (CMS assigns 0.01–0.04 for "residual UE levels").

Written against js-contrib `zjet` 72cb15e; paths and line numbers follow the layout of branch
`analysis_layout` (43981bd, below). Data in hand: campaigns `zjet_c00` and `zjet_c01` (19,200
Pb+Pb 0–30 % events on 800 hydro backgrounds, `/data/zjet_c0*`), the c00 jet map step at R = 0.3,
the c00 and c00+c01 Z–hadron pools, and the pp reference. Items marked **verify** were inferred
from reading the code and have not been run.

**Where the code and the data are** (`analysis/` and `analysis_root/` are
`contribs/PyJetscape/example/…`; `analysis/zjet/README.md` has the layout):

| what | where |
|---|---|
| shared and trigger-free: run paths, portable random choices, event weights, jet finding (`Clusterer`, `StripRho`), figure data | `analysis/jetcorr/` (`runs.py`, `keys.py`, `weights.py`, `jetmap.py`, `figdata.py`); C++ `analysis_root/common/` (`AnaCommon.h`, `JetMap.h`) |
| the Z+jet workflow: QA, Z–hadron pools and mixing, the jet map step and final stages | `analysis/zjet/` (`zjet_jets_map.py`, `zjet_jets.py`, `zjet_jets_qa.py`, `mix.py`, `zjet_mixing.py`, `check_cxx.py`); C++ `analysis_root/zjet/` (`zjets`, `zhadron_pools`) |
| this analysis | `analysis/dijet/` (`gap_count.py`; new: `deta.py`, `ANA_Deta.md`), its library `analysis/jetcorr/deta_asym.py` (new) |
| data and figures | `analysis/runs/<run>/{data,figures}/`, git-ignored, one run per campaign; scripts take `--run` |

The runs of this plan. The c00 outputs move on the analysis host from the old code folder
`analysis/zjet_test/` (the "formerly" column is relative to it; `mv`, git-ignored):

| run | holds | formerly |
|---|---|---|
| `runs/zjet_c00/` | `data/jets/` (the map step at R = 0.3), `data/mix/` (pools), `data/logs/`, `figures/` (`ANALYSIS_c00.md`, `mixing_summary_c00.txt`, …) | `data/jets_c00/`, `data/mix_c00/`, `data/logs/`, `figures/` (the c00 files) |
| `runs/zjet_c01/` | c01, mapped on its own (A0) | – |
| `runs/zjet_c00-01/` | the combination's figures (on the GB10 also its pools and QA) | `data-c00-01/`, `figures-c00-01/` |
| `runs/zjet_test/` | the 96-event test (`zjet/test/`), the set-up samples, the GB10's stand-in pp reference `vacuum_all` | `data/`, `figures/` (the rest) |
| `runs/hepdata/` | CMS's HEPData records (`zjet/get_hepdata.sh`), the final stages' default `--hepdata` | `data/hepdata/`, `data/hepdata_zjet/` |

## 0. The measurement and the proxy

**CMS** (2602.19431): anti-k_T **R = 0.4**, particle flow, constituent subtraction. Jet1 p_T > 130 GeV,
|η| < 1.0; jet2 p_T > 50 GeV, |η| < 2.0; Δφ_jet1,jet2 > 5π/6. Gap classes |Δη_jet1,jet2| < 0.5
(R_sym) and 0.5–1.0, 1.0–1.5, 1.5–2.0 (R_asym). Charged tracks 1–2 and 2–4 GeV, |η| < 2.4,
correlated with jet1's axis in (Δη, Δφ), normalized per dijet. Each event is reflected so that
η_jet1 > η_jet2: jet2 and its wake then always sit at negative Δη. The pair acceptance is corrected
with mixed events (jet1 of one event with the tracks of 30 others, matched in v_z and centrality).
Near side |Δφ| < π/2 projected onto Δη; the uncorrelated level is taken from 1.5 < Δη < 2.5 and
subtracted from R_sym and R_asym separately; the signal is R_asym − R_sym in −2.4 < Δη < −0.6, and
PbPb − pp. Fig. 3 integrates it over that range against centrality.

What is measured is hadrons. The jets select the event, give the reference axis (jet1) and sort
the event into a gap class (jet2's η, which also fixes the reflection). The signal on the near
side is **jet2's** diffusion wake: opposite to jet2 in φ, at jet2's rapidity.

**The Z proxy:** jet1 = the Z, at (y_Z, φ_Z), with |y_Z| < 1 (rapidity, not η: for a Z of 50 GeV,
y = 1 is η ≈ 1.6). jet2 = the recoil jet from the map step at R = 0.4. The same classes,
reflection (when η_jet2 > y_Z), projection, sideband and subtraction.

What it captures: jet2's wake and the whole analysis chain, with the model's truth next to it.
What it misses:

- **Jet1's own near-side physics.** Its fragments, wake front and medium-induced radiation sit
  around Δη ≈ 0 to −1 and differ between classes; CMS names this as what weakens the small-gap
  signal. With the Z the near side is clean, so the proxy shows the full dip *and* the bump at
  Δη ≈ 0 (R_sym's dip subtracted), which in dijets is under jet1's peak.
- **The selection bias.** Jet1 > 130 GeV after quenching favours a surface jet1 and a long path
  for jet2, and "subleading" picks the more quenched jet; dijets are gluon-richer, the Z's recoil
  mostly a quark. The wake per jet differs, so the proxy's significance does not scale one-to-one
  to dijets.
- **p_T.** Most of the Z's are at 40–100 GeV, so the proxy runs at lower thresholds than CMS
  (§2, D2).

## 1. Model results to compare with

- **Yang & Wang 2508.04194** (CoLBT-hydro, 0–10 %): γ+jet with p_T^γ > 100 GeV, |η_γ| < 1.44, jets
  R = 0.4 > 50 GeV in |η| < 1.6, Δφ > 7π/8. Their **background-free** form: the hadrons' *lab*
  η distribution on the trigger side (Δφ_jet,h > π/2, or Δφ_γ,h < π/2 in their Fig. 3), for
  |η_jet| > 0.8 against |η_jet| < 0.8 with η_jet > 0 by reflection; the trigger's rapidity
  range is the same for both classes, so the bulk cancels in the difference. Dip near η_h ≈ 1 at
  1–2 GeV, small at 2–4 GeV, pp subtraction negligible. Their §III does dijets with hemisphere
  classes (η_jet1 η_jet2 > 0 against < 0). The proxy computes both forms (B3).
- **Yang & Wang 2501.03419**: the original dijet rapidity-asymmetry proposal (CoLBT-hydro).
- **CMS's Fig. 2** compares data with PYTHIA (flat), HYBRID (fully incoherent energy loss;
  overpredicts the 1–2 GeV depletion and predicts an enhancement near Δη ≈ 0 from jet2's wake in
  R_sym) and CoLBT-hydro (0–10 % only; overpredicts both). Both media models get the dip's mean
  position. These are for the dijet stage (Phase C).
- **Our own Z–hadron result** (`analysis/runs/zjet_c00/figures/ANALYSIS_c00.md` §6): the model's
  Z-side depletion at 1–2 GeV is about half as deep as CMS 2507.09307's and absent at 2–4 GeV.
  The proxy resolves the same wake by jet rapidity, so it shows whether that deficit depends on
  where the jet goes.

## 2. Statistics

Counted on c00 at **R = 0.3** (the existing map step; to be redone at R = 0.4, A5) with
`analysis/dijet/gap_count.py`
(Z: mass 70–110 GeV and the leptons of `zjet_jets.z_selection`, |y_Z| < 1; jet2:
the leading jet, |η| < 2, Δφ > 5π/6, per oversample). Effective events = N_eff of the Z's (window
and N_coll weights) × the class's weighted fraction:

| p_T^Z / jet2 (GeV) | N_eff Z's | with a jet2 | < 0.5 | 0.5–1.0 | 1.0–1.5 | 1.5–2.0 |
|---|---|---|---|---|---|---|
| > 40 / > 30 | 2,083 | 43 % | 314 | 271 | 170 | 104 |
| > 60 / > 30 | 1,270 | 61 % | 285 | 236 | 140 | 82 |
| > 100 / > 50 | 318 | 74 % | 95 | 73 | 42 | 21 |
| > 130 / > 50 (CMS's) | 125 | 81 % | 38 | 31 | 21 | 10 |

The classes split about 37 / 31 / 18 / 11 % of the events with a jet2, in Pb+Pb and pp alike;
pp (9,667 vacuum events) has a jet2 in 79 % at 60/30. c00 + c01 doubles the counts, ten campaigns
multiply them by 10.

Precision (rough, factor 2: scaled from the per-Z noise of the c00 Z–hadron Δy at 1–2 GeV in
`runs/zjet_c00/figures/mixing_summary_c00.txt`, × √2 for the unfolded near side): the error of R_asym − R_sym
(1.0–1.5 minus < 0.5) per 0.4-wide Δη bin at 60/30 is ±0.12 (c00) / ±0.04 (ten campaigns) on the
truth path, ±0.5 / ±0.16 experiment-like with all 40 oversamples, ±1.5 / ±0.5 for the strict
one-oversample emulation. CMS's dip is a few tenths per unit Δη; the model's may be about half.
So: **c00 builds and closes the chain; the truth path shows the signal with the full campaigns;
the experiment-like path stays near 1σ per bin.** B5 replaces these estimates with measured ones.

## 3. Phase A — code changes

### A0. Map outputs and pools that combine across campaigns

Today a new campaign means remapping all campaigns together, although the expensive part does not
depend on the set: clustering, embedding and random cones work file by file, and the random cones
and the pools' `k_one` are keyed by (seed, file name, local event). Only bookkeeping computed over
the whole reader is written into every file's output:

- map step (`zjet/zjet_jets_map.py:86–94`, `:140–144`): `event` = the event's index over all files,
  `file` = the file's index, `bg` = an index from `np.unique` over (file, `bg_unit`), and
  `weight_mb` = σ_k / N_k, with σ_k the files' Pythia cross sections weighted by their accepted
  events (`HadronFileReader.pthat_bin_sigma`) and N_k the stored events of window k, both over
  all files;
- pools (`zjet/mix.py:76–106`): the same weights, event ids and background ids are passed into
  `zjet_mixing.file_pool`; the per-background sums (`SW`, `T`, `R`, `zjet/zjet_mixing.py:230–239`)
  carry the weights; and the cache manifest is keyed by the whole source, so a new campaign
  rebuilds every pool.

The fix: every output self-contained, ids and weights at load time.

- **Map step:** keys instead of indices: the file name (unique across campaigns through the
  `TAG_PREFIX` naming), `local_event` (already stored) and `bg_unit`. A one-entry-per-file
  `file_info` tree: per p̂_T window the file's stored events (before any cut), σ_gen, σ_err and
  accepted events. `weight_mb` stays for older readers.
- **Loader** (`zjet_jets.load_jets`, also used by `zjet_jets_qa.py`): takes several directories
  or globs (or several `--run`s); event ids from (file name, local event), background ids from (file name, `bg_unit`),
  both ordered by file name as the reader orders files; σ_k = Σ n_acc σ_gen / Σ n_acc,
  N_k = Σ n_stored, `weight_mb` = σ_k / N_k: the rule of `HadronFileReader`, so the numbers of a
  combined run. The combination rule and the key-based ids know no trigger: they go into
  `jetcorr/weights.py` (C++ `common/AnaCommon.h`, next to `event_weights` and `background_ids`),
  so that Phase C's dijet map step and loader take them over.
- **Pools:** the same keys and `file_info` in every per-file pool; the per-background sums stored
  **per window and unweighted** (Σ S and the lab density over the background's events of window
  k, with their count). An event's weight factorizes as (σ_k / N_k) × N_coll with N_coll the same
  for all events of a background (verified on the test: one value per background; the pool
  writers raise otherwise), so `Pool.load` rebuilds `SW` (w), `T` (w²), `R`,
  `W`, `W2` exactly; the per-background arrays grow ×3 (six windows instead of the two weighted
  sums). One manifest per campaign directory.
- **C++ twins:** `zjets.cxx`, `zhadron_pools.cxx` and `read_root_pool` the same.
- **Layout:** one run per campaign (`runs/zjet_c00/data/jets_r04/`, `runs/zjet_c01/data/jets_r04/`,
  …; `runs/zjet_cNN/data/mix/`); a combination is a list of runs at load time with its own run
  for the figures (`runs/zjet_c00-01/figures/`), and a new campaign maps only itself.

### A1. The jet radius as an option (map steps, QA, final stage)

R is a constant today: `R_JET = 0.3` in `jetcorr/jetmap.py:10` and `common/JetMap.h:32` (the
shared jet finding of both map steps), and in `zjet/zjet_jets_qa.py:41`. The k_T R = 0.4 of the ρ
estimate (`jetcorr/jetmap.py:64`, `common/JetMap.h:36`) is a separate setting and stays.

- R a parameter of `Clusterer` (`jetcorr/jetmap.py`, `common/JetMap.h`, default 0.3), so the
  dijet map step of Phase C has it too; `--R` in both Z+jet map steps, **default 0.3** so that the
  existing outputs stay bit for bit. It sets the anti-k_T radius, the random cones' radius and
  their ρπR² term (`zjet/zjet_jets_map.py:204–216`, `zjets.cxx:263–266`), and the embedding's match
  radius `DR_MATCH` = R/2 (0.15 at 0.3).
- The map step writes R (and `--ptmin`, `--cones`, `--seed`) into a one-entry `run_info` tree;
  `zjet_jets.py` and `zjet_jets_qa.py` read it (the area normalization A/πR², the labels) and fall
  back to 0.3 for older files.
- `zjet_jets.py`'s CMS 1702.01060 comparison (the resolution `RES`, N = 8.08 GeV, `zjet/zjet_jets.py:57`, and the
  HEPData χ²) is for R = 0.3: at another R it is skipped with a note, not silently applied.
- Coverage at R = 0.4 holds: jets to |η| < 2 reach 2.4, the ghosts go to 2.6
  (`GhostedAreaSpec(2.6, …)`, `jetcorr/jetmap.py:66`) and the particles are stored to |η| < 2.6.
- The ghost area **stays 0.01** on the fixed grid (`jetcorr/jetmap.py:66`, `common/JetMap.h:38`;
  decided 2026-10-18, D8) and is not made an option; `run_info` records it. Why, from the test of
  0.05 on one c00 file (4 oversamples, R = 0.3, both settings run side by side): 0.05 runs
  ~15 % faster (45 s against 53 s), but the hard jets' areas spread 3× more (A/πR² of the jets
  above 30 GeV: 0.82 ± 0.19 against 1.02 ± 0.06), 73 % more jets pass 30 GeV (535 against 309,
  upward fluctuations and fakes), and the embedding's JER at 30–50 GeV goes from 0.24 to 0.38
  (20–30 GeV: 0.44 → 0.55). A toy of a perfect cone on the fixed grid gives the same: the area's
  jitter is ±0.047 at 0.05 against ±0.013 at 0.01, i.e. ±13 GeV in ρA against ±3.6 GeV at
  ⟨ρ⟩ ≈ 273 GeV, as large as σ_RC itself (13.1 GeV at R = 0.3). The random cones do not use
  ghosts and would not show it; the embedding does.

### A2. The Z cone (jet1's p_T variant)

Per oversample, the Σp_T of the event's particles within R of (y_Z, φ_Z), leptons excluded, ρ of
the strip at y_Z, and δp_T = Σp_T − ρπR²; also in the no-wake twin. Stored as `zcone.{sum, rho,
dpt}` in the `jets` tree. Then jet1's p_T can be the bare p_T^Z (baseline) or p_T^Z + δp_T: the
reconstructed jet1 of a detector, including the real sym/asym difference that for small gaps
jet2's wake sits inside jet1's cone. Cheaper than inserting the Z into the event and reclustering,
and the same to first order; the random cones in `bulk_bg` stay as the cross-check.

### A3. Per-event contents in the pools

The pools (`zjet_mixing.file_pool`, `zjet/zjet_mixing.py:145`; `zhadron_pools.cxx`) contain no jets
and no R. Per
event they hold `S_full` (40-sample mean) and `S_one` (one oversample) in the Z frame
(p_T × Δy = η − y_Z, 0.4 wide × Δφ, π/10), the one oversample's tracks, and **per background
only** the Z-correlated yield. To split the truth by gap class, add per event:

- `S_zc` (wake + fragments: `SJ − SB + SF`, already computed at `:216`) and `S_frag` (`SF`), in
  the Z frame. With `S_full` they give every content of B2.
- `L_full`, `L_zc`, `L_frag`: the lab-frame η distribution (the existing `ETA` grid, 0.1) on the
  Z side and the away side (|Δφ_Z| < π/2, > π/2), for the Yang–Wang form (B3).

About 13 kB more per event (~100 MB for c00, ~1 GB for ten campaigns). `Pool`, `_pack`,
`read_root_pool` and the C++ writer get the same layout; pools without the new arrays still load,
so the Z–hadron analysis is unaffected.

**As implemented (2026-10-18, pool version 4; D5):** the new per-event contents are on a 0.1
grid in Δy per side of the Z (side 0 the Z side |Δφ| < π/2, the DPHI bins of `ZSIDE`), not on
the 0.4 × π/10 Z frame: `Y_full`, `Y_one`, `Y_zc`, `Y_frag` (`zjet_mixing.side_hist`, shape
(3, 2, 96)), and `L_full`, `L_zc`, `L_frag` (`lab_side_hist`, (3, 2, 48)); the old `S_full`,
`S_one` stay. Compressed, the v4 pools are about as large as v3's per background (the mini
campaign: ~210 kB per background and file). The pools stay in `zjet/` here, tied to the Z frame;
Phase C's pools in jet1's frame are the point to move them into `jetcorr/` with the frame as an
input.

### A4. Regression and the C++ twins

- Map step at R = 0.3 and ghost area 0.01: identical outputs to `runs/zjet_c00/data/jets/` on a few files.
- A0: on c00, the load-time event ids, background ids and `weight_mb` equal today's columns
  exactly (map step and pools); the first files of c00 and c01 mapped separately and loaded
  together give bit for bit what the same files mapped in one run give, through the final stages
  (`zjet_jets.py`, `mix.py`).
- `zjet/check_cxx.py jets` at R = 0.3 and R = 0.4 (Python against `zjets`), including `zcone`.
- **One FastJet for both routes.** On rhicML the `fastjet-config` on `PATH` is `/usr/local`'s
  FastJet 3.4.2, while the Python map step uses the pip wheel's 3.5.2. `make FASTJET_PY=<the
  wheel's package directory>` (Makefile, 2026-10-18) builds the C++ map steps against the
  wheel, the same FastJet and the fjcontrib it bundles (`libfastjetcontribfragile.so`, all
  contribs; the wheel's own `fastjet-config` carries its build paths, so the Makefile sets the
  flags). Checked on one c00 file (`--samples 2`, converted with `run_h5toROOT.py`):
  `check_cxx.py jets` 0 differences in every tree; C++ 19.5 s against Python 29 s.
- Pools: the existing arrays identical to `runs/zjet_c00/data/mix/`'s; `check_cxx.py pools` covers the new
  arrays; `S_full − S_zc + S_frag` and `S_zc − S_frag` are checked against the per-background
  sums.

### A5. Reruns

- Map step at R = 0.4 on c00 (Pb+Pb and pp) → `runs/zjet_c00/data/jets_r04/`; then c01 →
  `runs/zjet_c01/data/jets_r04/`, each once (A0). The last c00 map step took 6,335 s wall with 16
  Python processes (24 CPU-hours, `runs/zjet_c00/data/logs/jets_c00_nw_map_done.log`); the box has 48 cores, so more
  `--files` slices should bring it to ~40 min if memory allows (**verify**).
- The pools with the new arrays, c00 and c00+c01 (time **verify**; the mixing alone took 514 s on
  c00).
- `dijet/gap_count.py` again at R = 0.4 (the §2 table).

**Remark: the reruns can take the C++ route** (`zjets`, `zhadron_pools` on the ROOT
conversion), since its outputs are the Python map step's on the HDF5 files. Checked
2026-10-19 on one 48-event campaign-format file, all 40 oversamples, R = 0.4: `check_cxx.py
jets` 0 differences (every integer column, the jets and their order, `file_info`/`run_info`
equal; floats within 3·10⁻¹³), and `check_cxx.py pools` 0 differences for the v4 pools; the
conversion is bit-exact (`prod_AuAu_0_10_jet/root_export/check_h5_vs_root.py`). The final stages read either
(`load_jets`, `Pool.load`: `*_jets.root` of both, `*_pool.root` or `.npz`). What it gains,
measured on the GB10 with one process alone on that file:

| | time | memory (max RSS) |
|---|---|---|
| C++ `zjets` (ROOT) | 180 s | 0.99 GB |
| Python `zjet_jets_map.py` (HDF5) | 203 s | 2.56 GB |

C++ is only ~1.1× faster per process (on rhicML at R = 0.3 with 2 samples 19.5 s against
29 s), but needs 2.6× less memory, which sets how many processes run side by side on the box.
Per campaign (100 files of 96 events) about 10 CPU-hours in C++ (6 min per file on the GB10),
so ~15–20 min with 30–40 processes if the box's cores are as fast (**verify** there). The cost
is one conversion per campaign, reused by every later map step (any R, the pools, `zqa`): 2
files of 48 events in 9 s with one process, the ROOT files ~0.8× the HDF5's; c00 may already
have one from `ANA_Zjet.md` route B (B1). One route per campaign keeps a campaign's outputs
bit-reproducible; mixing routes across campaigns is fine (agreement to 10⁻¹³).

Outline (per campaign `NN`, the hadron files `PB="/data/zjet_cNN/gpu*/*_hadrons_jet_frag.h5"`;
the commands in `analysis/dijet/ANA_Deta.md` §1–2):

1. Build once against the Python wheel's FastJet: `make clean && make FASTJET_PY=…` in
   `analysis_root/zjet` (A4: one FastJet for both routes).
2. Convert once: `run_h5toROOT.py "$PB" -j 8 --out-dir runs/zjet_cNN/data/root --campaign-file
   …/zjet_cNN_campaign.root`, then `prod_AuAu_0_10_jet/root_export/check_h5_vs_root.py "$PB"
   runs/zjet_cNN/data/root --campaign …` → 0 differences; the pp alike (`runs/zjet_pp/data/root`).
3. Map, one process per file, as many in parallel as memory allows (~1 GB each):
   `seq 0 99 | xargs -P 40 -I{} sh -c 'zjets runs/zjet_cNN/data/root runs/zjet_cNN/data/jets_r04
   --R 0.4 --files {}:$(({} + 1)) > runs/zjet_cNN/data/logs/zjets_r04_{}.log 2>&1'`; the pp in
   one process (seconds).
4. Pools (v4) in C++: `zhadron_pools runs/zjet_cNN/data/root runs/zjet_cNN/data/mix/pbpb`
   (`--files` slices in parallel likewise); the pp into `runs/zjet_pp/data/mix/vacuum`.
5. Check (A4): the jets on one file, `zjet_jets_map.py "$PB" … --R 0.4 --samples 2 --files 0:1`
   against `zjets … --R 0.4 --samples 2 --files 0:1` with `check_cxx.py jets`; the pools of the
   whole campaign (both take their stored weights from the whole directory), `mix.py
   --pools-only` (minutes) against `zhadron_pools` with `check_cxx.py pools`.

## 4. Phase B — the Z-proxy analysis

New: a trigger-independent library `analysis/jetcorr/deta_asym.py` (classes, reflection, contents,
observables; the dijet stage reuses it) and the final stage `analysis/dijet/deta.py`
(pools + R = 0.4 jet trees, Pb+Pb and pp → figures and summary). `deta.py --trigger z` is the Z
proxy on the Z+jet runs (the Z from the pools and the `jet_events` tree); Phase C adds
`--trigger jet1` on the dijet runs, the rest unchanged. Seconds to minutes per run.

### B1. Selection and classes

- Z: the pools' definition (2507.09307: 40–350 GeV, |y_Z| < 2.4, 60–120 GeV, leptons > 20 GeV in
  |η| < 2.4), then |y_Z| < 1 and p_T^Z (or p_T^Z + δp_T) above the threshold (D2, D3).
- jet2: the leading R = 0.4 jet in |η| < 2 above its threshold, Δφ > 5π/6 (variant: the leading
  jet among those with Δφ > 5π/6). Joined to the pools by the event key.
- Per event the fraction of its oversamples in each class (as in §2) weights the 40-sample
  contents; the strict mode uses the jets of the event's own oversample `k_one`. Reflection by the
  event's majority. All contents use the full event's classes, so their differences are hadron
  content only (the no-wake twin's own jets are a variant).

### B2. The contents (the model's ladder)

| | content | per event | role |
|---|---|---|---|
| T | truth: wake + fragments | `Y_zc` | the reference; the bulk cancels exactly, no sideband needed |
| W | wake only | `Y_zc − Y_frag` | what is wake |
| F | full event, 40 samples | `Y_full` | experiment-like, with acceptance and sideband |
| O | one oversample | `Y_one` | strict emulation, the experiment's errors |
| N | no-wake twin (`bulk_bg` + the same fragments) | `Y_full − Y_zc + Y_frag` | must give a flat R_asym − R_sym |
| B | bulk only | `Y_full − Y_zc` | what the bulk leaks into the difference |
| pp | fragments (vacuum) | `Y_full` | the baseline |

(The `Y_*` of A3 as implemented; the lab-η `L_*` alike for the Yang–Wang form.)

### B3. Observables

- **CMS form:** per class, near side (|Δφ| < π/2) per trigger per unit Δη, reflected; the pair
  acceptance from the class's y_Z distribution and the sharp |η_ch| < 2.4 (the pooled-density
  kernel of `zjet_mixing`, normalized as CMS's mixed events); the 1.5 < Δη < 2.5 level subtracted
  per class (F, O, N, B; for T both with and without, to measure what the sideband removes);
  R_asym − R_sym for the three large-gap classes; PbPb − pp; the integral over
  −2.4 < Δη < −0.6 (Fig. 3's quantity). Track p_T 1–2 and 2–4 GeV (4–10 as a check).
- **Yang–Wang form:** from `L_*`, the lab η distribution on the Z side for |η_jet| > 0.8 against
  < 0.8 (η_jet > 0 by reflection), and for hemisphere classes; the difference, with and without
  the bulk (F against T: background-free means equal) and pp.
- Errors: jackknife over blocks of backgrounds (as `mix.py`, 100 blocks); the strict mode's spread
  for experiment-like errors.

### B4. Closure tests

1. **Class sum:** the classes summed (unreflected) give the Z–hadron Z-side Δy of the same events
   to rounding (bookkeeping).
2. **N flat:** the no-wake twin's R_asym − R_sym is zero within errors; a residual is a method or
   fragment artifact.
3. **B flat:** the bulk alone, after acceptance and sideband, is zero; the residual is the
   dN/dη-shape leak that Yang–Wang's form avoids.
4. **F = T:** the experiment-like result closes on the truth (the analogue of Z–hadron step 2
   against step 1).
5. **Sideband loss:** T's yield in 1.5 < Δη < 2.5 per class, mostly for R_sym, where the wake
   sits near Δη ≈ 0 and is broad.
6. **pp flat**, as in CMS.
7. **Variants:** jet1 bare or with the Z cone; jet2 leading or leading back-to-back; R 0.3 against
   0.4; thresholds (D2).

### B5. Results and comparisons

- **R1.** The Z-proxy R_asym − R_sym (Pb+Pb, pp, Pb+Pb − pp) at 1–2 and 2–4 GeV for the three
  large-gap classes, and the integrated dip per class: the dip moving to more negative Δη with
  the gap, as in CMS's Fig. 2, and the bump at Δη ≈ 0.
- **R2.** The Yang–Wang form against their CoLBT-hydro γ-jet (2508.04194 Figs. 2–3), with the
  closest cuts statistics allow: p_T^Z > 100 GeV (about 300 effective Z's per campaign) and a
  0–10 %-like subset by N_coll (a third of the events). Differences to state: Z against γ (both
  colourless), 0–30 % against 0–10 %, their jet |η| < 1.6.
- **R3.** Against our Z–hadron result: whether the model's deficit against CMS depends on the jet
  rapidity.
- **R4.** The measured noise per event for T, F, O → the precision per campaign and the dijet
  production's size (replacing §2's estimate).
- If the hard-momentum test production of `PLAN_hard_momentum_subtraction.md` exists, run the
  proxy on it as well: a string depletion tied to y_jet would show in B (bulk), the wake in T, so
  the proxy separates the two candidates for the Z–hadron deficit.

### B6. Outputs and documents

`runs/<run>/figures/deta_{asym,closure,yw,integrated}_<tag>.png` with their `.npz`/`.root`
(`jetcorr/figdata.py`), `deta_summary_<tag>.txt`; the step-by-step `analysis/dijet/ANA_Deta.md`
in the form of `zjet/ANA_Zjet.md`; the result note `runs/zjet_c00/figures/ANALYSIS_c00_deta_proxy.md`
(`runs/zjet_c00-01/…` for the combination; git-ignored with the figures).

## 5. Phase C — the dijet production (outline, after B)

1. **Generator:** PythiaGun hard QCD (the jet productions' settings, **verify**) with p̂_T
   windows chosen, as `zjet/test/window_layout.py` did for the Z, so that most events have jet1 > 130 GeV at
   R = 0.4 after quenching; the same Pb+Pb 0–30 % chain, the correlated `bulk_bg` twin kept (the
   truth path needs ~20× fewer events than the experiment-like one), all particles to
   |η| < 2.6.
2. **Map step without a Z:** `dijet/dijet_jets_map.py` and a C++ twin in `analysis_root/dijet/`,
   on `jetcorr/jetmap.py` / `common/JetMap.h` (R from A1) and the keys, weights and `file_info`
   of A0: jet1 and jet2 from the clustering, all particles but neutrinos (the Z map step's lepton
   removal, `keep_mask`, stays in `zjet/`), the embedding and random cones as there, the
   `jet_events` tree without Z columns.
3. **Pools in jet1's frame:** `zjet_mixing`'s pools moved into `jetcorr/` with the frame as an
   input: from the jet trees instead of `info.z`, per oversample, since jet1's axis moves with the
   sample in Pb+Pb; the pool map then reads the jet trees.
4. **Comparison with data:** CMS-HIN-25-012's HEPData record (the paper's ref. [31]; a reader in
   `dijet/` as `zjet/cms2507.py` is for 2507.09307), the model bands of CMS's Fig. 2, and
   2501.03419.
5. **Size:** from R4. Today's rough estimate for ±0.03 per 0.4 bin in the 1.0–1.5 class: ~10⁴
   selected dijets on the truth path, ~2 × 10⁵ experiment-like.
6. **Settings:** the same physics settings as the final Z+jet campaigns, in particular the
   decision of `PLAN_hard_momentum_subtraction.md`.

## 6. Decisions

| | question | proposal |
|---|---|---|
| D1 | Z definition | **decided 2026-10-18:** the pools' (2507.09307), so pools and jets select the same Z's; then \|y_Z\| < 1 |
| D2 | thresholds | **decided 2026-10-18:** p_T^Z > 60, jet2 > 30 GeV as the baseline, **to revisit against 60/40 once A5 gives the fake fraction at R = 0.4** (`deta.py` prints the fake jet2 per class: the cuts on the Z's own `bulk_bg`); 40/30, 60/40, 80/50, 100/50 and CMS's 130/50 as variants. **Added: jet2 < jet1 (x_jZ < 1) in the baseline** (CMS's jet1 is the leading jet), no x_jZ cut as the variant |
| D3 | jet1's p_T | **decided 2026-10-18:** bare p_T^Z; p_T^Z + δp_T (A2) as the variant |
| D4 | jet2's subtraction | **decided 2026-10-18:** ρA (in hand); constituent subtraction (CMS's: Berta et al., arXiv:1403.3108; settings not in the paper) only if the R = 0.4 embedding shows a JES bias that moves the class fractions. Then in C++ first: `ConstituentSubtractor` from the wheel's fjcontrib (`make FASTJET_PY=…`, A4; event-wide subtraction tested in a toy, 2026-10-18). The Python `fastjet` has no contrib binding (only SoftDrop, Lund and EnergyCorrelator through its awkward interface), so Python would need a small pybind11 module on the same library, or the subtraction stays C++-only. The campaign image has no fjcontrib, FastJet headers, ROOT or Python `fastjet`: the analyses run on the host |
| D5 | Δη binning | **decided 2026-10-18: a 0.1 grid in Δy, near and away side only**, for the new per-event contents (`Y_*`, and `L_*` in lab η). The pools' 0.4 came from CMS 2507.09307's Z–hadron tables (pp 0.4, Pb+Pb 0.8). CMS HIN-25-012's HEPData record (ins3123406, public: Tables 1–4) bins Δη in **0.2 with edges at odd tenths** (−2.5, −2.3, …, 2.1; centred on 0), which the 0.4 grid cannot form, nor the sideband 1.5–2.5 or the signal edge −0.6; the 0.1 grid rebins exactly to all of them and to 0.4/0.8. CMS's Fig. 3 values are the **plain sums** of Fig. 2's over the 10 bins centred −2.4 … −0.6 (edges −2.5 … −0.5; checked on all four 0–30 % entries, e.g. −1.29): either Fig. 2 is per 0.2 bin or Fig. 3 omits the width (a factor 5), open; it matters only for magnitudes against CMS and the models |
| D6 | classes | **decided 2026-10-18:** CMS's four; Yang–Wang's \|η_jet\| ≷ 0.8 (their jets \|η\| < 1.6) and hemispheres |
| D7 | centrality | **decided 2026-10-18:** 0–30 % (CMS's central bin); a 0–10 %-like subset N_coll > 1210 for the CoLBT comparisons (the model's 10 % boundary from the 200k min-bias 3dMCGlauber events of `zjet/test/centrality_check.py`, b < 4.99 fm; 34 % of the 0–30 % backgrounds, 51 % of the N_coll-weighted Z yield) |
| D8 | ghost area | **decided 2026-10-18: 0.01.** 0.05 was tested (A1): ~15 % faster, but JER 0.24 → 0.38 at 30–50 GeV and 73 % more jets above 30 GeV; more processes give the speed instead |

## 7. Order of work and compute

| step | what | compute |
|---|---|---|
| A0 | keys, `file_info`, load-time ids and weights (`jetcorr/weights.py`, `common/AnaCommon.h`); per-window background sums in the pools; one run per campaign | – |
| A1–A2 | R in `Clusterer`, `--R`, `run_info`, `zcone` in both map steps; QA and final stage read R | – |
| A3 | per-event `S_zc`, `S_frag`, `L_*` in both pool writers | – |
| A4 | regressions, `check_cxx.py jets` / `pools` | minutes |
| A5 | map step at R = 0.4 (c00, then c01), pools rebuilt, counts again | C++ route (A5's remark): ~10 CPU-hours per campaign, ~15–20 min with 30–40 processes of ~1 GB (**verify** on the box); pools seconds per file; one conversion per campaign |
| B1–B4 | `jetcorr/deta_asym.py`, `dijet/deta.py --trigger z`, the closure tests on c00 | seconds to minutes per run |
| B5–B6 | results on c00 + c01, comparisons, documents; again with each new campaign | minutes |
| C | dijet production: windows, `dijet/` map step, pools in jet1's frame (`jetcorr/`), HEPData | after B5's R4 |

## 8. Summary

The Z+jet campaigns carry the η-separated analysis with the Z as jet1: the same hadron
observable, CMS's cuts on jet2 at R = 0.4, and the model's truth, no-wake twin and bulk next to
the experiment-like result for closure. Three code changes: map outputs and pools that combine
across campaigns (ids and weights at load time, so each campaign is mapped once), the jet radius
as an option in the shared jet finding and the map steps (R is used only there, the QA and the
final stage's labels; the ghost area stays 0.01), and the truth yield per event in the pools (which
have no R). The shared parts go into `jetcorr/` and `analysis_root/common/`, the Z-specific ones
stay in `zjet/`, and the analysis itself is `dijet/`, so the dijet stage reuses all of it. c00 +
c01 build and close the chain;
the full campaigns give a result to set against Yang & Wang's CoLBT-hydro γ-jet, and the noise
measured on the way sizes the dijet production.
