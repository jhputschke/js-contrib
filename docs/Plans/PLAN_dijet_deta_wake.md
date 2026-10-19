# PLAN: CMS's η-separated dijet wake analysis (HIN-25-012), first on the Z+jet campaigns with the Z as jet1

2026-10-18. Why: CMS observed the diffusion wake with dijets separated in pseudorapidity
([arXiv:2602.19431](https://arxiv.org/abs/2602.19431), CMS-HIN-25-012, > 5σ at 1–2 GeV in
0–30 %). A dijet production with the full chain (a hydro run with the wake per event) is
expensive, so the analysis is first built, tested and closed on the Z+jet campaigns that already
exist, with the Z standing in for jet1. That is more than a prototype: Yang & Wang
([arXiv:2508.04194](https://arxiv.org/abs/2508.04194)) proposed the same rapidity asymmetry for
γ/Z+jet and computed it with CoLBT-hydro, so the Z version is a result in its own right. The Z
test also measures the noise per event that sets the size of the later dijet production.

Written against js-contrib `zjet` 72cb15e. Data in hand: campaigns `zjet_c00` and `zjet_c01`
(19,200 Pb+Pb 0–30 % events on 800 hydro backgrounds, `/data/zjet_c0*`), the c00 jet map step at
R = 0.3 (`analysis/zjet_test/data/jets_c00/`), the c00 and c00+c01 Z–hadron pools, and the pp
reference. Items marked **verify** were inferred from reading the code and have not been run.

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
- **Our own Z–hadron result** (`analysis/zjet_test/figures/ANALYSIS_c00.md` §6): the model's
  Z-side depletion at 1–2 GeV is about half as deep as CMS 2507.09307's and absent at 2–4 GeV.
  The proxy resolves the same wake by jet rapidity, so it shows whether that deficit depends on
  where the jet goes.

## 2. Statistics

Counted on c00 at **R = 0.3** (the existing map step; to be redone at R = 0.4, A5) with
`gap_count.py` (2026-10-18, so far outside the repo; it goes to `analysis/zjet_test/` with A5)
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
`figures/mixing_summary_c00.txt`, × √2 for the unfolded near side): the error of R_asym − R_sym
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

- map step (`zjet_jets_map.py:183–191`, `:237–241`): `event` = the event's index over all files,
  `file` = the file's index, `bg` = an index from `np.unique` over (file, `bg_unit`), and
  `weight_mb` = σ_k / N_k, with σ_k the files' Pythia cross sections weighted by their accepted
  events (`HadronFileReader.pthat_bin_sigma`) and N_k the stored events of window k, both over
  all files;
- pools (`mix.py:76–106`): the same weights, event ids and background ids are passed into
  `zjet_mixing.file_pool`; the per-background sums (`SW`, `T`, `R`, `zjet_mixing.py:230–239`)
  carry the weights; and the cache manifest is keyed by the whole source, so a new campaign
  rebuilds every pool.

The fix: every output self-contained, ids and weights at load time.

- **Map step:** keys instead of indices: the file name (unique across campaigns through the
  `TAG_PREFIX` naming), `local_event` (already stored) and `bg_unit`. A one-entry-per-file
  `file_info` tree: per p̂_T window the file's stored events (before any cut), σ_gen, σ_err and
  accepted events. `weight_mb` stays for older readers.
- **Loader** (`zjet_jets.load_jets`, also used by `zjet_jets_qa.py`): takes several directories
  or globs; event ids from (file name, local event), background ids from (file name, `bg_unit`),
  both ordered by file name as the reader orders files; σ_k = Σ n_acc σ_gen / Σ n_acc,
  N_k = Σ n_stored, `weight_mb` = σ_k / N_k: the rule of `HadronFileReader`, so the numbers of a
  combined run.
- **Pools:** the same keys and `file_info` in every per-file pool; the per-background sums stored
  **per window and unweighted** (Σ S and the lab density over the background's events of window
  k, with their count). An event's weight factorizes as (σ_k / N_k) × N_coll with N_coll the same
  for all events of a background (**verify**), so `Pool.load` rebuilds `SW` (w), `T` (w²), `R`,
  `W`, `W2` exactly; the per-background arrays grow ×3 (six windows instead of the two weighted
  sums). One manifest per campaign directory.
- **C++ twins:** `zjets.cxx`, `zhadron_pools.cxx` and `read_root_pool` the same.
- **Layout:** one directory per campaign (`data/jets_r04/c00/`, `c01/`, …; `data/mix/c00/`, …);
  a combination is a list of directories at load time, and a new campaign maps only itself.

### A1. The jet radius as an option (map steps, QA, final stage)

R is a constant today: `R_JET = 0.3` in `zjet_jets_map.py:42`, `zjets.cxx:63`,
`zjet_jets_qa.py:39`. The k_T R = 0.4 of the ρ estimate (`zjet_jets_map.py:119`, `zjets.cxx:80`)
is a separate setting and stays.

- `--R` in both map steps, **default 0.3** so that the existing outputs stay bit for bit. It sets
  the anti-k_T radius, the random cones' radius and their ρπR² term (`zjet_jets_map.py:301–313`,
  `zjets.cxx:371–374`), and the embedding's match radius `DR_MATCH` = R/2 (0.15 at 0.3).
- The map step writes R (and `--ptmin`, `--cones`, `--seed`) into a one-entry `run_info` tree;
  `zjet_jets.py` and `zjet_jets_qa.py` read it (the area normalization A/πR², the labels) and fall
  back to 0.3 for older files.
- `zjet_jets.py`'s CMS 1702.01060 comparison (the resolution `RES`, N = 8.08 GeV, `:56`, and the
  HEPData χ²) is for R = 0.3: at another R it is skipped with a note, not silently applied.
- Coverage at R = 0.4 holds: jets to |η| < 2 reach 2.4, the ghosts go to 2.6
  (`GhostedAreaSpec(2.6, …)`, `:121`) and the particles are stored to |η| < 2.6.
- The ghost area **stays 0.01** on the fixed grid (`zjet_jets_map.py:121`, `zjets.cxx:82`;
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

The pools (`zjet_mixing.file_pool`, `:145`; `zhadron_pools.cxx`) contain no jets and no R. Per
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

### A4. Regression and the C++ twins

- Map step at R = 0.3 and ghost area 0.01: identical outputs to `data/jets_c00` on a few files.
- A0: on c00, the load-time event ids, background ids and `weight_mb` equal today's columns
  exactly (map step and pools); the first files of c00 and c01 mapped separately and loaded
  together give bit for bit what the same files mapped in one run give, through the final stages
  (`zjet_jets.py`, `mix.py`).
- `check_cxx.py jets` at R = 0.3 and R = 0.4 (Python against `zjets`), including `zcone`.
- **One FastJet for both routes.** On rhicML the `fastjet-config` on `PATH` is `/usr/local`'s
  FastJet 3.4.2, while the Python map step uses the pip wheel's 3.5.2. `make FASTJET_PY=<the
  wheel's package directory>` (Makefile, 2026-10-18) builds the C++ map steps against the
  wheel, the same FastJet and the fjcontrib it bundles (`libfastjetcontribfragile.so`, all
  contribs; the wheel's own `fastjet-config` carries its build paths, so the Makefile sets the
  flags). Checked on one c00 file (`--samples 2`, converted with `run_h5toROOT.py`):
  `check_cxx.py jets` 0 differences in every tree; C++ 19.5 s against Python 29 s.
- Pools: the existing arrays identical to `data/mix_c00`'s; `check_cxx.py pools` covers the new
  arrays; `S_full − S_zc + S_frag` and `S_zc − S_frag` are checked against the per-background
  sums.

### A5. Reruns

- Map step at R = 0.4 on c00 (Pb+Pb and pp) → `data/jets_r04/c00/`; then c01 →
  `data/jets_r04/c01/`, each once (A0). The last c00 map step took 6,335 s wall with 16 Python
  processes (24 CPU-hours, `data/logs/jets_c00_nw_map_done.log`); the box has 48 cores, so more
  `--files` slices should bring it to ~40 min if memory allows (**verify**).
- The pools with the new arrays, c00 and c00+c01 (time **verify**; the mixing alone took 514 s on
  c00).
- `gap_count.py` again at R = 0.4 (the §2 table).

## 4. Phase B — the Z-proxy analysis

New: a trigger-independent library `analysis/deta_asym.py` (classes, reflection, contents,
observables; the dijet stage reuses it) and the final stage `analysis/zjet_test/zjet_deta.py`
(pools + R = 0.4 jet trees, Pb+Pb and pp → figures and summary). Seconds to minutes per run.

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
| T | truth: wake + fragments | `S_zc` | the reference; the bulk cancels exactly, no sideband needed |
| W | wake only | `S_zc − S_frag` | what is wake |
| F | full event, 40 samples | `S_full` | experiment-like, with acceptance and sideband |
| O | one oversample | `S_one` | strict emulation, the experiment's errors |
| N | no-wake twin (`bulk_bg` + the same fragments) | `S_full − S_zc + S_frag` | must give a flat R_asym − R_sym |
| B | bulk only | `S_full − S_zc` | what the bulk leaks into the difference |
| pp | fragments (vacuum) | `S_full` | the baseline |

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

`figures/zjet_deta_{asym,closure,yw,integrated}_<tag>.png` with their `.npz`/`.root`
(`figdata.py`), `zjet_deta_summary_<tag>.txt`; the step-by-step `analysis/ANA_ZjetDeta.md` in
the form of `ANA_Zjet.md`; the result note `figures/ANALYSIS_c00_deta_proxy.md` (git-ignored with
the figures).

## 5. Phase C — the dijet production (outline, after B)

1. **Generator:** PythiaGun hard QCD (the jet productions' settings, **verify**) with p̂_T
   windows chosen, as `window_layout.py` did for the Z, so that most events have jet1 > 130 GeV at
   R = 0.4 after quenching; the same Pb+Pb 0–30 % chain, the correlated `bulk_bg` twin kept (the
   truth path needs ~20× fewer events than the experiment-like one), all particles to
   |η| < 2.6.
2. **Map step without a Z:** jet1 and jet2 from the clustering (the lepton removal already skips
   events without a Z, `keep_mask`; the `jet_events` tree without Z columns, **verify**).
3. **Pools in jet1's frame:** the frame from the jet trees instead of `info.z`, per oversample,
   since jet1's axis moves with the sample in Pb+Pb; the pool map then reads the jet trees.
4. **Comparison with data:** CMS-HIN-25-012's HEPData record (the paper's ref. [31]; a reader as
   for `hepdata_zjet`), the model bands of CMS's Fig. 2, and 2501.03419.
5. **Size:** from R4. Today's rough estimate for ±0.03 per 0.4 bin in the 1.0–1.5 class: ~10⁴
   selected dijets on the truth path, ~2 × 10⁵ experiment-like.
6. **Settings:** the same physics settings as the final Z+jet campaigns, in particular the
   decision of `PLAN_hard_momentum_subtraction.md`.

## 6. Decisions

| | question | proposal |
|---|---|---|
| D1 | Z definition | the pools' (2507.09307), so pools and jets select the same Z's |
| D2 | thresholds | p_T^Z > 60, jet2 > 30 GeV as the baseline; 40/30, 80/50, 100/50 and CMS's 130/50 as variants |
| D3 | jet1's p_T | bare p_T^Z; p_T^Z + δp_T (A2) as the variant |
| D4 | jet2's subtraction | ρA (in hand); constituent subtraction (CMS's: Berta et al., arXiv:1403.3108; settings not in the paper) only if the R = 0.4 embedding shows a JES bias that moves the class fractions. Then in C++ first: `ConstituentSubtractor` from the wheel's fjcontrib (`make FASTJET_PY=…`, A4; event-wide subtraction tested in a toy, 2026-10-18). The Python `fastjet` has no contrib binding (only SoftDrop, Lund and EnergyCorrelator through its awkward interface), so Python would need a small pybind11 module on the same library, or the subtraction stays C++-only. The campaign image has no fjcontrib, FastJet headers, ROOT or Python `fastjet`: the analyses run on the host |
| D5 | Δη binning | the pools' 0.4; check CMS's binning in its HEPData record, and add a finer per-event grid only if needed |
| D6 | classes | CMS's four; Yang–Wang's |η_jet| ≷ 0.8 and hemispheres |
| D7 | centrality | 0–30 % (CMS's central bin); a 0–10 %-like N_coll subset for the CoLBT comparisons |
| D8 | ghost area | **decided 2026-10-18: 0.01.** 0.05 was tested (A1): ~15 % faster, but JER 0.24 → 0.38 at 30–50 GeV and 73 % more jets above 30 GeV; more processes give the speed instead |

## 7. Order of work and compute

| step | what | compute |
|---|---|---|
| A0 | keys, `file_info`, load-time ids and weights; per-window background sums in the pools; one directory per campaign | – |
| A1–A2 | `--R`, `run_info`, `zcone` in both map steps; QA and final stage read R | – |
| A3 | per-event `S_zc`, `S_frag`, `L_*` in both pool writers | – |
| A4 | regressions, `check_cxx.py jets` / `pools` | minutes |
| A5 | map step at R = 0.4 (c00, then c01), pools rebuilt, counts again | ~1.8 h per campaign for the map step; pools **verify** |
| B1–B4 | `deta_asym.py`, `zjet_deta.py`, the closure tests on c00 | seconds to minutes per run |
| B5–B6 | results on c00 + c01, comparisons, documents; again with each new campaign | minutes |
| C | dijet production: windows, map step without Z, pools in jet1's frame, HEPData | after B5's R4 |

## 8. Summary

The Z+jet campaigns carry the η-separated analysis with the Z as jet1: the same hadron
observable, CMS's cuts on jet2 at R = 0.4, and the model's truth, no-wake twin and bulk next to
the experiment-like result for closure. Three code changes: map outputs and pools that combine
across campaigns (ids and weights at load time, so each campaign is mapped once), the jet radius
as an option in the map steps (R is used only there, the QA and the final stage's labels; the
ghost area stays 0.01), and the truth yield per event in the pools (which have no R). c00 + c01 build and close the chain;
the full campaigns give a result to set against Yang & Wang's CoLBT-hydro γ-jet, and the noise
measured on the way sizes the dijet production.
