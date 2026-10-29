# Z–hadron event mixing: S, B and the four modes

What `mix.py` (with `zjet_mixing.py`) computes as S − B in each of its modes, written out in one
notation: what is averaged over an event's oversamples, which events are partners, which are
excluded, how the events are weighted. Then the pp result (the "factorized" one) and its relation
to the Pb+Pb modes, and `strict` and `cms10` against CMS 2507.09307's procedure. The commands, the
ladder of steps and the checks are in [ANA_Zhadron.md](ANA_Zhadron.md) and the
[README](README.md#event-mixing-mixpy); this document has the definitions only, no results.

## 1. What one event is

```
hydro background b   (Pb+Pb: one 3dMCGlauber + MUSIC_1 fluid; in a campaign 24 Z events share
 │                    it, 4 per p̂T window; those passing the Z selection enter)
 └─ Z event i        (one PythiaGun hard process: the Z, its partons, Matter + LBT, its
     │                droplets -> its own MUSIC_2 fluid)
     └─ oversample k = 0 … K−1   (K = 40: one particlization and one hadronization of event i)
```

| tag | sample k of event i, Pb+Pb | pp |
|---|---|---|
| `bulk_jet` | **J_i^k**: iSS sample k of event i's MUSIC_2 surface (the fluid with the wake) | – |
| `bulk_bg` | **G_b^k**: iSS sample k of background b's MUSIC_1 surface, with the same random numbers as J_i^k (correlated sampling), so J_i^k − G_b^k is the wake, sample by sample | – |
| `jet_frag` | **F_i^k**: Colorless hadronization (k mod n_frag) of event i's final partons | **F_i^k**: hadronization k of the partons Matter showered in vacuum (n_frag = 40; the GB10 stand-in 10) |

The oversamples of one event share the Z, its partons and (Pb+Pb) its fluid; only the random
numbers of particlization and hadronization differ. **An average over k is the mean hadronized
event, not more events:** it removes the hadronization noise, and an event still counts once. Data
have one sample per collision.

## 2. Notation

**Tracks**: charged, |η| < 2.4, p_T classes 1–2, 2–4, 4–10 GeV, without the Z's two leptons
(`zjet_wake.track_mask`). **The frame of Z i**: Δy = η − y_i, Δφ = φ − φ_i in [−π/2, 3π/2).

| symbol | what |
|---|---|
| H_i[X] | the histogram of the track set X in Z i's frame: p_T class × Δy (0.4) × Δφ (π/10) |
| ρ[X] | the lab-frame density of X: p_T class × η (0.1) × φ (6°) |
| shift_i(ρ) | a lab density seen from Z i: each lab bin centre moved by (−y_i, −φ_i) into Z i's frame (`ZKernel`) |
| ⟨·⟩_k | the mean over the event's K oversamples |
| k1(i) | `k_one`: the one oversample kept per event as "the recorded collision", fixed by a hash of (seed, file, event) |

**What an event contributes** (its content):

| content | Pb+Pb, sample k | pp |
|---|---|---|
| full | J_i^k + F_i^k: the whole event, as recorded | F_i^k |
| zc (Z-correlated) | J_i^k − G_b^k + F_i^k: wake + fragments, the bulk removed sample by sample | F_i^k (the same: no bulk) |

**Per Z i** (S) and **per partner event j** (what B is made of), all from the pools:

| | definition | pool name |
|---|---|---|
| S_i^full | ⟨H_i[J_i^k + F_i^k]⟩_k | `S_full` |
| S_i^zc | ⟨H_i[J_i^k − G_b^k + F_i^k]⟩_k | in `bgw_S_zc` (summed per background and window) |
| S_i^one | H_i[J_i^k1 + F_i^k1] | `S_one` |
| ρ_j^full | ⟨ρ[J_j^k + F_j^k]⟩_k | in `bgw_R_full` |
| ρ_j^zc | ⟨ρ[J_j^k − G_c^k + F_j^k]⟩_k (c the background of j) | `bgw_R_full` − `bgr_RB` |
| T_j | the tracks of J_j^k1 + F_j^k1: the only individual tracks kept | `trk_pt`, `trk_eta`, `trk_phi` |

pp: J and G are absent, so every "full" equals its "zc".

**Weights**: w_i = σ_window / N_window × N_coll (pp: no N_coll), normalized to mean 1 over the
selected Z's; W_b = Σ_{i∈b} w_i, the weight of background b. Why: §6.

**Per Z** D_i = S_i − B_i; **the result** ⟨D⟩ = Σ_i w_i D_i / Σ_i w_i (`strict`: its Z's weigh
W_b). **Observables** (`zjet_mixing.observables`, CMS's): Δφ summed over Δy, ±Δφ merged and
scaled by 0.5, per radian; Δy on the Z side |Δφ| < π/2, ±Δy merged and scaled by 0.5, per unit Δy.

**B with all partners (pooled)**, pairs weighted w_i w_j, P(i) the partners the rule allows:

  B = Σ_i w_i Σ_{j∈P(i)} w_j shift_i(ρ_j) / Σ_i w_i Σ_{j∈P(i)} w_j

Rules (`zjet_mixing.RULES`): `all` (every selected event, i included), `self` (all but i: the
exact self pair w_i² S_i taken out), `background` (all but i's background). `phi_flat`: ρ summed
over φ and spread evenly over Δφ, B flat in Δφ by construction. Since B is linear in ρ, the sum
over partners is one pooled density Σ_j w_j ρ_j shifted into each Z's frame: exactly the loop
over every pair and every oversample of every partner, without the loop and without noise of its
own.

**B with explicit partners** (`zjet_mixing.explicit_partners`): B_i = (1/10) Σ_{j∈P10(i)} H_i[T_j],
10 recorded collisions of partner events.

**The bulk term.** For one pair (i, j), i in background b, j in c:

  H_i[J_i + F_i] − shift_i(ρ[J_j + F_j]) = {H_i[J_i − G_b + F_i] − shift_i(ρ[J_j − G_c + F_j])} + {H_i[G_b] − shift_i(ρ[G_c])}

the Z-correlated pair plus the bulk term. The bulk term averages to zero when the bulk does not
know the Z (3dMCGlauber `Subtract_hard_momentum` 0; φ_Z independent of the event plane), and it
carries the bulk's fluctuations. So the full-event modes and the factorized one have the same
mean (the closure, check 3) and differ in noise.

## 3. The modes (Pb+Pb)

| mode (ANA step) | S | oversamples in S | partners | oversamples per partner | own event | own background | backgrounds per Z | Z weight | error |
|---|---|---|---|---|---|---|---|---|---|
| **factorized** (1) | S_i^zc | mean of all 40 | every selected Z event of every background, pooled (ρ^zc) | mean of all 40 | Δφ: in (B flat); Δy: out (`self`) | in | all (~400 per campaign) | w_i | jackknife, 100 blocks of backgrounds |
| **all** (2) | S_i^full | mean of all 40 | every selected Z event, pooled (ρ^full) | mean of all 40 | in | in | all | w_i | jackknife |
| **cms10** (3) | S_i^full, every selected Z | mean of all 40 | 10 events of 10 distinct other backgrounds | **one** (T_j) | out | **out** | 10 unique | w_i | jackknife, blocks out of S and B together |
| **strict** (4) | S_i^one, **one Z per background** | **one** (k1) | 10 events of the strict set, 10 distinct other backgrounds | **one** (T_j) | out | out | 10 unique | W_b | **spread over the Z's** (the jackknife, blocks out of S and B together, beside it) |

- **Partner draws in cms10 and strict** (the same code): 10 backgrounds without replacement, each
  ∝ the summed weight of its candidates (W_c), the Z's own excluded; then one candidate event in
  each ∝ w_j. cms10's candidates are all selected events (up to 24 per background); strict's are the strict set itself (one per background, picked ∝ w_j up front), so
  the same ~400 events are the Z's and the partners, as in data, and an event's recorded
  collision is both its S and what it gives as a partner. The marginal distribution of a partner
  (background ∝ W_c, event ∝ w_j) is the same in both.
- **The pooled modes keep the own background**: in `factorized` the shared fluid is already
  subtracted sample by sample; in `all` the own background is ~1/400 of a B without noise, and
  leaving it out would bias B (the exclusion bias, §6); with every partner S − B integrates to
  zero exactly. **The explicit modes exclude it**: a partner on the same fluid shares the Z's
  flow pattern and its random numbers and would cancel part of S's fluctuations, which mixing in
  data never can.
- **Rotation** (Pb+Pb, default): every background with its Z's, tracks and lab densities is
  turned by its own random multiple of 6° before mixing (3dMCGlauber puts all event planes near
  φ = 0). It changes the noise of the full-event modes, not their means; pp is not rotated.
- **Diagnostic variants** (checks 2–4, no results of their own): factorized with rule `all`;
  factorized and `all` with rule `background`.
- **All modes have the same mean**; they differ in how much is averaged and so in their noise:
  the pooled B has none, cms10's B the noise of 10 single collisions, strict's S and B both the
  noise of single collisions and its Z's are independent.

## 4. pp: the factorized result

**What a pp event is**: F_i^k only, i.e. Pythia's Z + parton (ISR and MPI on; the partons above
2 GeV handed over), Matter in vacuum, Colorless, no fluid. So "full" = "zc", and every event is
its own background (`bg_unit` = the event; the jackknife blocks are blocks of events).

**S − B** (`mix.py`: `from_all(pp, "full", ("all", "self"), phi_flat=True)`):

- S_i = ⟨H_i[F_i^k]⟩_k, the mean over the event's fragmentations;
- B pooled from the selected pp Z events (Δy: all but i), each the mean over its
  fragmentations, weighted w_j = σ_window / N_window;
- **Δφ** (rule `all`, B flat): B is the weighted mean multiplicity per Z spread evenly over Δφ, so
  S − B is S around the mean and Σ w_i (S_i − B) integrates to zero over Δφ;
- **Δy** (rule `self`, Z side): B is the pooled η density folded with each y_i, all events but
  i itself; the own event is left out because its tracks are correlated with its own y_Z (the
  recoil jet's rapidity, momentum balance), which would move part of S into B;
- Δy in 0.4 (CMS's pp tables), merged to 0.8 for Pb+Pb − pp.

It is **the infinite-partner limit of CMS's pp procedure** (each Z with 10 other Z events): the
same mean, without partner noise. That the Z side comes out negative is this subtraction at work,
as in CMS's pp data: S − B integrates to zero over Δφ and the recoil jet holds its tracks at
Δφ ≈ π, so the Z side lies below zero. The model's pp event has no soft underlying event below the
2 GeV parton cutoff (only the handed-over partons, beam remnants only where a string needs
closing); an uncorrelated underlying event cancels in S − B on average, so this matters for the
explicit modes' noise, not for the mean.

**Its relation to the Pb+Pb modes**:

| | pp factorized | Pb+Pb factorized (step 1) | Pb+Pb all / cms10 / strict (steps 2–4) |
|---|---|---|---|
| S | ⟨H_i[F_i]⟩_k | ⟨H_i[J_i − G_b + F_i]⟩_k | full event: ⟨H_i[J_i + F_i]⟩_k, or one oversample (strict) |
| B | pooled ⟨ρ[F_j]⟩_k, all other events | pooled ⟨ρ[J_j − G_c + F_j]⟩_k, all other events (own fluid included) | pooled or explicit full events |
| Δφ / Δy rules | all, flat / self | all, flat / self | all / all; explicit: own background out |
| bulk term | none (no bulk) | removed sample by sample | present: mean zero, adds noise |

- **Pb+Pb factorized is built exactly like pp**: S the Z-correlated content of the event (mean
  over its oversamples), B the pooled Z-correlated content of the other events, the same rules.
  Only the content differs: fragments in pp, fragments + wake in Pb+Pb. **Pb+Pb − pp in step 1 is
  therefore like for like**: the wake and the quenched fragments, through the same subtraction.
- In Pb+Pb the pooled B includes the other events on the Z's own fluid: harmless, because the
  bulk is subtracted per sample (check 2: excluding the own background agrees within errors).
- The full-event Pb+Pb modes add the bulk term, mean zero (check 3), so every Pb+Pb mode
  estimates what the factorized one does, and Pb+Pb (any mode) − pp (factorized) is legitimate
  for the mean. pp needs no other mode: without a fluid "full" and "zc" coincide, and its pooled
  result is CMS's pp mixing without partner noise. pp's `strict` set (one Z per event, one
  fragmentation, 10 partners) enters only check 5, the test of the strict error.

## 5. `strict` and `cms10` against CMS

CMS 2507.09307 (§5, Eqs. 1–2), the same in pp and Pb+Pb: Δ⟨N_ch⟩ = S − B, averaged over the Z's,
every Z counting 1; S the Z with all tracks of its own event; B the Z with the tracks of "a
different Z boson event within the sample", repeated with 10 other events per Z; no further event
criteria (their MC closure found the effect negligible). How the 10 are chosen is not stated.

| | CMS | `cms10` | `strict` |
|---|---|---|---|
| Z's | every Z of the sample, each one collision with its own medium | every selected Z, ~24 per fluid | **one per fluid**: independent, as in data |
| S | one collision | mean of 40 oversamples | **one** oversample |
| partner pool | the other Z events of the sample | all selected Z events of the other backgrounds | the other Z events of the strict set (~400 per campaign) |
| a partner | all tracks of one collision | one oversample | one oversample |
| partners per Z | 10 | 10, distinct backgrounds | 10, distinct backgrounds |
| excluded | the own event | the own background (the own event and the others on its fluid) | the own event (= its background) |
| partner draw | not stated (any unbiased choice) | backgrounds ∝ W_c, event ∝ w_j | ∝ W_c |
| matching | none | none, planes randomized | none, planes randomized |
| Z weights | none | w_i | W_b |
| statistical error | from its Z sample | jackknife over backgrounds | spread over the Z's |
| size | ~1,550 Z's in 0–30 % | – | ~400 per campaign; errors scaled by √(N_eff / N_CMS) (`--n-cms`) |
| tracks | efficiency-corrected; muons out by a 0.0025 cone | particle level; the two leptons out by identity | the same |

**`strict` is the closest to CMS's measurement**: one collision per Z, every Z its own medium, 10
single-collision partners from other Z events, the spread over the Z's as the error. What remains
different: the weights (importance weighting of a generated sample instead of an unweighted
physical one: the same mean, slightly different noise, the small exclusion bias), the sample size
(scaled, or several campaigns combined: ~4 give CMS's ~1,550 Z's, fewer effective ones with
the weights), the detector (`--efficiency` for
the Poisson part of corrected tracks), centrality by b with N_coll weights instead of HF classes,
and the model's physics (e.g. no minijets from the other nucleon–nucleon collisions in the
background), which is not a question of the mixing. **`cms10`** is CMS's B on every Z: the
model's full statistics with CMS's partner rule; its Z's are not independent and its S is averaged,
so its spread is not data's error (hence the jackknife).

**The explicit modes' errors.** The partners come from the same sample as the Z's: a fluid with
more particles than average raises the S of its own Z's and the B of the Z's that drew its
events, and the two cancel in the mean. A jackknife therefore leaves a block of backgrounds out
of S (its Z's) and of B (its events as partners) together, B the mean over the remaining (Z,
partner) pairs, each weighing w_i / m_i (`zjet_mixing.explicit_replicates`). Leaving it out of S
alone, each Z's S − B fixed (`cms10`'s error before 2026-10-29), counts the cancelled
fluctuations: on zjet_c00-01 5–10× the scatter of disjoint sub-campaigns at 1–4 GeV, right at
4–10 GeV. `strict`'s spread errs the same way, less (1.4–3× at low p_T); its jackknife is right
(check 5). If CMS's statistical errors are such a spread over its Z's, they carry the same
excess; how CMS computes them is not stated.

**Which to compare with what**: the model's values with the factorized (or `all`) result, which
has the same mean as every mode and the smallest error; the model's error bars with `strict`'s.
That the modes agree in the mean is the closure: CMS's procedure recovers the model's truth.

## 6. Why the events are weighted, and why partners are drawn ∝ weight

| | the production | nature, i.e. a data sample | weight |
|---|---|---|---|
| p̂T | the same number of events in every window (event i: window i mod 6) | Z's follow the cross section, which falls several-fold over the windows | σ_window / N_window |
| centrality | b sampled ∝ b on 0–8.57 fm, one Z per event | Z production ∝ N_coll: central collisions give more Z's | × N_coll |

With w the model's events stand for the physical Z population, the one CMS records unweighted.

- **Z's** are weighted by w (cms10), or one per background is picked ∝ w inside it and carries
  W_b (strict): both unbiased for the w-weighted average. Resampling Z's ∝ w and counting each
  once would look exactly like data, with the same mean and more noise.
- **Partners** are drawn ∝ w and each counts 1/10: CMS's B is an equal mean of 10 partners from
  the physical population, and this reproduces it. Uniform partners would have geometric
  centrality (too many peripheral events, too low a multiplicity: B too low, a spurious flat
  S − B) and the wrong p̂T mix; uniform partners *weighted* by w_j give the right mean but fewer
  effective partners, (Σw)² / Σw² < 10, and a small ratio bias: not CMS's noise.
- **Side effects**: the effective number of Z's N_eff = (Σw)² / Σw²; and leaving the own
  background out shifts B by a flat level, because heavier backgrounds (larger N_coll) also have
  more tracks (`zjet_mixing.exclusion_bias`): Σ_b W_b (N_b − ⟨N⟩) / (W − W_b) for explicit
  partners, Σ_b W_b² (N_b − ⟨N⟩) / (W² − Σ_b W_b²) for pooled ones, of order 1/(number of
  backgrounds). Data have no weights, so no such term.

## 7. Why one Z per fluid in `strict`

The m Z's of one background are correlated: they share the fluid's multiplicity (its total
entropy), which sets the largest part of a Z's S − B noise, the flat level of its event (~90 % of
the per-Z variance at 1–4 GeV on the test, check 6); partly its flow pattern (each Z sees it at its
own angle); and, with correlated sampling, the bulk's random numbers of equal oversample indices.
With an intra-background correlation r, a background's m Z's count as m / (1 + (m − 1) r)
independent ones:

| r | m = 1 (strict) | m = 6 (one per window) | m = 24 (all) |
|---|---|---|---|
| 0 | 1 | 6 | 24 |
| 0.5 | 1 | 1.7 | 1.9 |
| 0.9 | 1 | 1.1 | 1.1 |

So more Z's per fluid add little for a data-like error if the flat level dominates, and the spread
over correlated Z's understates the error (a jackknife over backgrounds does not, but then the
number of independent Z's to scale to CMS's N is unknown). The shared random numbers could be
avoided (Z's of one background with distinct oversample indices), the shared fluid cannot: for a
data-like error the number of independent fluids is the limit, and combining campaigns (pools of
version 4) raises it. The ratio of the jackknife variance to the naive spread variance on all
Z's would measure 1 + (m − 1) r.

## 8. In the code

| what | where |
|---|---|
| S_full, S_zc, S_one, the lab densities, the recorded tracks, per file | `zjet_mixing.file_pool` (route B: `../../analysis_root/zjet/zhadron_pools`) |
| the weights and ids of any set of pools | `zjet_mixing.assemble`, `jetcorr.weights` |
| the grid's edges in every pool, pools binned otherwise refused | `zjet_mixing.GRID`, `check_grid` |
| the pools' contents and the results as `hist.Hist` (notebooks, mplhep) | `zjet_mixing.Pool.to_hist`, `zframe_hist`, `observables_hist`, `obs_hist` ([README](README.md#the-pools-and-the-results-as-hist-histograms)) |
| the procedure step by step: S, B, S − B in 2D, projections, errors, the modes side by side | [`zhadron_mixing.ipynb`](zhadron_mixing.ipynb) |
| pooled B, rules `all` / `self` / `background`, `phi_flat` | `zjet_mixing.all_partners`, `ZKernel` |
| explicit partners (cms10, strict) | `zjet_mixing.explicit_partners`; the strict set `strict_set` |
| their jackknife, blocks out of S and B together | `zjet_mixing.explicit_replicates`; tested by `mix.subsample_pulls` (check 5) |
| the exclusion bias | `zjet_mixing.exclusion_bias` |
| Δφ / Δy as CMS (± merged, × 0.5, Z side) | `zjet_mixing.observables` |
| rotation of the backgrounds | `zjet_mixing.Pool.rotate_backgrounds` |
| which mode uses which content and rules | `mix.py` (`from_all`, `from_explicit`) |
| the checks 1–6 | [README](README.md#event-mixing-mixpy), `mixing_summary.txt` |
