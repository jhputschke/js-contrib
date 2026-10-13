# The jet source in MUSIC4GPU: normalisation of the liquefier's droplets on the hydro grid

Status 2026-10-13. The fix is on X-SCAPE `contrib` since 2026-09-29 (PR #155) and in every
production image since `cu126`/`cu130` of 2026-09-30. Its extension to MUSIC's late time step is on
`contrib` since 2026-10-13 (PR #171).

## 1. Summary

MUSIC_2 (the jet leg of the two-stage production) receives the jet's energy loss as droplets from
X-SCAPE's `CausalLiquefier`. MUSIC samples the liquefier's smearing kernel once per cell centre, in
the one time step that deposits the droplet, and adds τ Δτ Δx Δy Δη J^μ per cell. Nothing made that
sum equal the droplet's four-momentum. Where the grid does not resolve the kernel (large |η_s|, late
τ), MUSIC received between 0.004 and 6 times the droplet. In the 0–10 % Au+Au pT̂ 10–40 production,
40 % of the events had their injected energy off by more than 20 %. Droplets at the hard vertex were
lost entirely (η = 0/0).

The kernel is the analytic causal-diffusion solution, normalised in the continuum and validated on a
0.05 fm grid; the production grid (0.3 fm, 0.2 in η) does not resolve it (§3). The fix normalises each
droplet by the sampled sum of its own kernel on MUSIC's grid, computed with the same code path MUSIC
will query, so the deposit is exact by construction. Parton level and the
background leg are unchanged; the jet leg's wake changes event by event, mostly forward and late.

## 2. The problem

**How the source reaches MUSIC.** MUSIC4GPU has a second source slot next to the initial-state
strings: `hydro_source_terms_from_jets_ptr_`
([MUSIC4GPU `b9cc8be`](https://github.com/jhputschke/MUSIC4GPU/commit/b9cc8be6f086e7f7803f9fc463f0b3657ac9dbdd),
on both the CPU and the GPU path). X-SCAPE's `HydroSourceJETSCAPE` (in `MusicWrapper`) implements
it and forwards each query to `LiquefierBase::get_source(τ, x, y, η)`, which sums the smearing
kernels of all droplets at that point. MUSIC calls it at every cell centre, at the two Runge–Kutta
query times of a step, and adds τ Δτ ΔV J^μ to the cell.

**Why the sum is not the droplet.** The `CausalLiquefier` kernel of a droplet deposited at
(t_d, x_d) is a ball of radius c_diff (t − t_d) in the lab, seen by MUSIC at t − t_d ≈ cosh η_d · τ_delay.
In the lab z direction one η cell is τ Δη cosh η long. So at large |η_s|, or late τ, the ball covers
only a few cell centres, and the point-sampled sum of the kernel can be anything from ~0 (the ball
falls between cell centres) to several times 1 (a few cells, each weighted as if the kernel were
flat across them). The τ surface is also tilted against the droplet at large |η_s|. The kernel is
normalised analytically, not on the grid it is sampled on.

**A second, smaller loss.** A droplet created at the hard vertex (t = z = 0) got η = atanh(z/t) =
0/0 = NaN in `add_hydro_sources`. The kernel is then NaN everywhere and MUSIC never received it:
13 droplets, 56 GeV, in the 300 events of the pT̂ 10–40 production.

**A third, independent problem found in the same analysis** (not a normalisation issue, listed
for completeness): a free-streaming LBT parton that the liquefier absorbed was deposited as a
droplet and also hadronised, so 13.9 % of the initial jet energy on average (up to 67 %) was
counted twice. Fixed in
[X-SCAPE PR #154](https://github.com/JETSCAPE/X-SCAPE/pull/154)
([`780f5036`](https://github.com/JETSCAPE/X-SCAPE/commit/780f5036)); old files can be repaired
with `fix_lbt_double_counting.py`.

## 3. What the original kernel assumes, and where it is valid

The `CausalLiquefier` kernel is the analytic Green's function of the causal diffusion equation
(`kernel_rho`, `kernel_j`): a smooth interior (Bessel functions I₁, I₂ inside the ball
r < c_diff t) plus a wave-front shell at r = c_diff t (`rho_delta`, a δ-function of the
analytic solution given the finite width `width_delta`), damped by e^{−γt}/4π. It is normalised
**in the continuum**: ∫ 4π r² ρ(t, r) dr = 1 for every t. The original unit tests check exactly
that:

| test (`examples/unittests/causal_liquifier.cc`) | grid | tolerance | status |
|---|---|---|---|
| `TEST_CONSERVATION` | radial, dr = dt = 0.005 fm | 5 % | runs |
| `TEST_GRID_CARTESIAN_CONSERVATION` | dx = 0.05, 0.1, 0.3 fm, dt = 0.1 | 5 % | **disabled**: the loop bound is `i < 0`, so it has never run |
| `TEST_GRID_TAU_ETA_CONSERVATION` | dx = dy = dη = 0.05, dτ = 0.1, droplet at η = 1, τ_delay = t | 5 %, only for t > 3 fm | runs |

So the kernel was validated on a 0.05 fm grid, with the droplet seen after a delay of more than
3 fm, when the ball has a radius of ~2.7 fm (c_diff = √(d_diff/τ_relax) = 0.89 for the
production parameters) and spans ~100 cells. It was never checked at the production setting,
where it is sampled on a 0.3 fm × 0.3 fm × 0.2 (η) grid with τ_delay = 1 fm/c, so the ball has
radius ~0.9 fm · cosh η_d at midrapidity, i.e. ~6 cells across, and three things break the
continuum assumption:

- **The wave-front shell is thinner than a cell.** `width_delta` = 0.1 fm against 0.3 fm cells.
  At midrapidity and τ_delay = 1 the shell carries e^{−x}(1 + x + x²/2) with x = γt = 5, about
  12 % of the droplet, and it is point-sampled by whichever cell centres happen to fall inside
  a 0.1 fm thick sphere. The shell term alone can therefore contribute anywhere between 0 and
  several times its share. That is the main source of the 6× factors near midrapidity.
- **Cells are long in z at large |η_s|**: one η cell is τ Δη cosh η in the lab, so at η_s = 2
  and τ = 3 fm/c a cell is 2.3 fm long and the whole ball sits in one or two η slices; the
  interior's cosh/sinh variation across the cell is not sampled.
- **The τ = const surface is tilted** against the droplet's light cone at large |η_s|, so the
  Runge–Kutta query times miss or double-count part of the deposit.

With cells small against both the shell width and the ball, the point-sampled sum converges to
the continuum integral and the original source is correct; the normalisation factor then tends
to 1. On the production grid it does not, by factors of 0.004 to 6. The fix keeps the kernel and
makes the sampled sum what the continuum integral would have given, per droplet.

## 4. How it was discovered

The wake analysis of the first full production
([js-contrib `3f9eeee`, 2026-09-28](https://github.com/jhputschke/js-contrib/commit/3f9eeee5d76acc891a508f9710d7450ce213a0fa),
`contribs/PyJetscape/example/analysis/wake_observables.py`) does the energy bookkeeping of every
event exactly through the shower graph: initial partons = surviving partons + droplets, and
compares the droplets to the wake's ΔP^μ = P^μ(jet leg) − P^μ(background), integrated over every τ
surface with MUSIC's EoS.

- The parton-side balance did not close: that was the LBT double counting
  ([js-contrib `d19553d`](https://github.com/jhputschke/js-contrib/commit/d19553d7e4ed48581946f3d175145abd17fdce25),
  `jet_edep_balance_check.py`).
- After that, the wake's ΔP^μ still did not follow the droplets. The script ports the kernel to
  Python (`KernelFlux`, checked against the C++ `LiquefierBase::get_source` to 5e-5) and computes
  each droplet's `kernel_flux`, the fraction MUSIC actually received: 0.01 to 6, worst at large
  |η_s| and late τ. The wake's ΔP^μ followed the kernel-weighted deposit to ~2 % until τ ≈ 8 fm,
  not the droplet energy. The correlation of the wake's ΔP⁰ with its own droplets at τ = 7.5 fm
  was 0.55.

The analysis README (`contribs/PyJetscape/example/analysis/README.md`, section 6) records both
findings.

## 5. The solution

Normalise each droplet on the grid it will be sampled on, with the sampling MUSIC will do:

1. `MusicWrapper` passes MUSIC's computational grid (nx, ny, nη, spacings, origin) to the liquefier
   (`HydroSourceJETSCAPE::set_hydro_grid`).
2. In MUSIC's per-step hook `prepare_list_for_current_tau_frame(τ)` (the same hook that prunes the
   droplets to those that can deposit in this step), `LiquefierBase::normalize_active_droplets(τ, Δτ)`
   computes, once per droplet and before its first query, `sampled_flux()`: the sum over all cell
   centres and over the hydro's query times near the deposit of `smearing_kernel()` applied to a
   unit droplet at the same position, times ΔV and the Runge–Kutta weights. This is exactly what
   `get_source()` will return summed over MUSIC's calls.
3. `get_source()` scales the droplet by 1/flux.
4. Edge cases: a kernel that misses every cell centre is deposited whole into the nearest cell; a
   droplet off the grid is counted as lost and reported. MUSIC logs one summary line per event
   (droplets, fluxes, lost energy).
5. A vertex droplet gets τ = η = 0 (any η is the same point there).
6. `<Liquefier><normalize_on_hydro_grid>` (default 1) switches it; with 0 the source is
   bit-identical to the old behaviour. `CausalLiquefier` opts in through `normalizable_on_grid`,
   `deposit_time` and `deposit_half_width`; a liquefier without them is unchanged.

The normalisation fixes the integral, not the shape: a large-|η_s| droplet still lands in a few
cells (§3, §8). The step follows MUSIC's actual Δτ (see §9).

## 6. Implementation

**X-SCAPE** ([PR #155](https://github.com/JETSCAPE/X-SCAPE/pull/155), branch
`liquifier_kernel_normalization`, commit
[`dab5f944`](https://github.com/JETSCAPE/X-SCAPE/commit/dab5f944e451e8185768e80871c3441c52c684bb),
merged to `contrib` 2026-09-29 as `4a7a158f`):
- `src/framework/LiquefierBase.{h,cc}`: `HydroGrid`, `set_hydro_grid()`, `sampled_flux()`,
  `normalize_active_droplets()`, `get_droplet_flux()`, the switch, the per-event summary;
  `get_source()` applies the factor. `XSCAPE_LIQUEFIER_GRID_NORMALIZATION` is defined for code that
  must build against both.
- `src/liquefier/CausalLiquefier.{h,cc}`: the opt-in (`deposit_time`, `deposit_half_width`), the
  XML read.
- `src/hydro/MusicWrapper.{h,cc}`: the grid handed over at every initial condition;
  `HydroSourceJETSCAPE::prepare_list_for_current_tau_frame` calls the normalisation.
- `src/framework/JetEnergyLoss.cc`: the vertex droplet.
- `examples/unittests/causal_liquifier.cc`: four new tests that run the liquefier the way MUSIC
  does (cell centres, Heun weights).

**MUSIC4GPU** (no change for the fix itself; the hooks it relies on):
- the jet source slot
  ([`b9cc8be`](https://github.com/jhputschke/MUSIC4GPU/commit/b9cc8be6f086e7f7803f9fc463f0b3657ac9dbdd))
  and `prepare_list_for_current_tau_frame` called on it once per step
  ([`3037be7`](https://github.com/jhputschke/MUSIC4GPU/commit/3037be76518e4ae17b42c2236392bc93603c9f89)),
  both merged to `XSCAPE` with [PR #9](https://github.com/jhputschke/MUSIC4GPU/pull/9)
  (2026-09-25). The hook is where the droplet pruning
  ([X-SCAPE PR #141](https://github.com/JETSCAPE/X-SCAPE/pull/141)) and the normalisation run;
  `has_active_sources_current_tau()` ([X-SCAPE PR #144](https://github.com/JETSCAPE/X-SCAPE/pull/144),
  [MUSIC4GPU PR #10](https://github.com/jhputschke/MUSIC4GPU/pull/10)) lets MUSIC skip the source
  at steps with no droplet.

**js-contrib** ([PR #29](https://github.com/jhputschke/js-contrib/pull/29), commits
[`fa946f7`](https://github.com/jhputschke/js-contrib/commit/fa946f7e0a107ad4d671900b157b29edef676098),
[`5f92bcf`](https://github.com/jhputschke/js-contrib/commit/5f92bcff45dc277bbf2580bb0d9da2cea4a6c1d0)):
- bindings `get/set_normalize_on_hydro_grid`, `droplet_fluxes_numpy`, `normalization_summary`;
- `PairH5Writer` stores the root attribute `liquefier_normalize_on_hydro_grid` (1 / 0 / −1 for a
  build without it) and each droplet's in-run C++ flux in `source/flux`;
- the production XML sets `<normalize_on_hydro_grid>1</normalize_on_hydro_grid>` explicitly;
- `wake_observables.py` uses the stored flux (`inj_factor` = 1 for normalised files, the Python
  port for older ones).

## 7. Validation

- **Unit tests:** three production droplets deposit exactly their four-momentum with the
  normalisation and 6.19 / 0.978 / 0.0125 times it without; several droplets per step; the
  nearest-cell deposit; a droplet off the grid; a vertex droplet. 9/9 liquefier tests pass.
- **End to end**, job 0003 of the pT̂ 10–40 production rerun with the same seeds (15 events):
  - droplets, showers, the background leg and its freeze-out are bit-identical;
  - the in-run fluxes equal the standalone test's;
  - the wake's ΔP⁰ / droplet energy at τ = 7.5 fm: median 1.02, 16–84 % range 0.95–1.06, against
    0.70–1.11 before; over 285 events of the two productions the correlation of wake and droplets
    goes from 0.55 to 0.97;
  - with the switch off, the jet leg is bit-identical to the old run in the 14 events without a
    vertex droplet.
- **Where the change shows up** (job 0003, wake ΔP⁰ at τ = 7.5 fm by |η_s| band, GeV):

  | \|η_s\| | droplets deposited | old wake | fixed wake | Σ \|fixed − old\| per cell |
  |---|---|---|---|---|
  | 0 – 0.5 | 342 | 285 | 291 | 36 |
  | 0.5 – 1 | 105 | 172 | 180 | 60 |
  | 1 – 2 | 120 | 130 | 122 | 152 |
  | 2 – 5 | 86 | 53 | 76 | 217 |

  Near midrapidity the wake changes by ~2 %; forward, the energy had been in the wrong places
  (the cell-by-cell difference exceeds the forward wake). Droplets deposited after τ ≈ 7 fm had
  fluxes down to 0.013, so the late wake and the reheating near freeze-out were underestimated.
- **Hadron level** (`Production_readiness_summary.md` §5): jet − background bulk hadrons
  +24.6 ± 5.6 GeV at |η| < 2 against 25.0 GeV deposited; jet fragments equal to the surviving
  partons.

## 8. Suggested test: the current source against the original in the fine-grid limit

The normalised source should agree with the original (unnormalised) source wherever the
original is valid, and the two should differ only by the original's discretisation error. That
is not yet tested. Three levels, cheapest first:

1. **Unit test, convergence of the raw flux** (`causal_liquifier.cc`, new test; minutes). For the
   three production droplets already in `TEST_NORMALIZED_ON_HYDRO_GRID` (fluxes 6.19, 0.978,
   0.0125 on the production grid) and a few more at η_s = 0, 1, 2, 3 and τ_d = 0.5, 3, 7 fm/c,
   compute `sampled_flux()` on grids refined by 2, 4, 8 in x, y, η together with Δτ (0.3/0.2/0.02
   → 0.0375/0.025/0.0025). Expect the raw flux → 1 within 5 % (the original tests' tolerance) at
   the finest grid, monotonically, and the normalised deposit = p at every resolution. The
   same loop re-enables `TEST_GRID_CARTESIAN_CONSERVATION` (fix the loop bound) with
   `width_delta` ≥ 2 dx as a precondition it should state.
2. **Unit test, shape** (same file; minutes). On the fine grid, average the original source
   field into the production cells; compare with the normalised source sampled on the production
   grid, per droplet: L2 of the difference over the energy, and the first moments (centroid,
   r.m.s. radius in x, y, η). The normalisation fixes the integral only, so this measures what
   remains; the expectation is a per-droplet shape error of order the shell share (~10 %) at
   midrapidity and larger forward, which the hydro then smooths. A tolerance can be set from
   the numbers once measured; the test's value is the record of the size.
3. **End to end, MUSIC on a fine grid** (GB10 or 3090 box; hours). `prod_AuAu_0_10_jet`, a few
   events with fixed seeds, three runs: (a) production grid, normalisation on (the current
   default); (b) a 0.1 fm × 0.1 fm × 0.1 (η) MUSIC grid with Δτ = 0.01 fm/c
   (`grid_step_x/y/z`, `delta_tau`; ~27× cells, ~3× steps, ~80× GPU time per event, i.e.
   ~10–15 min per jet event on a 3090), normalisation **off**: the original source in the
   regime it was made for; (c) the same fine grid with normalisation on. Compare with
   `wake_observables.py`: the deposited energy per droplet (the in-run `source/flux`, which in (b)
   should be 1 within a few %), the wake's ΔP^μ(τ) of all three legs, and the wake field of (a)
   resampled onto (b)'s grid at τ = 4, 6, 8 fm/c. Expect (b) ≈ (c) at the percent level
   (the original converged), and (a) to agree with them in ΔP^μ at the percent level and in the
   wake field at the level the shape error of step 2 predicts, concentrated forward. This is
   the direct check that the normalised production source reproduces what the original would
   give with enough resolution, and it also bounds the production grid's own discretisation of
   the wake, which no present check does. The late step (D1) should be off in (b) and (c).

The same three runs serve as the reference for any later change of the kernel's sampling (for
instance depositing the shell term by cell-volume integration instead of point sampling, which
would remove most of the shape error).

## 9. Follow-up: the late time step (D1, `final_optim` branches, 2026-10-12)

MUSIC's `late_delta_tau` / `late_tau_switch` (0.05 fm/c from τ = 3 fm/c in the productions,
[MUSIC4GPU `547da26`](https://github.com/jhputschke/MUSIC4GPU/commit/547da26d7a0635d485ee05f1cd92d173fb51963c))
changes Δτ during a run. The liquefier's deposit window and its normalisation had assumed the
step given at the start, so every droplet after the switch would have been deposited 2.5× too
strongly or missed. [X-SCAPE `d425575e`](https://github.com/JETSCAPE/X-SCAPE/commit/d425575e4997d0aa3c15631ebcb75c45b3cf095d)
(branch `final_optim`, PR pending against `contrib`):
- `HydroSourceBase::hydro_dtau_changed(Δτ, τ)` (MUSIC4GPU) tells the source about the switch;
  `HydroSourceJETSCAPE` updates the pruning window and the 1/Δτ factor.
- `CausalLiquefier` knows the schedule (`set_late_hydro_step`): the deposition window of a query
  at τ is [τ − step_before(τ)/2, τ + step_from(τ)/2), so the windows tile the axis across the
  switch and both Runge–Kutta stages at the switch time see the same droplets.
- `sampled_flux()` takes its query times from `query_time(τ, Δτ, k)`, which walks the schedule.
- With a constant step everything is bit-identical to PR #155.

Checked on the 3090 box (`REPORT_FINAL_OPTIM.md`, 300 Au+Au events, D1 on vs off): the deposited
energy of same-shower events agrees to 1e-4, and the wake carries it (wake energy / deposit
0.9836 → 0.9864, +0.3 % ± 0.2 %). The switch is logged once per run (`late step: delta_tau 0.02 ->
0.05 fm/c at step 130 ...`).

## 10. Known limits and consequences

- **Integral, not shape.** A large-|η_s| droplet still lands in a few cells. A hole (negative
  droplet) concentrated there can push a cell below zero energy, which MUSIC resets silently
  (`reconst.cpp`), adding energy: +23 % in one of the 15 events of job 0003.
- **Productions made before 2026-09-29 cannot be repaired:** their jet leg was evolved with the
  wrong source. Rerun them; with the same seeds only `arr` (the jet leg) changes. Files carry the
  attribute `liquefier_normalize_on_hydro_grid` so the analysis can tell.
- **FNO training:** the model learns the map from droplets to Δe. Before the fix, Δe did not
  correspond to the droplets the model is given, biased forward and late. Fixed productions give
  consistent pairs; D1 changes the cell-level wake at the ~5 % level on top (see
  `REPORT_FINAL_OPTIM.md`), so one setting per campaign.

## 11. Links

| what | where |
|---|---|
| the fix | [X-SCAPE PR #155](https://github.com/JETSCAPE/X-SCAPE/pull/155), [`dab5f944`](https://github.com/JETSCAPE/X-SCAPE/commit/dab5f944e451e8185768e80871c3441c52c684bb) |
| LBT double counting (found with it) | [X-SCAPE PR #154](https://github.com/JETSCAPE/X-SCAPE/pull/154), [`780f5036`](https://github.com/JETSCAPE/X-SCAPE/commit/780f5036) |
| the analysis that found both | [js-contrib `3f9eeee`](https://github.com/jhputschke/js-contrib/commit/3f9eeee5d76acc891a508f9710d7450ce213a0fa), [`d19553d`](https://github.com/jhputschke/js-contrib/commit/d19553d7e4ed48581946f3d175145abd17fdce25); `contribs/PyJetscape/example/analysis/README.md` §6 |
| recording the flux, analysis of the fix | [js-contrib PR #29](https://github.com/jhputschke/js-contrib/pull/29), [`fa946f7`](https://github.com/jhputschke/js-contrib/commit/fa946f7e0a107ad4d671900b157b29edef676098), [`5f92bcf`](https://github.com/jhputschke/js-contrib/commit/5f92bcff45dc277bbf2580bb0d9da2cea4a6c1d0) |
| MUSIC4GPU jet source slot and per-step hook | [`b9cc8be`](https://github.com/jhputschke/MUSIC4GPU/commit/b9cc8be6f086e7f7803f9fc463f0b3657ac9dbdd), [`3037be7`](https://github.com/jhputschke/MUSIC4GPU/commit/3037be76518e4ae17b42c2236392bc93603c9f89), [PR #9](https://github.com/jhputschke/MUSIC4GPU/pull/9), [PR #10](https://github.com/jhputschke/MUSIC4GPU/pull/10) |
| droplet pruning, source skip | [X-SCAPE PR #141](https://github.com/JETSCAPE/X-SCAPE/pull/141), [PR #144](https://github.com/JETSCAPE/X-SCAPE/pull/144) |
| late time step (D1) | [MUSIC4GPU `547da26`](https://github.com/jhputschke/MUSIC4GPU/commit/547da26d7a0635d485ee05f1cd92d173fb51963c), [X-SCAPE `d425575e`](https://github.com/JETSCAPE/X-SCAPE/commit/d425575e4997d0aa3c15631ebcb75c45b3cf095d) (branches `final_optim`, PRs pending) |
| physics check of D1 | `docs/REPORT_FINAL_OPTIM.md` §2 |
| summary of all jet–medium fixes | `docs/Production_readiness_summary.md` §5 |
