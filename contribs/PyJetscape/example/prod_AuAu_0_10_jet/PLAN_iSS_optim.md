<!-- Plan, written 2026-09-26. Status: Part A (A1-A3) implemented 2026-09-26 on iSS branch
     `yield_cache` (fork jhputschke/iSS, 01f7cf9; pinned on X-SCAPE branch `iss_speedup`);
     A4 and Part B open. -->

# Plan: faster iSS, and correlated jet/background sampling

## Context

`hadronize.py` (`PLAN_particlize_h5.md`) samples each stored MUSIC freeze-out surface with
X-SCAPE's iSS. It is now the most expensive part of the hadron-level workflow: ~58 s of one CPU
core per event (both legs, 500 oversamples, 50 fragmentations), against ~35 s of GPU hydro per
event with both surfaces. This plan collects two independent improvements:

- **Part A: speed.** Remove the redundant work in iSS's per-cell yield loop. The goal is
  bit-identical hadrons, several times faster.
- **Part B: statistics.** Correlated ("common random numbers") sampling of the jet leg and its
  background, so identical parts of the two surfaces cancel in jet − background. The goal is
  ~2× smaller error bars on the wake for the same oversamples. This changes the sampling
  algorithm, so the output is not bit-identical.

Both need changes in iSS itself: upstream `chunshen1987/iSS`, branch `XSCAPE`, pinned in
X-SCAPE's `external_packages/get_iSS.sh` at `d242555`. That means a branch (or a fork) of
iSS, a new pin in X-SCAPE and, ideally, a PR upstream.

## Measurements this plan is based on

GB10, seed 1 of `prod_AuAu_0_10_jet`, 1.0 M surface cells per leg, EOS 9 (UrQMD list, 321
sampled species), iSS single-threaded.

**Timing per surface** (`hadronize.py`, idle machine):

| oversamples | wall time | peak memory |
|---|---|---|
| 1 | 18.5 s | 1.4 GB |
| 100 | 20.6 s | 1.4 GB |
| 500 | 29.3 s | 2.5 GB |
| 1000 | 41.6 s | 3.9 GB |

That is ~16.5 s fixed per surface plus ~23 ms per oversample; ~2 s of process start-up (once
per run); Colorless ~4 ms per fragmentation. Memory grows ~2.4 MB per oversample: iSS keeps
all hadrons as `shared_ptr<Hadron>` objects, ~350 B each.

**Profile of the fixed part** (`py-spy --native`, one oversample):

| share | what |
|---|---|
| 84% | `FSSW::calculate_dN_dxtdy_for_one_particle_species` (called once per species, loops over all cells), of which: |
| 49% | `calculate_dN_analytic`: thermal yield, Bessel-function series of up to 10 terms, `pow`, `exp` |
| 18% | `get22momNEOSBQSCoefficients`: δf coefficients from a table, per cell **and per species** |
| ~13% | the rest of the loop (dσ·u, bulk term) |
| ~7% | `malloc` / `free` / `new` |
| ~2% | surface conversion (`getSurfCellVector`) |

That is 321 species × 1.0 M cells = 3.2 × 10⁸ yield evaluations at ~46 ns each, independent
of the number of oversamples. The loop is `FSSW::sample_using_dN_dxtdy_4all_particles_conventional`:
for each species, compute the yield in every cell, build an inverse CDF over the cells
(`RandomVariable1DArray`), then sample.

**Why most of it is redundant.** The surface is an isotherm:
- 990,295 of 1,001,592 cells (99%) have T = 0.150 exactly, and μ = 0 everywhere, so the
  thermal yield of a species is the same number in 99% of the cells;
- the δf coefficients depend only on the cell's (e, n_B) (11,290 distinct values), not on the
  species, yet they are recomputed 321 times per cell;
- only dσ·u and Π really vary per cell.

## Part A: yield loop speed-up (bit-identical)

**A1. δf coefficients once per cell.** Compute `visCoefficients` for every cell once per
surface, before the species loop. Store them next to `FO_surf_LRF` (6 doubles per cell,
~48 MB), or inside it (it already has a `visCoeffs` member used for EOS 20). The species loop
then reads them. Removes ~18%.

**A2. Cache the thermal yield per (species, T, μ).** `calculate_dN_analytic(particle, mu, T,
results)` is a pure function. Cache its 6 results on the exact key (T, μ), per species: a small
map, or a check against the previous cell's (T, μ), since cells come in long runs with the same
T. The hit rate here is 99%. The cached values are the same doubles the call would return, so
this is bit-identical. Removes most of the 49%.

**A3. OpenMP over cells** in `calculate_dN_dxtdy_for_one_particle_species`. Each cell writes
only `dN_dxtdy_for_one_particle_species[l]`. Make `visCoefficients` loop-local (or read the
A1 array); the A2 cache is per thread or read-only after a serial fill. Per-element arithmetic
is unchanged, so this is bit-identical. The inverse CDF and all sampling stay sequential (RNG
order).

**A4 (optional). Memory.** Hand the samples to the framework without building one
`shared_ptr<Hadron>` per hadron (e.g. a binding that copies iSS's own sample arrays straight
into numpy). This would cut the ~2.4 MB per oversample and allow more oversamples per process.
It touches `iSpectraSamplerWrapper` and `bind_hadronization.cc`, not iSS.

**Expected result:** A1 + A2 take the fixed ~16.5 s per surface to ~1–2 s on one core; A3
makes the remainder negligible. The per-oversample part (~23 ms each, sequential) then
dominates: ~12 s for 500 oversamples. Per event with both legs: ~58 s → ~28 s
single-threaded (2 × (1.5 + 11.5) s + ~2 s start-up).

A3 helps latency more than throughput: in a campaign one `hadronize.py` already runs per core.
A1 and A2 help both.

**Validation.**
1. Bit-identical hadrons against the current iSS on seed 1: same particlize file, same unit
   seeds (`hadronize.py`, base seed 1), `--oversample 500`, both bulk tags. Reference files:
   `out/AuAu_0_10_jet_seed0001_hadrons_bulk_{jet,bg}.h5`.
2. The in-job vs offline check (`--validate-inline` + `--use-stored-seeds`) still bit-identical.
3. FastHydro's tests (it also runs iSS through the framework): `pytest FastHydro/tests`.
4. Timing table above, redone; `py-spy --native` profile to confirm the yield loop is gone.
5. Thread counts 1, 5, 20 give identical output (A3).

**Where.** iSS branch off `XSCAPE` at `d242555` (e.g. `yield_cache`), X-SCAPE pin in
`get_iSS.sh`, upstream PR. Watch for the `pretty_ostream` clash in `build_gpu` (see
`-Wl,-Bsymbolic` on libiSS in X-SCAPE's top-level CMakeLists): rebuild and run one
`hadronize.py` to be sure.

### Part A: done (2026-09-26), results

iSS branch `yield_cache` off `d242555` on the fork `jhputschke/iSS` (commit `01f7cf9`),
`src/FSSW.{h,cpp}` only:
- **A1:** `prepare_cell_visCoefficients()` fills the δf coefficients of all cells once per
  surface (8 doubles + a size per cell, ~65 MB for 1 M cells, freed after sampling). The yield
  loop and the per-hadron sampling both read them; `getCellVisCoefficients()` holds the one
  copy of the logic the two loops used to repeat.
- **A2:** the yield loop reuses `calculate_dN_analytic`'s result while (μ, T) stays exactly the
  same as in the previous cell (per thread).
- **A3:** `#pragma omp parallel for schedule(static)` over cells in the yield loop and the
  precompute. The CDF and all sampling stay sequential.
- Also bit-identical: the δf coefficients go to `sample_momemtum_from_a_fluid_cell` and
  `get_deltaf_bulk` by reference (they were copied per call and per trial), and
  `add_one_sampled_particle` no longer leaks one `new iSS_Hadron` per hadron.

FSSW's layout changes, and `iSS.h` inlines FSSW accessors into libJetScape and
pyjetscape_core, so **rebuild everything** (`cmake --build build_gpu`), not only libiSS:
rebuilding only libiSS segfaults in `iSpectraSamplerWrapper::PassHadronListToJetscape`.

Validation (GB10, seed 1, `--oversample 500`, both bulk tags):
1. Bit-identical to `out/..._hadrons_bulk_{jet,bg}.h5` (the old iSS also reproduced them
   first) with `OMP_NUM_THREADS` = 1, 5 and 20.
2. `--validate-inline` job (seed 900, 2 events, 100 oversamples, in-job iSS with the new code)
   against `hadronize.py --use-stored-seeds`: bit-identical (1,544,557 iSS hadrons, 94
   Colorless).
3. `pytest FastHydro/tests`: 316 passed, 8 skipped (none of them iSS tests).

Timing per surface (`hadronize.py --tags bulk_jet`, one thread, whole process):

| oversamples | before | after | peak memory before → after |
|---|---|---|---|
| 1 | 18.5 s | 5.0 s | 1.4 → 1.4 GB |
| 100 | 20.6 s | 7.2 s | 1.4 → 1.4 GB |
| 500 | 29.3 s | 16.0 s | 2.5 → 2.4 GB |
| 1000 | 41.6 s | 27.2 s | 3.9 → 3.7 GB |

With one oversample, 5 threads: 3.1 s, 20 threads: 2.9 s. Both legs, 500 oversamples:
56.5 s → 29.1 s (1 thread), 26.6 s (5), 26.0 s (20). The per-oversample part (~22 ms)
is unchanged and now dominates, as expected. `py-spy --native`, one oversample, one thread:
the yield loop is still ~50% of the process (~2.5 s), all of it the loop body itself: it
streams the ~200-byte `FO_surf_LRF` of every cell once per species (321 × 1 M), so it is
memory-bound. With OpenMP that drops to ~0.5 s. A compact per-cell array (only dσ·u, T, μ's,
Π and the coefficients) would cut it further single-threaded; not done.

**Follow-ups outside iSS (2026-09-26), also bit-identical.** A profile at 500 oversamples
showed ~30% in HDF5 writing and ~16% in `ParameterReader::getVal`:
- `HadronH5Writer.append_unit` wrote one append per oversample into 131,072-row chunks, so
  every partly filled chunk was decompressed, extended and recompressed ~18 times. It now
  writes a unit's samples in one go (new `RaggedGroup.append_many`, js-contrib).
- `iSpectraSamplerWrapper::PassHadronListToJetscape` called `iSS::get_hadron` per hadron,
  which looks `MC_sampling` up by name each time; it now takes each sample's vector once
  (X-SCAPE).

Both legs, 500 oversamples: 29.1 s → 14.6 s with one thread (56.5 s before Part A), 11.6 s
with 5. Same checks as above: bit-identical to the reference files (1 and 5 threads) and in
the in-job vs offline check; PyJetscape and FastHydro tests pass (apart from the
pre-existing `test_surface_finder` lattice-spacing failure). Profile of one surface now: the
yield loop ~28% with one thread, sampling proper ~30%, Python imports ~16%, HDF5 ~4%.

X-SCAPE branch `iss_speedup` has the wrapper fix and pins the fork in `get_iSS.sh`
(`jhputschke/iSS -b yield_cache`, `01f7cf9`). Still open: the upstream PR to
`chunshen1987/iSS` (upstream `XSCAPE` is now `cd5b7ce`, with a small overlap in
`add_one_sampled_particle`), after which the pin goes back to upstream; and A4.

## Part B: correlated jet/background sampling (common random numbers)

**The problem.** The wake is a small difference of two large numbers: a few hadrons per bin
out of ~7,000 per oversample. With independent seeds for the two legs, the noise of
jet − background is √2 × the noise of the whole bulk, including everywhere the two fireballs
are the same.

**The idea.** If both legs used the same random numbers wherever their surfaces agree, they
would produce the same hadrons there, and those cancel exactly in the difference. Only the part
of the surface the jet changed contributes noise. (Null test: identical surfaces give a
difference of exactly zero, not zero within errors.)

**How much is shared** (seed 1: each jet-leg cell against its nearest background cell in
(τ, x, y, η); emission weighted roughly by |dσ₀ u⁰|):

| match to within (position < 10⁻³, u^μ and dσ_μ) | share of cells | share of emission |
|---|---|---|
| bit-identical (all 32 fields) | 16% | – |
| 10⁻⁶ | 21% | 20% |
| 10⁻⁴ | 69% | 66% |
| 10⁻³ | 82% | 75% |
| 10⁻² | 88% | 81% |

The bit-identical cells are all from before the first deposit (τ < 1.01). Position offsets to
the nearest background cell: median 1.1 × 10⁻⁶, 90% below 9 × 10⁻⁴, 99% below 0.3. Only ~20%
of the emission differs appreciably (the wake region, plus a surface shifted slightly by the
extra energy).

**Expected gain (upper bound).** If the matched ~75–80% cancelled, the variance of the
difference would drop from ~2 × bulk to ~2 × 0.22 × bulk: ~4.5× smaller variance, ~2× smaller
errors, i.e. ~4.5× fewer oversamples for the same precision. Event-by-event wake measurements
would become practical.

**Why the same seed with today's iSS is not enough.** iSS draws an event's random numbers from
one sequence, in order:
- per species, a multiplicity (Poisson, sampling model 30);
- per hadron, a cell from the inverse CDF over the whole cell list;
- per hadron, a momentum by rejection sampling, with a variable number of draws.

The two legs' sequences stay aligned only while every draw is the same. They drift apart at the
first multiplicity that differs by one, the first rejection accepted one try later, or the
first cell index shifted by the jet leg's ~12k extra cells. After that everything is
independent. Expected: alignment is lost within the first species, so little gain.

**Design that would work** (an iSS option, e.g. `correlated_sampling = 1`): make the random
numbers a function of the physical cell, not of the order of draws.
1. **Per-cell sampling:** a Poisson multiplicity per (cell, species) with that cell's mean,
   instead of one total multiplicity plus cell selection. This is statistically equivalent (a
   Poisson superposition). It needs the A2 yields per cell anyway.
2. **Addressable random numbers:** a counter-based generator (Philox from Random123, already
   bundled at `X-SCAPE/external_packages/Random123`) keyed on (event seed, species, quantized
   cell position, hadron index). The quantization must be coarser than the Cornelius jitter
   (see offsets above) and finer than the cell size, e.g. 10⁻² in τ, x, y and η.
3. **Inversion where possible:** Poisson by inversion from one uniform, and momentum sampling
   with a fixed number of draws (or with one uniform per trial from the keyed stream), so a
   slightly different cell gives a slightly different hadron, or the same one.

Cells the wake creates or removes have no partner and sample independently; that is where the
signal is anyway.

**Consequences elsewhere.**
- **Seeds per background.** All events sharing a background (`--reuse`) use the background's
  seed for their jet leg, so each event is correlated with the same background samples.
  `hadronize.py` would pass one seed per (event, background) pair and record it.
- **Errors must be paired.** Adding the two legs' compound-Poisson errors is wrong once they
  are correlated. `HadronFileReader.jet_minus_background` needs a paired mode: per event,
  oversample k of the jet leg minus oversample k of its background, with the error from the
  spread of those per-oversample differences. That needs equal oversample counts per pair,
  so `--oversample-bg` > `--oversample` would not combine with it directly.
- **Not bit-identical** to today's iSS: the in-job/offline check needs the same mode on both
  sides.

**Validation.**
1. Null test (`--no-deposit`): jet − background exactly 0, sample by sample.
2. Means unchanged: spectra, yields and v₂ of each leg agree with independent sampling within
   errors (seed 1, many oversamples).
3. Gain measured: spread of the per-oversample wake difference against the independent case,
   same oversamples. Compare with the ~2× upper bound.
4. Energy balance of notebook §10 (bulk excess vs deposited energy) unchanged within errors,
   with smaller errors.

**Cheap first step, no iSS change.** Add a switch to `hadronize.py` that gives `bulk_bg` the
jet leg's seed. On seed 1, compare the spread of per-oversample differences with that of
independent seeds. Small gain (expected): per-cell sampling is justified. Large gain: we get
it for free.

**Effort.** It redesigns iSS's sampling loop, a clearly bigger job than Part A. Do Part A
first: it is needed anyway (per-cell yields) and speeds up every validation run of Part B.

## Related items (from the same discussion): done 2026-09-26

- **`--oversample-bg`** in `hadronize.py`, implemented as proposed. `--oversample` stays the
  jet leg's count and the default. `--oversample-bg M` overrides only `bulk_bg`;
  `--oversample-bg auto` gives each background N × `--oversample` (N = events using it),
  capped at `--oversample-bg-max` (default 2000). iSS takes the count per surface through the
  new `SoftParticlization::SetNumberOfSamples` (X-SCAPE), since its sampler reads it per event.
- **`run_hadronize.py`**, implemented:
  - `-j P`, `--skip-complete` by default, appended logs, a summary and exit code;
  - only complete particlize files, and a memory warning;
  - Ctrl-C closes the children's outputs as incomplete;
  - `--follow` stops at `run_jobs.sh`'s new `run_jobs.finished` marker.

  Tested on seeds 1 and 3, with a live `run_jobs.sh` campaign (`--follow`), a rerun, an
  incomplete input and Ctrl-C.
