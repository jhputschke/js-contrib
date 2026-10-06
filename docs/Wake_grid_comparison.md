# Wake on the output grid: 0.3875 fm vs. 0.3125 fm vs. MUSIC's 0.3 fm

**Question.** The default output grid (`grid_fno.yaml`, js-contrib PR #45) went from
65 × 65 × 33 cells at 0.3125 fm (x, y −10 … 10 fm) to 64 × 64 × 32 at 0.3875 fm × 0.3875 fm
× 0.3125 (x, y −12.2 … 12.2 fm). MUSIC evolves at 0.3 fm × 0.3 fm × 0.2. How much of the
jet's wake, Δe = e_jet − e_bg, does the coarser grid lose?

**Answer.** About as much as the old grid. The integrated wake is within 1–4% of MUSIC's on
either grid, and its shape at scales of 0.5 fm and up within about 10%. Every grid loses
the structure on the scale of one MUSIC cell, even one at MUSIC's own 0.3 fm spacing
shifted by half a cell. The new grid's only measurable cost is lower peaks (0.76 of MUSIC's
against 0.81 on the old grid) and a larger error in its worst frames, mostly in weak, narrow
wakes.

*October 2026, GB10. 6 events, one pT̂ window: see [Limits](#limits).*

## Method

The point is to vary only the sampling, not the hydro.

1. **One run on MUSIC's grid.** `run_prod_jet.py --native` writes both legs exactly as
   MUSIC stores them: x, y −15 … 14.7 fm (100 cells of 0.3 fm), η_s −6 … 5.8 (60 cells of
   0.2), one frame every 0.1 fm/c from τ = 0.4.
2. **Offline resampling.** Each event's Δe goes onto the test grids with
   `jetscape.bulk_sources.resample`, the function `PairH5Writer` itself calls in a
   production. It interpolates trilinearly between MUSIC's points; it does not average
   over cells.
3. **Measures, per frame**, in the region every grid covers (|x|, |y| ≤ 10 fm,
   |η_s| ≤ 4.84):
   - **wake energy**: Σ Δe · τ · Δx Δy Δη_s, net and split into its positive part (the
     excess, wake) and negative part (the depletion, diffusion wake), as grid / native;
   - **peak**: the largest Δe, grid / native;
   - **point error**: the grid's Δe interpolated back onto MUSIC's points, relative L2
     difference to the native Δe (what the grid lost);
   - **error at ≥ 0.5 fm**: the same after a Gaussian smoothing of both, σ = 0.5 fm in x, y
     and 0.3 in η_s;
   - **background**: Σ e_bg · τ · ΔV, grid / native.

   Σ e · τ · ΔV is a sampling measure, not the conserved energy (that needs
   T^ττ = (e + P) u^τ u^τ − P); it is computed the same way on every grid.
4. **Selection for the summary**: frames with τ ≥ 2 fm/c while both legs run (the
   background's freeze-out ends the comparison), and a real wake, i.e. native positive
   part ≥ 1 GeV and negative part ≤ −0.5 GeV, so that every ratio is finite.

**Grids compared** (min, max = the outermost cell centres):

| grid | x, y | η_s |
|---|---|---|
| **new** (`grid_fno.yaml` since PR #45) | −12.20625 … 12.20625, n = 64, step 0.3875 fm | −4.84375 … 4.84375, n = 32, step 0.3125 |
| **old** (before October 2026, and `fastdata_AuAu200_tune_0_10.h5`) | −10 … 10, n = 65, step 0.3125 | −5 … 5, n = 33, step 0.3125 |
| **0.3 shifted** (reference) | −12.15 … 12.15, n = 82, step 0.3 | −4.9 … 4.9, n = 50, step 0.2 |

The shifted grid has MUSIC's own spacing, with every point halfway between MUSIC's
points. It measures what interpolating at all costs, with no coarsening.

**Events.** 0–10% Au+Au 200 GeV, PythiaGun pT̂ 50–70 GeV, Matter + LBT and the
CausalLiquefier as in production (`AuAu_MCGlauber_MUSIC_0_10_jet.xml`). `--seed 7`,
`--reuse 2`: 6 jets on 3 backgrounds. Built from X-SCAPE `cbc72639` (MUSIC4GPU `49439c0`),
js-contrib `c8b682d`.

| event | droplets | E_droplets [GeV] | frames (jet / bg) |
|---|---|---|---|
| 0 | 11 | 8.5 | 117 / 105 |
| 1 | 27 | 40.2 | 128 / 105 |
| 2 | 19 | 35.6 | 108 / 101 |
| 3 | 21 | 40.1 | 105 / 101 |
| 4 | 26 | 34.6 | 109 / 108 |
| 5 | 17 | 27.8 | 124 / 108 |

## Results

Median over all selected frames of the 6 events, with the 5–95% range in brackets:

| | new 0.3875 | old 0.3125 | 0.3 shifted |
|---|---|---|---|
| wake energy, net | 0.999 [0.944, 1.044] | 1.003 [0.948, 1.112] | 1.000 [0.987, 1.001] |
| positive part | 0.986 [0.937, 1.003] | 0.994 [0.948, 1.084] | 0.980 [0.958, 0.987] |
| negative part | 0.961 [0.871, 0.983] | 0.969 [0.896, 1.027] | 0.950 [0.839, 0.978] |
| peak Δe | 0.758 [0.422, 0.902] | 0.813 [0.581, 0.963] | 0.708 [0.550, 0.890] |
| point error | 0.308 [0.190, 0.568] | 0.285 [0.162, 0.406] | 0.286 [0.188, 0.449] |
| error at ≥ 0.5 fm | **0.096** [0.055, 0.198] | **0.109** [0.058, 0.200] | **0.083** [0.059, 0.109] |
| background | 1.000 [1.000, 1.002] | 1.000 [1.000, 1.006] | 0.998 [0.989, 0.999] |

### What it means

- **The wake's energy is kept.** The net wake agrees with MUSIC's to about 1% (median) on
  every grid. The depletion comes out 3–5% low on all three, including the shifted 0.3 fm
  grid, so that loss comes from interpolating, not from the larger step. On the old grid
  the 95th percentile of the net is 1.11: event 0 at τ = 6–8 fm/c (below).
- **The point error is the same floor everywhere.** About 30% of Δe's norm is lost
  point by point on every grid, also on the 0.3 fm grid that only shifts the points.
  MUSIC's wake has structure on the scale of one cell: the liquefier's kernels are
  narrower than 0.3 fm (`width_delta` 0.1, `d_diff` 0.08 fm). Any resampling smooths it,
  and the 0.3875 fm step does not raise the median.
- **The shape at ≥ 0.5 fm is kept to ~10%** on all three grids (new 0.096, old 0.109): the
  scales a neural operator trained on these files and the hadrons respond to.
- **The background does not care** (within 0.2%, the shifted grid's 0.998 being its edge
  in η).
- **The new grid's cost is in the peaks and the worst frames.** Its peaks are 0.76 of
  MUSIC's against 0.81 on the old grid, and its point error reaches 0.57 in the worst 5% of
  frames against 0.41. The worst case is event 0, the weakest wake (8.5 GeV of droplets):

  | event 0 | net | peak | point error |
  |---|---|---|---|
  | τ = 6 fm/c, new | 0.94 | 0.45 | 0.63 |
  | τ = 6 fm/c, old | 1.13 | 0.99 | 0.31 |
  | τ = 6 fm/c, 0.3 shifted | 1.00 | 0.64 | 0.48 |
  | τ = 8 fm/c, new | 0.95 | 0.57 | 0.50 |
  | τ = 8 fm/c, old | 1.12 | 0.96 | 0.24 |
  | τ = 8 fm/c, 0.3 shifted | 0.99 | 0.66 | 0.37 |

  A narrow structure here falls between the new grid's points. The shifted 0.3 fm grid
  misses its peak too (0.64), while the old grid happens to land near it (0.99, but with
  the net 12–13% high). So which grid "sees" a narrow peak depends on where its points
  fall; a coarser step makes a miss more likely, not certain. The other five events show
  no such pattern (`--per-event` output of the script).

## Limits

- 6 events, one pT̂ window (50–70 GeV), 3 backgrounds. Weaker jets (10–40 GeV, the
  `pth10-40` campaigns) deposit less and in narrower wakes, where the new grid's peak loss
  matters most; not tested.
- Only frames while both legs run: the jet leg's late frames after the background froze
  out are not compared.
- Energy density only, not the flow velocities.
- No hadron-level observable, and no test of what an FNO trained on either grid learns.

## Options if the peaks matter

- `{min: -12.4, max: 12.4, n: 80}` in x, y: step 0.314 fm, the old grid's resolution over
  the new range, 1.56× the cells of the 64 grid (80 is not a power of two).
- Keep the new grid and accept that the wake is resolved at ≥ 0.5 fm, which is what this
  comparison finds every grid, MUSIC's own spacing included, delivers.

## Reproduce

```bash
conda activate js_fno
cd external_packages/js-contrib/contribs/PyJetscape/example/prod_AuAu_0_10_jet
python run_prod_jet.py --native --events 6 --reuse 2 --seed 7 --outdir OUT --seed-registry none
python wake_grid_compare.py OUT/AuAu_0_10_jet_seed0007.h5 --per-event
```

The run took 2.6 min on the GB10 and writes 4.3 GB (both legs on MUSIC's grid); the
comparison takes a few minutes and ~1 GB per leg and event. The grids compared are the
`GRIDS` table at the top of
[`wake_grid_compare.py`](../contribs/PyJetscape/example/prod_AuAu_0_10_jet/wake_grid_compare.py).
