# hydro_hist_vs_surface — hadrons from the saved hydro history vs. MUSIC's full surface

The pair files of [`../prod_AuAu_0_10_jet`](../prod_AuAu_0_10_jet/README.md) (the FNO
training data) keep only the **hydro history**: $e, v_x, v_y, v_z$ on the output grid
(0.3125 fm × 0.3125 fm × 0.3125 in $\eta_s$, every 0.1 fm/c), with no viscous fields. An
FNO that predicts such a history can be hadronized only through a freeze-out surface
**found in that history**, sampled **without viscous corrections**. This study measures how
far those hadrons are from the correct ones: iSS on MUSIC's own surface with $\pi^{\mu\nu}$
and $\Pi$, which `run_prod_jet.py --write-particlize both` stores.

The comparison uses the same events, the same `hadronize.py` settings and the same seeds as
the reference hadrons. Only the surface changes:

| variant | surface | viscous corrections in iSS | isolates |
|---|---|---|---|
| `ref` | MUSIC's (native grid 0.3 fm × 0.2 in $\eta_s$, Δτ = 0.1 fm/c, $\lvert\eta_s\rvert$ ≤ 5.8) | shear + bulk δf | the correct answer |
| `ref_no_shear` | MUSIC's, $\pi^{\mu\nu}$ = 0 | bulk δf only | shear δf |
| `ref_no_bulk` | MUSIC's, $\Pi$ = 0 | shear δf only | bulk δf |
| `ref_ideal` | MUSIC's, $\pi^{\mu\nu}$ = $\Pi$ = 0 | none | all of δf |
| `hist` | Cornelius on the saved history (X-SCAPE `SurfaceFinder`), $T$ from MUSIC's EoS 9 | none | + the coarse history |

`hist − ref` is the total error of hadronizing from the history. It splits into the
viscous part, `ref_ideal − ref` (itself split into shear and bulk), and the history part,
`hist − ref_ideal`: grid resolution, interpolation, and the output grid's $\lvert\eta_s\rvert$ ≤ 5.

## Files

| file | purpose |
|---|---|
| `make_surfaces.py` | builds one variant's particlize files from the reference particlize files (and, for `hist`, their pair files) |
| `run_study.sh` | the whole study, resumable: all variants → `hadronize.py` → `analyze.py` |
| `analyze.py` | every variant's hadron and surface files → one small `summary.npz` (per event and leg) |
| `hydro_hist_vs_surface.ipynb` | reads `summary.npz`: figures, tables, findings |
| [`../../tests/test_hist_surface.py`](../../tests/test_hist_surface.py) | the `hist` surface builder on a synthetic evolution with a known surface |

## Requirements

- An X-SCAPE build with iSS and PyJetscape (`pyjetscape_core`), as for `hadronize.py`
  (`conda activate js_fno` on the GB10). The `hist` surfaces also need the
  `FluidDynamics.find_freezeout_surface` / `surface_to_numpy` bindings and MUSIC's EoS 9
  table (`<build>/EOS/hotQCD/hrg_hotqcd_eos_binary.dat`; `--eos` for another path).
- A production directory made with `run_prod_jet.py --write-particlize both`, i.e. the
  pair files `<stem>.h5` next to `<stem>_particlize.h5`, and the reference hadrons made from
  them with `hadronize.py` / `run_hadronize.py` (`<stem>_hadrons_{bulk_jet,bulk_bg,jet_frag}.h5`).

## Run

```bash
conda activate js_fno
cd external_packages/js-contrib/contribs/PyJetscape/example/hydro_hist_vs_surface

# everything (surfaces, hadrons, summary); re-running resumes
PYTHON=$(which python) ./run_study.sh /home/putschke/FNO_Hydro_Data/AuAu_0_10_h5_pair_surface_test

# then open the notebook (it reads DATA/hist_vs_surface/summary.npz, or $HVS_SUMMARY)
jupyter lab hydro_hist_vs_surface.ipynb
```

`run_study.sh [DATA [OUT]]` writes one directory per variant under `OUT` (default
`DATA/hist_vs_surface/<variant>/`), with the reference's file names, and
`OUT/summary.npz`. The hadronization settings have to be the reference's: read them off
`DATA/*_hadronize.log` and pass them as `HADRONIZE_OPTS` if they differ from the default,
`--oversample 100 --n-frag 50 --correlated --keep-bits-p 12 --keep-bits-x 8` (what the test
data used). `analyze.py` warns if a variant's hadron files were sampled differently from the
reference (seed, seed scheme, correlated sampling, number of samples).

Environment of `run_study.sh`: `PYTHON`, `HADRONIZE_OPTS`, `HADRONIZE_J` (parallel
`hadronize.py`, default 4, `OMP_NUM_THREADS` = cores / J each), `VARIANTS` (default
`ref_ideal ref_no_bulk ref_no_shear hist`).

The steps one by one:

```bash
DATA=/home/putschke/FNO_Hydro_Data/AuAu_0_10_h5_pair_surface_test
OUT=$DATA/hist_vs_surface
python make_surfaces.py hist      $DATA --out-dir $OUT/hist            # ~62 s per event (both legs)
python make_surfaces.py ref_ideal $DATA --out-dir $OUT/ref_ideal       # ~40 s per file
python ../prod_AuAu_0_10_jet/run_hadronize.py $OUT/hist -j 4 \
       --oversample 100 --n-frag 50 --correlated --keep-bits-p 12 --keep-bits-x 8
python analyze.py --ref $DATA --variants-dir $OUT \
       --variants ref ref_no_shear ref_no_bulk ref_ideal hist --out $OUT/summary.npz

# quick look at a few events: --events a:b in make_surfaces.py, hadronize.py and analyze.py
python make_surfaces.py hist $DATA/AuAu_0_10_jet_seed0001_particlize.h5 --out-dir /tmp/t/hist --events 0:2
```

**Cost (GB10, 20 cores, 4 files × 25 events).**

| step | time | disk |
|---|---|---|
| `make_surfaces.py hist` | 1 h 44 min: 31 s per leg with all 20 threads (breakdown below) | 1.8–1.9 GB per file (70% of the reference's cells) |
| `make_surfaces.py ref_*` | 35–47 s per file (a copy with columns zeroed) | 2.2 GB (`ref_ideal`), 2.3 GB (`ref_no_shear`), 4.2 GB (`ref_no_bulk`) per file; the reference is 4.3–4.6 GB |
| `hadronize.py`, both legs, 100 oversamples | 3.6–3.9 min for the 4 files at once (`-j 4`) | ~0.6 GB per file (`--keep-bits-p 12 --keep-bits-x 8`) |
| `analyze.py` | 5.4 min (all 5 variants, 100 events; 2 min hadrons, 3.4 min surfaces) | 1.9 MB |
| **all** (`run_study.sh`) | **2 h 12 min** (12:17 → 14:30) | **51 GB** (hist 9.3, ref_ideal 12, ref_no_shear 12, ref_no_bulk 19) |

`ref_no_bulk` is the largest because it keeps the ten $\pi^{\mu\nu}$ columns, which compress
much worse than zeros.

**The `hist` surfaces, in detail** (from `run_study.log`: the step ran 12:36:34 → 14:20:37,
6243 s; the 200 per-leg times in the log add up to 6241 s). One process at a time, every
leg with all 20 cores:

| | legs | per leg (mean) | range | total |
|---|---|---|---|---|
| jet leg (`arr`) | 100 | 32.7 s | 26.7–41.0 s | 54.4 min |
| background leg (`arr_bg`) | 100 | 29.8 s | 26.6–34.1 s | 49.6 min |
| **all** | **200** | **31.2 s** | | **104.0 min** |

- **Per file** it is very even: 25.9, 26.3, 25.5 and 26.3 min for seeds 1–4, i.e. ~62 s
  per event for both legs.
- **Jet legs take longer** because MUSIC_2 runs until the jet's droplets have frozen out,
  so its history has more frames (`ntau_jet` 112 vs `ntau_bg` 102 on average). They find
  about as many cells (800k vs 792k per leg), and time and cell count are only loosely
  related (correlation 0.5): the search volume matters, not the output.
- **What a leg contains:** reading the leg from the pair file, $T(e)$ and $P(e)$, loading
  it into `FluidDynamics`, the finder, and writing the cells. Only the leg as a whole is
  timed. The finder is most of it: on the untrimmed 143 frames it alone took ~38 s (all
  threads; 102 s with 5), and dropping the all-zero frames after freeze-out brought it
  down to the 31 s above.
- **Rebuilding `hist`** therefore takes ~1 h 48 min with its hadronization (+~4 min). The
  `ref_*` variants take ~6 min each, since they need no surface search.

## What `make_surfaces.py` does

All variants copy the reference particlize file (final partons, `events/`, `music_input`,
provenance) and replace only `surface/jet` and `surface/bg`. `hadronize.py` and
`HadronFileReader` then read them unchanged.

- **`ref_*`**: the reference's cells, with the $\pi^{\mu\nu}$ and/or $\Pi$ columns set to 0.
- **`hist`**, for the jet leg of every event (`arr[e]`) and every new background
  (`arr_bg[bg_id]`):
  1. $T(e)$ and $P(e)$ from MUSIC's EoS 9 table, which is linear in $e$ as MUSIC reads it.
     $e(T_{sw}=0.15) = 0.2342$ GeV/fm³ reproduces the reference surface's $e$ = 0.2341.
  2. The history is stored in a `FluidDynamics` (`temperature, energy_density, pressure, vx,
     vy, vz`). The frames after the leg's freeze-out are zero, and all but the first of them
     are dropped. Then `find_freezeout_surface(T_sw, dtau, dx, deta)` runs X-SCAPE's
     Cornelius `SurfaceFinder` on the lattice `--lattice` (default: the output grid spacing,
     0.1, 0.3125, 0.3125). This is exactly what `SoftParticlization` does for a hydro that
     hands over no surface.
  3. Per cell, $e$ and $P$ are recomputed from the interpolated $T$ through the EoS. The
     finder interpolates $e$ and $T$ separately, which leaves $e$ up to 60% off the isotherm.
     The charges and chemical potentials (left uninitialised by the finder, ~1e-43) and
     $\pi^{\mu\nu}$, $\Pi$ are set to 0.
  4. **Low-temperature cap** (`--no-cap` to switch off): MUSIC freezes out every cell with
     0.05 GeV/fm³ < e < e(T_sw) on a τ = const element at its first freeze-out step
     (`Do_FreezeOut_lowtemp`, `evolve.cpp: FreezeOut_equal_tau_Surface_XY`). That is about
     0.6% of the reference's volume. `hist` adds the same elements at the history's first
     frame (τ = 0.5 fm/c; MUSIC's is τ₀ + 2Δτ ≈ 0.46).
  5. `events/hist_*_{jet,bg}` record, per event: the frames used, the cap cells, and the
     largest $T$ in the first and last frames and on the transverse and η edges of the grid.
     An edge above $T_{sw}$ means an open surface there.
- **Seeds.** The outputs keep the reference's `file_uuid`, so `hadronize.py` gives every unit
  the reference's iSS seed. With `--correlated`, iSS addresses its random numbers by the
  space-time cell block, so a variant and the reference give the same hadrons wherever
  their surfaces agree. Variant − reference is then far less noisy than either. The
  jet fragments (`jet_frag`) are bit-identical to the reference's, which checks this. To
  sample independently, pass `--new-uuid`.

Checks that were made on the test data:
- Surface geometry and volume: see the notebook, §1.
- The $d^3\sigma_\mu$ convention: both surfaces are un-weighted by τ, with $u^3 = \tau u^\eta$, and
  $V = \tau(d\sigma_0 u^0 + d\sigma_1 u^1 + d\sigma_2 u^2) + d\sigma_3 u^3$ agrees between them.
- The finder at the grid spacing gives the exact area on a synthetic flat surface
  (`test_hist_surface.py`).

**Known limits of `hist`,** which are properties of the stored history, not of the method:
- The output grid ends at $\lvert\eta_s\rvert$ = 5, where the medium is still above $T_{sw}$
  (`hist_T_max_eta_edge` 0.16–0.19 GeV in the test data). The surface is therefore open there, and
  forward hadrons are missing. Compare at mid-rapidity only.
- There are no viscous fields. $\pi^{\mu\nu}$ and $\Pi$ could be estimated from the stored
  velocity gradients (first-order Navier–Stokes, with MUSIC's η/s(T) and ζ/s(T)), or stored
  on the surface cells of the history. Neither is done here (see *Next steps*).
- The transverse grid is ±10 fm. The reference surface stays inside ±7.9 fm for these
  events (`hist_T_max_xy_edge` < T_sw).

## `summary.npz`

`analyze.py --help` lists every key. In short:
- `had/<variant>/<leg>/<name>`, where `leg` is `jet` (bulk_jet) or `bg` (the event's
  bulk_bg). Axis 0 is the event; each value is the mean over the event's oversamples:
  - charged `dndeta`;
  - `dndy` and `sumpt` for π±, K±, p, p̄ at |y| < 0.5;
  - identified and charged `ptspec`;
  - flow vectors `Q`, `NQ` (charged, |η| < 1, per pT bin and integrated; $v_n\{2\}$ with the
    self-pairs removed is built in the notebook);
  - `E_eta1`, `N_all`, `E_all`;
  - jet-relative `dphi_jet`, `dphi_jet_pt`, `pt_near` (soft charged, pT < 4 GeV,
    |η − y_jet| < 1; the jet axis is the hardest shower initiator).
- `surf/{ref,hist}/<leg>/<name>`:
  - cells, volume, and its distributions in τ, η_s (each cell spread over its η width) and $u_T$;
  - the volume-weighted mean $\Pi$ and $\sqrt{\pi:\pi}$ of the reference.
- `jet/phi`, `jet/y`, `event/file`, `event/local`, `edges/*`, `meta` (JSON: files, settings
  per variant, hadronization attributes).

## Findings (test data: 4 files × 25 events, 0–10% Au+Au 200 GeV, PythiaGun 50–70 GeV)

All shifts are event by event relative to `ref`, for the background leg at mid-rapidity
(the jet leg is the same within errors). The errors are statistical, over 100 events.

**Surfaces agree to 2%.** The surface found in the history has 70% of MUSIC's cells (a
coarser lattice) and 0.981 of its freeze-out volume, in every event (0.980–0.982). Its
position and timing match: dV/dτ agrees within 2–3% from τ ≈ 1.5 to 9 fm/c. It has less
of the fastest fluid, though: dV/du_T falls below the reference above u_T ≈ 0.7 and is
20% low at u_T = 0.9. MUSIC's surface carries a bulk pressure ⟨Π⟩/(e+P) = −0.046 and a
shear stress ⟨√(π:π)⟩/(e+P) = 0.028, which `hist` does not have.

**Bulk hadrons: the missing viscous corrections dominate, not the coarse history.**

| observable | shear δf off | bulk δf off | both off (`ref_ideal`) | history on top (`hist` − `ref_ideal`) | **total (`hist` − `ref`)** |
|---|---|---|---|---|---|
| dN_ch/dη, \|η\|<0.5 | −0.2% | −0.2% | −0.4% | +0.5% | **+0.2%** |
| dN/dy p + p̄ | 0.0% | −29.3% | −29.3% | +1.0% | **−28.6%** |
| ⟨pT⟩ π | −1.9% | +15.7% | +13.5% | −1.2% | **+12.1%** |
| ⟨pT⟩ K | −2.2% | +16.0% | +13.4% | −1.6% | **+11.6%** |
| ⟨pT⟩ p | −1.6% | +10.2% | +7.6% | −1.8% | **+5.6%** |
| v₂{2} | +10.3% | −5.5% | +6.5% | −3.0% | **+3.3%** |
| v₃{2} | +24.6% | −8.9% | +21.4% | −4.9% | **+15.4%** |

- The statistical errors are 0.01–0.3% for yields and ⟨pT⟩, and 0.5–2.4% for v_n.
- **Bulk δf** sets the yields and the spectral shape. Without it, protons drop by 29%
  and the spectra harden (⟨pT⟩ +10–16%).
- **Shear δf** sets the flow harmonics. Without it, v₂, v₃ and v₄ rise by 10%, 25% and
  67%, increasingly with pT.
- **The coarse history** adds only 0.5–2% to yields and ⟨pT⟩, and 3–5% to v_n (lower,
  consistent with the missing fast fluid). It partly offsets the viscous shifts.
- The total hadron energy is not the same with and without δf. Over all rapidities it is
  +3.9% for `ref_ideal` over `ref`, and +4.9% at |η| < 1: iSS's viscous corrections, as
  applied here, do not conserve energy.
- Over all rapidities `hist` has 6% less energy than `ref_ideal`. That is the forward
  region beyond the history's |η_s| = 5; at |η| < 1 the difference is −1.1%.

**Jet wake (jet − background): δf cancels, the history loses about 10%.**
- ΔE (|η| < 1) of the soft bulk:
  - `ref`: 24.0 ± 1.2 GeV;
  - `ref_ideal`: +0.14 ± 0.21 GeV relative to `ref`. The viscous corrections do not matter
    for the wake at this precision.
  - `hist`: 21.5 ± 1.1 GeV, i.e. −2.47 ± 0.33 GeV (−10%) relative to `ref`.
- On the near side (|Δφ| < 1, soft charged) `hist` has −12 ± 2% fewer hadrons and
  −10 ± 2% less pT; the away-side deficit is smaller. Over all rapidities the wake
  energy is −4%.
- The deficit is a **scale, not a distortion**. Per event, hist's wake is highly
  correlated with ref's (r = 0.97), with a median ratio of 0.90, and the Δφ shape is the
  same.
- It is already in the surfaces: the jet-induced freeze-out volume is 61.6 vs
  65.0 fm³ (−5%).
- Likely cause: the wake is small and fast, so it suffers most from the resampling onto
  the coarser output grid (0.2 → 0.3125 in η_s) and from the reduced flow at the fireball
  edge. This is not pinned down (see *Next steps*).

**What this means for hadronizing an FNO history.**
- dN/dη and the wake's shape can be taken from the history as it is. The wake's size comes
  out about 10% low.
- Spectra, identified yields, ⟨pT⟩ and v_n cannot. The error there is 5–30%, and almost
  all of it is the missing δf, bulk for yields and ⟨pT⟩, shear for v_n.

The figures, all tables and the per-event checks are in `hydro_hist_vs_surface.ipynb`.

## Next steps

- **Restore δf from the history.** Estimate $\Pi = -\zeta\theta$ and
  $\pi^{\mu\nu} = 2\eta\sigma^{\mu\nu}$ (first-order Navier–Stokes, MUSIC's η/s(T) and
  ζ/s(T)) from the stored velocity gradients, and add them to the `hist` cells as a
  `hist_ns` variant. It would show whether the 5–30% can be recovered without storing
  anything new.
- **Or store them.** Bulk δf alone accounts for most of the yield and ⟨pT⟩ error, so
  one more channel, Π, in the pair files would be the cheapest fix. Shear, for v_n, needs
  the five independent π^{μν} components.
- **Pin down the wake deficit.** Write the pair files on MUSIC's native grid
  (`run_prod_jet.py --native`) and run `hist` on them. That separates the resampling onto
  the output grid from the surface finding itself. `--lattice` checks the Cornelius
  lattice on its own.
- A wider η grid (|η_s| ≤ 6) would close the surface at the ends and recover the forward
  hadrons.
