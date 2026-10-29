# Au+Au jets in two stages

`prod_AuAu_0_10_jet` is the main production: 0–10 % Au+Au collisions at 200 GeV (3D
MC-Glauber strings, MUSIC on the GPU) with a hard jet whose energy loss feeds back into the
medium. Each event is written as a **pair** of hydro evolutions, the background and the same
event with the jet, and, for hadron-level work, as the freeze-out surfaces and partons that
a second stage turns into hadrons on CPUs. Use it for the medium response to jets (the wake)
in hydro and in hadrons, for jet observables with a realistic background, and as training
data for the ML hydro surrogates. This page goes from a single job to a campaign; the first
two events are in [Your first event](first-event.md#2-two-auau-jet-events).

## How an event is made

```text
IS (3dMCGlauber) -> Hard (PythiaGun) -> NullPreDynamics
   -> MUSIC_1                    background leg               -> arr_bg
   -> Liquefier + Eloss          Matter + LBT on MUSIC_1's medium -> droplets
   -> MUSIC_2                    the same strings + the droplets  -> arr
```

**Why pairs.** Both legs start from the same initial condition and are bit-identical until
the first droplet deposits, so `arr - arr_bg` is the jet's effect on the medium and nothing
else: no event-by-event fluctuation of the bulk to subtract statistically. The same holds at
hadron level: iSS on the jet leg's surface minus iSS on the background's.

The background leg doesn't depend on the jet. So a background can be **shared** by several
jets (`--reuse N`, or several pT̂ windows with `--pthat-bins`): MUSIC_1 runs once per
background, MUSIC_2 once per jet. MUSIC_2 also starts from MUSIC_1's saved state just before
the first droplet can deposit (τ ≈ 1 fm/c), with bit-identical output; this is on by default.
The physics settings (EOS, liquefier, Matter and LBT) are discussed in
[The physics](../about/physics.md).

## 1. Checks before a campaign

The driver `run_prod_jet.py` runs one job: one seed, N events. Run these from the production
folder (in the container: `/opt/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape/example/prod_AuAu_0_10_jet`,
Docker's start directory; on a Mac: your `xscape-examples` copy), with the output under a
directory you can write:

```bash
# 1. the XML and the grid, checked; the job XML written, no event
python run_prod_jet.py --events 1 --seed 1 --dry-run --outdir out_val

# 2. null test: MUSIC_2 without the liquefier must reproduce the background bit for bit
python run_prod_jet.py --events 1 --seed 901 --write-particlize both --no-deposit --outdir out_val

# 3. validation: the hadrons are also made in the job, to compare with stage 2
python run_prod_jet.py --events 2 --seed 900 --write-particlize both --validate-inline --outdir out_val
```

- **Null test.** `arr == arr_bg` in every frame (`diag/frames_identical == ntau` in the
  pair file), and the two stored surfaces are identical:

  ```bash
  python -c "import numpy as np; from jetscape.particlize_h5 import ParticlizeFile as P; \
  p = P('out_val/AuAu_0_10_jet_seed0901_particlize.h5'); \
  print(np.array_equal(p.surface('jet', 0), p.surface('bg', 0)))"      # True
  ```

- **Validation.** `hadronize.py --use-stored-seeds` on the validation file must reproduce the
  in-job hadrons bit for bit (run it without `--keep-bits-*` or `--eta-max`). The script is in
  *Checks before a campaign* of the
  [folder's README](https://github.com/jhputschke/js-contrib/blob/main/contribs/PyJetscape/example/prod_AuAu_0_10_jet/README.md#checks-before-a-campaign).
  `--validate-inline` changes the jet sample of a seed (it adds tasks, which shifts the energy
  loss's random streams): use it for validation jobs only.
- **One event with a jet:** `diag/n_droplets > 0`, and the legs differ only after the first
  deposit.

With pT̂ windows, also run the null test with the windows
(`--pthat-bins … --no-deposit`). For hardware checks (GPU found, settings arrived) see
[Checking a run](clusters.md#checking-a-run).

## 2. Stage 1: hydro and jets on the GPU

### `run_jobs.sh`

```text
./run_jobs.sh [-j P] [--mps] [--stagger S] [--campaign NAME] NJOBS EVENTS_PER_JOB FIRST_SEED [OUTDIR] [run_prod_jet.py options]
```

| argument | |
|---|---|
| `NJOBS` | number of jobs = number of output files; total events = NJOBS × EVENTS_PER_JOB |
| `-j P` | at most P jobs at once (default 1); it doesn't create jobs |
| `FIRST_SEED` | `0`: a new seed per job, never repeated (a campaign, the usual choice); `> 0`: seeds FIRST_SEED, FIRST_SEED+1, … |
| `--campaign NAME` | names the files `AuAu_0_10_jet_NAME_0001.h5`, …; without it, the start time |
| `OUTDIR` | the output directory: the argument after FIRST_SEED (default `./out`), **not** `--outdir`. One campaign per OUTDIR |
| after OUTDIR | options for every job, e.g. `--write-particlize both`, `--pthat-bins …` |

- **Restarting:** run the same command again. Jobs whose `.json` says complete are skipped,
  unfinished ones are redone (in a seed-0 campaign with a new seed). The skip test looks only
  at file names and event counts: give every setting its own OUTDIR.
- **Seed registry:** `seeds_used.tsv` in OUTDIR's parent, shared by the campaigns side by
  side there, under a lock.
- **A seed is a set of collisions.** The same seed gives the same background in every
  campaign, whatever the jet settings. That is what paired comparisons want, and wrong for
  campaigns meant as more statistics: use `FIRST_SEED 0` for those.
- `--mps` (CUDA only) lets the P jobs share the GPU through CUDA MPS; `--stagger S` starts
  the first P jobs S seconds apart. Both are discussed in [Clusters](clusters.md#launch-settings).
- At the end, `OUTDIR/run_jobs.finished` is written; each job leaves `<stem>.log`,
  `<stem>.json` and `<stem>.xml`.

### What to store

| option | stores | use |
|---|---|---|
| (default) | the hydro pair `<stem>.h5` | FNO training data, hydro-level wake |
| `--write-particlize both` | + `<stem>_particlize.h5`: both freeze-out surfaces (full viscous information), the final partons, the shower initiators | hadron level: jet − background. **Add it to every job that should get hadrons later** |
| `--write-particlize jet` | + the jet leg's surface only | the jet event's hadrons without background subtraction (turns the MUSIC_2 restart off) |
| `--write-particlize both --particlize-only` | only the particlize file, no pair file | hadrons without the hydro history: ~2.5× less disk, ~3 s per event faster |
| `--hadrons-only` | only hadron files, made in the job; no stage 2 | little or no oversampling ([below](#hadrons-in-the-job-hadrons-only)) |

The pair file is unchanged, byte for byte, by `--write-particlize`.

### pT̂ windows and shared backgrounds

A background costs about as much as a jet leg. For a real campaign, share each background
among several jets in several pT̂ windows:

```bash
... --pthat-bins 10-20,20-30,30-40 --jets-per-bin 5
```

- **How.** One Pythia per window, each with its own seed and cross section; event *i* uses
  window *i* mod K. `--reuse` is set to K × M (here 3 × 5 = 15), so one background serves 15
  events, 5 in each window.
- **EVENTS_PER_JOB must be a multiple of K × M** (15, 30, 45, …).
- **Cost.** An event costs about (1 + 1/(K·M)) / 2 of an event with its own background:
  0.53 for K·M = 15.
- **Cross sections.** Each window records its σ; combine windows with those weights
  ([Hadrons](../analysis/hadrons.md)). Results stitched over windows have errors correlated
  through the shared backgrounds; per-window results and jet − background don't.
- `--reuse N` alone shares a background among N jets of one window. `--pthat-min/--pthat-max`
  set a single window (default 50–70 GeV); `--hard pgun --pgun-pt P` shoots one parton.

### The parton rapidity cut

At low pT̂ the jets spread far in rapidity, and much hydro time would go to events that cannot
reach a mid-rapidity measurement. `--parton-ymax Y` keeps only events whose shower-initiating
partons pass a rapidity cut; rejected events are regenerated in Pythia before any shower or
hydro, and the cross sections are corrected by the kept/tried fraction.

| `--parton-y-mode` | keeps the event if | for |
|---|---|---|
| `leading` (default) | the hardest parton has \|y\| < Y | the hardest jet in the acceptance |
| `both` | the two hardest have \|y\| < Y | dijets |
| `any` | either of the two hardest has \|y\| < Y | inclusive jets, the medium response |

The cut only selects events; the analysis applies its own. So Y must be looser than the
analysis acceptance: `--parton-ymax 0.6` (leading) is complete for jets up to
\|η_jet\| ≈ 0.4–0.5; for inclusive R = 0.4 jets at \|η_jet\| < 0.6 use
`--parton-ymax 0.7 --parton-y-mode any`. For wake hadrons at \|η\| < η_max, about
Y ≈ η_max + 1 with `any`.

### Defaults worth knowing

- **MUSIC's late time step.** From τ = 3 fm/c MUSIC steps 0.05 fm/c instead of 0.02. The
  stored frames stay at 0.1 fm/c; the medium changes at 1e-3 and the hadron level by less
  than ±2 %. **Keep one setting per campaign**: `--late-dtau 0` gives a constant step.
- **MUSIC_2 from MUSIC_1's state.** Bit-identical output, ~25 core-s less per jet event;
  `--no-music2-restart` turns it off. Every job log says `restart` at the start, on or off.
- **EOS 91** (hotQCD with the SMASH hadron list), which iSS decays itself; `--eos 9` only to
  rerun older campaigns.
- **The output grid** of the pair file is `example/config/grid_fno.yaml` (64 × 64 × 32 cells,
  x, y ±12.2 fm, η_s ±4.84); `--grid FILE` for another. Never mix grids in one training set.

Every option of `run_prod_jet.py`: the *Options* table of the
[folder's README](https://github.com/jhputschke/js-contrib/blob/main/contribs/PyJetscape/example/prod_AuAu_0_10_jet/README.md#options).

## 3. Stage 2: hadronization on CPUs

`hadronize.py` turns one particlize file into hadrons: iSS samples both stored surfaces
(with viscous corrections), Pythia fragments the stored partons. It needs no GPU and no MUSIC.
`run_hadronize.py` runs it over a whole campaign, `-j P` files at a time, and skips files whose
hadrons are complete:

=== "Docker"

    ```bash
    IMG=jhputschke/xscape-prod:cu126          # the image of stage 1
    docker run --rm --user "$(id -u):$(id -g)" -e OMP_NUM_THREADS=2 \
      -v "$HOME/prod:/work" $IMG \
      ./run_hadronize.py /work/AuAu_a -j 8 --oversample 500 --n-frag 50
    ```

=== "Apptainer"

    ```bash
    PROD=/opt/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape/example/prod_AuAu_0_10_jet
    apptainer exec --env OMP_NUM_THREADS=2 --pwd "$PROD" --bind "$SCRATCH/prod:/work" \
      xscape_prod.sif ./run_hadronize.py /work/AuAu_a -j 8 --oversample 500 --n-frag 50
    ```

=== "Mac / source build"

    ```bash
    python run_hadronize.py out_had -j 8 --oversample 500 --n-frag 50
    ```

Inputs are directories, particlize files or quoted globs. Per production file it writes
`<stem>_hadrons_bulk_jet.h5`, `<stem>_hadrons_bulk_bg.h5` and `<stem>_hadrons_jet_frag.h5`
next to it.

| option | |
|---|---|
| `-j P` | processes at once, ~1.4–1.7 GB and one core each |
| `--oversample N` | iSS samples per jet-leg surface (default 100) |
| `--n-frag K` | Pythia fragmentations of the partons per event |
| `--oversample-bg auto` | a background shared by N events gets N × the samples: the optimal split for jet − background under `--reuse` |
| `--oversample-bg per-pthat-bin` | for `--pthat-bins` campaigns analysed per window: M × the samples (M = `--jets-per-bin`) |
| `--correlated` | correlated jet/background sampling: jet − background ~8× less noisy at \|η\| < 1 for the same samples (not together with `--oversample-bg`) |
| `--keep-bits-p B`, `--keep-bits-x B` | round momenta and positions to B mantissa bits: `12` and `8` store 58 % of the bytes with no visible effect on ensemble observables |
| `--eta-max X`, `--charged`, `--no-x` | store only \|η\| < X (2: ~60 % of the bytes), charged hadrons, no positions |
| `--tags` | a subset of `bulk_jet,bulk_bg,jet_frag` |
| `--follow` | keep running next to a stage 1 that is still writing; stop when its campaign is finished |
| `--dry-run` | show the plan only |
| `--force` | redo complete files (e.g. with other settings) |

!!! warning "Decide the hadron settings once per campaign"
    The precision (`--keep-bits-*`), the selection (`--eta-max`, `--charged`, `--no-x`) and
    the background oversampling must be the same for every file: pass the same flags to every
    call, `--follow` and restarts included. Complete files are skipped whatever settings made
    them. Nothing is lost for good: the particlize file and the stored seeds reproduce the
    exact hadrons with `--force`. Keep the validation job at full precision and without cuts.

**Alongside production.** Start `run_hadronize.py --follow` next to `run_jobs.sh`; it
hadronizes each file as its job completes and stops when the campaign writes
`run_jobs.finished`. On a machine whose cores are the limit, its cores come out of stage 1's
rate, so on clusters stage 2 usually runs separately on CPU nodes
([SLURM](clusters.md#stage-2-on-cpu-nodes)). Only the particlize files need to travel there:
they are self-contained.

### Hadrons in the job (`--hadrons-only`)

For campaigns with little or no oversampling (1–10 samples per jet event), the hadrons can be
made in the GPU job itself: `run_prod_jet.py --hadrons-only` writes only the three hadron
files, the undecayed bulk hadrons (`*_primary.h5`, for SMASH later; `--keep-primaries 0`
turns them off) and the `.json`. No pair or particlize file, no stage 2:

```bash
./run_jobs.sh -j 8 --mps 60 45 0 /work/AuAu_ho --reuse 4 --hadrons-only
```

Its options are `hadronize.py`'s (`--oversample`, default 1; `--n-frag`, default 1;
`--correlated`; `--eta-max`, `--charged`, `--no-x`; the rounding is
`--hadron-keep-bits-p/-x`). The hadrons equal those of the two-stage path for the same seed,
bit for bit. What is lost: no surfaces (no new samples later) and no hydro history. On the
GB10 it cost ~4 % in events/h against `--particlize-only` and wrote 0.64 MB per event, against
~160 MB in two stages.

## 4. A realistic campaign

The campaign setting used for the production runs: three pT̂ windows, 5 jets per window per
background, the leading parton at \|y\| < 0.6, everything stored for hadronization.

=== "Docker"

    ```bash
    IMG=jhputschke/xscape-prod:cu126          # for a campaign: a dated tag or the digest
    mkdir -p ~/prod/lbt_cache
    docker run -d --name c1 --gpus all --user "$(id -u):$(id -g)" \
      -e OMP_NUM_THREADS=3 -e OMP_WAIT_POLICY=passive -e MUSIC_CUDA_SYNC=block \
      -e LBT_TABLE_CACHE=/work/lbt_cache/lbt_tables_v1.bin -v "$HOME/prod:/work" $IMG \
      ./run_jobs.sh -j 6 --mps --stagger 12 --campaign pth10-40_eta06_c1 67 15 0 \
        /work/AuAu_0_10_pth10-40_eta06_c1 \
        --pthat-bins 10-20,20-30,30-40 --jets-per-bin 5 --parton-ymax 0.6 --write-particlize both
    docker logs -f c1             # progress;  docker stop c1 stops it cleanly
    ```

    `-j 6` with 3 threads each suits one GPU with ~24 cores and 64 GB; size `-j` to your
    machine (~2–3 cores and ~7 GB per job).

=== "Apptainer"

    ```bash
    PROD=/opt/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape/example/prod_AuAu_0_10_jet
    mkdir -p $SCRATCH/prod/lbt_cache
    apptainer exec --nv \
      --env OMP_NUM_THREADS=3,OMP_WAIT_POLICY=passive,MUSIC_CUDA_SYNC=block \
      --env LBT_TABLE_CACHE=/work/lbt_cache/lbt_tables_v1.bin \
      --pwd "$PROD" --bind "$SCRATCH/prod:/work" xscape_prod.sif \
      ./run_jobs.sh -j 6 --stagger 12 --campaign pth10-40_eta06_c1 67 15 0 \
        /work/AuAu_0_10_pth10-40_eta06_c1 \
        --pthat-bins 10-20,20-30,30-40 --jets-per-bin 5 --parton-ymax 0.6 --write-particlize both
    ```

    On a SLURM cluster use the array script instead: [Clusters](clusters.md#slurm-one-gpu-per-array-task).

=== "Mac"

    ```bash
    cd ~/xscape/examples/prod_AuAu_0_10_jet
    OMP_NUM_THREADS=5 OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0 \
      caffeinate -i ./run_jobs.sh -j 3 --campaign pth10-40_eta06_c1 67 15 0 ~/xscape/prod/AuAu_c1 \
        --pthat-bins 10-20,20-30,30-40 --jets-per-bin 5 --parton-ymax 0.6 --write-particlize both
    ```

    On a Mac, split the cores between the jobs (`-j` × threads ≈ the performance cores) and
    use `passive`, or several jobs gain nothing; `--mps` is CUDA only. `caffeinate -i` keeps
    the Mac awake.

That is 67 jobs × 15 events = 1,005 events on 67 backgrounds. Then stage 2, with the
background sampled per window and the hadrons rounded:

```bash
./run_hadronize.py /work/AuAu_0_10_pth10-40_eta06_c1 -j 16 \
  --oversample 400 --oversample-bg per-pthat-bin --n-frag 50 --keep-bits-p 12 --keep-bits-x 8
```

400 samples per jet event, 5 × 400 = 2,000 per background and window, 50 fragmentations.

**What to expect.** Measured with the campaign setting:

| | |
|---|---|
| 2 × RTX 3090 (24 cores), 10 jobs per GPU | **2,160 events/h** (1,080 per GPU), ~25 s per event in a loaded job |
| GB10, `-j 4 --mps` (earlier code) | 336 events/h |
| host memory per job | ~6–7 GB with the shared LBT tables; plan 7 GB (5 GB with `--particlize-only`) |
| disk, stage 1 | **0.243 GB per event** (pair + particlize); 0.097 GB with `--particlize-only` |
| stage 2 | ~340 events/h per core at 500 oversamples (10.6 s per event, both legs; 8.8 s with `--correlated`): ~6–7 cores keep up with 2,000 events/h |
| disk, hadrons | ~100 MB per leg and event at 500 oversamples, full precision; 58 % of that with `--keep-bits-p 12 --keep-bits-x 8`, and ~60 % of that again with `--eta-max 2` |

On a cluster, the sizing per GPU type and a 12,000-event SLURM example are in
[Clusters, GPUs and monitoring](clusters.md); the hardware numbers in
[GPU acceleration](../about/gpu.md).

!!! tip "Pin the image, one per campaign"
    The moving tags (`cu126`) change with every build, and different builds can differ in
    the last digits or more. For a campaign use a dated tag (`cu126-<YYYYMMDD>-<commit>`) or
    the digest, the same one on every machine, and note it: the files don't record which
    image made them. Run the whole campaign on GPUs: MUSIC's CPU path differs systematically
    (~0.2 % in multiplicity).

## After the campaign

- **Watch and check it:** `utils/jobmon.py OUTDIR --watch`, and the deposition check
  `utils/check_deposition.py OUTDIR`, which flags the rare job whose jets deposited far
  less than the rest: [Clusters, GPUs and monitoring](clusters.md#monitoring-a-campaign).
- **Move the files:** `--what h5` (particlize + hadron files) for the analysis,
  `--what pair` for FNO training: [Data on the OSDF](osdf.md).
- **Analyse:** [Hadrons](../analysis/hadrons.md) (`HadronFileReader`, cross-section weights,
  jet − background), [Jets with FastJet](../analysis/jets.md),
  [ROOT and C++](../analysis/root.md) (`run_h5toROOT.py`).

Full details, every option and the measurements:
[`prod_AuAu_0_10_jet/README.md`](https://github.com/jhputschke/js-contrib/blob/main/contribs/PyJetscape/example/prod_AuAu_0_10_jet/README.md)
and [`docs/README_2stage.md`](https://github.com/jhputschke/js-contrib/blob/main/docs/README_2stage.md).
