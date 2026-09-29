# prod_AuAu_0_10_jet — background/jet hydro pairs for FNO4d, straight to HDF5

This folder is [`../prod_AuAu_0_10`](../prod_AuAu_0_10/README.md) with a jet. It runs the
same 0–10% Au+Au 200 GeV medium (3D MC-Glauber strings, MUSIC) through the X-SCAPE
**two-stage hydro** and writes each event as a **pair** in FastHydro's layout:

```
IS (3dMCGlauber) -> Hard (PythiaGun | PGun) -> NullPreDynamics
   -> MUSIC_1                    background leg  -> arr_bg
   -> Liquefier + Eloss          Matter + LBT on MUSIC_1's medium -> droplets
   -> MUSIC_2                    same strings + the droplets      -> arr
```

`arr - arr_bg` is the jet's effect on the medium and nothing else: both legs start from the
same initial condition, and before the first droplet deposits they are bit-identical.

> **Status (2026-09-24).** Runs on `build_gpu` (music4gpu, CUDA, GB10). It needs a MUSIC
> build with the jet source slot:
> - CPU: MUSIC `cee9460`, via X-SCAPE PR #138.
> - GPU: MUSIC4GPU `XSCAPE` from `3037be7` on.
>
> The `<freeze_out_surface>` setting in the XML needs X-SCAPE branch
> `pair_h5_music_surface_off` and MUSIC4GPU branch `XSCAPE_surface_off`.
>
> Without the slot, MUSIC_2 silently ignores the droplets, and the writer then warns that the
> jet leg is identical to the background. Plan and status: [`PLAN_pair_h5_music.md`](../../../../docs/PLAN_pair_h5_music.md).
>
> **Hadron level (2026-09-26, branches `surface_to_hadrons` in X-SCAPE and js-contrib).**
> `--write-particlize` stores both freeze-out surfaces and the final partons next to the pair;
> `hadronize.py` turns them into iSS and Colorless hadrons offline, bit-identical to running
> them inside the job. See [Hadron level](#hadron-level-surfaces-partons-hadronizepy) and
> [`PLAN_particlize_h5.md`](../../../../docs/PLAN_particlize_h5.md).
>
> **Several pT̂ windows per background (2026-09-27, branches `N_ptHat_per_hydro` in X-SCAPE
> and js-contrib).** `--pthat-bins 20-40,50-70,70-90` runs every background with one jet per
> window. See [D. Several pTHat windows per background](#d-several-pthat-windows-per-background---pthat-bins).

| file | purpose |
|---|---|
| `AuAu_MCGlauber_MUSIC_0_10_jet.xml` | user XML: the prod physics plus `Hard`, `Liquefier`, `Eloss` and a second `Hydro` (MUSIC_2) |
| `run_prod_jet.py` | one job: one seed, N events, one `.h5` file |
| `run_jobs.sh` | many jobs, `-j P` at a time, resumable (wraps `../prod_AuAu_0_10/run_jobs.sh`) |
| `hadronize.py` | offline: iSS on the stored surfaces, Colorless on the stored partons → hadron files |
| `run_hadronize.py` | `hadronize.py` over a whole campaign, `-j P` at a time; `--follow` runs it alongside `run_jobs.sh` |
| `paired_noise.py` | noise of jet − background per oversample, independent vs correlated legs (`--correlated`), and each leg's physics in both modes ([`PLAN_iSS_optim.md`](../../../../docs/PLAN_iSS_optim.md), Part B) |
| `hadronize.xml` | the iSS and jet-hadronization settings (used by `hadronize.py` and `--validate-inline`) |
| `jet_wake.ipynb` | one pair, from the energy density (§1–9) to hadrons (§10) |

Design notes and measurements are in [js-contrib `docs/`](../../../../docs/README.md): the hadron-level path
([`PLAN_particlize_h5.md`](../../../../docs/PLAN_particlize_h5.md)), faster iSS and correlated
jet/background sampling ([`PLAN_iSS_optim.md`](../../../../docs/PLAN_iSS_optim.md)), and the machine
settings ([`BENCHMARK_GB10.md`](../../../../docs/BENCHMARK_GB10.md),
[`BENCHMARK_M3MAX.md`](../../../../docs/BENCHMARK_M3MAX.md)).

The grid YAMLs (`../prod_AuAu_0_10/grid_fno.yaml` by default) and all grid and environment
checks are shared with the single-leg production. Pair files and single-leg files made with
the same YAML have the same spatial grid.

## Run

```bash
conda activate js_fno        # GB10; on macOS e.g. fno_env_mlx (see ../prod_AuAu_0_10/README.md)
cd external_packages/js-contrib/contribs/PyJetscape/example/prod_AuAu_0_10_jet

python run_prod_jet.py --events 1 --seed 1 --no-deposit      # null test first: arr == arr_bg
python run_prod_jet.py --events 10 --seed 1                  # PythiaGun, pTHat 50-70 GeV
python run_prod_jet.py --events 10 --seed 1 --pthat-min 20 --pthat-max 40
python run_prod_jet.py --events 10 --seed 1 --hard pgun --pgun-pt 60
python run_prod_jet.py --events 30 --seed 1 --reuse 3        # one background per 3 jets
python run_prod_jet.py --events 30 --seed 1 --pthat-bins 20-40,50-70,70-90
                                             # the same, one jet per background in each window
python run_prod_jet.py --events 1 --seed 1 --dry-run         # check the XML/grid only

# + the input for hadronization: both freeze-out surfaces and the final partons
python run_prod_jet.py --events 10 --seed 1 --write-particlize both
python hadronize.py out/AuAu_0_10_jet_seed0001_particlize.h5 --oversample 500 --n-frag 50
python run_hadronize.py out -j 4 --oversample 500 --n-frag 50    # every particlize file in out/
python run_hadronize.py out -j 4 --oversample 500 --n-frag 50 \
       --keep-bits-p 12 --keep-bits-x 8       # hadrons rounded: 58% of the disk (campaigns, see B)
python run_hadronize.py out -j 4 --oversample 500 --n-frag 50 \
       --eta-max 2                            # only hadrons at |η| < 2: ~60% of the disk (see B)
python run_hadronize.py out -j 4 --oversample 500 --n-frag 50 \
       --charged --no-x                       # charged hadrons, no positions: ~40% (see B)

./run_jobs.sh 20 25 0                                        # 20 jobs x 25 events, unique seeds
./run_jobs.sh 20 25 0 --campaign pth50                       # the same, files named ..._pth50_00NN
./run_jobs.sh -j 2 20 25 0 out_pgun --hard pgun
./run_jobs.sh -j 4 --mps 20 25 0                             # 4 at a time, GPU shared via CUDA MPS
./run_jobs.sh -j 4 --mps 20 25 0 out_had --write-particlize both   # + hadronization input

# GB10 (CUDA): 4 jobs sharing the GPU through MPS, the cores split between them (../../../../docs/BENCHMARK_GB10.md)
OMP_NUM_THREADS=5 ./run_jobs.sh -j 4 --mps 20 25 0
OMP_NUM_THREADS=5 ./run_jobs.sh -j 4 --mps 20 25 0 --campaign pth50     # named campaign
OMP_NUM_THREADS=5 ./run_jobs.sh -j 4 --mps 20 25 1                      # seeds 1..20

# macOS (Metal): split the cores between the jobs, or -j 3 gains nothing (../../../../docs/BENCHMARK_M3MAX.md)
OMP_NUM_THREADS=5 OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0 ./run_jobs.sh -j 3 20 25 0
OMP_NUM_THREADS=5 OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0 ./run_jobs.sh -j 3 20 25 0 --campaign pth50
OMP_NUM_THREADS=5 OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0 ./run_jobs.sh -j 3 20 25 1
```

**`FIRST_SEED`, the third number, decides the seeds and the file names:**

| command (after `./run_jobs.sh -j 4 --mps`) | seeds | files in `./out` |
|---|---|---|
| `20 25 0` | a new one per job, from OS entropy | `AuAu_0_10_jet_20260926-221530_0001.h5` … `_0020.h5` (the start time) |
| `20 25 0 --campaign pth50` | a new one per job, from OS entropy | `AuAu_0_10_jet_pth50_0001.h5` … `_0020.h5` |
| `20 25 1` | 1, 2, …, 20 | `AuAu_0_10_jet_seed0001.h5` … `_seed0020.h5` |
| `20 25 1 --campaign pth50` | 1, 2, …, 20 | `AuAu_0_10_jet_pth50_seed0001.h5` … `_seed0020.h5` |

- **`0` is for campaigns.** Every job gets a seed no other job has used: drawn from OS
  entropy (1…900,000,000) and checked against the registry `seeds_used.tsv` next to `out/`.
  So two campaigns never share a collision, whenever and wherever they run. The seed that ran
  is stored in the file (`prod_seed`) and its `.json`: `--seed <it>` reproduces the file.
- **`> 0` gives exactly those seeds:** the same number always means the same collisions. That
  is right for reproducing a file, for validation jobs, and for comparing settings on the same
  events (the same seeds with another pT̂ window give the same backgrounds). It is wrong for a
  second campaign meant to add statistics.
- **The name says where the seed came from.** `seedNNNN` is the seed itself: an explicit seed
  names the file, so the same name always means the same collisions. A plain `NNNN` is the job
  number, 1…NJOBS: the seed was drawn, and is in the file and its `.json`.
- **`--campaign NAME` only adds the name** (`<campaign>_NNNN` or `<campaign>_seedNNNN`). It
  can go before or after the numbers. Without it, a `0` campaign is named by its start time. The
  name is kept in `out/run_jobs.campaign`, so re-running the same command resumes the
  campaign; give each campaign its own `OUTDIR` (a second name in the same one is refused).
  See [Campaigns with `run_jobs.sh`](#campaigns-with-run_jobssh) for the details.

**Recommended campaign settings** (measured, hydro pairs only; machine-specific, so they are
not built into the scripts):

| machine | campaign | events/h | memory | one job alone | details |
|---|---|---|---|---|---|
| GB10 (CUDA, 20 cores, 121 GB) | `OMP_NUM_THREADS=5 ./run_jobs.sh -j 4 --mps ...` | ~190 | ~66 GB | defaults (all threads), 118 events/h | [BENCHMARK_GB10.md](../../../../docs/BENCHMARK_GB10.md) |
| Apple M3 Max (Metal, 16 cores, 64 GB) | `OMP_NUM_THREADS=5 OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0 ./run_jobs.sh -j 3 ...` | 213 | ~40 GB | defaults, 132 events/h | [BENCHMARK_M3MAX.md](../../../../docs/BENCHMARK_M3MAX.md) |

- **Several jobs at once:** set `OMP_NUM_THREADS` to about cores / jobs, otherwise the jobs'
  OpenMP threads oversubscribe the cores.
- **One job alone:** leave `OMP_NUM_THREADS` unset. On the GB10, fewer threads made a single
  job 24% slower.
- **`--mps` is CUDA only.** On the GB10 it takes four jobs from 159 to 173–179 events/h, and
  the thread split brings that to ~190.
- **With `--write-particlize`,** each job is slower (~35 s instead of 29.5 s per event alone
  on the GB10; 43.0 s before MUSIC4GPU `5058545` parallelized the surface finder) and needs
  +0.5 GB. The campaign throughput with it has not been measured.
- **On another machine,** re-measure as described in
  [Finding the settings on another machine](../../../../docs/BENCHMARK_GB10.md#finding-the-settings-on-another-machine).

Each job writes the following, next to each other. The stem is `AuAu_0_10_jet_seedNNNN` for
an explicit `--seed` (`AuAu_0_10_jet_<campaign>_seedNNNN` with `--campaign`), and
`AuAu_0_10_jet_<campaign>_NNNN`, NNNN the job number, for `--seed 0` (see the next section):
- `<stem>.h5`: the data.
- `<stem>_particlize.h5`: with `--write-particlize` only, the input for
  `hadronize.py` (see [Hadron level](#hadron-level-surfaces-partons-hadronizepy)).
- `.xml`: the exact job XML.
- `.json`: a summary.
- `.log`: only when the job is run through `run_jobs.sh`.

`hadronize.py` then writes `<stem>_hadrons_{bulk_jet,bulk_bg,jet_frag}.h5` next to the
particlize file. For many jobs at once, see the next section.

## Campaigns with `run_jobs.sh`

`run_jobs.sh` runs many `run_prod_jet.py` jobs, one seed and one output file per job:

```bash
./run_jobs.sh [-j P] [--mps] [--campaign NAME] NJOBS EVENTS_PER_JOB FIRST_SEED [OUTDIR] [run_prod_jet.py options ...]
```

- **`FIRST_SEED 0`: a campaign, the usual choice.** Every job draws its own seed from OS
  entropy (`run_prod_jet.py --seed 0`), in 1…900,000,000 and not yet in the seed registry.
  The files are `OUTDIR/AuAu_0_10_jet_<campaign>_NNNN.*`, NNNN = 1…NJOBS, `EVENTS_PER_JOB`
  events each (default `OUTDIR` is `./out`). `<campaign>` is `--campaign NAME`, else the start
  time (`20260926-2215`), and is kept in `OUTDIR/run_jobs.campaign`: re-running the command
  resumes it, and another `--campaign` in the same `OUTDIR` is refused.
- **`FIRST_SEED > 0`:** seeds `FIRST_SEED .. FIRST_SEED+NJOBS-1` as given, files named by the
  seed: `OUTDIR/AuAu_0_10_jet_seedNNNN.*`, or `AuAu_0_10_jet_<campaign>_seedNNNN.*` with
  `--campaign`. For validation jobs,
  reproducing files, and campaigns that are *meant* to share their collisions (below).
- **The seed that ran is recorded** in the job XML, the file (`prod_seed`,
  `prod_seed_source` = `os_entropy` / `explicit`, `prod_campaign`, `prod_index`) and the
  `.json`. `--seed <that seed>` reproduces the file bit for bit.
- **Seed registry.** Every job appends its seed to `seeds_used.tsv` next to `OUTDIR` (seed,
  source, campaign, file, host, date), under a lock, so jobs starting together never draw the
  same seed and campaigns kept side by side there never repeat one. An explicit seed already
  in it is reported (not refused). `--seed-registry PATH` shares one registry between data
  directories or machines (on a shared disk); `--seed-registry none` turns it off.
- **Not X-SCAPE's own seed 0.** Given to the framework directly, 0 lets each module seed
  itself: the framework's engine from the clock in nanoseconds (and in its one-engine-for-all
  mode), Pythia from `time(0)`, in seconds. Jobs started within the same second then get the
  same jets, and neither seed is recorded, so no job could be reproduced. `--seed 0` here
  resolves the seed before the framework starts. Seeds above 900,000,000 are refused: Pythia
  clamps them to 900,000,000, so every such job would get the same jets.
- `-j P` keeps P jobs running at once; `--mps` lets them share the GPU through CUDA MPS
  (CUDA only). Every job runs in its own working directory (`OUTDIR/work/<tag>`, removed when
  it succeeds), so the jobs can start together. Output is bit-identical per seed whatever
  `-j`.
- Everything after the numbers goes to every job unchanged (`--write-particlize`, `--reuse`,
  `--hard`, `--grid`, ...), except `--campaign`, which `run_jobs.sh` takes wherever it stands
  (`./run_jobs.sh -j 4 20 25 0 --campaign pth50` and `./run_jobs.sh --campaign pth50 -j 4 20 25 0`
  are the same). `--events`, `--seed`, `--index`, `--outdir` and `--out` are refused: the
  script sets them per job.
- Each job's output goes to `OUTDIR/<tag>.log`. A failed job is reported and the others
  continue.
- **End marker.** When a campaign ends (not on Ctrl-C), `OUTDIR/run_jobs.finished` is
  written: this is how `run_hadronize.py --follow` knows no more files are coming. A new
  campaign in the same `OUTDIR` removes it first.
- **Restarting.** A job whose `.json` says all events were written is skipped (with
  `--write-particlize`, its particlize file must be complete too). So an interrupted campaign
  resumes with the same command, and re-running it only redoes failed or missing jobs. In a
  campaign, a re-run job draws a new seed; its unfinished file is replaced.
- Give every distinct setting its own `OUTDIR`: the skip test looks only at the file name and
  the event count, not at the options.

> **A seed is a set of collisions, in every campaign.** The seed fixes the initial condition,
> Pythia and Matter/LBT. Two campaigns over the same seeds are therefore not independent:
> - **Same settings:** the same events, bit for bit. Merging them double-counts every event.
> - **Different jet settings** (pT̂ window, `--hard`, `--no-deposit`, liquefier or medium
>   parameters): the same backgrounds, with different jets on top. Seed 1 at pT̂ 20–40 GeV and
>   at 50–70 GeV gives a bit-identical background leg. Only a different task list changes the
>   stream: the hydro-only `../prod_AuAu_0_10` has other initial conditions at the same seed.
>
> That is useful for comparing settings, where the shared backgrounds cancel in the
> difference, and wrong for anything that treats the campaigns as more statistics: merged
> hadron or FNO training sets, or errors that assume independent events. Campaigns meant to
> be independent: `FIRST_SEED 0` (unique seeds, checked against the registry). Campaigns meant
> to be paired: the same explicit seeds. `hadronize.py` keys its seeds on the particlize file,
> so its samples of repeated events differ, but the fluid underneath is still the same.

### A. Hydro pairs only (FNO training data)

```bash
conda activate js_fno
./run_jobs.sh 1 1 1 out_null --no-deposit                  # first: null test, arr == arr_bg
OMP_NUM_THREADS=5 ./run_jobs.sh -j 4 --mps 20 25 0 out     # 20 jobs x 25 events, unique seeds
```

On the GB10, `OMP_NUM_THREADS=5 ... -j 4 --mps` gives about 190 events/h
([BENCHMARK_GB10.md](../../../../docs/BENCHMARK_GB10.md); on macOS see the Metal line under *Run*). No
freeze-out surface is built (`--surface none`, the default), which is the fastest setting.

### B. Hydro pairs + hadronization input

The same, plus `--write-particlize`. The pair files are unchanged, byte for byte, so they can
also serve as training data:

```bash
# first: one validation job, then one null-test job (see "Checks before a campaign")
python run_prod_jet.py --events 2 --seed 900 --write-particlize both --validate-inline --outdir out_val
python run_prod_jet.py --events 1 --seed 901 --write-particlize both --no-deposit --outdir out_val

# the campaign: both surfaces + final partons every event
OMP_NUM_THREADS=5 ./run_jobs.sh -j 4 --mps 20 25 0 out_had --write-particlize both

# one background per 3 jets: the background surface is stored once per background
OMP_NUM_THREADS=5 ./run_jobs.sh -j 4 --mps 20 30 0 out_had_reuse3 --write-particlize both --reuse 3
```

| `--write-particlize` | stores | use |
|---|---|---|
| `both` | jet and background surfaces, final partons | hadron-level wake (jet − background), the usual choice |
| `jet` | jet surface, final partons | when only the jet event's hadrons are wanted (no background subtraction) |
| `none` (default) | nothing extra | hydro only (A) |

Don't combine it with `--surface`: `--write-particlize` builds the surfaces it stores, and a
surface built but not stored only costs time (the job warns if you do). `--validate-inline`
is for validation jobs only: it changes the jet sample of a seed (see *Seeds*).

**Hadronizing the campaign: `run_hadronize.py`.** `hadronize.py` needs only the particlize
files and an X-SCAPE build with iSS: no GPU, no MUSIC. It runs on one core, one production
file at a time; `run_hadronize.py` runs it over a whole campaign, `-j P` at a time:

```bash
# after production (or on another machine): every complete *_particlize.h5 in out_had/
python run_hadronize.py out_had -j 8 --oversample 500 --n-frag 50

# alongside production: start it next to run_jobs.sh; it hadronizes each file as its job
# completes and stops when run_jobs.sh writes out_had/run_jobs.finished
OMP_NUM_THREADS=5 ./run_jobs.sh -j 4 --mps 20 25 0 out_had --write-particlize both &
python run_hadronize.py out_had -j 3 --follow --oversample 500 --n-frag 50

# a --reuse campaign: give each background N x the jet leg's oversamples (the optimal split)
python run_hadronize.py out_had_reuse3 -j 8 --oversample 200 --oversample-bg auto --n-frag 50

# a --pthat-bins campaign analysed per window: each background M x the jet leg's oversamples
python run_hadronize.py out_pth3 -j 8 --oversample 200 --oversample-bg per-pthat-bin --n-frag 50

# hadrons stored at reduced precision: 58% of the disk (decide before the campaign, see below)
python run_hadronize.py out_had -j 8 --oversample 500 --n-frag 50 --keep-bits-p 12 --keep-bits-x 8

# only the hadrons at |η| < 2: ~60% of the disk, on top of the rounding (see below)
python run_hadronize.py out_had -j 8 --oversample 500 --n-frag 50 --keep-bits-p 12 --keep-bits-x 8 \
       --eta-max 2

python run_hadronize.py out_had --dry-run --oversample 500    # what it would do
```

> **Decide on the hadron precision before a campaign starts.** At 500 oversamples the hadron
> files are ~200 MB per event, about a third of a campaign's disk.
> `--keep-bits-p 12 --keep-bits-x 8` cuts them to 58% with no visible effect on ensemble
> observables. It has to be the same for every file of the campaign: pass the same flags to
> every `run_hadronize.py` / `hadronize.py` call, including `--follow` and restarts. See
> [Hadron precision](#hadron-precision---keep-bits-p---keep-bits-x).

- **Inputs** are directories, particlize files or globs. Every option it doesn't know goes to
  each `hadronize.py` unchanged. These are checked once before anything starts.
- **Only complete inputs.** A particlize file being written by a running job is not touched.
  Without `--follow` it is reported as incomplete; with `--follow` it is picked up once its
  job has finished.
- **Restarting.** `--skip-complete` is passed unless you give `--force`. Files whose outputs
  are all complete are skipped without starting a process; missing or incomplete outputs are
  redone. Re-running the same command finishes an interrupted pass, and on a finished
  campaign it returns at once. The skip test does not compare settings (`--oversample`,
  `--n-frag`, `--seed`, ...): to redo files with other settings, use `--force` or a new
  `--out-dir`. A complete file of another precision is kept, with a warning.
- **Ctrl-C** stops the running `hadronize.py` processes. They close their outputs as
  incomplete once their current iSS pass is done (up to ~90 s; a second Ctrl-C kills them),
  and the next run redoes those files.
- **`--follow`** stops when every input directory has `run_jobs.finished` and nothing is
  left, or after `--idle-exit` minutes (default 30) without new work. `--poll` (default
  30 s) sets how often it looks for new files.
- **Logs and exit code.** Each file's log is appended to `<stem>_hadronize.log` next to its
  outputs. A summary at the end lists hadronized, already complete, failed and incomplete
  files; the exit code is 1 if any `hadronize.py` failed.
- **Memory.** Each process needs ~1.6 GB, and beyond ~1500 iSS oversamples of its largest
  surface ~0.3 MB more per oversample. A warning is printed if `-j` processes would not fit
  into the available memory.
- **Cores.** Next to four GPU jobs on the GB10, `-j 3` to `-j 4` keeps up; alone, up to one
  process per free core.

`hadronize.py` options worth knowing (all pass through `run_hadronize.py`):

| option | effect |
|---|---|
| `--oversample N` | iSS samples per jet-leg surface (default: `hadronize.xml`'s 100); also per background unless: |
| `--oversample-bg M` | iSS samples per background surface |
| `--oversample-bg auto` | per background: N × the number of events using it (`events/bg_unit`), capped at `--oversample-bg-max` (default 2000, ~1.7 GB) with a warning. Under `--reuse N` this minimizes the error of jet − background for the CPU spent: a reused background's noise averages down over N times fewer backgrounds. The count per background is in `units/n_samples` of the `bulk_bg` file |
| `--oversample-bg per-pthat-bin` | `--pthat-bins` files: per background, N × the number of events using it in **one** pT̂ window (`--jets-per-bin`), capped like `auto`. The optimal split for results per window. Refused for files without `events/pthat_bin`. See [Hadronizing a `--pthat-bins` campaign](#hadronizing-a---pthat-bins-campaign) |
| `--n-frag K` | Colorless fragmentations per event. With `K` equal to `N` every oversample gets its own fragmentation (`JetEvents.jet_event`) |
| `--tags` | a subset of `bulk_jet,bulk_bg,jet_frag`, e.g. `--tags jet_frag` to redo only the fragments with other settings |
| `--seed` | base seed; every unit's seed derives from it and the production file (`file_uuid`), and is stored in `units/seed` |
| `--legacy-seeds` | the seeds of hadron files made before the production file entered them (`seed_scheme` absent or `legacy`): to reproduce those. Their events share seeds across the files of a campaign |
| `--correlated` | correlated sampling: iSS's random numbers are addressed by the cell, and each event's `bulk_jet` gets its background's seed (`--common-seeds`), so jet and background give the same hadrons where their surfaces agree. Same physics per leg, but jet − background is ~8× less noisy at \|η\| < 1 and ~40× less over the full acceptance: the same precision with ~8× fewer oversamples. Sample k of the jet leg belongs to sample k of its background: `HadronFileReader.jet_minus_background` then errs from the per-sample differences (`paired`, automatic). Not with `--use-stored-seeds` or `--oversample-bg` (below) |
| `--correlated-block DTAU,DX,DETA` | its block size (default 0.5 fm/c, 1 fm, 0.5): the gain is flat from 0.25 to 1 |
| `--common-seeds` | only the seeds of `--correlated`: with iSS's conventional sampling the legs decorrelate at the first hadron, so no gain alone (the null test) |
| `--keep-bits-p B`, `--keep-bits-x B` | round the hadrons' momenta `p` and positions `x` to `B` float32 mantissa bits (1–23; default: full precision). `12` and `8` store 58% of the bytes. Set once per campaign (below) |
| `--eta-max X` | store only hadrons with pseudorapidity \|η\| < X, in all three tags (default: all). iSS still samples the whole surface: every sample and every result inside the cut is unchanged. `2` keeps ~54% of the bulk hadrons, ~60% of the bytes. Set once per campaign (below) |
| `--charged` | store only charged hadrons, in all three tags: 58% of the bulk hadrons (below) |
| `--no-x` | store no positions `t, x, y, z`: ~60% of the bytes; readers give `x` as empty `(N, 0)` arrays (below) |
| `--add-initiators` | only add `initiators/` (each event's shower-initiating partons, from the pair file) to existing `bulk_jet` / `jet_frag` outputs, without hadronizing again: for files made before hadronize.py copied them. `--force` replaces an existing group |
| `--no-initiators` | don't copy the initiators (by default they are copied whenever the pair file is next to the particlize file) |

#### Hadron precision (`--keep-bits-p`, `--keep-bits-x`)

`hadronize.py` stores each hadron's `p` = (E, px, py, pz) and `x` = (t, x, y, z) as float32.
At full precision these barely compress (27.4 of 40 raw bytes per hadron with Blosc-zstd),
because the low mantissa bits are noise. Rounding them to fewer mantissa bits (round half to
even, as `keep_bits` does for the hydro files) makes the files much smaller:

```bash
# recommended for a campaign
python run_hadronize.py out_had -j 8 --oversample 500 --n-frag 50 --keep-bits-p 12 --keep-bits-x 8
# one file
python hadronize.py out_had/AuAu_0_10_jet_seed0001_particlize.h5 --oversample 500 --n-frag 50 \
       --keep-bits-p 12 --keep-bits-x 8
```

Measured on the M3 Max, seed 1, both legs, 200 oversamples (1.6 M hadrons per leg):

| `p` / `x` bits | max. relative error (p / x) | bytes per hadron | file per leg | shift of jet − background, in σ (200 oversamples) |
|---|---|---|---|---|
| full (default) | 0 | 27.4 | 43.8 MB | 0 |
| 16 / 16 | 7.6e-6 | 21.3 | 78% | dN/dp_T ≤ 0.03, E(\|η\|<1) 0.001 |
| **12 / 8 (recommended)** | 1.2e-4 / 2.0e-3 | 15.8 | **25.3 MB (58%)** | dN/dp_T ≤ 0.14, E(\|η\|<1) 0.004 |
| 10 / 10 | 4.9e-4 | 15.2 | 55% | dN/dp_T ≤ 0.23, E(\|η\|<1) 0.015 |
| 8 / 8 | 2.0e-3 | 13.3 | 49% | dN/dp_T ≤ 0.74, E(\|η\|<1) 0.024 |

The dN/dp_T column is the largest shift over 60 bins of 50 MeV (charged, \|η\| < 1).

- **What it changes.** Only the stored digits. The same seeds sample the same hadrons: a
  rounded file equals the full-precision file rounded afterwards, hadron for hadron. `pid`,
  `pstat`, the offsets and `units/` stay exact.
- **Integrated observables** (energies, yields) do not see it: the rounding errors are
  unbiased and average out over thousands of hadrons.
- **Histograms.** Rounding moves a bin edge by up to half a rounding step: 1.2e-4 relative
  for `p` at 12 bits. This moves counts between neighbouring bins. It is systematic, so it
  does not average down with more events. In jet − background it cancels to first order,
  since both legs shift the same way. For single-leg spectra, bins should be much wider than
  that step. That is why `p` should stay at 12 bits or more; 8 bits is too coarse.
- **Positions.** 8 bits is ≤ 0.03 fm at t = 15 fm/c, well below MUSIC's 0.2 fm cells.
- **Nothing is lost for good.** The particlize file and `units/seed` reproduce the exact
  hadrons: `hadronize.py --force` without `--keep-bits-*` on any production file.
- **Provenance.** Each dataset records `keep_mantissa_bits` and `max_rel_error`.
  `jetscape.hadrons_h5.hadron_precision(path)` and `HadronFileReader.precision(i)` read them
  back. Reading is unchanged.

**For a campaign:**
1. Choose the precision before the first `run_hadronize.py` call. Pass the same flags to
   every call: `--follow`, restarts, other machines.
2. Don't change it in the middle of a campaign. `--skip-complete` keeps complete files
   whatever their precision; it prints a warning when it differs from the requested one, but
   does not redo them. `HadronFileReader` warns when a campaign's files mix precisions.
3. To change it later, redo the files with `--force` or into a new `--out-dir`.
4. Keep the validation job (check 1 in *Checks before a campaign*) at full precision: the
   in-job hadrons it compares against are not rounded.

#### Hadron selection (`--eta-max`, `--charged`, `--no-x`)

The hadron files hold every hadron of the whole event, over the full η range of the
freeze-out surface: ~7000 per sample in 0–10% Au+Au, so ~2.8 M per event at 400
oversamples, each with its position. Most analyses need less. Three options store only
part of it, in all three tags:

| option | stores | bulk hadrons kept | bytes |
|---|---|---|---|
| `--eta-max X` | hadrons with pseudorapidity \|η\| < X | 54% at X = 2 | ~60% |
| `--charged` | charged hadrons (\|pid\| in `jetscape.hadrons_h5.CHARGED`) | 58% | |
| `--no-x` | no positions `t, x, y, z` | 100% | ~60% |
| `--charged --no-x` | both | 58% | 39–42% |

```bash
python run_hadronize.py out_had -j 16 --oversample 400 --n-frag 50 \
       --keep-bits-p 12 --keep-bits-x 8 --eta-max 2 --charged --no-x
```

The byte fractions are for files rounded to 12/8 bits. `--charged --no-x` was measured on c1
file 0001 (2 events, 20 oversamples): `bulk_jet` came to 0.42 of the full file, `bulk_bg`
to 0.39, and the options combine.

##### `--charged` and `--no-x`

- **What they change.** Only what is stored, as for `--eta-max`. The same seeds give the
  same hadrons: on c1 file 0001, the `--charged --no-x` files are identical, hadron for
  hadron and sample by sample, to the full files filtered to charged hadrons. Charged
  jet − background + fragments at \|η\| < 1 is identical, errors included.
- **Without positions** the files have no `hadrons/x` dataset. Readers (`Hadrons`,
  `HadronFile`, `JetEvents`, `HadronFileReader`) give `x` as an empty `(N, 0)` array: code
  that needs positions fails at once instead of reading zeros. `--keep-bits-x` is then
  ignored. `run_h5toROOT.py` writes such files without `t, x, y, z`.
- **What they rule out.** `--charged`: neutral hadrons, photons from decays, and the total
  energy balance (neutral hadrons carry 44% of the bulk energy and 43% of the fragments'
  on c1 file 0001). `--no-x`: space-time observables
  (freeze-out positions, femtoscopy).
- **Provenance.** The files record `charged_only` and `positions`.
  `jetscape.hadrons_h5.hadron_selection(path)` reads all three selections back.
  `HadronFileReader.selection()` returns the campaign's, and `selection(i)` returns it per
  tag for production file i.

##### `--eta-max`

What a cut keeps. Measured on `pth10-40_eta06_gridnorm`, file 0001: 5 M `bulk_jet` hadrons,
and all of `jet_frag`:

| \|η\| < | bulk hadrons | bulk energy | jet fragments | fragment energy |
|---|---|---|---|---|
| 1 | 27% | 5% | 65% | 53% |
| 1.5 | 41% | 9% | 76% | 59% |
| **2** | **54%** | 14% | 82% | 63% |
| 3 | 76% | 30% | 91% | 72% |
| 4 | 90% | 55% | 97% | 84% |
| none (default) | 100% (up to \|η\| ≈ 12) | 100% | 100% | 100% |

Charged hadrons alone give the same fractions to within 1%.

- **What it changes.** Only what is stored. iSS still samples the whole surface with the
  same seeds, so the stored hadrons are exactly those of a run without the cut, less the ones
  outside it. Every sample keeps its place, empty if nothing in it passes, so per-sample
  averages inside the cut are unchanged. Checked on c1 file 0001 (2 events, 20
  oversamples, `--keep-bits-p 12 --keep-bits-x 8 --correlated`): `pid`, `pstat`, `p`, `x`
  and the samples are identical, hadron for hadron, to the uncut files filtered at
  \|η\| < 2. `jet_minus_background` for charged hadrons at \|η\| < 1 is identical,
  including its paired errors.
- **All three tags.** `bulk_jet`, `bulk_bg` and `jet_frag` get the same cut, so
  jet − background and bulk + fragments are complete inside it. `jet_frag` is tiny, so the
  saving there doesn't matter.
- **Disk.** The files shrink less than the hadron count: the forward hadrons that are
  dropped take fewer bytes each than the mid-rapidity ones that stay. At \|η\| < 2 the
  files are 56–63% of the uncut size (`bulk_bg` / `bulk_jet`, 12/8 bits), against 54% of
  the hadrons. The cut combines with `--keep-bits-*` and `--correlated`.
- **Time and memory** are unchanged: iSS samples the whole surface either way.
- **The definition.** The cut uses pseudorapidity η = atanh(p_z/\|p\|) of the stored,
  i.e. rounded, momenta (`jetscape.hadrons_h5.pseudorapidity`). `Hadrons.eta` and
  `EventHadrons.eta` use the same function, so every hadron read back has \|η\| < X exactly.
  It is a cut on η, not on the rapidity y. For massive hadrons \|y\| ≤ \|η\|, so a hadron at
  \|y\| < Y can have \|η\| > Y: an analysis in y at \|y\| < Y needs X well above Y (slow
  protons at low p_T reach η ≫ y).
- **Choosing X.** Leave a margin around what the analysis bins. R = 0.4 jets at
  \|η_jet\| < 0.6 need hadrons at \|η\| < 1.0. The wake and the recoil correlations reach
  further in η than the jet cone. \|η\| < 2 keeps a unit of η beyond such jets and still
  saves ~40% of the disk.
- **What it rules out.** Anything that needs the whole event: the total energy balance of
  jet − background over all η, forward observables, and dN/dη beyond X. Outside the cut the
  files have no hadrons, which is not the same as zero hadrons. Keep the validation job
  (check 1 in *Checks before a campaign*) without `--eta-max`: the in-job hadrons it
  compares against are not cut.
- **Provenance.** The files record the cut as the `eta_max` attribute (absent: no cut).
  `jetscape.hadrons_h5.hadron_eta_max(path)` reads it. `HadronFileReader.eta_max()` returns
  the campaign's cut: the smallest of all its files, None without one. `eta_max(i)` returns
  `{tag: cut}` for production file i.
- **Nothing is lost for good.** As with the precision, the particlize file and
  `units/seed` reproduce every hadron: `hadronize.py --force` without `--eta-max`.

**For a campaign**, the same rules as for the precision, for all three options:
1. Choose the selection before the first `run_hadronize.py` call, and pass it to every call.
2. `--skip-complete` keeps complete files whatever their selection. It prints a warning when
   it differs from the requested one, but doesn't redo them.
3. `HadronFileReader` warns when a campaign's files, or the tags of one file, mix cuts,
   charged-only and all hadrons, or files with and without positions. Results are then
   only right for what all of them hold.
4. To change the selection later, redo the files with `--force` or into a new `--out-dir`.

#### Export to ROOT (`run_h5toROOT.py`)

For analyses in ROOT, `run_h5toROOT.py` converts a hadronized campaign, `-j` files at a
time. Each production file becomes one ROOT file. It holds its hadrons (`bulk_jet`,
`bulk_bg`, `jet_frag`: one entry per oversample), an `events` table (the particlize file's
event columns, window, cross section, samples, seeds, shower initiators), a `windows`
table and the full provenance. A campaign file adds the cross section and the weight per
event of every pT̂ window over all files:

```bash
python run_h5toROOT.py out_had -j 4                       # RNTuple (ROOT >= 6.34), next to the inputs
python run_h5toROOT.py out_had -j 4 --out-dir out_root --no-x --eta-max 1 --charged
python run_h5toROOT.py out_had -j 4 --format ttree        # for older ROOT
```

It needs uproot. PyROOT gives ~20% smaller files but isn't required. Converted files are
skipped on a re-run. The file layout, the format study (RNTuple is 0.9× the HDF5 size and
reads ~1.8× faster than a TTree) and how to weight and pair events are in
[`root_export/README.md`](root_export/README.md). The 20-file `gridnorm` campaign took
2 min 40 s with `-j 4`.

### C. Analysing a campaign: `HadronFileReader`

The hadron files are read the way the hydro files are: side by side, one production file
per seed, no merging. `HadronFileReader` (in `jetscape.hadrons_h5`) opens every production
file of a campaign as one data set:

```python
import numpy as np
from jetscape.hadrons_h5 import HadronFileReader

def jet_axis(info):
    """phi and rapidity of the event's hardest shower-initiating parton."""
    ini = info.initiators()                 # (K, 11): shower, pid, pstat, px, py, pz, E, ...
    px, py, pz, E = ini[np.argmax(np.hypot(ini[:, 3], ini[:, 4])), 3:7]
    return np.arctan2(py, px), 0.5 * np.log((E + pz) / (E - pz))

def dphi(ev, info):                         # hadron azimuth relative to this event's jet
    return np.mod(ev.phi - jet_axis(info)[0], 2 * np.pi)

def soft_charged(ev, info):                 # charged, pT < 4 GeV, |eta - y_jet| < 1
    return ev.charged & (ev.pt < 4) & (np.abs(ev.eta - jet_axis(info)[1]) < 1)

with HadronFileReader("out_had") as r:      # every *_particlize.h5 in the directory
    print(r.n_files, "files,", r.n_events, "events, tags", r.tags())

    # the wake: <jet leg> - <background> per 30-degree bin, all events of all seeds
    bins = np.linspace(0, 2 * np.pi, 13)
    wake, err = r.jet_minus_background(dphi, bins, mask=soft_charged)
    # ... fragments=True adds the jet's own hadrons (jet_frag)

    # one tag, any observable: names of EventHadrons attributes, or callables
    E_mid, E_err = r.total("bulk_jet", weights="E", mask=lambda ev, info: np.abs(ev.eta) < 1)
    pt_spec, pt_err = r.hist("bulk_bg", "pt", np.linspace(0, 3, 31), mask="charged")

    # single events, numbered across all seeds
    ev = r.jet_event(2, 17)                 # global event 2, oversample 17 + fragments
    bg = r.background_event(2, 17)          # the background that event used
    info = r.event_info(2)                  # stem, local event, bg_unit, seed, initiators()
```

What it does:

- **Finds the files.** `source` is a directory, a glob pattern (of particlize or hadron
  files) or a list of stems. Each production file needs its `_particlize.h5`; hadron tags
  missing in some files show up in `r.tags(file_index)`, and `r.tags()` lists the tags every
  file has.
- **Numbers events globally,** in file-name (seed) order: `r.locate(g)` gives
  (file, local event) and `r.global_event(file, local)` the reverse.
- **Refuses mixed runs.** A hadron file whose recorded `source_uuid` isn't its particlize
  file's `file_uuid` (renamed, or from another run) is refused;
  `check_uuid=False` overrides.
- **Flags repeated collisions.** A background found in more than one production file gives a
  warning: bit-identical (`events/bg_key`), or probably the same collision on another output
  grid (same `prod_seed` and freeze-out cell count). That is what campaigns over the same seeds
  produce (see *Campaigns*). `r.duplicate_backgrounds()` lists them.
- **Averages event by event.** For every event, the samples of its unit are histogrammed and
  divided by that unit's number of samples. These per-event means are then averaged over the
  events, so every event counts the same, even when files were hadronized with different
  `--oversample`. Errors are compound-Poisson per event, added over events; for
  `jet_minus_background` of `--correlated` files, the spread of the per-sample differences
  (`paired=True`, chosen automatically), the only right error when the legs are correlated.
  (The compound-Poisson error treats every hadron as independent; resonance daughters in
  one bin make it ~20% too small for soft multiplicities, which `paired=True` avoids for
  independent legs too, given equal sample counts.)
- **Gives each event its own background.** A background shared by several events
  (`--reuse`) is evaluated once per event with that event's `info`, which is what
  jet-relative observables need. Its hadrons, counted several times, enter the error as the
  correlated sum they are.
- **Keeps jet and background consistent.** `jet_minus_background` uses only events whose jet
  leg *and* background have samples (an empty surface drops the event from both), each event
  against its own background.
- **Reads one file at a time.** Only the units it needs are loaded.
  - The four events of seeds 1 and 3 (500 and 100 oversamples, about 11 M hadrons) take
    2.7 s.
  - For seed 1 it reproduces the notebook's energy balance: 18.1 ± 2.7 GeV at |η| < 1.

The callables receive an `EventHadrons` (`pid`, `pstat`, `p`, `x`, `E`, `px`, `py`, `pz`,
`pt`, `eta`, `y`, `phi`, `charged`, `sample`, `n_samples`, `species()`: all samples of that
tag for that event) and an `EventInfo` (`event`, `stem`, `local_event`, `bg_unit`, `seed`,
`initiators()`). `values` may return a tuple for an N-d histogram. `initiators()` reads the
`initiators/` group `hadronize.py` copies into `bulk_jet` and `jet_frag`, so an analysis needs
only the particlize and hadron files. For hadron files without it, it falls back to the pair
file's `shower/` group (next to the particlize file); `hadronize.py --add-initiators` adds the
group to such files.

**Option if needed: merging into single files.** A campaign could also be merged into one
file per tag (`merge_hadrons.py`, not written). It would concatenate the units, shift the
offsets, add `seed`/`source_event` columns, and renumber backgrounds globally together with a
merged event → background map. That only helps for moving or archiving a campaign as three
files instead of 4 × N_seeds: a merged file at 500 oversamples is ~50 GB per 500 events, and
the reader above makes it unnecessary for analysis.

### D. Several pTHat windows per background (`--pthat-bins`)

The background leg doesn't depend on the pT̂ window: seed 1 gives the same background at
20–40 and at 50–70 GeV. So campaigns for several windows can share their backgrounds and
run MUSIC_1 once for all of them:

```bash
# 3 windows, one jet per window per background: MUSIC_1 once per 3 events
python run_prod_jet.py --events 30 --seed 1 --pthat-bins 20-40,50-70,70-90
OMP_NUM_THREADS=5 ./run_jobs.sh -j 4 --mps 20 30 0 out_pth3 --pthat-bins 20-40,50-70,70-90 \
       --write-particlize both
# 2 jets per window per background (--reuse 6)
python run_prod_jet.py --events 30 --seed 1 --pthat-bins 20-40,50-70,70-90 --jets-per-bin 2
```

- **How.** PythiaGun gets `<pTHatBins>` (`20 40 50 70 70 90`) and builds one Pythia
  instance per window: the same settings, its own pT̂ range, seed and cross section. Event
  i uses window i mod K. `--reuse` is set to K × `--jets-per-bin` (M), and `--events` must
  be a multiple of it. So every background gets exactly M jets in each window, and each of
  them has its own MUSIC_2.
- **Cost (measured, GB10, one job, seed 1, 3 windows, `--write-particlize both`).** An
  event that also runs MUSIC_1 took 34.1 and 41.3 s; the events reusing its background
  21.1–24.4 s. The job averaged 27.6 s per event, 0.73 of an all-new-background event
  (1.37× the throughput). In general the cost goes as (1 + 1/(K·M)) / 2 of a pair, since
  MUSIC_1 and MUSIC_2 cost about the same, down to half for many windows. Campaign
  throughput with `-j 4 --mps` has not been measured. With `--write-particlize both`,
  the background surface is also stored (and `hadronize.py` samples it) once per K·M events.
- **Memory (measured, GB10, seed 1: `--reuse 1` over 2 events against 3 windows over 6
  events, the same two backgrounds).** The same: peak RSS (`ru_maxrss`) 20.1 vs 20.5 GB,
  ~12.3 vs ~12.7 GB between MUSIC runs, ~4 MB per extra Pythia instance.
  - The background's framework copy stays in memory for its whole group of events
    (Matter/LBT query it for every jet). With `--reuse 1` it is held just as long within
    its event, so reuse keeps it longer, not larger.
  - Two backgrounds are never held at once: the old one is released before the next
    MUSIC_1 run.
  - The peak is a ~1 s spike while a background leg runs. Its size follows the
    background's length (here the 111-frame one), not the reuse, and with reuse it comes
    once per group instead of every event.
  - The kept background is host memory, not GPU memory. Each MUSIC instance keeps its
    GPU grid (100 × 100 × 60 cells) until its own next run, so MUSIC_1's and MUSIC_2's
    grids coexist in either mode. On a discrete GPU that is two grids either way; on the
    GB10 they are unified memory, which `nvidia-smi` does not count.
- **Statistics.** Per window, a campaign with M = 1 is what a `--reuse 1` campaign of that
  window would be: one jet per background. The windows share their backgrounds, so results
  combined over windows (a spectrum stitched from the windows) have errors correlated
  between them through the bulk; per-window results and jet − background don't.
- **Seeds.** PythiaGun takes `<Random><seed>` itself, not through the task number.
  Window 0 keeps that seed, so it gives the same Pythia events as a single-window job of
  that window with the same seed (checked: pT̂ and σ identical). Windows 1..K−1 get seeds
  derived from (seed, window) (splitmix64, 1 … 900,000,000), recorded in
  `pthat_bin_seeds`. The extra Pythia instances are not framework tasks, so no other
  module's seed moves: the backgrounds are those of a single-window job with the same seed
  (checked: seed 1, all 95 frames bit-identical). The jets after Matter/LBT are not,
  because the energy-loss random streams are used by other events.
- **What is recorded.** Per event `diag/pthat_bin`, `diag/pthat`, `diag/event_weight`
  (also in the particlize file's `events/`). Per file the windows (`pthat_bins`,
  `pthat_jets_per_bin`), and when the job ends each window's `pthat_bin_sigma_gen`,
  `pthat_bin_sigma_err` [mb], `pthat_bin_n_accepted` and `pthat_bin_seeds` (in the pair
  file, the particlize file and the `.json`). A single window (`--pthat-bins 50-70`) runs
  exactly as `--pthat-min 50 --pthat-max 70` and records the same.
- **Needs** X-SCAPE branch `N_ptHat_per_hydro` (PythiaGun and the `<pTHatBins>` default in
  `config/jetscape_main.xml`) and PyJetscape built against it. An older X-SCAPE refuses the
  XML (the tag has no default in its main XML), and the driver refuses to run unless
  PythiaGun reports exactly the requested windows.
- **Refused:** window edges with more than one decimal (PythiaGun passes them to Pythia
  with one, as `pTHatMin`), `PhaseSpace:pTHat…` in `<LinesToRead>` (it would override every
  window), and `nReuseHydro` not a multiple of K.

**Analysis.** `HadronFileReader` knows the windows:

```python
with HadronFileReader("out_pth3") as r:
    r.pthat_bins                       # (K, 2) array, the same in every file
    ev_k = r.pthat_bin_events(1)       # global events of window 1, for events=
    sigma, err = r.pthat_bin_sigma(1)  # window 1's cross section [mb] over the campaign
    info = r.event_info(4)             # info.pthat_bin, info.pthat

    # a spectrum over all windows: each window's per-event average times its cross section
    edges = np.linspace(0, 40, 41)
    spec = np.zeros(len(edges) - 1)
    var = np.zeros(len(edges) - 1)
    for k in range(len(r.pthat_bins)):
        h, e = r.hist("jet_frag", "pt", edges, mask="charged", events=r.pthat_bin_events(k))
        s, _ = r.pthat_bin_sigma(k)
        spec += s * h
        var += (s * e) ** 2            # jet_frag: windows independent; bulk tags are not
```

`pthat_bin_sigma` averages the files' Pythia estimates weighted by their accepted events.
Each window's `hist` is an event average, so the windows can have different event
counts.

#### Hadronizing a `--pthat-bins` campaign

`hadronize.py` and `run_hadronize.py` need nothing new for these files. A `--pthat-bins`
file is a `--reuse K·M` file (K windows, M = `--jets-per-bin`) with three more per-event
columns (`events/pthat_bin`, `pthat`, `event_weight`), and hadronization works on events and
backgrounds, not on windows:

| tag | unit | with K windows, M jets per window |
|---|---|---|
| `bulk_jet` | event | one per event, whatever its window: `--oversample` samples each |
| `jet_frag` | event | one per event: `--n-frag` fragmentations each |
| `bulk_bg` | background | one per background, shared by its K·M events across all windows |

- **Seeds** come from (`--seed`, the production file, tag, unit, sample), as for every
  file: the window plays no role, and the events of different windows are sampled
  independently.
- **`--correlated`** works as under `--reuse`: every event's `bulk_jet` gets its
  background's seed, in every window, and each sample k of an event pairs with sample k of
  its background. It still needs equal sample counts, so no `--oversample-bg`.
- **The background's oversampling is the one choice the windows affect.** At fixed CPU,
  the error of jet − background is smallest when a background gets U × the jet leg's
  samples, U = the number of events **in the average** that share it (the reasoning behind
  `auto`). Which U applies depends on the analysis:

| analysis | events sharing a background in the average | `--oversample-bg` |
|---|---|---|
| each window on its own (per-window spectra, wake per pT̂) | M | `per-pthat-bin` (M × `--oversample`); with M = 1 the same as leaving it out |
| all windows together, equal weights | K·M | `auto` (K·M × `--oversample`) |
| all windows stitched with their cross sections | between M·√K (one window dominates) and K·M | `per-pthat-bin` or `auto`; nearer `per-pthat-bin` when the σ weights differ by orders of magnitude, as for 20–40 vs 70–90 GeV |

- **None of the choices is wrong.** More background samples only lower the error, and
  `HadronFileReader` weighs every event the same whatever its unit's count
  (`units/n_samples`). `auto` on a per-window analysis spends K times the needed background
  CPU: per event, the background then costs about what the jet leg costs, instead of 1/K of
  it, so up to ~2× the hadronization time.
- **Both are capped** at `--oversample-bg-max` (default 2000, ~1.7 GB per iSS process).
  With `--oversample 500`, `auto` needs K·M ≤ 4 to stay under it; `per-pthat-bin` needs
  only M ≤ 4. `run_hadronize.py` estimates the memory from the same counts.
- **Decide once per campaign**, like the precision: `--skip-complete` does not compare
  sample counts, so a campaign hadronized half with `auto` and half with `per-pthat-bin`
  is mixed. Mixed files are still averaged correctly, event by event.
- **Reading back:** `HadronFileReader` evaluates each event against its own background
  (`events/bg_unit`); `events=r.pthat_bin_events(k)` restricts any histogram, total or
  `jet_minus_background` to window k (see *Analysis* above).

Measured (GB10, seed 5, 3 windows):
- A 3-event file (M = 1) with `--oversample 20 --oversample-bg auto` gave one `bulk_bg` unit
  with 60 samples and 20 per event in `bulk_jet`. `HadronFileReader` put each event in its
  window against that background.
- A 6-event file (M = 2, one background) with `--oversample 10`: `--oversample-bg
  per-pthat-bin` gave the background 20 samples (M × 10), `auto` 60 (K·M × 10).

**Checks before such a campaign:** the null test with the windows,
`python run_prod_jet.py --events 3 --seed 1 --pthat-bins 20-40,50-70,70-90 --no-deposit`,
must give `arr == arr_bg` in every event; and in any job, `diag/bg_id` must repeat K·M
times with `diag/pthat_bin` cycling 0 … K−1.

### E. Parton rapidity cut (`--parton-ymax`)

Most measurements are at mid-rapidity, but at low pT̂ the jets spread far in rapidity. In
the `pth10-40` campaign (100 events per window) the hardest shower-initiating parton had
|y| < 0.6 in only 43% (10–20 GeV), 66% (20–30) and 62% (30–40) of the events. Much of the
hydro time went to events that cannot put a jet into a mid-rapidity measurement.
`--parton-ymax` selects the events before they are simulated:

```bash
# the pth10-40 campaign, only events whose hardest parton has |y| < 0.6
OMP_NUM_THREADS=5 ./run_jobs.sh -j 4 --mps --campaign pth10-40-y06 20 15 0 out_pth10-40-y06 \
    --pthat-bins 10-20,20-30,30-40 --jets-per-bin 5 --parton-ymax 0.6 --write-particlize both

# for inclusive jets measured at |eta_jet| < 0.6: either of the two hardest partons at
# |y| < 0.7 (a margin for the jet axis, and the recoil leg counts too; see below)
    ... --parton-ymax 0.7 --parton-y-mode any ...
```

- **What is cut.** The partons PythiaGun hands to the framework: status 62, after ISR
  and MPI with primordial kT (the final partons with `FSR_on`), i.e. the partons that
  start the showers (`shower/initiators`). The test is on their two hardest by pT:

  | `--parton-y-mode` | keeps the event if | for |
  |---|---|---|
  | `leading` (default) | the hardest has \|y\| < Y | analyses that need the hardest jet (or a trigger on it) in the acceptance |
  | `both` | the two hardest have \|y\| < Y | dijets, back-to-back observables |
  | `any` | either of the two hardest has \|y\| < Y | inclusive jets and the medium response: either leg can put a jet or a wake into the acceptance |

  Not on the hard process itself: its two outgoing partons have exactly the same pT at
  leading order (checked: 400 of 400 events), so "the leading one" is only defined once
  ISR has recoiled against them. "Any parton" would almost always pass, because the list
  also holds soft MPI partons (2–12 partons per event).
- **It selects events; nothing downstream is restricted.** The cut decides which events
  are generated. An accepted event keeps all its partons, including a recoil parton at
  forward rapidity, and MATTER, LBT and the hydro see the full event, so energy, momentum
  and the medium response stay consistent. A rejected event is regenerated, like one with
  fewer than two partons, before any shower or hydro: its jets never exist in the files,
  wherever they would have ended up.
- **Cross sections.** Pythia's `sigmaGen` still counts the rejected events, so per window
  PythiaGun counts the events that reach the cut (`n_tried`) and pass it (`n_kept`), and
  its cross section is σ = sigmaGen × kept/tried (error including the acceptance's). The
  event header, `pthat_bin_sigma_gen` and `HadronFileReader.pthat_bin_sigma` are this σ;
  `pthat_bin_sigma_gen_raw` is Pythia's own. Checked (10–20 GeV, leading \|y\| < 0.6,
  30,000 events with and without the cut): σ with the cut = 2.3057e-3 mb, σ without ×
  the fraction of uncut events passing = 2.3025e-3 mb (ratio 1.0014); acceptance
  0.4027 ± 0.0018 counted, 0.4013 ± 0.0028 in the uncut sample.
- **Choosing Y and the mode.** The analysis still applies its own cut on the
  reconstructed jets or hadrons; the parton cut only has to leave out events that cannot
  contribute to it. So Y must be *looser* than the analysis acceptance, by how far the
  observable can move from its initiating parton. Otherwise events that would have put a
  jet just inside the acceptance are never generated, and the sample is short near the
  edge.
  - *Jets.* The jet axis stays close to its parton (estimate: within ~0.1 for R = 0.4;
    not yet measured at hadron level). For jets at \|η_jet\| < η_max, use
    Y ≈ η_max + 0.1–0.2. Example: `--parton-ymax 0.6` is complete for jets at
    \|η_jet\| up to about 0.4–0.5; for \|η_jet\| < 0.6 (R = 0.4), use 0.7–0.8.
  - *The mode matters more than the margin.* With `leading`, an event whose hardest
    parton is at y = 1.5 is rejected even if the second one, the recoil leg of similar pT,
    is at y = 0.2, and its perfectly good mid-rapidity jet is lost. Quenching can also make
    the second leg's jet the harder one. For inclusive jets use `any`: at 10–20 GeV and
    \|y\| < 0.6 it keeps 66% of the events against 40% for `leading`.
  - *Medium response.* The deposit sits at its parton's rapidity: in the `pth10-40`
    campaign 68% of the energy the hardest parton's shower deposits (\|Δφ\| < 0.4 of it)
    lies within \|η_s − y\| < 0.16. The wake's hadrons are then spread by Cooper–Frye,
    thermal pions over roughly ±1 in rapidity (estimate). For wake hadrons at
    \|η\| < η_max, use Y ≈ η_max + 1 with `any`.
  - The migration can be measured once from a hadronized campaign (reconstructed jet η
    or the wake's hadrons against `diag/parton_y_lead`) and Y fixed from it.
- **Recorded.** Per event `diag/parton_y_lead`, `parton_pt_lead`, `parton_y_sub`,
  `parton_pt_sub` (the two hardest handed-over partons; also in the particlize file's
  `events/`). Per file `parton_ymax`, `parton_y_mode`, and at the end per window
  `pthat_bin_n_tried`, `pthat_bin_n_kept`, `pthat_bin_acceptance`,
  `pthat_bin_sigma_gen_raw`, `pthat_bin_sigma_err_raw` (and the `.json`).
  `HadronFileReader.pthat_bin_acceptance(k)` sums them over a campaign.
- **Cost.** Pythia regenerates an event in well under a millisecond (10,000 kept of
  24,892 tried: 4.2 s, against 2.6 s without the cut, startup included); every hydro run is then an accepted event: 1/0.40 = 2.5×
  more usable 10–20 GeV events per GPU hour at \|y\| < 0.6.
- **Needs** X-SCAPE branch `pyGun_eta_cut` (PythiaGun and the `<partonYMax>` /
  `<partonYMode>` defaults in `config/jetscape_main.xml`) and PyJetscape built against it;
  the driver checks that PythiaGun reports the requested cut. Without `--parton-ymax`
  nothing is cut and the files are as before.

### Planning numbers (GB10, one 0–10% event, measured)

| | per event | disk per event |
|---|---|---|
| A: hydro pair (`grid_fno.yaml`, Blosc-zstd) | 29.5 s alone; ~190 events/h with `-j 4 --mps` | 285 MB |
| B: + `--write-particlize both` | 34.6–35.3 s alone (+5.4 s: MUSIC builds and hands over the two surfaces; +13.5 s before MUSIC4GPU `5058545`); `-j` throughput not measured | + 154 MB |
| A with `--reuse N` or `--pthat-bins` (shared `arr_bg`) | as A, MUSIC_1 once per N events | ~160 MB + 148/N MB (measured at N = 3: 210 MB) |
| B with `--reuse N` | the background surface once per N events | + 78 MB + 78/N MB |
| `hadronize.py`, both legs, 500 oversamples, 50 fragmentations | ~14 s on one core, ~10 s with `OMP_NUM_THREADS=5` (per surface ~5 s fixed + ~7 ms per oversample); ~58 s before [`PLAN_iSS_optim.md`](../../../../docs/PLAN_iSS_optim.md) Part A | ~100 MB per leg (~0.2 MB per oversample); 58% with `--keep-bits-p 12 --keep-bits-x 8`, and ~60% of that with `--eta-max 2` |

Peak memory: +0.5 GB per production job with surfaces; `hadronize.py` ~1.4 GB per surface
up to ~1000 oversamples (1.6 GB for both legs), 1.7 GB at 2000 (it was 2.5 GB at 500 and
3.9 GB at 1000 before the hadrons went to numpy as arrays). Three or four `hadronize.py`
processes keep up with a whole four-job GPU campaign.

## Options

| option | effect |
|---|---|
| `--hard pythia` (default) | PythiaGun. The vertex is a 3dMCGlauber binary-collision point (x, y; z = t = 0). `--pthat-min/--pthat-max` override the XML's 50–70 GeV. |
| `--hard pgun --pgun-pt P` | One parton at fixed pT, always from the origin: PGun zeroes the sampled vertex (`PGun.cc:117-120`). |
| `--reuse N` | `setReuseHydro`: MUSIC_1 runs once per N events, MUSIC_2 every event (a new jet each time). `arr_bg` still reads per event, so it stays aligned with `arr`; it is stored once per background (`--bg-layout`). `diag/bg_id` gives the first event that used each background. |
| `--bg-layout {auto,full,shared}` | How `arr_bg` is stored. `full`: one copy per event. `shared`: each background once, `arr_bg` a virtual dataset over it that reads the same (see [Shared backgrounds](#shared-backgrounds---bg-layout)). `auto` (default): `shared` when a background is reused (`--reuse` > 1, which includes `--pthat-bins` with more than one window or `--jets-per-bin` > 1), else `full`, so files without reuse are unchanged. |
| `--pthat-bins A-B,C-D,...` | K PythiaGun pT̂ windows in one job, one Pythia each; event i uses window i mod K, and `--reuse` is set to K × `--jets-per-bin`. `--events` must be a multiple of that. Not with `--reuse`, `--pthat-min/max` or `--hard pgun`. See [D](#d-several-pthat-windows-per-background---pthat-bins). |
| `--jets-per-bin M` | With `--pthat-bins`: jets per window per background (default 1). |
| `--parton-ymax Y` | PythiaGun keeps only events whose handed-over partons (status 62) pass \|y\| < Y on their two hardest (`--parton-y-mode`); rejected events are regenerated before any shower or hydro, and the cross sections are Pythia's × kept/tried. With or without `--pthat-bins`. See [E](#e-parton-rapidity-cut---parton-ymax). |
| `--parton-y-mode {leading,both,any}` | With `--parton-ymax`: the hardest parton (default), the two hardest, or either of them. |
| `--no-deposit` | Null test: MUSIC_2 without the liquefier. `arr` must equal `arr_bg` bit for bit (`diag/frames_identical == ntau`). Showers and droplets are still recorded. |
| `--native` | Both legs on MUSIC's own grid (100 × 100 × 60) instead of the YAML's. |
| `--workdir DIR` / `--keep-workdir` / `--in-build` | The job's working directory, as in `../prod_AuAu_0_10` (default `OUTDIR/work/<tag>`, removed after a successful job). |
| `--no-showers` | Skip `shower/`. |
| `--write-particlize {none,jet,both}` | Also write `<stem>_particlize.h5`: the jet leg's surface (`jet`) or both legs' (`both`, the background once per background), plus the final partons. Switches on those legs' surfaces (and their hand-off to the framework) on top of `--surface`. The pair file is unchanged (checked byte for byte). Costs +5.4 s per event for `both` (+13.5 s before MUSIC4GPU `5058545`; measured, below). |
| `--validate-inline` | Validation only: also run iSS on the jet leg and Colorless jet hadronization inside the job and store their hadrons and seeds (`<stem>_inline_{bulk_jet,jet_frag}.h5`), for `hadronize.py --use-stored-seeds`. **Changes the jet sample** of the seed (see Seeds below); the background is unchanged. |
| `--hadronize-xml FILE` | Settings for `--validate-inline` (default `hadronize.xml`). |
| `--surface {none,bg,jet,both}` | Which legs build MUSIC's freeze-out surface (`<freeze_out_surface>` in the first `<Hydro><MUSIC>` block, i.e. the background and the default, and in MUSIC_2's own block). On its own it produces nothing: only `--write-particlize` hands a surface to the framework and stores it, and it builds its legs itself. So a leg built but not stored costs ~3 s per MUSIC run (~6 s before MUSIC4GPU `5058545`) for no output, and the job warns about it. `none` (default) gives a bit-identical evolution. |

The job XML always contains **one** hard process. The automatic task list would run every
`<Hard>` child it finds, so the driver rebuilds that block from the option. Several pT̂
windows are one PythiaGun with `<pTHatBins>` (`--pthat-bins`), not several `<Hard>` blocks.

## What is written

On top of the single-leg schema (`arr`, `ntau_freezeout`, `tau_freezeout`, grid attributes):

| | |
|---|---|
| `arr` | the **jet leg** (MUSIC_2), `(nevents, 4, nx, ny, neta, ntau)` float32, channels `energy_density, vx, vy, vz` |
| `arr_bg`, `ntau_freezeout_bg`, `tau_freezeout_bg` | the **background leg** (MUSIC_1). Always the same shape as `arr`. The τ axis grows to the longer leg of any event, and each leg is exactly 0 after its own freeze-out. With a reused background, `arr_bg` is a virtual dataset over `arr_bg_store` (below); it reads the same. |
| `arr_bg_store`, `arr_bg_rows`, attribute `bg_layout = "shared"` | only with a reused background (`--bg-layout` shared, the default under reuse): each background once, in the row of the first event that used it (the other rows are never written and take no space), and every event's row. Read `arr_bg`, not the store. |
| `source/droplets`, `source/offsets` | the droplets MUSIC_2 was given, `(M, 8)`: `tau, x, y, eta, E, px, py, pz` (Milne position, Cartesian momentum). Event `i` is rows `offsets[i]:offsets[i+1]`. Each droplet deposits at `tau + liquefier_tau_delay`. |
| `shower/` | partons, vertices and initiators per event, as FastHydro writes them (`jetscape.showers`) |
| `diag/` | `n_droplets`, `E_droplets`, `n/E_droplets_late` (deposit after the jet leg froze out), `n/E_droplets_early`, `n_showers`, `n_partons`, `tau0_music`, `ntau_jet`, `ntau_bg`, `bg_id`, `frames_identical`, `wall_s`; with `--pthat-bins` also `pthat_bin`, `pthat`, `event_weight` |
| attributes | `pairing = "bg_jet"`, `arr_is`, `arr_bg_is`, `deposition`, `source_model`, `hard_vertex`, `liquefier_{dtau, tau_delay, time_relax, d_diff, width_delta, c_diff, gamma_relax}`, `freezeout_convention_id = "frames_written"`, and the run provenance (`prod_seed`, `prod_user_xml`, `prod_grid_yaml`, `prod_hard`, `prod_reuse`, ...); with `--pthat-bins` also `pthat_bins` (K, 2), `pthat_jets_per_bin` and, written when the job ends, per window `pthat_bin_sigma_gen`, `pthat_bin_sigma_err` [mb], `pthat_bin_n_accepted`, `pthat_bin_seeds` (the same in the particlize file) |

There is **no `source/S`** (`has_source = false`). MUSIC evaluates the liquefier kernel per
cell and step and never keeps a gridded source. A re-deposit on the output grid would not be
what MUSIC applied: FastHydro measured point sampling losing whole deposits at |η_d| ≥ 3.

## Reading it

```python
from fasthydro.browse import open_pair        # FastHydro's PairBrowser
with open_pair("out/AuAu_0_10_jet_seed0001.h5") as p:   # checks frame 0 is identical
    print(p.summary(0))
    de = p.diff(0, 20)                        # arr - arr_bg, energy density, frame 20
```

FastHydro's wake notebook and `Visualization/wake_pyvista.py` read the file the same way. With
plain h5py, the jet's effect in event `i` is `f["arr"][i] - f["arr_bg"][i]`.

To bring files of a campaign to one τ length, run `repad_h5.py`, as for the single-leg files.
It grows `arr` and `arr_bg` together (with shared backgrounds, it grows `arr_bg_store` and
rebuilds the `arr_bg` view):

```bash
python ../../python/jetscape/repad_h5.py out/AuAu_0_10_jet_seed*.h5
```

### Shared backgrounds (`--bg-layout`)

Under `--reuse N` (and `--pthat-bins`) N events share one background. With the default
`--bg-layout auto` it is stored once:

- `arr_bg_store` holds each background in the row of the first event that used it. The
  rows of the events that reuse it are never written, and HDF5 stores no data for them.
- `arr_bg_rows[i]` is event i's row in the store (`= diag/bg_id[i]`).
- `arr_bg` is an HDF5 virtual dataset whose row i is `arr_bg_store[arr_bg_rows[i]]`, in the
  same file. It has `arr`'s shape and reads exactly like a full copy: `f["arr_bg"][i]`,
  `PairBrowser`, the wake notebook, FNO4d's loader (`h5_key = arr_bg`, whose `unique_bg`
  dedupe still works from `diag/bg_id`) need no change.

Files without reuse keep the full layout and do not change. `--bg-layout full` forces it
under reuse, for a tool that writes into `arr_bg` (a virtual dataset cannot be written to).
Things to know about the shared layout:

- **Copy the file whole** (`cp`, `rsync`, `h5repack`). The view points into its own file
  (HDF5's `"."`), so moving or copying the file works (checked: a moved file, and
  `h5repack`, read the same). Copying *only* the `arr_bg` dataset into another file
  (`h5py`'s `copy`, `h5copy`) copies the view without its store, and it reads all zeros
  (checked with h5py). To get a standalone array, read it (`f["arr_bg"][...]`) and write
  that.
- **Never read `arr_bg_store` directly** for an event that reused its background: its row
  is empty (zeros). `arr_bg` resolves that.
- **Needs HDF5 ≥ 1.10** to read (virtual datasets); every h5py since 2.9 has it.

Measured (GB10, seed 2, 3 pT̂ windows = one background for 3 events): 629 MB shared
against 925 MB full (210 vs 308 MB per event); `arr`, `arr_bg`, the freeze-out vectors
and `diag/` bit-identical. A background copy is ~148 MB and the rest of an event ~160 MB, so
a pair file takes about **160 + 148/N MB per event** with N events per background (the full
layout 308 MB whatever N): 170 MB at N = 15, 45% less. `PairBrowser`, FNO4d's
`read_3d_data_hdf5` (lazy and not, with and without `unique_bg`), both `h5_inspect`s and
`repad_h5` give the same results on both layouts. A job without reuse writes the same
datasets, attributes and file size as before.

## Hadron level: surfaces, partons, `hadronize.py`

The hydro pair stops at the fluid. To compare the wake in hadrons, the job stores the
**input** to hadronization, not hadrons, and hadronization runs later ([`PLAN_particlize_h5.md`](../../../../docs/PLAN_particlize_h5.md)
has the reasons and the alternatives):

```bash
python run_prod_jet.py --events 10 --seed 1 --write-particlize both        # GPU
python hadronize.py out/AuAu_0_10_jet_seed0001_particlize.h5 --oversample 500 --n-frag 50
```

**`<stem>_particlize.h5`** (`jetscape.particlize_h5`, read with `ParticlizeFile`):

| | |
|---|---|
| `surface/jet/cells` | MUSIC_2's freeze-out surface, `(N, 32)` float32, columns `surface_columns`: exactly the fields iSS reads (x^μ, dσ_μ, u^μ, e, T, P, charges, μ's, π^{μν}, Π). The framework's `SurfaceCellInfo` is float, so this is lossless. One unit per event (`offsets`). |
| `surface/bg/cells`, `surface/bg/bg_id` | MUSIC_1's, one unit per **new** background (with `--reuse N`, once per N events); `events/bg_unit` says which unit an event used. |
| `partons/data` | the final partons the jet hadronization receives (`JetEnergyLossManager::GetFinalStatePartons`), `(K, 14)`: shower, pid, pstat, E, p, t, x, y, z, mass, col, acol |
| `events/` | per event: `bg_id`, `bg_unit`, `bg_key` (hash of the whole background leg), cell and parton counts, droplet energies, MUSIC τ0, boundary flags |
| attributes | `music_input` (the job's own, verbatim: iSS reads its EoS id and `Include_Bulk_Visc` flag from it), `T_fo`, `pair_file`, `file_uuid`, provenance |

**`hadronize.py`** (CPU only: no MUSIC, no GPU) writes one file per tag (`jetscape.hadrons_h5`,
read with `Hadrons.from_h5`):

| tag | from | unit |
|---|---|---|
| `bulk_jet` | iSS on `surface/jet`: bulk + wake | event |
| `bulk_bg` | iSS on `surface/bg`: bulk | background |
| `jet_frag` | `ColorlessHadronization` on `partons/` (`eCMforHadronization` 200, `take_recoil` 1) | event |

Each file keeps every sample apart (`sample_offsets`, `unit_offsets`), so averages are over
oversamples of one event, with compound-Poisson errors (`Hadrons.hist`, `Hadrons.total`).
`bulk_jet` and `jet_frag` also carry `initiators/data` `(K, 11)` + `initiators/offsets`: each
event's shower-initiating partons (`shower, pid, pstat, px, py, pz, E, x, y, z, t`), copied from
the pair file's `shower/initiators` (`HadronFile.initiators(event)`). Without the pair file
they are written without it, with a warning.
`p` and `x` are full float32 unless `--keep-bits-p/-x` rounded them; the precision is
recorded per dataset (see [Hadron precision](#hadron-precision---keep-bits-p---keep-bits-x)).
All hadrons and their positions are stored unless `--eta-max`, `--charged` or `--no-x`
selected them. The file's `eta_max`, `charged_only` and `positions` attributes record that
(see [Hadron selection](#hadron-selection---eta-max---charged---no-x)).

**Every oversample is an event on its own**, and `JetEvents` puts the tags together per event:

```python
from jetscape.hadrons_h5 import JetEvents, HadronFile, ORIGIN
with JetEvents.from_stem("out/AuAu_0_10_jet_seed0001") as je:   # the three files + particlize
    ev = je.jet_event(0, 17)          # event 0: bulk_jet oversample 17 + one fragmentation
    ev["pid"], ev["p"], ev["x"]       # p = [E, px, py, pz]; ev["origin"]: 0 bulk, 1 fragment
    bg = je.background_event(0, 17)   # oversample 17 of the background event 0 used
    for ev in je.iter_jet_events(0):  # all oversamples of event 0
        ...
with HadronFile("out/AuAu_0_10_jet_seed0001_hadrons_bulk_jet.h5") as hf:
    one = hf.sample_event(0, 17)      # any single sample, read from disk (not the whole file)
```

For many production files at once, use `HadronFileReader` (see
[C. Analysing a campaign](#c-analysing-a-campaign-hadronfilereader)).

The fragmentation paired with oversample `k` is `k mod n_frag` (or `frag_sample=`); run
`hadronize.py` with `--n-frag` equal to `--oversample` to give every oversample its own. The
background comes from the particlize file's `events/bg_unit`, so reused backgrounds resolve.
Oversamples of one event share one fluid: independent Cooper–Frye samplings, not independent
collisions.
Every unit's seed is derived from (`--seed`, the particlize file's `file_uuid`, tag, unit,
sample) and stored in `units/seed`, so any unit can be regenerated alone, and the files of a
campaign, which all get the same `--seed`, are sampled independently. Files made before that
(`seed_scheme` attribute absent or `legacy`) gave event 0 of every file the same iSS and Pythia
seeds; with `--correlated` their events are then correlated with each other. `--legacy-seeds`
reproduces them, and `--skip-complete` warns when a complete file has the other scheme. A jet event is `bulk_jet` + `jet_frag`; its background is
`bulk_bg` unit `events/bg_unit`. An empty surface (MUSIC stopped at the grid boundary) gives a
unit with no samples.

**Measured (GB10, seed 1, one 0–10% event):**
- Surfaces: 1,001,592 cells (jet) and 989,732 (background), 128 MB each raw, 77.6 MB with
  Blosc-zstd (the charge and μ columns are zero here). The particlize file is 154.5 MB, about
  half the pair file (285 MB).
- Time: 29.5 s per event without surfaces, 34.6–35.3 s with `--write-particlize both`
  (+5.4 s: the parallel surface search 1.2 s, copies of the previous time step 1.8 s, the
  extra GPU→host copies 1.2 s, the hand-off and the write ~1.4 s). With the serial surface
  finder of MUSIC4GPU before `5058545` it was 43.0 s (+13.5 s, ~9.6 s of it the search).
  The surfaces are bit-identical either way, apart from the pressure column (see
  [`PLAN_particlize_h5.md`](../../../../docs/PLAN_particlize_h5.md), *Surface finder*). Peak memory
  +0.5 GB.
- `hadronize.py`: ~6 s per surface for 100 iSS oversamples and ~8 s for 500 on one core
  (~20 s and ~30 s before [`PLAN_iSS_optim.md`](../../../../docs/PLAN_iSS_optim.md) Part A); Colorless is negligible.
- **Exact.** A `--validate-inline` job (2 events, 100 oversamples) and
  `hadronize.py --use-stored-seeds` on its particlize file give bit-identical hadrons:
  1,710,050 iSS hadrons and all Colorless fragments.

**Things to know:**
- **Colored jet hadronization is not supported** for this setup: LBT assigns no colour tags and
  the liquefier removes partons from colour chains. `hadronize.py --diagnose-colored` measures
  it; [`PLAN_particlize_h5.md`](../../../../docs/PLAN_particlize_h5.md) has the details.
- **pstat decides what is fragmented.** Colorless takes 0 (shower), 1 (recoil), 22 and −1.
  Partons the liquefier absorbed (−11), absorbed holes (−17) and the momentum missing at a
  vertex (−13) went into the droplets, so they are never hadronized twice. In the seed-1 event
  only 5 of 57 final partons survive.
- **Beam remnants.** Colorless needs an even number of string ends: with an odd number of
  quarks it adds one remnant quark (0.2, 0.2, ±√s/6 = 33 GeV along the beam), with none it adds
  two (`ColorlessHadronization.cc`). That energy is not the jet's, and the string to a remnant
  spreads hadrons across the rapidity gap, so a rapidity cut removes only part of it. The
  seed-1 event needs none: its fragments carry exactly the surviving partons' 106.3 GeV.

## How the two legs are read

- **Background (MUSIC_1) from the framework copy.** Matter, LBT and the liquefier query the
  *first* hydro through `bulk_info`, so MUSIC_1 must fill it (`<dump_hydro_only>0`), and
  filling it releases MUSIC's native store. The copy is cell for cell the native store: both
  go through `get_fluid_cell_with_index` and a float copy.
- **Jet leg (MUSIC_2) from its native store.** Every MUSIC instance reads the *first*
  `<Hydro><MUSIC>` block, so `dump_hydro_only` cannot differ per leg in XML. `PairH5Writer.attach()`
  switches it on for MUSIC_2 alone, after `Init()`.
- **One output grid.** Both legs are resampled with the same code onto the same output grid.
  `diag/frames_identical` counts the leading bit-identical frames.
  - The writer warns if it is 0: the legs started from different initial conditions.
  - It also warns if a jet leg that received droplets matches the background completely:
    MUSIC ignored the liquefier.

## Choices worth knowing

- **`<CausalLiquefier><dtau>` must equal MUSIC's `Delta_Tau`** (0.02 in `music_input`). The
  kernel is divided by `dtau` and applied in one hydro step (`CausalLiquefier.cc:117-129`).
  `run_prod_jet.py` refuses a mismatch. `dx/dy/deta` are only printed by the C++.
- **Liquefier and energy-loss parameters** are those of FastHydro's
  `AuAu_FastHydro_tune_0_10_wake.xml` (`tau_delay` 1 fm, Matter + LBT with α_s 0.25, Q0 2
  GeV), so MUSIC pairs and FastHydro wake files can be compared directly.
- **Late droplets are lost.** Once the string sources are done, MUSIC stops at freeze-out,
  whether or not droplets are still due. Anything depositing later never reaches MUSIC_2;
  `diag/E_droplets_late` says how much.
- **Seeds.** `<Random><seed>` drives 3dMCGlauber, Pythia and the energy loss, and MUSIC is
  deterministic, so a seed reproduces a pair. With a seed ≠ 0 there is no shared random
  stream: every module has its own generator, seeded from (seed, the module's **task number**)
  (`JetScapeTaskSupport.cc:104-118`), and the task number counts every task created before it.
  - The energy-loss modules are cloned per shower at the first event, and the clones get new
    task numbers (`JetEnergyLoss`'s copy constructor default-constructs its base, which
    registers a task). So adding modules at Init shifts the jets: `--validate-inline` (4 extra
    tasks) gives a different jet for the same seed (measured: seed 1, 26 droplets / 33.1 GeV
    instead of 25 / 25.0 GeV), while the background leg stays identical (989,732 surface cells
    in both). `--write-particlize` adds no task and leaves the pair file byte-identical.
  - The same seed does **not** give the same Glauber event as `prod_AuAu_0_10` (measured: seed
    1 differs at frame 0). The earlier explanation here ("the extra modules draw from the
    framework's random stream first") does not match the code; the cause is still open.
- **Speed and memory (measured, GB10, seed 1, one 0–10% event, PythiaGun 50–70 GeV, 25
  droplets).**
  - Null test (`--no-deposit`): 58 s per event, about twice the single-leg ~25 s.
  - With deposition: 60 s per event. The droplet source is computed on the CPU, but each
    step evaluates only the droplets that can deposit in it (X-SCAPE `896e3d1c`, MUSIC4GPU
    `3037be7`). Before that, every droplet was evaluated at every step and the same event
    took 184 s; the output is bit-identical.
  - Without the freeze-out surface on either leg (`--surface none`, the default): 49.1 s
    per event. With the surface on the jet leg only (`--surface jet`): 55.3 s. Both are
    bit-identical to the 60 s run, which had the surface on both legs.
  - Peak memory 16 GB.
  - Several jobs at once (after the single-job speed-ups): one job does ~118 events/h;
    `-j 2` 133, `-j 3` 155, `-j 4` 159. The GPU is the shared bottleneck (~70 % busy at
    `-j 3`/`-j 4`). With CUDA MPS (`--mps`) the jobs share it better: `-j 4 --mps` gives 179
    events/h and `-j 3 --mps` 163. **On the GB10,
    `OMP_NUM_THREADS=5 ./run_jobs.sh -j 4 --mps …` gives ~190 events/h** (recommended,
    ~66 GB). These are machine-specific: see "Recommended settings" in
    [BENCHMARK_GB10.md](../../../../docs/BENCHMARK_GB10.md) for how to find them elsewhere. Jobs no longer need to be staggered: each
    runs in its own working directory (see
    [`../prod_AuAu_0_10/README.md`](../prod_AuAu_0_10/README.md)). Details, profile and
    the former startup hang: [BENCHMARK_GB10.md](../../../../docs/BENCHMARK_GB10.md).
  - **Apple M3 Max (Metal, 16 cores, 64 GB):** 26.1 / 28.7 s per event for seed 1, one
    job ~132 events/h. Several jobs at once only pay off with the cores split between
    them: **`OMP_NUM_THREADS=5 OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0 ./run_jobs.sh -j 3
    …` gives 213 events/h (1.61×, recommended, ~40 GB)**, and `-j 4` with
    `OMP_NUM_THREADS=4` 221. With the default settings `-j 3` gives only 148 and `-j 4`
    129. `--mps` is CUDA-only. Details: [BENCHMARK_M3MAX.md](../../../../docs/BENCHMARK_M3MAX.md).

## Checks before a campaign

For a campaign with `--write-particlize`, also check the two test jobs of recipe B:

```bash
# 1. offline = in-job, bit for bit
python hadronize.py out_val/AuAu_0_10_jet_seed0900_particlize.h5 --use-stored-seeds \
       --tags bulk_jet,jet_frag --out-dir out_val/offline
python - <<'EOF'
import h5py, numpy as np, jetscape.h5_compression
for tag in ("bulk_jet", "jet_frag"):
    a = h5py.File(f"out_val/AuAu_0_10_jet_seed0900_inline_{tag}.h5")["hadrons"]
    b = h5py.File(f"out_val/offline/AuAu_0_10_jet_seed0900_hadrons_{tag}.h5")["hadrons"]
    print(tag, all(np.array_equal(a[k][:], b[k][:]) for k in ("pid", "p", "sample_offsets")))
EOF
# 2. null test: the two surfaces are identical
python -c "import numpy as np; from jetscape.particlize_h5 import ParticlizeFile as P; \
p = P('out_val/AuAu_0_10_jet_seed0901_particlize.h5'); \
print(np.array_equal(p.surface('jet', 0), p.surface('bg', 0)))"
```

Both must print `True`. (Run them from this folder with `../../python` on `PYTHONPATH`, or
from anywhere after `pip install -e ../..`.) Run check 1 without `--keep-bits-p/-x`: the
in-job hadrons it compares against are full precision.

1. `--no-deposit`, 1 event: `arr == arr_bg` and `frames_identical == ntau`.
2. 1 event with a jet: `n_droplets > 0`; the legs differ only after the first deposit; the
   difference sits along the droplets (PairBrowser plot).
3. CPU vs GPU, same seed, small grid (`MUSIC_FORCE_CPU=1` runs music4gpu's CPU path): the Δe
   maps agree.

## Potential next steps

### Switch off MUSIC's momentum-anisotropy output

On hold. Measured in [BENCHMARK_GB10.md](../../../../docs/BENCHMARK_GB10.md).

**What the code does now.** MUSIC4GPU writes these diagnostics unconditionally
(`evolve.cpp:202`, `Cell_info::output_momentum_anisotropy_vs_etas` in `grid_info.cpp`).

- **When:** at four time steps, `it = iFreezeStart`, `+10`, `+30` and `+50`.
  `iFreezeStart` is where the freeze-out check starts, which with string sources is
  shortly after the last string deposits. So these are **early times**: τ ≈ 0.46, 0.66,
  1.06 and 1.46 fm/c in our runs.
- **What it computes:** a scan of the whole grid at each η slice, written to five files
  per step:
  - `momentum_anisotropy_tau_X.dat`: ε_p vs η_s, from the ideal, ideal + shear, and full
    T^μν;
  - `eccentricities_evo_{ed,nB,nQ}_tau_X.dat`: ε_n (n = 1–6) vs η;
  - `meanpT_estimators_tau_X.dat`.
- **Volume:** 20 files per MUSIC leg, 40 per pair event.
- **On the GPU path:** before each of these steps, the current grid (primitives and
  W^μν) is copied back to the host (`sync_curr_from_gpu_readonly`).

**What omitting it would change:**

| | Effect |
|---|---|
| Hydro evolution and h5 output | None. The copy is read-only (the GPU keeps the authoritative state) and the analysis only reads the grid. |
| Time | About 1–1.5 s per event, ~3 % of the 38–47 s an event takes after the `hydro_data_optim` changes: full-grid loops at 4 steps × 2 legs plus the GPU→host copies. |
| Files lost | The 40 files per event. They are named by τ only and written into the working directory (`build_gpu`), so each event overwrites the previous one, MUSIC_2 overwrites MUSIC_1, and concurrent jobs overwrite each other. After a campaign they hold whatever the last writer left. |
| Who reads them | Nothing in X-SCAPE, js-contrib or FNO4d. MUSIC4GPU's own `tests/testIPGlasma2D/TestOutputFiles.py` expects them, so standalone MUSIC should keep the output on by default. |

**What it would take.** The same three-repo pattern as `<freeze_out_surface>`:

1. **MUSIC4GPU:** a parameter `output_momentum_anisotropy` (default 1), read from the
   input file and settable through `set_parameter`, gating that one block in
   `evolve.cpp`.
2. **X-SCAPE `MusicWrapper`:** read `<output_momentum_anisotropy>` from the XML
   (global or per MUSIC instance) and pass it with `set_parameter`. It should not go
   through `music_input`, which MUSIC reads through its slow per-parameter `StringFind4`
   (and which was shared between jobs before the per-job working directories).
3. **js-contrib:** set it to 0 in the production XMLs.

**If the numbers are wanted per event.** Store them in the h5 (e.g. `diag/`) rather than
in these files. Part of it can also be computed afterwards from `arr`: ε_n of the energy
density, and the ideal part of ε_p. The shear part cannot, because π^μν is not stored.
