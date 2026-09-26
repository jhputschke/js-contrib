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
> jet leg is identical to the background. Plan and status: `../../PLAN_pair_h5_music.md`.
>
> **Hadron level (2026-09-26, branches `surface_to_hadrons` in X-SCAPE and js-contrib).**
> `--write-particlize` stores both freeze-out surfaces and the final partons next to the pair;
> `hadronize.py` turns them into iSS and Colorless hadrons offline, bit-identical to running
> them inside the job. See [Hadron level](#hadron-level-surfaces-partons-hadronizepy) and
> `PLAN_particlize_h5.md`.

| file | purpose |
|---|---|
| `AuAu_MCGlauber_MUSIC_0_10_jet.xml` | user XML: the prod physics plus `Hard`, `Liquefier`, `Eloss` and a second `Hydro` (MUSIC_2) |
| `run_prod_jet.py` | one job: one seed, N events, one `.h5` file |
| `run_jobs.sh` | many jobs, `-j P` at a time, resumable (wraps `../prod_AuAu_0_10/run_jobs.sh`) |
| `hadronize.py` | offline: iSS on the stored surfaces, Colorless on the stored partons → hadron files |
| `run_hadronize.py` | `hadronize.py` over a whole campaign, `-j P` at a time; `--follow` runs it alongside `run_jobs.sh` |
| `hadronize.xml` | the iSS and jet-hadronization settings (used by `hadronize.py` and `--validate-inline`) |
| `jet_wake.ipynb` | one pair, from the energy density (§1–9) to hadrons (§10) |
| `PLAN_particlize_h5.md` | design, decisions and validation of the hadron-level path |
| `PLAN_iSS_optim.md` | for later: faster iSS (bit-identical) and correlated jet/background sampling |

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
python run_prod_jet.py --events 1 --seed 1 --dry-run         # check the XML/grid only

# + the input for hadronization: both freeze-out surfaces and the final partons
python run_prod_jet.py --events 10 --seed 1 --write-particlize both
python hadronize.py out/AuAu_0_10_jet_seed0001_particlize.h5 --oversample 500 --n-frag 50
python run_hadronize.py out -j 4 --oversample 500 --n-frag 50    # every particlize file in out/

./run_jobs.sh 20 25 1                                        # 20 jobs x 25 events
./run_jobs.sh -j 2 20 25 1 out_pgun --hard pgun
./run_jobs.sh -j 4 --mps 20 25 1                             # 4 at a time, GPU shared via CUDA MPS
./run_jobs.sh -j 4 --mps 20 25 1 out_had --write-particlize both   # + hadronization input

# GB10 (CUDA): 4 jobs sharing the GPU through MPS, the cores split between them (BENCHMARK_GB10.md)
OMP_NUM_THREADS=5 ./run_jobs.sh -j 4 --mps 20 25 1

# macOS (Metal): split the cores between the jobs, or -j 3 gains nothing (BENCHMARK_M3MAX.md)
OMP_NUM_THREADS=5 OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0 ./run_jobs.sh -j 3 20 25 1
```

**Recommended campaign settings** (measured, hydro pairs only; machine-specific, so they are
not built into the scripts):

| machine | campaign | events/h | memory | one job alone | details |
|---|---|---|---|---|---|
| GB10 (CUDA, 20 cores, 121 GB) | `OMP_NUM_THREADS=5 ./run_jobs.sh -j 4 --mps ...` | ~190 | ~66 GB | defaults (all threads), 118 events/h | [BENCHMARK_GB10.md](BENCHMARK_GB10.md) |
| Apple M3 Max (Metal, 16 cores, 64 GB) | `OMP_NUM_THREADS=5 OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0 ./run_jobs.sh -j 3 ...` | 213 | ~40 GB | defaults, 132 events/h | [BENCHMARK_M3MAX.md](BENCHMARK_M3MAX.md) |

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
  [Finding the settings on another machine](BENCHMARK_GB10.md#finding-the-settings-on-another-machine).

Each job writes the following, next to each other:
- `AuAu_0_10_jet_seedNNNN.h5`: the data.
- `AuAu_0_10_jet_seedNNNN_particlize.h5`: with `--write-particlize` only, the input for
  `hadronize.py` (see [Hadron level](#hadron-level-surfaces-partons-hadronizepy)).
- `.xml`: the exact job XML.
- `.json`: a summary.
- `.log`: only when the job is run through `run_jobs.sh`.

`hadronize.py` then writes `AuAu_0_10_jet_seedNNNN_hadrons_{bulk_jet,bulk_bg,jet_frag}.h5`
next to the particlize file. For many seeds at once, see the next section.

## Campaigns with `run_jobs.sh`

`run_jobs.sh` runs many `run_prod_jet.py` jobs, one seed and one output file per job:

```bash
./run_jobs.sh [-j P] [--mps] NJOBS EVENTS_PER_JOB FIRST_SEED [OUTDIR] [run_prod_jet.py options ...]
```

- Seeds `FIRST_SEED .. FIRST_SEED+NJOBS-1`, `EVENTS_PER_JOB` events each. The files are
  `OUTDIR/AuAu_0_10_jet_seedNNNN.*` (default `OUTDIR` is `./out`).
- `-j P` keeps P jobs running at once; `--mps` lets them share the GPU through CUDA MPS
  (CUDA only). Every job runs in its own working directory (`OUTDIR/work/<tag>`, removed when
  it succeeds), so the jobs can start together. Output is bit-identical per seed whatever
  `-j`.
- Everything after `OUTDIR` goes to every job unchanged (`--write-particlize`, `--reuse`,
  `--hard`, `--grid`, ...). Don't pass `--events`, `--seed` or `--outdir`: the script sets them.
- Each job's output goes to `OUTDIR/<tag>.log`. A failed job is reported and the others
  continue.
- **End marker.** When a campaign ends (not on Ctrl-C), `OUTDIR/run_jobs.finished` is
  written: this is how `run_hadronize.py --follow` knows no more files are coming. A new
  campaign in the same `OUTDIR` removes it first.
- **Restarting.** A seed whose `.json` says all events were written is skipped (with
  `--write-particlize`, its particlize file must be complete too). So an interrupted campaign
  resumes with the same command, and re-running it only redoes failed or missing seeds.
- Give every distinct setting its own `OUTDIR`: the skip test looks only at the seed and the
  event count, not at the options.

### A. Hydro pairs only (FNO training data)

```bash
conda activate js_fno
./run_jobs.sh 1 1 1 out_null --no-deposit                  # first: null test, arr == arr_bg
OMP_NUM_THREADS=5 ./run_jobs.sh -j 4 --mps 20 25 1 out     # 20 seeds x 25 events
```

On the GB10, `OMP_NUM_THREADS=5 ... -j 4 --mps` gives about 190 events/h
([BENCHMARK_GB10.md](BENCHMARK_GB10.md); on macOS see the Metal line under *Run*). No
freeze-out surface is built (`--surface none`, the default), which is the fastest setting.

### B. Hydro pairs + hadronization input

The same, plus `--write-particlize`. The pair files are unchanged, byte for byte, so they can
also serve as training data:

```bash
# first: one validation job, then one null-test job (see "Checks before a campaign")
python run_prod_jet.py --events 2 --seed 900 --write-particlize both --validate-inline --outdir out_val
python run_prod_jet.py --events 1 --seed 901 --write-particlize both --no-deposit --outdir out_val

# the campaign: both surfaces + final partons every event
OMP_NUM_THREADS=5 ./run_jobs.sh -j 4 --mps 20 25 1 out_had --write-particlize both

# one background per 3 jets: the background surface is stored once per background
OMP_NUM_THREADS=5 ./run_jobs.sh -j 4 --mps 20 30 1 out_had_reuse3 --write-particlize both --reuse 3
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
OMP_NUM_THREADS=5 ./run_jobs.sh -j 4 --mps 20 25 1 out_had --write-particlize both &
python run_hadronize.py out_had -j 3 --follow --oversample 500 --n-frag 50

# a --reuse campaign: give each background N x the jet leg's oversamples (the optimal split)
python run_hadronize.py out_had_reuse3 -j 8 --oversample 200 --oversample-bg auto --n-frag 50

python run_hadronize.py out_had --dry-run --oversample 500    # what it would do
```

- **Inputs** are directories, particlize files or globs. Every option it doesn't know goes to
  each `hadronize.py` unchanged. These are checked once before anything starts.
- **Only complete inputs.** A particlize file being written by a running job is not touched.
  Without `--follow` it is reported as incomplete; with `--follow` it is picked up once its
  job has finished.
- **Restarting.** `--skip-complete` is passed unless you give `--force`. Files whose outputs
  are all complete are skipped without starting a process; missing or incomplete outputs are
  redone. Re-running the same command finishes an interrupted pass, and on a finished
  campaign it returns at once.
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
| `--n-frag K` | Colorless fragmentations per event. With `K` equal to `N` every oversample gets its own fragmentation (`JetEvents.jet_event`) |
| `--tags` | a subset of `bulk_jet,bulk_bg,jet_frag`, e.g. `--tags jet_frag` to redo only the fragments with other settings |
| `--seed` | base seed; every unit's seed derives from it and is stored in `units/seed` |

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
- **Averages event by event.** For every event, the samples of its unit are histogrammed and
  divided by that unit's number of samples. These per-event means are then averaged over the
  events, so every event counts the same, even when files were hadronized with different
  `--oversample`. Errors are compound-Poisson per event, added over events.
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
pair file's `shower/` group, so keep the pair files next to the particlize files.

**Option if needed: merging into single files.** A campaign could also be merged into one
file per tag (`merge_hadrons.py`, not written). It would concatenate the units, shift the
offsets, add `seed`/`source_event` columns, and renumber backgrounds globally together with a
merged event → background map. That only helps for moving or archiving a campaign as three
files instead of 4 × N_seeds: a merged file at 500 oversamples is ~50 GB per 500 events, and
the reader above makes it unnecessary for analysis.

### Planning numbers (GB10, one 0–10% event, measured)

| | per event | disk per event |
|---|---|---|
| A: hydro pair (`grid_fno.yaml`, Blosc-zstd) | 29.5 s alone; ~190 events/h with `-j 4 --mps` | 285 MB |
| B: + `--write-particlize both` | 34.6–35.3 s alone (+5.4 s: MUSIC builds and hands over the two surfaces; +13.5 s before MUSIC4GPU `5058545`); `-j` throughput not measured | + 154 MB |
| B with `--reuse N` | the background surface once per N events | + 78 MB + 78/N MB |
| `hadronize.py`, both legs, 500 oversamples, 50 fragmentations | ~14 s on one core, ~10 s with `OMP_NUM_THREADS=5` (per surface ~5 s fixed + ~7 ms per oversample); ~58 s before `PLAN_iSS_optim.md` Part A | ~100 MB per leg (~0.2 MB per oversample) |

Peak memory: +0.5 GB per production job with surfaces; `hadronize.py` ~1.4 GB per surface
up to ~1000 oversamples (1.6 GB for both legs), 1.7 GB at 2000 (it was 2.5 GB at 500 and
3.9 GB at 1000 before the hadrons went to numpy as arrays). Three or four `hadronize.py`
processes keep up with a whole four-job GPU campaign.

## Options

| option | effect |
|---|---|
| `--hard pythia` (default) | PythiaGun. The vertex is a 3dMCGlauber binary-collision point (x, y; z = t = 0). `--pthat-min/--pthat-max` override the XML's 50–70 GeV. |
| `--hard pgun --pgun-pt P` | One parton at fixed pT, always from the origin: PGun zeroes the sampled vertex (`PGun.cc:117-120`). |
| `--reuse N` | `setReuseHydro`: MUSIC_1 runs once per N events, MUSIC_2 every event (a new jet each time). `arr_bg` is still written per event, so it stays aligned with `arr`. `diag/bg_id` gives the first event that used each background. |
| `--no-deposit` | Null test: MUSIC_2 without the liquefier. `arr` must equal `arr_bg` bit for bit (`diag/frames_identical == ntau`). Showers and droplets are still recorded. |
| `--native` | Both legs on MUSIC's own grid (100 × 100 × 60) instead of the YAML's. |
| `--workdir DIR` / `--keep-workdir` / `--in-build` | The job's working directory, as in `../prod_AuAu_0_10` (default `OUTDIR/work/<tag>`, removed after a successful job). |
| `--no-showers` | Skip `shower/`. |
| `--write-particlize {none,jet,both}` | Also write `<stem>_particlize.h5`: the jet leg's surface (`jet`) or both legs' (`both`, the background once per background), plus the final partons. Switches on those legs' surfaces (and their hand-off to the framework) on top of `--surface`. The pair file is unchanged (checked byte for byte). Costs +5.4 s per event for `both` (+13.5 s before MUSIC4GPU `5058545`; measured, below). |
| `--validate-inline` | Validation only: also run iSS on the jet leg and Colorless jet hadronization inside the job and store their hadrons and seeds (`<stem>_inline_{bulk_jet,jet_frag}.h5`), for `hadronize.py --use-stored-seeds`. **Changes the jet sample** of the seed (see Seeds below); the background is unchanged. |
| `--hadronize-xml FILE` | Settings for `--validate-inline` (default `hadronize.xml`). |
| `--surface {none,bg,jet,both}` | Which legs build MUSIC's freeze-out surface (`<freeze_out_surface>` in the first `<Hydro><MUSIC>` block, i.e. the background and the default, and in MUSIC_2's own block). On its own it produces nothing: only `--write-particlize` hands a surface to the framework and stores it, and it builds its legs itself. So a leg built but not stored costs ~3 s per MUSIC run (~6 s before MUSIC4GPU `5058545`) for no output, and the job warns about it. `none` (default) gives a bit-identical evolution. |

The job XML always contains **one** hard process. The automatic task list would run every
`<Hard>` child it finds, so the driver rebuilds that block from the option.

## What is written

On top of the single-leg schema (`arr`, `ntau_freezeout`, `tau_freezeout`, grid attributes):

| | |
|---|---|
| `arr` | the **jet leg** (MUSIC_2), `(nevents, 4, nx, ny, neta, ntau)` float32, channels `energy_density, vx, vy, vz` |
| `arr_bg`, `ntau_freezeout_bg`, `tau_freezeout_bg` | the **background leg** (MUSIC_1). Always the same shape as `arr`. The τ axis grows to the longer leg of any event, and each leg is exactly 0 after its own freeze-out. |
| `source/droplets`, `source/offsets` | the droplets MUSIC_2 was given, `(M, 8)`: `tau, x, y, eta, E, px, py, pz` (Milne position, Cartesian momentum). Event `i` is rows `offsets[i]:offsets[i+1]`. Each droplet deposits at `tau + liquefier_tau_delay`. |
| `shower/` | partons, vertices and initiators per event, as FastHydro writes them (`jetscape.showers`) |
| `diag/` | `n_droplets`, `E_droplets`, `n/E_droplets_late` (deposit after the jet leg froze out), `n/E_droplets_early`, `n_showers`, `n_partons`, `tau0_music`, `ntau_jet`, `ntau_bg`, `bg_id`, `frames_identical`, `wall_s` |
| attributes | `pairing = "bg_jet"`, `arr_is`, `arr_bg_is`, `deposition`, `source_model`, `hard_vertex`, `liquefier_{dtau, tau_delay, time_relax, d_diff, width_delta, c_diff, gamma_relax}`, `freezeout_convention_id = "frames_written"`, and the run provenance (`prod_seed`, `prod_user_xml`, `prod_grid_yaml`, `prod_hard`, `prod_reuse`, ...) |

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
It grows `arr` and `arr_bg` together:

```bash
python ../../python/jetscape/repad_h5.py out/AuAu_0_10_jet_seed*.h5
```

## Hadron level: surfaces, partons, `hadronize.py`

The hydro pair stops at the fluid. To compare the wake in hadrons, the job stores the
**input** to hadronization, not hadrons, and hadronization runs later (`PLAN_particlize_h5.md`
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
Every unit's seed is derived from (`--seed`, tag, unit, sample) and stored in `units/seed`, so
any unit can be regenerated alone. A jet event is `bulk_jet` + `jet_frag`; its background is
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
  `PLAN_particlize_h5.md`, *Surface finder*). Peak memory
  +0.5 GB.
- `hadronize.py`: ~6 s per surface for 100 iSS oversamples and ~8 s for 500 on one core
  (~20 s and ~30 s before `PLAN_iSS_optim.md` Part A); Colorless is negligible.
- **Exact.** A `--validate-inline` job (2 events, 100 oversamples) and
  `hadronize.py --use-stored-seeds` on its particlize file give bit-identical hadrons:
  1,710,050 iSS hadrons and all Colorless fragments.

**Things to know:**
- **Colored jet hadronization is not supported** for this setup: LBT assigns no colour tags and
  the liquefier removes partons from colour chains. `hadronize.py --diagnose-colored` measures
  it; `PLAN_particlize_h5.md` has the details.
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
    [BENCHMARK_GB10.md](BENCHMARK_GB10.md) for how to find them elsewhere. Jobs no longer need to be staggered: each
    runs in its own working directory (see
    [`../prod_AuAu_0_10/README.md`](../prod_AuAu_0_10/README.md)). Details, profile and
    the former startup hang: [BENCHMARK_GB10.md](BENCHMARK_GB10.md).
  - **Apple M3 Max (Metal, 16 cores, 64 GB):** 26.1 / 28.7 s per event for seed 1, one
    job ~132 events/h. Several jobs at once only pay off with the cores split between
    them: **`OMP_NUM_THREADS=5 OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0 ./run_jobs.sh -j 3
    …` gives 213 events/h (1.61×, recommended, ~40 GB)**, and `-j 4` with
    `OMP_NUM_THREADS=4` 221. With the default settings `-j 3` gives only 148 and `-j 4`
    129. `--mps` is CUDA-only. Details: [BENCHMARK_M3MAX.md](BENCHMARK_M3MAX.md).

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
from anywhere after `pip install -e ../..`.)

1. `--no-deposit`, 1 event: `arr == arr_bg` and `frames_identical == ntau`.
2. 1 event with a jet: `n_droplets > 0`; the legs differ only after the first deposit; the
   difference sits along the droplets (PairBrowser plot).
3. CPU vs GPU, same seed, small grid (`MUSIC_FORCE_CPU=1` runs music4gpu's CPU path): the Δe
   maps agree.

## Potential next steps

### Switch off MUSIC's momentum-anisotropy output

On hold. Measured in [BENCHMARK_GB10.md](BENCHMARK_GB10.md).

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
