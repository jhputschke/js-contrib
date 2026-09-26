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
| `hadronize.xml` | the iSS and jet-hadronization settings (used by `hadronize.py` and `--validate-inline`) |
| `jet_wake.ipynb` | one pair, from the energy density (§1–9) to hadrons (§10) |
| `PLAN_particlize_h5.md` | design, decisions and validation of the hadron-level path |

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
./run_jobs.sh 20 25 1                                        # 20 jobs x 25 events
./run_jobs.sh -j 2 20 25 1 out_pgun --hard pgun
./run_jobs.sh -j 4 --mps 20 25 1                             # 4 at a time, GPU shared via CUDA MPS

# macOS (Metal): split the cores between the jobs, or -j 3 gains nothing (BENCHMARK_M3MAX.md)
OMP_NUM_THREADS=5 OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0 ./run_jobs.sh -j 3 20 25 1
```

Each job writes the following, next to each other:
- `AuAu_0_10_jet_seedNNNN.h5`: the data.
- `.xml`: the exact job XML.
- `.json`: a summary.
- `.log`: only when the job is run through `run_jobs.sh`.

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
| `--write-particlize {none,jet,both}` | Also write `<stem>_particlize.h5`: the jet leg's surface (`jet`) or both legs' (`both`, the background once per background), plus the final partons. Switches on those legs' surfaces (and their hand-off to the framework) on top of `--surface`. The pair file is unchanged (checked byte for byte). Costs +13.5 s per event for `both` (measured, below). |
| `--validate-inline` | Validation only: also run iSS on the jet leg and Colorless jet hadronization inside the job and store their hadrons and seeds (`<stem>_inline_{bulk_jet,jet_frag}.h5`), for `hadronize.py --use-stored-seeds`. **Changes the jet sample** of the seed (see Seeds below); the background is unchanged. |
| `--hadronize-xml FILE` | Settings for `--validate-inline` (default `hadronize.xml`). |
| `--surface {none,bg,jet,both}` | Which legs build MUSIC's freeze-out surface. It is needed only to particlize a leg, e.g. `jet` for hadrons from the jet leg. `none` (default) is ~6 s per MUSIC run faster, with a bit-identical evolution. It sets `<freeze_out_surface>` in the first `<Hydro><MUSIC>` block (background, and the default) and in MUSIC_2's own block (jet leg). |

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
- Time: 29.5 s per event without surfaces, 43.0 s with `--write-particlize both` (+13.5 s:
  ~12 s for MUSIC to build and hand over the two surfaces, 1.35 s for the write). Peak memory
  +0.5 GB.
- `hadronize.py`: ~20 s per surface for 100 iSS oversamples and ~30 s for 500 on the CPU
  (most of it is fixed cost); Colorless is negligible.
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

For a campaign with `--write-particlize`, also: one `--validate-inline` job and
`hadronize.py --use-stored-seeds` on it must give bit-identical hadrons, and a `--no-deposit`
job must give `surface/jet` = `surface/bg` cell for cell.

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
