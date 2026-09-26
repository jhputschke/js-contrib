# prod_AuAu_0_10 — MUSIC hydro evolutions for FNO4d, straight to HDF5

This folder produces 0–10% Au+Au 200 GeV hydro evolutions (3D MC-Glauber strings, then
MUSIC on the GPU) and writes them directly to FNO4d training HDF5 files. There is no ROOT
file, no framework copy of the evolution and no `root2hdf5` step.
`jetscape.fast_h5_bulk.H5BulkWriter` reads MUSIC's native in-memory store after each event
and resamples it onto an output grid you choose in a YAML file (default: the FNO4d grid).

| file | purpose |
|---|---|
| `AuAu_MCGlauber_MUSIC_0_10_fast.xml` | user XML: calibrated 3D-Glauber + MUSIC, hydro-only output settings |
| `grid_fno.yaml` | default output grid: the FNO4d Au+Au grid (65 × 65 × 33, τ from 0.5 in steps of 0.1) |
| `grid_x10_eta2p5.yaml` | example: the same grid cut to \|η_s\| ≤ 2.5 (17 η cells, about half the storage) |
| `run_prod.py` | one job: one seed, N events, one `.h5` file |
| `run_jobs.sh` | runs many jobs on one GPU (`-j P` at a time), one seed per job, and can resume |
| `check_output.ipynb` | checks the `.h5` output: sanity scan, ε and flow vs τ, freeze-out times, x–y viewer |

## Run

```bash
conda activate js_fno        # GB10 env; sets PYTHIA8DATA (its conda Pythia aborts the import without it)
                             # macOS: any env pyjetscape_core was built for, e.g. fno_env_mlx;
                             # Homebrew Pythia needs no PYTHIA8DATA
cd external_packages/js-contrib/contribs/PyJetscape/example/prod_AuAu_0_10

python run_prod.py --events 10 --seed 1         # -> out/AuAu_0_10_seed0001.h5
python run_prod.py --events 1 --seed 1 --grid my_grid.yaml --dry-run   # check a grid
./run_jobs.sh 20 25 1                           # 20 jobs x 25 events, seeds 1..20, into ./out
./run_jobs.sh -j 2 20 25 1                      # same, two jobs at a time (~1.7x throughput)
./run_jobs.sh -j 4 --mps 20 25 1                # four at a time, GPU shared via CUDA MPS
./run_jobs.sh 20 25 1 /data/AuAu_0_10           # same, into another directory
./run_jobs.sh 20 25 1 out_eta2p5 --grid grid_x10_eta2p5.yaml
```

You can launch it from any directory.

On macOS, `run_jobs.sh` runs under the system bash (3.2). `--mps` is CUDA-only. With
`-j` > 1, split the cores between the jobs, e.g. `OMP_NUM_THREADS=5 OMP_WAIT_POLICY=passive
KMP_BLOCKTIME=0 ./run_jobs.sh -j 3 …` on a 16-core M3 Max. Without that, the jobs' OpenMP
threads oversubscribe the cores. For the jet production this took `-j 3` from 148 to 213
events/h ([BENCHMARK_M3MAX.md](../prod_AuAu_0_10_jet/BENCHMARK_M3MAX.md)). This
production's concurrency was not measured on the Mac.

**Working directory.** Each job runs in its own working directory, `OUTDIR/work/<tag>`, and
reads the shared assets from the X-SCAPE build tree (`--build`, default `build_gpu`). This is
the Python counterpart of X-SCAPE's `examples/run_in_workdir.sh`:
- It holds a private copy of `music_input`, which MUSIC rewrites at every init.
- `XSCAPE_DATA_DIR`, `HYDROPROGRAMPATH` and `LBT_TABLES_PATH` point at the build tree, for
  `mcglauber.input`, the EoS tables and the LBT tables.
- `tables/` and the other directories the vendored code opens by relative path are
  symlinked.
- `../` paths in the main and job XML are made absolute.

So concurrent jobs share no file. In the build tree they used to share `music_input`, whose
rewrite at MUSIC init could leave a starting job spinning forever, as well as 3dMCGlauber's
and MUSIC's side files. The directory is removed after a successful job.

| option | effect |
|---|---|
| `--workdir DIR` | use `DIR` instead of `OUTDIR/work/<tag>` |
| `--keep-workdir` | keep it after a successful job (a failed job always keeps it) |
| `--in-build` | the old behaviour: run in the build tree itself |

Each job writes four files:

- `AuAu_0_10_seedNNNN.h5`: the data.
- `.xml`: the exact user XML the job ran.
- `.json`: a summary (events written, grid file, wall time).
- `.log`: only when the job is run through `run_jobs.sh`.

`run_jobs.sh` skips any seed whose `.json` already reports all events written, so you can
restart an interrupted campaign with the same command. That check looks only at the seed
and event count, so give each grid its own output directory.

To check the output, open `check_output.ipynb` in Jupyter (it reads `out/AuAu_0_10_seed*.h5`;
set `FILES` in its first cell or `PROD_H5_GLOB` for another directory). It needs only
`numpy`, `h5py`, `matplotlib`, `pandas` and `ipywidgets`, no X-SCAPE build.

Measured on the GB10 (build_gpu, music4gpu CUDA): about **25 s and about 200 MB (lzf) per
event** on the default grid (about 95 MB with `grid_x10_eta2p5.yaml`), with a peak RSS of
about 4.4 GB. The output grid barely changes the run time; MUSIC dominates. The default
compression is now Blosc-zstd, about 1.4x smaller than those lzf sizes, and
`--keep-bits 12` halves that again ([README_h5_optim.md](../../README_h5_optim.md)).

The XML now sets `<freeze_out_surface>0`: MUSIC builds no freeze-out surface, which a
hydro-only dump never uses, and stops on the equivalent max(e) < e_fo test instead. That
takes the seed-1 event from 23.3 s to **17.1 s**, with bit-identical output. It needs
X-SCAPE branch `pair_h5_music_surface_off` and MUSIC4GPU branch `XSCAPE_surface_off`; with
an older X-SCAPE, delete that line. The timings below were measured with the surface on.

Two jobs at once (`run_jobs.sh -j 2`) give about **1.66× the throughput** (17.3 s/event
overall, against 28.6 s/event for one job on events of 101–109 frames), because each job's
CPU stages (string deposition, h5 writing) overlap the other job's GPU evolution. Each job
gets about 21% slower, peak RSS stays about 4.7 GB per job, and a seed's file is
bit-identical whether it ran alone or alongside another. Ctrl-C (or `kill`) stops all
running seeds; restart with the same command to resume.

## The output grid (YAML)

```yaml
grid:
  x:   {min: -10.0, max: 10.0, n: 65}   # n cell centres, min..max inclusive
  y:   {min: -10.0, max: 10.0, n: 65}   # step = (max - min)/(n - 1)
  eta: {min: -5.0,  max: 5.0,  n: 33}   # n = 1 stores one slice (min = max)
tau:
  min: 0.5        # first frame [fm/c], must be >= every event's MUSIC tau0 (~0.4)
  dtau: 0.1       # multiples of 0.1 land on stored MUSIC frames
  max_ntau: 0     # 0 = τ axis grows (default); N > 0 pins it at N frames
```

- **Ranges and bin counts are free**, and they need not be symmetric about 0. The box must
  lie inside MUSIC's grid, which comes from the XML's `<IS>` block: x, y ∈ [−15, 14.7] fm
  (100 cells of 0.3 fm) and η_s ∈ [−6, 5.8] (60 cells of 0.2). `run_prod.py` checks this
  before it starts and refuses a box that leaves MUSIC's grid. Cells outside the source are
  written as zeros, not clamped, so a ±15 fm or ±6 request would leave a zero edge.
- **Resampling is trilinear** from MUSIC's 0.3 fm / 0.2 grid. A cell that coincides with a
  MUSIC cell is copied exactly. A cut grid on the same spacing gives exactly the same values
  as the matching cells of the full grid (checked: |η| ≤ 2.5 and η = 0 against the default
  grid, same seed, difference 0).
- **Every file trained on together must share one whole grid.** The YAML is stored in each
  file (`prod_grid_yaml` attribute) so you can check this afterwards. The τ length is the
  exception: it can differ between files and be reconciled afterwards (next section).
- The YAML must contain exactly these keys; anything else, a non-integer `n`, `max ≤ min`, or
  `n = 1` with `min ≠ max` is an error.

## τ length: grow, then repad

FNO4d's `MultiH5Array` needs every file to have the same τ length (`choose_ntau`). There are
two ways to get there, set by `tau.max_ntau`.

- **`max_ntau: 0` (default): grow, then repad.** Each file's τ axis grows to its longest
  event, and nothing is clipped. The axis stays extendible, so after the campaign you bring
  all files to one length in place:

  ```bash
  python ../../python/jetscape/repad_h5.py out/AuAu_0_10_seed*.h5                    # to the longest file
  python ../../python/jetscape/repad_h5.py out/AuAu_0_10_seed*.h5 --choose-ntau 145  # e.g. = fast_data tune
  python ../../python/jetscape/repad_h5.py out/AuAu_0_10_seed*.h5 --dry-run          # report only
  ```

  (`repad_h5.py` runs as a plain script, so it needs no install; the paths above are
  relative to this folder. With PyJetscape pip-installed, `python -m jetscape.repad_h5`
  does the same.) This is a metadata resize. No data moves, and the files don't grow, because the new frames
  are unallocated chunks that read back as exactly 0. You can pick a generous length (it's
  free) and repad again later if a new batch runs longer. `repad_to` never shrinks a file.
- **`max_ntau: N`: keep the first N frames.** Every event stores only its first N frames,
  from `tau.min` to `tau.min + (N−1)·dtau`. For example, `max_ntau: 20` keeps early times only
  (τ = 0.5 … 2.4 fm/c) at about a fifth of the storage.
  - Only those N frames are resampled. MUSIC still evolves each event to freeze-out, though,
    so the run time per event doesn't change.
  - Each job's log says `kept the first N of M frames`, and `run_jobs.sh` prints how many
    events were cut. Neither is treated as an error.
  - `ntau_freezeout` is then N (the frames stored). `tau_freezeout` still records when the
    hydro actually ended.
  - The file's τ axis is created at exactly N and fixed there, so files match as written with
    no repad step. `repad_to` can never grow them past N; to get later times, rerun with a
    larger N or 0.

0–10% events end near τ ≈ 10 fm/c, about 100 frames from τ = 0.5. The fast_data tune file
has 145 frames and a growable τ axis, so repadding the MUSIC files to 145 makes them load
together with it.

## What is written

Shown for the default `grid_fno.yaml`, the FNO4d Au+Au grid, the same as
`fastdata_AuAu200_tune_0_10.h5`:

| | |
|---|---|
| `arr` | `(nevents, 4, 65, 65, 33, ntau)` float32, channels `energy_density, vx, vy, vz`; ntau = longest event (or `max_ntau`) |
| x, y | −10 … 10 fm, dx = 0.3125 |
| η_s | −5 … 5, dη = 0.3125 |
| τ | 0.5 + k·0.1 fm/c |
| `ntau_freezeout`, `tau_freezeout` | per event; frames from `ntau_freezeout` onward are 0 |
| `diag/wall_s`, `diag/tau0_music` | per-event wall time and MUSIC start time |
| attrs | the grid, MUSIC's source grid (`*_MUSIC`), `prod_seed`, `prod_user_xml` (the full job XML), `prod_grid_yaml`, host and build |

Files from different seeds with the same grid YAML share one spatial shape. After a repad to
one τ length, FNO4d reads them as one set with no merge step:

```python
from loc_libs.read_3d_hdf5 import read_3d_data_hdf5
d = read_3d_data_hdf5(sorted(glob.glob("out/AuAu_0_10_seed*.h5")))   # lazy MultiH5Array
```

From PyJetscape, `jetscape.fast_h5_bulk.read_fast_h5_bulk(path)` returns each event as
`(ntau, nx, ny, neta, 4)`.

## Choices worth knowing

- **Centrality comes from b, not from `cenMin`/`cenMax`.** Inside X-SCAPE, 3dMCGlauber samples
  b on `[b_min, b_max]` and ignores the centrality cut, so `b_max = 4.7` fm (the geometric
  0–10%) is what selects the class. See the comment in the XML.
- **τ axis.** With dynamical strings (`InitialProfile` 131), MUSIC does not start at
  `Initial_time_tau_0`. It starts each event at
  `τ0 = floor((earliest string τ − 0.02)/0.02)·0.02` (0.4 fm/c in the test events) and stores
  a frame every 5 × `Delta_Tau` = 0.1 fm/c.
  - The default output axis starts at `τ = 0.5`, which is after every event's τ0. The log
    prints each event's τ0 and warns if one starts later than `tau.min`; its first frames
    would then be zeros.
  - When τ0 is a multiple of 0.1, every output frame is a stored MUSIC frame. Otherwise τ is
    interpolated linearly between frames 0.1 apart.
- **Seeds.** `<Random><seed>` drives 3dMCGlauber, and MUSIC itself is deterministic. A given
  seed and build reproduce an event bit for bit (checked). Give every job its own seed.
  Seed 0 means random, and `run_prod.py` refuses it.
- **Hydro only.** `dump_hydro_only = 1` leaves the framework medium empty. Adding `<Eloss>`,
  `<SoftParticlization>` or an afterburner would throw, and `run_prod.py` rejects such an XML.
  For soft hadrons from the same physics, use the source XML below with iSS.
- **Native grid.** `--native` ignores the YAML's grid and writes MUSIC's own 100 × 100 × 60
  grid (`max_ntau` still applies). That is about 10 MB per frame, roughly 1 GB per event
  before compression.

## Relation to `FastHydro/config/AuAu_MCGlauber_MUSIC_0_10.xml`

The XML here is derived from that file. The `<MCGlauber>`, `<IS>` grid, `<Preequilibrium>`
and MUSIC transport and freeze-out settings are unchanged: η/s(T), ζ/s(T), EOS 9,
T_fo = 0.15 GeV. Only output settings differ:

| | source XML (iSS reference) | this folder |
|---|---|---|
| evolution | `RootBulkWriter` via the framework copy (~65 GB RSS per event) | native store to HDF5 (~4 GB) |
| MUSIC | `output_evolution_every_N_timesteps` 1 | 5 (dtau 0.1), `dump_hydro_only` 1, `skip_surface` 1, `freeze_out_surface` 0 |
| particlization | iSS plus final-state hadron writer | none (hydro only) |

## Build requirement

`build_gpu` needs lib3dMCGlb linked with `-Wl,-Bsymbolic`. The top-level `CMakeLists.txt`
does this, next to the same fix for libiSS.

The reason: 3dMCGlauber and music4gpu both export an unnamespaced `pretty_ostream` class, and
music4gpu's version is larger. `pyjetscape_core` loads libmusic before lib3dMCGlb. Without the
flag, 3dMCGlauber's objects are built by music4gpu's constructor, and the job aborts with
`free(): invalid pointer` right after 3dMCGlauber prints its parameter list. `runJetscape`
loads the libraries in the opposite order, so it never showed the problem.
