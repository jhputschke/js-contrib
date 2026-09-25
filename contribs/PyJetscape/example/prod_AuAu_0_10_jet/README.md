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

| file | purpose |
|---|---|
| `AuAu_MCGlauber_MUSIC_0_10_jet.xml` | user XML: the prod physics plus `Hard`, `Liquefier`, `Eloss` and a second `Hydro` (MUSIC_2) |
| `run_prod_jet.py` | one job: one seed, N events, one `.h5` file |
| `run_jobs.sh` | many jobs, `-j P` at a time, resumable (wraps `../prod_AuAu_0_10/run_jobs.sh`) |

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
| `--no-showers` | Skip `shower/`. |
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
  deterministic, so a seed reproduces a pair. The same seed does **not** give the same Glauber
  event as `prod_AuAu_0_10`: the extra modules draw from the framework's random stream first,
  so the background leg is a different event from the hydro-only file's (measured: seed 1
  differs at frame 0).
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
  - Several jobs at once: `-j 3` gives about 2× the throughput and `-j 4` about 2.3×. Start them
    about 20 s apart, or they can hang at `Initialize MUSIC` (shared `music_input` rewrite).
    Details, profile and the bug: [BENCHMARK_GB10.md](BENCHMARK_GB10.md).

## Checks before a campaign

1. `--no-deposit`, 1 event: `arr == arr_bg` and `frames_identical == ntau`.
2. 1 event with a jet: `n_droplets > 0`; the legs differ only after the first deposit; the
   difference sits along the droplets (PairBrowser plot).
3. CPU vs GPU, same seed, small grid (`MUSIC_FORCE_CPU=1` runs music4gpu's CPU path): the Δe
   maps agree.
