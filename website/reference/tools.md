# Tools

The [`utils/`](https://github.com/jhputschke/js-contrib/tree/main/utils) folder holds the
tools around the simulations. They help you plan a production, watch it run, check its files,
and move them. Each one is a plain script, and `--help` lists all of its options.

## Planning: `prodwizard.py`

You choose a production preset and a machine. The wizard writes the launch command with the
settings measured on that machine. It estimates events per hour, duration, disk and memory
(each marked measured or estimated) and checks the plan. **It never starts jobs.**

```bash
python utils/prodwizard.py --list                                  # presets and machines
python utils/prodwizard.py --preset auau_jet --machine 2x3090 --events 2000
python utils/prodwizard.py --preset oo_hadrons --machine slurm_a100 --events 10000 --save launch_oo.sh
python utils/prodwizard_tui.py                                     # the same as an interactive wizard (Textual)
```

| Presets | Machines |
|---|---|
| `auau_jet`, `auau_particlize_only`, `auau_hadrons_only`, `oo_hadrons`, `oo_full`, `pbpb_zjet` | `2x3090`, `gb10`, `m3max`, `slurm_v100`, `slurm_a40`, `slurm_a100`, `slurm_l40s` |

## Launching

| Script | What it does |
|---|---|
| `run_jobs.sh` (in each production folder) | runs a campaign of jobs with unique seeds, several at a time on one GPU; see [the productions](../run/index.md) |
| [`launch_2gpu.sh`](https://github.com/jhputschke/js-contrib/blob/main/utils/launch_2gpu.sh), [`launch_2gpu_oo.sh`](https://github.com/jhputschke/js-contrib/blob/main/utils/launch_2gpu_oo.sh) | one Docker container per GPU on a two-GPU machine, a disk guard, and optional upload to the OSDF while it runs |
| [`slurm_prod_array.sh`](https://github.com/jhputschke/js-contrib/blob/main/utils/slurm_prod_array.sh) | stage 1 (hydro + jets) as a SLURM array in the production container, one GPU per task; `PROD=oo` for O+O |
| [`slurm_hadronize.sh`](https://github.com/jhputschke/js-contrib/blob/main/utils/slurm_hadronize.sh) | stage 2 (hadronization) as a SLURM array on a CPU partition |

See [Clusters, GPUs and monitoring](../run/clusters.md) for how they fit together.

## Monitoring: `jobmon.py`

`jobmon.py` watches a running (or finished) campaign using only what the jobs already write
(logs and `.json`). It reports jobs done, running and failed, events per hour, ETA, disk per
event, the machine's GPUs and memory, and the most frequent warnings. It also flags failed
jobs with their last log lines.

```bash
python utils/jobmon.py out/                 # one summary; exit code 1 if a job failed or is flagged
python utils/jobmon.py out/ --watch         # live view, refreshed every 10 s
python utils/jobmon.py out/ --tui           # terminal UI: sortable job table, live log, problems tab
```

[`check_deposition.py`](https://github.com/jhputschke/js-contrib/blob/main/utils/check_deposition.py)
flags Au+Au jet jobs whose deposited droplets per event fall below half the campaign's median.
This is a rare failure that leaves no log message.

## Inspecting files: `h5_inspect.py`

`h5_inspect.py` lists every group and dataset of an HDF5 file with its shape, dtype, storage
and attributes. For the js-contrib hydro formats it also summarises the grid, the axes, the
freeze-out, the `shower/`, `source/` and `diag/` tables, and flags inconsistencies. It needs
only h5py and numpy.

```bash
python utils/h5_inspect.py out/AuAu_0_10_jet_seed0001.h5           # summary + tree
python utils/h5_inspect.py FILE.h5 --stats [--event 0]              # per-feature min/max/mean, NaN/inf
python utils/h5_inspect.py FILE.h5 --attr prod_user_xml             # one attribute in full
```

## Moving files: OSDF and Google Cloud Storage

| Script | What it does |
|---|---|
| [`remote_transfer/js_osdf.py`](https://github.com/jhputschke/js-contrib/blob/main/utils/remote_transfer/js_osdf.py) | `ls`, `du`, `download` and `upload` for the Pelican namespace `osdf:///fno4hic` (reading is public; writing needs a token) |
| [`remote_transfer/js_gcs.py`](https://github.com/jhputschke/js-contrib/blob/main/utils/remote_transfer/js_gcs.py) | the same commands for a Google Cloud Storage bucket |
| [`upload_follow.py`](https://github.com/jhputschke/js-contrib/blob/main/utils/upload_follow.py) | uploads a campaign's finished jobs to the OSDF **while it runs**, and can delete the verified uploads locally |

Both transfer scripts select files by kind (`--what pair|h5|root|all`), skip files that are
already there, and check every transfer by size and CRC32C. Each one builds its own Python
environment on first use. See [Data on the OSDF](../run/osdf.md).

## Environments and builds

| Tool | What it does |
|---|---|
| [`analysis_env/setup_analysis_env.sh`](https://github.com/jhputschke/js-contrib/blob/main/utils/analysis_env/setup_analysis_env.sh), `check_env.py` | the analysis venv, and a check that opens every file of a production ([Analysis environment](../install/analysis-env.md)) |
| [`conda_install/`](https://github.com/jhputschke/js-contrib/tree/main/utils/conda_install) | creates and checks the `js_fno` conda environment ([Source build](../install/source.md)) |
| [`configure.sh`](https://github.com/jhputschke/js-contrib/blob/main/utils/configure.sh) | interactive CMake configurator for X-SCAPE + js-contrib (needs `dialog`) |
| [`gen_pyjetscape_stubs.py`](https://github.com/jhputschke/js-contrib/blob/main/utils/gen_pyjetscape_stubs.py) | regenerates the `.pyi` type stubs of `pyjetscape_core` for editors |
| `Dockerfile.dev`, `Dockerfile.prod` | the development and production images ([BuildContainerProd.md](https://github.com/jhputschke/js-contrib/blob/main/utils/BuildContainerProd.md), [BuildContainerDev.md](https://github.com/jhputschke/js-contrib/blob/main/utils/BuildContainerDev.md)) |
