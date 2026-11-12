# Tools

The [`utils/`](https://github.com/jhputschke/js-contrib/tree/main/utils) folder holds the
tools around the simulations. They help you plan a production, watch it run, check its files,
and move them. Each one is a plain script, and `--help` lists all of its options.

!!! info "On a Mac: commands of the xscape wheel"
    From 0.2.0 the [xscape wheel](../install/mac.md) ships `utils/` and has the tools as
    commands, run in the wheel's environment with the same options (`--help` shows them under
    the command's name). `xscape-utils DEST` copies `utils/` to read or adapt.

    | command | script |
    |---|---|
    | `xscape-prodwizard` | `prodwizard.py` (machine `mac_wheel`) |
    | `xscape-jobmon` | `jobmon.py` |
    | `xscape-check-deposition` | `check_deposition.py` |
    | `xscape-h5-inspect` | `h5_inspect.py` |
    | `xscape-osdf` | `remote_transfer/js_osdf.py`, in the wheel's environment (no `~/.cache/js_osdf`) |
    | `xscape-upload-follow` | `upload_follow.py`, likewise |
    | `xscape-gcs` | `remote_transfer/js_gcs.py`, with its own environment in `~/.cache/js_gcs` |
    | `xscape-selftest` | `container/image_check.py --analysis`, plus the LBT tables |

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
| `auau_jet`, `auau_particlize_only`, `auau_hadrons_only`, `oo_hadrons`, `oo_full`, `pbpb_zjet` | `2x3090`, `gb10`, `m3max`, `mac_wheel`, `slurm_v100`, `slurm_a40`, `slurm_a100`, `slurm_l40s` |

`m3max` is a Mac with a js-contrib checkout and the `fno_env_mlx` conda environment.
`mac_wheel` is a Mac with the xscape wheel: the commands activate `$VIRTUAL_ENV` (default
`~/.venvs/xscape`), run `run_jobs.sh` in your copy of the examples, `$XSCAPE_EXAMPLES`
(default `~/xscape/examples`), and watch it with `xscape-jobmon`. Run inside the wheel's
environment, the wizard picks `mac_wheel` by itself; the settings are the M3 Max's.

```bash
xscape-prodwizard --preset auau_jet --machine mac_wheel --events 300    # on a Mac with the wheel
```

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
xscape-jobmon out/ --tui                    # the same with the Mac wheel
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
xscape-h5-inspect FILE.h5                                           # the Mac wheel's command
```

## Comparing files: `h5_compare.py`

[`h5_compare.py`](https://github.com/jhputschke/js-contrib/blob/main/utils/h5_compare.py)
compares two HDF5 files, or every `*.h5` of the same name in two directories. It is the check
after a change that should not alter the output: run the same job with the same seed before and
after, and compare.

```bash
python utils/h5_compare.py A.h5 B.h5                               # datasets and attributes that differ
python utils/h5_compare.py before/ after/ --ignore-stamps          # two directories; run stamps skipped
python utils/h5_compare.py A.h5 B.h5 --ignore 'diag/*' --no-attrs  # skip by pattern; datasets only
python utils/h5_compare.py A.h5 B.h5 --rtol 1e-6                   # floats may differ by 1e-6 (relative)
python utils/h5_compare.py A.h5 B.h5 -q && echo same               # exit code: 0 same, 1 different, 2 error
```

The comparison is bit-exact unless you give a tolerance: a dataset is the same only if shape,
dtype and every byte agree. For a dataset that differs, the script prints how many values
differ, the largest absolute and relative difference, and the index of the first one. For an
attribute it prints both values.

For example, a hadron file against a copy with three momenta, one cell count and one attribute
changed:

```text
datasets:   10 in A, 10 in B; 8 identical, 2 differ
  hadrons/p        3 of 1438404 values differ, max |a-b| 9.83e-06, max rel 1e-05, first at [1000, 0]
  units/n_cells    1 of 1 values differ, max |a-b| 1, max rel 1.01e-06, first at [0]
attributes: 25 in A, 27 in B; 24 identical, 1 differ
  @oversample      only in B: 101
ignored:    2 (-v lists them)
AuAu_0_10_jet_pth10-40_eta06_c1_0013_hadrons_bulk_bg.h5: DIFFERENT (3 differences)
```

Two runs of the same job never agree in everything: the wall times, the UUIDs that tie the
files of a job together, the host name and the dates differ. `--ignore-stamps` skips these
(`*wall_s*`, `*@*uuid*`, `*@prod_host`, `*@created`, `*@updated`). `--ignore PATTERN` skips
more; a dataset is named by its path (`diag/wall_s`) and an attribute by `PATH@NAME`
(`@nevents` for a file attribute). Large datasets are read in blocks, so evolution files of
several GB need about 1 GB of memory. The script needs only h5py and numpy, plus hdf5plugin
for Blosc-compressed files.

## Moving files: OSDF and Google Cloud Storage

| Script | What it does |
|---|---|
| [`remote_transfer/js_osdf.py`](https://github.com/jhputschke/js-contrib/blob/main/utils/remote_transfer/js_osdf.py) | `ls`, `du`, `download` and `upload` for the Pelican namespace `osdf:///fno4hic` (reading is public; writing needs a token) |
| [`remote_transfer/js_gcs.py`](https://github.com/jhputschke/js-contrib/blob/main/utils/remote_transfer/js_gcs.py) | the same commands for a Google Cloud Storage bucket |
| [`upload_follow.py`](https://github.com/jhputschke/js-contrib/blob/main/utils/upload_follow.py) | uploads a campaign's finished jobs to the OSDF **while it runs**, and can delete the verified uploads locally |

Both transfer scripts select files by kind (`--what pair|h5|root|all`), skip files that are
already there, and check every transfer by size and CRC32C. Each one builds its own Python
environment on first use; the Mac wheel's `xscape-osdf` and `xscape-upload-follow` use the
wheel's environment instead. See [Data on the OSDF](../run/osdf.md).

## Environments and builds

| Tool | What it does |
|---|---|
| [`container/image_check.py`](https://github.com/jhputschke/js-contrib/blob/main/utils/container/image_check.py) | the production image's check: the bindings, HDF5 filters, pelicanfs, paths; `--analysis` the analysis packages, one synthetic event and MUSIC's EoS table (`$MUSIC_EOS_TABLE`, else the build's); `xscape-selftest` on the Mac |
| [`analysis_env/setup_analysis_env.sh`](https://github.com/jhputschke/js-contrib/blob/main/utils/analysis_env/setup_analysis_env.sh), `check_env.py` | the analysis venv, and a check that opens every file of a production ([Analysis environment](../install/analysis-env.md)) |
| [`conda_install/`](https://github.com/jhputschke/js-contrib/tree/main/utils/conda_install) | creates and checks the `js_fno` conda environment ([Source build](../install/source.md)) |
| [`configure.sh`](https://github.com/jhputschke/js-contrib/blob/main/utils/configure.sh) | interactive CMake configurator for X-SCAPE + js-contrib (needs `dialog`) |
| [`gen_pyjetscape_stubs.py`](https://github.com/jhputschke/js-contrib/blob/main/utils/gen_pyjetscape_stubs.py) | regenerates the `.pyi` type stubs of `pyjetscape_core` for editors |
| `Dockerfile.dev`, `Dockerfile.prod` | the development and production images ([BuildContainerProd.md](https://github.com/jhputschke/js-contrib/blob/main/utils/BuildContainerProd.md), [BuildContainerDev.md](https://github.com/jhputschke/js-contrib/blob/main/utils/BuildContainerDev.md)) |
