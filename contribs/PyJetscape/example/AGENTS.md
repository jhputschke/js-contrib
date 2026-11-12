# X-SCAPE / js-contrib examples: notes for AI coding assistants

This directory holds the PyJetscape examples of [js-contrib](https://github.com/jhputschke/js-contrib):
GPU-accelerated X-SCAPE heavy-ion simulations driven from Python. It is either a user's copy
made with `xscape-examples DIR` (production container, Mac wheel) or the folder
`contribs/PyJetscape/example` of a source checkout. The user is usually a physicist who wants
to run events, a production, or an analysis of production files, not to develop X-SCAPE.

The full documentation is the website, <https://jhputschke.github.io/js-contrib/>. Its index
for language models is <https://jhputschke.github.io/js-contrib/llms.txt> (every page as
Markdown, one line each); read the page for a topic before you guess an option or a file layout.

## 1. Find out which installation this is

Ask the user if it is not clear; the commands differ.

| installation | how to recognise it | how a command runs |
|---|---|---|
| **production container** (Linux, NVIDIA GPU) | `docker` or `apptainer` on the host, an image `jhputschke/xscape-prod:cuNNN` or a `.sif` file; `import jetscape` fails on the host | inside the container only, see below |
| **Mac wheel** (Apple Silicon) | `~/.venvs/xscape-VER`, commands `xscape-selftest`, `xscape-run` | `source ~/.venvs/xscape-VER/bin/activate`, then `python script.py` or `xscape-*` |
| **source build** | this directory is inside `X-SCAPE/external_packages/js-contrib` | in the environment X-SCAPE was built in (often conda `js_fno`), build in `X-SCAPE/build_gpu` |
| **analysis only** | no X-SCAPE; a venv with `pyjetscape[analysis]` | reads files; cannot run events or `hadronize.py` |

**Container.** You run on the host, X-SCAPE runs in the image. Wrap every command. `WORK` is
the host directory that is `/work` inside: if this copy was made with
`xscape-examples /work/examples`, it is this directory's parent.

```bash
# Docker: --gpus all for the GPU, --user so that the outputs belong to the user
docker run --rm --gpus all --user "$(id -u):$(id -g)" -v "$WORK:/work" -w /work/examples/pythonic \
  jhputschke/xscape-prod:cu126 python oo_workflow.py --events 1 --outdir /work/out_oo
# Apptainer: --nv for the GPU, --pwd for the working directory
apptainer exec --nv --pwd /work/examples/pythonic --bind "$WORK:/work" xscape_prod.sif \
  python oo_workflow.py --events 1 --outdir /work/out_oo
```

- The image is read-only: the tree is `/opt/X-SCAPE`, the examples in the image are
  `/opt/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape/example`, js-contrib's tools
  `/opt/X-SCAPE/external_packages/js-contrib/utils`. Write only below `/work`.
- Without `--gpus all` / `--nv` MUSIC falls back to the CPU: about 10 times slower, and the
  numbers are not the GPU's. Never mix such output into a GPU campaign.
- Use the image tag the user already has (`cu124`, `cu126`, `cu130`); do not pull another one
  unasked.

**Mac wheel.** MUSIC runs on the Mac's GPU without a setting. Tools are commands:
`xscape-run`, `xscape-prodwizard`, `xscape-jobmon`, `xscape-h5-inspect`, `xscape-osdf`,
`xscape-selftest`, `xscape-fetch-data`; `xscape-utils DIR` copies js-contrib's `utils/`. The
wheel has no ROOT, PyTorch or SMASH.

## 2. First check: one event

Before anything else, see that one event runs (under a minute with a GPU):

```bash
cd pythonic && python oo_workflow.py --events 1 --outdir OUT     # one O+O event, all stages
```

Then, for the Au+Au jet production (a few minutes):

```bash
cd prod_AuAu_0_10_jet
python run_prod_jet.py --events 2 --seed 1 --write-particlize both --outdir OUT/out   # GPU
python run_hadronize.py OUT/out -j 4 --oversample 500 --n-frag 50                      # CPU
```

On the Mac wheel, `xscape-selftest` checks the installation itself.

## 3. What is here

| path | what |
|---|---|
| `pythonic/` | the Pythonic interface (`import jetscape as js`): `oo_workflow.py`, its XML, notebooks |
| `prod_AuAu_0_10/` | Au+Au 0–10 % hydro only: `run_prod.py`, `run_jobs.sh` |
| `prod_AuAu_0_10_jet/` | Au+Au (and Pb+Pb, Z+jet) with a jet, two hydro legs per event: `run_prod_jet.py`, `run_jobs.sh`, `hadronize.py`, `run_hadronize.py`, `run_h5toROOT.py` |
| `prod_OO_isr/` | O+O (and Ne+Ne) with initial-state radiation: `run_prod_oo_isr.py`, `run_jobs.sh` |
| `analysis/`, `analysis_root/` | Python and ROOT/C++ analyses of the production files |
| `python_modules/`, the scripts at the top | X-SCAPE modules written in Python; single features of the bindings |
| `config/` | user XMLs |

Every folder has a `README.md`, and every driver prints its options with `--help`. Read both
before you change a command line.

## 4. Running a production

- **One job** is one driver call: one seed, N events (`run_prod_jet.py`, `run_prod_oo_isr.py`).
  `--dry-run` checks the XML and the grid and prints the plan without running.
- **A campaign** is `./run_jobs.sh [-j P] NJOBS EVENTS_PER_JOB FIRST_SEED [OUTDIR] [driver options]`.
  `FIRST_SEED 0` gives every job a new seed that is never repeated (the usual choice).
  OUTDIR is the argument after FIRST_SEED, not `--outdir`. One campaign per OUTDIR. Put OUTDIR
  one level inside a writable directory (`/work/out`, not `/work`): the seed registry
  `seeds_used.tsv` is written to its parent. Finished jobs are skipped when the same command
  runs again.
- **The physics is in the user XML** the driver names (`--help`, the folder's README). Copy it
  and pass the copy; do not edit `jetscape_main.xml` or files of the installed tree.
- **Centrality comes from the impact parameter.** Inside X-SCAPE, 3dMCGlauber ignores
  `cenMin`/`cenMax`; set `<b_max>` (Au+Au 0–10 % is 4.7 fm, 20 fm is minimum bias).
- **Hadrons are a second stage.** With `--write-particlize` the job stores the freeze-out
  surfaces and partons; `run_hadronize.py OUTDIR -j P --oversample N --n-frag M` makes the
  hadron files on CPUs and can be repeated with other settings.
- **Planning and watching:** `prodwizard.py` writes the launch commands for a machine,
  `jobmon.py OUTDIR` shows jobs, events per hour and failures, `check_deposition.py OUTDIR`
  flags jobs that lost almost no jet energy. A failed job's reason is in `OUTDIR/<stem>.log`.

## 5. The files and how to read them

| file | content | reader |
|---|---|---|
| `<stem>.h5` | the hydro pair: `arr` (jet leg), `arr_bg` (background), shape `(events, 4, nx, ny, neta, ntau)` | `h5py` |
| `<stem>_particlize.h5` | freeze-out surfaces, final partons, event table | `jetscape.particlize_h5.ParticlizeFile` |
| `<stem>_hadrons_{bulk_jet,bulk_bg,jet_frag,extra_frag}.h5` | hadrons: medium + wake, medium alone, jet fragments, fragments of extra hard scatterings | `jetscape.hadrons_h5.HadronFileReader` |
| `<stem>_hadrons.root`, `<campaign>_campaign.root` | the hadrons as ROOT files (`run_h5toROOT.py`) | uproot, `analysis_root/HadronFileReader.h` |
| `<stem>.json`, `.xml`, `.log` | job summary, the exact XML, the log | |

```python
import numpy as np
from jetscape.hadrons_h5 import HadronFileReader
with HadronFileReader("/path/to/campaign") as r:            # a directory, a file or a glob
    print(r.n_files, r.n_events, r.tags())
    dn, err = r.hist("bulk_bg", "pt", np.linspace(0, 4, 41), mask="charged")   # per event
    wake, werr = r.jet_minus_background("phi", np.linspace(-np.pi, np.pi, 33), mask="charged")
```

- Always `import hdf5plugin` before reading with plain `h5py`: the files are Blosc-compressed
  and every read fails without it (`OSError: Can't synchronously read data`). The readers of
  `jetscape` import it themselves.
- Look at a file first: `h5_inspect.py FILE`. Compare two files or directories bit by bit:
  `h5_compare.py A B --ignore-stamps`. Both are in js-contrib's `utils/`.
- `HadronFileReader` takes the directory that holds the `_particlize.h5` files with their
  hadron files next to them (or the hadron files of a `--hadrons-only` job).
- The analyses need no X-SCAPE and no GPU; `hadronize.py` needs X-SCAPE, but no GPU.

## 6. Rules

1. **Test small first.** One or two events with a fixed `--seed` and a new output directory
   before any campaign. Do not start a campaign, a long job or a large download without the
   user's explicit go: they take hours of GPU time and many GB of disk.
2. **Never write into the installation**: not into `/opt/X-SCAPE`, the wheel's
   `site-packages`, or (unless the user develops it) the source tree. Work in this copy and in
   the user's output directories. `xscape-examples` never overwrites; a fresh copy needs a
   new directory.
3. **Do not delete or overwrite production output.** A new setting gets a new OUTDIR.
4. **Oversamples are not events.** The N oversamples of one hadron file unit are N samplings
   of the same hydro event. Average them per event, then average over events; the readers'
   `hist`, `total` and `jet_minus_background` do this. Statistical errors come from the number
   of events.
5. **Know which error you report.** The reader's default error treats every hadron as
   independent (compound Poisson); resonance decays make it about 20 % too small for soft
   multiplicities. Files hadronized with `--correlated` need the paired error, which
   `jet_minus_background` selects by itself.
6. **The analysis scripts are examples, not vetted results.** Say so when you show a physics
   number, and state the cuts, the number of events and the campaign it came from.
7. **Do not invent options, XML tags or dataset names.** Check `--help`, the folder's
   `README.md`, `h5_inspect.py`, or the website.
8. **Report what ran.** Quote the command, where the output is, and the lines of the log that
   show success or failure (`... done in ... s` per event; `No CUDA device found` means the
   run was on the CPU).

## 7. Where to look things up

| topic | page |
|---|---|
| install: container, Mac wheel, analysis venv, source build | <https://jhputschke.github.io/js-contrib/install/> |
| first event, the productions, clusters, the OSDF data | <https://jhputschke.github.io/js-contrib/run/> |
| file layouts, hadrons, jets, the wake, ROOT | <https://jhputschke.github.io/js-contrib/analysis/> |
| the Pythonic interface, modules in Python | <https://jhputschke.github.io/js-contrib/python/> |
| tools (`prodwizard`, `jobmon`, `h5_inspect`, `h5_compare`, `js_osdf`) | <https://jhputschke.github.io/js-contrib/reference/tools/> |
| the whole site in one file | <https://jhputschke.github.io/js-contrib/llms-full.txt> |
