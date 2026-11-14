# Mac (xscape wheel)

On an Apple Silicon Mac, X-SCAPE runs natively with MUSIC on the Mac's own GPU through Metal.
It comes as a Python wheel, `xscape`, installed with [uv](https://docs.astral.sh/uv/) from a
lock file. There is no compiler and no container involved. Docker on a Mac can't reach the
GPU, so on a Mac the wheel takes the place of the [production container](container.md); the
scripts, notebooks and outputs are the same. The environment holds X-SCAPE with MUSIC
(MUSIC4GPU, Metal), iSS, 3dMCGlauber, Matter, LBT, iColored and Pythia, PyJetscape and its
Pythonic layer, and the analysis packages (awkward, vector, fastjet, particle, hist, mplhep,
matplotlib, scipy, pandas, uproot, JupyterLab), pelicanfs for `osdf://` paths, and js-contrib's
production tools as commands (`xscape-prodwizard`, `xscape-jobmon`, `xscape-osdf`, ...).

**What it can't do:** rebuild X-SCAPE or the bindings (that needs the
[source build](source.md)), ROOT/PyROOT (uproot, included, reads ROOT files), and PyTorch (not
in the environment). Some modules and data are not in it: [What runs with the
wheel](#what-runs-with-the-wheel). Intel Macs are not supported.

## Requirements

- A Mac with Apple Silicon (M1 or later) on **macOS 26** or later (Apple menu → About This
  Mac).
- About 3 GB of disk, 1 GB of it uv's download cache (`uv cache clean` frees it).
- Internet access during the installation.

You do **not** need Xcode, Homebrew, conda, Docker, git or an Apple developer account.

!!! info "Releases"
    The wheels are published on the
    [Releases page of jhputschke/xscape-macos](https://github.com/jhputschke/xscape-macos/releases).
    The current release is **v1.5**; check that page for the latest. Each release is a set
    of three files: the `xscape` wheel (X-SCAPE, 175 MB), the `pyjetscape` wheel (the Python
    package), and a **lock file** `xscape-<version>-cp313.lock.txt`, which names every package
    of the environment with its exact version and checksum. You install from the lock file,
    so every Mac gets exactly the same software.

    **Version numbers:** from 1.5 on, a release has the number of the js-contrib release it
    is built from (xscape 1.5 = js-contrib `v1.5` = pyjetscape 1.5), the same number as the
    production containers of that release (`cu126-v1.5`, `cu130-v1.5`, `cu124-v1.5`), which
    are built from the same commits. Earlier releases were numbered on their own (0.1.0,
    0.2.0); 1.5 follows 0.2.0.

## Install with one command

```bash
curl -LsSf https://raw.githubusercontent.com/jhputschke/xscape-macos/main/install.sh | sh
```

The script, [`install.sh`](https://github.com/jhputschke/xscape-macos/blob/main/install.sh),
does steps 1–4 below for the latest release:

1. checks the Mac (Apple Silicon, macOS 26 or later, 3 GB free) before downloading anything;
2. installs uv, unless a recent one is already there;
3. creates the environment `~/.venvs/xscape-VER` from the release's lock file;
4. fetches the LBT tables and runs `xscape-selftest`.

It takes a few minutes (93 s on an M3 Max with a fast connection, everything downloaded) and ends with `xscape VER is installed in ...` and
the line that activates the environment. Then run your first event (step 5):

```bash
source ~/.venvs/xscape-1.5/bin/activate       # the line it printed; in every new Terminal window
xscape-examples ~/xscape/examples
cd ~/xscape/examples/pythonic
python oo_workflow.py --events 1 --outdir ~/xscape/out_oo
```

Options go after `sh -s --`, for example
`... | sh -s -- --version 1.5 --examples ~/xscape/examples`:

| option | does |
|---|---|
| `--version VER` | installs this release (default: the latest on the Releases page) |
| `--venv DIR` | puts the environment in `DIR` (default: `~/.venvs/xscape-VER`) |
| `--examples DIR` | also copies the examples to `DIR` (never overwrites) |
| `--force` | deletes an existing environment first |
| `--no-lbt`, `--no-selftest` | skips the LBT tables, the self-test |

Running it again is safe: it reuses the environment and the LBT tables, and the lock file
makes the environment exactly the release again (`--force` builds it new). Apart from the
environment it writes only uv (in `~/.local/bin`; uv's installer adds that directory to your
shell's `PATH`), uv's cache and Python, and the LBT tables in `~/Library/Caches/xscape`.

The steps below do the same by hand.

## Step 1: install uv

uv is Astral's Python package manager: a single program in `~/.local/bin`.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then open a new Terminal window (or run `source ~/.local/bin/env`) so that `uv` is found. With
Homebrew, `brew install uv` works too.

## Step 2: create the environment from the lock file

```bash
VER=1.5                       # the latest: github.com/jhputschke/xscape-macos/releases
mkdir -p ~/xscape && cd ~/xscape
curl -LO https://github.com/jhputschke/xscape-macos/releases/download/v$VER/xscape-$VER-cp313.lock.txt
uv venv --managed-python ~/.venvs/xscape-$VER --python 3.13
source ~/.venvs/xscape-$VER/bin/activate
uv pip sync xscape-$VER-cp313.lock.txt
```

uv downloads Python 3.13 and about 200 MB of packages, which takes seconds to a few minutes
depending on the connection.

- `--managed-python` makes uv use its own Python, never one from Homebrew, conda or python.org
  that happens to be installed, so nothing outside the environment is used.
- `uv pip sync` installs exactly what the lock lists, hash-checked, and removes anything else.
- `source ~/.venvs/xscape-$VER/bin/activate` is needed in **every new Terminal window** where
  you use X-SCAPE. Your prompt then starts with `(xscape-1.5)`.

## Step 3: fetch the LBT tables

LBT, the jet energy-loss module, needs its scattering tables: a 240 MB download, 1.1 GB on
disk. They are kept in `~/Library/Caches/xscape` and shared by all your environments.

```bash
xscape-fetch-data lbt
```

It checks every file against the release's checksums and ends with
`xscape-fetch-data: LBT tables ... (checked) -> ...`. If you already have the tables on disk,
`xscape-fetch-data lbt --from DIR` uses them instead of downloading.

## Step 4: check the environment

```bash
xscape-selftest
# ... image_check: all ok
python -c "import jetscape as js; print(js.HAS_CORE, js.pyjetscape_core.MUSIC_BACKEND)"
# -> True music4gpu-metal
```

`xscape-selftest` checks the bindings, the HDF5 filters, pelicanfs, the paths to the installed
X-SCAPE, the analysis packages, MUSIC's EoS table and the LBT tables, and sends one synthetic
event through awkward, FastJet and mplhep. It stops at the first failure. `True` means the
compiled bindings load; `music4gpu-metal` means they were built with the Metal MUSIC. The first
`import` after the installation takes a little longer, because matplotlib builds its font cache
once.

When an event runs, MUSIC prints the GPU it uses when it starts:
`[MUSIC-GPU] Metal device: <your chip>`. There is nothing to switch on. If instead you see
`[MUSIC-GPU] Metal init failed, falling back to CPU.`, MUSIC runs on the CPU; that happens
inside a virtual machine or a container, so run natively in Terminal.

## Step 5: copy the examples and run one event

```bash
xscape-examples ~/xscape/examples
xscape-utils ~/xscape/utils          # optional: js-contrib's utils/, to read or adapt
cd ~/xscape/examples/pythonic
python oo_workflow.py --events 1 --outdir ~/xscape/out_oo
```

`xscape-examples` copies every example into a directory of yours: the production scripts, the
Pythonic examples (`pythonic/`), the toy Python modules (`python_modules/`), the analysis
notebooks and their XMLs. The examples write next to themselves, so they run from this copy.
`xscape-utils` does the same for js-contrib's `utils/` (`jobmon.py`, `prodwizard.py`,
`h5_inspect.py`, `remote_transfer/`, ...); you don't need the copy to use them, they are
commands of the environment ([Command reference](#command-reference)). Neither ever
overwrites: for a fresh copy, give a new directory.

The event is one O+O collision through the whole chain: 3dMCGlauber, Pythia with initial-state
radiation, MUSIC on the GPU, Matter + LBT, iColored + iSS. It takes under a minute (27 s on an
M3 Max) and writes two hadron files to `~/xscape/out_oo/`. What it did:
[Your first event](../run/first-event.md).

## The rules

1. **Your files live in your own directories** (e.g. `~/xscape`). The installed X-SCAPE
   inside `~/.venvs/...` is not edited; the examples run from your copy.
2. **Activate the environment** in every new Terminal window:
   `source ~/.venvs/xscape-VER/bin/activate`.
3. **One environment per release.** A release fixes all the software. For a campaign, keep to
   one release, and don't mix files of two releases, since physics fixes change results. The
   commits a release was built from:
   `cat "$(python -c 'import xscape; print(xscape.TREE)')/BUILD_INFO.txt"`.
4. **Everything finds the installed X-SCAPE by itself** ([`jetscape.paths`][jetscape.paths]). Leave `XSCAPE`,
   `JETSCAPE_BUILD`, `PYTHONPATH` and `DYLD_LIBRARY_PATH` unset in this environment: they belong
   to a source build and would point it elsewhere.
5. **Results agree statistically, not bit for bit,** across Macs and macOS versions.

| | where |
|---|---|
| the environment | `~/.venvs/xscape-VER` |
| the installed X-SCAPE (data, `config/`, `BUILD_INFO.txt`) | `python -c "import xscape; print(xscape.TREE)"` |
| X-SCAPE's default parameters | `config/jetscape_main.xml` in that tree |
| the examples | your copy, e.g. `~/xscape/examples` (`xscape-examples`) |
| js-contrib's `utils/` | inside the installed tree (`utils/`); your copy with `xscape-utils` |
| the LBT tables | `~/Library/Caches/xscape/LBT-tables-<commit>` |

## One XML: `xscape-run`

`xscape-run` runs X-SCAPE's `runJetscape` on a user XML in a directory of its own, and takes any
XML as it is: one of the installed X-SCAPE's `config/`, one of the examples, or your own.

```bash
xscape-run -w ~/xscape/runs/oo -u ~/xscape/examples/pythonic/oo_workflow.xml
xscape-run -w ~/xscape/runs/pp -u "$(python -c 'import xscape; print(xscape.TREE)')/config/jetscape_user.xml"
```

- It copies both XMLs into the run directory (`jetscape_main.xml`, `<name>.run.xml`; your file
  is not changed), with every `../` path made absolute. A `../` path with no file behind it is
  looked up in the installed X-SCAPE by its last parts: `../external_packages/iSS/iSS_tables`
  becomes its `iSS_tables`, `../config/HH_LightExcitedMesons.xml` its `config/...`. Give inputs
  of your own as absolute paths.
- It links the tables 3dMCGlauber and LBT read from the working directory and gives the run a
  private `music_input`, so several runs can go at once.
- **Before anything runs it checks the XML** and stops (exit code 3) with the reason:
    - a tag the main XML lacks: X-SCAPE would stop at the first with `User XML tag <...> is
      unrecognized`. Either the tag is misspelled or the XML is newer than the environment
      (`<extraHard>` in an environment before 0.2.0);
    - an input with no file behind it (the test hydro files, SMASH's configuration);
    - a module this wheel does not have ([below](#what-runs-with-the-wheel)), which X-SCAPE would
      only warn about and then run without;
    - automatic module setup turned off (`<enableAutomaticTaskListDetermination> false`): such
      XMLs are written for C++ test programs, and `runJetscape` would run nothing;
    - an LHAPDF6 set for Pythia (`PDF:pSet = LHAPDF6:...`), which the wheel's Pythia can't load.

  `xscape-run -f ...` runs anyway.

`runJetscape USER_XML` runs X-SCAPE in the current directory, as it is. When the XML needs
files there that the directory lacks (3dMCGlauber's `tables/`, MUSIC's `music_input`, `../`
paths), it stops and points to `xscape-run`, rather than failing in the middle with
`sh: ./Metropolis.e: No such file or directory` (`XSCAPE_RUN_HERE=1` runs it anyway). Use
`xscape-run`.

## Running the productions

The productions run from your copy of the examples, with the same options as on a GPU server
or in the container.

```bash
cd ~/xscape/examples/prod_AuAu_0_10_jet

# Au+Au 0-10 % with a jet: 2 events (about 40 s on an M3 Max)
python run_prod_jet.py --events 2 --seed 1 --write-particlize both --outdir ~/xscape/prod/out

# a campaign: 20 jobs x 25 events, unique seeds, 3 jobs at a time on the GPU
OMP_NUM_THREADS=5 OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0 \
  ./run_jobs.sh -j 3 --campaign test 20 25 0 ~/xscape/prod/out

# stage 2 (hadronization) on the CPU
python run_hadronize.py ~/xscape/prod/out -j 4 --oversample 500 --n-frag 50

# O+O with initial-state radiation
cd ~/xscape/examples/prod_OO_isr
python run_prod_oo_isr.py --events 2 --seed 5 --outdir ~/xscape/prod/out_oo
```

- **Put OUTDIR one level inside a directory of yours** (`~/xscape/prod/out`): every job records
  its seed in `OUTDIR/../seeds_used.tsv`.
- **Your own physics settings:** copy a user XML from `~/xscape/examples/config/`, edit it and
  pass it with `--user-xml my.xml`.
- **Several jobs at once:** on an M3 Max, `-j 3` with `OMP_NUM_THREADS=5` and
  `OMP_WAIT_POLICY=passive` gave the most events per hour; aim for `-j` × threads ≈ the
  number of performance cores. Without `passive`, idle OpenMP threads spin and slow the other
  jobs down.
- **Long campaigns:** keep the Mac awake, e.g. `caffeinate -i ./run_jobs.sh ...`.
- **ROOT output** (`run_h5toROOT.py` with ROOT) needs PyROOT, which this environment does not
  have; uproot writes and reads the ROOT files instead.

- **The extra hard scatterings, minimum bias, O+O and Ne+Ne at 5.36 TeV**
  (`run_prod_jet.py --extra-hard`, `--hard none`, `../config/OO_5360_jet.xml`,
  `NeNe_5360_jet.xml`) need the 0.2.0 environment or later. An older one stops with the
  unknown-tag message for `<extraHard>`: install the new release next to it.

**Plan, watch and upload with the tools.** `xscape-prodwizard` writes the launch commands for
this Mac with the measured settings (machine `mac_wheel`: this environment, your copy of the
examples, `run_jobs.sh`) and estimates the duration, disk and memory; it never starts jobs.
`xscape-jobmon` watches the campaign.

```bash
xscape-prodwizard --preset auau_jet --machine mac_wheel --events 300 --campaign test
xscape-prodwizard --preset oo_hadrons --machine mac_wheel --events 1000 --save launch_oo.sh
xscape-jobmon ~/xscape/examples/prod_AuAu_0_10_jet/out/test --tui     # while it runs
xscape-check-deposition ~/xscape/examples/prod_AuAu_0_10_jet/out/test
xscape-upload-follow ~/xscape/examples/prod_AuAu_0_10_jet/out AuAu_test --follow   # to the OSDF
```

`mac_wheel` takes the environment from `$VIRTUAL_ENV` (default `~/.venvs/xscape`) and the
examples from `$XSCAPE_EXAMPLES` (default `~/xscape/examples`); inside this environment the
wizard picks it by itself. The rates are the M3 Max's. The tools and their options:
[Tools](../reference/tools.md); the OSDF: [Data on the OSDF](../run/osdf.md#on-a-mac-the-xscape-wheel).

The productions themselves: [Au+Au jets (two stages)](../run/auau-jet.md),
[O+O with ISR](../run/oo-isr.md), [Z+jet and other systems](../run/zjet-other.md),
[Extra hard scatterings](../run/extra-hard.md).

## Your own scripts and Python modules

Put your script and its user XML in a directory of yours, starting from a copy of
`examples/pythonic/oo_workflow.py` and `oo_workflow.xml` (`oo_workflow.py` takes any user XML
with `--xml`). The skeleton every such script has:

```python
# ~/xscape/mine/my_job.py
import os
import jetscape as js
from jetscape.hadrons_h5 import HadronH5Writer

out = os.path.expanduser("~/xscape/out_mine")
with js.JobDir(out, os.path.expanduser("~/xscape/mine/my_job.xml")) as job:  # private working dir
    with js.per_event(job.main_xml, job.user_xml) as driver, \
         HadronH5Writer(f"{out}/my_job_hadrons.h5", tag="jet_frag", n_samples=1) as h5:
        for i in driver.events():                                         # one iteration per event
            jets = driver.hadronization_manager.hadron_array()
            h5.append_unit(jets.to_dict(), unit=i, event=i)
            print(i, len(jets), "jet hadrons")
```

`js.JobDir(out_dir, user_xml)` copies your XML to `out_dir/`, points it at the installed
X-SCAPE's tables, runs the job in a private directory `out_dir/work/<name>` and removes it
after a successful run. Inside the `with` block the current directory is that working
directory, so use absolute output paths. Details: [The Pythonic interface](../python/pythonic.md).

**Physics modules in Python** (a hard process, energy loss, hadronization, an afterburner, a
writer) run inside X-SCAPE's event loop with this environment as it is, next to the C++
modules. Start from `examples/python_modules/` (`python run_by_hand_pythonic.py --style new`)
and read [Writing modules in Python](../python/modules.md).

**More Python packages:** `uv pip install seaborn`. A later `uv pip sync` (of the same or a new
lock) removes them again, so keep your additions in a file and run
`uv pip install -r my_packages.txt` after each sync. Don't change numpy or h5py this way.

## Analysis

The same environment is an analysis environment; analysis needs no GPU. Notebooks to start
from, in your copy of the examples:

| notebook | reads |
|---|---|
| `pythonic/oo_workflow.ipynb` (sections 4–6) | the hadron files of your own run |
| `pythonic/oo_analysis.ipynb` | the O+O hadron files streamed from the OSDF (pelicanfs is in the environment) |
| `analysis/tutorials/hadron_distributions.ipynb`, `jet_fastjet*.ipynb` | a production directory |
| `prod_AuAu_0_10_jet/jet_wake.ipynb` | a pair file and its hadron files |
| `prod_OO_isr/oo_isr_overview.ipynb` | an O+O production |
| `analysis/bulk/` (`bulk_obs.py`, notebooks) | Nch, [pT] and vₙ of extras or minimum-bias files |

`osdf://` paths work as they are (`fsspec`, `uproot`, `h5py` through pelicanfs), and
`xscape-osdf download` fetches whole productions: [Data on the OSDF](../run/osdf.md).
`xscape-h5-inspect FILE.h5` shows any HDF5 file, with js-contrib's formats decoded.

To analyse on another computer (Linux, an Intel Mac, a cluster) you need only the Python
package, not X-SCAPE: see [the analysis environment](analysis-env.md#without-a-checkout-the-pyjetscape-wheel).

!!! important "3D visualization is not in the wheel"
    The wheel has neither PyVista nor the [3D visualization](../python/visualization.md)
    scripts (`wake_pyvista.py`, `hydro_jet_particles_pyvista.py`). For movies of the medium,
    the jet wake and the hadrons, clone js-contrib and create the
    [analysis environment](analysis-env.md) next to this one, with
    `setup_analysis_env.sh --python /opt/homebrew/bin/python3`: its visualization stack is
    installed by default. [How, step by step](../python/visualization.md#requirements).

## Jupyter and VS Code

```bash
source ~/.venvs/xscape-VER/bin/activate
cd ~/xscape/examples
jupyter lab
```

JupyterLab opens in your browser. Notebooks run with the environment's Python ("Python 3
(ipykernel)"), and simulations in a notebook use the GPU as scripts do. Stop it with Ctrl-C
twice in the Terminal.

**VS Code:** open `~/xscape/examples`, then "Python: Select Interpreter" (for notebooks also
"Select Kernel") and choose `~/.venvs/xscape-VER/bin/python`.

## Command reference

In the activated environment:

| command | does |
|---|---|
| `xscape-fetch-data lbt [--from DIR]` | the LBT tables into `~/Library/Caches/xscape` (or `$XSCAPE_CACHE`), checked and linked into this environment |
| `xscape-selftest` | checks the environment: bindings, HDF5 filters, pelicanfs, paths, analysis packages, EoS table, LBT tables, one synthetic event |
| `xscape-examples [DEST]` | copies the examples to `DEST` (default `./examples`); never overwrites |
| `xscape-utils [DEST]` | copies js-contrib's `utils/` to `DEST` (default `./utils`); never overwrites |
| `xscape-run -w RUN_DIR -u USER_XML [-f]` | X-SCAPE's `runJetscape` on any user XML, in a directory of its own; checks the XML first ([above](#one-xml-xscape-run)) |
| `runJetscape USER_XML` | X-SCAPE's executable with this environment's data, in the current directory; stops if the directory lacks what the XML needs |
| `xscape-prodwizard` | plans a production (`utils/prodwizard.py`; `--machine mac_wheel`) |
| `xscape-jobmon OUTDIR` | watches a `run_jobs.sh` production (`utils/jobmon.py`; `--watch`, `--tui`) |
| `xscape-check-deposition DIR` | flags Au+Au jet jobs with far too little deposition (`utils/check_deposition.py`) |
| `xscape-h5-inspect FILE.h5` | any HDF5 file, js-contrib's formats decoded (`utils/h5_inspect.py`) |
| `xscape-osdf` | `ls`, `du`, `download`, `upload`, `rm` on the OSDF, `osdf:///fno4hic` (`utils/remote_transfer/js_osdf.py`), in this environment |
| `xscape-upload-follow OUTBASE REMOTE` | uploads a running production's finished jobs to the OSDF (`utils/upload_follow.py`) |
| `xscape-gcs` | the same transfers for Google Cloud Storage (`utils/remote_transfer/js_gcs.py`); makes its own environment in `~/.cache/js_gcs` on first use; needs a service-account key |

The tool commands run the copies of the `utils/` scripts installed with the wheel, with the
same options; `--help` shows them under the command's name.

## What runs with the wheel

The wheel is built without ROOT, HepMC, SMASH, FreestreamMilne, CLVisc, IPGlasma and the FNO
hydro (PyTorch); its Pythia has no LHAPDF6, and Martini's data and X-SCAPE's test hydro files
are not installed. `xscape-run` names what an XML needs of these and stops. For them use the
[production container](container.md) or a [source build](source.md).

| | runs | does not run (why) |
|---|---|---|
| **productions** | `prod_AuAu_0_10/run_prod.py`, `prod_AuAu_0_10_jet/run_prod_jet.py` (also `--extra-hard`, `--hard none`, the 5.36 TeV O+O and Ne+Ne XMLs), `run_prod_vacuum.py`, `hadronize.py`, `run_hadronize.py`, `prod_OO_isr/run_prod_oo_isr.py`, `run_jobs.sh` | `run_h5toROOT.py` with ROOT (PyROOT) |
| **examples** | `pythonic/` (scripts and both notebooks), `python_modules/`, `per_event_loop.py`, `python_bulk_h5_writer.py`, `inspect_bulk_info.py`, `python_bulk_root_writer.py` (uproot), `analysis/` | `python_fno_test.py` (PyTorch), `python_fast_bulk_root_writer.py`, `validate_h5_vs_root.py`, `analysis_root/` (ROOT), `prod_AuAu_0_10_jet/music_cpu_gpu_hydro.py` (a CPU MUSIC), `prod_OO_isr/hadro_compare/run_hadro_compare.sh` (two builds), the movie step of `analysis/zjet/test/make_data.sh` (Visualization) |
| **X-SCAPE's test XMLs** (installed `config/`) | `jetscape_user.xml`, `_MUSIC`, `_DIS`, `_photoproduction`, `_twostagehydro`, `_brick_hybrid_hadronization`, `_OO_ISR_MatterLBT_iColored` | the rest: the HepMC writer (`_rMatterLbt`, `_root_test`; `-f` runs them without it), ROOT writers (`BulkFastTest/`, `_dAu3d`), SMASH, FreestreamMilne (`_AA_dukeTune*`, `_FSFileTest`, `_sims`), CLVisc, IPGlasma, test hydro files (`_HydroFile`, `_PbPb_hard_nPDF`), LHAPDF6 (`_nPDF_test`), automatic module setup off (`FVvsMUSIC/`, `_iMATTERMCGlauber*`, `_MUSICMainClockTest`, `_SMASH*`, `_test`); `_MCGlauberMUSIC` and `3D/` use a tag no main XML has (`<nEvents_MCGla>`) |

## Updating and removing

**A new release** goes into a new environment next to the old one, so a running campaign keeps
its software. With the script (the latest release, or `sh -s -- --version VER`):

```bash
curl -LsSf https://raw.githubusercontent.com/jhputschke/xscape-macos/main/install.sh | sh
source ~/.venvs/xscape-<new version>/bin/activate
xscape-examples ~/xscape/examples-<new version>   # the examples of this release
```

or by hand:

```bash
VER=<new version>
cd ~/xscape
curl -LO https://github.com/jhputschke/xscape-macos/releases/download/v$VER/xscape-$VER-cp313.lock.txt
uv venv --managed-python ~/.venvs/xscape-$VER --python 3.13 && source ~/.venvs/xscape-$VER/bin/activate
uv pip sync xscape-$VER-cp313.lock.txt
xscape-fetch-data lbt                    # reuses the cached tables when they match
xscape-examples ~/xscape/examples-$VER   # the examples of this release
```

**Removing:**

```bash
rm -rf ~/.venvs/xscape-VER               # one environment
rm -rf ~/Library/Caches/xscape           # the LBT tables
uv cache clean                           # uv's downloads
rm ~/.local/bin/uv ~/.local/bin/uvx      # uv itself
```

## Troubleshooting

| symptom | cause and fix |
|---|---|
| `zsh: command not found: uv` | open a new Terminal window, or `source ~/.local/bin/env` |
| `install.sh: this shell runs under Rosetta (x86_64)` | Terminal is set to open using Rosetta: Finder → Applications → Utilities → Terminal → Get Info → untick "Open using Rosetta" |
| `install.sh: ... exists but is no environment` | `--venv` names a directory that is not an environment: give another directory, or `--force` to replace it |
| `... has no wheels with a matching platform tag` | macOS older than 26, or an Intel Mac: update macOS; Intel Macs are not supported |
| `zsh: no matches found: pyjetscape[analysis]...` | quote the argument: `"pyjetscape[analysis] @ ..."` |
| `ModuleNotFoundError: No module named 'jetscape'` | the environment is not active: `source ~/.venvs/xscape-VER/bin/activate` |
| `import jetscape` picks up a source checkout, or `JobDir: not found: .../music_input` | `PYTHONPATH`, `XSCAPE` or `JETSCAPE_BUILD` set (e.g. in `~/.zshrc`): `unset PYTHONPATH XSCAPE JETSCAPE_BUILD` |
| LBT: `cannot open ... LBT-tables` | step 3 not done for this environment: `xscape-fetch-data lbt` |
| `xscape-fetch-data: ... file(s) missing or different` | an interrupted download: `rm -rf ~/Library/Caches/xscape/LBT-tables-*` and run it again |
| `xscape-examples: ... exists and is not empty` | it never overwrites: give another directory |
| `python: can't open file 'run_prod_jet.py'` | run it from your copy, `~/xscape/examples/prod_AuAu_0_10_jet` |
| MUSIC slow, no GPU | inside a virtual machine or a container there is no Metal GPU: run natively in Terminal |
| `ModuleNotFoundError: ROOT` | ROOT is not part of this environment; uproot reads ROOT files |
| `sh: ./Metropolis.e: No such file or directory` (3dMCGlauber), from `runJetscape` | an environment before 0.2.0, run in a directory without `tables/`: use `xscape-run -w RUN_DIR -u USER_XML` |
| `runJetscape: ... reads files relative to the current directory` | the same, caught: use `xscape-run` |
| `User XML tag <extraHard> is unrecognized`, or `xscape-run: tags of ... that the main XML lacks` | the XML is newer than the environment (the extra hard scatterings need 0.2.0), or a tag is misspelled: install the current release |
| `xscape-run: ... needs modules this xscape wheel does not have` | the XML uses ROOT, SMASH, FreestreamMilne, CLVisc, IPGlasma or HepMC: [What runs with the wheel](#what-runs-with-the-wheel) |
| `xscape-run: ... turns automatic module setup off` | an XML of an X-SCAPE C++ test program: run it with its program in a source build |
| [`JobDir`][jetscape.workdir.JobDir] / `enter_workdir` raise `XMLProblem` | the same checks from Python, before X-SCAPE starts: the message names the tags or inputs |
| `ValueError: Protocol not known: osdf` | an environment before 0.2.0 has no pelicanfs: install the current release (or `uv pip install pelicanfs`) |
| `AttributeError: 'Client' object has no attribute 'info'` on an `osdf://` path | installed after 2026-11-02, the environment has aiowebdav2 1.x, which pelicanfs (up to 1.5.0) cannot use: `uv pip install "aiowebdav2<1"` in the wheel's environment |

The same guide, with every detail:
[`utils/README_Mac_Usage.md`](https://github.com/jhputschke/js-contrib/blob/main/utils/README_Mac_Usage.md)
(also the README of [jhputschke/xscape-macos](https://github.com/jhputschke/xscape-macos)).

## Next: run your first event

You ran one event in step 5; [Your first event](../run/first-event.md) explains what it did,
what the output files hold and where to go from there.
