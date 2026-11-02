# X-SCAPE on a Mac

> **Install:** the wheels and lock files are released by [jhputschke/xscape-macos](https://github.com/jhputschke/xscape-macos) ([Releases](https://github.com/jhputschke/xscape-macos/releases)); this guide is also its README. The container guide for Linux with NVIDIA GPUs: [`README_Container_Usage.md`](README_Container_Usage.md).

X-SCAPE with MUSIC on the Mac's own GPU (Metal), installed with one command, no compiler
and no containers. Docker and other containers on a Mac can't reach the GPU, so on a Mac
this replaces the [production image](README_Container_Usage.md);
the scripts, notebooks and outputs are the same.

One environment, three uses:

| | use | GPU | you bring |
|---|---|---|---|
| [1](#1-the-production-scripts) | **the production scripts** (`run_prod_jet.py`, `run_prod_oo_isr.py`, `run_jobs.sh`, `hadronize.py`, ...) | Metal (stage 2 hadronization: CPU) | an output directory |
| [2](#2-your-own-scripts-and-notebooks) | **your own scripts and notebooks**, production-like (the O+O workflow, the examples) | Metal | your scripts, notebooks, XMLs |
| [3](#3-analysis) | **analysis**: hadron and pair files, FastJet, plots, notebooks | no | the files |

The environment holds X-SCAPE with MUSIC (music4gpu, Metal), iSS, 3dMCGlauber, Matter, LBT,
iColored and Pythia, compiled, with PyJetscape and its Pythonic layer
([`PYTHONIC.md`](../contribs/PyJetscape/PYTHONIC.md)),
and the analysis packages (awkward, vector, fastjet, particle, hist, mplhep, matplotlib,
scipy, pandas, uproot, Jupyter), precompiled: for developing X-SCAPE itself see
[2.5](#25-developing-x-scape-or-pyjetscape-not-with-this-environment). How it is built: [`BUILDING.md`](https://github.com/jhputschke/xscape-macos/blob/main/BUILDING.md) in the wheel's repository.

Contents: [Get started](#get-started) · [The rules](#the-rules) ·
[1 Productions](#1-the-production-scripts) · [2 Own scripts and notebooks](#2-your-own-scripts-and-notebooks) ·
[3 Analysis](#3-analysis) · [4 Productions and data](#4-productions-and-data) ·
[Jupyter](#jupyter) · [What runs with the wheel](#what-runs-with-the-wheel) ·
[Updating and removing](#updating-and-removing) · [Troubleshooting](#troubleshooting) ·
[Reference](#reference)

---

## Get started

**Prerequisites.** A Mac with Apple Silicon (M1 or later) on **macOS 26** or later
( → About This Mac), about 3 GB of disk (1 GB of it uv's download cache, `uv cache clean`), internet for the installation. Not needed: Xcode,
Homebrew, conda, Docker, git, an Apple developer account.

**Releases** are on the [Releases page of jhputschke/xscape-macos](https://github.com/jhputschke/xscape-macos/releases), the repository that builds the wheel.
Each one is a set of three files: the `xscape` wheel (X-SCAPE, 175 MB), the `pyjetscape`
wheel (the Python package) and a **lock file**, `xscape-<version>-cp313.lock.txt`, which
names every package of the environment with its exact version and checksum. You install
from the lock file, so every Mac gets the same software. Below, `VER` is the release's
version (e.g. `0.2.0`).

**Step 1: install uv** (Astral's Python package manager; one program, in `~/.local/bin`):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then open a new Terminal window (or `source ~/.local/bin/env`), so that `uv` is found.
With Homebrew, `brew install uv` works too.

**Step 2: the environment** (uv downloads Python 3.13 and about 200 MB of packages; seconds to a few minutes, depending on the connection):

```bash
VER=0.2.0
mkdir -p ~/xscape && cd ~/xscape
curl -LO https://github.com/jhputschke/xscape-macos/releases/download/v$VER/xscape-$VER-cp313.lock.txt
uv venv --managed-python ~/.venvs/xscape-$VER --python 3.13
source ~/.venvs/xscape-$VER/bin/activate
uv pip sync xscape-$VER-cp313.lock.txt
```

`--managed-python` makes uv use its own Python, never one from Homebrew, conda or
python.org that happens to be installed, so nothing outside the environment is used.
`source .../activate` is needed in every new Terminal window in which you use X-SCAPE (your
prompt then starts with `(xscape-VER)`).

**Step 3: the LBT tables** (the jet energy-loss tables: a 240 MB download, 1.1 GB on disk,
kept in `~/Library/Caches/xscape` and shared by all your environments):

```bash
xscape-fetch-data lbt
```

It checks every file against the release's checksums and ends with
`xscape-fetch-data: LBT tables ... (checked) -> ...`.

**Step 4: check it**

```bash
xscape-selftest
# ... image_check: ok    pyjetscape_core imports ...  ... image_check: all ok
```

It checks this install, the LBT tables, the bindings, the HDF5 filters, pelicanfs, the
analysis packages, MUSIC's EoS table, and runs one synthetic event through awkward, FastJet
and mplhep. It ends with `image_check: all ok`, and stops at the first failure with
`image_check: FAIL: ...`. The first run after the installation takes a little longer
(matplotlib builds its font cache once).

**Step 5: your first event** (one O+O event through the whole chain: 3dMCGlauber → Pythia
with initial-state radiation → MUSIC on the GPU → Matter + LBT → iColored + iSS; under a
minute (27 s on an M3 Max), two hadron files in `~/xscape/out_oo/`):

```bash
xscape-examples ~/xscape/examples
cd ~/xscape/examples/pythonic
python oo_workflow.py --events 1 --outdir ~/xscape/out_oo
```

The same event from its XML alone, without Python: `xscape-run -w ~/xscape/runs/oo -u
~/xscape/examples/pythonic/oo_workflow.xml` (section [2.2](#22-one-xml-xscape-run)).

`xscape-utils ~/xscape/utils` copies js-contrib's tools (`jobmon.py`, `prodwizard.py`,
`h5_inspect.py`, `remote_transfer/`, ...), to read or adapt. You don't need the copy to run
them: they are commands of the environment ([section 4](#4-productions-and-data)).

---

## The rules

1. **Your files live in your own directories** (e.g. `~/xscape`). The installed X-SCAPE
   (inside `~/.venvs/...`) is not edited; the examples run from a copy (`xscape-examples`),
   because they write next to themselves.
2. **Activate the environment** in every new Terminal window:
   `source ~/.venvs/xscape-VER/bin/activate`.
3. **One environment per release.** A release fixes all the software; for a campaign, keep
   to one release, and don't mix files of two releases (physics fixes change results). The
   environment's `BUILD_INFO.txt` lists the commits it was built from:
   `cat "$(python -c 'import xscape; print(xscape.TREE)')/BUILD_INFO.txt"`.
4. **Everything finds the installed X-SCAPE by itself** (`jetscape.paths`). Leave the
   variables `XSCAPE`, `JETSCAPE_BUILD`, `PYTHONPATH` and `DYLD_LIBRARY_PATH` unset in this
   environment: they are for a source build and would point it elsewhere.
5. **The GPU is used automatically.** MUSIC runs on Metal; there is nothing to switch on.
   Results across Macs or macOS versions agree statistically, not bit for bit.

Where things are:

| | where |
|---|---|
| the environment | `~/.venvs/xscape-VER` |
| the installed X-SCAPE (data, config, BUILD_INFO.txt) | `python -c "import xscape; print(xscape.TREE)"` |
| the examples: production scripts, notebooks, XMLs | your copy, e.g. `~/xscape/examples` (`xscape-examples`) |
| the LBT tables | `~/Library/Caches/xscape/LBT-tables-<commit>` |

---

## 1 The production scripts

The productions run from your copy of the examples, with the same options as on a GPU
server or in the container. Their full documentation:
[`prod_AuAu_0_10_jet/README.md`](../contribs/PyJetscape/example/prod_AuAu_0_10_jet/README.md)
(Au+Au with jets), [`prod_OO_isr/README.md`](../contribs/PyJetscape/example/prod_OO_isr/README.md)
(O+O with ISR), [`README_2stage.md`](../docs/README_2stage.md)
(the two stages).

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

- **Put OUTDIR one level inside a directory of yours** (`~/xscape/prod/out`): every job
  records its seed in `OUTDIR/../seeds_used.tsv`.
- **Your own physics settings:** copy a user XML from `~/xscape/examples/config/`, edit it,
  and pass it with `--user-xml my.xml`.
- **Several jobs at once:** on an M3 Max, `-j 3` with `OMP_NUM_THREADS=5` and
  `OMP_WAIT_POLICY=passive` gave the most events per hour (`-j` × threads ≈ the
  performance cores). Without `passive`, idle OpenMP threads spin and slow the other jobs.
- **Long campaigns:** keep the Mac awake, e.g. `caffeinate -i ./run_jobs.sh ...`.
- **ROOT output** (`run_h5toROOT.py`) needs PyROOT, which this environment does not have;
  uproot (included) reads ROOT files.
- **The event's other hard scatterings** (from 0.2.0; [`README_extra_hard.md`](../docs/README_extra_hard.md)):
  `--extra-hard` adds them to the triggering one, `--hard none --extra-hard` makes minimum-bias
  events. The O+O and Ne+Ne XMLs at 5.36 TeV are in `config/`:

  ```bash
  cd ~/xscape/examples/prod_AuAu_0_10_jet
  python run_prod_jet.py --events 10 --seed 1 --extra-hard --hadrons-only --outdir ~/xscape/prod/extra
  python run_prod_jet.py --events 10 --seed 1 --user-xml ../config/NeNe_5360_jet.xml \
      --grid ../config/grid_oo.yaml --extra-hard --hadrons-only --out NeNe_5360_seed0001.h5 \
      --outdir ~/xscape/prod/nene
  TAG_PREFIX=OO_5360_seed ./run_jobs.sh --campaign mb1 4 30 1 ~/xscape/prod/oo_mb \
      --user-xml ../config/OO_5360_jet.xml --grid ../config/grid_oo.yaml \
      --hard none --extra-hard --hadrons-only
  python ../analysis/bulk/bulk_obs.py ~/xscape/prod/oo_mb      # Nch, [pT], v_n{2}
  ```

  These XMLs need X-SCAPE's `<extraHard>` (0.2.0 or later). In an older environment
  `xscape-run` and the scripts stop with `tags of ... that the main XML ... lacks: <extraHard>`.

---

## 2 Your own scripts and notebooks

### 2.1 Start from the examples

`xscape-examples DEST` copies every example of PyJetscape into `DEST`: the production
scripts, the Pythonic examples (`pythonic/`), the toy modules (`python_modules/`), the
analysis notebooks and their XMLs. It never overwrites; for a fresh copy give a new
directory.

```bash
xscape-examples ~/xscape/examples          # once (done in Get started)
cd ~/xscape/examples/pythonic
python oo_workflow.py --events 3 --seed 7 --outdir ~/xscape/out_oo
```

[`oo_workflow.py`](../contribs/PyJetscape/example/pythonic/oo_workflow.py)
takes any user XML with `--xml`. Its notebook version, `oo_workflow.ipynb` (the same run, then
the hadron files read back, awkward arrays, spectra with errors, FastJet), runs in
[Jupyter](#jupyter).

### 2.2 One XML: xscape-run

`xscape-run -w RUN_DIR -u USER_XML` runs X-SCAPE's `runJetscape` on a user XML in a directory
of its own, with no Python of yours. It takes any XML as it is: the examples' and X-SCAPE's
own test XMLs (in `xscape.TREE / "config"`). Your file is not changed:

```bash
xscape-run -w ~/xscape/runs/oo -u ~/xscape/examples/pythonic/oo_workflow.xml
xscape-run -w ~/xscape/runs/pp -u "$(python -c 'import xscape; print(xscape.TREE)')/config/jetscape_user.xml"
```

`RUN_DIR` gets copies of both XMLs, `jetscape_main.xml` and `<name>.run.xml`, with every
`../` path made absolute. These paths are written relative to an X-SCAPE build directory
(`../external_packages/iSS/...`, `../config/...`) and are found in the installed X-SCAPE. Also
in `RUN_DIR`: links to the tables, a private `music_input` and every output. Give your own
input files as absolute paths.

Before anything runs, `xscape-run` checks the XML and stops (exit status 3, nothing run) if:

| it reports | because |
|---|---|
| `needs modules this xscape wheel does not have` | the XML asks for ROOT writers, HepMC, SMASH, FreestreamMilne, CLVisc or IPGlasma, which the wheel is built without; X-SCAPE would only warn and run without them |
| `turns automatic module setup off` | the XML is written for a C++ test program that builds its own pipeline; `runJetscape` would run nothing |
| `asks Pythia for LHAPDF6 sets` | the wheel's Pythia has no LHAPDF6 plugin and would crash |
| `tags of ... that the main XML ... lacks` | a misspelled tag, or an XML newer than this X-SCAPE (e.g. `<extraHard>` in an environment older than 0.2.0) |
| `inputs with no file behind them` | an input this X-SCAPE does not have (SMASH's, the test hydro files, Martini's data), or a wrong path |

`-f` runs anyway. Plain `runJetscape USER_XML` runs in the current directory, which must have
what the XML reads there (3dMCGlauber's `tables/`, MUSIC's `music_input`, the `../` paths). It
stops and points to `xscape-run` when those are missing.

### 2.3 A script of your own

Put your script and its user XML in a directory of yours (start from a copy of
`examples/pythonic/oo_workflow.xml` and `oo_workflow.py`). The skeleton every such script has:

```python
# ~/xscape/mine/my_job.py
import os
import jetscape as js
from jetscape.hadrons_h5 import HadronH5Writer

out = os.path.expanduser("~/xscape/out_mine")
with js.JobDir(out, os.path.expanduser("~/xscape/mine/my_job.xml")) as job:   # private working directory
    with js.per_event(job.main_xml, job.user_xml) as driver, \
         HadronH5Writer(f"{out}/my_job_hadrons.h5", tag="jet_frag", n_samples=1) as h5:
        for i in driver.events():                                          # one iteration per event
            jets = driver.hadronization_manager.hadron_array()
            h5.append_unit(jets.to_dict(), unit=i, event=i)
            print(i, len(jets), "jet hadrons")
```

```bash
cd ~/xscape/mine && python my_job.py
```

- `js.JobDir(out_dir, user_xml)` copies your XML to `out_dir/`, points it at the installed
  X-SCAPE's tables, gives the job a private working directory `out_dir/work/<name>` and removes
  it after a successful run. Use absolute output paths: inside the `with` the current directory
  is that working directory. It checks the XMLs as `xscape-run` does: a tag the main XML lacks
  or an input with no file behind it raises `jetscape.workdir.XMLProblem` before X-SCAPE starts.
- `driver.events()`, `final_parton_array()`, `hadron_array()`, the hadron files, writing a
  module of your own in Python: [`PYTHONIC.md`](../contribs/PyJetscape/PYTHONIC.md).
- **Modules of your own in Python** (a hard process, energy loss, hadronization, an afterburner,
  a writer) run in X-SCAPE's event loop with this environment as they are, next to the C++
  modules: start from `examples/python_modules/` (`python run_by_hand_pythonic.py --style new`;
  the modules are in `toy_modules_pythonic.py`, an XML that names them in `user_from_xml.xml`).
- X-SCAPE's defaults are in `config/jetscape_main.xml` of the installed X-SCAPE
  (`python -c "import xscape; print(xscape.TREE / 'config')"`).

### 2.4 Python packages that are not in the environment

```bash
uv pip install seaborn
```

`uv pip sync` (of the same or a new lock) removes them again: keep your additions in a file
(`my_packages.txt`) and run `uv pip install -r my_packages.txt` after each sync. Don't
change numpy or h5py this way.

### 2.5 Developing X-SCAPE or PyJetscape: not with this environment

This environment is precompiled: X-SCAPE, MUSIC4GPU, iSS, 3dMCGlauber, the PyJetscape
bindings and the libraries they need were built once, for the release. It runs them, and
you can build on top of them in Python, but it can't rebuild them.

| you want to | use |
|---|---|
| run productions; write your own scripts, notebooks and Python modules ([`PYTHONIC.md`](../contribs/PyJetscape/PYTHONIC.md)); analyse | this environment |
| change X-SCAPE's C++ (modules, framework), MUSIC4GPU (Metal), iSS, 3dMCGlauber, the PyJetscape bindings, or js-contrib's `jetscape` package | a **source build** in the **native conda environment**: Xcode with its Metal toolchain, Homebrew's libraries, a conda environment with Python and pybind11 ([js-contrib README, *Getting started*](../README.md#getting-started-pick-your-setup); step by step: [`contribs/README.md`](../contribs/README.md)) |

Both can live on one Mac. Use them in separate Terminal windows: the source build's conda
environment and variables (`PYTHONPATH`, `XSCAPE`, ...) must not be active where this
environment runs ([rule 4](#the-rules)).

---

## 3 Analysis

The same environment is the analysis environment; no GPU is needed.

```bash
cd ~/xscape/examples
python my_analysis.py ~/xscape/prod/out
```

Reading the files: `js.HadronArray.from_h5("..._hadrons_jet_frag.h5")`, `.to_awkward()`,
`js.select(...)`, FastJet on the jagged arrays
([`PYTHONIC.md`](../contribs/PyJetscape/PYTHONIC.md));
readers for whole campaigns and the analysis notebooks:
[`README_analysis.md`](../docs/README_analysis.md).
Notebooks to start from, in your copy of the examples:

| notebook | reads |
|---|---|
| `pythonic/oo_workflow.ipynb` (sections 4-6) | the hadron files of your own run |
| `pythonic/oo_analysis.ipynb` | the O+O hadron files streamed from the OSDF (pelicanfs is in the environment from 0.2.0) |
| `analysis/tutorials/hadron_distributions.ipynb`, `jet_fastjet*.ipynb` | a production directory |
| `prod_AuAu_0_10_jet/jet_wake.ipynb` | a pair file and its hadron files |
| `prod_OO_isr/oo_isr_overview.ipynb` | an O+O production |
| `analysis/bulk/extra_hard_bulk_jets.ipynb`, `bulk_obs.py` | a minimum-bias or extras campaign (`--extra-hard`) |

**Analysis on another computer** (Linux, an Intel Mac, a cluster) needs only the Python
package, not X-SCAPE: in any Python 3.10+ environment,

```bash
uv pip install "pyjetscape[analysis] @ https://github.com/jhputschke/xscape-macos/releases/download/v$VER/pyjetscape-1.4-py3-none-any.whl"
```

(quote it: zsh treats `[...]` as a pattern). The readers, `HadronArray`, the awkward views and
the notebooks work there; running X-SCAPE does not.

---

## 4 Productions and data

js-contrib's production tools are commands of the environment. They run the copies of
js-contrib's `utils/` scripts that the wheel ships, in this environment, from any directory.
Each command's `--help` is the script's.

**Plan** a production with the settings measured on an M3 Max: the commands, events per hour,
duration, disk and memory, with checks (events per job, memory, campaign name). It prints the
plan, or saves it as a script (`--save launch.sh`). It never starts jobs:

```bash
xscape-prodwizard --list                                   # productions and machines
xscape-prodwizard --preset auau_jet --machine mac_wheel --events 300 --campaign c1
```

`--machine mac_wheel` (the default inside this environment) writes commands for it. It
activates `$VIRTUAL_ENV` (default `~/.venvs/xscape`), `cd`s into your copy of the examples
(`$XSCAPE_EXAMPLES`, default `~/xscape/examples`), runs `run_jobs.sh` and watches with
`xscape-jobmon`.

**Run and watch:**

```bash
cd ~/xscape/examples/prod_AuAu_0_10_jet
OMP_NUM_THREADS=5 OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0 \
  caffeinate -i ./run_jobs.sh -j 3 --campaign c1 20 15 0 ~/xscape/prod/c1
xscape-jobmon ~/xscape/prod/c1 --tui                       # in a second Terminal window
xscape-check-deposition ~/xscape/prod/c1                   # jobs whose jets deposited too little
```

**Look into a file:** `xscape-h5-inspect FILE.h5 [--stats]` shows any HDF5 file, with
js-contrib's pair, hydro and hadron formats decoded.

**Move files to and from the OSDF** (`osdf:///fno4hic`). Reading the public namespace needs no
login. Writing needs a browser login once per Mac, or a token file (`--token-file`):

```bash
xscape-osdf login                                          # once; xscape-osdf status: who, how long
xscape-osdf upload ~/xscape/prod/c1 --what h5              # -> osdf:///fno4hic/c1/
xscape-osdf ls c1 --what h5
xscape-osdf download c1 --what h5 --to ~/xscape/data       # -> ~/xscape/data/c1/
xscape-upload-follow ~/xscape/prod/c1 c1 --follow          # upload jobs as they finish
```

These run in this environment, which has pelicanfs: they make no environment of their own in
`~/.cache/js_osdf`, as the scripts do elsewhere. `xscape-gcs` does the same for Google Cloud
Storage with a service-account key; on its first use it makes its own environment in
`~/.cache/js_gcs` (google-cloud-storage is not in this one).

The tools' documentation: [`utils/README.md`](README.md),
[`remote_transfer/README.md`](remote_transfer/README.md),
[`README_launch.md`](README_launch.md).

---

## Jupyter

```bash
source ~/.venvs/xscape-VER/bin/activate
cd ~/xscape/examples
jupyter lab
```

It opens in your browser. The notebooks run with the environment's Python ("Python 3
(ipykernel)"); simulations in a notebook use the GPU as scripts do. Stop it with Ctrl-C
(twice) in the Terminal.

**VS Code:** open `~/xscape/examples`, then "Python: Select Interpreter" (and for notebooks
"Select Kernel") → `~/.venvs/xscape-VER/bin/python`.

---

## What runs with the wheel

The wheel is built without ROOT, HepMC, SMASH, FreestreamMilne, CLVisc, IPGlasma, the FNO
hydro (torch), LHAPDF6 for Pythia, Martini's data and X-SCAPE's test hydro files. What needs
them stops with a message naming what is missing; for those, use a source build or the
production container.

**X-SCAPE's test XMLs** (`xscape.TREE / "config"`), with `xscape-run`:

| | XMLs |
|---|---|
| run | `jetscape_user.xml`, `_brick_hybrid_hadronization`, `_DIS`, `_MUSIC`, `_OO_ISR_MatterLBT_iColored`, `_photoproduction`, `_twostagehydro` |
| stop: module not in the wheel | `_rMatterLbt`, `_root_test` (the HepMC writer: X-SCAPE would skip it without a word; `-f` runs them without it); `BulkFastTest/*` (3), `_dAu3d` (ROOT bulk writer); `_3DGlauber_MUSIC_iSS_SMASH_test`, `_test_smash_RC1_1` (SMASH); `_AA_dukeTune`, `_AA_dukeTune_InitialFromFile`, `_FSFileTest`, `_sims` (FreestreamMilne); `jetscape_clvisc` (CLVisc); `_IPGlasmaMUSIC` (IPGlasma) |
| stop: automatic module setup off (for C++ test programs) | `FVvsMUSIC/*` (3), `_iMATTERMCGlauber*` (3), `_MUSICMainClockTest`, `_SMASHInitialCondition`, `_SMASHNucleusTest`, `_test` |
| stop: input not shipped | `_HydroFile`, `_PbPb_hard_nPDF` (`../examples/test_hydro_files`) |
| stop: tag unknown to X-SCAPE | `_MCGlauberMUSIC`, `3D/*` (2): `<nEvents_MCGla>` is in no main XML |
| stop: LHAPDF6 | `_nPDF_test` |

**The examples** (your `xscape-examples` copy): everything runs but:

| example | needs |
|---|---|
| `python_fno_test.py` | torch, the FNO hydro |
| `python_fast_bulk_root_writer.py`, `validate_h5_vs_root.py`, `analysis_root/`, `run_h5toROOT.py` | ROOT (PyROOT) |
| `music_cpu_gpu_hydro.py` | a CPU MUSIC build next to the GPU one |
| `hadro_compare/run_hadro_compare.sh` | two builds |
| the movie step of `analysis/zjet/test/make_data.sh` | js-contrib's Visualization contrib |

The analysis notebooks and scripts need production output: yours, or files from the OSDF.

---

## Updating and removing

**A new release:** a new environment next to the old one, so a running campaign keeps its
software:

```bash
VER=<new version>
cd ~/xscape
curl -LO https://github.com/jhputschke/xscape-macos/releases/download/v$VER/xscape-$VER-cp313.lock.txt
uv venv --managed-python ~/.venvs/xscape-$VER --python 3.13 && source ~/.venvs/xscape-$VER/bin/activate
uv pip sync xscape-$VER-cp313.lock.txt
xscape-fetch-data lbt                    # reuses the tables in the cache when they match
xscape-selftest
xscape-examples ~/xscape/examples-$VER   # the examples of this release
xscape-utils ~/xscape/utils-$VER         # its tools, to read (optional)
```

**Removing:** `rm -rf ~/.venvs/xscape-VER` (one environment),
`rm -rf ~/Library/Caches/xscape` (the LBT tables), `uv cache clean` (uv's downloads).
uv itself: `rm ~/.local/bin/uv ~/.local/bin/uvx`.

---

## Troubleshooting

| symptom | cause, fix |
|---|---|
| `zsh: command not found: uv` | open a new Terminal window, or `source ~/.local/bin/env` |
| `... has no wheels with a matching platform tag` | the Mac runs macOS older than 26, or has an Intel processor: update macOS; Intel Macs are not supported |
| `zsh: no matches found: pyjetscape[analysis]...` | quote the argument: `"pyjetscape[analysis] @ ..."` |
| `ModuleNotFoundError: No module named 'jetscape'` | the environment is not active: `source ~/.venvs/xscape-VER/bin/activate` |
| `import jetscape` takes a source checkout, or `JobDir: not found: .../music_input` | `PYTHONPATH`, `XSCAPE` or `JETSCAPE_BUILD` set (e.g. in `~/.zshrc`): `unset PYTHONPATH XSCAPE JETSCAPE_BUILD` |
| LBT: `cannot open ... LBT-tables` | step 3 not done in this environment: `xscape-fetch-data lbt` |
| `xscape-fetch-data: ... file(s) missing or different` | an interrupted download: `rm -rf ~/Library/Caches/xscape/LBT-tables-*` and run it again |
| `xscape-examples: ... exists and is not empty` | it never overwrites: give another directory |
| `python: can't open file 'run_prod_jet.py'` | run it from `~/xscape/examples/prod_AuAu_0_10_jet` (the copy), not from elsewhere |
| `sh: ./Metropolis.e: No such file or directory`, then `Can not generate ...` | `runJetscape` (0.1.0) in a directory without 3dMCGlauber's `tables/`: use `xscape-run -w RUN_DIR -u USER_XML`. From 0.2.0 `runJetscape` stops first and says so |
| `runJetscape: ... reads files relative to the current directory` | the same: `xscape-run` (section [2.2](#22-one-xml-xscape-run)); `XSCAPE_RUN_HERE=1` runs there anyway |
| `xscape-run: ... needs modules this xscape wheel does not have`, `... automatic module setup off`, `... LHAPDF6` | the XML can't run with the wheel ([What runs with the wheel](#what-runs-with-the-wheel)); `-f` runs anyway |
| `tags of ... that the main XML ... lacks: <extraHard>` | an XML newer than the environment's X-SCAPE: install the current release |
| `ValueError: Protocol not known: osdf` | an environment older than 0.2.0, which has no pelicanfs: install the current release, or `uv pip install pelicanfs` |
| `js_osdf: No module named 'google_crc32c' ... ~/.cache/js_osdf/venv` | the same, with `utils/remote_transfer/js_osdf.py` from a copy: use `xscape-osdf` of a 0.2.0 environment |
| MUSIC slow, no GPU | inside a virtual machine or a container there is no Metal GPU: run natively in Terminal |
| `ModuleNotFoundError: ROOT` | ROOT is not in this environment (see section 1); uproot reads ROOT files |

---

## Reference

**Commands** (in the activated environment):

| command | does |
|---|---|
| `xscape-fetch-data lbt [--from DIR]` | the LBT tables into `~/Library/Caches/xscape` (`$XSCAPE_CACHE`), checked, linked into this environment; `--from` uses tables already on disk |
| `xscape-selftest` | checks the environment ([Step 4](#get-started)) |
| `xscape-examples [DEST]` | copies the examples to `DEST` (default `./examples`); never overwrites |
| `xscape-utils [DEST]` | copies js-contrib's `utils/` to `DEST` (default `./utils`); never overwrites |
| `xscape-run -w RUN_DIR -u USER_XML [-m MAIN_XML] [-f]` | X-SCAPE's `runJetscape` on a user XML in its own directory (`run_in_workdir.sh`), the XML's `../` paths made absolute in copies, checked first ([2.2](#22-one-xml-xscape-run)); e.g. 2 pp events: `xscape-run -w ~/xscape/runs/pp -u "$(python -c 'import xscape; print(xscape.TREE)')/config/jetscape_user.xml"` |
| `runJetscape USER_XML [MAIN_XML]` | X-SCAPE's executable with this environment's data, in the current directory (stops if the XML needs files the directory lacks) |
| `xscape-prodwizard [--preset P --machine mac_wheel --events N ...]` | plans a production: commands, rates, disk, memory, checks (`utils/prodwizard.py`; no options: the terminal UI) |
| `xscape-jobmon OUTDIR [--tui] [--watch]` | a `run_jobs.sh` production's progress, rate, ETA and failures (`utils/jobmon.py`) |
| `xscape-check-deposition DIR...` | jet jobs whose jets deposited far less than the campaign's median (`utils/check_deposition.py`) |
| `xscape-h5-inspect FILE... [--stats]` | any HDF5 file, js-contrib's formats decoded (`utils/h5_inspect.py`) |
| `xscape-osdf {login,status,ls,du,upload,download,rm} ...` | production files on the OSDF, `osdf:///fno4hic` (`utils/remote_transfer/js_osdf.py`, in this environment) |
| `xscape-upload-follow OUTBASE REMOTE [--follow] [--delete ...]` | uploads a production's finished jobs while it runs (`utils/upload_follow.py`) |
| `xscape-gcs {ls,upload,download,...} ...` | the same for Google Cloud Storage (`utils/remote_transfer/js_gcs.py`; its own environment in `~/.cache/js_gcs`) |

**Python:** `import jetscape as js` (PyJetscape; `js.JobDir`, `js.per_event`, the readers);
`jetscape.paths.xscape_root()`, `build_dir()`, `main_xml()`, `examples_dir()`, `utils_dir()`;
`xscape.TREE` (the installed X-SCAPE), `xscape.PACKAGE`.

**Variables:**

| variable | default | for |
|---|---|---|
| `XSCAPE_CACHE` | `~/Library/Caches/xscape` | where `xscape-fetch-data` keeps the LBT tables |
| `XSCAPE`, `JETSCAPE_BUILD` | unset | a source build instead of the installed X-SCAPE; leave unset |
| `XSCAPE_RUN_HERE` | unset | `1`: `runJetscape` runs in the current directory even if it lacks what the XML reads there |
| `XSCAPE_EXAMPLES` | `~/xscape/examples` | `xscape-prodwizard --machine mac_wheel`: your copy of the examples |
| `OMP_NUM_THREADS`, `OMP_WAIT_POLICY` | all cores, active | threads per job; `passive` when several jobs share the Mac |

`PYTHIA8DATA`, `XSCAPE_DATA_DIR`, `HYDROPROGRAMPATH` and `LBT_TABLES_PATH` are set by the
environment itself for each run.
