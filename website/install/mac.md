# Mac (xscape wheel)

On an Apple Silicon Mac, X-SCAPE runs natively with MUSIC on the Mac's own GPU through Metal.
It comes as a Python wheel, `xscape`, installed with [uv](https://docs.astral.sh/uv/) from a
lock file. There is no compiler and no container involved. Docker on a Mac can't reach the
GPU, so on a Mac the wheel takes the place of the [production container](container.md); the
scripts, notebooks and outputs are the same. The environment holds X-SCAPE with MUSIC
(MUSIC4GPU, Metal), iSS, 3dMCGlauber, Matter, LBT, iColored and Pythia, PyJetscape and its
Pythonic layer, and the analysis packages (awkward, vector, fastjet, particle, hist, mplhep,
matplotlib, scipy, pandas, uproot, JupyterLab).

**What it can't do:** rebuild X-SCAPE or the bindings (that needs the
[source build](source.md)), ROOT/PyROOT (uproot, included, reads ROOT files), and PyTorch (not
in the environment). Intel Macs are not supported.

## Requirements

- A Mac with Apple Silicon (M1 or later) on **macOS 26** or later (Apple menu → About This
  Mac).
- About 3 GB of disk, 1 GB of it uv's download cache (`uv cache clean` frees it).
- Internet access during the installation.

You do **not** need Xcode, Homebrew, conda, Docker, git or an Apple developer account.

!!! info "Releases"
    The wheels are published on the
    [Releases page of jhputschke/xscape-macos](https://github.com/jhputschke/xscape-macos/releases).
    The current release is **v0.1.0**; check that page for the latest. Each release is a set
    of three files: the `xscape` wheel (X-SCAPE, 175 MB), the `pyjetscape` wheel (the Python
    package), and a **lock file** `xscape-<version>-cp313.lock.txt`, which names every package
    of the environment with its exact version and checksum. You install from the lock file,
    so every Mac gets exactly the same software.

## Step 1: install uv

uv is Astral's Python package manager: a single program in `~/.local/bin`.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then open a new Terminal window (or run `source ~/.local/bin/env`) so that `uv` is found. With
Homebrew, `brew install uv` works too.

## Step 2: create the environment from the lock file

```bash
VER=0.1.0                     # the latest: github.com/jhputschke/xscape-macos/releases
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
  you use X-SCAPE. Your prompt then starts with `(xscape-0.1.0)`.

## Step 3: fetch the LBT tables

LBT, the jet energy-loss module, needs its scattering tables: a 240 MB download, 1.1 GB on
disk. They are kept in `~/Library/Caches/xscape` and shared by all your environments.

```bash
xscape-fetch-data lbt
```

It checks every file against the release's checksums and ends with
`xscape-fetch-data: LBT tables ... (checked) -> ...`. If you already have the tables on disk,
`xscape-fetch-data lbt --from DIR` uses them instead of downloading.

## Step 4: check that the Metal GPU is used

```bash
python -c "import jetscape as js; print(js.HAS_CORE, js.pyjetscape_core.MUSIC_BACKEND)"
# -> True music4gpu-metal
```

`True` means the compiled bindings load; `music4gpu-metal` means they were built with the Metal
MUSIC. The first `import` after the installation takes a little longer, because matplotlib
builds its font cache once.

When an event runs, MUSIC prints the GPU it uses when it starts:
`[MUSIC-GPU] Metal device: <your chip>`. There is nothing to switch on. If instead you see
`[MUSIC-GPU] Metal init failed, falling back to CPU.`, MUSIC runs on the CPU; that happens
inside a virtual machine or a container, so run natively in Terminal.

## Step 5: copy the examples and run one event

```bash
xscape-examples ~/xscape/examples
cd ~/xscape/examples/pythonic
python oo_workflow.py --events 1 --outdir ~/xscape/out_oo
```

`xscape-examples` copies every example into a directory of yours: the production scripts, the
Pythonic examples (`pythonic/`), the toy Python modules (`python_modules/`), the analysis
notebooks and their XMLs. The examples write next to themselves, so they run from this copy.
It never overwrites: for a fresh copy, give it a new directory.

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
4. **Everything finds the installed X-SCAPE by itself** (`jetscape.paths`). Leave `XSCAPE`,
   `JETSCAPE_BUILD`, `PYTHONPATH` and `DYLD_LIBRARY_PATH` unset in this environment: they belong
   to a source build and would point it elsewhere.
5. **Results agree statistically, not bit for bit,** across Macs and macOS versions.

| | where |
|---|---|
| the environment | `~/.venvs/xscape-VER` |
| the installed X-SCAPE (data, `config/`, `BUILD_INFO.txt`) | `python -c "import xscape; print(xscape.TREE)"` |
| X-SCAPE's default parameters | `config/jetscape_main.xml` in that tree |
| the examples | your copy, e.g. `~/xscape/examples` |
| the LBT tables | `~/Library/Caches/xscape/LBT-tables-<commit>` |

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

The productions themselves: [Au+Au jets (two stages)](../run/auau-jet.md),
[O+O with ISR](../run/oo-isr.md), [Z+jet and other systems](../run/zjet-other.md).

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
| `pythonic/oo_analysis.ipynb` | the O+O hadron files streamed from the OSDF (first `uv pip install pelicanfs`) |
| `analysis/tutorials/hadron_distributions.ipynb`, `jet_fastjet*.ipynb` | a production directory |
| `prod_AuAu_0_10_jet/jet_wake.ipynb` | a pair file and its hadron files |
| `prod_OO_isr/oo_isr_overview.ipynb` | an O+O production |

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
| `xscape-examples [DEST]` | copies the examples to `DEST` (default `./examples`); never overwrites |
| `xscape-run -w RUN_DIR -u USER_XML` | X-SCAPE's `runJetscape` on a user XML, in a directory of its own |
| `runJetscape USER_XML` | X-SCAPE's executable with this environment's data, in the current directory |

For example, two proton-proton events with X-SCAPE's default user XML:

```bash
xscape-run -w ~/xscape/runs/pp -u "$(python -c 'import xscape; print(xscape.TREE)')/config/jetscape_user.xml"
```

## Updating and removing

**A new release** goes into a new environment next to the old one, so a running campaign keeps
its software:

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

The same guide, with every detail:
[`utils/README_Mac_Usage.md`](https://github.com/jhputschke/js-contrib/blob/main/utils/README_Mac_Usage.md)
(also the README of [jhputschke/xscape-macos](https://github.com/jhputschke/xscape-macos)).

## Next: run your first event

You ran one event in step 5; [Your first event](../run/first-event.md) explains what it did,
what the output files hold and where to go from there.
