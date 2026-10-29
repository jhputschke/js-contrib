# Analysis environment

The analysis environment is a plain Python venv for reading and analysing the files a
production writes, on any computer with Python 3.10 or later. It needs no GPU, no X-SCAPE
build, no compiled `pyjetscape_core`, no conda and no ROOT. Use it when someone else runs the
productions and you get the files (from a disk or from the [OSDF](../run/osdf.md)), or when you
analyse on a laptop and simulate elsewhere. It holds the HDF5 readers of the `jetscape`
package, uproot and awkward, FastJet, vector, particle, hist, mplhep, scipy, pandas, Jupyter,
pyvista for the 3D visualization, and pelicanfs for `osdf://` files.

!!! tip "You may already have one"
    The [production container](container.md) (run without `--gpus`) and the
    [xscape wheel](mac.md) on a Mac carry the same analysis packages. This page is for
    computers that have neither.

## What works in it, and what doesn't

| works | doesn't |
|---|---|
| every HDF5 file of a production: pair, particlize and hadron files (`jetscape.hadrons_h5`, `jetscape.particlize_h5`, `jetscape.showers`) | anything that runs X-SCAPE: `run_prod_jet.py`, `run_jobs.sh` |
| `run_h5toROOT.py` (hadron files → ROOT ntuples, written with uproot) and reading ROOT files with uproot | re-hadronization (`hadronize.py`, `run_hadronize.py`): iSS and Pythia live in the compiled core |
| the analysis scripts and notebooks (`analysis/`, `jet_wake.ipynb`, the FastJet tutorials) | the ROOT macros and the C++ reader, unless ROOT is added ([below](#4-root-next-to-it)) |
| the visualization of stored files (`wake_pyvista.py`, `hydro_jet_particles_pyvista.py --file`) | the visualization of a live event |

`jetscape.HAS_CORE` is `False` in this venv. That is expected: every reader works without the
compiled core.

## 1. Make the venv

```bash
git clone https://github.com/jhputschke/js-contrib.git
cd js-contrib
./utils/analysis_env/setup_analysis_env.sh           # venv in ~/.venvs/js_analysis
```

The script

1. creates the venv (default `~/.venvs/js_analysis`; give another directory as the first
   argument). Re-running it on an existing venv updates it;
2. installs the analysis packages from `utils/analysis_env/requirements.txt`, plus pyvista/vtk
   and pelicanfs (both on by default), about 1.6 GB in total;
3. installs PyJetscape from your checkout in editable mode, so `import jetscape` works from
   anywhere and follows `git pull`;
4. downloads MUSIC's hotQCD EoS table (3.2 MB, md5-checked) into the venv and sets
   `MUSIC_EOS_TABLE` in its `activate` script. The pair files store the energy density, not
   the temperature; `jet_wake.ipynb`, `analysis/wake_*.py` and the visualization use the table
   to convert e → T;
5. runs `check_env.py`.

Nothing outside the venv directory is written, except a Jupyter kernel spec if you ask for one.

## 2. Activate and check

```bash
source ~/.venvs/js_analysis/bin/activate
python utils/analysis_env/check_env.py /path/to/out    # opens every file of a production directory
```

`check_env.py` checks the packages, the Blosc filter (the HDF5 files are Blosc-compressed and
don't open without `hdf5plugin`), PyROOT if present, the EoS table and off-screen rendering.
With a directory it opens its pair, hadron and ROOT files. Two flags:

| flag | does |
|---|---|
| `--no-render` | skips the off-screen rendering test |
| `--remote-test` | also reads one public object over `gs://` and `osdf://` |

## 3. Options

```bash
./utils/analysis_env/setup_analysis_env.sh [VENV_DIR] [options]
```

| option | |
|---|---|
| `--uv` | create the venv and install with [uv](https://docs.astral.sh/uv/): much faster (uv must be on `PATH`) |
| `--python PY` | the interpreter for the venv (default `python3`; 3.10 or later) |
| `--no-viz` | leave out pyvista/vtk/ffmpeg (about 0.6 GB of the 1.6 GB) |
| `--with-fasthydro` | also install [FastHydro](../python/fasthydro.md), for FastHydro's own files and readers; **not needed for the production files**. Its solver also needs `pip install torch` |
| `--system-site-packages` | let the venv see the packages of the Python it is made from, e.g. PyROOT of a conda env ([§4](#4-root-next-to-it)); on an existing venv it switches this on |
| `--with-gcs` | add Google Cloud Storage (`gcsfs`, `google-cloud-storage`): `gs://` URLs for fsspec, h5py and uproot |
| `--no-pelican` | leave out pelicanfs, which is installed by default: `osdf://` and `pelican://` URLs |
| `--eos-table PATH` | copy the EoS table from `PATH` (an `EOS/hotQCD` directory of a MUSIC or X-SCAPE build, or the `.dat` file) instead of downloading it |
| `--no-eos` | neither download nor copy it |
| `--kernel NAME` | register a Jupyter kernel `NAME` for the venv (for VS Code or a JupyterLab elsewhere) |
| `--check DIR` | at the end, open the production files in `DIR` |

Full description: [`utils/analysis_env/README.md`](https://github.com/jhputschke/js-contrib/blob/main/utils/analysis_env/README.md)
and `./utils/analysis_env/setup_analysis_env.sh --help`.

## 4. ROOT next to it

ROOT isn't needed: `run_h5toROOT.py` falls back to uproot, and the ntuples hold the same
hadrons. You want ROOT for PyROOT and RDataFrame, the `root` prompt and browser, the ROOT macros
and the C++ reader, or the `--bits-p`/`--bits-x` options of `run_h5toROOT.py`. A venv sees
ROOT only when it is made **from the Python that ROOT was built for**, with
`--system-site-packages`.

=== "conda with ROOT + the venv (tested)"

    ```bash
    # 1. conda, once (skip if you have it): Miniforge, conda-forge by default
    curl -fLO "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash "Miniforge3-$(uname)-$(uname -m).sh" -b -p ~/miniforge3
    source ~/miniforge3/bin/activate

    # 2. an env with ROOT (ROOT >= 6.34 reads the RNTuple files run_h5toROOT.py writes)
    conda create -n root -c conda-forge "root>=6.34"
    conda activate root

    # 3. the venv, from this env's Python
    cd js-contrib
    ./utils/analysis_env/setup_analysis_env.sh ~/.venvs/js_analysis_root --system-site-packages
    ```

    `check_env.py` then prints `ok PyROOT 6.xx ... writes with ROOT`. Afterwards PyROOT works
    from the venv alone; for the `root` prompt and compiled macros, `conda activate root`
    (without it ACLiC doesn't find the system headers: `'assert.h' file not found`):

    ```bash
    source ~/.venvs/js_analysis_root/bin/activate
    python contribs/PyJetscape/example/prod_AuAu_0_10_jet/run_h5toROOT.py /path/to/out -j 4

    conda activate root
    root -l -b -q 'contribs/PyJetscape/example/analysis_root/hadron_distributions.C+("/path/to/out")'
    ```

    The same works with any other ROOT that has PyROOT (LCG/CVMFS, Homebrew, a source build):
    make the venv from the `python3` in which `import ROOT` works, with
    `--python "$(which python3)" --system-site-packages`.

=== "one conda env with everything (untested)"

    ```bash
    cd js-contrib
    conda env create -f utils/analysis_env/environment.yml    # env "js_analysis"
    conda activate js_analysis
    python utils/analysis_env/check_env.py /path/to/out
    ```

    Everything comes from conda-forge except the scikit-hep `fastjet` and PyJetscape (editable).
    The EoS table is not part of it; add it with

    ```bash
    mkdir -p "$CONDA_PREFIX/share/music_eos/hotQCD"
    curl -fL -o "$CONDA_PREFIX/share/music_eos/hotQCD/hrg_hotqcd_eos_binary.dat" \
      https://api.bitbucket.org/2.0/repositories/wayne_state_nuclear_theory/hotqcd/src/main/hrg_hotqcd_eos_binary.dat
    conda env config vars set MUSIC_EOS_TABLE="$CONDA_PREFIX/share/music_eos/hotQCD/hrg_hotqcd_eos_binary.dat"
    conda activate js_analysis                               # again, to pick up the variable
    ```

More on ROOT, RDataFrame and the C++ reader: [ROOT and C++](../analysis/root.md).

## 5. FastJet

| | Python | C++ (ROOT macros, programs) |
|---|---|---|
| package | scikit-hep `fastjet` (the FastJet library is in the wheel): the classic SWIG interface and the array (awkward) interface | conda-forge `fastjet-cxx` (FastJet, `fastjet-config`) and `fastjet-contrib` (fjcontrib as one library, `libfastjetcontribfragile`) |
| install | already in the venv | `conda install -c conda-forge fastjet-cxx fastjet-contrib` in the ROOT env |
| fjcontrib | only SoftDrop, Lund declusterings, energy correlators and N-subjettiness, through the array interface | all of it (`#include <fastjet/contrib/SoftDrop.hh>`, ...; link `-lfastjetcontribfragile`) |

Jet finding on the hadron files: [Jets with FastJet](../analysis/jets.md).

## 6. Files on the OSDF or in Google Cloud Storage

pelicanfs (default) and gcsfs (`--with-gcs`) are fsspec file systems, so `osdf://` and `gs://`
URLs work wherever fsspec does: uproot takes the URL directly, and h5py takes the file object
fsspec opens (with `hdf5plugin` imported). Only the chunks you read travel over the network.
Reading the public `osdf:///fno4hic` namespace needs no login or token. How to find, download
and read the published productions: [Data on the OSDF](../run/osdf.md).

## 7. Jupyter

```bash
source ~/.venvs/js_analysis/bin/activate
cd js-contrib/contribs/PyJetscape/example/prod_AuAu_0_10_jet
jupyter notebook jet_wake.ipynb
```

The notebooks name the kernel they were saved with. If Jupyter asks, choose
"Python 3 (ipykernel)": started from the activated venv, that is the venv. For a Jupyter
elsewhere (VS Code, a JupyterLab of another env), register the venv with `--kernel js_analysis`
and choose that kernel. The notebooks read `out/` by default; point them elsewhere with
`WAKE_H5=/path/to/<stem>.h5` (`jet_wake.ipynb`) or `HADRON_DIR=/path/to/out` (the `analysis/`
notebooks).

**Headless machines:** the visualization renders off-screen and `check_env.py` tests that it
works. If it fails on a machine without a display, run under `xvfb-run` (Linux,
`apt install xvfb`).

## Without a checkout: the pyjetscape wheel

The `jetscape` package with the same analysis packages also comes as a pure-Python wheel, part
of every [xscape-macos release](https://github.com/jhputschke/xscape-macos/releases). It
installs on any platform, in any Python 3.10+ environment:

```bash
uv pip install "pyjetscape[analysis] @ https://github.com/jhputschke/xscape-macos/releases/download/v0.1.0/pyjetscape-1.3-py3-none-any.whl"
```

Quote the argument: zsh treats `[...]` as a pattern. The readers, `HadronArray`, the awkward
views and the notebooks work with it; running X-SCAPE does not. Unlike the setup script, it
installs neither pyvista nor pelicanfs (add `pelicanfs` with `uv pip install pelicanfs` for the
OSDF), and it does not fetch the EoS table: download `hrg_hotqcd_eos_binary.dat` as in the
conda tab above and point `MUSIC_EOS_TABLE` at it.

## Threads and memory

The analysis tools are mostly single-threaded Python. Parallelise over files with processes,
and keep processes × threads at or below the number of cores.

| tool | parallel with | memory |
|---|---|---|
| `HadronFileReader` | one process; split a campaign by `events=` over processes | the units it reads: a background of 2000 oversamples is ~14 M hadrons, ~1.5 GB with the derived kinematics (estimate) |
| `wake_observables.py -j N`, `wake_hadrons.py -j N` | N processes over files | about +0.5 GB per process for the read-ahead (`wake_observables.py`) |
| `run_h5toROOT.py -j N` | N processes | ~80 B per hadron of the largest tag: ~3.5 GB for a 15-event file at 400 oversamples |
| `hadron_distributions.C` (RDataFrame) | its `threads` argument | |
| numpy (OpenBLAS) | `OPENBLAS_NUM_THREADS` | set it to 1 when running with `-j`, or the processes oversubscribe the cores |
| Blosc decompression (h5py) | `BLOSC_NTHREADS` | helps little; leave it unset with `-j` |

Measured times and more: [`docs/README_analysis.md`](https://github.com/jhputschke/js-contrib/blob/main/docs/README_analysis.md#threads-and-memory).

## Next: start analysing

[Analyse](../analysis/index.md) walks through what a production writes and how to read it:
hadrons, jets with FastJet, hydro and hadrons together, O+O, and ROOT.
