# analysis_env — analyse `prod_AuAu_0_10_jet` files without X-SCAPE

A Python venv for people who get the files of a production and want to analyse or look at
them, but don't run X-SCAPE. It needs this js-contrib checkout and a Python ≥ 3.10. It
doesn't need X-SCAPE, MUSIC, ROOT, conda or the compiled `pyjetscape_core`.

```bash
git clone https://github.com/jhputschke/js-contrib.git
cd js-contrib
./utils/analysis_env/setup_analysis_env.sh           # venv in ~/.venvs/js_analysis
source ~/.venvs/js_analysis/bin/activate
python utils/analysis_env/check_env.py /path/to/out  # opens every file in a production directory
```

| file | purpose |
|---|---|
| `setup_analysis_env.sh` | makes or updates the venv, gets MUSIC's EoS table, runs `check_env.py` |
| `requirements.txt` | HDF5 readers, uproot, the analysis notebooks (fastjet, vector), Jupyter |
| `requirements_viz.txt` | pyvista (+ vtk), imageio, imageio-ffmpeg for `contribs/Visualization` |
| `requirements_gcs.txt`, `requirements_pelican.txt` | optional: Google Cloud Storage (gcsfs, google-cloud-storage) and Pelican/OSDF (pelicanfs), see [Remote files](#remote-files-google-cloud-storage-and-pelicanosdf) |
| `check_env.py` | checks the packages, the Blosc filter, PyROOT, the EoS table and off-screen rendering; with a directory, opens its pair, hadron and ROOT files |
| `environment.yml` | the same as one conda env, ROOT included (see [With ROOT](#with-root)) |

## What works in it

| | files it reads |
|---|---|
| `jetscape.hadrons_h5` (`HadronFileReader`, `HadronFile`), `jetscape.particlize_h5`, `jetscape.showers` | `<stem>_particlize.h5`, `<stem>_hadrons_*.h5`, `<stem>.h5` |
| [`run_h5toROOT.py`](../../contribs/PyJetscape/example/prod_AuAu_0_10_jet/run_h5toROOT.py): hadron files → RNTuple or TTree, written with uproot, or with ROOT if PyROOT imports ([With ROOT](#with-root)) | the hadron files |
| the ROOT files, read with uproot (`uproot.open(...)["bulk_jet"].arrays()`) | `*_hadrons.root`, `campaign_*.root` |
| [`jet_wake.ipynb`](../../contribs/PyJetscape/example/prod_AuAu_0_10_jet/jet_wake.ipynb) | a pair file and its hadron files |
| [`PyJetscape/example/analysis`](../../contribs/PyJetscape/example/analysis/README.md): `jet_edep_balance_check.py`, `wake_observables.py`, `wake_hadrons.py`, their notebooks, `hadron_distributions.ipynb`, `jet_fastjet*.ipynb` | a production directory |
| [`Visualization`](../../contribs/Visualization/README.md): `hydro_jet_particles_pyvista.py`, `wake_pyvista.py` (`--movie`, `--vtk-dir`) | a pair file (+ its hadron files) |

**What doesn't work in it:** anything that runs X-SCAPE, i.e. `run_prod_jet.py`,
`run_jobs.sh`, `hadronize.py`/`run_hadronize.py` (iSS and the Pythia fragmentation are in
`pyjetscape_core`), and `hydro_pyvista.py`/`hydro_jet_pyvista.py` without `--file`, which
evolve a live event. The ROOT macros and the C++ reader in
[`analysis_root`](../../contribs/PyJetscape/example/analysis_root/README.md) need ROOT itself
([With ROOT](#with-root)); the distributions of `hadron_distributions.C` are also in
`hadron_distributions.ipynb`.

## Options

```bash
./utils/analysis_env/setup_analysis_env.sh [VENV_DIR] [options]
```

| option | |
|---|---|
| `VENV_DIR` | where the venv goes (default `~/.venvs/js_analysis`). An existing venv is updated. |
| `--python PY` | the interpreter for the venv (default `python3`; ≥ 3.10) |
| `--uv` | make the venv and install with [uv](https://docs.astral.sh/uv/), much faster |
| `--system-site-packages` | let the venv see the packages of the Python it is made from, e.g. PyROOT of a conda env with ROOT ([With ROOT](#with-root)). On an existing venv, it switches this on. |
| `--no-viz` | skip `requirements_viz.txt` (pyvista/vtk/ffmpeg: ~0.6 GB of the ~1.6 GB) |
| `--with-fasthydro` | also install [FastHydro](../../contribs/FastHydro/README.md) (`jetscape-fasthydro`), for FastHydro's own files and readers (`fast_data`, `fasthydro.browse`). **Not needed for the `prod_AuAu_0_10_jet` files**; the visualization falls back to `jetscape.showers` without it. The FastHydro solver also needs `pip install torch`. |
| `--with-gcs` | also install the Google Cloud Storage interfaces, `requirements_gcs.txt`: gcsfs (`gs://` URLs for fsspec, h5py and uproot) and Google's `google-cloud-storage` client |
| `--with-pelican` | also install pelicanfs, `requirements_pelican.txt`: `osdf://` and `pelican://` URLs for fsspec, h5py and uproot |
| `--eos-table PATH` | copy MUSIC's hotQCD table from `PATH` (an `EOS/hotQCD` directory of a MUSIC or X-SCAPE build, or its `hrg_hotqcd_eos_binary.dat`) instead of downloading it |
| `--no-eos` | neither download nor copy it |
| `--kernel NAME` | register a Jupyter kernel `NAME` for the venv (in `~/.local/share/jupyter`) |
| `--check DIR` | at the end, open the production files in `DIR` |

PyJetscape is installed editable (`pip install -e contribs/PyJetscape --no-deps`), so
`import jetscape` works from anywhere and follows this checkout. `jetscape.HAS_CORE` is
`False` in the venv; that is expected, and every reader works without the core.

## With ROOT

ROOT isn't needed: `run_h5toROOT.py` falls back to uproot, and the ntuples it writes hold
the same hadrons. With PyROOT, `run_h5toROOT.py` writes with ROOT by default
(`--writer auto`), which is needed for `--bits-p`/`--bits-x`. With ROOT itself, you also
get the `root` prompt, the browser and the macros of `analysis_root`. A venv only
sees ROOT when it is made **from the Python that ROOT was built for**, with
`--system-site-packages`. There are two ways to get there.

### A. conda with ROOT, then the venv script (recommended)

conda provides ROOT and its Python, and the venv adds everything else on top. The venv
reuses what the conda env already has, such as its numpy, so pip replaces nothing under
ROOT.

```bash
# 1. conda, once: Miniforge (conda-forge by default). Skip this if you already have conda.
curl -fLO "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash "Miniforge3-$(uname)-$(uname -m).sh" -b -p ~/miniforge3      # -b: doesn't edit ~/.bashrc
source ~/miniforge3/bin/activate                                   # conda in this shell
#    (`conda init` makes this permanent; it edits your shell's startup file)

# 2. an env with ROOT (ROOT >= 6.34 reads the RNTuple files run_h5toROOT.py writes)
conda create -n root -c conda-forge "root>=6.34"
conda activate root

# 3. the venv, from this env's Python (python3 is now the env's)
cd js-contrib
./utils/analysis_env/setup_analysis_env.sh ~/.venvs/js_analysis_root --system-site-packages
```

`check_env.py`, which the script runs at the end, then prints `ok PyROOT 6.xx ... writes
with ROOT`. Afterwards:

```bash
source ~/.venvs/js_analysis_root/bin/activate      # PyROOT works from the venv alone
python contribs/PyJetscape/example/prod_AuAu_0_10_jet/run_h5toROOT.py /path/to/out -j 4

conda activate root                                # the root prompt and compiled macros
root -l -b -q 'contribs/PyJetscape/example/analysis_root/hadron_distributions.C+("/path/to/out")'
```

The macro needs `conda activate`: without it, ACLiC doesn't find the system headers
(`'assert.h' file not found`). PyROOT inside the venv doesn't need it. An existing venv can
be switched over by re-running the script with `--system-site-packages` from the activated
env. It must be a venv made from that env's Python, since `--system-site-packages` shows the
packages of the venv's own base Python.

The same works with any other ROOT that has PyROOT (LCG/CVMFS, Homebrew, a source build).
Make the venv from the `python3` that `import ROOT` works in:
`--python "$(which python3)" --system-site-packages`.

Tested on the GB10 (Linux aarch64) with an existing Miniconda env holding conda-forge
ROOT 6.36.06 and Python 3.14. The Miniforge installation in step 1 was not repeated here.
- The script made the venv in 22 s with a warm pip cache (1.2 GB). pip kept conda's numpy.
- `check_env.py --check` passed, including off-screen rendering.
- `run_h5toROOT.py` wrote with ROOT, from the venv with no conda env active. The ntuples are
  identical to those uproot writes.
- `hadron_distributions.C+` ran on them in the activated env.
- Re-running on an isolated venv with `--system-site-packages` switched it over.

### B. One conda env with everything: `environment.yml`

```bash
cd js-contrib
conda env create -f utils/analysis_env/environment.yml    # env "js_analysis"
conda activate js_analysis
python utils/analysis_env/check_env.py /path/to/out
```

Everything available on conda-forge comes from conda-forge. pip adds only the scikit-hep
`fastjet` and PyJetscape (editable). The EoS table isn't part of it; add it with

```bash
mkdir -p "$CONDA_PREFIX/share/music_eos/hotQCD"
curl -fL -o "$CONDA_PREFIX/share/music_eos/hotQCD/hrg_hotqcd_eos_binary.dat" \
  https://api.bitbucket.org/2.0/repositories/wayne_state_nuclear_theory/hotqcd/src/main/hrg_hotqcd_eos_binary.dat
conda env config vars set MUSIC_EOS_TABLE="$CONDA_PREFIX/share/music_eos/hotQCD/hrg_hotqcd_eos_binary.dat"
conda activate js_analysis                               # again, to pick up the variable
```

**`environment.yml` has not been tested yet**; A is the tested way. Its package list
duplicates `requirements*.txt`, so a change to one needs the same change in the other.

## Remote files: Google Cloud Storage and Pelican/OSDF

For productions kept in a Google Cloud Storage bucket or in a Pelican federation such as the
OSDF, the venv can get the Python interfaces of either or both:

```bash
./utils/analysis_env/setup_analysis_env.sh --with-gcs --with-pelican   # or one of them
```

Both are [fsspec](https://filesystem-spec.readthedocs.io) file systems, so the URLs work
wherever fsspec does. `check_env.py --remote-test` reads one public object from each (the
setup script does that when either option is given). On an existing venv, re-run the script
with the option. In the conda env of `environment.yml`, run
`pip install -r utils/analysis_env/requirements_gcs.txt` (or `_pelican.txt`) instead.

**Reading files in place.** uproot takes the URL directly; h5py takes the file object
fsspec opens (with `hdf5plugin` imported for Blosc). Only the chunks read travel over the
network:

```python
import fsspec, h5py, hdf5plugin, uproot

# ROOT files of run_h5toROOT.py
events = uproot.open("gs://BUCKET/campaign/AuAu_0_10_jet_c1_0001_hadrons.root")["events"]
bulk = uproot.open("osdf:///NAMESPACE/campaign/AuAu_0_10_jet_c1_0001_hadrons.root")["bulk_jet"]

# HDF5 files: pair, particlize and hadron files
with fsspec.open("gs://BUCKET/campaign/AuAu_0_10_jet_c1_0001.h5", "rb") as fo, \
        h5py.File(fo, "r") as f:
    e_last = f["arr"][0, 0, :, :, :, -1]
```

**Whole campaigns, for the repo's tools.** `HadronFileReader`, `run_h5toROOT.py`, the
`analysis/` scripts and the Visualization scripts take local paths. Copy the files first,
optionally only what the analysis reads:

```python
import fsspec

fs = fsspec.filesystem("gs")                  # or "osdf"
fs.get("BUCKET/campaign/*_hadrons_*.h5", "out/")          # hadron files
fs.get("BUCKET/campaign/*_particlize.h5", "out/")         # HadronFileReader needs these too
```

The command-line tools do the same: `gcloud storage cp -r gs://BUCKET/campaign out/`, and
`pelican object get osdf:///NAMESPACE/campaign/FILE out/`.
[`utils/gcs_transfer/js_gcs.py`](../gcs_transfer/README.md) uploads and downloads
production directories, single files or patterns with a service-account key, by kind
(`--what pair|h5|root|all`), and skips the files already there. It runs in an environment
of its own.

**Credentials.**
- **GCS:** gcsfs and `google-cloud-storage` use Google's default credentials: those of
  `gcloud auth application-default login`, or a service-account key in
  `GOOGLE_APPLICATION_CREDENTIALS`. A public bucket needs none:
  `fsspec.filesystem("gs", token="anon")`, `storage.Client.create_anonymous_client()`.
- **Pelican/OSDF:** public namespaces need no token. For a protected one, pelicanfs takes a
  bearer token from `BEARER_TOKEN` or the file in `BEARER_TOKEN_FILE`, from the WLCG default
  token location, from `TOKEN`, or from HTCondor's credentials, in that order. It can also be
  passed explicitly: `fsspec.filesystem("osdf", headers={"Authorization": "Bearer " + tok})`.
  With the `pelican` binary on `PATH`, pelicanfs can also get one through OAuth.

Tested 2026-09-30 on the GB10 with gcsfs 2026.8.1, google-cloud-storage 3.15.1 and
pelicanfs 1.4.1:
- a public GCS bucket listed anonymously with gcsfs and with `google-cloud-storage`; without
  credentials gcsfs falls back to anonymous access, with a warning;
- a public OSDF object read, and a glob over its directory, with pelicanfs;
- uproot's reads over `gs://` and `osdf://` (on public files that aren't ROOT files, so it
  stopped at their first bytes), and h5py with Blosc through an fsspec file object.

Not tested: our own productions in a bucket or a namespace, a whole remote ROOT file, and
credentials.

## MUSIC's EoS table

The pair files store the energy density, not the temperature. `jet_wake.ipynb` (freeze-out
contours, Mach angle), `analysis/wake_observables.py`, `wake_hadrons.py` and the
Visualization scripts (freeze-out isosurface) convert e → T with MUSIC's hotQCD table. The
files say `eos_kind = "hotqcd (MUSIC EOS 9)"`. Normally these tools find the table in the
producing X-SCAPE build, but in this venv there is none. So the setup script downloads
`hrg_hotqcd_eos_binary.dat` (3.2 MB) into `VENV/share/music_eos/hotQCD`, from the same
place as MUSIC's `EOS/download_hotQCD.sh`, and checks its md5. It then sets
`MUSIC_EOS_TABLE` to that file, in `VENV/bin/activate` and in the `--kernel` kernel spec.
Without the table, the visualization uses a conformal EoS and prints a warning,
`jet_wake.ipynb` skips the contours, and `wake_*.py` stop and ask for `--eos`.

## Jupyter

```bash
source ~/.venvs/js_analysis/bin/activate
cd js-contrib/contribs/PyJetscape/example/prod_AuAu_0_10_jet
jupyter notebook jet_wake.ipynb
```

The notebooks name the kernel they were saved with (`js_fno`, `fno_env_mlx`, …). If Jupyter
asks, choose `Python 3 (ipykernel)`: started from the activated venv, that is the venv. If you
use a Jupyter elsewhere (VS Code, JupyterLab of another env), register the venv with
`--kernel js_analysis` and choose that. The notebooks read `out/` by default. Point them
elsewhere with `WAKE_H5=/path/to/<stem>.h5` (`jet_wake.ipynb`) or `HADRON_DIR=/path/to/out`
(`analysis/*.ipynb`).

## Headless machines

The visualization renders off-screen (`--movie`, `--vtk-dir`). `check_env.py` tests that it
works. With the current wheels (pyvista 0.49, vtk 9.7), rendering worked without a display on
the GB10. If it fails on another machine, run under `xvfb-run` (Linux, `apt install xvfb`).

## Tested

Tested 2026-09-29 on the GB10 (Linux aarch64), with the system Python 3.12 and none of
conda, X-SCAPE or ROOT on the path. Both venvs were fresh; the first has no FastHydro, the
second was made with `--uv --no-viz --with-fasthydro --kernel`. Everything was then run
again on the updated venvs.
- `setup_analysis_env.sh --check` on the seed-1 `out/` made a 1.6 GB venv. The `--uv` install took 10 s from
  a warm uv cache.
- `run_h5toROOT.py` converted the seed-1 production.
- `jet_wake.ipynb`, `analysis/hadron_distributions.ipynb`, `jet_fastjet.ipynb` and
  `jet_fastjet_awkward.ipynb` ran through with `jupyter nbconvert --execute`.
- `analysis/jet_edep_balance_check.py` and `wake_observables.py` ran.
- `hydro_jet_particles_pyvista.py --movie` and `wake_pyvista.py --movie` wrote their movies
  off-screen, without a display. T came from the downloaded table.
- `pytest contribs/Visualization/tests` passed (45).

`analysis/wake_hadrons.py` could not be tested on the seed-1 files. Their hadron files
predate `initiators/`, and it stops the same way in `js_fno`.
