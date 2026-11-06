# PyJetscape tutorials

Hands-on tutorials for X-SCAPE with js-contrib:
- running events from Python and the production scripts
- the files a production writes: the hydro pair, the hadron pair and the hadrons
- analysis in Python (hist, FastJet) and in ROOT
- writing your own modules

Each tutorial explains the physics behind what it shows, and ends with a check of its
results and suggestions for what to try next.

> [!WARNING]
> Work in progress: the tutorials are examples of how to run the simulation and use its
> outputs, not vetted physics results. Their data sets are small, so the statistics are
> low, and their analyses have not been checked to the standard of a publication.

Start with **`01_check_setup.ipynb`**. It runs anywhere, tells you which of the other
tutorials run in your setup, and reads a first file. The full list is in
[`tutorials.json`](tutorials.json).

## 1. Get the tutorials

```bash
curl -L https://github.com/jhputschke/js-contrib/releases/download/tutorials-v1/pyjetscape-tutorials.tar.gz | tar xz
cd pyjetscape-tutorials
```

Other ways to get them (the `main` branch, a checkout) are in
[`../README.md`](../README.md).

## 2. Start Jupyter

Pick the setup you have. Every setup below runs the analysis tutorials. Only the production
container and the Mac wheel also run X-SCAPE itself.

**Production container** (`jhputschke/xscape-prod`, images from 2026-10-28 on). Download the
tutorials on the host, into the directory you mount; the image has `wget` but not `curl`.

```bash
mkdir -p work && cd work
curl -L https://github.com/jhputschke/js-contrib/releases/download/tutorials-v1/pyjetscape-tutorials.tar.gz | tar xz
cd ..
docker run --rm -it --gpus all --user "$(id -u):$(id -g)" -p 8888:8888 \
  -v "$PWD/work:/work" jhputschke/xscape-prod:cu126 xscape-jupyter /work/pyjetscape-tutorials
```

Then open the `http://127.0.0.1:8888/lab?token=...` line it prints. Leave out `--gpus all`
on a computer without an NVIDIA GPU: the analysis tutorials do not need one. With
Apptainer on a cluster:

```bash
apptainer exec --nv --bind "$PWD/work:/work" xscape_prod.sif xscape-jupyter /work/pyjetscape-tutorials
```

On a cluster you also need an ssh tunnel to reach Jupyter; see the
[container page](https://jhputschke.github.io/js-contrib/install/container/#8-jupyter).

**Mac, xscape wheel** (0.2.0 or later):

```bash
source ~/.venvs/xscape-0.2.0/bin/activate      # the line install.sh printed
jupyter lab pyjetscape-tutorials
```

The wheel runs the hydro (MUSIC) on the Apple Silicon GPU through Metal, by default. Run it
natively in Terminal: inside a virtual machine or a container there is no Metal GPU, and
MUSIC falls back to the CPU.

**Analysis environment** (`setup_analysis_env.sh`, or the pyjetscape wheel):

```bash
source ~/.venvs/js_analysis/bin/activate
jupyter lab pyjetscape-tutorials
```

**Google Colab**: open a notebook from GitHub with the *Open in Colab* badge at its top.
The first cell fetches the helpers and installs the readers. This covers the analysis
tutorials only.

## 3. The data

The tutorials read example files from the OSDF over HTTPS. This needs no account, and
nothing is installed for it. The files are streamed as they are read. To download them
instead, for working offline or for the larger sets:

```bash
python get_data.py --list                        # the sets and their sizes
python get_data.py oo_test auau_jet_event        # -> data/<set>/
python get_data.py auau_jet_event --to ~/xscape_data && export JS_TUTORIAL_DATA=~/xscape_data
```

`get_data.py` needs only Python, so it runs in every setup, including inside the container.
It checks the size of every file, and its CRC32C checksum too when `google-crc32c` or `crc32c`
is installed. Interrupted downloads resume where they stopped. The tutorials take a set
from the data folder (`$JS_TUTORIAL_DATA`, else `data/` here) when it is there, and stream
it otherwise.

## What is in this folder

| file | |
|---|---|
| `NN_*.ipynb` | the tutorials, in order |
| `tutorials.json` | the index: titles, what each tutorial needs, its data |
| `datasets.json` | the data sets: their OSDF folders, sizes and contents |
| `get_data.py` | downloads data sets (Python standard library only) |
| `_common/tutorial_env.py` | the helpers each notebook loads first: detects the setup, registers the HDF5 compression filters, finds the data |
| `expected/` | the reference results the "check your result" cells compare against |
| `VERSION.txt` | the version of the tutorials, printed by every notebook (not `VERSION`: on macOS ROOT's compiler would take it for the C++ header `<version>`) |
