# Setting up the tutorials

The tutorials are Jupyter notebooks that you run on your own computer, in a setup you may
already have: the production container, the xscape wheel on a Mac, or the analysis
environment. They work with the releases already out. Everything a tutorial needs, apart
from the setup itself, comes with the tutorials or is fetched by them.

| setup | runs X-SCAPE | the hydro runs on | the analysis tutorials | install |
|---|---|---|---|---|
| production container | yes | an NVIDIA GPU (CUDA), when passed in | yes | [Production container](../install/container.md), images from 2026-10-28 on |
| Mac, xscape wheel | yes | the Apple Silicon GPU (Metal), by default | yes | [Mac (xscape wheel)](../install/mac.md), 0.2.0 or later |
| analysis environment | no | – | yes | [Analysis environment](../install/analysis-env.md) |
| Google Colab | no | – | yes | nothing: open a notebook from GitHub |

ROOT's C++ (tutorial 10) needs a ROOT installation next to the setup, for example from conda
or, on a Mac, from Homebrew ([ROOT and C++](../analysis/root.md#1-set-up)). Neither the
container nor the wheel brings ROOT.

## 1. Get the tutorials

In a checkout of js-contrib, they are in `tutorials/pyjetscape/`. Without one, download them
with curl:

=== "A release"

    ```bash
    curl -L https://github.com/jhputschke/js-contrib/releases/download/tutorials-v2/pyjetscape-tutorials.tar.gz | tar xz
    ```

    A release (`tutorials-vN`) is fixed, and goes with a fixed version of the data:
    `tutorials-v2` reads the data in `tutorials/v2/` on the OSDF.

=== "The main branch"

    ```bash
    mkdir pyjetscape-tutorials && curl -L https://codeload.github.com/jhputschke/js-contrib/tar.gz/refs/heads/main \
      | tar xz -C pyjetscape-tutorials --strip-components=3 js-contrib-main/tutorials/pyjetscape
    ```

    The latest version. This downloads the whole repository (about 20 MB) and unpacks only the
    tutorials.

Both commands create a folder `pyjetscape-tutorials/`. Later releases of the container and
the wheel will also carry the tutorials.

## 2. Start Jupyter

=== "Production container"

    Download the tutorials on the host, into the directory you mount (the image has `wget`
    but not `curl`). Then start JupyterLab in the image with `xscape-jupyter`:

    ```bash
    mkdir -p work && (cd work && curl -L https://github.com/jhputschke/js-contrib/releases/download/tutorials-v2/pyjetscape-tutorials.tar.gz | tar xz)
    docker run --rm -it --gpus all --user "$(id -u):$(id -g)" -p 8888:8888 \
      -v "$PWD/work:/work" jhputschke/xscape-prod:cu126 xscape-jupyter /work/pyjetscape-tutorials
    ```

    Open the `http://127.0.0.1:8888/lab?token=...` line it prints.

    - **`--gpus all`** passes the NVIDIA GPU in. MUSIC runs on it with CUDA; without it, MUSIC
      runs on its CPU path, about 10 times slower. The analysis tutorials need no GPU, so
      leave it out on a computer without one.
    - **Apptainer** on a cluster:
      `apptainer exec --nv --bind "$PWD/work:/work" xscape_prod.sif xscape-jupyter /work/pyjetscape-tutorials`,
      with an ssh tunnel to the node ([Jupyter in the container](../install/container.md#8-jupyter)).
    - Images older than 2026-10-28 have no Jupyter: pull a newer one.

=== "Mac (xscape wheel)"

    ```bash
    source ~/.venvs/xscape-0.2.0/bin/activate      # the line install.sh printed
    jupyter lab pyjetscape-tutorials
    ```

    JupyterLab opens in your browser. **The wheel runs the hydro on the Apple Silicon GPU, by
    default**, through Metal; there is nothing to switch on. When an event starts, MUSIC
    prints `[MUSIC-GPU] Metal device: <your chip>`. Run Jupyter natively in Terminal: inside a
    virtual machine or a container there is no Metal GPU, and MUSIC falls back to the CPU.

=== "Analysis environment"

    ```bash
    source ~/.venvs/js_analysis/bin/activate
    jupyter lab pyjetscape-tutorials
    ```

    For VS Code or a JupyterLab of another environment, register the environment as a
    kernel (`setup_analysis_env.sh --kernel js_analysis`) and choose it in the notebook.

=== "Google Colab"

    Open a tutorial from its page here or on GitHub with the **Open in Colab** badge at its
    top. The first cell fetches the tutorial helpers and installs the jetscape readers with
    the analysis packages, which takes about a minute. Colab runs the analysis tutorials;
    it cannot run X-SCAPE. Its sessions are temporary, so the installation and any
    downloaded data are gone the next time.

Then open **`01_check_setup.ipynb`**. It names your setup, says where the hydro would run,
checks that the data can be reached, reads a first file and lists the tutorials that run
here.

## 3. The data

The tutorials read example files from the OSDF, in the public namespace `/fno4hic`. They
read them over plain HTTPS: the OSDF director sends each request to a cache near you. This
needs no account, no `pelicanfs` and no `curl`. When a notebook opens a file, it takes the
local copy if there is one, and otherwise streams only the parts it reads.

To download a set once, for working offline or for the large files (the hydro pairs are
0.2–0.9 GB each), run `get_data.py` in the tutorial folder:

```bash
python get_data.py --list                          # the sets, their sizes and contents
python get_data.py oo_test auau_jet_event          # -> data/oo_test/, data/auau_jet_event/
python get_data.py auau_hadrons_root --max-files 3 # the first three files of a large set
```

- `get_data.py` uses only the Python standard library, so it runs in every setup, inside
  the container too.
- It takes the file names and sizes from the manifest that each OSDF folder holds, and
  checks every file's size. It also checks the CRC32C checksum when `google-crc32c` or
  `crc32c` is installed.
- Interrupted downloads resume where they stopped.
- To keep the data elsewhere, use `--to DIR` and set `JS_TUTORIAL_DATA=DIR` for the
  notebooks.

The tutorials' data sets (a complete Au+Au pair set, a hadron-level Au+Au campaign in three
p̂<sub>T</sub> windows, its p+p reference, a small O+O campaign) are in
`osdf:///fno4hic/tutorials/v2/`; `data_recipes/make_v2.sh` in the tutorial folder says how
each was made. They will never be
overwritten, so a tutorial release and its data stay together.

## Troubleshooting

| problem | cause and fix |
|---|---|
| `OSError: Can't synchronously read data (can't open directory .../hdf5/lib/plugin)` | the files are compressed with Blosc and h5py has no filter for it: `import hdf5plugin` before reading. The tutorials' first cell does this (`te.setup()`). |
| the OSDF is not reachable | a firewall or proxy blocks `osdf-director.osg-htc.org`. Download the data elsewhere with `get_data.py` and copy the `data/` folder. |
| `ModuleNotFoundError: _common` | the notebook was started outside its folder. Start Jupyter in the tutorial folder, or open the notebook from there. |
| the container's Jupyter page does not load | Docker: `-p 8888:8888` is missing. Apptainer: tunnel to the compute node, not the login node. |
| MUSIC prints `Metal init failed, falling back to CPU` | the Mac wheel runs in a virtual machine or a container: run it natively in Terminal. |
