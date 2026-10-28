# Using the X-SCAPE container

One image, three uses:

| | use | GPU | you bring |
|---|---|---|---|
| [1](#1-the-production-scripts) | **the production scripts** (`run_prod_jet.py`, `run_prod_oo_isr.py`, `run_jobs.sh`, `hadronize.py`, ...) | yes (stage 2 hadronization: no) | an output directory |
| [2](#2-your-own-scripts-and-notebooks) | **your own scripts and notebooks**, production-like (the O+O workflow, the examples) | yes | your scripts/notebooks/XMLs in a directory |
| [3](#3-analysis-only) | **analysis only**: hadron and pair files, FastJet, plots, notebooks | no | the files (or OSDF paths) |

The image holds X-SCAPE with MUSIC (GPU), iSS, 3dMCGlauber, Matter, LBT, iColored, Pythia,
compiled, with PyJetscape and its Pythonic layer
([`PYTHONIC.md`](../contribs/PyJetscape/PYTHONIC.md)), and the analysis packages (awkward,
vector, fastjet, particle, hist, mplhep, matplotlib, scipy, pandas, uproot, Jupyter). How it is
built: [`BuildContainerProd.md`](BuildContainerProd.md). This page: how to use it.

Contents: [Get started](#get-started) · [The rules of the container](#the-rules-of-the-container)
· [1 Productions](#1-the-production-scripts) · [2 Own scripts and notebooks](#2-your-own-scripts-and-notebooks)
· [3 Analysis only](#3-analysis-only) · [Jupyter](#jupyter) · [HPC with Apptainer](#hpc-apptainer)
· [Troubleshooting](#troubleshooting) · [Reference](#reference)

---

## Get started

**Prerequisites.** Docker with the NVIDIA Container Toolkit (workstation, cloud VM), or
Apptainer/Singularity (HPC); an NVIDIA GPU and driver for uses 1 and 2. Use 3 runs on any
machine, CPU only.

**Pick the image.** `jhputschke/xscape-prod:<tag>` on Docker Hub; each tag holds amd64 and
arm64 (cu124: amd64 only), `docker pull` / `apptainer pull` take the right one.

| tag | CUDA | GPUs | host driver |
|---|---|---|---|
| `cu126` | 12.6 | V100, RTX 20xx-40xx, A100, A40, L4, H100, GH200 | supports CUDA ≥ 12.6 |
| `cu130` | 13.2 | GH200, B200/GB200, RTX 50xx / RTX PRO Blackwell, GB10 | supports CUDA ≥ 13.2 |
| `cu124` | 12.4 | as cu126 | older drivers (R550), amd64 only |

`nvidia-smi` on the host shows the highest CUDA version its driver supports (top right).
Each tag comes as:

- `cu126` — moving: the latest build. For a campaign take a fixed one,
  `cu126-<YYYYMMDD>-<xscape commit>` (e.g. `cu126-20261013-9e84450`), so every job runs the
  same code. Don't mix images of different builds in one campaign: physics fixes change the
  results (the image's `/opt/X-SCAPE/BUILD_INFO.txt` lists its commits).
- `cu126-slim` — production only, without the analysis packages and Jupyter (0.6 GB
  smaller: 4.75 GB instead of 5.36 GB for cu130 arm64); fine for use 1.
- `cu126-viz` — with pyvista/vtk for `contribs/Visualization` (off-screen rendering).
- `...-gcs` — with Google Cloud Storage for Python.

The analysis packages, the helpers (`xscape-examples`, `xscape-jupyter`) and `PYTHONPATH` are
in the images built since 2026-10-28 (js-contrib `container_user`); older images only run
the production scripts.

**Pull it and check it:**

```bash
docker pull jhputschke/xscape-prod:cu126
docker run --rm --gpus all jhputschke/xscape-prod:cu126 nvidia-smi        # the GPU is seen
docker run --rm jhputschke/xscape-prod:cu126 \
  python /opt/X-SCAPE/external_packages/js-contrib/utils/container/image_check.py --analysis

# Apptainer
apptainer pull xscape_prod.sif docker://jhputschke/xscape-prod:cu126
apptainer exec --nv xscape_prod.sif nvidia-smi
```

`image_check.py` ends with `image_check: all ok` (it also runs when the image is built).

**Your first event** (one O+O event through the whole chain, under a minute on a GPU, the
hadron files in `./work/out_oo/`):

```bash
mkdir -p work
docker run --rm --gpus all --user "$(id -u):$(id -g)" -v "$PWD/work:/work" -w /work \
  jhputschke/xscape-prod:cu126 bash -c \
  'xscape-examples /work/examples && cd /work/examples/pythonic &&
   python oo_workflow.py --events 1 --outdir /work/out_oo'
```

---

## The rules of the container

Everything below follows from four rules.

1. **The image is read-only; your files live in `/work`.** The X-SCAPE tree is
   `/opt/X-SCAPE` (Apptainer: read-only; Docker: owned by root). Mount a directory of yours
   at `/work` (`-v "$PWD/work:/work"`, Apptainer `--bind "$PWD/work:/work"`) and write every
   output there. A file written anywhere else is gone when the container ends.
2. **Run as yourself.** Docker: `--user "$(id -u):$(id -g)"`, else the outputs belong to root.
   Apptainer runs as you anyway.
3. **GPU access is explicit.** Docker `--gpus all` (or `--gpus device=1`), Apptainer `--nv`.
   Without it MUSIC falls back to its CPU path: ~10× slower, and its numbers are not the GPU's.
4. **Everything finds the tree by itself.** `import jetscape` works everywhere (`PYTHONPATH`);
   scripts and notebooks find the build through `$XSCAPE=/opt/X-SCAPE`
   (`jetscape.paths`); each job runs in a private working directory under its output
   directory (`jetscape.workdir.JobDir`, the production scripts' `OUTDIR/work/`), so jobs never
   share files and nothing writes into the image.

Where things are:

| | path in the container |
|---|---|
| X-SCAPE tree, build | `/opt/X-SCAPE`, `/opt/X-SCAPE/build_gpu` |
| js-contrib | `/opt/X-SCAPE/external_packages/js-contrib` |
| the examples and production scripts | `.../js-contrib/contribs/PyJetscape/example` (`$EX` below) |
| Docker's start directory | `$EX/prod_AuAu_0_10_jet` (Apptainer: the directory you start it from) |
| your files | `/work` (you mount it) |

```bash
EX=/opt/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape/example
```

---

## 1 The production scripts

The productions run from the image's scripts, with their outputs in `/work`. Their full
documentation: [`prod_AuAu_0_10_jet/README.md`](../contribs/PyJetscape/example/prod_AuAu_0_10_jet/README.md)
(Au+Au with jets), [`prod_OO_isr/README.md`](../contribs/PyJetscape/example/prod_OO_isr/README.md)
(O+O with ISR), [`README_2stage.md`](../docs/README_2stage.md) (the two stages);
for clusters, SLURM arrays and several GPUs: [`BuildContainerProd.md`](BuildContainerProd.md#running-the-image).

```bash
IMG=jhputschke/xscape-prod:cu126
mkdir -p prod

# Au+Au 0-10 % with a jet: 2 events (Docker starts in prod_AuAu_0_10_jet)
docker run --rm --gpus all --user "$(id -u):$(id -g)" -v "$PWD/prod:/work" $IMG \
  python run_prod_jet.py --events 2 --seed 1 --write-particlize both --outdir /work/out

# a campaign: 20 jobs x 25 events, unique seeds, 4 jobs at a time on the GPU
docker run --rm --gpus all --user "$(id -u):$(id -g)" -v "$PWD/prod:/work" $IMG \
  ./run_jobs.sh -j 4 --campaign test 20 25 0 /work/out

# O+O with initial-state radiation (another directory: -w)
docker run --rm --gpus all --user "$(id -u):$(id -g)" -v "$PWD/prod:/work" \
  -w /opt/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape/example/prod_OO_isr $IMG \
  python run_prod_oo_isr.py --events 2 --seed 5 --outdir /work/out_oo

# stage 2 (hadronization) needs no GPU
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/prod:/work" $IMG \
  python run_hadronize.py /work/out -j 4 --oversample 500 --n-frag 50
```

- **Put OUTDIR one level inside `/work`** (`/work/out`, not `/work`): every job records its
  seed in `OUTDIR/../seeds_used.tsv`.
- **Your own physics settings:** mount a directory with your user XML and pass it,
  `-v "$PWD/xml:/xml:ro"` and `--user-xml /xml/PbPb_0_10.xml`
  ([details](BuildContainerProd.md#a-different-collision-system-your-own-user-xml)).
- **Long campaigns:** `docker run -d ...` (detached; `docker logs -f <id>` to follow), or the
  launchers [`launch_2gpu.sh`](launch_2gpu.sh) / [`launch_2gpu_oo.sh`](launch_2gpu_oo.sh)
  for one campaign per GPU ([`README_launch.md`](README_launch.md)).
- **Getting the outputs home** (OSDF, Google Cloud Storage):
  [`BuildContainerProd.md`](BuildContainerProd.md#getting-the-outputs-home).

---

## 2 Your own scripts and notebooks

### 2.1 Start from the examples

The examples can't run in the read-only image tree (they write next to themselves). Copy them
into `/work` once and work on the copy:

```bash
mkdir -p work
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/work:/work" jhputschke/xscape-prod:cu126 \
  xscape-examples /work/examples
```

`work/examples/` now holds every example of `contribs/PyJetscape/example`: the production
scripts, the Pythonic examples (`pythonic/`), the toy modules (`python_modules/`), the analysis
notebooks and their XMLs, editable on the host with your own editor. `xscape-examples`
never overwrites: for a fresh copy give a new directory.

Run one, here the O+O workflow as a script ([`oo_workflow.py`](../contribs/PyJetscape/example/pythonic/oo_workflow.py):
3dMCGlauber → Pythia + ISR → MUSIC → Matter + LBT → iColored + iSS → two hadron files):

```bash
docker run --rm --gpus all --user "$(id -u):$(id -g)" -v "$PWD/work:/work" \
  -w /work/examples/pythonic jhputschke/xscape-prod:cu126 \
  python oo_workflow.py --events 3 --seed 7 --outdir /work/out_oo
```

The notebook version, [`oo_workflow.ipynb`](../contribs/PyJetscape/example/pythonic/oo_workflow.ipynb)
(the same run, then the hadron files read back, awkward arrays, spectra with errors, FastJet),
runs in Jupyter: [Jupyter](#jupyter). The copied production scripts work too: they find the
build through `$XSCAPE`, exactly as the image's own.

### 2.2 A script of your own

Put your script and its user XML in `work/` (e.g. start from a copy of
`examples/pythonic/oo_workflow.xml` and `oo_workflow.py`). The skeleton every such script has:

```python
# work/my_job.py
import jetscape as js                  # on PYTHONPATH in the image
from jetscape.hadrons_h5 import HadronH5Writer

with js.JobDir("/work/out_mine", "/work/my_job.xml") as job:   # private working directory
    with js.per_event(job.main_xml, job.user_xml) as driver, \
         HadronH5Writer("/work/out_mine/my_job_hadrons.h5", tag="jet_frag", n_samples=1) as out:
        for i in driver.events():                              # one iteration per event
            jets = driver.hadronization_manager.hadron_array()
            out.append_unit(jets.to_dict(), unit=i, event=i)
            print(i, len(jets), "jet hadrons")
```

```bash
docker run --rm --gpus all --user "$(id -u):$(id -g)" -v "$PWD/work:/work" -w /work \
  jhputschke/xscape-prod:cu126 python my_job.py
```

- `js.JobDir(out_dir, user_xml)` copies your XML to `out_dir/`, makes its `../` paths
  (e.g. iSS's tables) absolute for the image's build, gives the job a private working directory
  `out_dir/work/<name>` (MUSIC and 3dMCGlauber write side files there) and removes it after a
  successful run. Use absolute output paths: inside the `with` the current directory is that
  working directory.
- `driver.events()`, `final_parton_array()`, `hadron_array()`, the hadron files, writing a
  module of your own in Python: [`PYTHONIC.md`](../contribs/PyJetscape/PYTHONIC.md).
- The physics is in the user XML; X-SCAPE's defaults are in
  `/opt/X-SCAPE/config/jetscape_main.xml`.

### 2.3 Python packages that are not in the image

Install them into `/work`, once, and put that directory on the path:

```bash
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/work:/work" jhputschke/xscape-prod:cu126 \
  pip install --target /work/pylib seaborn
docker run ... -e PYTHONPATH=/work/pylib:/opt/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape/python ...
```

Don't upgrade numpy or h5py this way: `pyjetscape_core` is compiled against the image's numpy.
For a group, a small Dockerfile of your own is cleaner:
`FROM jhputschke/xscape-prod:cu126-<date>-<commit>` + `RUN pip install --no-cache-dir seaborn`.

### 2.4 Changing js-contrib's Python code

The image's `jetscape` package is used unless yours comes first on the path. Mount your
js-contrib checkout and put its `python/` in front:

```bash
docker run ... -v "$HOME/js-contrib:/src/js-contrib:ro" \
  -e PYTHONPATH=/src/js-contrib/contribs/PyJetscape/python ...
```

This works for pure-Python changes only, and only if your checkout is close to the image's
js-contrib commit (`BUILD_INFO.txt`): the compiled `pyjetscape_core` comes from the image
(copy `pyjetscape_core*.so` from the image's `python/jetscape/` into your checkout's).
Changes to C++ (X-SCAPE, the bindings) need a build of your own: the development image
([`BuildContainerDev.md`](BuildContainerDev.md)) or a native build.

---

## 3 Analysis only

No GPU and no simulation: the image is the analysis environment. Leave out `--gpus` / `--nv`.

```bash
# a script on your files
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/work:/work" -w /work \
  jhputschke/xscape-prod:cu126 python my_analysis.py /work/data

# the production's file checks on a directory of outputs
docker run --rm -v "$PWD/prod:/work:ro" jhputschke/xscape-prod:cu126 \
  python /opt/X-SCAPE/external_packages/js-contrib/utils/analysis_env/check_env.py /work/out --no-render
```

Reading the files: `js.HadronArray.from_h5("..._hadrons_jet_frag.h5")`, `.to_awkward()`,
`js.select(...)`, FastJet on the jagged arrays ([`PYTHONIC.md`](../contribs/PyJetscape/PYTHONIC.md));
the readers for whole campaigns, ROOT files with uproot, the analysis notebooks:
[`README_analysis.md`](../docs/README_analysis.md). Notebooks to start from (in
`xscape-examples`' copy):

| notebook | reads |
|---|---|
| `pythonic/oo_analysis.ipynb` | the O+O hadron files **streamed from the OSDF** (public, nothing to download) |
| `pythonic/oo_workflow.ipynb` (sections 4-6) | the hadron files of your own run |
| `analysis/tutorials/hadron_distributions.ipynb`, `jet_fastjet*.ipynb` | a production directory |
| `prod_AuAu_0_10_jet/jet_wake.ipynb` | a pair file and its hadron files |
| `prod_OO_isr/oo_isr_overview.ipynb` | an O+O production |

Remote files: pelicanfs reads `osdf://` paths in place (public namespaces need no token; a
protected one: `-e BEARER_TOKEN_FILE=/work/token`); the `-gcs` images read `gs://`.
Without a container, the same packages come as a venv: [`analysis_env/`](analysis_env/README.md).

---

## Jupyter

`xscape-jupyter [DIR]` starts JupyterLab on port 8888 with `DIR` (default `/work`) as its
root, and prints the URL with its access token.

**On your workstation (Docker):**

```bash
docker run --rm -it --gpus all --user "$(id -u):$(id -g)" -p 8888:8888 \
  -v "$PWD/work:/work" jhputschke/xscape-prod:cu126 xscape-jupyter
```

Open the `http://127.0.0.1:8888/lab?token=...` line it prints. The notebooks in
`work/examples/` (from `xscape-examples`) open with the image's Python ("Python 3
(ipykernel)"); simulations in a notebook use the GPU as scripts do. Stop with Ctrl-C (twice).

- **Another port:** `-p 8890:8888`, then `127.0.0.1:8890`.
- **On a remote machine** (a GPU server): start it there as above, then from your laptop
  `ssh -N -L 8888:localhost:8888 user@server` and open the URL locally.
- **Jupyter's settings and history** live in `$HOME`, which is a temporary directory with
  `--user`; to keep them, `-e HOME=/work/.home`.
- **Analysis only:** the same without `--gpus all`.

**On a cluster (Apptainer):** below.

---

## HPC: Apptainer

Apptainer runs the same image as you, with your home directory mounted and the host's network.
Three differences to Docker: `--nv` for the GPU, `--bind` for `/work`, and it starts in the
current directory (not the image's start directory), so give `--pwd` or full paths.

```bash
apptainer pull xscape_prod.sif docker://jhputschke/xscape-prod:cu126-<date>-<commit>
W=$SCRATCH/xscape; mkdir -p $W
EX=/opt/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape/example

# 1 productions (more in BuildContainerProd.md: SLURM arrays, several GPUs)
apptainer exec --nv --pwd $EX/prod_AuAu_0_10_jet --bind $W:/work xscape_prod.sif \
  python run_prod_jet.py --events 2 --seed 1 --outdir /work/out

# 2 the examples, a script of your own
apptainer exec --bind $W:/work xscape_prod.sif xscape-examples /work/examples
apptainer exec --nv --pwd /work/examples/pythonic --bind $W:/work xscape_prod.sif \
  python oo_workflow.py --events 3 --outdir /work/out_oo

# 3 analysis, no GPU
apptainer exec --pwd /work --bind $W:/work xscape_prod.sif python my_analysis.py
```

In a SLURM batch script these lines run as they are (after `#SBATCH --gpus=1` etc.).

**Jupyter on a compute node:**

```bash
salloc --gpus=1 --time=2:00:00                  # your site's options; you land on a node
apptainer exec --nv --bind $SCRATCH/xscape:/work xscape_prod.sif xscape-jupyter
#   -> prints http://127.0.0.1:8888/lab?token=... and the node's name is `hostname`
```

In Apptainer, `xscape-jupyter` listens on 127.0.0.1 only (the node is shared). From your
laptop, tunnel through the login node to the compute node:

```bash
ssh -N -L 8888:localhost:8888 -J user@login.cluster user@<node>   # if nodes accept ssh
ssh -N -L 8888:<node>:8888 user@login.cluster                     # else: JUPYTER_IP=0.0.0.0
```

For the second form start it with `apptainer exec --env JUPYTER_IP=0.0.0.0 ...` (the token
still protects it). Where the centre runs Open OnDemand or JupyterHub, ask how to register a
container kernel; the kernel's command is the image's `python -m ipykernel_launcher`.

- **Mixed clusters:** a `.sif` holds one CPU architecture; on x86 login nodes with arm64 GPU
  nodes (GH200), `apptainer pull --arch arm64`.
- **Home directory:** Apptainer mounts it, but write outputs to `/work` (scratch), not `~`.

---

## Troubleshooting

| symptom | cause, fix |
|---|---|
| `Permission denied` / `Read-only file system` when writing | writing into the image or into `/work` itself where OUTDIR's parent must be writable: write to `/work/<something>`; Docker: `--user` and a directory you own |
| outputs owned by root | Docker without `--user "$(id -u):$(id -g)"`; `sudo chown -R $USER work/` once |
| `No CUDA device found ... falling back to CPU` | no `--gpus all` / `--nv`, or the driver is older than the image's CUDA: `nvidia-smi`, take `cu124` for R550 drivers |
| `docker: could not select device driver "" with capabilities: [[gpu]]` | the NVIDIA Container Toolkit is not installed on the host |
| `python: can't open file 'run_prod_jet.py'` | Apptainer: add `--pwd $EX/prod_AuAu_0_10_jet` or use the full path |
| `xscape-examples: ... exists and is not empty` | it never overwrites: give another directory |
| `JobDir: not found: .../music_input` | `JETSCAPE_BUILD` or `XSCAPE` set to something else; in the image leave them unset |
| `ModuleNotFoundError: awkward` (fastjet, Jupyter, ...) | a `-slim` image, or an image older than 2026-10-28 |
| Jupyter: page does not load | Docker: `-p 8888:8888` missing; Apptainer: tunnel to the node, not the login node |
| Matter warnings `Second instance ... el_vector` for ISR partons | harmless: an ISR parton before t = 0 outside the light cone, no medium there |

---

## Reference

**Helpers** (on `PATH`; sources in [`utils/container/`](container/)):

| command | does |
|---|---|
| `xscape-examples [DEST]` | copies the examples to `DEST` (default `/work/examples`); never overwrites |
| `xscape-jupyter [DIR] [options]` | JupyterLab in `DIR` (default `/work`), port `$JUPYTER_PORT` (8888); 0.0.0.0 in Docker, 127.0.0.1 in Apptainer (`$JUPYTER_IP`) |
| `xscape-entry` | the entrypoint: a writable `HOME` for `docker --user`, then your command |
| `python .../utils/container/image_check.py [--analysis] [--viz]` | checks the image |

**Environment** (set in the image):

| variable | value | used by |
|---|---|---|
| `XSCAPE` | `/opt/X-SCAPE` | `jetscape.paths`, the production scripts: the tree |
| `JETSCAPE_BUILD` | unset (→ `$XSCAPE/build_gpu`) | `jetscape.paths`: another build |
| `PYTHONPATH` | PyJetscape's and FastHydro's `python/` | `import jetscape`, `import fasthydro` |
| `PYTHIA8DATA` | Pythia's xmldoc | Pythia |
| `MUSIC_EOS_TABLE` | MUSIC's EoS 9 table | e → T for pair files (analysis) |
| `MPLBACKEND` | `Agg` | matplotlib without a display (Jupyter shows plots inline anyway) |

**In Python:** `jetscape.paths.xscape_root()`, `build_dir()`, `main_xml()`, `examples_dir()`;
`js.JobDir(out_dir, user_xml, *, name=None, main_xml=None, build=None, keep=False)`
(`job.main_xml`, `job.user_xml`, `job.path`).
