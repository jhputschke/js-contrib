# Production container

The production image `jhputschke/xscape-prod` is the ready-made setup for Linux machines with
an NVIDIA GPU: workstations, cloud VMs (Docker) and HPC clusters (Apptainer). It holds X-SCAPE
with MUSIC on the GPU (MUSIC4GPU, CUDA), iSS, 3dMCGlauber, Matter, LBT, iColored and Pythia,
all compiled, together with PyJetscape and its Pythonic layer and the analysis packages
(awkward, vector, fastjet, particle, hist, mplhep, matplotlib, scipy, pandas, uproot,
JupyterLab). Nothing is compiled on your machine. One image serves three uses:

| use | GPU | you bring |
|---|---|---|
| **the production scripts** (`run_prod_jet.py`, `run_prod_oo_isr.py`, `run_jobs.sh`, `run_hadronize.py`, ...) | yes; stage 2 (hadronization) no | an output directory |
| **your own scripts and notebooks** (the O+O workflow, the examples, Python modules of your own) | yes | your scripts, notebooks and XMLs in a directory |
| **analysis only** (hadron and pair files, FastJet, plots, notebooks) | no | the files, or OSDF paths |

!!! note "On a Mac"
    Containers on macOS can't reach the GPU: Docker runs them in a Linux VM without Metal.
    Use the [xscape wheel](mac.md) there; the scripts, notebooks and outputs are the same.

## 1. Requirements

- **For simulations:** an NVIDIA GPU with a driver recent enough for the image's CUDA
  version (table below), and either
    - **Docker** with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
      (workstation, cloud VM), or
    - **Apptainer** (or Singularity) on an HPC cluster; it needs no extra toolkit.
- **For analysis and hadronization only:** any Linux machine with Docker or Apptainer; no GPU.
- **Disk:** the full image is about 5.4 GB (5.36 GB for `cu130` on arm64); the `-slim`
  variant is 0.6 GB smaller.

Pick the tag by the CUDA version your driver supports. `nvidia-smi` on the host prints it in
the top right corner ("CUDA Version").

| tag | CUDA | GPUs | host driver | CPU architectures |
|---|---|---|---|---|
| `cu126` | 12.6 | V100, RTX 20xx–40xx, A100, A40, L4, H100, GH200 | supports CUDA 12.6 or later (R560+) | amd64, arm64 |
| `cu130` | 13.2 | GH200, B200/GB200, RTX 50xx / RTX PRO Blackwell, GB10; **no V100** | supports CUDA 13.2 or later (R580+) | amd64, arm64 |
| `cu124` | 12.4 | as `cu126` | older drivers (R550) | amd64 only |

Each tag is a multi-architecture manifest: `docker pull` and `apptainer pull` take the image
for the machine's CPU (x86_64 or ARM, e.g. GH200 and GB10). There is no performance difference
to expect between `cu124` and `cu126`.

!!! tip "For a campaign, pin the image"
    `cu126` is a **moving** tag: it always points at the latest build. For a production
    campaign take a fixed, dated tag, `cu126-<YYYYMMDD>-<xscape commit>`, or the image digest,
    and use the same one on every machine. Physics fixes change results, so don't mix images
    of different builds in one campaign, and record which image you used: the output files
    don't say. The image's `/opt/X-SCAPE/BUILD_INFO.txt` lists the commits it was built from.

## 2. Pull the image and check it

=== "Docker"

    ```bash
    docker pull jhputschke/xscape-prod:cu126
    docker run --rm --gpus all jhputschke/xscape-prod:cu126 nvidia-smi       # the GPU is seen
    docker run --rm jhputschke/xscape-prod:cu126 \
      python /opt/X-SCAPE/external_packages/js-contrib/utils/container/image_check.py --analysis
    ```

=== "Apptainer"

    ```bash
    apptainer pull xscape_prod.sif docker://jhputschke/xscape-prod:cu126
    apptainer exec --nv xscape_prod.sif nvidia-smi
    apptainer exec xscape_prod.sif \
      python /opt/X-SCAPE/external_packages/js-contrib/utils/container/image_check.py --analysis
    ```

`image_check.py` ends with `image_check: all ok`. To confirm that the bindings were built
against the GPU MUSIC:

```bash
docker run --rm jhputschke/xscape-prod:cu126 \
  python -c "import jetscape as js; print(js.HAS_CORE, js.pyjetscape_core.MUSIC_BACKEND)"
# -> True music4gpu-cuda
```

## 3. The rules of the container

Everything else on this page follows from four rules.

1. **The image is read-only; your files live in `/work`.** The X-SCAPE tree is
   `/opt/X-SCAPE` (read-only in Apptainer, owned by root in Docker). Mount a directory of
   yours at `/work` (Docker `-v "$PWD/work:/work"`, Apptainer `--bind "$PWD/work:/work"`) and
   write every output there. A file written anywhere else is gone when the container ends.
2. **Run as yourself.** With Docker, pass `--user "$(id -u):$(id -g)"`, or the outputs belong
   to root. Apptainer runs as you anyway.
3. **GPU access is explicit.** Docker `--gpus all` (or `--gpus device=1`), Apptainer `--nv`.
   Without it MUSIC falls back to its CPU path: about 10× slower, and its numbers are not the
   GPU's.
4. **Everything finds the tree by itself.** `import jetscape` works everywhere; scripts and
   notebooks find the build through `$XSCAPE=/opt/X-SCAPE` (`jetscape.paths`), and each job
   runs in a private working directory under its output directory, so jobs never share files
   and nothing writes into the image. Leave `XSCAPE` and `JETSCAPE_BUILD` as the image sets
   them.

Where things are inside the container:

| | path |
|---|---|
| X-SCAPE tree, build | `/opt/X-SCAPE`, `/opt/X-SCAPE/build_gpu` |
| js-contrib | `/opt/X-SCAPE/external_packages/js-contrib` |
| examples and production scripts | `/opt/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape/example` |
| start directory | Docker: `.../example/prod_AuAu_0_10_jet`; Apptainer: the directory you start it from |
| your files | `/work` (you mount it) |
| X-SCAPE's default parameters | `/opt/X-SCAPE/config/jetscape_main.xml` |

## 4. Your first event

`xscape-examples` copies every example into a writable directory (the examples write next to
themselves, so they can't run inside the read-only image). Then one O+O event runs through the
whole chain: 3dMCGlauber, Pythia with initial-state radiation, MUSIC on the GPU, Matter + LBT,
iColored + iSS. It takes under a minute on a GPU and leaves two hadron files in
`./work/out_oo/`.

=== "Docker"

    ```bash
    mkdir -p work
    docker run --rm --gpus all --user "$(id -u):$(id -g)" -v "$PWD/work:/work" -w /work \
      jhputschke/xscape-prod:cu126 bash -c \
      'xscape-examples /work/examples && cd /work/examples/pythonic &&
       python oo_workflow.py --events 1 --outdir /work/out_oo'
    ```

=== "Apptainer"

    ```bash
    W=$SCRATCH/xscape; mkdir -p $W
    apptainer exec --bind $W:/work xscape_prod.sif xscape-examples /work/examples
    apptainer exec --nv --pwd /work/examples/pythonic --bind $W:/work xscape_prod.sif \
      python oo_workflow.py --events 1 --outdir /work/out_oo
    ```

`work/examples/` now holds the production scripts, the Pythonic examples (`pythonic/`), the toy
Python modules (`python_modules/`), the analysis notebooks and their XMLs, editable on the host
with your own editor. `xscape-examples` never overwrites: for a fresh copy, give it a new
directory. What the event does and what the files contain:
[Your first event](../run/first-event.md).

## 5. Running the productions

The production scripts run from the image's own copy, with their outputs in `/work`. Docker
starts in `prod_AuAu_0_10_jet`; for O+O, change the directory with `-w`.

```bash
IMG=jhputschke/xscape-prod:cu126
mkdir -p prod

# Au+Au 0-10 % with a jet: 2 events
docker run --rm --gpus all --user "$(id -u):$(id -g)" -v "$PWD/prod:/work" $IMG \
  python run_prod_jet.py --events 2 --seed 1 --write-particlize both --outdir /work/out

# stage 2 (hadronization) needs no GPU
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/prod:/work" $IMG \
  python run_hadronize.py /work/out -j 4 --oversample 500 --n-frag 50

# O+O with initial-state radiation
docker run --rm --gpus all --user "$(id -u):$(id -g)" -v "$PWD/prod:/work" \
  -w /opt/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape/example/prod_OO_isr $IMG \
  python run_prod_oo_isr.py --events 2 --seed 5 --outdir /work/out_oo
```

!!! warning "Put OUTDIR one level inside `/work`"
    Use `/work/out`, not `/work`: every job records its seed in `OUTDIR/../seeds_used.tsv`, so
    the parent of OUTDIR must be writable too.

Your own physics settings go in a user XML that you mount and pass:
`-v "$PWD/xml:/xml:ro"` and `--user-xml /xml/PbPb_0_10.xml`. Campaigns, several jobs per GPU,
SLURM arrays and the two stages are explained in
[Au+Au jets (two stages)](../run/auau-jet.md), [O+O with ISR](../run/oo-isr.md),
[Z+jet and other systems](../run/zjet-other.md) and
[Clusters, GPUs and monitoring](../run/clusters.md).

## 6. Your own scripts and notebooks

### A script of your own

Put the script and its user XML in `work/` (start from a copy of
`examples/pythonic/oo_workflow.py` and `oo_workflow.xml`). Every such script has the same
skeleton:

```python
# work/my_job.py
import jetscape as js
from jetscape.hadrons_h5 import HadronH5Writer

with js.JobDir("/work/out_mine", "/work/my_job.xml") as job:      # private working directory
    with js.per_event(job.main_xml, job.user_xml) as driver, \
         HadronH5Writer("/work/out_mine/my_job_hadrons.h5", tag="jet_frag", n_samples=1) as out:
        for i in driver.events():                                 # one iteration per event
            jets = driver.hadronization_manager.hadron_array()
            out.append_unit(jets.to_dict(), unit=i, event=i)
            print(i, len(jets), "jet hadrons")
```

```bash
docker run --rm --gpus all --user "$(id -u):$(id -g)" -v "$PWD/work:/work" -w /work \
  jhputschke/xscape-prod:cu126 python my_job.py
```

`js.JobDir(out_dir, user_xml)` copies your XML to `out_dir/`, makes its relative paths (such
as iSS's tables) absolute for the image's build, runs the job in a private directory
`out_dir/work/<name>` and removes that directory after a successful run. Inside the `with`
block the current directory is that working directory, so use absolute output paths. The
interface (`driver.events()`, `hadron_array()`, the hadron files) is described in
[The Pythonic interface](../python/pythonic.md); writing physics modules in Python in
[Writing modules in Python](../python/modules.md).

### Python packages that are not in the image

Install them once into `/work` and put that directory on the path:

```bash
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/work:/work" jhputschke/xscape-prod:cu126 \
  pip install --target /work/pylib seaborn
docker run ... -e PYTHONPATH=/work/pylib:/opt/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape/python ...
```

Don't upgrade numpy or h5py this way: `pyjetscape_core` is compiled against the image's numpy.
For a group, a small Dockerfile of your own is cleaner:
`FROM jhputschke/xscape-prod:cu126-<date>-<commit>` plus `RUN pip install --no-cache-dir seaborn`.

### Changing js-contrib's Python code

The image uses its own `jetscape` package unless yours comes first on the path. For changes to
pure Python, mount your js-contrib checkout and put its package in front:

```bash
docker run ... -v "$HOME/js-contrib:/src/js-contrib:ro" \
  -e PYTHONPATH=/src/js-contrib/contribs/PyJetscape/python ...
```

This works only if your checkout is close to the image's js-contrib commit (`BUILD_INFO.txt`),
and the compiled `pyjetscape_core*.so` must come from the image (copy it from the image's
`python/jetscape/` into your checkout's). Changes to C++ need the [source build](source.md).

## 7. Analysis without a GPU

Without `--gpus` (Docker) or `--nv` (Apptainer) the image is an analysis environment. It also
runs stage 2 (hadronization) on CPU-only nodes, unchanged.

```bash
# a script of yours on your files
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/work:/work" -w /work \
  jhputschke/xscape-prod:cu126 python my_analysis.py /work/data

# open every file of a production directory
docker run --rm -v "$PWD/prod:/work:ro" jhputschke/xscape-prod:cu126 \
  python /opt/X-SCAPE/external_packages/js-contrib/utils/analysis_env/check_env.py /work/out --no-render
```

pelicanfs reads `osdf://` paths in place; public namespaces need no token, a protected one
takes `-e BEARER_TOKEN_FILE=/work/token`. Notebooks to start from (in the `xscape-examples`
copy):

| notebook | reads |
|---|---|
| `pythonic/oo_analysis.ipynb` | the O+O hadron files, streamed from the OSDF |
| `pythonic/oo_workflow.ipynb` (sections 4–6) | the hadron files of your own run |
| `analysis/tutorials/hadron_distributions.ipynb`, `jet_fastjet*.ipynb` | a production directory |
| `prod_AuAu_0_10_jet/jet_wake.ipynb` | a pair file and its hadron files |
| `prod_OO_isr/oo_isr_overview.ipynb` | an O+O production |

How to read the files: [Analyse](../analysis/index.md).

## 8. Jupyter

`xscape-jupyter [DIR]` starts JupyterLab on port 8888 with `DIR` (default `/work`) as its
root and prints the URL with its access token.

```bash
docker run --rm -it --gpus all --user "$(id -u):$(id -g)" -p 8888:8888 \
  -v "$PWD/work:/work" jhputschke/xscape-prod:cu126 xscape-jupyter
```

Open the `http://127.0.0.1:8888/lab?token=...` line it prints. Notebooks run with the image's
Python ("Python 3 (ipykernel)"), and simulations in a notebook use the GPU as scripts do. Stop
it with Ctrl-C twice.

- **Another port:** `-p 8890:8888`, then open `127.0.0.1:8890`.
- **On a remote GPU server:** start it there as above, then on your laptop run
  `ssh -N -L 8888:localhost:8888 user@server` and open the URL locally.
- **Keep Jupyter's settings and history:** `$HOME` is a temporary directory with `--user`;
  add `-e HOME=/work/.home`.
- **Analysis only:** the same command without `--gpus all`.

## 9. HPC clusters with Apptainer

Apptainer runs the same image as you, with your home directory mounted and the host's network.
It differs from Docker in three ways: `--nv` for the GPU, `--bind` for `/work`, and it starts in
the current directory, not the image's start directory, so pass `--pwd` or full paths.

```bash
apptainer pull xscape_prod.sif docker://jhputschke/xscape-prod:cu126-<date>-<commit>
W=$SCRATCH/xscape; mkdir -p $W
EX=/opt/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape/example

# a production
apptainer exec --nv --pwd $EX/prod_AuAu_0_10_jet --bind $W:/work xscape_prod.sif \
  python run_prod_jet.py --events 2 --seed 1 --outdir /work/out

# a script of your own
apptainer exec --nv --pwd /work/examples/pythonic --bind $W:/work xscape_prod.sif \
  python oo_workflow.py --events 3 --outdir /work/out_oo

# analysis, no GPU
apptainer exec --pwd /work --bind $W:/work xscape_prod.sif python my_analysis.py
```

In a SLURM batch script these lines run as they are (after `#SBATCH --gpus=1` or your site's
equivalent). Job arrays, several GPUs and the LBT table cache: [Clusters, GPUs and
monitoring](../run/clusters.md).

- **Mixed clusters:** a `.sif` file holds one CPU architecture. Where the login nodes are x86
  but the GPU nodes are ARM (GH200), pull with `apptainer pull --arch arm64`, or pull on a
  compute node.
- **Home directory:** Apptainer mounts it, but write outputs to `/work` on scratch, not to `~`.

**Jupyter on a compute node:**

```bash
salloc --gpus=1 --time=2:00:00                  # your site's options; you land on a node
apptainer exec --nv --bind $SCRATCH/xscape:/work xscape_prod.sif xscape-jupyter
```

In Apptainer, `xscape-jupyter` listens on 127.0.0.1 only, since the node is shared. Tunnel from
your laptop through the login node to the compute node:

```bash
ssh -N -L 8888:localhost:8888 -J user@login.cluster user@<node>   # if nodes accept ssh
ssh -N -L 8888:<node>:8888 user@login.cluster                     # otherwise, with JUPYTER_IP=0.0.0.0
```

For the second form start Jupyter with `apptainer exec --env JUPYTER_IP=0.0.0.0 ...`; the token
still protects it.

## 10. Image variants

Each CUDA tag comes in flavours, marked by a suffix; suffixes combine (`cu126-slim-gcs`).

| suffix | what changes | for |
|---|---|---|
| (none) | the full image: productions, analysis packages, JupyterLab, the helpers | everything on this page |
| `-slim` | no analysis packages and no Jupyter; 0.6 GB smaller | production runs only |
| `-viz` | adds pyvista/vtk with off-screen rendering | [3D visualization](../python/visualization.md) |
| `-gcs` | adds Google Cloud Storage for Python (`gcsfs`, `google-cloud-storage`) | reading and writing `gs://` |

Which variants are published at a given time is visible on
[Docker Hub](https://hub.docker.com/r/jhputschke/xscape-prod/tags). Every image has
Pelican/OSDF support (`pelicanfs` and the `pelican` command-line tool). The images contain
neither ROOT nor PyTorch.

Images built before 2026-10-28 (the date is in the dated tag) lack the analysis packages, the
helpers (`xscape-examples`, `xscape-jupyter`) and the preset `PYTHONPATH`; they only run the
production scripts.

## 11. Troubleshooting

| symptom | cause and fix |
|---|---|
| `Permission denied` or `Read-only file system` when writing | writing into the image, or into `/work` itself where OUTDIR's parent must be writable: write to `/work/<something>`; with Docker use `--user` and a directory you own |
| outputs owned by root | Docker without `--user "$(id -u):$(id -g)"`; fix once with `sudo chown -R $USER work/` |
| `No CUDA device found ... falling back to CPU` | no `--gpus all` / `--nv`, or the driver is older than the image's CUDA: check `nvidia-smi`; take `cu124` for R550 drivers |
| `docker: could not select device driver "" with capabilities: [[gpu]]` | the NVIDIA Container Toolkit is not installed on the host |
| `no kernel image is available for execution on the device` | the image has no code for this GPU (e.g. V100 with `cu130`): take `cu126` or `cu124`; discard files written by such a run |
| `python: can't open file 'run_prod_jet.py'` | Apptainer: add `--pwd $EX/prod_AuAu_0_10_jet` or use the full path |
| `xscape-examples: ... exists and is not empty` | it never overwrites: give another directory |
| `JobDir: not found: .../music_input` | `JETSCAPE_BUILD` or `XSCAPE` set to something else; in the image leave them as they are |
| `ModuleNotFoundError: awkward` (fastjet, Jupyter, ...) | a `-slim` image, or an image built before 2026-10-28 |
| Jupyter page does not load | Docker: `-p 8888:8888` missing; Apptainer: tunnel to the compute node, not the login node |
| Matter warnings `Second instance ... el_vector` for ISR partons | harmless: an ISR parton before t = 0 outside the light cone, where there is no medium |

Full reference (helpers, environment variables, the image build):
[`utils/README_Container_Usage.md`](https://github.com/jhputschke/js-contrib/blob/main/utils/README_Container_Usage.md),
[`utils/BuildContainerProd.md`](https://github.com/jhputschke/js-contrib/blob/main/utils/BuildContainerProd.md).

## Next: run your first event

You ran one event above; [Your first event](../run/first-event.md) explains what it did, what
the output files hold and where to go from there.
