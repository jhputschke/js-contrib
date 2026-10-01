# utils — environments, containers, data tools

Tools around the js-contrib code: setting up an environment, building the container
images, running productions on a cluster, inspecting and moving the files. Each entry links
to its own documentation.

## Environments

| | what | docs |
|---|---|---|
| [`conda_install/`](conda_install/) | creates and checks the `js_fno` conda env: the Python/ML stack, optionally with the C++ build tools for X-SCAPE and js-contrib (`install_js_fno_minimal.sh`, `install_js_fno_build_minimal.sh`), or pinned versions (`pinned/`); `test_js_fno_build_env.sh` checks every package without installing | [`contribs/README.md`](../contribs/README.md) |
| [`analysis_env/`](analysis_env/README.md) | a Python venv for analysing the files of a production without X-SCAPE: HDF5 readers, uproot, the notebooks, the Visualization scripts; optionally PyROOT, GCS and Pelican/OSDF for Python | [`analysis_env/README.md`](analysis_env/README.md) |
| [`configure.sh`](configure.sh) | interactive CMake configurator for X-SCAPE + js-contrib (needs `dialog`) | the script |

## Containers and clusters

| | what | docs |
|---|---|---|
| [`Dockerfile.dev`](Dockerfile.dev), [`Dockerfile.dev.blackwell`](Dockerfile.dev.blackwell) | development images: the build and runtime environment without sources (CUDA 12.6 / 13.2) | [`BuildContainerDev.md`](BuildContainerDev.md) |
| [`Dockerfile.prod`](Dockerfile.prod) | production images: X-SCAPE, MUSIC4GPU, iSS, 3dMCGlauber and PyJetscape built in, ready for `prod_AuAu_0_10_jet` | [`BuildContainerProd.md`](BuildContainerProd.md); running them: [`docs/README_2stage.md`](../docs/README_2stage.md) |
| [`slurm_prod_array.sh`](slurm_prod_array.sh) | a production campaign as a SLURM array in the production container, one GPU per task | [`BuildContainerProd.md`](BuildContainerProd.md), *SLURM* |

## Data

| | what | docs |
|---|---|---|
| [`remote_transfer/`](remote_transfer/README.md) | `js_gcs.py` (Google Cloud Storage) and `js_osdf.py` (Pelican/OSDF, pelicanfs): upload and download production directories, single files or patterns, picked by kind (`--what pair\|h5\|root\|all`). They skip files already there, check every transfer, and each runs in an environment of its own | [`remote_transfer/README.md`](remote_transfer/README.md) |
| [`h5_inspect.py`](h5_inspect.py) | every group and dataset of an HDF5 file, plus a summary of the js-contrib hydro formats (grid, axes, freeze-out, `shower/`, `source/`, `diag/`) with consistency checks | the script; [main README](../README.md#utilities) |
| [`h5_compression_bench.py`](h5_compression_bench.py) | compression ratio, write and read speed of the HDF5 filters on a real evolution dataset | [`docs/README_h5_optim.md`](../docs/README_h5_optim.md) |

```bash
python utils/h5_inspect.py out/AuAu_0_10_jet_seed0001.h5            # summary + tree
utils/remote_transfer/js_gcs.py upload /data/AuAu_c1 --what root      # -> gs://test_fno/AuAu_c1/
utils/remote_transfer/js_osdf.py download AuAu_c1 --what h5 --to /scratch   # osdf:///fno4hic/AuAu_c1/
```
