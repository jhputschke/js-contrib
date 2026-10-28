<!-- Design note, 2026-10-28. Branch `container_user` from js-contrib `main` 8a1165d.
     X-SCAPE unchanged (`contrib` 38e90519). Decisions taken 2026-10-28 (see Decisions). -->

# Plan: one production image for productions, own scripts and notebooks, and analysis

The production image (`utils/Dockerfile.prod`, `xscape-prod:cu126/cu130/cu124`) is built
for one thing: running the production scripts from the image tree with outputs in `/work`.
It should cover three uses:

| use | how it runs |
|---|---|
| production scripts | as now: `run_jobs.sh`, `run_prod_jet.py`, `run_prod_oo_isr.py`, `launch_2gpu*.sh`, outputs in `/work` |
| own scripts and notebooks | your files in `/work` (e.g. a copy of `oo_workflow.ipynb`), X-SCAPE, `jetscape` and the tables from the image; Jupyter optional |
| analysis only | the same image with or without a GPU, on files in `/work` or remote (pelicanfs), e.g. `oo_analysis.ipynb`, the `analysis/tutorials` notebooks |

## Status

- **2026-10-28:** plan and decisions; nothing implemented yet.

## What stands in the way today

1. **Python packages:** the image has numpy, h5py, hdf5plugin, pyyaml, pelicanfs only. No
   awkward, vector, fastjet, particle, matplotlib, hist, mplhep, scipy, pandas, uproot or
   Jupyter: `HadronArray`, `per_event`, `HadronH5Writer` work, `to_awkward`, `select`,
   FastJet and every plot don't.
2. **Image age:** the latest images (`cu1xx-20261013-9e84450`) predate js-contrib #75/#76
   (the Pythonic layer) and X-SCAPE #181/#182 (colour tags).
3. **`import jetscape`:** no `PYTHONPATH` in the image; the production scripts put
   `PyJetscape/python` on `sys.path` themselves.
4. **Paths from the file's own location:** `oo_workflow.ipynb` (and several examples) take
   the X-SCAPE tree from `Path.cwd().parents[...]` and write next to themselves. In the
   container that is `/opt/X-SCAPE`: read-only (Apptainer) or root-owned (Docker `--user`).
   A copy in `/work` finds neither the build nor the main XML.
5. **Own scripts reach the work-directory plumbing only by loading an example by path**
   (`importlib.util.spec_from_file_location(".../prod_AuAu_0_10/run_prod.py")`), with its
   Au+Au defaults and module-level paths.
6. **No Jupyter, and no writable HOME** with Docker `--user <uid>` (HOME is `/`): Jupyter
   and matplotlib cannot write their config/runtime files.

## Decisions (2026-10-28)

- **D1 Default tags are the full image** (analysis + Jupyter packages);
  `--build-arg WITH_ANALYSIS=0` builds `-slim` tags (production only, as today).
- **D2 pyvista/vtk optional:** `--build-arg WITH_VIZ=1` (`-viz` tags on request); not in
  the default image. `contribs/Visualization` otherwise runs in `analysis_env`.
- **D3 Work-directory and seed helpers move into the package** as `jetscape.workdir`,
  with a `JobDir` context manager; `run_prod.py` re-exports them under their old names,
  so every production script, `run_jobs.sh` and the launchers stay as they are.
- **D4** This plan on `container_user`; implementation in the order of the steps below.

## Rules

- **R1 Productions unchanged:** same commands, options, work directories, environment
  variables, file names, seeds and file contents. Checked by the regression in step 5.
- **R2 One package list:** the image installs the analysis packages from
  `utils/analysis_env/requirements.txt` (the build context is `utils/`), so the image and
  the venv cannot drift apart.
- **R3 numpy and h5py are not touched by pip:** a constraints file with the versions of the
  production layer (`pyjetscape_core` is compiled against the conda numpy).
- **R4 Image tree read-only:** nothing in the image writes into `/opt/X-SCAPE` at run time.

## Step 1: `jetscape.paths` and `jetscape.workdir` (js-contrib, code)

- `python/jetscape/paths.py`: `xscape_root()` (`$XSCAPE`, else from the package's
  location: `<root>/external_packages/js-contrib/contribs/PyJetscape/python/jetscape`),
  `build_dir()` (`$JETSCAPE_BUILD`, else `<root>/build_gpu`), `main_xml()`,
  `pyjetscape_dir()`.
- `python/jetscape/workdir.py`, moved from `example/prod_AuAu_0_10/run_prod.py` without
  behaviour change: `WORKDIR_LINKS`, `_absolute_parent_paths`, `enter_workdir`,
  `job_workdir`, `leave_workdir`, `check_env` (+ `pythia_data_problem`), and the seed
  helpers (`SEED_MAX`, the registry, `add_seed_args`, `seed_args_problems`,
  `output_name`, `seed_registry`, `resolve_seed`, `seed_provenance`), plus
  `add_workdir_args`.
  - The functions take plain arguments; `run_prod.py` keeps thin wrappers taking the
    parsed-argument object `a`, under the old names.
  - `JobDir(out_dir, user_xml, main_xml=None, build=None, keep=False)`: context manager,
    enters a private work directory under `out_dir/work/<name>`, yields an object with
    `main_xml`, `user_xml`, `path`; removes it on success unless `keep`.
- `run_prod.py`: imports and re-exports; its Au+Au parser, `job_xml`, grid YAML checks
  and `main` stay.
- Tests: `tests/test_workdir.py` (JobDir in a temporary directory, `../` rewriting, links,
  cleanup); `test_prod_seeds.py` reads `run_prod.py` and `workdir.py` where it inspects the
  source text (line ~239).

## Step 2: examples and templates (js-contrib)

- `oo_workflow.ipynb`: the tree from `jetscape.paths`, `JobDir` instead of loading
  `run_prod.py`, outputs to `$OO_OUT` (default `./out_oo` relative to the current
  directory, not to the notebook's location). Runs from a copy in `/work`.
- `example/pythonic/oo_workflow.py`: the notebook's simulation half as a script
  (`--events`, `--outdir`, `--xml`).
- `example/pythonic/my_production.py`: minimal template: `JobDir` + `js.per_event` +
  `driver.events()` + `HadronH5Writer`.
- Audit: outputs to `--outdir` or the current directory, never next to the script:
  `prod_OO_isr/hard_sub_compare`, `prod_OO_isr/hadro_compare`, `hydro_hist_vs_surface`,
  `python_modules`, `per_event_loop.py`, the analysis scripts' default `runs/` directories.
- `utils/xscape-examples` (installed into the image's PATH): copies `example/` (scripts,
  notebooks, XMLs, configs; no outputs) into a given directory, e.g. `/work/examples`.

## Step 3: the image (`utils/Dockerfile.prod`)

- `ARG WITH_ANALYSIS=1`, `ARG WITH_VIZ=0`. With analysis: pip install of
  `analysis_env/requirements.txt` with a constraints file (R3) generated from the runtime
  env (`numpy==`, `h5py==`, `hdf5plugin==`); with viz: `requirements_viz.txt` + the
  off-screen libraries (OSMesa or EGL, to be decided when built).
- `ENV PYTHONPATH=<PyJetscape>/python:<FastHydro>/python`, `MPLBACKEND=Agg`,
  `OMP_WAIT_POLICY=passive` (an open point of BuildContainerProd.md).
- Entrypoint `xscape-entry`: if HOME is not writable, `HOME=/tmp/home` (created); then
  `exec "$@"`. `--entrypoint test` in `launch_2gpu.sh` bypasses it, as now.
- `xscape-jupyter`: `jupyter lab --ip 0.0.0.0 --no-browser --notebook-dir ${1:-/work}`;
  `EXPOSE 8888`.
- `XSCAPE_REF` default → `contrib` (the CI passes the resolved commit anyway).
- Build-time checks: import every analysis package, `check_env.py` without a directory,
  a `to_awkward`/`select`/FastJet smoke test on a small synthetic event, an Agg plot;
  `pip freeze` in `BUILD_INFO.txt` as now.
- Before the build: FastHydro's matplotlib 3.11 deprecation (`cmap.set_bad` →
  `cmap.with_extremes(bad=...)`).

## Step 4: validation

On the GB10 (arm64, cu130, local build), then amd64 via CI:

1. **Regression (R1):** Au+Au jet and O+O productions natively on `main` and on this branch,
   and in the new image: same seeds, identical pair and hadron files. PyJetscape tests,
   including `test_prod_seeds.py` and the new `test_workdir.py`.
2. **Own notebook:** `oo_workflow.ipynb` copied to `/work`, `nbconvert --execute` with the
   GPU: same results as natively.
3. **Analysis only, no GPU:** `oo_analysis.ipynb` (OSDF), `analysis/tutorials/jet_fastjet*`,
   `hadron_distributions`, `check_env.py` on a production directory.
4. **Docker `--user <uid>`:** Jupyter starts (HOME fix, port, token), matplotlib writes.
5. **Launchers:** `launch_2gpu.sh` / `launch_2gpu_oo.sh` with a few events on the RTX box
   against the new image: checks pass, throughput as before.
6. Open beyond this branch: Apptainer, amd64 runtime on an amd64 machine.

## Step 5: CI (`.github/workflows/docker-prod.yml`)

- Input `with_analysis` (default true; false → `-slim` tags), `with_viz` (default false →
  `-viz` tags), alongside `with_gcs`; tag suffixes combine (`cu126-slim-gcs`, ...).
- Rebuild from X-SCAPE `contrib` and js-contrib `main` after the merge. Note in the
  release notes: the new images contain X-SCAPE #181 (jet hadrons with iColored change by
  5–17 % in the O+O Matter-only test), so old and new images are not mixed in one campaign.

## Step 6: documentation

- `utils/BuildContainerProd.md`: section "Using the image" with three recipes
  (productions as now; own scripts and notebooks: `/work`, `xscape-examples`,
  `JobDir`, Jupyter with Docker `-p 8888:8888` and with Apptainer on HPC: interactive
  SLURM job, `xscape-jupyter`, ssh tunnel to the node; analysis only: no `--gpus`,
  pelicanfs); the tags (`-slim`, `-viz`); a developer note (own js-contrib `python/` mounted
  in front works for pure-Python changes with an unchanged `pyjetscape_core`, else the dev
  image).
- `PYTHONIC.md`: `jetscape.paths`, `jetscape.workdir`/`JobDir`; links to the recipes.
- `utils/analysis_env/README.md`, `docs/README_analysis.md`: the image as the alternative
  to the venv.

## Open questions

- Off-screen rendering for `WITH_VIZ` (OSMesa vs EGL with the NVIDIA driver).
- Image size: measure full vs slim; if the 1.2 GB LBT tables matter, a bind-mounted
  `LBT_TABLES_PATH` (already an open point).
- Registry for clusters (Docker Hub limits vs `ghcr.io`), unchanged by this plan.
