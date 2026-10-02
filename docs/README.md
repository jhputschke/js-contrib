# js-contrib design notes and benchmarks

Plans, measurements and findings behind the contribs, the productions and the containers.
The user documentation is in the contrib READMEs ([`contribs/`](../contribs/README.md),
[PyJetscape](../contribs/PyJetscape/README.md), [FastHydro](../contribs/FastHydro/README.md))
and the production READMEs
([`prod_AuAu_0_10`](../contribs/PyJetscape/example/prod_AuAu_0_10/README.md),
[`prod_AuAu_0_10_jet`](../contribs/PyJetscape/example/prod_AuAu_0_10_jet/README.md)).

**Running a production with the containers** (Docker, Apptainer, SLURM; hadronization on
CPUs; analysis and ROOT export): [`README_2stage.md`](README_2stage.md).

The plans (`PLAN_*.md`, `PlanContainer*.md`) are in [`Plans/`](Plans/); the user guide,
the HDF5 study and the benchmarks stay here.

## HDF5 writers and productions

| file | contents |
|---|---|
| [`PLAN_consolidate_h5_writer.md`](Plans/PLAN_consolidate_h5_writer.md) | one FNO4d HDF5 writer shared by `fast_data`, FastHydro and PyJetscape (decisions, phases 0–5) |
| [`PLAN_pair_h5_music.md`](Plans/PLAN_pair_h5_music.md) | background/jet MUSIC pairs → pair files (`pair_h5.py`, `prod_AuAu_0_10_jet`) |
| [`PLAN_particlize_h5.md`](Plans/PLAN_particlize_h5.md) | hadron level: stored surfaces and partons, `hadronize.py`, validation |
| [`PLAN_iSS_optim.md`](Plans/PLAN_iSS_optim.md) | faster iSS (Part A, bit-identical) and correlated jet/background sampling (Part B) |
| [`PLAN_analysis_decays.md`](Plans/PLAN_analysis_decays.md) | π⁰ and weak decays of the hadron files at the analysis level (`jetscape.decays`, `run_h5toROOT.py --decays`; not started) |
| [`PLAN_slim_bulk_info.md`](Plans/PLAN_slim_bulk_info.md) | a 6-field background copy (`bulk_info`): ~20 → ~14 GB peak per jet job, bit-identical (not started) |
| [`README_h5_optim.md`](README_h5_optim.md) | HDF5 compression of `arr` / `arr_bg`: Blosc-zstd default, `keep_bits` |

## Benchmarks

| file | contents |
|---|---|
| [`BENCHMARK_GB10.md`](BENCHMARK_GB10.md) | GB10 (CUDA): per-event profile, speed-ups, concurrent jobs, MPS, thread settings |
| [`BENCHMARK_M3MAX.md`](BENCHMARK_M3MAX.md) | Apple M3 Max (Metal): the same measurements, `run_jobs.sh` on macOS |

## Containers

| file | contents |
|---|---|
| [`README_2stage.md`](README_2stage.md) | **user guide**: the two-stage production with the production images, from GPU jobs to hadrons and ROOT files |
| [`PlanContainer.md`](Plans/PlanContainer.md) | one Docker image with X-SCAPE, music4gpu, js-contrib and FNO4d built in |
| [`PlanContainerDev.md`](Plans/PlanContainerDev.md) | source-free dev containers (`utils/Dockerfile.dev`, `utils/Dockerfile.dev.blackwell`) |

How to build and publish the images: [`utils/BuildContainerDev.md`](../utils/BuildContainerDev.md)
(dev) and [`utils/BuildContainerProd.md`](../utils/BuildContainerProd.md) (production).

## Retired

Two finished plans were removed and are kept in the git history:
[`FastHydro/PLAN_hadronization.md`](https://github.com/jhputschke/js-contrib/blob/ece9d53/contribs/FastHydro/PLAN_hadronization.md)
(FastHydro soft particlization) and
[`Visualization/PlanVisualization.md`](https://github.com/jhputschke/js-contrib/blob/ece9d53/contribs/Visualization/PlanVisualization.md)
(PyVista visualizer).
