# js-contrib design notes and benchmarks

Plans, measurements and findings behind the contribs, the productions and the containers.
The user documentation is in the contrib READMEs ([`contribs/`](../contribs/README.md),
[PyJetscape](../contribs/PyJetscape/README.md), [FastHydro](../contribs/FastHydro/README.md))
and the production READMEs
([`prod_AuAu_0_10`](../contribs/PyJetscape/example/prod_AuAu_0_10/README.md),
[`prod_AuAu_0_10_jet`](../contribs/PyJetscape/example/prod_AuAu_0_10_jet/README.md)).

## HDF5 writers and productions

| file | contents |
|---|---|
| [`PLAN_consolidate_h5_writer.md`](PLAN_consolidate_h5_writer.md) | one FNO4d HDF5 writer shared by `fast_data`, FastHydro and PyJetscape (decisions, phases 0–5) |
| [`PLAN_pair_h5_music.md`](PLAN_pair_h5_music.md) | background/jet MUSIC pairs → pair files (`pair_h5.py`, `prod_AuAu_0_10_jet`) |
| [`PLAN_particlize_h5.md`](PLAN_particlize_h5.md) | hadron level: stored surfaces and partons, `hadronize.py`, validation |
| [`PLAN_iSS_optim.md`](PLAN_iSS_optim.md) | faster iSS (Part A, bit-identical) and correlated jet/background sampling (Part B) |
| [`README_h5_optim.md`](README_h5_optim.md) | HDF5 compression of `arr` / `arr_bg`: Blosc-zstd default, `keep_bits` |

## Benchmarks

| file | contents |
|---|---|
| [`BENCHMARK_GB10.md`](BENCHMARK_GB10.md) | GB10 (CUDA): per-event profile, speed-ups, concurrent jobs, MPS, thread settings |
| [`BENCHMARK_M3MAX.md`](BENCHMARK_M3MAX.md) | Apple M3 Max (Metal): the same measurements, `run_jobs.sh` on macOS |

## Containers

| file | contents |
|---|---|
| [`PlanContainer.md`](PlanContainer.md) | one Docker image with X-SCAPE, music4gpu, js-contrib and FNO4d built in |
| [`PlanContainerDev.md`](PlanContainerDev.md) | source-free dev containers (`utils/Dockerfile.dev`, `utils/Dockerfile.dev.blackwell`) |

How to build and publish the dev images: [`utils/BuildContainerDev.md`](../utils/BuildContainerDev.md).

## Retired

Two finished plans were removed and are kept in the git history:
[`FastHydro/PLAN_hadronization.md`](https://github.com/jhputschke/js-contrib/blob/ece9d53/contribs/FastHydro/PLAN_hadronization.md)
(FastHydro soft particlization) and
[`Visualization/PlanVisualization.md`](https://github.com/jhputschke/js-contrib/blob/ece9d53/contribs/Visualization/PlanVisualization.md)
(PyVista visualizer).
