# js-contrib design notes and benchmarks

Plans, measurements and findings behind the contribs, the productions and the containers.

**Where things stand:** [`Production_readiness_summary.md`](Production_readiness_summary.md) summarizes the
changes in MUSIC4GPU, X-SCAPE and js-contrib for the jet-wake analysis and the FNO training data:
motivation, benchmarks, validation, physics tests, open points.
The user documentation is in the contrib READMEs ([`contribs/`](../contribs/README.md),
[PyJetscape](../contribs/PyJetscape/README.md), [FastHydro](../contribs/FastHydro/README.md))
and the production READMEs
([`prod_AuAu_0_10`](../contribs/PyJetscape/example/prod_AuAu_0_10/README.md),
[`prod_AuAu_0_10_jet`](../contribs/PyJetscape/example/prod_AuAu_0_10_jet/README.md)).

**Running a production with the containers** (Docker, Apptainer, SLURM; hadronization on
CPUs; analysis and ROOT export): [`README_2stage.md`](README_2stage.md).

**Analysing the files** (environments, reading hydro, particlize and hadron files, FastJet in
Python and in ROOT, the ROOT export, the notebooks): [`README_analysis.md`](README_analysis.md);
for the O+O production, [`README_OO.md`](README_OO.md). The scripts behind their tested code:
[`analysis_snippets/`](analysis_snippets/README.md); the Pb+Pb 5.02 TeV input files of
README_2stage §9: [`PbPb_5020/`](PbPb_5020/).

The plans (`PLAN_*.md`, `PlanContainer*.md`) are in [`Plans/`](Plans/); the user guide,
the HDF5 study and the benchmarks stay here.

## HDF5 writers and productions

| file | contents |
|---|---|
| [`PLAN_consolidate_h5_writer.md`](Plans/PLAN_consolidate_h5_writer.md) | one FNO4d HDF5 writer shared by `fast_data`, FastHydro and PyJetscape (decisions, phases 0–5) |
| [`PLAN_pair_h5_music.md`](Plans/PLAN_pair_h5_music.md) | background/jet MUSIC pairs → pair files (`pair_h5.py`, `prod_AuAu_0_10_jet`) |
| [`PLAN_particlize_h5.md`](Plans/PLAN_particlize_h5.md) | hadron level: stored surfaces and partons, `hadronize.py`, validation |
| [`PLAN_iSS_optim.md`](Plans/PLAN_iSS_optim.md) | faster iSS (Parts A and C, bit-identical) and correlated jet/background sampling (Part B) |
| [`Final_Optims.md`](Plans/Final_Optims.md) | what is left to speed up the O+O and Au+Au jet productions (2026-10-12): the first kernel profile at the production grid (`delta_qi` 52 %), less GPU work per event (late Δτ, active region, `delta_qi` restructure), host → GPU moves, and which side limits the rate on WSU Grid / ACCESS GPU nodes |
| [`PLAN_OO_root_reader.md`](Plans/PLAN_OO_root_reader.md) | O+O hadrons in `HadronFileReader` and `run_h5toROOT.py` (a `single_leg` layout, streaming, weights), a flat ROOT ntuple with the event record and weights per sample (not started) |
| [`PLAN_analysis_decays.md`](Plans/PLAN_analysis_decays.md) | π⁰ and weak decays of the hadron files at the analysis level (`jetscape.decays`, `run_h5toROOT.py --decays`; not started) |
| [`PLAN_smash_ready_hadrons.md`](Plans/PLAN_smash_ready_hadrons.md) | EOS 91 with iSS decaying the SMASH list, undecayed bulk and jet primaries for a later SMASH run, space-time positions for the jet hadrons, and SMASH as a separate CPU step on the primaries (Au+Au and O+O; not started) |
| [`PLAN_AuAu_hadrons_only.md`](Plans/PLAN_AuAu_hadrons_only.md) | Au+Au jet jobs that write hadron files only, in one job like O+O: `hadronize.py`'s per-unit work on the particlize writer's in-memory data, one iSS called on given cells (a small X-SCAPE wrapper method), little or no oversampling, primaries and partons kept for SMASH and a later jet hadronization (not started; after PLAN_smash_ready_hadrons) |
| [`PLAN_slim_bulk_info.md`](Plans/PLAN_slim_bulk_info.md) | the jet job's memory: three fixes and a 6-field background copy (`bulk_info`, `--bulk-info slim`): ~20 → 8.7 GB peak per job, bit-identical; `-j 4` in 29 GB instead of 66–69 (done) |
| [`README_h5_optim.md`](README_h5_optim.md) | HDF5 compression of `arr` / `arr_bg`: Blosc-zstd default, `keep_bits` |
| [`Wake_grid_comparison.md`](Wake_grid_comparison.md) | the jet's wake on the 0.3875 fm output grid vs. the old 0.3125 fm and MUSIC's 0.3 fm: integrated wake within 1–4%, shape at ≥ 0.5 fm within ~10% on every grid; the coarser grid lowers the peaks |
| [`SOURCE_NORMALIZATION.md`](SOURCE_NORMALIZATION.md) | the jet source in MUSIC4GPU: the liquefier kernel's continuum normalisation and why the production grid breaks it (0.004–6× per droplet), how the wake analysis found it, the per-droplet normalisation on MUSIC's grid (X-SCAPE #155), its validation, the late-step follow-up, and a proposed fine-grid test of the current source against the original |
| [`OPTIMIZATION_CHANGES.md`](OPTIMIZATION_CHANGES.md) | every optimisation and related fix since the GPU port, by module (MUSIC4GPU, X-SCAPE, iSS, js-contrib, 3dMCGlauber), split into bit-identical and not, with the validation behind each and the net events/h (2026-10-13) |
| [`MUSIC_CPU_vs_GPU.md`](MUSIC_CPU_vs_GPU.md) | MUSIC's CPU path (double) vs. MUSIC4GPU (float, vacuum at rest) on the same events, CUDA (GB10) and Metal (M3 Max): fields within 0.2%, identical freeze-out; the GPU gives a systematic 0.19% smaller freeze-out volume and 0.2% fewer hadrons, flow and wake unchanged; whether FP64 kernels would be worth it (an opt-in build, ~1 week) |

## Benchmarks

| file | contents |
|---|---|
| [`BENCHMARK_GB10.md`](BENCHMARK_GB10.md) | GB10 (CUDA): per-event profile, speed-ups, concurrent jobs, MPS, thread settings |
| [`BENCHMARK_M3MAX.md`](BENCHMARK_M3MAX.md) | Apple M3 Max (Metal): the same measurements, `run_jobs.sh` on macOS |
| [`BENCHMARK_2x3090.md`](BENCHMARK_2x3090.md) | 2 × RTX 3090 (Docker): jobs per GPU, threads, MPS, memory, the campaign default, `final_optim` |
| [`REPORT_FINAL_OPTIM.md`](REPORT_FINAL_OPTIM.md) | `cu124-final_optim` on 2 × RTX 3090: throughput, and the physics check of C4 and the late time step (D1) |
| [`REPORT_SUMMARY_CLUSTER.md`](REPORT_SUMMARY_CLUSTER.md) | throughput per machine, and the per-GPU settings extrapolated to cluster GPUs (V100, A40, A100, L40S) |
| [`REPORT_CLUSTER_ACCESS.md`](REPORT_CLUSTER_ACCESS.md) | which NSF ACCESS GPU resource to request: ACCESS exchange rates, estimated events/h and events per credit per resource, the DeltaAI (first) and Delta A40 (second) settings, request sizing, scratch space and the upload to OSDF; Lambda 8 × A100 and A10 for comparison (2026-10-13) |

## Containers

| file | contents |
|---|---|
| [`README_2stage.md`](README_2stage.md) | **user guide**: the two-stage production with the production images, from GPU jobs to hadrons and ROOT files; image status, launch settings, per-GPU sizing, the late time step |
| [`README_analysis.md`](README_analysis.md) | **user guide**: the analysis of the production files: environments, threads, HDF5 readers for hydro and hadrons, FastJet (Python and C++), ROOT conversion and macros, notebooks |
| [`README_OO.md`](README_OO.md) | **user guide**: the analysis of the O+O ISR production: the event, the per-event record, cross-section and Ncoll weights, checks, multiplicity, centrality, flow from the oversamples, jets, ROOT |
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
