# PyJetscape examples — productions, analyses, single-feature scripts

Everything here uses PyJetscape, the Python bindings of X-SCAPE
([`../README.md`](../README.md)). The folders are complete workflows: productions of hydro
data for FNO4d training, hadronization, and the analyses of their outputs. The scripts at
the top each show one feature of the bindings.

**What a part needs.**
- **X-SCAPE build:** the productions and the scripts run X-SCAPE. They need its build with
  PyJetscape (`pyjetscape_core`), usually in the `js_fno` conda env, or the production
  container.
- **Python venv:** the analyses only read files, in the venv of
  [`utils/analysis_env`](../../../utils/analysis_env/README.md). No X-SCAPE build is needed.
- **ROOT:** `analysis_root` needs ROOT itself.

The guide to all of the analysis side (environments, readers, FastJet in Python and C++, the
ROOT export, the notebooks) is [`docs/README_analysis.md`](../../../docs/README_analysis.md).

```
 prod_AuAu_0_10            hydro only        → FNO4d HDF5 (MUSIC evolutions)
 prod_AuAu_0_10_jet        hydro + jet       → pair files (background + jet leg)
   │  --write-particlize                     → particlize files (surfaces, partons)
   ├─ hadronize.py                           → hadron files (HDF5)
   │    ├─ analysis/         Python          → balance checks, wake observables, spectra, jets
   │    ├─ hydro_hist_vs_surface/            → hadrons from the saved history vs MUSIC's surface
   │    └─ run_h5toROOT.py                   → ROOT files
   │         └─ analysis_root/  C++/ROOT     → HadronFileReader.h, distributions
   └─ moved between machines with utils/remote_transfer (GCS, OSDF)
 prod_OO_isr               O+O, ISR + jet    → hydro file, jet hadrons (in the job)
   │  --bulk-hadrons N                       → bulk hadrons in the job too
   │  (default: particlize file)             → hadronize.py --tags bulk_bg → bulk hadrons
```

## Productions

| folder | what | docs |
|---|---|---|
| [`prod_AuAu_0_10/`](prod_AuAu_0_10/README.md) | 0–10% Au+Au 200 GeV hydro evolutions (3D MC-Glauber, MUSIC on the GPU), written straight to FNO4d training HDF5 on an output grid chosen in a YAML file; `run_prod.py`, `run_jobs.sh` | [README](prod_AuAu_0_10/README.md) |
| [`prod_AuAu_0_10_jet/`](prod_AuAu_0_10_jet/README.md) | the same medium with a jet: the two-stage hydro writes background/jet **pairs** (`run_prod_jet.py`, `run_jobs.sh`, optional pT̂ windows). With `--write-particlize` it also stores both freeze-out surfaces and the partons, which `hadronize.py` / `run_hadronize.py` turn into hadrons later, on CPUs. `run_h5toROOT.py` converts a hadronized campaign to ROOT ([`root_export/`](prod_AuAu_0_10_jet/root_export/README.md)) | [README](prod_AuAu_0_10_jet/README.md); in the containers: [`docs/README_2stage.md`](../../../docs/README_2stage.md) |
| [`prod_OO_isr/`](prod_OO_isr/README.md) | O+O 200 GeV with initial-state radiation (PythiaIsrGun + iMATTER, strings from what the hard scattering left) and Matter in one MUSIC medium per event, tuned O+O 3D MC-Glauber/MUSIC parameters: the hydro evolution, the jet hadrons of iColoredHadronization (always in the job), and the bulk hadrons either in the job (`--bulk-hadrons N`, with `--no-particlize` nothing is left to hadronize) or later from the stored surfaces (`hadronize.py --tags bulk_bg`); `run_prod_oo_isr.py`, `run_jobs.sh`. Needs X-SCAPE's `<Hard><ISR>` (branch `oo_isr_test`) | [README](prod_OO_isr/README.md) |

## Analyses

| folder | what | needs |
|---|---|---|
| [`analysis/`](analysis/README.md) | Python analyses of a jet production: the jet energy balance and the fix for the LBT/liquefier double counting; wake observables at parton/hydro and at hadron level (`wake_observables.py`, `wake_hadrons.py` + notebooks); hadron η, φ, pT distributions; jets with FastJet. The Z+jet campaigns, step by step: [`ANA_Zhadron.md`](analysis/zjet/ANA_Zhadron.md) (Z–hadron wake, CMS 2507.09307, event mixing) and [`ANA_Zjet.md`](analysis/zjet/ANA_Zjet.md) (Z+jet, CMS 1702.01060), each in Python and in ROOT | the venv |
| [`analysis_root/`](analysis_root/README.md) | ROOT analyses of the files of `run_h5toROOT.py`: `HadronFileReader.h` (an event's background, background + deposition and whole jet event as `std::vector<Hadron>`), `read_hadrons.C`, `hadron_distributions.C` | ROOT ≥ 6.34 |
| [`hydro_hist_vs_surface/`](hydro_hist_vs_surface/README.md) | a study: hadrons from a freeze-out surface found in the saved hydro history (what an FNO prediction can be hadronized with), against iSS on MUSIC's own surface with viscous corrections | X-SCAPE build with iSS |

**Multi-GPU output (`OUTBASE/gpu0/`, `OUTBASE/gpu1/`, …; `utils/launch_2gpu.sh`).** The readers
look only in the directory they are given, not below it (2026-10-16). Pass the subdirectories:
the Python `HadronFileReader` a glob, `'OUTBASE/gpu*/*_hadrons_jet_frag.h5'` (or
`*_particlize.h5`); `run_h5toROOT.py` the directories themselves, `OUTBASE/gpu0 OUTBASE/gpu1`,
with `--out-dir` so that the ROOT files end up in one directory, which the C++
`HadronFileReader.h` reads. The two readers number events in file order: Python's glob sorts by
path (`gpu0/` first), the C++ reader a flat directory by file name. With the launchers' fixed
seeds (GPU0 the first block, GPU1 the next; zero-padded in the names) the orders agree; with
`FIRST_SEED 0` (random seeds) they can differ, which changes no physics but the global event
numbers, and with them the jackknife blocks and the strict-mixing draws of the Z analyses. More
in [`analysis/README.md`](analysis/README.md) and [`analysis_root/README.md`](analysis_root/README.md).

## Notebooks

Every notebook sits next to the code or the production it belongs to; this is the index.
Inputs come from environment variables (defaults in each notebook's first code cell); "stored"
says whether the notebook is committed with its outputs.

| notebook | what it shows | input | needs | stored |
|---|---|---|---|---|
| **Productions** | | | | |
| [`prod_AuAu_0_10/check_output.ipynb`](prod_AuAu_0_10/check_output.ipynb) | checks the hydro files of `run_prod.py`: summary, sanity scan (finite, ε ≥ 0, \|v\| < 1), the frames past freeze-out | the folder's `out/*.h5` | h5py, pandas | no |
| [`prod_AuAu_0_10_jet/jet_wake.ipynb`](prod_AuAu_0_10_jet/jet_wake.ipynb) | Δe = jet leg − background of one pair in MUSIC, the shower and its droplets; §11 is where the jet energy balance was first checked | a pair file and its hadron files | `jetscape`, scipy (`MUSIC_EOS_TABLE`) | yes |
| [`prod_OO_isr/oo_isr_overview.ipynb`](prod_OO_isr/oo_isr_overview.ipynb) | one O+O ISR job: the medium at three proper times with the hard scattering, and the hadrons | `OO_STEM`, `XSCAPE_BUILD` | h5py | yes |
| [`prod_OO_isr/hard_sub_compare/hard_sub_compare.ipynb`](prod_OO_isr/hard_sub_compare/hard_sub_compare.ipynb) | two O+O campaigns that differ only in 3dMCGlauber's `Subtract_hard_momentum` | `HSC_DIR` (`run_compare.sh`'s output) | h5py | yes |
| **The Pythonic interface** ([`PYTHONIC.md`](../PYTHONIC.md)) | | | | |
| [`pythonic/oo_workflow.ipynb`](pythonic/oo_workflow.ipynb) | 3 O+O events through the full chain (3dMCGlauber, ISR, MUSIC, Matter + LBT, iColored, iSS) with `js.per_event`; the hadron files written from `HadronArray`s and read back; awkward views, PDG selections; FastJet on every event and every oversample in one call, a jet embedded in the bulk | the user XML `pythonic/oo_workflow.xml`; runs X-SCAPE (`JETSCAPE_BUILD`, default `build_gpu`); writes `pythonic/out_oo/` (git-ignored) | X-SCAPE build with 3dMCGlauber, music4gpu, iSS, iColored closure; awkward, vector, fastjet, particle, pandas | yes (the modules' log removed) |
| **Analyses** | | | | |
| [`analysis/wake_observables.ipynb`](analysis/wake_observables.ipynb) | jet energy loss and the wake at parton and hydro level | `WAKE_OBS`: `wake_observables.h5` of `wake_observables.py` | pandas | yes |
| [`analysis/wake_hadrons.ipynb`](analysis/wake_hadrons.ipynb) | the same at hadron level | `WAKE_HAD`: `wake_hadrons.h5` of `wake_hadrons.py` (+ `WAKE_OBS`) | pandas | yes |
| [`analysis/zjet/zjet_wake.ipynb`](analysis/zjet/zjet_wake.ipynb) | the Z–hadron wake (paired truth) in Δφ and Δη, Pb+Pb − pp, one event's hydro wake and movie frame | `ZJET_HADRONS`, `ZJET_VACUUM`, `ZJET_PAIR`, `ZJET_MOVIE`; test data in `analysis/runs/zjet_test/` ([`zjet/test/README.md`](analysis/zjet/test/README.md)) | `jetscape`, `zjet_wake.py` | no |
| [`analysis_root/read_hadrons.ipynb`](analysis_root/read_hadrons.ipynb) | `read_hadrons.C` one step per cell: the C++ reader by example, with the histograms drawn | a campaign converted by `run_h5toROOT.py` | the ROOT C++ kernel, `HadronFileReader.h` | yes |
| [`hydro_hist_vs_surface/hydro_hist_vs_surface.ipynb`](hydro_hist_vs_surface/hydro_hist_vs_surface.ipynb) | hadrons from a surface found in the saved hydro history (what an FNO prediction can give) against MUSIC's full surface | the study's outputs ([README](hydro_hist_vs_surface/README.md)) | numpy, matplotlib | yes |
| **Tutorials** (reading the hadron files) | | | | |
| [`analysis/tutorials/hadron_distributions.ipynb`](analysis/tutorials/hadron_distributions.ipynb) | η, φ, p_T of background, background + wake and jet hadrons; the correlated-sampling flags checked | `HADRON_DIR`: hadron files of `hadronize.py` or `--hadrons-only` | `jetscape` | yes |
| [`analysis/tutorials/jet_fastjet.ipynb`](analysis/tutorials/jet_fastjet.ipynb) | FastJet's classic interface: jet spectrum, fragmentation function, jet areas, ρ subtraction | `HADRON_DIR` | `fastjet` | yes |
| [`analysis/tutorials/jet_fastjet_awkward.ipynb`](analysis/tutorials/jet_fastjet_awkward.ipynb) | the same with the array interface (awkward), every event clustered in one call; §6 substructure | `HADRON_DIR` | `fastjet`, awkward, vector | yes |

FastHydro's notebooks (`jet_wake`, `hadron_wake`, `bulk_vs_music`) are in
[`../../FastHydro/notebooks/`](../../FastHydro/notebooks/).

## Single-feature scripts

| script | shows | docs |
|---|---|---|
| [`per_event_loop.py`](per_event_loop.py) | the event loop in Python (Mode C, `JetScapePerEvent`): each event's modules can be inspected before they are cleared; XML task list or a manual pipeline | [PyJetscape README, *Example Scripts*](../README.md#example-scripts) |
| [`python_bulk_h5_writer.py`](python_bulk_h5_writer.py) | `H5BulkWriter`: the hydro evolution straight to FNO4d HDF5, no ROOT; native, grid and framework modes | [*Python HDF5 Bulk Writer*](../README.md#python-hdf5-bulk-writer-fast_h5_bulkpy) |
| [`validate_h5_vs_root.py`](validate_h5_vs_root.py) | the C++ `FastRootBulkWriter` and `H5BulkWriter` on the same events, compared (bit for bit in native mode) | [*Python HDF5 Bulk Writer*, Examples and tests](../README.md#examples-and-tests) |
| [`python_fast_bulk_root_writer.py`](python_fast_bulk_root_writer.py) | the C++ `FastRootBulkWriter` from Python, the file read back; `--check-numpy` compares MUSIC's native store in numpy with it | [*C++ `FastRootBulkWriter` from Python*](../README.md#c-fastrootbulkwriter-from-python) |
| [`python_bulk_root_writer.py`](python_bulk_root_writer.py) | `PyBulkRootWriter`, a Python module writing the bulk evolution to ROOT with uproot | [*Python ROOT Bulk Writer*](../README.md#python-root-bulk-writer-bulk_root_writerpy) |
| [`inspect_bulk_info.py`](inspect_bulk_info.py) | the evolution history of a run as numpy arrays and PyTorch tensors, with plots of e, T and v slices | the script |
| [`python_fno_test.py`](python_fno_test.py) | `PyFNOHydro`: an FNO in place of the hydro module (TRENTo → free streaming → FNO → MATTER → hadronization) | [*`PyFNOHydro`*](../README.md#pyfnohydro--python-fno-hydro-module) |

**Running them.** The scripts run from the X-SCAPE build directory, with PyJetscape
importable (`PYTHONPATH` or the editable install, [PyJetscape README,
*Installation*](../README.md#installation)). The XML paths are those of the X-SCAPE
`config/` folder, e.g.:

```bash
cd X-SCAPE/build_gpu
python ../external_packages/js-contrib/contribs/PyJetscape/example/python_bulk_h5_writer.py \
  --user ../config/BulkFastTest/OO_one_event_fast.xml --out hydro_evo.h5
```

Their XML defaults are X-SCAPE's `config/` and FnoHydro's `config/`, found from where
js-contrib sits (`X-SCAPE/external_packages/js-contrib`). `python_fno_test.py` also needs
an FNO model in `contribs/FnoHydro/models/`; the `.pt` files aren't in git, see that
folder's README.

## Tests

The tests are in [`../tests`](../tests/):
- the hydro HDF5 writer (`test_h5_bulk.py`) and the pair files (`test_pair_h5.py`);
- particlize and hadron files and hadronization (`test_particlize_h5.py`);
- seeds and pT̂ windows of the productions (`test_prod_seeds.py`, `test_pthat_bins.py`);
- the ROOT export (`test_hadrons_to_root.py`, `test_run_h5toROOT.py`) and
  `HadronFileReader.h` (`test_analysis_root_reader.py`, needs PyROOT);
- the wake observables (`test_wake_observables.py`);
- the surface finder and the history surfaces (`test_surface_finder.py`,
  `test_hist_surface.py`).

```bash
pytest external_packages/js-contrib/contribs/PyJetscape/tests -q
```
