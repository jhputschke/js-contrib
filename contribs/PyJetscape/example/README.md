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
```

## Productions

| folder | what | docs |
|---|---|---|
| [`prod_AuAu_0_10/`](prod_AuAu_0_10/README.md) | 0–10% Au+Au 200 GeV hydro evolutions (3D MC-Glauber, MUSIC on the GPU), written straight to FNO4d training HDF5 on an output grid chosen in a YAML file; `run_prod.py`, `run_jobs.sh` | [README](prod_AuAu_0_10/README.md) |
| [`prod_AuAu_0_10_jet/`](prod_AuAu_0_10_jet/README.md) | the same medium with a jet: the two-stage hydro writes background/jet **pairs** (`run_prod_jet.py`, `run_jobs.sh`, optional pT̂ windows). With `--write-particlize` it also stores both freeze-out surfaces and the partons, which `hadronize.py` / `run_hadronize.py` turn into hadrons later, on CPUs. `run_h5toROOT.py` converts a hadronized campaign to ROOT ([`root_export/`](prod_AuAu_0_10_jet/root_export/README.md)) | [README](prod_AuAu_0_10_jet/README.md); in the containers: [`docs/README_2stage.md`](../../../docs/README_2stage.md) |

## Analyses

| folder | what | needs |
|---|---|---|
| [`analysis/`](analysis/README.md) | Python analyses of a jet production: the jet energy balance and the fix for the LBT/liquefier double counting; wake observables at parton/hydro and at hadron level (`wake_observables.py`, `wake_hadrons.py` + notebooks); hadron η, φ, pT distributions; jets with FastJet | the venv |
| [`analysis_root/`](analysis_root/README.md) | ROOT analyses of the files of `run_h5toROOT.py`: `HadronFileReader.h` (an event's background, background + deposition and whole jet event as `std::vector<Hadron>`), `read_hadrons.C`, `hadron_distributions.C` | ROOT ≥ 6.34 |
| [`hydro_hist_vs_surface/`](hydro_hist_vs_surface/README.md) | a study: hadrons from a freeze-out surface found in the saved hydro history (what an FNO prediction can be hadronized with), against iSS on MUSIC's own surface with viscous corrections | X-SCAPE build with iSS |

## Single-feature scripts

| script | shows | docs |
|---|---|---|
| [`per_event_loop.py`](per_event_loop.py) | the event loop in Python (Mode C, `JetScapePerEvent`): each event's modules can be inspected before they are cleared; XML task list or a manual pipeline | [PyJetscape README, *Example Script*](../README.md#example-script) |
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
