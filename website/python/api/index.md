# Python API

Every class and function of PyJetscape with its signature and documentation, generated from the
source: the docstrings of the `jetscape` package and, for the compiled C++ bindings, the type
stubs `pyjetscape_core.pyi`. The pages follow the order of the rest of this site: run events,
write the files, analyse them. Each page starts with short examples of its most used calls.

| page | what you do with it | modules |
|---|---|---|
| [Run events](run.md) | run X-SCAPE from Python, one event at a time; a working directory per job; modules in Python | `pythonic`, `run_jetscape`, `workdir`, `paths`, `copy_examples` |
| [Write the files](write.md) | store hadrons from your own runs; the writers behind the production files | `hadrons_h5` (writer), `particlize_h5` (writer), `hadronizer`, `pair_h5`, `surface_replay`, `liquefier_io` |
| [Analyse the files](analyse.md) | read a campaign: histograms with per-event errors, the wake, single events, jets, surfaces, the hydro pair | `hadrons_h5` (readers), `particlize_h5` (reader), `hadron_array`, `awkward_views`, `records`, `particles`, `showers` |
| [ROOT, bulk and FNO files](bulk.md) | the hydro evolution as ROOT or HDF5 for machine learning, and the PyTorch hydro module | `fast_root_bulk`, `bulk_root_writer`, `fast_h5_bulk`, `bulk_sources`, `fno_h5_writer`, `h5_compression`, `repad_h5`, `fno_hydro`, `utils` |
| [C++ bindings](core.md) | X-SCAPE's classes with their C++ names, and what the Pythonic layer adds to them | `pyjetscape_core` |

## Reading these pages

- **Names.** The package is imported as `import jetscape as js`; the bound C++ classes are in
  `jetscape.pyjetscape_core` (`from jetscape import pyjetscape_core as core`). `import jetscape`
  also re-exports the most used names at the top level (`js.JetScape`, `js.HadronArray`, ...).
- **What needs a build.** Running events needs the compiled extension (container, Mac wheel or a
  source build). Reading files needs only the [analysis environment](../../install/analysis-env.md).
- **The Pythonic layer on the C++ classes.** `import jetscape` adds methods and properties to the
  bound classes (`final_parton_array()`, `hadron_array()`, ...). They are listed on the
  [C++ bindings](core.md) page under the class they belong to.
- **Build flags.** The stubs follow the build they were made from. Names behind a build option
  (ROOT, the iColored colour functions, iSS primaries) are listed only when that build had them.
- **No docstring.** A C++ method without one is listed with its signature only; its C++
  namesake in X-SCAPE behaves the same.

The guides explain the ideas behind these calls: [The Pythonic interface](../pythonic.md),
[Writing modules in Python](../modules.md), [What a production writes](../../analysis/files.md)
and [Hadrons](../../analysis/hadrons.md).
