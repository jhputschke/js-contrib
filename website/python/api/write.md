# Write the files

Store the hadrons of your own runs in the production's file format, and the writers behind the
production files. Which file holds what is on
[What a production writes](../../analysis/files.md).

## Examples

### Hadrons of your own run

`HadronH5Writer` writes one unit per event; the files it makes are read with everything on
[Analyse the files](analyse.md). Give it an absolute path: `JobDir` changes the directory.

```python
import os
import jetscape as js
from jetscape.hadrons_h5 import HadronH5Writer
from jetscape.records import initiator_rows

out = os.path.abspath("my_hadrons_jet_frag.h5")     # absolute: JobDir changes directory
with js.JobDir("out", "oo_workflow.xml") as job, \
     js.per_event(job.main_xml, job.user_xml, n_events=2) as driver, \
     HadronH5Writer(out, tag="jet_frag", n_samples=1, initiators=True) as w:
    had, mgr = driver.hadronization_manager, driver.energy_loss_manager
    n = 0
    for i in driver.events():
        w.append_unit(had.hadron_array().to_dict(), unit=n, event=i,
                      initiators=initiator_rows(mgr.initiator_array()),
                      seed=had.last_random_seed)
        n += 1
```

- **The bulk hadrons** of iSS go into a second writer with `tag="bulk_bg"` and
  `n_samples=` the number of oversamples: call `driver.soft_particlization.use_compact_output()`
  before the loop and append `driver.soft_particlization.hadron_array().to_dict()` per event.
  The complete version is `pythonic/oo_workflow.py` in the examples.
- **Extra keywords** of `append_unit` (`seed=`, `pthat=`, `n_cells=`, ...) are stored per unit,
  in the file's `units/` table.

### What the productions write

The drivers use the other writers on this page themselves:

| file | written by | class |
|---|---|---|
| `<stem>_particlize.h5` | `run_prod_jet.py --write-particlize` | `ParticlizeH5Writer` |
| `<stem>_hadrons_<tag>.h5` | `hadronize.py` (per unit: `Hadronizer`) | `HadronH5Writer` |
| `<stem>.h5`, the hydro pair | `run_prod_jet.py` | `PairH5Writer` |

`SurfaceReplay` is a hydro module that hands a stored freeze-out surface to iSS inside a
JETSCAPE run, so iSS samples it as it would the live hydro's. `liquefier_io` reads the jet's
droplets out of the liquefier during a run.

## Reference

::: jetscape.hadrons_h5.HadronH5Writer

::: jetscape.hadrons_h5.add_initiators

::: jetscape.hadrons_h5.particlize_initiators

::: jetscape.hadrons_h5.pair_initiators

::: jetscape.hadrons_h5.z_record

::: jetscape.hadrons_h5.hadronization_input_rows

::: jetscape.hadrons_h5.hadronization_input_dict

::: jetscape.particlize_h5.ParticlizeH5Writer

::: jetscape.particlize_h5.add_initiators_from_pair

::: jetscape.hadronizer

::: jetscape.pair_h5

::: jetscape.surface_replay

::: jetscape.liquefier_io
