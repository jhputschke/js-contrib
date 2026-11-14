# Run events

Run X-SCAPE from Python and read each event's results while it is still in memory. The guide is
[The Pythonic interface](../pythonic.md); modules of your own are on
[Writing modules in Python](../modules.md).

## Examples

### One event, results in memory

`JobDir` gives the run a private working directory (MUSIC, 3dMCGlauber and iSS write side files
into the current directory) and points the XML's relative paths at the build. `per_event` runs
`Init()` on entry and `Finish()` on exit; inside the loop the event's data is still there.

```python
import numpy as np
import jetscape as js

with js.JobDir("out_first", "oo_workflow.xml") as job:          # the O+O example's XML
    with js.per_event(job.main_xml, job.user_xml, n_events=1) as driver:
        for i in driver.events():                                # after the event, before clearing
            partons = driver.energy_loss_manager.final_parton_array()   # numpy records
            jet = driver.hadronization_manager.hadron_array()          # HadronArray
            print(f"event {i}: {len(partons)} final partons,",
                  f"leading parton pT {np.hypot(partons['px'], partons['py']).max():.1f} GeV,",
                  f"{len(jet)} jet hadrons")
```

`final_parton_array()` has the fields `shower, pid, pstat, E, px, py, pz, t, x, y, z, mass, col,
acol`; `hadron_array()` holds `pid`, `pstat`, `p` (E, px, py, pz) and `x` (t, x, y, z) per hadron.
The other modules are driver properties too: `driver.hydro`, `driver.soft_particlization` (iSS),
`driver.initial_state`.

### Skip events that fail

A rare module error (Matter's "minus log of sudakov negative") would stop the loop. With
`on_error` the event is reported, cleared and skipped, as the productions do:

```python
skipped = []
for i in driver.events(on_error=lambda i, exc: skipped.append((i, str(exc)))):
    ...
```

### Run a whole XML without a Python loop

```python
from jetscape.run_jetscape import run_automatic

run_automatic(main_xml, user_xml)          # modules from the XML, the C++ event loop
```

### A module in Python

Subclass the C++ base class and register it; an XML can then name it, and `xml_parameters`
become XML parameters (from [Writing modules in Python](../modules.md)):

```python
import jetscape as js
from jetscape import pyjetscape_core as core

@js.module("CustomModuleToySplitV2", section="Eloss")
class ToySplitV2(core.JetEnergyLoss):
    """A parton above e_min splits into two halves at t >= t_split."""

    xml_parameters = {"t_split": 1.0, "e_min": 30.0}

    def DoEnergyLoss(self, deltaT, time, Q2, pIn, pOut):
        p = pIn[0]
        if time >= self.t_split and p.e() > self.e_min:
            E, px, py, pz = p.momentum
            return [js.make_parton(p.pid(), (E / 2, px / 2, s * 1.0, pz / 2),
                                   position=p.position) for s in (+1, -1)]
```

## Reference

::: jetscape.pythonic

::: jetscape.run_jetscape

::: jetscape.workdir

::: jetscape.paths

::: jetscape.copy_examples
