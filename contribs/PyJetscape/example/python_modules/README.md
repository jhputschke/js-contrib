# python_modules — JETSCAPE modules written in Python

One toy module per stage in `toy_modules.py`, as templates: each subclasses the C++ base
class and overrides what JETSCAPE calls, with the C++ names and argument order.

| module | base class | overrides | C++ counterpart |
|---|---|---|---|
| `ToyGun` | `HardProcess` | `InitTask`, `ExecuteTask` (`AddParton`), `GetSigmaGen` | PGun, PythiaGun |
| `ToySplit` | `JetEnergyLoss` | `DoEnergyLoss(deltaT, time, Q2, pIn, pOut)` | Matter, LBT |
| `ToyHadronizer` | `Hadronization` | `DoHadronization(shower, hOut, pOut)` | ColorlessHadronization |
| `ToySampler` | `SoftParticlization` | `ExecuteTask` (`set_hadron_list`) | iSS |
| `ToyAfterburner` | `Afterburner` | `ExecuteTask`, `GetCurrentHadronList` | SMASH |
| `ToyWriter` | `JetScapeWriter` | `GetStatus`, `Write(obj)`, `WriteEvent` | JetScapeWriterAscii |

The same pipeline two ways, each with the C++ ASCII writer next to the Python one and the
file read back with `JetScapeReaderAscii`:

```bash
conda activate js_fno
python run_from_xml.py --outdir /tmp/toy   # the XML names them: <Eloss><CustomModuleToySplit>, ...
python run_by_hand.py --outdir /tmp/toy    # built in Python (Mode B), automatic task list off
```

Things to know:

- **Names for the XML.** `register_module(name, cls, section=, defaults=)` puts a class into
  JETSCAPE's module factory. Every XML section takes a module whose element name contains
  `CustomModule`; `<JetHadronization>` takes it as `<name>CustomModuleX</name>`.
  `JetScape.Init()` adds the default entry (`<section><name>defaults</name>`) to the main
  XML in memory, as JetScape refuses user tags the main XML lacks; a module reads its
  parameters with `get_xml_element_*([section, name, ...])`.
- **Copies.** The energy-loss manager copies its modules once per shower with `Clone()`:
  the C++ copy constructor plus a shallow copy of the attributes (override `Clone` for
  more). Task numbers, and so the random seeds, follow a C++ run.
- **Random numbers.** `self.GetMt19937Generator()` is the task's engine, as for a C++
  module (`.uniform()`, or seed numpy from one draw).
- **Lists.** The input partons of `DoEnergyLoss` are the C++ objects (valid during the call);
  the output lists are copied back after the call.
- **Cost.** A Python energy-loss module is called per parton per time step.

The full API and the design: `docs/Plans/PLAN_pyjetscape_complete.md` in js-contrib.
