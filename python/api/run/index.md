# Run events

Run X-SCAPE from Python and read each event's results while it is still in memory. The guide is [The Pythonic interface](https://jhputschke.github.io/js-contrib/python/pythonic/index.md); modules of your own are on [Writing modules in Python](https://jhputschke.github.io/js-contrib/python/modules/index.md).

## Examples

### One event, results in memory

`JobDir` gives the run a private working directory (MUSIC, 3dMCGlauber and iSS write side files into the current directory) and points the XML's relative paths at the build. `per_event` runs `Init()` on entry and `Finish()` on exit; inside the loop the event's data is still there.

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

`final_parton_array()` has the fields `shower, pid, pstat, E, px, py, pz, t, x, y, z, mass, col, acol`; `hadron_array()` holds `pid`, `pstat`, `p` (E, px, py, pz) and `x` (t, x, y, z) per hadron. The other modules are driver properties too: `driver.hydro`, `driver.soft_particlization` (iSS), `driver.initial_state`.

### Skip events that fail

A rare module error (Matter's "minus log of sudakov negative") would stop the loop. With `on_error` the event is reported, cleared and skipped, as the productions do:

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

Subclass the C++ base class and register it; an XML can then name it, and `xml_parameters` become XML parameters (from [Writing modules in Python](https://jhputschke.github.io/js-contrib/python/modules/index.md)):

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

### `jetscape.pythonic`

python/jetscape/pythonic.py

The Pythonic layer over the bound classes (docs/Plans/PLAN_pyjetscape_pythonic.md). Additive: install() puts new methods and properties onto the classes of pyjetscape_core and refuses any name a class already has, so no bound method can be shadowed and every existing script keeps working. jetscape/**init** calls install() once; importing jetscape.pyjetscape_core directly runs jetscape/**init** first, so the layer is always there.

Naming: a method ending in \_array() returns numpy data, filled once per event; a plain noun (showers, nodes, final_partons()) returns objects. Momentum (E, px, py, pz), position (t, x, y, z): the order of FourVector.comp(i).

Properties on the bound classes are non-data descriptors (\_Prop): an instance attribute of the same name, as a Python module may set (self.rng = ...), still wins.

#### `ShowerNode(shower, node_id: int)`

A vertex of a PartonShower by id, with its position, the partons in and out and the split variables (PartonShower's id-based methods).

##### `position: 'core.FourVector'`

FourVector (iterates as t, x, y, z).

##### `parents: list`

The partons that end here.

##### `children: list`

The partons that start here.

#### `installed_names() -> list[str]`

'Class.name' for everything the layer installed (for docs and tests).

#### `make_parton(pid: int, momentum, position=(0.0, 0.0, 0.0, 0.0), stat: int = 0, label: int = 0, *, color: int = 0, anti_color: int = 0, cls=None) -> 'core.Parton'`

A Parton by keywords: momentum (E, px, py, pz) and position (t, x, y, z) as any 4-sequence or a FourVector; colour tags optional. cls=Photon for a photon.

#### `make_photon(momentum, position=(0.0, 0.0, 0.0, 0.0), stat: int = 0, label: int = 0) -> 'core.Photon'`

A Photon (pid 22) by keywords, as make_parton.

#### `make_hadron(pid: int, momentum, position=(0.0, 0.0, 0.0, 0.0), stat: int = 0, label: int = 0, mass: float | None = None) -> 'core.Hadron'`

A Hadron by keywords: momentum (E, px, py, pz), position (t, x, y, z) as any 4-sequence or a FourVector; without a mass the PDG mass is used (C++ Hadron).

#### `module(name: str, section: str = '') -> 'Callable[[type], type]'`

Class decorator: register a Python module under `name` (register_module), with a main-XML default built from the class's `xml_parameters` ({key: default}, int, float, str or bool). The class's InitTask first sets self. from [section, name, key] in the XML; the defaults are also class attributes, so a module built by hand has them before Init. Use a CustomModule... name for modules the XML names ().

#### `per_event(main_xml: str | None = None, user_xml: str | None = None, tasks: Sequence | None = None, n_events: int | None = None) -> 'core.JetScapePerEvent'`

A JetScapePerEvent driver by keywords: the XML files, a by-hand pipeline (tasks added in order; else the XML's task list) and the number of events.

```text
with js.per_event(main_xml, user_xml) as driver:
    for _ in driver.events():
        fp = driver.energy_loss_manager.final_parton_array()
```

The bound constructor JetScapePerEvent() stays as it is.

#### `install() -> None`

Put the layer onto the bound classes and the core module (once).

### `jetscape.run_jetscape`

python/jetscape/run_jetscape.py

High-level simulation driver helpers.

Two modes are supported — matching the two values of in the JETSCAPE XML:

Mode A — automatic task list (enableAutomaticTaskListDetermination = true) run_automatic(main_xml, user_xml) All modules are instantiated from the XML by the C++ factory. No Python module injection is possible in this mode.

Mode B — manual task list (enableAutomaticTaskListDetermination = false) run_manual(main_xml, user_xml, modules) Modules are supplied explicitly as a Python list. Python trampoline modules (e.g. PyFNOHydro) are supported. The helper validates that the supplied list is in a legal pipeline order before calling js.Add().

Mode C — per-event external loop (either XML configuration) per_event_loop(main_xml, user_xml, ...) # generator, yields once per event run_per_event(main_xml, user_xml, on_event, ...) # callback per event Uses JetScapePerEvent so the event loop runs in Python. The per-event module results are accessible (e.g. via JetScapeSignalManager) *after* the event executes and *before* its memory is released. Works with the automatic (modules=None) or manual (modules=[...]) task list.

Usage example (Mode C): from jetscape import JetScapeSignalManager from jetscape.run_jetscape import per_event_loop

```text
for js in per_event_loop("config/jetscape_main.xml",
                         "config/jetscape_user.xml"):
    hydro = JetScapeSignalManager.Instance().GetHydroPointer()
    if hydro is not None:
        info = hydro.get_bulk_info()
```

Usage example (Mode A): from python.jetscape.run_jetscape import run_automatic js = run_automatic("config/jetscape_main.xml", "config/jetscape_user_MUSIC.xml")

Usage example (Mode B): from python.jetscape import create_module from python.jetscape.fno_hydro import PyFNOHydro from python.jetscape.run_jetscape import run_manual

```text
ini      = create_module("TrentoInitial")
preeq    = create_module("FreestreamMilne")
fno      = PyFNOHydro("models/traced.pt", config)
# JetEnergyLossManager / JetEnergyLoss are not in the module factory: for jet
# energy loss construct pyjetscape_core.JetEnergyLossManager() and JetEnergyLoss()
# directly and Add() a create_module("Matter") to them.

js = run_manual(
    "config/jetscape_main.xml",
    "jetscape_user_fno.xml",           # a user XML with a <Hydro><FNO> block
    [ini, preeq, fno],
)   # runs Init(), Exec() and Finish()
```

#### `run_automatic(main_xml: str, user_xml: str) -> JetScape`

Run a fully XML-driven JETSCAPE pipeline.

The XML must contain::

```text
<enableAutomaticTaskListDetermination>true</enableAutomaticTaskListDetermination>
```

All modules are instantiated from the XML by the C++ factory. No Python module injection is possible in this mode.

Parameters:

- **`main_xml`** (`str`) – Path to the main (default parameters) XML file.
- **`user_xml`** (`str`) – Path to the user (overrides) XML file.

Returns:

- `JetScape` – The framework object after Finish() has been called.

#### `run_manual(main_xml: str, user_xml: str, modules: Sequence, n_events: int | None = None) -> JetScape`

Run a JETSCAPE pipeline with an explicit module list.

The XML must contain::

```text
<enableAutomaticTaskListDetermination>false</enableAutomaticTaskListDetermination>
```

Modules are added in the order supplied. Python trampoline modules (e.g. PyFNOHydro) are supported alongside C++ modules obtained via create_module().

The helper validates that the supplied top-level modules are in a legal pipeline stage order before handing them to JetScape.Add().

Parameters:

- **`main_xml`** (`str`) – Path to the main (default parameters) XML file.
- **`user_xml`** (`str`) – Path to the user (overrides) XML file.
- **`modules`** (`sequence of JetScapeTask`) – Top-level modules added to JetScape in pipeline order, e.g.:: [ini, preeq, fno, jloss_manager, hadro_manager] Note: sub-modules (JetEnergyLoss, Hadronization, etc.) must already be nested inside their managers via manager.Add() before calling run_manual().
- **`n_events`** (`int`, default: `None` ) – Override the number of events from the XML.

Returns:

- `JetScape` – The framework object after Finish() has been called.

Raises:

- `ValueError` – If the module list is in an illegal pipeline order.

#### `per_event_loop(main_xml: str, user_xml: str, modules: Sequence | None = None, n_events: int | None = None, start_event: int = 0) -> Iterator[JetScapePerEvent]`

Iterate over a JETSCAPE run one event at a time.

Yields the JetScapePerEvent driver once per event, *after* the event has been executed but *before* its memory is released, so the caller can read the live module results for that event::

```text
from jetscape import JetScapeSignalManager
from jetscape.run_jetscape import per_event_loop

for js in per_event_loop(main_xml, user_xml):
    hydro = JetScapeSignalManager.Instance().GetHydroPointer()
    if hydro is not None:
        info = hydro.get_bulk_info()
    # ... analyse this event ...
# Finish() is called automatically when the loop ends.
```

The memory for each event is released (ClearPerEvent) immediately after the consumer resumes the generator, and Finish() is called when iteration ends or the generator is closed (e.g. on an early `break`).

Parameters:

- **`main_xml`** (`str`) – Path to the main (default parameters) XML file.
- **`user_xml`** (`str`) – Path to the user (overrides) XML file.
- **`modules`** (`sequence of JetScapeTask`, default: `None` ) – Manual (Mode B) pipeline. If omitted, the task list is taken from the XML (Mode A). When given, the same pipeline-order validation as run_manual() is applied.
- **`n_events`** (`int`, default: `None` ) – Override the number of events from the XML.
- **`start_event`** (`int`, default: `0` ) – Starting event number for the global event counter.

Yields:

- `JetScapePerEvent` – The driver object, with the current event's data available in memory.

#### `run_per_event(main_xml: str, user_xml: str, on_event: Callable[[JetScapePerEvent], None], modules: Sequence | None = None, n_events: int | None = None, start_event: int = 0) -> JetScapePerEvent`

Run a JETSCAPE simulation, invoking a callback once per event.

Callback-based counterpart to per_event_loop(). `on_event(js)` is called for each event after ExecPerEvent() and before ClearPerEvent(), i.e. while the event's module results are still in memory::

```text
def analyse(js):
    hydro = JetScapeSignalManager.Instance().GetHydroPointer()
    ...

run_per_event(main_xml, user_xml, analyse)
```

Parameters:

- **`main_xml`** (`str`) – XML configuration file paths.
- **`user_xml`** (`str`) – XML configuration file paths.
- **`on_event`** (`callable`) – Called as on_event(js) for each event, with the live data in memory.
- **`modules`** (`sequence of JetScapeTask`, default: `None` ) – Manual (Mode B) pipeline; if omitted the XML task list (Mode A) is used.
- **`n_events`** (`int`, default: `None` ) – Override the number of events from the XML.
- **`start_event`** (`int`, default: `0` ) – Starting event number for the global event counter.

Returns:

- `JetScapePerEvent` – The driver object after Finish() has been called.

### `jetscape.workdir`

python/jetscape/workdir.py

A job's private working directory and its seed, for the production scripts and for scripts and notebooks of one's own (docs/Plans/PLAN_container_user.md, step 1). Pure Python, no build needed to import.

The functions moved here unchanged from example/prod_AuAu_0_10/run_prod.py, which re-exports them under the same names, so every production script keeps calling `rp.enter_workdir(a, ...)` etc. They take the scripts' parsed arguments `a` (any object with the attributes they read: `build`, `main_xml`, `workdir`, `in_build`, `keep_workdir`, `seed`, ...).

For a script of one's own, JobDir does the same with plain arguments:

```text
import jetscape as js
from jetscape.workdir import JobDir

with JobDir("out", "my_job.xml") as job:            # -> out/my_job.xml, out/work/my_job
    with js.per_event(job.main_xml, job.user_xml) as driver:
        for i in driver.events():
            ...
```

#### `XMLProblem(problems)`

Bases: `FileNotFoundError`

The job's XMLs can't run with this X-SCAPE: a user XML tag the main XML lacks, or an input of the run with no file behind it. Raised before anything runs. `problems` lists them; the message says what to do.

#### `JobDir(out_dir, user_xml, *, name=None, main_xml=None, build=None, keep=False)`

A job's private working directory, as the production scripts make it, for a script or notebook of one's own -- runnable from any writable directory (a container's /work).

On entering: `user_xml` is copied to `<out_dir>/<name>.xml` (`job.user_xml`), the main XML to `<out_dir>/work/<name>/jetscape_main.xml` (`job.main_xml`), their '../' paths made absolute against the build, music_input copied and the build's tables linked (enter_workdir), and the current directory changed to `job.path`, where the modules write their side files (3dMCGlauber's events_summary.dat, ...).

On leaving: back to the directory it was entered from; `job.path` is removed unless `keep` or an exception left the block (then it is kept and printed).

`build`: default jetscape.paths.build_dir() ($JETSCAPE_BUILD, else /build_gpu); `main_xml`: default jetscape.paths.main_xml(); `name`: default the user XML's stem.

#### `add_seed_args(p: argparse.ArgumentParser, drives: str) -> None`

--seed, --campaign, --index and --seed-registry (run_prod.py and run_prod_jet.py).

#### `output_name(a, base: str) -> str`

The output file name: --out, else by seed when the seed is given -- [\*]*seed.h5 -- and by job number when it is drawn (--seed 0, with the start time as the campaign if none was given):*\* .h5.

#### `resolve_seed(a, out_h5: str, record: bool = True) -> None`

Make `a.seed` concrete and note where it came from (`a.seed_source`).

--seed 0 draws from OS entropy (`secrets`) a seed in 1..SEED_MAX that the registry does not hold yet. With `record` the seed is appended to the registry (under an exclusive lock, so jobs starting together cannot draw the same one); an explicit seed is recorded too, and reported when the registry has it for another file (the same collisions: fine for paired comparisons, not for more statistics).

#### `resolve_parent_path(text: str, build: str) -> tuple`

A '../' path of an XML -> (absolute path, whether it exists).

The path relative to `build` (the working directory of an in-build run); if that's missing, `build` joined with each tail of its parts, longest first. An installed tree (the xscape wheel's share/xscape) has no ../external_packages or ../config, and keeps iSS_tables, iSS_parameters.dat and config/ directly in it. So ../external_packages/iSS/iSS_tables resolves to /iSS_tables and ../config/HH_LightExcitedMesons.xml to /config/HH_LightExcitedMesons.xml. The same rule as X-SCAPE's examples/run_xml.py, which run_in_workdir.sh uses.

#### `xml_problems(main_xml: str, user_xml: str, build: str) -> list`

What would stop the job: user XML tags the main XML lacks (X-SCAPE exits the process at the first one, Python included), and the run's '../' inputs with no file behind them, in the build or by their trailing parts in it. Nothing is written. The inputs are not checked when the XML turns automatic module setup off.

#### `enter_workdir(a, workdir: str, job_xml_path: str) -> str`

Make `workdir` this job's working directory -> the main XML to use.

The Python counterpart of X-SCAPE's examples/run_in_workdir.sh (which execs the runJetscape binary, so it cannot drive PyJetscape). In the build tree, concurrent jobs share music_input -- MusicWrapper truncates and rewrites it at every MUSIC init, and a job reading it in that window spins forever in MUSIC's StringFind4 -- as well as every file the modules write to the working directory (3dMCGlauber strings_event\_.dat and events_summary.dat, MUSIC's momentum_anisotropy / eccentricities / meanpT files, ...). Here each job has its own directory with a private music_input, and the read-only assets come from the build tree:

- `XSCAPE_DATA_DIR` -> mcglauber.input (MCGlauberWrapper), LBT-tables fallback
- `HYDROPROGRAMPATH` -> MUSIC's EOS tables
- `LBT_TABLES_PATH` -> LBT tables
- symlinks for the dirs still opened relative to the working directory (WORKDIR_LINKS)
- '../' paths in the main and job XML made absolute (see \_absolute_parent_paths)

Before any of it, the XMLs are checked (xml_problems): XMLProblem, a FileNotFoundError, names the user XML tags the main XML lacks and the inputs with no file behind them.

#### `job_workdir(a, out_h5: str) -> str`

This job's working directory: --workdir, else OUTDIR/work/.

#### `leave_workdir(a, workdir: str, ok: bool) -> None`

Remove the working directory after a successful job unless --keep-workdir.

#### `pythia_data_problem()`

None if pyjetscape_core can be imported, else the reason.

Pythia finds its xmldoc through PYTHIA8DATA or the path compiled into libpythia8. Homebrew's Pythia (macOS) has a working compiled-in path; a relocated conda Pythia (the js_fno env) may not, and then the import aborts the process -- so it is probed in a subprocess, and only when PYTHIA8DATA is unset.

#### `hydro_label(build: str) -> str`

The production files' `hydro` attribute: "MUSIC (music4gpu)" or "MUSIC".

From the compiled core's MUSIC_BACKEND (the MUSIC it was linked against: "music", "music4gpu-cuda", "music4gpu-metal"); a core without it (built before) falls back to the old test, "gpu" in the build folder's name (build_gpu).

#### `build_or_job_dir(out_dir, user_xml, main_xml=None, *, name=None, keep=False)`

Where an example script runs X-SCAPE -> an object with `main_xml`, `user_xml`, `path`.

In a build tree (`music_input` in the current directory: the old way, `cd X-SCAPE/build_gpu` and run the script from there) it runs right here with the XMLs as given, as these scripts always did. Anywhere else -- a copy of the examples (xscape-examples), the container's /work, the macOS wheel -- it runs in a JobDir under `out_dir` (`keep`: leave its working directory, e.g. for files the XML writes there). Give output paths absolute: a JobDir changes the current directory.

### `jetscape.paths`

python/jetscape/paths.py

Where the X-SCAPE tree, its build and the main XML are, for scripts and notebooks that live anywhere -- a copy in a container's /work, a user's own directory -- instead of inside the tree (docs/Plans/PLAN_container_user.md, step 1). Pure Python, no build needed. Paths are absolute but not resolved (symlinks kept), as the production scripts make them.

```text
XSCAPE          the X-SCAPE tree (the production image sets it to /opt/X-SCAPE);
                default: the checkout this package sits in (X-SCAPE/external_packages/
                js-contrib/contribs/PyJetscape/python/jetscape), else the installed
                tree of the xscape wheel (macOS: site-packages/xscape/share/xscape)
JETSCAPE_BUILD  the build tree to run; default: <XSCAPE>/build_gpu, or the installed
                tree itself, which holds music_input, EOS and the tables
```

The installed tree of the wheel (`cmake --install` of X-SCAPE) has config/ and the data of a build tree side by side, so it serves as both XSCAPE and the build.

#### `pyjetscape_dir() -> Path`

contribs/PyJetscape (this package is its python/jetscape) -- in a checkout.

#### `installed_package() -> Path | None`

The xscape wheel's package directory (site-packages/xscape: bin, lib, share, the compiled core), or None when that wheel is not installed. Found without importing it.

#### `installed_tree() -> Path | None`

The xscape wheel's X-SCAPE tree (/share/xscape), or None.

#### `examples_dir() -> Path`

contribs/PyJetscape/example: the production scripts, notebooks and their XMLs -- the checkout's, else the copy in the xscape wheel's installed tree.

#### `utils_dir() -> Path`

js-contrib's utils/ (jobmon.py, prodwizard.py, h5_inspect.py, remote_transfer/, ...): the checkout's, else the one in $XSCAPE's js-contrib (the production image), else the copy in the xscape wheel's installed tree (share/xscape/utils).

#### `xscape_root() -> Path`

The X-SCAPE tree: $XSCAPE, else the checkout this package sits in, else the xscape wheel's installed tree.

#### `build_dir() -> Path`

The build tree: $JETSCAPE_BUILD, else /build_gpu, else -- an installed tree, which has music_input but no build_gpu -- the X-SCAPE tree itself.

#### `main_xml() -> Path`

X-SCAPE's config/jetscape_main.xml.

### `jetscape.copy_examples`

python/jetscape/copy_examples.py

`xscape-examples [DEST]` -- copy PyJetscape's examples into a writable directory: the console script of the installed package (the xscape wheel), the Python twin of the production image's utils/container/xscape-examples.

`xscape-utils [DEST]` -- the same for js-contrib's utils/ (jobmon.py, prodwizard.py, h5_inspect.py, remote_transfer/, ...), to read or adapt; jetscape.paths.utils_dir().

The examples write next to themselves, and an installed or image tree is read-only, so they run from a copy:

```text
xscape-examples                 # -> /work/examples in the image, else ./examples
xscape-examples ~/xscape/examples
```

Copies jetscape.paths.examples_dir() (production scripts, notebooks, user XMLs, configs), without outputs (out/, out\_\*/, \*.h5, \*.root) and caches; in a git checkout, the tracked files only. Never overwrites: DEST must be new or empty. The copies find the X-SCAPE tree and its build through jetscape.paths ($XSCAPE, $JETSCAPE_BUILD, else the checkout or the installed tree).

#### `copy_tree(src: str, dest: str, prog: str = 'xscape-examples') -> str`

Copy `src` into `dest` (new or empty) -> `src`: in a git checkout the tracked files only, else everything but outputs and caches (\_EXCLUDE).

#### `copy_examples(dest: str) -> str`

Copy the examples into `dest` (new or empty) -> the source directory.

#### `copy_utils(dest: str) -> str`

Copy js-contrib's utils/ into `dest` (new or empty) -> the source directory.
