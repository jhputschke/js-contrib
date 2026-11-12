# 03 · The production scripts

**Time:** 45 minutes · **Needs:** X-SCAPE (the production container, the Mac wheel or a source build) · **Data:** `auau_stats` and `auau_pair_mini`: a few job records (kB) and one table; the notebook makes its own job (about 0.7 GB in `out/`)

Tutorial 02 ran events from a Python loop. For physics you need thousands of events, made the same way, recorded so that they can be checked, combined and shared. That is what the *production scripts* do. This tutorial runs one real production job, a central Au+Au collision with two jets, through both stages, and looks at every file each step writes:

- the driver `run_prod_jet.py`, its user XML and its options;
- seeds, and why a seed is a set of collisions;
- stage 2: hadronization with `hadronize.py`, and the ROOT conversion;
- `$\hat p_T$` windows and their cross sections, hadrons made in the job, correlated sampling;
- how many jobs make a campaign, how it is watched, and how its files are shared on the OSDF.

The job is a small copy of the one behind the data set `auau_pair_mini` of tutorials 04, 06 and 07, and the last section shows the recipe of all the tutorial data.

Reference pages: [Running the productions](https://jhputschke.github.io/js-contrib/run/), [Au+Au jets](https://jhputschke.github.io/js-contrib/run/auau-jet/), [Clusters, GPUs and monitoring](https://jhputschke.github.io/js-contrib/run/clusters/), [Data on the OSDF](https://jhputschke.github.io/js-contrib/run/osdf/).

> **Work in progress: examples, not vetted physics.** The tutorials show how to run the simulation and use its outputs. The data sets are small, so the statistics are low, and the analyses have not been checked to the standard of a publication: treat every number and figure as an illustration, not as a physics result.

> **The outputs shown** were produced on an Apple Silicon Mac, in the environment the xscape wheel 0.2.0 is built in, with the hydro on the GPU through Metal.

```python
# Setup: the tutorial helpers sit next to this notebook (_common/).
import os, sys
sys.path.insert(0, os.getcwd())
from _common import tutorial_env as te
env = te.setup()
```

```
tutorials v2 | setup: source (runs X-SCAPE; hydro: Metal GPU) | Python 3.13.1 | data: data/
```

```python
ok, missing = te.meets(['xscape'], env.caps)
assert ok, f"this tutorial needs {missing}: see 01_check_setup.ipynb"
```

## 1. Jobs and campaigns

A **production** is a folder of ready-made scripts for one physics setup. Its **driver** runs one **job**: one seed, N events, one set of output files. A **campaign** is many jobs of the same setup, each with its own seed, run side by side on one or many GPUs. Every production works this way:

| production           | physics                                                                          | driver               |
| -------------------- | -------------------------------------------------------------------------------- | -------------------- |
| `prod_AuAu_0_10`     | 0–10 % Au+Au at 200 GeV, hydro only                                              | `run_prod.py`        |
| `prod_AuAu_0_10_jet` | the same medium with a jet, whose lost energy flows back into a second hydro run | `run_prod_jet.py`    |
| `prod_OO_isr`        | O+O at 200 GeV with initial-state radiation (tutorial 02's chain)                | `run_prod_oo_isr.py` |

We use the main production, `prod_AuAu_0_10_jet`. Its folder holds the driver, the stage-2 scripts and `run_jobs.sh` for campaigns. In the container and the Mac wheel the scripts sit in a read-only place, which is fine: they write only into the output directory we give them. The helper `script` runs one of them with this notebook's Python, in a process of its own, and shortens the paths in what it prints:

```python
import json, re, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from jetscape import paths

EX = paths.examples_dir()                      # the examples: the productions and their XMLs
PROD = EX / "prod_AuAu_0_10_jet"
OUT = Path("out/03_production").resolve()      # this tutorial's files (out/ is git-ignored)
JOB = OUT / "AuAu"                             # the job's output directory
OUT.mkdir(parents=True, exist_ok=True)
for old in [JOB, OUT / "demo"]:                # a second run of the notebook starts afresh
    shutil.rmtree(old, ignore_errors=True)

SHORT = [(str(JOB), "out/03_production/AuAu"), (str(OUT), "out/03_production"),
         (str(PROD), "<prod>"), (str(EX), "<examples>"), (str(paths.build_dir()), "<build>"),
         (str(paths.xscape_root()), "<X-SCAPE>")]

def tidy(text):
    """This machine's paths -> short names."""
    for long, short in SHORT:
        text = text.replace(long, short)
    return text

def script(name, *args, log=None):
    """Run a script of the production folder in a new process -> (its output, seconds).
    The full output also goes to the file `log`, if given."""
    t0 = time.time()
    done = subprocess.run([sys.executable, PROD / name, *map(str, args)], cwd=OUT, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if log:
        Path(log).write_text(done.stdout)
    if done.returncode != 0:
        print(tidy(done.stdout[-3000:]))
        raise RuntimeError(f"{name} failed")
    return tidy(done.stdout), time.time() - t0

print(sorted(p.name for p in PROD.iterdir() if p.suffix in (".py", ".sh")))
```

```
['add_initiators.py', 'hadronize.py', 'music_cpu_gpu_hadrons.py', 'music_cpu_gpu_hydro.py', 'paired_noise.py', 'run_h5toROOT.py', 'run_hadronize.py', 'run_jobs.sh', 'run_prod_jet.py', 'run_prod_vacuum.py', 'wake_grid_compare.py']
```

Every script explains itself with `--help`. The driver has many options; most of them pick what to simulate and what to store. Here are the ones this tutorial uses:

```python
help_text, _ = script("run_prod_jet.py", "--help")
print(len(re.findall(r"^  --", help_text, re.M)), "options\n")

def option_help(opt):
    """The --help entry of one option."""
    m = re.search(rf"^  {re.escape(opt)}\b.*?(?=^  -|^\S|\Z)", help_text, re.M | re.S)
    return m.group(0).rstrip()

for opt in ["--events", "--reuse", "--pthat-min", "--parton-ymax", "--write-particlize",
            "--seed", "--outdir"]:
    print(option_help(opt))
```

```
65 options

  --events EVENTS       events in this job (default 10)
  --reuse REUSE         jet events per background (setReuseHydro); 1 = a new
                        background every event (default)
  --pthat-min PTHAT_MIN
                        PythiaGun pTHatMin [GeV] (default: the XML's)
  --parton-ymax PARTON_YMAX
                        PythiaGun: keep only events whose partons handed to
                        the framework (status 62 after ISR/MPI; final partons
                        with FSR_on) pass |y| < YMAX on their two hardest (see
                        --parton-y-mode; leptons do not count). Rejected
                        events are regenerated before any shower or hydro; the
                        cross sections are Pythia's x kept/tried per window.
                        Records the acceptance and the partons' rapidity per
                        event. Needs X-SCAPE's PythiaGun with <partonYMax>
  --write-particlize {none,jet,both}
                        also write <stem>_particlize.h5: the freeze-out
                        surface of the jet leg (jet) or of both legs (both,
                        the background once per background) plus the final
                        partons, everything hadronize.py needs to hadronize
                        the event exactly. Switches on the surface of those
                        legs (see --surface). Default none
  --seed SEED           framework seed (drives 3dMCGlauber, Pythia,
                        Matter/LBT), 1..900000000; the same seed gives the
                        same collisions. 0: draw a unique one from OS entropy,
                        recorded in the file, the .json and the seed registry
                        (the job stays reproducible); the file is then named
                        by --campaign. Not X-SCAPE's own seed 0, which Pythia
                        takes from the clock in seconds (jobs started together
                        get the same jets) and records nowhere
  --outdir OUTDIR       output directory (default: ./out next to this script)
```

## 2. The user XML and the job XML

The physics is in a **user XML**, as in tutorial 02. The production's template is `config/AuAu_MCGlauber_MUSIC_0_10_jet.xml`: the initial state, the hydro with its viscosities, Matter and LBT, the liquefier that turns the jet's lost energy into sources for the fluid. The driver takes it, or your own with `--user-xml my.xml`, and fills in what belongs to the job: the number of events, the seed, the hard process and its window, what to store. It checks the result and writes it next to the output, as `<stem>.xml`; the same text goes into the files (tutorial 04).

So, **to change the physics**, copy the template, edit the copy and pass it with `--user-xml`; **to change the job**, use the options. A dry run (`--dry-run`) checks everything, writes the job XML and prints the plan, without running an event. Here is our job, a small copy of the one behind `auau_pair_mini`:

```python
JOB_ARGS = ["--events", 2, "--reuse", 2, "--seed", 101, "--pthat-min", 30, "--pthat-max", 40,
            "--parton-ymax", 0.6, "--write-particlize", "both", "--outdir", JOB]
plan, _ = script("run_prod_jet.py", *JOB_ARGS, "--dry-run")
print(plan)
```

```
prod_AuAu_0_10_jet: 2 event(s), seed 101, grid_mode grid, PythiaGun pTHat 30.0-40.0 GeV, partons |y| < 0.6 (leading), reuse 2, deposition on, freeze-out surface: bg+jet, particlize input: both
  build    <build>
  job XML  out/03_production/AuAu/AuAu_0_10_jet_seed0101.xml
  grid     <examples>/config/grid_fno.yaml
             x -12.2063 .. 12.2063, n = 64, step = 0.3875
             y -12.2063 .. 12.2063, n = 64, step = 0.3875
           eta -4.84375 .. 4.84375, n = 32, step = 0.3125
           tau 0.5 + k*0.1, max_ntau = auto
               2.0 MB per frame uncompressed
  edge     flag e > 0.2431 GeV/fm^3 on the x/y edge of the output grid (e(T_fo = 0.15 GeV), EOS 91)
  output   out/03_production/AuAu/AuAu_0_10_jet_seed0101.h5
  output   out/03_production/AuAu/AuAu_0_10_jet_seed0101_particlize.h5
```

The first line is the plan: 2 events with seed 101, PYTHIA with `$\hat p_T$` 30–40 GeV and the leading parton at `$|y| < 0.6$`, one background for both jets (`reuse 2`), the jet's energy deposited in the fluid, and both freeze-out surfaces stored for hadronization. The output grid is the one the hydro history is stored on (tutorial 06). What did the driver change in the template? We compare the two XMLs element by element:

```python
import xml.etree.ElementTree as ET

def leaves(el, path=""):
    """{path: text} of every element without children."""
    out = {}
    for child in el:
        if not isinstance(child.tag, str):
            continue                                  # a comment
        p = f"{path}/{child.tag}"
        while p in out:
            p += "+"                                  # a repeated tag
        out.update(leaves(child, p) if len(child) else {p: (child.text or "").strip()})
    return out

template = leaves(ET.parse(EX / "config" / "AuAu_MCGlauber_MUSIC_0_10_jet.xml").getroot())
job_xml = leaves(ET.parse(JOB / "AuAu_0_10_jet_seed0101.xml").getroot())
changed = [(k, template.get(k, "(absent)"), job_xml.get(k, "(absent)"))
           for k in sorted(set(template) | set(job_xml)) if template.get(k) != job_xml.get(k)]
pd.DataFrame(changed, columns=["element", "template", "job"]).set_index("element")
```

|                                     | template | job     |
| ----------------------------------- | -------- | ------- |
| element                             |          |         |
| /Hard/PythiaGun/extraHard/noTrigger | (absent) | 0       |
| /Hard/PythiaGun/pTHatMax            | 70       | 40.0    |
| /Hard/PythiaGun/pTHatMin            | 50       | 30.0    |
| /Hard/PythiaGun/partonYMax          | (absent) | 0.6     |
| /Hard/PythiaGun/partonYMode         | (absent) | leading |
| /Hydro/MUSIC/freeze_out_surface     | 0        | 1       |
| /Hydro/MUSIC/slim_bulk_info         | (absent) | 1       |
| /Random/seed                        | 1        | 101     |
| /nEvents                            | 10       | 2       |
| /nReuseHydro                        | 1        | 2       |
| /setReuseHydro                      | false    | true    |

**What to notice.** Only the job changed: the number of events, the seed, the window and the parton cut of PYTHIA, the reuse of the background, and the freeze-out surface that the particlize file needs. Two technical switches come with them: `slim_bulk_info` (a smaller copy of the medium for the energy loss, with the same output) and the extra hard scatterings, off. Every physics setting is the template's. This is what makes the jobs of a campaign, and campaigns made months apart, comparable: each records its full XML, and you can check that the physics agrees (tutorial 04, section 7).

## 3. Seeds

`--seed S` sets X-SCAPE's `<Random><seed>`. Every random number of the job derives from it: the nucleons and strings of the initial state, PYTHIA's hard scattering, the energy loss. So **a seed is a set of collisions**: the same seed, XML and build give the same events, bit for bit. That is what you want to compare two settings on the same collisions (tutorial 02 did this with `$\eta/s$`), and exactly what you must avoid in a campaign meant to add statistics.

- **An explicit seed** (`--seed 101`) names the file (`..._seed0101.h5`) and reproduces it.
- **`--seed 0`** draws a new seed from the operating system's entropy, one that the *seed registry* has not seen, and records it in the files and the job record, so the job stays reproducible. The file is then named by the campaign and a job number (`AuAu_0_10_jet_<campaign>_0001.h5`). This is what campaigns use.
- **Not X-SCAPE's own seed 0.** In a plain X-SCAPE XML, seed 0 means "take the clock": PYTHIA then seeds itself from the time in seconds, so jobs started together get the same jets, and the seed is recorded nowhere. The drivers never pass 0 to X-SCAPE.
- **The seed registry** is `seeds_used.tsv` in the *parent* of the output directory: every job appends its seed, under a lock, and a seed used before is reported. Campaigns kept side by side in one directory share it. Our job writes to `out/03_production/AuAu`, so its registry is `out/03_production/seeds_used.tsv`.

## 4. Stage 1: hydro and jets

Now the job itself. The background (MUSIC_1) runs once; each of the two jets showers in it with Matter and LBT; the energy and momentum the jet loses becomes *droplets*, sources for a second hydro run with the same strings (MUSIC_2). One Au+Au jet event takes about 10–15 s on an Apple Silicon GPU, so the job takes well under a minute. Like `run_jobs.sh`, we keep the driver's output as `<stem>.log` in the output directory:

```python
STEM = "AuAu_0_10_jet_seed0101"
text, seconds = script("run_prod_jet.py", *JOB_ARGS, log=JOB / f"{STEM}.log")
for line in text.splitlines():
    if re.match(r"(PairH5Writer|ParticlizeH5Writer): event|\s+pTHat|restart", line):
        print(line)
print(f"stage 1: {seconds:.0f} s")
```

```
  pTHat    window 0: 30-40 GeV (Pythia seed 101)
PairH5Writer: event 0: jet 131 / bg 103 frames, 24 droplets (54.14 GeV), 6 leading frames identical, bg_id 0
ParticlizeH5Writer: event 0: jet 1165243 cells, bg 1151293 cells, 49 final partons, bg_unit 0
PairH5Writer: event 1: jet 103 / bg 103 frames, 14 droplets (34.45 GeV), 9 leading frames identical, bg_id 0
ParticlizeH5Writer: event 1: jet 1160069 cells, bg 1151293 cells, 41 final partons, bg_unit 0
stage 1: 41 s
```

Per event, the pair writer reports how many time frames each hydro leg has, how many droplets the jets deposited and how much energy they carried, and how many leading frames of the two legs are identical (before the first droplet arrives). The particlize writer reports the cells of each leg's freeze-out surface. The background (`bg`) is the same for both events: it was computed once.

The files of the job:

```python
def listing(folder):
    return pd.DataFrame([(p.name, p.stat().st_size / 1e6) for p in sorted(folder.iterdir()) if p.is_file()],
                        columns=["file", "MB"]).set_index("file").round(2)
listing(JOB)
```

|                                      | MB     |
| ------------------------------------ | ------ |
| file                                 |        |
| AuAu_0_10_jet_seed0101.h5            | 411.34 |
| AuAu_0_10_jet_seed0101.json          | 0.00   |
| AuAu_0_10_jet_seed0101.log           | 0.03   |
| AuAu_0_10_jet_seed0101.xml           | 0.01   |
| AuAu_0_10_jet_seed0101_particlize.h5 | 271.31 |

- **`<stem>.h5`**, the hydro **pair**: the jet leg and the background on the output grid. With a shared background it is stored once. This is the big one, and it is only needed for the hydro (tutorial 06) and for training the ML hydro.
- **`<stem>_particlize.h5`**: both freeze-out surfaces with their full viscous information, the final partons and the shower initiators: everything stage 2 needs, on its own.
- **`<stem>.xml`**: the job XML as run. **`<stem>.json`**: the **job record**, written at the end, below. **`<stem>.log`**: the driver's output.

The job record says what was done and how it went:

```python
record = json.loads((JOB / f"{STEM}.json").read_text())
keys = ["seed", "seed_source", "campaign", "hard", "reuse", "events_requested", "events_written",
        "droplets_total", "events_hit_edge", "wall_s", "pthat_bins", "pthat_bin_n_tried",
        "pthat_bin_acceptance", "pthat_bin_sigma_gen_mb", "particlize_legs"]
for k in keys:
    print(f"{k:24s} {record[k]}")
```

```
seed                     101
seed_source              explicit
campaign                 None
hard                     PythiaGun pTHat 30.0-40.0 GeV, partons |y| < 0.6 (leading)
reuse                    2
events_requested         2
events_written           2
droplets_total           38
events_hit_edge          {'bg': 0, 'jet': 0}
wall_s                   25.1
pthat_bins               [[30.0, 40.0]]
pthat_bin_n_tried        [2]
pthat_bin_acceptance     [1.0]
pthat_bin_sigma_gen_mb   [1.3311443618951287e-06]
particlize_legs          ['jet', 'bg']
```

`run_jobs.sh` looks at `events_written` to decide whether a job is complete when a campaign is restarted, and the monitoring tools read `droplets_total`, `events_hit_edge` (fluid that reached the edge of the output grid) and `wall_s`. The `$\hat p_T$` entries are the subject of section 7. And the registry now knows seed 101:

```python
reg = pd.read_csv(OUT / "seeds_used.tsv", sep="\t").rename(columns={"# seed": "seed"})
reg["file"] = reg["file"].map(tidy)
reg[["seed", "source", "file"]]
```

|     | seed | source   | file                                             |
| --- | ---- | -------- | ------------------------------------------------ |
| 0   | 101  | explicit | out/03_production/AuAu/AuAu_0_10_jet_seed0101.h5 |

### A seed is a set of collisions

The data set `auau_pair_mini` of tutorials 04, 06 and 07 was made with the same template, the same window and seed 101, but five jets per background (`--reuse 5`). A background is made once per `reuse` events, so its first two events should be ours (with the build the data were made with). The pair files record each event's `$\hat p_T$` and droplets (`diag/`):

```python
import h5py, hdf5plugin

cols = ["pthat", "n_droplets", "E_droplets", "ntau_jet", "ntau_bg"]
with h5py.File(JOB / f"{STEM}.h5") as f:
    ours = pd.DataFrame({c: f["diag"][c][:] for c in cols})
with te.h5("auau_pair_mini", f"{STEM}.h5") as f:
    mini = pd.DataFrame({c: f["diag"][c][:] for c in cols})
pd.concat({"this job": ours, "auau_pair_mini": mini}, axis=1).round(3).fillna("")
```

|     | this job |            |            |          |         | auau_pair_mini |            |            |          |         |
| --- | -------- | ---------- | ---------- | -------- | ------- | -------------- | ---------- | ---------- | -------- | ------- |
|     | pthat    | n_droplets | E_droplets | ntau_jet | ntau_bg | pthat          | n_droplets | E_droplets | ntau_jet | ntau_bg |
| 0   | 31.6     | 24.0       | 54.136     | 131.0    | 103.0   | 31.600         | 24.0       | 54.136     | 131.0    | 103.0   |
| 1   | 33.995   | 14.0       | 34.446     | 103.0    | 103.0   | 33.995         | 14.0       | 34.446     | 103.0    | 103.0   |
| 2   |          |            |            |          |         | 30.156         | 22.0       | 27.857     | 109.0    | 103.0   |
| 3   |          |            |            |          |         | 34.729         | 11.0       | 15.765     | 123.0    | 103.0   |
| 4   |          |            |            |          |         | 30.714         | 17.0       | 19.398     | 114.0    | 103.0   |

**What to notice.** The first two events of the data set are our two events: the same hard scattering, the same 24 and 14 droplets with the same energy, the same hydro lifetimes. The other three events of the data set are the next jets on the same background. This is also how to extend a sample on purpose: same seed and settings, more events. And it is why two campaigns meant as independent statistics must never share seeds.

## 5. Stage 2: hadrons

`hadronize.py` turns one particlize file into hadrons, on the CPU, without MUSIC: iSS samples each stored freeze-out surface (the *bulk*), and PYTHIA fragments the stored final partons (the *jet fragments*). Since the surfaces are stored, stage 2 can be repeated any number of times, with more oversamples or other settings, without the expensive stage 1. The options we use:

- `--oversample 20`: iSS samples per surface. Productions use 100–500; we keep it small.
- `--n-frag 5`: PYTHIA fragmentations of each event's partons.
- `--correlated --common-seeds`: correlated sampling. Sample `$k$` of a jet event and sample `$k$` of its background use the same random numbers, addressed by the surface cell. Where the two surfaces agree, they give the same hadrons, so jet minus background, the jet's wake in hadrons, is far less noisy (tutorial 07).

Each unit (an event, or a background) gets its own seed, derived from the base seed (`--seed`, default 1), the particlize file's unique id, the tag and the unit, and recorded in the file. For a whole campaign, `run_hadronize.py` runs `hadronize.py` over a directory, several files at once, and skips files whose hadrons are complete.

```python
text, seconds = script("hadronize.py", JOB / f"{STEM}_particlize.h5", "--correlated",
                       "--common-seeds", "--oversample", 20, "--n-frag", 5,
                       log=JOB / f"{STEM}_hadronize.log")
for line in text.splitlines():
    if re.match(r"\s+(event|\d+ event|bulk_|jet_frag)", line):
        print(line)
print(f"stage 2: {seconds:.0f} s")
```

```
  2 event(s), tags bulk_jet, bulk_bg, jet_frag, iSS oversample 20, 5 fragmentation(s)/event, base seed 1, EOS 91 (SMASH list, decays: iSS, Breit-Wigner masses for the unstable daughters of channels closed at the pole masses; eta, Sigma0 stable), eCMforHadronization 200
  event 0, bulk_jet 169040 hadrons, bulk_bg (unit 0) 168432, jet_frag 36.6/sample
  event 1, bulk_jet 169041 hadrons, jet_frag 28.8/sample
  bulk_jet  -> out/03_production/AuAu/AuAu_0_10_jet_seed0101_hadrons_bulk_jet.h5 (10.2 MB)
  bulk_bg   -> out/03_production/AuAu/AuAu_0_10_jet_seed0101_hadrons_bulk_bg.h5 (4.6 MB)
  jet_frag  -> out/03_production/AuAu/AuAu_0_10_jet_seed0101_hadrons_jet_frag.h5 (0.2 MB)
stage 2: 8 s
```

Three hadron files, one per *tag*: `bulk_jet` (the jet event's fluid: the medium with the jet's wake), `bulk_bg` (the background's fluid) and `jet_frag` (the jet's own hadrons). What each holds, and the seeds:

```python
with h5py.File(JOB / f"{STEM}_particlize.h5") as f:
    uuid = f.attrs["file_uuid"]
rows = []
for tag in ["bulk_jet", "bulk_bg", "jet_frag"]:
    with h5py.File(JOB / f"{STEM}_hadrons_{tag}.h5") as f:
        u = f["units"]
        n = len(f["hadrons/pid"])
        rows.append({"tag": tag, "units (events)": u["event"][:].tolist(),
                     "samples per unit": u["n_samples"][:].tolist(),
                     "hadrons per sample": round(n / u["n_samples"][:].sum(), 1),
                     "seeds": u["seed"][:].tolist(),
                     "correlated": bool(f.attrs.get("correlated_sampling", False)),
                     "source_uuid matches": f.attrs["source_uuid"] == uuid})
hadrons = pd.DataFrame(rows).set_index("tag")
hadrons
```

|          | units (events) | samples per unit | hadrons per sample | seeds                  | correlated | source_uuid matches |
| -------- | -------------- | ---------------- | ------------------ | ---------------------- | ---------- | ------------------- |
| tag      |                |                  |                    |                        |            |                     |
| bulk_jet | [0, 1]         | [20, 20]         | 8452.0             | [12608650, 12608650]   | True       | True                |
| bulk_bg  | [0]            | [20]             | 8421.6             | [12608650]             | True       | True                |
| jet_frag | [0, 1]         | [5, 5]           | 32.7               | [862761277, 702944669] | False      | True                |

**What to notice.**

- **One background unit for two events.** `bulk_bg` has a single unit: the background both jets shared. A jet event is `bulk_jet` + `jet_frag`; its background is `bulk_bg`'s unit `bg_unit` of the particlize file.
- **The common seeds.** Both `bulk_jet` units carry the seed of their background unit: that is `--common-seeds`. The jet fragments have their own seed per event.
- **About 8,400 hadrons per bulk sample** (all species, all rapidities: one central Au+Au collision), and a few tens per fragmentation of the jets.
- **`source_uuid`** ties each hadron file to its particlize file; the readers refuse mismatched files (tutorial 04).

### ROOT files

`run_h5toROOT.py` converts a directory of hadronized jobs into ROOT files, one per job (RNTuple), plus a *campaign file* with the cross sections of all jobs together. It needs no ROOT installation with `--writer uproot`. Tutorials 09 and 10 work with these files.

```python
text, seconds = script("run_h5toROOT.py", JOB, "--writer", "uproot")
print(text.strip())

import uproot
with uproot.open(JOB / f"{STEM}_hadrons.root") as rf:
    print("\nobjects:", rf.keys())
    print("bulk_jet entries (samples):", rf["bulk_jet"].num_entries)
    events = pd.DataFrame(rf["events"].arrays(["event", "pthat", "bg_unit", "E_droplets",
                                               "sigma_file_mb"], library="np"))
events.round(3).assign(sigma_file_mb=events["sigma_file_mb"].map("{:.2e}".format))
```

```
run_h5toROOT.py: 1 production file(s): 1 to convert, 0 already converted, 0 not ready; rntuple by uproot, tags bulk_jet,bulk_bg,jet_frag, -j 1; campaign file out/03_production/AuAu/campaign_campaign.root
  done        AuAu_0_10_jet_seed0101_hadrons.root: 2 events, bulk_jet 0.3 M, bulk_bg 0.2 M, jet_frag 0.0 M hadrons, 14 MB in 0 s
  campaign    out/03_production/AuAu/campaign_campaign.root: 1 file(s), windows 30-40 GeV: sigma 1.33e-06 mb, 2 events
run_h5toROOT.py: 1 converted, 0 failed, 0 not ready, in 0 s

objects: ['events;1', 'windows;1', 'provenance;1', 'bulk_jet;1', 'bulk_bg;1', 'jet_frag;1']
bulk_jet entries (samples): 40
```

|     | E_droplets | bg_unit | event | pthat  | sigma_file_mb |
| --- | ---------- | ------- | ----- | ------ | ------------- |
| 0   | 54.136     | 0       | 0     | 31.600 | 1.33e-06      |
| 1   | 34.446     | 0       | 1     | 33.995 | 1.33e-06      |

## 6. The job, step by step: what we have now

| step    | command                                                         | writes                                               | needs                 |
| ------- | --------------------------------------------------------------- | ---------------------------------------------------- | --------------------- |
| stage 1 | `run_prod_jet.py ... --write-particlize both`                   | `<stem>.h5`, `<stem>_particlize.h5`, `.json`, `.xml` | a GPU (or a slow CPU) |
| stage 2 | `hadronize.py <stem>_particlize.h5` (or `run_hadronize.py DIR`) | `<stem>_hadrons_{bulk_jet,bulk_bg,jet_frag}.h5`      | CPUs                  |
| ROOT    | `run_h5toROOT.py DIR`                                           | `<stem>_hadrons.root`, `<campaign>_campaign.root`    | CPUs                  |

The particlize file is self-contained: it can travel to another cluster for stage 2 without the pair file. Our files, all together:

```python
listing(JOB)
```

|                                            | MB     |
| ------------------------------------------ | ------ |
| file                                       |        |
| AuAu_0_10_jet_seed0101.h5                  | 411.34 |
| AuAu_0_10_jet_seed0101.json                | 0.00   |
| AuAu_0_10_jet_seed0101.log                 | 0.03   |
| AuAu_0_10_jet_seed0101.xml                 | 0.01   |
| AuAu_0_10_jet_seed0101_hadronize.log       | 0.01   |
| AuAu_0_10_jet_seed0101_hadrons.root        | 13.90  |
| AuAu_0_10_jet_seed0101_hadrons_bulk_bg.h5  | 4.63   |
| AuAu_0_10_jet_seed0101_hadrons_bulk_jet.h5 | 10.16  |
| AuAu_0_10_jet_seed0101_hadrons_jet_frag.h5 | 0.15   |
| AuAu_0_10_jet_seed0101_particlize.h5       | 271.31 |
| campaign_campaign.root                     | 0.01   |

## 7. `$\hat p_T$` windows and cross sections

The jet cross section falls steeply with `$p_T$`. A sample generated over one wide `$\hat p_T$` range would be almost all low-`$\hat p_T$` events. So the productions generate in **windows** of `$\hat p_T$` and weight each window by its cross section afterwards.

- `--pthat-min/--pthat-max` set one window (our job: 30–40 GeV).
- `--pthat-bins 10-20,20-30,30-50` runs K windows in one job, one PYTHIA per window with its own seed and cross section; event `$i$` uses window `$i \bmod K$`.
- `--jets-per-bin M` gives every background M jets in each window: `reuse` becomes `$K \times M$`, and the events per job must be a multiple of it. A background costs about as much as a jet leg, so sharing it saves nearly half the time.

PYTHIA estimates each window's cross section `$\sigma$` from the events it generates. With the parton cut (`--parton-ymax`) rejected events are regenerated, and `$\sigma$` is multiplied by the accepted fraction. Our job tried only 2 events, so its `$\sigma$` is a rough estimate; `auau_pair_mini` tried 6 for the same window:

```python
mini_record = json.loads(te.read_text("auau_pair_mini", f"{STEM}.json"))
for name, r in [("this job", record), ("auau_pair_mini", mini_record)]:
    print(f"{name:15s} sigma = {r['pthat_bin_sigma_gen_mb'][0]:.2e} mb from "
          f"{r['pthat_bin_n_tried'][0]} events tried, acceptance {r['pthat_bin_acceptance'][0]}")
```

```
this job        sigma = 1.33e-06 mb from 2 events tried, acceptance 1.0
auau_pair_mini  sigma = 2.09e-06 mb from 6 events tried, acceptance 0.8333
```

The two estimates of the same quantity differ by a factor 1.6: a single job's `$\sigma$` is noisy. A campaign combines the estimates of all its jobs. The data set `auau_stats` is a small campaign: 4 jobs of 30 events, three windows 10–20, 20–30 and 30–50 GeV, 2 jets per window per background. Each job recorded its own estimates, and the campaign file of `run_h5toROOT.py` holds the combination:

```python
per_job = []
for seed in [201, 202, 203, 204]:
    r = json.loads(te.read_text("auau_stats", f"AuAu_0_10_jet_stats_seed{seed:04d}.json"))
    per_job.append({"seed": seed, **{f"sigma {lo:.0f}-{hi:.0f} [mb]": s for (lo, hi), s in
                                     zip(r["pthat_bins"], r["pthat_bin_sigma_gen_mb"])},
                    "tried": r["pthat_bin_n_tried"], "reuse": r["reuse"], "wall_s": r["wall_s"]})
display(pd.DataFrame(per_job).set_index("seed").map(lambda v: f"{v:.2e}" if isinstance(v, float) and v < 0.1 else v))

with uproot.open(te.data("auau_stats", "stats_campaign.root")) as rf:
    windows = pd.DataFrame(rf["windows"].arrays(["pthat_lo", "pthat_hi", "sigma_mb", "sigma_err_mb",
                                                  "acceptance", "n_events", "weight_mb"], library="np"))
windows.map(lambda v: f"{v:.3g}")
```

|      | sigma 10-20 [mb] | sigma 20-30 [mb] | sigma 30-50 [mb] | tried        | reuse | wall_s |
| ---- | ---------------- | ---------------- | ---------------- | ------------ | ----- | ------ |
| seed |                  |                  |                  |              |       |        |
| 201  | 3.46e-03         | 4.97e-05         | 2.27e-06         | [19, 16, 15] | 6     | 327.8  |
| 202  | 2.37e-03         | 3.53e-05         | 2.13e-06         | [24, 20, 12] | 6     | 315.7  |
| 203  | 3.09e-03         | 3.95e-05         | 3.29e-06         | [14, 21, 12] | 6     | 377.7  |
| 204  | 2.69e-03         | 2.88e-05         | 2.65e-06         | [21, 24, 14] | 6     | 400.8  |

|     | pthat_lo | pthat_hi | sigma_mb | sigma_err_mb | acceptance | n_events | weight_mb |
| --- | -------- | -------- | -------- | ------------ | ---------- | -------- | --------- |
| 0   | 10       | 20       | 0.00285  | 0.000356     | 0.513      | 40       | 7.13e-05  |
| 1   | 20       | 30       | 3.73e-05 | 4.67e-06     | 0.494      | 40       | 9.33e-07  |
| 2   | 30       | 50       | 2.57e-06 | 2.95e-07     | 0.755      | 40       | 6.42e-08  |

**What to notice.**

- **The cross section falls by three orders of magnitude** from the 10–20 to the 30–50 GeV window, yet each window has 40 events. Generated over 10–50 GeV in one go, only about one event in a thousand would land in the 30–50 GeV window.
- **The jobs' estimates scatter by up to about 30 %** around the combined value: each job saw only 12–24 PYTHIA events per window. The combined value carries an error of 11–13 %.
- **`weight_mb` = `$\sigma$` / `n_events`** is the weight of one event of that window. A spectrum over all windows is the sum of the windows' histograms, each event weighted by its window's `weight_mb` (tutorial 05). Results for one window alone need no weights.

A dry run shows the plan of one such job. We give it `--seed 0` and a campaign name, as a campaign would: the seed is drawn from the operating system (yours will differ), the file is named by campaign and job number, and a dry run records nothing in the registry:

```python
plan, _ = script("run_prod_jet.py", "--events", 30, "--seed", 0, "--campaign", "demo",
                 "--pthat-bins", "10-20,20-30,30-50", "--jets-per-bin", 2, "--parton-ymax", 0.6,
                 "--hadrons-only", "--oversample", 10, "--n-frag", 10, "--no-x",
                 "--outdir", OUT / "demo", "--dry-run")
print(plan.splitlines()[0])
print([line.strip() for line in plan.splitlines() if line.strip().startswith("output")])
```

```
prod_AuAu_0_10_jet: 30 event(s), seed 693593494 (from OS entropy), campaign demo #1, grid_mode grid, PythiaGun pTHat windows 10-20, 20-30, 30-50 GeV, 2 jet(s) per window per background, partons |y| < 0.6 (leading), reuse 6, deposition on, freeze-out surface: bg+jet, particlize input: both, HADRONS ONLY (no pair or particlize file)
['output   out/03_production/demo/AuAu_0_10_jet_demo_0001_hadrons_{bulk_jet,bulk_bg,jet_frag}.h5 + *_primary.h5']
```

## 8. Hadrons in the job, and correlated sampling

That dry run also used **`--hadrons-only`**: iSS and the jet fragmentation run inside the GPU job, exactly as `hadronize.py` would (same code, settings and seeds), and the job writes only the hadron files and its record: no pair file, no particlize file, no stage 2. It suits campaigns with little oversampling, here 10 samples per jet event. The price: no surfaces, so no new samples later, and no hydro history. Measured on the GB10 (the [Au+Au jets](https://jhputschke.github.io/js-contrib/run/auau-jet/#hadrons-in-the-job-hadrons-only) page), it wrote 0.64 MB per event, against about 160 MB in two stages. `auau_stats` was made this way; its job record keeps the hadronization settings:

```python
r = json.loads(te.read_text("auau_stats", "AuAu_0_10_jet_stats_seed0201.json"))
print("hadrons_only:", r["hadrons_only"])
print("hadronize_settings:", r["hadronize_settings"])
```

```
hadrons_only: True
hadronize_settings: {'oversample': 10, 'oversample_bg': 'per-pthat-bin', 'n_frag': 10, 'keep_primaries': 'all', 'base_seed': 1, 'correlated': False}
```

`oversample_bg: per-pthat-bin` gives each background `$M \times$` `oversample` samples, as many as the `$M$` jet events of one window have together: the right balance for results per window. **Correlated sampling** (`--correlated`) works in the job as in stage 2; `auau_stats` did not use it, our job and `auau_pair_mini` did. Use it when the analysis is jet minus background, and analyse such files sample by sample (tutorial 07).

**Decide the hadron settings once per campaign.** The number of oversamples, the correlation, the precision (`--keep-bits-p/-x`) and the selection (`--eta-max`, `--charged`, `--no-x`) must be the same for every job: restarts skip complete files whatever settings made them.

## 9. From a job to a campaign

Nothing below runs in this notebook; each is a command for a terminal.

**`run_jobs.sh`** runs many jobs of one production, `-j P` at a time on one GPU, each in its own working directory:

```
cd <your copy of the examples>/prod_AuAu_0_10_jet
./run_jobs.sh [-j P] [--mps] [--stagger S] [--campaign NAME] NJOBS EVENTS_PER_JOB FIRST_SEED [OUTDIR] [driver options]

# 67 jobs x 15 events, new seeds, three windows, 5 jets per window per background
./run_jobs.sh -j 3 --campaign pth10-40_c1 67 15 0 ~/xscape/prod/AuAu_c1 \
    --pthat-bins 10-20,20-30,30-40 --jets-per-bin 5 --parton-ymax 0.6 --write-particlize both
```

- `FIRST_SEED 0` gives every job a new seed through `--seed 0` (a campaign); `FIRST_SEED 1` gives seeds 1, 2, ... (to reproduce files, or to compare settings on the same collisions).
- OUTDIR is the argument after FIRST_SEED; one campaign per OUTDIR. Every job leaves `<stem>.log`, `.json`, `.xml` there, and the campaign ends with `run_jobs.finished`.
- **Restart** with the same command: jobs whose `.json` says complete are skipped, the others are redone.
- On a Mac, split the cores between the jobs: `OMP_NUM_THREADS=5 OMP_WAIT_POLICY=passive ./run_jobs.sh -j 3 ...` on a 16-core M3 Max. On CUDA, `--mps` lets the jobs share the GPU.

**`utils/prodwizard.py`** plans a campaign before it runs: pick a preset (`auau_jet`, `auau_hadrons_only`, `oo_hadrons`, ...) and a machine (`mac_wheel`, `2x3090`, `slurm_a100`, ...), and it writes the launch command with the measured settings for that machine and estimates events per hour, duration, disk and memory. It checks the plan (events per job a multiple of windows × jets, memory, disk) and never starts jobs:

```
utils/prodwizard.py --preset auau_jet --machine mac_wheel --events 300     # xscape-prodwizard in the wheel
```

**`utils/jobmon.py`** watches a running or finished campaign from what the jobs write (logs, job records): jobs done, running and failed, events per hour, the time left, disk per event, and checks such as the deposition check, which flags the rare job whose jets deposited far less energy than the rest:

```
utils/jobmon.py ~/xscape/prod/AuAu_c1 --watch      # xscape-jobmon in the wheel
```

**On a cluster** with SLURM, `utils/slurm_prod_array.sh` runs a campaign as a job array in the production container: each array task gets one GPU and runs `P` jobs on it, all tasks sharing one seed registry. A GPU needs several jobs at once to stay busy, so one SLURM job per production job would waste it. Stage 2 runs on CPU nodes with `utils/slurm_hadronize.sh`:

```
SIF=$PWD/xscape_prod.sif CAMPAIGN=pth10-40_c2 NJOBS=40 EVENTS=15 \
  EXTRA_ARGS="--pthat-bins 10-20,20-30,30-40 --jets-per-bin 5 --parton-ymax 0.6 --write-particlize both" \
  sbatch --array=0-19 utils/slurm_prod_array.sh
```

The sizes per GPU type, the launch settings and a 12,000-event example are on [Clusters, GPUs and monitoring](https://jhputschke.github.io/js-contrib/run/clusters/).

## 10. Sharing the files on the OSDF

The tutorial data come from the **OSDF** (Open Science Data Federation), a network of data origins and caches. Reading is public: with pelicanfs in Python (`osdf://` URLs, tutorial 01), or over plain HTTPS. Writing to the namespace `osdf:///fno4hic` needs a login. `utils/remote_transfer/js_osdf.py` (`xscape-osdf` in the Mac wheel) moves whole productions or kinds of files:

```
./js_osdf.py login                                         # once: a link to log in from any browser
./js_osdf.py upload ~/xscape/prod/AuAu_c1 --what h5 --dry-run     # what would go
./js_osdf.py upload ~/xscape/prod/AuAu_c1 --what h5        # -> osdf:///fno4hic/AuAu_c1/
./js_osdf.py download AuAu_c1 --what root --to /scratch    # anyone, no login
```

- `--what` picks files by kind: `h5` (particlize and hadron files: everything an analysis needs), `root`, `pair` (the hydro, for the ML hydro), `all`.
- Incomplete files (an HDF5 file whose `complete` attribute is false, `*.part`) are left out. Each upload is checked: the size at the origin, and a CRC32C recorded in a manifest in the remote folder, which downloads (and `get_data.py`) check against.
- **Whatever is uploaded can be read by everyone.**
- **Uploaded files are never overwritten.** The OSDF caches keep a copy of a file for a long time, so a file replaced at the origin can still be served in its old version, and readers would get different data under one name. A changed data set is a new version under a new name instead: the tutorial data live under `tutorials/v2/`, and a change would become `v3`.

This notebook uploads nothing.

## 11. A worked example: how the tutorial data were made

The data sets of the tutorials were made by one script, `data_recipes/make_v2.sh` in this folder: each set is a function of a few production commands. Here are the two Au+Au sets:

```python
recipe = Path("data_recipes/make_v2.sh").read_text()
for name in ["pair_mini", "stats"]:
    print(re.search(rf"^{name}\(\) \{{.*?^\}}", recipe, re.M | re.S).group(0), "\n")
```

```
pair_mini() {
  local D="$OUT/AuAu_pair_mini"
  log "AuAu_pair_mini -> $D"
  (cd "$EX/prod_AuAu_0_10_jet" &&
   $PY run_prod_jet.py --events 5 --reuse 5 --seed 101 --pthat-min 30 --pthat-max 40 \
       --parton-ymax 0.6 --write-particlize both --outdir "$D" &&
   $PY hadronize.py "$D/AuAu_0_10_jet_seed0101_particlize.h5" \
       --correlated --common-seeds --oversample 200 --n-frag 20 &&
   $PY run_h5toROOT.py "$D" --writer uproot)
} 

stats() {
  local D="$OUT/AuAu_stats"
  log "AuAu_stats -> $D"
  (cd "$EX/prod_AuAu_0_10_jet" &&
   printf '%s\n' 201 202 203 204 | xargs -P "$JOBS" -I{} \
     $PY run_prod_jet.py --events 30 --seed {} --campaign stats \
       --pthat-bins 10-20,20-30,30-50 --jets-per-bin 2 --parton-ymax 0.6 \
       --hadrons-only --oversample 10 --n-frag 10 --no-x --outdir "$D" &&
   mkdir -p "$OUT/AuAu_stats_primaries" &&       # the undecayed hadrons (for an afterburner):
   mv "$D"/*_primary.h5 "$OUT/AuAu_stats_primaries/" &&   # not part of the set
   $PY run_h5toROOT.py "$D" --writer uproot -j "$JOBS")
}
```

**Read them with this tutorial in mind.**

- **`pair_mini`** is our job with five jets: `run_prod_jet.py --events 5 --reuse 5 --seed 101 ...`, then `hadronize.py --correlated --common-seeds --oversample 200 --n-frag 20` and `run_h5toROOT.py --writer uproot`. Its first two events are ours (section 4).
- **`stats`** is a campaign of four jobs with seeds 201–204, three windows and two jets per window per background, hadrons made in the job (`--hadrons-only`, 10 oversamples, no positions), started three at a time by `xargs -P` instead of `run_jobs.sh`. It moves the undecayed hadrons (`*_primary.h5`, kept for a later hadronic afterburner) out of the set, then writes the ROOT files and the campaign file.
- **Fixed seeds** make every set reproducible with the same build. The script's header states the machine and environment, and that uploaded sets are never overwritten.
- The last function, `upload`, sends each set with `js_osdf.py upload --what all --prefix tutorials/v2`.

### Check your result

Your job ran on your setup. Its collisions are fixed by the seed and the build: another GPU or MUSIC on the CPU changes the numbers slightly, another build of PYTHIA or 3dMCGlauber can give other collisions. The hadrons of stage 2 also depend on the particlize file's unique id, so they change with every run of stage 1. The check allows for both:

```python
exp = json.load(open("expected/03_production_scripts.json"))
assert record["events_written"] == exp["events"] and record["reuse"] == exp["reuse"] and record["seed"] == 101
assert hadrons["source_uuid matches"].all() and hadrons.loc["bulk_bg", "units (events)"] == [0]
assert set(hadrons.loc["bulk_jet", "seeds"]) == set(hadrons.loc["bulk_bg", "seeds"])   # common seeds
assert len(events) == 2 and 101 in reg["seed"].values
# the same collisions as shown here, unless your build of PYTHIA or 3dMCGlauber differs
if np.allclose(ours["pthat"], exp["pthat"], rtol=1e-6):
    np.testing.assert_allclose(ours["E_droplets"], exp["E_droplets"], rtol=0.1)
    np.testing.assert_allclose(hadrons.loc["bulk_jet", "hadrons per sample"], exp["bulk_jet_per_sample"], rtol=0.03)
else:
    print("Your collisions differ from the ones shown (another build): only the job's structure was checked.")
print("All checks pass.")
```

```
All checks pass.
```

## Exercises

1. **The null test.** Run one event with `--no-deposit` (and its own seed and output directory): MUSIC_2 then runs without the droplets, and the two legs must be identical. Check `diag/frames_identical` against `diag/ntau_jet` in the pair file. Why is this the first thing to run on a new machine or build?
1. **Stage 2 again.** Run `hadronize.py` on our particlize file once more, with `--tags jet_frag --seed 2 --force`. Which hadron file changes, and which seeds? Then without `--force`: what happens?
1. **A campaign's arithmetic.** With `--dry-run`, try `--events 20 --pthat-bins 10-20,20-30,30-40 --jets-per-bin 5`. What does the driver say, and why? Which numbers of events work?
1. **Plan a campaign.** Run `utils/prodwizard.py --list`, then a plan for 300 Au+Au events on your machine (`--preset auau_jet --machine mac_wheel --events 300`, or a cluster). How long would it take, and how much disk would it need?

## Next steps

- Every file of a job and its provenance, from the same recipe: tutorial 04.
- Spectra over many events, oversamples and the cross-section weights of the windows: tutorial 05.
- The pair file and the wake in the fluid: tutorial 06; the hadron files and the wake in hadrons: tutorial 07.
- Every option and measurement: [Au+Au jets](https://jhputschke.github.io/js-contrib/run/auau-jet/), [Clusters, GPUs and monitoring](https://jhputschke.github.io/js-contrib/run/clusters/), [Data on the OSDF](https://jhputschke.github.io/js-contrib/run/osdf/).
- The files of this tutorial are in `out/03_production/` (about 0.7 GB): delete the folder when you are done.
