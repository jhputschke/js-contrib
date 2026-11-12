# 04 · The files and their provenance

**Time:** 30 minutes · **Needs:** any setup · **Data:** `auau_pair_mini` (streamed: a few MB are read)

A production job writes several files, each for a different use. This tutorial opens every file of one job: a Au+Au collision with five jets. Along the way it shows what each stage of the simulation leaves behind, and where the job records how it was made: the seed, the physics settings, the software and the grid. Those records let you check that files belong together before you combine them.

Reference pages: [What a production writes](https://jhputschke.github.io/js-contrib/analysis/files/) and [The physics](https://jhputschke.github.io/js-contrib/about/physics/).

> **Work in progress: examples, not vetted physics.** The tutorials show how to run the simulation and use its outputs. The data sets are small, so the statistics are low, and the analyses have not been checked to the standard of a publication: treat every number and figure as an illustration, not as a physics result.

> **The outputs shown** were produced on an Apple Silicon Mac, in the environment the xscape wheel 0.2.0 is built in, with the hydro on the GPU through Metal.

```python
# Setup: the tutorial helpers sit next to this notebook (_common/). In Colab they are
# fetched first, with the jetscape readers; locally nothing is installed.
import os, sys, subprocess
if "google.colab" in sys.modules and not os.path.isdir("_common"):
    subprocess.run("curl -sL https://codeload.github.com/jhputschke/js-contrib/tar.gz/refs/heads/main"
                   " | tar xz --strip-components=3 js-contrib-main/tutorials/pyjetscape",
                   shell=True, check=True)
sys.path.insert(0, os.getcwd())
from _common import tutorial_env as te
te.colab_install()
env = te.setup()
```

```
tutorials v2 | setup: source (runs X-SCAPE; hydro: Metal GPU) | Python 3.13.1 | data: data/
```

```python
SET = "auau_pair_mini"
STEM = "AuAu_0_10_jet_seed0101"          # one job: one seed, one stem for all its files
```

## 1. One job, its files

A jet production runs in two stages:

1. **Stage 1, on the GPU** (`run_prod_jet.py`): the initial state, the hard scattering, the hydro of the background (MUSIC_1), the jet's energy loss in it (Matter + LBT), and the hydro again with the energy the jet deposited (MUSIC_2). It writes the **pair file** (both hydro evolutions), the **particlize file** (both freeze-out surfaces and the final partons), and the job record (`.json`, `.xml`).
1. **Stage 2, on the CPU** (`hadronize.py`): Cooper–Frye sampling of each freeze-out surface and hadronization of the jet partons. It writes the **hadron files**: `bulk_jet` (the medium with the jet's wake), `bulk_bg` (the medium alone) and `jet_frag` (the jet's own hadrons). `run_h5toROOT.py` then copies them into a **ROOT file**.

Stage 2 can be repeated any number of times from the particlize file, with more oversamples or other settings, without rerunning the hydro.

All files of a job share its *stem*. The files of this set:

```python
for name, size in te.files(SET).items():
    print(f"{size / 1e6:9.1f} MB  {name}")
```

```
    835.7 MB  AuAu_0_10_jet_seed0101.h5
      0.0 MB  AuAu_0_10_jet_seed0101.json
      0.0 MB  AuAu_0_10_jet_seed0101.xml
    263.2 MB  AuAu_0_10_jet_seed0101_hadrons.root
     45.5 MB  AuAu_0_10_jet_seed0101_hadrons_bulk_bg.h5
    229.2 MB  AuAu_0_10_jet_seed0101_hadrons_bulk_jet.h5
      0.2 MB  AuAu_0_10_jet_seed0101_hadrons_jet_frag.h5
    540.3 MB  AuAu_0_10_jet_seed0101_particlize.h5
      0.0 MB  campaign_campaign.root
```

## 2. The job record

`<stem>.json` summarises the job: its seed, what was simulated, how many events were written, and the cross section of the p̂T window.

```python
import json
job = json.loads(te.read_text(SET, STEM + ".json"))
for key in ["seed", "hard", "reuse", "events_written", "droplets_total", "wall_s",
            "pthat_bins", "pthat_bin_sigma_gen_mb", "events_hit_edge"]:
    print(f"{key:24s} {job[key]}")
```

```
seed                     101
hard                     PythiaGun pTHat 30.0-40.0 GeV, partons |y| < 0.6 (leading)
reuse                    5
events_written           5
droplets_total           88
wall_s                   42.3
pthat_bins               [[30.0, 40.0]]
pthat_bin_sigma_gen_mb   [2.093149017109464e-06]
events_hit_edge          {'bg': 0, 'jet': 0}
```

- **`reuse` 5**: the five jets share **one** background. The initial state and MUSIC_1 run once, and each jet is followed through the same medium. This saves time, and it makes the five events directly comparable.
- **`pthat_bin_sigma_gen_mb`**: PYTHIA's cross section for this p̂T window, in mb. Results from several windows are combined with these weights (tutorial 05).
- **`wall_s`**: the whole stage 1 took 42 s on an Apple M3 Max.

`<stem>.xml` is the complete X-SCAPE input of the job, verbatim. Every physics setting is there. Some of the most important ones:

```python
import xml.etree.ElementTree as ET
xml = ET.fromstring(te.read_text(SET, STEM + ".xml"))

def setting(path):
    """The text of the first element at an XPath-like path, stripped."""
    el = xml.find(path)
    return el.text.strip() if el is not None and el.text else "-"

settings = {
    "collision":              f"{setting('.//MCGlauber/projectile')}+{setting('.//MCGlauber/target')} at "
                              f"{setting('.//MCGlauber/sqrts')} GeV, b = {setting('.//MCGlauber/b_min')}-"
                              f"{setting('.//MCGlauber/b_max')} fm (centrality {setting('.//MCGlauber/cenMin')}-"
                              f"{setting('.//MCGlauber/cenMax')} %)",
    "hard scattering":        f"PYTHIA pTHat {setting('.//PythiaGun/pTHatMin')}-{setting('.//PythiaGun/pTHatMax')} GeV, "
                              f"partons |y| < {setting('.//PythiaGun/partonYMax')}",
    "hydro start tau_0":      setting(".//MUSIC/Initial_time_tau_0") + " fm/c",
    "eta/s at the kink":      f"{setting('.//MUSIC/shear_viscosity_3_at_kink')} at T = "
                              f"{setting('.//MUSIC/shear_viscosity_3_T_kink_in_GeV')} GeV",
    "zeta/s maximum":         f"{setting('.//MUSIC/bulk_viscosity_3_max')} at T = "
                              f"{setting('.//MUSIC/bulk_viscosity_3_T_peak_in_GeV')} GeV",
    "freeze-out temperature": setting(".//MUSIC/freezeout_temperature") + " GeV",
    "MATTER / LBT switch Q0":  setting(".//Matter/Q0") + " GeV",
    "alpha_s (MATTER, LBT)":  f"{setting('.//Matter/alphas')}, {setting('.//Lbt/alphas')}",
    "liquefier tau_delay":    setting(".//CausalLiquefier/tau_delay") + " fm/c",
}
for k, v in settings.items():
    print(f"{k:24s} {v}")
```

```
collision                Au+Au at 200 GeV, b = 0.0-4.7 fm (centrality 0.0-10.0 %)
hard scattering          PYTHIA pTHat 30.0-40.0 GeV, partons |y| < 0.6
hydro start tau_0        0.424 fm/c
eta/s at the kink        0.108 at T = 0.206 GeV
zeta/s maximum           0.092 at T = 0.180 GeV
freeze-out temperature   0.15 GeV
MATTER / LBT switch Q0   2.0 GeV
alpha_s (MATTER, LBT)    0.25, 0.25
liquefier tau_delay      1.0 fm/c
```

In order: the **initial state** is a 3D MC-Glauber model with strings (0–10 % central Au+Au). The **hard scattering** comes from PYTHIA. The **fluid** is MUSIC with temperature-dependent shear and bulk viscosities, which ends where the temperature falls below 150 MeV. The **jet** loses energy with MATTER at high virtuality and LBT below `$Q_0 = 2$` GeV. The **CausalLiquefier** turns the energy and momentum the jet loses into *droplets* that the fluid absorbs after a delay. Each step is explained in [The physics](https://jhputschke.github.io/js-contrib/about/physics/).

## 3. The pair file: two hydro evolutions

`<stem>.h5` holds the energy density and flow velocity of **both** hydro runs on a fixed grid in `$(x, y, \eta_s, \tau)$`: `arr` is the jet leg (MUSIC_2), `arr_bg` the background (MUSIC_1). Both start from the same initial condition, so `arr - arr_bg` is exactly what the jet did to the fluid. Tutorial 06 looks at that difference. Here we look at the layout and the per-event diagnostics.

The file is streamed: opening it and reading the diagnostics fetches a few MB of its 836 MB.

```python
import numpy as np
import pandas as pd

with te.h5(SET, STEM + ".h5") as f:
    A = dict(f.attrs)                    # a copy: the attributes stay readable after closing
    print("arr:", f["arr"].shape, "= (event, channel, x, y, eta_s, tau);",
          "channels", list(A["feature_names"]))
    print(f"grid: x, y in {A['nx']} cells of {A['dx']} fm, eta_s in {A['neta']} cells of {A['deta']},"
          f" tau from {A['tau_min']} fm/c every {A['dtau']} fm/c")
    print("chunks:", f["arr"].chunks, "-> read one event and one tau frame at a time")
    cols = ["pthat", "parton_pt_lead", "parton_pt_sub", "n_droplets", "E_droplets",
            "frames_identical", "ntau_jet", "ntau_bg"]
    diag = pd.DataFrame({c: f["diag"][c][:] for c in cols})

diag["tau_identical"] = A["tau_min"] + A["dtau"] * diag["frames_identical"]   # fm/c
diag["tau_fo_jet"] = A["tau_min"] + A["dtau"] * diag["ntau_jet"]
diag["tau_fo_bg"] = A["tau_min"] + A["dtau"] * diag["ntau_bg"]
diag.round(2)
```

```
arr: (5, 4, 64, 64, 32, 131) = (event, channel, x, y, eta_s, tau); channels ['energy_density', 'vx', 'vy', 'vz']
grid: x, y in 64 cells of 0.3875 fm, eta_s in 32 cells of 0.3125, tau from 0.5 fm/c every 0.1 fm/c
chunks: (1, 4, 64, 64, 32, 1) -> read one event and one tau frame at a time
```

|     | pthat | parton_pt_lead | parton_pt_sub | n_droplets | E_droplets | frames_identical | ntau_jet | ntau_bg | tau_identical | tau_fo_jet | tau_fo_bg |
| --- | ----- | -------------- | ------------- | ---------- | ---------- | ---------------- | -------- | ------- | ------------- | ---------- | --------- |
| 0   | 31.60 | 31.92          | 31.60         | 24.0       | 54.14      | 6.0              | 131.0    | 103.0   | 1.1           | 13.6       | 10.8      |
| 1   | 34.00 | 35.84          | 32.41         | 14.0       | 34.45      | 9.0              | 103.0    | 103.0   | 1.4           | 10.8       | 10.8      |
| 2   | 30.16 | 30.32          | 30.06         | 22.0       | 27.86      | 6.0              | 109.0    | 103.0   | 1.1           | 11.4       | 10.8      |
| 3   | 34.73 | 34.85          | 34.74         | 11.0       | 15.77      | 7.0              | 123.0    | 103.0   | 1.2           | 12.8       | 10.8      |
| 4   | 30.71 | 33.96          | 29.40         | 17.0       | 19.40      | 6.0              | 114.0    | 103.0   | 1.1           | 11.9       | 10.8      |

**What to notice.**

- **`E_droplets`**: the energy the jet handed to the fluid, from 16 to 54 GeV. The two hard partons together carry about `$2\hat p_T \approx$` 60–70 GeV, so in 0–10 % central Au+Au a large part of the jets' energy ends up in the medium. How much depends on the path through the fireball.
- **`tau_identical`**: the two legs stay bit-identical until the first droplet arrives, about 1.1–1.4 fm/c. That is the jet's start (`$\tau = 0.5$` fm/c) plus the liquefier's delay.
- **`tau_fo_jet` vs `tau_fo_bg`**: the jet leg freezes out later, by up to 2.8 fm/c. The deposited energy reheats the fluid, so its last part takes longer to cool to 150 MeV. This is why `arr - arr_bg` is only a "wake" while **both** legs still exist.

## 4. The particlize file: the input of hadronization

`<stem>_particlize.h5` holds what stage 2 needs:

- the **freeze-out surface** of each leg: a list of hypersurface cells with their position, normal vector `$d\Sigma_\mu$`, flow `$u^\mu$`, temperature and viscous corrections (32 columns per cell)
- the **final partons** of the jet
- the **shower initiators**: the partons that started the jet showers

The background's surface is stored once, and every event points to it through `events/bg_unit`.

```python
with te.h5(SET, STEM + "_particlize.h5") as f:
    print("legs:", list(f.attrs["legs"]), "| T_fo =", f.attrs["T_fo"], "GeV",
          "| file_uuid", f.attrs["file_uuid"])
    print("surface cells: jet leg", f["surface/jet/cells"].shape,
          "(all events), background", f["surface/bg/cells"].shape, "(once)")
    ev = pd.DataFrame({c: f["events"][c][:] for c in ["bg_unit", "n_cells_jet", "n_cells_bg", "n_partons"]})
    # one event's surface: the cells of event 0
    o = f["surface/jet/offsets"][:]
    cells = f["surface/jet/cells"][o[0]:o[1]]
    columns = list(f["surface/jet/cells"].attrs.get("columns", []))
ev
```

```
legs: ['jet', 'bg'] | T_fo = 0.15 GeV | file_uuid da36d68d-b279-4190-8a1d-553d1ffc7d09
surface cells: jet leg (5803677, 32) (all events), background (1151293, 32) (once)
```

|     | bg_unit | n_cells_jet | n_cells_bg | n_partons |
| --- | ------- | ----------- | ---------- | --------- |
| 0   | 0       | 1165243     | 1151293    | 49        |
| 1   | 0       | 1160069     | 1151293    | 41        |
| 2   | 0       | 1162421     | 1151293    | 56        |
| 3   | 0       | 1157113     | 1151293    | 26        |
| 4   | 0       | 1158831     | 1151293    | 41        |

```python
from jetscape.particlize_h5 import SURFACE_COLUMNS
T = cells[:, SURFACE_COLUMNS.index("T")]
tau = cells[:, SURFACE_COLUMNS.index("tau")]
at_fo = np.abs(T - 0.15) < 5e-4
print(f"event 0: {len(cells):,} surface cells, tau = {tau.min():.2f} .. {tau.max():.2f} fm/c")
print(f"  {at_fo.mean():.1%} at T = T_fo; the others: T = {T[~at_fo].min():.3f} .. {T[~at_fo].max():.3f} GeV,"
      f" all at tau = {np.unique(tau[~at_fo].round(2))} fm/c")
```

```
event 0: 1,165,243 surface cells, tau = 0.46 .. 13.38 fm/c
  98.9% at T = T_fo; the others: T = 0.120 .. 0.149 GeV, all at tau = [0.46] fm/c
```

**What to notice.**

- **Almost every cell sits at `$T = T_\text{fo} = 0.15$` GeV:** the surface is where the fluid cools through the freeze-out temperature.
- **About 1 % of the cells are colder, all at the very start of the hydro (`$\tau \approx 0.46$` fm/c).** They are fluid that was already below 150 MeV when the hydro began, at the dilute edges of the fireball (large radius or large `$|\eta_s|$`). MUSIC freezes them out at once (`Do_FreezeOut_lowtemp` in the XML), so their energy is not lost.
- **The jet leg has about 1 % more cells than the background,** from the reheated, slightly larger fluid. A leg with very different numbers would point to a broken hydro run.

## 5. The hadron files

Each hadron file holds one *tag*. The hadrons are in one flat table; the offsets mark the **units** (an event, or for `bulk_bg` a background) and their **samples** (Cooper–Frye oversamples or fragmentations):

| tag        | from                     | unit       | samples per unit here             |
| ---------- | ------------------------ | ---------- | --------------------------------- |
| `bulk_jet` | the jet leg's surface    | event      | 100                               |
| `bulk_bg`  | the background's surface | background | 500 = 100 × the 5 events using it |
| `jet_frag` | the final partons        | event      | 20                                |

Each hadron file records the `file_uuid` of the particlize file it came from, as its `source_uuid`. The readers refuse a hadron file that does not match, which catches files renamed or mixed up between runs.

```python
with te.h5(SET, STEM + "_particlize.h5") as f:
    uuid = f.attrs["file_uuid"]
for tag in ["bulk_jet", "bulk_bg", "jet_frag"]:
    with te.h5(SET, f"{STEM}_hadrons_{tag}.h5") as f:
        n_units = int(f.attrs["nunits_written"])
        samples = f["units/n_samples"][:]
        n = len(f["hadrons/pid"])
        same = f.attrs["source_uuid"] == uuid
    print(f"{tag:9s} {n_units} unit(s), samples per unit {samples.tolist()},"
          f" {n:>9,} hadrons, {n / samples.sum():7.1f} per sample, source_uuid matches: {same}")
```

```
bulk_jet  5 unit(s), samples per unit [200, 200, 200, 200, 200], 8,401,278 hadrons,  8401.3 per sample, source_uuid matches: True
bulk_bg   1 unit(s), samples per unit [200], 1,676,110 hadrons,  8380.5 per sample, source_uuid matches: True
jet_frag  5 unit(s), samples per unit [20, 20, 20, 20, 20],     2,626 hadrons,    26.3 per sample, source_uuid matches: True
```

**What to notice.** A sample of the bulk has about 8,400 hadrons, all species and charges in the full rapidity range: this is what one central Au+Au collision looks like. A jet's own fragmentation gives about 26 hadrons per sample. The jet leg's bulk has slightly *more* hadrons than the background, because the deposited energy is also turned into hadrons. That excess is the hadronic wake of tutorial 07.

## 6. The ROOT file

`run_h5toROOT.py` puts the three hadron tags and the event information into one ROOT file per job (RNTuple), plus a campaign file with the cross sections. uproot reads it without ROOT, also straight from the OSDF:

```python
import uproot
with uproot.open(te.data(SET, STEM + "_hadrons.root")) as rf:
    print("objects:", rf.keys())
    print("bulk_jet: one entry per sample ->", rf["bulk_jet"].num_entries, "entries")
    events = pd.DataFrame(rf["events"].arrays(["event", "pthat", "E_droplets", "n_samples_jet",
                                               "n_samples_bg"], library="np"))
    windows = pd.DataFrame(rf["windows"].arrays(library="np"))
events
```

```
objects: ['events;1', 'windows;1', 'provenance;1', 'bulk_jet;1', 'bulk_bg;1', 'jet_frag;1']
bulk_jet: one entry per sample -> 1000 entries
```

|     | E_droplets | event | pthat     | n_samples_jet | n_samples_bg |
| --- | ---------- | ----- | --------- | ------------- | ------------ |
| 0   | 54.135521  | 0     | 31.599779 | 200           | 200          |
| 1   | 34.446343  | 1     | 33.995441 | 200           | 200          |
| 2   | 27.856571  | 2     | 30.156334 | 200           | 200          |
| 3   | 15.765037  | 3     | 34.729335 | 200           | 200          |
| 4   | 19.397569  | 4     | 30.714280 | 200           | 200          |

```python
windows
```

|     | window | pthat_lo | pthat_hi | sigma_mb | sigma_err_mb | acceptance | n_tried | n_kept | n_accepted | seed |
| --- | ------ | -------- | -------- | -------- | ------------ | ---------- | ------- | ------ | ---------- | ---- |
| 0   | 0      | 30.0     | 40.0     | 0.000002 | 0.000001     | 0.833333   | 6       | 5      | 6          | 101  |

The ROOT file has the same events and the same window cross section as the HDF5 files. The tutorials 09 and 10 work with ROOT files in Python and in C++.

## 7. Before combining files: the settings

Files can look alike and still differ in how they were made. Before you combine files from different jobs or campaigns, compare their settings. All of them are recorded in the attributes of the pair and particlize files:

```python
import yaml

def production_settings(h5file):
    """The settings that must agree between files you combine."""
    A = h5file.attrs
    xml = ET.fromstring(A["prod_user_xml"])
    late = xml.findtext(".//Hydro/MUSIC/late_delta_tau")
    grid = yaml.safe_load(A["prod_grid_yaml"])["grid"]
    return {
        "hard process": A["prod_hard"],
        "background reuse": int(A["prod_reuse"]),
        "MUSIC late time step": late.strip() if late else "absent (constant step)",
        "output grid x": grid["x"],
        "EoS": A["eos_kind"],
        "build": te.short(A["prod_build"]),
    }

with te.h5(SET, STEM + ".h5") as f:
    mine = production_settings(f)
for k, v in mine.items():
    print(f"{k:22s} {v}")
```

```
hard process           PythiaGun pTHat 30.0-40.0 GeV, partons |y| < 0.6 (leading)
background reuse       5
MUSIC late time step   0.05
output grid x          {'min': -12.20625, 'max': 12.20625, 'n': 64}
EoS                    hotqcd (MUSIC EOS 91)
build                  ~/JetScape/X-SCAPE/build_gpu
```

### Check your result

```python
exp = json.load(open("expected/04_files_and_provenance.json"))
assert job["events_written"] == exp["events"] and int(job["reuse"]) == exp["reuse"]
np.testing.assert_allclose(diag["E_droplets"], exp["E_droplets"], rtol=1e-5)
np.testing.assert_array_equal(diag["ntau_jet"], exp["ntau_jet"])
np.testing.assert_array_equal(ev["n_cells_jet"], exp["n_cells_jet"])
assert at_fo.mean() > 0.98 and np.all(tau[~at_fo] < 0.5)
print("All checks pass.")
```

```
All checks pass.
```

## Exercises

1. **Which jet lost the most energy?** Event 0 deposited 54 GeV. Look at its partons' `$p_T$` and rapidity in `diag` (`parton_pt_lead`, `parton_y_lead`, ...). Is there a simple reason? (Tutorial 06 shows its path through the medium.)
1. **Another job.** Compare `production_settings` with the older example event in the set `auau_jet_event` (file `AuAu_0_10_jet_seed0003.h5`). Which settings differ? Could the two be combined in one analysis?
1. **The surface.** Histogram the `$\tau$` of event 0's surface cells (`tau` above) for the jet leg, and the same for the background (`surface/bg`). Where do the extra cells of the jet leg come from?

## Next steps

- The wake in the fluid, from the pair file: tutorial 06.
- The wake in hadrons, from the hadron files: tutorial 07.
- Every dataset and attribute: [What a production writes](https://jhputschke.github.io/js-contrib/analysis/files/).
- Inspect any file from a terminal: `python utils/h5_inspect.py FILE` in js-contrib.
