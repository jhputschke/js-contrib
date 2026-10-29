# Your first event

Before a campaign, run one event through the whole chain and look at what comes out. This
page does that twice: first one O+O event at 200 GeV with the Pythonic example
`oo_workflow.py` (3D MC-Glauber, Pythia with initial-state radiation, MUSIC on the GPU,
Matter + LBT, iColored and iSS: under a minute), then two Au+Au jet events with the
production driver of the [two-stage production](auau-jet.md). You need the
[production container](../install/container.md), the [xscape wheel](../install/mac.md) on
a Mac, or a [source build](../install/source.md).

## 1. One O+O event

`oo_workflow.py` reads a user XML (default: `oo_workflow.xml` next to it, made from the O+O
production's template), runs the events and writes the bulk and jet hadrons of each event to
two hadron files. The examples can't run inside the read-only image or the installed wheel
(they write next to themselves), so `xscape-examples` first copies them into a directory of
yours.

=== "Docker"

    ```bash
    mkdir -p work
    docker run --rm --gpus all --user "$(id -u):$(id -g)" -v "$PWD/work:/work" -w /work \
      jhputschke/xscape-prod:cu126 bash -c \
      'xscape-examples /work/examples && cd /work/examples/pythonic &&
       python oo_workflow.py --events 1 --outdir /work/out_oo'
    ```

    The hadron files land in `./work/out_oo/`. `--gpus all` gives the container the GPU and
    `--user` makes the files yours. Other CUDA versions (`cu130` for Blackwell and GH200,
    `cu124` for older drivers): [Production container](../install/container.md).

=== "Apptainer"

    ```bash
    apptainer pull xscape_prod.sif docker://jhputschke/xscape-prod:cu126
    W=$SCRATCH/xscape; mkdir -p $W
    apptainer exec --bind $W:/work xscape_prod.sif xscape-examples /work/examples
    apptainer exec --nv --pwd /work/examples/pythonic --bind $W:/work xscape_prod.sif \
      python oo_workflow.py --events 1 --outdir /work/out_oo
    ```

    Apptainer needs `--nv` for the GPU and `--pwd` for the working directory: it ignores the
    image's own and starts where you launched it. On a cluster run this on a GPU node (in a
    batch script, or after `salloc`).

=== "Mac"

    ```bash
    source ~/.venvs/xscape-0.1.0/bin/activate      # the environment of the installation
    xscape-examples ~/xscape/examples
    cd ~/xscape/examples/pythonic
    python oo_workflow.py --events 1 --outdir ~/xscape/out_oo
    ```

    MUSIC runs on the Mac's GPU (Metal) without any setting; the event takes under a minute
    (27 s on an M3 Max). The LBT tables must have been fetched once (`xscape-fetch-data lbt`,
    part of the [installation](../install/mac.md)).

=== "Source build"

    ```bash
    cd X-SCAPE/external_packages/js-contrib/contribs/PyJetscape/example/pythonic
    python oo_workflow.py --events 1 --outdir out_oo
    ```

    In the environment PyJetscape was built in, with the `jetscape` package importable
    ([Source build](../install/source.md)). The script finds the X-SCAPE tree it sits in and
    runs the build `X-SCAPE/build_gpu`, or the one `$JETSCAPE_BUILD` names (`--build DIR`
    overrides both).

The options of `oo_workflow.py`:

| option | default | |
|---|---|---|
| `--xml FILE` | `oo_workflow.xml` | the user XML: O+O 200 GeV, b < 3 fm, pT̂ 20–40 GeV, 20 iSS oversamples per event |
| `--events N` | the XML's (3) | events |
| `--seed S` | the XML's (7) | the framework seed |
| `--outdir DIR` | `./out_oo` | output directory |
| `--name NAME` | the XML's file name (`oo_workflow`) | the file stem |
| `--build DIR` | `$JETSCAPE_BUILD`, else `$XSCAPE/build_gpu` | the X-SCAPE build (leave it alone in the container and the wheel) |
| `--keep-workdir` | off | keep the job's working directory `OUTDIR/work/<name>` with the modules' side files |

### What it prints and writes

Between the modules' own log lines (MUSIC, iSS, LBT, Matter), the script prints one line per
event and a summary. One event on an M3 Max (source build) gave:

```text
oo_workflow: /…/out_oo/oo_workflow.xml
event 0: 17 final partons, 157350 surface cells, 8361 bulk hadrons, 116 jet hadrons, 15.7 s
oo_workflow: 1 event(s) written, skipped none
  /…/out_oo/oo_workflow_hadrons_jet_frag.h5
  /…/out_oo/oo_workflow_hadrons_bulk_bg.h5
```

The bulk-hadron count is summed over the event's 20 oversamples. In `out_oo/`:

| file | content |
|---|---|
| `oo_workflow.xml` | the job's XML as run, with `--events` and `--seed` applied |
| `oo_workflow_hadrons_bulk_bg.h5` | the bulk: iSS on MUSIC's freeze-out surface, one sample per oversample, with the event record (pT̂, the Glauber numbers, seeds) per unit |
| `oo_workflow_hadrons_jet_frag.h5` | the jet hadrons of iColoredHadronization, one sample per event, with the partons that started the showers (`initiators/`) |

A module that throws inside an event costs that event only: it is listed as skipped, and the
run goes on.

### A look at the hadrons

The files are in the `jetscape.hadrons_h5` format of all productions. In the same environment
(or the [analysis environment](../install/analysis-env.md), which needs no X-SCAPE):

```python
import awkward as ak
import jetscape as js

jet = js.HadronArray.from_h5("out_oo/oo_workflow_hadrons_jet_frag.h5")
bulk = js.HadronArray.from_h5("out_oo/oo_workflow_hadrons_bulk_bg.h5")
print(jet, "|", bulk)          # HadronArray(116 hadrons in 1 samples) | HadronArray(8361 hadrons in 20 samples)
print(jet.info["units"]["pthat"])                 # the event record: pT-hat, Glauber numbers, seeds

b = bulk.to_awkward(by_unit=True)                 # events x oversamples x hadrons
ch = b[js.select(b, "charged")]
print(ak.mean(ak.num(ch[abs(ch.eta) < 0.5], axis=2)))   # dN_ch/deta at |eta| < 0.5, averaged over oversamples
```

The oversamples of one event share one fluid: average over them, they are not independent
collisions. The notebook version of the run,
[`pythonic/oo_workflow.ipynb`](https://github.com/jhputschke/js-contrib/blob/main/contribs/PyJetscape/example/pythonic/oo_workflow.ipynb),
goes on from here: awkward views, PDG selections, spectra with errors, FastJet on every event
and every oversample in one call. The interface is described in
[The Pythonic interface](../python/pythonic.md); the O+O analysis in [O+O](../analysis/oo.md).

!!! tip "Make it yours"
    `oo_workflow.py` is also a template: change the XML (`--xml my_job.xml`) and the per-event
    block that decides what is stored. For a production with campaigns, seeds and restarts,
    use the O+O production itself: [O+O with ISR](oo-isr.md).

## 2. Two Au+Au jet events

Now the main production's driver, `run_prod_jet.py`: two 0–10 % Au+Au events at 200 GeV,
each a background and a jet leg (two MUSIC runs), with the freeze-out surfaces and the
partons stored for hadronization (`--write-particlize both`). Then stage 2 turns them into
hadrons on the CPU.

=== "Docker"

    ```bash
    IMG=jhputschke/xscape-prod:cu126
    mkdir -p prod
    # stage 1 (Docker starts in prod_AuAu_0_10_jet)
    docker run --rm --gpus all --user "$(id -u):$(id -g)" -v "$PWD/prod:/work" $IMG \
      python run_prod_jet.py --events 2 --seed 1 --write-particlize both --outdir /work/out
    # stage 2: no GPU
    docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/prod:/work" $IMG \
      python run_hadronize.py /work/out -j 4 --oversample 500 --n-frag 50
    ```

=== "Apptainer"

    ```bash
    EX=/opt/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape/example
    W=$SCRATCH/xscape; mkdir -p $W
    # stage 1
    apptainer exec --nv --pwd $EX/prod_AuAu_0_10_jet --bind $W:/work xscape_prod.sif \
      python run_prod_jet.py --events 2 --seed 1 --write-particlize both --outdir /work/out
    # stage 2: no GPU, no --nv
    apptainer exec --pwd $EX/prod_AuAu_0_10_jet --bind $W:/work xscape_prod.sif \
      python run_hadronize.py /work/out -j 4 --oversample 500 --n-frag 50
    ```

=== "Mac"

    ```bash
    cd ~/xscape/examples/prod_AuAu_0_10_jet
    # stage 1: about 40 s on an M3 Max
    python run_prod_jet.py --events 2 --seed 1 --write-particlize both --outdir ~/xscape/prod/out
    # stage 2, on the CPU
    python run_hadronize.py ~/xscape/prod/out -j 4 --oversample 500 --n-frag 50
    ```

=== "Source build"

    ```bash
    cd X-SCAPE/external_packages/js-contrib/contribs/PyJetscape/example/prod_AuAu_0_10_jet
    python run_prod_jet.py --events 2 --seed 1 --write-particlize both --outdir out
    python run_hadronize.py out -j 4 --oversample 500 --n-frag 50
    ```

!!! warning "Put the output one level inside your directory"
    Every job records its seed in `seeds_used.tsv` in the **parent** of its output directory.
    So write to `/work/out` (or `~/xscape/prod/out`), not to `/work` itself, whose parent is
    inside the read-only image.

With an explicit `--seed 1` the stem is `AuAu_0_10_jet_seed0001`. Stage 1 writes
`AuAu_0_10_jet_seed0001.h5` (the hydro pair: the jet leg `arr` and the background `arr_bg`),
`AuAu_0_10_jet_seed0001_particlize.h5` (both freeze-out surfaces and the final partons), the
job's `.xml` and `.json`. Stage 2 writes next to the particlize file:

| file | hadrons |
|---|---|
| `<stem>_hadrons_bulk_jet.h5` | iSS on the jet leg's surface: the bulk with the jet's wake |
| `<stem>_hadrons_bulk_bg.h5` | iSS on the background's surface |
| `<stem>_hadrons_jet_frag.h5` | the jet partons, fragmented by Pythia |

The difference between the first two is the medium's response to the jet, at hadron level.
Read the three together with `HadronFileReader`:
[Hadrons](../analysis/hadrons.md).

!!! note "A first look at the hydro pair"
    `arr - arr_bg` is the jet's effect on the medium and nothing else: both legs start from the
    same initial condition and are bit-identical until the first droplet deposits. With h5py:
    `f["arr"][0] - f["arr_bg"][0]` for event 0 (channels energy density, vx, vy, vz). More in
    [Hydro and hadrons together](../analysis/hydro-hadrons.md).

## Next steps

- [Au+Au jets (two stages)](auau-jet.md): from these two events to a campaign.
- [O+O with ISR](oo-isr.md): the O+O production behind `oo_workflow.py`.
- [Notebooks](../analysis/notebooks.md): the analysis notebooks to run on these files.
