# O+O with initial-state radiation

`prod_OO_isr` runs O+O collisions at 200 GeV with X-SCAPE's initial-state radiation (ISR)
workflow: the hard scattering's incoming partons are showered backward in time by iMATTER,
the 3D MC-Glauber strings are made from what the hard scattering left, MUSIC evolves that
medium, and Matter + LBT shower the jets through it. The jet hadrons come from
iColoredHadronization, which follows the colour flow of the ISR and the beam remnants; the
bulk hadrons from iSS. Use it for small-system studies where the hard process and the bulk
come from the same nucleon–nucleon collisions: multiplicity and centrality biases of
hard-triggered events, flow, jets in a small medium. The tuned O+O 3D MC-Glauber and MUSIC
parameters are those of X-SCAPE's O+O example, tuned at 200 GeV.

## What is different from Au+Au

```text
IS: 3dMCGlauber (O16, positions only)
 -> Hard: PythiaIsrGun at one binary collision
          -> IsrManager / iMATTER       ISR, backward in time
          -> MCGlauberGenString         strings, the hard partons' momentum subtracted
 -> MUSIC                              ONE hydro run per event
 -> Eloss: Matter + LBT in the medium   the final-state shower
 -> JetHadronization: iColored          jet hadrons, in the job
 -> SoftParticlization: iSS             bulk hadrons, in the job or later
```

- **One MUSIC run per event, no pair.** The medium is made from what each event's hard
  scattering left (its partons' momentum is subtracted from the colliding nucleons), so a
  medium belongs to one hard event: there is no background leg, no jet − background, no
  hydro reuse, and no jet deposition into the fluid.
- **Every event is hard-triggered:** exactly one hard scattering, at a binary collision drawn
  uniformly. `glauber_ncoll` is recorded per event to weight the geometry
  ([O+O analysis](../analysis/oo.md)). Centrality comes from the impact parameter: the
  default b < 3 fm is about 0–24 % of the O+O cross section.
- **The jet hadrons are made in the job.** iColoredHadronization needs the event's beam
  remnants and ISR colour connections, which exist only while the job runs. The bulk can be
  made in the job or later from stored surfaces.
- **Events without a medium:** when the hard collision is the only one, no string is made;
  Matter showers in vacuum and the bulk unit is empty (`has_medium` 0).
- **The cores set the rate.** An O+O event is small (MUSIC's grid ±10.5 fm), so an O+O job
  needs ~1.4 GB and ~6 core-s per event; the GPU is far from full.

## Running it

The driver is `run_prod_oo_isr.py` (one job), with the same `run_jobs.sh` as the Au+Au
productions: the same seeds, campaign names, seed registry and restarts
([Au+Au jets](auau-jet.md#run_jobssh)).

```bash
python run_prod_oo_isr.py --events 1 --seed 1 --dry-run      # check the XML/grid, print the plan
python run_prod_oo_isr.py --events 10 --seed 1               # pT-hat 20-40 GeV, b < 3 fm
python run_prod_oo_isr.py --events 30 --seed 1 --pthat-bins 10-20,20-30,30-40 --parton-ymax 0.6
./run_jobs.sh 20 25 0                                        # 20 jobs x 25 events, unique seeds
./run_jobs.sh -j 2 20 25 1 out_had --bulk-hadrons 100 --no-particlize
```

The image starts in the Au+Au folder, so point it at the O+O one, or call the scripts by
their full path (they find their XML and grid next to themselves):

=== "Docker"

    ```bash
    OO=/opt/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape/example/prod_OO_isr
    docker run --rm --gpus all --user "$(id -u):$(id -g)" \
      -e OMP_NUM_THREADS=3 -e OMP_WAIT_POLICY=passive -e MUSIC_CUDA_SYNC=block \
      -v "$HOME/prod:/work" -w "$OO" jhputschke/xscape-prod:cu126 \
      ./run_jobs.sh -j 8 --mps 32 60 0 /work/OO_a --bulk-hadrons 100 --no-particlize --no-hydro-file
    ```

=== "Apptainer"

    ```bash
    OO=/opt/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape/example/prod_OO_isr
    apptainer exec --nv --env OMP_NUM_THREADS=3,OMP_WAIT_POLICY=passive,MUSIC_CUDA_SYNC=block \
      --pwd "$OO" --bind "$SCRATCH/prod:/work" xscape_prod.sif \
      ./run_jobs.sh -j 8 32 60 0 /work/OO_a --bulk-hadrons 100 --no-particlize --no-hydro-file
    ```

=== "Mac"

    ```bash
    cd ~/xscape/examples/prod_OO_isr
    python run_prod_oo_isr.py --events 2 --seed 5 --outdir ~/xscape/prod/out_oo
    OMP_NUM_THREADS=5 OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0 \
      ./run_jobs.sh -j 3 --campaign oo_a 20 25 0 ~/xscape/prod/OO_a --bulk-hadrons 100 --no-particlize
    ```

That is 32 jobs × 60 events, 8 at a time on one GPU, with all hadrons made in the jobs and
nothing else kept. Write the output under the bound `/work`: the default `./out` is inside
the read-only image. Nothing needs to be set for iMATTER in the container (the image sets
`PYTHIA8DATA`), and the LBT table cache and `--stagger` don't apply to these small jobs.

## Hadronization: in the job or afterwards

| setup | options | files | afterwards |
|---|---|---|---|
| **A. everything in the job** | `--bulk-hadrons 100 --no-particlize` | `<stem>.h5`, `_hadrons_bulk_bg.h5`, `_hadrons_jet_frag.h5` | nothing to do |
| **B. in the job, surfaces kept** | `--bulk-hadrons 100` | A + `_particlize.h5` | more oversamples or other iSS settings any time |
| **C. bulk afterwards** (the default) | none | `<stem>.h5`, `_particlize.h5`, `_hadrons_jet_frag.h5` | `hadronize.py --tags bulk_bg` |

Add `--no-hydro-file` to any of them to leave out the hydro evolution (MUSIC still runs and
Matter still reads it). A + `--no-hydro-file` is a hadrons-only job: the usual production
setting, at ~0.13 MB per event with 10 oversamples.

For setup C, the bulk is made on CPUs with the Au+Au production's `hadronize.py`, **always
with `--tags bulk_bg`** (the jet hadrons were made in the job; it refuses `jet_frag` on these
files):

```bash
python ../prod_AuAu_0_10_jet/hadronize.py out/OO_isr_seed0001_particlize.h5 \
       --tags bulk_bg --oversample 100
python ../prod_AuAu_0_10_jet/run_hadronize.py out -j 4 --tags bulk_bg --oversample 100
```

In the container, with `-w "$OO"` (Docker) or `--pwd "$OO"` (Apptainer) and without
`--gpus`/`--nv`: `../prod_AuAu_0_10_jet/run_hadronize.py /work/OO_a -j 8 --tags bulk_bg --oversample 100`.
On SLURM: `PROD=oo` for `utils/slurm_hadronize.sh` ([Clusters](clusters.md#stage-2-on-cpu-nodes)).

The bulk made in the job and afterwards are sampled alike (both use `config/hadronize.xml`);
only the seeds differ. An event at hadron level is unit *i* of the jet file plus unit *i* of
the bulk file; the bulk's N oversamples share one fluid.

## Settings

| option | default | |
|---|---|---|
| `--pthat-min`, `--pthat-max` | 20, 40 GeV | one pT̂ window |
| `--pthat-bins A-B,C-D,…` | | K windows in one job, one Pythia each, event *i* in window *i* mod K; no background is shared. Use a multiple of K events |
| `--parton-ymax Y`, `--parton-y-mode {both,any}` | off, `both` | regenerate the event until the two outgoing partons of the hard scattering have \|y\| < Y; cross sections corrected by the acceptance |
| `--b-min`, `--b-max` | 0, 3 fm | the impact-parameter range (centrality) |
| `--sqrts` | 200 GeV | the collision energy (the parameters are tuned at 200 GeV) |
| `--bulk-hadrons N` | 0 | iSS in the job, N oversamples per event |
| `--no-particlize`, `--no-hydro-file`, `--no-jet-hadrons` | | leave out the particlize file, the hydro file, the jet hadrons |
| `--eta-max`, `--charged`, `--no-x`, `--keep-bits-p/-x` | | store less, as for `hadronize.py`, on both in-job hadron files |
| `--keep-primaries N` | | with `--bulk-hadrons`: also the undecayed bulk hadrons of the first N oversamples, for SMASH later |
| `--colour-closure MODE` | the XML's (`reconnect`) | how iColored closes colour lines that end in the medium: `reconnect`, `reconnect_global`, `thermal`, `fake`, `none` |
| `--matter-only` | | Matter alone, no LBT and no recoils: a reference for colour studies |
| `--store-hadronization-input` | | keep what was handed to Pythia, so `rehadronize_jets.py` can redo the jet hadrons later |
| `--no-subtraction` | | the reference without the hard-momentum subtraction (counts the hard pair's energy twice, on purpose); keep it in its own OUTDIR |
| `--late-dtau 0` | late step on | MUSIC at a constant 0.02 fm/c; keep one setting per campaign |
| `--max-skipped N` | 3 | stop the job when more events fail |

!!! note "Choices that change the jets of a seed"
    X-SCAPE seeds each module from the order it was made. `--bulk-hadrons` and
    `--no-hydro-file` change that order, so the same seed gives other, equally valid jet
    showers (the hydro is identical). Keep one choice per campaign if jets of different jobs
    are to be compared event by event.

Before combining files, check the settings every file records (subtraction, b range,
late time step, energy loss, colour closure): [O+O analysis](../analysis/oo.md). Every
option and the physics notes: the folder's
[README](https://github.com/jhputschke/js-contrib/blob/main/contribs/PyJetscape/example/prod_OO_isr/README.md).

## Output and throughput

Each job writes (stem `OO_isr_seedNNNN`, or `OO_isr_<campaign>_NNNN`):

| file | content | size per event |
|---|---|---|
| `<stem>_hadrons_jet_frag.h5` | the jet hadrons (dijet, ISR emissions, Pythia's MPI partons, the hard pair's remnants), one unit per event, with the whole event record in `units/` | small |
| `<stem>_hadrons_bulk_bg.h5` | the bulk, N oversamples per event | ~13 kB per oversample |
| `<stem>_particlize.h5` | MUSIC's freeze-out surface, the final partons, the initiators | ~10 MB |
| `<stem>.h5` | MUSIC's evolution (e, vx, vy, vz) on `config/grid_oo.yaml` (64 × 64 × 32, x, y ±9.84 fm), with `diag/` per event | ~17 MB |
| `<stem>.xml`, `.json`, `.log` | the job XML, the summary (cross sections, skipped events), the log | |

A module that throws inside an event costs that event only; the `.json` lists it under
`events_skipped`.

Measured on 2 × RTX 3090 (24 cores), 8 jobs per GPU with 3 threads, MPS,
`MUSIC_CUDA_SYNC=block`, 60 events per job:

| output | events/h | per job | disk per event |
|---|---|---|---|
| hadron files only (`--bulk-hadrons 10 --no-particlize --no-hydro-file`) | **13,728** | 1.43 GB | 0.13 MB |
| full outputs (hydro, particlize, jet hadrons) | 12,700 | 1.40 GB | 28 MB |

The cores set the rate (~570 events/h per core). Full outputs write ~360 GB/h, more than an
OSDF upload takes: use hadron files only, or a large local disk. On two GPUs,
`utils/launch_2gpu_oo.sh` runs one container per GPU with these settings, and on SLURM
`PROD=oo` selects this production in `slurm_prod_array.sh`: [Clusters](clusters.md).

## Next steps

- [O+O analysis](../analysis/oo.md): the per-event record, cross-section and N_coll weights,
  multiplicity, centrality, flow from the oversamples, jets.
- [`prod_OO_isr/oo_isr_overview.ipynb`](https://github.com/jhputschke/js-contrib/blob/main/contribs/PyJetscape/example/prod_OO_isr/oo_isr_overview.ipynb):
  one job's medium and hadrons (needs Jupyter: the analysis venv or the container's
  `xscape-jupyter`).
- [Your first event](first-event.md): the same chain from Python with `oo_workflow.py`.
