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

## The p+p reference: `--vacuum`

`--vacuum` runs the same chain without a medium: no pre-equilibrium, MUSIC or LBT, Matter in
vacuum, no hydro or particlize file. 3D MC-Glauber still runs, because PythiaIsrGun puts the
hard scattering at a binary collision and iColoredHadronization takes the beam remnants from
it. p+p and O+O are thus hadronized the same way, which is what R_AA needs, and the jet file
has the same per-event record and per-window cross sections as the O+O one, so the same
analysis reads both.

```bash
python run_prod_oo_isr.py --vacuum --user-xml ../config/pp200_isr_vacuum.xml \
       --events 2100 --seed 1 --pthat-bins 5-7,7-9,9-11,11-15,15-20,20-30,30-60 \
       --pythia-line "SigmaProcess:alphaSvalue = 0.13" \
       --pythia-line "MultipartonInteractions:alphaSvalue = 0.13" \
       --store-hadronization-input --eloss-threads 0
./run_jobs.sh -j 16 40 2100 0 out_pp --vacuum --user-xml ../config/pp200_isr_vacuum.xml \
       --pthat-bins 5-7,7-9,9-11,11-15,15-20,20-30,30-60 --eloss-threads 0
python run_prod_oo_isr.py --vacuum --events 20 --seed 1 --pthat-bins 10-20,20-40
            # O+O (the default XML) without a medium: the vacuum reference of the same system
```

- **`config/pp200_isr_vacuum.xml`**: p+p at 200 GeV with the O+O 3D MC-Glauber settings,
  pT̂ 5–60 GeV, `MultipartonInteractions:pTmin = 5.0`, Matter `Q0` 2 and `vir_factor` 0.25
  (iMATTER reads both there), iColored. The windows start at 5 GeV, above iMATTER's floor
  Q0/√vir_factor (4 GeV at 0.25): below it a hard scattering gets no initial-state radiation.
  Copy it to change `Q0` or `vir_factor`.
- **Pythia settings from the command line**: `--pythia-line` goes to the hard process
  (α_s, the MPI cut, the PDF), `--hadronization-line` to iColored's Pythia (e.g.
  `"StringFlav:probStoUD = 0.25"`). Both repeat and are recorded in the files.
- **Fragmentation settings afterwards**: with `--store-hadronization-input`,
  `rehadronize_jets.py --line "StringFlav:probStoUD = 0.3"` redoes the jet hadrons of the same
  showers with that setting; with the stored seeds the result is exactly that of a job run with
  it. A fragmentation parameter can be scanned without rerunning the events.
- **Output**: only `<proj><targ>_isr_vacuum_seedNNNN_hadrons_jet_frag.h5` and the `.json`.
  `HadronFileReader(OUTDIR)` reads the campaign (the jet file is marked as a hadrons-only
  job's, like every job without a particlize file), with the windows and
  `pthat_bin_sigma(k)`. Weight with the per-window cross sections as for O+O
  ([O+O analysis](../analysis/oo.md)); every event is one hard scattering, no `glauber_ncoll`
  weight in p+p.
- **Speed**: about 11 ms per event on one core (GB10, seven windows 5–60 GeV, about 4.6 ms at
  5–7 GeV), the same as `runJetscape` on the job's XML. Set `OMP_NUM_THREADS=1` and
  `--eloss-threads 0` and run one job per core; no GPU is used.

## Au+Au with the same chain

The O+O chain runs Au+Au too, with one MUSIC run per event and no hydro reuse, so jets and
the medium come from the same nucleon–nucleon collisions as in O+O: ISR, strings, iColored with
the real beam remnants. Use it to test an O+O setting (energy loss, hadronization) against Au+Au
data. Add to the O+O command:

```bash
python run_prod_oo_isr.py --user-xml ../config/OO_MCGlauber_MUSIC_isr_rhictune.xml \
    --events 50 --seed 1 --pthat-bins 23-30 --b-min 0 --b-max 4.7 \
    --xml-set=IS/MCGlauber/projectile=Au --xml-set=IS/MCGlauber/target=Au \
    --xml-set=IS/MCGlauber/nucleon_configuration_from_file=0 \
    --xml-set=IS/grid_max_x=13 --xml-set=IS/grid_max_y=13 \
    --parton-ymax 1.0 --parton-y-mode any \
    --no-particlize --no-hydro-file --eta-max 5.2 \
    --store-partons --store-hadronization-input --outdir OUT
```

- `--b-max 4.7` is 0–10 % of the Au+Au cross section (geometric); the generated nucleon
  positions replace O+O's from file, and MUSIC's grid is ±13 fm.
- `--store-partons --store-hadronization-input` keep the holes and the hadronization input, for
  the hole subtraction and for re-hadronizing the same partons later.
- An Au+Au event is much heavier than an O+O one: about 3,000 events/h on 2 × RTX 3090
  (10 jobs per GPU), against O+O's 13,700.

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
| `--flush-every N` | 100 with `--vacuum`, else 1 | commit the in-job hadron files every N events (offsets, per-event record, flush) instead of every event: the same file, much faster for events of a few ms; after a crash every committed event reads back |
| `--vacuum` | | no medium: the [p+p reference](#the-pp-reference-vacuum), or O+O without a medium |
| `--pythia-line LINE`, `--hadronization-line LINE` | | add a Pythia setting to the hard process or to iColored's Pythia (repeatable, recorded) |
| `--keep-primaries N` | | with `--bulk-hadrons`: also the undecayed bulk hadrons of the first N oversamples, for SMASH later |
| `--colour-closure MODE` | the XML's (`reconnect`) | how iColored closes colour lines that end in the medium: `reconnect`, `reconnect_global`, `thermal`, `fake`, `none` |
| `--matter-only` | | Matter alone, no LBT and no recoils: a reference for colour studies |
| `--store-hadronization-input` | | keep what was handed to Pythia, so `rehadronize_jets.py` can redo the jet hadrons later: decays off, new seeds, other Pythia settings (`--line`) |
| `--no-subtraction` | | the reference without the hard-momentum subtraction (counts the hard pair's energy twice, on purpose); keep it in its own OUTDIR |
| `--late-dtau 0` | late step on | MUSIC at a constant 0.02 fm/c; keep one setting per campaign |
| `--eloss-threads 0` | -1: in parallel | the final-state showers of an event run in parallel on `OMP_NUM_THREADS` threads (set it: unset means all cores per job): events ~13 % faster (GB10, 4 threads). 0 runs them serially, N on N threads. Serial and parallel give other showers for a seed, the thread count does not: one setting per campaign |
| `--active-region 1e-5` | off (0) | the GPU evolves only the region with matter and keeps the vacuum frozen (CUDA and Metal): GPU kernel time ×0.47, job −19 % alone (GB10). The wake energy changes by ~0.8 % per event in Au+Au, unbiased; hadron-level checks are still open. One setting per campaign: [the active region](../about/gpu.md#the-active-region-opt-in) |
| `--no-hydro-fixes` | fixes on | MUSIC4GPU's [conservation fixes](../about/gpu.md#the-conservation-fixes-on-by-default) off (the old rules, the default before 2026-11-17): only to continue a campaign begun before them |
| `--max-skipped N` | 3 | stop the job when more events fail |

!!! note "Choices that change the jets of a seed"
    X-SCAPE seeds each module from the order it was made. `--bulk-hadrons` and
    `--no-hydro-file` change that order, so the same seed gives other, equally valid jet
    showers (the hydro is identical). So do parallel showers, this driver's default, against
    `--eloss-threads 0`: every shower gets streams of its own (the thread count does not
    matter). Keep one choice per campaign if jets of different jobs are to be compared event
    by event; the files record it as `prod_eloss_threads`.

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
