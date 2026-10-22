# prod_OO_isr — O+O with initial-state radiation and jet energy loss, straight to HDF5

This folder runs O+O 200 GeV events with X-SCAPE's initial-state radiation (ISR) workflow:
the hard scattering's initial-state partons are showered backward in time by iMATTER, the
3D MC-Glauber strings are made from what the hard scattering left, MUSIC (music4gpu) evolves
that medium, and Matter showers the jet through it. There is **one** MUSIC run per event (no
two-stage hydro, no jet deposition). The hadrons can be made in the same job, or the bulk
later from stored freeze-out surfaces (see [Hadronization](#hadronization-in-the-job-or-afterwards)).

```
IS: 3dMCGlauber (O16, positions only)
 -> Hard: PythiaIsrGun at one binary collision
          -> IsrManager / iMATTER       ISR, backward from 0 to -maxT
          -> MCGlauberGenString         strings, the hard partons' momentum subtracted
 -> NullPreDynamics -> MUSIC            the medium
 -> Eloss: Matter in MUSIC's medium     the final-state shower, from -maxT
 -> JetHadronization: iColored          jet hadrons (final partons + beam remnants)
 -> SoftParticlization: iSS             bulk hadrons, only with --bulk-hadrons
```

> **Analysing the output:** [`docs/README_OO.md`](../../../../docs/README_OO.md) (the per-event
> record, cross-section and Ncoll weights, multiplicity, centrality, flow, jets, ROOT).

The chain is what X-SCAPE's `examples/custom_examples/PythiaIsrMUSIC.cc` builds by hand.
Here X-SCAPE builds it from the XML: a `<Hard><ISR>` block (next to `<PythiaGun>`) switches
the automatic task list to it.

> **Status (2026-10-10, branches `oo_isr_test` in X-SCAPE and js-contrib).** Runs on
> `build_gpu` (music4gpu, CUDA, GB10). It needs X-SCAPE's `<Hard><ISR>` task list,
> `<JetHadronization><name>iColored` and the iSS fix for events without a medium (all on
> branch `oo_isr_test`).
>
> **Tuned O+O parameters.** The 3dMCGlauber and MUSIC settings in
> `OO_MCGlauber_MUSIC_isr.xml` are the tuned O+O set of X-SCAPE's
> `config/BulkFastTest/OO_one_event.xml` (O16 configurations from `nucleusConfigs`,
> `tau_form_mean` 0.5, MUSIC's viscosities and freeze-out, EOS 91), tuned at √s_NN =
> 200 GeV. Three settings differ on purpose (see [Physics notes](#physics-notes-and-limits)):
> `Subtract_hard_momentum` 1 instead of 0 (with ISR the hard partons' momentum must come out
> of the strings), `generateOnlyPositions` 1 (the strings are made after the ISR) and
> b ≤ 3 fm. (Until October 2026 the production ran EOS 9, because iSS could not decay the
> SMASH hadron list of EOS 91; `--eos 9` reruns such campaigns.)

| file | purpose |
|---|---|
| `../config/OO_MCGlauber_MUSIC_isr.xml` | user XML: tuned 3dMCGlauber + MUSIC for O+O, `<Hard>` with `<PythiaGun>` + `<ISR>`, Matter, iColored |
| `../config/grid_oo.yaml` | output grid of the hydro file: 64 × 64 × 32 at 0.3125 fm × 0.3125 fm × 0.3125 (x, y ±9.84 fm) |
| `run_prod_oo_isr.py` | one job: one seed, N events |
| `run_jobs.sh` | many jobs, `-j P` at a time, resumable (wraps `../prod_AuAu_0_10/run_jobs.sh`) |
| `rehadronize_jets.py` | hadronize a `--store-hadronization-input` job's jets again from the stored Pythia input: `--check` against the job, or decays off / new seeds ([Redoing the jet hadronization](#redoing-the-jet-hadronization)) |
| `hard_sub_compare/` | two campaigns with and without the hard-momentum subtraction, and a notebook comparing their hadrons ([README](hard_sub_compare/README.md)) |
| `oo_isr_overview.ipynb` | one job's output: the medium's energy density at three τ (with the freeze-out contour and the dijet), and η, φ, pT and Δφ-to-the-leading-parton distributions of the bulk and jet hadrons |

## Run

```bash
conda activate js_fno        # sets PYTHIA8DATA: iMATTER reads Pythia's PDF tables next to it
cd external_packages/js-contrib/contribs/PyJetscape/example/prod_OO_isr

python run_prod_oo_isr.py --events 1 --seed 1 --dry-run     # check the XML/grid, print the plan
python run_prod_oo_isr.py --events 10 --seed 1              # pTHat 20-40 GeV, b < 3 fm
python run_prod_oo_isr.py --events 10 --seed 1 --pthat-min 10 --pthat-max 20
python run_prod_oo_isr.py --events 10 --seed 1 --b-max 5    # more peripheral
python run_prod_oo_isr.py --events 10 --seed 1 --sqrts 5360 # LHC energy (tuned at 200 GeV)
python run_prod_oo_isr.py --events 30 --seed 1 --pthat-bins 10-20,20-40,40-60
                                                            # 3 pTHat windows, every event complete
python run_prod_oo_isr.py --events 30 --seed 1 --pthat-bins 10-20,20-30,30-40 --parton-ymax 0.6
                                                            # both hard partons at |y| < 0.6

./run_jobs.sh 20 25 0                                       # 20 jobs x 25 events, unique seeds -> ./out
./run_jobs.sh -j 2 20 25 1 out_had --bulk-hadrons 100 --no-particlize
                                                            # seeds 1..20, all hadrons made in the jobs
```

**Look at a job:** [`oo_isr_overview.ipynb`](oo_isr_overview.ipynb) reads `<stem>.h5`,
`<stem>_hadrons_bulk_bg.h5` and `<stem>_hadrons_jet_frag.h5` (stem from `OO_STEM`, default
`out/OO_isr_seed0001`); it is stored with the output of a 20-event job made with
`--events 20 --seed 1 --bulk-hadrons 100 --no-particlize`.

```bash
OO_STEM=out/OO_isr_seed0001 jupyter nbconvert --to notebook --execute --inplace oo_isr_overview.ipynb
```

**Events that fail are skipped, not the job.** A module that throws inside an event (seen:
Matter's "minus log of sudakov negative", 1 event in ~400 in the benchmark below, with
`-j 12` under MPS; the same seed alone did not fail again) costs that event only: nothing of
it is written, the job goes on, and the `.json`
lists it (`events_skipped`, `skipped` with the event and the error; `events_skipped` also as
a file attribute). Rows stay consistent in every file; `events/job_event` (particlize) and
`units/event` (hadron files) give the event's number in the job, as `diag/event` does.
`run_jobs.sh` counts written + skipped events as complete, so a campaign does not rerun
the job. More than `--max-skipped` (default 3) skipped events stop the job: then something
else is wrong. The cause of those skips was found on 2026-10-12: Matter read a garbage qhat
for the ISR partons before t = 0, fixed in X-SCAPE `8c975d37` ([Physics
notes](#physics-notes-and-limits)); with the fix, two runs of 100 events had no skip and were
bit-identical. Images up to `cu124-final_optim` still have the skips.

**No hydro reuse, also with pT̂ windows.** With ISR the strings, and so the medium, are made
from what each event's hard scattering left (its partons' momentum is subtracted from the
colliding nucleons), so a medium belongs to one hard event. Every event runs the whole chain
(3dMCGlauber, ISR, strings, MUSIC, Matter, hadronization); X-SCAPE refuses
`<setReuseHydro>` with ISR (`JetScape::Init`), and this driver has no `--reuse`.
`--pthat-bins 10-20,20-40,40-60` runs one Pythia per window (PythiaIsrGun's `<pTHatBins>`,
as PythiaGun's) and event *i* uses window *i* mod K; unlike the Au+Au `--pthat-bins`, no
background is shared between the windows. Per event `pthat_bin` is recorded (`diag/`,
`events/`); per window the file attributes `pthat_bins`, `pthat_bin_sigma_gen`,
`pthat_bin_sigma_err`, `pthat_bin_n_accepted`, `pthat_bin_seeds` (and the `.json`) give the
cross sections for weighting. Use a multiple of K events for the same number per window.

**Rapidity cut on the hard partons: `--parton-ymax Y`** (`--parton-y-mode both`, the
default, or `any`). PythiaIsrGun regenerates the event until the two outgoing partons of the
hard scattering (Pythia's 2→2, before the ISR) have |y| < Y (both, or either), before
anything else runs. Their pT is equal at that stage, so there is no `leading` mode as in the
Au+Au driver. The cross sections are Pythia's × the acceptance kept/tried per window: the
file attributes and the `.json` give `pthat_bin_sigma_gen` (corrected),
`pthat_bin_acceptance`, `pthat_bin_n_tried`, `pthat_bin_n_kept` and Pythia's own
`pthat_bin_sigma_gen_raw`. Per event `parton_y_lead`, `parton_pt_lead`, `parton_y_sub`,
`parton_pt_sub` are the two hardest partons the hard process holds after the ISR (iMATTER
changes their momenta, so |y| can exceed Y slightly). At 200 GeV with Y = 0.6 (both) the
acceptance is 0.14, 0.30 and 0.49 in the 10–20, 20–30 and 30–40 GeV windows; this costs no
hydro time. Needs X-SCAPE's PythiaIsrGun with `<partonYMax>` (branch `oo_hard_sub_compare`).

The seeds, campaign names, the seed registry and the working directories are those of the
Au+Au productions: see [`../prod_AuAu_0_10_jet/README.md`](../prod_AuAu_0_10_jet/README.md)
(*Campaigns with `run_jobs.sh`*). Every option of `run_prod_oo_isr.py` can follow the
numbers of `run_jobs.sh`. The same options and seed give the same files, bit for bit.

## Containers

The production image (`utils/Dockerfile.prod`, [`docs/README_2stage.md`](../../../../docs/README_2stage.md))
starts in `prod_AuAu_0_10_jet`. For O+O, point its working directory at this folder, or call
the scripts by their full path: they find their XML, grid and helpers next to themselves.

```
/opt/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape/example/prod_OO_isr
```

> **Image (2026-10-13).** `jhputschke/xscape-prod:cu124-final_optim` (= `cu124`, digest
> `sha256:91c04102…`) has this folder, the ±10.5 fm grid and the late time step; the O+O
> numbers below were measured with it. The images of 2026-09-30 and 2026-10-03 have neither
> the folder nor X-SCAPE's `<Hard><ISR>` task list. Two O+O fixes came after
> `cu124-final_optim` (X-SCAPE `final_optim`): Matter's guard against a qhat lookup before
> t = 0 (`8c975d37`, see [Physics notes](#physics-notes-and-limits)) and iSS's leak fix
> (`50226a7c`). For an O+O campaign build from those branches:
>
> ```bash
> docker build -t xscape-prod:oo -f utils/Dockerfile.prod \
>   --build-arg XSCAPE_REF=final_optim --build-arg JS_CONTRIB_REF=final_optim .
> ```

**Docker:** `-w` sets the working directory.

```bash
OO=/opt/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape/example/prod_OO_isr
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  -e OMP_NUM_THREADS=3 -e OMP_WAIT_POLICY=passive -e MUSIC_CUDA_SYNC=block \
  -v "$HOME/prod:/work" -w "$OO" $IMG \
  ./run_jobs.sh -j 8 --mps 32 60 0 /work/OO_a --bulk-hadrons 100 --no-particlize
```

On a two-GPU machine, [`utils/launch_2gpu_oo.sh`](../../../../utils/launch_2gpu_oo.sh) starts
one container per GPU with these settings (8 jobs per GPU, 3 threads,
`MUSIC_CUDA_SYNC=block`, MPS).

**Apptainer:** `--pwd` sets it (Apptainer ignores the image's working directory anyway).

```bash
OO=/opt/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape/example/prod_OO_isr
apptainer exec --nv --env OMP_NUM_THREADS=3,OMP_WAIT_POLICY=passive,MUSIC_CUDA_SYNC=block \
  --pwd "$OO" --bind "$SCRATCH/prod:/work" xscape_prod.sif \
  ./run_jobs.sh -j 8 --mps 32 60 0 /work/OO_a --bulk-hadrons 100 --no-particlize
```

- **Without changing the directory:** `"$OO"/run_jobs.sh ...` or `python "$OO"/run_prod_oo_isr.py ...`
  work from anywhere, e.g. from the default `prod_AuAu_0_10_jet`.
- **Output under the bound `/work`** (`/work/OO_a`): the default `./out` is inside the
  image, read-only under Apptainer. The seed registry goes next to OUTDIR, so `/work` too.
- **The bulk afterwards** (setup C of [Hadronization](#hadronization-in-the-job-or-afterwards)): `hadronize.py` is in the Au+Au folder, e.g. with
  `-w "$OO"`: `../prod_AuAu_0_10_jet/run_hadronize.py /work/OO_a -j 8 --tags bulk_bg --oversample 100`
  (no GPU needed).
- **Nothing to set for iMATTER:** the image sets `PYTHIA8DATA`, and iMATTER finds Pythia's
  PDF tables next to it.
- **The notebook** `oo_isr_overview.ipynb` needs Jupyter, which the image does not have: run
  it on the host or in the analysis venv ([`utils/analysis_env`](../../../../utils/analysis_env/README.md))
  on the files in `/work`.

## Hadronization: in the job or afterwards

An O+O event's hadrons come in two files, both in the `jetscape.hadrons_h5` format of the
Au+Au productions:

| hadrons | file | made by | when |
|---|---|---|---|
| **jet** | `<stem>_hadrons_jet_frag.h5` | iColoredHadronization: the final partons and the beam remnants, through the ISR colour flow | **in the job** (unless `--no-jet-hadrons`); **again afterwards** only with `--store-hadronization-input` (`rehadronize_jets.py`) |
| **bulk** | `<stem>_hadrons_bulk_bg.h5` | iSS on MUSIC's freeze-out surface, N oversamples per event | **in the job** with `--bulk-hadrons N`, **or afterwards** with `hadronize.py --tags bulk_bg` on the particlize file |

The jet hadrons are made in the job: iColoredHadronization needs the event's beam remnants
(momenta from 3dMCGlauber) and the ISR colour connections, which exist only during the job.
With `--store-hadronization-input` the job keeps what it handed to Pythia (see
[Redoing the jet hadronization](#redoing-the-jet-hadronization)), so the jets can be
hadronized again later; without it, they cannot. The bulk can be made either way. Choose one
of three setups:

| setup | command | files | afterwards |
|---|---|---|---|
| **A. everything in the job**, no surfaces kept | `run_prod_oo_isr.py ... --bulk-hadrons 100 --no-particlize` | `.h5`, `_hadrons_bulk_bg.h5`, `_hadrons_jet_frag.h5` | nothing to do |
| **B. in the job, surfaces kept** | `run_prod_oo_isr.py ... --bulk-hadrons 100` | A + `_particlize.h5` | more oversamples or other iSS settings any time |
| **C. bulk afterwards** (the default) | `run_prod_oo_isr.py ...` | `.h5`, `_particlize.h5`, `_hadrons_jet_frag.h5` | `hadronize.py --tags bulk_bg` |

Add `--no-hydro-file` to any of them to leave out the hydro evolution (`<stem>.h5`): MUSIC
still runs and Matter still reads its medium, only nothing of it is stored. A + `--no-hydro-file`
is a hadrons-only job: `_hadrons_bulk_bg.h5`, `_hadrons_jet_frag.h5`, `.json`, `.xml`.

```bash
# A: all hadrons in the job, nothing stored to hadronize later
python run_prod_oo_isr.py --events 10 --seed 1 --bulk-hadrons 100 --no-particlize
# A without the hydro evolution: hadrons only
python run_prod_oo_isr.py --events 10 --seed 1 --bulk-hadrons 100 --no-particlize --no-hydro-file
# A with smaller hadron files (hadronize.py's selection options, on both hadron files)
python run_prod_oo_isr.py --events 10 --seed 1 --bulk-hadrons 100 --no-particlize \
       --eta-max 2 --charged --keep-bits-p 12

# C: the bulk afterwards, on CPUs, from the stored surfaces
python run_prod_oo_isr.py --events 10 --seed 1
python ../prod_AuAu_0_10_jet/hadronize.py out/OO_isr_seed0001_particlize.h5 \
       --tags bulk_bg --oversample 100
python ../prod_AuAu_0_10_jet/run_hadronize.py out -j 4 --tags bulk_bg --oversample 100
                                                    # every particlize file in out/
```

Options for the hadrons made in the job:

| option | effect |
|---|---|
| `--bulk-hadrons N` | run iSS in the job, N oversamples per event → `<stem>_hadrons_bulk_bg.h5` (default 0: no bulk hadrons in the job) |
| `--hadronize-xml FILE` | iSS settings for `--bulk-hadrons` (default `../config/hadronize.xml`, the settings `hadronize.py` runs, so in-job and offline bulk are sampled alike) |
| `--no-particlize` | no `<stem>_particlize.h5`: no surfaces, partons or initiators stored |
| `--no-hydro-file` | no `<stem>.h5`: the hydro evolution is not stored (`hydro_row` is then -1 for every event) |
| `--no-jet-hadrons` | no iColoredHadronization, no `<stem>_hadrons_jet_frag.h5` |
| `--store-hadronization-input` | opt-in: also store each event's Pythia input of the jet hadronization in the jet file (`hadronization_input/`, ~2 kB per event), so `rehadronize_jets.py` can redo the jet hadrons later |
| `--keep-primaries N` | with `--bulk-hadrons`: also write the undecayed bulk hadrons of the first N oversamples → `<stem>_hadrons_bulk_bg_primary.h5`, for SMASH later (full precision, positions, no selection; the decayed hadrons are the same with or without it) |
| `--eos 9\|91` | MUSIC's EOS (default: the XML's, 91); 9 only to rerun older campaigns |
| `--eta-max X`, `--charged`, `--no-x`, `--keep-bits-p B`, `--keep-bits-x B` | store only \|η\| < X / charged hadrons / no positions / rounded p, x — as `hadronize.py`, on both in-job hadron files; iSS and Pythia still sample everything |

**Jet hadronization seeds.** iColoredHadronization reseeds Pythia every event from the
job's seed (`<reseed_per_event>` 1, always on in this driver) and records the seed: `units/seed`
of the jet file and `jet_pythia_seed` in every event record. Before October 2026 it ran from
Pythia's default seed, the same stream in every job, so those jobs' jet hadrons cannot be
reproduced event by event and differ from a rerun with this driver.

**Jet hadronization colour checks** (X-SCAPE and js-contrib `lbt_colored_hadro`,
[`PLAN_lbt_colour_tags.md`](../../../../docs/Plans/PLAN_lbt_colour_tags.md)):
- **Colour tags were duplicated before.** Matter took its new tags from per-parton blocks
  that overlap iMATTER's ISR blocks once there are MPI scatterings: in two 75-event test jobs
  131 of 150 events had a tag used more than once, which Pythia strings silently. Matter
  now draws its tags from one counter per event (`HardProcess::NextColorTag`), and those jobs
  have none. The parton kinematics are unchanged (bit-identical), the jet hadrons are not:
  **jobs before and after are not combinable.**
- **A failed fragmentation fails the event.** Pythia's `next()` was not checked: when it gave
  up ("junction topology inconsistent"), the input partons were written as jet hadrons. In
  `hard_sub_compare/` that happened in `sub/` seed 0103 from event 46 and seed 0129 from event
  3, to the end of each job (100 of 6031 events). Now the event throws and is skipped
  (`skipped` in the job summary). Duplicate tags also fail the event unless
  `<JetHadronization><colour_duplicates>relabel` (a counted repair).
- **Lines that end in the medium** (a scattering's hole took the thermal parton's colour
  with it: with LBT or Matter's recoils) are closed as `<JetHadronization><colour_closure>`
  says:

  | `<colour_closure>` | what it does |
  |---|---|
  | `reconnect` (default) | join the open ends within a shower, the shortest string first; the holes' medium partners (the conjugate parton with the hole's momentum) for the rest |
  | `reconnect_global` | the same, joining across showers too |
  | `thermal` | every open line gets its hole's medium partner |
  | `fake` | a soft (1 GeV) light (anti)quark along the beam per open end |
  | `none` | throw (an event with an open line is skipped) |

  The holes themselves are not hadronized; their energy is recorded (`colour_hole_energy`).
  A fragmentation that fails after a closure is redone once with `fake` (`colour_retries`).
- **`units/colour_*`** of the jet file: the event's colour audit (partons and remnants
  handed to Pythia, holes and their energy, other negative statuses skipped, open colour /
  anticolour tags before the closure, duplicate and relabelled tags, `reconnected` pairs,
  `reconnected_cross` (across showers, as a fallback), `partners` and their energy, `fakes`
  and their energy, `retries`). The job summary (`<stem>.json`) has them added up over the
  job (`jet_colour_totals`, with `failed` = events thrown), as the log's colour summary at
  the end of the job.

### Redoing the jet hadronization

With `--store-hadronization-input` the jet file keeps, per event, the parton list
iColoredHadronization handed to Pythia (`hadronization_input/`: pid, col, acol, E, px, py, pz,
m of the final partons, the beam remnants with their 3dMCGlauber momenta and the
colour-balancing parton), next to the seed, the Pythia version (`pythia_version`) and the
effective `<JetHadronization>` settings (`jet_hadronization_xml`). `rehadronize_jets.py` then
hadronizes the jets again with Pythia alone, no 3dMCGlauber and no hydro:

```bash
python run_prod_oo_isr.py --events 10 --seed 1 --store-hadronization-input
python rehadronize_jets.py out/OO_isr_seed0001_hadrons_jet_frag.h5 --check
            # the stored seeds give the job's jet hadrons exactly (same Pythia version)
python rehadronize_jets.py out/OO_isr_seed0001_hadrons_jet_frag.h5 --decays off \
       --out out/OO_isr_seed0001_hadrons_jet_frag_nodecay.h5   # the primary hadrons
python rehadronize_jets.py out/OO_isr_seed0001_hadrons_jet_frag.h5 --new-seeds 7 \
       --out out/OO_isr_seed0001_hadrons_jet_frag_seed7.h5     # another fragmentation
```

What it cannot change: anything before the final partons (Matter, the ISR), and the colour
flow. Hadron positions (for SMASH) are a later step of
[`PLAN_smash_ready_hadrons.md`](../../../../docs/Plans/PLAN_smash_ready_hadrons.md)
(2.2–2.3): they also need each parton's hadronization point, which needs the hydro in the job.

Au+Au needs none of this: its Colorless jet hadronization runs in `hadronize.py`, from the
final partons in the particlize file, so it can always be redone there; `hadronize.py --check`
is the counterpart of `--check` here
([Re-hadronizing the jets](../prod_AuAu_0_10_jet/README.md#re-hadronizing-the-jets)).

**An event at hadron level** is unit *i* of the jet file plus unit *i* of the bulk file
(units are events in both): sample *k* of the bulk together with the jet hadrons of the same
event. The *N* bulk oversamples share one fluid: average over them, they are not independent
collisions.

Good to know:
- **In the job and afterwards agree.** Both run iSS with `hadronize.xml`'s settings; only
  the seeds differ. In a 3-event test with 50 oversamples each (made before the subtraction
  was switched on, see *Physics notes*), dN/dη at |η| < 0.5 per event
  was 32.0 / 37.9 / 32.1 in the job and 32.0 / 36.8 / 31.2 from `hadronize.py`; ⟨pT⟩, the
  charged fraction and the π⁺, K⁺, p yields agree as well. The in-job file has the attribute
  `inline`, and each unit its iSS `seed`, `n_cells` and `hydro_row`.
- **B and C combined.** `hadronize.py` keeps an existing `_hadrons_bulk_bg.h5` (it stops
  with "exists"): write the offline set elsewhere (`--out-dir`) or replace it (`--force`).
- **Always `--tags bulk_bg` with `hadronize.py`.** It refuses `jet_frag` on these
  particlize files (attribute `jet_frag_inline`): ColorlessHadronization would drop the hard
  partons (status 1000, see below) and overwrite the in-job jet hadrons.
- **`--bulk-hadrons` and `--no-hydro-file` change the jet showers' random numbers.**
  X-SCAPE seeds each task from the order it was made, and the per-parton Matter copies are
  made during the event, after the job's other tasks: iSS (`--bulk-hadrons`) and the hydro
  writer (absent with `--no-hydro-file`) change their number, so they get other seeds. The
  showers are then a different, equally valid sample (the hydro is identical). Keep one
  choice per campaign if jets of different jobs are to be compared event by event.
- **Events without a medium** (no string, `hydro_row` -1) get an empty bulk unit; the run
  goes on.

## Output

Each job writes, next to each other (stem `OO_isr_seedNNNN`, or `OO_isr_<campaign>_NNNN`):

| file | content | written by |
|---|---|---|
| `<stem>.h5` | MUSIC's evolution (e, vx, vy, vz) on `grid_oo.yaml`, read from the framework copy Matter also reads; `diag/` per event: `event`, `pthat`, `sigma_gen`, `sigma_err`, `event_weight`, `glauber_npart`, `glauber_ncoll`, `glauber_nstrings`, `glauber_b`, `tau0_music`, `has_medium`, `hit_grid_boundary`, the ISR energy bookkeeping
`isr_sub_*`, `remnant_*`, `strings_*` (see *Physics notes*) (`--no-hydro-file`: not written) | `jetscape.fast_h5_bulk.H5BulkWriter` (`grid_mode="framework"`) |
| `<stem>_particlize.h5` | MUSIC's freeze-out surface (`surface/bg`, one unit per event), the final partons (`partons/`) and the shower initiators; `events/` as `diag/` plus `hydro_row`, `wall_s` (`--no-particlize`: not written) | `jetscape.particlize_h5.ParticlizeH5Writer` (`legs=("bg",)`) |
| `<stem>_hadrons_bulk_bg.h5` | the bulk hadrons: in the job with `--bulk-hadrons N`, else later from `hadronize.py` | `jetscape.hadrons_h5.HadronH5Writer` |
| `<stem>_hadrons_jet_frag.h5` | the jet hadrons of iColoredHadronization, one unit per event, with the initiators | `jetscape.hadrons_h5.HadronH5Writer`, in the job |
| `<stem>.xml`, `.json` | the job XML; a summary (events, cross section, outputs, wall time) | |

- **The hadron files' `units/`** carry the event's whole record (as `diag/`: pTHat, window,
  Npart, `isr_sub_*`, `remnant_*`, `strings_*`, the hard partons), so hadron-only jobs
  (`--no-hydro-file --no-particlize`) keep it.
- **`hydro_row`** links the files: the row of the event in `<stem>.h5`, or -1 when it has
  none: for an event without a medium, and for every event with `--no-hydro-file`.
  **`has_medium`** (1/0, in `diag/`, `events/` and the hadron files' `units/`) says which:
  without a medium (no string) there is no evolution, Matter showers in vacuum, and the
  surface and bulk unit are empty.
- **The jet hadrons** are those of the whole hard-scattered nucleon pair: dijet, Pythia's MPI
  partons, the ISR emissions and the remnants.
- **Final-parton status codes.** PythiaIsrGun marks the outgoing hard partons 1000 and
  Matter's daughters keep that status, so the showered jet partons are mostly status 1000;
  the showers of the ISR emissions are status 0. Status 1000 does not mean "not showered".

Measured on the GB10 (2026-10-10, one job, default settings): ~5 s per event, 2.8 GB peak
memory; per event ~17 MB hydro file, ~10 MB particlize file. `--bulk-hadrons 50` adds ~0.3 s
per event (~5,600 hadrons per event over the 50 oversamples).

### Throughput on 2 × RTX 3090 (campaigns, current image)

Hadron files only (`--bulk-hadrons 10 --no-particlize --no-hydro-file`), 8 jobs per GPU with
3 threads, MPS, `MUSIC_CUDA_SYNC=block`, 60 events per job, ±10.5 fm MUSIC grid
(`launch_2gpu_oo.sh`; [`REPORT_FINAL_OPTIM.md`](../../../../docs/REPORT_FINAL_OPTIM.md),
[`BENCHMARK_2x3090.md`](../../../../docs/BENCHMARK_2x3090.md)):

| image | output | events/h (machine) | GPU busy | core-s per event | GPU-s per event | per job (VmHWM) | disk per event |
|---|---|---|---|---|---|---|---|
| **`cu124-final_optim`** (C4, late Δτ, A12) | hadron files only | **13,728** | 65 % | 6.1 | 0.344 | 1.43 GB | 0.13 MB |
| `cu124-final_optim`, `--late-dtau 0` | hadron files only | 12,722 | 74 % | 6.1 | 0.421 | 1.42 GB | 0.13 MB |
| `cu124-final_optim` | full outputs (hydro, particlize, jet hadrons) | 12,700 | 60 % | 5.9 | 0.342 | 1.40 GB | 28 MB |
| `cu124-freezeoutcopies` (A1–A11) | hadron files only | 10,313 | 79 % | 7.3 | 0.55 | 1.43 GB | 0.12 MB |

- **The cores set the rate:** ~24 busy threads, the box's physical cores; ~570 events/h per
  core. The GPU limit is ~18,800 events/h.
- **Full outputs write ~360 GB/h,** more than the ~290 GB/h an OSDF upload takes: hadron
  files only, or a large local disk.
- **The late step** saves 20 % of the GPU time per O+O event (the events end at τ ≈ 4.4 fm/c,
  so less of each is after the switch). It makes the freeze-out surface 0.20 % larger and
  freezes out one frame later in 11 % of events; the bulk hadrons change by +0.05 % ± 0.03 %
  (+0.4 % ± 0.2 % at \|η\| < 1). Keep one setting per campaign.

### Throughput on the GB10 (campaigns, 2026-10-10)

Hadrons only (`--bulk-hadrons 10 --no-particlize --no-hydro-file`), one wave of j jobs x 10
events, CUDA MPS, `OMP_NUM_THREADS` = 20 cores / j (2026-10-10, GB10: 20 cores, 121 GB):

| `-j` | `OMP_NUM_THREADS` | events/h | s per event in a job | peak memory (machine, ~7 GB idle) | GPU busy (mean) |
|---|---|---|---|---|---|
| 4 | 5 | 1,520 | 9.0 | 18.5 GB | 77% |
| 8 | 2 | 1,620 | 17.1 | 28.5 GB | 92% |
| 12 | 1 | ~1,600 | 23.7 | 38 GB | 92% |
| 16 | 1 | 1,650 | 33.7 | 48 GB | 93% |

- **The GPU is the limit**, not memory or cores: from `-j 8` it is ~92% busy and the
  throughput stays at ~1,600-1,650 events/h. Memory is no constraint (~2.5-3 GB per job).
- **Recommended:** `OMP_NUM_THREADS=2 ./run_jobs.sh -j 8 --mps ...` (`-j 4` is within 6% if
  the GPU should stay less loaded). That is ~8-9x the Au+Au jet production (~190 events/h,
  [`BENCHMARK_GB10.md`](../../../../docs/BENCHMARK_GB10.md)).
- With the hydro file and the particlize file the jobs write more (~27 MB per event) but do
  the same GPU work; not measured as a campaign.
- **MUSIC's grid is ±10.5 fm × ±6 (70 × 70 × 60 cells) since 2026-10-12** (`<IS><grid_max_x/y>` in
  the XML; it was ±15 fm when the table above was measured). On the 2 × RTX 3090 machine the
  smaller grid gave +82 % events/h and half the CPU and memory per job, with the hydro and the
  freeze-out surface the same to float32 rounding and no grid-edge hit in 2,994 events
  ([`BENCHMARK_2x3090.md`](../../../../docs/BENCHMARK_2x3090.md), "O+O campaign"); the GB10
  campaign is not re-measured. For larger b or another system check `diag/hit_grid_boundary`
  and widen the grid; the output grid (±9.84 fm) must stay inside it.
- **MUSIC's time step is 0.05 fm/c from 3 fm/c on** (`<late_delta_tau>`, `<late_tau_switch>` in the
  XML; `--late-dtau 0` for a constant 0.02, `--late-tau TAU` for another switch time): about
  half the steps of an event after that time, the stored frames and the freeze-out checks kept
  at 0.1 fm/c (docs/Plans/Final_Optims.md, D1). Not bit-identical (above); details in
  [`../prod_AuAu_0_10_jet/README.md`](../prod_AuAu_0_10_jet/README.md#musics-late-time-step-default).
- **The GB10 is not re-measured** with the ±10.5 fm grid or `final_optim`. It is GPU-bound
  (92 %), so both should help it most: ~3,500–4,500 events/h expected
  ([`REPORT_SUMMARY_CLUSTER.md`](../../../../docs/REPORT_SUMMARY_CLUSTER.md)).

## Physics notes and limits

- **MUSIC's anisotropy diagnostic is off.** The XML sets
  `<output_momentum_anisotropy_vs_etas>0`: no `momentum_anisotropy_*` / `eccentricities_*`
  `.dat` files, which nothing reads, and no serial pass over the grid at four steps per run.
  The output is unchanged. Set 1 in a user XML to get them back. Builds older than X-SCAPE
  `6480dbeb` ignore the tag. The other host-CPU fixes of 2026-10-11 need no setting
  ([`PLAN_cpu_fixes_A.md`](../../../../docs/Plans/PLAN_cpu_fixes_A.md)).
- **EOS 91, the tuned set's.** iSS samples the SMASH hadron list and, with
  `hadronize.xml`'s `<decay_smash_list>` 1 (iSS from `smash_ready` on), decays it itself:
  the channels the pole masses close (ω → πρ, η′ → ησ, f₂ → ρρ, ... 215 of them) get
  Breit–Wigner masses for their unstable daughters. η and Σ⁰ stay stable, as in SMASH, so
  they are among the final hadrons (EOS 9 decays them). An iSS without it leaves the SMASH
  list undecayed (both scripts warn): in a 3-event test that gave a midrapidity dN/dη of
  15.5 per sample against 34.8, decayed, with EOS 9. Until October 2026 the production ran
  EOS 9 (UrQMD list) for that reason; `--eos 9` reruns such campaigns.
- **One hard scattering per event**, at a binary collision of 3dMCGlauber drawn uniformly.
  The events are therefore hard-triggered with no weight for the number of collisions:
  `glauber_ncoll` is recorded per event for that. `b` selects the centrality
  (`cenMin`/`cenMax` act only in the standalone 3dMCGlauber); `b < 3 fm` is about 0–20% of
  the O+O cross section.
- **The hard partons' momentum is taken from the colliding nucleons**
  (`<Subtract_hard_momentum>1`, the X-SCAPE default; the BulkFastTest O+O example, without
  ISR, has 0). The hard pair's valence quark gives up the momentum the ISR hands over, and
  its remnants go to the jet hadronization instead of making strings. With 0 3dMCGlauber
  zeroes that momentum first: the strings keep the full Npart √s/2 although the hard partons
  and the remnants are hadronized too, so 6-12% of the event's energy counted twice (in the
  5-event test below; earlier O+O files of this production had 0). X-SCAPE now warns about 0 with ISR, and
  `run_prod_oo_isr.py` sets 1 unless `--no-subtraction` (below) asks for the reference. **Recorded per event** (`diag/`, `events/`):
  `isr_sub_{E,px,py,pz}_{proj,targ}` and `isr_sub_E_total` (handed over for subtraction from
  the projectile, +z, and target, -z, nucleon), `remnant_E_{proj,targ}` and
  `remnant_pz_{proj,targ}` (the two nucleons' remnants, to the jet hadronization), and
  `strings_E_total`, `strings_pz_net` (the strings and remnant ends MUSIC deposits, its own
  sum). Per event, exactly: `strings_E_total + isr_sub_E_total + remnant_E_proj +
  remnant_E_targ = Npart √s/2` (5-event test: 1855.7 + 89.7 + 54.6 = 2000.0 GeV for
  Npart 20; the subtracted energy was 66-136 GeV, the remnants 24-55 GeV).
- **Reference without the subtraction: `--no-subtraction`.** Runs the same chain with
  `Subtract_hard_momentum` 0 and still hadronizes the jets (iColored): the strings keep the
  full Npart √s/2 while the jet hadrons also carry the hard partons and the hard pair's
  remnants, so that energy is counted twice, on purpose -- the difference to the default is
  what the subtraction does in a small system. Every file and the `.json` record
  `subtract_hard_momentum` (0 here, 1 by default; check it before combining files), the
  start prints a warning, and the bookkeeping is the same (`isr_sub_*` is then what would
  have been subtracted; `strings_E_total` = Npart √s/2). Keep such jobs in their own OUTDIR
  or campaign (`--campaign nosub`): the file names are not changed, so that `run_jobs.sh`
  can resume them. **Compare ensembles, not events:** the same seed gives the same hard
  events (Pythia has its own stream), but 3dMCGlauber's geometry agrees only for the first
  event, because the subtraction draws from its random stream (4-event test: event 0 paired
  exactly, the strings carried 144.3 GeV more = subtracted + remnants; from event 1 on Npart
  differed).

```bash
python run_prod_oo_isr.py --events 20 --seed 1 --outdir out_nosub --bulk-hadrons 100 \
       --no-particlize --no-subtraction                # the reference, next to out/
```
- **The hard nucleon pair makes strings only if it is not yet connected**
  (`Glauber::decide_produce_string_num`). Peripheral events can end up with no string at all
  (`hydro_row` -1); in p+p this is most events (9 of 12 in a test).
- **Matter and ISR partons before t = 0.** The final-state partons that iMATTER hands over
  start at negative times, and Matter's qhat table lookups bounded the index only from above:
  a negative index read memory before the table (~3,800 lookups per event), a garbage qhat.
  That made runs irreproducible: Matter's "minus log of sudakov negative" skips changed from
  run to run with the same seed, and a far negative index crashed. X-SCAPE `8c975d37` (branch
  `final_optim`) takes qhat = 0 before t = 0, where there is no medium (a window that starts
  there averages with 0); two runs of 100 events are then bit-identical, with no skip or
  crash. `cu124-final_optim` doesn't have it yet (see
  *Containers*). The Au+Au production has no ISR and is unchanged.
- **Jet energy loss: Matter, or Matter + LBT** (`lbt_colored_hadro`). LBT now gives its
  partons the large-Nc colour flow (Matter's rules per channel), Matter's recoils keep theirs,
  and iColored closes the colour lines that end in the medium (`<JetHadronization>
  <colour_closure>`, above). The production XML stays Matter-only (`recoil_on` 0); add an
  `<Eloss><Lbt>` block (and `recoil_on` 1, as in the Au+Au XML) for LBT. A build without the
  closure still refuses `<Eloss><Lbt>` unless `--no-jet-hadrons`. Tested 2026-10-21 on two
  75-event jobs: 632 holes, every line closed (no soft fallback, no failure), every tag paired
  once, hadron energy = Pythia input energy.
- **ISR range.** iMATTER runs from 0 to -`<Eloss><maxT>` (50 fm/c, as in X-SCAPE's ISR
  examples) and the final-state shower starts there; `<Hard><ISR><maxT>`/`<deltaT>` override.
- **Not included:** X-SCAPE branch `Changing_iMatter_pTDefinition` (upstream, August 2025,
  not merged) changes iMATTER's sibling-pT and recoil kinematics; this branch runs iMATTER as
  it is on `contrib`.

## Not working yet: Matter + LBT with ColorlessHadronization

The Au+Au production's combination, Matter + LBT with ColorlessHadronization (here without
energy deposition, i.e. no liquefier), **runs with ISR but gives wrong jet hadrons**. It is not
an option of `run_prod_oo_isr.py`. Tested 2026-10-10 with `runJetscape` on this chain
(`<Hard><ISR>`, `<Eloss>` Matter + LBT, `<JetHadronization><name>colorless`), 5 events each:

| run | jet-parton energy with status 1000 (dropped) | energy Colorless takes | events with no jet hadrons |
|---|---|---|---|
| Matter + LBT | 45–92 GeV/event | 18–77 GeV/event | 0 of 5 |
| Matter only | 66–136 GeV/event | 0–44 GeV/event | 3 of 5 |

Two problems:

1. **Colorless drops the hard-scattered partons.** ColorlessHadronization takes only pstat 0
   (shower), 1 (recoil, `take_recoil`), 22 (photon) and -1 (hole, subtracted). With ISR the
   hard partons from PythiaIsrGun and all their Matter daughters have status 1000, so they are
   left out silently. With LBT, the partons LBT took over come back with status 0 and are
   kept; what left the medium, or never reached LBT, stays 1000 and is lost. With Matter alone
   almost nothing is hadronized.
2. **The fake beam remnant gets the wrong energy.** Colorless closes open strings with a fake
   remnant of momentum ~√s/6 along the beam, from `<JetHadronization><eCMforHadronization>`,
   whose main-XML default is 5020 GeV: such events carry 840–1740 GeV of hadron energy. The
   Au+Au `hadronize.xml` sets it to 200.

What a fix would take:
- X-SCAPE `src/hadronization/ColorlessHadronization.cc`: take status 1000 like status 0 (one
  line; only PythiaIsrGun makes status 1000, so other workflows are unchanged).
- The O+O XMLs: `<eCMforHadronization>` 200 with colorless.
- `run_prod_oo_isr.py`: a `--jet-hadronization colorless` choice that allows `<Eloss><Lbt>`.
- With Colorless the jet hadrons need no live event (no remnants): `hadronize.py --tags
  jet_frag` could then make them later from the particlize file's partons, and the
  `jet_frag_inline` guard would apply to iColored only.

Even fixed, Colorless with ISR ignores the ISR colour connections and the 3dMCGlauber beam
remnants: it hadronizes the final partons as independent strings, as the Au+Au Matter + LBT
production does, with LBT's holes and recoils through `take_recoil`.
