# Z+jet and other collision systems

The physics of a production is in its user XML, so a new collision system or a different hard
process is mostly a copy of an XML with a few lines changed. This page covers the two cases
that already exist: the **Z+jet production** in Pb+Pb at 5.02 TeV, made for the CMS and ATLAS
Z–hadron and Z+jet measurements, and **Pb+Pb jets at the LHC** from your own XML, with the
steps to take for any other system. Both run through the Au+Au jet production's driver
`run_prod_jet.py`, so everything in [Au+Au jets](auau-jet.md) applies. At the end, the
production wizard plans the launch command for any of the productions.

## Z+jet in Pb+Pb at 5.02 TeV

The Z+jet production runs Pb+Pb collisions at √s_NN = 5.02 TeV, 0–30 %, with a hard
process that makes a Z boson decaying to e⁺e⁻ or μ⁺μ⁻ and a recoiling parton. The Z's
leptons pass through the medium untouched, so the Z tags the initial jet momentum; the recoil
jet loses energy and leaves a wake. The inputs are in
[`example/config/`](https://github.com/jhputschke/js-contrib/tree/main/contribs/PyJetscape/example/config)
(in the image: `/opt/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape/example/config`):

- `PbPb_0_30_ZJet.xml`: the Pb+Pb jet XML at 0–30 % (`b_max` 8.57 fm) with a Z+jet
  `<PythiaGun>` (Z + parton processes, Z → ee/μμ, `<trigger>Z`);
- `grid_PbPb.yaml`: the larger Pb+Pb output grid (80 × 80 × 40 cells, x, y ±15.3 fm), for the
  edge flags;
- `hadronize.xml`: the shared hadronization settings; its `eCMforHadronization` is set to the
  collision's 5020 GeV from the XML's PythiaGun automatically.

### The settings

The campaigns use six pT̂ windows with 4 jets per window per background (24 events per
background), the Z's leptons inside the detector acceptance, and hadron files made in the job:

| option | |
|---|---|
| `--user-xml …/PbPb_0_30_ZJet.xml --grid …/grid_PbPb.yaml` | the Z+jet XML and the Pb+Pb grid |
| `--pthat-bins 20-30,30-35,35-45,45-55,55-70,70-200 --jets-per-bin 4` | the windows: the p_T^Z > 40 GeV cross section is spread evenly over them, so little is lost to the window weights |
| `--lepton-ptmin 20 --lepton-etamax 2.5` | both leptons p_T > 20 GeV, \|η\| < 2.5 (the CMS and ATLAS Z–hadron cuts); failing events are regenerated before any hydro, and the cross sections corrected |
| `--hadrons-only --oversample 40 --n-frag 40 --correlated` | hadron files only, 40 correlated samples per jet event |
| `--eta-max 2.6 --no-x --hadron-keep-bits-p 12 --hadron-keep-bits-x 8 --keep-primaries 0` | all final particles at \|η\| < 2.6, no positions, rounded |
| `TAG_PREFIX=PbPb_0_30_seed` (environment) | names the files `PbPb_0_30_<campaign>_…` instead of `AuAu_0_10_jet_…` |

`--trigger Z` comes from the XML; `--trigger-ymax Y` would add a Z rapidity cut (the
campaigns apply \|y_Z\| offline). There is no cut on the recoiling parton.

!!! warning "A current image"
    The Z trigger, `--hadrons-only` and `TAG_PREFIX` need an image built from the current code.
    Check it first: `python run_prod_jet.py --help` must list `--trigger`, and a `--dry-run`
    must print `trigger Z (lepton pT > 20 GeV, lepton |eta| < 2.5)`, `EOS 91` and
    `eCMforHadronization 5020`. Pin the image by digest for the campaign.

### Running it

Check with `--dry-run` inside the image (no GPU work), then run. In Docker:

```bash
IMG=jhputschke/xscape-prod:cu126                      # pinned by digest for a campaign
D=/opt/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape/example/config
mkdir -p ~/prod/lbt_cache
docker run -d --name zjet --gpus all --user "$(id -u):$(id -g)" \
  -e TAG_PREFIX=PbPb_0_30_seed -e OMP_NUM_THREADS=5 -e OMP_WAIT_POLICY=passive \
  -e MUSIC_CUDA_SYNC=block -e LBT_TABLE_CACHE=/work/lbt_cache/lbt_tables_v1.bin \
  -v "$HOME/prod:/work" $IMG \
  ./run_jobs.sh -j 3 --stagger 12 --campaign zjet_test 10 96 0 /work/zjet_test \
    --user-xml $D/PbPb_0_30_ZJet.xml --grid $D/grid_PbPb.yaml \
    --pthat-bins 20-30,30-35,35-45,45-55,55-70,70-200 --jets-per-bin 4 \
    --lepton-ptmin 20 --lepton-etamax 2.5 \
    --hadrons-only --oversample 40 --n-frag 40 --correlated \
    --eta-max 2.6 --no-x --hadron-keep-bits-p 12 --hadron-keep-bits-x 8 --keep-primaries 0
```

Ten jobs of 96 events (4 backgrounds each), three at a time: a Pb+Pb Z+jet job peaks at
**14–16 GB** of host memory (2.1× an Au+Au job; measured on the GB10), mostly MUSIC's larger
box. It takes 2.1× the time of an Au+Au jet job and writes 1.7× the disk.

**The scripts of the published campaigns** are in
[`utils/zjet/`](https://github.com/jhputschke/js-contrib/tree/main/utils/zjet): ten campaigns
`zjet_c00` … `zjet_c09` of 9,600 events each (100 jobs × 96 events), with disjoint seed ranges
(campaign C uses seeds from 1 + 1000 C), so campaigns from different machines combine. Run
them from a js-contrib checkout:

| script | runs |
|---|---|
| `zjet_slurm_campaign.sh` | one campaign as a SLURM array through `utils/slurm_prod_array.sh`: `C=0 sbatch zjet_slurm_campaign.sh` (set up for NCSA Delta's A40 nodes, 3 jobs per GPU; edit the `#SBATCH` lines for your site). `C=0 NJOBS=1` is a ~1,000-event smoke test |
| `zjet_3090_campaign.sh` | one campaign on a two-GPU workstation through `utils/launch_2gpu.sh`: `IMAGE=… ./zjet_3090_campaign.sh 0` |
| `zjet_3090_test.sh` | before the first campaign on a workstation: `info` (digest, commits, dry run), `bench 3`, `bench 4`, `report`, `smoke` |

After each campaign: `utils/check_deposition.py OUTDIR` ([Clusters](clusters.md#monitoring-a-campaign)),
then the upload to the OSDF ([Data on the OSDF](osdf.md)).

### The pp reference

`prod_AuAu_0_10_jet/run_prod_vacuum.py` runs the same XML's PythiaGun and Matter in vacuum (no
initial state, hydro, LBT or liquefier), with Colorless hadronization at the PythiaGun's
energy. It runs on the CPU, ~0.02 s per event on the GB10, so 12,000 events take minutes:

```bash
python run_prod_vacuum.py --events 12000 --seed 900001 --campaign zjet_pp \
  --user-xml $D/PbPb_0_30_ZJet.xml --pthat-bins 20-30,30-35,35-45,45-55,55-70,70-200 \
  --lepton-ptmin 20 --lepton-etamax 2.5 \
  --n-frag 40 --eta-max 2.6 --no-x --hadron-keep-bits-p 12 --outdir /work/zjet_pp
```

It writes one `<stem>_hadrons_jet_frag.h5` (attribute `vacuum`), read like the Pb+Pb jet files.
Give it seeds of its own: the Pb+Pb campaigns' seeds would give the same Pythia events.

### What to know in the files

- **The Z per event** is in `units/` of the hadron files (`z_pt`, `z_y`, `z_phi`, `z_m`,
  `lep{1,2}_{pid,pt,eta,phi}`); every paper's p_T^Z and mass window is applied offline.
- **Weight every event by its `n_coll`** (`units/n_coll`): the geometry is drawn with
  P(b) ∝ b over 0–30 % and every event has one Z, while Z production scales with N_coll.
  Weight also by the window's cross section.
- **The two leptons are among the jet hadrons** (`jet_frag`, pid ±11/±13) and in
  `initiators/`: remove them before a track or jet analysis.
- **Every background has its event plane near φ = 0** (3dMCGlauber puts the impact parameter
  along x). The means of a Z-frame analysis are unaffected; event mixing needs each
  background rotated by a random angle, which the Z-analysis scripts do by default.

The Z–hadron and Z+jet analyses, step by step:
[`analysis/zjet/`](https://github.com/jhputschke/js-contrib/tree/main/contribs/PyJetscape/example/analysis/zjet)
and [Hadrons](../analysis/hadrons.md). The campaign design (windows, cuts, expected
statistics): [`utils/zjet/README.md`](https://github.com/jhputschke/js-contrib/blob/main/utils/zjet/README.md).

## Another collision system: Pb+Pb at the LHC

`run_prod_jet.py` takes a copy of `AuAu_MCGlauber_MUSIC_0_10_jet.xml` with `--user-xml`, fills
in the job and checks it. So another system or energy is a copy of the XML with the
collision changed. Here: Pb+Pb at √s_NN = 5.02 TeV, 0–10 %; the same steps apply to other
nuclei and energies.

### 1. The XML

| where | Au+Au 200 GeV | Pb+Pb 5.02 TeV | why |
|---|---|---|---|
| `<IS><MCGlauber><projectile>`, `<target>` | `Au` | `Pb` | |
| `<IS><MCGlauber><sqrts>` | `200` | `5020` | |
| `<IS><MCGlauber><b_max>` | `4.7` fm | `4.95` fm | centrality comes from b inside X-SCAPE (`cenMin`/`cenMax` are ignored): the geometric 0–10 %, π b² = 0.1 × 7.7 b |
| `<IS><grid_max_x>`, `<grid_max_y>` | `15` | `20` | MUSIC's grid: the fireball is larger and lives longer |
| `<IS><grid_max_z>` | `6` | `8` | MUSIC's η_s range: the beam rapidity is 8.5 instead of 5.4 |
| `<Hard><PythiaGun><eCM>` | `200` | `5020` | |
| the pT̂ windows | 10–40 GeV | your choice | `--pthat-bins` |

`eCMforHadronization` follows the collision automatically, so the one `hadronize.xml` serves
every system. From a js-contrib checkout:

```bash
mkdir -p xml
T=contribs/PyJetscape/example/config
sed -e 's#<projectile>Au</projectile>#<projectile>Pb</projectile>#' \
    -e 's#<target>Au</target>#<target>Pb</target>#' \
    -e 's#<sqrts>200</sqrts>#<sqrts>5020</sqrts>#' \
    -e 's#<b_max>4.7</b_max>#<b_max>4.95</b_max>#' \
    -e 's#<grid_max_x> 15 </grid_max_x>#<grid_max_x> 20 </grid_max_x>#' \
    -e 's#<grid_max_y> 15 </grid_max_y>#<grid_max_y> 20 </grid_max_y>#' \
    -e 's#<grid_max_z> 6  </grid_max_z>#<grid_max_z> 8  </grid_max_z>#' \
    -e 's#<eCM>200</eCM>#<eCM>5020</eCM>#' \
    $T/AuAu_MCGlauber_MUSIC_0_10_jet.xml > xml/PbPb_0_10_jet.xml
diff $T/AuAu_MCGlauber_MUSIC_0_10_jet.xml xml/PbPb_0_10_jet.xml     # 8 lines must differ
```

The result is already in `example/config/` as `PbPb_0_10_jet.xml`, with `grid_PbPb.yaml`;
use them as they are or as a start for another system.

**What the driver checks** (it refuses an XML that breaks the two-stage setup, before any
event): exactly two `<Hydro><MUSIC>` blocks, `MUSIC_1` (background) and `MUSIC_2` (jet leg), all
MUSIC settings in the first; the liquefier's `<dtau>` equal to MUSIC's starting step; a
`<Hard><PythiaGun>` block; no `<SoftParticlization>`, `<JetHadronization>`, `<Afterburner>`
or writers (the hadrons come from stage 2); an output grid inside MUSIC's grid.

**Choices to make before a production:**

- **The tune.** The 3D MC-Glauber, viscosity, freeze-out and energy-loss parameters are those
  of the Au+Au 200 GeV production; none is tuned at the LHC. Check the bulk against data
  first (ALICE: dN_ch/dη at \|η\| < 0.5 is 1943 ± 54 at 0–5 % and 1587 ± 47 at 5–10 %).
- **Pythia's MPI.** PythiaGun switches multi-parton interactions on, and every parton it hands
  over is showered and can deposit; at 5.02 TeV that is many more partons than at 200 GeV.
  `PartonLevel:MPI = off` in `<LinesToRead>` leaves the hard process and its ISR.
- **The output grid.** The default grid (±12.2 fm) fits, but the jet leg can reach its edge:
  `grid_PbPb.yaml` is larger with the same cells (files of different grids don't go into one
  training set).
- **Names.** `TAG_PREFIX=PbPb_0_10_seed` names the files `PbPb_0_10_…` (letters, digits,
  `.`, `_`, `-`, ending in `_seed`; one per OUTDIR), or put the system into the campaign name.

### 2. Your XML from outside the container

The XML doesn't need to be in the image. Bind its directory and give the driver the path **as
the container sees it**, as an absolute path:

| | bind | then |
|---|---|---|
| Docker | `-v "$PWD/xml:/xml:ro"` | `--user-xml /xml/PbPb_0_10_jet.xml` |
| Apptainer | `--bind "$PWD/xml:/xml"`, or nothing for a file under `$HOME` | `--user-xml /xml/PbPb_0_10_jet.xml` |
| SLURM (`slurm_prod_array.sh`, `slurm_hadronize.sh`) | `BINDS="$PWD/xml:/xml"` | in `EXTRA_ARGS` (stage 1) or `HAD_ARGS` (stage 2) |

The same holds for `--grid`, `--main-xml` and `--hadronize-xml`. The file is read when a job
starts, and each job stores the XML it ran (`<stem>.xml`, `prod_user_xml`): don't edit it
during a campaign.

### 3. A first event, then the campaign

```bash
# in the container (Apptainer; Docker with -v "$PWD/xml:/xml:ro" -v "$PWD/prod:/work")
PROD=/opt/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape/example/prod_AuAu_0_10_jet
apptainer exec --nv --env MUSIC_CUDA_SYNC=block --pwd "$PROD" \
  --bind "$PWD/xml:/xml,$SCRATCH/prod:/work" xscape_prod.sif \
  python run_prod_jet.py --user-xml /xml/PbPb_0_10_jet.xml --events 3 --seed 1 \
    --write-particlize both --outdir /work/PbPb_test --out PbPb_0_10_jet_seed0001.h5 --dry-run
# ... the same without --dry-run, then stage 2
apptainer exec --pwd "$PROD" --bind "$PWD/xml:/xml,$SCRATCH/prod:/work" xscape_prod.sif \
  ./run_hadronize.py /work/PbPb_test -j 3 --oversample 100 --n-frag 20
```

One event against Au+Au, measured on the GB10 (one job alone, pT̂ 50–70 GeV,
`--write-particlize both`, default output grid, 100 oversamples):

| | Pb+Pb 5.02 TeV | Au+Au 200 GeV |
|---|---|---|
| wall time of the event (both legs) | 38.9 s | 17.1 s |
| peak host memory of the job | 15.7 GB | 6.1 GB |
| freeze-out cells per leg | 2.55 M | 1.0 M |
| showers, droplets, deposited energy | 18, 74, 336 GeV | 2, 25, 32 GeV |
| pair file, particlize file | 321 MB, 395 MB | 245 MB, 155 MB |
| hadron file per leg (100 oversamples) | 79 MB | 19 MB |
| dN_ch/dη, \|η\| < 0.5, background | 1534 ± 4 (ALICE 0–10 %: ~1765) | 577 ± 2 |

The campaign is the Au+Au one with the XML (and the grid) passed after OUTDIR:

=== "Docker"

    ```bash
    docker run -d --gpus all --user "$(id -u):$(id -g)" \
      -e OMP_NUM_THREADS=3 -e OMP_WAIT_POLICY=passive -e MUSIC_CUDA_SYNC=block \
      -e LBT_TABLE_CACHE=/work/lbt_cache/lbt_tables_v1.bin \
      -v "$PWD/xml:/xml:ro" -v "$HOME/prod:/work" $IMG \
      ./run_jobs.sh -j 4 --stagger 12 --campaign PbPb_c1 20 15 0 /work/PbPb_c1 \
        --user-xml /xml/PbPb_0_10_jet.xml --grid /xml/grid_PbPb.yaml \
        --pthat-bins 40-60,60-80,80-100 --jets-per-bin 5 --parton-ymax 0.6 --write-particlize both
    ```

=== "SLURM"

    ```bash
    # stage 1: ~16 GB per job, so fewer jobs per GPU than for Au+Au
    SIF=$PWD/xscape_prod.sif CAMPAIGN=PbPb_c1 P=5 BINDS="$PWD/xml:/xml" \
      EXTRA_ARGS="--user-xml /xml/PbPb_0_10_jet.xml --grid /xml/grid_PbPb.yaml --pthat-bins 40-60,60-80,80-100 --jets-per-bin 5 --parton-ymax 0.6 --write-particlize both" \
      sbatch --array=0-19 --cpus-per-task=20 --mem=96G utils/slurm_prod_array.sh
    # stage 2
    SIF=$PWD/xscape_prod.sif CAMPAIGN=PbPb_c1 \
      HAD_ARGS="--oversample 200 --n-frag 50 --keep-bits-p 12 --keep-bits-x 8" \
      sbatch --array=0-19 --mem=48G utils/slurm_hadronize.sh
    ```

- **Memory:** plan ~16 GB per job, 4–5 jobs on 96 GB; measure events/h with a short sweep of
  `P` on the first node.
- **Disk:** ~0.7 GB per event with a new background every event; a background shared by 15
  jets saves most of the background's share. Hadrons are ~4× Au+Au per oversample, so 200
  oversamples give the statistics of 400–500 in Au+Au.
- **Stage 2 memory:** ~2.3 GB per process at 100 oversamples.
- Hydro-only Pb+Pb (`prod_AuAu_0_10/run_prod.py`) takes the same `<IS>` changes in a copy of
  its XML.

Here the grid is bound from `xml/`: copy `grid_PbPb.yaml` there, or pass the image's own
path. Full details:
[`docs/README_2stage.md`, §9](https://github.com/jhputschke/js-contrib/blob/main/docs/README_2stage.md#9-another-collision-system-pbpb-at-the-lhc).

## Planning a production: the wizard

[`utils/prodwizard.py`](https://github.com/jhputschke/js-contrib/blob/main/utils/prodwizard.py)
plans a campaign before anything runs. Pick a production preset and a machine; it writes the
launch command with the settings measured for that machine (`launch_2gpu.sh` on a two-GPU
workstation, `run_jobs.sh` on a single machine or Mac, `slurm_prod_array.sh` on a cluster)
and estimates events/h, duration, disk and memory, each marked as measured or estimated with
its source. It checks the plan: events per job a multiple of the pT̂ windows × jets, memory,
disk, upload rate, the image, the campaign name. It never starts jobs.

```bash
utils/prodwizard.py --list                 # presets and machines
utils/prodwizard.py --preset auau_jet --machine 2x3090 --events 2700 --campaign c5 --out /data/c5
utils/prodwizard.py --preset pbpb_zjet --machine slurm_a40 --events 19200 --gpus 20 \
    --campaign zjet_c10 --save launch_zjet_c10.sh      # a script, the plan in its header
utils/prodwizard.py                        # the interactive wizard: a form, the plan beside it
```

| presets (`--preset`) | |
|---|---|
| `auau_jet` | Au+Au jets, pair + particlize files (the campaign setting) |
| `auau_particlize_only` | Au+Au jets, particlize file only |
| `auau_hadrons_only` | Au+Au jets, hadron files only, in the job |
| `oo_hadrons`, `oo_full` | O+O with ISR: hadron files only; or the default outputs |
| `pbpb_zjet` | Pb+Pb 5.02 TeV 0–30 % Z+jet, hadron files only |

The machines (`--machine`) are `2x3090`, `gb10`, `m3max` and SLURM nodes with one
`slurm_v100`, `slurm_a40`, `slurm_a100` or `slurm_l40s` per array task; further options set
`--events-per-job`, `--image`, `--first-seed`, `--jobs-per-gpu`, `--omp`, `--upload`,
`--delete` and `--min-free`. The command-line mode needs only the standard library; the
interactive wizard needs `textual` (in the analysis venv), and runs in a browser with
`textual serve "utils/prodwizard.py"`. The presets and measured rates are in
`utils/prod_presets.json`.

## Next steps

- [Clusters, GPUs and monitoring](clusters.md): SLURM, several GPUs, `jobmon.py`.
- [Hadrons](../analysis/hadrons.md) and [Jets with FastJet](../analysis/jets.md): reading
  the outputs.
