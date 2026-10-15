# Z+jet campaigns (docs/Plans/PLAN_ZJET_CMS10x.md, sections 5a–5b)

Ten campaigns `zjet_c00` … `zjet_c09`, 9,600 Pb+Pb 5.02 TeV 0–30 % Z+jet events each: 6 p̂T
windows (20–30, 30–40, 40–60, 60–80, 80–120, 120–200 GeV) × 4 jets per window per background = 24
events per background, 96 events per job (4 backgrounds), 100 jobs; 96k events, 16k per window.
Seeds: campaign C uses 1 + 1000 C … 1000 C + 100, so campaigns from Delta and from the 3090 box
combine and never overlap.

The selection (decided 2026-10-15, docs/Plans/PLAN_ZJET_CMS10x.md):

- **0–30 %**, the class of all three CMS measurements (`b_max` 8.57 fm); every event is weighted
  by its `n_coll` (below).
- **Both decay leptons p_T > 20 GeV, |η| < 2.5** (`--lepton-ptmin 20 --lepton-etamax 2.5`): the
  lepton cuts of CMS Z–hadron (2103.04377), CMS Z–hadron Δφ/Δy (2507.09307) and ATLAS
  (2008.09811); CMS Z+jet's 10–20 GeV muons are not generated. Events failing it are regenerated
  before any hydro.
- **No Z rapidity cut** in the generator (2507.09307's |y_Z| < 2.4 is applied offline); no
  p_T^Z cut either: every paper's p_T^Z and mass window is applied offline from `units/z_*`.
- **p̂T from 20 GeV**: the windows hold 99.1 % of the p_T^Z > 40 and 97 % of the p_T^Z > 30 cross
  section (table below); ATLAS's 15–30 GeV p_T^Z bin is out of reach (Z + parton matrix element).

### Where the p_T^Z > 40 sample comes from

p̂T is the hard scattering's, not the Z's: with ISR, a Z of given p_T comes from a spread of p̂T.
With this selection (both leptons p_T > 20 GeV, |η| < 2.5; hardest parton |y| < 1.9; standalone
Pythia 8.312 with PythiaGun's settings, no medium, 20,000 events per window, 2026-10-15):

| p̂T window [GeV] | σ × acceptance, p_T^Z > 40 | share of p_T^Z > 40 | the window's accepted events with p_T^Z > 40 |
|---|---|---|---|
| 10–20 (not produced) | 0.02 pb | 0.1 % | 0.0 % |
| **20–30** | 1.46 pb | **3.9 %** | 4.2 % |
| **30–40** | 5.48 pb | **14.6 %** | 29 % |
| 40–60 | 15.8 pb | 42.0 % | 85 % |
| 60–80 | 7.56 pb | 20.1 % | 99 % |
| 80–120 | 5.10 pb | 13.6 % | 99.8 % |
| 120–200 | 1.88 pb | 5.0 % | 100 % |
| 200–400 (not produced) | 0.31 pb | 0.8 % | 100 % |
| **total** | **37.6 pb** | | |

- **The windows 20–30 and 30–40 supply 18.5 % of the p_T^Z > 40 cross section**, at p_T^Z just
  above 40 where the steep spectrum has most Zs; windows from 40 GeV held only 81 %. The six
  windows cover 99.1 %; missing are p̂T > 200 (0.8 %; 1.8 % of p_T^Z > 60) and 10–20 (0.1 %).
- **Few of the low windows' events pass p_T^Z > 40** (4 % of 20–30, 29 % of 30–40): with 16k
  events per window the p_T^Z > 40 sample has ~67k events, and since those from low windows carry
  large weights (the window's σ / N), its effective size for a cross-section-weighted p_T^Z > 40
  quantity is ~42k (before the N_coll weights).
- **p_T^Z > 30** (CMS 2103.04377): shares 2.4 / 16.0 / 25.6 / 30.7 / 12.9 / 8.7 / 3.2 / 0.5 % for
  the windows 10–20 … 200–400; the six windows cover 97 %, and the 20–30 window serves mainly this
  sample.
- **p_T^Z > 60**: 0.0 / 0.1 / 1.4 / 19.0 / 37.5 / 29.4 / 10.9 / 1.8 %.

| file | use |
|---|---|
| `zjet_slurm_campaign.sh` | Delta: `C=<n> sbatch zjet_slurm_campaign.sh` (10 A40s, ~2.5 h) |
| `zjet_3090_campaign.sh` | 3090 box: `IMAGE=… ./zjet_3090_campaign.sh <n>` (~9 h) |

The inputs are the image's own `contribs/PyJetscape/example/config/` files, so they always match
its code: `PbPb_0_30_ZJet.xml` (0–30 %, the centrality class of the CMS measurements; the Z+jet
`<PythiaGun>`: Z + parton processes, Z → ee/μμ, `<trigger>Z`, `<minPartons>1`) and `grid_PbPb.yaml` (the Pb+Pb output grid, for the edge flags).
The hadronization is the shared `config/hadronize.xml` (iSS decaying the SMASH list for EOS 91),
with `eCMforHadronization` the collision's 5020 from the XML's PythiaGun: no second XML.
`XML_DIR=…` points the scripts at copies instead.

## Before campaign 0 (once)

1. **The image is mandatory:** the Z trigger and `--hadrons-only` exist only in an image built
   from X-SCAPE `contrib` and js-contrib `main` with the `zjet` changes (plan 5a step 9); they
   cannot be bind-mounted in (C++ and compiled bindings). Pin it by digest:
   `apptainer pull xscape_prod_zjet.sif docker://jhputschke/xscape-prod@sha256:…` (Delta), `IMAGE=…`
   (box). Check: `BUILD_INFO.txt` lists the commits, `run_prod_jet.py --help` shows `--trigger`.
2. `--dry-run` inside the image (no GPU work):
   `python run_prod_jet.py --events 96 --seed 1 --user-xml $D/PbPb_0_30_ZJet.xml
   --grid $D/grid_PbPb.yaml --pthat-bins 20-30,30-40,40-60,60-80,80-120,120-200 --jets-per-bin 4
   --lepton-ptmin 20 --lepton-etamax 2.5 --hadrons-only --dry-run` with
   `D=/opt/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape/example/config`: the plan
   must say `trigger Z (lepton pT > 20 GeV, lepton |eta| < 2.5)`, `EOS 91` and
   `eCMforHadronization 5020`, and the job XML it writes (`<stem>.xml`) `<b_max>8.57`.
3. LBT cache: one 1-event job writes `lbt_tables_v1_amd64.bin` (Delta `/work/hdd`) /
   `lbt_tables_v1.bin` (box).
4. Benchmark (plan 5a step 10): 24 events/job (one background: 6 windows × 4 jets) at several jobs per GPU; set `P` / `JOBS_PER_GPU`
   from the memory peak (`sacct -o MaxRSS` / `mem.log`), note events/h. On the GB10 a 4-event
   Pb+Pb Z+jet `--particlize-only` job peaked at 13.5 GB, not the plan's 8–9 GB: expect fewer
   jobs per GPU than the scripts' defaults (6).
5. Smoke test = one CMS-equivalent: `C=0 NJOBS=1 sbatch …` (Delta) or
   `./zjet_3090_campaign.sh 0 10` (box), ~1,000 events. Check: two leptons per event in
   `initiators/` and in every `jet_frag` sample, `units/z_pt` ≥ ~the window's lower edge for most
   events, `units/n_coll` filled, `pthat_bin_acceptance` in the `.json`s (~0.4–0.65 with
   `--parton-ymax 1.9`, lowest in the 20–30 window), `droplets_total/events_written`, no edge
   flags, R_jZ of a few hundred events sane.

## Per campaign

1. Submit / launch; resubmit the same command to resume (finished jobs are skipped).
2. After the jobs: `utils/check_deposition.py <OUTDIR>` → move flagged jobs away, resubmit.
3. Upload: `js_osdf.py upload <OUTDIR> osdf:///fno4hic/zjet/zjet_cNN` (or `--upload` in
   `launch_2gpu.sh`); ~35 GB per campaign at 40 samples (`--charged --no-x`, 12/8 bits).
4. Record in the campaign log: image digest, events/h by makespan, flagged jobs.

## In the files

- The Z per event: `units/` of `bulk_jet` and `jet_frag` (`z_pt, z_y, z_phi, z_m,
  lep{1,2}_{pid,pt,eta,phi}`, `jetscape.hadrons_h5.Z_COLUMNS`); `HadronFileReader`'s
  `EventInfo.z`. File attributes `trigger = "Z"`, `lepton_ptmin`, `lepton_etamax`,
  `min_partons`.
- **Weight every event by its `n_coll`** (`units/n_coll`, `EventInfo.n_coll`, the 3dMCGlauber
  event's binary collisions): the geometry is drawn with P(b) ∝ b over 0–30 % and every event
  gets one Z, while Z production scales with N_coll (CMS: 78 % of all Zs fall in 0–30 %). A
  narrower class (e.g. ATLAS's 0–10 %) is a cut on the geometry, weighted the same way.
- The two leptons are in `jet_frag` (pid ±11/±13; `--charged` keeps them) and in `initiators/`:
  **remove them before a track or jet analysis** (match `lep{1,2}_*`).
- `pthat_bin_sigma_gen` is Pythia's σ × kept/tried of both cuts (`--parton-ymax` and the trigger).

## pp reference

`contribs/PyJetscape/example/prod_AuAu_0_10_jet/run_prod_vacuum.py` (plan 5a step 8): the same
XML's PythiaGun and Matter in vacuum (`<in_vac> 1`, recoils off), no initial state, hydro, LBT or
liquefier, Colorless with eCMforHadronization = the PythiaGun's eCM; ~0.02 s per event with
4 fragmentations on the GB10, so 12,000 events take minutes on any machine (CPU only). With
`D=/opt/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape/example/config` inside the image:

    python run_prod_vacuum.py --events 12000 --seed 900001 --campaign zjet_pp \
      --user-xml $D/PbPb_0_30_ZJet.xml --pthat-bins 20-30,30-40,40-60,60-80,80-120,120-200 \
      --parton-ymax 1.9 --lepton-ptmin 20 --lepton-etamax 2.5 \
      --n-frag 40 --charged --eta-max 2.6 --no-x --hadron-keep-bits-p 12 --outdir /data/zjet_pp

One `<stem>_hadrons_jet_frag.h5` (attribute `vacuum`), read like the Pb+Pb jet files
(`HadronFileReader`, `EventInfo.z`). Seeds of its own: the Pb+Pb campaigns' seeds would give the
same Pythia events (fine for a paired comparison, not for statistics).
