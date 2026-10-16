# Z+jet campaigns (docs/Plans/PLAN_ZJET_CMS10x.md, sections 5a–5b)

Ten campaigns `zjet_c00` … `zjet_c09`, 9,600 Pb+Pb 5.02 TeV 0–30 % Z+jet events each: 6 p̂T
windows (20–30, 30–40, 40–60, 60–80, 80–120, 120–200 GeV) × 4 jets per window per background = 24
events per background, 96 events per job (4 backgrounds), 100 jobs; 96k events, 16k per window.
Seeds: campaign C uses 1 + 1000 C … 1000 C + 100, so campaigns from Delta and from the 3090 box
combine and never overlap.

The selection (decided 2026-10-15, docs/Plans/PLAN_ZJET_CMS10x.md):

- **0–30 %**, the class of all three CMS measurements (`b_max` 8.57 fm); every event is weighted
  by its `n_coll` (below). Centrality is selected on b only: X-SCAPE ignores `cenMin`/`cenMax`
  (3dMCGlauber applies them only in its standalone program). Checked with 200,000 minimum-bias
  events of the same `<MCGlauber>` settings (2026-10-16,
  `contribs/PyJetscape/example/analysis/zjet_test/centrality_check.py`): this model's own σ_inel is 7.89 b (3dMCGlauber, σ_NN 67.6 mb), not the 7.7 b behind b_max, so b < 8.57 fm is its 0–29.3 % (exactly 0–30 %: 8.68 fm); classes defined by event activity (the string count, which is 3dMCGlauber's own centrality cut, N_part, or a two-component N_part/N_coll mix) agree with the b class for 96–97 % of their events, with the same ⟨N_coll⟩ (1,030) and N_coll-weighted ⟨b⟩ (4.9 fm). The
  cut is a good stand-in for an activity-based class; CMS's HF-energy class is not modelled
  beyond that.
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
| `zjet_slurm_campaign.sh` | Delta: `C=<n> sbatch zjet_slurm_campaign.sh` (10 A40s, `P=3`, ~3–4.5 h, estimate) |
| `zjet_3090_campaign.sh` | 3090 box: `IMAGE=… ./zjet_3090_campaign.sh <n>` (3 jobs per GPU, ~14–19 h, estimate) |
| `zjet_3090_test.sh` | 3090 box, before campaign 0: `IMAGE=… ./zjet_3090_test.sh info \| bench 3 \| bench 4 \| report \| smoke` (steps 1, 2, 4, 5 below) |

**Run them from a checkout of this repo** (e.g. `git clone -b zjet https://github.com/jhputschke/js-contrib`
on the box, then `utils/zjet/…`): the 3090 scripts use `../launch_2gpu.sh` of the same checkout
(`LAUNCH=…` for another), the Slurm one the `slurm_prod_array.sh` next to where it was submitted
from (`utils/zjet`, `utils` or the repo top; `SCRIPT=…` for another). Both refuse a launcher
that does not pass `TAG_PREFIX` on (copies from before 2026-10-16, e.g. an old
`~/prod_test/launch_2gpu.sh`).

**File names:** `TAG_PREFIX=PbPb_0_30_seed`, so `PbPb_0_30_zjet_cNN_seedNNNN.*` on the box and
`PbPb_0_30_zjet_cNN_tTTT_seedNNNN.*` on Delta, not `AuAu_0_10_jet_*` (README_launch.md,
`TAG_PREFIX`). This needs an image of js-contrib 2026-10-16 or later; the launchers check it
before any job starts.

**Cost (GB10, 2026-10-15; [docs/BENCHMARK_PbPb_vs_AuAu.md](../../docs/BENCHMARK_PbPb_vs_AuAu.md)):**
a Pb+Pb Z+jet job takes 2.1× the time of an Au+Au jet job (323 against 693 events/h, 2 jobs
at once), peaks at **14–16 GB** (2.1×) and writes 1.7× the disk. The memory is mostly MUSIC's
larger box (x, y ±20 fm, η_s ±8: 2.4× the cells): the background's evolution is kept in
memory on the whole grid. The launchers' defaults (3 jobs per 3090, `P=3` per A40) and the
times above follow from these GB10 numbers and the box's Au+Au rates: ~500–720 events/h on
the box, all ten campaigns in 6–8 days ([PLAN_ZJET_CMS10x.md](../../docs/Plans/PLAN_ZJET_CMS10x.md)
§6); **to be confirmed on the RTX box** (step 4 below).

The inputs are the image's own `contribs/PyJetscape/example/config/` files, so they always match
its code: `PbPb_0_30_ZJet.xml` (0–30 %, the centrality class of the CMS measurements; the Z+jet
`<PythiaGun>`: Z + parton processes, Z → ee/μμ, `<trigger>Z`, `<minPartons>1`) and `grid_PbPb.yaml` (the Pb+Pb output grid, for the edge flags).
The hadronization is the shared `config/hadronize.xml` (iSS decaying the SMASH list for EOS 91),
with `eCMforHadronization` the collision's 5020 from the XML's PythiaGun: no second XML.
`XML_DIR=…` points the scripts at copies instead.

## Before campaign 0 (once)

1. **The image is mandatory:** the Z trigger, `--hadrons-only` and the `TAG_PREFIX` naming exist
   only in an image with the `zjet` changes (plan 5a step 9); they cannot be bind-mounted in
   (C++ and compiled bindings). Before the PRs: the test image
   `jhputschke/xscape-prod:cu124-zjet`, built by `docker-prod.yml` from the `zjet` branches with
   `moving_tag=false` (the moving `cu124` stays on the production image;
   `utils/BuildContainerProd.md`). After the PRs: a build from X-SCAPE `contrib` and js-contrib
   `main`. Pin it by digest:
   `apptainer pull xscape_prod_zjet.sif docker://jhputschke/xscape-prod@sha256:…` (Delta), `IMAGE=…`
   (box). Check: `BUILD_INFO.txt` lists the commits, `run_prod_jet.py --help` shows `--trigger`
   (`zjet_3090_test.sh info` does both, the dry run of step 2 and the `TAG_PREFIX` checks).
2. `--dry-run` inside the image (no GPU work):
   `python run_prod_jet.py --events 96 --seed 1 --user-xml $D/PbPb_0_30_ZJet.xml
   --grid $D/grid_PbPb.yaml --pthat-bins 20-30,30-40,40-60,60-80,80-120,120-200 --jets-per-bin 4
   --lepton-ptmin 20 --lepton-etamax 2.5 --hadrons-only --dry-run` with
   `D=/opt/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape/example/config`: the plan
   must say `trigger Z (lepton pT > 20 GeV, lepton |eta| < 2.5)`, `EOS 91` and
   `eCMforHadronization 5020`, and the job XML it writes (`<stem>.xml`) `<b_max>8.57`.
3. LBT cache: one 1-event job writes `lbt_tables_v1_amd64.bin` (Delta `/work/hdd`) /
   `lbt_tables_v1.bin` (box).
4. Benchmark (plan 5a step 10), first on the RTX box (`zjet_3090_test.sh bench 3`, `bench 4`,
   `report`; the test image is `jhputschke/xscape-prod:cu124-zjet`): 24 events/job (one background: 6
   windows × 4 jets) at 3 and 4 jobs per GPU; set `JOBS_PER_GPU` / `P` from the memory peak
   (`mem.log` / `sacct -o MaxRSS`, with ~10 % headroom for jobs peaking together), note
   events/h, and replace the estimates here and in the scripts. The defaults (3) are from the
   GB10's 14–16 GB per job; on discrete GPUs the hydro's arrays are in device memory, so the
   host peak may be lower (Au+Au peaks the same on both, 6.4–7.6 GB). VRAM: ~1 GB per Au+Au
   job on a 3090 (9.6 GB at 10 per GPU), so ~2.4 GB per Pb+Pb job: no limit at 3–4 per GPU.
5. Smoke test = one CMS-equivalent: `C=0 NJOBS=1 sbatch zjet_slurm_campaign.sh` (Delta) or
   `./zjet_3090_campaign.sh 0 10` (box; `zjet_3090_test.sh smoke`), ~1,000 events, files
   `PbPb_0_30_zjet_c00_….*`. Check: two leptons per event in
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

## Analysis and movies

- `contribs/PyJetscape/example/analysis/zjet_wake.ipynb` (helpers in `zjet_wake.py`): the
  Z–hadron wake in the Z frame (Δη = η_h − y_Z, Δφ = φ_h − φ_Z) as CMS 2507.09307 measures it:
  (Δη, Δφ) maps, Δφ and Z-side Δη distributions per track p_T class, Pb+Pb − pp with the vacuum
  reference, the p_T^Z dependence, one event's hydro wake. Events are weighted by window
  cross section × N_coll; the Z's leptons are removed from the tracks.
- `contribs/PyJetscape/example/analysis/zjet_test/`: a small test set to run the notebook and
  the movies on (96 Pb+Pb events, the pp reference, one movie event) and the commands that
  make it (`make_data.sh`); data and figures are git-ignored.
- Movies of single events need a pair file, so a few events run without `--hadrons-only`
  (`--write-particlize both`): `contribs/Visualization/wake_pyvista.py` and
  `hydro_jet_particles_pyvista.py` draw the Z's leptons as straight tracks and an arrow along
  the Z (Visualization README, "Z+jet events").
- **Correlated sampling needs equal sample counts on both legs.** With
  `--oversample-bg per-pthat-bin` (160 background samples for 40 per event) iSS samples the
  background *uncorrelated* with the jets (measured 2026-10-15: sd(J_k − B_k) equals that of
  mismatched pairs), and paired errors refuse such files; with 40 and 40 the legs are strongly
  correlated (sd(J_k − B_k) ≈ 19 against ≈ 50 mismatched).

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
