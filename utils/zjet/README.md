# Z+jet campaigns (docs/Plans/PLAN_ZJET_CMS10x.md, sections 5a–5b)

Ten campaigns `zjet_c00` … `zjet_c09`, 10,000 Pb+Pb 5.02 TeV Z+jet events each: 4 p̂T windows
(40–60, 60–80, 80–120, 120–200 GeV) × 5 jets per window per background = 20 events per background,
100 events per job (5 backgrounds), 100 jobs. One campaign ≈ 10× CMS's Z count per p_T^Z bin
(~3× its precision); ten ≈ 100× (10× precision). Seeds: campaign C uses 1 + 1000 C … 1000 C + 100,
so campaigns from Delta and from the 3090 box combine and never overlap.

| file | use |
|---|---|
| `zjet_slurm_campaign.sh` | Delta: `C=<n> sbatch zjet_slurm_campaign.sh` (10 A40s, ~2.5 h) |
| `zjet_3090_campaign.sh` | 3090 box: `IMAGE=… ./zjet_3090_campaign.sh <n>` (~9 h) |

The inputs are the image's own `docs/PbPb_5020/` files, so they always match its code:
`PbPb_0_10_ZJet.xml` (the Z+jet `<PythiaGun>`: Z + parton processes, Z → ee/μμ, `<trigger>Z`,
`<minPartons>1`), `hadronize_5020.xml` (Colorless with `eCMforHadronization` 5020, iSS decaying the
SMASH list for EOS 91) and `grid_PbPb.yaml` (the Pb+Pb output grid, for the edge flags).
`XML_DIR=…` points the scripts at copies instead.

## Before campaign 0 (once)

1. **The image is mandatory:** the Z trigger and `--hadrons-only` exist only in an image built
   from X-SCAPE `contrib` and js-contrib `main` with the `zjet` changes (plan 5a step 9); they
   cannot be bind-mounted in (C++ and compiled bindings). Pin it by digest:
   `apptainer pull xscape_prod_zjet.sif docker://jhputschke/xscape-prod@sha256:…` (Delta), `IMAGE=…`
   (box). Check: `BUILD_INFO.txt` lists the commits, `run_prod_jet.py --help` shows `--trigger`.
2. `--dry-run` inside the image (no GPU work):
   `python run_prod_jet.py --events 20 --seed 1 --user-xml $D/PbPb_0_10_ZJet.xml --hadronize-xml
   $D/hadronize_5020.xml --grid $D/grid_PbPb.yaml --pthat-bins 40-60,60-80,80-120,120-200
   --jets-per-bin 5 --trigger-ymax 2.4 --hadrons-only --dry-run` with
   `D=/opt/X-SCAPE/external_packages/js-contrib/docs/PbPb_5020`: the plan line must say
   `trigger Z (…)` and `EOS 91`.
3. LBT cache: one 1-event job writes `lbt_tables_v1_amd64.bin` (Delta `/work/hdd`) /
   `lbt_tables_v1.bin` (box).
4. Benchmark (plan 5a step 10): 15 events/job at several jobs per GPU; set `P` / `JOBS_PER_GPU`
   from the memory peak (`sacct -o MaxRSS` / `mem.log`), note events/h. On the GB10 a 4-event
   Pb+Pb Z+jet `--particlize-only` job peaked at 13.5 GB, not the plan's 8–9 GB: expect fewer
   jobs per GPU than the scripts' defaults (6).
5. Smoke test = one CMS-equivalent: `C=0 NJOBS=1 sbatch …` (Delta) or
   `./zjet_3090_campaign.sh 0 10` (box), ~1,000 events. Check: two leptons per event in
   `initiators/` and in every `jet_frag` sample, `units/z_pt` ≥ ~the window's lower edge for most
   events, `pthat_bin_acceptance` in the `.json`s (~0.45–0.6 with `--parton-ymax 1.9`),
   `droplets_total/events_written`, no edge flags, R_jZ of a few hundred events sane.

## Per campaign

1. Submit / launch; resubmit the same command to resume (finished jobs are skipped).
2. After the jobs: `utils/check_deposition.py <OUTDIR>` → move flagged jobs away, resubmit.
3. Upload: `js_osdf.py upload <OUTDIR> osdf:///fno4hic/zjet/zjet_cNN` (or `--upload` in
   `launch_2gpu.sh`); ~35 GB per campaign at 40 samples (`--charged --no-x`, 12/8 bits).
4. Record in the campaign log: image digest, events/h by makespan, flagged jobs.

## In the files

- The Z per event: `units/` of `bulk_jet` and `jet_frag` (`z_pt, z_y, z_phi, z_m,
  lep{1,2}_{pid,pt,eta,phi}`, `jetscape.hadrons_h5.Z_COLUMNS`); `HadronFileReader`'s
  `EventInfo.z`. File attributes `trigger = "Z"`, `trigger_ymax`, `lepton_ptmin`,
  `lepton_etamax`, `min_partons`.
- The two leptons are in `jet_frag` (pid ±11/±13; `--charged` keeps them) and in `initiators/`:
  **remove them before a track or jet analysis** (match `lep{1,2}_*`).
- `pthat_bin_sigma_gen` is Pythia's σ × kept/tried of both cuts (`--parton-ymax` and the trigger).

## pp reference

`run_prod_vacuum.py` (plan 5a step 8, not written yet), 10,000 events with the same XML and
windows, no hydro: minutes on any machine; one file set `zjet_pp`.
