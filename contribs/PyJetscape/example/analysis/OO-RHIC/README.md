# OO-RHIC: the RHIC p+p 200 GeV hard tune

The hard sector of X-SCAPE's ISR chain for p+p at √s = 200 GeV, tuned to high-pT hadron and jet
data: the p+p reference that the O+O 200 GeV jet production (`prod_OO_isr`) takes its hard
settings from. It follows the plan on the "RHIC p+p and O+O Tuning" page (steps 1 and 2).

**Model results, not vetted physics results.** State the campaign, cuts and number of events
with any number taken from here.

## What is tuned

| | |
|---|---|
| chain | 3dMCGlauber (p+p, positions and beam remnants) → PythiaIsrGun → iMATTER (ISR) → Matter in vacuum → iColored hadronization; `prod_OO_isr/run_prod_oo_isr.py --vacuum` with `config/pp200_isr_vacuum.xml` |
| varied | α_s(M_Z) of the hard process and the MPI (0.11–0.20), C_v = `vir_factor` (0.16–0.50), Q0 (1.0–2.0 GeV); iMATTER and Matter read C_v and Q0 |
| fixed | `PDF:pSet = 20` (NNPDF3.1 NNLO, the set iMATTER has hard-coded), Λ_QCD = 0.2 GeV, `probStoUD` 0.217 (Pythia's default), `MultipartonInteractions:pTmin = 5`, which Pythia lowers to 1.7 GeV at 200 GeV (its MPI needs σ_int > 1.1 σ_ND; `best_mpi4` = `best`) |
| cut scheme | pT̂ windows 5–7, 7–9, 9–11, 11–14, 14–18, 18–23, 23–30, 30–40, 40–55, 55–80 GeV, weighted by Pythia's cross section per window; 5 GeV is above iMATTER's floor Q0/√C_v for every design point |
| fit data | PHENIX π⁰ E d³σ/dp³, \|η\| < 0.35, 6–20 GeV (PRD 76, 051106; HEPData ins749394); STAR inclusive jets R = 0.5, d²σ/dp_T dη, \|η\| < 0.5 and 0.5 < \|η\| < 0.9, 8.2–52 GeV (arXiv:2603.28695; ins3137518) |
| references | JETSCAPE PP19 (arXiv:1910.05481) at 200 GeV, Pythia 8.312 default (Monash) and STAR's Detroit tune, same windows and observables (not tuned) |
| validation | STAR π⁰ and η/π⁰, 0 < η < 1 (PRC 81, 064904; ins840766); STAR underlying event vs leading jet (PRD 101, 052004; ins1771348), `ue_check.py` |
| method | maximin Latin hypercube of 32 points × 300k events (+180k in the windows 5–23 GeV, for the π⁰ statistics in PHENIX's 0.5 GeV bins); Gaussian-process emulator per bin (ln σ); emulator σ scaled to leave-one-out pull rms 1; posterior with emcee; χ² of the simulated design points as a cross-check; validation runs at the posterior median (`best`), a reference point (`ref`), the ends of the α_s–C_v valley and with other MPI settings |

The fit ranges come from `truncation_check.py`: events with pT̂ < 5 GeV never give a π⁰ above
1.14 p̂T (so nothing above 5.7 GeV) or a jet above ~1.6 p̂T (so nothing above 8.2 GeV).

## Reproduce

In the `js_fno` environment (X-SCAPE `build_gpu` with PyJetscape). Events and observables go
to `~/prod/OO-RHIC_pp_hard` (`$OO_RHIC_PROD` overrides); code, design, HEPData tables and fit
results stay here.

**Extra packages for the fit (`tune.py`, the notebook).** `emcee` (the MCMC sampler; pure Python,
needs only numpy) and `scikit-learn` (the Gaussian-process emulator) are **not** installed by
default in any environment: neither is in js-contrib's analysis requirements
(`utils/analysis_env/requirements.txt`, the `pyjetscape[analysis]` extra), so the production
container, the Mac wheel and a fresh analysis venv lack them. Install them once with pip:

```bash
pip install emcee scikit-learn        # tested: emcee 3.1.6, scikit-learn 1.9.0
```

In the production container the image is read-only: install into the user site
(`pip install --user ...`, with `PYTHONUSERBASE` below `/work` so it persists) or use an analysis
venv. The rest (numpy, scipy, h5py, hdf5plugin, awkward, fastjet, pyyaml, matplotlib, jupyter) is
in the analysis requirements. Running events needs X-SCAPE (`run_runs.py`); `observables.py`,
`ue_check.py`, `tune.py` and the notebook only read files.

**Not in git** (`.gitignore`): `data/` (HEPData tables, `fetch_data.py` makes them again),
`design/` (`make_design.py`; the design is fixed by `config.DESIGN_SEED`, the extra runs by the
`--add` commands below) and `results/` (`truncation_check.py`, `tune.py`, `ue_check.py`).

```bash
conda activate js_fno
cd contribs/PyJetscape/example/analysis/OO-RHIC
python fetch_data.py                         # data/hepdata/*.yaml, data/data.json
python make_design.py                        # design/design.json, design/xml/p000..p031.xml

# truncation check (30k events, ~1 min on 10 cores)
python make_design.py --add trunc 0.155 0.33 1.5 --jobs 10 --windows 4-5,5-7,7-9,9-11,11-14,14-18
python run_runs.py trunc -j 10 && python observables.py trunc && python truncation_check.py

# the design: 32 points x 10 jobs x 30k events, ~80 min on 20 cores (GB10, one core per job)
python run_runs.py --design -j 20
python run_runs.py --low -j 20               # + 30k events per window 5-23 GeV (pi0 statistics)
python observables.py --design -j 8           # PROD/obs/pNNN.npz
python tune.py                               # results/summary.json, chains, emulator, ...

# validation runs (values of the run of 2026-11-22): the posterior median (the MAP sits on the
# box edge Cv = 0.5 of a flat alpha_s-Cv valley) and the reference point, 30 jobs x 30k events
python make_design.py --add best 0.1530 0.319 1.52 --jobs 30     # results/summary.json "median"
python make_design.py --add ref 0.15 0.25 2.0 --jobs 30
# is fixing the MPI cut justified? the best point with the cut at 4 GeV and without MPI
python make_design.py --add best_mpi4 0.1530 0.319 1.52 --jobs 20 --pythia-line "MultipartonInteractions:pTmin = 4.0"
python make_design.py --add best_nompi 0.1530 0.319 1.52 --jobs 20 --pythia-line "PartonLevel:MPI = off"
# the ends of the valley (profile-likelihood alpha_s and Q0 at Cv = 0.16 and 0.5)
python make_design.py --add valley_lo 0.1493 0.16 1.51 --jobs 10
python make_design.py --add valley_hi 0.1559 0.50 1.49 --jobs 10
RUNS="best ref best_mpi4 best_nompi valley_lo valley_hi"
python run_runs.py $RUNS -j 20 && python observables.py $RUNS
# Pythia's full soft MPI in the X-SCAPE chain (vacuum; ~15x the CPU per event: 20 x 10k events, ~30 min)
python make_design.py --add best_mpidef 0.1530 0.319 1.52 --jobs 20 --pythia-line "MultipartonInteractions:pTmin = 0.2"
python run_runs.py best_mpidef --events 10000 -j 20 && python observables.py best_mpidef
# other p+p models for orientation: JETSCAPE PP19, Pythia 8 default (Monash), Detroit (STAR)
python refs/run_refs.py -j 20                # 3 x 20 jobs x 30k events, ~10 min
python observables.py pp19 pythia8_default pythia8_detroit
python ue_check.py trunc --check-truncation
python refs/run_refs.py pythia8_nocr --jobs 10 -j 10      # diagnostic: Pythia default without colour reconnection
python ue_check.py $RUNS best_mpidef pp19 pythia8_default pythia8_detroit pythia8_nocr -j 16   # results/ue_check.json
python ue_check.py $RUNS best_mpidef pp19 pythia8_default pythia8_detroit pythia8_nocr --cached   # again from results/ue_<run>.npz
# where in eta the particles are made: full-eta samples at pTHat 9-11 and 23-30 GeV (3000 events each;
# all models, the four MPI settings, Pythia with the tune's coupling, CR on/off; ~10 min, full MPI the longest)
python dndeta.py run -j 20 && python dndeta.py analyse                       # results/dndeta.json
P=$HOME/prod/OO-RHIC_pp_hard/diag/eta/9-11
python diag_strings.py $P/best_nompi $P/best $P/best_mpi4 $P/best_mpidef   # results/diag_strings.json
python conditional.py --fix Q0=2.0          # posterior at Q0 = 2 (Matter + LBT in O+O), emulator only
# R3 + R4: p+p reference from the O+O tune XML and the O+O vacuum closure (CPU, ~40 min on 20 cores, 3 GB)
python closure_oo.py run -j 20 && python closure_oo.py analyse -j 10
python centrality_proxy.py run -j 20 && python centrality_proxy.py analyse     # R4b part 1 (CPU, ~8 min)
# with the string switch (X-SCAPE oo_rhic_run): jet samples with --hard-pair-string-veto 0 (closure + subtraction alone)
python centrality_proxy.py run jet_sub_noveto jet_nosub_noveto -j 20 && python centrality_proxy.py analyse && python centrality_proxy.py smear
jupyter nbconvert --to notebook --execute closure_oo.ipynb --output results/closure_oo_executed.ipynb
# R4b full test and R4c (GPU, O+O with hydro; oo_bias.py --help): minimum bias, jets in vacuum, jets quenched
python oo_bias.py run mb -j 8 --mps && python oo_bias.py run jets -j 8 --mps     # ~9 h on the GB10
python oo_bias.py run jets_quench -j 8 --mps                                     # the jet in the medium
python oo_bias.py analyse && python oo_bias_jets.py && python oo_bias_jets.py --sample jets_quench && python oo_bulk.py analyse
# holes and partners: 4,500 quenched events with --store-partons --store-hadronization-input (~50 min on the GB10),
# negative hadrons per job (CPU), then the validation
python oo_bias.py run jets_holes -j 8 --mps
for f in ~/prod/OO-RHIC_OO/bias/jets_holes/*_hadrons_jet_frag.h5; do python ../../prod_OO_isr/negative_hadrons.py "$f"; done
python validate_holes.py
jupyter nbconvert --to notebook --execute oo_centrality_bias.ipynb --output-dir results --output oo_centrality_bias_executed.ipynb
jupyter nbconvert --to notebook --execute oo_bulk_comparison.ipynb --output-dir results --output oo_bulk_comparison_executed.ipynb
jupyter nbconvert --to notebook --execute --inplace tune_comparison.ipynb
# p+p / O+O bulk tune, step A (screening) and follow-ups, on a multi-GPU Docker host (one container per GPU)
./oo_bias_docker.sh --dev /data/OO-RHIC_bulk screen      # map, pp2407, m3: 10k p+p + 10k O+O each
./oo_bias_docker.sh --dev /data/OO-RHIC_bulk var         # pp2407 with yloss variance 0.3
./oo_bias_docker.sh --dev /data/OO-RHIC_bulk grid        # variance 0.4/0.45 x nodes 2.15/2.25
./oo_bias_docker.sh --dev /data/OO-RHIC_bulk rem         # alpha_rem 0.4/0.65/0.8/1.0 on g_v045_n215 (p+p)
./oo_bias_docker.sh --dev /data/OO-RHIC_bulk dau_map dau_g_v045_n215   # d+Au, the MAP's calibration data
./oo_bias_docker.sh --dev /data/OO-RHIC_bulk shad        # shadowing 0.6/0.8, ylossParam4At2 1.1 (d+Au, O+O)
./oo_bias_docker.sh --dev /data/OO-RHIC_bulk tau         # tau_form_mean 0.424 = the final XML (p+p, O+O, d+Au)
python pp_bulk.py fetch && python pp_bulk.py analyse --out /data/OO-RHIC_bulk   # UA5 + refs/rhic_pp200 (PHOBOS, STAR, PHENIX)
python oo_bulk.py analyse --sample oo_pp2407 --out /data/OO-RHIC_bulk          # per O+O sample
python centrality_norm.py
python dau_bulk.py fetch && python dau_bulk.py analyse --out /data/OO-RHIC_bulk   # d+Au vs PHENIX, STAR (TPC, FTPC), BRAHMS
python dau_bulk.py analyse --out /data/OO-RHIC_bulk --class-frac 1.1 dau_g_v045_n215   # classes as the top 110 %
python pt_bulk.py fetch && python pt_bulk.py analyse g_v045_n215 tau0424 map --out /data/OO-RHIC_bulk   # identified pT, <pT> vs STAR
# the adopted set g_v045_n215 is the 3dMCGlauber block of config/OO_MCGlauber_MUSIC_isr_rhictune.xml and
# config/pp200_isr_vacuum.xml since 2026-11-24, with tau_form_mean 0.424 (= sample tau0424; mb, jets,
# jets_quench run with it); the earlier sets name every parameter they change (map: oo_bias.MAP, and
# tau_form_mean 0.5), so their samples are unchanged
# the physics run (tuning page "Physics run plan"): phase 1 + 2 (mb, MPI check), phase 3 after gate G2
./oo_bias_docker.sh --dev /data/OO-RHIC_phys phys1      # mb first (its analysis runs during phys2)
./oo_bias_docker.sh --dev /data/OO-RHIC_phys phys2      # after docker rm oo_bias_gpu0 oo_bias_gpu1
./oo_bias_docker.sh --dev /data/OO-RHIC_phys phys3
OO_PHYS_CENTRAL_BMAX=<b_max of 0-20 %> ./oo_bias_docker.sh --dev /data/OO-RHIC_phys phys3_central
python g2_check.py --out /data/OO-RHIC_phys -j 16   # gate G2 after phase 2 -> results/g2_check.json
jupyter nbconvert --to notebook --execute g2_mpi_check.ipynb --output-dir results --output g2_mpi_check_executed.ipynb
python fwd_check.py --out /data/OO-RHIC_phys -j 16  # forward (EPD, MBD) activity of p+p jet events / minimum bias -> results/fwd_check.json
./oo_bias_docker.sh --dev /data/OO-RHIC_phys phys2_soft   # soft-MPI check: p+p with hydro at pT0Ref 1.42 / 1.45 / 1.48 (~1.2 h)
python soft_mpi_check.py --out /data/OO-RHIC_phys -j 16   # its UE vs STAR, forward ratios, pT0Ref at STAR's level -> results/soft_mpi_check.json
# its jet and hadron analysis: a family name stitches the one-window sub-samples (<name>_w<lo>_<hi>)
# by their cross sections; the minimum-bias reference is phys_mb
python oo_bias_jets.py --out /data/OO-RHIC_phys --sample oo_vac -j 16   # phase 3 (soft MPI) -> results/oo_bias_jets_oo_vac.json
python oo_bias_jets.py --out /data/OO-RHIC_phys --sample oo_med -j 16
python oo_bias_jets.py --out /data/OO-RHIC_phys --sample jets_vac -j 16  # phase 2, old MPI (two windows)
# the phase-3 analysis: selection (oo_vac) x quenching (oo_med / oo_vac) per class, R_AA vs pp_ref, STAR I_CP, sPHENIX R_CP
python oo_analysis_rhic.py --out /data/OO-RHIC_phys -j 16     # -> results/oo_analysis_rhic.json (+ per-sample caches; --cached reuses them)
jupyter nbconvert --to notebook --execute oo_analysis_rhic.ipynb --output-dir results --output oo_analysis_rhic_executed.ipynb
# hard-tune recheck with the new vacuum XML: the reference point with the new and the old block, same seeds
git show a3bb4c2:contribs/PyJetscape/example/config/pp200_isr_vacuum.xml > design/pp200_isr_vacuum_mapbulk.xml
python make_design.py --add ref_tune 0.15 0.25 2.0 --jobs 10
python make_design.py --add ref_mapbulk 0.15 0.25 2.0 --jobs 10 --seeds-of ref_tune --template design/pp200_isr_vacuum_mapbulk.xml
python run_runs.py ref_tune ref_mapbulk -j 12 && python observables.py ref_tune ref_mapbulk
python hard_recheck.py ref_tune ref_mapbulk && python ue_check.py ref_tune ref_mapbulk
jupyter nbconvert --to notebook --execute bulk_tune_screening.ipynb --output-dir results --output bulk_tune_screening_executed.ipynb
# why the vacuum UE is flat (ue_trend_cv.ipynb): standalone Pythia with ISR off, and the decomposition
# MPI off -> FSR off (matter_on 0) -> bare (also Q0 50: no iMATTER ISR), on best_nompi's XML and seeds
for ts in default:981000 detroit:982000; do t=${ts%:*}; for j in $(seq 0 9); do s=$((${ts#*:} + j)); \
  python refs/pythia_ref.py --tune $t --events 30000 --seed $s --line "PartonLevel:ISR = off" \
  --out $OO_RHIC_PROD/events/pythia8_${t}_isroff/pythia8_${t}_isroff_seed${s}_hadrons_jet_frag.h5; done; done
#   (the same with MPI off, + FSR off, + ISR off: runs pythia8_default_nompi, _nompi_fsroff, _nompi_bare)
python make_design.py --add best_nompi_fsroff 0.153 0.319 1.52 --jobs 10 --seeds-of best_nompi --pythia-line "PartonLevel:MPI = off"
python make_design.py --add best_nompi_bare 0.153 0.319 1.52 --jobs 10 --seeds-of best_nompi --pythia-line "PartonLevel:MPI = off"
for n in best_nompi_fsroff best_nompi_bare; do cp design/xml/best_nompi.xml design/xml/$n.xml; \
  sed -i 's|<matter_on> 1 </matter_on>|<matter_on> 0 </matter_on>|' design/xml/$n.xml; done
sed -i 's|<Q0> 1.52 </Q0>|<Q0> 50 </Q0>|' design/xml/best_nompi_bare.xml
python run_runs.py best_nompi_fsroff best_nompi_bare -j 20
python ue_check.py best_nompi_fsroff best_nompi_bare pythia8_default_isroff pythia8_detroit_isroff -j 16
jupyter nbconvert --to notebook --execute ue_trend_cv.ipynb --output results/ue_trend_cv_executed.ipynb
# the hard tune in the A+A jet scheme (scheme_recheck.ipynb), at the tune XML point, OO_RHIC_PROD=~/prod/OO-RHIC_aascheme_recheck
# (from example/, R=$OO_RHIC_PROD). Vacuum chain, pi0/eta undecayed; the pSet/alpha_s lines are needed (the vacuum XML has neither):
W=5-7,7-9,9-11,11-14,14-18,18-23,23-30
C0="--vacuum --user-xml config/pp200_isr_vacuum.xml --events 29400 --pthat-bins $W --eta-max 1.5 --max-skipped 30 --eloss-threads 0 \
  --hadronization-line 111:mayDecay=off --hadronization-line 221:mayDecay=off --pythia-line 'PDF:pSet = 20' --pythia-line 'SigmaProcess:alphaSvalue = 0.152'"
for j in $(seq 0 9); do s=$((933000 + j)); \
  eval python prod_OO_isr/run_prod_oo_isr.py $C0 --seed $s --outdir $R/events/vc_ref2; \
  eval python prod_OO_isr/run_prod_oo_isr.py $C0 --aa-vacuum --seed $s --outdir $R/events/vc_aa2; done
# full chain (2000 events per job, 4 jobs per arm): tune scheme (--vacuum-jet) against no QGP anywhere (Matter T0 = hydro_Tc = 100)
W=5-7,7-9,9-11,11-14,14-18,18-23,23-30,30-40,40-55,55-80
COMMON="--user-xml config/pp200_isr_rhictune.xml --events 2000 --pthat-bins $W --b-min 0 --b-max 3 --bulk-hadrons 1 --no-particlize \
  --no-hydro-file --eta-max 5.2 --max-skipped 30 --eloss-threads 0"
for j in 0 1 2 3; do s=$((930000 + j)); \
  python prod_OO_isr/run_prod_oo_isr.py $COMMON --seed $s --vacuum-jet --outdir $R/events/aa_ref_h; \
  python prod_OO_isr/run_prod_oo_isr.py $COMMON --seed $s --matter-only --xml-set Eloss/Matter/T0=100 \
    --xml-set Eloss/Matter/hydro_Tc=100 --outdir $R/events/aa_vac_h; done
python observables.py vc_ref2 vc_aa2 aa_ref_h aa_vac_h -j 2
jupyter nbconvert --to notebook --execute scheme_recheck.ipynb --output-dir results --output scheme_recheck_executed.ipynb
```

`run_runs.py` is resumable (finished jobs are skipped) and gives every job its own seed.

## Files

| file | |
|---|---|
| `config.py` | every setting of the tune: windows, parameters, events, data, fit ranges, paths |
| `fetch_data.py` | HEPData download (curl) and parsing → `data/data.json` |
| `make_design.py` | the design (maximin LHS) and the per-point user XMLs; `--add` for extra runs |
| `run_runs.py` | runs the jobs (`run_prod_oo_isr.py --vacuum`), resumable |
| `observables.py` | model observables in the data binning, per pT̂ window (π⁰, η/π⁰, STAR jets with off-axis-cone UE subtraction) |
| `truncation_check.py` | what the 5 GeV lowest window misses |
| `ue_check.py` | STAR's transverse charged density vs leading jet (R = 0.6) from the runs' hadrons |
| `diag_strings.py` | the colour-string topology iColored hands to Pythia (stored input) against standalone Pythia's parton level, CR on/off, tune vs default coupling; writes `results/diag_strings.json` for the notebook |
| `closure_oo.py`, `closure_oo.ipynb` | R3 + R4 of the plan forward: the p+p reference from the O+O tune XML (`config/OO_MCGlauber_MUSIC_isr_rhictune.xml`, projectile/target p) and the O+O vacuum closure in impact-parameter classes; the notebook explains the test and shows the figures (`PROD/closure/`, `results/closure_oo*.{json,npz}`) |
| `centrality_proxy.py` | R4b part 1: centrality bias from a string local-energy proxy (`smear`: negative-binomial fluctuations anchored to sPHENIX's MBD efficiency) (`run_prod_oo_isr.py`'s `strings_eloc_*` units) in STAR's estimator windows; minimum-bias 3dMCGlauber alone vs N_coll-weighted jet events, subtraction on/off → `results/centrality_proxy.json`; figures in `closure_oo.ipynb` |
| `oo_bias.py`, `oo_centrality_bias.ipynb` | R4b full test: O+O with hydro and one iSS sample per event, minimum bias and N_coll-weighted jet events (jet in vacuum: `--vacuum-jet`; `jets_quench`: in the medium), classes from the multiplicity in STAR's EPD, mid-rapidity and sPHENIX's MBD; R_AA^vac, bias factor, R_CP with errors → `results/oo_bias.json` |
| `oo_bias_jets.py` | the same against the jet p_T and the charged-hadron p_T: anti-k_T R = 0.4 jets (\|η\| < 0.7; charged jets for `--charged` samples, full jets for `jets_quench`) and charged hadrons (\|η\| < 1) on the hard event's hadrons, per p̂_T window and stitched over the windows (a window enters a bin only with ≥ `--min-entries` (10) raw entries in every class); class sums for quenched/vacuum double ratios; LBT hole energies of the colour audit → `results/oo_bias_jets.json`, `results/oo_bias_jets_jets_quench.json`. `--sample` also takes a comma-separated list or a physics-run family (`jets_vac`, `jets_med`, ...): its one-window sub-samples `<name>_w<lo>_<hi>` (own directories) are stitched on one window list by their cross sections, minimum bias `phys_mb` (`--mb` overrides) |
| `validate_holes.py` | validation of the hole and partner treatment on the `jets_holes` sample (made with `--store-partons --store-hadronization-input`, negative hadrons from `prod_OO_isr/negative_hadrons.py`): bookkeeping against iColored's colour audit, hole and partner energy per class, full-jet R<sub>CP</sub> uncorrected / parton-level subtraction / negative hadrons, charged jets, leading hadrons → `results/validate_holes.json` |
| `oo_bulk.py`, `oo_bulk_comparison.ipynb` | R4c: the O+O bulk of the same events against STAR (p(N_ch), ⟨N_ch⟩ vs centrality, EPD nMIP of minimum-bias and triggered events) → `results/oo_bulk.json` |
| `oo_bias_docker.sh` | `oo_bias.py` on a multi-GPU Docker host: one container per GPU, each its shard (`--shard g/N`), the settings benchmarked on the 2 × RTX 3090 (8 jobs per GPU, OMP 3, MPS); `--dev` mounts this checkout's `example/` over the image's |
| `pp_bulk.py` | p+p 200 GeV bulk of the `pp_<set>` samples against UA5 (dN/dη NSD and inelastic, NSD P(n) in four η windows; HEPData ins233599, ins267179 → `refs/ua5_200/`) → `results/pp_bulk.json` |
| `centrality_norm.py` | how STAR's O+O centrality classes map onto the model's: STAR's normalisation from its own Fig. 5a/5b, the model centrality at STAR's N_ch and EPD edges, the flattest normalisation f, STAR's minimum-bias trigger emulated (from `oo_bulk.py`'s quantiles) → `results/centrality_norm.json` |
| `bulk_tune_screening.ipynb` | the bulk parameter exploration (step A of the p+p / O+O bulk tune and its follow-ups): the sets and why, p+p against UA5 and the RHIC data, O+O p(N_ch) shape, ⟨N_ch⟩ vs centrality with and without the normalisation, the summary and the grid, α<sub>rem</sub>, d+Au, shadowing and the y<sub>init</sub> = 2 node (section 8), the formation time (section 9), identified p<sub>T</sub> spectra and ⟨p<sub>T</sub>⟩ (section 10) |
| `pt_bulk.py` | identified π±, K±, p, p̄ spectra, ⟨p<sub>T</sub>⟩ and dN/dy of the bulk samples (p+p, d+Au, O+O) against STAR: TOF spectra in NSD p+p and d+Au classes (nucl-ex/0309012, ins628232), dE/dx in d+Au (0808.2041, ins793126, and its ⟨p<sub>T</sub>⟩ / dN/dy tables transcribed in `refs/pt_200/star_0808_2041_tables.json`), p+p charged ⟨p<sub>T</sub>⟩ vs multiplicity (nucl-ex/0606028, ins719969); each experiment's trigger and centrality estimator emulated → `results/pt_bulk.json` |
| `hard_recheck.py` | two hard-tune runs compared bin by bin (ratio, largest pull) and each run's χ² against the fit bins: the check that the bulk revision of `pp200_isr_vacuum.xml` leaves the hard tune unchanged (2026-11-24: χ² 35.4 vs 33.8 over 35 bins) |
| `g2_check.py`, `g2_mpi_check.ipynb` | gate G2 of the physics run, with the frozen hard + bulk tune (XML 788ca1e; the settings are read from the XML and the jobs' files): p+p underlying event in the production chain (`pp_bulk_w*` MPI on, `pp_bulk_mpioff_w*`, `pp_bulk_switch_w*`) against STAR (ue_check.py's definitions) and the minimum-bias level of `pp_tau0424`; O+O vacuum R<sub>AA</sub>/R<sub>CP</sub> with MPI on / off (`jets_vac[_mpioff]_w18_23`, `_w30_40`, classes from `phys_mb`), ⟨N<sub>EPD</sub>⟩ vs N<sub>coll</sub>; the page's decision rule applied → `results/g2_check.json` |
| `fwd_check.py` | the forward side of gate G2: per leading-jet bin, charged activity of the p+p jet events (`pp_bulk_w*`, `_mpioff`, `_switch`; jet + bulk hadrons) in the EPD (2.1–5.1) and MBD (3.51–4.61) acceptances and in the transverse region, each over the minimum bias of `pp_tau0424`; EPD split into the leading jet's hemisphere and the opposite one → `results/fwd_check.json`. Model result (2026-11-24): the settings that span a factor 2.2 at mid-rapidity differ by ±0.15 in the EPD and not at all in the MBD; the forward activity falls with jet pT in all of them (energy conservation) |
| `soft_mpi_check.py` | the soft-MPI check after gate G2 (`phys2_soft`: `pp_bulk_soft142_w*`, `_soft145_w*`, `_soft148_w*`, Pythia's soft MPI with MPI &alpha;<sub>s</sub> 0.130, pTmin 0.2, ecmRef 200, pT0Ref 1.42 / 1.45 / 1.48 appended to the tune XML's lines): transverse density, p<sub>T</sub> &gt; 0.5 density and &lang;p<sub>T</sub>&rang; against STAR, EPD / MBD over minimum bias, against MPI on and off; pT0Ref at STAR's level interpolated in ln(ratio) (bracketing pair and a parabola), &chi;<sup>2</sup> minimum → `results/soft_mpi_check.json`. Model result (2026-11-24): &chi;<sup>2</sup> over the complete bins 30.9 / 16.1 / 8.1 at pT0Ref 1.42 / 1.45 / 1.48 (MPI on 26.3; parabola minimum 6.5 at 1.50), STAR's level at 1.44 (6&ndash;15 GeV), 1.465 (11&ndash;20 GeV), 1.49 (11&ndash;45 GeV); trend still flat, &lang;p<sub>T</sub>&rang; 0.76&ndash;0.79 &times; STAR. `g2_mpi_check.ipynb` shows it as Part 4 (Part 3: `fwd_check.py`) |
| `rhictune_xml.py` | writes `config/pp200_isr_rhictune.xml` (p+p 200 GeV in the full chain: strings, hydro, iSS, Matter/LBT; the physics run's p+p reference `pp_ref_w*`) from the O+O tune XML with only the nuclei changed, and checks (`check`; also `oo_bias.py` before launching) that the two differ nowhere else. Not to be confused with `pp200_isr_vacuum.xml`, the hard tune's chain without strings and hydro. The earlier p+p samples (O+O XML with `--xml-set` nuclei) give the same job XMLs (driver dry run, 163 elements identical) |
| `oo_analysis_rhic.py`, `oo_analysis_rhic.ipynb` | the main analysis of the physics run (phase 3; replaces `oo_centrality_bias.ipynb` for it): the samples `oo_vac`, `oo_med`, `pp_ref`, `pp_med` stitched over the p̂<sub>T</sub> windows, classes from `phys_mb` (EPD, MBD, mid-rapidity, impact parameter as truth). Selection = R<sub>AA</sub> of `oo_vac` per class against N<sub>coll</sub> × `pp_ref`; quenching Q = `oo_med` / `oo_vac` in the same classes (independent errors: the same seeds are not the same events), Q<sub>b</sub> vs Q<sub>mult</sub>; total R<sub>AA</sub> / R<sub>CP</sub> of `oo_med`. Observables: charged hadrons, sPHENIX full jets R = 0.4, STAR charged jets R = 0.2 / 0.5, STAR's I<sub>CP</sub> (h–h, h+jet; arXiv:2604.13935's definitions), I<sub>AA</sub>, `pp_med` / `pp_ref` and the QS check. The medium's holes: jets subtract the holes and partner copies in the cone at parton level (full four-momentum for the full jets, 2/3 of p<sub>T</sub> for the charged jets; `*_raw` without); hadron-level observables (charged-hadron R<sub>AA</sub>, h–h I<sub>CP</sub>) need `prod_OO_isr/negative_hadrons.py` (follow-up). Stitching: a p̂<sub>T</sub> window enters a bin where every compared sample has ≥ 10 raw entries pooled over the classes (one choice for all classes and estimators); per bin the dropped share of each sample's yield and the incomplete low-p<sub>T</sub> bins (lowest window > 2 %) are reported; histograms window × class × bin with `hist`. Overlapping p̂<sub>T</sub> windows (3e: narrow 35–40 and 47–55 GeV samples inside 30–40 and 40–55) are combined piece by piece (every window cut at all window edges; per piece σ from the covering windows, weight σ<sub>piece</sub> / events in the piece; `--split-at` for the closure test). Full jets are also split by flavour (`sph_jet_q`, `sph_jet_g`: the nearest hard outgoing parton within ΔR < 0.4). `--phase3` restricts to the phase-3 jobs (no 3b extensions, no 3e samples; outputs with the suffix `_phase3`), `--phase3 --with-3e` adds the 3e samples (suffix `_phase3e`); without either, all finished jobs. The notebook uses all jobs and shows what 3e and 3b changed (sections 5b, 5c). Data overlays from `refs/star_oo_icp/` and `refs/sphenix_oo_jet/` → `results/oo_analysis_rhic.json`; `-j` parallel file workers, `--cached` Re-hadronized samples (`../../prod_OO_isr/rehadronize_colorless.py`): `--out ROOT_REHAD --est-root ROOT_ORIG --tag colorless` keeps every event in the class of its original estimators and bulk, and subtracts the hole hadrons (pstat −1) at hadron level (notebook §6b). |
| `refs/star_oo_icp/` | STAR O+O h–h and h+jet I<sub>CP</sub> (arXiv:2604.13935, HEPData ins3145466 v1: Figs. 3, 5, 6 as `figure*.yaml`; git-ignored like all HEPData YAML). hepdata.net blocks this box (Cloudflare challenge): download the record's submission zip in a browser (https://www.hepdata.net/record/ins3145466, "Download All", YAML) and unzip its `*.yaml` here; read by `oo_analysis_rhic.py` (`load_data`). The record has no EPD multiplicity distributions (Figs. 1, 4) |
| `pp_jet_spectra.py` | the model's p+p jet cross sections in the experiments' own definitions and bins, for `jet_spectrum_slope.ipynb`: `pp_ref`, all finished jobs (phase 3 + 3e; `--phase3` for phase 3 alone), full jets R = 0.5 at \|η\| < 0.5 and 0.5–0.9 (STAR's bins) and R = 0.4 at \|η\| < 0.7 (sPHENIX's bins), stitched with `stitch_pieces`, absolute → `results/pp_jet_spectra.json` (`_phase3.json`). Model / STAR (\|η\| < 0.5): 0.86–1.14 to 31 GeV, 0.70–0.71 at 37–52 GeV (the hard tune's high-p<sub>T</sub> shortfall); model / sPHENIX: 1.53 at 15–19 GeV → 1.00–1.04 at 34–63 GeV |
| `quench_diag.py`, `quench_diag.ipynb` | follow-up A1, A2, B4: per hard outgoing parton (`initiators/` pstat 1000, p<sub>T</sub> > 10 GeV, \|η\| < 0.5) the energy given to the medium (recoils − holes, in the hemisphere and the cone) and the jet loss 1 − ⟨p<sub>T,jet</sub>/p<sub>T,parton</sub>⟩<sub>med</sub>/⟨…⟩<sub>vac</sub> at parton level (without and with the response) and hadron level; quark vs gluon, production radius; the medium's time scales (MUSIC start, hole times, bulk freeze-out); and for the selection bias the forward signal of `oo_vac` jet events over minimum bias at the same N<sub>coll</sub> against p̂<sub>T</sub> and x, with the energy taken from each nucleus and the beam remnants' energy. `python quench_diag.py --out ROOT -j N both` → `results/quench_diag_eloss.npz`, `results/quench_diag_bias.npz` (reads the `oo_analysis_rhic.py` caches) |
| `jet_spectrum_slope.ipynb` | how steep the 200 GeV p+p jet spectrum is, and what that does to R<sub>AA</sub>/R<sub>CP</sub>: local power n(p<sub>T</sub>) = −d ln σ/d ln p<sub>T</sub> of sPHENIX (R = 0.4, preliminary), STAR (R = 0.5) and our `pp_ref` (R = 0.4), fits A p<sub>T</sub><sup>−n₀</sup>(1−x<sub>T</sub>)<sup>m</sup> integrated over the bins. Result: n rises from ≈ 6 at 15–20 GeV to ≈ 11 at 40 and ≈ 15 at 50 GeV in every spectrum, the model's included; a constant fractional loss ε then gives a suppression falling with p<sub>T</sub>. sPHENIX's R<sub>CP</sub> / bias reads as ε ≈ 4–7 % (flat or slowly falling), the model as ε ≈ 0.4–2 % (≈ 1 % at 36–60 GeV). Also the model's checks: jet flavour (each full jet tagged by the nearest hard outgoing parton, `initiators/` pstat 1000: gluon share 34 % at 10 GeV → 5 % at 55 GeV; ε per flavour, gluon/quark ε ≈ 1.3 ± 0.6, no significant Casimir ordering) and cone size (charged jets R = 0.2 vs 0.5). Reads the all-jobs outputs of `oo_analysis_rhic.py` (`results/oo_analysis_rhic.json`, `*.npz`) and `pp_jet_spectra.py`, `data/data.json`, `refs/sphenix_pp_jet/` |
| `auau_raa.py` | the Au+Au 200 GeV 0–10 % energy-loss calibration (`oo_bias.py` groups `auau_b1` = `auau_med_w*`, Type 2 α<sub>s</sub> 0.25; `auau_b2` = `auau_t3_w*`, Type 3 α<sub>s</sub> 0.3): R<sub>AA</sub> against `pp_ref` (the same \|y\| < 1 parton cut) for pions (PHENIX's and STAR's acceptances), charged hadrons, charged jets R = 0.2–0.4 (holes + partners subtracted with 2/3; without and with the 5 GeV leading-track cut; `_star`: the cut on Au+Au only, STAR's R<sub>AA</sub><sup>Pythia</sup> definition), full jets R = 0.4; every jet also unsubtracted (`_raw`); the bulk at mid-rapidity. `python auau_raa.py --family auau_med -j 6` → `results/auau_raa_<family>.json`. Holes: parton level for iColored files (holes + partner copies, charged × 2/3), hadron level (JETSCAPE's: pstat −1 hadrons subtracted) for Colorless files (`run_prod_jet_single.py`, `rehadronize_colorless.py`); `--auau-glob` / `--pp-glob` take other samples (run C: `/data/AuAu-RHIC_phys/js_chain_t3`, against PP19) |
| `pp_ref_check.py` | the p+p reference of `auau_raa.py` against the measured p+p spectra (STAR π±, h±; PHENIX π⁰) and against Pythia 8 Monash / Detroit (`refs/pythia_ref.py` with iColored's decays: `--line "111:mayDecay = on" --line "221:mayDecay = on" --line "ParticleDecays:limitTau0 = on" --line "ParticleDecays:tau0Max = 10.0"`, in `/data/OO-RHIC_pp_refs/pythia8_<tune>_icdecays`): hadron spectra, charged jets with and without a perpendicular-cone UE subtraction. `python pp_ref_check.py -j 6` → `results/pp_ref_check.json` |
| `chain_diag.py` | where between energy loss, partons and hadrons the Au+Au samples part: per sample and p̂<sub>T</sub> window the energy given to the medium (recoils − holes), holes, shower initiators, final partons and charged hadrons above thresholds, hadrons per parton (B1, B1 re-hadronized with Colorless, B2, run C, `pp_ref` both ways, PP19). `python chain_diag.py -j 6` → `results/chain_diag.json` (`auau_jet_raa.ipynb` Fig. 6) |
| `auau_jet_raa.ipynb` | Au+Au 0–10 %: model charged-jet R<sub>AA</sub> against STAR (2006.00582), pion and hadron R<sub>AA</sub> against PHENIX π⁰ and STAR π±, h±; the p+p reference against data and Pythia 8, and the jet R<sub>AA</sub> with Pythia 8 as the reference (as STAR); per-parton loss Au+Au vs O+O; the reading. Runs `auau_raa.py` (families in `SHOW`) and `pp_ref_check.py` if needed. Generated without outputs; executed copy: `results/auau_jet_raa_executed.ipynb` |
| `auau_bulk_phobos.ipynb` | the Au+Au (and d+Au) bulk of the RHIC tune against PHOBOS dN/dη (with `hist`), the MAP block for comparison, and the runs B1/B2. Executed copy in `results/` |
| `refs/star_auau200_chjet/` | STAR charged-jet R<sub>AA</sub><sup>Pythia</sup>, Au+Au 0–10 % and 60–80 %, R = 0.2–0.4 (arXiv:2006.00582 Fig. 17), read from the vector figure: `extract_star_chjet_raa.py PAPER.pdf` → `star_chjet_raa.json` |
| `refs/phenix_auau200_pi0/` | PHENIX π⁰ R<sub>AA</sub> Au+Au 0–10 % (PRC 87, 034911, Table VII): `phenix_pi0_raa_0_10.json` |
| `refs/star_auau200_hadrons/` | STAR π± R<sub>AA</sub> 0–12 % (1110.0579, HEPData ins930463) and (h⁺+h⁻)/2 R<sub>AA</sub> 0–5 %, 10–20 % (nucl-ex/0305015, HEPData ins619063), and their measured p+p spectra: `fetch_star_hadron_raa.py` → `star_hadron_raa.json` (YAML git-ignored) |
| `refs/jetscape_2204_fig14/` | JETSCAPE's MATTER+LBT Type 3 (α<sub>s</sub> 0.3; TRENTo + VISHNU, PP19) Au+Au 0–10 % charged-jet (R = 0.2–0.4) and π R<sub>AA</sub>, arXiv:2204.01163 Fig. 14, read from the vector figure: `extract_js_fig14.py PAPER.pdf` → `js_fig14.json` |
| `refs/hadronize_pp19_icdecays.xml` | Colorless with iColored's decays (π⁰, η decay; K⁰<sub>S</sub>, Λ, Σ, Ξ, Ω do not), for the PP19 p+p reference (`../prod_AuAu_0_10_jet/run_prod_vacuum.py --user-xml refs/pp19_200.xml`) and for JETSCAPE's chain on our medium (`../prod_AuAu_0_10_jet/run_prod_jet_single.py`, run C) |
| `refs/sphenix_pp_jet/` | sPHENIX's preliminary p+p 200 GeV inclusive jet cross section (sPH-CONF-JET-2025-03; anti-k<sub>T</sub> R = 0.4, \|η\| < 0.7, 15–72 GeV, unfolded): `extract_sphenix_pp_jet.py NOTE.pdf` reads Fig. 5 from the vector paths (log axis from the decade ticks) → `sphenix_pp_jet_prelim.json`. The drawn statistical bars are the same in every bin (a drawing size: upper bounds); the systematic boxes are largely correlated. Preliminary, for comparison only |
| `refs/sphenix_oo_jet/` | sPHENIX's preliminary O+O jet R<sub>CP</sub> (vs 60–80 % and vs 40–60 %) and R<sub>AA</sub> 0–100 % (sPH-CONF-JET-2026-02; R = 0.4, \|η\| < 0.7, MBD classes, 19–60 GeV): the note has no value tables, so `extract_sphenix_oo_jet.py NOTE.pdf` reads the points, statistical bars and asymmetric systematic boxes from the PDF's vector paths (`pdftocairo -svg`; axes mapped with their major ticks) → `sphenix_oo_jet_prelim.json`. The lowest bins' statistical bars (±0.017–0.018 in every panel) are probably the drawing's minimum: upper bounds there. Also the note's model curves (key `theory`: per vertex the line's value and the band's edges): JEWEL 2.4.0 recoils off / on (R<sub>AA</sub> 0–100 %, R<sub>CP</sub>; Figs. 6, 9, 10), Angantyr's centrality-bias factors on R<sub>CP</sub> (Fig. 8), the no-quenching nPDF baseline with nNNPDF3.0 / EPPS21 / TUJU21 (Fig. 6); `oo_analysis_rhic.ipynb` Figure 3c sets them against `oo_vac` (selection) and `oo_med` / `oo_vac` (quenching). Preliminary, for comparison only |
| `refs/rhic_pp200/` | RHIC p+p 200 GeV multiplicity references besides UA5 (none on HEPData): PHOBOS inelastic dN/dη for \|η\| ≤ 5.3 read from the vector figure of arXiv:1011.1940 (Fig. 28a, 90 % C.L. band), STAR NSD and PHENIX minimum-bias mid-rapidity values (`extract_rhic_pp200.py --phobos <source dir>` → `rhic_pp200.json`); used by `pp_bulk.py` |
| `dau_bulk.py` | d+Au 200 GeV minimum-bias bulk (`oo_bias.py` samples `dau_<set>`: d on Au, b < 12 fm) against PHENIX (dN/dη, \|η\| < 2.6, six classes from the Au-going BBC; HEPData ins1684475), STAR (\|η\| < 1 in Au-going FTPC classes, ins656934; FTPC points at 3.0 ≤ \|η\| ≤ 3.5 in \|η\| < 0.5 classes) and BRAHMS (minimum bias to η = 4.2, ins643085): the check that a bulk set does not break the MAP's calibration → `results/dau_bulk.json`; `--class-frac F`: the model's classes as fractions of its highest F of events (→ `results/dau_bulk_f<F>.json`) |
| `refs/dau_200/` | d+Au references: the HEPData tables (fetched, git-ignored) and `extract_star_ftpc_dau.py` → `star_dau_ftpc.json`, STAR's FTPC d+Au dN/dη read from a ROOT figure of the d+Au FTPC analysis (2004; the figure is not public and stays local, git-ignored; its 40–100 % and minimum-bias points are rescaled to the published \|η\| < 1 normalisation) |
| `refs/star_oo/` | STAR O+O references read from the vector figures of arXiv:2510.19645 and 2604.13935 (`extract_star_oo.py` → `star_oo_bulk.json`; no HEPData) |
| `conditional.py` | the posterior with a parameter held fixed (emulator only), e.g. Q0 = 2 GeV for Matter + LBT → `results/conditional_Q0_2.json` |
| `dndeta.py` | charged $dN/d\eta$ over the full η range at fixed pT̂ for all models and MPI settings (own small samples under `PROD/diag/eta/`) |
| `refs/run_refs.py` | the reference models' events (PP19 via `prod_AuAu_0_10_jet/run_prod_vacuum.py`, Pythia standalone) |
| `ue_trend_cv.ipynb` | why the vacuum p+p UE is flat while STAR's falls (tuning page, "Why the UE is flat" and "UE trend vs C_v", 2026-11-24): standalone Pythia with ISR off keeps the fall; the decomposition (MPI off, FSR off, ISR off) of the chain and of Pythia Monash; ⟨N<sub>MPI</sub>⟩ vs p̂<sub>T</sub> (parton level, run in the notebook); the 32 design points' trend UE(11&ndash;15)/UE(35&ndash;45) against C<sub>v</sub>, &alpha;<sub>s</sub>, Q<sub>0</sub> and Pythia's &sigma;<sub>int</sub> (read from the job logs: with pTmin 5 it steps between 31 and 53 mb), fitted along the valley, and the STAR jets at 37&ndash;52 GeV. Reads the `ue_check.py` sums (`results/ue_<run>.npz`; recomputed when missing), the runs' observables and `results/design_chi2.json`; old MPI, vacuum chain. Model result: remnants and MPI fall as in Pythia, iMATTER ISR and MATTER FSR rise with jet p<sub>T</sub>; trend 1.17 at C<sub>v</sub> 0.16 to 0.82 at 0.50 (STAR 1.34), jets 0.70&ndash;0.76 along the valley, set by &alpha;<sub>s</sub> |
| `scheme_recheck.ipynb` | the p+p hard tune in the A+A jet scheme (tuning page, "Phase 4", 2026-11-25): MATTER down to QS = 0.9 outside the QGP (`in_vac 0`, as in the O+O and `pp_med` runs) against the tune's vacuum mode (down to Q<sub>0</sub>), at the tune XML point. Vacuum chain (`--vacuum` vs `--vacuum --aa-vacuum`, 294k events per arm, π⁰ undecayed) and full chain (`--vacuum-jet` vs `--matter-only` with T<sub>0</sub> = hydro_Tc = 100, 8k per arm): A+A/tune ratio per bin and, for the π⁰, in wide p<sub>T</sub> ranges (job-to-job spread); model/data and χ² (tune.py's covariance) for PHENIX and STAR π⁰ and the STAR jets. Reads the runs' observables and the vacuum-chain event files. Model result: π⁰ 0.97&ndash;0.98 at 2&ndash;4 GeV, consistent with 1 above 6 GeV; jets unchanged; equal χ² in both schemes |
| `refs/pythia_ref.py` | standalone Pythia 8 (default, Detroit, no CR; `--windows`, `--eta-max 0`, `--line` for diagnostics) in the tune's hadron-file layout |
| `refs/pp19_200.xml`, `refs/hadronize_pp19.xml` | PP19 at 200 GeV: PythiaGun + vacuum Matter (Q0 1, C_v 0.25) + Colorless, π⁰ and η undecayed |
| `tune.py` | emulator, posterior (emcee), design-point χ² |
| `tune_comparison.ipynb` | posterior, χ² cross-check, data vs model, validation |
| `hstat.py` | statistical errors for the analyses and notebooks (hist `Weight`/`Mean` storages): histograms, means per bin or class (weighted with importance weights), per-event particle histograms (event-to-event spread), class fractions, experiment-style R_AA^vac and R_CP of N_coll-weighted jet events; plotting helpers (bands, error bars) |
| `results/` | `summary.json` (MAP, intervals, χ²), `design_chi2.json`, `chains.npz`, `posterior_predictive.npz`, `emulator.pkl`, `truncation.json` |

## Definitions and caveats

- **Holes and partners (the jet in the medium).** Matter (`recoil_on` 1) and LBT leave holes
  (pstat −1) that iColored drops, and for part of them it inserts colour partners with the
  hole's three-momentum. **The physics runs store both with `--store-partons
  --store-hadronization-input`** (rule and details: `prod_OO_isr/README.md`, "Holes and
  partners: what a physics run must store"). Then: full jets subtract holes + partners at
  parton level in the cone; leading hadrons above 3 GeV need nothing; charged jets, hadrons
  below 3 GeV, substructure, soft and recoil observables subtract the negative hadrons of
  `negative_hadrons.py`. The 50k `jets_quench` sample has neither: its full-jet and
  charged-jet R<sub>CP</sub> carry a ≈ 2 % systematic (too high in central events; from
  `validate_holes.py`), its hadron R<sub>CP</sub> above 3 GeV none.
- Statistical errors: every model point carries one (`hstat.py`). Each event has one sample (no
  hydro or iSS oversampling), so events are independent; particle distributions take the
  event-to-event spread (√N of particles is too small: particles of one event are correlated),
  weighted quantities use Σw². The bands of the posterior are not statistical errors.
- Model jets: anti-k_T R = 0.5 (FastJet) on all visible final particles with p_T > 0.2 GeV
  (stored |η| < 1.5), ≥ 1 charged constituent, p_T corrected with STAR's off-axis cones (two
  cones of radius R at the jet's η and φ ± π/2, A_jet = πR²). In vacuum mode there is no bulk;
  the underlying event is MPI (effective cut 1.7 GeV), ISR/FSR and beam remnants.
- π⁰ and η are undecayed in the model (`111:mayDecay = off`, `221:mayDecay = off` in iColored);
  invariant cross sections use rapidity windows.
- Covariance: stat ⊕ sys per bin (systematics uncorrelated between bins: the tables give no
  correlations) plus the normalisation (PHENIX 9.7 %, STAR 5.6 %) fully correlated within a data
  set, plus the emulator's (or the simulation's) statistical variance; residuals in ln σ.
