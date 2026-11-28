# hard_tune: the RHIC p+p 200 GeV hard tune

The hard sector of X-SCAPE's ISR chain for p+p at √s = 200 GeV, tuned to high-p<sub>T</sub> hadron
and jet data: the p+p reference whose hard settings the O+O 200 GeV jet production
(`prod_OO_isr`) takes. Steps 1 and 2 of the "RHIC p+p and O+O Tuning" page, done 2026-11-22,
with the later rechecks of the tune (bulk XML revision, the underlying-event trend, the A+A jet
scheme).

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

## Run it

In the `js_fno` environment (X-SCAPE `build_gpu` with PyJetscape), from this directory. Events
and observables go to `~/prod/OO-RHIC_pp_hard` (`$OO_RHIC_PROD` overrides); the HEPData tables
(`../data/`), the design (`../design/`) and the fit results (`../results/`) are shared with the
rest of `analysis/OO-RHIC` and git-ignored. `fetch_data.py` makes `../data/` again; the design is
fixed by `config.DESIGN_SEED`, the extra runs by the `--add` commands below.

The fit needs `emcee` and `scikit-learn`, which no environment installs by default (not in
`utils/analysis_env/requirements.txt`, the `pyjetscape[analysis]` extra, the production container
or the Mac wheel): `pip install emcee scikit-learn` (tested: emcee 3.1.6, scikit-learn 1.9.0; in
the read-only container `pip install --user` with `PYTHONUSERBASE` below `/work`). Running events
needs X-SCAPE (`run_runs.py`); `observables.py`, `ue_check.py`, `tune.py` and the notebooks only
read files.

```bash
conda activate js_fno
cd contribs/PyJetscape/example/analysis/OO-RHIC/hard_tune
python fetch_data.py                         # ../data/hepdata/*.yaml, ../data/data.json
python make_design.py                        # ../design/design.json, ../design/xml/p000..p031.xml

# truncation check (30k events, ~1 min on 10 cores)
python make_design.py --add trunc 0.155 0.33 1.5 --jobs 10 --windows 4-5,5-7,7-9,9-11,11-14,14-18
python run_runs.py trunc -j 10 && python observables.py trunc && python truncation_check.py

# the design: 32 points x 10 jobs x 30k events, ~80 min on 20 cores (GB10, one core per job)
python run_runs.py --design -j 20
python run_runs.py --low -j 20               # + 30k events per window 5-23 GeV (pi0 statistics)
python observables.py --design -j 8           # PROD/obs/pNNN.npz
python tune.py                               # ../results/summary.json, chains, emulator, ...

# validation runs (values of the run of 2026-11-22): the posterior median (the MAP sits on the
# box edge Cv = 0.5 of a flat alpha_s-Cv valley) and the reference point, 30 jobs x 30k events
python make_design.py --add best 0.1530 0.319 1.52 --jobs 30     # ../results/summary.json "median"
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
python ../refs/run_refs.py -j 20                # 3 x 20 jobs x 30k events, ~10 min
python observables.py pp19 pythia8_default pythia8_detroit
python ue_check.py trunc --check-truncation
python ../refs/run_refs.py pythia8_nocr --jobs 10 -j 10      # diagnostic: Pythia default without colour reconnection
python ue_check.py $RUNS best_mpidef pp19 pythia8_default pythia8_detroit pythia8_nocr -j 16   # ../results/ue_check.json
python ue_check.py $RUNS best_mpidef pp19 pythia8_default pythia8_detroit pythia8_nocr --cached   # again from ../results/ue_<run>.npz
# where in eta the particles are made: full-eta samples at pTHat 9-11 and 23-30 GeV (3000 events each;
# all models, the four MPI settings, Pythia with the tune's coupling, CR on/off; ~10 min, full MPI the longest)
python dndeta.py run -j 20 && python dndeta.py analyse                       # ../results/dndeta.json
P=$HOME/prod/OO-RHIC_pp_hard/diag/eta/9-11
python diag_strings.py $P/best_nompi $P/best $P/best_mpi4 $P/best_mpidef   # ../results/diag_strings.json
python conditional.py --fix Q0=2.0          # posterior at Q0 = 2 (Matter + LBT in O+O), emulator only

# hard-tune recheck with the new vacuum XML: the reference point with the new and the old block, same seeds
git show a3bb4c2:contribs/PyJetscape/example/config/pp200_isr_vacuum.xml > ../design/pp200_isr_vacuum_mapbulk.xml
python make_design.py --add ref_tune 0.15 0.25 2.0 --jobs 10
python make_design.py --add ref_mapbulk 0.15 0.25 2.0 --jobs 10 --seeds-of ref_tune --template ../design/pp200_isr_vacuum_mapbulk.xml
python run_runs.py ref_tune ref_mapbulk -j 12 && python observables.py ref_tune ref_mapbulk
python hard_recheck.py ref_tune ref_mapbulk && python ue_check.py ref_tune ref_mapbulk

# why the vacuum UE is flat (ue_trend_cv.ipynb): standalone Pythia with ISR off, and the decomposition
# MPI off -> FSR off (matter_on 0) -> bare (also Q0 50: no iMATTER ISR), on best_nompi's XML and seeds
for ts in default:981000 detroit:982000; do t=${ts%:*}; for j in $(seq 0 9); do s=$((${ts#*:} + j)); \
  python ../refs/pythia_ref.py --tune $t --events 30000 --seed $s --line "PartonLevel:ISR = off" \
  --out $OO_RHIC_PROD/events/pythia8_${t}_isroff/pythia8_${t}_isroff_seed${s}_hadrons_jet_frag.h5; done; done
#   (the same with MPI off, + FSR off, + ISR off: runs pythia8_default_nompi, _nompi_fsroff, _nompi_bare)
python make_design.py --add best_nompi_fsroff 0.153 0.319 1.52 --jobs 10 --seeds-of best_nompi --pythia-line "PartonLevel:MPI = off"
python make_design.py --add best_nompi_bare 0.153 0.319 1.52 --jobs 10 --seeds-of best_nompi --pythia-line "PartonLevel:MPI = off"
for n in best_nompi_fsroff best_nompi_bare; do cp ../design/xml/best_nompi.xml ../design/xml/$n.xml; \
  sed -i 's|<matter_on> 1 </matter_on>|<matter_on> 0 </matter_on>|' ../design/xml/$n.xml; done
sed -i 's|<Q0> 1.52 </Q0>|<Q0> 50 </Q0>|' ../design/xml/best_nompi_bare.xml
python run_runs.py best_nompi_fsroff best_nompi_bare -j 20
python ue_check.py best_nompi_fsroff best_nompi_bare pythia8_default_isroff pythia8_detroit_isroff -j 16
jupyter nbconvert --to notebook --execute ue_trend_cv.ipynb --output-dir ../results --output ue_trend_cv_executed.ipynb
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
jupyter nbconvert --to notebook --execute scheme_recheck.ipynb --output-dir ../results --output scheme_recheck_executed.ipynb
jupyter nbconvert --to notebook --execute --inplace tune_comparison.ipynb
```

`run_runs.py` is resumable (finished jobs are skipped) and gives every job its own seed.

## Results

Not public yet: the numbers and conclusions are kept in a local, git-ignored `phys_readme.md` next to this file.

The settings taken from the tune are in `config/OO_MCGlauber_MUSIC_isr_rhictune.xml` and its p+p twin
`config/pp200_isr_rhictune.xml`.

## Files

| file | |
|---|---|
| `config.py` | every setting of the tune: windows, parameters, events, data, fit ranges, paths |
| `fetch_data.py` | HEPData download (curl) and parsing → `../data/data.json` |
| `make_design.py` | the design (maximin LHS) and the per-point user XMLs; `--add` for extra runs |
| `run_runs.py` | runs the jobs (`run_prod_oo_isr.py --vacuum`), resumable |
| `observables.py` | model observables in the data binning, per pT̂ window (π⁰, η/π⁰, STAR jets with off-axis-cone UE subtraction) |
| `truncation_check.py` | what the 5 GeV lowest window misses |
| `ue_check.py` | STAR's transverse charged density vs leading jet (R = 0.6) from the runs' hadrons |
| `diag_strings.py` | the colour-string topology iColored hands to Pythia (stored input) against standalone Pythia's parton level, CR on/off, tune vs default coupling; writes `../results/diag_strings.json` for the notebook |
| `hard_recheck.py` | two hard-tune runs compared bin by bin (ratio, largest pull) and each run's χ² against the fit bins: the check that the bulk revision of `pp200_isr_vacuum.xml` leaves the hard tune unchanged |
| `conditional.py` | the posterior with a parameter held fixed (emulator only), e.g. Q0 = 2 GeV for Matter + LBT → `../results/conditional_Q0_2.json` |
| `dndeta.py` | charged $dN/d\eta$ over the full η range at fixed pT̂ for all models and MPI settings (own small samples under `PROD/diag/eta/`) |
| `ue_trend_cv.ipynb` | why the vacuum p+p UE is flat while STAR's falls (tuning page, "Why the UE is flat" and "UE trend vs C_v", 2026-11-24): standalone Pythia with ISR off keeps the fall; the decomposition (MPI off, FSR off, ISR off) of the chain and of Pythia Monash; ⟨N<sub>MPI</sub>⟩ vs p̂<sub>T</sub> (parton level, run in the notebook); the 32 design points' trend UE(11&ndash;15)/UE(35&ndash;45) against C<sub>v</sub>, &alpha;<sub>s</sub>, Q<sub>0</sub> and Pythia's &sigma;<sub>int</sub> (read from the job logs: with pTmin 5 it steps between 31 and 53 mb), fitted along the valley, and the STAR jets at 37&ndash;52 GeV. Reads the `ue_check.py` sums (`../results/ue_<run>.npz`; recomputed when missing), the runs' observables and `../results/design_chi2.json`; old MPI, vacuum chain. |
| `scheme_recheck.ipynb` | the p+p hard tune in the A+A jet scheme (tuning page, "Phase 4", 2026-11-25): MATTER down to QS = 0.9 outside the QGP (`in_vac 0`, as in the O+O and `pp_med` runs) against the tune's vacuum mode (down to Q<sub>0</sub>), at the tune XML point. Vacuum chain (`--vacuum` vs `--vacuum --aa-vacuum`, 294k events per arm, π⁰ undecayed) and full chain (`--vacuum-jet` vs `--matter-only` with T<sub>0</sub> = hydro_Tc = 100, 8k per arm): A+A/tune ratio per bin and, for the π⁰, in wide p<sub>T</sub> ranges (job-to-job spread); model/data and χ² (tune.py's covariance) for PHENIX and STAR π⁰ and the STAR jets. Reads the runs' observables and the vacuum-chain event files. |
| `tune.py` | emulator, posterior (emcee), design-point χ² |
| `tune_comparison.ipynb` | posterior, χ² cross-check, data vs model, validation |
| `../refs/run_refs.py` | the reference models' events (PP19 via `prod_AuAu_0_10_jet/run_prod_vacuum.py`, Pythia standalone) |
| `../refs/pythia_ref.py` | standalone Pythia 8 (default, Detroit, no CR; `--windows`, `--eta-max 0`, `--line` for diagnostics) in the tune's hadron-file layout |
| `../refs/pp19_200.xml`, `../refs/hadronize_pp19.xml` | PP19 at 200 GeV: PythiaGun + vacuum Matter (Q0 1, C_v 0.25) + Colorless, π⁰ and η undecayed |

Outputs in `../results/`: `summary.json` (MAP, intervals, χ²), `design_chi2.json`, `chains.npz`,
`posterior_predictive.npz`, `emulator.pkl`, `truncation.json`, `ue_check.json`, `ue_<run>.npz`,
`dndeta.json`, `diag_strings.json`, `conditional_Q0_2.json`.

## Definitions and caveats

- Model jets: anti-k_T R = 0.5 (FastJet) on all visible final particles with p_T > 0.2 GeV
  (stored |η| < 1.5), ≥ 1 charged constituent, p_T corrected with STAR's off-axis cones (two
  cones of radius R at the jet's η and φ ± π/2, A_jet = πR²). In vacuum mode there is no bulk;
  the underlying event is MPI (effective cut 1.7 GeV), ISR/FSR and beam remnants.
- π⁰ and η are undecayed in the model (`111:mayDecay = off`, `221:mayDecay = off` in iColored);
  invariant cross sections use rapidity windows.
- Covariance: stat ⊕ sys per bin (systematics uncorrelated between bins: the tables give no
  correlations) plus the normalisation (PHENIX 9.7 %, STAR 5.6 %) fully correlated within a data
  set, plus the emulator's (or the simulation's) statistical variance; residuals in ln σ.
- The bands of the posterior are not statistical errors.
