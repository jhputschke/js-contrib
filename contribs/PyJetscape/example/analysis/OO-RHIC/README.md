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
# hard-tune recheck with the new vacuum XML: the reference point with the new and the old block, same seeds
git show a3bb4c2:contribs/PyJetscape/example/config/pp200_isr_vacuum.xml > design/pp200_isr_vacuum_mapbulk.xml
python make_design.py --add ref_tune 0.15 0.25 2.0 --jobs 10
python make_design.py --add ref_mapbulk 0.15 0.25 2.0 --jobs 10 --seeds-of ref_tune --template design/pp200_isr_vacuum_mapbulk.xml
python run_runs.py ref_tune ref_mapbulk -j 12 && python observables.py ref_tune ref_mapbulk
python hard_recheck.py ref_tune ref_mapbulk && python ue_check.py ref_tune ref_mapbulk
jupyter nbconvert --to notebook --execute bulk_tune_screening.ipynb --output-dir results --output bulk_tune_screening_executed.ipynb
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
| `oo_bias_jets.py` | the same against the jet p_T and the charged-hadron p_T: anti-k_T R = 0.4 jets (\|η\| < 0.7; charged jets for `--charged` samples, full jets for `jets_quench`) and charged hadrons (\|η\| < 1) on the hard event's hadrons, per p̂_T window and stitched over the windows (a window enters a bin only with ≥ `--min-entries` (10) raw entries in every class); class sums for quenched/vacuum double ratios; LBT hole energies of the colour audit → `results/oo_bias_jets.json`, `results/oo_bias_jets_jets_quench.json` |
| `validate_holes.py` | validation of the hole and partner treatment on the `jets_holes` sample (made with `--store-partons --store-hadronization-input`, negative hadrons from `prod_OO_isr/negative_hadrons.py`): bookkeeping against iColored's colour audit, hole and partner energy per class, full-jet R<sub>CP</sub> uncorrected / parton-level subtraction / negative hadrons, charged jets, leading hadrons → `results/validate_holes.json` |
| `oo_bulk.py`, `oo_bulk_comparison.ipynb` | R4c: the O+O bulk of the same events against STAR (p(N_ch), ⟨N_ch⟩ vs centrality, EPD nMIP of minimum-bias and triggered events) → `results/oo_bulk.json` |
| `oo_bias_docker.sh` | `oo_bias.py` on a multi-GPU Docker host: one container per GPU, each its shard (`--shard g/N`), the settings benchmarked on the 2 × RTX 3090 (8 jobs per GPU, OMP 3, MPS); `--dev` mounts this checkout's `example/` over the image's |
| `pp_bulk.py` | p+p 200 GeV bulk of the `pp_<set>` samples against UA5 (dN/dη NSD and inelastic, NSD P(n) in four η windows; HEPData ins233599, ins267179 → `refs/ua5_200/`) → `results/pp_bulk.json` |
| `centrality_norm.py` | how STAR's O+O centrality classes map onto the model's: STAR's normalisation from its own Fig. 5a/5b, the model centrality at STAR's N_ch and EPD edges, the flattest normalisation f, STAR's minimum-bias trigger emulated (from `oo_bulk.py`'s quantiles) → `results/centrality_norm.json` |
| `bulk_tune_screening.ipynb` | the bulk parameter exploration (step A of the p+p / O+O bulk tune and its follow-ups): the sets and why, p+p against UA5 and the RHIC data, O+O p(N_ch) shape, ⟨N_ch⟩ vs centrality with and without the normalisation, the summary and the grid, α<sub>rem</sub>, d+Au, shadowing and the y<sub>init</sub> = 2 node (section 8), the formation time (section 9), identified p<sub>T</sub> spectra and ⟨p<sub>T</sub>⟩ (section 10) |
| `pt_bulk.py` | identified π±, K±, p, p̄ spectra, ⟨p<sub>T</sub>⟩ and dN/dy of the bulk samples (p+p, d+Au, O+O) against STAR: TOF spectra in NSD p+p and d+Au classes (nucl-ex/0309012, ins628232), dE/dx in d+Au (0808.2041, ins793126, and its ⟨p<sub>T</sub>⟩ / dN/dy tables transcribed in `refs/pt_200/star_0808_2041_tables.json`), p+p charged ⟨p<sub>T</sub>⟩ vs multiplicity (nucl-ex/0606028, ins719969); each experiment's trigger and centrality estimator emulated → `results/pt_bulk.json` |
| `hard_recheck.py` | two hard-tune runs compared bin by bin (ratio, largest pull) and each run's χ² against the fit bins: the check that the bulk revision of `pp200_isr_vacuum.xml` leaves the hard tune unchanged (2026-11-24: χ² 35.4 vs 33.8 over 35 bins) |
| `refs/rhic_pp200/` | RHIC p+p 200 GeV multiplicity references besides UA5 (none on HEPData): PHOBOS inelastic dN/dη for \|η\| ≤ 5.3 read from the vector figure of arXiv:1011.1940 (Fig. 28a, 90 % C.L. band), STAR NSD and PHENIX minimum-bias mid-rapidity values (`extract_rhic_pp200.py --phobos <source dir>` → `rhic_pp200.json`); used by `pp_bulk.py` |
| `dau_bulk.py` | d+Au 200 GeV minimum-bias bulk (`oo_bias.py` samples `dau_<set>`: d on Au, b < 12 fm) against PHENIX (dN/dη, \|η\| < 2.6, six classes from the Au-going BBC; HEPData ins1684475), STAR (\|η\| < 1 in Au-going FTPC classes, ins656934; FTPC points at 3.0 ≤ \|η\| ≤ 3.5 in \|η\| < 0.5 classes) and BRAHMS (minimum bias to η = 4.2, ins643085): the check that a bulk set does not break the MAP's calibration → `results/dau_bulk.json`; `--class-frac F`: the model's classes as fractions of its highest F of events (→ `results/dau_bulk_f<F>.json`) |
| `refs/dau_200/` | d+Au references: the HEPData tables (fetched, git-ignored) and `extract_star_ftpc_dau.py` → `star_dau_ftpc.json`, STAR's FTPC d+Au dN/dη read from a ROOT figure of the d+Au FTPC analysis (2004; the figure is not public and stays local, git-ignored; its 40–100 % and minimum-bias points are rescaled to the published \|η\| < 1 normalisation) |
| `refs/star_oo/` | STAR O+O references read from the vector figures of arXiv:2510.19645 and 2604.13935 (`extract_star_oo.py` → `star_oo_bulk.json`; no HEPData) |
| `conditional.py` | the posterior with a parameter held fixed (emulator only), e.g. Q0 = 2 GeV for Matter + LBT → `results/conditional_Q0_2.json` |
| `dndeta.py` | charged $dN/d\eta$ over the full η range at fixed pT̂ for all models and MPI settings (own small samples under `PROD/diag/eta/`) |
| `refs/run_refs.py` | the reference models' events (PP19 via `prod_AuAu_0_10_jet/run_prod_vacuum.py`, Pythia standalone) |
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
