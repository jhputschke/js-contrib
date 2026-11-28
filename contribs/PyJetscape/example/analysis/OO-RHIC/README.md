# OO-RHIC: p+p, O+O and Au+Au at √s<sub>NN</sub> = 200 GeV

The X-SCAPE O+O physics run at RHIC and everything it rests on: the p+p hard tune, the p+p / O+O
bulk tune, the closure and centrality-bias tests, gate G2, the Au+Au energy-loss calibration and
the hadronization study. It follows the "RHIC p+p and O+O Tuning" page. The final O+O analysis
stays in this directory; each test has its own directory with a README (intent, how to run it,
results).

**Model results, not vetted physics results.** State the campaign, cuts and number of events
with any number taken from here.

## Layout

| directory | the question | when |
|---|---|---|
| [`hard_tune/`](hard_tune/README.md) | the p+p 200 GeV hard sector (α<sub>s</sub>, C<sub>v</sub>, Q<sub>0</sub>) against PHENIX π⁰ and STAR jets; the underlying event; rechecks | 2026-11-22 … 25 |
| [`bulk_tune/`](bulk_tune/README.md) | the 3dMCGlauber + MUSIC bulk against p+p, O+O, d+Au (and Au+Au) multiplicities; identified p<sub>T</sub> | 2026-11-23 … 25 |
| [`closure_bias/`](closure_bias/README.md) | R3, R4, R4b: the O+O vacuum closure, the centrality bias, the holes and partners | 2026-11-22 … 23 |
| [`g2_mpi_check/`](g2_mpi_check/README.md) | gate G2: MPI setting, p+p underlying event, forward activity, soft-MPI scan | 2026-11-24 |
| *this directory* | **the O+O physics run (phase 3): selection × quenching, R<sub>AA</sub>, R<sub>CP</sub>, I<sub>CP</sub> against STAR and sPHENIX** | 2026-11-24 … 25 |
| [`oo_followup/`](oo_followup/README.md) | reading the O+O results: energy loss, selection bias, the jet-spectrum slope | 2026-11-25 |
| [`auau_calibration/`](auau_calibration/README.md) | central Au+Au against STAR / PHENIX: is the energy loss right? | 2026-11-26 |
| [`hadronization/`](hadronization/README.md) | Colorless's close-pair shift and iColored: how the hadronization of medium-modified showers enters R<sub>AA</sub> | 2026-11-26 … 27 |

Shared by all: `results/` (every script's output), `data/` (HEPData tables of the hard tune),
`design/` (the hard tune's design and XMLs), all generated and git-ignored, and `refs/` (the
reference data, below). `_paths.py` puts this directory and the test directories on `sys.path`, so
every module imports the others by name wherever it lives:

```python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # in a test directory
import _paths                                                      # noqa: E402,F401
ROOT = _paths.ROOT                                                 # analysis/OO-RHIC: results/, refs/, data/
```

Notebooks in a test directory start with `sys.path.insert(0, "..")` + `import _paths` and read
`../results/`, `../refs/`. Write executed copies with `--output-dir ../results`.

## The O+O physics run

Samples (`oo_bias.py`, phase 3; tune XMLs of js-contrib `oo_rhic_run` 073085b and later):

| sample | | role |
|---|---|---|
| `oo_vac` | O+O, the jet in vacuum (`--vacuum-jet`; the medium is made) | selection only |
| `oo_med` | O+O, the jet in the medium (Matter + LBT), the same seeds | selection × quenching |
| `pp_ref` | p+p (`config/pp200_isr_rhictune.xml`), the jet in vacuum | the R<sub>AA</sub> reference |
| `pp_med` | p+p, the jet in p+p's own medium (3c) | consistent reference, Q<sub>S</sub> check |
| `phys_mb` | O+O minimum bias (bulk hadrons only) | class edges, ⟨N<sub>coll</sub>⟩ per class |

Each sample is split into one-window sub-samples `<name>_w<lo>_<hi>` (own directories and seeds),
stitched by the windows' cross sections. The same seeds do not give the same events (LBT shifts the
random stream): every ratio of two samples is statistical, with independent errors.

```bash
cd contribs/PyJetscape/example/analysis/OO-RHIC
# production on a multi-GPU Docker host, one container per GPU (oo_bias.py --help, oo_bias_docker.sh)
./oo_bias_docker.sh --dev /data/OO-RHIC_phys phys1      # phys_mb first (its analysis runs during phys2)
./oo_bias_docker.sh --dev /data/OO-RHIC_phys phys2      # gate G2 samples (g2_mpi_check/), after docker rm oo_bias_gpu0 oo_bias_gpu1
./oo_bias_docker.sh --dev /data/OO-RHIC_phys phys3      # oo_vac, oo_med, pp_ref
OO_PHYS_CENTRAL_BMAX=<b_max of 0-20 %> ./oo_bias_docker.sh --dev /data/OO-RHIC_phys phys3_central
# a family name stitches the one-window sub-samples; the minimum-bias reference is phys_mb
python oo_bias_jets.py --out /data/OO-RHIC_phys --sample oo_vac -j 16   # -> results/oo_bias_jets_oo_vac.json
python oo_bias_jets.py --out /data/OO-RHIC_phys --sample oo_med -j 16
# the phase-3 analysis: selection (oo_vac) x quenching (oo_med / oo_vac) per class, R_AA vs pp_ref, STAR I_CP, sPHENIX R_CP
python oo_analysis_rhic.py --out /data/OO-RHIC_phys -j 16     # -> results/oo_analysis_rhic.json (+ per-sample caches; --cached reuses them)
jupyter nbconvert --to notebook --execute oo_analysis_rhic.ipynb --output-dir results --output oo_analysis_rhic_executed.ipynb
```

### Results

Not public yet: the numbers and conclusions are kept in a local, git-ignored `phys_readme.md` next to this file.

## Files in this directory

| file | |
|---|---|
| `oo_analysis_rhic.py`, `oo_analysis_rhic.ipynb` | the main analysis of the physics run (phase 3; replaces `closure_bias/oo_centrality_bias.ipynb` for it): the samples `oo_vac`, `oo_med`, `pp_ref`, `pp_med` stitched over the p̂<sub>T</sub> windows, classes from `phys_mb` (EPD, MBD, mid-rapidity, impact parameter as truth). Selection = R<sub>AA</sub> of `oo_vac` per class against N<sub>coll</sub> × `pp_ref`; quenching Q = `oo_med` / `oo_vac` in the same classes (independent errors: the same seeds are not the same events), Q<sub>b</sub> vs Q<sub>mult</sub>; total R<sub>AA</sub> / R<sub>CP</sub> of `oo_med`. Observables: charged hadrons, sPHENIX full jets R = 0.4, STAR charged jets R = 0.2 / 0.5, STAR's I<sub>CP</sub> (h–h, h+jet; arXiv:2604.13935's definitions), I<sub>AA</sub>, `pp_med` / `pp_ref` and the QS check. The medium's holes: jets subtract the holes and partner copies in the cone at parton level (full four-momentum for the full jets, 2/3 of p<sub>T</sub> for the charged jets; `*_raw` without); hadron-level observables (charged-hadron R<sub>AA</sub>, h–h I<sub>CP</sub>) need `prod_OO_isr/negative_hadrons.py` (follow-up). Stitching: a p̂<sub>T</sub> window enters a bin where every compared sample has ≥ 10 raw entries pooled over the classes (one choice for all classes and estimators); per bin the dropped share of each sample's yield and the incomplete low-p<sub>T</sub> bins (lowest window > 2 %) are reported; histograms window × class × bin with `hist`. Overlapping p̂<sub>T</sub> windows (3e: narrow 35–40 and 47–55 GeV samples inside 30–40 and 40–55) are combined piece by piece (every window cut at all window edges; per piece σ from the covering windows, weight σ<sub>piece</sub> / events in the piece; `--split-at` for the closure test). Full jets are also split by flavour (`sph_jet_q`, `sph_jet_g`: the nearest hard outgoing parton within ΔR < 0.4). `--phase3` restricts to the phase-3 jobs (no 3b extensions, no 3e samples; outputs with the suffix `_phase3`), `--phase3 --with-3e` adds the 3e samples (suffix `_phase3e`); without either, all finished jobs. The notebook uses all jobs and shows what 3e and 3b changed (sections 5b, 5c). Data overlays from `refs/star_oo_icp/` and `refs/sphenix_oo_jet/` → `results/oo_analysis_rhic.json`; `-j` parallel file workers, `--cached` Re-hadronized samples (`../../prod_OO_isr/rehadronize_colorless.py`): `--out ROOT_REHAD --est-root ROOT_ORIG --tag colorless` keeps every event in the class of its original estimators and bulk, and subtracts the hole hadrons (pstat −1) at hadron level (notebook §6b). Fragmentation oversamples (`--n-frag` of the rehadronize tools): each sample is a sub-event (jets and trigger per sample, the event's holes, partners and bulk copied to it), averaged into the event's row, as `auau_calibration/auau_raa.py`; the classes use the first sample's hadrons; samples Pythia could not hadronize (no hadrons) are left out of the average. |
| `oo_bias.py` | the GPU sample driver of the whole program (`run`, `analyse`, `--help`): the sets and samples of the bulk tune (`pp_`, `oo_`, `dau_<set>`), R4b (`mb`, `jets`, `jets_quench`, `jets_holes`), gate G2 and the physics run (`phys_mb`, `pp_bulk_*`, `jets_vac*`, `oo_vac`, `oo_med`, `pp_ref`, `pp_med`, one sub-sample per p̂<sub>T</sub> window) and Au+Au (`auau_med`, `auau_t3`); checks the tune XMLs (`rhictune_xml.py`) before launching; `analyse`: R4b classes, R<sub>AA</sub><sup>vac</sup>, bias factor, R<sub>CP</sub> → `results/oo_bias.json` |
| `oo_bias_jets.py` | the R<sub>AA</sub><sup>vac</sup> / R<sub>CP</sub> of `oo_bias.py`'s samples against the jet p_T and the charged-hadron p_T: anti-k_T R = 0.4 jets (\|η\| < 0.7; charged jets for `--charged` samples, full jets for `jets_quench`) and charged hadrons (\|η\| < 1) on the hard event's hadrons, per p̂_T window and stitched over the windows (a window enters a bin only with ≥ `--min-entries` (10) raw entries in every class); class sums for quenched/vacuum double ratios; LBT hole energies of the colour audit → `results/oo_bias_jets.json`, `results/oo_bias_jets_jets_quench.json`. `--sample` also takes a comma-separated list or a physics-run family (`jets_vac`, `jets_med`, ...): its one-window sub-samples `<name>_w<lo>_<hi>` (own directories) are stitched on one window list by their cross sections, minimum bias `phys_mb` (`--mb` overrides) |
| `oo_bias_docker.sh` | `oo_bias.py` on a multi-GPU Docker host: one container per GPU, each its shard (`--shard g/N`), the settings benchmarked on the 2 × RTX 3090 (8 jobs per GPU, OMP 3, MPS); `--dev` mounts this checkout's `example/` over the image's |
| `hstat.py` | statistical errors for the analyses and notebooks (hist `Weight`/`Mean` storages): histograms, means per bin or class (weighted with importance weights), per-event particle histograms (event-to-event spread), class fractions, experiment-style R_AA^vac and R_CP of N_coll-weighted jet events; plotting helpers (bands, error bars) |
| `rhictune_xml.py` | writes `config/pp200_isr_rhictune.xml` (p+p 200 GeV in the full chain: strings, hydro, iSS, Matter/LBT; the physics run's p+p reference `pp_ref_w*`) from the O+O tune XML with only the nuclei changed, and checks (`check`; also `oo_bias.py` before launching) that the two differ nowhere else. Not to be confused with `pp200_isr_vacuum.xml`, the hard tune's chain without strings and hydro. The earlier p+p samples (O+O XML with `--xml-set` nuclei) give the same job XMLs (driver dry run, 163 elements identical) |
| `_paths.py` | puts this directory and the test directories on `sys.path` (see Layout) |

## Reference data (`refs/`)

| file | |
|---|---|
| `refs/star_oo_icp/` | STAR O+O h–h and h+jet I<sub>CP</sub> (arXiv:2604.13935, HEPData ins3145466 v1: Figs. 3, 5, 6 as `figure*.yaml`; git-ignored like all HEPData YAML). hepdata.net blocks this box (Cloudflare challenge): download the record's submission zip in a browser (https://www.hepdata.net/record/ins3145466, "Download All", YAML) and unzip its `*.yaml` here; read by `oo_analysis_rhic.py` (`load_data`). The record has no EPD multiplicity distributions (Figs. 1, 4) |
| `refs/star_auau200_chjet/` | STAR charged-jet R<sub>AA</sub><sup>Pythia</sup>, Au+Au 0–10 % and 60–80 %, R = 0.2–0.4 (arXiv:2006.00582 Fig. 17), read from the vector figure: `extract_star_chjet_raa.py PAPER.pdf` → `star_chjet_raa.json` |
| `refs/phenix_auau200_pi0/` | PHENIX π⁰ R<sub>AA</sub> Au+Au 0–10 % (PRC 87, 034911, Table VII): `phenix_pi0_raa_0_10.json` |
| `refs/star_auau200_hadrons/` | STAR π± R<sub>AA</sub> 0–12 % (1110.0579, HEPData ins930463) and (h⁺+h⁻)/2 R<sub>AA</sub> 0–5 %, 10–20 % (nucl-ex/0305015, HEPData ins619063), and their measured p+p spectra: `fetch_star_hadron_raa.py` → `star_hadron_raa.json` (YAML git-ignored) |
| `refs/jetscape_2204_fig14/` | JETSCAPE's MATTER+LBT Type 3 (α<sub>s</sub> 0.3; TRENTo + VISHNU, PP19) Au+Au 0–10 % charged-jet (R = 0.2–0.4) and π R<sub>AA</sub>, arXiv:2204.01163 Fig. 14, read from the vector figure: `extract_js_fig14.py PAPER.pdf` → `js_fig14.json` |
| `refs/hadronize_pp19_icdecays.xml` | Colorless with iColored's decays (π⁰, η decay; K⁰<sub>S</sub>, Λ, Σ, Ξ, Ω do not), for the PP19 p+p reference (`prod_AuAu_0_10_jet/run_prod_vacuum.py --user-xml refs/pp19_200.xml`) and for JETSCAPE's chain on our medium (`prod_AuAu_0_10_jet/run_prod_jet_single.py`, run C) |
| `refs/sphenix_pp_jet/` | sPHENIX's preliminary p+p 200 GeV inclusive jet cross section (sPH-CONF-JET-2025-03; anti-k<sub>T</sub> R = 0.4, \|η\| < 0.7, 15–72 GeV, unfolded): `extract_sphenix_pp_jet.py NOTE.pdf` reads Fig. 5 from the vector paths (log axis from the decade ticks) → `sphenix_pp_jet_prelim.json`. The drawn statistical bars are the same in every bin (a drawing size: upper bounds); the systematic boxes are largely correlated. Preliminary, for comparison only |
| `refs/sphenix_oo_jet/` | sPHENIX's preliminary O+O jet R<sub>CP</sub> (vs 60–80 % and vs 40–60 %) and R<sub>AA</sub> 0–100 % (sPH-CONF-JET-2026-02; R = 0.4, \|η\| < 0.7, MBD classes, 19–60 GeV): the note has no value tables, so `extract_sphenix_oo_jet.py NOTE.pdf` reads the points, statistical bars and asymmetric systematic boxes from the PDF's vector paths (`pdftocairo -svg`; axes mapped with their major ticks) → `sphenix_oo_jet_prelim.json`. The lowest bins' statistical bars (±0.017–0.018 in every panel) are probably the drawing's minimum: upper bounds there. Also the note's model curves (key `theory`: per vertex the line's value and the band's edges): JEWEL 2.4.0 recoils off / on (R<sub>AA</sub> 0–100 %, R<sub>CP</sub>; Figs. 6, 9, 10), Angantyr's centrality-bias factors on R<sub>CP</sub> (Fig. 8), the no-quenching nPDF baseline with nNNPDF3.0 / EPPS21 / TUJU21 (Fig. 6); `oo_analysis_rhic.ipynb` Figure 3c sets them against `oo_vac` (selection) and `oo_med` / `oo_vac` (quenching). Preliminary, for comparison only |
| `refs/rhic_pp200/` | RHIC p+p 200 GeV multiplicity references besides UA5 (none on HEPData): PHOBOS inelastic dN/dη for \|η\| ≤ 5.3 read from the vector figure of arXiv:1011.1940 (Fig. 28a, 90 % C.L. band), STAR NSD and PHENIX minimum-bias mid-rapidity values (`extract_rhic_pp200.py --phobos <source dir>` → `rhic_pp200.json`); used by `bulk_tune/pp_bulk.py` |
| `refs/dau_200/` | d+Au references: the HEPData tables (fetched, git-ignored) and `extract_star_ftpc_dau.py` → `star_dau_ftpc.json`, STAR's FTPC d+Au dN/dη read from a ROOT figure of the d+Au FTPC analysis (2004; the figure is not public and stays local, git-ignored; its 40–100 % and minimum-bias points are rescaled to the published \|η\| < 1 normalisation) |
| `refs/star_oo/` | STAR O+O references read from the vector figures of arXiv:2510.19645 and 2604.13935 (`extract_star_oo.py` → `star_oo_bulk.json`; no HEPData) |
| `refs/run_refs.py` | the reference models' events (PP19 via `prod_AuAu_0_10_jet/run_prod_vacuum.py`, Pythia standalone) |
| `refs/pythia_ref.py` | standalone Pythia 8 (default, Detroit, no CR; `--windows`, `--eta-max 0`, `--line` for diagnostics) in the tune's hadron-file layout |
| `refs/pp19_200.xml`, `refs/hadronize_pp19.xml` | PP19 at 200 GeV: PythiaGun + vacuum Matter (Q0 1, C_v 0.25) + Colorless, π⁰ and η undecayed |

HEPData YAML tables are git-ignored (`hard_tune/fetch_data.py`, `pp_bulk.py fetch`, `dau_bulk.py fetch`,
`pt_bulk.py fetch` and the `fetch_*.py` / `extract_*.py` scripts in `refs/` make them again).

## Environment

The analyses need only the analysis environment (`utils/analysis_env`, `pyjetscape[analysis]`:
numpy, scipy, h5py, hdf5plugin, awkward, fastjet, hist, pyyaml, matplotlib, jupyter); running events
needs X-SCAPE (the production container or a source build). The hard-tune fit needs `emcee` and
`scikit-learn` (`hard_tune/README.md`); the string and hadronization diagnostics need Pythia 8's
Python module (`pythia8` in the image, the PyPI wheel `pythia8mc` in the analysis venv).

## Definitions and caveats

- **Holes and partners (the jet in the medium).** Matter (`recoil_on` 1) and LBT leave holes
  (pstat −1) that iColored drops, and for part of them it inserts colour partners with the
  hole's three-momentum. **The physics runs store both with `--store-partons
  --store-hadronization-input`** (rule and details: `prod_OO_isr/README.md`, "Holes and
  partners: what a physics run must store"). Then: full jets subtract holes + partners at
  parton level in the cone; leading hadrons above 3 GeV need nothing; charged jets, hadrons
  below 3 GeV, substructure, soft and recoil observables subtract the negative hadrons of
  `negative_hadrons.py`. Validation: `closure_bias/validate_holes.py`.
- Statistical errors: every model point carries one (`hstat.py`). Each event has one sample (no
  hydro or iSS oversampling), so events are independent; particle distributions take the
  event-to-event spread (√N of particles is too small: particles of one event are correlated),
  weighted quantities use Σw². Fragmentation oversamples (`--n-frag` of the rehadronize tools) are
  averaged per event before any sum. Stitched spectra report n<sub>eff</sub> = (Σw)²/Σw² per bin where
  thin p̂<sub>T</sub> windows matter (`auau_calibration/auau_raa.py`).
