# closure_bias: the O+O vacuum closure and the centrality bias (R3, R4, R4b)

The tests of the "plan forward" before the O+O physics run (tuning page, 2026-11-22/23): does the
O+O chain reproduce N<sub>coll</sub> × p+p when nothing is quenched, and how much does the selection by
multiplicity bias the jet events' classes?

- **R3** — the p+p reference from the O+O tune XML (projectile/target p).
- **R4** — the O+O vacuum closure in impact-parameter classes: R<sub>AA</sub> = 1 up to energy
  conservation in the hard yield per binary collision?
- **R4b** — the centrality bias from the hard-momentum subtraction: first a string-energy proxy
  (CPU), then the full test with hydro (GPU): minimum bias, jets in vacuum, jets in the medium.
- **Holes and partners** — the validation of the hole and partner treatment that set the storage
  rule of the physics runs (`--store-partons --store-hadronization-input`).

**Model results, not vetted physics results; statistical errors only.** The physics run itself and
its analysis are one directory up (`../oo_analysis_rhic.py`); these samples are not reused there
(old bulk set, gaps in p̂<sub>T</sub>).

## Run it

```bash
cd contribs/PyJetscape/example/analysis/OO-RHIC/closure_bias
# R3 + R4: p+p reference from the O+O tune XML and the O+O vacuum closure (CPU, ~40 min on 20 cores, 3 GB)
python closure_oo.py run -j 20 && python closure_oo.py analyse -j 10
python centrality_proxy.py run -j 20 && python centrality_proxy.py analyse     # R4b part 1 (CPU, ~8 min)
# with the string switch (X-SCAPE oo_rhic_run): jet samples with --hard-pair-string-veto 0 (closure + subtraction alone)
python centrality_proxy.py run jet_sub_noveto jet_nosub_noveto -j 20 && python centrality_proxy.py analyse && python centrality_proxy.py smear
jupyter nbconvert --to notebook --execute closure_oo.ipynb --output-dir ../results --output closure_oo_executed.ipynb
# R4b full test and R4c (GPU, O+O with hydro; oo_bias.py --help): minimum bias, jets in vacuum, jets quenched
python ../oo_bias.py run mb -j 8 --mps && python ../oo_bias.py run jets -j 8 --mps     # ~9 h on the GB10
python ../oo_bias.py run jets_quench -j 8 --mps                                     # the jet in the medium
python ../oo_bias.py analyse && python ../oo_bias_jets.py && python ../oo_bias_jets.py --sample jets_quench   # oo_bulk.py: ../bulk_tune/
# holes and partners: 4,500 quenched events with --store-partons --store-hadronization-input (~50 min on the GB10),
# negative hadrons per job (CPU), then the validation
python ../oo_bias.py run jets_holes -j 8 --mps
for f in ~/prod/OO-RHIC_OO/bias/jets_holes/*_hadrons_jet_frag.h5; do python ../../../prod_OO_isr/negative_hadrons.py "$f"; done
python validate_holes.py
jupyter nbconvert --to notebook --execute oo_centrality_bias.ipynb --output-dir ../results --output oo_centrality_bias_executed.ipynb
```

## Results

Not public yet: the numbers and conclusions are kept in a local, git-ignored `phys_readme.md` next to this file.

## Files

| file | |
|---|---|
| `closure_oo.py`, `closure_oo.ipynb` | R3 + R4 of the plan forward: the p+p reference from the O+O tune XML (`config/OO_MCGlauber_MUSIC_isr_rhictune.xml`, projectile/target p) and the O+O vacuum closure in impact-parameter classes; the notebook explains the test and shows the figures (`PROD/closure/`, `../results/closure_oo*.{json,npz}`) |
| `centrality_proxy.py` | R4b part 1: centrality bias from a string local-energy proxy (`smear`: negative-binomial fluctuations anchored to sPHENIX's MBD efficiency) (`run_prod_oo_isr.py`'s `strings_eloc_*` units) in STAR's estimator windows; minimum-bias 3dMCGlauber alone vs N_coll-weighted jet events, subtraction on/off → `../results/centrality_proxy.json`; figures in `closure_oo.ipynb` |
| `oo_centrality_bias.ipynb` (samples: `../oo_bias.py`) | R4b full test: O+O with hydro and one iSS sample per event, minimum bias and N_coll-weighted jet events (jet in vacuum: `--vacuum-jet`; `jets_quench`: in the medium), classes from the multiplicity in STAR's EPD, mid-rapidity and sPHENIX's MBD; R_AA^vac, bias factor, R_CP with errors → `../results/oo_bias.json` |
| `validate_holes.py` | validation of the hole and partner treatment on the `jets_holes` sample (made with `--store-partons --store-hadronization-input`, negative hadrons from `prod_OO_isr/negative_hadrons.py`): bookkeeping against iColored's colour audit, hole and partner energy per class, full-jet R<sub>CP</sub> uncorrected / parton-level subtraction / negative hadrons, charged jets, leading hadrons → `../results/validate_holes.json` |

The samples (`mb`, `jets`, `jets_quench`, `jets_holes`) are defined and launched by `../oo_bias.py`;
`../oo_bias_jets.py` gives their jet and hadron R<sub>CP</sub>.
