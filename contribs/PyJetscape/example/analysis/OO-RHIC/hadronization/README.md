# hadronization: jets hadronized in a medium (Colorless's close-pair shift, iColored)

How the Au+Au R<sub>AA</sub> of `../auau_calibration/` depends on the hadronization of medium-modified
showers, and which hadronization to use for the energy-loss calibration. Studied 2026-11-26/27 (GB10 session): X-SCAPE's ColorlessHadronization gives every pair of partons closer than
4 Λ<sub>QCD</sub> a ±0.8 GeV p<sub>z</sub> kick without taking the energy from anywhere ("Making collinear partons
not collinear", in JETSCAPE since v3.0, so also in 2204.01163). It fires once per close pair, so it
grows with the square of the number of soft recoils and holes.

**Model results, not vetted physics results.**

## Samples and names

| name in the study | sample |
|---|---|
| JS T3-20k = run C | JETSCAPE's chain (PythiaGun, Matter + LBT Type 3, Colorless) on our medium, 20k events |
| JS T3-78k = run 1 | the same, 78k events (a: 20k at hydro reuse 20; b: 42k at reuse 21; c: 16k in the 5–9 GeV windows) |
| XS-ISR T2 / T3 = B1 / B2 | our ISR chain (iColored), Type 2 / Type 3, a hydro per event |
| XS-ISR T3 (reuse) = run 1′ | our ISR chain, B2's settings, hydro reused 20×, 78k events |
| XS-ISR pp = pp_ref; PP19 | the p+p references of the two chains |

Every sample is re-hadronized with n<sub>frag</sub> = 10 per event: CL+CPS (Colorless as in production),
CL (Colorless without the shift), iCol-R (iColored with the real stored beam remnants,
`prod_OO_isr/rehadronize_icolored.py --beam-ends stored`), iCol-B (iColored with remnant-like ends
at ±√s/6), each against its own p+p reference hadronized the same way. The "without the shift"
samples need X-SCAPE with the loop behind `<JetHadronization><close_pair_shift>`
(`colorless_close_pair_shift.diff`, X-SCAPE 299fd3f; not in X-SCAPE, local build on the GB10).

## Run it

The data live on the GB10 (`~/prod/...`; `$CSS_ROOT` points the scripts elsewhere, e.g. a mirror).

```bash
cd contribs/PyJetscape/example/analysis/OO-RHIC/hadronization
python make_colorless_shift_nb.py                 # writes colorless_shift_study.ipynb (no outputs)
CSS_ROOT=/data/AuAu-RHIC_phys/gb10 jupyter nbconvert --to notebook --execute colorless_shift_study.ipynb \
    --output-dir ../results --output colorless_shift_study_executed.ipynb
# the notebook runs ../auau_calibration/auau_raa.py per sample and hadronization where
# ../results/auau_raa_css_<tag>.json is missing, and the two diagnostics:
python css_gap_diag.py -j 12                      # run C vs run 1 per pTHat window -> ../results/css_gap_diag.json
python css_parton_jets.py -j 12                   # parton-level reference, energy balance -> ../results/css_parton_jets.json
python colorless_shift_count.py                   # firings and added energy per event (vacuum ... Au+Au, O+O)
python icolored_diag.py -j 16                     # iColored replayed in standalone Pythia, string anatomy
python colorless_emul.py -j 16                    # Colorless re-implemented on Pythia, ablations
```

`colorless_emul.py`, `icolored_diag.py` and `../hard_tune/diag_strings.py` need Pythia 8's Python
module: the production image has `pythia8`; in the analysis venv `setup_analysis_env.sh` installs the
PyPI wheel `pythia8mc` (8.312), which the scripts import as a fallback.

## Results

Not public yet: the numbers and conclusions are kept in a local, git-ignored `phys_readme.md` next to this file.

## Files

| file | |
|---|---|
| `icolored_diag.py` | why iColored and Colorless differ on the same partons (GB10 layout, `~/prod/...`): iColored replayed in standalone Pythia from `hadronization_input/` (reproduces the stored hadrons), every high-p<sub>T</sub> hadron traced to its string and the string's partons (jet, recoil, closure partner, remnant); per-event energy and p<sub>T</sub> bookkeeping; z of the leading hadron. Model results, not vetted |
| `colorless_emul.py` | X-SCAPE's ColorlessHadronization re-implemented on standalone Pythia (string ends, gluon chaining, the close-pair shift, fakes at ±√s/6) for ablations: with and without the shift, without recoils |
| `colorless_shift_count.py` | how often Colorless's close-pair shift fires per event and how much energy it adds (vacuum, T1, B1, B2, run C, O+O) |
| `colorless_shift_study.ipynb`, `make_colorless_shift_nb.py` | the close-pair shift (CPS) study: Au+Au 0–10 % R<sub>AA</sub> of the JS chain (JS T3-20k = run C, JS T3-78k = run 1) and the XS-ISR chain (XS-ISR T2/T3 = B1/B2, XS-ISR T3 (reuse) = run 1′), each hadronized with CL+CPS, CL (no shift), iColored with real remnants / √s/6 beam ends, against its own p+p reference hadronized the same way; spectra with n<sub>eff</sub>, the chains compared, per-window diagnostics, the contribution from thinly sampled windows (`auau_raa.py --min-entries 5`), the holes' colour closure, the parton-level reference and Colorless's energy balance; motivation, reading and conclusions. Generated without outputs (`python make_colorless_shift_nb.py`; data under `$CSS_ROOT`, GB10 layout); executed copy and HTML in `../results/` |
| `css_gap_diag.py` | JS T3-20k vs JS T3-78k without the CPS, per p̂<sub>T</sub> window: entries and yield shares per h± bin, R<sub>AA</sub> with a minimum-entries rule, per-window contributions to the gap, parton-level comparison. → `../results/css_gap_diag.json` |
| `css_parton_jets.py` | the parton-level jet reference (anti-k<sub>T</sub> R = 0.2, 0.4; shower + ISR partons, LBT recoils, holes subtracted in the cone, as `../oo_followup/quench_diag.py` A1), its R<sub>AA</sub> with `../auau_calibration/auau_raa.py`'s weights, hadron-level jets of the same definition per hadronization, the matched ⟨p<sub>T,had</sub>/p<sub>T,parton</sub>⟩ in Au+Au and p+p and their double ratio, and Colorless's energy balance (hole subtraction, inclusive and in the jet cone). → `../results/css_parton_jets.json` |
| `collinear_split.py` | collinear-safety test: splits every gluon above a p<sub>T</sub> into exactly collinear colour-chained gluons in `partons/` and `hadronization_input/` |
| `colorless_close_pair_shift.diff` | the X-SCAPE patch (299fd3f) that puts Colorless's close-pair shift behind `<JetHadronization><close_pair_shift>` (default 1 = unchanged); used for the "no shift" samples of `../auau_calibration/auau_jet_raa.ipynb` Fig. 8. Not in X-SCAPE |
