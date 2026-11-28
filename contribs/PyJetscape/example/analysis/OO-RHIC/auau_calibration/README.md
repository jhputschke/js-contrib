# auau_calibration: the Au+Au 200 GeV 0–10 % energy-loss calibration

Is the energy loss of the O+O chain (Matter + LBT) right? Tested in central Au+Au (0–10 %,
b < 4.7 fm) against STAR's charged-jet R<sub>AA</sub> (arXiv:2006.00582, STAR's definition: leading-track
cut on Au+Au only), PHENIX π⁰ and STAR π±, h±, and against JETSCAPE's own calculation
(2204.01163 Fig. 14), all on the same medium (the RHIC-tune 3dMCGlauber strings + 3D MUSIC);
tuning page "Au+Au calibration + hadronization", 2026-11-26. Samples: B1 (`auau_med_w*`, Type 2,
α<sub>s</sub> 0.25), B2 (`auau_t3_w*`, Type 3, α<sub>s</sub> 0.3), both from `../oo_bias.py` (our ISR chain,
iColored), and run C (JETSCAPE's chain on our medium: PythiaGun, Matter + LBT Type 3, Colorless;
`prod_AuAu_0_10_jet/run_prod_jet_single.py`), each against its own p+p reference.

**Model results, not vetted physics results.**

`auau_raa.py` is also the R<sub>AA</sub> engine of `../hadronization/` (the high-statistics study that
superseded the hadronization numbers of `auau_jet_raa.ipynb`).

## Run it

```bash
cd contribs/PyJetscape/example/analysis/OO-RHIC/auau_calibration
python auau_raa.py --family auau_med -j 6          # B1 -> ../results/auau_raa_auau_med.json
python auau_raa.py --family auau_t3 -j 6           # B2
python auau_raa.py --family js_chain_t3 -j 6 \
    --auau-glob '/data/AuAu-RHIC_phys/js_chain_t3/*_hadrons_jet_frag.h5' \
    --pp-glob '/data/OO-RHIC_pp_refs/pp19_icdecays/*_hadrons_jet_frag.h5'   # run C against PP19 (quote the globs)
python auau_raa.py ... --min-entries 5             # the stitching rule (counts p+p entries; --min-entries-on both)
python pp_ref_check.py -j 6                        # p+p reference vs data and Pythia 8 -> ../results/pp_ref_check.json
python chain_diag.py -j 6                          # per window: energy loss, partons, hadrons -> ../results/chain_diag.json
# the notebook runs the scripts above itself where their JSON is missing
jupyter nbconvert --to notebook --execute auau_jet_raa.ipynb --output-dir ../results --output auau_jet_raa_executed.ipynb
```

## Results

Not public yet: the numbers and conclusions are kept in a local, git-ignored `phys_readme.md` next to this file.

## Files

| file | |
|---|---|
| `auau_raa.py` | the Au+Au 200 GeV 0–10 % energy-loss calibration (`oo_bias.py` groups `auau_b1` = `auau_med_w*`, Type 2 α<sub>s</sub> 0.25; `auau_b2` = `auau_t3_w*`, Type 3 α<sub>s</sub> 0.3): R<sub>AA</sub> against `pp_ref` (the same \|y\| < 1 parton cut) for pions (PHENIX's and STAR's acceptances), charged hadrons, charged jets R = 0.2–0.4 (holes + partners subtracted with 2/3; without and with the 5 GeV leading-track cut; `_star`: the cut on Au+Au only, STAR's R<sub>AA</sub><sup>Pythia</sup> definition), full jets R = 0.4; every jet also unsubtracted (`_raw`); the bulk at mid-rapidity. `python auau_raa.py --family auau_med -j 6` → `../results/auau_raa_<family>.json`. Holes: parton level for iColored files (holes + partner copies, charged × 2/3), hadron level (JETSCAPE's: pstat −1 hadrons subtracted) for Colorless files (`run_prod_jet_single.py`, `rehadronize_colorless.py`); `--auau-glob` / `--pp-glob` take other samples (run C: `/data/AuAu-RHIC_phys/js_chain_t3`, against PP19) |
| `pp_ref_check.py` | the p+p reference of `auau_raa.py` against the measured p+p spectra (STAR π±, h±; PHENIX π⁰) and against Pythia 8 Monash / Detroit (`../refs/pythia_ref.py` with iColored's decays: `--line "111:mayDecay = on" --line "221:mayDecay = on" --line "ParticleDecays:limitTau0 = on" --line "ParticleDecays:tau0Max = 10.0"`, in `/data/OO-RHIC_pp_refs/pythia8_<tune>_icdecays`): hadron spectra, charged jets with and without a perpendicular-cone UE subtraction. `python pp_ref_check.py -j 6` → `../results/pp_ref_check.json` |
| `chain_diag.py` | where between energy loss, partons and hadrons the Au+Au samples part: per sample and p̂<sub>T</sub> window the energy given to the medium (recoils − holes), holes, shower initiators, final partons and charged hadrons above thresholds, hadrons per parton (B1, B1 re-hadronized with Colorless, B2, run C, `pp_ref` both ways, PP19). `python chain_diag.py -j 6` → `../results/chain_diag.json` (`auau_jet_raa.ipynb` Fig. 6) |
| `auau_jet_raa.ipynb` | Au+Au 0–10 %: model charged-jet R<sub>AA</sub> against STAR (2006.00582), pion and hadron R<sub>AA</sub> against PHENIX π⁰ and STAR π±, h±; the p+p reference against data and Pythia 8, and the jet R<sub>AA</sub> with Pythia 8 as the reference (as STAR); per-parton loss Au+Au vs O+O; the reading. Runs `auau_raa.py` (families in `SHOW`) and `pp_ref_check.py` if needed. Generated without outputs; executed copy: `../results/auau_jet_raa_executed.ipynb` |
