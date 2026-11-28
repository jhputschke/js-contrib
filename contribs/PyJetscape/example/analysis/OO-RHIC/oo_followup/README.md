# oo_followup: reading the O+O results (energy loss, selection bias, spectrum slope)

Follow-ups of the phase-3 O+O results of `../oo_analysis_rhic.py` (tuning page "Follow-up" A1, A2,
B4, and the jet-spectrum slope, 2026-11-25): how much energy do the jets lose in O+O and where does
it go, how does the selection bias arise, and how does the steepness of the 200 GeV jet spectrum
turn a small fractional loss into the measured R<sub>CP</sub>? They read the caches of
`../oo_analysis_rhic.py` (`../results/oo_analysis_rhic_<family>.npz`) and the physics-run files.

**Model results, not vetted physics results.** sPHENIX's data are preliminary.

## Run it

```bash
cd contribs/PyJetscape/example/analysis/OO-RHIC/oo_followup
python pp_jet_spectra.py -j 16                    # pp_ref jet cross sections in STAR's / sPHENIX's bins -> ../results/pp_jet_spectra.json (--phase3)
python quench_diag.py --out /data/OO-RHIC_phys -j 16 both   # -> ../results/quench_diag_eloss.npz, quench_diag_bias.npz
jupyter nbconvert --to notebook --execute quench_diag.ipynb --output-dir ../results --output quench_diag_executed.ipynb
jupyter nbconvert --to notebook --execute jet_spectrum_slope.ipynb --output-dir ../results --output jet_spectrum_slope_executed.ipynb
```

## Results

Not public yet: the numbers and conclusions are kept in a local, git-ignored `phys_readme.md` next to this file.

## Files

| file | |
|---|---|
| `pp_jet_spectra.py` | the model's p+p jet cross sections in the experiments' own definitions and bins, for `jet_spectrum_slope.ipynb`: `pp_ref`, all finished jobs (phase 3 + 3e; `--phase3` for phase 3 alone), full jets R = 0.5 at \|η\| < 0.5 and 0.5–0.9 (STAR's bins) and R = 0.4 at \|η\| < 0.7 (sPHENIX's bins), stitched with `stitch_pieces`, absolute → `../results/pp_jet_spectra.json` (`_phase3.json`). |
| `quench_diag.py`, `quench_diag.ipynb` | follow-up A1, A2, B4: per hard outgoing parton (`initiators/` pstat 1000, p<sub>T</sub> > 10 GeV, \|η\| < 0.5) the energy given to the medium (recoils − holes, in the hemisphere and the cone) and the jet loss 1 − ⟨p<sub>T,jet</sub>/p<sub>T,parton</sub>⟩<sub>med</sub>/⟨…⟩<sub>vac</sub> at parton level (without and with the response) and hadron level; quark vs gluon, production radius; the medium's time scales (MUSIC start, hole times, bulk freeze-out); and for the selection bias the forward signal of `oo_vac` jet events over minimum bias at the same N<sub>coll</sub> against p̂<sub>T</sub> and x, with the energy taken from each nucleus and the beam remnants' energy. `python quench_diag.py --out ROOT -j N both` → `../results/quench_diag_eloss.npz`, `../results/quench_diag_bias.npz` (reads the `oo_analysis_rhic.py` caches) |
| `jet_spectrum_slope.ipynb` | how steep the 200 GeV p+p jet spectrum is, and what that does to R<sub>AA</sub>/R<sub>CP</sub>: local power n(p<sub>T</sub>) = −d ln σ/d ln p<sub>T</sub> of sPHENIX (R = 0.4, preliminary), STAR (R = 0.5) and our `pp_ref` (R = 0.4), fits A p<sub>T</sub><sup>−n₀</sup>(1−x<sub>T</sub>)<sup>m</sup> integrated over the bins. Also the model's checks: jet flavour (each full jet tagged by the nearest hard outgoing parton, `initiators/` pstat 1000; ε per flavour) and cone size (charged jets R = 0.2 vs 0.5). Reads the all-jobs outputs of `oo_analysis_rhic.py` (`../results/oo_analysis_rhic.json`, `*.npz`) and `pp_jet_spectra.py`, `../data/data.json`, `../refs/sphenix_pp_jet/` |
