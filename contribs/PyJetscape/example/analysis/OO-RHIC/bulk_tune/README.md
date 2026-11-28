# bulk_tune: the p+p / O+O / d+Au bulk (3dMCGlauber + MUSIC) at 200 GeV

The soft sector of the O+O chain: 3dMCGlauber strings + MUSIC + iSS bulk hadrons, checked and
hand-tuned against the RHIC multiplicity data so that O+O centrality classes and the p+p
reference sit at the right activity. Plan step A (screening) of the "RHIC p+p and O+O Tuning"
page and its follow-ups (a smaller yloss variance, a 2 × 2 grid in variance and node level, the
remnant α<sub>rem</sub> scan, the d+Au check, shadowing, τ<sub>form</sub>, identified p<sub>T</sub>), 2026-11-23/24; the
Au+Au comparison with PHOBOS followed on 2026-11-25.

**Model results, not vetted physics results; statistical errors only.**

## Run it

The GPU samples come from `../oo_bias.py` (sets, groups `screen`, `var`, `grid`, `rem`, `shad`,
`tau`, samples `pp_<set>`, `oo_<set>`, `dau_<set>`) on a multi-GPU Docker host, one container per
GPU (`../oo_bias_docker.sh`); events in `/data/OO-RHIC_bulk`. The analyses here only read files
and write to `../results/`.

```bash
cd contribs/PyJetscape/example/analysis/OO-RHIC/bulk_tune
# p+p / O+O bulk tune, step A (screening) and follow-ups, on a multi-GPU Docker host (one container per GPU)
../oo_bias_docker.sh --dev /data/OO-RHIC_bulk screen      # map, pp2407, m3: 10k p+p + 10k O+O each
../oo_bias_docker.sh --dev /data/OO-RHIC_bulk var         # pp2407 with yloss variance 0.3
../oo_bias_docker.sh --dev /data/OO-RHIC_bulk grid        # variance 0.4/0.45 x nodes 2.15/2.25
../oo_bias_docker.sh --dev /data/OO-RHIC_bulk rem         # alpha_rem 0.4/0.65/0.8/1.0 on g_v045_n215 (p+p)
../oo_bias_docker.sh --dev /data/OO-RHIC_bulk dau_map dau_g_v045_n215   # d+Au, the MAP's calibration data
../oo_bias_docker.sh --dev /data/OO-RHIC_bulk shad        # shadowing 0.6/0.8, ylossParam4At2 1.1 (d+Au, O+O)
../oo_bias_docker.sh --dev /data/OO-RHIC_bulk tau         # tau_form_mean 0.424 = the final XML (p+p, O+O, d+Au)
python pp_bulk.py fetch && python pp_bulk.py analyse --out /data/OO-RHIC_bulk   # UA5 + refs/rhic_pp200 (PHOBOS, STAR, PHENIX)
python oo_bulk.py analyse --sample oo_pp2407 --out /data/OO-RHIC_bulk          # per O+O sample
python centrality_norm.py
python dau_bulk.py fetch && python dau_bulk.py analyse --out /data/OO-RHIC_bulk   # d+Au vs PHENIX, STAR (TPC, FTPC), BRAHMS
python dau_bulk.py analyse --out /data/OO-RHIC_bulk --class-frac 1.1 dau_g_v045_n215   # classes as the top 110 %
python pt_bulk.py fetch && python pt_bulk.py analyse g_v045_n215 tau0424 map --out /data/OO-RHIC_bulk   # identified pT, <pT> vs STAR
python oo_bulk.py analyse                       # R4c: O+O bulk vs STAR (the R4b mb sample) -> ../results/oo_bulk.json
jupyter nbconvert --to notebook --execute oo_bulk_comparison.ipynb --output-dir ../results --output oo_bulk_comparison_executed.ipynb
jupyter nbconvert --to notebook --execute bulk_tune_screening.ipynb --output-dir ../results --output bulk_tune_screening_executed.ipynb
jupyter nbconvert --to notebook --execute auau_bulk_phobos.ipynb --output-dir ../results --output auau_bulk_phobos_executed.ipynb
```

The adopted set `g_v045_n215` is the 3dMCGlauber block of `config/OO_MCGlauber_MUSIC_isr_rhictune.xml`
and `config/pp200_isr_vacuum.xml` since 2026-11-24, with `tau_form_mean` 0.424 (= sample `tau0424`);
the earlier sets name every parameter they change (`oo_bias.MAP`, `tau_form_mean` 0.5), so their
samples resolve to the parameters they ran with.

## Results

Not public yet: the numbers and conclusions are kept in a local, git-ignored `phys_readme.md` next to this file.

## Files

| file | |
|---|---|
| `oo_bulk.py`, `oo_bulk_comparison.ipynb` | R4c: the O+O bulk of the same events against STAR (p(N_ch), ⟨N_ch⟩ vs centrality, EPD nMIP of minimum-bias and triggered events) → `../results/oo_bulk.json` |
| `pp_bulk.py` | p+p 200 GeV bulk of the `pp_<set>` samples against UA5 (dN/dη NSD and inelastic, NSD P(n) in four η windows; HEPData ins233599, ins267179 → `../refs/ua5_200/`) → `../results/pp_bulk.json` |
| `centrality_norm.py` | how STAR's O+O centrality classes map onto the model's: STAR's normalisation from its own Fig. 5a/5b, the model centrality at STAR's N_ch and EPD edges, the flattest normalisation f, STAR's minimum-bias trigger emulated (from `oo_bulk.py`'s quantiles) → `../results/centrality_norm.json` |
| `bulk_tune_screening.ipynb` | the bulk parameter exploration (step A of the p+p / O+O bulk tune and its follow-ups): the sets and why, p+p against UA5 and the RHIC data, O+O p(N_ch) shape, ⟨N_ch⟩ vs centrality with and without the normalisation, the summary and the grid, α<sub>rem</sub>, d+Au, shadowing and the y<sub>init</sub> = 2 node (section 8), the formation time (section 9), identified p<sub>T</sub> spectra and ⟨p<sub>T</sub>⟩ (section 10) |
| `pt_bulk.py` | identified π±, K±, p, p̄ spectra, ⟨p<sub>T</sub>⟩ and dN/dy of the bulk samples (p+p, d+Au, O+O) against STAR: TOF spectra in NSD p+p and d+Au classes (nucl-ex/0309012, ins628232), dE/dx in d+Au (0808.2041, ins793126, and its ⟨p<sub>T</sub>⟩ / dN/dy tables transcribed in `../refs/pt_200/star_0808_2041_tables.json`), p+p charged ⟨p<sub>T</sub>⟩ vs multiplicity (nucl-ex/0606028, ins719969); each experiment's trigger and centrality estimator emulated → `../results/pt_bulk.json` |
| `auau_bulk_phobos.ipynb` | the Au+Au (and d+Au) bulk of the RHIC tune against PHOBOS dN/dη (with `hist`), the MAP block for comparison, and the runs B1/B2. Executed copy in `../results/` |
| `dau_bulk.py` | d+Au 200 GeV minimum-bias bulk (`oo_bias.py` samples `dau_<set>`: d on Au, b < 12 fm) against PHENIX (dN/dη, \|η\| < 2.6, six classes from the Au-going BBC; HEPData ins1684475), STAR (\|η\| < 1 in Au-going FTPC classes, ins656934; FTPC points at 3.0 ≤ \|η\| ≤ 3.5 in \|η\| < 0.5 classes) and BRAHMS (minimum bias to η = 4.2, ins643085): the check that a bulk set does not break the MAP's calibration → `../results/dau_bulk.json`; `--class-frac F`: the model's classes as fractions of its highest F of events (→ `../results/dau_bulk_f<F>.json`) |

`oo_bulk.py` is R4c of the plan forward (the O+O bulk against STAR); the jet side of that plan is in
`../closure_bias/`.
