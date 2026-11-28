# g2_mpi_check: gate G2 of the physics run (MPI, underlying event, forward activity)

Gate G2 of the O+O physics run (tuning page "Physics run plan"), 2026-11-24, with the frozen hard
+ bulk tune (XML js-contrib 788ca1e; the settings are read back from the XML and every job's
files): which multiparton-interaction setting goes into phase 3?

- **Part 1** — the p+p underlying event in the production chain (strings + hydro) against STAR.
- **Part 2** — the O+O vacuum R<sub>CP</sub> with MPI on and off.
- **Part 3** — the forward side: does the mid-rapidity UE carry over into the centrality acceptances
  (EPD, MBD)?
- **Part 4** — a scan of Pythia's soft MPI (pT0Ref) against STAR's UE.

**Model results, not vetted physics results; statistical errors only.**

## Run it

Samples from `../oo_bias.py` (phases `phys1`, `phys2`, `phys2_soft`) on the multi-GPU host; events in
`/data/OO-RHIC_phys`.

```bash
cd contribs/PyJetscape/example/analysis/OO-RHIC/g2_mpi_check
../oo_bias_docker.sh --dev /data/OO-RHIC_phys phys1        # phys_mb
../oo_bias_docker.sh --dev /data/OO-RHIC_phys phys2        # pp_bulk_w*, _mpioff, _switch; jets_vac[_mpioff]_w18_23, _w30_40
python g2_check.py --out /data/OO-RHIC_phys -j 16   # gate G2 after phase 2 -> ../results/g2_check.json
jupyter nbconvert --to notebook --execute g2_mpi_check.ipynb --output-dir ../results --output g2_mpi_check_executed.ipynb
python fwd_check.py --out /data/OO-RHIC_phys -j 16  # forward (EPD, MBD) activity of p+p jet events / minimum bias -> ../results/fwd_check.json
../oo_bias_docker.sh --dev /data/OO-RHIC_phys phys2_soft   # soft-MPI check: p+p with hydro at pT0Ref 1.42 / 1.45 / 1.48 (~1.2 h)
python soft_mpi_check.py --out /data/OO-RHIC_phys -j 16   # its UE vs STAR, forward ratios, pT0Ref at STAR's level -> ../results/soft_mpi_check.json
```

## Results

Not public yet: the numbers and conclusions are kept in a local, git-ignored `phys_readme.md` next to this file.

The MPI setting chosen at the gate is in the tune XMLs (`config/OO_MCGlauber_MUSIC_isr_rhictune.xml`,
`MultipartonInteractions:*` lines; js-contrib 073085b).

## Files

| file | |
|---|---|
| `g2_check.py`, `g2_mpi_check.ipynb` | gate G2 of the physics run, with the frozen hard + bulk tune (XML 788ca1e; the settings are read from the XML and the jobs' files): p+p underlying event in the production chain (`pp_bulk_w*` MPI on, `pp_bulk_mpioff_w*`, `pp_bulk_switch_w*`) against STAR (ue_check.py's definitions) and the minimum-bias level of `pp_tau0424`; O+O vacuum R<sub>AA</sub>/R<sub>CP</sub> with MPI on / off (`jets_vac[_mpioff]_w18_23`, `_w30_40`, classes from `phys_mb`), ⟨N<sub>EPD</sub>⟩ vs N<sub>coll</sub>; the page's decision rule applied → `../results/g2_check.json` |
| `fwd_check.py` | the forward side of gate G2: per leading-jet bin, charged activity of the p+p jet events (`pp_bulk_w*`, `_mpioff`, `_switch`; jet + bulk hadrons) in the EPD (2.1–5.1) and MBD (3.51–4.61) acceptances and in the transverse region, each over the minimum bias of `pp_tau0424`; EPD split into the leading jet's hemisphere and the opposite one → `../results/fwd_check.json`. |
| `soft_mpi_check.py` | the soft-MPI check after gate G2 (`phys2_soft`: `pp_bulk_soft142_w*`, `_soft145_w*`, `_soft148_w*`, Pythia's soft MPI with MPI &alpha;<sub>s</sub> 0.130, pTmin 0.2, ecmRef 200, pT0Ref 1.42 / 1.45 / 1.48 appended to the tune XML's lines): transverse density, p<sub>T</sub> &gt; 0.5 density and &lang;p<sub>T</sub>&rang; against STAR, EPD / MBD over minimum bias, against MPI on and off; pT0Ref at STAR's level interpolated in ln(ratio) (bracketing pair and a parabola), &chi;<sup>2</sup> minimum → `../results/soft_mpi_check.json`. `g2_mpi_check.ipynb` shows it as Part 4 (Part 3: `fwd_check.py`) |
